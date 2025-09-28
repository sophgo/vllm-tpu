# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2.5-VL model compatible with HuggingFace weights."""
from functools import cached_property, partial
from typing import (Callable, Iterable, List, Literal, Mapping, Optional, Set,
                    Tuple, TypedDict, Union)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BatchFeature
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig)

import numpy as np

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed import parallel_state, tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
# from vllm.model_executor.layers.soph_linear import (SophQKVParallelLinear,
#                                                     SophColumnParallelLinear,
#                                                     SophRowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig)
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, NestedTensors
from vllm.platforms import _Backend
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope

from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.qwen2_vl import Qwen2VLDummyInputsBuilder as Qwen2_5_VLDummyInputsBuilder
from vllm.model_executor.models.qwen2_vl import (Qwen2VLMultiModalProcessor, Qwen2VLProcessingInfo,
                       apply_rotary_pos_emb_vision)
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)
from vllm.model_executor.models.vision import get_vit_attn_backend


logger = init_logger(__name__)

# === Vision Inputs === #


class Qwen2_5_VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """Shape:
    `(num_patches, num_channels * patch_size * patch_size)`
    """

    image_grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


class Qwen2_5_VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: torch.Tensor
    """Supported types:
    - List[`torch.Tensor`]: A list of tensors holding all images' features.
        Each tensor holds an image's features.
    - `torch.Tensor`: A tensor holding all images' features
        (concatenation of all images' feature tensors).

    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on
        the number and resolution of the images.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    image_grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


Qwen2_5_VLImageInputs = Union[Qwen2_5_VLImagePixelInputs,
                              Qwen2_5_VLImageEmbeddingInputs]


class Qwen2_5_VLVideoPixelInputs(TypedDict):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: torch.Tensor
    """Shape:
    `(num_patches,
      num_channels * temporal_patch_size * patch_size * patch_size)`
    """

    video_grid_thw: torch.Tensor
    """Shape: `(num_videos, 3)`

    This should be in `(grid_t, grid_h, grid_w)` format.
    """

    second_per_grid_ts: torch.Tensor
    """
    The video time interval (in seconds) for each grid along the temporal
    dimension in the 3D position IDs. Returned when `videos` is not `None`.
    """


class Qwen2_5_VLVideoEmbeddingInputs(TypedDict):
    type: Literal["video_embeds"]
    video_embeds: torch.Tensor
    """Supported types:
    - List[`torch.Tensor`]: A list of tensors holding all videos' features.
        Each tensor holds an video's features.
    - `torch.Tensor`: A tensor holding all videos' features
      (concatenation of all videos' feature tensors).

    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on
        the number and resolution of the videos.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    video_grid_thw: torch.Tensor
    """Shape: `(num_videos, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


Qwen2_5_VLVideoInputs = Union[Qwen2_5_VLVideoPixelInputs,
                              Qwen2_5_VLVideoEmbeddingInputs]

# === Vision Encoder === #

def soph_rmsnorm(weight, output, hidden_states, variance_epsilon, residual=None):
    if residual is not None:
        hidden_states += residual
    residual = hidden_states
    torch.ops.my_ops.rmsnorm_forward(
        hidden_states,
        weight,
        None,
        output,
        hidden_states.dim() - 1,
        variance_epsilon,
    )
    return residual

class Qwen2_5_VisionMLP(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 bias: bool = False,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.gate_proj = ColumnParallelLinear(in_features,
                                              hidden_features,
                                              bias=bias,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.gate_proj")
        self.up_proj = ColumnParallelLinear(in_features,
                                            hidden_features,
                                            bias=bias,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.up_proj")
        self.down_proj = RowParallelLinear(hidden_features,
                                           in_features,
                                           bias=bias,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj")

        self.register_buffer("down_weight_bias", None)
        if self.tp_rank == 0:
            self.register_buffer("down_weight_bias", self.down_proj.bias)

        self.act_fn = act_fn

    def forward(self, x: torch.Tensor, mlp_buffer):
        torch.ops.my_ops.llama_mlp_forward(x, self.up_proj.weight, self.gate_proj.weight, self.down_proj.weight.transpose(0, 1).contiguous(), self.up_proj.bias, self.gate_proj.bias, self.down_weight_bias, None, None, mlp_buffer, False)
        if self.tp_size > 1:
            mlp_buffer = tensor_model_parallel_all_reduce(mlp_buffer)
        return mlp_buffer


class Qwen2_5_VisionAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        hidden_size: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size_per_partition = dist_utils.divide(
            projection_size, self.tp_size)
        self.head_dim = hidden_size // num_heads
        self.num_attention_heads_per_partition = num_heads // self.tp_size

        self.qkv = ColumnParallelLinear(input_size=embed_dim,
                                        output_size=3 * projection_size,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.qkv")
        self.proj = RowParallelLinear(input_size=projection_size,
                                      output_size=embed_dim,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.proj")

        self.softmax_scale = 1.0 / np.sqrt(self.head_dim)


    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # [s, 3 * head * head_dim] - 没有batch维度
        seq_len, _ = qkv.shape

        if self.tp_size > 1:
            qkv = tensor_model_parallel_all_gather(qkv)

        # [s, 3 * head * head_dim] -> 3 * [s, head * head_dim]
        q, k, v = qkv.chunk(3, dim=1)

        # 3 * [s, head * head_dim]
        if self.tp_size > 1:
            splitter = partial(dist_utils.split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
            v = splitter(v)[self.tp_rank]

        # 3 * [s, head * head_dim] -> 3 * [s, head, head_dim]
        new_shape = (seq_len, self.num_attention_heads_per_partition,
                     self.head_dim)
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cos,
        sin,
        attention_output,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        qkv, _ = self.qkv(x)

        q, k, v = self.split_qkv(qkv)

        _shape = (
            1,
            x.shape[0],
            self.num_attention_heads_per_partition,
            self.head_dim
        )

        q = q.view(*_shape).contiguous()
        k   = k  .view(*_shape).contiguous()
        v = v.view(*_shape).contiguous()

        ranges = [(cu_seqlens[i-1], cu_seqlens[i]) for i in range(1, len(cu_seqlens))]

        for start, end in ranges:
            torch.ops.my_ops.llava_attention(
                attention_output[:, start:end],
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                cos[:, start:end],
                sin[:, start:end],
                None,
                self.softmax_scale
            )

        attn_output = attention_output
        attn_output = attn_output.reshape(x.shape[0], -1)
        output, _ = self.proj(attn_output)

        return output


class Qwen2_5_VisionBlock(nn.Module):

    def __init__(
        self,
        vision_config,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = vision_config.hidden_size

        self.norm1 = RMSNorm(vision_config.hidden_size,
                                       eps=1e-6)
        self.norm2 = RMSNorm(vision_config.hidden_size,
                                                eps=1e-6)
        self.eps = 1e-6
        self.norm1_weight = self.norm1.weight.data
        self.norm2_weight = self.norm2.weight.data
        self.attn = Qwen2_5_VisionAttention(embed_dim=dim,
                                            hidden_size=self.hidden_size,
                                            num_heads=num_heads,
                                            projection_size=dim,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.attn")
        self.mlp = Qwen2_5_VisionMLP(dim,
                                     mlp_hidden_dim,
                                     act_fn=act_fn,
                                     bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.mlp")
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()

    def forward(self, x: torch.Tensor, residual, cos, sin, attn_buffer, mlp_buffer, rms_buffer, cu_seqlens: torch.Tensor) -> torch.Tensor:

        res = soph_rmsnorm(self.norm1_weight, rms_buffer, x, self.eps, residual)
        attn_out = self.attn(rms_buffer, cos, sin, attn_buffer,
                             cu_seqlens=cu_seqlens)
        attn_res = soph_rmsnorm(self.norm2_weight, rms_buffer, attn_out, self.eps, res)
        mlp_out = self.mlp(rms_buffer, mlp_buffer)

        return attn_res, mlp_out



class Qwen2_5_VisionPatchEmbed(nn.Module):

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels,
                              hidden_size,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size,
                   self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x


class Qwen2_5_VisionPatchMerger(nn.Module):

    def __init__(
        self,
        vision_config,
        d_model: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size
        self.ln_q = RMSNorm(vision_config.hidden_size,
                                       eps=1e-6)
        self.eps = 1e-6
        self.ln_q_weight = self.ln_q.weight.data
        self.mlp = nn.ModuleList([
            ColumnParallelLinear(self.hidden_size,
                                 self.hidden_size,
                                 bias=True,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.mlp.0"),
            nn.GELU(),
            RowParallelLinear(self.hidden_size,
                              d_model,
                              bias=True,
                              quant_config=quant_config,
                              prefix=f"{prefix}.mlp.2"),
        ])

        self.register_buffer("w2", self.mlp[2].weight.transpose(0, 1).contiguous())
        self.register_buffer("b2", None)
        if self.tp_rank == 0:
            bias = self.mlp[2].bias.contiguous()
            self.register_buffer("b2", bias)


    def forward(self, x: torch.Tensor, residual, rms_buffer, mlp_out) -> torch.Tensor:
        res = soph_rmsnorm(self.ln_q_weight, rms_buffer, x, self.eps, residual)
        rms_buffer = rms_buffer.view(1, -1, self.hidden_size)
        torch.ops.my_ops.llava_mlp(rms_buffer, self.mlp[0].weight.transpose(0, 1).contiguous(), self.w2, self.mlp[0].bias, self.b2, mlp_out)
        if self.tp_size > 1:
            mlp_out = tensor_model_parallel_all_reduce(mlp_out)
        return mlp_out.squeeze(0)


class Qwen2_5_VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta
                          **(torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (self.theta**(torch.arange(
                0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device)
                                                / self.dim))
            seq = torch.arange(seqlen,
                               device=self.inv_freq.device,
                               dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class Qwen2_5_VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        # args for get_window_index
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2

        self.rms_buffer = None

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        norm_layer = RMSNorm(vision_config.hidden_size,
                                       eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            Qwen2_5_VisionBlock(
                vision_config,
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}")
            for layer_idx in range(depth)
        ])
        self.merger = Qwen2_5_VisionPatchMerger(
            vision_config,
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )

        self.mlp_buffer  = None
        self.attn_buffer = None
        self.rms_buffer  = None

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            h_scalar = h.item()
            w_scalar = w.item()
            spatial_merge_size = self.spatial_merge_size
            hpos_ids = torch.arange(h_scalar, device=h.device).unsqueeze(1).expand(-1, w_scalar)
            wpos_ids = torch.arange(w_scalar, device=w.device).unsqueeze(0).expand(h_scalar, -1)
            hpos_ids = hpos_ids.reshape(
                h_scalar // spatial_merge_size,
                spatial_merge_size,
                w_scalar // spatial_merge_size,
                spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
        
            wpos_ids = wpos_ids.reshape(
                h_scalar // spatial_merge_size,
                spatial_merge_size,
                w_scalar // spatial_merge_size,
                spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t.item(), 1)
            )
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].float().max().to(dtype=torch.int32)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        pos_ids = pos_ids.to(rotary_pos_emb_full.device)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (self.window_size //
                                  self.spatial_merge_size // self.patch_size)

        for grid_t, grid_h, grid_w in grid_thw:
            grid_h_scalar = grid_h.item()
            grid_w_scalar = grid_w.item()
            spatial_merge_size = self.spatial_merge_size
            llm_grid_h = grid_h_scalar // spatial_merge_size
            llm_grid_w = grid_w_scalar // spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), 'constant', -100)
            index_padded = index_padded.reshape(grid_t, num_windows_h,
                                                vit_merger_window_size,
                                                num_windows_w,
                                                vit_merger_window_size)
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t, num_windows_h * num_windows_w, vit_merger_window_size,
                vit_merger_window_size)
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(
                0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:

        # patchify
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # windows attention
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        reverse_indices = torch.argsort(window_index).to(hidden_states.device)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            # device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        seq_len, _ = hidden_states.size()

        patch_shape = (seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        og_shape = (seq_len, -1)
        window_index = window_index.to(hidden_states.device)
        rotary_pos_emb = rotary_pos_emb.to(hidden_states.device)
        hidden_states = hidden_states.view(patch_shape)[window_index, :, :].view(
            og_shape
        )
        hidden_states = hidden_states.contiguous()
        rotary_pos_emb = rotary_pos_emb.view(patch_shape)[window_index, :, :].view(
            og_shape
        )
        rotary_pos_emb = rotary_pos_emb.contiguous()
        cos = rotary_pos_emb.cos().to(hidden_states.dtype).unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)
        sin = rotary_pos_emb.sin().to(hidden_states.dtype).unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)

        # compute cu_seqlens
        values = grid_thw[:, 1] * grid_thw[:, 2]
        cu_seqlens = values.cumsum(dim=0)

        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        if self.mlp_buffer is None or hidden_states.shape[0] != self.mlp_buffer.shape[0]:
            self.mlp_buffer = torch.empty_like(hidden_states)
            self.attn_buffer = self.mlp_buffer.new_empty((1, hidden_states.shape[0], self.blocks[0].attn.num_attention_heads_per_partition, self.blocks[0].attn.head_dim))
            self.rms_buffer = torch.empty_like(hidden_states)

        mlp_buffer = self.mlp_buffer
        rms_buffer = self.rms_buffer
        attn_buffer = self.attn_buffer
        merger_out = mlp_buffer.new_empty(1, mlp_buffer.shape[0]//self.spatial_merge_unit, self.merger.w2.shape[1])
        residual = None

        # transformers
        for layer_num, blk in enumerate(self.blocks):

            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            hidden_states, residual = blk(hidden_states, residual, cos, sin, attn_buffer, mlp_buffer, rms_buffer,
                                cu_seqlens=cu_seqlens_now)

        # adapter
        hidden_states = self.merger(hidden_states, residual, rms_buffer, merger_out)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen2_5_VLProcessingInfo(Qwen2VLProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2_5_VLConfig)

    def get_hf_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        fps: Optional[Union[float, List[float]]] = None,
        **kwargs: object,
    ) -> Qwen2_5_VLProcessor:
        if fps is not None:
            kwargs["fps"] = fps

        return self.ctx.get_hf_processor(
            Qwen2_5_VLProcessor,
            image_processor=self.get_image_processor(min_pixels=min_pixels,
                                                     max_pixels=max_pixels,
                                                     size=size),
            **kwargs,
        )


class Qwen2_5_VLMultiModalProcessor(Qwen2VLMultiModalProcessor):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            **super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
            second_per_grid_ts=MultiModalFieldConfig.batched("video"),
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder)
class Qwen2_5_VLForConditionalGeneration(nn.Module, SupportsMultiModal,
                                         SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    # LoRA specific attributes
    supported_lora_modules = [
        # language model
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",  # Same name with vision encoder
        # vision tower
        "qkv",
        "gate_proj",
        "up_proj",
        "attn.proj",  # Distinguish patch_embed.proj
        "fc1",
        "fc2",
        # projector
        "mlp.0",
        "mlp.2"
    ]

    embedding_modules = {}
    embedding_padding_modules = []

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen2_5_VLConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.visual = Qwen2_5_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=maybe_prefix(prefix, "visual"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        # GPTQ configs do not have a list of ignored modules, however AutoGPTQ
        # seems to avoid vision encoder sections for some models.
        if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
            return None
        return quant_config

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                 f"Got ndim: {mm_input.ndim} "
                                 f"(shape={mm_input.shape})")
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Qwen2_5_VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen2_5_VLImagePixelInputs(type="pixel_values",
                                              pixel_values=pixel_values,
                                              image_grid_thw=image_grid_thw)

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw)

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[Qwen2_5_VLVideoInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw)

    def _process_image_input(
            self,
            image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = []
        for grid_t, grid_h, grid_w in grid_thw:
            # 将 Tensor 转换为标量后再计算
            t, h, w = grid_t.item(), grid_h.item(), grid_w.item()
            size = (t * h * w) // (merge_size * merge_size)
            sizes.append(size)
        sizes = torch.tensor(sizes, dtype=torch.int32, device=grid_thw.device)

        return image_embeds.split(sizes.tolist())

    def _process_video_input(
            self,
            video_input: Qwen2_5_VLVideoInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
            if input_key in ("pixel_values_videos",
                             "video_embeds") and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(
                    **kwargs)
        return modalities

    def get_multimodal_embeddings(
            self, **kwargs) -> Optional[tuple[torch.Tensor, ...]]:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += vision_embeddings
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_video_input(video_input)
                multimodal_embeddings += video_embeddings
        return multimodal_embeddings

    def _flatten_embeddings(
        self,
        embeddings: NestedTensors,
    ) -> torch.Tensor:
        """
        Recursively flattens and concatenates NestedTensors on all but the last
        dimension.
        """
        if isinstance(embeddings, torch.Tensor):
            # Flatten all but the last dimension.
            return embeddings.flatten(0, -2)

        return torch.cat(tuple(self._flatten_embeddings(t) for t in embeddings))

    def _merge_multimodal_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        is_multimodal: torch.Tensor,
        multimodal_embeddings: NestedTensors,
    ) -> torch.Tensor:
        """
        Merge multimodal embeddings into inputs_embeds by overwriting the positions
        corresponding to placeholder tokens.
        """

        # Check the number of tokens matches
        num_expected_tokens = int(is_multimodal.sum().item())
        assert isinstance(num_expected_tokens, int)

        if is_multimodal.dtype != torch.bool:
            is_multimodal = is_multimodal.bool()
        # Flatten the multimodal embeddings
        flattened = self._flatten_embeddings(multimodal_embeddings)
        if flattened.shape[0] != num_expected_tokens:
            raise ValueError(
                f"Attempted to assign {flattened.shape[0]} multimodal tokens "
                f"to {num_expected_tokens} placeholders")

        #improve performence
        diffs = is_multimodal[1:].int() - is_multimodal[:-1].int()
        starts = torch.where(diffs == 1)[0]
        ends = torch.where(diffs == -1)[0]
        if len(starts) != len(ends):
            inputs_embeds[is_multimodal] = flattened.view(-1, flattened.shape[-1])
        else:
            current_ptr = 0
            for start_idx, end_idx in zip(starts, ends):
                region_length = end_idx - start_idx
                inputs_embeds[start_idx+1:end_idx+1] = flattened[current_ptr:current_ptr + region_length]
                current_ptr += region_length
        return inputs_embeds

    def _merge_multimodal_embeddings_wrapper(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: NestedTensors,
        placeholder_token_id: Union[int, List[int]],
    ) -> torch.Tensor:
        """
        Wrapper function that handles the merging of multimodal embeddings.
        """
        if isinstance(placeholder_token_id, list):
            placeholder_token_id = torch.tensor(
                placeholder_token_id, device=input_ids.device)
            is_multimodal = torch.isin(input_ids, placeholder_token_id).float()
        else:
            is_multimodal = (input_ids == placeholder_token_id)

        return self._merge_multimodal_embeddings(
            inputs_embeds, is_multimodal, multimodal_embeddings)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = self._merge_multimodal_embeddings_wrapper(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id])
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[tuple[torch.Tensor, ...]] = None,
        video_input: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:

        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_id,
            )

        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds,
                placeholder_token_id=self.config.video_token_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for Qwen2.5-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2.5-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            pixel_values: Pixel values to be fed to a model.
                `None` if no images are passed.
            image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in LLM.
                `None` if no images are passed.
            pixel_values_videos: Pixel values of videos to be fed to a model.
                `None` if no videos are passed.
            video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in LLM.
                `None` if no videos are passed.
            second_per_grid_ts: Tensor `(num_videos)` of video time interval (
                in seconds) for each grid along the temporal dimension in the
                3D position IDs. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                if uses_mrope(self.config):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}")
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input)
                input_ids = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.",
            tower_model="visual.merger.")
