# coding=utf-8
# Copyright 2023, 2024 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
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

from typing import List, Optional, Tuple, Type

import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn
import torch_tpu
import torch.nn.functional as F
from transformers import PretrainedConfig
import numpy as np

from vllm.platforms import soph_config
from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.soph_linear import (SophColumnParallelLinear,
                                                    SophReplicatedLinear,
                                                    SophRowParallelLinear)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear,)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.soph_embedding import SophEmbedding
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


# Copied from transformers.models.llama.modeling_llama.rotate_half

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

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

# class Timer:
#     def __init__(self, name):
#         self.start(name)

#     def start(self, name):
#         self.name = name
#         torch_tpu.tpu.synchronize()
#         self.start_time = time.time_ns()

#     def end(self):
#         torch_tpu.tpu.synchronize()
#         end_time = time.time_ns()
#         latency = (end_time - self.start_time) // 1e3
#         logger.info(f'Time {self.name} {latency}us')

class DeepseekV3Config(PretrainedConfig):
    def __init__(
        self,
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=11008,
        moe_intermediate_size=1407,
        num_hidden_layers=30,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts=2,
        n_routed_experts=160,
        ep_size=1,
        routed_scaling_factor=1.0,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        topk_method="gready",
        n_group=8,
        topk_group=3,
        num_experts_per_tok=6,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        norm_topk_prob=False,
        scoring_func="softmax",
        aux_loss_alpha=0.001,
        seq_aux=True,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        if tie_word_embeddings:
            raise ValueError(
                "tie_word_embeddings is not supported for Deepseek V2 models."
            )

        if ep_size != 1:
            raise ValueError(
                f"Currently only ep_size == 1 is supported for Deepseek V2 models, was {ep_size}"
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

class DeepseekV3Attention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        attn_type: str = AttentionType.DECODER
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.head_size = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.value_head_size = config.v_head_dim
        self.head_pad_size = max(self.head_size, self.value_head_size)
        self.tp_size = get_tensor_model_parallel_world_size()

        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.softmax_scale = self.head_size**-0.5 * mscale * mscale

        if self.num_heads % self.tp_size != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {self.tp_size}"
            )

        self.num_heads = self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.weight_block_size = 128
        if self.q_lora_rank is None:
            self.q_proj = SophColumnParallelLinear(self.hidden_size,
                                               config.num_attention_heads *
                                               self.head_size,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")
        else:
            self.q_a_proj = SophReplicatedLinear(self.hidden_size,
                                             self.q_lora_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.q_a_proj")
            self.q_a_proj_weight = (self.q_a_proj.weight.data, self.q_a_proj.weight_scale_inv.data)

            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)

            # load q-b fp8 weights and f16 scales, with parallel
            self.q_b_proj = SophColumnParallelLinear(self.q_lora_rank,
                                                 config.num_attention_heads *
                                                 self.head_size,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")

        # self.kv_scales = get_kv_scales(weights, f"{prefix}")

        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)

        # load kv-a fp8 weights and f16 scales, no parallel
        self.kv_a_proj_with_mqa = SophReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_weight_with_mqa = (self.kv_a_proj_with_mqa.weight.data, self.kv_a_proj_with_mqa.weight_scale_inv.data)
        # load kv-b fp8 weights and f16 scales, with parallel
        self.kv_b_proj = SophColumnParallelLinear(
            self.kv_lora_rank,
            config.num_attention_heads * (self.qk_nope_head_dim + self.value_head_size),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj")

        self.o_proj = SophRowParallelLinear(config.num_attention_heads * self.value_head_size,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")
        self.o_proj_weight = (self.o_proj.weight.data, self.o_proj.weight_scale_inv.data)
        self.num_groups = self.num_heads // self.num_key_value_heads
        # self.quantized = quant_config

        # register kv-cache buffer and pe-cache buffer, {batch, seqlen, kv_lora_rank}
        import os
        self.max_cache_size = soph_config.CONTEXT_LEN + soph_config.DECODE_TOKEN_LEN + int(os.getenv("chat_token_num", "0"))

        self.kv_cache = None
        self.pe_cache = None
        self.use_paged_kv_cache = True

        self.eps=config.rms_norm_eps

        self.attn = Attention(self.num_heads,
                              self.head_size,
                              self.softmax_scale,
                              num_kv_heads=self.num_key_value_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=attn_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlen_prefill: torch.Tensor,
        kv_cache,
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths,
        max_s: int,
        mask,
        attention_output,
    ):
        self.kv_b_weight = (self.kv_b_proj.weight.data, self.kv_b_proj.weight_scale_inv.data)
        self.q_b_weight = (self.q_b_proj.weight.data, self.q_b_proj.weight_scale_inv.data)

        batch = input_lengths.shape[0]
        if self.kv_cache is None and self.use_paged_kv_cache == False:
            self.kv_cache = torch.empty(batch, self.max_cache_size, self.kv_lora_rank,
                                        device=hidden_states.device, dtype=hidden_states.dtype)
            self.pe_cache = torch.empty(batch, self.max_cache_size, self.qk_rope_head_dim,
                                        device=hidden_states.device, dtype=hidden_states.dtype)
        seqlen = hidden_states.shape[0]
        lquery = attention_output["lquery"]
        normed_lquery = attention_output["normed_lquery"]
        lkv = attention_output["lkv"]
        normed_lkv_nope = attention_output["normed_lkv_nope"]
 
        lkv_nope = attention_output["lkv_nope"]
        normed_lkv_nope = attention_output["normed_lkv_nope"]
        key_pe = attention_output["key_pe"]
 
        if self.q_lora_rank is None:
            query = self.q_proj(hidden_states)
        else:
            torch.ops.my_ops.mm_w8a16_dq_forward(
                hidden_states,
                *self.q_a_proj_weight,
                lquery,
                self.weight_block_size)
            soph_rmsnorm(self.q_a_layernorm.weight.data, normed_lquery, lquery, self.eps)
            query = normed_lquery
 
        # kv-compress and k-pe
        torch.ops.my_ops.mm_w8a16_dq_forward(
            hidden_states,
            *self.kv_a_weight_with_mqa,
            lkv,
            self.weight_block_size)
 
        if seqlen > 1:
            lkv_nope.copy_(lkv[:, :self.kv_lora_rank])
            key_pe.copy_(lkv[:, self.kv_lora_rank:])
        else:
            lkv_nope = lkv[0, :self.kv_lora_rank]
            key_pe = lkv[0, self.kv_lora_rank:]
 
        soph_rmsnorm(self.kv_a_layernorm.weight.data, normed_lkv_nope, lkv_nope, self.eps)
 
        if self.use_paged_kv_cache == False:
            torch.ops.my_ops.latent_attention_fp8(
                attention_output["attn_out"],
                query,
                normed_lkv_nope,
                key_pe,
                self.q_b_weight[0],
                self.kv_b_weight[0],
                self.kv_cache,
                self.pe_cache,
                cos,
                sin,
                self.q_b_weight[1],
                self.kv_b_weight[1],
                mask,
                input_lengths,
                self.num_heads,
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.value_head_size,
                max_s,
                self.weight_block_size,
                self.kv_cache.shape[1],
                self.softmax_scale,
                0 if cu_seqlen_prefill is not None else 1)
        else:
            torch.ops.my_ops.paged_latent_attention_fp8(
                attention_output["attn_out"],
                query,
                normed_lkv_nope,
                key_pe,
                self.q_b_weight[0],
                self.kv_b_weight[0],
                kv_cache[0],
                kv_cache[1],
                cos,
                sin,
                self.q_b_weight[1],
                self.kv_b_weight[1],
                block_tables,
                slots if cu_seqlen_prefill is None else block_tables,
                mask,
                input_lengths,
                self.num_heads,
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.value_head_size,
                max_s,
                self.weight_block_size,
                block_tables.shape[1],
                kv_cache[0].shape[1],
                self.softmax_scale,
                2 if cu_seqlen_prefill is not None else 3)
 
        # output proj
        torch.ops.my_ops.mm_w8a16_dq_forward(
            attention_output["attn_out"].reshape(-1, self.num_heads * self.value_head_size),
            self.o_proj_weight[0],
            self.o_proj_weight[1],
            attention_output["o_proj_out"],
            self.weight_block_size)

        if self.tp_size > 1:
            attention_output["o_proj_out"] = tensor_model_parallel_all_reduce(attention_output["o_proj_out"])
        return attention_output["o_proj_out"]


class DeepseekV3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.quant_config = quant_config
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

        self.up_weight = self.up_proj.weight.data
        self.up_scale = self.up_proj.weight_scale_inv.data
        self.gate_weight = self.gate_proj.weight.data
        self.gate_scale = self.gate_proj.weight_scale_inv.data
        self.down_proj_weight = self.down_proj.weight.data
        down_proj_dtype = self.down_proj_weight.dtype
        self.down_weight = self.down_proj_weight.to('cpu').transpose(0,1).contiguous()
        self.down_weight = self.down_weight.to(self.down_proj_weight.device)
        self.down_scale = self.down_proj.weight_scale_inv.data.transpose(0,1).contiguous()

        self.blocksize = self.quant_config.weight_block_size[0]
        self.tp_size = get_tensor_model_parallel_world_size()

    def forward(self, x, output, *args):
        torch.ops.my_ops.mlp_w8a16_dq_forward(x, self.gate_weight, self.up_weight, self.down_weight,
                                                self.gate_scale, self.up_scale, self.down_scale,
                                                output, self.blocksize)
        return output

class DeepseekV3MoE(nn.Module):
    def __init__(
        self,
        prefix,
        config: DeepseekV3Config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_dim = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.scoring_func = config.scoring_func
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.topk_method = config.topk_method
        self.routed_scaling_factor = config.routed_scaling_factor

        # Gating
        self.gate = SophReplicatedLinear(config.hidden_size,
                                config.n_routed_experts,
                                bias=False,
                                quant_config=quant_config,
                                prefix=f"{prefix}.gate")

        # Routed experts
        gate_weights = []
        gate_scales = []
        up_weights = []
        up_scales = []
        down_weights = []
        down_scales = []

        self.experts = nn.ModuleList([
            DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.experts.{idx}", 
            )
            for idx in range(self.n_routed_experts)
        ])

        for expert in self.experts:
            up_weights.append(expert.up_weight)
            up_scales.append(expert.up_scale)

            gate_weights.append(expert.gate_weight)
            gate_scales.append(expert.gate_scale)

            down_proj_dtype = expert.down_weight.dtype
            down_weights.append(expert.down_weight)
            down_scales.append(expert.down_scale)

        def concat(tensors):
            origDtype = tensors[0].dtype
            origShape = tensors[0].shape
            origDevice = tensors[0].device
            acc = torch.empty(len(tensors), *origShape, dtype=origDtype, device=origDevice)
            for idx, t in enumerate(tensors):
                acc[idx] = t
            tensors.clear()
            return acc

        self.gate_weights = concat(gate_weights)
        self.gate_scales = concat(gate_scales)
        self.up_weights = concat(up_weights)
        self.up_scales = concat(up_scales)
        self.down_weights = concat(down_weights)
        self.down_scales = concat(down_scales)
        self.down_weights = self.down_weights.contiguous()
        self.down_scales = self.down_scales.contiguous()

        # shared experts
        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)
            self.shared_experts = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        self.max_seq_len = 0
        

    def update_buffer(self, seqlen, device):
        if seqlen <= self.max_seq_len:
            return
        self.max_seq_len = seqlen

        opts = dict(device=device, dtype=torch.bfloat16)
        self.router_logits = torch.empty(self.max_seq_len, self.n_routed_experts, **opts)
        self.routing_weights = torch.empty(self.max_seq_len, self.top_k, **opts)
        self.selected_experts = torch.empty(self.max_seq_len, self.top_k, device=device, dtype=torch.int32)
        self.denominator = torch.empty(self.max_seq_len, 1, **opts)

    def forward(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        gathered_experts_out_buf: torch.Tensor,
        shared_expert_out_buf: torch.Tensor,
        default_device,
        default_dtype):
 
        seq_len, h = x.shape
        self.update_buffer(seq_len, x.device)
        router_logits = self.router_logits[:seq_len]
        routing_weights = self.routing_weights[:seq_len]
        selected_experts = self.selected_experts[:seq_len]
        denominator = self.denominator[:seq_len]
 
        torch.matmul(x, self.gate.weight.data.T, out=router_logits)
 
        if self.scoring_func == "sigmoid":
            torch.sigmoid(router_logits, out=router_logits)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )
 
        torch.topk(
            router_logits, k=self.top_k, dim=-1, sorted=False,
            out=(routing_weights, selected_experts))
 
        if self.shared_experts is not None:
            shared_output = self.shared_experts(x, shared_expert_out_buf)
        else:
            shared_output = None
 
        # gathered_experts_out_buf is of shape (seq x k x hs)
        output_sample = None
        input_sample = None
        selected_experts_middle = None
        routing_weights_middle = None
        num_select_experts = None
 
        block_size = 128
        num_experts = self.n_routed_experts
        num_experts_per_tok = self.top_k
        use_grouped_topk = True
        num_expert_group = self.n_group
        topk_group = self.topk_group
        torch.ops.my_ops.fused_moe_fused_experts(gathered_experts_out_buf, x,
                                                output_sample, input_sample,
                                                self.gate_weights, self.up_weights, self.down_weights,
                                                self.gate_scales, self.up_scales, self.down_scales,
                                                selected_experts, routing_weights,
                                                num_select_experts,
                                                selected_experts_middle, routing_weights_middle,
                                                block_size, num_experts, num_experts_per_tok,
                                                use_grouped_topk, num_expert_group, topk_group,
                                                None, None, None, False)
 
        if self.norm_topk_prob:
            torch.sum(routing_weights, dim=-1, keepdim=True, out=denominator) # + 1e-20
            torch.mul(routing_weights, self.routed_scaling_factor, out=routing_weights)
            torch.div(routing_weights, denominator, out=routing_weights)
 
        torch.bmm(gathered_experts_out_buf.transpose(-2, -1), routing_weights.unsqueeze(2), out=out.unsqueeze(2))
        if shared_output is not None:
            out += shared_output

        # Reduce sum

        return out

class DeepseekV3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep='.')[-1])
        # if model_config.use_mla:
        #     attn_cls = DeepseekV2MLAAttention
        # else:
        #     attn_cls = DeepseekV2Attention
        self.self_attn = DeepseekV3Attention(
            prefix=f"{prefix}.self_attn",
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
        )

        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = DeepseekV3MoE(
                prefix=f"{prefix}.mlp",
                config=config,
                quant_config=quant_config,
            )
        else:
            self.mlp = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        
        self.eps = config.rms_norm_eps
        self.tp_size = get_tensor_model_parallel_world_size()

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlen_prefill: torch.Tensor,
        kv_cache,
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths,
        max_s: int,
        mask,
        attn_buffer,
        rms_buffer,
        mlp_buffer,
        gathered_experts_out_buf: torch.Tensor,
        shared_expert_out_buf: torch.Tensor,
        default_device,
        default_dtype
    ):
        res = soph_rmsnorm(self.input_layernorm.weight, rms_buffer, hidden_states, self.eps, residual)
        attn_output = self.self_attn(
            rms_buffer,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
            mask,
            attn_buffer
        )
        attn_res = soph_rmsnorm(self.post_attention_layernorm.weight, rms_buffer, attn_output, self.eps, res)

        mlp_output = self.mlp(
            rms_buffer, mlp_buffer,
            gathered_experts_out_buf,
            shared_expert_out_buf,
            default_device,
            default_dtype)
        
        if self.tp_size > 1:
            mlp_output = tensor_model_parallel_all_reduce(mlp_output)

        return mlp_output, attn_res

class SophTensorEmbedding(nn.Module):
    def __init__(self, prefix: str, weights):
        super().__init__()
        self.weight = weights.get_tensor(f"{prefix}.weight")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.embedding(input, self.weight)

        return out

class DeepseekV3Model(torch.nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            # self.embed_tokens = VocabParallelEmbedding(
            #     config.vocab_size,
            #     config.hidden_size,
            #     quant_config=quant_config,
            #     prefix=f"{prefix}.embed_tokens")
            self.embed_tokens = SophEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens")
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV3DecoderLayer(
                config,
                prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.head_size = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.value_head_size = config.v_head_dim
        self.head_pad_size = max(self.head_size, self.value_head_size)
        self.tp_size = get_tensor_model_parallel_world_size()
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.rotary_emb = get_rope(
            self.head_size,
            rotary_dim = self.qk_rope_head_dim,
            max_position = config.max_position_embeddings,
            base = rope_theta,
            rope_scaling = rope_scaling,
        )

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_experts_per_tok = config.num_experts_per_tok
        self.mlp_buffer = None
        self.attn_buffer = None
        self.rms_buffer = None

        self.eps = config.rms_norm_eps

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        default_device = hidden_states.device
        default_dtype = hidden_states.dtype

        if self.mlp_buffer is None or hidden_states.shape[0] != self.mlp_buffer.shape[0]:
            self.rms_buffer = torch.empty_like(hidden_states)
            self.attn_buffer = {}
            seqlen = hidden_states.shape[0]
            self.attn_buffer["attn_out"] = hidden_states.new_empty((seqlen, self.num_heads, self.value_head_size))
            self.attn_buffer["o_proj_out"] = torch.empty_like(hidden_states)
            self.attn_buffer["lquery"] = hidden_states.new_empty((seqlen, self.q_lora_rank))
            self.attn_buffer["normed_lquery"] = hidden_states.new_empty((seqlen, self.q_lora_rank))
            self.attn_buffer["lkv_nope"] = hidden_states.new_empty((seqlen, self.kv_lora_rank))
            self.attn_buffer["normed_lkv_nope"] = hidden_states.new_empty((seqlen, self.kv_lora_rank))
            self.attn_buffer["lkv"] = hidden_states.new_empty((seqlen, self.kv_lora_rank + self.qk_rope_head_dim))
            self.attn_buffer["key_pe"] = hidden_states.new_empty((seqlen, self.qk_rope_head_dim))

            self.mlp_buffer = torch.empty_like(hidden_states)
            self.gathered_experts_out_buf = torch.empty(
                hidden_states.size(0),
                self.num_experts_per_tok,
                hidden_states.size(1),
                device=default_device, dtype=default_dtype)
            self.shared_expert_out_buf =torch.empty_like(hidden_states)

        mlp_buffer = self.mlp_buffer
        rms_buffer = self.rms_buffer
        attn_buffer = self.attn_buffer

        cos, sin = self.rotary_emb(positions)
        cos = cos.contiguous().unsqueeze(1).repeat(1, 1, 2)
        sin = sin.contiguous().unsqueeze(1).repeat(1, 1, 2)

        # Attention metadata
        cu_seqlen_prefill = attn_metadata.num_prefills
        if cu_seqlen_prefill is not None:
            input_lengths = attn_metadata.effective_query_lens
        else:
            input_lengths = attn_metadata.context_lens
        max_s = input_lengths.max().item()
        mask = None
        if cu_seqlen_prefill is not None:
            mask = torch.triu(torch.full((max_s, max_s), -65504.0, dtype=default_dtype), diagonal=1).to(default_device)
        block_tables = attn_metadata.block_tables
        slots = attn_metadata.slot_mapping

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_caches[i - self.start_layer],
                block_tables,
                slots,
                input_lengths,
                max_s,
                mask,
                attn_buffer,
                rms_buffer,
                mlp_buffer,
                self.gathered_experts_out_buf,
                self.shared_expert_out_buf,
                default_device,
                default_dtype
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        #hidden_states, _ = self.norm(hidden_states, residual)
        soph_rmsnorm(self.norm.weight, rms_buffer, hidden_states, self.eps, residual)
        #return hidden_states
        return rms_buffer


class DeepseekV3ForCausalLM(nn.Module, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.model = DeepseekV3Model(vllm_config=vllm_config,
                                     prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # ("gate_up_proj", "gate_proj", 0),
            # ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            exclude_list = ['embed_tokens'] + [f'model.layers.{i}' for i in range(self.config.num_hidden_layers)]  # 前config.num_hidden_layer层
            if all(exclude_str not in name for exclude_str in exclude_list):
                continue
            
            if "rotary_emb.inv_freq" in name:
                continue

            # print(f'name:{name};dtype:{loaded_weight.dtype}')

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    # name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                #   name,
                                #   shard_id=shard_id,
                                #   expert_id=expert_id
                                  )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if 'e_score_correction_bias' in name:
                        continue

                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

class DeepseekV2ForCausalLM(DeepseekV3ForCausalLM):
    pass

def get_spec_layer_idx_from_weight_name(config: PretrainedConfig,
                                        weight_name: str) -> Optional[int]:
    if hasattr(config,
               "num_nextn_predict_layers") and (config.num_nextn_predict_layers
                                                > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx+i}."):
                return layer_idx + i
    return None
