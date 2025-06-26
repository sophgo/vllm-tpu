# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only DeepseekV2/DeepseekV3 model."""
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn
import torch_tpu
import torch.nn.functional as F
from transformers import PretrainedConfig
import numpy as np

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

def apply_rotary_pos_emb(q, k, cos, sin):
    b, h, d = q.shape
    q = q.view(b, h, d // 2, 2).transpose(3, 2).reshape(b, h, d)

    b, h, d = k.shape
    k = k.view(b, h, d // 2, 2).transpose(3, 2).reshape(b, h, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MoEGate(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        # gate weight
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)

    def forward(self, hidden_states):
        logits = torch.nn.functional.linear(hidden_states, self.weight)
        return logits

class DeepseekV2MLP(nn.Module):

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
        if quant_config:
            self.bits = quant_config.weight_bits
            self.groupsize = quant_config.group_size
            #decode
            self.up_qweight, self.up_qzeros, self.up_scales = self.up_proj.qweight.data, self.up_proj.qzeros.data, self.up_proj.scales.data
            self.gate_qweight, self.gate_qzeros, self.gate_scales = self.gate_proj.qweight.data, self.gate_proj.qzeros.data, self.gate_proj.scales.data
            self.down_qweight, self.down_qzeros, self.down_scales = self.down_proj.qweight.data, self.down_proj.qzeros.data, self.down_proj.scales.data
        else:
            self.up_proj_weight, self.gate_proj_weight = self.up_proj.weight.data, self.gate_proj.weight.data
            # self.down_proj_weight = self.down_proj.weight.data.transpose(0,1).contiguous()
        self.tp_size = get_tensor_model_parallel_world_size()

    def forward(self, x, output, *args):
        if self.quant_config:
            torch.ops.my_ops.llama_mlp_gptq_forward(x,
                                                    self.up_qweight, self.up_qzeros, self.up_scales,
                                                    self.gate_qweight, self.gate_qzeros, self.gate_scales,
                                                    self.down_qweight, self.down_qzeros, self.down_scales,
                                                    self.groupsize,
                                                    self.bits,
                                                    output)
        else:
            self.down_proj_weight = self.down_proj.weight.data.transpose(0,1).contiguous()
            torch.ops.my_ops.llama_mlp_forward(x, self.up_weight, self.gate_weight, self.down_weight, None, None, None, None, None, output, False)
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output


class DeepseekV2MoE(nn.Module):
    def __init__(
        self,
        config,
        quant_config,
        prefix,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_dim = config.hidden_size
        self.moe_intermediate_size = (
            config.moe_intermediate_size // self.tp_size
        )
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_shared_experts = config.n_shared_experts
        # Gating
        # self.gate = MoEGate(prefix=f"{prefix}.gate", config=config, weights=weights)
        self.gate = ReplicatedLinear(config.hidden_size,
                                config.n_routed_experts,
                                bias=False,
                                quant_config=None,
                                prefix=f"{prefix}.gate")
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts))
        else:
            self.gate.e_score_correction_bias = None
        # routed experts
        self.experts = nn.ModuleList([
            DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.experts.{idx}", 
            )
            for idx in range(self.n_routed_experts)
        ])
        # self.experts = FusedMoE(
        #     num_experts=config.n_routed_experts,
        #     top_k=config.num_experts_per_tok,
        #     hidden_size=config.hidden_size,
        #     intermediate_size=config.moe_intermediate_size,
        #     reduce_results=False,
        #     renormalize=config.norm_topk_prob,
        #     quant_config=quant_config,
        #     use_grouped_topk=True,
        #     num_expert_group=config.n_group,
        #     topk_group=config.topk_group,
        #     prefix=f"{prefix}.experts",
        #     scoring_func=config.scoring_func,
        #     e_score_correction_bias=self.gate.e_score_correction_bias)
        # shared experts
        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob


    def forward(
        self, x: torch.Tensor, out: torch.Tensor,
        gathered_experts_out_buf: torch.Tensor,
        shared_expert_out_buf: torch.Tensor,
        default_device,
        default_dtype
        ):

        router_logits = self.gate(x)
        router_logits = router_logits[0].to('cpu')

        if self.shared_experts is not None:
            shared_output = self.shared_experts(x, shared_expert_out_buf, None)
        else:
            shared_output = None

        routing_weights = F.softmax(router_logits, dim=1, dtype=default_dtype)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.n_routed_experts).permute(2, 1, 0)

        routing_weights_tpu = routing_weights.to(default_device, non_blocking=True)

        for expert_idx in range(self.n_routed_experts):
            k_expert_idx, tok_idx = torch.where(expert_mask[expert_idx])
            if expert_idx == 0:
                torch_tpu.tpu.synchronize()
            if tok_idx.size(0):
                expert_layer = self.experts[expert_idx]
                tok_idx_numpy = tok_idx.numpy()
                k_expert_idx_numpy = k_expert_idx.numpy()

                num = tok_idx.shape[0]
                current_states = x[tok_idx_numpy] if np.any(tok_idx_numpy) else x[:1]
                current_hidden_states = expert_layer(current_states, out[0:num])
                for i, tok_i in enumerate(tok_idx_numpy):
                    gathered_experts_out_buf[tok_i, :, k_expert_idx_numpy[i]] = current_hidden_states[i, :]

        torch.bmm(gathered_experts_out_buf, routing_weights_tpu.unsqueeze(2), out=out.unsqueeze(2))

        if shared_output is not None:
            out += shared_output

        # Reduce sum
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)

        return out
    # def forward(self, x, output):
    #     num_tokens, hidden_dim = x.shape
    #     x = x.view(-1, hidden_dim)
    #     if self.n_shared_experts is not None:
    #         shared_output = self.shared_experts(x, output)
    #     # router_logits: (num_tokens, n_experts)
    #     router_logits, _ = self.gate(x)
    #     final_hidden_states = self.experts(
    #         hidden_states=x,
    #         router_logits=router_logits) * self.routed_scaling_factor
    #     if shared_output is not None:
    #         final_hidden_states = final_hidden_states + shared_output
    #     if self.tp_size > 1:
    #         final_hidden_states = tensor_model_parallel_all_reduce(
    #             final_hidden_states)

    #     return final_hidden_states.view(num_tokens, hidden_dim)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2Attention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % self.tp_size == 0
        self.num_heads = num_heads
        self.num_local_heads = num_heads // self.tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.v_head_dim = v_head_dim

        self.head_size = qk_nope_head_dim + qk_rope_head_dim
        self.value_head_size = v_head_dim
        self.head_pad_size = max(self.head_size, self.value_head_size)

        self.eps=config.rms_norm_eps

        self.num_key_value_heads = (
            config.num_key_value_heads // self.tp_size
        )

        if self.q_lora_rank is not None:
            self.q_a_proj = SophReplicatedLinear(self.hidden_size,
                                             self.q_lora_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.q_a_proj")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.q_b_proj = SophColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = SophColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")

        self.kv_a_proj_with_mqa = SophReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = SophColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj")
        # O projection.
        self.o_proj = SophRowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")
        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # SophTPU Attention may not need，just use native Attention to init metadata
        self.attn = Attention(self.num_heads,
                              self.qk_head_dim,
                              self.scaling,
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
        kv_cache: torch.Tensor,
        attn_metadata,
        attention_output,
    ):
        if self.q_lora_rank is None:
            query, _ = self.q_proj(hidden_states)
        else:
            query, _ = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)[0])[0])[0]
        query = query.view(-1, self.num_local_heads, self.head_size)

        _, query_pe = torch.split(
            query, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv, _ = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, key_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        key_pe = key_pe.view(-1, 1, self.qk_rope_head_dim)
        # rmsnorm
        compressed_kv_con = compressed_kv.contiguous()
        compressed_kv_norm = torch.empty_like(compressed_kv)
        # self.kv_a_layernorm(compressed_kv_norm, compressed_kv_con)
        soph_rmsnorm(self.kv_a_layernorm.weight, compressed_kv_norm, compressed_kv_con, self.eps)
        # kv_b_proj
        # #print(f'compressed_kv:{compressed_kv.mean()}')
        #print(f'compressed_kv_con:{compressed_kv_con.mean()}')
        #print(f'kv_b_proj.weight.mean:{self.kv_b_proj.weight.data.mean()}')
        #print(f'compressed_kv_norm:{compressed_kv_norm.mean()}')
        kv = self.kv_b_proj(compressed_kv_norm)[0].view(
            -1, self.num_key_value_heads, self.qk_nope_head_dim + self.value_head_size
        )

        key_nope, value = torch.split(
            kv, [self.qk_nope_head_dim, self.value_head_size], dim=-1
        )

        query_pe, key_pe = apply_rotary_pos_emb(query_pe, key_pe, cos, sin)
        
        query[..., self.qk_nope_head_dim :] = query_pe
        key = torch.empty_like(query)
        key[..., : self.qk_nope_head_dim] = key_nope
        key[..., self.qk_nope_head_dim :] = key_pe

        # We need to pad the heads because Flash Attention does not support
        # qk and v with different head sizes.
        paded_value = torch.zeros_like(query, device='cpu').to(query.device)
        paded_value[..., :self.value_head_size] = value
        value = paded_value

        block_size = int(kv_cache[1].shape[1])
        #print(f'query:{query.mean()}')
        #print(f'key:{key.mean()}')
        #print(f'value:{value.mean()}')
        if attn_metadata.num_prefills > 0:

            # Prefill
            save_slots = attn_metadata.block_tables
            fetch_slots = None
            input_lengths = attn_metadata.effective_query_lens
            max_s = input_lengths.max().item()

            torch.ops.my_ops.llama_attention(attention_output[1], query, key, value, kv_cache[0], kv_cache[1],
                                    None, None, input_lengths, save_slots, fetch_slots, None,
                                    attn_metadata.block_tables.size(1), max_s, block_size, self.scaling, 2)

        else:
            # Decoding
            save_slots = attn_metadata.slot_mapping
            fetch_slots = attn_metadata.block_tables
            input_lengths = attn_metadata.context_lens
            max_s = input_lengths.max().item()

            torch.ops.my_ops.llama_attention(attention_output[1], query, key, value, kv_cache[0], kv_cache[1],
                                    None, None, input_lengths, save_slots, fetch_slots, None,
                                    attn_metadata.block_tables.size(1), max_s, block_size, self.scaling, 3)

        # Reshape the output tensor.
        #print(f'attention_output[1]:{attention_output[1].mean()}')
        output, _ = self.o_proj(attention_output[1][..., : self.value_head_size].contiguous().view(-1, self.num_local_heads * self.value_head_size), 
                              attention_output[2])
        return output


class DeepseekV2DecoderLayer(nn.Module):

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
        self.self_attn = DeepseekV2Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_buffer: torch.Tensor,
        mlp_buffer: torch.Tensor,
        rms_buffer: torch.Tensor,
        gathered_experts_out_buf,
        shared_expert_out_buf,
        default_device,
        default_dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention

        """
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

        """
        res = soph_rmsnorm(self.input_layernorm.weight, rms_buffer, hidden_states, self.eps, residual)
        #print(f'input_layernorm:f{rms_buffer}')
        attn_output = self.self_attn(
            rms_buffer,
            cos,
            sin,
            kv_cache,
            attn_metadata,
            attn_buffer
        )
        #print(f'attn_output:f{attn_output}')
        attn_res = soph_rmsnorm(self.post_attention_layernorm.weight, rms_buffer, attn_output, self.eps, res)

        #print(f'post_attention_layernorm:f{rms_buffer}')
        mlp_output = self.mlp(
            rms_buffer, mlp_buffer,
            gathered_experts_out_buf,
            shared_expert_out_buf,
            default_device,
            default_dtype)
        #print(f'mlp_output:f{mlp_output}')
        return mlp_output, attn_res


@support_torch_compile
class DeepseekV2Model(nn.Module):

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
            lambda prefix: DeepseekV2DecoderLayer(
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

        self.head_size = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
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
        self.tp_size = get_tensor_model_parallel_world_size()

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

        cos, sin = self.rotary_emb(positions)
        cos = cos.contiguous().unsqueeze(1).repeat(1, 1, 2)
        sin = sin.contiguous().unsqueeze(1).repeat(1, 1, 2)

        default_device = hidden_states.device
        default_dtype = hidden_states.dtype
        residual = None

        if self.mlp_buffer is None or hidden_states.shape[0] != self.mlp_buffer.shape[0]:
            self.rms_buffer = torch.empty_like(hidden_states)
            self.attn_buffer = []
            self.attn_buffer.append(hidden_states.new_empty(hidden_states.shape[0], (self.num_heads + 2 * self.num_kv_heads) * self.head_size))
            self.attn_buffer.append(hidden_states.new_empty((hidden_states.shape[0], self.num_heads // self.tp_size, self.head_size)))
            self.attn_buffer.append(torch.empty_like(hidden_states))

            self.mlp_buffer = torch.empty_like(hidden_states)
            self.gathered_experts_out_buf = torch.empty(
                hidden_states.size(0),
                hidden_states.size(1),
                self.num_experts_per_tok,
                device=default_device, dtype=default_dtype)
            self.shared_expert_out_buf =torch.empty_like(hidden_states)
        # import pdb
        # pdb.set_trace()
        mlp_buffer = self.mlp_buffer
        rms_buffer = self.rms_buffer
        attn_buffer = self.attn_buffer

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            #hidden_states, residual = layer(
            #    positions,
            #    hidden_states,
            #    kv_caches[i - self.start_layer],
            #    attn_metadata,
            #    residual,
            #)
            hidden_states, residual = layer(
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
                cos,
                sin,
                attn_buffer,
                mlp_buffer,
                rms_buffer,
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


class DeepseekV2ForCausalLM(nn.Module, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = DeepseekV2Model(vllm_config=vllm_config,
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
            if "rotary_emb.inv_freq" in name:
                continue

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


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
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
