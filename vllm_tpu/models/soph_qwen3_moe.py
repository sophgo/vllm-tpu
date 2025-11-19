# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""
import typing
from collections.abc import Callable, Iterable
from typing import Any, Optional, Union, List, Tuple
 
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen3MoeConfig

from vllm.logger import init_logger
from vllm.config import CacheConfig, VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.forward_context import get_forward_context
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size,
                              get_tensor_model_parallel_rank,
                              tensor_model_parallel_all_reduce)
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.tpu.sampler import Sampler
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention 
from vllm.model_executor.models.utils import (extract_layer_index,
                                              is_pp_missing_parameter,
                                              make_empty_intermediate_tensors_factory, make_layers,
                                              maybe_prefix)
from vllm_sophon.attention.attention import SophAttentionState, SophTPUMetadata
from vllm_sophon.ops.layernorm import SophonRMSNorm
from vllm_sophon.ops.soph_embedding import SophEmbedding
from vllm_sophon.ops.soph_fused_moe import SophDeepseekV3FusedMoE
from vllm_sophon.ops.soph_linear import (SophQKVParallelLinear,
                                                   SophColumnParallelLinear,
                                                   SophRowParallelLinear,
                                                   SophReplicatedLinear)

logger = init_logger(__name__)

class Qwen3MoeMLP(nn.Module):

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
        self.gate_proj = SophColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = SophColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = SophRowParallelLinear(intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj")
        if quant_config:
            self.bits = quant_config.weight_bits
            self.groupsize = quant_config.group_size
            self.up_qweight, self.up_qzeros, self.up_scales = self.up_proj.qweight.data, self.up_proj.qzeros.data, self.up_proj.scales.data
            self.gate_qweight, self.gate_qzeros, self.gate_scales = self.gate_proj.qweight.data, self.gate_proj.qzeros.data, self.gate_proj.scales.data
            self.down_qweight, self.down_qzeros, self.down_scales = self.down_proj.qweight.data, self.down_proj.qzeros.data, self.down_proj.scales.data
        else:
            self.up_proj_weight, self.gate_proj_weight = self.up_proj.weight.data, self.gate_proj.weight.data
        self.tp_size = get_tensor_model_parallel_world_size()
 
    def forward(self, x, output):
        if self.quant_config:
            torch.ops.my_ops.llama_mlp_gptq_forward(x,
                                                    self.up_qweight, self.up_qzeros, self.up_scales,
                                                    self.gate_qweight, self.gate_qzeros, self.gate_scales,
                                                    self.down_qweight, self.down_qzeros, self.down_scales,
                                                    self.groupsize,
                                                    self.bits,
                                                    output)
        else:
            self.down_proj_weight = self.down_proj.weight.data
            torch.ops.my_ops.llama_mlp_forward(x, self.up_proj_weight, self.gate_proj_weight, self.down_proj_weight, None, None, None, None, None, output, False)
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output
 
class Qwen3MoeSparseMoeBlock(nn.Module):
 
    def __init__(
        self,
        config: Qwen3MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.norm_topk_prob = config.norm_topk_prob
        self.max_seq_len = 0
        self.prefix = prefix
 
        if self.tp_size > self.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.num_experts}.")
            
        self.experts = SophDeepseekV3FusedMoE(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            n_routed_experts = self.num_experts,
            prefix=f"{prefix}.experts",
            )
 
        self.gate = SophReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=f"{prefix}.gate",
        )
        # Buffer management for fused MoE
        self.router_logits = None
        self.routing_weights = None
        self.selected_experts = None
 
    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        # GPTQ configs do not have a list of ignored modules, however AutoGPTQ
        # seems to avoid gate quantization.
        # See: https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4
        if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
            return None
        return quant_config
 
    def update_buffer(self, seqlen, device):
        tpu_graph_enabled = os.environ.get("PYTORCH_TPU_ALLOCATOR")
        if not tpu_graph_enabled and seqlen <= self.max_seq_len:
            return
        self.max_seq_len = seqlen
 
        opts = dict(device=device, dtype=torch.bfloat16)
        self.router_logits = torch.empty(self.max_seq_len, self.num_experts, **opts)
        self.routing_weights = torch.empty(self.max_seq_len, self.top_k, **opts)
        self.selected_experts = torch.empty(self.max_seq_len, self.top_k, device=device, dtype=torch.int32)
 
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        default_dtype: torch.dtype,
        mlp_buffer: torch.Tensor,
        gathered_experts_out_buf: torch.Tensor,
        ) -> torch.Tensor:
 
        assert hidden_states.dim(
        ) <= 2, "Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs"
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
 
        # Update buffer for current sequence length
        seq_len = hidden_states.shape[0]
        self.update_buffer(seq_len, hidden_states.device)
 
        # Get buffer views for current sequence length
        router_logits = self.router_logits[:seq_len]
        routing_weights = self.routing_weights[:seq_len]
        selected_experts = self.selected_experts[:seq_len]
 
        # Compute router logits
        torch.matmul(hidden_states, self.gate.weight.data.T, out=router_logits)
 
        router_logits = F.softmax(router_logits, dim=1, dtype=default_dtype)
 
        # Use torch.topk to get routing weights and selected experts
        torch.topk(
            router_logits, k=self.top_k, dim=-1, sorted=False,
            out=(routing_weights, selected_experts))
 
        # Prepare parameters for fused MoE
        output_sample = None
        input_sample = None
        selected_experts_middle = None
        routing_weights_middle = None
        num_select_experts = None
 
        block_size = 128
        num_experts = self.num_experts
        num_experts_per_tok = self.top_k
        use_grouped_topk = False
        num_expert_group = 1
        topk_group = 1
 
        self.experts(gathered_experts_out_buf, hidden_states,
                    output_sample, input_sample,
                    selected_experts, routing_weights,
                    num_select_experts,
                    selected_experts_middle, routing_weights_middle,
                    block_size, num_experts, num_experts_per_tok,
                    use_grouped_topk, num_expert_group, topk_group,)
 
        # Apply normalization and combine expert outputs
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        torch.bmm(gathered_experts_out_buf.transpose(-2, -1), routing_weights.unsqueeze(2), out=mlp_buffer.unsqueeze(2))
        if self.tp_size > 1:
            mlp_buffer = tensor_model_parallel_all_reduce(mlp_buffer)

        return mlp_buffer.view(orig_shape)

class SophQwen3MoeAttention(Qwen3MoeAttention):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 rope_theta: float = 10000,
                 rope_scaling: Optional[dict[str, Any]] = None,
                 max_position_embeddings: int = 8192,
                 head_dim: Optional[int] = None,
                 rms_norm_eps: float = 1e-06,
                 qkv_bias: bool = False,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 dual_chunk_attention_config: Optional[dict[str, Any]] = None,
                 ) -> None:

        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         num_kv_heads=num_kv_heads,
                         rope_theta=rope_theta,
                         rope_scaling=rope_scaling,
                         max_position_embeddings=max_position_embeddings,
                         head_dim=head_dim,
                         rms_norm_eps=rms_norm_eps,
                         qkv_bias=qkv_bias,
                         cache_config=cache_config,
                         quant_config=quant_config,
                         prefix=prefix,
                         dual_chunk_attention_config=dual_chunk_attention_config)

        self.attn_metadata_prefix = f"{prefix}.attn"
        self.qkv_proj = SophQKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = SophRowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.eps = rms_norm_eps
        self.q_norm = SophonRMSNorm(self.head_dim, eps=self.eps)
        self.k_norm = SophonRMSNorm(self.head_dim, eps=self.eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:

        qkv, _ = self.qkv_proj(hidden_states, attention_output[0])
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        query.copy_(q, non_blocking = True)
        key.copy_(k, non_blocking = True)
        value.copy_(v, non_blocking = True)

        # norm of q,k
        self.q_norm(query, query)
        self.k_norm(key, key)

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None:
            attn_metadata = attn_metadata[self.attn_metadata_prefix]
            assert isinstance(attn_metadata, SophTPUMetadata)
            attn_metadata.cos = cos
            attn_metadata.sin = sin
            attn_metadata.buffer = attention_output[1]
            attn_out = self.attn(query, key, value)
        output, _ = self.o_proj(attention_output[1].view(-1, self.num_heads * self.head_dim), attention_output[2])

        return output


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)

        self.self_attn = SophQwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None), 
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(config=config,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.mlp")
        else:
            self.mlp = Qwen3MoeMLP(hidden_size=config.hidden_size,
                                   intermediate_size=config.intermediate_size,
                                   hidden_act=config.hidden_act,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp")
        self.input_layernorm = SophonRMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = SophonRMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_buffer: torch.Tensor,
        mlp_buffer: torch.Tensor,
        gathered_experts_out_buf: torch.Tensor,
        rms_buffer: torch.Tensor,
        query_buffer: torch.Tensor,
        key_buffer: torch.Tensor,
        value_buffer: torch.Tensor,
        default_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, rms_buffer, residual)

        attn_output = self.self_attn(
            rms_buffer,
            cos,
            sin,
            attn_buffer,
            query_buffer,
            key_buffer,
            value_buffer,
        )

        # Apply SophonRMSNorm to attention output
        hidden_states, residual = self.post_attention_layernorm(attn_output, rms_buffer, residual)

        assert isinstance(self.mlp, Qwen3MoeSparseMoeBlock)
        # 创建输出缓冲区
        hidden_states = self.mlp(hidden_states,  default_dtype, mlp_buffer, gathered_experts_out_buf)

        return hidden_states, residual

@support_torch_compile
class Qwen3MoeModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.quant_config = quant_config
        self.embed_tokens = SophEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens")
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3MoeDecoderLayer(vllm_config=vllm_config,
                                                prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = SophonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.head_dim = config.head_dim
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim = self.head_dim,
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
        self.query_buffer = None
        self.key_buffer = None
        self.value_buffer = None

        self.eps = config.rms_norm_eps
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_dim = config.hidden_size
        logger.info(f"Qwen3MoeModel initialized with hidden_dim: {self.hidden_dim}")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
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

        cos, sin = self.rotary_emb(positions)
        cos = cos.contiguous().unsqueeze(1).repeat(1, 1, 2)
        sin = sin.contiguous().unsqueeze(1).repeat(1, 1, 2)

        tpu_graph_enabled = os.environ.get("PYTORCH_TPU_ALLOCATOR")
        if tpu_graph_enabled or self.mlp_buffer is None or hidden_states.shape[0] != self.mlp_buffer.shape[0]:
            self.mlp_buffer = torch.empty_like(hidden_states)
            # {MM-QKV output, Attention output, Attention_FC output}
            self.attn_buffer = []
            self.attn_buffer.append(hidden_states.new_empty(hidden_states.shape[0], (self.num_heads + 2 * self.num_kv_heads) * self.head_dim // self.tp_size))
            self.attn_buffer.append(hidden_states.new_empty(hidden_states.shape[0], self.num_heads // self.tp_size, self.head_dim))
            self.attn_buffer.append(torch.empty_like(hidden_states))
            self.rms_buffer = torch.empty_like(hidden_states)
            self.gathered_experts_out_buf = torch.empty(
                hidden_states.size(0),
                self.num_experts_per_tok,
                hidden_states.size(1),
                device=default_device, dtype=default_dtype)
            self.query_buffer = torch.empty(
                hidden_states.shape[0],
                max(1, self.num_heads // self.tp_size),
                self.head_dim,
                dtype = default_dtype, device = default_device)
            self.key_buffer = torch.empty(
                hidden_states.shape[0],
                max(1, self.num_kv_heads // self.tp_size),
                self.head_dim,
                dtype = default_dtype, device = default_device)
            self.value_buffer = torch.empty(
                hidden_states.shape[0],
                max(1, self.num_kv_heads // self.tp_size),
                self.head_dim,
                dtype = default_dtype, device = default_device)
        mlp_buffer = self.mlp_buffer
        rms_buffer = self.rms_buffer
        attn_buffer = self.attn_buffer
        query_buffer = self.query_buffer
        key_buffer = self.key_buffer
        value_buffer = self.value_buffer

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                attn_buffer,
                mlp_buffer,
                self.gathered_experts_out_buf,
                rms_buffer,
                query_buffer,
                key_buffer,
                value_buffer,
                default_dtype
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, residual = self.norm(hidden_states, rms_buffer, residual)
        return rms_buffer

class Qwen3MoeForCausalLM(nn.Module, SupportsPP, SupportsLoRA):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3MoeModel(vllm_config=vllm_config,
                                   prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        self.sampler = Sampler()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids,
                                   positions,
                                   intermediate_tensors,
                                   inputs_embeds)
        return hidden_states
 
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        self.logits_processor.use_all_gather = True
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
 
    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
 
    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        name_flags_1 = ["up_proj", "gate_proj", "down_proj"]
        name_flags_2 = [".weights", ".scales"]
        do_resize_list = [a + b for a in name_flags_1 for b in name_flags_2]
 
        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (".bias", "_bias", ".k_scale", "_k_scale",
                           ".v_scale", "_v_scale", ".weight_scale",
                           "_weight_scale", ".input_scale", "_input_scale")
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        has_quantization = hasattr(self.quant_config, 'weight_block_size') if self.quant_config else False 
        expert_params_mapping = SophDeepseekV3FusedMoE.make_expert_params_mapping(
            num_experts=self.config.num_experts, has_quantization=has_quantization)
        for name, loaded_weight in weights:
            if any(do_resize_name in name for do_resize_name in do_resize_list):
                param = params_dict[name]
                if 'down_proj' in name:
                    dim_ = getattr(param, "input_dim", 0)
                else:
                    dim_ = getattr(param, "output_dim", 0)
                tp_rank = get_tensor_model_parallel_rank()
                shard_size = getattr(param, "data", None).shape[dim_]
                n_pad = shard_size* self.tp_size - loaded_weight.shape[dim_]
                if (tp_rank == self.tp_size - 1) and (n_pad > 0):
                    if dim_ == 0:
                        loaded_weight = torch.nn.functional.pad(loaded_weight, (0,0,0,n_pad), 'constant', 0).contiguous()
                    elif dim_ == 1:
                        loaded_weight = torch.nn.functional.pad(loaded_weight, (0,n_pad,0,0), 'constant', 0).contiguous()
                    else:
                        raise NotImplementedError("Let's make that generic when needed")
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id = mapping
                    if weight_name not in name:
                        continue
 
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
 
                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)
 
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
 
                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name_mapped.endswith(
                            ignore_suffixes
                    ) and name_mapped not in params_dict:
                        continue
 
                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(Callable[..., bool],
                                                param.weight_loader)
                    if hasattr(weight_loader, '__name__') and weight_loader.__name__ == 'scaled_weight_loader':
                        success = weight_loader(loaded_weight, param, name_mapped,
                                            expert_id=expert_id,
                                            return_success=True)
                    else:
                        weight_loader(loaded_weight, param, expert_id)
                        success = True
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
 
                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(
                            ignore_suffixes) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale")
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv-scale is not loaded.",  # noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
