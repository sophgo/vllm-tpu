"""Inference-only Qwen3 model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Set, Tuple, Union

import os
import torch
from torch import nn
from transformers import Qwen3Config

from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, ParallelConfig
from vllm.config.utils import config
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.tpu.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import PoolerOutput

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              PPMissingLayer,
                                              WeightsMapper,
                                              is_pp_missing_parameter,
                                              make_empty_intermediate_tensors_factory,
                                              make_layers,
                                              maybe_prefix)

from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.models.qwen3 import Qwen3Attention

from vllm_sophon.attention.attention import SophTPUMetadata
from vllm_sophon.ops.soph_linear import (SophQKVParallelLinear,
                                         SophColumnParallelLinear,
                                         SophRowParallelLinear)
from vllm_sophon.ops.soph_embedding import SophEmbedding
from vllm_sophon.ops.layernorm import SophonRMSNorm

logger = init_logger(__name__)

class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
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
        self.down_proj = SophRowParallelLinear(
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

class SophonQwen3Attention(Qwen3Attention):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[Tuple] = None,
                 prefix: str = "",
                 attn_type: str = AttentionType.DECODER,
                 eps: float = 1e-6,
                 head_dim: int = None,
                 ) -> None:

        super().__init__(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads, max_position=max_position,head_dim=head_dim,
                         rope_theta=rope_theta, cache_config=cache_config, quant_config=quant_config,
                         rope_scaling=rope_scaling, prefix=prefix, attn_type=attn_type)

        self.attn_metadata_prefix = f"{prefix}.attn"
        self.qkv_proj = SophQKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
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

        self.eps = eps
        self.q_norm = SophonRMSNorm(self.head_dim, eps=self.eps)
        self.k_norm = SophonRMSNorm(self.head_dim, eps=self.eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_output: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
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
        self.q_norm(query, query, residual=None)
        self.k_norm(key, key, residual=None)

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


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = SophonQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            eps=config.rms_norm_eps,
            head_dim=config.head_dim
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = SophonRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = SophonRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        attn_buffer: torch.Tensor,
        mlp_buffer: torch.Tensor,
        rms_buffer: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        query_buffer: torch.Tensor,
        key_buffer: torch.Tensor,
        value_buffer: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, rms_buffer, residual)

        attn_output = self.self_attn(hidden_states, attn_buffer, cos, sin, query_buffer, key_buffer, value_buffer)

        hidden_states, residual = self.post_attention_layernorm(attn_output, rms_buffer, residual)

        self.mlp(hidden_states, mlp_buffer)

        return mlp_buffer, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen3-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Qwen3Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # TODO (@robertgshaw2): see if this can be moved out
        if config.sliding_window is not None:
            assert config.max_window_layers >= config.num_hidden_layers, (
                "Sliding window for some but all layers is not supported. "
                "This model uses sliding window but `max_window_layers` = {} "
                "is less than `num_hidden_layers` = {}. Please open an issue "
                "to discuss this feature.".format(
                    config.max_window_layers,
                    config.num_hidden_layers,
                ))

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = SophEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3DecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = SophonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

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

        self.mlp_buffer = None
        self.rms_buffer = None

        self.tp_size = get_tensor_model_parallel_world_size()

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

        cos, sin = self.rotary_emb(positions)
        cos = cos.contiguous().unsqueeze(1).repeat(1, 1, 2)
        sin = sin.contiguous().unsqueeze(1).repeat(1, 1, 2)

        tpu_graph_enabled = os.environ.get("PYTORCH_TPU_ALLOCATOR")
        if tpu_graph_enabled or self.mlp_buffer is None or hidden_states.shape[0] != self.mlp_buffer.shape[0]:
            self.mlp_buffer = torch.empty_like(hidden_states)
            self.rms_buffer = torch.empty_like(hidden_states)
            self.attn_buffer = []
            self.attn_buffer.append(hidden_states.new_empty(hidden_states.shape[0], (self.num_heads + 2 * self.num_kv_heads) * self.head_dim // self.tp_size))
            self.attn_buffer.append(hidden_states.new_empty((hidden_states.shape[0], self.num_heads // self.tp_size, self.head_dim)))
            self.attn_buffer.append(torch.empty_like(hidden_states))
        mlp_buffer = self.mlp_buffer
        rms_buffer = self.rms_buffer
        attn_buffer = self.attn_buffer
        default_device = hidden_states.device
        default_dtype = hidden_states.dtype
        query_buffer = torch.empty(hidden_states.shape[0], max(1, self.num_heads // self.tp_size), self.head_dim, dtype = default_dtype, device = default_device)
        key_buffer = torch.empty(hidden_states.shape[0], max(1, self.num_kv_heads // self.tp_size), self.head_dim, dtype = default_dtype, device = default_device)
        value_buffer = torch.empty(hidden_states.shape[0], max(1, self.num_kv_heads // self.tp_size), self.head_dim, dtype = default_dtype, device = default_device)
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                hidden_states,
                residual,
                attn_buffer,
                mlp_buffer,
                rms_buffer,
                cos,
                sin,
                query_buffer,
                key_buffer,
                value_buffer,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, rms_buffer, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # ("gate_up_proj", "gate_proj", 0),
            # ("gate_up_proj", "up_proj", 1),
        ]
        name_flags_1 = ["up_proj", "gate_proj", "down_proj"]
        name_flags_2 = [".qweight", ".qzeros", ".scales"]
        do_resize_list = [a + b for a in name_flags_1 for b in name_flags_2]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            elif any(do_resize_name in name for do_resize_name in do_resize_list):
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


class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen3Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

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
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

class Qwen3EmbeddingModel(nn.Module, SupportsLoRA, SupportsPP):
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
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        pooler_config = vllm_config.model_config.pooler_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen3Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        # TODO: Replace this model class with as_embedding_model(
        # Qwen3ForCausalLM) after changing the default pooling method
        if pooler_config.pooling_type is None:
            logger.warning(
                "This embedding model will default to last-token pooling in "
                "an upcoming version. To avoid breaking changes, you should "
                "pass `--override-pooler-config '{\"pooling_type\": \"MEAN\"}'`"
                " explicitly.")

        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.MEAN,
            normalize=True,
            softmax=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, kv_caches, attn_metadata,
                          intermediate_tensors)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = self.hf_to_vllm_mapper.apply(weights)
        weights = ((name, data) for name, data in weights
                   if not name.startswith("lm_head."))
        self.model.load_weights(weights)
