from typing import List, Optional, Tuple, Type

import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from vllm.distributed import get_tensor_model_parallel_rank

import torch
from torch import nn
import torch_tpu
import torch.nn.functional as F
from transformers import PretrainedConfig
import numpy as np

from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.soph_linear import soph_to_dtype
from vllm.platforms import soph_config
from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)

class SophDeepseekV3FusedMoE(nn.Module):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 hidden_act,
                 quant_config,
                 n_routed_experts,
                 prefix,
                 params_dtype = None
                 ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.prefix = prefix 
        self.n_routed_experts = n_routed_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.quant_config = quant_config

        self.create_weight()

    def _get_weight_shape_dtype(self, intermediate_size, hidden_size, quant_config, proj_type):
        tp_size = get_tensor_model_parallel_world_size()
        block_quant = hasattr(quant_config, 'weight_block_size')

        if proj_type in ['gate_proj', 'up_proj']:
            input_size = hidden_size
            output_size = intermediate_size
            input_size_per_partition = input_size
            output_size_per_partition = output_size // tp_size
        else:  # down_proj
            input_size = intermediate_size
            output_size = hidden_size
            input_size_per_partition = input_size // tp_size
            output_size_per_partition = output_size

        weight_dtype = (torch.float8_e4m3fn if getattr(quant_config, 'is_checkpoint_fp8_serialized', False)
                    else torch.get_default_dtype())
        scale_dtype = torch.float32

        if block_quant:
            block_n, block_k = quant_config.weight_block_size
            if proj_type in ['gate_proj', 'up_proj']:
                assert output_size_per_partition % block_n == 0, (
                    f"{proj_type}: output_partition_size ({output_size_per_partition}) "
                    f"must be divisible by block_n ({block_n})"
                )
            else:  # down_proj
                assert input_size_per_partition % block_k == 0, (
                    f"{proj_type}: input_size_per_partition ({input_size_per_partition}) "
                    f"must be divisible by block_k ({block_k})"
                )

            rows = output_size_per_partition
            cols = input_size_per_partition
            n_blocks = (rows + block_n - 1) // block_n
            k_blocks = (cols + block_k - 1) // block_k

        return (output_size_per_partition, input_size_per_partition),(n_blocks, k_blocks) if block_quant else None,weight_dtype,scale_dtype

    def create_weight(self):
        block_quant = hasattr(self.quant_config, 'weight_block_size')

        gate_shape_info = self._get_weight_shape_dtype(self.intermediate_size, self.hidden_size, self.quant_config, 'gate_proj')
        up_shape_info = self._get_weight_shape_dtype(self.intermediate_size, self.hidden_size, self.quant_config, 'up_proj')
        down_shape_info = self._get_weight_shape_dtype(self.intermediate_size, self.hidden_size, self.quant_config, 'down_proj')

        gate_weight_shape, gate_scale_shape, gate_weight_dtype, gate_scale_dtype = gate_shape_info
        up_weight_shape, up_scale_shape, up_weight_dtype, up_scale_dtype = up_shape_info
        down_weight_shape, down_scale_shape, down_weight_dtype, down_scale_dtype = down_shape_info

        gate_weights = torch.nn.Parameter(torch.empty(self.n_routed_experts, *gate_weight_shape, dtype=gate_weight_dtype),requires_grad=False)
        gate_scales = torch.nn.Parameter(torch.empty(self.n_routed_experts, *gate_scale_shape, dtype=gate_scale_dtype),requires_grad=False)
        up_weights = torch.nn.Parameter(torch.empty(self.n_routed_experts, *up_weight_shape, dtype=up_weight_dtype),requires_grad=False)
        up_scales = torch.nn.Parameter(torch.empty(self.n_routed_experts, *up_scale_shape, dtype=up_scale_dtype),requires_grad=False)
        down_weights = torch.nn.Parameter(torch.empty(self.n_routed_experts, *down_weight_shape, dtype=down_weight_dtype),requires_grad=False)
        down_scales = torch.nn.Parameter(torch.empty(self.n_routed_experts, *down_scale_shape, dtype=down_scale_dtype),requires_grad=False)

        self.register_parameter("gate_weights", gate_weights)
        set_weight_attrs(gate_weights, {
            "quant_method": "BLOCK" if block_quant else "TENSOR",
            "output_dim": 0,
            "is_sharded_weight": True,
            "weight_loader": self.weight_loader
        })

        self.register_parameter("gate_scales", gate_scales)
        set_weight_attrs(gate_scales, {
            "quant_method": "BLOCK" if block_quant else "TENSOR",
            "output_dim": 0,
            "is_sharded_weight": True,
            "weight_loader": self.weight_loader
        })

        self.register_parameter("up_weights", up_weights)
        set_weight_attrs(up_weights, {
            "quant_method": "BLOCK" if block_quant else "TENSOR",
            "output_dim": 0,
            "is_sharded_weight": True,
            "weight_loader": self.weight_loader
        })

        self.register_parameter("up_scales", up_scales)
        set_weight_attrs(up_scales, {
            "quant_method": "BLOCK" if block_quant else "TENSOR",
            "output_dim": 0,
            "is_sharded_weight": True,
            "weight_loader": self.weight_loader
        })

        self.register_parameter("down_weights", down_weights)
        set_weight_attrs(down_weights, {
            "quant_method": "BLOCK" if block_quant else "TENSOR",
            "output_dim": 1 ,
            "is_sharded_weight": True,
            "weight_loader": self.weight_loader
        })

        self.register_parameter("down_scales", down_scales)
        set_weight_attrs(down_scales, {
            "quant_method": "BLOCK" if block_quant else "TENSOR",
            "output_dim": 1 ,
            "is_sharded_weight": True,
            "weight_loader": self.weight_loader
        })

    @classmethod
    def make_expert_params_mapping(
            cls, num_experts: int) -> List[Tuple[str, str, int]]:

        return [
            # (param_name, weight_name, expert_id)
            (  "experts.gate_weights" if weight_name == 'gate_proj.weight'
                else "experts.gate_scales" if weight_name == 'gate_proj.weight_scale_inv'
                else "experts.up_weights" if weight_name == 'up_proj.weight'
                else "experts.up_scales" if weight_name == 'up_proj.weight_scale_inv'
                else "experts.down_weights" if weight_name == 'down_proj.weight'
                else "experts.down_scales",  # down_proj.weight_scale_inv
                f"experts.{expert_id}.{weight_name}",
                expert_id
            )
            for expert_id in range(num_experts) for weight_name in [
                'gate_proj.weight',
                'gate_proj.weight_scale_inv',
                'up_proj.weight',
                'up_proj.weight_scale_inv',
                'down_proj.weight',
                'down_proj.weight_scale_inv'
            ]
        ]

    def weight_loader(self, loaded_weight, param, expert_id):

        expert_data = param.data[expert_id]
        tp_rank = get_tensor_model_parallel_rank()

        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        output_dim = getattr(param, "output_dim", None)

        if is_sharded_weight and output_dim is not None:
            shard_size = expert_data.shape[output_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        assert expert_data.shape == loaded_weight.shape, \
            f"Shape mismatch: {expert_data.shape} vs {loaded_weight.shape} for {self.prefix}.{self.proj_type}"
        expert_data.copy_(loaded_weight)

    def process_weights_after_loading(self, *prefix):

        self.gate_weights.data = soph_to_dtype(self.gate_weights.data, self.params_dtype)
        self.gate_scales.data = soph_to_dtype(self.gate_scales.data, self.params_dtype)
        self.up_weights.data = soph_to_dtype(self.up_weights.data, self.params_dtype)
        self.up_scales.data = soph_to_dtype(self.up_scales.data, self.params_dtype)
        self.down_weights.data = soph_to_dtype(self.down_weights.data, self.params_dtype)
        self.down_scales.data = soph_to_dtype(self.down_scales.data, self.params_dtype)

        self.down_weights.data = self.down_weights.data.transpose(1,2).contiguous()
        self.down_scales.data = self.down_scales.data.transpose(1,2).contiguous()

    def forward(self, gathered_experts_out_buf, x,
                output_sample, input_sample,
                selected_experts, routing_weights,
                num_select_experts,
                selected_experts_middle, routing_weights_middle,
                block_size, num_experts, num_experts_per_tok,
                use_grouped_topk, num_expert_group, topk_group,):
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
        return gathered_experts_out_buf
