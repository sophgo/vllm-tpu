# SPDX-License-Identifier: Apache-2.0

import itertools
from abc import abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter

from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
# yapf: disable
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           BlockQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           PerTensorScaleParameter,
                                           RowvLLMParameter)
# yapf: enable
from vllm.model_executor.utils import set_weight_attrs

from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear,
                                               ColumnParallelLinear,
                                               ReplicatedLinear)

logger = init_logger(__name__)

def soph_to_dtype(tensor, dtype, to_dtype=True):
    if (
        tensor.dtype
        not in [
            torch.float8_e4m3fn,
            torch.int16,
            torch.int32,
            torch.int64,
        ]
        and to_dtype
    ):
        return tensor.to(dtype=dtype)
    else:
        return tensor

class SophQKVParallelLinear(QKVParallelLinear):
    """
    Inherits from `QKVParallelLinear` and adds the `output_gptq` parameter
    to the `forward` method, designed to capture outputs from the SophTPU
    gptq inference function `torch.ops.my_ops.matmul_gptq_forward`.
    """
    def __init__(self,
                hidden_size: int,
                head_size: int,
                total_num_heads: int,
                total_num_kv_heads: Optional[int] = None,
                bias: bool = True,
                skip_bias_add: bool = False,
                params_dtype: Optional[torch.dtype] = None,
                quant_config: Optional[QuantizationConfig] = None,
                prefix: str = ""):
        super().__init__(hidden_size, head_size, total_num_heads, total_num_kv_heads, bias, skip_bias_add, params_dtype, quant_config, prefix)

    def forward(self, input_, output_gptq=None) -> tuple[torch.Tensor, Optional[Parameter]]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        if getattr(self.quant_method, 'quant_config', None):
            output_parallel = self.quant_method.apply(self,
                                                      input_,
                                                      output_gptq,
                                                      bias)
        else:
            output_parallel = F.linear(input_, self.weight, bias=bias, out=output_gptq)

        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

class SophRowParallelLinear(RowParallelLinear):
    """
    Inherits from `RowParallelLinear` and adds the `output_gptq` parameter
    to the `forward` method, designed to capture outputs from the SophTPU
    gptq inference function `torch.ops.my_ops.matmul_gptq_forward`.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 input_is_parallel: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 reduce_results: bool = True,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__(input_size,
                 output_size,
                 bias,
                 input_is_parallel,
                 skip_bias_add,
                 params_dtype,
                 reduce_results,
                 quant_config,
                 prefix)
    def forward(self, input_, output_gptq=None) -> tuple[torch.Tensor, Optional[Parameter]]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        if getattr(self.quant_method, 'quant_config', None):
            output_parallel = self.quant_method.apply(self,
                                                    input_parallel,
                                                    output_gptq,
                                                    bias=bias_)
        else:
            output_parallel = F.linear(input_parallel, self.weight, bias=bias_, out=output_gptq)

        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
    
    def process_weights_after_loading(self, name, *prefix):
        self.weight.data = soph_to_dtype(self.weight.data, self.params_dtype)
        if hasattr(self, 'weight_scale_inv') and self.weight_scale_inv is not None:
            self.weight_scale_inv.data = soph_to_dtype(self.weight_scale_inv.data, self.params_dtype)
        if 'mlp' in name:
            self.weight.data = self.weight.data.transpose(0,1).contiguous()
            if hasattr(self, 'weight_scale_inv') and self.weight_scale_inv is not None:
                self.weight_scale_inv.data = self.weight_scale_inv.data.transpose(0,1).contiguous()
        if 'linear_1' in name:
            self.weight.data = self.weight.data.transpose(0,1).contiguous()

class SophColumnParallelLinear(ColumnParallelLinear):
    """
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 gather_output: bool = False,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 output_sizes: Optional[list[int]] = None,
                 prefix: str = ""):

        super().__init__(input_size, output_size, bias, gather_output, skip_bias_add, params_dtype,
                         quant_config, output_sizes, prefix)

    def forward(self, input_, output_gptq=None) -> tuple[torch.Tensor, Optional[Parameter]]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        if getattr(self.quant_method, 'quant_config', None):
            output_parallel = self.quant_method.apply(self,
                                                    input_,
                                                    output_gptq,
                                                    bias=bias)
        else:
            output_parallel = F.linear(input_, self.weight, bias=bias_, out=output_gptq)

        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def process_weights_after_loading(self, name, *prefix):
        self.weight.data = soph_to_dtype(self.weight.data, self.params_dtype)
        if hasattr(self, 'weight_scale_inv') and self.weight_scale_inv is not None:
            self.weight_scale_inv.data = soph_to_dtype(self.weight_scale_inv.data, self.params_dtype)
        if 'linear_1' in name:
            self.weight.data = self.weight.data.transpose(0,1).contiguous()

class SophReplicatedLinear(ReplicatedLinear):
    """
    Inherits from `ReplicatedLinear` and adds the `output_gptq` parameter
    to the `forward` method, designed to capture outputs from the SophTPU
    gptq inference function `torch.ops.my_ops.matmul_gptq_forward`.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__(input_size,
                         output_size,
                         bias,
                         skip_bias_add,
                         params_dtype,
                         quant_config,
                         prefix=prefix)

    def forward(self,
                x: torch.Tensor,
                output_gptq: torch.Tensor=None) -> tuple[torch.Tensor, Optional[Parameter]]:
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None
        if getattr(self.quant_method, 'quant_config', None):
            output = self.quant_method.apply(self,
                                                x,
                                                output_gptq,
                                                bias)
        else:
            output_parallel = F.linear(x, self.weight, bias=bias, out=output_gptq)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def process_weights_after_loading(self, *prefix):
        self.weight.data = soph_to_dtype(self.weight.data, self.params_dtype)
        if hasattr(self, 'weight_scale_inv') and self.weight_scale_inv is not None:
            self.weight_scale_inv.data = soph_to_dtype(self.weight_scale_inv.data, self.params_dtype)
