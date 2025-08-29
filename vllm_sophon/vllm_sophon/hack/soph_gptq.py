from typing import Any, Dict, List, Optional, Union
import torch
from copy import deepcopy
from vllm.config import QuantizationConfig
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.quantization.gptq import GPTQConfig, ExllamaState, GPTQLinearMethod
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.vocab_parallel_embedding import (ParallelLMHead,
                                                                 UnquantizedEmbeddingMethod)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (override_config,
                                                                      get_dynamic_override)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)

# hack for get_linear_quant_method
def soph_get_linear_quant_method(
    config: QuantizationConfig,
    layer: torch.nn.Module,
    prefix: str,
    linear_method_cls: type,
):
    cloned_config = deepcopy(config)
    parallel_lm_head_quantized = isinstance(
        layer, ParallelLMHead) and cloned_config.lm_head_quantized
    if isinstance(layer, LinearBase) or parallel_lm_head_quantized:
        # False = skip module, None = no override, else = Positive match
        if get_dynamic_override(  # noqa: E712
                cloned_config,  # noqa: E712
                layer_name=prefix) == False:  # noqa: E712
            if parallel_lm_head_quantized:
                return UnquantizedEmbeddingMethod()
            return UnquantizedLinearMethod()

        if prefix:
            # Dynamic per module/layer rules may override base config
            override_config(cloned_config, prefix=prefix)
            
        return linear_method_cls(cloned_config, prefix)
    return None


# hack for get_quant_method
def soph_get_quant_method(self, layer: torch.nn.Module,
                          prefix: str) -> Optional["GPTQLinearMethod"]:
    return soph_get_linear_quant_method(self, layer, prefix, SophGPTQLinearMethod)

GPTQConfig.get_quant_method = soph_get_quant_method


class SophGPTQLinearMethod(GPTQLinearMethod):
    """Linear method for GPTQ on SophTPU.

    Inherits from `GPTQLinearMethod` with the following customizations:
    1. Overrides `create_weights` - Adapts to reordered weights and parameters
    2. Overrides `process_weights_after_loading` - Implements post-load weight processing
    3. Overrides `apply` - Integrates SophTPU's GPTQ operator via `torch.ops.my_ops.matmul_gptq_forward`

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQConfig, prefix):
        self.prefix = prefix
        self.tp_size = get_tensor_model_parallel_world_size()
        super().__init__(quant_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size  # Unused.
        weight_loader = extra_weight_attrs.get("weight_loader")
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        output_size_per_partition = sum(output_partition_sizes)
        if (output_size_per_partition % self.quant_config.pack_factor.numerator
                != 0):
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size
        exllama_state = ExllamaState.UNINITIALIZED
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None

        assert not self.quant_config.desc_act, (
        "For GPTQ-quantized model inference on Soph_TPU, "
        "models quantized with quant_config.desc_act=FALSE are required. ")
        if (input_size != input_size_per_partition
                and self.quant_config.group_size != -1):
            # we need to partition qzeros and scales for exllama kernel
            scale_and_zero_size = input_size_per_partition // group_size
            scale_and_zero_input_dim = 0

        """
        Adaptations cover three aspects:
        1. Shape
        2. Dtype
        3. output_dim and packed_dim_qw

        The shape transformation logic follows the implementation in `weight_reorder()` method.

        Note: `g_idx` and `bias` remain unchanged during reordering.
        """
        pack_factor = torch.iinfo(torch.uint8).bits // 4 # 2
        adjusted_input_size = input_size_per_partition // pack_factor
        adjusted_output_size = output_size_per_partition // pack_factor

        is_attention = "self_attn" in self.prefix
        is_mlp_gate_up = ("mlp.gate_proj" in self.prefix ) or ("mlp.up_proj" in self.prefix)
        is_mlp_down = "mlp.down_proj" in self.prefix

        if is_attention:
            qweight_shape = (output_size_per_partition, adjusted_input_size)
            qzeros_shape = (adjusted_output_size*2, # The compression ratio is reduced to pack_factor_uint8 with separated high/low 4-bit extended storage.
                            scale_and_zero_size//2  # After transposition, every 4 bits in the column direction are packed into 8 bits.
                            )
            scales_shape = (output_size_per_partition, scale_and_zero_size * int(16 // 8))
            scale_dtype = torch.uint8
            input_dim=1
            output_dim=0
        elif is_mlp_down:
            # down_qweight
            size = adjusted_input_size * self.tp_size // group_size
            n_align = self.tp_size
            block_size = self._compute_block_size(size, n_align)
            qweight_shape = (block_size, output_size_per_partition * group_size)

            # down_qzeros
            size = scale_and_zero_size * self.tp_size // pack_factor
            n_align = self.tp_size
            block_size = self._compute_block_size(size, n_align)
            qzeros_shape = (block_size, output_size_per_partition)

            # down_scales
            size = scale_and_zero_size * self.tp_size
            n_align = self.tp_size * 2
            block_size = self._compute_block_size(size, n_align)
            scales_shape = (block_size, output_size_per_partition)

            scale_dtype = params_dtype
            input_dim = 0
            output_dim = 1
        elif is_mlp_gate_up:
            size = output_size_per_partition * self.tp_size
            n_align = self.tp_size*256
            block_size = self._compute_block_size(size, n_align)

            qweight_shape = (block_size, adjusted_input_size)
            qzeros_shape = (block_size, scale_and_zero_size // pack_factor)
            scales_shape = (block_size, scale_and_zero_size)
            scale_dtype = params_dtype
            input_dim=1
            output_dim=0

        qweight = PackedvLLMParameter(
            data=torch.empty(
                qweight_shape, 
                dtype=torch.uint8
            ),
            input_dim=input_dim,
            output_dim=output_dim,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        g_idx = RowvLLMParameter(data=torch.tensor(
            [
                i // self.quant_config.group_size
                for i in range(input_size_per_partition)
            ],
            dtype=torch.int32,
        ),
                                 input_dim=0,
                                 weight_loader=weight_loader)
        qzeros_args = {
            "data":
            torch.empty(
                qzeros_shape, 
                dtype=torch.uint8
            ),
            "weight_loader":
            weight_loader
        }
        weight_scale_args = {
            "data":
            torch.empty(
                scales_shape, 
                dtype=scale_dtype
            ),
            "weight_loader":
            weight_loader
        }
        if scale_and_zero_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=output_dim,
                                                **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=output_dim,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        else:
            scales = GroupQuantScaleParameter(output_dim=output_dim,
                                              input_dim=input_dim,
                                              **weight_scale_args)
            qzeros = PackedvLLMParameter(
                input_dim=input_dim,
                output_dim=output_dim,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

        layer.exllama_state = exllama_state

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if "self_attn" in self.prefix:
            layer.scales.data = torch.cat((layer.scales.data, layer.qzeros.data),dim=-1).contiguous()

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              output_gptq: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        torch.ops.my_ops.matmul_gptq_forward(
            x,
            layer.qweight.data,
            bias,
            layer.scales,
            layer.scales,
            self.quant_config.group_size,
            self.quant_config.weight_bits,
            output_gptq,
        )
        return output_gptq

    def _compute_block_size(self, size: int, n_align: int) -> int:
        new_size = (size + n_align - 1) // n_align * n_align
        n_pad = new_size - size
        if n_pad > 0:
            size = new_size
        block_size = (size + self.tp_size - 1) // self.tp_size
        return block_size
