# SPDX-License-Identifier: Apache-2.0

import enum
from enum import Enum
from fractions import Fraction
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_linear_quant_method)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)
from vllm.platforms import current_platform


class GPTQConfig(QuantizationConfig):
    """Config class for GPTQ.

    Reference: https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        lm_head_quantized: bool,
        dynamic: Dict[str, Dict[str, Union[int, bool]]],
    ) -> None:
        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # Format is Dict[str, Dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # Negative matching will skip quantization init for this module
        # entirely:
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # Example:
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix
        #  # positive match layer 10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers
        # }
        super().__init__()
        self.dynamic = dynamic

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.pack_factor = Fraction(32, self.weight_bits)
        if self.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ, but got {self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"GPTQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act}),"
                f"lm_head_quantized={self.lm_head_quantized}), "
                f"dynamic={self.dynamic}")

    @classmethod
    def get_name(cls) -> str:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQConfig":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic

        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        return cls(weight_bits, group_size, desc_act, lm_head_quantized,
                   dynamic)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["GPTQLinearMethod"]:
        if current_platform.is_sophtpu():
            return get_linear_quant_method(self, layer, prefix, SophGPTQLinearMethod)
        else:
            return get_linear_quant_method(self, layer, prefix, GPTQLinearMethod)


class ExllamaState(Enum):

    UNUSED = enum.auto()
    UNINITIALIZED = enum.auto()
    READY = enum.auto()


class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

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
        if (input_size != input_size_per_partition
                and self.quant_config.group_size != -1):
            # For act-order models, we cannot use Exllama for row parallel layer
            if self.quant_config.desc_act:
                exllama_state = ExllamaState.UNUSED
            else:
                # we need to partition qzeros and scales for exllama kernel
                scale_and_zero_size = input_size_per_partition // group_size
                scale_and_zero_input_dim = 0

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
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
                scale_and_zero_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader":
            weight_loader
        }
        weight_scale_args = {
            "data":
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader":
            weight_loader
        }
        if scale_and_zero_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1,
                                                **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        else:
            scales = GroupQuantScaleParameter(output_dim=1,
                                              input_dim=0,
                                              **weight_scale_args)
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

        layer.exllama_state = exllama_state

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # for torch.compile
        layer.qzeros = Parameter(layer.qzeros.data, requires_grad=False)
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.g_idx = Parameter(layer.g_idx.data, requires_grad=False)
        layer.scales = Parameter(layer.scales.data, requires_grad=False)

        # exllama needs to shuffle the weight after the weight is loaded
        # here we do the shuffle on first forward pass
        if layer.exllama_state == ExllamaState.UNINITIALIZED:
            if self.quant_config.desc_act:
                layer.g_idx.data = torch.argsort(layer.g_idx).to(torch.int)
            else:
                layer.g_idx.data = torch.empty((0, ),
                                               dtype=torch.int,
                                               device=layer.g_idx.device)
            layer.exllama_state = ExllamaState.READY
            ops.gptq_shuffle(layer.qweight, layer.g_idx,
                             self.quant_config.weight_bits)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        out_shape = x.shape[:-1] + (layer.qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])

        output = ops.gptq_gemm(reshaped_x, layer.qweight, layer.qzeros,
                               layer.scales, layer.g_idx,
                               layer.exllama_state == ExllamaState.READY,
                               self.quant_config.weight_bits)
        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)

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
        if (input_size != input_size_per_partition
                and self.quant_config.group_size != -1):
            # For act-order models, we cannot use Exllama for row parallel layer
            if self.quant_config.desc_act:
                exllama_state = ExllamaState.UNUSED
            else:
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
        pack_factor_int32 = self.quant_config.pack_factor.numerator # pack_factor_int32 = 8
        pack_factor_uint8 = torch.iinfo(torch.uint8).bits // 4 # 2
        pack_factor = pack_factor_uint8 if current_platform.is_sophtpu() else pack_factor_int32
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
