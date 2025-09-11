from vllm.model_executor.layers.quantization.awq import AWQConfig, is_layer_skipped_awq
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
import torch
from typing import Optional, Union

from vllm_sophon.hack.soph_gptq import SophGPTQLinearMethod

def awq_get_quant_method(
    self, layer: torch.nn.Module, prefix: str
) -> Optional[Union["LinearMethodBase", "QuantizeMethodBase"]]:
    if isinstance(layer, LinearBase):
        if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
            return UnquantizedLinearMethod()
        return SophonAWQLinearMethod(self, prefix)

AWQConfig.get_quant_method = awq_get_quant_method

class SophonAWQLinearMethod(SophGPTQLinearMethod):
    def __init__(self, quant_config: AWQConfig, prefix: str):
        # The TPU backend only implements linear with gptq quantization.
        # So we use the gptq config to initialize the linear method.
        # The weight has been reordered to the gptq format before loading.
        assert quant_config.weight_bits == 4, \
                "Only 4-bit weight quantization is supported for AWQ."
        assert len(quant_config.modules_to_not_convert) == 0, \
                "modules_to_not_convert is not supported for AWQ."
        gptq_config = GPTQConfig(
            weight_bits=quant_config.weight_bits,
            group_size=quant_config.group_size,
            desc_act=False,
            lm_head_quantized=False,
            dynamic={}
        )
        super().__init__(gptq_config, prefix)