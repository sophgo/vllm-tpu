import os
import glob
import torch
from torch import nn

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from typing import (Any, Callable, Dict, Generator, Iterable, List, Optional,
                    Tuple, cast)

import vllm
from vllm.config import ModelConfig, VllmConfig
from vllm.attention import Attention
from vllm.transformers_utils.config import get_config
from vllm.model_executor.layers.linear import QKVCrossParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.utils import (device_loading_context,
                                                    initialize_model,
                                                    set_default_torch_dtype)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf, download_weights_from_hf,
    filter_duplicate_safetensors_files, filter_files_not_needed_for_inference,
    maybe_download_from_modelscope)

from vllm_sophon.hack.soph_utils import weight_reorder
from vllm_sophon.ops.soph_linear import SophRowParallelLinear, SophColumnParallelLinear, SophReplicatedLinear
from vllm_sophon.ops.soph_fused_moe import SophDeepseekV3FusedMoE

from vllm.logger import init_logger
logger = init_logger(__name__)

# hack for process_weights_after_loading
def process_weights_after_loading(model: nn.Module, model_config: ModelConfig,
                                   target_device: torch.device) -> None:
    for _, module in model.named_modules():
        if isinstance(module, QKVCrossParallelLinear):
            # NOTE(Isotr0py): special case for cross QKV layer because
            # q and kv proj aren't registered as submodules intentionally
            module.process_weights_after_loading()
            continue
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase) and not bool(getattr(getattr(quant_method, "quant_config", None), "is_checkpoint_fp8_serialized", False)):
            # When quant methods need to process weights after loading
            # (for repacking, quantizing, etc), they expect parameters
            # to be on the global target device. This scope is for the
            # case where cpu offloading is used, where we will move the
            # parameters onto device for processing and back off after.
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)

    # Currently only used by MLA.
    # NOTE: This intentionally happens after other modules so we can easily
    # decompress the weights for MLA.
    for _, module in model.named_modules():
        if isinstance(module, Attention) and \
            hasattr(module, "process_weights_after_loading"):
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            module.process_weights_after_loading(model_config.dtype)

    if model_config.quantization not in ['gptq', 'awq']:
        for name, module in model.named_modules():
            if isinstance(module, (SophRowParallelLinear, SophColumnParallelLinear, SophReplicatedLinear, SophDeepseekV3FusedMoE)):
                # For all model weights used on SophTPU, perform type checking and conversion.
                # Transpose and make contiguous the weights of down_proj in mlp.
                # The data types of embedding and norm are BF16, so they will not be modified for now.
                module.process_weights_after_loading(name)

def _prepare_weights(
    self,
    model_name_or_path: str,
    revision: Optional[str],
    fall_back_to_pt: bool,
    allow_patterns_overrides: Optional[list[str]],
) -> Tuple[str, List[str], bool]:
    """Prepare weights for the model.

    If the model is not local, it will be downloaded."""
    model_name_or_path = (maybe_download_from_modelscope(
        model_name_or_path, revision) or model_name_or_path)

    is_local = os.path.isdir(model_name_or_path)
    load_format = self.load_config.load_format
    use_safetensors = False
    index_file = SAFE_WEIGHTS_INDEX_NAME
    # Some quantized models use .pt files for storing the weights.
    if load_format == "auto":
        allow_patterns = ["*.safetensors", "*.bin"]
    elif (load_format == "safetensors"
            or load_format == "fastsafetensors"):
        use_safetensors = True
        allow_patterns = ["*.safetensors"]
    elif load_format == "mistral":
        use_safetensors = True
        allow_patterns = ["consolidated*.safetensors"]
        index_file = "consolidated.safetensors.index.json"
    elif load_format == "pt":
        allow_patterns = ["*.pt"]
    elif load_format == "npcache":
        allow_patterns = ["*.bin"]
    else:
        raise ValueError(f"Unknown load_format: {load_format}")

    if fall_back_to_pt:
        allow_patterns += ["*.pt"]

    if allow_patterns_overrides is not None:
        allow_patterns = allow_patterns_overrides

    if not is_local:
        hf_folder = download_weights_from_hf(
            model_name_or_path,
            self.load_config.download_dir,
            allow_patterns,
            revision,
            ignore_patterns=self.load_config.ignore_patterns,
        )
    else:
        hf_folder = model_name_or_path

    hf_config = get_config(hf_folder, trust_remote_code = True) # Load the ​**config.json** file from a Hugging Face model directory
    quantization_config = getattr(hf_config, "quantization_config", None)
    if quantization_config:
        reorder_id = weight_reorder(hf_folder, quantization_config.get('quant_method', None), hf_config.torch_dtype, quantization_config.get('group_size', None))
        if reorder_id:
            hf_folder = reorder_id

    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if len(hf_weights_files) > 0:
            if pattern == "*.safetensors":
                use_safetensors = True
            break

    if use_safetensors:
        # For models like Mistral-7B-Instruct-v0.3
        # there are both sharded safetensors files and a consolidated
        # safetensors file. Using both breaks.
        # Here, we download the `model.safetensors.index.json` and filter
        # any files not found in the index.
        if not is_local:
            download_safetensors_index_file_from_hf(
                model_name_or_path,
                index_file,
                self.load_config.download_dir,
                revision,
            )
        hf_weights_files = filter_duplicate_safetensors_files(
            hf_weights_files, hf_folder, index_file)
    else:
        hf_weights_files = filter_files_not_needed_for_inference(
            hf_weights_files)

    if len(hf_weights_files) == 0:
        raise RuntimeError(
            f"Cannot find any model weights with `{model_name_or_path}`")

    return hf_folder, hf_weights_files, use_safetensors


def load_model(self, vllm_config: VllmConfig,
               model_config: ModelConfig) -> nn.Module:
    """Load a model with the given configurations."""
    device_config = vllm_config.device_config
    load_config = vllm_config.load_config
    load_device = device_config.device if load_config.device is None else \
                  load_config.device
    target_device = torch.device(load_device)
    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            model = initialize_model(vllm_config=vllm_config,
                                     model_config=model_config)

        logger.debug("Loading weights on %s ...", load_device)
        # Quantization does not happen in `load_weights` but after it
        self.load_weights(model, model_config)
        process_weights_after_loading(model, model_config, target_device)
    return model.eval()

vllm.model_executor.model_loader.base_loader.BaseModelLoader.load_model = load_model
vllm.model_executor.model_loader.default_loader.DefaultModelLoader._prepare_weights = _prepare_weights
