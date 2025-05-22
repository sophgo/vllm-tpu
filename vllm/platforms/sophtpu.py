from typing import TYPE_CHECKING, Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class SophTpuPlatform(Platform):
    _enum = PlatformEnum.SOPHTPU
    device_name: str = "sophtpu"
    device_type: str = "sophtpu"
    dispatch_key: str = "Torch-TPU"
    ray_device_key: str = "SOPHTPU"
    device_control_env_var: str = "SOPHTPU_VISIBLE_CHIPS"

    supported_quantization: list[str] = [
        "gptq",
    ]

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool,
                             use_mla: bool) -> str:
        if (selected_backend != _Backend.SOPHTPU_ATTN):
            logger.info("Cannot use %s backend on Sophon TPU.", selected_backend)

        logger.info("Using Sophon TPU Attention backend.")
        return "vllm.attention.backends.sophtpu_attn.SophTPUAttentionBackend"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "sophtpu"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return not envs.VLLM_USE_V1

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm.config import CompilationLevel

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        compilation_config = vllm_config.compilation_config

        if compilation_config.level != CompilationLevel.DYNAMO_ONCE:
            logger.info("[Sophon TPU] Forcing DYNAMO_ONCE compilation level")
            compilation_config.level = CompilationLevel.DYNAMO_ONCE

        if compilation_config.backend == "":
            compilation_config.backend = "torch-tpu"

        assert vllm_config.speculative_config is None, \
            "Sophon TPU does not support speculative decoding"

        if vllm_config.model_config.dtype in (torch.float16, torch.float32):
            logger.warning(
                "The Sophon TPU backend currently does not support %s. "
                "Using bfloat16 instead.", vllm_config.model_config.dtype)
            vllm_config.model_config.dtype = torch.bfloat16

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        if parallel_config.worker_cls == "auto":
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = \
                    "vllm.v1.worker.sophtpu_worker.SophTPUWorker"
            else:
                #if scheduler_config.is_multi_step:
                #    parallel_config.worker_cls = \
                #        "vllm.worker.multi_step_tpu_worker.MultiStepSophTPUWorker"
                #else:
                #    parallel_config.worker_cls = \
                #        "vllm.worker.sophtpu_worker.SophTPUWorker"
                assert not scheduler_config.is_multi_step
                parallel_config.worker_cls = \
                    "vllm.worker.sophtpu_worker.SophTPUWorker"

        # Adjust scheduler config for V1
        # TODO: Add support for these
        if envs.VLLM_USE_V1 and vllm_config.cache_config.enable_prefix_caching:
            logger.warning("[V1][Sophon TPU] Disable prefix caching")
            vllm_config.cache_config.enable_prefix_caching = False

        assert not vllm_config.speculative_config, (
            "Speculative decoding is not yet supported for Sophon TPU backend")

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on Sophon TPU.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.sophtpu_communicator.SophTpuCommunicator"  # noqa
