#
# Copyright (c) 2025 SOPHGO Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-sophon project.
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
#

from typing import TYPE_CHECKING, Optional

import torch
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import vllm.envs as envs
from vllm.logger import init_logger

from vllm.platforms.interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.utils import FlexibleArgumentParser
else:
    VllmConfig = None
    FlexibleArgumentParser = None

logger = init_logger(__name__)


class SophTpuPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "sophtpu"
    device_type: str = "sophtpu"
    simple_compile_backend: str = "eager"  # Disable torch.compile()
    dispatch_key: str = "PrivateUse1"
    ray_device_key: str = ""
    device_control_env_var: str = "SOPHTPU_VISIBLE_CHIPS"

    supported_quantization: list[str] = [
        "gptq","fp8",
    ]

    @classmethod
    def pre_register_and_update(cls,
                                parser: Optional[FlexibleArgumentParser] = None
                                ) -> None:
        from vllm_sophon import hack

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool,
                             use_mla: bool) -> str:
        return "vllm_sophon.attention.attention.SophTPUAttentionBackend"

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
                    "vllm_sophon.v1.worker.sophtpu_worker.SophTPUWorker"
            else:
                #if scheduler_config.is_multi_step:
                #    parallel_config.worker_cls = \
                #        "vllm.worker.multi_step_tpu_worker.MultiStepSophTPUWorker"
                #else:
                #    parallel_config.worker_cls = \
                #        "vllm.worker.sophtpu_worker.SophTPUWorker"
                assert not scheduler_config.is_multi_step
                parallel_config.worker_cls = \
                    "vllm_sophon.worker.sophtpu_worker.SophTPUWorker"

        assert not vllm_config.speculative_config, (
            "Speculative decoding is not yet supported for Sophon TPU backend")

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on Sophon TPU.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_sophon.communicator.SophTpuCommunicator"  # noqa

    @classmethod
    def seed_everything(cls, seed: Optional[int] = None) -> None:
        if seed is not None:
            import random
            import numpy as np
            import torch_tpu
            random.seed(seed)
            np.random.seed(seed)
            torch_tpu.tpu.manual_seed_all(seed)

class SophConfigManager:
    def __init__(self):
        self.config: Dict[str, Any] = {}
        default_config = self._create_default_config()
        self.config = default_config

    def _create_default_config(self) -> None:
        return {
            "SIMULATE_RANK_NUM": int(os.getenv("SIMULATE_RANK_NUM", "1")),
            "SLOG_LEVEL": os.getenv("SLOG_LEVEL", "INFO"),
            "USE_SOPHTPU": os.getenv("DEVICE", "").upper() != "GPU",
            "KVCACHE_BLOCKS": int(os.getenv('KVCACHE_BLOCKS', '1024')),

            # Some global variables about DEBUG
            "DEBUG_MODE": os.getenv("DEBUG_MODE", "OFF").upper() == "ON",
            "DECODE_TOKEN_LEN": int(os.getenv("DECODE_TOKEN_LEN", "128")),
            "DEBUG_HIDDEN_LAYERS": int(os.getenv("DEBUG_HIDDEN_LAYERS", "1")),
            "TENSOR_DUMP": os.getenv("TENSOR_DUMP", "OFF").upper() == "ON",
            "TENSOR_DUMP_PATH": os.getenv("TENSOR_DUMP_PATH", "/workspace/dumped_tensors/"),

            "CONTEXT_LEN": int(os.getenv("CONTEXT_LEN", "6")),
            "MAX_IMG_TOKEN": int(os.getenv("MAX_IMG_TOKEN", "3000")),
            "ENABLE_PROFILE": int(os.getenv("ENABLE_PROFILE", "0")),
            "PROFILE_BOOK_KEEPING": int(os.getenv("PROFILE_BOOK_KEEPING", "1")),
            "PROFILE_STARTING_TOKEN": int(os.getenv("PROFILE_STARTING_TOKEN", "1")),
            "MAX_TOTAL_TOKENS": int(os.getenv('MAX_TOTAL_TOKENS', '4046')),

            "RANK": self._get_rank_value(),
            "WORLD_SIZE": self._get_world_size_value(),
            "LOCAL_WORLD_SIZE": str(os.getenv('WORLD_SIZE', '1')),
            "LOCAL_RANK": str(os.getenv('RANK', '0')),

            "SKIP_H2D": os.getenv("SKIP_H2D", "OFF").upper() == "ON",
            "USE_DUMMY_DATA": os.getenv("USE_DUMMY_DATA", "OFF").upper() == "ON",
            "BACKBONE_CMD_FORBID": os.getenv("BACKBONE_CMD_FORBID", "OFF").upper() == "ON",

            # Runtime dynamic variables
            "CURRENT_BATCH_SIZE": None,
            "NUM_PROMPTS_TOTAL_TOKENS": None
        }

    def _get_rank_value(self) -> int:
        if os.getenv("OMPI_COMM_WORLD_RANK"):
            return int(os.getenv("OMPI_COMM_WORLD_RANK"))
        return int(os.getenv("RANK", "0"))

    def _get_world_size_value(self) -> int:
        if os.getenv("OMPI_COMM_WORLD_SIZE"):
            return int(os.getenv("OMPI_COMM_WORLD_SIZE"))
        return int(os.getenv("WORLD_SIZE", "1"))

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

_config_manager: Optional[SophConfigManager] = None

def get_soph_config_manager() -> SophConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = SophConfigManager()
    return _config_manager
