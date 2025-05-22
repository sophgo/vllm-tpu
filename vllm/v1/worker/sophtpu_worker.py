
import os
from typing import Dict, List, Optional

import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (init_distributed_environment, ensure_model_parallel_initialized,
                              init_world_group, GroupCoordinator)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.sophtpu_model_runner import ExecutionMode, SophTPUModelRunner

logger = init_logger(__name__)


class SophTPUWorker:

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

    def init_device(self):
        os.environ["PJRT_DEVICE"] = "SophTPU"
        torch.set_grad_enabled(False)
        torch.set_default_dtype(self.model_config.dtype)

        # Initialize the distributed environment.
        init_sophtpu_worker_distributed_environment(self.parallel_config,
                                                    self.rank,
                                                    self.distributed_init_method,
                                                    self.local_rank)

        # Device initialization should happen after initializing
        # the distributed runtime.
        self.device = torch.device(f"tpu:{self.local_rank}")
        self.device_config.device = self.device

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Increase the cache size limit, which is the maximum number of
        # dynamo graphs that can be compiled.
        # NOTE(woosuk): Usually, we compile 10-15 graphs for prefill and
        # 30-40 graphs for decode. 128 is an arbitrary safe number.
        torch._dynamo.config.cache_size_limit = 128

        # Init ModelRunner here, so that we have access to self.device.
        self.model_runner = SophTPUModelRunner(self.vllm_config, self.device)

    def determine_available_memory(self) -> int:
        '''
        Within the range of GPU memory utilization, 
        determine the available memory excluding that used by model operation; 
        subsequent updates should be implemented as soon as possible, 
        and the maximum number of generated TOKENs 
        and block size can also be used for calculation. 
        Unit: bytes        
        '''
        return 15769803770

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.rank == 0 else None

    def load_model(self) -> None:
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> KVCacheSpec:
        return self.model_runner.get_kv_cache_spec()

    def initialize_cache(self, kv_cache_configs: List[KVCacheConfig]) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        kv_cache_config = kv_cache_configs[self.rank]
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

_WORLD: Optional[GroupCoordinator] = None

def init_sophtpu_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""

    init_distributed_environment(
        world_size=parallel_config.world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=distributed_init_method,
        backend="sccl",
    )
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)
