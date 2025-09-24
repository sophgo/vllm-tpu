import os
from typing import Dict, List, Optional

import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (init_distributed_environment, ensure_model_parallel_initialized,
                              get_tensor_model_parallel_world_size,
                              init_world_group, GroupCoordinator)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.distributed.parallel_state import _WORLD, init_world_group
from vllm_sophon.platform import get_soph_config_manager
from vllm_sophon.v1.worker.sophtpu_model_runner import ExecutionMode, SophTPUModelRunner, SophAttentionSpec, SophTPUAttentionBackend
import math



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

        from vllm_sophon import ops
        from vllm_sophon import hack

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
        torch.set_grad_enabled(False)
        torch.set_default_dtype(self.model_config.dtype)

        import os
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.parallel_config.world_size)
        os.environ["LOCAL_RANK"] = str(self.rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(self.parallel_config.world_size)
        os.environ["OMPI_COMM_WORLD_RANK"] = str(self.rank)
        os.environ["OMPI_COMM_WORLD_SIZE"] = str(self.parallel_config.world_size)

        # logger.info("Initializing SCCL backend...")

        def set_rank_affinity():
            import psutil
            cpu_affinity = [self.rank]
            p = psutil.Process(os.getpid())
            p.cpu_affinity(cpu_affinity)

        set_rank_affinity()

        # Initialize the distributed environment.
        init_sophon_worker_distributed_environment(self.parallel_config,
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


    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the free memory that can be used for KV cache in
        bytes.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        kv_caches: Dict[str, torch.Tensor] = {}
        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        for layer_name, layer_spec in kv_cache_spec.items():
            if isinstance(layer_spec, SophAttentionSpec):
                dtype = layer_spec.dtype
                min_num_block = (
                    self.scheduler_config.max_num_batched_tokens
                    + self.cache_config.block_size
                ) // self.cache_config.block_size
                # dummy kv cache for profiling
                kv_cache_shape = SophTPUAttentionBackend.get_kv_cache_shape(
                    min_num_block, self.cache_config.block_size, 1, 1)
                dtype = layer_spec.dtype

                if self.model_config.is_deepseek_mla:
                    assert self.model_config.use_mla
                    qk_rope_head_dim = getattr(self.model_config.hf_text_config, "qk_rope_head_dim", 0)
                    kv_lora_rank = self.model_config.hf_text_config.kv_lora_rank
                    tpu_kv_cache = torch.empty((kv_cache_shape[0], kv_cache_shape[1], kv_lora_rank),
                                               dtype=dtype,
                                               device=self.device)
                    tpu_pe_cache = torch.empty((kv_cache_shape[0], kv_cache_shape[1], qk_rope_head_dim),
                                               dtype=dtype,
                                               device=self.device)

                    kv_caches[layer_name] = (tpu_kv_cache, tpu_pe_cache)
                else:
                    tpu_k_cache = torch.empty(kv_cache_shape,
                                              dtype=dtype,
                                              device=self.device)
                    tpu_v_cache = torch.empty_like(tpu_k_cache)

                    kv_caches[layer_name] = (tpu_k_cache, tpu_v_cache)
            else:
                raise NotImplementedError

        torch.tpu.empty_cache()
        torch.tpu.synchronize()
        torch.tpu.reset_peak_memory_stats()
        _, total_gpu_memory = torch.tpu.mem_get_info()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        runner_kv_caches: List[torch.Tensor] = []
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            runner_kv_caches)
        
        logger.info(f'---- START DUMMY RUN ----')

        self.model_runner.dummy_run(
            runner_kv_caches,
            num_tokens=1,
            seq_len=self.scheduler_config.max_num_batched_tokens,
            exec_mode=ExecutionMode.PREFILL,
        )

        # Get the peak memory allocation recorded by torch
        peak_memory = torch.tpu.memory_stats()["allocated_bytes.all.peak"]

        # Check for any memory left around that may have been allocated on the
        # gpu outside of `torch`. NCCL operations, for example, can use a few
        # GB during a forward pass
        torch.tpu.empty_cache()
        torch.tpu.synchronize()
        logger.info(f'---- FINISH DUMMY RUN ----')
        torch_allocated_bytes = torch.tpu.memory_stats(
        )["allocated_bytes.all.current"]
        total_allocated_bytes = torch.tpu.mem_get_info(
        )[1] - torch.tpu.mem_get_info()[0]
        non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations
        available_kv_cache_memory = (
            total_gpu_memory * self.cache_config.gpu_memory_utilization -
            peak_memory)

        return int(available_kv_cache_memory)

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

def init_sophon_worker_distributed_environment(
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
