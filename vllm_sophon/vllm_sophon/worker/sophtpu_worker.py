import gc
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, bind_kv_cache, get_dtype_size
from vllm.platforms import current_platform
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                     LoraNotSupportedWorkerBase, WorkerBase,
                                     WorkerInput)

from vllm_sophon.worker.sophtpu_model_runner import SophTPUModelRunner

logger = init_logger(__name__)

class SophTPUWorker(LoraNotSupportedWorkerBase, LocalOrDistributedWorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool,
    ) -> None:
        WorkerBase.__init__(self, vllm_config=vllm_config)
        device_config = self.device_config
        parallel_config = self.parallel_config
        assert device_config.device_type == "sophtpu"

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if parallel_config and is_driver_worker:
            assert rank % parallel_config.tensor_parallel_size == 0, \
                   "Driver worker should be rank 0 of tensor parallel group."

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        self.model_runner = SophTPUModelRunner(
            vllm_config=vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
        )

    def init_device(self) -> None:
        torch.set_grad_enabled(False)
        torch.set_default_dtype(self.model_config.dtype)

        init_distributed_environment(
            world_size=self.parallel_config.world_size,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size)

        self.device = torch.device(f"tpu:{self.local_rank}")
        self.device_config.device = self.device
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        head_size = self.model_config.get_head_size()
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)

        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value ``None``.
        import math
        num_sophtpu_blocks = max(1024, math.ceil(self.model_config.max_model_len / self.cache_config.block_size))
        num_sophtpu_blocks = (num_sophtpu_blocks // 8) * 8  # Round down to 8.

        # Calculate the CPU KV cache size based on the config.
        dtype_btyes = get_dtype_size(self.cache_dtype)
        block_size_bytes = (dtype_btyes * self.cache_config.block_size *
                            num_layers * 2 * head_size * num_kv_heads)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             block_size_bytes)
        num_cpu_blocks = (num_cpu_blocks // 8) * 8  # Round down to 8.
        return num_sophtpu_blocks, num_cpu_blocks

    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        self.block_size = self.cache_config.block_size

        dtype = self.cache_dtype
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()

        self.cpu_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.sophtpu_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        sophtpu_cache_shape = self.model_runner.attn_backend.get_kv_cache_shape(
            num_gpu_blocks, self.block_size, num_kv_heads, head_size)
        cpu_cache_shape = self.model_runner.attn_backend.get_kv_cache_shape(
            num_cpu_blocks, self.block_size, num_kv_heads, head_size)
        for _ in range(num_layers):
            sophtpu_k_cache = torch.empty(sophtpu_cache_shape,
                                      dtype=dtype,
                                      device=self.device)
            sophtpu_v_cache = torch.empty_like(sophtpu_k_cache)
            self.sophtpu_cache.append((sophtpu_k_cache, sophtpu_v_cache))
            cpu_k_cache = torch.empty(cpu_cache_shape,
                                      dtype=dtype,
                                      device="cpu")
            cpu_v_cache = torch.empty_like(cpu_k_cache)
            self.cpu_cache.append((cpu_k_cache, cpu_v_cache))
        bind_kv_cache(self.compilation_config.static_forward_context,
                      [self.sophtpu_cache])
        self._warmup_model()

    def _warmup_model(self) -> None:
        # FIXME(woosuk): Here we are abusing `enforce_eager` which is defined
        # for CUDA graphs. We should refactor this part.
        if not self.model_config.enforce_eager:
            # Warm up the model with all possible input shapes so that
            # compilation never happens during the actual execution.
            # This may take ~30 mins for the first run and ~20 mins for the
            # subsequent runs.
            # If `enforce_eager` is True, the ahead-of-time compilation is
            # skipped and the compilation happens during the actual execution,
            # which is bad for performance but useful for development.
            self.model_runner.warmup_model(self.sophtpu_cache)

    def get_cache_block_size_bytes(self) -> int:
        head_size = self.model_config.get_head_size()
        num_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        num_layers = self.model_config.get_num_layers(self.parallel_config)

        key_cache_block = self.cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = get_dtype_size(self.cache_dtype)
        return dtype_size * total

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        # NOTE(woosuk): This assumes virtual_engine == 0, i.e., no pipeline
        # parallelism.
        return [self.sophtpu_cache]

    def prepare_worker_input(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> WorkerInput:
        virtual_engine = execute_model_req.virtual_engine
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        blocks_to_swap_in = _make_src_to_dst(
            execute_model_req.blocks_to_swap_in, "cpu", self.device)
        blocks_to_swap_out = _make_src_to_dst(
            execute_model_req.blocks_to_swap_out, self.device, "cpu")
        blocks_to_copy = _make_src_to_dst(execute_model_req.blocks_to_copy,
                                          self.device, self.device)
        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
        )

    def execute_worker(self, worker_input: WorkerInput) -> None:
        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine[virtual_engine].swap_in(
                worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            self.cache_engine[virtual_engine].swap_out(
                worker_input.blocks_to_swap_out)
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)

def _make_src_to_dst(
    mapping: List[Tuple[int, int]],
    src_device: Union[torch.device, str],
    dst_device: Union[torch.device, str],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not mapping:
        return None

    src_indices = [i for i, _ in mapping]
    dst_indices = [i for _, i in mapping]
    src_indices = torch.tensor(src_indices,
                               device=src_device,
                               dtype=torch.int64)
    dst_indices = torch.tensor(dst_indices,
                               device=dst_device,
                               dtype=torch.int64)
    return src_indices, dst_indices

