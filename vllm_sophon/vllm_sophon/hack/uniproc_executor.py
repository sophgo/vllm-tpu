from vllm.utils import (get_distributed_init_method, get_ip, get_open_port)
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.executor.uniproc_executor import UniProcExecutor
from concurrent.futures import Future, ThreadPoolExecutor
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import worker_receiver_cache_from_config
from multiprocessing import Lock
def UniProcExecutor_init_executor(self) -> None:
    """Initialize the worker and load the model.
        """
    self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config,
                                            rpc_rank=0)
    distributed_init_method, rank, local_rank = self._distributed_args()
    is_driver_worker = True
    kwargs = dict(
        vllm_config=self.vllm_config,
        local_rank=local_rank,
        rank=rank,
        distributed_init_method=distributed_init_method,
        is_driver_worker=is_driver_worker,
    )
    self.mm_receiver_cache = worker_receiver_cache_from_config(
        self.vllm_config, MULTIMODAL_REGISTRY, Lock())

    self.async_output_thread: Optional[ThreadPoolExecutor] = None
    if self.max_concurrent_batches > 1:
        self.async_output_thread = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="WorkerAsyncOutput")

    self.collective_rpc("init_worker", args=([kwargs], ))
    self.collective_rpc("init_device")
    self.collective_rpc("load_model")

def UniProcExecutor_distributed_args(self) -> tuple[str, int, int]:
    """Return (distributed_init_method, rank, local_rank)."""
    distributed_init_method = get_distributed_init_method(
        get_ip(), get_open_port())
    # set local rank as the device index if specified
    return distributed_init_method, 0, 0

UniProcExecutor._init_executor = UniProcExecutor_init_executor
UniProcExecutor._distributed_args = UniProcExecutor_distributed_args
