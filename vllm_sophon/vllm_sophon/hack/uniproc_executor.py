from vllm.utils import (get_distributed_init_method, get_ip, get_open_port)
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.executor.uniproc_executor import UniProcExecutor

def UniProcExecutor_init_executor(self) -> None:
    """Initialize the worker and load the model.
        """
    self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config,
                                            rpc_rank=0)
    distributed_init_method = get_distributed_init_method(
        get_ip(), get_open_port())
    local_rank = 0
    rank = 0
    kwargs = dict(
        vllm_config=self.vllm_config,
        local_rank=local_rank,
        rank=rank,
        distributed_init_method=distributed_init_method,
        is_driver_worker=(not self.parallel_config)
        or (rank % self.parallel_config.tensor_parallel_size == 0),
    )
    self.collective_rpc("init_worker", args=([kwargs], ))
    self.collective_rpc("init_device")
    self.collective_rpc("load_model")

UniProcExecutor._init_executor = UniProcExecutor_init_executor