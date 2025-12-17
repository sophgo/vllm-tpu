from functools import wraps

import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup

import torch_tpu

from torch.distributed.constants import default_pg_timeout
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union)

import vllm
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.distributed.parallel_state import _get_unique_name, _register_group, init_world_group
#from vllm.distributed.parallel_state import init_distributed_environment

from vllm.utils import resolve_obj_by_qualname
"""
def new_group_wrapper(func):
    @wraps(func)
    def wrapper(ranks=None, timeout=default_pg_timeout, backend=None, pg_options=None, use_local_synchronization=False):
        if len(ranks) == 1:
            return func(ranks=ranks,
                        timeout=timeout,
                        backend="sccl",
                        pg_options=pg_options,
                        use_local_synchronization=use_local_synchronization)
        if pg_options is None:
            pg_options = torch_tpu.ProcessGroupSCCLOptions()
            torch_tpu.tpu.set_chip_map(pg_options, use_rank_table=False)
        return func(ranks=ranks, timeout=timeout, backend=backend, pg_options=pg_options, use_local_synchronization=use_local_synchronization)
    return wrapper


torch.distributed.new_group = new_group_wrapper(torch.distributed.new_group)
"""

def init_wrapper(func):
    @wraps(func)
    def wrapper(backend=None, init_method=None, timeout=default_pg_timeout, world_size=-1, rank=-1, store=None, group_name="", pg_options=None):
        if backend != "sccl":
            return func(backend=backend, init_method=init_method, timeout=timeout, world_size=world_size, rank=rank, store=store, group_name=group_name, pg_options=pg_options)
        if pg_options is None:
            pg_options = torch_tpu.ProcessGroupSCCLOptions()
        if world_size == -1:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        pg_options.chip_map = list(range(world_size))
        if rank == -1:
            rank = int(os.environ.get('RANK', '0'))
        torch_tpu.tpu.set_chip_map(pg_options, use_rank_table=False)
        torch_tpu.tpu.set_device(rank)
        return func(backend=backend, init_method=init_method, timeout=timeout, world_size=world_size, rank=rank, store=store, group_name=group_name, pg_options=pg_options)
    return wrapper

torch.distributed.init_process_group = init_wrapper(torch.distributed.init_process_group)


def GroupCoordinator__init__(
    self,
    group_ranks: List[List[int]],
    local_rank: int,
    torch_distributed_backend: Union[str, Backend],
    use_device_communicator: bool,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
):
    group_name = group_name or "anonymous"
    self.unique_name = _get_unique_name(group_name)
    _register_group(self)

    self.rank = torch.distributed.get_rank()
    self.local_rank = local_rank
    self.device_group = None
    self.cpu_group = None

    for ranks in group_ranks:
        import torch_tpu
        # device_group = torch.distributed.group.WORLD
        options = torch_tpu.ProcessGroupSCCLOptions()
        torch_tpu.tpu.set_chip_map(options, use_rank_table=False)
        device_group = torch.distributed.new_group(
            ranks, backend=torch_distributed_backend,pg_options=options,)
        # a group with `gloo` backend, to allow direct coordination between
        # processes through the CPU.
        cpu_group = torch.distributed.new_group(ranks, backend="gloo")
        if self.rank in ranks:
            self.ranks = ranks
            self.world_size = len(ranks)
            self.rank_in_group = ranks.index(self.rank)
            self.device_group = device_group
            self.cpu_group = cpu_group

    assert self.cpu_group is not None
    assert self.device_group is not None

    from vllm.platforms import current_platform
    # TODO: fix it for other platforms
    if current_platform.is_cuda_alike():
        self.device = torch.device(f"cuda:{local_rank}")
    elif current_platform.is_out_of_tree():
        self.device = torch.device(f"tpu:{local_rank}")
    else:
        self.device = torch.device("cpu")

    self.use_device_communicator = use_device_communicator

    self.device_communicator: DeviceCommunicatorBase = None  # type: ignore
    if use_device_communicator and self.world_size > 1:
        device_comm_cls = resolve_obj_by_qualname(
            current_platform.get_device_communicator_cls())
        self.device_communicator = device_comm_cls(
            cpu_group=self.cpu_group,
            device=self.device,
            device_group=self.device_group,
            unique_name=self.unique_name,
        )

    from vllm.distributed.device_communicators.shm_broadcast import (
        MessageQueue)
    self.mq_broadcaster: Optional[MessageQueue] = None
    if use_message_queue_broadcaster and self.world_size > 1:
        self.mq_broadcaster = MessageQueue.create_from_process_group(
            self.cpu_group, 1 << 22, 6)

    from vllm.platforms import current_platform
    self.use_custom_op_call = current_platform.is_cuda_alike()

#__init__ warpper can del after achieve the new_group wrapper && chip_map in torch_tpu
GroupCoordinator.__init__ = GroupCoordinator__init__

