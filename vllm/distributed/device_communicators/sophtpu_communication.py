import os
from typing import Optional

import torch
from torch.distributed import ProcessGroup

from vllm.platforms import current_platform

from .base_device_communicator import DeviceCommunicatorBase


class SophTpuCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)

    def all_reduce(self, input_) -> torch.Tensor:
        dist.all_reduce(input_, group=self.device_group)
        return input_
