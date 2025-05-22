import os
from typing import Optional

import torch
import torch_tpu
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
        self.gather_outputs = None
        self.world_size = self.device_group.size()

    def all_reduce(self, input_) -> torch.Tensor:
        torch.distributed.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_, dim) -> torch.Tensor:
        #fix after, init gather output before model infer
        if self.gather_outputs is None or self.gather_outputs[0].shape[0] != input_.shape[0]:
            buffer_shape = (self.world_size,) + input_.shape
            full_buffer = input_.new_empty(buffer_shape)
            self.gather_outputs = [full_buffer[i] for i in range(self.world_size)]

        torch.distributed.all_gather(
            self.gather_outputs,
            input_,
            group=self.device_group
        )
        return torch.cat(self.gather_outputs, dim=dim)
