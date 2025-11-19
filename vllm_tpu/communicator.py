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
from typing import Optional

import torch
from torch.distributed import ProcessGroup
from vllm.distributed.device_communicators.base_device_communicator import \
    DeviceCommunicatorBase

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
        tpu_graph_enabled = os.environ.get("PYTORCH_TPU_ALLOCATOR")
        if tpu_graph_enabled or self.gather_outputs is None or self.gather_outputs[0].shape[0] != input_.shape[0]:
            buffer_shape = (self.world_size,) + input_.shape
            full_buffer = input_.new_empty(buffer_shape)
            self.gather_outputs = [full_buffer[i] for i in range(self.world_size)]

        torch.distributed.all_gather(
            self.gather_outputs,
            input_,
            group=self.device_group
        )
        return torch.cat(self.gather_outputs, dim=dim)
