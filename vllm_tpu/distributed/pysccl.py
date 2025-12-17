# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp
import torch_tpu

from vllm_sophon.distributed.pysccl_wrapper import (
    SCCLLibrary, scclHandle_t, scclComm_t, scclDataTypeEnum,
    scclReduceTypeEnum, scclUniqueId)
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

logger = init_logger(__name__)


class PyScclCommunicator:

    def __init__(
        self,
        group: Union[ProcessGroup, StatelessProcessGroup],
        device: Union[int, str, torch.device],
        library_path: Optional[str] = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyScclCommunicator to. If None,
                it will be bind to f"tpu:{local_rank}".
            library_path: the path to the SCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert dist.get_backend(group) != dist.Backend.SCCL, (
                "PyScclCommunicator should be attached to a non-SCCL group.")
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        if self.world_size == 1:
            self.available = False
            self.disabled = True
            return
        try:
            self.sccl = SCCLLibrary(library_path)
        except Exception:
            # disable because of missing SCCL library
            # e.g. in a non-TPUenvironment
            self.available = False
            self.disabled = True
            raise ValueError("can not find SCCLLibrary")

        self.sccl = SCCLLibrary(library_path)

        self.available = True
        self.disabled = False

        if self.rank == 0:
            # get the unique id from SCCL
            stream = torch_tpu.tpu.current_stream()
            self.unique_id = self.sccl.scclGetUniqueId(stream.tpudnn_handle)
        else:
            # construct an empty unique id
            self.unique_id = scclUniqueId()

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            # arg `src` in `broadcast` is the global rank
            dist.broadcast(tensor, src=ranks[0], group=group)
            char_list = tensor.tolist()
            for i, char in enumerate(char_list):
                self.unique_id.internal[i] = char
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)

        if isinstance(device, int):
            device = torch.device(f"tpu:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        # sccl communicator and stream will use this device
        with torch_tpu.tpu.device(device):
            self.comm: scclComm_t = self.sccl.scclCommInitRank(
                self.world_size, self.unique_id, self.rank)

            stream = torch_tpu.tpu.current_stream()
            # A small all_reduce for warmup. like barrier
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            stream.synchronize()
            del data

    def all_reduce(self,
                   input_tensor: torch.Tensor,
                   op: ReduceOp = ReduceOp.SUM,
                   stream=None) -> torch.Tensor:
        if self.disabled:
            return None
        # sccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this sccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")

        output_tensor = torch.empty_like(input_tensor)
        if stream is None:
            stream = torch_tpu.tpu.current_stream()
        sendBuff = self.sccl.scclPhysToVirt(scclHandle_t(stream.tpudnn_handle), input_tensor.data_ptr())
        recvBuff = self.sccl.scclPhysToVirt(scclHandle_t(stream.tpudnn_handle), output_tensor.data_ptr())
        self.sccl.scclAllReduce(sendBuff,
                                recvBuff,
                                input_tensor.numel(),
                                scclDataTypeEnum.from_torch(input_tensor.dtype),
                                scclReduceTypeEnum.from_torch(op), self.comm,
                                scclHandle_t(stream.tpudnn_handle))
        return output_tensor

    def all_gather(self,
                   output_tensor: torch.Tensor,
                   input_tensor: torch.Tensor,
                   stream=None):
        if self.disabled:
            return
        # sccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this sccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")
        if stream is None:
            stream = torch_tpu.tpu.current_stream()
        sendBuff = self.sccl.scclPhysToVirt(scclHandle_t(stream.tpudnn_handle), input_tensor.data_ptr())
        recvBuff = self.sccl.scclPhysToVirt(scclHandle_t(stream.tpudnn_handle), output_tensor.data_ptr())
        self.sccl.scclAllGather(
            sendBuff, recvBuff, input_tensor.numel(),
            scclDataTypeEnum.from_torch(input_tensor.dtype), self.comm,
            scclHandle_t(stream.tpudnn_handle))

    def scatter(self,
                output_tensor: torch.Tensor,
                input_tensor: torch.Tensor,
                src: int,
                stream=None):
        if self.disabled:
            return
        assert input_tensor.device == self.device, (
            f"this sccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")
        if stream is None:
            stream = torch_tpu.tpu.current_stream()
        sendBuff = self.sccl.scclPhysToVirt(scclHandle_t(stream.tpudnn_handle), input_tensor.data_ptr())
        recvBuff = self.sccl.scclPhysToVirt(scclHandle_t(stream.tpudnn_handle), output_tensor.data_ptr())
        self.sccl.scclScatter(
            sendBuff, recvBuff, output_tensor.numel(),
            scclDataTypeEnum.from_torch(input_tensor.dtype), src,
            self.comm, scclHandle_t(stream.tpudnn_handle))

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this sccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = torch_tpu.tpu.current_stream()
        sendBuff = self.sccl.scclPhysToVirt(scclHandle_t(stream.tpudnn_handle), tensor.data_ptr())
        self.sccl.scclSend(sendBuff, tensor.numel(),
                           scclDataTypeEnum.from_torch(tensor.dtype), dst,
                           self.comm, scclHandle_t(stream.tpudnn_handle))

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this sccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = torch_tpu.tpu.current_stream()
        recvBuff = self.sccl.scclPhysToVirt(scclHandle_t(stream.tpudnn_handle), tensor.data_ptr())
        self.sccl.scclRecv(recvBuff, tensor.numel(),
                           scclDataTypeEnum.from_torch(tensor.dtype), src,
                           self.comm, scclHandle_t(stream.tpudnn_handle))

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this sccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = torch_tpu.tpu.current_stream()

        buff = self.sccl.scclPhysToVirt(scclHandle_t(stream.tpudnn_handle), tensor.data_ptr())
        self.sccl.scclBroadcast(buff, tensor.numel(),
                                scclDataTypeEnum.from_torch(tensor.dtype), src,
                                self.comm, scclHandle_t(stream.tpudnn_handle))

    def alltoall(self,
                output_tensor: torch.Tensor,
                input_tensor: torch.Tensor,
                stream=None):
        if self.disabled:
            return
        assert input_tensor.device == self.device, (
            f"this sccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")
        if stream is None:
            stream = torch_tpu.tpu.current_stream()
        sendBuff = self.sccl.scclPhysToVirt(scclHandle_t(stream.tpudnn_handle), input_tensor.data_ptr())
        recvBuff = self.sccl.scclPhysToVirt(scclHandle_t(stream.tpudnn_handle), output_tensor.data_ptr())
        self.sccl.scclAllToAll(
            sendBuff, recvBuff, output_tensor.numel(),
            scclDataTypeEnum.from_torch(input_tensor.dtype),
            self.comm, scclHandle_t(stream.tpudnn_handle))
