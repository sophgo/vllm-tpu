# SPDX-License-Identifier: Apache-2.0

# This file is a pure Python wrapper for the SCCL library.
# A C/C++ binding is not flexible enough to handle this. It requires
# recompilation of the code every time we want to switch between different
# versions. This current implementation, with a **pure** Python wrapper, is
# more flexible. We can easily switch between different versions of SCCL by
# changing the environment variable `VLLM_SCCL_SO_PATH`, or the `so_file`
# variable in the code.

import ctypes
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.distributed import ReduceOp

from vllm.logger import init_logger
import os

logger = init_logger(__name__)

# === export types and functions from sccl to Python ===

def find_sccl_library() -> str:
        """
        We either use the library file specified by the `VLLM_SCCL_SO_PATH`
        environment variable, or we find the library file brought by PyTorch.
        After importing `torch_tpu`, `libsccl.so can be found by `ctypes` automatically.
        """
        so_file = os.environ.get("VLLM_NCCL_SO_PATH", None)

        # manually load the sccl library
        if so_file:
            logger.info(
                "Found sccl from environment variable VLLM_SCCL_SO_PATH=%s",
                so_file)
        else:
            import subprocess

            cmd = "python3 -m pip show torch_tpu | grep Location | awk '{print $2}'"
            torch_tpu_dist_path = subprocess.check_output(cmd, shell=True, text=True).strip()
            so_file = f"{torch_tpu_dist_path}/torch_tpu/lib/libsccl.so"
            logger.info("Found sccl from library %s", so_file)
        return so_file

scclResult_t = ctypes.c_int
scclComm_t = ctypes.c_void_p

class scclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]

scclHandle_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

scclDataType_t = ctypes.c_int


class scclDataTypeEnum:
    SCCL_DTYPE_FP32 = 0
    SCCL_DTYPE_FP16 = 1
    SCCL_DTYPE_INT8 = 2
    SCCL_DTYPE_UINT8 = 3
    SCCL_DTYPE_INT16 = 4
    SCCL_DTYPE_UINT16 = 5
    SCCL_DTYPE_INT32 = 6
    SCCL_DTYPE_UINT32 = 7
    SCCL_DTYPE_BF16 = 8
    SCCL_DTYPE_INT4 = 9
    SCCL_DTYPE_UINT4 = 10
    SCCL_DTYPE_FP20 = 11
    SCCL_DTYPE_FP8E5M2 = 12
    SCCL_DTYPE_FP8E4M3 = 13
    SCCL_DTYPE_INT64 = 14
    SCCL_DTYPE_BOOL = 15

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.SCCL_DTYPE_INT8
        if dtype == torch.uint8:
            return cls.SCCL_DTYPE_UINT8
        if dtype == torch.int32:
            return cls.SCCL_DTYPE_INT32
        if dtype == torch.int64:
            return cls.SCCL_DTYPE_INT64
        if dtype == torch.float16:
            return cls.SCCL_DTYPE_FP16
        if dtype == torch.float32:
            return cls.SCCL_DTYPE_FP32
        if dtype == torch.bfloat16:
            return cls.SCCL_DTYPE_BF16
        raise ValueError(f"Unsupported dtype: {dtype}")


scclReduceType_t = ctypes.c_int


class scclReduceTypeEnum:
    SCCL_REDUCE_MEAN = 0
    SCCL_REDUCE_SUM  = 1
    SCCL_REDUCE_MAX  = 2
    SCCL_REDUCE_MIN  = 3
    SCCL_REDUCE_PROD = 4

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.SCCL_REDUCE_SUM
        if op == ReduceOp.PRODUCT:
            return cls.SCCL_REDUCE_PROD
        if op == ReduceOp.MAX:
            return cls.SCCL_REDUCE_MAX
        if op == ReduceOp.MIN:
            return cls.SCCL_REDUCE_MIN
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class SCCLLibrary:
    exported_functions = [
        # scclResult_t scclGetUniqueId(
        #     scclHandle_t handle, scclUniqueId *uniqueId);
        # note that scclHandle_t is a pointer type, so the first argument
        # is a pointer
        Function("scclGetUniqueId", scclResult_t,
                 [scclHandle_t, ctypes.POINTER(scclUniqueId)]),

        # scclResult_t scclCommInitRank(
        #     scclComm_t *comm, int nRanks, scclUniqueId uniqueId,
        #     int rank, const int *chipMap);
        Function("scclCommInitRank", scclResult_t, [
            ctypes.POINTER(scclComm_t), ctypes.c_int, scclUniqueId,
            ctypes.c_int, ctypes.POINTER(ctypes.c_int)
        ]),

        Function("scclPhysToVirt", ctypes.c_void_p,
                 [scclHandle_t, ctypes.c_uint64]),

        # scclResult_t scclAllReduce(
        #     const void *sendBuff, void *recvBuff, uint64_t count,
        #     scclDataType_t dtype, scclReduceType_t op,
        #     scclComm_t comm, scclHandle_t handle);
        Function("scclAllReduce", scclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, scclDataType_t,
            scclReduceType_t, scclComm_t, scclHandle_t
        ]),

        # scclResult_t scclAllGather(
        #     const void *sendBuff, void *recvBuff,
        #     uint64_t send_count, scclDataType_t dtype,
        #     scclComm_t comm, scclHandle_t handle);
        # note that scclHandle_t is a pointer type, so the last argument
        # is a pointer
        Function("scclAllGather", scclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, scclDataType_t,
            scclComm_t, scclHandle_t
        ]),

        # scclResult_t scclScatter(
        #     const void *sendBuff, void *recvBuff,
        #     uint64_t recv_count, scclDataType_t dtype, int root,
        #     scclComm_t comm, scclHandle_t handle);
        Function("scclScatter", scclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, scclDataType_t,
            ctypes.c_int, scclComm_t, scclHandle_t
        ]),

        # scclResult_t scclSend(
        #     const void *send_buff, uint64_t send_count,
        #     scclDataType_t dtype, int dst_rank,
        #     scclComm_t comm, scclHandle_t handle);
        # note that scclHandle_t is a pointer type, so the last argument
        # is a pointer
        Function("scclSend", scclResult_t, [
            buffer_type, ctypes.c_size_t, scclDataType_t, ctypes.c_int,
            scclComm_t, scclHandle_t
        ]),

        # scclResult_t scclRecv(
        #     const void *recv_buff, uint64_t recv_count,
        #     scclDataType_t dtype, int src_rank,
        #     scclComm_t comm, scclHandle_t handle);
        # note that scclHandle_t is a pointer type, so the last argument
        # is a pointer
        Function("scclRecv", scclResult_t, [
            buffer_type, ctypes.c_size_t, scclDataType_t, ctypes.c_int,
            scclComm_t, scclHandle_t
        ]),

        # scclResult_t scclBroadcast(
        #     void *buff, uint64_t count, scclDataType_t dtype,
        #     int root, scclComm_t comm, scclHandle_t handle);
        Function("scclBroadcast", scclResult_t, [
            buffer_type, ctypes.c_size_t, scclDataType_t,
            ctypes.c_int, scclComm_t, scclHandle_t
        ]),

        # scclResult_t scclAllToAll(
        #     const void *sendBuff, void *recvBuff,
        #     uint64_t recv_count, scclDataType_t dtype,
        #     scclComm_t comm, scclHandle_t handle);
        Function("scclAllToAll", scclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, scclDataType_t,
            scclComm_t, scclHandle_t
        ]),

        # be cautious! this is a collective call, it will block until all
        # processes in the communicator have called this function.
        # because Python object destruction can happen in random order,
        # it is better not to call it at all.
        # scclResult_t scclCommDestroy(scclComm_t comm);
        Function("scclCommDestroy", scclResult_t, [scclComm_t]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        so_file = so_file or find_sccl_library()

        try:
            if so_file not in SCCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                SCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = SCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load SCCL library from %s ."
                "It is expected if you are not running on Sophon TPUs."
                "Otherwise, the sccl library might not exist, be corrupted "
                "or it does not support the current platform %s."
                "If you already have the library, please set the "
                "environment variable VLLM_SCCL_SO_PATH"
                " to point to the correct sccl library path.", so_file,
                platform.platform())
            raise e

        if so_file not in SCCLLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            for func in SCCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            SCCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = SCCLLibrary.path_to_dict_mapping[so_file]

    def SCCL_CHECK(self, result: scclResult_t) -> None:
        if result != 0:
            raise RuntimeError(f"SCCL error: {result}")

    def scclGetUniqueId(self, handle: scclHandle_t) -> scclUniqueId:
        unique_id = scclUniqueId()
        self.SCCL_CHECK(self._funcs["scclGetUniqueId"](
            handle, ctypes.byref(unique_id)))
        return unique_id

    def scclPhysToVirt(self, handle: scclHandle_t, addr: ctypes.c_uint64) -> ctypes.c_void_p:
        device_addr = (addr & ((1 << 64) - 1)) & ((1 << 58) - 1)
        virt_ptr = self._funcs["scclPhysToVirt"](handle, device_addr)
        return virt_ptr

    def scclCommInitRank(self, world_size: int, unique_id: scclUniqueId,
                         rank: int, chipMap=None) -> scclComm_t:
        comm = scclComm_t()
        chipMap = [0,1] # default for 1p1d
        if os.environ.get("CHIP_MAP"):
            chipMap = list(map(int, os.environ.get("CHIP_MAP").split(",")))
        if isinstance(chipMap, (list, tuple)):
            ArrT = ctypes.c_int * len(chipMap)
            chip_buf = ArrT(*chipMap)
            chip_ptr = chip_buf
        else:
            raise ValueError("CHIP_MAP is None")

        self.SCCL_CHECK(self._funcs["scclCommInitRank"](ctypes.byref(comm),
                                                        world_size, unique_id,
                                                        rank, chip_ptr))
        return comm

    def scclAllReduce(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, op: int, comm: scclComm_t,
                      handle: scclHandle_t) -> None:
        # `datatype` actually should be `scclDataType_t`
        # and `op` should be `scclReduceType_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.SCCL_CHECK(self._funcs["scclAllReduce"](sendbuff, recvbuff, count,
                                                     datatype, op, comm, handle))

    def scclAllGather(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, comm: scclComm_t,
                      handle: scclHandle_t) -> None:
        # `datatype` actually should be `scclDataType_t`
        # which is an aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.SCCL_CHECK(self._funcs["scclAllGather"](sendbuff, recvbuff, count,
                                                     datatype, comm, handle))

    def scclScatter(self, sendbuff: buffer_type, recvbuff: buffer_type,
                          count: int, datatype: int, root: int, comm: scclComm_t,
                          handle: scclHandle_t) -> None:
        # `datatype` actually should be `scclDataType_t`
        # and `op` should be `scclReduceType_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.SCCL_CHECK(self._funcs["scclScatter"](sendbuff, recvbuff,
                                                    count, datatype, root,
                                                    comm, handle))

    def scclSend(self, sendbuff: buffer_type, count: int, datatype: int,
                 dest: int, comm: scclComm_t, handle: scclHandle_t) -> None:
        self.SCCL_CHECK(self._funcs["scclSend"](sendbuff, count, datatype,
                                                dest, comm, handle))

    def scclRecv(self, recvbuff: buffer_type, count: int, datatype: int,
                 src: int, comm: scclComm_t, handle: scclHandle_t) -> None:
        self.SCCL_CHECK(self._funcs["scclRecv"](recvbuff, count, datatype, 
                                                src, comm, handle))

    def scclBroadcast(self, buff: buffer_type, count: int,
                      datatype: int, root: int, comm: scclComm_t,
                      handle: scclHandle_t) -> None:
        self.SCCL_CHECK(self._funcs["scclBroadcast"](buff, count, datatype,
                                                    root, comm, handle))

    def scclAllToAll(self, sendbuff: buffer_type, recvbuff: buffer_type,
                          count: int, datatype: int, comm: scclComm_t,
                          handle: scclHandle_t) -> None:
        # `datatype` actually should be `scclDataType_t`
        # and `op` should be `scclReduceType_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.SCCL_CHECK(self._funcs["scclAllToAll"](sendbuff, recvbuff,
                                                    count, datatype,
                                                    comm, handle))

    def scclCommDestroy(self, comm: scclComm_t) -> None:
        self.SCCL_CHECK(self._funcs["scclCommDestroy"](comm))


__all__ = [
    "SCCLLibrary", "scclDataTypeEnum", "scclReduceTypeEnum", "scclUniqueId",
    "scclComm_t", "scclHandle_t", "buffer_type"
]
