#!/bin/bash

TORCH_TPU_TOP=$(pip show torch_tpu | grep Location | cut -d " " -f 2)/torch_tpu/
RUNTIME_TOP="/opt/tpuv7/tpuv7-current"

export LD_LIBRARY_PATH=${TORCH_TPU_TOP}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${RUNTIME_TOP}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${TPUDNN_TOP}/lib:$LD_LIBRARY_PATH

export TPUDNN_PATH=${TORCH_TPU_TOP}/lib

# cmodel
export TPU_EMULATOR_PATH=${TORCH_TPU_TOP}/lib/sg2260_cmodel_firmware.so
export TPU_KERNEL_MODULE_PATH=${TORCH_TPU_TOP}/lib/sg2260_cmodel_firmware.so

# decice
# set a specific kernel_module
# export TPUKERNEL_FIRMWARE_PATH=${TORCH_TPU_TOP}/lib/sg2260_kernel_module.so

export CMODEL_GLOBAL_MEM_SIZE=30719476736

export TPU_ALLOCATOR_FREE_DELAY_IN_MS=2000
export TPU_ALLOCATOR_REUSE=1
