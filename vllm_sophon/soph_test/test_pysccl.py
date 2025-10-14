import os
import torch
import torch_tpu
import torch.distributed as dist
from vllm.distributed.utils import StatelessProcessGroup
import torch.multiprocessing as mp

from vllm_sophon.distributed.pysccl import PyScclCommunicator

def all_gather_worker_for_single_rank(rank=0, world_size=1):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "6000"
    local_rank = rank
    print(f"Hello from rank {local_rank}/{world_size}.")

    device = torch.device(f"tpu:{local_rank}")
    torch_tpu.tpu.set_device(local_rank)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    group = dist.group.WORLD
    comm = PyScclCommunicator(group=group, device=local_rank)

    tensor_len = 16
    dtype = torch.float16
    input_tensor = torch.ones(tensor_len, dtype=dtype)
    if torch_tpu.tpu.current_device() == device.index:
        input_tensor = input_tensor.to(device)
    print("created input_tensor", input_tensor.shape, input_tensor.dtype, input_tensor.device, "numel", input_tensor.numel())

    output_tensor = torch.zeros(tensor_len * world_size, dtype=dtype)
    if torch_tpu.tpu.current_device() == device.index:
        output_tensor = output_tensor.to(device)
    print("created output_tensor", output_tensor.shape, output_tensor.dtype, output_tensor.device, "numel", output_tensor.numel())

    comm.all_gather(output_tensor, input_tensor)
    print(f"Rank {rank} result: {output_tensor.cpu()}")

    dist.destroy_process_group()


def all_gather_worker_for_nranks():
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    print(f"Hello from rank {local_rank}/{world_size}.")

    device = torch.device(f"tpu:{local_rank}")
    torch_tpu.tpu.set_device(local_rank)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    group = dist.group.WORLD
    comm = PyScclCommunicator(group=group, device=local_rank)

    tensor_len = 8
    dtype = torch.float16
    input_tensor = torch.ones(tensor_len, dtype=dtype)
    if torch_tpu.tpu.current_device() == device.index:
        input_tensor = input_tensor.to(device)
    print(f"Rank {rank}: created input_tensor", input_tensor.shape, input_tensor.dtype, input_tensor.device, "numel", input_tensor.numel())
    
    output_tensor = torch.zeros(tensor_len * world_size, dtype=dtype)
    if torch_tpu.tpu.current_device() == device.index:
        output_tensor = output_tensor.to(device)
    print(f"Rank {rank}: created output_tensor", output_tensor.shape, output_tensor.dtype, output_tensor.device, "numel", output_tensor.numel())

    comm.all_gather(output_tensor, input_tensor)
    print(f"Rank {rank} result: {output_tensor.cpu()}")

    dist.destroy_process_group()

def all_reduce_worker_for_nranks():
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    print(f"Hello from rank {local_rank}/{world_size}.")

    device = torch.device(f"tpu:{local_rank}")
    torch_tpu.tpu.set_device(local_rank)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    group = dist.group.WORLD
    comm = PyScclCommunicator(group=group, device=local_rank)

    tensor_len = 16
    dtype = torch.float16
    tensor = torch.ones(tensor_len, dtype=dtype, device=device)
    print("created tensor", tensor.shape, tensor.dtype, tensor.device, "numel", tensor.numel())

    tensor = comm.all_reduce(tensor)
    assert torch.all(tensor == comm.world_size).cpu().item()

    print(f"Rank {rank} result: {tensor.cpu()}")
    dist.destroy_process_group()


def send_recv_worker_for_nranks():
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    print(f"Hello from rank {local_rank}/{world_size}.")

    device = torch.device(f"tpu:{local_rank}")
    torch_tpu.tpu.set_device(local_rank)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    group = dist.group.WORLD
    comm = PyScclCommunicator(group=group, device=local_rank)

    tensor_len = 16
    dtype = torch.float16
    if rank == 0:
        tensor = torch.rand(tensor_len, dtype=dtype)
        if torch_tpu.tpu.current_device() == device.index:
            tensor = tensor.to(device)
        print("created input_tensor", tensor.shape, tensor.dtype, tensor.device, "numel", tensor.numel())
        comm.send(tensor, dst=1)
        print('Rank 0 has sent the tensor to Rank 1:', tensor.cpu())
    elif rank == 1:
        tensor = torch.zeros(tensor_len, dtype=dtype)
        if torch_tpu.tpu.current_device() == device.index:
            tensor = tensor.to(device)
        print("created output_tensor", tensor.shape, tensor.dtype, tensor.device, "numel", tensor.numel())
        comm.recv(tensor, src=0)
        print('Rank 1 has received the tensor:', tensor.cpu())

    dist.destroy_process_group()

if __name__ == "__main__":
    import os
    # all_gather_worker_for_single_rank()

    # all_gather_worker_for_nranks()

    # all_reduce_worker_for_nranks()

    send_recv_worker_for_nranks()
