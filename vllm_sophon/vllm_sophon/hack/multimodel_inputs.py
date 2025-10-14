import torch
from vllm.multimodel.inputs import BatchedTensorInputs
from typing import cast

from vllm.utils import JSONTree, json_map_leaves

@staticmethod
def MultiModelKwargs_as_kwargs(
    batched_inputs: BatchedTensorInputs,
    *,
    device: torch.types.Device,
) -> BatchedTensorInputs:
    json_inputs = cast(JSONTree[torch.Tensor], batched_inputs)

    json_mapped = json_map_leaves(
        lambda x: x.to(device, non_blocking=False),
        json_inputs,
    )

    return cast(BatchedTensorInputs, json_mapped)

MultiModelKwargs.as_kwargs = MultiModelKwargs_as_kwargs
