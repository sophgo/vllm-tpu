
import torch
from collections.abc import Mapping
from typing_extensions import TypeAlias
from typing import (Union, cast)
from vllm.utils import JSONTree, json_map_leaves
from vllm.multimodal.inputs import BatchedTensorInputs, MultiModalKwargs

@staticmethod
def MultiModalKwargs_as_kwargs(
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

MultiModalKwargs.as_kwargs = MultiModalKwargs_as_kwargs