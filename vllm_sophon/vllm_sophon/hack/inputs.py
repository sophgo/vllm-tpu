import torch
from collections.abc import Mapping
from typing_extensions import TypeAlias
from typing import (Union, cast)
from vllm.utils.jsontree import json_map_leaves
from vllm.multimodal.inputs import BatchedTensorInputs, MultiModalKwargs

@staticmethod
def MultiModalKwargs_as_kwargs(
    batched_inputs: BatchedTensorInputs,
    *,
    device: torch.types.Device,
) -> BatchedTensorInputs:
    return json_map_leaves(
        lambda x: x.to(device, non_blocking=False),
        batched_inputs,
    )

MultiModalKwargs.as_kwargs = MultiModalKwargs_as_kwargs
