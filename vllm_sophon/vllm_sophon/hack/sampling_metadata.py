# SPDX-License-Identifier: Apache-2.0

from array import array
from typing import Dict, List, Optional

import torch

from vllm.sequence import SequenceGroupMetadata
from vllm.utils import is_pin_memory_available, make_tensor_with_pad

from vllm.model_executor.sampling_metadata import SamplingMetadataCache, SamplingMetadata, SamplingTensors
from vllm.model_executor.sampling_metadata import _prepare_seq_groups

@staticmethod
def SamplingMetadata_prepare(
    seq_group_metadata_list: List[SequenceGroupMetadata],
    seq_lens: List[int],
    query_lens: List[int],
    device: str,
    pin_memory: bool,
    generators: Optional[Dict[str, torch.Generator]] = None,
    cache: Optional[SamplingMetadataCache] = None,
) -> "SamplingMetadata":
    (
        seq_groups,
        selected_token_indices,
        categorized_sample_indices,
        num_prompts,
    ) = _prepare_seq_groups(seq_group_metadata_list, seq_lens, query_lens,
                            device, generators, cache)
    selected_token_indices = torch.tensor(
        selected_token_indices,
        dtype=torch.long,
        pin_memory=pin_memory, 
        device=device
        )
    categorized_sample_indices = {
        t:
        torch.tensor(
            seq_ids, 
            dtype=torch.int, 
            pin_memory=pin_memory, 
            device=device
            )
        for t, seq_ids in categorized_sample_indices.items()
    }

    sampling_metadata = SamplingMetadata(
        seq_groups=seq_groups,
        selected_token_indices=selected_token_indices,
        categorized_sample_indices=categorized_sample_indices,
        num_prompts=num_prompts,
    )
    return sampling_metadata


@classmethod
def SamplingTensors_from_lists(
    cls,
    temperatures: List[float],
    top_ps: List[float],
    top_ks: List[int],
    min_ps: List[float],
    presence_penalties: List[float],
    frequency_penalties: List[float],
    repetition_penalties: List[float],
    prompt_tokens: List[array],
    output_tokens: List[array],
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> "SamplingTensors":
    # Note that the performance will be very bad without
    # pinned memory.
    pin_memory = is_pin_memory_available()

    do_penalties = prompt_tokens or output_tokens

    if do_penalties:
        prompt_t = make_tensor_with_pad(
            prompt_tokens,
            vocab_size,
            device=device,
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        output_t = make_tensor_with_pad(
            output_tokens,
            vocab_size,
            device=device,
            dtype=torch.int,
            pin_memory=pin_memory,
        )
    else:
        empty_tensor = torch.empty(0, device=device, dtype=torch.int32)
        prompt_t = empty_tensor
        output_t = empty_tensor

    temperatures_t = torch.tensor(
        temperatures,
        device=device,
        dtype=dtype,
        pin_memory=pin_memory,
    )
    top_ps_t = torch.tensor(
        top_ps,
        device=device,
        dtype=dtype,
        pin_memory=pin_memory,
    )
    min_ps_t = torch.tensor(
        min_ps,
        device=device,
        dtype=dtype,
        pin_memory=pin_memory,
    )
    presence_penalties_t = torch.tensor(
        presence_penalties,
        device=device,
        dtype=dtype,
        pin_memory=pin_memory,
    )
    frequency_penalties_t = torch.tensor(
        frequency_penalties,
        device=device,
        dtype=dtype,
        pin_memory=pin_memory,
    )
    repetition_penalties_t = torch.tensor(
        repetition_penalties,
        device=device,
        dtype=dtype,
        pin_memory=pin_memory,
    )
    top_ks_t = torch.tensor(
        top_ks,
        device=device,
        dtype=torch.int32,
        pin_memory=pin_memory,
    )
    # Because the memory is pinned, we can do non-blocking
    # transfer to device.

    return cls(
        temperatures=temperatures_t,
        top_ps=top_ps_t,
        top_ks=top_ks_t,
        min_ps=min_ps_t,
        presence_penalties=presence_penalties_t,
        frequency_penalties=frequency_penalties_t,
        repetition_penalties=repetition_penalties_t,
        prompt_tokens=prompt_t,
        output_tokens=output_t,
    )


SamplingMetadata.prepare = SamplingMetadata_prepare
SamplingTensors.from_lists = SamplingTensors_from_lists
