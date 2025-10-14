# hack for Sampler.forward in vllm/vllm/v1/sample/sampler.py for we not support long type now

from typing import Optional

import torch

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.sampler import Sampler

def sampler_forward(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> SamplerOutput:
    # NOTE(woosuk): Use the original logits (before any penalties or
    # temperature scaling) for the top-k logprobs.
    # This is different from the V0 sampler, which uses the logits that
    # is used for sampling (after penalties and temperature scaling).
    num_logprobs = sampling_metadata.max_num_logprobs
    if num_logprobs is not None:
        if self.logprobs_mode == "raw_logprobs":
            raw_logprobs = self.compute_logprobs(logits)
        elif self.logprobs_mode == "raw_logits":
            raw_logprobs = logits.clone()

    # Use float32 for the logits.
    logits = logits.to(torch.float32)
    # Apply allowed token ids.
    logits = self.apply_allowed_token_ids(logits, sampling_metadata)
    # Apply bad words exclusion.
    logits = self.apply_bad_words(logits, sampling_metadata)

    # Apply logits processors which can impact greedy sampling
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        logits = processor.apply(logits)

    # Apply penalties (e.g., min_tokens, freq_penalties).
    logits = self.apply_penalties(logits, sampling_metadata)

    # Sample the next token.
    sampled, processed_logprobs = self.sample(logits, sampling_metadata)
    if processed_logprobs is not None:
        raw_logprobs = processed_logprobs
    # Convert sampled token ids to int64 (long) type to ensure compatibility
    # with subsequent operations that may use these values as indices.
    # This conversion is necessary because FlashInfer sampling operations
    # return int32 (while PyTorch argmax and topk return int64).
    #sampled = sampled.long()
    sampled = sampled.to(torch.int32) #hack in here, sophon tpu not support ing64 now

    # Gather the logprobs of the topk and sampled token (if requested).
    # Get logprobs and rank tensors (if requested)
    logprobs_tensors = None if num_logprobs is None else \
        self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)

    # Use int32 to reduce the tensor size.
    sampled = sampled.to(torch.int32)

    # These are GPU tensors.
    sampler_output = SamplerOutput(
        # The sampled tokens are expanded to 2D tensor with shape
        # [num_requests, 1], where each row represents one generated
        # token per request.
        sampled_token_ids=sampled.unsqueeze(-1),
        logprobs_tensors=logprobs_tensors,
    )
    return sampler_output

Sampler.forward = sampler_forward
