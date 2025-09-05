from typing import Optional, Tuple

import torch
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding)
from vllm.model_executor.layers.rotary_embedding import (
    _yarn_find_correction_range, _yarn_linear_ramp_mask)

def rope_forward_oot(
    self,
    positions: torch.Tensor,
    #query: torch.Tensor,
    #key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A PyTorch-native implementation of forward()."""
    if offsets is not None:
        positions = positions + offsets
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = self.cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    return cos, sin


def rope_deepseek_compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
    pos_freqs = self.base**(torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

    low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                            self.rotary_dim, self.base,
                                            self.max_position_embeddings)
    # Get n-d rotational scaling corrected for extrapolation
    inv_freq_mask = (1 - _yarn_linear_ramp_mask(
        low, high, self.rotary_dim // 2,
        dtype=torch.float)) * self.extrapolation_factor
    inv_freq = inv_freq_interpolation * (
        1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
    return inv_freq

def rope_deepseek_compute_cos_sin_cache(self) -> torch.Tensor:
    inv_freq = self._compute_inv_freq(self.scaling_factor)
    t = torch.arange(self.max_position_embeddings * self.scaling_factor, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = (freqs.cos() * self.mscale)
    sin = (freqs.sin() * self.mscale)
    cache = torch.cat((cos, sin), dim=-1)
    return cache


def rope_deepseek_forward_oot(
    self,
    positions: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(
        positions.device)
    cos_sin = self.cos_sin_cache[torch.add(positions, offsets)
                                 if offsets is not None else positions]
    cos, sin = cos_sin.chunk(2, dim=-1)

    return cos, sin


RotaryEmbedding.forward_oot = rope_forward_oot
DeepseekScalingRotaryEmbedding._compute_inv_freq = rope_deepseek_compute_inv_freq
DeepseekScalingRotaryEmbedding._compute_cos_sin_cache = rope_deepseek_compute_cos_sin_cache
DeepseekScalingRotaryEmbedding.forward = rope_deepseek_forward_oot
