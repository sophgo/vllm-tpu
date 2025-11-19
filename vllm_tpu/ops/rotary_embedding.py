from typing import Optional, Tuple

import os
import torch
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding, MRotaryEmbedding)
from vllm.model_executor.layers.rotary_embedding.common import (
    yarn_find_correction_range, yarn_linear_ramp_mask)

#add device for torch.arange
def rope_compute_inv_freq(self, base: float) -> torch.Tensor:
    """Compute the inverse frequency."""
    # NOTE(woosuk): To exactly match the HF implementation, we need to
    # use CPU to compute the cache and then move it to GPU. However, we
    # create the cache on GPU for faster initialization. This may cause
    # a slight numerical difference between the HF implementation and ours.
    device = "tpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    inv_freq = 1.0 / (base**(torch.arange(
        #0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        0, self.rotary_dim, 2, dtype=torch.float, device=device) / self.rotary_dim))
    return inv_freq

#add device for torch.arange
def rope_compute_cos_sin_cache(self) -> torch.Tensor:
    """Compute the cos and sin cache."""
    inv_freq = self._compute_inv_freq(self.base)
    #t = torch.arange(self.max_position_embeddings, dtype=torch.float)
    device = "tpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    t = torch.arange(self.max_position_embeddings, dtype=torch.float, device=device)

    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache

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
    cos_sin = self.cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    return cos, sin


#hack from vllm:yarn_linear_ramp_mask
def rope_yarn_linear_ramp_mask(low: float, high: float, dim: int,
                          dtype: torch.dtype) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity

    device = "tpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    linear_func = (torch.arange(dim, dtype=dtype, device=device) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def rope_deepseek_compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
    device = "tpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    pos_freqs = self.base**(torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=device) / self.rotary_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

    low, high = yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                            self.rotary_dim, self.base,
                                            self.max_position_embeddings)
    # Get n-d rotational scaling corrected for extrapolation
    inv_freq_mask = (1 - rope_yarn_linear_ramp_mask(
        low, high, self.rotary_dim // 2,
        dtype=torch.float)) * self.extrapolation_factor
    inv_freq = inv_freq_interpolation * (
        1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
    return inv_freq

def rope_deepseek_compute_cos_sin_cache(self) -> torch.Tensor:
    inv_freq = self._compute_inv_freq(self.scaling_factor)
    device = "tpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    t = torch.arange(self.max_position_embeddings * self.scaling_factor, dtype=torch.float32, device=device)
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

def rope_qwen2_5_vl_forward_oot(
    self,
    positions: torch.Tensor,
    query: Optional[torch.Tensor] = None,  # 改为可选参数
    key: Optional[torch.Tensor] = None,    # 改为可选参数
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert positions.ndim == 1 or positions.ndim == 2

    num_tokens = positions.shape[-1]
    cos_sin = self.cos_sin_cache[positions.contiguous()]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        assert self.mrope_section

        cos = torch.cat([
            m[i]
            for i, m in enumerate(cos.split(self.mrope_section, dim=-1))
        ],
                        dim=-1)
        sin = torch.cat([
            m[i]
            for i, m in enumerate(sin.split(self.mrope_section, dim=-1))
        ],
                        dim=-1)

    return cos, sin

RotaryEmbedding._compute_inv_freq = rope_compute_inv_freq
RotaryEmbedding._compute_cos_sin_cache = rope_compute_cos_sin_cache
RotaryEmbedding.forward_oot = rope_forward_oot
DeepseekScalingRotaryEmbedding._compute_inv_freq = rope_deepseek_compute_inv_freq
DeepseekScalingRotaryEmbedding._compute_cos_sin_cache = rope_deepseek_compute_cos_sin_cache
DeepseekScalingRotaryEmbedding.forward = rope_deepseek_forward_oot
MRotaryEmbedding.forward = rope_qwen2_5_vl_forward_oot
