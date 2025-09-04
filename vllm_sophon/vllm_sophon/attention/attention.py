from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.logger import init_logger
logger = init_logger(__name__)

class SophTPUAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "SOPHTPU_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["SophTPUAttentionBackendImpl"]:
        return SophTPUAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["SophTPUMetadata"]:
        return SophTPUMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        # return (num_kv_heads, num_blocks, block_size, head_size)
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the Sophon TPU backend.")

    @staticmethod
    def copy_blocks(
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        src_to_dists: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        src_indices, dst_indices = src_to_dists
        for k_cache, v_cache in kv_caches:
            torch.ops.xla.dynamo_set_buffer_donor_(k_cache, True)
            k_cache[:, dst_indices] = k_cache[:, src_indices]
            torch.ops.xla.dynamo_set_buffer_donor_(v_cache, True)
            v_cache[:, dst_indices] = v_cache[:, src_indices]

@dataclass
class SophTPUMetadata(AttentionMetadata):

    # Currently, input sequences can only contain all prefills
    # or all decoding.
    block_tables: Optional[torch.Tensor] = None
    input_lengths: Optional[torch.Tensor] = None
    cache_lengths: Optional[torch.Tensor] = None

    # positional embeddings parameters
    cos: Optional[torch.Tensor] = None
    sin: Optional[torch.Tensor] = None
    # buffers for storing the intermediate results of the attention computation
    buffer: Optional[Dict[str, torch.Tensor]] = None

    @property
    def prefill_metadata(self) -> Optional["SophTPUMetadata"]:
        if self.num_prefills == 0:
            return None

        assert self.num_decode_tokens == 0
        return self

    @property
    def decode_metadata(self) -> Optional["SophTPUMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.block_tables is not None
        assert self.context_lens is not None
        return self

class SophTPUAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.logits_soft_cap = logits_soft_cap
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")
        if sliding_window is not None:
            logger.warning("The context lengths must be less than sliding window.\
                           This is temporarily method to support QwQ-32B.")
            # raise NotImplementedError("Sliding window is not supported.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError("FP8 KV cache dtype is not supported.")
        if blocksparse_params is not None:
            raise NotImplementedError("Blocksparse is not supported.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "SophTPUAttentionBackendImpl")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Sophon TPU attention.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache[0] = [num_blocks, block_size, num_kv_head, head_size]
            kv_cache[1] = [num_blocks, block_size, num_kv_head, head_size]
                NOTE: kv_cache[0] and kv_cache[1] will be an empty tensor
                with shape [0] for profiling run.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        context_lengths = attn_metadata.input_lengths + attn_metadata.cache_lengths
        attn_output = attn_metadata.buffer['attn_out'] if output is None else output
        torch.ops.my_ops.hybrid_attention(
            attn_output,
            torch.where(attn_metadata.input_lengths > 1)[0],
            query,
            key,
            value,
            kv_cache[0],
            kv_cache[1],
            attn_metadata.cos,
            attn_metadata.sin,
            attn_metadata.input_lengths,
            attn_metadata.cache_lengths,
            context_lengths,
            attn_metadata.slot_mapping,
            attn_metadata.block_tables,
            None,
            attn_metadata.block_tables.size(1),
            context_lengths.max().item(),
            kv_cache[0].size(1),
            self.scale
        )
        return attn_output
