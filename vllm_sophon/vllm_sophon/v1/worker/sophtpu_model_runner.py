import enum
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from unittest.mock import patch
import time

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch_tpu

from vllm.attention import AttentionMetadata
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.sampling_params import SamplingType
from vllm.utils import LayerBlockType, cdiv, is_pin_memory_available, get_dtype_size
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_sophon.attention.attention import (SophTPUAttentionBackend,
                                             SophTPUMetadata)

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)

# Here we utilize the behavior that out-of-bound index is ignored.
# FIXME(woosuk): Find a more reliable way to prevent possible bugs.
_PAD_SLOT_ID = 1_000_000_000


class ExecutionMode(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()
    PREFIX_PREFILL = enum.auto()

    def is_prefill(self) -> bool:
        return self in (ExecutionMode.PREFILL, ExecutionMode.PREFIX_PREFILL)


@dataclass
class PromptDecodeInfo:
    prompt_req_ids: List[str]
    decode_req_ids: List[str]
    prompt_scheduled_tokens: List[int]


@dataclass
class PromptData:
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    attn_metadata: SophTPUMetadata


@dataclass
class DecodeData:
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    attn_metadata: Optional[SophTPUMetadata] = None


#vllm's FullAttentionSpec has bug, and fixed in latest version
#just implement here right now, can use vllm's AttentionSpec after
@dataclass
class SophAttentionSpec(FullAttentionSpec):
    use_mla: bool

    @property
    def page_size_bytes(self) -> int:
        coef = 1 if self.use_mla else 2
        return  coef * self.block_size * self.num_kv_heads * self.head_size \
                * get_dtype_size(self.dtype)


class SophTPUModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype

        self.is_multimodal_model = model_config.is_multimodal_model
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()

        self.model: Optional[nn.Module] = None

        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
        )

        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}

        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: Dict[str, Dict[int, torch.Tensor]] = {}

        # KV caches for forward pass
        self.kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Cached torch/numpy tensors
        self.num_swaps = 2
        self.cur_swap_id = 0
        self.input_ids_cpu = []
        self.input_ids_np = []
        self.input_positions_cpu = []
        self.input_positions_np = []
        self.slot_mapping_cpu = []
        self.slot_mapping_np = []
        self.prompt_context_lens_cpu = []
        self.prompt_effective_query_lens_cpu = []
        self.decode_context_lens_cpu = []
        self.decode_context_lens_np = []
        for _ in range(self.num_swaps):
            self.input_ids_cpu.append(
                torch.empty(self.max_num_tokens,
                            dtype=torch.int32,
                            device="cpu"))
            self.input_ids_np.append(self.input_ids_cpu[-1].numpy())

            self.input_positions_cpu.append(
                torch.empty(self.max_num_tokens,
                            dtype=torch.int32,
                            device="cpu"))
            self.input_positions_np.append(
                self.input_positions_cpu[-1].numpy())

            self.slot_mapping_cpu.append(
                torch.empty(self.max_num_tokens,
                            dtype=torch.int64,
                            device="cpu"))
            self.slot_mapping_np.append(self.slot_mapping_cpu[-1].numpy())

            self.prompt_context_lens_cpu.append(
                torch.empty((1), dtype=torch.int32, device="cpu"))
            self.prompt_effective_query_lens_cpu.append(
                torch.empty((1), dtype=torch.int32, device="cpu"))

            self.decode_context_lens_cpu.append(
                torch.empty(self.max_num_tokens,
                            dtype=torch.int32,
                            device="cpu"))
            self.decode_context_lens_np.append(
                self.decode_context_lens_cpu[-1].numpy())

        # Range tensor with values [0 .. self.max_num_tokens - 1].
        # Used to initialize positions / context_lens / seq_lens
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int32)

        self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=self.pin_memory)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()

        self.time_records = {
            "inference": [],
            "data_transfer": []
        }
        self.num_tokens = 0
    def _update_states(self, scheduler_output: "SchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        Returns:
            True if there is a new/resumed/paused/finished request in the batch.
            If False, we can skip copying SamplingMetadata to the GPU.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: List[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt=new_req_data.prompt,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                req_state.block_ids.extend(req_data.new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                req_data.num_computed_tokens)
            start_index = len(req_state.block_ids) - len(
                req_data.new_block_ids)
            self.input_batch.block_table.append_row(req_index, start_index,
                                                    req_data.new_block_ids)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)
        return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0

    def swap_step(self):
        self.cur_swap_id = (self.cur_swap_id + 1) % self.num_swaps

    def get_model(self) -> nn.Module:
        assert self.model is not None
        return self.model

    def get_kv_cache_spec(self) -> KVCacheSpec:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: KVCacheSpec = {}
        for layer_name, attn_module in forward_ctx.items():
            # TODO: Support other attention modules, e.g., sliding window,
            # cross-attention, MLA.
            assert isinstance(attn_module, Attention)
            if attn_module.attn_type == AttentionType.DECODER:
                kv_cache_spec[layer_name] = SophAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=attn_module.dtype,
                    use_mla=self.vllm_config.model_config.use_mla,
                )
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec

    def _get_prompts_and_decodes(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> PromptDecodeInfo:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        prompt_req_ids = []
        decode_req_ids = []
        prompt_scheduled_tokens = []

        for i in range(num_reqs):
            req_id = self.input_batch.req_ids[i]
            assert req_id is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[i]
            num_prompt_tokens = self.input_batch.num_prompt_tokens[i]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]

            if num_computed_tokens < num_prompt_tokens:
                # Prompt
                prompt_req_ids.append(req_id)
                prompt_scheduled_tokens.append(num_scheduled_tokens)
            else:
                # Decode
                assert num_scheduled_tokens == 1, (
                    f"Decode request {req_id} scheduled {num_scheduled_tokens} tokens")
                decode_req_ids.append(req_id)

        if not (prompt_req_ids or decode_req_ids):
            raise RuntimeError("No valid prompt or decode requests identified")

        return PromptDecodeInfo(prompt_req_ids, decode_req_ids,
                                prompt_scheduled_tokens)

    def _prepare_prompt(
        self,
        pd_info,
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, List[int]]:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        prompt_lens: List[int] = []
        cache_lengths: List[int] = []

        req_ids = pd_info.prompt_req_ids
        block_tables = []
        max_blocks = 0

        for i, req_id in enumerate(req_ids):
            req_state = self.requests[req_id]
            prompt_token_ids = req_state.prompt_token_ids
            block_ids = req_state.block_ids
            num_computed = req_state.num_computed_tokens
            start_idx = max(0, len(prompt_token_ids) - self.sliding_window) if self.sliding_window else 0
            process_tokens = prompt_token_ids[num_computed:num_computed+pd_info.prompt_scheduled_tokens[i]]
            # input_tokens\input_positions\prompt_lens\context_lens
            input_tokens.extend(process_tokens)
            input_positions.extend(range(num_computed, num_computed + len(process_tokens)))
            prompt_lens.append(len(process_tokens))
            cache_lengths.append(num_computed)

            # slot_mapping
            for i in range(len(process_tokens)):
                global_pos = num_computed + i
                if global_pos < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue
                block_idx = (global_pos - start_idx) // self.block_size
                block_offset = (global_pos - start_idx) % self.block_size
                slot = block_ids[block_idx] * self.block_size + block_offset
                slot_mapping.append(slot)

            # block_tables
            req_blocks = [bid for bid in block_ids
                        if bid >= (start_idx // self.block_size)]
            block_tables.append(req_blocks)
            max_blocks = max(max_blocks, len(req_blocks))

        # to tensor
        input_tokens = torch.tensor(input_tokens, dtype=torch.int32, device=self.device)
        input_positions = torch.tensor(input_positions, dtype=torch.int32, device=self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        cache_lengths = torch.tensor(cache_lengths, dtype=torch.int32, device="cpu")
        prompt_lens_tensor = torch.tensor(prompt_lens, dtype=torch.int32, device="cpu")
        padded_block_tables = [seq + [0] * (max_blocks - len(seq)) for seq in block_tables]
        block_tables_tensor = torch.tensor(padded_block_tables, dtype=torch.int32, device=self.device)

        attn_metadata = SophTPUMetadata(
            num_decode_tokens = 0,
            slot_mapping = slot_mapping,
            num_prefills=len(req_ids),
            num_prefill_tokens=len(input_tokens),
            enable_kv_scales_calculation=True,
            block_tables=block_tables_tensor,
            cache_lengths=cache_lengths,
            input_lengths=prompt_lens_tensor,
            multi_modal_placeholder_index_maps=None
        )

        return PromptData(input_tokens, input_positions, attn_metadata)


    def _prepare_decode(
        self,
        req_ids,
    ) -> DecodeData:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        cache_lengths: List[int] = []
        input_lengths: List[int] = []

        block_tables = []
        max_blocks = 0
        for req_id in req_ids:
            req_state = self.requests[req_id]
            output_token_ids = req_state.output_token_ids
            generation_token = output_token_ids[-1]
            seq_len = req_state.num_tokens
            num_computed = req_state.num_computed_tokens
            position = seq_len - 1
            seq_len = seq_len if self.sliding_window is None else min(
                seq_len, self.sliding_window)
            block_ids = req_state.block_ids
            start_idx = max(0, seq_len - self.sliding_window) if self.sliding_window else 0
            
            # input_tokens\input_positions\cache_lengths
            input_tokens.append(generation_token)
            input_positions.append(position)
            cache_lengths.append(num_computed)
            input_lengths.append(int(1))

            # slot_mapping
            global_pos = req_state.num_computed_tokens
            if global_pos < start_idx:
                slot_mapping.append(_PAD_SLOT_ID)
                continue
            block_idx = (global_pos - start_idx) // self.block_size
            block_offset = (global_pos - start_idx) % self.block_size
            slot = block_ids[block_idx] * self.block_size + block_offset
            slot_mapping.append(slot)

            # block_tables
            req_blocks = [bid for bid in block_ids
                        if bid >= (start_idx // self.block_size)]
            block_tables.append(req_blocks)
            max_blocks = max(max_blocks, len(req_blocks))

        # to tensor
        input_tokens = torch.tensor(input_tokens, dtype=torch.int32, device=self.device)
        input_positions = torch.tensor(input_positions, dtype=torch.int32, device=self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        cache_lengths = torch.tensor(cache_lengths, dtype=torch.int32, device="cpu")
        input_lengths = torch.tensor(input_lengths, dtype=torch.int32, device="cpu")
        padded_block_tables = [seq + [0] * (max_blocks - len(seq)) for seq in block_tables]
        block_tables_tensor = torch.tensor(padded_block_tables, dtype=torch.int32, device=self.device)

        attn_metadata = SophTPUMetadata(
            num_prefills=None,
            num_prefill_tokens=0,
            num_decode_tokens=len(req_ids),
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            block_tables=block_tables_tensor,
            cache_lengths=cache_lengths,
            input_lengths=input_lengths
        )

        return DecodeData(input_tokens=input_tokens,
                          input_positions=input_positions,
                          attn_metadata=attn_metadata)

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        # Update cached state
        self._update_states(scheduler_output)
        pd_info = self._get_prompts_and_decodes(scheduler_output)

        req_ids = pd_info.prompt_req_ids + pd_info.decode_req_ids

        num_reqs = self.input_batch.num_reqs
        assert(num_reqs == len(req_ids))
        num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
        max_num_scheduled_tokens = 0
        for i, req_id in enumerate(req_ids):
            req_index = self.input_batch.req_id_to_index[req_id]
            if req_id in pd_info.prompt_req_ids:
                num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            else:
                assert req_id in pd_info.decode_req_ids
                num_tokens = 1
            num_scheduled_tokens[i] = num_tokens
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)

        cu_num_tokens = np.cumsum(num_scheduled_tokens)

        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens
        query_start_loc = self.query_start_loc_cpu[:num_reqs + 1].to(
            self.device, non_blocking=True)
        logits_indices = query_start_loc[1:] - 1
        prompt_len = len(pd_info.prompt_req_ids)
        logits_indices = logits_indices[:prompt_len]

        num_prompts = len(pd_info.prompt_req_ids)
        sampled_token_ids = [0] * self.input_batch.num_reqs

        prompt_batch = self._prepare_prompt(pd_info) if pd_info.prompt_req_ids else None
        decode_batch = self._prepare_decode(pd_info.decode_req_ids) if pd_info.decode_req_ids else None

        hidden_or_intermediate_states = None
        if prompt_batch and decode_batch:
            # mixed batch sequential execution (prompt first, then decode)
            decode_hidden = self._run_model(decode_batch, intermediate_tensors)
            prompt_hidden = self._run_model(prompt_batch, intermediate_tensors)
            sample_prompt_hidden = prompt_hidden[logits_indices]
            hidden_or_intermediate_states = torch.cat([sample_prompt_hidden, decode_hidden], dim=0)
        elif prompt_batch:
            hidden_or_intermediate_states = self._run_model(prompt_batch, intermediate_tensors)
            hidden_or_intermediate_states = hidden_or_intermediate_states[logits_indices]
        elif decode_batch:
            hidden_or_intermediate_states = self._run_model(decode_batch, intermediate_tensors)
        else:
            raise RuntimeError(
                "At least one of `prompt_batch` or `decode_batch` must be non-empty. "
                "Received empty inputs.")

        logits = self.model.compute_logits(hidden_or_intermediate_states, None)
        # Greedy sampling.
        argmax_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        argmax_token_ids = argmax_token_ids.squeeze(dim=-1)
        # Transfer sampled tokens from TPU to CPU.
        generate_token_ids_cpu = argmax_token_ids.cpu()
        # self._statistics_time(t1, t2, t3, t4)
        generate_token_ids_list = generate_token_ids_cpu.tolist()

        # Update cached state
        for i, req_id in enumerate(req_ids):
            req_index = self.input_batch.req_id_to_index[req_id]
            req_state = self.requests[req_id]
            seq_len = req_state.num_computed_tokens + 1
            is_prefill = req_id in pd_info.prompt_req_ids

            # batch with prefill-chunking
            if is_prefill and req_state.num_computed_tokens < len(req_state.prompt_token_ids):
                token_id = generate_token_ids_list[i]
                # Determine whether prefill-chunking is complete
                num_prefill_tokens_remained = len(req_state.prompt_token_ids)-req_state.num_computed_tokens
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                if num_prefill_tokens_remained <= num_scheduled_tokens:
                    # end of prefill-chunking, generate the first token
                    req_state.output_token_ids.append(token_id)
            elif num_prompts > 0:
                # decode in mixed batch
                token_id = generate_token_ids_list[i]
                req_state.output_token_ids.append(token_id)
            else:
                # decode
                token_id = generate_token_ids_list[i]
                req_state.output_token_ids.append(token_id)

            sampled_token_ids[req_index] = token_id
            self.input_batch.token_ids_cpu[req_index, seq_len] = token_id
            self.input_batch.num_tokens[req_index] += 1

        # Create output.
        prompt_logprobs_dict: Dict[str, Optional[LogprobsTensors]] = {}
        for req_id in req_ids:
            prompt_logprobs_dict[req_id] = None

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[[tid] for tid in sampled_token_ids],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={req_id: None for req_id in req_ids},
        )

    def _run_model(self, model_input, intermediate_tensors):
        with set_forward_context(model_input.attn_metadata, self.vllm_config):
            return self.model(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                kv_caches=self.kv_caches,
                attn_metadata=model_input.attn_metadata,
                intermediate_tensors=intermediate_tensors,
            )

    def load_model(self) -> None:
        self.model = get_model(vllm_config=self.vllm_config)

    def dummy_run(
        self,
        kv_caches,
        num_tokens: int,
        seq_len: Optional[int] = None,
        exec_mode: Optional[ExecutionMode] = None,
    ) -> None:
        assert seq_len is not None
        assert exec_mode is not None

        exec_mode = ExecutionMode(exec_mode)
        if exec_mode.is_prefill():
            seq_len = (seq_len + 15) // 16 * 16
            token_ids = torch.zeros((num_tokens, seq_len),
                                    dtype=torch.int32,
                                    device=self.device)
            position_ids = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int32,
                                       device=self.device)
            slot_mapping = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int64,
                                       device=self.device)
            if exec_mode == ExecutionMode.PREFILL:
                attn_metadata = SophTPUMetadata(
                    num_prefills=num_tokens,
                    num_prefill_tokens=num_tokens * seq_len,
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping,
                    multi_modal_placeholder_index_maps=None,
                    enable_kv_scales_calculation=True,
                    block_tables=None,
                    context_lens=None,
                    effective_query_lens=None,
                )

            else:
                context_lens = torch.ones((num_tokens, ),
                                          dtype=torch.int32,
                                          device=self.device)

                block_tables = torch.zeros(
                    (num_tokens, self.max_num_blocks_per_req),
                    dtype=torch.int32,
                    device=self.device)

                effective_query_lens = torch.ones_like(context_lens)

                attn_metadata = SophTPUMetadata(
                    num_prefills=num_tokens,
                    num_prefill_tokens=num_tokens * seq_len,
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping,
                    multi_modal_placeholder_index_maps=None,
                    enable_kv_scales_calculation=True,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    effective_query_lens=effective_query_lens,
                )
        else:
            assert seq_len == 1
            token_ids = torch.zeros((num_tokens, seq_len),
                                    dtype=torch.int32,
                                    device=self.device)
            position_ids = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int32,
                                       device=self.device)
            slot_mapping = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int64,
                                       device=self.device)
            block_tables = torch.zeros(
                (num_tokens, self.max_num_blocks_per_req),
                dtype=torch.int32,
                device=self.device)
            context_lens = torch.ones((num_tokens, ),
                                      dtype=torch.int32,
                                      device=self.device)
            attn_metadata = SophTPUMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=num_tokens * seq_len,
                slot_mapping=slot_mapping,
                multi_modal_placeholder_index_maps=None,
                enable_kv_scales_calculation=True,
                block_tables=block_tables,
                context_lens=context_lens,
            )

        with set_forward_context(attn_metadata, self.vllm_config, 0):
            assert self.model is not None
            self.model(token_ids, position_ids, attn_metadata, kv_caches)


    def capture_model(self) -> None:
        """Compile the model."""

        # Prefill
        logger.info(
            "Compiling the model with different input shapes for prefill:")
        start = time.time()
        for batch_size in [1]:
            seq_len = 16
            while seq_len <= self.model_config.max_model_len:
                self.dummy_run(self.kv_caches,
                               batch_size,
                               seq_len,
                               exec_mode=ExecutionMode.PREFILL)
                torch_tpu.tpu.synchronize()
                logger.info("  batch_size: %d, seq_len: %d", batch_size,
                            seq_len)
                num_tokens = batch_size * seq_len
                if num_tokens >= self.scheduler_config.max_num_batched_tokens:
                    break
                seq_len = seq_len * 2

        end = time.time()
        logger.info("    -- Compilation for prefill done in %.2f [secs].",
                    end - start)

        # Prefix prefill
        if self.scheduler_config.enable_chunked_prefill:
            logger.info("Compiling the model with different input shapes for "
                        "prefix prefill:")
            start = time.time()
            for batch_size in [1]:
                seq_len = 16
                while seq_len <= self.model_config.max_model_len:
                    self.dummy_run(self.kv_caches,
                                   batch_size,
                                   seq_len,
                                   exec_mode=ExecutionMode.PREFIX_PREFILL)
                    torch_tpu.tpu.synchronize()
                    logger.info("  batch_size: %d, seq_len: %d", batch_size,
                                seq_len)
                    num_tokens = batch_size * seq_len
                    if (num_tokens
                            >= self.scheduler_config.max_num_batched_tokens):
                        break
                    seq_len = seq_len * 2
            end = time.time()
            logger.info(
                "    -- Compilation for prefix prefill done in %.2f [secs].",
                end - start)

        # Decode
        logger.info(
            "Compiling the model with different input shapes for decode:")
        start = time.time()
        seq_len = 1
        batch_size = 8  # Must be in sync with _get_padded_batch_size()
        while True:
            self.dummy_run(self.kv_caches,
                           batch_size,
                           seq_len,
                           exec_mode=ExecutionMode.DECODE)
            torch_tpu.tpu.synchronize()
            logger.info("  batch_size: %d, seq_len: %d", batch_size, seq_len)

            if batch_size >= self.scheduler_config.max_num_seqs:
                break
            batch_size = batch_size + 16 if batch_size >= 16 else batch_size * 2

        end = time.time()
        logger.info("    -- Compilation for decode done in %.2f [secs].",
                    end - start)


    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if len(kv_cache_config.groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        kv_caches: Dict[str, torch.Tensor] = {}

        for layer_name, layer_spec in kv_cache_config.kv_cache_spec.items():
            tensor_config = kv_cache_config.tensors[layer_name]
            assert tensor_config.size % layer_spec.page_size_bytes == 0
            num_blocks = tensor_config.size // layer_spec.page_size_bytes
            if isinstance(layer_spec, SophAttentionSpec):
                kv_cache_shape = SophTPUAttentionBackend.get_kv_cache_shape(
                    num_blocks, layer_spec.block_size, layer_spec.num_kv_heads,
                    layer_spec.head_size)
                dtype = layer_spec.dtype

                if self.model_config.is_deepseek_mla:
                    assert self.model_config.use_mla
                    qk_rope_head_dim = getattr(self.model_config.hf_text_config, "qk_rope_head_dim", 0)
                    kv_lora_rank = self.model_config.hf_text_config.kv_lora_rank
                    tpu_kv_cache = torch.empty((kv_cache_shape[0], kv_cache_shape[1], kv_lora_rank),
                                               dtype=dtype,
                                               device=self.device)
                    tpu_pe_cache = torch.empty((kv_cache_shape[0], kv_cache_shape[1], qk_rope_head_dim),
                                               dtype=dtype,
                                               device=self.device)

                    kv_caches[layer_name] = (tpu_kv_cache, tpu_pe_cache)
                else:
                    tpu_k_cache = torch.empty(kv_cache_shape,
                                              dtype=dtype,
                                              device=self.device)
                    tpu_v_cache = torch.empty_like(tpu_k_cache)

                    kv_caches[layer_name] = (tpu_k_cache, tpu_v_cache)
            else:
                raise NotImplementedError

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

    def _statistics_time(self, t1, t2, t3, t4):
        import os
        DECODE_TOKEN_LEN = int(os.getenv("DECODE_TOKEN_LEN", "128"))
        self.num_tokens += 1
        if self.num_tokens > DECODE_TOKEN_LEN:
            self.time_records["inference"].append((t2 - t1)/1e6)
            self.time_records["data_transfer"].append((t4 - t3)/1e6)
            if self.num_tokens == int(DECODE_TOKEN_LEN * 2):
                results = {}
                for key, values in self.time_records.items():
                    if not values:
                        print(f"警告: {key}列表为空，无法统计时间")
                        continue
                    arr = np.array(values)
                    stats = {
                        "count": len(arr),
                        "mean": np.mean(arr),
                        "median": np.median(arr),
                        "min": np.min(arr),
                        "max": np.max(arr),
                        "std": np.std(arr)
                    }
                    results[key] = stats
                print("\n===== 时间性能分析报告 =====")
                for key, stats in results.items():
                    print(f"\n【{key.upper()} 统计】")
                    print(f"  推理token数: {stats['count']}")
                    print(f"  平均时间: {stats['mean']:.2f} ms")
                    print(f"  中位数: {stats['median']:.2f} ms")
                    print(f"  最短时间: {stats['min']:.2f} ms")
                    print(f"  最长时间: {stats['max']:.2f} ms")
                    print(f"  标准差: {stats['std']:.2f} ms")

class ModelWrapperV1(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Executes the forward pass of the model and samples the next token.

        Args:
            token_ids: The input token IDs of shape [batch_size, seq_len].
            position_ids: The input position IDs of shape [batch_size, seq_len].
            attn_metadata: The attention metadata.
            input_lens: The actual input lengths of shape [batch_size].
            t: The sampling temperature of shape [batch_size].
            p: The top-p probability of shape [batch_size].
            num_samples: Number of samples to draw from each logits vector.
            kv_caches: The key and value caches. They can be None during the
                memory profiling at initialization.
        """
        # Skip this in memory profiling at initialization.
        if attn_metadata is not None and kv_caches[0][0].numel() > 0:
            # index_copy_(slot_mapping) only works when the inserted dimension
            # is 0. However, the KV cache in the SophTPU backend has the shape
            # [num_kv_heads, num_blocks, block_size, head_size]. To make it
            # work, we need to flatten the first three dimensions and modify
            # the slot_mapping accordingly.
            num_kv_heads, num_blocks, block_size, _ = kv_caches[0][0].shape
            slot_mapping = attn_metadata.slot_mapping
            slot_mapping = slot_mapping.flatten()
            head_indicies = torch.arange(0,
                                         num_kv_heads,
                                         device=slot_mapping.device,
                                         dtype=slot_mapping.dtype)
            head_indicies *= block_size * num_blocks
            slot_mapping = slot_mapping.repeat_interleave(num_kv_heads).view(
                -1, num_kv_heads)
            slot_mapping = slot_mapping + head_indicies.view(1, -1)
            slot_mapping = slot_mapping.flatten()
            attn_metadata.slot_mapping = slot_mapping

        assert self.model is not None
        hidden_states = self.model(
            token_ids,
            position_ids,
            kv_caches,
            attn_metadata,
        )

        hidden_states = hidden_states.flatten(0, 1)
        logits = self.model.compute_logits(hidden_states, None)

        # Greedy sampling.
        argmax_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        argmax_token_ids = argmax_token_ids.squeeze(dim=-1)
        return argmax_token_ids

def swap_positions(b: InputBatch, id_1, id_2):
    assert id_1 != id_2
    req_id_1 = b.req_ids[id_1]
    req_id_2 = b.req_ids[id_2]
    assert req_id_1 is not None
    assert req_id_2 is not None
    assert id_1 == b.req_id_to_index[req_id_1]
    assert id_2 == b.req_id_to_index[req_id_2]

    b.req_ids[id_1], b.req_ids[id_2] = b.req_ids[id_2], b.req_ids[id_1]
    b.req_id_to_index[req_id_1], b.req_id_to_index[
        req_id_2] = b.req_id_to_index[req_id_2], b.req_id_to_index[req_id_1]

    ids = [id_1, id_2]
    rev_ids = [id_2, id_1]
    b.num_tokens[ids] = b.num_tokens[rev_ids]
    b.token_ids_cpu[ids] = b.token_ids_cpu[rev_ids]
    b.num_prompt_tokens[ids] = b.num_prompt_tokens[rev_ids]
    b.num_computed_tokens_cpu[ids] = b.num_computed_tokens_cpu[rev_ids]

    b.block_table.swap_row(id_1, id_2)

    b.temperature_cpu[ids] = b.temperature_cpu[rev_ids]
    b.top_p_cpu[ids] = b.top_p_cpu[rev_ids]
    b.top_k_cpu[ids] = b.top_k_cpu[rev_ids]
    b.frequency_penalties_cpu[ids] = b.frequency_penalties_cpu[rev_ids]
    b.presence_penalties_cpu[ids] = b.presence_penalties_cpu[rev_ids]
    b.repetition_penalties_cpu[ids] = b.repetition_penalties_cpu[rev_ids]

    b.min_tokens[id_1], b.min_tokens[id_2] = b.min_tokens[id_2], b.min_tokens[
        id_1]

    gen_1 = b.generators.pop(id_1, None)
    gen_2 = b.generators.pop(id_2, None)
    if gen_1 is not None:
        b.generators[id_2] = gen_1
    if gen_2 is not None:
        b.generators[id_1] = gen_2


def ensure_decodes_first(b: InputBatch):
    num_reqs = b.num_reqs
    while True:
        # Find the first prompt index
        first_prompt_index = None
        for i in range(num_reqs):
            if b.num_computed_tokens_cpu[i] < b.num_prompt_tokens[i]:
                first_prompt_index = i
                break
        if first_prompt_index is None:
            break

        # Find the last decode index
        last_decode_index = None
        for i in reversed(range(num_reqs)):
            if b.num_computed_tokens_cpu[i] >= b.num_prompt_tokens[i]:
                last_decode_index = i
                break
        if last_decode_index is None:
            break

        # Sanity
        assert first_prompt_index != last_decode_index

        # Check if done
        if first_prompt_index > last_decode_index:
            break

        # Swap
        swap_positions(b, first_prompt_index, last_decode_index)


def _get_padded_prompt_len(x: int) -> int:
    # NOTE(woosuk): The pallas FlashAttention kernel requires the sequence
    # length to be a multiple of 16. We pad the prompt length to the nearest
    # multiple of 16. This is also good for performance.
    if x <= 16:
        return 16
    return 1 << (x - 1).bit_length()


def _get_padded_batch_size(batch_size: int) -> int:
    # The GMM Pallas kernel requires num_tokens * topk to be a multiple of 16.
    # To meet this requirement in the simplest way, we set the minimal batch
    # size to 8.
    if batch_size <= 8:
        return 8
    else:
        return ((batch_size + 15) // 16) * 16
