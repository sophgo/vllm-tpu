# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch
import torch_tpu
import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import weak_ref_tensors

logger = init_logger(__name__)

@dataclasses.dataclass
class TPUGraphEntry:
    batch_descriptor: BatchDescriptor
    tpugraph: Optional[torch.tpu.TPUGraph] = None
    output: Optional[Any] = None

    # for tpugraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None


class TPUGraphWrapper:
    """Wraps a runnable to add tpu graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the tpugraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for tpugraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform tpugraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: TPUGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    def __init__(self,
                 runnable: Callable,
                 vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode,
                 graph_pool: Any = None,
                 cudagraph_options: Optional[CUDAGraphOptions] = None):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.graph_pool = graph_pool
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # assert runtime_mode is not NONE(no tpugraph), otherwise, we don't
        # need to initialize a ACLGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.tpugraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # tpugraphs for.
        self.concrete_tpugraph_entries: dict[BatchDescriptor, TPUGraphEntry] = {}

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of "
                             f"tpugraph wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        tpugraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if tpugraph_runtime_mode == CUDAGraphMode.NONE or \
                            tpugraph_runtime_mode != self.runtime_mode:
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without tpugraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_tpugraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_tpugraph_entries[batch_descriptor] = \
                TPUGraphEntry(batch_descriptor=batch_descriptor)

        entry = self.concrete_tpugraph_entries[batch_descriptor]

        if entry.tpugraph is None:
            #if self.tpugraph_options.debug_log_enable:
                # Since we capture tpugraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
            logger.info_once("Capturing tpugraph on (%s,%s)",
                              self.runtime_mode.name, entry.batch_descriptor)
            # validate that tpugraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            input_addresses = [
                x.data_ptr() for x in kwargs if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            tpugraph = torch_tpu.tpu.TPUGraph(keep_graph=True)

            with ExitStack() as stack:
                if self.tpugraph_options.gc_disable:
                    # during every model forward for piecewise tpugraph
                    # mode, we will capture many pieces of tpugraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the tpugraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("torch.tpu.empty_cache", lambda: None))

                self.runnable(*args, **kwargs)

                # mind-exploding: carefully manage the reference and memory.
                forward_context.capturing = True
                torch_tpu.tpu.synchronize()
                with torch_tpu.tpu.graph(tpugraph):
                    # `output` is managed by pytorch's tpugraph pool
                    output = self.runnable(*args, **kwargs)
                torch_tpu.tpu.synchronize()

            entry.output = output
            entry.tpugraph = tpugraph

            compilation_counter.num_cudagraph_captured += 1

            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for tpugraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}")

        logger.info_once("Replaying tpugraph on (%s, %s)",
                          self.runtime_mode.name, entry.batch_descriptor)
        entry.tpugraph.replay()
        return entry.output

