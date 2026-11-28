# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import time

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput

# LOCAL PATCH (two-stage output wait). VLLM_TWO_STAGE_WAIT=0 keeps one blocking
# wait for every step.
_TWO_STAGE_WAIT = os.environ.get("VLLM_TWO_STAGE_WAIT", "1") != "0"
# The longest the tail of a step is polled before the wait blocks again.
_TAIL_POLL_NS = 20_000_000


def wait_for_output(copy_event, forward_done=None, clock=time.perf_counter_ns) -> None:
    """Return once `copy_event` has completed.

    A blocking event wait sleeps in the driver and wakes a few hundred
    microseconds after the GPU is done (measured on GB10: about 0.33 ms). A
    decode step does not care: the next step is already queued on the device
    while the host waits. A prefill step's caller does: nothing else is queued
    and the reply waits for this wake-up. For such a step the runner passes
    `forward_done`, an event recorded in stream order after the forward and
    before the sampling tail (vocabulary head, sampler, output copy, about
    3 ms). The host sleeps on that one, so the late wake-up lands inside the
    tail, and then polls the copy event through the rest of the tail. Polling
    a whole step instead would keep one CPU core busy for its full length,
    which on a shared CPU/GPU power budget slows the GPU more than the
    wake-up costs. The same events complete in the same order either way; the
    wait changes nothing the device computes.
    """
    if forward_done is None:
        copy_event.synchronize()
        return
    forward_done.synchronize()
    deadline = clock() + _TAIL_POLL_NS
    done = copy_event.query
    while not done():
        if clock() > deadline:
            copy_event.synchronize()
            return


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        check_ep_fault: bool = False,
        known_single_token: bool = False,
        forward_done: torch.cuda.Event | None = None,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        # LOCAL PATCH (SP5D, Yukon submission 98c1361e; without its spin wait): one request, no draft tokens, a
        # decode step: exactly one sampled token, so num_sampled_tokens is not copied back and the output is
        # [[sampled_token_ids[0, 0]]], what the general path below produces from a count of 1.
        self.known_single_token = known_single_token
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)
        # See wait_for_output: set by the runner for a prefill step only.
        self.forward_done = forward_done if _TWO_STAGE_WAIT else None
        self._has_fault: torch.Tensor | None = None

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)

            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            self.num_sampled_tokens_np = (
                None if known_single_token else async_copy_to_np(num_sampled_tokens)
            )
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            if check_ep_fault:
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        wait_for_output(self.copy_event, self.forward_done)

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        if self.known_single_token:
            self.model_runner_output.sampled_token_ids = [
                [int(self.sampled_token_ids[0, 0])]
            ]
        else:
            sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
            num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
            for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
                del token_ids[num_tokens:]
            self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict

        if self._has_fault is not None and self._has_fault.item():
            mask = get_ep_all2all_manager().query_active_mask()
            raise RuntimeError(
                "Fault detected in EP all2all communication: "
                "one or more ranks timed out during dispatch/combine. "
                f"Mask: {mask.cpu().tolist()}"
            )

        return self.model_runner_output


class PreRecordedAsyncOutput(AsyncModelRunnerOutput):
    """Zero-DMA UVA pinned host ring output for compound verify+draft CUDA graphs."""

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        host_row_np: np.ndarray,
        copy_event: torch.cuda.Event,
        forward_done: torch.cuda.Event | None = None,
    ):
        self.model_runner_output = model_runner_output
        self.host_row_np = host_row_np
        self.copy_event = copy_event
        self.forward_done = forward_done if _TWO_STAGE_WAIT else None

    def get_output(self) -> ModelRunnerOutput:
        wait_for_output(self.copy_event, self.forward_done)
        if self.host_row_np.shape[0] >= 16:
            ns = int(self.host_row_np[15])
            max_ns = 13
        else:
            ns = int(self.host_row_np[7])
            max_ns = 7
        if ns < 0:
            ns = 0
        elif ns > max_ns:
            ns = max_ns
        self.model_runner_output.sampled_token_ids = [
            self.host_row_np[:ns].tolist()
        ]
        return self.model_runner_output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: PoolerOutput,
        finished_mask: list[bool],
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            if isinstance(self.pooler_output, torch.Tensor) and all(finished_mask):
                self.pooler_output_cpu: PoolerOutput = self.pooler_output.to(
                    "cpu", non_blocking=True
                )
            else:
                outputs = (
                    self.pooler_output.unbind()
                    if isinstance(self.pooler_output, torch.Tensor)
                    else self.pooler_output
                )
                self.pooler_output_cpu = [
                    None
                    if output is None or not is_finished
                    else output.to("cpu", non_blocking=True)
                    for output, is_finished in zip(outputs, finished_mask, strict=True)
                ]
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        if isinstance(self.pooler_output_cpu, torch.Tensor):
            pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        else:
            pooler_output = self.pooler_output_cpu
        self.copy_event.synchronize()
        self.model_runner_output.pooler_output = pooler_output
        return self.model_runner_output


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()


@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """Lightweight version of torch.cuda.stream() context manager which
    avoids current_stream and device lookups.
    """
    try:
        torch.cuda.set_stream(to_stream)
        yield
    finally:
        torch.cuda.set_stream(from_stream)
