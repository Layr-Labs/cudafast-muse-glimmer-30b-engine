# SPDX-License-Identifier: Apache-2.0
"""Boot-time whole-forward CUDA graph for one fresh prefill of a fixed length
(V2 model runner), supporting FlashAttention (FA2 on sm_12x) and models that
feed the decoder `inputs_embeds` (multimodal checkpoints running text
embedding outside the compiled model).

A single-request, unpadded prefill of exactly `num_tokens` tokens with nothing
computed before it is dispatched eagerly (cg_mode NONE) when the cudagraph
capture sizes stop below it. At boot, after the dispatcher's own graphs, this
module builds that step's inputs the way the V2 FULL-graph capture does (the
runner's persistent input buffers, dummy block tables and PAD slot mappings, so
nothing is written to the KV cache), runs the forward eagerly twice (JIT,
autotune, allocator growth) and captures it once into a CUDA graph with its
own memory pool.

Exactness: the graph holds the eager forward's own kernels, launched by the
same host code from the same host values. `try_replay` refuses (the caller
then runs eager) unless every host scalar of the step's attention metadata
(num_actual_tokens, max_query_len, max_seq_len, max_num_splits, causal,
sliding_window, use_cascade, ...) and of the model inputs equals the captured
one and every device tensor has the captured shape, dtype, strides and device;
CPU tensors must be equal. The runner writes a step's inputs into the same
persistent buffers the capture read, so normally nothing is copied; a device
tensor at a different address is copied into the captured one before replay.
The logits gather and compute_logits stay outside the graph, unchanged.

Set VLLM_PREFILL_GRAPH=0 to disable; VLLM_PREFILL_GRAPH_TOKENS picks the
length (default 512).
"""

from __future__ import annotations

import dataclasses
import enum
import os
import time
from typing import Any

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

_SCALARS = (
    int,
    float,
    bool,
    str,
    bytes,
    type(None),
    torch.dtype,
    torch.device,
    enum.Enum,
)


def _plan(dst: Any, src: Any, pairs: list, seen: dict, path: str) -> str | None:
    """Match the step's `src` against the captured `dst`.

    Collects (dst, src) device copies for tensors at a different address.
    Returns None on a match, else the path of the first difference."""
    if type(dst) is not type(src):
        return f"{path}: type {type(dst).__name__} != {type(src).__name__}"
    if isinstance(dst, torch.Tensor):
        prev = seen.get(id(dst))
        if prev is not None:
            # A tensor the capture shared must be fed one memory region.
            same = prev is src or (
                prev.data_ptr() == src.data_ptr()
                and prev.shape == src.shape
                and prev.stride() == src.stride()
                and prev.dtype == src.dtype
            )
            return None if same else f"{path}: shared tensor fed two regions"
        seen[id(dst)] = src
        if (
            dst.shape != src.shape
            or dst.dtype != src.dtype
            or dst.stride() != src.stride()
            or dst.device != src.device
        ):
            return (
                f"{path}: tensor {tuple(dst.shape)}/{dst.dtype}/{dst.stride()} != "
                f"{tuple(src.shape)}/{src.dtype}/{src.stride()}"
            )
        if dst.device.type == "cpu":
            return None if torch.equal(dst, src) else f"{path}: cpu tensor values"
        if dst.data_ptr() != src.data_ptr():
            pairs.append((dst, src))
        return None
    if isinstance(dst, np.ndarray):
        same = dst.dtype == src.dtype and np.array_equal(dst, src)
        return None if same else f"{path}: ndarray values"
    if isinstance(dst, (dict, list, tuple)) or (
        dataclasses.is_dataclass(dst) and not isinstance(dst, type)
    ):
        # Per-layer metadata shares one object across the layers of a group.
        if seen.get(("node", id(dst))) is src:
            return None
        seen[("node", id(dst))] = src
    if dataclasses.is_dataclass(dst) and not isinstance(dst, type):
        for f in dataclasses.fields(dst):
            diff = _plan(
                getattr(dst, f.name), getattr(src, f.name), pairs, seen,
                f"{path}.{f.name}",
            )
            if diff is not None:
                return diff
        return None
    if isinstance(dst, dict):
        if dst.keys() != src.keys():
            return f"{path}: keys differ"
        for k in dst:
            diff = _plan(dst[k], src[k], pairs, seen, f"{path}[{k!r}]")
            if diff is not None:
                return diff
        return None
    if isinstance(dst, (list, tuple)):
        if len(dst) != len(src):
            return f"{path}: length {len(dst)} != {len(src)}"
        for i, (a, b) in enumerate(zip(dst, src)):
            diff = _plan(a, b, pairs, seen, f"{path}[{i}]")
            if diff is not None:
                return diff
        return None
    if dst is src:
        return None
    if isinstance(dst, _SCALARS):
        return None if dst == src else f"{path}: {dst!r} != {src!r}"
    # Unknown object: only identity or a plain-bool equality counts.
    try:
        same = dst == src
    except Exception:
        same = False
    return None if same is True else f"{path}: {type(dst).__name__} differs"


def prefill_graph_unsupported(runner, num_tokens: int) -> str | None:
    """Why the whole-forward prefill graph cannot serve this engine."""
    if os.environ.get("VLLM_PREFILL_GRAPH", "1") == "0":
        return "VLLM_PREFILL_GRAPH=0"
    if not 0 < num_tokens <= min(runner.max_num_tokens, runner.max_model_len):
        return f"{num_tokens} tokens do not fit the engine"
    if runner.device.type != "cuda":
        return f"device {runner.device.type}"
    manager = runner.cudagraph_manager
    if manager is None or not manager.needs_capture():
        return "CUDA graphs are off"
    if any(key[0] == num_tokens for key in manager._candidates):
        return "the dispatcher already graphs this size"
    pc = runner.parallel_config
    if (
        pc.tensor_parallel_size,
        pc.pipeline_parallel_size,
        pc.data_parallel_size,
    ) != (1, 1, 1) or runner.use_dcp:
        return "parallelism"
    if (
        runner.speculator is not None
        or runner.speculative_config is not None
        or runner.lora_config
        or runner.is_pooling_model
        or runner.use_aux_hidden_state_outputs
        or runner.is_encoder_decoder
        or runner.pcp_manager is not None
        or getattr(runner.cache_config, "calculate_kv_scales", False)
    ):
        return "engine feature outside the captured shape"
    from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR

    if runner.kv_connector is not NO_OP_KV_CONNECTOR:
        return "kv connector"
    for groups in runner.attn_groups:
        for group in groups:
            name = group.backend.get_name()
            if name != "FLASH_ATTN":
                return f"attention backend {name}"
    return None


def capture_prefill_graph(runner) -> PrefillGraph | None:
    """Capture the graph for the runner, or None (with a log line)."""
    num_tokens = int(os.environ.get("VLLM_PREFILL_GRAPH_TOKENS", "512"))
    reason = prefill_graph_unsupported(runner, num_tokens)
    if reason is not None:
        logger.info("Prefill graph off: %s", reason)
        return None
    started = time.perf_counter()
    graph = None
    prepare_tile = getattr(runner.model, "prepare_final_sample_tile", None)
    if prepare_tile is not None:
        try:
            sample_tile = prepare_tile(num_tokens)
            if sample_tile is not None:
                with graph_capture(device=runner.device):
                    graph = PrefillGraph(runner, num_tokens, sample_tile=sample_tile)
        except Exception as exc:  # noqa: BLE001 - retry with the original full graph
            logger.warning(
                "Compact prefill graph capture failed (%s: %s); trying the full graph",
                type(exc).__name__, exc,
            )
            try:
                torch.accelerator.synchronize()
            except Exception:  # noqa: BLE001
                pass
    try:
        if graph is None:
            with graph_capture(device=runner.device):
                graph = PrefillGraph(runner, num_tokens)
    except Exception as exc:  # noqa: BLE001 - any failure falls back to eager
        logger.warning(
            "Prefill graph off: capture failed (%s: %s); prefills run eager",
            type(exc).__name__,
            exc,
        )
        try:
            torch.accelerator.synchronize()
        except Exception:  # noqa: BLE001
            pass
        return None
    logger.info(
        "Prefill graph: %d-token whole forward captured in %.2f s",
        num_tokens,
        time.perf_counter() - started,
    )
    return graph


class PrefillGraph:
    def __init__(self, runner, num_tokens: int, *, sample_tile=None):
        n = num_tokens
        self.num_tokens = n
        self.vllm_config = runner.vllm_config
        self.model = runner.model
        self.sample_tile = sample_tile
        self.final_sample_tile = sample_tile is not None
        self.replays = 0
        self._logged_refusal = False

        model_state = runner.model_state
        input_buffers = runner.input_buffers
        # The step a fresh one-request prefill of n tokens builds
        # (prepare_inputs): seq_len == query_len == n, nothing computed.
        input_batch = InputBatch.make_dummy(1, n, input_buffers)
        input_batch.prefill_len_np = np.full(1, n, dtype=np.int32)
        input_batch.is_prefilling_np = np.ones(1, dtype=np.bool_)
        block_tables = runner.block_tables.get_dummy_block_tables(1)
        # PAD slots: the warm-up runs and the capture write no KV.
        slot_mappings = runner.block_tables.get_dummy_slot_mappings(n)
        self.slot_mappings = build_slot_mappings_by_layer(
            slot_mappings, runner.kv_cache_config
        )
        self.attn_metadata = model_state.prepare_attn(
            input_batch,
            CUDAGraphMode.NONE,
            block_tables,
            slot_mappings,
            runner.attn_groups,
            runner.kv_cache_config,
            for_capture=False,
        )
        # Same dict execute_model builds for an eager step.
        dummy = model_state.prepare_dummy_inputs(1, n)
        input_ids = input_batch.input_ids
        inputs_embeds = None
        self.embed_input_ids: torch.Tensor | None = None
        if runner.supports_mm_inputs and runner.is_first_pp_rank:
            inputs_embeds = dummy.get("inputs_embeds")
            if inputs_embeds is not None and not runner.model.requires_raw_input_tokens:
                self.embed_input_ids = input_batch.input_ids
                input_ids = None
        self.model_inputs = {
            "input_ids": input_ids,
            "positions": dummy.get("positions", input_batch.positions),
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": None,
        }
        self.is_padding = input_batch.is_padding
        self.batch_descriptor = BatchDescriptor(
            num_tokens=n, has_lora=False, num_active_loras=0
        )

        for _ in range(2):
            self._forward()
        torch.accelerator.synchronize()
        # Micro-attribution: Capturing a dedicated fixed-length single-request prefill CUDA graph
        # with dummy PAD slot mappings at boot is inspired by @ArthurOstapenko (Yukon submission c1e4b913);
        # the V2 runner persistent-buffer binding, final-slab last-chunk routing, and zero-VRAM fp8ws
        # dual-TMA integration around it are custom-implemented here.
        # Own pool: nothing this graph allocates aliases the decode graphs'
        # memory, so no replay order between them can clobber an output.
        self.pool = torch.cuda.graph_pool_handle()
        self.graph = torch.cuda.CUDAGraph()
        get_offloader().sync_prev_onload()
        with torch.cuda.graph(self.graph, self.pool):
            self.hidden_states = self._forward()
            get_offloader().join_after_forward()
        torch.accelerator.synchronize()
        if not isinstance(self.hidden_states, torch.Tensor):
            raise TypeError(f"model returned {type(self.hidden_states).__name__}")
        if self.final_sample_tile:
            logger.info("Prefill graph captured both final FFN tiles on the compiled backbone")

    def _forward(self):
        with set_forward_context(
            self.attn_metadata,
            self.vllm_config,
            num_tokens=self.num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=None,
            batch_descriptor=self.batch_descriptor,
            slot_mapping=self.slot_mappings,
            is_padding=self.is_padding,
        ):
            if (
                self.embed_input_ids is not None
                and self.model_inputs.get("inputs_embeds") is not None
            ):
                lm = (
                    self.model.get_language_model()
                    if hasattr(self.model, "get_language_model")
                    else self.model
                )
                self.model_inputs["inputs_embeds"].copy_(
                    lm.embed_input_ids(self.embed_input_ids)
                )
            if self.sample_tile is None:
                return self.model(**self.model_inputs)
            with self.sample_tile.capture():
                hidden_states = self.model(**self.model_inputs)
                return hidden_states[-64:]

    def try_replay(
        self,
        attn_metadata: Any,
        slot_mappings: Any,
        model_inputs: dict[str, Any],
        is_padding: torch.Tensor,
        batch_descriptor: BatchDescriptor,
        raw_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Hidden states from a replay, or None when this step differs from
        the captured one in anything but device bytes."""
        if self.embed_input_ids is not None and raw_input_ids is None:
            return None
        pairs: list = []
        seen: dict = {}
        diff = None
        if batch_descriptor != self.batch_descriptor:
            diff = f"batch_descriptor {batch_descriptor}"
        checks = [
            ("attn_metadata", self.attn_metadata, attn_metadata),
            ("slot_mapping", self.slot_mappings, slot_mappings),
            ("model_inputs", self.model_inputs, model_inputs),
            ("is_padding", self.is_padding, is_padding),
        ]
        if self.embed_input_ids is not None:
            checks.append(("raw_input_ids", self.embed_input_ids, raw_input_ids))
        for name, dst, src in checks:
            if diff is not None:
                break
            diff = _plan(dst, src, pairs, seen, name)
        if diff is not None:
            if not self._logged_refusal:
                self._logged_refusal = True
                logger.warning(
                    "Prefill graph: %d-token prefill ran eager (%s)",
                    self.num_tokens,
                    diff,
                )
            return None
        for dst, src in pairs:
            dst.copy_(src, non_blocking=True)
        get_offloader().sync_prev_onload()
        self.graph.replay()
        if self.replays == 0:
            logger.info("Prefill graph: first %d-token replay", self.num_tokens)
        self.replays += 1
        return self.hidden_states
