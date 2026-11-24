# SPDX-License-Identifier: Apache-2.0
"""Boot-time whole-forward CUDA graph for one fresh prefill of a fixed length
(V2 model runner).

Port of the whole-forward prefill graph by ArthurOstapenko (Yukon submission
c1e4b913), adapted to the V2 GPU model runner, to FlashAttention (FA2 on
sm_12x) and to models that feed the decoder `inputs_embeds` (multimodal
checkpoints run the text embedding outside the compiled model).

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
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import upload_cudagraph
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

try:
    import muse_embed as _muse_embed  # prefill3, harness/kernels/muse_embed
except (ImportError, OSError):
    _muse_embed = None

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
        self._fast_replays = 0
        self._fast_verified = False
        self._zp_enabled = os.environ.get("VLLM_PREFILL_GRAPH_ZERO_PREP", "1") != "0"
        self._zp_verified = False
        self._zp_failed = False
        self._zp_replays = 0
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
        self._raw_slot_sig = (slot_mappings.data_ptr(), tuple(slot_mappings.shape), tuple(slot_mappings.stride()))
        self._raw_bt_sig = tuple(
            (t.data_ptr(), tuple(t.shape), tuple(t.stride())) for t in block_tables
        )
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
        # LOCAL PATCH (SP5D, Yukon submission 98c1361e): a model fed `inputs_embeds` embeds the prompt's token ids
        # inside the graph (the same eager embed_input_ids, captured), so a replay needs no eager embedding first.
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
        # Own pool: nothing this graph allocates aliases the decode graphs'
        # memory, so no replay order between them can clobber an output.
        self.pool = torch.cuda.graph_pool_handle()
        self.graph = torch.cuda.CUDAGraph()
        get_offloader().sync_prev_onload()
        with torch.cuda.graph(self.graph, self.pool):
            self.hidden_states = self._forward()
            get_offloader().join_after_forward()
        upload_cudagraph(self.graph)
        torch.accelerator.synchronize()
        if not isinstance(self.hidden_states, torch.Tensor):
            raise TypeError(f"model returned {type(self.hidden_states).__name__}")
        if self.final_sample_tile:
            logger.info("Prefill graph captured both final FFN tiles on the compiled backbone")
            plans = getattr(self.sample_tile, "attn_plans", None)
            if plans is not None:
                try:
                    import muse_final_slab
                    calls = muse_final_slab.attn_stats["calls"]
                except (ImportError, OSError, AttributeError):
                    calls = -1
                logger.info("Prefill graph final-layer attention slab: %d of 2 projections admitted on the last "
                            "64 rows (%d slab launches so far)", len(plans), calls)
                try:
                    import muse_final_slab
                    logger.info("Prefill graph final rows only: %s launches narrowed to rows 448..511 so far",
                                muse_final_slab.restrict_stats)
                except (ImportError, OSError, AttributeError):
                    pass
        if _muse_embed is not None:
            logger.info("Prefill graph embedding: fused muse_embed %s (%d fused calls)",
                        _muse_embed.state["reason"] or "not used", _muse_embed.state["fused_calls"])
        self._init_zero_prep(runner)

    def _init_zero_prep(self, runner) -> None:
        try:
            n = self.num_tokens
            dev = runner.device
            ib = runner.input_buffers
            bt = runner.block_tables
            self._zp_positions_gpu = torch.arange(n, dtype=torch.int64, device=dev)
            self._zp_seq_lens_gpu = torch.zeros_like(ib.seq_lens)
            self._zp_seq_lens_gpu[0] = n
            self._zp_qsl_gpu = torch.full_like(ib.query_start_loc, n)
            self._zp_qsl_gpu[0] = 0
            self._zp_logits_indices_gpu = torch.tensor([n - 1], dtype=torch.int32, device=dev)
            self._zp_pin_ids = torch.zeros((n,), dtype=torch.int32, pin_memory=True)
            self._zp_pin_ids_np = self._zp_pin_ids.numpy()
            num_g = bt.num_kv_cache_groups
            self._zp_pin_slots = torch.zeros((num_g, n), dtype=torch.int64, pin_memory=True)
            self._zp_pin_slots_np = self._zp_pin_slots.numpy()
            self._zp_pin_bt = [
                torch.zeros((bt.input_block_tables[g].shape[1],), dtype=torch.int32, pin_memory=True)
                for g in range(num_g)
            ]
            self._zp_pin_bt_np = [t.numpy() for t in self._zp_pin_bt]
            self._zp_bs_offsets = [np.arange(bs, dtype=np.int64) for bs in bt.block_sizes]
            self._zp_nb_per_group = [(n + bs - 1) // bs for bs in bt.block_sizes]
            self._zp_cached_bids: list[list[int] | None] = [None] * num_g
            for g in range(num_g):
                nb_g = self._zp_nb_per_group[g]
                bs = bt.block_sizes[g]
                if nb_g <= self._zp_pin_bt_np[g].shape[0]:
                    canon = list(range(1, nb_g + 1))
                    bids_arr = np.asarray(canon, dtype=np.int64)
                    slots_full = (
                        bids_arr[:, None] * bs + self._zp_bs_offsets[g][None, :]
                    ).reshape(-1)
                    self._zp_pin_slots_np[g, :] = slots_full[:n]
                    self._zp_pin_bt_np[g][:nb_g] = bids_arr
                    self._zp_cached_bids[g] = canon
            self._zp_num_sched_np = np.array([n], dtype=np.int32)
            self._zp_qsl_np = np.array([0, n], dtype=np.int32)
            self._zp_prefill_len_np = np.array([n], dtype=np.int32)
            self._zp_zero_1_np = np.zeros(1, dtype=np.int32)
            self._zp_is_pref_np = np.array([True], dtype=np.bool_)
            self._zp_seq_ub = torch.from_numpy(self._zp_num_sched_np.copy())
            self._zp_const_graph: torch.cuda.CUDAGraph | None = None
            self._zp_const_skip_pad = False
            try:
                from vllm import envs
                skip_pad = bool(envs.VLLM_MOE_SKIP_PADDING)
                raw_ids = ib.input_ids[:n]
                raw_pos = ib.positions[:n]
                raw_pad = ib.is_padding[:n]
                raw_slots = bt.slot_mappings[:, :n]
                if any(
                    self._zp_nb_per_group[g] > self._zp_pin_bt_np[g].shape[0]
                    for g in range(num_g)
                ):
                    raise RuntimeError("prefill blocks exceed block_table width")

                def _stage_ops() -> None:
                    if skip_pad:
                        raw_pad.fill_(False)
                    raw_pos.copy_(self._zp_positions_gpu, non_blocking=True)
                    ib.seq_lens.copy_(self._zp_seq_lens_gpu, non_blocking=True)
                    ib.query_start_loc.copy_(self._zp_qsl_gpu, non_blocking=True)
                    raw_ids.copy_(self._zp_pin_ids, non_blocking=True)
                    for g in range(num_g):
                        nb_g = self._zp_nb_per_group[g]
                        bt.input_block_tables[g][0, :nb_g].copy_(
                            self._zp_pin_bt[g][:nb_g], non_blocking=True
                        )
                    raw_slots.copy_(self._zp_pin_slots, non_blocking=True)

                # Warmup before capture, then capture all const + 4 pinned H2D copies into one graph.
                _stage_ops()
                torch.cuda.synchronize(dev)
                cg = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cg):
                    _stage_ops()
                upload_cudagraph(cg)
                # Leave persistent dummy buffers clean for initial warmup passes.
                raw_ids.zero_()
                for g in range(num_g):
                    bt.input_block_tables[g].zero_()
                bt.slot_mappings.fill_(PAD_SLOT_ID)
                ib.is_padding.fill_(True)
                torch.cuda.synchronize(dev)
                self._zp_const_graph = cg
                self._zp_const_skip_pad = skip_pad
            except Exception:  # noqa: BLE001
                self._zp_const_graph = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prefill graph zero-prep init skipped: %s", exc)
            self._zp_failed = True

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
                # prefill3: the same embedding + weightless embedding norm in three kernels written straight into
                # inputs_embeds (harness/kernels/muse_embed; admitted by bitwise checks on its first eager call).
                if _muse_embed is None or not _muse_embed.embed_into(
                    lm, self.embed_input_ids, self.model_inputs["inputs_embeds"]
                ):
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
            if not pairs and self.embed_input_ids is not None and os.environ.get("VLLM_PREFILL_GRAPH_FAST", "1") != "0":
                self._fast_verified = True
        self.replays += 1
        return self.hidden_states

    def can_fast_replay(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        raw_input_ids: torch.Tensor | None,
    ) -> bool:
        """True when all persistent GPU buffers match the first verified zero-copy replay."""
        if not self._fast_verified or raw_input_ids is None or self.embed_input_ids is None:
            return False
        pos = self.model_inputs.get("positions")
        if pos is None:
            return False
        if (
            raw_input_ids.data_ptr() != self.embed_input_ids.data_ptr()
            or raw_input_ids.shape != self.embed_input_ids.shape
            or raw_input_ids.stride() != self.embed_input_ids.stride()
            or input_batch.positions.data_ptr() != pos.data_ptr()
            or input_batch.positions.shape != pos.shape
            or input_batch.positions.stride() != pos.stride()
            or input_batch.is_padding.data_ptr() != self.is_padding.data_ptr()
            or input_batch.is_padding.shape != self.is_padding.shape
            or slot_mappings.data_ptr() != self._raw_slot_sig[0]
            or tuple(slot_mappings.shape) != self._raw_slot_sig[1]
            or tuple(slot_mappings.stride()) != self._raw_slot_sig[2]
            or len(block_tables) != len(self._raw_bt_sig)
        ):
            return False
        for t, sig in zip(block_tables, self._raw_bt_sig):
            if (t.data_ptr(), tuple(t.shape), tuple(t.stride())) != sig:
                return False
        if (
            input_batch.prefill_len_np is None
            or input_batch.is_prefilling_np is None
            or int(input_batch.prefill_len_np[0]) != self.num_tokens
            or not bool(input_batch.is_prefilling_np[0])
        ):
            return False
        return True

    def fast_replay(self) -> torch.Tensor:
        get_offloader().sync_prev_onload()
        self.graph.replay()
        self.replays += 1
        self._fast_replays += 1
        if self._fast_replays == 1:
            logger.info("Prefill graph: fast replay active (skipping 52-layer metadata and _plan)")
        return self.hidden_states

    def stage_zero_prep(
        self,
        input_buffers,
        block_tables,
        prefill_token_ids: list[int],
        block_ids: tuple[list[int], ...],
        pad_id: int,
        skip_padding: bool,
    ) -> bool:
        if not self._fast_verified or not self._zp_enabled or self._zp_failed or self.embed_input_ids is None:
            return False
        n = self.num_tokens
        pos = self.model_inputs.get("positions")
        if pos is None or len(prefill_token_ids) != n or len(block_ids) != block_tables.num_kv_cache_groups:
            return False
        raw_ids = input_buffers.input_ids[:n]
        raw_pos = input_buffers.positions[:n]
        raw_pad = input_buffers.is_padding[:n]
        raw_slots = block_tables.slot_mappings[:, :n]
        if (
            raw_ids.data_ptr() != self.embed_input_ids.data_ptr()
            or raw_pos.data_ptr() != pos.data_ptr()
            or raw_pad.data_ptr() != self.is_padding.data_ptr()
            or raw_slots.data_ptr() != self._raw_slot_sig[0]
            or len(block_tables.input_block_tables) != len(self._raw_bt_sig)
        ):
            return False
        for g, sig in enumerate(self._raw_bt_sig):
            bt_view = block_tables.input_block_tables[g][:1]
            if (bt_view.data_ptr(), tuple(bt_view.shape), tuple(bt_view.stride())) != sig:
                return False
        can_use_const_graph = (
            self._zp_const_graph is not None
            and skip_padding == self._zp_const_skip_pad
        )
        for g in range(block_tables.num_kv_cache_groups):
            bs = block_tables.block_sizes[g]
            bids = block_ids[g]
            nb = len(bids)
            if nb * bs < n or nb > self._zp_pin_bt_np[g].shape[0]:
                return False
            if nb != self._zp_nb_per_group[g]:
                can_use_const_graph = False
            if bids != self._zp_cached_bids[g]:
                bids_arr = np.asarray(bids, dtype=np.int64)
                slots_full = (
                    bids_arr[:, None] * bs + self._zp_bs_offsets[g][None, :]
                ).reshape(-1)
                self._zp_pin_slots_np[g, :] = slots_full[:n]
                self._zp_pin_bt_np[g][:nb] = bids_arr
                self._zp_cached_bids[g] = list(bids)
        self._zp_pin_ids_np[:] = prefill_token_ids
        if can_use_const_graph:
            self._zp_const_graph.replay()
        else:
            if skip_padding:
                raw_pad.fill_(False)
            raw_pos.copy_(self._zp_positions_gpu, non_blocking=True)
            input_buffers.seq_lens.copy_(self._zp_seq_lens_gpu, non_blocking=True)
            input_buffers.query_start_loc.copy_(self._zp_qsl_gpu, non_blocking=True)
            raw_ids.copy_(self._zp_pin_ids, non_blocking=True)
            for g in range(block_tables.num_kv_cache_groups):
                nb = len(block_ids[g])
                block_tables.input_block_tables[g][0, :nb].copy_(
                    self._zp_pin_bt[g][:nb], non_blocking=True
                )
            raw_slots.copy_(self._zp_pin_slots, non_blocking=True)
        return True
