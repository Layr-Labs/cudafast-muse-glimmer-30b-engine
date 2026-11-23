# SPDX-License-Identifier: Apache-2.0
"""Where the one-row decode, 512-row prefill weights, and KV cache live in device memory.

On a unified-memory GPU (GB10), streaming GEMV bandwidth and 512-row L2 reuse
depend on the physical DRAM pages backing a weight or KV tensor. After loading,
we drop clean checkpoint page cache via posix_fadvise(DONTNEED), allocate
spare buffers, flush L2 with a read-only pass (avoiding dirty-line write
penalties and L2-warm spare bias), use two-stage successive halving to eliminate
winner's curse cleanly, hold rejected slow buffers across decode/prefill/KV
allocation so later groups do not recycle slow pages, and exchange slow homes
with faster spares after a bitwise equality check.

  MUSE_DECODE_PLACEMENT=0             no decode placement step
  MUSE_DECODE_PLACEMENT_SURPLUS=<r>   spare buffers per decode weight (default 3.5)
  MUSE_DECODE_MIN_BYTES=<n>           minimum decode payload bytes (default 512 KiB)
  MUSE_DECODE_SCALE_MIN_BYTES=<n>     minimum decode scale bytes (default 512 KiB)
  MUSE_PREFILL_PLACEMENT=0            no prefill placement step
  MUSE_PREFILL_PLACEMENT_SURPLUS=<r>  spare buffers per prefill weight (default 2.0)
  MUSE_KV_PLACEMENT=0                 no KV cache backing-buffer swap
  MUSE_KV_BLOCK_SORT=0                no KV free-block-queue sorting
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from statistics import median

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("MUSE_DECODE_PLACEMENT", "1").strip() not in ("0", "", "false", "no")
SURPLUS = float(os.environ.get("MUSE_DECODE_PLACEMENT_SURPLUS", "3.5"))
LM_HEAD_SPARES = int(os.environ.get("MUSE_LM_HEAD_SPARES", "16"))
ROUNDS = 5
SCREEN_ROUNDS = max(1, int(os.environ.get("MUSE_PLACEMENT_SCREEN_ROUNDS", "3")))
REFINE_ROUNDS = max(0, int(os.environ.get("MUSE_PLACEMENT_REFINE_ROUNDS", "8")))
# Weights that fit in L2 several times over are not bandwidth-bound per prefill call.
MIN_PAYLOAD_BYTES = 48 << 20
# Decode tensors >= 512 KiB (including o_proj, qkv, and NVFP4/rANS scales) stream cold from DRAM per token.
DECODE_MIN_BYTES = int(os.environ.get("MUSE_DECODE_MIN_BYTES", str(512 << 10)))
SCALE_MIN_BYTES = int(os.environ.get("MUSE_DECODE_SCALE_MIN_BYTES", str(512 << 10)))
STREAM_READ_BLOCKS = int(os.environ.get("MUSE_STREAM_READ_BLOCKS", "192"))
# Host memory that must stay available while the spares exist.
HOST_HEADROOM_BYTES = 16 << 30
_NAME = "_marlin_decode_weight"
_L2_FLUSH_INTS = 16 << 20  # 64 MiB (> 2.5x GB10's 24 MiB L2)
# Quarantine below-median (slow) buffers from place_decode_weights through initialize_kv_cache.
_SLOW_HOLD: list[torch.Tensor] = []

KV_ENABLED = os.environ.get("MUSE_KV_PLACEMENT", "1").strip() not in ("0", "", "false", "no")
KV_BLOCK_SORT_ENABLED = os.environ.get("MUSE_KV_BLOCK_SORT", "1").strip() not in ("0", "", "false", "no")
KV_SURPLUS = float(os.environ.get("MUSE_KV_PLACEMENT_SURPLUS", "2.0"))
KV_ROUNDS = 3
KV_WINDOW_BYTES = 32 << 20
_KV_BLOCK_RANKS: list[float] | None = None


def _drop_model_page_cache() -> None:
    """Release clean checkpoint file pages back to Linux free memory on unified-RAM hosts."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    model_dir = os.environ.get("RESIDENT_MODEL_DIR", "")
    if not model_dir:
        return
    root = Path(model_dir)
    if not root.is_dir():
        return
    dropped = 0
    for path in root.glob("*.safetensors"):
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                dropped += 1
            finally:
                os.close(fd)
        except OSError:
            pass
    if dropped:
        logger.info("Released page cache for %d safetensors shards via posix_fadvise(DONTNEED)", dropped)


def _flush_l2(evict_buf: torch.Tensor, flush_out: torch.Tensor | None = None) -> None:
    """Read-only L2 eviction pass (never writes, avoiding dirty-line writeback overhead)."""
    if flush_out is not None:
        torch.sum(evict_buf.reshape(-1), dim=0, out=flush_out)
    else:
        _ = evict_buf.sum()


def _mem_available() -> int | None:
    try:
        with open("/proc/meminfo") as fp:
            for line in fp:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _spares_allowed(wanted: int, payload_bytes: int) -> int:
    """Spare buffers the host can carry right now, at most ``wanted``."""
    available = _mem_available()
    if available is None:
        return 0
    return max(0, min(wanted, (available - HOST_HEADROOM_BYTES) // payload_bytes))


def _ensure_spares_room(wanted: int, payload_bytes: int, *hold_lists: list) -> int:
    """Release oldest quarantined buffers from hold_lists (in order) only if needed for headroom."""
    spares_count = _spares_allowed(wanted, payload_bytes)
    for hold_list in hold_lists:
        if spares_count >= wanted:
            break
        while hold_list and spares_count < wanted:
            del hold_list[: max(1, len(hold_list) // 4)]
            torch.cuda.empty_cache()
            spares_count = _spares_allowed(wanted, payload_bytes)
    return spares_count


def ensure_kv_cache_room(required_bytes: int) -> None:
    """Release oldest quarantined slow buffers only if host RAM lacks room for KV cache."""
    if not _SLOW_HOLD or required_bytes <= 0:
        return
    needed = max(1, required_bytes)
    while _SLOW_HOLD:
        available = _mem_available()
        if available is None or available - HOST_HEADROOM_BYTES >= needed:
            break
        del _SLOW_HOLD[: max(1, len(_SLOW_HOLD) // 4)]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def release_slow_hold() -> None:
    """Release quarantined slow buffers once KV cache allocation has finished."""
    if not _SLOW_HOLD:
        return
    _SLOW_HOLD.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_kv_block_ranks(num_blocks: int) -> list[float] | None:
    """Return per-block read-latency ranks measured during KV cache allocation."""
    ranks = _KV_BLOCK_RANKS
    if ranks is None or len(ranks) != num_blocks:
        return None
    return ranks


def _halving_keep_count(total: int, num_homes: int | None) -> int:
    count = max(1, num_homes) if num_homes is not None else 1
    spares = max(0, total - count)
    min_keep = count + min(spares, max(1, min(count // 2, spares // 2)))
    return min(total, max(min_keep, math.ceil(0.4 * total)))


def _resolve_halving_rounds(
    rounds: int,
    default_rounds: int,
    default_screen: int,
    default_refine: int,
    screen_rounds: int | None,
    refine_rounds: int | None,
) -> tuple[int, int]:
    if screen_rounds is not None and refine_rounds is not None:
        return max(1, screen_rounds), max(0, refine_rounds)
    if rounds == default_rounds:
        return default_screen, default_refine
    if rounds <= 2:
        return max(1, rounds), 0
    scr = max(2, rounds // 2)
    return scr, rounds - scr


def _partition_candidates(times: list[float], count: int) -> tuple[float, list[int], list[int]]:
    """Split candidates into slow threshold, above-count fast spares, and below-count movers."""
    slow_cut = median(times)
    ranked = sorted(range(len(times)), key=times.__getitem__)
    kept = set(ranked[:count])
    free_fast = [index for index in ranked[:count] if index >= count]
    movers = sorted((index for index in range(count) if index not in kept), key=times.__getitem__, reverse=True)
    return slow_cut, free_fast, movers


def _hold_rejected(buf: torch.Tensor, elapsed: float, slow_cut: float, excess_hold: list) -> None:
    """Quarantine slow buffers across phases; keep above-median spares in excess_hold."""
    if elapsed > slow_cut:
        _SLOW_HOLD.append(buf)
    else:
        excess_hold.append(buf)


def _time_calls(
    run,
    candidates,
    rounds: int,
    evict_buf: torch.Tensor | None = None,
    *,
    num_homes: int | None = None,
    screen_rounds: int | None = None,
    refine_rounds: int | None = None,
    flush_each: bool = False,
) -> list[float]:
    """Median time (ms) of run(candidate) per candidate via two-stage successive halving."""
    total = len(candidates)
    if total == 0:
        return []
    device = candidates[0].device
    flush_out = torch.empty((), dtype=torch.int64, device=device) if evict_buf is not None else None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    run(candidates[0])
    if evict_buf is not None:
        _flush_l2(evict_buf, flush_out)
    torch.cuda.synchronize(device)

    scr, ref = _resolve_halving_rounds(rounds, ROUNDS, SCREEN_ROUNDS, REFINE_ROUNDS, screen_rounds, refine_rounds)
    keep = _halving_keep_count(total, num_homes)
    samples: list[list[float]] = [[] for _ in range(total)]
    shuffler = random.Random(20261122)

    def _run_rounds(order: list[int], n_rounds: int) -> None:
        for _ in range(n_rounds):
            shuffler.shuffle(order)
            for index in order:
                if flush_each and evict_buf is not None:
                    _flush_l2(evict_buf, flush_out)
                start.record()
                run(candidates[index])
                end.record()
                end.synchronize()
                samples[index].append(start.elapsed_time(end))

    _run_rounds(list(range(total)), scr)
    if ref > 0:
        if keep < total:
            stage1 = [median(values) for values in samples]
            finalists = sorted(range(total), key=stage1.__getitem__)[:keep]
        else:
            finalists = list(range(total))
        _run_rounds(finalists, ref)
    return [median(values) for values in samples]


def _time_candidates(
    decode_linear,
    layer,
    candidates,
    n: int,
    k: int,
    rounds: int,
    evict_buf: torch.Tensor,
    flush_each: bool = False,
    num_homes: int | None = None,
) -> list[float]:
    """Median GEMV time (ms) of the tiled decode kernel on each candidate buffer."""
    device = candidates[0].device
    # A fixed finite row; the kernel's timing does not depend on values.
    x = ((torch.arange(k, device=device) % 7).float() - 3.0).to(torch.bfloat16).reshape(1, k)
    scales, global_scale, workspace = layer._marlin_decode_scales, layer.weight_global_scale, layer.workspace
    return _time_calls(
        lambda buf: decode_linear(x, buf, scales, global_scale, workspace, n, k),
        candidates,
        rounds,
        evict_buf,
        num_homes=num_homes,
        flush_each=flush_each,
    )


def _place_group(
    decode_linear,
    layers,
    n: int,
    k: int,
    evict_buf: torch.Tensor,
    excess_hold: list,
    stream_read_fn=None,
    sink: torch.Tensor | None = None,
) -> dict | None:
    homes = [layer._buffers[_NAME] for layer in layers]
    count = len(homes)
    payload_bytes = homes[0].numel() * homes[0].element_size()
    wanted = max(LM_HEAD_SPARES if count == 1 else 1, math.ceil(count * SURPLUS)) if SURPLUS > 0 else 0
    spares_count = _ensure_spares_room(wanted, payload_bytes, excess_hold)
    if spares_count <= 0:
        return None
    spares = []
    try:
        for _ in range(spares_count):
            spare = torch.empty_like(homes[0])
            spare.copy_(homes[0])  # every page written once before it is timed
            spares.append(spare)
    except RuntimeError:  # out of device memory: time what exists
        pass
    if not spares:
        return None
    candidates = homes + spares
    flush_each = payload_bytes < MIN_PAYLOAD_BYTES
    if flush_each and stream_read_fn is not None and sink is not None and payload_bytes >= 16 and homes[0].data_ptr() % 16 == 0:
        times = _time_calls(
            lambda buf: stream_read_fn(buf, sink, blocks=STREAM_READ_BLOCKS),
            candidates,
            ROUNDS,
            evict_buf,
            num_homes=count,
            flush_each=True,
        )
    else:
        times = _time_candidates(
            decode_linear, layers[0], candidates, n, k, ROUNDS, evict_buf, flush_each=flush_each, num_homes=count
        )
    slow_cut, free_fast, movers = _partition_candidates(times, count)
    moved = 0
    after = list(times[:count])
    used_targets = set()
    for home_index, target_index in zip(movers, free_fast):
        target, source = candidates[target_index], homes[home_index]
        target_u8 = target.reshape(-1).view(torch.uint8)
        source_u8 = source.reshape(-1).view(torch.uint8)
        target_u8.copy_(source_u8)
        if not torch.equal(target_u8, source_u8):
            continue  # keep the loader's buffer
        layers[home_index]._buffers[_NAME] = target
        used_targets.add(target_index)
        _hold_rejected(source, times[home_index], slow_cut, excess_hold)
        after[home_index] = times[target_index]
        moved += 1
    for i, spare in enumerate(spares):
        idx = count + i
        if idx not in used_targets:
            _hold_rejected(spare, times[idx], slow_cut, excess_hold)
    before = sorted(times[:count])
    after.sort()
    return {
        "count": count, "spares": len(spares), "moved": moved, "payload_bytes": payload_bytes,
        "before": (before[0], median(before), before[-1], sum(before)),
        "after": (after[0], median(after), after[-1], sum(after)),
    }


def _place_muserans_group(
    rans_ext,
    layers,
    n: int,
    k: int,
    evict_buf: torch.Tensor,
    excess_hold: list,
    stream_read_fn=None,
    sink: torch.Tensor | None = None,
) -> dict | None:
    homes = [layer._buffers["muse_fp8_packed"] for layer in layers]
    count = len(homes)
    max_bytes = max(h.numel() for h in homes)
    wanted = max(LM_HEAD_SPARES if count == 1 else 1, math.ceil(count * SURPLUS)) if SURPLUS > 0 else 0
    spares_count = _ensure_spares_room(wanted, max_bytes, excess_hold)
    if spares_count <= 0:
        return None
    ref_len = homes[0].numel()
    # Micro-attribution: Storage-swapping source.data in _place_muserans_group is inspired by @kaankolcu
    # (Yukon submission 9daf69d2); cross-phase _SLOW_HOLD quarantine and total-bytes group sorting are
    # inspired by @zereraz (Yukon submission 63ea2ee2).
    spares = []
    try:
        for _ in range(spares_count):
            spare = torch.empty((max_bytes,), dtype=torch.uint8, device=homes[0].device)
            spare[:ref_len].copy_(homes[0])
            if max_bytes > ref_len:
                spare[ref_len:].zero_()
            spares.append(spare)
    except RuntimeError:
        pass
    if not spares:
        return None
    flush_each = max_bytes < MIN_PAYLOAD_BYTES
    probe_bytes = min(h.numel() for h in homes) & ~15
    if (
        flush_each
        and stream_read_fn is not None
        and sink is not None
        and probe_bytes >= 16
        and all(h.data_ptr() % 16 == 0 for h in homes)
    ):
        cand_views = [h[:probe_bytes] for h in homes] + [s[:probe_bytes] for s in spares]
        times = _time_calls(
            lambda buf: stream_read_fn(buf, sink, blocks=STREAM_READ_BLOCKS),
            cand_views,
            ROUNDS,
            evict_buf,
            num_homes=count,
            flush_each=True,
        )
    else:
        ref = layers[0]
        device = homes[0].device
        xq = torch.zeros((k,), dtype=torch.float8_e4m3fn, device=device)
        ws = ref.weight_scale.reshape(-1)
        xs = ref.input_scale.reshape(-1) if getattr(ref, "input_scale", None) is not None else torch.ones((1,), dtype=torch.float32, device=device)
        out = torch.empty((1, n), dtype=torch.bfloat16, device=device)

        def run_cand(buf):
            rans_ext.rans_gemv_auto(buf, xq, ws, xs, out, n, k)

        cand_views = list(homes) + [s[:ref_len] for s in spares]
        times = _time_calls(run_cand, cand_views, ROUNDS, evict_buf, num_homes=count, flush_each=flush_each)
    slow_cut, free_fast, movers = _partition_candidates(times, count)
    moved = 0
    after = list(times[:count])
    used_targets = set()
    for home_index, target_index in zip(movers, free_fast):
        source = homes[home_index]
        target = spares[target_index - count][: source.numel()]
        target.copy_(source)
        if not torch.equal(target, source):
            continue
        old = source.data
        source.data = target
        if source.data_ptr() != target.data_ptr():
            source.data = old
            continue
        used_targets.add(target_index)
        _hold_rejected(old, times[home_index], slow_cut, excess_hold)
        after[home_index] = times[target_index]
        moved += 1
    for i, spare in enumerate(spares):
        idx = count + i
        if idx not in used_targets:
            _hold_rejected(spare, times[idx], slow_cut, excess_hold)
    before = sorted(times[:count])
    after.sort()
    return {
        "count": count, "spares": len(spares), "moved": moved, "payload_bytes": max_bytes,
        "before": (before[0], median(before), before[-1], sum(before)),
        "after": (after[0], median(after), after[-1], sum(after)),
    }


def _place_scale_group(
    stream_read_fn,
    sink: torch.Tensor,
    layers,
    scale_name: str,
    evict_buf: torch.Tensor,
    excess_hold: list,
) -> dict | None:
    """Place active NVFP4 scale / rANS-scale buffers via pure DRAM stream_read probing."""
    homes = [layer._buffers[scale_name] for layer in layers]
    count = len(homes)
    byte_sizes = [h.numel() * h.element_size() for h in homes]
    max_bytes = max(byte_sizes)
    probe_bytes = min(byte_sizes) & ~15
    if probe_bytes < 16 or not all(h.data_ptr() % 16 == 0 for h in homes):
        return None
    wanted = max(LM_HEAD_SPARES if count == 1 else 1, math.ceil(count * SURPLUS)) if SURPLUS > 0 else 0
    spares_count = _ensure_spares_room(wanted, max_bytes, excess_hold)
    if spares_count <= 0:
        return None
    ref_u8 = homes[0].reshape(-1).view(torch.uint8)
    ref_bytes = byte_sizes[0]
    spares = []
    try:
        for _ in range(spares_count):
            spare = torch.empty((max_bytes,), dtype=torch.uint8, device=homes[0].device)
            spare[:ref_bytes].copy_(ref_u8)
            if max_bytes > ref_bytes:
                spare[ref_bytes:].zero_()
            spares.append(spare)
    except RuntimeError:
        pass
    if not spares:
        return None
    cand_views = [h.reshape(-1).view(torch.uint8)[:probe_bytes] for h in homes] + [s[:probe_bytes] for s in spares]
    flush_each = probe_bytes < MIN_PAYLOAD_BYTES
    times = _time_calls(
        lambda buf: stream_read_fn(buf, sink, blocks=STREAM_READ_BLOCKS),
        cand_views,
        ROUNDS,
        evict_buf,
        num_homes=count,
        flush_each=flush_each,
    )
    slow_cut, free_fast, movers = _partition_candidates(times, count)
    moved = 0
    after = list(times[:count])
    used_targets = set()
    for home_index, target_index in zip(movers, free_fast):
        source = homes[home_index]
        nbytes = byte_sizes[home_index]
        target = spares[target_index - count][:nbytes].view(source.dtype).reshape(source.shape)
        target_u8 = target.reshape(-1).view(torch.uint8)
        source_u8 = source.reshape(-1).view(torch.uint8)
        target_u8.copy_(source_u8)
        if not torch.equal(target_u8, source_u8):
            continue
        layers[home_index]._buffers[scale_name] = target
        used_targets.add(target_index)
        _hold_rejected(source, times[home_index], slow_cut, excess_hold)
        after[home_index] = times[target_index]
        moved += 1
    for i, spare in enumerate(spares):
        idx = count + i
        if idx not in used_targets:
            _hold_rejected(spare, times[idx], slow_cut, excess_hold)
    before = sorted(times[:count])
    after.sort()
    return {
        "count": count, "spares": len(spares), "moved": moved, "payload_bytes": probe_bytes,
        "before": (before[0], median(before), before[-1], sum(before)),
        "after": (after[0], median(after), after[-1], sum(after)),
    }


# --------------------------------------------------------------------------- #
# KV cache placement & block_id latency ranking
# --------------------------------------------------------------------------- #
def _stream_read_sum(buf: torch.Tensor) -> torch.Tensor:
    if buf.is_contiguous() and buf.data_ptr() % 8 == 0 and buf.numel() % 8 == 0:
        return buf.view(torch.int64).sum()
    return buf.sum()


def _probe_kv_block_ranks(
    unique_tensors: list[torch.Tensor],
    num_blocks: int,
    evict_buf: torch.Tensor,
) -> list[float] | None:
    """Measure read latency across ~32 MiB block_id windows spanning all KV tensors."""
    if not KV_BLOCK_SORT_ENABLED or num_blocks <= 2 or not unique_tensors:
        return None
    if not all(
        t.is_cuda
        and t.dim() == 1
        and t.is_contiguous()
        and t.numel() >= num_blocks
        and t.numel() % num_blocks == 0
        for t in unique_tensors
    ):
        return None
    tensor_strides = [(t, t.numel() // num_blocks) for t in unique_tensors]
    sum_stride = sum(st for _, st in tensor_strides)
    if sum_stride <= 0:
        return None
    win_blocks = max(1, min(num_blocks - 1, round(KV_WINDOW_BYTES / sum_stride)))
    windows = [
        (b0, min(b0 + win_blocks, num_blocks))
        for b0 in range(1, num_blocks, win_blocks)
    ]
    if len(windows) <= 1:
        return None

    device = unique_tensors[0].device
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    w0_b0, w0_b1 = windows[0]
    for t, st in tensor_strides:
        _stream_read_sum(t[w0_b0 * st : w0_b1 * st])
    _flush_l2(evict_buf)
    torch.cuda.synchronize(device)

    samples: list[list[float]] = [[] for _ in windows]
    order = list(range(len(windows)))
    shuffler = random.Random(20261122)
    for _ in range(KV_ROUNDS):
        shuffler.shuffle(order)
        for w_idx in order:
            b0, b1 = windows[w_idx]
            start.record()
            for t, st in tensor_strides:
                _stream_read_sum(t[b0 * st : b1 * st])
            end.record()
            end.synchronize()
            samples[w_idx].append(start.elapsed_time(end))

    ranks = [0.0] * num_blocks
    per_block_rates: list[float] = []
    for (b0, b1), vals in zip(windows, samples):
        rate = median(vals) / (b1 - b0)
        per_block_rates.append(rate)
        for blk in range(b0, b1):
            ranks[blk] = rate
    per_block_rates.sort()
    logger.info(
        "KV block probe: %d blocks in %d windows (%d blocks/win, %.1f MiB/win); "
        "ms/block min/median/max %.5f/%.5f/%.5f",
        num_blocks,
        len(windows),
        win_blocks,
        win_blocks * sum_stride / (1 << 20),
        per_block_rates[0],
        median(per_block_rates),
        per_block_rates[-1],
    )
    return ranks


def place_kv_cache_tensors(
    raw_tensors: dict[str, torch.Tensor],
    num_blocks: int,
) -> dict[str, torch.Tensor]:
    """Swap slow KV backing buffers with faster zeroed spares and rank block_ids; never raises."""
    global _KV_BLOCK_RANKS
    _KV_BLOCK_RANKS = None
    if (not KV_ENABLED and not KV_BLOCK_SORT_ENABLED) or not torch.cuda.is_available() or not raw_tensors:
        return raw_tensors

    began = time.perf_counter()
    try:
        unique_by_ptr: dict[int, torch.Tensor] = {}
        for t in raw_tensors.values():
            if t.is_cuda:
                unique_by_ptr.setdefault(t.data_ptr(), t)
        unique_tensors = list(unique_by_ptr.values())
        if not unique_tensors:
            return raw_tensors
        total_bytes = sum(t.numel() * t.element_size() for t in unique_tensors)
        if total_bytes < MIN_PAYLOAD_BYTES:
            return raw_tensors

        evict_buf = torch.empty(
            (_L2_FLUSH_INTS,), dtype=torch.int32, device=unique_tensors[0].device
        )
        replacements: dict[int, torch.Tensor] = {}
        moved_total = 0

        if KV_ENABLED and KV_SURPLUS > 0:
            groups: dict[tuple, list[torch.Tensor]] = {}
            for t in unique_tensors:
                groups.setdefault((t.numel(), t.dtype, t.device), []).append(t)

            for (numel, dtype, device), homes in groups.items():
                payload_bytes = numel * homes[0].element_size()
                batch_sz = min(len(homes), 13)
                for offset in range(0, len(homes), batch_sz):
                    batch = homes[offset : offset + batch_sz]
                    wanted = (
                        max(2, math.ceil(KV_SURPLUS))
                        if len(batch) == 1
                        else math.ceil(len(batch) * KV_SURPLUS)
                    )
                    spares_count = _spares_allowed(wanted, payload_bytes)
                    if spares_count <= 0:
                        continue
                    spares: list[torch.Tensor] = []
                    try:
                        for _ in range(spares_count):
                            spares.append(torch.zeros_like(batch[0]))
                    except RuntimeError:
                        pass
                    if not spares:
                        continue
                    candidates = batch + spares
                    times = _time_calls(
                        _stream_read_sum,
                        candidates,
                        KV_ROUNDS,
                        evict_buf,
                        num_homes=len(batch),
                        screen_rounds=KV_ROUNDS,
                        refine_rounds=0,
                    )
                    slow_cut, free_fast, movers = _partition_candidates(times, len(batch))
                    used_targets = set()
                    for home_idx, target_idx in zip(movers, free_fast):
                        source = batch[home_idx]
                        target = candidates[target_idx]
                        if target.shape != source.shape or target.dtype != source.dtype:
                            continue
                        replacements[id(source)] = target
                        used_targets.add(target_idx)
                        if times[home_idx] > slow_cut and _spares_allowed(1, payload_bytes) > 0:
                            _SLOW_HOLD.append(source)
                        moved_total += 1
                    for i, spare in enumerate(spares):
                        idx = len(batch) + i
                        if idx not in used_targets and times[idx] > slow_cut and _spares_allowed(1, payload_bytes) > 0:
                            _SLOW_HOLD.append(spare)
                    del spares
                    torch.cuda.empty_cache()

        if replacements:
            raw_tensors = {
                ln: replacements.get(id(t), t) for ln, t in raw_tensors.items()
            }

        final_by_ptr: dict[int, torch.Tensor] = {}
        for t in raw_tensors.values():
            if t.is_cuda:
                final_by_ptr.setdefault(t.data_ptr(), t)
        final_tensors = list(final_by_ptr.values())

        _KV_BLOCK_RANKS = _probe_kv_block_ranks(final_tensors, num_blocks, evict_buf)
        logger.info(
            "KV cache placement: %d tensors (%.0f MiB), %d swapped, sorted=%s, %.2f s",
            len(final_tensors),
            total_bytes / (1 << 20),
            moved_total,
            _KV_BLOCK_RANKS is not None,
            time.perf_counter() - began,
        )
    except Exception as err:  # noqa: BLE001 - placement never blocks KV init
        logger.warning("KV cache placement skipped: %s", err)
    return raw_tensors


# --------------------------------------------------------------------------- #
# Prompt-sized (prefill) weights
# --------------------------------------------------------------------------- #
PREFILL_ENABLED = os.environ.get("MUSE_PREFILL_PLACEMENT", "1").strip() not in ("0", "", "false", "no")
PREFILL_SURPLUS = float(os.environ.get("MUSE_PREFILL_PLACEMENT_SURPLUS", "2.0"))
PREFILL_ROUNDS = 4
PREFILL_SCREEN_ROUNDS = max(1, int(os.environ.get("MUSE_PREFILL_SCREEN_ROUNDS", "2")))
PREFILL_REFINE_ROUNDS = max(0, int(os.environ.get("MUSE_PREFILL_REFINE_ROUNDS", "4")))


def _place_prefill_group(
    layers, run, on_move=None, evict_buf=None, rejected_hold=None, buf_name: str | None = None
) -> dict | None:
    homes = [layer._buffers[buf_name] if buf_name else layer.weight.data for layer in layers]
    count = len(homes)
    payload_bytes = homes[0].numel() * homes[0].element_size()
    wanted = math.ceil(count * PREFILL_SURPLUS) if PREFILL_SURPLUS > 0 else 0
    spares_count = _ensure_spares_room(wanted, payload_bytes, _SLOW_HOLD)
    col_major = homes[0].dim() == 2 and homes[0].stride() == (1, homes[0].shape[0])
    spares = []
    try:
        for _ in range(spares_count):
            if col_major:
                spare = torch.empty(
                    (homes[0].shape[1], homes[0].shape[0]),
                    dtype=homes[0].dtype,
                    device=homes[0].device,
                ).t()
            else:
                spare = torch.empty_like(homes[0])
            spare.copy_(homes[0])  # every page written once before it is timed
            spares.append(spare)
    except RuntimeError:  # out of device memory: time what exists
        pass
    if not spares:
        return None
    candidates = homes + spares
    times = _time_calls(
        run,
        candidates,
        PREFILL_ROUNDS,
        evict_buf,
        num_homes=count,
        screen_rounds=PREFILL_SCREEN_ROUNDS,
        refine_rounds=PREFILL_REFINE_ROUNDS,
        flush_each=False,
    )
    slow_cut, free_fast, movers = _partition_candidates(times, count)
    moved = 0
    after = list(times[:count])
    used_targets = set()
    for home_index, target_index in zip(movers, free_fast):
        target, source = candidates[target_index], homes[home_index]
        target.copy_(source)
        if not torch.equal(target, source):
            continue  # keep the loader's buffer
        if buf_name:
            layers[home_index]._buffers[buf_name] = target
        else:
            layers[home_index].weight.data = target
        if on_move is not None:
            on_move(source.data_ptr(), target.data_ptr())
        used_targets.add(target_index)
        if times[home_index] > slow_cut and _spares_allowed(1, payload_bytes) > 0:
            _SLOW_HOLD.append(source)
        elif rejected_hold is not None:
            rejected_hold.append(source)
        after[home_index] = times[target_index]
        moved += 1
    for i, spare in enumerate(spares):
        idx = count + i
        if idx not in used_targets:
            if times[idx] > slow_cut and _spares_allowed(1, payload_bytes) > 0:
                _SLOW_HOLD.append(spare)
            elif rejected_hold is not None:
                rejected_hold.append(spare)
    before = sorted(times[:count])
    after.sort()
    pool = sorted(times)
    return {
        "count": count, "spares": len(spares), "moved": moved, "payload_bytes": payload_bytes,
        "before": (before[0], median(before), before[-1], sum(before)),
        "after": (after[0], median(after), after[-1], sum(after)),
        "pool": (pool[0], median(pool), pool[-1]),
    }


def place_prefill_weights(model: torch.nn.Module) -> None:
    """Move 512-row NVFP4 & FP8 MLP weights that landed on slow device memory; never raises."""
    if not PREFILL_ENABLED or not torch.cuda.is_available():
        return
    began = time.perf_counter()
    rejected_hold: list[torch.Tensor] = []
    try:
        try:
            from marlinw import swiglu as _swiglu
        except (ImportError, OSError):
            _swiglu = None
        try:
            import marlinw_down as _down
        except (ImportError, OSError):
            _down = None
        try:
            from vllm.model_executor.kernels.linear.nvfp4 import marlin as _route
        except (ImportError, OSError):
            _route = None
        try:
            from vllm.model_executor.layers.quantization.utils import glimmer_fp8 as _gfp8
            import muse_fp8_prefill as _fp8pf
        except (ImportError, OSError):
            _gfp8 = None
            _fp8pf = None
        try:
            import muse_fp8_swiglu as _fp8sw
        except (ImportError, OSError):
            _fp8sw = None
        gate_layers, down_layers, fp8_down_layers, fp8_swiglu_layers = [], [], [], []
        for layer in model.modules():
            bufs = getattr(layer, "_buffers", {})
            pw = bufs.get("paired_fp8_swiglu_weight")
            if (
                _fp8sw is not None
                and os.environ.get("MUSE_PREFILL_FP8_SWIGLU", "1") != "0"
                and pw is not None
                and pw.is_cuda
                and pw.numel() * pw.element_size() >= MIN_PAYLOAD_BYTES
                and getattr(layer, "weight_scale", None) is not None
            ):
                fp8_swiglu_layers.append(layer)
            weight = getattr(layer, "weight", None)
            if not isinstance(weight, torch.nn.Parameter) or not weight.is_cuda:
                continue
            if (
                _gfp8 is not None
                and _fp8pf is not None
                and weight.dtype == torch.float8_e4m3fn
                and _gfp8.prefill_admitted(weight)
                and weight.numel() >= MIN_PAYLOAD_BYTES
                and tuple(weight.shape) == (19968, 6656)
            ):
                fp8_down_layers.append(layer)
                continue
            if (
                weight.dtype != torch.int32
                or not weight.is_contiguous()
                or weight.numel() * 4 < MIN_PAYLOAD_BYTES
                or getattr(layer, "weight_scale", None) is None
                or getattr(layer, "weight_global_scale", None) is None
            ):
                continue
            n = getattr(layer, "output_size_per_partition", 0)
            k = getattr(layer, "input_size_per_partition", 0)
            if tuple(weight.shape) != (k // 16, 2 * n):
                continue
            path = getattr(layer, "marlin_prompt_path", None)
            if _swiglu is not None and getattr(layer, "marlin_swiglu_ready", False) and (n, k) == (39936, 6656):
                gate_layers.append(layer)
            elif (
                _down is not None and path is not None and path[0] == "wide" and path[2][0] is not None
                and (n, k) == (_down.COLUMNS, _down.INNER)
            ):
                down_layers.append(layer)
        evict_buf = torch.empty((_L2_FLUSH_INTS,), dtype=torch.int32, device="cuda")
        saved_ms = 0.0
        groups = []
        if len(gate_layers) > 1:
            ref = gate_layers[0]
            x = ((torch.arange(512 * 6656, device=ref.weight.device) % 13).float() - 6.0).mul(0.11)
            x = x.to(torch.bfloat16).reshape(512, 6656).contiguous()

            fast = hasattr(_swiglu, "uses_128") and all(_swiglu.uses_128(layer.weight) for layer in gate_layers)
            if hasattr(_swiglu, "gemm_with"):
                def run_gate(weight, ref=ref, x=x, fast=fast):
                    _swiglu.gemm_with(x, weight, ref.weight_scale, ref.weight_global_scale, fast)
            else:
                def run_gate(weight, ref=ref, x=x):
                    _swiglu.gemm(x, weight, ref.weight_scale, ref.weight_global_scale)

            groups.append(("39936x6656", gate_layers, run_gate, getattr(_swiglu, "moved", None), fast, None))
        if len(down_layers) > 1:
            ref = down_layers[0]
            plan = ref.marlin_prompt_path[2]
            x = ((torch.arange(512 * _down.INNER, device=ref.weight.device) % 11).float() - 5.0).mul(0.07)
            x = x.to(torch.bfloat16).reshape(512, _down.INNER).contiguous()

            fast = _route is not None and all(_route.down128_admitted(layer.weight) for layer in down_layers)
            if fast:
                def run_down(weight, ref=ref, x=x):
                    if _route.marlinw_down128.gemm_auto(x, weight, ref.weight_scale, ref.weight_global_scale) is None:
                        raise RuntimeError("down128 kernel refused the probe input")
            else:
                def run_down(weight, ref=ref, x=x, plan=plan):
                    if _down.gemm(x, weight, ref.weight_scale, ref.weight_global_scale, *plan) is None:
                        raise RuntimeError("down kernel refused the probe input")

            groups.append((f"{_down.COLUMNS}x{_down.INNER}", down_layers, run_down,
                           _route.down128_moved if _route is not None else None, fast, None))
        if len(fp8_down_layers) > 1:
            ref = fp8_down_layers[0]
            dev = ref.weight.device
            x = torch.zeros((512, 19968), dtype=torch.float8_e4m3fn, device=dev)
            sa = ref.input_scale if getattr(ref, "input_scale", None) is not None else torch.ones((1,), dtype=torch.float32, device=dev)
            sb = ref.weight_scale
            cfg_dn = _fp8pf.config(6656, 19968) or 1

            def run_fp8_down(weight, x=x, sa=sa, sb=sb, cfg_dn=cfg_dn):
                _fp8pf.gemm(x, weight, sa, sb, cfg_dn)

            groups.append(("fp8_6656x19968", fp8_down_layers, run_fp8_down, _gfp8.prefill_moved, False, None))
        if len(fp8_swiglu_layers) > 1:
            ref = fp8_swiglu_layers[0]
            dev = ref._buffers["paired_fp8_swiglu_weight"].device
            x = torch.zeros((512, 6656), dtype=torch.float8_e4m3fn, device=dev)
            sa = ref.input_scale if getattr(ref, "input_scale", None) is not None else torch.ones((1,), dtype=torch.float32, device=dev)
            sb = ref.weight_scale.reshape(1)
            ds = torch.ones((1,), dtype=torch.float32, device=dev)

            def run_fp8_sw(weight, x=x, sa=sa, sb=sb, ds=ds):
                _fp8sw.apply(x, weight, sa, sb, ds)

            groups.append(("fp8_swiglu_39936x6656", fp8_swiglu_layers, run_fp8_sw,
                           None, False, "paired_fp8_swiglu_weight"))
        for name, layers, run, on_move, fast, buf_name in groups:
            try:
                stats = _place_prefill_group(layers, run, on_move, evict_buf, rejected_hold, buf_name=buf_name)
            except Exception as err:  # noqa: BLE001 - this group stays where it was
                logger.warning("Prefill weight placement skipped for %s: %s", name, err)
                continue
            if stats is None:
                continue
            b, a, pool = stats["before"], stats["after"], stats["pool"]
            saved_ms += b[3] - a[3]
            logger.info(
                "Prefill weight placement %s: %d weights of %.0f MiB, %d spare buffers, %d moved; 512-row kernel%s ms "
                "min/median/max %.3f/%.3f/%.3f -> %.3f/%.3f/%.3f (all candidates %.3f/%.3f/%.3f), sum %.2f -> %.2f ms",
                name, stats["count"], stats["payload_bytes"] / (1 << 20), stats["spares"], stats["moved"],
                " (128-row tile)" if fast else "",
                b[0], b[1], b[2], a[0], a[1], a[2], pool[0], pool[1], pool[2], b[3], a[3],
            )
        logger.info(
            "Prefill weight placement: %d groups, timed 512-row kernel sum reduced by %.2f ms per prefill, %.2f s",
            len(groups), saved_ms, time.perf_counter() - began,
        )
    except Exception as err:  # noqa: BLE001 - the load must go on
        logger.warning("Prefill weight placement failed and was skipped: %s", err)
    finally:
        # Keep _SLOW_HOLD quarantined until gpu_worker finishes initialize_kv_cache.
        rejected_hold.clear()
        torch.cuda.empty_cache()


def place_decode_weights(model: torch.nn.Module) -> None:
    """Move decode weights and block scales that landed on slow device memory; never raises."""
    global _KV_BLOCK_RANKS
    if not ENABLED or not torch.cuda.is_available():
        return
    try:
        from muse_marlin_gemv import decode_linear
    except (ImportError, OSError):
        return
    try:
        import l2pf as _l2pf
        stream_read_fn = _l2pf.stream_read if (_l2pf.lib() is not None and hasattr(_l2pf, "stream_read")) else None
    except Exception:  # noqa: BLE001
        stream_read_fn = None
    began = time.perf_counter()
    _SLOW_HOLD.clear()
    _KV_BLOCK_RANKS = None
    excess_hold: list[torch.Tensor] = []
    try:
        _drop_model_page_cache()
        evict_buf = torch.empty((_L2_FLUSH_INTS,), dtype=torch.int32, device="cuda")
        sink = torch.zeros((1,), dtype=torch.int32, device="cuda") if stream_read_fn is not None else None
        groups: dict[tuple, list] = {}
        rans_groups: dict[tuple[int, int], list] = {}
        scale_groups: dict[tuple, list] = {}
        for layer in model.modules():
            bufs = getattr(layer, "_buffers", {})
            weight = bufs.get(_NAME)
            if (
                weight is not None
                and weight.is_cuda
                and weight.is_contiguous()
                and weight.numel() * weight.element_size() >= DECODE_MIN_BYTES
                and getattr(layer, "_marlin_decode_scales", None) is not None
                and getattr(layer, "weight_global_scale", None) is not None
                and getattr(layer, "workspace", None) is not None
            ):
                n, k = layer.output_size_per_partition, layer.input_size_per_partition
                groups.setdefault((n, k, tuple(weight.shape), weight.dtype, weight.device), []).append(layer)
            scales6 = bufs.get("_marlin_decode_scales6")
            palette = bufs.get("_marlin_decode_palette")
            if scales6 is not None and palette is not None:
                scale_name, scale_buf = "_marlin_decode_scales6", scales6
            else:
                scale_name, scale_buf = "_marlin_decode_scales", bufs.get("_marlin_decode_scales")
            if (
                stream_read_fn is not None
                and scale_buf is not None
                and scale_buf.is_cuda
                and scale_buf.is_contiguous()
                and scale_buf.data_ptr() % 16 == 0
                and scale_buf.numel() * scale_buf.element_size() >= max(16, SCALE_MIN_BYTES)
            ):
                n = getattr(layer, "output_size_per_partition", 0)
                k = getattr(layer, "input_size_per_partition", 0)
                if n > 0 and k > 0:
                    scale_groups.setdefault((n, k, scale_name, scale_buf.dtype, scale_buf.device), []).append(layer)
            rblob = bufs.get("muse_fp8_packed")
            if (
                rblob is not None
                and rblob.is_cuda
                and rblob.dtype == torch.uint8
                and rblob.dim() == 1
                and rblob.is_contiguous()
                and rblob.numel() >= DECODE_MIN_BYTES
                and getattr(layer, "weight_scale", None) is not None
            ):
                n = getattr(layer, "output_size_per_partition", 0)
                k = getattr(layer, "input_size_per_partition", 0)
                if n > 0 and k > 0:
                    rans_groups.setdefault((n, k), []).append(layer)
        _rans_ext = None
        if rans_groups:
            try:
                import muse_rans_ext as _rans_ext
            except (ImportError, OSError):
                _rans_ext = None

        # Place largest total-payload groups first across weights, rANS weights, and scales
        # while quarantining rejected slow buffers in _SLOW_HOLD.
        tasks: list[tuple[int, str, tuple, list]] = []
        for key, layers in groups.items():
            total_b = len(layers) * layers[0]._buffers[_NAME].numel() * layers[0]._buffers[_NAME].element_size()
            tasks.append((total_b, "nvfp4", key, layers))
        for key, layers in rans_groups.items():
            total_b = sum(layer._buffers["muse_fp8_packed"].numel() for layer in layers)
            tasks.append((total_b, "rans", key, layers))
        for key, layers in scale_groups.items():
            scale_name = key[2]
            total_b = sum(
                layer._buffers[scale_name].numel() * layer._buffers[scale_name].element_size()
                for layer in layers
            )
            tasks.append((total_b, "scale", key, layers))
        tasks.sort(key=lambda item: -item[0])

        saved_ms = 0.0
        n_groups = 0
        for _, kind, key, layers in tasks:
            n, k = key[0], key[1]
            try:
                if kind == "nvfp4":
                    stats = _place_group(decode_linear, layers, n, k, evict_buf, excess_hold, stream_read_fn, sink)
                elif kind == "rans":
                    if _rans_ext is None and (
                        stream_read_fn is None or max(l._buffers["muse_fp8_packed"].numel() for l in layers) >= MIN_PAYLOAD_BYTES
                    ):
                        continue
                    stats = _place_muserans_group(_rans_ext, layers, n, k, evict_buf, excess_hold, stream_read_fn, sink)
                else:
                    stats = _place_scale_group(stream_read_fn, sink, layers, key[2], evict_buf, excess_hold)
            except Exception as err:  # noqa: BLE001 - this group stays where it was
                logger.warning("Decode %s placement skipped for %dx%d: %s", kind, n, k, err)
                continue
            if stats is None:
                continue
            n_groups += 1
            b, a = stats["before"], stats["after"]
            saved_ms += b[3] - a[3]
            gbps = stats["payload_bytes"] / 1e6
            if kind == "nvfp4":
                logger.info(
                    "Decode weight placement %dx%d: %d weights of %.1f MiB, %d spare buffers, %d moved; "
                    "ms min/median/max %.4f/%.4f/%.4f -> %.4f/%.4f/%.4f "
                    "(weight-only %.0f/%.0f/%.0f -> %.0f/%.0f/%.0f GB/s), sum %.3f -> %.3f ms",
                    n, k, stats["count"], stats["payload_bytes"] / (1 << 20), stats["spares"], stats["moved"],
                    b[0], b[1], b[2], a[0], a[1], a[2],
                    gbps / b[0], gbps / b[1], gbps / b[2], gbps / a[0], gbps / a[1], gbps / a[2], b[3], a[3],
                )
            elif kind == "rans":
                logger.info(
                    "Decode rANS placement %dx%d: %d weights of ~%.1f MiB, %d spare buffers, %d moved; "
                    "ms min/median/max %.4f/%.4f/%.4f -> %.4f/%.4f/%.4f, sum %.3f -> %.3f ms",
                    n, k, stats["count"], stats["payload_bytes"] / (1 << 20), stats["spares"], stats["moved"],
                    b[0], b[1], b[2], a[0], a[1], a[2], b[3], a[3],
                )
            else:
                logger.info(
                    "Decode scale (%s) placement %dx%d: %d tensors of ~%.1f MiB, %d spare buffers, %d moved; "
                    "read ms min/median/max %.4f/%.4f/%.4f -> %.4f/%.4f/%.4f "
                    "(%.0f/%.0f/%.0f -> %.0f/%.0f/%.0f GB/s), sum %.3f -> %.3f ms",
                    key[2], n, k, stats["count"], stats["payload_bytes"] / (1 << 20), stats["spares"], stats["moved"],
                    b[0], b[1], b[2], a[0], a[1], a[2],
                    gbps / b[0], gbps / b[1], gbps / b[2], gbps / a[0], gbps / a[1], gbps / a[2], b[3], a[3],
                )
        logger.info(
            "Decode weight placement: %d groups, timed sum reduced by %.3f ms per token, %.2f s",
            n_groups, saved_ms, time.perf_counter() - began,
        )
    except Exception as err:  # noqa: BLE001 - the load must go on
        logger.warning("Decode weight placement failed and was skipped: %s", err)
    finally:
        # Keep _SLOW_HOLD quarantined through initialize_kv_cache (even if prefill is off).
        excess_hold.clear()
        torch.cuda.empty_cache()
