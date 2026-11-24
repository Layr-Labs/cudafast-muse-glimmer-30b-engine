# SPDX-License-Identifier: Apache-2.0
"""Where the one-row decode and 512-row prefill weights live in device memory.

On a unified-memory GPU (GB10), streaming GEMV bandwidth and 512-row L2 reuse
depend on the physical DRAM pages backing a weight tensor. After loading,
we drop clean checkpoint page cache via posix_fadvise(DONTNEED), allocate
spare buffers, flush L2 with a read-only pass (avoiding dirty-line write
penalties and L2-warm spare bias), hold rejected slow buffers across groups
so later groups do not recycle slow pages, and exchange slow homes with faster
spares after a bitwise equality check.

  MUSE_DECODE_PLACEMENT=0             no decode placement step
  MUSE_DECODE_PLACEMENT_SURPLUS=<r>   spare buffers per decode weight (default 2.0)
  MUSE_PREFILL_PLACEMENT=0            no prefill placement step
  MUSE_PREFILL_PLACEMENT_SURPLUS=<r>  spare buffers per prefill weight (default 1.5)
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
FULL_RAM_SCAN = os.environ.get("MUSE_FULL_RAM_SCAN", "1").strip() not in ("0", "", "false", "no")
DIRECT_FAST_PACK = os.environ.get("MUSE_DIRECT_FAST_PACK", "1").strip() not in ("0", "", "false", "no")
PLACE_SCALES6 = os.environ.get("MUSE_PLACE_SCALES6", "1").strip() not in ("0", "", "false", "no")
SURPLUS = float(os.environ.get("MUSE_DECODE_PLACEMENT_SURPLUS", "3.0"))
LARGE_RANS_SURPLUS = float(os.environ.get("MUSE_LARGE_RANS_SURPLUS", "5.0"))
LM_HEAD_SPARES = int(os.environ.get("MUSE_LM_HEAD_SPARES", "32"))
ROUNDS = 6
SCREEN_ROUNDS = max(1, int(os.environ.get("MUSE_PLACEMENT_SCREEN_ROUNDS", "2")))
REFINE_ROUNDS = max(0, int(os.environ.get("MUSE_PLACEMENT_REFINE_ROUNDS", "4")))
FULL_SCAN_SCREEN_ROUNDS = max(1, int(os.environ.get("MUSE_FULL_SCAN_SCREEN_ROUNDS", "2")))
FULL_SCAN_REFINE_ROUNDS = max(0, int(os.environ.get("MUSE_FULL_SCAN_REFINE_ROUNDS", "4")))
# Weights that fit in L2 several times over are not bandwidth-bound per call.
MIN_PAYLOAD_BYTES = 48 << 20
# Host memory that must stay available while the spares exist.
HOST_HEADROOM_BYTES = 20 << 30
KV_HEADROOM_BYTES = 16 << 30
_NAME = "_marlin_decode_weight"
_L2_FLUSH_INTS = 16 << 20  # 64 MiB (> 2.5x GB10's 24 MiB L2)
# Quarantine below-median (slow) buffers from place_decode_weights through initialize_kv_cache.
_SLOW_HOLD: list[torch.Tensor] = []
PLACE_ATTN = os.environ.get("MUSE_PLACE_ATTN", "1").strip() not in ("0", "", "false", "no")
PREFILL_PLACE_ATTN = os.environ.get("MUSE_PREFILL_PLACE_ATTN", "1").strip() not in ("0", "", "false", "no")
ATTN_MIN_BYTES = 10 << 20
ATTN_SURPLUS = float(os.environ.get("MUSE_ATTN_PLACEMENT_SURPLUS", "2.5"))

KV_ENABLED = os.environ.get("MUSE_KV_PLACEMENT", "1").strip() not in ("0", "", "false", "no")
KV_BLOCK_SORT_ENABLED = os.environ.get("MUSE_KV_BLOCK_SORT", "1").strip() not in ("0", "", "false", "no")
KV_SURPLUS = float(os.environ.get("MUSE_KV_PLACEMENT_SURPLUS", "2.0"))
KV_ROUNDS = 3
KV_WINDOW_BYTES = 32 << 20
_KV_BLOCK_RANKS: list[float] | None = None


def drop_safetensors_shard(path: str | Path) -> bool:
    """Drop clean Linux page cache for a single .safetensors file via posix_fadvise(DONTNEED)."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            return True
        finally:
            os.close(fd)
    except OSError:
        return False


def _resolve_model_dir(model_dir: str | None = None) -> Path | None:
    cand = model_dir or os.environ.get("RESIDENT_MODEL_DIR", "")
    if not cand:
        try:
            from vllm.config import get_current_vllm_config

            cfg = get_current_vllm_config()
            cand = getattr(getattr(cfg, "model_config", None), "model", "") or ""
        except Exception:
            cand = ""
    if not cand:
        return None
    root = Path(cand)
    return root if root.is_dir() else None


def _drop_model_page_cache(model_dir: str | None = None) -> None:
    """Release clean checkpoint file pages back to Linux free memory on unified-RAM hosts."""
    root = _resolve_model_dir(model_dir)
    if root is None:
        return
    dropped = 0
    for path in root.glob("*.safetensors"):
        if drop_safetensors_shard(path):
            dropped += 1
    if dropped:
        logger.info("Released page cache for %d safetensors shards via posix_fadvise(DONTNEED)", dropped)


def _flush_l2(evict_buf: torch.Tensor) -> None:
    """Read-only L2 eviction pass (never writes, avoiding dirty-line writeback overhead)."""
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
    """Spare buffers the host can carry right now, at most ``wanted``.
    Only count PyTorch's cached-free 127 MiB fast pool for allocations <= 128 MiB that can actually reuse it."""
    available = _mem_available()
    if available is None:
        return 0
    if payload_bytes <= (128 << 20):
        cached_free = max(0, torch.cuda.memory_reserved() - torch.cuda.memory_allocated())
        effective = available + cached_free
    else:
        effective = available
    return max(0, min(wanted, (effective - HOST_HEADROOM_BYTES) // payload_bytes))


def _ensure_spares_room(wanted: int, payload_bytes: int, hold_list: list, allow_slow_trim: bool = True) -> int:
    """Release oldest fast spares back to PyTorch's cache first; only trim _SLOW_HOLD + empty_cache if still short."""
    spares_count = _spares_allowed(wanted, payload_bytes)
    if spares_count >= wanted:
        return spares_count
    need_flush = payload_bytes > (128 << 20)
    while hold_list and _spares_allowed(wanted, payload_bytes) < wanted:
        del hold_list[: max(1, len(hold_list) // 4)]
        if need_flush:
            torch.cuda.empty_cache()
    spares_count = _spares_allowed(wanted, payload_bytes)
    if need_flush and spares_count < wanted:
        torch.cuda.empty_cache()
        spares_count = _spares_allowed(wanted, payload_bytes)
    if allow_slow_trim and spares_count < wanted and _SLOW_HOLD:
        while _SLOW_HOLD and _spares_allowed(wanted, payload_bytes) < wanted:
            del _SLOW_HOLD[: max(1, len(_SLOW_HOLD) // 4)]
            torch.cuda.empty_cache()
    return _spares_allowed(wanted, payload_bytes)


def ensure_kv_cache_room(required_bytes: int) -> None:
    """Release oldest quarantined slow buffers only if host RAM lacks room for KV cache."""
    if not _SLOW_HOLD or required_bytes <= 0:
        return
    needed = max(1, required_bytes)
    while _SLOW_HOLD:
        available = _mem_available()
        if available is None or available - KV_HEADROOM_BYTES >= needed:
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


def get_kv_block_ranks(num_blocks: int | None = None) -> list[float] | None:
    """Return per-block read-latency ranks measured during KV cache allocation."""
    ranks = _KV_BLOCK_RANKS
    if ranks is None:
        return None
    if num_blocks is not None and len(ranks) != num_blocks:
        return None
    return ranks


def _halving_keep_count(total: int, num_homes: int | None) -> int:
    count = max(1, num_homes) if num_homes is not None else 1
    spares = max(0, total - count)
    min_keep = count + min(spares, max(1, min(count, spares // 2)))
    return min(total, max(min_keep, math.ceil(0.35 * total)))


def _partition_candidates(times: list[float], count: int) -> tuple[float, list[int], list[int]]:
    """Split candidates into slow threshold, above-count fast spares, and below-count movers."""
    slow_cut = median(times)
    ranked = sorted(range(len(times)), key=times.__getitem__)
    kept = set(ranked[:count])
    free_fast = [index for index in ranked[:count] if index >= count]
    movers = sorted((index for index in range(count) if index not in kept), key=times.__getitem__, reverse=True)
    return slow_cut, free_fast, movers


def _time_candidates(
    decode_linear,
    layer,
    candidates,
    n: int,
    k: int,
    rounds: int,
    evict_buf: torch.Tensor,
    *,
    num_homes: int | None = None,
    screen_rounds: int | None = None,
    refine_rounds: int | None = None,
) -> list[float]:
    """Median GEMV time (ms) of the tiled decode kernel on each candidate buffer."""
    device = candidates[0].device
    # A fixed finite row; the kernel's timing does not depend on values.
    x = ((torch.arange(k, device=device) % 7).float() - 3.0).to(torch.bfloat16).reshape(1, k)
    global_scale, workspace = layer.weight_global_scale, layer.workspace
    palette = getattr(layer, "_marlin_decode_palette", None)
    scales6 = getattr(layer, "_marlin_decode_scales6", None)
    fixed_grid = bool(getattr(layer, "marlin_decode6_fixed_grid", False))
    use_6 = False
    if palette is not None and scales6 is not None:
        try:
            from muse_marlin6_gemv import decode_linear6
            use_6 = True
        except (ImportError, OSError):
            use_6 = False
    if use_6:
        def run_one(buf):
            decode_linear6(x, buf, scales6, palette, global_scale, workspace, n, k, fixed_grid)
    else:
        scales = layer._marlin_decode_scales
        def run_one(buf):
            decode_linear(x, buf, scales, global_scale, workspace, n, k)
    return _time_calls(
        run_one,
        candidates,
        rounds,
        evict_buf,
        num_homes=num_homes,
        screen_rounds=screen_rounds,
        refine_rounds=refine_rounds,
    )


class _FastSlabPacker:
    """Bump-allocates 4 KiB-aligned contiguous slices from ranked 127 MiB fast DRAM slabs, fastest first."""

    def __init__(self, slabs: list[torch.Tensor]) -> None:
        self._slabs = [s.reshape(-1).view(torch.uint8) for s in slabs]
        self._idx = 0
        self._off = 0
        self.used_slabs = 0

    def alloc_u8(
        self, nbytes: int, align: int = 4096, *, cap_bytes: int | None = None
    ) -> torch.Tensor | None:
        if nbytes <= 0:
            return None
        cap = nbytes if cap_bytes is None else max(nbytes, cap_bytes)
        while self._idx < len(self._slabs):
            slab = self._slabs[self._idx]
            pad = (-(slab.data_ptr() + self._off)) & (align - 1) if align > 1 else 0
            off = self._off + pad
            if off + cap <= slab.numel():
                self._off = off + cap
                self.used_slabs = self._idx + 1
                sub = torch.empty((0,), dtype=torch.uint8, device=slab.device).set_(
                    slab.untyped_storage(), slab.storage_offset() + off, (cap,), (1,)
                )
                return sub if cap == nbytes else sub[:nbytes]
            self._idx += 1
            self._off = 0
        return None

    def alloc_like(
        self, src: torch.Tensor, align: int = 4096, *, cap_bytes: int | None = None
    ) -> torch.Tensor | None:
        nbytes = src.numel() * src.element_size()
        raw = self.alloc_u8(nbytes, align=align, cap_bytes=cap_bytes)
        if raw is None:
            return None
        if src.dtype == torch.uint8:
            return raw.view(src.shape)
        return raw.view(src.dtype).view(src.shape)

    def release_unused(self) -> None:
        if self.used_slabs < len(self._slabs):
            del self._slabs[self.used_slabs :]


def _relocate_scales6_ptr(m6_mod, old_ptr: int, new_ptr: int) -> None:
    if m6_mod is None or old_ptr == new_ptr:
        return
    if hasattr(m6_mod, "relocate_scale_ptr"):
        m6_mod.relocate_scale_ptr(old_ptr, new_ptr)
    for attr in (
        "_DOWN_ILV_SCALES",
        "_LM_SPLIT2_SCALES",
        "_GATE_UP_SPLIT2_SCALES",
        "_DOWN_SPLIT2_SCALES",
    ):
        pset = getattr(m6_mod, attr, None)
        if pset is not None and old_ptr in pset:
            pset.discard(old_ptr)
            pset.add(new_ptr)


def _place_group_full_ram_scan(decode_linear, layers, n: int, k: int, evict_buf: torch.Tensor) -> dict | None:
    """Scan available host RAM in 127 MiB blocks (~4s), place gate_up into the fastest 45,
    return the next fastest ranked slabs in exact speed order for direct fast-slab packing,
    and quarantine the slowest slabs in _SLOW_HOLD."""
    homes = [layer._buffers[_NAME] for layer in layers]
    count = len(homes)
    payload_bytes = homes[0].numel() * homes[0].element_size()
    spares_count = _spares_allowed(600, payload_bytes)
    if spares_count < count * 3:
        return None
    spares = []
    try:
        for _ in range(spares_count):
            spare = torch.empty_like(homes[0])
            spare.copy_(homes[0])
            spares.append(spare)
    except RuntimeError:
        pass
    num_spares = len(spares)
    if num_spares < count * 3:
        spares.clear()
        return None
    candidates = homes + spares
    times = _time_candidates(
        decode_linear,
        layers[0],
        candidates,
        n,
        k,
        FULL_SCAN_SCREEN_ROUNDS + FULL_SCAN_REFINE_ROUNDS,
        evict_buf,
        num_homes=count,
        screen_rounds=FULL_SCAN_SCREEN_ROUNDS,
        refine_rounds=FULL_SCAN_REFINE_ROUNDS,
    )
    ranked = sorted(range(len(candidates)), key=times.__getitem__)
    kept = set(ranked[:count])
    free_fast = [index for index in ranked[:count] if index >= count]
    movers = sorted((index for index in range(count) if index not in kept), key=times.__getitem__, reverse=True)
    moved = 0
    after = list(times[:count])
    used_targets = set()
    displaced_homes = []
    for home_index, target_index in zip(movers, free_fast):
        target, source = candidates[target_index], homes[home_index]
        target.copy_(source)
        if not torch.equal(target, source):
            continue
        layers[home_index]._buffers[_NAME] = target
        used_targets.add(target_index)
        displaced_homes.append((times[home_index], source))
        after[home_index] = times[target_index]
        moved += 1
    unused = [(times[count + i], spare) for i, spare in enumerate(spares) if (count + i) not in used_targets]
    unused.extend(displaced_homes)
    spares.clear()
    candidates.clear()
    homes.clear()
    displaced_homes.clear()
    spare = target = source = None
    unused.sort(key=lambda pair: pair[0])
    fast_keep = min(len(unused), max(140 if DIRECT_FAST_PACK else 180, int(len(unused) * 0.42)))
    fast_hold = [buf for _, buf in unused[:fast_keep]]
    mid_and_slow = unused[fast_keep:]
    slow_tail = mid_and_slow[-140:] if len(mid_and_slow) > 140 else mid_and_slow
    _SLOW_HOLD.extend(slow_buf for _, slow_buf in slow_tail)
    del unused, mid_and_slow, slow_tail
    if DIRECT_FAST_PACK:
        fast_slabs = fast_hold
    else:
        torch.cuda.empty_cache()
        del fast_hold
        fast_slabs = []
    before = sorted(times[:count])
    after.sort()
    return {
        "count": count, "spares": num_spares, "moved": moved, "payload_bytes": payload_bytes,
        "before": (before[0], median(before), before[-1], sum(before)),
        "after": (after[0], median(after), after[-1], sum(after)),
        "fast_slabs": fast_slabs,
    }


def _place_group(decode_linear, layers, n: int, k: int, evict_buf: torch.Tensor, rejected_hold: list) -> dict | None:
    homes = [layer._buffers[_NAME] for layer in layers]
    count = len(homes)
    payload_bytes = homes[0].numel() * homes[0].element_size()
    is_small = payload_bytes < MIN_PAYLOAD_BYTES
    surplus = ATTN_SURPLUS if is_small else SURPLUS
    wanted = max(LM_HEAD_SPARES if count == 1 else 1, math.ceil(count * surplus)) if surplus > 0 else 0
    spares_count = _ensure_spares_room(wanted, payload_bytes, rejected_hold, allow_slow_trim=not is_small)
    if spares_count <= 0 or (is_small and spares_count < count):
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
    times = _time_candidates(
        decode_linear,
        layers[0],
        candidates,
        n,
        k,
        ROUNDS,
        evict_buf,
        num_homes=count,
        screen_rounds=SCREEN_ROUNDS,
        refine_rounds=REFINE_ROUNDS,
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
        layers[home_index]._buffers[_NAME] = target
        used_targets.add(target_index)
        rejected_hold.append(source)
        if times[home_index] > slow_cut:
            _SLOW_HOLD.append(source)
        after[home_index] = times[target_index]
        moved += 1
    for i, spare in enumerate(spares):
        idx = count + i
        if idx not in used_targets:
            rejected_hold.append(spare)
            if times[idx] > slow_cut and len(_SLOW_HOLD) < 100:
                _SLOW_HOLD.append(spare)
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
    rejected_hold: list,
    surplus_override: float | None = None,
) -> dict | None:
    homes = [layer._buffers["muse_fp8_packed"] for layer in layers]
    count = len(homes)
    max_bytes = max(h.numel() for h in homes)
    is_small = max_bytes < MIN_PAYLOAD_BYTES
    surplus = surplus_override if surplus_override is not None else (ATTN_SURPLUS if is_small else SURPLUS)
    wanted = max(1, math.ceil(count * surplus)) if surplus > 0 else 0
    spares_count = _ensure_spares_room(wanted, max_bytes, rejected_hold, allow_slow_trim=not is_small)
    if spares_count <= 0 or (is_small and spares_count < count):
        return None
    # Time every home and spare with the EXACT same smallest blob so per-layer compression ratio
    # differences do not confound physical-page speed measurements. Back up real blobs to host first.
    min_idx = min(range(count), key=lambda i: homes[i].numel())
    ref_blob = homes[min_idx].clone()
    ref_len = ref_blob.numel()
    saved_host = [h.detach().cpu() for h in homes]
    spares = []
    try:
        for _ in range(spares_count):
            spare = torch.empty((max_bytes,), dtype=torch.uint8, device=homes[0].device)
            spare[:ref_len].copy_(ref_blob)
            spares.append(spare)
    except RuntimeError:
        pass
    if not spares:
        return None
    for h in homes:
        h[:ref_len].copy_(ref_blob)
    ref = layers[0]
    device = homes[0].device
    xq = torch.zeros((k,), dtype=torch.float8_e4m3fn, device=device)
    ws = ref.weight_scale.reshape(-1)
    xs = ref.input_scale.reshape(-1) if getattr(ref, "input_scale", None) is not None else torch.ones((1,), dtype=torch.float32, device=device)
    out = torch.empty((1, n), dtype=torch.bfloat16, device=device)

    def run_cand(buf):
        rans_ext.rans_gemv_auto(buf, xq, ws, xs, out, n, k)

    cand_views = [h[:ref_len] for h in homes] + [s[:ref_len] for s in spares]
    try:
        times = _time_calls(
            run_cand,
            cand_views,
            ROUNDS,
            evict_buf,
            num_homes=count,
            screen_rounds=SCREEN_ROUNDS,
            refine_rounds=REFINE_ROUNDS,
        )
    except Exception:
        for h, sh in zip(homes, saved_host):
            h.copy_(sh, non_blocking=False)
        raise
    ranked = sorted(range(len(cand_views)), key=times.__getitem__)
    kept = set(ranked[:count])
    free_fast = [index for index in ranked[:count] if index >= count]
    movers = sorted((index for index in range(count) if index not in kept), key=times.__getitem__, reverse=True)
    # Restore every kept home's real bytes first.
    for idx in kept:
        if idx < count:
            homes[idx].copy_(saved_host[idx], non_blocking=False)
    moved = 0
    after = list(times[:count])
    used_targets = set()
    slow_cut = median(times)
    for home_index, target_index in zip(movers, free_fast):
        source = homes[home_index]
        real_dev = saved_host[home_index].to(device=device)
        target_full = spares[target_index - count]
        target = target_full[: source.numel()]
        target.copy_(real_dev)
        if not torch.equal(target, real_dev):
            source.copy_(real_dev)
            continue
        # integration3: swap the storage of the home tensor object itself. The FP8 linear kernel keeps its own
        # reference to this object (cutlass.py self.muse_packed), so replacing only the layer buffer entry
        # would leave the kernel reading the old blob and the move would have no effect.
        old = source.data
        source.data = target
        if source.data_ptr() != target.data_ptr():
            source.data = old
            source.copy_(real_dev)
            continue
        used_targets.add(target_index)
        rejected_hold.append(old)
        if times[home_index] > slow_cut:
            _SLOW_HOLD.append(old)
        after[home_index] = times[target_index]
        moved += 1
    for i, spare in enumerate(spares):
        idx = count + i
        if idx not in used_targets:
            rejected_hold.append(spare)
            if times[idx] > slow_cut and len(_SLOW_HOLD) < 100:
                _SLOW_HOLD.append(spare)
    before = sorted(times[:count])
    after.sort()
    return {
        "count": count, "spares": len(spares), "moved": moved, "payload_bytes": max_bytes,
        "before": (before[0], median(before), before[-1], sum(before)),
        "after": (after[0], median(after), after[-1], sum(after)),
    }


def _place_bf16_rans_group(bf16_lib, layers, n: int, k: int, evict_buf: torch.Tensor, rejected_hold: list) -> dict | None:
    homes = [layer._buffers["muse_bf16_lane_weight"] for layer in layers]
    count = len(homes)
    max_bytes = max(h.numel() for h in homes)
    wanted = max(1, math.ceil(count * ATTN_SURPLUS)) if ATTN_SURPLUS > 0 else 0
    spares_count = _ensure_spares_room(wanted, max_bytes, rejected_hold, allow_slow_trim=False)
    if spares_count < count:
        return None
    min_idx = min(range(count), key=lambda i: homes[i].numel())
    ref_blob = homes[min_idx].clone()
    ref_len = ref_blob.numel()
    saved_host = [h.detach().cpu() for h in homes]
    spares = []
    try:
        for _ in range(spares_count):
            spare = torch.empty((max_bytes,), dtype=torch.uint8, device=homes[0].device)
            spare[:ref_len].copy_(ref_blob)
            spares.append(spare)
    except RuntimeError:
        pass
    if not spares:
        return None
    for h in homes:
        h[:ref_len].copy_(ref_blob)
    device = homes[0].device
    x = torch.zeros((1, k), dtype=torch.bfloat16, device=device)
    out = torch.empty((1, n), dtype=torch.bfloat16, device=device)

    def run_cand(buf):
        stream = torch.cuda.current_stream(device).cuda_stream
        rc = bf16_lib.muse_bf16_rans_gemv(n, ref_len, x.data_ptr(), buf.data_ptr(), out.data_ptr(), stream)
        if rc:
            raise RuntimeError(f"muse_bf16_rans_gemv returned {rc}")

    cand_views = [h[:ref_len] for h in homes] + [s[:ref_len] for s in spares]
    try:
        with torch.cuda.device(device):
            times = _time_calls(
                run_cand,
                cand_views,
                ROUNDS,
                evict_buf,
                num_homes=count,
                screen_rounds=SCREEN_ROUNDS,
                refine_rounds=REFINE_ROUNDS,
            )
    except Exception:
        for h, sh in zip(homes, saved_host):
            h.copy_(sh, non_blocking=False)
        raise
    ranked = sorted(range(len(cand_views)), key=times.__getitem__)
    kept = set(ranked[:count])
    free_fast = [index for index in ranked[:count] if index >= count]
    movers = sorted((index for index in range(count) if index not in kept), key=times.__getitem__, reverse=True)
    for idx in kept:
        if idx < count:
            homes[idx].copy_(saved_host[idx], non_blocking=False)
    moved = 0
    after = list(times[:count])
    used_targets = set()
    slow_cut = median(times)
    for home_index, target_index in zip(movers, free_fast):
        source = homes[home_index]
        real_dev = saved_host[home_index].to(device=device)
        target_full = spares[target_index - count]
        target = target_full[: source.numel()]
        target.copy_(real_dev)
        if not torch.equal(target, real_dev):
            source.copy_(real_dev)
            continue
        old = source.data
        source.data = target
        if source.data_ptr() != target.data_ptr():
            source.data = old
            source.copy_(real_dev)
            continue
        used_targets.add(target_index)
        rejected_hold.append(old)
        if times[home_index] > slow_cut:
            _SLOW_HOLD.append(old)
        after[home_index] = times[target_index]
        moved += 1
    for i, spare in enumerate(spares):
        idx = count + i
        if idx not in used_targets:
            rejected_hold.append(spare)
            if times[idx] > slow_cut and len(_SLOW_HOLD) < 100:
                _SLOW_HOLD.append(spare)
    before = sorted(times[:count])
    after.sort()
    return {
        "count": count, "spares": len(spares), "moved": moved, "payload_bytes": max_bytes,
        "before": (before[0], median(before), before[-1], sum(before)),
        "after": (after[0], median(after), after[-1], sum(after)),
    }


# --------------------------------------------------------------------------- #
# KV cache placement & block_id latency ranking
# --------------------------------------------------------------------------- #
def _stream_read_sum(buf: torch.Tensor) -> torch.Tensor:
    if buf.is_contiguous() and buf.data_ptr() % 8 == 0 and (buf.numel() * buf.element_size()) % 8 == 0:
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
PREFILL_ROUNDS = 5
PREFILL_PASSES = int(os.environ.get("MUSE_PREFILL_PLACEMENT_PASSES", "1"))
DECODE_DOWN_PASSES = int(os.environ.get("MUSE_DECODE_DOWN_PASSES", "2"))


def _time_calls(
    run,
    candidates,
    rounds: int,
    evict_buf: torch.Tensor | None = None,
    *,
    num_homes: int | None = None,
    screen_rounds: int | None = None,
    refine_rounds: int | None = None,
) -> list[float]:
    """Median time (ms) of run(candidate) per candidate, with optional two-stage successive halving."""
    total = len(candidates)
    if total == 0:
        return []
    device = candidates[0].device
    payload_bytes = candidates[0].numel() * candidates[0].element_size()
    total_bytes = sum(c.numel() * c.element_size() for c in candidates)
    flush_per_cand = evict_buf is not None and payload_bytes < (32 << 20) and total_bytes < (48 << 20)
    flush_per_round = evict_buf is not None and payload_bytes < (32 << 20) and not flush_per_cand
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    run(candidates[0])
    if evict_buf is not None:
        _flush_l2(evict_buf)
    torch.cuda.synchronize(device)

    samples: list[list[float]] = [[] for _ in range(total)]
    shuffler = random.Random(20261122)

    if screen_rounds is not None and refine_rounds is not None and refine_rounds > 0:
        scr = max(1, screen_rounds)
        ref = max(0, refine_rounds)
    else:
        scr = max(1, rounds)
        ref = 0

    def _run_stage(indices: list[int], n_rounds: int, first_stage: bool) -> None:
        order = list(indices)
        stage_bytes = payload_bytes * len(order)
        stage_flush_cand = evict_buf is not None and payload_bytes < (32 << 20) and stage_bytes < (48 << 20)
        stage_flush_round = evict_buf is not None and payload_bytes < (32 << 20) and not stage_flush_cand
        for r_idx in range(n_rounds):
            if (not first_stage or r_idx > 0) and (flush_per_round or stage_flush_round):
                _flush_l2(evict_buf)
            shuffler.shuffle(order)
            for step_idx, index in enumerate(order):
                if (flush_per_cand or stage_flush_cand) and (not first_stage or r_idx > 0 or step_idx > 0):
                    _flush_l2(evict_buf)
                start.record()
                run(candidates[index])
                end.record()
                end.synchronize()
                samples[index].append(start.elapsed_time(end))

    _run_stage(list(range(total)), scr, True)
    if ref > 0:
        keep = _halving_keep_count(total, num_homes)
        if keep < total:
            stage1 = [median(values) for values in samples]
            finalists = sorted(range(total), key=stage1.__getitem__)[:keep]
        else:
            finalists = list(range(total))
        _run_stage(finalists, ref, False)
    return [median(values) for values in samples]


def _place_prefill_group(layers, run, on_move=None, evict_buf=None, rejected_hold=None,
                         buf_name: str | None = None) -> dict | None:
    homes = [layer._buffers[buf_name] if buf_name else layer.weight.data for layer in layers]
    count = len(homes)
    payload_bytes = homes[0].numel() * homes[0].element_size()
    wanted = math.ceil(count * PREFILL_SURPLUS) if PREFILL_SURPLUS > 0 else 0
    spares_count = _ensure_spares_room(
        wanted, payload_bytes, rejected_hold if rejected_hold is not None else [], allow_slow_trim=True
    )
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
    times = _time_calls(run, candidates, PREFILL_ROUNDS, evict_buf)
    ranked = sorted(range(len(candidates)), key=times.__getitem__)
    kept = set(ranked[:count])
    free_fast = [index for index in ranked[:count] if index >= count]
    movers = sorted((index for index in range(count) if index not in kept), key=times.__getitem__, reverse=True)
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
        if rejected_hold is not None:
            rejected_hold.append(source)
        after[home_index] = times[target_index]
        moved += 1
    if rejected_hold is not None:
        for i, spare in enumerate(spares):
            if (count + i) not in used_targets:
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
        try:
            import fp8ws as _fp8ws
        except (ImportError, OSError):
            _fp8ws = None
        try:
            import marlinw_attn as _marlin_attn
        except (ImportError, OSError):
            _marlin_attn = None
        gate_layers, down_layers, fp8_down_layers, fp8_swiglu_layers, bf16_attn_layers = [], [], [], [], []
        fp8_attn_o, fp8_attn_qkv, fp8_attn_gate = [], [], []
        nvfp4_attn_qkv, nvfp4_attn_gate, nvfp4_attn_o = [], [], []
        for layer in model.modules():
            bufs = getattr(layer, "_buffers", {})
            bf16_cat = bufs.get("_bf16_qkv_gate_w")
            if (
                bf16_cat is not None
                and bf16_cat.is_cuda
                and bf16_cat.dtype == torch.bfloat16
                and tuple(bf16_cat.shape) == (8704, 6656)
                and bf16_cat.numel() * bf16_cat.element_size() >= MIN_PAYLOAD_BYTES
            ):
                bf16_attn_layers.append(layer)
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
                and getattr(layer, "weight_scale", None) is not None
            ):
                wshape = tuple(weight.shape)
                if _gfp8.prefill_admitted(weight) and wshape == (19968, 6656):
                    fp8_down_layers.append(layer)
                    continue
                if PREFILL_PLACE_ATTN:
                    if _gfp8.prefill_admitted(weight) and wshape == (4096, 6656):
                        fp8_attn_o.append(layer)
                        continue
                    if _fp8ws is not None and wshape == (6656, 4608) and weight.stride() == (1, 6656):
                        fp8_attn_qkv.append(layer)
                        continue
                    if _fp8ws is not None and wshape == (6656, 4096) and weight.stride() == (1, 6656):
                        fp8_attn_gate.append(layer)
                        continue
            if (
                weight.dtype != torch.int32
                or not weight.is_contiguous()
                or getattr(layer, "weight_scale", None) is None
                or getattr(layer, "weight_global_scale", None) is None
            ):
                continue
            n = getattr(layer, "output_size_per_partition", 0)
            k = getattr(layer, "input_size_per_partition", 0)
            if tuple(weight.shape) != (k // 16, 2 * n):
                continue
            path = getattr(layer, "marlin_prompt_path", None)
            if weight.numel() * 4 >= MIN_PAYLOAD_BYTES:
                if _swiglu is not None and getattr(layer, "marlin_swiglu_ready", False) and (n, k) == (39936, 6656):
                    gate_layers.append(layer)
                elif (
                    _down is not None and path is not None and path[0] == "wide" and path[2][0] is not None
                    and (n, k) == (_down.COLUMNS, _down.INNER)
                ):
                    down_layers.append(layer)
            elif PREFILL_PLACE_ATTN and _marlin_attn is not None and path is not None and path[0] == "narrow_attn":
                if (n, k) == (4608, 6656):
                    nvfp4_attn_qkv.append(layer)
                elif (n, k) == (4096, 6656):
                    nvfp4_attn_gate.append(layer)
                elif (n, k) == (6656, 4096):
                    nvfp4_attn_o.append(layer)
        evict_buf = torch.empty((_L2_FLUSH_INTS,), dtype=torch.int32, device="cuda")
        saved_ms = 0.0
        groups = []
        if len(gate_layers) > 1:
            ref = gate_layers[0]
            x = ((torch.arange(512 * 6656, device=ref.weight.device) % 13).float() - 6.0).mul(0.11)
            x = x.to(torch.bfloat16).reshape(512, 6656).contiguous()

            fast = hasattr(_swiglu, "uses_128") and all(_swiglu.uses_128(layer.weight) for layer in gate_layers)
            if fast and hasattr(_swiglu, "choose_feed") and os.environ.get("MUSE_PREFILL_FEED_CHOICE", "1") != "0":
                try:
                    choice = _swiglu.choose_feed(
                        gate_layers, x, lambda run, ws: _time_calls(run, ws, PREFILL_ROUNDS, evict_buf)
                    )
                except Exception as err:  # noqa: BLE001 - keep the admitted kernels as they are
                    choice = None
                    logger.warning("512-row gate/up feed choice skipped: %s", err)
                if choice is not None:
                    logger.info(
                        "512-row gate/up feed choice over %d layers: cp.async-fed %.2f ms, TMA-fed %.2f ms per "
                        "prefill -> %s", len(gate_layers), choice[0], choice[1],
                        "TMA-fed kept" if choice[2] else "cp.async-fed on every layer",
                    )
            if hasattr(_swiglu, "gemm_with"):
                def run_gate(weight, ref=ref, x=x, fast=fast):
                    _swiglu.gemm_with(x, weight, ref.weight_scale, ref.weight_global_scale, fast)
            else:
                def run_gate(weight, ref=ref, x=x):
                    _swiglu.gemm(x, weight, ref.weight_scale, ref.weight_global_scale)

            groups.append(("39936x6656", gate_layers, run_gate, getattr(_swiglu, "moved", None), fast))
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
                           _route.down128_moved if _route is not None else None, fast))
        if len(fp8_down_layers) > 1:
            ref = fp8_down_layers[0]
            dev = ref.weight.device
            x = torch.zeros((512, 19968), dtype=torch.float8_e4m3fn, device=dev)
            sa = ref.input_scale if getattr(ref, "input_scale", None) is not None else torch.ones((1,), dtype=torch.float32, device=dev)
            sb = ref.weight_scale
            cfg_dn = _fp8pf.config(6656, 19968) or 1
            use_ws_dn = (
                _fp8ws is not None
                and _fp8ws.ready(dev)
                and os.environ.get("FP8WS", "1") != "0"
                and (6656, 19968) in getattr(_fp8ws, "SHAPES", ())
            )

            def run_fp8_down(weight, x=x, sa=sa, sb=sb, cfg_dn=cfg_dn, use_ws_dn=use_ws_dn):
                if use_ws_dn:
                    _fp8ws.gemm(x, weight, sa, sb)
                else:
                    _fp8pf.gemm(x, weight, sa, sb, cfg_dn)

            groups.append(("fp8_6656x19968", fp8_down_layers, run_fp8_down, _gfp8.prefill_moved, False))
        if len(fp8_swiglu_layers) > 1:
            ref = fp8_swiglu_layers[0]
            dev = ref._buffers["paired_fp8_swiglu_weight"].device
            x = torch.zeros((512, 6656), dtype=torch.float8_e4m3fn, device=dev)
            sa = ref.input_scale if getattr(ref, "input_scale", None) is not None else torch.ones((1,), dtype=torch.float32, device=dev)
            sb = ref.weight_scale
            ds = torch.ones((1,), dtype=torch.float32, device=dev)

            def run_fp8_sw(weight, x=x, sa=sa, sb=sb, ds=ds):
                _fp8sw.apply(x, weight, sa, sb, ds)

            groups.append(("fp8_swiglu_39936x6656", fp8_swiglu_layers, run_fp8_sw, None, False))
        if len(bf16_attn_layers) > 1:
            dev = bf16_attn_layers[0]._buffers["_bf16_qkv_gate_w"].device
            x = torch.zeros((512, 6656), dtype=torch.bfloat16, device=dev)
            use_bf16_ws = _fp8ws is not None and _fp8ws.ready(dev) and hasattr(_fp8ws, "bf16_gemm_dual_stacked")

            def run_bf16_attn(weight, x=x, use_bf16_ws=use_bf16_ws):
                if use_bf16_ws:
                    res = _fp8ws.bf16_gemm_dual_stacked(x, weight)
                    if res is not None:
                        return
                torch.mm(x, weight.t())

            def on_bf16_attn_move(old_ptr, new_ptr, layers=bf16_attn_layers):
                try:
                    from vllm.model_executor.models.muse_glimmer import _BF16_DUAL_WS_ADMITTED
                    if old_ptr in _BF16_DUAL_WS_ADMITTED:
                        _BF16_DUAL_WS_ADMITTED.discard(old_ptr)
                        _BF16_DUAL_WS_ADMITTED.add(new_ptr)
                except Exception:
                    pass
                for attn_mod in layers:
                    w_cat = attn_mod._buffers["_bf16_qkv_gate_w"]
                    if w_cat.data_ptr() == new_ptr:
                        attn_mod.qkv_proj.weight.data = w_cat[:4608]
                        attn_mod.output_gate_proj.weight.data = w_cat[4608:]
                        break

            groups.append(("bf16_attn_8704x6656", bf16_attn_layers, run_bf16_attn, on_bf16_attn_move, False))
        if PREFILL_PLACE_ATTN and len(fp8_attn_o) > 1:
            ref = fp8_attn_o[0]
            dev = ref.weight.device
            x = torch.zeros((512, 4096), dtype=torch.float8_e4m3fn, device=dev)
            sa = ref.input_scale if getattr(ref, "input_scale", None) is not None else torch.ones((1,), dtype=torch.float32, device=dev)
            sb = ref.weight_scale
            cfg_o = _fp8pf.config(6656, 4096) or 3
            use_ws_o = (
                _fp8ws is not None
                and _fp8ws.ready(dev)
                and os.environ.get("FP8WS", "1") != "0"
                and (6656, 4096) in getattr(_fp8ws, "SHAPES", ())
            )

            def run_fp8_o(weight, x=x, sa=sa, sb=sb, cfg_o=cfg_o, use_ws_o=use_ws_o):
                if use_ws_o:
                    _fp8ws.gemm(x, weight, sa, sb)
                else:
                    _fp8pf.gemm(x, weight, sa, sb, cfg_o)

            groups.append(("fp8_attn_6656x4096", fp8_attn_o, run_fp8_o, _gfp8.prefill_moved, False))
        if PREFILL_PLACE_ATTN and _fp8ws is not None and _fp8ws.ready(torch.device("cuda")):
            for gname, glayers, gk in (
                ("fp8_attn_4608x6656", fp8_attn_qkv, 6656),
                ("fp8_attn_4096x6656", fp8_attn_gate, 6656),
            ):
                if len(glayers) > 1:
                    ref = glayers[0]
                    dev = ref.weight.device
                    x = torch.zeros((512, gk), dtype=torch.float8_e4m3fn, device=dev)
                    sa = ref.input_scale if getattr(ref, "input_scale", None) is not None else torch.ones((1,), dtype=torch.float32, device=dev)
                    sb = ref.weight_scale

                    def run_fp8_ws(weight, x=x, sa=sa, sb=sb):
                        _fp8ws.gemm(x, weight, sa, sb)

                    groups.append((gname, glayers, run_fp8_ws, _gfp8.prefill_moved, False))
        if PREFILL_PLACE_ATTN and _marlin_attn is not None:
            for gname, glayers, gn, gk in (
                ("nvfp4_attn_4608x6656", nvfp4_attn_qkv, 4608, 6656),
                ("nvfp4_attn_4096x6656", nvfp4_attn_gate, 4096, 6656),
                ("nvfp4_attn_6656x4096", nvfp4_attn_o, 6656, 4096),
            ):
                if len(glayers) > 1:
                    ref = glayers[0]
                    dev = ref.weight.device
                    x = ((torch.arange(512 * gk, device=dev) % 11).float() - 5.0).mul(0.07).to(torch.bfloat16).reshape(512, gk).contiguous()
                    if (
                        (gn, gk) in ((4608, 6656), (4096, 6656), (6656, 4096))
                        and _route is not None
                        and _route.marlinw_down128 is not None
                        and all(l.weight.data_ptr() in _route._attn128_weights for l in glayers)
                    ):
                        def run_nv_attn(weight, ref=ref, x=x, gn=gn, gk=gk):
                            _route.marlinw_down128.gemm_shape(x, weight, ref.weight_scale, ref.weight_global_scale, gn, gk)
                        on_mv = _route.down128_moved
                    else:
                        plan = ref.marlin_prompt_path[1:]
                        def run_nv_attn(weight, ref=ref, x=x, plan=plan):
                            _marlin_attn.gemm(x, weight, ref.weight_scale, ref.weight_global_scale, *plan)
                        on_mv = _route.down128_moved if _route is not None else None
                    groups.append((gname, glayers, run_nv_attn, on_mv, False))
        for name, layers, run, on_move, fast in groups:
            try:
                if name == "fp8_swiglu_39936x6656":
                    buf_name = "paired_fp8_swiglu_weight"
                elif name == "bf16_attn_8704x6656":
                    buf_name = "_bf16_qkv_gate_w"
                else:
                    buf_name = None
                stats = _place_prefill_group(layers, run, on_move, evict_buf, rejected_hold, buf_name=buf_name)
                if (
                    stats is not None
                    and PREFILL_PASSES >= 2
                    and name in ("39936x6656", "6656x19968", "fp8_6656x19968")
                    and (stats["before"][3] - stats["after"][3]) >= 0.05
                ):
                    for _pass in range(1, PREFILL_PASSES):
                        stats2 = _place_prefill_group(layers, run, on_move, evict_buf, rejected_hold, buf_name=buf_name)
                        if stats2 is None:
                            break
                        gain2 = stats2["before"][3] - stats2["after"][3]
                        stats["moved"] += stats2["moved"]
                        stats["after"] = stats2["after"]
                        if gain2 < 0.05:
                            break
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
        # Keep _SLOW_HOLD quarantined until gpu_worker calls release_slow_hold() after initialize_kv_cache().
        had_hold = bool(rejected_hold)
        rejected_hold.clear()
        if had_hold and torch.cuda.is_available():
            avail = _mem_available()
            if avail is None or avail < HOST_HEADROOM_BYTES:
                torch.cuda.empty_cache()


def place_decode_weights(model: torch.nn.Module) -> None:
    """Move decode weights that landed on slow device memory; never raises."""
    global _KV_BLOCK_RANKS
    if not ENABLED or not torch.cuda.is_available():
        return
    try:
        from muse_marlin_gemv import decode_linear
    except (ImportError, OSError):
        return
    began = time.perf_counter()
    _SLOW_HOLD.clear()
    _KV_BLOCK_RANKS = None
    rejected_hold: list[torch.Tensor] = []
    try:
        _drop_model_page_cache()
        evict_buf = torch.empty((_L2_FLUSH_INTS,), dtype=torch.int32, device="cuda")
        groups: dict[tuple, list] = {}
        rans_groups: dict[tuple[int, int], list] = {}
        attn_nvfp4_groups: dict[tuple, list] = {}
        attn_rans_groups: dict[tuple[int, int], list] = {}
        attn_bf16_groups: dict[tuple[int, int], list] = {}
        for layer in model.modules():
            bufs = getattr(layer, "_buffers", {})
            weight = bufs.get(_NAME)
            if (
                weight is not None
                and weight.is_cuda
                and weight.is_contiguous()
                and getattr(layer, "_marlin_decode_scales", None) is not None
                and getattr(layer, "weight_global_scale", None) is not None
                and getattr(layer, "workspace", None) is not None
            ):
                nbytes = weight.numel() * weight.element_size()
                n, k = layer.output_size_per_partition, layer.input_size_per_partition
                key = (n, k, tuple(weight.shape), weight.dtype, weight.device)
                if nbytes >= MIN_PAYLOAD_BYTES:
                    groups.setdefault(key, []).append(layer)
                elif PLACE_ATTN and nbytes >= ATTN_MIN_BYTES:
                    attn_nvfp4_groups.setdefault(key, []).append(layer)
            rblob = bufs.get("muse_fp8_packed")
            if (
                rblob is not None
                and rblob.is_cuda
                and rblob.dtype == torch.uint8
                and rblob.dim() == 1
                and rblob.is_contiguous()
                and getattr(layer, "weight_scale", None) is not None
            ):
                n = getattr(layer, "output_size_per_partition", 0)
                k = getattr(layer, "input_size_per_partition", 0)
                if n > 0 and k > 0:
                    if rblob.numel() >= MIN_PAYLOAD_BYTES:
                        rans_groups.setdefault((n, k), []).append(layer)
                    elif PLACE_ATTN and rblob.numel() >= ATTN_MIN_BYTES:
                        attn_rans_groups.setdefault((n, k), []).append(layer)
            bblob = bufs.get("muse_bf16_lane_weight")
            bidx = bufs.get("muse_bf16_lane_index")
            if (
                PLACE_ATTN
                and bblob is not None
                and bidx is not None
                and bidx.numel() == 1
                and bblob.is_cuda
                and bblob.dtype == torch.uint8
                and bblob.dim() == 1
                and bblob.is_contiguous()
                and bblob.numel() >= ATTN_MIN_BYTES
            ):
                n = getattr(layer, "output_size_per_partition", 0)
                k = getattr(layer, "input_size_per_partition", 0)
                if n > 0 and k > 0:
                    attn_bf16_groups.setdefault((n, k), []).append(layer)
        saved_ms = 0.0
        n_groups = 0
        # Place 39936x6656 first (full-RAM scan seeds PyTorch's fast cache and quarantines slow pages in _SLOW_HOLD),
        # then 6656x19968 down, then FP8 rANS MLPs, and defer 641 MiB lm_head until after rANS MLPs so its spares
        # never crowd the fast pool.
        ordered_groups = sorted(
            groups.items(),
            key=lambda kv: (1 if len(kv[1]) == 1 else 0, -(len(kv[1]) * kv[1][0]._buffers[_NAME].numel())),
        )
        deferred_single = [(k, v) for k, v in ordered_groups if len(v) == 1]
        multi_nvfp4 = [(k, v) for k, v in ordered_groups if len(v) > 1]
        packer: _FastSlabPacker | None = None
        packed_groups: set[tuple[str, int, int]] = set()
        for (n, k, *_), layers in multi_nvfp4:
            if ("nvfp4", n, k) in packed_groups:
                continue
            try:
                stats = None
                if FULL_RAM_SCAN and (n, k) == (39936, 6656) and len(layers) >= 32:
                    stats = _place_group_full_ram_scan(decode_linear, layers, n, k, evict_buf)
                    if stats is not None:
                        fast_slabs = stats.pop("fast_slabs", [])
                        if fast_slabs:
                            packer = _FastSlabPacker(fast_slabs)
                            # Pack all remaining <= 127 MiB decode weights directly into the top ranked
                            # fast slabs in descending decode-impact order, before releasing unused slabs.
                            p_moved, p_bytes = 0, 0
                            # 1) NVFP4 multi-layer groups (e.g. 6656x19968 down: 40 x 63 MiB -> 20 slabs)
                            for (gn, gk, *_), glayers in multi_nvfp4:
                                if (gn, gk) == (39936, 6656):
                                    continue
                                ok_all = True
                                for gl in glayers:
                                    src = gl._buffers[_NAME]
                                    dst = packer.alloc_like(src)
                                    if dst is None:
                                        ok_all = False
                                        break
                                    dst.copy_(src)
                                    if torch.equal(dst, src):
                                        gl._buffers[_NAME] = dst
                                        p_moved += 1
                                        p_bytes += dst.numel() * dst.element_size()
                                    else:
                                        ok_all = False
                                if ok_all:
                                    packed_groups.add(("nvfp4", gn, gk))
                            # 2) FP8 rANS <= 127 MiB (6656x19968 down: 12 x ~105 MiB -> 12 slabs)
                            for (gn, gk), glayers in sorted(
                                rans_groups.items(),
                                key=lambda kv: -(len(kv[1]) * max(l._buffers["muse_fp8_packed"].numel() for l in kv[1])),
                            ):
                                max_b = max(l._buffers["muse_fp8_packed"].numel() for l in glayers)
                                if max_b > (128 << 20):
                                    continue
                                ok_all = True
                                for gl in glayers:
                                    src = gl._buffers["muse_fp8_packed"]
                                    dst = packer.alloc_u8(src.numel())
                                    if dst is None:
                                        ok_all = False
                                        break
                                    dst.copy_(src)
                                    if torch.equal(dst, src):
                                        old = src.data
                                        src.data = dst
                                        if src.data_ptr() != dst.data_ptr():
                                            src.data = old
                                            ok_all = False
                                        else:
                                            p_moved += 1
                                            p_bytes += dst.numel()
                                    else:
                                        ok_all = False
                                if ok_all:
                                    packed_groups.add(("rans", gn, gk))
                            # 3) FP8 rANS attention (98 x ~23 MiB -> 19 slabs)
                            if PLACE_ATTN:
                                for (gn, gk), glayers in sorted(
                                    attn_rans_groups.items(),
                                    key=lambda kv: -(len(kv[1]) * max(l._buffers["muse_fp8_packed"].numel() for l in kv[1])),
                                ):
                                    ok_all = True
                                    for gl in glayers:
                                        src = gl._buffers["muse_fp8_packed"]
                                        dst = packer.alloc_u8(src.numel())
                                        if dst is None:
                                            ok_all = False
                                            break
                                        dst.copy_(src)
                                        if torch.equal(dst, src):
                                            old = src.data
                                            src.data = dst
                                            if src.data_ptr() != dst.data_ptr():
                                                src.data = old
                                                ok_all = False
                                            else:
                                                p_moved += 1
                                                p_bytes += dst.numel()
                                        else:
                                            ok_all = False
                                    if ok_all:
                                        packed_groups.add(("attn_rans", gn, gk))
                                # 4) BF16 rANS attention (12 x ~37 MiB -> 4 slabs)
                                for (gn, gk), glayers in sorted(
                                    attn_bf16_groups.items(),
                                    key=lambda kv: -(len(kv[1]) * max(l._buffers["muse_bf16_lane_weight"].numel() for l in kv[1])),
                                ):
                                    ok_all = True
                                    for gl in glayers:
                                        src = gl._buffers["muse_bf16_lane_weight"]
                                        dst = packer.alloc_u8(src.numel())
                                        if dst is None:
                                            ok_all = False
                                            break
                                        dst.copy_(src)
                                        if torch.equal(dst, src):
                                            old = src.data
                                            src.data = dst
                                            if src.data_ptr() != dst.data_ptr():
                                                src.data = old
                                                ok_all = False
                                            else:
                                                p_moved += 1
                                                p_bytes += dst.numel()
                                        else:
                                            ok_all = False
                                    if ok_all:
                                        packed_groups.add(("attn_bf16", gn, gk))
                                # 5) NVFP4 attention (46 x ~14 MiB -> 6 slabs)
                                for (gn, gk, *_), glayers in sorted(
                                    attn_nvfp4_groups.items(),
                                    key=lambda kv: -(len(kv[1]) * kv[1][0]._buffers[_NAME].numel()),
                                ):
                                    ok_all = True
                                    for gl in glayers:
                                        src = gl._buffers[_NAME]
                                        dst = packer.alloc_like(src)
                                        if dst is None:
                                            ok_all = False
                                            break
                                        dst.copy_(src)
                                        if torch.equal(dst, src):
                                            gl._buffers[_NAME] = dst
                                            p_moved += 1
                                            p_bytes += dst.numel() * dst.element_size()
                                        else:
                                            ok_all = False
                                    if ok_all:
                                        packed_groups.add(("attn_nvfp4", gn, gk))
                            # 6) scales6 / TMA4 scale + palette buffers (all NVFP4 layers incl. gate_up & lm_head)
                            if PLACE_SCALES6:
                                try:
                                    import muse_marlin6_gemv as _m6
                                except (ImportError, OSError):
                                    _m6 = None
                                moved_s6, bytes_s6 = 0, 0
                                for m in model.modules():
                                    s6 = m._buffers.get("_marlin_decode_scales6")
                                    if s6 is not None and s6.is_cuda and s6.numel() * s6.element_size() >= (64 << 10):
                                        old_ptr = s6.data_ptr()
                                        s6_nbytes = s6.numel() * s6.element_size()
                                        cap_b = None
                                        if (
                                            s6.dtype == torch.uint8
                                            and (
                                                getattr(m, "output_size_per_partition", 0),
                                                getattr(m, "input_size_per_partition", 0),
                                            )
                                            == (39936, 6656)
                                        ):
                                            base_cap = (
                                                s6._base.numel() * s6._base.element_size()
                                                if s6._base is not None
                                                else 12_459_648
                                            )
                                            cap_b = min(
                                                max(s6_nbytes, base_cap),
                                                max(s6_nbytes + (256 << 10), ((int(s6_nbytes * 1.05) + 31) & ~31)),
                                            )
                                        dst = packer.alloc_like(s6, cap_bytes=cap_b)
                                        if dst is None:
                                            break
                                        dst.copy_(s6)
                                        if torch.equal(dst, s6):
                                            m._buffers["_marlin_decode_scales6"] = dst
                                            _relocate_scales6_ptr(_m6, old_ptr, dst.data_ptr())
                                            moved_s6 += 1
                                            bytes_s6 += s6_nbytes
                                            p_moved += 1
                                            p_bytes += s6_nbytes
                                    elif s6 is None:
                                        s_raw = m._buffers.get("_marlin_decode_scales")
                                        if (
                                            s_raw is not None
                                            and s_raw.is_cuda
                                            and (64 << 10) <= s_raw.numel() * s_raw.element_size() <= (126 << 20)
                                        ):
                                            dst_raw = packer.alloc_like(s_raw)
                                            if dst_raw is not None:
                                                dst_raw.copy_(s_raw)
                                                if torch.equal(dst_raw, s_raw):
                                                    m._buffers["_marlin_decode_scales"] = dst_raw
                                                    p_moved += 1
                                                    p_bytes += dst_raw.numel() * dst_raw.element_size()
                                    pal = m._buffers.get("_marlin_decode_palette")
                                    if pal is not None and pal.is_cuda and pal.dtype == torch.uint8 and 0 < pal.numel() <= 256:
                                        dst_pal = packer.alloc_u8(pal.numel(), align=64)
                                        if dst_pal is not None:
                                            dst_pal.copy_(pal)
                                            if torch.equal(dst_pal, pal):
                                                old_pal = pal.data
                                                pal.data = dst_pal
                                                if pal.data_ptr() != dst_pal.data_ptr():
                                                    pal.data = old_pal
                                if moved_s6:
                                    logger.info(
                                        "Decode scales6 direct fast-slab packing: %d buffers (%.1f MiB)",
                                        moved_s6, bytes_s6 / (1 << 20),
                                    )
                            used_slabs = packer.used_slabs
                            packer.release_unused()
                            del packer, fast_slabs
                            packer = None
                            logger.info(
                                "Decode direct fast-slab packing: %d tensors (%.1f MiB) packed into top %d fast slabs",
                                p_moved, p_bytes / (1 << 20), used_slabs,
                            )
                        elif PLACE_SCALES6:
                            try:
                                import muse_marlin6_gemv as _m6
                            except (ImportError, OSError):
                                _m6 = None
                            moved_s6, bytes_s6 = 0, 0
                            for m in model.modules():
                                s6 = m._buffers.get("_marlin_decode_scales6")
                                if s6 is not None and s6.is_cuda and s6.numel() * s6.element_size() >= (64 << 10):
                                    try:
                                        old_ptr = s6.data_ptr()
                                        dst = torch.empty_like(s6)
                                        dst.copy_(s6)
                                        if torch.equal(dst, s6):
                                            m._buffers["_marlin_decode_scales6"] = dst
                                            rejected_hold.append(s6)
                                            _relocate_scales6_ptr(_m6, old_ptr, dst.data_ptr())
                                            moved_s6 += 1
                                            bytes_s6 += dst.numel() * dst.element_size()
                                    except RuntimeError:
                                        break
                            if moved_s6:
                                logger.info(
                                    "Decode scales6 fast-pool relocation: %d buffers (%.1f MiB) moved into fast pool",
                                    moved_s6, bytes_s6 / (1 << 20),
                                )
                if stats is None:
                    stats = _place_group(decode_linear, layers, n, k, evict_buf, rejected_hold)
                    if (
                        stats is not None
                        and DECODE_DOWN_PASSES >= 2
                        and (n, k) == (6656, 19968)
                        and (stats["before"][3] - stats["after"][3]) >= 0.05
                    ):
                        for _pass in range(1, DECODE_DOWN_PASSES):
                            stats2 = _place_group(decode_linear, layers, n, k, evict_buf, rejected_hold)
                            if stats2 is None:
                                break
                            gain2 = stats2["before"][3] - stats2["after"][3]
                            stats["moved"] += stats2["moved"]
                            stats["after"] = stats2["after"]
                            if gain2 < 0.03:
                                break
            except Exception as err:  # noqa: BLE001 - this group stays where it was
                logger.warning("Decode weight placement skipped for %dx%d: %s", n, k, err)
                continue
            if stats is None:
                continue
            n_groups += 1
            b, a = stats["before"], stats["after"]
            saved_ms += b[3] - a[3]
            gbps = stats["payload_bytes"] / 1e6
            logger.info(
                "Decode weight placement %dx%d: %d weights of %.0f MiB, %d spare buffers, %d moved; "
                "GEMV ms min/median/max %.4f/%.4f/%.4f -> %.4f/%.4f/%.4f "
                "(weight-only %.0f/%.0f/%.0f -> %.0f/%.0f/%.0f GB/s), sum %.3f -> %.3f ms",
                n, k, stats["count"], stats["payload_bytes"] / (1 << 20), stats["spares"], stats["moved"],
                b[0], b[1], b[2], a[0], a[1], a[2],
                gbps / b[0], gbps / b[1], gbps / b[2], gbps / a[0], gbps / a[1], gbps / a[2], b[3], a[3],
            )
        deferred_large_rans: list[tuple[tuple[int, int], list]] = []
        _rans_ext = None
        if rans_groups or (PLACE_ATTN and attn_rans_groups):
            try:
                import muse_rans_ext as _rans_ext
            except (ImportError, OSError):
                _rans_ext = None
        if rans_groups and _rans_ext is not None:
            ordered_rans = sorted(
                rans_groups.items(),
                key=lambda kv: -(len(kv[1]) * max(l._buffers["muse_fp8_packed"].numel() for l in kv[1])),
            )
            for (n, k), layers in ordered_rans:
                max_b = max(l._buffers["muse_fp8_packed"].numel() for l in layers)
                if max_b > (128 << 20):
                    deferred_large_rans.append(((n, k), layers))
                    continue
                if ("rans", n, k) in packed_groups:
                    continue
                try:
                    stats = _place_muserans_group(_rans_ext, layers, n, k, evict_buf, rejected_hold)
                except Exception as err:  # noqa: BLE001
                    logger.warning("Decode rANS placement skipped for %dx%d: %s", n, k, err)
                    continue
                if stats is None:
                    continue
                n_groups += 1
                b, a = stats["before"], stats["after"]
                saved_ms += b[3] - a[3]
                logger.info(
                    "Decode rANS placement %dx%d: %d weights of ~%.0f MiB, %d spare buffers, %d moved; "
                    "GEMV ms min/median/max %.4f/%.4f/%.4f -> %.4f/%.4f/%.4f, sum %.3f -> %.3f ms",
                    n, k, stats["count"], stats["payload_bytes"] / (1 << 20), stats["spares"], stats["moved"],
                    b[0], b[1], b[2], a[0], a[1], a[2], b[3], a[3],
                )
        if PLACE_ATTN and attn_bf16_groups:
            try:
                import muse_bf16_lane as _bf16_mod
                _bf16_lib = _bf16_mod._lib()
            except (ImportError, OSError, AttributeError):
                _bf16_lib = None
            if _bf16_lib is not None and hasattr(_bf16_lib, "muse_bf16_rans_gemv"):
                for (n, k), layers in sorted(
                    attn_bf16_groups.items(),
                    key=lambda kv: -(len(kv[1]) * max(l._buffers["muse_bf16_lane_weight"].numel() for l in kv[1])),
                ):
                    if ("attn_bf16", n, k) in packed_groups:
                        continue
                    try:
                        stats = _place_bf16_rans_group(_bf16_lib, layers, n, k, evict_buf, rejected_hold)
                    except Exception as err:  # noqa: BLE001
                        logger.warning("Decode BF16 rANS attn placement skipped for %dx%d: %s", n, k, err)
                        continue
                    if stats is None:
                        continue
                    n_groups += 1
                    b, a = stats["before"], stats["after"]
                    saved_ms += b[3] - a[3]
                    logger.info(
                        "Decode BF16 rANS attn placement %dx%d: %d weights of ~%.0f MiB, %d spare buffers, %d moved; "
                        "GEMV ms min/median/max %.4f/%.4f/%.4f -> %.4f/%.4f/%.4f, sum %.3f -> %.3f ms",
                        n, k, stats["count"], stats["payload_bytes"] / (1 << 20), stats["spares"], stats["moved"],
                        b[0], b[1], b[2], a[0], a[1], a[2], b[3], a[3],
                    )
        if PLACE_ATTN and attn_rans_groups and _rans_ext is not None:
            for (n, k), layers in sorted(
                attn_rans_groups.items(),
                key=lambda kv: -(len(kv[1]) * max(l._buffers["muse_fp8_packed"].numel() for l in kv[1])),
            ):
                if ("attn_rans", n, k) in packed_groups:
                    continue
                try:
                    stats = _place_muserans_group(_rans_ext, layers, n, k, evict_buf, rejected_hold)
                except Exception as err:  # noqa: BLE001
                    logger.warning("Decode rANS attn placement skipped for %dx%d: %s", n, k, err)
                    continue
                if stats is None:
                    continue
                n_groups += 1
                b, a = stats["before"], stats["after"]
                saved_ms += b[3] - a[3]
                logger.info(
                    "Decode rANS attn placement %dx%d: %d weights of ~%.0f MiB, %d spare buffers, %d moved; "
                    "GEMV ms min/median/max %.4f/%.4f/%.4f -> %.4f/%.4f/%.4f, sum %.3f -> %.3f ms",
                    n, k, stats["count"], stats["payload_bytes"] / (1 << 20), stats["spares"], stats["moved"],
                    b[0], b[1], b[2], a[0], a[1], a[2], b[3], a[3],
                )
        if PLACE_ATTN and attn_nvfp4_groups:
            for (n, k, *_), layers in sorted(
                attn_nvfp4_groups.items(),
                key=lambda kv: -(len(kv[1]) * kv[1][0]._buffers[_NAME].numel()),
            ):
                if ("attn_nvfp4", n, k) in packed_groups:
                    continue
                try:
                    stats = _place_group(decode_linear, layers, n, k, evict_buf, rejected_hold)
                except Exception as err:  # noqa: BLE001
                    logger.warning("Decode NVFP4 attn placement skipped for %dx%d: %s", n, k, err)
                    continue
                if stats is None:
                    continue
                n_groups += 1
                b, a = stats["before"], stats["after"]
                saved_ms += b[3] - a[3]
                logger.info(
                    "Decode NVFP4 attn placement %dx%d: %d weights of %.0f MiB, %d spare buffers, %d moved; "
                    "GEMV ms min/median/max %.4f/%.4f/%.4f -> %.4f/%.4f/%.4f, sum %.3f -> %.3f ms",
                    n, k, stats["count"], stats["payload_bytes"] / (1 << 20), stats["spares"], stats["moved"],
                    b[0], b[1], b[2], a[0], a[1], a[2], b[3], a[3],
                )
        if deferred_large_rans and _rans_ext is not None:
            for (n, k), layers in deferred_large_rans:
                try:
                    stats = _place_muserans_group(
                        _rans_ext, layers, n, k, evict_buf, rejected_hold, surplus_override=LARGE_RANS_SURPLUS
                    )
                except Exception as err:  # noqa: BLE001
                    logger.warning("Decode rANS placement skipped for %dx%d: %s", n, k, err)
                    continue
                if stats is None:
                    continue
                n_groups += 1
                b, a = stats["before"], stats["after"]
                saved_ms += b[3] - a[3]
                logger.info(
                    "Decode rANS placement %dx%d: %d weights of ~%.0f MiB, %d spare buffers, %d moved; "
                    "GEMV ms min/median/max %.4f/%.4f/%.4f -> %.4f/%.4f/%.4f, sum %.3f -> %.3f ms",
                    n, k, stats["count"], stats["payload_bytes"] / (1 << 20), stats["spares"], stats["moved"],
                    b[0], b[1], b[2], a[0], a[1], a[2], b[3], a[3],
                )
        for (n, k, *_), layers in deferred_single:
            try:
                stats = _place_group(decode_linear, layers, n, k, evict_buf, rejected_hold)
            except Exception as err:  # noqa: BLE001
                logger.warning("Decode weight placement skipped for %dx%d: %s", n, k, err)
                continue
            if stats is None:
                continue
            n_groups += 1
            b, a = stats["before"], stats["after"]
            saved_ms += b[3] - a[3]
            gbps = stats["payload_bytes"] / 1e6
            logger.info(
                "Decode weight placement %dx%d: %d weights of %.0f MiB, %d spare buffers, %d moved; "
                "GEMV ms min/median/max %.4f/%.4f/%.4f -> %.4f/%.4f/%.4f "
                "(weight-only %.0f/%.0f/%.0f -> %.0f/%.0f/%.0f GB/s), sum %.3f -> %.3f ms",
                n, k, stats["count"], stats["payload_bytes"] / (1 << 20), stats["spares"], stats["moved"],
                b[0], b[1], b[2], a[0], a[1], a[2],
                gbps / b[0], gbps / b[1], gbps / b[2], gbps / a[0], gbps / a[1], gbps / a[2], b[3], a[3],
            )
        logger.info(
            "Decode weight placement: %d groups, timed GEMV sum reduced by %.3f ms per token, %.2f s",
            n_groups, saved_ms, time.perf_counter() - began,
        )
    except Exception as err:  # noqa: BLE001 - the load must go on
        logger.warning("Decode weight placement failed and was skipped: %s", err)
    finally:
        # Keep _SLOW_HOLD quarantined until gpu_worker calls release_slow_hold() after initialize_kv_cache().
        had_hold = bool(rejected_hold)
        rejected_hold.clear()
        if had_hold and torch.cuda.is_available():
            avail = _mem_available()
            if avail is None or avail < HOST_HEADROOM_BYTES:
                torch.cuda.empty_cache()
    try:
        from vllm.model_executor.kernels.linear.scaled_mm import muse_fp8 as _muse_fp8

        _muse_fp8.refresh_rans_meta(model)
    except Exception as err:  # noqa: BLE001
        logger.warning("muserans fast-meta refresh skipped: %s", err)
    try:
        import muse_marlin6_gemv as _m6
        if hasattr(_m6, "register_all_gate_up_swiglu"):
            _m6.register_all_gate_up_swiglu(model)
        if hasattr(_m6, "register_all_oproj_sigmoid"):
            _m6.register_all_oproj_sigmoid(model)
        if hasattr(_m6, "register_all_down_swiglu"):
            _m6.register_all_down_swiglu(model)
    except Exception as err:  # noqa: BLE001
        logger.warning("Decode6 post-placement registration skipped: %s", err)
