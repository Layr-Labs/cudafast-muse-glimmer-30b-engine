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
PLACE_SCALES6 = os.environ.get("MUSE_PLACE_SCALES6", "1").strip() not in ("0", "", "false", "no")
SURPLUS = float(os.environ.get("MUSE_DECODE_PLACEMENT_SURPLUS", "3.0"))
LARGE_RANS_SURPLUS = float(os.environ.get("MUSE_LARGE_RANS_SURPLUS", "6.0"))
LM_HEAD_SPARES = int(os.environ.get("MUSE_LM_HEAD_SPARES", "24"))
ROUNDS = 5
# Weights that fit in L2 several times over are not bandwidth-bound per call.
MIN_PAYLOAD_BYTES = 48 << 20
# Host memory that must stay available while the spares exist.
HOST_HEADROOM_BYTES = 20 << 30
_NAME = "_marlin_decode_weight"
_L2_FLUSH_INTS = 16 << 20  # 64 MiB (> 2.5x GB10's 24 MiB L2)
# Quarantine below-median (slow) buffers from place_decode_weights through place_prefill_weights (and KV-cache init).
_SLOW_HOLD: list[torch.Tensor] = []
HOLD_SLOW_FOR_KV = os.environ.get("MUSE_HOLD_SLOW_FOR_KV", "1").strip() not in ("0", "", "false", "no")
_SPONGE_STATE: dict[str, int] = {"ptr": 0, "len": 0}


def take_buddyinfo_sponge(need_bytes: int = 36 << 30) -> int:
    """Temporarily commit fragmented small free blocks via cuMemAlloc_v2 before weight loading
    so initial weights land on 32 MiB extents, reserving >=54 GiB free RAM (zero OOM risk)."""
    if not ENABLED or os.environ.get("MUSE_BUDDY_SPONGE", "1").strip() in ("0", "", "false", "no"):
        return 0
    if _SPONGE_STATE["ptr"] != 0:
        return _SPONGE_STATE["len"]
    try:
        import ctypes

        cap_gib = int(os.environ.get("MUSE_BUDDY_SPONGE_CAP_GIB", "24"))
        if cap_gib <= 0:
            return 0
        usable = _usable_buddy_bytes(2 << 20)
        avail = _mem_available()
        if avail is not None:
            usable = min(usable, avail)
        headroom = 18 << 30
        granule = 2 << 20
        least = 1 << 30
        most = cap_gib << 30
        if usable <= need_bytes + headroom:
            return 0
        length = (min(usable - (need_bytes + headroom), most) // granule) * granule
        if length < least:
            return 0
        cuda_drv = ctypes.CDLL("libcuda.so.1")
        dptr = ctypes.c_uint64(0)
        if cuda_drv.cuMemAlloc_v2(ctypes.byref(dptr), ctypes.c_size_t(length)) != 0 or dptr.value == 0:
            return 0
        if cuda_drv.cuMemsetD8_v2(dptr, ctypes.c_ubyte(0), ctypes.c_size_t(length)) != 0:
            cuda_drv.cuMemFree_v2(dptr)
            return 0
        _SPONGE_STATE["ptr"] = int(dptr.value)
        _SPONGE_STATE["len"] = int(length)
        logger.info(
            "Buddyinfo fragmentation sponge: held %.1f GiB (usable=%.1f GiB, reserve=%.1f GiB) while weights commit",
            length / (1 << 30), usable / (1 << 30), (need_bytes + headroom) / (1 << 30),
        )
        return int(length)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Buddyinfo sponge skipped: %s", exc)
        return 0


def release_buddyinfo_sponge() -> None:
    """Release temporary buddyinfo sponge buffer directly via cuMemFree_v2."""
    ptr = _SPONGE_STATE.get("ptr", 0)
    length = _SPONGE_STATE.get("len", 0)
    if not ptr:
        return
    _SPONGE_STATE["ptr"] = 0
    _SPONGE_STATE["len"] = 0
    try:
        import ctypes

        cuda_drv = ctypes.CDLL("libcuda.so.1")
        cuda_drv.cuMemFree_v2(ctypes.c_uint64(ptr))
        logger.info("Buddyinfo fragmentation sponge: released %.1f GiB", length / (1 << 30))
    except Exception:  # noqa: BLE001
        pass


def ensure_slow_hold_headroom(min_free_bytes: int) -> None:
    """Trim oldest _SLOW_HOLD slabs only if available host memory is below min_free_bytes."""
    if not _SLOW_HOLD or not torch.cuda.is_available():
        return
    while _SLOW_HOLD:
        avail = _mem_available()
        if avail is None or avail >= min_free_bytes:
            break
        del _SLOW_HOLD[: max(1, len(_SLOW_HOLD) // 4)]
        torch.cuda.empty_cache()


def release_slow_hold() -> None:
    """Release quarantined slow DRAM slabs after KV-cache allocation completes."""
    if _SLOW_HOLD:
        n = len(_SLOW_HOLD)
        _SLOW_HOLD.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Released %d quarantined slow-DRAM slabs after KV-cache allocation", n)
PLACE_ATTN = os.environ.get("MUSE_PLACE_ATTN", "1").strip() not in ("0", "", "false", "no")
PREFILL_PLACE_ATTN = os.environ.get("MUSE_PREFILL_PLACE_ATTN", "1").strip() not in ("0", "", "false", "no")
ATTN_MIN_BYTES = 10 << 20
ATTN_SURPLUS = float(os.environ.get("MUSE_ATTN_PLACEMENT_SURPLUS", "2.5"))


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


def _flush_l2(evict_buf: torch.Tensor) -> None:
    """Read-only L2 eviction pass (never writes, avoiding dirty-line writeback overhead)."""
    _ = evict_buf.sum()


def _usable_buddy_bytes(min_chunk: int = 2 << 20) -> int:
    """Free bytes in Linux buddy orders >= min_chunk across /proc/buddyinfo."""
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = 0
        with open("/proc/buddyinfo") as fp:
            for line in fp:
                parts = line.replace(",", " ").split()
                if len(parts) <= 4:
                    continue
                for order, tok in enumerate(parts[4:]):
                    cnt = int(tok)
                    blk = page << order
                    if blk >= min_chunk:
                        total += cnt * blk
        return total
    except Exception:
        return 0


def _alloc_buddy_sponge(need_bytes: int) -> torch.Tensor | None:
    """Temporarily absorb fragmented lower-order free pages before allocating large
    unfragmented Order-13 (32 MiB) weight spares; never fails the load."""
    if os.environ.get("MUSE_LM_BUDDY_SPONGE", "0").strip() in ("0", "", "false", "no"):
        return None
    try:
        usable = _usable_buddy_bytes(2 << 20)
        avail = _mem_available()
        if avail is not None:
            usable = min(usable, avail)
        keep = need_bytes + HOST_HEADROOM_BYTES
        if usable <= keep:
            return None
        sponge_bytes = min(usable - keep, 48 << 30) & ~((2 << 20) - 1)
        if sponge_bytes < (1 << 30):
            return None
        t = torch.empty((sponge_bytes,), dtype=torch.uint8, device="cuda")
        t[::4096] = 0  # commit physical pages on first touch
        torch.cuda.synchronize()
        logger.info(
            "Buddyinfo fragmentation sponge: committed %.1f GiB (usable >=2MiB: %.1f GiB) for high-order placement",
            sponge_bytes / (1 << 30), usable / (1 << 30),
        )
        return t
    except Exception:
        return None


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


def _time_candidates(decode_linear, layer, candidates, n: int, k: int, rounds: int, evict_buf: torch.Tensor) -> list[float]:
    """Median GEMV time (ms) of the tiled decode kernel on each candidate buffer."""
    device = candidates[0].device
    # A fixed finite row; the kernel's timing does not depend on values.
    x = ((torch.arange(k, device=device) % 7).float() - 3.0).to(torch.bfloat16).reshape(1, k)
    global_scale, workspace = layer.weight_global_scale, layer.workspace
    palette = getattr(layer, "_marlin_decode_palette", None)
    scales6 = getattr(layer, "_marlin_decode_scales6", None)
    fixed_grid = bool(getattr(layer, "marlin_decode6_fixed_grid", False))
    use_6 = False
    _m6_mod = None
    if palette is not None and scales6 is not None:
        try:
            import muse_marlin6_gemv as _m6_mod
            decode_linear6 = _m6_mod.decode_linear6
            use_6 = True
        except (ImportError, OSError):
            use_6 = False
    if use_6:
        # Pre-warm Split2/Pad128 scale header checks and use fused lm_head+lmpost kernel when active on lm_head.
        _is_pad128 = hasattr(_m6_mod, "is_lm_pad128") and _m6_mod.is_lm_pad128(scales6)
        _is_split2 = hasattr(_m6_mod, "is_lm_split2") and _m6_mod.is_lm_split2(scales6)
        use_lm_lmpost = (
            (n, k) == (202048, 6656)
            and os.environ.get("MUSE_PLACE_LMHEAD_SPLIT2", "1") != "0"
            and hasattr(_m6_mod, "is_tma4_ilv")
            and getattr(_m6_mod, "_sk_tma4_ilv_lmpost", None) is not None
            and _m6_mod.is_tma4_ilv(scales6)
            and (_is_split2 or _is_pad128)
        )
        if use_lm_lmpost:
            scratch, partials = _m6_mod._get_lmpost_bufs(device)
            dummy_lut = torch.zeros(65536, dtype=torch.bfloat16, device=device)
            dummy_tok = torch.zeros((1, 1), dtype=torch.int64, device=device)
            org_vocab = int(getattr(layer, "org_vocab_size", n))
            eff_n = 202112 if _is_pad128 else n
            lm_blocks = 96 if _is_split2 else 48
            dev_idx = device.index if device.index is not None else torch.cuda.current_device()
            lmpost_fn = _m6_mod._sk_tma4_ilv_lmpost

            def run_one(buf):
                rc = lmpost_fn(
                    x.data_ptr(), buf.data_ptr(), scratch.data_ptr(), partials.data_ptr(),
                    scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(),
                    dummy_lut.data_ptr(), dummy_tok.data_ptr(), org_vocab, eff_n, k, lm_blocks, dev_idx,
                    torch.cuda.current_stream(device).cuda_stream,
                )
                if rc:
                    raise RuntimeError(f"lm_head placement probe returned {rc}")
        else:
            def run_one(buf):
                decode_linear6(x, buf, scales6, palette, global_scale, workspace, n, k, fixed_grid)
    else:
        scales = layer._marlin_decode_scales
        def run_one(buf):
            decode_linear(x, buf, scales, global_scale, workspace, n, k)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    run_one(candidates[0])
    _flush_l2(evict_buf)
    torch.cuda.synchronize(device)
    samples: list[list[float]] = [[] for _ in candidates]
    order = list(range(len(candidates)))
    shuffler = random.Random(20261122)
    for _ in range(rounds):
        shuffler.shuffle(order)
        for index in order:
            start.record()
            run_one(candidates[index])
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end))
    return [median(values) for values in samples]


def _place_group_full_ram_scan(decode_linear, layers, n: int, k: int, evict_buf: torch.Tensor) -> dict | None:
    """Scan available host RAM in 127 MiB blocks (~4s), place gate_up into the fastest 45,
    seed PyTorch's caching allocator with the next ~24 GiB of fastest blocks, and quarantine
    the slowest ~44 GiB in _SLOW_HOLD so all subsequent decode & prefill groups draw only
    from the fast physical DRAM band."""
    homes = [layer._buffers[_NAME] for layer in layers]
    count = len(homes)
    payload_bytes = homes[0].numel() * homes[0].element_size()
    spares_count = _spares_allowed(600, payload_bytes)
    if spares_count < count * 4:
        return None
    spares = []
    try:
        for _ in range(spares_count):
            spare = torch.empty_like(homes[0])
            spare.copy_(homes[0])
            spares.append(spare)
    except RuntimeError:
        pass
    if len(spares) < count * 4:
        spares.clear()
        return None
    candidates = homes + spares
    times = _time_candidates(decode_linear, layers[0], candidates, n, k, 5, evict_buf)
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
    # Partition all unused candidates (spares + displaced homes) by measured speed:
    # release the fastest ~24 GiB (190 blocks) to PyTorch's caching allocator (no empty_cache),
    # quarantine up to ~22 GiB (175 slowest blocks) in _SLOW_HOLD, and release any middle overflow.
    unused = [(times[count + i], spare) for i, spare in enumerate(spares) if (count + i) not in used_targets]
    unused.extend(displaced_homes)
    unused.sort(key=lambda pair: pair[0])
    fast_keep = min(len(unused), max(180, int(len(unused) * 0.42)))
    slow_tail = unused[fast_keep:]
    if len(slow_tail) > 175:
        slow_tail = slow_tail[-175:]
    for _, slow_buf in slow_tail:
        _SLOW_HOLD.append(slow_buf)
    del unused, slow_tail
    before = sorted(times[:count])
    after.sort()
    return {
        "count": count, "spares": len(spares), "moved": moved, "payload_bytes": payload_bytes,
        "before": (before[0], median(before), before[-1], sum(before)),
        "after": (after[0], median(after), after[-1], sum(after)),
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
    times = _time_candidates(decode_linear, layers[0], candidates, n, k, ROUNDS, evict_buf)
    ranked = sorted(range(len(candidates)), key=times.__getitem__)
    kept = set(ranked[:count])
    free_fast = [index for index in ranked[:count] if index >= count]
    movers = sorted((index for index in range(count) if index not in kept), key=times.__getitem__, reverse=True)
    moved = 0
    after = list(times[:count])
    used_targets = set()
    slow_cut = median(times)
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


def _place_muserans_group(rans_ext, layers, n: int, k: int, evict_buf: torch.Tensor, rejected_hold: list) -> dict | None:
    homes = [layer._buffers["muse_fp8_packed"] for layer in layers]
    count = len(homes)
    max_bytes = max(h.numel() for h in homes)
    is_small = max_bytes < MIN_PAYLOAD_BYTES
    surplus = ATTN_SURPLUS if is_small else SURPLUS
    wanted = max(1, math.ceil(count * surplus)) if surplus > 0 else 0
    spares_count = _ensure_spares_room(wanted, max_bytes, rejected_hold, allow_slow_trim=not is_small)
    if not is_small and max_bytes > (128 << 20) and LARGE_RANS_SURPLUS > surplus:
        spares_count = max(spares_count, _spares_allowed(math.ceil(count * LARGE_RANS_SURPLUS), max_bytes))
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
        times = _time_calls(run_cand, cand_views, ROUNDS, evict_buf)
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
            times = _time_calls(run_cand, cand_views, ROUNDS, evict_buf)
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
# Prompt-sized (prefill) weights
# --------------------------------------------------------------------------- #
PREFILL_ENABLED = os.environ.get("MUSE_PREFILL_PLACEMENT", "1").strip() not in ("0", "", "false", "no")
PREFILL_SURPLUS = float(os.environ.get("MUSE_PREFILL_PLACEMENT_SURPLUS", "2.0"))
PREFILL_ROUNDS = 5


def _time_calls(run, candidates, rounds: int, evict_buf: torch.Tensor | None = None) -> list[float]:
    """Median time (ms) of run(candidate) per candidate, in shuffled rounds."""
    device = candidates[0].device
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    run(candidates[0])
    if evict_buf is not None:
        _flush_l2(evict_buf)
    torch.cuda.synchronize(device)
    samples: list[list[float]] = [[] for _ in candidates]
    order = list(range(len(candidates)))
    shuffler = random.Random(20261122)
    for _ in range(rounds):
        shuffler.shuffle(order)
        for index in order:
            start.record()
            run(candidates[index])
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end))
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

            use_ilv = hasattr(_swiglu, "uses_ilv") and all(_swiglu.uses_ilv(layer.weight) for layer in gate_layers)
            if use_ilv:
                ref_s_ilv = ref._buffers["paired_marlin_swiglu_scale"]

                def run_gate_ilv(weight, ref=ref, ref_s_ilv=ref_s_ilv, x=x):
                    _swiglu.gemm_ilv(x, weight, ref_s_ilv, ref.weight_global_scale)

                groups.append(("39936x6656_ilv", gate_layers, run_gate_ilv, None, True))
            else:
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
                elif name == "39936x6656_ilv":
                    buf_name = "paired_marlin_swiglu_weight"
                elif name == "bf16_attn_8704x6656":
                    buf_name = "_bf16_qkv_gate_w"
                else:
                    buf_name = None
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
        rejected_hold.clear()
        if HOLD_SLOW_FOR_KV and _SLOW_HOLD:
            ensure_slow_hold_headroom(HOST_HEADROOM_BYTES)
        else:
            _SLOW_HOLD.clear()
            torch.cuda.empty_cache()


def _run_post_decode_registrations(model: torch.nn.Module) -> None:
    try:
        import muse_marlin6_gemv as _m6
        if hasattr(_m6, "register_all_gate_up_swiglu"):
            _m6.register_all_gate_up_swiglu(model)
        if hasattr(_m6, "register_all_oproj_sigmoid"):
            _m6.register_all_oproj_sigmoid(model)
    except Exception as err:  # noqa: BLE001
        import traceback
        logger.warning("Decode6 post-placement registration skipped: %s\n%s", err, traceback.format_exc())


def place_decode_weights(model: torch.nn.Module) -> None:
    """Move decode weights that landed on slow device memory; never raises."""
    if not torch.cuda.is_available():
        return
    if not ENABLED:
        _run_post_decode_registrations(model)
        return
    try:
        from muse_marlin_gemv import decode_linear
    except (ImportError, OSError):
        return
    began = time.perf_counter()
    _SLOW_HOLD.clear()
    rejected_hold: list[torch.Tensor] = []
    _post_reg_done = False
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
        # Place 641 MiB lm_head FIRST (while physical RAM is completely unfragmented by _SLOW_HOLD)
        # using a short-lived /proc/buddyinfo sponge so lm_head lands on contiguous Order-13 (32 MiB) extents,
        # then immediately clear PyTorch's cache before FULL_RAM_SCAN scans 127 MiB slabs.
        if deferred_single and os.environ.get("MUSE_EARLY_LM_HEAD_PLACE", "1").strip() not in ("0", "", "false", "no"):
            lm_bytes = sum(
                layers[0]._buffers[_NAME].numel() * layers[0]._buffers[_NAME].element_size() * (LM_HEAD_SPARES + 2)
                for _, layers in deferred_single
            )
            sponge = _alloc_buddy_sponge(lm_bytes)
            early_hold: list[torch.Tensor] = []
            try:
                for (n, k, *_), layers in deferred_single:
                    stats = _place_group(decode_linear, layers, n, k, evict_buf, early_hold)
                    if stats is None:
                        continue
                    n_groups += 1
                    b, a = stats["before"], stats["after"]
                    saved_ms += b[3] - a[3]
                    gbps = stats["payload_bytes"] / 1e6
                    logger.info(
                        "Early Order-13 weight placement %dx%d: %d weights of %.0f MiB, %d spare buffers, %d moved; "
                        "GEMV ms min/median/max %.4f/%.4f/%.4f -> %.4f/%.4f/%.4f "
                        "(weight-only %.0f/%.0f/%.0f -> %.0f/%.0f/%.0f GB/s), sum %.3f -> %.3f ms",
                        n, k, stats["count"], stats["payload_bytes"] / (1 << 20), stats["spares"], stats["moved"],
                        b[0], b[1], b[2], a[0], a[1], a[2],
                        gbps / b[0], gbps / b[1], gbps / b[2], gbps / a[0], gbps / a[1], gbps / a[2], b[3], a[3],
                    )
                deferred_single = []
            except Exception as err:  # noqa: BLE001
                logger.warning("Early lm_head placement skipped (will retry deferred): %s", err)
            finally:
                early_hold.clear()
                del sponge
                torch.cuda.empty_cache()
        for (n, k, *_), layers in multi_nvfp4:
            try:
                stats = None
                if FULL_RAM_SCAN and (n, k) == (39936, 6656) and len(layers) >= 32:
                    stats = _place_group_full_ram_scan(decode_linear, layers, n, k, evict_buf)
                    if stats is not None and PLACE_SCALES6:
                        try:
                            import muse_marlin6_gemv as _m6
                            ilv_set = getattr(_m6, "_DOWN_ILV_SCALES", None)
                            lm_split2_set = getattr(_m6, "_LM_SPLIT2_SCALES", None)
                            lm_split2_cache = getattr(_m6, "_LM_SPLIT2_CACHE", None)
                            lm_pad128_set = getattr(_m6, "_LM_PAD128_SCALES", None)
                            lm_pad128_cache = getattr(_m6, "_LM_PAD128_CACHE", None)
                        except (ImportError, OSError):
                            ilv_set = None
                            lm_split2_set = None
                            lm_split2_cache = None
                            lm_pad128_set = None
                            lm_pad128_cache = None
                        moved_s6, bytes_s6 = 0, 0
                        for m in model.modules():
                            s6 = m._buffers.get("_marlin_decode_scales6")
                            if s6 is not None and s6.is_cuda and s6.numel() * s6.element_size() >= (64 << 10):
                                try:
                                    old_ptr = s6.data_ptr()
                                    if s6.dim() == 2 and s6.shape[1] == 32 and (s6.shape[0] + 64) * 32 < (1 << 20):
                                        dst_base = torch.empty((s6.shape[0] + 64, 32), dtype=s6.dtype, device=s6.device)
                                        dst = dst_base[: s6.shape[0]]
                                    elif (
                                        s6.dim() == 2
                                        and s6.shape[1] == 32
                                        and getattr(m, "output_size_per_partition", None) == 39936
                                        and getattr(m, "input_size_per_partition", None) == 6656
                                    ):
                                        # Reserve +128 KiB fast-pool headroom so register_all_gate_up_swiglu
                                        # always updates paired scales6 in-place without spilling to slow DRAM.
                                        dst_base = torch.empty((s6.shape[0] + 4096, 32), dtype=s6.dtype, device=s6.device)
                                        dst = dst_base[: s6.shape[0]]
                                    else:
                                        dst = torch.empty_like(s6)
                                    dst.copy_(s6)
                                    if torch.equal(dst, s6):
                                        m._buffers["_marlin_decode_scales6"] = dst
                                        if ilv_set is not None and old_ptr in ilv_set:
                                            ilv_set.discard(old_ptr)
                                            ilv_set.add(dst.data_ptr())
                                        if lm_split2_set is not None and old_ptr in lm_split2_set:
                                            lm_split2_set.discard(old_ptr)
                                            lm_split2_set.add(dst.data_ptr())
                                        if lm_split2_cache is not None and old_ptr in lm_split2_cache:
                                            lm_split2_cache[dst.data_ptr()] = lm_split2_cache.pop(old_ptr)
                                        if lm_pad128_set is not None and old_ptr in lm_pad128_set:
                                            lm_pad128_set.discard(old_ptr)
                                            lm_pad128_set.add(dst.data_ptr())
                                        if lm_pad128_cache is not None and old_ptr in lm_pad128_cache:
                                            lm_pad128_cache[dst.data_ptr()] = lm_pad128_cache.pop(old_ptr)
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
            try:
                from vllm.model_executor.models.muse_glimmer import (
                    _BF16_DUAL_RANS_M,
                    _BF16_DUAL_RANS_M_GATE_PTRS,
                    update_bf16_dual_rans_m,
                )
            except Exception:
                _BF16_DUAL_RANS_M = {}
                _BF16_DUAL_RANS_M_GATE_PTRS = set()
                update_bf16_dual_rans_m = None
            place_bf16_m1_dual = os.environ.get("MUSE_PLACE_BF16_M1_DUAL", "0").strip() not in ("0", "", "false", "no")
            if _bf16_lib is not None and hasattr(_bf16_lib, "muse_bf16_rans_gemv"):
                for (n, k), layers in sorted(
                    attn_bf16_groups.items(),
                    key=lambda kv: -(len(kv[1]) * max(l._buffers["muse_bf16_lane_weight"].numel() for l in kv[1])),
                ):
                    bf16_layers = [
                        l for l in layers
                        if place_bf16_m1_dual
                        or getattr(l, "muse_bf16_lane_index", None) is None
                        or (
                            l.muse_bf16_lane_index.data_ptr() not in _BF16_DUAL_RANS_M
                            and l.muse_bf16_lane_index.data_ptr() not in _BF16_DUAL_RANS_M_GATE_PTRS
                        )
                    ]
                    if not bf16_layers:
                        continue
                    try:
                        stats = _place_bf16_rans_group(_bf16_lib, bf16_layers, n, k, evict_buf, rejected_hold)
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
            # Also place multi-M BF16 rANS MMA blobs (muse_bf16_rans_m) for verify-width target BF16 attention
            try:
                import muse_bfx_ext as _bfx_ext
                if hasattr(_bfx_ext, "rans_gemm_out") and hasattr(_bf16_mod, "get_m_rans"):
                    for (n, k), layers in attn_bf16_groups.items():
                        m_layers = [
                            l for l in layers
                            if l._buffers.get("muse_bf16_rans_m") is not None
                            and l.muse_bf16_lane_index.data_ptr() not in _BF16_DUAL_RANS_M
                            and l.muse_bf16_lane_index.data_ptr() not in _BF16_DUAL_RANS_M_GATE_PTRS
                        ]
                        if not m_layers:
                            continue
                        dev = m_layers[0]._buffers["muse_bf16_rans_m"].device
                        max_len = max(l._buffers["muse_bf16_rans_m"].numel() for l in m_layers)
                        x7 = torch.zeros((7, k), dtype=torch.bfloat16, device=dev)
                        out7 = torch.empty((7, n), dtype=torch.bfloat16, device=dev)
                        bexps = [_bf16_mod.get_m_rans(l.muse_bf16_lane_index)[1] for l in m_layers]
                        def _tm(buf, be):
                            ts = []
                            for _ in range(4):
                                _flush_l2(evict_buf)
                                e0 = torch.cuda.Event(enable_timing=True)
                                e1 = torch.cuda.Event(enable_timing=True)
                                e0.record()
                                _bfx_ext.rans_gemm_out(x7, buf, out7, n, k, be)
                                e1.record()
                                torch.cuda.synchronize(dev)
                                ts.append(e0.elapsed_time(e1))
                            ts.sort()
                            return ts[len(ts) // 2]
                        _bfx_ext.rans_gemm_out(x7, m_layers[0]._buffers["muse_bf16_rans_m"], out7, n, k, bexps[0])
                        torch.cuda.synchronize(dev)
                        ht = [_tm(l._buffers["muse_bf16_rans_m"], bexps[i]) for i, l in enumerate(m_layers)]
                        ref_b = m_layers[0]._buffers["muse_bf16_rans_m"]
                        spares, st_list = [], []
                        for si in range(min(15, len(m_layers) * 2)):
                            sb = torch.empty(max_len, dtype=torch.uint8, device=dev)
                            sb[:ref_b.numel()].copy_(ref_b)
                            st_list.append((_tm(sb[:ref_b.numel()], bexps[0]), si))
                            spares.append(sb)
                        st_list.sort(key=lambda pair: pair[0])
                        slow_cut = median(ht + [t for t, _ in st_list])
                        at = list(ht)
                        used_spares: set[int] = set()
                        sp_i = moved_m = 0
                        for hi in sorted(range(len(m_layers)), key=lambda idx: -ht[idx]):
                            if sp_i >= len(st_list):
                                break
                            t_sp, si = st_list[sp_i]
                            if t_sp < ht[hi] * 0.99:
                                h = m_layers[hi]._buffers["muse_bf16_rans_m"]
                                dst = spares[si][:h.numel()]
                                dst.copy_(h)
                                tn = _tm(dst, bexps[hi])
                                if tn < ht[hi]:
                                    m_layers[hi]._buffers["muse_bf16_rans_m"] = dst
                                    _bf16_mod.update_m_rans(m_layers[hi].muse_bf16_lane_index, dst)
                                    used_spares.add(si)
                                    rejected_hold.append(h)
                                    if ht[hi] > slow_cut:
                                        _SLOW_HOLD.append(h)
                                    at[hi] = tn
                                    moved_m += 1
                                sp_i += 1
                        for t_sp, si in st_list:
                            if si not in used_spares:
                                rejected_hold.append(spares[si])
                                if t_sp > slow_cut:
                                    _SLOW_HOLD.append(spares[si])
                        del spares, x7, out7
                        logger.info(
                            "Decode BF16 multi-M rANS placement %dx%d: %d blobs (~%.0f MiB), %d moved; sum %.3f -> %.3f ms",
                            n, k, len(m_layers), max_len / (1 << 20), moved_m, sum(ht), sum(at),
                        )
                if (
                    hasattr(_bfx_ext, "rans_gemm_dual_out")
                    and update_bf16_dual_rans_m is not None
                    and _BF16_DUAL_RANS_M
                ):
                    dual_mods = [
                        m for m in model.modules()
                        if m._buffers.get("_bf16_qkv_gate_rans_m") is not None
                        and hasattr(m, "qkv_proj")
                        and getattr(m.qkv_proj, "muse_bf16_lane_index", None) is not None
                        and m.qkv_proj.muse_bf16_lane_index.data_ptr() in _BF16_DUAL_RANS_M
                    ]
                    if dual_mods:
                        dev = dual_mods[0]._buffers["_bf16_qkv_gate_rans_m"].device
                        max_len = max(m._buffers["_bf16_qkv_gate_rans_m"].numel() for m in dual_mods)
                        x16 = torch.zeros((16, 6656), dtype=torch.bfloat16, device=dev)
                        oq16 = torch.empty((16, 4608), dtype=torch.bfloat16, device=dev)
                        og16 = torch.empty((16, 4096), dtype=torch.bfloat16, device=dev)
                        bexps = [
                            _BF16_DUAL_RANS_M[m.qkv_proj.muse_bf16_lane_index.data_ptr()][1]
                            for m in dual_mods
                        ]
                        def _tm_dual(buf, be):
                            ts = []
                            for _ in range(4):
                                _flush_l2(evict_buf)
                                e0 = torch.cuda.Event(enable_timing=True)
                                e1 = torch.cuda.Event(enable_timing=True)
                                e0.record()
                                _bfx_ext.rans_gemm_dual_out(x16, buf, oq16, og16, 4608, 4096, 6656, be)
                                e1.record()
                                torch.cuda.synchronize(dev)
                                ts.append(e0.elapsed_time(e1))
                            ts.sort()
                            return ts[len(ts) // 2]
                        _bfx_ext.rans_gemm_dual_out(
                            x16, dual_mods[0]._buffers["_bf16_qkv_gate_rans_m"], oq16, og16, 4608, 4096, 6656, bexps[0]
                        )
                        torch.cuda.synchronize(dev)
                        ht = [_tm_dual(m._buffers["_bf16_qkv_gate_rans_m"], bexps[i]) for i, m in enumerate(dual_mods)]
                        ref_b = dual_mods[0]._buffers["_bf16_qkv_gate_rans_m"]
                        spares, st_list = [], []
                        for si in range(min(15, len(dual_mods) * 2)):
                            sb = torch.empty(max_len, dtype=torch.uint8, device=dev)
                            sb[:ref_b.numel()].copy_(ref_b)
                            st_list.append((_tm_dual(sb[:ref_b.numel()], bexps[0]), si))
                            spares.append(sb)
                        st_list.sort(key=lambda pair: pair[0])
                        slow_cut = median(ht + [t for t, _ in st_list])
                        at = list(ht)
                        used_spares_d: set[int] = set()
                        sp_i = moved_d = 0
                        for hi in sorted(range(len(dual_mods)), key=lambda idx: -ht[idx]):
                            if sp_i >= len(st_list):
                                break
                            t_sp, si = st_list[sp_i]
                            if t_sp < ht[hi] * 0.99:
                                h = dual_mods[hi]._buffers["_bf16_qkv_gate_rans_m"]
                                dst = spares[si][:h.numel()]
                                dst.copy_(h)
                                tn = _tm_dual(dst, bexps[hi])
                                if tn < ht[hi]:
                                    dual_mods[hi]._buffers["_bf16_qkv_gate_rans_m"] = dst
                                    update_bf16_dual_rans_m(dual_mods[hi].qkv_proj.muse_bf16_lane_index, dst)
                                    used_spares_d.add(si)
                                    rejected_hold.append(h)
                                    if ht[hi] > slow_cut:
                                        _SLOW_HOLD.append(h)
                                    at[hi] = tn
                                    moved_d += 1
                                sp_i += 1
                        for t_sp, si in st_list:
                            if si not in used_spares_d:
                                rejected_hold.append(spares[si])
                                if t_sp > slow_cut:
                                    _SLOW_HOLD.append(spares[si])
                        del spares, x16, oq16, og16
                        logger.info(
                            "Decode BF16 stacked qkv+gate rANS placement 8704x6656: %d blobs (~%.0f MiB), %d moved; sum %.3f -> %.3f ms",
                            len(dual_mods), max_len / (1 << 20), moved_d, sum(ht), sum(at),
                        )
            except Exception as err:  # noqa: BLE001
                logger.warning("Decode BF16 multi-M rANS placement skipped: %s", err)
        if PLACE_ATTN and attn_rans_groups and _rans_ext is not None:
            for (n, k), layers in sorted(
                attn_rans_groups.items(),
                key=lambda kv: -(len(kv[1]) * max(l._buffers["muse_fp8_packed"].numel() for l in kv[1])),
            ):
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
        _run_post_decode_registrations(model)
        _post_reg_done = True
    except Exception as err:  # noqa: BLE001 - the load must go on
        logger.warning("Decode weight placement failed and was skipped: %s", err)
    finally:
        rejected_hold.clear()
        if not PREFILL_ENABLED:
            _SLOW_HOLD.clear()
            torch.cuda.empty_cache()
    if not _post_reg_done:
        _run_post_decode_registrations(model)
