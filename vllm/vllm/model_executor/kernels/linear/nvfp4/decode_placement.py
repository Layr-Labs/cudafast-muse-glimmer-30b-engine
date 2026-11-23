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
SURPLUS = float(os.environ.get("MUSE_DECODE_PLACEMENT_SURPLUS", "3.5"))
LM_HEAD_SPARES = int(os.environ.get("MUSE_LM_HEAD_SPARES", "16"))
ROUNDS = 5
# Weights that fit in L2 several times over are not bandwidth-bound per call.
MIN_PAYLOAD_BYTES = 48 << 20
# Host memory that must stay available while the spares exist.
HOST_HEADROOM_BYTES = 20 << 30
_NAME = "_marlin_decode_weight"
_L2_FLUSH_INTS = 16 << 20  # 64 MiB (> 2.5x GB10's 24 MiB L2)


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


def _place_group(decode_linear, layers, n: int, k: int, evict_buf: torch.Tensor, rejected_hold: list) -> dict | None:
    homes = [layer._buffers[_NAME] for layer in layers]
    count = len(homes)
    payload_bytes = homes[0].numel() * homes[0].element_size()
    wanted = max(LM_HEAD_SPARES if count == 1 else 1, math.ceil(count * SURPLUS)) if SURPLUS > 0 else 0
    spares_count = _spares_allowed(wanted, payload_bytes)
    if spares_count < count and rejected_hold:
        rejected_hold.clear()
        torch.cuda.empty_cache()
        spares_count = _spares_allowed(wanted, payload_bytes)
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
    times = _time_candidates(decode_linear, layers[0], candidates, n, k, ROUNDS, evict_buf)
    ranked = sorted(range(len(candidates)), key=times.__getitem__)
    kept = set(ranked[:count])
    free_fast = [index for index in ranked[:count] if index >= count]
    movers = sorted((index for index in range(count) if index not in kept), key=times.__getitem__, reverse=True)
    moved = 0
    after = list(times[:count])
    used_targets = set()
    pool_med = median(times)
    hold_all = os.environ.get("MUSE_DECODE_HOLD_ALL", "1") != "0"
    for home_index, target_index in zip(movers, free_fast):
        target, source = candidates[target_index], homes[home_index]
        target.copy_(source)
        if not torch.equal(target, source):
            continue  # keep the loader's buffer
        layers[home_index]._buffers[_NAME] = target
        used_targets.add(target_index)
        if hold_all or times[home_index] > pool_med:
            rejected_hold.append(source)
        after[home_index] = times[target_index]
        moved += 1
    for i, spare in enumerate(spares):
        if (count + i) not in used_targets and (hold_all or times[count + i] > pool_med):
            rejected_hold.append(spare)
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
    wanted = max(1, math.ceil(count * SURPLUS)) if SURPLUS > 0 else 0
    spares_count = _spares_allowed(wanted, max_bytes)
    if spares_count < count and rejected_hold:
        rejected_hold.clear()
        torch.cuda.empty_cache()
        spares_count = _spares_allowed(wanted, max_bytes)
    if spares_count <= 0:
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
    pool_med = median(times)
    hold_all = os.environ.get("MUSE_DECODE_HOLD_ALL", "1") != "0"
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
        if hold_all or times[home_index] > pool_med:
            rejected_hold.append(old)
        after[home_index] = times[target_index]
        moved += 1
    for i, spare in enumerate(spares):
        if (count + i) not in used_targets and (hold_all or times[count + i] > pool_med):
            rejected_hold.append(spare)
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
PREFILL_SURPLUS = float(os.environ.get("MUSE_PREFILL_PLACEMENT_SURPLUS", "1.5"))
PREFILL_ROUNDS = 4


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
    spares = []
    try:
        for _ in range(_spares_allowed(wanted, payload_bytes)):
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

            def run_fp8_down(weight, x=x, sa=sa, sb=sb, cfg_dn=cfg_dn):
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
        for name, layers, run, on_move, fast in groups:
            try:
                buf_name = "paired_fp8_swiglu_weight" if name == "fp8_swiglu_39936x6656" else None
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
        torch.cuda.empty_cache()


def place_decode_weights(model: torch.nn.Module) -> None:
    """Move decode weights that landed on slow device memory; never raises."""
    if not ENABLED or not torch.cuda.is_available():
        return
    try:
        from muse_marlin_gemv import decode_linear
    except (ImportError, OSError):
        return
    began = time.perf_counter()
    rejected_hold: list[torch.Tensor] = []
    try:
        _drop_model_page_cache()
        evict_buf = torch.empty((_L2_FLUSH_INTS,), dtype=torch.int32, device="cuda")
        groups: dict[tuple, list] = {}
        rans_groups: dict[tuple[int, int], list] = {}
        for layer in model.modules():
            bufs = getattr(layer, "_buffers", {})
            weight = bufs.get(_NAME)
            if (
                weight is not None
                and weight.is_cuda
                and weight.is_contiguous()
                and weight.numel() * weight.element_size() >= MIN_PAYLOAD_BYTES
                and getattr(layer, "_marlin_decode_scales", None) is not None
                and getattr(layer, "weight_global_scale", None) is not None
                and getattr(layer, "workspace", None) is not None
            ):
                n, k = layer.output_size_per_partition, layer.input_size_per_partition
                groups.setdefault((n, k, tuple(weight.shape), weight.dtype, weight.device), []).append(layer)
            rblob = bufs.get("muse_fp8_packed")
            if (
                rblob is not None
                and rblob.is_cuda
                and rblob.dtype == torch.uint8
                and rblob.dim() == 1
                and rblob.is_contiguous()
                and rblob.numel() >= MIN_PAYLOAD_BYTES
                and getattr(layer, "weight_scale", None) is not None
            ):
                n = getattr(layer, "output_size_per_partition", 0)
                k = getattr(layer, "input_size_per_partition", 0)
                if n > 0 and k > 0:
                    rans_groups.setdefault((n, k), []).append(layer)
        saved_ms = 0.0
        n_groups = 0
        # Place largest / highest-impact groups first (lm_head, gate_up, down) while holding rejected slow buffers.
        ordered_groups = sorted(
            groups.items(),
            key=lambda kv: -(kv[0][0] * kv[0][1] if len(kv[1]) == 1 else len(kv[1]) * kv[1][0]._buffers[_NAME].numel()),
        )
        for (n, k, *_), layers in ordered_groups:
            try:
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
        if rans_groups:
            try:
                import muse_rans_ext as _rans_ext
            except (ImportError, OSError):
                _rans_ext = None
            if _rans_ext is not None:
                ordered_rans = sorted(
                    rans_groups.items(),
                    key=lambda kv: -max(h.numel() for h in (l._buffers["muse_fp8_packed"] for l in kv[1])),
                )
                for (n, k), layers in ordered_rans:
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
        logger.info(
            "Decode weight placement: %d groups, timed GEMV sum reduced by %.3f ms per token, %.2f s",
            n_groups, saved_ms, time.perf_counter() - began,
        )
    except Exception as err:  # noqa: BLE001 - the load must go on
        logger.warning("Decode weight placement failed and was skipped: %s", err)
    finally:
        rejected_hold.clear()
        torch.cuda.empty_cache()
