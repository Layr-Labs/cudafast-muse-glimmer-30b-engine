# SPDX-License-Identifier: Apache-2.0
"""Where the one-row NVFP4 decode weights live in device memory.

A one-row (decode) Marlin GEMV streams its whole weight once per token at the
DRAM limit, and on a unified-memory GPU (GB10) the rate it reaches is a
property of the physical memory the weight happens to sit on: the same bytes in
two allocations differ by several percent, the difference follows the buffer
and not its contents, and it persists for the life of the process. Which
buffers are fast is different on every boot.

``place_decode_weights`` runs once after the weights are loaded. For every
group of same-shaped decode weights it allocates spare buffers of the same
size, times the production tiled GEMV on every home and every spare (the
kernel's own address pattern, contents irrelevant to its timing), keeps the
fastest buffers and moves the weights that sat on slow ones. A move is a copy
of the same bytes followed by a bitwise comparison, so nothing a kernel
computes can change; the prompt-sized (prefill) weights are never touched.

Best effort throughout: any error, a short host, or the switch below leaves
every weight where the loader put it.

  MUSE_DECODE_PLACEMENT=0             no placement step
  MUSE_DECODE_PLACEMENT_SURPLUS=<r>   spare buffers per decode weight (default 1.0)
"""

from __future__ import annotations

import math
import os
import random
import time
from statistics import median

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("MUSE_DECODE_PLACEMENT", "1").strip() not in ("0", "", "false", "no")
SURPLUS = float(os.environ.get("MUSE_DECODE_PLACEMENT_SURPLUS", "1.0"))
ROUNDS = 5
# Weights that fit in L2 several times over are not bandwidth-bound per call.
MIN_PAYLOAD_BYTES = 48 << 20
# Host memory that must stay available while the spares exist.
HOST_HEADROOM_BYTES = 24 << 30
_NAME = "_marlin_decode_weight"


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


def _time_candidates(decode_linear, layer, candidates, n: int, k: int, rounds: int) -> list[float]:
    """Median GEMV time (ms) of the tiled decode kernel on each candidate buffer."""
    device = candidates[0].device
    # A fixed finite row; the kernel's timing does not depend on values.
    x = ((torch.arange(k, device=device) % 7).float() - 3.0).to(torch.bfloat16).reshape(1, k)
    scales, global_scale, workspace = layer._marlin_decode_scales, layer.weight_global_scale, layer.workspace
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    decode_linear(x, candidates[0], scales, global_scale, workspace, n, k)
    torch.cuda.synchronize(device)
    samples: list[list[float]] = [[] for _ in candidates]
    order = list(range(len(candidates)))
    shuffler = random.Random(20261122)
    for _ in range(rounds):
        shuffler.shuffle(order)
        for index in order:
            start.record()
            decode_linear(x, candidates[index], scales, global_scale, workspace, n, k)
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end))
    return [median(values) for values in samples]


def _place_group(decode_linear, layers, n: int, k: int) -> dict | None:
    homes = [layer._buffers[_NAME] for layer in layers]
    count = len(homes)
    payload_bytes = homes[0].numel() * homes[0].element_size()
    wanted = max(2 if count == 1 else 1, math.ceil(count * SURPLUS)) if SURPLUS > 0 else 0
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
    times = _time_candidates(decode_linear, layers[0], candidates, n, k, ROUNDS)
    ranked = sorted(range(len(candidates)), key=times.__getitem__)
    kept = set(ranked[:count])
    free_fast = [index for index in ranked[:count] if index >= count]
    movers = sorted((index for index in range(count) if index not in kept), key=times.__getitem__, reverse=True)
    moved = 0
    after = list(times[:count])
    for home_index, target_index in zip(movers, free_fast):
        target, source = candidates[target_index], homes[home_index]
        target.copy_(source)
        if not torch.equal(target, source):
            continue  # keep the loader's buffer
        layers[home_index]._buffers[_NAME] = target
        after[home_index] = times[target_index]
        moved += 1
    before = sorted(times[:count])
    after.sort()
    return {
        "count": count, "spares": len(spares), "moved": moved, "payload_bytes": payload_bytes,
        "before": (before[0], median(before), before[-1], sum(before)),
        "after": (after[0], median(after), after[-1], sum(after)),
    }


def place_decode_weights(model: torch.nn.Module) -> None:
    """Move decode weights that landed on slow device memory; never raises."""
    if not ENABLED or not torch.cuda.is_available():
        return
    try:
        from muse_marlin_gemv import decode_linear
    except (ImportError, OSError):
        return
    began = time.perf_counter()
    try:
        groups: dict[tuple, list] = {}
        for layer in model.modules():
            weight = getattr(layer, "_buffers", {}).get(_NAME)
            if (
                weight is None
                or not weight.is_cuda
                or not weight.is_contiguous()
                or weight.numel() * weight.element_size() < MIN_PAYLOAD_BYTES
                or getattr(layer, "_marlin_decode_scales", None) is None
                or getattr(layer, "weight_global_scale", None) is None
                or getattr(layer, "workspace", None) is None
            ):
                continue
            n, k = layer.output_size_per_partition, layer.input_size_per_partition
            groups.setdefault((n, k, tuple(weight.shape), weight.dtype, weight.device), []).append(layer)
        saved_ms = 0.0
        for (n, k, *_), layers in groups.items():
            try:
                stats = _place_group(decode_linear, layers, n, k)
            except Exception as err:  # noqa: BLE001 - this group stays where it was
                logger.warning("Decode weight placement skipped for %dx%d: %s", n, k, err)
                continue
            if stats is None:
                continue
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
            len(groups), saved_ms, time.perf_counter() - began,
        )
    except Exception as err:  # noqa: BLE001 - the load must go on
        logger.warning("Decode weight placement failed and was skipped: %s", err)
