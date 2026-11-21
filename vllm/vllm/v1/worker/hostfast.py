# SPDX-License-Identifier: Apache-2.0
"""Host-side trims for the single-request engine loop.

Nothing here changes a kernel, a shape, a batch or a value the GPU reads.

* Garbage collection. Automatic collection is off while the engine has work,
  so a collection never fires inside the scheduling and launch path, and it is
  back on when the engine goes idle.
* CPU placement. ``pin_to_fast_cpus`` restricts the calling process to the
  highest-capacity cores when the cores differ (the big cluster on a
  heterogeneous CPU). It fails soft: a sandbox that refuses affinity changes,
  or a CPU whose cores are all alike, leaves everything as it was.

Switches, read once at import (all default on):
  VLLM_HOSTFAST_GC=0    automatic GC stays on
  VLLM_HOSTFAST_PIN=0   no CPU affinity change
"""

from __future__ import annotations

import gc
import os


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip() not in ("0", "", "false", "no")


GC = _flag("VLLM_HOSTFAST_GC", "1")
PIN = _flag("VLLM_HOSTFAST_PIN", "1")

# --------------------------------------------------------------------------- #
# GC
# --------------------------------------------------------------------------- #

_gc_paused = False


def gc_busy() -> bool:
    """The engine has work: stop automatic collection. True when this call
    is the one that stopped it."""
    global _gc_paused
    if GC and not _gc_paused and gc.isenabled():
        gc.disable()
        _gc_paused = True
        return True
    return False


def gc_idle() -> None:
    """The engine is idle: give collection back to the interpreter."""
    global _gc_paused
    if _gc_paused:
        gc.enable()
        _gc_paused = False


# --------------------------------------------------------------------------- #
# CPU placement
# --------------------------------------------------------------------------- #


def _read_int(path: str) -> int | None:
    try:
        with open(path) as fp:
            return int(fp.read().strip())
    except (OSError, ValueError):
        return None


def fast_cpus(
    allowed: set[int],
    sysfs: str = "/sys/devices/system/cpu",
    ratio: float = 0.85,
) -> set[int]:
    """The allowed CPUs whose ``cpu_capacity`` (else ``cpuinfo_max_freq``) is
    within ``ratio`` of the largest. Empty when the CPUs do not differ (or
    nothing is readable), so the caller leaves the affinity alone."""
    for leaf in ("cpu_capacity", "cpufreq/cpuinfo_max_freq"):
        values = {c: _read_int(f"{sysfs}/cpu{c}/{leaf}") for c in allowed}
        known = {c: v for c, v in values.items() if v is not None and v > 0}
        if len(known) != len(allowed) or not known:
            continue
        top = max(known.values())
        fast = {c for c, v in known.items() if v >= ratio * top}
        if fast and fast != allowed:
            return fast
    return set()


def format_cpus(cpus: list[int]) -> str:
    """``[5, 6, 7, 9]`` -> ``"5-7,9"``."""
    runs: list[tuple[int, int]] = []
    for cpu in sorted(cpus):
        if runs and cpu == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], cpu)
        else:
            runs.append((cpu, cpu))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def pin_to_fast_cpus() -> list[int]:
    """Restrict every thread of this process (and so the threads and processes
    it starts afterwards) to the fast cores. Returns the CPU list applied, empty
    when nothing changed. Call it after startup: compile workers and other
    startup parallelism keep every core."""
    if not PIN or not hasattr(os, "sched_setaffinity"):
        return []
    try:
        allowed = set(os.sched_getaffinity(0))
        fast = fast_cpus(allowed)
        if not fast:
            return []
        try:
            tids = [int(t) for t in os.listdir("/proc/self/task")]
        except OSError:
            tids = [0]
        for tid in tids:
            try:
                os.sched_setaffinity(tid, fast)
            except OSError:
                pass  # a thread that exited meanwhile
        os.sched_setaffinity(0, fast)
        return sorted(fast)
    except OSError:
        return []
