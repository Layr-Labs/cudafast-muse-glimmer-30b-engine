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

Switches, read once at import (default on unless noted):
  VLLM_HOSTFAST_GC=0    automatic GC stays on
  VLLM_HOSTFAST_PIN=0   no CPU affinity change
  VLLM_HOSTFAST_TICKLE=1   start the SCHED_IDLE idle ticklers (default off)
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip() not in ("0", "", "false", "no")


GC = _flag("VLLM_HOSTFAST_GC", "1")
PIN = _flag("VLLM_HOSTFAST_PIN", "1")
SINGLE_CLUSTER = _flag("VLLM_HOSTFAST_SINGLE_CLUSTER", "1")
# The SCHED_IDLE idle ticklers (start_idle_tickle) are off by default in this
# tree: they keep the pinned cores out of deep idle between decode steps,
# which only changes host power state, never a value the GPU computes.
# VLLM_HOSTFAST_TICKLE=1 turns them on.
TICKLE = _flag("VLLM_HOSTFAST_TICKLE", "0")

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


def _read_str(path: str) -> str | None:
    try:
        with open(path) as fp:
            return fp.read().strip()
    except OSError:
        return None


def _pick_single_cluster(fast: set[int], known: dict[int, int], sysfs: str) -> set[int]:
    """If ``fast`` spans multiple L3/CPU clusters, pick the highest-capacity cluster."""
    if not SINGLE_CLUSTER or len(fast) < 2:
        return fast
    by_cluster: dict[str, set[int]] = {}
    for c in fast:
        cl = _read_str(f"{sysfs}/cpu{c}/topology/cluster_cpus_list")
        if not cl:
            return fast
        by_cluster.setdefault(cl, set()).add(c)
    if len(by_cluster) <= 1:
        return fast
    return max(
        by_cluster.values(),
        key=lambda grp: (sum(known.get(c, 0) for c in grp), len(grp), max(grp)),
    )


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
            return _pick_single_cluster(fast, known, sysfs)
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


# --------------------------------------------------------------------------- #
# Deep-idle prevention (pure-C SCHED_IDLE 1 ms nanosleep tickler via libl2pf)
# --------------------------------------------------------------------------- #

_tickle_started = False


def start_idle_tickle(cpus: list[int]) -> int:
    """Spawn one pure-C SCHED_IDLE 1 ms nanosleep pthread per pinned CPU (zero
    Python/GIL involvement) so ARM acpi_idle does not enter deep LPI-3 (~433 us
    exit) between decode steps. Fails soft; returns thread count started."""
    global _tickle_started
    if not TICKLE or _tickle_started or not cpus or sys.platform != "linux":
        return 0
    try:
        import l2pf

        lib_path = os.path.join(os.path.dirname(os.path.abspath(l2pf.__file__)), "libl2pf.so")
        lib = ctypes.CDLL(lib_path)
        fn = getattr(lib, "l2pf_start_idle_tickle", None)
        if fn is None:
            return 0
        fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        fn.restype = ctypes.c_int
        arr = (ctypes.c_int * len(cpus))(*cpus)
        started = int(fn(arr, len(cpus)))
        if started > 0:
            _tickle_started = True
        return started
    except Exception:
        return 0
