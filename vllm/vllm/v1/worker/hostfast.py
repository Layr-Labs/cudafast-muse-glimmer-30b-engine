# SPDX-License-Identifier: Apache-2.0
"""Host-side trims for the single-request engine loop.

Nothing here changes a kernel, a shape, a batch or a value the GPU reads.

* Garbage collection. Automatic collection is off while the engine has work,
  so a collection never fires inside the scheduling and launch path, and it is
  back on when the engine goes idle.
* CPU placement. ``pin_to_fast_cpus`` restricts the calling process to the
  highest-capacity 16 MiB-L3 cluster when the cores differ (the big X925
  cluster on a heterogeneous CPU), and ``pin_main_thread`` pins the critical
  main thread to 1 core ``c`` on that cluster alongside a single pure-C
  ``SCHED_IDLE`` tickler thread on core ``c``. Everything fails soft: a sandbox
  that refuses affinity changes, or a CPU whose cores are all alike, leaves
  everything as it was.

Switches, read once at import (all default on):
  VLLM_HOSTFAST_GC=0              automatic GC stays on
  VLLM_HOSTFAST_PIN=0             no CPU affinity change
  VLLM_HOSTFAST_SINGLE_CLUSTER=0  keep all fast clusters instead of one
  VLLM_HOSTFAST_PIN_MAIN=0        keep main thread across the whole fast cluster
  VLLM_HOSTFAST_TICKLE=0          do not start the SCHED_IDLE deep-idle tickler
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys
import threading

_L3_16_MIB = 16 * 1024 * 1024


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip() not in ("0", "", "false", "no")


GC = _flag("VLLM_HOSTFAST_GC", "1")
PIN = _flag("VLLM_HOSTFAST_PIN", "1")
SINGLE_CLUSTER = _flag("VLLM_HOSTFAST_SINGLE_CLUSTER", "1")
PIN_MAIN = _flag("VLLM_HOSTFAST_PIN_MAIN", "1")
TICKLE = _flag("VLLM_HOSTFAST_TICKLE", "1")

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
            val = fp.read().strip()
            return val or None
    except OSError:
        return None


def _parse_cache_size(raw: str | None) -> int:
    """Parse sysfs cache size strings such as ``'16384K'`` or ``'16M'`` into bytes."""
    if not raw:
        return 0
    s = raw.strip()
    if not s:
        return 0
    mult = 1
    tail = s[-1]
    if tail in ("K", "k"):
        mult = 1024
        s = s[:-1]
    elif tail in ("M", "m"):
        mult = 1024 * 1024
        s = s[:-1]
    elif tail in ("G", "g"):
        mult = 1024 * 1024 * 1024
        s = s[:-1]
    try:
        val = int(s.strip())
    except ValueError:
        return 0
    return val * mult if val > 0 else 0


def _pick_single_cluster(
    fast: set[int],
    known: dict[int, int],
    sysfs: str,
    *,
    require_asymmetric_l3: bool = False,
) -> set[int]:
    """If ``fast`` spans multiple L3/CPU clusters, dynamically select the
    16 MiB-L3 cluster (falling back to the largest L3 / highest-capacity
    cluster when ``cache/index3/size`` is absent)."""
    if not SINGLE_CLUSTER or len(fast) < 2:
        return set() if require_asymmetric_l3 else fast
    by_cluster: dict[str, set[int]] = {}
    for c in fast:
        cl = _read_str(f"{sysfs}/cpu{c}/topology/cluster_cpus_list") or _read_str(
            f"{sysfs}/cpu{c}/cache/index3/shared_cpu_list"
        )
        if not cl:
            return set() if require_asymmetric_l3 else fast
        by_cluster.setdefault(cl, set()).add(c)
    if len(by_cluster) <= 1:
        return set() if require_asymmetric_l3 else fast
    l3_by_key = {
        cl: max(
            (
                _parse_cache_size(_read_str(f"{sysfs}/cpu{c}/cache/index3/size"))
                for c in grp
            ),
            default=0,
        )
        for cl, grp in by_cluster.items()
    }
    if require_asymmetric_l3 and len(set(l3_by_key.values())) <= 1:
        return set()
    best_cl = max(
        by_cluster,
        key=lambda cl: (
            l3_by_key[cl] == _L3_16_MIB,
            l3_by_key[cl],
            sum(known.get(c, 0) for c in by_cluster[cl]),
            len(by_cluster[cl]),
            max(by_cluster[cl]),
        ),
    )
    return by_cluster[best_cl]


def fast_cpus(
    allowed: set[int],
    sysfs: str = "/sys/devices/system/cpu",
    ratio: float = 0.85,
) -> set[int]:
    """The allowed CPUs whose ``cpu_capacity`` (else ``cpuinfo_max_freq``) is
    within ``ratio`` of the largest, narrowed to the 16 MiB-L3 cluster when
    clusters differ. Empty when the CPUs do not differ (or nothing is readable),
    so the caller leaves the affinity alone."""
    for leaf in ("cpu_capacity", "cpufreq/cpuinfo_max_freq"):
        values = {c: _read_int(f"{sysfs}/cpu{c}/{leaf}") for c in allowed}
        known = {c: v for c, v in values.items() if v is not None and v > 0}
        if len(known) != len(allowed) or not known:
            continue
        top = max(known.values())
        fast = {c for c, v in known.items() if v >= ratio * top}
        if fast and fast != allowed:
            return _pick_single_cluster(fast, known, sysfs)
        if fast and fast == allowed:
            narrowed = _pick_single_cluster(
                fast, known, sysfs, require_asymmetric_l3=True
            )
            if narrowed and narrowed != allowed:
                return narrowed
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


def pin_main_thread(cpus: list[int]) -> int | None:
    """Pin the calling (critical main) thread to 1 core ``c`` on the selected
    fast cluster while keeping background threads on the remaining cores of the
    same cluster. Fails soft; returns ``c`` when applied."""
    if not PIN or not PIN_MAIN or not cpus or not hasattr(os, "sched_setaffinity"):
        return None
    ordered = sorted(cpus)
    c = ordered[-1]
    try:
        if len(ordered) > 1:
            bg = set(ordered[:-1])
            main_tid = (
                threading.get_native_id()
                if hasattr(threading, "get_native_id")
                else os.getpid()
            )
            try:
                tids = [int(t) for t in os.listdir("/proc/self/task")]
            except OSError:
                tids = []
            for tid in tids:
                if tid != main_tid:
                    try:
                        os.sched_setaffinity(tid, bg)
                    except OSError:
                        pass
        os.sched_setaffinity(0, {c})
        return c
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Deep-idle prevention (pure-C SCHED_IDLE nanosleep tickler via libl2pf)
# --------------------------------------------------------------------------- #

_tickle_started = False
_tickle_lib: ctypes.CDLL | None = None
_last_tickle_phase: bool | None = None


def _get_l2pf_cdll() -> ctypes.CDLL | None:
    global _tickle_lib
    if _tickle_lib is not None:
        return _tickle_lib
    try:
        import l2pf

        lib_path = os.path.join(
            os.path.dirname(os.path.abspath(l2pf.__file__)), "libl2pf.so"
        )
        _tickle_lib = ctypes.CDLL(lib_path)
        return _tickle_lib
    except Exception:
        return None


def start_idle_tickle(cpus: list[int], core: int | None = None) -> int:
    """Spawn ONLY 1 pure-C SCHED_IDLE nanosleep pthread on core ``c`` (zero
    Python/GIL involvement) so ARM acpi_idle does not enter deep LPI-3 (~433 us
    exit) on the critical main thread's core between decode steps. Fails soft;
    returns thread count started (0 or 1)."""
    global _tickle_started
    if not TICKLE or _tickle_started or not cpus or sys.platform != "linux":
        return 0
    c = sorted(cpus)[-1] if core is None else int(core)
    try:
        lib = _get_l2pf_cdll()
        if lib is None:
            return 0
        fn = getattr(lib, "l2pf_start_idle_tickle", None)
        if fn is None:
            return 0
        fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        fn.restype = ctypes.c_int
        arr = (ctypes.c_int * 1)(c)
        started = int(fn(arr, 1))
        if started > 0:
            _tickle_started = True
        return started
    except Exception:
        return 0


def set_tickle_phase(is_decode: bool) -> None:
    """Notify ``libl2pf`` of the active phase (``False`` = prefill, ``True`` =
    decode). By default ``libl2pf`` keeps the 1 ms tickler active on the single
    pinned X925 core in both prefill and decode (unless overridden via
    ``L2PF_PREFILL_TICKLE_MS``) to avoid ARM ``acpi_idle`` LPI-3 wakeup
    penalties. Fails soft."""
    global _last_tickle_phase
    flag = bool(is_decode)
    if _last_tickle_phase is flag:
        return
    try:
        import l2pf

        setter = getattr(l2pf, "set_tickle_phase", None)
        if callable(setter):
            setter(flag)
            _last_tickle_phase = flag
            return
    except Exception:
        pass
    try:
        lib = _get_l2pf_cdll()
        if lib is None:
            return
        c_fn = getattr(lib, "l2pf_set_tickle_phase", None)
        if c_fn is None:
            return
        c_fn.argtypes = [ctypes.c_int]
        c_fn.restype = None
        c_fn(1 if flag else 0)
        _last_tickle_phase = flag
    except Exception:
        pass
