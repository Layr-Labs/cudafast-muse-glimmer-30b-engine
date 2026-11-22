"""Report ready on a quiet host.

The decode phase streams every weight once per token at the limit of the
memory bus, and the bus is shared with the CPUs: measured on GB10, one host
core copying memory costs the GPU stream about 5%, and 1.2 GB/s of host copying
costs 1% of decode. The load that precedes the timed phases is exactly the kind
of event that leaves such work behind (page reclaim, compaction, swap-in,
write-back of the caches the engine just wrote), and anything else that is busy
on the box when the resident reports ready lands in the timed windows too.

``settle`` runs once, after the model is loaded and warmed and before the
socket is bound. It returns the freed part of the Python heap to the kernel,
then waits until the kernel's memory housekeeping counters and the CPU time of
every other process have been idle for a few seconds, bounded by a ceiling. It
computes nothing the engine reads and never fails the boot: an unreadable
counter is treated as idle, and the ceiling ends the wait whatever the host is
doing.

  RESIDENT_SETTLE=0   skip the wait
"""

from __future__ import annotations

import ctypes
import gc
import os
import time
from collections import Counter
from dataclasses import dataclass, field

ENABLED = os.environ.get("RESIDENT_SETTLE", "1").strip() not in ("0", "", "false", "no")
MAX_WAIT_S = 45.0
QUIET_S = 3.0
INTERVAL_S = 0.5
# /proc/vmstat counters of kernel memory housekeeping and the rate (events per
# second) below which each is idle. Direct reclaim and allocation stalls are
# never idle while they move.
HOUSEKEEPING = {
    "compact_migrate_scanned": ("compaction", 2048),
    "compact_free_scanned": ("compaction", 2048),
    "pgmigrate_success": ("compaction", 256),
    "pgscan_kswapd": ("reclaim", 2048),
    "pgsteal_kswapd": ("reclaim", 512),
    "pgscan_direct": ("reclaim", 0),
    "pgsteal_direct": ("reclaim", 0),
    "allocstall_normal": ("reclaim", 0),
    "allocstall_movable": ("reclaim", 0),
    "pswpin": ("swap", 32),
    "pswpout": ("swap", 32),
}
# CPU time of everything on the host except this process, in cores.
OTHERS_CPU_CORES = 0.25


@dataclass
class Sample:
    t: float
    vm: dict[str, int] = field(default_factory=dict)
    busy_ticks: int | None = None
    own_ticks: float = 0.0


def read_vmstat(path: str = "/proc/vmstat") -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open(path) as fp:
            for line in fp:
                key, _, value = line.partition(" ")
                if key in HOUSEKEEPING:
                    out[key] = int(value)
    except (OSError, ValueError):
        return {}
    return out


def read_busy_ticks(path: str = "/proc/stat") -> int | None:
    """Clock ticks all CPUs spent on anything but idle and I/O wait."""
    try:
        with open(path) as fp:
            fields = fp.readline().split()
        if not fields or fields[0] != "cpu":
            return None
        ticks = [int(x) for x in fields[1:9]]
        return sum(ticks) - ticks[3] - (ticks[4] if len(ticks) > 4 else 0)
    except (OSError, ValueError, IndexError):
        return None


def sample() -> Sample:
    times = os.times()
    hz = os.sysconf("SC_CLK_TCK")
    own = (times.user + times.system + times.children_user + times.children_system) * hz
    return Sample(time.monotonic(), read_vmstat(), read_busy_ticks(), own)


def busy_reasons(a: Sample, b: Sample) -> list[str]:
    """Why the interval a -> b is not quiet; empty when it is."""
    dt = max(b.t - a.t, 1e-6)
    reasons = set()
    for key, (name, idle_rate) in HOUSEKEEPING.items():
        if key in a.vm and key in b.vm and (b.vm[key] - a.vm[key]) / dt > idle_rate:
            reasons.add(name)
    if a.busy_ticks is not None and b.busy_ticks is not None:
        hz = os.sysconf("SC_CLK_TCK")
        others = ((b.busy_ticks - a.busy_ticks) - (b.own_ticks - a.own_ticks)) / hz / dt
        if others > OTHERS_CPU_CORES:
            reasons.add("other processes")
    return sorted(reasons)


def trim_heap() -> None:
    """Give the freed part of the heap back to the kernel now, not during a timed phase."""
    try:
        gc.collect()
        ctypes.CDLL(None).malloc_trim(0)
    except (OSError, AttributeError):
        pass


def settle(
    max_wait_s: float = MAX_WAIT_S,
    quiet_s: float = QUIET_S,
    interval_s: float = INTERVAL_S,
    sampler=sample,
    sleep=time.sleep,
) -> dict:
    """Wait until the host has been quiet for quiet_s, at most max_wait_s.
    Returns {"waited", "quiet", "reasons"}; never raises."""
    result = {"waited": 0.0, "quiet": False, "reasons": Counter()}
    try:
        first = last = sampler()
        quiet_since = first.t
        while True:
            sleep(interval_s)
            now = sampler()
            reasons = busy_reasons(last, now)
            last = now
            result["waited"] = now.t - first.t
            if reasons:
                quiet_since = now.t
                result["reasons"].update(reasons)
            elif now.t - quiet_since >= quiet_s:
                result["quiet"] = True
                return result
            if result["waited"] >= max_wait_s:
                return result
    except Exception:  # noqa: BLE001 - the boot goes on
        return result


def settle_and_log(log) -> None:
    if not ENABLED:
        return
    try:
        trim_heap()
        result = settle()
        seen = ", ".join(f"{name} in {count} intervals" for name, count in sorted(result["reasons"].items()))
        log(
            f"host {'quiet' if result['quiet'] else 'still busy at the ceiling'} after {result['waited']:.1f} s"
            + (f" (waited on: {seen})" if seen else "")
        )
    except Exception as err:  # noqa: BLE001
        log(f"settle skipped: {err}")
