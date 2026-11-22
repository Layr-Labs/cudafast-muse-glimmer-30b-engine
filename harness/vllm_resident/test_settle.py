"""settle.py: the bounded wait for a quiet host before the resident reports ready. Standard library only."""

from __future__ import annotations

import os
import tempfile
import unittest

try:
    from . import settle
except ImportError:  # run as a script / discovered from this directory
    import settle  # type: ignore

HZ = os.sysconf("SC_CLK_TCK")


class Clock:
    """A scripted host: each call to the sampler advances half a second and applies the next step."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.t = 0.0
        self.vm = {key: 0 for key in settle.HOUSEKEEPING}
        self.busy = 0
        self.own = 0.0
        self.calls = 0

    def sampler(self):
        if self.calls:
            self.t += 0.5
            step = self.steps.pop(0) if self.steps else {}
            for key, value in step.get("vm", {}).items():
                self.vm[key] += value
            self.busy += int(step.get("others", 0.0) * HZ * 0.5) + int(step.get("own", 0.0) * HZ * 0.5)
            self.own += step.get("own", 0.0) * HZ * 0.5
        self.calls += 1
        return settle.Sample(self.t, dict(self.vm), self.busy, self.own)


class SettleTests(unittest.TestCase):
    def run_settle(self, steps, **kwargs):
        clock = Clock(steps)
        return settle.settle(sampler=clock.sampler, sleep=lambda _: None, **kwargs), clock

    def test_quiet_host_returns_after_the_quiet_span(self):
        result, clock = self.run_settle([])
        self.assertTrue(result["quiet"])
        self.assertAlmostEqual(result["waited"], settle.QUIET_S)
        self.assertFalse(result["reasons"])

    def test_own_cpu_is_not_other_processes(self):
        result, _ = self.run_settle([{"own": 4.0}] * 20)
        self.assertTrue(result["quiet"])
        self.assertAlmostEqual(result["waited"], settle.QUIET_S)

    def test_waits_out_reclaim_then_needs_a_full_quiet_span(self):
        busy = [{"vm": {"pgscan_kswapd": 50_000}}] * 8  # 4 s of kswapd
        result, _ = self.run_settle(busy)
        self.assertTrue(result["quiet"])
        self.assertAlmostEqual(result["waited"], 4.0 + settle.QUIET_S)
        self.assertEqual(result["reasons"], {"reclaim": 8})

    def test_every_class_of_work_is_named(self):
        steps = [{"vm": {"compact_migrate_scanned": 10_000}}, {"vm": {"pswpin": 100}}, {"vm": {"pgscan_direct": 1}},
                 {"others": 1.0}]
        result, _ = self.run_settle(steps)
        self.assertEqual(set(result["reasons"]), {"compaction", "swap", "reclaim", "other processes"})
        self.assertTrue(result["quiet"])

    def test_small_background_rates_are_idle(self):
        steps = [{"vm": {"pswpin": 4, "pgscan_kswapd": 100, "compact_free_scanned": 500}, "others": 0.1}] * 10
        result, _ = self.run_settle(steps)
        self.assertTrue(result["quiet"])
        self.assertAlmostEqual(result["waited"], settle.QUIET_S)

    def test_ceiling_ends_the_wait_on_a_busy_host(self):
        result, _ = self.run_settle([{"others": 3.0}] * 1000, max_wait_s=5.0)
        self.assertFalse(result["quiet"])
        self.assertAlmostEqual(result["waited"], 5.0)

    def test_unreadable_counters_are_idle(self):
        def sampler(state={"t": 0.0}):
            state["t"] += 0.5
            return settle.Sample(state["t"])

        result = settle.settle(sampler=sampler, sleep=lambda _: None)
        self.assertTrue(result["quiet"])

    def test_a_failing_sampler_never_raises(self):
        def sampler():
            raise OSError("no /proc")

        result = settle.settle(sampler=sampler, sleep=lambda _: None)
        self.assertFalse(result["quiet"])

    def test_readers(self):
        with tempfile.TemporaryDirectory() as d:
            vm = os.path.join(d, "vmstat")
            with open(vm, "w") as fp:
                fp.write("nr_free_pages 5\npgscan_kswapd 7\npswpin 3\nunrelated 9\n")
            self.assertEqual(settle.read_vmstat(vm), {"pgscan_kswapd": 7, "pswpin": 3})
            st = os.path.join(d, "stat")
            with open(st, "w") as fp:
                fp.write("cpu  100 5 50 1000 20 3 2 1 0 0\ncpu0 1 1 1 1 1 1 1 1 0 0\n")
            self.assertEqual(settle.read_busy_ticks(st), 100 + 5 + 50 + 3 + 2 + 1)
            self.assertEqual(settle.read_vmstat(os.path.join(d, "missing")), {})
            self.assertIsNone(settle.read_busy_ticks(os.path.join(d, "missing")))

    def test_settle_and_log_writes_one_line(self):
        lines = []
        real = settle.settle
        settle.settle = lambda: {"waited": 4.5, "quiet": True, "reasons": {"reclaim": 3}}
        try:
            settle.settle_and_log(lines.append)
        finally:
            settle.settle = real
        self.assertEqual(lines, ["host quiet after 4.5 s (waited on: reclaim in 3 intervals)"])


if __name__ == "__main__":
    unittest.main()
