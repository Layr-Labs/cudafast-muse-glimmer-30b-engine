"""Host-side trims of the single-request engine loop (vllm/v1/worker/hostfast.py
and the in-process client in vllm/v1/engine/core_client.py).

`HostfastTests` need only the standard library. `InprocClientTests` need the
staged overlay (vLLM importable) and are skipped elsewhere. In the pinned
container:
  PYTHONPATH=.build/vllm-overlay python3 -m unittest harness/vllm_resident/test_hostfast.py -v
"""

from __future__ import annotations

import gc
import importlib.util
import os
import tempfile
import unittest
from collections import deque
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
HOSTFAST = os.path.join(HERE, "..", "..", "vllm", "vllm", "v1", "worker", "hostfast.py")


def load_hostfast():
    spec = importlib.util.spec_from_file_location("hostfast_under_test", HOSTFAST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_sysfs(capacities: dict[int, int | None], leaf: str = "cpu_capacity") -> str:
    root = tempfile.mkdtemp()
    for cpu, value in capacities.items():
        d = os.path.join(root, f"cpu{cpu}")
        os.makedirs(os.path.join(d, os.path.dirname(leaf)), exist_ok=True)
        if value is not None:
            with open(os.path.join(d, leaf), "w") as fp:
                fp.write(f"{value}\n")
    return root


class HostfastTests(unittest.TestCase):
    def setUp(self):
        self.hf = load_hostfast()
        self.addCleanup(self.hf.gc_idle)

    def test_gc_pauses_once_and_comes_back(self):
        self.assertTrue(gc.isenabled())
        self.assertTrue(self.hf.gc_busy(), "the first call is the one that paused it")
        self.assertFalse(gc.isenabled())
        self.assertFalse(self.hf.gc_busy(), "already paused")
        self.hf.gc_idle()
        self.assertTrue(gc.isenabled())
        self.hf.gc_idle()  # no-op
        self.assertTrue(gc.isenabled())

    def test_gc_is_left_alone_when_it_was_off_already(self):
        gc.disable()
        self.addCleanup(gc.enable)
        self.assertFalse(self.hf.gc_busy())
        self.hf.gc_idle()
        self.assertFalse(gc.isenabled(), "not ours to turn on")

    def test_gc_switch_off(self):
        with mock.patch.object(self.hf, "GC", False):
            self.assertFalse(self.hf.gc_busy())
        self.assertTrue(gc.isenabled())

    def test_fast_cpus_picks_the_high_capacity_cores(self):
        root = fake_sysfs({0: 725, 1: 725, 2: 1024, 3: 1024, 4: 997})
        self.assertEqual(self.hf.fast_cpus({0, 1, 2, 3, 4}, sysfs=root), {2, 3, 4})

    def test_fast_cpus_is_empty_when_the_cores_are_alike(self):
        root = fake_sysfs({0: 1024, 1: 1024})
        self.assertEqual(self.hf.fast_cpus({0, 1}, sysfs=root), set())

    def test_fast_cpus_is_empty_when_nothing_is_readable(self):
        self.assertEqual(self.hf.fast_cpus({0, 1}, sysfs=tempfile.mkdtemp()), set())
        root = fake_sysfs({0: 1024, 1: None})
        self.assertEqual(self.hf.fast_cpus({0, 1}, sysfs=root), set(), "one core unreadable")

    def test_fast_cpus_falls_back_to_the_max_frequency(self):
        root = fake_sysfs({0: 2000000, 1: 3900000}, leaf="cpufreq/cpuinfo_max_freq")
        self.assertEqual(self.hf.fast_cpus({0, 1}, sysfs=root), {1})

    def test_fast_cpus_only_looks_at_the_allowed_cores(self):
        root = fake_sysfs({0: 725, 1: 1024, 2: 1024})
        self.assertEqual(self.hf.fast_cpus({0, 1}, sysfs=root), {1})
        self.assertEqual(self.hf.fast_cpus({1, 2}, sysfs=root), set(), "alike within the cgroup")

    def test_format_cpus(self):
        self.assertEqual(self.hf.format_cpus([5, 6, 7, 9]), "5-7,9")
        self.assertEqual(self.hf.format_cpus([3]), "3")
        self.assertEqual(self.hf.format_cpus([]), "")

    def test_pin_sets_every_thread_and_reports_the_cores(self):
        calls = []

        def record(tid, cpus):
            calls.append((tid, set(cpus)))

        with (
            mock.patch.object(os, "sched_getaffinity", create=True, return_value={0, 1, 2, 3}),
            mock.patch.object(os, "sched_setaffinity", create=True, side_effect=record),
            mock.patch.object(os, "listdir", return_value=["11", "12"]),
            mock.patch.object(self.hf, "fast_cpus", return_value={2, 3}),
        ):
            self.assertEqual(self.hf.pin_to_fast_cpus(), [2, 3])
        self.assertEqual(calls, [(11, {2, 3}), (12, {2, 3}), (0, {2, 3})])

    def test_pin_fails_soft(self):
        with (
            mock.patch.object(os, "sched_getaffinity", create=True, return_value={0, 1}),
            mock.patch.object(os, "sched_setaffinity", create=True, side_effect=PermissionError()),
            mock.patch.object(os, "listdir", return_value=["11"]),
            mock.patch.object(self.hf, "fast_cpus", return_value={1}),
        ):
            self.assertEqual(self.hf.pin_to_fast_cpus(), [], "refused: nothing changed")
        with mock.patch.object(os, "sched_getaffinity", create=True, side_effect=OSError()):
            self.assertEqual(self.hf.pin_to_fast_cpus(), [])

    def test_pin_does_nothing_when_the_cores_are_alike_or_the_switch_is_off(self):
        with (
            mock.patch.object(os, "sched_getaffinity", create=True, return_value={0, 1}),
            mock.patch.object(os, "sched_setaffinity", create=True) as setter,
            mock.patch.object(self.hf, "fast_cpus", return_value=set()),
        ):
            self.assertEqual(self.hf.pin_to_fast_cpus(), [])
            setter.assert_not_called()
        with mock.patch.object(self.hf, "PIN", False):
            self.assertEqual(self.hf.pin_to_fast_cpus(), [])


try:
    from vllm.v1.engine.core_client import InprocClient
    from vllm.v1.worker import hostfast as live_hostfast
except ImportError:  # no vLLM on this host
    InprocClient = None


class FakeScheduler:
    def __init__(self):
        self.unfinished = False
        self.finished_pending = False

    def has_unfinished_requests(self):
        return self.unfinished

    def has_requests(self):
        return self.unfinished or self.finished_pending


class FakeCore:
    """The slice of EngineCore the in-process client touches."""

    def __init__(self):
        self.scheduler = FakeScheduler()
        self.batch_queue = deque()
        self.steps = 0
        self.aborted: list[list[str]] = []
        self.fail = False

    def abort_requests(self, request_ids):
        self.aborted.append(list(request_ids))
        self.scheduler.unfinished = False
        self.scheduler.finished_pending = True

    def step_fn(self):
        self.steps += 1
        if self.fail:
            raise RuntimeError("boom")
        if self.batch_queue:
            self.batch_queue.pop()
        self.scheduler.finished_pending = False
        return None, False

    def post_step(self, model_executed):
        pass

    def shutdown(self):
        pass


@unittest.skipIf(InprocClient is None, "vLLM is not importable here")
class InprocClientTests(unittest.TestCase):
    def setUp(self):
        self.core = FakeCore()
        self.client = InprocClient.__new__(InprocClient)
        self.client.engine_core = self.core
        self.addCleanup(live_hostfast.gc_idle)

    def test_abort_drains_the_queued_batch_and_the_cleanup(self):
        self.core.batch_queue.append(object())
        self.core.scheduler.finished_pending = True
        self.client.abort_requests(["r1"])
        self.assertEqual(self.core.aborted, [["r1"]])
        self.assertFalse(self.core.batch_queue)
        self.assertFalse(self.core.scheduler.has_requests())
        self.assertTrue(gc.isenabled())

    def test_an_empty_abort_drains_after_the_last_request_finished(self):
        self.core.batch_queue.append(object())
        self.client.abort_requests([])
        self.assertEqual(self.core.aborted, [])
        self.assertFalse(self.core.batch_queue)

    def test_a_deferred_empty_abort_leaves_the_drain_to_settle(self):
        self.client.defer_settle = True
        self.core.batch_queue.append(object())
        self.core.scheduler.finished_pending = True
        self.client.abort_requests([])
        self.assertEqual(self.core.steps, 0, "the output goes out before the cleanup")
        self.assertEqual(len(self.core.batch_queue), 1)
        self.client.settle()
        self.assertFalse(self.core.batch_queue)
        self.assertFalse(self.core.scheduler.has_requests())
        self.assertTrue(gc.isenabled())

    def test_a_deferred_client_still_drains_a_real_abort_inline(self):
        self.client.defer_settle = True
        self.core.batch_queue.append(object())
        self.client.abort_requests(["r1"])
        self.assertEqual(self.core.aborted, [["r1"]])
        self.assertFalse(self.core.batch_queue)
        self.assertFalse(self.core.scheduler.has_requests())

    def test_settle_never_advances_a_live_request(self):
        self.client.defer_settle = True
        self.core.scheduler.unfinished = True
        self.core.batch_queue.append(object())
        self.client.settle()
        self.assertEqual(self.core.steps, 0)
        self.assertEqual(len(self.core.batch_queue), 1)

    def test_a_live_request_is_never_advanced_by_an_abort(self):
        self.core.scheduler.unfinished = True
        self.core.batch_queue.append(object())
        self.client.abort_requests([])
        self.assertEqual(self.core.steps, 0)
        self.assertEqual(len(self.core.batch_queue), 1)

    def test_get_output_pauses_gc_while_there_is_work_and_resumes_when_idle(self):
        self.core.scheduler.unfinished = True
        self.client.get_output()
        self.assertFalse(gc.isenabled(), "a live request keeps the collector paused")
        self.core.scheduler.unfinished = False
        self.client.get_output()
        self.assertTrue(gc.isenabled(), "nothing left: the collector is back")

    def test_an_error_gives_the_collector_back(self):
        self.core.scheduler.unfinished = True
        self.core.fail = True
        with self.assertRaises(RuntimeError):
            self.client.get_output()
        self.assertTrue(gc.isenabled())


try:
    from vllm.v1.worker.gpu import async_utils
except ImportError:  # no vLLM on this host
    async_utils = None


class FakeEvent:
    """A CUDA event that completes after `polls` queries (or at once)."""

    def __init__(self, polls=0):
        self.polls = polls
        self.queries = 0
        self.synchronized = 0

    def query(self):
        self.queries += 1
        return self.queries > self.polls

    def synchronize(self):
        self.synchronized += 1


@unittest.skipIf(async_utils is None, "vLLM is not importable here")
class TwoStageWaitTests(unittest.TestCase):
    def test_a_decode_step_blocks_once_and_never_polls(self):
        copy = FakeEvent()
        async_utils.wait_for_output(copy)
        self.assertEqual((copy.synchronized, copy.queries), (1, 0))

    def test_a_prefill_step_sleeps_through_the_forward_then_polls_the_tail(self):
        forward, copy = FakeEvent(), FakeEvent(polls=7)
        async_utils.wait_for_output(copy, forward)
        self.assertEqual(forward.synchronized, 1)
        self.assertEqual(copy.queries, 8)
        self.assertEqual(copy.synchronized, 0, "the copy event was polled, not slept on")

    def test_a_long_tail_falls_back_to_the_blocking_wait(self):
        forward, copy = FakeEvent(), FakeEvent(polls=10**9)
        ticks = iter(range(0, 10**12, async_utils._TAIL_POLL_NS // 4))
        async_utils.wait_for_output(copy, forward, clock=lambda: next(ticks))
        self.assertEqual(forward.synchronized, 1)
        self.assertEqual(copy.synchronized, 1)
        self.assertLess(copy.queries, 10)


if __name__ == "__main__":
    unittest.main()
