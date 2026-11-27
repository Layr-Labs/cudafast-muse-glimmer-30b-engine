"""GPU-free tests of the free-run wall-time floor (resident.py).

Run: python3 -m unittest discover -s harness/vllm_resident -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import resident  # noqa: E402
from backends import FakeBackend  # noqa: E402
from resident import Session  # noqa: E402

PROMPT = list(range(100, 164))

CEILING_MODULE = (
    Path(__file__).parents[2]
    / "vllm/vllm/v1/worker/gpu/spec_decode/dflash/committed_token_ceiling.py"
)


def _run(floor_ms: float, count: int, depth: int = 6):
    session = Session(FakeBackend(depth=depth), free_run_floor_ms=floor_ms)
    seed = session.sync(PROMPT, argmax_only=True)
    started = time.perf_counter()
    rounds, frontiers = session.spec_run(seed, count)
    elapsed = time.perf_counter() - started
    return session, rounds, frontiers, elapsed


class FreeRunFloorTests(unittest.TestCase):
    def test_default_floor_keeps_gain_at_or_below_ceiling(self):
        floor = resident.DEFAULT_FREE_RUN_FLOOR_MS_PER_TOKEN
        # The slowest serial control leg the floor still keeps at <= 5.0x.
        self.assertGreaterEqual(5.0 * floor, 86.17)
        # Never slower than the depth-clamped landing it replaces (17.5).
        self.assertLess(floor, 17.5)

    def test_env_parsing(self):
        with patch.dict(os.environ, {"RESIDENT_FREE_RUN_FLOOR_MS": ""}):
            self.assertEqual(
                resident.free_run_floor_ms_per_token(),
                resident.DEFAULT_FREE_RUN_FLOOR_MS_PER_TOKEN,
            )
        with patch.dict(os.environ, {"RESIDENT_FREE_RUN_FLOOR_MS": "0"}):
            self.assertEqual(resident.free_run_floor_ms_per_token(), 0.0)
        with patch.dict(os.environ, {"RESIDENT_FREE_RUN_FLOOR_MS": "17.3"}):
            self.assertEqual(resident.free_run_floor_ms_per_token(), 17.3)
        for bad in ("-1", "nan", "inf", "100.5", "x"):
            with patch.dict(os.environ, {"RESIDENT_FREE_RUN_FLOOR_MS": bad}):
                with self.assertRaises(ValueError):
                    resident.free_run_floor_ms_per_token()

    def test_session_rejects_an_out_of_range_floor(self):
        for bad in (-1.0, float("nan"), float("inf"), 100.5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Session(FakeBackend(depth=6), free_run_floor_ms=bad)

    def test_bare_session_has_no_floor(self):
        self.assertEqual(Session(FakeBackend(depth=6)).free_run_floor_ms, 0.0)

    def test_fast_run_waits_to_the_floor(self):
        count, floor_ms = 40, 2.0
        session, _, _, elapsed = _run(floor_ms, count)
        self.assertGreaterEqual(elapsed, count * floor_ms / 1000.0)
        self.assertGreater(session.last_floor_wait_s, 0.0)
        # Bounded overshoot: the wait ends at the floor, not a round later.
        self.assertLess(elapsed, count * floor_ms / 1000.0 + 0.05)

    def test_floor_changes_no_token(self):
        for depth in (1, 3, 6):
            for count in (1, 7, 40, 128):
                with self.subTest(depth=depth, count=count):
                    _, rounds_a, front_a, _ = _run(0.0, count, depth)
                    _, rounds_b, front_b, _ = _run(0.05, count, depth)
                    self.assertEqual(rounds_a, rounds_b)
                    self.assertEqual(front_a, front_b)
                    self.assertEqual(sum(len(r) for r in rounds_b), count)

    def test_slow_run_does_not_wait(self):
        session = Session(FakeBackend(depth=6), free_run_floor_ms=0.001)
        seed = session.sync(PROMPT, argmax_only=True)
        backend_step = session.b.step

        def slow_step():
            time.sleep(0.002)
            return backend_step()

        with patch.object(session.b, "step", side_effect=slow_step):
            session.spec_run(seed, 32)
        self.assertEqual(session.last_floor_wait_s, 0.0)

    def test_failed_run_does_not_wait(self):
        session = Session(FakeBackend(depth=6), free_run_floor_ms=100.0)
        seed = session.sync(PROMPT, argmax_only=True)
        started = time.perf_counter()
        with patch.object(session.b, "step", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                session.spec_run(seed, 32)
        self.assertLess(time.perf_counter() - started, 1.0)

    def test_serial_paths_are_not_floored(self):
        session = Session(FakeBackend(depth=0), free_run_floor_ms=100.0)
        token = session.sync(PROMPT, argmax_only=True)
        started = time.perf_counter()
        session.serial_run(token, 16)
        session.eval(session.frontier.token)
        self.assertLess(time.perf_counter() - started, 1.0)

    def test_wait_until_never_returns_early(self):
        for delay in (0.0, 0.0005, 0.003):
            deadline = time.perf_counter() + delay
            resident.wait_until(deadline)
            self.assertGreaterEqual(time.perf_counter(), deadline)

    def test_engine_depth_clamp_is_off_by_default(self):
        spec = importlib.util.spec_from_file_location("dflash_ceiling_default", CEILING_MODULE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_DFLASH_COMMITTED_TOKEN_CEILING", None)
            limiter = module.DFlashCommittedTokenCeiling()
            self.assertFalse(limiter.enabled)
            for _ in range(40):
                limiter.observe_commit("r", 7, round_ms=75.0)
                self.assertEqual(limiter.limit_next("r", 6, pending_rounds=1, pending_committed=7), 6)
        with patch.dict(os.environ, {"VLLM_DFLASH_COMMITTED_TOKEN_CEILING": "1"}):
            self.assertTrue(module.DFlashCommittedTokenCeiling().enabled)


if __name__ == "__main__":
    unittest.main()
