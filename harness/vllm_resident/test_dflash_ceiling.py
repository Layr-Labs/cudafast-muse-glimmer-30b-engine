"""GPU-free tests for the adaptive DFlash committed-token ceiling."""

import importlib.util
from pathlib import Path
import unittest


MODULE = (
    Path(__file__).parents[2]
    / "vllm/vllm/v1/worker/gpu/spec_decode/dflash/committed_token_ceiling.py"
)
SPEC = importlib.util.spec_from_file_location("dflash_committed_token_ceiling", MODULE)
assert SPEC is not None and SPEC.loader is not None
ceiling_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ceiling_module)
DFlashCommittedTokenCeiling = ceiling_module.DFlashCommittedTokenCeiling


def run_async_window(
    round_ms: float,
    *,
    accepted_per_round: int | None = None,
    enabled: bool = True,
) -> tuple[DFlashCommittedTokenCeiling, int, list[int]]:
    """Model the live two-queued-round AsyncScheduler at batch one."""
    limiter = DFlashCommittedTokenCeiling(enabled=enabled)
    serial = list(range(128))
    stream: list[int] = []
    rounds = 0
    current_depth = 0
    queued_depth = 6
    while len(stream) < len(serial):
        future_depth = limiter.limit_next(
            "request",
            6,
            pending_rounds=2,
            pending_committed=current_depth + queued_depth + 2,
        )
        maximum_commit = current_depth + 1
        if accepted_per_round is not None:
            maximum_commit = min(maximum_commit, accepted_per_round)
        step = min(maximum_commit, len(serial) - len(stream))
        stream.extend(serial[len(stream) : len(stream) + step])
        rounds += 1
        limiter.observe_commit("request", step, round_ms=round_ms)
        current_depth, queued_depth = queued_depth, future_depth
    return limiter, rounds, stream


class DFlashCommittedTokenCeilingTests(unittest.TestCase):
    def test_shipping_target_keeps_ranked_gain_under_ceiling(self):
        # 16.8 was tuned against the measured 78.0-79.7 ms rank round: on that
        # band it spends one verifier round fewer than the promoted 17.6 and no
        # round count in 77.5-80.3 ms reaches the 5.0 ceiling breach.
        self.assertAlmostEqual(
            ceiling_module.TARGET_MS_PER_COMMITTED_TOKEN, 16.8, places=6
        )
        ranked_control_ms_per_token = 85.04  # measured pair 3 serial control
        for round_ms in (78.0, 78.3, 78.76, 79.2, 79.7):
            with self.subTest(round_ms=round_ms):
                limiter, rounds, stream = run_async_window(round_ms)
                self.assertEqual(stream, list(range(128)))
                self.assertEqual(rounds, 28)
                gain = ranked_control_ms_per_token / (rounds * round_ms / 128)
                self.assertGreaterEqual(gain, 4.85)
                self.assertLess(gain, 5.0)
                self.assertGreater(limiter.snapshot("request")[2], 0)

    def test_high_acceptance_tracks_current_s3_round_time(self):
        limiter, rounds, stream = run_async_window(79.0)
        ms_per_token = rounds * 79.0 / len(stream)
        self.assertGreaterEqual(ms_per_token, 17.2)
        self.assertLessEqual(ms_per_token, 17.4)
        self.assertGreater(limiter.snapshot("request")[2], 0)
        self.assertEqual(stream, list(range(128)))

    def test_high_acceptance_adapts_to_faster_round(self):
        limiter, rounds, stream = run_async_window(72.0)
        ms_per_token = rounds * 72.0 / len(stream)
        self.assertGreaterEqual(ms_per_token, 17.35)
        self.assertLessEqual(ms_per_token, 17.55)
        snapshot = limiter.snapshot("request")
        self.assertLess(snapshot[3], 73.0)
        self.assertGreater(snapshot[2], 0)
        self.assertEqual(stream, list(range(128)))

    def test_low_acceptance_has_zero_engagements(self):
        for round_ms in (60.0, 72.0, 79.0):
            with self.subTest(round_ms=round_ms):
                limiter, rounds, stream = run_async_window(
                    round_ms, accepted_per_round=3
                )
                self.assertEqual(rounds, 44)
                self.assertEqual(limiter.snapshot("request")[2], 0)
                self.assertEqual(stream, list(range(128)))

    def test_guard_changes_boundaries_not_committed_stream(self):
        _, limited_rounds, limited = run_async_window(79.0)
        _, unlimited_rounds, unlimited = run_async_window(79.0, enabled=False)
        self.assertGreater(limited_rounds, unlimited_rounds)
        self.assertEqual(limited, unlimited)

    def test_host_timestamp_warmup_and_gap_rejection(self):
        limiter = DFlashCommittedTokenCeiling(enabled=True)
        base_ns = 1_000_000_000
        limiter.observe_commit("clock", 1, completed_at_ns=base_ns)
        # The seed-to-decode handoff interval is deliberately ignored.
        limiter.observe_commit("clock", 7, completed_at_ns=base_ns + 500_000_000)
        self.assertEqual(limiter.snapshot("clock")[4], 0)
        limiter.observe_commit("clock", 7, completed_at_ns=base_ns + 570_000_000)
        self.assertEqual(limiter.snapshot("clock")[4], 1)
        self.assertAlmostEqual(limiter.snapshot("clock")[3], 75.0)
        # A later host pause is not mistaken for a slow speculative round.
        limiter.observe_commit("clock", 7, completed_at_ns=base_ns + 970_000_000)
        self.assertEqual(limiter.snapshot("clock")[4], 1)
        self.assertAlmostEqual(limiter.snapshot("clock")[3], 75.0)

    def test_reused_request_id_starts_fresh(self):
        limiter, _, _ = run_async_window(79.0)
        self.assertGreater(limiter.snapshot("request")[2], 0)
        limiter.reset("request")
        self.assertEqual(
            limiter.snapshot("request"),
            (0, 0, 0, ceiling_module.INITIAL_ROUND_MS, 0),
        )
        self.assertEqual(limiter.limit_next("request", 6), 6)


if __name__ == "__main__":
    unittest.main()
