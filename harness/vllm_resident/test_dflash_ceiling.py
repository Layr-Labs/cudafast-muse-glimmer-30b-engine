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
    def test_explicit_safety_constant_matches_control_gain(self):
        self.assertEqual(ceiling_module.TARGET_MS_PER_COMMITTED_TOKEN, 17.22)
        self.assertEqual(ceiling_module.REFERENCE_MAX_DECODE_GAIN, 5.0)

    def test_round_quantized_endpoint_stays_under_ceiling(self):
        # 85.0 ms is the fastest pair-3 serial control observed; 5.0x of it
        # is 17.0 ms/token.  Sweep round times and acceptance caps.
        for quarter_ms in range(280, 340):
            round_ms = quarter_ms / 4
            for accepted in (None, 6, 5):
                _, rounds, stream = run_async_window(
                    round_ms, accepted_per_round=accepted
                )
                self.assertEqual(stream, list(range(128)))
                self.assertGreater(rounds * round_ms / len(stream), 17.0)

    def test_high_acceptance_tracks_current_s3_round_time(self):
        limiter, rounds, stream = run_async_window(79.0)
        ms_per_token = rounds * 79.0 / len(stream)
        self.assertGreater(ms_per_token, 17.0)
        self.assertLessEqual(ms_per_token, 18.0)
        self.assertGreater(limiter.snapshot("request")[2], 0)
        self.assertEqual(stream, list(range(128)))

    def test_high_acceptance_adapts_to_faster_round(self):
        limiter, rounds, stream = run_async_window(72.0)
        ms_per_token = rounds * 72.0 / len(stream)
        self.assertGreater(ms_per_token, 17.0)
        self.assertLessEqual(ms_per_token, 18.0)
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
