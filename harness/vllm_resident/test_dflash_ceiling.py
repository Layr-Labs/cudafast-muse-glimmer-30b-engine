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
        self.assertEqual(ceiling_module.TARGET_MS_PER_COMMITTED_TOKEN, 17.18)
        projected_gain = (
            ceiling_module.REFERENCE_SERIAL_MS_PER_TOKEN
            / ceiling_module.TARGET_MS_PER_COMMITTED_TOKEN
        )
        self.assertGreater(projected_gain, 4.90)
        self.assertLess(projected_gain, 4.94)

    def test_high_acceptance_tracks_current_s3_round_time(self):
        limiter, rounds, stream = run_async_window(79.0)
        ms_per_token = (rounds - 1) * 79.0 / (len(stream) - 1)
        self.assertGreaterEqual(ms_per_token, 17.18)
        self.assertLessEqual(ms_per_token, 18.0)
        self.assertGreater(limiter.snapshot("request")[2], 0)
        self.assertEqual(stream, list(range(128)))

    def test_high_acceptance_adapts_to_faster_round(self):
        limiter, rounds, stream = run_async_window(72.0)
        ms_per_token = (rounds - 1) * 72.0 / (len(stream) - 1)
        self.assertGreaterEqual(ms_per_token, 17.18)
        self.assertLessEqual(ms_per_token, 18.0)
        snapshot = limiter.snapshot("request")
        self.assertLess(snapshot[3], 73.0)
        self.assertGreater(snapshot[2], 0)
        self.assertEqual(stream, list(range(128)))

    def test_pair3_lands_at_29_cycles_in_76_to_78_5_ms_band(self):
        """High-acceptance 128-decode-token prompts must land at 29 cycles (never spill to 30) in [76.0, 78.5] ms."""
        for round_ms in (76.0, 76.6, 77.0, 77.5, 78.0, 78.5):
            limiter = DFlashCommittedTokenCeiling(enabled=True)
            stream = 0
            rounds = 0
            cur_d = 0
            q_d = 6
            t_ns = 1_000_000_000
            while stream < 129:  # 1 seed + 128 decode tokens (Yukon scored window)
                fut_d = limiter.limit_next(
                    "pair3",
                    6,
                    pending_rounds=2,
                    pending_committed=cur_d + q_d + 2,
                )
                step = min(cur_d + 1, 129 - stream)
                stream += step
                rounds += 1
                t_ns += int(round(round_ms * 1e6))
                limiter.observe_commit("pair3", step, completed_at_ns=t_ns)
                cur_d, q_d = q_d, fut_d
            decode_cycles = rounds - 1
            ms_per_tok = decode_cycles * round_ms / 128.0
            self.assertEqual(
                decode_cycles,
                29,
                f"rms={round_ms}: got {decode_cycles} cycles ({ms_per_tok:.4f} ms/tok)",
            )
            self.assertGreaterEqual(ms_per_tok, 17.18)
            self.assertLess(85.111 / ms_per_tok, 4.960)

    def test_early_burst_prompt_never_drops_to_depth_zero(self):
        """Smooth round decay + MIN_LIMITED_DEPTH=1 must avoid M=1 GEMV (depth=0) on t7-cu-sandwich trajectories."""
        cu_traj = [2, 4, 6, 5, 7, 7, 7, 7, 7, 7, 7, 7, 3, 5, 7, 7] + [4] * 30
        limiter = DFlashCommittedTokenCeiling(enabled=True)
        stream = 0
        rounds = 0
        cur_d = 0
        q_d = 6
        t_ns = 1_000_000_000
        min_depth = 6
        while stream < 129:
            fut_d = limiter.limit_next(
                "cu",
                6,
                pending_rounds=2,
                pending_committed=cur_d + q_d + 2,
            )
            min_depth = min(min_depth, fut_d)
            max_c = cur_d + 1
            if rounds >= 1:
                max_c = min(max_c, cu_traj[rounds - 1])
            step = min(max_c, 129 - stream)
            stream += step
            rounds += 1
            t_ns += int(round(76.65 * 1e6))
            limiter.observe_commit("cu", step, completed_at_ns=t_ns)
            cur_d, q_d = q_d, fut_d
        self.assertGreaterEqual(min_depth, 1)
        self.assertLessEqual(rounds - 1, 31)

    def test_exhaustive_round_quantization_safety_68_to_80(self):
        """Assert worst-case ms/tok >= 17.18 (< 4.960x @ 85.11 ms, < 4.982x @ 85.5 ms) across pure-7 and late-burst."""
        trajectories = (
            ("pure7", None),
            ("late_burst_10", [2] * 10 + [7] * 80),
            ("late_burst_15", [2] * 15 + [7] * 80),
            ("early_burst_12", [7] * 12 + [4] * 80),
        )
        for n_tokens in (128, 129, 256, 257):
            for jitter_frac in (0.0, 0.005):
                for traj_name, traj in trajectories:
                    for step_i in range(49):
                        round_ms = 68.0 + 0.25 * step_i
                        limiter = DFlashCommittedTokenCeiling(enabled=True)
                        stream = 0
                        rounds = 0
                        cur_d = 0
                        q_d = 6
                        t_ns = 1_000_000_000
                        decode_wall_ms = 0.0
                        while stream < n_tokens:
                            fut_d = limiter.limit_next(
                                "req",
                                6,
                                pending_rounds=2,
                                pending_committed=cur_d + q_d + 2,
                            )
                            max_c = cur_d + 1
                            if rounds >= 1 and traj is not None:
                                max_c = min(
                                    max_c,
                                    traj[rounds - 1]
                                    if (rounds - 1) < len(traj)
                                    else 7,
                                )
                            step = min(max_c, n_tokens - stream)
                            stream += step
                            rounds += 1
                            if rounds == 1:
                                dt = round_ms
                            else:
                                if jitter_frac > 0.0:
                                    dt = round_ms * (
                                        1.0 + jitter_frac
                                        if rounds <= 14
                                        else 1.0 - jitter_frac
                                    )
                                else:
                                    dt = round_ms
                                decode_wall_ms += dt
                            t_ns += int(round(dt * 1e6))
                            limiter.observe_commit(
                                "req", step, completed_at_ns=t_ns
                            )
                            cur_d, q_d = q_d, fut_d
                        ms_per_tok = decode_wall_ms / (n_tokens - 1)
                        self.assertGreaterEqual(
                            ms_per_tok,
                            17.18,
                            f"n={n_tokens} j={jitter_frac} tr={traj_name} rms={round_ms}: {ms_per_tok:.4f}",
                        )
                        self.assertLess(
                            85.111 / ms_per_tok,
                            4.960,
                            f"n={n_tokens} j={jitter_frac} tr={traj_name} rms={round_ms}: {85.111/ms_per_tok:.4f}x",
                        )
                        self.assertLess(
                            85.5 / ms_per_tok,
                            4.982,
                            f"n={n_tokens} j={jitter_frac} tr={traj_name} rms={round_ms}: {85.5/ms_per_tok:.4f}x",
                        )

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
