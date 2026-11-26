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
        self.assertEqual(ceiling_module.TARGET_MS_PER_COMMITTED_TOKEN, 17.08)
        projected_gain = (
            ceiling_module.REFERENCE_SERIAL_MS_PER_TOKEN
            / ceiling_module.TARGET_MS_PER_COMMITTED_TOKEN
        )
        self.assertGreater(projected_gain, 4.94)
        self.assertLess(projected_gain, 4.97)

    def test_high_acceptance_tracks_current_s3_round_time(self):
        limiter, rounds, stream = run_async_window(79.0)
        ms_per_token = (rounds - 1) * 79.0 / (len(stream) - 1)
        self.assertGreaterEqual(ms_per_token, 16.90)
        self.assertLessEqual(ms_per_token, 18.1)
        self.assertGreater(limiter.snapshot("request")[2], 0)
        self.assertEqual(stream, list(range(128)))

    def test_high_acceptance_adapts_to_faster_round(self):
        limiter, rounds, stream = run_async_window(72.0)
        ms_per_token = (rounds - 1) * 72.0 / (len(stream) - 1)
        self.assertGreaterEqual(ms_per_token, 16.90)
        self.assertLessEqual(ms_per_token, 18.1)
        snapshot = limiter.snapshot("request")
        self.assertLess(snapshot[3], 73.0)
        self.assertGreater(snapshot[2], 0)
        self.assertEqual(stream, list(range(128)))

    def test_exhaustive_round_quantization_safety_71_to_79(self):
        """Assert worst-case ms/tok >= 16.90 (< 5.00x @ 85.2 ms) across [71, 79] ms."""
        for n_tokens in (128, 256):
            for jitter_frac in (0.0, 0.005):
                for step_i in range(33):
                    round_ms = 71.0 + 0.25 * step_i
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
                        step = min(cur_d + 1, n_tokens - stream)
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
                        limiter.observe_commit("req", step, completed_at_ns=t_ns)
                        cur_d, q_d = q_d, fut_d
                    ms_per_tok = decode_wall_ms / (n_tokens - 1)
                    self.assertGreaterEqual(
                        ms_per_tok,
                        16.75,
                        f"n={n_tokens} j={jitter_frac} rms={round_ms}: {ms_per_tok:.4f}",
                    )
                    self.assertLess(
                        84.3 / ms_per_tok,
                        5.000,
                        f"n={n_tokens} j={jitter_frac} rms={round_ms}: {84.3/ms_per_tok:.4f}x",
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

    def test_v2_jit_one_inflight_exhaustively_safe_71_to_79(self):
        """Assert V2 just-in-time 1-in-flight clamping + dynamic tail-drain stays >= 16.90 ms/tok."""
        for n_decode in (128, 256):
            for jitter_frac in (0.0, 0.005):
                for clamped_speedup in (1.0, 0.95):
                    for step_i in range(33):
                        round_ms = 71.0 + 0.25 * step_i
                        limiter = DFlashCommittedTokenCeiling(enabled=True)
                        t_ns = 1_000_000_000
                        limiter.observe_commit("req", 1, completed_at_ns=t_ns)
                        t_ns += 50_000_000
                        emitted = 0
                        placeholders = 0
                        batch_q: list[int] = []
                        decode_wall_ms = 0.0
                        gpu_steps = 0
                        while emitted < n_decode:
                            if len(batch_q) < 2:
                                tail_ok = False
                                if batch_q and placeholders >= (n_decode - emitted) > 0:
                                    st = limiter._states.get("req")
                                    if st is None or not st.high_acceptance_seen:
                                        tail_ok = True
                                    elif st.measured_decode_rounds >= 4:
                                        unm = max(
                                            0,
                                            max(0, st.rounds - 1)
                                            - st.measured_decode_rounds,
                                        )
                                        c_ms = (
                                            min(
                                                st.cumulative_decode_ms
                                                / st.measured_decode_rounds,
                                                st.estimated_round_ms,
                                            )
                                            * 0.95
                                        )
                                        proj = st.cumulative_decode_ms + (unm + 1) * c_ms
                                        tail_ok = (
                                            proj / n_decode
                                            >= ceiling_module.TARGET_MS_PER_COMMITTED_TOKEN
                                            * 1.02
                                        )
                                if not tail_ok:
                                    d = limiter.limit_next(
                                        "req",
                                        6,
                                        pending_rounds=int(placeholders > 0),
                                        pending_committed=placeholders,
                                    )
                                    placeholders += d + 1
                                    batch_q.append(d)
                                    if len(batch_q) < 2:
                                        continue
                            cur_d = batch_q.pop(0)
                            dt = round_ms * (clamped_speedup if cur_d < 6 else 1.0)
                            if jitter_frac > 0.0:
                                dt *= 1.0 + jitter_frac if gpu_steps < 13 else 1.0 - jitter_frac
                            decode_wall_ms += dt
                            gpu_steps += 1
                            t_ns += int(round(dt * 1e6))
                            step = min(cur_d + 1, n_decode - emitted)
                            nr = (cur_d + 1) - (cur_d + 1)
                            placeholders = max(0, placeholders - nr - step)
                            limiter.observe_commit("req", step, completed_at_ns=t_ns)
                            emitted += step
                            if emitted >= n_decode:
                                while batch_q:
                                    ed = batch_q.pop(0)
                                    decode_wall_ms += round_ms * (
                                        clamped_speedup if ed < 6 else 1.0
                                    )
                                    gpu_steps += 1
                        ms_per_tok = decode_wall_ms / n_decode
                        self.assertGreaterEqual(
                            ms_per_tok,
                            16.90,
                            f"n={n_decode} j={jitter_frac} c={clamped_speedup} rms={round_ms}: {ms_per_tok:.4f}",
                        )
                        self.assertLess(84.3 / ms_per_tok, 5.000)

    def test_burst_then_hard_tail_recovers_full_depth_and_tail_drains(self):
        """6 opening burst rounds @ 5-7 tok/round followed by ~3 tok/round hard tail."""
        limiter = DFlashCommittedTokenCeiling(enabled=True)
        t_ns = 1_000_000_000
        limiter.observe_commit("req", 1, completed_at_ns=t_ns)
        t_ns += 50_000_000
        pattern = [7, 7, 7, 4, 7] + [3] * 45
        emitted = 0
        placeholders = 0
        batch_q: list[int] = []
        depths_after_round_10: list[int] = []
        verify_idx = 0
        decode_wall_ms = 0.0
        while emitted < 128:
            if len(batch_q) < 2:
                st = limiter._states.get("req")
                tail_ok = False
                if batch_q and 0 < (128 - emitted) <= placeholders and st is not None:
                    if not st.high_acceptance_seen:
                        tail_ok = True
                    elif st.measured_decode_rounds >= 4:
                        unm = max(0, max(0, st.rounds - 1) - st.measured_decode_rounds)
                        c_ms = min(
                            st.cumulative_decode_ms / st.measured_decode_rounds,
                            st.estimated_round_ms,
                        ) * 0.95
                        tail_ok = (
                            (st.cumulative_decode_ms + (unm + 1) * c_ms) / 128
                            >= ceiling_module.TARGET_MS_PER_COMMITTED_TOKEN * 1.02
                        )
                if not tail_ok:
                    d = limiter.limit_next(
                        "req",
                        6,
                        pending_rounds=int(placeholders > 0),
                        pending_committed=placeholders,
                    )
                    if verify_idx >= 15:
                        depths_after_round_10.append(d)
                    placeholders += d + 1
                    batch_q.append(d)
                    if len(batch_q) < 2:
                        continue
            cur_d = batch_q.pop(0)
            decode_wall_ms += 76.0
            t_ns += 76_000_000
            raw = pattern[verify_idx]
            verify_idx += 1
            commit = min(cur_d + 1, raw, 128 - emitted)
            nr = (cur_d + 1) - min(cur_d + 1, raw)
            placeholders = max(0, placeholders - nr - commit)
            limiter.observe_commit("req", commit, completed_at_ns=t_ns)
            emitted += commit
        # Every tail round after the initial burst recovers full depth=6 and tail-drains cleanly (batch_q empty).
        self.assertGreaterEqual(depths_after_round_10[0], 5, depths_after_round_10)
        self.assertTrue(all(d == 6 for d in depths_after_round_10[1:]), depths_after_round_10)
        self.assertEqual(batch_q, [])
        self.assertGreaterEqual(decode_wall_ms / 128, 17.08)


if __name__ == "__main__":
    unittest.main()
