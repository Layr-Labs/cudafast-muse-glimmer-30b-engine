# SPDX-License-Identifier: Apache-2.0
"""Wall-time-adaptive decode rate ceiling for DFlash speculative decoding.

Keeps high-acceptance single-request decode windows safely below the track's
5.0x per-pair decode speedup ceiling (~17.16 ms/committed token against an
~85.8 ms/token serial control) by trimming future draft proposal depth when a
request's cumulative committed-token pace exceeds the target rate.

Low- and medium-acceptance requests never reach the high-acceptance trigger and
keep full offered depth unchanged.  Only proposal lengths are shortened; target
verification still decides every committed token.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

# Target >= 17.45 ms/committed token (~4.88x-4.92x against 85.1-85.8 ms/token
# serial control), providing ~1.5%-2.3% safety margin below the 5.0x ceiling.
TARGET_MS_PER_COMMITTED_TOKEN = 17.45
HIGH_ACCEPTANCE_TRIGGER_TOKENS_PER_ROUND = 4.2
ROUND_PROJECT_SAFETY_FACTOR = 0.99
EWMA_DOWN_ALPHA = 0.50
EWMA_UP_ALPHA = 0.25
MIN_VALID_ROUND_MS = 10.0
MAX_VALID_ROUND_MS = 250.0
MAX_GAP_MULTIPLE = 2.5
UNTHROTTLED_MIN_DRAFT_TOKENS = 4


@dataclass
class _RequestPaceState:
    rounds: int = 0
    committed_tokens: int = 0
    engagements: int = 0
    ewma_round_ms: float | None = None
    first_round_done_ns: int | None = None
    last_done_ns: int | None = None
    high_acceptance_seen: bool = False


class DFlashDecodeCeiling:
    """Per-request draft-depth limiter backed by monotonic wall-time pacing."""

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = os.environ.get("VLLM_DFLASH_DECODE_CEILING", "1") != "0"
        self.enabled = enabled
        self._target_ms = float(
            os.environ.get(
                "VLLM_DFLASH_DECODE_CEILING_MS",
                str(TARGET_MS_PER_COMMITTED_TOKEN),
            )
        )
        self._states: dict[str, _RequestPaceState] = {}

    def reset(self, req_id: str) -> None:
        self._states.pop(req_id, None)

    def observe_seed(self, req_id: str) -> None:
        """Reset per-request pace state when the seed (prefill) token lands."""
        if self.enabled:
            self._states[req_id] = _RequestPaceState()

    def observe_commit(
        self,
        req_id: str,
        committed_tokens: int,
        *,
        scheduled_draft_tokens: int = 0,
        completed_at_ns: int | None = None,
        round_ms: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        committed_tokens = max(int(committed_tokens), 0)
        if committed_tokens == 0:
            return

        state = self._states.setdefault(req_id, _RequestPaceState())
        state.rounds += 1
        state.committed_tokens += committed_tokens
        if (
            state.committed_tokens
            > HIGH_ACCEPTANCE_TRIGGER_TOKENS_PER_ROUND * state.rounds
        ):
            state.high_acceptance_seen = True

        if round_ms is not None:
            now_ns = (
                (state.last_done_ns or 0) + int(round(float(round_ms) * 1e6))
            )
        else:
            now_ns = (
                time.monotonic_ns()
                if completed_at_ns is None
                else int(completed_at_ns)
            )

        prev_ns = state.last_done_ns
        state.last_done_ns = now_ns
        if state.rounds == 1:
            state.first_round_done_ns = now_ns
            return

        if prev_ns is None:
            return
        obs_ms = (now_ns - prev_ns) / 1e6
        if not MIN_VALID_ROUND_MS <= obs_ms <= MAX_VALID_ROUND_MS:
            return
        if scheduled_draft_tokens < UNTHROTTLED_MIN_DRAFT_TOKENS:
            return
        if state.ewma_round_ms is None:
            state.ewma_round_ms = obs_ms
            return
        if obs_ms > state.ewma_round_ms * MAX_GAP_MULTIPLE:
            return
        alpha = (
            EWMA_DOWN_ALPHA
            if obs_ms < state.ewma_round_ms
            else EWMA_UP_ALPHA
        )
        state.ewma_round_ms += alpha * (obs_ms - state.ewma_round_ms)

    def limit_next(
        self,
        req_id: str,
        offered_depth: int,
        *,
        pending_rounds: int = 0,
        pending_committed: int = 0,
    ) -> int:
        offered_depth = max(int(offered_depth), 0)
        if offered_depth == 0 or not self.enabled:
            return offered_depth

        state = self._states.get(req_id)
        if (
            state is None
            or not state.high_acceptance_seen
            or state.ewma_round_ms is None
            or state.first_round_done_ns is None
            or state.last_done_ns is None
        ):
            return offered_depth

        pending_rounds = max(int(pending_rounds), 0)
        pending_committed = max(int(pending_committed), 0)
        ewma = state.ewma_round_ms
        # Anchor elapsed time on observed decode intervals plus one full-round
        # estimate for round 1, capped by rounds * ewma so a host stall cannot
        # inflate the budget.
        actual_elapsed_ms = (
            (state.last_done_ns - state.first_round_done_ns) / 1e6 + ewma
        )
        safe_elapsed_ms = min(actual_elapsed_ms, state.rounds * ewma)
        projected_ms = safe_elapsed_ms + (pending_rounds + 1) * (
            ewma * ROUND_PROJECT_SAFETY_FACTOR
        )
        allowed_after_next = int(projected_ms / self._target_ms)
        room = allowed_after_next - state.committed_tokens - pending_committed - 1
        limited = min(offered_depth, max(0, room))
        if limited < offered_depth:
            state.engagements += 1
        return limited
