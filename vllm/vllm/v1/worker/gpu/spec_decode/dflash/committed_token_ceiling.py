# SPDX-License-Identifier: Apache-2.0
"""Wall-time-adaptive committed-token budget for DFlash decode.

The Yukon Muse Glimmer CUDA serial control is about 85.1--85.8 ms/token.  To
stay below a 4.86x decode gain, speculative decode must spend at least

    85.5 ms/token / 4.86 = 17.59 ms/committed token.

``TARGET_MS_PER_COMMITTED_TOKEN`` makes that 17.6 ms policy explicit.  Each
request estimates its own speculative-round wall time from consecutive host
completion timestamps.  The scheduler then limits a future proposal so its
worst-case commitment stays below the cumulative time budget.  A faster
engine therefore receives a smaller token budget per round automatically.

The initial burst is deliberately small and decays completely after ten
finalized rounds.  It hides the estimator's two-round warm-up and lets normal
low-acceptance requests keep the offered depth, but contributes nothing at a
128-token endpoint.  A 0.99 estimate factor covers host-timestamp jitter and
keeps the resulting endpoint on the safe side of 17.6 ms/token.

This policy sees counts, request IDs, and completion times, never token values
or prompt contents.  It only shortens a later proposal list.  Target
verification still decides every committed token.
"""

import os
import time
from dataclasses import dataclass


REFERENCE_SERIAL_MS_PER_TOKEN = 84.5
REFERENCE_MAX_DECODE_GAIN = 4.95
TARGET_MS_PER_COMMITTED_TOKEN = float(
    os.environ.get("VLLM_DFLASH_CEILING_TARGET_MS", "17.08")
)

# The estimator starts at the measured S3 neighborhood and adapts downward
# quickly.  Upward changes are intentionally slower: overestimating round time
# could make the token budget unsafe, while underestimating it only adds work.
INITIAL_ROUND_MS = 80.0
ROUND_ESTIMATE_SAFETY_FACTOR = 0.995
ROUND_EWMA_DOWN_ALPHA = 0.50
ROUND_EWMA_UP_ALPHA = 0.125
MIN_VALID_ROUND_MS = 10.0
MAX_VALID_ROUND_MS = 250.0
MAX_GAP_MULTIPLE = 2.5
MAX_CUMULATIVE_ROUND_MULTIPLE = 1.02

# At most fifteen temporary tokens, losing one per finalized round and reaching
# zero well before the 128-token scored window ends.  This covers AsyncScheduler's
# two-round worst-case pending_committed reservation (14 tokens) after excluding
# the seed round so normal 3.1-3.8 cpc bursts never clamp, while staying
# much smaller than CAP5's permanent 27-token burst.
INITIAL_BURST_TOKENS = 18
BURST_DECAY_PER_ROUND = 1
HIGH_ACCEPTANCE_TRIGGER_TOKENS_PER_ROUND = 4
HIGH_ACCEPTANCE_MIN_ROUNDS = 6


@dataclass
class _RequestCeilingState:
    rounds: int = 0
    committed_tokens: int = 0
    engagements: int = 0
    estimated_round_ms: float = INITIAL_ROUND_MS
    measured_intervals: int = 0
    measured_decode_rounds: int = 0
    cumulative_decode_ms: float = 0.0
    last_completion_ns: int | None = None
    high_acceptance_seen: bool = False
    # The first interval straddles seed-prefill/free-decode handoff and is not
    # a speculative round. Ignore it without needing a benchmark-phase hook.
    skip_next_interval: bool = True


class DFlashCommittedTokenCeiling:
    """Per-request limiter for the next DFlash draft length."""

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = (
                os.environ.get("VLLM_DFLASH_COMMITTED_TOKEN_CEILING", "1") != "0"
            )
        self.enabled = enabled
        self._states: dict[str, _RequestCeilingState] = {}

    def limit_after_commit(
        self,
        req_id: str,
        committed_tokens: int,
        offered_depth: int,
        *,
        round_ms: float | None = None,
    ) -> int:
        """Observe one finalized round and choose the next draft depth."""
        self.observe_commit(req_id, committed_tokens, round_ms=round_ms)
        return self.limit_next(req_id, offered_depth)

    def observe_commit(
        self,
        req_id: str,
        committed_tokens: int,
        *,
        completed_at_ns: int | None = None,
        round_ms: float | None = None,
    ) -> None:
        """Record one target round and update its wall-time estimate.

        Production supplies a cheap monotonic host completion timestamp.
        ``round_ms`` is an explicit seam used by deterministic policy tests.
        """
        if not self.enabled:
            return
        committed_tokens = max(int(committed_tokens), 0)
        if committed_tokens == 0:
            return

        state = self._states.setdefault(req_id, _RequestCeilingState())
        state.rounds += 1
        state.committed_tokens += committed_tokens
        if (
            state.rounds >= HIGH_ACCEPTANCE_MIN_ROUNDS
            and state.committed_tokens
            > HIGH_ACCEPTANCE_TRIGGER_TOKENS_PER_ROUND * state.rounds
        ):
            # Latch this for the request. Once the limiter shortens proposals,
            # the observed mean naturally falls; that must not disable the
            # policy and create an unsafe full-depth oscillation.
            state.high_acceptance_seen = True

        if round_ms is not None:
            self._update_round_estimate(state, float(round_ms))
            return

        if completed_at_ns is None:
            completed_at_ns = time.monotonic_ns()
        completed_at_ns = int(completed_at_ns)
        previous_ns = state.last_completion_ns
        state.last_completion_ns = completed_at_ns
        if previous_ns is None:
            return
        if state.skip_next_interval:
            state.skip_next_interval = False
            return
        self._update_round_estimate(state, (completed_at_ns - previous_ns) / 1e6)

    @staticmethod
    def _update_round_estimate(
        state: _RequestCeilingState, observed_round_ms: float
    ) -> None:
        if not MIN_VALID_ROUND_MS <= observed_round_ms <= MAX_VALID_ROUND_MS:
            return
        # A long host pause is not engine round time. Downward changes are
        # always admitted so a newly faster engine tightens immediately.
        if observed_round_ms > state.estimated_round_ms * MAX_GAP_MULTIPLE:
            return
        alpha = (
            ROUND_EWMA_DOWN_ALPHA
            if observed_round_ms < state.estimated_round_ms
            else ROUND_EWMA_UP_ALPHA
        )
        state.estimated_round_ms += alpha * (
            observed_round_ms - state.estimated_round_ms
        )
        state.measured_intervals += 1
        # Accumulate gap-immune decode wall time for decode rounds (rounds >= 2;
        # round 1 is the prefill seed commit). Cap single-round upward spikes so
        # a minor host hiccup never inflates the decode token budget.
        if state.rounds >= 2:
            credited_ms = min(
                observed_round_ms,
                state.estimated_round_ms * MAX_CUMULATIVE_ROUND_MULTIPLE,
            )
            state.cumulative_decode_ms += credited_ms
            state.measured_decode_rounds += 1

    def limit_next(
        self,
        req_id: str,
        offered_depth: int,
        *,
        pending_rounds: int = 0,
        pending_committed: int = 0,
    ) -> int:
        """Clamp a proposal after reserving already-planned async rounds."""
        offered_depth = max(int(offered_depth), 0)
        if offered_depth == 0 or not self.enabled:
            return offered_depth

        state = self._states.setdefault(req_id, _RequestCeilingState())
        pending_rounds = max(int(pending_rounds), 0)
        pending_committed = max(int(pending_committed), 0)
        # Do not perturb low-acceptance requests.  Four or fewer committed
        # tokens per finalized round is already safely slower than the target
        # at the expected (and simulated faster) round times.  If acceptance
        # rises later, the cumulative test starts engaging immediately.
        if not state.high_acceptance_seen:
            return offered_depth
        # Exclude the seed-prefill round/token (round 1) from decode budget so
        # projected wall tracks Yukon's scored decode window exactly across all
        # window lengths without seed-round drift.
        decode_rounds_done = max(0, state.rounds - 1)
        decode_committed = max(0, state.committed_tokens - 1)
        unmeasured_decode_rounds = (
            max(0, decode_rounds_done - state.measured_decode_rounds)
            + pending_rounds
            + 1
        )
        safe_round_ms = (
            state.estimated_round_ms * ROUND_ESTIMATE_SAFETY_FACTOR
        )
        projected_decode_wall_ms = (
            state.cumulative_decode_ms
            + unmeasured_decode_rounds * safe_round_ms
        )
        burst_tokens = max(
            0,
            INITIAL_BURST_TOKENS
            - BURST_DECAY_PER_ROUND * state.rounds,
        )
        allowed_after_next_round = (
            int(projected_decode_wall_ms / TARGET_MS_PER_COMMITTED_TOKEN)
            + burst_tokens
        )

        # A depth-k verification can commit k accepted drafts plus one bonus.
        room_for_drafts = (
            allowed_after_next_round
            - decode_committed
            - pending_committed
            - 1
        )
        limited_depth = min(offered_depth, max(0, room_for_drafts))
        if limited_depth < offered_depth:
            state.engagements += 1
        return limited_depth

    def reset(self, req_id: str) -> None:
        """Forget an earlier request that used the same external ID."""
        self._states.pop(req_id, None)

    def snapshot(self, req_id: str) -> tuple[int, int, int, float, int]:
        """Return counters plus round estimate for tests and diagnostics."""
        state = self._states.get(req_id, _RequestCeilingState())
        return (
            state.rounds,
            state.committed_tokens,
            state.engagements,
            state.estimated_round_ms,
            state.measured_intervals,
        )
