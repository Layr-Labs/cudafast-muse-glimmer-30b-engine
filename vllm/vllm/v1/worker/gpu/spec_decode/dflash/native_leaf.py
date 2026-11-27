# SPDX-License-Identifier: Apache-2.0
"""Zero-overhead native 1-leaf branch for depth-6 DFlash verify (proposals only; target exact).

Replaces verify row 6 (d6, rarely reached in low-acceptance regimes) with `alt`, the drafter's
2nd-best token for position p+1, when the prior verify committed <= MUSE_NATIVE_LEAF_THRESH
tokens (default 3). Unlike side-stream 2-row attention (+1.8 ms/step), this uses a 5-line
logical KV-index remap inside gqa16_fa2_block128_pipe_kernel<true>:
  - Row 6 writes its RoPE(p+1) K/V directly into slot p+6 (never read by causal rows 0..5).
  - gqa16_verify runs row 6 with seqlen_k = p+2 and remaps tok==(p+1) -> (p+6) for both K and V,
    yielding 0-ULP bit-exact attention (identical window, tiles, MMA lanes) at ~0.00 ms overhead.
  - On a leaf hit (d1 missed and target(t0) == alt): commits [alt, u], copies row 6 -> row 1 of
    hidden/aux states, and copies slot(p+6) -> slot(p+1) across all 52 target layers in a single
    52-CTA x 64-thread kernel (~4 us on hit, ~1.5 us early-exit on miss).

Controls:
  MUSE_NATIVE_LEAF=1          enable (0 disables completely)
  MUSE_NATIVE_LEAF_THRESH=3   activate leaf mode when prior verify committed <= thresh
  MUSE_NATIVE_LEAF_SELFTEST=0 1 places d1 in row 6 to verify bitwise logits[6] == logits[1]
  MUSE_NATIVE_LEAF_NOHIT=0    1 disables hit promotion (diagnostic)
"""
from __future__ import annotations

import atexit
import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_TREE_M = "16"
DEFAULT_TREE_WIDE_M = "32"
ENABLED = os.environ.get("MUSE_NATIVE_LEAF", "1").strip() not in ("0", "", "false", "no")
M8_ENABLED = os.environ.get("MUSE_NATIVE_LEAF_M8", "1").strip() not in ("0", "", "false", "no")
try:
    _TREE_M_RAW = int(os.environ.get("MUSE_NATIVE_TREE_M", DEFAULT_TREE_M).strip() or "0")
except ValueError:
    _TREE_M_RAW = 0
TREE_M = _TREE_M_RAW if _TREE_M_RAW in (12, 16, 20, 24, 32) else 0
try:
    _TREE_WIDE_M_RAW = int(os.environ.get("MUSE_NATIVE_TREE_WIDE_M", DEFAULT_TREE_WIDE_M).strip() or "0")
except ValueError:
    _TREE_WIDE_M_RAW = 0
TREE_WIDE_M = (
    _TREE_WIDE_M_RAW
    if (_TREE_WIDE_M_RAW in (20, 24, 32) and TREE_M > 0 and _TREE_WIDE_M_RAW > TREE_M)
    else 0
)
# Default "always": every bs=1 depth-6 verify round runs the wide (M=32) tree. The "cpc" gate below reads
# decoded_tokens from the scheduler's optimistic CPU mirror (num_computed_tokens - prefill_len), which under the
# batch-queue async scheduler already counts the in-flight round's 7 scheduled tokens. At round k it therefore
# compares (C[k-2] + 7) / (k-1) with TREE_WIDE_CPC instead of the true running rate C[k-1] / (k-1): a prompt at
# ~3.0 committed tokens/round only goes wide after round ~17 and one at ~3.2 practically never does. Neither width
# changes a committed token (both are captured as FULL graphs and admitted at boot; the target's greedy verify
# decides every commit), and the M=32 tree is the M=16 tree plus 16 more best-first leaves, so it only trades about
# 1 ms of verify time per round for a longer expected accept. "cpc" and "sm" stay selectable through the env var.
TREE_WIDE_GATE = os.environ.get("MUSE_NATIVE_TREE_WIDE_GATE", "always").strip().lower()
TREE_WIDE_CPC = float(os.environ.get("MUSE_NATIVE_TREE_WIDE_CPC", "3.25"))
TREE_WIDE_MIN_ROUNDS = int(os.environ.get("MUSE_NATIVE_TREE_WIDE_MIN_ROUNDS", "3"))
TREE_WIDE_SM = float(os.environ.get("MUSE_NATIVE_TREE_WIDE_SM", "2.0"))


def desired_verify_ms() -> tuple[int, ...]:
    """Configured bs=1 depth-6 verify widths > 7 to capture as FULL CUDA graphs."""
    if not ENABLED:
        return ()
    if TREE_M in (12, 16, 20, 24, 32):
        if TREE_WIDE_M > TREE_M:
            return (TREE_M, TREE_WIDE_M)
        return (TREE_M,)
    if M8_ENABLED:
        return (8,)
    return ()


def desired_verify_m() -> int:
    """Single source of truth for max configured bs=1 depth-6 verify width (32..7)."""
    ms = desired_verify_ms()
    return max(ms) if ms else 7
LOW_THRESH = int(os.environ.get("MUSE_NATIVE_LEAF_THRESH", "3"))
_POLICY_RAW = os.environ.get("MUSE_NATIVE_LEAF_POLICY", "d1").strip().lower()
POLICY = 1 if _POLICY_RAW == "margin" else 0
MARGIN_ALPHA = float(os.environ.get("MUSE_NATIVE_LEAF_MARGIN_ALPHA", "1.0"))
TREE_DYNAMIC = os.environ.get("MUSE_TREE_DYNAMIC", "1").strip() not in ("0", "", "false", "no")
TREE_TEMP = float(os.environ.get("MUSE_TREE_TEMP", "1.35"))
TREE_TAIL = float(os.environ.get("MUSE_TREE_TAIL", "4.0"))
TREE_GAMMA = float(os.environ.get("MUSE_TREE_GAMMA", "0.92"))
TREE_MAX_DEPTH = max(1, min(6, int(os.environ.get("MUSE_TREE_MAX_DEPTH", "6"))))
TREE_SUB_COEF = float(os.environ.get("MUSE_TREE_SUB_COEF", "0.55"))
TREE_SHF_COEF = float(os.environ.get("MUSE_TREE_SHF_COEF", "0.20"))
TREE_CONT_DECAY = float(os.environ.get("MUSE_TREE_CONT_DECAY", "0.88"))
TREE_ADAPT_GAMMA = os.environ.get("MUSE_TREE_ADAPT_GAMMA", "1").strip() not in ("0", "", "false", "no")
TREE_CALIB = os.environ.get("MUSE_TREE_CALIB", "1").strip() not in ("0", "", "false", "no")
TREE_HIGH_ACC_CPC = float(os.environ.get("MUSE_TREE_HIGH_ACC_CPC", "0"))
LEAF_STATS = os.environ.get("MUSE_LEAF_STATS", "0").strip() in ("1", "true", "yes")
TR_ENABLED = os.environ.get("MUSE_TR_MODE", "1").strip() not in ("0", "", "false", "no")
OOV_PREADMIT = os.environ.get("MUSE_OOV_PREADMIT", "2").strip() not in ("0", "", "false", "no")
try:
    _TREE_K_RAW = int(os.environ.get("MUSE_TREE_K", "8").strip() or "8")
except ValueError:
    _TREE_K_RAW = 8
TREE_K = 8 if _TREE_K_RAW >= 8 else 6
TREE_TOP_LEN = 6 * TREE_K
HIST_CAP = 16384
TR_LOG_CAP = 4096
_ST_RAW = os.environ.get("MUSE_NATIVE_LEAF_SELFTEST", "0").strip().lower()
SELFTEST = 1 if _ST_RAW in ("1", "true", "yes") else (2 if _ST_RAW == "2" else 0)
NOHIT = os.environ.get("MUSE_NATIVE_LEAF_NOHIT", "0").strip() in ("1", "true", "yes")


class _NativeLeafState:
    def __init__(self) -> None:
        self.ready: bool = False
        self.m8_ready: bool = False
        self.tree_m_ready: int = 0
        self.tree_wide_m_ready: int = 0
        self.vocab_size: int = 202048
        self.device: torch.device | None = None
        self.leaf_flag: torch.Tensor | None = None          # int32[97] shared with muse_gqa16_attn
        self.prev_ns: torch.Tensor | None = None            # int32[1]
        self.hit_flag: torch.Tensor | None = None           # int32[14] [dst0, hit_len, dst0, src0, ..., dst5, src5]
        self.hit_src_row: torch.Tensor | None = None        # int32[1] (0=miss, r=winning initial leaf row 7..31)
        self.alt_ids: torch.Tensor | None = None            # int64[4] (d1_b1, d1_b2, d2_b1, d2_b2)
        self.alt_margins: torch.Tensor | None = None        # float32[2] (margin_d1, margin_d2)
        self.tree_top: torch.Tensor | None = None           # int64[36] ([6, 6] top-6 for d1..d6)
        self.tree_vals: torch.Tensor | None = None          # float32[36] ([6, 6] top-6 logits for d1..d6)
        self.req_acc_state: torch.Tensor | None = None      # int32[3] [rounds, committed, high_acc_flag]
        self.saved_alt_tok: torch.Tensor | None = None      # int64[1] (legacy M=7/M=8)
        self.saved_leaf_toks: torch.Tensor | None = None    # int64[32] (tree mode)
        self.saved_leaf_k: torch.Tensor | None = None       # int32[32] (tree mode)
        self.saved_leaf_parent: torch.Tensor | None = None  # int32[32] (-1=1-level root, 7..31=parent row)
        self.hist: torch.Tensor | None = None               # int32[16384] exact committed token history
        self.hist_len: torch.Tensor | None = None           # int32[1] valid history length
        self.saved_pos0: torch.Tensor | None = None         # int64[1] step anchor position p0
        self.saved_row_ctx: torch.Tensor | None = None      # int32[32, 4] per-row (cur, prv1, prv2, pos_r)
        self.tr_log: torch.Tensor | None = None             # int32[4096, 8] per-request 30B target recycle log
        self.tr_state: torch.Tensor | None = None           # int32[4] [tr_len, tr_step, last_p0, reserved]
        self.in_static_vocab: torch.Tensor | None = None    # uint8[vocab] drafter static vocab mask
        self.tail_ids: torch.Tensor | None = None           # int32[512] drafter dynamic OOV tail IDs
        self.tail_count: torch.Tensor | None = None         # int32[1] drafter dynamic OOV tail count
        self.tail_seen: torch.Tensor | None = None          # int32[vocab] drafter OOV dedup bitmap
        self.stats: torch.Tensor | None = None              # int64[52] telemetry counters
        self.cascade_meta: torch.Tensor | None = None       # int32[8] Pass-2 cascade metadata & row mask
        self.cascade_toks: torch.Tensor | None = None       # int64[36] Pass-2 / backup top-6 tokens
        self.cascade_vals: torch.Tensor | None = None       # float32[36] Pass-2 / backup top-6 logits
        self.p1_save_draft: torch.Tensor | None = None      # int64[6] saved Pass-1 draft_tokens
        self.p1_save_top: torch.Tensor | None = None        # int64[36] saved Pass-1 tree_top
        self.p1_save_vals: torch.Tensor | None = None       # float32[36] saved Pass-1 tree_vals
        self.p1_save_alt: torch.Tensor | None = None        # int64[4] saved Pass-1 alt_ids
        self.p1_save_margins: torch.Tensor | None = None    # float32[2] saved Pass-1 alt_margins
        # Static 52-layer target KV metadata for 1-launch all-layer KV fix on hit.
        self.num_layers: int = 0
        self.kv_ptrs: torch.Tensor | None = None
        self.stride_block: torch.Tensor | None = None
        self.stride_head: torch.Tensor | None = None
        self.stride_slot: torch.Tensor | None = None
        self.block_size: torch.Tensor | None = None
        self.layer_gid: torch.Tensor | None = None
        # Cached device pointer array for hidden_states + aux_hidden_states (0 H2D copies steady state).
        self._hid_ptr_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._last_req_id: str | None = None
        self._hist_req_id: str | None = None
        self._host_req_id: str | None = None
        self._host_rounds: int = 0
        self._logged_active: bool = False
        self._atexit_registered: bool = False


_S = _NativeLeafState()


def register_draft_tail_bufs(
    in_static_vocab: torch.Tensor | None,
    tail_ids: torch.Tensor | None,
    tail_count: torch.Tensor | None,
    tail_seen: torch.Tensor | None,
) -> None:
    """Register drafter lm_head OOV tail buffers for 32-row 30B cone OOV pre-admission."""
    if (
        not OOV_PREADMIT
        or in_static_vocab is None
        or tail_ids is None
        or tail_count is None
        or tail_seen is None
    ):
        return
    if _S.tail_ids is tail_ids and _S.in_static_vocab is in_static_vocab:
        return
    _S.in_static_vocab = in_static_vocab
    _S.tail_ids = tail_ids
    _S.tail_count = tail_count
    _S.tail_seen = tail_seen


def _reset_draft_tail_state() -> None:
    if torch.cuda.is_current_stream_capturing():
        return
    if _S.tail_ids is not None and not _S.tail_ids.is_inference():
        _S.tail_ids.fill_(-1)
    if _S.tail_count is not None and not _S.tail_count.is_inference():
        _S.tail_count.zero_()
    if _S.tail_seen is not None and not _S.tail_seen.is_inference():
        _S.tail_seen.zero_()


def _reset_tr_state() -> None:
    if _S.tr_state is not None:
        _S.tr_state[:2].zero_()
        _S.tr_state[2:3].fill_(-1)
    if _S.saved_row_ctx is not None:
        _S.saved_row_ctx.fill_(-1)
    _reset_draft_tail_state()


def _dump_stats(tag: str) -> None:
    if not LEAF_STATS or not _S.ready or _S.stats is None:
        return
    try:
        st = _S.stats.detach().cpu().tolist()
        if len(st) >= 32 and st[0] > 0:
            pl_extra = ""
            if len(st) >= 36:
                pl_extra = f" hist_ok/bad={st[32]}/{st[33]} pl_roots={st[34]} pl_boosts={st[35]}"
            if len(st) >= 40:
                pl_extra += f" tr_boosts={st[36]} tr_roots={st[37]} tr_conts={st[38]} tr_rows={st[39]}"
            if len(st) >= 48 and st[40] > 0:
                pl_extra += (
                    f" casc[ev={st[40]} br={st[41]} dn={st[42]} "
                    f"pl={st[43]} hit={st[44]} ex={st[45]} dn_chg/hit={st[46]}/{st[47]}]"
                )
            logger.info(
                "native_leaf_stats[%s]: steps=%d active=%d hits=%d deep_hits=%d extra_toks=%d deep_bonus=%d "
                "total_ns=%d max_len=%d len1..6=%s k1..6=%s slots0..8=%s reached_d2/3/4+=%d/%d/%d%s",
                tag, st[0], st[1], st[2], st[3], st[25], st[26], st[27], st[28],
                st[19:25], st[4:10], st[10:19], st[29], st[30], st[31], pl_extra,
            )
            _S.stats.zero_()
    except Exception:  # noqa: BLE001
        pass


def is_ready() -> bool:
    return ENABLED and _S.ready


def is_m8_ready() -> bool:
    return ENABLED and M8_ENABLED and _S.ready and _S.m8_ready


def active_verify_m() -> int:
    """Return active base verify width for bs=1 depth-6 verify: 32, 24, 20, 16, 12, 8, or 7."""
    if not (ENABLED and _S.ready):
        return 7
    if _S.tree_m_ready in (12, 16, 20, 24, 32):
        return _S.tree_m_ready
    if M8_ENABLED and _S.m8_ready:
        return 8
    return 7


def select_verify_m(req_id: str | None, decoded_tokens: int) -> int:
    """Select per-step verify width (adaptive wide M=32 on hard steps vs base M=16 on easy steps).

    Default `cpc` gate is 100% zero-sync on the host: uses `decoded_tokens` (from
    `req_states.num_computed_tokens_np - prefill_len`) divided by host-counted verify rounds.
    """
    base_m = active_verify_m()
    wide_m = _S.tree_wide_m_ready
    if wide_m <= base_m or req_id is None:
        return base_m
    if req_id != _S._host_req_id or decoded_tokens <= 0:
        _S._host_req_id = req_id
        _S._host_rounds = 0
    if TREE_WIDE_GATE == "always":
        return wide_m
    if TREE_WIDE_GATE == "sm":
        if _S._host_rounds == 0 or _S.tree_vals is None or _S.tree_vals.numel() < 36:
            return wide_m
        _tk = max(6, int(_S.tree_vals.numel()) // 6)
        sm_val = float((_S.tree_vals[0 : 6 * _tk : _tk] - _S.tree_vals[1 : 6 * _tk : _tk]).mean().item())
        return wide_m if sm_val < TREE_WIDE_SM else base_m
    # Default zero-sync `cpc` gate: start on fast M=16 for the first TREE_WIDE_MIN_ROUNDS steps,
    # then switch to wide M=32 only when running verify cpc < TREE_WIDE_CPC (hard <3.25 tok/step prose!).
    if _S._host_rounds < TREE_WIDE_MIN_ROUNDS:
        return base_m
    verify_committed = max(0, int(decoded_tokens))
    return wide_m if (float(verify_committed) < TREE_WIDE_CPC * float(_S._host_rounds)) else base_m


def _get_bfx():
    try:
        import muse_bfx_ext as _bfx

        if (
            hasattr(_bfx, "native_leaf_setup")
            and hasattr(_bfx, "native_leaf_commit")
            and hasattr(_bfx, "native_leaf_fix_hidden")
            and hasattr(_bfx, "native_leaf_fix_kv")
        ):
            return _bfx
    except Exception:  # noqa: BLE001
        pass
    return None


def _get_gqa16():
    try:
        import muse_gqa16_attn as _g16

        if hasattr(_g16, "get_leaf_flag") and hasattr(_g16, "is_leaf_verify_admitted"):
            return _g16
    except Exception:  # noqa: BLE001
        pass
    return None


def get_alt_buf(device: torch.device) -> torch.Tensor | None:
    """Return fixed-address int64[4] buffer for drafter rows 0..1 top-2 (CUDA-graph safe)."""
    if not ENABLED or device.type != "cuda":
        return None
    if (
        _S.alt_ids is not None
        and _S.alt_ids.device == device
        and _S.alt_ids.numel() >= 4
        and not _S.alt_ids.is_inference()
    ):
        return _S.alt_ids
    if torch.cuda.is_current_stream_capturing():
        return _S.alt_ids if (_S.alt_ids is not None and _S.alt_ids.device == device) else None
    with torch.inference_mode(False):
        _S.alt_ids = torch.full((4,), -1, dtype=torch.int64, device=device)
    return _S.alt_ids


def get_margin_buf(device: torch.device) -> torch.Tensor | None:
    """Return fixed-address float32[2] buffer for drafter rows 0..1 logit margins (CUDA-graph safe)."""
    if not ENABLED or device.type != "cuda":
        return None
    if (
        _S.alt_margins is not None
        and _S.alt_margins.device == device
        and _S.alt_margins.numel() >= 2
        and not _S.alt_margins.is_inference()
    ):
        return _S.alt_margins
    if torch.cuda.is_current_stream_capturing():
        return _S.alt_margins if (_S.alt_margins is not None and _S.alt_margins.device == device) else None
    with torch.inference_mode(False):
        _S.alt_margins = torch.full((2,), 1e30, dtype=torch.float32, device=device)
    return _S.alt_margins


def get_tree_top_buf(device: torch.device) -> torch.Tensor | None:
    """Return fixed-address int64[6*TREE_K] buffer for drafter rows 0..5 top-K IDs (CUDA-graph safe)."""
    if not ENABLED or TREE_M == 0 or device.type != "cuda":
        return None
    if (
        _S.tree_top is not None
        and _S.tree_top.device == device
        and _S.tree_top.numel() >= TREE_TOP_LEN
        and not _S.tree_top.is_inference()
    ):
        return _S.tree_top
    if torch.cuda.is_current_stream_capturing():
        return _S.tree_top if (_S.tree_top is not None and _S.tree_top.device == device) else None
    with torch.inference_mode(False):
        _S.tree_top = torch.full((TREE_TOP_LEN,), -1, dtype=torch.int64, device=device)
    return _S.tree_top


def get_tree_vals_buf(device: torch.device) -> torch.Tensor | None:
    """Return fixed-address float32[6*TREE_K] buffer for drafter rows 0..5 top-K logits (CUDA-graph safe)."""
    if not ENABLED or TREE_M == 0 or device.type != "cuda":
        return None
    if (
        _S.tree_vals is not None
        and _S.tree_vals.device == device
        and _S.tree_vals.numel() >= TREE_TOP_LEN
        and not _S.tree_vals.is_inference()
    ):
        return _S.tree_vals
    if torch.cuda.is_current_stream_capturing():
        return _S.tree_vals if (_S.tree_vals is not None and _S.tree_vals.device == device) else None
    with torch.inference_mode(False):
        _S.tree_vals = torch.full((TREE_TOP_LEN,), -1e30, dtype=torch.float32, device=device)
    return _S.tree_vals


def get_cascade_bufs(
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
] | None:
    """Return fixed-address Pass-2 cascade buffers (CUDA-graph safe)."""
    if not ENABLED or TREE_M == 0 or device.type != "cuda":
        return None
    if (
        _S.cascade_meta is not None
        and _S.cascade_meta.device == device
        and not _S.cascade_meta.is_inference()
        and _S.cascade_toks is not None
        and _S.cascade_vals is not None
        and _S.p1_save_draft is not None
        and _S.p1_save_top is not None
        and _S.p1_save_vals is not None
        and _S.p1_save_alt is not None
        and _S.p1_save_margins is not None
        and _S.req_acc_state is not None
        and _S.stats is not None
        and _S.stats.numel() >= 52
    ):
        return (
            _S.cascade_meta,
            _S.cascade_toks,
            _S.cascade_vals,
            _S.p1_save_draft,
            _S.p1_save_top,
            _S.p1_save_vals,
            _S.p1_save_alt,
            _S.p1_save_margins,
            _S.req_acc_state,
            _S.stats,
        )
    if torch.cuda.is_current_stream_capturing():
        return None
    with torch.inference_mode(False):
        _S.cascade_meta = torch.zeros((8,), dtype=torch.int32, device=device)
        _S.cascade_toks = torch.full((36,), -1, dtype=torch.int64, device=device)
        _S.cascade_vals = torch.full((36,), -1e30, dtype=torch.float32, device=device)
        _S.p1_save_draft = torch.full((6,), -1, dtype=torch.int64, device=device)
        _S.p1_save_top = torch.full((36,), -1, dtype=torch.int64, device=device)
        _S.p1_save_vals = torch.full((36,), -1e30, dtype=torch.float32, device=device)
        _S.p1_save_alt = torch.full((4,), -1, dtype=torch.int64, device=device)
        _S.p1_save_margins = torch.full((2,), 1e30, dtype=torch.float32, device=device)
        if (
            _S.req_acc_state is None
            or _S.req_acc_state.device != device
            or _S.req_acc_state.numel() < 3
            or _S.req_acc_state.is_inference()
        ):
            _S.req_acc_state = torch.zeros((3,), dtype=torch.int32, device=device)
        if (
            _S.stats is None
            or _S.stats.device != device
            or _S.stats.numel() < 52
            or _S.stats.is_inference()
        ):
            _S.stats = torch.zeros((52,), dtype=torch.int64, device=device)
    return (
        _S.cascade_meta,
        _S.cascade_toks,
        _S.cascade_vals,
        _S.p1_save_draft,
        _S.p1_save_top,
        _S.p1_save_vals,
        _S.p1_save_alt,
        _S.p1_save_margins,
        _S.req_acc_state,
        _S.stats,
    )


def _reset_hist_for_req(req_id: str | None) -> None:
    if req_id == _S._hist_req_id and req_id is not None:
        return
    _S._hist_req_id = req_id
    if _S.hist_len is not None:
        _S.hist_len.zero_()
    if _S.saved_pos0 is not None:
        _S.saved_pos0.fill_(-1)
    _reset_tr_state()


def get_hist_bufs(
    device: torch.device,
    req_id: str | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return fixed-address (hist int32[16384], hist_len int32[1]) buffers for exact committed token history."""
    if not (ENABLED and _S.ready) or device.type != "cuda":
        return None, None
    if (
        _S.hist is not None
        and _S.hist_len is not None
        and _S.hist.device == device
        and _S.hist_len.device == device
    ):
        if req_id is not None and req_id != _S._hist_req_id:
            _reset_hist_for_req(req_id)
        return _S.hist, _S.hist_len
    return None, None


def init_kv_table(
    kv_caches_dict: dict[str, torch.Tensor],
    kv_cache_config: Any,
    draft_attn_layer_names: set[str] | list[str] | None,
    device: torch.device,
    vocab_size: int | None = None,
) -> bool:
    """Build static 52-layer KV pointer table once after KV cache init."""
    _S.ready = False
    if not ENABLED or device.type != "cuda":
        return False
    bfx = _get_bfx()
    g16 = _get_gqa16()
    if bfx is None or g16 is None:
        logger.info("native_leaf: disabled (missing muse_bfx_ext or muse_gqa16_attn leaf symbols)")
        return False
    if not g16.is_leaf_verify_admitted(device):
        logger.info("native_leaf: disabled (gqa16 native leaf verify not admitted)")
        return False
    lf = g16.get_leaf_flag(device)
    if lf is None:
        return False

    draft_set = set(draft_attn_layer_names or ())
    name_to_gid: dict[str, int] = {}
    for gi, grp in enumerate(kv_cache_config.kv_cache_groups):
        for ln in grp.layer_names:
            name_to_gid[ln] = gi

    kv_ptrs: list[int] = []
    s_blk: list[int] = []
    s_head: list[int] = []
    s_slot: list[int] = []
    b_sz: list[int] = []
    gids: list[int] = []

    for lname, kv in kv_caches_dict.items():
        if lname in draft_set:
            continue
        gi = name_to_gid.get(lname)
        if gi is None:
            continue
        if (
            not isinstance(kv, torch.Tensor)
            or not kv.is_cuda
            or kv.dtype != torch.bfloat16
            or kv.ndim != 4
            or kv.shape[1] != 2
            or kv.shape[3] != 256
            or kv.stride(3) != 1
        ):
            # Target model has a non-GQA-16 layer -> cannot guarantee all 52 layers use gqa16_verify!
            logger.warning(
                "native_leaf: target layer %s has unsupported kv shape=%s stride=%s; disabling",
                lname, tuple(kv.shape), kv.stride(),
            )
            return False
        bs = int(kv.shape[2])
        if bs < 16 or (bs & 15) != 0:
            return False
        ptr = kv.data_ptr()
        sb, sh, ss = int(kv.stride(0)), int(kv.stride(1)), int(kv.stride(2))
        if (ptr & 15) or (sb & 7) or (sh & 7) or (ss & 7):
            return False
        kv_ptrs.append(ptr)
        s_blk.append(sb)
        s_head.append(sh)
        s_slot.append(ss)
        b_sz.append(bs)
        gids.append(gi)

    if not kv_ptrs:
        return False

    with torch.inference_mode(False):
        _S.device = device
        if vocab_size is not None and int(vocab_size) > 0:
            _S.vocab_size = int(vocab_size)
        _S.leaf_flag = lf
        _S.leaf_flag.zero_()
        _S.prev_ns = torch.full((1,), 7, dtype=torch.int32, device=device)
        _S.hit_flag = torch.zeros((14,), dtype=torch.int32, device=device)
        _S.hit_src_row = torch.zeros((1,), dtype=torch.int32, device=device)
        if (
            _S.alt_ids is None
            or _S.alt_ids.device != device
            or _S.alt_ids.numel() < 4
            or _S.alt_ids.is_inference()
        ):
            _S.alt_ids = torch.full((4,), -1, dtype=torch.int64, device=device)
        else:
            _S.alt_ids.fill_(-1)
        if (
            _S.alt_margins is None
            or _S.alt_margins.device != device
            or _S.alt_margins.numel() < 2
            or _S.alt_margins.is_inference()
        ):
            _S.alt_margins = torch.full((2,), 1e30, dtype=torch.float32, device=device)
        else:
            _S.alt_margins.fill_(1e30)
        if (
            _S.tree_top is None
            or _S.tree_top.device != device
            or _S.tree_top.numel() < TREE_TOP_LEN
            or _S.tree_top.is_inference()
        ):
            _S.tree_top = torch.full((TREE_TOP_LEN,), -1, dtype=torch.int64, device=device)
        else:
            _S.tree_top.fill_(-1)
        if (
            _S.tree_vals is None
            or _S.tree_vals.device != device
            or _S.tree_vals.numel() < TREE_TOP_LEN
            or _S.tree_vals.is_inference()
        ):
            _S.tree_vals = torch.full((TREE_TOP_LEN,), -1e30, dtype=torch.float32, device=device)
        else:
            _S.tree_vals.fill_(-1e30)
        if (
            _S.req_acc_state is None
            or _S.req_acc_state.device != device
            or _S.req_acc_state.numel() < 3
            or _S.req_acc_state.is_inference()
        ):
            _S.req_acc_state = torch.zeros((3,), dtype=torch.int32, device=device)
        else:
            _S.req_acc_state.zero_()
        _S.saved_alt_tok = torch.full((1,), -1, dtype=torch.int64, device=device)
        _S.saved_leaf_toks = torch.full((32,), -1, dtype=torch.int64, device=device)
        _S.saved_leaf_k = torch.zeros((32,), dtype=torch.int32, device=device)
        _S.saved_leaf_parent = torch.full((32,), -1, dtype=torch.int32, device=device)
        _S.hist = torch.full((HIST_CAP,), -1, dtype=torch.int32, device=device)
        _S.hist_len = torch.zeros((1,), dtype=torch.int32, device=device)
        _S.saved_pos0 = torch.full((1,), -1, dtype=torch.int64, device=device)
        _S.saved_row_ctx = torch.full((32, 4), -1, dtype=torch.int32, device=device)
        _S.tr_log = torch.zeros((TR_LOG_CAP, 8), dtype=torch.int32, device=device)
        _S.tr_state = torch.tensor([0, 0, -1, 0], dtype=torch.int32, device=device)
        if (
            _S.stats is None
            or _S.stats.device != device
            or _S.stats.numel() < 52
            or _S.stats.is_inference()
        ):
            _S.stats = torch.zeros((52,), dtype=torch.int64, device=device)
        else:
            _S.stats.zero_()
        get_cascade_bufs(device)
        _S.num_layers = len(kv_ptrs)
        _S.kv_ptrs = torch.tensor(kv_ptrs, dtype=torch.int64, device=device)
        _S.stride_block = torch.tensor(s_blk, dtype=torch.int64, device=device)
        _S.stride_head = torch.tensor(s_head, dtype=torch.int64, device=device)
        _S.stride_slot = torch.tensor(s_slot, dtype=torch.int64, device=device)
        _S.block_size = torch.tensor(b_sz, dtype=torch.int32, device=device)
        _S.layer_gid = torch.tensor(gids, dtype=torch.int32, device=device)
        _S._hid_ptr_cache.clear()
        _S._last_req_id = None
        _S._hist_req_id = None
        _S._host_req_id = None
        _S._host_rounds = 0
        _S.m8_ready = bool(M8_ENABLED and hasattr(bfx, "native_leaf_prepare_m8"))
        _has_tree_ops = (
            hasattr(bfx, "native_leaf_prepare_tree")
            and hasattr(bfx, "native_leaf_setup_tree")
            and hasattr(bfx, "native_leaf_commit_tree")
        )
        _S.tree_m_ready = (
            TREE_M
            if (
                TREE_M in (12, 16, 20, 24, 32)
                and lf.numel() >= 1 + 3 * TREE_M
                and _has_tree_ops
            )
            else 0
        )
        _S.tree_wide_m_ready = (
            TREE_WIDE_M
            if (
                _S.tree_m_ready > 0
                and TREE_WIDE_M > _S.tree_m_ready
                and lf.numel() >= 1 + 3 * TREE_WIDE_M
                and _has_tree_ops
            )
            else 0
        )
        _S.ready = True
        if LEAF_STATS and not _S._atexit_registered:
            _S._atexit_registered = True
            atexit.register(_dump_stats, "atexit")
    logger.info(
        "native_leaf: armed (%d target GQA-16 layers, tree_m=%d, wide_m=%d [gate=%s,cpc=%.2f,sm=%.2f], dynamic=%d [calib=%d,depth<=%d,T=%.2f,tail=%.1f,g=%.2f,adapt=%d,sub=%.2f,shf=%.2f,cont=%.2f,hi_cpc=%.1f], m8=%d, policy=%s, alpha=%.2f, thresh<=%d, selftest=%d, nohit=%d, stats=%d)",
        _S.num_layers, _S.tree_m_ready, _S.tree_wide_m_ready, TREE_WIDE_GATE, TREE_WIDE_CPC, TREE_WIDE_SM,
        int(TREE_DYNAMIC), int(TREE_CALIB), TREE_MAX_DEPTH, TREE_TEMP, TREE_TAIL, TREE_GAMMA,
        int(TREE_ADAPT_GAMMA), TREE_SUB_COEF, TREE_SHF_COEF, TREE_CONT_DECAY, TREE_HIGH_ACC_CPC,
        int(_S.m8_ready), _POLICY_RAW, MARGIN_ALPHA, LOW_THRESH, int(SELFTEST), int(NOHIT), int(LEAF_STATS),
    )
    return True


@torch.no_grad()
def on_non_leaf_step(
    req_id: str | None,
    num_draft_tokens: int,
    *,
    is_prefill: bool | None = None,
) -> None:
    """Clear leaf flag on any non-leaf step and reset state on request switch / prefill.

    Never permanently latch ``req_acc_state[2]`` off when ``committed_token_ceiling``
    shortens a single proposal (``0 <= num_draft_tokens < 6``): bypassing leaves on
    that shortened step is already handled by ``leaf_flag[0] = 0``, while subsequent
    ``num_draft_tokens == 6`` steps run ``FULL(16)`` at fixed cost and keep all 9
    speculative tree leaves active.
    """
    if not _S.ready:
        return
    assert (
        _S.leaf_flag is not None
        and _S.prev_ns is not None
        and _S.alt_ids is not None
        and _S.req_acc_state is not None
    )
    _S.leaf_flag[:1].zero_()
    is_req_switch = req_id is None or req_id != _S._last_req_id
    if req_id is None or req_id != _S._hist_req_id:
        _reset_hist_for_req(req_id)
    reset_req = is_req_switch or (
        num_draft_tokens == 0 if is_prefill is None else is_prefill
    )
    if reset_req:
        if LEAF_STATS and _S._last_req_id is not None:
            _dump_stats(str(_S._last_req_id))
        _S._last_req_id = req_id
        _S._host_req_id = req_id
        _S._host_rounds = 0
        _S.prev_ns.fill_(7)
        _S.alt_ids.fill_(-1)
        if _S.alt_margins is not None:
            _S.alt_margins.fill_(1e30)
        if _S.tree_top is not None:
            _S.tree_top.fill_(-1)
        if _S.tree_vals is not None:
            _S.tree_vals.fill_(-1e30)
        if _S.cascade_meta is not None:
            _S.cascade_meta.zero_()
        _S.req_acc_state.zero_()
        _reset_tr_state()


@torch.no_grad()
def prepare_m8(positions: torch.Tensor, seq_lens: torch.Tensor) -> bool:
    """Zero-sync GPU prep: positions[7] = positions[6] + 1, seq_lens[0] += 1 for M=8 slot mapping."""
    return prepare_tree(positions, seq_lens, 8)


@torch.no_grad()
def prepare_tree(positions: torch.Tensor, seq_lens: torch.Tensor, verify_m: int) -> bool:
    """Zero-sync GPU prep: positions[7..verify_m-1] = p+7..p+verify_m-1, seq_lens[0] += verify_m-7."""
    if not is_ready() or verify_m < 8 or verify_m > 32:
        return False
    bfx = _get_bfx()
    if bfx is None:
        return False
    if verify_m == 8 and hasattr(bfx, "native_leaf_prepare_m8"):
        bfx.native_leaf_prepare_m8(positions, seq_lens)
        return True
    if hasattr(bfx, "native_leaf_prepare_tree"):
        bfx.native_leaf_prepare_tree(positions, seq_lens, verify_m)
        return True
    return False


@torch.no_grad()
def setup_step(
    req_id: str,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    slot_mappings: torch.Tensor,
    m8_mode: bool = False,
    verify_m: int | None = None,
) -> bool:
    """Zero-sync GPU setup for a bs=1, depth=6 verify step (tree M=12/16, M=8 zero-clamp, or legacy M=7)."""
    if not _S.ready:
        return False
    bfx = _get_bfx()
    if bfx is None:
        return False
    assert (
        _S.leaf_flag is not None
        and _S.prev_ns is not None
        and _S.alt_ids is not None
        and _S.saved_alt_tok is not None
        and _S.req_acc_state is not None
    )
    eff_m = verify_m if verify_m is not None else (8 if m8_mode else 7)
    if (
        input_ids.dtype != torch.int32
        or input_ids.numel() < eff_m
        or positions.dtype != torch.int64
        or positions.numel() < eff_m
        or slot_mappings.dtype != torch.int64
        or slot_mappings.ndim != 2
        or slot_mappings.shape[1] < eff_m
        or slot_mappings.stride(1) != 1
    ):
        _S.leaf_flag[:1].zero_()
        return False
    if req_id != _S._hist_req_id:
        _reset_hist_for_req(req_id)
    if req_id != _S._last_req_id:
        if LEAF_STATS and _S._last_req_id is not None:
            _dump_stats(str(_S._last_req_id))
        _S._last_req_id = req_id
        _S._host_req_id = req_id
        _S._host_rounds = 0
        _S.prev_ns.fill_(7)
        _S.alt_ids.fill_(-1)
        if _S.alt_margins is not None:
            _S.alt_margins.fill_(1e30)
        if _S.tree_top is not None:
            _S.tree_top.fill_(-1)
        if _S.tree_vals is not None:
            _S.tree_vals.fill_(-1e30)
        if _S.cascade_meta is not None:
            _S.cascade_meta.zero_()
        _S.req_acc_state.zero_()
        _reset_tr_state()
    if eff_m > 8:
        assert (
            _S.tree_top is not None
            and _S.saved_leaf_toks is not None
            and _S.saved_leaf_k is not None
        )
        tr_log_arg = _S.tr_log if TR_ENABLED else None
        tr_state_arg = _S.tr_state if TR_ENABLED else None
        try:
            bfx.native_leaf_setup_tree(
                _S.leaf_flag,
                _S.tree_top,
                input_ids,
                positions,
                _S.saved_leaf_toks,
                _S.saved_leaf_k,
                eff_m,
                SELFTEST,
                _S.req_acc_state,
                _S.tree_vals,
                TREE_DYNAMIC,
                TREE_TEMP,
                TREE_TAIL,
                TREE_GAMMA,
                _S.saved_leaf_parent,
                TREE_MAX_DEPTH,
                TREE_SUB_COEF,
                TREE_SHF_COEF,
                TREE_CONT_DECAY,
                _S.hist,
                _S.hist_len,
                _S.saved_pos0,
                _S.stats,
                _S.saved_row_ctx,
                tr_log_arg,
                tr_state_arg,
                _S.vocab_size,
                _S.cascade_meta,
                _S.cascade_toks,
                _S.cascade_vals,
            )
        except TypeError:
            try:
                bfx.native_leaf_setup_tree(
                    _S.leaf_flag,
                    _S.tree_top,
                    input_ids,
                    positions,
                    _S.saved_leaf_toks,
                    _S.saved_leaf_k,
                    eff_m,
                    SELFTEST,
                    _S.req_acc_state,
                    _S.tree_vals,
                    TREE_DYNAMIC,
                    TREE_TEMP,
                    TREE_TAIL,
                    TREE_GAMMA,
                    _S.saved_leaf_parent,
                    TREE_MAX_DEPTH,
                    TREE_SUB_COEF,
                    TREE_SHF_COEF,
                    TREE_CONT_DECAY,
                    _S.hist,
                    _S.hist_len,
                    _S.saved_pos0,
                    _S.stats,
                )
            except TypeError:
                bfx.native_leaf_setup_tree(
                    _S.leaf_flag,
                    _S.tree_top,
                    input_ids,
                    positions,
                    _S.saved_leaf_toks,
                    _S.saved_leaf_k,
                    eff_m,
                    SELFTEST,
                    _S.req_acc_state,
                )
        _S._host_rounds += 1
        return True
    bfx.native_leaf_setup(
        _S.leaf_flag,
        _S.prev_ns,
        _S.alt_ids,
        input_ids,
        positions,
        _S.saved_alt_tok,
        LOW_THRESH,
        SELFTEST,
        _S.alt_margins,
        _S.req_acc_state,
        eff_m == 8,
        POLICY,
        MARGIN_ALPHA,
    )
    return True


@torch.no_grad()
def commit_and_fix(
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    leaf_u_tok: torch.Tensor,
    hidden_states: torch.Tensor | None,
    aux_hidden_states: list[torch.Tensor] | None,
    slot_mappings: torch.Tensor | None,
    extra_hidden: torch.Tensor | None = None,
    m8_mode: bool = False,
    verify_m: int | None = None,
    chain_all_valid: bool = False,
) -> None:
    """Commit leaf hit/clamp and fix hidden/aux states + all 52 target KV layers in 3 tiny launches."""
    if not _S.ready:
        return
    bfx = _get_bfx()
    if bfx is None:
        return
    assert (
        _S.leaf_flag is not None
        and _S.prev_ns is not None
        and _S.hit_flag is not None
        and _S.saved_alt_tok is not None
        and _S.kv_ptrs is not None
        and _S.stride_block is not None
        and _S.stride_head is not None
        and _S.stride_slot is not None
        and _S.block_size is not None
        and _S.layer_gid is not None
    )

    eff_m = verify_m if verify_m is not None else (8 if m8_mode else 7)
    is_tree = eff_m > 8
    src_row = eff_m - 1
    src_row_buf = None
    if is_tree:
        assert (
            _S.hit_src_row is not None
            and _S.saved_leaf_toks is not None
            and _S.saved_leaf_k is not None
        )
        tr_log_arg = _S.tr_log if TR_ENABLED else None
        tr_state_arg = _S.tr_state if TR_ENABLED else None
        in_sv_arg = _S.in_static_vocab if OOV_PREADMIT else None
        t_ids_arg = _S.tail_ids if OOV_PREADMIT else None
        t_cnt_arg = _S.tail_count if OOV_PREADMIT else None
        t_seen_arg = _S.tail_seen if OOV_PREADMIT else None
        try:
            bfx.native_leaf_commit_tree(
                _S.leaf_flag,
                _S.prev_ns,
                _S.hit_flag,
                _S.hit_src_row,
                _S.saved_leaf_toks,
                _S.saved_leaf_k,
                leaf_u_tok,
                sampled_tokens,
                num_sampled,
                num_rejected,
                eff_m,
                _S.stats,
                NOHIT,
                _S.req_acc_state,
                _S.saved_leaf_parent,
                _S.hist,
                _S.hist_len,
                _S.saved_pos0,
                _S.saved_row_ctx,
                tr_log_arg,
                tr_state_arg,
                _S.vocab_size,
                chain_all_valid,
                _S.cascade_meta,
                in_sv_arg,
                t_ids_arg,
                t_cnt_arg,
                t_seen_arg,
            )
        except TypeError:
            try:
                bfx.native_leaf_commit_tree(
                    _S.leaf_flag,
                    _S.prev_ns,
                    _S.hit_flag,
                    _S.hit_src_row,
                    _S.saved_leaf_toks,
                    _S.saved_leaf_k,
                    leaf_u_tok,
                    sampled_tokens,
                    num_sampled,
                    num_rejected,
                    eff_m,
                    _S.stats,
                    NOHIT,
                    _S.req_acc_state,
                    _S.saved_leaf_parent,
                    _S.hist,
                    _S.hist_len,
                    _S.saved_pos0,
                )
            except TypeError:
                bfx.native_leaf_commit_tree(
                    _S.leaf_flag,
                    _S.prev_ns,
                    _S.hit_flag,
                    _S.hit_src_row,
                    _S.saved_leaf_toks,
                    _S.saved_leaf_k,
                    leaf_u_tok,
                    sampled_tokens,
                    num_sampled,
                    num_rejected,
                    eff_m,
                    _S.stats,
                    NOHIT,
                    _S.req_acc_state,
                )
        src_row_buf = _S.hit_src_row
    else:
        bfx.native_leaf_commit(
            _S.leaf_flag,
            _S.prev_ns,
            _S.hit_flag,
            _S.saved_alt_tok,
            leaf_u_tok,
            sampled_tokens,
            num_sampled,
            num_rejected,
            _S.stats,
            NOHIT,
            _S.req_acc_state,
            eff_m == 8,
        )

    # Fix hidden_states[k] <- [src_row] and aux_hidden_states[i][k] <- [src_row] on leaf hit.
    min_rows = eff_m
    tensors: list[torch.Tensor] = []
    seen_ptrs: set[int] = set()
    for t in ([hidden_states, extra_hidden] + (list(aux_hidden_states) if aux_hidden_states else [])):
        if (
            t is not None
            and t.is_cuda
            and t.dim() == 2
            and t.shape[0] >= min_rows
            and t.stride(1) == 1
            and t.data_ptr() not in seen_ptrs
        ):
            seen_ptrs.add(t.data_ptr())
            tensors.append(t)

    if tensors:
        groups: dict[tuple[int, int], list[int]] = {}
        for t in tensors:
            eb = t.element_size()
            rsb = t.stride(0) * eb
            cb = t.shape[1] * eb
            if (rsb & 15) == 0 and (cb & 15) == 0 and (t.data_ptr() & 15) == 0:
                groups.setdefault((rsb, cb), []).append(t.data_ptr())
        for (rsb, cb), ptrs in groups.items():
            key = tuple(ptrs)
            ptr_t = _S._hid_ptr_cache.get(key)
            if ptr_t is None:
                if len(_S._hid_ptr_cache) >= 32:
                    _S._hid_ptr_cache.clear()
                with torch.inference_mode(False):
                    ptr_t = torch.tensor(ptrs, dtype=torch.int64, device=sampled_tokens.device)
                _S._hid_ptr_cache[key] = ptr_t
            if src_row_buf is not None:
                bfx.native_leaf_fix_hidden(_S.hit_flag, ptr_t, rsb, cb, src_row, src_row_buf)
            else:
                bfx.native_leaf_fix_hidden(_S.hit_flag, ptr_t, rsb, cb, src_row)

    # Fix KV cache slot(p+src_row) -> slot(p+k) across all 52 target layers (1 launch, ~1.5-4 us).
    if slot_mappings is not None and _S.num_layers > 0:
        if src_row_buf is not None:
            bfx.native_leaf_fix_kv(
                _S.hit_flag,
                slot_mappings,
                _S.kv_ptrs,
                _S.stride_block,
                _S.stride_head,
                _S.stride_slot,
                _S.block_size,
                _S.layer_gid,
                src_row,
                src_row_buf,
            )
        else:
            bfx.native_leaf_fix_kv(
                _S.hit_flag,
                slot_mappings,
                _S.kv_ptrs,
                _S.stride_block,
                _S.stride_head,
                _S.stride_slot,
                _S.block_size,
                _S.layer_gid,
                src_row,
            )

    if not _S._logged_active:
        _S._logged_active = True
        logger.info(
            "native_leaf: active on verify path (verify_m=%d, policy=%s, 52-layer KV-remap + 3-kernel commit)",
            eff_m, _POLICY_RAW,
        )
