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

DEFAULT_TREE_M = "24"
ENABLED = os.environ.get("MUSE_NATIVE_LEAF", "1").strip() not in ("0", "", "false", "no")
M8_ENABLED = os.environ.get("MUSE_NATIVE_LEAF_M8", "1").strip() not in ("0", "", "false", "no")
try:
    _TREE_M_RAW = int(os.environ.get("MUSE_NATIVE_TREE_M", DEFAULT_TREE_M).strip() or "0")
except ValueError:
    _TREE_M_RAW = 0
TREE_M = _TREE_M_RAW if _TREE_M_RAW in (12, 16, 24) else 0


def desired_verify_m() -> int:
    """Single source of truth for configured bs=1 depth-6 verify width (16, 12, 8, or 7)."""
    if not ENABLED:
        return 7
    if TREE_M in (12, 16, 24):
        return TREE_M
    if M8_ENABLED:
        return 8
    return 7


# Adaptive width: with the M=24 tree armed, high-acceptance requests (the
# ceiling limiter's latched flag, known host-side on the scheduler) run the
# record M=16 tree instead; the allocator disables every leaf for them, so rows
# 7..23 would be pure dummies.  Both widths are exact verifications.
ADAPTIVE_M24 = os.environ.get("MUSE_M24_ADAPTIVE", "1").strip() not in ("0", "", "false", "no")


def capture_verify_ms() -> list[int]:
    """Widths whose bs=1 FULL verify graphs must be captured (widest first)."""
    m = desired_verify_m()
    if m <= 7:
        return []
    if m == 24:
        # Never capture (or later dispatch) a 24-row graph unless the M=24 kernels were actually admitted
        # (init_kv_table runs before graph capture); otherwise the capture would run un-admitted generic paths.
        if _S.ready and _S.tree_m_ready == 24:
            return [24, 16] if ADAPTIVE_M24 else [24]
        return [16]
    return [m]


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
_ST_RAW = os.environ.get("MUSE_NATIVE_LEAF_SELFTEST", "0").strip().lower()
SELFTEST = 1 if _ST_RAW in ("1", "true", "yes") else (2 if _ST_RAW == "2" else 0)
NOHIT = os.environ.get("MUSE_NATIVE_LEAF_NOHIT", "0").strip() in ("1", "true", "yes")


class _NativeLeafState:
    def __init__(self) -> None:
        self.ready: bool = False
        self.m8_ready: bool = False
        self.tree_m_ready: int = 0
        self.device: torch.device | None = None
        self.leaf_flag: torch.Tensor | None = None          # int32[73] shared with muse_gqa16_attn
        self.prev_ns: torch.Tensor | None = None            # int32[1]
        self.hit_flag: torch.Tensor | None = None           # int32[14] [dst0, hit_len, dst0, src0, ..., dst5, src5]
        self.hit_src_row: torch.Tensor | None = None        # int32[1] (0=miss, r=winning initial leaf row 7..15)
        self.alt_ids: torch.Tensor | None = None            # int64[4] (d1_b1, d1_b2, d2_b1, d2_b2)
        self.alt_margins: torch.Tensor | None = None        # float32[2] (margin_d1, margin_d2)
        self.tree_top: torch.Tensor | None = None           # int64[36] ([6, 6] top-6 for d1..d6)
        self.tree_vals: torch.Tensor | None = None          # float32[36] ([6, 6] top-6 logits for d1..d6)
        self.req_acc_state: torch.Tensor | None = None      # int32[3] [rounds, committed, high_acc_flag]
        self.saved_alt_tok: torch.Tensor | None = None      # int64[1] (legacy M=7/M=8)
        self.saved_leaf_toks: torch.Tensor | None = None    # int64[24] (tree mode)
        self.saved_leaf_k: torch.Tensor | None = None       # int32[24] (tree mode)
        self.saved_leaf_parent: torch.Tensor | None = None  # int32[24] (-1=1-level root, 7..23=parent row)
        self.stats: torch.Tensor | None = None              # int64[32] telemetry counters
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
        self._logged_active: bool = False
        self._atexit_registered: bool = False


_S = _NativeLeafState()


def _dump_stats(tag: str) -> None:
    if not LEAF_STATS or not _S.ready or _S.stats is None:
        return
    try:
        st = _S.stats.detach().cpu().tolist()
        if len(st) >= 32 and st[0] > 0:
            logger.info(
                "native_leaf_stats[%s]: steps=%d active=%d hits=%d deep_hits=%d extra_toks=%d deep_bonus=%d "
                "total_ns=%d max_len=%d len1..6=%s k1..6=%s slots0..8=%s reached_d2/3/4+=%d/%d/%d",
                tag, st[0], st[1], st[2], st[3], st[25], st[26], st[27], st[28],
                st[19:25], st[4:10], st[10:19], st[29], st[30], st[31],
            )
            _S.stats.zero_()
    except Exception:  # noqa: BLE001
        pass


def is_ready() -> bool:
    return ENABLED and _S.ready


def is_m8_ready() -> bool:
    return ENABLED and M8_ENABLED and _S.ready and _S.m8_ready


def active_verify_m(high_acceptance: bool = False) -> int:
    """Return active verify width for bs=1 depth-6 verify: 24, 16, 12, 8, or 7.

    ``high_acceptance`` is the scheduler-side latched host flag; it demotes an
    armed M=24 tree to the exact record M=16 tree.
    """
    if not (ENABLED and _S.ready):
        return 7
    if _S.tree_m_ready == 24 and high_acceptance and ADAPTIVE_M24:
        return 16
    if _S.tree_m_ready in (12, 16, 24):
        return _S.tree_m_ready
    if M8_ENABLED and _S.m8_ready:
        return 8
    return 7


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


def _all_m24_kernels_admitted(device: torch.device, g16: Any) -> bool:
    """Admission is fail-closed: a missing/unchecked M=24 participant keeps record M=16.

    Evaluated in init_kv_table, i.e. after load (decode6 / rANS / BF16 admissions, early lm_head+lmpost admission in
    model_runner.initialize_kv_cache) and before CUDA graph capture."""
    verdicts: dict[str, bool] = {}
    verdicts["gqa16_attention"] = bool(getattr(g16, "is_m24_verify_admitted", lambda _d: False)(device))
    for module_name, flag_name in (
        ("muse_marlin6_gemv", "decode6_m24_admitted"),
        ("muse_rans_ext", "rans_m24_admitted"),
    ):
        try:
            module = __import__(module_name)
            verdicts[module_name] = bool(getattr(module, flag_name, lambda _d: False)(device))
        except Exception:  # noqa: BLE001
            verdicts[module_name] = False
    try:
        from vllm.model_executor.kernels.linear.scaled_mm import muse_fp8 as _mf8

        verdicts["fp8_non_rans_layers"] = bool(_mf8.m24_fp8_ok())
    except Exception:  # noqa: BLE001
        verdicts["fp8_non_rans_layers"] = False
    try:
        from vllm.model_executor.models import muse_glimmer as _mg

        verdicts["bf16_stacked_attention"] = bool(getattr(_mg, "bf16_rans_m24_admitted", lambda: False)())
    except Exception:  # noqa: BLE001
        verdicts["bf16_stacked_attention"] = False
    ok = all(verdicts.values())
    logger.info(
        "native_leaf: M=24 arming verdicts %s -> %s",
        ", ".join(f"{k}={'ok' if v else 'NO'}" for k, v in verdicts.items()),
        "ARMED M=24 tree" if ok else "keeping record M=16 tree",
    )
    return ok


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
    """Return fixed-address int64[36] ([6, 6]) buffer for drafter rows 0..5 top-6 IDs (CUDA-graph safe)."""
    if not ENABLED or TREE_M == 0 or device.type != "cuda":
        return None
    if (
        _S.tree_top is not None
        and _S.tree_top.device == device
        and _S.tree_top.numel() >= 36
        and not _S.tree_top.is_inference()
    ):
        return _S.tree_top
    if torch.cuda.is_current_stream_capturing():
        return _S.tree_top if (_S.tree_top is not None and _S.tree_top.device == device) else None
    with torch.inference_mode(False):
        _S.tree_top = torch.full((36,), -1, dtype=torch.int64, device=device)
    return _S.tree_top


def get_tree_vals_buf(device: torch.device) -> torch.Tensor | None:
    """Return fixed-address float32[36] ([6, 6]) buffer for drafter rows 0..5 top-6 logits (CUDA-graph safe)."""
    if not ENABLED or TREE_M == 0 or device.type != "cuda":
        return None
    if (
        _S.tree_vals is not None
        and _S.tree_vals.device == device
        and _S.tree_vals.numel() >= 36
        and not _S.tree_vals.is_inference()
    ):
        return _S.tree_vals
    if torch.cuda.is_current_stream_capturing():
        return _S.tree_vals if (_S.tree_vals is not None and _S.tree_vals.device == device) else None
    with torch.inference_mode(False):
        _S.tree_vals = torch.full((36,), -1e30, dtype=torch.float32, device=device)
    return _S.tree_vals


def init_kv_table(
    kv_caches_dict: dict[str, torch.Tensor],
    kv_cache_config: Any,
    draft_attn_layer_names: set[str] | list[str] | None,
    device: torch.device,
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
            or _S.tree_top.numel() < 36
            or _S.tree_top.is_inference()
        ):
            _S.tree_top = torch.full((36,), -1, dtype=torch.int64, device=device)
        else:
            _S.tree_top.fill_(-1)
        if (
            _S.tree_vals is None
            or _S.tree_vals.device != device
            or _S.tree_vals.numel() < 36
            or _S.tree_vals.is_inference()
        ):
            _S.tree_vals = torch.full((36,), -1e30, dtype=torch.float32, device=device)
        else:
            _S.tree_vals.fill_(-1e30)
        _S.req_acc_state = torch.zeros((3,), dtype=torch.int32, device=device)
        _S.saved_alt_tok = torch.full((1,), -1, dtype=torch.int64, device=device)
        _S.saved_leaf_toks = torch.full((24,), -1, dtype=torch.int64, device=device)
        _S.saved_leaf_k = torch.zeros((24,), dtype=torch.int32, device=device)
        _S.saved_leaf_parent = torch.full((24,), -1, dtype=torch.int32, device=device)
        _S.stats = torch.zeros((32,), dtype=torch.int64, device=device)
        _S.num_layers = len(kv_ptrs)
        _S.kv_ptrs = torch.tensor(kv_ptrs, dtype=torch.int64, device=device)
        _S.stride_block = torch.tensor(s_blk, dtype=torch.int64, device=device)
        _S.stride_head = torch.tensor(s_head, dtype=torch.int64, device=device)
        _S.stride_slot = torch.tensor(s_slot, dtype=torch.int64, device=device)
        _S.block_size = torch.tensor(b_sz, dtype=torch.int32, device=device)
        _S.layer_gid = torch.tensor(gids, dtype=torch.int32, device=device)
        _S._hid_ptr_cache.clear()
        _S._last_req_id = None
        _S.m8_ready = bool(M8_ENABLED and hasattr(bfx, "native_leaf_prepare_m8"))
        m24_ok = TREE_M == 24 and _all_m24_kernels_admitted(device, g16)
        requested_tree_m = 24 if m24_ok else (16 if TREE_M == 24 else TREE_M)
        _S.tree_m_ready = (
            requested_tree_m
            if (
                requested_tree_m in (12, 16, 24)
                and lf.numel() >= 1 + 3 * requested_tree_m
                and hasattr(bfx, "native_leaf_prepare_tree")
                and hasattr(bfx, "native_leaf_setup_tree")
                and hasattr(bfx, "native_leaf_commit_tree")
            )
            else 0
        )
        _S.ready = True
        if LEAF_STATS and not _S._atexit_registered:
            _S._atexit_registered = True
            atexit.register(_dump_stats, "atexit")
    logger.info(
        "native_leaf: armed (%d target GQA-16 layers, tree_m=%d, dynamic=%d [calib=%d,depth<=%d,T=%.2f,tail=%.1f,g=%.2f,adapt=%d,sub=%.2f,shf=%.2f,cont=%.2f,hi_cpc=%.1f], m8=%d, policy=%s, alpha=%.2f, thresh<=%d, selftest=%d, nohit=%d, stats=%d)",
        _S.num_layers, _S.tree_m_ready, int(TREE_DYNAMIC), int(TREE_CALIB), TREE_MAX_DEPTH, TREE_TEMP, TREE_TAIL, TREE_GAMMA,
        int(TREE_ADAPT_GAMMA), TREE_SUB_COEF, TREE_SHF_COEF, TREE_CONT_DECAY, TREE_HIGH_ACC_CPC,
        int(_S.m8_ready), _POLICY_RAW, MARGIN_ALPHA, LOW_THRESH, int(SELFTEST), int(NOHIT), int(LEAF_STATS),
    )
    return True


@torch.no_grad()
def on_non_leaf_step(req_id: str | None, num_draft_tokens: int) -> None:
    """Clear leaf flag on any non-leaf step and reset state on request switch / prefill / ceiling clamp."""
    if not _S.ready:
        return
    assert (
        _S.leaf_flag is not None
        and _S.prev_ns is not None
        and _S.alt_ids is not None
        and _S.req_acc_state is not None
    )
    _S.leaf_flag[:1].zero_()
    if req_id is None or req_id != _S._last_req_id or num_draft_tokens == 0:
        if LEAF_STATS and _S._last_req_id is not None and (req_id != _S._last_req_id or num_draft_tokens == 0):
            _dump_stats(str(_S._last_req_id))
        _S._last_req_id = req_id
        _S.prev_ns.fill_(7)
        _S.alt_ids.fill_(-1)
        if _S.alt_margins is not None:
            _S.alt_margins.fill_(1e30)
        if _S.tree_top is not None:
            _S.tree_top.fill_(-1)
        if _S.tree_vals is not None:
            _S.tree_vals.fill_(-1e30)
        _S.req_acc_state.zero_()
    elif 0 < num_draft_tokens < 6:
        # Ceiling shortened depth on high-acceptance request -> hard-latch leaf mode off.
        _S.prev_ns.fill_(7)
        _S.req_acc_state[2:3].fill_(2)


@torch.no_grad()
def prepare_m8(positions: torch.Tensor, seq_lens: torch.Tensor) -> bool:
    """Zero-sync GPU prep: positions[7] = positions[6] + 1, seq_lens[0] += 1 for M=8 slot mapping."""
    return prepare_tree(positions, seq_lens, 8)


@torch.no_grad()
def prepare_tree(positions: torch.Tensor, seq_lens: torch.Tensor, verify_m: int) -> bool:
    """Zero-sync GPU prep: positions[7..verify_m-1] = p+7..p+verify_m-1, seq_lens[0] += verify_m-7."""
    if not is_ready() or verify_m < 8 or verify_m > 24:
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
    if req_id != _S._last_req_id:
        if LEAF_STATS and _S._last_req_id is not None:
            _dump_stats(str(_S._last_req_id))
        _S._last_req_id = req_id
        _S.prev_ns.fill_(7)
        _S.alt_ids.fill_(-1)
        if _S.alt_margins is not None:
            _S.alt_margins.fill_(1e30)
        if _S.tree_top is not None:
            _S.tree_top.fill_(-1)
        if _S.tree_vals is not None:
            _S.tree_vals.fill_(-1e30)
        _S.req_acc_state.zero_()
    if eff_m > 8:
        assert (
            _S.tree_top is not None
            and _S.saved_leaf_toks is not None
            and _S.saved_leaf_k is not None
        )
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
