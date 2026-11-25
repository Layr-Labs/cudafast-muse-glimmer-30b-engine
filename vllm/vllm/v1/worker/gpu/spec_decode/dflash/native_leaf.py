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

import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("MUSE_NATIVE_LEAF", "1").strip() not in ("0", "", "false", "no")
M8_ENABLED = os.environ.get("MUSE_NATIVE_LEAF_M8", "1").strip() not in ("0", "", "false", "no")
try:
    _TREE_M_RAW = int(os.environ.get("MUSE_NATIVE_TREE_M", "12").strip() or "0")
except ValueError:
    _TREE_M_RAW = 0
TREE_M = _TREE_M_RAW if _TREE_M_RAW in (12, 16) else 0
LOW_THRESH = int(os.environ.get("MUSE_NATIVE_LEAF_THRESH", "3"))
_POLICY_RAW = os.environ.get("MUSE_NATIVE_LEAF_POLICY", "d1").strip().lower()
POLICY = 1 if _POLICY_RAW == "margin" else 0
MARGIN_ALPHA = float(os.environ.get("MUSE_NATIVE_LEAF_MARGIN_ALPHA", "1.0"))
_ST_RAW = os.environ.get("MUSE_NATIVE_LEAF_SELFTEST", "0").strip().lower()
SELFTEST = 1 if _ST_RAW in ("1", "true", "yes") else (2 if _ST_RAW == "2" else 0)
NOHIT = os.environ.get("MUSE_NATIVE_LEAF_NOHIT", "0").strip() in ("1", "true", "yes")


class _NativeLeafState:
    def __init__(self) -> None:
        self.ready: bool = False
        self.m8_ready: bool = False
        self.tree_m_ready: int = 0
        self.device: torch.device | None = None
        self.leaf_flag: torch.Tensor | None = None       # int32[49] shared with muse_gqa16_attn
        self.prev_ns: torch.Tensor | None = None         # int32[1]
        self.hit_flag: torch.Tensor | None = None        # int32[1] (0=miss, k=hit depth 1..5)
        self.hit_src_row: torch.Tensor | None = None     # int32[1] (0=miss, r=winning leaf row 7..15)
        self.alt_ids: torch.Tensor | None = None         # int64[4] (d1_b1, d1_b2, d2_b1, d2_b2)
        self.alt_margins: torch.Tensor | None = None     # float32[2] (margin_d1, margin_d2)
        self.tree_top: torch.Tensor | None = None        # int64[20] ([5, 4] top-4 for d1..d5)
        self.req_acc_state: torch.Tensor | None = None   # int32[3] [rounds, committed, high_acc_flag]
        self.saved_alt_tok: torch.Tensor | None = None   # int64[1] (legacy M=7/M=8)
        self.saved_leaf_toks: torch.Tensor | None = None # int64[16] (tree mode)
        self.saved_leaf_k: torch.Tensor | None = None    # int32[16] (tree mode)
        self.stats: torch.Tensor | None = None           # int64[18] [steps, mode, hits, clamps, k1..5, li0..8]
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


_S = _NativeLeafState()


def is_ready() -> bool:
    return ENABLED and _S.ready


def is_m8_ready() -> bool:
    return ENABLED and M8_ENABLED and _S.ready and _S.m8_ready


def active_verify_m() -> int:
    """Return active verify width for bs=1 depth-6 verify: 16, 12, 8, or 7."""
    if not (ENABLED and _S.ready):
        return 7
    if _S.tree_m_ready in (12, 16):
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
    """Return fixed-address int64[20] ([5, 4]) buffer for drafter rows 0..4 top-4/top-2 (CUDA-graph safe)."""
    if not ENABLED or TREE_M == 0 or device.type != "cuda":
        return None
    if (
        _S.tree_top is not None
        and _S.tree_top.device == device
        and _S.tree_top.numel() >= 20
        and not _S.tree_top.is_inference()
    ):
        return _S.tree_top
    if torch.cuda.is_current_stream_capturing():
        return _S.tree_top if (_S.tree_top is not None and _S.tree_top.device == device) else None
    with torch.inference_mode(False):
        _S.tree_top = torch.full((20,), -1, dtype=torch.int64, device=device)
    return _S.tree_top


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
        _S.hit_flag = torch.zeros((1,), dtype=torch.int32, device=device)
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
            or _S.tree_top.numel() < 20
            or _S.tree_top.is_inference()
        ):
            _S.tree_top = torch.full((20,), -1, dtype=torch.int64, device=device)
        else:
            _S.tree_top.fill_(-1)
        _S.req_acc_state = torch.zeros((3,), dtype=torch.int32, device=device)
        _S.saved_alt_tok = torch.full((1,), -1, dtype=torch.int64, device=device)
        _S.saved_leaf_toks = torch.full((16,), -1, dtype=torch.int64, device=device)
        _S.saved_leaf_k = torch.zeros((16,), dtype=torch.int32, device=device)
        _S.stats = torch.zeros((18,), dtype=torch.int64, device=device)
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
        _S.tree_m_ready = (
            TREE_M
            if (
                TREE_M in (12, 16)
                and lf.numel() >= 49
                and hasattr(bfx, "native_leaf_prepare_tree")
                and hasattr(bfx, "native_leaf_setup_tree")
                and hasattr(bfx, "native_leaf_commit_tree")
            )
            else 0
        )
        _S.ready = True
    logger.info(
        "native_leaf: armed (%d target GQA-16 layers, tree_m=%d, m8=%d, policy=%s, alpha=%.2f, thresh<=%d, selftest=%d, nohit=%d)",
        _S.num_layers, _S.tree_m_ready, int(_S.m8_ready), _POLICY_RAW, MARGIN_ALPHA, LOW_THRESH, int(SELFTEST), int(NOHIT),
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
        _S._last_req_id = req_id
        _S.prev_ns.fill_(7)
        _S.alt_ids.fill_(-1)
        if _S.alt_margins is not None:
            _S.alt_margins.fill_(1e30)
        if _S.tree_top is not None:
            _S.tree_top.fill_(-1)
        _S.req_acc_state.zero_()
    elif 0 < num_draft_tokens < 6:
        # Ceiling shortened depth on high-acceptance request -> keep leaf mode off.
        _S.prev_ns.fill_(7)
        _S.req_acc_state[2:3].fill_(1)


@torch.no_grad()
def prepare_m8(positions: torch.Tensor, seq_lens: torch.Tensor) -> bool:
    """Zero-sync GPU prep: positions[7] = positions[6] + 1, seq_lens[0] += 1 for M=8 slot mapping."""
    return prepare_tree(positions, seq_lens, 8)


@torch.no_grad()
def prepare_tree(positions: torch.Tensor, seq_lens: torch.Tensor, verify_m: int) -> bool:
    """Zero-sync GPU prep: positions[7..verify_m-1] = p+7..p+verify_m-1, seq_lens[0] += verify_m-7."""
    if not is_ready() or verify_m < 8 or verify_m > 16:
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
        _S._last_req_id = req_id
        _S.prev_ns.fill_(7)
        _S.alt_ids.fill_(-1)
        if _S.alt_margins is not None:
            _S.alt_margins.fill_(1e30)
        if _S.tree_top is not None:
            _S.tree_top.fill_(-1)
        _S.req_acc_state.zero_()
    if eff_m > 8:
        assert (
            _S.tree_top is not None
            and _S.saved_leaf_toks is not None
            and _S.saved_leaf_k is not None
        )
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
