"""16-CTA GQA-16 Tensor-Core M=1 decode attention for MuseGlimmer (32Q / 2KV / D=128).

Controls (no VLLM_* env vars required):
  MUSE_GQA16_DECODE=1          opt-in (off by default: not bit-identical to FlashAttention-2 in every case).
  MUSE_GQA16_MODE=fa2_8sp      default: 16 CTAs (kBlockN=128, 8 splits on SW, 32 splits on Full; 8.95-11.15 us).
  MUSE_GQA16_MODE=fa2_exact    0-ULP FA2-matched kBlockN=128 mode (2 CTAs on SW, 16 CTAs on Full).
  MUSE_GQA16_MODE=split8       16 CTAs (8 splits x 2 KV heads, B_N=32) on all 52 layers.
  MUSE_GQA16_SPLITS=8          KV split worker count per KV head (default 8 -> 16 CTAs).
"""
from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

import torch

try:
    from vllm.logger import init_logger
    logger = init_logger("vllm.muse_gqa16_attn")
except Exception:  # noqa: BLE001
    logger = logging.getLogger("vllm.muse_gqa16_attn")

MAX_SPLITS = 64
ENABLED = os.environ.get("MUSE_GQA16_DECODE", "0").strip() not in ("0", "", "false", "no")
MODE_NAME = os.environ.get("MUSE_GQA16_MODE", "fa2_exact").strip().lower()
NUM_SPLITS = max(1, min(MAX_SPLITS, int(os.environ.get("MUSE_GQA16_SPLITS", "8"))))
FA2_FULL_SPLITS = int(os.environ.get("MUSE_GQA16_FA2_FULL_SPLITS", "32"))

_lib = None
try:
    _lib = ctypes.CDLL(str(Path(__file__).with_name("libgqa16_decode.so")))
    _lib.muse_gqa16_decode.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p,
    ]
    _lib.muse_gqa16_decode.restype = ctypes.c_int
except OSError:
    _lib = None

_scratch: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_state: dict[int, tuple[bool, int]] = {}  # dev_idx -> (admitted, fa2_exact_submode)


def _get_scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    bufs = _scratch.get(idx)
    if bufs is not None:
        return bufs
    if torch.cuda.is_current_stream_capturing():
        return None
    o_part = torch.zeros((2, MAX_SPLITS, 16, 128), dtype=torch.float32, device=device)
    lse_part = torch.full((2, MAX_SPLITS, 16), float("-inf"), dtype=torch.float32, device=device)
    cnt = torch.zeros((2,), dtype=torch.int32, device=device)
    bufs = (o_part, lse_part, cnt)
    _scratch[idx] = bufs
    return bufs


def _mode_params(w_left: int, exact_submode: int) -> tuple[int, int, int]:
    if MODE_NAME == "fa2_exact":
        g_splits = 1 if w_left >= 0 else NUM_SPLITS
        f_splits = 1 if w_left >= 0 else FA2_FULL_SPLITS
        return g_splits, f_splits, exact_submode
    if MODE_NAME == "split8":
        return NUM_SPLITS, NUM_SPLITS, 0
    # Default "fa2_8sp": kBlockN=128 online softmax across 8 worker splits (16 CTAs) on SW
    # and FA2_FULL_SPLITS across 8 worker splits (16 CTAs, 0-ULP vs FA2) on Full-Attention.
    f_splits = NUM_SPLITS if w_left >= 0 else FA2_FULL_SPLITS
    return NUM_SPLITS, f_splits, exact_submode


def self_check(device: torch.device) -> bool:
    if not ENABLED or _lib is None or device.type != "cuda":
        return False
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _state.get(idx)
    if cached is not None:
        return cached[0]
    if torch.cuda.is_current_stream_capturing():
        return False

    ok = False
    exact_submode = 1
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

        bufs = _get_scratch(device)
        assert bufs is not None
        o_part, lse_part, cnt = bufs
        gen = torch.Generator(device=device).manual_seed(20261123)
        scale = float(128.0 ** -0.5)
        num_blocks = 192
        block_size = 16
        raw_kv = (torch.randn((num_blocks, block_size, 2, 256), generator=gen, device=device) * 0.8).to(torch.bfloat16)
        raw_kv[0].zero_()
        kv_cache = raw_kv.permute(0, 2, 1, 3)
        k_cache, v_cache = kv_cache.transpose(1, 2).split(128, dim=-1)

        perm = torch.randperm(num_blocks - 1, generator=gen, device=device, dtype=torch.int32) + 1
        bt = perm.unsqueeze(0).contiguous()
        cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
        stream = torch.cuda.current_stream(device).cuda_stream

        for sub in (1, 2):
            sub_mismatches = 0
            for seq_k in (1, 16, 64, 129, 577, 2048):
                for win in ((-1, -1), (2047, 0)):
                    q = (torch.randn((1, 32, 128), generator=gen, device=device) * 1.5).to(torch.bfloat16).contiguous()
                    sk = torch.tensor([seq_k], dtype=torch.int32, device=device)
                    ref = torch.empty_like(q)
                    got = torch.empty_like(q)
                    flash_attn_varlen_func(
                        q=q, k=k_cache, v=v_cache, out=ref,
                        cu_seqlens_q=cu_q, max_seqlen_q=1,
                        seqused_k=sk, max_seqlen_k=seq_k,
                        softmax_scale=scale, causal=True,
                        window_size=list(win), block_table=bt,
                        fa_version=2, num_splits=1,
                    )
                    st = _lib.muse_gqa16_decode(
                        q.data_ptr(), kv_cache.data_ptr(), got.data_ptr(),
                        o_part.data_ptr(), lse_part.data_ptr(), cnt.data_ptr(),
                        sk.data_ptr(), bt.data_ptr(),
                        block_size, win[0], 1, 1, sub, scale,
                        q.stride(1), got.stride(1),
                        kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1),
                        stream,
                    )
                    torch.cuda.synchronize(device)
                    if st != 0 or not torch.isfinite(got).all():
                        sub_mismatches += 100000
                        break
                    sub_mismatches += int((ref.view(torch.int16) != got.view(torch.int16)).sum().item())
            if sub_mismatches == 0:
                exact_submode = sub
                break

        ok = True
        for seq_k in (1, 16, 17, 32, 33, 64, 128, 513, 576, 640, 2048, 2500):
            for win in ((-1, -1), (2047, 0)):
                bt_cur = bt.clone()
                if win[0] >= 0 and seq_k > 2048:
                    expired_pages = (seq_k - 2048) // block_size
                    if expired_pages > 0:
                        bt_cur[0, :expired_pages] = 0
                q = (torch.randn((1, 32, 128), generator=gen, device=device) * 1.5).to(torch.bfloat16).contiguous()
                sk = torch.tensor([seq_k], dtype=torch.int32, device=device)
                ref = torch.empty_like(q)
                got = torch.empty_like(q)
                flash_attn_varlen_func(
                    q=q, k=k_cache, v=v_cache, out=ref,
                    cu_seqlens_q=cu_q, max_seqlen_q=1,
                    seqused_k=sk, max_seqlen_k=8192,
                    softmax_scale=scale, causal=True,
                    window_size=list(win), block_table=bt_cur,
                    fa_version=2, num_splits=0,
                )
                g_splits, f_splits, k_mode = _mode_params(win[0], exact_submode)
                st = _lib.muse_gqa16_decode(
                    q.data_ptr(), kv_cache.data_ptr(), got.data_ptr(),
                    o_part.data_ptr(), lse_part.data_ptr(), cnt.data_ptr(),
                    sk.data_ptr(), bt_cur.data_ptr(),
                    block_size, win[0], g_splits, f_splits, k_mode, scale,
                    q.stride(1), got.stride(1),
                    kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1),
                    stream,
                )
                torch.cuda.synchronize(device)
                if st != 0 or not torch.isfinite(got).all() or int(cnt.abs().max().item()) != 0:
                    ok = False
                    break
                max_err = float((ref.float() - got.float()).abs().max().item())
                mismatch = int((ref.view(torch.int16) != got.view(torch.int16)).sum().item())
                if MODE_NAME == "fa2_exact":
                    if mismatch > 0:
                        ok = False
                        break
                elif MODE_NAME == "fa2_8sp":
                    # Full-attention (-1) must be near 0-ULP; SW (2047) 8-split LSE combine <= 1 ULP.
                    if (win[0] < 0 and mismatch > 4) or max_err > 4e-3:
                        ok = False
                        break
                else:
                    if max_err > 1e-2:
                        ok = False
                        break
            if not ok:
                break
    except Exception:  # noqa: BLE001
        logger.warning("muse_gqa16_attn: self-check raised on cuda:%d", idx, exc_info=True)
        ok = False

    _state[idx] = (ok, exact_submode)
    if ok:
        logger.info(
            "muse_gqa16_attn: admitted on cuda:%d (mode=%s, splits=%d -> %d CTAs, exact_submode=%d)",
            idx, MODE_NAME, NUM_SPLITS, NUM_SPLITS * 2, exact_submode,
        )
    else:
        logger.warning("muse_gqa16_attn: self-check failed on cuda:%d; keeping stock FlashAttention-2", idx)
    return ok


def try_decode_gqa16(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    seqused_k: torch.Tensor | None,
    block_table: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
    window: tuple[int, int] | None,
) -> bool:
    if not ENABLED or _lib is None:
        return False
    device = query.device
    idx = device.index if device.index is not None else torch.cuda.current_device()
    st = _state.get(idx)
    if st is None:
        if not self_check(device):
            return False
        st = _state[idx]
    elif not st[0]:
        return False

    if (
        seqused_k is None
        or block_table is None
        or query.dtype != torch.bfloat16
        or kv_cache.dtype != torch.bfloat16
        or output.dtype != torch.bfloat16
        or query.ndim != 3
        or output.ndim != 3
        or kv_cache.ndim != 4
        or kv_cache.shape[1] != 2
        or kv_cache.shape[3] != 256
        or kv_cache.stride(3) != 1
        or query.stride(2) != 1
        or output.stride(2) != 1
        or cu_seqlens_q.shape[0] != 2
        or seqused_k.dtype != torch.int32
        or block_table.dtype != torch.int32
        or block_table.ndim != 2
        or block_table.stride(1) != 1
        or (query.data_ptr() | kv_cache.data_ptr() | output.data_ptr()) & 15
    ):
        return False

    block_size = int(kv_cache.shape[2])
    if block_size < 16 or (block_size & 15) != 0:
        return False

    w_left = -1
    if window is not None:
        if window[1] > 0:
            return False
        w_left = int(window[0])

    bufs = _get_scratch(device)
    if bufs is None:
        return False
    o_part, lse_part, cnt = bufs

    g_splits, f_splits, k_mode = _mode_params(w_left, st[1])
    stream = torch.cuda.current_stream(device).cuda_stream
    rc = _lib.muse_gqa16_decode(
        query.data_ptr(),
        kv_cache.data_ptr(),
        output.data_ptr(),
        o_part.data_ptr(),
        lse_part.data_ptr(),
        cnt.data_ptr(),
        seqused_k.data_ptr(),
        block_table.data_ptr(),
        block_size,
        w_left,
        g_splits,
        f_splits,
        k_mode,
        softmax_scale,
        query.stride(1),
        output.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(1),
        stream,
    )
    return rc == 0
