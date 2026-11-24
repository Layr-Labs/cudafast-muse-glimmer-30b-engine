"""Strict 0-ULP FA2-exact 16-CTA GQA-16 Tensor-Core M=1 decode attention for MuseGlimmer (32Q / 2KV / D=128).

Controls (no VLLM_* env vars required):
  MUSE_GQA16_DECODE=1          enable 0-ULP GQA-16 M=1 decode kernel (admitted only after strict 0-ULP self-check).
  MUSE_GQA16_SPLITS=8          worker CTA count per KV head (default 8 -> 16 CTAs).
"""
from __future__ import annotations

import ctypes
import logging
import math
import os
from pathlib import Path

import torch

try:
    from vllm.logger import init_logger
    logger = init_logger("vllm.muse_gqa16_attn")
except Exception:  # noqa: BLE001
    logger = logging.getLogger("vllm.muse_gqa16_attn")

MAX_SPLITS = 64
MAX_NBLOCKS64 = 256
ENABLED = os.environ.get("MUSE_GQA16_DECODE", "1").strip() not in ("0", "", "false", "no")
NUM_SPLITS = max(1, min(MAX_SPLITS, int(os.environ.get("MUSE_GQA16_SPLITS", "8"))))
ONE_PASS = os.environ.get("MUSE_GQA16_1PASS", "1").strip() not in ("0", "", "false", "no")
ONE_PASS_MAX_TOKENS = int(os.environ.get("MUSE_GQA16_1PASS_MAX_TOKENS", "2176"))

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
_state: dict[int, tuple[bool, int, int]] = {}  # dev_idx -> (admitted, scale_mode, num_sms)


def _get_scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    bufs = _scratch.get(idx)
    if bufs is not None:
        return bufs
    if torch.cuda.is_current_stream_capturing():
        return None
    # Sized to hold both 2-pass kBlockN=64 scratch (2 x 256 x 1024 f32 = 2 MiB)
    # and multi-split kBlockN=128 O/LSE accumulators (2 x 64 x 16 x 128 f32 = 1 MiB).
    o_part = torch.zeros((2, MAX_NBLOCKS64, 1024), dtype=torch.float32, device=device)
    lse_part = torch.full((2, MAX_NBLOCKS64, 16), float("-inf"), dtype=torch.float32, device=device)
    cnt = torch.zeros((2,), dtype=torch.int32, device=device)
    bufs = (o_part, lse_part, cnt)
    _scratch[idx] = bufs
    return bufs


def fa2_num_splits_heuristic(max_seqlen_k: int, num_sms: int, batch_nheads_mblocks: int = 2) -> int:
    """Exact replica of FlashAttention-2's host num_splits_heuristic for kBlockN=128."""
    if batch_nheads_mblocks >= 0.8 * num_sms:
        return 1
    num_n_blocks = (max_seqlen_k + 127) // 128
    if num_n_blocks <= 1:
        return 1
    max_splits = min(128, num_sms, num_n_blocks)
    if max_splits > MAX_SPLITS:
        max_splits = MAX_SPLITS

    def ceildiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    def eligible(s: int) -> bool:
        return s == 1 or ceildiv(num_n_blocks, s) != ceildiv(num_n_blocks, s - 1)

    max_eff = 0.0
    effs: list[float] = []
    for s in range(1, max_splits + 1):
        if not eligible(s):
            effs.append(0.0)
        else:
            n_waves = float(batch_nheads_mblocks * s) / float(num_sms)
            eff = n_waves / math.ceil(n_waves)
            if eff > max_eff:
                max_eff = eff
            effs.append(eff)
    for s in range(1, max_splits + 1):
        if eligible(s) and effs[s - 1] >= 0.85 * max_eff:
            return s
    return 1


def _resolve_splits(
    w_left: int, max_seqlen_k: int, max_num_splits: int, num_sms: int, base_scale_mode: int = 1
) -> tuple[int, int, int]:
    """Return (grid_splits, fa2_splits, mode) matching _vllm_fa2_C::varlen_fwd."""
    if max_num_splits == 1:
        max_active = (w_left + 65) if w_left >= 0 else max_seqlen_k
        g_splits = 8 if (NUM_SPLITS >= 8 and max_active <= MAX_NBLOCKS64 * 64) else 1
        one_pass_bit = 8 if (ONE_PASS and g_splits in (4, 8, 16) and max_active <= ONE_PASS_MAX_TOKENS) else 0
        return g_splits, 1, (base_scale_mode | one_pass_bit)
    if w_left >= 0:
        # With num_splits=0 and sliding window on paged KV, FA2 dispatches to non-splitkv kBlockN=128.
        max_active = w_left + 129
        g_splits = 8 if (NUM_SPLITS >= 8 and max_active <= 128 * 128) else 1
        one_pass_bit = 8 if (ONE_PASS and g_splits in (4, 8, 16) and max_active <= ONE_PASS_MAX_TOKENS) else 0
        return g_splits, 1, (base_scale_mode | 4 | one_pass_bit)
    if max_num_splits > 1:
        return NUM_SPLITS, min(MAX_SPLITS, max_num_splits), base_scale_mode
    f_splits = fa2_num_splits_heuristic(max_seqlen_k, num_sms)
    if f_splits == 1:
        g_splits = 8 if (NUM_SPLITS >= 8 and max_seqlen_k <= 128 * 128) else 1
        one_pass_bit = 8 if (ONE_PASS and g_splits in (4, 8, 16) and max_seqlen_k <= ONE_PASS_MAX_TOKENS) else 0
        return g_splits, 1, (base_scale_mode | 4 | one_pass_bit)
    return NUM_SPLITS, f_splits, base_scale_mode


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
    scale_mode = 1
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
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

        # 1. Pick scale_mode (1 = double(scale)*M_LOG2E, 3 = float(scale)*float(M_LOG2E)) with 0 mismatches.
        found_mode = False
        for cand_mode in (1, 3):
            mismatches = 0
            for seq_k in (1, 17, 64, 128, 255, 512, 576, 1024):
                for win in ((-1, -1), (511, 0)):
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
                        block_size, win[0], NUM_SPLITS, 1, cand_mode, scale,
                        q.stride(1), got.stride(1),
                        kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1),
                        stream,
                    )
                    torch.cuda.synchronize(device)
                    if st != 0 or not torch.isfinite(got).all():
                        mismatches += 100000
                        break
                    mismatches += int((ref.view(torch.int16) != got.view(torch.int16)).sum().item())
                if mismatches > 0:
                    break
            if mismatches == 0:
                scale_mode = cand_mode
                found_mode = True
                break

        if found_mode:
            ok = True
            # 2. Verify strict 0-ULP against BOTH num_splits=1 AND production (max_seqlen_k=8192, num_splits=0)
            for seq_k in (1, 17, 64, 128, 255, 511, 512, 513, 576, 639, 640, 1024, 1152, 2048, 2500):
                for win in ((-1, -1), (511, 0), (2047, 0)):
                    bt_cur = bt.clone()
                    if win[0] >= 0 and seq_k > win[0] + 1:
                        expired_pages = (seq_k - (win[0] + 1)) // block_size
                        if expired_pages > 0:
                            bt_cur[0, :expired_pages] = 0
                    for max_sk, ns in ((seq_k, 1), (8192, 0)):
                        q = (torch.randn((1, 32, 128), generator=gen, device=device) * 1.5).to(torch.bfloat16).contiguous()
                        sk = torch.tensor([seq_k], dtype=torch.int32, device=device)
                        ref = torch.empty_like(q)
                        got = torch.empty_like(q)
                        flash_attn_varlen_func(
                            q=q, k=k_cache, v=v_cache, out=ref,
                            cu_seqlens_q=cu_q, max_seqlen_q=1,
                            seqused_k=sk, max_seqlen_k=max_sk,
                            softmax_scale=scale, causal=True,
                            window_size=list(win), block_table=bt_cur,
                            fa_version=2, num_splits=ns,
                        )
                        g_splits, f_splits, run_mode = _resolve_splits(win[0], max_sk, ns, num_sms, scale_mode)
                        st = _lib.muse_gqa16_decode(
                            q.data_ptr(), kv_cache.data_ptr(), got.data_ptr(),
                            o_part.data_ptr(), lse_part.data_ptr(), cnt.data_ptr(),
                            sk.data_ptr(), bt_cur.data_ptr(),
                            block_size, win[0], g_splits, f_splits, run_mode, scale,
                            q.stride(1), got.stride(1),
                            kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1),
                            stream,
                        )
                        torch.cuda.synchronize(device)
                        if (
                            st != 0
                            or not torch.isfinite(got).all()
                            or int(cnt.abs().max().item()) != 0
                            or not torch.equal(ref.view(torch.int16), got.view(torch.int16))
                        ):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
    except Exception:  # noqa: BLE001
        logger.warning("muse_gqa16_attn: self-check raised on cuda:%d", idx, exc_info=True)
        ok = False

    _state[idx] = (ok, scale_mode, num_sms)
    if ok:
        logger.info(
            "muse_gqa16_attn: strict 0-ULP admitted on cuda:%d (splits=%d -> %d CTAs, 1pass_sw=%d, scale_mode=%d, full_splits_8k=%d)",
            idx, NUM_SPLITS, NUM_SPLITS * 2, int(ONE_PASS), scale_mode, fa2_num_splits_heuristic(8192, num_sms),
        )
    else:
        logger.warning("muse_gqa16_attn: strict 0-ULP self-check failed on cuda:%d; keeping stock FlashAttention-2", idx)
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
    max_seqlen_k: int = 8192,
    max_num_splits: int = 0,
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

    g_splits, f_splits, run_mode = _resolve_splits(
        w_left, int(max_seqlen_k), int(max_num_splits), st[2], st[1]
    )
    if f_splits > MAX_SPLITS:
        return False
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
        run_mode,
        softmax_scale,
        query.stride(1),
        output.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(1),
        stream,
    )
    return rc == 0
