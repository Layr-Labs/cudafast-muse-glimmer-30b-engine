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
SW_SPLITS = max(1, min(MAX_SPLITS, int(os.environ.get("MUSE_GQA16_SW_SPLITS", "4"))))
SW_PIPE = os.environ.get("MUSE_GQA16_SW_PIPE", "0").strip() not in ("0", "", "false", "no")
FULL_SPLITS = max(1, min(MAX_SPLITS, int(os.environ.get("MUSE_GQA16_FULL_SPLITS", "4"))))
FULL_PIPE = os.environ.get("MUSE_GQA16_FULL_PIPE", "1").strip() not in ("0", "", "false", "no")
ONE_PASS = os.environ.get("MUSE_GQA16_1PASS", "1").strip() not in ("0", "", "false", "no")
ONE_PASS_MAX_TOKENS = int(os.environ.get("MUSE_GQA16_1PASS_MAX_TOKENS", "2176"))
PREFILL_ENABLED = os.environ.get("MUSE_GQA16_PREFILL", "1").strip() not in ("0", "", "false", "no")
PREFILL_FUSE_ENABLED = os.environ.get("MUSE_GQA16_PREFILL_FUSE", "1").strip() not in ("0", "", "false", "no")
PREFILL_PDL_ENABLED = os.environ.get("MUSE_GQA16_PREFILL_PDL", "1").strip() not in ("0", "", "false", "no")
VERIFY_ENABLED = os.environ.get("MUSE_GQA16_VERIFY", "1").strip() not in ("0", "", "false", "no")
VERIFY_SPLITS = int(os.environ.get("MUSE_GQA16_VERIFY_SPLITS", "1"))
VERIFY_PIPE = os.environ.get("MUSE_GQA16_VERIFY_PIPE", "1").strip() not in ("0", "", "false", "no")
NATIVE_LEAF_ENABLED = os.environ.get("MUSE_NATIVE_LEAF", "1").strip() not in ("0", "", "false", "no")

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
    if hasattr(_lib, "muse_gqa16_prefill"):
        _lib.muse_gqa16_prefill.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
        ]
        _lib.muse_gqa16_prefill.restype = ctypes.c_int
    if hasattr(_lib, "muse_gqa16_verify"):
        _lib.muse_gqa16_verify.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
        ]
        _lib.muse_gqa16_verify.restype = ctypes.c_int
    if hasattr(_lib, "muse_gqa16_verify_leaf"):
        _lib.muse_gqa16_verify_leaf.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        _lib.muse_gqa16_verify_leaf.restype = ctypes.c_int
except OSError:
    _lib = None

_scratch: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_state: dict[int, tuple[bool, int, int]] = {}  # dev_idx -> (admitted, scale_mode, num_sms)
_prefill_state: dict[int, tuple[bool, int]] = {}  # dev_idx -> (admitted, scale_mode)
_verify_state: dict[int, tuple[bool, int]] = {}  # dev_idx -> (admitted, scale_mode)
_leaf_verify_admitted: dict[int, bool] = {}  # dev_idx -> 0-ULP native 1-leaf verify admitted
_leaf_flag_bufs: dict[int, torch.Tensor] = {}  # dev_idx -> static int32[1] device flag
_pending_prefill: tuple | None = None
_pf_counts: dict[str, int] = {"defer": 0, "fused1": 0, "fused2": 0, "direct0": 0, "flush": 0, "miss": 0}


def get_leaf_flag(device: torch.device) -> torch.Tensor | None:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    buf = _leaf_flag_bufs.get(idx)
    if buf is not None:
        return buf
    if torch.cuda.is_current_stream_capturing():
        return None
    with torch.inference_mode(False):
        buf = torch.zeros((49,), dtype=torch.int32, device=device)
    _leaf_flag_bufs[idx] = buf
    return buf


def is_leaf_verify_admitted(device: torch.device) -> bool:
    if not NATIVE_LEAF_ENABLED or VERIFY_SPLITS != 1:
        return False
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _verify_state:
        if torch.cuda.is_current_stream_capturing():
            return False
        self_check_verify(device)
    return bool(_leaf_verify_admitted.get(idx, False))


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
        g_splits = SW_SPLITS if (SW_SPLITS in (4, 8, 16) and max_active <= ONE_PASS_MAX_TOKENS and ONE_PASS) else (
            8 if (NUM_SPLITS >= 8 and max_active <= 128 * 128) else 1
        )
        one_pass_bit = 8 if (ONE_PASS and g_splits in (4, 8, 16) and max_active <= ONE_PASS_MAX_TOKENS) else 0
        sw_pipe_bit = 16 if (SW_PIPE and one_pass_bit != 0) else 0
        return g_splits, 1, (base_scale_mode | 4 | one_pass_bit | sw_pipe_bit)
    pipe_bit = 16 if FULL_PIPE else 0
    if max_num_splits > 1:
        return FULL_SPLITS, min(MAX_SPLITS, max_num_splits), (base_scale_mode | pipe_bit)
    f_splits = fa2_num_splits_heuristic(max_seqlen_k, num_sms)
    if f_splits == 1:
        g_splits = 8 if (NUM_SPLITS >= 8 and max_seqlen_k <= 128 * 128) else 1
        one_pass_bit = 8 if (ONE_PASS and g_splits in (4, 8, 16) and max_seqlen_k <= ONE_PASS_MAX_TOKENS) else 0
        return g_splits, 1, (base_scale_mode | 4 | one_pass_bit)
    return FULL_SPLITS, f_splits, (base_scale_mode | pipe_bit)


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
            "muse_gqa16_attn: strict 0-ULP admitted on cuda:%d (sw_splits=%d -> %d CTAs, full_splits=%d -> %d CTAs, 1pass_sw=%d, sw_pipe=%d, full_pipe=%d, scale_mode=%d, full_splits_8k=%d)",
            idx, SW_SPLITS, SW_SPLITS * 2, FULL_SPLITS, FULL_SPLITS * 2, int(ONE_PASS), int(SW_PIPE), int(FULL_PIPE), scale_mode, fa2_num_splits_heuristic(8192, num_sms),
        )
        if PREFILL_ENABLED and idx not in _prefill_state:
            self_check_prefill(device)
        if VERIFY_ENABLED and idx not in _verify_state:
            self_check_verify(device)
    else:
        logger.warning("muse_gqa16_attn: strict 0-ULP self-check failed on cuda:%d; keeping stock FlashAttention-2", idx)
    return ok


def self_check_verify(device: torch.device) -> bool:
    if not VERIFY_ENABLED or _lib is None or not hasattr(_lib, "muse_gqa16_verify") or device.type != "cuda":
        return False
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _verify_state.get(idx)
    if cached is not None:
        return cached[0]
    if torch.cuda.is_current_stream_capturing():
        return False

    ok = False
    scale_mode = 1
    g_splits = VERIFY_SPLITS if VERIFY_SPLITS in (1, 4, 8, 16) else 1
    pipe_bit = 16 if VERIFY_PIPE else 0
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

        gen = torch.Generator(device=device).manual_seed(20261126)
        scale = float(128.0 ** -0.5)
        block_size = 16
        num_blocks = 512
        kv_cache = (torch.randn((num_blocks, 2, block_size, 256), generator=gen, device=device) * 0.6).to(torch.bfloat16)
        k_cache, v_cache = kv_cache.transpose(1, 2).split(128, dim=-1)
        perm = torch.randperm(num_blocks - 1, generator=gen, device=device, dtype=torch.int32) + 1
        bt = perm.unsqueeze(0).contiguous()
        stream = torch.cuda.current_stream(device).cuda_stream

        for cand_mode in (1, 3):
            all_match = True
            for m_tok in (2, 4, 7, 8, 12, 16):
                q = (torch.randn((m_tok, 32, 128), generator=gen, device=device) * 1.1).to(torch.bfloat16).contiguous()
                cu_q = torch.tensor([0, m_tok], dtype=torch.int32, device=device)
                for seq_k in (518, 640):
                    sk = torch.tensor([seq_k], dtype=torch.int32, device=device)
                    for win in ((511, 0), (-1, -1)):
                        ref = torch.empty_like(q)
                        flash_attn_varlen_func(
                            q=q, k=k_cache, v=v_cache, out=ref,
                            cu_seqlens_q=cu_q, max_seqlen_q=m_tok,
                            seqused_k=sk, max_seqlen_k=8192,
                            softmax_scale=scale, causal=True,
                            window_size=win, block_table=bt, fa_version=2,
                        )
                        got = torch.empty_like(q)
                        rc = _lib.muse_gqa16_verify(
                            q.data_ptr(), kv_cache.data_ptr(), got.data_ptr(),
                            sk.data_ptr(), bt.data_ptr(), cu_q.data_ptr(),
                            m_tok, block_size, win[0], g_splits, cand_mode | pipe_bit, scale,
                            q.stride(0), q.stride(1), got.stride(0), got.stride(1),
                            kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1),
                            stream,
                        )
                        torch.cuda.synchronize(device)
                        if rc != 0 or not torch.isfinite(got).all() or not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
                            all_match = False
                            break
                    if not all_match:
                        break
                if not all_match:
                    break
            # Also verify padded CUDA-graph replay (grid max_q=8, runtime actual_m=7).
            if all_match:
                q8 = (torch.randn((8, 32, 128), generator=gen, device=device) * 1.1).to(torch.bfloat16).contiguous()
                cu_q7 = torch.tensor([0, 7], dtype=torch.int32, device=device)
                sk = torch.tensor([640], dtype=torch.int32, device=device)
                ref7 = torch.empty((7, 32, 128), dtype=torch.bfloat16, device=device)
                flash_attn_varlen_func(
                    q=q8[:7], k=k_cache, v=v_cache, out=ref7,
                    cu_seqlens_q=cu_q7, max_seqlen_q=7,
                    seqused_k=sk, max_seqlen_k=8192,
                    softmax_scale=scale, causal=True,
                    window_size=(511, 0), block_table=bt, fa_version=2,
                )
                got8 = torch.zeros_like(q8)
                rc = _lib.muse_gqa16_verify(
                    q8.data_ptr(), kv_cache.data_ptr(), got8.data_ptr(),
                    sk.data_ptr(), bt.data_ptr(), cu_q7.data_ptr(),
                    8, block_size, 511, g_splits, cand_mode | pipe_bit, scale,
                    q8.stride(0), q8.stride(1), got8.stride(0), got8.stride(1),
                    kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1),
                    stream,
                )
                torch.cuda.synchronize(device)
                if rc != 0 or not torch.equal(ref7.view(torch.int16), got8[:7].view(torch.int16)):
                    all_match = False
            if all_match:
                ok = True
                scale_mode = cand_mode
                break
    except Exception:  # noqa: BLE001
        logger.warning("muse_gqa16_attn: verify self-check raised on cuda:%d", idx, exc_info=True)
        ok = False

    _verify_state[idx] = (ok, scale_mode)
    leaf_ok = False
    if ok and NATIVE_LEAF_ENABLED and g_splits == 1 and hasattr(_lib, "muse_gqa16_verify_leaf"):
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

            gen_l = torch.Generator(device=device).manual_seed(20261127)
            scale_l = float(128.0 ** -0.5)
            bs_l = 16
            nblks_l = 64
            kv_l = (torch.randn((nblks_l, 2, bs_l, 256), generator=gen_l, device=device) * 0.6).to(torch.bfloat16)
            perm_l = torch.randperm(nblks_l - 1, generator=gen_l, device=device, dtype=torch.int32) + 1
            bt_l = perm_l.unsqueeze(0).contiguous()
            lf_on = torch.ones((1,), dtype=torch.int32, device=device)
            lf_k1 = torch.tensor([1], dtype=torch.int32, device=device)
            lf_k2 = torch.tensor([2], dtype=torch.int32, device=device)
            cu_q7 = torch.tensor([0, 7], dtype=torch.int32, device=device)
            cu_q8 = torch.tensor([0, 8], dtype=torch.int32, device=device)
            st_l = torch.cuda.current_stream(device).cuda_stream
            leaf_all = True
            for seq_k in (16, 17, 32, 33, 45, 46, 48, 128, 129, 134, 141, 142, 520):
                # 1) Legacy M=7: identical K/V at p+1 (seq_k-6) and p+6 (seq_k-1), Q[6] == Q[1].
                kv_cur = kv_l.clone()
                p1 = seq_k - 6
                p6 = seq_k - 1
                b1, s1 = int(bt_l[0, p1 // bs_l].item()), p1 % bs_l
                b6, s6 = int(bt_l[0, p6 // bs_l].item()), p6 % bs_l
                kv_cur[b6, :, s6, :].copy_(kv_cur[b1, :, s1, :])
                k_c, v_c = kv_cur.transpose(1, 2).split(128, dim=-1)
                sk = torch.tensor([seq_k], dtype=torch.int32, device=device)
                for max_q in (7, 8):
                    q_l = (torch.randn((max_q, 32, 128), generator=gen_l, device=device) * 1.1).to(torch.bfloat16).contiguous()
                    q_l[6].copy_(q_l[1])
                    for win in ((-1, -1), (511, 0)):
                        ref_l = torch.empty((7, 32, 128), dtype=torch.bfloat16, device=device)
                        flash_attn_varlen_func(
                            q=q_l[:7], k=k_c, v=v_c, out=ref_l,
                            cu_seqlens_q=cu_q7, max_seqlen_q=7,
                            seqused_k=sk, max_seqlen_k=8192,
                            softmax_scale=scale_l, causal=True,
                            window_size=win, block_table=bt_l, fa_version=2,
                        )
                        got_l = torch.zeros_like(q_l)
                        rc_l = _lib.muse_gqa16_verify_leaf(
                            q_l.data_ptr(), kv_cur.data_ptr(), got_l.data_ptr(),
                            sk.data_ptr(), bt_l.data_ptr(), cu_q7.data_ptr(),
                            max_q, bs_l, win[0], 1, scale_mode | pipe_bit, scale_l,
                            q_l.stride(0), q_l.stride(1), got_l.stride(0), got_l.stride(1),
                            kv_cur.stride(0), kv_cur.stride(2), kv_cur.stride(1),
                            lf_on.data_ptr(), st_l,
                        )
                        torch.cuda.synchronize(device)
                        if (
                            rc_l != 0
                            or not torch.isfinite(got_l[:7]).all()
                            or not torch.equal(got_l[6].view(torch.int16), got_l[1].view(torch.int16))
                            or not torch.equal(got_l[:6].view(torch.int16), ref_l[:6].view(torch.int16))
                        ):
                            leaf_all = False
                            break
                    if not leaf_all:
                        break
                if not leaf_all:
                    break
                # 2) M=8 zero-clamp: test both lk=1 (row 7 == row 1) and lk=2 (row 7 == row 2) plus full chain got[:7] == ref[:7].
                for lk, lf_t in ((1, lf_k1), (2, lf_k2)):
                    kv_m8 = kv_l.clone()
                    pk = seq_k - 8 + lk
                    p7 = seq_k - 1
                    bk, sk_slot = int(bt_l[0, pk // bs_l].item()), pk % bs_l
                    b7, s7_slot = int(bt_l[0, p7 // bs_l].item()), p7 % bs_l
                    kv_m8[b7, :, s7_slot, :].copy_(kv_m8[bk, :, sk_slot, :])
                    k_c8, v_c8 = kv_m8.transpose(1, 2).split(128, dim=-1)
                    q_m8 = (torch.randn((8, 32, 128), generator=gen_l, device=device) * 1.1).to(torch.bfloat16).contiguous()
                    q_m8[7].copy_(q_m8[lk])
                    for win in ((-1, -1), (511, 0)):
                        ref_m8 = torch.empty((8, 32, 128), dtype=torch.bfloat16, device=device)
                        flash_attn_varlen_func(
                            q=q_m8, k=k_c8, v=v_c8, out=ref_m8,
                            cu_seqlens_q=cu_q8, max_seqlen_q=8,
                            seqused_k=sk, max_seqlen_k=8192,
                            softmax_scale=scale_l, causal=True,
                            window_size=win, block_table=bt_l, fa_version=2,
                        )
                        got_m8 = torch.zeros_like(q_m8)
                        rc_m8 = _lib.muse_gqa16_verify_leaf(
                            q_m8.data_ptr(), kv_m8.data_ptr(), got_m8.data_ptr(),
                            sk.data_ptr(), bt_l.data_ptr(), cu_q8.data_ptr(),
                            8, bs_l, win[0], 1, scale_mode | pipe_bit, scale_l,
                            q_m8.stride(0), q_m8.stride(1), got_m8.stride(0), got_m8.stride(1),
                            kv_m8.stride(0), kv_m8.stride(2), kv_m8.stride(1),
                            lf_t.data_ptr(), st_l,
                        )
                        torch.cuda.synchronize(device)
                        if (
                            rc_m8 != 0
                            or not torch.isfinite(got_m8).all()
                            or not torch.equal(got_m8[7].view(torch.int16), got_m8[lk].view(torch.int16))
                            or not torch.equal(got_m8[:7].view(torch.int16), ref_m8[:7].view(torch.int16))
                        ):
                            leaf_all = False
                            break
                    if not leaf_all:
                        break
                if not leaf_all:
                    break
                # 3) Generalized bit-packed arbitrary-depth (1..6-level) tree table mode (sentinel -1, int32[49]):
                #    Test M=12 1-level (5 leaves), M=16 1-level (9 leaves), AND M=16 deep 6-level + 3-level branches!
                #    Each tree spec is (tree_m, tuple of (k_r, parent_row)) for rows r = 7..tree_m-1.
                for tree_m, tree_nodes in (
                    (12, ((1, -1), (1, -1), (2, -1), (2, -1), (3, -1))),
                    (16, ((1, -1), (1, -1), (2, -1), (2, -1), (3, -1), (1, -1), (3, -1), (4, -1), (5, -1))),
                    # Deep 6-level chain (rows 7..12 -> k=1..6) + 3-level branch (rows 13..15 -> k=2..4):
                    (16, ((1, -1), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11), (2, -1), (3, 13), (4, 14))),
                ):
                    if seq_k < tree_m:
                        continue
                    kv_t = kv_l.clone()
                    q_t = (torch.randn((tree_m, 32, 128), generator=gen_l, device=device) * 1.1).to(torch.bfloat16).contiguous()
                    tbl = [-1] + [0] * 48
                    for q_i in range(7):
                        tbl[1 + 3 * q_i] = tree_m - 1 - q_i
                        tbl[1 + 3 * q_i + 1] = -1
                        tbl[1 + 3 * q_i + 2] = 0
                    row_base_sub = [-1] * tree_m
                    row_depth = [0] * tree_m
                    row_packed = [0] * tree_m
                    for li, (lk, par) in enumerate(tree_nodes):
                        r_row = 7 + li
                        pk = seq_k - tree_m + lk
                        pr = seq_k - tree_m + r_row
                        bk, sk_slot = int(bt_l[0, pk // bs_l].item()), pk % bs_l
                        br, sr_slot = int(bt_l[0, pr // bs_l].item()), pr % bs_l
                        kv_t[br, :, sr_slot, :].copy_(kv_t[bk, :, sk_slot, :])
                        q_t[r_row].copy_(q_t[lk])
                        if par < 0:
                            b_sub = tree_m - lk
                            dp = 1
                            pkd = r_row & 0xF
                        else:
                            b_sub = row_base_sub[par]
                            dp = row_depth[par] + 1
                            pkd = row_packed[par] | ((r_row & 0xF) << (4 * (dp - 1)))
                        row_base_sub[r_row] = b_sub
                        row_depth[r_row] = dp
                        row_packed[r_row] = pkd
                        tbl[1 + 3 * r_row] = tree_m - (lk + 1)
                        tbl[1 + 3 * r_row + 1] = b_sub
                        tbl[1 + 3 * r_row + 2] = pkd
                    lf_tbl = torch.tensor(tbl, dtype=torch.int32, device=device)
                    cu_qt = torch.tensor([0, tree_m], dtype=torch.int32, device=device)
                    k_ct, v_ct = kv_t.transpose(1, 2).split(128, dim=-1)
                    for win in ((-1, -1), (511, 0)):
                        ref_t = torch.empty((tree_m, 32, 128), dtype=torch.bfloat16, device=device)
                        flash_attn_varlen_func(
                            q=q_t, k=k_ct, v=v_ct, out=ref_t,
                            cu_seqlens_q=cu_qt, max_seqlen_q=tree_m,
                            seqused_k=sk, max_seqlen_k=8192,
                            softmax_scale=scale_l, causal=True,
                            window_size=win, block_table=bt_l, fa_version=2,
                        )
                        got_t = torch.zeros_like(q_t)
                        rc_t = _lib.muse_gqa16_verify_leaf(
                            q_t.data_ptr(), kv_t.data_ptr(), got_t.data_ptr(),
                            sk.data_ptr(), bt_l.data_ptr(), cu_qt.data_ptr(),
                            tree_m, bs_l, win[0], 1, scale_mode | pipe_bit, scale_l,
                            q_t.stride(0), q_t.stride(1), got_t.stride(0), got_t.stride(1),
                            kv_t.stride(0), kv_t.stride(2), kv_t.stride(1),
                            lf_tbl.data_ptr(), st_l,
                        )
                        torch.cuda.synchronize(device)
                        if (
                            rc_t != 0
                            or not torch.isfinite(got_t).all()
                            or not torch.equal(got_t[:7].view(torch.int16), ref_t[:7].view(torch.int16))
                        ):
                            leaf_all = False
                            break
                        for li, (lk, _par) in enumerate(tree_nodes):
                            if not torch.equal(got_t[7 + li].view(torch.int16), ref_t[lk].view(torch.int16)):
                                leaf_all = False
                                break
                        if not leaf_all:
                            break
                    if not leaf_all:
                        break
                if not leaf_all:
                    break
            leaf_ok = leaf_all
        except Exception:  # noqa: BLE001
            logger.warning("muse_gqa16_attn: native 1-leaf verify self-check raised on cuda:%d", idx, exc_info=True)
            leaf_ok = False
    _leaf_verify_admitted[idx] = leaf_ok
    if ok:
        logger.info(
            "muse_gqa16_attn: strict 0-ULP verify (M=2..16) admitted on cuda:%d (grid_splits=%d, pipe=%d, scale_mode=%d, native_leaf=%d)",
            idx, g_splits, int(VERIFY_PIPE), scale_mode, int(leaf_ok),
        )
    else:
        logger.warning("muse_gqa16_attn: strict 0-ULP verify self-check failed on cuda:%d; keeping stock FA2", idx)
    return ok


def try_verify_gqa16(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    seqused_k: torch.Tensor | None,
    block_table: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor,
    num_actual_tokens: int,
    softmax_scale: float,
    window: tuple[int, int] | None,
    max_num_splits: int = 0,
) -> bool:
    if not VERIFY_ENABLED or _lib is None or not hasattr(_lib, "muse_gqa16_verify"):
        return False
    if num_actual_tokens < 2 or num_actual_tokens > 16 or max_num_splits not in (0, 1):
        return False
    device = query.device
    idx = device.index if device.index is not None else torch.cuda.current_device()
    vst = _verify_state.get(idx)
    if vst is None:
        if not self_check_verify(device):
            return False
        vst = _verify_state[idx]
    elif not vst[0]:
        return False

    if (
        seqused_k is None
        or block_table is None
        or query.dtype != torch.bfloat16
        or kv_cache.dtype != torch.bfloat16
        or output.dtype != torch.bfloat16
        or query.ndim != 3
        or output.ndim != 3
        or query.shape[0] < num_actual_tokens
        or output.shape[0] < num_actual_tokens
        or kv_cache.ndim != 4
        or kv_cache.shape[1] != 2
        or kv_cache.shape[3] != 256
        or kv_cache.stride(3) != 1
        or query.stride(2) != 1
        or output.stride(2) != 1
        or cu_seqlens_q.shape[0] != 2
        or cu_seqlens_q.dtype != torch.int32
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

    g_splits = VERIFY_SPLITS if VERIFY_SPLITS in (1, 4, 8, 16) else 1
    pipe_bit = 16 if VERIFY_PIPE else 0
    stream = torch.cuda.current_stream(device).cuda_stream
    leaf_ptr = 0
    if (
        NATIVE_LEAF_ENABLED
        and g_splits == 1
        and 7 <= num_actual_tokens <= 16
        and _leaf_verify_admitted.get(idx, False)
        and hasattr(_lib, "muse_gqa16_verify_leaf")
    ):
        lf = get_leaf_flag(device)
        if lf is not None:
            leaf_ptr = lf.data_ptr()
    if leaf_ptr != 0:
        rc = _lib.muse_gqa16_verify_leaf(
            query.data_ptr(),
            kv_cache.data_ptr(),
            output.data_ptr(),
            seqused_k.data_ptr(),
            block_table.data_ptr(),
            cu_seqlens_q.data_ptr(),
            int(num_actual_tokens),
            block_size,
            w_left,
            g_splits,
            vst[1] | pipe_bit,
            softmax_scale,
            query.stride(0),
            query.stride(1),
            output.stride(0),
            output.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(2),
            kv_cache.stride(1),
            leaf_ptr,
            stream,
        )
    else:
        rc = _lib.muse_gqa16_verify(
            query.data_ptr(),
            kv_cache.data_ptr(),
            output.data_ptr(),
            seqused_k.data_ptr(),
            block_table.data_ptr(),
            cu_seqlens_q.data_ptr(),
            int(num_actual_tokens),
            block_size,
            w_left,
            g_splits,
            vst[1] | pipe_bit,
            softmax_scale,
            query.stride(0),
            query.stride(1),
            output.stride(0),
            output.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(2),
            kv_cache.stride(1),
            stream,
        )
    return rc == 0


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


def self_check_prefill(device: torch.device) -> bool:
    if not PREFILL_ENABLED or _lib is None or not hasattr(_lib, "muse_gqa16_prefill") or device.type != "cuda":
        return False
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _prefill_state.get(idx)
    if cached is not None:
        return cached[0]
    if torch.cuda.is_current_stream_capturing():
        return False

    ok = False
    scale_mode = 1
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
        import muse_norm_glue as G

        gen = torch.Generator(device=device).manual_seed(20261125)
        scale = float(128.0 ** -0.5)
        num_blocks = 40
        block_size = 16
        raw_kv = (torch.randn((num_blocks, block_size, 2, 256), generator=gen, device=device) * 0.8).to(torch.bfloat16)
        raw_kv[0].zero_()
        kv_cache = raw_kv.permute(0, 2, 1, 3)
        k_cache, v_cache = kv_cache.transpose(1, 2).split(128, dim=-1)
        perm = torch.randperm(num_blocks - 1, generator=gen, device=device, dtype=torch.int32) + 1
        bt = perm[:32].unsqueeze(0).contiguous()
        sk = torch.tensor([512], dtype=torch.int32, device=device)
        stream = torch.cuda.current_stream(device).cuda_stream
        fp8_scale = torch.tensor([0.045], dtype=torch.float32, device=device)

        for cand_mode in (1, 3):
            all_match = True
            for block_size in (16, 64):
                num_blocks = (512 // block_size) + 8
                raw_kv = (torch.randn((num_blocks, block_size, 2, 256), generator=gen, device=device) * 0.8).to(torch.bfloat16)
                raw_kv[0].zero_()
                kv_cache = raw_kv.permute(0, 2, 1, 3)
                k_cache, v_cache = kv_cache.transpose(1, 2).split(128, dim=-1)
                perm = torch.randperm(num_blocks - 1, generator=gen, device=device, dtype=torch.int32) + 1
                bt = perm[: (512 // block_size)].unsqueeze(0).contiguous()
                for seq_q, q_off in ((512, 0), (64, 448)):
                    q = (torch.randn((seq_q, 32, 128), generator=gen, device=device) * 1.4).to(torch.bfloat16).contiguous()
                    gate = (torch.randn((seq_q, 4096), generator=gen, device=device) * 1.3).to(torch.bfloat16).contiguous()
                    cu_q = torch.tensor([0, seq_q], dtype=torch.int32, device=device)
                    for win in ((-1, -1), (511, 0)):
                        ref = torch.empty_like(q)
                        flash_attn_varlen_func(
                            q=q, k=k_cache, v=v_cache, out=ref,
                            cu_seqlens_q=cu_q, max_seqlen_q=seq_q,
                            seqused_k=sk, max_seqlen_k=512,
                            softmax_scale=scale, causal=True,
                            window_size=list(win), block_table=bt,
                            fa_version=2, num_splits=0,
                        )
                        # Mode 0: plain BF16
                        got0 = torch.empty_like(q)
                        rc0 = _lib.muse_gqa16_prefill(
                            q.data_ptr(), kv_cache.data_ptr(), 0, 0, got0.data_ptr(), bt.data_ptr(),
                            seq_q, q_off, block_size, 0, cand_mode, 0, scale,
                            q.stride(0), q.stride(1), got0.stride(0),
                            kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1), stream,
                        )
                        # Mode 1: fused mul_sigmoid BF16 (with PDL producer)
                        ref1 = torch.empty((seq_q, 4096), dtype=torch.bfloat16, device=device)
                        G.mul_sigmoid_bf16(gate, ref.view(seq_q, 4096), ref1, seq_q * 4096, stream)
                        got1 = torch.empty_like(ref1)
                        rc1 = _lib.muse_gqa16_prefill(
                            q.data_ptr(), kv_cache.data_ptr(), gate.data_ptr(), 0, got1.data_ptr(), bt.data_ptr(),
                            seq_q, q_off, block_size, 1, cand_mode, int(PREFILL_PDL_ENABLED), scale,
                            q.stride(0), q.stride(1), 4096,
                            kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1), stream,
                        )
                        # Mode 2: fused mul_sigmoid_fp8 (with PDL producer)
                        ref2 = torch.empty((seq_q, 4096), dtype=torch.float8_e4m3fn, device=device)
                        G.mul_sigmoid_fp8(gate, ref.view(seq_q, 4096), fp8_scale, ref2, seq_q * 4096, stream)
                        got2 = torch.empty_like(ref2)
                        rc2 = _lib.muse_gqa16_prefill(
                            q.data_ptr(), kv_cache.data_ptr(), gate.data_ptr(), fp8_scale.data_ptr(),
                            got2.data_ptr(), bt.data_ptr(),
                            seq_q, q_off, block_size, 2, cand_mode, int(PREFILL_PDL_ENABLED), scale,
                            q.stride(0), q.stride(1), 4096,
                            kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1), stream,
                        )
                        torch.cuda.synchronize(device)
                        if (
                            rc0 != 0 or rc1 != 0 or rc2 != 0
                            or not torch.equal(ref.view(torch.int16), got0.view(torch.int16))
                            or not torch.equal(ref1.view(torch.int16), got1.view(torch.int16))
                            or not torch.equal(ref2.view(torch.uint8), got2.view(torch.uint8))
                        ):
                            all_match = False
                            break
                    if not all_match:
                        break
                if not all_match:
                    break
            if all_match:
                scale_mode = cand_mode
                ok = True
                break
    except Exception:  # noqa: BLE001
        logger.warning("muse_gqa16_attn: prefill self-check raised on cuda:%d", idx, exc_info=True)
        ok = False

    _prefill_state[idx] = (ok, scale_mode)
    if ok:
        logger.info(
            "muse_gqa16_attn: prefill strict 0-ULP admitted on cuda:%d (fuse=%d, pdl=%d, scale_mode=%d)",
            idx, int(PREFILL_FUSE_ENABLED), int(PREFILL_PDL_ENABLED), scale_mode,
        )
    else:
        logger.warning("muse_gqa16_attn: prefill strict 0-ULP self-check failed on cuda:%d", idx)
    return ok


def flush_pending_prefill(stream: int | None = None) -> None:
    global _pending_prefill
    pend = _pending_prefill
    if pend is None:
        return
    _pending_prefill = None
    _out_ptr, _out_numel, q, kv_cache, bt, out, seqlen_q, q_offset, scale, scale_mode, saved_stream = pend
    st = saved_stream if stream is None else int(stream)
    bs = int(kv_cache.shape[2])
    rc = _lib.muse_gqa16_prefill(
        q.data_ptr(), kv_cache.data_ptr(), 0, 0, out.data_ptr(), bt.data_ptr(),
        seqlen_q, q_offset, bs, 0, scale_mode, 0, scale,
        q.stride(0), q.stride(1), out.stride(0),
        kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1), st,
    )
    if rc != 0:
        raise RuntimeError(f"muse_gqa16_prefill flush failed with cuda error {rc}")


def consume_prefill_mul_sigmoid(
    attn: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
    fp8_scale: torch.Tensor | None = None,
    stream: int | None = None,
) -> bool:
    global _pending_prefill
    pend = _pending_prefill
    if pend is None:
        return False
    _pending_prefill = None
    out_ptr, out_numel, q, kv_cache, bt, out_arg, seqlen_q, q_offset, scale, scale_mode, saved_stream = pend
    st = saved_stream if stream is None else int(stream)
    bs = int(kv_cache.shape[2])
    is_fp8 = fp8_scale is not None
    out_align_mask = 7 if is_fp8 else 15
    if (
        attn.data_ptr() != out_ptr
        or attn.numel() != out_numel
        or gate.numel() != out_numel
        or out.numel() != out_numel
        or (gate.data_ptr() & 15) != 0
        or (out.data_ptr() & out_align_mask) != 0
        or not gate.is_contiguous()
        or not out.is_contiguous()
    ):
        rc_f = _lib.muse_gqa16_prefill(
            q.data_ptr(), kv_cache.data_ptr(), 0, 0, out_arg.data_ptr(), bt.data_ptr(),
            seqlen_q, q_offset, bs, 0, scale_mode, 0, scale,
            q.stride(0), q.stride(1), out_arg.stride(0),
            kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1), st,
        )
        if rc_f != 0:
            raise RuntimeError(f"muse_gqa16_prefill fallback flush failed with cuda error {rc_f}")
        return False
    mode = 2 if is_fp8 else 1
    fp8_ptr = fp8_scale.data_ptr() if is_fp8 else 0
    pdl = 1 if PREFILL_PDL_ENABLED else 0
    rc = _lib.muse_gqa16_prefill(
        q.data_ptr(), kv_cache.data_ptr(), gate.data_ptr(), fp8_ptr, out.data_ptr(), bt.data_ptr(),
        seqlen_q, q_offset, bs, mode, scale_mode, pdl, scale,
        q.stride(0), q.stride(1), 4096,
        kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1), st,
    )
    if rc != 0:
        raise RuntimeError(f"muse_gqa16_prefill fused mode={mode} failed with cuda error {rc}")
    k = "fused2" if is_fp8 else "fused1"
    _pf_counts[k] += 1
    if _pf_counts["fused1"] + _pf_counts["fused2"] in (1, 51):
        logger.info("muse_gqa16_attn: prefill fused counts=%s", _pf_counts)
    return True


def try_prefill_gqa16(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    block_table: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor,
    orig_max_seqlen_q: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    window: tuple[int, int] | None,
    allow_defer: bool = True,
) -> bool:
    global _pending_prefill
    if not PREFILL_ENABLED or _lib is None or not hasattr(_lib, "muse_gqa16_prefill"):
        return False
    if _pending_prefill is not None:
        flush_pending_prefill()
    device = query.device
    idx = device.index if device.index is not None else torch.cuda.current_device()
    st = _prefill_state.get(idx)
    if st is None:
        if torch.cuda.is_current_stream_capturing() or not self_check_prefill(device):
            return False
        st = _prefill_state[idx]
    elif not st[0]:
        return False

    if (
        block_table is None
        or orig_max_seqlen_q != max_seqlen_k
        or max_seqlen_k < 128
        or (max_seqlen_k & 127) != 0
        or max_seqlen_q <= 0
        or max_seqlen_q > max_seqlen_k
        or (max_seqlen_q & 7) != 0
        or cu_seqlens_q.shape[0] != 2
        or query.dtype != torch.bfloat16
        or kv_cache.dtype != torch.bfloat16
        or output.dtype != torch.bfloat16
        or query.ndim != 3
        or output.ndim != 3
        or query.shape[0] != max_seqlen_q
        or query.shape[1] != 32
        or query.shape[2] != 128
        or output.shape != (max_seqlen_q, 32, 128)
        or not output.is_contiguous()
        or query.stride(2) != 1
        or (query.stride(0) & 7) != 0
        or (query.stride(1) & 7) != 0
        or kv_cache.ndim != 4
        or kv_cache.shape[1] != 2
        or kv_cache.shape[2] not in (16, 64)
        or kv_cache.shape[3] != 256
        or kv_cache.stride(3) != 1
        or block_table.dtype != torch.int32
        or block_table.ndim != 2
        or block_table.stride(1) != 1
        or block_table.shape[1] < (max_seqlen_k // int(kv_cache.shape[2]))
        or ((query.data_ptr() | kv_cache.data_ptr() | output.data_ptr()) & 15) != 0
    ):
        _pf_counts["miss"] += 1
        if _pf_counts["miss"] <= 3 and max_seqlen_k >= 128:
            logger.info(
                "muse_gqa16_attn: prefill skip orig_mq=%s mq=%s mk=%s cu_q=%s q=%s/%s out=%s/%s kv=%s bt=%s win=%s",
                orig_max_seqlen_q, max_seqlen_q, max_seqlen_k, tuple(cu_seqlens_q.shape),
                tuple(query.shape), query.stride(), tuple(output.shape), output.stride(),
                tuple(kv_cache.shape), None if block_table is None else tuple(block_table.shape), window,
            )
        return False

    if window is not None:
        if window[1] > 0:
            return False
        if window[0] >= 0 and window[0] < max_seqlen_k - 1:
            return False

    q_offset = max_seqlen_k - max_seqlen_q
    if (q_offset & 7) != 0:
        return False

    stream = torch.cuda.current_stream(device).cuda_stream
    can_defer = allow_defer and q_offset == 0 and PREFILL_FUSE_ENABLED
    if can_defer:
        try:
            from vllm.model_executor.models import muse_glimmer_norm_glue as _ng_mod
            can_defer = bool(getattr(_ng_mod, "_installed", False) and getattr(_ng_mod, "_MUL_SIGMOID_GLUE", False))
        except Exception:
            can_defer = False
    if can_defer:
        _pending_prefill = (
            output.data_ptr(),
            output.numel(),
            query,
            kv_cache,
            block_table,
            output,
            max_seqlen_q,
            q_offset,
            softmax_scale,
            st[1],
            stream,
        )
        return True

    bs = int(kv_cache.shape[2])
    rc = _lib.muse_gqa16_prefill(
        query.data_ptr(), kv_cache.data_ptr(), 0, 0, output.data_ptr(), block_table.data_ptr(),
        max_seqlen_q, q_offset, bs, 0, st[1], 0, softmax_scale,
        query.stride(0), query.stride(1), output.stride(0),
        kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1), stream,
    )
    return rc == 0
