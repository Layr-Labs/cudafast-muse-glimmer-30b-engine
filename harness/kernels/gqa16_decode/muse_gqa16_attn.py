"""Strict 0-ULP FA2-exact GQA-16 Tensor-Core decode & causal prefill attention for MuseGlimmer (32Q / 2KV / D=128).

Controls (no VLLM_* env vars required):
  MUSE_GQA16_DECODE=1            enable 0-ULP GQA-16 M=1 decode kernel (admitted only after strict 0-ULP self-check).
  MUSE_GQA16_SPLITS=8            worker CTA count per KV head (default 8 -> 16 CTAs).
  MUSE_GQA16_PREFILL=1           enable 0-ULP GQA-16 causal prefill kernel (0 disables).
  MUSE_GQA16_PREFILL_SIGMOID=1   fuse mul_sigmoid_{bf16,fp8} into prefill attention epilogue after per-layer bitwise admission (0 disables).
"""
from __future__ import annotations

import ctypes
import hashlib
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

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
SW_SPLITS = max(1, min(MAX_SPLITS, int(os.environ.get("MUSE_GQA16_SW_SPLITS", str(NUM_SPLITS)))))
SW_PIPE = os.environ.get("MUSE_GQA16_SW_PIPE", "0").strip() not in ("0", "", "false", "no")
FULL_SPLITS = max(1, min(MAX_SPLITS, int(os.environ.get("MUSE_GQA16_FULL_SPLITS", "4"))))
FULL_PIPE = os.environ.get("MUSE_GQA16_FULL_PIPE", "1").strip() not in ("0", "", "false", "no")
ONE_PASS = os.environ.get("MUSE_GQA16_1PASS", "1").strip() not in ("0", "", "false", "no")
ONE_PASS_MAX_TOKENS = int(os.environ.get("MUSE_GQA16_1PASS_MAX_TOKENS", "2176"))

PREFILL_ENABLED = os.environ.get("MUSE_GQA16_PREFILL", "1").strip() not in ("0", "", "false", "no")
PREFILL_SIGMOID = os.environ.get(
    "MUSE_GQA16_PREFILL_SIGMOID",
    os.environ.get("MUSE_GQA16_PREFILL_FUSE_GATE", "1"),
).strip() not in ("0", "", "false", "no")

_MUL_SIGMOID_BF16_HASH = "d7dbdd6fbb45cd65ebdcb6c1fde8d461e74484b611165ad886a6d2e589ff6573"
_SIGMOID_FP8_HASH = "68ddaaf112b7f2fbd8d452b5eb8b93c39567cfe29e167a625d3194458669ad01"

_lib = None
_has_prefill_sym = False
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
    if hasattr(_lib, "muse_gqa16_prefill") and hasattr(_lib, "muse_gqa16_prefill_init"):
        _lib.muse_gqa16_prefill_init.argtypes = []
        _lib.muse_gqa16_prefill_init.restype = ctypes.c_int
        _lib.muse_gqa16_prefill.argtypes = [
            ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
        ]
        _lib.muse_gqa16_prefill.restype = ctypes.c_int
        _has_prefill_sym = True
except OSError:
    _lib = None

_scratch: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_state: dict[int, tuple[bool, int, int]] = {}  # dev_idx -> (admitted, scale_mode, num_sms)
_prefill_state: dict[int, tuple[bool, int]] = {}  # dev_idx -> (admitted, scale_mode)

_sigmoid_hook_installed = False
_pending_prefill: tuple[Any, ...] | None = None
_probe_prefill: tuple[Any, ...] | None = None
_prefill_fuse_counts = {"admitted": 0, "rejected": 0, "fused_bf16": 0, "fused_fp8": 0}


def _canonical_triton_hash(src: str) -> str:
    m = re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE)
    if m is None:
        return ""
    body = re.sub(r"^def\s+\w+\s*\(", "def KERNEL(", src[m.start() :]).rstrip()
    body = "\n".join(line.rstrip() for line in body.splitlines())
    return hashlib.sha256(body.encode()).hexdigest()


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
    else:
        logger.warning("muse_gqa16_attn: strict 0-ULP self-check failed on cuda:%d; keeping stock FlashAttention-2", idx)
    return ok


def self_check_prefill(device: torch.device) -> bool:
    if not PREFILL_ENABLED or _lib is None or not _has_prefill_sym or device.type != "cuda":
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

        with torch.cuda.device(device):
            if _lib.muse_gqa16_prefill_init() != 0:
                _prefill_state[idx] = (False, 1)
                return False

            gen = torch.Generator(device=device).manual_seed(20261130)
            scale = float(128.0 ** -0.5)
            stream = torch.cuda.current_stream(device).cuda_stream
            try:
                import muse_norm_glue as _ng
                has_ng = _ng.lib() is not None
            except Exception:  # noqa: BLE001
                _ng = None
                has_ng = False

            for cand_mode in (1, 3):
                mode_ok = True
                for block_size in (16, 32):
                    num_blocks = 64
                    raw_kv = (torch.randn((num_blocks, block_size, 2, 256), generator=gen, device=device) * 0.7).to(torch.bfloat16)
                    raw_kv[0].zero_()
                    kv_cache = raw_kv.permute(0, 2, 1, 3)
                    k_cache, v_cache = kv_cache.transpose(1, 2).split(128, dim=-1)
                    bt = (torch.randperm(num_blocks - 1, generator=gen, device=device, dtype=torch.int32) + 1).unsqueeze(0).contiguous()
                    for mq, seq_k in ((512, 512), (64, 512), (17, 17), (512, 640)):
                        q = (torch.randn((mq, 32, 128), generator=gen, device=device) * 1.4).to(torch.bfloat16).contiguous()
                        cu_q = torch.tensor([0, mq], dtype=torch.int32, device=device)
                        sk = torch.tensor([seq_k], dtype=torch.int32, device=device)
                        for win in ((-1, -1), (511, 0), (2047, 0)):
                            ref = torch.empty_like(q)
                            got = torch.empty_like(q)
                            flash_attn_varlen_func(
                                q=q, k=k_cache, v=v_cache, out=ref,
                                cu_seqlens_q=cu_q, max_seqlen_q=mq,
                                seqused_k=sk, max_seqlen_k=seq_k,
                                softmax_scale=scale, causal=True,
                                window_size=list(win), block_table=bt,
                                fa_version=2, num_splits=0,
                            )
                            rc = _lib.muse_gqa16_prefill(
                                0, cand_mode,
                                q.data_ptr(), kv_cache.data_ptr(), got.data_ptr(),
                                0, 0,
                                sk.data_ptr(), bt.data_ptr(),
                                mq, block_size, win[0], scale,
                                q.stride(0), q.stride(1), got.stride(0), got.stride(1),
                                kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1),
                                stream,
                            )
                            torch.cuda.synchronize(device)
                            if rc != 0 or not torch.isfinite(got).all() or not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
                                mode_ok = False
                                break
                            if has_ng and win == (-1, -1) and block_size == 16 and mq in (64, 512):
                                gate = (torch.randn((mq, 4096), generator=gen, device=device) * 2.2).to(torch.bfloat16)
                                fp8_sc = torch.tensor([0.02345], dtype=torch.float32, device=device)
                                ref_b = torch.empty((mq, 4096), dtype=torch.bfloat16, device=device)
                                got_b = torch.empty_like(ref_b)
                                _ng.mul_sigmoid_bf16(gate, ref.view(mq, 4096), ref_b, mq * 4096, stream)
                                rc1 = _lib.muse_gqa16_prefill(
                                    1, cand_mode,
                                    q.data_ptr(), kv_cache.data_ptr(), got_b.data_ptr(),
                                    gate.data_ptr(), 0,
                                    sk.data_ptr(), bt.data_ptr(),
                                    mq, block_size, win[0], scale,
                                    q.stride(0), q.stride(1), 0, 0,
                                    kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1),
                                    stream,
                                )
                                ref_f = torch.empty((mq, 4096), dtype=torch.float8_e4m3fn, device=device)
                                got_f = torch.empty_like(ref_f)
                                _ng.mul_sigmoid_fp8(gate, ref.view(mq, 4096), fp8_sc, ref_f, mq * 4096, stream)
                                rc2 = _lib.muse_gqa16_prefill(
                                    2, cand_mode,
                                    q.data_ptr(), kv_cache.data_ptr(), got_f.data_ptr(),
                                    gate.data_ptr(), fp8_sc.data_ptr(),
                                    sk.data_ptr(), bt.data_ptr(),
                                    mq, block_size, win[0], scale,
                                    q.stride(0), q.stride(1), 0, 0,
                                    kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1),
                                    stream,
                                )
                                torch.cuda.synchronize(device)
                                if (
                                    rc1 != 0
                                    or rc2 != 0
                                    or not torch.equal(ref_b.view(torch.int16), got_b.view(torch.int16))
                                    or not torch.equal(ref_f.view(torch.uint8), got_f.view(torch.uint8))
                                ):
                                    mode_ok = False
                                    break
                        if not mode_ok:
                            break
                    if not mode_ok:
                        break
                if mode_ok:
                    scale_mode = cand_mode
                    ok = True
                    break
    except Exception:  # noqa: BLE001
        logger.warning("muse_gqa16_attn: prefill self-check raised on cuda:%d", idx, exc_info=True)
        ok = False

    _prefill_state[idx] = (ok, scale_mode)
    if ok:
        logger.info(
            "muse_gqa16_attn: prefill 48KB 2-CTA/SM strict 0-ULP admitted on cuda:%d (scale_mode=%d, fuse_sigmoid=%d)",
            idx, scale_mode, int(PREFILL_SIGMOID),
        )
    else:
        logger.warning("muse_gqa16_attn: prefill 0-ULP self-check failed on cuda:%d; keeping stock prefill FlashAttention-2", idx)
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


def try_prefill_gqa16(
    layer: Any,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    seqused_k: torch.Tensor | None,
    block_table: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
    window: tuple[int, int] | None,
) -> bool:
    global _pending_prefill, _probe_prefill
    if _pending_prefill is not None:
        _pending_prefill = None
        raise RuntimeError("muse_gqa16_attn: deferred prefill attention was never consumed by mul_sigmoid")
    if _probe_prefill is not None:
        prev_layer = _probe_prefill[0]
        _probe_prefill = None
        if prev_layer is not None:
            prev_layer._gqa16_prefill_gate_state = "rejected"

    if not PREFILL_ENABLED or _lib is None or not _has_prefill_sym:
        return False
    device = query.device
    idx = device.index if device.index is not None else torch.cuda.current_device()
    st = _prefill_state.get(idx)
    if st is None:
        if not self_check_prefill(device):
            return False
        st = _prefill_state[idx]
    elif not st[0]:
        return False

    mq = int(query.shape[0])
    if (
        mq < 64
        or seqused_k is None
        or block_table is None
        or query.dtype != torch.bfloat16
        or kv_cache.dtype != torch.bfloat16
        or output.dtype != torch.bfloat16
        or query.ndim != 3
        or output.ndim != 3
        or query.shape != (mq, 32, 128)
        or output.shape != (mq, 32, 128)
        or kv_cache.ndim != 4
        or kv_cache.shape[1] != 2
        or kv_cache.shape[3] != 256
        or kv_cache.stride(3) != 1
        or query.stride(2) != 1
        or output.stride(2) != 1
        or (query.stride(0) & 7) != 0
        or (query.stride(1) & 7) != 0
        or (output.stride(0) & 7) != 0
        or (output.stride(1) & 7) != 0
        or cu_seqlens_q.ndim != 1
        or cu_seqlens_q.shape[0] != 2
        or seqused_k.dtype != torch.int32
        or seqused_k.numel() < 1
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

    scale_mode = st[1]
    stream = torch.cuda.current_stream(device).cuda_stream

    can_defer_sigmoid = False
    if (
        PREFILL_SIGMOID
        and _sigmoid_hook_installed
        and layer is not None
        and getattr(layer, "_gqa16_has_output_gate", False)
    ):
        try:
            from vllm.config.compilation import CUDAGraphMode
            from vllm.forward_context import get_forward_context

            can_defer_sigmoid = get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE
        except Exception:  # noqa: BLE001
            can_defer_sigmoid = True

    if can_defer_sigmoid:
        gate_state = getattr(layer, "_gqa16_prefill_gate_state", "probing")
        if gate_state == "admitted":
            # Hold tensor references in _pending_prefill so the caching allocator cannot reuse query's storage
            # during output_gate_proj before consume_prefill_sigmoid launches the fused kernel.
            _pending_prefill = (
                layer, query, kv_cache, output.data_ptr(), seqused_k, block_table,
                mq, block_size, w_left, scale_mode, float(softmax_scale), stream,
            )
            return True
        if gate_state == "probing" and not torch.cuda.is_current_stream_capturing():
            rc = _lib.muse_gqa16_prefill(
                0, scale_mode,
                query.data_ptr(), kv_cache.data_ptr(), output.data_ptr(),
                0, 0,
                seqused_k.data_ptr(), block_table.data_ptr(),
                mq, block_size, w_left, float(softmax_scale),
                query.stride(0), query.stride(1), output.stride(0), output.stride(1),
                kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1),
                stream,
            )
            if rc != 0:
                return False
            _probe_prefill = (
                layer, query, kv_cache, output.data_ptr(), seqused_k, block_table,
                mq, block_size, w_left, scale_mode, float(softmax_scale), stream,
            )
            return True

    rc = _lib.muse_gqa16_prefill(
        0, scale_mode,
        query.data_ptr(), kv_cache.data_ptr(), output.data_ptr(),
        0, 0,
        seqused_k.data_ptr(), block_table.data_ptr(),
        mq, block_size, w_left, float(softmax_scale),
        query.stride(0), query.stride(1), output.stride(0), output.stride(1),
        kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1),
        stream,
    )
    return rc == 0


def _ranges_disjoint(ptr_a: int, bytes_a: int, ptr_b: int, bytes_b: int) -> bool:
    return (ptr_a + bytes_a <= ptr_b) or (ptr_b + bytes_b <= ptr_a)


def _consume_prefill_sigmoid(kind: int, autotuner: Any, prev_run: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    global _pending_prefill, _probe_prefill
    stream = kwargs.get("stream")
    if kind == 1:
        if len(args) != 4:
            return False
        gate, attn, out, numel = args
        scale = None
    else:
        if len(args) != 5:
            return False
        gate, attn, scale, out, numel = args

    try:
        n_el = int(numel)
    except Exception:  # noqa: BLE001
        return False

    if _pending_prefill is not None:
        rec = _pending_prefill
        _pending_prefill = None
        (
            layer, query, kv_cache, exp_out_ptr, seqused_k, block_table,
            mq, block_size, w_left, scale_mode, softmax_scale, exp_stream,
        ) = rec
        if (
            stream is None
            or stream != exp_stream
            or n_el != mq * 4096
            or not isinstance(gate, torch.Tensor)
            or not isinstance(attn, torch.Tensor)
            or not isinstance(out, torch.Tensor)
            or attn.data_ptr() != exp_out_ptr
            or not gate.is_cuda or not gate.is_contiguous() or gate.dtype != torch.bfloat16 or gate.numel() != n_el
            or not attn.is_cuda or not attn.is_contiguous() or attn.dtype != torch.bfloat16 or attn.numel() != n_el
            or not out.is_cuda or not out.is_contiguous() or out.numel() != n_el
            or (gate.data_ptr() & 15) != 0
            or (kind == 1 and (out.dtype != torch.bfloat16 or (out.data_ptr() & 15) != 0))
            or (
                kind == 2
                and (
                    scale is None
                    or not isinstance(scale, torch.Tensor)
                    or not scale.is_cuda
                    or not scale.is_contiguous()
                    or scale.dtype != torch.float32
                    or scale.numel() != 1
                    or out.dtype != torch.float8_e4m3fn
                    or (out.data_ptr() & 7) != 0
                )
            )
            or not _ranges_disjoint(query.data_ptr(), mq * 4096 * 2, out.data_ptr(), n_el * out.element_size())
        ):
            raise RuntimeError("muse_gqa16_attn: admitted prefill sigmoid arguments diverged from probed shape")
        rc = _lib.muse_gqa16_prefill(
            kind, scale_mode,
            query.data_ptr(), kv_cache.data_ptr(), out.data_ptr(),
            gate.data_ptr(), 0 if scale is None else scale.data_ptr(),
            seqused_k.data_ptr(), block_table.data_ptr(),
            mq, block_size, w_left, softmax_scale,
            query.stride(0), query.stride(1), 0, 0,
            kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1),
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"muse_gqa16_attn: muse_gqa16_prefill(mode={kind}) failed with rc={rc}")
        if kind == 1:
            _prefill_fuse_counts["fused_bf16"] += 1
        else:
            _prefill_fuse_counts["fused_fp8"] += 1
        del rec, query
        return True

    if _probe_prefill is None:
        return False

    rec = _probe_prefill
    _probe_prefill = None
    (
        layer, query, kv_cache, exp_out_ptr, seqused_k, block_table,
        mq, block_size, w_left, scale_mode, softmax_scale, exp_stream,
    ) = rec
    if (
        stream is None
        or stream != exp_stream
        or torch.cuda.is_current_stream_capturing()
        or n_el != mq * 4096
        or not isinstance(gate, torch.Tensor)
        or not isinstance(attn, torch.Tensor)
        or not isinstance(out, torch.Tensor)
        or attn.data_ptr() != exp_out_ptr
        or not gate.is_cuda or not gate.is_contiguous() or gate.dtype != torch.bfloat16 or gate.numel() != n_el
        or not attn.is_cuda or not attn.is_contiguous() or attn.dtype != torch.bfloat16 or attn.numel() != n_el
        or not out.is_cuda or not out.is_contiguous() or out.numel() != n_el
        or (gate.data_ptr() & 15) != 0
        or (kind == 1 and (out.dtype != torch.bfloat16 or (out.data_ptr() & 15) != 0))
        or (
            kind == 2
            and (
                scale is None
                or not isinstance(scale, torch.Tensor)
                or not scale.is_cuda
                or not scale.is_contiguous()
                or scale.dtype != torch.float32
                or scale.numel() != 1
                or out.dtype != torch.float8_e4m3fn
                or (out.data_ptr() & 7) != 0
            )
        )
    ):
        layer._gqa16_prefill_gate_state = "rejected"
        _prefill_fuse_counts["rejected"] += 1
        return False

    # Probing already wrote `attn` via epilogue_mode=0. Run stock/norm_glue into `out`, then verify 0-ULP bitwise
    # on BOTH the live tensors AND non-trivial random synthetic KV/Q/gate tensors (covering zeroed-KV warmup runs).
    gate_saved = gate.clone()
    scale_saved = scale.clone() if scale is not None else None
    prev_run(autotuner, *args, **kwargs)
    ok = False
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

        got_live = torch.empty_like(out)
        rc_live = _lib.muse_gqa16_prefill(
            kind, scale_mode,
            query.data_ptr(), kv_cache.data_ptr(), got_live.data_ptr(),
            gate_saved.data_ptr(), 0 if scale_saved is None else scale_saved.data_ptr(),
            seqused_k.data_ptr(), block_table.data_ptr(),
            mq, block_size, w_left, softmax_scale,
            query.stride(0), query.stride(1), 0, 0,
            kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1),
            stream,
        )
        torch.cuda.synchronize(out.device)
        if kind == 1:
            ok = rc_live == 0 and torch.equal(out.view(torch.int16), got_live.view(torch.int16))
        else:
            ok = rc_live == 0 and torch.equal(out.view(torch.uint8), got_live.view(torch.uint8))

        if ok:
            dev = out.device
            gen = torch.Generator(device=dev).manual_seed(20261130 + int(mq))
            sk_val = max(mq, 512) if mq <= 512 else mq
            nb_syn = (sk_val + block_size - 1) // block_size + 8
            kv_syn = (torch.randn((nb_syn, 2, block_size, 256), generator=gen, device=dev) * 0.7).to(torch.bfloat16)
            k_syn, v_syn = kv_syn.transpose(1, 2).split(128, dim=-1)
            bt_syn = torch.arange(1, nb_syn, dtype=torch.int32, device=dev).unsqueeze(0).contiguous()
            cu_syn = torch.tensor([0, mq], dtype=torch.int32, device=dev)
            sk_syn = torch.tensor([sk_val], dtype=torch.int32, device=dev)
            win_list = [w_left, 0] if w_left >= 0 else [-1, -1]
            for sig in (0.2, 1.5, 8.0):
                q_syn = (torch.randn((mq, 32, 128), generator=gen, device=dev) * sig).to(torch.bfloat16)
                g_syn = (torch.randn((mq, 4096), generator=gen, device=dev) * sig).to(torch.bfloat16)
                a_ref = torch.empty_like(q_syn)
                flash_attn_varlen_func(
                    q=q_syn, k=k_syn, v=v_syn, out=a_ref,
                    cu_seqlens_q=cu_syn, max_seqlen_q=mq,
                    seqused_k=sk_syn, max_seqlen_k=sk_val,
                    softmax_scale=softmax_scale, causal=True,
                    window_size=win_list, block_table=bt_syn,
                    fa_version=2, num_splits=0,
                )
                o_ref = torch.empty_like(out)
                o_got = torch.empty_like(out)
                if kind == 1:
                    prev_run(autotuner, g_syn, a_ref.view(mq, 4096), o_ref, n_el, **kwargs)
                else:
                    assert scale_saved is not None
                    prev_run(autotuner, g_syn, a_ref.view(mq, 4096), scale_saved, o_ref, n_el, **kwargs)
                rc_s = _lib.muse_gqa16_prefill(
                    kind, scale_mode,
                    q_syn.data_ptr(), kv_syn.data_ptr(), o_got.data_ptr(),
                    g_syn.data_ptr(), 0 if scale_saved is None else scale_saved.data_ptr(),
                    sk_syn.data_ptr(), bt_syn.data_ptr(),
                    mq, block_size, w_left, softmax_scale,
                    q_syn.stride(0), q_syn.stride(1), 0, 0,
                    kv_syn.stride(0), kv_syn.stride(2), kv_syn.stride(1),
                    stream,
                )
                torch.cuda.synchronize(dev)
                same = (
                    torch.equal(o_ref.view(torch.int16), o_got.view(torch.int16))
                    if kind == 1
                    else torch.equal(o_ref.view(torch.uint8), o_got.view(torch.uint8))
                )
                if rc_s != 0 or not same:
                    ok = False
                    break
    except Exception:  # noqa: BLE001
        logger.warning("muse_gqa16_attn: prefill sigmoid probe failed", exc_info=True)
        ok = False

    if ok:
        layer._gqa16_prefill_gate_state = "admitted"
        _prefill_fuse_counts["admitted"] += 1
        if _prefill_fuse_counts["admitted"] in (1, 52):
            logger.info(
                "muse_gqa16_attn: prefill fused mul_sigmoid epilogue admitted bitwise on %d layers",
                _prefill_fuse_counts["admitted"],
            )
    else:
        layer._gqa16_prefill_gate_state = "rejected"
        _prefill_fuse_counts["rejected"] += 1
    return True


def install_prefill_sigmoid_hook() -> None:
    """Install CachingAutotuner.run consumer for prefill attention + mul_sigmoid_{bf16,fp8} epilogue fusion."""
    global _sigmoid_hook_installed
    if _sigmoid_hook_installed or not PREFILL_ENABLED or not PREFILL_SIGMOID or _lib is None or not _has_prefill_sym:
        return
    try:
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner, get_active_debug_mode
    except Exception:  # noqa: BLE001
        return

    prev_run = CachingAutotuner.run

    def run(autotuner, *args, **kwargs):
        if (
            (_pending_prefill is not None or _probe_prefill is not None)
            and not kwargs.get("benchmark_run", False)
            and not getattr(autotuner, "triton_interpret", False)
            and not get_active_debug_mode()
        ):
            kind = autotuner.__dict__.get("_gqa16_pf_sig", 0)
            if kind == 0:
                src = getattr(getattr(autotuner, "fn", None), "src", "")
                h = _canonical_triton_hash(src) if isinstance(src, str) else ""
                if h == _MUL_SIGMOID_BF16_HASH:
                    kind = 1
                elif h == _SIGMOID_FP8_HASH:
                    kind = 2
                else:
                    kind = None
                autotuner.__dict__["_gqa16_pf_sig"] = kind
            if kind in (1, 2):
                if _consume_prefill_sigmoid(kind, autotuner, prev_run, args, kwargs):
                    return None
        return prev_run(autotuner, *args, **kwargs)

    CachingAutotuner.run = run
    _sigmoid_hook_installed = True
