# SPDX-License-Identifier: Apache-2.0
"""Strict 0-ULP fused M=512 prefill QK-norm/RoPE/KV-cache + 48-SM balanced causal GQA-16 attention.

Orthogonal standalone package (zero overlap with harness/kernels/gqa16_decode).
Kill switch: MUSE_PREFILL_ATTN_FUSED=0 disables this path and falls back to stock qk_glue + gqa16_prefill.
"""
from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

import torch

try:
    from vllm.logger import init_logger
    logger = init_logger("vllm.muse_prefill_attn")
except Exception:  # noqa: BLE001
    logger = logging.getLogger("vllm.muse_prefill_attn")

ENABLED = os.environ.get("MUSE_PREFILL_ATTN_FUSED", "1").strip() not in ("0", "", "false", "no")
PDL_ENABLED = os.environ.get("MUSE_PREFILL_ATTN_PDL", "1").strip() not in ("0", "", "false", "no")
DISCARD_L2 = os.environ.get("MUSE_PREFILL_ATTN_DISCARD_L2", "1").strip() not in ("0", "", "false", "no")
POST_ATTN_ENABLED = os.environ.get("MUSE_PREFILL_POST_ATTN_FAST", "1").strip() not in ("0", "", "false", "no")
POST_ATTN_CS_MODE = int(os.environ.get("MUSE_PREFILL_POST_ATTN_CS_MODE", "1"))

_lib = None
try:
    _lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_prefill_attn.so")))
    _p = ctypes.c_void_p
    _lib.muse_prefill_kv_only.argtypes = [
        _p, _p, _p, _p, _p, _p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, _p,
    ]
    _lib.muse_prefill_kv_only.restype = ctypes.c_int
    _lib.muse_prefill_kv_only_ex.argtypes = [
        _p, _p, _p, _p, _p, _p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        _p, ctypes.c_size_t, _p, ctypes.c_size_t, _p,
    ]
    _lib.muse_prefill_kv_only_ex.restype = ctypes.c_int
    _lib.muse_prefill_attn_fused_512.argtypes = [
        _p, _p, _p, _p, _p, _p, _p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, _p,
    ]
    _lib.muse_prefill_attn_fused_512.restype = ctypes.c_int
    _lib.muse_prefill_post_attn_norm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        _p, _p, _p, _p, _p, _p, _p, _p,
        _p, ctypes.c_size_t, _p,
    ]
    _lib.muse_prefill_post_attn_norm.restype = ctypes.c_int
    _lib.muse_prefill_discard_l2.argtypes = [_p, ctypes.c_size_t, _p]
    _lib.muse_prefill_discard_l2.restype = ctypes.c_int
    _lib.muse_prefill_discard_l2_multi2.argtypes = [_p, ctypes.c_size_t, _p, ctypes.c_size_t, _p]
    _lib.muse_prefill_discard_l2_multi2.restype = ctypes.c_int
except OSError:
    _lib = None

_state: dict[int, tuple[bool, int]] = {}  # dev_idx -> (admitted, scale_mode)
_scratch: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}  # dev_idx -> (kv_flat, q_stub, k_stub)

# Pending fused prefill QK state (set by launch_kv_only, consumed by try_prefill -> consume_mul_sigmoid)
_stashed_qk: tuple | None = None
_pending_attn: tuple | None = None
_last_attn_out: torch.Tensor | None = None
_counts = {"kv_only": 0, "fused0": 0, "fused1": 0, "fused2": 0, "fallback_q": 0, "post_attn0": 0, "post_attn1": 0}


def _get_scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    s = _scratch.get(idx)
    if s is None:
        kv_flat = torch.empty((2, 512, 128), dtype=torch.bfloat16, device=device)
        q_stub = torch.empty((512, 4096), dtype=torch.bfloat16, device=device)
        k_stub = torch.empty((512, 256), dtype=torch.bfloat16, device=device)
        s = (kv_flat, q_stub, k_stub)
        _scratch[idx] = s
    return s


def discard_l2(t: torch.Tensor, stream: int | None = None) -> bool:
    if _lib is None or not t.is_cuda or not t.is_contiguous():
        return False
    nbytes = t.numel() * t.element_size()
    if nbytes < 128:
        return False
    st = torch.cuda.current_stream(t.device).cuda_stream if stream is None else int(stream)
    return _lib.muse_prefill_discard_l2(t.data_ptr(), nbytes, st) == 0


def self_check(device: torch.device) -> bool:
    """Strict 0-ULP bitwise check against muse_qk_glue + flash_attn_varlen_func + muse_norm_glue."""
    if not ENABLED or _lib is None or device.type != "cuda":
        return False
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _state.get(idx)
    if cached is not None:
        return cached[0]
    if torch.cuda.is_current_stream_capturing():
        return False

    props = torch.cuda.get_device_properties(device)
    if props.multi_processor_count != 48 or torch.cuda.get_device_capability(device) != (12, 1):
        _state[idx] = (False, 1)
        return False

    ok = False
    scale_mode = 1
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
        import muse_qk_glue as QK
        import muse_norm_glue as NG

        if not QK.available(device) or NG.lib() is None:
            _state[idx] = (False, 1)
            return False

        gen = torch.Generator(device=device).manual_seed(20261126)
        scale = float(128.0 ** -0.5)
        eps, qscale = 1e-5, 3.869140625
        pos = torch.arange(512, dtype=torch.int64, device=device)
        t_idx = torch.arange(1024, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, device=device, dtype=torch.float32) / 128.0))
        freqs = torch.einsum("i,j->ij", t_idx, inv_freq)
        cache_f32 = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()
        cache_bf16 = cache_f32.to(torch.bfloat16).contiguous()
        fp8_scale = torch.tensor([0.045], dtype=torch.float32, device=device)
        cu_512 = torch.tensor([0, 512], dtype=torch.int32, device=device)
        sk_512 = torch.tensor([512], dtype=torch.int32, device=device)
        kv_flat, _, _ = _get_scratch(device)
        stream = torch.cuda.current_stream(device).cuda_stream

        for cand_mode in (1, 3):
            all_match = True
            for block_size in (16, 64):
                num_blocks = (512 // block_size) + 8
                perm = torch.randperm(num_blocks - 1, generator=gen, device=device, dtype=torch.int32) + 1
                bt = perm[: (512 // block_size)].unsqueeze(0).contiguous()
                slots = (bt[0].to(torch.int64)[:, None] * block_size + torch.arange(block_size, dtype=torch.int64, device=device)[None, :]).reshape(-1)
                for use_rope, cache_t in ((True, cache_f32), (True, cache_bf16), (False, cache_f32)):
                    qkv_ref = (torch.randn((512, 4608), generator=gen, device=device) * 1.4).to(torch.bfloat16).contiguous()
                    qkv_got = qkv_ref.clone()
                    gate_ref = (torch.randn((512, 4096), generator=gen, device=device) * 1.3).to(torch.bfloat16).contiguous()
                    gate_got = gate_ref.clone()

                    raw_kv_ref = torch.zeros((num_blocks, block_size, 2, 256), dtype=torch.bfloat16, device=device)
                    raw_kv_got = torch.zeros_like(raw_kv_ref)
                    kvc_ref = raw_kv_ref.permute(0, 2, 1, 3)
                    kvc_got = raw_kv_got.permute(0, 2, 1, 3)

                    # Reference: muse_qk_glue -> FA2 -> mul_sigmoid / mul_sigmoid_fp8
                    q_ref, _, wrote_ref = QK.run_qk_norm_rope_kvcache_prefill(
                        qkv_ref, pos, cache_t, eps, qscale, use_rope, kvc_ref, slots, q_first_row=0
                    )
                    if not wrote_ref:
                        all_match = False
                        break
                    k_cache, v_cache = kvc_ref.transpose(1, 2).split(128, dim=-1)
                    ref0 = torch.empty((512, 32, 128), dtype=torch.bfloat16, device=device)
                    flash_attn_varlen_func(
                        q=q_ref.view(512, 32, 128), k=k_cache, v=v_cache, out=ref0,
                        cu_seqlens_q=cu_512, max_seqlen_q=512, seqused_k=sk_512, max_seqlen_k=512,
                        softmax_scale=scale, causal=True, window_size=[-1, -1], block_table=bt,
                        fa_version=2, num_splits=0,
                    )
                    ref1 = torch.empty((512, 4096), dtype=torch.bfloat16, device=device)
                    NG.mul_sigmoid_bf16(gate_ref, ref0.view(512, 4096), ref1, 512 * 4096, stream)
                    ref2 = torch.empty((512, 4096), dtype=torch.float8_e4m3fn, device=device)
                    NG.mul_sigmoid_fp8(gate_ref, ref0.view(512, 4096), fp8_scale, ref2, 512 * 4096, stream)

                    # Candidate: kv_only -> fused_512 (modes 0, 1, 2)
                    cf32 = int(cache_t.dtype == torch.float32)
                    c_rows = int(cache_t.shape[0]) if use_rope else 0
                    p_ptr = pos.data_ptr() if use_rope else 0
                    c_ptr = cache_t.data_ptr() if use_rope else 0
                    rc_kv = _lib.muse_prefill_kv_only(
                        qkv_got.data_ptr(), p_ptr, c_ptr,
                        kvc_got.data_ptr(), kv_flat.data_ptr(), slots.data_ptr(),
                        512, c_rows, block_size,
                        kvc_got.stride(0), kvc_got.stride(2), kvc_got.stride(1),
                        eps, int(use_rope), cf32, int(PDL_ENABLED), 0, stream,
                    )
                    got0 = torch.empty((512, 4096), dtype=torch.bfloat16, device=device)
                    rc0 = _lib.muse_prefill_attn_fused_512(
                        qkv_got.data_ptr(), p_ptr, c_ptr, kv_flat.data_ptr(), 0, 0, got0.data_ptr(),
                        c_rows, 0, cand_mode, int(use_rope), cf32, int(PDL_ENABLED), 0,
                        eps, qscale, scale, stream,
                    )
                    got1 = torch.empty((512, 4096), dtype=torch.bfloat16, device=device)
                    rc1 = _lib.muse_prefill_attn_fused_512(
                        qkv_got.data_ptr(), p_ptr, c_ptr, kv_flat.data_ptr(), gate_got.data_ptr(), 0, got1.data_ptr(),
                        c_rows, 1, cand_mode, int(use_rope), cf32, int(PDL_ENABLED), 0,
                        eps, qscale, scale, stream,
                    )
                    got2 = torch.empty((512, 4096), dtype=torch.float8_e4m3fn, device=device)
                    rc2 = _lib.muse_prefill_attn_fused_512(
                        qkv_got.data_ptr(), p_ptr, c_ptr, kv_flat.data_ptr(), gate_got.data_ptr(), fp8_scale.data_ptr(), got2.data_ptr(),
                        c_rows, 2, cand_mode, int(use_rope), cf32, int(PDL_ENABLED), int(DISCARD_L2),
                        eps, qscale, scale, stream,
                    )
                    torch.cuda.synchronize(device)
                    if (
                        rc_kv != 0 or rc0 != 0 or rc1 != 0 or rc2 != 0
                        or not torch.equal(raw_kv_ref.view(torch.int16), raw_kv_got.view(torch.int16))
                        or not torch.equal(ref0.view(512, 4096).view(torch.int16), got0.view(torch.int16))
                        or not torch.equal(ref1.view(torch.int16), got1.view(torch.int16))
                        or not torch.equal(ref2.view(torch.uint8), got2.view(torch.uint8))
                    ):
                        all_match = False
                        break
                if not all_match:
                    break
            if all_match:
                scale_mode = cand_mode
                ok = True
                break

        if ok and POST_ATTN_ENABLED:
            # Strict 0-ULP check for fused M=512 prefill post_attn_norm (kind=0 BF16, kind=1 FP8)
            x_pa = (torch.randn((512, 6656), generator=gen, device=device) * 1.3).to(torch.bfloat16).contiguous()
            res_pa = (torch.randn((512, 6656), generator=gen, device=device) * 1.1).to(torch.bfloat16).contiguous()
            wp_pa = (torch.randn((6656,), generator=gen, device=device) * 0.2).to(torch.bfloat16).contiguous()
            wq_pa = (torch.randn((6656,), generator=gen, device=device) * 0.2).to(torch.bfloat16).contiguous()
            sc_pa = torch.tensor([0.055], dtype=torch.float32, device=device)
            s_ref0 = torch.empty((512,), dtype=torch.float32, device=device)
            o_ref0 = torch.empty((512, 6656), dtype=torch.bfloat16, device=device)
            s_got0 = torch.empty_like(s_ref0)
            o_got0 = torch.empty_like(o_ref0)
            NG.post_attn(0, 512, x_pa, res_pa, wp_pa, wq_pa, None, s_ref0, o_ref0, None, stream)
            rc_p0 = _lib.muse_prefill_post_attn_norm(
                0, 512, int(POST_ATTN_CS_MODE),
                x_pa.data_ptr(), res_pa.data_ptr(), wp_pa.data_ptr(), wq_pa.data_ptr(),
                0, s_got0.data_ptr(), o_got0.data_ptr(), 0, 0, 0, stream,
            )
            s_ref1 = torch.empty((512,), dtype=torch.float32, device=device)
            q_ref1 = torch.empty((512, 6656), dtype=torch.float8_e4m3fn, device=device)
            s_got1 = torch.empty_like(s_ref1)
            q_got1 = torch.empty_like(q_ref1)
            NG.post_attn(1, 512, x_pa, res_pa, wp_pa, wq_pa, sc_pa, s_ref1, None, q_ref1, stream)
            rc_p1 = _lib.muse_prefill_post_attn_norm(
                1, 512, int(POST_ATTN_CS_MODE),
                x_pa.data_ptr(), res_pa.data_ptr(), wp_pa.data_ptr(), wq_pa.data_ptr(),
                sc_pa.data_ptr(), s_got1.data_ptr(), 0, q_got1.data_ptr(), 0, 0, stream,
            )
            torch.cuda.synchronize(device)
            if (
                rc_p0 != 0 or rc_p1 != 0
                or not torch.equal(s_ref0.view(torch.int32), s_got0.view(torch.int32))
                or not torch.equal(o_ref0.view(torch.int16), o_got0.view(torch.int16))
                or not torch.equal(s_ref1.view(torch.int32), s_got1.view(torch.int32))
                or not torch.equal(q_ref1.view(torch.uint8), q_got1.view(torch.uint8))
            ):
                ok = False
    except Exception:  # noqa: BLE001
        logger.warning("muse_prefill_attn: strict 0-ULP self-check raised on cuda:%d", idx, exc_info=True)
        ok = False

    _state[idx] = (ok, scale_mode)
    if ok:
        logger.info(
            "muse_prefill_attn: strict 0-ULP admitted on cuda:%d (pdl=%d, discard_l2=%d, scale_mode=%d)",
            idx, int(PDL_ENABLED), int(DISCARD_L2), scale_mode,
        )
    else:
        logger.warning("muse_prefill_attn: strict 0-ULP self-check failed on cuda:%d", idx)
    return ok


def run_kv_only_and_stash(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    eps: float,
    qscale: float,
    use_rope: bool,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    q_first_row: int = 0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """For fresh M=512 prefill (q_first_row==0), run tiny KV-only kernel and stash raw Q for fused attn."""
    global _stashed_qk, _last_attn_out
    _last_attn_out = None
    if not ENABLED or _lib is None or q_first_row != 0:
        return None
    if (
        qkv.ndim != 2
        or qkv.shape != (512, 4608)
        or not qkv.is_cuda
        or not qkv.is_contiguous()
        or qkv.dtype != torch.bfloat16
        or (qkv.data_ptr() & 127) != 0
        or kv_cache.device != qkv.device
        or kv_cache.dtype != torch.bfloat16
        or kv_cache.ndim != 4
        or kv_cache.shape[1] != 2
        or kv_cache.shape[2] not in (16, 64)
        or kv_cache.shape[3] != 256
        or kv_cache.stride(3) != 1
        or slot_mapping.device != qkv.device
        or slot_mapping.ndim != 1
        or slot_mapping.dtype != torch.int64
        or not slot_mapping.is_contiguous()
        or slot_mapping.numel() < 512
    ):
        return None

    device = qkv.device
    idx = device.index if device.index is not None else torch.cuda.current_device()
    st = _state.get(idx)
    if st is None:
        if torch.cuda.is_current_stream_capturing() or not self_check(device):
            return None
        st = _state[idx]
    elif not st[0]:
        return None

    if _stashed_qk is not None:
        _stashed_qk = None

    if use_rope:
        if (
            cos_sin_cache is None
            or cos_sin_cache.device != device
            or cos_sin_cache.ndim != 2
            or cos_sin_cache.shape[0] <= 0
            or cos_sin_cache.shape[1] != 128
            or cos_sin_cache.dtype not in (torch.bfloat16, torch.float32)
            or positions.device != device
            or positions.numel() < 512
        ):
            return None
        if positions.dtype != torch.int64:
            positions = positions.to(torch.int64)
        if not positions.is_contiguous():
            positions = positions.contiguous()
        if not cos_sin_cache.is_contiguous():
            cos_sin_cache = cos_sin_cache.contiguous()
        pos_ptr = positions.data_ptr()
        cache_ptr = cos_sin_cache.data_ptr()
        cache_rows = int(cos_sin_cache.shape[0])
        cache_fp32 = int(cos_sin_cache.dtype == torch.float32)
    else:
        pos_ptr = 0
        cache_ptr = 0
        cache_rows = 0
        cache_fp32 = 1

    kv_flat, q_stub, k_stub = _get_scratch(device)
    block_size = int(kv_cache.shape[2])
    stream = torch.cuda.current_stream(device).cuda_stream
    rc = _lib.muse_prefill_kv_only(
        qkv.data_ptr(), pos_ptr, cache_ptr,
        kv_cache.data_ptr(), kv_flat.data_ptr(), slot_mapping.data_ptr(),
        512, cache_rows, block_size,
        kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1),
        float(eps), int(use_rope), cache_fp32, int(PDL_ENABLED), int(DISCARD_L2), stream,
    )
    if rc != 0:
        raise RuntimeError(f"muse_prefill_kv_only failed with CUDA error {rc}")
    _counts["kv_only"] += 1
    _stashed_qk = (
        q_stub.data_ptr(),
        qkv,
        positions,
        cos_sin_cache,
        pos_ptr,
        cache_ptr,
        cache_rows,
        cache_fp32,
        float(eps),
        float(qscale),
        int(use_rope),
        kv_flat,
        st[1],
    )
    return q_stub, k_stub


def _materialize_q_fallback(q_dst: torch.Tensor, stash: tuple, stream: int) -> None:
    """If attention cannot consume our fused stash (e.g. non-standard window/chunk), materialize Q exactly."""
    import muse_qk_glue as QK

    (_, qkv, positions, cos_sin_cache, _, _, _, _, eps, qscale, use_rope, _, _) = stash
    q_ref, _ = QK.run_qk_norm_rope(qkv, positions, cos_sin_cache, eps, qscale, bool(use_rope))
    q_dst.copy_(q_ref.view_as(q_dst))
    _counts["fallback_q"] += 1


def try_prefill_fused(
    query: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    orig_max_seqlen_q: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    window: tuple[int, int] | None,
    allow_defer: bool = True,
    eligible: bool = True,
) -> bool:
    global _stashed_qk, _pending_attn
    stash = _stashed_qk
    if stash is None:
        return False
    if query.data_ptr() != stash[0]:
        return False
    _stashed_qk = None

    if _pending_attn is not None:
        flush_pending()

    win_ok = window is None or (window[1] <= 0 and (window[0] < 0 or window[0] >= 511))
    stream = torch.cuda.current_stream(query.device).cuda_stream
    if (
        not eligible
        or orig_max_seqlen_q != 512
        or max_seqlen_q != 512
        or max_seqlen_k != 512
        or not win_ok
        or cu_seqlens_q.shape[0] != 2
        or output.shape != (512, 32, 128)
        or not output.is_contiguous()
        or (output.data_ptr() & 15) != 0
    ):
        _materialize_q_fallback(query, stash, stream)
        return False

    can_defer = allow_defer
    if can_defer:
        try:
            from vllm.model_executor.models import muse_glimmer_norm_glue as _ng_mod
            can_defer = bool(getattr(_ng_mod, "_installed", False) and getattr(_ng_mod, "_MUL_SIGMOID_GLUE", False))
        except Exception:
            can_defer = False

    if can_defer:
        _pending_attn = (output.data_ptr(), output.numel(), output, stash, float(softmax_scale), stream)
        return True

    (_, qkv, _pos, _cs, pos_ptr, cache_ptr, cache_rows, cache_fp32, eps, qscale, use_rope, kv_flat, scale_mode) = stash
    rc = _lib.muse_prefill_attn_fused_512(
        qkv.data_ptr(), pos_ptr, cache_ptr, kv_flat.data_ptr(), 0, 0, output.data_ptr(),
        cache_rows, 0, scale_mode, use_rope, cache_fp32, int(PDL_ENABLED), int(DISCARD_L2),
        eps, qscale, float(softmax_scale), stream,
    )
    if rc != 0:
        raise RuntimeError(f"muse_prefill_attn_fused_512 mode=0 failed with CUDA error {rc}")
    _counts["fused0"] += 1
    return True


def flush_pending(stream: int | None = None) -> None:
    global _pending_attn
    pend = _pending_attn
    if pend is None:
        return
    _pending_attn = None
    _out_ptr, _out_numel, output, stash, softmax_scale, saved_stream = pend
    st = saved_stream if stream is None else int(stream)
    (_, qkv, _pos, _cs, pos_ptr, cache_ptr, cache_rows, cache_fp32, eps, qscale, use_rope, kv_flat, scale_mode) = stash
    rc = _lib.muse_prefill_attn_fused_512(
        qkv.data_ptr(), pos_ptr, cache_ptr, kv_flat.data_ptr(), 0, 0, output.data_ptr(),
        cache_rows, 0, scale_mode, use_rope, cache_fp32, int(PDL_ENABLED), int(DISCARD_L2),
        eps, qscale, softmax_scale, st,
    )
    if rc != 0:
        raise RuntimeError(f"muse_prefill_attn_fused_512 flush failed with CUDA error {rc}")
    _counts["fused0"] += 1


def consume_prefill_mul_sigmoid(
    attn: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
    fp8_scale: torch.Tensor | None = None,
    stream: int | None = None,
) -> bool:
    global _pending_attn, _last_attn_out
    pend = _pending_attn
    if pend is None:
        return False
    _pending_attn = None
    _last_attn_out = None
    out_ptr, out_numel, out_arg, stash, softmax_scale, saved_stream = pend
    st = saved_stream if stream is None else int(stream)
    is_fp8 = fp8_scale is not None
    out_align_mask = 7 if is_fp8 else 15
    (_, qkv, _pos, _cs, pos_ptr, cache_ptr, cache_rows, cache_fp32, eps, qscale, use_rope, kv_flat, scale_mode) = stash
    if (
        attn.data_ptr() != out_ptr
        or attn.numel() != out_numel
        or gate.numel() != out_numel
        or out.numel() != out_numel
        or (gate.data_ptr() & 127) != 0
        or (out.data_ptr() & out_align_mask) != 0
        or not gate.is_contiguous()
        or not out.is_contiguous()
    ):
        rc_f = _lib.muse_prefill_attn_fused_512(
            qkv.data_ptr(), pos_ptr, cache_ptr, kv_flat.data_ptr(), 0, 0, out_arg.data_ptr(),
            cache_rows, 0, scale_mode, use_rope, cache_fp32, int(PDL_ENABLED), int(DISCARD_L2),
            eps, qscale, softmax_scale, st,
        )
        if rc_f != 0:
            raise RuntimeError(f"muse_prefill_attn_fused_512 fallback flush failed with CUDA error {rc_f}")
        _counts["fused0"] += 1
        return False
    mode = 2 if is_fp8 else 1
    fp8_ptr = fp8_scale.data_ptr() if is_fp8 else 0
    rc = _lib.muse_prefill_attn_fused_512(
        qkv.data_ptr(), pos_ptr, cache_ptr, kv_flat.data_ptr(), gate.data_ptr(), fp8_ptr, out.data_ptr(),
        cache_rows, mode, scale_mode, use_rope, cache_fp32, int(PDL_ENABLED), int(DISCARD_L2),
        eps, qscale, softmax_scale, st,
    )
    if rc != 0:
        raise RuntimeError(f"muse_prefill_attn_fused_512 mode={mode} failed with CUDA error {rc}")
    _last_attn_out = out
    k = "fused2" if is_fp8 else "fused1"
    _counts[k] += 1
    if _counts["fused1"] + _counts["fused2"] in (1, 51):
        logger.info("muse_prefill_attn: fused counts=%s", _counts)
    return True


def _ranges_disjoint(a_ptr: int, a_bytes: int, tensors: tuple[torch.Tensor | None, ...]) -> bool:
    a_end = a_ptr + a_bytes
    for t in tensors:
        if t is None:
            continue
        t_ptr = t.data_ptr()
        t_end = t_ptr + t.numel() * t.element_size()
        if not (a_end <= t_ptr or t_end <= a_ptr):
            return False
    return True


def post_attn_norm(
    kind: int,
    rows: int,
    x: torch.Tensor,
    res: torch.Tensor,
    wp: torch.Tensor,
    wq: torch.Tensor,
    scale: torch.Tensor | None,
    sumsq: torch.Tensor,
    out_bf16: torch.Tensor | None,
    out_fp8: torch.Tensor | None,
    stream: int,
    for_check: bool = False,
) -> bool:
    """Fast 16B-coalesced M=512 prefill post_attn_norm with safe inline L2 discard of dead attn_out."""
    global _last_attn_out
    if not POST_ATTN_ENABLED or _lib is None or rows != 512:
        if not for_check:
            _last_attn_out = None
        return False
    device = x.device
    idx = device.index if device.index is not None else torch.cuda.current_device()
    st = _state.get(idx)
    if st is None:
        if torch.cuda.is_current_stream_capturing() or not self_check(device):
            if not for_check:
                _last_attn_out = None
            return False
        st = _state[idx]
    if not st[0]:
        if not for_check:
            _last_attn_out = None
        return False
    # Verify 16B alignment for uint4 loads/stores (8B for FP8 output).
    if ((x.data_ptr() | res.data_ptr() | wp.data_ptr() | wq.data_ptr()) & 15) != 0:
        if not for_check:
            _last_attn_out = None
        return False
    if kind == 0:
        if out_bf16 is None or (out_bf16.data_ptr() & 15) != 0:
            if not for_check:
                _last_attn_out = None
            return False
    elif kind == 1:
        if scale is None or out_fp8 is None or (out_fp8.data_ptr() & 7) != 0:
            if not for_check:
                _last_attn_out = None
            return False
    else:
        if not for_check:
            _last_attn_out = None
        return False

    dead_ptr = 0
    dead_bytes = 0
    dead_t = None
    if not for_check:
        dead_t = _last_attn_out
        _last_attn_out = None
        if DISCARD_L2 and dead_t is not None and dead_t.is_cuda and dead_t.device == device and dead_t.is_contiguous():
            dp = dead_t.data_ptr()
            db = dead_t.numel() * dead_t.element_size()
            if db >= 128 and _ranges_disjoint(dp, db, (x, res, wp, wq, scale, sumsq, out_bf16, out_fp8)):
                dead_ptr = dp
                dead_bytes = db

    rc = _lib.muse_prefill_post_attn_norm(
        int(kind),
        int(rows),
        int(POST_ATTN_CS_MODE),
        x.data_ptr(),
        res.data_ptr(),
        wp.data_ptr(),
        wq.data_ptr(),
        scale.data_ptr() if scale is not None else 0,
        sumsq.data_ptr(),
        out_bf16.data_ptr() if out_bf16 is not None else 0,
        out_fp8.data_ptr() if out_fp8 is not None else 0,
        dead_ptr,
        dead_bytes,
        int(stream),
    )
    del dead_t
    if rc != 0:
        raise RuntimeError(f"muse_prefill_post_attn_norm kind={kind} failed with CUDA error {rc}")
    _counts["post_attn0" if kind == 0 else "post_attn1"] += 1
    return True
