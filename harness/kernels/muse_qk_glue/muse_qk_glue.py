"""One-pass fused Q/K RMSNorm + query pre-scale + NEOX RoPE for MuseGlimmerAttention.

Kill switch: ``VLLM_MUSE_GLIMMER_QK_FUSED=0`` restores the stock PyTorch/Inductor path.
"""
from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

import torch

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except Exception:  # noqa: BLE001
    logger = logging.getLogger(__name__)

_lib = None
_multi = None
_kv_m1 = None
_kv_prefill = None
try:
    _lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_qk_glue.so")))
    _lib.muse_qk_norm_rope.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    _lib.muse_qk_norm_rope.restype = ctypes.c_int
    _multi = getattr(_lib, "muse_qk_norm_rope_multi", None)
    if _multi is not None:
        _multi.argtypes = list(_lib.muse_qk_norm_rope.argtypes[:-1]) + [ctypes.c_int, ctypes.c_void_p]
        _multi.restype = ctypes.c_int
    _kv_m1 = getattr(_lib, "muse_qk_norm_rope_kvcache_m1", None)
    if _kv_m1 is not None:
        _kv_m1.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        _kv_m1.restype = ctypes.c_int
    _kv_prefill = getattr(_lib, "muse_qk_norm_rope_kvcache_prefill", None)
    if _kv_prefill is not None:
        _kv_prefill.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        _kv_prefill.restype = ctypes.c_int
    _set_pdl_fn = getattr(_lib, "muse_qk_glue_set_pdl", None)
    if _set_pdl_fn is not None:
        _set_pdl_fn.argtypes = [ctypes.c_int]
        _set_pdl_fn.restype = ctypes.c_int
except OSError:
    _lib = None
    _multi = None
    _kv_m1 = None
    _kv_prefill = None
    _set_pdl_fn = None


def set_pdl(on: bool) -> bool | None:
    if _set_pdl_fn is None:
        return None
    return bool(_set_pdl_fn(1 if on else 0))

MULTI_HPW = int(os.environ.get("MUSE_QK_MULTI_HPW", "9"))
MULTI_MIN_TOKENS = 64
DECODE_HPW = int(os.environ.get("MUSE_QK_DECODE_HPW", "2"))
DECODE_KV_HPW = int(os.environ.get("MUSE_QK_DECODE_KV_HPW", "6"))
PREFILL_KV_HPW = int(os.environ.get("MUSE_QK_PREFILL_KV_HPW", "6"))
# Wide tree verify (17..32 rows): 9 heads per warp measured -0.20 ms/round in situ at M=32 (glimmer-28).
WIDE_KV_HPW = int(os.environ.get("MUSE_QK_WIDE_KV_HPW", "9"))


def _kv_hpw(num_tokens: int) -> int:
    if num_tokens <= 16:
        return DECODE_KV_HPW
    if num_tokens <= 32:
        return WIDE_KV_HPW
    return PREFILL_KV_HPW
FUSE_KVCACHE = os.environ.get("MUSE_QK_FUSE_KVCACHE", "1").strip() not in ("0", "", "false", "no")
FUSE_KVCACHE_PREFILL = FUSE_KVCACHE and os.environ.get("MUSE_QK_FUSE_KVCACHE_PREFILL", "1").strip() not in ("0", "", "false", "no")
_multi_state: dict[str, bool | None] = {"admitted": None}
_decode_state: dict[str, bool | None] = {"admitted": None}
_kv_m1_state: dict[str, bool | None] = {"admitted": None}
_kv_prefill_state: dict[str, bool | None] = {"admitted": None}

ENABLED = os.environ.get("VLLM_MUSE_GLIMMER_QK_FUSED", "1").strip() not in ("0", "", "false", "no")
_checked: dict[int, bool] = {}


def _ref_qk_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    qscale: float,
    use_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, _ = qkv.split([4096, 256, 256], dim=-1)
    qf = q.reshape(-1, 128).float()
    qf = (qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + eps)).reshape(-1, 4096) * qscale
    kf = k.reshape(-1, 128).float()
    kf = (kf * torch.rsqrt(kf.pow(2).mean(-1, keepdim=True) + eps)).reshape(-1, 256)
    if use_rope:
        cs = cos_sin_cache[positions]
        cos, sin = cs.chunk(2, dim=-1)
        cos = cos.float().unsqueeze(1)
        sin = sin.float().unsqueeze(1)
        q1, q2 = qf.view(-1, 32, 128).chunk(2, dim=-1)
        qf = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin), dim=-1).view(-1, 4096)
        k1, k2 = kf.view(-1, 2, 128).chunk(2, dim=-1)
        kf = torch.cat((k1 * cos - k2 * sin, k2 * cos + k1 * sin), dim=-1).view(-1, 256)
    return qf.to(qkv.dtype), kf.to(qkv.dtype)


def self_check(device: torch.device) -> bool:
    """Verify the native kernel matches FP32 reference outputs on M=1 and M=16 (RoPE and NoPE)."""
    if not ENABLED or _lib is None or device.type != "cuda":
        return False
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _checked.get(idx)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        return False
    ok = False
    try:
        gen = torch.Generator(device=device).manual_seed(20261123)
        t_idx = torch.arange(512, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, device=device, dtype=torch.float32) / 128.0))
        freqs = torch.einsum("i,j->ij", t_idx, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16).contiguous()
        eps, qscale = 1e-5, 3.869140625
        ok = True
        for m in (1, 16):
            qkv = (torch.randn((m, 4608), generator=gen, device=device) * 2.0).to(torch.bfloat16).contiguous()
            pos = torch.arange(m, dtype=torch.int64, device=device)
            for use_rope in (True, False):
                rq, rk = _ref_qk_norm_rope(qkv, pos, cache, eps, qscale, use_rope)
                oq, ok_t = run_qk_norm_rope(qkv, pos, cache, eps, qscale, use_rope)
                # Eager PyTorch uses IEEE rsqrt while Inductor/our kernel uses rsqrt.approx.ftz;
                # near-zero cancellations in RoPE can flip sign (+/-1e-8) between IEEE and approx
                # rsqrt, so check finiteness, small absolute error, and >=99.9% exact bit agreement.
                if not (torch.isfinite(oq).all() and torch.isfinite(ok_t).all()):
                    ok = False
                    break
                max_q = (rq.float() - oq.float()).abs().max().item()
                max_k = (rk.float() - ok_t.float()).abs().max().item()
                q_mismatch = (rq.view(torch.int16) != oq.view(torch.int16)).sum().item()
                k_mismatch = (rk.view(torch.int16) != ok_t.view(torch.int16)).sum().item()
                if max_q > 1e-3 or max_k > 1e-3:
                    ok = False
                    break
                if (not use_rope) and (q_mismatch > 0 or k_mismatch > 0):
                    ok = False
                    break
                if q_mismatch > max(4, oq.numel() // 1000) or k_mismatch > max(2, ok_t.numel() // 1000):
                    ok = False
                    break
            if not ok:
                break
        if ok and _multi is not None and DECODE_HPW > 0 and _decode_state["admitted"] is None:
            dok = True
            stream = torch.cuda.current_stream(device).cuda_stream
            pos1 = torch.zeros((1,), dtype=torch.int64, device=device)
            for use_rope in (True, False):
                for cache_t in ((cache, cache.float()) if use_rope else (cache,)):
                    for mag in (1e-3, 0.5, 4.0, 300.0):
                        r = torch.randn((1, 4608), generator=gen, device=device) * mag
                        x = (r * torch.exp(torch.randn((1, 4608), generator=gen, device=device))).to(torch.bfloat16)
                        q0, k0 = torch.empty((1, 4096), dtype=x.dtype, device=device), torch.empty((1, 256), dtype=x.dtype, device=device)
                        q1, k1 = torch.empty_like(q0), torch.empty_like(k0)
                        cf32 = int(cache_t.dtype == torch.float32)
                        c_rows = int(cache_t.shape[0]) if use_rope else 0
                        p_ptr = pos1.data_ptr() if use_rope else 0
                        c_ptr = cache_t.data_ptr() if use_rope else 0
                        st0 = _lib.muse_qk_norm_rope(x.data_ptr(), p_ptr, c_ptr, q0.data_ptr(), k0.data_ptr(), 1, c_rows, eps, qscale, int(use_rope), cf32, stream)
                        st1 = _multi(x.data_ptr(), p_ptr, c_ptr, q1.data_ptr(), k1.data_ptr(), 1, c_rows, eps, qscale, int(use_rope), cf32, DECODE_HPW, stream)
                        torch.cuda.synchronize()
                        if st0 or st1 or not (torch.equal(q0.view(torch.int16), q1.view(torch.int16)) and torch.equal(k0.view(torch.int16), k1.view(torch.int16))):
                            dok = False
                            break
                    if not dok:
                        break
                if not dok:
                    break
            _decode_state["admitted"] = dok
            logger.info("muse_qk_glue: M=1 %d-heads-per-warp decode kernel %s", DECODE_HPW, "admitted (2 blocks)" if dok else "REJECTED")
        if ok and _kv_m1 is not None and FUSE_KVCACHE and _kv_m1_state["admitted"] is None:
            kvok = True
            stream = torch.cuda.current_stream(device).cuda_stream
            pos1 = torch.tensor([17], dtype=torch.int64, device=device)
            for layout in ("NHD", "HND"):
                for block_size in (16, 32):
                    raw_shape = (6, block_size, 2, 256) if layout == "NHD" else (6, 2, block_size, 256)
                    for use_rope in (True, False):
                        for cache_t in ((cache, cache.float()) if use_rope else (cache,)):
                            for slot_val in (-1, 0, block_size + 5, 5 * block_size + block_size - 1):
                                x = (torch.randn((1, 4608), generator=gen, device=device) * 3.0).to(torch.bfloat16).contiguous()
                                raw_ref = torch.randn(raw_shape, generator=gen, device=device, dtype=torch.bfloat16)
                                raw_got = raw_ref.clone()
                                kv_ref = raw_ref.permute(0, 2, 1, 3) if layout == "NHD" else raw_ref
                                kv_got = raw_got.permute(0, 2, 1, 3) if layout == "NHD" else raw_got
                                q0, k0 = torch.empty((1, 4096), dtype=x.dtype, device=device), torch.empty((1, 256), dtype=x.dtype, device=device)
                                q1, k1 = torch.empty_like(q0), torch.empty_like(k0)
                                slot_t = torch.tensor([slot_val], dtype=torch.int64, device=device)
                                cf32 = int(cache_t.dtype == torch.float32)
                                c_rows = int(cache_t.shape[0]) if use_rope else 0
                                p_ptr = pos1.data_ptr() if use_rope else 0
                                c_ptr = cache_t.data_ptr() if use_rope else 0
                                st0 = _lib.muse_qk_norm_rope(x.data_ptr(), p_ptr, c_ptr, q0.data_ptr(), k0.data_ptr(), 1, c_rows, eps, qscale, int(use_rope), cf32, stream)
                                if slot_val >= 0:
                                    blk, off = slot_val // block_size, slot_val % block_size
                                    kv_ref[blk, :, off, :128].copy_(k0.view(2, 128))
                                    kv_ref[blk, :, off, 128:].copy_(x[0, 4352:].view(2, 128))
                                st1 = _kv_m1(
                                    x.data_ptr(), p_ptr, c_ptr, q1.data_ptr(), k1.data_ptr(),
                                    kv_got.data_ptr(), slot_t.data_ptr(),
                                    c_rows, block_size,
                                    kv_got.stride(0), kv_got.stride(2), kv_got.stride(1),
                                    eps, qscale, int(use_rope), cf32, stream,
                                )
                                torch.cuda.synchronize()
                                if (
                                    st0 or st1
                                    or not torch.equal(q0.view(torch.int16), q1.view(torch.int16))
                                    or not torch.equal(k0.view(torch.int16), k1.view(torch.int16))
                                    or not torch.equal(raw_ref.view(torch.int16), raw_got.view(torch.int16))
                                ):
                                    kvok = False
                                    break
                            if not kvok:
                                break
                        if not kvok:
                            break
                    if not kvok:
                        break
                if not kvok:
                    break
            _kv_m1_state["admitted"] = kvok
            logger.info("muse_qk_glue: M=1 fused QK-norm+RoPE+KV-cache kernel %s", "admitted" if kvok else "REJECTED")
        if ok and _kv_prefill is not None and FUSE_KVCACHE_PREFILL and _kv_prefill_state["admitted"] is None:
            pkvok = PREFILL_KV_HPW in (2, 4, 5, 6, 9) and DECODE_KV_HPW in (2, 4, 5, 6, 9) and WIDE_KV_HPW in (2, 4, 5, 6, 9)
            stream = torch.cuda.current_stream(device).cuda_stream
            if pkvok:
                for m_p in (16, 32, 512):
                    hpw_p = _kv_hpw(m_p)
                    pos_p = torch.arange(m_p, dtype=torch.int64, device=device) % int(cache.shape[0])
                    for layout in ("NHD", "HND"):
                        for block_size in (16, 32):
                            n_blk = (m_p // block_size) + 4
                            raw_shape = (n_blk, block_size, 2, 256) if layout == "NHD" else (n_blk, 2, block_size, 256)
                            for use_rope in (True, False):
                                for cache_t in ((cache, cache.float()) if use_rope else (cache,)):
                                    for q_first in (0, m_p - 64 if m_p >= 128 else m_p - 4):
                                        x = (torch.randn((m_p, 4608), generator=gen, device=device) * 3.0).to(torch.bfloat16).contiguous()
                                        raw_ref = torch.randn(raw_shape, generator=gen, device=device, dtype=torch.bfloat16)
                                        raw_got = raw_ref.clone()
                                        kv_ref = raw_ref.permute(0, 2, 1, 3) if layout == "NHD" else raw_ref
                                        kv_got = raw_got.permute(0, 2, 1, 3) if layout == "NHD" else raw_got
                                        slots_p = torch.arange(m_p, dtype=torch.int64, device=device)
                                        slots_p[m_p // 2] = -1
                                        slots_p[-1] = (n_blk - 1) * block_size + (block_size - 1)
                                        q0 = torch.empty((m_p, 4096), dtype=x.dtype, device=device)
                                        k0 = torch.empty((m_p, 256), dtype=x.dtype, device=device)
                                        q1 = torch.empty_like(q0)
                                        cf32 = int(cache_t.dtype == torch.float32)
                                        c_rows = int(cache_t.shape[0]) if use_rope else 0
                                        p_ptr = pos_p.data_ptr() if use_rope else 0
                                        c_ptr = cache_t.data_ptr() if use_rope else 0
                                        st0 = _lib.muse_qk_norm_rope(
                                            x.data_ptr(), p_ptr, c_ptr, q0.data_ptr(), k0.data_ptr(),
                                            m_p, c_rows, eps, qscale, int(use_rope), cf32, stream,
                                        )
                                        valid_mask = slots_p >= 0
                                        valid_slots = slots_p[valid_mask]
                                        blks = valid_slots // block_size
                                        offs = valid_slots % block_size
                                        kv_ref[blks, :, offs, :128] = k0[valid_mask].view(-1, 2, 128)
                                        kv_ref[blks, :, offs, 128:] = x[valid_mask, 4352:].view(-1, 2, 128)
                                        st1 = _kv_prefill(
                                            x.data_ptr(), p_ptr, c_ptr, q1.data_ptr(),
                                            kv_got.data_ptr(), slots_p.data_ptr(),
                                            m_p, q_first, c_rows, block_size,
                                            kv_got.stride(0), kv_got.stride(2), kv_got.stride(1),
                                            eps, qscale, int(use_rope), cf32, hpw_p, stream,
                                        )
                                        torch.cuda.synchronize()
                                        if (
                                            st0 or st1
                                            or not torch.equal(q0[q_first:].view(torch.int16), q1[q_first:].view(torch.int16))
                                            or not torch.equal(raw_ref.view(torch.int16), raw_got.view(torch.int16))
                                        ):
                                            pkvok = False
                                            break
                                    if not pkvok:
                                        break
                                if not pkvok:
                                    break
                            if not pkvok:
                                break
                        if not pkvok:
                            break
                    if not pkvok:
                        break
            _kv_prefill_state["admitted"] = pkvok
            logger.info(
                "muse_qk_glue: M>1 (decode HPW=%d, prefill HPW=%d) fused QK-norm+RoPE+KV-cache kernel %s",
                DECODE_KV_HPW, PREFILL_KV_HPW, "admitted" if pkvok else "REJECTED",
            )
    except Exception:  # noqa: BLE001
        logger.warning("muse_qk_glue: self-check raised on cuda:%d", idx, exc_info=True)
        ok = False
    _checked[idx] = ok
    if ok:
        logger.info("muse_qk_glue: one-pass Q/K RMSNorm + RoPE kernel admitted on cuda:%d", idx)
    else:
        logger.warning("muse_qk_glue: self-check failed on cuda:%d; keeping stock Q/K path", idx)
    return ok


def available(device: torch.device | None = None) -> bool:
    if not ENABLED or _lib is None:
        return False
    if device is None:
        return True
    return self_check(device)


def run_qk_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    eps: float,
    qscale: float,
    use_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _lib is None:
        raise RuntimeError("libmuse_qk_glue.so unavailable")
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()
    num_tokens = int(qkv.shape[0])
    q_out = qkv.new_empty((num_tokens, 4096))
    k_out = qkv.new_empty((num_tokens, 256))
    if num_tokens == 0:
        return q_out, k_out
    if use_rope:
        assert cos_sin_cache is not None
        if positions.dtype != torch.int64:
            positions = positions.to(torch.int64)
        if not positions.is_contiguous():
            positions = positions.contiguous()
        pos_ptr = positions.data_ptr()
        cache_ptr = cos_sin_cache.data_ptr()
        cache_rows = int(cos_sin_cache.shape[0])
        cache_fp32 = int(cos_sin_cache.dtype == torch.float32)
    else:
        pos_ptr = 0
        cache_ptr = 0
        cache_rows = 0
        cache_fp32 = 1
    with torch.cuda.device(qkv.device):
        stream = torch.cuda.current_stream(qkv.device).cuda_stream
        args = (
            qkv.data_ptr(), pos_ptr, cache_ptr, q_out.data_ptr(), k_out.data_ptr(),
            num_tokens, cache_rows, float(eps), float(qscale), int(use_rope), cache_fp32,
        )
        if num_tokens == 1 and DECODE_HPW > 0 and _multi is not None and _decode_state["admitted"]:
            st = _multi(*args, DECODE_HPW, stream)
        elif (
            num_tokens > MULTI_MIN_TOKENS
            and MULTI_HPW > 0
            and _multi is not None
            and _multi_ok(args, qkv, positions, cos_sin_cache, stream)
        ):
            st = _multi(*args, MULTI_HPW, stream)
        else:
            st = _lib.muse_qk_norm_rope(*args, stream)
    if st:
        raise RuntimeError(f"muse_qk_norm_rope failed with CUDA error {st}")
    return q_out, k_out


def run_qk_norm_rope_kvcache_m1(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    eps: float,
    qscale: float,
    use_rope: bool,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if (
        not FUSE_KVCACHE
        or _kv_m1 is None
        or not _kv_m1_state.get("admitted")
        or qkv.shape != (1, 4608)
        or qkv.dtype != torch.bfloat16
        or kv_cache.dtype != torch.bfloat16
        or kv_cache.ndim != 4
        or kv_cache.shape[1] != 2
        or kv_cache.shape[3] != 256
        or kv_cache.stride(3) != 1
        or slot_mapping.dtype != torch.int64
        or slot_mapping.numel() != 1
    ):
        q_out, k_out = run_qk_norm_rope(qkv, positions, cos_sin_cache, eps, qscale, use_rope)
        return q_out, k_out, False

    if not qkv.is_contiguous():
        qkv = qkv.contiguous()
    s_blk = int(kv_cache.stride(0))
    s_slot = int(kv_cache.stride(2))
    s_head = int(kv_cache.stride(1))
    block_size = int(kv_cache.shape[2])
    if (
        block_size <= 0
        or ((qkv.data_ptr() | kv_cache.data_ptr()) & 7) != 0
        or ((s_blk | s_slot | s_head) & 3) != 0
    ):
        q_out, k_out = run_qk_norm_rope(qkv, positions, cos_sin_cache, eps, qscale, use_rope)
        return q_out, k_out, False

    q_out = qkv.new_empty((1, 4096))
    k_out = qkv.new_empty((1, 256))
    if use_rope:
        assert cos_sin_cache is not None
        if positions.dtype != torch.int64:
            positions = positions.to(torch.int64)
        if not positions.is_contiguous():
            positions = positions.contiguous()
        pos_ptr = positions.data_ptr()
        cache_ptr = cos_sin_cache.data_ptr()
        cache_rows = int(cos_sin_cache.shape[0])
        cache_fp32 = int(cos_sin_cache.dtype == torch.float32)
    else:
        pos_ptr = 0
        cache_ptr = 0
        cache_rows = 0
        cache_fp32 = 1

    with torch.cuda.device(qkv.device):
        stream = torch.cuda.current_stream(qkv.device).cuda_stream
        st = _kv_m1(
            qkv.data_ptr(),
            pos_ptr,
            cache_ptr,
            q_out.data_ptr(),
            k_out.data_ptr(),
            kv_cache.data_ptr(),
            slot_mapping.data_ptr(),
            cache_rows,
            block_size,
            s_blk,
            s_slot,
            s_head,
            float(eps),
            float(qscale),
            int(use_rope),
            cache_fp32,
            stream,
        )
    if st:
        raise RuntimeError(f"muse_qk_norm_rope_kvcache_m1 failed with CUDA error {st}")
    return q_out, k_out, True


def run_qk_norm_rope_kvcache_prefill(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    eps: float,
    qscale: float,
    use_rope: bool,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    q_first_row: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    num_tokens = int(qkv.shape[0]) if qkv.ndim == 2 else -1
    if (
        not FUSE_KVCACHE_PREFILL
        or _kv_prefill is None
        or not _kv_prefill_state.get("admitted")
        or PREFILL_KV_HPW not in (2, 4, 5, 6, 9)
        or DECODE_KV_HPW not in (2, 4, 5, 6, 9)
        or qkv.ndim != 2
        or num_tokens <= 1
        or qkv.shape[1] != 4608
        or not qkv.is_cuda
        or qkv.dtype != torch.bfloat16
        or kv_cache.device != qkv.device
        or kv_cache.dtype != torch.bfloat16
        or kv_cache.ndim != 4
        or kv_cache.shape[1] != 2
        or kv_cache.shape[3] != 256
        or kv_cache.stride(3) != 1
        or slot_mapping.device != qkv.device
        or slot_mapping.ndim != 1
        or slot_mapping.dtype != torch.int64
        or not slot_mapping.is_contiguous()
        or slot_mapping.numel() < num_tokens
    ):
        q_out, k_out = run_qk_norm_rope(qkv, positions, cos_sin_cache, eps, qscale, use_rope)
        return q_out, k_out, False

    if not qkv.is_contiguous():
        qkv = qkv.contiguous()
    s_blk = int(kv_cache.stride(0))
    s_slot = int(kv_cache.stride(2))
    s_head = int(kv_cache.stride(1))
    block_size = int(kv_cache.shape[2])
    q_first = int(q_first_row) if 0 <= int(q_first_row) < num_tokens else 0
    if (
        block_size <= 0
        or ((qkv.data_ptr() | kv_cache.data_ptr()) & 7) != 0
        or ((s_blk | s_slot | s_head) & 3) != 0
    ):
        q_out, k_out = run_qk_norm_rope(qkv, positions, cos_sin_cache, eps, qscale, use_rope)
        return q_out, k_out, False

    if use_rope:
        if (
            cos_sin_cache is None
            or cos_sin_cache.device != qkv.device
            or cos_sin_cache.ndim != 2
            or cos_sin_cache.shape[0] <= 0
            or cos_sin_cache.shape[1] != 128
            or cos_sin_cache.dtype not in (torch.bfloat16, torch.float32)
            or positions.device != qkv.device
            or positions.numel() < num_tokens
        ):
            q_out, k_out = run_qk_norm_rope(qkv, positions, cos_sin_cache, eps, qscale, use_rope)
            return q_out, k_out, False
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

    q_out = qkv.new_empty((num_tokens, 4096))
    k_out = qkv.new_empty((num_tokens, 256))

    hpw = _kv_hpw(num_tokens)
    with torch.cuda.device(qkv.device):
        stream = torch.cuda.current_stream(qkv.device).cuda_stream
        st = _kv_prefill(
            qkv.data_ptr(),
            pos_ptr,
            cache_ptr,
            q_out.data_ptr(),
            kv_cache.data_ptr(),
            slot_mapping.data_ptr(),
            num_tokens,
            q_first,
            cache_rows,
            block_size,
            s_blk,
            s_slot,
            s_head,
            float(eps),
            float(qscale),
            int(use_rope),
            cache_fp32,
            hpw,
            stream,
        )
    if st:
        raise RuntimeError(f"muse_qk_norm_rope_kvcache_prefill failed with CUDA error {st}")
    return q_out, k_out, True


def _multi_ok(args, qkv, positions, cos_sin_cache, stream) -> bool:
    """Bitwise equality on the first prompt call (outside capture), adversarial magnitudes, and both RoPE/NoPE."""
    state = _multi_state["admitted"]
    if state is not None:
        return state
    if torch.cuda.is_current_stream_capturing():
        return False
    ok = True
    try:
        n = args[5]
        gen = torch.Generator(device=qkv.device).manual_seed(20261123)
        cases = [(qkv, bool(args[9]))]
        for mag in (1e-3, 0.5, 4.0, 300.0):
            r = torch.randn(qkv.shape, generator=gen, device=qkv.device) * mag
            cases.append(((r * torch.exp(torch.randn(qkv.shape, generator=gen, device=qkv.device))).to(qkv.dtype), bool(args[9])))
        alt_r = (torch.randn(qkv.shape, generator=gen, device=qkv.device) * 2.0).to(qkv.dtype)
        cases.append((alt_r, not bool(args[9])))
        alt_pos = torch.zeros((n,), dtype=torch.int64, device=qkv.device)
        alt_cache = torch.ones((1, 128), dtype=torch.float32, device=qkv.device)
        for x, rope_flag in cases:
            p_ptr = args[1] if (rope_flag and bool(args[9])) else (alt_pos.data_ptr() if rope_flag else 0)
            c_ptr = args[2] if (rope_flag and bool(args[9])) else (alt_cache.data_ptr() if rope_flag else 0)
            c_rows = args[6] if (rope_flag and bool(args[9])) else (1 if rope_flag else 0)
            c_fp32 = args[10] if (rope_flag and bool(args[9])) else 1
            outs = []
            for multi in (False, True):
                q = torch.empty((n, 4096), dtype=qkv.dtype, device=qkv.device)
                k = torch.empty((n, 256), dtype=qkv.dtype, device=qkv.device)
                a = (x.data_ptr(), p_ptr, c_ptr, q.data_ptr(), k.data_ptr(), n, c_rows, args[7], args[8], int(rope_flag), c_fp32)
                st = _multi(*a, MULTI_HPW, stream) if multi else _lib.muse_qk_norm_rope(*a, stream)
                if st:
                    ok = False
                outs.append((q, k))
            torch.cuda.synchronize()
            if not (torch.equal(outs[0][0].view(torch.int16), outs[1][0].view(torch.int16))
                    and torch.equal(outs[0][1].view(torch.int16), outs[1][1].view(torch.int16))):
                ok = False
    except Exception:  # noqa: BLE001
        ok = False
    _multi_state["admitted"] = ok
    logger.info("muse_qk_glue: %d-heads-per-warp prompt kernel %s", MULTI_HPW, "admitted" if ok else "REJECTED")
    return ok
