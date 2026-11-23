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
try:
    _lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_qk_glue.so")))
    _lib.muse_qk_norm_rope.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    _lib.muse_qk_norm_rope.restype = ctypes.c_int
except OSError:
    _lib = None

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
        t_idx = torch.arange(128, device=device, dtype=torch.float32)
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
        st = _lib.muse_qk_norm_rope(
            qkv.data_ptr(), pos_ptr, cache_ptr, q_out.data_ptr(), k_out.data_ptr(),
            num_tokens, cache_rows, float(eps), float(qscale),
            int(use_rope), cache_fp32, stream,
        )
    if st:
        raise RuntimeError(f"muse_qk_norm_rope failed with CUDA error {st}")
    return q_out, k_out
