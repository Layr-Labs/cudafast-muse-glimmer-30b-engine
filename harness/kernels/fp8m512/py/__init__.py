"""fp8m512: the 512-row FP8 (e4m3) GEMM of the prompt pass for the FP8 down projections (csrc/fp8m512.cu).

``pack(w)`` copies a layer's [K, N] column-major e4m3 weight once into 128-column, 64-deep tiles in the kernel's
shared-memory layout; ``gemm`` computes bf16 [M, N] for M a multiple of 128. The kernel issues mma.sync m16n8k32
e4m3 in ascending k with one fp32 accumulator per element and the epilogue bf16_rn((acc * scale_b) * scale_a); the
caller admits a layer only after comparing it bitwise with ops.cutlass_scaled_mm on that layer's weight.

The kernel is from public Yukon submission 98cd9fcd-35ae-4820-a97c-e61177a5cdd3 (commit e3f124f6e127).
``MUSE_FP8_M512=0`` in the environment disables it.
"""

from __future__ import annotations

import ctypes
import os

import torch

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfp8m512.so")
ENABLED = os.environ.get("MUSE_FP8_M512", "1") != "0"

_lib = None
_tried = False


def lib():
    """The loaded library with its shared-memory attribute set, or None when missing, disabled or failing."""
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    if not ENABLED:
        return None
    try:
        h = ctypes.CDLL(LIB_PATH)
    except OSError:
        return None
    vp, ci = ctypes.c_void_p, ctypes.c_int
    h.fp8m512_init.argtypes = []
    h.fp8m512_init.restype = ci
    h.fp8m512_repack.argtypes = [vp, vp, ci, ci, vp]
    h.fp8m512_repack.restype = ci
    h.fp8m512_gemm.argtypes = [vp, vp, vp, vp, vp, ci, ci, ci, ci, vp]
    h.fp8m512_gemm.restype = ci
    if torch.cuda.is_current_stream_capturing() or h.fp8m512_init() != 0:
        return None
    _lib = h
    return h


def supported(n: int, k: int) -> bool:
    return n % 128 == 0 and k % 64 == 0


def pack(w_kn: torch.Tensor) -> torch.Tensor:
    """[K, N] column-major e4m3 weight (stride(0) == 1, CUDA) -> int32 tensor of N * K bytes in tile order."""
    h = lib()
    if h is None:
        raise RuntimeError("fp8m512 library not available")
    k, n = w_kn.shape
    if not (w_kn.is_cuda and w_kn.element_size() == 1 and w_kn.stride(0) == 1 and w_kn.stride(1) == k
            and supported(n, k)):
        raise ValueError("unsupported weight for fp8m512")
    packed = torch.empty((n // 128) * (k // 64) * 512 * 4, dtype=torch.int32, device=w_kn.device)
    with torch.cuda.device(w_kn.device):
        rc = h.fp8m512_repack(w_kn.data_ptr(), packed.data_ptr(), n, k,
                              torch.cuda.current_stream(w_kn.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"fp8m512_repack returned CUDA status {rc}")
    return packed


def gemm(a: torch.Tensor, packed: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, n: int,
         k: int) -> torch.Tensor:
    """a: e4m3 [M, K] contiguous, M a multiple of 128; scale_a, scale_b: fp32 with one element. -> bf16 [M, N]."""
    m = a.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    with torch.cuda.device(a.device):
        rc = _lib.fp8m512_gemm(a.data_ptr(), packed.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(),
                               out.data_ptr(), m, n, k, 0, torch.cuda.current_stream(a.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"fp8m512_gemm returned CUDA status {rc}")
    return out
