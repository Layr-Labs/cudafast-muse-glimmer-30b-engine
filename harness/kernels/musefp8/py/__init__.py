"""musefp8: bit-exact M == 1 FP8 (e4m3) GEMV for the decode step (csrc/musefp8.cu).

``pack(w)`` re-packs a layer's [N, K] row-major e4m3 weight once at load into mma-fragment order; ``gemv`` then computes
one token's output with the same instruction chain and epilogue as vLLM's cutlass_scaled_mm M16 kernel, so the bf16
result is bitwise identical (``self_check`` compares the two on the layer's own weight before a layer is admitted).

``MUSEFP8=0`` in the environment disables it (every call then goes to cutlass_scaled_mm).
"""

from __future__ import annotations

import ctypes
import os

import torch

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmusefp8.so")
ENABLED = os.environ.get("MUSEFP8", "1") != "0"

_lib = None
_tried = False


def lib():
    """The loaded library, or None when it is missing or disabled."""
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
    h.musefp8_pack.argtypes = [vp, vp, ci, ci, vp]
    h.musefp8_pack.restype = ci
    h.musefp8_gemv_auto.argtypes = [vp, vp, vp, ci, vp, vp, ci, ci, vp]
    h.musefp8_gemv_auto.restype = ci
    h.musefp8_gemv.argtypes = [vp, vp, vp, ci, vp, vp, ci, ci, ci, ci, ci, vp]
    h.musefp8_gemv.restype = ci
    _lib = h
    return h


def supported(n: int, k: int) -> bool:
    return n % 8 == 0 and k % 64 == 0 and k // 32 * 4 * 8 <= 48 * 1024


def pack(w_nk: torch.Tensor) -> torch.Tensor:
    """[N, K] row-major one-byte weight (contiguous, CUDA) -> packed uint8 tensor of N*K bytes."""
    h = lib()
    if h is None:
        raise RuntimeError("musefp8 library not available")
    n, k = w_nk.shape
    if not (w_nk.is_cuda and w_nk.is_contiguous() and w_nk.element_size() == 1 and supported(n, k)):
        raise ValueError("unsupported weight for musefp8")
    packed = torch.empty(n * k, dtype=torch.uint8, device=w_nk.device)
    rc = h.musefp8_pack(w_nk.data_ptr(), packed.data_ptr(), n, k, torch.cuda.current_stream(w_nk.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"musefp8_pack failed: {rc}")
    return packed


# Launch shape: BLOCKS blocks (GB10 has 48 SMs; fewer leaves SMs to kernels on other streams, e.g. the attention chain
# that runs beside the output-gate GEMM and the L2 prefetch), about one warp per 8 columns, U loads in flight per lane.
BLOCKS = int(os.environ.get("MUSEFP8_BLOCKS", "48"))
LOADS = int(os.environ.get("MUSEFP8_U", "8"))


# Warps per block for (N, K) shapes that run beside other kernels. The attention output gate GEMM (N 4096, K 6656) runs
# while the attention chain runs on a side stream (muse_glimmer.py, _GateSideStream): flash attention's decode kernel
# needs 32.5K registers of an SM, so a GEMV block must leave them free (8 warps x 96 registers = 24.6K) or the
# attention kernel waits for the GEMV to finish.
_WPB = {(4096, 6656): int(os.environ.get("MUSEFP8_GATE_WPB", "8"))}
_SM120_CFG = {
    (4608, 6656): (144, 4),
    (4096, 6656): (128, 4),
    (6656, 4096): (139, 6),
    (6656, 19968): (139, 6),
}
_IS_SM120_CACHE: dict[int, bool] = {}


def _is_sm120(device: torch.device) -> bool:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _IS_SM120_CACHE.get(idx)
    if cached is None:
        cached = (
            torch.cuda.get_device_capability(device) == (12, 0)
            and torch.cuda.get_device_properties(device).multi_processor_count > 48
        )
        _IS_SM120_CACHE[idx] = cached
    return cached


def launch_config(n: int, k: int = 0, blocks: int = 0, is_sm120: bool = False) -> tuple[int, int]:
    if is_sm120 and not blocks and (n, k) in _SM120_CFG:
        return _SM120_CFG[(n, k)]
    b = blocks or BLOCKS
    wpb = _WPB.get((n, k)) or (n // 8 + b - 1) // b
    return b, max(4, min(18 if b == 48 else 16, wpb))


def gemv(packed: torch.Tensor, x: torch.Tensor, sb: torch.Tensor, sa: torch.Tensor, out: torch.Tensor, n: int, k: int,
         blocks: int = 0) -> None:
    """out[0:n] (bf16) = bf16_rn(sa * (sb * (x . W))) for one e4m3 activation row x[0:k]; sb has 1 or n fp32 entries, sa 1."""
    b, wpb = launch_config(n, k, blocks, is_sm120=_is_sm120(x.device))
    rc = lib().musefp8_gemv(
        packed.data_ptr(), x.data_ptr(), sb.data_ptr(), int(sb.numel() == 1), sa.data_ptr(), out.data_ptr(), n, k,
        b, wpb, LOADS, torch.cuda.current_stream(x.device).cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"musefp8_gemv failed: {rc}")


def self_check(packed: torch.Tensor, w_kn: torch.Tensor, sb: torch.Tensor, sa: torch.Tensor, reference) -> bool:
    """Compare gemv with reference(A, B, As, Bs) -> bf16 [1, N] (the stock cutlass_scaled_mm) bitwise on this weight,
    for activations with large and cancelling entries and every scale shape the layer uses."""
    k, n = w_kn.shape
    dev = w_kn.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(n * 131 + k)
    for trial in range(4):
        x = torch.randn(1, k, generator=gen, device=dev) * (0.05, 1.0, 8.0, 40.0)[trial]
        x[:, ::3] *= 16
        x[:, 1::2] = -x[:, ::2]
        xq = (x / sa.float().reshape(1, 1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        ref = reference(xq, w_kn, sa, sb)
        got = torch.empty(1, n, dtype=torch.bfloat16, device=dev)
        gemv(packed, xq, sb, sa, got, n, k)
        if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
            return False
    return True
