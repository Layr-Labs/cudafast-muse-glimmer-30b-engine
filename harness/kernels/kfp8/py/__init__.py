"""kfp8: CUTLASS-exact FP8 w8a8 GEMV for 1 <= M <= 16 rows (attention projections).

libkfp8.so computes exactly what vLLM v0.27.1's ``ops.cutlass_scaled_mm`` computes on
sm_120/sm_121 for M <= 16 with per-tensor static scales and bf16 output: the same native
e4m3 m16n8k32 MMA, each output one ascending-K chain from +0.0, then
bf16_rn((acc * scale_b) * scale_a). It reads the layer's own column-major [K, N] weight in
place (no pack, no extra memory) and schedules one warp per 8 output columns, so the
GEMV is one full wave instead of CUTLASS's 16x64 tiles (72/64/104 CTAs on 48 SMs).
Shapes: merged qkv (N, K) = (4608, 6656), output_gate_proj (4096, 6656), o_proj (6656, 4096).
Lineage: ivanfioravanti's fp8small (Nemotron CUDA track, submission 31d30b53); the PDL
launch follows the fp8small PDL variant of submission 0852152c (kshitij-hash). See csrc/kfp8.cu.

Dispatch is one torch custom op, ``torch.ops.kfp8.scaled_mm``, opaque to torch.compile
(fake impl: an empty [M, N] tensor). Its eager impl runs kfp8 when 1 <= M <= KFP8_MAX_M and
the operands are as expected, else the stock ``ops.cutlass_scaled_mm`` call with the same
arguments, so prefill and every other case stay on CUTLASS.

Environment: ``KFP8=0`` disables it (prepare() returns False: stock path everywhere).
``KFP8_VARIANT`` picks the memory-hint/launch variant (see csrc; every variant gives the
same bits): bit 0 = L2 evict_first on weight loads, bit 1 = L1::no_allocate, bit 2 = L2
prefetch ahead, bit 3 = PDL launch, bit 4 = 8-deep ring. Built by default: 0, 8, 16, 24;
default 16 (8-deep ring, 66 registers, so even o_proj's 208 CTAs are one wave on 48 SMs).
``KFP8_MAX_M`` (default 2, at most 16): the kernel is exact for every M <= 16, but its
per-row loads make M >= 4 slower than CUTLASS on a 48-SM partition; decode is M = 1.
"""

from __future__ import annotations

import ctypes
import os

import torch

__all__ = ["LIB_PATH", "SHAPES", "available", "prepare", "apply", "gemm", "regs"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libkfp8.so")

# (N, K) of the layers this was built for.
SHAPES = frozenset({(4608, 6656), (4096, 6656), (6656, 4096)})
ENABLED = os.environ.get("KFP8", "1") != "0"
VARIANT = int(os.environ.get("KFP8_VARIANT", "16"))
MAX_M = max(1, min(16, int(os.environ.get("KFP8_MAX_M", "2"))))

_lib = None
_load_error: str | None = None
_ready = False


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        lib.kfp8_init.argtypes = []
        lib.kfp8_init.restype = ctypes.c_int
        lib.kfp8_supported.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.kfp8_supported.restype = ctypes.c_int
        lib.kfp8_regs.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.kfp8_regs.restype = ctypes.c_int
        lib.kfp8_gemm.argtypes = [
            ctypes.c_void_p, ctypes.c_longlong,  # a lda
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # b sa sb
            ctypes.c_void_p, ctypes.c_longlong,  # c ldc
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # m n k variant
            ctypes.c_void_p,  # stream
        ]
        lib.kfp8_gemm.restype = ctypes.c_int
        _lib = lib
    except OSError as exc:  # library missing or unloadable: callers fall back
        _load_error = str(exc)
    return _lib


def available() -> bool:
    return _load() is not None


def _init() -> bool:
    """Loads the kernels into the context (must run outside CUDA graph capture)."""
    global _ready
    if _ready:
        return True
    lib = _load()
    if lib is None or torch.cuda.is_current_stream_capturing():
        return False
    if not lib.kfp8_supported(6656, VARIANT):
        raise RuntimeError(f"kfp8: KFP8_VARIANT={VARIANT} is not built in")
    rc = lib.kfp8_init()
    if rc != 0:
        raise RuntimeError(f"kfp8_init failed: {rc}")
    _ready = True
    return True


def regs(k: int, m16: bool = False, variant: int = VARIANT) -> int:
    lib = _load()
    assert lib is not None, _load_error
    return lib.kfp8_regs(k, int(m16), variant)


def gemm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor,
         variant: int = VARIANT) -> torch.Tensor:
    """bf16 out[M, N] for e4m3 a[M, K] and column-major e4m3 b[K, N]; no guard beyond the C side's."""
    lib = _load()
    assert lib is not None, _load_error
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    rc = lib.kfp8_gemm(
        a.data_ptr(), a.stride(0), b.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(),
        out.data_ptr(), n, m, n, k, variant,
        torch.cuda.current_stream(a.device).cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"kfp8_gemm failed: {rc} (M={m} N={n} K={k} variant={variant})")
    return out


def _stock(a, b, scale_a, scale_b):
    from vllm import _custom_ops as ops

    return ops.cutlass_scaled_mm(a, b, out_dtype=torch.bfloat16, scale_a=scale_a,
                                 scale_b=scale_b, bias=None)


def _eligible(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor,
              scale_b: torch.Tensor) -> bool:
    m = a.shape[0]
    k, n = b.shape
    return (
        1 <= m <= MAX_M
        and a.dim() == 2
        and a.shape[1] == k
        and a.dtype == torch.float8_e4m3fn
        and b.dtype == torch.float8_e4m3fn
        and (n, k) in SHAPES
        and a.stride(1) == 1
        and a.stride(0) % 16 == 0
        and a.data_ptr() % 16 == 0
        and b.stride() == (1, k)
        and b.data_ptr() % 16 == 0
        and scale_a.dtype == torch.float32
        and scale_b.dtype == torch.float32
        and scale_a.numel() == 1
        and scale_b.numel() == 1
        and a.device == b.device == scale_a.device == scale_b.device
        and (_ready or _init())
    )


@torch.library.custom_op("kfp8::scaled_mm", mutates_args=())
def scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor,
              scale_b: torch.Tensor) -> torch.Tensor:
    """ops.cutlass_scaled_mm(a, b, scale_a, scale_b, bf16) for a 2-D a: kfp8 for small M."""
    if _eligible(a, b, scale_a, scale_b):
        return gemm(a, b, scale_a, scale_b)
    return _stock(a, b, scale_a, scale_b)


@scaled_mm.register_fake
def _(a, b, scale_a, scale_b):
    return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.bfloat16)


def prepare(weight: torch.Tensor) -> bool:
    """True when a CUTLASS FP8 layer's (unpadded) weight is one kfp8 serves."""
    if not ENABLED or _load() is None:
        return False
    if weight.dim() != 2 or not weight.is_cuda or weight.dtype != torch.float8_e4m3fn:
        return False
    k, n = weight.shape
    if (n, k) not in SHAPES or weight.stride() != (1, k) or weight.data_ptr() % 16:
        return False
    if torch.cuda.get_device_capability(weight.device)[0] != 12:
        return False
    with torch.cuda.device(weight.device):
        return _init()


def apply(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor,
          scale_b: torch.Tensor) -> torch.Tensor:
    """The stock result for a prepared layer (2-D a); kfp8 when the runtime guard holds."""
    return torch.ops.kfp8.scaled_mm(a, b, scale_a, scale_b)
