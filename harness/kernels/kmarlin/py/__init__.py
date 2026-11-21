"""kmarlin: Marlin-exact NVFP4 W4A16 GEMM for M = 1 decode rows.

libkmarlin.so computes exactly what vLLM v0.27.1's ``ops.marlin_gemm`` computes for the
NVFP4 W4A16 linears of Muse-Glimmer-30B (MLP gate_up 6656->39936, down 19968->6656,
lm_head 6656->202048) from the same Marlin-format tensors (no repack, no extra memory),
with a schedule that streams the weights faster. Port of dadon's "marlinx" (Nemotron CUDA
track, submission 100549d4). See ../README.md and csrc/kmarlin.cu for the arithmetic kept.

Dispatch is one torch custom op, ``torch.ops.kmarlin.nvfp4_gemm``, opaque to torch.compile
(fake impl: an empty [M, N] bf16 tensor). Its eager impl runs kmarlin when M == 1 and the
row is aligned, and stock Marlin (``apply_fp4_marlin_linear``, same arguments as the stock
path) otherwise, so prefill (M = 512) and any other shape stay on Marlin.

Environment: ``KMARLIN=0`` disables it (prepare() returns False, stock path everywhere).
``KMARLIN_VARIANT`` (default 0) picks the memory-hint variant, bit 0 = L2 evict_first on
the weight and scale stream, bit 1 = prefetch.global.L2 8 blocks past the ring, bit 2 = PDL
launch. Every variant gives the same bits (on an RTX 5090, evict_first and prefetch were
slower in isolation, so the default is the plain register ring). ``KMARLIN_MAX_M``
(default 1) widens the M range (<= 8). ``KMARLIN_SKIP`` = comma list of gate_up, down,
lm_head keeps those layers on stock Marlin.
"""

from __future__ import annotations

import ctypes
import os

import torch

__all__ = ["LIB_PATH", "SHAPES", "available", "prepare", "apply", "gemm", "regs"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libkmarlin.so")

# (N, K) per partition of the layers this was built for: gate_up, down, lm_head.
SHAPES = frozenset({(39936, 6656), (6656, 19968), (202048, 6656)})
ENABLED = os.environ.get("KMARLIN", "1") != "0"
VARIANT = int(os.environ.get("KMARLIN_VARIANT", "0"))
_NAMES = {(39936, 6656): "gate_up", (6656, 19968): "down", (202048, 6656): "lm_head"}
SKIP = frozenset(t for t in os.environ.get("KMARLIN_SKIP", "").split(",") if t)
MAX_M = max(1, min(8, int(os.environ.get("KMARLIN_MAX_M", "1"))))

_lib = None
_load_error: str | None = None
_ready = False
_sms: dict[int, int] = {}


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        lib.kmarlin_init.argtypes = []
        lib.kmarlin_init.restype = ctypes.c_int
        lib.kmarlin_regs.argtypes = [ctypes.c_int]
        lib.kmarlin_regs.restype = ctypes.c_int
        lib.kmarlin_marlin_tile_n.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.kmarlin_marlin_tile_n.restype = ctypes.c_int
        lib.kmarlin_gemm.argtypes = [
            ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int,  # A lda M
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # B S GS
            ctypes.c_void_p, ctypes.c_longlong,  # C ldc
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # N K sms variant
            ctypes.c_void_p,  # stream
        ]
        lib.kmarlin_gemm.restype = ctypes.c_int
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
    rc = lib.kmarlin_init()
    if rc != 0:
        raise RuntimeError(f"kmarlin_init failed: {rc}")
    _ready = True
    return True


def regs(variant: int = VARIANT) -> int:
    lib = _load()
    assert lib is not None, _load_error
    return lib.kmarlin_regs(variant)


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _sms.get(idx)
    if n is None:
        n = torch.cuda.get_device_properties(idx).multi_processor_count
        _sms[idx] = n
    return n


def gemm(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
         global_scale: torch.Tensor, n: int, k: int, variant: int = VARIANT) -> torch.Tensor:
    """out[M, n] for a 2-D bf16 x[M, k] (1 <= M <= 8); no guard beyond the C side's."""
    lib = _load()
    assert lib is not None, _load_error
    m = x.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    rc = lib.kmarlin_gemm(
        x.data_ptr(), x.stride(0), m,
        weight.data_ptr(), weight_scale.data_ptr(), global_scale.data_ptr(),
        out.data_ptr(), n, n, k, _sm_count(x.device), variant,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"kmarlin_gemm failed: {rc} (M={m} N={n} K={k} variant={variant})")
    return out


def _stock(x, weight, weight_scale, global_scale, workspace, n, k):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
    )

    return apply_fp4_marlin_linear(
        input=x, weight=weight, weight_scale=weight_scale,
        weight_global_scale=global_scale, workspace=workspace,
        size_n=n, size_k=k, bias=None,
    )


@torch.library.custom_op("kmarlin::nvfp4_gemm", mutates_args=())
def nvfp4_gemm(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
               global_scale: torch.Tensor, workspace: torch.Tensor, n: int,
               k: int) -> torch.Tensor:
    """Marlin's NVFP4 W4A16 output for a 2-D bf16 x[M, k]: kmarlin for small M, else Marlin."""
    m = x.shape[0]
    if (
        1 <= m <= MAX_M
        and x.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and (_ready or _init())
    ):
        return gemm(x, weight, weight_scale, global_scale, n, k)
    return _stock(x, weight, weight_scale, global_scale, workspace, n, k)


@nvfp4_gemm.register_fake
def _(x, weight, weight_scale, global_scale, workspace, n, k):
    return x.new_empty((x.shape[0], n), dtype=torch.bfloat16)


def prepare(layer) -> bool:
    """Marks a Marlin NVFP4 linear layer (after prepare_fp4_layer_for_marlin) as eligible."""
    layer._kmarlin_ok = False
    if not ENABLED or _load() is None:
        return False
    n = layer.output_size_per_partition
    k = layer.input_size_per_partition
    w, s, gs = layer.weight, layer.weight_scale, getattr(layer, "weight_global_scale", None)
    ok = (
        (n, k) in SHAPES
        and _NAMES[(n, k)] not in SKIP
        and getattr(layer, "params_dtype", torch.bfloat16) == torch.bfloat16
        and gs is not None
        and gs.dtype == torch.float32
        and gs.numel() == 1
        and w.dtype == torch.int32
        and tuple(w.shape) == (k // 16, 2 * n)
        and w.is_contiguous()
        and s.dtype == torch.float8_e4m3fn
        and tuple(s.shape) == (k // 16, n)
        and s.is_contiguous()
        and w.is_cuda
        and getattr(layer, "bias", None) is None
        and _load().kmarlin_marlin_tile_n(n, k, _sm_count(w.device)) > 0
    )
    if ok:
        ok = _init()
    layer._kmarlin_ok = ok
    return ok


def apply(layer, x: torch.Tensor) -> torch.Tensor | None:
    """Marlin-exact output for an eligible layer, or None (caller runs stock Marlin)."""
    if not getattr(layer, "_kmarlin_ok", False):
        return None
    k = layer.input_size_per_partition
    n = layer.output_size_per_partition
    if x.shape[-1] != k:
        return None
    x2 = x.reshape(-1, k)
    out = torch.ops.kmarlin.nvfp4_gemm(
        x2, layer.weight, layer.weight_scale, layer.weight_global_scale, layer.workspace, n, k)
    return out.reshape(*x.shape[:-1], n)
