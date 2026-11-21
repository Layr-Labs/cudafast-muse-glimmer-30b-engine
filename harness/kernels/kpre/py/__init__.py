"""kpre: Marlin-exact NVFP4 W4A16 GEMM for prefill (M a multiple of 64, 128..1024).

libkpre.so is vLLM v0.27.1's own Marlin kernel (same template instantiation stock
``ops.marlin_gemm`` launches at M = 512: thread_m_blocks 4, thread_k 64, thread_n 256,
256 threads, 4 stages, grid = SM count) with one scheduling change: the part-1 tiles,
which stock reduces whole-K on one CTA each, are visited column-major over the 64-row M
slabs instead of slab-major. A round of CTAs then shares each weight tile through L2,
instead of the whole weight being streamed from DRAM once per 64-row slab (8 times at
M = 512). Part 2 (the K-split tiles and their lock-ordered fp32 reduce) is stock, so
every output element comes from the same instructions in the same order: same bits.

``maybe_gemm`` returns None when the call is outside the guard (the caller then runs stock
Marlin). ``KPRE=0`` disables it.
"""

from __future__ import annotations

import ctypes
import os

import torch

__all__ = ["LIB_PATH", "apply", "available", "maybe_gemm", "prepare"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libkpre.so")
ENABLED = os.environ.get("KPRE", "1") != "0"
# (N, K): MLP gate_up and down of Muse-Glimmer-30B. lm_head (N = 202048) is never M > 1.
SHAPES = frozenset({(39936, 6656), (6656, 19968)})
MIN_M = 17

_lib = None
_load_error: str | None = None
_ready = False
_sms: dict[int, int] = {}
_ctmp: dict[int, torch.Tensor] = {}


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    if not ENABLED:
        _load_error = "disabled (KPRE=0)"
        return None
    try:
        lib = ctypes.CDLL(LIB_PATH)
        lib.kpre_init.argtypes = []
        lib.kpre_init.restype = ctypes.c_int
        lib.kpre_eligible.argtypes = [ctypes.c_int] * 4
        lib.kpre_eligible.restype = ctypes.c_int
        lib.kpre_gemm.argtypes = [
            ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int,  # A lda M
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # B S GS
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # C C_tmp locks
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # N K sms
            ctypes.c_void_p,  # stream
        ]
        lib.kpre_gemm.restype = ctypes.c_int
        _lib = lib
    except OSError as exc:
        _load_error = str(exc)
    return _lib


def available() -> bool:
    return _load() is not None


def _init() -> bool:
    global _ready
    if _ready:
        return True
    lib = _load()
    if lib is None or torch.cuda.is_current_stream_capturing():
        return False
    rc = lib.kpre_init()
    if rc != 0:
        raise RuntimeError(f"kpre_init failed: {rc}")
    _ready = True
    return True


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _sms.get(idx)
    if n is None:
        n = torch.cuda.get_device_properties(idx).multi_processor_count
        _sms[idx] = n
    return n


def _alloc_ctmp(device: torch.device) -> torch.Tensor:
    # fp32 reduce scratch, as stock marlin_gemm allocates (sms * 64 * max_thread_n) per call.
    idx = device.index if device.index is not None else torch.cuda.current_device()
    ctmp = torch.empty(_sm_count(device) * 64 * 256, dtype=torch.float32, device=device)
    _ctmp[idx] = ctmp
    return ctmp


def maybe_gemm(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
               global_scale: torch.Tensor, workspace: torch.Tensor, n: int,
               k: int) -> torch.Tensor | None:
    """Stock-Marlin-identical out[M, n] for eligible prefill calls, else None."""
    m = x.shape[0]
    if (
        m < MIN_M
        or (n, k) not in SHAPES
        or x.dim() != 2
        or x.shape[1] != k
        or x.dtype != torch.bfloat16
        or x.stride(-1) != 1
        or x.stride(0) % 8 != 0
        or x.data_ptr() % 16 != 0
        or weight.dtype != torch.int32
        or not weight.is_contiguous()
        or weight_scale.dtype != torch.float8_e4m3fn
        or not weight_scale.is_contiguous()
        or global_scale is None
        or global_scale.dtype != torch.float32
        or not (_ready or _init())
    ):
        return None
    sms = _sm_count(x.device)
    if workspace.numel() < sms or not _lib.kpre_eligible(m, n, k, sms):
        return None
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    ctmp = _ctmp.get(idx)
    if ctmp is None:
        if torch.cuda.is_current_stream_capturing():
            return None  # prepare() allocates it before any capture
        ctmp = _alloc_ctmp(x.device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    rc = _lib.kpre_gemm(
        x.data_ptr(), x.stride(0), m,
        weight.data_ptr(), weight_scale.data_ptr(), global_scale.data_ptr(),
        out.data_ptr(), ctmp.data_ptr(), workspace.data_ptr(), n, k, sms,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if rc > 0:
        return None
    if rc != 0:
        raise RuntimeError(f"kpre_gemm failed: {rc} (M={m} N={n} K={k})")
    return out


def _stock(x, weight, weight_scale, global_scale, workspace, n, k):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
    )
    return apply_fp4_marlin_linear(
        input=x, weight=weight, weight_scale=weight_scale, weight_global_scale=global_scale,
        workspace=workspace, size_n=n, size_k=k, bias=None)


@torch.library.custom_op("kpre::nvfp4_gemm", mutates_args=())
def nvfp4_gemm(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
               global_scale: torch.Tensor, workspace: torch.Tensor, n: int, k: int,
               small: bool) -> torch.Tensor:
    """Marlin's NVFP4 output for bf16 x[M, k]. The M branch lives inside this opaque op so a
    torch.compile'd graph (symbolic M) never specialises on it: kpre for eligible prefill M,
    else kmarlin's op when the layer has it (small), else stock Marlin."""
    out = maybe_gemm(x, weight, weight_scale, global_scale, workspace, n, k)
    if out is not None:
        return out
    if small:
        return torch.ops.kmarlin.nvfp4_gemm(x, weight, weight_scale, global_scale, workspace,
                                            n, k)
    return _stock(x, weight, weight_scale, global_scale, workspace, n, k)


@nvfp4_gemm.register_fake
def _(x, weight, weight_scale, global_scale, workspace, n, k, small):
    return x.new_empty((x.shape[0], n), dtype=torch.bfloat16)


def prepare(layer) -> bool:
    """Marks a Marlin NVFP4 layer (after prepare_fp4_layer_for_marlin) of a kpre shape."""
    layer._kpre_ok = False
    n = layer.output_size_per_partition
    k = layer.input_size_per_partition
    w, s, gs = layer.weight, layer.weight_scale, getattr(layer, "weight_global_scale", None)
    ok = (
        (n, k) in SHAPES
        and available()
        and gs is not None and gs.dtype == torch.float32 and gs.numel() == 1
        and w.dtype == torch.int32 and w.is_contiguous() and w.is_cuda
        and s.dtype == torch.float8_e4m3fn and s.is_contiguous()
        and getattr(layer, "bias", None) is None
        and layer.workspace.numel() >= _sm_count(w.device)
        and _init()
    )
    if ok:
        idx = w.device.index if w.device.index is not None else torch.cuda.current_device()
        if idx not in _ctmp:
            _alloc_ctmp(w.device)
    layer._kpre_ok = ok
    return ok


def apply(layer, x: torch.Tensor) -> torch.Tensor | None:
    """Output for a prepared layer via the kpre op, or None (caller keeps its own path)."""
    if not getattr(layer, "_kpre_ok", False):
        return None
    k = layer.input_size_per_partition
    n = layer.output_size_per_partition
    if x.shape[-1] != k or x.dtype != torch.bfloat16:
        return None
    out = torch.ops.kpre.nvfp4_gemm(
        x.reshape(-1, k), layer.weight, layer.weight_scale, layer.weight_global_scale,
        layer.workspace, n, k, bool(getattr(layer, "_kmarlin_ok", False)))
    return out.reshape(*x.shape[:-1], n)
