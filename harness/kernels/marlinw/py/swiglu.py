"""Complete-K paired gate/up and exact compiled SwiGLU for the SM121 prefill.

Two kernels with the same C ABI and the same bits: ``libmarlinw_swiglu`` (64-row
tiles, the reference) and ``libmarlinw_swiglu2`` (128-row tiles, each weight
fragment dequantized once for twice the rows). A layer uses the second one only
after a bitwise comparison with the first on its own weights.
"""
import ctypes
import os
from pathlib import Path

import torch

_ready = set()
_lib = None
_lib2 = None
_fast = set()  # weight data pointers whose layer passed the comparison


def _load(name):
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name(name)))
    except OSError:
        return None
    lib.swiglu_prepare.argtypes = []
    lib.swiglu_prepare.restype = ctypes.c_int
    lib.swiglu_gemm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_void_p]
    lib.swiglu_gemm.restype = ctypes.c_int
    return lib


def _run(lib, x, weight, scales, global_scale):
    out = x.new_empty((512, 19968))
    with torch.cuda.device(x.device):
        status = lib.swiglu_gemm(x.data_ptr(), weight.data_ptr(), scales.data_ptr(),
                                 global_scale.data_ptr(), out.data_ptr(), x.stride(0),
                                 torch.cuda.current_stream(x.device).cuda_stream)
    if status:
        raise RuntimeError(f"Paired SwiGLU launch returned CUDA status {status}")
    return out


def _check_input(gen, device):
    x = torch.randn(512, 6656, generator=gen, device=device)
    # Large and cancelling terms make a change of summation order show.
    x[:, ::3] *= 64
    x[1::2, 1::2] = -x[1::2, ::2]
    return x.to(torch.bfloat16)


def prepare(layer):
    global _lib, _lib2
    device = layer.weight.device
    if (device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 1)
            or torch.cuda.get_device_properties(device).multi_processor_count != 48):
        return False
    if _lib is None:
        _lib = _load("libmarlinw_swiglu.so")
        if _lib is None:
            return False
        if os.environ.get("MARLINW_SWIGLU2", "1") != "0":
            _lib2 = _load("libmarlinw_swiglu2.so")
    if device not in _ready:
        with torch.cuda.device(device):
            if _lib.swiglu_prepare() != 0:
                return False
            if _lib2 is not None and _lib2.swiglu_prepare() != 0:
                _lib2 = None
        _ready.add(device)
    if _lib2 is not None and _check(layer):
        _fast.add(layer.weight.data_ptr())
    return True


def _check(layer):
    """Bitwise comparison of the 128-row kernel with the 64-row one on this layer."""
    try:
        weight, scales, gs = layer.weight, layer.weight_scale, layer.weight_global_scale
        gen = torch.Generator(device=weight.device)
        gen.manual_seed(6656 * 31 + 39936)
        for _ in range(2):
            x = _check_input(gen, weight.device)
            a = _run(_lib, x, weight, scales, gs)
            b = _run(_lib2, x, weight, scales, gs)
            if not torch.equal(a.view(torch.int16), b.view(torch.int16)):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def uses_128(weight) -> bool:
    """True if this weight's layer passed the 128-row comparison (its calls run libmarlinw_swiglu2)."""
    return _lib2 is not None and weight.data_ptr() in _fast


def gemm_with(x, weight, scales, global_scale, fast: bool):
    """Load-time prefill weight placement: time the kernel the group's layers will run (Muse Glimmer)."""
    if x.device not in _ready:
        raise RuntimeError("Paired SwiGLU was called before layer preparation")
    return _run(_lib2 if (fast and _lib2 is not None) else _lib, x, weight, scales, global_scale)


def moved(old_ptr: int, new_ptr: int) -> None:
    """A placed weight moved to another buffer with the same bytes: it keeps its kernel (Muse Glimmer)."""
    if old_ptr in _fast:
        _fast.discard(old_ptr)
        _fast.add(new_ptr)


def gemm(x, weight, scales, global_scale):
    if (x.ndim != 2 or x.shape != (512, 6656) or x.dtype != torch.bfloat16
            or x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16):
        raise ValueError("Unsupported paired SwiGLU input")
    if x.device not in _ready:
        raise RuntimeError("Paired SwiGLU was called before layer preparation")
    lib = _lib2 if weight.data_ptr() in _fast else _lib
    return _run(lib, x, weight, scales, global_scale)
