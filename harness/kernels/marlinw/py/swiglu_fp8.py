"""Complete-K paired gate/up and exact compiled SwiGLU and E4M3 activation for the SM121 prefill.

Two kernels with the same C ABI and the same bits: ``libmarlinw_swiglu_fp8`` (64-row tiles, the
reference) and ``libmarlinw_swiglu3_fp8`` (128-row TMA-fed tiles with direct SiLU and fused E4M3
quantization). A layer uses the TMA 128-row kernel only after a bitwise comparison with the 64-row
reference on its own weight.
"""
import ctypes
import os
from pathlib import Path

import torch

_ready = set()
_lib = None
_lib3 = None
_fast3 = set()
_ilv_layers: dict[int, object] = {}


def _ilv_enabled() -> bool:
    return os.environ.get("MARLINW_GATEUP_ILV", "1") != "0"


def _load(name):
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name(name)))
    except OSError:
        return None
    lib.swiglu_fp8_prepare.argtypes = []
    lib.swiglu_fp8_prepare.restype = ctypes.c_int
    lib.swiglu_fp8_gemm.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_void_p]
    lib.swiglu_fp8_gemm.restype = ctypes.c_int
    ilv = getattr(lib, "swiglu_fp8_gemm_ilv", None)
    if ilv is not None:
        ilv.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_void_p]
        ilv.restype = ctypes.c_int
    sp = getattr(lib, "marlinw_tma_set_pdl", None)
    if sp is not None:
        sp.argtypes = [ctypes.c_int]
        sp.restype = ctypes.c_int
    return lib


def set_tma_pdl(mode: int) -> int | None:
    fn = getattr(_lib3, "marlinw_tma_set_pdl", None) if _lib3 is not None else None
    if fn is None:
        return None
    return int(fn(int(mode)))


def _run(lib, x, weight, scales, global_scale, down_scale):
    out = x.new_empty((512, 19968), dtype=torch.float8_e4m3fn)
    with torch.cuda.device(x.device):
        status = lib.swiglu_fp8_gemm(
            x.data_ptr(), weight.data_ptr(), scales.data_ptr(),
            global_scale.data_ptr(), down_scale.data_ptr(), out.data_ptr(), x.stride(0),
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"Paired SwiGLU FP8 launch returned CUDA status {status}")
    return out


def _run_ilv(lib, x, w_ilv, s_ilv, global_scale, down_scale):
    out = x.new_empty((512, 19968), dtype=torch.float8_e4m3fn)
    with torch.cuda.device(x.device):
        status = lib.swiglu_fp8_gemm_ilv(
            x.data_ptr(), w_ilv.data_ptr(), s_ilv.data_ptr(),
            global_scale.data_ptr(), down_scale.data_ptr(), out.data_ptr(), x.stride(0),
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"Paired SwiGLU FP8 ilv launch returned CUDA status {status}")
    return out


def _check_input(gen, device):
    x = torch.randn((512, 6656), generator=gen, device=device, dtype=torch.float32)
    x[:, ::7] *= 12.0
    x[:, 1::2] = -x[:, ::2]
    return x.to(torch.bfloat16)


def _check(layer, lib):
    try:
        weight, scales, gs = layer.weight, layer.weight_scale, layer.weight_global_scale
        gen = torch.Generator(device=weight.device)
        gen.manual_seed(0x57161F8 + weight.data_ptr())
        for ds_val in (0.015625, 0.125):
            x = _check_input(gen, weight.device)
            ds = torch.tensor([ds_val], dtype=torch.float32, device=weight.device)
            a = _run(_lib, x, weight, scales, gs, ds)
            b = _run(lib, x, weight, scales, gs, ds)
            if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                return False
        return True
    except Exception:
        return False


def _check_ilv(layer, lib, w_ilv, s_ilv):
    try:
        weight, scales, gs = layer.weight, layer.weight_scale, layer.weight_global_scale
        gen = torch.Generator(device=weight.device)
        gen.manual_seed(0x57161F8 + weight.data_ptr())
        for ds_val in (0.015625, 0.125):
            x = _check_input(gen, weight.device)
            ds = torch.tensor([ds_val], dtype=torch.float32, device=weight.device)
            a = _run(_lib, x, weight, scales, gs, ds)
            b = _run_ilv(lib, x, w_ilv, s_ilv, gs, ds)
            if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                return False
        return True
    except Exception:
        return False


def prepare(layer):
    global _lib, _lib3
    device = layer.weight.device
    if (device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 1)
            or torch.cuda.get_device_properties(device).multi_processor_count != 48):
        return False
    if _lib is None:
        _lib = _load("libmarlinw_swiglu_fp8.so")
        if _lib is None:
            return False
        if os.environ.get("MARLINW_SWIGLU3_FP8", "1") != "0":
            _lib3 = _load("libmarlinw_swiglu3_fp8.so")
    if device not in _ready:
        with torch.cuda.device(device):
            if _lib.swiglu_fp8_prepare() != 0:
                return False
            if _lib3 is not None and _lib3.swiglu_fp8_prepare() != 0:
                _lib3 = None
        _ready.add(device)
    if _lib3 is not None and _check(layer, _lib3):
        _fast3.add(layer.weight.data_ptr())
        if _ilv_enabled() and hasattr(_lib3, "swiglu_fp8_gemm_ilv"):
            from . import swiglu as _swiglu
            w_ilv, s_ilv = _swiglu.ensure_interleaved_buffers(layer)
            if w_ilv is not None and _check_ilv(layer, _lib3, w_ilv, s_ilv):
                _ilv_layers[layer.weight.data_ptr()] = layer
    return True


def moved(old_ptr: int, new_ptr: int) -> None:
    if old_ptr in _fast3:
        _fast3.discard(old_ptr)
        _fast3.add(new_ptr)
    if old_ptr in _ilv_layers:
        _ilv_layers[new_ptr] = _ilv_layers.pop(old_ptr)


def counts() -> int:
    return len(_fast3)


def ilv_count() -> int:
    return len(_ilv_layers)


def gemm(x, weight, scales, global_scale, down_scale):
    if (x.ndim != 2 or x.shape != (512, 6656) or x.dtype != torch.bfloat16
            or x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16):
        raise ValueError("Unsupported paired SwiGLU input")
    if x.device not in _ready:
        raise RuntimeError("Paired SwiGLU was called before layer preparation")
    if down_scale.dtype != torch.float32 or down_scale.numel() != 1 or down_scale.device != x.device:
        raise ValueError("The next FP8 linear requires its original per-tensor FP32 activation scale")
    ptr = weight.data_ptr()
    layer = _ilv_layers.get(ptr)
    if layer is not None and _lib3 is not None:
        bufs = layer._buffers
        return _run_ilv(
            _lib3, x, bufs["paired_marlin_swiglu_weight"], bufs["paired_marlin_swiglu_scale"],
            global_scale, down_scale,
        )
    lib = _lib3 if (_lib3 is not None and ptr in _fast3) else _lib
    return _run(lib, x, weight, scales, global_scale, down_scale)
