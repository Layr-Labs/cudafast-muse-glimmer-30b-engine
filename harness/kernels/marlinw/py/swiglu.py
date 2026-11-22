"""Complete-K paired gate/up and exact compiled SwiGLU for the SM121 prefill."""
import ctypes
from pathlib import Path

import torch

_ready = set()
_lib = None


def prepare(layer):
    global _lib
    device = layer.weight.device
    if (device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 1)
            or torch.cuda.get_device_properties(device).multi_processor_count != 48):
        return False
    if device in _ready:
        return True
    if _lib is None:
        try:
            _lib = ctypes.CDLL(str(Path(__file__).with_name("libmarlinw_swiglu.so")))
        except OSError:
            return False
        _lib.swiglu_prepare.argtypes = []
        _lib.swiglu_prepare.restype = ctypes.c_int
        _lib.swiglu_gemm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_void_p]
        _lib.swiglu_gemm.restype = ctypes.c_int
    with torch.cuda.device(device):
        if _lib.swiglu_prepare() != 0:
            return False
    _ready.add(device)
    return True


def gemm(x, weight, scales, global_scale):
    if (x.ndim != 2 or x.shape != (512, 6656) or x.dtype != torch.bfloat16
            or x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16):
        raise ValueError("Unsupported paired SwiGLU input")
    if x.device not in _ready:
        raise RuntimeError("Paired SwiGLU was called before layer preparation")
    out = x.new_empty((512, 19968))
    with torch.cuda.device(x.device):
        status = _lib.swiglu_gemm(x.data_ptr(), weight.data_ptr(), scales.data_ptr(),
                                  global_scale.data_ptr(), out.data_ptr(), x.stride(0),
                                  torch.cuda.current_stream(x.device).cuda_stream)
    if status:
        raise RuntimeError(f"Paired SwiGLU launch returned CUDA status {status}")
    return out
