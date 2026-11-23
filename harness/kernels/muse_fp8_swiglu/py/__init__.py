# SPDX-License-Identifier: Apache-2.0
"""GB10 512-row FP8 gate/up, SiLU and static down-projection quantization."""
import ctypes
from pathlib import Path
import threading

import torch

_HIDDEN = 6656
_INNER = 19968
_ROWS = 512
_library = None
_sms = {}
_scratch_buf: dict[int, torch.Tensor] = {}
_lock = threading.Lock()


def _load():
    global _library
    if _library is None:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_fp8_swiglu.so")))
        ptr = ctypes.c_void_p
        lib.init_silu.argtypes = [ptr]
        lib.init_silu.restype = ctypes.c_int
        lib.prefill.argtypes = [ctypes.c_int] * 3 + [ptr] * 7 + [
            ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ptr,
        ]
        lib.prefill.restype = ctypes.c_int
        _library = lib
    return _library


def prepare(weight):
    """Return a column-major FP8 weight with alternating gate and up columns."""
    if (not weight.is_cuda or weight.dtype != torch.float8_e4m3fn
            or tuple(weight.shape) != (_HIDDEN, 2 * _INNER)
            or weight.stride() != (1, _HIDDEN)
            or torch.cuda.get_device_capability(weight.device) != (12, 1)):
        return None
    device = weight.device.index
    with _lock, torch.cuda.device(device):
        if device not in _sms:
            try:
                lib = _load()
            except OSError:
                return None
            stream = torch.cuda.current_stream(device)
            if lib.init_silu(stream.cuda_stream):
                return None
            # Weight preparation runs outside graph capture. Every subsequent
            # stream may read the same completed, immutable BF16 SiLU table.
            stream.synchronize()
            _sms[device] = torch.cuda.get_device_properties(device).multi_processor_count
    original = weight.t().view(torch.uint8)
    gate, up = original.split(_INNER)
    return torch.stack((gate, up), dim=1).reshape(2 * _INNER, _HIDDEN).view(weight.dtype).t()


def apply(a, b, scale_a, scale_b, down_scale):
    if (a.shape != (_ROWS, _HIDDEN) or a.dtype != torch.float8_e4m3fn or not a.is_contiguous()
            or b.shape != (_HIDDEN, 2 * _INNER) or b.stride() != (1, _HIDDEN)
            or b.dtype != a.dtype or not a.is_cuda or b.device != a.device
            or any(s.numel() != 1 or s.dtype != torch.float32 or s.device != a.device
                   for s in (scale_a, scale_b, down_scale))):
        raise ValueError("invalid GB10 FP8 gate/up operands")
    device = a.device.index
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        out = a.new_empty((_ROWS, _INNER))
        if torch.cuda.is_current_stream_capturing():
            scratch = torch.empty(1 << 20, dtype=torch.uint8, device=a.device)
        else:
            scratch = _scratch_buf.get(device)
            if scratch is None:
                scratch = torch.empty(1 << 20, dtype=torch.uint8, device=a.device)
                _scratch_buf[device] = scratch
        status = _library.prefill(
            _ROWS, 2 * _INNER, _HIDDEN, a.data_ptr(), b.data_ptr(), scale_a.data_ptr(),
            scale_b.data_ptr(), down_scale.data_ptr(), out.data_ptr(), scratch.data_ptr(),
            scratch.numel(), device, _sms[device], stream.cuda_stream,
        )
        if status:
            raise RuntimeError(f"GB10 FP8 gate/up CUTLASS status {status}")
        return out
