# SPDX-License-Identifier: Apache-2.0
"""GB10 512-row FP8 gate/up, SiLU and static down-projection quantization."""
import ctypes
import os
from pathlib import Path
import threading

import torch

_HIDDEN = 6656
_INNER = 19968
_ROWS = 512
_USE_WS = os.environ.get("MUSE_FP8_SWIGLU_WS", "1") != "0"
_library = None
_sms = {}
_ws_admitted: dict[int, bool] = {}
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
        if hasattr(lib, "swiglu_ws_init") and hasattr(lib, "swiglu_ws_run"):
            lib.swiglu_ws_init.argtypes = [ptr]
            lib.swiglu_ws_init.restype = ctypes.c_int
            lib.swiglu_ws_run.argtypes = [ptr] * 7
            lib.swiglu_ws_run.restype = ctypes.c_int
        _library = lib
    return _library


def _admit_ws(device: int, packed: torch.Tensor) -> bool:
    if not _USE_WS or _sms.get(device) != 48 or not hasattr(_library, "swiglu_ws_run"):
        return False
    try:
        dev = packed.device
        stream = torch.cuda.current_stream(device)
        if _library.swiglu_ws_init(stream.cuda_stream) != 0:
            return False
        g = torch.Generator(device=dev).manual_seed(20261131)
        a_rand = (torch.randn((_ROWS, _HIDDEN), device=dev, generator=g, dtype=torch.float32) * 12.0).clamp(
            -448.0, 448.0
        ).to(torch.float8_e4m3fn)
        bits = (
            torch.arange(_ROWS * _HIDDEN, device=dev, dtype=torch.int64) * 7
            + torch.arange(_ROWS, device=dev, dtype=torch.int64).repeat_interleave(_HIDDEN)
        )
        a_bits = (bits % 256).to(torch.uint8).view(torch.float8_e4m3fn).reshape(_ROWS, _HIDDEN)
        sa = torch.tensor([0.04], dtype=torch.float32, device=dev)
        sb = torch.tensor([0.03], dtype=torch.float32, device=dev)
        sd = torch.tensor([0.08], dtype=torch.float32, device=dev)
        scratch = torch.empty(1 << 20, dtype=torch.uint8, device=dev)
        for act in (a_rand, a_bits):
            ref = torch.empty((_ROWS, _INNER), dtype=torch.float8_e4m3fn, device=dev)
            got = torch.empty((_ROWS, _INNER), dtype=torch.float8_e4m3fn, device=dev)
            rc0 = _library.prefill(
                _ROWS, 2 * _INNER, _HIDDEN, act.data_ptr(), packed.data_ptr(), sa.data_ptr(),
                sb.data_ptr(), sd.data_ptr(), ref.data_ptr(), scratch.data_ptr(),
                scratch.numel(), device, _sms[device], stream.cuda_stream,
            )
            rc1 = _library.swiglu_ws_run(
                act.data_ptr(), packed.data_ptr(), sa.data_ptr(), sb.data_ptr(), sd.data_ptr(),
                got.data_ptr(), stream.cuda_stream,
            )
            stream.synchronize()
            if rc0 != 0 or rc1 != 0 or not torch.equal(ref.view(torch.uint8), got.view(torch.uint8)):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


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
    packed = torch.stack((gate, up), dim=1).reshape(2 * _INNER, _HIDDEN).view(weight.dtype).t()
    with _lock, torch.cuda.device(device):
        if device not in _ws_admitted and not torch.cuda.is_current_stream_capturing():
            _ws_admitted[device] = _admit_ws(device, packed)
    return packed


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
        if _USE_WS and _ws_admitted.get(device) and (((a.data_ptr() | b.data_ptr() | out.data_ptr()) & 15) == 0):
            rc = _library.swiglu_ws_run(
                a.data_ptr(), b.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(),
                down_scale.data_ptr(), out.data_ptr(), stream.cuda_stream,
            )
            if rc == 0:
                return out
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
