# SPDX-License-Identifier: Apache-2.0
"""lmpost: bit-exact BF16 logit scaling/softcap via PyTorch LUT + fused argmax."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

ENABLED = os.environ.get("LMPOST", "1") != "0"
_lib = None
_load_failed = False
_ready_devices: set[int] = set()
_luts: dict[tuple[int, float, float | None], torch.Tensor] = {}
_scratch: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_out_slot: dict[int, int] = {}


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("liblmpost.so")))
        vp, ci = ctypes.c_void_p, ctypes.c_int
        lib.lmpost_init.argtypes = []
        lib.lmpost_init.restype = ci
        lib.lmpost_softcap_inplace.argtypes = [vp, vp, ci, vp]
        lib.lmpost_softcap_inplace.restype = ci
        lib.lmpost_softcap_argmax.argtypes = [vp, vp, vp, vp, vp, vp, ci, vp]
        lib.lmpost_softcap_argmax.restype = ci
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


def available(device: torch.device) -> bool:
    if not ENABLED or _load() is None or not device.type == "cuda":
        return False
    with torch.cuda.device(device):
        idx = torch.cuda.current_device()
        if idx not in _ready_devices:
            if _lib.lmpost_init() != 0:
                return False
            _ready_devices.add(idx)
    return True


def _get_lut(device: torch.device, mult: float, cap: float | None) -> torch.Tensor:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    c_val = None if cap is None else float(cap)
    key = (idx, float(mult), c_val)
    lut = _luts.get(key)
    if lut is None:
        raw_u16 = torch.arange(65536, dtype=torch.int32, device=device).to(torch.uint16)
        ref = raw_u16.view(torch.bfloat16) * float(mult)
        if c_val is not None:
            ref = c_val * torch.tanh(ref / c_val)
        lut = ref.view(torch.uint16).contiguous()
        _luts[key] = lut
    return lut


def _get_scratch(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    bufs = _scratch.get(idx)
    if bufs is None:
        bufs = (
            torch.empty(128, dtype=torch.float32, device=device),
            torch.empty(128, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.empty((1, 1), dtype=torch.int64, device=device),
            torch.empty((1, 1), dtype=torch.int64, device=device),
        )
        _scratch[idx] = bufs
    return bufs


def softcap_inplace(logits: torch.Tensor, mult: float, cap: float | None) -> bool:
    """Apply ``logits = cap * tanh((logits * mult) / cap)`` in-place, bit-exact for BF16."""
    if (
        logits.dtype != torch.bfloat16
        or not logits.is_contiguous()
        or logits.numel() == 0
        or not available(logits.device)
    ):
        return False
    lut = _get_lut(logits.device, mult, cap)
    rc = _lib.lmpost_softcap_inplace(
        logits.data_ptr(),
        lut.data_ptr(),
        logits.numel(),
        torch.cuda.current_stream(logits.device).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"lmpost_softcap_inplace failed: {rc}")
    return True


def softcap_argmax(logits: torch.Tensor, mult: float, cap: float | None) -> torch.Tensor | None:
    """Compute [1, 1] int64 argmax of ``cap * tanh((logits * mult) / cap)`` directly."""
    if (
        logits.ndim != 2
        or logits.shape[0] != 1
        or logits.shape[1] == 0
        or logits.dtype != torch.bfloat16
        or not logits.is_contiguous()
        or not available(logits.device)
    ):
        return None
    dev = logits.device
    lut = _get_lut(dev, mult, cap)
    sv, si, dc, out0, out1 = _get_scratch(dev)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    slot = _out_slot.get(idx, 0)
    _out_slot[idx] = slot ^ 1
    out = out0 if slot == 0 else out1
    rc = _lib.lmpost_softcap_argmax(
        logits.data_ptr(),
        lut.data_ptr(),
        out.data_ptr(),
        sv.data_ptr(),
        si.data_ptr(),
        dc.data_ptr(),
        logits.shape[1],
        torch.cuda.current_stream(dev).cuda_stream,
    )
    if rc:
        raise RuntimeError(f"lmpost_softcap_argmax failed: {rc}")
    return out
