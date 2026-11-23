# SPDX-License-Identifier: Apache-2.0
"""muse_fp8_prefill: prompt-sized (M > 16) FP8 w8a8 GEMMs on GB10 through CUTLASS SM120 kernels tuned for 512 rows.

``config(n, k)`` gives the kernel id for a weight shape (None: not served). ``gemm(a, b, sa, sb, cfg)`` returns the
bf16 product with the stock ScaledEpilogue arithmetic. ``admit(b, sb, sa, reference)`` compares this kernel with the
stock op on the layer's own weight for several row counts and activation patterns and returns True only when every
output bit matches. ``MUSE_FP8_PREFILL=0`` disables the package (every call then takes the stock op).
"""

from __future__ import annotations

import ctypes
import os
import threading

import torch

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmuse_fp8_prefill.so")
ENABLED = os.environ.get("MUSE_FP8_PREFILL", "1") != "0"
# Micro-attribution: The k=128 CUTLASS SM120 Cooperative config choices Cfg2 (128x192x128) for down
# and Cfg3 (128x96x128) for qkv/o_proj are inspired by @spariev (Yukon submission 262ee4f1);
# keeping 4096x6656 output_gate on fp8ws and caching per-device scratch buffers are retained here.
_CFG: dict[tuple[int, int], int] = {
    (6656, 19968): 2,  # down: 128x192x128
    (6656, 4096): 3,   # o_proj: 128x96x128
    (4608, 6656): 2,   # qkv: 128x192x128 (exact 2.00 waves on 48 SMs; faster cold-L2 than 128x96x128)
}
_lib = None
_tried = False
_sms: dict[int, int] = {}
_scratch_buf: dict[int, torch.Tensor] = {}
_lock = threading.Lock()


def lib():
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
    vp = ctypes.c_void_p
    h.fp8_prefill_gemm.argtypes = [ctypes.c_int] * 4 + [vp] * 6 + [ctypes.c_size_t, ctypes.c_int, ctypes.c_int, vp]
    h.fp8_prefill_gemm.restype = ctypes.c_int
    _lib = h
    return _lib


def config(n: int, k: int) -> int | None:
    return _CFG.get((n, k))


def supported(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> bool:
    return (a.dim() == 2 and a.size(0) > 16 and a.dtype == b.dtype == torch.float8_e4m3fn
            and a.is_cuda and a.device == b.device == sa.device == sb.device
            and a.is_contiguous() and b.dim() == 2 and b.stride(0) == 1 and b.stride(1) == b.size(0)
            and a.size(1) == b.size(0) and a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0
            and sa.numel() == sb.numel() == 1 and sa.dtype == sb.dtype == torch.float32
            and torch.cuda.get_device_capability(a.device) == (12, 1))


def gemm(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, cfg: int) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    dev = a.device.index
    if dev not in _sms:
        with _lock:
            _sms[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    if torch.cuda.is_current_stream_capturing():
        scratch = torch.empty(1 << 20, dtype=torch.uint8, device=a.device)
    else:
        scratch = _scratch_buf.get(dev)
        if scratch is None:
            scratch = torch.empty(1 << 20, dtype=torch.uint8, device=a.device)
            _scratch_buf[dev] = scratch
    status = _lib.fp8_prefill_gemm(
        cfg, m, n, k, a.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr(), out.data_ptr(),
        scratch.data_ptr(), scratch.numel(), dev, _sms[dev], torch.cuda.current_stream(a.device).cuda_stream,
    )
    if status:
        raise RuntimeError(f"muse_fp8_prefill: CUTLASS status {status} (cfg {cfg}, {m}x{n}x{k})")
    return out


def admit(b: torch.Tensor, sb: torch.Tensor, sa: torch.Tensor, reference) -> bool:
    """Bitwise comparison with reference(a, b, sa, sb) on this weight: 512 / 300 / 33 rows, three activation patterns
    (wide normal, large columns cancelling their neighbours, small values), saturating at the e4m3 range."""
    if lib() is None:
        return False
    k, n = b.shape
    cfg = config(n, k)
    if cfg is None:
        return False
    gen = torch.Generator(device=b.device)
    gen.manual_seed(n * 7919 + k)
    for m, pattern in ((512, 0), (512, 1), (512, 2), (300, 1), (33, 0)):
        x = torch.randn(m, k, generator=gen, device=b.device) * 40.0
        if pattern == 1:
            x[:, ::7] *= 12
            x[:, 1::2] = -x[:, ::2][:, : x[:, 1::2].shape[1]]
        elif pattern == 2:
            x = x * 0.0125
        a = x.clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
        if not supported(a, b, sa, sb):
            return False
        ref = reference(a, b, sa, sb)
        got = gemm(a, b, sa, sb, cfg)
        if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
            return False
    return True
