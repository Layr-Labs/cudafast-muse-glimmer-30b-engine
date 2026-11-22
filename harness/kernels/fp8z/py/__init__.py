# SPDX-License-Identifier: Apache-2.0
"""fp8z: lossless 7-bit coding of FP8 (E4M3) linear weights for the 1..8-row (decode) GEMM.

``prepare(w)`` encodes the original weight bytes ``w`` [N, K] (one row per output column, the
memory of vLLM's column-major B = W^T) into one buffer, decodes the whole buffer with the
kernel's own decoder and compares every byte with ``w``. It returns None (the caller keeps
CUTLASS) on any mismatch, unsupported shape or GPU, or an encoding larger than MAX_RATIO of
the original. ``gemm`` then computes what ops.cutlass_scaled_mm computes for 1..8 rows with
per-tensor scales: one ascending-K e4m3 m16n8k32 chain per output, fp32 * scale_b, then
bf16_rn(* scale_a).

Derived from fp8c (our nemotron engine's harness/kernels/fp8c): same code and layout, any
N % 32 == 0 and K % 128 == 0, encoding from the original [N, K] bytes.

FP8Z=0 disables it (prepare returns None).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

ENABLED = os.environ.get("FP8Z", "1") != "0"
MAX_RATIO = 0.95
MAIN = 3584  # bytes per (tile, chunk of 4 K-steps)
MAX_WARP_CHUNK = 124  # escapes per (tile, chunk, warp): the 32-lane window holds 128 bytes
# Kernel variant <ring depth in chunks, minimum CTAs per SM, warps per CTA>: 2 = <2,4,4>, 3 = <3,2,4>,
# 33 = <3,4,4>, 4 = <4,2,4>, 6 = <6,2,4>, 32 = <3,8,2>, 31 = <3,16,1>, 21 = <2,16,1>. FP8Z_RING overrides the
# default.
VARIANTS = (2, 3, 33, 4, 6, 32, 31, 21)
RING = int(os.environ.get("FP8Z_RING", "0"))
if RING not in VARIANTS:
    RING = 0
DEFAULT_VARIANT = 3
# Per-shape default (N, K) -> variant, from tests/fp8z_time.py on GB10 (run glim-exec-dec2-fp8zt-20261122-030138:
# cold back-to-back calls over every layer of the shape, ms per call vs the record's G2 tile).
SHAPE_VARIANT: dict[tuple[int, int], int] = {
    (4608, 6656): 3,    # qkv        0.1208 vs G2 0.1372
    (4096, 6656): 33,   # attn gate  0.1078 vs G2 0.1208
    (6656, 4096): 4,    # o_proj     0.1083 vs G2 0.1215
    (6656, 19968): 4,   # down       0.5203 vs G2 0.5924
    (39936, 6656): 6,   # gate_up    1.0041 vs G2 1.1543
}
_lib = None
_load_failed = False
_ready_devices = set()
stats = {"tensors": 0, "raw_bytes": 0, "bytes": 0, "rejected": 0, "last_reason": ""}


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libfp8z.so")))
        lib.fp8z_init.argtypes = []
        lib.fp8z_init.restype = ctypes.c_int
        lib.fp8z_run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        lib.fp8z_run.restype = ctypes.c_int
        lib.fp8z_decode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_void_p]
        lib.fp8z_decode.restype = ctypes.c_int
        lib.fp8z_header_bytes.argtypes = [ctypes.c_int]
        lib.fp8z_header_bytes.restype = ctypes.c_int
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


def available() -> bool:
    return ENABLED and _load() is not None


def header_bytes(k: int) -> int:
    c = k // 128
    return (((4 * c + 1) * 2 + 64) + 127) // 128 * 128


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@torch.no_grad()
def encode(w: torch.Tensor):
    """Encode the weight bytes w [N, K] (uint8, row = output column). Returns (buffer
    [N/32, tile_bytes] uint8, info dict) or None when a warp-chunk holds more escapes than its
    window or a tile more than the u16 offsets address."""
    n, k = w.shape
    t, c = n // 32, k // 128
    dev = w.device
    # B-fragment order: [tile, chunk, warp, lane=(col8, quad), word=(step, half), byte]; column
    # tile*32 + warp*8 + col8 holds k = (chunk*4 + step)*32 + half*16 + quad*4 + byte.
    r = (
        w.view(t, 4, 8, c, 4, 2, 4, 4)
        .permute(0, 3, 1, 2, 6, 4, 5, 7)
        .reshape(t, c, 4, 32, 8, 4)
    )
    rc = r.view(t, c, 4, 8, 4, 8, 4)  # lane -> (lane//4, lane%4)
    exp = (rc >> 3) & 15
    hc = torch.stack([(exp == x).sum((1, 4, 5, 6)) for x in range(16)], -1)  # [t, 4, 8, 16]
    cs = torch.stack([hc[..., x: x + 8].sum(-1) for x in range(9)], -1)  # [t, 4, 8, 9]
    # The 8-exponent window holding most of the column (ties: the higher window).
    e0c = (cs * 16 + torch.arange(9, device=dev)).argmax(-1).to(torch.uint8)  # [t, 4, 8]
    del hc, cs
    e0b = e0c.view(t, 1, 4, 8, 1, 1, 1)
    win = (exp >= e0b) & (exp <= e0b + 7)
    code = ((rc & 0x80) >> 1) | (((exp - e0b) & 7) << 3) | (rc & 7)
    del exp
    # Escape code per column: its least frequent window code (ties: the lowest code).
    col = torch.arange(n, device=dev, dtype=torch.int32).view(t, 1, 4, 8, 1, 1, 1)
    idx = torch.where(win, col * 128 + code.to(torch.int32), torch.full_like(col, n * 128))
    counts = torch.bincount(idx.reshape(-1), minlength=n * 128 + 1)[: n * 128].view(t, 4, 8, 128)
    del idx
    escc = counts.argmin(-1).to(torch.uint8)  # [t, 4, 8]
    escb = escc.view(t, 1, 4, 8, 1, 1, 1)
    esc = ~win | (code == escb)
    del win
    code = torch.where(esc, escb.expand_as(code), code).reshape(t, c, 4, 32, 8, 4)
    esc = esc.reshape(t, c, 4, 32, 8, 4)
    # word 7's 7 code bits ride in bit 7 of words 0..6
    shifts = torch.arange(7, device=dev, dtype=torch.uint8).view(7, 1)
    hi = ((code[..., 7:8, :] >> shifts) & 1) << 7
    cw = code[..., :7, :] | hi
    del hi, code
    main = torch.cat(
        [
            cw[..., 0:4, :].reshape(t, c, 2048),
            cw[..., 4:6, :].reshape(t, c, 1024),
            cw[..., 6, :].reshape(t, c, 512),
        ],
        dim=2,
    ).reshape(t, c * MAIN)
    del cw
    escf = esc.reshape(t, c * 4, 1024)
    cnt = escf.sum(-1, dtype=torch.int32)
    max_wc = int(cnt.max().item())
    if max_wc > MAX_WARP_CHUNK:
        return None
    tot = cnt.sum(-1)
    max_tot = int(tot.max().item())
    if max_tot >= 32768:
        return None
    offs = torch.zeros((t, 4 * c + 1), dtype=torch.int32, device=dev)
    offs[:, 1:] = cnt.cumsum(-1)
    cap = max(_round_up(max_tot, 16), 16)
    vals = r.reshape(t, -1)[esc.reshape(t, -1)]
    del esc, escf
    tile_of = torch.repeat_interleave(torch.arange(t, device=dev), tot)
    start = tot.cumsum(0) - tot
    pos = torch.arange(vals.numel(), device=dev) - start[tile_of]
    exc = torch.zeros((t, cap), dtype=torch.uint8, device=dev)
    exc[tile_of, pos] = vals
    del vals, tile_of, pos, r
    hdr_b = header_bytes(k)
    hdr = torch.zeros((t, hdr_b), dtype=torch.uint8, device=dev)
    hdr[:, : (4 * c + 1) * 2] = offs.to(torch.int16).view(torch.uint8)
    params = (e0c.to(torch.int32) | (escc.to(torch.int32) << 8)).view(t, 32)
    hdr[:, hdr_b - 64:] = params.to(torch.int16).view(torch.uint8)
    tile_bytes = _round_up(hdr_b + c * MAIN + cap, 128)
    pad = torch.zeros((t, tile_bytes - hdr_b - c * MAIN - cap), dtype=torch.uint8, device=dev)
    buf = torch.cat([hdr, main, exc, pad], dim=1).contiguous()
    n_esc = int(tot.sum().item())
    read = t * (hdr_b + c * MAIN) + n_esc  # bytes the kernel reads (padding is never read)
    info = dict(
        escapes=n_esc, escape_rate=n_esc / (n * k), max_warp_chunk=max_wc, cap=cap,
        tile_bytes=tile_bytes, bytes=buf.numel(), read_bytes=read, ratio=read / (n * k),
        bits_per_weight=8.0 * read / (n * k),
    )
    return buf, info


def _ready(device) -> bool:
    if _load() is None:
        return False
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        if index not in _ready_devices:
            rc = _lib.fp8z_init()
            if rc:
                raise RuntimeError(f"fp8z initialization failed: {rc}")
            _ready_devices.add(index)
    return True


def _reject(reason: str):
    stats["rejected"] += 1
    stats["last_reason"] = reason
    return None


def decode(buf: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The kernel decoder's output in the original layout [N, K] uint8."""
    out = torch.empty((n, k), dtype=torch.uint8, device=buf.device)
    rc = _lib.fp8z_decode(buf.data_ptr(), buf.shape[1], out.data_ptr(), n, k,
                          torch.cuda.current_stream(buf.device).cuda_stream)
    if rc:
        raise RuntimeError(f"fp8z decode failed: {rc}")
    return out


@torch.no_grad()
def prepare(w: torch.Tensor, return_info: bool = False):
    """Encode and verify the weight bytes w [N, K] (uint8 or float8_e4m3fn); None keeps CUTLASS."""
    if not ENABLED or w is None:
        return None
    if w.dtype == torch.float8_e4m3fn:
        w = w.view(torch.uint8)
    if not w.is_cuda or w.dtype != torch.uint8 or w.ndim != 2 or not w.is_contiguous():
        return None
    n, k = w.shape
    if n % 32 or k % 128 or n == 0 or k == 0:
        return _reject(f"shape {n}x{k} not tiled")
    if torch.cuda.get_device_capability(w.device) != (12, 1) or not _ready(w.device):
        return None
    if _lib.fp8z_header_bytes(k) != header_bytes(k):
        return None
    with torch.cuda.device(w.device):
        enc = encode(w)
        if enc is None:
            return _reject("a warp-chunk or tile exceeds the exception capacity")
        buf, info = enc
        if info["read_bytes"] > MAX_RATIO * n * k:
            return _reject(f"encoded ratio {info['ratio']:.4f} > {MAX_RATIO}")
        if buf.data_ptr() % 256:
            return _reject("buffer misaligned")
        dec = decode(buf, n, k)
        if not torch.equal(dec, w):
            bad = int((dec != w).sum().item())
            return _reject(f"decode differs from the weight in {bad} bytes")
        del dec
    stats["tensors"] += 1
    stats["raw_bytes"] += n * k
    stats["bytes"] += info["read_bytes"]
    return (buf, info) if return_info else buf


def gemm(a: torch.Tensor, buf: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, n: int, k: int,
         ring: int = 0, out: torch.Tensor | None = None) -> torch.Tensor:
    """a [M<=8, K] e4m3 contiguous; sa, sb fp32 scalars. Called only after prepare()."""
    m = a.shape[0]
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    rc = _lib.fp8z_run(a.data_ptr(), buf.data_ptr(), sa.data_ptr(), sb.data_ptr(), out.data_ptr(),
                       m, n, k, buf.shape[1], ring or RING or SHAPE_VARIANT.get((n, k), DEFAULT_VARIANT),
                       torch.cuda.current_stream(a.device).cuda_stream)
    if rc:
        raise RuntimeError(f"fp8z launch failed: {rc}, M={m}, N={n}, K={k}")
    return out
