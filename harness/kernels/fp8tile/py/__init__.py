# SPDX-License-Identifier: Apache-2.0
"""fp8tile: vLLM's SM120 FP8 w8a8 CUTLASS GEMM for M <= 16 with another tile.

libfp8tile.so instantiates the CUTLASS template that vLLM v0.27.1's
``ops.cutlass_scaled_mm`` uses for M <= 16 on SM120 (same element types,
layouts, pingpong schedule, persistent scheduler and ScaledEpilogue tree) with
a different TileShape; see csrc/fp8tile_gemm.cuh for why every output element
is computed by the same instruction sequence. It reads the layer's own [K, N]
column-major weight: nothing is packed or copied.

``FP8TILE=0`` in the environment disables it. A layer uses it only after
``prepare`` compared it bit for bit with the stock kernel on that layer's weight.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

__all__ = ["available", "built_configs", "config_for", "gemm", "prepare", "MAX_M", "TILES"]

LIB_PATH = str(Path(__file__).with_name("libfp8tile.so"))
MAX_M = 16
ENABLED = os.environ.get("FP8TILE", "1") != "0"
# Configuration index -> TileShape (M, N, K) (csrc/fp8tile_gemm.cuh).
TILES = {0: (16, 64, 128), 1: (16, 32, 128), 3: (16, 32, 256), 4: (16, 64, 256),
         6: (16, 32, 512), 7: (16, 64, 512)}
# Per weight shape (N, K) -> configuration; anything else uses DEFAULT.
BY_SHAPE: dict[tuple[int, int], int] = {}
DEFAULT = 6  # 16x32x512

_lib = None
_load_error: str | None = None
_ws: dict[tuple[int, int, int, int], int] = {}


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        lib.fp8tile_configs.argtypes = []
        lib.fp8tile_configs.restype = ctypes.c_int
        lib.fp8tile_workspace.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_size_t)]
        lib.fp8tile_workspace.restype = ctypes.c_int
        lib.fp8tile_gemm.argtypes = [
            ctypes.c_int,  # cfg
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # a b sa sb
            ctypes.c_void_p,  # d
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # m n k
            ctypes.c_void_p, ctypes.c_void_p,  # workspace stream
        ]
        lib.fp8tile_gemm.restype = ctypes.c_int
        _lib = lib
    except OSError as e:
        _load_error = str(e)
    return _lib


def available() -> bool:
    return ENABLED and _load() is not None


def built_configs() -> set[int]:
    if not available():
        return set()
    mask = _lib.fp8tile_configs()
    return {i for i in TILES if mask & (1 << i)}


def config_for(n: int, k: int) -> int:
    return BY_SHAPE.get((n, k), DEFAULT)


def _workspace_bytes(cfg: int, m: int, n: int, k: int) -> int:
    key = (cfg, m, n, k)
    if key not in _ws:
        size = ctypes.c_size_t(0)
        rc = _lib.fp8tile_workspace(cfg, m, n, k, ctypes.byref(size))
        if rc != 0:
            raise RuntimeError(f"fp8tile workspace query failed: rc={rc}")
        _ws[key] = int(size.value)
    return _ws[key]


def gemm(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor,
         cfg: int | None = None) -> torch.Tensor:
    """bf16 [M, N] = bf16_rn(sa * fp32_rn(sb * (a @ b))); a [M, K] row-major
    e4m3 (1 <= M <= 16), b the [K, N] column-major e4m3 operand, sa and sb fp32
    scalars on the device. Callers check the guard."""
    m, k = a.shape
    n = b.shape[1]
    if cfg is None:
        cfg = config_for(n, k)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    nws = _workspace_bytes(cfg, m, n, k)
    ws = torch.empty((nws,), dtype=torch.uint8, device=a.device) if nws else None
    rc = _lib.fp8tile_gemm(
        cfg, a.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr(), out.data_ptr(), m, n, k,
        ws.data_ptr() if ws is not None else None, torch.cuda.current_stream(a.device).cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"fp8tile launch failed: rc={rc} cfg={cfg} M={m} N={n} K={k}")
    return out


def supported(b: torch.Tensor, cfg: int) -> bool:
    """b is the [K, N] column-major e4m3 weight operand of cutlass_scaled_mm."""
    if not isinstance(b, torch.Tensor) or not b.is_cuda or cfg not in built_configs():
        return False
    if b.dtype != torch.float8_e4m3fn or b.ndim != 2:
        return False
    k, n = b.shape
    if k <= 0 or n <= 0 or k % 16 or n % 16 or b.stride() != (1, k) or b.data_ptr() % 16:
        return False
    # Whole tiles only (no K residue, no partial N tile): true for every FP8 shape here.
    _, tn, tk = TILES[cfg]
    if n % tn or k % tk:
        return False
    return torch.cuda.get_device_capability(b.device) == (12, 1)


def _check_inputs(k: int, device: torch.device) -> list[torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(20261121)
    xs = []
    for m, scale in ((1, 20.0), (2, 0.05), (16, 3.0), (1, 300.0)):
        x = torch.randn((m, k), generator=gen, device=device) * scale
        xs.append(x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn))
    spikes = torch.randn((1, k), generator=gen, device=device) * 0.05
    idx = torch.randint(0, k, (1, 16), generator=gen, device=device)
    spikes.scatter_(1, idx, torch.randn((1, 16), generator=gen, device=device).sign() * 448.0)
    xs.append(spikes.to(torch.float8_e4m3fn))
    xs.append(torch.arange(k, device=device, dtype=torch.int64).remainder(256).to(torch.uint8)
              .view(torch.float8_e4m3fn).reshape(1, k))
    return xs


def prepare(b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, stock) -> int | None:
    """Load-time admission of one layer. Returns the configuration to use, or
    None when the layer must keep the stock path: fp8tile must equal
    ``stock(a, b, sa, sb)`` bit for bit (bf16 bit patterns, NaNs included) on
    six inputs (M = 1, 2 and 16) over this layer's weight."""
    if not available() or torch.cuda.is_current_stream_capturing():
        return None
    k, n = b.shape
    cfg = config_for(n, k)
    if not supported(b, cfg):
        return None
    if sa.numel() != 1 or sb.numel() != 1 or sa.dtype != torch.float32 or sb.dtype != torch.float32:
        return None
    try:
        for x in _check_inputs(k, b.device):
            ref = stock(x, b, sa, sb)
            got = gemm(x, b, sa, sb, cfg)
            if not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
                return None
        return cfg
    except Exception:  # noqa: BLE001 - any failure keeps the stock path
        return None
