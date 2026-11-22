"""marlint: ops.marlin_gemm for NVFP4 W4A16 layers at 1 to 8 rows, same bits,
reading a tile-major copy of the prepared weight.

For M <= 8, ops.marlin_gemm (vLLM v0.27.1) runs one launch of the Marlin kernel
with thread_m_blocks 1 and m_block_size_8, one block per SM; every block walks
column tiles of 16 * thread_n_blocks columns (128, or 64 when N is not a
multiple of 128) down a K range in 128-deep k-tiles. In the prepared [K/16, N]
layout one k16 row of a tile is 1 KB (512 B) out of a row of N * 8 bytes, so a
block's stream jumps by a full matrix row per k16 row. libmarlint.so is the
same kernel (marlin_template.h, stock template arguments) that reads the same
bytes from a copy in which every column tile is contiguous over K
(``relayout``); only the global load addresses change.

``MARLINT=0`` disables it. Calls the copy does not cover run the stock op.
"""

from __future__ import annotations

import ctypes
import os
import threading

import torch

__all__ = ["LIB_PATH", "available", "eligible", "relayout", "gemm", "stock_launches",
           "tile_n", "MAX_M"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmarlint.so")
ENABLED = os.environ.get("MARLINT", "1") != "0"
MAX_M = 8

_STAGES = 4
_LARGE = ((64, 256, 256), (64, 128, 128), (128, 64, 128))
_SMALL = ((128, 128, 256), (64, 128, 128), (128, 64, 128))
# (thread_k, thread_n, threads) -> index of the kernel in libmarlint.so
_KERNELS = {(128, 128, 256): 0, (128, 64, 128): 1}

_lib = None
_load_error: str | None = None
_lock = threading.Lock()
_ready: dict[int, int] = {}
_cfgs: dict = {}
_scratch: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


# ----------------------------------------------------------------- stock replay
# marlin.cu: get_kernel_cache_size / is_valid_config / the marlin_mm host loop,
# for bf16 activations, fe2m1f weights, group 16, no act order, no zero points.


def _cache_size(cfg, tmb: int) -> int:
    tk, tn, _ = cfg
    tb_m = tmb * 16
    sh_a = _STAGES * tb_m * tk * 2
    sh_b = _STAGES * (tk * tn // 8) * 4
    sh_red = tb_m * (tn + 8) * 2
    sh_bias = tn * 2
    tmp = (sh_red if sh_b > sh_red else sh_b) + sh_bias
    tmp = max(max(sh_b, sh_red), tmp)
    sh_s = _cdiv(tk, 16) * tn * 2 * _STAGES
    return tmp + sh_a + sh_s


def _valid(cfg, tmb: int, n: int, k: int, smem: int) -> bool:
    tk, tn, th = cfg
    if k % tk or n % tn or tn < 64 or tk < 64 or th < 128:
        return False
    return _cache_size(cfg, tmb) <= smem


def stock_launches(m: int, n: int, k: int, sms: int, smem: int):
    """[(row0, rows, thread_m_blocks, m_block_size_8, (thread_k, thread_n, threads))]"""
    out = []
    max_par = 16 * 8 if n <= 4096 else 16
    rest, row0, max_tmb = m, 0, 4
    while rest:
        par_count = min(rest // (max_tmb * 16), max_par)
        rows = par_count * max_tmb * 16 if par_count > 0 else rest
        tmb = min(_cdiv(rows, 16), max_tmb)
        m8 = rows <= 8
        cfg = None
        for c in (_LARGE if tmb > 1 else _SMALL):
            if _valid(c, tmb, n, k, smem - 512):
                cfg = c
                break
        if cfg is not None and n // cfg[1] * _cdiv(rows, tmb * 16) * 4 <= sms:
            if _valid((128, 64, 128), tmb, n, k, smem):
                cfg = (128, 64, 128)
        if cfg is None:
            if max_tmb > 1:
                max_tmb -= 1
                continue
            return None
        out.append((row0, rows, tmb, m8, cfg))
        row0 += rows
        rest -= rows
    return out


def _kernel_for(m: int, n: int, k: int, sms: int, smem: int):
    """Kernel index for an m-row call if ops.marlin_gemm issues exactly one
    launch with one of the configurations in libmarlint.so, else None."""
    ls = stock_launches(m, n, k, sms, smem)
    if not ls or len(ls) != 1:
        return None
    (_, rows, tmb, m8, cfg) = ls[0]
    if rows != m or tmb != 1 or not m8:
        return None
    return _KERNELS.get(cfg)


def tile_n(n: int, k: int, sms: int, smem: int) -> int | None:
    """Column-tile width the copy must use: the thread_n of the M <= 8 launch,
    which must be the same for every M in 1..8."""
    tns = set()
    for m in range(1, MAX_M + 1):
        ls = stock_launches(m, n, k, sms, smem)
        if not ls or len(ls) != 1 or _kernel_for(m, n, k, sms, smem) is None:
            return None
        tns.add(ls[0][4][1])
    return tns.pop() if len(tns) == 1 else None


# ----------------------------------------------------------------- library


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        lib.marlint_init.argtypes = [ctypes.c_int]
        lib.marlint_init.restype = ctypes.c_int
        lib.marlint_tile_major.argtypes = []
        lib.marlint_tile_major.restype = ctypes.c_int
        lib.marlint_gemm.argtypes = [
            ctypes.c_int,  # cfg
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # A B C C_tmp
            ctypes.c_void_p, ctypes.c_void_p,  # scales global_scale
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # M N K lda
            ctypes.c_void_p,  # locks
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p,  # grid smem stream
        ]
        lib.marlint_gemm.restype = ctypes.c_int
        if lib.marlint_tile_major() != 1:
            raise OSError("libmarlint.so is not the tile-major build")
        _lib = lib
    except OSError as exc:
        _load_error = str(exc)
    return _lib


def available() -> bool:
    return ENABLED and _load() is not None


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _init(dev: int, device: torch.device) -> int | None:
    """Shared-memory size the kernel runs with on this device (as ops.marlin_gemm:
    the opt-in maximum), after setting the kernel attribute, and the scratch
    buffers. Outside capture only."""
    smem = _ready.get(dev)
    if smem is not None:
        return smem
    lib = _load()
    if lib is None or torch.cuda.is_current_stream_capturing():
        return None
    props = torch.cuda.get_device_properties(dev)
    smem = int(props.shared_memory_per_block_optin)
    with torch.cuda.device(dev):
        rc = lib.marlint_init(smem)
    if rc != 0:
        raise RuntimeError(f"marlint_init failed: {rc}")
    sms = props.multi_processor_count
    # The global-reduce locks (zero, and left zero by the kernel) and the fp32
    # scratch ops.marlin_gemm allocates per call (sms * 64 * 256 floats).
    locks = torch.zeros(sms * 4, dtype=torch.int32, device=device)
    c_tmp = torch.empty(sms * 64 * 256, dtype=torch.float32, device=device)
    _scratch[dev] = (locks, c_tmp)
    _ready[dev] = smem
    return smem


def eligible(layer) -> int | None:
    """Tile width of the copy for a Marlin NVFP4 layer (after
    prepare_fp4_layer_for_marlin), or None if the layer is not covered."""
    if not available():
        return None
    n = layer.output_size_per_partition
    k = layer.input_size_per_partition
    w, s = layer.weight, layer.weight_scale
    gs = getattr(layer, "weight_global_scale", None)
    if not (
        k % 128 == 0 and n % 64 == 0
        and gs is not None and gs.dtype == torch.float32 and gs.numel() == 1
        and w.dtype == torch.int32 and tuple(w.shape) == (k // 16, 2 * n) and w.is_contiguous()
        and s.dtype == torch.float8_e4m3fn and tuple(s.shape) == (k // 16, n)
        and s.is_contiguous() and w.is_cuda and getattr(layer, "bias", None) is None
    ):
        return None
    dev = _device_index(w.device)
    smem = _init(dev, w.device)
    if smem is None:
        return None
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    tn = tile_n(n, k, sms, smem)
    if tn is None or n % tn:
        return None
    return tn


def relayout(weight: torch.Tensor, weight_scale: torch.Tensor, tn: int):
    """Tile-major copies of the prepared codes [K/16, 2N] int32 and scales
    [K/16, N] fe4m3fn: for every tn-column tile, its K/16 rows back to back.
    A permutation of whole 16-byte pieces; every value is kept."""
    kr, n2 = weight.shape
    n = n2 // 2
    nt = n // tn
    w = weight.view(kr, nt, 2 * tn).transpose(0, 1).contiguous()
    s = weight_scale.view(torch.uint8).view(kr, nt, tn).transpose(0, 1).contiguous()
    return w.view(nt * kr, 2 * tn), s.view(weight_scale.dtype).view(nt * kr, tn)


def gemm(x2: torch.Tensor, weight_tm: torch.Tensor, scale_tm: torch.Tensor,
         global_scale: torch.Tensor, n: int, k: int) -> torch.Tensor | None:
    """x2 [M, K] bf16, M <= 8 -> [M, N] bf16, equal bit for bit to
    ops.marlin_gemm on the stock layout, or None when not covered."""
    m = x2.shape[0]
    if not ENABLED or m < 1 or m > MAX_M:
        return None
    if (x2.dtype != torch.bfloat16 or x2.stride(-1) != 1 or x2.stride(0) % 8
            or x2.data_ptr() % 16 or not x2.is_cuda):
        return None
    dev = _device_index(x2.device)
    smem = _ready.get(dev)
    if smem is None:
        smem = _init(dev, x2.device)
        if smem is None:
            return None
    key = (m, n, k, dev)
    cfg = _cfgs.get(key)
    if cfg is None:
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        c = _kernel_for(m, n, k, sms, smem)
        cfg = (c, sms) if c is not None else (-1, sms)
        _cfgs[key] = cfg
    if cfg[0] < 0:
        return None
    locks, c_tmp = _scratch[dev]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x2.device)
    rc = _lib.marlint_gemm(
        cfg[0], x2.data_ptr(), weight_tm.data_ptr(), out.data_ptr(), c_tmp.data_ptr(),
        scale_tm.data_ptr(), global_scale.data_ptr(), m, n, k, x2.stride(0),
        locks.data_ptr(), cfg[1], smem,
        torch.cuda.current_stream(x2.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"marlint_gemm failed: {rc}")
    return out
