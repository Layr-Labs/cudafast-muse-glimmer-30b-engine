"""marlinw: ops.marlin_gemm for NVFP4 W4A16 layers at prefill widths, same bits.

For M >= 128 rows, ops.marlin_gemm (vLLM v0.27.1) splits the rows into 64-row
chunks and stripes (chunk, 256-column tile, 64-deep k-tile) work over one block
per SM in chunk-major order, so a weight tile is fetched once per chunk, far
apart in time. libmarlinw.so runs the same kernel body on the same slices
(chunk, column tile, k range, reduction index), computed here on the host by
replaying the stock partition, but hands the full-K tiles out column-major: the
chunks that share a weight tile run at the same time and share its L2 lines.
Slices of k-split tiles keep their stock reduction order (the same lock
protocol and fp32 scratch). See README.md for the arithmetic it preserves.

``MARLINW=0`` disables it. Anything the replica does not cover (other
configurations, the rows of a last partial chunk) runs through the stock op on
those rows.

``gemm_silu_mul`` computes silu(gate) * up of a [gate | up] weight in the same
call (see ``silu_mul``); ``MARLINW_SILU=0`` disables that fusion.
"""

from __future__ import annotations

import ctypes
import os
import threading

import torch

__all__ = ["LIB_PATH", "available", "prepare", "gemm", "gemm_silu_mul", "silu_mul",
           "plan", "stock_launches", "stock_slices", "MIN_M"]

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmarlinw.so")
ENABLED = os.environ.get("MARLINW", "1") != "0"
SILU_ENABLED = os.environ.get("MARLINW_SILU", "1") != "0"
MIN_M = 128
GROUP_CHUNKS = 8

_STAGES = 4
_LARGE = ((64, 256, 256), (64, 128, 128), (128, 64, 128))
_SMALL = ((128, 128, 256), (64, 128, 128), (128, 64, 128))
_KERNEL_CFG = (64, 256, 256)
_KERNEL_TMB = 4

_lib = None
_load_error: str | None = None
_lock = threading.Lock()
_ready: dict[int, int] = {}
_plans: dict = {}
_scratch: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_retired: list[tuple[torch.Tensor, torch.Tensor]] = []


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


def stock_slices(rows: int, n: int, k: int, tmb: int, m8: bool, cfg, grid: int):
    """Per block, the slices marlin_template.h's init_slice walks:
    [(par, col, k_tile0, k_tiles, slice_idx, slice_count)]."""
    tk, tn, _ = cfg
    m_block = 8 if m8 else 16 * tmb
    parallel = rows // m_block if rows > m_block else 1
    k_tiles, n_tiles = k // tk, n // tn
    glob = parallel * n_tiles
    part2, part1 = glob, 0
    if glob > grid:
        part2 = glob % grid
        if part2 * 3 <= grid:
            part2 += grid
        part1 = (glob - part2) // grid
    iters = _cdiv(k_tiles * part2, grid)  # group_blocks 1 < thread_k_blocks: no rounding
    work = []
    for b in range(grid):
        mine = []
        col_par, srow = b, 0
        left1 = part1
        while left1:
            left1 -= 1
            mine.append((col_par // n_tiles, col_par % n_tiles, 0, k_tiles, 0, 1))
            col_par += grid
        col_par = (iters * b) // k_tiles
        srow = (iters * b) % k_tiles
        col = (col_par + glob - part2) % n_tiles
        par = (col_par + glob - part2) // n_tiles
        while True:
            it = iters * (b + 1) - (k_tiles * col_par + srow)
            if it < 0 or col_par >= part2:
                it = 0
            if it == 0:
                break
            if srow + it > k_tiles:
                it = k_tiles - srow
            cnt, idx = 1, 0
            col_first = iters * _cdiv(k_tiles * col_par, iters)
            if col_first <= k_tiles * (col_par + 1):
                col_off = col_first - k_tiles * col_par
                cnt = _cdiv(k_tiles - col_off, iters)
                if col_off > 0:
                    cnt += 1
                delta_first = iters * b - col_first
                if delta_first < 0 or (col_off == 0 and delta_first == 0):
                    idx = cnt - 1
                else:
                    idx = cnt - 1 - delta_first // iters
                    if col_off > 0:
                        idx -= 1
            if col == n_tiles:
                col = 0
                par += 1
            mine.append((par, col, srow, it, idx, cnt))
            srow = 0
            col_par += 1
            col += 1
        work.append(mine)
    return work


# ----------------------------------------------------------------- plan


class Plan:
    __slots__ = ("m", "n", "k", "sms", "rows", "stock_rows", "items", "ptr", "slots",
                 "work", "work_ptr", "max_load", "min_load", "silu_mul")


def plan(m: int, n: int, k: int, sms: int, smem: int,
         silu_mul: bool = False) -> Plan | None:
    """Work list for the rows the replica covers, or None (stock for all rows).

    silu_mul: list for the fused gated-MLP epilogue, or None unless every row
    is covered and every slice is a whole-K tile. A block then runs the gate
    tile (chunk, c) and next the up tile (chunk, c + n_tiles / 2); pairs are
    handed out column-major within groups of GROUP_CHUNKS chunks."""
    launches = stock_launches(m, n, k, sms, smem)
    if launches is None:
        return None
    covered = []
    stock_rows = []
    for (row0, rows, tmb, m8, cfg) in launches:
        if tmb == _KERNEL_TMB and not m8 and cfg == _KERNEL_CFG and rows % 64 == 0:
            if stock_rows:  # covered rows must be one prefix
                return None
            covered.append((row0, rows, tmb, m8, cfg))
        else:
            stock_rows.append((row0, row0 + rows))
    if not covered:
        return None
    if silu_mul and (stock_rows or n % 512):
        return None
    full = []
    split = []
    for (row0, rows, tmb, m8, cfg) in covered:
        chunk0 = row0 // 64
        for mine in stock_slices(rows, n, k, tmb, m8, cfg, sms):
            for (par, col, k0, kn, idx, cnt) in mine:
                if cnt == 1:
                    c = chunk0 + par
                    full.append((c // GROUP_CHUNKS, col, c, k0, kn))
                else:
                    split.append((chunk0 + par, col, k0, kn, idx, cnt))
    if silu_mul:
        return _plan_silu_mul(m, n, k, sms, covered, full, split)
    slot_of: dict = {}
    for (chunk, col, *_rest) in sorted(split):
        slot_of.setdefault((chunk, col), len(slot_of))
    lists: list[list] = [[] for _ in range(sms)]
    load = [0] * sms
    # Column-major inside groups of GROUP_CHUNKS chunks: the blocks that run
    # together share weight tiles, and a group's activations stay in L2.
    full.sort()
    for j, (_g, col, chunk, k0, kn) in enumerate(full):
        b = j % sms
        lists[b].append((chunk, col, k0, kn, 0, 1, 0))
        load[b] += kn
    tail: list[list] = [[] for _ in range(sms)]
    for (chunk, col, k0, kn, idx, cnt) in sorted(split, key=lambda s: (-s[3], s[4], s[0], s[1])):
        b = min(range(sms), key=lambda i: (load[i], i))
        tail[b].append((chunk, col, k0, kn, idx, cnt, slot_of[(chunk, col)]))
        load[b] += kn
    for b in range(sms):
        # Every block runs its unsynchronised slices first and its reduction
        # slices by increasing index: a slice waits only for slices of lower
        # index, so with all blocks resident every wait ends.
        lists[b].extend(sorted(tail[b], key=lambda s: (s[4], s[0], s[1])))
    items = []
    ptr = [0]
    for b in range(sms):
        for (chunk, col, k0, kn, idx, cnt, slot) in lists[b]:
            items.append((chunk, col, k0, kn, idx, cnt, slot, 0))
        ptr.append(len(items))
    p = Plan()
    p.m, p.n, p.k, p.sms = m, n, k, sms
    p.rows = covered[-1][0] + covered[-1][1]
    p.stock_rows = stock_rows
    p.items, p.ptr, p.slots = items, ptr, len(slot_of)
    p.max_load, p.min_load = max(load), min(load)
    p.work = p.work_ptr = None
    p.silu_mul = False
    return p


def _plan_silu_mul(m, n, k, sms, covered, full, split) -> Plan | None:
    if split:
        return None
    half = n // 256 // 2
    k_tiles = k // 64
    tiles = {(chunk, col) for (_g, col, chunk, k0, kn) in full}
    pairs = []
    for (g, col, chunk, k0, kn) in sorted(full):
        if k0 != 0 or kn != k_tiles:
            return None
        if col < half:
            if (chunk, col + half) not in tiles:
                return None
            pairs.append((g, col, chunk))
    if 2 * len(pairs) != len(full):
        return None
    lists: list[list] = [[] for _ in range(sms)]
    for j, (_g, col, chunk) in enumerate(pairs):
        lists[j % sms].append((chunk, col))
    items = []
    ptr = [0]
    load = []
    for b in range(sms):
        for (chunk, col) in lists[b]:
            items.append((chunk, col, 0, k_tiles, 0, 1, 0, 0))
            items.append((chunk, col + half, 0, k_tiles, 0, 1, 0, 0))
        ptr.append(len(items))
        load.append(2 * k_tiles * len(lists[b]))
    p = Plan()
    p.m, p.n, p.k, p.sms = m, n, k, sms
    p.rows = covered[-1][0] + covered[-1][1]
    p.stock_rows = []
    p.items, p.ptr, p.slots = items, ptr, 0
    p.max_load, p.min_load = max(load), min(load)
    p.work = p.work_ptr = None
    p.silu_mul = True
    return p


# ----------------------------------------------------------------- library


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        lib.marlinw_init.argtypes = [ctypes.c_int]
        lib.marlinw_init.restype = ctypes.c_int
        lib.marlinw_regs.argtypes = [ctypes.c_int]
        lib.marlinw_regs.restype = ctypes.c_int
        lib.marlinw_gemm.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # A B C C_tmp
            ctypes.c_void_p, ctypes.c_void_p,  # scales global_scale
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # M N K lda
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # locks work work_ptr
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,  # grid smem silu stream
        ]
        lib.marlinw_gemm.restype = ctypes.c_int
        lib.marlinw_silu_mul.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.marlinw_silu_mul.restype = ctypes.c_int
        _lib = lib
    except OSError as exc:
        _load_error = str(exc)
    return _lib


def available() -> bool:
    return _load() is not None


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _init(dev: int) -> int | None:
    """Shared-memory size the kernel runs with on this device (as ops.marlin_gemm:
    the opt-in maximum), after setting the kernel attribute. Outside capture."""
    smem = _ready.get(dev)
    if smem is not None:
        return smem
    lib = _load()
    if lib is None or torch.cuda.is_current_stream_capturing():
        return None
    props = torch.cuda.get_device_properties(dev)
    smem = int(props.shared_memory_per_block_optin)
    with torch.cuda.device(dev):
        rc = lib.marlinw_init(smem)
    if rc != 0:
        raise RuntimeError(f"marlinw_init failed: {rc}")
    _ready[dev] = smem
    return smem


def _get_plan(m: int, n: int, k: int, device: torch.device, silu_mul: bool = False):
    dev = _device_index(device)
    key = (m, n, k, dev, silu_mul)
    p = _plans.get(key)
    if p is not None:
        return p
    if torch.cuda.is_current_stream_capturing():
        return None
    smem = _init(dev)
    if smem is None:
        return None
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    with _lock:
        p = plan(m, n, k, sms, smem, silu_mul)
        if p is not None:
            p.work = torch.tensor(p.items, dtype=torch.int32).reshape(-1).to(device)
            p.work_ptr = torch.tensor(p.ptr, dtype=torch.int32).to(device)
            _ensure_scratch(dev, device, max(p.slots, 1))
        _plans[key] = p if p is not None else False
    return p


def _ensure_scratch(dev: int, device: torch.device, slots: int):
    cur = _scratch.get(dev)
    if cur is not None and cur[0].numel() >= slots:
        return
    if cur is not None:
        # A captured CUDA graph may hold the old buffers: never free them.
        _retired.append(cur)
    locks = torch.zeros(slots, dtype=torch.int32, device=device)
    c_tmp = torch.empty(slots * 64 * 256, dtype=torch.float32, device=device)
    _scratch[dev] = (locks, c_tmp)


# ----------------------------------------------------------------- calls


def gemm(x2: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
         global_scale: torch.Tensor, workspace: torch.Tensor, n: int, k: int,
         stock) -> torch.Tensor | None:
    """x2 [M, K] bf16 -> [M, N] bf16 equal to stock(x2) bit for bit, or None.
    stock(a, c) must run the stock op on rows a into the preallocated rows c."""
    m = x2.shape[0]
    if not ENABLED or m < MIN_M:
        return None
    if (x2.dtype != torch.bfloat16 or x2.stride(-1) != 1 or x2.stride(0) % 8
            or x2.data_ptr() % 16 or not x2.is_cuda):
        return None
    p = _get_plan(m, n, k, x2.device)
    if not p:
        return None
    dev = _device_index(x2.device)
    locks, c_tmp = _scratch[dev]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x2.device)
    rc = _lib.marlinw_gemm(
        x2.data_ptr(), weight.data_ptr(), out.data_ptr(), c_tmp.data_ptr(),
        weight_scale.data_ptr(), global_scale.data_ptr(), p.rows, n, k, x2.stride(0),
        locks.data_ptr(), p.work.data_ptr(), p.work_ptr.data_ptr(), p.sms, _ready[dev],
        0, torch.cuda.current_stream(x2.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"marlinw_gemm failed: {rc}")
    for (r0, r1) in p.stock_rows:
        stock(x2[r0:r1], out[r0:r1])
    return out


def _x_ok(x2: torch.Tensor) -> bool:
    return (x2.dtype == torch.bfloat16 and x2.stride(-1) == 1 and x2.stride(0) % 8 == 0
            and x2.data_ptr() % 16 == 0 and x2.is_cuda)


def gemm_silu_mul(x2: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                  global_scale: torch.Tensor, n: int, k: int) -> torch.Tensor | None:
    """x2 [M, K] bf16 -> [M, N / 2] bf16: silu(gate) * up of the [M, N] result
    [gate | up] of the stock op, bit for bit as silu_mul(stock(x2)); or None
    (no fused plan for this M)."""
    m = x2.shape[0]
    if not (ENABLED and SILU_ENABLED) or m < MIN_M or not _x_ok(x2):
        return None
    p = _get_plan(m, n, k, x2.device, True)
    if not p:
        return None
    dev = _device_index(x2.device)
    locks, c_tmp = _scratch[dev]
    out = torch.empty((m, n // 2), dtype=torch.bfloat16, device=x2.device)
    rc = _lib.marlinw_gemm(
        x2.data_ptr(), weight.data_ptr(), out.data_ptr(), c_tmp.data_ptr(),
        weight_scale.data_ptr(), global_scale.data_ptr(), p.rows, n, k, x2.stride(0),
        locks.data_ptr(), p.work.data_ptr(), p.work_ptr.data_ptr(), p.sms, _ready[dev],
        1, torch.cuda.current_stream(x2.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"marlinw_gemm failed: {rc}")
    return out


def silu_mul(y: torch.Tensor) -> torch.Tensor:
    """y [M, 2D] bf16 (gate | up) -> [M, D] bf16, silu(gate) * up with the
    arithmetic of the compiled SiluAndMul glue (see csrc/silu_mul.cuh)."""
    lib = _load()
    m, n2 = y.shape
    d = n2 // 2
    if (lib is None or y.dtype != torch.bfloat16 or n2 % 16 or y.stride(-1) != 1
            or y.stride(0) % 8 or y.data_ptr() % 16 or not y.is_cuda):
        raise RuntimeError("marlinw.silu_mul: unsupported input")
    out = torch.empty((m, d), dtype=torch.bfloat16, device=y.device)
    rc = lib.marlinw_silu_mul(y.data_ptr(), out.data_ptr(), m, d, y.stride(0),
                              torch.cuda.current_stream(y.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"marlinw_silu_mul failed: {rc}")
    return out


def prepare(layer) -> bool:
    """Static eligibility of a Marlin NVFP4 layer (after prepare_fp4_layer_for_marlin)."""
    if not ENABLED or _load() is None:
        return False
    n = layer.output_size_per_partition
    k = layer.input_size_per_partition
    w, s = layer.weight, layer.weight_scale
    gs = getattr(layer, "weight_global_scale", None)
    return (
        n % 256 == 0 and k % 64 == 0
        and gs is not None and gs.dtype == torch.float32 and gs.numel() == 1
        and w.dtype == torch.int32 and tuple(w.shape) == (k // 16, 2 * n) and w.is_contiguous()
        and s.dtype == torch.float8_e4m3fn and tuple(s.shape) == (k // 16, n)
        and s.is_contiguous() and w.is_cuda and getattr(layer, "bias", None) is None
    )
