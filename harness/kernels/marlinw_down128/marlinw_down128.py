# SPDX-License-Identifier: Apache-2.0
"""128x128-tile MarlinW prompt GEMM (full dequant sharing) for NVFP4 down and attention projections."""

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path

import torch

import marlinw

ROWS = 512
COLUMNS = 6656
INNER = 19968
SMS = 48
ENABLED = os.environ.get("MARLIN_DOWN128", "1") == "1"
ATTN_ENABLED = os.environ.get("MARLIN_ATTN128", "1") == "1"
ATTN_SHAPES = {(4608, 6656): "qkv"}
_DEFAULT_TAIL = {(6656, 19968): 16, (4608, 6656): 0, (4096, 6656): 8, (6656, 4096): 16}
_plans: dict = {}
_scratch_cache: dict[tuple[int, int], torch.Tensor] = {}

_library = ctypes.CDLL(str(Path(__file__).with_name("libmarlinw_down128.so")))
_init = _library.marlinw_down128_init
_init.argtypes = [ctypes.POINTER(ctypes.c_int)]
_init.restype = ctypes.c_int
_main = _library.marlinw_down128_main
_main.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_main.restype = ctypes.c_int
_shape_init = _library.marlinw_down128_shape_init
_shape_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_shape_init.restype = ctypes.c_int
_shape_main = _library.marlinw_down128_shape_main
_shape_main.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
_shape_main.restype = ctypes.c_int


def _get_scratch(device: torch.device, slots: int) -> torch.Tensor:
    slots = max(int(slots), 1)
    key = (device.index or 0, slots)
    buf = _scratch_cache.get(key)
    if buf is None:
        buf = torch.empty((slots, 256, 32, 4), dtype=torch.float32, device=device)
        _scratch_cache[key] = buf
    return buf


def work_lists_shape(n: int, k: int, shared_bytes: int):
    launches = marlinw.stock_launches(ROWS, n, k, SMS, shared_bytes)
    config = (64, 256, 256)
    if launches != [(0, ROWS, 4, False, config)]:
        raise ValueError(f"projection {n}x{k} does not select the expected stock tile")
    stock = marlinw.stock_slices(ROWS, n, k, 4, False, config, SMS)
    k_tiles = k // 64
    cuts_map: dict[tuple[int, int], list[int]] = {}
    for block in stock:
        for chunk, col, k_start, k_count, index, count in block:
            if count > 3:
                raise ValueError("more than three K slices per unit")
            lst = cuts_map.setdefault((col, chunk), [])
            if count == 1:
                assert k_start == 0 and k_count == k_tiles
            elif k_start > 0:
                lst.append(k_start)
    assert len(cuts_map) == (n // 256) * (ROWS // 64)
    cut = {}  # (col256, chunk) -> (cut1, cut2)
    for key, lst in cuts_map.items():
        lst.sort()
        c1 = lst[0] if len(lst) >= 1 else -1
        c2 = lst[1] if len(lst) >= 2 else -1
        cut[key] = (c1, c2)
    default_tail = _DEFAULT_TAIL[(n, k)]
    tail_tiles = int(os.environ.get("D128_TAIL", str(default_tail))) if (n, k) == (COLUMNS, INNER) else default_tail
    pg = int(os.environ.get("D128_PG", "2"))
    n_col = n // 128
    order = [(col, pair) for g in range(0, 4, pg) for col in range(n_col) for pair in range(g, g + pg)]
    items = []  # (row0, col128, mb, ca, cb, ca2, cb2)
    tail = []
    for idx, (col, pair) in enumerate(order):
        (c0, c0_2), (c1, c1_2) = cut[(col // 2, 2 * pair)], cut[(col // 2, 2 * pair + 1)]
        base = 128 * pair
        if idx >= len(order) - tail_tiles:
            for r0, mb in ((0, 3), (48, 3), (96, 2)):
                chunk0 = (base + r0) // 64 - 2 * pair
                ca, ca2 = (c0, c0_2) if chunk0 == 0 else (c1, c1_2)
                cb, cb2 = (c1, c1_2) if chunk0 == 0 else (-1, -1)
                tail.append((base + r0, col, mb, ca, cb, ca2, cb2))
        else:
            items.append((base, col, 8, c0, c1, c0_2, c1_2))
    items += tail
    work, slots = [], 0
    iters_val = int(os.environ.get("D128_ITERS", k_tiles)) if (n, k) == (COLUMNS, INNER) else k_tiles
    for row0, col, mb, ca, cb, ca2, cb2 in items:
        has1 = ca >= 0 or cb >= 0
        has2 = ca2 >= 0 or cb2 >= 0
        w1w = (ca2 & 0xFFFF) | ((cb2 & 0xFFFF) << 16)
        if w1w >= 0x80000000:
            w1w -= 0x100000000
        work.append((row0, col, mb, slots if has1 else 0))
        work.append((ca, cb, iters_val, w1w))
        slots += 2 if has2 else (1 if has1 else 0)
    return work, len(items), slots


def work_lists(shared_bytes: int):
    return work_lists_shape(COLUMNS, INNER, shared_bytes)


@dataclass(frozen=True, slots=True)
class Prepared:
    work: torch.Tensor
    grid: int
    slots: int
    active_blocks_per_sm: int
    n: int = COLUMNS
    k: int = INNER


def prepare_shape(device: torch.device, n: int, k: int) -> Prepared:
    props = torch.cuda.get_device_properties(device)
    if props.multi_processor_count != SMS or torch.cuda.get_device_capability(device) != (12, 1):
        raise ValueError("the 128x128 tile kernel requires a 48-SM SM121 device")
    work, grid, slots = work_lists_shape(n, k, props.shared_memory_per_block_optin)
    active = ctypes.c_int()
    with torch.cuda.device(device):
        status = _shape_init(n, k, ctypes.byref(active))
    if status:
        raise RuntimeError(f"MarlinW 128x128 ({n}x{k}) initialization returned CUDA error {status}")
    if (n, k) == (COLUMNS, INNER):
        grid = int(os.environ.get("D128_GRID", grid))
    return Prepared(torch.tensor(work, dtype=torch.int32, device=device), grid, slots, active.value, n, k)


def prepare(device: torch.device) -> Prepared:
    return prepare_shape(device, COLUMNS, INNER)


def gemm(x, weight, scales, global_scale, p: Prepared):
    if (
        x.shape != (ROWS, p.k)
        or x.dtype != torch.bfloat16
        or x.device != p.work.device
        or x.stride(1) != 1
        or x.stride(0) % 8
        or x.data_ptr() % 16
    ):
        return None
    out = x.new_empty((ROWS, p.n))
    scratch = _get_scratch(x.device, p.slots)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    with torch.cuda.device(x.device):
        if (p.n, p.k) == (COLUMNS, INNER):
            status = _main(
                x.data_ptr(), weight.data_ptr(), out.data_ptr(), scratch.data_ptr(),
                scales.data_ptr(), global_scale.data_ptr(), p.work.data_ptr(), p.grid,
                x.stride(0), stream)
        else:
            status = _shape_main(
                x.data_ptr(), weight.data_ptr(), out.data_ptr(), scratch.data_ptr(),
                scales.data_ptr(), global_scale.data_ptr(), p.work.data_ptr(), p.grid,
                p.n, p.k, x.stride(0), stream)
    if status:
        raise RuntimeError(f"MarlinW 128x128 ({p.n}x{p.k}) launch returned CUDA error {status}")
    return out


def gemm_shape(x, weight, scales, global_scale, n: int, k: int):
    key = (x.device, n, k)
    plan = _plans.get(key)
    if plan is None:
        plan = _plans[key] = prepare_shape(x.device, n, k)
    return gemm(x, weight, scales, global_scale, plan)


def gemm_auto(x, weight, scales, global_scale):
    """gemm with the per-device cached work list (None when x does not fit the kernel)."""
    return gemm_shape(x, weight, scales, global_scale, COLUMNS, INNER)
