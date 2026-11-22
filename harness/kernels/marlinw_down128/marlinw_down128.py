# SPDX-License-Identifier: Apache-2.0
"""128x128-tile MarlinW down projection (full dequant sharing); leftover units use the record tile."""

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
_plans = {}

_library = ctypes.CDLL(str(Path(__file__).with_name("libmarlinw_down128.so")))
_init = _library.marlinw_down128_init
_init.argtypes = [ctypes.POINTER(ctypes.c_int)]
_init.restype = ctypes.c_int
_main = _library.marlinw_down128_main
_main.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_main.restype = ctypes.c_int


def work_lists(shared_bytes: int):
    launches = marlinw.stock_launches(ROWS, COLUMNS, INNER, SMS, shared_bytes)
    config = (64, 256, 256)
    if launches != [(0, ROWS, 4, False, config)]:
        raise ValueError("down projection does not select the expected stock tile")
    stock = marlinw.stock_slices(ROWS, COLUMNS, INNER, 4, False, config, SMS)
    k_tiles = INNER // 64
    cut = {}  # (col256, chunk) -> k-tile where the stock chain restarts, or -1
    for block in stock:
        for chunk, col, k_start, k_count, index, count in block:
            if count > 2:
                raise ValueError("more than two K slices per unit")
            if count == 1:
                assert k_start == 0 and k_count == k_tiles
                cut[(col, chunk)] = -1
            elif k_start > 0:
                cut[(col, chunk)] = k_start
    assert len(cut) == 208
    assert all(c < 0 or c % 4 == 0 for c in cut.values())
    tail_tiles = int(os.environ.get("D128_TAIL", "16"))
    pg = int(os.environ.get("D128_PG", "2"))
    n_col = COLUMNS // 128
    order = [(col, pair) for g in range(0, 4, pg) for col in range(n_col) for pair in range(g, g + pg)]
    items = []  # (row0, col128, mb, cut_a, cut_b)
    tail = []
    for idx, (col, pair) in enumerate(order):
        c0, c1 = cut[(col // 2, 2 * pair)], cut[(col // 2, 2 * pair + 1)]
        base = 128 * pair
        if idx >= len(order) - tail_tiles:
            for r0, mb in ((0, 3), (48, 3), (96, 2)):
                chunk0 = (base + r0) // 64 - 2 * pair
                ca = (c0, c1)[chunk0]
                cb = c1 if chunk0 == 0 else -1
                tail.append((base + r0, col, mb, ca, cb))
        else:
            items.append((base, col, 8, c0, c1))
    items += tail
    work, slots = [], 0
    for row0, col, mb, ca, cb in items:
        has = ca >= 0 or cb >= 0
        work.append((row0, col, mb, slots if has else 0))
        work.append((ca, cb, int(os.environ.get("D128_ITERS", k_tiles)), 0))
        slots += has
    return work, len(items), slots


@dataclass(frozen=True, slots=True)
class Prepared:
    work: torch.Tensor
    grid: int
    slots: int
    active_blocks_per_sm: int


def prepare(device: torch.device) -> Prepared:
    props = torch.cuda.get_device_properties(device)
    if props.multi_processor_count != SMS or torch.cuda.get_device_capability(device) != (12, 1):
        raise ValueError("the down tile trial requires a 48-SM SM121 device")
    work, grid, slots = work_lists(props.shared_memory_per_block_optin)
    active = ctypes.c_int()
    with torch.cuda.device(device):
        status = _init(ctypes.byref(active))
    if status:
        raise RuntimeError(f"MarlinW down128 initialization returned CUDA error {status}")
    grid = int(os.environ.get('D128_GRID', grid))
    return Prepared(torch.tensor(work, dtype=torch.int32, device=device), grid, slots, active.value)


def gemm(x, weight, scales, global_scale, p: Prepared):
    if (
        x.shape != (ROWS, INNER)
        or x.dtype != torch.bfloat16
        or x.device != p.work.device
        or x.stride(1) != 1
        or x.stride(0) % 8
        or x.data_ptr() % 16
    ):
        return None
    out = x.new_empty((ROWS, COLUMNS))
    scratch = x.new_empty((max(p.slots, 1), 256, 32, 4), dtype=torch.float32)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    with torch.cuda.device(x.device):
        status = _main(
            x.data_ptr(), weight.data_ptr(), out.data_ptr(), scratch.data_ptr(),
            scales.data_ptr(), global_scale.data_ptr(), p.work.data_ptr(), p.grid,
            x.stride(0), stream)
    if status:
        raise RuntimeError(f"MarlinW down128 launch returned CUDA error {status}")
    return out


def gemm_auto(x, weight, scales, global_scale):
    """gemm with the per-device cached work list (None when x does not fit the kernel)."""
    plan = _plans.get(x.device)
    if plan is None:
        plan = _plans[x.device] = prepare(x.device)
    return gemm(x, weight, scales, global_scale, plan)
