# SPDX-License-Identifier: Apache-2.0
"""Narrow MarlinW down projection with the original split-K order."""

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
ENABLED = os.environ.get("MARLIN_NARROW_DOWN", "1") == "1"

_library = ctypes.CDLL(str(Path(__file__).with_name("libmarlinw_down.so")))
_init = _library.marlinw_down_init
_init.argtypes = [ctypes.POINTER(ctypes.c_int)]
_init.restype = ctypes.c_int
_launch = _library.marlinw_down_gemm
_launch.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int, ctypes.c_void_p]
_launch.restype = ctypes.c_int


def work_lists(shared_bytes: int):
    launches = marlinw.stock_launches(ROWS, COLUMNS, INNER, SMS, shared_bytes)
    config = (64, 256, 256)
    if launches != [(0, ROWS, 4, False, config)]:
        raise ValueError("down projection does not select the expected stock tile")
    stock = marlinw.stock_slices(ROWS, COLUMNS, INNER, 4, False, config, SMS)
    tiles: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for block in stock:
        for chunk, col, k_start, k_count, index, count in block:
            tiles.setdefault((col, chunk), []).append((index, k_start, k_count, count))

    starts = [0]
    work = []
    slots = 0
    for (col, chunk), slices in sorted(tiles.items()):
        for half in range(2):
            slot = slots if len(slices) > 1 else 0
            for index, k_start, k_count, count in sorted(slices):
                work.append((chunk, col * 2 + half, k_start, k_count, index, count, slot, 0))
            starts.append(len(work))
            if len(slices) > 1:
                slots += 1
    return work, starts, slots


@dataclass(frozen=True, slots=True)
class Prepared:
    work: torch.Tensor
    work_ptr: torch.Tensor
    scratch_slots: int
    active_blocks_per_sm: int


def prepare(device: torch.device) -> Prepared:
    props = torch.cuda.get_device_properties(device)
    if props.multi_processor_count != SMS or torch.cuda.get_device_capability(device) != (12, 1):
        raise ValueError("the down tile trial requires a 48-SM SM121 device")
    work, starts, slots = work_lists(props.shared_memory_per_block_optin)
    active = ctypes.c_int()
    with torch.cuda.device(device):
        status = _init(ctypes.byref(active))
    if status:
        raise RuntimeError(f"MarlinW down initialization returned CUDA error {status}")
    return Prepared(torch.tensor(work, dtype=torch.int32, device=device),
                    torch.tensor(starts, dtype=torch.int32, device=device),
                    slots, active.value)


_SCRATCH_CACHE: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_scratch(device: torch.device, scratch_slots: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, scratch_slots)
    cached = _SCRATCH_CACHE.get(key)
    if cached is not None:
        return cached
    partial = torch.empty((scratch_slots, 64, 128), dtype=torch.float32, device=device)
    locks = torch.zeros((scratch_slots,), dtype=torch.int32, device=device)
    _SCRATCH_CACHE[key] = (partial, locks)
    return partial, locks


def gemm(x, weight, scales, global_scale, work, work_ptr, scratch_slots):
    if (
        x.shape != (ROWS, INNER)
        or x.dtype != torch.bfloat16
        or x.device != work.device
        or x.stride(1) != 1
        or x.stride(0) % 8
        or x.data_ptr() % 16
    ):
        return None
    out = x.new_empty((ROWS, COLUMNS))
    partial, locks = _get_scratch(x.device, scratch_slots)
    with torch.cuda.device(x.device):
        status = _launch(
            x.data_ptr(), weight.data_ptr(), out.data_ptr(), partial.data_ptr(),
            scales.data_ptr(), global_scale.data_ptr(), locks.data_ptr(),
            work.data_ptr(), work_ptr.data_ptr(), x.stride(0),
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"MarlinW down launch returned CUDA error {status}")
    return out
