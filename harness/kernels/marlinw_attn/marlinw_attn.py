# SPDX-License-Identifier: Apache-2.0
"""Narrow MarlinW for the 512-row NVFP4 attention projections (qkv, gate, o).

At 512 rows ops.marlin_gemm (vLLM v0.27.1) runs these L2-sized layers as one
launch of 48 blocks, one per SM, with the (thread_k 64, thread_n 256, 256
threads) config, four stages and 101 KB of shared memory. ``work_lists``
replays that partition (marlinw.stock_launches / stock_slices) and hands every
stock slice, cut into its two 128-column halves, to its own block of the
marlinw_down kernel (64x128 tile, 128 threads, three stages, 45 KB, two blocks
per SM). Each slice keeps its 64-row chunk, K range, slice_idx and
slice_count, and each cut (tile, half) gets its own FP32 scratch slot and lock,
so every output element is the same K16 MMA chains, the same two-warp FP32
sum, the same FP32 split-K reduction order and the same BF16 epilogue.

Blocks are listed for the hardware's in-order dispatch: the unsplit tiles
first, by (256-column tile, chunk, half) so the chunks that share a weight
tile run together, then the slices of the cut tiles, longest first. A slice
with slice_idx > 0 waits (after its K loop) for its predecessor; at most
``forward_waits`` blocks (46 for gate, 40 for o, 0 for qkv) can wait for a
block listed after them, fewer than the resident blocks, so every wait ends.

``MARLIN_NARROW_ATTN=0`` disables it.
"""

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path

import torch

import marlinw

ROWS = 512
SMS = 48
CONFIG = (64, 256, 256)
SHAPES = {(4608, 6656): "qkv", (4096, 6656): "gate", (6656, 4096): "o"}
ENABLED = os.environ.get("MARLIN_NARROW_ATTN", "1") == "1"

_library = ctypes.CDLL(str(Path(__file__).with_name("libmarlinw_attn.so")))
_init = _library.marlinw_attn_init
_init.argtypes = [ctypes.POINTER(ctypes.c_int)]
_init.restype = ctypes.c_int
_launch = _library.marlinw_attn_gemm
_launch.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
_launch.restype = ctypes.c_int


def work_lists(n: int, k: int, shared_bytes: int):
    """(work items, per-block starts, scratch slots); one item per block."""
    launches = marlinw.stock_launches(ROWS, n, k, SMS, shared_bytes)
    if launches != [(0, ROWS, 4, False, CONFIG)]:
        raise ValueError(f"attention projection {n}x{k} does not select the expected stock tile")
    stock = marlinw.stock_slices(ROWS, n, k, 4, False, CONFIG, SMS)
    cut_tiles = sorted({(c, col) for blk in stock for (c, col, _k0, _kn, _i, cnt) in blk if cnt > 1})
    slot_of = {}
    for tile in cut_tiles:
        for half in range(2):
            slot_of[(tile, half)] = len(slot_of)
    full, cut = [], []
    for blk in stock:
        for (c, col, k0, kn, idx, cnt) in blk:
            for half in range(2):
                item = (c, 2 * col + half, k0, kn, idx, cnt,
                        slot_of[((c, col), half)] if cnt > 1 else 0, 0)
                (full if cnt == 1 else cut).append(item)
    full.sort(key=lambda it: (it[1] // 2, it[0], it[1] % 2))
    cut.sort(key=lambda it: (-it[3], it[4], it[1] // 2, it[0], it[1] % 2))
    work = full + cut
    starts = list(range(len(work) + 1))
    _check(n, k, stock, work, len(slot_of))
    return work, starts, len(slot_of)


def _check(n, k, stock, work, slots):
    """Same slices as stock (per half), whole K per tile, one slot per cut tile."""
    mine = {}
    for it in work:
        key = (it[0], it[1] // 2, it[2], it[3], it[4], it[5])
        mine[key] = mine.get(key, 0) + 1
    ref = {}
    for blk in stock:
        for s in blk:
            ref[tuple(s)] = ref.get(tuple(s), 0) + 2
    if mine != ref:
        raise ValueError("narrow attention plan changed the stock slices")
    tiles = {}
    for it in work:
        tiles.setdefault((it[0], it[1]), []).append(it)
    if len(tiles) != (ROWS // 64) * (n // 128):
        raise ValueError("narrow attention plan misses an output tile")
    used = set()
    for items in tiles.values():
        pos = 0
        for it in sorted(items, key=lambda it: it[2]):
            if it[2] != pos:
                raise ValueError("narrow attention plan has a K gap")
            pos += it[3]
        if pos != k // 64 or sorted(it[4] for it in items) != list(range(len(items))):
            raise ValueError("narrow attention plan has a bad K partition")
        if len(items) > 1:
            slot = {it[6] for it in items}
            if len(slot) != 1 or slot & used:
                raise ValueError("narrow attention plan shares a scratch slot")
            used |= slot
    if len(used) != slots:
        raise ValueError("narrow attention plan has unused scratch slots")


def forward_waits(work) -> int:
    """Blocks whose predecessor slice is listed after them."""
    pos = {(it[0], it[1], it[4]): b for b, it in enumerate(work)}
    return sum(1 for b, it in enumerate(work) if it[4] > 0 and pos[(it[0], it[1], it[4] - 1)] > b)


@dataclass(frozen=True, slots=True)
class Prepared:
    kind: str
    work: torch.Tensor
    work_ptr: torch.Tensor
    scratch_slots: int
    forward_waits: int
    active_blocks_per_sm: int


_plans: dict = {}
_scratch: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_scratch(device: torch.device, slots: int) -> tuple[torch.Tensor, torch.Tensor]:
    slots = max(int(slots), 1)
    key = (device.index or 0, slots)
    buf = _scratch.get(key)
    if buf is None:
        buf = (
            torch.empty((slots, 64, 128), dtype=torch.float32, device=device),
            torch.zeros((slots,), dtype=torch.int32, device=device),
        )
        _scratch[key] = buf
    return buf


def prepare(device: torch.device, n: int, k: int) -> Prepared:
    kind = SHAPES.get((n, k))
    if kind is None:
        raise ValueError(f"no narrow attention plan for {n}x{k}")
    props = torch.cuda.get_device_properties(device)
    if props.multi_processor_count != SMS or torch.cuda.get_device_capability(device) != (12, 1):
        raise ValueError("the narrow attention plan requires a 48-SM SM121 device")
    key = (props.name, device.index, n, k)
    plan = _plans.get(key)
    if plan is not None:
        return plan
    work, starts, slots = work_lists(n, k, props.shared_memory_per_block_optin)
    active = ctypes.c_int()
    with torch.cuda.device(device):
        status = _init(ctypes.byref(active))
    if status:
        raise RuntimeError(f"narrow attention initialization returned CUDA error {status}")
    waits = forward_waits(work)
    if active.value < 2 or waits >= active.value * SMS:
        raise ValueError(f"{active.value} resident blocks per SM cannot hold {waits} waiting blocks")
    plan = Prepared(kind, torch.tensor(work, dtype=torch.int32, device=device),
                    torch.tensor(starts, dtype=torch.int32, device=device),
                    slots, waits, active.value)
    _plans[key] = plan
    return plan


def gemm(x, weight, scales, global_scale, work, work_ptr, scratch_slots):
    """x [512, K] bf16 -> [512, N] bf16 equal to ops.marlin_gemm bit for bit, or None."""
    n = scales.shape[1]
    k = 16 * scales.shape[0]
    if (
        x.dim() != 2
        or x.shape != (ROWS, k)
        or (n, k) not in SHAPES
        or x.dtype != torch.bfloat16
        or x.device != work.device
        or x.stride(1) != 1
        or x.stride(0) % 8
        or x.stride(0) < k
        or x.data_ptr() % 16
    ):
        return None
    out = x.new_empty((ROWS, n))
    partial, locks = _get_scratch(x.device, scratch_slots)
    with torch.cuda.device(x.device):
        status = _launch(
            x.data_ptr(), weight.data_ptr(), out.data_ptr(), partial.data_ptr(),
            scales.data_ptr(), global_scale.data_ptr(), locks.data_ptr(),
            work.data_ptr(), work_ptr.data_ptr(), ROWS, n, k, x.stride(0),
            work_ptr.numel() - 1, torch.cuda.current_stream(x.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"narrow attention launch returned CUDA error {status}")
    return out
