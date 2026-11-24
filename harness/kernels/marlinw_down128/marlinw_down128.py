# SPDX-License-Identifier: Apache-2.0
"""128x128-tile MarlinW prompt GEMM (full dequant sharing) for NVFP4 down and attention projections.

The down projection (6656 x 19968) has a second kernel with the same bits: ``libmarlinw_down128t`` runs the same
128x128 tiles with their operands copied by TMA, and splits the last 16 tiles in K into 48 full-size CTAs
(``tma_work_lists``) instead of 48 row-split CTAs. A weight runs it only after its layer passed the caller's
bitwise comparison with it (``admit_tma``).
"""

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
_ALL_ATTN_SHAPES = {(4608, 6656): "qkv", (6656, 4096): "o_proj", (4096, 6656): "gate"}
_raw_attn_shapes = os.environ.get("MARLIN_ATTN128_SHAPES", "qkv,o_proj")
_enabled_attn = set() if _raw_attn_shapes in ("", "0") else set(_raw_attn_shapes.split(","))
ATTN_SHAPES = {k: v for k, v in _ALL_ATTN_SHAPES.items() if v in _enabled_attn}
ATTN_TMA_ENABLED = ATTN_ENABLED and os.environ.get("MARLIN_ATTN_TMA", "1") == "1"
TMA_SHFL_EPI = os.environ.get("MARLIN_TMA_SHFL_EPI", "0") != "0"
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

TMA_ENABLED = ENABLED and os.environ.get("MARLIN_DOWN128_TMA", "1") == "1"
_tma_init = _tma_main = _tma_shape_init = _tma_shape_main = None
if TMA_ENABLED or ATTN_TMA_ENABLED:
    try:
        _tma_library = ctypes.CDLL(str(Path(__file__).with_name("libmarlinw_down128t.so")))
    except OSError:
        TMA_ENABLED = False
        ATTN_TMA_ENABLED = False
    else:
        _tma_init = _tma_library.marlinw_down128_init
        _tma_init.argtypes = [ctypes.POINTER(ctypes.c_int)]
        _tma_init.restype = ctypes.c_int
        _tma_main = _tma_library.marlinw_down128_main
        _tma_main.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _tma_main.restype = ctypes.c_int
        _tma_shape_init = getattr(_tma_library, "marlinw_down128_shape_tma_init", None)
        if _tma_shape_init is not None:
            _tma_shape_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            _tma_shape_init.restype = ctypes.c_int
        _tma_shape_main = getattr(_tma_library, "marlinw_down128_shape_tma_main", None)
        if _tma_shape_main is not None:
            _tma_shape_main.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
            _tma_shape_main.restype = ctypes.c_int
        _tma_set_shfl_epi = getattr(_tma_library, "marlinw_down128_set_shfl_epi", None)
        if _tma_set_shfl_epi is not None:
            _tma_set_shfl_epi.argtypes = [ctypes.c_int]
            _tma_set_shfl_epi.restype = None
            _tma_set_shfl_epi(1 if TMA_SHFL_EPI else 0)
_tma_ready = set()  # devices on which the TMA-fed kernel initialized
_tma_weights = set()  # weight data pointers whose layer passed the comparison with the TMA-fed kernel
_admitted = 0  # layers offered to admit_tma (those on the 128x128 tile)
SPLIT = 16  # tiles the TMA-fed kernel runs as three K pieces (= hand-over states = flag words)
PIECES = (120, 96, 96)  # k-tiles of the first / middle / last piece (the later two run a slower loop copy)


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
    pg = int(os.environ.get("D128_PG", "4"))
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


def _shape_cuts(n: int, k: int, shared_bytes: int):
    """(col256, 64-row chunk) -> k-tile at which the stock partition restarts chains, or -1."""
    launches = marlinw.stock_launches(ROWS, n, k, SMS, shared_bytes)
    if launches != [(0, ROWS, 4, False, (64, 256, 256))]:
        raise ValueError(f"projection {n}x{k} does not select the expected stock tile")
    k_tiles = k // 64
    cut = {}
    for block in marlinw.stock_slices(ROWS, n, k, 4, False, (64, 256, 256), SMS):
        for chunk, col, k_start, k_count, index, count in block:
            if count > 2:
                raise ValueError("more than two K slices per unit")
            if count == 1:
                assert k_start == 0 and k_count == k_tiles
                cut[(col, chunk)] = -1
            elif k_start > 0:
                cut[(col, chunk)] = k_start
    assert len(cut) == (n // 256) * (ROWS // 64)
    return cut


def _down_cuts(shared_bytes: int):
    return _shape_cuts(COLUMNS, INNER, shared_bytes)


def tma_work_lists_shape(n: int, k: int, shared_bytes: int):
    """Work list for the TMA-fed 128x128 kernel across down (6656x19968), qkv (4608x6656),
    o_proj (6656x4096), and gate (4096x6656)."""
    if (n, k) == (COLUMNS, INNER):
        return tma_work_lists(shared_bytes)
    cut = _shape_cuts(n, k, shared_bytes)
    k_tiles = k // 64
    n_col = n // 128
    tma_pg = int(os.environ.get("D128_TMA_PG", "4"))
    order = [(col, pair) for g in range(0, 4, tma_pg) for col in range(n_col) for pair in range(g, g + tma_pg)]
    slots = 0
    slot_of = {}

    def block(tile, k0, count, role, index):
        nonlocal slots
        col, pair = tile
        ca, cb = cut[(col // 2, 2 * pair)], cut[(col // 2, 2 * pair + 1)]
        slot = 0
        if ca >= 0 or cb >= 0:
            if tile not in slot_of:
                slot_of[tile] = slots
                slots += 1
            slot = slot_of[tile]
        return [(128 * pair, col, 8, slot), (ca, cb, count, k0 | role << 10 | index << 12)]

    if (n, k) == (4608, 6656):
        # 144 tiles = 3 exact waves of 48 SMs; zero split tiles, zero K-cuts.
        blocks = [block(tile, 0, k_tiles, 0, 0) for tile in order]
        work = [row for both in blocks for row in both] + [(0, 0, 0, 0)]
        return work, len(blocks), max(slots, 1)
    if (n, k) == (6656, 4096):
        # 208 tiles = 192 whole + 16 split into 3 K pieces (24, 24, 16) -> exact same wave schedule as down.
        split_n = 16
        sizes = (24, 24, 16)
        starts = (0, sizes[0], sizes[0] + sizes[1])
        whole, split = order[:-split_n], order[-split_n:]
        slots = split_n
        whole_blocks = lambda a, b: [block(tile, 0, k_tiles, 0, 0) for tile in whole[a:b]]
        pieces = lambda p: [block(tile, starts[p], sizes[p], p + 1, idx) for idx, tile in enumerate(split)]
        blocks = (pieces(0) + whole_blocks(0, 32) + whole_blocks(32, 48) + pieces(1) + whole_blocks(48, 64)
                  + whole_blocks(64, 96) + pieces(2) + whole_blocks(96, 192))
        work = [row for both in blocks for row in both] + [(0, 0, 0, 0)] * ((split_n + 3) // 4)
        return work, len(blocks), slots
    if (n, k) == (4096, 6656):
        # 128 tiles = 96 whole + 32 split into 3 K pieces (40, 32, 32).
        # Wave schedule keeps all 48 SMs balanced at 272-280 k-tiles (99% SM efficiency) with zero stall.
        split_n = 32
        sizes = (40, 32, 32)
        starts = (0, sizes[0], sizes[0] + sizes[1])
        whole, split = order[:-split_n], order[-split_n:]
        slots = split_n
        whole_blocks = lambda a, b: [block(tile, 0, k_tiles, 0, 0) for tile in whole[a:b]]
        pieces = lambda p, a, b: [block(split[i], starts[p], sizes[p], p + 1, i) for i in range(a, b)]
        blocks = (pieces(0, 0, 32) + whole_blocks(0, 16)
                  + pieces(1, 0, 16) + whole_blocks(16, 32) + whole_blocks(32, 48)
                  + pieces(1, 16, 32) + pieces(2, 0, 16) + pieces(2, 16, 32)
                  + whole_blocks(48, 96))
        work = [row for both in blocks for row in both] + [(0, 0, 0, 0)] * ((split_n + 3) // 4)
        return work, len(blocks), slots
    raise ValueError(f"unsupported TMA shape {n}x{k}")


def tma_work_lists(shared_bytes: int, sizes=PIECES):
    """Work list of the TMA-fed down kernel: K pieces instead of a row-split tail wave.

    208 tiles on 48 SMs are 4 1/3 tiles per SM. The 16 tiles that ``work_lists`` splits by rows (3/3/2 m-blocks,
    far below the tensor-core rate) are split in K here into three pieces of ``sizes`` k-tiles. A piece is a full
    128x128 CTA; a later piece starts from the accumulator registers its predecessor stored (scratch slots
    0..SPLIT-1), so every chain is the chain of the whole tile, HMMA for HMMA, and the stock K cuts inside a piece
    are taken as in a whole tile. The block order puts the three pieces of a tile at different times on different
    SMs (first pieces and 32 whole tiles first, 16 whole tiles behind the first pieces, then the middle pieces, ...,
    the last pieces before the final 96 whole tiles), so every SM runs four whole tiles and one piece. A piece that
    is dispatched before its predecessor finished waits for it (flag words behind the work list: 0 -> 1 -> 2 -> 0,
    written only by the kernel, 0 between calls); every predecessor has a lower block index, so with in-order
    block dispatch it is already running. The first piece is longer because the later pieces, which load a state,
    run a loop copy that ptxas schedules about 8% slower per k-tile.

    Returns (rows: two per block {row0, col128, 8, slot}, {cut_a, cut_b, k_tiles, k0 | role << 10 | index << 12},
    then SPLIT flag words; blocks; scratch slots)."""
    cut = _down_cuts(shared_bytes)
    k_tiles = INNER // 64
    starts = (0, sizes[0], sizes[0] + sizes[1])
    n_col = COLUMNS // 128
    assert sum(sizes) == k_tiles and all(v % 8 == 0 for v in starts) and starts[2] < 1024
    assert n_col * 4 == 4 * SMS + SPLIT
    tma_pg = int(os.environ.get("D128_TMA_PG", "4"))
    order = [(col, pair) for g in range(0, 4, tma_pg) for col in range(n_col) for pair in range(g, g + tma_pg)]
    whole, split = order[:-SPLIT], order[-SPLIT:]
    slots = SPLIT  # scratch: SPLIT hand-over states, then one slot per tile with a stock cut
    slot_of = {}

    def block(tile, k0, count, role, index):
        nonlocal slots
        col, pair = tile
        ca, cb = cut[(col // 2, 2 * pair)], cut[(col // 2, 2 * pair + 1)]
        slot = 0
        if ca >= 0 or cb >= 0:
            if tile not in slot_of:
                slot_of[tile] = slots
                slots += 1
            slot = slot_of[tile]
        return [(128 * pair, col, 8, slot), (ca, cb, count, k0 | role << 10 | index << 12)]

    def whole_blocks(first, last):
        return [block(tile, 0, k_tiles, 0, 0) for tile in whole[first:last]]

    def pieces(n):
        return [block(tile, starts[n], sizes[n], n + 1, index) for index, tile in enumerate(split)]

    blocks = (pieces(0) + whole_blocks(0, 32) + whole_blocks(32, 48) + pieces(1) + whole_blocks(48, 64)
              + whole_blocks(64, 96) + pieces(2) + whole_blocks(96, 192))
    assert len(blocks) == len(whole) + 3 * SPLIT
    work = [row for both in blocks for row in both] + [(0, 0, 0, 0)] * ((SPLIT + 3) // 4)
    return work, len(blocks), slots


@dataclass(frozen=True, slots=True)
class Prepared:
    work: torch.Tensor
    grid: int
    slots: int
    active_blocks_per_sm: int
    n: int = COLUMNS
    k: int = INNER
    tma_work: torch.Tensor | None = None  # work list of the TMA-fed down kernel, then its SPLIT flag words
    tma_grid: int = 0
    tma_slots: int = 0


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
    plan = Prepared(torch.tensor(work, dtype=torch.int32, device=device), grid, slots, active.value, n, k)
    if (n, k) == (COLUMNS, INNER) and _tma_init is not None:
        with torch.cuda.device(device):
            status = _tma_init(ctypes.byref(active))
        if status == 0 and active.value == 1:  # one CTA per SM: the waiting pieces (at most 32) never fill the GPU
            work, grid, slots = tma_work_lists(props.shared_memory_per_block_optin)
            plan = Prepared(plan.work, plan.grid, plan.slots, plan.active_blocks_per_sm, n, k,
                            torch.tensor(work, dtype=torch.int32, device=device), grid, slots)
            _tma_ready.add(device)
    elif ATTN_TMA_ENABLED and _tma_shape_init is not None and (n, k) in _ALL_ATTN_SHAPES:
        with torch.cuda.device(device):
            status = _tma_shape_init(n, k, ctypes.byref(active))
        if status == 0 and active.value == 1:
            work, grid, slots = tma_work_lists_shape(n, k, props.shared_memory_per_block_optin)
            plan = Prepared(plan.work, plan.grid, plan.slots, plan.active_blocks_per_sm, n, k,
                            torch.tensor(work, dtype=torch.int32, device=device), grid, slots)
    return plan


def prepare(device: torch.device) -> Prepared:
    return prepare_shape(device, COLUMNS, INNER)


def gemm(x, weight, scales, global_scale, p: Prepared, tma: bool = False):
    if (
        x.shape != (ROWS, p.k)
        or x.dtype != torch.bfloat16
        or x.device != p.work.device
        or x.stride(1) != 1
        or x.stride(0) % 8
        or x.data_ptr() % 16
    ):
        return None
    if tma and p.tma_work is None:
        return None
    out = x.new_empty((ROWS, p.n))
    scratch = _get_scratch(x.device, p.tma_slots if tma else p.slots)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    with torch.cuda.device(x.device):
        if tma:
            if (p.n, p.k) == (COLUMNS, INNER):
                status = _tma_main(
                    x.data_ptr(), weight.data_ptr(), out.data_ptr(), scratch.data_ptr(),
                    scales.data_ptr(), global_scale.data_ptr(), p.tma_work.data_ptr(), p.tma_grid,
                    x.stride(0), stream)
            else:
                status = _tma_shape_main(
                    x.data_ptr(), weight.data_ptr(), out.data_ptr(), scratch.data_ptr(),
                    scales.data_ptr(), global_scale.data_ptr(), p.tma_work.data_ptr(), p.tma_grid,
                    p.n, p.k, x.stride(0), stream)
        elif (p.n, p.k) == (COLUMNS, INNER):
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


def gemm_shape(x, weight, scales, global_scale, n: int, k: int, tma: bool | None = None):
    key = (x.device, n, k)
    plan = _plans.get(key)
    if plan is None:
        plan = _plans[key] = prepare_shape(x.device, n, k)
    use_tma = (plan.tma_work is not None and ATTN_TMA_ENABLED) if tma is None else tma
    return gemm(x, weight, scales, global_scale, plan, use_tma)


def _plan(device):
    key = (device, COLUMNS, INNER)
    plan = _plans.get(key)
    if plan is None:
        plan = _plans[key] = prepare_shape(device, COLUMNS, INNER)
    return plan


def uses_tma(weight) -> bool:
    """True if calls on this weight run the TMA-fed kernel: its layer passed the comparison, or (a buffer no layer
    was compared on, i.e. a prefill placement candidate) every layer on the 128x128 tile did."""
    if not _tma_weights:
        return False
    return weight.data_ptr() in _tma_weights or len(_tma_weights) == _admitted


def gemm_auto(x, weight, scales, global_scale):
    """Down projection with the per-device cached work list (None when x does not fit the kernel)."""
    plan = _plan(x.device)
    return gemm(x, weight, scales, global_scale, plan, x.device in _tma_ready and uses_tma(weight))


def gemm_tma(x, weight, scales, global_scale):
    """The TMA-fed down kernel regardless of admission (None when it is unavailable or x does not fit)."""
    plan = _plan(x.device)
    if x.device not in _tma_ready:
        return None
    return gemm(x, weight, scales, global_scale, plan, True)


def admit_tma(weight, passed: bool) -> None:
    """Record the caller's bitwise comparison of gemm_tma on this layer's weight."""
    global _admitted
    _admitted += 1
    if passed:
        _tma_weights.add(weight.data_ptr())


def moved(old_ptr: int, new_ptr: int) -> None:
    """A placed weight moved to another buffer with the same bytes: it keeps its kernel."""
    if old_ptr in _tma_weights:
        _tma_weights.discard(old_ptr)
        _tma_weights.add(new_ptr)


def counts() -> tuple[int, int]:
    """Layers on the 128x128 tile that were offered the TMA-fed kernel, and those that passed."""
    return _admitted, len(_tma_weights)
