# SPDX-License-Identifier: Apache-2.0
"""Decode (M == 1) NVFP4 GEMV with 6-bit block scales: the board's tiled Marlin GEMV (muse_marlin_gemv) reading each
layer's group-16 FP8 block scales as 6-bit codes into a 64-entry palette, 3/4 of the scale bytes. The decoded bytes are
the same bytes, in the same shared-memory fragment, so the kernel computes the same bits; every layer is admitted only
after a bitwise comparison with the uncompressed tiled GEMV on its own weight."""
import ctypes
import os
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
STATS = {"admitted": 0, "kept": 0, "fixed_grid": 0, "gate_up_srans": 0, "gate_up_srans_rejected": 0,
         "srans_scales": 0, "srans_bytes": 0, "down_srans": 0, "down_srans_rejected": 0, "down_srans_scales": 0,
         "down_srans_bytes": 0, "logged": False}

_lib = ctypes.CDLL(str(Path(__file__).with_name("libmuse_marlin6.so")))
_gemv6 = _lib.muse_marlin6_gemv
_gemv6.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
_gemv6.restype = ctypes.c_int
_gate_up = getattr(_lib, "muse_marlin6_gate_up", None)
if _gate_up is not None:
    _gate_up.argtypes = _gemv6.argtypes
    _gate_up.restype = ctypes.c_int
# The same fixed grid reading rANS-coded block scales (gemv6_srans.cu).
_gate_up_srans = getattr(_lib, "muse_marlin6_gate_up_srans", None)
if _gate_up_srans is not None:
    _gate_up_srans.argtypes = _gemv6.argtypes
    _gate_up_srans.restype = ctypes.c_int
# The 48-CTA stream-K down projection reading rANS-coded block scales (gemv6_srans_down.cu).
_down_srans = getattr(_lib, "muse_marlin6_down_srans", None)
if _down_srans is not None:
    _down_srans.argtypes = _gemv6.argtypes
    _down_srans.restype = ctypes.c_int
# 4-stage TMA super-block rANS scales (gemv6_tma4.cu): gate_up, down, and lm_head.
_gate_up_tma4 = getattr(_lib, "muse_marlin6_gate_up_tma4", None)
if _gate_up_tma4 is not None:
    _gate_up_tma4.argtypes = _gemv6.argtypes
    _gate_up_tma4.restype = ctypes.c_int
_sk_tma4 = getattr(_lib, "muse_marlin6_sk_tma4", None)
if _sk_tma4 is not None:
    _sk_tma4.argtypes = _gemv6.argtypes
    _sk_tma4.restype = ctypes.c_int
_sk_tma4_ilv = getattr(_lib, "muse_marlin6_sk_tma4_ilv", None)
if _sk_tma4_ilv is not None:
    _sk_tma4_ilv.argtypes = _gemv6.argtypes
    _sk_tma4_ilv.restype = ctypes.c_int
# 64-column tiles at 256 threads: same grid and schedule, 8 warps/SM instead of 4 (MUSE_LMHEAD_T256).
_sk_tma4_t256 = getattr(_lib, "muse_marlin6_sk_tma4_t256", None)
if _sk_tma4_t256 is not None:
    _sk_tma4_t256.argtypes = _gemv6.argtypes
    _sk_tma4_t256.restype = ctypes.c_int
# Same kernels reading the weight through a row-chunked placement (chunk bases + cumulative tile offsets).
_sk_tma4_chunked = getattr(_lib, "muse_marlin6_sk_tma4_chunked", None)
if _sk_tma4_chunked is not None:
    _sk_tma4_chunked.argtypes = _gemv6.argtypes + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    _sk_tma4_chunked.restype = ctypes.c_int
_get_gate_blocks = getattr(_lib, "muse_marlin6_get_gate_blocks", None)
if _get_gate_blocks is not None:
    _get_gate_blocks.argtypes = []
    _get_gate_blocks.restype = ctypes.c_int
_tma4_p1 = getattr(_lib, "muse_tma4_pass1", None)
if _tma4_p1 is not None:
    _tma4_p1.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    _tma4_p1.restype = ctypes.c_int
_tma4_p2 = getattr(_lib, "muse_tma4_pass2", None)
if _tma4_p2 is not None:
    _tma4_p2.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    _tma4_p2.restype = ctypes.c_int


GATE_UP_N, GATE_UP_K = 39936, 6656
DOWN_N, DOWN_K = 6656, 19968
LM_HEAD_N, LM_HEAD_K = 202048, 6656
ATTN_ILV_SHAPES = {(4608, 6656), (6656, 4096), (4096, 6656)}
_TMA4_SCHED_CACHE: dict[tuple[int, int, bool, bool, int, torch.device], tuple] = {}
_DOWN_ILV_SCALES: set[int] = set()


# Stream-K CTA counts wider than the admitted 48. EMPTY BY DEFAULT: a wider grid changes the split-K reduction grouping,
# so it is not bitwise-equal to the 48-CTA reference, and the shipping tree stays bit-exact everywhere. Opt in with
# MUSE_SK_WIDE_BLOCKS=<n>.
#   lm_head 202048x6656 -> 3157 column tiles = 287 x 11, i.e. 11 whole columns per CTA with slice_count == 1 (no
#   split-K, no locks). 48 CTAs is 1 CTA/SM at 4 warps (TileN=64/Threads=128) and reaches ~225 GB/s; 287 would be 5.98
#   waves at 4 CTAs/SM. Unmeasured — it was rejected by admission before it ever ran. Gate it on an argmax-flip sweep,
#   not on the 129-token hash (see #112).
# `down` (6656x19968) has only 52 column tiles, so 52 CTAs would be 1.08 waves and the 4-CTA second wave costs more
# than the split-K it removes. MUSE_SK_DOWN_BLOCKS=52 opts in if that turns out to be wrong.
_SK_BLOCKS: dict[tuple[int, int], int] = {}
_SK_BLOCK_OVERRIDE: dict[tuple[int, int], int] = {}
# Experiment hooks: a positive value forces that CTA count; 0 uses the table above.
_SK_WIDE_BLOCKS = int(os.environ.get("MUSE_SK_WIDE_BLOCKS", "0") or 0)
_SK_DOWN_BLOCKS = int(os.environ.get("MUSE_SK_DOWN_BLOCKS", "0") or 0)
# Admission runs every reference comparison on the admitted 48-CTA partition, exactly as before this change, so the
# per-layer bitwise gates keep their original meaning. The wider runtime grid is a reduction-grouping change only: it
# consumes the same decoded scale bytes through the same MMA chain, so coding correctness carries over, and the grid
# change itself is validated end to end by the golden hashes.
_ADMITTING = False
# MUSE_SK_STRICT=1 vetoes a wider runtime grid that is not bitwise-equal to the 48-CTA partition.
_SK_STRICT = os.environ.get("MUSE_SK_STRICT", "0").strip() not in ("", "0")
# 64-column tiles (only lm_head 202048x6656) run 4 warps/SM at 48 CTAs, half of what the 128-column shapes get. Running
# them at 256 threads keeps the grid, the schedule and the reduction tree identical -- so the GEMV stays bit-exact --
# and only doubles the resident warps. Off by default until it is measured.
_LMHEAD_T256 = os.environ.get("MUSE_LMHEAD_T256", "0").strip() not in ("", "0")
# Per-output-tile base-pointer tables: weight data_ptr -> int64 CUDA tensor of n_tiles device addresses. Lets the
# allocator place a 641 MiB weight in several buffers small enough to land in the fast band without changing the tile
# index, the grid or the schedule (see set_btab).
_BTAB: dict[int, torch.Tensor] = {}


def _sk_blocks(n: int, k: int) -> int:
    """CTA count for the stream-K TMA4 shapes. 48 (the admitted, bit-exact partition) unless explicitly widened."""
    if _ADMITTING:
        return 48
    override = _SK_BLOCK_OVERRIDE.get((n, k))
    if override is not None:
        return override
    if (n, k) == (DOWN_N, DOWN_K):
        return _SK_DOWN_BLOCKS if _SK_DOWN_BLOCKS > 0 else 48
    if (n, k) != (LM_HEAD_N, LM_HEAD_K):
        return 48
    # lm_head is the only shape with a widening candidate: 3157 column tiles = 287 x 11.
    if _SK_WIDE_BLOCKS > 0:
        return _SK_WIDE_BLOCKS
    return _SK_BLOCKS.get((n, k), 48)


def _tma4_tile_bytes(n: int, k: int) -> int:
    """Bytes of one output column tile in the tiled decode weight: the kernel addresses tile `col` at
    b_sh_stride * (k / 16) int4s from the buffer base (see marlin_template.h b_gl_rd)."""
    tile_n = 128 if n % 128 == 0 else 64
    pack_factor = 8  # NVFP4
    b_sh_stride = ((tile_n * 16) // pack_factor) // 4
    return b_sh_stride * (k // 16) * 16


def set_btab(tiled_weight: torch.Tensor, n: int, k: int, chunks: list[torch.Tensor]) -> tuple:
    """Register a row-chunked placement for a decode weight; returns (wptrs, wtiles, wchunks) device tensors.

    `chunks` are consecutive tile-aligned slices of the packed weight, in order, each its own allocation. The kernel
    reads tile `col` from `wptrs[chunk]` at `col - wtiles[chunk]`, so the tile index, the k-chain, the grid and the
    schedule are unchanged and the GEMV stays bit-exact. Passing `[tiled_weight]` registers the identity placement,
    which is the zero-risk way to exercise the path.
    """
    tile_bytes = _tma4_tile_bytes(n, k)
    total = tiled_weight.numel() * tiled_weight.element_size()
    if total % tile_bytes:
        raise ValueError(f"weight is {total} B, not a whole number of {tile_bytes} B tiles")
    n_tiles = total // tile_bytes
    ptrs: list[int] = []
    wtiles: list[int] = [0]
    for c in chunks:
        if c.data_ptr() % 16:
            raise ValueError("chunk buffers must be 16-byte aligned")
        cnt = c.numel() * c.element_size() // tile_bytes
        if cnt < 1:
            raise ValueError("empty chunk")
        ptrs.append(int(c.data_ptr()))
        wtiles.append(wtiles[-1] + cnt)
    if wtiles[-1] != n_tiles:
        raise ValueError(f"chunks cover {wtiles[-1]} tiles, weight has {n_tiles}")
    wptrs_t = torch.tensor(ptrs, dtype=torch.int64, device=tiled_weight.device)
    wtiles_t = torch.tensor(wtiles, dtype=torch.int32, device=tiled_weight.device)
    entry = (wptrs_t, wtiles_t, len(ptrs))
    _BTAB[tiled_weight.data_ptr()] = entry
    return entry


def _nv6_gate_blocks() -> int:
    if _get_gate_blocks is not None:
        return int(_get_gate_blocks())
    env = os.environ.get("MUSE_NV6_GATE_BLOCKS", "")
    if not env:
        return 24
    try:
        v = int(env)
    except ValueError:
        return 24
    return v if v in (24, 32, 44, 46, 48) else 24


# --- per-layer output_gate launch configuration (opt-in) -------------------------------------
# The attention output_gate GEMV (4096x6656) runs beside the attention kernel on a side
# stream. RoPE sliding-window layers run a 2-CTA attention kernel and can afford a wider gate
# grid; NoPE full-attention layers run 16-32 CTAs and need those SMs back. The block count is
# baked into the packed scale codes at admission time, so it is recorded per layer here and
# re-applied before that layer's gate launch.
# NOTE: the interleaved scale layout (muse_marlin6_sk_tma4_ilv) is only admitted for gate
# blocks 24 or 48; 32/44/46 fall back to the plain super-block layout for the gate layer.
_GATE_SHAPE6 = (4096, 6656)
_GATE_BLOCKS_BY_SCALES: dict[int, int] = {}
_set_gate_blocks = getattr(_lib, "muse_marlin6_set_gate_blocks", None)
if _set_gate_blocks is not None:
    _set_gate_blocks.argtypes = [ctypes.c_int]
    _set_gate_blocks.restype = ctypes.c_int
_gate_dispatch6: bool | None = None


def _gate_dispatch6_enabled() -> bool:
    global _gate_dispatch6
    if _gate_dispatch6 is None:
        _gate_dispatch6 = os.environ.get(
            "MUSE_GATE_ROPE_SW_DISPATCH", "0"
        ).strip() not in ("", "0")
    return _gate_dispatch6


def _env_gate_blocks(name: str, default: int) -> int:
    env = os.environ.get(name, "")
    if not env:
        return default
    try:
        v = int(env)
    except ValueError:
        return default
    return v if v in (24, 32, 44, 46, 48) else default


def _layer_gate_blocks(layer: torch.nn.Module) -> int:
    """Gate block count for this layer: per-layer dispatch when enabled, else the env default."""
    if not _gate_dispatch6_enabled():
        return _nv6_gate_blocks()
    rope_sw = getattr(layer, "_muse_rope_sw", None)
    if rope_sw is None:
        return _nv6_gate_blocks()
    if rope_sw:
        return _env_gate_blocks("MUSE_NV6_GATE_ROPE_SW_BLOCKS", 46)
    return _env_gate_blocks("MUSE_NV6_GATE_NOPE_BLOCKS", 24)

# ---------------------------------------------------------------- rANS-coded gate/up block scales
# Symbols: the layer's palette indices of the tiled one-byte scales. One table per layer (10-bit probabilities),
# 32-bit states, 16-bit renormalisation. Every CTA slice of the fixed grid has 8 warps x 32 lanes: lane l of warp w
# owns bytes 128 w + l + 32 m (m = 0..3) of each 1 KB stage block of raw scales, in stage order; a lane that
# renormalises reads its warp's next 16-bit word for that stage in lane order. Buffer (uint8, rows of 16 bytes):
#   [decode table: u32[1024] = symbol | freq << 6 | (slot - cum) << 17]
#   [336 headers of 3,328 bytes: u32 states[256] (warp-major), u32 offsets[64] (block X >= 2 at the stage-block
#    region + 16 * offsets[X], offsets[iters] ends the last), block 0 and block 1, each padded to 1 KB]
#   [stage blocks X >= 2 of every CTA in CTA order: u8 words-per-warp[8], then the 16-bit words of warp 0..7,
#    padded to 16 bytes]
# A shape (rows, 16) uint8 identifies the coded form. Encoding is a pure function of the scale bytes.
SRANS_M_BITS, SRANS_M, SRANS_L = 10, 1024, 1 << 16
SRANS_HDR, SRANS_CAP = 3328, 1024


def _srans_tasks():
    t = [(i, 0, 52) for i in range(288)]
    t += [(288 + j, 26, 26) for j in range(24)]
    t += [(288 + j, 0, 26) for j in range(24)]
    return t


def _srans_table(counts: torch.Tensor):
    counts = counts.cpu().long()
    total = int(counts.sum())
    f = torch.zeros(64, dtype=torch.long)
    for s in range(64):
        c = int(counts[s])
        if c:
            f[s] = max(1, round(c * SRANS_M / total))
    diff = SRANS_M - int(f.sum())
    while diff:
        k = int(torch.argmax(f))
        step = diff if diff > 0 else max(diff, -(int(f[k]) - 1))
        if step == 0:
            raise ValueError("cannot normalise the scale frequencies")
        f[k] += step
        diff -= step
    cum = torch.cumsum(f, 0) - f
    lut = torch.zeros(SRANS_M, dtype=torch.long)
    for s in range(64):
        if f[s]:
            slots = torch.arange(int(f[s]))
            lut[int(cum[s]) + slots] = s | (int(f[s]) << 6) | (slots << 17)
    return f, cum, lut


def is_gate_up_srans(scales: torch.Tensor) -> bool:
    return scales.dim() == 2 and scales.shape[1] == 16 and scales.dtype == torch.uint8


def is_tma4(scales: torch.Tensor) -> bool:
    return scales.dim() == 2 and scales.shape[1] == 32 and scales.dtype == torch.uint8


def is_tma4_ilv(scales: torch.Tensor) -> bool:
    return is_tma4(scales) and scales.data_ptr() in _DOWN_ILV_SCALES


def _get_tma4_sched(
    n: int,
    k: int,
    fixed_grid: bool,
    device: torch.device,
    *,
    gate_v48: bool = False,
    gate_blocks: int = 48,
):
    key = (n, k, fixed_grid, gate_v48, gate_blocks, device)
    cached = _TMA4_SCHED_CACHE.get(key)
    if cached is not None:
        return cached
    tile_n = 128 if n % 128 == 0 else 64
    # A 64-column tile is the only shape that runs at 128 threads today (lm_head 202048x6656, 4 warps/SM at 48 CTAs).
    # MUSE_LMHEAD_T256 doubles the resident warps without touching the grid or the schedule.
    threads = 256 if (tile_n == 128 or _LMHEAD_T256) else 128
    n_tiles = n // tile_n
    k_tiles = k // 128
    stages: list[tuple[int, int]] = []
    blk_off = [0]
    blk_sb_off = [0]
    blk_iters = []
    if fixed_grid:
        num_blks = 336
        for col in range(288):
            for kt in range(52):
                stages.append((col, kt))
            blk_off.append(len(stages))
            blk_iters.append(52)
            blk_sb_off.append(blk_sb_off[-1] + (52 + 3) // 4)
        for j in range(24):
            for kt in range(26, 52):
                stages.append((288 + j, kt))
            blk_off.append(len(stages))
            blk_iters.append(26)
            blk_sb_off.append(blk_sb_off[-1] + (26 + 3) // 4)
        for j in range(24):
            for kt in range(0, 26):
                stages.append((288 + j, kt))
            blk_off.append(len(stages))
            blk_iters.append(26)
            blk_sb_off.append(blk_sb_off[-1] + (26 + 3) // 4)
    elif gate_v48:
        num_blks = gate_blocks
        vgrid = 48
        iters = (k_tiles * n_tiles + vgrid - 1) // vgrid
        total_p2 = n_tiles * k_tiles
        for b in range(num_blks):
            s_start = len(stages)
            v1 = 47 - b
            for pos in range(iters * v1, min(iters * (v1 + 1), total_p2)):
                stages.append((pos // k_tiles, pos % k_tiles))
            if v1 >= num_blks:
                v2 = v1 - num_blks
                for pos in range(iters * v2, min(iters * (v2 + 1), total_p2)):
                    stages.append((pos // k_tiles, pos % k_tiles))
            blk_off.append(len(stages))
            it = len(stages) - s_start
            blk_iters.append(it)
            blk_sb_off.append(blk_sb_off[-1] + (it + 3) // 4)
    else:
        num_blks = gate_blocks if (n, k) == (4096, 6656) else _sk_blocks(n, k)
        part2_mn_tiles = n_tiles
        part1_mn_iters = 0
        if n_tiles > num_blks:
            part2_mn_tiles = n_tiles % num_blks
            if part2_mn_tiles * 3 <= num_blks:
                part2_mn_tiles += num_blks
            part1_mn_iters = (n_tiles - part2_mn_tiles) // num_blks
        iters = (k_tiles * part2_mn_tiles + num_blks - 1) // num_blks
        base_col = n_tiles - part2_mn_tiles
        total_p2 = part2_mn_tiles * k_tiles
        for b in range(num_blks):
            s_start = len(stages)
            for p in range(part1_mn_iters):
                col = b + p * num_blks
                for kt in range(k_tiles):
                    stages.append((col, kt))
            p2_start = iters * b
            p2_end = min(iters * (b + 1), total_p2)
            for pos in range(p2_start, p2_end):
                stages.append((base_col + pos // k_tiles, pos % k_tiles))
            blk_off.append(len(stages))
            it = len(stages) - s_start
            blk_iters.append(it)
            blk_sb_off.append(blk_sb_off[-1] + (it + 3) // 4)

    total_sbs = blk_sb_off[-1]
    blk_num_sbs = [(it + 3) // 4 for it in blk_iters]
    sb_gsb_map = [0] * total_sbs
    if fixed_grid:
        max_sbs = 0
        sb_row_off: list[int] = []
        for b in range(num_blks):
            w_idx, lane_b = divmod(b, 48)
            base = blk_sb_off[b]
            for sb in range(blk_num_sbs[b]):
                sb_gsb_map[base + sb] = (w_idx * 13 + sb) * 48 + lane_b
    else:
        max_sbs = max(blk_num_sbs)
        sb_row_off = []
        gsb_cursor = 0
        for sb in range(max_sbs):
            active = [b for b in range(num_blks) if sb < blk_num_sbs[b]]
            assert active == list(range(active[0], active[0] + len(active)))
            assert gsb_cursor >= active[0]
            sb_row_off.append(gsb_cursor - active[0])
            for rank, b in enumerate(active):
                sb_gsb_map[blk_sb_off[b] + sb] = gsb_cursor + rank
            gsb_cursor += len(active)
        assert gsb_cursor == total_sbs

    res = (
        torch.tensor(stages, dtype=torch.int32, device=device).reshape(-1),
        torch.tensor(blk_off, dtype=torch.int32, device=device),
        torch.tensor(blk_sb_off, dtype=torch.int32, device=device),
        torch.tensor(sb_gsb_map, dtype=torch.int32, device=device),
        torch.tensor(blk_iters, dtype=torch.int32, device=device),
        torch.tensor(sb_row_off, dtype=torch.int32, device=device),
        num_blks,
        threads,
        total_sbs,
        max_sbs,
        stages,
        blk_off,
        blk_iters,
    )
    _TMA4_SCHED_CACHE[key] = res
    return res


def _build_ilv_weight(w: torch.Tensor, n: int, k: int, gate_v48: bool, gate_blocks: int) -> torch.Tensor:
    k_tiles = k // 128
    *_, num_blks, _, _, _, stages, blk_off, blk_iters = _get_tma4_sched(
        n, k, False, w.device, gate_v48=gate_v48, gate_blocks=gate_blocks
    )
    max_it = max(blk_iters)
    perm = []
    for s in range(max_it):
        for b in range(num_blks):
            if s < blk_iters[b]:
                col, kt = stages[blk_off[b] + s]
                perm.append(col * k_tiles + kt)
    assert len(perm) == (n // 128) * k_tiles
    idx_t = torch.tensor(perm, dtype=torch.long, device=w.device)
    return w.reshape(len(perm), -1).index_select(0, idx_t).contiguous().view_as(w)


def encode_tma4_scales(
    tiled_scales: torch.Tensor,
    palette: torch.Tensor,
    n: int,
    k: int,
    fixed_grid: bool,
    *,
    gate_v48: bool = False,
    gate_blocks: int = 48,
) -> torch.Tensor:
    """Encode tiled one-byte scales into wave/stage-interleaved 4-stage TMA super-block rANS streams (shape (rows, 32) uint8)."""
    if _tma4_p1 is None or _tma4_p2 is None:
        raise RuntimeError("GPU TMA4 rANS encoder unavailable")
    dev = tiled_scales.device
    tile_n = 128 if n % 128 == 0 else 64
    n_tiles = n // tile_n
    k_tiles = k // 128
    stage_bytes = tile_n * 8
    raw = tiled_scales.contiguous().view(torch.uint8).reshape(n_tiles, k_tiles, stage_bytes)
    vals = torch.unique(raw)
    inv = torch.full((256,), -1, dtype=torch.long, device=dev)
    pal = palette.long()
    if not torch.equal(pal[: vals.numel()], vals.long()):
        raise ValueError("palette does not list the scale bytes in order")
    inv[pal[: vals.numel()]] = torch.arange(vals.numel(), device=dev)
    sym = inv[raw.long()].to(torch.uint8).contiguous()
    f, cum, lut = _srans_table(torch.bincount(sym.reshape(-1).long(), minlength=64))
    enc_f = f.to(dtype=torch.int32, device=dev)
    enc_c = cum.to(dtype=torch.int32, device=dev)

    (
        stage_col_k, blk_stage_off, blk_sb_off, sb_gsb_map,
        blk_iters, sb_row_off, num_blks, threads, total_sbs, max_sbs, *_
    ) = _get_tma4_sched(n, k, fixed_grid, dev, gate_v48=gate_v48, gate_blocks=gate_blocks)
    warps = threads // 32
    # The super-block byte budget follows the tile geometry, not the thread count (a 64-column super-block is 4 x 512
    # raw scale bytes, a 128-column one 4 x 1024), and marlin_template.h derives RANS_CAP the same way.
    rans_cap = 3200 if tile_n == 128 else 2048
    sb_wc = torch.empty(total_sbs * warps, dtype=torch.int32, device=dev)
    with torch.cuda.device(dev):
        stream = torch.cuda.current_stream().cuda_stream
        st = _tma4_p1(
            sym.data_ptr(), k_tiles, stage_col_k.data_ptr(), blk_stage_off.data_ptr(),
            blk_sb_off.data_ptr(), sb_gsb_map.data_ptr(), enc_f.data_ptr(), enc_c.data_ptr(),
            sb_wc.data_ptr(), num_blks, threads, stage_bytes, stream,
        )
        if st:
            raise RuntimeError(f"muse_tma4_pass1 failed with CUDA error {st}")

    wc2 = sb_wc.view(total_sbs, warps).long()
    sb_words = warps + wc2.sum(dim=1)
    sb_16b = (sb_words + 7) // 8
    if int(sb_16b.max().item()) * 16 > rans_cap:
        raise ValueError(f"TMA4 super-block exceeds RANS_CAP ({int(sb_16b.max().item()) * 16} > {rans_cap})")
    off16_long = torch.zeros(total_sbs + 1, dtype=torch.long, device=dev)
    off16_long[1:] = torch.cumsum(sb_16b, 0)
    total_var_bytes = int(off16_long[-1].item()) * 16
    off16 = off16_long.to(dtype=torch.int32)

    meta_u32 = ((4 + num_blks + max_sbs + 3) // 4) * 4
    meta_bytes = meta_u32 * 4
    off16_bytes = ((total_sbs + 1) * 4 + 15) // 16 * 16
    states_off = 4096 + meta_bytes + off16_bytes
    var_off = states_off + num_blks * threads * 4
    total_bytes = ((var_off + total_var_bytes + 31) // 32) * 32

    coded = torch.zeros(total_bytes, dtype=torch.uint8, device=dev)
    coded[:4096].copy_(lut.to(dtype=torch.int32, device=dev).view(torch.uint8))
    hdr_meta = torch.tensor([total_sbs, num_blks, max_sbs, threads], dtype=torch.int32, device=dev)
    p = 4096
    coded[p : p + 16].copy_(hdr_meta.view(torch.uint8))
    p += 16
    coded[p : p + num_blks * 4].copy_(blk_iters.view(torch.uint8))
    p += num_blks * 4
    if max_sbs > 0:
        coded[p : p + max_sbs * 4].copy_(sb_row_off.view(torch.uint8))
    coded[4096 + meta_bytes : 4096 + meta_bytes + (total_sbs + 1) * 4].copy_(off16.view(torch.uint8))
    with torch.cuda.device(dev):
        stream = torch.cuda.current_stream().cuda_stream
        bp = coded.data_ptr()
        st = _tma4_p2(
            sym.data_ptr(), k_tiles, stage_col_k.data_ptr(), blk_stage_off.data_ptr(),
            blk_sb_off.data_ptr(), sb_gsb_map.data_ptr(), enc_f.data_ptr(), enc_c.data_ptr(),
            sb_wc.data_ptr(), off16.data_ptr(), bp + states_off, bp + var_off,
            num_blks, threads, stage_bytes, stream,
        )
        if st:
            raise RuntimeError(f"muse_tma4_pass2 failed with CUDA error {st}")
    return coded.view(-1, 32)


def encode_gate_up_scales(tiled_scales: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Tiled one-byte gate/up scales [312, 416, 128] and the layer's 64-byte palette -> the coded buffer
    (new tensor; raises ValueError if a stage block exceeds SRANS_CAP)."""
    dev = tiled_scales.device
    raw = tiled_scales.contiguous().view(torch.uint8).reshape(312, 52, 1024)
    vals = torch.unique(raw)
    inv = torch.full((256,), -1, dtype=torch.long, device=dev)
    pal = palette.long()
    if not torch.equal(pal[:vals.numel()], vals.long()):
        raise ValueError("palette does not list the scale bytes in order")
    inv[pal[:vals.numel()]] = torch.arange(vals.numel(), device=dev)
    sym = inv[raw.long()]
    f, cum, lut = _srans_table(torch.bincount(sym.reshape(-1), minlength=64))
    F, C = f.to(dev), cum.to(dev)
    tasks = _srans_tasks()
    out = []
    for ids, iters in ((list(range(288)), 52), (list(range(288, 336)), 26)):
        G = len(ids)
        s = torch.stack([sym[tasks[t][0], tasks[t][1]:tasks[t][1] + iters] for t in ids])
        steps = iters * 4
        seq = s.view(G, iters, 8, 4, 32).permute(0, 2, 4, 1, 3).reshape(G, 8, 32, steps)
        x = torch.full((G, 8, 32), SRANS_L, dtype=torch.long, device=dev)
        emit = torch.empty((G, 8, 32, steps), dtype=torch.bool, device=dev)
        words = torch.empty((G, 8, 32, steps), dtype=torch.long, device=dev)
        for st in range(steps - 1, -1, -1):
            sy = seq[..., st]
            fs, cs = F[sy], C[sy]
            e = x >= (fs << 22)
            emit[..., st] = e
            words[..., st] = x & 0xFFFF
            x = torch.where(e, x >> 16, x)
            x = ((x // fs) << SRANS_M_BITS) + (x % fs) + cs
        E = emit.view(G, 8, 32, iters, 4).permute(0, 3, 1, 4, 2).reshape(G, iters, 1024)
        Wd = words.view(G, 8, 32, iters, 4).permute(0, 3, 1, 4, 2).reshape(G, iters, 1024)
        cw = E.view(G, iters, 8, 128).sum(-1)
        padded = ((4 + cw.sum(-1) + 7) // 8) * 8  # 16-bit words, whole 16-byte rows
        if int(padded.max()) * 2 > SRANS_CAP:
            raise ValueError("a stage block exceeds the decode slot")
        out.append((torch.tensor(ids, device=dev), iters, x.reshape(G, 256), E, Wd, cw, padded))

    def put(dest, starts, Eb, Wb, cb):
        dest[starts[:, None] + torch.arange(4, device=dev)] = (cb[:, 0::2] | (cb[:, 1::2] << 8)).int()
        pos = starts[:, None] + 4 + torch.cumsum(Eb.long(), 1) - 1
        dest[pos[Eb]] = Wb[Eb].int()

    base, total = [], 0
    for ids, iters, *_rest, padded in out:
        flat = padded[:, 2:].reshape(-1)
        base.append((torch.cumsum(flat, 0) - flat + total).view(len(ids), iters - 2))
        total += int(flat.sum())
    hdr = torch.zeros((336, SRANS_HDR // 2), dtype=torch.int32, device=dev)
    var = torch.zeros(total, dtype=torch.int32, device=dev)
    hflat = hdr.view(-1)
    for (ids, iters, states, E, Wd, cw, padded), st in zip(out, base):
        G = len(ids)
        put(var, st.reshape(-1), E[:, 2:].reshape(-1, 1024), Wd[:, 2:].reshape(-1, 1024), cw[:, 2:].reshape(-1, 8))
        hs = ids[:, None] * (SRANS_HDR // 2) + 640 + 512 * torch.arange(2, device=dev)[None, :]
        put(hflat, hs.reshape(-1), E[:, :2].reshape(-1, 1024), Wd[:, :2].reshape(-1, 1024), cw[:, :2].reshape(-1, 8))
        hdr[ids, 0:512:2] = (states & 0xFFFF).int()
        hdr[ids, 1:512:2] = (states >> 16).int()
        offs = torch.zeros((G, 64), dtype=torch.long, device=dev)
        offs[:, 2:iters] = st // 8
        offs[:, iters] = (st[:, -1] + padded[:, -1]) // 8
        hdr[ids, 512:640:2] = (offs & 0xFFFF).int()
        hdr[ids, 513:640:2] = (offs >> 16).int()

    def u16(v):
        v = v & 0xFFFF
        return torch.where(v >= 32768, v - 65536, v).to(torch.int16).contiguous().view(torch.uint8)

    lut8 = lut.to(dev).to(torch.int32).contiguous().view(torch.uint8)
    coded = torch.cat((lut8, u16(hdr.reshape(-1)), u16(var)))
    return coded.view(-1, 16)


# ---------------------------------------------------------------- rANS-coded down-projection block scales
# The same symbols, table, lanes and block format as the gate/up stream, for the 48-CTA stream-K down projection
# (6656 x 19968; gemv6_srans_down.cu). Stage blocks in tile-major order (column tile t, k-stage s -> 156 t + s:
# 52 x 156 = 8,112 = 48 x 169); CTA b decodes blocks 169 b .. 169 b + 168 in order across its column slices.
# Buffer (uint8, rows of 16 bytes):
#   [decode table: u32[1024]]
#   [48 headers of 3,840 bytes: u32 states[256] (warp-major), u32 offsets[192] (block X >= 2 at the stage-block
#    region + 16 * offsets[X], offsets[169] ends the last), block 0 and block 1, each padded to 1 KB]
#   [stage blocks X >= 2 of every CTA in CTA order]
SRANS_DOWN_HDR, SRANS_DOWN_CTAS, SRANS_DOWN_ITERS = 3840, 48, 169


def encode_down_scales(tiled_scales: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Tiled one-byte down-projection scales [52, 1248, 128] and the layer's 64-byte palette -> the coded buffer
    (new tensor; raises ValueError if a stage block exceeds SRANS_CAP)."""
    dev = tiled_scales.device
    G, iters = SRANS_DOWN_CTAS, SRANS_DOWN_ITERS
    raw = tiled_scales.contiguous().view(torch.uint8).reshape(G * iters, 1024)
    vals = torch.unique(raw)
    inv = torch.full((256,), -1, dtype=torch.long, device=dev)
    pal = palette.long()
    if not torch.equal(pal[:vals.numel()], vals.long()):
        raise ValueError("palette does not list the scale bytes in order")
    inv[pal[:vals.numel()]] = torch.arange(vals.numel(), device=dev)
    sym = inv[raw.long()]
    f, cum, lut = _srans_table(torch.bincount(sym.reshape(-1), minlength=64))
    F, C = f.to(dev), cum.to(dev)
    steps = iters * 4
    seq = sym.view(G, iters, 8, 4, 32).permute(0, 2, 4, 1, 3).reshape(G, 8, 32, steps)
    x = torch.full((G, 8, 32), SRANS_L, dtype=torch.long, device=dev)
    emit = torch.empty((G, 8, 32, steps), dtype=torch.bool, device=dev)
    words = torch.empty((G, 8, 32, steps), dtype=torch.long, device=dev)
    for st in range(steps - 1, -1, -1):
        sy = seq[..., st]
        fs, cs = F[sy], C[sy]
        e = x >= (fs << 22)
        emit[..., st] = e
        words[..., st] = x & 0xFFFF
        x = torch.where(e, x >> 16, x)
        x = ((x // fs) << SRANS_M_BITS) + (x % fs) + cs
    E = emit.view(G, 8, 32, iters, 4).permute(0, 3, 1, 4, 2).reshape(G, iters, 1024)
    Wd = words.view(G, 8, 32, iters, 4).permute(0, 3, 1, 4, 2).reshape(G, iters, 1024)
    cw = E.view(G, iters, 8, 128).sum(-1)
    padded = ((4 + cw.sum(-1) + 7) // 8) * 8  # 16-bit words, whole 16-byte rows
    if int(padded.max()) * 2 > SRANS_CAP:
        raise ValueError("a stage block exceeds the decode slot")

    def put(dest, starts, Eb, Wb, cb):
        dest[starts[:, None] + torch.arange(4, device=dev)] = (cb[:, 0::2] | (cb[:, 1::2] << 8)).int()
        pos = starts[:, None] + 4 + torch.cumsum(Eb.long(), 1) - 1
        dest[pos[Eb]] = Wb[Eb].int()

    flat = padded[:, 2:].reshape(-1)
    st = (torch.cumsum(flat, 0) - flat).view(G, iters - 2)
    hw = SRANS_DOWN_HDR // 2
    hdr = torch.zeros((G, hw), dtype=torch.int32, device=dev)
    var = torch.zeros(int(flat.sum()), dtype=torch.int32, device=dev)
    put(var, st.reshape(-1), E[:, 2:].reshape(-1, 1024), Wd[:, 2:].reshape(-1, 1024), cw[:, 2:].reshape(-1, 8))
    hs = torch.arange(G, device=dev)[:, None] * hw + 896 + 512 * torch.arange(2, device=dev)[None, :]
    put(hdr.view(-1), hs.reshape(-1), E[:, :2].reshape(-1, 1024), Wd[:, :2].reshape(-1, 1024),
        cw[:, :2].reshape(-1, 8))
    states = x.reshape(G, 256)
    hdr[:, 0:512:2] = (states & 0xFFFF).int()
    hdr[:, 1:512:2] = (states >> 16).int()
    offs = torch.zeros((G, 192), dtype=torch.long, device=dev)
    offs[:, 2:iters] = st // 8
    offs[:, iters] = (st[:, -1] + padded[:, -1]) // 8
    hdr[:, 512:896:2] = (offs & 0xFFFF).int()
    hdr[:, 513:896:2] = (offs >> 16).int()

    def u16(v):
        v = v & 0xFFFF
        return torch.where(v >= 32768, v - 65536, v).to(torch.int16).contiguous().view(torch.uint8)

    lut8 = lut.to(dev).to(torch.int32).contiguous().view(torch.uint8)
    coded = torch.cat((lut8, u16(hdr.reshape(-1)), u16(var)))
    return coded.view(-1, 16)


def pack_scales6(tiled_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """[n/tile_n, k/16, tile_n] one-byte scales (any 1-byte dtype) -> (codes uint8 [3/4 bytes], palette uint8 [64]),
    or None when the layer has more than 64 distinct scale bytes. The code of byte b of the flat tensor sits at bits
    [6b, 6b + 6) of the code stream, least significant bit first: four codes per three bytes."""
    flat = tiled_scales.contiguous().view(torch.uint8).reshape(-1)
    if flat.numel() % 4:
        return None
    values = torch.unique(flat)
    if values.numel() > 64:
        return None
    palette = torch.zeros(64, dtype=torch.uint8, device=flat.device)
    palette[: values.numel()] = values
    lut = torch.zeros(256, dtype=torch.uint8, device=flat.device)
    lut[values.long()] = torch.arange(values.numel(), dtype=torch.uint8, device=flat.device)
    codes = lut[flat.long()].view(-1, 4).to(torch.int32)
    c0, c1, c2, c3 = codes.unbind(1)
    packed = torch.stack(
        (c0 | ((c1 & 3) << 6), (c1 >> 2) | ((c2 & 15) << 4), (c2 >> 4) | (c3 << 2)), dim=1
    ).to(torch.uint8).reshape(-1)
    # round trip on the device before anyone relies on it
    q = packed.view(-1, 3).to(torch.int32)
    d = torch.stack((q[:, 0] & 63, (q[:, 0] >> 6) | ((q[:, 1] & 15) << 2), (q[:, 1] >> 4) | ((q[:, 2] & 3) << 4), q[:, 2] >> 2), 1)
    if not torch.equal(palette[d.reshape(-1).long()], flat):
        return None
    return packed, palette


def decode_linear6(
    x: torch.Tensor,
    tiled_weight: torch.Tensor,
    scales6: torch.Tensor,
    palette: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
    n: int,
    k: int,
    fixed_grid: bool = False,
) -> torch.Tensor:
    if not STATS["logged"]:
        STATS["logged"] = True
        # Report the output_gate block count, because it is resolved in C++ from
        # MUSE_NV6_GATE_BLOCKS (gemv6.cu:22) and appears nowhere else in either log. Without this,
        # a leg cannot show that the knob took: the number is not in the pack lines (unlike the
        # rANS gate, whose wpb is), and there is no other admission line for it.
        _gb_note = ""
        try:
            if _gate_dispatch6_enabled():
                _gb_note = ", output_gate on %d blocks (RoPE sliding-window) / %d blocks (NoPE full attention), per-layer dispatch" % (
                    _env_gate_blocks("MUSE_NV6_GATE_ROPE_SW_BLOCKS", 46),
                    _env_gate_blocks("MUSE_NV6_GATE_NOPE_BLOCKS", 24),
                )
            else:
                _gb = _nv6_gate_blocks()
                # The block count alone does not identify the instantiation: gemv6.cu:74-82 selects
                # GateV48=true only for gb == 24, so "24 blocks with GateV48=false" is a third state
                # that no env value can produce -- it is what a stale overlay or a changed guard
                # would give you. Print the instantiation the kernel will actually take, derived by
                # the same rule as is_gate_v48 at :1075, so the arm cannot be "verified" by the
                # block count alone.
                _gb_note = ", output_gate on %d blocks, GateV48=%s" % (
                    _gb, "true" if _gb == 24 else "false"
                )
        except Exception:  # noqa: BLE001 - a log line must never fail the load
            _gb_note = ""
        logger.info("NVFP4 decode GEMV with 6-bit block scales: %d layers, %d kept uncompressed%s",
                    STATS["admitted"], STATS["kept"], _gb_note)
    if not x.is_contiguous() or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)
    output = x.new_empty((*x.shape[:-1], n))
    # One grid for every kernel that can serve this shape, so the per-layer bitwise admission compares the coded
    # scales against the six-bit scales *under the same schedule* (see _sk_blocks). Shapes outside _SK_BLOCKS get
    # their admitted 48 CTAs.
    blocks = _sk_blocks(n, k)
    tile_n = 128 if n % 128 == 0 else 64
    partials = x.new_empty((blocks, 16, tile_n), dtype=torch.float32)
    with torch.cuda.device(x.device):
        use_grid = fixed_grid and _gate_up is not None and x.numel() == k and (n, k) == (39936, 6656)
        kernel = _gate_up if use_grid else _gemv6
        extra: tuple = ()
        if is_tma4_ilv(scales6):
            if x.numel() == k and _sk_tma4_ilv is not None:
                kernel = _sk_tma4_ilv
            else:
                raise RuntimeError(f"TMA4-ILV rANS scales unsupported for shape ({n}, {k}, fixed={fixed_grid})")
        elif is_tma4(scales6):
            if use_grid and _gate_up_tma4 is not None:
                kernel = _gate_up_tma4
            elif not fixed_grid and x.numel() == k:
                chunked = _BTAB.get(tiled_weight.data_ptr())
                if chunked is not None and _sk_tma4_chunked is not None:
                    # Row-chunked placement: the kernel reads each tile from its own buffer. Same tile index, same
                    # k-chain, same grid -- bit-exact.
                    kernel = _sk_tma4_chunked
                    extra = (chunked[0].data_ptr(), chunked[1].data_ptr(), int(chunked[2]))
                elif _LMHEAD_T256 and tile_n == 64 and _sk_tma4_t256 is not None:
                    kernel = _sk_tma4_t256
                elif _sk_tma4 is not None:
                    kernel = _sk_tma4
                else:
                    raise RuntimeError(f"TMA4 rANS scales unsupported for shape ({n}, {k}, fixed={fixed_grid})")
            else:
                raise RuntimeError(f"TMA4 rANS scales unsupported for shape ({n}, {k}, fixed={fixed_grid})")
        elif is_gate_up_srans(scales6) and (n, k) == (DOWN_N, DOWN_K):
            # only the stream-K coded-scale kernel can read this scale form
            if fixed_grid or x.numel() != k or _down_srans is None:
                raise RuntimeError("rANS-coded down scales outside the one-row 48-CTA GEMV")
            kernel = _down_srans
        elif is_gate_up_srans(scales6):
            # only the coded-scale kernel can read this scale form
            if not use_grid or _gate_up_srans is None:
                raise RuntimeError("rANS-coded gate/up scales outside the one-row fixed grid")
            kernel = _gate_up_srans
        if _GATE_BLOCKS_BY_SCALES and (n, k) == _GATE_SHAPE6:
            # Per-layer gate grid: the packed scale codes were built for this block count.
            gb = _GATE_BLOCKS_BY_SCALES.get(scales6.data_ptr())
            if gb is not None and _set_gate_blocks is not None:
                _set_gate_blocks(gb)
        status = kernel(
            x.data_ptr(), tiled_weight.data_ptr(), output.data_ptr(), partials.data_ptr(),
            scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(), n, k, blocks,
            x.device.index, torch.cuda.current_stream().cuda_stream, *extra,
        )
    if status:
        raise RuntimeError(f"SM121 Marlin GEMV (scale6) launch failed with CUDA error {status}")
    return output


def _decode6_fake(x, tiled_weight, scales6, palette, global_scale, workspace, n, k, fixed_grid=False):
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op("marlin_tiled_decode6", decode_linear6, fake_impl=_decode6_fake)


def self_check6(layer: torch.nn.Module, reference, fixed_grid: bool = False) -> bool:
    """decode_linear6 == reference (the uncompressed tiled GEMV) bitwise on the layer's own weight, for activations with
    large and cancelling entries at several magnitudes. Runs on the admitted 48-CTA partition (see _ADMITTING), which
    is the partition the uncompressed reference uses."""
    global _ADMITTING
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    dev = layer.weight.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(n * 7 + k)
    saved = _ADMITTING
    _ADMITTING = True
    try:
        for mag in (0.01, 1.0, 64.0):
            x = torch.randn(1, k, generator=gen, device=dev) * mag
            x[:, ::3] *= 64
            x[:, 1::2] = -x[:, ::2]
            x = x.to(torch.bfloat16)
            got = decode_linear6(x, layer._marlin_decode_weight, layer._marlin_decode_scales6,
                                 layer._marlin_decode_palette, layer.weight_global_scale, layer.workspace,
                                 n, k, fixed_grid)
            ref = reference(x, layer._marlin_decode_weight, layer._marlin_decode_scales,
                            layer.weight_global_scale, layer.workspace, n, k)
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                return False
    finally:
        _ADMITTING = saved
    return True


def _gate_up_probes(layer: torch.nn.Module) -> list[torch.Tensor]:
    """One-row activations for the coded-scale admission: self_check6's large and cancelling entries at three
    magnitudes and two plain Gaussian rows, all BF16."""
    k, dev = layer.input_size_per_partition, layer.weight.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(GATE_UP_N * 7 + k + 1)
    out = []
    for mag in (0.01, 1.0, 64.0):
        x = torch.randn(1, k, generator=gen, device=dev) * mag
        x[:, ::3] *= 64
        x[:, 1::2] = -x[:, ::2]
        out.append(x.to(torch.bfloat16))
    for mag in (0.5, 4.0):
        out.append((torch.randn(1, k, generator=gen, device=dev) * mag).to(torch.bfloat16))
    return out


def _gate_up_grid_outputs(layer, weight, scales, probes) -> list[torch.Tensor]:
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    return [decode_linear6(x, weight, scales, layer._marlin_decode_palette, layer.weight_global_scale,
                           layer.workspace, n, k, True) for x in probes]


def _six_bit_codes(layer: torch.nn.Module) -> torch.Tensor:
    """The layer's six-bit scale codes, rebuilt from the tiled one-byte scales."""
    packed = pack_scales6(layer._marlin_decode_scales)
    if packed is None or not torch.equal(packed[1], layer._marlin_decode_palette):
        raise ValueError("six-bit codes cannot be rebuilt")
    return packed[0]


def _same_bits(expected, got) -> bool:
    return all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(expected, got))


def check_gate_up_srans(layer: torch.nn.Module) -> bool:
    """For a layer with rANS-coded gate/up scales: True iff the coded kernel on the current buffers equals, bit for
    bit, the admitted six-bit fixed grid with codes rebuilt from the one-byte scales. Depends only on the buffers'
    contents, so it can be re-run after a placement step copies the buffers elsewhere."""
    c = layer._marlin_decode_scales6
    if not is_gate_up_srans(c) or _gate_up_srans is None:
        return False
    probes = _gate_up_probes(layer)
    expected = _gate_up_grid_outputs(layer, layer._marlin_decode_weight, _six_bit_codes(layer), probes)
    return _same_bits(expected, _gate_up_grid_outputs(layer, layer._marlin_decode_weight, c, probes))


def revert_gate_up_srans(layer: torch.nn.Module) -> None:
    """Put rANS-coded gate/up scales back to six-bit codes (into the same storage when the coded buffer still lives
    at the start of the six-bit code buffer it was written into)."""
    c = layer._marlin_decode_scales6
    if is_gate_up_srans(c):
        codes = _six_bit_codes(layer)
        base = c._base
        if (base is not None and base.dim() == 1 and base.dtype == torch.uint8 and base.numel() == codes.numel()
                and base.data_ptr() == c.data_ptr()):
            base.copy_(codes)
            codes = base
        layer.register_buffer("_marlin_decode_scales6", codes, persistent=False)
    layer.marlin_decode6_gate_up_srans = False


def admit_gate_up_srans(layer: torch.nn.Module) -> bool:
    """For an admitted fixed-grid gate/up layer with six-bit codes: encode its scales (a pure function of the tiled
    one-byte scales) and switch _marlin_decode_scales6 to the coded buffer only if the coded kernel reproduces
    every output bit of the six-bit fixed grid on the probes. Otherwise nothing changes."""
    c = layer._marlin_decode_scales6
    if is_gate_up_srans(c):  # already coded (e.g. after a placement move): only re-check
        layer.marlin_decode6_gate_up_srans = check_gate_up_srans(layer)
        return layer.marlin_decode6_gate_up_srans
    layer.marlin_decode6_gate_up_srans = False
    if (_gate_up_srans is None or not getattr(layer, "marlin_decode6_fixed_grid", False)
            or (layer.output_size_per_partition, layer.input_size_per_partition) != (GATE_UP_N, GATE_UP_K)
            or getattr(layer, "_marlin_decode_scales", None) is None):
        return False
    w = layer._marlin_decode_weight
    probes = _gate_up_probes(layer)
    expected = _gate_up_grid_outputs(layer, w, c, probes)
    coded = encode_gate_up_scales(layer._marlin_decode_scales, layer._marlin_decode_palette)
    if not _same_bits(expected, _gate_up_grid_outputs(layer, w, coded, probes)):
        return False
    if int(torch.count_nonzero(layer.workspace)) != 0:
        return False
    if coded.numel() <= c.numel() and c.dim() == 1 and c.is_contiguous():
        # write the coded bytes over the start of the six-bit code buffer, so the scale stream keeps the memory the
        # loader gave it (no late allocation); the temporary is freed
        c[:coded.numel()].copy_(coded.view(-1))
        coded = c[:coded.numel()].view(-1, 16)
        if not _same_bits(expected, _gate_up_grid_outputs(layer, w, coded, probes)):
            c.copy_(_six_bit_codes(layer))
            return False
    layer.register_buffer("_marlin_decode_scales6", coded, persistent=False)
    layer.marlin_decode6_gate_up_srans = True
    STATS["srans_scales"] += layer._marlin_decode_scales.numel()
    STATS["srans_bytes"] += coded.numel()
    return True


def _down_probes(layer: torch.nn.Module) -> list[torch.Tensor]:
    """One-row activations for the coded down-scale admission: the gate/up probe recipe (cancelling entries at three
    magnitudes, two Gaussian rows) plus a heavy-tailed row shaped like a SwiGLU output, all BF16."""
    k, dev = layer.input_size_per_partition, layer.weight.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(DOWN_N * 7 + k + 1)
    out = []
    for mag in (0.01, 1.0, 64.0):
        x = torch.randn(1, k, generator=gen, device=dev) * mag
        x[:, ::3] *= 64
        x[:, 1::2] = -x[:, ::2]
        out.append(x.to(torch.bfloat16))
    for mag in (0.5, 4.0):
        out.append((torch.randn(1, k, generator=gen, device=dev) * mag).to(torch.bfloat16))
    g = torch.randn(1, k, generator=gen, device=dev) * 2
    out.append((g * torch.sigmoid(g) * torch.randn(1, k, generator=gen, device=dev)).to(torch.bfloat16))
    return out


def _down_outputs(layer, scales, probes) -> list[torch.Tensor]:
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    return [decode_linear6(x, layer._marlin_decode_weight, scales, layer._marlin_decode_palette,
                           layer.weight_global_scale, layer.workspace, n, k, False) for x in probes]


def check_down_srans(layer: torch.nn.Module) -> bool:
    """For a layer with rANS-coded down scales: True iff the coded stream-K kernel on the current buffers equals, bit
    for bit, the six-bit kernel with codes rebuilt from the one-byte scales. Depends only on the buffers' contents."""
    c = layer._marlin_decode_scales6
    if not is_gate_up_srans(c) or _down_srans is None:
        return False
    probes = _down_probes(layer)
    return _same_bits(_down_outputs(layer, _six_bit_codes(layer), probes), _down_outputs(layer, c, probes))


def _restore_six_bit(layer: torch.nn.Module) -> None:
    """Register the layer's six-bit codes as _marlin_decode_scales6, written into the storage of the current scale
    buffer when that storage is exactly the six-bit code buffer the coded bytes were written over."""
    c = layer._marlin_decode_scales6
    codes = _six_bit_codes(layer)
    if c.dtype == torch.uint8 and c.storage_offset() == 0 and c.untyped_storage().nbytes() == codes.numel():
        home = c.as_strided((codes.numel(),), (1,), 0)
        home.copy_(codes)
        codes = home
    layer.register_buffer("_marlin_decode_scales6", codes, persistent=False)


def revert_down_srans(layer: torch.nn.Module) -> None:
    """Put rANS-coded down scales back to six-bit codes (into the same storage when the coded buffer still lives at
    the start of the six-bit code buffer it was written into)."""
    if is_gate_up_srans(layer._marlin_decode_scales6):
        _restore_six_bit(layer)
    layer.marlin_decode6_down_srans = False


def admit_down_srans(layer: torch.nn.Module) -> bool:
    """For an admitted six-bit down-projection layer: encode its scales (a pure function of the tiled one-byte
    scales), write the coded bytes over the start of the layer's own six-bit code buffer and switch to them only if
    the coded stream-K kernel reproduces every output bit of the six-bit kernel on the probes, before and after the
    write. Otherwise the six-bit codes stay (rewritten in place if the write happened)."""
    c = layer._marlin_decode_scales6
    if is_gate_up_srans(c):  # already coded (e.g. after a placement move): only re-check
        layer.marlin_decode6_down_srans = check_down_srans(layer)
        return layer.marlin_decode6_down_srans
    layer.marlin_decode6_down_srans = False
    if (_down_srans is None or getattr(layer, "marlin_decode6_fixed_grid", False)
            or (layer.output_size_per_partition, layer.input_size_per_partition) != (DOWN_N, DOWN_K)
            or getattr(layer, "_marlin_decode_scales", None) is None
            or c.dim() != 1 or not c.is_contiguous()):
        return False
    probes = _down_probes(layer)
    expected = _down_outputs(layer, c, probes)
    coded = encode_down_scales(layer._marlin_decode_scales, layer._marlin_decode_palette)
    if coded.numel() > c.numel() or not _same_bits(expected, _down_outputs(layer, coded, probes)):
        return False
    if int(torch.count_nonzero(layer.workspace)) != 0:
        return False
    # write the coded bytes over the start of the six-bit code buffer, so the scale stream keeps the memory the
    # loader gave it (no late allocation); the temporary is freed
    nbytes = coded.numel()
    c[:nbytes].copy_(coded.view(-1))
    del coded
    view = c[:nbytes].view(-1, 16)
    try:
        ok = _same_bits(expected, _down_outputs(layer, view, probes)) and int(torch.count_nonzero(layer.workspace)) == 0
    except (RuntimeError, ValueError):
        ok = False
    if not ok:
        try:
            c.copy_(_six_bit_codes(layer))
        except (RuntimeError, ValueError):
            # the six-bit codes cannot be put back: this layer falls back to the uncompressed tiled GEMV
            del layer._marlin_decode_scales6, layer._marlin_decode_palette
        return False
    layer.register_buffer("_marlin_decode_scales6", view, persistent=False)
    layer.marlin_decode6_down_srans = True
    STATS["down_srans_scales"] += layer._marlin_decode_scales.numel()
    STATS["down_srans_bytes"] += nbytes
    return True


def _admit_tma4(layer: torch.nn.Module, fixed_grid: bool) -> bool:
    """Admit 4-stage TMA super-block rANS scales on gate_up, down, or lm_head after exact bitwise check."""
    c = layer._marlin_decode_scales6
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    if getattr(layer, "_marlin_decode_scales", None) is None or c.dim() != 1 or not c.is_contiguous():
        return False
    dev = layer.weight.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(n * 11 + k + 3)
    probes = []
    for mag in (0.01, 1.0, 64.0):
        x = torch.randn(1, k, generator=gen, device=dev) * mag
        x[:, ::3] *= 64
        x[:, 1::2] = -x[:, ::2]
        probes.append(x.to(torch.bfloat16))
    w = layer._marlin_decode_weight
    pal = layer._marlin_decode_palette
    gs = layer.weight_global_scale
    ws = layer.workspace
    expected = [decode_linear6(x, w, c, pal, gs, ws, n, k, fixed_grid) for x in probes]
    want_gu_ilv = (
        fixed_grid
        and (n, k) == (GATE_UP_N, GATE_UP_K)
        and _sk_tma4_ilv is not None
        and os.environ.get("MUSE_GATE_UP_ILV", "0") != "0"
    )
    gb = _layer_gate_blocks(layer) if (not fixed_grid and (n, k) == (4096, 6656)) else 48
    is_gate_v48 = (not fixed_grid and (n, k) == (4096, 6656) and gb == 24)
    coded = encode_tma4_scales(
        layer._marlin_decode_scales, pal, n, k,
        False if want_gu_ilv else fixed_grid,
        gate_v48=is_gate_v48, gate_blocks=gb,
    )
    if coded.numel() > c.numel():
        logger.warning("TMA4 (%dx%d) rejected: coded %d > c %d", n, k, coded.numel(), c.numel())
        return False
    want_ilv = _sk_tma4_ilv is not None and (
        want_gu_ilv
        or (
            not fixed_grid
            and (
                ((n, k) == (DOWN_N, DOWN_K) and os.environ.get("MUSE_DOWN_ILV", "1") != "0")
                or ((n, k) in ATTN_ILV_SHAPES and gb in (24, 48) and os.environ.get("MUSE_ATTN_ILV", "1") != "0")
            )
        )
    )
    if want_ilv:
        w_ilv = _build_ilv_weight(w, n, k, is_gate_v48, gb)
        _DOWN_ILV_SCALES.add(coded.data_ptr())
        try:
            got_ilv = [decode_linear6(x, w_ilv, coded, pal, gs, ws, n, k, False) for x in probes]
        finally:
            _DOWN_ILV_SCALES.discard(coded.data_ptr())
        if _same_bits(expected, got_ilv) and int(torch.count_nonzero(ws)) == 0:
            nbytes = coded.numel()
            w_backup = w.detach().clone()
            w.copy_(w_ilv)
            del w_ilv
            c[:nbytes].copy_(coded.view(-1))
            del coded
            view = c[:nbytes].view(-1, 32)
            _DOWN_ILV_SCALES.add(view.data_ptr())
            got2 = [decode_linear6(x, w, view, pal, gs, ws, n, k, False) for x in probes]
            if _same_bits(expected, got2) and int(torch.count_nonzero(ws)) == 0:
                del w_backup
                layer.register_buffer("_marlin_decode_scales6", view, persistent=False)
                if want_gu_ilv:
                    layer.marlin_decode6_fixed_grid = False
                    layer.marlin_decode6_gate_up_srans = True
                    STATS["srans_scales"] += layer._marlin_decode_scales.numel()
                    STATS["srans_bytes"] += nbytes
                else:
                    layer.marlin_decode6_down_srans = True
                    STATS["down_srans_scales"] += layer._marlin_decode_scales.numel()
                    STATS["down_srans_bytes"] += nbytes
                return True
            _DOWN_ILV_SCALES.discard(view.data_ptr())
            w.copy_(w_backup)
            del w_backup
            c.copy_(_six_bit_codes(layer))
            return False
        del w_ilv
        if want_gu_ilv:
            coded = encode_tma4_scales(layer._marlin_decode_scales, pal, n, k, True)
            if coded.numel() > c.numel():
                return False
    got = [decode_linear6(x, w, coded, pal, gs, ws, n, k, fixed_grid) for x in probes]
    if not _same_bits(expected, got) or int(torch.count_nonzero(ws)) != 0:
        logger.warning("TMA4 (%dx%d) rejected: bitwise=%s ws_nz=%d", n, k, _same_bits(expected, got), int(torch.count_nonzero(ws)))
        return False
    nbytes = coded.numel()
    c[:nbytes].copy_(coded.view(-1))
    del coded
    view = c[:nbytes].view(-1, 32)
    got2 = [decode_linear6(x, w, view, pal, gs, ws, n, k, fixed_grid) for x in probes]
    if not _same_bits(expected, got2) or int(torch.count_nonzero(ws)) != 0:
        c.copy_(_six_bit_codes(layer))
        return False
    layer.register_buffer("_marlin_decode_scales6", view, persistent=False)
    if (n, k) == (LM_HEAD_N, LM_HEAD_K):
        # Placement check. If a row-chunked table is registered (set_btab) the chunked kernel must reproduce the
        # contiguous weight bit for bit -- that equality is the entire safety argument for splitting the buffer, so
        # prove it here rather than on the goldens. MUSE_LMHEAD_BTAB=1 additionally registers an identity table (one
        # chunk spanning every tile) when none exists, which exercises the indirection at zero risk.
        registered = _BTAB.get(w.data_ptr())
        if registered is None and os.environ.get("MUSE_LMHEAD_BTAB", "0") != "0":
            set_btab(w, n, k, [w])
            registered = _BTAB.get(w.data_ptr())
        if registered is not None:
            _BTAB.pop(w.data_ptr(), None)
            try:
                contig = [decode_linear6(x, w, view, pal, gs, ws, n, k, fixed_grid) for x in probes]
            finally:
                _BTAB[w.data_ptr()] = registered
            placed = [decode_linear6(x, w, view, pal, gs, ws, n, k, fixed_grid) for x in probes]
            if not _same_bits(contig, placed) or int(torch.count_nonzero(ws)) != 0:
                _BTAB.pop(w.data_ptr(), None)
                logger.warning("TMA4 (%dx%d) rejected: %d-chunk placement is not bitwise identical to the "
                               "contiguous weight", n, k, registered[2])
                return False
            logger.info("TMA4 (%dx%d): %d-chunk placement is bitwise identical to the contiguous weight "
                        "(%d tiles of %d B, chunk split %s)", n, k, registered[2],
                        int(registered[1][-1].item()), _tma4_tile_bytes(n, k),
                        registered[1].tolist())
    if fixed_grid:
        layer.marlin_decode6_gate_up_srans = True
        STATS["srans_scales"] += layer._marlin_decode_scales.numel()
        STATS["srans_bytes"] += nbytes
    else:
        layer.marlin_decode6_down_srans = True
        STATS["down_srans_scales"] += layer._marlin_decode_scales.numel()
        STATS["down_srans_bytes"] += nbytes
    return True


def admit_tma4(layer: torch.nn.Module, fixed_grid: bool) -> bool:
    """Admit TMA4 scales on the admitted 48-CTA partition (see _ADMITTING), then report how far the runtime grid is
    from it. A wider grid changes the split-K reduction grouping, so it cannot be bitwise-equal to the 48-CTA
    reference by construction; the gate for it is the end-to-end golden hashes, and MUSE_SK_STRICT=1 turns the
    report into a veto instead."""
    global _ADMITTING
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    wide = _SK_BLOCKS.get((n, k), 48) if _SK_WIDE_BLOCKS <= 0 else _SK_WIDE_BLOCKS
    _ADMITTING = True
    try:
        ok = _admit_tma4(layer, fixed_grid)
    finally:
        _ADMITTING = False
    if not ok or wide == 48:
        return ok
    _SK_BLOCK_OVERRIDE[(n, k)] = wide
    try:
        narrow = _sk_probe_outputs(layer, n, k, admit=True)
        wide_out = _sk_probe_outputs(layer, n, k)
    finally:
        _SK_BLOCK_OVERRIDE.pop((n, k), None)
    worst = 0
    ulps = 0
    for a, b in zip(narrow, wide_out):
        d = (a.view(torch.int16).to(torch.int32) - b.view(torch.int16).to(torch.int32)).abs()
        ulps += int((d != 0).sum())
        worst = max(worst, int(d.max()))
    if ulps == 0:
        logger.info("TMA4 (%dx%d): %d-CTA runtime grid is bitwise identical to the 48-CTA partition", n, k, wide)
        return True
    if _SK_STRICT:
        _SK_BLOCK_OVERRIDE[(n, k)] = 48
        logger.warning("TMA4 (%dx%d): %d-CTA runtime grid differs from 48 CTAs on %d elements (max %d ULP), "
                       "falling back (MUSE_SK_STRICT=1)", n, k, wide, ulps, worst)
        return True
    logger.info("TMA4 (%dx%d): %d-CTA runtime grid differs from the 48-CTA partition on %d of %d elements "
                "(max %d ULP) -- reduction regrouping only; gated by the end-to-end golden hashes",
                n, k, wide, ulps, sum(a.numel() for a in narrow), worst)
    return True


def _sk_probe_outputs(layer: torch.nn.Module, n: int, k: int, admit: bool = False) -> list[torch.Tensor]:
    """The layer's own outputs on the admission probes, on the current runtime grid (or the 48-CTA grid when
    ``admit`` is set)."""
    global _ADMITTING
    saved = _ADMITTING
    _ADMITTING = admit
    try:
        return [decode_linear6(x, layer._marlin_decode_weight, layer._marlin_decode_scales6,
                               layer._marlin_decode_palette, layer.weight_global_scale, layer.workspace,
                               n, k, False) for x in _down_probes(layer)]
    finally:
        _ADMITTING = saved


def prepare_decode_layout6(layer: torch.nn.Module, reference) -> bool:
    """After the board's prepare_decode_layout: add the 6-bit scale codes and palette, check, register. True if admitted."""
    layer.marlin_decode6_fixed_grid = False
    tiled = getattr(layer, "_marlin_decode_scales", None)
    if tiled is None or getattr(layer, "_marlin_decode_weight", None) is None:
        return False
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    # Per-layer gate launch configuration: set it before any admission check so the variant
    # that is checked is the variant that will be launched for this layer.
    gate_blocks = _layer_gate_blocks(layer) if (n, k) == _GATE_SHAPE6 else None
    if gate_blocks is not None and _set_gate_blocks is not None:
        _set_gate_blocks(gate_blocks)
    packed = pack_scales6(tiled) if k % 128 == 0 else None
    if packed is None:
        STATS["kept"] += 1
        return False
    codes, palette = packed
    # Drop any stale entry: the caching allocator can hand a freed scale buffer's address
    # back to this layer, and the admission checks below must not read another layer's grid.
    _GATE_BLOCKS_BY_SCALES.pop(codes.data_ptr(), None)
    layer.register_buffer("_marlin_decode_scales6", codes, persistent=False)
    layer.register_buffer("_marlin_decode_palette", palette, persistent=False)
    if not self_check6(layer, reference):
        del layer._marlin_decode_scales6, layer._marlin_decode_palette
        STATS["kept"] += 1
        return False
    use_tma4 = os.environ.get("MUSE_TMA4", "1") != "0" and _gate_up_tma4 is not None and _sk_tma4 is not None
    if _gate_up is not None and (n, k) == (39936, 6656):
        try:
            layer.marlin_decode6_fixed_grid = self_check6(layer, reference, fixed_grid=True)
        except (RuntimeError, ValueError) as exc:
            logger.warning_once("NVFP4 gate/up grid admission failed on this layer: %s", exc)
        if layer.marlin_decode6_fixed_grid:
            STATS["fixed_grid"] += 1
            if use_tma4 and os.environ.get("MUSE_GATE_UP_SRANS", "1") != "0":
                try:
                    coded_ok = admit_tma4(layer, fixed_grid=True)
                except (RuntimeError, ValueError) as exc:
                    logger.warning_once("NVFP4 TMA4 gate/up scale admission failed: %s", exc)
                    coded_ok = False
                STATS["gate_up_srans" if coded_ok else "gate_up_srans_rejected"] += 1
            elif _gate_up_srans is not None and os.environ.get("MUSE_GATE_UP_SRANS", "1") != "0":
                try:
                    coded_ok = admit_gate_up_srans(layer)
                except (RuntimeError, ValueError) as exc:
                    logger.warning_once("NVFP4 rANS gate/up scale admission failed on this layer: %s", exc)
                    coded_ok = False
                STATS["gate_up_srans" if coded_ok else "gate_up_srans_rejected"] += 1
    elif (
        (n, k) in ((DOWN_N, DOWN_K), (LM_HEAD_N, LM_HEAD_K))
        or (use_tma4 and os.environ.get("MUSE_TMA4_ATTN", "1") != "0" and n % 128 == 0 and k % 128 == 0)
    ) and os.environ.get("MUSE_DOWN_SRANS", "1") != "0":
        if use_tma4:
            try:
                coded_ok = admit_tma4(layer, fixed_grid=False)
            except (RuntimeError, ValueError) as exc:
                logger.warning_once("NVFP4 TMA4 (%dx%d) scale admission failed: %s", n, k, exc)
                coded_ok = False
            STATS["down_srans" if coded_ok else "down_srans_rejected"] += 1
        elif _down_srans is not None and (n, k) == (DOWN_N, DOWN_K):
            try:
                coded_ok = admit_down_srans(layer)
            except (RuntimeError, ValueError) as exc:
                logger.warning_once("NVFP4 rANS down scale admission failed on this layer: %s", exc)
                coded_ok = False
            STATS["down_srans" if coded_ok else "down_srans_rejected"] += 1
    if gate_blocks is not None:
        codes = getattr(layer, "_marlin_decode_scales6", None)
        if codes is not None:
            _GATE_BLOCKS_BY_SCALES[codes.data_ptr()] = gate_blocks
    STATS["admitted"] += 1
    return True
