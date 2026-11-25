# SPDX-License-Identifier: Apache-2.0
"""Decode (M == 1) NVFP4 GEMV with 6-bit block scales: the board's tiled Marlin GEMV (muse_marlin_gemv) reading each
layer's group-16 FP8 block scales as 6-bit codes into a 64-entry palette, 3/4 of the scale bytes. The decoded bytes are
the same bytes, in the same shared-memory fragment, so the kernel computes the same bits; every layer is admitted only
after a bitwise comparison with the uncompressed tiled GEMV on its own weight."""
import ctypes
import math
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
_gate_up_tma4_swiglu = getattr(_lib, "muse_marlin6_gate_up_tma4_swiglu", None)
if _gate_up_tma4_swiglu is not None:
    _gate_up_tma4_swiglu.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    _gate_up_tma4_swiglu.restype = ctypes.c_int
_gate_up_tma4_swiglu_fp8 = getattr(_lib, "muse_marlin6_gate_up_tma4_swiglu_fp8", None)
if _gate_up_tma4_swiglu_fp8 is not None:
    _gate_up_tma4_swiglu_fp8.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    _gate_up_tma4_swiglu_fp8.restype = ctypes.c_int
_sk_tma4_ilv_swiglu = getattr(_lib, "muse_marlin6_sk_tma4_ilv_swiglu", None)
if _sk_tma4_ilv_swiglu is not None:
    _sk_tma4_ilv_swiglu.argtypes = _gemv6.argtypes
    _sk_tma4_ilv_swiglu.restype = ctypes.c_int
_sk_tma4_ilv_swiglu_m = getattr(_lib, "muse_marlin6_sk_tma4_ilv_swiglu_m", None)
if _sk_tma4_ilv_swiglu_m is not None:
    _sk_tma4_ilv_swiglu_m.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    _sk_tma4_ilv_swiglu_m.restype = ctypes.c_int
_sk_tma4_ilv_swiglu_fp8 = getattr(_lib, "muse_marlin6_sk_tma4_ilv_swiglu_fp8", None)
if _sk_tma4_ilv_swiglu_fp8 is not None:
    _sk_tma4_ilv_swiglu_fp8.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    _sk_tma4_ilv_swiglu_fp8.restype = ctypes.c_int
_sk_tma4 = getattr(_lib, "muse_marlin6_sk_tma4", None)
if _sk_tma4 is not None:
    _sk_tma4.argtypes = _gemv6.argtypes
    _sk_tma4.restype = ctypes.c_int
_sk_tma4_lmpost = getattr(_lib, "muse_marlin6_sk_tma4_lmpost", None)
if _sk_tma4_lmpost is not None:
    _sk_tma4_lmpost.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    _sk_tma4_lmpost.restype = ctypes.c_int
_sk_tma4_ilv_lmpost = getattr(_lib, "muse_marlin6_sk_tma4_ilv_lmpost", None)
if _sk_tma4_ilv_lmpost is not None:
    _sk_tma4_ilv_lmpost.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    _sk_tma4_ilv_lmpost.restype = ctypes.c_int
_sk_tma4_ilv_lmpost_m = getattr(_lib, "muse_marlin6_sk_tma4_ilv_lmpost_m", None)
if _sk_tma4_ilv_lmpost_m is not None:
    _sk_tma4_ilv_lmpost_m.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
    _sk_tma4_ilv_lmpost_m.restype = ctypes.c_int
_sk_tma4_ilv = getattr(_lib, "muse_marlin6_sk_tma4_ilv", None)
if _sk_tma4_ilv is not None:
    _sk_tma4_ilv.argtypes = _gemv6.argtypes
    _sk_tma4_ilv.restype = ctypes.c_int
_sk_tma4_ilv_m = getattr(_lib, "muse_marlin6_sk_tma4_ilv_m", None)
if _sk_tma4_ilv_m is not None:
    _sk_tma4_ilv_m.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    _sk_tma4_ilv_m.restype = ctypes.c_int
_sk_tma4_gated = getattr(_lib, "muse_marlin6_sk_tma4_gated", None)
if _sk_tma4_gated is not None:
    _sk_tma4_gated.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    _sk_tma4_gated.restype = ctypes.c_int
_sk_tma4_ilv_gated = getattr(_lib, "muse_marlin6_sk_tma4_ilv_gated", None)
if _sk_tma4_ilv_gated is not None:
    _sk_tma4_ilv_gated.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    _sk_tma4_ilv_gated.restype = ctypes.c_int
_sk_tma4_ilv_gated_m = getattr(_lib, "muse_marlin6_sk_tma4_ilv_gated_m", None)
if _sk_tma4_ilv_gated_m is not None:
    _sk_tma4_ilv_gated_m.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    _sk_tma4_ilv_gated_m.restype = ctypes.c_int
_get_gate_blocks = getattr(_lib, "muse_marlin6_get_gate_blocks", None)
if _get_gate_blocks is not None:
    _get_gate_blocks.argtypes = []
    _get_gate_blocks.restype = ctypes.c_int
_set_pdl = getattr(_lib, "muse_marlin6_set_pdl", None)
if _set_pdl is not None:
    _set_pdl.argtypes = [ctypes.c_int]
    _set_pdl.restype = ctypes.c_int


def set_pdl(mode: int) -> int | None:
    """Set SM121 PDL mode for decode6 TMA4 launches (0=off, 1=role-smart, 2=all, <0=env default)."""
    if _set_pdl is None:
        return None
    return int(_set_pdl(int(mode)))
_tma4_p1 = getattr(_lib, "muse_tma4_pass1", None)
if _tma4_p1 is not None:
    _tma4_p1.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    _tma4_p1.restype = ctypes.c_int
_tma4_p2 = getattr(_lib, "muse_tma4_pass2", None)
if _tma4_p2 is not None:
    _tma4_p2.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    _tma4_p2.restype = ctypes.c_int


GATE_UP_N, GATE_UP_K = 39936, 6656
DOWN_N, DOWN_K = 6656, 19968
OPROJ_N, OPROJ_K = 6656, 4096
LM_HEAD_N, LM_HEAD_K = 202048, 6656
LM_HEAD_PAD128_N = 202112
ATTN_ILV_SHAPES = {(4608, 6656), (6656, 4096), (4096, 6656)}
_TMA4_SCHED_CACHE: dict[tuple, tuple] = {}
_DOWN_ILV_SCALES: set[int] = set()
_LM_SPLIT2_SCALES: set[int] = set()
_LM_SPLIT2_CACHE: dict[int, bool] = {}
_LM_PAD128_SCALES: set[int] = set()
_LM_PAD128_CACHE: dict[int, bool] = {}
_SWIGLU_GATE_UP: dict[int, dict] = {}
# The paired gate_up layout has 624 BF16 tiles of 64 columns; the multi-row verify output is un-paired by one gather.
# Built once at weight registration, before CUDA graph capture.
_GATE_UP_TILE_ORDER: dict[torch.device, torch.Tensor] = {}
_GATED_OPROJ: set[int] = set()
_ALL_OPROJ_GATED_READY: bool = False
_GATED_OPROJ_M: set[int] = set()
_ALL_OPROJ_GATED_M_READY: bool = False
_LMPOST_BUFS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_LMPOST_M_BUFS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_consume_down_swiglu = None
_notify_gate_up_swiglu = None
_consume_oproj_sigmoid = None


def _nv6_gate_blocks() -> int:
    if _get_gate_blocks is not None:
        return int(_get_gate_blocks())
    env = os.environ.get("MUSE_NV6_GATE_BLOCKS", "")
    if not env:
        return 48
    try:
        v = int(env)
    except ValueError:
        return 48
    return 24 if v in (0, 24) else (v if v in (32, 48) else 48)

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
    if os.environ.get("MUSE_TMA4_KL_TABLE", "1") != "0":
        nz = [s for s in range(64) if int(counts[s]) > 0]
        if nz:
            for s in nz:
                f[s] = max(1, (int(counts[s]) * SRANS_M) // total)
            diff = SRANS_M - int(f.sum())
            while diff > 0:
                best_s = max(nz, key=lambda s: int(counts[s]) * math.log((int(f[s]) + 1) / int(f[s])))
                f[best_s] += 1
                diff -= 1
            while diff < 0:
                cands = [s for s in nz if int(f[s]) > 1]
                if not cands:
                    raise ValueError("cannot normalise the scale frequencies")
                best_s = min(cands, key=lambda s: int(counts[s]) * math.log(int(f[s]) / (int(f[s]) - 1)))
                f[best_s] -= 1
                diff += 1
    else:
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


def is_lm_split2(scales: torch.Tensor) -> bool:
    ptr = scales.data_ptr()
    if ptr in _LM_SPLIT2_SCALES:
        return True
    cached = _LM_SPLIT2_CACHE.get(ptr)
    if cached is not None:
        return cached
    if not is_tma4_ilv(scales) or scales.numel() < 4112 or torch.cuda.is_current_stream_capturing():
        return False
    res = int(scales.reshape(-1)[4100:4104].view(torch.int32).item()) == 96
    _LM_SPLIT2_CACHE[ptr] = res
    if res:
        _LM_SPLIT2_SCALES.add(ptr)
    return res


def is_lm_pad128(scales: torch.Tensor) -> bool:
    ptr = scales.data_ptr()
    if ptr in _LM_PAD128_SCALES:
        return True
    cached = _LM_PAD128_CACHE.get(ptr)
    if cached is not None:
        return cached
    if not is_tma4_ilv(scales) or scales.numel() < 4112 or torch.cuda.is_current_stream_capturing():
        return False
    res = int(scales.reshape(-1)[4108:4112].view(torch.int32).item()) == 256
    _LM_PAD128_CACHE[ptr] = res
    if res:
        _LM_PAD128_SCALES.add(ptr)
    return res


def _get_tma4_sched(
    n: int,
    k: int,
    fixed_grid: bool,
    device: torch.device,
    *,
    gate_v48: bool = False,
    gate_blocks: int = 48,
    split2: bool = False,
    lm_pad128: bool = False,
    lm_blocks: int = 48,
):
    key = (n, k, fixed_grid, gate_v48, gate_blocks, split2, lm_pad128, lm_blocks, device)
    cached = _TMA4_SCHED_CACHE.get(key)
    if cached is not None:
        return cached
    tile_n = 128 if n % 128 == 0 else 64
    threads = 256 if tile_n == 128 else 128
    n_tiles = n // tile_n
    k_tiles = k // 128
    stages: list[tuple[int, int]] = []
    blk_off = [0]
    blk_sb_off = [0]
    blk_iters = []
    if lm_pad128:
        num_blks = lm_blocks
        for b in range(num_blks):
            s_start = len(stages)
            col = b
            while col < n_tiles:
                for kt in range(k_tiles):
                    stages.append((col, kt))
                col += num_blks
            blk_off.append(len(stages))
            it = len(stages) - s_start
            blk_iters.append(it)
            blk_sb_off.append(blk_sb_off[-1] + (it + 3) // 4)
    elif fixed_grid:
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
    elif split2:
        vgrid = 48
        num_blks = 96
        part2_mn_tiles = n_tiles
        part1_mn_iters = 0
        if n_tiles > vgrid:
            part2_mn_tiles = n_tiles % vgrid
            if part2_mn_tiles * 3 <= vgrid:
                part2_mn_tiles += vgrid
            part1_mn_iters = (n_tiles - part2_mn_tiles) // vgrid
        lo_p1 = (part1_mn_iters + 1) // 2
        iters = (k_tiles * part2_mn_tiles + vgrid - 1) // vgrid
        base_col = n_tiles - part2_mn_tiles
        total_p2 = part2_mn_tiles * k_tiles
        for b in range(vgrid):
            s_start = len(stages)
            for p in range(lo_p1):
                col = b + p * vgrid
                for kt in range(k_tiles):
                    stages.append((col, kt))
            blk_off.append(len(stages))
            it = len(stages) - s_start
            blk_iters.append(it)
            blk_sb_off.append(blk_sb_off[-1] + (it + 3) // 4)
        for b in range(vgrid):
            s_start = len(stages)
            for p in range(lo_p1, part1_mn_iters):
                col = b + p * vgrid
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
    else:
        num_blks = 48
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


def _build_ilv_weight(
    w: torch.Tensor,
    n: int,
    k: int,
    gate_v48: bool,
    gate_blocks: int,
    split2: bool = False,
    lm_pad128: bool = False,
    lm_blocks: int = 48,
) -> torch.Tensor:
    tile_n = 128 if n % 128 == 0 else 64
    k_tiles = k // 128
    *_, num_blks, _, _, _, stages, blk_off, blk_iters = _get_tma4_sched(
        n, k, False, w.device, gate_v48=gate_v48, gate_blocks=gate_blocks,
        split2=split2, lm_pad128=lm_pad128, lm_blocks=lm_blocks,
    )
    max_it = max(blk_iters)
    perm = []
    for s in range(max_it):
        for b in range(num_blks):
            if s < blk_iters[b]:
                col, kt = stages[blk_off[b] + s]
                perm.append(col * k_tiles + kt)
    assert len(perm) == (n // tile_n) * k_tiles
    idx_t = torch.tensor(perm, dtype=torch.long, device=w.device)
    return w.reshape(len(perm), -1).index_select(0, idx_t).contiguous().view_as(w)


def _pad_lm_head_128(
    w64: torch.Tensor, s64: torch.Tensor, pal: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad [3157, 416, 128] int32 weights and [3157, 416, 64] uint8 scales by 64 zero cols -> TileN=128 (202112)."""
    k_groups = w64.shape[1]
    w_pad = torch.zeros((1, k_groups, 128), dtype=w64.dtype, device=w64.device)
    w128 = (
        torch.cat((w64, w_pad), dim=0)
        .view(LM_HEAD_PAD128_N // 128, 2, k_groups, 128)
        .permute(0, 2, 1, 3)
        .reshape(LM_HEAD_PAD128_N // 128, k_groups, 256)
        .contiguous()
    )
    s64_u8 = s64.contiguous().view(torch.uint8)
    fill_byte = int(pal[0].item())
    s_pad = torch.full((1, k_groups, 64), fill_byte, dtype=torch.uint8, device=s64.device)
    s128 = (
        torch.cat((s64_u8, s_pad), dim=0)
        .view(LM_HEAD_PAD128_N // 128, 2, k_groups, 64)
        .permute(0, 2, 1, 3)
        .reshape(LM_HEAD_PAD128_N // 128, k_groups, 128)
        .contiguous()
    )
    return w128, s128


def _unbuild_ilv_weight(
    w_ilv: torch.Tensor, n: int, k: int, gate_v48: bool = False, gate_blocks: int = 48, split2: bool = False
) -> torch.Tensor:
    tile_n = 128 if n % 128 == 0 else 64
    k_tiles = k // 128
    *_, num_blks, _, _, _, stages, blk_off, blk_iters = _get_tma4_sched(
        n, k, False, w_ilv.device, gate_v48=gate_v48, gate_blocks=gate_blocks, split2=split2
    )
    max_it = max(blk_iters)
    perm = []
    for s in range(max_it):
        for b in range(num_blks):
            if s < blk_iters[b]:
                col, kt = stages[blk_off[b] + s]
                perm.append(col * k_tiles + kt)
    assert len(perm) == (n // tile_n) * k_tiles
    inv = [0] * len(perm)
    for dst_i, src_i in enumerate(perm):
        inv[src_i] = dst_i
    idx_t = torch.tensor(inv, dtype=torch.long, device=w_ilv.device)
    return w_ilv.reshape(len(perm), -1).index_select(0, idx_t).contiguous().view_as(w_ilv)


def encode_tma4_scales(
    tiled_scales: torch.Tensor,
    palette: torch.Tensor,
    n: int,
    k: int,
    fixed_grid: bool,
    *,
    gate_v48: bool = False,
    gate_blocks: int = 48,
    split2: bool = False,
    lm_pad128: bool = False,
    lm_blocks: int = 48,
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
    ) = _get_tma4_sched(
        n, k, fixed_grid, dev, gate_v48=gate_v48, gate_blocks=gate_blocks,
        split2=split2, lm_pad128=lm_pad128, lm_blocks=lm_blocks,
    )
    warps = threads // 32
    rans_cap = 4096 if lm_pad128 else (3200 if threads == 256 else 2048)
    sb_wc = torch.empty(total_sbs * warps, dtype=torch.int32, device=dev)
    with torch.cuda.device(dev):
        stream = torch.cuda.current_stream().cuda_stream
        st = _tma4_p1(
            sym.data_ptr(), k_tiles, stage_col_k.data_ptr(), blk_stage_off.data_ptr(),
            blk_sb_off.data_ptr(), sb_gsb_map.data_ptr(), enc_f.data_ptr(), enc_c.data_ptr(),
            sb_wc.data_ptr(), num_blks, threads, stream,
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
            num_blks, threads, stream,
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
        logger.info("NVFP4 decode GEMV with 6-bit block scales: %d layers, %d kept uncompressed",
                    STATS["admitted"], STATS["kept"])
    if (
        _consume_down_swiglu is not None
        and not fixed_grid
        and (n, k) == (DOWN_N, DOWN_K)
        and (x.numel() == k or (x.ndim == 2 and 1 <= x.shape[0] <= 16 and x.shape[1] == k))
    ):
        x_sw = _consume_down_swiglu(x, global_scale.data_ptr())
        if x_sw is not None:
            x = x_sw
    if (
        _consume_oproj_sigmoid is not None
        and not fixed_grid
        and (n, k) == (OPROJ_N, OPROJ_K)
        and (
            (x.numel() == k and global_scale.data_ptr() in _GATED_OPROJ and is_tma4(scales6))
            or (
                x.ndim == 2
                and 1 < x.shape[0] <= 16
                and x.shape[1] == k
                and global_scale.data_ptr() in _GATED_OPROJ_M
                and is_tma4_ilv(scales6)
            )
        )
    ):
        op_res = _consume_oproj_sigmoid(
            x, global_scale.data_ptr(), tiled_weight, scales6, palette, global_scale, workspace
        )
        if op_res is not None:
            return op_res
    if not x.is_contiguous() or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)
    lm_pad128 = (n, k) == (LM_HEAD_N, LM_HEAD_K) and is_lm_pad128(scales6)
    eff_n = LM_HEAD_PAD128_N if lm_pad128 else n
    blocks = 96 if (n, k) == (LM_HEAD_N, LM_HEAD_K) and is_lm_split2(scales6) else 48
    tile_n = 128 if eff_n % 128 == 0 else 64
    partials = x.new_empty((blocks, 16, tile_n), dtype=torch.float32)
    with torch.cuda.device(x.device):
        use_grid = fixed_grid and _gate_up is not None and x.numel() == k and (n, k) == (39936, 6656)
        sw_entry = (
            _SWIGLU_GATE_UP.get(global_scale.data_ptr())
            if (x.numel() == k and (n, k) == (39936, 6656) and _notify_gate_up_swiglu is not None and is_tma4(scales6))
            else None
        )
        if sw_entry is not None and sw_entry["state"] == "admitted":
            out_buf = sw_entry["swiglu_out"]
            st_stream = torch.cuda.current_stream().cuda_stream
            if sw_entry.get("is_fp8", False):
                fp8_buf = sw_entry["swiglu_fp8_out"]
                fp8_sc_ptr = sw_entry["fp8_scale_ptr"]
                if sw_entry["is_ilv"]:
                    status = _sk_tma4_ilv_swiglu_fp8(
                        x.data_ptr(), tiled_weight.data_ptr(), out_buf.data_ptr(), partials.data_ptr(),
                        scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(),
                        fp8_sc_ptr, fp8_buf.data_ptr(), n, k, 48, x.device.index, st_stream,
                    )
                else:
                    status = _gate_up_tma4_swiglu_fp8(
                        x.data_ptr(), tiled_weight.data_ptr(), out_buf.data_ptr(), partials.data_ptr(),
                        scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(),
                        fp8_sc_ptr, fp8_buf.data_ptr(), n, k, 48, x.device.index, st_stream,
                    )
            elif sw_entry["is_ilv"]:
                status = _sk_tma4_ilv_swiglu(
                    x.data_ptr(), tiled_weight.data_ptr(), out_buf.data_ptr(), partials.data_ptr(),
                    scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(),
                    n, k, 48, x.device.index, st_stream,
                )
            else:
                status = _gate_up_tma4_swiglu(
                    x.data_ptr(), tiled_weight.data_ptr(), out_buf.data_ptr(), partials.data_ptr(),
                    scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(),
                    0, n, k, 48, x.device.index, st_stream,
                )
            if status:
                raise RuntimeError(f"SM121 Marlin gate_up SwiGLU launch failed with CUDA error {status}")
            _notify_gate_up_swiglu(out_buf.data_ptr(), sw_entry, True, None)
            return out_buf
        if x.ndim == 2 and 1 < x.shape[0] <= 16 and is_tma4_ilv(scales6) and _sk_tma4_ilv_m is not None:
            m = int(x.shape[0])
            sw_entry_m = (
                _SWIGLU_GATE_UP.get(global_scale.data_ptr())
                if (
                    (n, k) == (GATE_UP_N, GATE_UP_K)
                    and _sk_tma4_ilv_swiglu_m is not None
                    and _notify_gate_up_swiglu is not None
                )
                else None
            )
            if (
                sw_entry_m is not None
                and sw_entry_m["state"] == "admitted"
                and sw_entry_m.get("swiglu_out_m") is not None
            ):
                out_buf_m = sw_entry_m["swiglu_out_m"][:m]
                status = _sk_tma4_ilv_swiglu_m(
                    x.data_ptr(), tiled_weight.data_ptr(), out_buf_m.data_ptr(), partials.data_ptr(),
                    scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(),
                    m, n, k, 48, x.device.index, torch.cuda.current_stream().cuda_stream,
                )
                if status:
                    raise RuntimeError(f"SM121 Marlin multi-M gate_up SwiGLU launch failed with CUDA error {status}")
                _notify_gate_up_swiglu(out_buf_m.data_ptr(), sw_entry_m, True, None)
                return out_buf_m
            output = x.new_empty((m, eff_n))
            status = _sk_tma4_ilv_m(
                x.data_ptr(), tiled_weight.data_ptr(), output.data_ptr(), partials.data_ptr(),
                scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(),
                m, eff_n, k, blocks, x.device.index, torch.cuda.current_stream().cuda_stream,
            )
            if status:
                raise RuntimeError(f"SM121 Marlin multi-M (scale6) launch failed with CUDA error {status}")
            if lm_pad128:
                return output[:, :n].contiguous()
            if (n, k) == (GATE_UP_N, GATE_UP_K) and global_scale.data_ptr() in _SWIGLU_GATE_UP:
                order = _GATE_UP_TILE_ORDER.get(output.device)
                if order is not None and output.is_contiguous():
                    # One exact gather of the 624 64-column tiles (newjordan 833fb6c4) instead of two strided
                    # copies plus a concatenation: same values, same positions.
                    return output.view(m, 624, 64).index_select(1, order).view(m, n)
                head = output[:, :33792].view(m, 264, 2, 64)
                return torch.cat(
                    (
                        head[:, :, 0, :].reshape(m, 16896),
                        output[:, 33792:36864],
                        head[:, :, 1, :].reshape(m, 16896),
                        output[:, 36864:],
                    ),
                    dim=1,
                )
            return output
        output = x.new_empty((*x.shape[:-1], n))
        kernel = _gate_up if use_grid else _gemv6
        if is_tma4_ilv(scales6):
            if x.numel() == k and _sk_tma4_ilv is not None:
                kernel = _sk_tma4_ilv
            else:
                raise RuntimeError(f"TMA4-ILV rANS scales unsupported for shape ({n}, {k}, fixed={fixed_grid})")
        elif is_tma4(scales6):
            if use_grid and _gate_up_tma4 is not None:
                kernel = _gate_up_tma4
            elif not fixed_grid and x.numel() == k and _sk_tma4 is not None:
                kernel = _sk_tma4
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
        status = kernel(
            x.data_ptr(), tiled_weight.data_ptr(), output.data_ptr(), partials.data_ptr(),
            scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), workspace.data_ptr(), eff_n, k, blocks,
            x.device.index, torch.cuda.current_stream().cuda_stream,
        )
    if status:
        raise RuntimeError(f"SM121 Marlin GEMV (scale6) launch failed with CUDA error {status}")
    if sw_entry is not None and sw_entry["state"] == "prechecked" and not torch.cuda.is_current_stream_capturing():
        _notify_gate_up_swiglu(
            output.data_ptr(),
            sw_entry,
            False,
            (x, tiled_weight, scales6, palette, global_scale, workspace, partials),
        )
    return output


def _get_lmpost_bufs(dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    bufs = _LMPOST_BUFS.get(idx)
    if bufs is None:
        bufs = (
            torch.zeros(256, dtype=torch.int64, device=dev),
            torch.empty((96, 16, 128), dtype=torch.float32, device=dev),
        )
        _LMPOST_BUFS[idx] = bufs
    return bufs


def _get_lmpost_m_bufs(dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Dedicated scratch/partials for MultiM (M=2..16) lm_head+lmpost so g_bv[0..2047]
    and g_bi[0..2047] never overlap M=1's g_done counter (g_done sits at int32[4096])."""
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    bufs = _LMPOST_M_BUFS.get(idx)
    if bufs is None:
        bufs = (
            torch.zeros(2064, dtype=torch.int64, device=dev),
            torch.empty((96, 16, 128), dtype=torch.float32, device=dev),
        )
        _LMPOST_M_BUFS[idx] = bufs
    return bufs


def launch_lmhead_lmpost6(
    x: torch.Tensor,
    layer: torch.nn.Module,
    lut: torch.Tensor,
    out_token: torch.Tensor,
    org_vocab_size: int,
) -> bool:
    s6 = getattr(layer, "_marlin_decode_scales6", None)
    if (
        s6 is None
        or not x.is_contiguous()
        or x.data_ptr() % 16 != 0
        or x.shape != (1, layer.input_size_per_partition)
        or out_token.shape != (1, 1)
        or out_token.dtype != torch.int64
    ):
        return False
    is_ilv = is_tma4_ilv(s6)
    fn = _sk_tma4_ilv_lmpost if is_ilv else _sk_tma4_lmpost
    if fn is None:
        return False
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    lm_pad128 = is_ilv and (n, k) == (LM_HEAD_N, LM_HEAD_K) and is_lm_pad128(s6)
    eff_n = LM_HEAD_PAD128_N if lm_pad128 else n
    blocks = 96 if (is_ilv and is_lm_split2(s6)) else 48
    scratch, partials = _get_lmpost_bufs(x.device)
    with torch.cuda.device(x.device):
        rc = fn(
            x.data_ptr(),
            layer._marlin_decode_weight.data_ptr(),
            scratch.data_ptr(),
            partials.data_ptr(),
            s6.data_ptr(),
            layer._marlin_decode_palette.data_ptr(),
            layer.weight_global_scale.data_ptr(),
            layer.workspace.data_ptr(),
            lut.data_ptr(),
            out_token.data_ptr(),
            int(org_vocab_size),
            eff_n,
            k,
            blocks,
            x.device.index,
            torch.cuda.current_stream().cuda_stream,
        )
    if rc:
        raise RuntimeError(f"muse_marlin6_sk_tma4_lmpost failed with CUDA error {rc}")
    return True


def admit_tma4_lmpost(
    layer: torch.nn.Module,
    lut: torch.Tensor,
    org_vocab_size: int,
    lmpost_argmax,
    mult: float,
    cap: float | None,
) -> bool:
    if os.environ.get("MUSE_FUSE_LMHEAD_LMPOST", "1") == "0":
        return False
    scales6 = getattr(layer, "_marlin_decode_scales6", None)
    if (
        scales6 is None
        or getattr(layer, "_marlin_decode_weight", None) is None
        or getattr(layer, "_marlin_decode_palette", None) is None
        or getattr(layer, "marlin_decode6_fixed_grid", False)
        or not is_tma4(scales6)
    ):
        return False
    is_ilv = is_tma4_ilv(scales6)
    if (is_ilv and _sk_tma4_ilv_lmpost is None) or (not is_ilv and _sk_tma4_lmpost is None):
        return False
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    pad128 = is_ilv and (n, k) == (LM_HEAD_N, LM_HEAD_K) and is_lm_pad128(scales6)
    if k % 128 != 0 or (not pad128 and (n % 128 == 0 or n % 64 != 0)) or not (0 < org_vocab_size <= n):
        return False
    dev = scales6.device
    scratch, _ = _get_lmpost_bufs(dev)
    gen = torch.Generator(device=dev).manual_seed(20261125)
    out_tok = torch.empty((1, 1), dtype=torch.int64, device=dev)
    for mag in (0.01, 0.5, 2.0, 32.0):
        x = (torch.randn((1, k), generator=gen, device=dev) * mag).to(torch.bfloat16).contiguous()
        raw = decode_linear6(
            x, layer._marlin_decode_weight, scales6, layer._marlin_decode_palette,
            layer.weight_global_scale, layer.workspace, n, k, False,
        )
        if org_vocab_size < n:
            raw = raw[:, :org_vocab_size].contiguous()
        ref = lmpost_argmax(raw, mult, cap)
        out_tok.fill_(-1)
        if not launch_lmhead_lmpost6(x, layer, lut, out_tok, org_vocab_size):
            return False
        torch.cuda.synchronize(dev)
        if (
            ref is None
            or not torch.equal(out_tok, ref)
            or int(torch.count_nonzero(layer.workspace).item()) != 0
            or int(scratch.view(torch.int32)[256].item()) != 0
        ):
            logger.warning("Fused lm_head+lmpost self-check mismatch; keeping 2-kernel path")
            return False
    layer.marlin_decode6_lmpost_admitted = True
    logger.info(
        "Fused lm_head+lmpost decode6 kernel admitted bitwise (n=%d, vocab=%d, k=%d, ilv=%s, split2=%s, pad128=%s)",
        n, org_vocab_size, k, is_ilv, is_lm_split2(scales6), pad128,
    )
    return True


def launch_lmhead_lmpost6_m(
    x: torch.Tensor,
    layer: torch.nn.Module,
    lut: torch.Tensor,
    out_tokens: torch.Tensor,
    org_vocab_size: int,
) -> bool:
    s6 = getattr(layer, "_marlin_decode_scales6", None)
    if (
        s6 is None
        or _sk_tma4_ilv_lmpost_m is None
        or not is_tma4_ilv(s6)
        or not x.is_contiguous()
        or x.data_ptr() % 16 != 0
        or x.ndim != 2
        or not (1 <= x.shape[0] <= 16)
        or x.shape[1] != layer.input_size_per_partition
        or not out_tokens.is_contiguous()
        or out_tokens.dtype != torch.int64
        or out_tokens.numel() < x.shape[0]
    ):
        return False
    m = int(x.shape[0])
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    lm_pad128 = (n, k) == (LM_HEAD_N, LM_HEAD_K) and is_lm_pad128(s6)
    eff_n = LM_HEAD_PAD128_N if lm_pad128 else n
    blocks = 96 if is_lm_split2(s6) else 48
    scratch, partials = _get_lmpost_m_bufs(x.device)
    with torch.cuda.device(x.device):
        rc = _sk_tma4_ilv_lmpost_m(
            x.data_ptr(),
            layer._marlin_decode_weight.data_ptr(),
            scratch.data_ptr(),
            partials.data_ptr(),
            s6.data_ptr(),
            layer._marlin_decode_palette.data_ptr(),
            layer.weight_global_scale.data_ptr(),
            layer.workspace.data_ptr(),
            lut.data_ptr(),
            out_tokens.data_ptr(),
            int(org_vocab_size),
            m,
            eff_n,
            k,
            blocks,
            x.device.index,
            torch.cuda.current_stream().cuda_stream,
        )
    if rc:
        raise RuntimeError(f"muse_marlin6_sk_tma4_ilv_lmpost_m failed with CUDA error {rc}")
    return True


def admit_tma4_lmpost_m(
    layer: torch.nn.Module,
    lut: torch.Tensor,
    org_vocab_size: int,
    lmpost_argmax,
    mult: float,
    cap: float | None,
) -> bool:
    if os.environ.get("MUSE_FUSE_LMHEAD_LMPOST_M", "1") == "0":
        return False
    scales6 = getattr(layer, "_marlin_decode_scales6", None)
    if (
        scales6 is None
        or _sk_tma4_ilv_lmpost_m is None
        or getattr(layer, "_marlin_decode_weight", None) is None
        or getattr(layer, "_marlin_decode_palette", None) is None
        or getattr(layer, "marlin_decode6_fixed_grid", False)
        or not is_tma4_ilv(scales6)
    ):
        return False
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    pad128 = (n, k) == (LM_HEAD_N, LM_HEAD_K) and is_lm_pad128(scales6)
    if (n, k) != (LM_HEAD_N, LM_HEAD_K) or not (0 < org_vocab_size <= n):
        return False
    dev = scales6.device
    scratch, _ = _get_lmpost_m_bufs(dev)
    gen = torch.Generator(device=dev).manual_seed(20261128)
    for cm in (7, 12, 16, 2):
        out_tok = torch.empty((1, cm), dtype=torch.int64, device=dev)
        for mag in (0.05, 1.5, 16.0):
            x = (torch.randn((cm, k), generator=gen, device=dev) * mag).to(torch.bfloat16).contiguous()
            raw = decode_linear6(
                x, layer._marlin_decode_weight, scales6, layer._marlin_decode_palette,
                layer.weight_global_scale, layer.workspace, n, k, False,
            )
            if org_vocab_size < n:
                raw = raw[:, :org_vocab_size].contiguous()
            refs = []
            for r in range(cm):
                ref_r = lmpost_argmax(raw[r : r + 1].contiguous(), mult, cap)
                if ref_r is None:
                    return False
                refs.append(ref_r.view(-1))
            ref = torch.cat(refs, dim=0).view(1, cm)
            out_tok.fill_(-1)
            if not launch_lmhead_lmpost6_m(x, layer, lut, out_tok, org_vocab_size):
                return False
            torch.cuda.synchronize(dev)
            if (
                not torch.equal(out_tok, ref)
                or int(torch.count_nonzero(layer.workspace).item()) != 0
                or int(scratch.view(torch.int32)[4096].item()) != 0
            ):
                logger.warning("Fused multi-M lm_head+lmpost self-check mismatch (m=%d); keeping 2-kernel path", cm)
                return False
    layer.marlin_decode6_lmpost_m_admitted = True
    logger.info(
        "Fused multi-M lm_head+lmpost decode6 kernel admitted bitwise (n=%d, vocab=%d, k=%d, split2=%s, pad128=%s)",
        n, org_vocab_size, k, is_lm_split2(scales6), pad128,
    )
    return True


def _pair_gate_up_tensors(w_raw: torch.Tensor, s_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Repack [312, 416, 256] int32 weights and [312, 416, 128] uint8 scales so part-1 tiles 0..263
    pair 64 gate + 64 up cols in shared memory, part-1 tiles 264..287 hold pure full-K gate cols
    16896..19967, and part-2 tiles 288..311 hold pure split-K up cols 16896..19967."""
    w_head = (
        torch.stack((w_raw[:132], w_raw[156:288]), dim=0)
        .view(2, 132, 416, 2, 128)
        .permute(1, 3, 2, 0, 4)
        .reshape(264, 416, 256)
    )
    s_head = (
        torch.stack((s_raw[:132], s_raw[156:288]), dim=0)
        .view(2, 132, 416, 2, 64)
        .permute(1, 3, 2, 0, 4)
        .reshape(264, 416, 128)
    )
    w_p = torch.cat((w_head, w_raw[132:156], w_raw[288:312]), dim=0).contiguous()
    s_p = torch.cat((s_head, s_raw[132:156], s_raw[288:312]), dim=0).contiguous()
    return w_p, s_p


def register_all_gate_up_swiglu(model: torch.nn.Module) -> int:
    if os.environ.get("MUSE_GATE_UP_SWIGLU", "1") == "0":
        return 0
    want_fp8 = os.environ.get("MUSE_GATE_UP_SWIGLU_FP8", "0") != "0"
    try:
        import muse_norm_glue as _ng
        if _ng.lib() is None or not hasattr(_ng, "mul_silu_slice_bf16"):
            return 0
    except Exception:
        return 0
    try:
        import muse_rans_ext as _rans
        if not hasattr(_rans, "rans_gemv_auto") or not hasattr(_rans, "rans_gemv_fused_auto"):
            _rans = None
    except Exception:
        _rans = None
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return 0
    count_bf16 = 0
    count_fp8 = 0
    count_m = 0
    count_reused = 0
    for lyr in layers:
        mlp = getattr(lyr, "mlp", None)
        gu = getattr(mlp, "gate_up_proj", None)
        dn = getattr(mlp, "down_proj", None)
        if gu is None or dn is None:
            continue
        s6 = getattr(gu, "_marlin_decode_scales6", None)
        s_raw = getattr(gu, "_marlin_decode_scales", None)
        w = getattr(gu, "_marlin_decode_weight", None)
        pal = getattr(gu, "_marlin_decode_palette", None)
        ws = getattr(gu, "workspace", None)
        if (
            s6 is None or s_raw is None or w is None or pal is None
            or ws is None or ws.numel() < 48
            or not is_tma4(s6)
            or (gu.output_size_per_partition, gu.input_size_per_partition) != (GATE_UP_N, GATE_UP_K)
            or (dn.output_size_per_partition, dn.input_size_per_partition) != (DOWN_N, DOWN_K)
        ):
            continue
        dn_is_nvfp4 = getattr(dn, "_marlin_decode_scales6", None) is not None
        dn_packed = getattr(dn, "muse_fp8_packed", None)
        dn_in_sc = getattr(dn, "input_scale", None)
        dn_w_sc = getattr(dn, "weight_scale", None)
        is_fp8 = False
        if not dn_is_nvfp4:
            if (
                not want_fp8
                or _rans is None
                or dn_packed is None or not dn_packed.is_cuda or dn_packed.dtype != torch.uint8
                or dn_packed.numel() == 0 or dn_packed.numel() == DOWN_N * DOWN_K
                or dn_in_sc is None or not dn_in_sc.is_cuda or dn_in_sc.dtype != torch.float32 or dn_in_sc.numel() != 1
                or dn_w_sc is None or not dn_w_sc.is_cuda or dn_w_sc.dtype != torch.float32 or dn_w_sc.numel() not in (1, DOWN_N)
            ):
                continue
            is_fp8 = True
        is_ilv = is_tma4_ilv(s6)
        is_fg = bool(getattr(gu, "marlin_decode6_fixed_grid", False))
        if is_ilv:
            if (is_fp8 and _sk_tma4_ilv_swiglu_fp8 is None) or (not is_fp8 and _sk_tma4_ilv_swiglu is None):
                continue
        elif is_fg:
            if (is_fp8 and _gate_up_tma4_swiglu_fp8 is None) or (not is_fp8 and _gate_up_tma4_swiglu is None):
                continue
        else:
            continue
        gs_key = gu.weight_global_scale.data_ptr()
        if gs_key in _SWIGLU_GATE_UP:
            continue
        dev = s6.device
        gen = torch.Generator(device=dev).manual_seed(20261126 + count_bf16 + count_fp8)
        probes = []
        for mag, shift in ((0.05, 0.01), (0.8, -0.15), (3.0, 0.4), (16.0, 0.0)):
            xp = torch.randn((1, GATE_UP_K), generator=gen, device=dev) * mag + shift
            probes.append(xp.to(torch.bfloat16).contiguous())
        xp_ht = torch.randn((1, GATE_UP_K), generator=gen, device=dev) * torch.exp(
            torch.randn((1, GATE_UP_K), generator=gen, device=dev) * 0.75
        )
        probes.append(xp_ht.to(torch.bfloat16).contiguous())
        xp_canc = torch.randn((1, GATE_UP_K), generator=gen, device=dev) * 2.0
        xp_canc[:, 1::2] = -xp_canc[:, ::2]
        probes.append(xp_canc.to(torch.bfloat16).contiguous())
        # Compute exact 0-ULP reference outputs BEFORE modifying buffers
        refs = []
        with torch.cuda.device(dev):
            stream = torch.cuda.current_stream().cuda_stream
            for xp in probes:
                gu_ref = decode_linear6(
                    xp, w, s6, pal, gu.weight_global_scale, gu.workspace, GATE_UP_N, GATE_UP_K, is_fg
                )
                if is_fp8:
                    dn_ref = torch.empty((1, DOWN_N), dtype=torch.bfloat16, device=dev)
                    _rans.rans_gemv_fused_auto(
                        dn_packed, gu_ref.view(-1), gu_ref.view(-1),
                        dn_w_sc.reshape(-1), dn_in_sc.reshape(-1), dn_ref, DOWN_N, DOWN_K, 2,
                    )
                    refs.append(dn_ref)
                else:
                    sw_ref = torch.empty((1, DOWN_K), dtype=torch.bfloat16, device=dev)
                    _ng.mul_silu_slice_bf16(gu_ref, sw_ref, 1, stream)
                    refs.append(sw_ref)
            torch.cuda.synchronize(dev)

        w_unilv = _unbuild_ilv_weight(w, GATE_UP_N, GATE_UP_K) if is_ilv else w
        w_p, s_p = _pair_gate_up_tensors(w_unilv, s_raw)
        del w_unilv
        try:
            s6_p = encode_tma4_scales(s_p, pal, GATE_UP_N, GATE_UP_K, not is_ilv)
        except Exception:
            continue
        if is_ilv:
            w_p = _build_ilv_weight(w_p, GATE_UP_N, GATE_UP_K, False, 48)

        swiglu_out = torch.empty((1, GATE_UP_N), dtype=torch.bfloat16, device=dev)
        swiglu_fp8_out = torch.empty((1, DOWN_K), dtype=torch.float8_e4m3fn, device=dev) if is_fp8 else None
        partials = torch.empty((48, 16, 128), dtype=torch.float32, device=dev)
        ok = True
        with torch.cuda.device(dev):
            stream = torch.cuda.current_stream().cuda_stream
            for xp, ref_t in zip(probes, refs):
                swiglu_out.zero_()
                if is_fp8:
                    swiglu_fp8_out.zero_()
                    fn_fp8 = _sk_tma4_ilv_swiglu_fp8 if is_ilv else _gate_up_tma4_swiglu_fp8
                    rc = fn_fp8(
                        xp.data_ptr(), w_p.data_ptr(), swiglu_out.data_ptr(), partials.data_ptr(),
                        s6_p.data_ptr(), pal.data_ptr(), gu.weight_global_scale.data_ptr(),
                        gu.workspace.data_ptr(), dn_in_sc.data_ptr(), swiglu_fp8_out.data_ptr(),
                        GATE_UP_N, GATE_UP_K, 48, dev.index, stream,
                    )
                    dn_got = torch.empty((1, DOWN_N), dtype=torch.bfloat16, device=dev)
                    _rans.rans_gemv_auto(
                        dn_packed, swiglu_fp8_out.view(-1), dn_w_sc.reshape(-1), dn_in_sc.reshape(-1),
                        dn_got, DOWN_N, DOWN_K,
                    )
                    torch.cuda.synchronize(dev)
                    cmp_ok = torch.equal(dn_got.view(torch.int16), ref_t.view(torch.int16))
                else:
                    if is_ilv:
                        rc = _sk_tma4_ilv_swiglu(
                            xp.data_ptr(), w_p.data_ptr(), swiglu_out.data_ptr(), partials.data_ptr(),
                            s6_p.data_ptr(), pal.data_ptr(), gu.weight_global_scale.data_ptr(),
                            gu.workspace.data_ptr(), GATE_UP_N, GATE_UP_K, 48, dev.index, stream,
                        )
                    else:
                        rc = _gate_up_tma4_swiglu(
                            xp.data_ptr(), w_p.data_ptr(), swiglu_out.data_ptr(), partials.data_ptr(),
                            s6_p.data_ptr(), pal.data_ptr(), gu.weight_global_scale.data_ptr(),
                            gu.workspace.data_ptr(), 0, GATE_UP_N, GATE_UP_K, 48, dev.index, stream,
                        )
                    torch.cuda.synchronize(dev)
                    cmp_ok = torch.equal(swiglu_out[:, :DOWN_K].view(torch.int16), ref_t.view(torch.int16))
                if rc != 0 or int(torch.count_nonzero(gu.workspace).item()) != 0 or not cmp_ok:
                    ok = False
                    break
        if not ok:
            continue
        # Commit paired weights IN-PLACE into placed fast DRAM buffer (keeps w.data_ptr()!)
        w_backup = w.detach().clone()
        s6_backup = s6.detach().clone()
        w.copy_(w_p)
        del w_p
        s6_buf = s6._base if (s6._base is not None and s6._base.data_ptr() == s6.data_ptr()) else s6
        if s6_buf.is_contiguous() and s6_buf.numel() >= s6_p.numel():
            if is_ilv:
                _DOWN_ILV_SCALES.discard(s6.data_ptr())
            s6_flat = s6_buf.view(-1)
            s6_flat[: s6_p.numel()].copy_(s6_p.view(-1))
            s6_new = s6_flat[: s6_p.numel()].view(-1, 32)
            reused_s6 = True
        else:
            if is_ilv:
                _DOWN_ILV_SCALES.discard(s6.data_ptr())
            s6_new = s6_p
            reused_s6 = False
        if is_ilv:
            _DOWN_ILV_SCALES.add(s6_new.data_ptr())
        gu.register_buffer("_marlin_decode_scales6", s6_new, persistent=False)
        # Re-verify in-place buffer bitwise on probe 0!
        swiglu_out.zero_()
        with torch.cuda.device(dev):
            stream = torch.cuda.current_stream().cuda_stream
            if is_fp8:
                swiglu_fp8_out.zero_()
                fn_fp8 = _sk_tma4_ilv_swiglu_fp8 if is_ilv else _gate_up_tma4_swiglu_fp8
                rc = fn_fp8(
                    probes[0].data_ptr(), w.data_ptr(), swiglu_out.data_ptr(), partials.data_ptr(),
                    s6_new.data_ptr(), pal.data_ptr(), gu.weight_global_scale.data_ptr(),
                    gu.workspace.data_ptr(), dn_in_sc.data_ptr(), swiglu_fp8_out.data_ptr(),
                    GATE_UP_N, GATE_UP_K, 48, dev.index, stream,
                )
                dn_got0 = torch.empty((1, DOWN_N), dtype=torch.bfloat16, device=dev)
                _rans.rans_gemv_auto(
                    dn_packed, swiglu_fp8_out.view(-1), dn_w_sc.reshape(-1), dn_in_sc.reshape(-1),
                    dn_got0, DOWN_N, DOWN_K,
                )
                torch.cuda.synchronize(dev)
                ok0 = rc == 0 and torch.equal(dn_got0.view(torch.int16), refs[0].view(torch.int16))
            else:
                if is_ilv:
                    rc = _sk_tma4_ilv_swiglu(
                        probes[0].data_ptr(), w.data_ptr(), swiglu_out.data_ptr(), partials.data_ptr(),
                        s6_new.data_ptr(), pal.data_ptr(), gu.weight_global_scale.data_ptr(),
                        gu.workspace.data_ptr(), GATE_UP_N, GATE_UP_K, 48, dev.index, stream,
                    )
                else:
                    rc = _gate_up_tma4_swiglu(
                        probes[0].data_ptr(), w.data_ptr(), swiglu_out.data_ptr(), partials.data_ptr(),
                        s6_new.data_ptr(), pal.data_ptr(), gu.weight_global_scale.data_ptr(),
                        gu.workspace.data_ptr(), 0, GATE_UP_N, GATE_UP_K, 48, dev.index, stream,
                    )
                torch.cuda.synchronize(dev)
                ok0 = rc == 0 and torch.equal(swiglu_out[:, :DOWN_K].view(torch.int16), refs[0].view(torch.int16))
        if not ok0:
            w.copy_(w_backup)
            if is_ilv:
                _DOWN_ILV_SCALES.discard(s6_new.data_ptr())
            if reused_s6:
                s6_buf.view(-1)[: s6_backup.numel()].copy_(s6_backup.view(-1))
                if is_ilv:
                    _DOWN_ILV_SCALES.add(s6.data_ptr())
                gu.register_buffer("_marlin_decode_scales6", s6, persistent=False)
            else:
                if is_ilv:
                    _DOWN_ILV_SCALES.add(s6_backup.data_ptr())
                gu.register_buffer("_marlin_decode_scales6", s6_backup, persistent=False)
            continue
        del w_backup, s6_backup
        if reused_s6:
            count_reused += 1
        if dev not in _GATE_UP_TILE_ORDER:
            tiles = [*range(0, 528, 2), *range(528, 576), *range(1, 528, 2), *range(576, 624)]
            _GATE_UP_TILE_ORDER[dev] = torch.tensor(tiles, dtype=torch.int64, device=dev)
        if is_fp8:
            _SWIGLU_GATE_UP[gs_key] = {
                "swiglu_out": swiglu_out,
                "swiglu_fp8_out": swiglu_fp8_out,
                "fp8_scale": dn_in_sc,
                "fp8_scale_ptr": dn_in_sc.data_ptr(),
                "is_fp8": True,
                "is_ilv": is_ilv,
                "state": "admitted",
            }
            count_fp8 += 1
        else:
            swiglu_out_m = None
            if is_ilv and _sk_tma4_ilv_swiglu_m is not None and _sk_tma4_ilv_m is not None and os.environ.get("MUSE_GATE_UP_SWIGLU_M", "1") != "0":
                try:
                    import muse_norm_glue as G
                    gen_m = torch.Generator(device=dev).manual_seed(20261127 + count_bf16)
                    sw_buf_m = torch.empty((16, GATE_UP_N), dtype=torch.bfloat16, device=dev)
                    check_ms = (7, 8, 12, 16) if count_bf16 == 0 else (7, 12)
                    m_ok = True
                    for cm in check_ms:
                        xm = (torch.randn((cm, GATE_UP_K), generator=gen_m, device=dev) * 0.6).to(torch.bfloat16).contiguous()
                        out_raw = torch.empty((cm, GATE_UP_N), dtype=torch.bfloat16, device=dev)
                        ref_sw_m = torch.empty((cm, DOWN_K), dtype=torch.bfloat16, device=dev)
                        with torch.cuda.device(dev):
                            st_m = torch.cuda.current_stream().cuda_stream
                            rc_r = _sk_tma4_ilv_m(
                                xm.data_ptr(), w.data_ptr(), out_raw.data_ptr(), partials.data_ptr(),
                                s6_new.data_ptr(), pal.data_ptr(), gu.weight_global_scale.data_ptr(),
                                gu.workspace.data_ptr(), cm, GATE_UP_N, GATE_UP_K, 48, dev.index, st_m,
                            )
                            unpaired = out_raw.view(cm, 624, 64).index_select(1, _GATE_UP_TILE_ORDER[dev]).reshape(cm, GATE_UP_N).contiguous()
                            G.mul_silu_slice_bf16(unpaired, ref_sw_m, cm, st_m)
                            rc_f = _sk_tma4_ilv_swiglu_m(
                                xm.data_ptr(), w.data_ptr(), sw_buf_m.data_ptr(), partials.data_ptr(),
                                s6_new.data_ptr(), pal.data_ptr(), gu.weight_global_scale.data_ptr(),
                                gu.workspace.data_ptr(), cm, GATE_UP_N, GATE_UP_K, 48, dev.index, st_m,
                            )
                        torch.cuda.synchronize(dev)
                        got_sw_m = sw_buf_m.view(-1)[: cm * DOWN_K].view(cm, DOWN_K)
                        if not (rc_r == 0 and rc_f == 0 and int(torch.count_nonzero(gu.workspace).item()) == 0 and torch.equal(got_sw_m.view(torch.int16), ref_sw_m.view(torch.int16))):
                            m_ok = False
                            del xm, out_raw, unpaired, ref_sw_m
                            break
                        del xm, out_raw, unpaired, ref_sw_m
                    if m_ok:
                        swiglu_out_m = sw_buf_m
                        count_m += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning_once("Decode6 multi-M gate_up SwiGLU check failed: %s", exc)
                    swiglu_out_m = None
            _SWIGLU_GATE_UP[gs_key] = {
                "swiglu_out": swiglu_out,
                "swiglu_out_m": swiglu_out_m,
                "down_gs_ptr": dn.weight_global_scale.data_ptr(),
                "is_fp8": False,
                "is_ilv": is_ilv,
                "state": "admitted",
            }
            count_bf16 += 1
    total = count_bf16 + count_fp8
    if total:
        logger.info(
            "Decode6 paired gate_up SwiGLU admitted bitwise on %d MLP layers (%d BF16->NVFP4, %d FP8->rANS, %d multi-M, %d/%d reused fast-pool scales6)",
            total, count_bf16, count_fp8, count_m, count_reused, total,
        )
    return total


def launch_oproj_gated6(
    gate: torch.Tensor,
    attn: torch.Tensor,
    tiled_weight: torch.Tensor,
    scales6: torch.Tensor,
    palette: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
    out_shape: tuple[int, ...],
) -> torch.Tensor:
    g_kernel = _sk_tma4_ilv_gated if is_tma4_ilv(scales6) else _sk_tma4_gated
    if g_kernel is None:
        raise RuntimeError("SM121 Marlin o_proj gated kernel unavailable")
    partials = gate.new_empty((48, 16, 128), dtype=torch.float32)
    output = gate.new_empty(out_shape)
    with torch.cuda.device(gate.device):
        status = g_kernel(
            gate.data_ptr(), attn.data_ptr(), tiled_weight.data_ptr(), output.data_ptr(),
            partials.data_ptr(), scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(),
            workspace.data_ptr(), OPROJ_N, OPROJ_K, 48,
            gate.device.index, torch.cuda.current_stream().cuda_stream,
        )
    if status:
        raise RuntimeError(f"SM121 Marlin o_proj gated launch failed with CUDA error {status}")
    return output


def launch_oproj_gated6_m(
    gate: torch.Tensor,
    attn: torch.Tensor,
    tiled_weight: torch.Tensor,
    scales6: torch.Tensor,
    palette: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
    out_shape: tuple[int, ...],
) -> torch.Tensor:
    if _sk_tma4_ilv_gated_m is None or not is_tma4_ilv(scales6):
        raise RuntimeError("SM121 Marlin multi-M o_proj gated kernel unavailable")
    m = int(gate.numel() // OPROJ_K)
    if m < 1 or m > 16 or gate.numel() != m * OPROJ_K or attn.numel() != m * OPROJ_K:
        raise RuntimeError(f"SM121 Marlin multi-M o_proj gated invalid shape: m={m}")
    if not gate.is_contiguous() or gate.data_ptr() % 16:
        gate = gate.clone(memory_format=torch.contiguous_format)
    if not attn.is_contiguous() or attn.data_ptr() % 16:
        attn = attn.clone(memory_format=torch.contiguous_format)
    partials = gate.new_empty((48, 16, 128), dtype=torch.float32)
    output = gate.new_empty(out_shape)
    with torch.cuda.device(gate.device):
        status = _sk_tma4_ilv_gated_m(
            gate.data_ptr(), attn.data_ptr(), tiled_weight.data_ptr(), output.data_ptr(),
            partials.data_ptr(), scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(),
            workspace.data_ptr(), m, OPROJ_N, OPROJ_K, 48,
            gate.device.index, torch.cuda.current_stream().cuda_stream,
        )
    if status:
        raise RuntimeError(f"SM121 Marlin multi-M o_proj gated launch failed with CUDA error {status}")
    return output


def register_all_oproj_sigmoid(model: torch.nn.Module) -> int:
    global _ALL_OPROJ_GATED_READY, _ALL_OPROJ_GATED_M_READY
    _ALL_OPROJ_GATED_READY = False
    _ALL_OPROJ_GATED_M_READY = False
    _GATED_OPROJ.clear()
    _GATED_OPROJ_M.clear()
    if os.environ.get("MUSE_OPROJ_SIGMOID_FUSE", "1") == "0" or (_sk_tma4_ilv_gated is None and _sk_tma4_gated is None):
        return 0
    want_m = os.environ.get("MUSE_OPROJ_SIGMOID_FUSE_M", "1") != "0" and _sk_tma4_ilv_gated_m is not None
    try:
        import muse_norm_glue as _ng
        if _ng.lib() is None or not hasattr(_ng, "mul_sigmoid_bf16"):
            return 0
    except Exception:
        return 0
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return 0
    non_fp8_total = 0
    admitted = 0
    admitted_m = 0
    for lyr in layers:
        sa = getattr(lyr, "self_attn", None)
        op = getattr(sa, "o_proj", None)
        if op is None:
            continue
        w_raw = getattr(op, "weight", None)
        if w_raw is not None and w_raw.dtype == torch.float8_e4m3fn:
            continue
        non_fp8_total += 1
        s6 = getattr(op, "_marlin_decode_scales6", None)
        w = getattr(op, "_marlin_decode_weight", None)
        pal = getattr(op, "_marlin_decode_palette", None)
        gs = getattr(op, "weight_global_scale", None)
        ws = getattr(op, "workspace", None)
        if (
            s6 is None or w is None or pal is None or gs is None or ws is None
            or getattr(op, "marlin_decode6_fixed_grid", False)
            or not is_tma4(s6)
            or (op.output_size_per_partition, op.input_size_per_partition) != (OPROJ_N, OPROJ_K)
        ):
            continue
        g_kernel = _sk_tma4_ilv_gated if is_tma4_ilv(s6) else _sk_tma4_gated
        if g_kernel is None:
            continue
        dev = s6.device
        gen = torch.Generator(device=dev).manual_seed(20261127 + non_fp8_total)
        tmp = torch.empty((1, OPROJ_K), dtype=torch.bfloat16, device=dev)
        ok = True
        with torch.cuda.device(dev):
            stream = torch.cuda.current_stream().cuda_stream
            for sig in (0.05, 1.0, 5.0, 24.0):
                rg = (torch.randn((1, OPROJ_K), generator=gen, device=dev) * sig).to(torch.bfloat16).contiguous()
                ra = (torch.randn((1, OPROJ_K), generator=gen, device=dev) * sig).to(torch.bfloat16).contiguous()
                ra[:, 1::2] = -ra[:, ::2]
                _ng.mul_sigmoid_bf16(rg, ra, tmp, OPROJ_K, stream)
                ref = decode_linear6(tmp, w, s6, pal, gs, ws, OPROJ_N, OPROJ_K, False)
                got = launch_oproj_gated6(rg, ra, w, s6, pal, gs, ws, (1, OPROJ_N))
                torch.cuda.synchronize(dev)
                if not torch.equal(got.view(torch.int16), ref.view(torch.int16)) or int(torch.count_nonzero(ws).item()) != 0:
                    ok = False
                    break
        if ok:
            _GATED_OPROJ.add(gs.data_ptr())
            admitted += 1
            if want_m and is_tma4_ilv(s6):
                ok_m = True
                try:
                    with torch.cuda.device(dev):
                        stream = torch.cuda.current_stream().cuda_stream
                        m_checks = ((7, (0.05, 1.0, 5.0, 24.0)), (12, (0.05, 5.0)), (16, (1.0,))) if admitted_m == 0 else ((7, (0.05, 1.0, 5.0, 24.0)), (12, (1.0,)))
                        for cm, sigs in m_checks:
                            tmpm = torch.empty((cm, OPROJ_K), dtype=torch.bfloat16, device=dev)
                            for sig in sigs:
                                rgm = (torch.randn((cm, OPROJ_K), generator=gen, device=dev) * sig).to(torch.bfloat16).contiguous()
                                ram = (torch.randn((cm, OPROJ_K), generator=gen, device=dev) * sig).to(torch.bfloat16).contiguous()
                                ram[:, 1::2] = -ram[:, ::2]
                                _ng.mul_sigmoid_bf16(rgm, ram, tmpm, cm * OPROJ_K, stream)
                                refm = decode_linear6(tmpm, w, s6, pal, gs, ws, OPROJ_N, OPROJ_K, False)
                                gotm = launch_oproj_gated6_m(rgm, ram, w, s6, pal, gs, ws, (cm, OPROJ_N))
                                torch.cuda.synchronize(dev)
                                if not torch.equal(gotm.view(torch.int16), refm.view(torch.int16)) or int(torch.count_nonzero(ws).item()) != 0:
                                    ok_m = False
                                    break
                            if not ok_m:
                                break
                except Exception:
                    ok_m = False
                if ok_m:
                    _GATED_OPROJ_M.add(gs.data_ptr())
                    admitted_m += 1
    if non_fp8_total > 0 and admitted == non_fp8_total:
        _ALL_OPROJ_GATED_READY = True
        logger.info("Decode6 o_proj fused mul_sigmoid prologue prechecked bitwise on %d/%d NVFP4 layers", admitted, non_fp8_total)
    if non_fp8_total > 0 and admitted_m == non_fp8_total:
        _ALL_OPROJ_GATED_M_READY = True
        logger.info("Decode6 o_proj fused mul_sigmoid M=2..8 prologue prechecked bitwise on %d/%d NVFP4 layers", admitted_m, non_fp8_total)
    return admitted


def _decode6_fake(x, tiled_weight, scales6, palette, global_scale, workspace, n, k, fixed_grid=False):
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op("marlin_tiled_decode6", decode_linear6, fake_impl=_decode6_fake)


def self_check6(layer: torch.nn.Module, reference, fixed_grid: bool = False) -> bool:
    """decode_linear6 == reference (the uncompressed tiled GEMV) bitwise on the layer's own weight, for activations with
    large and cancelling entries at several magnitudes."""
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    dev = layer.weight.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(n * 7 + k)
    for mag in (0.01, 1.0, 64.0):
        x = torch.randn(1, k, generator=gen, device=dev) * mag
        x[:, ::3] *= 64
        x[:, 1::2] = -x[:, ::2]
        x = x.to(torch.bfloat16)
        got = decode_linear6(x, layer._marlin_decode_weight, layer._marlin_decode_scales6, layer._marlin_decode_palette,
                             layer.weight_global_scale, layer.workspace, n, k, fixed_grid)
        ref = reference(x, layer._marlin_decode_weight, layer._marlin_decode_scales, layer.weight_global_scale,
                        layer.workspace, n, k)
        if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
            return False
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


def admit_tma4(layer: torch.nn.Module, fixed_grid: bool) -> bool:
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
        and os.environ.get("MUSE_GATE_UP_ILV", "1") != "0"
    )
    gb = _nv6_gate_blocks() if (not fixed_grid and (n, k) == (4096, 6656)) else 48
    is_gate_v48 = (not fixed_grid and (n, k) == (4096, 6656) and gb != 48)
    want_lm_ilv = (
        not fixed_grid
        and (n, k) == (LM_HEAD_N, LM_HEAD_K)
        and _sk_tma4_ilv is not None
        and os.environ.get("MUSE_LM_HEAD_ILV", "1") != "0"
    )
    want_lm_pad128 = want_lm_ilv and os.environ.get("MUSE_LM_HEAD_PAD128", "1") != "0"
    if want_lm_pad128:
        lm_blocks = 96 if os.environ.get("MUSE_LM_HEAD_PAD128_BLOCKS", "48") == "96" else 48
        w128, s128 = _pad_lm_head_128(w, layer._marlin_decode_scales, pal)
        coded128 = encode_tma4_scales(
            s128, pal, LM_HEAD_PAD128_N, k, False, lm_pad128=True, lm_blocks=lm_blocks
        )
        del s128
        if coded128.numel() <= c.numel():
            w128_ilv = _build_ilv_weight(
                w128, LM_HEAD_PAD128_N, k, False, 48, lm_pad128=True, lm_blocks=lm_blocks
            )
            del w128
            _DOWN_ILV_SCALES.add(coded128.data_ptr())
            _LM_PAD128_SCALES.add(coded128.data_ptr())
            if lm_blocks == 96:
                _LM_SPLIT2_SCALES.add(coded128.data_ptr())
            try:
                got128 = [decode_linear6(x, w128_ilv, coded128, pal, gs, ws, n, k, False) for x in probes]
            finally:
                _DOWN_ILV_SCALES.discard(coded128.data_ptr())
                _LM_PAD128_SCALES.discard(coded128.data_ptr())
                _LM_SPLIT2_SCALES.discard(coded128.data_ptr())
            if _same_bits(expected, got128) and int(torch.count_nonzero(ws)) == 0:
                nbytes = coded128.numel()
                c[:nbytes].copy_(coded128.view(-1))
                del coded128
                view = c[:nbytes].view(-1, 32)
                _DOWN_ILV_SCALES.add(view.data_ptr())
                _LM_PAD128_SCALES.add(view.data_ptr())
                if lm_blocks == 96:
                    _LM_SPLIT2_SCALES.add(view.data_ptr())
                got128_2 = [decode_linear6(x, w128_ilv, view, pal, gs, ws, n, k, False) for x in probes]
                if _same_bits(expected, got128_2) and int(torch.count_nonzero(ws)) == 0:
                    layer.register_buffer("_marlin_decode_weight", w128_ilv, persistent=False)
                    layer.register_buffer("_marlin_decode_scales6", view, persistent=False)
                    layer.marlin_decode6_down_srans = True
                    STATS["down_srans_scales"] += layer._marlin_decode_scales.numel()
                    STATS["down_srans_bytes"] += nbytes
                    logger.info(
                        "Admitted lm_head Pad128 TileN=128 TMA4-ILV bitwise (202048->202112, blocks=%d, scales=%.2f MiB)",
                        lm_blocks, nbytes / (1 << 20),
                    )
                    return True
                _DOWN_ILV_SCALES.discard(view.data_ptr())
                _LM_PAD128_SCALES.discard(view.data_ptr())
                _LM_SPLIT2_SCALES.discard(view.data_ptr())
                c.copy_(_six_bit_codes(layer))
            del w128_ilv
            logger.warning("lm_head Pad128 TileN=128 failed bitwise check; falling back to TileN=64")
        else:
            del w128, coded128
            logger.warning("lm_head Pad128 coded scales exceeded buffer; falling back to TileN=64")
    want_lm_split2 = want_lm_ilv and os.environ.get("MUSE_LM_HEAD_SPLIT2", "1") != "0"
    coded = encode_tma4_scales(
        layer._marlin_decode_scales, pal, n, k,
        False if want_gu_ilv else fixed_grid,
        gate_v48=is_gate_v48, gate_blocks=gb, split2=want_lm_split2,
    )
    if coded.numel() > c.numel():
        logger.warning("TMA4 (%dx%d) rejected: coded %d > c %d", n, k, coded.numel(), c.numel())
        return False
    want_draft_lm_ilv = (
        not fixed_grid
        and k == LM_HEAD_K
        and n >= 16384
        and n % 128 == 0
        and n != GATE_UP_N
        and os.environ.get("MUSE_LM_HEAD_ILV", "1") != "0"
    )
    want_ilv = _sk_tma4_ilv is not None and (
        want_gu_ilv
        or want_lm_ilv
        or want_draft_lm_ilv
        or (
            not fixed_grid
            and (
                ((n, k) == (DOWN_N, DOWN_K) and os.environ.get("MUSE_DOWN_ILV", "1") != "0")
                or ((n, k) in ATTN_ILV_SHAPES and gb in (24, 48) and os.environ.get("MUSE_ATTN_ILV", "1") != "0")
            )
        )
    )
    if want_ilv:
        w_ilv = _build_ilv_weight(w, n, k, is_gate_v48, gb, split2=want_lm_split2)
        _DOWN_ILV_SCALES.add(coded.data_ptr())
        if want_lm_split2:
            _LM_SPLIT2_SCALES.add(coded.data_ptr())
        try:
            got_ilv = [decode_linear6(x, w_ilv, coded, pal, gs, ws, n, k, False) for x in probes]
        finally:
            _DOWN_ILV_SCALES.discard(coded.data_ptr())
            _LM_SPLIT2_SCALES.discard(coded.data_ptr())
        if _same_bits(expected, got_ilv) and int(torch.count_nonzero(ws)) == 0:
            nbytes = coded.numel()
            w_backup = w.detach().clone() if w.nbytes <= (256 << 20) else None
            w.copy_(w_ilv)
            del w_ilv
            c[:nbytes].copy_(coded.view(-1))
            del coded
            view = c[:nbytes].view(-1, 32)
            _DOWN_ILV_SCALES.add(view.data_ptr())
            if want_lm_split2:
                _LM_SPLIT2_SCALES.add(view.data_ptr())
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
            _LM_SPLIT2_SCALES.discard(view.data_ptr())
            if w_backup is not None:
                w.copy_(w_backup)
                del w_backup
            c.copy_(_six_bit_codes(layer))
            return False
        del w_ilv
        if want_gu_ilv or want_lm_ilv:
            coded = encode_tma4_scales(layer._marlin_decode_scales, pal, n, k, fixed_grid)
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
    if fixed_grid:
        layer.marlin_decode6_gate_up_srans = True
        STATS["srans_scales"] += layer._marlin_decode_scales.numel()
        STATS["srans_bytes"] += nbytes
    else:
        layer.marlin_decode6_down_srans = True
        STATS["down_srans_scales"] += layer._marlin_decode_scales.numel()
        STATS["down_srans_bytes"] += nbytes
    return True


def prepare_decode_layout6(layer: torch.nn.Module, reference) -> bool:
    """After the board's prepare_decode_layout: add the 6-bit scale codes and palette, check, register. True if admitted."""
    layer.marlin_decode6_fixed_grid = False
    tiled = getattr(layer, "_marlin_decode_scales", None)
    if tiled is None or getattr(layer, "_marlin_decode_weight", None) is None:
        return False
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    packed = pack_scales6(tiled) if k % 128 == 0 else None
    if packed is None:
        STATS["kept"] += 1
        return False
    codes, palette = packed
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
    STATS["admitted"] += 1
    return True
