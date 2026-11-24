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
_gate_up_tma4_swiglu = getattr(_lib, "muse_marlin6_gate_up_tma4_swiglu", None)
if _gate_up_tma4_swiglu is not None:
    _gate_up_tma4_swiglu.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    _gate_up_tma4_swiglu.restype = ctypes.c_int
_sk_tma4_ilv_swiglu = getattr(_lib, "muse_marlin6_sk_tma4_ilv_swiglu", None)
if _sk_tma4_ilv_swiglu is not None:
    _sk_tma4_ilv_swiglu.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    _sk_tma4_ilv_swiglu.restype = ctypes.c_int
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
_sk_tma4_ilv = getattr(_lib, "muse_marlin6_sk_tma4_ilv", None)
if _sk_tma4_ilv is not None:
    _sk_tma4_ilv.argtypes = _gemv6.argtypes
    _sk_tma4_ilv.restype = ctypes.c_int
_sk_tma4_gated = getattr(_lib, "muse_marlin6_sk_tma4_gated", None)
if _sk_tma4_gated is not None:
    _sk_tma4_gated.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    _sk_tma4_gated.restype = ctypes.c_int
_sk_tma4_ilv_gated = getattr(_lib, "muse_marlin6_sk_tma4_ilv_gated", None)
if _sk_tma4_ilv_gated is not None:
    _sk_tma4_ilv_gated.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    _sk_tma4_ilv_gated.restype = ctypes.c_int
_sk_tma4_swiglu_a = getattr(_lib, "muse_marlin6_sk_tma4_swiglu_a", None)
if _sk_tma4_swiglu_a is not None:
    _sk_tma4_swiglu_a.argtypes = _gemv6.argtypes
    _sk_tma4_swiglu_a.restype = ctypes.c_int
_sk_tma4_ilv_swiglu_a = getattr(_lib, "muse_marlin6_sk_tma4_ilv_swiglu_a", None)
if _sk_tma4_ilv_swiglu_a is not None:
    _sk_tma4_ilv_swiglu_a.argtypes = _gemv6.argtypes
    _sk_tma4_ilv_swiglu_a.restype = ctypes.c_int
_set_gate_blocks = getattr(_lib, "muse_marlin6_set_gate_blocks", None)
if _set_gate_blocks is not None:
    _set_gate_blocks.argtypes = [ctypes.c_int]
    _set_gate_blocks.restype = ctypes.c_int
_get_gate_blocks = getattr(_lib, "muse_marlin6_get_gate_blocks", None)
if _get_gate_blocks is not None:
    _get_gate_blocks.argtypes = []
    _get_gate_blocks.restype = ctypes.c_int
_set_tma4_stages = getattr(_lib, "muse_marlin6_set_tma4_stages", None)
if _set_tma4_stages is not None:
    _set_tma4_stages.argtypes = [ctypes.c_int]
    _set_tma4_stages.restype = ctypes.c_int
_get_tma4_stages = getattr(_lib, "muse_marlin6_get_tma4_stages", None)
if _get_tma4_stages is not None:
    _get_tma4_stages.argtypes = []
    _get_tma4_stages.restype = ctypes.c_int
_set_pdl = getattr(_lib, "muse_marlin6_set_pdl", None)
if _set_pdl is not None:
    _set_pdl.argtypes = [ctypes.c_int]
    _set_pdl.restype = ctypes.c_int
_get_pdl = getattr(_lib, "muse_marlin6_get_pdl", None)
if _get_pdl is not None:
    _get_pdl.argtypes = []
    _get_pdl.restype = ctypes.c_int
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
ATTN_ILV_SHAPES = {(4608, 6656), (6656, 4096), (4096, 6656)}
_TMA4_SCHED_CACHE: dict[tuple, tuple] = {}
_DOWN_ILV_SCALES: set[int] = set()
_LM_SPLIT2_SCALES: set[int] = set()
_GATE_UP_SPLIT2_SCALES: set[int] = set()
_DOWN_SPLIT2_SCALES: set[int] = set()
_SWIGLU_GATE_UP: dict[int, dict] = {}
_GATED_OPROJ: set[int] = set()
_ALL_OPROJ_GATED_READY: bool = False
_SWIGLU_DOWN_FUSE: set[int] = set()
_ALL_DOWN_SWIGLU_READY: bool = False
_LMPOST_BUFS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_SPLIT2_LOCKS_BUFS: dict[int, torch.Tensor] = {}
_PAIR_LOCKS_BUFS: dict[int, torch.Tensor] = {}
_SHARED_PARTIALS_BUFS: dict[int, torch.Tensor] = {}
_SHARED_PARTIALS_RAW: dict[int, torch.Tensor] = {}
_RUNTIME_TMA4_STAGES: int | None = None
_RUNTIME_PDL: bool | None = None
_consume_down_swiglu = None
_consume_down_swiglu_fuse = None
_notify_gate_up_swiglu = None
_consume_oproj_sigmoid = None


def relocate_scale_ptr(old_ptr: int, new_ptr: int) -> None:
    for s in (_DOWN_ILV_SCALES, _LM_SPLIT2_SCALES, _GATE_UP_SPLIT2_SCALES, _DOWN_SPLIT2_SCALES):
        if old_ptr in s:
            s.discard(old_ptr)
            s.add(new_ptr)


def set_runtime_tma4_stages(stages: int | None) -> None:
    global _RUNTIME_TMA4_STAGES
    if stages is not None and int(stages) not in (4, 8):
        raise ValueError(f"Unsupported TMA4 stages: {stages}")
    v = 0 if stages is None else int(stages)
    _RUNTIME_TMA4_STAGES = None if v == 0 else v
    if _set_tma4_stages is not None:
        rc = _set_tma4_stages(v)
        if rc != 0:
            raise RuntimeError(f"muse_marlin6_set_tma4_stages({v}) failed with error {rc}")


def get_runtime_tma4_stages() -> int | None:
    if _get_tma4_stages is not None:
        v = int(_get_tma4_stages())
        return v if v in (4, 8) else None
    return _RUNTIME_TMA4_STAGES


def set_runtime_pdl(enabled: bool | None) -> None:
    global _RUNTIME_PDL
    _RUNTIME_PDL = None if enabled is None else bool(enabled)
    v = -1 if enabled is None else (1 if enabled else 0)
    if _set_pdl is not None:
        rc = _set_pdl(v)
        if rc != 0:
            raise RuntimeError(f"muse_marlin6_set_pdl({v}) failed with error {rc}")


def get_runtime_pdl() -> bool:
    if _get_pdl is not None:
        return bool(_get_pdl())
    if _RUNTIME_PDL is not None:
        return _RUNTIME_PDL
    return os.environ.get("MUSE_DECODE6_PDL", "1") != "0"


def _get_shared_partials(dev: torch.device) -> torch.Tensor:
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    buf = _SHARED_PARTIALS_BUFS.get(idx)
    if buf is None:
        # 64 KiB (128 slices x 512 B), explicitly 256-byte (>=128B) aligned for discard.global.L2.
        n_f32 = 16384
        raw = torch.empty(n_f32 + 64, dtype=torch.float32, device=dev)
        mis = raw.data_ptr() & 255
        off = ((256 - mis) & 255) // 4
        buf = raw[off : off + n_f32]
        _SHARED_PARTIALS_RAW[idx] = raw
        _SHARED_PARTIALS_BUFS[idx] = buf
    return buf


def get_shared_partials(dev: torch.device) -> torch.Tensor:
    return _get_shared_partials(dev)


def _get_split2_locks(dev: torch.device) -> torch.Tensor:
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    buf = _SPLIT2_LOCKS_BUFS.get(idx)
    if buf is None:
        buf = torch.zeros(256, dtype=torch.int32, device=dev)
        _SPLIT2_LOCKS_BUFS[idx] = buf
    return buf


def _get_pair_locks(dev: torch.device) -> torch.Tensor:
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    buf = _PAIR_LOCKS_BUFS.get(idx)
    if buf is None:
        buf = torch.zeros(160, dtype=torch.int32, device=dev)
        _PAIR_LOCKS_BUFS[idx] = buf
    return buf


def set_runtime_gate_blocks(blocks: int) -> None:
    b = int(blocks)
    if b not in (24, 32, 48):
        raise ValueError(f"Unsupported NVFP4 gate blocks: {blocks}")
    if _set_gate_blocks is not None:
        rc = _set_gate_blocks(b)
        if rc != 0:
            raise RuntimeError(f"muse_marlin6_set_gate_blocks({b}) failed with error {rc}")


def _nv6_gate_blocks(dev: torch.device | None = None) -> int:
    env = os.environ.get("MUSE_NV6_GATE_BLOCKS", "").strip()
    if env:
        try:
            v = int(env)
            if v in (24, 32, 48):
                return v
        except ValueError:
            pass
    if dev is not None:
        try:
            import muse_gqa16_attn as _gqa
            adm = _gqa.is_admitted(dev) if hasattr(_gqa, "is_admitted") else None
            if adm is None and hasattr(_gqa, "self_check"):
                adm = _gqa.self_check(dev)
            if adm is False:
                if _set_gate_blocks is not None:
                    _set_gate_blocks(32)
                return 32
        except Exception:
            pass
    if _get_gate_blocks is not None:
        return int(_get_gate_blocks())
    return 48

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
    split2: bool = False,
    down_split2: bool = False,
    grid: int = 48,
):
    key = (n, k, fixed_grid, gate_v48, gate_blocks, split2, down_split2, grid, device)
    cached = _TMA4_SCHED_CACHE.get(key)
    if cached is not None:
        return cached
    use_64 = (n % 128 != 0) or split2 or down_split2 or (grid == 96)
    tile_n = 64 if use_64 else 128
    threads = 128 if tile_n == 64 else 256
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
    elif down_split2:
        vgrid = 48
        num_blks = 96
        v_n_tiles = n_tiles // 2
        part2_mn_tiles = v_n_tiles
        part1_mn_iters = 0
        if v_n_tiles > vgrid:
            part2_mn_tiles = v_n_tiles % vgrid
            if part2_mn_tiles * 3 <= vgrid:
                part2_mn_tiles += vgrid
            part1_mn_iters = (v_n_tiles - part2_mn_tiles) // vgrid
        iters = (k_tiles * part2_mn_tiles + vgrid - 1) // vgrid
        base_col = v_n_tiles - part2_mn_tiles
        total_p2 = part2_mn_tiles * k_tiles
        for b in range(num_blks):
            v = b % vgrid
            h = b // vgrid
            s_start = len(stages)
            for p in range(part1_mn_iters):
                col64 = 2 * (v + p * vgrid) + h
                for kt in range(k_tiles):
                    stages.append((col64, kt))
            p2_start = iters * v
            p2_end = min(iters * (v + 1), total_p2)
            for pos in range(p2_start, p2_end):
                col64 = 2 * (base_col + pos // k_tiles) + h
                stages.append((col64, pos % k_tiles))
            blk_off.append(len(stages))
            it = len(stages) - s_start
            blk_iters.append(it)
            blk_sb_off.append(blk_sb_off[-1] + (it + 3) // 4)
    else:
        num_blks = grid
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
    down_split2: bool = False,
    grid: int = 48,
) -> torch.Tensor:
    use_64 = (n % 128 != 0) or split2 or down_split2 or (grid == 96)
    tile_n = 64 if use_64 else 128
    k_tiles = k // 128
    *_, num_blks, _, _, _, stages, blk_off, blk_iters = _get_tma4_sched(
        n, k, False, w.device, gate_v48=gate_v48, gate_blocks=gate_blocks,
        split2=split2, down_split2=down_split2, grid=grid,
    )
    max_it = max(blk_iters)
    perm = []
    for s in range(max_it):
        for b in range(num_blks):
            if s < blk_iters[b]:
                col, kt = stages[blk_off[b] + s]
                perm.append(col * k_tiles + kt)
    assert len(perm) == (n // tile_n) * k_tiles
    w_src = w
    if n % 128 == 0 and tile_n == 64:
        w_src = w.view(n // 128, k_tiles, 8, 2, 128).permute(0, 3, 1, 2, 4).contiguous()
    idx_t = torch.tensor(perm, dtype=torch.long, device=w.device)
    return w_src.reshape(len(perm), -1).index_select(0, idx_t).contiguous().view_as(w)


def _unbuild_ilv_weight(
    w_ilv: torch.Tensor,
    n: int,
    k: int,
    gate_v48: bool = False,
    gate_blocks: int = 48,
    split2: bool = False,
    down_split2: bool = False,
    grid: int = 48,
) -> torch.Tensor:
    use_64 = (n % 128 != 0) or split2 or down_split2 or (grid == 96)
    tile_n = 64 if use_64 else 128
    k_tiles = k // 128
    *_, num_blks, _, _, _, stages, blk_off, blk_iters = _get_tma4_sched(
        n, k, False, w_ilv.device, gate_v48=gate_v48, gate_blocks=gate_blocks,
        split2=split2, down_split2=down_split2, grid=grid,
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
    unilv = w_ilv.reshape(len(perm), -1).index_select(0, idx_t).contiguous()
    if n % 128 == 0 and tile_n == 64:
        unilv = unilv.view(n // 128, 2, k_tiles, 8, 128).permute(0, 2, 3, 1, 4).contiguous()
    return unilv.view_as(w_ilv)


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
    down_split2: bool = False,
    grid: int = 48,
) -> torch.Tensor:
    """Encode tiled one-byte scales into wave/stage-interleaved 4-stage TMA super-block rANS streams (shape (rows, 32) uint8)."""
    if _tma4_p1 is None or _tma4_p2 is None:
        raise RuntimeError("GPU TMA4 rANS encoder unavailable")
    dev = tiled_scales.device
    use_64 = (n % 128 != 0) or split2 or down_split2 or (grid == 96)
    tile_n = 64 if use_64 else 128
    n_tiles = n // tile_n
    k_tiles = k // 128
    stage_bytes = tile_n * 8
    if n % 128 == 0 and tile_n == 64:
        raw = (
            tiled_scales.contiguous()
            .view(torch.uint8)
            .view(n // 128, k_tiles, 8, 2, 64)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
            .reshape(n_tiles, k_tiles, stage_bytes)
        )
    else:
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
        split2=split2, down_split2=down_split2, grid=grid,
    )
    warps = threads // 32
    rans_cap = 3200 if threads == 256 else 2048
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

    pal_cpu = palette.detach().to(device="cpu", dtype=torch.long) & 0xFF
    lut_tma4 = ((lut & ~63) << 2) | pal_cpu[lut & 63]
    coded = torch.zeros(total_bytes, dtype=torch.uint8, device=dev)
    coded[:4096].copy_(lut_tma4.to(dtype=torch.int32, device=dev).view(torch.uint8))
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
    if _consume_down_swiglu is not None and not fixed_grid and (n, k) == (DOWN_N, DOWN_K) and x.numel() == k:
        x_sw = _consume_down_swiglu(x, global_scale.data_ptr())
        if x_sw is not None:
            x = x_sw
    if (
        _consume_down_swiglu_fuse is not None
        and not fixed_grid
        and (n, k) == (DOWN_N, DOWN_K)
        and x.numel() == k
        and global_scale.data_ptr() in _SWIGLU_DOWN_FUSE
        and is_tma4(scales6)
    ):
        dn_res = _consume_down_swiglu_fuse(
            x, global_scale.data_ptr(), tiled_weight, scales6, palette, global_scale, workspace
        )
        if dn_res is not None:
            return dn_res
    if (
        _consume_oproj_sigmoid is not None
        and not fixed_grid
        and (n, k) == (OPROJ_N, OPROJ_K)
        and x.numel() == k
        and global_scale.data_ptr() in _GATED_OPROJ
        and is_tma4(scales6)
    ):
        op_res = _consume_oproj_sigmoid(
            x, global_scale.data_ptr(), tiled_weight, scales6, palette, global_scale, workspace
        )
        if op_res is not None:
            return op_res
    if not x.is_contiguous() or x.data_ptr() % 16:
        x = x.clone(memory_format=torch.contiguous_format)
    s6_ptr = scales6.data_ptr()
    is_split2_96 = (
        ((n, k) == (LM_HEAD_N, LM_HEAD_K) and s6_ptr in _LM_SPLIT2_SCALES)
        or ((n, k) == (GATE_UP_N, GATE_UP_K) and s6_ptr in _GATE_UP_SPLIT2_SCALES)
        or ((n, k) == (DOWN_N, DOWN_K) and s6_ptr in _DOWN_SPLIT2_SCALES)
    )
    blocks = 96 if is_split2_96 else 48
    partials = _get_shared_partials(x.device)
    locks_t = _get_split2_locks(x.device) if (blocks == 96 and workspace.numel() < 256) else workspace
    with torch.cuda.device(x.device):
        use_grid = fixed_grid and _gate_up is not None and x.numel() == k and (n, k) == (39936, 6656)
        sw_entry = (
            _SWIGLU_GATE_UP.get(global_scale.data_ptr())
            if (x.numel() == k and (n, k) == (39936, 6656) and _notify_gate_up_swiglu is not None and is_tma4(scales6))
            else None
        )
        if sw_entry is not None and sw_entry["state"] == "admitted":
            out_buf = sw_entry["swiglu_out"]
            sw_blocks = int(sw_entry.get("blocks", 48))
            sw_locks = _get_split2_locks(x.device) if (sw_blocks == 96 and workspace.numel() < 256) else workspace
            pair_locks = _get_pair_locks(x.device)
            if sw_entry["is_ilv"]:
                status = _sk_tma4_ilv_swiglu(
                    x.data_ptr(), tiled_weight.data_ptr(), out_buf.data_ptr(), partials.data_ptr(),
                    scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), sw_locks.data_ptr(),
                    pair_locks.data_ptr(), n, k, sw_blocks, x.device.index, torch.cuda.current_stream().cuda_stream,
                )
            else:
                status = _gate_up_tma4_swiglu(
                    x.data_ptr(), tiled_weight.data_ptr(), out_buf.data_ptr(), partials.data_ptr(),
                    scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), sw_locks.data_ptr(),
                    pair_locks.data_ptr(), n, k, 48, x.device.index, torch.cuda.current_stream().cuda_stream,
                )
            if status:
                raise RuntimeError(f"SM121 Marlin gate_up SwiGLU launch failed with CUDA error {status}")
            _notify_gate_up_swiglu(out_buf.data_ptr(), sw_entry, True, None)
            return out_buf
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
            scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), locks_t.data_ptr(), n, k, blocks,
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
            _get_shared_partials(dev),
        )
        _LMPOST_BUFS[idx] = bufs
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
    blocks = 96 if (is_ilv and s6.data_ptr() in _LM_SPLIT2_SCALES) else 48
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
            n,
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
    if float(mult) < 0.0 or (cap is not None and float(cap) <= 0.0):
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
    if k % 128 != 0 or n % 128 == 0 or n % 64 != 0 or not (0 < org_vocab_size <= n):
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
        "Fused lm_head+lmpost decode6 kernel admitted bitwise (n=%d, vocab=%d, k=%d, ilv=%s, split2=%s)",
        n, org_vocab_size, k, is_ilv, scales6.data_ptr() in _LM_SPLIT2_SCALES,
    )
    return True


def _pair_gate_up_tensors(w_raw: torch.Tensor, s_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Repack [312, 416, 256] int32 weights and [312, 416, 128] uint8 scales so each 128-col tile p (0..311)
    holds 64 cols of gate in its left half and 64 cols of up in its right half."""
    w_p = w_raw.view(2, 156, 416, 2, 128).permute(1, 3, 2, 0, 4).contiguous().view(312, 416, 256)
    s_p = s_raw.view(2, 156, 416, 2, 64).permute(1, 3, 2, 0, 4).contiguous().view(312, 416, 128)
    return w_p, s_p


def _pair_gate_up_tensors_split2(w_raw: torch.Tensor, s_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Repack [312, 416, 256] int32 weights and [312, 416, 128] uint8 scales for 96-CTA FixedShape=3 (624 64-col tiles):
    - Tiles 0..527 (528 tiles = cols 0..16895): paired 32 gate (j=0,1) + 32 up (j=2,3) in Part 1 (unsplit K=0..51).
    - Tiles 528..575 (48 tiles = cols 16896..19967): pure 64-col gate in Part 1 (unsplit K=0..51).
    - Tiles 576..623 (48 tiles = cols 16896..19967): pure 64-col up in Part 2 (K-split 26+26, identical to stock)."""
    w64 = w_raw.view(312, 416, 2, 128).permute(0, 2, 1, 3).contiguous().view(624, 416, 128)
    g_w32 = w64[:312].view(312, 416, 32, 2, 2).permute(0, 3, 1, 2, 4).reshape(624, 416, 32, 2)
    u_w32 = w64[312:].view(312, 416, 32, 2, 2).permute(0, 3, 1, 2, 4).reshape(624, 416, 32, 2)
    head_w = torch.stack((g_w32[:528], u_w32[:528]), dim=3).reshape(528, 416, 128)
    w64_out = torch.cat((head_w, w64[264:312], w64[576:624]), dim=0)
    w_p = w64_out.view(312, 2, 416, 128).permute(0, 2, 1, 3).contiguous().view(312, 416, 256)

    s64 = s_raw.view(312, 416, 2, 64).permute(0, 2, 1, 3).contiguous().view(624, 416, 64)
    g_s32 = s64[:312].view(312, 416, 8, 2, 4).permute(0, 3, 1, 2, 4).reshape(624, 416, 8, 4)
    u_s32 = s64[312:].view(312, 416, 8, 2, 4).permute(0, 3, 1, 2, 4).reshape(624, 416, 8, 4)
    head_s = torch.stack((g_s32[:528], u_s32[:528]), dim=3).reshape(528, 416, 64)
    s64_out = torch.cat((head_s, s64[264:312], s64[576:624]), dim=0)
    s_p = s64_out.view(312, 2, 416, 64).permute(0, 2, 1, 3).contiguous().view(312, 416, 128)
    return w_p, s_p


def register_all_gate_up_swiglu(model: torch.nn.Module) -> int:
    sw_env = os.environ.get("MUSE_GATE_UP_SWIGLU")
    if sw_env == "0":
        return 0
    try:
        import muse_norm_glue as _ng
        if _ng.lib() is None or not hasattr(_ng, "mul_silu_slice_bf16"):
            return 0
    except Exception:
        return 0
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return 0
    count = 0
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
        if (
            s6 is None or s_raw is None or w is None or pal is None
            or not is_tma4(s6)
            or (gu.output_size_per_partition, gu.input_size_per_partition) != (GATE_UP_N, GATE_UP_K)
            or getattr(dn, "_marlin_decode_scales6", None) is None
            or (dn.output_size_per_partition, dn.input_size_per_partition) != (DOWN_N, DOWN_K)
        ):
            continue
        is_ilv = is_tma4_ilv(s6)
        is_fg = bool(getattr(gu, "marlin_decode6_fixed_grid", False))
        is_split2 = is_ilv and (s6.data_ptr() in _GATE_UP_SPLIT2_SCALES)
        if not is_split2 and sw_env != "1":
            continue
        if is_ilv:
            if _sk_tma4_ilv_swiglu is None:
                continue
        elif is_fg:
            if _gate_up_tma4_swiglu is None:
                continue
        else:
            continue
        gs_key = gu.weight_global_scale.data_ptr()
        if gs_key in _SWIGLU_GATE_UP:
            continue
        dev = s6.device
        blocks = 96 if is_split2 else 48
        locks_t = _get_split2_locks(dev) if (blocks == 96 and gu.workspace.numel() < 256) else gu.workspace
        pair_locks = _get_pair_locks(dev)
        gen = torch.Generator(device=dev).manual_seed(20261126 + count)
        probes = []
        for mag in (0.02, 0.5, 2.0, 24.0):
            xp = (torch.randn((1, GATE_UP_K), generator=gen, device=dev) * mag)
            xp[:, 1::2] = -xp[:, ::2]
            probes.append(xp.to(torch.bfloat16).contiguous())
        # Compute exact 0-ULP reference outputs BEFORE modifying buffers
        refs = []
        with torch.cuda.device(dev):
            stream = torch.cuda.current_stream().cuda_stream
            for xp in probes:
                gu_ref = decode_linear6(
                    xp, w, s6, pal, gu.weight_global_scale, gu.workspace, GATE_UP_N, GATE_UP_K, is_fg
                )
                sw_ref = torch.empty((1, DOWN_K), dtype=torch.bfloat16, device=dev)
                _ng.mul_silu_slice_bf16(gu_ref, sw_ref, 1, stream)
                refs.append(sw_ref)
            torch.cuda.synchronize(dev)

        w_unilv = _unbuild_ilv_weight(w, GATE_UP_N, GATE_UP_K, grid=blocks) if is_ilv else w
        w_p, s_p = (
            _pair_gate_up_tensors_split2(w_unilv, s_raw)
            if is_split2
            else _pair_gate_up_tensors(w_unilv, s_raw)
        )
        del w_unilv
        try:
            s6_p = encode_tma4_scales(s_p, pal, GATE_UP_N, GATE_UP_K, not is_ilv, grid=blocks)
        except Exception:
            continue
        if is_ilv:
            w_p = _build_ilv_weight(w_p, GATE_UP_N, GATE_UP_K, False, 48, grid=blocks)

        swiglu_out = torch.empty((1, GATE_UP_N), dtype=torch.bfloat16, device=dev)
        partials = _get_shared_partials(dev)
        ok = True
        with torch.cuda.device(dev):
            stream = torch.cuda.current_stream().cuda_stream
            for xp, sw_ref in zip(probes, refs):
                swiglu_out.zero_()
                if is_ilv:
                    rc = _sk_tma4_ilv_swiglu(
                        xp.data_ptr(), w_p.data_ptr(), swiglu_out.data_ptr(), partials.data_ptr(),
                        s6_p.data_ptr(), pal.data_ptr(), gu.weight_global_scale.data_ptr(),
                        locks_t.data_ptr(), pair_locks.data_ptr(), GATE_UP_N, GATE_UP_K, blocks, dev.index, stream,
                    )
                else:
                    rc = _gate_up_tma4_swiglu(
                        xp.data_ptr(), w_p.data_ptr(), swiglu_out.data_ptr(), partials.data_ptr(),
                        s6_p.data_ptr(), pal.data_ptr(), gu.weight_global_scale.data_ptr(),
                        locks_t.data_ptr(), pair_locks.data_ptr(), GATE_UP_N, GATE_UP_K, 48, dev.index, stream,
                    )
                torch.cuda.synchronize(dev)
                if (
                    rc != 0
                    or int(torch.count_nonzero(gu.workspace).item()) != 0
                    or int(torch.count_nonzero(locks_t).item()) != 0
                    or int(torch.count_nonzero(pair_locks).item()) != 0
                    or not torch.equal(swiglu_out[:, :DOWN_K].view(torch.int16), sw_ref.view(torch.int16))
                ):
                    ok = False
                    break
        if not ok:
            continue
        # Commit paired weights IN-PLACE into placed fast DRAM buffer (keeps w.data_ptr()!)
        old_s6_ptr = s6.data_ptr()
        w_backup = w.detach().clone()
        s6_backup = s6.detach().clone()
        w.copy_(w_p)
        del w_p
        s6_storage_buf = (
            s6._base
            if (
                s6._base is not None
                and s6._base.dtype == torch.uint8
                and s6._base.is_contiguous()
                and s6._base.data_ptr() == old_s6_ptr
            )
            else (s6 if (s6.dtype == torch.uint8 and s6.is_contiguous()) else None)
        )
        if is_ilv:
            _DOWN_ILV_SCALES.discard(old_s6_ptr)
        if is_split2:
            _GATE_UP_SPLIT2_SCALES.discard(old_s6_ptr)
        if s6_storage_buf is not None and s6_storage_buf.numel() >= s6_p.numel():
            s6_flat = s6_storage_buf.view(-1)
            s6_flat[: s6_p.numel()].copy_(s6_p.view(-1))
            s6_new = s6_flat[: s6_p.numel()].view(-1, 32)
        else:
            s6_new = s6_p
        if is_ilv:
            _DOWN_ILV_SCALES.add(s6_new.data_ptr())
        if is_split2:
            _GATE_UP_SPLIT2_SCALES.add(s6_new.data_ptr())
        gu.register_buffer("_marlin_decode_scales6", s6_new, persistent=False)
        # Re-verify in-place buffer bitwise on probe 0!
        swiglu_out.zero_()
        with torch.cuda.device(dev):
            stream = torch.cuda.current_stream().cuda_stream
            if is_ilv:
                rc = _sk_tma4_ilv_swiglu(
                    probes[0].data_ptr(), w.data_ptr(), swiglu_out.data_ptr(), partials.data_ptr(),
                    s6_new.data_ptr(), pal.data_ptr(), gu.weight_global_scale.data_ptr(),
                    locks_t.data_ptr(), pair_locks.data_ptr(), GATE_UP_N, GATE_UP_K, blocks, dev.index, stream,
                )
            else:
                rc = _gate_up_tma4_swiglu(
                    probes[0].data_ptr(), w.data_ptr(), swiglu_out.data_ptr(), partials.data_ptr(),
                    s6_new.data_ptr(), pal.data_ptr(), gu.weight_global_scale.data_ptr(),
                    locks_t.data_ptr(), pair_locks.data_ptr(), GATE_UP_N, GATE_UP_K, 48, dev.index, stream,
                )
            torch.cuda.synchronize(dev)
        if (
            rc != 0
            or int(torch.count_nonzero(locks_t).item()) != 0
            or int(torch.count_nonzero(pair_locks).item()) != 0
            or not torch.equal(swiglu_out[:, :DOWN_K].view(torch.int16), refs[0].view(torch.int16))
        ):
            w.copy_(w_backup)
            if is_ilv:
                _DOWN_ILV_SCALES.discard(s6_new.data_ptr())
                _DOWN_ILV_SCALES.add(s6_backup.data_ptr())
            if is_split2:
                _GATE_UP_SPLIT2_SCALES.discard(s6_new.data_ptr())
                _GATE_UP_SPLIT2_SCALES.add(s6_backup.data_ptr())
            gu.register_buffer("_marlin_decode_scales6", s6_backup, persistent=False)
            continue
        del w_backup, s6_backup
        _SWIGLU_GATE_UP[gs_key] = {
            "swiglu_out": swiglu_out,
            "down_gs_ptr": dn.weight_global_scale.data_ptr(),
            "is_ilv": is_ilv,
            "blocks": blocks,
            "state": "admitted",
        }
        count += 1
    if count:
        logger.info("Decode6 paired gate_up SwiGLU admitted bitwise on %d NVFP4 MLP layers", count)
    return count


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
    partials = _get_shared_partials(gate.device)
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


def register_all_oproj_sigmoid(model: torch.nn.Module) -> int:
    global _ALL_OPROJ_GATED_READY
    _ALL_OPROJ_GATED_READY = False
    _GATED_OPROJ.clear()
    if os.environ.get("MUSE_OPROJ_SIGMOID_FUSE", "1") == "0" or (_sk_tma4_ilv_gated is None and _sk_tma4_gated is None):
        return 0
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
    if non_fp8_total > 0 and admitted == non_fp8_total:
        _ALL_OPROJ_GATED_READY = True
        logger.info("Decode6 o_proj fused mul_sigmoid prologue prechecked bitwise on %d/%d NVFP4 layers", admitted, non_fp8_total)
    return admitted


def launch_down_swiglu6(
    gate_up_x: torch.Tensor,
    tiled_weight: torch.Tensor,
    scales6: torch.Tensor,
    palette: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: torch.Tensor,
    out_shape: tuple[int, ...],
) -> torch.Tensor:
    if not gate_up_x.is_contiguous() or gate_up_x.data_ptr() % 16:
        gate_up_x = gate_up_x.clone(memory_format=torch.contiguous_format)
    is_ilv = is_tma4_ilv(scales6)
    is_split2 = is_ilv and scales6.data_ptr() in _DOWN_SPLIT2_SCALES
    fn = _sk_tma4_ilv_swiglu_a if is_ilv else _sk_tma4_swiglu_a
    if fn is None:
        raise RuntimeError("SM121 Marlin down_proj SwiGLU-A kernel unavailable")
    blocks = 96 if is_split2 else 48
    locks_t = _get_split2_locks(gate_up_x.device) if (blocks == 96 and workspace.numel() < 256) else workspace
    partials = _get_shared_partials(gate_up_x.device)
    output = gate_up_x.new_empty(out_shape)
    with torch.cuda.device(gate_up_x.device):
        status = fn(
            gate_up_x.data_ptr(), tiled_weight.data_ptr(), output.data_ptr(), partials.data_ptr(),
            scales6.data_ptr(), palette.data_ptr(), global_scale.data_ptr(), locks_t.data_ptr(),
            DOWN_N, DOWN_K, blocks,
            gate_up_x.device.index, torch.cuda.current_stream().cuda_stream,
        )
    if status:
        raise RuntimeError(f"SM121 Marlin down_proj SwiGLU-A launch failed with CUDA error {status}")
    return output


def register_all_down_swiglu(model: torch.nn.Module) -> int:
    global _ALL_DOWN_SWIGLU_READY
    _ALL_DOWN_SWIGLU_READY = False
    _SWIGLU_DOWN_FUSE.clear()
    if os.environ.get("MUSE_DOWN_SWIGLU_FUSE", "0") == "0" or (_sk_tma4_ilv_swiglu_a is None and _sk_tma4_swiglu_a is None):
        return 0
    try:
        import muse_norm_glue as _ng
        if _ng.lib() is None or not hasattr(_ng, "mul_silu_slice_bf16"):
            return 0
    except Exception:
        return 0
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return 0
    gu_fused_downs = {v.get("down_gs_ptr") for v in _SWIGLU_GATE_UP.values()}
    non_fp8_total = 0
    admitted = 0
    for lyr in layers:
        mlp = getattr(lyr, "mlp", None)
        dn = getattr(mlp, "down_proj", None)
        if dn is None:
            continue
        w_raw = getattr(dn, "weight", None)
        if w_raw is not None and w_raw.dtype == torch.float8_e4m3fn:
            continue
        non_fp8_total += 1
        s6 = getattr(dn, "_marlin_decode_scales6", None)
        w = getattr(dn, "_marlin_decode_weight", None)
        pal = getattr(dn, "_marlin_decode_palette", None)
        gs = getattr(dn, "weight_global_scale", None)
        ws = getattr(dn, "workspace", None)
        if (
            s6 is None or w is None or pal is None or gs is None or ws is None
            or gs.data_ptr() in gu_fused_downs
            or getattr(dn, "marlin_decode6_fixed_grid", False)
            or not is_tma4(s6)
            or (dn.output_size_per_partition, dn.input_size_per_partition) != (DOWN_N, DOWN_K)
        ):
            continue
        is_ilv = is_tma4_ilv(s6)
        fn = _sk_tma4_ilv_swiglu_a if is_ilv else _sk_tma4_swiglu_a
        if fn is None:
            continue
        is_split2 = is_ilv and s6.data_ptr() in _DOWN_SPLIT2_SCALES
        dev = s6.device
        locks_t = _get_split2_locks(dev) if (is_split2 and ws.numel() < 256) else ws
        gen = torch.Generator(device=dev).manual_seed(20261128 + non_fp8_total)
        tmp = torch.empty((1, DOWN_K), dtype=torch.bfloat16, device=dev)
        ok = True
        with torch.cuda.device(dev):
            stream = torch.cuda.current_stream().cuda_stream
            for idx_p, sig in enumerate((0.05, 1.0, 5.0, 24.0)):
                rgu = torch.randn((1, 2 * DOWN_K), generator=gen, device=dev) * sig
                if idx_p % 2 == 0:
                    rgu[:, 1::2] = -rgu[:, ::2]
                rgu = rgu.to(torch.bfloat16).contiguous()
                _ng.mul_silu_slice_bf16(rgu, tmp, 1, stream)
                ref = decode_linear6(tmp, w, s6, pal, gs, ws, DOWN_N, DOWN_K, False)
                got = launch_down_swiglu6(rgu, w, s6, pal, gs, ws, (1, DOWN_N))
                torch.cuda.synchronize(dev)
                if (
                    not torch.equal(got.view(torch.int16), ref.view(torch.int16))
                    or int(torch.count_nonzero(ws).item()) != 0
                    or int(torch.count_nonzero(locks_t).item()) != 0
                ):
                    ok = False
                    break
        if ok:
            _SWIGLU_DOWN_FUSE.add(gs.data_ptr())
            admitted += 1
    if non_fp8_total > 0 and admitted == non_fp8_total:
        _ALL_DOWN_SWIGLU_READY = True
        logger.info("Decode6 down_proj fused mul_silu_slice prologue prechecked bitwise on %d/%d NVFP4 layers", admitted, non_fp8_total)
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
    """Admit 4-stage TMA super-block rANS scales on gate_up, down, attn, or lm_head after exact bitwise check."""
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
    for mag in (0.5, 4.0):
        probes.append((torch.randn(1, k, generator=gen, device=dev) * mag).to(torch.bfloat16))
    w = layer._marlin_decode_weight
    pal = layer._marlin_decode_palette
    gs = layer.weight_global_scale
    ws = layer.workspace
    expected = [decode_linear6(x, w, c, pal, gs, ws, n, k, fixed_grid) for x in probes]

    gb = _nv6_gate_blocks(dev) if (not fixed_grid and (n, k) == (4096, 6656)) else 48
    is_gate_v48 = not fixed_grid and (n, k) == (4096, 6656) and gb != 48
    is_gu_shape = fixed_grid and (n, k) == (GATE_UP_N, GATE_UP_K)
    is_dn_shape = not fixed_grid and (n, k) == (DOWN_N, DOWN_K)
    is_lm_shape = not fixed_grid and (n, k) == (LM_HEAD_N, LM_HEAD_K)

    def _try_commit_ilv(
        split2_set: set[int] | None,
        *,
        is_gu: bool = False,
        gate_v48: bool = False,
        gate_blocks: int = 48,
        split2: bool = False,
        down_split2: bool = False,
        grid: int = 48,
    ) -> bool:
        if _sk_tma4_ilv is None:
            return False
        try:
            coded_ilv = encode_tma4_scales(
                layer._marlin_decode_scales, pal, n, k, False,
                gate_v48=gate_v48, gate_blocks=gate_blocks,
                split2=split2, down_split2=down_split2, grid=grid,
            )
        except (RuntimeError, ValueError) as exc:
            logger.warning("TMA4-ILV (%dx%d, grid=%d) encode failed: %s", n, k, grid, exc)
            return False
        if coded_ilv.numel() > c.numel():
            logger.warning("TMA4-ILV (%dx%d, grid=%d) rejected: coded %d > c %d", n, k, grid, coded_ilv.numel(), c.numel())
            return False
        w_ilv = _build_ilv_weight(
            w, n, k, gate_v48, gate_blocks, split2=split2, down_split2=down_split2, grid=grid,
        )
        locks_t = _get_split2_locks(dev) if (grid == 96 and ws.numel() < 256) else ws
        _DOWN_ILV_SCALES.add(coded_ilv.data_ptr())
        if split2_set is not None:
            split2_set.add(coded_ilv.data_ptr())
        try:
            got_ilv = [decode_linear6(x, w_ilv, coded_ilv, pal, gs, ws, n, k, False) for x in probes]
        finally:
            _DOWN_ILV_SCALES.discard(coded_ilv.data_ptr())
            if split2_set is not None:
                split2_set.discard(coded_ilv.data_ptr())
        if not _same_bits(expected, got_ilv) or int(torch.count_nonzero(ws).item()) != 0 or int(torch.count_nonzero(locks_t).item()) != 0:
            del w_ilv, coded_ilv
            return False
        nbytes = coded_ilv.numel()
        w_backup = w.detach().clone() if w.nbytes <= (256 << 20) else None
        w.copy_(w_ilv)
        del w_ilv
        c[:nbytes].copy_(coded_ilv.view(-1))
        del coded_ilv
        view = c[:nbytes].view(-1, 32)
        _DOWN_ILV_SCALES.add(view.data_ptr())
        if split2_set is not None:
            split2_set.add(view.data_ptr())
        got2 = [decode_linear6(x, w, view, pal, gs, ws, n, k, False) for x in probes]
        if _same_bits(expected, got2) and int(torch.count_nonzero(ws).item()) == 0 and int(torch.count_nonzero(locks_t).item()) == 0:
            del w_backup
            layer.register_buffer("_marlin_decode_scales6", view, persistent=False)
            if is_gu:
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
        if split2_set is not None:
            split2_set.discard(view.data_ptr())
        if w_backup is not None:
            w.copy_(w_backup)
            del w_backup
        c.copy_(_six_bit_codes(layer))
        return False

    if is_gu_shape:
        if os.environ.get("MUSE_GATE_UP_SPLIT2", "1") != "0":
            if _try_commit_ilv(_GATE_UP_SPLIT2_SCALES, is_gu=True, grid=96):
                return True
        if os.environ.get("MUSE_GATE_UP_ILV", "1") != "0":
            if _try_commit_ilv(None, is_gu=True, grid=48):
                return True
    elif is_dn_shape:
        if os.environ.get("MUSE_DOWN_SPLIT2", "1") != "0":
            if _try_commit_ilv(_DOWN_SPLIT2_SCALES, down_split2=True, grid=96):
                return True
        if os.environ.get("MUSE_DOWN_ILV", "1") != "0":
            if _try_commit_ilv(None, grid=48):
                return True
    elif is_lm_shape:
        want_lm_ilv = os.environ.get("MUSE_LM_HEAD_ILV", "1") != "0"
        if want_lm_ilv and os.environ.get("MUSE_LM_HEAD_SPLIT2", "1") != "0":
            if _try_commit_ilv(_LM_SPLIT2_SCALES, split2=True, grid=96):
                return True
        if want_lm_ilv:
            if _try_commit_ilv(None, grid=48):
                return True
    elif not fixed_grid and (n, k) in ATTN_ILV_SHAPES and gb in (24, 48) and os.environ.get("MUSE_ATTN_ILV", "1") != "0":
        if _try_commit_ilv(None, gate_v48=is_gate_v48, gate_blocks=gb, grid=gb):
            return True

    coded = encode_tma4_scales(
        layer._marlin_decode_scales, pal, n, k, fixed_grid,
        gate_v48=is_gate_v48, gate_blocks=gb,
    )
    if coded.numel() > c.numel():
        logger.warning("TMA4 (%dx%d) rejected: coded %d > c %d", n, k, coded.numel(), c.numel())
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
