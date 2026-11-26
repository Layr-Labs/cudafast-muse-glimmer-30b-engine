# SPDX-License-Identifier: Apache-2.0
"""lmmips: H2 exact Cauchy-Schwarz / rank-32 subspace pruning of the target lm_head argmax (MUSE_LM_MIPS=1).

Greedy decode only needs argmax(LUT(bf16(z_v))) over the 202,048 vocab rows, z_v = fl(fl_TC(x . b_v) * g) the dense
fused kernel's value. At load we take the dense kernel's own effective weights (W~ = bf16(g * b), probed through the
dense GEMV; ``weight_rounding_delta`` bounds ||g b_v - W~_v||, 0 for rows that decode uniquely), fit a rank-32 basis V
(randomized range finder; any V is valid), and store per row fp16 coordinates c^_v plus three rounded-up norms
(d_v, r_v, e_v). Per call:

    U_{i,T} = max_{v in T} fl32(q_i . c^_v) + ||q_i|| d_v + ||y_i|| r_v + ||x_i|| e_v  >=  x_i . (g b_v)
    pass 1: dense masked kernel over seed tiles (top-2 tiles by U per row)      -> F_i (softcapped seed winner)
    pass 2: dense masked kernel over tiles with U_{i,T} >= RD(zlo(F_i) - eps_i) -> merge (value, lowest index)

eps_i = coef ||x_i|| max_v ||g b_v|| (rigorous fp32 / tensor-core dot error, u = 2^-22) + MUSE_LM_MIPS_EPS (>= 0.10).
zlo(F) is the largest z whose RNE bf16 rounding has LUT value < F, so every skipped column is strictly below the seed
winner: it can neither win nor tie and the merged argmax equals the dense one bit for bit. Kept tiles run the dense
kernel itself (bit-identical logits). Everything is fixed-address and stream-ordered (CUDA-graph capturable).

Admission (once per lm_head): masked-kernel bitwise check, dense error-model audit, then a 500-vector (default)
exact-argmax self-check against the dense fused kernel for each path (M=1 and multi-M); any failure leaves the
dense path in place.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "RANK", "TILE", "ROW_F", "MAX_M", "EPS_FLOOR_MIN",
    "enabled", "eps_floor", "err_coef", "up_f32", "down_f32",
    "build_zlo_table", "weight_rounding_delta", "range_finder", "SideData", "precompute_side_data",
    "reference_rowbuf", "reference_bounds", "reference_seed", "reference_thresh", "dense_final",
    "masked_argmax", "merge", "reference_pruned_argmax", "tma4_tile_states",
    "State", "run", "build_state", "probe_vectors", "self_check",
]

RANK = 32
TILE = 128
ROW_F = 40  # rowbuf floats per row (matches lmmips.cu kRowF)
MAX_M = 16
EPS_FLOOR_MIN = 0.10


def enabled() -> bool:
    return os.environ.get("MUSE_LM_MIPS", "0") not in ("", "0")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def eps_floor() -> float:
    return max(EPS_FLOOR_MIN, _env_float("MUSE_LM_MIPS_EPS", EPS_FLOOR_MIN))


def err_coef(k: int, u_log2: int | None = None) -> float:
    """Relative bound on |z - x.(g b)| / (||x|| max||g b||): gamma_K for the tensor-core fp32 accumulation (unit
    roundoff 2^u_log2, default 2^-22 = 4x fp32 RN to cover truncating accumulators), one fp32 multiply by g, plus
    1e-5 for the fp32 evaluation of the bound itself."""
    if u_log2 is None:
        u_log2 = _env_int("MUSE_LM_MIPS_UTC_LOG2", -22)
    u = 2.0 ** min(u_log2, -22)
    ku = k * u
    if ku >= 0.5:
        raise ValueError("K too large for the dot-product error bound")
    return ku / (1.0 - ku) + 2.0 ** -23 + 1e-5


# ----------------------------------------------------------------------------------------------- fp helpers
def up_f32(v: torch.Tensor) -> torch.Tensor:
    """fp64 -> fp32 rounded toward +inf (elementwise)."""
    f = v.to(torch.float32)
    lo = f.to(torch.float64) < v
    return torch.where(lo, torch.nextafter(f, torch.full_like(f, math.inf)), f)


def down_f32(v: torch.Tensor) -> torch.Tensor:
    f = v.to(torch.float32)
    hi = f.to(torch.float64) > v
    return torch.where(hi, torch.nextafter(f, torch.full_like(f, -math.inf)), f)


def _codes_i16(codes: torch.Tensor) -> torch.Tensor:
    c = codes.to(torch.int32) & 0xFFFF
    return torch.where(c >= 32768, c - 65536, c).to(torch.int16)


def _bf16_from_codes(codes: torch.Tensor) -> torch.Tensor:
    return _codes_i16(codes).view(torch.bfloat16)


def _codes_from_bf16(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def _key_from_code(c: torch.Tensor) -> torch.Tensor:
    """Order-preserving integer key of bf16 codes (-0 and +0 adjacent)."""
    return torch.where(c < 0x8000, c ^ 0x8000, 0xFFFF - c)


def _code_from_key(k: torch.Tensor) -> torch.Tensor:
    return torch.where(k >= 0x8000, k ^ 0x8000, 0xFFFF - k)


# ----------------------------------------------------------------------------------------------- zlo table
def build_zlo_table(lut: torch.Tensor) -> torch.Tensor:
    """fp32[65536] indexed by a final (softcapped) bf16 code F: zlo(F) = the lower RNE edge of the smallest raw bf16
    value b* with LUT(b*) >= F. Any fp32 z < zlo(F) rounds to a raw value below b*, hence LUT(bf16(z)) < F.
    No monotonicity of the LUT is assumed (prefix max). Runs on CPU."""
    lut = lut.detach().to("cpu")
    if lut.numel() != 65536:
        raise ValueError("lmpost LUT must have 65536 entries")
    lut_codes = lut.view(torch.int16).to(torch.int32) & 0xFFFF
    codes = torch.arange(65536, dtype=torch.int32)
    raw = _bf16_from_codes(codes).to(torch.float64)
    fin = _bf16_from_codes(lut_codes).to(torch.float64)
    fin = torch.where(torch.isnan(fin), torch.full_like(fin, -math.inf), fin)
    valid = ~torch.isnan(raw)
    vc = codes[valid]
    order = torch.argsort(_key_from_code(vc))
    vc = vc[order]
    vals = raw[vc]
    lutv = fin[vc]
    new_g = torch.ones_like(vals, dtype=torch.bool)
    new_g[1:] = vals[1:] != vals[:-1]
    gid = torch.cumsum(new_g.to(torch.int64), 0) - 1
    n_g = int(gid[-1].item()) + 1
    gval = vals[new_g]
    glut = torch.full((n_g,), -math.inf, dtype=torch.float64).scatter_reduce(0, gid, lutv, reduce="amax")
    pref = torch.cummax(glut, 0).values
    max_bf16 = (2.0 - 2.0 ** -7) * 2.0 ** 127
    half_over = (2.0 - 2.0 ** -8) * 2.0 ** 127  # RNE edge between max finite bf16 and inf
    edge = torch.empty(n_g, dtype=torch.float64)
    edge[0] = -math.inf
    prev, cur = gval[:-1], gval[1:]
    mid = (prev + cur) * 0.5
    mid = torch.where(torch.isinf(cur) & (cur > 0), torch.full_like(mid, half_over), mid)
    mid = torch.where(torch.isinf(prev) & (prev < 0), torch.full_like(mid, -half_over), mid)
    edge[1:] = mid
    assert float(gval[1]) == -max_bf16 and float(gval[-2]) == max_bf16
    fvals = _bf16_from_codes(codes).to(torch.float64)
    idx = torch.searchsorted(pref, torch.where(torch.isnan(fvals), torch.full_like(fvals, -math.inf), fvals))
    z = torch.where(idx < n_g, edge[idx.clamp(max=n_g - 1)], torch.full_like(fvals, math.inf))
    z = torch.where(torch.isnan(fvals), torch.full_like(fvals, -math.inf), z)
    z32 = z.to(torch.float32)
    if not torch.equal(z32.to(torch.float64), z):
        raise AssertionError("zlo edge not exactly representable in fp32")
    return z32.contiguous()


# ----------------------------------------------------------------------------------------------- weights
def weight_rounding_delta(wt: torch.Tensor, g: float, chunk: int = 2048) -> tuple[torch.Tensor, int]:
    """wt: bf16 [N, K] = bf16(fl32(b * g)) as probed from the dense kernel (b the bf16 dequantized NVFP4 weight,
    g the fp32 global scale). Returns (delta fp64 [N] with ||g b_v - wt_v|| <= delta_v, #ambiguous elements).
    The preimage {b : bf16(fl32(b g)) == wt} is searched within +-2 bf16 steps of bf16(wt / g); elements whose
    preimage is not confined there fall back to |wt| (2^-8 + 2^-23) + 2^-133."""
    g32 = torch.tensor(g, dtype=torch.float32, device=wt.device)
    g64 = float(g32.item())
    n = wt.shape[0]
    out = torch.empty(n, dtype=torch.float64, device=wt.device)
    n_amb = 0
    for a in range(0, n, chunk):
        w = wt[a : a + chunk]
        wf = w.to(torch.float32)
        k0 = _key_from_code(_codes_from_bf16((wf / g32).to(torch.bfloat16)))
        cand_err = []
        inpre = []
        for dk in (-2, -1, 0, 1, 2):
            kk = (k0 + dk).clamp(0, 0xFFFF)
            b = _bf16_from_codes(_code_from_key(kk))
            bf = b.to(torch.float32)
            img = (bf * g32).to(torch.bfloat16).to(torch.float32)
            ok = (img == wf) & torch.isfinite(bf)
            inpre.append(ok)
            cand_err.append(torch.where(ok, (bf.to(torch.float64) * g64 - wf.to(torch.float64)).abs(),
                                        torch.zeros_like(wf, dtype=torch.float64)))
        ok_any = inpre[1] | inpre[2] | inpre[3]
        confined = ok_any & ~inpre[0] & ~inpre[4]
        err = torch.stack(cand_err[1:4]).amax(0)
        loose = wf.to(torch.float64).abs() * (2.0 ** -8 + 2.0 ** -23) + 2.0 ** -133
        err = torch.where(confined, err, loose)
        n_amb += int((~confined).sum().item())
        out[a : a + chunk] = torch.sqrt((err * err).sum(1)) * (1.0 + 1e-9)
    return out, n_amb


def range_finder(wt: torch.Tensor, r: int = RANK, oversample: int = 16, power: int = 2, seed: int = 20261127,
                 chunk: int = 8192) -> torch.Tensor:
    """Top-r right singular subspace of W (rows of wt) via a randomized range finder; returns fp32 [K, r]
    (orthonormal to ~1e-7; the bound does not need orthonormality)."""
    dev = wt.device
    n, k = wt.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((k, r + oversample), generator=gen, dtype=torch.float32).to(dev)

    def wtw(m: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros((k, m.shape[1]), dtype=torch.float64, device=dev)
        for a in range(0, n, chunk):
            wc = wt[a : a + chunk].to(torch.float32)
            acc += (wc.T @ (wc @ m)).to(torch.float64)
        return acc

    for _ in range(power):
        q, _ = torch.linalg.qr(wtw(q))
        q = q.to(torch.float32)
    gram = wtw(q).T.to(torch.float64) @ q.to(torch.float64)  # Q^T W^T W Q
    gram = 0.5 * (gram + gram.T)
    _, evec = torch.linalg.eigh(gram)
    v = q.to(torch.float64) @ evec[:, -r:].flip(1)
    v, _ = torch.linalg.qr(v)
    return v.to(torch.float32).contiguous()


@dataclass
class SideData:
    vmat: torch.Tensor  # fp32 [K, 32]
    chat: torch.Tensor  # fp16 [Npad, 32]
    aux: torch.Tensor  # fp32 [Npad, 4]: d', r', e', 0
    wmax: float
    n_tiles: int
    org_vocab: int
    resid_frac: float = 0.0


def precompute_side_data(wt: torch.Tensor, delta: torch.Tensor, org_vocab: int, vmat: torch.Tensor | None = None,
                         chunk: int = 4096) -> SideData:
    """wt bf16 [N, K] (effective rows, any device), delta fp64 [N]. All norms in fp64, stored rounded up."""
    dev = wt.device
    n, k = wt.shape
    if vmat is None:
        vmat = range_finder(wt)
    v64 = vmat.to(torch.float64)
    n_tiles = -(-n // TILE)
    npad = n_tiles * TILE
    chat = torch.zeros((npad, RANK), dtype=torch.float16, device=dev)
    aux = torch.zeros((npad, 4), dtype=torch.float32, device=dev)
    wmax = 0.0
    rsum = 0.0
    wsum = 0.0
    u24 = 2.0 ** -24
    for a in range(0, n, chunk):
        w = wt[a : a + chunk].to(torch.float64)
        c = w @ v64
        ch = c.to(torch.float16)
        chd = ch.to(torch.float64)
        rho = w - chd @ v64.T
        r = torch.linalg.vector_norm(rho, dim=1)
        d = torch.linalg.vector_norm(rho @ v64, dim=1)
        cn = torch.linalg.vector_norm(chd, dim=1)
        wn = torch.linalg.vector_norm(w, dim=1)
        g = 1.0 + 1e-9
        dl = delta[a : a + chunk].to(torch.float64)
        aux[a : a + chunk, 0] = up_f32(d * g + 34.0 * u24 * cn * g + 1e-30)
        aux[a : a + chunk, 1] = up_f32(r * g + 1e-30)
        aux[a : a + chunk, 2] = up_f32(dl * g + 1e-10 * cn + 1e-30)
        chat[a : a + chunk] = ch
        live = min(org_vocab, a + w.shape[0]) - a
        if live > 0:
            wmax = max(wmax, float((wn[:live] * g + dl[:live]).max().item()))
            rsum += float(r[:live].sum().item())
            wsum += float(wn[:live].sum().item())
    wmax32 = float(up_f32(torch.tensor([wmax * (1.0 + 1e-6)], dtype=torch.float64))[0].item())
    return SideData(vmat.contiguous(), chat.contiguous(), aux.contiguous(), wmax32, n_tiles, org_vocab,
                    rsum / max(wsum, 1e-30))


# ----------------------------------------------------------------------------------------------- reference
def reference_rowbuf(x: torch.Tensor, sd: SideData, eps_fl: float, coef: float) -> torch.Tensor:
    """CPU/torch mirror of lmmips_project_kernel: [m, ROW_F] fp32."""
    m, k = x.shape
    xd = x.to(torch.float32).to(torch.float64)
    v64 = sd.vmat.to(torch.float64)
    q32 = (xd @ v64).to(torch.float32)
    y = xd - q32.to(torch.float64) @ v64.T
    g = 1.0 + 1e-9
    qn = up_f32(torch.linalg.vector_norm(q32.to(torch.float64), dim=1) * g)
    yn = up_f32(torch.linalg.vector_norm(y, dim=1) * g)
    xn = up_f32(torch.linalg.vector_norm(xd, dim=1) * g)
    eps = up_f32(coef * xn.to(torch.float64) * sd.wmax + eps_fl)
    finite = torch.isfinite(xd).all(1) & torch.isfinite(eps)
    eps = torch.where(finite, eps, torch.full_like(eps, math.inf))
    rb = torch.zeros((m, ROW_F), dtype=torch.float32, device=x.device)
    rb[:, :RANK] = q32
    rb[:, RANK] = qn
    rb[:, RANK + 1] = yn
    rb[:, RANK + 2] = xn
    rb[:, RANK + 3] = eps
    rb[:, RANK + 4] = finite.to(torch.float32)
    return rb


def reference_bounds(rb: torch.Tensor, sd: SideData, residual_term: bool = True) -> torch.Tensor:
    """Mirror of lmmips_bound_kernel: U fp32 [m, n_tiles]."""
    q = rb[:, :RANK]
    s = q @ sd.chat.to(torch.float32).T
    a = sd.aux
    u = s + rb[:, RANK : RANK + 1] * a[:, 0][None]
    if residual_term:
        u = u + rb[:, RANK + 1 : RANK + 2] * a[:, 1][None]
    u = u + rb[:, RANK + 2 : RANK + 3] * a[:, 2][None]
    u[:, sd.org_vocab :] = -math.inf
    u = torch.where(torch.isnan(u), torch.full_like(u, math.inf), u)
    return u.view(u.shape[0], sd.n_tiles, TILE).amax(2)


def reference_seed(ub: torch.Tensor, rb: torch.Tensor, topk: int, hot_tiles: int) -> torch.Tensor:
    m, nt = ub.shape
    seed = torch.zeros(nt, dtype=torch.bool, device=ub.device)
    seed[: min(hot_tiles, nt)] = True
    for i in range(m):
        order = torch.argsort(-ub[i], stable=True)  # descending, ties -> lowest tile
        seed[order[: min(topk, 4)]] = True
    if bool((rb[:, RANK + 4] == 0).any()):
        seed[:] = False
    return seed


def reference_thresh(ub: torch.Tensor, rb: torch.Tensor, val1: torch.Tensor, zlo: torch.Tensor,
                     seed: torch.Tensor) -> torch.Tensor:
    thr = torch.full((ub.shape[0],), -math.inf, dtype=torch.float32)
    for i in range(ub.shape[0]):
        f = float(val1[i])
        if math.isnan(f) or f == -math.inf:
            continue
        code = int(_codes_from_bf16(val1[i : i + 1].to(torch.bfloat16))[0])
        t = down_f32(torch.tensor([float(zlo[code]) - float(rb[i, RANK + 3])], dtype=torch.float64))[0]
        thr[i] = -math.inf if torch.isnan(t) else t
    keep = (~(ub < thr[:, None].to(ub.device))).any(0)
    return keep & ~seed


def dense_final(x: torch.Tensor, b: torch.Tensor, g: float, lut: torch.Tensor) -> torch.Tensor:
    """Model of the dense fused kernel: z = fl32(fl32(x . b) * g) -> bf16 -> LUT; returns final fp32 [m, N]."""
    acc = x.to(torch.float32) @ b.to(torch.float32).T
    z = acc * torch.tensor(g, dtype=torch.float32)
    raw = _codes_from_bf16(z.to(torch.bfloat16))
    lut_codes = lut.view(torch.int16).to(torch.int32) & 0xFFFF
    return _bf16_from_codes(lut_codes[raw.long()]).to(torch.float32)


def masked_argmax(final: torch.Tensor, tile_keep: torch.Tensor, org_vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(value, lowest index) over kept tiles of each row; (-inf, 2^31-1) when nothing is kept."""
    m, n = final.shape
    col_keep = tile_keep.repeat_interleave(TILE)[:n].clone()
    col_keep[org_vocab:] = False
    f = torch.where(col_keep[None], final, torch.full_like(final, -math.inf))
    val = f.amax(1)
    hit = (f == val[:, None]) & col_keep[None]
    idx = torch.where(hit, torch.arange(n)[None].expand(m, n), torch.full((m, n), 2 ** 31 - 1))
    tok = idx.amin(1)
    return val, tok


def merge(val1, tok1, val2, tok2) -> torch.Tensor:
    pick1 = (val1 > val2) | ((val1 == val2) & (tok1 < tok2))
    return torch.where(pick1, tok1, tok2)


def reference_pruned_argmax(x: torch.Tensor, final: torch.Tensor, sd: SideData, zlo: torch.Tensor, eps_fl: float,
                            coef: float, topk: int = 2, hot_tiles: int = 0, residual_term: bool = True):
    """Full CPU mirror of one H2 call given the dense finals (only kept tiles are read). Returns
    (tokens, keep_union bool [n_tiles], U, thresholds-per-row-pass)."""
    rb = reference_rowbuf(x, sd, eps_fl, coef)
    ub = reference_bounds(rb, sd, residual_term)
    seed = reference_seed(ub, rb, topk, hot_tiles)
    val1, tok1 = masked_argmax(final, seed, sd.org_vocab)
    mask2 = reference_thresh(ub, rb, val1, zlo, seed)
    val2, tok2 = masked_argmax(final, mask2, sd.org_vocab)
    return merge(val1, tok1, val2, tok2), seed | mask2, ub, val1


# ----------------------------------------------------------------------------------- masked-tile rANS entry states
def _u32(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32).long() & 0xFFFFFFFF


def tma4_tile_states(coded: torch.Tensor, n_tiles: int = 1579, stages_per_tile: int = 52) -> torch.Tensor:
    """Decoder state at the first scale symbol of every tile of a fixed-grid TMA4 lm_head scale stream.

    The masked lm_head kernel walks only kept tiles, so it resumes each CTA's rANS stream from these states.
    Replays the kernel's tma_decode (marlin_template.h) for all CTAs at once: CTA b owns tiles b, b + grid, ...;
    a 4-stage super-block starts warp w at u16 word warps + sum(cw[:w]); per stage each lane decodes 4 symbols
    (e = lut[x & 1023]; x = f * (x >> 10) + bias; if x < 2^16 the lane takes the warp's next word in lane order).
    Returns int32 [n_tiles, threads] (u32 bit patterns); row b (< grid) equals CTA b's header state. Raises
    ValueError on a layout mismatch, including any warp not consuming exactly its stored word count."""
    raw = coded.contiguous().view(torch.uint8).reshape(-1)
    total_sbs, grid, max_sbs, threads = (int(v) for v in _u32(raw[4096:4112]).tolist())
    warps = threads // 32
    blk_it = _u32(raw[4112 : 4112 + 4 * grid])
    sb_row = _u32(raw[4112 + 4 * grid : 4112 + 4 * (grid + max_sbs)])
    meta_bytes = ((4 + grid + max_sbs + 3) // 4) * 16
    off16 = _u32(raw[4096 + meta_bytes : 4096 + meta_bytes + 4 * (total_sbs + 1)])
    states_off = 4096 + meta_bytes + ((total_sbs + 1) * 4 + 15) // 16 * 16
    var_off = states_off + grid * threads * 4
    if stages_per_tile % 4 or bool((blk_it % stages_per_tile).any()):
        raise ValueError("TMA4 stream is not tile-aligned")
    if int(blk_it.sum().item()) != n_tiles * stages_per_tile:
        raise ValueError("TMA4 stream does not cover the lm_head tiles")
    lut = _u32(raw[:4096])
    var = raw[var_off : var_off + (raw.numel() - var_off) // 2 * 2]
    words = var.view(torch.int16).long() & 0xFFFF
    dev = raw.device
    x = _u32(raw[states_off:var_off]).reshape(grid, warps, 32)
    out = torch.zeros((n_tiles, threads), dtype=torch.long, device=dev)
    out[:grid] = x.reshape(grid, threads)
    cta = torch.arange(grid, device=dev)
    n_sbs = blk_it // 4
    sb_per_tile = stages_per_tile // 4
    lanes = torch.arange(32, device=dev)
    below = (lanes[None, :] < lanes[:, None]).long()  # below[l, j] = 1 iff lane j < lane l
    w_idx = torch.arange(warps - 1, device=dev)
    for s in range(int(n_sbs.max().item())):
        act = n_sbs > s
        gsb = (sb_row[s] + cta).clamp(max=total_sbs - 1)
        base = 8 * off16[gsb]  # u16 index of the super-block (16-byte units)
        cw = words[base[:, None] + w_idx[None, :]]
        start = torch.zeros((grid, warps), dtype=torch.long, device=dev)
        start[:, 1:] = torch.cumsum(cw, 1)
        wp0 = base[:, None] + warps + start
        wp = wp0
        for _ in range(16):  # 4 stages x 4 symbols
            e = lut[x & 1023]
            xn = ((e >> 6) & 2047) * (x >> 10) + (e >> 17)
            need = (xn < 65536) & act[:, None, None]
            rank = (need.long()[..., None, :] * below).sum(-1)
            w = words[(wp[..., None] + rank).clamp(max=words.numel() - 1)]
            x = torch.where(act[:, None, None], torch.where(need, (xn << 16) | w, xn), x)
            wp = wp + need.long().sum(-1)
        if bool(((wp - wp0)[:, : warps - 1] != cw)[act].any()):
            raise ValueError(f"TMA4 replay word count mismatch at super-block {s}")
        if (s + 1) % sb_per_tile == 0:
            tiles = cta + ((s + 1) // sb_per_tile) * grid
            keep = n_sbs > s + 1
            out[tiles[keep]] = x.reshape(grid, threads)[keep]
    return torch.where(out >= 2**31, out - 2**32, out).to(torch.int32).contiguous()


# ----------------------------------------------------------------------------------------------- GPU path
_lib = None
_load_failed = False


def _load():
    global _lib, _load_failed
    if _lib is not None or _load_failed:
        return _lib
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("liblmmips.so")))
        vp, ci, cf = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        lib.lmmips_project.argtypes = [vp, vp, ci, ci, cf, cf, cf, vp, vp]
        lib.lmmips_bound.argtypes = [vp, vp, vp, ci, ci, ci, vp, vp]
        lib.lmmips_seed.argtypes = [vp, vp, ci, ci, ci, ci, vp, ci, vp]
        lib.lmmips_thresh.argtypes = [vp, vp, vp, vp, vp, ci, ci, ci, vp, vp, vp]
        lib.lmmips_merge.argtypes = [vp, vp, vp, vp, ci, vp, vp]
        for fn in ("lmmips_project", "lmmips_bound", "lmmips_seed", "lmmips_thresh", "lmmips_merge"):
            getattr(lib, fn).restype = ci
        _lib = lib
    except OSError:
        _load_failed = True
    return _lib


@dataclass
class State:
    sd: SideData
    zlo: torch.Tensor
    eps_floor: float
    coef: float
    topk: int
    hot_tiles: int
    n_words: int
    rowbuf: torch.Tensor
    ubuf: torch.Tensor
    seed_mask: torch.Tensor
    mask2: torch.Tensor
    tok1: torch.Tensor
    val1: torch.Tensor
    tok2: torch.Tensor
    val2: torch.Tensor
    stats: torch.Tensor
    ok: dict = field(default_factory=dict)  # "multi" / "single" -> bool
    launch: object = None  # masked dense lm_head launcher (set by muse_marlin6_gemv at admission)


def _check(rc: int, name: str) -> None:
    if rc:
        raise RuntimeError(f"{name} failed with CUDA error {rc}")


def run(state: State, x: torch.Tensor, out_tokens: torch.Tensor, masked_fn, multi: bool) -> None:
    """One H2 call: x bf16 [m, K] contiguous, out_tokens int64 with >= m elements (first m written).
    masked_fn(x, tile_mask, tok_out, val_out, multi) launches the dense masked lm_head kernel."""
    lib = _lib
    sd = state.sd
    m, k = int(x.shape[0]), int(x.shape[1])
    st = torch.cuda.current_stream(x.device).cuda_stream
    _check(lib.lmmips_project(x.data_ptr(), sd.vmat.data_ptr(), k, m, sd.wmax, state.coef, state.eps_floor,
                              state.rowbuf.data_ptr(), st), "lmmips_project")
    _check(lib.lmmips_bound(sd.chat.data_ptr(), sd.aux.data_ptr(), state.rowbuf.data_ptr(), m, sd.org_vocab,
                            sd.n_tiles, state.ubuf.data_ptr(), st), "lmmips_bound")
    _check(lib.lmmips_seed(state.ubuf.data_ptr(), state.rowbuf.data_ptr(), m, sd.n_tiles, state.hot_tiles,
                           state.topk, state.seed_mask.data_ptr(), state.n_words, st), "lmmips_seed")
    masked_fn(x, state.seed_mask, state.tok1, state.val1, multi)
    _check(lib.lmmips_thresh(state.ubuf.data_ptr(), state.rowbuf.data_ptr(), state.val1.data_ptr(),
                             state.zlo.data_ptr(), state.seed_mask.data_ptr(), m, sd.n_tiles, state.n_words,
                             state.mask2.data_ptr(), state.stats.data_ptr(), st), "lmmips_thresh")
    masked_fn(x, state.mask2, state.tok2, state.val2, multi)
    _check(lib.lmmips_merge(state.tok1.data_ptr(), state.val1.data_ptr(), state.tok2.data_ptr(),
                            state.val2.data_ptr(), m, out_tokens.data_ptr(), st), "lmmips_merge")


def build_state(wt: torch.Tensor, g: float, lut: torch.Tensor, org_vocab: int) -> State:
    dev = wt.device
    delta, n_amb = weight_rounding_delta(wt, g)
    sd = precompute_side_data(wt, delta, org_vocab)
    zlo = build_zlo_table(lut).to(dev)
    n_words = -(-sd.n_tiles // 32)
    if n_words > 64:
        raise ValueError("too many tiles for the 64-word masks")
    st = State(
        sd=sd, zlo=zlo, eps_floor=eps_floor(), coef=err_coef(wt.shape[1]),
        topk=max(1, min(4, _env_int("MUSE_LM_MIPS_TOPK", 2))), hot_tiles=max(0, _env_int("MUSE_LM_MIPS_HOT_TILES", 0)),
        n_words=n_words,
        rowbuf=torch.zeros((MAX_M, ROW_F), dtype=torch.float32, device=dev),
        ubuf=torch.zeros((MAX_M, sd.n_tiles), dtype=torch.float32, device=dev),
        seed_mask=torch.zeros(64, dtype=torch.int32, device=dev),
        mask2=torch.zeros(64, dtype=torch.int32, device=dev),
        tok1=torch.zeros(MAX_M, dtype=torch.int64, device=dev),
        val1=torch.zeros(MAX_M, dtype=torch.float32, device=dev),
        tok2=torch.zeros(MAX_M, dtype=torch.int64, device=dev),
        val2=torch.zeros(MAX_M, dtype=torch.float32, device=dev),
        stats=torch.zeros(2, dtype=torch.int64, device=dev),
    )
    logger.info(
        "H2 lm_head MIPS side data: rank %d, %d tiles, %.1f MiB, wmax %.4f, mean residual frac %.3f, "
        "ambiguous probe elements %d, eps floor %.2f, err coef %.3g",
        RANK, sd.n_tiles, (sd.chat.numel() * 2 + sd.aux.numel() * 4 + sd.vmat.numel() * 4) / 2 ** 20,
        sd.wmax, sd.resid_frac, n_amb, st.eps_floor, st.coef,
    )
    return st


def probe_vectors(wt: torch.Tensor, n: int, k: int, seed: int, org_vocab: int) -> torch.Tensor:
    """Self-check queries (bf16 [n, K]): peaked on real lm_head rows at several cosines/magnitudes (the regime where
    pruning skips), near-tie pairs of two real rows, and isotropic gaussians."""
    dev = wt.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    xs = []
    for j in range(n):
        kind = j % 5
        noise = torch.randn(k, generator=gen, dtype=torch.float64)
        noise /= noise.norm()
        if kind == 4:
            mag = float(torch.empty(1).uniform_(20.0, 160.0, generator=gen))
            xs.append(noise * mag)
            continue
        t = int(torch.randint(0, org_vocab, (1,), generator=gen))
        wtv = wt[t].to(torch.float64).cpu()
        wtv = wtv / wtv.norm().clamp_min(1e-30)
        if kind == 3:
            u = int(torch.randint(0, org_vocab, (1,), generator=gen))
            wu = wt[u].to(torch.float64).cpu()
            wu = wu / wu.norm().clamp_min(1e-30)
            d = wtv + wu
            base = d / d.norm().clamp_min(1e-30)
        else:
            base = wtv
        cos = (0.9, 0.6, 0.35)[kind % 3]
        noise = noise - (noise @ base) * base
        noise /= noise.norm().clamp_min(1e-30)
        mag = float(torch.empty(1).uniform_(20.0, 160.0, generator=gen))
        xs.append(mag * (cos * base + math.sqrt(1 - cos * cos) * noise))
    return torch.stack(xs).to(torch.bfloat16).to(dev).contiguous()


def self_check(state: State, wt: torch.Tensor, masked_fn, dense_fn, multi: bool, n_vec: int) -> bool:
    """n_vec-vector exact argmax check of the full H2 path vs the dense fused kernel (0-ULP: identical token ids)."""
    k = wt.shape[1]
    dev = wt.device
    xs = probe_vectors(wt, n_vec, k, 20261127 + int(multi), state.sd.org_vocab)
    sizes = (16, 12, 7, 2, 1, 16, 9, 16) if multi else (1,)
    a = 0
    j = 0
    bad = 0
    st0 = state.stats.clone()
    while a < n_vec:
        m = min(sizes[j % len(sizes)], n_vec - a)
        j += 1
        x = xs[a : a + m].contiguous()
        a += m
        ref = torch.full((1, MAX_M), -1, dtype=torch.int64, device=dev)
        got = torch.full((1, MAX_M), -2, dtype=torch.int64, device=dev)
        if not dense_fn(x, ref[:, :m] if multi else ref[:, :1]):
            return False
        run(state, x, got, masked_fn, multi)
        torch.cuda.synchronize(dev)
        if not torch.equal(ref[0, :m], got[0, :m]):
            bad += int((ref[0, :m] != got[0, :m]).sum().item())
    kept = state.stats - st0
    calls = max(int(kept[1].item()), 1)
    frac = float(kept[0].item()) / (calls * state.sd.n_tiles)
    logger.info("H2 lm_head MIPS self-check (%s): %d/%d exact argmax, pass-2 tiles kept %.1f%%",
                "multi-M" if multi else "M=1", n_vec - bad, n_vec, 100.0 * frac)
    return bad == 0
