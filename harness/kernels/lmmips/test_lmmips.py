"""GPU-free unit tests for H2 (harness/kernels/lmmips): exact subspace branch-and-bound lm_head argmax pruning.

A synthetic NVFP4 head (e2m1 x e4m3 group-16 scales x fp32 global scale, decaying spectrum, real-head-like row norms)
goes through the production precompute (probe-row rounding preimages, range finder, fp16 coordinates, rounded-up
norms, zlo table from an lmpost-style softcap LUT) and a torch mirror of every GPU kernel. The dense fused kernel is
modelled as z = fl32(fl32(x . b) * g) -> bf16 -> LUT -> lowest-index argmax.

Run: python3 -m unittest harness/kernels/lmmips/test_lmmips.py -v
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


def _load():
    path = Path(__file__).resolve().parent / "py" / "__init__.py"
    spec = importlib.util.spec_from_file_location("lmmips_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


L = _load()
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def make_lut(mult: float, cap: float) -> torch.Tensor:
    """Same ops as lmpost._get_lut (BF16 tensor ops), on CPU."""
    raw = L._bf16_from_codes(torch.arange(65536, dtype=torch.int32))
    ref = raw * float(mult)
    ref = cap * torch.tanh(ref / cap)
    return ref.view(torch.int16).contiguous()


def make_head(n: int, k: int, g: float, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    s = torch.arange(1, k + 1, dtype=torch.float64) ** -0.8
    s[:8] *= 3.0
    s = s / s.norm()
    w = torch.randn((n, k), generator=gen, dtype=torch.float64) * s
    w *= torch.exp(0.15 * torch.randn((n, 1), generator=gen, dtype=torch.float64))
    sc = w / g
    grp = sc.view(n, k // 16, 16)
    scale = (grp.abs().amax(2, keepdim=True) / 6.0).clamp_min(2.0 ** -9)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float64).clamp_min(2.0 ** -9)
    q = (grp / scale).clamp(-6, 6)
    mag = q.abs()
    idx = (mag[..., None] - E2M1.double()).abs().argmin(-1)
    e = E2M1.double()[idx] * torch.sign(q)
    b64 = (e * scale).view(n, k)
    b = b64.to(torch.bfloat16)
    assert torch.equal(b.double(), b64), "e2m1 x e4m3 must be exact in bf16"
    g32 = torch.tensor(g, dtype=torch.float32)
    wt = (b.float() * g32).to(torch.bfloat16)
    return b, wt, float(g32)


def queries(wt: torch.Tensor, n: int, mode: str, cos: float, scale: float, seed: int, org_vocab: int):
    gen = torch.Generator().manual_seed(seed)
    k = wt.shape[1]
    x = torch.randn((n, k), generator=gen, dtype=torch.float64)
    x /= x.norm(dim=1, keepdim=True)
    if mode == "isotropic":
        return (x * scale).to(torch.bfloat16)
    zipf = torch.distributions.Pareto(torch.tensor(1.0), torch.tensor(0.2)).sample((n,))
    t = (zipf.floor().long() - 1).clamp(0, org_vocab - 1)
    wtv = wt[t].double()
    wtv /= wtv.norm(dim=1, keepdim=True)
    x -= (x * wtv).sum(1, keepdim=True) * wtv
    x /= x.norm(dim=1, keepdim=True)
    x = scale * (cos * wtv + math.sqrt(1 - cos * cos) * x)
    if mode == "tie":
        u = torch.randint(org_vocab // 2, org_vocab, (n,), generator=gen)
        gdir = wt[u].double() - wt[t].double()
        gap = (x * gdir).sum(1) + 1e-3
        x = x - (gap / (gdir * gdir).sum(1))[:, None] * gdir
    return x.to(torch.bfloat16)


class H2Fixture:
    N, ORG, K, G = 128 * 200, 128 * 200 - 30, 1024, 0.0137
    MULT, CAP = 1.0, 30.0

    def __init__(self):
        self.b, self.wt, self.g = make_head(self.N, self.K, self.G)
        self.delta, self.n_amb = L.weight_rounding_delta(self.wt, self.g)
        self.sd = L.precompute_side_data(self.wt, self.delta, self.ORG)
        self.lut = make_lut(self.MULT, self.CAP)
        self.zlo = L.build_zlo_table(self.lut)
        self.coef = L.err_coef(self.K)
        self.eps = L.eps_floor()
        self.weff = self.b.double() * self.g  # exact real effective weights


_FIX: H2Fixture | None = None


def fix() -> H2Fixture:
    global _FIX
    if _FIX is None:
        torch.manual_seed(0)
        _FIX = H2Fixture()
    return _FIX


class TestZloTable(unittest.TestCase):
    def test_zlo_sound_and_tight(self):
        f = fix()
        lut_codes = f.lut.to(torch.int32) & 0xFFFF
        final_of = lambda z: L._bf16_from_codes(lut_codes[L._codes_from_bf16(z.to(torch.bfloat16)).long()]).float()
        finals = torch.unique(L._bf16_from_codes(lut_codes).float())
        finals = finals[torch.isfinite(finals)]
        codes = L._codes_from_bf16(finals.to(torch.bfloat16)).long()
        zl = f.zlo[codes]
        fin = torch.isfinite(zl)
        below = torch.nextafter(zl[fin], torch.full_like(zl[fin], -math.inf))
        # largest fp32 strictly below zlo(F) already maps under F; zlo(F) itself (the RNE midpoint) may reach F
        self.assertTrue(bool((final_of(below) < finals[fin]).all()))
        gen = torch.Generator().manual_seed(1)
        z = torch.randn(400000, generator=gen) * 40.0
        fz = final_of(z)
        for fv in (-29.5, -3.0, 0.0, 1.0, 7.25, 18.0, 25.5, 29.875):
            code = int(L._codes_from_bf16(torch.tensor([fv]).to(torch.bfloat16))[0])
            fb = float(L._bf16_from_codes(torch.tensor([code]))[0].float())
            sel = z < f.zlo[code]
            self.assertTrue(bool((fz[sel] < fb).all()), fv)

    def test_zlo_nan_and_neg_inf_keep_everything(self):
        f = fix()
        self.assertEqual(float(f.zlo[0xFF80]), -math.inf)  # F = -inf
        self.assertEqual(float(f.zlo[0x7FC0]), -math.inf)  # F = NaN


class TestPrecompute(unittest.TestCase):
    def test_probe_rounding_delta_sound(self):
        f = fix()
        true = torch.linalg.vector_norm(f.weff - f.wt.double(), dim=1)
        self.assertTrue(bool((f.delta >= true).all()))
        self.assertLess(f.n_amb, f.wt.numel() // 2)

    def test_side_data_shapes_and_rounding(self):
        f = fix()
        sd = f.sd
        self.assertEqual(sd.vmat.shape, (f.K, L.RANK))
        self.assertEqual(sd.chat.shape, (sd.n_tiles * L.TILE, L.RANK))
        self.assertEqual(sd.n_tiles, f.N // L.TILE)
        orth = (sd.vmat.double().T @ sd.vmat.double() - torch.eye(L.RANK, dtype=torch.float64)).abs().max()
        self.assertLess(float(orth), 1e-5)
        v64 = sd.vmat.double()
        rho = f.wt.double() - sd.chat[: f.N].double() @ v64.T
        self.assertTrue(bool((sd.aux[: f.N, 1].double() >= torch.linalg.vector_norm(rho, dim=1)).all()))
        self.assertTrue(bool((sd.aux[: f.N, 0].double() >= torch.linalg.vector_norm(rho @ v64, dim=1)).all()))
        self.assertGreaterEqual(sd.wmax, float(torch.linalg.vector_norm(f.weff[: f.ORG], dim=1).max()))
        self.assertLess(sd.resid_frac, 0.9)

    def test_up_down_rounding(self):
        v = torch.tensor([1.0 + 2.0 ** -40, -1.0 - 2.0 ** -40, 3.0, -0.1], dtype=torch.float64)
        self.assertTrue(bool((L.up_f32(v).double() >= v).all()))
        self.assertTrue(bool((L.down_f32(v).double() <= v).all()))


class TestExactness(unittest.TestCase):
    def _bound_audit(self, x, residual_term=True):
        f = fix()
        rb = L.reference_rowbuf(x, f.sd, f.eps, f.coef)
        ub = L.reference_bounds(rb, f.sd, residual_term)
        exact = x.double() @ f.weff.T
        exact = torch.cat([exact, torch.full((x.shape[0], f.sd.n_tiles * L.TILE - f.N), -math.inf,
                                             dtype=torch.float64)], 1)
        exact[:, f.ORG :] = -math.inf
        tmax = exact.view(x.shape[0], f.sd.n_tiles, L.TILE).amax(2)
        return ub.double(), tmax

    def test_bound_dominates_exact_tile_max(self):
        f = fix()
        x = torch.cat([queries(f.wt, 64, "peaked", c, 40.0, 10 + i, f.ORG) for i, c in enumerate((0.9, 0.6, 0.35))]
                      + [queries(f.wt, 64, "isotropic", 0.0, 80.0, 20, f.ORG)])
        ub, tmax = self._bound_audit(x)
        self.assertTrue(bool((ub >= tmax).all()))

    def test_negative_control_unsound_bound_is_caught(self):
        f = fix()
        x = queries(f.wt, 128, "peaked", 0.6, 40.0, 30, f.ORG)
        ub, tmax = self._bound_audit(x, residual_term=False)
        self.assertTrue(bool((ub < tmax).any()), "dropping ||y|| r_v must violate the bound somewhere")

    def _regime(self, x, m):
        f = fix()
        final = L.dense_final(x, f.b, f.g, f.lut)
        dense = final[:, : f.ORG].argmax(1)  # kernel stops at org_vocab; torch argmax = lowest-index tie-break
        n = x.shape[0]
        skipped = []
        ok = 0
        for a in range(0, n, m):
            xb, fb = x[a : a + m], final[a : a + m]
            tok, keep, _, val1 = L.reference_pruned_argmax(xb, fb, f.sd, f.zlo, f.eps, f.coef)
            ok += int((tok == dense[a : a + m]).sum())
            # audit: every skipped tile strictly below each row's dense max (so it cannot win or tie)
            col_keep = keep.repeat_interleave(L.TILE)[: f.N]
            col_keep[f.ORG :] = True
            sk = fb[:, ~col_keep]
            if sk.numel():
                self.assertTrue(bool((sk.amax(1) < fb.amax(1)).all()))
            self.assertTrue(bool((val1 <= fb.amax(1)).all()))
            skipped.append(1.0 - keep.float().mean().item())
        return ok, n, sum(skipped) / len(skipped)

    def test_500_vector_exact_argmax_all_regimes(self):
        f = fix()
        n = int(os.environ.get("LMMIPS_TEST_N", "500"))
        regimes = [("peaked cos=0.9", "peaked", 0.9), ("peaked cos=0.6", "peaked", 0.6),
                   ("peaked cos=0.35", "peaked", 0.35), ("tie-stress", "tie", 0.6), ("isotropic", "isotropic", 0.0)]
        report = {}
        for i, (name, mode, cos) in enumerate(regimes):
            x = queries(f.wt, n, mode, cos, 40.0, 100 + i, f.ORG)
            for m in (16, 1):
                xs = x if m == 16 else x[: max(64, n // 5)]
                ok, tot, skip = self._regime(xs, m)
                self.assertEqual(ok, tot, f"{name} M={m}: {ok}/{tot} exact argmax")
                report[(name, m)] = skip
        print("\nH2 CPU mirror: 100.0% exact argmax; tiles skipped per call (M=16 group / M=1):")
        for name, _, _ in regimes:
            print(f"  {name:16s} {100 * report[(name, 16)]:5.1f}% / {100 * report[(name, 1)]:5.1f}%")
        self.assertGreater(report[("peaked cos=0.9", 1)], 0.5)

    def test_nonfinite_query_keeps_every_tile(self):
        f = fix()
        x = queries(f.wt, 4, "peaked", 0.9, 40.0, 7, f.ORG)
        x[2, 5] = float("nan")
        final = L.dense_final(x, f.b, f.g, f.lut)
        _, keep, _, _ = L.reference_pruned_argmax(x, final, f.sd, f.zlo, f.eps, f.coef)
        self.assertTrue(bool(keep.all()))


def _synthetic_tma4_stream(grid: int, n_tiles: int, stages_per_tile: int, seed: int):
    """A fixed-grid TMA4 stream in the encoder's layout plus the per-tile entry states from a scalar, line-by-line
    transliteration of the CUDA tma_decode (random words; each warp's stored count is what it consumed)."""
    g = torch.Generator().manual_seed(seed)
    threads, warps = 256, 8
    f = torch.randint(1, 60, (40,), generator=g)
    f = (f * 1024 // f.sum()).clamp(min=1)
    f[0] += 1024 - int(f.sum())
    lut = []
    for s, fs in enumerate(f.tolist()):
        lut += [s | (fs << 6) | (i << 17) for i in range(fs)]
    tiles_of = [list(range(b, n_tiles, grid)) for b in range(grid)]
    n_sbs = [len(t) * stages_per_tile // 4 for t in tiles_of]
    max_sbs = max(n_sbs)
    sb_row = [sum(1 for b in range(grid) if n_sbs[b] > r) for r in range(max_sbs)]
    sb_row = [sum(sb_row[:r]) for r in range(max_sbs)]
    total_sbs = sum(n_sbs)
    x = [[int(v) for v in torch.randint(1 << 16, 1 << 31, (threads,), generator=g)] for _ in range(grid)]
    head = [list(r) for r in x]
    expect = {}
    blocks = [None] * total_sbs
    for b in range(grid):
        for s in range(n_sbs[b]):
            if s % (stages_per_tile // 4) == 0:
                expect[tiles_of[b][s // (stages_per_tile // 4)]] = list(x[b])
            pool = [int(v) for v in torch.randint(0, 65536, (4096,), generator=g)]
            used = []
            for w in range(warps):
                start, pos = sum(used), 0
                for _ in range(16):
                    need = []
                    for lane in range(32):
                        t = 32 * w + lane
                        e = lut[x[b][t] & 1023]
                        x[b][t] = ((e >> 6) & 2047) * (x[b][t] >> 10) + (e >> 17)
                        need.append(x[b][t] < 65536)
                    for lane in range(32):
                        if need[lane]:
                            t = 32 * w + lane
                            x[b][t] = (x[b][t] << 16) | pool[start + pos + sum(need[:lane])]
                    pos += sum(need)
                used.append(pos)
            body = used[: warps - 1] + [0] + pool[: sum(used)]
            body += [0] * (-len(body) % 8)
            blocks[sb_row[s] + b] = body
    off16 = [0]
    for body in blocks:
        off16.append(off16[-1] + len(body) // 8)
    meta_u32 = ((4 + grid + max_sbs + 3) // 4) * 4
    u32 = lut + [total_sbs, grid, max_sbs, threads] + [n * 4 for n in n_sbs] + sb_row
    u32 += [0] * (1024 + meta_u32 - len(u32)) + off16
    u32 += [0] * (-len(u32) % 4) + [v for r in head for v in r]
    words = [w for body in blocks for w in body]
    coded = torch.cat([torch.tensor(u32, dtype=torch.int64).to(torch.int32).view(torch.uint8),
                       torch.tensor(words, dtype=torch.int32).to(torch.int16).view(torch.uint8)])
    coded = torch.cat([coded, torch.zeros(-coded.numel() % 32, dtype=torch.uint8)]).view(-1, 32)
    ref = torch.tensor([expect[t] for t in range(n_tiles)], dtype=torch.int64)
    return coded, torch.where(ref >= 2**31, ref - 2**32, ref).to(torch.int32)


class TestTileStates(unittest.TestCase):
    def test_replay_matches_scalar_decoder(self):
        coded, ref = _synthetic_tma4_stream(grid=3, n_tiles=7, stages_per_tile=8, seed=7)
        got = L.tma4_tile_states(coded, n_tiles=7, stages_per_tile=8)
        self.assertTrue(torch.equal(got, ref))

    def test_replay_rejects_corrupt_word_counts(self):
        coded, _ = _synthetic_tma4_stream(grid=3, n_tiles=7, stages_per_tile=8, seed=7)
        raw = coded.reshape(-1).clone()
        hdr = 4096 + ((4 + 3 + 6 + 3) // 4) * 16 + ((14 + 1) * 4 + 15) // 16 * 16 + 3 * 256 * 4
        raw[hdr] ^= 1  # first super-block, warp-0 word count
        with self.assertRaises(ValueError):
            L.tma4_tile_states(raw.view(-1, 32), n_tiles=7, stages_per_tile=8)


class TestConfig(unittest.TestCase):
    def test_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MUSE_LM_MIPS", None)
            self.assertFalse(L.enabled())
        with mock.patch.dict(os.environ, {"MUSE_LM_MIPS": "1"}):
            self.assertTrue(L.enabled())
        with mock.patch.dict(os.environ, {"MUSE_LM_MIPS": "0"}):
            self.assertFalse(L.enabled())

    def test_eps_floor_at_least_0_10(self):
        with mock.patch.dict(os.environ, {"MUSE_LM_MIPS_EPS": "0.01"}):
            self.assertGreaterEqual(L.eps_floor(), 0.10)
        with mock.patch.dict(os.environ, {"MUSE_LM_MIPS_EPS": "0.25"}):
            self.assertEqual(L.eps_floor(), 0.25)

    def test_err_coef_covers_fp32_gamma(self):
        k = 6656
        self.assertGreater(L.err_coef(k), k * 2.0 ** -24)


if __name__ == "__main__":
    unittest.main()
