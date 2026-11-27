"""Bit-level models of decode- and prefill-path kernel edits, checked against the code they replace.

Pure Python, no GPU. Each test transcribes the old and the new instruction sequence of one edit and asserts that
they produce the same 32-bit words (or the same select state) on every input the kernel can see:

  * muserans/csrc/muse_rans.cu (par4 main loop and fallback path): the FP8 exponent nibbles are pre-shifted into
    bits 3..6 of each byte before `__byte_perm`, instead of shifting the permuted word left by 3 afterwards.
  * musebfx/bf16_rans_mma.cu (br_pair_mma_gemm_kernel and br_pair_unpack_kernel): the BF16 pair is assembled as
    `lop3(dup(sm), 0x807F807F, perm(e_lo, e_hi) << 7, 0xEA)` instead of the sign/mantissa shift-and-or sequence.
  * musebfx/draft_head_fast.cu (dhf_select_topk_kernel): the serial `tid == 0` scan for the radix bucket that holds
    the `need`-th largest key is replaced by an 8-warp inclusive prefix sum over s_hist[255..0].
  * norm_glue.cu, prefill_attn.cu, marlinw_tma.cu, swiglu.cuh: the [-448, 448] clamp in front of the saturating
    `cvt.rn.satfinite.e4m3x2.f32` is dropped (NaN operands reach the converter unchanged).

Run: python3 -m unittest discover -s harness/vllm_resident -p test_decode_unpack_exact.py -v
"""

import math
import random
import struct
import unittest

M32 = 0xFFFFFFFF


def byte_perm(x: int, y: int, s: int) -> int:
    """CUDA __byte_perm(x, y, s) for selectors whose nibbles are 0..7 (no sign-replicate mode)."""
    src = (x & M32) | ((y & M32) << 32)
    out = 0
    for i in range(4):
        sel = (s >> (4 * i)) & 0x7
        out |= ((src >> (8 * sel)) & 0xFF) << (8 * i)
    return out


def lop3_ea(a: int, b: int, c: int) -> int:
    """lop3.b32 with immediate 0xEA = (a & b) | c."""
    return ((a & b) | c) & M32


# --- muse_rans.cu: FP8 e4m3x4 exponent assembly -----------------------------------------------------------------


def rans_exp_old(pw: int) -> tuple[int, int]:
    hi_e = (pw >> 4) & 0x0F0F0F0F
    lo_e = pw & 0x0F0F0F0F
    e0 = byte_perm(hi_e, lo_e, 0x5140)
    e1 = byte_perm(hi_e, lo_e, 0x7362)
    return (e0 << 3) & M32, (e1 << 3) & M32


def rans_exp_new(pw: int) -> tuple[int, int]:
    hi_e3 = (pw >> 1) & 0x78787878
    lo_e3 = (pw << 3) & 0x78787878
    return byte_perm(hi_e3, lo_e3, 0x5140), byte_perm(hi_e3, lo_e3, 0x7362)


# --- bf16_rans_mma.cu: BF16 pair assembly ------------------------------------------------------------------------


def bf16_pair_old(sm_w: int, e_lo: int, e_hi: int, upper: bool) -> int:
    sm = byte_perm(sm_w, 0, 0x4342 if upper else 0x4140)
    return (((sm & 0x00800080) << 8) | (sm & 0x007F007F) | (e_lo << 7) | (e_hi << 23)) & M32


def bf16_pair_new(sm_w: int, e_lo: int, e_hi: int, upper: bool) -> int:
    u_dup = byte_perm(sm_w, 0, 0x3322 if upper else 0x1100)
    e_sh7 = (byte_perm(e_lo, e_hi, 0x5410) << 7) & M32
    return lop3_ea(u_dup, 0x807F807F, e_sh7)


# --- draft_head_fast.cu: radix-select bucket scan ----------------------------------------------------------------


def select_serial(hist: list[int], need: int, pref: int, shift: int) -> tuple[int, int]:
    """Old tid == 0 scan. Returns (s_prefix, s_rem_k) as written to shared memory."""
    acc = 0
    chosen = 0
    rem_k = need
    for b in range(255, -1, -1):
        c = hist[b]
        if acc + c >= need:
            chosen = b
            rem_k = need - acc
            break
        acc += c
    return pref | (chosen << shift), rem_k


def select_parallel(hist: list[int], need: int, pref: int, shift: int) -> tuple[int, int]:
    """New 8-warp reverse prefix sum; every thread's write is applied (the predicates make at most one fire)."""
    c = [hist[255 - tid] for tid in range(256)]
    winc = [0] * 256
    wsum = [0] * 8
    for w in range(8):
        run = 0
        for lane in range(32):
            run += c[32 * w + lane]
            winc[32 * w + lane] = run
        wsum[w] = run
    wbase = [sum(wsum[:w]) for w in range(8)]
    s_prefix, s_rem_k = None, need  # s_prefix keeps its previous value unless written; check below
    writes = 0
    for tid in range(256):
        inc = winc[tid] + wbase[tid >> 5]
        exc = inc - c[tid]
        if need == 0:
            if tid == 0:
                s_prefix, s_rem_k = pref | (255 << shift), 0
                writes += 1
        elif exc < need <= inc:
            s_prefix, s_rem_k = pref | ((255 - tid) << shift), need - exc
            writes += 1
        elif tid == 255 and inc < need:
            s_prefix = pref
            writes += 1
    if writes != 1:
        raise AssertionError(f"{writes} threads wrote the select state")
    return s_prefix, s_rem_k


E4M3_MAX = 448.0


def e4m3_satfinite(x: float) -> int:
    """cvt.rn.satfinite.e4m3 of one finite or infinite FP32 value (NaN handled by the callers): RNE, +-448 clamp."""
    sign = 0x80 if (x < 0 or (x == 0 and math.copysign(1.0, x) < 0)) else 0
    a = abs(x)
    if a >= E4M3_MAX:  # includes inf; values in (448, 464) also round to 448 under RNE
        return sign | 0x7E
    if a < 2.0 ** -6:  # subnormal: step 2^-9
        q = a / 2.0 ** -9
        n = math.floor(q)
        r = q - n
        n = n + 1 if (r > 0.5 or (r == 0.5 and n % 2 == 1)) else n
        return sign | n  # n == 8 lands exactly on the smallest normal 0x08
    e = math.floor(math.log2(a))
    while 2.0 ** e > a:
        e -= 1
    while 2.0 ** (e + 1) <= a:
        e += 1
    q = a / 2.0 ** e * 8.0 - 8.0  # mantissa in eighths
    n = math.floor(q)
    r = q - n
    n = n + 1 if (r > 0.5 or (r == 0.5 and n % 2 == 1)) else n
    if n == 8:
        n, e = 0, e + 1
    if e > 8 or (e == 8 and n > 6):
        return sign | 0x7E
    return sign | ((e + 7) << 3) | n


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


class TestE4m3SaturationFold(unittest.TestCase):
    """63759117: clamps to [-448, 448] ahead of a satfinite E4M3 conversion are dropped (NaN handling preserved)."""

    def values(self):
        rng = random.Random(4)
        edge = [0.0, -0.0, 2.0 ** -10, 2.0 ** -9, 1.5 * 2.0 ** -9, 2.0 ** -6, 415.9, 416.0, 432.0, 440.0, 447.9,
                448.0, 448.1, 463.9, 464.0, 464.1, 480.0, 1e30, math.inf]
        vals = edge + [-v for v in edge]
        for _ in range(20000):
            vals.append(f32(rng.uniform(-600.0, 600.0)))
            vals.append(f32(math.copysign(2.0 ** rng.uniform(-12, 12), rng.random() - 0.5)))
            bits = rng.getrandbits(32)
            v = struct.unpack("<f", struct.pack("<I", bits))[0]
            if not math.isnan(v):
                vals.append(v)
        return vals

    def test_ordered_clamp_matches_satfinite(self):
        # norm_glue clamp448 / marlinw epilogues: non-NaN values clamped to [-448, 448], NaN forwarded unchanged.
        for v in self.values():
            clamped = -E4M3_MAX if v < -E4M3_MAX else (E4M3_MAX if v > E4M3_MAX else v)
            self.assertEqual(e4m3_satfinite(clamped), e4m3_satfinite(v), repr(v))

    def test_encoder_reference_points(self):
        self.assertEqual(e4m3_satfinite(448.0), 0x7E)
        self.assertEqual(e4m3_satfinite(-448.0), 0xFE)
        self.assertEqual(e4m3_satfinite(1.0), 0x38)
        self.assertEqual(e4m3_satfinite(2.0 ** -9), 0x01)
        self.assertEqual(e4m3_satfinite(2.0 ** -6), 0x08)
        self.assertEqual(e4m3_satfinite(1.0625), 0x38)  # tie -> even mantissa 0
        self.assertEqual(e4m3_satfinite(1.1875), 0x3A)  # tie -> even mantissa 2


class TestRansExponentPreshift(unittest.TestCase):
    def test_every_byte_value_in_every_lane(self):
        for lane in range(4):
            for v in range(256):
                pw = v << (8 * lane)
                self.assertEqual(rans_exp_old(pw), rans_exp_new(pw), hex(pw))

    def test_every_adjacent_byte_pair(self):
        # Pre-shifting crosses byte boundaries (>> 1, << 3) before masking; all 2^16 neighbour pairs cover it.
        for lane in range(3):
            for v in range(1 << 16):
                pw = v << (8 * lane)
                self.assertEqual(rans_exp_old(pw), rans_exp_new(pw), hex(pw))

    def test_random_words(self):
        rng = random.Random(0x5EED)
        for _ in range(200_000):
            pw = rng.getrandbits(32)
            self.assertEqual(rans_exp_old(pw), rans_exp_new(pw), hex(pw))


class TestBf16PairLop3(unittest.TestCase):
    def test_all_sign_mantissa_pairs(self):
        rng = random.Random(1)
        for upper in (False, True):
            for v in range(1 << 16):
                sm_w = (v << 16) if upper else v
                sm_w |= rng.getrandbits(16) << (0 if upper else 16)  # the other half must not leak in
                e_lo, e_hi = rng.randrange(256), rng.randrange(256)
                self.assertEqual(bf16_pair_old(sm_w, e_lo, e_hi, upper), bf16_pair_new(sm_w, e_lo, e_hi, upper))

    def test_all_exponent_pairs(self):
        rng = random.Random(2)
        for e_lo in range(256):
            for e_hi in range(256):
                sm_w = rng.getrandbits(32)
                for upper in (False, True):
                    self.assertEqual(bf16_pair_old(sm_w, e_lo, e_hi, upper), bf16_pair_new(sm_w, e_lo, e_hi, upper))


class TestRadixSelectScan(unittest.TestCase):
    def check(self, hist, need, pref=0x12000000, shift=16):
        self.assertEqual(select_serial(hist, need, pref, shift), select_parallel(hist, need, pref, shift), (need,))

    def test_edges(self):
        zero = [0] * 256
        self.check(zero, 0)
        self.check(zero, 5)  # total < need: prefix unchanged, rem_k unchanged
        for b in (0, 31, 32, 128, 224, 255):
            h = [0] * 256
            h[b] = 7
            for need in (0, 1, 6, 7, 8):
                self.check(h, need)
        flat = [1] * 256
        for need in (0, 1, 2, 31, 32, 33, 255, 256, 257):
            self.check(flat, need)

    def test_random_histograms(self):
        rng = random.Random(3)
        for _ in range(3000):
            density = rng.choice((0.02, 0.2, 1.0))
            hist = [rng.randrange(0, 40) if rng.random() < density else 0 for _ in range(256)]
            total = sum(hist)
            need = rng.choice((0, 1, rng.randrange(0, total + 2), total, total + 1))
            self.check(hist, need, shift=rng.choice((0, 8, 16, 24)))


if __name__ == "__main__":
    unittest.main()
