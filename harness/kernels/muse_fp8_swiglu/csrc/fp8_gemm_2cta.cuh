// SPDX-License-Identifier: Apache-2.0
// Clean-room GB10 (sm_121a) 0-ULP M=512 prefill TMA kernels (0 register spills):
//   - swiglu_ws9_st3_lut_kernel: 9-warp ST=3 SwiGLU with LUT SiLU, PRMT 4x4 byte transpose, and dual-sided SM121 PDL
//   - swiglu_ws8_256x128_st2_kernel: 8-warp BM=256, BN=128, ST=2 (4.00 QMMA/LDSM, 13 exact 48-SM waves)
//   - fp8_6656_ws8_256_st2_st4_kernel: 8-warp BM=256 ST1=2 (4.00 QMMA/LDSM) + BM=128 ST2=4 Phase-2 for N=6656
//   - fp8_dual_a_ws8_256_kernel: 8-warp BM=256 ST=2 3-wave seamless FP8 qkv+gate dual GEMM (4.00 QMMA/LDSM)
//   - bf16_dual_ws8_256_kernel: 8-warp BM=256 ST=2 3-wave seamless BF16 qkv+gate dual GEMM (4.00 HMMA/LDSM)
#pragma once

namespace swiglu_ws {

__device__ __forceinline__ uint32_t cvt2_scaled_bf16(float lo, float hi, float sa, float sb) {
  const float x = __fmul_rn(__fmul_rn(lo, sb), sa);
  const float y = __fmul_rn(__fmul_rn(hi, sb), sa);
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(x), "f"(y));
  return r;
}

__device__ __forceinline__ uint32_t cvt2_unscaled_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}

__device__ __forceinline__ void mma_bf16_k16(const uint32_t (&a)[4], uint32_t b0, uint32_t b1, float (&c)[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

template <bool IsBf16>
__device__ __forceinline__ void mma_op(const uint32_t (&a)[4], uint32_t b0, uint32_t b1, float (&c)[4]) {
  if constexpr (IsBf16) mma_bf16_k16(a, b0, b1, c);
  else                  mma_fp8(a, b0, b1, c);
}

template <bool kLutSilu>
__device__ __forceinline__ float eval_pair_opt(float g_acc, float u_acc, float sa, float sb, float inv_down) {
  const float g_f = __fmul_rn(__fmul_rn(g_acc, sb), sa);
  const float u_f = __fmul_rn(__fmul_rn(u_acc, sb), sa);
  uint32_t packed_bf16;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(packed_bf16) : "f"(g_f), "f"(u_f));
  const float u_bf = __uint_as_float(packed_bf16 & 0xffff0000u);
  float silu;
  if constexpr (kLutSilu) {
    silu = __ldg(bf16_silu + (packed_bf16 & 0xffffu));
  } else {
    silu = swiglu_silu(__uint_as_float(packed_bf16 << 16));
  }
  return __fmul_rn(__fmul_rn(silu, u_bf), inv_down);
}

__device__ __forceinline__ uint32_t transpose4x4_bytes(uint32_t pack, int q) {
  const uint32_t sel1 = (q & 1) ? 0x3715u : 0x6240u;
  const uint32_t sel2 = (q & 2) ? 0x3276u : 0x5410u;
  const uint32_t u = __shfl_xor_sync(0xffffffffu, pack, 1);
  const uint32_t x = __byte_perm(pack, u, sel1);
  const uint32_t v = __shfl_xor_sync(0xffffffffu, x, 2);
  return __byte_perm(x, v, sel2);
}

template <int NW_CUR, bool kLutSilu>
__device__ __forceinline__ void epilogue_swiglu_prmt(const float (&acc)[NW_CUR][4], uint8_t* out, int row0, int out_col0,
                                                     int lane, float sa, float sb, float inv_down) {
  static_assert(NW_CUR == 4 || NW_CUR == 8, "NW_CUR must be 4 or 8");
  const int q = lane & 3, rq = lane >> 2;
  uint32_t* out32 = reinterpret_cast<uint32_t*>(out);
#pragma unroll
  for (int half = 0; half < NW_CUR / 4; ++half) {
    const int j0 = 4 * half;
    uint16_t conv[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const float q_lo = eval_pair_opt<kLutSilu>(acc[j0 + m][0], acc[j0 + m][1], sa, sb, inv_down);
      const float q_hi = eval_pair_opt<kLutSilu>(acc[j0 + m][2], acc[j0 + m][3], sa, sb, inv_down);
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(conv[m]) : "f"(q_hi), "f"(q_lo));
    }
    const uint32_t c01 = static_cast<uint32_t>(conv[0]) | (static_cast<uint32_t>(conv[1]) << 16);
    const uint32_t c23 = static_cast<uint32_t>(conv[2]) | (static_cast<uint32_t>(conv[3]) << 16);
    const uint32_t w_lo = transpose4x4_bytes(__byte_perm(c01, c23, 0x6420u), q);
    const uint32_t w_hi = transpose4x4_bytes(__byte_perm(c01, c23, 0x7531u), q);
    out32[((size_t)(row0 + rq) * OUT_COLS + out_col0) / 4 + 4 * half + q] = w_lo;
    out32[((size_t)(row0 + rq + 8) * OUT_COLS + out_col0) / 4 + 4 * half + q] = w_hi;
  }
}

template <int N_OUT, int NW_CUR, bool IsScaled = true>
__device__ __forceinline__ void epilogue_bf16_sel4(const float (&acc)[NW_CUR][4], __nv_bfloat16* out,
                                                   int row0, int col0, int lane, float sa, float sb) {
  const int q = lane & 3, rq = lane >> 2;
#pragma unroll
  for (int pp = 0; pp < NW_CUR / 2; ++pp) {
    const int j = 2 * pp;
    const uint32_t v0 = IsScaled ? cvt2_scaled_bf16(acc[j][0], acc[j][1], sa, sb) : cvt2_unscaled_bf16(acc[j][0], acc[j][1]);
    const uint32_t v1 = IsScaled ? cvt2_scaled_bf16(acc[j][2], acc[j][3], sa, sb) : cvt2_unscaled_bf16(acc[j][2], acc[j][3]);
    const uint32_t v2 = IsScaled ? cvt2_scaled_bf16(acc[j + 1][0], acc[j + 1][1], sa, sb) : cvt2_unscaled_bf16(acc[j + 1][0], acc[j + 1][1]);
    const uint32_t v3 = IsScaled ? cvt2_scaled_bf16(acc[j + 1][2], acc[j + 1][3], sa, sb) : cvt2_unscaled_bf16(acc[j + 1][2], acc[j + 1][3]);
    const uint32_t r1 = __shfl_xor_sync(0xffffffffu, sel4(v0, v1, v2, v3, q ^ 1), 1);
    const uint32_t r2 = __shfl_xor_sync(0xffffffffu, sel4(v0, v1, v2, v3, q ^ 2), 2);
    const uint32_t r3 = __shfl_xor_sync(0xffffffffu, sel4(v0, v1, v2, v3, q ^ 3), 3);
    const uint32_t own = sel4(v0, v1, v2, v3, q);
    uint4 w;
    w.x = sel4(own, r1, r2, r3, 0 ^ q);
    w.y = sel4(own, r1, r2, r3, 1 ^ q);
    w.z = sel4(own, r1, r2, r3, 2 ^ q);
    w.w = sel4(own, r1, r2, r3, 3 ^ q);
    const int row = row0 + rq + (q & 1) * 8, col = col0 + (j + (q >> 1)) * 8;
    *reinterpret_cast<uint4*>(out + (size_t)row * N_OUT + col) = w;
  }
}

template <int NW_CUR, bool IsBf16 = false>
__device__ __forceinline__ void consume_stage_qpipe_2xN(
    uint32_t abase, uint32_t wbase, int wm, int wn, int ar, int ach, int br, int bch, float (&acc)[2][NW_CUR][4]) {
  static_assert(NW_CUR % 2 == 0, "NW_CUR must be even");
  constexpr int NP = NW_CUR / 2;
  auto load_a_i = [&](int ks, int i, uint32_t (&ai)[4]) {
    const int r = wm * 32 + i * 16 + ar;
    ldsm4(abase + r * KT + swz(r, 2 * ks + ach) * 16, ai);
  };
  auto load_b_p = [&](int ks, int p, uint32_t (&bp)[4]) {
    const int r = wn * NW_CUR * 8 + p * 16 + br;
    ldsm4(wbase + r * KT + swz(r, 2 * ks + bch) * 16, bp);
  };
  uint32_t a[2][4], bbuf[2][4];
  load_a_i(0, 0, a[0]);
  load_a_i(0, 1, a[1]);
  load_b_p(0, 0, bbuf[0]);
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
    for (int p = 0; p < NP; ++p) {
      const int q = ks * NP + p;
      uint32_t (&bc)[4] = bbuf[q & 1];
      uint32_t (&bn)[4] = bbuf[(q + 1) & 1];
      if (p + 1 < NP)      load_b_p(ks, p + 1, bn);
      else if (ks + 1 < 4) load_b_p(ks + 1, 0, bn);
      mma_op<IsBf16>(a[0], bc[0], bc[1], acc[0][2 * p]);
      mma_op<IsBf16>(a[0], bc[2], bc[3], acc[0][2 * p + 1]);
      if (p == NP - 1 && ks + 1 < 4) load_a_i(ks + 1, 0, a[0]);
      mma_op<IsBf16>(a[1], bc[0], bc[1], acc[1][2 * p]);
      mma_op<IsBf16>(a[1], bc[2], bc[3], acc[1][2 * p + 1]);
      if (p == NP - 1 && ks + 1 < 4) load_a_i(ks + 1, 1, a[1]);
    }
  }
}

template <bool IsBf16 = false>
__device__ __forceinline__ void consume_stage_pipe_4x8(
    uint32_t abase, uint32_t wbase, int wm, int wn, int ar, int ach, int br, int bch, float (&acc)[4][8][4]) {
  auto load_a_i = [&](int ks, int i, uint32_t (&ai)[4]) {
    const int r = wm * 64 + i * 16 + ar;
    ldsm4(abase + r * KT + swz(r, 2 * ks + ach) * 16, ai);
  };
  auto load_b_p = [&](int ks, int p, uint32_t (&bp)[4]) {
    const int r = wn * 64 + p * 16 + br;
    ldsm4(wbase + r * KT + swz(r, 2 * ks + bch) * 16, bp);
  };
  // 2-slot streaming-B ring (8 regs vs 24 regs) + single A buffer reloaded on jp==3 (16 regs vs 32 regs):
  // saves 32 operand regs/thread while keeping 4 QMMAs per B-pair LDSM and exact ascending-ks 0-ULP order.
  uint32_t a[4][4], bbuf[2][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) load_a_i(0, i, a[i]);
  load_b_p(0, 0, bbuf[0]);
#pragma unroll
  for (int ks = 0; ks < KT / 32; ++ks) {
#pragma unroll
    for (int jp = 0; jp < 4; ++jp) {
      const int q = (ks << 2) + jp;
      uint32_t (&bc)[4] = bbuf[q & 1];
      uint32_t (&bn)[4] = bbuf[(q + 1) & 1];
      if (jp + 1 < 4)            load_b_p(ks, jp + 1, bn);
      else if (ks + 1 < KT / 32) load_b_p(ks + 1, 0, bn);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        mma_op<IsBf16>(a[i], bc[0], bc[1], acc[i][2 * jp]);
        mma_op<IsBf16>(a[i], bc[2], bc[3], acc[i][2 * jp + 1]);
        if (jp == 3 && ks + 1 < KT / 32) load_a_i(ks + 1, i, a[i]);
      }
    }
  }
}

template <int MW, int NW_CUR, bool IsBf16 = false>
__device__ __forceinline__ void consume_stage_pipe_opt(
    uint32_t abase, uint32_t wbase, int wm, int wn, int ar, int ach, int br, int bch, float (&acc)[MW][NW_CUR][4]) {
  static_assert(NW_CUR % 2 == 0, "NW_CUR must be even");
  constexpr int NP = NW_CUR / 2;
  auto load_a_i = [&](int ks, int i, uint32_t (&ai)[4]) {
    const int r = wm * MW * 16 + i * 16 + ar;
    ldsm4(abase + r * KT + swz(r, 2 * ks + ach) * 16, ai);
  };
  auto load_b_p = [&](int ks, int p, uint32_t (&bp)[4]) {
    const int r = wn * NW_CUR * 8 + p * 16 + br;
    ldsm4(wbase + r * KT + swz(r, 2 * ks + bch) * 16, bp);
  };
  // Parity-indexed 2-slot streaming-B ring (handles odd NP=3 cleanly) + jp-outer 4-QMMA/LDSM.
  uint32_t a[MW][4], bbuf[2][4];
#pragma unroll
  for (int i = 0; i < MW; ++i) load_a_i(0, i, a[i]);
  load_b_p(0, 0, bbuf[0]);
#pragma unroll
  for (int ks = 0; ks < KT / 32; ++ks) {
#pragma unroll
    for (int jp = 0; jp < NP; ++jp) {
      const int q = ks * NP + jp;
      uint32_t (&bc)[4] = bbuf[q & 1];
      uint32_t (&bn)[4] = bbuf[(q + 1) & 1];
      if (jp + 1 < NP)           load_b_p(ks, jp + 1, bn);
      else if (ks + 1 < KT / 32) load_b_p(ks + 1, 0, bn);
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        mma_op<IsBf16>(a[i], bc[0], bc[1], acc[i][2 * jp]);
        mma_op<IsBf16>(a[i], bc[2], bc[3], acc[i][2 * jp + 1]);
        if (jp == NP - 1 && ks + 1 < KT / 32) load_a_i(ks + 1, i, a[i]);
      }
    }
  }
}

template <bool kPdlPrologue, bool kPdlLaunch, bool kLutSilu>
__global__ void __launch_bounds__(NT, 1)
swiglu_ws8_256x128_st2_kernel(const __grid_constant__ CUtensorMap tmA256, const __grid_constant__ CUtensorMap tmB128,
                              const float* sa_p, const float* sb_p, const float* down_p, uint8_t* out) {
  constexpr int BM_T = 256, ST_T = 2, ASZ = BM_T * KT, STB = ASZ + WSZ, NKT = K_DIM / KT, TOTAL = 13 * NKT;
  constexpr int kPdlTrig = TOTAL - 12;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST_T];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST_T;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST_T; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();

  auto produce = [&](int gl) {
    const int s = gl & 1, n = gl >> 1;
    if (gl >= ST_T) {
      mbar_wait(empty0 + 8 * s, (n - 1) & 1);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    const int ti = gl / NKT, kt = gl - ti * NKT;
    const int t = b + ti * 48, nt = t >> 1, band = t & 1;
    const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
    if constexpr (kPdlPrologue) {
      if (gl == 1) {
        mbar_expect_tx(bar, ASZ);
        tma2d_hint(dst, &tmA256, kt * KT, band * BM_T, bar, tma_pol_el());
        return;
      }
    }
    mbar_expect_tx(bar, STB);
    tma2d_hint(dst, &tmA256, kt * KT, band * BM_T, bar, tma_pol_el());
    tma2d_hint(dst + ASZ, &tmB128, kt * KT, nt * BN, bar, tma_pol_ef());
  };

  if constexpr (kPdlPrologue) {
    const int nt0 = b >> 1, band0 = b & 1;
    if (tid == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA256) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB128) : "memory");
      mbar_expect_tx_only(full0, WSZ);
      tma2d_hint(sbase + ASZ, &tmB128, 0, nt0 * BN, full0, tma_pol_ef());
      mbar_expect_tx_only(full0 + 8, WSZ);
      tma2d_hint(sbase + STB + ASZ, &tmB128, KT, nt0 * BN, full0 + 8, tma_pol_ef());
    }
    pdl_wait();
    if (tid == 0) {
      mbar_expect_tx(full0, ASZ);
      tma2d_hint(sbase, &tmA256, 0, band0 * BM_T, full0, tma_pol_el());
    }
  } else if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA256) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB128) : "memory");
    produce(0);
  }

  const int wm = warp % WM, wn = warp / WM;
  const float sa = __ldg(sa_p), sb = __ldg(sb_p);
  float inv_down;
  asm("div.full.f32 %0, 0f3F800000, %1;" : "=f"(inv_down) : "f"(__ldg(down_p)));
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  float acc[4][8][4] = {};
  int kt = 0, ti = 0;
  for (int g = 0; g < TOTAL; ++g) {
    if constexpr (kPdlLaunch) {
      if (g == kPdlTrig) pdl_launch_dependents();
    }
    const int s = g & 1, n = g >> 1;
    if (tid == 0 && g + 1 < TOTAL) produce(g + 1);
    mbar_wait(full0 + 8 * s, n & 1);
    const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
    consume_stage_pipe_4x8<false>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * s);
    if (++kt == NKT) {
      const int t = b + ti * 48, nt = t >> 1, band = t & 1;
      kt = 0; ++ti;
      const int rb0 = band * BM_T + wm * 64;
      const int out_col0 = (nt * BN + wn * 64) / 2;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        epilogue_swiglu_prmt<8, kLutSilu>(acc[i], out, rb0 + i * 16, out_col0, lane, sa, sb, inv_down);
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
      }
    }
  }
}

template <bool kPdlPrologue, bool kPdlLaunch, bool kLutSilu>
__global__ void __launch_bounds__(NT_WS9, 1)
swiglu_ws9_st3_lut_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                          const float* sa_p, const float* sb_p, const float* down_p, uint8_t* out) {
  constexpr int BM_T = 128, ST_T = 3, ASZ = BM_T * KT, STB = ASZ + WSZ, NKT = K_DIM / KT;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST_T];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST_T;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST_T; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();

  if (warp == NWARP) {
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
      int s = 0, par = 0, g = 0;
      const uint64_t pol_el = tma_pol_el(), pol_ef = tma_pol_ef();
      const uint64_t pol_b = ((b & 3) < 3) ? tma_pol_b_shared() : pol_ef;
      if constexpr (kPdlPrologue) {
        const int nt0 = b >> 2;
#pragma unroll
        for (int ps = 0; ps < ST_T; ++ps) {
          const uint32_t bar = full0 + 8 * ps, dst = sbase + ps * STB;
          mbar_expect_tx_only(bar, WSZ);
          tma2d_hint(dst + ASZ, &tmB, ps * KT, nt0 * BN, bar, pol_b);
        }
        pdl_wait();
      }
      for (int ti = 0; ti < 26; ++ti) {
        const int t = b + ti * 48, nt = t >> 2, band = t & 3;
        const int row0 = band * BM_T, col0 = nt * BN;
        for (int kt = 0; kt < NKT; ++kt) {
          if (g >= ST_T) {
            mbar_wait(empty0 + 8 * s, par ^ 1);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          }
          const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
          if constexpr (kPdlPrologue) {
            if (g < ST_T) {
              mbar_expect_tx(bar, ASZ);
              tma2d_hint(dst, &tmA, kt * KT, row0, bar, pol_el);
              ++g;
              if (++s == ST_T) { s = 0; par ^= 1; }
              continue;
            }
          }
          mbar_expect_tx(bar, STB);
          tma2d_hint(dst, &tmA, kt * KT, row0, bar, pol_el);
          tma2d_hint(dst + ASZ, &tmB, kt * KT, col0, bar, pol_b);
          ++g;
          if (++s == ST_T) { s = 0; par ^= 1; }
        }
      }
    }
    return;
  }

  const int wm = warp % WM, wn = warp / WM;
  const float sa = __ldg(sa_p), sb = __ldg(sb_p);
  float inv_down;
  asm("div.full.f32 %0, 0f3F800000, %1;" : "=f"(inv_down) : "f"(__ldg(down_p)));
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  float acc[2][NW][4] = {};
  int s = 0, par = 0;
  for (int ti = 0; ti < 26; ++ti) {
    const int t = b + ti * 48, nt = t >> 2, band = t & 3;
    for (int kt = 0; kt < NKT; ++kt) {
      if constexpr (kPdlLaunch) { if (ti == 25 && kt == NKT - 12) pdl_launch_dependents(); }
      mbar_wait(full0 + 8 * s, par);
      const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
      consume_stage_pipe_sw(abase, wbase, wm, wn, ar, ach, br, bch, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++s == ST_T) { s = 0; par ^= 1; }
    }
    const int rb0 = band * BM_T + wm * 32;
    const int out_col0 = (nt * BN + wn * NW * 8) / 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      epilogue_swiglu_prmt<NW, kLutSilu>(acc[i], out, rb0 + i * 16, out_col0, lane, sa, sb, inv_down);
#pragma unroll
      for (int j = 0; j < NW; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
    }
  }
}

template <int K, bool kPdl>
__global__ void __launch_bounds__(NT, 1)
fp8_6656_ws8_256_st2_st4_kernel(const __grid_constant__ CUtensorMap tmA256,
                                const __grid_constant__ CUtensorMap tmA128,
                                const __grid_constant__ CUtensorMap tmB128,
                                const __grid_constant__ CUtensorMap tmB48,
                                const __grid_constant__ CUtensorMap tmB32,
                                const float* sa_p, const float* sb_p, __nv_bfloat16* out) {
  constexpr int N_OUT = 6656, ST1 = 2, BM1 = 256, ASZ1 = BM1 * KT, STB1 = ASZ1 + WSZ, NKT = K / KT, TOTAL1 = 2 * NKT;
  constexpr int ST2 = 4, ASZ2 = 128 * KT, WSZ2 = 48 * KT, STB2 = ASZ2 + WSZ2;
  static_assert(STB2 % 1024 == 0 && ST2 * STB2 + 1024 <= SMEM_MAX, "Phase-2 SWIZZLE_128B alignment");
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST2];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST2;
  const int wm = warp % WM, wn = warp / WM;
  const float sa = __ldg(sa_p), sb = __ldg(sb_p);
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;

  {
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < ST1; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    auto produce1 = [&](int gl) {
      const int s = gl & 1, n = gl >> 1;
      if (gl >= ST1) {
        mbar_wait(empty0 + 8 * s, (n - 1) & 1);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      }
      const int ti = gl / NKT, kt = gl - ti * NKT;
      const int t = b + ti * 48, nt = t >> 1, band = t & 1;
      const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB1;
      if constexpr (kPdl) {
        if (gl == 1) {
          mbar_expect_tx(bar, ASZ1);
          tma2d_hint(dst, &tmA256, kt * KT, band * BM1, bar, tma_pol_el());
          return;
        }
      }
      mbar_expect_tx(bar, STB1);
      tma2d_hint(dst, &tmA256, kt * KT, band * BM1, bar, tma_pol_el());
      tma2d_hint(dst + ASZ1, &tmB128, kt * KT, nt * BN, bar, tma_pol_ef());
    };
    if constexpr (kPdl) {
      const int nt0 = b >> 1, band0 = b & 1;
      if (tid == 0) {
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA256) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA128) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB128) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB48) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB32) : "memory");
        mbar_expect_tx_only(full0, WSZ);
        tma2d_hint(sbase + ASZ1, &tmB128, 0, nt0 * BN, full0, tma_pol_ef());
        mbar_expect_tx_only(full0 + 8, WSZ);
        tma2d_hint(sbase + STB1 + ASZ1, &tmB128, KT, nt0 * BN, full0 + 8, tma_pol_ef());
      }
      pdl_wait();
      if (tid == 0) {
        mbar_expect_tx(full0, ASZ1);
        tma2d_hint(sbase, &tmA256, 0, band0 * BM1, full0, tma_pol_el());
      }
    } else if (tid == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA256) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA128) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB128) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB48) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB32) : "memory");
      produce1(0);
    }
    float acc[4][8][4] = {};
    int kt = 0, ti = 0;
    for (int g = 0; g < TOTAL1; ++g) {
      const int s = g & 1, n = g >> 1;
      if (tid == 0 && g + 1 < TOTAL1) produce1(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB1, wbase = abase + ASZ1;
      consume_stage_pipe_4x8<false>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++kt == NKT) {
        const int t = b + ti * 48, nt = t >> 1, band = t & 1;
        kt = 0; ++ti;
        const int rb0 = band * BM1 + wm * 64;
        const int col0 = nt * BN + wn * 64;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          epilogue_bf16_sel4<N_OUT, 8, true>(acc[i], out, rb0 + i * 16, col0, lane, sa, sb);
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
        }
      }
    }
  }

  __syncthreads();

  {
    const int band2 = b & 3, c2 = b >> 2;
    const bool is48 = (c2 < 8);
    const int row0_2 = band2 * 128;
    const int col0_2 = is48 ? (6144 + c2 * 48) : (6528 + (c2 - 8) * 32);
    const int tx2 = ASZ2 + (is48 ? (48 * KT) : (32 * KT));
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < ST2; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    auto produce2 = [&](int gl) {
      const int s = gl & 3, n = gl >> 2;
      if (gl >= ST2) {
        mbar_wait(empty0 + 8 * s, (n - 1) & 1);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      }
      const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB2;
      mbar_expect_tx(bar, tx2);
      tma2d_hint(dst, &tmA128, gl * KT, row0_2, bar, tma_pol_el());
      if (is48) tma2d_hint(dst + ASZ2, &tmB48, gl * KT, col0_2, bar, tma_pol_el());
      else      tma2d_hint(dst + ASZ2, &tmB32, gl * KT, col0_2, bar, tma_pol_el());
    };
    if (tid == 0) {
      for (int s = 0; s < ST2 - 1 && s < NKT; ++s) produce2(s);
    }
    const int rb0 = row0_2 + warp * 16;
    constexpr int kPdlTrig = (K == 19968) ? (NKT - 20) : ((NKT > 10) ? (NKT - 10) : 0);
    if (is48) {
      float acc2[1][6][4] = {};
      for (int g = 0; g < NKT; ++g) {
        if constexpr (kPdl) { if (g == kPdlTrig) pdl_launch_dependents(); }
        const int s = g & 3, n = g >> 2;
        if (tid == 0 && g + ST2 - 1 < NKT) produce2(g + ST2 - 1);
        mbar_wait(full0 + 8 * s, n & 1);
        const uint32_t abase = sbase + s * STB2, wbase = abase + ASZ2;
        consume_stage_pipe_opt<1, 6, false>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
      }
      epilogue_bf16_sel4<N_OUT, 6, true>(acc2[0], out, rb0, col0_2, lane, sa, sb);
    } else {
      float acc2[1][4][4] = {};
      for (int g = 0; g < NKT; ++g) {
        if constexpr (kPdl) { if (g == kPdlTrig) pdl_launch_dependents(); }
        const int s = g & 3, n = g >> 2;
        if (tid == 0 && g + ST2 - 1 < NKT) produce2(g + ST2 - 1);
        mbar_wait(full0 + 8 * s, n & 1);
        const uint32_t abase = sbase + s * STB2, wbase = abase + ASZ2;
        consume_stage_pipe_opt<1, 4, false>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
      }
      epilogue_bf16_sel4<N_OUT, 4, true>(acc2[0], out, rb0, col0_2, lane, sa, sb);
    }
  }
}

template <bool kPdl>
__global__ void __launch_bounds__(NT, 1)
fp8_dual_a_ws8_256_kernel(const __grid_constant__ CUtensorMap tmA0,
                          const __grid_constant__ CUtensorMap tmA1,
                          const __grid_constant__ CUtensorMap tmB0,
                          const __grid_constant__ CUtensorMap tmB1_128,
                          const __grid_constant__ CUtensorMap tmB1_112,
                          const __grid_constant__ CUtensorMap tmB1_96,
                          const float* sa0_p, const float* sa1_p,
                          const float* sb0_p, const float* sb1_p,
                          __nv_bfloat16* out0, __nv_bfloat16* out1) {
  constexpr int N0 = 4608, N1 = 4096, K = 6656, NTN0 = 36, NKT = K / KT;
  constexpr int ST_T = 2, ASZ = 256 * KT, STB = ASZ + WSZ, TOTAL1 = 2 * NKT, TOTAL_ALL = 3 * NKT;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST_T];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST_T;

  const int band2 = b & 1, c2 = b >> 1;
  const bool is112 = (c2 < 16);
  const int row0_2 = band2 * 256;
  const int col0_2 = is112 ? (1536 + c2 * 112) : (3328 + (c2 - 16) * 96);
  const int tx2 = ASZ + (is112 ? (112 * KT) : (96 * KT));

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST_T; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();

  auto produce = [&](int gl) {
    const int s = gl & 1, n = gl >> 1;
    if (gl >= ST_T) {
      mbar_wait(empty0 + 8 * s, (n - 1) & 1);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
    if (gl < TOTAL1) {
      const int ti = gl / NKT, kt = gl - ti * NKT;
      const int t = b + ti * 48, nt = t >> 1, band = t & 1;
      if constexpr (kPdl) {
        if (gl == 1) {
          mbar_expect_tx(bar, ASZ);
          if (nt < NTN0) tma2d_hint(dst, &tmA0, kt * KT, band * 256, bar, tma_pol_el());
          else           tma2d_hint(dst, &tmA1, kt * KT, band * 256, bar, tma_pol_el());
          return;
        }
      }
      mbar_expect_tx(bar, STB);
      if (nt < NTN0) {
        tma2d_hint(dst, &tmA0, kt * KT, band * 256, bar, tma_pol_el());
        tma2d_hint(dst + ASZ, &tmB0, kt * KT, nt * BN, bar, tma_pol_ef());
      } else {
        tma2d_hint(dst, &tmA1, kt * KT, band * 256, bar, tma_pol_el());
        tma2d_hint(dst + ASZ, &tmB1_128, kt * KT, (nt - NTN0) * BN, bar, tma_pol_ef());
      }
    } else {
      const int kt = gl - TOTAL1;
      mbar_expect_tx(bar, tx2);
      tma2d_hint(dst, &tmA1, kt * KT, row0_2, bar, tma_pol_el());
      if (is112) tma2d_hint(dst + ASZ, &tmB1_112, kt * KT, col0_2, bar, tma_pol_ef());
      else       tma2d_hint(dst + ASZ, &tmB1_96,  kt * KT, col0_2, bar, tma_pol_ef());
    }
  };

  if constexpr (kPdl) {
    const int nt0 = b >> 1, band0 = b & 1;
    if (tid == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA0) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA1) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB0) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB1_128) : "memory");
      mbar_expect_tx_only(full0, WSZ);
      mbar_expect_tx_only(full0 + 8, WSZ);
      if (nt0 < NTN0) {
        tma2d_hint(sbase + ASZ, &tmB0, 0, nt0 * BN, full0, tma_pol_ef());
        tma2d_hint(sbase + STB + ASZ, &tmB0, KT, nt0 * BN, full0 + 8, tma_pol_ef());
      } else {
        tma2d_hint(sbase + ASZ, &tmB1_128, 0, (nt0 - NTN0) * BN, full0, tma_pol_ef());
        tma2d_hint(sbase + STB + ASZ, &tmB1_128, KT, (nt0 - NTN0) * BN, full0 + 8, tma_pol_ef());
      }
    }
    pdl_wait();
    if (tid == 0) {
      mbar_expect_tx(full0, ASZ);
      if (nt0 < NTN0) tma2d_hint(sbase, &tmA0, 0, band0 * 256, full0, tma_pol_el());
      else            tma2d_hint(sbase, &tmA1, 0, band0 * 256, full0, tma_pol_el());
    }
  } else if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA0) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA1) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB0) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB1_128) : "memory");
    produce(0);
  }

  const int wm = warp % WM, wn = warp / WM;
  const float sa0 = __ldg(sa0_p), sa1 = __ldg(sa1_p), sb0 = __ldg(sb0_p), sb1 = __ldg(sb1_p);
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;

  {
    float acc[4][8][4] = {};
    int kt = 0, ti = 0;
    for (int g = 0; g < TOTAL1; ++g) {
      const int s = g & 1, n = g >> 1;
      if (tid == 0) produce(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
      consume_stage_pipe_4x8<false>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++kt == NKT) {
        const int t = b + ti * 48, nt = t >> 1, band = t & 1;
        kt = 0; ++ti;
        const bool is1 = nt >= NTN0;
        const int rb0 = band * 256 + wm * 64;
        const int col0 = (is1 ? (nt - NTN0) : nt) * BN + wn * 64;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (is1) epilogue_bf16_sel4<N1, 8, true>(acc[i], out1, rb0 + i * 16, col0, lane, sa1, sb1);
          else     epilogue_bf16_sel4<N0, 8, true>(acc[i], out0, rb0 + i * 16, col0, lane, sa0, sb0);
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
        }
      }
    }
  }

  const int rb0 = row0_2 + warp * 32;
  constexpr int kPdlTrig = TOTAL_ALL - 8;
  if (is112) {
    float acc2[2][14][4] = {};
    for (int g = TOTAL1; g < TOTAL_ALL; ++g) {
      if constexpr (kPdl) { if (g == kPdlTrig) pdl_launch_dependents(); }
      const int s = g & 1, n = g >> 1;
      if (tid == 0 && g + 1 < TOTAL_ALL) produce(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
      consume_stage_qpipe_2xN<14, false>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) epilogue_bf16_sel4<N1, 14, true>(acc2[i], out1, rb0 + i * 16, col0_2, lane, sa1, sb1);
  } else {
    float acc2[2][12][4] = {};
    for (int g = TOTAL1; g < TOTAL_ALL; ++g) {
      if constexpr (kPdl) { if (g == kPdlTrig) pdl_launch_dependents(); }
      const int s = g & 1, n = g >> 1;
      if (tid == 0 && g + 1 < TOTAL_ALL) produce(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
      consume_stage_qpipe_2xN<12, false>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) epilogue_bf16_sel4<N1, 12, true>(acc2[i], out1, rb0 + i * 16, col0_2, lane, sa1, sb1);
  }
}

template <bool kPdl>
__global__ void __launch_bounds__(NT, 1)
bf16_dual_ws8_256_kernel(const __grid_constant__ CUtensorMap tmA256,
                         const __grid_constant__ CUtensorMap tmB_128,
                         const __grid_constant__ CUtensorMap tmB_112,
                         const __grid_constant__ CUtensorMap tmB_96,
                         __nv_bfloat16* out0, __nv_bfloat16* out1) {
  constexpr int N0 = 4608, N1 = 4096, K = 6656, NTN0 = 36, NKT = (2 * K) / KT;
  constexpr int ST_T = 2, ASZ = 256 * KT, STB = ASZ + WSZ, TOTAL1 = 2 * NKT, TOTAL_ALL = 3 * NKT;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST_T];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST_T;

  const int band2 = b & 1, c2 = b >> 1;
  const bool is112 = (c2 < 16);
  const int row0_2 = band2 * 256;
  const int w_row0_2 = is112 ? (6144 + c2 * 112) : (7936 + (c2 - 16) * 96);
  const int col0_2 = w_row0_2 - N0;
  const int tx2 = ASZ + (is112 ? (112 * KT) : (96 * KT));

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST_T; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();

  auto produce = [&](int gl) {
    const int s = gl & 1, n = gl >> 1;
    if (gl >= ST_T) {
      mbar_wait(empty0 + 8 * s, (n - 1) & 1);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
    if (gl < TOTAL1) {
      const int ti = gl / NKT, kt = gl - ti * NKT;
      const int t = b + ti * 48, nt = t >> 1, band = t & 1;
      if constexpr (kPdl) {
        if (gl == 1) {
          mbar_expect_tx(bar, ASZ);
          tma2d_hint(dst, &tmA256, kt * KT, band * 256, bar, tma_pol_el());
          return;
        }
      }
      mbar_expect_tx(bar, STB);
      tma2d_hint(dst, &tmA256, kt * KT, band * 256, bar, tma_pol_el());
      tma2d_hint(dst + ASZ, &tmB_128, kt * KT, nt * BN, bar, tma_pol_ef());
    } else {
      const int kt = gl - TOTAL1;
      mbar_expect_tx(bar, tx2);
      tma2d_hint(dst, &tmA256, kt * KT, row0_2, bar, tma_pol_el());
      if (is112) tma2d_hint(dst + ASZ, &tmB_112, kt * KT, w_row0_2, bar, tma_pol_ef());
      else       tma2d_hint(dst + ASZ, &tmB_96,  kt * KT, w_row0_2, bar, tma_pol_ef());
    }
  };

  if constexpr (kPdl) {
    const int nt0 = b >> 1, band0 = b & 1;
    if (tid == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA256) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_128) : "memory");
      mbar_expect_tx_only(full0, WSZ);
      tma2d_hint(sbase + ASZ, &tmB_128, 0, nt0 * BN, full0, tma_pol_ef());
      mbar_expect_tx_only(full0 + 8, WSZ);
      tma2d_hint(sbase + STB + ASZ, &tmB_128, KT, nt0 * BN, full0 + 8, tma_pol_ef());
    }
    pdl_wait();
    if (tid == 0) {
      mbar_expect_tx(full0, ASZ);
      tma2d_hint(sbase, &tmA256, 0, band0 * 256, full0, tma_pol_el());
    }
  } else if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA256) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_128) : "memory");
    produce(0);
  }

  const int wm = warp % WM, wn = warp / WM;
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;

  {
    float acc[4][8][4] = {};
    int kt = 0, ti = 0;
    for (int g = 0; g < TOTAL1; ++g) {
      const int s = g & 1, n = g >> 1;
      if (tid == 0) produce(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
      consume_stage_pipe_4x8<true>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++kt == NKT) {
        const int t = b + ti * 48, nt = t >> 1, band = t & 1;
        kt = 0; ++ti;
        const bool is1 = nt >= NTN0;
        const int rb0 = band * 256 + wm * 64;
        const int col0 = (is1 ? (nt - NTN0) : nt) * BN + wn * 64;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (is1) epilogue_bf16_sel4<N1, 8, false>(acc[i], out1, rb0 + i * 16, col0, lane, 1.f, 1.f);
          else     epilogue_bf16_sel4<N0, 8, false>(acc[i], out0, rb0 + i * 16, col0, lane, 1.f, 1.f);
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
        }
      }
    }
  }

  const int rb0 = row0_2 + warp * 32;
  constexpr int kPdlTrig = TOTAL_ALL - 16;
  if (is112) {
    float acc2[2][14][4] = {};
    for (int g = TOTAL1; g < TOTAL_ALL; ++g) {
      if constexpr (kPdl) { if (g == kPdlTrig) pdl_launch_dependents(); }
      const int s = g & 1, n = g >> 1;
      if (tid == 0 && g + 1 < TOTAL_ALL) produce(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
      consume_stage_qpipe_2xN<14, true>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) epilogue_bf16_sel4<N1, 14, false>(acc2[i], out1, rb0 + i * 16, col0_2, lane, 1.f, 1.f);
  } else {
    float acc2[2][12][4] = {};
    for (int g = TOTAL1; g < TOTAL_ALL; ++g) {
      if constexpr (kPdl) { if (g == kPdlTrig) pdl_launch_dependents(); }
      const int s = g & 1, n = g >> 1;
      if (tid == 0 && g + 1 < TOTAL_ALL) produce(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
      consume_stage_qpipe_2xN<12, true>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) epilogue_bf16_sel4<N1, 12, false>(acc2[i], out1, rb0 + i * 16, col0_2, lane, 1.f, 1.f);
  }
}

template <typename Kernel, typename... Args>
static inline cudaError_t launch_pdl_ex(Kernel kernel, int blocks, int threads, size_t smem, cudaStream_t st,
                                        const Args&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks, 1, 1);
  cfg.blockDim = dim3(threads, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace swiglu_ws
