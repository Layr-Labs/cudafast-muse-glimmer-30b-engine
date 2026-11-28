// SPDX-License-Identifier: Apache-2.0
// fp8ws: zero-VRAM dual-TMA M=512 prefill FP8 GEMM for MuseGlimmer-30B-NVFP4,
// C[512,N] = A[512,K] x W[N,K]^T, BF16 out, across all 5 MuseGlimmer FP8 shapes:
// (4608,6656), (4096,6656), (6656,4096), (39936,6656), (6656,19968).
//
// Arithmetic: one ascending-K chain of mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
// from +0.0f (no split-K), then bf16_rn(fmul(fmul(acc, scale_b), scale_a)).
// Dual 2-D TMA (tmA on A[M,K], tmB directly on column-major B[K,N] == row-major W[N,K])
// with SWIZZLE_128B: zero extra weight VRAM and zero load-time packing.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <stdint.h>

namespace fp8ws {
constexpr int M = 512;
constexpr int KT = 128;
constexpr int WM = 4, WN = 2, NW = 8;
constexpr int BN = WN * NW * 8;        // 128
constexpr int NWARP = WM * WN, NT = NWARP * 32;
constexpr int WSZ = BN * KT;           // 16384 B
constexpr int SMEM_MAX = 99328;

template <int N, int K>
struct Shape {
  static constexpr int ST = (N <= 4608) ? 3 : 2;
  static constexpr int MW = (N <= 4608) ? 2 : 4;
  static constexpr int BM = WM * MW * 16;          // 128 for N<=4608, 256 otherwise
  static constexpr int NB = M / BM;                // 4 bands for BM=128, 2 for BM=256
  static constexpr int ASZ = BM * KT;              // 16384 or 32768 B
  static constexpr int STB = ASZ + WSZ;            // 32768 or 49152 B
  static constexpr int SMEM = ST * STB + 1024;     // 99328 B <= 99 KiB
  static_assert(M % BM == 0 && STB % 1024 == 0, "stage");
  static_assert(N % BN == 0 && K % KT == 0, "shape");
  static constexpr int NTN = N / BN, NTILES = NB * NTN, NKT = K / KT;
};

__host__ __device__ constexpr int swz(int r, int c) { return c ^ (r & 7); }

__device__ __forceinline__ void mbar_init(uint32_t a, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, int tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx_only(uint32_t a, int tx) {
  asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(tx) : "memory");
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;"); }
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, int parity) {
  asm volatile(
      "{\n .reg .pred p;\n W_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra W_%=;\n}\n" ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ uint64_t tma_pol_el() {
  uint64_t p;
  asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t tma_pol_ef() {
  uint64_t p;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void tma2d_hint(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar,
                                           uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;"
      ::"r"(dst), "l"(map), "r"(x), "r"(y), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      ::"r"(dst), "l"(map), "r"(x), "r"(y), "r"(bar) : "memory");
}
template <bool kPolEf>
__device__ __forceinline__ void tma2d_b(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar,
                                        uint64_t pol_ef) {
  if constexpr (kPolEf) tma2d_hint(dst, map, x, y, bar, pol_ef);
  else                  tma2d(dst, map, x, y, bar);
}
__device__ __forceinline__ void prewarm_tlb_2m(const uint8_t* ptr, size_t nbytes, int tid, int nthreads) {
  if (ptr == nullptr) return;
  constexpr size_t kStep = 2048u << 10;
  for (size_t off = static_cast<size_t>(tid) * kStep; off < nbytes; off += static_cast<size_t>(nthreads) * kStep) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr + off));
  }
}
__device__ __forceinline__ void ldsm4(uint32_t a, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma(const uint32_t (&a)[4], uint32_t b0, uint32_t b1, float (&c)[4]) {
  asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ uint32_t cvt2(float lo, float hi, float sa, float sb) {
  float x = __fmul_rn(__fmul_rn(lo, sb), sa);
  float y = __fmul_rn(__fmul_rn(hi, sb), sa);
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(x), "f"(y));
  return r;
}
__device__ __forceinline__ uint32_t cvt2_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}
__device__ __forceinline__ void mma_bf16(const uint32_t (&a)[4], uint32_t b0, uint32_t b1, float (&c)[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ uint32_t selp(uint32_t a, uint32_t b, bool c) {
  uint32_t r;
  asm("{\n .reg .pred p;\n setp.ne.b32 p, %3, 0;\n selp.b32 %0, %1, %2, p;\n}"
      : "=r"(r) : "r"(a), "r"(b), "r"((uint32_t)c));
  return r;
}
__device__ __forceinline__ uint32_t sel4(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, int i) {
  const uint32_t lo = selp(v1, v0, i & 1), hi = selp(v3, v2, i & 1);
  return selp(hi, lo, i & 2);
}

struct Params {
  const float* sa;
  const float* sb;
  __nv_bfloat16* out;
  const uint8_t* b_raw = nullptr;
  size_t b_bytes = 0;
};

struct ParamsDual {
  const float* sa;
  const float* sb0;
  const float* sb1;
  __nv_bfloat16* out0;
  __nv_bfloat16* out1;
};

struct ParamsDualA {
  const float* sa0;
  const float* sa1;
  const float* sb0;
  const float* sb1;
  __nv_bfloat16* out0;
  __nv_bfloat16* out1;
  const uint8_t* b0_raw = nullptr;
  size_t b0_bytes = 0;
  const uint8_t* b1_raw = nullptr;
  size_t b1_bytes = 0;
};

struct ParamsBf16Dual {
  __nv_bfloat16* out0;
  __nv_bfloat16* out1;
  const uint8_t* b_raw = nullptr;
  size_t b_bytes = 0;
};

template <int N0, int N1, int K>
struct ShapeDual {
  static constexpr int ST = 3;
  static constexpr int MW = 2;
  static constexpr int BM = WM * MW * 16;          // 128
  static constexpr int NB = M / BM;                // 4 bands
  static constexpr int ASZ = BM * KT;              // 16384 B
  static constexpr int STB = ASZ + WSZ;            // 32768 B
  static constexpr int SMEM = ST * STB + 1024;     // 99328 B <= 99 KiB
  static_assert(M % BM == 0 && STB % 1024 == 0, "stage");
  static_assert(N0 % BN == 0 && N1 % BN == 0 && K % KT == 0, "shape");
  static constexpr int NTN0 = N0 / BN, NTN1 = N1 / BN, NTN = NTN0 + NTN1;
  static constexpr int NTILES = NB * NTN, NKT = K / KT;
};

template <int N0, int N1, int K>
struct ShapeDual256 {
  static constexpr int ST = 2;
  static constexpr int MW = 4;
  static constexpr int BM = WM * MW * 16;          // 256
  static constexpr int NB = M / BM;                // 2 bands
  static constexpr int ASZ = BM * KT;              // 32768 B
  static constexpr int STB = ASZ + WSZ;            // 49152 B
  static constexpr int SMEM = ST * STB + 1024;     // 99328 B <= 99 KiB
  static_assert(M % BM == 0 && STB % 1024 == 0, "stage");
  static_assert(N0 % BN == 0 && N1 % BN == 0 && K % KT == 0, "shape");
  static constexpr int NTN0 = N0 / BN, NTN1 = N1 / BN, NTN = NTN0 + NTN1;
  static constexpr int NTILES = NB * NTN, NKT = K / KT;
};

template <int N, int NW_CUR = NW>
__device__ __forceinline__ void epilogue(const float (&acc)[NW_CUR][4], __nv_bfloat16* out, int row0, int col0,
                                         int lane, float sa, float sb) {
  const int q = lane & 3, rq = lane >> 2;
#pragma unroll
  for (int pp = 0; pp < NW_CUR / 2; ++pp) {
    const int j = 2 * pp;
    const uint32_t v0 = cvt2(acc[j][0], acc[j][1], sa, sb);
    const uint32_t v1 = cvt2(acc[j][2], acc[j][3], sa, sb);
    const uint32_t v2 = cvt2(acc[j + 1][0], acc[j + 1][1], sa, sb);
    const uint32_t v3 = cvt2(acc[j + 1][2], acc[j + 1][3], sa, sb);
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
    *reinterpret_cast<uint4*>(out + (size_t)row * N + col) = w;
  }
}

template <int N, int NW_CUR = NW>
__device__ __forceinline__ void epilogue_bf16(const float (&acc)[NW_CUR][4], __nv_bfloat16* out, int row0, int col0,
                                              int lane) {
  const int q = lane & 3, rq = lane >> 2;
#pragma unroll
  for (int pp = 0; pp < NW_CUR / 2; ++pp) {
    const int j = 2 * pp;
    const uint32_t v0 = cvt2_bf16(acc[j][0], acc[j][1]);
    const uint32_t v1 = cvt2_bf16(acc[j][2], acc[j][3]);
    const uint32_t v2 = cvt2_bf16(acc[j + 1][0], acc[j + 1][1]);
    const uint32_t v3 = cvt2_bf16(acc[j + 1][2], acc[j + 1][3]);
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
    *reinterpret_cast<uint4*>(out + (size_t)row * N + col) = w;
  }
}

template <int MW, int NW_CUR = NW, bool IsBf16 = false>
__device__ __forceinline__ void consume_stage(uint32_t abase, uint32_t wbase, int wm, int wn, int ar, int ach,
                                              int br, int bch, float (&acc)[MW][NW_CUR][4]) {
#pragma unroll
  for (int ks = 0; ks < KT / 32; ++ks) {
    uint32_t a[MW][4], b[NW_CUR][2];
#pragma unroll
    for (int i = 0; i < MW; ++i) {
      const int r = wm * MW * 16 + i * 16 + ar;
      ldsm4(abase + r * KT + swz(r, 2 * ks + ach) * 16, a[i]);
    }
#pragma unroll
    for (int j = 0; j < NW_CUR / 2; ++j) {
      const int r = wn * NW_CUR * 8 + j * 16 + br;
      uint32_t qd[4];
      ldsm4(wbase + r * KT + swz(r, 2 * ks + bch) * 16, qd);
      b[2 * j][0] = qd[0];
      b[2 * j][1] = qd[1];
      b[2 * j + 1][0] = qd[2];
      b[2 * j + 1][1] = qd[3];
    }
#pragma unroll
    for (int i = 0; i < MW; ++i)
#pragma unroll
      for (int j = 0; j < NW_CUR; ++j) {
        if constexpr (IsBf16) mma_bf16(a[i], b[j][0], b[j][1], acc[i][j]);
        else mma(a[i], b[j][0], b[j][1], acc[i][j]);
      }
  }
}

template <int MW, int NW_CUR = NW, bool IsBf16 = false>
__device__ __forceinline__ void consume_stage_pipe(uint32_t abase, uint32_t wbase, int wm, int wn, int ar, int ach,
                                                   int br, int bch, float (&acc)[MW][NW_CUR][4]) {
  static_assert(NW_CUR % 2 == 0, "streaming-B consume needs even NW_CUR");
  constexpr int NP = NW_CUR / 2;
  constexpr int KSTEPS = KT / 32;
  auto load_a_i = [&](int ks, int i, uint32_t (&ai)[4]) {
    const int r = wm * MW * 16 + i * 16 + ar;
    ldsm4(abase + r * KT + swz(r, 2 * ks + ach) * 16, ai);
  };
  auto load_b_p = [&](int ks, int p, uint32_t (&bp)[4]) {
    const int r = wn * NW_CUR * 8 + p * 16 + br;
    ldsm4(wbase + r * KT + swz(r, 2 * ks + bch) * 16, bp);
  };
  uint32_t a[MW][4];
  uint32_t bbuf[2][4];
#pragma unroll
  for (int i = 0; i < MW; ++i) load_a_i(0, i, a[i]);
  load_b_p(0, 0, bbuf[0]);
#pragma unroll
  for (int ks = 0; ks < KSTEPS; ++ks) {
#pragma unroll
    for (int p = 0; p < NP; ++p) {
      const int q = ks * NP + p;
      uint32_t(&bc)[4] = bbuf[q & 1];
      uint32_t(&bn)[4] = bbuf[(q + 1) & 1];
      if (p + 1 < NP) load_b_p(ks, p + 1, bn);
      else if (ks + 1 < KSTEPS) load_b_p(ks + 1, 0, bn);
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        if constexpr (IsBf16) {
          mma_bf16(a[i], bc[0], bc[1], acc[i][2 * p]);
          mma_bf16(a[i], bc[2], bc[3], acc[i][2 * p + 1]);
        } else {
          mma(a[i], bc[0], bc[1], acc[i][2 * p]);
          mma(a[i], bc[2], bc[3], acc[i][2 * p + 1]);
        }
        if (p == NP - 1 && ks + 1 < KSTEPS) load_a_i(ks + 1, i, a[i]);
      }
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
  uint32_t a[2][4][4], bbuf[2][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) load_a_i(0, i, a[0][i]);
  load_b_p(0, 0, bbuf[0]);
#pragma unroll
  for (int ks = 0; ks < KT / 32; ++ks) {
    const int acur = ks & 1, anxt = acur ^ 1;
#pragma unroll
    for (int jp = 0; jp < 4; ++jp) {
      const int q = (ks << 2) + jp;
      uint32_t (&bc)[4] = bbuf[q & 1];
      uint32_t (&bn)[4] = bbuf[(q + 1) & 1];
      if (jp + 1 < 4)            load_b_p(ks, jp + 1, bn);
      else if (ks + 1 < KT / 32) load_b_p(ks + 1, 0, bn);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if constexpr (IsBf16) {
          mma_bf16(a[acur][i], bc[0], bc[1], acc[i][2 * jp]);
          mma_bf16(a[acur][i], bc[2], bc[3], acc[i][2 * jp + 1]);
        } else {
          mma(a[acur][i], bc[0], bc[1], acc[i][2 * jp]);
          mma(a[acur][i], bc[2], bc[3], acc[i][2 * jp + 1]);
        }
        if (jp == 2 && ks + 1 < KT / 32) load_a_i(ks + 1, i, a[anxt][i]);
      }
    }
  }
}

template <int MW, int NW_CUR, bool IsBf16 = false>
__device__ __forceinline__ void consume_stage_pipe_opt(
    uint32_t abase, uint32_t wbase, int wm, int wn, int ar, int ach, int br, int bch, float (&acc)[MW][NW_CUR][4]) {
  constexpr int NP = NW_CUR / 2;
  auto load_a_i = [&](int ks, int i, uint32_t (&ai)[4]) {
    const int r = wm * MW * 16 + i * 16 + ar;
    ldsm4(abase + r * KT + swz(r, 2 * ks + ach) * 16, ai);
  };
  auto load_b_p = [&](int ks, int p, uint32_t (&bp)[4]) {
    const int r = wn * NW_CUR * 8 + p * 16 + br;
    ldsm4(wbase + r * KT + swz(r, 2 * ks + bch) * 16, bp);
  };
  uint32_t a[2][MW][4], bbuf[2][4];
#pragma unroll
  for (int i = 0; i < MW; ++i) load_a_i(0, i, a[0][i]);
  load_b_p(0, 0, bbuf[0]);
#pragma unroll
  for (int ks = 0; ks < KT / 32; ++ks) {
    const int acur = ks & 1, anxt = acur ^ 1;
#pragma unroll
    for (int jp = 0; jp < NP; ++jp) {
      const int q = ks * NP + jp;
      uint32_t (&bc)[4] = bbuf[q & 1];
      uint32_t (&bn)[4] = bbuf[(q + 1) & 1];
      if (jp + 1 < NP)           load_b_p(ks, jp + 1, bn);
      else if (ks + 1 < KT / 32) load_b_p(ks + 1, 0, bn);
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        if constexpr (IsBf16) {
          mma_bf16(a[acur][i], bc[0], bc[1], acc[i][2 * jp]);
          mma_bf16(a[acur][i], bc[2], bc[3], acc[i][2 * jp + 1]);
        } else {
          mma(a[acur][i], bc[0], bc[1], acc[i][2 * jp]);
          mma(a[acur][i], bc[2], bc[3], acc[i][2 * jp + 1]);
        }
        if (jp == (NP >= 2 ? NP - 2 : 0) && ks + 1 < KT / 32) load_a_i(ks + 1, i, a[anxt][i]);
      }
    }
  }
}

template <int N, int K, bool kPdl = false>
__global__ void __launch_bounds__(NT, 1)
gemm(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB, const Params p) {
  using S = Shape<N, K>;
  constexpr int ST = S::ST;
  constexpr int MW = S::MW, BM = S::BM, NB = S::NB, ASZ = S::ASZ, STB = S::STB;
  constexpr int NTILES = S::NTILES, NKT = S::NKT;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST;
  const int grid = gridDim.x;
  if ((int)blockIdx.x >= NTILES) {
    if constexpr (kPdl) pdl_launch_dependents();
    return;
  }
  const int nmine = (NTILES - blockIdx.x + grid - 1) / grid;
  const int total = nmine * NKT;
  const int pdl_trig = (total > 10) ? (total - 10) : 0;
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  auto produce = [&](int gl) {
    const int s = gl % ST, n = gl / ST;
    if (gl >= ST) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
    const int ti = gl / NKT, kt = gl - ti * NKT;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
    mbar_expect_tx(bar, STB);
    tma2d_hint(dst, &tmA, kt * KT, band * BM, bar, tma_pol_el());
    tma2d_hint(dst + ASZ, &tmB, kt * KT, nt * BN, bar, tma_pol_ef());
  };
  if constexpr (kPdl) {
    if (tid == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
      const uint64_t pol_ef = tma_pol_ef();
      for (int s = 0; s < ST - 1 && s < total; ++s) {
        const int ti0 = s / NKT, kt0 = s - ti0 * NKT;
        const int t0 = blockIdx.x + ti0 * grid, nt0 = t0 / NB;
        const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
        mbar_expect_tx_only(bar, WSZ);
        tma2d_hint(dst + ASZ, &tmB, kt0 * KT, nt0 * BN, bar, pol_ef);
      }
    }
    pdl_wait();
    if (tid == 0) {
      const uint64_t pol_el = tma_pol_el();
      for (int s = 0; s < ST - 1 && s < total; ++s) {
        const int ti0 = s / NKT, kt0 = s - ti0 * NKT;
        const int t0 = blockIdx.x + ti0 * grid, nt0 = t0 / NB, band0 = t0 - nt0 * NB;
        const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
        mbar_expect_tx(bar, ASZ);
        tma2d_hint(dst, &tmA, kt0 * KT, band0 * BM, bar, pol_el);
      }
    }
  } else if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
    for (int s = 0; s < ST - 1 && s < total; ++s) produce(s);
  }
  const int wm = warp % WM, wn = warp / WM;
  const float sa = __ldg(p.sa), sb = __ldg(p.sb);
  float acc[MW][NW][4];
#pragma unroll
  for (int i = 0; i < MW; ++i)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  int kt = 0, ti = 0;
  for (int g = 0; g < total; ++g) {
    if constexpr (kPdl) {
      if (g == pdl_trig) pdl_launch_dependents();
    }
    const int s = g % ST, n = g / ST;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const int rb0 = band * BM + wm * MW * 16;
    if (tid == 0 && g + ST - 1 < total) produce(g + ST - 1);
    mbar_wait(full0 + 8 * s, n & 1);
    const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
    if constexpr (MW == 4)      consume_stage_pipe_4x8<false>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
    else if constexpr (MW == 2) consume_stage_pipe<MW>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
    else                        consume_stage<MW>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * s);
    if (++kt == NKT) {
      kt = 0;
      ++ti;
      const int col0 = nt * BN + wn * NW * 8;
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        epilogue<N>(acc[i], p.out, rb0 + i * 16, col0, lane, sa, sb);
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
      }
    }
  }
}

template <int K>
__global__ void __launch_bounds__(NT, 1)
gemm_hybrid_6656(const __grid_constant__ CUtensorMap tmA256, const __grid_constant__ CUtensorMap tmA64,
                 const __grid_constant__ CUtensorMap tmB, const Params p) {
  constexpr int N = 6656;
  constexpr int NKT = K / KT;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * 4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * 4;
  const int b = blockIdx.x;
  const int wm = warp % WM, wn = warp / WM;
  const float sa = __ldg(p.sa), sb = __ldg(p.sb);
  float acc[4][NW][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;

  {  // Phase 1: 96 tiles of 256x128 (cols 0..47), 2 full 48-SM waves
    constexpr int ST1 = 2, MW1 = 4, BM1 = 256, ASZ1 = BM1 * KT, STB1 = ASZ1 + WSZ;
    constexpr int total1 = 2 * NKT;
    if (tid == 0) {
      for (int s = 0; s < ST1; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    auto produce1 = [&](int gl) {
      const int s = gl % ST1, n = gl / ST1;
      if (gl >= ST1) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
      const int ti = gl / NKT, kt = gl - ti * NKT;
      const int t = b + ti * 48, nt = t >> 1, band = t & 1;
      const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB1;
      mbar_expect_tx(bar, STB1);
      tma2d_hint(dst, &tmA256, kt * KT, band * BM1, bar, tma_pol_el());
      tma2d_hint(dst + ASZ1, &tmB, kt * KT, nt * BN, bar, tma_pol_ef());
    };
    if (tid == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA256) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
      produce1(0);
    }
    int kt = 0, ti = 0;
    for (int g = 0; g < total1; ++g) {
      const int s = g % ST1, n = g / ST1;
      const int t = b + ti * 48, nt = t >> 1, band = t & 1;
      const int rb0 = band * BM1 + wm * MW1 * 16;
      if (tid == 0 && g + ST1 - 1 < total1) produce1(g + ST1 - 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB1, wbase = abase + ASZ1;
      consume_stage_pipe_4x8<false>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++kt == NKT) {
        kt = 0;
        ++ti;
        const int col0 = nt * BN + wn * NW * 8;
#pragma unroll
        for (int i = 0; i < MW1; ++i) {
          epilogue<N>(acc[i], p.out, rb0 + i * 16, col0, lane, sa, sb);
#pragma unroll
          for (int j = 0; j < NW; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
        }
      }
    }
  }

  __syncthreads();
  if (b >= 32) return;

  {  // Phase 2: 32 tiles of 64x128 (cols 48..51, 8x weight reuse), ST=4
    constexpr int ST2 = 4, BM2 = 64, ASZ2 = BM2 * KT, STB2 = ASZ2 + WSZ;
    const int nt = 48 + (b >> 3), band = b & 7;
    if (tid == 0) {
      for (int s = 0; s < ST2; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    auto produce2 = [&](int gl) {
      const int s = gl & 3, n = gl >> 2;
      if (gl >= ST2) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
      const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB2;
      mbar_expect_tx(bar, STB2);
      tma2d_hint(dst, &tmA64, gl * KT, band * BM2, bar, tma_pol_el());
      tma2d_hint(dst + ASZ2, &tmB, gl * KT, nt * BN, bar, tma_pol_el());
    };
    if (tid == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA64) : "memory");
      for (int s = 0; s < ST2 - 1 && s < NKT; ++s) produce2(s);
    }
    float (&acc2)[1][NW][4] = reinterpret_cast<float (&)[1][NW][4]>(acc[0]);
    const int rb0 = band * BM2 + wm * 16;
    const int col0 = nt * BN + wn * NW * 8;
    for (int g = 0; g < NKT; ++g) {
      const int s = g & 3, n = g >> 2;
      if (tid == 0 && g + ST2 - 1 < NKT) produce2(g + ST2 - 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB2, wbase = abase + ASZ2;
      consume_stage_pipe_opt<1, NW, false>(abase, wbase, wm, wn, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
    epilogue<N>(acc[0], p.out, rb0, col0, lane, sa, sb);
  }
}

// Wave-29b: 48-SM Phase-2 N-split (4 bands of 128 rows x 12 col-CTAs: 8x 128x48 + 4x 128x32, ST=4)
template <int K, bool kPdl = false>
__global__ void __launch_bounds__(NT, 1)
gemm_hybrid_6656_nsplit128(const __grid_constant__ CUtensorMap tmA256,
                           const __grid_constant__ CUtensorMap tmA128,
                           const __grid_constant__ CUtensorMap tmB128,
                           const __grid_constant__ CUtensorMap tmB48,
                           const __grid_constant__ CUtensorMap tmB32,
                           const Params p) {
  constexpr int N = 6656, NKT = K / KT;
  constexpr int kPdlTrig = (K == 19968) ? (NKT - 32) : (NKT - 8);
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * 4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * 4;
  const int wm = warp % WM, wn = warp / WM;
  const float sa = __ldg(p.sa), sb = __ldg(p.sb);
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;

  {  // Phase 1: 96 tiles of 256x128 (cols 0..6143), ST = 2
    constexpr int ST1 = 2, MW1 = 4, BM1 = 256, ASZ1 = BM1 * KT, STB1 = ASZ1 + WSZ, total1 = 2 * NKT;
    float acc[MW1][NW][4] = {};
    if (tid == 0) {
      for (int s = 0; s < ST1; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    auto produce1 = [&](int gl) {
      const int s = gl & 1, n = gl >> 1;
      if (gl >= ST1) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
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
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB128) : "memory");
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
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB128) : "memory");
      produce1(0);
    }
    int kt = 0, ti = 0;
    for (int g = 0; g < total1; ++g) {
      const int s = g & 1, n = g >> 1;
      const int t = b + ti * 48, nt = t >> 1, band = t & 1;
      const int rb0 = band * BM1 + wm * MW1 * 16;
      if (tid == 0 && g + 1 < total1) produce1(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB1, wbase = abase + ASZ1;
      consume_stage_pipe_4x8<false>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++kt == NKT) {
        kt = 0; ++ti;
        const int col0 = nt * BN + wn * NW * 8;
#pragma unroll
        for (int i = 0; i < MW1; ++i) {
          epilogue<N, NW>(acc[i], p.out, rb0 + i * 16, col0, lane, sa, sb);
#pragma unroll
          for (int j = 0; j < NW; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
        }
      }
    }
  }

  __syncthreads();

  {  // Phase 2: ALL 48 SMs on cols 6144..6655 (4 bands of 128 rows x 12 col-CTAs: 8x 128x48 + 4x 128x32), ST = 4
    constexpr int ST2 = 4, ASZ2 = 128 * KT, WSZ_MAX = 48 * KT, STB2 = ASZ2 + WSZ_MAX;
    const int band = b & 3, c = b >> 2;
    const bool is48 = (c < 8);
    const int row0 = band * 128;
    const int col0 = is48 ? (6144 + c * 48) : (6144 + 384 + (c - 8) * 32);
    const int tx_bytes = ASZ2 + (is48 ? (48 * KT) : (32 * KT));
    if (tid == 0) {
      for (int s = 0; s < ST2; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    auto produce2 = [&](int gl) {
      const int s = gl & 3, n = gl >> 2;
      if (gl >= ST2) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
      const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB2;
      mbar_expect_tx(bar, tx_bytes);
      tma2d_hint(dst, &tmA128, gl * KT, row0, bar, tma_pol_el());
      if (is48) tma2d_hint(dst + ASZ2, &tmB48, gl * KT, col0, bar, tma_pol_el());
      else      tma2d_hint(dst + ASZ2, &tmB32, gl * KT, col0, bar, tma_pol_el());
    };
    if (tid == 0) {
      for (int s = 0; s < ST2 - 1 && s < NKT; ++s) produce2(s);
    }
    const int rb0 = row0 + warp * 16;
    if (is48) {
      float acc2[1][6][4] = {};
      for (int g = 0; g < NKT; ++g) {
        if constexpr (kPdl) {
          if (g == kPdlTrig) pdl_launch_dependents();
        }
        const int s = g & 3, n = g >> 2;
        if (tid == 0 && g + ST2 - 1 < NKT) produce2(g + ST2 - 1);
        mbar_wait(full0 + 8 * s, n & 1);
        const uint32_t abase = sbase + s * STB2, wbase = abase + ASZ2;
        consume_stage_pipe_opt<1, 6, false>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
      }
      epilogue<N, 6>(acc2[0], p.out, rb0, col0, lane, sa, sb);
    } else {
      float acc2[1][4][4] = {};
      for (int g = 0; g < NKT; ++g) {
        if constexpr (kPdl) {
          if (g == kPdlTrig) pdl_launch_dependents();
        }
        const int s = g & 3, n = g >> 2;
        if (tid == 0 && g + ST2 - 1 < NKT) produce2(g + ST2 - 1);
        mbar_wait(full0 + 8 * s, n & 1);
        const uint32_t abase = sbase + s * STB2, wbase = abase + ASZ2;
        consume_stage_pipe_opt<1, 4, false>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
      }
      epilogue<N, 4>(acc2[0], p.out, rb0, col0, lane, sa, sb);
    }
  }
}

template <int N0, int N1, int K>
__global__ void __launch_bounds__(NT, 1)
gemm_dual(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB0,
          const __grid_constant__ CUtensorMap tmB1, const ParamsDual p) {
  using S = ShapeDual256<N0, N1, K>;
  constexpr int ST = S::ST;
  constexpr int MW = S::MW, BM = S::BM, NB = S::NB, ASZ = S::ASZ, STB = S::STB;
  constexpr int NTN0 = S::NTN0, NTILES = S::NTILES, NKT = S::NKT;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST;
  const int grid = gridDim.x;
  if ((int)blockIdx.x >= NTILES) return;
  const int nmine = (NTILES - blockIdx.x + grid - 1) / grid;
  const int total = nmine * NKT;
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  auto produce = [&](int gl) {
    const int s = gl % ST, n = gl / ST;
    if (gl >= ST) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
    const int ti = gl / NKT, kt = gl - ti * NKT;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
    mbar_expect_tx(bar, STB);
    tma2d_hint(dst, &tmA, kt * KT, band * BM, bar, tma_pol_el());
    if (nt < NTN0) {
      tma2d_hint(dst + ASZ, &tmB0, kt * KT, nt * BN, bar, tma_pol_ef());
    } else {
      tma2d_hint(dst + ASZ, &tmB1, kt * KT, (nt - NTN0) * BN, bar, tma_pol_ef());
    }
  };
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB0) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB1) : "memory");
    for (int s = 0; s < ST - 1 && s < total; ++s) produce(s);
  }
  const int wm = warp % WM, wn = warp / WM;
  const float sa = __ldg(p.sa), sb0 = __ldg(p.sb0), sb1 = __ldg(p.sb1);
  float acc[MW][NW][4];
#pragma unroll
  for (int i = 0; i < MW; ++i)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  int kt = 0, ti = 0;
  for (int g = 0; g < total; ++g) {
    const int s = g % ST, n = g / ST;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const int rb0 = band * BM + wm * MW * 16;
    if (tid == 0 && g + ST - 1 < total) produce(g + ST - 1);
    mbar_wait(full0 + 8 * s, n & 1);
    const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
    consume_stage_pipe_4x8<false>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * s);
    if (++kt == NKT) {
      kt = 0;
      ++ti;
      const bool is1 = nt >= NTN0;
      const int col0 = (is1 ? (nt - NTN0) : nt) * BN + wn * NW * 8;
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        if (is1) {
          epilogue<N1>(acc[i], p.out1, rb0 + i * 16, col0, lane, sa, sb1);
        } else {
          epilogue<N0>(acc[i], p.out0, rb0 + i * 16, col0, lane, sa, sb0);
        }
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
      }
    }
  }
}

template <int N0, int N1, int K, int ST2>
__global__ void __launch_bounds__(NT, 1)
gemm_dual_a(const __grid_constant__ CUtensorMap tmA0, const __grid_constant__ CUtensorMap tmA1,
            const __grid_constant__ CUtensorMap tmB0, const __grid_constant__ CUtensorMap tmB1,
            const ParamsDualA p) {
  using S = ShapeDual256<N0, N1, K>;
  constexpr int MW = S::MW, BM = S::BM, NB = S::NB, ASZ = S::ASZ, STB = S::STB;
  constexpr int NTN0 = S::NTN0, NTILES = S::NTILES, NKT = S::NKT;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST2];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST2;
  const int grid = gridDim.x;
  if ((int)blockIdx.x >= NTILES) return;
  const int nmine = (NTILES - blockIdx.x + grid - 1) / grid;
  const int total = nmine * NKT;
  if (tid == 0) {
    for (int s = 0; s < ST2; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  auto produce = [&](int gl) {
    const int s = gl % ST2, n = gl / ST2;
    if (gl >= ST2) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
    const int ti = gl / NKT, kt = gl - ti * NKT;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
    mbar_expect_tx(bar, STB);
    if (nt < NTN0) {
      tma2d_hint(dst, &tmA0, kt * KT, band * BM, bar, tma_pol_el());
      tma2d_hint(dst + ASZ, &tmB0, kt * KT, nt * BN, bar, tma_pol_ef());
    } else {
      tma2d_hint(dst, &tmA1, kt * KT, band * BM, bar, tma_pol_el());
      tma2d_hint(dst + ASZ, &tmB1, kt * KT, (nt - NTN0) * BN, bar, tma_pol_ef());
    }
  };
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA0) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA1) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB0) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB1) : "memory");
    for (int s = 0; s < ST2 - 1 && s < total; ++s) produce(s);
  }
  const int wm = warp % WM, wn = warp / WM;
  const float sa0 = __ldg(p.sa0), sa1 = __ldg(p.sa1), sb0 = __ldg(p.sb0), sb1 = __ldg(p.sb1);
  float acc[MW][NW][4];
#pragma unroll
  for (int i = 0; i < MW; ++i)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  int kt = 0, ti = 0;
  for (int g = 0; g < total; ++g) {
    const int s = g % ST2, n = g / ST2;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const int rb0 = band * BM + wm * MW * 16;
    if (tid == 0 && g + ST2 - 1 < total) produce(g + ST2 - 1);
    mbar_wait(full0 + 8 * s, n & 1);
    const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
    consume_stage_pipe_4x8<false>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * s);
    if (++kt == NKT) {
      kt = 0;
      ++ti;
      const bool is1 = nt >= NTN0;
      const int col0 = (is1 ? (nt - NTN0) : nt) * BN + wn * NW * 8;
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        if (is1) {
          epilogue<N1>(acc[i], p.out1, rb0 + i * 16, col0, lane, sa1, sb1);
        } else {
          epilogue<N0>(acc[i], p.out0, rb0 + i * 16, col0, lane, sa0, sb0);
        }
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
      }
    }
  }
}

// Wave-29b: Seamless zero-drain 48-SM Wave-3 N-split for gemm_dual_a (BM=256 preserved, 32x 256x112 + 16x 256x96)
template <bool kPdl = false>
__global__ void __launch_bounds__(NT, 1)
gemm_dual_a_seamless(const __grid_constant__ CUtensorMap tmA0,
                     const __grid_constant__ CUtensorMap tmA1,
                     const __grid_constant__ CUtensorMap tmB0,
                     const __grid_constant__ CUtensorMap tmB1_128,
                     const __grid_constant__ CUtensorMap tmB1_112,
                     const __grid_constant__ CUtensorMap tmB1_96,
                     const ParamsDualA p) {
  constexpr int N0 = 4608, N1 = 4096, K = 6656, NTN0 = 36, NKT = K / KT;
  constexpr int ASZ1 = 256 * KT, STB1 = ASZ1 + WSZ;
  constexpr int total1 = 2 * NKT, total_all = 3 * NKT;
  constexpr int kPdlTrig = total_all - 8;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 16;

  const int band2 = b & 1, c2 = b >> 1;
  const bool is112 = (c2 < 16);
  const int row0_2 = band2 * 256;
  const int col0_2 = is112 ? (1536 + c2 * 112) : (3328 + (c2 - 16) * 96);
  const int tx2 = ASZ1 + (is112 ? (112 * KT) : (96 * KT));

  if (tid == 0) {
    mbar_init(full0, 1); mbar_init(full0 + 8, 1);
    mbar_init(empty0, NWARP); mbar_init(empty0 + 8, NWARP);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();

  auto produce = [&](int gl) {
    const int s = gl & 1, n = gl >> 1;
    if (gl >= 2) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
    const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB1;
    if (gl < total1) {
      const int ti = gl / NKT, kt = gl - ti * NKT;
      const int t = b + ti * 48, nt = t >> 1, band = t & 1;
      if constexpr (kPdl) {
        if (gl == 1) {
          mbar_expect_tx(bar, ASZ1);
          if (nt < NTN0) tma2d_hint(dst, &tmA0, kt * KT, band * 256, bar, tma_pol_el());
          else           tma2d_hint(dst, &tmA1, kt * KT, band * 256, bar, tma_pol_el());
          return;
        }
      }
      mbar_expect_tx(bar, STB1);
      if (nt < NTN0) {
        tma2d_hint(dst, &tmA0, kt * KT, band * 256, bar, tma_pol_el());
        tma2d_hint(dst + ASZ1, &tmB0, kt * KT, nt * BN, bar, tma_pol_ef());
      } else {
        tma2d_hint(dst, &tmA1, kt * KT, band * 256, bar, tma_pol_el());
        tma2d_hint(dst + ASZ1, &tmB1_128, kt * KT, (nt - NTN0) * BN, bar, tma_pol_ef());
      }
    } else {
      const int kt = gl - total1;
      mbar_expect_tx(bar, tx2);
      tma2d_hint(dst, &tmA1, kt * KT, row0_2, bar, tma_pol_el());
      if (is112) tma2d_hint(dst + ASZ1, &tmB1_112, kt * KT, col0_2, bar, tma_pol_ef());
      else       tma2d_hint(dst + ASZ1, &tmB1_96,  kt * KT, col0_2, bar, tma_pol_ef());
    }
  };

  const float sa0 = __ldg(p.sa0), sa1 = __ldg(p.sa1), sb0 = __ldg(p.sb0), sb1 = __ldg(p.sb1);
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
        tma2d_hint(sbase + ASZ1, &tmB0, 0, nt0 * BN, full0, tma_pol_ef());
        tma2d_hint(sbase + STB1 + ASZ1, &tmB0, KT, nt0 * BN, full0 + 8, tma_pol_ef());
      } else {
        tma2d_hint(sbase + ASZ1, &tmB1_128, 0, (nt0 - NTN0) * BN, full0, tma_pol_ef());
        tma2d_hint(sbase + STB1 + ASZ1, &tmB1_128, KT, (nt0 - NTN0) * BN, full0 + 8, tma_pol_ef());
      }
    }
    pdl_wait();
    if (tid == 0) {
      mbar_expect_tx(full0, ASZ1);
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
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;

  {
    float acc[4][NW][4] = {};
    int kt = 0, ti = 0;
    for (int g = 0; g < total1; ++g) {
      const int s = g & 1, n = g >> 1;
      const int t = b + ti * 48, nt = t >> 1, band = t & 1;
      const int rb0 = band * 256 + wm * 64;
      if (tid == 0) produce(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB1, wbase = abase + ASZ1;
      consume_stage_pipe_4x8<false>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++kt == NKT) {
        kt = 0; ++ti;
        const bool is1 = nt >= NTN0;
        const int col0 = (is1 ? (nt - NTN0) : nt) * BN + wn * 64;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (is1) epilogue<N1, NW>(acc[i], p.out1, rb0 + i * 16, col0, lane, sa1, sb1);
          else     epilogue<N0, NW>(acc[i], p.out0, rb0 + i * 16, col0, lane, sa0, sb0);
#pragma unroll
          for (int j = 0; j < NW; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
        }
      }
    }
  }

  const int rb0 = row0_2 + warp * 32;
  if (is112) {
    float acc2[2][14][4] = {};
    for (int g = total1; g < total_all; ++g) {
      if constexpr (kPdl) {
        if (g == kPdlTrig) pdl_launch_dependents();
      }
      const int s = g & 1, n = g >> 1;
      if (tid == 0 && g + 1 < total_all) produce(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB1, wbase = abase + ASZ1;
      consume_stage_pipe_opt<2, 14, false>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) epilogue<N1, 14>(acc2[i], p.out1, rb0 + i * 16, col0_2, lane, sa1, sb1);
  } else {
    float acc2[2][12][4] = {};
    for (int g = total1; g < total_all; ++g) {
      if constexpr (kPdl) {
        if (g == kPdlTrig) pdl_launch_dependents();
      }
      const int s = g & 1, n = g >> 1;
      if (tid == 0 && g + 1 < total_all) produce(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB1, wbase = abase + ASZ1;
      consume_stage_pipe_opt<2, 12, false>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) epilogue<N1, 12>(acc2[i], p.out1, rb0 + i * 16, col0_2, lane, sa1, sb1);
  }
}

template <int N0, int N1, int K>
__global__ void __launch_bounds__(NT, 1)
gemm_bf16_dual_stacked(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                       const ParamsBf16Dual p) {
  constexpr int ST = 2, MW = 4, BM = WM * MW * 16, NB = M / BM;
  constexpr int ASZ = BM * KT, STB = ASZ + WSZ;
  constexpr int NTN0 = N0 / BN, NTN = (N0 + N1) / BN, NTILES = NB * NTN, NKT = (2 * K) / KT;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST;
  const int grid = gridDim.x;
  if ((int)blockIdx.x >= NTILES) return;
  const int nmine = (NTILES - blockIdx.x + grid - 1) / grid;
  const int total = nmine * NKT;
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  auto produce = [&](int gl) {
    const int s = gl % ST, n = gl / ST;
    if (gl >= ST) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
    const int ti = gl / NKT, kt = gl - ti * NKT;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
    mbar_expect_tx(bar, STB);
    tma2d_hint(dst, &tmA, kt * KT, band * BM, bar, tma_pol_el());
    tma2d_hint(dst + ASZ, &tmB, kt * KT, nt * BN, bar, tma_pol_ef());
  };
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
    for (int s = 0; s < ST - 1 && s < total; ++s) produce(s);
  }
  const int wm = warp % WM, wn = warp / WM;
  float acc[MW][NW][4];
#pragma unroll
  for (int i = 0; i < MW; ++i)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  int kt = 0, ti = 0;
  for (int g = 0; g < total; ++g) {
    const int s = g % ST, n = g / ST;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const int rb0 = band * BM + wm * MW * 16;
    if (tid == 0 && g + ST - 1 < total) produce(g + ST - 1);
    mbar_wait(full0 + 8 * s, n & 1);
    const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
    consume_stage_pipe_4x8<true>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * s);
    if (++kt == NKT) {
      kt = 0;
      ++ti;
      const bool is1 = nt >= NTN0;
      const int col0 = (is1 ? (nt - NTN0) : nt) * BN + wn * NW * 8;
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        if (is1) {
          epilogue_bf16<N1>(acc[i], p.out1, rb0 + i * 16, col0, lane);
        } else {
          epilogue_bf16<N0>(acc[i], p.out0, rb0 + i * 16, col0, lane);
        }
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
      }
    }
  }
}

// Wave-29b: Seamless zero-drain 48-SM Wave-3 N-split for gemm_bf16_dual_stacked (BM=256 preserved, 32x 256x112 + 16x 256x96)
template <bool kPdl = false>
__global__ void __launch_bounds__(NT, 1)
gemm_bf16_dual_seamless(const __grid_constant__ CUtensorMap tmA,
                        const __grid_constant__ CUtensorMap tmB_128,
                        const __grid_constant__ CUtensorMap tmB_112,
                        const __grid_constant__ CUtensorMap tmB_96,
                        const ParamsBf16Dual p) {
  constexpr int N0 = 4608, N1 = 4096, K = 6656, NTN0 = 36, NKT = (2 * K) / KT;
  constexpr int ASZ1 = 256 * KT, STB1 = ASZ1 + WSZ;
  constexpr int total1 = 2 * NKT, total_all = 3 * NKT;
  constexpr int kPdlTrig = total_all - 16;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 16;

  const int band2 = b & 1, c2 = b >> 1;
  const bool is112 = (c2 < 16);
  const int row0_2 = band2 * 256;
  const int w_row0_2 = is112 ? (6144 + c2 * 112) : (7936 + (c2 - 16) * 96);
  const int col0_2 = w_row0_2 - N0;
  const int tx2 = ASZ1 + (is112 ? (112 * KT) : (96 * KT));

  if (tid == 0) {
    mbar_init(full0, 1); mbar_init(full0 + 8, 1);
    mbar_init(empty0, NWARP); mbar_init(empty0 + 8, NWARP);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();

  auto produce = [&](int gl) {
    const int s = gl & 1, n = gl >> 1;
    if (gl >= 2) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
    const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB1;
    if (gl < total1) {
      const int ti = gl / NKT, kt = gl - ti * NKT;
      const int t = b + ti * 48, nt = t >> 1, band = t & 1;
      if constexpr (kPdl) {
        if (gl == 1) {
          mbar_expect_tx(bar, ASZ1);
          tma2d_hint(dst, &tmA, kt * KT, band * 256, bar, tma_pol_el());
          return;
        }
      }
      mbar_expect_tx(bar, STB1);
      tma2d_hint(dst, &tmA, kt * KT, band * 256, bar, tma_pol_el());
      tma2d_hint(dst + ASZ1, &tmB_128, kt * KT, nt * BN, bar, tma_pol_ef());
    } else {
      const int kt = gl - total1;
      mbar_expect_tx(bar, tx2);
      tma2d_hint(dst, &tmA, kt * KT, row0_2, bar, tma_pol_el());
      if (is112) tma2d_hint(dst + ASZ1, &tmB_112, kt * KT, w_row0_2, bar, tma_pol_ef());
      else       tma2d_hint(dst + ASZ1, &tmB_96,  kt * KT, w_row0_2, bar, tma_pol_ef());
    }
  };

  if constexpr (kPdl) {
    const int nt0 = b >> 1, band0 = b & 1;
    if (tid == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_128) : "memory");
      mbar_expect_tx_only(full0, WSZ);
      tma2d_hint(sbase + ASZ1, &tmB_128, 0, nt0 * BN, full0, tma_pol_ef());
      mbar_expect_tx_only(full0 + 8, WSZ);
      tma2d_hint(sbase + STB1 + ASZ1, &tmB_128, KT, nt0 * BN, full0 + 8, tma_pol_ef());
    }
    pdl_wait();
    if (tid == 0) {
      mbar_expect_tx(full0, ASZ1);
      tma2d_hint(sbase, &tmA, 0, band0 * 256, full0, tma_pol_el());
    }
  } else if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_128) : "memory");
    produce(0);
  }
  const int wm = warp % WM, wn = warp / WM;
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;

  {
    float acc[4][NW][4] = {};
    int kt = 0, ti = 0;
    for (int g = 0; g < total1; ++g) {
      const int s = g & 1, n = g >> 1;
      const int t = b + ti * 48, nt = t >> 1, band = t & 1;
      const int rb0 = band * 256 + wm * 64;
      if (tid == 0) produce(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB1, wbase = abase + ASZ1;
      consume_stage_pipe_4x8<true>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++kt == NKT) {
        kt = 0; ++ti;
        const bool is1 = nt >= NTN0;
        const int col0 = (is1 ? (nt - NTN0) : nt) * BN + wn * 64;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (is1) epilogue_bf16<N1, NW>(acc[i], p.out1, rb0 + i * 16, col0, lane);
          else     epilogue_bf16<N0, NW>(acc[i], p.out0, rb0 + i * 16, col0, lane);
#pragma unroll
          for (int j = 0; j < NW; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
        }
      }
    }
  }

  const int rb0 = row0_2 + warp * 32;
  if (is112) {
    float acc2[2][14][4] = {};
    for (int g = total1; g < total_all; ++g) {
      if constexpr (kPdl) {
        if (g == kPdlTrig) pdl_launch_dependents();
      }
      const int s = g & 1, n = g >> 1;
      if (tid == 0 && g + 1 < total_all) produce(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB1, wbase = abase + ASZ1;
      consume_stage_pipe_opt<2, 14, true>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) epilogue_bf16<N1, 14>(acc2[i], p.out1, rb0 + i * 16, col0_2, lane);
  } else {
    float acc2[2][12][4] = {};
    for (int g = total1; g < total_all; ++g) {
      if constexpr (kPdl) {
        if (g == kPdlTrig) pdl_launch_dependents();
      }
      const int s = g & 1, n = g >> 1;
      if (tid == 0 && g + 1 < total_all) produce(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB1, wbase = abase + ASZ1;
      consume_stage_pipe_opt<2, 12, true>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) epilogue_bf16<N1, 12>(acc2[i], p.out1, rb0 + i * 16, col0_2, lane);
  }
}

// =================================================================================================
// Wave-34: 9-warp warp-specialized KT=128 ST=3 (2-stage TMA lookahead), BM=128, seamless zero-drain
// Phase 1 -> Phase 2 for FP8 o_proj (6656x4096, 42x) and FP8 qkv+gate dual (4608+4096x6656, 28x).
// Fits in 99,328 B <= 101,376 B GB10 opt-in SMEM with 4-SM lockstep L2 weight sharing and explicit
// fence.proxy.async.shared::cta across the warp-8 async TMA <-> warps-0..7 generic ldmatrix proxy.
// =================================================================================================
constexpr int NT_WS = (NWARP + 1) * 32;  // 288 threads: warps 0..7 MMA consumers, warp 8 TMA producer

template <int K, bool kPdl = false, bool kPolEf = false>
__global__ void __launch_bounds__(NT_WS, 1)
gemm_hybrid_6656_st3(const __grid_constant__ CUtensorMap tmA128,
                     const __grid_constant__ CUtensorMap tmB128,
                     const __grid_constant__ CUtensorMap tmB48,
                     const __grid_constant__ CUtensorMap tmB32,
                     const Params p) {
  constexpr int N = 6656, ST = 3, BM = 128, ASZ = BM * KT, STB = ASZ + WSZ, NKT = K / KT;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST;

  const int band2 = b & 3, c2 = b >> 2;
  const bool is48 = (c2 < 8);
  const int row0_2 = band2 * 128;
  const int col0_2 = is48 ? (6144 + c2 * 48) : (6528 + (c2 - 8) * 32);
  const int tx2 = ASZ + (is48 ? (48 * KT) : (32 * KT));

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();

  if (warp == NWARP) {
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA128) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB128) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB48) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB32) : "memory");
      int s = 0, par = 0, g = 0;
      const uint64_t pol_el = tma_pol_el(), pol_ef = tma_pol_ef();
      if constexpr (kPdl) {
        const int nt0 = b >> 2;
        const int col0_0 = nt0 * BN;
#pragma unroll
        for (int ps = 0; ps < ST; ++ps) {
          const uint32_t bar = full0 + 8 * ps, dst = sbase + ps * STB;
          mbar_expect_tx_only(bar, WSZ);
          tma2d_b<kPolEf>(dst + ASZ, &tmB128, ps * KT, col0_0, bar, pol_ef);
        }
        pdl_wait();
      }
      for (int ti = 0; ti < 4; ++ti) {
        const int t = b + ti * 48, nt = t >> 2, band = t & 3;
        const int row0 = band * BM, col0 = nt * BN;
        for (int kt = 0; kt < NKT; ++kt) {
          if (g >= ST) {
            mbar_wait(empty0 + 8 * s, par ^ 1);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          }
          const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
          if constexpr (kPdl) {
            if (g < ST) {
              mbar_expect_tx(bar, ASZ);
              tma2d_hint(dst, &tmA128, kt * KT, row0, bar, pol_el);
              ++g;
              if (++s == ST) { s = 0; par ^= 1; }
              continue;
            }
          }
          mbar_expect_tx(bar, STB);
          tma2d_hint(dst, &tmA128, kt * KT, row0, bar, pol_el);
          tma2d_b<kPolEf>(dst + ASZ, &tmB128, kt * KT, col0, bar, pol_ef);
          ++g;
          if (++s == ST) { s = 0; par ^= 1; }
        }
      }
      for (int kt = 0; kt < NKT; ++kt) {
        mbar_wait(empty0 + 8 * s, par ^ 1);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
        mbar_expect_tx(bar, tx2);
        tma2d_hint(dst, &tmA128, kt * KT, row0_2, bar, pol_el);
        if (is48) tma2d_hint(dst + ASZ, &tmB48, kt * KT, col0_2, bar, pol_el);
        else      tma2d_hint(dst + ASZ, &tmB32, kt * KT, col0_2, bar, pol_el);
        if (++s == ST) { s = 0; par ^= 1; }
      }
    }
    return;
  }

  const int wm = warp % WM, wn = warp / WM;
  const float sa = __ldg(p.sa), sb = __ldg(p.sb);
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  int s = 0, par = 0;

  {
    float acc[2][NW][4] = {};
    for (int ti = 0; ti < 4; ++ti) {
      const int t = b + ti * 48, nt = t >> 2, band = t & 3;
      for (int kt = 0; kt < NKT; ++kt) {
        mbar_wait(full0 + 8 * s, par);
        const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
        consume_stage_pipe<2, NW, false>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
        if (++s == ST) { s = 0; par ^= 1; }
      }
      const int rb0 = band * BM + wm * 32;
      const int col0 = nt * BN + wn * 64;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        epilogue<N, NW>(acc[i], p.out, rb0 + i * 16, col0, lane, sa, sb);
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
      }
    }
  }

  const int rb0 = row0_2 + warp * 16;
  constexpr int kPdlTrig = (K == 19968) ? (NKT - 20) : ((NKT > 10) ? (NKT - 10) : 0);
  if (is48) {
    float acc2[1][6][4] = {};
    for (int kt = 0; kt < NKT; ++kt) {
      if constexpr (kPdl) { if (kt == kPdlTrig) pdl_launch_dependents(); }
      mbar_wait(full0 + 8 * s, par);
      const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
      consume_stage_pipe<1, 6, false>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++s == ST) { s = 0; par ^= 1; }
    }
    epilogue<N, 6>(acc2[0], p.out, rb0, col0_2, lane, sa, sb);
  } else {
    float acc2[1][4][4] = {};
    for (int kt = 0; kt < NKT; ++kt) {
      if constexpr (kPdl) { if (kt == kPdlTrig) pdl_launch_dependents(); }
      mbar_wait(full0 + 8 * s, par);
      const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
      consume_stage_pipe<1, 4, false>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++s == ST) { s = 0; par ^= 1; }
    }
    epilogue<N, 4>(acc2[0], p.out, rb0, col0_2, lane, sa, sb);
  }
}

template <bool kPdl = false, bool kPolEf = false>
__global__ void __launch_bounds__(NT_WS, 1)
gemm_dual_a_st3(const __grid_constant__ CUtensorMap tmA0,
                const __grid_constant__ CUtensorMap tmA1,
                const __grid_constant__ CUtensorMap tmB0,
                const __grid_constant__ CUtensorMap tmB1_128,
                const __grid_constant__ CUtensorMap tmB1_96,
                const __grid_constant__ CUtensorMap tmB1_64,
                const ParamsDualA p) {
  constexpr int N0 = 4608, N1 = 4096, K = 6656, NTN0 = 36, NKT = K / KT;
  constexpr int ST = 3, BM = 128, ASZ = BM * KT, STB = ASZ + WSZ;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST;

  const int band2 = b & 3, c2 = b >> 2;
  const bool is96 = (c2 < 8);
  const int row0_2 = band2 * 128;
  const int col0_2 = is96 ? (3072 + c2 * 96) : (3840 + (c2 - 8) * 64);
  const int tx2 = ASZ + (is96 ? (96 * KT) : (64 * KT));

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();

  if (warp == NWARP) {
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA0) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA1) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB0) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB1_128) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB1_96) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB1_64) : "memory");
      int s = 0, par = 0, g = 0;
      const uint64_t pol_el = tma_pol_el(), pol_ef = tma_pol_ef();
      if constexpr (kPdl) {
        const int nt0 = b >> 2;
        const int col0_0 = nt0 * BN;
#pragma unroll
        for (int ps = 0; ps < ST; ++ps) {
          const uint32_t bar = full0 + 8 * ps, dst = sbase + ps * STB;
          mbar_expect_tx_only(bar, WSZ);
          tma2d_b<kPolEf>(dst + ASZ, &tmB0, ps * KT, col0_0, bar, pol_ef);
        }
        pdl_wait();
      }
      for (int ti = 0; ti < 5; ++ti) {
        const int t = b + ti * 48, nt = t >> 2, band = t & 3;
        const int row0 = band * BM;
        const bool is1 = nt >= NTN0;
        const int col0 = (is1 ? (nt - NTN0) : nt) * BN;
        const CUtensorMap* mapA = is1 ? &tmA1 : &tmA0;
        const CUtensorMap* mapB = is1 ? &tmB1_128 : &tmB0;
        for (int kt = 0; kt < NKT; ++kt) {
          if (g >= ST) {
            mbar_wait(empty0 + 8 * s, par ^ 1);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          }
          const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
          if constexpr (kPdl) {
            if (g < ST) {
              mbar_expect_tx(bar, ASZ);
              tma2d_hint(dst, mapA, kt * KT, row0, bar, pol_el);
              ++g;
              if (++s == ST) { s = 0; par ^= 1; }
              continue;
            }
          }
          mbar_expect_tx(bar, STB);
          tma2d_hint(dst, mapA, kt * KT, row0, bar, pol_el);
          tma2d_b<kPolEf>(dst + ASZ, mapB, kt * KT, col0, bar, pol_ef);
          ++g;
          if (++s == ST) { s = 0; par ^= 1; }
        }
      }
      const CUtensorMap* mapB2 = is96 ? &tmB1_96 : &tmB1_64;
      for (int kt = 0; kt < NKT; ++kt) {
        mbar_wait(empty0 + 8 * s, par ^ 1);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
        mbar_expect_tx(bar, tx2);
        tma2d_hint(dst, &tmA1, kt * KT, row0_2, bar, pol_el);
        tma2d_b<kPolEf>(dst + ASZ, mapB2, kt * KT, col0_2, bar, pol_ef);
        if (++s == ST) { s = 0; par ^= 1; }
      }
    }
    return;
  }

  const int wm = warp % WM, wn = warp / WM;
  const float sa0 = __ldg(p.sa0), sa1 = __ldg(p.sa1), sb0 = __ldg(p.sb0), sb1 = __ldg(p.sb1);
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  int s = 0, par = 0;

  {
    float acc[2][NW][4] = {};
    for (int ti = 0; ti < 5; ++ti) {
      const int t = b + ti * 48, nt = t >> 2, band = t & 3;
      for (int kt = 0; kt < NKT; ++kt) {
        mbar_wait(full0 + 8 * s, par);
        const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
        consume_stage_pipe<2, NW, false>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
        if (++s == ST) { s = 0; par ^= 1; }
      }
      const bool is1 = nt >= NTN0;
      const int rb0 = band * BM + wm * 32;
      const int col0 = (is1 ? (nt - NTN0) : nt) * BN + wn * 64;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        if (is1) epilogue<N1, NW>(acc[i], p.out1, rb0 + i * 16, col0, lane, sa1, sb1);
        else     epilogue<N0, NW>(acc[i], p.out0, rb0 + i * 16, col0, lane, sa0, sb0);
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
      }
    }
  }

  const int rb0 = row0_2 + wm * 32;
  constexpr int kPdlTrig = NKT - 12;
  if (is96) {
    float acc2[2][6][4] = {};
    for (int kt = 0; kt < NKT; ++kt) {
      if constexpr (kPdl) { if (kt == kPdlTrig) pdl_launch_dependents(); }
      mbar_wait(full0 + 8 * s, par);
      const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
      consume_stage_pipe<2, 6, false>(abase, wbase, wm, wn, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++s == ST) { s = 0; par ^= 1; }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) epilogue<N1, 6>(acc2[i], p.out1, rb0 + i * 16, col0_2 + wn * 48, lane, sa1, sb1);
  } else {
    float acc2[2][4][4] = {};
    for (int kt = 0; kt < NKT; ++kt) {
      if constexpr (kPdl) { if (kt == kPdlTrig) pdl_launch_dependents(); }
      mbar_wait(full0 + 8 * s, par);
      const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
      consume_stage_pipe<2, 4, false>(abase, wbase, wm, wn, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++s == ST) { s = 0; par ^= 1; }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) epilogue<N1, 4>(acc2[i], p.out1, rb0 + i * 16, col0_2 + wn * 32, lane, sa1, sb1);
  }
}

static bool g_st3_bf16 = []() {
  const char* env = std::getenv("FP8WS_ST3_BF16");
  return !env || !*env || std::strtol(env, nullptr, 10) != 0;
}();

template <bool kPdl = false, bool kPolEf = false>
__global__ void __launch_bounds__(NT_WS, 1)
gemm_bf16_dual_st3(const __grid_constant__ CUtensorMap tmA,
                   const __grid_constant__ CUtensorMap tmB_128,
                   const __grid_constant__ CUtensorMap tmB_96,
                   const __grid_constant__ CUtensorMap tmB_64,
                   const ParamsBf16Dual p) {
  constexpr int N0 = 4608, N1 = 4096, K = 6656, NTN0 = 36, NKT = (2 * K) / KT;
  constexpr int ST = 3, BM = 128, ASZ = BM * KT, STB = ASZ + WSZ;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST;

  const int band2 = b & 3, c2 = b >> 2;
  const bool is96 = (c2 < 8);
  const int row0_2 = band2 * 128;
  const int w_row0_2 = is96 ? (7680 + c2 * 96) : (8448 + (c2 - 8) * 64);
  const int col0_2 = w_row0_2 - N0;
  const int tx2 = ASZ + (is96 ? (96 * KT) : (64 * KT));

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();

  if (warp == NWARP) {
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_128) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_96) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_64) : "memory");
      int s = 0, par = 0, g = 0;
      const uint64_t pol_el = tma_pol_el(), pol_ef = tma_pol_ef();
      if constexpr (kPdl) {
        const int nt0 = b >> 2;
        const int wrow0_0 = nt0 * BN;
#pragma unroll
        for (int ps = 0; ps < ST; ++ps) {
          const uint32_t bar = full0 + 8 * ps, dst = sbase + ps * STB;
          mbar_expect_tx_only(bar, WSZ);
          tma2d_b<kPolEf>(dst + ASZ, &tmB_128, ps * KT, wrow0_0, bar, pol_ef);
        }
        pdl_wait();
      }
      for (int ti = 0; ti < 5; ++ti) {
        const int t = b + ti * 48, nt = t >> 2, band = t & 3;
        const int row0 = band * BM;
        const int wrow0 = nt * BN;
        for (int kt = 0; kt < NKT; ++kt) {
          if (g >= ST) {
            mbar_wait(empty0 + 8 * s, par ^ 1);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          }
          const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
          if constexpr (kPdl) {
            if (g < ST) {
              mbar_expect_tx(bar, ASZ);
              tma2d_hint(dst, &tmA, kt * KT, row0, bar, pol_el);
              ++g;
              if (++s == ST) { s = 0; par ^= 1; }
              continue;
            }
          }
          mbar_expect_tx(bar, STB);
          tma2d_hint(dst, &tmA, kt * KT, row0, bar, pol_el);
          tma2d_b<kPolEf>(dst + ASZ, &tmB_128, kt * KT, wrow0, bar, pol_ef);
          ++g;
          if (++s == ST) { s = 0; par ^= 1; }
        }
      }
      const CUtensorMap* mapB2 = is96 ? &tmB_96 : &tmB_64;
      for (int kt = 0; kt < NKT; ++kt) {
        mbar_wait(empty0 + 8 * s, par ^ 1);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
        mbar_expect_tx(bar, tx2);
        tma2d_hint(dst, &tmA, kt * KT, row0_2, bar, pol_el);
        tma2d_b<kPolEf>(dst + ASZ, mapB2, kt * KT, w_row0_2, bar, pol_ef);
        if (++s == ST) { s = 0; par ^= 1; }
      }
    }
    return;
  }

  const int wm = warp % WM, wn = warp / WM;
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  int s = 0, par = 0;

  {
    float acc[2][NW][4] = {};
    for (int ti = 0; ti < 5; ++ti) {
      const int t = b + ti * 48, nt = t >> 2, band = t & 3;
      for (int kt = 0; kt < NKT; ++kt) {
        mbar_wait(full0 + 8 * s, par);
        const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
        consume_stage_pipe<2, NW, true>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
        if (++s == ST) { s = 0; par ^= 1; }
      }
      const bool is1 = nt >= NTN0;
      const int rb0 = band * BM + wm * 32;
      const int col0 = (is1 ? (nt - NTN0) : nt) * BN + wn * 64;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        if (is1) epilogue_bf16<N1, NW>(acc[i], p.out1, rb0 + i * 16, col0, lane);
        else     epilogue_bf16<N0, NW>(acc[i], p.out0, rb0 + i * 16, col0, lane);
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
      }
    }
  }

  const int rb0 = row0_2 + wm * 32;
  constexpr int kPdlTrig = NKT - 12;
  if (is96) {
    float acc2[2][6][4] = {};
    for (int kt = 0; kt < NKT; ++kt) {
      if constexpr (kPdl) { if (kt == kPdlTrig) pdl_launch_dependents(); }
      mbar_wait(full0 + 8 * s, par);
      const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
      consume_stage_pipe<2, 6, true>(abase, wbase, wm, wn, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++s == ST) { s = 0; par ^= 1; }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) epilogue_bf16<N1, 6>(acc2[i], p.out1, rb0 + i * 16, col0_2 + wn * 48, lane);
  } else {
    float acc2[2][4][4] = {};
    for (int kt = 0; kt < NKT; ++kt) {
      if constexpr (kPdl) { if (kt == kPdlTrig) pdl_launch_dependents(); }
      mbar_wait(full0 + 8 * s, par);
      const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
      consume_stage_pipe<2, 4, true>(abase, wbase, wm, wn, ar, ach, br, bch, acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++s == ST) { s = 0; par ^= 1; }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) epilogue_bf16<N1, 4>(acc2[i], p.out1, rb0 + i * 16, col0_2 + wn * 32, lane);
  }
}

template <int K, bool kPdl>
__global__ void __launch_bounds__(NT, 1)
gemm_6656_ws8_256_st2_st4(const __grid_constant__ CUtensorMap tmA256,
                          const __grid_constant__ CUtensorMap tmA128,
                          const __grid_constant__ CUtensorMap tmB128,
                          const __grid_constant__ CUtensorMap tmB48,
                          const __grid_constant__ CUtensorMap tmB32,
                          const Params p) {
  constexpr int N_OUT = 6656, ST1 = 2, BM1 = 256, ASZ1 = BM1 * KT, STB1 = ASZ1 + WSZ, NKT = K / KT, TOTAL1 = 2 * NKT;
  constexpr int ST2 = 4, ASZ2 = 128 * KT, WSZ2 = 48 * KT, STB2 = ASZ2 + WSZ2;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST2];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST2;
  const int wm = warp % WM, wn = warp / WM;
  const float sa = __ldg(p.sa), sb = __ldg(p.sb);
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
      if (gl >= ST1) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
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
      consume_stage_pipe_4x8(abase, wbase, wm, wn, ar, ach, br, bch, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++kt == NKT) {
        const int t = b + ti * 48, nt = t >> 1, band = t & 1;
        kt = 0; ++ti;
        const int rb0 = band * BM1 + wm * 64;
        const int col0 = nt * BN + wn * 64;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          epilogue<N_OUT, 8>(acc[i], p.out, rb0 + i * 16, col0, lane, sa, sb);
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
      if (gl >= ST2) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
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
        consume_stage_pipe_opt<1, 6>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
      }
      epilogue<N_OUT, 6>(acc2[0], p.out, rb0, col0_2, lane, sa, sb);
    } else {
      float acc2[1][4][4] = {};
      for (int g = 0; g < NKT; ++g) {
        if constexpr (kPdl) { if (g == kPdlTrig) pdl_launch_dependents(); }
        const int s = g & 3, n = g >> 2;
        if (tid == 0 && g + ST2 - 1 < NKT) produce2(g + ST2 - 1);
        mbar_wait(full0 + 8 * s, n & 1);
        const uint32_t abase = sbase + s * STB2, wbase = abase + ASZ2;
        consume_stage_pipe_opt<1, 4>(abase, wbase, warp, 0, ar, ach, br, bch, acc2);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
      }
      epilogue<N_OUT, 4>(acc2[0], p.out, rb0, col0_2, lane, sa, sb);
    }
  }
}

// Wave-81: 8-warp BM=256 ST1=2 (6144 cols = 70.6%) + seamless BM=128 ST2=3 (1536 + 1024 cols) for 28x FP8 dual_a
template <bool kPdl>
__global__ void __launch_bounds__(NT, 1)
gemm_dual_a_ws8_256_st2_st3(const __grid_constant__ CUtensorMap tmA0_256,
                            const __grid_constant__ CUtensorMap tmA1_256,
                            const __grid_constant__ CUtensorMap tmA1_128,
                            const __grid_constant__ CUtensorMap tmB0,
                            const __grid_constant__ CUtensorMap tmB1_128,
                            const __grid_constant__ CUtensorMap tmB1_96,
                            const __grid_constant__ CUtensorMap tmB1_64,
                            const ParamsDualA p) {
  constexpr int N0 = 4608, N1 = 4096, K = 6656, NTN0 = 36, NKT = K / KT;
  constexpr int ST1 = 2, BM1 = 256, ASZ1 = BM1 * KT, STB1 = ASZ1 + WSZ, TOTAL1 = 2 * NKT;
  constexpr int ST2 = 3, BM2 = 128, ASZ2 = BM2 * KT, STB2 = ASZ2 + WSZ, TOTAL2 = 2 * NKT;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST2];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST2;
  const int wm = warp % WM, wn = warp / WM;
  const float sa0 = __ldg(p.sa0), sa1 = __ldg(p.sa1), sb0 = __ldg(p.sb0), sb1 = __ldg(p.sb1);
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
      if (gl >= ST1) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
      const int ti = gl / NKT, kt = gl - ti * NKT;
      const int t = b + ti * 48, nt = t >> 1, band = t & 1;
      const bool is1 = nt >= NTN0;
      const int col0 = (is1 ? (nt - NTN0) : nt) * BN;
      const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB1;
      if constexpr (kPdl) {
        if (gl == 1) {
          mbar_expect_tx(bar, ASZ1);
          tma2d_hint(dst, &tmA0_256, kt * KT, band * BM1, bar, tma_pol_el());
          return;
        }
      }
      mbar_expect_tx(bar, STB1);
      tma2d_hint(dst, is1 ? &tmA1_256 : &tmA0_256, kt * KT, band * BM1, bar, tma_pol_el());
      tma2d_hint(dst + ASZ1, is1 ? &tmB1_128 : &tmB0, kt * KT, col0, bar, tma_pol_ef());
    };
    if constexpr (kPdl) {
      const int nt0 = b >> 1, band0 = b & 1;
      if (tid == 0) {
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA0_256) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA1_256) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA1_128) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB0) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB1_128) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB1_96) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB1_64) : "memory");
        mbar_expect_tx_only(full0, WSZ);
        tma2d_hint(sbase + ASZ1, &tmB0, 0, nt0 * BN, full0, tma_pol_ef());
        mbar_expect_tx_only(full0 + 8, WSZ);
        tma2d_hint(sbase + STB1 + ASZ1, &tmB0, KT, nt0 * BN, full0 + 8, tma_pol_ef());
      }
      pdl_wait();
      if (tid == 0) {
        mbar_expect_tx(full0, ASZ1);
        tma2d_hint(sbase, &tmA0_256, 0, band0 * BM1, full0, tma_pol_el());
      }
    } else if (tid == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA0_256) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA1_256) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA1_128) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB0) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB1_128) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB1_96) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB1_64) : "memory");
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
        const bool is1 = nt >= NTN0;
        const int rb0 = band * BM1 + wm * 64;
        const int col0 = (is1 ? (nt - NTN0) : nt) * BN + wn * 64;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (is1) epilogue<N1, 8>(acc[i], p.out1, rb0 + i * 16, col0, lane, sa1, sb1);
          else     epilogue<N0, 8>(acc[i], p.out0, rb0 + i * 16, col0, lane, sa0, sb0);
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
    const bool is96 = (c2 < 8);
    const int row0_2 = band2 * BM2;
    const int col_2a = 1536 + c2 * 128;
    const int col_2b = is96 ? (3072 + c2 * 96) : (3840 + (c2 - 8) * 64);
    const int tx_2b = ASZ2 + (is96 ? (96 * KT) : (64 * KT));
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < ST2; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    auto produce2 = [&](int gl) {
      const int s = gl % ST2, n = gl / ST2;
      if (gl >= ST2) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
      const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB2;
      if (gl < NKT) {
        mbar_expect_tx(bar, STB2);
        tma2d_hint(dst, &tmA1_128, gl * KT, row0_2, bar, tma_pol_el());
        tma2d_hint(dst + ASZ2, &tmB1_128, gl * KT, col_2a, bar, tma_pol_ef());
      } else {
        const int kt = gl - NKT;
        mbar_expect_tx(bar, tx_2b);
        tma2d_hint(dst, &tmA1_128, kt * KT, row0_2, bar, tma_pol_el());
        tma2d_hint(dst + ASZ2, is96 ? &tmB1_96 : &tmB1_64, kt * KT, col_2b, bar, tma_pol_ef());
      }
    };
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < ST2 - 1; ++s) produce2(s);
    }
    const int rb0 = row0_2 + wm * 32;
    {
      float acc2a[2][8][4] = {};
      for (int g = 0; g < NKT; ++g) {
        const int s = g % ST2, n = g / ST2;
        if (tid == 0) produce2(g + ST2 - 1);
        mbar_wait(full0 + 8 * s, n & 1);
        const uint32_t abase = sbase + s * STB2, wbase = abase + ASZ2;
        consume_stage_pipe_opt<2, 8, false>(abase, wbase, wm, wn, ar, ach, br, bch, acc2a);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) epilogue<N1, 8>(acc2a[i], p.out1, rb0 + i * 16, col_2a + wn * 64, lane, sa1, sb1);
    }
    constexpr int kPdlTrig = TOTAL2 - 12;
    if (is96) {
      float acc2b[2][6][4] = {};
      for (int g = NKT; g < TOTAL2; ++g) {
        if constexpr (kPdl) { if (g == kPdlTrig) pdl_launch_dependents(); }
        const int s = g % ST2, n = g / ST2;
        if (tid == 0 && g + ST2 - 1 < TOTAL2) produce2(g + ST2 - 1);
        mbar_wait(full0 + 8 * s, n & 1);
        const uint32_t abase = sbase + s * STB2, wbase = abase + ASZ2;
        consume_stage_pipe_opt<2, 6, false>(abase, wbase, wm, wn, ar, ach, br, bch, acc2b);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) epilogue<N1, 6>(acc2b[i], p.out1, rb0 + i * 16, col_2b + wn * 48, lane, sa1, sb1);
    } else {
      float acc2b[2][4][4] = {};
      for (int g = NKT; g < TOTAL2; ++g) {
        if constexpr (kPdl) { if (g == kPdlTrig) pdl_launch_dependents(); }
        const int s = g % ST2, n = g / ST2;
        if (tid == 0 && g + ST2 - 1 < TOTAL2) produce2(g + ST2 - 1);
        mbar_wait(full0 + 8 * s, n & 1);
        const uint32_t abase = sbase + s * STB2, wbase = abase + ASZ2;
        consume_stage_pipe_opt<2, 4, false>(abase, wbase, wm, wn, ar, ach, br, bch, acc2b);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) epilogue<N1, 4>(acc2b[i], p.out1, rb0 + i * 16, col_2b + wn * 32, lane, sa1, sb1);
    }
  }
}

// Wave-81: 8-warp BM=256 ST1=2 + seamless BM=128 ST2=3 for 18x BF16 dual_stacked
template <bool kPdl>
__global__ void __launch_bounds__(NT, 1)
gemm_bf16_dual_ws8_256_st2_st3(const __grid_constant__ CUtensorMap tmA256,
                               const __grid_constant__ CUtensorMap tmA128,
                               const __grid_constant__ CUtensorMap tmB_128,
                               const __grid_constant__ CUtensorMap tmB_96,
                               const __grid_constant__ CUtensorMap tmB_64,
                               const ParamsBf16Dual p) {
  constexpr int N0 = 4608, N1 = 4096, K = 6656, NTN0 = 36, NKT = (2 * K) / KT;
  constexpr int ST1 = 2, BM1 = 256, ASZ1 = BM1 * KT, STB1 = ASZ1 + WSZ, TOTAL1 = 2 * NKT;
  constexpr int ST2 = 3, BM2 = 128, ASZ2 = BM2 * KT, STB2 = ASZ2 + WSZ, TOTAL2 = 2 * NKT;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST2];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST2;
  const int wm = warp % WM, wn = warp / WM;
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
      if (gl >= ST1) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
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
      tma2d_hint(dst + ASZ1, &tmB_128, kt * KT, nt * BN, bar, tma_pol_ef());
    };
    if constexpr (kPdl) {
      const int nt0 = b >> 1, band0 = b & 1;
      if (tid == 0) {
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA256) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA128) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_128) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_96) : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_64) : "memory");
        mbar_expect_tx_only(full0, WSZ);
        tma2d_hint(sbase + ASZ1, &tmB_128, 0, nt0 * BN, full0, tma_pol_ef());
        mbar_expect_tx_only(full0 + 8, WSZ);
        tma2d_hint(sbase + STB1 + ASZ1, &tmB_128, KT, nt0 * BN, full0 + 8, tma_pol_ef());
      }
      pdl_wait();
      if (tid == 0) {
        mbar_expect_tx(full0, ASZ1);
        tma2d_hint(sbase, &tmA256, 0, band0 * BM1, full0, tma_pol_el());
      }
    } else if (tid == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA256) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA128) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_128) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_96) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_64) : "memory");
      produce1(0);
    }
    float acc[4][8][4] = {};
    int kt = 0, ti = 0;
    for (int g = 0; g < TOTAL1; ++g) {
      const int s = g & 1, n = g >> 1;
      if (tid == 0 && g + 1 < TOTAL1) produce1(g + 1);
      mbar_wait(full0 + 8 * s, n & 1);
      const uint32_t abase = sbase + s * STB1, wbase = abase + ASZ1;
      consume_stage_pipe_4x8<true>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++kt == NKT) {
        const int t = b + ti * 48, nt = t >> 1, band = t & 1;
        kt = 0; ++ti;
        const bool is1 = nt >= NTN0;
        const int rb0 = band * BM1 + wm * 64;
        const int col0 = (is1 ? (nt - NTN0) : nt) * BN + wn * 64;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (is1) epilogue_bf16<N1, 8>(acc[i], p.out1, rb0 + i * 16, col0, lane);
          else     epilogue_bf16<N0, 8>(acc[i], p.out0, rb0 + i * 16, col0, lane);
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
    const bool is96 = (c2 < 8);
    const int row0_2 = band2 * BM2;
    const int w_row_2a = 6144 + c2 * 128;
    const int col_2a = 1536 + c2 * 128;
    const int w_row_2b = is96 ? (7680 + c2 * 96) : (8448 + (c2 - 8) * 64);
    const int col_2b = w_row_2b - N0;
    const int tx_2b = ASZ2 + (is96 ? (96 * KT) : (64 * KT));
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < ST2; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    auto produce2 = [&](int gl) {
      const int s = gl % ST2, n = gl / ST2;
      if (gl >= ST2) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
      const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB2;
      if (gl < NKT) {
        mbar_expect_tx(bar, STB2);
        tma2d_hint(dst, &tmA128, gl * KT, row0_2, bar, tma_pol_el());
        tma2d_hint(dst + ASZ2, &tmB_128, gl * KT, w_row_2a, bar, tma_pol_ef());
      } else {
        const int kt = gl - NKT;
        mbar_expect_tx(bar, tx_2b);
        tma2d_hint(dst, &tmA128, kt * KT, row0_2, bar, tma_pol_el());
        tma2d_hint(dst + ASZ2, is96 ? &tmB_96 : &tmB_64, kt * KT, w_row_2b, bar, tma_pol_ef());
      }
    };
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < ST2 - 1; ++s) produce2(s);
    }
    const int rb0 = row0_2 + wm * 32;
    {
      float acc2a[2][8][4] = {};
      for (int g = 0; g < NKT; ++g) {
        const int s = g % ST2, n = g / ST2;
        if (tid == 0) produce2(g + ST2 - 1);
        mbar_wait(full0 + 8 * s, n & 1);
        const uint32_t abase = sbase + s * STB2, wbase = abase + ASZ2;
        consume_stage_pipe_opt<2, 8, true>(abase, wbase, wm, wn, ar, ach, br, bch, acc2a);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) epilogue_bf16<N1, 8>(acc2a[i], p.out1, rb0 + i * 16, col_2a + wn * 64, lane);
    }
    constexpr int kPdlTrig = TOTAL2 - 12;
    if (is96) {
      float acc2b[2][6][4] = {};
      for (int g = NKT; g < TOTAL2; ++g) {
        if constexpr (kPdl) { if (g == kPdlTrig) pdl_launch_dependents(); }
        const int s = g % ST2, n = g / ST2;
        if (tid == 0 && g + ST2 - 1 < TOTAL2) produce2(g + ST2 - 1);
        mbar_wait(full0 + 8 * s, n & 1);
        const uint32_t abase = sbase + s * STB2, wbase = abase + ASZ2;
        consume_stage_pipe_opt<2, 6, true>(abase, wbase, wm, wn, ar, ach, br, bch, acc2b);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) epilogue_bf16<N1, 6>(acc2b[i], p.out1, rb0 + i * 16, col_2b + wn * 48, lane);
    } else {
      float acc2b[2][4][4] = {};
      for (int g = NKT; g < TOTAL2; ++g) {
        if constexpr (kPdl) { if (g == kPdlTrig) pdl_launch_dependents(); }
        const int s = g % ST2, n = g / ST2;
        if (tid == 0 && g + ST2 - 1 < TOTAL2) produce2(g + ST2 - 1);
        mbar_wait(full0 + 8 * s, n & 1);
        const uint32_t abase = sbase + s * STB2, wbase = abase + ASZ2;
        consume_stage_pipe_opt<2, 4, true>(abase, wbase, wm, wn, ar, ach, br, bch, acc2b);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) epilogue_bf16<N1, 4>(acc2b[i], p.out1, rb0 + i * 16, col_2b + wn * 32, lane);
    }
  }
}

static int g_fp8ws_pdl_override = -1;
static bool fp8ws_pdl_enabled() {
  if (g_fp8ws_pdl_override >= 0) return g_fp8ws_pdl_override != 0;
  static const bool env_on = []() {
    const char* env = std::getenv("FP8WS_PDL");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  return env_on;
}
static bool fp8ws_tmab_ef() {
  static const bool env_on = []() {
    const char* env = std::getenv("MUSE_FP8_TMAB_EF");
    return env && *env && std::strtol(env, nullptr, 10) != 0;
  }();
  return env_on;
}

template <typename Kernel, typename... Args>
static cudaError_t launch_ex_pdl(Kernel kernel, int blocks, int threads, size_t smem, cudaStream_t st,
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

typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                             const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                             CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn encode = nullptr;
static int num_sms = 0;
static bool g_nsplit_6656 = true;
static bool g_dual_seamless = true;
static bool g_bf16_seamless = true;
static bool g_st3_oproj = true;
static bool g_st3_down = true;
static bool g_st3_dual = true;

static bool encode_tma2d(CUtensorMap* tm, const void* ptr, int k, int rows, int box_rows) {
  cuuint64_t dims[2] = {(cuuint64_t)k, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)k};
  cuuint32_t box[2] = {(cuuint32_t)KT, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1, 1};
  return encode(tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims, strides, box, es,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <int N, int K>
static cudaError_t launch_shape(const CUtensorMap& tmA, const CUtensorMap& tmB, const Params& p, cudaStream_t st) {
  using S = Shape<N, K>;
  constexpr int T = S::NTILES;
  const int grid = num_sms < T ? num_sms : T;
  if (fp8ws_pdl_enabled()) {
    return launch_ex_pdl(gemm<N, K, true>, grid, NT, S::SMEM, st, tmA, tmB, p);
  }
  gemm<N, K, false><<<grid, NT, S::SMEM, st>>>(tmA, tmB, p);
  return cudaGetLastError();
}

template <int K>
static cudaError_t launch_hybrid_6656(const CUtensorMap& tmA256, const CUtensorMap& tmA64, const CUtensorMap& tmB,
                                      const Params& p, cudaStream_t st) {
  using S = Shape<6656, K>;
  if (num_sms == 48) {
    gemm_hybrid_6656<K><<<48, NT, S::SMEM, st>>>(tmA256, tmA64, tmB, p);
    return cudaGetLastError();
  }
  return launch_shape<6656, K>(tmA256, tmB, p, st);
}

template <int N0, int N1, int K>
static cudaError_t launch_dual(const CUtensorMap& tmA, const CUtensorMap& tmB0, const CUtensorMap& tmB1,
                               const ParamsDual& p, cudaStream_t st) {
  using S = ShapeDual256<N0, N1, K>;
  constexpr int T = S::NTILES;
  gemm_dual<N0, N1, K><<<num_sms < T ? num_sms : T, NT, S::SMEM, st>>>(tmA, tmB0, tmB1, p);
  return cudaGetLastError();
}
}  // namespace fp8ws

extern "C" void fp8ws_set_flags(int nsplit_6656, int dual_seamless, int bf16_seamless) {
  fp8ws::g_nsplit_6656 = (nsplit_6656 != 0);
  fp8ws::g_dual_seamless = (dual_seamless != 0);
  fp8ws::g_bf16_seamless = (bf16_seamless != 0);
}

extern "C" void fp8ws_set_st3(int st3_oproj, int st3_dual) {
  fp8ws::g_st3_oproj = (st3_oproj != 0);
  fp8ws::g_st3_dual = (st3_dual != 0);
}

extern "C" int fp8ws_set_pdl(int on) {
  const int old = fp8ws::fp8ws_pdl_enabled() ? 1 : 0;
  fp8ws::g_fp8ws_pdl_override = (on < 0) ? -1 : (on != 0 ? 1 : 0);
  return old;
}

extern "C" int fp8ws_init() {
  using namespace fp8ws;
  if (!encode) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
    if (e != cudaSuccess || !fn || q != cudaDriverEntryPointSuccess) return e != cudaSuccess ? (int)e : -2;
    encode = (EncodeFn)fn;
  }
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e == cudaSuccess) e = cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
#define FP8WS_ATTR(N, K)                                                                                       \
  if (e == cudaSuccess)                                                                                        \
    e = cudaFuncSetAttribute(gemm<N, K, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Shape<N, K>::SMEM); \
  if (e == cudaSuccess)                                                                                        \
    e = cudaFuncSetAttribute(gemm<N, K, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, Shape<N, K>::SMEM)
  FP8WS_ATTR(4608, 6656);
  FP8WS_ATTR(4096, 6656);
  FP8WS_ATTR(6656, 4096);
  FP8WS_ATTR(39936, 6656);
  FP8WS_ATTR(6656, 19968);
#undef FP8WS_ATTR
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_hybrid_6656<4096>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             Shape<6656, 4096>::SMEM);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_hybrid_6656<19968>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             Shape<6656, 19968>::SMEM);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_hybrid_6656_nsplit128<4096, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_hybrid_6656_nsplit128<4096, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_hybrid_6656_nsplit128<19968, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_hybrid_6656_nsplit128<19968, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_dual<4608, 4096, 6656>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             ShapeDual256<4608, 4096, 6656>::SMEM);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_dual_a<4608, 4096, 6656, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             2 * ShapeDual256<4608, 4096, 6656>::STB + 1024);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_dual_a_seamless<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_dual_a_seamless<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_bf16_dual_stacked<4608, 4096, 6656>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * 49152 + 1024);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_bf16_dual_seamless<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_bf16_dual_seamless<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_6656_ws8_256_st2_st4<4096, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_6656_ws8_256_st2_st4<4096, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_6656_ws8_256_st2_st4<19968, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_6656_ws8_256_st2_st4<19968, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_dual_a_ws8_256_st2_st3<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_dual_a_ws8_256_st2_st3<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_bf16_dual_ws8_256_st2_st3<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_bf16_dual_ws8_256_st2_st3<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_hybrid_6656_st3<4096, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_hybrid_6656_st3<4096, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_hybrid_6656_st3<4096, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_hybrid_6656_st3<4096, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_hybrid_6656_st3<19968, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_hybrid_6656_st3<19968, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_hybrid_6656_st3<19968, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_hybrid_6656_st3<19968, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_dual_a_st3<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_dual_a_st3<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_dual_a_st3<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_dual_a_st3<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_bf16_dual_st3<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_bf16_dual_st3<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_bf16_dual_st3<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_bf16_dual_st3<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  }
  return (int)e;
}

extern "C" int fp8ws_run(const void* a, const void* b, const void* sa, const void* sb, void* c, int m, int n,
                         int k, void* stream) {
  using namespace fp8ws;
  if (m != M || !encode || num_sms <= 0) return (int)cudaErrorInvalidValue;
  if (((uintptr_t)a & 15) || ((uintptr_t)b & 15) || ((uintptr_t)c & 15)) return (int)cudaErrorInvalidValue;
  if (n == 6656 && (k == 4096 || k == 19968)) {
    Params p{(const float*)sa, (const float*)sb, (__nv_bfloat16*)c, (const uint8_t*)b, (size_t)n * (size_t)k};
    cudaStream_t st = (cudaStream_t)stream;
    if (num_sms == 48 && g_nsplit_6656) {
      static const bool bm256_oproj = []() {
        const char* env = std::getenv("FP8WS_BM256_OPROJ");
        return !env || !*env || std::strtol(env, nullptr, 10) != 0;
      }();
      static const bool bm256_down = []() {
        const char* env = std::getenv("FP8WS_BM256_DOWN");
        return !env || !*env || std::strtol(env, nullptr, 10) != 0;
      }();
      if ((k == 4096 && bm256_oproj) || (k == 19968 && bm256_down)) {
        CUtensorMap tmA256, tmA128, tmB128, tmB48, tmB32;
        if (!encode_tma2d(&tmA256, a, k, M, 256) || !encode_tma2d(&tmA128, a, k, M, 128)) return -3;
        if (!encode_tma2d(&tmB128, b, k, n, 128) || !encode_tma2d(&tmB48, b, k, n, 48) ||
            !encode_tma2d(&tmB32, b, k, n, 32)) return -4;
        if (fp8ws_pdl_enabled()) {
          if (k == 19968) {
            return (int)launch_ex_pdl(gemm_6656_ws8_256_st2_st4<19968, true>, 48, NT, SMEM_MAX, st,
                                      tmA256, tmA128, tmB128, tmB48, tmB32, p);
          }
          return (int)launch_ex_pdl(gemm_6656_ws8_256_st2_st4<4096, true>, 48, NT, SMEM_MAX, st,
                                    tmA256, tmA128, tmB128, tmB48, tmB32, p);
        }
        if (k == 19968) {
          gemm_6656_ws8_256_st2_st4<19968, false><<<48, NT, SMEM_MAX, st>>>(tmA256, tmA128, tmB128, tmB48, tmB32, p);
        } else {
          gemm_6656_ws8_256_st2_st4<4096, false><<<48, NT, SMEM_MAX, st>>>(tmA256, tmA128, tmB128, tmB48, tmB32, p);
        }
        return (int)cudaGetLastError();
      }
      if ((k == 4096 && g_st3_oproj) || (k == 19968 && g_st3_down)) {
        CUtensorMap tmA128, tmB128, tmB48, tmB32;
        if (!encode_tma2d(&tmA128, a, k, M, 128)) return -3;
        if (!encode_tma2d(&tmB128, b, k, n, 128) || !encode_tma2d(&tmB48, b, k, n, 48) ||
            !encode_tma2d(&tmB32, b, k, n, 32)) return -4;
        const bool ef = fp8ws_tmab_ef();
        if (fp8ws_pdl_enabled()) {
          if (k == 19968) {
            return ef ? (int)launch_ex_pdl(gemm_hybrid_6656_st3<19968, true, true>, 48, NT_WS, SMEM_MAX, st, tmA128, tmB128, tmB48, tmB32, p)
                      : (int)launch_ex_pdl(gemm_hybrid_6656_st3<19968, true, false>, 48, NT_WS, SMEM_MAX, st, tmA128, tmB128, tmB48, tmB32, p);
          }
          return ef ? (int)launch_ex_pdl(gemm_hybrid_6656_st3<4096, true, true>, 48, NT_WS, SMEM_MAX, st, tmA128, tmB128, tmB48, tmB32, p)
                    : (int)launch_ex_pdl(gemm_hybrid_6656_st3<4096, true, false>, 48, NT_WS, SMEM_MAX, st, tmA128, tmB128, tmB48, tmB32, p);
        }
        if (k == 19968) {
          if (ef) gemm_hybrid_6656_st3<19968, false, true><<<48, NT_WS, SMEM_MAX, st>>>(tmA128, tmB128, tmB48, tmB32, p);
          else    gemm_hybrid_6656_st3<19968, false, false><<<48, NT_WS, SMEM_MAX, st>>>(tmA128, tmB128, tmB48, tmB32, p);
        } else {
          if (ef) gemm_hybrid_6656_st3<4096, false, true><<<48, NT_WS, SMEM_MAX, st>>>(tmA128, tmB128, tmB48, tmB32, p);
          else    gemm_hybrid_6656_st3<4096, false, false><<<48, NT_WS, SMEM_MAX, st>>>(tmA128, tmB128, tmB48, tmB32, p);
        }
        return (int)cudaGetLastError();
      }
      CUtensorMap tmA256, tmA128, tmB128, tmB48, tmB32;
      if (!encode_tma2d(&tmA256, a, k, M, 256) || !encode_tma2d(&tmA128, a, k, M, 128)) return -3;
      if (!encode_tma2d(&tmB128, b, k, n, 128) || !encode_tma2d(&tmB48, b, k, n, 48) ||
          !encode_tma2d(&tmB32, b, k, n, 32)) return -4;
      if (fp8ws_pdl_enabled()) {
        if (k == 4096) return (int)launch_ex_pdl(gemm_hybrid_6656_nsplit128<4096, true>, 48, NT, SMEM_MAX, st,
                                                 tmA256, tmA128, tmB128, tmB48, tmB32, p);
        return (int)launch_ex_pdl(gemm_hybrid_6656_nsplit128<19968, true>, 48, NT, SMEM_MAX, st,
                                  tmA256, tmA128, tmB128, tmB48, tmB32, p);
      }
      if (k == 4096) gemm_hybrid_6656_nsplit128<4096, false><<<48, NT, SMEM_MAX, st>>>(tmA256, tmA128, tmB128, tmB48, tmB32, p);
      else           gemm_hybrid_6656_nsplit128<19968, false><<<48, NT, SMEM_MAX, st>>>(tmA256, tmA128, tmB128, tmB48, tmB32, p);
      return (int)cudaGetLastError();
    }
    CUtensorMap tmA256, tmA64, tmB;
    if (!encode_tma2d(&tmA256, a, k, M, 256)) return -3;
    if (!encode_tma2d(&tmA64, a, k, M, 64)) return -3;
    if (!encode_tma2d(&tmB, b, k, n, BN)) return -4;
    if (k == 4096) return (int)launch_hybrid_6656<4096>(tmA256, tmA64, tmB, p, st);
    return (int)launch_hybrid_6656<19968>(tmA256, tmA64, tmB, p, st);
  }
  const int bm = (n <= 4608) ? 128 : 256;
  CUtensorMap tmA, tmB;
  if (!encode_tma2d(&tmA, a, k, M, bm)) return -3;
  if (!encode_tma2d(&tmB, b, k, n, BN)) return -4;
  Params p{(const float*)sa, (const float*)sb, (__nv_bfloat16*)c};
  cudaStream_t st = (cudaStream_t)stream;
  if (n == 4608 && k == 6656) return (int)launch_shape<4608, 6656>(tmA, tmB, p, st);
  if (n == 4096 && k == 6656) return (int)launch_shape<4096, 6656>(tmA, tmB, p, st);
  if (n == 6656 && k == 4096) return (int)launch_shape<6656, 4096>(tmA, tmB, p, st);
  if (n == 39936 && k == 6656) return (int)launch_shape<39936, 6656>(tmA, tmB, p, st);
  if (n == 6656 && k == 19968) return (int)launch_shape<6656, 19968>(tmA, tmB, p, st);
  return (int)cudaErrorInvalidValue;
}

extern "C" int fp8ws_run_dual(const void* a, const void* b0, const void* b1, const void* sa, const void* sb0,
                              const void* sb1, void* c0, void* c1, int m, int n0, int n1, int k, void* stream) {
  using namespace fp8ws;
  if (m != M || n0 != 4608 || n1 != 4096 || k != 6656 || !encode || num_sms <= 0)
    return (int)cudaErrorInvalidValue;
  if (((uintptr_t)a & 15) || ((uintptr_t)b0 & 15) || ((uintptr_t)b1 & 15) ||
      ((uintptr_t)c0 & 15) || ((uintptr_t)c1 & 15))
    return (int)cudaErrorInvalidValue;
  CUtensorMap tmA, tmB0, tmB1;
  if (!encode_tma2d(&tmA, a, k, M, 256)) return -3;
  if (!encode_tma2d(&tmB0, b0, k, n0, BN)) return -4;
  if (!encode_tma2d(&tmB1, b1, k, n1, BN)) return -5;
  ParamsDual p{(const float*)sa, (const float*)sb0, (const float*)sb1, (__nv_bfloat16*)c0, (__nv_bfloat16*)c1};
  return (int)launch_dual<4608, 4096, 6656>(tmA, tmB0, tmB1, p, (cudaStream_t)stream);
}

extern "C" int fp8ws_run_dual_a(const void* a0, const void* a1, const void* b0, const void* b1, const void* sa0,
                                const void* sa1, const void* sb0, const void* sb1, void* c0, void* c1, int m, int n0,
                                int n1, int k, int stages, void* stream) {
  using namespace fp8ws;
  if (m != M || n0 != 4608 || n1 != 4096 || k != 6656 || !encode || num_sms <= 0 || (stages != 2 && stages != 3))
    return (int)cudaErrorInvalidValue;
  if (((uintptr_t)a0 & 15) || ((uintptr_t)a1 & 15) || ((uintptr_t)b0 & 15) || ((uintptr_t)b1 & 15) ||
      ((uintptr_t)c0 & 15) || ((uintptr_t)c1 & 15))
    return (int)cudaErrorInvalidValue;
  ParamsDualA p{(const float*)sa0, (const float*)sa1, (const float*)sb0, (const float*)sb1, (__nv_bfloat16*)c0,
                (__nv_bfloat16*)c1, (const uint8_t*)b0, (size_t)n0 * (size_t)k, (const uint8_t*)b1,
                (size_t)n1 * (size_t)k};
  if (num_sms == 48 && g_dual_seamless) {
    static const bool bm256_dual = []() {
      const char* env = std::getenv("FP8WS_BM256_DUAL");
      return env && *env && std::strtol(env, nullptr, 10) != 0;
    }();
    if (bm256_dual) {
      CUtensorMap tmA0_256, tmA1_256, tmA1_128, tmB0, tmB1_128, tmB1_96, tmB1_64;
      if (!encode_tma2d(&tmA0_256, a0, k, M, 256) || !encode_tma2d(&tmA1_256, a1, k, M, 256) ||
          !encode_tma2d(&tmA1_128, a1, k, M, 128)) return -3;
      if (!encode_tma2d(&tmB0, b0, k, n0, BN)) return -4;
      if (!encode_tma2d(&tmB1_128, b1, k, n1, 128) || !encode_tma2d(&tmB1_96, b1, k, n1, 96) ||
          !encode_tma2d(&tmB1_64, b1, k, n1, 64)) return -5;
      if (fp8ws_pdl_enabled()) {
        return (int)launch_ex_pdl(gemm_dual_a_ws8_256_st2_st3<true>, 48, NT, SMEM_MAX, (cudaStream_t)stream,
                                  tmA0_256, tmA1_256, tmA1_128, tmB0, tmB1_128, tmB1_96, tmB1_64, p);
      }
      gemm_dual_a_ws8_256_st2_st3<false><<<48, NT, SMEM_MAX, (cudaStream_t)stream>>>(
          tmA0_256, tmA1_256, tmA1_128, tmB0, tmB1_128, tmB1_96, tmB1_64, p);
      return (int)cudaGetLastError();
    }
    if (g_st3_dual) {
      CUtensorMap tmA0, tmA1, tmB0, tmB1_128, tmB1_96, tmB1_64;
      if (!encode_tma2d(&tmA0, a0, k, M, 128) || !encode_tma2d(&tmA1, a1, k, M, 128)) return -3;
      if (!encode_tma2d(&tmB0, b0, k, n0, BN)) return -4;
      if (!encode_tma2d(&tmB1_128, b1, k, n1, 128) || !encode_tma2d(&tmB1_96, b1, k, n1, 96) ||
          !encode_tma2d(&tmB1_64, b1, k, n1, 64)) return -5;
      const bool ef = fp8ws_tmab_ef();
      if (fp8ws_pdl_enabled()) {
        return ef ? (int)launch_ex_pdl(gemm_dual_a_st3<true, true>, 48, NT_WS, SMEM_MAX, (cudaStream_t)stream,
                                       tmA0, tmA1, tmB0, tmB1_128, tmB1_96, tmB1_64, p)
                  : (int)launch_ex_pdl(gemm_dual_a_st3<true, false>, 48, NT_WS, SMEM_MAX, (cudaStream_t)stream,
                                       tmA0, tmA1, tmB0, tmB1_128, tmB1_96, tmB1_64, p);
      }
      if (ef) gemm_dual_a_st3<false, true><<<48, NT_WS, SMEM_MAX, (cudaStream_t)stream>>>(
          tmA0, tmA1, tmB0, tmB1_128, tmB1_96, tmB1_64, p);
      else    gemm_dual_a_st3<false, false><<<48, NT_WS, SMEM_MAX, (cudaStream_t)stream>>>(
          tmA0, tmA1, tmB0, tmB1_128, tmB1_96, tmB1_64, p);
      return (int)cudaGetLastError();
    }
    CUtensorMap tmA0, tmA1, tmB0, tmB1_128, tmB1_112, tmB1_96;
    if (!encode_tma2d(&tmA0, a0, k, M, 256) || !encode_tma2d(&tmA1, a1, k, M, 256)) return -3;
    if (!encode_tma2d(&tmB0, b0, k, n0, BN)) return -4;
    if (!encode_tma2d(&tmB1_128, b1, k, n1, 128) || !encode_tma2d(&tmB1_112, b1, k, n1, 112) ||
        !encode_tma2d(&tmB1_96, b1, k, n1, 96)) return -5;
    if (fp8ws_pdl_enabled()) {
      return (int)launch_ex_pdl(gemm_dual_a_seamless<true>, 48, NT, SMEM_MAX, (cudaStream_t)stream,
                                tmA0, tmA1, tmB0, tmB1_128, tmB1_112, tmB1_96, p);
    }
    gemm_dual_a_seamless<false><<<48, NT, SMEM_MAX, (cudaStream_t)stream>>>(
        tmA0, tmA1, tmB0, tmB1_128, tmB1_112, tmB1_96, p);
    return (int)cudaGetLastError();
  }
  CUtensorMap tmA0, tmA1, tmB0, tmB1;
  if (!encode_tma2d(&tmA0, a0, k, M, 256)) return -3;
  if (!encode_tma2d(&tmA1, a1, k, M, 256)) return -3;
  if (!encode_tma2d(&tmB0, b0, k, n0, BN)) return -4;
  if (!encode_tma2d(&tmB1, b1, k, n1, BN)) return -5;
  using S = ShapeDual256<4608, 4096, 6656>;
  const int grid = num_sms < S::NTILES ? num_sms : S::NTILES;
  gemm_dual_a<4608, 4096, 6656, 2><<<grid, NT, 2 * S::STB + 1024, (cudaStream_t)stream>>>(tmA0, tmA1, tmB0, tmB1, p);
  return (int)cudaGetLastError();
}

extern "C" int bf16ws_run_dual_stacked(const void* a, const void* w_cat, void* c0, void* c1,
                                       int m, int n0, int n1, int k, void* stream) {
  using namespace fp8ws;
  if (m != M || n0 != 4608 || n1 != 4096 || k != 6656 || !encode || num_sms <= 0)
    return (int)cudaErrorInvalidValue;
  if (((uintptr_t)a & 15) || ((uintptr_t)w_cat & 15) || ((uintptr_t)c0 & 15) || ((uintptr_t)c1 & 15))
    return (int)cudaErrorInvalidValue;
  ParamsBf16Dual p{(__nv_bfloat16*)c0, (__nv_bfloat16*)c1, (const uint8_t*)w_cat,
                   (size_t)(n0 + n1) * (size_t)k * 2u};
  if (num_sms == 48 && g_bf16_seamless) {
    static const bool bm256_bf16 = []() {
      const char* env = std::getenv("FP8WS_BM256_BF16");
      return !env || !*env || std::strtol(env, nullptr, 10) != 0;
    }();
    if (bm256_bf16) {
      CUtensorMap tmA256, tmA128, tmB_128, tmB_96, tmB_64;
      if (!encode_tma2d(&tmA256, a, 2 * k, M, 256) || !encode_tma2d(&tmA128, a, 2 * k, M, 128)) return -3;
      if (!encode_tma2d(&tmB_128, w_cat, 2 * k, n0 + n1, 128) ||
          !encode_tma2d(&tmB_96, w_cat, 2 * k, n0 + n1, 96) ||
          !encode_tma2d(&tmB_64, w_cat, 2 * k, n0 + n1, 64)) return -4;
      if (fp8ws_pdl_enabled()) {
        return (int)launch_ex_pdl(gemm_bf16_dual_ws8_256_st2_st3<true>, 48, NT, SMEM_MAX, (cudaStream_t)stream,
                                  tmA256, tmA128, tmB_128, tmB_96, tmB_64, p);
      }
      gemm_bf16_dual_ws8_256_st2_st3<false><<<48, NT, SMEM_MAX, (cudaStream_t)stream>>>(
          tmA256, tmA128, tmB_128, tmB_96, tmB_64, p);
      return (int)cudaGetLastError();
    }
    if (g_st3_bf16) {
      CUtensorMap tmA, tmB_128, tmB_96, tmB_64;
      if (!encode_tma2d(&tmA, a, 2 * k, M, 128)) return -3;
      if (!encode_tma2d(&tmB_128, w_cat, 2 * k, n0 + n1, 128) ||
          !encode_tma2d(&tmB_96, w_cat, 2 * k, n0 + n1, 96) ||
          !encode_tma2d(&tmB_64, w_cat, 2 * k, n0 + n1, 64)) return -4;
      const bool ef = fp8ws_tmab_ef();
      if (fp8ws_pdl_enabled()) {
        return ef ? (int)launch_ex_pdl(gemm_bf16_dual_st3<true, true>, 48, NT_WS, SMEM_MAX, (cudaStream_t)stream,
                                       tmA, tmB_128, tmB_96, tmB_64, p)
                  : (int)launch_ex_pdl(gemm_bf16_dual_st3<true, false>, 48, NT_WS, SMEM_MAX, (cudaStream_t)stream,
                                       tmA, tmB_128, tmB_96, tmB_64, p);
      }
      if (ef) gemm_bf16_dual_st3<false, true><<<48, NT_WS, SMEM_MAX, (cudaStream_t)stream>>>(tmA, tmB_128, tmB_96, tmB_64, p);
      else    gemm_bf16_dual_st3<false, false><<<48, NT_WS, SMEM_MAX, (cudaStream_t)stream>>>(tmA, tmB_128, tmB_96, tmB_64, p);
      return (int)cudaGetLastError();
    }
    CUtensorMap tmA, tmB_128, tmB_112, tmB_96;
    if (!encode_tma2d(&tmA, a, 2 * k, M, 256)) return -3;
    if (!encode_tma2d(&tmB_128, w_cat, 2 * k, n0 + n1, 128) ||
        !encode_tma2d(&tmB_112, w_cat, 2 * k, n0 + n1, 112) ||
        !encode_tma2d(&tmB_96, w_cat, 2 * k, n0 + n1, 96)) return -4;
    if (fp8ws_pdl_enabled()) {
      return (int)launch_ex_pdl(gemm_bf16_dual_seamless<true>, 48, NT, SMEM_MAX, (cudaStream_t)stream,
                                tmA, tmB_128, tmB_112, tmB_96, p);
    }
    gemm_bf16_dual_seamless<false><<<48, NT, SMEM_MAX, (cudaStream_t)stream>>>(tmA, tmB_128, tmB_112, tmB_96, p);
    return (int)cudaGetLastError();
  }
  CUtensorMap tmA, tmB;
  if (!encode_tma2d(&tmA, a, 2 * k, M, 256)) return -3;
  if (!encode_tma2d(&tmB, w_cat, 2 * k, n0 + n1, BN)) return -4;
  constexpr int T = 2 * ((4608 + 4096) / BN);
  const int grid = num_sms < T ? num_sms : T;
  gemm_bf16_dual_stacked<4608, 4096, 6656><<<grid, NT, 2 * 49152 + 1024, (cudaStream_t)stream>>>(tmA, tmB, p);
  return (int)cudaGetLastError();
}
#define GAUNTLET_REDRAW_00447c9a_20261128T035820Z 1
