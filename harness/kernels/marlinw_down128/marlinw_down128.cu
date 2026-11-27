// SPDX-License-Identifier: Apache-2.0
// 128-row x 128-col NVFP4 W4A16 prompt GEMM for the MLP down projection (M=512).
// Every B fragment is dequantized once per CTA (not once per 64-row chunk).
// Per-element MMA chains match marlinw_down exactly (see NOTES): A0 over k16 blocks
// {4t,4t+1}, A1 over {4t+2,4t+3}, P = A0 + A1, slice fold through fp32 scratch,
// bf16_rn(P * global_scale).
#define MARLIN_NAMESPACE_NAME marlinw_down128
#include <cuda_runtime.h>
#include "marlinw_template.h"

namespace marlinw_down128 {

constexpr int kRows = 512;
constexpr int kColumns = 6656;
constexpr int kInner = 19968;
constexpr int kThreads = 256;
#ifndef D128_STAGES
#define D128_STAGES 4
#endif
#ifndef D128_ORDER
#define D128_ORDER 0
#endif
constexpr int kStages = D128_STAGES;
#ifndef D128_DIST
#define D128_DIST 3
#endif
constexpr int kDist = D128_DIST;  // prefetch distance in tiles (<= kStages - 1)
constexpr int kAStage = 1024;  // int4: 128 rows x 8
constexpr int kBStage = 256;
constexpr int kSStage = 32;
constexpr int kStageInt4 = kAStage + kBStage + kSStage;
constexpr int kShared = kStages * kStageInt4 * 16 + 256;
constexpr int kBGlStride = kColumns / 2;   // int4 per k16 row of B
constexpr int kSGlStride = kColumns / 16;  // int4 per k16 row of scales
constexpr int kBF16 = 0;

using T = MarlinScalarType<vllm::kBFloat16.id()>;
using FragA = T::FragA;
using FragB = T::FragB;

__device__ inline void mma16(const FragA& a, const FragB& b, float* c) {
  const uint32_t* ap = reinterpret_cast<const uint32_t*>(&a);
  const uint32_t* bp = reinterpret_cast<const uint32_t*>(&b);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(ap[0]), "r"(ap[1]), "r"(ap[2]), "r"(ap[3]), "r"(bp[0]), "r"(bp[1]));
}

// One CTA = one work item w[2b] = {row0, col128, mb, slot}, w[2b+1] = {cut_a, cut_b, 0, 0}.
// The CTA computes rows [row0, row0 + 16 mb) x cols [128 col128, +128) over the full K (312 k-tiles).
// cut_a / cut_b are the k-tile at which the stock partition restarts the accumulator chain for the
// 64-row chunk containing row0 / the next chunk (-1: no cut). Chain (A0+A1) of the first range is saved
// at the cut and folded as P_a + P_b at the end.
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t a, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("{ .reg .b64 st; mbarrier.arrive.shared::cta.b64 st, [%0]; }" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_cp_arrive(uint32_t a) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, int parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(a), "r"(parity) : "memory");
}

#ifdef D128_PROF
__device__ unsigned long long g_prof[512][8];
#define PROF_T(v) long long v = clock64();
#else
#define PROF_T(v)
#endif

template <int MB, int kCols = kColumns, int kIn = kInner>
__device__ __forceinline__ void tile_body(
    const int4* __restrict__ A0, const int4* __restrict__ B, int4* __restrict__ C0,
    float4* __restrict__ scratch, const int4* __restrict__ scales, float gs, int lda,
    int row0, int col, int slot, int cut_a, int cut_b, int iters) {
  constexpr int kBStride = kCols / 2;
  constexpr int kSStride = kCols / 16;
  extern __shared__ int4 sh[];
#ifdef D128_PROF
  const long long p_k0 = clock64();
#endif
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int q = warp & 3;
  const int kw = warp >> 2;
  const int nb = q >> 1;
  const int sh2 = q & 1;
  const int a_gl_stride = lda / 8;
  constexpr int rows = MB * 16;
  constexpr int a_iters = (rows + 31) / 32;

  int a_wr[a_iters];
#pragma unroll
  for (int i = 0; i < a_iters; i++) {
    int row = (tid >> 3) + 32 * i;
    a_wr[i] = 8 * row + ((tid & 7) ^ (row & 7));
  }
  int a_rd[2][MB];
  {
    int base = 8 * (lane & 15) + (lane >> 4) + 4 * kw;
#pragma unroll
    for (int kk = 0; kk < 2; kk++)
#pragma unroll
      for (int j = 0; j < MB; j++) {
        int idx = 2 * kk + 128 * j + base;
        int row = idx / 8;
        a_rd[kk][j] = 8 * row + ((idx & 7) ^ (row & 7));
      }
  }
  const int s_rd = 2 * (8 * nb + (lane >> 2)) + sh2;

  float acc[MB][2][2][4];

  const int4* a_src = A0 + (size_t)a_gl_stride * (row0 + (tid >> 3)) + (tid & 7);
  const int4* b_src = B + (size_t)kBStride * (tid >> 6) + (tid & 63) + 64 * col;
  const int4* s_src = scales + (size_t)kSStride * (tid >> 3) + 8 * col + (tid & 7);
  int ftile = 0;
  const uint32_t full_bar = smem_u32(sh + kStages * kStageInt4);
  const uint32_t empty_bar = full_bar + 8 * kStages;
  if (tid == 0) {
    for (int i = 0; i < kStages; i++) {
      mbar_init(full_bar + 8 * i, kThreads);
      mbar_init(empty_bar + 8 * i, kThreads);
    }
  }
  __syncthreads();

  auto fetch = [&](int stage) {
    if (ftile < iters) {
#ifndef D128_NOLOAD
      int4* sa = sh + kStageInt4 * stage;
#ifndef D128_NOA
#pragma unroll
      for (int i = 0; i < a_iters; i++)
        if (rows == 128 || 32 * i + (tid >> 3) < rows)
          cp_async4(&sa[a_wr[i]], a_src + (size_t)a_gl_stride * 32 * i);
#endif
#ifndef D128_NOB
      cp_async4(&sa[kAStage + tid], b_src);
      if (tid < 32) cp_async4(&sa[kAStage + kBStage + tid], s_src);
#endif
#endif
#ifdef D128_PF
      if (ftile + D128_PF < iters) {
        if ((tid & 7) == 0)
          asm volatile("prefetch.global.L2 [%0];" ::"l"(b_src + (size_t)kBStride * 4 * D128_PF));
        if (tid < 32 && (tid & 7) == 0)
          asm volatile("prefetch.global.L2 [%0];" ::"l"(s_src + (size_t)kSStride * 4 * D128_PF));
      }
#endif
      a_src += 8;
      b_src += kBStride * 4;
      s_src += kSStride * 4;
      ftile++;
      mbar_cp_arrive(full_bar + 8 * stage);
    }
  };

  FragA fa[2][MB];
  int2 fb[2];
  int fs[2];

  auto load_regs = [&](int k, int stage) {
    const int g = 2 * kw + (k & 1);
    int4* sa = sh + kStageInt4 * stage;
#pragma unroll
    for (int j = 0; j < MB; j++) {
#ifdef D128_HALFLDSM
      if (j >= 4) { fa[k & 1][j] = fa[k & 1][j - 4]; continue; }
#endif
#if 1
      ldsm<4, vllm::kBFloat16.id()>(fa[k & 1][j], &sa[a_rd[k & 1][j]]);
#endif
    }
    const int* bi = reinterpret_cast<const int*>(&sa[kAStage + 64 * g + 32 * nb + lane]);
    fb[k & 1] = *reinterpret_cast<const int2*>(bi + 2 * sh2);
    fs[k & 1] = reinterpret_cast<const int*>(&sa[kAStage + kBStage])[s_rd + 32 * g];
  };

  auto wait_stage = [&]() {
    cp_async_wait<kStages - 2>();
#ifndef D128_NOSYNC
    __syncthreads();
#endif
  };

  FragB bb[2][2][2];  // dequantized B operands [buffer][jj][frag]
  auto dequant_to = [&](int k2) {
    nv_bfloat162 sc[2];
    dequant_fp8_scales<nv_bfloat162, vllm::kFE4M3fn.id()>(fs[k2], sc);
#pragma unroll
    for (int jj = 0; jj < 2; jj++) {
      FragB& b0 = bb[k2][jj][0];
      FragB& b1 = bb[k2][jj][1];
      int bq1 = jj == 0 ? fb[k2].x : fb[k2].y;
      int bq0 = bq1 << 8;
#ifdef D128_NODQ
      *reinterpret_cast<int*>(&b0[0]) = bq0; *reinterpret_cast<int*>(&b0[1]) = bq1; *reinterpret_cast<int*>(&b1[0]) = bq0 ^ 1; *reinterpret_cast<int*>(&b1[1]) = bq1 ^ 3;
#else
      {
        const unsigned u = static_cast<unsigned>(bq1);
        reinterpret_cast<unsigned*>(&b1)[1] =
            lop3<(0xf0 & 0xcc) | 0xaa>(u, 0x80008000, (u >> 6) & 0x01c001c0u);
        reinterpret_cast<unsigned*>(&b1)[0] = (((u & 0x0f000f00u) >> 2) * 65u) & 0x81c081c0u;
        reinterpret_cast<unsigned*>(&b0)[1] = ((u & 0x00f000f0u) * 260u) & 0x81c081c0u;
        reinterpret_cast<unsigned*>(&b0)[0] = ((u & 0x000f000fu) * 4160u) & 0x81c081c0u;
      }
#endif
      nv_bfloat162 s = sc[jj];
      nv_bfloat162 s0 = __bfloat162bfloat162(reinterpret_cast<nv_bfloat16*>(&s)[0]);
      nv_bfloat162 s1 = __bfloat162bfloat162(reinterpret_cast<nv_bfloat16*>(&s)[1]);
      b0[0] = __hmul2(b0[0], s0);
      b0[1] = __hmul2(b0[1], s0);
      b1[0] = __hmul2(b1[0], s1);
      b1[1] = __hmul2(b1[1], s1);
    }
  };
  auto mma_k = [&](int k2) {
#if D128_ORDER == 1
#pragma unroll
    for (int jj = 0; jj < 2; jj++)
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int i = 0; i < MB; i++) mma16(fa[k2][i], bb[k2][jj][h], acc[i][jj][h]);
#elif D128_ORDER == 2
#pragma unroll
    for (int i = 0; i < MB; i++)
#pragma unroll
      for (int jj = 0; jj < 2; jj++)
#pragma unroll
        for (int h = 0; h < 2; h++) mma16(fa[k2][i], bb[k2][jj][h], acc[i][jj][h]);
#else
#pragma unroll
    for (int jj = 0; jj < 2; jj++) {
#pragma unroll
      for (int i = 0; i < MB; i++) {
        mma16(fa[k2][i], bb[k2][jj][0], acc[i][jj][0]);
        mma16(fa[k2][i], bb[k2][jj][1], acc[i][jj][1]);
      }
    }
#endif
  };

  const int n_first = 4 - ((row0 >> 4) & 3);  // m-blocks in the chunk containing row0
  float4* my_scratch = scratch + ((size_t)slot * 256 + tid) * 32;  // fixed 32 float4 per thread
  auto do_cuts = [&](int t) {
#pragma unroll
    for (int i = 0; i < MB; i++) {
      int ci = i < n_first ? cut_a : cut_b;
      if (ci == t) {
#pragma unroll
        for (int e = 0; e < 4; e++) my_scratch[i * 4 + e] = reinterpret_cast<float4*>(acc[i])[e];
#pragma unroll
        for (int e = 0; e < 16; e++) reinterpret_cast<float*>(acc[i])[e] = 0.0f;
      }
    }
  };

#pragma unroll
  for (int i = 0; i < kDist; i++) fetch(i);
#pragma unroll
  for (int i = 0; i < MB * 16; i++) reinterpret_cast<float*>(acc)[i] = 0.0f;
  mbar_wait(full_bar, 0);
  load_regs(0, 0);
  dequant_to(0);

  static_assert((kStages & (kStages - 1)) == 0, "stages must be a power of two");
#ifdef D128_PROF
  long long p_start = clock64(), p_full = 0, p_empty = 0;
#endif
#ifdef D128_STAGGER
  if (kw == 1) {
    long long c0 = clock64();
    while (clock64() - c0 < D128_STAGGER) {
    }
  }
#endif
#pragma unroll 1
  for (int t = 0; t < iters; t++) {
    if (t == cut_a || t == cut_b) do_cuts(t);
    const int s = t & (kStages - 1);
    const int ns = (t + 1) & (kStages - 1);
    const int ps = (t + kDist) & (kStages - 1);
    load_regs(1, s);
    mbar_arrive(empty_bar + 8 * s);
    if (ftile < iters) {
#ifdef D128_PROF
      long long e0 = clock64();
#endif
      if (t + kDist >= kStages) mbar_wait(empty_bar + 8 * ps, ((t + kDist - kStages) / kStages) & 1);
#ifdef D128_PROF
      p_empty += clock64() - e0;
#endif
      fetch(ps);
    }
    dequant_to(1);
    mma_k(0);
#ifdef D128_PROF
    long long f0 = clock64();
#endif
    if (t + 1 < iters) mbar_wait(full_bar + 8 * ns, ((t + 1) / kStages) & 1);
#ifdef D128_PROF
    p_full += clock64() - f0;
#endif
    load_regs(2, ns);
    dequant_to(0);
    mma_k(1);
  }
#ifdef D128_PROF
  if ((tid & 31) == 0) {
    atomicAdd(&g_prof[blockIdx.x][0], clock64() - p_start);
    atomicAdd(&g_prof[blockIdx.x][5], p_start - p_k0);
    atomicAdd(&g_prof[blockIdx.x][1], p_full);
    atomicAdd(&g_prof[blockIdx.x][2], p_empty);
  }
#endif
  __syncthreads();

  float4* red = reinterpret_cast<float4*>(sh);
  if constexpr (MB == 8) {
    const int t128 = tid & 127;
    if (kw == 0) {
#pragma unroll
      for (int k = 0; k < 16; k++) red[k * 128 + t128] = reinterpret_cast<float4*>(&acc[4])[k];
    } else {
#pragma unroll
      for (int k = 0; k < 16; k++) red[(16 + k) * 128 + t128] = reinterpret_cast<float4*>(&acc[0])[k];
    }
    __syncthreads();
    __nv_bfloat16* C = reinterpret_cast<__nv_bfloat16*>(C0);
    const int r0 = row0 + (lane >> 2);
    const int col0 = 128 * col + 64 * nb + 32 * sh2 + 2 * (lane & 3);
    const float4* s0 = scratch + ((size_t)slot * 256 + t128) * 32;
    const float4* s1 = scratch + ((size_t)slot * 256 + t128 + 128) * 32;
    if (kw == 0) {
#pragma unroll
      for (int k = 0; k < 16; k++) {
        float4 r = red[(16 + k) * 128 + t128];
        float* c = reinterpret_cast<float*>(&acc[0]) + 4 * k;
        c[0] += r.x; c[1] += r.y; c[2] += r.z; c[3] += r.w;
      }
      if (cut_a >= 0 || cut_b >= 0) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
          int ci = i < n_first ? cut_a : cut_b;
          if (ci >= 0) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
              float4 x = s0[i * 4 + e], y = s1[i * 4 + e];
              float* c = reinterpret_cast<float*>(acc[i]) + 4 * e;
              c[0] = (x.x + y.x) + c[0];
              c[1] = (x.y + y.y) + c[1];
              c[2] = (x.z + y.z) + c[2];
              c[3] = (x.w + y.w) + c[3];
            }
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int jj = 0; jj < 2; jj++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            float* c = acc[i][jj][h];
            nv_bfloat162 lo = __halves2bfloat162(__float2bfloat16(c[0] * gs), __float2bfloat16(c[1] * gs));
            nv_bfloat162 hi = __halves2bfloat162(__float2bfloat16(c[2] * gs), __float2bfloat16(c[3] * gs));
            int cc = col0 + 16 * jj + 8 * h;
            *reinterpret_cast<nv_bfloat162*>(&C[(size_t)(r0 + 16 * i) * kCols + cc]) = lo;
            *reinterpret_cast<nv_bfloat162*>(&C[(size_t)(r0 + 16 * i + 8) * kCols + cc]) = hi;
          }
    } else {
#pragma unroll
      for (int k = 0; k < 16; k++) {
        float4 r = red[k * 128 + t128];
        float* c = reinterpret_cast<float*>(&acc[4]) + 4 * k;
        c[0] = r.x + c[0]; c[1] = r.y + c[1]; c[2] = r.z + c[2]; c[3] = r.w + c[3];
      }
      if (cut_a >= 0 || cut_b >= 0) {
#pragma unroll
        for (int i = 4; i < 8; i++) {
          int ci = i < n_first ? cut_a : cut_b;
          if (ci >= 0) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
              float4 x = s0[i * 4 + e], y = s1[i * 4 + e];
              float* c = reinterpret_cast<float*>(acc[i]) + 4 * e;
              c[0] = (x.x + y.x) + c[0];
              c[1] = (x.y + y.y) + c[1];
              c[2] = (x.z + y.z) + c[2];
              c[3] = (x.w + y.w) + c[3];
            }
          }
        }
      }
#pragma unroll
      for (int i = 4; i < 8; i++)
#pragma unroll
        for (int jj = 0; jj < 2; jj++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            float* c = acc[i][jj][h];
            nv_bfloat162 lo = __halves2bfloat162(__float2bfloat16(c[0] * gs), __float2bfloat16(c[1] * gs));
            nv_bfloat162 hi = __halves2bfloat162(__float2bfloat16(c[2] * gs), __float2bfloat16(c[3] * gs));
            int cc = col0 + 16 * jj + 8 * h;
            *reinterpret_cast<nv_bfloat162*>(&C[(size_t)(r0 + 16 * i) * kCols + cc]) = lo;
            *reinterpret_cast<nv_bfloat162*>(&C[(size_t)(r0 + 16 * i + 8) * kCols + cc]) = hi;
          }
    }
  } else {
    if (kw == 1) {
#pragma unroll
      for (int k = 0; k < MB * 4; k++) red[k * 128 + (tid - 128)] = reinterpret_cast<float4*>(acc)[k];
    }
    __syncthreads();
    if (kw == 0) {
#pragma unroll
      for (int k = 0; k < MB * 4; k++) {
        float4 r = red[k * 128 + tid];
        float* c = reinterpret_cast<float*>(acc) + 4 * k;
        c[0] += r.x; c[1] += r.y; c[2] += r.z; c[3] += r.w;
      }
      if (cut_a >= 0 || cut_b >= 0) {
        const float4* s0 = scratch + ((size_t)slot * 256 + tid) * 32;
        const float4* s1 = scratch + ((size_t)slot * 256 + tid + 128) * 32;
#pragma unroll
        for (int i = 0; i < MB; i++) {
          int ci = i < n_first ? cut_a : cut_b;
          if (ci >= 0) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
              float4 x = s0[i * 4 + e], y = s1[i * 4 + e];
              float* c = reinterpret_cast<float*>(acc[i]) + 4 * e;
              c[0] = (x.x + y.x) + c[0];
              c[1] = (x.y + y.y) + c[1];
              c[2] = (x.z + y.z) + c[2];
              c[3] = (x.w + y.w) + c[3];
            }
          }
        }
      }
      __nv_bfloat16* C = reinterpret_cast<__nv_bfloat16*>(C0);
      const int r0 = row0 + (lane >> 2);
      const int col0 = 128 * col + 64 * nb + 32 * sh2 + 2 * (lane & 3);
#pragma unroll
      for (int i = 0; i < MB; i++)
#pragma unroll
        for (int jj = 0; jj < 2; jj++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            float* c = acc[i][jj][h];
            nv_bfloat162 lo = __halves2bfloat162(__float2bfloat16(c[0] * gs), __float2bfloat16(c[1] * gs));
            nv_bfloat162 hi = __halves2bfloat162(__float2bfloat16(c[2] * gs), __float2bfloat16(c[3] * gs));
            int cc = col0 + 16 * jj + 8 * h;
            *reinterpret_cast<nv_bfloat162*>(&C[(size_t)(r0 + 16 * i) * kCols + cc]) = lo;
            *reinterpret_cast<nv_bfloat162*>(&C[(size_t)(r0 + 16 * i + 8) * kCols + cc]) = hi;
          }
    }
  }
#ifdef D128_PROF
  if ((tid & 31) == 0) atomicAdd(&g_prof[blockIdx.x][3], clock64() - p_k0);
#endif
}

template <int kCols, int kIn>
__global__ void __launch_bounds__(kThreads, 1) gemm128(
    const int4* __restrict__ A0, const int4* __restrict__ B,
    int4* __restrict__ C0, float4* __restrict__ scratch,
    const int4* __restrict__ scales, const float* __restrict__ gs_ptr,
    int lda, const int4* __restrict__ work) {
  const int4 w0 = work[2 * blockIdx.x];
  const int4 w1 = work[2 * blockIdx.x + 1];
  const float gs = gs_ptr[0];
#ifdef D128_ONLY8
  tile_body<8, kCols, kIn>(A0, B, C0, scratch, scales, gs, lda, w0.x, w0.y, w0.w, w1.x, w1.y, w1.z);
#else
  if (w0.z == 8)
    tile_body<8, kCols, kIn>(A0, B, C0, scratch, scales, gs, lda, w0.x, w0.y, w0.w, w1.x, w1.y, w1.z);
  else if (w0.z == 3)
    tile_body<3, kCols, kIn>(A0, B, C0, scratch, scales, gs, lda, w0.x, w0.y, w0.w, w1.x, w1.y, w1.z);
  else
    tile_body<2, kCols, kIn>(A0, B, C0, scratch, scales, gs, lda, w0.x, w0.y, w0.w, w1.x, w1.y, w1.z);
#endif
}

}  // namespace marlinw_down128

extern "C" int marlinw_down128_init(int* active_blocks) {
  using namespace marlinw_down128;
  auto k = gemm128<6656, 19968>;
  auto status = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(active_blocks, k, kThreads, kShared);
}

extern "C" int marlinw_down128_shape_init(int n, int k_in, int* active_blocks) {
  using namespace marlinw_down128;
  auto fn = (n == 6656 && k_in == 19968) ? gemm128<6656, 19968>
          : (n == 4608 && k_in == 6656)  ? gemm128<4608, 6656>
          : (n == 4096 && k_in == 6656)  ? gemm128<4096, 6656>
          : (n == 6656 && k_in == 4096)  ? gemm128<6656, 4096>
          : nullptr;
  if (!fn) return cudaErrorInvalidValue;
  auto status = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(active_blocks, fn, kThreads, kShared);
}

#ifdef D128_PROF
extern "C" int marlinw_down128_prof(unsigned long long* host) {
  return cudaMemcpyFromSymbol(host, marlinw_down128::g_prof, sizeof(unsigned long long) * 512 * 8);
}
#endif

extern "C" int marlinw_down128_main(
    const int4* input, const int4* weight, int4* output, float4* scratch,
    const int4* scales, const float* global_scale, const int4* work, int grid,
    int row_stride, cudaStream_t stream) {
  using namespace marlinw_down128;
  gemm128<6656, 19968><<<grid, kThreads, kShared, stream>>>(
      input, weight, output, scratch, scales, global_scale, row_stride, work);
  return cudaGetLastError();
}

extern "C" int marlinw_down128_shape_main(
    const int4* input, const int4* weight, int4* output, float4* scratch,
    const int4* scales, const float* global_scale, const int4* work, int grid,
    int n, int k_in, int row_stride, cudaStream_t stream) {
  using namespace marlinw_down128;
  if (n == 6656 && k_in == 19968)
    gemm128<6656, 19968><<<grid, kThreads, kShared, stream>>>(
        input, weight, output, scratch, scales, global_scale, row_stride, work);
  else if (n == 4608 && k_in == 6656)
    gemm128<4608, 6656><<<grid, kThreads, kShared, stream>>>(
        input, weight, output, scratch, scales, global_scale, row_stride, work);
  else if (n == 4096 && k_in == 6656)
    gemm128<4096, 6656><<<grid, kThreads, kShared, stream>>>(
        input, weight, output, scratch, scales, global_scale, row_stride, work);
  else if (n == 6656 && k_in == 4096)
    gemm128<6656, 4096><<<grid, kThreads, kShared, stream>>>(
        input, weight, output, scratch, scales, global_scale, row_stride, work);
  else
    return cudaErrorInvalidValue;
  return cudaGetLastError();
}