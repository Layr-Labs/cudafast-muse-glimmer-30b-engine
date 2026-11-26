// SPDX-License-Identifier: Apache-2.0
// 128-row x 128-column (64 gate + 64 up) NVFP4 W4A16 tile with fused SwiGLU.
// Same per-element MMA chains as marlinw_swiglu (A0 over k16 blocks 4t,4t+1,
// A1 over 4t+2,4t+3, P = A0 + A1, bf16_rn(P * global_scale), swiglu_product),
// but each weight fragment is dequantized once for 128 rows instead of 64.
// 8 warps: kw (k-half of each 64-deep tile) x cw (gate/up) x jh (n-tile pair).
#define MARLINW_SWIGLU
#define MARLIN_NAMESPACE_NAME marlinw_swiglu2
#include <cuda_runtime.h>
#include "marlinw_template.h"

#ifndef STAGES
#define STAGES 3
#endif

namespace marlinw_swiglu2 {
using namespace vllm;
constexpr auto kA = kBFloat16.id();
using Adtype = MarlinScalarType<kA>;
using FragA = typename Adtype::FragA;
using FragB = typename Adtype::FragB;
using FragC = typename Adtype::FragC;
using FragS = typename Adtype::FragS;
using scalar_t2 = typename Adtype::scalar_t2;
using scalar_32bit_t = typename Adtype::scalar_32bit_t;

constexpr int ST = STAGES;
constexpr int KTILES = 6656 / 64;  // 104
constexpr int A_STAGE = 128 * 8;   // int4
constexpr int B_STAGE = 256;
constexpr int S_STAGE = 32;
constexpr int SH_A = 0, SH_B = ST * A_STAGE, SH_S = SH_B + ST * B_STAGE;
constexpr int SH_RED_I4 = 128 * 17;                       // epilogue staging
constexpr int SH_TOTAL = SH_S + ST * S_STAGE > SH_RED_I4 ? SH_S + ST * S_STAGE : SH_RED_I4;
constexpr int shared_bytes = SH_TOTAL * 16;

__device__ __forceinline__ void cp_async4_ca(void* smem_ptr, const void* glob_ptr) {
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(glob_ptr));
}

__device__ __forceinline__ void cp_async4_cg256(void* smem_ptr, const void* glob_ptr) {
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" ::"r"(smem), "l"(glob_ptr));
}

__global__ void __launch_bounds__(256, 1)
kernel(const int4* __restrict__ A0, const int4* __restrict__ B,
       char* __restrict__ out, const int4* __restrict__ scales_ptr,
       const float* __restrict__ global_scale_ptr, int lda, int prob_n) {
  extern __shared__ int4 sh[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int kw = warp >> 2, cw = (warp >> 1) & 1, jh = warp & 1;
  const int chunk = blockIdx.x & 3, slice_col = blockIdx.x >> 2;
  const float gs = global_scale_ptr[0];

  const int a_stride = lda / 8;
  const int4* Ap = A0 + (size_t)chunk * 128 * a_stride;
  // A global/shared addressing
  const int a_row0 = tid >> 3, a_c = tid & 7;
  const int4* a_src = Ap + (size_t)a_row0 * a_stride + a_c;
  const int a_dst = a_row0 * 8 + (a_c ^ (a_row0 & 7));
  // B
  const int b_gl_stride = prob_n / 2;
  const int bc = tid & 63;
  const int4* b_src = B + (size_t)b_gl_stride * (tid >> 6) + bc + 32 * slice_col +
                      ((bc & 32) ? (b_gl_stride - 64) / 2 : 0);
  const int b_step = b_gl_stride * 4;
  // S
  const int s_gl_stride = prob_n / 16;
  const int4* s_src = scales_ptr + (size_t)s_gl_stride * (tid >> 3) + 4 * slice_col + (tid & 7) +
                      ((tid & 4) ? (s_gl_stride - 8) / 2 : 0);
  const int s_step = s_gl_stride * 4;

  int fetch_kt = 0;
  auto fetch_to_shared = [&](int stage) {
    if (fetch_kt < KTILES) {
      int4* da = sh + SH_A + stage * A_STAGE + a_dst;
#pragma unroll
      for (int it = 0; it < 4; it++)
        cp_async4_ca(da + it * 256, a_src + (size_t)it * 32 * a_stride + 8 * fetch_kt);
      cp_async4_cg256(sh + SH_B + stage * B_STAGE + tid, b_src + (size_t)b_step * fetch_kt);
      if (tid < 32) cp_async4_cg256(sh + SH_S + stage * S_STAGE + tid, s_src + (size_t)s_step * fetch_kt);
    }
    fetch_kt++;
    cp_async_fence();
  };

  // ldmatrix addressing
  const int a_rd_row = lane & 15, a_rd_col = lane >> 4;
  const int a_rd_off = a_rd_row * 8;
  const int a_rd_x = lane & 7;
  const int s_rd = (8 * cw + (lane >> 2)) * 2 + jh;  // int index within a k16 row (32 ints per row)

  FragA frag_a[2][8];
  int2 bq[2];
  int sq[2];
  FragC c[8][2][2];
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int g = 0; g < 4; g++) c[i][j][h][g] = 0.0f;

  auto fetch_regs = [&](int k, int stage) {  // k: 0/1 (parity)
    const int kk = k & 1;
    const int4* sa = sh + SH_A + stage * A_STAGE + a_rd_off + ((4 * kw + 2 * kk + a_rd_col) ^ a_rd_x);
#pragma unroll
    for (int i = 0; i < 8; i++) ldsm<4, kA>(frag_a[kk][i], sa + 128 * i);
    bq[kk] = reinterpret_cast<const int2*>(sh + SH_B + stage * B_STAGE + 64 * (2 * kw + kk) + 32 * cw + lane)[jh];
    sq[kk] = reinterpret_cast<const int*>(sh + SH_S + stage * S_STAGE)[(2 * kw + kk) * 32 + s_rd];
  };

  FragB fb[2][2][2];
  FragS fsc[2][2];
  auto deq = [&](int kk, int jl) {
    if (jl == 0)
      dequant_fp8_scales<scalar_t2, kFE4M3fn.id()>(sq[kk], reinterpret_cast<scalar_t2*>(fsc[kk]));
    const unsigned u = static_cast<unsigned>(jl == 0 ? bq[kk].x : bq[kk].y);
    reinterpret_cast<unsigned*>(&fb[kk][jl][1])[1] = (u & 0x80008000u) | ((u >> 6) & 0x01c001c0u);
    reinterpret_cast<unsigned*>(&fb[kk][jl][1])[0] = (((u & 0x0f000f00u) >> 2) * 65u) & 0x81c081c0u;
    reinterpret_cast<unsigned*>(&fb[kk][jl][0])[1] = ((u & 0x00f000f0u) * 260u) & 0x81c081c0u;
    reinterpret_cast<unsigned*>(&fb[kk][jl][0])[0] = ((u & 0x000f000fu) * 4160u) & 0x81c081c0u;
    scale<kA>(fb[kk][jl][0], fsc[kk][jl], 0);
    scale<kA>(fb[kk][jl][1], fsc[kk][jl], 1);
  };
  auto mmas = [&](int kk, int jl) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      mma<kA, false>(frag_a[kk][i], fb[kk][jl][0], c[i][jl][0]);
      mma<kA, false>(frag_a[kk][i], fb[kk][jl][1], c[i][jl][1]);
    }
  };
  auto matmul = [&](int k) {
    const int kk = k & 1;
    mmas(kk, 0);
    deq(kk ^ 1, 0);
    mmas(kk, 1);
    deq(kk ^ 1, 1);
  };

  // Pipeline
#pragma unroll
  for (int s = 0; s < ST - 1; s++) fetch_to_shared(s);
  cp_async_wait<ST - 2>();
  __syncthreads();
  fetch_regs(0, 0);
  deq(0, 0);
  deq(0, 1);

  int iters = KTILES;
  while (iters) {
#pragma unroll
    for (int pipe = 0; pipe < ST;) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        fetch_regs(k + 1, pipe % ST);
        if (k == 0) {
          fetch_to_shared((pipe + ST - 1) % ST);
          pipe++;
          cp_async_wait<ST - 2>();
          __syncthreads();
        }
        matmul(k);
      }
      if (--iters == 0) break;
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  // Parallel k-warp reduction & BF16 staging within 32 KiB SMEM (keeps 61.5 KiB shared_bytes & 64 KiB L1 cache):
  // kw=0 owns i in {0,1,4,5}, kw=1 owns i in {2,3,6,7}, both accumulating in exact A0 + A1 order.
  int4* red = sh;
  const int t128 = tid & 127;
#pragma unroll
  for (int half = 0; half < 2; half++) {
    if (kw == 1) {
#pragma unroll
      for (int q = 0; q < 8; q++)
        red[q * 128 + t128] = reinterpret_cast<int4*>(&c[4 * half])[q];
    } else {
#pragma unroll
      for (int q = 0; q < 8; q++)
        red[(8 + q) * 128 + t128] = reinterpret_cast<int4*>(&c[4 * half + 2])[q];
    }
    __syncthreads();
    if (kw == 0) {
#pragma unroll
      for (int q = 0; q < 8; q++) {
        const int4 v = red[q * 128 + t128];
        float* cf = reinterpret_cast<float*>(&c[4 * half]) + 4 * q;
        cf[0] += __int_as_float(v.x);
        cf[1] += __int_as_float(v.y);
        cf[2] += __int_as_float(v.z);
        cf[3] += __int_as_float(v.w);
      }
    } else {
#pragma unroll
      for (int q = 0; q < 8; q++) {
        const int4 v = red[(8 + q) * 128 + t128];
        float* cf = reinterpret_cast<float*>(&c[4 * half + 2]) + 4 * q;
        cf[0] = __int_as_float(v.x) + cf[0];
        cf[1] = __int_as_float(v.y) + cf[1];
        cf[2] = __int_as_float(v.z) + cf[2];
        cf[3] = __int_as_float(v.w) + cf[3];
      }
    }
    __syncthreads();
  }

  {
    // words (bf16x2) in rows of 68 words; kw=0 writes i={0,1,4,5}, kw=1 writes i={2,3,6,7} in parallel
    // using strictly compile-time indices into c[] so ptxas keeps 0 bytes stack frame / 0 STL/LDL.
    uint32_t* shw = reinterpret_cast<uint32_t*>(sh);
    auto put = [&](int idx, float c0, float c1) {
      __nv_bfloat162 r = __halves2bfloat162(__float2bfloat16_rn(c0 * gs),
                                            __float2bfloat16_rn(c1 * gs));
      shw[idx] = *reinterpret_cast<uint32_t*>(&r);
    };
    if (kw == 0) {
#pragma unroll
      for (int half = 0; half < 2; half++) {
        int wr = ((4 * half) * 16 + (lane >> 2)) * 68 + (lane & 3) + 32 * cw;
#pragma unroll
        for (int il = 0; il < 2; il++) {
          const int i = 4 * half + il;
#pragma unroll
          for (int jl = 0; jl < 2; jl++) {
            const int j = 2 * jh + jl;
            const int w = wr + 8 * j;
            put(w, c[i][jl][0][0], c[i][jl][0][1]);
            put(w + 68 * 8, c[i][jl][0][2], c[i][jl][0][3]);
            put(w + 4, c[i][jl][1][0], c[i][jl][1][1]);
            put(w + 68 * 8 + 4, c[i][jl][1][2], c[i][jl][1][3]);
          }
          wr += 16 * 68;
        }
      }
    } else {
#pragma unroll
      for (int half = 0; half < 2; half++) {
        int wr = ((4 * half + 2) * 16 + (lane >> 2)) * 68 + (lane & 3) + 32 * cw;
#pragma unroll
        for (int il = 0; il < 2; il++) {
          const int i = 4 * half + 2 + il;
#pragma unroll
          for (int jl = 0; jl < 2; jl++) {
            const int j = 2 * jh + jl;
            const int w = wr + 8 * j;
            put(w, c[i][jl][0][0], c[i][jl][0][1]);
            put(w + 68 * 8, c[i][jl][0][2], c[i][jl][0][3]);
            put(w + 4, c[i][jl][1][0], c[i][jl][1][1]);
            put(w + 68 * 8 + 4, c[i][jl][1][2], c[i][jl][1][3]);
          }
          wr += 16 * 68;
        }
      }
    }
  }
  __syncthreads();
  {
    const uint32_t* shw = reinterpret_cast<const uint32_t*>(sh);
    uint32_t* o = reinterpret_cast<uint32_t*>(out);
#pragma unroll 4
    for (int row = warp; row < 128; row += 8) {
      uint32_t gates = shw[row * 68 + lane], ups = shw[row * 68 + lane + 32];
      uint32_t prod;
#pragma unroll
      for (int e = 0; e < 2; e++) {
        auto gate = reinterpret_cast<const __nv_bfloat16*>(&gates)[e];
        auto up = reinterpret_cast<const __nv_bfloat16*>(&ups)[e];
        const float gf = __uint_as_float(static_cast<uint32_t>(__bfloat16_as_ushort(gate)) << 16);
        reinterpret_cast<__nv_bfloat16*>(&prod)[e] =
            __float2bfloat16_rn(__fmul_rn(swiglu_silu(gf), __bfloat162float(up)));
      }
      o[(size_t)(chunk * 128 + row) * (prob_n / 4) + slice_col * 32 + lane] = prod;
    }
  }
}
}  // namespace marlinw_swiglu2

extern "C" int swiglu_prepare() {
  auto status = cudaFuncSetAttribute(marlinw_swiglu2::kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     marlinw_swiglu2::shared_bytes);
  if (status != cudaSuccess) return status;
  swiglu_initialize<<<256, 256>>>();
  return cudaDeviceSynchronize();
}

extern "C" int swiglu_gemm(const void* x, const void* w, const void* scales, const void* global,
                           void* out, int stride, cudaStream_t stream) {
  constexpr int m = 512, n = 39936, k = 6656;
  if (stride < k || stride % 8) return cudaErrorInvalidValue;
  marlinw_swiglu2::kernel<<<(m / 128) * (n / 128), 256, marlinw_swiglu2::shared_bytes, stream>>>(
      static_cast<const int4*>(x), static_cast<const int4*>(w), static_cast<char*>(out),
      static_cast<const int4*>(scales), static_cast<const float*>(global), stride, n);
  return cudaGetLastError();
}
