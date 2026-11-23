// SPDX-License-Identifier: Apache-2.0
// Exact one-pass replacement for the compiled prompt-batch query/key RMSNorm (+ NEOX RoPE) kernel pair of the Muse
// Glimmer attention (32 query heads, 2 key heads, head size 128, packed qkv rows of 4608).
//
// The compiled graph runs two kernels per layer: a reduction (one warp per 128-wide head row) that writes the query
// sums of squares and the normalized (and rotated) keys, then a pointwise kernel that re-reads the queries and the
// sums and writes the normalized, pre-scaled (and rotated) queries. This kernel does both in one pass: one warp per
// (token, head) row, the row read once, the same float operations in the same order as the compiled code:
//   * lane l holds elements 4l..4l+3 of the row (the compiled layout [1,4] x [1,32]);
//   * per lane ((x0^2 + x1^2) + x2^2) + x3^2 with each square an FFMA into +0, then a warp butterfly 16, 8, 4, 2, 1;
//   * r = rsqrt.approx.ftz(div.full(s, 128) + eps);
//   * a = (r * x) for keys, (r * x) * qscale for queries, each product rounded to fp32 (no bf16 rounding);
//   * RoPE: d < 64: fma(a[d], cos[d], -(a[d+64] * sin[d])); d >= 64: fma(a[d], cos[d-64], a[d-64] * sin[d-64]);
//   * bf16 round to nearest even on store; the query sums of squares are stored as well.
// Explicitly rounded intrinsics are used everywhere, so this compiler cannot contract differently.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int kHeadDim = 128;
constexpr int kQHeads = 32;
constexpr int kKHeads = 2;
constexpr int kHeads = kQHeads + kKHeads;
constexpr int kQkvStride = (kQHeads + 2 * kKHeads) * kHeadDim;  // q, k, v of one token
constexpr int kWarps = 8;

__device__ __forceinline__ float rsqrt_approx_ftz(float v) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v));
  return r;
}

__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}

template <bool kRope>
__global__ void __launch_bounds__(kWarps * 32) qk_norm(
    const __nv_bfloat16* __restrict__ qkv, const int64_t* __restrict__ pos,
    const __nv_bfloat16* __restrict__ cache, long long cache_rows, float qscale, float eps, int tokens,
    float* __restrict__ q_sumsq, __nv_bfloat16* __restrict__ k_out, __nv_bfloat16* __restrict__ q_out) {
  const int lane = threadIdx.x & 31;
  const long long row = static_cast<long long>(blockIdx.x) * kWarps + (threadIdx.x >> 5);
  if (row >= static_cast<long long>(tokens) * kHeads) return;  // whole warps only
  const long long t = row / kHeads;
  const int h = static_cast<int>(row - t * kHeads);
  const bool is_q = h < kQHeads;
  const uint2 u = *reinterpret_cast<const uint2*>(qkv + t * kQkvStride + h * kHeadDim + lane * 4);
  float x[4] = {bf_lo(u.x), bf_hi(u.x), bf_lo(u.y), bf_hi(u.y)};

  float sq[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) sq[e] = __fmaf_rn(x[e], x[e], 0.0f);
  float s = __fadd_rn(__fadd_rn(__fadd_rn(sq[0], sq[1]), sq[2]), sq[3]);
  s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 16));
  s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 8));
  s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 4));
  s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 2));
  s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 1));
  if (is_q && lane == 0) q_sumsq[t * kQHeads + h] = s;
  const float r = rsqrt_approx_ftz(__fadd_rn(div_full(s, 128.0f), eps));

  float a[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    a[e] = __fmul_rn(r, x[e]);
    if (is_q) a[e] = __fmul_rn(a[e], qscale);
  }

  float y[4];
  if constexpr (kRope) {
    long long p = pos[t];
    if (p < 0) p += cache_rows;
    const int c0 = (lane & 15) * 4;  // d mod 64
    const uint2 uc = *reinterpret_cast<const uint2*>(cache + p * kHeadDim + c0);
    const uint2 us = *reinterpret_cast<const uint2*>(cache + p * kHeadDim + 64 + c0);
    const float cs[4] = {bf_lo(uc.x), bf_hi(uc.x), bf_lo(uc.y), bf_hi(uc.y)};
    const float sn[4] = {bf_lo(us.x), bf_hi(us.x), bf_lo(us.y), bf_hi(us.y)};
    const bool lo = lane < 16;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float other = __shfl_xor_sync(0xffffffffu, a[e], 16);  // a[d + 64] below, a[d - 64] above
      const float prod = __fmul_rn(other, sn[e]);
      y[e] = __fmaf_rn(a[e], cs[e], lo ? -prod : prod);
    }
  } else {
#pragma unroll
    for (int e = 0; e < 4; ++e) y[e] = a[e];
  }

  uint2 o;
  o.x = pack_bf16x2(y[0], y[1]);
  o.y = pack_bf16x2(y[2], y[3]);
  __nv_bfloat16* dst = is_q ? q_out + (t * kQHeads + h) * kHeadDim : k_out + (t * kKHeads + (h - kQHeads)) * kHeadDim;
  *reinterpret_cast<uint2*>(dst + lane * 4) = o;
}

}  // namespace

extern "C" {

// rope: 0 = no rotary embedding (NoPE layers), 1 = NEOX RoPE from the cos|sin cache rows (pos, cache, cache_rows).
int muse_qk_glue_launch(int rope, int tokens, const void* qkv, const void* pos, const void* cache,
                        long long cache_rows, float qscale, float eps, void* q_sumsq, void* k_out, void* q_out,
                        cudaStream_t stream) {
  if (tokens <= 0) return 0;
  const long long rows = static_cast<long long>(tokens) * kHeads;
  const unsigned blocks = static_cast<unsigned>((rows + kWarps - 1) / kWarps);
  const auto* Q = static_cast<const __nv_bfloat16*>(qkv);
  auto* S = static_cast<float*>(q_sumsq);
  auto* K = static_cast<__nv_bfloat16*>(k_out);
  auto* O = static_cast<__nv_bfloat16*>(q_out);
  if (rope) {
    qk_norm<true><<<blocks, kWarps * 32, 0, stream>>>(Q, static_cast<const int64_t*>(pos),
        static_cast<const __nv_bfloat16*>(cache), cache_rows, qscale, eps, tokens, S, K, O);
  } else {
    qk_norm<false><<<blocks, kWarps * 32, 0, stream>>>(Q, nullptr, nullptr, 0, qscale, eps, tokens, S, K, O);
  }
  return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
