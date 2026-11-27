// SPDX-License-Identifier: Apache-2.0
// Exact few-row replacements for the four compiled sandwich-norm reduction kernels of the Muse Glimmer decoder
// (hidden size 6656), for the decode step (one row per CTA).
//
// The compiled kernels (Inductor/Triton, R0_BLOCK 1024, 8 warps, 4 lanes per thread) are latency bound in
// decode: one CTA walks the row three times with one dependent global round trip per 1024-element chunk.
// These kernels issue every load of a row up front and keep the row in registers, but evaluate exactly the
// same float operations as the compiled SASS, in the same order:
//   * per-lane sums of squares: one FFMA chain per lane over the 7 chunks (lane j covers j + 1024 c);
//   * per-thread partial ((l0 + l1) + l2) + l3, warp butterfly 16, 8, 4, 2, 1, then the 8 warp sums
//     butterflied 4, 2, 1 (Triton's tl.sum lowering for this layout);
//   * mean + eps folded by ptxas into one FFMA with the reciprocal constant of div.full(., 6656.0):
//     rsqrt.approx.ftz(fma(s, 0x391d89d9, eps));
//   * the per-element chains below, with FMUL/FADD/FFMA exactly where the compiled SASS has them
//     (explicitly rounded intrinsics, so this compiler cannot contract them differently);
//   * FP8: y * rcp.approx.ftz(scale) (div.full(1, scale) for a normal scale), NaN-propagating clamp to
//     +-448, cvt.rn.satfinite.e4m3.
// The wrapper (vllm .../muse_glimmer_glue.py) only routes launches whose compiled source matches the
// expected value chain, and admits each compiled kernel only after a bitwise comparison on device.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdlib>
#include <stdint.h>

namespace {

constexpr int kHidden = 6656;
constexpr int kThreads = 256;
constexpr int kChunks = 7;  // ceil(6656 / 1024); chunk 6 covers lanes 0..511 only

__device__ __forceinline__ float rsqrt_approx_ftz(float v) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v));
  return r;
}

__device__ __forceinline__ float rcp_approx_ftz(float v) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v));
  return r;
}

__device__ __forceinline__ float full_divide(float num, float denom) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(num), "f"(denom));
  return r;
}

// rsqrt(s / 6656 + eps) exactly as the compiled SASS evaluates it.
__device__ __forceinline__ float inv_rms(float s, float eps) {
  return rsqrt_approx_ftz(__fmaf_rn(s, __int_as_float(0x391d89d9), eps));
}

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

struct Row4 {  // 4 consecutive bf16 of one chunk, as floats (exact)
  float v[4];
};

__device__ __forceinline__ Row4 unpack(uint2 u) {
  Row4 r;
  r.v[0] = bf_lo(u.x);
  r.v[1] = bf_hi(u.x);
  r.v[2] = bf_lo(u.y);
  r.v[3] = bf_hi(u.y);
  return r;
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&p);
}

__device__ __forceinline__ float clamp448(float y) {
  // triton_helpers.maximum / minimum (NaN propagating) against -448 / 448
  if (!(y > -448.0f) && !(y != y)) y = -448.0f;
  if (!(y < 448.0f) && !(y != y)) y = 448.0f;
  return y;
}

__device__ __forceinline__ uint16_t fp8x2(float lo, float hi) {
  __nv_fp8x2_storage_t s = __nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
  return static_cast<uint16_t>(s);
}

__device__ __forceinline__ bool chunk_valid(int c) {
  return c < kChunks - 1 || threadIdx.x < 128;
}

// Sum over the row in Triton's order; every thread returns the same value. `red` holds 8 floats and must not
// be reused by a later reduction of the same kernel without a barrier (each reduction gets its own slot).
__device__ __forceinline__ float row_sum(const float (&acc)[4], float* red) {
  float p = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
  p = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, 16));
  p = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, 8));
  p = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, 4));
  p = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, 2));
  p = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, 1));
  const int lane = threadIdx.x & 31;
  if (lane == 0) red[threadIdx.x >> 5] = p;
  __syncthreads();
  float v = red[lane & 7];
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return v;  // lanes 0..7 hold the total in Triton; lanes 8..31 hold the same values here (8-lane groups)
}

// SM121 Programmatic Dependent Launch (PDL): preload all inputs independent of the preceding GEMV
// (and precompute h1 for post_ff / post_ff_final) before waiting for the preceding grid's completion,
// then immediately signal dependent launch for the following GEMV so it can overlap our reduction.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;"); }

__device__ __forceinline__ uint2 ld8(const __nv_bfloat16* base, int c) {
  return *reinterpret_cast<const uint2*>(base + c * 1024 + threadIdx.x * 4);
}

__device__ __forceinline__ uint2 ld8_cs(const __nv_bfloat16* base, int c) {
  uint2 r;
  const void* ptr = base + c * 1024 + threadIdx.x * 4;
  asm("ld.global.cs.v2.u32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(ptr));
  return r;
}

__device__ __forceinline__ uint2 ldg8(const __nv_bfloat16* base, int c) {
  return __ldg(reinterpret_cast<const uint2*>(base + c * 1024 + threadIdx.x * 4));
}

__device__ __forceinline__ uint4 ld16_cs(const uint4* ptr) {
  uint4 r;
  asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(ptr));
  return r;
}

// Direct 8B/4B coalesced load/store (0-ULP identical per-thread layout; eliminates even/odd lane
// divergence and __shfl_xor_sync overhead across all 52 layers on SM121, matching prefill_attn.cu).
__device__ __forceinline__ uint2 ld8_v4(const __nv_bfloat16* base, int c) {
  return ld8(base, c);
}

__device__ __forceinline__ uint2 ld8_cs_v4(const __nv_bfloat16* base, int c) {
  return ld8_cs(base, c);
}

__device__ __forceinline__ uint2 ldg8_v4(const __nv_bfloat16* base, int c) {
  return ldg8(base, c);
}

__device__ __forceinline__ void st8_v4(__nv_bfloat16* base, int c, uint2 val) {
  *reinterpret_cast<uint2*>(base + c * 1024 + threadIdx.x * 4) = val;
}

__device__ __forceinline__ void st4_v2(uint8_t* base, int c, uint32_t val) {
  *reinterpret_cast<uint32_t*>(base + c * 1024 + threadIdx.x * 4) = val;
}

// ---------------------------------------------------------------------------------------------------------
// Post-attention norm + residual + pre-feedforward norm (compiled ..._mean_mul_pow_rsqrt_1 and the FP8 variant
// ..._clamp_..._reciprocal_rsqrt_1):
//   s_x = sum x^2 (stored),  a = inv_rms(s_x, 1e-8)
//   h   = fma(a * x, w_post + 1, res)          (FMUL, FADD, FFMA)
//   s_h = sum h^2,           b = inv_rms(s_h, 1e-5)
//   y   = (b * h) * (w_pre + 1)                 (FMUL, FADD, FMUL)
//   out_bf16 = bf16(y);  out_fp8 = e4m3(clamp(rcp(scale) * y))
template <bool kBf16Out, bool kFp8Out, bool kPdl = false>
__global__ void __launch_bounds__(kThreads) post_attn_norm_decode(
    const __nv_bfloat16* x_all, const __nv_bfloat16* res_all,
    const __nv_bfloat16* __restrict__ w_post, const __nv_bfloat16* __restrict__ w_pre,
    const float* scale, float* sumsq_out,
    __nv_bfloat16* out_bf16_all, uint8_t* out_fp8_all) {
  __shared__ float red[2][8];
  const size_t row = blockIdx.x;
  const __nv_bfloat16* x = x_all + row * kHidden;
  const __nv_bfloat16* res = res_all + row * kHidden;
  uint2 ux[kChunks], ur[kChunks], uw[kChunks], up[kChunks];
  float inv_scale = 0.f;
  if constexpr (kPdl) {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid(c)) {
        ur[c] = ld8_v4(res, c);
        uw[c] = ldg8_v4(w_post, c);
        up[c] = ldg8_v4(w_pre, c);
      }
    }
    if constexpr (kFp8Out) inv_scale = rcp_approx_ftz(__ldg(scale));
    pdl_wait();
    pdl_launch_dependents();
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid(c)) ux[c] = ld8_v4(x, c);
    }
  } else {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid(c)) {
        ux[c] = ld8_v4(x, c);
        ur[c] = ld8_v4(res, c);
        uw[c] = ldg8_v4(w_post, c);
        up[c] = ldg8_v4(w_pre, c);
      }
    }
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 v = unpack(ux[c]);
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[e] = __fmaf_rn(v.v[e], v.v[e], acc[e]);
    }
  }
  const float s_x = row_sum(acc, red[0]);
  if (threadIdx.x == 0) sumsq_out[row] = s_x;
  const float a = inv_rms(s_x, 1e-08f);
  float h[kChunks][4];
  float acc2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 xv = unpack(ux[c]), rv = unpack(ur[c]), wv = unpack(uw[c]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h[c][e] = __fmaf_rn(__fmul_rn(a, xv.v[e]), __fadd_rn(wv.v[e], 1.0f), rv.v[e]);
        acc2[e] = __fmaf_rn(h[c][e], h[c][e], acc2[e]);
      }
    }
  }
  const float s_h = row_sum(acc2, red[1]);
  const float b = inv_rms(s_h, 1e-05f);
  if constexpr (!kPdl && kFp8Out) inv_scale = rcp_approx_ftz(__ldg(scale));
  __nv_bfloat16* const out_bf16_row = kBf16Out ? (out_bf16_all + row * kHidden) : nullptr;
  uint8_t* const out_fp8_row = kFp8Out ? (out_fp8_all + row * kHidden) : nullptr;
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 pv = unpack(up[c]);
      float y[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) y[e] = __fmul_rn(__fmul_rn(b, h[c][e]), __fadd_rn(pv.v[e], 1.0f));
      if constexpr (kBf16Out) {
        uint2 o;
        o.x = pack_bf16x2(y[0], y[1]);
        o.y = pack_bf16x2(y[2], y[3]);
        st8_v4(out_bf16_row, c, o);
      }
      if constexpr (kFp8Out) {
        uint32_t q = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[0])), clamp448(__fmul_rn(inv_scale, y[1])))) |
                     (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[2])), clamp448(__fmul_rn(inv_scale, y[3])))) << 16);
        st4_v2(out_fp8_row, c, q);
      }
    }
  }
}

template <bool kBf16Out, bool kFp8Out, bool kPdl = false>
__global__ void __launch_bounds__(kThreads, 4) post_attn_norm(
    const __nv_bfloat16* __restrict__ x_all, const __nv_bfloat16* __restrict__ res_all,
    const __nv_bfloat16* __restrict__ w_post, const __nv_bfloat16* __restrict__ w_pre,
    const float* __restrict__ scale, float* __restrict__ sumsq_out,
    __nv_bfloat16* __restrict__ out_bf16_all, uint8_t* __restrict__ out_fp8_all) {
  __shared__ float red[2][8];
  __shared__ uint2 sh_res[kPdl ? (kHidden / 4) : 1];
  const size_t row = blockIdx.x;
  const __nv_bfloat16* x = x_all + row * kHidden;
  const __nv_bfloat16* res = res_all + row * kHidden;
  uint2 ux[kChunks];
  float inv_scale = 0.f;
  if constexpr (kPdl) {
    // Preload predecessor-independent residual (6.8 MB across 512 rows) into SMEM and scale before waiting for o_proj.
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid(c)) sh_res[c * 256 + threadIdx.x] = ld8_cs_v4(res, c);
    }
    if constexpr (kFp8Out) inv_scale = rcp_approx_ftz(__ldg(scale));
    pdl_wait();
    pdl_launch_dependents();
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      ux[c] = ld8_cs_v4(x, c);
      Row4 v = unpack(ux[c]);
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[e] = __fmaf_rn(v.v[e], v.v[e], acc[e]);
    }
  }
  const float s_x = row_sum(acc, red[0]);
  if (threadIdx.x == 0) sumsq_out[row] = s_x;
  const float a = inv_rms(s_x, 1e-08f);
  float h[kChunks][4];
  float acc2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 xv = unpack(ux[c]);
      Row4 rv = unpack(kPdl ? sh_res[c * 256 + threadIdx.x] : ld8_cs_v4(res, c));
      Row4 wv = unpack(ldg8_v4(w_post, c));
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h[c][e] = __fmaf_rn(__fmul_rn(a, xv.v[e]), __fadd_rn(wv.v[e], 1.0f), rv.v[e]);
        acc2[e] = __fmaf_rn(h[c][e], h[c][e], acc2[e]);
      }
    }
  }
  const float s_h = row_sum(acc2, red[1]);
  const float b = inv_rms(s_h, 1e-05f);
  if constexpr (!kPdl && kFp8Out) inv_scale = rcp_approx_ftz(__ldg(scale));
  __nv_bfloat16* const out_bf16_row = kBf16Out ? (out_bf16_all + row * kHidden) : nullptr;
  uint8_t* const out_fp8_row = kFp8Out ? (out_fp8_all + row * kHidden) : nullptr;
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 pv = unpack(ldg8_v4(w_pre, c));
      float y[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) y[e] = __fmul_rn(__fmul_rn(b, h[c][e]), __fadd_rn(pv.v[e], 1.0f));
      if constexpr (kBf16Out) {
        uint2 o;
        o.x = pack_bf16x2(y[0], y[1]);
        o.y = pack_bf16x2(y[2], y[3]);
        st8_v4(out_bf16_row, c, o);
      }
      if constexpr (kFp8Out) {
        uint32_t q = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[0])), clamp448(__fmul_rn(inv_scale, y[1])))) |
                     (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[2])), clamp448(__fmul_rn(inv_scale, y[3])))) << 16);
        st4_v2(out_fp8_row, c, q);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------
// Post-feedforward norm + residual + next layer's input norm (compiled ..._mean_mul_pow_rsqrt_3 and the FP8
// variant ..._clamp_..._reciprocal_rsqrt_3):
//   s_d = sum d^2,  c = inv_rms(s_d, 1e-8),  a = inv_rms(s_x (from the first kernel), 1e-8)
//   h1  = res + (a * x) * (w_post + 1)          (FMUL, FADD, FMUL, then an explicitly rounded add)
//   h2  = fma(c * d, w_ff + 1, h1)               (FMUL, FADD, FFMA)
//   x  <- bf16(h2) (in place);  s_h = sum h2^2 (unrounded h2);  e = inv_rms(s_h, 1e-5)
//   y   = (e * bf16(h2)) * (w_in + 1);  out_bf16 = bf16(y);  out_fp8 = e4m3(clamp(rcp(scale) * y))
template <bool kFp8Out, bool kGateOut = false, bool kWriteBf16 = true, bool kPdl = false>
__global__ void __launch_bounds__(kThreads) post_ff_norm_decode(
    __nv_bfloat16* x_all, const __nv_bfloat16* d_all,
    const __nv_bfloat16* res_all, const float* sumsq_x,
    const __nv_bfloat16* __restrict__ w_post, const __nv_bfloat16* __restrict__ w_ff,
    const __nv_bfloat16* w_in, const float* scale,
    __nv_bfloat16* out_bf16_all, uint8_t* out_fp8_all,
    const float* gate_scale = nullptr, uint8_t* out_gate_all = nullptr) {
  __shared__ float red[2][8];
  const size_t row = blockIdx.x;
  __nv_bfloat16* x = x_all + row * kHidden;
  const __nv_bfloat16* d = d_all + row * kHidden;
  const __nv_bfloat16* res = res_all + row * kHidden;
  uint2 ud[kChunks], uf[kChunks], ui[kChunks];
  float h1_pre[kChunks][4];
  float inv_scale = 0.f;
  float gate_inv = 0.f;
  if constexpr (kPdl) {
    const float s_x = __ldg(sumsq_x + row);
    const float a = inv_rms(s_x, 1e-08f);
    if constexpr (kFp8Out) inv_scale = rcp_approx_ftz(__ldg(scale));
    if constexpr (kGateOut) gate_inv = full_divide(1.0f, __ldg(gate_scale));
    // Precompute h1_pre chunk-by-chunk before pdl_wait() so ux/ur/uw never consume live regs post-wait.
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid(c)) {
        const uint2 uxc = ld8_v4(x, c);
        const uint2 urc = ld8_v4(res, c);
        const uint2 uwc = ldg8_v4(w_post, c);
        uf[c] = ldg8_v4(w_ff, c);
        ui[c] = ldg8_v4(w_in, c);
        Row4 xv = unpack(uxc), rv = unpack(urc), wv = unpack(uwc);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float t1 = __fmul_rn(__fmul_rn(a, xv.v[e]), __fadd_rn(wv.v[e], 1.0f));
          h1_pre[c][e] = __fadd_rn(rv.v[e], t1);
        }
      }
    }
    pdl_wait();
    pdl_launch_dependents();
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid(c)) ud[c] = ld8_v4(d, c);
    }
  } else {
    const float a = inv_rms(__ldg(sumsq_x + row), 1e-08f);
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid(c)) {
        ud[c] = ld8_v4(d, c);
        const uint2 uxc = ld8_v4(x, c);
        const uint2 urc = ld8_v4(res, c);
        const uint2 uwc = ldg8_v4(w_post, c);
        uf[c] = ldg8_v4(w_ff, c);
        ui[c] = ldg8_v4(w_in, c);
        Row4 xv = unpack(uxc), rv = unpack(urc), wv = unpack(uwc);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float t1 = __fmul_rn(__fmul_rn(a, xv.v[e]), __fadd_rn(wv.v[e], 1.0f));
          h1_pre[c][e] = __fadd_rn(rv.v[e], t1);
        }
      }
    }
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 v = unpack(ud[c]);
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[e] = __fmaf_rn(v.v[e], v.v[e], acc[e]);
    }
  }
  const float s_d = row_sum(acc, red[0]);
  const float cd = inv_rms(s_d, 1e-08f);
  float acc2[4] = {0.f, 0.f, 0.f, 0.f};
  uint2 hb[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 dv = unpack(ud[c]), fv = unpack(uf[c]);
      float h2[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h2[e] = __fmaf_rn(__fmul_rn(cd, dv.v[e]), __fadd_rn(fv.v[e], 1.0f), h1_pre[c][e]);
        acc2[e] = __fmaf_rn(h2[e], h2[e], acc2[e]);
      }
      hb[c].x = pack_bf16x2(h2[0], h2[1]);
      hb[c].y = pack_bf16x2(h2[2], h2[3]);
      st8_v4(x, c, hb[c]);
    }
  }
  const float s_h = row_sum(acc2, red[1]);
  const float e_r = inv_rms(s_h, 1e-05f);
  if constexpr (!kPdl && kFp8Out) inv_scale = rcp_approx_ftz(__ldg(scale));
  if constexpr (!kPdl && kGateOut) gate_inv = full_divide(1.0f, __ldg(gate_scale));
  __nv_bfloat16* const out_bf16_row = kWriteBf16 ? (out_bf16_all + row * kHidden) : nullptr;
  uint8_t* const out_fp8_row = kFp8Out ? (out_fp8_all + row * kHidden) : nullptr;
  uint8_t* const out_gate_row = kGateOut ? (out_gate_all + row * kHidden) : nullptr;
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 hv = unpack(hb[c]), iv = unpack(ui[c]);
      float y[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) y[e] = __fmul_rn(__fmul_rn(e_r, hv.v[e]), __fadd_rn(iv.v[e], 1.0f));
      uint2 o;
      o.x = pack_bf16x2(y[0], y[1]);
      o.y = pack_bf16x2(y[2], y[3]);
      if constexpr (kWriteBf16) {
        st8_v4(out_bf16_row, c, o);
      }
      if constexpr (kFp8Out) {
        uint32_t q = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[0])), clamp448(__fmul_rn(inv_scale, y[1])))) |
                     (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[2])), clamp448(__fmul_rn(inv_scale, y[3])))) << 16);
        st4_v2(out_fp8_row, c, q);
      }
      if constexpr (kGateOut) {
        const float g0 = bf_lo(o.x), g1 = bf_hi(o.x), g2 = bf_lo(o.y), g3 = bf_hi(o.y);
        uint32_t gq = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(gate_inv, g0)), clamp448(__fmul_rn(gate_inv, g1)))) |
                      (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(gate_inv, g2)), clamp448(__fmul_rn(gate_inv, g3)))) << 16);
        st4_v2(out_gate_row, c, gq);
      }
    }
  }
}

template <bool kFp8Out, bool kPdl = false>
__global__ void __launch_bounds__(kThreads, 4) post_ff_norm_prefill(
    __nv_bfloat16* __restrict__ x_all, const __nv_bfloat16* __restrict__ d_all,
    const __nv_bfloat16* __restrict__ res_all, const float* __restrict__ sumsq_x,
    const __nv_bfloat16* __restrict__ w_post, const __nv_bfloat16* __restrict__ w_ff,
    const __nv_bfloat16* __restrict__ w_in, const float* __restrict__ scale,
    __nv_bfloat16* __restrict__ out_bf16_all, uint8_t* __restrict__ out_fp8_all) {
  __shared__ float red[2][8];
  __shared__ uint2 sh_res[kPdl ? (kHidden / 4) : 1];
  const size_t row = blockIdx.x;
  __nv_bfloat16* x = x_all + row * kHidden;
  const __nv_bfloat16* d = d_all + row * kHidden;
  const __nv_bfloat16* res = res_all + row * kHidden;
  uint2 ud[kChunks];
  float a = 0.f;
  float inv_scale = 0.f;
  if constexpr (kPdl) {
    const float s_x = __ldg(sumsq_x + row);
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid(c)) sh_res[c * 256 + threadIdx.x] = ld8_cs_v4(res, c);
    }
    a = inv_rms(s_x, 1e-08f);
    if constexpr (kFp8Out) inv_scale = rcp_approx_ftz(__ldg(scale));
    pdl_wait();
    pdl_launch_dependents();
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      ud[c] = ld8_cs_v4(d, c);
      Row4 v = unpack(ud[c]);
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[e] = __fmaf_rn(v.v[e], v.v[e], acc[e]);
    }
  }
  if constexpr (!kPdl) {
    const float s_x = __ldg(sumsq_x + row);
    a = inv_rms(s_x, 1e-08f);
  }
  const float s_d = row_sum(acc, red[0]);
  const float cd = inv_rms(s_d, 1e-08f);
  float acc2[4] = {0.f, 0.f, 0.f, 0.f};
  uint2 hb[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 xv = unpack(ld8_cs_v4(x, c)), dv = unpack(ud[c]),
           rv = unpack(kPdl ? sh_res[c * 256 + threadIdx.x] : ld8_cs_v4(res, c));
      Row4 wv = unpack(ldg8_v4(w_post, c)), fv = unpack(ldg8_v4(w_ff, c));
      float h2[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float t1 = __fmul_rn(__fmul_rn(a, xv.v[e]), __fadd_rn(wv.v[e], 1.0f));
        const float h1 = __fadd_rn(rv.v[e], t1);
        h2[e] = __fmaf_rn(__fmul_rn(cd, dv.v[e]), __fadd_rn(fv.v[e], 1.0f), h1);
        acc2[e] = __fmaf_rn(h2[e], h2[e], acc2[e]);
      }
      hb[c].x = pack_bf16x2(h2[0], h2[1]);
      hb[c].y = pack_bf16x2(h2[2], h2[3]);
      st8_v4(x, c, hb[c]);
    }
  }
  const float s_h = row_sum(acc2, red[1]);
  const float e_r = inv_rms(s_h, 1e-05f);
  if constexpr (!kPdl && kFp8Out) inv_scale = rcp_approx_ftz(__ldg(scale));
  __nv_bfloat16* const out_bf16_row = out_bf16_all + row * kHidden;
  uint8_t* const out_fp8_row = kFp8Out ? (out_fp8_all + row * kHidden) : nullptr;
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 hv = unpack(hb[c]), iv = unpack(ldg8_v4(w_in, c));
      float y[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) y[e] = __fmul_rn(__fmul_rn(e_r, hv.v[e]), __fadd_rn(iv.v[e], 1.0f));
      uint2 o;
      o.x = pack_bf16x2(y[0], y[1]);
      o.y = pack_bf16x2(y[2], y[3]);
      st8_v4(out_bf16_row, c, o);
      if constexpr (kFp8Out) {
        uint32_t q = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[0])), clamp448(__fmul_rn(inv_scale, y[1])))) |
                     (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[2])), clamp448(__fmul_rn(inv_scale, y[3])))) << 16);
        st4_v2(out_fp8_row, c, q);
      }
    }
  }
}

__global__ void gate_formula_kernel(const uint16_t* bits, const float* gate_scale, uint8_t* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float gate_inv = full_divide(1.0f, *gate_scale);
  const float v = __uint_as_float(static_cast<uint32_t>(bits[i]) << 16);
  out[i] = static_cast<uint8_t>(fp8x2(clamp448(__fmul_rn(gate_inv, v)), 0.0f) & 0xff);
}

__device__ __forceinline__ float triton_sigmoid(float g) {
  float sig;
  asm("{\n\t"
      ".reg .f32 ng, t, e, d;\n\t"
      "sub.f32 ng, 0f00000000, %1;\n\t"
      "mul.f32 t, ng, 0f3FB8AA3B;\n\t"
      "ex2.approx.f32 e, t;\n\t"
      "add.f32 d, e, 0f3F800000;\n\t"
      "div.full.f32 %0, 0f3F800000, d;\n\t"
      "}" : "=f"(sig) : "f"(g));
  return sig;
}

__device__ __forceinline__ float triton_silu(float gate) {
  float result;
  asm("{\n\t"
      ".reg .f32 x, a, b, c, d, e, f, g, h;\n\t"
      ".reg .b32 exp2_scale;\n\t"
      "sub.f32 x, 0f00000000, %1;\n\t"
      "fma.rn.ftz.f32 a, x, 0f3BBB989D, 0f3F000000;\n\t"
      "cvt.ftz.sat.f32.f32 b, a;\n\t"
      "fma.rm.ftz.f32 c, b, 0f437C0000, 0f4B400001;\n\t"
      "add.f32 d, c, 0fCB40007F;\n\t"
      "neg.f32 d, d;\n\t"
      "fma.rn.ftz.f32 e, x, 0f3FB8AA3B, d;\n\t"
      "fma.rn.ftz.f32 f, x, 0f32A57060, e;\n\t"
      "shl.b32 exp2_scale, c, 23;\n\t"
      "ex2.approx.ftz.f32 g, f;\n\t"
      "fma.rn.f32 h, g, exp2_scale, 0f3F800000;\n\t"
      "div.full.f32 %0, %1, h;\n\t"
      "}" : "=f"(result) : "f"(gate));
  return result;
}

template <bool kStreamCS = false, bool kPdl = false>
__global__ void __launch_bounds__(256) mul_sigmoid_bf16_kernel(
    const uint4* __restrict__ gate, const uint4* __restrict__ attn, uint4* __restrict__ out, int vec8) {
  const int idx = blockIdx.x * 256 + threadIdx.x;
  if constexpr (kPdl) pdl_launch_dependents();
  if (idx >= vec8) return;
  const uint4 g = kStreamCS ? ld16_cs(gate + idx) : gate[idx];
  const uint4 a = kStreamCS ? ld16_cs(attn + idx) : attn[idx];
  const uint32_t gw[4] = {g.x, g.y, g.z, g.w}, aw[4] = {a.x, a.y, a.z, a.w};
  uint32_t ow[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float gl = bf_lo(gw[i]), gh = bf_hi(gw[i]);
    const float al = bf_lo(aw[i]), ah = bf_hi(aw[i]);
    ow[i] = pack_bf16x2(__fmul_rn(triton_sigmoid(gl), al), __fmul_rn(triton_sigmoid(gh), ah));
  }
  out[idx] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
}

template <bool kPdl = false>
__global__ void __launch_bounds__(256) mul_sigmoid_fp8_kernel(
    const uint4* __restrict__ gate, const uint4* __restrict__ attn,
    const float* __restrict__ scale, uint2* __restrict__ out, int vec8) {
  const int idx = blockIdx.x * 256 + threadIdx.x;
  if constexpr (kPdl) pdl_launch_dependents();
  if (idx >= vec8) return;
  const float inv_s = full_divide(1.0f, __ldg(scale));
  const uint4 g = ld16_cs(gate + idx), a = ld16_cs(attn + idx);
  const uint32_t gw[4] = {g.x, g.y, g.z, g.w}, aw[4] = {a.x, a.y, a.z, a.w};
  uint16_t ow[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float gl = bf_lo(gw[i]), gh = bf_hi(gw[i]);
    const float al = bf_lo(aw[i]), ah = bf_hi(aw[i]);
    const float ql = clamp448(__fmul_rn(__fmul_rn(triton_sigmoid(gl), al), inv_s));
    const float qh = clamp448(__fmul_rn(__fmul_rn(triton_sigmoid(gh), ah), inv_s));
    ow[i] = fp8x2(ql, qh);
  }
  const uint32_t lo = static_cast<uint32_t>(ow[0]) | (static_cast<uint32_t>(ow[1]) << 16);
  const uint32_t hi = static_cast<uint32_t>(ow[2]) | (static_cast<uint32_t>(ow[3]) << 16);
  out[idx] = make_uint2(lo, hi);
}

constexpr int kInterVec8 = 19968 / 8;  // 2496 uint4s per 19968-element BF16 half-row

__global__ void __launch_bounds__(128) mul_silu_slice_bf16_kernel(
    const uint4* __restrict__ gu_all, uint4* __restrict__ out_all) {
  const int col = blockIdx.x * 128 + threadIdx.x;
  if (col >= kInterVec8) return;
  const size_t row = blockIdx.y;
  const uint4* gu = gu_all + row * (2 * kInterVec8);
  const uint4 g = gu[col];
  const uint4 u = gu[col + kInterVec8];
  const uint32_t gw[4] = {g.x, g.y, g.z, g.w}, uw[4] = {u.x, u.y, u.z, u.w};
  uint32_t ow[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float gl = bf_lo(gw[i]), gh = bf_hi(gw[i]);
    const float ul = bf_lo(uw[i]), uh = bf_hi(uw[i]);
    ow[i] = pack_bf16x2(__fmul_rn(triton_silu(gl), ul), __fmul_rn(triton_silu(gh), uh));
  }
  out_all[row * kInterVec8 + col] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
}

__global__ void __launch_bounds__(128) mul_silu_slice_fp8_kernel(
    const uint4* __restrict__ gu_all, const float* __restrict__ scale, uint2* __restrict__ out_all) {
  const int col = blockIdx.x * 128 + threadIdx.x;
  if (col >= kInterVec8) return;
  const float inv_s = full_divide(1.0f, __ldg(scale));
  const size_t row = blockIdx.y;
  const uint4* gu = gu_all + row * (2 * kInterVec8);
  const uint4 g = ld16_cs(gu + col);
  const uint4 u = ld16_cs(gu + col + kInterVec8);
  const uint32_t gw[4] = {g.x, g.y, g.z, g.w}, uw[4] = {u.x, u.y, u.z, u.w};
  uint16_t ow[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float gl = bf_lo(gw[i]), gh = bf_hi(gw[i]);
    const float ul = bf_lo(uw[i]), uh = bf_hi(uw[i]);
    const float ql = clamp448(__fmul_rn(__fmul_rn(triton_silu(gl), ul), inv_s));
    const float qh = clamp448(__fmul_rn(__fmul_rn(triton_silu(gh), uh), inv_s));
    ow[i] = fp8x2(ql, qh);
  }
  const uint32_t lo = static_cast<uint32_t>(ow[0]) | (static_cast<uint32_t>(ow[1]) << 16);
  const uint32_t hi = static_cast<uint32_t>(ow[2]) | (static_cast<uint32_t>(ow[3]) << 16);
  out_all[row * kInterVec8 + col] = make_uint2(lo, hi);
}

// Layer-51 final post-FF norm + residual + model.norm (compiled bf5b0726...):
//   tensors: (d, res, x, sumsq_x, w_post, w_ff, w_in, out_h2_fp32, out_bf16)
//   h1 = res + (a * x) * (w_post + 1)             (FMUL, FADD, FMUL, FADD)
//   h2 = fma(cd * d, w_ff + 1, h1)                (FMUL, FADD, FFMA — matches compiled SASS)
//   out_h2_fp32 <- h2 (if kWriteH2);  s_h = sum h2^2;  e_r = inv_rms(s_h, 1e-5)
//   y  = (e_r * h2) * (w_in + 0.0f);  out_bf16 <- bf16(y)
template <bool kWriteH2 = true, bool kPdl = false>
__global__ void __launch_bounds__(kThreads) post_ff_final_norm(
    const __nv_bfloat16* d_all, const __nv_bfloat16* res_all,
    const __nv_bfloat16* x_all, const float* sumsq_x,
    const __nv_bfloat16* __restrict__ w_post, const __nv_bfloat16* __restrict__ w_ff,
    const __nv_bfloat16* __restrict__ w_in,
    float* out_h2_all, __nv_bfloat16* out_bf16_all) {
  __shared__ float red[2][8];
  const size_t row = blockIdx.x;
  const __nv_bfloat16* d = d_all + row * kHidden;
  const __nv_bfloat16* res = res_all + row * kHidden;
  const __nv_bfloat16* x = x_all + row * kHidden;
  uint2 ud[kChunks], uf[kChunks], ui[kChunks];
  float h1_pre[kChunks][4];
  const float a = inv_rms(__ldg(sumsq_x + row), 1e-08f);
  if constexpr (kPdl) {
    // Precompute h1_pre chunk-by-chunk before pdl_wait() so ux/ur/uw never consume live regs post-wait.
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid(c)) {
        const uint2 urc = ld8_v4(res, c);
        const uint2 uxc = ld8_v4(x, c);
        const uint2 uwc = ldg8_v4(w_post, c);
        uf[c] = ldg8_v4(w_ff, c);
        ui[c] = ldg8_v4(w_in, c);
        Row4 xv = unpack(uxc), rv = unpack(urc), wv = unpack(uwc);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float t1 = __fmul_rn(__fmul_rn(a, xv.v[e]), __fadd_rn(wv.v[e], 1.0f));
          h1_pre[c][e] = __fadd_rn(rv.v[e], t1);
        }
      }
    }
    pdl_wait();
    pdl_launch_dependents();
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid(c)) ud[c] = ld8_v4(d, c);
    }
  } else {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid(c)) {
        ud[c] = ld8_v4(d, c);
        const uint2 urc = ld8_v4(res, c);
        const uint2 uxc = ld8_v4(x, c);
        const uint2 uwc = ldg8_v4(w_post, c);
        uf[c] = ldg8_v4(w_ff, c);
        ui[c] = ldg8_v4(w_in, c);
        Row4 xv = unpack(uxc), rv = unpack(urc), wv = unpack(uwc);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float t1 = __fmul_rn(__fmul_rn(a, xv.v[e]), __fadd_rn(wv.v[e], 1.0f));
          h1_pre[c][e] = __fadd_rn(rv.v[e], t1);
        }
      }
    }
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 v = unpack(ud[c]);
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[e] = __fmaf_rn(v.v[e], v.v[e], acc[e]);
    }
  }
  const float s_d = row_sum(acc, red[0]);
  const float cd = inv_rms(s_d, 1e-08f);
  float h2[kChunks][4];
  float acc2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 dv = unpack(ud[c]), fv = unpack(uf[c]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h2[c][e] = __fmaf_rn(__fmul_rn(cd, dv.v[e]), __fadd_rn(fv.v[e], 1.0f), h1_pre[c][e]);
        acc2[e] = __fmaf_rn(h2[c][e], h2[c][e], acc2[e]);
      }
      if constexpr (kWriteH2) {
        const size_t off = row * kHidden + c * 1024 + threadIdx.x * 4;
        *reinterpret_cast<float4*>(out_h2_all + off) =
            make_float4(h2[c][0], h2[c][1], h2[c][2], h2[c][3]);
      }
    }
  }
  const float s_h = row_sum(acc2, red[1]);
  const float e_r = inv_rms(s_h, 1e-05f);
  __nv_bfloat16* const out_bf16_row = out_bf16_all + row * kHidden;
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 iv = unpack(ui[c]);
      float y[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        y[e] = __fmul_rn(__fmul_rn(e_r, h2[c][e]), __fadd_rn(iv.v[e], 0.0f));
      }
      uint2 o;
      o.x = pack_bf16x2(y[0], y[1]);
      o.y = pack_bf16x2(y[2], y[3]);
      st8_v4(out_bf16_row, c, o);
    }
  }
}

// Layer-0 initial input_layernorm + FP8 qkv quant (+ optional gate FP8 quant fold), 8-warp (R0_BLOCK=1024) layout:
template <bool kGateOut = false, bool kWriteBf16 = true, bool kPdl = false>
__global__ void __launch_bounds__(kThreads) input_norm_fp8_w8(
    const __nv_bfloat16* x_all, const __nv_bfloat16* __restrict__ w_in,
    const float* scale, __nv_bfloat16* out_bf16_all, uint8_t* out_fp8_all,
    const float* gate_scale = nullptr, uint8_t* out_gate_all = nullptr) {
  __shared__ float red[8];
  const size_t row = blockIdx.x;
  const __nv_bfloat16* x = x_all + row * kHidden;
  uint2 ux[kChunks], ui[kChunks];
  float inv_scale = 0.f;
  float gate_inv = 0.f;
  if constexpr (kPdl) {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid(c)) ui[c] = ldg8_v4(w_in, c);
    }
    inv_scale = full_divide(1.0f, __ldg(scale));
    if constexpr (kGateOut) gate_inv = full_divide(1.0f, __ldg(gate_scale));
    pdl_wait();
    pdl_launch_dependents();
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid(c)) ux[c] = ld8_v4(x, c);
    }
  } else {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid(c)) {
        ux[c] = ld8_v4(x, c);
        ui[c] = ldg8_v4(w_in, c);
      }
    }
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 v = unpack(ux[c]);
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[e] = __fmaf_rn(v.v[e], v.v[e], acc[e]);
    }
  }
  const float s_x = row_sum(acc, red);
  const float r = inv_rms(s_x, 1e-05f);
  if constexpr (!kPdl) {
    inv_scale = full_divide(1.0f, __ldg(scale));
    if constexpr (kGateOut) gate_inv = full_divide(1.0f, __ldg(gate_scale));
  }
  __nv_bfloat16* const out_bf16_row = kWriteBf16 ? (out_bf16_all + row * kHidden) : nullptr;
  uint8_t* const out_fp8_row = out_fp8_all + row * kHidden;
  uint8_t* const out_gate_row = kGateOut ? (out_gate_all + row * kHidden) : nullptr;
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 xv = unpack(ux[c]), iv = unpack(ui[c]);
      float y[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) y[e] = __fmul_rn(__fmul_rn(r, xv.v[e]), __fadd_rn(iv.v[e], 1.0f));
      uint2 o;
      o.x = pack_bf16x2(y[0], y[1]);
      o.y = pack_bf16x2(y[2], y[3]);
      if constexpr (kWriteBf16) {
        st8_v4(out_bf16_row, c, o);
      }
      uint32_t q = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[0])), clamp448(__fmul_rn(inv_scale, y[1])))) |
                   (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[2])), clamp448(__fmul_rn(inv_scale, y[3])))) << 16);
      st4_v2(out_fp8_row, c, q);
      if constexpr (kGateOut) {
        const float g0 = bf_lo(o.x), g1 = bf_hi(o.x), g2 = bf_lo(o.y), g3 = bf_hi(o.y);
        uint32_t gq = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(gate_inv, g0)), clamp448(__fmul_rn(gate_inv, g1)))) |
                      (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(gate_inv, g2)), clamp448(__fmul_rn(gate_inv, g3)))) << 16);
        st4_v2(out_gate_row, c, gq);
      }
    }
  }
}

// Layer-0 initial input_layernorm + FP8 qkv quant (+ optional gate FP8 quant fold), 16-warp (R0_BLOCK=8192) layout:
//   sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 16] (512 threads; t < 416 owns 16 elements).
template <bool kGateOut = false, bool kWriteBf16 = true, bool kPdl = false>
__global__ void __launch_bounds__(512) input_norm_fp8_w16(
    const __nv_bfloat16* x_all, const __nv_bfloat16* __restrict__ w_in,
    const float* scale, __nv_bfloat16* out_bf16_all, uint8_t* out_fp8_all,
    const float* gate_scale = nullptr, uint8_t* out_gate_all = nullptr) {
  __shared__ float red[16];
  const size_t row = blockIdx.x;
  const int t = threadIdx.x;
  const bool valid = t < 416;
  uint4 ux0 = {0, 0, 0, 0}, ux1 = {0, 0, 0, 0};
  uint4 ui0 = {0, 0, 0, 0}, ui1 = {0, 0, 0, 0};
  float inv_scale = 0.f;
  float gate_inv = 0.f;
  if constexpr (kPdl) {
    if (valid) {
      const uint4* ip = reinterpret_cast<const uint4*>(w_in + t * 16);
      ui0 = ip[0];
      ui1 = ip[1];
      inv_scale = full_divide(1.0f, *scale);
      if constexpr (kGateOut) gate_inv = full_divide(1.0f, *gate_scale);
    }
    pdl_wait();
    pdl_launch_dependents();
    if (valid) {
      const uint4* xp = reinterpret_cast<const uint4*>(x_all + row * kHidden + t * 16);
      ux0 = xp[0];
      ux1 = xp[1];
    }
  } else if (valid) {
    const uint4* xp = reinterpret_cast<const uint4*>(x_all + row * kHidden + t * 16);
    const uint4* ip = reinterpret_cast<const uint4*>(w_in + t * 16);
    ux0 = xp[0];
    ux1 = xp[1];
    ui0 = ip[0];
    ui1 = ip[1];
  }
  float xv[16];
  const uint32_t xw[8] = {ux0.x, ux0.y, ux0.z, ux0.w, ux1.x, ux1.y, ux1.z, ux1.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    xv[2 * i] = bf_lo(xw[i]);
    xv[2 * i + 1] = bf_hi(xw[i]);
  }
  float p = 0.0f;
  if (valid) {
    p = __fadd_rn(__fmul_rn(xv[0], xv[0]), __fmul_rn(xv[1], xv[1]));
#pragma unroll
    for (int i = 2; i < 16; ++i) {
      p = __fadd_rn(__fmul_rn(xv[i], xv[i]), p);
    }
  }
  p = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, 16));
  p = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, 8));
  p = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, 4));
  p = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, 2));
  p = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, 1));
  if ((t & 31) == 0) red[t >> 5] = p;
  __syncthreads();
  if (t < 32) {
    float v = (t < 16) ? red[t] : 0.0f;
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
    if (t == 0) red[0] = v;
  }
  __syncthreads();
  if (!valid) return;
  const float r = inv_rms(red[0], 1e-05f);
  if constexpr (!kPdl) {
    inv_scale = full_divide(1.0f, *scale);
    if constexpr (kGateOut) gate_inv = full_divide(1.0f, *gate_scale);
  }
  const uint32_t iw[8] = {ui0.x, ui0.y, ui0.z, ui0.w, ui1.x, ui1.y, ui1.z, ui1.w};
  uint32_t ob[8];
  uint16_t qf[8];
  uint16_t qg[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float y0 = __fmul_rn(__fmul_rn(r, xv[2 * i]), __fadd_rn(bf_lo(iw[i]), 1.0f));
    const float y1 = __fmul_rn(__fmul_rn(r, xv[2 * i + 1]), __fadd_rn(bf_hi(iw[i]), 1.0f));
    ob[i] = pack_bf16x2(y0, y1);
    qf[i] = fp8x2(clamp448(__fmul_rn(inv_scale, y0)), clamp448(__fmul_rn(inv_scale, y1)));
    if constexpr (kGateOut) {
      qg[i] = fp8x2(clamp448(__fmul_rn(gate_inv, bf_lo(ob[i]))),
                    clamp448(__fmul_rn(gate_inv, bf_hi(ob[i]))));
    }
  }
  const size_t off = row * kHidden + t * 16;
  if constexpr (kWriteBf16) {
    uint4* dst_b = reinterpret_cast<uint4*>(out_bf16_all + off);
    dst_b[0] = make_uint4(ob[0], ob[1], ob[2], ob[3]);
    dst_b[1] = make_uint4(ob[4], ob[5], ob[6], ob[7]);
  }
  auto pack4x16 = [](uint16_t a0, uint16_t a1, uint16_t a2, uint16_t a3,
                     uint16_t a4, uint16_t a5, uint16_t a6, uint16_t a7) -> uint4 {
    return make_uint4(
        static_cast<uint32_t>(a0) | (static_cast<uint32_t>(a1) << 16),
        static_cast<uint32_t>(a2) | (static_cast<uint32_t>(a3) << 16),
        static_cast<uint32_t>(a4) | (static_cast<uint32_t>(a5) << 16),
        static_cast<uint32_t>(a6) | (static_cast<uint32_t>(a7) << 16));
  };
  *reinterpret_cast<uint4*>(out_fp8_all + off) =
      pack4x16(qf[0], qf[1], qf[2], qf[3], qf[4], qf[5], qf[6], qf[7]);
  if constexpr (kGateOut) {
    *reinterpret_cast<uint4*>(out_gate_all + off) =
        pack4x16(qg[0], qg[1], qg[2], qg[3], qg[4], qg[5], qg[6], qg[7]);
  }
}

int g_pdl_override = -1;

static bool norm_glue_pdl_enabled() {
  if (g_pdl_override >= 0) return g_pdl_override != 0;
  static const bool env_on = []() {
    const char* env = std::getenv("MUSE_NORM_GLUE_PDL");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  return env_on;
}

static bool norm_glue_prefill_pdl_enabled() {
  if (g_pdl_override >= 0) return g_pdl_override != 0;
  static const bool env_on = []() {
    const char* e1 = std::getenv("MUSE_NORM_GLUE_PDL");
    if (e1 && *e1 && std::strtol(e1, nullptr, 10) == 0) return false;
    const char* e2 = std::getenv("MUSE_NORM_GLUE_PREFILL_PDL");
    return !e2 || !*e2 || std::strtol(e2, nullptr, 10) != 0;
  }();
  return env_on;
}

template <typename... KArgs, typename... Args>
cudaError_t launch_pdl(void (*kernel)(KArgs...), int rows, int threads, cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(rows, 1, 1);
  cfg.blockDim = dim3(threads, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...);
}

}  // namespace

extern "C" {

int muse_norm_glue_set_pdl(int on) {
  const int old = norm_glue_pdl_enabled() ? 1 : 0;
  g_pdl_override = (on < 0) ? -1 : (on != 0 ? 1 : 0);
  return old;
}

// kind: 0 = bf16 out, 1 = fp8 out (no bf16 out), as the compiled ..._rsqrt_1 kernels.
int muse_norm_glue_post_attn(int kind, int rows, const void* x, const void* res, const void* w_post,
                             const void* w_pre, const void* scale, void* sumsq_out, void* out_bf16,
                             void* out_fp8, cudaStream_t stream) {
  if (rows <= 0) return 0;
  const auto* X = static_cast<const __nv_bfloat16*>(x);
  const auto* Rs = static_cast<const __nv_bfloat16*>(res);
  const auto* Wp = static_cast<const __nv_bfloat16*>(w_post);
  const auto* Wq = static_cast<const __nv_bfloat16*>(w_pre);
  const bool use_pdl = (rows <= 32) && norm_glue_pdl_enabled();
  const bool use_prefill_pdl = (rows > 32) && norm_glue_prefill_pdl_enabled();
  if (kind == 0) {
    if (use_pdl) {
      return static_cast<int>(launch_pdl(post_attn_norm_decode<true, false, true>, rows, kThreads, stream,
          X, Rs, Wp, Wq, static_cast<const float*>(nullptr), static_cast<float*>(sumsq_out),
          static_cast<__nv_bfloat16*>(out_bf16), static_cast<uint8_t*>(nullptr)));
    } else if (rows <= 32) {
      post_attn_norm_decode<true, false><<<rows, kThreads, 0, stream>>>(X, Rs, Wp, Wq, nullptr,
          static_cast<float*>(sumsq_out), static_cast<__nv_bfloat16*>(out_bf16), nullptr);
    } else if (use_prefill_pdl) {
      return static_cast<int>(launch_pdl(post_attn_norm<true, false, true>, rows, kThreads, stream,
          X, Rs, Wp, Wq, static_cast<const float*>(nullptr), static_cast<float*>(sumsq_out),
          static_cast<__nv_bfloat16*>(out_bf16), static_cast<uint8_t*>(nullptr)));
    } else {
      post_attn_norm<true, false><<<rows, kThreads, 0, stream>>>(X, Rs, Wp, Wq, nullptr,
          static_cast<float*>(sumsq_out), static_cast<__nv_bfloat16*>(out_bf16), nullptr);
    }
  } else if (kind == 1) {
    if (use_pdl) {
      return static_cast<int>(launch_pdl(post_attn_norm_decode<false, true, true>, rows, kThreads, stream,
          X, Rs, Wp, Wq, static_cast<const float*>(scale), static_cast<float*>(sumsq_out),
          static_cast<__nv_bfloat16*>(nullptr), static_cast<uint8_t*>(out_fp8)));
    } else if (rows <= 32) {
      post_attn_norm_decode<false, true><<<rows, kThreads, 0, stream>>>(X, Rs, Wp, Wq, static_cast<const float*>(scale),
          static_cast<float*>(sumsq_out), nullptr, static_cast<uint8_t*>(out_fp8));
    } else if (use_prefill_pdl) {
      return static_cast<int>(launch_pdl(post_attn_norm<false, true, true>, rows, kThreads, stream,
          X, Rs, Wp, Wq, static_cast<const float*>(scale), static_cast<float*>(sumsq_out),
          static_cast<__nv_bfloat16*>(nullptr), static_cast<uint8_t*>(out_fp8)));
    } else {
      post_attn_norm<false, true><<<rows, kThreads, 0, stream>>>(X, Rs, Wp, Wq, static_cast<const float*>(scale),
          static_cast<float*>(sumsq_out), nullptr, static_cast<uint8_t*>(out_fp8));
    }
  } else {
    return -1;
  }
  return static_cast<int>(cudaGetLastError());
}

// kind: 0 = bf16 out, 1 = bf16 + fp8 out, as the compiled ..._rsqrt_3 kernels.
int muse_norm_glue_post_ff(int kind, int rows, void* x, const void* d, const void* res, const void* sumsq_x,
                           const void* w_post, const void* w_ff, const void* w_in, const void* scale,
                           void* out_bf16, void* out_fp8, cudaStream_t stream) {
  if (rows <= 0) return 0;
  auto* X = static_cast<__nv_bfloat16*>(x);
  const auto* D = static_cast<const __nv_bfloat16*>(d);
  const auto* Rs = static_cast<const __nv_bfloat16*>(res);
  const auto* S = static_cast<const float*>(sumsq_x);
  const auto* Wp = static_cast<const __nv_bfloat16*>(w_post);
  const auto* Wf = static_cast<const __nv_bfloat16*>(w_ff);
  const auto* Wi = static_cast<const __nv_bfloat16*>(w_in);
  static const bool prefill_cs = []() {
    const char* env = std::getenv("MUSE_POST_FF_PREFILL_CS");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  const bool use_pdl = (rows <= 32) && norm_glue_pdl_enabled();
  const bool use_prefill_pdl = (rows > 32) && norm_glue_prefill_pdl_enabled();
  if (kind == 0) {
    if (use_pdl) {
      return static_cast<int>(launch_pdl(post_ff_norm_decode<false, false, true, true>, rows, kThreads, stream,
          X, D, Rs, S, Wp, Wf, Wi, static_cast<const float*>(nullptr), static_cast<__nv_bfloat16*>(out_bf16),
          static_cast<uint8_t*>(nullptr), static_cast<const float*>(nullptr), static_cast<uint8_t*>(nullptr)));
    } else if (rows <= 32 || !prefill_cs) {
      post_ff_norm_decode<false><<<rows, kThreads, 0, stream>>>(X, D, Rs, S, Wp, Wf, Wi, nullptr,
          static_cast<__nv_bfloat16*>(out_bf16), nullptr);
    } else if (use_prefill_pdl) {
      return static_cast<int>(launch_pdl(post_ff_norm_prefill<false, true>, rows, kThreads, stream,
          X, D, Rs, S, Wp, Wf, Wi, static_cast<const float*>(nullptr),
          static_cast<__nv_bfloat16*>(out_bf16), static_cast<uint8_t*>(nullptr)));
    } else {
      post_ff_norm_prefill<false><<<rows, kThreads, 0, stream>>>(X, D, Rs, S, Wp, Wf, Wi, nullptr,
          static_cast<__nv_bfloat16*>(out_bf16), nullptr);
    }
  } else if (kind == 1) {
    if (use_pdl) {
      return static_cast<int>(launch_pdl(post_ff_norm_decode<true, false, true, true>, rows, kThreads, stream,
          X, D, Rs, S, Wp, Wf, Wi, static_cast<const float*>(scale), static_cast<__nv_bfloat16*>(out_bf16),
          static_cast<uint8_t*>(out_fp8), static_cast<const float*>(nullptr), static_cast<uint8_t*>(nullptr)));
    } else if (rows <= 32 || !prefill_cs) {
      post_ff_norm_decode<true><<<rows, kThreads, 0, stream>>>(X, D, Rs, S, Wp, Wf, Wi, static_cast<const float*>(scale),
          static_cast<__nv_bfloat16*>(out_bf16), static_cast<uint8_t*>(out_fp8));
    } else if (use_prefill_pdl) {
      return static_cast<int>(launch_pdl(post_ff_norm_prefill<true, true>, rows, kThreads, stream,
          X, D, Rs, S, Wp, Wf, Wi, static_cast<const float*>(scale),
          static_cast<__nv_bfloat16*>(out_bf16), static_cast<uint8_t*>(out_fp8)));
    } else {
      post_ff_norm_prefill<true><<<rows, kThreads, 0, stream>>>(X, D, Rs, S, Wp, Wf, Wi, static_cast<const float*>(scale),
          static_cast<__nv_bfloat16*>(out_bf16), static_cast<uint8_t*>(out_fp8));
    }
  } else {
    return -1;
  }
  return static_cast<int>(cudaGetLastError());
}

int muse_norm_glue_post_ff_final(int rows, const void* d, const void* res, const void* x, const void* sumsq_x,
                                 const void* w_post, const void* w_ff, const void* w_in,
                                 void* out_h2, void* out_bf16, int write_h2, cudaStream_t stream) {
  if (rows <= 0) return 0;
  if (rows <= 32 && norm_glue_pdl_enabled()) {
    auto k = write_h2 ? post_ff_final_norm<true, true> : post_ff_final_norm<false, true>;
    return static_cast<int>(launch_pdl(k, rows, kThreads, stream,
        static_cast<const __nv_bfloat16*>(d), static_cast<const __nv_bfloat16*>(res),
        static_cast<const __nv_bfloat16*>(x), static_cast<const float*>(sumsq_x),
        static_cast<const __nv_bfloat16*>(w_post), static_cast<const __nv_bfloat16*>(w_ff),
        static_cast<const __nv_bfloat16*>(w_in),
        static_cast<float*>(out_h2), static_cast<__nv_bfloat16*>(out_bf16)));
  }
  if (write_h2) {
    post_ff_final_norm<true><<<rows, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(d), static_cast<const __nv_bfloat16*>(res),
        static_cast<const __nv_bfloat16*>(x), static_cast<const float*>(sumsq_x),
        static_cast<const __nv_bfloat16*>(w_post), static_cast<const __nv_bfloat16*>(w_ff),
        static_cast<const __nv_bfloat16*>(w_in),
        static_cast<float*>(out_h2), static_cast<__nv_bfloat16*>(out_bf16));
  } else {
    post_ff_final_norm<false><<<rows, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(d), static_cast<const __nv_bfloat16*>(res),
        static_cast<const __nv_bfloat16*>(x), static_cast<const float*>(sumsq_x),
        static_cast<const __nv_bfloat16*>(w_post), static_cast<const __nv_bfloat16*>(w_ff),
        static_cast<const __nv_bfloat16*>(w_in),
        static_cast<float*>(out_h2), static_cast<__nv_bfloat16*>(out_bf16));
  }
  return static_cast<int>(cudaGetLastError());
}

int muse_norm_glue_input_fp8(int warps, int rows, const void* x, const void* w_in, const void* scale,
                             void* out_bf16, void* out_fp8, cudaStream_t stream) {
  if (rows <= 0) return 0;
  const auto* X = static_cast<const __nv_bfloat16*>(x);
  const auto* Wi = static_cast<const __nv_bfloat16*>(w_in);
  const auto* Sc = static_cast<const float*>(scale);
  auto* Ob = static_cast<__nv_bfloat16*>(out_bf16);
  auto* Of = static_cast<uint8_t*>(out_fp8);
  const bool use_pdl = (rows <= 32) && norm_glue_pdl_enabled();
  if (warps == 16) {
    if (use_pdl) {
      return static_cast<int>(launch_pdl(input_norm_fp8_w16<false, true, true>, rows, 512, stream,
          X, Wi, Sc, Ob, Of, static_cast<const float*>(nullptr), static_cast<uint8_t*>(nullptr)));
    }
    input_norm_fp8_w16<false, true><<<rows, 512, 0, stream>>>(X, Wi, Sc, Ob, Of);
  } else if (warps == 8) {
    if (use_pdl) {
      return static_cast<int>(launch_pdl(input_norm_fp8_w8<false, true, true>, rows, kThreads, stream,
          X, Wi, Sc, Ob, Of, static_cast<const float*>(nullptr), static_cast<uint8_t*>(nullptr)));
    }
    input_norm_fp8_w8<false, true><<<rows, kThreads, 0, stream>>>(X, Wi, Sc, Ob, Of);
  } else {
    return -1;
  }
  return static_cast<int>(cudaGetLastError());
}

int muse_norm_glue_input_fp8_gate(int warps, int rows, const void* x, const void* w_in, const void* scale,
                                  void* out_bf16, void* out_fp8, const void* gate_scale, void* out_gate,
                                  int write_bf16, cudaStream_t stream) {
  if (rows <= 0) return 0;
  const auto* X = static_cast<const __nv_bfloat16*>(x);
  const auto* Wi = static_cast<const __nv_bfloat16*>(w_in);
  const auto* Sc = static_cast<const float*>(scale);
  auto* Ob = static_cast<__nv_bfloat16*>(out_bf16);
  auto* Of = static_cast<uint8_t*>(out_fp8);
  const auto* Gs = static_cast<const float*>(gate_scale);
  auto* Og = static_cast<uint8_t*>(out_gate);
  const bool use_pdl = (rows <= 32) && norm_glue_pdl_enabled();
  if (warps == 16) {
    if (use_pdl) {
      auto k = write_bf16 ? input_norm_fp8_w16<true, true, true> : input_norm_fp8_w16<true, false, true>;
      return static_cast<int>(launch_pdl(k, rows, 512, stream, X, Wi, Sc, Ob, Of, Gs, Og));
    }
    if (write_bf16) {
      input_norm_fp8_w16<true, true><<<rows, 512, 0, stream>>>(X, Wi, Sc, Ob, Of, Gs, Og);
    } else {
      input_norm_fp8_w16<true, false><<<rows, 512, 0, stream>>>(X, Wi, Sc, Ob, Of, Gs, Og);
    }
  } else if (warps == 8) {
    if (use_pdl) {
      auto k = write_bf16 ? input_norm_fp8_w8<true, true, true> : input_norm_fp8_w8<true, false, true>;
      return static_cast<int>(launch_pdl(k, rows, kThreads, stream, X, Wi, Sc, Ob, Of, Gs, Og));
    }
    if (write_bf16) {
      input_norm_fp8_w8<true, true><<<rows, kThreads, 0, stream>>>(X, Wi, Sc, Ob, Of, Gs, Og);
    } else {
      input_norm_fp8_w8<true, false><<<rows, kThreads, 0, stream>>>(X, Wi, Sc, Ob, Of, Gs, Og);
    }
  } else {
    return -1;
  }
  return static_cast<int>(cudaGetLastError());
}

int muse_norm_glue_post_ff_gate(int rows, void* x, const void* d, const void* res, const void* sumsq_x,
                                const void* w_post, const void* w_ff, const void* w_in, const void* scale,
                                void* out_bf16, void* out_fp8, const void* gate_scale, void* out_gate,
                                int write_bf16, cudaStream_t stream) {
  if (rows <= 0) return 0;
  auto* X = static_cast<__nv_bfloat16*>(x);
  const auto* D = static_cast<const __nv_bfloat16*>(d);
  const auto* Rs = static_cast<const __nv_bfloat16*>(res);
  const auto* S = static_cast<const float*>(sumsq_x);
  const auto* Wp = static_cast<const __nv_bfloat16*>(w_post);
  const auto* Wf = static_cast<const __nv_bfloat16*>(w_ff);
  const auto* Wi = static_cast<const __nv_bfloat16*>(w_in);
  const auto* Sc = static_cast<const float*>(scale);
  const auto* Gs = static_cast<const float*>(gate_scale);
  auto* Ob = static_cast<__nv_bfloat16*>(out_bf16);
  auto* Of = static_cast<uint8_t*>(out_fp8);
  auto* Og = static_cast<uint8_t*>(out_gate);
  if (rows <= 32 && norm_glue_pdl_enabled()) {
    auto k = write_bf16 ? post_ff_norm_decode<true, true, true, true> : post_ff_norm_decode<true, true, false, true>;
    return static_cast<int>(launch_pdl(k, rows, kThreads, stream, X, D, Rs, S, Wp, Wf, Wi, Sc, Ob, Of, Gs, Og));
  }
  if (write_bf16) {
    post_ff_norm_decode<true, true, true><<<rows, kThreads, 0, stream>>>(X, D, Rs, S, Wp, Wf, Wi, Sc, Ob, Of, Gs, Og);
  } else {
    post_ff_norm_decode<true, true, false><<<rows, kThreads, 0, stream>>>(X, D, Rs, S, Wp, Wf, Wi, Sc, Ob, Of, Gs, Og);
  }
  return static_cast<int>(cudaGetLastError());
}

int muse_norm_glue_gate_formula(const void* bits, const void* gate_scale, void* out, int n,
                                cudaStream_t stream) {
  if (n <= 0) return 0;
  gate_formula_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      static_cast<const uint16_t*>(bits), static_cast<const float*>(gate_scale),
      static_cast<uint8_t*>(out), n);
  return static_cast<int>(cudaGetLastError());
}

int muse_norm_glue_mul_sigmoid_bf16(const void* gate, const void* attn, void* out, int n,
                                    cudaStream_t stream) {
  if (n <= 0 || (n & 7) != 0) return -1;
  const int vec8 = n >> 3;
  const int blocks = (vec8 + 255) / 256;
  if (n <= 4096 * 16) {
    mul_sigmoid_bf16_kernel<false, false><<<blocks, 256, 0, stream>>>(
        static_cast<const uint4*>(gate), static_cast<const uint4*>(attn),
        static_cast<uint4*>(out), vec8);
  } else if (norm_glue_prefill_pdl_enabled()) {
    mul_sigmoid_bf16_kernel<true, true><<<blocks, 256, 0, stream>>>(
        static_cast<const uint4*>(gate), static_cast<const uint4*>(attn),
        static_cast<uint4*>(out), vec8);
  } else {
    mul_sigmoid_bf16_kernel<true, false><<<blocks, 256, 0, stream>>>(
        static_cast<const uint4*>(gate), static_cast<const uint4*>(attn),
        static_cast<uint4*>(out), vec8);
  }
  return static_cast<int>(cudaGetLastError());
}

int muse_norm_glue_mul_sigmoid_fp8(const void* gate, const void* attn, const void* scale, void* out, int n,
                                   cudaStream_t stream) {
  if (n <= 0 || (n & 7) != 0) return -1;
  const int vec8 = n >> 3;
  const int blocks = (vec8 + 255) / 256;
  if (n > 4096 * 16 && norm_glue_prefill_pdl_enabled()) {
    mul_sigmoid_fp8_kernel<true><<<blocks, 256, 0, stream>>>(
        static_cast<const uint4*>(gate), static_cast<const uint4*>(attn),
        static_cast<const float*>(scale), static_cast<uint2*>(out), vec8);
  } else {
    mul_sigmoid_fp8_kernel<false><<<blocks, 256, 0, stream>>>(
        static_cast<const uint4*>(gate), static_cast<const uint4*>(attn),
        static_cast<const float*>(scale), static_cast<uint2*>(out), vec8);
  }
  return static_cast<int>(cudaGetLastError());
}

int muse_norm_glue_mul_silu_slice_bf16(const void* gu, void* out, int rows, cudaStream_t stream) {
  if (rows <= 0) return -1;
  const dim3 grid((kInterVec8 + 127) / 128, static_cast<unsigned>(rows));
  mul_silu_slice_bf16_kernel<<<grid, 128, 0, stream>>>(
      static_cast<const uint4*>(gu), static_cast<uint4*>(out));
  return static_cast<int>(cudaGetLastError());
}

int muse_norm_glue_mul_silu_slice_fp8(const void* gu, const void* scale, void* out, int rows, cudaStream_t stream) {
  if (rows <= 0) return -1;
  const dim3 grid((kInterVec8 + 127) / 128, static_cast<unsigned>(rows));
  mul_silu_slice_fp8_kernel<<<grid, 128, 0, stream>>>(
      static_cast<const uint4*>(gu), static_cast<const float*>(scale), static_cast<uint2*>(out));
  return static_cast<int>(cudaGetLastError());
}

}  // extern "C"