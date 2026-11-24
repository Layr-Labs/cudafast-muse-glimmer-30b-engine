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
#include <stdint.h>
#include <stdlib.h>
#include <atomic>

namespace {

constexpr int kHidden = 6656;
constexpr int kThreads = 256;
constexpr int kChunks = 7;  // ceil(6656 / 1024); chunk 6 covers lanes 0..511 only
constexpr int kRowCacheLines = (kHidden * 2) / 128;  // 13,312 B / 128 B = 104 exact cache lines

__device__ __forceinline__ void gdc_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void gdc_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ void discard_l2_128(const void* addr) {
  asm volatile("discard.global.L2 [%0], 128;" :: "l"(addr) : "memory");
}

__device__ __forceinline__ void st_el_b32(void* addr, uint32_t v) {
  *reinterpret_cast<uint32_t*>(addr) = v;
}

__device__ __forceinline__ void st_el_v2_b32(void* addr, uint2 v) {
  *reinterpret_cast<uint2*>(addr) = v;
}

__device__ __forceinline__ void st_el_v4_b32(void* addr, uint4 v) {
  *reinterpret_cast<uint4*>(addr) = v;
}

__device__ __forceinline__ void st_el_v4_f32(float* addr, float x, float y, float z, float w) {
  *reinterpret_cast<float4*>(addr) = make_float4(x, y, z, w);
}

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

__device__ __forceinline__ uint2 ld8(const __nv_bfloat16* base, int c) {
  return *reinterpret_cast<const uint2*>(base + c * 1024 + threadIdx.x * 4);
}

__device__ __forceinline__ uint2 ldg8(const __nv_bfloat16* base, int c) {
  return __ldg(reinterpret_cast<const uint2*>(base + c * 1024 + threadIdx.x * 4));
}

// ---------------------------------------------------------------------------------------------------------
// Post-attention norm + residual + pre-feedforward norm (compiled ..._mean_mul_pow_rsqrt_1 and the FP8 variant
// ..._clamp_..._reciprocal_rsqrt_1):
//   s_x = sum x^2 (stored),  a = inv_rms(s_x, 1e-8)
//   h   = fma(a * x, w_post + 1, res)          (FMUL, FADD, FFMA)
//   s_h = sum h^2,           b = inv_rms(s_h, 1e-5)
//   y   = (b * h) * (w_pre + 1)                 (FMUL, FADD, FMUL)
//   out_bf16 = bf16(y);  out_fp8 = e4m3(clamp(rcp(scale) * y))
template <bool kBf16Out, bool kFp8Out, bool kWriteH1 = false>
__global__ void __launch_bounds__(kThreads) post_attn_norm_decode(
    const __nv_bfloat16* x_all, const __nv_bfloat16* res_all,
    const __nv_bfloat16* __restrict__ w_post, const __nv_bfloat16* __restrict__ w_pre,
    const float* scale, float* sumsq_out,
    __nv_bfloat16* out_bf16_all, uint8_t* out_fp8_all,
    float* h1_out = nullptr, int discard_mask = 0) {
  __shared__ float red[2][8];
  const size_t row = blockIdx.x;
  const __nv_bfloat16* x = x_all + row * kHidden;
  const __nv_bfloat16* res = res_all + row * kHidden;
  uint2 ux[kChunks], ur[kChunks], uw[kChunks], up[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      uw[c] = ld8(w_post, c);
      up[c] = ld8(w_pre, c);
    }
  }
  gdc_wait();
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      ux[c] = ld8(x, c);
      ur[c] = ld8(res, c);
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
  if (threadIdx.x == 0 && sumsq_out != nullptr) sumsq_out[row] = s_x;
  const float a = inv_rms(s_x, 1e-08f);
  float h[kChunks][4];
  float acc2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 xv = unpack(ux[c]), rv = unpack(ur[c]), wv = unpack(uw[c]);
      float h1_vals[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float w1 = __fadd_rn(wv.v[e], 1.0f);
        const float ax = __fmul_rn(a, xv.v[e]);
        h[c][e] = __fmaf_rn(ax, w1, rv.v[e]);
        acc2[e] = __fmaf_rn(h[c][e], h[c][e], acc2[e]);
        if constexpr (kWriteH1) {
          h1_vals[e] = __fadd_rn(rv.v[e], __fmul_rn(ax, w1));
        }
      }
      if constexpr (kWriteH1) {
        const size_t off = row * kHidden + c * 1024 + threadIdx.x * 4;
        st_el_v4_f32(h1_out + off, h1_vals[0], h1_vals[1], h1_vals[2], h1_vals[3]);
      }
    }
  }
  const float s_h = row_sum(acc2, red[1]);
  const float b = inv_rms(s_h, 1e-05f);
  float inv_scale = 0.f;
  if constexpr (kFp8Out) inv_scale = rcp_approx_ftz(*scale);
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 pv = unpack(up[c]);
      float y[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) y[e] = __fmul_rn(__fmul_rn(b, h[c][e]), __fadd_rn(pv.v[e], 1.0f));
      const size_t off = row * kHidden + c * 1024 + threadIdx.x * 4;
      if constexpr (kBf16Out) {
        uint2 o;
        o.x = pack_bf16x2(y[0], y[1]);
        o.y = pack_bf16x2(y[2], y[3]);
        st_el_v2_b32(out_bf16_all + off, o);
      }
      if constexpr (kFp8Out) {
        uint32_t q = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[0])), clamp448(__fmul_rn(inv_scale, y[1])))) |
                     (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[2])), clamp448(__fmul_rn(inv_scale, y[3])))) << 16);
        st_el_b32(out_fp8_all + off, q);
      }
    }
  }
  if constexpr (kWriteH1) {
    __syncthreads();
    if (gridDim.x == 1 && threadIdx.x < kRowCacheLines) {
      const uintptr_t x_p = reinterpret_cast<uintptr_t>(x);
      const uintptr_t res_p = reinterpret_cast<uintptr_t>(res);
      const uintptr_t h1_p = reinterpret_cast<uintptr_t>(h1_out);
      const uintptr_t ob_p = reinterpret_cast<uintptr_t>(out_bf16_all);
      const uintptr_t of_p = reinterpret_cast<uintptr_t>(out_fp8_all);
      auto disjoint_1row = [](uintptr_t a, uintptr_t b, size_t b_bytes) {
        if (b == 0u || b_bytes == 0u) return true;
        return (a < b) ? (b - a >= 13312u) : (a - b >= b_bytes);
      };
      if ((discard_mask & 1) && (x_p & 127u) == 0u &&
          disjoint_1row(x_p, h1_p, 26624u) &&
          disjoint_1row(x_p, ob_p, 13312u) &&
          disjoint_1row(x_p, of_p, 6656u) &&
          disjoint_1row(x_p, res_p, 13312u)) {
        discard_l2_128(reinterpret_cast<const uint8_t*>(x) + threadIdx.x * 128);
      }
      if ((discard_mask & 2) && (res_p & 127u) == 0u &&
          disjoint_1row(res_p, h1_p, 26624u) &&
          disjoint_1row(res_p, ob_p, 13312u) &&
          disjoint_1row(res_p, of_p, 6656u)) {
        discard_l2_128(reinterpret_cast<const uint8_t*>(res) + threadIdx.x * 128);
      }
    }
    __syncthreads();
  }
  gdc_launch_dependents();
}

template <bool kBf16Out, bool kFp8Out>
__global__ void __launch_bounds__(kThreads, 4) post_attn_norm(
    const __nv_bfloat16* __restrict__ x_all, const __nv_bfloat16* __restrict__ res_all,
    const __nv_bfloat16* __restrict__ w_post, const __nv_bfloat16* __restrict__ w_pre,
    const float* __restrict__ scale, float* __restrict__ sumsq_out,
    __nv_bfloat16* __restrict__ out_bf16_all, uint8_t* __restrict__ out_fp8_all) {
  __shared__ float red[2][8];
  const size_t row = blockIdx.x;
  const __nv_bfloat16* x = x_all + row * kHidden;
  const __nv_bfloat16* res = res_all + row * kHidden;
  uint2 ux[kChunks];
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      ux[c] = ld8(x, c);
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
      Row4 xv = unpack(ux[c]), rv = unpack(ld8(res, c)), wv = unpack(ldg8(w_post, c));
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h[c][e] = __fmaf_rn(__fmul_rn(a, xv.v[e]), __fadd_rn(wv.v[e], 1.0f), rv.v[e]);
        acc2[e] = __fmaf_rn(h[c][e], h[c][e], acc2[e]);
      }
    }
  }
  const float s_h = row_sum(acc2, red[1]);
  const float b = inv_rms(s_h, 1e-05f);
  float inv_scale = 0.f;
  if constexpr (kFp8Out) inv_scale = rcp_approx_ftz(*scale);
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 pv = unpack(ldg8(w_pre, c));
      float y[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) y[e] = __fmul_rn(__fmul_rn(b, h[c][e]), __fadd_rn(pv.v[e], 1.0f));
      const size_t off = row * kHidden + c * 1024 + threadIdx.x * 4;
      if constexpr (kBf16Out) {
        uint2 o;
        o.x = pack_bf16x2(y[0], y[1]);
        o.y = pack_bf16x2(y[2], y[3]);
        *reinterpret_cast<uint2*>(out_bf16_all + off) = o;
      }
      if constexpr (kFp8Out) {
        uint32_t q = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[0])), clamp448(__fmul_rn(inv_scale, y[1])))) |
                     (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[2])), clamp448(__fmul_rn(inv_scale, y[3])))) << 16);
        *reinterpret_cast<uint32_t*>(out_fp8_all + off) = q;
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
template <bool kFp8Out, bool kGateOut = false, bool kWriteBf16 = true, bool kReadH1 = false>
__global__ void __launch_bounds__(kThreads) post_ff_norm(
    __nv_bfloat16* x_all, const __nv_bfloat16* d_all,
    const __nv_bfloat16* res_all, const float* sumsq_x,
    const __nv_bfloat16* __restrict__ w_post, const __nv_bfloat16* __restrict__ w_ff,
    const __nv_bfloat16* w_in, const float* scale,
    __nv_bfloat16* out_bf16_all, uint8_t* out_fp8_all,
    const float* gate_scale = nullptr, uint8_t* out_gate_all = nullptr,
    const float* h1_all = nullptr) {
  __shared__ float red[2][8];
  const size_t row = blockIdx.x;
  __nv_bfloat16* x = x_all + row * kHidden;
  const __nv_bfloat16* d = d_all + row * kHidden;
  const __nv_bfloat16* res = res_all ? (res_all + row * kHidden) : nullptr;
  const float* h1_row = h1_all ? (h1_all + row * kHidden) : nullptr;
  uint2 ux[kChunks], ud[kChunks], ur[kChunks], uw[kChunks], uf[kChunks], ui[kChunks];
  float4 uh1[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      if constexpr (!kReadH1) {
        uw[c] = ld8(w_post, c);
      }
      uf[c] = ld8(w_ff, c);
      ui[c] = ld8(w_in, c);
    }
  }
  gdc_wait();
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      ud[c] = ld8(d, c);
      if constexpr (kReadH1) {
        uh1[c] = *reinterpret_cast<const float4*>(h1_row + c * 1024 + threadIdx.x * 4);
      } else {
        ux[c] = ld8(x, c);
        ur[c] = ld8(res, c);
      }
    }
  }
  const float s_x = kReadH1 ? 0.0f : sumsq_x[row];
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
  const float a = kReadH1 ? 0.0f : inv_rms(s_x, 1e-08f);
  float acc2[4] = {0.f, 0.f, 0.f, 0.f};
  uint2 hb[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 dv = unpack(ud[c]), fv = unpack(uf[c]);
      float h1_vals[4];
      if constexpr (kReadH1) {
        h1_vals[0] = uh1[c].x;
        h1_vals[1] = uh1[c].y;
        h1_vals[2] = uh1[c].z;
        h1_vals[3] = uh1[c].w;
      } else {
        Row4 xv = unpack(ux[c]), rv = unpack(ur[c]), wv = unpack(uw[c]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float t1 = __fmul_rn(__fmul_rn(a, xv.v[e]), __fadd_rn(wv.v[e], 1.0f));
          h1_vals[e] = __fadd_rn(rv.v[e], t1);
        }
      }
      float h2[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h2[e] = __fmaf_rn(__fmul_rn(cd, dv.v[e]), __fadd_rn(fv.v[e], 1.0f), h1_vals[e]);
        acc2[e] = __fmaf_rn(h2[e], h2[e], acc2[e]);
      }
      hb[c].x = pack_bf16x2(h2[0], h2[1]);
      hb[c].y = pack_bf16x2(h2[2], h2[3]);
      *reinterpret_cast<uint2*>(x + c * 1024 + threadIdx.x * 4) = hb[c];
    }
  }
  const float s_h = row_sum(acc2, red[1]);
  const float e_r = inv_rms(s_h, 1e-05f);
  float inv_scale = 0.f;
  float gate_inv = 0.f;
  if constexpr (kFp8Out) inv_scale = rcp_approx_ftz(*scale);
  if constexpr (kGateOut) gate_inv = full_divide(1.0f, *gate_scale);
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 hv = unpack(hb[c]), iv = unpack(ui[c]);
      float y[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) y[e] = __fmul_rn(__fmul_rn(e_r, hv.v[e]), __fadd_rn(iv.v[e], 1.0f));
      const size_t off = row * kHidden + c * 1024 + threadIdx.x * 4;
      uint2 o;
      o.x = pack_bf16x2(y[0], y[1]);
      o.y = pack_bf16x2(y[2], y[3]);
      if constexpr (kWriteBf16) {
        st_el_v2_b32(out_bf16_all + off, o);
      }
      if constexpr (kFp8Out) {
        uint32_t q = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[0])), clamp448(__fmul_rn(inv_scale, y[1])))) |
                     (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[2])), clamp448(__fmul_rn(inv_scale, y[3])))) << 16);
        st_el_b32(out_fp8_all + off, q);
      }
      if constexpr (kGateOut) {
        const float g0 = bf_lo(o.x), g1 = bf_hi(o.x), g2 = bf_lo(o.y), g3 = bf_hi(o.y);
        uint32_t gq = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(gate_inv, g0)), clamp448(__fmul_rn(gate_inv, g1)))) |
                      (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(gate_inv, g2)), clamp448(__fmul_rn(gate_inv, g3)))) << 16);
        st_el_b32(out_gate_all + off, gq);
      }
    }
  }
  gdc_launch_dependents();
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

__global__ void __launch_bounds__(256) mul_sigmoid_bf16_kernel(
    const uint4* __restrict__ gate, const uint4* __restrict__ attn, uint4* __restrict__ out, int vec8) {
  const int idx = blockIdx.x * 256 + threadIdx.x;
  gdc_wait();
  if (idx >= vec8) return;
  const uint4 g = gate[idx], a = attn[idx];
  const uint32_t gw[4] = {g.x, g.y, g.z, g.w}, aw[4] = {a.x, a.y, a.z, a.w};
  uint32_t ow[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float gl = bf_lo(gw[i]), gh = bf_hi(gw[i]);
    const float al = bf_lo(aw[i]), ah = bf_hi(aw[i]);
    ow[i] = pack_bf16x2(__fmul_rn(triton_sigmoid(gl), al), __fmul_rn(triton_sigmoid(gh), ah));
  }
  out[idx] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
  gdc_launch_dependents();
}

__global__ void __launch_bounds__(256) mul_sigmoid_fp8_kernel(
    const uint4* __restrict__ gate, const uint4* __restrict__ attn,
    const float* __restrict__ scale, uint2* __restrict__ out, int vec8) {
  const int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= vec8) return;
  const float inv_s = full_divide(1.0f, __ldg(scale));
  const uint4 g = gate[idx], a = attn[idx];
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
  gdc_wait();
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
  gdc_launch_dependents();
}

// Layer-51 final post-FF norm + residual + model.norm (compiled bf5b0726...):
//   tensors: (d, res, x, sumsq_x, w_post, w_ff, w_in, out_h2_fp32, out_bf16)
//   h1 = res + (a * x) * (w_post + 1)             (FMUL, FADD, FMUL, FADD)
//   h2 = fma(cd * d, w_ff + 1, h1)                (FMUL, FADD, FFMA — matches compiled SASS)
//   out_h2_fp32 <- h2 (if kWriteH2);  s_h = sum h2^2;  e_r = inv_rms(s_h, 1e-5)
//   y  = (e_r * h2) * (w_in + 0.0f);  out_bf16 <- bf16(y)
template <bool kWriteH2 = true, bool kReadH1 = false>
__global__ void __launch_bounds__(kThreads) post_ff_final_norm(
    const __nv_bfloat16* d_all, const __nv_bfloat16* res_all,
    const __nv_bfloat16* x_all, const float* sumsq_x,
    const __nv_bfloat16* __restrict__ w_post, const __nv_bfloat16* __restrict__ w_ff,
    const __nv_bfloat16* __restrict__ w_in,
    float* out_h2_all, __nv_bfloat16* out_bf16_all,
    const float* h1_all = nullptr) {
  __shared__ float red[2][8];
  const size_t row = blockIdx.x;
  const __nv_bfloat16* d = d_all + row * kHidden;
  const __nv_bfloat16* res = res_all ? (res_all + row * kHidden) : nullptr;
  const __nv_bfloat16* x = x_all ? (x_all + row * kHidden) : nullptr;
  const float* h1_row = h1_all ? (h1_all + row * kHidden) : nullptr;
  uint2 ud[kChunks], ur[kChunks], ux[kChunks], uw[kChunks], uf[kChunks], ui[kChunks];
  float4 uh1[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      if constexpr (!kReadH1) {
        uw[c] = ld8(w_post, c);
      }
      uf[c] = ld8(w_ff, c);
      ui[c] = ld8(w_in, c);
    }
  }
  gdc_wait();
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      ud[c] = ld8(d, c);
      if constexpr (kReadH1) {
        uh1[c] = *reinterpret_cast<const float4*>(h1_row + c * 1024 + threadIdx.x * 4);
      } else {
        ur[c] = ld8(res, c);
        ux[c] = ld8(x, c);
      }
    }
  }
  const float s_x = kReadH1 ? 0.0f : sumsq_x[row];
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
  const float a = kReadH1 ? 0.0f : inv_rms(s_x, 1e-08f);
  const float cd = inv_rms(s_d, 1e-08f);
  float h2[kChunks][4];
  float acc2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 dv = unpack(ud[c]), fv = unpack(uf[c]);
      float h1_vals[4];
      if constexpr (kReadH1) {
        h1_vals[0] = uh1[c].x;
        h1_vals[1] = uh1[c].y;
        h1_vals[2] = uh1[c].z;
        h1_vals[3] = uh1[c].w;
      } else {
        Row4 xv = unpack(ux[c]), rv = unpack(ur[c]), wv = unpack(uw[c]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float t1 = __fmul_rn(__fmul_rn(a, xv.v[e]), __fadd_rn(wv.v[e], 1.0f));
          h1_vals[e] = __fadd_rn(rv.v[e], t1);
        }
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h2[c][e] = __fmaf_rn(__fmul_rn(cd, dv.v[e]), __fadd_rn(fv.v[e], 1.0f), h1_vals[e]);
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
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 iv = unpack(ui[c]);
      float y[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        y[e] = __fmul_rn(__fmul_rn(e_r, h2[c][e]), __fadd_rn(iv.v[e], 0.0f));
      }
      const size_t off = row * kHidden + c * 1024 + threadIdx.x * 4;
      uint2 o;
      o.x = pack_bf16x2(y[0], y[1]);
      o.y = pack_bf16x2(y[2], y[3]);
      st_el_v2_b32(out_bf16_all + off, o);
    }
  }
  gdc_launch_dependents();
}

// Layer-0 initial input_layernorm + FP8 qkv quant (+ optional gate FP8 quant fold), 8-warp (R0_BLOCK=1024) layout:
template <bool kGateOut = false, bool kWriteBf16 = true>
__global__ void __launch_bounds__(kThreads) input_norm_fp8_w8(
    const __nv_bfloat16* x_all, const __nv_bfloat16* __restrict__ w_in,
    const float* scale, __nv_bfloat16* out_bf16_all, uint8_t* out_fp8_all,
    const float* gate_scale = nullptr, uint8_t* out_gate_all = nullptr) {
  __shared__ float red[8];
  const size_t row = blockIdx.x;
  const __nv_bfloat16* x = x_all + row * kHidden;
  uint2 ux[kChunks], ui[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      ui[c] = ld8(w_in, c);
    }
  }
  gdc_wait();
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      ux[c] = ld8(x, c);
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
  const float inv_scale = full_divide(1.0f, *scale);
  float gate_inv = 0.f;
  if constexpr (kGateOut) gate_inv = full_divide(1.0f, *gate_scale);
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 xv = unpack(ux[c]), iv = unpack(ui[c]);
      float y[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) y[e] = __fmul_rn(__fmul_rn(r, xv.v[e]), __fadd_rn(iv.v[e], 1.0f));
      const size_t off = row * kHidden + c * 1024 + threadIdx.x * 4;
      uint2 o;
      o.x = pack_bf16x2(y[0], y[1]);
      o.y = pack_bf16x2(y[2], y[3]);
      if constexpr (kWriteBf16) {
        st_el_v2_b32(out_bf16_all + off, o);
      }
      uint32_t q = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[0])), clamp448(__fmul_rn(inv_scale, y[1])))) |
                   (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[2])), clamp448(__fmul_rn(inv_scale, y[3])))) << 16);
      st_el_b32(out_fp8_all + off, q);
      if constexpr (kGateOut) {
        const float g0 = bf_lo(o.x), g1 = bf_hi(o.x), g2 = bf_lo(o.y), g3 = bf_hi(o.y);
        uint32_t gq = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(gate_inv, g0)), clamp448(__fmul_rn(gate_inv, g1)))) |
                      (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(gate_inv, g2)), clamp448(__fmul_rn(gate_inv, g3)))) << 16);
        st_el_b32(out_gate_all + off, gq);
      }
    }
  }
  gdc_launch_dependents();
}

// Layer-0 initial input_layernorm + FP8 qkv quant (+ optional gate FP8 quant fold), 16-warp (R0_BLOCK=8192) layout:
//   sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 16] (512 threads; t < 416 owns 16 elements).
template <bool kGateOut = false, bool kWriteBf16 = true>
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
  if (valid) {
    const uint4* ip = reinterpret_cast<const uint4*>(w_in + t * 16);
    ui0 = ip[0];
    ui1 = ip[1];
  }
  gdc_wait();
  if (valid) {
    const uint4* xp = reinterpret_cast<const uint4*>(x_all + row * kHidden + t * 16);
    ux0 = xp[0];
    ux1 = xp[1];
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
  const float inv_scale = full_divide(1.0f, *scale);
  float gate_inv = 0.f;
  if constexpr (kGateOut) gate_inv = full_divide(1.0f, *gate_scale);
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
    st_el_v4_b32(&dst_b[0], make_uint4(ob[0], ob[1], ob[2], ob[3]));
    st_el_v4_b32(&dst_b[1], make_uint4(ob[4], ob[5], ob[6], ob[7]));
  }
  auto pack4x16 = [](uint16_t a0, uint16_t a1, uint16_t a2, uint16_t a3,
                     uint16_t a4, uint16_t a5, uint16_t a6, uint16_t a7) -> uint4 {
    return make_uint4(
        static_cast<uint32_t>(a0) | (static_cast<uint32_t>(a1) << 16),
        static_cast<uint32_t>(a2) | (static_cast<uint32_t>(a3) << 16),
        static_cast<uint32_t>(a4) | (static_cast<uint32_t>(a5) << 16),
        static_cast<uint32_t>(a6) | (static_cast<uint32_t>(a7) << 16));
  };
  st_el_v4_b32(out_fp8_all + off, pack4x16(qf[0], qf[1], qf[2], qf[3], qf[4], qf[5], qf[6], qf[7]));
  if constexpr (kGateOut) {
    st_el_v4_b32(out_gate_all + off, pack4x16(qg[0], qg[1], qg[2], qg[3], qg[4], qg[5], qg[6], qg[7]));
  }
  gdc_launch_dependents();
}

static std::atomic<int> g_norm_pdl_state{-1};  // -1 = uninitialized (use env), 0 = off, 1 = on

inline bool norm_pdl_enabled() {
  int v = g_norm_pdl_state.load(std::memory_order_relaxed);
  if (v < 0) {
    const char* e = getenv("MUSE_NORM_PDL");
    v = (e && e[0] == '0' && e[1] == '\0') ? 0 : 1;
    int exp = -1;
    g_norm_pdl_state.compare_exchange_strong(exp, v, std::memory_order_relaxed);
    if (exp >= 0) v = exp;
  }
  return v != 0;
}

template <typename KernelT, typename... Args>
inline cudaError_t launch_norm_kernel(
    dim3 grid, dim3 block, cudaStream_t stream, bool allow_pdl, KernelT kernel, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = (allow_pdl && norm_pdl_enabled()) ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace

extern "C" {

void muse_norm_glue_set_pdl(int enabled) {
  if (enabled < 0) {
    const char* e = getenv("MUSE_NORM_PDL");
    g_norm_pdl_state.store((e && e[0] == '0' && e[1] == '\0') ? 0 : 1, std::memory_order_relaxed);
  } else {
    g_norm_pdl_state.store(enabled ? 1 : 0, std::memory_order_relaxed);
  }
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
  cudaError_t err = cudaSuccess;
  if (kind == 0) {
    if (rows <= 16) {
      err = launch_norm_kernel(
          dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
          post_attn_norm_decode<true, false, false>,
          X, Rs, Wp, Wq, static_cast<const float*>(nullptr),
          static_cast<float*>(sumsq_out), static_cast<__nv_bfloat16*>(out_bf16),
          static_cast<uint8_t*>(nullptr), static_cast<float*>(nullptr), 0);
    } else {
      post_attn_norm<true, false><<<rows, kThreads, 0, stream>>>(X, Rs, Wp, Wq, nullptr,
          static_cast<float*>(sumsq_out), static_cast<__nv_bfloat16*>(out_bf16), nullptr);
      err = cudaGetLastError();
    }
  } else if (kind == 1) {
    if (rows <= 16) {
      err = launch_norm_kernel(
          dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
          post_attn_norm_decode<false, true, false>,
          X, Rs, Wp, Wq, static_cast<const float*>(scale),
          static_cast<float*>(sumsq_out), static_cast<__nv_bfloat16*>(nullptr),
          static_cast<uint8_t*>(out_fp8), static_cast<float*>(nullptr), 0);
    } else {
      post_attn_norm<false, true><<<rows, kThreads, 0, stream>>>(X, Rs, Wp, Wq, static_cast<const float*>(scale),
          static_cast<float*>(sumsq_out), nullptr, static_cast<uint8_t*>(out_fp8));
      err = cudaGetLastError();
    }
  } else {
    return -1;
  }
  return static_cast<int>(err);
}

int muse_norm_glue_post_attn_h1(int kind, int rows, const void* x, const void* res, const void* w_post,
                                const void* w_pre, const void* scale, void* sumsq_out, void* out_bf16,
                                void* out_fp8, void* h1_out, int discard_mask, cudaStream_t stream) {
  if (rows <= 0 || rows > 16 || h1_out == nullptr) return -1;
  const auto* X = static_cast<const __nv_bfloat16*>(x);
  const auto* Rs = static_cast<const __nv_bfloat16*>(res);
  const auto* Wp = static_cast<const __nv_bfloat16*>(w_post);
  const auto* Wq = static_cast<const __nv_bfloat16*>(w_pre);
  cudaError_t err = cudaSuccess;
  if (kind == 0) {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        post_attn_norm_decode<true, false, true>,
        X, Rs, Wp, Wq, static_cast<const float*>(nullptr),
        static_cast<float*>(sumsq_out), static_cast<__nv_bfloat16*>(out_bf16),
        static_cast<uint8_t*>(nullptr), static_cast<float*>(h1_out), discard_mask);
  } else if (kind == 1) {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        post_attn_norm_decode<false, true, true>,
        X, Rs, Wp, Wq, static_cast<const float*>(scale),
        static_cast<float*>(sumsq_out), static_cast<__nv_bfloat16*>(nullptr),
        static_cast<uint8_t*>(out_fp8), static_cast<float*>(h1_out), discard_mask);
  } else {
    return -1;
  }
  return static_cast<int>(err);
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
  cudaError_t err = cudaSuccess;
  if (kind == 0) {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        post_ff_norm<false, false, true, false>,
        X, D, Rs, S, Wp, Wf, Wi, static_cast<const float*>(nullptr),
        static_cast<__nv_bfloat16*>(out_bf16), static_cast<uint8_t*>(nullptr),
        static_cast<const float*>(nullptr), static_cast<uint8_t*>(nullptr),
        static_cast<const float*>(nullptr));
  } else if (kind == 1) {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        post_ff_norm<true, false, true, false>,
        X, D, Rs, S, Wp, Wf, Wi, static_cast<const float*>(scale),
        static_cast<__nv_bfloat16*>(out_bf16), static_cast<uint8_t*>(out_fp8),
        static_cast<const float*>(nullptr), static_cast<uint8_t*>(nullptr),
        static_cast<const float*>(nullptr));
  } else {
    return -1;
  }
  return static_cast<int>(err);
}

int muse_norm_glue_post_ff_h1(int kind, int rows, void* x, const void* d, const void* h1_in,
                              const void* w_ff, const void* w_in, const void* scale,
                              void* out_bf16, void* out_fp8, cudaStream_t stream) {
  if (rows <= 0 || h1_in == nullptr) return -1;
  auto* X = static_cast<__nv_bfloat16*>(x);
  const auto* D = static_cast<const __nv_bfloat16*>(d);
  const auto* H1 = static_cast<const float*>(h1_in);
  const auto* Wf = static_cast<const __nv_bfloat16*>(w_ff);
  const auto* Wi = static_cast<const __nv_bfloat16*>(w_in);
  cudaError_t err = cudaSuccess;
  if (kind == 0) {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        post_ff_norm<false, false, true, true>,
        X, D, static_cast<const __nv_bfloat16*>(nullptr), static_cast<const float*>(nullptr),
        static_cast<const __nv_bfloat16*>(nullptr), Wf, Wi, static_cast<const float*>(nullptr),
        static_cast<__nv_bfloat16*>(out_bf16), static_cast<uint8_t*>(nullptr),
        static_cast<const float*>(nullptr), static_cast<uint8_t*>(nullptr), H1);
  } else if (kind == 1) {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        post_ff_norm<true, false, true, true>,
        X, D, static_cast<const __nv_bfloat16*>(nullptr), static_cast<const float*>(nullptr),
        static_cast<const __nv_bfloat16*>(nullptr), Wf, Wi, static_cast<const float*>(scale),
        static_cast<__nv_bfloat16*>(out_bf16), static_cast<uint8_t*>(out_fp8),
        static_cast<const float*>(nullptr), static_cast<uint8_t*>(nullptr), H1);
  } else {
    return -1;
  }
  return static_cast<int>(err);
}

int muse_norm_glue_post_ff_final(int rows, const void* d, const void* res, const void* x, const void* sumsq_x,
                                 const void* w_post, const void* w_ff, const void* w_in,
                                 void* out_h2, void* out_bf16, int write_h2, cudaStream_t stream) {
  if (rows <= 0) return 0;
  cudaError_t err = cudaSuccess;
  if (write_h2) {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        post_ff_final_norm<true, false>,
        static_cast<const __nv_bfloat16*>(d), static_cast<const __nv_bfloat16*>(res),
        static_cast<const __nv_bfloat16*>(x), static_cast<const float*>(sumsq_x),
        static_cast<const __nv_bfloat16*>(w_post), static_cast<const __nv_bfloat16*>(w_ff),
        static_cast<const __nv_bfloat16*>(w_in),
        static_cast<float*>(out_h2), static_cast<__nv_bfloat16*>(out_bf16),
        static_cast<const float*>(nullptr));
  } else {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        post_ff_final_norm<false, false>,
        static_cast<const __nv_bfloat16*>(d), static_cast<const __nv_bfloat16*>(res),
        static_cast<const __nv_bfloat16*>(x), static_cast<const float*>(sumsq_x),
        static_cast<const __nv_bfloat16*>(w_post), static_cast<const __nv_bfloat16*>(w_ff),
        static_cast<const __nv_bfloat16*>(w_in),
        static_cast<float*>(out_h2), static_cast<__nv_bfloat16*>(out_bf16),
        static_cast<const float*>(nullptr));
  }
  return static_cast<int>(err);
}

int muse_norm_glue_post_ff_final_h1(int rows, const void* d, const void* h1_in,
                                    const void* w_ff, const void* w_in,
                                    void* out_h2, void* out_bf16, int write_h2, cudaStream_t stream) {
  if (rows <= 0 || h1_in == nullptr) return -1;
  cudaError_t err = cudaSuccess;
  if (write_h2) {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        post_ff_final_norm<true, true>,
        static_cast<const __nv_bfloat16*>(d), static_cast<const __nv_bfloat16*>(nullptr),
        static_cast<const __nv_bfloat16*>(nullptr), static_cast<const float*>(nullptr),
        static_cast<const __nv_bfloat16*>(nullptr), static_cast<const __nv_bfloat16*>(w_ff),
        static_cast<const __nv_bfloat16*>(w_in),
        static_cast<float*>(out_h2), static_cast<__nv_bfloat16*>(out_bf16),
        static_cast<const float*>(h1_in));
  } else {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        post_ff_final_norm<false, true>,
        static_cast<const __nv_bfloat16*>(d), static_cast<const __nv_bfloat16*>(nullptr),
        static_cast<const __nv_bfloat16*>(nullptr), static_cast<const float*>(nullptr),
        static_cast<const __nv_bfloat16*>(nullptr), static_cast<const __nv_bfloat16*>(w_ff),
        static_cast<const __nv_bfloat16*>(w_in),
        static_cast<float*>(out_h2), static_cast<__nv_bfloat16*>(out_bf16),
        static_cast<const float*>(h1_in));
  }
  return static_cast<int>(err);
}

int muse_norm_glue_input_fp8(int warps, int rows, const void* x, const void* w_in, const void* scale,
                             void* out_bf16, void* out_fp8, cudaStream_t stream) {
  if (rows <= 0) return 0;
  const auto* X = static_cast<const __nv_bfloat16*>(x);
  const auto* Wi = static_cast<const __nv_bfloat16*>(w_in);
  const auto* Sc = static_cast<const float*>(scale);
  auto* Ob = static_cast<__nv_bfloat16*>(out_bf16);
  auto* Of = static_cast<uint8_t*>(out_fp8);
  cudaError_t err = cudaSuccess;
  if (warps == 16) {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(512), stream, rows == 1,
        input_norm_fp8_w16<false, true>,
        X, Wi, Sc, Ob, Of, static_cast<const float*>(nullptr), static_cast<uint8_t*>(nullptr));
  } else if (warps == 8) {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        input_norm_fp8_w8<false, true>,
        X, Wi, Sc, Ob, Of, static_cast<const float*>(nullptr), static_cast<uint8_t*>(nullptr));
  } else {
    return -1;
  }
  return static_cast<int>(err);
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
  cudaError_t err = cudaSuccess;
  if (warps == 16) {
    if (write_bf16) {
      err = launch_norm_kernel(
          dim3(static_cast<unsigned>(rows)), dim3(512), stream, rows == 1,
          input_norm_fp8_w16<true, true>, X, Wi, Sc, Ob, Of, Gs, Og);
    } else {
      err = launch_norm_kernel(
          dim3(static_cast<unsigned>(rows)), dim3(512), stream, rows == 1,
          input_norm_fp8_w16<true, false>, X, Wi, Sc, Ob, Of, Gs, Og);
    }
  } else if (warps == 8) {
    if (write_bf16) {
      err = launch_norm_kernel(
          dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
          input_norm_fp8_w8<true, true>, X, Wi, Sc, Ob, Of, Gs, Og);
    } else {
      err = launch_norm_kernel(
          dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
          input_norm_fp8_w8<true, false>, X, Wi, Sc, Ob, Of, Gs, Og);
    }
  } else {
    return -1;
  }
  return static_cast<int>(err);
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
  cudaError_t err = cudaSuccess;
  if (write_bf16) {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        post_ff_norm<true, true, true, false>,
        X, D, Rs, S, Wp, Wf, Wi, Sc, Ob, Of, Gs, Og, static_cast<const float*>(nullptr));
  } else {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        post_ff_norm<true, true, false, false>,
        X, D, Rs, S, Wp, Wf, Wi, Sc, Ob, Of, Gs, Og, static_cast<const float*>(nullptr));
  }
  return static_cast<int>(err);
}

int muse_norm_glue_post_ff_gate_h1(int rows, void* x, const void* d, const void* h1_in,
                                   const void* w_ff, const void* w_in, const void* scale,
                                   void* out_bf16, void* out_fp8, const void* gate_scale, void* out_gate,
                                   int write_bf16, cudaStream_t stream) {
  if (rows <= 0 || h1_in == nullptr) return -1;
  auto* X = static_cast<__nv_bfloat16*>(x);
  const auto* D = static_cast<const __nv_bfloat16*>(d);
  const auto* H1 = static_cast<const float*>(h1_in);
  const auto* Wf = static_cast<const __nv_bfloat16*>(w_ff);
  const auto* Wi = static_cast<const __nv_bfloat16*>(w_in);
  const auto* Sc = static_cast<const float*>(scale);
  const auto* Gs = static_cast<const float*>(gate_scale);
  auto* Ob = static_cast<__nv_bfloat16*>(out_bf16);
  auto* Of = static_cast<uint8_t*>(out_fp8);
  auto* Og = static_cast<uint8_t*>(out_gate);
  cudaError_t err = cudaSuccess;
  if (write_bf16) {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        post_ff_norm<true, true, true, true>,
        X, D, static_cast<const __nv_bfloat16*>(nullptr), static_cast<const float*>(nullptr),
        static_cast<const __nv_bfloat16*>(nullptr), Wf, Wi, Sc, Ob, Of, Gs, Og, H1);
  } else {
    err = launch_norm_kernel(
        dim3(static_cast<unsigned>(rows)), dim3(kThreads), stream, rows == 1,
        post_ff_norm<true, true, false, true>,
        X, D, static_cast<const __nv_bfloat16*>(nullptr), static_cast<const float*>(nullptr),
        static_cast<const __nv_bfloat16*>(nullptr), Wf, Wi, Sc, Ob, Of, Gs, Og, H1);
  }
  return static_cast<int>(err);
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
  cudaError_t err = launch_norm_kernel(
      dim3((vec8 + 255) / 256), dim3(256), stream, vec8 == 512,
      mul_sigmoid_bf16_kernel,
      static_cast<const uint4*>(gate), static_cast<const uint4*>(attn),
      static_cast<uint4*>(out), vec8);
  return static_cast<int>(err);
}

int muse_norm_glue_mul_sigmoid_fp8(const void* gate, const void* attn, const void* scale, void* out, int n,
                                   cudaStream_t stream) {
  if (n <= 0 || (n & 7) != 0) return -1;
  const int vec8 = n >> 3;
  mul_sigmoid_fp8_kernel<<<(vec8 + 255) / 256, 256, 0, stream>>>(
      static_cast<const uint4*>(gate), static_cast<const uint4*>(attn),
      static_cast<const float*>(scale), static_cast<uint2*>(out), vec8);
  return static_cast<int>(cudaGetLastError());
}

int muse_norm_glue_mul_silu_slice_bf16(const void* gu, void* out, int rows, cudaStream_t stream) {
  if (rows <= 0) return -1;
  const dim3 grid((kInterVec8 + 127) / 128, static_cast<unsigned>(rows));
  cudaError_t err = launch_norm_kernel(
      grid, dim3(128), stream, rows == 1,
      mul_silu_slice_bf16_kernel,
      static_cast<const uint4*>(gu), static_cast<uint4*>(out));
  return static_cast<int>(err);
}

}  // extern "C"
