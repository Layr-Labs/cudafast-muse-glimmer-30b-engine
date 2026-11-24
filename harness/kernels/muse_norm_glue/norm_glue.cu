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
template <bool kBf16Out, bool kFp8Out>
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
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      ux[c] = ld8(x, c);
      ur[c] = ld8(res, c);
      uw[c] = ld8(w_post, c);
      up[c] = ld8(w_pre, c);
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
template <bool kFp8Out, bool kGateOut = false, bool kWriteBf16 = true>
__global__ void __launch_bounds__(kThreads) post_ff_norm(
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
  uint2 ux[kChunks], ud[kChunks], ur[kChunks], uw[kChunks], uf[kChunks], ui[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      ud[c] = ld8(d, c);
      ux[c] = ld8(x, c);
      ur[c] = ld8(res, c);
      uw[c] = ld8(w_post, c);
      uf[c] = ld8(w_ff, c);
      ui[c] = ld8(w_in, c);
    }
  }
  const float s_x = sumsq_x[row];
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
  const float a = inv_rms(s_x, 1e-08f);
  float acc2[4] = {0.f, 0.f, 0.f, 0.f};
  uint2 hb[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid(c)) {
      Row4 xv = unpack(ux[c]), dv = unpack(ud[c]), rv = unpack(ur[c]), wv = unpack(uw[c]), fv = unpack(uf[c]);
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
        *reinterpret_cast<uint2*>(out_bf16_all + off) = o;
      }
      if constexpr (kFp8Out) {
        uint32_t q = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[0])), clamp448(__fmul_rn(inv_scale, y[1])))) |
                     (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[2])), clamp448(__fmul_rn(inv_scale, y[3])))) << 16);
        *reinterpret_cast<uint32_t*>(out_fp8_all + off) = q;
      }
      if constexpr (kGateOut) {
        const float g0 = bf_lo(o.x), g1 = bf_hi(o.x), g2 = bf_lo(o.y), g3 = bf_hi(o.y);
        uint32_t gq = static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(gate_inv, g0)), clamp448(__fmul_rn(gate_inv, g1)))) |
                      (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(gate_inv, g2)), clamp448(__fmul_rn(gate_inv, g3)))) << 16);
        *reinterpret_cast<uint32_t*>(out_gate_all + off) = gq;
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

__global__ void __launch_bounds__(256) mul_sigmoid_bf16_kernel(
    const uint4* __restrict__ gate, const uint4* __restrict__ attn, uint4* __restrict__ out, int vec8) {
  const int idx = blockIdx.x * 256 + threadIdx.x;
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

}  // namespace

extern "C" {

// kind: 0 = bf16 out, 1 = fp8 out (no bf16 out), as the compiled ..._rsqrt_1 kernels.
int muse_norm_glue_post_attn(int kind, int rows, const void* x, const void* res, const void* w_post,
                             const void* w_pre, const void* scale, void* sumsq_out, void* out_bf16,
                             void* out_fp8, cudaStream_t stream) {
  if (rows <= 0) return 0;
  const auto* X = static_cast<const __nv_bfloat16*>(x);
  const auto* Rs = static_cast<const __nv_bfloat16*>(res);
  const auto* Wp = static_cast<const __nv_bfloat16*>(w_post);
  const auto* Wq = static_cast<const __nv_bfloat16*>(w_pre);
  if (kind == 0) {
    if (rows <= 16) {
      post_attn_norm_decode<true, false><<<rows, kThreads, 0, stream>>>(X, Rs, Wp, Wq, nullptr,
          static_cast<float*>(sumsq_out), static_cast<__nv_bfloat16*>(out_bf16), nullptr);
    } else {
      post_attn_norm<true, false><<<rows, kThreads, 0, stream>>>(X, Rs, Wp, Wq, nullptr,
          static_cast<float*>(sumsq_out), static_cast<__nv_bfloat16*>(out_bf16), nullptr);
    }
  } else if (kind == 1) {
    if (rows <= 16) {
      post_attn_norm_decode<false, true><<<rows, kThreads, 0, stream>>>(X, Rs, Wp, Wq, static_cast<const float*>(scale),
          static_cast<float*>(sumsq_out), nullptr, static_cast<uint8_t*>(out_fp8));
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
  if (kind == 0) {
    post_ff_norm<false><<<rows, kThreads, 0, stream>>>(X, D, Rs, S, Wp, Wf, Wi, nullptr,
        static_cast<__nv_bfloat16*>(out_bf16), nullptr);
  } else if (kind == 1) {
    post_ff_norm<true><<<rows, kThreads, 0, stream>>>(X, D, Rs, S, Wp, Wf, Wi, static_cast<const float*>(scale),
        static_cast<__nv_bfloat16*>(out_bf16), static_cast<uint8_t*>(out_fp8));
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
  if (write_bf16) {
    post_ff_norm<true, true, true><<<rows, kThreads, 0, stream>>>(X, D, Rs, S, Wp, Wf, Wi, Sc, Ob, Of, Gs, Og);
  } else {
    post_ff_norm<true, true, false><<<rows, kThreads, 0, stream>>>(X, D, Rs, S, Wp, Wf, Wi, Sc, Ob, Of, Gs, Og);
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
  mul_sigmoid_bf16_kernel<<<(vec8 + 255) / 256, 256, 0, stream>>>(
      static_cast<const uint4*>(gate), static_cast<const uint4*>(attn),
      static_cast<uint4*>(out), vec8);
  return static_cast<int>(cudaGetLastError());
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

}  // extern "C"
