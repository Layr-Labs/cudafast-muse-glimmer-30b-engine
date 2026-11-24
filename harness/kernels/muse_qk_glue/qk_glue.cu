// SPDX-License-Identifier: Apache-2.0
// One-pass fused Q/K RMSNorm + query pre-scale + NEOX RoPE for MuseGlimmerAttention
// (head_dim = 128, 32 Q heads + 2 K heads, BF16 I/O, FP32 internal arithmetic).
//
// Evaluates the exact same FP32 sum-of-squares reduction order (4 elements per lane,
// xor-shuffle 16/8/4/2/1), div.full.f32(/128), rsqrt.approx.ftz.f32, query pre-scale,
// and NEOX RoPE FMA chain as the compiled Inductor kernels in a single global pass
// for both prompt prefill (M > 1) and single-token decode (M = 1).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace muse_qk_glue {

template <bool UseRope, bool CacheFP32>
__global__ void qk_norm_rope_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const int64_t* __restrict__ positions,
    const void* __restrict__ cos_sin_cache,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int num_tokens,
    int cache_rows,
    float eps,
    float qscale) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int total_warps = num_tokens * 34;
  if (warp_id >= total_warps) return;
  const int lane = threadIdx.x & 31;
  const int t = warp_id / 34;
  const int h = warp_id - t * 34;

  const uint2 raw = *reinterpret_cast<const uint2*>(
      qkv + static_cast<size_t>(t) * 4608 + h * 128 + lane * 4);
  const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&raw);
  const float x0 = __bfloat162float(vb[0]);
  const float x1 = __bfloat162float(vb[1]);
  const float x2 = __bfloat162float(vb[2]);
  const float x3 = __bfloat162float(vb[3]);

  // Exact Triton reduction order: ((x0^2 + x1^2) + x2^2) + x3^2, then xor-shuffle 16,8,4,2,1.
  float ss = __fmaf_rn(x0, x0, 0.0f);
  ss = __fmaf_rn(x1, x1, ss);
  ss = __fmaf_rn(x2, x2, ss);
  ss = __fmaf_rn(x3, x3, ss);
  ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 16));
  ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 8));
  ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 4));
  ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 2));
  ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 1));

  float mean_v;
  asm("div.full.f32 %0, %1, 0f43000000;" : "=f"(mean_v) : "f"(ss));
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(mean_v + eps));

  float y0 = __fmul_rn(x0, r);
  float y1 = __fmul_rn(x1, r);
  float y2 = __fmul_rn(x2, r);
  float y3 = __fmul_rn(x3, r);
  if (h < 32) {
    y0 = __fmul_rn(y0, qscale);
    y1 = __fmul_rn(y1, qscale);
    y2 = __fmul_rn(y2, qscale);
    y3 = __fmul_rn(y3, qscale);
  }

  float o0 = y0, o1 = y1, o2 = y2, o3 = y3;
  if constexpr (UseRope) {
    // Lane l holds elements 4*l..4*l+3, so lane (l ^ 16) holds the exact +/-64 NEOX partner.
    const float p0 = __shfl_xor_sync(0xffffffffu, y0, 16);
    const float p1 = __shfl_xor_sync(0xffffffffu, y1, 16);
    const float p2 = __shfl_xor_sync(0xffffffffu, y2, 16);
    const float p3 = __shfl_xor_sync(0xffffffffu, y3, 16);
    int64_t pos = positions[t];
    if (pos < 0) pos += cache_rows;
    const int d_lo = (lane & 15) * 4;
    float c0, c1, c2, c3, s0, s1, s2, s3;
    if constexpr (CacheFP32) {
      const float* row = reinterpret_cast<const float*>(cos_sin_cache) + pos * 128;
      const float4 cv = *reinterpret_cast<const float4*>(row + d_lo);
      const float4 sv = *reinterpret_cast<const float4*>(row + 64 + d_lo);
      c0 = cv.x; c1 = cv.y; c2 = cv.z; c3 = cv.w;
      s0 = sv.x; s1 = sv.y; s2 = sv.z; s3 = sv.w;
    } else {
      const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache) + pos * 128;
      const uint2 cv = *reinterpret_cast<const uint2*>(row + d_lo);
      const uint2 sv = *reinterpret_cast<const uint2*>(row + 64 + d_lo);
      const __nv_bfloat16* cb = reinterpret_cast<const __nv_bfloat16*>(&cv);
      const __nv_bfloat16* sb = reinterpret_cast<const __nv_bfloat16*>(&sv);
      c0 = __bfloat162float(cb[0]); c1 = __bfloat162float(cb[1]);
      c2 = __bfloat162float(cb[2]); c3 = __bfloat162float(cb[3]);
      s0 = __bfloat162float(sb[0]); s1 = __bfloat162float(sb[1]);
      s2 = __bfloat162float(sb[2]); s3 = __bfloat162float(sb[3]);
    }
    if (lane < 16) {
      o0 = __fmaf_rn(y0, c0, -__fmul_rn(p0, s0));
      o1 = __fmaf_rn(y1, c1, -__fmul_rn(p1, s1));
      o2 = __fmaf_rn(y2, c2, -__fmul_rn(p2, s2));
      o3 = __fmaf_rn(y3, c3, -__fmul_rn(p3, s3));
    } else {
      o0 = __fmaf_rn(y0, c0, __fmul_rn(p0, s0));
      o1 = __fmaf_rn(y1, c1, __fmul_rn(p1, s1));
      o2 = __fmaf_rn(y2, c2, __fmul_rn(p2, s2));
      o3 = __fmaf_rn(y3, c3, __fmul_rn(p3, s3));
    }
  }

  uint2 out_v;
  __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&out_v);
  ob[0] = __float2bfloat16_rn(o0);
  ob[1] = __float2bfloat16_rn(o1);
  ob[2] = __float2bfloat16_rn(o2);
  ob[3] = __float2bfloat16_rn(o3);
  if (h < 32) {
    *reinterpret_cast<uint2*>(q_out + static_cast<size_t>(t) * 4096 + h * 128 + lane * 4) = out_v;
  } else {
    *reinterpret_cast<uint2*>(k_out + static_cast<size_t>(t) * 256 + (h - 32) * 128 + lane * 4) = out_v;
  }
}

template <bool UseRope, bool CacheFP32, int HPW>
__global__ void qk_norm_rope_multi_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const int64_t* __restrict__ positions,
    const void* __restrict__ cos_sin_cache,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int num_tokens,
    int cache_rows,
    float eps,
    float qscale) {
  constexpr int kGroups = (34 + HPW - 1) / HPW;
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp_id >= num_tokens * kGroups) return;
  const int lane = threadIdx.x & 31;
  const int t = warp_id / kGroups;
  const int h0 = (warp_id - t * kGroups) * HPW;
  uint2 raw[HPW];
#pragma unroll
  for (int j = 0; j < HPW; ++j) {
    if (h0 + j < 34)
      raw[j] = *reinterpret_cast<const uint2*>(qkv + static_cast<size_t>(t) * 4608 + (h0 + j) * 128 + lane * 4);
  }
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  if constexpr (UseRope) {
    int64_t pos = positions[t];
    if (pos < 0) pos += cache_rows;
    const int d_lo = (lane & 15) * 4;
    if constexpr (CacheFP32) {
      const float* row = reinterpret_cast<const float*>(cos_sin_cache) + pos * 128;
      const float4 cv = *reinterpret_cast<const float4*>(row + d_lo);
      const float4 sv = *reinterpret_cast<const float4*>(row + 64 + d_lo);
      c0 = cv.x; c1 = cv.y; c2 = cv.z; c3 = cv.w;
      s0 = sv.x; s1 = sv.y; s2 = sv.z; s3 = sv.w;
    } else {
      const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache) + pos * 128;
      const uint2 cv = *reinterpret_cast<const uint2*>(row + d_lo);
      const uint2 sv = *reinterpret_cast<const uint2*>(row + 64 + d_lo);
      const __nv_bfloat16* cb = reinterpret_cast<const __nv_bfloat16*>(&cv);
      const __nv_bfloat16* sb = reinterpret_cast<const __nv_bfloat16*>(&sv);
      c0 = __bfloat162float(cb[0]); c1 = __bfloat162float(cb[1]);
      c2 = __bfloat162float(cb[2]); c3 = __bfloat162float(cb[3]);
      s0 = __bfloat162float(sb[0]); s1 = __bfloat162float(sb[1]);
      s2 = __bfloat162float(sb[2]); s3 = __bfloat162float(sb[3]);
    }
  }
#pragma unroll
  for (int j = 0; j < HPW; ++j) {
    const int h = h0 + j;
    if (h >= 34) break;
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&raw[j]);
    const float x0 = __bfloat162float(vb[0]);
    const float x1 = __bfloat162float(vb[1]);
    const float x2 = __bfloat162float(vb[2]);
    const float x3 = __bfloat162float(vb[3]);
    float ss = __fmaf_rn(x0, x0, 0.0f);
    ss = __fmaf_rn(x1, x1, ss);
    ss = __fmaf_rn(x2, x2, ss);
    ss = __fmaf_rn(x3, x3, ss);
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 16));
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 8));
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 4));
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 2));
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 1));
    float mean_v;
    asm("div.full.f32 %0, %1, 0f43000000;" : "=f"(mean_v) : "f"(ss));
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(mean_v + eps));
    float y0 = __fmul_rn(x0, r);
    float y1 = __fmul_rn(x1, r);
    float y2 = __fmul_rn(x2, r);
    float y3 = __fmul_rn(x3, r);
    if (h < 32) {
      y0 = __fmul_rn(y0, qscale);
      y1 = __fmul_rn(y1, qscale);
      y2 = __fmul_rn(y2, qscale);
      y3 = __fmul_rn(y3, qscale);
    }
    float o0 = y0, o1 = y1, o2 = y2, o3 = y3;
    if constexpr (UseRope) {
      const float p0 = __shfl_xor_sync(0xffffffffu, y0, 16);
      const float p1 = __shfl_xor_sync(0xffffffffu, y1, 16);
      const float p2 = __shfl_xor_sync(0xffffffffu, y2, 16);
      const float p3 = __shfl_xor_sync(0xffffffffu, y3, 16);
      if (lane < 16) {
        o0 = __fmaf_rn(y0, c0, -__fmul_rn(p0, s0));
        o1 = __fmaf_rn(y1, c1, -__fmul_rn(p1, s1));
        o2 = __fmaf_rn(y2, c2, -__fmul_rn(p2, s2));
        o3 = __fmaf_rn(y3, c3, -__fmul_rn(p3, s3));
      } else {
        o0 = __fmaf_rn(y0, c0, __fmul_rn(p0, s0));
        o1 = __fmaf_rn(y1, c1, __fmul_rn(p1, s1));
        o2 = __fmaf_rn(y2, c2, __fmul_rn(p2, s2));
        o3 = __fmaf_rn(y3, c3, __fmul_rn(p3, s3));
      }
    }
    uint2 out_v;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&out_v);
    ob[0] = __float2bfloat16_rn(o0);
    ob[1] = __float2bfloat16_rn(o1);
    ob[2] = __float2bfloat16_rn(o2);
    ob[3] = __float2bfloat16_rn(o3);
    if (h < 32) {
      *reinterpret_cast<uint2*>(q_out + static_cast<size_t>(t) * 4096 + h * 128 + lane * 4) = out_v;
    } else {
      *reinterpret_cast<uint2*>(k_out + static_cast<size_t>(t) * 256 + (h - 32) * 128 + lane * 4) = out_v;
    }
  }
}

template <bool UseRope, bool CacheFP32>
__global__ __launch_bounds__(288) void qk_norm_rope_kvcache_m1_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const int64_t* __restrict__ positions,
    const void* __restrict__ cos_sin_cache,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping,
    int cache_rows,
    int block_size,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    float eps,
    float qscale) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp_id >= 17) {
    if (warp_id == 17) {
      const int64_t slot = slot_mapping[0];
      if (slot >= 0) {
        const uint2 v0 = *reinterpret_cast<const uint2*>(qkv + 4352 + lane * 4);
        const uint2 v1 = *reinterpret_cast<const uint2*>(qkv + 4480 + lane * 4);
        const int64_t blk = (block_size == 16) ? (slot >> 4) : (slot / block_size);
        const int64_t off = (block_size == 16) ? (slot & 15) : (slot - blk * block_size);
        __nv_bfloat16* dst = kv_cache + blk * stride_block + off * stride_slot + 128 + lane * 4;
        *reinterpret_cast<uint2*>(dst) = v0;
        *reinterpret_cast<uint2*>(dst + stride_kv_head) = v1;
      }
    }
    return;
  }

  const int h0 = warp_id * 2;
  const int64_t k_slot = (warp_id == 16) ? slot_mapping[0] : -1;
  uint2 raw[2];
  raw[0] = *reinterpret_cast<const uint2*>(qkv + h0 * 128 + lane * 4);
  raw[1] = *reinterpret_cast<const uint2*>(qkv + (h0 + 1) * 128 + lane * 4);

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  if constexpr (UseRope) {
    int64_t pos = positions[0];
    if (pos < 0) pos += cache_rows;
    const int d_lo = (lane & 15) * 4;
    if constexpr (CacheFP32) {
      const float* row = reinterpret_cast<const float*>(cos_sin_cache) + pos * 128;
      const float4 cv = *reinterpret_cast<const float4*>(row + d_lo);
      const float4 sv = *reinterpret_cast<const float4*>(row + 64 + d_lo);
      c0 = cv.x; c1 = cv.y; c2 = cv.z; c3 = cv.w;
      s0 = sv.x; s1 = sv.y; s2 = sv.z; s3 = sv.w;
    } else {
      const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache) + pos * 128;
      const uint2 cv = *reinterpret_cast<const uint2*>(row + d_lo);
      const uint2 sv = *reinterpret_cast<const uint2*>(row + 64 + d_lo);
      const __nv_bfloat16* cb = reinterpret_cast<const __nv_bfloat16*>(&cv);
      const __nv_bfloat16* sb = reinterpret_cast<const __nv_bfloat16*>(&sv);
      c0 = __bfloat162float(cb[0]); c1 = __bfloat162float(cb[1]);
      c2 = __bfloat162float(cb[2]); c3 = __bfloat162float(cb[3]);
      s0 = __bfloat162float(sb[0]); s1 = __bfloat162float(sb[1]);
      s2 = __bfloat162float(sb[2]); s3 = __bfloat162float(sb[3]);
    }
  }

  __nv_bfloat16* k_cache_base = nullptr;
  if (warp_id == 16 && k_slot >= 0) {
    const int64_t blk = (block_size == 16) ? (k_slot >> 4) : (k_slot / block_size);
    const int64_t off = (block_size == 16) ? (k_slot & 15) : (k_slot - blk * block_size);
    k_cache_base = kv_cache + blk * stride_block + off * stride_slot + lane * 4;
  }

#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int h = h0 + j;
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&raw[j]);
    const float x0 = __bfloat162float(vb[0]);
    const float x1 = __bfloat162float(vb[1]);
    const float x2 = __bfloat162float(vb[2]);
    const float x3 = __bfloat162float(vb[3]);
    float ss = __fmaf_rn(x0, x0, 0.0f);
    ss = __fmaf_rn(x1, x1, ss);
    ss = __fmaf_rn(x2, x2, ss);
    ss = __fmaf_rn(x3, x3, ss);
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 16));
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 8));
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 4));
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 2));
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 1));
    float mean_v;
    asm("div.full.f32 %0, %1, 0f43000000;" : "=f"(mean_v) : "f"(ss));
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(mean_v + eps));
    float y0 = __fmul_rn(x0, r);
    float y1 = __fmul_rn(x1, r);
    float y2 = __fmul_rn(x2, r);
    float y3 = __fmul_rn(x3, r);
    if (h < 32) {
      y0 = __fmul_rn(y0, qscale);
      y1 = __fmul_rn(y1, qscale);
      y2 = __fmul_rn(y2, qscale);
      y3 = __fmul_rn(y3, qscale);
    }
    float o0 = y0, o1 = y1, o2 = y2, o3 = y3;
    if constexpr (UseRope) {
      const float p0 = __shfl_xor_sync(0xffffffffu, y0, 16);
      const float p1 = __shfl_xor_sync(0xffffffffu, y1, 16);
      const float p2 = __shfl_xor_sync(0xffffffffu, y2, 16);
      const float p3 = __shfl_xor_sync(0xffffffffu, y3, 16);
      if (lane < 16) {
        o0 = __fmaf_rn(y0, c0, -__fmul_rn(p0, s0));
        o1 = __fmaf_rn(y1, c1, -__fmul_rn(p1, s1));
        o2 = __fmaf_rn(y2, c2, -__fmul_rn(p2, s2));
        o3 = __fmaf_rn(y3, c3, -__fmul_rn(p3, s3));
      } else {
        o0 = __fmaf_rn(y0, c0, __fmul_rn(p0, s0));
        o1 = __fmaf_rn(y1, c1, __fmul_rn(p1, s1));
        o2 = __fmaf_rn(y2, c2, __fmul_rn(p2, s2));
        o3 = __fmaf_rn(y3, c3, __fmul_rn(p3, s3));
      }
    }
    uint2 out_v;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&out_v);
    ob[0] = __float2bfloat16_rn(o0);
    ob[1] = __float2bfloat16_rn(o1);
    ob[2] = __float2bfloat16_rn(o2);
    ob[3] = __float2bfloat16_rn(o3);
    if (h < 32) {
      *reinterpret_cast<uint2*>(q_out + h * 128 + lane * 4) = out_v;
    } else {
      const int hk = h - 32;
      if (k_out != nullptr) {
        *reinterpret_cast<uint2*>(k_out + hk * 128 + lane * 4) = out_v;
      }
      if (k_cache_base != nullptr) {
        *reinterpret_cast<uint2*>(k_cache_base + hk * stride_kv_head) = out_v;
      }
    }
  }
}

template <int HPW>
int launch_multi(const void* qkv, const void* positions, const void* cos_sin_cache, void* q_out, void* k_out,
                 int num_tokens, int cache_rows, float eps, float qscale, int use_rope, int cache_fp32,
                 cudaStream_t stream) {
  constexpr int kGroups = (34 + HPW - 1) / HPW;
  const int threads = (HPW == 2 && num_tokens == 1) ? 288 : 128;
  const int wpb = threads >> 5;
  const int blocks = (num_tokens * kGroups + wpb - 1) / wpb;
  const auto* q = static_cast<const __nv_bfloat16*>(qkv);
  const auto* pos = static_cast<const int64_t*>(positions);
  auto* qo = static_cast<__nv_bfloat16*>(q_out);
  auto* ko = static_cast<__nv_bfloat16*>(k_out);
  if (use_rope) {
    if (cache_fp32)
      qk_norm_rope_multi_kernel<true, true, HPW><<<blocks, threads, 0, stream>>>(
          q, pos, cos_sin_cache, qo, ko, num_tokens, cache_rows, eps, qscale);
    else
      qk_norm_rope_multi_kernel<true, false, HPW><<<blocks, threads, 0, stream>>>(
          q, pos, cos_sin_cache, qo, ko, num_tokens, cache_rows, eps, qscale);
  } else {
    qk_norm_rope_multi_kernel<false, true, HPW><<<blocks, threads, 0, stream>>>(
        q, pos, cos_sin_cache, qo, ko, num_tokens, cache_rows, eps, qscale);
  }
  return static_cast<int>(cudaGetLastError());
}

template <bool UseRope, bool CacheFP32, int HPW>
__global__ void qk_norm_rope_kvcache_prefill_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const int64_t* __restrict__ positions,
    const void* __restrict__ cos_sin_cache,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping,
    int num_tokens,
    int q_first_row,
    int cache_rows,
    int block_size,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    float eps,
    float qscale) {
  constexpr int kGroups = (34 + HPW - 1) / HPW;
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp_id >= num_tokens * kGroups) return;
  const int lane = threadIdx.x & 31;
  const int t = warp_id / kGroups;
  const int h0 = (warp_id - t * kGroups) * HPW;
  if (t < q_first_row && h0 + HPW <= 32) return;

  int64_t slot = -1;
  if (h0 + HPW > 32) {
    slot = slot_mapping[t];
    if (slot < 0 && h0 >= 32) return;
  }

  uint2 raw[HPW];
#pragma unroll
  for (int j = 0; j < HPW; ++j) {
    const int h = h0 + j;
    if (h < 34 && (t >= q_first_row || h >= 32)) {
      raw[j] = *reinterpret_cast<const uint2*>(qkv + static_cast<size_t>(t) * 4608 + h * 128 + lane * 4);
    }
  }
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  if constexpr (UseRope) {
    int64_t pos = positions[t];
    if (pos < 0) pos += cache_rows;
    const int d_lo = (lane & 15) * 4;
    if constexpr (CacheFP32) {
      const float* row = reinterpret_cast<const float*>(cos_sin_cache) + pos * 128;
      const float4 cv = *reinterpret_cast<const float4*>(row + d_lo);
      const float4 sv = *reinterpret_cast<const float4*>(row + 64 + d_lo);
      c0 = cv.x; c1 = cv.y; c2 = cv.z; c3 = cv.w;
      s0 = sv.x; s1 = sv.y; s2 = sv.z; s3 = sv.w;
    } else {
      const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache) + pos * 128;
      const uint2 cv = *reinterpret_cast<const uint2*>(row + d_lo);
      const uint2 sv = *reinterpret_cast<const uint2*>(row + 64 + d_lo);
      const __nv_bfloat16* cb = reinterpret_cast<const __nv_bfloat16*>(&cv);
      const __nv_bfloat16* sb = reinterpret_cast<const __nv_bfloat16*>(&sv);
      c0 = __bfloat162float(cb[0]); c1 = __bfloat162float(cb[1]);
      c2 = __bfloat162float(cb[2]); c3 = __bfloat162float(cb[3]);
      s0 = __bfloat162float(sb[0]); s1 = __bfloat162float(sb[1]);
      s2 = __bfloat162float(sb[2]); s3 = __bfloat162float(sb[3]);
    }
  }

  __nv_bfloat16* kv_base = nullptr;
  if (slot >= 0) {
    const int64_t blk = (block_size == 16) ? (slot >> 4) : (slot / block_size);
    const int64_t off = (block_size == 16) ? (slot & 15) : (slot - blk * block_size);
    kv_base = kv_cache + blk * stride_block + off * stride_slot + lane * 4;
  }

#pragma unroll
  for (int j = 0; j < HPW; ++j) {
    const int h = h0 + j;
    if (h >= 34) break;
    if (t < q_first_row && h < 32) continue;
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&raw[j]);
    const float x0 = __bfloat162float(vb[0]);
    const float x1 = __bfloat162float(vb[1]);
    const float x2 = __bfloat162float(vb[2]);
    const float x3 = __bfloat162float(vb[3]);
    float ss = __fmaf_rn(x0, x0, 0.0f);
    ss = __fmaf_rn(x1, x1, ss);
    ss = __fmaf_rn(x2, x2, ss);
    ss = __fmaf_rn(x3, x3, ss);
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 16));
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 8));
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 4));
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 2));
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 1));
    float mean_v;
    asm("div.full.f32 %0, %1, 0f43000000;" : "=f"(mean_v) : "f"(ss));
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(mean_v + eps));
    float y0 = __fmul_rn(x0, r);
    float y1 = __fmul_rn(x1, r);
    float y2 = __fmul_rn(x2, r);
    float y3 = __fmul_rn(x3, r);
    if (h < 32) {
      y0 = __fmul_rn(y0, qscale);
      y1 = __fmul_rn(y1, qscale);
      y2 = __fmul_rn(y2, qscale);
      y3 = __fmul_rn(y3, qscale);
    }
    float o0 = y0, o1 = y1, o2 = y2, o3 = y3;
    if constexpr (UseRope) {
      const float p0 = __shfl_xor_sync(0xffffffffu, y0, 16);
      const float p1 = __shfl_xor_sync(0xffffffffu, y1, 16);
      const float p2 = __shfl_xor_sync(0xffffffffu, y2, 16);
      const float p3 = __shfl_xor_sync(0xffffffffu, y3, 16);
      if (lane < 16) {
        o0 = __fmaf_rn(y0, c0, -__fmul_rn(p0, s0));
        o1 = __fmaf_rn(y1, c1, -__fmul_rn(p1, s1));
        o2 = __fmaf_rn(y2, c2, -__fmul_rn(p2, s2));
        o3 = __fmaf_rn(y3, c3, -__fmul_rn(p3, s3));
      } else {
        o0 = __fmaf_rn(y0, c0, __fmul_rn(p0, s0));
        o1 = __fmaf_rn(y1, c1, __fmul_rn(p1, s1));
        o2 = __fmaf_rn(y2, c2, __fmul_rn(p2, s2));
        o3 = __fmaf_rn(y3, c3, __fmul_rn(p3, s3));
      }
    }
    uint2 out_v;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&out_v);
    ob[0] = __float2bfloat16_rn(o0);
    ob[1] = __float2bfloat16_rn(o1);
    ob[2] = __float2bfloat16_rn(o2);
    ob[3] = __float2bfloat16_rn(o3);
    if (h < 32) {
      *reinterpret_cast<uint2*>(q_out + static_cast<size_t>(t) * 4096 + h * 128 + lane * 4) = out_v;
    } else if (kv_base != nullptr) {
      const int64_t hk = static_cast<int64_t>(h - 32);
      const uint2 v_raw = *reinterpret_cast<const uint2*>(
          qkv + static_cast<size_t>(t) * 4608 + 4352 + hk * 128 + lane * 4);
      __nv_bfloat16* dst = kv_base + hk * stride_kv_head;
      *reinterpret_cast<uint2*>(dst) = out_v;
      *reinterpret_cast<uint2*>(dst + 128) = v_raw;
    }
  }
}

template <int HPW>
int launch_kvcache_prefill(
    const void* qkv, const void* positions, const void* cos_sin_cache,
    void* q_out, void* kv_cache, const void* slot_mapping,
    int num_tokens, int q_first_row, int cache_rows, int block_size,
    int64_t stride_block, int64_t stride_slot, int64_t stride_kv_head,
    float eps, float qscale, int use_rope, int cache_fp32, cudaStream_t stream) {
  constexpr int kGroups = (34 + HPW - 1) / HPW;
  const int threads = 128;
  const int wpb = threads >> 5;
  const int blocks = (num_tokens * kGroups + wpb - 1) / wpb;
  const auto* q = static_cast<const __nv_bfloat16*>(qkv);
  const auto* pos = static_cast<const int64_t*>(positions);
  auto* qo = static_cast<__nv_bfloat16*>(q_out);
  auto* kv = static_cast<__nv_bfloat16*>(kv_cache);
  const auto* sm = static_cast<const int64_t*>(slot_mapping);
  if (use_rope) {
    if (cache_fp32)
      qk_norm_rope_kvcache_prefill_kernel<true, true, HPW><<<blocks, threads, 0, stream>>>(
          q, pos, cos_sin_cache, qo, kv, sm, num_tokens, q_first_row, cache_rows, block_size,
          stride_block, stride_slot, stride_kv_head, eps, qscale);
    else
      qk_norm_rope_kvcache_prefill_kernel<true, false, HPW><<<blocks, threads, 0, stream>>>(
          q, pos, cos_sin_cache, qo, kv, sm, num_tokens, q_first_row, cache_rows, block_size,
          stride_block, stride_slot, stride_kv_head, eps, qscale);
  } else {
    qk_norm_rope_kvcache_prefill_kernel<false, true, HPW><<<blocks, threads, 0, stream>>>(
        q, pos, cos_sin_cache, qo, kv, sm, num_tokens, q_first_row, cache_rows, block_size,
        stride_block, stride_slot, stride_kv_head, eps, qscale);
  }
  return static_cast<int>(cudaGetLastError());
}

}  // namespace muse_qk_glue

extern "C" int muse_qk_norm_rope_kvcache_prefill(
    const void* qkv, const void* positions, const void* cos_sin_cache,
    void* q_out, void* kv_cache, const void* slot_mapping,
    int num_tokens, int q_first_row, int cache_rows, int block_size,
    int64_t stride_block, int64_t stride_slot, int64_t stride_kv_head,
    float eps, float qscale, int use_rope, int cache_fp32, int hpw, cudaStream_t stream) {
  if (num_tokens <= 0) return 0;
  if (block_size <= 0) return 1;
  switch (hpw) {
    case 2: return muse_qk_glue::launch_kvcache_prefill<2>(qkv, positions, cos_sin_cache, q_out, kv_cache, slot_mapping, num_tokens, q_first_row, cache_rows, block_size, stride_block, stride_slot, stride_kv_head, eps, qscale, use_rope, cache_fp32, stream);
    case 5: return muse_qk_glue::launch_kvcache_prefill<5>(qkv, positions, cos_sin_cache, q_out, kv_cache, slot_mapping, num_tokens, q_first_row, cache_rows, block_size, stride_block, stride_slot, stride_kv_head, eps, qscale, use_rope, cache_fp32, stream);
    case 9: return muse_qk_glue::launch_kvcache_prefill<9>(qkv, positions, cos_sin_cache, q_out, kv_cache, slot_mapping, num_tokens, q_first_row, cache_rows, block_size, stride_block, stride_slot, stride_kv_head, eps, qscale, use_rope, cache_fp32, stream);
    default: return 1;
  }
}

extern "C" int muse_qk_norm_rope_kvcache_m1(
    const void* qkv,
    const void* positions,
    const void* cos_sin_cache,
    void* q_out,
    void* k_out,
    void* kv_cache,
    const void* slot_mapping,
    int cache_rows,
    int block_size,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    float eps,
    float qscale,
    int use_rope,
    int cache_fp32,
    cudaStream_t stream) {
  if (block_size <= 0) return 1;
  const auto* q = static_cast<const __nv_bfloat16*>(qkv);
  const auto* pos = static_cast<const int64_t*>(positions);
  auto* qo = static_cast<__nv_bfloat16*>(q_out);
  auto* ko = static_cast<__nv_bfloat16*>(k_out);
  auto* kvc = static_cast<__nv_bfloat16*>(kv_cache);
  const auto* sm = static_cast<const int64_t*>(slot_mapping);
  if (use_rope) {
    if (cache_fp32) {
      muse_qk_glue::qk_norm_rope_kvcache_m1_kernel<true, true><<<2, 288, 0, stream>>>(
          q, pos, cos_sin_cache, qo, ko, kvc, sm,
          cache_rows, block_size, stride_block, stride_slot, stride_kv_head, eps, qscale);
    } else {
      muse_qk_glue::qk_norm_rope_kvcache_m1_kernel<true, false><<<2, 288, 0, stream>>>(
          q, pos, cos_sin_cache, qo, ko, kvc, sm,
          cache_rows, block_size, stride_block, stride_slot, stride_kv_head, eps, qscale);
    }
  } else {
    muse_qk_glue::qk_norm_rope_kvcache_m1_kernel<false, true><<<2, 288, 0, stream>>>(
        q, pos, cos_sin_cache, qo, ko, kvc, sm,
        cache_rows, block_size, stride_block, stride_slot, stride_kv_head, eps, qscale);
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_qk_norm_rope_multi(
    const void* qkv, const void* positions, const void* cos_sin_cache, void* q_out, void* k_out,
    int num_tokens, int cache_rows, float eps, float qscale, int use_rope, int cache_fp32, int hpw,
    cudaStream_t stream) {
  if (num_tokens <= 0) return 0;
  switch (hpw) {
    case 2: return muse_qk_glue::launch_multi<2>(qkv, positions, cos_sin_cache, q_out, k_out, num_tokens, cache_rows, eps, qscale, use_rope, cache_fp32, stream);
    case 6: return muse_qk_glue::launch_multi<6>(qkv, positions, cos_sin_cache, q_out, k_out, num_tokens, cache_rows, eps, qscale, use_rope, cache_fp32, stream);
    case 9: return muse_qk_glue::launch_multi<9>(qkv, positions, cos_sin_cache, q_out, k_out, num_tokens, cache_rows, eps, qscale, use_rope, cache_fp32, stream);
    case 17: return muse_qk_glue::launch_multi<17>(qkv, positions, cos_sin_cache, q_out, k_out, num_tokens, cache_rows, eps, qscale, use_rope, cache_fp32, stream);
    case 34: return muse_qk_glue::launch_multi<34>(qkv, positions, cos_sin_cache, q_out, k_out, num_tokens, cache_rows, eps, qscale, use_rope, cache_fp32, stream);
    default: return 1;
  }
}

extern "C" int muse_qk_norm_rope(
    const void* qkv,
    const void* positions,
    const void* cos_sin_cache,
    void* q_out,
    void* k_out,
    int num_tokens,
    int cache_rows,
    float eps,
    float qscale,
    int use_rope,
    int cache_fp32,
    cudaStream_t stream) {
  if (num_tokens <= 0) return 0;
  const int total_warps = num_tokens * 34;
  const int threads = (num_tokens == 1) ? 32 : 256;
  const int wpb = threads >> 5;
  const int blocks = (total_warps + wpb - 1) / wpb;
  if (use_rope) {
    if (cache_fp32) {
      muse_qk_glue::qk_norm_rope_kernel<true, true><<<blocks, threads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(qkv),
          static_cast<const int64_t*>(positions),
          cos_sin_cache,
          static_cast<__nv_bfloat16*>(q_out),
          static_cast<__nv_bfloat16*>(k_out),
          num_tokens, cache_rows, eps, qscale);
    } else {
      muse_qk_glue::qk_norm_rope_kernel<true, false><<<blocks, threads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(qkv),
          static_cast<const int64_t*>(positions),
          cos_sin_cache,
          static_cast<__nv_bfloat16*>(q_out),
          static_cast<__nv_bfloat16*>(k_out),
          num_tokens, cache_rows, eps, qscale);
    }
  } else {
    muse_qk_glue::qk_norm_rope_kernel<false, true><<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(qkv),
        static_cast<const int64_t*>(positions),
        cos_sin_cache,
        static_cast<__nv_bfloat16*>(q_out),
        static_cast<__nv_bfloat16*>(k_out),
        num_tokens, cache_rows, eps, qscale);
  }
  return static_cast<int>(cudaGetLastError());
}
