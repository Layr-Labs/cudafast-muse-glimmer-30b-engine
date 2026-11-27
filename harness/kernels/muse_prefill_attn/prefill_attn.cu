// SPDX-License-Identifier: Apache-2.0
// Strict 0-ULP FA2-bit-exact fused M=512 prefill attention, prefill post_attn_norm, and
// L2 dirty-writeback discard kernels for MuseGlimmer (GB10 sm_121a, 48 SMs).
//
// Orthogonal standalone package (zero overlap with harness/kernels/gqa16_decode and
// harness/kernels/muse_norm_glue).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace muse_pf_attn {

constexpr float kLog2e = 1.44269504088896340736f;  // 0x3FB8AA3B

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ float max_ftz_f32(float a, float b) {
  float r;
  asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float add_ftz_f32(float a, float b) {
  float r;
  asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float sub_ftz_f32(float a, float b) {
  float r;
  asm("sub.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float mul_ftz_f32(float a, float b) {
  float r;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float fma_rn_ftz_f32(float a, float b, float c) {
  float r;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}

__device__ __forceinline__ float fast_exp2f(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float fast_rcpf(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float rsqrt_approx_ftz(float v) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v));
  return r;
}

__device__ __forceinline__ float full_divide(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float triton_sigmoid(float x) {
  float e;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(__fmul_rn(-x, 1.4426950408889634f)));
  return full_divide(1.0f, __fadd_rn(1.0f, e));
}

// Single-instruction NaN-propagating clamp matching triton_helpers.maximum/minimum on sm_121a!
__device__ __forceinline__ float clamp448(float y) {
  float t, r;
  asm("max.NaN.f32 %0, %1, 0fC3E00000;" : "=f"(t) : "f"(y));
  asm("min.NaN.f32 %0, %1, 0f43E00000;" : "=f"(r) : "f"(t));
  return r;
}

__device__ __forceinline__ uint16_t fp8x2(float a, float b) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(b), "f"(a));
  return r;
}

__device__ __forceinline__ uint2 ld2_cs(const void* ptr) {
  uint2 r;
  asm("ld.global.cs.v2.u32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(ptr));
  return r;
}

__device__ __forceinline__ uint4 ld4_cs(const void* ptr) {
  uint4 r;
  asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(ptr));
  return r;
}

__device__ __forceinline__ void discard_l2_128(const void* ptr) {
  asm volatile("discard.global.L2 [%0], 128;" :: "l"(ptr) : "memory");
}

__device__ __forceinline__ void cp_async_16(uint32_t smem_dst, const void* gmem_src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_dst), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    const uint32_t (&a)[4], uint32_t b0, uint32_t b1, float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b0), "r"(b1),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void ldsm4_swizzled(
    uint32_t (&a)[4], const uint4 tile[16][16], int ks, int lane) {
  const int m_row = lane & 15;
  const int c_log = (ks << 1) + (lane >> 4);
  const int c_phys = c_log ^ (m_row & 7);
  const uint32_t addr = smem_u32(&tile[m_row][c_phys]);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
}

__device__ __forceinline__ void ldsm4_k_pair64(
    uint32_t (&b)[4], const uint4 tile[64][16], int sub_pair, int ks, int lane) {
  const int r_local = lane & 7;
  const int sub = (sub_pair << 1) + (lane >> 4);
  const int sub_k = (lane >> 3) & 1;
  const int r = (sub << 3) + r_local;
  const int c_log = (ks << 1) + sub_k;
  const int c_phys = c_log ^ r_local;
  const uint32_t addr = smem_u32(&tile[r][c_phys]);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
}

__device__ __forceinline__ void ldsm4_v_pair64(
    uint32_t (&v)[4], const uint4 tile[64][16], int dt_pair, int ks_local, int lane) {
  const int r_in_16 = lane & 15;
  const int r = (ks_local << 4) + r_in_16;
  const int dt = (dt_pair << 1) + (lane >> 4);
  const int c_phys = dt ^ (r & 7);
  const uint32_t addr = smem_u32(&tile[r][c_phys]);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(addr));
}

__device__ __forceinline__ void issue_kv64_flat(
    uint4 smem_tile[64][16],
    const __nv_bfloat16* __restrict__ flat_base,
    int tok_base,
    int tid,
    int n_u = 4) {
  const int r_in_16 = tid >> 4;
  const int c = tid & 15;
  const int c_phys = c ^ (r_in_16 & 7);
  const uint4* src = reinterpret_cast<const uint4*>(
      flat_base + static_cast<int64_t>(tok_base) * 128) + r_in_16 * 16 + c;
  #pragma unroll
  for (int u = 0; u < 4; ++u) {
    if (u < n_u) {
      const int r = (u << 4) + r_in_16;
      cp_async_16(smem_u32(&smem_tile[r][c_phys]), src + u * 256);
    }
  }
}

template <bool UseRope, bool CacheFP32>
__device__ __forceinline__ void load_rope_cs(
    int64_t pos, int cache_rows, const void* __restrict__ cos_sin_cache, int lane,
    float& c0, float& c1, float& c2, float& c3,
    float& s0, float& s1, float& s2, float& s3) {
  if constexpr (UseRope) {
    if (pos < 0) pos += cache_rows;
    const int d_lo = (lane & 15) * 4;
    if constexpr (CacheFP32) {
      const float* row = reinterpret_cast<const float*>(cos_sin_cache) + pos * 128;
      const float4 cv = __ldg(reinterpret_cast<const float4*>(row + d_lo));
      const float4 sv = __ldg(reinterpret_cast<const float4*>(row + 64 + d_lo));
      c0 = cv.x; c1 = cv.y; c2 = cv.z; c3 = cv.w;
      s0 = sv.x; s1 = sv.y; s2 = sv.z; s3 = sv.w;
    } else {
      const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache) + pos * 128;
      const uint2 cv = __ldg(reinterpret_cast<const uint2*>(row + d_lo));
      const uint2 sv = __ldg(reinterpret_cast<const uint2*>(row + 64 + d_lo));
      const __nv_bfloat16* cb = reinterpret_cast<const __nv_bfloat16*>(&cv);
      const __nv_bfloat16* sb = reinterpret_cast<const __nv_bfloat16*>(&sv);
      c0 = __bfloat162float(cb[0]); c1 = __bfloat162float(cb[1]);
      c2 = __bfloat162float(cb[2]); c3 = __bfloat162float(cb[3]);
      s0 = __bfloat162float(sb[0]); s1 = __bfloat162float(sb[1]);
      s2 = __bfloat162float(sb[2]); s3 = __bfloat162float(sb[3]);
    }
  }
}

template <bool UseRope, bool ApplyQScale>
__device__ __forceinline__ uint2 qk_norm_rope_head(
    uint2 raw, float eps, float qscale, int lane,
    float c0, float c1, float c2, float c3,
    float s0, float s1, float s2, float s3) {
  const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&raw);
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
  if constexpr (ApplyQScale) {
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
  return out_v;
}

// 4-head interleaved Q-RMSNorm + qscale + RoPE directly from registers (4x ILP on shfl/div/rsqrt!)
template <bool UseRope>
__device__ __forceinline__ void q_norm_rope_4heads(
    const uint2 (&raw)[4], uint2 (&out_v)[4], float eps, float qscale, int lane,
    float c0, float c1, float c2, float c3,
    float s0, float s1, float s2, float s3) {
  float x[4][4];
  float ss[4];
  #pragma unroll
  for (int h = 0; h < 4; ++h) {
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&raw[h]);
    x[h][0] = __bfloat162float(vb[0]);
    x[h][1] = __bfloat162float(vb[1]);
    x[h][2] = __bfloat162float(vb[2]);
    x[h][3] = __bfloat162float(vb[3]);
    float s = __fmaf_rn(x[h][0], x[h][0], 0.0f);
    s = __fmaf_rn(x[h][1], x[h][1], s);
    s = __fmaf_rn(x[h][2], x[h][2], s);
    ss[h] = __fmaf_rn(x[h][3], x[h][3], s);
  }
  #pragma unroll
  for (int delta = 16; delta >= 1; delta >>= 1) {
    #pragma unroll
    for (int h = 0; h < 4; ++h) {
      ss[h] = __fadd_rn(ss[h], __shfl_xor_sync(0xffffffffu, ss[h], delta));
    }
  }
  float r[4];
  #pragma unroll
  for (int h = 0; h < 4; ++h) {
    float mv;
    asm("div.full.f32 %0, %1, 0f43000000;" : "=f"(mv) : "f"(ss[h]));
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r[h]) : "f"(mv + eps));
  }
  #pragma unroll
  for (int h = 0; h < 4; ++h) {
    float y0 = __fmul_rn(__fmul_rn(x[h][0], r[h]), qscale);
    float y1 = __fmul_rn(__fmul_rn(x[h][1], r[h]), qscale);
    float y2 = __fmul_rn(__fmul_rn(x[h][2], r[h]), qscale);
    float y3 = __fmul_rn(__fmul_rn(x[h][3], r[h]), qscale);
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
    const __nv_bfloat162 b0 = __floats2bfloat162_rn(o0, o1);
    const __nv_bfloat162 b1 = __floats2bfloat162_rn(o2, o3);
    out_v[h] = make_uint2(*reinterpret_cast<const uint32_t*>(&b0), *reinterpret_cast<const uint32_t*>(&b1));
  }
}

// =================================================================================================
// Stage 1: Tiny KV-Only K-RMSNorm + RoPE + paged kv_cache + contiguous kv_flat[2][2][seqlen][128]
// =================================================================================================
template <bool UseRope, bool CacheFP32, bool kPdl, bool kDiscardKV>
__global__ __launch_bounds__(64) void prefill_kv_only_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const int64_t* __restrict__ positions,
    const void* __restrict__ cos_sin_cache,
    __nv_bfloat16* __restrict__ kv_cache,
    __nv_bfloat16* __restrict__ kv_flat,
    const int64_t* __restrict__ slot_mapping,
    int num_tokens,
    int cache_rows,
    int block_size,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    float eps,
    const char* __restrict__ dead_ptr0,
    size_t dead_lines0,
    const char* __restrict__ dead_ptr1,
    size_t dead_lines1) {
  const int tid_global = static_cast<int>(blockIdx.x) * 64 + static_cast<int>(threadIdx.x);
  const int total_threads = static_cast<int>(gridDim.x) * 64;
  const int warp_global = (static_cast<int>(blockIdx.x) << 1) + (static_cast<int>(threadIdx.x) >> 5);
  const int total_warps = static_cast<int>(gridDim.x) << 1;
  const int lane = static_cast<int>(threadIdx.x) & 31;

  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
  }

  const int64_t flat_v_stride = static_cast<int64_t>(num_tokens) * 128;
  const int64_t flat_h_stride = flat_v_stride * 2;

  for (int t = warp_global; t < num_tokens; t += total_warps) {
    const int64_t slot = __ldg(slot_mapping + t);
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    if constexpr (UseRope) {
      load_rope_cs<true, CacheFP32>(__ldg(positions + t), cache_rows, cos_sin_cache, lane,
                                    c0, c1, c2, c3, s0, s1, s2, s3);
    }

    const __nv_bfloat16* row_kv = qkv + static_cast<int64_t>(t) * 4608 + 4096;
    const uint2 k0_raw = ld2_cs(row_kv + lane * 4);
    const uint2 k1_raw = ld2_cs(row_kv + 128 + lane * 4);
    const uint2 v0_raw = ld2_cs(row_kv + 256 + lane * 4);
    const uint2 v1_raw = ld2_cs(row_kv + 384 + lane * 4);

    if constexpr (kDiscardKV) {
      __syncwarp();
      if (lane < 8) {
        discard_l2_128(reinterpret_cast<const char*>(row_kv) + (lane << 7));
      }
    }

    const uint2 k0_out = qk_norm_rope_head<UseRope, false>(k0_raw, eps, 1.0f, lane, c0, c1, c2, c3, s0, s1, s2, s3);
    const uint2 k1_out = qk_norm_rope_head<UseRope, false>(k1_raw, eps, 1.0f, lane, c0, c1, c2, c3, s0, s1, s2, s3);

    const int64_t t_off = static_cast<int64_t>(t) * 128 + lane * 4;
    *reinterpret_cast<uint2*>(kv_flat + t_off) = k0_out;
    *reinterpret_cast<uint2*>(kv_flat + flat_v_stride + t_off) = v0_raw;
    *reinterpret_cast<uint2*>(kv_flat + flat_h_stride + t_off) = k1_out;
    *reinterpret_cast<uint2*>(kv_flat + flat_h_stride + flat_v_stride + t_off) = v1_raw;

    if (slot >= 0) {
      const int64_t blk = (block_size == 16) ? (slot >> 4)
                        : ((block_size == 64) ? (slot >> 6) : (slot / block_size));
      const int64_t off = (block_size == 16) ? (slot & 15)
                        : ((block_size == 64) ? (slot & 63) : (slot - blk * block_size));
      __nv_bfloat16* dst0 = kv_cache + blk * stride_block + off * stride_slot + lane * 4;
      __nv_bfloat16* dst1 = dst0 + stride_kv_head;
      *reinterpret_cast<uint2*>(dst0) = k0_out;
      *reinterpret_cast<uint2*>(dst0 + 128) = v0_raw;
      *reinterpret_cast<uint2*>(dst1) = k1_out;
      *reinterpret_cast<uint2*>(dst1 + 128) = v1_raw;
    }
  }

  if constexpr (kDiscardKV) {
    if (dead_ptr0) {
      for (size_t i = static_cast<size_t>(tid_global); i < dead_lines0; i += static_cast<size_t>(total_threads)) {
        discard_l2_128(dead_ptr0 + (i << 7));
      }
    }
    if (dead_ptr1) {
      for (size_t i = static_cast<size_t>(tid_global); i < dead_lines1; i += static_cast<size_t>(total_threads)) {
        discard_l2_128(dead_ptr1 + (i << 7));
      }
    }
  }
}

// =================================================================================================
// Stage 2 (v3): Lean 128-CTA Exact-12-Tile Ordered Fused Q-Norm/RoPE + Causal GQA-16 Attention
//   - Single unified loop (NO duplicated half_only body -> compact ~5,600 SASS insns!)
//   - 4-head interleaved Q-norm/RoPE directly from L1-cached qkv -> SMEM (4x ILP, 0 extra SMEM pass)
//   - Single-insn min.NaN / max.NaN clamp448
//   - Normal L1-cached gate load + inline discard.global.L2 on dead qkv & gate lines
// =================================================================================================
__device__ __forceinline__ int map_m_grp_128(int g) {
  const int grp = g >> 3;
  const int idx = g & 7;
  int base;
  switch (grp) {
    case 0: base = 56; break;
    case 1: base = 48; break;
    case 2: base = 40; break;
    case 3: base = 24; break;
    case 4: base = 32; break;
    case 5: base = 16; break;
    case 6: base = 8;  break;
    default: base = 0; break;
  }
  return base + idx;
}

template <bool kTrim>
__device__ __forceinline__ void prefill_qk_half(
    const uint32_t (&q)[8][4], const uint4 (&k)[64][16],
    float (*scores)[4], int lane, int pairs) {
  uint32_t b_cur[4], b_nxt[4];
  ldsm4_k_pair64(b_cur, k, 0, 0, lane);
  #pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    #pragma unroll
    for (int sp = 0; sp < 4; ++sp) {
      if (!kTrim || sp < pairs) {
        const bool next_pair = sp + 1 < (kTrim ? pairs : 4);
        const int next_ks = ks + (next_pair ? 0 : 1);
        if (next_ks < 8) {
          ldsm4_k_pair64(b_nxt, k, next_pair ? sp + 1 : 0, next_ks, lane);
        }
        mma_m16n8k16_bf16(q[ks], b_cur[0], b_cur[1], scores[sp * 2]);
        mma_m16n8k16_bf16(q[ks], b_cur[2], b_cur[3], scores[sp * 2 + 1]);
        if (next_ks < 8) {
          b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1];
          b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
        }
      }
    }
  }
}

template <bool kTrim>
__device__ __forceinline__ void prefill_pv_half(
    const uint32_t (*p)[4], const uint4 (&v)[64][16],
    float (*o)[4], int lane, int pairs) {
  uint32_t v_cur[4], v_nxt[4];
  ldsm4_v_pair64(v_cur, v, 0, 0, lane);
  #pragma unroll
  for (int ks_local = 0; ks_local < 4; ++ks_local) {
    if (!kTrim || ks_local < pairs) {
      #pragma unroll
      for (int dp = 0; dp < 8; ++dp) {
        const int n_dp = (dp + 1) & 7;
        const int n_ks = ks_local + (dp == 7 ? 1 : 0);
        if (n_ks < (kTrim ? pairs : 4)) {
          ldsm4_v_pair64(v_nxt, v, n_dp, n_ks, lane);
        }
        mma_m16n8k16_bf16(p[ks_local], v_cur[0], v_cur[1], o[dp * 2 + 0]);
        mma_m16n8k16_bf16(p[ks_local], v_cur[2], v_cur[3], o[dp * 2 + 1]);
        if (n_ks < (kTrim ? pairs : 4)) {
          v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1];
          v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
        }
      }
    }
  }
}

template <int Mode, bool UseRope, bool CacheFP32, bool kPdl, bool kDiscardL2>
__global__ __launch_bounds__(256, 1) void prefill_attn_fused_512_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const int64_t* __restrict__ positions,
    const void* __restrict__ cos_sin_cache,
    const __nv_bfloat16* __restrict__ kv_flat,
    const __nv_bfloat16* __restrict__ gate_ptr,
    const float* __restrict__ fp8_scale_ptr,
    void* __restrict__ out_ptr,
    int cache_rows,
    float eps,
    float qscale,
    float scale_log2) {
  __shared__ __align__(128) uint4 kv_smem[2][64][16];  // 32 KiB static SMEM

  const int lin = static_cast<int>(blockIdx.x);   // 0..127
  const int kv_head = lin & 1;
  const int m_grp = map_m_grp_128(lin >> 1);      // 0..63

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int group_id = lane >> 2;
  const int tid_in_grp = lane & 3;

  const int q_tok = (m_grp << 3) + warp_id;
  const int nb_max = (m_grp >> 4) + 1;           // 1..4
  const bool half_only = ((m_grp & 15) < 8);     // true when diagonal upper 64 KV tokens are 100% masked

  // ---------------------------------------------------------------------------
  // 1. 4-head Interleaved Fused Q-RMSNorm + qscale + NEOX RoPE directly from raw qkv!
  //    Loads 4 heads per batch via L1-cached 16B even-lane loads + 4x ILP norm/RoPE -> swizzled SMEM.
  //    Runs BEFORE griddepcontrol.wait so Wave-1 CTAs hide kv_only completely!
  // ---------------------------------------------------------------------------
  uint32_t q_frag[8][4];
  {
    uint4 (*q_tiles)[16][16] = reinterpret_cast<uint4 (*)[16][16]>(kv_smem);
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    if constexpr (UseRope) {
      load_rope_cs<true, CacheFP32>(__ldg(positions + q_tok), cache_rows, cos_sin_cache, lane,
                                    c0, c1, c2, c3, s0, s1, s2, s3);
    }
    const __nv_bfloat16* const q_row =
        qkv + static_cast<int64_t>(q_tok) * 4608 + static_cast<int64_t>(kv_head * 16) * 128;
    const int c_log = lane >> 1;
    const int c_half = lane & 1;
    const int even_lane = lane & ~1;

    // Issue ALL 16 heads in 4 batches of 4 heads (16B coalesced loads via even-lane uint4 + shfl)
    #pragma unroll
    for (int hb = 0; hb < 16; hb += 4) {
      uint2 raw[4];
      #pragma unroll
      for (int h = 0; h < 4; ++h) {
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if ((lane & 1) == 0) {
          v = *reinterpret_cast<const uint4*>(q_row + (hb + h) * 128 + even_lane * 4);
        }
        const uint32_t hi_x = __shfl_xor_sync(0xffffffffu, v.z, 1);
        const uint32_t hi_y = __shfl_xor_sync(0xffffffffu, v.w, 1);
        raw[h] = ((lane & 1) == 0) ? make_uint2(v.x, v.y) : make_uint2(hi_x, hi_y);
      }
      uint2 ov[4];
      q_norm_rope_4heads<UseRope>(raw, ov, eps, qscale, lane, c0, c1, c2, c3, s0, s1, s2, s3);
      #pragma unroll
      for (int h = 0; h < 4; ++h) {
        const int h_loc = hb + h;
        const int c_phys = c_log ^ (h_loc & 7);
        reinterpret_cast<uint2*>(&q_tiles[warp_id][h_loc][c_phys])[c_half] = ov[h];
      }
    }
    __syncwarp();
    if constexpr (kDiscardL2) {
      discard_l2_128(reinterpret_cast<const char*>(q_row) + (lane << 7));
    }
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      ldsm4_swizzled(q_frag[ks], q_tiles[warp_id], ks, lane);
    }
  }

  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  __syncthreads();

  // ---------------------------------------------------------------------------
  // 2. Single Unified 2-Buffer 32 KiB SMEM Causal Attention Loop (strict 0-ULP FA2 order)
  // ---------------------------------------------------------------------------
  const __nv_bfloat16* const k_flat = kv_flat + static_cast<int64_t>(kv_head) * (2 * 512 * 128);
  const __nv_bfloat16* const v_flat = k_flat + (512 * 128);

  float o_frag[16][4];
  #pragma unroll
  for (int dt = 0; dt < 16; ++dt) {
    o_frag[dt][0] = 0.0f; o_frag[dt][1] = 0.0f; o_frag[dt][2] = 0.0f; o_frag[dt][3] = 0.0f;
  }
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  const int init_diag_pairs = ((m_grp & 15) >> 1) + 1;
  issue_kv64_flat(kv_smem[0], k_flat, (nb_max - 1) << 7, tid, half_only ? init_diag_pairs : 4);
  cp_async_commit();

  for (int nb = nb_max - 1; nb >= 0; --nb) {
    const int block_tok_base = nb << 7;
    const bool is_diag = (nb == nb_max - 1);
    const bool skip_hi = is_diag && half_only;
    const int diag_pairs = ((m_grp & 15) >> 1) + 1;
    const int lo_pairs = is_diag && skip_hi ? diag_pairs : 4;
    const int hi_pairs = is_diag ? diag_pairs - 4 : 4;

    float s_frag[16][4];
    #pragma unroll
    for (int ni = 0; ni < 16; ++ni) {
      s_frag[ni][0] = 0.0f; s_frag[ni][1] = 0.0f; s_frag[ni][2] = 0.0f; s_frag[ni][3] = 0.0f;
    }

    if (!skip_hi) {
      issue_kv64_flat(kv_smem[1], k_flat, block_tok_base + 64, tid, hi_pairs);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      issue_kv64_flat(kv_smem[1], v_flat, block_tok_base, tid, lo_pairs);
      cp_async_commit();
      cp_async_wait<1>();
    }
    __syncthreads();
    if (lo_pairs < 4) {
      prefill_qk_half<true>(q_frag, kv_smem[0], s_frag, lane, lo_pairs);
    } else {
      prefill_qk_half<false>(q_frag, kv_smem[0], s_frag, lane, 4);
    }
    __syncthreads();

    if (!skip_hi) {
      issue_kv64_flat(kv_smem[0], v_flat, block_tok_base, tid);
      cp_async_commit();
      cp_async_wait<1>();
      __syncthreads();
      if (hi_pairs < 4) {
        prefill_qk_half<true>(q_frag, kv_smem[1], s_frag + 8, lane, hi_pairs);
      } else {
        prefill_qk_half<false>(q_frag, kv_smem[1], s_frag + 8, lane, 4);
      }
      __syncthreads();
      issue_kv64_flat(kv_smem[1], v_flat, block_tok_base + 64, tid, hi_pairs);
      cp_async_commit();
    }

    float m0 = -INFINITY, m1 = -INFINITY;
    #pragma unroll
    for (int ni = 0; ni < 16; ++ni) {
      if (is_diag && ni >= 2 * diag_pairs) {
        s_frag[ni][0] = s_frag[ni][1] = s_frag[ni][2] = s_frag[ni][3] = -INFINITY;
        continue;
      }
      if (is_diag) {
        const int col0 = block_tok_base + (ni << 3) + (tid_in_grp << 1);
        const int col1 = col0 + 1;
        if (col0 > q_tok) { s_frag[ni][0] = -INFINITY; s_frag[ni][2] = -INFINITY; }
        if (col1 > q_tok) { s_frag[ni][1] = -INFINITY; s_frag[ni][3] = -INFINITY; }
      }
      m0 = max_ftz_f32(m0, max_ftz_f32(s_frag[ni][0], s_frag[ni][1]));
      m1 = max_ftz_f32(m1, max_ftz_f32(s_frag[ni][2], s_frag[ni][3]));
    }
    m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 1));

    float sc0 = 1.0f, sc1 = 1.0f;
    if (is_diag) {
      row_max0 = m0;
      row_max1 = m1;
    } else {
      const float prev0 = row_max0, prev1 = row_max1;
      row_max0 = max_ftz_f32(prev0, m0);
      row_max1 = max_ftz_f32(prev1, m1);
      const float cur0 = (row_max0 == -INFINITY) ? 0.0f : row_max0;
      const float cur1 = (row_max1 == -INFINITY) ? 0.0f : row_max1;
      sc0 = fast_exp2f(mul_ftz_f32(scale_log2, sub_ftz_f32(prev0, cur0)));
      sc1 = fast_exp2f(mul_ftz_f32(scale_log2, sub_ftz_f32(prev1, cur1)));
      #pragma unroll
      for (int dt = 0; dt < 16; ++dt) {
        o_frag[dt][0] = mul_ftz_f32(o_frag[dt][0], sc0);
        o_frag[dt][1] = mul_ftz_f32(o_frag[dt][1], sc0);
        o_frag[dt][2] = mul_ftz_f32(o_frag[dt][2], sc1);
        o_frag[dt][3] = mul_ftz_f32(o_frag[dt][3], sc1);
      }
    }

    const float max_sc0 = (row_max0 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max0);
    const float max_sc1 = (row_max1 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max1);
    const bool skip_masked_exp = is_diag && scale_log2 >= 0x1p-126f &&
        scale_log2 < INFINITY && fabsf(max_sc0) < INFINITY && fabsf(max_sc1) < INFINITY;

    uint32_t p_frag[8][4];
    #pragma unroll
    for (int ni = 0; ni < 16; ++ni) {
      if (skip_masked_exp && ni >= 2 * diag_pairs) {
        p_frag[ni >> 1][(ni & 1) << 1] = 0;
        p_frag[ni >> 1][((ni & 1) << 1) + 1] = 0;
        continue;
      }
      const float p0 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ni][0], scale_log2), max_sc0));
      const float p1 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ni][1], scale_log2), max_sc0));
      const float p2 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ni][2], scale_log2), max_sc1));
      const float p3 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ni][3], scale_log2), max_sc1));
      if (ni == 0) {
        if (is_diag) {
          row_sum0 = add_ftz_f32(p0, p1);
          row_sum1 = add_ftz_f32(p2, p3);
        } else {
          row_sum0 = add_ftz_f32(fma_rn_ftz_f32(row_sum0, sc0, p0), p1);
          row_sum1 = add_ftz_f32(fma_rn_ftz_f32(row_sum1, sc1, p2), p3);
        }
      } else {
        row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, p0), p1);
        row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, p2), p3);
      }
      const __nv_bfloat162 pb0 = __floats2bfloat162_rn(p0, p1);
      const __nv_bfloat162 pb1 = __floats2bfloat162_rn(p2, p3);
      const int kp = ni >> 1;
      const int hi = (ni & 1) << 1;
      p_frag[kp][hi + 0] = *reinterpret_cast<const uint32_t*>(&pb0);
      p_frag[kp][hi + 1] = *reinterpret_cast<const uint32_t*>(&pb1);
    }

    if (skip_hi) {
      // V0 is in kv_smem[1]; pre-issue next K0 into kv_smem[0] right now!
      if (nb > 0) {
        issue_kv64_flat(kv_smem[0], k_flat, (nb - 1) << 7, tid);
        cp_async_commit();
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
        if constexpr (kPdl) {
          asm volatile("griddepcontrol.launch_dependents;");
        }
      }
      __syncthreads();
      if (skip_masked_exp && lo_pairs < 4) {
        prefill_pv_half<true>(p_frag, kv_smem[1], o_frag, lane, lo_pairs);
      } else {
        prefill_pv_half<false>(p_frag, kv_smem[1], o_frag, lane, 4);
      }
      __syncthreads();
    } else {
      cp_async_wait<1>();
      __syncthreads();
      prefill_pv_half<false>(p_frag, kv_smem[0], o_frag, lane, 4);
      __syncthreads();

      if (nb > 0) {
        issue_kv64_flat(kv_smem[0], k_flat, (nb - 1) << 7, tid);
        cp_async_commit();
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
        if constexpr (kPdl) {
          asm volatile("griddepcontrol.launch_dependents;");
        }
      }
      __syncthreads();
      if (skip_masked_exp && hi_pairs < 4) {
        prefill_pv_half<true>(p_frag + 4, kv_smem[1], o_frag, lane, hi_pairs);
      } else {
        prefill_pv_half<false>(p_frag + 4, kv_smem[1], o_frag, lane, 4);
      }
      __syncthreads();
    }
  }

  // ---------------------------------------------------------------------------
  // 3. Fused Epilogue (L1-cached gate load + single-insn NaN clamp + inline L2 discard)
  // ---------------------------------------------------------------------------
  row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 2));
  row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 1));
  row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 2));
  row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 1));
  const float inv0 = (row_sum0 == 0.0f || row_sum0 != row_sum0) ? 1.0f : fast_rcpf(row_sum0);
  const float inv1 = (row_sum1 == 0.0f || row_sum1 != row_sum1) ? 1.0f : fast_rcpf(row_sum1);

  uint32_t (*o_tiles)[16][64] = reinterpret_cast<uint32_t (*)[16][64]>(kv_smem);
  #pragma unroll
  for (int dt = 0; dt < 16; ++dt) {
    const int c32 = (dt << 2) + tid_in_grp;
    const __nv_bfloat162 ob0 = __floats2bfloat162_rn(mul_ftz_f32(o_frag[dt][0], inv0), mul_ftz_f32(o_frag[dt][1], inv0));
    const __nv_bfloat162 ob1 = __floats2bfloat162_rn(mul_ftz_f32(o_frag[dt][2], inv1), mul_ftz_f32(o_frag[dt][3], inv1));
    o_tiles[warp_id][group_id][c32]     = *reinterpret_cast<const uint32_t*>(&ob0);
    o_tiles[warp_id][group_id + 8][c32] = *reinterpret_cast<const uint32_t*>(&ob1);
  }
  __syncwarp();

  const int64_t tok_head_base = static_cast<int64_t>(q_tok) * 4096 + static_cast<int64_t>(kv_head * 16) * 128;
  const uint4* o_v4 = reinterpret_cast<const uint4*>(o_tiles[warp_id]);
  if constexpr (Mode == 0) {
    uint4* dst = reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(out_ptr) + tok_head_base);
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = (u << 5) + lane;
      dst[idx] = o_v4[idx];
    }
  } else if constexpr (Mode == 1) {
    const uint4* g_v4 = reinterpret_cast<const uint4*>(gate_ptr + tok_head_base);
    uint4* dst = reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(out_ptr) + tok_head_base);
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = (u << 5) + lane;
      const uint4 ov = o_v4[idx];
      const uint4 gv = g_v4[idx];
      const uint32_t ow[4] = {ov.x, ov.y, ov.z, ov.w};
      const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
      uint32_t rw[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 of = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ow[i]));
        const float2 gf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gw[i]));
        const __nv_bfloat162 r = __floats2bfloat162_rn(__fmul_rn(triton_sigmoid(gf.x), of.x),
                                                       __fmul_rn(triton_sigmoid(gf.y), of.y));
        rw[i] = *reinterpret_cast<const uint32_t*>(&r);
      }
      dst[idx] = make_uint4(rw[0], rw[1], rw[2], rw[3]);
    }
    if constexpr (kDiscardL2) {
      __syncwarp();
      discard_l2_128(reinterpret_cast<const char*>(g_v4) + (lane << 7));
    }
  } else {
    const float inv_s = full_divide(1.0f, __ldg(fp8_scale_ptr));
    const uint4* g_v4 = reinterpret_cast<const uint4*>(gate_ptr + tok_head_base);
    uint2* dst = reinterpret_cast<uint2*>(static_cast<uint8_t*>(out_ptr) + tok_head_base);
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = (u << 5) + lane;
      const uint4 ov = o_v4[idx];
      const uint4 gv = g_v4[idx];
      const uint32_t ow[4] = {ov.x, ov.y, ov.z, ov.w};
      const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
      uint16_t qw[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 of = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ow[i]));
        const float2 gf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gw[i]));
        qw[i] = fp8x2(clamp448(__fmul_rn(__fmul_rn(triton_sigmoid(gf.x), of.x), inv_s)),
                      clamp448(__fmul_rn(__fmul_rn(triton_sigmoid(gf.y), of.y), inv_s)));
      }
      dst[idx] = make_uint2(static_cast<uint32_t>(qw[0]) | (static_cast<uint32_t>(qw[1]) << 16),
                            static_cast<uint32_t>(qw[2]) | (static_cast<uint32_t>(qw[3]) << 16));
    }
    if constexpr (kDiscardL2) {
      __syncwarp();
      discard_l2_128(reinterpret_cast<const char*>(g_v4) + (lane << 7));
    }
  }
}

// =================================================================================================
// Stage 3: Fast Prefill post_attn_norm (0-ULP exact, 16B uint4 coalesced, no harmful sh_res SMEM,
//          optional inline L2 discard of dead attn_out in 0 extra launches!)
// =================================================================================================
constexpr int kHidden = 6656;
constexpr int kChunks = 7;

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&p);
}

__device__ __forceinline__ float inv_rms_6656(float s, float eps) {
  return rsqrt_approx_ftz(__fmaf_rn(s, __int_as_float(0x391d89d9), eps));
}

__device__ __forceinline__ bool chunk_valid_6656(int c) {
  return c < kChunks - 1 || threadIdx.x < 128;
}

__device__ __forceinline__ float row_sum_256(const float (&acc)[4], float* red) {
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
  return v;
}

template <bool kCs>
__device__ __forceinline__ uint2 ld8_direct(const __nv_bfloat16* base, int c) {
  const __nv_bfloat16* ptr = base + c * 1024 + threadIdx.x * 4;
  return kCs ? ld2_cs(ptr) : *reinterpret_cast<const uint2*>(ptr);
}

__device__ __forceinline__ uint2 ldg8_direct(const __nv_bfloat16* base, int c) {
  return __ldg(reinterpret_cast<const uint2*>(base + c * 1024 + threadIdx.x * 4));
}

template <bool kBf16Out, bool kFp8Out, bool kCsX, bool kCsRes, bool kPdl = true>
__global__ void __launch_bounds__(256, 4) prefill_post_attn_norm_kernel(
    const __nv_bfloat16* __restrict__ x_all,
    const __nv_bfloat16* __restrict__ res_all,
    const __nv_bfloat16* __restrict__ w_post,
    const __nv_bfloat16* __restrict__ w_pre,
    const float* __restrict__ scale,
    float* __restrict__ sumsq_out,
    __nv_bfloat16* __restrict__ out_bf16_all,
    uint8_t* __restrict__ out_fp8_all,
    const char* __restrict__ dead_ptr,
    size_t dead_lines) {
  __shared__ float red[2][8];
  __shared__ uint2 sh_res[kPdl ? (kHidden / 4) : 1];
  const size_t row = blockIdx.x;
  const __nv_bfloat16* x = x_all + row * kHidden;
  const __nv_bfloat16* res = res_all + row * kHidden;
  float inv_scale = 0.0f;

  if constexpr (kPdl) {
    // Preload predecessor-independent residual (6.8 MB) into SMEM before waiting on o_proj!
    #pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      if (chunk_valid_6656(c)) {
        sh_res[c * 256 + threadIdx.x] = ld8_direct<kCsRes>(res, c);
      }
    }
    if constexpr (kFp8Out) inv_scale = fast_rcpf(__ldg(scale));
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
  }

  // Discard dead attn_out strictly AFTER pdl_wait() so o_proj has finished reading it!
  if (dead_ptr) {
    const size_t gid = row * 256 + threadIdx.x;
    const size_t gstride = static_cast<size_t>(gridDim.x) * 256;
    for (size_t i = gid; i < dead_lines; i += gstride) {
      discard_l2_128(dead_ptr + (i << 7));
    }
  }

  uint2 ux[kChunks];
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  #pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid_6656(c)) {
      ux[c] = ld8_direct<kCsX>(x, c);
      const float v0 = bf_lo(ux[c].x), v1 = bf_hi(ux[c].x);
      const float v2 = bf_lo(ux[c].y), v3 = bf_hi(ux[c].y);
      acc[0] = __fmaf_rn(v0, v0, acc[0]);
      acc[1] = __fmaf_rn(v1, v1, acc[1]);
      acc[2] = __fmaf_rn(v2, v2, acc[2]);
      acc[3] = __fmaf_rn(v3, v3, acc[3]);
    }
  }
  const float s_x = row_sum_256(acc, red[0]);
  if (threadIdx.x == 0) sumsq_out[row] = s_x;
  const float a = inv_rms_6656(s_x, 1e-08f);

  float h[kChunks][4];
  float acc2[4] = {0.f, 0.f, 0.f, 0.f};
  #pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid_6656(c)) {
      const uint2 rv_u = kPdl ? sh_res[c * 256 + threadIdx.x] : ld8_direct<kCsRes>(res, c);
      const uint2 wv_u = ldg8_direct(w_post, c);
      const float xv[4] = {bf_lo(ux[c].x), bf_hi(ux[c].x), bf_lo(ux[c].y), bf_hi(ux[c].y)};
      const float rv[4] = {bf_lo(rv_u.x), bf_hi(rv_u.x), bf_lo(rv_u.y), bf_hi(rv_u.y)};
      const float wv[4] = {bf_lo(wv_u.x), bf_hi(wv_u.x), bf_lo(wv_u.y), bf_hi(wv_u.y)};
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        h[c][e] = __fmaf_rn(__fmul_rn(a, xv[e]), __fadd_rn(wv[e], 1.0f), rv[e]);
        acc2[e] = __fmaf_rn(h[c][e], h[c][e], acc2[e]);
      }
    }
  }
  const float s_h = row_sum_256(acc2, red[1]);
  const float b = inv_rms_6656(s_h, 1e-05f);
  if constexpr (!kPdl && kFp8Out) inv_scale = fast_rcpf(__ldg(scale));

  __nv_bfloat16* const out_bf16_row = kBf16Out ? (out_bf16_all + row * kHidden) : nullptr;
  uint8_t* const out_fp8_row = kFp8Out ? (out_fp8_all + row * kHidden) : nullptr;
  #pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (chunk_valid_6656(c)) {
      const uint2 pv_u = ldg8_direct(w_pre, c);
      const float pv[4] = {bf_lo(pv_u.x), bf_hi(pv_u.x), bf_lo(pv_u.y), bf_hi(pv_u.y)};
      float y[4];
      #pragma unroll
      for (int e = 0; e < 4; ++e) y[e] = __fmul_rn(__fmul_rn(b, h[c][e]), __fadd_rn(pv[e], 1.0f));
      const size_t off = c * 1024 + threadIdx.x * 4;
      if constexpr (kBf16Out) {
        *reinterpret_cast<uint2*>(out_bf16_row + off) = make_uint2(pack_bf16x2(y[0], y[1]), pack_bf16x2(y[2], y[3]));
      }
      if constexpr (kFp8Out) {
        const uint32_t q =
            static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[0])), clamp448(__fmul_rn(inv_scale, y[1])))) |
            (static_cast<uint32_t>(fp8x2(clamp448(__fmul_rn(inv_scale, y[2])), clamp448(__fmul_rn(inv_scale, y[3])))) << 16);
        *reinterpret_cast<uint32_t*>(out_fp8_row + off) = q;
      }
    }
  }
}

static bool post_attn_pdl_enabled() {
  static const bool on = []() {
    const char* env = std::getenv("MUSE_PREFILL_POST_ATTN_PDL");
    return env && *env && std::strtol(env, nullptr, 10) != 0;
  }();
  return on;
}

// =================================================================================================
// Launch & Dispatch Helpers
// =================================================================================================
template <typename... KArgs, typename... Args>
static inline cudaError_t launch_ex(
    bool pdl, dim3 grid, dim3 block, cudaStream_t st,
    void (*kernel)(KArgs...), Args... args) {
  if (pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...);
  }
  kernel<<<grid, block, 0, st>>>(static_cast<KArgs>(args)...);
  return cudaGetLastError();
}

static inline void align_discard(const void* ptr, size_t n_bytes, const char*& out_ptr, size_t& out_lines) {
  out_ptr = nullptr;
  out_lines = 0;
  if (!ptr || n_bytes < 128) return;
  const uintptr_t a = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t aligned = (a + 127u) & ~static_cast<uintptr_t>(127u);
  if (aligned >= a + n_bytes) return;
  const size_t n_lines = (a + n_bytes - aligned) >> 7;
  if (n_lines == 0) return;
  out_ptr = reinterpret_cast<const char*>(aligned);
  out_lines = n_lines;
}

template <int Mode, bool kPdl, bool kDiscard>
static inline cudaError_t dispatch_attn_rope(
    int use_rope, int cache_fp32, cudaStream_t st,
    const __nv_bfloat16* qkv, const int64_t* pos, const void* cos_sin,
    const __nv_bfloat16* kv_flat, const __nv_bfloat16* gate, const float* fp8_scale,
    void* out, int cache_rows, float eps, float qscale, float scale_log2) {
  const dim3 grid(128u, 1u, 1u);
  const dim3 block(256u, 1u, 1u);
  if (use_rope) {
    if (cache_fp32) {
      return launch_ex(kPdl, grid, block, st,
                       prefill_attn_fused_512_kernel<Mode, true, true, kPdl, kDiscard>,
                       qkv, pos, cos_sin, kv_flat, gate, fp8_scale, out, cache_rows, eps, qscale, scale_log2);
    }
    return launch_ex(kPdl, grid, block, st,
                     prefill_attn_fused_512_kernel<Mode, true, false, kPdl, kDiscard>,
                     qkv, pos, cos_sin, kv_flat, gate, fp8_scale, out, cache_rows, eps, qscale, scale_log2);
  }
  return launch_ex(kPdl, grid, block, st,
                   prefill_attn_fused_512_kernel<Mode, false, true, kPdl, kDiscard>,
                   qkv, pos, cos_sin, kv_flat, gate, fp8_scale, out, cache_rows, eps, qscale, scale_log2);
}

template <int Mode>
static inline cudaError_t dispatch_attn_pdl(
    int pdl, int discard_l2, int use_rope, int cache_fp32, cudaStream_t st,
    const __nv_bfloat16* qkv, const int64_t* pos, const void* cos_sin,
    const __nv_bfloat16* kv_flat, const __nv_bfloat16* gate, const float* fp8_scale,
    void* out, int cache_rows, float eps, float qscale, float scale_log2) {
  if (pdl) {
    if (discard_l2) return dispatch_attn_rope<Mode, true, true>(use_rope, cache_fp32, st, qkv, pos, cos_sin, kv_flat, gate, fp8_scale, out, cache_rows, eps, qscale, scale_log2);
    return dispatch_attn_rope<Mode, true, false>(use_rope, cache_fp32, st, qkv, pos, cos_sin, kv_flat, gate, fp8_scale, out, cache_rows, eps, qscale, scale_log2);
  }
  if (discard_l2) return dispatch_attn_rope<Mode, false, true>(use_rope, cache_fp32, st, qkv, pos, cos_sin, kv_flat, gate, fp8_scale, out, cache_rows, eps, qscale, scale_log2);
  return dispatch_attn_rope<Mode, false, false>(use_rope, cache_fp32, st, qkv, pos, cos_sin, kv_flat, gate, fp8_scale, out, cache_rows, eps, qscale, scale_log2);
}

template <bool kPdl, bool kDiscard>
static inline cudaError_t dispatch_kv_only(
    int use_rope, int cache_fp32, cudaStream_t st,
    const __nv_bfloat16* qkv, const int64_t* pos, const void* cos_sin,
    __nv_bfloat16* kvc, __nv_bfloat16* kvf, const int64_t* slots,
    int num_tokens, int cache_rows, int block_size,
    int64_t stride_block, int64_t stride_slot, int64_t stride_kv_head, float eps,
    const char* d0, size_t l0, const char* d1, size_t l1) {
  const dim3 grid(48u, 1u, 1u);
  const dim3 block(64u, 1u, 1u);
  if (use_rope) {
    if (cache_fp32) {
      return launch_ex(kPdl, grid, block, st,
          prefill_kv_only_kernel<true, true, kPdl, kDiscard>,
          qkv, pos, cos_sin, kvc, kvf, slots, num_tokens, cache_rows, block_size,
          stride_block, stride_slot, stride_kv_head, eps, d0, l0, d1, l1);
    }
    return launch_ex(kPdl, grid, block, st,
        prefill_kv_only_kernel<true, false, kPdl, kDiscard>,
        qkv, pos, cos_sin, kvc, kvf, slots, num_tokens, cache_rows, block_size,
        stride_block, stride_slot, stride_kv_head, eps, d0, l0, d1, l1);
  }
  return launch_ex(kPdl, grid, block, st,
      prefill_kv_only_kernel<false, true, kPdl, kDiscard>,
      qkv, pos, cos_sin, kvc, kvf, slots, num_tokens, cache_rows, block_size,
      stride_block, stride_slot, stride_kv_head, eps, d0, l0, d1, l1);
}

__global__ __launch_bounds__(256) void discard_l2_range_kernel(const char* ptr, size_t n_lines) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
       i < n_lines;
       i += static_cast<size_t>(gridDim.x) * 256) {
    discard_l2_128(ptr + (i << 7));
  }
}

__global__ __launch_bounds__(256) void discard_l2_multi2_kernel(
    const char* p0, size_t l0, const char* p1, size_t l1) {
  const size_t gid = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
  const size_t gstride = static_cast<size_t>(gridDim.x) * 256;
  const size_t total = l0 + l1;
  for (size_t i = gid; i < total; i += gstride) {
    if (i < l0) discard_l2_128(p0 + (i << 7));
    else discard_l2_128(p1 + ((i - l0) << 7));
  }
}

}  // namespace muse_pf_attn

extern "C" int muse_prefill_kv_only(
    const void* qkv_ptr, const void* pos_ptr, const void* cos_sin_ptr,
    void* kv_cache_ptr, void* kv_flat_ptr, const void* slot_ptr,
    int num_tokens, int cache_rows, int block_size,
    int64_t stride_block, int64_t stride_slot, int64_t stride_kv_head,
    float eps, int use_rope, int cache_fp32, int pdl, int discard_l2, void* cuda_stream) {
  if (!qkv_ptr || !kv_cache_ptr || !kv_flat_ptr || !slot_ptr || num_tokens <= 0 || block_size <= 0) {
    return 1;
  }
  const auto* qkv = static_cast<const __nv_bfloat16*>(qkv_ptr);
  const auto* pos = static_cast<const int64_t*>(pos_ptr);
  auto* kvc = static_cast<__nv_bfloat16*>(kv_cache_ptr);
  auto* kvf = static_cast<__nv_bfloat16*>(kv_flat_ptr);
  const auto* slots = static_cast<const int64_t*>(slot_ptr);
  const cudaStream_t st = static_cast<cudaStream_t>(cuda_stream);
  cudaError_t err;
  if (pdl) {
    err = discard_l2
        ? muse_pf_attn::dispatch_kv_only<true, true>(use_rope, cache_fp32, st, qkv, pos, cos_sin_ptr, kvc, kvf, slots, num_tokens, cache_rows, block_size, stride_block, stride_slot, stride_kv_head, eps, nullptr, 0, nullptr, 0)
        : muse_pf_attn::dispatch_kv_only<true, false>(use_rope, cache_fp32, st, qkv, pos, cos_sin_ptr, kvc, kvf, slots, num_tokens, cache_rows, block_size, stride_block, stride_slot, stride_kv_head, eps, nullptr, 0, nullptr, 0);
  } else {
    err = discard_l2
        ? muse_pf_attn::dispatch_kv_only<false, true>(use_rope, cache_fp32, st, qkv, pos, cos_sin_ptr, kvc, kvf, slots, num_tokens, cache_rows, block_size, stride_block, stride_slot, stride_kv_head, eps, nullptr, 0, nullptr, 0)
        : muse_pf_attn::dispatch_kv_only<false, false>(use_rope, cache_fp32, st, qkv, pos, cos_sin_ptr, kvc, kvf, slots, num_tokens, cache_rows, block_size, stride_block, stride_slot, stride_kv_head, eps, nullptr, 0, nullptr, 0);
  }
  return static_cast<int>(err);
}

extern "C" int muse_prefill_kv_only_ex(
    const void* qkv_ptr, const void* pos_ptr, const void* cos_sin_ptr,
    void* kv_cache_ptr, void* kv_flat_ptr, const void* slot_ptr,
    int num_tokens, int cache_rows, int block_size,
    int64_t stride_block, int64_t stride_slot, int64_t stride_kv_head,
    float eps, int use_rope, int cache_fp32, int pdl, int discard_l2,
    const void* dead_ptr0, size_t dead_bytes0,
    const void* dead_ptr1, size_t dead_bytes1, void* cuda_stream) {
  if (!qkv_ptr || !kv_cache_ptr || !kv_flat_ptr || !slot_ptr || num_tokens <= 0 || block_size <= 0) {
    return 1;
  }
  const auto* qkv = static_cast<const __nv_bfloat16*>(qkv_ptr);
  const auto* pos = static_cast<const int64_t*>(pos_ptr);
  auto* kvc = static_cast<__nv_bfloat16*>(kv_cache_ptr);
  auto* kvf = static_cast<__nv_bfloat16*>(kv_flat_ptr);
  const auto* slots = static_cast<const int64_t*>(slot_ptr);
  const cudaStream_t st = static_cast<cudaStream_t>(cuda_stream);
  const char* d0 = nullptr; size_t l0 = 0;
  const char* d1 = nullptr; size_t l1 = 0;
  if (discard_l2) {
    muse_pf_attn::align_discard(dead_ptr0, dead_bytes0, d0, l0);
    muse_pf_attn::align_discard(dead_ptr1, dead_bytes1, d1, l1);
  }
  cudaError_t err;
  if (pdl) {
    err = discard_l2
        ? muse_pf_attn::dispatch_kv_only<true, true>(use_rope, cache_fp32, st, qkv, pos, cos_sin_ptr, kvc, kvf, slots, num_tokens, cache_rows, block_size, stride_block, stride_slot, stride_kv_head, eps, d0, l0, d1, l1)
        : muse_pf_attn::dispatch_kv_only<true, false>(use_rope, cache_fp32, st, qkv, pos, cos_sin_ptr, kvc, kvf, slots, num_tokens, cache_rows, block_size, stride_block, stride_slot, stride_kv_head, eps, nullptr, 0, nullptr, 0);
  } else {
    err = discard_l2
        ? muse_pf_attn::dispatch_kv_only<false, true>(use_rope, cache_fp32, st, qkv, pos, cos_sin_ptr, kvc, kvf, slots, num_tokens, cache_rows, block_size, stride_block, stride_slot, stride_kv_head, eps, d0, l0, d1, l1)
        : muse_pf_attn::dispatch_kv_only<false, false>(use_rope, cache_fp32, st, qkv, pos, cos_sin_ptr, kvc, kvf, slots, num_tokens, cache_rows, block_size, stride_block, stride_slot, stride_kv_head, eps, nullptr, 0, nullptr, 0);
  }
  return static_cast<int>(err);
}

extern "C" int muse_prefill_attn_fused_512(
    const void* qkv_ptr, const void* pos_ptr, const void* cos_sin_ptr,
    const void* kv_flat_ptr, const void* gate_ptr, const void* fp8_scale_ptr,
    void* out_ptr, int cache_rows, int mode, int scale_mode,
    int use_rope, int cache_fp32, int pdl, int discard_l2,
    float eps, float qscale, float softmax_scale, void* cuda_stream) {
  if (!qkv_ptr || !kv_flat_ptr || !out_ptr || mode < 0 || mode > 2) return 1;
  const bool use_scale_mode3 = ((scale_mode & 3) == 3);
  const float scale_log2 =
      use_scale_mode3
          ? (softmax_scale * muse_pf_attn::kLog2e)
          : static_cast<float>(static_cast<double>(softmax_scale) * 1.44269504088896340736);
  const cudaStream_t st = static_cast<cudaStream_t>(cuda_stream);
  const auto* qkv = static_cast<const __nv_bfloat16*>(qkv_ptr);
  const auto* pos = static_cast<const int64_t*>(pos_ptr);
  const auto* kvf = static_cast<const __nv_bfloat16*>(kv_flat_ptr);
  const auto* gate = static_cast<const __nv_bfloat16*>(gate_ptr);
  const auto* fp8s = static_cast<const float*>(fp8_scale_ptr);

  cudaError_t err;
  if (mode == 0) {
    err = muse_pf_attn::dispatch_attn_pdl<0>(pdl, discard_l2, use_rope, cache_fp32, st, qkv, pos, cos_sin_ptr, kvf, gate, fp8s, out_ptr, cache_rows, eps, qscale, scale_log2);
  } else if (mode == 1) {
    err = muse_pf_attn::dispatch_attn_pdl<1>(pdl, discard_l2, use_rope, cache_fp32, st, qkv, pos, cos_sin_ptr, kvf, gate, fp8s, out_ptr, cache_rows, eps, qscale, scale_log2);
  } else {
    err = muse_pf_attn::dispatch_attn_pdl<2>(pdl, discard_l2, use_rope, cache_fp32, st, qkv, pos, cos_sin_ptr, kvf, gate, fp8s, out_ptr, cache_rows, eps, qscale, scale_log2);
  }
  return static_cast<int>(err);
}

extern "C" int muse_prefill_post_attn_norm(
    int kind, int rows, int cs_mode,
    const void* x, const void* res, const void* w_post, const void* w_pre,
    const void* scale, void* sumsq_out, void* out_bf16, void* out_fp8,
    const void* dead_ptr, size_t dead_bytes, void* cuda_stream) {
  if (rows <= 0) return 0;
  const auto* X = static_cast<const __nv_bfloat16*>(x);
  const auto* Rs = static_cast<const __nv_bfloat16*>(res);
  const auto* Wp = static_cast<const __nv_bfloat16*>(w_post);
  const auto* Wq = static_cast<const __nv_bfloat16*>(w_pre);
  const auto* Sc = static_cast<const float*>(scale);
  auto* Ss = static_cast<float*>(sumsq_out);
  auto* Ob = static_cast<__nv_bfloat16*>(out_bf16);
  auto* Of = static_cast<uint8_t*>(out_fp8);
  const cudaStream_t st = static_cast<cudaStream_t>(cuda_stream);
  const char* dp = nullptr; size_t dl = 0;
  muse_pf_attn::align_discard(dead_ptr, dead_bytes, dp, dl);
  const bool pdl = muse_pf_attn::post_attn_pdl_enabled();
  const dim3 grid(static_cast<unsigned>(rows), 1u, 1u);
  const dim3 block(256u, 1u, 1u);
  if (kind == 0) {
    if (pdl) {
      if (cs_mode == 0)
        return static_cast<int>(muse_pf_attn::launch_ex(true, grid, block, st, muse_pf_attn::prefill_post_attn_norm_kernel<true, false, false, false, true>, X, Rs, Wp, Wq, static_cast<const float*>(nullptr), Ss, Ob, static_cast<uint8_t*>(nullptr), dp, dl));
      if (cs_mode == 2)
        return static_cast<int>(muse_pf_attn::launch_ex(true, grid, block, st, muse_pf_attn::prefill_post_attn_norm_kernel<true, false, false, true, true>, X, Rs, Wp, Wq, static_cast<const float*>(nullptr), Ss, Ob, static_cast<uint8_t*>(nullptr), dp, dl));
      return static_cast<int>(muse_pf_attn::launch_ex(true, grid, block, st, muse_pf_attn::prefill_post_attn_norm_kernel<true, false, true, true, true>, X, Rs, Wp, Wq, static_cast<const float*>(nullptr), Ss, Ob, static_cast<uint8_t*>(nullptr), dp, dl));
    }
    if (cs_mode == 0)
      muse_pf_attn::prefill_post_attn_norm_kernel<true, false, false, false, false><<<rows, 256, 0, st>>>(X, Rs, Wp, Wq, nullptr, Ss, Ob, nullptr, dp, dl);
    else if (cs_mode == 2)
      muse_pf_attn::prefill_post_attn_norm_kernel<true, false, false, true, false><<<rows, 256, 0, st>>>(X, Rs, Wp, Wq, nullptr, Ss, Ob, nullptr, dp, dl);
    else
      muse_pf_attn::prefill_post_attn_norm_kernel<true, false, true, true, false><<<rows, 256, 0, st>>>(X, Rs, Wp, Wq, nullptr, Ss, Ob, nullptr, dp, dl);
  } else if (kind == 1) {
    if (pdl) {
      if (cs_mode == 0)
        return static_cast<int>(muse_pf_attn::launch_ex(true, grid, block, st, muse_pf_attn::prefill_post_attn_norm_kernel<false, true, false, false, true>, X, Rs, Wp, Wq, Sc, Ss, static_cast<__nv_bfloat16*>(nullptr), Of, dp, dl));
      if (cs_mode == 2)
        return static_cast<int>(muse_pf_attn::launch_ex(true, grid, block, st, muse_pf_attn::prefill_post_attn_norm_kernel<false, true, false, true, true>, X, Rs, Wp, Wq, Sc, Ss, static_cast<__nv_bfloat16*>(nullptr), Of, dp, dl));
      return static_cast<int>(muse_pf_attn::launch_ex(true, grid, block, st, muse_pf_attn::prefill_post_attn_norm_kernel<false, true, true, true, true>, X, Rs, Wp, Wq, Sc, Ss, static_cast<__nv_bfloat16*>(nullptr), Of, dp, dl));
    }
    if (cs_mode == 0)
      muse_pf_attn::prefill_post_attn_norm_kernel<false, true, false, false, false><<<rows, 256, 0, st>>>(X, Rs, Wp, Wq, Sc, Ss, nullptr, Of, dp, dl);
    else if (cs_mode == 2)
      muse_pf_attn::prefill_post_attn_norm_kernel<false, true, false, true, false><<<rows, 256, 0, st>>>(X, Rs, Wp, Wq, Sc, Ss, nullptr, Of, dp, dl);
    else
      muse_pf_attn::prefill_post_attn_norm_kernel<false, true, true, true, false><<<rows, 256, 0, st>>>(X, Rs, Wp, Wq, Sc, Ss, nullptr, Of, dp, dl);
  } else {
    return 1;
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_prefill_discard_l2(const void* ptr, size_t n_bytes, void* cuda_stream) {
  const char* p = nullptr; size_t n_lines = 0;
  muse_pf_attn::align_discard(ptr, n_bytes, p, n_lines);
  if (!p || n_lines == 0) return 0;
  muse_pf_attn::discard_l2_range_kernel<<<48, 256, 0, static_cast<cudaStream_t>(cuda_stream)>>>(p, n_lines);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_prefill_discard_l2_multi2(
    const void* ptr0, size_t n_bytes0,
    const void* ptr1, size_t n_bytes1, void* cuda_stream) {
  const char* p0 = nullptr; size_t l0 = 0;
  const char* p1 = nullptr; size_t l1 = 0;
  muse_pf_attn::align_discard(ptr0, n_bytes0, p0, l0);
  muse_pf_attn::align_discard(ptr1, n_bytes1, p1, l1);
  if (l0 == 0 && l1 == 0) return 0;
  if (l0 == 0) { p0 = p1; l0 = l1; p1 = nullptr; l1 = 0; }
  muse_pf_attn::discard_l2_multi2_kernel<<<48, 256, 0, static_cast<cudaStream_t>(cuda_stream)>>>(p0, l0, p1, l1);
  return static_cast<int>(cudaGetLastError());
}
