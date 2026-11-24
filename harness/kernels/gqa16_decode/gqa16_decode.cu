// SPDX-License-Identifier: Apache-2.0
// GQA-16 grouped Tensor-Core M=1 decode attention for MuseGlimmer (32 Q heads, 2 KV heads,
// head_dim=128, BF16 Q/K/V, paged KV cache, sliding_window=2048 or full causal attention).
//
// Target: NVIDIA GB10 (sm_121a, 48 SMs).
// Grid (default split8): dim3(8, 2, 1) = 16 CTAs x 128 threads (28.5 KiB static smem),
// fitting 100% inside the 16 idle SMs beside 32-CTA FP8/BF16 output_gate_proj (or 24 free
// SMs beside 24-CTA NV6 GateV48) in a single wave while reading each KV page 1x per split.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace muse_gqa16 {

constexpr int kMaxSplits = 64;
constexpr float kLog2e = 1.44269504088896340736f;
constexpr float kLn2   = 0.69314718055994530942f;

struct __align__(128) SharedStorage {
  uint4 kv[3][32][16];     // 3 x 8 KiB = 24 KiB (128B XOR swizzled: c_phys = c ^ (r & 7))
  uint4 p[16][16];         // 4 KiB (128B XOR swizzled: c_phys = c ^ (m & 7))
  float warp_max[4][16];   // 256 B
  float warp_sum[4][16];   // 256 B
};                         // Total = 29,184 B (28.5 KiB < 32 KiB)

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ float fast_exp2f(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float fast_log2f(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float fast_rcpf(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ void cp_async_16(uint32_t smem_dst, const void* gmem_src) {
  // Plain cp.async.cg without L2::cache_hint -> 0 uniform-register cache descriptors (0 odd-UR risk).
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_dst), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ float ld_cg_f32(const float* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ float4 ld_cg_f32x4(const float* p) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ void st_cg_f32(float* p, float v) {
  asm volatile("st.global.cg.f32 [%0], %1;" :: "l"(p), "f"(v));
}

__device__ __forceinline__ void st_cg_f32x2(float* p, float x, float y) {
  asm volatile("st.global.cg.v2.f32 [%0], {%1, %2};" :: "l"(p), "f"(x), "f"(y));
}

// Load m16k16 FragA from swizzled [16][16] uint4 tile at k-step ks (covering cols 16*ks .. 16*ks+15).
__device__ __forceinline__ void ldsm4_swizzled(
    uint32_t (&a)[4],
    const uint4 tile[16][16],
    int ks,
    int lane) {
  const int m_row = lane & 15;
  const int c_log = (ks << 1) + (lane >> 4);
  const int c_phys = c_log ^ (m_row & 7);
  const uint32_t addr = smem_u32(&tile[m_row][c_phys]);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(addr));
}

// Load n8k16 FragB (K[8*w .. 8*w+7, 16*ks .. 16*ks+15]) from swizzled [32][16] uint4 K tile.
__device__ __forceinline__ void ldsm2_k_swizzled(
    uint32_t (&b)[2],
    const uint4 tile[32][16],
    int warp_id,
    int ks,
    int lane) {
  const int r_local = lane & 7;
  const int sub = (lane >> 3) & 1;
  const int r = (warp_id << 3) + r_local;
  const int c_log = (ks << 1) + sub;
  const int c_phys = c_log ^ r_local;
  const uint32_t addr = smem_u32(&tile[r][c_phys]);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(b[0]), "=r"(b[1])
      : "r"(addr));
}

// Load k16n8 FragB (V[16*ks_local .. +15, 32*w + 8*dt .. +7]) transposed from swizzled [32][16] V tile.
__device__ __forceinline__ void ldsm2_v_trans_swizzled(
    uint32_t (&v)[2],
    const uint4 tile[32][16],
    int warp_id,
    int dt,
    int ks_local,
    int lane) {
  const int r_in_16 = lane & 15;
  const int r = (ks_local << 4) + r_in_16;
  const int c_log = (warp_id << 2) + dt;
  const int c_phys = c_log ^ (r & 7);
  const uint32_t addr = smem_u32(&tile[r][c_phys]);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(v[0]), "=r"(v[1])
      : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
    float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

template <bool IsV>
__device__ __forceinline__ void issue_kv_tile_async(
    uint4 smem_tile[32][16],
    const __nv_bfloat16* __restrict__ kv_ptr,
    const int32_t* __restrict__ block_table,
    int tok_base,
    int k_start,
    int seqlen_k,
    int block_size,
    int kv_head,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    int tid) {
  int phys_blk0 = 0, slot_base0 = 0;
  const bool half0_any = (tok_base < seqlen_k) && (tok_base + 16 > k_start);
  if (half0_any) {
    const int lb0 = (block_size == 16) ? (tok_base >> 4) : (tok_base / block_size);
    slot_base0 = (block_size == 16) ? 0 : (tok_base - lb0 * block_size);
    phys_blk0 = __ldg(block_table + lb0);
  }
  int phys_blk1 = 0, slot_base1 = 0;
  const int tok_hi = tok_base + 16;
  const bool half1_any = (tok_hi < seqlen_k) && (tok_hi + 16 > k_start);
  if (half1_any) {
    const int lb1 = (block_size == 16) ? (tok_hi >> 4) : (tok_hi / block_size);
    slot_base1 = (block_size == 16) ? 0 : (tok_hi - lb1 * block_size);
    phys_blk1 = __ldg(block_table + lb1);
  }

  const int64_t head_v_off = static_cast<int64_t>(kv_head) * stride_kv_head + (IsV ? 128 : 0);
  const bool full_valid = (tok_base >= k_start) && (tok_base + 32 <= seqlen_k);

  if (full_valid) {
    const int64_t base0 = static_cast<int64_t>(phys_blk0) * stride_block +
                          static_cast<int64_t>(slot_base0) * stride_slot + head_v_off;
    const int64_t base1 = static_cast<int64_t>(phys_blk1) * stride_block +
                          static_cast<int64_t>(slot_base1) * stride_slot + head_v_off;
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int idx = (u << 7) + tid;
      const int r = idx >> 4;
      const int c = idx & 15;
      const int64_t row_off = (u < 2)
          ? (base0 + static_cast<int64_t>(r) * stride_slot)
          : (base1 + static_cast<int64_t>(r - 16) * stride_slot);
      cp_async_16(smem_u32(&smem_tile[r][c ^ (r & 7)]), kv_ptr + row_off + (c << 3));
    }
  } else {
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int idx = (u << 7) + tid;
      const int r = idx >> 4;
      const int c = idx & 15;
      const int tok = tok_base + r;
      if (tok >= k_start && tok < seqlen_k) {
        const int phys_blk = (u < 2) ? phys_blk0 : phys_blk1;
        const int slot = (u < 2) ? (slot_base0 + r) : (slot_base1 + (r - 16));
        const int64_t row_off = static_cast<int64_t>(phys_blk) * stride_block +
                                static_cast<int64_t>(slot) * stride_slot + head_v_off;
        cp_async_16(smem_u32(&smem_tile[r][c ^ (r & 7)]), kv_ptr + row_off + (c << 3));
      } else {
        smem_tile[r][c ^ (r & 7)] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  }
}

__device__ __forceinline__ void load_q_group_to_regs(
    uint32_t (&q_frag)[8][4],
    SharedStorage& smem,
    const __nv_bfloat16* __restrict__ q_ptr,
    int kv_head,
    int64_t q_head_stride,
    int tid,
    int lane) {
  const __nv_bfloat16* q_group = q_ptr + static_cast<int64_t>(kv_head * 16) * q_head_stride;
  #pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int idx = (u << 7) + tid;
    const int m = idx >> 4;
    const int c = idx & 15;
    smem.kv[0][m][c ^ (m & 7)] =
        *reinterpret_cast<const uint4*>(q_group + static_cast<int64_t>(m) * q_head_stride + (c << 3));
  }
  __syncthreads();
  #pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    ldsm4_swizzled(q_frag[ks], smem.kv[0], ks, lane);
  }
  __syncthreads();
}

template <bool Base2LSE>
__device__ __forceinline__ void combine_splits_epilogue(
    __nv_bfloat16* __restrict__ out_ptr,
    const float* __restrict__ o_partial,
    const float* __restrict__ lse_partial,
    int32_t* __restrict__ split_cnt,
    int kv_head,
    int grid_splits,
    int active_splits,
    int64_t out_head_stride,
    int tid) {
  __threadfence();
  __syncthreads();
  int is_last = 0;
  if (tid == 0) {
    const int prev = atomicAdd(&split_cnt[kv_head], 1);
    if (prev == grid_splits - 1) {
      atomicExch(&split_cnt[kv_head], 0);
      is_last = 1;
    }
  }
  if (!__syncthreads_or(is_last)) return;

  const int q_local = tid >> 3;
  const int d_base = (tid & 7) << 4;
  const float* lse_head = lse_partial + kv_head * kMaxSplits * 16;
  const float* o_head   = o_partial   + kv_head * kMaxSplits * 16 * 128;

  float lse_max = -INFINITY;
  for (int s = 0; s < active_splits; ++s) {
    const float lse_s = ld_cg_f32(lse_head + s * 16 + q_local);
    lse_max = fmaxf(lse_max, lse_s);
  }

  if (lse_max == -INFINITY) {
    __nv_bfloat16* dst = out_ptr + static_cast<int64_t>(kv_head * 16 + q_local) * out_head_stride + d_base;
    *reinterpret_cast<uint4*>(dst)     = make_uint4(0u, 0u, 0u, 0u);
    *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0u, 0u, 0u, 0u);
    return;
  }

  float sum_exp = 0.0f;
  for (int s = 0; s < active_splits; ++s) {
    const float lse_s = ld_cg_f32(lse_head + s * 16 + q_local);
    if (lse_s != -INFINITY) {
      sum_exp += Base2LSE ? fast_exp2f(lse_s - lse_max)
                          : fast_exp2f((lse_s - lse_max) * kLog2e);
    }
  }

  float acc[16] = {0.f};
  if constexpr (Base2LSE) {
    const float inv_sum = (sum_exp == 0.0f) ? 0.0f : (1.0f / sum_exp);
    for (int s = 0; s < active_splits; ++s) {
      const float lse_s = ld_cg_f32(lse_head + s * 16 + q_local);
      if (lse_s == -INFINITY) continue;
      const float w = fast_exp2f(lse_s - lse_max) * inv_sum;
      const float* src = o_head + (s * 16 + q_local) * 128 + d_base;
      const float4 v0 = ld_cg_f32x4(src + 0);
      const float4 v1 = ld_cg_f32x4(src + 4);
      const float4 v2 = ld_cg_f32x4(src + 8);
      const float4 v3 = ld_cg_f32x4(src + 12);
      acc[0]  = __fmaf_rn(w, v0.x, acc[0]);  acc[1]  = __fmaf_rn(w, v0.y, acc[1]);
      acc[2]  = __fmaf_rn(w, v0.z, acc[2]);  acc[3]  = __fmaf_rn(w, v0.w, acc[3]);
      acc[4]  = __fmaf_rn(w, v1.x, acc[4]);  acc[5]  = __fmaf_rn(w, v1.y, acc[5]);
      acc[6]  = __fmaf_rn(w, v1.z, acc[6]);  acc[7]  = __fmaf_rn(w, v1.w, acc[7]);
      acc[8]  = __fmaf_rn(w, v2.x, acc[8]);  acc[9]  = __fmaf_rn(w, v2.y, acc[9]);
      acc[10] = __fmaf_rn(w, v2.z, acc[10]); acc[11] = __fmaf_rn(w, v2.w, acc[11]);
      acc[12] = __fmaf_rn(w, v3.x, acc[12]); acc[13] = __fmaf_rn(w, v3.y, acc[13]);
      acc[14] = __fmaf_rn(w, v3.z, acc[14]); acc[15] = __fmaf_rn(w, v3.w, acc[15]);
    }
  } else {
    const float lse_logsum = (sum_exp == 0.0f || sum_exp != sum_exp)
        ? INFINITY
        : (fast_log2f(sum_exp) * kLn2 + lse_max);
    for (int s = 0; s < active_splits; ++s) {
      const float lse_s = ld_cg_f32(lse_head + s * 16 + q_local);
      if (lse_s == -INFINITY || lse_logsum == INFINITY) continue;
      const float w = fast_exp2f((lse_s - lse_logsum) * kLog2e);
      const float* src = o_head + (s * 16 + q_local) * 128 + d_base;
      const float4 v0 = ld_cg_f32x4(src + 0);
      const float4 v1 = ld_cg_f32x4(src + 4);
      const float4 v2 = ld_cg_f32x4(src + 8);
      const float4 v3 = ld_cg_f32x4(src + 12);
      acc[0]  = __fmaf_rn(w, v0.x, acc[0]);  acc[1]  = __fmaf_rn(w, v0.y, acc[1]);
      acc[2]  = __fmaf_rn(w, v0.z, acc[2]);  acc[3]  = __fmaf_rn(w, v0.w, acc[3]);
      acc[4]  = __fmaf_rn(w, v1.x, acc[4]);  acc[5]  = __fmaf_rn(w, v1.y, acc[5]);
      acc[6]  = __fmaf_rn(w, v1.z, acc[6]);  acc[7]  = __fmaf_rn(w, v1.w, acc[7]);
      acc[8]  = __fmaf_rn(w, v2.x, acc[8]);  acc[9]  = __fmaf_rn(w, v2.y, acc[9]);
      acc[10] = __fmaf_rn(w, v2.z, acc[10]); acc[11] = __fmaf_rn(w, v2.w, acc[11]);
      acc[12] = __fmaf_rn(w, v3.x, acc[12]); acc[13] = __fmaf_rn(w, v3.y, acc[13]);
      acc[14] = __fmaf_rn(w, v3.z, acc[14]); acc[15] = __fmaf_rn(w, v3.w, acc[15]);
    }
  }

  uint32_t packed[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    __nv_bfloat162 b2 = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
    packed[i] = *reinterpret_cast<const uint32_t*>(&b2);
  }
  __nv_bfloat16* dst = out_ptr + static_cast<int64_t>(kv_head * 16 + q_local) * out_head_stride + d_base;
  *reinterpret_cast<uint4*>(dst)     = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  *reinterpret_cast<uint4*>(dst + 8) = make_uint4(packed[4], packed[5], packed[6], packed[7]);
}

__global__ __launch_bounds__(128, 2) void gqa16_split_chunk32_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ kv_ptr,
    __nv_bfloat16* __restrict__ out_ptr,
    float* __restrict__ o_partial,
    float* __restrict__ lse_partial,
    int32_t* __restrict__ split_cnt,
    const int32_t* __restrict__ seqused_k_ptr,
    const int32_t* __restrict__ block_table,
    int block_size,
    int window_left,
    float scale_log2,
    int64_t q_head_stride,
    int64_t out_head_stride,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head) {
  __shared__ SharedStorage smem;

  const int split_idx   = blockIdx.x;
  const int grid_splits = gridDim.x;
  const int kv_head     = blockIdx.y;
  const int tid         = threadIdx.x;
  const int warp_id     = tid >> 5;
  const int lane        = tid & 31;
  const int group_id    = lane >> 2;
  const int tid_in_grp  = lane & 3;

  const int seqlen_k = seqused_k_ptr[0];
  const int k_start = (window_left >= 0)
      ? ((seqlen_k - 1 - window_left > 0) ? (seqlen_k - 1 - window_left) : 0)
      : 0;
  const int first_tile  = (seqlen_k > 0) ? (k_start >> 5) : 0;
  const int end_tile    = (seqlen_k > 0) ? ((seqlen_k + 31) >> 5) : 0;
  const int total_tiles = (end_tile > first_tile) ? (end_tile - first_tile) : 0;

  const int base_t   = total_tiles / grid_splits;
  const int rem_t    = total_tiles - base_t * grid_splits;
  const int my_count = base_t + (split_idx < rem_t ? 1 : 0);
  const int my_start = first_tile + split_idx * base_t + (split_idx < rem_t ? split_idx : rem_t);
  const int my_end   = my_start + my_count;

  if (my_count == 0) {
    if (grid_splits == 1) {
      const int q_local = tid >> 3;
      const int d_base = (tid & 7) << 4;
      __nv_bfloat16* dst = out_ptr + static_cast<int64_t>(kv_head * 16 + q_local) * out_head_stride + d_base;
      *reinterpret_cast<uint4*>(dst)     = make_uint4(0u, 0u, 0u, 0u);
      *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0u, 0u, 0u, 0u);
      return;
    }
    if (tid < 16) {
      st_cg_f32(lse_partial + (kv_head * kMaxSplits + split_idx) * 16 + tid, -INFINITY);
    }
    combine_splits_epilogue<true>(
        out_ptr, o_partial, lse_partial, split_cnt,
        kv_head, grid_splits, grid_splits, out_head_stride, tid);
    return;
  }

  uint32_t q_frag[8][4];
  load_q_group_to_regs(q_frag, smem, q_ptr, kv_head, q_head_stride, tid, lane);

  float o_frag[4][4] = {0.f};
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  const int tok0 = (my_end - 1) << 5;
  issue_kv_tile_async<false>(smem.kv[0], kv_ptr, block_table, tok0, k_start, seqlen_k,
                             block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
  cp_async_commit();
  issue_kv_tile_async<true>(smem.kv[1], kv_ptr, block_table, tok0, k_start, seqlen_k,
                            block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
  cp_async_commit();

  for (int it = 0; it < my_count; ++it) {
    const int tile_cur     = (my_end - 1) - it;
    const int tok_cur      = tile_cur << 5;
    const int ks_slot      = (2 * it) % 3;
    const int vs_slot      = (2 * it + 1) % 3;
    const int next_ks_slot = (2 * it + 2) % 3;

    cp_async_wait<1>();
    __syncthreads();

    if (it + 1 < my_count) {
      const int tok_next = (tile_cur - 1) << 5;
      issue_kv_tile_async<false>(smem.kv[next_ks_slot], kv_ptr, block_table, tok_next, k_start, seqlen_k,
                                 block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
    }

    float s_frag[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t b_frag[2];
      ldsm2_k_swizzled(b_frag, smem.kv[ks_slot], warp_id, ks, lane);
      mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag);
    }

    const int col0 = tok_cur + (warp_id << 3) + (tid_in_grp << 1);
    const int col1 = col0 + 1;
    if (col0 < k_start || col0 >= seqlen_k) { s_frag[0] = -INFINITY; s_frag[2] = -INFINITY; }
    if (col1 < k_start || col1 >= seqlen_k) { s_frag[1] = -INFINITY; s_frag[3] = -INFINITY; }

    float m0 = fmaxf(s_frag[0], s_frag[1]);
    float m1 = fmaxf(s_frag[2], s_frag[3]);
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    if (tid_in_grp == 0) {
      smem.warp_max[warp_id][group_id]     = m0;
      smem.warp_max[warp_id][group_id + 8] = m1;
    }

    if (it + 1 < my_count) {
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    if (it + 1 < my_count) {
      const int tok_next = (tile_cur - 1) << 5;
      issue_kv_tile_async<true>(smem.kv[ks_slot], kv_ptr, block_table, tok_next, k_start, seqlen_k,
                                block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
    }

    const float tile_max0 = fmaxf(fmaxf(smem.warp_max[0][group_id], smem.warp_max[1][group_id]),
                                  fmaxf(smem.warp_max[2][group_id], smem.warp_max[3][group_id]));
    const float tile_max1 = fmaxf(fmaxf(smem.warp_max[0][group_id + 8], smem.warp_max[1][group_id + 8]),
                                  fmaxf(smem.warp_max[2][group_id + 8], smem.warp_max[3][group_id + 8]));

    if (it == 0) {
      row_max0 = tile_max0;
      row_max1 = tile_max1;
    } else {
      const float prev0 = row_max0, prev1 = row_max1;
      row_max0 = fmaxf(prev0, tile_max0);
      row_max1 = fmaxf(prev1, tile_max1);
      const float cur0 = (row_max0 == -INFINITY) ? 0.0f : row_max0;
      const float cur1 = (row_max1 == -INFINITY) ? 0.0f : row_max1;
      const float sc0 = (prev0 == -INFINITY) ? 0.0f : fast_exp2f((prev0 - cur0) * scale_log2);
      const float sc1 = (prev1 == -INFINITY) ? 0.0f : fast_exp2f((prev1 - cur1) * scale_log2);
      row_sum0 *= sc0;
      row_sum1 *= sc1;
      #pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        o_frag[dt][0] *= sc0; o_frag[dt][1] *= sc0;
        o_frag[dt][2] *= sc1; o_frag[dt][3] *= sc1;
      }
    }

    const float max_sc0 = (row_max0 == -INFINITY) ? 0.0f : (row_max0 * scale_log2);
    const float max_sc1 = (row_max1 == -INFINITY) ? 0.0f : (row_max1 * scale_log2);
    const float p0 = (s_frag[0] == -INFINITY) ? 0.0f : fast_exp2f(__fmaf_rn(s_frag[0], scale_log2, -max_sc0));
    const float p1 = (s_frag[1] == -INFINITY) ? 0.0f : fast_exp2f(__fmaf_rn(s_frag[1], scale_log2, -max_sc0));
    const float p2 = (s_frag[2] == -INFINITY) ? 0.0f : fast_exp2f(__fmaf_rn(s_frag[2], scale_log2, -max_sc1));
    const float p3 = (s_frag[3] == -INFINITY) ? 0.0f : fast_exp2f(__fmaf_rn(s_frag[3], scale_log2, -max_sc1));
    row_sum0 = (row_sum0 + p0) + p1;
    row_sum1 = (row_sum1 + p2) + p3;

    const __nv_bfloat162 pb0 = __floats2bfloat162_rn(p0, p1);
    const __nv_bfloat162 pb1 = __floats2bfloat162_rn(p2, p3);
    const int c_phys_p = warp_id ^ group_id;
    reinterpret_cast<uint32_t*>(&smem.p[group_id][c_phys_p])[tid_in_grp] =
        *reinterpret_cast<const uint32_t*>(&pb0);
    reinterpret_cast<uint32_t*>(&smem.p[group_id + 8][c_phys_p])[tid_in_grp] =
        *reinterpret_cast<const uint32_t*>(&pb1);
    __syncthreads();

    #pragma unroll
    for (int ks_p = 0; ks_p < 2; ++ks_p) {
      uint32_t p_frag[4];
      ldsm4_swizzled(p_frag, smem.p, ks_p, lane);
      #pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        uint32_t v_frag[2];
        ldsm2_v_trans_swizzled(v_frag, smem.kv[vs_slot], warp_id, dt, ks_p, lane);
        mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
      }
    }
  }

  row_sum0 += __shfl_xor_sync(0xffffffffu, row_sum0, 2);
  row_sum0 += __shfl_xor_sync(0xffffffffu, row_sum0, 1);
  row_sum1 += __shfl_xor_sync(0xffffffffu, row_sum1, 2);
  row_sum1 += __shfl_xor_sync(0xffffffffu, row_sum1, 1);
  if (tid_in_grp == 0) {
    smem.warp_sum[warp_id][group_id]     = row_sum0;
    smem.warp_sum[warp_id][group_id + 8] = row_sum1;
  }
  __syncthreads();

  const float tot0 = (smem.warp_sum[0][group_id] + smem.warp_sum[1][group_id]) +
                     (smem.warp_sum[2][group_id] + smem.warp_sum[3][group_id]);
  const float tot1 = (smem.warp_sum[0][group_id + 8] + smem.warp_sum[1][group_id + 8]) +
                     (smem.warp_sum[2][group_id + 8] + smem.warp_sum[3][group_id + 8]);
  const bool empty0 = (tot0 == 0.0f || tot0 != tot0 || row_max0 == -INFINITY);
  const bool empty1 = (tot1 == 0.0f || tot1 != tot1 || row_max1 == -INFINITY);
  const float inv0 = empty0 ? 0.0f : (1.0f / tot0);
  const float inv1 = empty1 ? 0.0f : (1.0f / tot1);

  if (grid_splits == 1) {
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      const int d_col = (warp_id << 5) + (dt << 3) + (tid_in_grp << 1);
      const __nv_bfloat162 ob0 = __floats2bfloat162_rn(o_frag[dt][0] * inv0, o_frag[dt][1] * inv0);
      const __nv_bfloat162 ob1 = __floats2bfloat162_rn(o_frag[dt][2] * inv1, o_frag[dt][3] * inv1);
      *reinterpret_cast<uint32_t*>(
          out_ptr + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob0);
      *reinterpret_cast<uint32_t*>(
          out_ptr + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob1);
    }
    return;
  }

  float* o_slot = o_partial + (kv_head * kMaxSplits + split_idx) * 16 * 128;
  #pragma unroll
  for (int dt = 0; dt < 4; ++dt) {
    const int d_col = (warp_id << 5) + (dt << 3) + (tid_in_grp << 1);
    st_cg_f32x2(o_slot + group_id * 128 + d_col, o_frag[dt][0] * inv0, o_frag[dt][1] * inv0);
    st_cg_f32x2(o_slot + (group_id + 8) * 128 + d_col, o_frag[dt][2] * inv1, o_frag[dt][3] * inv1);
  }
  if (warp_id == 0 && tid_in_grp == 0) {
    const float lse0 = empty0 ? -INFINITY : __fmaf_rn(row_max0, scale_log2, fast_log2f(tot0));
    const float lse1 = empty1 ? -INFINITY : __fmaf_rn(row_max1, scale_log2, fast_log2f(tot1));
    float* lse_slot = lse_partial + (kv_head * kMaxSplits + split_idx) * 16;
    st_cg_f32(lse_slot + group_id,     lse0);
    st_cg_f32(lse_slot + group_id + 8, lse1);
  }

  combine_splits_epilogue<true>(
      out_ptr, o_partial, lse_partial, split_cnt,
      kv_head, grid_splits, grid_splits, out_head_stride, tid);
}

template <bool UseFastRcp>
__global__ __launch_bounds__(128, 2) void gqa16_fa2_block128_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ kv_ptr,
    __nv_bfloat16* __restrict__ out_ptr,
    float* __restrict__ o_partial,
    float* __restrict__ lse_partial,
    int32_t* __restrict__ split_cnt,
    const int32_t* __restrict__ seqused_k_ptr,
    const int32_t* __restrict__ block_table,
    int block_size,
    int window_left,
    int fa2_splits,
    float softmax_scale,
    float scale_log2,
    int64_t q_head_stride,
    int64_t out_head_stride,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head) {
  __shared__ SharedStorage smem;

  const int worker_idx  = blockIdx.x;
  const int grid_splits = gridDim.x;
  const int kv_head     = blockIdx.y;
  const int tid         = threadIdx.x;
  const int warp_id     = tid >> 5;
  const int lane        = tid & 31;
  const int group_id    = lane >> 2;
  const int tid_in_grp  = lane & 3;

  const int seqlen_k = seqused_k_ptr[0];
  const int k_start = (window_left >= 0)
      ? ((seqlen_k - 1 - window_left > 0) ? (seqlen_k - 1 - window_left) : 0)
      : 0;
  const int num_n_blocks = (seqlen_k > 0) ? ((seqlen_k + 127) >> 7) : 0;
  const int n_blocks_per_split = (num_n_blocks + fa2_splits - 1) / fa2_splits;
  const int global_nb_min = (window_left >= 0 && seqlen_k > 0) ? (k_start >> 7) : 0;

  uint32_t q_frag[8][4];
  load_q_group_to_regs(q_frag, smem, q_ptr, kv_head, q_head_stride, tid, lane);

  for (int s_slot = worker_idx; s_slot < fa2_splits; s_slot += grid_splits) {
    int nb_min = (fa2_splits == 1) ? global_nb_min : (s_slot * n_blocks_per_split);
    int nb_max = (fa2_splits == 1) ? num_n_blocks
                                   : ((s_slot + 1) * n_blocks_per_split < num_n_blocks
                                          ? (s_slot + 1) * n_blocks_per_split
                                          : num_n_blocks);
    if (nb_min < global_nb_min) nb_min = global_nb_min;

    if (nb_min >= nb_max || n_blocks_per_split == 0) {
      if (fa2_splits == 1) {
        const int q_local = tid >> 3;
        const int d_base = (tid & 7) << 4;
        __nv_bfloat16* dst = out_ptr + static_cast<int64_t>(kv_head * 16 + q_local) * out_head_stride + d_base;
        *reinterpret_cast<uint4*>(dst)     = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0u, 0u, 0u, 0u);
        return;
      }
      if (tid < 16) {
        st_cg_f32(lse_partial + (kv_head * kMaxSplits + s_slot) * 16 + tid, -INFINITY);
      }
      __syncthreads();
      continue;
    }

    float o_frag[4][4] = {0.f};
    float row_max0 = -INFINITY, row_max1 = -INFINITY;
    float row_sum0 = 0.0f, row_sum1 = 0.0f;

    for (int nb = nb_max - 1; nb >= nb_min; --nb) {
      const int block_tok_base = nb << 7;
      float s_frag[4][4] = {0.f};

      issue_kv_tile_async<false>(smem.kv[0], kv_ptr, block_table, block_tok_base, k_start, seqlen_k,
                                 block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
      #pragma unroll
      for (int ch = 0; ch < 4; ++ch) {
        if (ch + 1 < 4) {
          issue_kv_tile_async<false>(smem.kv[(ch + 1) & 1], kv_ptr, block_table,
                                     block_tok_base + ((ch + 1) << 5), k_start, seqlen_k,
                                     block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
          cp_async_commit();
          cp_async_wait<1>();
        } else {
          cp_async_wait<0>();
        }
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          uint32_t b_frag[2];
          ldsm2_k_swizzled(b_frag, smem.kv[ch & 1], warp_id, ks, lane);
          mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[ch]);
        }
        __syncthreads();
      }

      issue_kv_tile_async<true>(smem.kv[0], kv_ptr, block_table, block_tok_base, k_start, seqlen_k,
                                block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();

      float m0 = -INFINITY, m1 = -INFINITY;
      #pragma unroll
      for (int ch = 0; ch < 4; ++ch) {
        const int col0 = block_tok_base + (ch << 5) + (warp_id << 3) + (tid_in_grp << 1);
        const int col1 = col0 + 1;
        if (col0 < k_start || col0 >= seqlen_k) { s_frag[ch][0] = -INFINITY; s_frag[ch][2] = -INFINITY; }
        if (col1 < k_start || col1 >= seqlen_k) { s_frag[ch][1] = -INFINITY; s_frag[ch][3] = -INFINITY; }
        m0 = fmaxf(m0, fmaxf(s_frag[ch][0], s_frag[ch][1]));
        m1 = fmaxf(m1, fmaxf(s_frag[ch][2], s_frag[ch][3]));
      }
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
      m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
      m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
      if (tid_in_grp == 0) {
        smem.warp_max[warp_id][group_id]     = m0;
        smem.warp_max[warp_id][group_id + 8] = m1;
      }
      __syncthreads();

      const float tile_max0 = fmaxf(fmaxf(smem.warp_max[0][group_id], smem.warp_max[1][group_id]),
                                    fmaxf(smem.warp_max[2][group_id], smem.warp_max[3][group_id]));
      const float tile_max1 = fmaxf(fmaxf(smem.warp_max[0][group_id + 8], smem.warp_max[1][group_id + 8]),
                                    fmaxf(smem.warp_max[2][group_id + 8], smem.warp_max[3][group_id + 8]));
      if (nb == nb_max - 1) {
        row_max0 = tile_max0;
        row_max1 = tile_max1;
      } else {
        const float prev0 = row_max0, prev1 = row_max1;
        row_max0 = fmaxf(prev0, tile_max0);
        row_max1 = fmaxf(prev1, tile_max1);
        const float cur0 = (row_max0 == -INFINITY) ? 0.0f : row_max0;
        const float cur1 = (row_max1 == -INFINITY) ? 0.0f : row_max1;
        const float sc0 = (prev0 == -INFINITY) ? 0.0f : fast_exp2f((prev0 - cur0) * scale_log2);
        const float sc1 = (prev1 == -INFINITY) ? 0.0f : fast_exp2f((prev1 - cur1) * scale_log2);
        row_sum0 *= sc0;
        row_sum1 *= sc1;
        #pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
          o_frag[dt][0] *= sc0; o_frag[dt][1] *= sc0;
          o_frag[dt][2] *= sc1; o_frag[dt][3] *= sc1;
        }
      }

      const float max_sc0 = (row_max0 == -INFINITY) ? 0.0f : (row_max0 * scale_log2);
      const float max_sc1 = (row_max1 == -INFINITY) ? 0.0f : (row_max1 * scale_log2);
      // Reuse unused smem.kv[2] (8 KiB) to stage FP32 p[16][16][4][2] for exact FA2 1-warp sum order
      float (*p_f32)[16][4][2] = reinterpret_cast<float (*)[16][4][2]>(smem.kv[2]);
      #pragma unroll
      for (int ch = 0; ch < 4; ++ch) {
        const float p0 = (s_frag[ch][0] == -INFINITY) ? 0.0f : fast_exp2f(__fmaf_rn(s_frag[ch][0], scale_log2, -max_sc0));
        const float p1 = (s_frag[ch][1] == -INFINITY) ? 0.0f : fast_exp2f(__fmaf_rn(s_frag[ch][1], scale_log2, -max_sc0));
        const float p2 = (s_frag[ch][2] == -INFINITY) ? 0.0f : fast_exp2f(__fmaf_rn(s_frag[ch][2], scale_log2, -max_sc1));
        const float p3 = (s_frag[ch][3] == -INFINITY) ? 0.0f : fast_exp2f(__fmaf_rn(s_frag[ch][3], scale_log2, -max_sc1));
        const int j = (ch << 2) + warp_id;
        p_f32[group_id][j][tid_in_grp][0] = p0;
        p_f32[group_id][j][tid_in_grp][1] = p1;
        p_f32[group_id + 8][j][tid_in_grp][0] = p2;
        p_f32[group_id + 8][j][tid_in_grp][1] = p3;
        const __nv_bfloat162 pb0 = __floats2bfloat162_rn(p0, p1);
        const __nv_bfloat162 pb1 = __floats2bfloat162_rn(p2, p3);
        const int c_phys_p = j ^ group_id;
        reinterpret_cast<uint32_t*>(&smem.p[group_id][c_phys_p])[tid_in_grp] =
            *reinterpret_cast<const uint32_t*>(&pb0);
        reinterpret_cast<uint32_t*>(&smem.p[group_id + 8][c_phys_p])[tid_in_grp] =
            *reinterpret_cast<const uint32_t*>(&pb1);
      }
      __syncthreads();
      if (warp_id == 0) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
          row_sum0 = (row_sum0 + p_f32[group_id][j][tid_in_grp][0]) + p_f32[group_id][j][tid_in_grp][1];
          row_sum1 = (row_sum1 + p_f32[group_id + 8][j][tid_in_grp][0]) + p_f32[group_id + 8][j][tid_in_grp][1];
        }
      }

      #pragma unroll
      for (int ch = 0; ch < 4; ++ch) {
        if (ch + 1 < 4) {
          issue_kv_tile_async<true>(smem.kv[(ch + 1) & 1], kv_ptr, block_table,
                                    block_tok_base + ((ch + 1) << 5), k_start, seqlen_k,
                                    block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
          cp_async_commit();
          cp_async_wait<1>();
        } else {
          cp_async_wait<0>();
        }
        __syncthreads();
        #pragma unroll
        for (int ks_p = 0; ks_p < 2; ++ks_p) {
          uint32_t p_frag[4];
          ldsm4_swizzled(p_frag, smem.p, (ch << 1) + ks_p, lane);
          #pragma unroll
          for (int dt = 0; dt < 4; ++dt) {
            uint32_t v_frag[2];
            ldsm2_v_trans_swizzled(v_frag, smem.kv[ch & 1], warp_id, dt, ks_p, lane);
            mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
          }
        }
        __syncthreads();
      }
    }

    if (warp_id == 0) {
      row_sum0 += __shfl_xor_sync(0xffffffffu, row_sum0, 2);
      row_sum0 += __shfl_xor_sync(0xffffffffu, row_sum0, 1);
      row_sum1 += __shfl_xor_sync(0xffffffffu, row_sum1, 2);
      row_sum1 += __shfl_xor_sync(0xffffffffu, row_sum1, 1);
      if (tid_in_grp == 0) {
        smem.warp_sum[0][group_id]     = row_sum0;
        smem.warp_sum[0][group_id + 8] = row_sum1;
      }
    }
    __syncthreads();

    const float tot0 = smem.warp_sum[0][group_id];
    const float tot1 = smem.warp_sum[0][group_id + 8];
    const bool empty0 = (tot0 == 0.0f || tot0 != tot0 || row_max0 == -INFINITY);
    const bool empty1 = (tot1 == 0.0f || tot1 != tot1 || row_max1 == -INFINITY);
    const float inv0 = empty0 ? 0.0f : (UseFastRcp ? fast_rcpf(tot0) : (1.0f / tot0));
    const float inv1 = empty1 ? 0.0f : (UseFastRcp ? fast_rcpf(tot1) : (1.0f / tot1));

    if (fa2_splits == 1) {
      #pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        const int d_col = (warp_id << 5) + (dt << 3) + (tid_in_grp << 1);
        const __nv_bfloat162 ob0 = __floats2bfloat162_rn(o_frag[dt][0] * inv0, o_frag[dt][1] * inv0);
        const __nv_bfloat162 ob1 = __floats2bfloat162_rn(o_frag[dt][2] * inv1, o_frag[dt][3] * inv1);
        *reinterpret_cast<uint32_t*>(
            out_ptr + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) =
            *reinterpret_cast<const uint32_t*>(&ob0);
        *reinterpret_cast<uint32_t*>(
            out_ptr + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) =
            *reinterpret_cast<const uint32_t*>(&ob1);
      }
      return;
    }

    float* o_slot = o_partial + (kv_head * kMaxSplits + s_slot) * 16 * 128;
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      const int d_col = (warp_id << 5) + (dt << 3) + (tid_in_grp << 1);
      st_cg_f32x2(o_slot + group_id * 128 + d_col, o_frag[dt][0] * inv0, o_frag[dt][1] * inv0);
      st_cg_f32x2(o_slot + (group_id + 8) * 128 + d_col, o_frag[dt][2] * inv1, o_frag[dt][3] * inv1);
    }
    if (warp_id == 0 && tid_in_grp == 0) {
      const float lse0 = empty0 ? -INFINITY : __fmaf_rn(row_max0, softmax_scale, fast_log2f(tot0) * kLn2);
      const float lse1 = empty1 ? -INFINITY : __fmaf_rn(row_max1, softmax_scale, fast_log2f(tot1) * kLn2);
      float* lse_slot = lse_partial + (kv_head * kMaxSplits + s_slot) * 16;
      st_cg_f32(lse_slot + group_id,     lse0);
      st_cg_f32(lse_slot + group_id + 8, lse1);
    }
    __syncthreads();
  }

  if (fa2_splits > 1) {
    combine_splits_epilogue<false>(
        out_ptr, o_partial, lse_partial, split_cnt,
        kv_head, grid_splits, fa2_splits, out_head_stride, tid);
  }
}

}  // namespace muse_gqa16

extern "C" int muse_gqa16_decode(
    const void* q_ptr,
    const void* kv_ptr,
    void* out_ptr,
    void* o_partial_ptr,
    void* lse_partial_ptr,
    void* split_cnt_ptr,
    const void* seqused_k_ptr,
    const void* block_table_ptr,
    int block_size,
    int window_left,
    int grid_splits,
    int fa2_splits,
    int mode,
    float softmax_scale,
    int64_t q_head_stride,
    int64_t out_head_stride,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    void* cuda_stream) {
  if (grid_splits < 1 || grid_splits > muse_gqa16::kMaxSplits ||
      fa2_splits < 1 || fa2_splits > muse_gqa16::kMaxSplits ||
      block_size < 16 || (block_size & 15) != 0) {
    return 1;
  }
  const float scale_log2 = static_cast<float>(static_cast<double>(softmax_scale) * 1.44269504088896340736);
  const dim3 grid(static_cast<unsigned>(grid_splits), 2u, 1u);
  const dim3 block(128u, 1u, 1u);
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

  if (mode == 0) {
    muse_gqa16::gqa16_split_chunk32_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q_ptr),
        static_cast<const __nv_bfloat16*>(kv_ptr),
        static_cast<__nv_bfloat16*>(out_ptr),
        static_cast<float*>(o_partial_ptr),
        static_cast<float*>(lse_partial_ptr),
        static_cast<int32_t*>(split_cnt_ptr),
        static_cast<const int32_t*>(seqused_k_ptr),
        static_cast<const int32_t*>(block_table_ptr),
        block_size, window_left, scale_log2,
        q_head_stride, out_head_stride,
        stride_block, stride_slot, stride_kv_head);
  } else if (mode == 1) {
    muse_gqa16::gqa16_fa2_block128_kernel<true><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q_ptr),
        static_cast<const __nv_bfloat16*>(kv_ptr),
        static_cast<__nv_bfloat16*>(out_ptr),
        static_cast<float*>(o_partial_ptr),
        static_cast<float*>(lse_partial_ptr),
        static_cast<int32_t*>(split_cnt_ptr),
        static_cast<const int32_t*>(seqused_k_ptr),
        static_cast<const int32_t*>(block_table_ptr),
        block_size, window_left, fa2_splits, softmax_scale, scale_log2,
        q_head_stride, out_head_stride,
        stride_block, stride_slot, stride_kv_head);
  } else {
    muse_gqa16::gqa16_fa2_block128_kernel<false><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q_ptr),
        static_cast<const __nv_bfloat16*>(kv_ptr),
        static_cast<__nv_bfloat16*>(out_ptr),
        static_cast<float*>(o_partial_ptr),
        static_cast<float*>(lse_partial_ptr),
        static_cast<int32_t*>(split_cnt_ptr),
        static_cast<const int32_t*>(seqused_k_ptr),
        static_cast<const int32_t*>(block_table_ptr),
        block_size, window_left, fa2_splits, softmax_scale, scale_log2,
        q_head_stride, out_head_stride,
        stride_block, stride_slot, stride_kv_head);
  }
  return static_cast<int>(cudaGetLastError());
}
