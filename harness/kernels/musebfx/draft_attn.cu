// SPDX-License-Identifier: Apache-2.0
// dflash_draft_attn: 1-pass PDL non-causal BF16 Tensor-Core GQA attention kernel (M <= 16, D = 128)
// for the pinned DFlash drafter on SM121 (GB10).
//
// Design:
//   - Grid: (num_heads, 1, 1) — 1 CTA per query head, kv_head = q_head / (num_heads / num_kv_heads).
//     Works for any num_heads, num_kv_heads (num_heads % num_kv_heads == 0), any M in [1..16]
//     (including M=16 and M=8), arbitrary seqlen_k, optional sliding window_left, optional sinks,
//     and arbitrary 4D/5D paged KV cache strides (last dim contiguous, head_dim = 128).
//   - Block: 128 threads (4 warps), __launch_bounds__(128, 2).
//   - Static SMEM: 37,376 B (36.5 KiB < 48 KiB), 4-slot zero-stall kBlockN=64 cp.async ring
//     (kv[0]=K0, kv[1]=K1, kv[2]=V0, kv[3]=V1), 128B XOR swizzle, 0 bank conflicts.
//   - SM121 odd-UR safe: plain cp.async.cg.shared.global ONLY (zero createpolicy / L2::cache_hint).
//   - PDL chain: loads predecessor-independent metadata before griddepcontrol.wait, and emits early
//     griddepcontrol.launch_dependents on the final K-tile before PV(V1) + epilogue stores so
//     dflash_qkv_norm_rope -> dflash_draft_attn -> o_proj forms an unbroken SM121 PDL chain.
//   - 100% CUDA-graph capture safe: reads actual_m from cu_seqlens_q[0..1] and seqlen_k from
//     seqused_k[0] directly on device; zero host syncs; zero dynamic allocations.

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace muse_draft_attn {

constexpr float kLog2e = 1.44269504088896340736f;  // 0x3FB8AA3B

struct __align__(128) DraftAttnSmem {
  uint4 kv[4][32][16];    // 4 x 8 KiB = 32,768 B (0=K0, 1=K1, 2=V0, 3=V1; c_phys = c ^ (r & 7))
  uint4 p[16][16];        // 4,096 B (c_phys = c ^ (m & 7); aliased for prologue Q load)
  float warp_max[4][16];  // 256 B
  float warp_sum[4][16];  // 256 B
};                        // Total = 37,376 B (36.5 KiB < 48 KiB)

static_assert(sizeof(DraftAttnSmem) == 37376, "DraftAttnSmem size mismatch");

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

__device__ __forceinline__ void cp_async_16(uint32_t smem_dst, const void* gmem_src) {
  // Plain cp.async.cg (no createpolicy / L2::cache_hint -> 0 SM121 odd-UR risk).
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_dst), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Load m16k16 FragA from swizzled [16][16] uint4 tile at k-step ks (cols 16*ks .. 16*ks+15).
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
    int64_t stride_head,
    int tid) {
  const bool is_bs64_aligned = (block_size == 64) && ((tok_base & 31) == 0);
  const bool is_bs16_aligned = (block_size == 16) && ((tok_base & 15) == 0);
  int phys_blk0 = 0, slot_base0 = 0;
  const bool half0_any = (tok_base < seqlen_k) && (tok_base + 16 > k_start);
  if (half0_any) {
    const int lb0 = (block_size == 16) ? (tok_base >> 4)
                  : ((block_size == 64) ? (tok_base >> 6) : (tok_base / block_size));
    slot_base0 = (block_size == 16) ? (tok_base & 15)
               : ((block_size == 64) ? (tok_base & 63) : (tok_base - lb0 * block_size));
    phys_blk0 = __ldg(block_table + lb0);
  }
  int phys_blk1 = phys_blk0, slot_base1 = slot_base0 + 16;
  const int tok_hi = tok_base + 16;
  const bool half1_any = (tok_hi < seqlen_k) && (tok_hi + 16 > k_start);
  if (half1_any && !is_bs64_aligned) {
    const int lb1 = (block_size == 16) ? (tok_hi >> 4)
                  : ((block_size == 64) ? (tok_hi >> 6) : (tok_hi / block_size));
    slot_base1 = (block_size == 16) ? (tok_hi & 15)
               : ((block_size == 64) ? (tok_hi & 63) : (tok_hi - lb1 * block_size));
    phys_blk1 = __ldg(block_table + lb1);
  }

  const int64_t head_off = static_cast<int64_t>(kv_head) * stride_head;
  const bool full_valid = (tok_base >= k_start) && (tok_base + 32 <= seqlen_k) &&
                          (is_bs16_aligned || is_bs64_aligned ||
                           (slot_base0 + 16 <= block_size && slot_base1 + 16 <= block_size));

  if (full_valid) {
    const int64_t base0 = static_cast<int64_t>(phys_blk0) * stride_block +
                          static_cast<int64_t>(slot_base0) * stride_slot + head_off;
    if (is_bs64_aligned) {
      #pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int idx = (u << 7) + tid;
        const int r = idx >> 4;
        const int c = idx & 15;
        const int64_t row_off = base0 + static_cast<int64_t>(r) * stride_slot;
        cp_async_16(smem_u32(&smem_tile[r][c ^ (r & 7)]), kv_ptr + row_off + (c << 3));
      }
    } else {
      const int64_t base1 = static_cast<int64_t>(phys_blk1) * stride_block +
                            static_cast<int64_t>(slot_base1) * stride_slot + head_off;
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
    }
  } else {
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int idx = (u << 7) + tid;
      const int r = idx >> 4;
      const int c = idx & 15;
      const int tok = tok_base + r;
      if (tok >= k_start && tok < seqlen_k) {
        const int lb = (block_size == 16) ? (tok >> 4)
                     : ((block_size == 64) ? (tok >> 6) : (tok / block_size));
        const int slot = (block_size == 16) ? (tok & 15)
                       : ((block_size == 64) ? (tok & 63) : (tok - lb * block_size));
        const int phys_blk = __ldg(block_table + lb);
        const int64_t row_off = static_cast<int64_t>(phys_blk) * stride_block +
                                static_cast<int64_t>(slot) * stride_slot + head_off;
        cp_async_16(smem_u32(&smem_tile[r][c ^ (r & 7)]), kv_ptr + row_off + (c << 3));
      } else {
        smem_tile[r][c ^ (r & 7)] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  }
}

__global__ __launch_bounds__(128, 2)
void dflash_draft_attn_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ k_ptr,
    const __nv_bfloat16* __restrict__ v_ptr,
    __nv_bfloat16* __restrict__ out_ptr,
    const int32_t* __restrict__ seqused_k_ptr,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seqlens_q,
    const void* __restrict__ sinks_ptr,
    int sinks_is_bf16,
    int gqa_ratio,
    int block_size,
    int window_left,
    int window_right,
    int max_q,
    int q_rows,
    float softmax_scale,
    int64_t q_tok_stride,
    int64_t q_head_stride,
    int64_t out_tok_stride,
    int64_t out_head_stride,
    int64_t k_stride_block,
    int64_t k_stride_slot,
    int64_t k_stride_head,
    int64_t v_stride_block,
    int64_t v_stride_slot,
    int64_t v_stride_head,
    int pdl_mode) {
  __shared__ DraftAttnSmem smem;

  // Predecessor-independent thread/head setup BEFORE griddepcontrol.wait.
  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int group_id = lane >> 2;
  const int tid_in_grp = lane & 3;
  const int q_head = static_cast<int>(blockIdx.x);
  const int kv_head = q_head / gqa_ratio;
  const float scale_log2 = softmax_scale * kLog2e;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (pdl_mode & 1) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif

  // Load device metadata strictly AFTER griddepcontrol.wait so predecessor-written
  // cu_seqlens_q / seqused_k are guaranteed visible.
  const int q0 = __ldg(cu_seqlens_q);
  const int q1 = __ldg(cu_seqlens_q + 1);
  int actual_m = q1 - q0;
  if (q0 < 0 || q0 >= q_rows || actual_m <= 0) {
    actual_m = 0;
  } else {
    const int rem = q_rows - q0;
    if (actual_m > rem) actual_m = rem;
    if (actual_m > max_q) actual_m = max_q;
    if (actual_m > 16) actual_m = 16;
  }
  const int seqlen_k = __ldg(seqused_k_ptr);
  // FlashAttention-2 non-causal local window convention:
  // Query row m in [0..actual_m-1] aligns to k_center = seqlen_k - actual_m + m,
  // attending to keys in [k_center - window_left, k_center + window_right] inclusive,
  // clipped to [0, seqlen_k).
  const int base_pos = seqlen_k - actual_m;
  const int k_start_cta = (window_left >= 0) ? max(0, base_pos - window_left) : 0;
  const int k_end_cta = (window_right >= 0)
      ? min(seqlen_k, base_pos + max(0, actual_m - 1) + window_right + 1)
      : seqlen_k;
  const int k_inner_lo = (window_left >= 0)
      ? max(0, base_pos + max(0, actual_m - 1) - window_left)
      : 0;
  const int k_inner_hi = (window_right >= 0)
      ? min(seqlen_k, base_pos + window_right + 1)
      : seqlen_k;
  const int k_lo0 = (window_left >= 0) ? max(0, base_pos + group_id - window_left) : 0;
  const int k_lo1 = (window_left >= 0) ? max(0, base_pos + (group_id + 8) - window_left) : 0;
  const int k_hi0 = (window_right >= 0)
      ? min(seqlen_k, base_pos + group_id + window_right + 1)
      : seqlen_k;
  const int k_hi1 = (window_right >= 0)
      ? min(seqlen_k, base_pos + (group_id + 8) + window_right + 1)
      : seqlen_k;
  const int nb_min = (k_end_cta > k_start_cta) ? (k_start_cta >> 6) : 0;
  const int nb_max = (k_end_cta > k_start_cta) ? ((k_end_cta + 63) >> 6) : 0;

  if (actual_m <= 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (pdl_mode & 2) {
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#endif
    return;
  }

  // Empty KV edge case (e.g. warmup dummy capture): zero-fill out rows cleanly.
  if (nb_min >= nb_max) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (pdl_mode & 2) {
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#endif
    const int m0 = group_id;
    const int m1 = group_id + 8;
    const int64_t ob0 = static_cast<int64_t>(q0 + m0) * out_tok_stride +
                        static_cast<int64_t>(q_head) * out_head_stride;
    const int64_t ob1 = static_cast<int64_t>(q0 + m1) * out_tok_stride +
                        static_cast<int64_t>(q_head) * out_head_stride;
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      const int d_col = (warp_id << 5) + (dt << 3) + (tid_in_grp << 1);
      if (m0 < actual_m) *reinterpret_cast<uint32_t*>(out_ptr + ob0 + d_col) = 0u;
      if (m1 < actual_m) *reinterpret_cast<uint32_t*>(out_ptr + ob1 + d_col) = 0u;
    }
    return;
  }

  // Prologue: stage Q[0..actual_m-1, q_head, 0..127] into smem.p[16][16] (zero-filling m >= actual_m),
  // and prefetch K0(nb_min) -> smem.kv[0], K1(nb_min) -> smem.kv[1].
  const int64_t q_head_off = static_cast<int64_t>(q_head) * q_head_stride;
  #pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int idx = (u << 7) + tid;
    const int m = idx >> 4;
    const int c = idx & 15;
    if (m < actual_m) {
      const __nv_bfloat16* src =
          q_ptr + static_cast<int64_t>(q0 + m) * q_tok_stride + q_head_off + (c << 3);
      smem.p[m][c ^ (m & 7)] = *reinterpret_cast<const uint4*>(src);
    } else {
      smem.p[m][c ^ (m & 7)] = make_uint4(0u, 0u, 0u, 0u);
    }
  }

  const int first_tok = nb_min << 6;
  issue_kv_tile_async(
      smem.kv[0], k_ptr, block_table, first_tok, k_start_cta, k_end_cta,
      block_size, kv_head, k_stride_block, k_stride_slot, k_stride_head, tid);
  cp_async_commit();
  issue_kv_tile_async(
      smem.kv[1], k_ptr, block_table, first_tok + 32, k_start_cta, k_end_cta,
      block_size, kv_head, k_stride_block, k_stride_slot, k_stride_head, tid);
  cp_async_commit();
  __syncthreads();

  uint32_t q_frag[8][4];
  #pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    ldsm4_swizzled(q_frag[ks], smem.p, ks, lane);
  }

  float o_frag[4][4];
  #pragma unroll
  for (int dt = 0; dt < 4; ++dt) {
    o_frag[dt][0] = 0.0f;
    o_frag[dt][1] = 0.0f;
    o_frag[dt][2] = 0.0f;
    o_frag[dt][3] = 0.0f;
  }
  float row_max0 = -INFINITY;
  float row_max1 = -INFINITY;
  float row_sum0 = 0.0f;
  float row_sum1 = 0.0f;

  // Main 1-pass FlashAttention loop over 64-token K/V tiles.
  // Entry invariant: in-flight cp.async queue = [K0(nb), K1(nb)] (2 groups);
  // smem.kv[2] and smem.kv[3] are free.
  for (int nb = nb_min; nb < nb_max; ++nb) {
    const int tok_base = nb << 6;

    // Step 1: Wait K0(nb) -> prefetch V0(nb) into smem.kv[2] -> compute QK^T(K0).
    cp_async_wait<1>();
    __syncthreads();
    issue_kv_tile_async(
        smem.kv[2], v_ptr, block_table, tok_base, k_start_cta, k_end_cta,
        block_size, kv_head, v_stride_block, v_stride_slot, v_stride_head, tid);
    cp_async_commit();

    float s_frag[2][4];
    s_frag[0][0] = 0.0f; s_frag[0][1] = 0.0f; s_frag[0][2] = 0.0f; s_frag[0][3] = 0.0f;
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t k_frag[2];
      ldsm2_k_swizzled(k_frag, smem.kv[0], warp_id, ks, lane);
      mma_m16n8k16_bf16(q_frag[ks], k_frag, s_frag[0]);
    }

    // Step 2: Wait K1(nb) -> prefetch V1(nb) into smem.kv[3] -> compute QK^T(K1).
    cp_async_wait<1>();
    __syncthreads();
    issue_kv_tile_async(
        smem.kv[3], v_ptr, block_table, tok_base + 32, k_start_cta, k_end_cta,
        block_size, kv_head, v_stride_block, v_stride_slot, v_stride_head, tid);
    cp_async_commit();

    s_frag[1][0] = 0.0f; s_frag[1][1] = 0.0f; s_frag[1][2] = 0.0f; s_frag[1][3] = 0.0f;
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t k_frag[2];
      ldsm2_k_swizzled(k_frag, smem.kv[1], warp_id, ks, lane);
      mma_m16n8k16_bf16(q_frag[ks], k_frag, s_frag[1]);
    }

    // Step 3: Non-causal per-row sliding-window & boundary masking + online softmax.
    if (tok_base < k_inner_lo || tok_base + 64 > k_inner_hi) {
      #pragma unroll
      for (int ch = 0; ch < 2; ++ch) {
        const int col0 = tok_base + (ch << 5) + (warp_id << 3) + (tid_in_grp << 1);
        const int col1 = col0 + 1;
        if (col0 < k_lo0 || col0 >= k_hi0) s_frag[ch][0] = -INFINITY;
        if (col1 < k_lo0 || col1 >= k_hi0) s_frag[ch][1] = -INFINITY;
        if (col0 < k_lo1 || col0 >= k_hi1) s_frag[ch][2] = -INFINITY;
        if (col1 < k_lo1 || col1 >= k_hi1) s_frag[ch][3] = -INFINITY;
      }
    }

    float m0 = max_ftz_f32(max_ftz_f32(s_frag[0][0], s_frag[0][1]),
                           max_ftz_f32(s_frag[1][0], s_frag[1][1]));
    float m1 = max_ftz_f32(max_ftz_f32(s_frag[0][2], s_frag[0][3]),
                           max_ftz_f32(s_frag[1][2], s_frag[1][3]));
    m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    if (tid_in_grp == 0) {
      smem.warp_max[warp_id][group_id]     = m0;
      smem.warp_max[warp_id][group_id + 8] = m1;
    }
    __syncthreads();

    const float tile_max0 = max_ftz_f32(
        max_ftz_f32(smem.warp_max[0][group_id], smem.warp_max[1][group_id]),
        max_ftz_f32(smem.warp_max[2][group_id], smem.warp_max[3][group_id]));
    const float tile_max1 = max_ftz_f32(
        max_ftz_f32(smem.warp_max[0][group_id + 8], smem.warp_max[1][group_id + 8]),
        max_ftz_f32(smem.warp_max[2][group_id + 8], smem.warp_max[3][group_id + 8]));

    float sc0 = 1.0f;
    float sc1 = 1.0f;
    if (nb == nb_min) {
      row_max0 = tile_max0;
      row_max1 = tile_max1;
    } else {
      const float prev0 = row_max0;
      const float prev1 = row_max1;
      row_max0 = max_ftz_f32(prev0, tile_max0);
      row_max1 = max_ftz_f32(prev1, tile_max1);
      const float cur0 = (row_max0 == -INFINITY) ? 0.0f : row_max0;
      const float cur1 = (row_max1 == -INFINITY) ? 0.0f : row_max1;
      sc0 = (prev0 == -INFINITY) ? 0.0f : fast_exp2f(mul_ftz_f32(scale_log2, sub_ftz_f32(prev0, cur0)));
      sc1 = (prev1 == -INFINITY) ? 0.0f : fast_exp2f(mul_ftz_f32(scale_log2, sub_ftz_f32(prev1, cur1)));
      #pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        o_frag[dt][0] = mul_ftz_f32(o_frag[dt][0], sc0);
        o_frag[dt][1] = mul_ftz_f32(o_frag[dt][1], sc0);
        o_frag[dt][2] = mul_ftz_f32(o_frag[dt][2], sc1);
        o_frag[dt][3] = mul_ftz_f32(o_frag[dt][3], sc1);
      }
    }

    const float max_sc0 = (row_max0 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max0);
    const float max_sc1 = (row_max1 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max1);
    float wsum0 = 0.0f;
    float wsum1 = 0.0f;
    #pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      const float p0 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][0], scale_log2), max_sc0));
      const float p1 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][1], scale_log2), max_sc0));
      const float p2 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][2], scale_log2), max_sc1));
      const float p3 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][3], scale_log2), max_sc1));
      wsum0 = add_ftz_f32(wsum0, add_ftz_f32(p0, p1));
      wsum1 = add_ftz_f32(wsum1, add_ftz_f32(p2, p3));
      const int ni = (ch << 2) + warp_id;
      const __nv_bfloat162 pb0 = __floats2bfloat162_rn(p0, p1);
      const __nv_bfloat162 pb1 = __floats2bfloat162_rn(p2, p3);
      const int c_phys_p = ni ^ group_id;
      reinterpret_cast<uint32_t*>(&smem.p[group_id][c_phys_p])[tid_in_grp] =
          *reinterpret_cast<const uint32_t*>(&pb0);
      reinterpret_cast<uint32_t*>(&smem.p[group_id + 8][c_phys_p])[tid_in_grp] =
          *reinterpret_cast<const uint32_t*>(&pb1);
    }
    wsum0 = add_ftz_f32(wsum0, __shfl_xor_sync(0xffffffffu, wsum0, 2));
    wsum0 = add_ftz_f32(wsum0, __shfl_xor_sync(0xffffffffu, wsum0, 1));
    wsum1 = add_ftz_f32(wsum1, __shfl_xor_sync(0xffffffffu, wsum1, 2));
    wsum1 = add_ftz_f32(wsum1, __shfl_xor_sync(0xffffffffu, wsum1, 1));
    if (tid_in_grp == 0) {
      smem.warp_sum[warp_id][group_id]     = wsum0;
      smem.warp_sum[warp_id][group_id + 8] = wsum1;
    }

    // Step 4: Wait V0(nb) -> update row_sum -> prefetch next K0 -> compute PV(V0).
    cp_async_wait<1>();
    __syncthreads();

    const float tile_sum0 = add_ftz_f32(
        add_ftz_f32(smem.warp_sum[0][group_id], smem.warp_sum[1][group_id]),
        add_ftz_f32(smem.warp_sum[2][group_id], smem.warp_sum[3][group_id]));
    const float tile_sum1 = add_ftz_f32(
        add_ftz_f32(smem.warp_sum[0][group_id + 8], smem.warp_sum[1][group_id + 8]),
        add_ftz_f32(smem.warp_sum[2][group_id + 8], smem.warp_sum[3][group_id + 8]));
    row_sum0 = (nb == nb_min) ? tile_sum0 : fma_rn_ftz_f32(row_sum0, sc0, tile_sum0);
    row_sum1 = (nb == nb_min) ? tile_sum1 : fma_rn_ftz_f32(row_sum1, sc1, tile_sum1);

    const bool has_next = (nb + 1 < nb_max);
    if (has_next) {
      issue_kv_tile_async(
          smem.kv[0], k_ptr, block_table, tok_base + 64, k_start_cta, k_end_cta,
          block_size, kv_head, k_stride_block, k_stride_slot, k_stride_head, tid);
      cp_async_commit();
    }

    #pragma unroll
    for (int ks_p = 0; ks_p < 2; ++ks_p) {
      uint32_t p_frag[4];
      ldsm4_swizzled(p_frag, smem.p, ks_p, lane);
      #pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        uint32_t v_frag[2];
        ldsm2_v_trans_swizzled(v_frag, smem.kv[2], warp_id, dt, ks_p, lane);
        mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
      }
    }

    // Step 5: Wait V1(nb) -> prefetch next K1 (or emit early PDL) -> compute PV(V1).
    if (has_next) {
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    if (has_next) {
      issue_kv_tile_async(
          smem.kv[1], k_ptr, block_table, tok_base + 96, k_start_cta, k_end_cta,
          block_size, kv_head, k_stride_block, k_stride_slot, k_stride_head, tid);
      cp_async_commit();
    } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      if (pdl_mode & 2) {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
      }
#endif
    }

    #pragma unroll
    for (int ks_p = 0; ks_p < 2; ++ks_p) {
      uint32_t p_frag[4];
      ldsm4_swizzled(p_frag, smem.p, 2 + ks_p, lane);
      #pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        uint32_t v_frag[2];
        ldsm2_v_trans_swizzled(v_frag, smem.kv[3], warp_id, dt, ks_p, lane);
        mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
      }
    }

    if (has_next) {
      __syncthreads();
    }
  }

  // Optional attention sinks (virtual key with natural-log logit sinks[q_head] and value 0).
  if (sinks_ptr != nullptr) {
    const float sink_val = sinks_is_bf16
        ? __bfloat162float(__ldg(reinterpret_cast<const __nv_bfloat16*>(sinks_ptr) + q_head))
        : __ldg(reinterpret_cast<const float*>(sinks_ptr) + q_head);
    const float sink_log2 = sink_val * kLog2e;
    const float m0_log2 = (row_max0 == -INFINITY) ? -INFINITY : mul_ftz_f32(row_max0, scale_log2);
    const float m1_log2 = (row_max1 == -INFINITY) ? -INFINITY : mul_ftz_f32(row_max1, scale_log2);
    if (sink_log2 > m0_log2) {
      const float sc = (m0_log2 == -INFINITY) ? 0.0f : fast_exp2f(sub_ftz_f32(m0_log2, sink_log2));
      row_sum0 = fma_rn_ftz_f32(row_sum0, sc, 1.0f);
      #pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        o_frag[dt][0] = mul_ftz_f32(o_frag[dt][0], sc);
        o_frag[dt][1] = mul_ftz_f32(o_frag[dt][1], sc);
      }
    } else {
      row_sum0 = add_ftz_f32(row_sum0, fast_exp2f(sub_ftz_f32(sink_log2, m0_log2)));
    }
    if (sink_log2 > m1_log2) {
      const float sc = (m1_log2 == -INFINITY) ? 0.0f : fast_exp2f(sub_ftz_f32(m1_log2, sink_log2));
      row_sum1 = fma_rn_ftz_f32(row_sum1, sc, 1.0f);
      #pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        o_frag[dt][2] = mul_ftz_f32(o_frag[dt][2], sc);
        o_frag[dt][3] = mul_ftz_f32(o_frag[dt][3], sc);
      }
    } else {
      row_sum1 = add_ftz_f32(row_sum1, fast_exp2f(sub_ftz_f32(sink_log2, m1_log2)));
    }
  }

  // Epilogue: normalize and store BF16x2 to out[q0 + m, q_head, d_col..d_col+1] for m < actual_m.
  const int m0 = group_id;
  const int m1 = group_id + 8;
  const bool valid0 = (m0 < actual_m);
  const bool valid1 = (m1 < actual_m);
  const float inv0 = (row_sum0 > 0.0f) ? fast_rcpf(row_sum0) : 0.0f;
  const float inv1 = (row_sum1 > 0.0f) ? fast_rcpf(row_sum1) : 0.0f;
  const int64_t out_base0 = static_cast<int64_t>(q0 + m0) * out_tok_stride +
                            static_cast<int64_t>(q_head) * out_head_stride;
  const int64_t out_base1 = static_cast<int64_t>(q0 + m1) * out_tok_stride +
                            static_cast<int64_t>(q_head) * out_head_stride;
  #pragma unroll
  for (int dt = 0; dt < 4; ++dt) {
    const int d_col = (warp_id << 5) + (dt << 3) + (tid_in_grp << 1);
    if (valid0) {
      const __nv_bfloat162 ob0 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[dt][0], inv0), mul_ftz_f32(o_frag[dt][1], inv0));
      *reinterpret_cast<uint32_t*>(out_ptr + out_base0 + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob0);
    }
    if (valid1) {
      const __nv_bfloat162 ob1 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[dt][2], inv1), mul_ftz_f32(o_frag[dt][3], inv1));
      *reinterpret_cast<uint32_t*>(out_ptr + out_base1 + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob1);
    }
  }
}

static int draft_attn_pdl_mode() {
  static const int mode = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_ATTN_PDL");
    if (!s || !*s) s = std::getenv("MUSE_DRAFT_PDL");
    if (!s || !*s) return 3;
    const int v = std::atoi(s);
    if (v <= 0) return 0;
    return (v == 1 || v == 2 || v == 3) ? v : 3;
  }();
  return mode;
}

static void ensure_draft_attn_carveout() {
  static bool done = false;
  if (done) return;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  const bool swapped = (cudaThreadExchangeStreamCaptureMode(&mode) == cudaSuccess);
  (void)cudaFuncSetAttribute(
      dflash_draft_attn_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
  (void)cudaGetLastError();
  if (swapped) {
    (void)cudaThreadExchangeStreamCaptureMode(&mode);
  }
  done = true;
}

}  // namespace muse_draft_attn

void dflash_draft_attn(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor out,
    torch::Tensor seqused_k,
    torch::Tensor block_table,
    torch::Tensor cu_seqlens_q,
    double softmax_scale,
    int64_t window_left,
    int64_t window_right,
    int64_t max_q,
    c10::optional<torch::Tensor> sinks) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.dim() == 3,
              "dflash_draft_attn expects CUDA BF16 3D q [M, nq, 128]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.dim() == 3,
              "dflash_draft_attn expects CUDA BF16 3D out [M, nq, 128]");
  TORCH_CHECK(k_cache.is_cuda() && k_cache.scalar_type() == torch::kBFloat16 && k_cache.dim() == 4,
              "dflash_draft_attn expects CUDA BF16 4D k_cache [B, N, nkv, 128]");
  TORCH_CHECK(v_cache.is_cuda() && v_cache.scalar_type() == torch::kBFloat16 && v_cache.dim() == 4,
              "dflash_draft_attn expects CUDA BF16 4D v_cache [B, N, nkv, 128]");
  TORCH_CHECK(seqused_k.is_cuda() && seqused_k.scalar_type() == torch::kInt32 && seqused_k.numel() >= 1,
              "dflash_draft_attn expects CUDA int32 seqused_k");
  TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_q.scalar_type() == torch::kInt32 &&
                  cu_seqlens_q.is_contiguous() && cu_seqlens_q.numel() >= 2,
              "dflash_draft_attn expects contiguous CUDA int32 cu_seqlens_q[>=2]");
  TORCH_CHECK(block_table.is_cuda() && block_table.scalar_type() == torch::kInt32 &&
                  (block_table.dim() == 1 || block_table.dim() == 2) &&
                  block_table.stride(block_table.dim() - 1) == 1,
              "dflash_draft_attn expects CUDA int32 block_table with contiguous last dim");

  const int num_heads = static_cast<int>(q.size(1));
  const int head_dim = static_cast<int>(q.size(2));
  const int block_size = static_cast<int>(k_cache.size(1));
  const int num_kv_heads = static_cast<int>(k_cache.size(2));
  const int q_rows = static_cast<int>(q.size(0));
  TORCH_CHECK(head_dim == 128 && out.size(1) == num_heads && out.size(2) == 128 &&
                  out.size(0) == q_rows && k_cache.size(3) == 128 && v_cache.size(3) == 128 &&
                  v_cache.size(1) == block_size && v_cache.size(2) == num_kv_heads,
              "dflash_draft_attn shape mismatch (requires head_dim=128)");
  TORCH_CHECK(num_heads > 0 && num_kv_heads > 0 && (num_heads % num_kv_heads) == 0,
              "dflash_draft_attn requires num_heads % num_kv_heads == 0");
  TORCH_CHECK(block_size > 0 && softmax_scale > 0.0,
              "dflash_draft_attn requires block_size > 0 and softmax_scale > 0");
  TORCH_CHECK(q.stride(2) == 1 && out.stride(2) == 1 &&
                  k_cache.stride(3) == 1 && v_cache.stride(3) == 1,
              "dflash_draft_attn requires contiguous head_dim");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(q.data_ptr()) & 15u) == 0u &&
                  (reinterpret_cast<uintptr_t>(k_cache.data_ptr()) & 15u) == 0u &&
                  (reinterpret_cast<uintptr_t>(v_cache.data_ptr()) & 15u) == 0u &&
                  (reinterpret_cast<uintptr_t>(out.data_ptr()) & 3u) == 0u,
              "dflash_draft_attn requires 16B-aligned Q/K/V and 4B-aligned out");
  TORCH_CHECK((q.stride(0) & 7) == 0 && (q.stride(1) & 7) == 0 &&
                  (k_cache.stride(0) & 7) == 0 && (k_cache.stride(1) & 7) == 0 &&
                  (k_cache.stride(2) & 7) == 0 &&
                  (v_cache.stride(0) & 7) == 0 && (v_cache.stride(1) & 7) == 0 &&
                  (v_cache.stride(2) & 7) == 0 &&
                  (out.stride(0) & 1) == 0 && (out.stride(1) & 1) == 0,
              "dflash_draft_attn requires 16B-aligned Q/K/V strides and 4B-aligned out strides");

  const void* sinks_ptr = nullptr;
  int sinks_is_bf16 = 0;
  if (sinks.has_value() && sinks->defined()) {
    const auto& s = *sinks;
    TORCH_CHECK(s.is_cuda() && s.dim() == 1 && s.is_contiguous() && s.size(0) == num_heads &&
                    (s.scalar_type() == torch::kFloat32 || s.scalar_type() == torch::kBFloat16),
                "dflash_draft_attn expects contiguous 1D FP32/BF16 sinks[num_heads]");
    sinks_ptr = s.data_ptr();
    sinks_is_bf16 = (s.scalar_type() == torch::kBFloat16) ? 1 : 0;
  }

  muse_draft_attn::ensure_draft_attn_carveout();
  const int pdl = muse_draft_attn::draft_attn_pdl_mode();
  const int gqa_ratio = num_heads / num_kv_heads;
  const int eff_max_q = (max_q > 0 && max_q < 16) ? static_cast<int>(max_q) : 16;
  auto st = c10::cuda::getCurrentCUDAStream().stream();

  cudaLaunchConfig_t lcfg = {};
  lcfg.gridDim = dim3(static_cast<unsigned>(num_heads), 1u, 1u);
  lcfg.blockDim = dim3(128u, 1u, 1u);
  lcfg.dynamicSmemBytes = 0;
  lcfg.stream = st;
  cudaLaunchAttribute attr[1];
  if ((pdl & 1) != 0) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    lcfg.attrs = attr;
    lcfg.numAttrs = 1;
  }

  cudaError_t err = cudaLaunchKernelEx(
      &lcfg,
      muse_draft_attn::dflash_draft_attn_kernel,
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      seqused_k.data_ptr<int32_t>(),
      block_table.data_ptr<int32_t>(),
      cu_seqlens_q.data_ptr<int32_t>(),
      sinks_ptr,
      sinks_is_bf16,
      gqa_ratio,
      block_size,
      static_cast<int>(window_left),
      static_cast<int>(window_right),
      eff_max_q,
      q_rows,
      static_cast<float>(softmax_scale),
      static_cast<int64_t>(q.stride(0)),
      static_cast<int64_t>(q.stride(1)),
      static_cast<int64_t>(out.stride(0)),
      static_cast<int64_t>(out.stride(1)),
      static_cast<int64_t>(k_cache.stride(0)),
      static_cast<int64_t>(k_cache.stride(1)),
      static_cast<int64_t>(k_cache.stride(2)),
      static_cast<int64_t>(v_cache.stride(0)),
      static_cast<int64_t>(v_cache.stride(1)),
      static_cast<int64_t>(v_cache.stride(2)),
      pdl);
  TORCH_CHECK(err == cudaSuccess, "dflash_draft_attn launch failed: ", cudaGetErrorString(err));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
