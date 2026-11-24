// SPDX-License-Identifier: Apache-2.0
// Strict 0-ULP FA2-bit-exact GQA-16 Tensor-Core M=1 decode attention for MuseGlimmer
// (32 Q heads, 2 KV heads, head_dim=128, BF16 Q/K/V, paged KV cache, sliding_window or full).
//
// Target: NVIDIA GB10 (sm_121a, 48 SMs).
// Modes:
//   - fa2_splits == 1 (matches FlashAttention-2 splitkv num_splits=1, kBlockN=64):
//       * grid_splits == 8: 2-pass parallel K-split (16 CTAs x 128 threads) + V-dim-split
//         (16 CTAs x 64 threads), 100.000% 0-ULP bit-identical to 1-split FA2, deadlock-free.
//       * grid_splits == 1 (or seqlen_k > 16384): 2-CTA single-pass kBlockN=64 exact FA2.
//   - fa2_splits > 1 (matches FlashAttention-2 splitkv num_splits>1, kBlockN=128 + butterfly LSE):
//       * 16 CTAs (grid_splits x 2 KV heads) + exact power-of-2 butterfly LSE combine, 0-ULP.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace muse_gqa16 {

constexpr int kMaxSplits = 64;
constexpr float kLog2e = 1.44269504088896340736f;  // 0x3FB8AA3B
constexpr float kLn2   = 0.69314718055994530942f;  // 0x3F317218

struct __align__(128) SharedStorage {
  uint4 kv[3][32][16];     // 3 x 8 KiB = 24 KiB (128B XOR swizzled: c_phys = c ^ (r & 7))
  uint4 p[16][16];         // 4 KiB (128B XOR swizzled: c_phys = c ^ (m & 7))
  float warp_max[4][16];   // 256 B
  float warp_sum[4][16];   // 256 B
};                         // Total = 29,184 B (28.5 KiB < 32 KiB)

struct __align__(128) Pass1Storage {
  uint4 kv[2][32][16];     // 16 KiB
  float warp_max[4][16];   // 256 B
};                         // Total = 16,640 B

struct __align__(128) Pass2Storage {
  uint4 v[2][128][3];      // 2 x (128 x 48 B) = 12,288 B (conflict-free 12-word stride!)
  float inv[16];           // 64 B
};                         // Total = 12,352 B

template <int kBlockN, int kVRowStride, int kStagesK>
struct __align__(128) OnePassDSplitStorage {
  union {
    uint4 k[kStagesK][32][16];    // 16 or 32 KiB swizzled K chunks
    float4 p_f32[kBlockN << 2];   // 4 or 8 KiB (live only between K MMA and next-nb K prefetch)
  };
  uint4 p[16][16];                // 4 KiB swizzled BF16 P
  uint4 v[kBlockN][kVRowStride];  // 2..10 KiB conflict-free V slice
  float warp_max[4][16];          // 256 B
  float warp_sum[16];             // 64 B
};

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

__device__ __forceinline__ void st_cg_f32x4(float* p, float x, float y, float z, float w) {
  asm volatile("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};"
               :: "l"(p), "f"(x), "f"(y), "f"(z), "f"(w));
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

// Load k16n8 FragB from conflict-free [128][3] uint4 V tile in Pass 2.
__device__ __forceinline__ void ldsm2_v_trans_pass2(
    uint32_t (&v)[2],
    const uint4 tile[128][3],
    int warp_id,
    int ks_p,
    int lane) {
  const int r = (ks_p << 4) + (lane & 15);
  const uint32_t addr = smem_u32(&tile[r][warp_id]);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(v[0]), "=r"(v[1])
      : "r"(addr));
}

template <int kBlockN, int kVRowStride>
__device__ __forceinline__ void ldsm2_v_trans_dsplit(
    uint32_t (&v)[2],
    const uint4 tile[kBlockN][kVRowStride],
    int warp_id,
    int ks_p,
    int lane) {
  const int r = (ks_p << 4) + (lane & 15);
  const uint32_t addr = smem_u32(&tile[r][warp_id]);
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
  const bool full_valid = (tok_base >= k_start) && (tok_base + 32 <= seqlen_k) &&
                          ((block_size == 16) || (slot_base0 + 16 <= block_size && slot_base1 + 16 <= block_size));

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
        const int lb = (block_size == 16) ? (tok >> 4) : (tok / block_size);
        const int slot = (block_size == 16) ? (tok & 15) : (tok - lb * block_size);
        const int phys_blk = __ldg(block_table + lb);
        const int64_t row_off = static_cast<int64_t>(phys_blk) * stride_block +
                                static_cast<int64_t>(slot) * stride_slot + head_v_off;
        cp_async_16(smem_u32(&smem_tile[r][c ^ (r & 7)]), kv_ptr + row_off + (c << 3));
      } else {
        smem_tile[r][c ^ (r & 7)] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  }
}

template <int kBlockN>
__device__ __forceinline__ void issue_v_col16_tile_async(
    uint4 v_tile[128][3],
    const __nv_bfloat16* __restrict__ kv_ptr,
    const int32_t* __restrict__ block_table,
    int tok_base,
    int k_start,
    int seqlen_k,
    int block_size,
    int kv_head,
    int d_cta,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    int tid) {
  #pragma unroll
  for (int u = 0; u < (kBlockN >> 6); ++u) {
    const int r = (u << 6) + tid;
    const int tok = tok_base + r;
    if (tok >= k_start && tok < seqlen_k) {
      const int lb = (block_size == 16) ? (tok >> 4) : (tok / block_size);
      const int slot = (block_size == 16) ? (tok & 15) : (tok - lb * block_size);
      const int phys_blk = __ldg(block_table + lb);
      const __nv_bfloat16* src =
          kv_ptr + static_cast<int64_t>(phys_blk) * stride_block +
          static_cast<int64_t>(slot) * stride_slot +
          static_cast<int64_t>(kv_head) * stride_kv_head + 128 + (d_cta << 4);
      cp_async_16(smem_u32(&v_tile[r][0]), src);
      cp_async_16(smem_u32(&v_tile[r][1]), src + 8);
    } else {
      v_tile[r][0] = make_uint4(0u, 0u, 0u, 0u);
      v_tile[r][1] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
}

__device__ __forceinline__ void load_q_group_to_regs(
    uint32_t (&q_frag)[8][4],
    uint4 q_smem[32][16],
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
    q_smem[m][c ^ (m & 7)] =
        *reinterpret_cast<const uint4*>(q_group + static_cast<int64_t>(m) * q_head_stride + (c << 3));
  }
  __syncthreads();
  #pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    ldsm4_swizzled(q_frag[ks], q_smem, ks, lane);
  }
  __syncthreads();
}

// ============================================================================
// Exact 1-split FA2 kernel (kBlockN = 64, single CTA per KV head)
// ============================================================================
__device__ __forceinline__ void run_k64_single_cta_body(
    SharedStorage& smem,
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ kv_ptr,
    __nv_bfloat16* __restrict__ out_ptr,
    const int32_t* __restrict__ block_table,
    int seqlen_k,
    int k_start,
    int nb_min,
    int nb_max,
    int block_size,
    int kv_head,
    float scale_log2,
    int64_t q_head_stride,
    int64_t out_head_stride,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    int tid) {
  const int warp_id    = tid >> 5;
  const int lane       = tid & 31;
  const int group_id   = lane >> 2;
  const int tid_in_grp = lane & 3;

  if (nb_min >= nb_max) {
    const int q_local = tid >> 3;
    const int d_base = (tid & 7) << 4;
    __nv_bfloat16* dst = out_ptr + static_cast<int64_t>(kv_head * 16 + q_local) * out_head_stride + d_base;
    *reinterpret_cast<uint4*>(dst)     = make_uint4(0u, 0u, 0u, 0u);
    *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0u, 0u, 0u, 0u);
    return;
  }

  uint32_t q_frag[8][4];
  load_q_group_to_regs(q_frag, smem.kv[0], q_ptr, kv_head, q_head_stride, tid, lane);

  float o_frag[4][4] = {0.f};
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  for (int nb = nb_max - 1; nb >= nb_min; --nb) {
    const int block_tok_base = nb << 6;
    float s_frag[2][4] = {0.f};

    issue_kv_tile_async<false>(smem.kv[0], kv_ptr, block_table, block_tok_base, k_start, seqlen_k,
                               block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();
    issue_kv_tile_async<false>(smem.kv[1], kv_ptr, block_table, block_tok_base + 32, k_start, seqlen_k,
                               block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();

    cp_async_wait<1>();
    __syncthreads();
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t b_frag[2];
      ldsm2_k_swizzled(b_frag, smem.kv[0], warp_id, ks, lane);
      mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[0]);
    }
    cp_async_wait<0>();
    __syncthreads();
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t b_frag[2];
      ldsm2_k_swizzled(b_frag, smem.kv[1], warp_id, ks, lane);
      mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[1]);
    }
    __syncthreads();

    issue_kv_tile_async<true>(smem.kv[0], kv_ptr, block_table, block_tok_base, k_start, seqlen_k,
                              block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();
    issue_kv_tile_async<true>(smem.kv[1], kv_ptr, block_table, block_tok_base + 32, k_start, seqlen_k,
                              block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();

    float m0 = -INFINITY, m1 = -INFINITY;
    #pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      const int col0 = block_tok_base + (ch << 5) + (warp_id << 3) + (tid_in_grp << 1);
      const int col1 = col0 + 1;
      if (col0 < k_start || col0 >= seqlen_k) { s_frag[ch][0] = -INFINITY; s_frag[ch][2] = -INFINITY; }
      if (col1 < k_start || col1 >= seqlen_k) { s_frag[ch][1] = -INFINITY; s_frag[ch][3] = -INFINITY; }
      m0 = max_ftz_f32(m0, max_ftz_f32(s_frag[ch][0], s_frag[ch][1]));
      m1 = max_ftz_f32(m1, max_ftz_f32(s_frag[ch][2], s_frag[ch][3]));
    }
    m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    if (tid_in_grp == 0) {
      smem.warp_max[warp_id][group_id]     = m0;
      smem.warp_max[warp_id][group_id + 8] = m1;
    }
    __syncthreads();

    const float tile_max0 = max_ftz_f32(max_ftz_f32(smem.warp_max[0][group_id], smem.warp_max[1][group_id]),
                                        max_ftz_f32(smem.warp_max[2][group_id], smem.warp_max[3][group_id]));
    const float tile_max1 = max_ftz_f32(max_ftz_f32(smem.warp_max[0][group_id + 8], smem.warp_max[1][group_id + 8]),
                                        max_ftz_f32(smem.warp_max[2][group_id + 8], smem.warp_max[3][group_id + 8]));
    float sc0 = 1.0f, sc1 = 1.0f;
    if (nb == nb_max - 1) {
      row_max0 = tile_max0;
      row_max1 = tile_max1;
    } else {
      const float prev0 = row_max0, prev1 = row_max1;
      row_max0 = max_ftz_f32(prev0, tile_max0);
      row_max1 = max_ftz_f32(prev1, tile_max1);
      const float cur0 = (row_max0 == -INFINITY) ? 0.0f : row_max0;
      const float cur1 = (row_max1 == -INFINITY) ? 0.0f : row_max1;
      sc0 = fast_exp2f(mul_ftz_f32(scale_log2, sub_ftz_f32(prev0, cur0)));
      sc1 = fast_exp2f(mul_ftz_f32(scale_log2, sub_ftz_f32(prev1, cur1)));
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
    float4* p_f32_smem = reinterpret_cast<float4*>(smem.kv[2]);
    #pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      const float p0 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][0], scale_log2), max_sc0));
      const float p1 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][1], scale_log2), max_sc0));
      const float p2 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][2], scale_log2), max_sc1));
      const float p3 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][3], scale_log2), max_sc1));
      const int ni = (ch << 2) + warp_id;
      p_f32_smem[ni * 32 + lane] = make_float4(p0, p1, p2, p3);
      const __nv_bfloat162 pb0 = __floats2bfloat162_rn(p0, p1);
      const __nv_bfloat162 pb1 = __floats2bfloat162_rn(p2, p3);
      const int c_phys_p = ni ^ group_id;
      reinterpret_cast<uint32_t*>(&smem.p[group_id][c_phys_p])[tid_in_grp] =
          *reinterpret_cast<const uint32_t*>(&pb0);
      reinterpret_cast<uint32_t*>(&smem.p[group_id + 8][c_phys_p])[tid_in_grp] =
          *reinterpret_cast<const uint32_t*>(&pb1);
    }

    cp_async_wait<1>();
    __syncthreads();

    if (warp_id == 0) {
      const float4 pv0 = p_f32_smem[0 * 32 + lane];
      if (nb == nb_max - 1) {
        row_sum0 = add_ftz_f32(pv0.x, pv0.y);
        row_sum1 = add_ftz_f32(pv0.z, pv0.w);
      } else {
        row_sum0 = add_ftz_f32(fma_rn_ftz_f32(row_sum0, sc0, pv0.x), pv0.y);
        row_sum1 = add_ftz_f32(fma_rn_ftz_f32(row_sum1, sc1, pv0.z), pv0.w);
      }
      #pragma unroll
      for (int ni = 1; ni < 8; ++ni) {
        const float4 pv = p_f32_smem[ni * 32 + lane];
        row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, pv.x), pv.y);
        row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, pv.z), pv.w);
      }
    }

    #pragma unroll
    for (int ks_p = 0; ks_p < 2; ++ks_p) {
      uint32_t p_frag[4];
      ldsm4_swizzled(p_frag, smem.p, ks_p, lane);
      #pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        uint32_t v_frag[2];
        ldsm2_v_trans_swizzled(v_frag, smem.kv[0], warp_id, dt, ks_p, lane);
        mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
      }
    }
    cp_async_wait<0>();
    __syncthreads();
    #pragma unroll
    for (int ks_p = 0; ks_p < 2; ++ks_p) {
      uint32_t p_frag[4];
      ldsm4_swizzled(p_frag, smem.p, 2 + ks_p, lane);
      #pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        uint32_t v_frag[2];
        ldsm2_v_trans_swizzled(v_frag, smem.kv[1], warp_id, dt, ks_p, lane);
        mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
      }
    }
    __syncthreads();
  }

  if (warp_id == 0) {
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 2));
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 1));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 2));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 1));
    const float inv0 = (row_sum0 == 0.0f || row_sum0 != row_sum0) ? 1.0f : fast_rcpf(row_sum0);
    const float inv1 = (row_sum1 == 0.0f || row_sum1 != row_sum1) ? 1.0f : fast_rcpf(row_sum1);
    if (tid_in_grp == 0) {
      smem.warp_sum[0][group_id]     = inv0;
      smem.warp_sum[0][group_id + 8] = inv1;
    }
  }
  __syncthreads();

  const float inv0 = smem.warp_sum[0][group_id];
  const float inv1 = smem.warp_sum[0][group_id + 8];
  #pragma unroll
  for (int dt = 0; dt < 4; ++dt) {
    const int d_col = (warp_id << 5) + (dt << 3) + (tid_in_grp << 1);
    const __nv_bfloat162 ob0 = __floats2bfloat162_rn(
        mul_ftz_f32(o_frag[dt][0], inv0), mul_ftz_f32(o_frag[dt][1], inv0));
    const __nv_bfloat162 ob1 = __floats2bfloat162_rn(
        mul_ftz_f32(o_frag[dt][2], inv1), mul_ftz_f32(o_frag[dt][3], inv1));
    *reinterpret_cast<uint32_t*>(
        out_ptr + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) =
        *reinterpret_cast<const uint32_t*>(&ob0);
    *reinterpret_cast<uint32_t*>(
        out_ptr + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) =
        *reinterpret_cast<const uint32_t*>(&ob1);
  }
}

__global__ __launch_bounds__(128, 2) void gqa16_fa2_k64_single_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ kv_ptr,
    __nv_bfloat16* __restrict__ out_ptr,
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
  if (blockIdx.x > 0) return;
  const int kv_head  = blockIdx.y;
  const int tid      = threadIdx.x;
  const int seqlen_k = seqused_k_ptr[0];
  const int k_start  = (window_left >= 0)
      ? ((seqlen_k - 1 - window_left > 0) ? (seqlen_k - 1 - window_left) : 0)
      : 0;
  const int nb_max = (seqlen_k > 0) ? ((seqlen_k + 63) >> 6) : 0;
  const int nb_min = (window_left >= 0 && seqlen_k > 0) ? (k_start >> 6) : 0;
  run_k64_single_cta_body(
      smem, q_ptr, kv_ptr, out_ptr, block_table,
      seqlen_k, k_start, nb_min, nb_max, block_size, kv_head, scale_log2,
      q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head, tid);
}

// ============================================================================
// Parallel 0-ULP 2-Pass Exact kBlockN={64,128} Kernel (16 CTAs across 2 KV heads)
// Pass 1: Parallel K-split Q @ K^T + tile_max -> L2 scratch
// Pass 2: Parallel V-dim-split (8 CTAs x 2 warps = 16 d_tiles) exact online softmax + P @ V
// ============================================================================
template <int kBlockN>
__global__ __launch_bounds__(128, 2) void gqa16_pass1_qk_max_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ kv_ptr,
    float* __restrict__ s_scratch,
    float* __restrict__ m_scratch,
    const int32_t* __restrict__ seqused_k_ptr,
    const int32_t* __restrict__ block_table,
    int block_size,
    int window_left,
    int64_t q_head_stride,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head) {
  static_assert(kBlockN == 64 || kBlockN == 128, "kBlockN must be 64 or 128");
  constexpr int kShift  = (kBlockN == 128) ? 7 : 6;
  constexpr int kChunks = kBlockN >> 5;
  constexpr int kMaxNB  = (kBlockN == 128) ? 128 : 256;
  constexpr int kSStride = kChunks * 512;

  __shared__ Pass1Storage smem;

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
  const int nb_max = (seqlen_k > 0) ? ((seqlen_k + kBlockN - 1) >> kShift) : 0;
  const int nb_min = (window_left >= 0 && seqlen_k > 0) ? (k_start >> kShift) : 0;
  const int total_nb = (nb_max > nb_min) ? (nb_max - nb_min) : 0;

  if (worker_idx >= total_nb || total_nb > kMaxNB) return;

  uint32_t q_frag[8][4];
  load_q_group_to_regs(q_frag, smem.kv[0], q_ptr, kv_head, q_head_stride, tid, lane);

  for (int idx = worker_idx; idx < total_nb; idx += grid_splits) {
    const int nb = (nb_max - 1) - idx;
    const int block_tok_base = nb << kShift;
    float s_frag[kChunks][4] = {0.f};

    issue_kv_tile_async<false>(smem.kv[0], kv_ptr, block_table, block_tok_base, k_start, seqlen_k,
                               block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();
    #pragma unroll
    for (int ch = 0; ch < kChunks; ++ch) {
      if (ch + 1 < kChunks) {
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

    float m0 = -INFINITY, m1 = -INFINITY;
    float* s_dst = s_scratch + (kv_head * kMaxNB + idx) * kSStride;
    #pragma unroll
    for (int ch = 0; ch < kChunks; ++ch) {
      const int col0 = block_tok_base + (ch << 5) + (warp_id << 3) + (tid_in_grp << 1);
      const int col1 = col0 + 1;
      if (col0 < k_start || col0 >= seqlen_k) { s_frag[ch][0] = -INFINITY; s_frag[ch][2] = -INFINITY; }
      if (col1 < k_start || col1 >= seqlen_k) { s_frag[ch][1] = -INFINITY; s_frag[ch][3] = -INFINITY; }
      m0 = max_ftz_f32(m0, max_ftz_f32(s_frag[ch][0], s_frag[ch][1]));
      m1 = max_ftz_f32(m1, max_ftz_f32(s_frag[ch][2], s_frag[ch][3]));
      st_cg_f32x4(s_dst + (((ch << 2) + warp_id) * 32 + lane) * 4,
                  s_frag[ch][0], s_frag[ch][1], s_frag[ch][2], s_frag[ch][3]);
    }

    m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    if (tid_in_grp == 0) {
      smem.warp_max[warp_id][group_id]     = m0;
      smem.warp_max[warp_id][group_id + 8] = m1;
    }
    __syncthreads();

    if (tid < 16) {
      const float tm = max_ftz_f32(max_ftz_f32(smem.warp_max[0][tid], smem.warp_max[1][tid]),
                                   max_ftz_f32(smem.warp_max[2][tid], smem.warp_max[3][tid]));
      st_cg_f32(m_scratch + (kv_head * kMaxNB + idx) * 16 + tid, tm);
    }
    __syncthreads();
  }
}

template <int kBlockN>
__global__ __launch_bounds__(64, 4) void gqa16_pass2_pv_dim_kernel(
    const __nv_bfloat16* __restrict__ kv_ptr,
    __nv_bfloat16* __restrict__ out_ptr,
    const float* __restrict__ s_scratch,
    const float* __restrict__ m_scratch,
    const int32_t* __restrict__ seqused_k_ptr,
    const int32_t* __restrict__ block_table,
    int block_size,
    int window_left,
    float scale_log2,
    int64_t out_head_stride,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head) {
  static_assert(kBlockN == 64 || kBlockN == 128, "kBlockN must be 64 or 128");
  constexpr int kShift      = (kBlockN == 128) ? 7 : 6;
  constexpr int kHalfBlocks = kBlockN >> 4;
  constexpr int kMaxNB      = (kBlockN == 128) ? 128 : 256;
  constexpr int kSStride    = (kBlockN == 128) ? 2048 : 1024;

  __shared__ Pass2Storage smem;

  const int d_cta    = blockIdx.x;
  const int kv_head  = blockIdx.y;
  const int tid      = threadIdx.x;
  const int seqlen_k = seqused_k_ptr[0];
  const int k_start  = (window_left >= 0)
      ? ((seqlen_k - 1 - window_left > 0) ? (seqlen_k - 1 - window_left) : 0)
      : 0;
  const int nb_max   = (seqlen_k > 0) ? ((seqlen_k + kBlockN - 1) >> kShift) : 0;
  const int nb_min   = (window_left >= 0 && seqlen_k > 0) ? (k_start >> kShift) : 0;
  const int total_nb = (nb_max > nb_min) ? (nb_max - nb_min) : 0;

  const int warp_id    = tid >> 5;
  const int lane       = tid & 31;
  const int group_id   = lane >> 2;
  const int tid_in_grp = lane & 3;

  if (total_nb <= 0) {
    const int d_col = (d_cta << 4) + (warp_id << 3) + (tid_in_grp << 1);
    *reinterpret_cast<uint32_t*>(
        out_ptr + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) = 0u;
    *reinterpret_cast<uint32_t*>(
        out_ptr + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) = 0u;
    return;
  }

  float o_frag[4] = {0.f, 0.f, 0.f, 0.f};
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  issue_v_col16_tile_async<kBlockN>(
      smem.v[0], kv_ptr, block_table, (nb_max - 1) << kShift, k_start, seqlen_k,
      block_size, kv_head, d_cta, stride_block, stride_slot, stride_kv_head, tid);
  cp_async_commit();

  for (int idx = 0; idx < total_nb; ++idx) {
    if (idx + 1 < total_nb) {
      const int nb_next = (nb_max - 1) - (idx + 1);
      issue_v_col16_tile_async<kBlockN>(
          smem.v[(idx + 1) & 1], kv_ptr, block_table, nb_next << kShift, k_start, seqlen_k,
          block_size, kv_head, d_cta, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
    }

    const float* m_src = m_scratch + (kv_head * kMaxNB + idx) * 16;
    const float tile_max0 = ld_cg_f32(m_src + group_id);
    const float tile_max1 = ld_cg_f32(m_src + group_id + 8);

    float sc0 = 1.0f, sc1 = 1.0f;
    if (idx == 0) {
      row_max0 = tile_max0;
      row_max1 = tile_max1;
    } else {
      const float prev0 = row_max0, prev1 = row_max1;
      row_max0 = max_ftz_f32(prev0, tile_max0);
      row_max1 = max_ftz_f32(prev1, tile_max1);
      const float cur0 = (row_max0 == -INFINITY) ? 0.0f : row_max0;
      const float cur1 = (row_max1 == -INFINITY) ? 0.0f : row_max1;
      sc0 = fast_exp2f(mul_ftz_f32(scale_log2, sub_ftz_f32(prev0, cur0)));
      sc1 = fast_exp2f(mul_ftz_f32(scale_log2, sub_ftz_f32(prev1, cur1)));
      o_frag[0] = mul_ftz_f32(o_frag[0], sc0);
      o_frag[1] = mul_ftz_f32(o_frag[1], sc0);
      o_frag[2] = mul_ftz_f32(o_frag[2], sc1);
      o_frag[3] = mul_ftz_f32(o_frag[3], sc1);
    }

    const float max_sc0 = (row_max0 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max0);
    const float max_sc1 = (row_max1 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max1);
    const float* s_src = s_scratch + (kv_head * kMaxNB + idx) * kSStride;

    uint32_t p_frag[kHalfBlocks][4];
    #pragma unroll
    for (int ks_p = 0; ks_p < kHalfBlocks; ++ks_p) {
      const int ni0 = ks_p << 1;
      const int ni1 = ni0 + 1;
      const float4 s0 = ld_cg_f32x4(s_src + (ni0 * 32 + lane) * 4);
      const float4 s1 = ld_cg_f32x4(s_src + (ni1 * 32 + lane) * 4);

      const float p00 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s0.x, scale_log2), max_sc0));
      const float p01 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s0.y, scale_log2), max_sc0));
      const float p02 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s0.z, scale_log2), max_sc1));
      const float p03 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s0.w, scale_log2), max_sc1));

      const float p10 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s1.x, scale_log2), max_sc0));
      const float p11 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s1.y, scale_log2), max_sc0));
      const float p12 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s1.z, scale_log2), max_sc1));
      const float p13 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s1.w, scale_log2), max_sc1));

      if (warp_id == 0) {
        if (ks_p == 0) {
          if (idx == 0) {
            row_sum0 = add_ftz_f32(p00, p01);
            row_sum1 = add_ftz_f32(p02, p03);
          } else {
            row_sum0 = add_ftz_f32(fma_rn_ftz_f32(row_sum0, sc0, p00), p01);
            row_sum1 = add_ftz_f32(fma_rn_ftz_f32(row_sum1, sc1, p02), p03);
          }
        } else {
          row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, p00), p01);
          row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, p02), p03);
        }
        row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, p10), p11);
        row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, p12), p13);
      }

      const __nv_bfloat162 b0 = __floats2bfloat162_rn(p00, p01);
      const __nv_bfloat162 b1 = __floats2bfloat162_rn(p02, p03);
      const __nv_bfloat162 b2 = __floats2bfloat162_rn(p10, p11);
      const __nv_bfloat162 b3 = __floats2bfloat162_rn(p12, p13);
      p_frag[ks_p][0] = *reinterpret_cast<const uint32_t*>(&b0);
      p_frag[ks_p][1] = *reinterpret_cast<const uint32_t*>(&b1);
      p_frag[ks_p][2] = *reinterpret_cast<const uint32_t*>(&b2);
      p_frag[ks_p][3] = *reinterpret_cast<const uint32_t*>(&b3);
    }

    if (idx + 1 < total_nb) {
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    #pragma unroll
    for (int ks_p = 0; ks_p < kHalfBlocks; ++ks_p) {
      uint32_t v_frag[2];
      ldsm2_v_trans_pass2(v_frag, smem.v[idx & 1], warp_id, ks_p, lane);
      mma_m16n8k16_bf16(p_frag[ks_p], v_frag, o_frag);
    }
    __syncthreads();
  }

  if (warp_id == 0) {
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 2));
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 1));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 2));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 1));
    const float inv0 = (row_sum0 == 0.0f || row_sum0 != row_sum0) ? 1.0f : fast_rcpf(row_sum0);
    const float inv1 = (row_sum1 == 0.0f || row_sum1 != row_sum1) ? 1.0f : fast_rcpf(row_sum1);
    if (tid_in_grp == 0) {
      smem.inv[group_id]     = inv0;
      smem.inv[group_id + 8] = inv1;
    }
  }
  __syncthreads();

  const float inv0 = smem.inv[group_id];
  const float inv1 = smem.inv[group_id + 8];
  const int d_col = (d_cta << 4) + (warp_id << 3) + (tid_in_grp << 1);
  const __nv_bfloat162 ob0 = __floats2bfloat162_rn(
      mul_ftz_f32(o_frag[0], inv0), mul_ftz_f32(o_frag[1], inv0));
  const __nv_bfloat162 ob1 = __floats2bfloat162_rn(
      mul_ftz_f32(o_frag[2], inv1), mul_ftz_f32(o_frag[3], inv1));
  *reinterpret_cast<uint32_t*>(
      out_ptr + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) =
      *reinterpret_cast<const uint32_t*>(&ob0);
  *reinterpret_cast<uint32_t*>(
      out_ptr + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) =
      *reinterpret_cast<const uint32_t*>(&ob1);
}

// ============================================================================
// Single-Pass 0-ULP D-Split Kernel (K_splits = 1, D_splits in {4, 8, 16})
// Zero workspace reads/writes, 1 kernel launch per layer instead of 2!
// ============================================================================
template <int kBlockN, int kActiveVWarps, int kVRowStride>
__device__ __forceinline__ void issue_v_dsplit_async(
    uint4 v_tile[kBlockN][kVRowStride],
    const __nv_bfloat16* __restrict__ kv_ptr,
    const int32_t* __restrict__ block_table,
    int tok_base,
    int k_start,
    int seqlen_k,
    int block_size,
    int kv_head,
    int d_cta,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    int tid) {
  const int d_off = 128 + d_cta * (kActiveVWarps * 8);
  if constexpr (kBlockN == 128) {
    const int tok = tok_base + tid;
    if (tok >= k_start && tok < seqlen_k) {
      const int lb = (block_size == 16) ? (tok >> 4) : (tok / block_size);
      const int slot = (block_size == 16) ? (tok & 15) : (tok - lb * block_size);
      const int phys_blk = __ldg(block_table + lb);
      const __nv_bfloat16* src =
          kv_ptr + static_cast<int64_t>(phys_blk) * stride_block +
          static_cast<int64_t>(slot) * stride_slot +
          static_cast<int64_t>(kv_head) * stride_kv_head + d_off;
      #pragma unroll
      for (int w = 0; w < kActiveVWarps; ++w) {
        cp_async_16(smem_u32(&v_tile[tid][w]), src + (w << 3));
      }
    } else {
      #pragma unroll
      for (int w = 0; w < kActiveVWarps; ++w) {
        v_tile[tid][w] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  } else {
    if (tid < 64) {
      const int tok = tok_base + tid;
      if (tok >= k_start && tok < seqlen_k) {
        const int lb = (block_size == 16) ? (tok >> 4) : (tok / block_size);
        const int slot = (block_size == 16) ? (tok & 15) : (tok - lb * block_size);
        const int phys_blk = __ldg(block_table + lb);
        const __nv_bfloat16* src =
            kv_ptr + static_cast<int64_t>(phys_blk) * stride_block +
            static_cast<int64_t>(slot) * stride_slot +
            static_cast<int64_t>(kv_head) * stride_kv_head + d_off;
        #pragma unroll
        for (int w = 0; w < kActiveVWarps; ++w) {
          cp_async_16(smem_u32(&v_tile[tid][w]), src + (w << 3));
        }
      } else {
        #pragma unroll
        for (int w = 0; w < kActiveVWarps; ++w) {
          v_tile[tid][w] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  }
}

template <int kBlockN, int kDSplits>
__global__ __launch_bounds__(128, 2) void gqa16_1pass_dsplit_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ kv_ptr,
    __nv_bfloat16* __restrict__ out_ptr,
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
  static_assert(kBlockN == 64 || kBlockN == 128, "kBlockN must be 64 or 128");
  static_assert(kDSplits == 4 || kDSplits == 8 || kDSplits == 16, "kDSplits must be 4, 8, or 16");
  constexpr int kShift        = (kBlockN == 128) ? 7 : 6;
  constexpr int kChunks       = kBlockN >> 5;
  constexpr int kHalfBlocks   = kBlockN >> 4;
  constexpr int kActiveVWarps = 16 / kDSplits;
  constexpr int kVRowStride   = (kActiveVWarps == 1) ? 1 : (kActiveVWarps + 1);
  constexpr int kSumWarp      = (kActiveVWarps < 4) ? kActiveVWarps : 0;
  constexpr int kStagesK      = kChunks;

  __shared__ OnePassDSplitStorage<kBlockN, kVRowStride, kStagesK> smem;

  const int d_cta      = blockIdx.x;
  const int kv_head    = blockIdx.y;
  const int tid        = threadIdx.x;
  const int warp_id    = tid >> 5;
  const int lane       = tid & 31;
  const int group_id   = lane >> 2;
  const int tid_in_grp = lane & 3;

  const int seqlen_k = seqused_k_ptr[0];
  const int k_start  = (window_left >= 0)
      ? ((seqlen_k - 1 - window_left > 0) ? (seqlen_k - 1 - window_left) : 0)
      : 0;
  const int nb_max = (seqlen_k > 0) ? ((seqlen_k + kBlockN - 1) >> kShift) : 0;
  const int nb_min = (window_left >= 0 && seqlen_k > 0) ? (k_start >> kShift) : 0;

  if (nb_min >= nb_max) {
    if (warp_id < kActiveVWarps) {
      const int d_col = d_cta * (kActiveVWarps * 8) + (warp_id << 3) + (tid_in_grp << 1);
      *reinterpret_cast<uint32_t*>(
          out_ptr + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) = 0u;
      *reinterpret_cast<uint32_t*>(
          out_ptr + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) = 0u;
    }
    return;
  }

  uint32_t q_frag[8][4];
  load_q_group_to_regs(q_frag, smem.k[0], q_ptr, kv_head, q_head_stride, tid, lane);

  float o_frag[4] = {0.f, 0.f, 0.f, 0.f};
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  // Prologue: prefetch all K chunks of nb_max - 1.
  #pragma unroll
  for (int ch = 0; ch < kChunks; ++ch) {
    issue_kv_tile_async<false>(smem.k[ch], kv_ptr, block_table, ((nb_max - 1) << kShift) + (ch << 5),
                               k_start, seqlen_k, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();
  }

  for (int nb = nb_max - 1; nb >= nb_min; --nb) {
    const int block_tok_base = nb << kShift;
    float s_frag[kChunks][4] = {0.f};

    // Issue V slice immediately so all kChunks K tiles + V slice stream concurrently!
    issue_v_dsplit_async<kBlockN, kActiveVWarps, kVRowStride>(
        smem.v, kv_ptr, block_table, block_tok_base, k_start, seqlen_k,
        block_size, kv_head, d_cta, stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();

    #pragma unroll
    for (int ch = 0; ch < kChunks; ++ch) {
      if (kChunks - ch == 4) cp_async_wait<4>();
      else if (kChunks - ch == 3) cp_async_wait<3>();
      else if (kChunks - ch == 2) cp_async_wait<2>();
      else cp_async_wait<1>();
      __syncthreads();
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t b_frag[2];
        ldsm2_k_swizzled(b_frag, smem.k[ch], warp_id, ks, lane);
        mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[ch]);
      }
    }

    float m0 = -INFINITY, m1 = -INFINITY;
    #pragma unroll
    for (int ch = 0; ch < kChunks; ++ch) {
      const int col0 = block_tok_base + (ch << 5) + (warp_id << 3) + (tid_in_grp << 1);
      const int col1 = col0 + 1;
      if (col0 < k_start || col0 >= seqlen_k) { s_frag[ch][0] = -INFINITY; s_frag[ch][2] = -INFINITY; }
      if (col1 < k_start || col1 >= seqlen_k) { s_frag[ch][1] = -INFINITY; s_frag[ch][3] = -INFINITY; }
      m0 = max_ftz_f32(m0, max_ftz_f32(s_frag[ch][0], s_frag[ch][1]));
      m1 = max_ftz_f32(m1, max_ftz_f32(s_frag[ch][2], s_frag[ch][3]));
    }
    m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    if (tid_in_grp == 0) {
      smem.warp_max[warp_id][group_id]     = m0;
      smem.warp_max[warp_id][group_id + 8] = m1;
    }
    __syncthreads();

    const float tile_max0 = max_ftz_f32(max_ftz_f32(smem.warp_max[0][group_id], smem.warp_max[1][group_id]),
                                        max_ftz_f32(smem.warp_max[2][group_id], smem.warp_max[3][group_id]));
    const float tile_max1 = max_ftz_f32(max_ftz_f32(smem.warp_max[0][group_id + 8], smem.warp_max[1][group_id + 8]),
                                        max_ftz_f32(smem.warp_max[2][group_id + 8], smem.warp_max[3][group_id + 8]));
    float sc0 = 1.0f, sc1 = 1.0f;
    if (nb == nb_max - 1) {
      row_max0 = tile_max0;
      row_max1 = tile_max1;
    } else {
      const float prev0 = row_max0, prev1 = row_max1;
      row_max0 = max_ftz_f32(prev0, tile_max0);
      row_max1 = max_ftz_f32(prev1, tile_max1);
      const float cur0 = (row_max0 == -INFINITY) ? 0.0f : row_max0;
      const float cur1 = (row_max1 == -INFINITY) ? 0.0f : row_max1;
      sc0 = fast_exp2f(mul_ftz_f32(scale_log2, sub_ftz_f32(prev0, cur0)));
      sc1 = fast_exp2f(mul_ftz_f32(scale_log2, sub_ftz_f32(prev1, cur1)));
      if (warp_id < kActiveVWarps) {
        o_frag[0] = mul_ftz_f32(o_frag[0], sc0);
        o_frag[1] = mul_ftz_f32(o_frag[1], sc0);
        o_frag[2] = mul_ftz_f32(o_frag[2], sc1);
        o_frag[3] = mul_ftz_f32(o_frag[3], sc1);
      }
    }

    const float max_sc0 = (row_max0 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max0);
    const float max_sc1 = (row_max1 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max1);
    #pragma unroll
    for (int ch = 0; ch < kChunks; ++ch) {
      const float p0 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][0], scale_log2), max_sc0));
      const float p1 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][1], scale_log2), max_sc0));
      const float p2 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][2], scale_log2), max_sc1));
      const float p3 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][3], scale_log2), max_sc1));
      const int ni = (ch << 2) + warp_id;
      smem.p_f32[ni * 32 + lane] = make_float4(p0, p1, p2, p3);
      const __nv_bfloat162 pb0 = __floats2bfloat162_rn(p0, p1);
      const __nv_bfloat162 pb1 = __floats2bfloat162_rn(p2, p3);
      const int c_phys_p = ni ^ group_id;
      reinterpret_cast<uint32_t*>(&smem.p[group_id][c_phys_p])[tid_in_grp] =
          *reinterpret_cast<const uint32_t*>(&pb0);
      reinterpret_cast<uint32_t*>(&smem.p[group_id + 8][c_phys_p])[tid_in_grp] =
          *reinterpret_cast<const uint32_t*>(&pb1);
    }

    cp_async_wait<0>();
    __syncthreads();

    if (warp_id == kSumWarp) {
      const float4 pv0 = smem.p_f32[0 * 32 + lane];
      if (nb == nb_max - 1) {
        row_sum0 = add_ftz_f32(pv0.x, pv0.y);
        row_sum1 = add_ftz_f32(pv0.z, pv0.w);
      } else {
        row_sum0 = add_ftz_f32(fma_rn_ftz_f32(row_sum0, sc0, pv0.x), pv0.y);
        row_sum1 = add_ftz_f32(fma_rn_ftz_f32(row_sum1, sc1, pv0.z), pv0.w);
      }
      #pragma unroll
      for (int ni = 1; ni < (kChunks << 2); ++ni) {
        const float4 pv = smem.p_f32[ni * 32 + lane];
        row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, pv.x), pv.y);
        row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, pv.z), pv.w);
      }
    }
    __syncthreads();

    if (nb > nb_min) {
      #pragma unroll
      for (int ch = 0; ch < kChunks; ++ch) {
        issue_kv_tile_async<false>(smem.k[ch], kv_ptr, block_table, ((nb - 1) << kShift) + (ch << 5),
                                   k_start, seqlen_k, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
        cp_async_commit();
      }
    }

    if (warp_id < kActiveVWarps) {
      #pragma unroll
      for (int ks_p = 0; ks_p < kHalfBlocks; ++ks_p) {
        uint32_t p_frag[4];
        uint32_t v_frag[2];
        ldsm4_swizzled(p_frag, smem.p, ks_p, lane);
        ldsm2_v_trans_dsplit<kBlockN, kVRowStride>(v_frag, smem.v, warp_id, ks_p, lane);
        mma_m16n8k16_bf16(p_frag, v_frag, o_frag);
      }
    }
    __syncthreads();
  }

  if (warp_id == kSumWarp) {
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 2));
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 1));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 2));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 1));
    const float inv0 = (row_sum0 == 0.0f || row_sum0 != row_sum0) ? 1.0f : fast_rcpf(row_sum0);
    const float inv1 = (row_sum1 == 0.0f || row_sum1 != row_sum1) ? 1.0f : fast_rcpf(row_sum1);
    if (tid_in_grp == 0) {
      smem.warp_sum[group_id]     = inv0;
      smem.warp_sum[group_id + 8] = inv1;
    }
  }
  __syncthreads();

  if (warp_id < kActiveVWarps) {
    const float inv0 = smem.warp_sum[group_id];
    const float inv1 = smem.warp_sum[group_id + 8];
    const int d_col = d_cta * (kActiveVWarps * 8) + (warp_id << 3) + (tid_in_grp << 1);
    const __nv_bfloat162 ob0 = __floats2bfloat162_rn(
        mul_ftz_f32(o_frag[0], inv0), mul_ftz_f32(o_frag[1], inv0));
    const __nv_bfloat162 ob1 = __floats2bfloat162_rn(
        mul_ftz_f32(o_frag[2], inv1), mul_ftz_f32(o_frag[3], inv1));
    *reinterpret_cast<uint32_t*>(
        out_ptr + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) =
        *reinterpret_cast<const uint32_t*>(&ob0);
    *reinterpret_cast<uint32_t*>(
        out_ptr + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) =
        *reinterpret_cast<const uint32_t*>(&ob1);
  }
}

// ============================================================================
// Exact multi-split FA2 kernel (kBlockN = 128 + exact butterfly LSE combine)
// ============================================================================
__device__ __forceinline__ void combine_splits_fa2_exact(
    SharedStorage& smem,
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

  float (*w_smem)[kMaxSplits] = reinterpret_cast<float (*)[kMaxSplits]>(smem.p);
  uint64_t* valid_mask_smem   = reinterpret_cast<uint64_t*>(smem.warp_max);

  const int warp_id = tid >> 5;
  const int lane    = tid & 31;
  const float* lse_head = lse_partial + kv_head * kMaxSplits * 16;
  const float* o_head   = o_partial   + kv_head * kMaxSplits * 16 * 128;

  // Power-of-2 max_splits matching FA2 kLogMaxSplits (2, 4, 8, 16, 32, or 64).
  int max_splits = 2;
  while (max_splits < active_splits) max_splits <<= 1;

  for (int q_local = warp_id; q_local < 16; q_local += 4) {
    if (max_splits <= 32) {
      const float lse_orig = (lane < active_splits)
          ? ld_cg_f32(lse_head + lane * 16 + q_local)
          : -INFINITY;
      float lse_max = lse_orig;
      for (int mask = max_splits >> 1; mask >= 1; mask >>= 1) {
        lse_max = max_ftz_f32(lse_max, __shfl_xor_sync(0xffffffffu, lse_max, mask));
      }
      const float lse_max_safe = (lse_max == -INFINITY) ? 0.0f : lse_max;
      float sum_exp = fast_exp2f(mul_ftz_f32(sub_ftz_f32(lse_orig, lse_max_safe), kLog2e));
      for (int mask = max_splits >> 1; mask >= 1; mask >>= 1) {
        sum_exp = add_ftz_f32(sum_exp, __shfl_xor_sync(0xffffffffu, sum_exp, mask));
      }
      const float lse_logsum = (sum_exp == 0.0f || sum_exp != sum_exp)
          ? INFINITY
          : fma_rn_ftz_f32(fast_log2f(sum_exp), kLn2, lse_max_safe);
      const bool valid = (lane < active_splits) && (lse_orig != -INFINITY);
      const uint32_t vmask = __ballot_sync(0xffffffffu, valid);
      if (lane == 0) {
        valid_mask_smem[q_local] = static_cast<uint64_t>(vmask);
      }
      if (lane < active_splits) {
        w_smem[q_local][lane] = fast_exp2f(mul_ftz_f32(sub_ftz_f32(lse_orig, lse_logsum), kLog2e));
      }
    } else {
      const int s0 = lane << 1;
      const int s1 = s0 + 1;
      const float l0 = (s0 < active_splits) ? ld_cg_f32(lse_head + s0 * 16 + q_local) : -INFINITY;
      const float l1 = (s1 < active_splits) ? ld_cg_f32(lse_head + s1 * 16 + q_local) : -INFINITY;
      float lse_max = max_ftz_f32(l0, l1);
      #pragma unroll
      for (int mask = 16; mask >= 1; mask >>= 1) {
        lse_max = max_ftz_f32(lse_max, __shfl_xor_sync(0xffffffffu, lse_max, mask));
      }
      const float lse_max_safe = (lse_max == -INFINITY) ? 0.0f : lse_max;
      const float e0 = fast_exp2f(mul_ftz_f32(sub_ftz_f32(l0, lse_max_safe), kLog2e));
      const float e1 = fast_exp2f(mul_ftz_f32(sub_ftz_f32(l1, lse_max_safe), kLog2e));
      float sum_exp = add_ftz_f32(e0, e1);
      #pragma unroll
      for (int mask = 16; mask >= 1; mask >>= 1) {
        sum_exp = add_ftz_f32(sum_exp, __shfl_xor_sync(0xffffffffu, sum_exp, mask));
      }
      const float lse_logsum = (sum_exp == 0.0f || sum_exp != sum_exp)
          ? INFINITY
          : fma_rn_ftz_f32(fast_log2f(sum_exp), kLn2, lse_max_safe);
      const uint32_t vm0 = __ballot_sync(0xffffffffu, (s0 < active_splits) && (l0 != -INFINITY));
      const uint32_t vm1 = __ballot_sync(0xffffffffu, (s1 < active_splits) && (l1 != -INFINITY));
      if (lane == 0) {
        uint64_t mask64 = 0ull;
        for (int b = 0; b < 32; ++b) {
          if ((vm0 >> b) & 1u) mask64 |= (1ull << (2 * b));
          if ((vm1 >> b) & 1u) mask64 |= (1ull << (2 * b + 1));
        }
        valid_mask_smem[q_local] = mask64;
      }
      if (s0 < active_splits) {
        w_smem[q_local][s0] = fast_exp2f(mul_ftz_f32(sub_ftz_f32(l0, lse_logsum), kLog2e));
      }
      if (s1 < active_splits) {
        w_smem[q_local][s1] = fast_exp2f(mul_ftz_f32(sub_ftz_f32(l1, lse_logsum), kLog2e));
      }
    }
  }
  __syncthreads();

  const int q_local = tid >> 3;
  const int d_base  = (tid & 7) << 4;
  const uint64_t vmask = valid_mask_smem[q_local];

  float acc[16] = {0.f};
  for (int s = 0; s < active_splits; ++s) {
    const float w = w_smem[q_local][s];
    if ((vmask >> s) & 1ull) {
      const float* src = o_head + (s * 16 + q_local) * 128 + d_base;
      const float4 v0 = ld_cg_f32x4(src + 0);
      const float4 v1 = ld_cg_f32x4(src + 4);
      const float4 v2 = ld_cg_f32x4(src + 8);
      const float4 v3 = ld_cg_f32x4(src + 12);
      acc[0]  = fma_rn_ftz_f32(w, v0.x, acc[0]);  acc[1]  = fma_rn_ftz_f32(w, v0.y, acc[1]);
      acc[2]  = fma_rn_ftz_f32(w, v0.z, acc[2]);  acc[3]  = fma_rn_ftz_f32(w, v0.w, acc[3]);
      acc[4]  = fma_rn_ftz_f32(w, v1.x, acc[4]);  acc[5]  = fma_rn_ftz_f32(w, v1.y, acc[5]);
      acc[6]  = fma_rn_ftz_f32(w, v1.z, acc[6]);  acc[7]  = fma_rn_ftz_f32(w, v1.w, acc[7]);
      acc[8]  = fma_rn_ftz_f32(w, v2.x, acc[8]);  acc[9]  = fma_rn_ftz_f32(w, v2.y, acc[9]);
      acc[10] = fma_rn_ftz_f32(w, v2.z, acc[10]); acc[11] = fma_rn_ftz_f32(w, v2.w, acc[11]);
      acc[12] = fma_rn_ftz_f32(w, v3.x, acc[12]); acc[13] = fma_rn_ftz_f32(w, v3.y, acc[13]);
      acc[14] = fma_rn_ftz_f32(w, v3.z, acc[14]); acc[15] = fma_rn_ftz_f32(w, v3.w, acc[15]);
    } else {
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        acc[i] = fma_rn_ftz_f32(w, 0.0f, acc[i]);
      }
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
  load_q_group_to_regs(q_frag, smem.kv[0], q_ptr, kv_head, q_head_stride, tid, lane);

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
        m0 = max_ftz_f32(m0, max_ftz_f32(s_frag[ch][0], s_frag[ch][1]));
        m1 = max_ftz_f32(m1, max_ftz_f32(s_frag[ch][2], s_frag[ch][3]));
      }
      m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
      m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
      m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
      m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
      if (tid_in_grp == 0) {
        smem.warp_max[warp_id][group_id]     = m0;
        smem.warp_max[warp_id][group_id + 8] = m1;
      }
      __syncthreads();

      const float tile_max0 = max_ftz_f32(max_ftz_f32(smem.warp_max[0][group_id], smem.warp_max[1][group_id]),
                                          max_ftz_f32(smem.warp_max[2][group_id], smem.warp_max[3][group_id]));
      const float tile_max1 = max_ftz_f32(max_ftz_f32(smem.warp_max[0][group_id + 8], smem.warp_max[1][group_id + 8]),
                                          max_ftz_f32(smem.warp_max[2][group_id + 8], smem.warp_max[3][group_id + 8]));
      float sc0 = 1.0f, sc1 = 1.0f;
      if (nb == nb_max - 1) {
        row_max0 = tile_max0;
        row_max1 = tile_max1;
      } else {
        const float prev0 = row_max0, prev1 = row_max1;
        row_max0 = max_ftz_f32(prev0, tile_max0);
        row_max1 = max_ftz_f32(prev1, tile_max1);
        const float cur0 = (row_max0 == -INFINITY) ? 0.0f : row_max0;
        const float cur1 = (row_max1 == -INFINITY) ? 0.0f : row_max1;
        sc0 = fast_exp2f(mul_ftz_f32(scale_log2, sub_ftz_f32(prev0, cur0)));
        sc1 = fast_exp2f(mul_ftz_f32(scale_log2, sub_ftz_f32(prev1, cur1)));
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
      float4* p_f32_smem = reinterpret_cast<float4*>(smem.kv[2]);
      #pragma unroll
      for (int ch = 0; ch < 4; ++ch) {
        const float p0 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][0], scale_log2), max_sc0));
        const float p1 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][1], scale_log2), max_sc0));
        const float p2 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][2], scale_log2), max_sc1));
        const float p3 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][3], scale_log2), max_sc1));
        const int ni = (ch << 2) + warp_id;
        p_f32_smem[ni * 32 + lane] = make_float4(p0, p1, p2, p3);
        const __nv_bfloat162 pb0 = __floats2bfloat162_rn(p0, p1);
        const __nv_bfloat162 pb1 = __floats2bfloat162_rn(p2, p3);
        const int c_phys_p = ni ^ group_id;
        reinterpret_cast<uint32_t*>(&smem.p[group_id][c_phys_p])[tid_in_grp] =
            *reinterpret_cast<const uint32_t*>(&pb0);
        reinterpret_cast<uint32_t*>(&smem.p[group_id + 8][c_phys_p])[tid_in_grp] =
            *reinterpret_cast<const uint32_t*>(&pb1);
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
        if (ch == 0 && warp_id == 0) {
          const float4 pv0 = p_f32_smem[0 * 32 + lane];
          if (nb == nb_max - 1) {
            row_sum0 = add_ftz_f32(pv0.x, pv0.y);
            row_sum1 = add_ftz_f32(pv0.z, pv0.w);
          } else {
            row_sum0 = add_ftz_f32(fma_rn_ftz_f32(row_sum0, sc0, pv0.x), pv0.y);
            row_sum1 = add_ftz_f32(fma_rn_ftz_f32(row_sum1, sc1, pv0.z), pv0.w);
          }
          #pragma unroll
          for (int ni = 1; ni < 16; ++ni) {
            const float4 pv = p_f32_smem[ni * 32 + lane];
            row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, pv.x), pv.y);
            row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, pv.z), pv.w);
          }
        }
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
      row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 2));
      row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 1));
      row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 2));
      row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 1));
      const bool empty0 = (row_sum0 == 0.0f || row_sum0 != row_sum0 || row_max0 == -INFINITY);
      const bool empty1 = (row_sum1 == 0.0f || row_sum1 != row_sum1 || row_max1 == -INFINITY);
      const float inv0 = empty0 ? 1.0f : fast_rcpf(row_sum0);
      const float inv1 = empty1 ? 1.0f : fast_rcpf(row_sum1);
      if (tid_in_grp == 0) {
        smem.warp_sum[0][group_id]     = inv0;
        smem.warp_sum[0][group_id + 8] = inv1;
        if (fa2_splits > 1) {
          const float lse0 = empty0 ? -INFINITY
                                    : fma_rn_ftz_f32(row_max0, softmax_scale, mul_ftz_f32(fast_log2f(row_sum0), kLn2));
          const float lse1 = empty1 ? -INFINITY
                                    : fma_rn_ftz_f32(row_max1, softmax_scale, mul_ftz_f32(fast_log2f(row_sum1), kLn2));
          float* lse_slot = lse_partial + (kv_head * kMaxSplits + s_slot) * 16;
          st_cg_f32(lse_slot + group_id,     lse0);
          st_cg_f32(lse_slot + group_id + 8, lse1);
        }
      }
    }
    __syncthreads();

    const float inv0 = smem.warp_sum[0][group_id];
    const float inv1 = smem.warp_sum[0][group_id + 8];

    if (fa2_splits == 1) {
      #pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        const int d_col = (warp_id << 5) + (dt << 3) + (tid_in_grp << 1);
        const __nv_bfloat162 ob0 = __floats2bfloat162_rn(
            mul_ftz_f32(o_frag[dt][0], inv0), mul_ftz_f32(o_frag[dt][1], inv0));
        const __nv_bfloat162 ob1 = __floats2bfloat162_rn(
            mul_ftz_f32(o_frag[dt][2], inv1), mul_ftz_f32(o_frag[dt][3], inv1));
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
      st_cg_f32x2(o_slot + group_id * 128 + d_col,
                  mul_ftz_f32(o_frag[dt][0], inv0), mul_ftz_f32(o_frag[dt][1], inv0));
      st_cg_f32x2(o_slot + (group_id + 8) * 128 + d_col,
                  mul_ftz_f32(o_frag[dt][2], inv1), mul_ftz_f32(o_frag[dt][3], inv1));
    }
    __syncthreads();
  }

  if (fa2_splits > 1) {
    combine_splits_fa2_exact(
        smem, out_ptr, o_partial, lse_partial, split_cnt,
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
  const bool use_scale_mode3 = ((mode & 3) == 3);
  const bool use_k128_single = ((mode & 4) != 0);
  const bool use_1pass_dsplit = ((mode & 8) != 0);
  const float scale_log2 =
      use_scale_mode3
          ? (softmax_scale * muse_gqa16::kLog2e)
          : static_cast<float>(static_cast<double>(softmax_scale) * 1.44269504088896340736);
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

  if (fa2_splits == 1 && use_1pass_dsplit && (grid_splits == 4 || grid_splits == 8 || grid_splits == 16)) {
    const dim3 grid(static_cast<unsigned>(grid_splits), 2u, 1u);
    if (use_k128_single) {
      if (grid_splits == 8) {
        muse_gqa16::gqa16_1pass_dsplit_kernel<128, 8><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_ptr), static_cast<const __nv_bfloat16*>(kv_ptr),
            static_cast<__nv_bfloat16*>(out_ptr), static_cast<const int32_t*>(seqused_k_ptr),
            static_cast<const int32_t*>(block_table_ptr), block_size, window_left, scale_log2,
            q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head);
      } else if (grid_splits == 4) {
        muse_gqa16::gqa16_1pass_dsplit_kernel<128, 4><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_ptr), static_cast<const __nv_bfloat16*>(kv_ptr),
            static_cast<__nv_bfloat16*>(out_ptr), static_cast<const int32_t*>(seqused_k_ptr),
            static_cast<const int32_t*>(block_table_ptr), block_size, window_left, scale_log2,
            q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head);
      } else {
        muse_gqa16::gqa16_1pass_dsplit_kernel<128, 16><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_ptr), static_cast<const __nv_bfloat16*>(kv_ptr),
            static_cast<__nv_bfloat16*>(out_ptr), static_cast<const int32_t*>(seqused_k_ptr),
            static_cast<const int32_t*>(block_table_ptr), block_size, window_left, scale_log2,
            q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head);
      }
    } else {
      if (grid_splits == 8) {
        muse_gqa16::gqa16_1pass_dsplit_kernel<64, 8><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_ptr), static_cast<const __nv_bfloat16*>(kv_ptr),
            static_cast<__nv_bfloat16*>(out_ptr), static_cast<const int32_t*>(seqused_k_ptr),
            static_cast<const int32_t*>(block_table_ptr), block_size, window_left, scale_log2,
            q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head);
      } else if (grid_splits == 4) {
        muse_gqa16::gqa16_1pass_dsplit_kernel<64, 4><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_ptr), static_cast<const __nv_bfloat16*>(kv_ptr),
            static_cast<__nv_bfloat16*>(out_ptr), static_cast<const int32_t*>(seqused_k_ptr),
            static_cast<const int32_t*>(block_table_ptr), block_size, window_left, scale_log2,
            q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head);
      } else {
        muse_gqa16::gqa16_1pass_dsplit_kernel<64, 16><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_ptr), static_cast<const __nv_bfloat16*>(kv_ptr),
            static_cast<__nv_bfloat16*>(out_ptr), static_cast<const int32_t*>(seqused_k_ptr),
            static_cast<const int32_t*>(block_table_ptr), block_size, window_left, scale_log2,
            q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head);
      }
    }
    return static_cast<int>(cudaGetLastError());
  }

  if (fa2_splits == 1 && !use_k128_single) {
    if (grid_splits == 8) {
      muse_gqa16::gqa16_pass1_qk_max_kernel<64><<<dim3(8u, 2u, 1u), dim3(128u, 1u, 1u), 0, stream>>>(
          static_cast<const __nv_bfloat16*>(q_ptr),
          static_cast<const __nv_bfloat16*>(kv_ptr),
          static_cast<float*>(o_partial_ptr),
          static_cast<float*>(lse_partial_ptr),
          static_cast<const int32_t*>(seqused_k_ptr),
          static_cast<const int32_t*>(block_table_ptr),
          block_size, window_left,
          q_head_stride, stride_block, stride_slot, stride_kv_head);
      muse_gqa16::gqa16_pass2_pv_dim_kernel<64><<<dim3(8u, 2u, 1u), dim3(64u, 1u, 1u), 0, stream>>>(
          static_cast<const __nv_bfloat16*>(kv_ptr),
          static_cast<__nv_bfloat16*>(out_ptr),
          static_cast<const float*>(o_partial_ptr),
          static_cast<const float*>(lse_partial_ptr),
          static_cast<const int32_t*>(seqused_k_ptr),
          static_cast<const int32_t*>(block_table_ptr),
          block_size, window_left, scale_log2,
          out_head_stride, stride_block, stride_slot, stride_kv_head);
    } else {
      muse_gqa16::gqa16_fa2_k64_single_kernel<<<dim3(1u, 2u, 1u), dim3(128u, 1u, 1u), 0, stream>>>(
          static_cast<const __nv_bfloat16*>(q_ptr),
          static_cast<const __nv_bfloat16*>(kv_ptr),
          static_cast<__nv_bfloat16*>(out_ptr),
          static_cast<const int32_t*>(seqused_k_ptr),
          static_cast<const int32_t*>(block_table_ptr),
          block_size, window_left, scale_log2,
          q_head_stride, out_head_stride,
          stride_block, stride_slot, stride_kv_head);
    }
  } else if (fa2_splits == 1 && use_k128_single && grid_splits == 8) {
    muse_gqa16::gqa16_pass1_qk_max_kernel<128><<<dim3(8u, 2u, 1u), dim3(128u, 1u, 1u), 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q_ptr),
        static_cast<const __nv_bfloat16*>(kv_ptr),
        static_cast<float*>(o_partial_ptr),
        static_cast<float*>(lse_partial_ptr),
        static_cast<const int32_t*>(seqused_k_ptr),
        static_cast<const int32_t*>(block_table_ptr),
        block_size, window_left,
        q_head_stride, stride_block, stride_slot, stride_kv_head);
    muse_gqa16::gqa16_pass2_pv_dim_kernel<128><<<dim3(8u, 2u, 1u), dim3(64u, 1u, 1u), 0, stream>>>(
        static_cast<const __nv_bfloat16*>(kv_ptr),
        static_cast<__nv_bfloat16*>(out_ptr),
        static_cast<const float*>(o_partial_ptr),
        static_cast<const float*>(lse_partial_ptr),
        static_cast<const int32_t*>(seqused_k_ptr),
        static_cast<const int32_t*>(block_table_ptr),
        block_size, window_left, scale_log2,
        out_head_stride, stride_block, stride_slot, stride_kv_head);
  } else {
    const dim3 grid(static_cast<unsigned>(grid_splits), 2u, 1u);
    muse_gqa16::gqa16_fa2_block128_kernel<<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
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
