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
#include <cstdlib>

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
    struct {
      uint4 k_keep[(kStagesK > 1) ? (kStagesK - 1) : 1][32][16];
      float4 p_f32[kBlockN << 2]; // 4 or 8 KiB (aliases only k[kStagesK - 1])
    };
  };
  uint4 p[16][16];                // 4 KiB swizzled BF16 P
  uint4 v[kBlockN][kVRowStride];  // 2..10 KiB conflict-free V slice
  float warp_max[4][16];          // 256 B
  float warp_sum[16];             // 64 B
};

struct __align__(128) Pipe128Storage {
  union {
    uint4 kv[5][32][16];          // 5 x 8 KiB = 40,960 B (4 K tiles + V[0] in flight simultaneously)
    struct {
      uint4 kv_keep[3][32][16];   // 24,576 B (kv[0..2] hold V[1..3] while p_f32 aliases kv[3])
      float4 p_f32[512];          // 8,192 B
    };
  };
  uint4 p[16][16];                // 4,096 B
  float warp_max[4][16];          // 256 B
  float warp_sum[16];             // 64 B
};                                // Total = 45,440 B (< 48 KiB static shared memory limit)

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

__device__ __forceinline__ uint64_t kv_pol_evict_last() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

template <int kKvL2 = 0>
__device__ __forceinline__ void cp_async_16_l2(uint32_t smem_dst, const void* gmem_src, uint64_t l2_pol) {
  if constexpr (kKvL2 == 2) {
    asm volatile("cp.async.cg.shared.global.L2::256B.L2::cache_hint [%0], [%1], 16, %2;\n"
                 :: "r"(smem_dst), "l"(gmem_src), "l"(l2_pol));
  } else if constexpr (kKvL2 == 1) {
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n"
                 :: "r"(smem_dst), "l"(gmem_src));
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_dst), "l"(gmem_src));
  }
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
    int tid,
    int leaf_remap_base = -1,
    int leaf_p_start = 0,
    uint32_t leaf_remap_packed = 0u) {
  const bool is_bs64_aligned = (block_size == 64) && ((tok_base & 31) == 0);
  int phys_blk0 = 0, slot_base0 = 0;
  const bool half0_any = (tok_base < seqlen_k) && (tok_base + 16 > k_start);
  if (half0_any) {
    const int lb0 = (block_size == 16) ? (tok_base >> 4)
                  : ((block_size == 64) ? (tok_base >> 6) : (tok_base / block_size));
    slot_base0 = (block_size == 16) ? 0
               : ((block_size == 64) ? (tok_base & 63) : (tok_base - lb0 * block_size));
    phys_blk0 = __ldg(block_table + lb0);
  }
  int phys_blk1 = phys_blk0, slot_base1 = slot_base0 + 16;
  const int tok_hi = tok_base + 16;
  const bool half1_any = (tok_hi < seqlen_k) && (tok_hi + 16 > k_start);
  if (half1_any && !is_bs64_aligned) {
    const int lb1 = (block_size == 16) ? (tok_hi >> 4)
                  : ((block_size == 64) ? (tok_hi >> 6) : (tok_hi / block_size));
    slot_base1 = (block_size == 16) ? 0
               : ((block_size == 64) ? (tok_hi & 63) : (tok_hi - lb1 * block_size));
    phys_blk1 = __ldg(block_table + lb1);
  }

  const int64_t head_v_off = static_cast<int64_t>(kv_head) * stride_kv_head + (IsV ? 128 : 0);
  const bool has_remap = (leaf_remap_base >= 0) && (tok_base + 32 > leaf_remap_base);
  const bool full_valid = !has_remap && (tok_base >= k_start) && (tok_base + 32 <= seqlen_k) &&
                          ((block_size == 16) || is_bs64_aligned ||
                           (slot_base0 + 16 <= block_size && slot_base1 + 16 <= block_size));

  if (full_valid) {
    const int64_t base0 = static_cast<int64_t>(phys_blk0) * stride_block +
                          static_cast<int64_t>(slot_base0) * stride_slot + head_v_off;
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
    }
  } else {
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int idx = (u << 7) + tid;
      const int r = idx >> 4;
      const int c = idx & 15;
      const int tok = tok_base + r;
      if (tok >= k_start && tok < seqlen_k) {
        const int tok_eff = (leaf_remap_base >= 0 && tok >= leaf_remap_base)
            ? (leaf_p_start + static_cast<int>((leaf_remap_packed >> ((tok - leaf_remap_base) * 5)) & 0x1Fu))
            : tok;
        const int lb = (block_size == 16) ? (tok_eff >> 4)
                     : ((block_size == 64) ? (tok_eff >> 6) : (tok_eff / block_size));
        const int slot = (block_size == 16) ? (tok_eff & 15)
                       : ((block_size == 64) ? (tok_eff & 63) : (tok_eff - lb * block_size));
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
      const int lb = (block_size == 16) ? (tok >> 4)
                   : ((block_size == 64) ? (tok >> 6) : (tok / block_size));
      const int slot = (block_size == 16) ? (tok & 15)
                     : ((block_size == 64) ? (tok & 63) : (tok - lb * block_size));
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
      const int lb = (block_size == 16) ? (tok >> 4)
                   : ((block_size == 64) ? (tok >> 6) : (tok / block_size));
      const int slot = (block_size == 16) ? (tok & 15)
                     : ((block_size == 64) ? (tok & 63) : (tok - lb * block_size));
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

template <int kBlockN, int kDSplits, bool kMultiQ = false>
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
    int64_t stride_kv_head,
    const int32_t* __restrict__ cu_seqlens_q_ptr = nullptr,
    int64_t q_tok_stride = 0,
    int64_t out_tok_stride = 0) {
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

  int seqlen_k = seqused_k_ptr[0];
  if constexpr (kMultiQ) {
    const int q0 = cu_seqlens_q_ptr[0];
    const int actual_m = cu_seqlens_q_ptr[1] - q0;
    const int q_idx = blockIdx.z;
    if (q_idx >= actual_m) return;
    q_ptr += static_cast<int64_t>(q0 + q_idx) * q_tok_stride;
    out_ptr += static_cast<int64_t>(q0 + q_idx) * out_tok_stride;
    seqlen_k -= (actual_m - 1 - q_idx);
    if (seqlen_k < 0) seqlen_k = 0;
  }
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

template <int kDSplits, bool kMultiQ = false>
__global__ __launch_bounds__(128, 2) void gqa16_1pass_dsplit_pipe_kernel(
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
    int64_t stride_kv_head,
    const int32_t* __restrict__ cu_seqlens_q_ptr = nullptr,
    int64_t q_tok_stride = 0,
    int64_t out_tok_stride = 0) {
  static_assert(kDSplits == 4 || kDSplits == 8 || kDSplits == 16, "kDSplits must be 4, 8, or 16");
  constexpr int kBlockN       = 128;
  constexpr int kShift        = 7;
  constexpr int kHalfBlocks   = 8;
  constexpr int kActiveVWarps = 16 / kDSplits;
  constexpr int kVRowStride   = (kActiveVWarps == 1) ? 1 : (kActiveVWarps + 1);
  constexpr int kSumWarp      = (kActiveVWarps < 4) ? kActiveVWarps : 0;

  __shared__ OnePassDSplitStorage<kBlockN, kVRowStride, 4> smem;

  const int d_cta      = blockIdx.x;
  const int kv_head    = blockIdx.y;
  const int tid        = threadIdx.x;
  const int warp_id    = tid >> 5;
  const int lane       = tid & 31;
  const int group_id   = lane >> 2;
  const int tid_in_grp = lane & 3;

  int seqlen_k = seqused_k_ptr[0];
  if constexpr (kMultiQ) {
    const int q0 = cu_seqlens_q_ptr[0];
    const int actual_m = cu_seqlens_q_ptr[1] - q0;
    const int q_idx = blockIdx.z;
    if (q_idx >= actual_m) return;
    q_ptr += static_cast<int64_t>(q0 + q_idx) * q_tok_stride;
    out_ptr += static_cast<int64_t>(q0 + q_idx) * out_tok_stride;
    seqlen_k -= (actual_m - 1 - q_idx);
    if (seqlen_k < 0) seqlen_k = 0;
  }
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

  // Prologue: issue 4 staggered 8 KiB K groups so ch=0 starts after just 8 KiB!
  const int nb0 = nb_max - 1;
  #pragma unroll
  for (int ch = 0; ch < 4; ++ch) {
    issue_kv_tile_async<false>(smem.k[ch], kv_ptr, block_table, (nb0 << kShift) + (ch << 5),
                               k_start, seqlen_k, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();
  }

  for (int nb = nb0; nb >= nb_min; --nb) {
    const int block_tok_base = nb << kShift;
    float s_frag[4][4] = {0.f};

    // Issue V slice as group #5 so all 4 K chunks + V stream concurrently.
    issue_v_dsplit_async<kBlockN, kActiveVWarps, kVRowStride>(
        smem.v, kv_ptr, block_table, block_tok_base, k_start, seqlen_k,
        block_size, kv_head, d_cta, stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();

    #pragma unroll
    for (int ch = 0; ch < 4; ++ch) {
      if (ch == 0) cp_async_wait<4>();
      else if (ch == 1) cp_async_wait<3>();
      else if (ch == 2) cp_async_wait<2>();
      else cp_async_wait<1>();
      __syncthreads();
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t b_frag[2];
        ldsm2_k_swizzled(b_frag, smem.k[ch], warp_id, ks, lane);
        mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[ch]);
      }
    }

    // K[0..2] (24 KiB) are completely consumed and do NOT alias p_f32 (which aliases K[3] only)!
    // Prefetch next block's K[0..2] NOW so they stream during softmax + row_sum + P@V!
    if (nb > nb_min) {
      const int next_base = (nb - 1) << kShift;
      #pragma unroll
      for (int ch = 0; ch < 3; ++ch) {
        issue_kv_tile_async<false>(smem.k[ch], kv_ptr, block_table, next_base + (ch << 5),
                                   k_start, seqlen_k, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
        cp_async_commit();
      }
    }

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
    if (nb == nb0) {
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
    for (int ch = 0; ch < 4; ++ch) {
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

    // Wait for V slice (leaving the 3 next-block K[0..2] groups in flight if nb > nb_min).
    if (nb > nb_min) cp_async_wait<3>();
    else cp_async_wait<0>();
    __syncthreads();

    // Execute P @ V and row_sum concurrently in the same barrier epoch (no intermediate __syncthreads()!)
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

    if (warp_id == kSumWarp) {
      const float4 pv0 = smem.p_f32[0 * 32 + lane];
      if (nb == nb0) {
        row_sum0 = add_ftz_f32(pv0.x, pv0.y);
        row_sum1 = add_ftz_f32(pv0.z, pv0.w);
      } else {
        row_sum0 = add_ftz_f32(fma_rn_ftz_f32(row_sum0, sc0, pv0.x), pv0.y);
        row_sum1 = add_ftz_f32(fma_rn_ftz_f32(row_sum1, sc1, pv0.z), pv0.w);
      }
      #pragma unroll
      for (int ni = 1; ni < 16; ++ni) {
        const float4 pv = smem.p_f32[ni * 32 + lane];
        row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, pv.x), pv.y);
        row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, pv.z), pv.w);
      }
    }
    __syncthreads();

    // p_f32 (in k[3]) is now done -> issue next iteration's K[3] as the 4th staggered group!
    if (nb > nb_min) {
      issue_kv_tile_async<false>(smem.k[3], kv_ptr, block_table, ((nb - 1) << kShift) + (3 << 5),
                                 k_start, seqlen_k, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
    }
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

// ============================================================================
// 5-Stage Pipelined Exact Multi-Split FA2 Kernel (8 CTAs = 4/head default)
// Fits 32 (gate) + 8 (l2pf) + 8 (attn) = 48 SMs with zero wave preemption,
// immediate idle-CTA exit, zero empty-split GMEM writes/loads, and 2x unrolled combine.
// ============================================================================
__device__ __forceinline__ void combine_splits_pipe_exact(
    Pipe128Storage& smem,
    __nv_bfloat16* __restrict__ out_ptr,
    const float* __restrict__ o_partial,
    const float* __restrict__ lse_partial,
    int32_t* __restrict__ split_cnt,
    int kv_head,
    int active_workers,
    int fa2_splits,
    int s_begin,
    int s_end,
    int64_t out_head_stride,
    int tid) {
  if (active_workers > 1) {
    __threadfence();
    __syncthreads();
    int is_last = 0;
    if (tid == 0) {
      const int prev = atomicAdd(&split_cnt[kv_head], 1);
      if (prev == active_workers - 1) {
        atomicExch(&split_cnt[kv_head], 0);
        is_last = 1;
      }
    }
    if (!__syncthreads_or(is_last)) return;
  } else {
    __syncthreads();
  }

  float (*w_smem)[kMaxSplits] = reinterpret_cast<float (*)[kMaxSplits]>(smem.p);
  uint64_t* valid_mask_smem   = reinterpret_cast<uint64_t*>(smem.warp_max);

  const int warp_id = tid >> 5;
  const int lane    = tid & 31;
  const float* lse_head = lse_partial + kv_head * kMaxSplits * 16;
  const float* o_head   = o_partial   + kv_head * kMaxSplits * 16 * 128;

  int max_splits = 2;
  while (max_splits < fa2_splits) max_splits <<= 1;

  for (int q_local = warp_id; q_local < 16; q_local += 4) {
    if (max_splits <= 32) {
      const bool in_range = (lane >= s_begin && lane < s_end);
      const float lse_orig = in_range
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
      const bool valid = in_range && (lse_orig != -INFINITY);
      const uint32_t vmask = __ballot_sync(0xffffffffu, valid);
      if (lane == 0) {
        valid_mask_smem[q_local] = static_cast<uint64_t>(vmask);
      }
      if (in_range) {
        w_smem[q_local][lane] = fast_exp2f(mul_ftz_f32(sub_ftz_f32(lse_orig, lse_logsum), kLog2e));
      }
    } else {
      const int s0 = lane << 1;
      const int s1 = s0 + 1;
      const bool in0 = (s0 >= s_begin && s0 < s_end);
      const bool in1 = (s1 >= s_begin && s1 < s_end);
      const float l0 = in0 ? ld_cg_f32(lse_head + s0 * 16 + q_local) : -INFINITY;
      const float l1 = in1 ? ld_cg_f32(lse_head + s1 * 16 + q_local) : -INFINITY;
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
      const uint32_t vm0 = __ballot_sync(0xffffffffu, in0 && (l0 != -INFINITY));
      const uint32_t vm1 = __ballot_sync(0xffffffffu, in1 && (l1 != -INFINITY));
      if (lane == 0) {
        uint64_t mask64 = 0ull;
        for (int b = 0; b < 32; ++b) {
          if ((vm0 >> b) & 1u) mask64 |= (1ull << (2 * b));
          if ((vm1 >> b) & 1u) mask64 |= (1ull << (2 * b + 1));
        }
        valid_mask_smem[q_local] = mask64;
      }
      if (in0) {
        w_smem[q_local][s0] = fast_exp2f(mul_ftz_f32(sub_ftz_f32(l0, lse_logsum), kLog2e));
      }
      if (in1) {
        w_smem[q_local][s1] = fast_exp2f(mul_ftz_f32(sub_ftz_f32(l1, lse_logsum), kLog2e));
      }
    }
  }
  __syncthreads();

  const int q_local = tid >> 3;
  const int d_base  = (tid & 7) << 4;
  const uint64_t vmask = valid_mask_smem[q_local];

  float acc[16] = {0.f};
  int s = s_begin;
  for (; s + 1 < s_end; s += 2) {
    const float w0 = w_smem[q_local][s];
    const float w1 = w_smem[q_local][s + 1];
    const bool v0_ok = ((vmask >> s) & 1ull) != 0ull;
    const bool v1_ok = ((vmask >> (s + 1)) & 1ull) != 0ull;
    if (v0_ok && v1_ok) {
      const float* src0 = o_head + (s * 16 + q_local) * 128 + d_base;
      const float* src1 = o_head + ((s + 1) * 16 + q_local) * 128 + d_base;
      const float4 a0 = ld_cg_f32x4(src0 + 0);
      const float4 a1 = ld_cg_f32x4(src0 + 4);
      const float4 a2 = ld_cg_f32x4(src0 + 8);
      const float4 a3 = ld_cg_f32x4(src0 + 12);
      const float4 b0 = ld_cg_f32x4(src1 + 0);
      const float4 b1 = ld_cg_f32x4(src1 + 4);
      const float4 b2 = ld_cg_f32x4(src1 + 8);
      const float4 b3 = ld_cg_f32x4(src1 + 12);
      acc[0]  = fma_rn_ftz_f32(w0, a0.x, acc[0]);  acc[1]  = fma_rn_ftz_f32(w0, a0.y, acc[1]);
      acc[2]  = fma_rn_ftz_f32(w0, a0.z, acc[2]);  acc[3]  = fma_rn_ftz_f32(w0, a0.w, acc[3]);
      acc[4]  = fma_rn_ftz_f32(w0, a1.x, acc[4]);  acc[5]  = fma_rn_ftz_f32(w0, a1.y, acc[5]);
      acc[6]  = fma_rn_ftz_f32(w0, a1.z, acc[6]);  acc[7]  = fma_rn_ftz_f32(w0, a1.w, acc[7]);
      acc[8]  = fma_rn_ftz_f32(w0, a2.x, acc[8]);  acc[9]  = fma_rn_ftz_f32(w0, a2.y, acc[9]);
      acc[10] = fma_rn_ftz_f32(w0, a2.z, acc[10]); acc[11] = fma_rn_ftz_f32(w0, a2.w, acc[11]);
      acc[12] = fma_rn_ftz_f32(w0, a3.x, acc[12]); acc[13] = fma_rn_ftz_f32(w0, a3.y, acc[13]);
      acc[14] = fma_rn_ftz_f32(w0, a3.z, acc[14]); acc[15] = fma_rn_ftz_f32(w0, a3.w, acc[15]);

      acc[0]  = fma_rn_ftz_f32(w1, b0.x, acc[0]);  acc[1]  = fma_rn_ftz_f32(w1, b0.y, acc[1]);
      acc[2]  = fma_rn_ftz_f32(w1, b0.z, acc[2]);  acc[3]  = fma_rn_ftz_f32(w1, b0.w, acc[3]);
      acc[4]  = fma_rn_ftz_f32(w1, b1.x, acc[4]);  acc[5]  = fma_rn_ftz_f32(w1, b1.y, acc[5]);
      acc[6]  = fma_rn_ftz_f32(w1, b1.z, acc[6]);  acc[7]  = fma_rn_ftz_f32(w1, b1.w, acc[7]);
      acc[8]  = fma_rn_ftz_f32(w1, b2.x, acc[8]);  acc[9]  = fma_rn_ftz_f32(w1, b2.y, acc[9]);
      acc[10] = fma_rn_ftz_f32(w1, b2.z, acc[10]); acc[11] = fma_rn_ftz_f32(w1, b2.w, acc[11]);
      acc[12] = fma_rn_ftz_f32(w1, b3.x, acc[12]); acc[13] = fma_rn_ftz_f32(w1, b3.y, acc[13]);
      acc[14] = fma_rn_ftz_f32(w1, b3.z, acc[14]); acc[15] = fma_rn_ftz_f32(w1, b3.w, acc[15]);
    } else {
      for (int u = 0; u < 2; ++u) {
        const int su = s + u;
        const float wu = w_smem[q_local][su];
        if ((vmask >> su) & 1ull) {
          const float* src = o_head + (su * 16 + q_local) * 128 + d_base;
          const float4 v0 = ld_cg_f32x4(src + 0);
          const float4 v1 = ld_cg_f32x4(src + 4);
          const float4 v2 = ld_cg_f32x4(src + 8);
          const float4 v3 = ld_cg_f32x4(src + 12);
          acc[0]  = fma_rn_ftz_f32(wu, v0.x, acc[0]);  acc[1]  = fma_rn_ftz_f32(wu, v0.y, acc[1]);
          acc[2]  = fma_rn_ftz_f32(wu, v0.z, acc[2]);  acc[3]  = fma_rn_ftz_f32(wu, v0.w, acc[3]);
          acc[4]  = fma_rn_ftz_f32(wu, v1.x, acc[4]);  acc[5]  = fma_rn_ftz_f32(wu, v1.y, acc[5]);
          acc[6]  = fma_rn_ftz_f32(wu, v1.z, acc[6]);  acc[7]  = fma_rn_ftz_f32(wu, v1.w, acc[7]);
          acc[8]  = fma_rn_ftz_f32(wu, v2.x, acc[8]);  acc[9]  = fma_rn_ftz_f32(wu, v2.y, acc[9]);
          acc[10] = fma_rn_ftz_f32(wu, v2.z, acc[10]); acc[11] = fma_rn_ftz_f32(wu, v2.w, acc[11]);
          acc[12] = fma_rn_ftz_f32(wu, v3.x, acc[12]); acc[13] = fma_rn_ftz_f32(wu, v3.y, acc[13]);
          acc[14] = fma_rn_ftz_f32(wu, v3.z, acc[14]); acc[15] = fma_rn_ftz_f32(wu, v3.w, acc[15]);
        } else {
          #pragma unroll
          for (int i = 0; i < 16; ++i) acc[i] = fma_rn_ftz_f32(wu, 0.0f, acc[i]);
        }
      }
    }
  }
  for (; s < s_end; ++s) {
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
      for (int i = 0; i < 16; ++i) acc[i] = fma_rn_ftz_f32(w, 0.0f, acc[i]);
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

template <bool kMultiQDirect = false, bool kPdl = false>
__global__ __launch_bounds__(128, 2) void gqa16_fa2_block128_pipe_kernel(
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
    int64_t stride_kv_head,
    const int32_t* __restrict__ cu_seqlens_q_ptr = nullptr,
    int64_t q_tok_stride = 0,
    int64_t out_tok_stride = 0,
    const int32_t* __restrict__ leaf_flag_ptr = nullptr) {
  __shared__ Pipe128Storage smem;

  const int worker_idx  = blockIdx.x;
  const int grid_splits = gridDim.x;
  const int kv_head     = blockIdx.y;
  const int tid         = threadIdx.x;
  const int warp_id     = tid >> 5;
  const int lane        = tid & 31;
  const int group_id    = lane >> 2;
  const int tid_in_grp  = lane & 3;

  int seqlen_k = seqused_k_ptr[0];
  int leaf_remap_base = -1;
  int leaf_p_start = 0;
  uint32_t leaf_remap_packed = 0u;
  if constexpr (kMultiQDirect) {
    const int q0 = cu_seqlens_q_ptr[0];
    const int actual_m = cu_seqlens_q_ptr[1] - q0;
    const int q_idx = blockIdx.z;
    if (q_idx >= actual_m) return;
    q_ptr += static_cast<int64_t>(q0 + q_idx) * q_tok_stride;
    out_ptr += static_cast<int64_t>(q0 + q_idx) * out_tok_stride;
    if (leaf_flag_ptr != nullptr && leaf_flag_ptr[0] < 0 && actual_m <= 32 && seqlen_k >= actual_m) {
      // Generalized arbitrary-depth (1..6-level) bit-packed tree remap table (sentinel leaf_flag_ptr[0] == -1):
      // entry q_idx sits at [1 + 3*q_idx + {0: seq_sub, 1: base_sub, 2: remap_packed}] inside int32[49].
      // For base_sub >= 0, logical KV positions tok = (seqlen_k - base_sub) + j (j = 0..5) remap to
      // physical slot (seqlen_k - actual_m) + ((remap_packed >> (4*j)) & 0xF).
      const int base = 1 + q_idx * 3;
      const int seq_sub = leaf_flag_ptr[base];
      const int base_sub = leaf_flag_ptr[base + 1];
      if (base_sub >= 0) {
        leaf_p_start = seqlen_k - actual_m;
        leaf_remap_base = seqlen_k - base_sub;
        leaf_remap_packed = static_cast<uint32_t>(leaf_flag_ptr[base + 2]);
      }
      seqlen_k -= seq_sub;
    } else if (q_idx == 7 && actual_m == 8 && leaf_flag_ptr != nullptr) {
      const int lk = leaf_flag_ptr[0];
      if ((lk == 1 || lk == 2) && seqlen_k >= 8) {
        leaf_p_start = seqlen_k - 8;
        leaf_remap_base = leaf_p_start + lk;  // p + lk (exact logical position of row lk)
        leaf_remap_packed = 7u;               // nibble 0 -> physical row 7 (p + 7)
        seqlen_k = leaf_p_start + lk + 1;     // p + lk + 1 (0-ULP identical to row lk!)
      } else {
        seqlen_k -= (actual_m - 1 - q_idx);
      }
    } else if (q_idx == 6 && actual_m == 7 && leaf_flag_ptr != nullptr && leaf_flag_ptr[0] > 0 && seqlen_k >= 7) {
      leaf_p_start = seqlen_k - 7;
      leaf_remap_base = leaf_p_start + 1;     // p + 1 (exact logical position of row 1)
      leaf_remap_packed = 6u;                 // nibble 0 -> physical row 6 (p + 6)
      seqlen_k -= 5;                          // p + 2 (0-ULP identical to row 1!)
    } else {
      seqlen_k -= (actual_m - 1 - q_idx);
    }
    if (seqlen_k < 0) seqlen_k = 0;
  }
  const int k_start = (window_left >= 0)
      ? ((seqlen_k - 1 - window_left > 0) ? (seqlen_k - 1 - window_left) : 0)
      : 0;
  const int num_n_blocks = (seqlen_k > 0) ? ((seqlen_k + 127) >> 7) : 0;
  const int global_nb_min = (window_left >= 0 && seqlen_k > 0) ? (k_start >> 7) : 0;
  const int n_blocks_per_split = (num_n_blocks + fa2_splits - 1) / fa2_splits;

  int s_begin = 0;
  int s_end = 0;
  if (n_blocks_per_split > 0 && global_nb_min < num_n_blocks) {
    s_begin = global_nb_min / n_blocks_per_split;
    s_end   = (num_n_blocks + n_blocks_per_split - 1) / n_blocks_per_split;
    if (s_end > fa2_splits) s_end = fa2_splits;
  }

  const int num_nonempty = s_end - s_begin;
  if (num_nonempty <= 0) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    if (worker_idx == 0) {
      const int q_local = tid >> 3;
      const int d_base = (tid & 7) << 4;
      __nv_bfloat16* dst = out_ptr + static_cast<int64_t>(kv_head * 16 + q_local) * out_head_stride + d_base;
      *reinterpret_cast<uint4*>(dst)     = make_uint4(0u, 0u, 0u, 0u);
      *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0u, 0u, 0u, 0u);
    }
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  const int active_workers = (num_nonempty < grid_splits) ? num_nonempty : grid_splits;
  if (worker_idx >= active_workers) return;

  // SM121 PDL consumer barrier: wait for preceding qk_norm_rope_kvcache strictly after independent metadata setup.
  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  uint32_t q_frag[8][4];
  load_q_group_to_regs(q_frag, smem.kv[0], q_ptr, kv_head, q_head_stride, tid, lane);

  for (int idx = worker_idx; idx < num_nonempty; idx += active_workers) {
    const int s_slot = s_begin + idx;
    int nb_min = s_slot * n_blocks_per_split;
    int nb_max = (s_slot + 1) * n_blocks_per_split;
    if (nb_max > num_n_blocks) nb_max = num_n_blocks;
    if (nb_min < global_nb_min) nb_min = global_nb_min;

    float o_frag[4][4] = {0.f};
    float row_max0 = -INFINITY, row_max1 = -INFINITY;
    float row_sum0 = 0.0f, row_sum1 = 0.0f;

    for (int nb = nb_max - 1; nb >= nb_min; --nb) {
      const int block_tok_base = nb << 7;
      float s_frag[4][4] = {0.f};

      #pragma unroll
      for (int ch = 0; ch < 4; ++ch) {
        issue_kv_tile_async<false>(smem.kv[ch], kv_ptr, block_table, block_tok_base + (ch << 5),
                                   k_start, seqlen_k, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid,
                                   leaf_remap_base, leaf_p_start, leaf_remap_packed);
        cp_async_commit();
      }
      issue_kv_tile_async<true>(smem.kv[4], kv_ptr, block_table, block_tok_base,
                                k_start, seqlen_k, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid,
                                leaf_remap_base, leaf_p_start, leaf_remap_packed);
      cp_async_commit();

      cp_async_wait<4>();
      __syncthreads();
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t b_frag[2];
        ldsm2_k_swizzled(b_frag, smem.kv[0], warp_id, ks, lane);
        mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[0]);
      }
      __syncthreads();
      issue_kv_tile_async<true>(smem.kv[0], kv_ptr, block_table, block_tok_base + 32,
                                k_start, seqlen_k, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid,
                                leaf_remap_base, leaf_p_start, leaf_remap_packed);
      cp_async_commit();

      cp_async_wait<4>();
      __syncthreads();
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t b_frag[2];
        ldsm2_k_swizzled(b_frag, smem.kv[1], warp_id, ks, lane);
        mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[1]);
      }
      __syncthreads();
      issue_kv_tile_async<true>(smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
                                k_start, seqlen_k, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid,
                                leaf_remap_base, leaf_p_start, leaf_remap_packed);
      cp_async_commit();

      cp_async_wait<4>();
      __syncthreads();
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t b_frag[2];
        ldsm2_k_swizzled(b_frag, smem.kv[2], warp_id, ks, lane);
        mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[2]);
      }
      __syncthreads();
      issue_kv_tile_async<true>(smem.kv[2], kv_ptr, block_table, block_tok_base + 96,
                                k_start, seqlen_k, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid,
                                leaf_remap_base, leaf_p_start, leaf_remap_packed);
      cp_async_commit();

      cp_async_wait<4>();
      __syncthreads();
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t b_frag[2];
        ldsm2_k_swizzled(b_frag, smem.kv[3], warp_id, ks, lane);
        mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[3]);
      }

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
      #pragma unroll
      for (int ch = 0; ch < 4; ++ch) {
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

      cp_async_wait<2>();
      __syncthreads();

      if (warp_id == 0) {
        const float4 pv0 = smem.p_f32[0 * 32 + lane];
        if (nb == nb_max - 1) {
          row_sum0 = add_ftz_f32(pv0.x, pv0.y);
          row_sum1 = add_ftz_f32(pv0.z, pv0.w);
        } else {
          row_sum0 = add_ftz_f32(fma_rn_ftz_f32(row_sum0, sc0, pv0.x), pv0.y);
          row_sum1 = add_ftz_f32(fma_rn_ftz_f32(row_sum1, sc1, pv0.z), pv0.w);
        }
        #pragma unroll
        for (int ni = 1; ni < 16; ++ni) {
          const float4 pv = smem.p_f32[ni * 32 + lane];
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
          ldsm2_v_trans_swizzled(v_frag, smem.kv[4], warp_id, dt, ks_p, lane);
          mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
        }
      }
      #pragma unroll
      for (int ks_p = 0; ks_p < 2; ++ks_p) {
        uint32_t p_frag[4];
        ldsm4_swizzled(p_frag, smem.p, 2 + ks_p, lane);
        #pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
          uint32_t v_frag[2];
          ldsm2_v_trans_swizzled(v_frag, smem.kv[0], warp_id, dt, ks_p, lane);
          mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
        }
      }

      if constexpr (kPdl) {
        if (nb == nb_min && idx + active_workers >= num_nonempty) {
          asm volatile("griddepcontrol.launch_dependents;");
        }
      }

      cp_async_wait<0>();
      __syncthreads();

      #pragma unroll
      for (int ks_p = 0; ks_p < 2; ++ks_p) {
        uint32_t p_frag[4];
        ldsm4_swizzled(p_frag, smem.p, 4 + ks_p, lane);
        #pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
          uint32_t v_frag[2];
          ldsm2_v_trans_swizzled(v_frag, smem.kv[1], warp_id, dt, ks_p, lane);
          mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
        }
      }
      #pragma unroll
      for (int ks_p = 0; ks_p < 2; ++ks_p) {
        uint32_t p_frag[4];
        ldsm4_swizzled(p_frag, smem.p, 6 + ks_p, lane);
        #pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
          uint32_t v_frag[2];
          ldsm2_v_trans_swizzled(v_frag, smem.kv[2], warp_id, dt, ks_p, lane);
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
      const bool empty0 = (row_sum0 == 0.0f || row_sum0 != row_sum0 || row_max0 == -INFINITY);
      const bool empty1 = (row_sum1 == 0.0f || row_sum1 != row_sum1 || row_max1 == -INFINITY);
      const float inv0 = empty0 ? 1.0f : fast_rcpf(row_sum0);
      const float inv1 = empty1 ? 1.0f : fast_rcpf(row_sum1);
      if (tid_in_grp == 0) {
        smem.warp_sum[group_id]     = inv0;
        smem.warp_sum[group_id + 8] = inv1;
        if constexpr (!kMultiQDirect) {
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

    const float inv0 = smem.warp_sum[group_id];
    const float inv1 = smem.warp_sum[group_id + 8];
    if constexpr (kMultiQDirect) {
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
    } else {
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
  }

  if constexpr (!kMultiQDirect) {
    combine_splits_pipe_exact(
        smem, out_ptr, o_partial, lse_partial, split_cnt,
        kv_head, active_workers, fa2_splits, s_begin, s_end, out_head_stride, tid);
  }
}

// ============================================================================
// Shared-KV Multi-Q Pipelined FA2 Verify Kernel (kQBatch = 2 or 4 rows / CTA)
//
// Strictly 0-ULP bit-identical to gqa16_fa2_block128_pipe_kernel<true>:
//   - Cuts KV cache reads, block_table loads, cp.async, and K ldmatrix by kQBatch
//     across the entire shared prefix (nb < safe_tok_end >> 7).
//   - Tail block(s) (<= 1..2 blocks touching causal/tree leaf_flag[49] remap) run
//     first per query row with exact per-row (k_start, seqlen_k, leaf_remap).
//   - With kQBatch=4 and M=16 (2 KV heads), grid=(1, 2, 4) = 8 CTAs fits cleanly
//     alongside 32-SM output_gate + 8-SM l2pf = 48 SMs in 1 wave!
// ============================================================================
template <int kQBatch>
struct __align__(128) MultiQPipe128Storage {
  Pipe128Storage base;            // 45,440 B
  uint4 q[kQBatch][16][16];       // kQBatch * 4,096 B (61,824 B total for kQBatch=4)
};

__device__ __forceinline__ void pipe128_softmax_and_pv_step(
    Pipe128Storage& smem,
    float (&s_frag)[4][4],
    float (&o_frag)[4][4],
    float& row_max0,
    float& row_max1,
    float& row_sum0,
    float& row_sum1,
    bool is_first,
    int block_tok_base,
    int k_start,
    int seqlen_k,
    float scale_log2,
    int warp_id,
    int lane,
    int group_id,
    int tid_in_grp) {
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
  if (is_first) {
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
  #pragma unroll
  for (int ch = 0; ch < 4; ++ch) {
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
  __syncthreads();

  if (warp_id == 0) {
    const float4 pv0 = smem.p_f32[0 * 32 + lane];
    if (is_first) {
      row_sum0 = add_ftz_f32(pv0.x, pv0.y);
      row_sum1 = add_ftz_f32(pv0.z, pv0.w);
    } else {
      row_sum0 = add_ftz_f32(fma_rn_ftz_f32(row_sum0, sc0, pv0.x), pv0.y);
      row_sum1 = add_ftz_f32(fma_rn_ftz_f32(row_sum1, sc1, pv0.z), pv0.w);
    }
    #pragma unroll
    for (int ni = 1; ni < 16; ++ni) {
      const float4 pv = smem.p_f32[ni * 32 + lane];
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
      ldsm2_v_trans_swizzled(v_frag, smem.kv[4], warp_id, dt, ks_p, lane);
      mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
    }
  }
  #pragma unroll
  for (int ks_p = 0; ks_p < 2; ++ks_p) {
    uint32_t p_frag[4];
    ldsm4_swizzled(p_frag, smem.p, 2 + ks_p, lane);
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      uint32_t v_frag[2];
      ldsm2_v_trans_swizzled(v_frag, smem.kv[0], warp_id, dt, ks_p, lane);
      mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
    }
  }
  #pragma unroll
  for (int ks_p = 0; ks_p < 2; ++ks_p) {
    uint32_t p_frag[4];
    ldsm4_swizzled(p_frag, smem.p, 4 + ks_p, lane);
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      uint32_t v_frag[2];
      ldsm2_v_trans_swizzled(v_frag, smem.kv[1], warp_id, dt, ks_p, lane);
      mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
    }
  }
  #pragma unroll
  for (int ks_p = 0; ks_p < 2; ++ks_p) {
    uint32_t p_frag[4];
    ldsm4_swizzled(p_frag, smem.p, 6 + ks_p, lane);
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      uint32_t v_frag[2];
      ldsm2_v_trans_swizzled(v_frag, smem.kv[2], warp_id, dt, ks_p, lane);
      mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
    }
  }
  __syncthreads();
}

template <int kQBatch, bool kPdl = false>
__global__ __launch_bounds__(128, 1) void gqa16_fa2_block128_pipe_multiq_kernel(
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
    int64_t stride_kv_head,
    const int32_t* __restrict__ cu_seqlens_q_ptr,
    int64_t q_tok_stride,
    int64_t out_tok_stride,
    const int32_t* __restrict__ leaf_flag_ptr) {
  extern __shared__ __align__(128) unsigned char dyn_smem[];
  auto& smem = *reinterpret_cast<MultiQPipe128Storage<kQBatch>*>(dyn_smem);

  const int kv_head     = blockIdx.y;
  const int tid         = threadIdx.x;
  const int warp_id     = tid >> 5;
  const int lane        = tid & 31;
  const int group_id    = lane >> 2;
  const int tid_in_grp  = lane & 3;

  const int q0 = cu_seqlens_q_ptr[0];
  const int actual_m = cu_seqlens_q_ptr[1] - q0;
  const int q_base = blockIdx.z * kQBatch;
  if (q_base >= actual_m) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  const int raw_seqlen_k = seqused_k_ptr[0];
  const int lf0 = (leaf_flag_ptr != nullptr) ? leaf_flag_ptr[0] : 0;

  bool active_q[kQBatch];
  bool empty_q[kQBatch];
  int seqlen_k_arr[kQBatch];
  int k_start_arr[kQBatch];
  int num_nb_arr[kQBatch];
  int nb_min_arr[kQBatch];
  int remap_base_arr[kQBatch];
  int p_start_arr[kQBatch];
  uint32_t remap_packed_arr[kQBatch];

  bool any_active = false;
  int safe_tok_end = raw_seqlen_k;
  int batch_nb_min = 0x7fffffff;
  int batch_k_start = 0x7fffffff;

  #pragma unroll
  for (int qb = 0; qb < kQBatch; ++qb) {
    const int q_idx = q_base + qb;
    active_q[qb] = false;
    empty_q[qb] = false;
    seqlen_k_arr[qb] = 0;
    k_start_arr[qb] = 0;
    num_nb_arr[qb] = 0;
    nb_min_arr[qb] = 0;
    remap_base_arr[qb] = -1;
    p_start_arr[qb] = 0;
    remap_packed_arr[qb] = 0u;
    if (q_idx >= actual_m) continue;

    int sk = raw_seqlen_k;
    int lrb = -1;
    int lps = 0;
    uint32_t lrp = 0u;
    if (leaf_flag_ptr != nullptr && lf0 < 0 && actual_m <= 32 && sk >= actual_m) {
      const int base = 1 + q_idx * 3;
      const int seq_sub = leaf_flag_ptr[base];
      const int base_sub = leaf_flag_ptr[base + 1];
      if (base_sub >= 0) {
        lps = sk - actual_m;
        lrb = sk - base_sub;
        lrp = static_cast<uint32_t>(leaf_flag_ptr[base + 2]);
      }
      sk -= seq_sub;
    } else if (q_idx == 7 && actual_m == 8 && leaf_flag_ptr != nullptr) {
      if ((lf0 == 1 || lf0 == 2) && sk >= 8) {
        lps = sk - 8;
        lrb = lps + lf0;
        lrp = 7u;
        sk = lps + lf0 + 1;
      } else {
        sk -= (actual_m - 1 - q_idx);
      }
    } else if (q_idx == 6 && actual_m == 7 && leaf_flag_ptr != nullptr && lf0 > 0 && sk >= 7) {
      lps = sk - 7;
      lrb = lps + 1;
      lrp = 6u;
      sk -= 5;
    } else {
      sk -= (actual_m - 1 - q_idx);
    }
    if (sk < 0) sk = 0;

    const int ks = (window_left >= 0) ? ((sk - 1 - window_left > 0) ? (sk - 1 - window_left) : 0) : 0;
    const int nnb = (sk > 0) ? ((sk + 127) >> 7) : 0;
    const int gnb = (window_left >= 0 && sk > 0) ? (ks >> 7) : 0;
    if (gnb >= nnb) {
      empty_q[qb] = true;
      continue;
    }

    active_q[qb] = true;
    any_active = true;
    seqlen_k_arr[qb] = sk;
    k_start_arr[qb] = ks;
    num_nb_arr[qb] = nnb;
    nb_min_arr[qb] = gnb;
    remap_base_arr[qb] = lrb;
    p_start_arr[qb] = lps;
    remap_packed_arr[qb] = lrp;

    if (sk < safe_tok_end) safe_tok_end = sk;
    if (lrb >= 0 && lrb < safe_tok_end) safe_tok_end = lrb;
    if (gnb < batch_nb_min) batch_nb_min = gnb;
    if (ks < batch_k_start) batch_k_start = ks;
  }

  // SM121 PDL consumer barrier: wait for preceding qk_norm_rope_kvcache after independent metadata/remap setup
  // and strictly before reading q_ptr / kv_ptr or writing out_ptr.
  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  #pragma unroll
  for (int qb = 0; qb < kQBatch; ++qb) {
    const int q_idx = q_base + qb;
    if (empty_q[qb]) {
      const int q_local = tid >> 3;
      const int d_base = (tid & 7) << 4;
      __nv_bfloat16* dst = out_ptr + static_cast<int64_t>(q0 + q_idx) * out_tok_stride +
                           static_cast<int64_t>(kv_head * 16 + q_local) * out_head_stride + d_base;
      *reinterpret_cast<uint4*>(dst)     = make_uint4(0u, 0u, 0u, 0u);
      *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0u, 0u, 0u, 0u);
      continue;
    }
    if (!active_q[qb]) continue;
    const __nv_bfloat16* q_group =
        q_ptr + static_cast<int64_t>(q0 + q_idx) * q_tok_stride +
        static_cast<int64_t>(kv_head * 16) * q_head_stride;
    #pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int idx = (u << 7) + tid;
      const int m = idx >> 4;
      const int c = idx & 15;
      smem.q[qb][m][c ^ (m & 7)] =
          *reinterpret_cast<const uint4*>(q_group + static_cast<int64_t>(m) * q_head_stride + (c << 3));
    }
  }
  if (!any_active) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }
  __syncthreads();

  const int shared_end_block = (safe_tok_end > 0) ? (safe_tok_end >> 7) : 0;

  float o_frag[kQBatch][4][4];
  float row_max0[kQBatch];
  float row_max1[kQBatch];
  float row_sum0[kQBatch];
  float row_sum1[kQBatch];
  bool is_first[kQBatch];
  #pragma unroll
  for (int qb = 0; qb < kQBatch; ++qb) {
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      o_frag[qb][dt][0] = 0.f; o_frag[qb][dt][1] = 0.f;
      o_frag[qb][dt][2] = 0.f; o_frag[qb][dt][3] = 0.f;
    }
    row_max0[qb] = -INFINITY;
    row_max1[qb] = -INFINITY;
    row_sum0[qb] = 0.0f;
    row_sum1[qb] = 0.0f;
    is_first[qb] = true;
  }

  // Phase 1: Per-row tail block(s) nb >= shared_end_block (typically 1 block touching causal/tree remap).
  #pragma unroll
  for (int qb = 0; qb < kQBatch; ++qb) {
    if (!active_q[qb]) continue;
    const int tail_lo = (nb_min_arr[qb] > shared_end_block) ? nb_min_arr[qb] : shared_end_block;
    for (int nb = num_nb_arr[qb] - 1; nb >= tail_lo; --nb) {
      const int block_tok_base = nb << 7;
      float s_tail[4][4] = {0.f};

      #pragma unroll
      for (int ch = 0; ch < 4; ++ch) {
        issue_kv_tile_async<false>(smem.base.kv[ch], kv_ptr, block_table, block_tok_base + (ch << 5),
                                   k_start_arr[qb], seqlen_k_arr[qb], block_size, kv_head,
                                   stride_block, stride_slot, stride_kv_head, tid,
                                   remap_base_arr[qb], p_start_arr[qb], remap_packed_arr[qb]);
        cp_async_commit();
      }
      issue_kv_tile_async<true>(smem.base.kv[4], kv_ptr, block_table, block_tok_base,
                                k_start_arr[qb], seqlen_k_arr[qb], block_size, kv_head,
                                stride_block, stride_slot, stride_kv_head, tid,
                                remap_base_arr[qb], p_start_arr[qb], remap_packed_arr[qb]);
      cp_async_commit();

      cp_async_wait<4>();
      __syncthreads();
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t q_cur[4], b_frag[2];
        ldsm4_swizzled(q_cur, smem.q[qb], ks, lane);
        ldsm2_k_swizzled(b_frag, smem.base.kv[0], warp_id, ks, lane);
        mma_m16n8k16_bf16(q_cur, b_frag, s_tail[0]);
      }
      __syncthreads();
      issue_kv_tile_async<true>(smem.base.kv[0], kv_ptr, block_table, block_tok_base + 32,
                                k_start_arr[qb], seqlen_k_arr[qb], block_size, kv_head,
                                stride_block, stride_slot, stride_kv_head, tid,
                                remap_base_arr[qb], p_start_arr[qb], remap_packed_arr[qb]);
      cp_async_commit();

      cp_async_wait<4>();
      __syncthreads();
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t q_cur[4], b_frag[2];
        ldsm4_swizzled(q_cur, smem.q[qb], ks, lane);
        ldsm2_k_swizzled(b_frag, smem.base.kv[1], warp_id, ks, lane);
        mma_m16n8k16_bf16(q_cur, b_frag, s_tail[1]);
      }
      __syncthreads();
      issue_kv_tile_async<true>(smem.base.kv[1], kv_ptr, block_table, block_tok_base + 64,
                                k_start_arr[qb], seqlen_k_arr[qb], block_size, kv_head,
                                stride_block, stride_slot, stride_kv_head, tid,
                                remap_base_arr[qb], p_start_arr[qb], remap_packed_arr[qb]);
      cp_async_commit();

      cp_async_wait<4>();
      __syncthreads();
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t q_cur[4], b_frag[2];
        ldsm4_swizzled(q_cur, smem.q[qb], ks, lane);
        ldsm2_k_swizzled(b_frag, smem.base.kv[2], warp_id, ks, lane);
        mma_m16n8k16_bf16(q_cur, b_frag, s_tail[2]);
      }
      __syncthreads();
      issue_kv_tile_async<true>(smem.base.kv[2], kv_ptr, block_table, block_tok_base + 96,
                                k_start_arr[qb], seqlen_k_arr[qb], block_size, kv_head,
                                stride_block, stride_slot, stride_kv_head, tid,
                                remap_base_arr[qb], p_start_arr[qb], remap_packed_arr[qb]);
      cp_async_commit();

      cp_async_wait<4>();
      __syncthreads();
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t q_cur[4], b_frag[2];
        ldsm4_swizzled(q_cur, smem.q[qb], ks, lane);
        ldsm2_k_swizzled(b_frag, smem.base.kv[3], warp_id, ks, lane);
        mma_m16n8k16_bf16(q_cur, b_frag, s_tail[3]);
      }
      cp_async_wait<0>();
      __syncthreads();

      pipe128_softmax_and_pv_step(
          smem.base, s_tail, o_frag[qb], row_max0[qb], row_max1[qb], row_sum0[qb], row_sum1[qb],
          is_first[qb], block_tok_base, k_start_arr[qb], seqlen_k_arr[qb], scale_log2,
          warp_id, lane, group_id, tid_in_grp);
      is_first[qb] = false;
    }
  }

  // Phase 2: Shared-KV prefix loop (nb < shared_end_block): 1 KV load shared by all kQBatch rows!
  for (int nb = shared_end_block - 1; nb >= batch_nb_min; --nb) {
    const int block_tok_base = nb << 7;
    float s_frag[kQBatch][4][4];
    #pragma unroll
    for (int qb = 0; qb < kQBatch; ++qb) {
      #pragma unroll
      for (int ch = 0; ch < 4; ++ch) {
        s_frag[qb][ch][0] = 0.f; s_frag[qb][ch][1] = 0.f;
        s_frag[qb][ch][2] = 0.f; s_frag[qb][ch][3] = 0.f;
      }
    }

    #pragma unroll
    for (int ch = 0; ch < 4; ++ch) {
      issue_kv_tile_async<false>(smem.base.kv[ch], kv_ptr, block_table, block_tok_base + (ch << 5),
                                 batch_k_start, safe_tok_end, block_size, kv_head,
                                 stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
    }
    issue_kv_tile_async<true>(smem.base.kv[4], kv_ptr, block_table, block_tok_base,
                              batch_k_start, safe_tok_end, block_size, kv_head,
                              stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();

    cp_async_wait<4>();
    __syncthreads();
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t b_frag[2];
      ldsm2_k_swizzled(b_frag, smem.base.kv[0], warp_id, ks, lane);
      #pragma unroll
      for (int qb = 0; qb < kQBatch; ++qb) {
        if (active_q[qb] && nb >= nb_min_arr[qb]) {
          uint32_t q_cur[4];
          ldsm4_swizzled(q_cur, smem.q[qb], ks, lane);
          mma_m16n8k16_bf16(q_cur, b_frag, s_frag[qb][0]);
        }
      }
    }
    __syncthreads();
    issue_kv_tile_async<true>(smem.base.kv[0], kv_ptr, block_table, block_tok_base + 32,
                              batch_k_start, safe_tok_end, block_size, kv_head,
                              stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();

    cp_async_wait<4>();
    __syncthreads();
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t b_frag[2];
      ldsm2_k_swizzled(b_frag, smem.base.kv[1], warp_id, ks, lane);
      #pragma unroll
      for (int qb = 0; qb < kQBatch; ++qb) {
        if (active_q[qb] && nb >= nb_min_arr[qb]) {
          uint32_t q_cur[4];
          ldsm4_swizzled(q_cur, smem.q[qb], ks, lane);
          mma_m16n8k16_bf16(q_cur, b_frag, s_frag[qb][1]);
        }
      }
    }
    __syncthreads();
    issue_kv_tile_async<true>(smem.base.kv[1], kv_ptr, block_table, block_tok_base + 64,
                              batch_k_start, safe_tok_end, block_size, kv_head,
                              stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();

    cp_async_wait<4>();
    __syncthreads();
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t b_frag[2];
      ldsm2_k_swizzled(b_frag, smem.base.kv[2], warp_id, ks, lane);
      #pragma unroll
      for (int qb = 0; qb < kQBatch; ++qb) {
        if (active_q[qb] && nb >= nb_min_arr[qb]) {
          uint32_t q_cur[4];
          ldsm4_swizzled(q_cur, smem.q[qb], ks, lane);
          mma_m16n8k16_bf16(q_cur, b_frag, s_frag[qb][2]);
        }
      }
    }
    __syncthreads();
    issue_kv_tile_async<true>(smem.base.kv[2], kv_ptr, block_table, block_tok_base + 96,
                              batch_k_start, safe_tok_end, block_size, kv_head,
                              stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();

    cp_async_wait<4>();
    __syncthreads();
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t b_frag[2];
      ldsm2_k_swizzled(b_frag, smem.base.kv[3], warp_id, ks, lane);
      #pragma unroll
      for (int qb = 0; qb < kQBatch; ++qb) {
        if (active_q[qb] && nb >= nb_min_arr[qb]) {
          uint32_t q_cur[4];
          ldsm4_swizzled(q_cur, smem.q[qb], ks, lane);
          mma_m16n8k16_bf16(q_cur, b_frag, s_frag[qb][3]);
        }
      }
    }
    if constexpr (kPdl) {
      if (nb == batch_nb_min) {
        asm volatile("griddepcontrol.launch_dependents;");
      }
    }
    cp_async_wait<0>();
    __syncthreads();

    #pragma unroll
    for (int qb = 0; qb < kQBatch; ++qb) {
      if (active_q[qb] && nb >= nb_min_arr[qb]) {
        pipe128_softmax_and_pv_step(
            smem.base, s_frag[qb], o_frag[qb], row_max0[qb], row_max1[qb], row_sum0[qb], row_sum1[qb],
            is_first[qb], block_tok_base, k_start_arr[qb], seqlen_k_arr[qb], scale_log2,
            warp_id, lane, group_id, tid_in_grp);
        is_first[qb] = false;
      }
    }
  }

  if constexpr (kPdl) {
    if (shared_end_block <= batch_nb_min) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
  }

  // Final normalization and BF16 output store for each active query row.
  #pragma unroll
  for (int qb = 0; qb < kQBatch; ++qb) {
    if (!active_q[qb]) continue;
    if (warp_id == 0) {
      float rs0 = add_ftz_f32(row_sum0[qb], __shfl_xor_sync(0xffffffffu, row_sum0[qb], 2));
      rs0 = add_ftz_f32(rs0, __shfl_xor_sync(0xffffffffu, rs0, 1));
      float rs1 = add_ftz_f32(row_sum1[qb], __shfl_xor_sync(0xffffffffu, row_sum1[qb], 2));
      rs1 = add_ftz_f32(rs1, __shfl_xor_sync(0xffffffffu, rs1, 1));
      const bool empty0 = (rs0 == 0.0f || rs0 != rs0 || row_max0[qb] == -INFINITY);
      const bool empty1 = (rs1 == 0.0f || rs1 != rs1 || row_max1[qb] == -INFINITY);
      const float inv0 = empty0 ? 1.0f : fast_rcpf(rs0);
      const float inv1 = empty1 ? 1.0f : fast_rcpf(rs1);
      if (tid_in_grp == 0) {
        smem.base.warp_sum[group_id]     = inv0;
        smem.base.warp_sum[group_id + 8] = inv1;
      }
    }
    __syncthreads();

    const float inv0 = smem.base.warp_sum[group_id];
    const float inv1 = smem.base.warp_sum[group_id + 8];
    __nv_bfloat16* const out_row = out_ptr + static_cast<int64_t>(q0 + q_base + qb) * out_tok_stride;
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      const int d_col = (warp_id << 5) + (dt << 3) + (tid_in_grp << 1);
      const __nv_bfloat162 ob0 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[qb][dt][0], inv0), mul_ftz_f32(o_frag[qb][dt][1], inv0));
      const __nv_bfloat162 ob1 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[qb][dt][2], inv1), mul_ftz_f32(o_frag[qb][dt][3], inv1));
      *reinterpret_cast<uint32_t*>(
          out_row + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob0);
      *reinterpret_cast<uint32_t*>(
          out_row + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob1);
    }
    __syncthreads();
  }
}

// ============================================================================
// Warp-Per-Query Shared-KV FA2 Verify Kernel (kWarps = 8 or 4 queries / CTA)
//
// Strictly 0-ULP bit-identical to FlashAttention-2 kBlockN=128 & pipe_kernel<true>:
//   - Each warp owns 1 complete query token (16 Q heads x 128 head_dim) in registers:
//     q_frag[8][4] + o_frag[16][4] + in-place s_frag[16][4]/p_frag[8][4] (0 SMEM q/p!).
//   - All kWarps warps share 1 cp.async KV load per 128-token block into kv[2][64][16]
//     (4x fewer KV L2 reads vs multiq<2>, zero softmax __syncthreads(), zero serial warp0
//     row_sum, and 8 CTAs @ M=32 with kWarps=8 leaves 40/48 SMs free for o_gate!).
//   - Tree-remapped tail blocks load the <=32 tree K/V rows ONCE into tree_kv[32][16]
//     and evaluate all kWarps queries in parallel via per-lane ldmatrix indirection.
// ============================================================================
struct __align__(128) VerifyV3Storage {
  uint4 kv[2][64][16];       // 32,768 B (aliases q_tiles[8][16][16] during Q preload)
  uint4 tree_kv[32][16];     //  8,192 B (up to 32 tree K/V rows for remapped tail block)
  uint4 zero_row[16];        //    256 B (constant +0.0 BF16 row for out-of-range lanes)
  int32_t nb_max_w[8];       //     32 B
  int32_t nb_min_w[8];       //     32 B
  int32_t k_start_w[8];      //     32 B
  int32_t seqlen_w[8];       //     32 B
  int32_t remap_base_w[8];   //     32 B
  uint8_t active_w[8];       //      8 B
};                           // 41,408 B (< 48 KiB static SMEM; 2 CTAs/SM for kWarps=4)
static_assert(sizeof(VerifyV3Storage) <= 45440,
              "VerifyV3Storage must stay <= 45,440 B for static SMEM / 2-CTA carveout");

__device__ __forceinline__ void mma_m16n8k16_bf16_u2(
    const uint32_t (&a)[4], uint32_t b0, uint32_t b1, float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b0), "r"(b1),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void ldsm4_k_pair64_v3(
    uint32_t (&b)[4], const uint4 tile[64][16], int sub_pair, int ks, int lane) {
  const int r_local = lane & 7;
  const int sub = (sub_pair << 1) + (lane >> 4);
  const int sub_k = (lane >> 3) & 1;
  const int r = (sub << 3) + r_local;
  const int c_log = (ks << 1) + sub_k;
  const uint32_t addr = smem_u32(&tile[r][c_log ^ r_local]);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
}

__device__ __forceinline__ void ldsm4_v_pair64_v3(
    uint32_t (&v)[4], const uint4 tile[64][16], int dt_pair, int ks_local, int lane) {
  const int r_in_16 = lane & 15;
  const int r = (ks_local << 4) + r_in_16;
  const int dt = (dt_pair << 1) + (lane >> 4);
  const uint32_t addr = smem_u32(&tile[r][dt ^ (r & 7)]);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(addr));
}

__device__ __forceinline__ void ldsm4_k_pair64_remap(
    uint32_t (&b)[4],
    const uint4 tile[64][16],
    const uint4 tree_tile[32][16],
    const uint4 zero_row[16],
    int chunk_tok_base,
    int my_k_start,
    int my_seqlen_k,
    int my_remap_base,
    uint32_t my_remap_packed,
    int sub_pair,
    int ks,
    int lane) {
  const int r_local = lane & 7;
  const int sub = (sub_pair << 1) + (lane >> 4);
  const int sub_k = (lane >> 3) & 1;
  const int r = (sub << 3) + r_local;
  const int c_log = (ks << 1) + sub_k;
  const int tok = chunk_tok_base + r;
  uint32_t addr;
  if (tok >= my_k_start && tok < my_seqlen_k) {
    if (my_remap_base >= 0 && tok >= my_remap_base) {
      const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
      addr = smem_u32(&tree_tile[rr][c_log ^ (rr & 7)]);
    } else {
      addr = smem_u32(&tile[r][c_log ^ r_local]);
    }
  } else {
    addr = smem_u32(&zero_row[c_log]);
  }
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
}

__device__ __forceinline__ void ldsm4_v_pair64_remap(
    uint32_t (&v)[4],
    const uint4 tile[64][16],
    const uint4 tree_tile[32][16],
    const uint4 zero_row[16],
    int chunk_tok_base,
    int my_k_start,
    int my_seqlen_k,
    int my_remap_base,
    uint32_t my_remap_packed,
    int dt_pair,
    int ks_local,
    int lane) {
  const int r_in_16 = lane & 15;
  const int r = (ks_local << 4) + r_in_16;
  const int dt = (dt_pair << 1) + (lane >> 4);
  const int tok = chunk_tok_base + r;
  uint32_t addr;
  if (tok >= my_k_start && tok < my_seqlen_k) {
    if (my_remap_base >= 0 && tok >= my_remap_base) {
      const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
      addr = smem_u32(&tree_tile[rr][dt ^ (rr & 7)]);
    } else {
      addr = smem_u32(&tile[r][dt ^ (r & 7)]);
    }
  } else {
    addr = smem_u32(&zero_row[dt]);
  }
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(addr));
}

template <bool IsV, int kWarps>
__device__ __forceinline__ void issue_kv64_verify(
    uint4 smem_tile[64][16],
    const __nv_bfloat16* __restrict__ kv_ptr,
    const int32_t* __restrict__ block_table,
    int tok_base,
    int cta_k_start,
    int cta_seqlen_k,
    int block_size,
    int kv_head,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    int tid) {
  constexpr int kRowsPerStep = (kWarps * 32) >> 4;  // 16 (kWarps=8) or 8 (kWarps=4)
  constexpr int kSteps = 64 / kRowsPerStep;         // 4 or 8
  const int64_t head_v_off = static_cast<int64_t>(kv_head) * stride_kv_head + (IsV ? 128 : 0);
  const int r_sub = tid >> 4;
  const int c = tid & 15;
  const int c_phys = c ^ (r_sub & 7);
  const bool full_valid = (tok_base >= cta_k_start) && (tok_base + 64 <= cta_seqlen_k);
  if (full_valid && block_size == 16) {
    const int lb0 = tok_base >> 4;
    #pragma unroll
    for (int u = 0; u < kSteps; ++u) {
      const int r = u * kRowsPerStep + r_sub;
      const int phys_blk = __ldg(block_table + lb0 + (r >> 4));
      const int64_t row_off = static_cast<int64_t>(phys_blk) * stride_block +
                              static_cast<int64_t>(r & 15) * stride_slot + head_v_off;
      cp_async_16(smem_u32(&smem_tile[r][c_phys]), kv_ptr + row_off + (c << 3));
    }
  } else {
    #pragma unroll
    for (int u = 0; u < kSteps; ++u) {
      const int r = u * kRowsPerStep + r_sub;
      const int tok = tok_base + r;
      if (tok >= cta_k_start && tok < cta_seqlen_k) {
        const int lb = (block_size == 16) ? (tok >> 4) : (tok / block_size);
        const int slot = (block_size == 16) ? (tok & 15) : (tok - lb * block_size);
        const int phys_blk = __ldg(block_table + lb);
        const int64_t row_off = static_cast<int64_t>(phys_blk) * stride_block +
                                static_cast<int64_t>(slot) * stride_slot + head_v_off;
        cp_async_16(smem_u32(&smem_tile[r][c_phys]), kv_ptr + row_off + (c << 3));
      } else {
        smem_tile[r][c_phys] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  }
}

template <bool IsV, int kWarps>
__device__ __forceinline__ void issue_tree_kv_verify(
    uint4 tree_tile[32][16],
    const __nv_bfloat16* __restrict__ kv_ptr,
    const int32_t* __restrict__ block_table,
    int cta_p_start,
    int actual_m,
    int block_size,
    int kv_head,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    int tid) {
  constexpr int kRowsPerStep = (kWarps * 32) >> 4;
  constexpr int kTreeSteps = 32 / kRowsPerStep;
  const int64_t head_v_off = static_cast<int64_t>(kv_head) * stride_kv_head + (IsV ? 128 : 0);
  const int r_sub = tid >> 4;
  const int c = tid & 15;
  const int c_phys = c ^ (r_sub & 7);
  #pragma unroll
  for (int u = 0; u < kTreeSteps; ++u) {
    const int r = u * kRowsPerStep + r_sub;
    if (r < actual_m) {
      const int tok = cta_p_start + r;
      const int lb = (block_size == 16) ? (tok >> 4) : (tok / block_size);
      const int slot = (block_size == 16) ? (tok & 15) : (tok - lb * block_size);
      const int phys_blk = __ldg(block_table + lb);
      const int64_t row_off = static_cast<int64_t>(phys_blk) * stride_block +
                              static_cast<int64_t>(slot) * stride_slot + head_v_off;
      cp_async_16(smem_u32(&tree_tile[r][c_phys]), kv_ptr + row_off + (c << 3));
    } else {
      tree_tile[r][c_phys] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
}

template <int kWarps, bool kPdl = false>
__global__ __launch_bounds__(kWarps * 32, (kWarps == 4 ? 2 : 1)) void gqa16_verify_v3_kernel(
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
    int64_t stride_kv_head,
    const int32_t* __restrict__ cu_seqlens_q_ptr,
    int64_t q_tok_stride,
    int64_t out_tok_stride,
    const int32_t* __restrict__ leaf_flag_ptr) {
  static_assert(kWarps == 4 || kWarps == 8, "kWarps must be 4 or 8");
  __shared__ VerifyV3Storage smem;

  const int kv_head    = blockIdx.y;
  const int tid        = threadIdx.x;
  const int warp_id    = tid >> 5;
  const int lane       = tid & 31;
  const int group_id   = lane >> 2;
  const int tid_in_grp = lane & 3;

  const int q0 = cu_seqlens_q_ptr[0];
  const int actual_m = cu_seqlens_q_ptr[1] - q0;
  const int q_base = blockIdx.z * kWarps;
  if (q_base >= actual_m) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  const int raw_seqlen_k = seqused_k_ptr[0];
  const int lf0 = (leaf_flag_ptr != nullptr) ? leaf_flag_ptr[0] : 0;
  const int my_q_idx = q_base + warp_id;

  bool my_active = false;
  bool my_empty = false;
  int my_seqlen_k = 0;
  int my_k_start = 0;
  int my_num_nb = 0;
  int my_nb_min = 0;
  int my_remap_base = -1;
  uint32_t my_remap_packed = 0u;

  if (my_q_idx < actual_m) {
    int sk = raw_seqlen_k;
    int lrb = -1;
    uint32_t lrp = 0u;
    if (leaf_flag_ptr != nullptr && lf0 < 0 && actual_m <= 32 && sk >= actual_m) {
      const int base = 1 + my_q_idx * 3;
      const int seq_sub = leaf_flag_ptr[base];
      const int base_sub = leaf_flag_ptr[base + 1];
      if (base_sub >= 0) {
        lrb = sk - base_sub;
        lrp = static_cast<uint32_t>(leaf_flag_ptr[base + 2]);
      }
      sk -= seq_sub;
    } else if (my_q_idx == 7 && actual_m == 8 && leaf_flag_ptr != nullptr) {
      if ((lf0 == 1 || lf0 == 2) && sk >= 8) {
        const int lps = sk - 8;
        lrb = lps + lf0;
        lrp = 7u;
        sk = lps + lf0 + 1;
      } else {
        sk -= (actual_m - 1 - my_q_idx);
      }
    } else if (my_q_idx == 6 && actual_m == 7 && leaf_flag_ptr != nullptr && lf0 > 0 && sk >= 7) {
      const int lps = sk - 7;
      lrb = lps + 1;
      lrp = 6u;
      sk -= 5;
    } else {
      sk -= (actual_m - 1 - my_q_idx);
    }
    if (sk < 0) sk = 0;

    const int ks = (window_left >= 0) ? ((sk - 1 - window_left > 0) ? (sk - 1 - window_left) : 0) : 0;
    const int nnb = (sk > 0) ? ((sk + 127) >> 7) : 0;
    const int gnb = (window_left >= 0 && sk > 0) ? (ks >> 7) : 0;
    if (gnb >= nnb) {
      my_empty = true;
    } else {
      my_active = true;
      my_seqlen_k = sk;
      my_k_start = ks;
      my_num_nb = nnb;
      my_nb_min = gnb;
      my_remap_base = lrb;
      my_remap_packed = lrp;
    }
  }

  if (lane == 0) {
    smem.active_w[warp_id] = my_active ? 1u : 0u;
    smem.nb_max_w[warp_id] = my_active ? my_num_nb : 0;
    smem.nb_min_w[warp_id] = my_active ? my_nb_min : 0x7fffffff;
    smem.k_start_w[warp_id] = my_active ? my_k_start : 0x7fffffff;
    smem.seqlen_w[warp_id] = my_active ? my_seqlen_k : 0;
    smem.remap_base_w[warp_id] = (my_active && my_remap_base >= 0) ? my_remap_base : 0x7fffffff;
  }
  if (tid < 16) {
    smem.zero_row[tid] = make_uint4(0u, 0u, 0u, 0u);
  }
  __syncthreads();

  bool cta_any_active = false;
  int cta_nb_max = 0;
  int cta_nb_min = 0x7fffffff;
  int cta_k_start = 0x7fffffff;
  int cta_seqlen_k = 0;
  int cta_remap_min_base = 0x7fffffff;
  #pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    if (smem.active_w[w]) {
      cta_any_active = true;
      if (smem.nb_max_w[w] > cta_nb_max) cta_nb_max = smem.nb_max_w[w];
      if (smem.nb_min_w[w] < cta_nb_min) cta_nb_min = smem.nb_min_w[w];
      if (smem.k_start_w[w] < cta_k_start) cta_k_start = smem.k_start_w[w];
      if (smem.seqlen_w[w] > cta_seqlen_k) cta_seqlen_k = smem.seqlen_w[w];
      if (smem.remap_base_w[w] < cta_remap_min_base) cta_remap_min_base = smem.remap_base_w[w];
    }
  }
  const bool cta_has_remap = (cta_remap_min_base < 0x7fffffff);
  const int cta_p_start = raw_seqlen_k - actual_m;

  // SM121 PDL consumer barrier: wait after independent metadata/remap setup and before Q/KV/out access.
  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  if (my_empty) {
    __nv_bfloat16* const dst_base = out_ptr + static_cast<int64_t>(q0 + my_q_idx) * out_tok_stride +
                                    static_cast<int64_t>(kv_head * 16) * out_head_stride;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = (u << 5) + lane;
      const int m = idx >> 4;
      const int c = idx & 15;
      *reinterpret_cast<uint4*>(dst_base + static_cast<int64_t>(m) * out_head_stride + (c << 3)) =
          make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (!cta_any_active) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  // Load each active warp's Q group (16 heads x 128 dims) through kv_smem into registers, then free kv_smem.
  uint32_t q_frag[8][4];
  if (my_active) {
    uint4 (*q_tiles)[16][16] = reinterpret_cast<uint4 (*)[16][16]>(smem.kv);
    const __nv_bfloat16* q_tok_base = q_ptr + static_cast<int64_t>(q0 + my_q_idx) * q_tok_stride +
                                      static_cast<int64_t>(kv_head * 16) * q_head_stride;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = (u << 5) + lane;
      const int m = idx >> 4;
      const int c = idx & 15;
      q_tiles[warp_id][m][c ^ (m & 7)] =
          *reinterpret_cast<const uint4*>(q_tok_base + static_cast<int64_t>(m) * q_head_stride + (c << 3));
    }
    __syncwarp();
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      ldsm4_swizzled(q_frag[ks], q_tiles[warp_id], ks, lane);
    }
  }
  __syncthreads();

  float o_frag[16][4];
  #pragma unroll
  for (int dt = 0; dt < 16; ++dt) {
    o_frag[dt][0] = 0.0f; o_frag[dt][1] = 0.0f; o_frag[dt][2] = 0.0f; o_frag[dt][3] = 0.0f;
  }
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;
  bool my_is_first = true;

  // Prime first block's K half 0 into smem.kv[0].
  issue_kv64_verify<false, kWarps>(
      smem.kv[0], kv_ptr, block_table, (cta_nb_max - 1) << 7,
      cta_k_start, cta_seqlen_k, block_size, kv_head,
      stride_block, stride_slot, stride_kv_head, tid);
  cp_async_commit();

  for (int nb = cta_nb_max - 1; nb >= cta_nb_min; --nb) {
    const int block_tok_base = nb << 7;
    const bool nb_has_remap = cta_has_remap && (block_tok_base + 128 > cta_remap_min_base);
    const bool my_in_block = my_active && (nb < my_num_nb) && (nb >= my_nb_min);

    float s_frag[16][4];
    #pragma unroll
    for (int ni = 0; ni < 16; ++ni) {
      s_frag[ni][0] = 0.0f; s_frag[ni][1] = 0.0f; s_frag[ni][2] = 0.0f; s_frag[ni][3] = 0.0f;
    }

    if (!nb_has_remap) {
      issue_kv64_verify<false, kWarps>(
          smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
      cp_async_wait<1>();
      __syncthreads();
      if (my_in_block) {
        uint32_t b_cur[4], b_nxt[4];
        ldsm4_k_pair64_v3(b_cur, smem.kv[0], 0, 0, lane);
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          #pragma unroll
          for (int sp = 0; sp < 4; ++sp) {
            const int n_sp = (sp + 1) & 3;
            const int n_ks = ks + (sp == 3 ? 1 : 0);
            if (n_ks < 8) ldsm4_k_pair64_v3(b_nxt, smem.kv[0], n_sp, n_ks, lane);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[0], b_cur[1], s_frag[sp * 2 + 0]);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[2], b_cur[3], s_frag[sp * 2 + 1]);
            b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1]; b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
          }
        }
      }
      __syncthreads();

      issue_kv64_verify<true, kWarps>(
          smem.kv[0], kv_ptr, block_table, block_tok_base,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
      cp_async_wait<1>();
      __syncthreads();
      if (my_in_block) {
        uint32_t b_cur[4], b_nxt[4];
        ldsm4_k_pair64_v3(b_cur, smem.kv[1], 0, 0, lane);
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          #pragma unroll
          for (int sp = 0; sp < 4; ++sp) {
            const int n_sp = (sp + 1) & 3;
            const int n_ks = ks + (sp == 3 ? 1 : 0);
            if (n_ks < 8) ldsm4_k_pair64_v3(b_nxt, smem.kv[1], n_sp, n_ks, lane);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[0], b_cur[1], s_frag[8 + sp * 2 + 0]);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[2], b_cur[3], s_frag[8 + sp * 2 + 1]);
            b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1]; b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
          }
        }
      }
      __syncthreads();

      issue_kv64_verify<true, kWarps>(
          smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
    } else {
      issue_kv64_verify<false, kWarps>(
          smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      issue_tree_kv_verify<false, kWarps>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();

      if (my_in_block) {
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          #pragma unroll
          for (int sp = 0; sp < 4; ++sp) {
            uint32_t b_cur[4];
            ldsm4_k_pair64_remap(
                b_cur, smem.kv[0], smem.tree_kv, smem.zero_row,
                block_tok_base, my_k_start, my_seqlen_k, my_remap_base, my_remap_packed,
                sp, ks, lane);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[0], b_cur[1], s_frag[sp * 2 + 0]);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[2], b_cur[3], s_frag[sp * 2 + 1]);
          }
        }
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          #pragma unroll
          for (int sp = 0; sp < 4; ++sp) {
            uint32_t b_cur[4];
            ldsm4_k_pair64_remap(
                b_cur, smem.kv[1], smem.tree_kv, smem.zero_row,
                block_tok_base + 64, my_k_start, my_seqlen_k, my_remap_base, my_remap_packed,
                sp, ks, lane);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[0], b_cur[1], s_frag[8 + sp * 2 + 0]);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[2], b_cur[3], s_frag[8 + sp * 2 + 1]);
          }
        }
      }
      __syncthreads();

      issue_kv64_verify<true, kWarps>(
          smem.kv[0], kv_ptr, block_table, block_tok_base,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      issue_kv64_verify<true, kWarps>(
          smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      issue_tree_kv_verify<true, kWarps>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
    }

    if constexpr (kPdl) {
      if (nb == cta_nb_min) {
        asm volatile("griddepcontrol.launch_dependents;");
      }
    }

    // Pure warp-local online softmax in registers (0 SMEM, 0 __syncthreads(), overlaps V cp.async!).
    // Packs BF16x2 P fragments in-place into s_frag[0..7][0..3] to save 32 registers/thread.
    if (my_in_block) {
      const bool need_mask = (block_tok_base < my_k_start) || (block_tok_base + 128 > my_seqlen_k);
      float m0 = -INFINITY, m1 = -INFINITY;
      #pragma unroll
      for (int ni = 0; ni < 16; ++ni) {
        if (need_mask) {
          const int col0 = block_tok_base + (ni << 3) + (tid_in_grp << 1);
          const int col1 = col0 + 1;
          if (col0 < my_k_start || col0 >= my_seqlen_k) { s_frag[ni][0] = -INFINITY; s_frag[ni][2] = -INFINITY; }
          if (col1 < my_k_start || col1 >= my_seqlen_k) { s_frag[ni][1] = -INFINITY; s_frag[ni][3] = -INFINITY; }
        }
        m0 = max_ftz_f32(m0, max_ftz_f32(s_frag[ni][0], s_frag[ni][1]));
        m1 = max_ftz_f32(m1, max_ftz_f32(s_frag[ni][2], s_frag[ni][3]));
      }
      m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
      m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
      m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
      m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 1));

      float sc0 = 1.0f, sc1 = 1.0f;
      if (my_is_first) {
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
      #pragma unroll
      for (int ni = 0; ni < 16; ++ni) {
        const float s0 = s_frag[ni][0];
        const float s1 = s_frag[ni][1];
        const float s2 = s_frag[ni][2];
        const float s3 = s_frag[ni][3];
        const float p0 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s0, scale_log2), max_sc0));
        const float p1 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s1, scale_log2), max_sc0));
        const float p2 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s2, scale_log2), max_sc1));
        const float p3 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s3, scale_log2), max_sc1));
        if (ni == 0) {
          if (my_is_first) {
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
        s_frag[kp][hi + 0] = __uint_as_float(*reinterpret_cast<const uint32_t*>(&pb0));
        s_frag[kp][hi + 1] = __uint_as_float(*reinterpret_cast<const uint32_t*>(&pb1));
      }
      my_is_first = false;
    }

    if (!nb_has_remap) {
      cp_async_wait<1>();
      __syncthreads();
      if (my_in_block) {
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_v_pair64_v3(v_cur, smem.kv[0], 0, 0, lane);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          const uint32_t p_cur[4] = {
              __float_as_uint(s_frag[ks_local][0]),
              __float_as_uint(s_frag[ks_local][1]),
              __float_as_uint(s_frag[ks_local][2]),
              __float_as_uint(s_frag[ks_local][3]),
          };
          #pragma unroll
          for (int dp = 0; dp < 8; ++dp) {
            const int n_dp = (dp + 1) & 7;
            const int n_ks = ks_local + (dp == 7 ? 1 : 0);
            if (n_ks < 4) ldsm4_v_pair64_v3(v_nxt, smem.kv[0], n_dp, n_ks, lane);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
      }
      __syncthreads();

      if (nb > cta_nb_min) {
        issue_kv64_verify<false, kWarps>(
            smem.kv[0], kv_ptr, block_table, (nb - 1) << 7,
            cta_k_start, cta_seqlen_k, block_size, kv_head,
            stride_block, stride_slot, stride_kv_head, tid);
        cp_async_commit();
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
      }
      __syncthreads();
      if (my_in_block) {
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_v_pair64_v3(v_cur, smem.kv[1], 0, 0, lane);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          const uint32_t p_cur[4] = {
              __float_as_uint(s_frag[4 + ks_local][0]),
              __float_as_uint(s_frag[4 + ks_local][1]),
              __float_as_uint(s_frag[4 + ks_local][2]),
              __float_as_uint(s_frag[4 + ks_local][3]),
          };
          #pragma unroll
          for (int dp = 0; dp < 8; ++dp) {
            const int n_dp = (dp + 1) & 7;
            const int n_ks = ks_local + (dp == 7 ? 1 : 0);
            if (n_ks < 4) ldsm4_v_pair64_v3(v_nxt, smem.kv[1], n_dp, n_ks, lane);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
      }
      __syncthreads();
    } else {
      cp_async_wait<0>();
      __syncthreads();
      if (my_in_block) {
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          const uint32_t p_cur[4] = {
              __float_as_uint(s_frag[ks_local][0]),
              __float_as_uint(s_frag[ks_local][1]),
              __float_as_uint(s_frag[ks_local][2]),
              __float_as_uint(s_frag[ks_local][3]),
          };
          #pragma unroll
          for (int dp = 0; dp < 8; ++dp) {
            uint32_t v_cur[4];
            ldsm4_v_pair64_remap(
                v_cur, smem.kv[0], smem.tree_kv, smem.zero_row,
                block_tok_base, my_k_start, my_seqlen_k, my_remap_base, my_remap_packed,
                dp, ks_local, lane);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
          }
        }
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          const uint32_t p_cur[4] = {
              __float_as_uint(s_frag[4 + ks_local][0]),
              __float_as_uint(s_frag[4 + ks_local][1]),
              __float_as_uint(s_frag[4 + ks_local][2]),
              __float_as_uint(s_frag[4 + ks_local][3]),
          };
          #pragma unroll
          for (int dp = 0; dp < 8; ++dp) {
            uint32_t v_cur[4];
            ldsm4_v_pair64_remap(
                v_cur, smem.kv[1], smem.tree_kv, smem.zero_row,
                block_tok_base + 64, my_k_start, my_seqlen_k, my_remap_base, my_remap_packed,
                dp, ks_local, lane);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
          }
        }
      }
      __syncthreads();
      if (nb > cta_nb_min) {
        issue_kv64_verify<false, kWarps>(
            smem.kv[0], kv_ptr, block_table, (nb - 1) << 7,
            cta_k_start, cta_seqlen_k, block_size, kv_head,
            stride_block, stride_slot, stride_kv_head, tid);
        cp_async_commit();
      }
    }
  }

  // Final warp-local normalization and direct BF16x2 output store (0 barriers!).
  if (my_active) {
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 2));
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 1));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 2));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 1));
    const bool empty0 = (row_sum0 == 0.0f || row_sum0 != row_sum0 || row_max0 == -INFINITY);
    const bool empty1 = (row_sum1 == 0.0f || row_sum1 != row_sum1 || row_max1 == -INFINITY);
    const float inv0 = empty0 ? 1.0f : fast_rcpf(row_sum0);
    const float inv1 = empty1 ? 1.0f : fast_rcpf(row_sum1);
    __nv_bfloat16* const out_row = out_ptr + static_cast<int64_t>(q0 + my_q_idx) * out_tok_stride;
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
      const int d_col = (dt << 3) + (tid_in_grp << 1);
      const __nv_bfloat162 ob0 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[dt][0], inv0), mul_ftz_f32(o_frag[dt][1], inv0));
      const __nv_bfloat162 ob1 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[dt][2], inv1), mul_ftz_f32(o_frag[dt][3], inv1));
      *reinterpret_cast<uint32_t*>(
          out_row + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob0);
      *reinterpret_cast<uint32_t*>(
          out_row + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob1);
    }
  }
}

// ============================================================================
// Wave-64 V4 Dual-Warp-Per-Query Shared-KV FA2 Verify Kernel (4Q x 2 warps/Q)
//
// Strictly 0-ULP bit-identical to FlashAttention-2 kBlockN=128 & verify_v3:
//   - 256 threads (8 warps = 4 Q tokens x 2 cooperating warps/Q):
//       M=32 -> 16 CTAs (128 warps across 16 SMs, 0 Wave-2 spill vs 32-CTA FP8 gate)
//       M=16 ->  8 CTAs ( 64 warps across  8 SMs, 4x less KV traffic vs pipe_kernel)
//   - N-half split for QK^T (w_sub=0 owns kv[0] 0..63, w_sub=1 owns kv[1] 64..127 ->
//     32 MMAs + 32 MUFU.EX2 per warp = 2x compute & softmax throughput!)
//   - D-half split for PV (w_sub=0 owns D=0..63, w_sub=1 owns D=64..127 -> o_frag[8][4]
//     + s_frag[8][4] cuts core regs 160 -> 96, eliminating v3's 232-reg pressure!)
//   - Exact FA2 descending block-128 order & exact ni=0..15 add_ftz_f32 row_sum order.
//   - Hoisted per-lane SMEM row pointers + double-buffered ldmatrix in nb_has_remap.
//   - Fast block_size == 64 (1 block_table load/chunk) & block_size == 16 paths.
// ============================================================================
struct __align__(128) VerifyV4Storage {
  uint4 kv[2][64][16];         // 32,768 B (aliases q_tiles[4][16][16] during Q preload)
  uint4 tree_kv[32][16];       //  8,192 B (up to 32 tree K/V rows for remapped tail block)
  uint4 zero_row[16];          //    256 B (constant +0.0 BF16 row for out-of-range lanes)
  float4 p_hi_f32[4][8][32];   // 16,384 B (w_sub=1 -> w_sub=0 exact ni=8..15 row_sum + p_hi)
  uint4 p_lo_bf16[4][4][32];   //  8,192 B (w_sub=0 -> w_sub=1 packed BF16x2 P[0..3])
  float max_exch[4][2][16];    //    512 B (per-warp QK row max exchange)
  float inv_exch[4][16];       //    256 B (final inv_row_sum broadcast to w_sub=1)
  int32_t nb_max_q[4];         //     16 B
  int32_t nb_min_q[4];         //     16 B
  int32_t k_start_q[4];        //     16 B
  int32_t seqlen_q[4];         //     16 B
  int32_t remap_base_q[4];     //     16 B
  uint8_t active_q[4];         //      4 B
};                             // 66,688 B (<= 99 KiB dynamic SMEM on sm_121a)
static_assert(sizeof(VerifyV4Storage) <= 99328, "VerifyV4Storage must fit SM121 dynamic SMEM");

template <bool IsV, int kKvL2 = 0>
__device__ __forceinline__ void issue_kv64_v4(
    uint4 smem_tile[64][16],
    const __nv_bfloat16* __restrict__ kv_ptr,
    const int32_t* __restrict__ block_table,
    int tok_base,
    int cta_k_start,
    int cta_seqlen_k,
    int block_size,
    int kv_head,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    int tid,
    uint64_t l2_pol = 0ull) {
  const int64_t head_v_off = static_cast<int64_t>(kv_head) * stride_kv_head + (IsV ? 128 : 0);
  const int r_sub = tid >> 4;  // 0..15 (256 threads -> 16 rows per step, 4 steps for 64 rows!)
  const int c = tid & 15;
  const int c_phys = c ^ (r_sub & 7);
  const bool full_valid = (tok_base >= cta_k_start) && (tok_base + 64 <= cta_seqlen_k);
  if (full_valid && block_size == 64 && ((tok_base & 63) == 0)) {
    const int phys_blk = __ldg(block_table + (tok_base >> 6));
    const int64_t blk_base = static_cast<int64_t>(phys_blk) * stride_block + head_v_off + (c << 3);
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int r = (u << 4) + r_sub;
      cp_async_16_l2<kKvL2>(smem_u32(&smem_tile[r][c_phys]), kv_ptr + blk_base + static_cast<int64_t>(r) * stride_slot, l2_pol);
    }
  } else if (full_valid && block_size == 16 && ((tok_base & 15) == 0)) {
    const int lb0 = tok_base >> 4;
    const int64_t slot_off = static_cast<int64_t>(r_sub) * stride_slot + head_v_off + (c << 3);
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int phys_blk = __ldg(block_table + lb0 + u);
      const int r = (u << 4) + r_sub;
      cp_async_16_l2<kKvL2>(smem_u32(&smem_tile[r][c_phys]), kv_ptr + static_cast<int64_t>(phys_blk) * stride_block + slot_off, l2_pol);
    }
  } else if (block_size == 16 && ((tok_base & 15) == 0)) {
    // Hoisted warp-uniform page-table lookup for partial first/last blocks (0 div/mod, 1 __ldg per valid page).
    const int lb0 = tok_base >> 4;
    const int64_t slot_off = static_cast<int64_t>(r_sub) * stride_slot + head_v_off + (c << 3);
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int r = (u << 4) + r_sub;
      const int page_base = tok_base + (u << 4);
      if (page_base < cta_seqlen_k && page_base + 16 > cta_k_start) {
        const int phys_blk = __ldg(block_table + lb0 + u);
        const int tok = page_base + r_sub;
        if (tok >= cta_k_start && tok < cta_seqlen_k) {
          cp_async_16_l2<kKvL2>(smem_u32(&smem_tile[r][c_phys]),
                                kv_ptr + static_cast<int64_t>(phys_blk) * stride_block + slot_off, l2_pol);
        } else {
          smem_tile[r][c_phys] = make_uint4(0u, 0u, 0u, 0u);
        }
      } else {
        smem_tile[r][c_phys] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  } else if (block_size == 64 && ((tok_base & 63) == 0)) {
    if (tok_base < cta_seqlen_k && tok_base + 64 > cta_k_start) {
      const int phys_blk = __ldg(block_table + (tok_base >> 6));
      const int64_t blk_base = static_cast<int64_t>(phys_blk) * stride_block + head_v_off + (c << 3);
      #pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int r = (u << 4) + r_sub;
        const int tok = tok_base + r;
        if (tok >= cta_k_start && tok < cta_seqlen_k) {
          cp_async_16_l2<kKvL2>(smem_u32(&smem_tile[r][c_phys]),
                                kv_ptr + blk_base + static_cast<int64_t>(r) * stride_slot, l2_pol);
        } else {
          smem_tile[r][c_phys] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    } else {
      #pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int r = (u << 4) + r_sub;
        smem_tile[r][c_phys] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  } else {
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int r = (u << 4) + r_sub;
      const int tok = tok_base + r;
      if (tok >= cta_k_start && tok < cta_seqlen_k) {
        const int lb = (block_size == 16) ? (tok >> 4)
                     : ((block_size == 64) ? (tok >> 6) : (tok / block_size));
        const int slot = (block_size == 16) ? (tok & 15)
                       : ((block_size == 64) ? (tok & 63) : (tok - lb * block_size));
        const int phys_blk = __ldg(block_table + lb);
        const int64_t row_off = static_cast<int64_t>(phys_blk) * stride_block +
                                static_cast<int64_t>(slot) * stride_slot + head_v_off;
        cp_async_16_l2<kKvL2>(smem_u32(&smem_tile[r][c_phys]), kv_ptr + row_off + (c << 3), l2_pol);
      } else {
        smem_tile[r][c_phys] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  }
}

template <bool IsV, int kKvL2 = 0>
__device__ __forceinline__ void issue_tree_kv_v4(
    uint4 tree_tile[32][16],
    const __nv_bfloat16* __restrict__ kv_ptr,
    const int32_t* __restrict__ block_table,
    int cta_p_start,
    int actual_m,
    int block_size,
    int kv_head,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    int tid,
    uint64_t l2_pol = 0ull) {
  const int64_t head_v_off = static_cast<int64_t>(kv_head) * stride_kv_head + (IsV ? 128 : 0);
  const int r_sub = tid >> 4;
  const int c = tid & 15;
  const int c_phys = c ^ (r_sub & 7);
  #pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int r = (u << 4) + r_sub;
    if (r < actual_m) {
      const int tok = cta_p_start + r;
      const int lb = (block_size == 16) ? (tok >> 4)
                   : ((block_size == 64) ? (tok >> 6) : (tok / block_size));
      const int slot = (block_size == 16) ? (tok & 15)
                     : ((block_size == 64) ? (tok & 63) : (tok - lb * block_size));
      const int phys_blk = __ldg(block_table + lb);
      const int64_t row_off = static_cast<int64_t>(phys_blk) * stride_block +
                              static_cast<int64_t>(slot) * stride_slot + head_v_off;
      cp_async_16_l2<kKvL2>(smem_u32(&tree_tile[r][c_phys]), kv_ptr + row_off + (c << 3), l2_pol);
    } else {
      tree_tile[r][c_phys] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
}

__device__ __forceinline__ void ldsm4_raw_addr(uint32_t (&b)[4], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
}

__device__ __forceinline__ void ldsm4_trans_raw_addr(uint32_t (&v)[4], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(addr));
}

template <bool kPdl = false>
__global__ __launch_bounds__(256, 1) void gqa16_verify_v4_kernel(
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
    int64_t stride_kv_head,
    const int32_t* __restrict__ cu_seqlens_q_ptr,
    int64_t q_tok_stride,
    int64_t out_tok_stride,
    const int32_t* __restrict__ leaf_flag_ptr) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  VerifyV4Storage& smem = *reinterpret_cast<VerifyV4Storage*>(smem_raw);

  const int kv_head    = blockIdx.y;
  const int tid        = threadIdx.x;
  const int warp_id    = tid >> 5;       // 0..7
  const int q_slot     = warp_id >> 1;   // 0..3 (4 Q tokens / CTA)
  const int w_sub      = warp_id & 1;    // 0 or 1 (2 cooperating warps / Q token)
  const int lane       = tid & 31;
  const int group_id   = lane >> 2;      // 0..7
  const int tid_in_grp = lane & 3;       // 0..3

  const int q0 = cu_seqlens_q_ptr[0];
  const int actual_m = cu_seqlens_q_ptr[1] - q0;
  const int q_base = blockIdx.z * 4;
  if (q_base >= actual_m) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  const int raw_seqlen_k = seqused_k_ptr[0];
  const int lf0 = (leaf_flag_ptr != nullptr) ? leaf_flag_ptr[0] : 0;
  const int my_q_idx = q_base + q_slot;

  bool my_active = false;
  bool my_empty = false;
  int my_seqlen_k = 0;
  int my_k_start = 0;
  int my_num_nb = 0;
  int my_nb_min = 0;
  int my_remap_base = -1;
  uint32_t my_remap_packed = 0u;

  if (my_q_idx < actual_m) {
    int sk = raw_seqlen_k;
    int lrb = -1;
    uint32_t lrp = 0u;
    if (leaf_flag_ptr != nullptr && lf0 < 0 && actual_m <= 32 && sk >= actual_m) {
      const int base = 1 + my_q_idx * 3;
      const int seq_sub = leaf_flag_ptr[base];
      const int base_sub = leaf_flag_ptr[base + 1];
      if (base_sub >= 0) {
        lrb = sk - base_sub;
        lrp = static_cast<uint32_t>(leaf_flag_ptr[base + 2]);
      }
      sk -= seq_sub;
    } else if (my_q_idx == 7 && actual_m == 8 && leaf_flag_ptr != nullptr) {
      if ((lf0 == 1 || lf0 == 2) && sk >= 8) {
        const int lps = sk - 8;
        lrb = lps + lf0;
        lrp = 7u;
        sk = lps + lf0 + 1;
      } else {
        sk -= (actual_m - 1 - my_q_idx);
      }
    } else if (my_q_idx == 6 && actual_m == 7 && leaf_flag_ptr != nullptr && lf0 > 0 && sk >= 7) {
      const int lps = sk - 7;
      lrb = lps + 1;
      lrp = 6u;
      sk -= 5;
    } else {
      sk -= (actual_m - 1 - my_q_idx);
    }
    if (sk < 0) sk = 0;

    const int ks = (window_left >= 0) ? ((sk - 1 - window_left > 0) ? (sk - 1 - window_left) : 0) : 0;
    const int nnb = (sk > 0) ? ((sk + 127) >> 7) : 0;
    const int gnb = (window_left >= 0 && sk > 0) ? (ks >> 7) : 0;
    if (gnb >= nnb) {
      my_empty = true;
    } else {
      my_active = true;
      my_seqlen_k = sk;
      my_k_start = ks;
      my_num_nb = nnb;
      my_nb_min = gnb;
      my_remap_base = lrb;
      my_remap_packed = lrp;
    }
  }

  if (w_sub == 0 && lane == 0) {
    smem.active_q[q_slot] = my_active ? 1u : 0u;
    smem.nb_max_q[q_slot] = my_active ? my_num_nb : 0;
    smem.nb_min_q[q_slot] = my_active ? my_nb_min : 0x7fffffff;
    smem.k_start_q[q_slot] = my_active ? my_k_start : 0x7fffffff;
    smem.seqlen_q[q_slot] = my_active ? my_seqlen_k : 0;
    smem.remap_base_q[q_slot] = (my_active && my_remap_base >= 0) ? my_remap_base : 0x7fffffff;
  }
  if (tid < 16) {
    smem.zero_row[tid] = make_uint4(0u, 0u, 0u, 0u);
  }
  __syncthreads();

  bool cta_any_active = false;
  int cta_nb_max = 0;
  int cta_nb_min = 0x7fffffff;
  int cta_k_start = 0x7fffffff;
  int cta_seqlen_k = 0;
  int cta_remap_min_base = 0x7fffffff;
  #pragma unroll
  for (int qs = 0; qs < 4; ++qs) {
    if (smem.active_q[qs]) {
      cta_any_active = true;
      if (smem.nb_max_q[qs] > cta_nb_max) cta_nb_max = smem.nb_max_q[qs];
      if (smem.nb_min_q[qs] < cta_nb_min) cta_nb_min = smem.nb_min_q[qs];
      if (smem.k_start_q[qs] < cta_k_start) cta_k_start = smem.k_start_q[qs];
      if (smem.seqlen_q[qs] > cta_seqlen_k) cta_seqlen_k = smem.seqlen_q[qs];
      if (smem.remap_base_q[qs] < cta_remap_min_base) cta_remap_min_base = smem.remap_base_q[qs];
    }
  }
  const bool cta_has_remap = (cta_remap_min_base < 0x7fffffff);
  const int cta_p_start = raw_seqlen_k - actual_m;

  // SM121 PDL consumer barrier: wait after independent metadata/remap setup and before Q/KV/out access.
  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  if (my_empty && w_sub == 0) {
    __nv_bfloat16* const dst_base = out_ptr + static_cast<int64_t>(q0 + my_q_idx) * out_tok_stride +
                                    static_cast<int64_t>(kv_head * 16) * out_head_stride;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = (u << 5) + lane;
      const int m = idx >> 4;
      const int c = idx & 15;
      *reinterpret_cast<uint4*>(dst_base + static_cast<int64_t>(m) * out_head_stride + (c << 3)) =
          make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (!cta_any_active) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  // Load each Q token ONCE via w_sub==0 into smem.kv (aliased as q_tiles[4][16][16]), then both warps ldsm4.
  uint32_t q_frag[8][4];
  {
    uint4 (*q_tiles)[16][16] = reinterpret_cast<uint4 (*)[16][16]>(smem.kv);
    if (my_active && w_sub == 0) {
      const __nv_bfloat16* q_tok_base = q_ptr + static_cast<int64_t>(q0 + my_q_idx) * q_tok_stride +
                                        static_cast<int64_t>(kv_head * 16) * q_head_stride;
      #pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int idx = (u << 5) + lane;
        const int m = idx >> 4;
        const int c = idx & 15;
        q_tiles[q_slot][m][c ^ (m & 7)] =
            *reinterpret_cast<const uint4*>(q_tok_base + static_cast<int64_t>(m) * q_head_stride + (c << 3));
      }
    }
    __syncthreads();
    if (my_active) {
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        ldsm4_swizzled(q_frag[ks], q_tiles[q_slot], ks, lane);
      }
    }
  }
  __syncthreads();

  // Each warp owns HALF of head_dim=128: w_sub=0 -> D=0..63 (dp=0..3), w_sub=1 -> D=64..127 (dp=4..7).
  float o_frag[8][4];
  #pragma unroll
  for (int dt = 0; dt < 8; ++dt) {
    o_frag[dt][0] = 0.0f; o_frag[dt][1] = 0.0f; o_frag[dt][2] = 0.0f; o_frag[dt][3] = 0.0f;
  }
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;
  bool my_is_first = true;

  // Prime first block's K half 0 into smem.kv[0].
  issue_kv64_v4<false>(
      smem.kv[0], kv_ptr, block_table, (cta_nb_max - 1) << 7,
      cta_k_start, cta_seqlen_k, block_size, kv_head,
      stride_block, stride_slot, stride_kv_head, tid);
  cp_async_commit();

  const int dp_base = w_sub << 2;  // 0 (D=0..63) or 4 (D=64..127)

  for (int nb = cta_nb_max - 1; nb >= cta_nb_min; --nb) {
    const int block_tok_base = nb << 7;
    const bool nb_has_remap = cta_has_remap && (block_tok_base + 128 > cta_remap_min_base);
    const bool my_in_block = my_active && (nb < my_num_nb) && (nb >= my_nb_min);

    float s_frag[8][4];
    #pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
      s_frag[ni][0] = 0.0f; s_frag[ni][1] = 0.0f; s_frag[ni][2] = 0.0f; s_frag[ni][3] = 0.0f;
    }

    // -------------------------------------------------------------------------
    // 1. QK^T: w_sub=0 owns kv[0] (tokens 0..63), w_sub=1 owns kv[1] (64..127).
    //    All 8 warps execute 32 MMAs in parallel!
    // -------------------------------------------------------------------------
    if (!nb_has_remap) {
      issue_kv64_v4<false>(
          smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();

      if (my_in_block) {
        const uint4 (*my_k_tile)[16] = smem.kv[w_sub];
        uint32_t b_cur[4], b_nxt[4];
        ldsm4_k_pair64_v3(b_cur, my_k_tile, 0, 0, lane);
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          #pragma unroll
          for (int sp = 0; sp < 4; ++sp) {
            const int n_sp = (sp + 1) & 3;
            const int n_ks = ks + (sp == 3 ? 1 : 0);
            if (n_ks < 8) ldsm4_k_pair64_v3(b_nxt, my_k_tile, n_sp, n_ks, lane);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[0], b_cur[1], s_frag[sp * 2 + 0]);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[2], b_cur[3], s_frag[sp * 2 + 1]);
            b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1]; b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
          }
        }
      }
    } else {
      issue_kv64_v4<false>(
          smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      issue_tree_kv_v4<false>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();

      if (my_in_block) {
        const uint4 (*my_k_tile)[16] = smem.kv[w_sub];
        const int chunk_base = block_tok_base + (w_sub << 6);
        const int r_local = lane & 7;
        const int sub_k = (lane >> 3) & 1;
        uint32_t k_row_base[4];
        int k_swz[4];
        #pragma unroll
        for (int sp = 0; sp < 4; ++sp) {
          const int sub = (sp << 1) + (lane >> 4);
          const int r = (sub << 3) + r_local;
          const int tok = chunk_base + r;
          if (tok >= my_k_start && tok < my_seqlen_k) {
            if (my_remap_base >= 0 && tok >= my_remap_base) {
              const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
              k_row_base[sp] = smem_u32(&smem.tree_kv[rr][0]);
              k_swz[sp] = rr & 7;
            } else {
              k_row_base[sp] = smem_u32(&my_k_tile[r][0]);
              k_swz[sp] = r_local;
            }
          } else {
            k_row_base[sp] = smem_u32(&smem.zero_row[0]);
            k_swz[sp] = 0;
          }
        }
        uint32_t b_cur[4], b_nxt[4];
        ldsm4_raw_addr(b_cur, k_row_base[0] + ((sub_k ^ k_swz[0]) << 4));
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          #pragma unroll
          for (int sp = 0; sp < 4; ++sp) {
            const int n_sp = (sp + 1) & 3;
            const int n_ks = ks + (sp == 3 ? 1 : 0);
            if (n_ks < 8) {
              ldsm4_raw_addr(b_nxt, k_row_base[n_sp] + ((((n_ks << 1) + sub_k) ^ k_swz[n_sp]) << 4));
            }
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[0], b_cur[1], s_frag[sp * 2 + 0]);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[2], b_cur[3], s_frag[sp * 2 + 1]);
            b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1]; b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
          }
        }
      }
    }

    // Local QK masking + row max per half (before barrier so barrier publishes max_exch & frees kv[0..1]).
    if (my_in_block) {
      const int chunk_base = block_tok_base + (w_sub << 6);
      const bool need_mask = (chunk_base < my_k_start) || (chunk_base + 64 > my_seqlen_k);
      float m0 = -INFINITY, m1 = -INFINITY;
      #pragma unroll
      for (int ni = 0; ni < 8; ++ni) {
        if (need_mask) {
          const int col0 = chunk_base + (ni << 3) + (tid_in_grp << 1);
          const int col1 = col0 + 1;
          if (col0 < my_k_start || col0 >= my_seqlen_k) { s_frag[ni][0] = -INFINITY; s_frag[ni][2] = -INFINITY; }
          if (col1 < my_k_start || col1 >= my_seqlen_k) { s_frag[ni][1] = -INFINITY; s_frag[ni][3] = -INFINITY; }
        }
        m0 = max_ftz_f32(m0, max_ftz_f32(s_frag[ni][0], s_frag[ni][1]));
        m1 = max_ftz_f32(m1, max_ftz_f32(s_frag[ni][2], s_frag[ni][3]));
      }
      m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
      m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
      m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
      m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
      if (tid_in_grp == 0) {
        smem.max_exch[q_slot][w_sub][group_id] = m0;
        smem.max_exch[q_slot][w_sub][group_id + 8] = m1;
      }
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // 2. Issue BOTH V[0] and V[1] (and tree_V if remap) behind parallel Softmax!
    // -------------------------------------------------------------------------
    if (!nb_has_remap) {
      issue_kv64_v4<true>(
          smem.kv[0], kv_ptr, block_table, block_tok_base,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
      issue_kv64_v4<true>(
          smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
    } else {
      issue_kv64_v4<true>(
          smem.kv[0], kv_ptr, block_table, block_tok_base,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      issue_kv64_v4<true>(
          smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      issue_tree_kv_v4<true>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
    }

    if constexpr (kPdl) {
      if (nb == cta_nb_min) {
        asm volatile("griddepcontrol.launch_dependents;");
      }
    }

    // Parallel softmax across both warps (32 MUFU.EX2 per warp = 2x softmax throughput!).
    // s_frag[0..3] holds packed BF16x2 P fragments in-place: p_lo (w_sub=0) or p_hi (w_sub=1).
    if (my_in_block) {
      const float m0 = max_ftz_f32(smem.max_exch[q_slot][0][group_id], smem.max_exch[q_slot][1][group_id]);
      const float m1 = max_ftz_f32(smem.max_exch[q_slot][0][group_id + 8], smem.max_exch[q_slot][1][group_id + 8]);

      float sc0 = 1.0f, sc1 = 1.0f;
      if (my_is_first) {
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
        for (int dt = 0; dt < 8; ++dt) {
          o_frag[dt][0] = mul_ftz_f32(o_frag[dt][0], sc0);
          o_frag[dt][1] = mul_ftz_f32(o_frag[dt][1], sc0);
          o_frag[dt][2] = mul_ftz_f32(o_frag[dt][2], sc1);
          o_frag[dt][3] = mul_ftz_f32(o_frag[dt][3], sc1);
        }
      }

      const float max_sc0 = (row_max0 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max0);
      const float max_sc1 = (row_max1 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max1);
      #pragma unroll
      for (int ni = 0; ni < 8; ++ni) {
        const float s0 = s_frag[ni][0];
        const float s1 = s_frag[ni][1];
        const float s2 = s_frag[ni][2];
        const float s3 = s_frag[ni][3];
        const float p0 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s0, scale_log2), max_sc0));
        const float p1 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s1, scale_log2), max_sc0));
        const float p2 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s2, scale_log2), max_sc1));
        const float p3 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s3, scale_log2), max_sc1));
        if (w_sub == 0) {
          if (ni == 0) {
            if (my_is_first) {
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
        } else {
          smem.p_hi_f32[q_slot][ni][lane] = make_float4(p0, p1, p2, p3);
        }
        const __nv_bfloat162 pb0 = __floats2bfloat162_rn(p0, p1);
        const __nv_bfloat162 pb1 = __floats2bfloat162_rn(p2, p3);
        const int kp = ni >> 1;
        const int hi = (ni & 1) << 1;
        s_frag[kp][hi + 0] = __uint_as_float(*reinterpret_cast<const uint32_t*>(&pb0));
        s_frag[kp][hi + 1] = __uint_as_float(*reinterpret_cast<const uint32_t*>(&pb1));
      }
      if (w_sub == 0) {
        #pragma unroll
        for (int kp = 0; kp < 4; ++kp) {
          smem.p_lo_bf16[q_slot][kp][lane] = make_uint4(
              __float_as_uint(s_frag[kp][0]),
              __float_as_uint(s_frag[kp][1]),
              __float_as_uint(s_frag[kp][2]),
              __float_as_uint(s_frag[kp][3]));
        }
      }
      my_is_first = false;
    }

    // -------------------------------------------------------------------------
    // 3. PV: All 8 warps execute 32 MMAs per V half in parallel (D-split)!
    // -------------------------------------------------------------------------
    if (!nb_has_remap) {
      cp_async_wait<1>();
      __syncthreads();
      if (my_in_block) {
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_v_pair64_v3(v_cur, smem.kv[0], dp_base, 0, lane);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          uint32_t p_cur[4];
          if (w_sub == 0) {
            p_cur[0] = __float_as_uint(s_frag[ks_local][0]);
            p_cur[1] = __float_as_uint(s_frag[ks_local][1]);
            p_cur[2] = __float_as_uint(s_frag[ks_local][2]);
            p_cur[3] = __float_as_uint(s_frag[ks_local][3]);
          } else {
            const uint4 pl = smem.p_lo_bf16[q_slot][ks_local][lane];
            p_cur[0] = pl.x; p_cur[1] = pl.y; p_cur[2] = pl.z; p_cur[3] = pl.w;
          }
          #pragma unroll
          for (int dp = 0; dp < 4; ++dp) {
            const int n_dp = (dp + 1) & 3;
            const int n_ks = ks_local + (dp == 3 ? 1 : 0);
            if (n_ks < 4) ldsm4_v_pair64_v3(v_nxt, smem.kv[0], dp_base + n_dp, n_ks, lane);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
        // w_sub=0 finishes exact ni=8..15 row_sum and packs p_hi into s_frag[0..3] while V[1] arrives!
        if (w_sub == 0) {
          #pragma unroll
          for (int ni = 0; ni < 8; ++ni) {
            const float4 ph = smem.p_hi_f32[q_slot][ni][lane];
            row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph.x), ph.y);
            row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph.z), ph.w);
            const __nv_bfloat162 pb0 = __floats2bfloat162_rn(ph.x, ph.y);
            const __nv_bfloat162 pb1 = __floats2bfloat162_rn(ph.z, ph.w);
            const int kp = ni >> 1;
            const int hi = (ni & 1) << 1;
            s_frag[kp][hi + 0] = __uint_as_float(*reinterpret_cast<const uint32_t*>(&pb0));
            s_frag[kp][hi + 1] = __uint_as_float(*reinterpret_cast<const uint32_t*>(&pb1));
          }
        }
      }
      __syncthreads();

      if (nb > cta_nb_min) {
        issue_kv64_v4<false>(
            smem.kv[0], kv_ptr, block_table, (nb - 1) << 7,
            cta_k_start, cta_seqlen_k, block_size, kv_head,
            stride_block, stride_slot, stride_kv_head, tid);
        cp_async_commit();
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
      }
      __syncthreads();

      if (my_in_block) {
        // Now BOTH w_sub=0 and w_sub=1 hold p_hi in registers s_frag[0..3]!
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_v_pair64_v3(v_cur, smem.kv[1], dp_base, 0, lane);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          const uint32_t p_cur[4] = {
              __float_as_uint(s_frag[ks_local][0]),
              __float_as_uint(s_frag[ks_local][1]),
              __float_as_uint(s_frag[ks_local][2]),
              __float_as_uint(s_frag[ks_local][3]),
          };
          #pragma unroll
          for (int dp = 0; dp < 4; ++dp) {
            const int n_dp = (dp + 1) & 3;
            const int n_ks = ks_local + (dp == 3 ? 1 : 0);
            if (n_ks < 4) ldsm4_v_pair64_v3(v_nxt, smem.kv[1], dp_base + n_dp, n_ks, lane);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
      }
      __syncthreads();
    } else {
      cp_async_wait<0>();
      __syncthreads();
      if (my_in_block) {
        const int r_in_16 = lane & 15;
        const int dt_sub = lane >> 4;
        // PV Half 0 (tokens 0..63) with hoisted per-lane SMEM row addresses + double-buffered ldmatrix!
        {
          uint32_t v_row_base[4];
          int v_swz[4];
          #pragma unroll
          for (int kl = 0; kl < 4; ++kl) {
            const int r = (kl << 4) + r_in_16;
            const int tok = block_tok_base + r;
            if (tok >= my_k_start && tok < my_seqlen_k) {
              if (my_remap_base >= 0 && tok >= my_remap_base) {
                const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
                v_row_base[kl] = smem_u32(&smem.tree_kv[rr][0]);
                v_swz[kl] = rr & 7;
              } else {
                v_row_base[kl] = smem_u32(&smem.kv[0][r][0]);
                v_swz[kl] = r & 7;
              }
            } else {
              v_row_base[kl] = smem_u32(&smem.zero_row[0]);
              v_swz[kl] = 0;
            }
          }
          uint32_t v_cur[4], v_nxt[4];
          ldsm4_trans_raw_addr(v_cur, v_row_base[0] + (((dp_base << 1) + dt_sub) ^ v_swz[0]) * 16u);
          #pragma unroll
          for (int ks_local = 0; ks_local < 4; ++ks_local) {
            uint32_t p_cur[4];
            if (w_sub == 0) {
              p_cur[0] = __float_as_uint(s_frag[ks_local][0]);
              p_cur[1] = __float_as_uint(s_frag[ks_local][1]);
              p_cur[2] = __float_as_uint(s_frag[ks_local][2]);
              p_cur[3] = __float_as_uint(s_frag[ks_local][3]);
            } else {
              const uint4 pl = smem.p_lo_bf16[q_slot][ks_local][lane];
              p_cur[0] = pl.x; p_cur[1] = pl.y; p_cur[2] = pl.z; p_cur[3] = pl.w;
            }
            #pragma unroll
            for (int dp = 0; dp < 4; ++dp) {
              const int n_dp = (dp + 1) & 3;
              const int n_ks = ks_local + (dp == 3 ? 1 : 0);
              if (n_ks < 4) {
                const int n_dt = ((dp_base + n_dp) << 1) + dt_sub;
                ldsm4_trans_raw_addr(v_nxt, v_row_base[n_ks] + ((n_dt ^ v_swz[n_ks]) << 4));
              }
              mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
              mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
              v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
            }
          }
        }
        if (w_sub == 0) {
          #pragma unroll
          for (int ni = 0; ni < 8; ++ni) {
            const float4 ph = smem.p_hi_f32[q_slot][ni][lane];
            row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph.x), ph.y);
            row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph.z), ph.w);
            const __nv_bfloat162 pb0 = __floats2bfloat162_rn(ph.x, ph.y);
            const __nv_bfloat162 pb1 = __floats2bfloat162_rn(ph.z, ph.w);
            const int kp = ni >> 1;
            const int hi = (ni & 1) << 1;
            s_frag[kp][hi + 0] = __uint_as_float(*reinterpret_cast<const uint32_t*>(&pb0));
            s_frag[kp][hi + 1] = __uint_as_float(*reinterpret_cast<const uint32_t*>(&pb1));
          }
        }
        // PV Half 1 (tokens 64..127) — V[1] & tree_kv already in SMEM, zero extra barrier!
        {
          uint32_t v_row_base[4];
          int v_swz[4];
          #pragma unroll
          for (int kl = 0; kl < 4; ++kl) {
            const int r = (kl << 4) + r_in_16;
            const int tok = block_tok_base + 64 + r;
            if (tok >= my_k_start && tok < my_seqlen_k) {
              if (my_remap_base >= 0 && tok >= my_remap_base) {
                const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
                v_row_base[kl] = smem_u32(&smem.tree_kv[rr][0]);
                v_swz[kl] = rr & 7;
              } else {
                v_row_base[kl] = smem_u32(&smem.kv[1][r][0]);
                v_swz[kl] = r & 7;
              }
            } else {
              v_row_base[kl] = smem_u32(&smem.zero_row[0]);
              v_swz[kl] = 0;
            }
          }
          uint32_t v_cur[4], v_nxt[4];
          ldsm4_trans_raw_addr(v_cur, v_row_base[0] + (((dp_base << 1) + dt_sub) ^ v_swz[0]) * 16u);
          #pragma unroll
          for (int ks_local = 0; ks_local < 4; ++ks_local) {
            const uint32_t p_cur[4] = {
                __float_as_uint(s_frag[ks_local][0]),
                __float_as_uint(s_frag[ks_local][1]),
                __float_as_uint(s_frag[ks_local][2]),
                __float_as_uint(s_frag[ks_local][3]),
            };
            #pragma unroll
            for (int dp = 0; dp < 4; ++dp) {
              const int n_dp = (dp + 1) & 3;
              const int n_ks = ks_local + (dp == 3 ? 1 : 0);
              if (n_ks < 4) {
                const int n_dt = ((dp_base + n_dp) << 1) + dt_sub;
                ldsm4_trans_raw_addr(v_nxt, v_row_base[n_ks] + ((n_dt ^ v_swz[n_ks]) << 4));
              }
              mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
              mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
              v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
            }
          }
        }
      }
      __syncthreads();
      if (nb > cta_nb_min) {
        issue_kv64_v4<false>(
            smem.kv[0], kv_ptr, block_table, (nb - 1) << 7,
            cta_k_start, cta_seqlen_k, block_size, kv_head,
            stride_block, stride_slot, stride_kv_head, tid);
        cp_async_commit();
      }
    }
  }

  // Final normalization: w_sub=0 reduces row_sum & broadcasts inv0/inv1, both warps store 8 D-tiles in parallel!
  if (my_active && w_sub == 0) {
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 2));
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 1));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 2));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 1));
    const bool empty0 = (row_sum0 == 0.0f || row_sum0 != row_sum0 || row_max0 == -INFINITY);
    const bool empty1 = (row_sum1 == 0.0f || row_sum1 != row_sum1 || row_max1 == -INFINITY);
    const float inv0 = empty0 ? 1.0f : fast_rcpf(row_sum0);
    const float inv1 = empty1 ? 1.0f : fast_rcpf(row_sum1);
    if (tid_in_grp == 0) {
      smem.inv_exch[q_slot][group_id] = inv0;
      smem.inv_exch[q_slot][group_id + 8] = inv1;
    }
  }
  __syncthreads();

  if (my_active) {
    const float inv0 = smem.inv_exch[q_slot][group_id];
    const float inv1 = smem.inv_exch[q_slot][group_id + 8];
    const int d_off = w_sub << 6;  // 0 or 64
    __nv_bfloat16* const out_row = out_ptr + static_cast<int64_t>(q0 + my_q_idx) * out_tok_stride;
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
      const int d_col = d_off + (dt << 3) + (tid_in_grp << 1);
      const __nv_bfloat162 ob0 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[dt][0], inv0), mul_ftz_f32(o_frag[dt][1], inv0));
      const __nv_bfloat162 ob1 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[dt][2], inv1), mul_ftz_f32(o_frag[dt][3], inv1));
      *reinterpret_cast<uint32_t*>(
          out_row + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob0);
      *reinterpret_cast<uint32_t*>(
          out_row + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob1);
    }
  }
}

// ============================================================================
// Wave-86d V4 Zero-Stall 3-Slot KV Ring Shared-KV FA2 Verify Kernel (4Q x 2 warps/Q)
//
// Strictly 0-ULP bit-identical to FlashAttention-2 kBlockN=128 & verify_v4:
//   - Same 16 CTAs at M=32 (100% single-wave co-residency with 32-CTA FP8/BF16 gate!).
//   - Expands KV SMEM from 2 -> 3 slots (kv[3][64][16] = 48 KiB; total 83,072 B <= 99,328 B,
//     zero occupancy or carveout change vs VerifyV4Storage's 66,688 B!).
//   - Prologue puts q_tiles[4][16][16] in kv[2] while prefetching block-0 K[0]/K[1](/tree_K)
//     into kv[0]/kv[1](/tree_kv) concurrently with Q ldsm4!
//   - Main loop rotates (s0, s1, s2):
//       * V[0] -> kv[s2] issued BEFORE QK^T (hidden behind QK^T + Softmax!)
//       * (tree_V -> tree_kv if remap) + V[1] -> kv[s0] + next K[0] -> kv[s1] issued after QK^T
//       * next K[1] -> kv[s2] issued BEFORE PV Half 1 (hidden behind PV Half 1!)
//     Eliminates the cold K[1] cp_async_wait<0>() stall, pipelines nb_has_remap, and cuts
//     CTA barriers from 6 -> 4 per 128-token block!
//   - Interleaves w_sub=0's p_hi_f32 row_sum + BF16 pack inside PV Half 0's 4-step ks_local
//     HMMA loop (exact ascending ni=0..7 add_ftz_f32 order, 0 extra regs, 0 serial tail!).
// ============================================================================
struct __align__(128) VerifyV4Ring3Storage {
  uint4 kv[3][64][16];         // 49,152 B (3x16 KiB ring; kv[2] aliases q_tiles[4][16][16] in prologue)
  uint4 tree_kv[32][16];       //  8,192 B (up to 32 tree K/V rows for remapped tail block)
  uint4 zero_row[16];          //    256 B (constant +0.0 BF16 row for out-of-range lanes)
  float4 p_hi_f32[4][8][32];   // 16,384 B (w_sub=1 -> w_sub=0 exact ni=8..15 row_sum + p_hi)
  uint4 p_lo_bf16[4][4][32];   //  8,192 B (w_sub=0 -> w_sub=1 packed BF16x2 P[0..3])
  float max_exch[4][2][16];    //    512 B (per-warp QK row max exchange)
  float inv_exch[4][16];       //    256 B (final inv_row_sum broadcast to w_sub=1)
  int32_t nb_max_q[4];         //     16 B
  int32_t nb_min_q[4];         //     16 B
  int32_t k_start_q[4];        //     16 B
  int32_t seqlen_q[4];         //     16 B
  int32_t remap_base_q[4];     //     16 B
  uint8_t active_q[4];         //      4 B
};                             // 83,072 B (<= 99,328 B dynamic SMEM on sm_121a)
static_assert(sizeof(VerifyV4Ring3Storage) <= 99328, "VerifyV4Ring3Storage must fit SM121 dynamic SMEM");

template <bool kPdl = false, int kKvL2 = 0>
__global__ __launch_bounds__(256, 1) void gqa16_verify_v4_ring3_kernel(
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
    int64_t stride_kv_head,
    const int32_t* __restrict__ cu_seqlens_q_ptr,
    int64_t q_tok_stride,
    int64_t out_tok_stride,
    const int32_t* __restrict__ leaf_flag_ptr) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  VerifyV4Ring3Storage& smem = *reinterpret_cast<VerifyV4Ring3Storage*>(smem_raw);

  const int kv_head    = blockIdx.y;
  const int tid        = threadIdx.x;
  const int warp_id    = tid >> 5;       // 0..7
  const int q_slot     = warp_id >> 1;   // 0..3 (4 Q tokens / CTA)
  const int w_sub      = warp_id & 1;    // 0 or 1 (2 cooperating warps / Q token)
  const int lane       = tid & 31;
  const int group_id   = lane >> 2;      // 0..7
  const int tid_in_grp = lane & 3;       // 0..3

  const int q0 = cu_seqlens_q_ptr[0];
  const int actual_m = cu_seqlens_q_ptr[1] - q0;
  const int q_base = blockIdx.z * 4;
  if (q_base >= actual_m) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  const int raw_seqlen_k = seqused_k_ptr[0];
  const int lf0 = (leaf_flag_ptr != nullptr) ? leaf_flag_ptr[0] : 0;
  const int my_q_idx = q_base + q_slot;

  bool my_active = false;
  bool my_empty = false;
  int my_seqlen_k = 0;
  int my_k_start = 0;
  int my_num_nb = 0;
  int my_nb_min = 0;
  int my_remap_base = -1;
  uint32_t my_remap_packed = 0u;

  if (my_q_idx < actual_m) {
    int sk = raw_seqlen_k;
    int lrb = -1;
    uint32_t lrp = 0u;
    if (leaf_flag_ptr != nullptr && lf0 < 0 && actual_m <= 32 && sk >= actual_m) {
      const int base = 1 + my_q_idx * 3;
      const int seq_sub = leaf_flag_ptr[base];
      const int base_sub = leaf_flag_ptr[base + 1];
      if (base_sub >= 0) {
        lrb = sk - base_sub;
        lrp = static_cast<uint32_t>(leaf_flag_ptr[base + 2]);
      }
      sk -= seq_sub;
    } else if (my_q_idx == 7 && actual_m == 8 && leaf_flag_ptr != nullptr) {
      if ((lf0 == 1 || lf0 == 2) && sk >= 8) {
        const int lps = sk - 8;
        lrb = lps + lf0;
        lrp = 7u;
        sk = lps + lf0 + 1;
      } else {
        sk -= (actual_m - 1 - my_q_idx);
      }
    } else if (my_q_idx == 6 && actual_m == 7 && leaf_flag_ptr != nullptr && lf0 > 0 && sk >= 7) {
      const int lps = sk - 7;
      lrb = lps + 1;
      lrp = 6u;
      sk -= 5;
    } else {
      sk -= (actual_m - 1 - my_q_idx);
    }
    if (sk < 0) sk = 0;

    const int ks = (window_left >= 0) ? ((sk - 1 - window_left > 0) ? (sk - 1 - window_left) : 0) : 0;
    const int nnb = (sk > 0) ? ((sk + 127) >> 7) : 0;
    const int gnb = (window_left >= 0 && sk > 0) ? (ks >> 7) : 0;
    if (gnb >= nnb) {
      my_empty = true;
    } else {
      my_active = true;
      my_seqlen_k = sk;
      my_k_start = ks;
      my_num_nb = nnb;
      my_nb_min = gnb;
      my_remap_base = lrb;
      my_remap_packed = lrp;
    }
  }

  if (w_sub == 0 && lane == 0) {
    smem.active_q[q_slot] = my_active ? 1u : 0u;
    smem.nb_max_q[q_slot] = my_active ? my_num_nb : 0;
    smem.nb_min_q[q_slot] = my_active ? my_nb_min : 0x7fffffff;
    smem.k_start_q[q_slot] = my_active ? my_k_start : 0x7fffffff;
    smem.seqlen_q[q_slot] = my_active ? my_seqlen_k : 0;
    smem.remap_base_q[q_slot] = (my_active && my_remap_base >= 0) ? my_remap_base : 0x7fffffff;
  }
  if (tid < 16) {
    smem.zero_row[tid] = make_uint4(0u, 0u, 0u, 0u);
  }
  __syncthreads();

  bool cta_any_active = false;
  int cta_nb_max = 0;
  int cta_nb_min = 0x7fffffff;
  int cta_k_start = 0x7fffffff;
  int cta_seqlen_k = 0;
  int cta_remap_min_base = 0x7fffffff;
  #pragma unroll
  for (int qs = 0; qs < 4; ++qs) {
    if (smem.active_q[qs]) {
      cta_any_active = true;
      if (smem.nb_max_q[qs] > cta_nb_max) cta_nb_max = smem.nb_max_q[qs];
      if (smem.nb_min_q[qs] < cta_nb_min) cta_nb_min = smem.nb_min_q[qs];
      if (smem.k_start_q[qs] < cta_k_start) cta_k_start = smem.k_start_q[qs];
      if (smem.seqlen_q[qs] > cta_seqlen_k) cta_seqlen_k = smem.seqlen_q[qs];
      if (smem.remap_base_q[qs] < cta_remap_min_base) cta_remap_min_base = smem.remap_base_q[qs];
    }
  }
  const bool cta_has_remap = (cta_remap_min_base < 0x7fffffff);
  const int cta_p_start = raw_seqlen_k - actual_m;

  // SM121 PDL consumer barrier: wait after independent metadata/remap setup and before Q/KV/out access.
  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  if (my_empty && w_sub == 0) {
    __nv_bfloat16* const dst_base = out_ptr + static_cast<int64_t>(q0 + my_q_idx) * out_tok_stride +
                                    static_cast<int64_t>(kv_head * 16) * out_head_stride;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = (u << 5) + lane;
      const int m = idx >> 4;
      const int c = idx & 15;
      *reinterpret_cast<uint4*>(dst_base + static_cast<int64_t>(m) * out_head_stride + (c << 3)) =
          make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (!cta_any_active) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  // Prologue: load Q into kv[2] (aliased as q_tiles[4][16][16]) AND prime block-0 K[0]/K[1](/tree_K)
  // into kv[0]/kv[1](/tree_kv) concurrently so block-0 K overlaps Q ldsm4 + accumulator init!
  const uint64_t kv_l2_pol = (kKvL2 == 2) ? kv_pol_evict_last() : 0ull;
  uint32_t q_frag[8][4];
  {
    uint4 (*q_tiles)[16][16] = reinterpret_cast<uint4 (*)[16][16]>(smem.kv[2]);
    if (my_active && w_sub == 0) {
      const __nv_bfloat16* q_tok_base = q_ptr + static_cast<int64_t>(q0 + my_q_idx) * q_tok_stride +
                                        static_cast<int64_t>(kv_head * 16) * q_head_stride;
      #pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int idx = (u << 5) + lane;
        const int m = idx >> 4;
        const int c = idx & 15;
        q_tiles[q_slot][m][c ^ (m & 7)] =
            *reinterpret_cast<const uint4*>(q_tok_base + static_cast<int64_t>(m) * q_head_stride + (c << 3));
      }
    }
    const int nb0_tok_base = (cta_nb_max - 1) << 7;
    issue_kv64_v4<false, kKvL2>(
        smem.kv[0], kv_ptr, block_table, nb0_tok_base,
        cta_k_start, cta_seqlen_k, block_size, kv_head,
        stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    issue_kv64_v4<false, kKvL2>(
        smem.kv[1], kv_ptr, block_table, nb0_tok_base + 64,
        cta_k_start, cta_seqlen_k, block_size, kv_head,
        stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    if (cta_has_remap && (nb0_tok_base + 128 > cta_remap_min_base)) {
      issue_tree_kv_v4<false, kKvL2>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    }
    cp_async_commit();
    __syncthreads();
    if (my_active) {
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        ldsm4_swizzled(q_frag[ks], q_tiles[q_slot], ks, lane);
      }
    }
  }

  float o_frag[8][4];
  #pragma unroll
  for (int dt = 0; dt < 8; ++dt) {
    o_frag[dt][0] = 0.0f; o_frag[dt][1] = 0.0f; o_frag[dt][2] = 0.0f; o_frag[dt][3] = 0.0f;
  }
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;
  bool my_is_first = true;

  const int dp_base = w_sub << 2;  // 0 (D=0..63) or 4 (D=64..127)
  int s0 = 0, s1 = 1, s2 = 2;      // kv[s0]=K[0], kv[s1]=K[1], kv[s2]=free after Barrier #1

  for (int nb = cta_nb_max - 1; nb >= cta_nb_min; --nb) {
    const int block_tok_base = nb << 7;
    const bool nb_has_remap = cta_has_remap && (block_tok_base + 128 > cta_remap_min_base);
    const bool my_in_block = my_active && (nb < my_num_nb) && (nb >= my_nb_min);

    // Barrier #1: K[0] (kv[s0]) & K[1] (kv[s1]) ready; kv[s2] (prior Q or PV1) is now free!
    cp_async_wait<0>();
    __syncthreads();

    // Rare boundary case: 2nd consecutive remapped block reloads tree_K after prior PV1 freed tree_kv.
    if (nb_has_remap && nb < cta_nb_max - 1) {
      issue_tree_kv_v4<false, kKvL2>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();
    }

    // Issue V[0] into free kv[s2] BEFORE QK^T so V[0] downloads in parallel with QK^T + Softmax!
    issue_kv64_v4<true, kKvL2>(
        smem.kv[s2], kv_ptr, block_table, block_tok_base,
        cta_k_start, cta_seqlen_k, block_size, kv_head,
        stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    cp_async_commit();

    float s_frag[8][4];
    #pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
      s_frag[ni][0] = 0.0f; s_frag[ni][1] = 0.0f; s_frag[ni][2] = 0.0f; s_frag[ni][3] = 0.0f;
    }

    // -------------------------------------------------------------------------
    // 1. QK^T: w_sub=0 reads kv[s0] (0..63), w_sub=1 reads kv[s1] (64..127).
    // -------------------------------------------------------------------------
    if (!nb_has_remap) {
      if (my_in_block) {
        const uint4 (*my_k_tile)[16] = (w_sub == 0) ? smem.kv[s0] : smem.kv[s1];
        uint32_t b_cur[4], b_nxt[4];
        ldsm4_k_pair64_v3(b_cur, my_k_tile, 0, 0, lane);
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          #pragma unroll
          for (int sp = 0; sp < 4; ++sp) {
            const int n_sp = (sp + 1) & 3;
            const int n_ks = ks + (sp == 3 ? 1 : 0);
            if (n_ks < 8) ldsm4_k_pair64_v3(b_nxt, my_k_tile, n_sp, n_ks, lane);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[0], b_cur[1], s_frag[sp * 2 + 0]);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[2], b_cur[3], s_frag[sp * 2 + 1]);
            b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1]; b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
          }
        }
      }
    } else {
      if (my_in_block) {
        const uint4 (*my_k_tile)[16] = (w_sub == 0) ? smem.kv[s0] : smem.kv[s1];
        const int chunk_base = block_tok_base + (w_sub << 6);
        const int r_local = lane & 7;
        const int sub_k = (lane >> 3) & 1;
        uint32_t k_row_base[4];
        int k_swz[4];
        #pragma unroll
        for (int sp = 0; sp < 4; ++sp) {
          const int sub = (sp << 1) + (lane >> 4);
          const int r = (sub << 3) + r_local;
          const int tok = chunk_base + r;
          if (tok >= my_k_start && tok < my_seqlen_k) {
            if (my_remap_base >= 0 && tok >= my_remap_base) {
              const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
              k_row_base[sp] = smem_u32(&smem.tree_kv[rr][0]);
              k_swz[sp] = rr & 7;
            } else {
              k_row_base[sp] = smem_u32(&my_k_tile[r][0]);
              k_swz[sp] = r_local;
            }
          } else {
            k_row_base[sp] = smem_u32(&smem.zero_row[0]);
            k_swz[sp] = 0;
          }
        }
        uint32_t b_cur[4], b_nxt[4];
        ldsm4_raw_addr(b_cur, k_row_base[0] + ((sub_k ^ k_swz[0]) << 4));
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          #pragma unroll
          for (int sp = 0; sp < 4; ++sp) {
            const int n_sp = (sp + 1) & 3;
            const int n_ks = ks + (sp == 3 ? 1 : 0);
            if (n_ks < 8) {
              ldsm4_raw_addr(b_nxt, k_row_base[n_sp] + ((((n_ks << 1) + sub_k) ^ k_swz[n_sp]) << 4));
            }
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[0], b_cur[1], s_frag[sp * 2 + 0]);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[2], b_cur[3], s_frag[sp * 2 + 1]);
            b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1]; b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
          }
        }
      }
    }

    // Local QK masking + row max per half before Barrier #2 (publishes max_exch & frees kv[s0]/kv[s1]/tree_kv).
    if (my_in_block) {
      const int chunk_base = block_tok_base + (w_sub << 6);
      const bool need_mask = (chunk_base < my_k_start) || (chunk_base + 64 > my_seqlen_k);
      float m0 = -INFINITY, m1 = -INFINITY;
      #pragma unroll
      for (int ni = 0; ni < 8; ++ni) {
        if (need_mask) {
          const int col0 = chunk_base + (ni << 3) + (tid_in_grp << 1);
          const int col1 = col0 + 1;
          if (col0 < my_k_start || col0 >= my_seqlen_k) { s_frag[ni][0] = -INFINITY; s_frag[ni][2] = -INFINITY; }
          if (col1 < my_k_start || col1 >= my_seqlen_k) { s_frag[ni][1] = -INFINITY; s_frag[ni][3] = -INFINITY; }
        }
        m0 = max_ftz_f32(m0, max_ftz_f32(s_frag[ni][0], s_frag[ni][1]));
        m1 = max_ftz_f32(m1, max_ftz_f32(s_frag[ni][2], s_frag[ni][3]));
      }
      m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
      m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
      m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
      m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
      if (tid_in_grp == 0) {
        smem.max_exch[q_slot][w_sub][group_id] = m0;
        smem.max_exch[q_slot][w_sub][group_id + 8] = m1;
      }
    }
    // Barrier #2: max_exch visible; kv[s0], kv[s1], and tree_kv are now free!
    __syncthreads();

    // -------------------------------------------------------------------------
    // 2. Issue (tree_V if remap) + V[1] -> kv[s0] + next K[0] -> kv[s1] behind Softmax!
    //    Pending groups after this section:
    //      * nb > cta_nb_min: {V[0], [tree_V], V[1], next K[0]} -> wait<2> waits V[0](+tree_V)!
    //      * nb == cta_nb_min: {V[0], [tree_V], V[1]}           -> wait<1> waits V[0](+tree_V)!
    // -------------------------------------------------------------------------
    if (nb_has_remap) {
      issue_tree_kv_v4<true, kKvL2>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
      cp_async_commit();
    }
    issue_kv64_v4<true, kKvL2>(
        smem.kv[s0], kv_ptr, block_table, block_tok_base + 64,
        cta_k_start, cta_seqlen_k, block_size, kv_head,
        stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    cp_async_commit();
    if (nb > cta_nb_min) {
      issue_kv64_v4<false, kKvL2>(
          smem.kv[s1], kv_ptr, block_table, (nb - 1) << 7,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
      cp_async_commit();
    }

    if constexpr (kPdl) {
      if (nb == cta_nb_min) {
        asm volatile("griddepcontrol.launch_dependents;");
      }
    }

    // Parallel softmax across both warps (exact FA2 math order).
    if (my_in_block) {
      const float m0 = max_ftz_f32(smem.max_exch[q_slot][0][group_id], smem.max_exch[q_slot][1][group_id]);
      const float m1 = max_ftz_f32(smem.max_exch[q_slot][0][group_id + 8], smem.max_exch[q_slot][1][group_id + 8]);

      float sc0 = 1.0f, sc1 = 1.0f;
      if (my_is_first) {
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
        for (int dt = 0; dt < 8; ++dt) {
          o_frag[dt][0] = mul_ftz_f32(o_frag[dt][0], sc0);
          o_frag[dt][1] = mul_ftz_f32(o_frag[dt][1], sc0);
          o_frag[dt][2] = mul_ftz_f32(o_frag[dt][2], sc1);
          o_frag[dt][3] = mul_ftz_f32(o_frag[dt][3], sc1);
        }
      }

      const float max_sc0 = (row_max0 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max0);
      const float max_sc1 = (row_max1 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max1);
      #pragma unroll
      for (int ni = 0; ni < 8; ++ni) {
        const float s0_v = s_frag[ni][0];
        const float s1_v = s_frag[ni][1];
        const float s2_v = s_frag[ni][2];
        const float s3_v = s_frag[ni][3];
        const float p0 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s0_v, scale_log2), max_sc0));
        const float p1 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s1_v, scale_log2), max_sc0));
        const float p2 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s2_v, scale_log2), max_sc1));
        const float p3 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s3_v, scale_log2), max_sc1));
        if (w_sub == 0) {
          if (ni == 0) {
            if (my_is_first) {
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
        } else {
          smem.p_hi_f32[q_slot][ni][lane] = make_float4(p0, p1, p2, p3);
        }
        const __nv_bfloat162 pb0 = __floats2bfloat162_rn(p0, p1);
        const __nv_bfloat162 pb1 = __floats2bfloat162_rn(p2, p3);
        const int kp = ni >> 1;
        const int hi = (ni & 1) << 1;
        s_frag[kp][hi + 0] = __uint_as_float(*reinterpret_cast<const uint32_t*>(&pb0));
        s_frag[kp][hi + 1] = __uint_as_float(*reinterpret_cast<const uint32_t*>(&pb1));
      }
      if (w_sub == 0) {
        #pragma unroll
        for (int kp = 0; kp < 4; ++kp) {
          smem.p_lo_bf16[q_slot][kp][lane] = make_uint4(
              __float_as_uint(s_frag[kp][0]),
              __float_as_uint(s_frag[kp][1]),
              __float_as_uint(s_frag[kp][2]),
              __float_as_uint(s_frag[kp][3]));
        }
      }
      my_is_first = false;
    }

    // Barrier #3: V[0] (kv[s2]) (+ tree_V if remap), p_lo_bf16, and p_hi_f32 are ready!
    if (nb > cta_nb_min) {
      cp_async_wait<2>();
    } else {
      cp_async_wait<1>();
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // 3A. PV Half 0 (reads kv[s2] [+ tree_kv]) + interleaved w_sub=0 p_hi row_sum/pack!
    // -------------------------------------------------------------------------
    if (!nb_has_remap) {
      if (my_in_block) {
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_v_pair64_v3(v_cur, smem.kv[s2], dp_base, 0, lane);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          uint32_t p_cur[4];
          if (w_sub == 0) {
            p_cur[0] = __float_as_uint(s_frag[ks_local][0]);
            p_cur[1] = __float_as_uint(s_frag[ks_local][1]);
            p_cur[2] = __float_as_uint(s_frag[ks_local][2]);
            p_cur[3] = __float_as_uint(s_frag[ks_local][3]);
          } else {
            const uint4 pl = smem.p_lo_bf16[q_slot][ks_local][lane];
            p_cur[0] = pl.x; p_cur[1] = pl.y; p_cur[2] = pl.z; p_cur[3] = pl.w;
          }
          #pragma unroll
          for (int dp = 0; dp < 4; ++dp) {
            const int n_dp = (dp + 1) & 3;
            const int n_ks = ks_local + (dp == 3 ? 1 : 0);
            if (n_ks < 4) ldsm4_v_pair64_v3(v_nxt, smem.kv[s2], dp_base + n_dp, n_ks, lane);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            if (w_sub == 0 && (dp == 0 || dp == 2)) {
              const int ni = (ks_local << 1) + (dp >> 1);
              const float4 ph = smem.p_hi_f32[q_slot][ni][lane];
              row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph.x), ph.y);
              row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph.z), ph.w);
              const __nv_bfloat162 pb0 = __floats2bfloat162_rn(ph.x, ph.y);
              const __nv_bfloat162 pb1 = __floats2bfloat162_rn(ph.z, ph.w);
              s_frag[ks_local][dp + 0] = __uint_as_float(*reinterpret_cast<const uint32_t*>(&pb0));
              s_frag[ks_local][dp + 1] = __uint_as_float(*reinterpret_cast<const uint32_t*>(&pb1));
            }
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
      }
    } else {
      if (my_in_block) {
        const int r_in_16 = lane & 15;
        const int dt_sub = lane >> 4;
        uint32_t v_row_base[4];
        int v_swz[4];
        #pragma unroll
        for (int kl = 0; kl < 4; ++kl) {
          const int r = (kl << 4) + r_in_16;
          const int tok = block_tok_base + r;
          if (tok >= my_k_start && tok < my_seqlen_k) {
            if (my_remap_base >= 0 && tok >= my_remap_base) {
              const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
              v_row_base[kl] = smem_u32(&smem.tree_kv[rr][0]);
              v_swz[kl] = rr & 7;
            } else {
              v_row_base[kl] = smem_u32(&smem.kv[s2][r][0]);
              v_swz[kl] = r & 7;
            }
          } else {
            v_row_base[kl] = smem_u32(&smem.zero_row[0]);
            v_swz[kl] = 0;
          }
        }
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_trans_raw_addr(v_cur, v_row_base[0] + (((dp_base << 1) + dt_sub) ^ v_swz[0]) * 16u);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          uint32_t p_cur[4];
          if (w_sub == 0) {
            p_cur[0] = __float_as_uint(s_frag[ks_local][0]);
            p_cur[1] = __float_as_uint(s_frag[ks_local][1]);
            p_cur[2] = __float_as_uint(s_frag[ks_local][2]);
            p_cur[3] = __float_as_uint(s_frag[ks_local][3]);
          } else {
            const uint4 pl = smem.p_lo_bf16[q_slot][ks_local][lane];
            p_cur[0] = pl.x; p_cur[1] = pl.y; p_cur[2] = pl.z; p_cur[3] = pl.w;
          }
          #pragma unroll
          for (int dp = 0; dp < 4; ++dp) {
            const int n_dp = (dp + 1) & 3;
            const int n_ks = ks_local + (dp == 3 ? 1 : 0);
            if (n_ks < 4) {
              const int n_dt = ((dp_base + n_dp) << 1) + dt_sub;
              ldsm4_trans_raw_addr(v_nxt, v_row_base[n_ks] + ((n_dt ^ v_swz[n_ks]) << 4));
            }
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            if (w_sub == 0 && (dp == 0 || dp == 2)) {
              const int ni = (ks_local << 1) + (dp >> 1);
              const float4 ph = smem.p_hi_f32[q_slot][ni][lane];
              row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph.x), ph.y);
              row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph.z), ph.w);
              const __nv_bfloat162 pb0 = __floats2bfloat162_rn(ph.x, ph.y);
              const __nv_bfloat162 pb1 = __floats2bfloat162_rn(ph.z, ph.w);
              s_frag[ks_local][dp + 0] = __uint_as_float(*reinterpret_cast<const uint32_t*>(&pb0));
              s_frag[ks_local][dp + 1] = __uint_as_float(*reinterpret_cast<const uint32_t*>(&pb1));
            }
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
      }
    }

    // Barrier #4: V[1] (kv[s0]) ready AND kv[s2] (V[0]) is now free for next K[1]!
    if (nb > cta_nb_min) {
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    // Issue next K[1] into free kv[s2] BEFORE PV Half 1 (hidden behind PV Half 1's 32 MMAs!).
    if (nb > cta_nb_min) {
      issue_kv64_v4<false, kKvL2>(
          smem.kv[s2], kv_ptr, block_table, ((nb - 1) << 7) + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
      cp_async_commit();
    }

    // -------------------------------------------------------------------------
    // 3B. PV Half 1 (reads kv[s0] [+ tree_kv]); BOTH warps hold p_hi in s_frag[0..3]!
    // -------------------------------------------------------------------------
    if (!nb_has_remap) {
      if (my_in_block) {
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_v_pair64_v3(v_cur, smem.kv[s0], dp_base, 0, lane);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          const uint32_t p_cur[4] = {
              __float_as_uint(s_frag[ks_local][0]),
              __float_as_uint(s_frag[ks_local][1]),
              __float_as_uint(s_frag[ks_local][2]),
              __float_as_uint(s_frag[ks_local][3]),
          };
          #pragma unroll
          for (int dp = 0; dp < 4; ++dp) {
            const int n_dp = (dp + 1) & 3;
            const int n_ks = ks_local + (dp == 3 ? 1 : 0);
            if (n_ks < 4) ldsm4_v_pair64_v3(v_nxt, smem.kv[s0], dp_base + n_dp, n_ks, lane);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
      }
    } else {
      if (my_in_block) {
        const int r_in_16 = lane & 15;
        const int dt_sub = lane >> 4;
        uint32_t v_row_base[4];
        int v_swz[4];
        #pragma unroll
        for (int kl = 0; kl < 4; ++kl) {
          const int r = (kl << 4) + r_in_16;
          const int tok = block_tok_base + 64 + r;
          if (tok >= my_k_start && tok < my_seqlen_k) {
            if (my_remap_base >= 0 && tok >= my_remap_base) {
              const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
              v_row_base[kl] = smem_u32(&smem.tree_kv[rr][0]);
              v_swz[kl] = rr & 7;
            } else {
              v_row_base[kl] = smem_u32(&smem.kv[s0][r][0]);
              v_swz[kl] = r & 7;
            }
          } else {
            v_row_base[kl] = smem_u32(&smem.zero_row[0]);
            v_swz[kl] = 0;
          }
        }
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_trans_raw_addr(v_cur, v_row_base[0] + (((dp_base << 1) + dt_sub) ^ v_swz[0]) * 16u);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          const uint32_t p_cur[4] = {
              __float_as_uint(s_frag[ks_local][0]),
              __float_as_uint(s_frag[ks_local][1]),
              __float_as_uint(s_frag[ks_local][2]),
              __float_as_uint(s_frag[ks_local][3]),
          };
          #pragma unroll
          for (int dp = 0; dp < 4; ++dp) {
            const int n_dp = (dp + 1) & 3;
            const int n_ks = ks_local + (dp == 3 ? 1 : 0);
            if (n_ks < 4) {
              const int n_dt = ((dp_base + n_dp) << 1) + dt_sub;
              ldsm4_trans_raw_addr(v_nxt, v_row_base[n_ks] + ((n_dt ^ v_swz[n_ks]) << 4));
            }
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
      }
    }

    // Rotate 3-slot ring with zero end-of-loop barrier:
    //   next s0 = s1 (next K[0]), next s1 = s2 (next K[1]), next s2 = s0 (old V[1], freed at next Barrier #1).
    const int ns0 = s1;
    const int ns1 = s2;
    const int ns2 = s0;
    s0 = ns0;
    s1 = ns1;
    s2 = ns2;
  }

  // Final normalization: w_sub=0 reduces row_sum & broadcasts inv0/inv1, both warps store 8 D-tiles in parallel!
  if (my_active && w_sub == 0) {
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 2));
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 1));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 2));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 1));
    const bool empty0 = (row_sum0 == 0.0f || row_sum0 != row_sum0 || row_max0 == -INFINITY);
    const bool empty1 = (row_sum1 == 0.0f || row_sum1 != row_sum1 || row_max1 == -INFINITY);
    const float inv0 = empty0 ? 1.0f : fast_rcpf(row_sum0);
    const float inv1 = empty1 ? 1.0f : fast_rcpf(row_sum1);
    if (tid_in_grp == 0) {
      smem.inv_exch[q_slot][group_id] = inv0;
      smem.inv_exch[q_slot][group_id + 8] = inv1;
    }
  }
  __syncthreads();

  if (my_active) {
    const float inv0 = smem.inv_exch[q_slot][group_id];
    const float inv1 = smem.inv_exch[q_slot][group_id + 8];
    const int d_off = w_sub << 6;  // 0 or 64
    __nv_bfloat16* const out_row = out_ptr + static_cast<int64_t>(q0 + my_q_idx) * out_tok_stride;
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
      const int d_col = d_off + (dt << 3) + (tid_in_grp << 1);
      const __nv_bfloat162 ob0 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[dt][0], inv0), mul_ftz_f32(o_frag[dt][1], inv0));
      const __nv_bfloat162 ob1 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[dt][2], inv1), mul_ftz_f32(o_frag[dt][3], inv1));
      *reinterpret_cast<uint32_t*>(
          out_row + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob0);
      *reinterpret_cast<uint32_t*>(
          out_row + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob1);
    }
  }
}

// ============================================================================
// Wave-64 V4.1 Quad-Warp-Per-Query Shared-KV FA2 Verify Kernel (2Q x 4 warps/Q)
//
// Strictly 0-ULP bit-identical to FlashAttention-2 kBlockN=128 & verify_v3/v4:
//   - 256 threads (8 warps = 2 Q tokens x 4 cooperating warps/Q):
//       M=32 -> 32 CTAs (256 warps; 2x SM parallelism on critical-path BF16/NVFP4 layers)
//       M=16 -> 16 CTAs (128 warps; exact 48-SM single-wave co-residency with 32-CTA FP8 gate!)
//   - 4-way N-split for QK^T (w_sub=0..3 owns 32 K-tokens each -> 16 MMAs + 16 MUFU.EX2/warp =
//     4x faster compute & softmax than v3, 2x faster than v4!)
//   - 4-way D-split for PV (w_sub=0..3 owns 32 output dims each -> o_frag[4][4] + s_frag[4][4] =
//     only 64 core regs/thread -> ~115-132 total regs, 0 spills!)
//   - Total SMEM = 62.4 KiB (<= 64 KiB -> 64 KiB L1 cache carveout on sm_121a!).
//   - Exact FA2 descending block-128 order, exact max_ftz_f32, exact ni=0..15 add_ftz_f32 row_sum.
// ============================================================================
struct __align__(128) VerifyV4_2Q4WStorage {
  uint4 kv[2][64][16];         // 32,768 B (aliases q_tiles[2][16][16] during Q preload)
  uint4 tree_kv[32][16];       //  8,192 B (up to 32 tree K/V rows for remapped tail block)
  uint4 zero_row[16];          //    256 B (constant +0.0 BF16 row for out-of-range lanes)
  float4 p_hi_f32[2][12][32];  // 12,288 B (w_sub=1..3 -> w_sub=0 exact ni=4..15 row_sum)
  uint4 p_bf16[2][8][32];      //  8,192 B (all 4 warps publish/read packed BF16x2 P[0..7])
  float max_exch[2][4][16];    //    512 B (4-warp QK row max exchange)
  float inv_exch[2][16];       //    128 B (final inv_row_sum broadcast to w_sub=1..3)
  int32_t nb_max_q[2];         //      8 B
  int32_t nb_min_q[2];         //      8 B
  int32_t k_start_q[2];        //      8 B
  int32_t seqlen_q[2];         //      8 B
  int32_t remap_base_q[2];     //      8 B
  uint8_t active_q[2];         //      2 B
};                             // 62,464 B (<= 64 KiB SMEM carveout on sm_121a!)
static_assert(sizeof(VerifyV4_2Q4WStorage) <= 65536, "VerifyV4_2Q4WStorage must fit 64 KiB carveout");

template <bool kPdl = false>
__global__ __launch_bounds__(256, 1) void gqa16_verify_v4_2q4w_kernel(
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
    int64_t stride_kv_head,
    const int32_t* __restrict__ cu_seqlens_q_ptr,
    int64_t q_tok_stride,
    int64_t out_tok_stride,
    const int32_t* __restrict__ leaf_flag_ptr) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  VerifyV4_2Q4WStorage& smem = *reinterpret_cast<VerifyV4_2Q4WStorage*>(smem_raw);

  const int kv_head    = blockIdx.y;
  const int tid        = threadIdx.x;
  const int warp_id    = tid >> 5;       // 0..7
  const int q_slot     = warp_id >> 2;   // 0..1 (2 Q tokens / CTA)
  const int w_sub      = warp_id & 3;    // 0..3 (4 cooperating warps / Q token)
  const int lane       = tid & 31;
  const int group_id   = lane >> 2;      // 0..7
  const int tid_in_grp = lane & 3;       // 0..3

  const int q0 = cu_seqlens_q_ptr[0];
  const int actual_m = cu_seqlens_q_ptr[1] - q0;
  const int q_base = blockIdx.z * 2;
  if (q_base >= actual_m) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  const int raw_seqlen_k = seqused_k_ptr[0];
  const int lf0 = (leaf_flag_ptr != nullptr) ? leaf_flag_ptr[0] : 0;
  const int my_q_idx = q_base + q_slot;

  bool my_active = false;
  bool my_empty = false;
  int my_seqlen_k = 0;
  int my_k_start = 0;
  int my_num_nb = 0;
  int my_nb_min = 0;
  int my_remap_base = -1;
  uint32_t my_remap_packed = 0u;

  if (my_q_idx < actual_m) {
    int sk = raw_seqlen_k;
    int lrb = -1;
    uint32_t lrp = 0u;
    if (leaf_flag_ptr != nullptr && lf0 < 0 && actual_m <= 32 && sk >= actual_m) {
      const int base = 1 + my_q_idx * 3;
      const int seq_sub = leaf_flag_ptr[base];
      const int base_sub = leaf_flag_ptr[base + 1];
      if (base_sub >= 0) {
        lrb = sk - base_sub;
        lrp = static_cast<uint32_t>(leaf_flag_ptr[base + 2]);
      }
      sk -= seq_sub;
    } else if (my_q_idx == 7 && actual_m == 8 && leaf_flag_ptr != nullptr) {
      if ((lf0 == 1 || lf0 == 2) && sk >= 8) {
        const int lps = sk - 8;
        lrb = lps + lf0;
        lrp = 7u;
        sk = lps + lf0 + 1;
      } else {
        sk -= (actual_m - 1 - my_q_idx);
      }
    } else if (my_q_idx == 6 && actual_m == 7 && leaf_flag_ptr != nullptr && lf0 > 0 && sk >= 7) {
      const int lps = sk - 7;
      lrb = lps + 1;
      lrp = 6u;
      sk -= 5;
    } else {
      sk -= (actual_m - 1 - my_q_idx);
    }
    if (sk < 0) sk = 0;

    const int ks = (window_left >= 0) ? ((sk - 1 - window_left > 0) ? (sk - 1 - window_left) : 0) : 0;
    const int nnb = (sk > 0) ? ((sk + 127) >> 7) : 0;
    const int gnb = (window_left >= 0 && sk > 0) ? (ks >> 7) : 0;
    if (gnb >= nnb) {
      my_empty = true;
    } else {
      my_active = true;
      my_seqlen_k = sk;
      my_k_start = ks;
      my_num_nb = nnb;
      my_nb_min = gnb;
      my_remap_base = lrb;
      my_remap_packed = lrp;
    }
  }

  if (w_sub == 0 && lane == 0) {
    smem.active_q[q_slot] = my_active ? 1u : 0u;
    smem.nb_max_q[q_slot] = my_active ? my_num_nb : 0;
    smem.nb_min_q[q_slot] = my_active ? my_nb_min : 0x7fffffff;
    smem.k_start_q[q_slot] = my_active ? my_k_start : 0x7fffffff;
    smem.seqlen_q[q_slot] = my_active ? my_seqlen_k : 0;
    smem.remap_base_q[q_slot] = (my_active && my_remap_base >= 0) ? my_remap_base : 0x7fffffff;
  }
  if (tid < 16) {
    smem.zero_row[tid] = make_uint4(0u, 0u, 0u, 0u);
  }
  __syncthreads();

  bool cta_any_active = false;
  int cta_nb_max = 0;
  int cta_nb_min = 0x7fffffff;
  int cta_k_start = 0x7fffffff;
  int cta_seqlen_k = 0;
  int cta_remap_min_base = 0x7fffffff;
  #pragma unroll
  for (int qs = 0; qs < 2; ++qs) {
    if (smem.active_q[qs]) {
      cta_any_active = true;
      if (smem.nb_max_q[qs] > cta_nb_max) cta_nb_max = smem.nb_max_q[qs];
      if (smem.nb_min_q[qs] < cta_nb_min) cta_nb_min = smem.nb_min_q[qs];
      if (smem.k_start_q[qs] < cta_k_start) cta_k_start = smem.k_start_q[qs];
      if (smem.seqlen_q[qs] > cta_seqlen_k) cta_seqlen_k = smem.seqlen_q[qs];
      if (smem.remap_base_q[qs] < cta_remap_min_base) cta_remap_min_base = smem.remap_base_q[qs];
    }
  }
  const bool cta_has_remap = (cta_remap_min_base < 0x7fffffff);
  const int cta_p_start = raw_seqlen_k - actual_m;

  // SM121 PDL consumer barrier: wait after independent metadata/remap setup and before Q/KV/out access.
  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  if (my_empty && w_sub == 0) {
    __nv_bfloat16* const dst_base = out_ptr + static_cast<int64_t>(q0 + my_q_idx) * out_tok_stride +
                                    static_cast<int64_t>(kv_head * 16) * out_head_stride;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = (u << 5) + lane;
      const int m = idx >> 4;
      const int c = idx & 15;
      *reinterpret_cast<uint4*>(dst_base + static_cast<int64_t>(m) * out_head_stride + (c << 3)) =
          make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (!cta_any_active) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  // Load each Q token ONCE via w_sub==0 into smem.kv (aliased as q_tiles[2][16][16]), then all 4 warps ldsm4.
  uint32_t q_frag[8][4];
  {
    uint4 (*q_tiles)[16][16] = reinterpret_cast<uint4 (*)[16][16]>(smem.kv);
    if (my_active && w_sub == 0) {
      const __nv_bfloat16* q_tok_base = q_ptr + static_cast<int64_t>(q0 + my_q_idx) * q_tok_stride +
                                        static_cast<int64_t>(kv_head * 16) * q_head_stride;
      #pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int idx = (u << 5) + lane;
        const int m = idx >> 4;
        const int c = idx & 15;
        q_tiles[q_slot][m][c ^ (m & 7)] =
            *reinterpret_cast<const uint4*>(q_tok_base + static_cast<int64_t>(m) * q_head_stride + (c << 3));
      }
    }
    __syncthreads();
    if (my_active) {
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        ldsm4_swizzled(q_frag[ks], q_tiles[q_slot], ks, lane);
      }
    }
  }
  __syncthreads();

  // Each warp owns 1/4 of head_dim=128: w_sub -> D = w_sub*32 .. w_sub*32+31 (dp_base = w_sub*2, 2 dp pairs = 4 dt tiles).
  float o_frag[4][4];
  #pragma unroll
  for (int dt = 0; dt < 4; ++dt) {
    o_frag[dt][0] = 0.0f; o_frag[dt][1] = 0.0f; o_frag[dt][2] = 0.0f; o_frag[dt][3] = 0.0f;
  }
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;
  bool my_is_first = true;

  // Prime first block's K half 0 into smem.kv[0].
  issue_kv64_v4<false>(
      smem.kv[0], kv_ptr, block_table, (cta_nb_max - 1) << 7,
      cta_k_start, cta_seqlen_k, block_size, kv_head,
      stride_block, stride_slot, stride_kv_head, tid);
  cp_async_commit();

  const int kv_half = w_sub >> 1;      // 0 (tokens 0..63) or 1 (tokens 64..127)
  const int sp_base = (w_sub & 1) << 1; // 0 (0..31 within half) or 2 (32..63 within half)
  const int dp_base = w_sub << 1;       // 0, 2, 4, 6 (D = 0..31, 32..63, 64..95, 96..127)

  for (int nb = cta_nb_max - 1; nb >= cta_nb_min; --nb) {
    const int block_tok_base = nb << 7;
    const bool nb_has_remap = cta_has_remap && (block_tok_base + 128 > cta_remap_min_base);
    const bool my_in_block = my_active && (nb < my_num_nb) && (nb >= my_nb_min);

    float s_frag[4][4];
    #pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      s_frag[ni][0] = 0.0f; s_frag[ni][1] = 0.0f; s_frag[ni][2] = 0.0f; s_frag[ni][3] = 0.0f;
    }

    // -------------------------------------------------------------------------
    // 1. QK^T: All 4 warps/Q split 128 K-tokens into 4x32 (16 MMAs/warp!).
    // -------------------------------------------------------------------------
    if (!nb_has_remap) {
      issue_kv64_v4<false>(
          smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();

      if (my_in_block) {
        const uint4 (*my_k_tile)[16] = smem.kv[kv_half];
        uint32_t b_cur[4], b_nxt[4];
        ldsm4_k_pair64_v3(b_cur, my_k_tile, sp_base, 0, lane);
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          #pragma unroll
          for (int sp = 0; sp < 2; ++sp) {
            const int n_sp = sp ^ 1;
            const int n_ks = ks + sp;
            if (n_ks < 8) ldsm4_k_pair64_v3(b_nxt, my_k_tile, sp_base + n_sp, n_ks, lane);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[0], b_cur[1], s_frag[sp * 2 + 0]);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[2], b_cur[3], s_frag[sp * 2 + 1]);
            b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1]; b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
          }
        }
      }
    } else {
      issue_kv64_v4<false>(
          smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      issue_tree_kv_v4<false>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();

      if (my_in_block) {
        const uint4 (*my_k_tile)[16] = smem.kv[kv_half];
        const int half_base = block_tok_base + (kv_half << 6);
        const int r_local = lane & 7;
        const int sub_k = (lane >> 3) & 1;
        uint32_t k_row_base[2];
        int k_swz[2];
        #pragma unroll
        for (int sp = 0; sp < 2; ++sp) {
          const int sub = ((sp_base + sp) << 1) + (lane >> 4);
          const int r = (sub << 3) + r_local;
          const int tok = half_base + r;
          if (tok >= my_k_start && tok < my_seqlen_k) {
            if (my_remap_base >= 0 && tok >= my_remap_base) {
              const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
              k_row_base[sp] = smem_u32(&smem.tree_kv[rr][0]);
              k_swz[sp] = rr & 7;
            } else {
              k_row_base[sp] = smem_u32(&my_k_tile[r][0]);
              k_swz[sp] = r_local;
            }
          } else {
            k_row_base[sp] = smem_u32(&smem.zero_row[0]);
            k_swz[sp] = 0;
          }
        }
        uint32_t b_cur[4], b_nxt[4];
        ldsm4_raw_addr(b_cur, k_row_base[0] + ((sub_k ^ k_swz[0]) << 4));
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          #pragma unroll
          for (int sp = 0; sp < 2; ++sp) {
            const int n_sp = sp ^ 1;
            const int n_ks = ks + sp;
            if (n_ks < 8) {
              ldsm4_raw_addr(b_nxt, k_row_base[n_sp] + ((((n_ks << 1) + sub_k) ^ k_swz[n_sp]) << 4));
            }
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[0], b_cur[1], s_frag[sp * 2 + 0]);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[2], b_cur[3], s_frag[sp * 2 + 1]);
            b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1]; b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
          }
        }
      }
    }

    // Local QK masking + row max over 32 K-tokens per warp.
    if (my_in_block) {
      const int chunk_base = block_tok_base + (w_sub << 5);
      const bool need_mask = (chunk_base < my_k_start) || (chunk_base + 32 > my_seqlen_k);
      float m0 = -INFINITY, m1 = -INFINITY;
      #pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        if (need_mask) {
          const int col0 = chunk_base + (ni << 3) + (tid_in_grp << 1);
          const int col1 = col0 + 1;
          if (col0 < my_k_start || col0 >= my_seqlen_k) { s_frag[ni][0] = -INFINITY; s_frag[ni][2] = -INFINITY; }
          if (col1 < my_k_start || col1 >= my_seqlen_k) { s_frag[ni][1] = -INFINITY; s_frag[ni][3] = -INFINITY; }
        }
        m0 = max_ftz_f32(m0, max_ftz_f32(s_frag[ni][0], s_frag[ni][1]));
        m1 = max_ftz_f32(m1, max_ftz_f32(s_frag[ni][2], s_frag[ni][3]));
      }
      m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
      m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
      m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
      m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
      if (tid_in_grp == 0) {
        smem.max_exch[q_slot][w_sub][group_id] = m0;
        smem.max_exch[q_slot][w_sub][group_id + 8] = m1;
      }
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // 2. Issue BOTH V[0] & V[1] (+ tree_V if remap) behind 4-way parallel Softmax!
    // -------------------------------------------------------------------------
    if (!nb_has_remap) {
      issue_kv64_v4<true>(
          smem.kv[0], kv_ptr, block_table, block_tok_base,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
      issue_kv64_v4<true>(
          smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
    } else {
      issue_kv64_v4<true>(
          smem.kv[0], kv_ptr, block_table, block_tok_base,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      issue_kv64_v4<true>(
          smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid);
      issue_tree_kv_v4<true>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
    }

    if constexpr (kPdl) {
      if (nb == cta_nb_min) {
        asm volatile("griddepcontrol.launch_dependents;");
      }
    }

    // 4-way parallel softmax (only 16 MUFU.EX2 per warp!).
    if (my_in_block) {
      const float m0 = max_ftz_f32(
          max_ftz_f32(smem.max_exch[q_slot][0][group_id], smem.max_exch[q_slot][1][group_id]),
          max_ftz_f32(smem.max_exch[q_slot][2][group_id], smem.max_exch[q_slot][3][group_id]));
      const float m1 = max_ftz_f32(
          max_ftz_f32(smem.max_exch[q_slot][0][group_id + 8], smem.max_exch[q_slot][1][group_id + 8]),
          max_ftz_f32(smem.max_exch[q_slot][2][group_id + 8], smem.max_exch[q_slot][3][group_id + 8]));

      float sc0 = 1.0f, sc1 = 1.0f;
      if (my_is_first) {
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
        for (int dt = 0; dt < 4; ++dt) {
          o_frag[dt][0] = mul_ftz_f32(o_frag[dt][0], sc0);
          o_frag[dt][1] = mul_ftz_f32(o_frag[dt][1], sc0);
          o_frag[dt][2] = mul_ftz_f32(o_frag[dt][2], sc1);
          o_frag[dt][3] = mul_ftz_f32(o_frag[dt][3], sc1);
        }
      }

      const float max_sc0 = (row_max0 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max0);
      const float max_sc1 = (row_max1 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max1);
      uint32_t p_pack[2][4];
      #pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const float s0 = s_frag[ni][0];
        const float s1 = s_frag[ni][1];
        const float s2 = s_frag[ni][2];
        const float s3 = s_frag[ni][3];
        const float p0 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s0, scale_log2), max_sc0));
        const float p1 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s1, scale_log2), max_sc0));
        const float p2 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s2, scale_log2), max_sc1));
        const float p3 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s3, scale_log2), max_sc1));
        if (w_sub == 0) {
          if (ni == 0) {
            if (my_is_first) {
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
        } else {
          smem.p_hi_f32[q_slot][((w_sub - 1) << 2) + ni][lane] = make_float4(p0, p1, p2, p3);
        }
        const __nv_bfloat162 pb0 = __floats2bfloat162_rn(p0, p1);
        const __nv_bfloat162 pb1 = __floats2bfloat162_rn(p2, p3);
        const int kp = ni >> 1;
        const int hi = (ni & 1) << 1;
        p_pack[kp][hi + 0] = *reinterpret_cast<const uint32_t*>(&pb0);
        p_pack[kp][hi + 1] = *reinterpret_cast<const uint32_t*>(&pb1);
      }
      #pragma unroll
      for (int kp = 0; kp < 2; ++kp) {
        smem.p_bf16[q_slot][(w_sub << 1) + kp][lane] =
            make_uint4(p_pack[kp][0], p_pack[kp][1], p_pack[kp][2], p_pack[kp][3]);
      }
      my_is_first = false;
    }

    // -------------------------------------------------------------------------
    // 3. PV: All 4 warps/Q execute 16 MMAs per V half in parallel (4-way D-split)!
    // -------------------------------------------------------------------------
    if (!nb_has_remap) {
      cp_async_wait<1>();
      __syncthreads();
      if (my_in_block) {
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_v_pair64_v3(v_cur, smem.kv[0], dp_base, 0, lane);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          const uint4 pl = smem.p_bf16[q_slot][ks_local][lane];
          const uint32_t p_cur[4] = {pl.x, pl.y, pl.z, pl.w};
          #pragma unroll
          for (int dp = 0; dp < 2; ++dp) {
            const int n_dp = dp ^ 1;
            const int n_ks = ks_local + dp;
            if (n_ks < 4) ldsm4_v_pair64_v3(v_nxt, smem.kv[0], dp_base + n_dp, n_ks, lane);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
        // w_sub=0 finishes exact ni=4..15 row_sum while V[1] arrives!
        if (w_sub == 0) {
          #pragma unroll
          for (int ni = 0; ni < 12; ++ni) {
            const float4 ph = smem.p_hi_f32[q_slot][ni][lane];
            row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph.x), ph.y);
            row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph.z), ph.w);
          }
        }
      }
      __syncthreads();

      if (nb > cta_nb_min) {
        issue_kv64_v4<false>(
            smem.kv[0], kv_ptr, block_table, (nb - 1) << 7,
            cta_k_start, cta_seqlen_k, block_size, kv_head,
            stride_block, stride_slot, stride_kv_head, tid);
        cp_async_commit();
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
      }
      __syncthreads();

      if (my_in_block) {
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_v_pair64_v3(v_cur, smem.kv[1], dp_base, 0, lane);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          const uint4 ph = smem.p_bf16[q_slot][4 + ks_local][lane];
          const uint32_t p_cur[4] = {ph.x, ph.y, ph.z, ph.w};
          #pragma unroll
          for (int dp = 0; dp < 2; ++dp) {
            const int n_dp = dp ^ 1;
            const int n_ks = ks_local + dp;
            if (n_ks < 4) ldsm4_v_pair64_v3(v_nxt, smem.kv[1], dp_base + n_dp, n_ks, lane);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
      }
      __syncthreads();
    } else {
      cp_async_wait<0>();
      __syncthreads();
      if (my_in_block) {
        const int r_in_16 = lane & 15;
        const int dt_sub = lane >> 4;
        // PV Half 0 (tokens 0..63) with hoisted per-lane SMEM row addresses + double-buffered ldmatrix!
        {
          uint32_t v_row_base[4];
          int v_swz[4];
          #pragma unroll
          for (int kl = 0; kl < 4; ++kl) {
            const int r = (kl << 4) + r_in_16;
            const int tok = block_tok_base + r;
            if (tok >= my_k_start && tok < my_seqlen_k) {
              if (my_remap_base >= 0 && tok >= my_remap_base) {
                const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
                v_row_base[kl] = smem_u32(&smem.tree_kv[rr][0]);
                v_swz[kl] = rr & 7;
              } else {
                v_row_base[kl] = smem_u32(&smem.kv[0][r][0]);
                v_swz[kl] = r & 7;
              }
            } else {
              v_row_base[kl] = smem_u32(&smem.zero_row[0]);
              v_swz[kl] = 0;
            }
          }
          uint32_t v_cur[4], v_nxt[4];
          ldsm4_trans_raw_addr(v_cur, v_row_base[0] + (((dp_base << 1) + dt_sub) ^ v_swz[0]) * 16u);
          #pragma unroll
          for (int ks_local = 0; ks_local < 4; ++ks_local) {
            const uint4 pl = smem.p_bf16[q_slot][ks_local][lane];
            const uint32_t p_cur[4] = {pl.x, pl.y, pl.z, pl.w};
            #pragma unroll
            for (int dp = 0; dp < 2; ++dp) {
              const int n_dp = dp ^ 1;
              const int n_ks = ks_local + dp;
              if (n_ks < 4) {
                const int n_dt = ((dp_base + n_dp) << 1) + dt_sub;
                ldsm4_trans_raw_addr(v_nxt, v_row_base[n_ks] + ((n_dt ^ v_swz[n_ks]) << 4));
              }
              mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
              mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
              v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
            }
          }
        }
        if (w_sub == 0) {
          #pragma unroll
          for (int ni = 0; ni < 12; ++ni) {
            const float4 ph = smem.p_hi_f32[q_slot][ni][lane];
            row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph.x), ph.y);
            row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph.z), ph.w);
          }
        }
        // PV Half 1 (tokens 64..127) — V[1] & tree_kv already in SMEM, zero extra barrier!
        {
          uint32_t v_row_base[4];
          int v_swz[4];
          #pragma unroll
          for (int kl = 0; kl < 4; ++kl) {
            const int r = (kl << 4) + r_in_16;
            const int tok = block_tok_base + 64 + r;
            if (tok >= my_k_start && tok < my_seqlen_k) {
              if (my_remap_base >= 0 && tok >= my_remap_base) {
                const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
                v_row_base[kl] = smem_u32(&smem.tree_kv[rr][0]);
                v_swz[kl] = rr & 7;
              } else {
                v_row_base[kl] = smem_u32(&smem.kv[1][r][0]);
                v_swz[kl] = r & 7;
              }
            } else {
              v_row_base[kl] = smem_u32(&smem.zero_row[0]);
              v_swz[kl] = 0;
            }
          }
          uint32_t v_cur[4], v_nxt[4];
          ldsm4_trans_raw_addr(v_cur, v_row_base[0] + (((dp_base << 1) + dt_sub) ^ v_swz[0]) * 16u);
          #pragma unroll
          for (int ks_local = 0; ks_local < 4; ++ks_local) {
            const uint4 ph = smem.p_bf16[q_slot][4 + ks_local][lane];
            const uint32_t p_cur[4] = {ph.x, ph.y, ph.z, ph.w};
            #pragma unroll
            for (int dp = 0; dp < 2; ++dp) {
              const int n_dp = dp ^ 1;
              const int n_ks = ks_local + dp;
              if (n_ks < 4) {
                const int n_dt = ((dp_base + n_dp) << 1) + dt_sub;
                ldsm4_trans_raw_addr(v_nxt, v_row_base[n_ks] + ((n_dt ^ v_swz[n_ks]) << 4));
              }
              mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
              mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
              v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
            }
          }
        }
      }
      __syncthreads();
      if (nb > cta_nb_min) {
        issue_kv64_v4<false>(
            smem.kv[0], kv_ptr, block_table, (nb - 1) << 7,
            cta_k_start, cta_seqlen_k, block_size, kv_head,
            stride_block, stride_slot, stride_kv_head, tid);
        cp_async_commit();
      }
    }
  }

  // Final normalization: w_sub=0 reduces row_sum & broadcasts inv0/inv1, all 4 warps store 4 D-tiles in parallel!
  if (my_active && w_sub == 0) {
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 2));
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 1));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 2));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 1));
    const bool empty0 = (row_sum0 == 0.0f || row_sum0 != row_sum0 || row_max0 == -INFINITY);
    const bool empty1 = (row_sum1 == 0.0f || row_sum1 != row_sum1 || row_max1 == -INFINITY);
    const float inv0 = empty0 ? 1.0f : fast_rcpf(row_sum0);
    const float inv1 = empty1 ? 1.0f : fast_rcpf(row_sum1);
    if (tid_in_grp == 0) {
      smem.inv_exch[q_slot][group_id] = inv0;
      smem.inv_exch[q_slot][group_id + 8] = inv1;
    }
  }
  __syncthreads();

  if (my_active) {
    const float inv0 = smem.inv_exch[q_slot][group_id];
    const float inv1 = smem.inv_exch[q_slot][group_id + 8];
    const int d_off = w_sub << 5;  // 0, 32, 64, 96
    __nv_bfloat16* const out_row = out_ptr + static_cast<int64_t>(q0 + my_q_idx) * out_tok_stride;
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      const int d_col = d_off + (dt << 3) + (tid_in_grp << 1);
      const __nv_bfloat162 ob0 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[dt][0], inv0), mul_ftz_f32(o_frag[dt][1], inv0));
      const __nv_bfloat162 ob1 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[dt][2], inv1), mul_ftz_f32(o_frag[dt][3], inv1));
      *reinterpret_cast<uint32_t*>(
          out_row + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob0);
      *reinterpret_cast<uint32_t*>(
          out_row + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob1);
    }
  }
}

// ============================================================================
// Wave-86d V4.1 Zero-Stall 3-Slot KV Ring Quad-Warp-Per-Query FA2 Verify Kernel (2Q x 4w/Q)
// Same 3-slot zero-stall KV ring + interleaved w_sub=0 p_hi_f32 row_sum (ni=0..11) for 32-CTA
// full-attention / V4=2 mode (78,848 B <= 99,328 B dynamic SMEM on sm_121a).
// ============================================================================
struct __align__(128) VerifyV4_2Q4W_Ring3Storage {
  uint4 kv[3][64][16];         // 49,152 B (3x16 KiB ring; kv[2] aliases q_tiles[2][16][16] in prologue)
  uint4 tree_kv[32][16];       //  8,192 B (up to 32 tree K/V rows for remapped tail block)
  uint4 zero_row[16];          //    256 B (constant +0.0 BF16 row for out-of-range lanes)
  float4 p_hi_f32[2][12][32];  // 12,288 B (w_sub=1..3 -> w_sub=0 exact ni=4..15 row_sum)
  uint4 p_bf16[2][8][32];      //  8,192 B (all 4 warps publish/read packed BF16x2 P[0..7])
  float max_exch[2][4][16];    //    512 B (4-warp QK row max exchange)
  float inv_exch[2][16];       //    128 B (final inv_row_sum broadcast to w_sub=1..3)
  int32_t nb_max_q[2];         //      8 B
  int32_t nb_min_q[2];         //      8 B
  int32_t k_start_q[2];        //      8 B
  int32_t seqlen_q[2];         //      8 B
  int32_t remap_base_q[2];     //      8 B
  uint8_t active_q[2];         //      2 B
};                             // 78,848 B (<= 99,328 B dynamic SMEM on sm_121a)
static_assert(sizeof(VerifyV4_2Q4W_Ring3Storage) <= 99328, "VerifyV4_2Q4W_Ring3Storage must fit SM121 dynamic SMEM");

template <bool kPdl = false, int kKvL2 = 0>
__global__ __launch_bounds__(256, 1) void gqa16_verify_v4_2q4w_ring3_kernel(
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
    int64_t stride_kv_head,
    const int32_t* __restrict__ cu_seqlens_q_ptr,
    int64_t q_tok_stride,
    int64_t out_tok_stride,
    const int32_t* __restrict__ leaf_flag_ptr) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  VerifyV4_2Q4W_Ring3Storage& smem = *reinterpret_cast<VerifyV4_2Q4W_Ring3Storage*>(smem_raw);

  const int kv_head    = blockIdx.y;
  const int tid        = threadIdx.x;
  const int warp_id    = tid >> 5;       // 0..7
  const int q_slot     = warp_id >> 2;   // 0..1 (2 Q tokens / CTA)
  const int w_sub      = warp_id & 3;    // 0..3 (4 cooperating warps / Q token)
  const int lane       = tid & 31;
  const int group_id   = lane >> 2;      // 0..7
  const int tid_in_grp = lane & 3;       // 0..3

  const int q0 = cu_seqlens_q_ptr[0];
  const int actual_m = cu_seqlens_q_ptr[1] - q0;
  const int q_base = blockIdx.z * 2;
  if (q_base >= actual_m) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  const int raw_seqlen_k = seqused_k_ptr[0];
  const int lf0 = (leaf_flag_ptr != nullptr) ? leaf_flag_ptr[0] : 0;
  const int my_q_idx = q_base + q_slot;

  bool my_active = false;
  bool my_empty = false;
  int my_seqlen_k = 0;
  int my_k_start = 0;
  int my_num_nb = 0;
  int my_nb_min = 0;
  int my_remap_base = -1;
  uint32_t my_remap_packed = 0u;

  if (my_q_idx < actual_m) {
    int sk = raw_seqlen_k;
    int lrb = -1;
    uint32_t lrp = 0u;
    if (leaf_flag_ptr != nullptr && lf0 < 0 && actual_m <= 32 && sk >= actual_m) {
      const int base = 1 + my_q_idx * 3;
      const int seq_sub = leaf_flag_ptr[base];
      const int base_sub = leaf_flag_ptr[base + 1];
      if (base_sub >= 0) {
        lrb = sk - base_sub;
        lrp = static_cast<uint32_t>(leaf_flag_ptr[base + 2]);
      }
      sk -= seq_sub;
    } else if (my_q_idx == 7 && actual_m == 8 && leaf_flag_ptr != nullptr) {
      if ((lf0 == 1 || lf0 == 2) && sk >= 8) {
        const int lps = sk - 8;
        lrb = lps + lf0;
        lrp = 7u;
        sk = lps + lf0 + 1;
      } else {
        sk -= (actual_m - 1 - my_q_idx);
      }
    } else if (my_q_idx == 6 && actual_m == 7 && leaf_flag_ptr != nullptr && lf0 > 0 && sk >= 7) {
      const int lps = sk - 7;
      lrb = lps + 1;
      lrp = 6u;
      sk -= 5;
    } else {
      sk -= (actual_m - 1 - my_q_idx);
    }
    if (sk < 0) sk = 0;

    const int ks = (window_left >= 0) ? ((sk - 1 - window_left > 0) ? (sk - 1 - window_left) : 0) : 0;
    const int nnb = (sk > 0) ? ((sk + 127) >> 7) : 0;
    const int gnb = (window_left >= 0 && sk > 0) ? (ks >> 7) : 0;
    if (gnb >= nnb) {
      my_empty = true;
    } else {
      my_active = true;
      my_seqlen_k = sk;
      my_k_start = ks;
      my_num_nb = nnb;
      my_nb_min = gnb;
      my_remap_base = lrb;
      my_remap_packed = lrp;
    }
  }

  if (w_sub == 0 && lane == 0) {
    smem.active_q[q_slot] = my_active ? 1u : 0u;
    smem.nb_max_q[q_slot] = my_active ? my_num_nb : 0;
    smem.nb_min_q[q_slot] = my_active ? my_nb_min : 0x7fffffff;
    smem.k_start_q[q_slot] = my_active ? my_k_start : 0x7fffffff;
    smem.seqlen_q[q_slot] = my_active ? my_seqlen_k : 0;
    smem.remap_base_q[q_slot] = (my_active && my_remap_base >= 0) ? my_remap_base : 0x7fffffff;
  }
  if (tid < 16) {
    smem.zero_row[tid] = make_uint4(0u, 0u, 0u, 0u);
  }
  __syncthreads();

  bool cta_any_active = false;
  int cta_nb_max = 0;
  int cta_nb_min = 0x7fffffff;
  int cta_k_start = 0x7fffffff;
  int cta_seqlen_k = 0;
  int cta_remap_min_base = 0x7fffffff;
  #pragma unroll
  for (int qs = 0; qs < 2; ++qs) {
    if (smem.active_q[qs]) {
      cta_any_active = true;
      if (smem.nb_max_q[qs] > cta_nb_max) cta_nb_max = smem.nb_max_q[qs];
      if (smem.nb_min_q[qs] < cta_nb_min) cta_nb_min = smem.nb_min_q[qs];
      if (smem.k_start_q[qs] < cta_k_start) cta_k_start = smem.k_start_q[qs];
      if (smem.seqlen_q[qs] > cta_seqlen_k) cta_seqlen_k = smem.seqlen_q[qs];
      if (smem.remap_base_q[qs] < cta_remap_min_base) cta_remap_min_base = smem.remap_base_q[qs];
    }
  }
  const bool cta_has_remap = (cta_remap_min_base < 0x7fffffff);
  const int cta_p_start = raw_seqlen_k - actual_m;

  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  if (my_empty && w_sub == 0) {
    __nv_bfloat16* const dst_base = out_ptr + static_cast<int64_t>(q0 + my_q_idx) * out_tok_stride +
                                    static_cast<int64_t>(kv_head * 16) * out_head_stride;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = (u << 5) + lane;
      const int m = idx >> 4;
      const int c = idx & 15;
      *reinterpret_cast<uint4*>(dst_base + static_cast<int64_t>(m) * out_head_stride + (c << 3)) =
          make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (!cta_any_active) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  // Prologue: load Q into kv[2] while priming block-0 K[0]/K[1](/tree_K) into kv[0]/kv[1](/tree_kv).
  const uint64_t kv_l2_pol = (kKvL2 == 2) ? kv_pol_evict_last() : 0ull;
  uint32_t q_frag[8][4];
  {
    uint4 (*q_tiles)[16][16] = reinterpret_cast<uint4 (*)[16][16]>(smem.kv[2]);
    if (my_active && w_sub == 0) {
      const __nv_bfloat16* q_tok_base = q_ptr + static_cast<int64_t>(q0 + my_q_idx) * q_tok_stride +
                                        static_cast<int64_t>(kv_head * 16) * q_head_stride;
      #pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int idx = (u << 5) + lane;
        const int m = idx >> 4;
        const int c = idx & 15;
        q_tiles[q_slot][m][c ^ (m & 7)] =
            *reinterpret_cast<const uint4*>(q_tok_base + static_cast<int64_t>(m) * q_head_stride + (c << 3));
      }
    }
    const int nb0_tok_base = (cta_nb_max - 1) << 7;
    issue_kv64_v4<false, kKvL2>(
        smem.kv[0], kv_ptr, block_table, nb0_tok_base,
        cta_k_start, cta_seqlen_k, block_size, kv_head,
        stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    issue_kv64_v4<false, kKvL2>(
        smem.kv[1], kv_ptr, block_table, nb0_tok_base + 64,
        cta_k_start, cta_seqlen_k, block_size, kv_head,
        stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    if (cta_has_remap && (nb0_tok_base + 128 > cta_remap_min_base)) {
      issue_tree_kv_v4<false, kKvL2>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    }
    cp_async_commit();
    __syncthreads();
    if (my_active) {
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        ldsm4_swizzled(q_frag[ks], q_tiles[q_slot], ks, lane);
      }
    }
  }

  float o_frag[4][4];
  #pragma unroll
  for (int dt = 0; dt < 4; ++dt) {
    o_frag[dt][0] = 0.0f; o_frag[dt][1] = 0.0f; o_frag[dt][2] = 0.0f; o_frag[dt][3] = 0.0f;
  }
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;
  bool my_is_first = true;

  const int kv_half = w_sub >> 1;
  const int sp_base = (w_sub & 1) << 1;
  const int dp_base = w_sub << 1;
  int s0 = 0, s1 = 1, s2 = 2;

  for (int nb = cta_nb_max - 1; nb >= cta_nb_min; --nb) {
    const int block_tok_base = nb << 7;
    const bool nb_has_remap = cta_has_remap && (block_tok_base + 128 > cta_remap_min_base);
    const bool my_in_block = my_active && (nb < my_num_nb) && (nb >= my_nb_min);

    cp_async_wait<0>();
    __syncthreads();

    if (nb_has_remap && nb < cta_nb_max - 1) {
      issue_tree_kv_v4<false, kKvL2>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();
    }

    issue_kv64_v4<true, kKvL2>(
        smem.kv[s2], kv_ptr, block_table, block_tok_base,
        cta_k_start, cta_seqlen_k, block_size, kv_head,
        stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    cp_async_commit();

    float s_frag[4][4];
    #pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      s_frag[ni][0] = 0.0f; s_frag[ni][1] = 0.0f; s_frag[ni][2] = 0.0f; s_frag[ni][3] = 0.0f;
    }

    if (!nb_has_remap) {
      if (my_in_block) {
        const uint4 (*my_k_tile)[16] = (kv_half == 0) ? smem.kv[s0] : smem.kv[s1];
        uint32_t b_cur[4], b_nxt[4];
        ldsm4_k_pair64_v3(b_cur, my_k_tile, sp_base, 0, lane);
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          #pragma unroll
          for (int sp = 0; sp < 2; ++sp) {
            const int n_sp = sp ^ 1;
            const int n_ks = ks + sp;
            if (n_ks < 8) ldsm4_k_pair64_v3(b_nxt, my_k_tile, sp_base + n_sp, n_ks, lane);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[0], b_cur[1], s_frag[sp * 2 + 0]);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[2], b_cur[3], s_frag[sp * 2 + 1]);
            b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1]; b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
          }
        }
      }
    } else {
      if (my_in_block) {
        const uint4 (*my_k_tile)[16] = (kv_half == 0) ? smem.kv[s0] : smem.kv[s1];
        const int half_base = block_tok_base + (kv_half << 6);
        const int r_local = lane & 7;
        const int sub_k = (lane >> 3) & 1;
        uint32_t k_row_base[2];
        int k_swz[2];
        #pragma unroll
        for (int sp = 0; sp < 2; ++sp) {
          const int sub = ((sp_base + sp) << 1) + (lane >> 4);
          const int r = (sub << 3) + r_local;
          const int tok = half_base + r;
          if (tok >= my_k_start && tok < my_seqlen_k) {
            if (my_remap_base >= 0 && tok >= my_remap_base) {
              const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
              k_row_base[sp] = smem_u32(&smem.tree_kv[rr][0]);
              k_swz[sp] = rr & 7;
            } else {
              k_row_base[sp] = smem_u32(&my_k_tile[r][0]);
              k_swz[sp] = r_local;
            }
          } else {
            k_row_base[sp] = smem_u32(&smem.zero_row[0]);
            k_swz[sp] = 0;
          }
        }
        uint32_t b_cur[4], b_nxt[4];
        ldsm4_raw_addr(b_cur, k_row_base[0] + ((sub_k ^ k_swz[0]) << 4));
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          #pragma unroll
          for (int sp = 0; sp < 2; ++sp) {
            const int n_sp = sp ^ 1;
            const int n_ks = ks + sp;
            if (n_ks < 8) {
              ldsm4_raw_addr(b_nxt, k_row_base[n_sp] + ((((n_ks << 1) + sub_k) ^ k_swz[n_sp]) << 4));
            }
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[0], b_cur[1], s_frag[sp * 2 + 0]);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[2], b_cur[3], s_frag[sp * 2 + 1]);
            b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1]; b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
          }
        }
      }
    }

    if (my_in_block) {
      const int chunk_base = block_tok_base + (w_sub << 5);
      const bool need_mask = (chunk_base < my_k_start) || (chunk_base + 32 > my_seqlen_k);
      float m0 = -INFINITY, m1 = -INFINITY;
      #pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        if (need_mask) {
          const int col0 = chunk_base + (ni << 3) + (tid_in_grp << 1);
          const int col1 = col0 + 1;
          if (col0 < my_k_start || col0 >= my_seqlen_k) { s_frag[ni][0] = -INFINITY; s_frag[ni][2] = -INFINITY; }
          if (col1 < my_k_start || col1 >= my_seqlen_k) { s_frag[ni][1] = -INFINITY; s_frag[ni][3] = -INFINITY; }
        }
        m0 = max_ftz_f32(m0, max_ftz_f32(s_frag[ni][0], s_frag[ni][1]));
        m1 = max_ftz_f32(m1, max_ftz_f32(s_frag[ni][2], s_frag[ni][3]));
      }
      m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
      m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
      m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
      m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
      if (tid_in_grp == 0) {
        smem.max_exch[q_slot][w_sub][group_id] = m0;
        smem.max_exch[q_slot][w_sub][group_id + 8] = m1;
      }
    }
    __syncthreads();

    if (nb_has_remap) {
      issue_tree_kv_v4<true, kKvL2>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
      cp_async_commit();
    }
    issue_kv64_v4<true, kKvL2>(
        smem.kv[s0], kv_ptr, block_table, block_tok_base + 64,
        cta_k_start, cta_seqlen_k, block_size, kv_head,
        stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    cp_async_commit();
    if (nb > cta_nb_min) {
      issue_kv64_v4<false, kKvL2>(
          smem.kv[s1], kv_ptr, block_table, (nb - 1) << 7,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
      cp_async_commit();
    }

    if constexpr (kPdl) {
      if (nb == cta_nb_min) {
        asm volatile("griddepcontrol.launch_dependents;");
      }
    }

    if (my_in_block) {
      const float m0 = max_ftz_f32(
          max_ftz_f32(smem.max_exch[q_slot][0][group_id], smem.max_exch[q_slot][1][group_id]),
          max_ftz_f32(smem.max_exch[q_slot][2][group_id], smem.max_exch[q_slot][3][group_id]));
      const float m1 = max_ftz_f32(
          max_ftz_f32(smem.max_exch[q_slot][0][group_id + 8], smem.max_exch[q_slot][1][group_id + 8]),
          max_ftz_f32(smem.max_exch[q_slot][2][group_id + 8], smem.max_exch[q_slot][3][group_id + 8]));

      float sc0 = 1.0f, sc1 = 1.0f;
      if (my_is_first) {
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
        for (int dt = 0; dt < 4; ++dt) {
          o_frag[dt][0] = mul_ftz_f32(o_frag[dt][0], sc0);
          o_frag[dt][1] = mul_ftz_f32(o_frag[dt][1], sc0);
          o_frag[dt][2] = mul_ftz_f32(o_frag[dt][2], sc1);
          o_frag[dt][3] = mul_ftz_f32(o_frag[dt][3], sc1);
        }
      }

      const float max_sc0 = (row_max0 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max0);
      const float max_sc1 = (row_max1 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max1);
      uint32_t p_pack[2][4];
      #pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const float s0_v = s_frag[ni][0];
        const float s1_v = s_frag[ni][1];
        const float s2_v = s_frag[ni][2];
        const float s3_v = s_frag[ni][3];
        const float p0 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s0_v, scale_log2), max_sc0));
        const float p1 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s1_v, scale_log2), max_sc0));
        const float p2 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s2_v, scale_log2), max_sc1));
        const float p3 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s3_v, scale_log2), max_sc1));
        if (w_sub == 0) {
          if (ni == 0) {
            if (my_is_first) {
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
        } else {
          smem.p_hi_f32[q_slot][((w_sub - 1) << 2) + ni][lane] = make_float4(p0, p1, p2, p3);
        }
        const __nv_bfloat162 pb0 = __floats2bfloat162_rn(p0, p1);
        const __nv_bfloat162 pb1 = __floats2bfloat162_rn(p2, p3);
        const int kp = ni >> 1;
        const int hi = (ni & 1) << 1;
        p_pack[kp][hi + 0] = *reinterpret_cast<const uint32_t*>(&pb0);
        p_pack[kp][hi + 1] = *reinterpret_cast<const uint32_t*>(&pb1);
      }
      #pragma unroll
      for (int kp = 0; kp < 2; ++kp) {
        smem.p_bf16[q_slot][(w_sub << 1) + kp][lane] =
            make_uint4(p_pack[kp][0], p_pack[kp][1], p_pack[kp][2], p_pack[kp][3]);
      }
      my_is_first = false;
    }

    if (nb > cta_nb_min) {
      cp_async_wait<2>();
    } else {
      cp_async_wait<1>();
    }
    __syncthreads();

    // 3A. PV Half 0 (reads kv[s2] [+ tree_kv]) + interleaved w_sub=0 exact ni=0..11 row_sum!
    if (!nb_has_remap) {
      if (my_in_block) {
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_v_pair64_v3(v_cur, smem.kv[s2], dp_base, 0, lane);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          const uint4 pl = smem.p_bf16[q_slot][ks_local][lane];
          const uint32_t p_cur[4] = {pl.x, pl.y, pl.z, pl.w};
          #pragma unroll
          for (int dp = 0; dp < 2; ++dp) {
            const int n_dp = dp ^ 1;
            const int n_ks = ks_local + dp;
            if (n_ks < 4) ldsm4_v_pair64_v3(v_nxt, smem.kv[s2], dp_base + n_dp, n_ks, lane);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            if (w_sub == 0) {
              if (dp == 0) {
                const float4 ph0 = smem.p_hi_f32[q_slot][ks_local * 3 + 0][lane];
                row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph0.x), ph0.y);
                row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph0.z), ph0.w);
                const float4 ph1 = smem.p_hi_f32[q_slot][ks_local * 3 + 1][lane];
                row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph1.x), ph1.y);
                row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph1.z), ph1.w);
              } else {
                const float4 ph2 = smem.p_hi_f32[q_slot][ks_local * 3 + 2][lane];
                row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph2.x), ph2.y);
                row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph2.z), ph2.w);
              }
            }
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
      }
    } else {
      if (my_in_block) {
        const int r_in_16 = lane & 15;
        const int dt_sub = lane >> 4;
        uint32_t v_row_base[4];
        int v_swz[4];
        #pragma unroll
        for (int kl = 0; kl < 4; ++kl) {
          const int r = (kl << 4) + r_in_16;
          const int tok = block_tok_base + r;
          if (tok >= my_k_start && tok < my_seqlen_k) {
            if (my_remap_base >= 0 && tok >= my_remap_base) {
              const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
              v_row_base[kl] = smem_u32(&smem.tree_kv[rr][0]);
              v_swz[kl] = rr & 7;
            } else {
              v_row_base[kl] = smem_u32(&smem.kv[s2][r][0]);
              v_swz[kl] = r & 7;
            }
          } else {
            v_row_base[kl] = smem_u32(&smem.zero_row[0]);
            v_swz[kl] = 0;
          }
        }
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_trans_raw_addr(v_cur, v_row_base[0] + (((dp_base << 1) + dt_sub) ^ v_swz[0]) * 16u);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          const uint4 pl = smem.p_bf16[q_slot][ks_local][lane];
          const uint32_t p_cur[4] = {pl.x, pl.y, pl.z, pl.w};
          #pragma unroll
          for (int dp = 0; dp < 2; ++dp) {
            const int n_dp = dp ^ 1;
            const int n_ks = ks_local + dp;
            if (n_ks < 4) {
              const int n_dt = ((dp_base + n_dp) << 1) + dt_sub;
              ldsm4_trans_raw_addr(v_nxt, v_row_base[n_ks] + ((n_dt ^ v_swz[n_ks]) << 4));
            }
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            if (w_sub == 0) {
              if (dp == 0) {
                const float4 ph0 = smem.p_hi_f32[q_slot][ks_local * 3 + 0][lane];
                row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph0.x), ph0.y);
                row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph0.z), ph0.w);
                const float4 ph1 = smem.p_hi_f32[q_slot][ks_local * 3 + 1][lane];
                row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph1.x), ph1.y);
                row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph1.z), ph1.w);
              } else {
                const float4 ph2 = smem.p_hi_f32[q_slot][ks_local * 3 + 2][lane];
                row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph2.x), ph2.y);
                row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph2.z), ph2.w);
              }
            }
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
      }
    }

    if (nb > cta_nb_min) {
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    if (nb > cta_nb_min) {
      issue_kv64_v4<false, kKvL2>(
          smem.kv[s2], kv_ptr, block_table, ((nb - 1) << 7) + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
      cp_async_commit();
    }

    // 3B. PV Half 1 (reads kv[s0] [+ tree_kv]).
    if (!nb_has_remap) {
      if (my_in_block) {
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_v_pair64_v3(v_cur, smem.kv[s0], dp_base, 0, lane);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          const uint4 ph = smem.p_bf16[q_slot][4 + ks_local][lane];
          const uint32_t p_cur[4] = {ph.x, ph.y, ph.z, ph.w};
          #pragma unroll
          for (int dp = 0; dp < 2; ++dp) {
            const int n_dp = dp ^ 1;
            const int n_ks = ks_local + dp;
            if (n_ks < 4) ldsm4_v_pair64_v3(v_nxt, smem.kv[s0], dp_base + n_dp, n_ks, lane);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
      }
    } else {
      if (my_in_block) {
        const int r_in_16 = lane & 15;
        const int dt_sub = lane >> 4;
        uint32_t v_row_base[4];
        int v_swz[4];
        #pragma unroll
        for (int kl = 0; kl < 4; ++kl) {
          const int r = (kl << 4) + r_in_16;
          const int tok = block_tok_base + 64 + r;
          if (tok >= my_k_start && tok < my_seqlen_k) {
            if (my_remap_base >= 0 && tok >= my_remap_base) {
              const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
              v_row_base[kl] = smem_u32(&smem.tree_kv[rr][0]);
              v_swz[kl] = rr & 7;
            } else {
              v_row_base[kl] = smem_u32(&smem.kv[s0][r][0]);
              v_swz[kl] = r & 7;
            }
          } else {
            v_row_base[kl] = smem_u32(&smem.zero_row[0]);
            v_swz[kl] = 0;
          }
        }
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_trans_raw_addr(v_cur, v_row_base[0] + (((dp_base << 1) + dt_sub) ^ v_swz[0]) * 16u);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          const uint4 ph = smem.p_bf16[q_slot][4 + ks_local][lane];
          const uint32_t p_cur[4] = {ph.x, ph.y, ph.z, ph.w};
          #pragma unroll
          for (int dp = 0; dp < 2; ++dp) {
            const int n_dp = dp ^ 1;
            const int n_ks = ks_local + dp;
            if (n_ks < 4) {
              const int n_dt = ((dp_base + n_dp) << 1) + dt_sub;
              ldsm4_trans_raw_addr(v_nxt, v_row_base[n_ks] + ((n_dt ^ v_swz[n_ks]) << 4));
            }
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
      }
    }

    const int ns0 = s1;
    const int ns1 = s2;
    const int ns2 = s0;
    s0 = ns0;
    s1 = ns1;
    s2 = ns2;
  }

  if (my_active && w_sub == 0) {
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 2));
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 1));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 2));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 1));
    const bool empty0 = (row_sum0 == 0.0f || row_sum0 != row_sum0 || row_max0 == -INFINITY);
    const bool empty1 = (row_sum1 == 0.0f || row_sum1 != row_sum1 || row_max1 == -INFINITY);
    const float inv0 = empty0 ? 1.0f : fast_rcpf(row_sum0);
    const float inv1 = empty1 ? 1.0f : fast_rcpf(row_sum1);
    if (tid_in_grp == 0) {
      smem.inv_exch[q_slot][group_id] = inv0;
      smem.inv_exch[q_slot][group_id + 8] = inv1;
    }
  }
  __syncthreads();

  if (my_active) {
    const float inv0 = smem.inv_exch[q_slot][group_id];
    const float inv1 = smem.inv_exch[q_slot][group_id + 8];
    const int d_off = w_sub << 5;
    __nv_bfloat16* const out_row = out_ptr + static_cast<int64_t>(q0 + my_q_idx) * out_tok_stride;
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      const int d_col = d_off + (dt << 3) + (tid_in_grp << 1);
      const __nv_bfloat162 ob0 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[dt][0], inv0), mul_ftz_f32(o_frag[dt][1], inv0));
      const __nv_bfloat162 ob1 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[dt][2], inv1), mul_ftz_f32(o_frag[dt][3], inv1));
      *reinterpret_cast<uint32_t*>(
          out_row + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob0);
      *reinterpret_cast<uint32_t*>(
          out_row + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob1);
    }
  }
}

// ============================================================================
// Wave-27 V4.2 Pipe-Four (4-Slot Dedicated K/V Ring) Quad-Warp FA2 Verify Kernel (2Q x 4w/Q)
//
// Strictly 0-ULP bit-identical to FlashAttention-2 kBlockN=128 & verify_v4_2q4w_ring3:
//   - Expands KV SMEM from 3 -> 4 slots (k_smem[2] + v_smem[2] = 64 KiB; total 95,232 B <= 99,328 B
//     on sm_121a, zero occupancy change: 16 CTAs at M<=16, 32 CTAs at M=32 on 48 SMs).
//   - Cuts CTA barriers from 4 -> 3 per 128-token block (25% barrier cut across all 52 layers):
//       * BOTH V[0] & V[1] issue BEFORE QK^T (hidden behind QK^T + max + Softmax).
//       * BOTH next K[0] & K[1] issue BEFORE Softmax (hidden behind Softmax + PV Half 0 + PV Half 1).
//       * PV Half 0 & PV Half 1 execute back-to-back with ZERO intermediate __syncthreads()!
//   - CTA-uniform dead upper-half skip (has_hi = block_tok_base + 64 < cta_seqlen_k):
//     on seqlen_k = 513..576 (50% of timed decode window), skips K[1]/V[1] SMEM zero-fills,
//     w_sub>=2 QK^T & Softmax, dead p_hi_f32[4..11] adds, and PV Half 1 (strict 0-ULP).
//   - Dual-warp cooperative Q load (w_sub < 2 -> 4 iters instead of 8) into v_smem[0]
//     overlapped with prologue K[0..1] cp.async into k_smem[0..1].
// ============================================================================
struct __align__(128) VerifyV4_2Q4W_Pipe4Storage {
  uint4 k_smem[2][64][16];     // 32,768 B (K[0], K[1] for current 128-token block)
  uint4 v_smem[2][64][16];     // 32,768 B (V[0], V[1]; v_smem[0] aliases q_tiles[2][16][16] in prologue)
  uint4 tree_kv[32][16];       //  8,192 B (up to 32 tree K/V rows for remapped tail block)
  uint4 zero_row[16];          //    256 B (constant +0.0 BF16 row for out-of-range lanes)
  float4 p_hi_f32[2][12][32];  // 12,288 B (w_sub=1..3 -> w_sub=0 exact ni=4..15 row_sum)
  uint4 p_bf16[2][8][32];      //  8,192 B (all 4 warps publish/read packed BF16x2 P[0..7])
  float max_exch[2][4][16];    //    512 B (4-warp QK row max exchange)
  float inv_exch[2][16];       //    128 B (final inv_row_sum broadcast to w_sub=1..3)
  int32_t nb_max_q[2];         //      8 B
  int32_t nb_min_q[2];         //      8 B
  int32_t k_start_q[2];        //      8 B
  int32_t seqlen_q[2];         //      8 B
  int32_t remap_base_q[2];     //      8 B
  uint8_t active_q[2];         //      2 B
};                             // 95,232 B (<= 99,328 B dynamic SMEM on sm_121a)
static_assert(sizeof(VerifyV4_2Q4W_Pipe4Storage) <= 99328, "VerifyV4_2Q4W_Pipe4Storage must fit SM121 dynamic SMEM");

template <bool kPdl = false, int kKvL2 = 0>
__global__ __launch_bounds__(256, 1) void gqa16_verify_v4_2q4w_pipe4_kernel(
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
    int64_t stride_kv_head,
    const int32_t* __restrict__ cu_seqlens_q_ptr,
    int64_t q_tok_stride,
    int64_t out_tok_stride,
    const int32_t* __restrict__ leaf_flag_ptr) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  VerifyV4_2Q4W_Pipe4Storage& smem = *reinterpret_cast<VerifyV4_2Q4W_Pipe4Storage*>(smem_raw);

  const int kv_head    = blockIdx.y;
  const int tid        = threadIdx.x;
  const int warp_id    = tid >> 5;       // 0..7
  const int q_slot     = warp_id >> 2;   // 0..1 (2 Q tokens / CTA)
  const int w_sub      = warp_id & 3;    // 0..3 (4 cooperating warps / Q token)
  const int lane       = tid & 31;
  const int group_id   = lane >> 2;      // 0..7
  const int tid_in_grp = lane & 3;       // 0..3

  const int q0 = cu_seqlens_q_ptr[0];
  const int actual_m = cu_seqlens_q_ptr[1] - q0;
  const int q_base = blockIdx.z * 2;
  if (q_base >= actual_m) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  const int raw_seqlen_k = seqused_k_ptr[0];
  const int lf0 = (leaf_flag_ptr != nullptr) ? leaf_flag_ptr[0] : 0;
  const int my_q_idx = q_base + q_slot;

  bool my_active = false;
  bool my_empty = false;
  int my_seqlen_k = 0;
  int my_k_start = 0;
  int my_num_nb = 0;
  int my_nb_min = 0;
  int my_remap_base = -1;
  uint32_t my_remap_packed = 0u;

  if (my_q_idx < actual_m) {
    int sk = raw_seqlen_k;
    int lrb = -1;
    uint32_t lrp = 0u;
    if (leaf_flag_ptr != nullptr && lf0 < 0 && actual_m <= 32 && sk >= actual_m) {
      const int base = 1 + my_q_idx * 3;
      const int seq_sub = leaf_flag_ptr[base];
      const int base_sub = leaf_flag_ptr[base + 1];
      if (base_sub >= 0) {
        lrb = sk - base_sub;
        lrp = static_cast<uint32_t>(leaf_flag_ptr[base + 2]);
      }
      sk -= seq_sub;
    } else if (my_q_idx == 7 && actual_m == 8 && leaf_flag_ptr != nullptr) {
      if ((lf0 == 1 || lf0 == 2) && sk >= 8) {
        const int lps = sk - 8;
        lrb = lps + lf0;
        lrp = 7u;
        sk = lps + lf0 + 1;
      } else {
        sk -= (actual_m - 1 - my_q_idx);
      }
    } else if (my_q_idx == 6 && actual_m == 7 && leaf_flag_ptr != nullptr && lf0 > 0 && sk >= 7) {
      const int lps = sk - 7;
      lrb = lps + 1;
      lrp = 6u;
      sk -= 5;
    } else {
      sk -= (actual_m - 1 - my_q_idx);
    }
    if (sk < 0) sk = 0;

    const int ks = (window_left >= 0) ? ((sk - 1 - window_left > 0) ? (sk - 1 - window_left) : 0) : 0;
    const int nnb = (sk > 0) ? ((sk + 127) >> 7) : 0;
    const int gnb = (window_left >= 0 && sk > 0) ? (ks >> 7) : 0;
    if (gnb >= nnb) {
      my_empty = true;
    } else {
      my_active = true;
      my_seqlen_k = sk;
      my_k_start = ks;
      my_num_nb = nnb;
      my_nb_min = gnb;
      my_remap_base = lrb;
      my_remap_packed = lrp;
    }
  }

  if (w_sub == 0 && lane == 0) {
    smem.active_q[q_slot] = my_active ? 1u : 0u;
    smem.nb_max_q[q_slot] = my_active ? my_num_nb : 0;
    smem.nb_min_q[q_slot] = my_active ? my_nb_min : 0x7fffffff;
    smem.k_start_q[q_slot] = my_active ? my_k_start : 0x7fffffff;
    smem.seqlen_q[q_slot] = my_active ? my_seqlen_k : 0;
    smem.remap_base_q[q_slot] = (my_active && my_remap_base >= 0) ? my_remap_base : 0x7fffffff;
  }
  if (tid < 16) {
    smem.zero_row[tid] = make_uint4(0u, 0u, 0u, 0u);
  }
  __syncthreads();

  bool cta_any_active = false;
  int cta_nb_max = 0;
  int cta_nb_min = 0x7fffffff;
  int cta_k_start = 0x7fffffff;
  int cta_seqlen_k = 0;
  int cta_remap_min_base = 0x7fffffff;
  #pragma unroll
  for (int qs = 0; qs < 2; ++qs) {
    if (smem.active_q[qs]) {
      cta_any_active = true;
      if (smem.nb_max_q[qs] > cta_nb_max) cta_nb_max = smem.nb_max_q[qs];
      if (smem.nb_min_q[qs] < cta_nb_min) cta_nb_min = smem.nb_min_q[qs];
      if (smem.k_start_q[qs] < cta_k_start) cta_k_start = smem.k_start_q[qs];
      if (smem.seqlen_q[qs] > cta_seqlen_k) cta_seqlen_k = smem.seqlen_q[qs];
      if (smem.remap_base_q[qs] < cta_remap_min_base) cta_remap_min_base = smem.remap_base_q[qs];
    }
  }
  const bool cta_has_remap = (cta_remap_min_base < 0x7fffffff);
  const int cta_p_start = raw_seqlen_k - actual_m;

  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  if (my_empty && w_sub == 0) {
    __nv_bfloat16* const dst_base = out_ptr + static_cast<int64_t>(q0 + my_q_idx) * out_tok_stride +
                                    static_cast<int64_t>(kv_head * 16) * out_head_stride;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = (u << 5) + lane;
      const int m = idx >> 4;
      const int c = idx & 15;
      *reinterpret_cast<uint4*>(dst_base + static_cast<int64_t>(m) * out_head_stride + (c << 3)) =
          make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (!cta_any_active) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  // Prologue: dual-warp cooperative Q load into v_smem[0] while priming block-0 K[0..1](/tree_K) into k_smem[0..1].
  const uint64_t kv_l2_pol = (kKvL2 == 2) ? kv_pol_evict_last() : 0ull;
  uint32_t q_frag[8][4];
  {
    uint4 (*q_tiles)[16][16] = reinterpret_cast<uint4 (*)[16][16]>(smem.v_smem[0]);
    if (my_active && w_sub < 2) {
      const __nv_bfloat16* q_tok_base = q_ptr + static_cast<int64_t>(q0 + my_q_idx) * q_tok_stride +
                                        static_cast<int64_t>(kv_head * 16) * q_head_stride;
      #pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int idx = (u << 6) + (w_sub << 5) + lane;
        const int m = idx >> 4;
        const int c = idx & 15;
        q_tiles[q_slot][m][c ^ (m & 7)] =
            *reinterpret_cast<const uint4*>(q_tok_base + static_cast<int64_t>(m) * q_head_stride + (c << 3));
      }
    }
    const int nb0_tok_base = (cta_nb_max - 1) << 7;
    const bool nb0_has_hi = (nb0_tok_base + 64 < cta_seqlen_k);
    issue_kv64_v4<false, kKvL2>(
        smem.k_smem[0], kv_ptr, block_table, nb0_tok_base,
        cta_k_start, cta_seqlen_k, block_size, kv_head,
        stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    if (nb0_has_hi) {
      issue_kv64_v4<false, kKvL2>(
          smem.k_smem[1], kv_ptr, block_table, nb0_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    }
    if (cta_has_remap && (nb0_tok_base + 128 > cta_remap_min_base)) {
      issue_tree_kv_v4<false, kKvL2>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    }
    cp_async_commit();
    __syncthreads();
    if (my_active) {
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        ldsm4_swizzled(q_frag[ks], q_tiles[q_slot], ks, lane);
      }
    }
  }

  float o_frag[4][4];
  #pragma unroll
  for (int dt = 0; dt < 4; ++dt) {
    o_frag[dt][0] = 0.0f; o_frag[dt][1] = 0.0f; o_frag[dt][2] = 0.0f; o_frag[dt][3] = 0.0f;
  }
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;
  bool my_is_first = true;

  const int kv_half = w_sub >> 1;
  const int sp_base = (w_sub & 1) << 1;
  const int dp_base = w_sub << 1;

  for (int nb = cta_nb_max - 1; nb >= cta_nb_min; --nb) {
    const int block_tok_base = nb << 7;
    const bool has_hi = (block_tok_base + 64 < cta_seqlen_k);
    const bool nb_has_remap = cta_has_remap && (block_tok_base + 128 > cta_remap_min_base);
    const bool my_in_block = my_active && (nb < my_num_nb) && (nb >= my_nb_min);

    // Barrier #1: k_smem[0..1] ready and v_smem[0..1] free (after prologue q_tiles or prior PV).
    cp_async_wait<0>();
    __syncthreads();

    if (nb_has_remap && nb < cta_nb_max - 1) {
      issue_tree_kv_v4<false, kKvL2>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();
    }

    // Issue BOTH V[0] & (if has_hi) V[1] BEFORE QK^T -> hidden behind QK^T + Softmax!
    issue_kv64_v4<true, kKvL2>(
        smem.v_smem[0], kv_ptr, block_table, block_tok_base,
        cta_k_start, cta_seqlen_k, block_size, kv_head,
        stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    if (has_hi) {
      issue_kv64_v4<true, kKvL2>(
          smem.v_smem[1], kv_ptr, block_table, block_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
    }
    cp_async_commit();

    float s_frag[4][4];
    #pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      s_frag[ni][0] = 0.0f; s_frag[ni][1] = 0.0f; s_frag[ni][2] = 0.0f; s_frag[ni][3] = 0.0f;
    }

    const bool my_qk_active = my_in_block && (has_hi || kv_half == 0);
    if (!nb_has_remap) {
      if (my_qk_active) {
        const uint4 (*my_k_tile)[16] = (kv_half == 0) ? smem.k_smem[0] : smem.k_smem[1];
        uint32_t b_cur[4], b_nxt[4];
        ldsm4_k_pair64_v3(b_cur, my_k_tile, sp_base, 0, lane);
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          #pragma unroll
          for (int sp = 0; sp < 2; ++sp) {
            const int n_sp = sp ^ 1;
            const int n_ks = ks + sp;
            if (n_ks < 8) ldsm4_k_pair64_v3(b_nxt, my_k_tile, sp_base + n_sp, n_ks, lane);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[0], b_cur[1], s_frag[sp * 2 + 0]);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[2], b_cur[3], s_frag[sp * 2 + 1]);
            b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1]; b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
          }
        }
      }
    } else {
      if (my_qk_active) {
        const uint4 (*my_k_tile)[16] = (kv_half == 0) ? smem.k_smem[0] : smem.k_smem[1];
        const int half_base = block_tok_base + (kv_half << 6);
        const int r_local = lane & 7;
        const int sub_k = (lane >> 3) & 1;
        uint32_t k_row_base[2];
        int k_swz[2];
        #pragma unroll
        for (int sp = 0; sp < 2; ++sp) {
          const int sub = ((sp_base + sp) << 1) + (lane >> 4);
          const int r = (sub << 3) + r_local;
          const int tok = half_base + r;
          if (tok >= my_k_start && tok < my_seqlen_k) {
            if (my_remap_base >= 0 && tok >= my_remap_base) {
              const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
              k_row_base[sp] = smem_u32(&smem.tree_kv[rr][0]);
              k_swz[sp] = rr & 7;
            } else {
              k_row_base[sp] = smem_u32(&my_k_tile[r][0]);
              k_swz[sp] = r_local;
            }
          } else {
            k_row_base[sp] = smem_u32(&smem.zero_row[0]);
            k_swz[sp] = 0;
          }
        }
        uint32_t b_cur[4], b_nxt[4];
        ldsm4_raw_addr(b_cur, k_row_base[0] + ((sub_k ^ k_swz[0]) << 4));
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          #pragma unroll
          for (int sp = 0; sp < 2; ++sp) {
            const int n_sp = sp ^ 1;
            const int n_ks = ks + sp;
            if (n_ks < 8) {
              ldsm4_raw_addr(b_nxt, k_row_base[n_sp] + ((((n_ks << 1) + sub_k) ^ k_swz[n_sp]) << 4));
            }
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[0], b_cur[1], s_frag[sp * 2 + 0]);
            mma_m16n8k16_bf16_u2(q_frag[ks], b_cur[2], b_cur[3], s_frag[sp * 2 + 1]);
            b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1]; b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
          }
        }
      }
    }

    if (my_in_block) {
      if (has_hi || kv_half == 0) {
        const int chunk_base = block_tok_base + (w_sub << 5);
        const bool need_mask = (chunk_base < my_k_start) || (chunk_base + 32 > my_seqlen_k);
        float m0 = -INFINITY, m1 = -INFINITY;
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          if (need_mask) {
            const int col0 = chunk_base + (ni << 3) + (tid_in_grp << 1);
            const int col1 = col0 + 1;
            if (col0 < my_k_start || col0 >= my_seqlen_k) { s_frag[ni][0] = -INFINITY; s_frag[ni][2] = -INFINITY; }
            if (col1 < my_k_start || col1 >= my_seqlen_k) { s_frag[ni][1] = -INFINITY; s_frag[ni][3] = -INFINITY; }
          }
          m0 = max_ftz_f32(m0, max_ftz_f32(s_frag[ni][0], s_frag[ni][1]));
          m1 = max_ftz_f32(m1, max_ftz_f32(s_frag[ni][2], s_frag[ni][3]));
        }
        m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
        m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
        m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
        m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
        if (tid_in_grp == 0) {
          smem.max_exch[q_slot][w_sub][group_id] = m0;
          smem.max_exch[q_slot][w_sub][group_id + 8] = m1;
        }
      } else if (tid_in_grp == 0) {
        smem.max_exch[q_slot][w_sub][group_id] = -INFINITY;
        smem.max_exch[q_slot][w_sub][group_id + 8] = -INFINITY;
      }
    }
    // Barrier #2: max_exch visible; k_smem[0..1] and tree_kv are now free!
    __syncthreads();

    // Issue (tree_V if remap) + BOTH next K[0..1] BEFORE Softmax -> K(nb-1) hidden behind Softmax + PV0 + PV1!
    if (nb_has_remap) {
      issue_tree_kv_v4<true, kKvL2>(
          smem.tree_kv, kv_ptr, block_table, cta_p_start, actual_m,
          block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
      cp_async_commit();
    }
    if (nb > cta_nb_min) {
      const int nxt_tok_base = (nb - 1) << 7;
      issue_kv64_v4<false, kKvL2>(
          smem.k_smem[0], kv_ptr, block_table, nxt_tok_base,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
      issue_kv64_v4<false, kKvL2>(
          smem.k_smem[1], kv_ptr, block_table, nxt_tok_base + 64,
          cta_k_start, cta_seqlen_k, block_size, kv_head,
          stride_block, stride_slot, stride_kv_head, tid, kv_l2_pol);
      cp_async_commit();
    }

    if constexpr (kPdl) {
      if (nb == cta_nb_min) {
        asm volatile("griddepcontrol.launch_dependents;");
      }
    }

    if (my_in_block) {
      const float m0 = has_hi
          ? max_ftz_f32(
                max_ftz_f32(smem.max_exch[q_slot][0][group_id], smem.max_exch[q_slot][1][group_id]),
                max_ftz_f32(smem.max_exch[q_slot][2][group_id], smem.max_exch[q_slot][3][group_id]))
          : max_ftz_f32(smem.max_exch[q_slot][0][group_id], smem.max_exch[q_slot][1][group_id]);
      const float m1 = has_hi
          ? max_ftz_f32(
                max_ftz_f32(smem.max_exch[q_slot][0][group_id + 8], smem.max_exch[q_slot][1][group_id + 8]),
                max_ftz_f32(smem.max_exch[q_slot][2][group_id + 8], smem.max_exch[q_slot][3][group_id + 8]))
          : max_ftz_f32(smem.max_exch[q_slot][0][group_id + 8], smem.max_exch[q_slot][1][group_id + 8]);

      float sc0 = 1.0f, sc1 = 1.0f;
      if (my_is_first) {
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
        for (int dt = 0; dt < 4; ++dt) {
          o_frag[dt][0] = mul_ftz_f32(o_frag[dt][0], sc0);
          o_frag[dt][1] = mul_ftz_f32(o_frag[dt][1], sc0);
          o_frag[dt][2] = mul_ftz_f32(o_frag[dt][2], sc1);
          o_frag[dt][3] = mul_ftz_f32(o_frag[dt][3], sc1);
        }
      }

      if (has_hi || kv_half == 0) {
        const float max_sc0 = (row_max0 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max0);
        const float max_sc1 = (row_max1 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max1);
        uint32_t p_pack[2][4];
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          const float s0_v = s_frag[ni][0];
          const float s1_v = s_frag[ni][1];
          const float s2_v = s_frag[ni][2];
          const float s3_v = s_frag[ni][3];
          const float p0 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s0_v, scale_log2), max_sc0));
          const float p1 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s1_v, scale_log2), max_sc0));
          const float p2 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s2_v, scale_log2), max_sc1));
          const float p3 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s3_v, scale_log2), max_sc1));
          if (w_sub == 0) {
            if (ni == 0) {
              if (my_is_first) {
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
          } else {
            smem.p_hi_f32[q_slot][((w_sub - 1) << 2) + ni][lane] = make_float4(p0, p1, p2, p3);
          }
          const __nv_bfloat162 pb0 = __floats2bfloat162_rn(p0, p1);
          const __nv_bfloat162 pb1 = __floats2bfloat162_rn(p2, p3);
          const int kp = ni >> 1;
          const int hi = (ni & 1) << 1;
          p_pack[kp][hi + 0] = *reinterpret_cast<const uint32_t*>(&pb0);
          p_pack[kp][hi + 1] = *reinterpret_cast<const uint32_t*>(&pb1);
        }
        #pragma unroll
        for (int kp = 0; kp < 2; ++kp) {
          smem.p_bf16[q_slot][(w_sub << 1) + kp][lane] =
              make_uint4(p_pack[kp][0], p_pack[kp][1], p_pack[kp][2], p_pack[kp][3]);
        }
      }
      my_is_first = false;
    }

    // Barrier #3: BOTH V[0..1] (+ tree_V if remap), p_bf16, and p_hi_f32 are ready!
    if (nb > cta_nb_min) {
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    // 3A+3B. PV Half 0 (v_smem[0]) + interleaved w_sub=0 exact row_sum + back-to-back PV Half 1 (v_smem[1])!
    if (!nb_has_remap) {
      if (my_in_block) {
        uint32_t v_cur[4], v_nxt[4];
        ldsm4_v_pair64_v3(v_cur, smem.v_smem[0], dp_base, 0, lane);
        #pragma unroll
        for (int ks_local = 0; ks_local < 4; ++ks_local) {
          const uint4 pl = smem.p_bf16[q_slot][ks_local][lane];
          const uint32_t p_cur[4] = {pl.x, pl.y, pl.z, pl.w};
          #pragma unroll
          for (int dp = 0; dp < 2; ++dp) {
            const int n_dp = dp ^ 1;
            const int n_ks = ks_local + dp;
            if (n_ks < 4) ldsm4_v_pair64_v3(v_nxt, smem.v_smem[0], dp_base + n_dp, n_ks, lane);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
            mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
            if (w_sub == 0) {
              if (has_hi) {
                if (dp == 0) {
                  const float4 ph0 = smem.p_hi_f32[q_slot][ks_local * 3 + 0][lane];
                  row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph0.x), ph0.y);
                  row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph0.z), ph0.w);
                  const float4 ph1 = smem.p_hi_f32[q_slot][ks_local * 3 + 1][lane];
                  row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph1.x), ph1.y);
                  row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph1.z), ph1.w);
                } else {
                  const float4 ph2 = smem.p_hi_f32[q_slot][ks_local * 3 + 2][lane];
                  row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph2.x), ph2.y);
                  row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph2.z), ph2.w);
                }
              } else if (dp == 0) {
                const float4 ph = smem.p_hi_f32[q_slot][ks_local][lane];
                row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph.x), ph.y);
                row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph.z), ph.w);
              }
            }
            v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
          }
        }
        if (has_hi) {
          ldsm4_v_pair64_v3(v_cur, smem.v_smem[1], dp_base, 0, lane);
          #pragma unroll
          for (int ks_local = 0; ks_local < 4; ++ks_local) {
            const uint4 ph = smem.p_bf16[q_slot][4 + ks_local][lane];
            const uint32_t p_cur[4] = {ph.x, ph.y, ph.z, ph.w};
            #pragma unroll
            for (int dp = 0; dp < 2; ++dp) {
              const int n_dp = dp ^ 1;
              const int n_ks = ks_local + dp;
              if (n_ks < 4) ldsm4_v_pair64_v3(v_nxt, smem.v_smem[1], dp_base + n_dp, n_ks, lane);
              mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
              mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
              v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
            }
          }
        }
      }
    } else {
      if (my_in_block) {
        const int r_in_16 = lane & 15;
        const int dt_sub = lane >> 4;
        {
          uint32_t v_row_base[4];
          int v_swz[4];
          #pragma unroll
          for (int kl = 0; kl < 4; ++kl) {
            const int r = (kl << 4) + r_in_16;
            const int tok = block_tok_base + r;
            if (tok >= my_k_start && tok < my_seqlen_k) {
              if (my_remap_base >= 0 && tok >= my_remap_base) {
                const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
                v_row_base[kl] = smem_u32(&smem.tree_kv[rr][0]);
                v_swz[kl] = rr & 7;
              } else {
                v_row_base[kl] = smem_u32(&smem.v_smem[0][r][0]);
                v_swz[kl] = r & 7;
              }
            } else {
              v_row_base[kl] = smem_u32(&smem.zero_row[0]);
              v_swz[kl] = 0;
            }
          }
          uint32_t v_cur[4], v_nxt[4];
          ldsm4_trans_raw_addr(v_cur, v_row_base[0] + (((dp_base << 1) + dt_sub) ^ v_swz[0]) * 16u);
          #pragma unroll
          for (int ks_local = 0; ks_local < 4; ++ks_local) {
            const uint4 pl = smem.p_bf16[q_slot][ks_local][lane];
            const uint32_t p_cur[4] = {pl.x, pl.y, pl.z, pl.w};
            #pragma unroll
            for (int dp = 0; dp < 2; ++dp) {
              const int n_dp = dp ^ 1;
              const int n_ks = ks_local + dp;
              if (n_ks < 4) {
                const int n_dt = ((dp_base + n_dp) << 1) + dt_sub;
                ldsm4_trans_raw_addr(v_nxt, v_row_base[n_ks] + ((n_dt ^ v_swz[n_ks]) << 4));
              }
              mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
              mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
              if (w_sub == 0) {
                if (has_hi) {
                  if (dp == 0) {
                    const float4 ph0 = smem.p_hi_f32[q_slot][ks_local * 3 + 0][lane];
                    row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph0.x), ph0.y);
                    row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph0.z), ph0.w);
                    const float4 ph1 = smem.p_hi_f32[q_slot][ks_local * 3 + 1][lane];
                    row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph1.x), ph1.y);
                    row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph1.z), ph1.w);
                  } else {
                    const float4 ph2 = smem.p_hi_f32[q_slot][ks_local * 3 + 2][lane];
                    row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph2.x), ph2.y);
                    row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph2.z), ph2.w);
                  }
                } else if (dp == 0) {
                  const float4 ph = smem.p_hi_f32[q_slot][ks_local][lane];
                  row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, ph.x), ph.y);
                  row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, ph.z), ph.w);
                }
              }
              v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
            }
          }
        }
        if (has_hi) {
          uint32_t v_row_base[4];
          int v_swz[4];
          #pragma unroll
          for (int kl = 0; kl < 4; ++kl) {
            const int r = (kl << 4) + r_in_16;
            const int tok = block_tok_base + 64 + r;
            if (tok >= my_k_start && tok < my_seqlen_k) {
              if (my_remap_base >= 0 && tok >= my_remap_base) {
                const int rr = static_cast<int>((my_remap_packed >> ((tok - my_remap_base) * 5)) & 0x1Fu);
                v_row_base[kl] = smem_u32(&smem.tree_kv[rr][0]);
                v_swz[kl] = rr & 7;
              } else {
                v_row_base[kl] = smem_u32(&smem.v_smem[1][r][0]);
                v_swz[kl] = r & 7;
              }
            } else {
              v_row_base[kl] = smem_u32(&smem.zero_row[0]);
              v_swz[kl] = 0;
            }
          }
          uint32_t v_cur[4], v_nxt[4];
          ldsm4_trans_raw_addr(v_cur, v_row_base[0] + (((dp_base << 1) + dt_sub) ^ v_swz[0]) * 16u);
          #pragma unroll
          for (int ks_local = 0; ks_local < 4; ++ks_local) {
            const uint4 ph = smem.p_bf16[q_slot][4 + ks_local][lane];
            const uint32_t p_cur[4] = {ph.x, ph.y, ph.z, ph.w};
            #pragma unroll
            for (int dp = 0; dp < 2; ++dp) {
              const int n_dp = dp ^ 1;
              const int n_ks = ks_local + dp;
              if (n_ks < 4) {
                const int n_dt = ((dp_base + n_dp) << 1) + dt_sub;
                ldsm4_trans_raw_addr(v_nxt, v_row_base[n_ks] + ((n_dt ^ v_swz[n_ks]) << 4));
              }
              mma_m16n8k16_bf16_u2(p_cur, v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
              mma_m16n8k16_bf16_u2(p_cur, v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
              v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
            }
          }
        }
      }
    }
  }

  if (my_active && w_sub == 0) {
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 2));
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 1));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 2));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 1));
    const bool empty0 = (row_sum0 == 0.0f || row_sum0 != row_sum0 || row_max0 == -INFINITY);
    const bool empty1 = (row_sum1 == 0.0f || row_sum1 != row_sum1 || row_max1 == -INFINITY);
    const float inv0 = empty0 ? 1.0f : fast_rcpf(row_sum0);
    const float inv1 = empty1 ? 1.0f : fast_rcpf(row_sum1);
    if (tid_in_grp == 0) {
      smem.inv_exch[q_slot][group_id] = inv0;
      smem.inv_exch[q_slot][group_id + 8] = inv1;
    }
  }
  __syncthreads();

  if (my_active) {
    const float inv0 = smem.inv_exch[q_slot][group_id];
    const float inv1 = smem.inv_exch[q_slot][group_id + 8];
    const int d_off = w_sub << 5;
    __nv_bfloat16* const out_row = out_ptr + static_cast<int64_t>(q0 + my_q_idx) * out_tok_stride;
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      const int d_col = d_off + (dt << 3) + (tid_in_grp << 1);
      const __nv_bfloat162 ob0 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[dt][0], inv0), mul_ftz_f32(o_frag[dt][1], inv0));
      const __nv_bfloat162 ob1 = __floats2bfloat162_rn(
          mul_ftz_f32(o_frag[dt][2], inv1), mul_ftz_f32(o_frag[dt][3], inv1));
      *reinterpret_cast<uint32_t*>(
          out_row + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob0);
      *reinterpret_cast<uint32_t*>(
          out_row + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) =
          *reinterpret_cast<const uint32_t*>(&ob1);
    }
  }
}

static bool gqa16_verify_pipe4_enabled() {
  static const bool enabled = []() -> bool {
    const char* env = std::getenv("MUSE_GQA16_PIPE4");
    if (!env || !*env) return true;  // default ON: 4-slot dedicated K/V ring (3 barriers, 16/32 CTAs, dead-hi skip)
    return std::strtol(env, nullptr, 10) != 0;
  }();
  return enabled;
}

static int gqa16_verify_v4_mode() {
  static const int mode = []() -> int {
    const char* env = std::getenv("MUSE_GQA16_VERIFY_V4");
    if (!env || !*env) return 1;  // default 1 = 4Q x 2-warp/Q (v4, min_q=17 -> M=32 + M=16 fa2/wg2); 2 = v4.1; 0 = legacy
    long v = std::strtol(env, nullptr, 10);
    if (v <= 0) return 0;
    if (v == 1) return 1;
    return 2;
  }();
  return mode;
}

static int gqa16_verify_v4_min_q() {
  static const int min_q = []() -> int {
    const char* env = std::getenv("MUSE_GQA16_VERIFY_V4_MIN_Q");
    if (!env || !*env) return 17;  // for mode==1 (4Q x 2w): skip M<=16 so M=16 stays fast on fa2/wg2
    return static_cast<int>(std::strtol(env, nullptr, 10));
  }();
  return min_q;
}

static bool gqa16_verify_v4_ring3_enabled() {
  static const bool enabled = []() -> bool {
    const char* env = std::getenv("MUSE_GQA16_V4_RING3");
    if (!env || !*env) return true;  // default ON: 3-slot zero-stall KV ring (83 KiB SMEM, 16 CTAs)
    return std::strtol(env, nullptr, 10) != 0;
  }();
  return enabled;
}

static bool gqa16_verify_v4_full_2q4w_enabled() {
  static const bool enabled = []() -> bool {
    const char* env = std::getenv("MUSE_GQA16_V4_FULL_2Q4W");
    if (!env || !*env) return false;  // env-gated: 1 promotes window_left < 0 full-attn layers to 32-CTA 2q4w
    return std::strtol(env, nullptr, 10) != 0;
  }();
  return enabled;
}

static int gqa16_verify_kv_l2_mode() {
  static const int mode = []() -> int {
    const char* env = std::getenv("MUSE_GQA16_KV_L2");
    if (!env || !*env) return 2;  // default 2: L2::256B + L2::evict_last (pins 640 KiB shared KV across 8 CTAs/head)
    long v = std::strtol(env, nullptr, 10);
    if (v <= 0) return 0;         // 0 = plain cp.async.cg
    if (v == 1) return 1;         // 1 = L2::256B only
    return 2;                     // 2 = L2::256B + L2::evict_last
  }();
  return mode;
}

static bool gqa16_verify_m16_ring3_enabled() {
  static const bool enabled = []() -> bool {
    const char* env = std::getenv("MUSE_GQA16_M16_RING3");
    if (!env || !*env) return true;  // default ON: route M<=16 verify to 16-CTA 2q4w_ring3 (0 spill, kv[3] ring)
    return std::strtol(env, nullptr, 10) != 0;
  }();
  return enabled;
}

static void gqa16_multiq_set_attrs_once() {
  static bool done = false;
  if (done) return;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  cudaThreadExchangeStreamCaptureMode(&mode);
  cudaFuncSetAttribute(gqa16_fa2_block128_pipe_multiq_kernel<4, false>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(sizeof(MultiQPipe128Storage<4>)));
  cudaFuncSetAttribute(gqa16_fa2_block128_pipe_multiq_kernel<4, true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(sizeof(MultiQPipe128Storage<4>)));
  cudaFuncSetAttribute(gqa16_fa2_block128_pipe_multiq_kernel<2, false>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(sizeof(MultiQPipe128Storage<2>)));
  cudaFuncSetAttribute(gqa16_fa2_block128_pipe_multiq_kernel<2, true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(sizeof(MultiQPipe128Storage<2>)));
  cudaFuncSetAttribute(gqa16_verify_v3_kernel<4, false>,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(gqa16_verify_v3_kernel<4, true>,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(gqa16_verify_v4_kernel<false>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(sizeof(VerifyV4Storage)));
  cudaFuncSetAttribute(gqa16_verify_v4_kernel<false>,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(gqa16_verify_v4_kernel<true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(sizeof(VerifyV4Storage)));
  cudaFuncSetAttribute(gqa16_verify_v4_kernel<true>,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  auto set_ring3_attrs = [](auto fn, size_t smem_bytes) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
    cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  };
  set_ring3_attrs(gqa16_verify_v4_ring3_kernel<false, 0>, sizeof(VerifyV4Ring3Storage));
  set_ring3_attrs(gqa16_verify_v4_ring3_kernel<true, 0>,  sizeof(VerifyV4Ring3Storage));
  set_ring3_attrs(gqa16_verify_v4_ring3_kernel<false, 1>, sizeof(VerifyV4Ring3Storage));
  set_ring3_attrs(gqa16_verify_v4_ring3_kernel<true, 1>,  sizeof(VerifyV4Ring3Storage));
  set_ring3_attrs(gqa16_verify_v4_ring3_kernel<false, 2>, sizeof(VerifyV4Ring3Storage));
  set_ring3_attrs(gqa16_verify_v4_ring3_kernel<true, 2>,  sizeof(VerifyV4Ring3Storage));
  cudaFuncSetAttribute(gqa16_verify_v4_2q4w_kernel<false>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(sizeof(VerifyV4_2Q4WStorage)));
  cudaFuncSetAttribute(gqa16_verify_v4_2q4w_kernel<false>,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 50);
  cudaFuncSetAttribute(gqa16_verify_v4_2q4w_kernel<true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(sizeof(VerifyV4_2Q4WStorage)));
  cudaFuncSetAttribute(gqa16_verify_v4_2q4w_kernel<true>,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 50);
  set_ring3_attrs(gqa16_verify_v4_2q4w_ring3_kernel<false, 0>, sizeof(VerifyV4_2Q4W_Ring3Storage));
  set_ring3_attrs(gqa16_verify_v4_2q4w_ring3_kernel<true, 0>,  sizeof(VerifyV4_2Q4W_Ring3Storage));
  set_ring3_attrs(gqa16_verify_v4_2q4w_ring3_kernel<false, 1>, sizeof(VerifyV4_2Q4W_Ring3Storage));
  set_ring3_attrs(gqa16_verify_v4_2q4w_ring3_kernel<true, 1>,  sizeof(VerifyV4_2Q4W_Ring3Storage));
  set_ring3_attrs(gqa16_verify_v4_2q4w_ring3_kernel<false, 2>, sizeof(VerifyV4_2Q4W_Ring3Storage));
  set_ring3_attrs(gqa16_verify_v4_2q4w_ring3_kernel<true, 2>,  sizeof(VerifyV4_2Q4W_Ring3Storage));
  set_ring3_attrs(gqa16_verify_v4_2q4w_pipe4_kernel<false, 0>, sizeof(VerifyV4_2Q4W_Pipe4Storage));
  set_ring3_attrs(gqa16_verify_v4_2q4w_pipe4_kernel<true, 0>,  sizeof(VerifyV4_2Q4W_Pipe4Storage));
  set_ring3_attrs(gqa16_verify_v4_2q4w_pipe4_kernel<false, 1>, sizeof(VerifyV4_2Q4W_Pipe4Storage));
  set_ring3_attrs(gqa16_verify_v4_2q4w_pipe4_kernel<true, 1>,  sizeof(VerifyV4_2Q4W_Pipe4Storage));
  set_ring3_attrs(gqa16_verify_v4_2q4w_pipe4_kernel<false, 2>, sizeof(VerifyV4_2Q4W_Pipe4Storage));
  set_ring3_attrs(gqa16_verify_v4_2q4w_pipe4_kernel<true, 2>,  sizeof(VerifyV4_2Q4W_Pipe4Storage));
  cudaThreadExchangeStreamCaptureMode(&mode);
  done = true;
}

static int gqa16_multiq_mode(int max_q) {
  static const int mode = []() -> int {
    const char* env = std::getenv("MUSE_GQA16_MULTIQ");
    if (!env || !*env) return -1;
    long v = std::strtol(env, nullptr, 10);
    if (v <= 0) return 0;
    if (v == 2) return 2;
    if (v == 4) return 4;
    if (v == 44) return 44;
    return 8;
  }();
  if (mode >= 0) return mode;
  return (max_q > 16) ? 8 : 0;
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
  const bool use_pipe128     = ((mode & 16) != 0);
  const float scale_log2 =
      use_scale_mode3
          ? (softmax_scale * muse_gqa16::kLog2e)
          : static_cast<float>(static_cast<double>(softmax_scale) * 1.44269504088896340736);
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

  if (fa2_splits == 1 && use_1pass_dsplit && (grid_splits == 4 || grid_splits == 8 || grid_splits == 16)) {
    const dim3 grid(static_cast<unsigned>(grid_splits), 2u, 1u);
    if (use_k128_single) {
      if (use_pipe128) {
        if (grid_splits == 8) {
          muse_gqa16::gqa16_1pass_dsplit_pipe_kernel<8><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
              static_cast<const __nv_bfloat16*>(q_ptr), static_cast<const __nv_bfloat16*>(kv_ptr),
              static_cast<__nv_bfloat16*>(out_ptr), static_cast<const int32_t*>(seqused_k_ptr),
              static_cast<const int32_t*>(block_table_ptr), block_size, window_left, scale_log2,
              q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head);
        } else if (grid_splits == 4) {
          muse_gqa16::gqa16_1pass_dsplit_pipe_kernel<4><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
              static_cast<const __nv_bfloat16*>(q_ptr), static_cast<const __nv_bfloat16*>(kv_ptr),
              static_cast<__nv_bfloat16*>(out_ptr), static_cast<const int32_t*>(seqused_k_ptr),
              static_cast<const int32_t*>(block_table_ptr), block_size, window_left, scale_log2,
              q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head);
        } else {
          muse_gqa16::gqa16_1pass_dsplit_pipe_kernel<16><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
              static_cast<const __nv_bfloat16*>(q_ptr), static_cast<const __nv_bfloat16*>(kv_ptr),
              static_cast<__nv_bfloat16*>(out_ptr), static_cast<const int32_t*>(seqused_k_ptr),
              static_cast<const int32_t*>(block_table_ptr), block_size, window_left, scale_log2,
              q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head);
        }
      } else if (grid_splits == 8) {
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
  } else if (fa2_splits > 1 && use_pipe128) {
    const dim3 grid(static_cast<unsigned>(grid_splits), 2u, 1u);
    muse_gqa16::gqa16_fa2_block128_pipe_kernel<<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
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

// =================================================================================================
// Strict 0-ULP FA2-exact GQA-16 causal prefill (M=512 / M=64 tail) + fused mul_sigmoid (BF16/FP8)
// + SM121 PDL producer trigger.
// =================================================================================================
namespace muse_gqa16_pf {

using muse_gqa16::smem_u32;
using muse_gqa16::max_ftz_f32;
using muse_gqa16::add_ftz_f32;
using muse_gqa16::sub_ftz_f32;
using muse_gqa16::mul_ftz_f32;
using muse_gqa16::fma_rn_ftz_f32;
using muse_gqa16::fast_exp2f;
using muse_gqa16::fast_rcpf;
using muse_gqa16::cp_async_16;
using muse_gqa16::cp_async_commit;
using muse_gqa16::cp_async_wait;
using muse_gqa16::ldsm4_swizzled;

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

__device__ __forceinline__ float clamp448(float x) {
  // Single-instruction fminf(x, 448.0f) maps NaN -> +448.0f (matching fminf(fmaxf(NaN,-448),448)=448)
  // while cvt.rn.satfinite.e4m3x2.f32 saturates all finite/±Inf values to [-448, +448] (strict 0-ULP).
  return fminf(x, 448.0f);
}

__device__ __forceinline__ uint16_t fp8x2(float a, float b) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(b), "f"(a));
  return r;
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

template <bool IsV, bool kBlock64>
__device__ __forceinline__ void issue_kv64_paged(
    uint4 smem_tile[64][16],
    const __nv_bfloat16* __restrict__ kv_ptr,
    const int32_t* __restrict__ block_table,
    int tok_base,
    int kv_head,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    int tid) {
  const int64_t head_v_off = static_cast<int64_t>(kv_head) * stride_kv_head + (IsV ? 128 : 0);
  const int r_in_16 = tid >> 4;
  const int c = tid & 15;
  const int c_phys = c ^ (r_in_16 & 7);
  if constexpr (kBlock64) {
    const int phys_blk = __ldg(block_table + (tok_base >> 6));
    const int64_t blk_base = static_cast<int64_t>(phys_blk) * stride_block + head_v_off;
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int r = (u << 4) + r_in_16;
      const int64_t row_off = blk_base + static_cast<int64_t>(r) * stride_slot;
      cp_async_16(smem_u32(&smem_tile[r][c_phys]), kv_ptr + row_off + (c << 3));
    }
  } else {
    const int lb_base = tok_base >> 4;
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int phys_blk = __ldg(block_table + lb_base + u);
      const int r = (u << 4) + r_in_16;
      const int64_t row_off = static_cast<int64_t>(phys_blk) * stride_block
                            + static_cast<int64_t>(r_in_16) * stride_slot + head_v_off;
      cp_async_16(smem_u32(&smem_tile[r][c_phys]), kv_ptr + row_off + (c << 3));
    }
  }
}

template <int Mode, bool kPdlProd, bool kBlock64>
__global__ __launch_bounds__(256, 1) void gqa16_prefill_v3_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ kv_ptr,
    const __nv_bfloat16* __restrict__ gate_ptr,
    const float* __restrict__ fp8_scale_ptr,
    void* __restrict__ out_ptr,
    const int32_t* __restrict__ block_table,
    int seqlen_q,
    int q_offset,
    float scale_log2,
    int64_t q_tok_stride,
    int64_t q_head_stride,
    int64_t out_tok_stride,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head) {
  __shared__ __align__(128) uint4 kv_smem[2][64][16];  // 32 KiB

  const int num_m_groups = seqlen_q >> 3;
  const int lin = static_cast<int>(blockIdx.x);
  const int m_grp = (num_m_groups - 1) - (lin >> 1);
  const int kv_head = lin & 1;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int group_id = lane >> 2;
  const int tid_in_grp = lane & 3;

  const int q_local_tok = (m_grp << 3) + warp_id;
  const int q_tok = q_offset + q_local_tok;
  const int nb_max = ((q_offset + (m_grp << 3)) >> 7) + 1;

  // SM121 PDL consumer barrier: wait for preceding qk_norm_rope_kvcache after independent tile math
  // and strictly before reading q_ptr or kv_ptr, then immediately launch dependents (o_proj) onto
  // wave-2 idle SMs.
  if constexpr (kPdlProd) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
  }

  {
    uint4 (*q_tiles)[16][16] = reinterpret_cast<uint4 (*)[16][16]>(kv_smem);
    const __nv_bfloat16* q_tok_base = q_ptr + static_cast<int64_t>(q_local_tok) * q_tok_stride
                                      + static_cast<int64_t>(kv_head * 16) * q_head_stride;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = (u << 5) + lane;
      const int m = idx >> 4;
      const int c = idx & 15;
      q_tiles[warp_id][m][c ^ (m & 7)] =
          *reinterpret_cast<const uint4*>(q_tok_base + static_cast<int64_t>(m) * q_head_stride + (c << 3));
    }
    __syncwarp();
  }
  uint32_t q_frag[8][4];
  {
    const uint4 (*q_tiles)[16][16] = reinterpret_cast<const uint4 (*)[16][16]>(kv_smem);
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      ldsm4_swizzled(q_frag[ks], q_tiles[warp_id], ks, lane);
    }
  }
  __syncthreads();

  float o_frag[16][4];
  #pragma unroll
  for (int dt = 0; dt < 16; ++dt) {
    o_frag[dt][0] = 0.0f; o_frag[dt][1] = 0.0f; o_frag[dt][2] = 0.0f; o_frag[dt][3] = 0.0f;
  }
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  issue_kv64_paged<false, kBlock64>(kv_smem[0], kv_ptr, block_table, (nb_max - 1) << 7,
                                    kv_head, stride_block, stride_slot, stride_kv_head, tid);
  cp_async_commit();

  for (int nb = nb_max - 1; nb >= 0; --nb) {
    const int block_tok_base = nb << 7;
    float s_frag[16][4];
    #pragma unroll
    for (int ni = 0; ni < 16; ++ni) {
      s_frag[ni][0] = 0.0f; s_frag[ni][1] = 0.0f; s_frag[ni][2] = 0.0f; s_frag[ni][3] = 0.0f;
    }

    issue_kv64_paged<false, kBlock64>(kv_smem[1], kv_ptr, block_table, block_tok_base + 64,
                                      kv_head, stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    {
      uint32_t b_cur[4], b_nxt[4];
      ldsm4_k_pair64(b_cur, kv_smem[0], 0, 0, lane);
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        #pragma unroll
        for (int sp = 0; sp < 4; ++sp) {
          const int n_sp = (sp + 1) & 3;
          const int n_ks = ks + (sp == 3 ? 1 : 0);
          if (n_ks < 8) ldsm4_k_pair64(b_nxt, kv_smem[0], n_sp, n_ks, lane);
          mma_m16n8k16_bf16(q_frag[ks], b_cur[0], b_cur[1], s_frag[sp * 2 + 0]);
          mma_m16n8k16_bf16(q_frag[ks], b_cur[2], b_cur[3], s_frag[sp * 2 + 1]);
          b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1]; b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
        }
      }
    }
    __syncthreads();

    issue_kv64_paged<true, kBlock64>(kv_smem[0], kv_ptr, block_table, block_tok_base,
                                     kv_head, stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    {
      uint32_t b_cur[4], b_nxt[4];
      ldsm4_k_pair64(b_cur, kv_smem[1], 0, 0, lane);
      #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        #pragma unroll
        for (int sp = 0; sp < 4; ++sp) {
          const int n_sp = (sp + 1) & 3;
          const int n_ks = ks + (sp == 3 ? 1 : 0);
          if (n_ks < 8) ldsm4_k_pair64(b_nxt, kv_smem[1], n_sp, n_ks, lane);
          mma_m16n8k16_bf16(q_frag[ks], b_cur[0], b_cur[1], s_frag[8 + sp * 2 + 0]);
          mma_m16n8k16_bf16(q_frag[ks], b_cur[2], b_cur[3], s_frag[8 + sp * 2 + 1]);
          b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1]; b_cur[2] = b_nxt[2]; b_cur[3] = b_nxt[3];
        }
      }
    }
    __syncthreads();

    issue_kv64_paged<true, kBlock64>(kv_smem[1], kv_ptr, block_table, block_tok_base + 64,
                                     kv_head, stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();

    float m0 = -INFINITY, m1 = -INFINITY;
    const bool is_diag = (nb == nb_max - 1);
    #pragma unroll
    for (int ni = 0; ni < 16; ++ni) {
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

    uint32_t p_frag[8][4];
    #pragma unroll
    for (int ni = 0; ni < 16; ++ni) {
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

    cp_async_wait<1>();
    __syncthreads();
    {
      uint32_t v_cur[4], v_nxt[4];
      ldsm4_v_pair64(v_cur, kv_smem[0], 0, 0, lane);
      #pragma unroll
      for (int ks_local = 0; ks_local < 4; ++ks_local) {
        #pragma unroll
        for (int dp = 0; dp < 8; ++dp) {
          const int n_dp = (dp + 1) & 7;
          const int n_ks = ks_local + (dp == 7 ? 1 : 0);
          if (n_ks < 4) ldsm4_v_pair64(v_nxt, kv_smem[0], n_dp, n_ks, lane);
          mma_m16n8k16_bf16(p_frag[ks_local], v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
          mma_m16n8k16_bf16(p_frag[ks_local], v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
          v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
        }
      }
    }
    __syncthreads();

    if (nb > 0) {
      issue_kv64_paged<false, kBlock64>(kv_smem[0], kv_ptr, block_table, (nb - 1) << 7,
                                        kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    {
      uint32_t v_cur[4], v_nxt[4];
      ldsm4_v_pair64(v_cur, kv_smem[1], 0, 0, lane);
      #pragma unroll
      for (int ks_local = 0; ks_local < 4; ++ks_local) {
        #pragma unroll
        for (int dp = 0; dp < 8; ++dp) {
          const int n_dp = (dp + 1) & 7;
          const int n_ks = ks_local + (dp == 7 ? 1 : 0);
          if (n_ks < 4) ldsm4_v_pair64(v_nxt, kv_smem[1], n_dp, n_ks, lane);
          mma_m16n8k16_bf16(p_frag[4 + ks_local], v_cur[0], v_cur[1], o_frag[dp * 2 + 0]);
          mma_m16n8k16_bf16(p_frag[4 + ks_local], v_cur[2], v_cur[3], o_frag[dp * 2 + 1]);
          v_cur[0] = v_nxt[0]; v_cur[1] = v_nxt[1]; v_cur[2] = v_nxt[2]; v_cur[3] = v_nxt[3];
        }
      }
    }
    __syncthreads();
  }

  row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 2));
  row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 1));
  row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 2));
  row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 1));
  const float inv0 = (row_sum0 == 0.0f || row_sum0 != row_sum0) ? 1.0f : fast_rcpf(row_sum0);
  const float inv1 = (row_sum1 == 0.0f || row_sum1 != row_sum1) ? 1.0f : fast_rcpf(row_sum1);

  uint32_t (*o_tiles)[16][64] = reinterpret_cast<uint32_t (*)[16][64]>(kv_smem);
  const int g_sw = (group_id & 7) << 2;
  #pragma unroll
  for (int dt = 0; dt < 16; ++dt) {
    const int c32 = ((dt << 2) ^ g_sw) + tid_in_grp;
    const __nv_bfloat162 ob0 = __floats2bfloat162_rn(mul_ftz_f32(o_frag[dt][0], inv0), mul_ftz_f32(o_frag[dt][1], inv0));
    const __nv_bfloat162 ob1 = __floats2bfloat162_rn(mul_ftz_f32(o_frag[dt][2], inv1), mul_ftz_f32(o_frag[dt][3], inv1));
    o_tiles[warp_id][group_id][c32]     = *reinterpret_cast<const uint32_t*>(&ob0);
    o_tiles[warp_id][group_id + 8][c32] = *reinterpret_cast<const uint32_t*>(&ob1);
  }
  __syncwarp();

  const int64_t tok_head_base = static_cast<int64_t>(q_local_tok) * out_tok_stride
                              + static_cast<int64_t>(kv_head * 16) * 128;
  const uint4 (*o_v4_rows)[16] = reinterpret_cast<const uint4 (*)[16]>(o_tiles[warp_id]);
  if constexpr (Mode == 0) {
    uint4* dst = reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(out_ptr) + tok_head_base);
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = (u << 5) + lane;
      const int h_idx = idx >> 4;
      const int dt = idx & 15;
      dst[idx] = o_v4_rows[h_idx][dt ^ (h_idx & 7)];
    }
  } else if constexpr (Mode == 1) {
    const uint4* g_v4 = reinterpret_cast<const uint4*>(gate_ptr + tok_head_base);
    uint4* dst = reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(out_ptr) + tok_head_base);
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = (u << 5) + lane;
      const int h_idx = idx >> 4;
      const int dt = idx & 15;
      uint4 gv;
      asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
          : "=r"(gv.x), "=r"(gv.y), "=r"(gv.z), "=r"(gv.w) : "l"(g_v4 + idx));
      const uint4 ov = o_v4_rows[h_idx][dt ^ (h_idx & 7)];
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
  } else {
    const float inv_s = full_divide(1.0f, __ldg(fp8_scale_ptr));
    const uint4* g_v4 = reinterpret_cast<const uint4*>(gate_ptr + tok_head_base);
    uint2* dst = reinterpret_cast<uint2*>(static_cast<uint8_t*>(out_ptr) + tok_head_base);
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = (u << 5) + lane;
      const int h_idx = idx >> 4;
      const int dt = idx & 15;
      uint4 gv;
      asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
          : "=r"(gv.x), "=r"(gv.y), "=r"(gv.z), "=r"(gv.w) : "l"(g_v4 + idx));
      const uint4 ov = o_v4_rows[h_idx][dt ^ (h_idx & 7)];
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
  }
}

template <int Mode, bool kBlock64>
static inline cudaError_t launch_prefill_mode(
    bool pdl_prod, dim3 grid, dim3 block, cudaStream_t st,
    const __nv_bfloat16* q, const __nv_bfloat16* kv, const __nv_bfloat16* gate,
    const float* fp8_scale, void* out, const int32_t* bt,
    int seqlen_q, int q_offset, float scale_log2,
    int64_t q_tok_stride, int64_t q_head_stride, int64_t out_tok_stride,
    int64_t stride_block, int64_t stride_slot, int64_t stride_kv_head) {
  if (pdl_prod) {
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
    return cudaLaunchKernelEx(
        &cfg, gqa16_prefill_v3_kernel<Mode, true, kBlock64>,
        q, kv, gate, fp8_scale, out, bt, seqlen_q, q_offset, scale_log2,
        q_tok_stride, q_head_stride, out_tok_stride, stride_block, stride_slot, stride_kv_head);
  }
  gqa16_prefill_v3_kernel<Mode, false, kBlock64><<<grid, block, 0, st>>>(
      q, kv, gate, fp8_scale, out, bt, seqlen_q, q_offset, scale_log2,
      q_tok_stride, q_head_stride, out_tok_stride, stride_block, stride_slot, stride_kv_head);
  return cudaGetLastError();
}

}  // namespace muse_gqa16_pf


// =================================================================================================
// Clean-room Wave 58 WG2 Shared-KV Tree Verify Kernel (256 threads = 2 warpgroups x 4 warps/token)
// - At M=16: launches (1, 2, 8) = 16 CTAs x 8 warps = 128 warps across the exact 16 free SMs
//   alongside 32-CTA output_gate_proj (48 SMs total = 100% single-wave co-residency, 0 Wave-2 spill!)
// - Preserves the exact 4-warp-per-token QK^T (16 MMAs/warp) + PV (16 MMAs/warp) decomposition of
//   gqa16_fa2_block128_pipe_kernel while both warpgroups execute QK^T, softmax, and PV IN PARALLEL
//   from a single shared 256-thread cp.async KV load (2x fewer KV DRAM/L2 reads, ~156 regs, 0 spills,
//   58.1 KiB dynamic SMEM -> 64 KiB L1 cache carveout on SM121!).
// =================================================================================================
namespace muse_gqa16_wg2 {

using muse_gqa16::smem_u32;
using muse_gqa16::max_ftz_f32;
using muse_gqa16::add_ftz_f32;
using muse_gqa16::sub_ftz_f32;
using muse_gqa16::mul_ftz_f32;
using muse_gqa16::fma_rn_ftz_f32;
using muse_gqa16::fast_exp2f;
using muse_gqa16::fast_rcpf;
using muse_gqa16::cp_async_16;
using muse_gqa16::cp_async_commit;
using muse_gqa16::cp_async_wait;
using muse_gqa16::ldsm4_swizzled;
using muse_gqa16::ldsm2_k_swizzled;
using muse_gqa16::ldsm2_v_trans_swizzled;
using muse_gqa16::mma_m16n8k16_bf16;

struct Wg2RowInfo {
  int sk;
  int ks;
  int nnb;
  int gnb;
  int lrb;
  int lps;
  uint32_t lrp;
  uint8_t active;
  uint8_t empty;
};

struct __align__(128) Wg2Pipe128Storage {
  union {
    uint4 kv[5][32][16];           // 40,960 B (K[0..3] + V[0]; then kv[0..2] hold V[1..3], kv[4] holds V[0])
    struct {
      uint4 kv_keep[3][32][16];    // 24,576 B (kv[0..2])
      float4 p_f32_wg0[512];       // 8,192 B (aliases dead kv[3] during softmax!)
    };
  };
  float4 p_f32_wg1[512];           // 8,192 B
  uint4 p[2][16][16];              // 8,192 B
  float warp_max[2][4][16];        // 512 B
  float warp_sum[2][16];           // 128 B
  Wg2RowInfo rows[2];
  int q0;
  int batch_nb_max;
  int batch_nb_min;
  int batch_k_start;
  int batch_max_sk;
  int shared_end_block;
  uint8_t any_active;
  uint8_t any_remap;
};                                 // Total = 58,112 B (<= 62 KiB -> 64 KiB L1 carveout on SM121!)

template <bool IsV>
__device__ __forceinline__ void issue_kv_tile_async_256(
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
    int tid,
    int leaf_remap_base = -1,
    int leaf_p_start = 0,
    uint32_t leaf_remap_packed = 0u) {
  const bool is_bs64_aligned = (block_size == 64) && ((tok_base & 31) == 0);
  int phys_blk0 = 0, slot_base0 = 0;
  const bool half0_any = (tok_base < seqlen_k) && (tok_base + 16 > k_start);
  if (half0_any) {
    const int lb0 = (block_size == 16) ? (tok_base >> 4)
                  : ((block_size == 64) ? (tok_base >> 6) : (tok_base / block_size));
    slot_base0 = (block_size == 16) ? 0
               : ((block_size == 64) ? (tok_base & 63) : (tok_base - lb0 * block_size));
    phys_blk0 = __ldg(block_table + lb0);
  }
  int phys_blk1 = phys_blk0, slot_base1 = slot_base0 + 16;
  const int tok_hi = tok_base + 16;
  const bool half1_any = (tok_hi < seqlen_k) && (tok_hi + 16 > k_start);
  if (half1_any && !is_bs64_aligned) {
    const int lb1 = (block_size == 16) ? (tok_hi >> 4)
                  : ((block_size == 64) ? (tok_hi >> 6) : (tok_hi / block_size));
    slot_base1 = (block_size == 16) ? 0
               : ((block_size == 64) ? (tok_hi & 63) : (tok_hi - lb1 * block_size));
    phys_blk1 = __ldg(block_table + lb1);
  }

  const int64_t head_v_off = static_cast<int64_t>(kv_head) * stride_kv_head + (IsV ? 128 : 0);
  const bool has_remap = (leaf_remap_base >= 0) && (tok_base + 32 > leaf_remap_base);
  const bool full_valid = !has_remap && (tok_base >= k_start) && (tok_base + 32 <= seqlen_k) &&
                          ((block_size == 16) || is_bs64_aligned ||
                           (slot_base0 + 16 <= block_size && slot_base1 + 16 <= block_size));

  const int r0 = tid >> 4;        // 0..15
  const int c = tid & 15;         // 0..15
  const int c_phys = c ^ (r0 & 7);

  if (full_valid) {
    const int64_t base0 = static_cast<int64_t>(phys_blk0) * stride_block +
                          static_cast<int64_t>(slot_base0) * stride_slot + head_v_off;
    cp_async_16(smem_u32(&smem_tile[r0][c_phys]), kv_ptr + base0 + static_cast<int64_t>(r0) * stride_slot + (c << 3));
    if (is_bs64_aligned) {
      cp_async_16(smem_u32(&smem_tile[r0 + 16][c_phys]), kv_ptr + base0 + static_cast<int64_t>(r0 + 16) * stride_slot + (c << 3));
    } else {
      const int64_t base1 = static_cast<int64_t>(phys_blk1) * stride_block +
                            static_cast<int64_t>(slot_base1) * stride_slot + head_v_off;
      cp_async_16(smem_u32(&smem_tile[r0 + 16][c_phys]), kv_ptr + base1 + static_cast<int64_t>(r0) * stride_slot + (c << 3));
    }
  } else {
    #pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int r = (u << 4) + r0;
      const int tok = tok_base + r;
      if (tok >= k_start && tok < seqlen_k) {
        const int tok_eff = (leaf_remap_base >= 0 && tok >= leaf_remap_base)
            ? (leaf_p_start + static_cast<int>((leaf_remap_packed >> ((tok - leaf_remap_base) * 5)) & 0x1Fu))
            : tok;
        const int lb = (block_size == 16) ? (tok_eff >> 4)
                     : ((block_size == 64) ? (tok_eff >> 6) : (tok_eff / block_size));
        const int slot = (block_size == 16) ? (tok_eff & 15)
                       : ((block_size == 64) ? (tok_eff & 63) : (tok_eff - lb * block_size));
        const int phys_blk = __ldg(block_table + lb);
        const int64_t row_off = static_cast<int64_t>(phys_blk) * stride_block +
                                static_cast<int64_t>(slot) * stride_slot + head_v_off;
        cp_async_16(smem_u32(&smem_tile[r][c_phys]), kv_ptr + row_off + (c << 3));
      } else {
        smem_tile[r][c_phys] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  }
}

__device__ __forceinline__ void wg_bar_sync(int wg) {
  // Named barrier 1 for wg=0 (128 threads), barrier 2 for wg=1 (128 threads)
  if (wg == 0) {
    asm volatile("bar.sync 1, 128;" ::: "memory");
  } else {
    asm volatile("bar.sync 2, 128;" ::: "memory");
  }
}

__device__ __forceinline__ void wg2_softmax_pre_pv(
    Wg2Pipe128Storage& smem,
    int wg,
    int sub_w,
    int lane,
    int group_id,
    int tid_in_grp,
    bool do_wg,
    float (&s_frag)[4][4],
    float (&o_frag)[4][4],
    float& row_max0,
    float& row_max1,
    float& sc0_out,
    float& sc1_out,
    bool is_first,
    int block_tok_base,
    int k_start,
    int seqlen_k,
    float scale_log2) {
  sc0_out = 1.0f;
  sc1_out = 1.0f;
  if (!do_wg) return;

  float m0 = -INFINITY, m1 = -INFINITY;
  const bool need_mask = (block_tok_base < k_start) || (block_tok_base + 128 > seqlen_k);
  #pragma unroll
  for (int ch = 0; ch < 4; ++ch) {
    if (need_mask) {
      const int col0 = block_tok_base + (ch << 5) + (sub_w << 3) + (tid_in_grp << 1);
      const int col1 = col0 + 1;
      if (col0 < k_start || col0 >= seqlen_k) { s_frag[ch][0] = -INFINITY; s_frag[ch][2] = -INFINITY; }
      if (col1 < k_start || col1 >= seqlen_k) { s_frag[ch][1] = -INFINITY; s_frag[ch][3] = -INFINITY; }
    }
    m0 = max_ftz_f32(m0, max_ftz_f32(s_frag[ch][0], s_frag[ch][1]));
    m1 = max_ftz_f32(m1, max_ftz_f32(s_frag[ch][2], s_frag[ch][3]));
  }
  m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
  m0 = max_ftz_f32(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
  m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
  m1 = max_ftz_f32(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
  if (tid_in_grp == 0) {
    smem.warp_max[wg][sub_w][group_id]     = m0;
    smem.warp_max[wg][sub_w][group_id + 8] = m1;
  }
  wg_bar_sync(wg);

  const float tile_max0 = max_ftz_f32(max_ftz_f32(smem.warp_max[wg][0][group_id], smem.warp_max[wg][1][group_id]),
                                      max_ftz_f32(smem.warp_max[wg][2][group_id], smem.warp_max[wg][3][group_id]));
  const float tile_max1 = max_ftz_f32(max_ftz_f32(smem.warp_max[wg][0][group_id + 8], smem.warp_max[wg][1][group_id + 8]),
                                      max_ftz_f32(smem.warp_max[wg][2][group_id + 8], smem.warp_max[wg][3][group_id + 8]));
  if (is_first) {
    row_max0 = tile_max0;
    row_max1 = tile_max1;
  } else {
    const float prev0 = row_max0, prev1 = row_max1;
    row_max0 = max_ftz_f32(prev0, tile_max0);
    row_max1 = max_ftz_f32(prev1, tile_max1);
    const float cur0 = (row_max0 == -INFINITY) ? 0.0f : row_max0;
    const float cur1 = (row_max1 == -INFINITY) ? 0.0f : row_max1;
    sc0_out = fast_exp2f(mul_ftz_f32(scale_log2, sub_ftz_f32(prev0, cur0)));
    sc1_out = fast_exp2f(mul_ftz_f32(scale_log2, sub_ftz_f32(prev1, cur1)));
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      o_frag[dt][0] = mul_ftz_f32(o_frag[dt][0], sc0_out);
      o_frag[dt][1] = mul_ftz_f32(o_frag[dt][1], sc0_out);
      o_frag[dt][2] = mul_ftz_f32(o_frag[dt][2], sc1_out);
      o_frag[dt][3] = mul_ftz_f32(o_frag[dt][3], sc1_out);
    }
  }

  float4* const pf32 = (wg == 0) ? smem.p_f32_wg0 : smem.p_f32_wg1;
  const float max_sc0 = (row_max0 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max0);
  const float max_sc1 = (row_max1 == -INFINITY) ? 0.0f : mul_ftz_f32(scale_log2, row_max1);
  #pragma unroll
  for (int ch = 0; ch < 4; ++ch) {
    const float p0 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][0], scale_log2), max_sc0));
    const float p1 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][1], scale_log2), max_sc0));
    const float p2 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][2], scale_log2), max_sc1));
    const float p3 = fast_exp2f(sub_ftz_f32(mul_ftz_f32(s_frag[ch][3], scale_log2), max_sc1));
    const int ni = (ch << 2) + sub_w;
    pf32[ni * 32 + lane] = make_float4(p0, p1, p2, p3);
    const __nv_bfloat162 pb0 = __floats2bfloat162_rn(p0, p1);
    const __nv_bfloat162 pb1 = __floats2bfloat162_rn(p2, p3);
    const int c_phys_p = ni ^ group_id;
    reinterpret_cast<uint32_t*>(&smem.p[wg][group_id][c_phys_p])[tid_in_grp] =
        *reinterpret_cast<const uint32_t*>(&pb0);
    reinterpret_cast<uint32_t*>(&smem.p[wg][group_id + 8][c_phys_p])[tid_in_grp] =
        *reinterpret_cast<const uint32_t*>(&pb1);
  }
}

__device__ __forceinline__ void wg2_row_sum_and_pv_lo(
    Wg2Pipe128Storage& smem,
    int wg,
    int sub_w,
    int lane,
    bool do_wg,
    bool is_first,
    float sc0,
    float sc1,
    float& row_sum0,
    float& row_sum1,
    float (&o_frag)[4][4]) {
  if (!do_wg) return;
  if (sub_w == 0) {
    const float4* const pf32 = (wg == 0) ? smem.p_f32_wg0 : smem.p_f32_wg1;
    const float4 pv0 = pf32[0 * 32 + lane];
    if (is_first) {
      row_sum0 = add_ftz_f32(pv0.x, pv0.y);
      row_sum1 = add_ftz_f32(pv0.z, pv0.w);
    } else {
      row_sum0 = add_ftz_f32(fma_rn_ftz_f32(row_sum0, sc0, pv0.x), pv0.y);
      row_sum1 = add_ftz_f32(fma_rn_ftz_f32(row_sum1, sc1, pv0.z), pv0.w);
    }
    #pragma unroll
    for (int ni = 1; ni < 16; ++ni) {
      const float4 pv = pf32[ni * 32 + lane];
      row_sum0 = add_ftz_f32(add_ftz_f32(row_sum0, pv.x), pv.y);
      row_sum1 = add_ftz_f32(add_ftz_f32(row_sum1, pv.z), pv.w);
    }
  }
  #pragma unroll
  for (int ks_p = 0; ks_p < 2; ++ks_p) {
    uint32_t p_frag[4];
    ldsm4_swizzled(p_frag, smem.p[wg], ks_p, lane);
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      uint32_t v_frag[2];
      ldsm2_v_trans_swizzled(v_frag, smem.kv[4], sub_w, dt, ks_p, lane);
      mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
    }
  }
  #pragma unroll
  for (int ks_p = 0; ks_p < 2; ++ks_p) {
    uint32_t p_frag[4];
    ldsm4_swizzled(p_frag, smem.p[wg], 2 + ks_p, lane);
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      uint32_t v_frag[2];
      ldsm2_v_trans_swizzled(v_frag, smem.kv[0], sub_w, dt, ks_p, lane);
      mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
    }
  }
}

__device__ __forceinline__ void wg2_pv_hi(
    Wg2Pipe128Storage& smem,
    int wg,
    int sub_w,
    int lane,
    bool do_wg,
    float (&o_frag)[4][4]) {
  if (!do_wg) return;
  #pragma unroll
  for (int ks_p = 0; ks_p < 2; ++ks_p) {
    uint32_t p_frag[4];
    ldsm4_swizzled(p_frag, smem.p[wg], 4 + ks_p, lane);
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      uint32_t v_frag[2];
      ldsm2_v_trans_swizzled(v_frag, smem.kv[1], sub_w, dt, ks_p, lane);
      mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
    }
  }
  #pragma unroll
  for (int ks_p = 0; ks_p < 2; ++ks_p) {
    uint32_t p_frag[4];
    ldsm4_swizzled(p_frag, smem.p[wg], 6 + ks_p, lane);
    #pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      uint32_t v_frag[2];
      ldsm2_v_trans_swizzled(v_frag, smem.kv[2], sub_w, dt, ks_p, lane);
      mma_m16n8k16_bf16(p_frag, v_frag, o_frag[dt]);
    }
  }
}

template <bool kPdl>
__global__ __launch_bounds__(256, 1) void gqa16_verify_wg2_kernel(
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
    int64_t stride_kv_head,
    const int32_t* __restrict__ cu_seqlens_q_ptr,
    int64_t q_tok_stride,
    int64_t out_tok_stride,
    const int32_t* __restrict__ leaf_flag_ptr) {
  extern __shared__ __align__(128) unsigned char dyn_smem[];
  auto& smem = *reinterpret_cast<Wg2Pipe128Storage*>(dyn_smem);

  const int kv_head    = static_cast<int>(blockIdx.y);
  const int q_grp      = (static_cast<int>(gridDim.z) - 1) - static_cast<int>(blockIdx.z);
  const int q_base     = q_grp << 1;
  const int tid        = static_cast<int>(threadIdx.x);
  const int warp_id    = tid >> 5;       // 0..7
  const int wg         = warp_id >> 2;   // 0..1 (which of the 2 query tokens in CTA)
  const int sub_w      = warp_id & 3;    // 0..3 (which warp within the 4-warp group)
  const int wg_tid     = tid & 127;      // 0..127
  const int lane       = tid & 31;
  const int group_id   = lane >> 2;
  const int tid_in_grp = lane & 3;

  const int q0 = cu_seqlens_q_ptr[0];
  const int actual_m = cu_seqlens_q_ptr[1] - q0;
  if (q_base >= actual_m) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  const int raw_seqlen_k = seqused_k_ptr[0];
  const int lf0 = (leaf_flag_ptr != nullptr) ? leaf_flag_ptr[0] : 0;

  if (tid < 2) {
    const int q_idx = q_base + tid;
    Wg2RowInfo r = {0, 0, 0, 0, -1, 0, 0u, 0u, 0u};
    if (q_idx < actual_m) {
      int sk = raw_seqlen_k;
      int lrb = -1, lps = 0;
      uint32_t lrp = 0u;
      if (leaf_flag_ptr != nullptr && lf0 < 0 && actual_m <= 32 && sk >= actual_m) {
        const int base = 1 + q_idx * 3;
        const int seq_sub = leaf_flag_ptr[base];
        const int base_sub = leaf_flag_ptr[base + 1];
        if (base_sub >= 0) {
          lps = sk - actual_m;
          lrb = sk - base_sub;
          lrp = static_cast<uint32_t>(leaf_flag_ptr[base + 2]);
        }
        sk -= seq_sub;
      } else if (q_idx == 7 && actual_m == 8 && leaf_flag_ptr != nullptr) {
        if ((lf0 == 1 || lf0 == 2) && sk >= 8) {
          lps = sk - 8; lrb = lps + lf0; lrp = 7u; sk = lps + lf0 + 1;
        } else {
          sk -= (actual_m - 1 - q_idx);
        }
      } else if (q_idx == 6 && actual_m == 7 && leaf_flag_ptr != nullptr && lf0 > 0 && sk >= 7) {
        lps = sk - 7; lrb = lps + 1; lrp = 6u; sk -= 5;
      } else {
        sk -= (actual_m - 1 - q_idx);
      }
      if (sk < 0) sk = 0;
      const int ks = (window_left >= 0) ? ((sk - 1 - window_left > 0) ? (sk - 1 - window_left) : 0) : 0;
      const int nnb = (sk > 0) ? ((sk + 127) >> 7) : 0;
      const int gnb = (window_left >= 0 && sk > 0) ? (ks >> 7) : 0;
      if (gnb >= nnb) {
        r.empty = 1u;
      } else {
        r.active = 1u; r.sk = sk; r.ks = ks; r.nnb = nnb; r.gnb = gnb;
        r.lrb = lrb; r.lps = lps; r.lrp = lrp;
      }
    }
    smem.rows[tid] = r;
  }
  __syncthreads();

  if (tid == 0) {
    smem.q0 = q0;
    int b_nb_max = 0, b_nb_min = 0x7fffffff, b_ks = 0x7fffffff, b_max_sk = 0;
    int min_remap_tok = raw_seqlen_k;
    uint8_t any_act = 0u, any_rem = 0u;
    #pragma unroll
    for (int qb = 0; qb < 2; ++qb) {
      const Wg2RowInfo& r = smem.rows[qb];
      if (!r.active) continue;
      any_act = 1u;
      if (r.nnb > b_nb_max) b_nb_max = r.nnb;
      if (r.gnb < b_nb_min) b_nb_min = r.gnb;
      if (r.ks < b_ks) b_ks = r.ks;
      if (r.sk > b_max_sk) b_max_sk = r.sk;
      if (r.lrb >= 0) {
        any_rem = 1u;
        if (r.lrb < min_remap_tok) min_remap_tok = r.lrb;
      }
    }
    smem.any_active = any_act;
    smem.any_remap = any_rem;
    smem.batch_nb_max = b_nb_max;
    smem.batch_nb_min = any_act ? b_nb_min : 0;
    smem.batch_k_start = any_act ? b_ks : 0;
    smem.batch_max_sk = b_max_sk;
    smem.shared_end_block = any_rem ? ((min_remap_tok > 0) ? (min_remap_tok >> 7) : 0) : b_nb_max;
  }

  if constexpr (kPdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  __syncthreads();

  if (smem.rows[wg].empty) {
    const int q_local = wg_tid >> 3;
    const int d_base = (wg_tid & 7) << 4;
    __nv_bfloat16* dst = out_ptr + static_cast<int64_t>(smem.q0 + q_base + wg) * out_tok_stride +
                         static_cast<int64_t>(kv_head * 16 + q_local) * out_head_stride + d_base;
    *reinterpret_cast<uint4*>(dst)     = make_uint4(0u, 0u, 0u, 0u);
    *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0u, 0u, 0u, 0u);
  }

  if (!smem.any_active) {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  // Load each warpgroup's Q[16, 128] into q_frag[8][4] in registers using smem.p[wg] (4 KiB)
  uint32_t q_frag[8][4];
  if (smem.rows[wg].active) {
    const __nv_bfloat16* q_group = q_ptr + static_cast<int64_t>(smem.q0 + q_base + wg) * q_tok_stride +
                                   static_cast<int64_t>(kv_head * 16) * q_head_stride;
    #pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int idx = (u << 7) + wg_tid;
      const int m = idx >> 4;
      const int c = idx & 15;
      smem.p[wg][m][c ^ (m & 7)] =
          *reinterpret_cast<const uint4*>(q_group + static_cast<int64_t>(m) * q_head_stride + (c << 3));
    }
    wg_bar_sync(wg);
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      ldsm4_swizzled(q_frag[ks], smem.p[wg], ks, lane);
    }
  }
  __syncthreads();

  float o_frag[4][4];
  #pragma unroll
  for (int dt = 0; dt < 4; ++dt) {
    o_frag[dt][0] = 0.0f; o_frag[dt][1] = 0.0f; o_frag[dt][2] = 0.0f; o_frag[dt][3] = 0.0f;
  }
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;
  bool my_is_first = true;

  const int shared_end_block = smem.shared_end_block;
  const int batch_nb_min = smem.batch_nb_min;

  // Phase 1: Only when at least one of the 2 rows in CTA has tree remap (lrb >= 0) — typically just 1 tail block!
  if (smem.any_remap) {
    #pragma unroll 1
    for (int qb = 0; qb < 2; ++qb) {
      const Wg2RowInfo r = smem.rows[qb];
      if (!r.active) continue;
      const int tail_lo = (r.gnb > shared_end_block) ? r.gnb : shared_end_block;
      for (int nb = r.nnb - 1; nb >= tail_lo; --nb) {
        const int block_tok_base = nb << 7;
        const bool do_wg = (wg == qb);
        float s_frag[4][4] = {0.f};

        #pragma unroll
        for (int ch = 0; ch < 4; ++ch) {
          issue_kv_tile_async_256<false>(
              smem.kv[ch], kv_ptr, block_table, block_tok_base + (ch << 5),
              r.ks, r.sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid,
              r.lrb, r.lps, r.lrp);
          cp_async_commit();
        }
        issue_kv_tile_async_256<true>(
            smem.kv[4], kv_ptr, block_table, block_tok_base,
            r.ks, r.sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid,
            r.lrb, r.lps, r.lrp);
        cp_async_commit();

        cp_async_wait<4>();
        __syncthreads();
        if (do_wg) {
          #pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            uint32_t b_frag[2];
            ldsm2_k_swizzled(b_frag, smem.kv[0], sub_w, ks, lane);
            mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[0]);
          }
        }
        __syncthreads();
        issue_kv_tile_async_256<true>(
            smem.kv[0], kv_ptr, block_table, block_tok_base + 32,
            r.ks, r.sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid,
            r.lrb, r.lps, r.lrp);
        cp_async_commit();

        cp_async_wait<4>();
        __syncthreads();
        if (do_wg) {
          #pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            uint32_t b_frag[2];
            ldsm2_k_swizzled(b_frag, smem.kv[1], sub_w, ks, lane);
            mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[1]);
          }
        }
        __syncthreads();
        issue_kv_tile_async_256<true>(
            smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
            r.ks, r.sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid,
            r.lrb, r.lps, r.lrp);
        cp_async_commit();

        cp_async_wait<4>();
        __syncthreads();
        if (do_wg) {
          #pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            uint32_t b_frag[2];
            ldsm2_k_swizzled(b_frag, smem.kv[2], sub_w, ks, lane);
            mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[2]);
          }
        }
        __syncthreads();
        issue_kv_tile_async_256<true>(
            smem.kv[2], kv_ptr, block_table, block_tok_base + 96,
            r.ks, r.sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid,
            r.lrb, r.lps, r.lrp);
        cp_async_commit();

        cp_async_wait<4>();
        __syncthreads();
        if (do_wg) {
          #pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            uint32_t b_frag[2];
            ldsm2_k_swizzled(b_frag, smem.kv[3], sub_w, ks, lane);
            mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[3]);
          }
        }
        __syncthreads();

        float sc0 = 1.0f, sc1 = 1.0f;
        wg2_softmax_pre_pv(
            smem, wg, sub_w, lane, group_id, tid_in_grp, do_wg, s_frag, o_frag,
            row_max0, row_max1, sc0, sc1, my_is_first, block_tok_base, r.ks, r.sk, scale_log2);

        cp_async_wait<2>();
        __syncthreads();
        wg2_row_sum_and_pv_lo(smem, wg, sub_w, lane, do_wg, my_is_first, sc0, sc1, row_sum0, row_sum1, o_frag);

        cp_async_wait<0>();
        __syncthreads();
        wg2_pv_hi(smem, wg, sub_w, lane, do_wg, o_frag);
        if (do_wg) my_is_first = false;
        __syncthreads();
      }
    }
  }

  // Phase 2: Cross-block pipelined Shared-KV mainloop — K[0] & K[3] of (nb-1) stream in during PV_hi(nb),
  // and BOTH warpgroups execute QK^T, softmax, and PV IN PARALLEL from 1 shared KV load!
  const int load_ks = smem.batch_k_start;
  const int load_sk = smem.batch_max_sk;
  if (shared_end_block > batch_nb_min) {
    const int first_base = (shared_end_block - 1) << 7;
    issue_kv_tile_async_256<false>(
        smem.kv[0], kv_ptr, block_table, first_base,
        load_ks, load_sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();
    issue_kv_tile_async_256<false>(
        smem.kv[3], kv_ptr, block_table, first_base + 96,
        load_ks, load_sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
    cp_async_commit();

    for (int nb = shared_end_block - 1; nb >= batch_nb_min; --nb) {
      const int block_tok_base = nb << 7;
      const Wg2RowInfo mr = smem.rows[wg];
      const bool do_wg = mr.active && (nb < mr.nnb) && (nb >= mr.gnb);
      float s_frag[4][4] = {0.f};

      issue_kv_tile_async_256<false>(
          smem.kv[1], kv_ptr, block_table, block_tok_base + 32,
          load_ks, load_sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
      issue_kv_tile_async_256<false>(
          smem.kv[2], kv_ptr, block_table, block_tok_base + 64,
          load_ks, load_sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();
      issue_kv_tile_async_256<true>(
          smem.kv[4], kv_ptr, block_table, block_tok_base,
          load_ks, load_sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();

      cp_async_wait<4>();
      __syncthreads();
      if (do_wg) {
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          uint32_t b_frag[2];
          ldsm2_k_swizzled(b_frag, smem.kv[0], sub_w, ks, lane);
          mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[0]);
        }
      }
      __syncthreads();
      issue_kv_tile_async_256<true>(
          smem.kv[0], kv_ptr, block_table, block_tok_base + 32,
          load_ks, load_sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();

      cp_async_wait<3>();
      __syncthreads();
      if (do_wg) {
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          uint32_t b_frag[2];
          ldsm2_k_swizzled(b_frag, smem.kv[1], sub_w, ks, lane);
          mma_m16n8k16_bf16(q_frag[ks], b_frag, s_frag[1]);
        }
      }
      __syncthreads();
      issue_kv_tile_async_256<true>(
          smem.kv[1], kv_ptr, block_table, block_tok_base + 64,
          load_ks, load_sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();

      cp_async_wait<3>();
      __syncthreads();
      if (do_wg) {
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          uint32_t b_frag2[2], b_frag3[2];
          ldsm2_k_swizzled(b_frag2, smem.kv[2], sub_w, ks, lane);
          ldsm2_k_swizzled(b_frag3, smem.kv[3], sub_w, ks, lane);
          mma_m16n8k16_bf16(q_frag[ks], b_frag2, s_frag[2]);
          mma_m16n8k16_bf16(q_frag[ks], b_frag3, s_frag[3]);
        }
      }
      __syncthreads();
      issue_kv_tile_async_256<true>(
          smem.kv[2], kv_ptr, block_table, block_tok_base + 96,
          load_ks, load_sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
      cp_async_commit();

      float sc0 = 1.0f, sc1 = 1.0f;
      wg2_softmax_pre_pv(
          smem, wg, sub_w, lane, group_id, tid_in_grp, do_wg, s_frag, o_frag,
          row_max0, row_max1, sc0, sc1, my_is_first, block_tok_base, mr.ks, mr.sk, scale_log2);

      cp_async_wait<2>();
      __syncthreads();
      wg2_row_sum_and_pv_lo(smem, wg, sub_w, lane, do_wg, my_is_first, sc0, sc1, row_sum0, row_sum1, o_frag);
      __syncthreads();

      if (nb > batch_nb_min) {
        const int next_base = (nb - 1) << 7;
        issue_kv_tile_async_256<false>(
            smem.kv[0], kv_ptr, block_table, next_base,
            load_ks, load_sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
        cp_async_commit();
        issue_kv_tile_async_256<false>(
            smem.kv[3], kv_ptr, block_table, next_base + 96,
            load_ks, load_sk, block_size, kv_head, stride_block, stride_slot, stride_kv_head, tid);
        cp_async_commit();
        cp_async_wait<2>();
      } else {
        if constexpr (kPdl) {
          asm volatile("griddepcontrol.launch_dependents;");
        }
        cp_async_wait<0>();
      }
      __syncthreads();
      wg2_pv_hi(smem, wg, sub_w, lane, do_wg, o_frag);
      if (do_wg) my_is_first = false;
      __syncthreads();
    }
  } else {
    if constexpr (kPdl) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
  }

  if (!smem.rows[wg].active) return;

  if (sub_w == 0) {
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 2));
    row_sum0 = add_ftz_f32(row_sum0, __shfl_xor_sync(0xffffffffu, row_sum0, 1));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 2));
    row_sum1 = add_ftz_f32(row_sum1, __shfl_xor_sync(0xffffffffu, row_sum1, 1));
    const bool empty0 = (row_sum0 == 0.0f || row_sum0 != row_sum0 || row_max0 == -INFINITY);
    const bool empty1 = (row_sum1 == 0.0f || row_sum1 != row_sum1 || row_max1 == -INFINITY);
    const float inv0 = empty0 ? 1.0f : fast_rcpf(row_sum0);
    const float inv1 = empty1 ? 1.0f : fast_rcpf(row_sum1);
    if (tid_in_grp == 0) {
      smem.warp_sum[wg][group_id]     = inv0;
      smem.warp_sum[wg][group_id + 8] = inv1;
    }
  }
  wg_bar_sync(wg);

  const float inv0 = smem.warp_sum[wg][group_id];
  const float inv1 = smem.warp_sum[wg][group_id + 8];
  __nv_bfloat16* const out_row = out_ptr + static_cast<int64_t>(smem.q0 + q_base + wg) * out_tok_stride;
  #pragma unroll
  for (int dt = 0; dt < 4; ++dt) {
    const int d_col = (sub_w << 5) + (dt << 3) + (tid_in_grp << 1);
    const __nv_bfloat162 ob0 = __floats2bfloat162_rn(mul_ftz_f32(o_frag[dt][0], inv0), mul_ftz_f32(o_frag[dt][1], inv0));
    const __nv_bfloat162 ob1 = __floats2bfloat162_rn(mul_ftz_f32(o_frag[dt][2], inv1), mul_ftz_f32(o_frag[dt][3], inv1));
    *reinterpret_cast<uint32_t*>(out_row + static_cast<int64_t>(kv_head * 16 + group_id) * out_head_stride + d_col) =
        *reinterpret_cast<const uint32_t*>(&ob0);
    *reinterpret_cast<uint32_t*>(out_row + static_cast<int64_t>(kv_head * 16 + group_id + 8) * out_head_stride + d_col) =
        *reinterpret_cast<const uint32_t*>(&ob1);
  }
}

static void gqa16_wg2_set_attrs_once() {
  static bool done = false;
  if (done) return;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  cudaThreadExchangeStreamCaptureMode(&mode);
  constexpr int kWg2Smem = static_cast<int>(sizeof(Wg2Pipe128Storage));
  cudaFuncSetAttribute(gqa16_verify_wg2_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kWg2Smem);
  cudaFuncSetAttribute(gqa16_verify_wg2_kernel<false>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
  cudaFuncSetAttribute(gqa16_verify_wg2_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kWg2Smem);
  cudaFuncSetAttribute(gqa16_verify_wg2_kernel<true>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
  cudaThreadExchangeStreamCaptureMode(&mode);
  done = true;
}

static int gqa16_wg2_min_q() {
  static const int min_q = []() -> int {
    const char* env = std::getenv("MUSE_GQA16_WG2");
    if (!env || !*env) return 13; // default: enable WG2 whenever max_q > 12 (>24 CTAs would spill alongside 24-SM gate!)
    long v = std::strtol(env, nullptr, 10);
    if (v <= 0) return 999;       // disabled
    if (v == 1) return 13;        // default threshold max_q >= 13
    return static_cast<int>(v);   // explicit min max_q (e.g. MUSE_GQA16_WG2=2 forces WG2 on all M>=2)
  }();
  return min_q;
}

}  // namespace muse_gqa16_wg2

extern "C" int muse_gqa16_prefill(
    const void* q_ptr, const void* kv_ptr, const void* gate_ptr, const void* fp8_scale_ptr,
    void* out_ptr, const void* block_table_ptr,
    int seqlen_q, int q_offset, int block_size, int mode, int scale_mode, int pdl_prod,
    float softmax_scale,
    int64_t q_tok_stride, int64_t q_head_stride, int64_t out_tok_stride,
    int64_t stride_block, int64_t stride_slot, int64_t stride_kv_head,
    void* cuda_stream) {
  if (!q_ptr || !kv_ptr || !out_ptr || !block_table_ptr ||
      seqlen_q <= 0 || (seqlen_q & 7) != 0 || q_offset < 0 || (q_offset & 7) != 0 ||
      (block_size != 16 && block_size != 64) || mode < 0 || mode > 2) {
    return 1;
  }
  const bool use_scale_mode3 = ((scale_mode & 3) == 3);
  const float scale_log2 =
      use_scale_mode3
          ? (softmax_scale * muse_gqa16::kLog2e)
          : static_cast<float>(static_cast<double>(softmax_scale) * 1.44269504088896340736);
  const dim3 grid(static_cast<unsigned>((seqlen_q >> 3) * 2), 1u, 1u);
  const dim3 block(256u, 1u, 1u);
  const cudaStream_t st = static_cast<cudaStream_t>(cuda_stream);
  const auto* q = static_cast<const __nv_bfloat16*>(q_ptr);
  const auto* kv = static_cast<const __nv_bfloat16*>(kv_ptr);
  const auto* gate = static_cast<const __nv_bfloat16*>(gate_ptr);
  const auto* fp8_scale = static_cast<const float*>(fp8_scale_ptr);
  const auto* bt = static_cast<const int32_t*>(block_table_ptr);
  cudaError_t err;
  if (block_size == 64) {
    if (mode == 0) {
      err = muse_gqa16_pf::launch_prefill_mode<0, true>(
          pdl_prod != 0, grid, block, st, q, kv, gate, fp8_scale, out_ptr, bt,
          seqlen_q, q_offset, scale_log2,
          q_tok_stride, q_head_stride, out_tok_stride, stride_block, stride_slot, stride_kv_head);
    } else if (mode == 1) {
      err = muse_gqa16_pf::launch_prefill_mode<1, true>(
          pdl_prod != 0, grid, block, st, q, kv, gate, fp8_scale, out_ptr, bt,
          seqlen_q, q_offset, scale_log2,
          q_tok_stride, q_head_stride, out_tok_stride, stride_block, stride_slot, stride_kv_head);
    } else {
      err = muse_gqa16_pf::launch_prefill_mode<2, true>(
          pdl_prod != 0, grid, block, st, q, kv, gate, fp8_scale, out_ptr, bt,
          seqlen_q, q_offset, scale_log2,
          q_tok_stride, q_head_stride, out_tok_stride, stride_block, stride_slot, stride_kv_head);
    }
  } else {
    if (mode == 0) {
      err = muse_gqa16_pf::launch_prefill_mode<0, false>(
          pdl_prod != 0, grid, block, st, q, kv, gate, fp8_scale, out_ptr, bt,
          seqlen_q, q_offset, scale_log2,
          q_tok_stride, q_head_stride, out_tok_stride, stride_block, stride_slot, stride_kv_head);
    } else if (mode == 1) {
      err = muse_gqa16_pf::launch_prefill_mode<1, false>(
          pdl_prod != 0, grid, block, st, q, kv, gate, fp8_scale, out_ptr, bt,
          seqlen_q, q_offset, scale_log2,
          q_tok_stride, q_head_stride, out_tok_stride, stride_block, stride_slot, stride_kv_head);
    } else {
      err = muse_gqa16_pf::launch_prefill_mode<2, false>(
          pdl_prod != 0, grid, block, st, q, kv, gate, fp8_scale, out_ptr, bt,
          seqlen_q, q_offset, scale_log2,
          q_tok_stride, q_head_stride, out_tok_stride, stride_block, stride_slot, stride_kv_head);
    }
  }
  return static_cast<int>(err);
}

extern "C" int muse_gqa16_verify_leaf(
    const void* q_ptr,
    const void* kv_ptr,
    void* out_ptr,
    const void* seqused_k_ptr,
    const void* block_table_ptr,
    const void* cu_seqlens_q_ptr,
    int max_q,
    int block_size,
    int window_left,
    int grid_splits,
    int mode,
    float softmax_scale,
    int64_t q_tok_stride,
    int64_t q_head_stride,
    int64_t out_tok_stride,
    int64_t out_head_stride,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    const void* leaf_flag_ptr,
    void* cuda_stream) {
  if (!q_ptr || !kv_ptr || !out_ptr || !seqused_k_ptr || !block_table_ptr || !cu_seqlens_q_ptr ||
      max_q < 1 || max_q > 32 ||
      (grid_splits != 1 && grid_splits != 4 && grid_splits != 8 && grid_splits != 16) ||
      block_size < 16 || (block_size & 15) != 0) {
    return 1;
  }
  const bool use_scale_mode3 = ((mode & 3) == 3);
  const bool use_pipe128     = ((mode & 16) != 0);
  const float scale_log2 =
      use_scale_mode3
          ? (softmax_scale * muse_gqa16::kLog2e)
          : static_cast<float>(static_cast<double>(softmax_scale) * 1.44269504088896340736);
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  const dim3 grid(static_cast<unsigned>(grid_splits), 2u, static_cast<unsigned>(max_q));
  const auto* q = static_cast<const __nv_bfloat16*>(q_ptr);
  const auto* kv = static_cast<const __nv_bfloat16*>(kv_ptr);
  auto* out = static_cast<__nv_bfloat16*>(out_ptr);
  const auto* sk = static_cast<const int32_t*>(seqused_k_ptr);
  const auto* bt = static_cast<const int32_t*>(block_table_ptr);
  const auto* cuq = static_cast<const int32_t*>(cu_seqlens_q_ptr);
  const auto* leaf_flag = static_cast<const int32_t*>(leaf_flag_ptr);

  if (grid_splits == 1) {
    static const bool verify_pdl = []() {
      const char* env = std::getenv("MUSE_GQA16_VERIFY_PDL");
      return !env || !*env || std::strtol(env, nullptr, 10) != 0;
    }();
    auto launch_pdl_blk = [&](auto kernel, dim3 lgrid, dim3 lblock, size_t dsmem, auto... args) -> int {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = lgrid;
      cfg.blockDim = lblock;
      cfg.dynamicSmemBytes = dsmem;
      cfg.stream = stream;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      return static_cast<int>(cudaLaunchKernelEx(&cfg, kernel, args...));
    };
    auto launch_pdl = [&](auto kernel, dim3 lgrid, size_t dsmem, auto... args) -> int {
      return launch_pdl_blk(kernel, lgrid, dim3(128u, 1u, 1u), dsmem, args...);
    };
    static const bool multiq_pdl = []() {
      const char* env = std::getenv("MUSE_GQA16_MULTIQ_PDL");
      return !env || !*env || std::strtol(env, nullptr, 10) != 0;
    }();
    const int v4_mode = (max_q >= 2) ? muse_gqa16::gqa16_verify_v4_mode() : 0;
    const bool v4_ring3 = muse_gqa16::gqa16_verify_v4_ring3_enabled();
    const bool v4_pipe4 = muse_gqa16::gqa16_verify_pipe4_enabled();
    const int kv_l2 = muse_gqa16::gqa16_verify_kv_l2_mode();
    const bool use_pipe4 = (v4_mode == 1 && v4_ring3 && v4_pipe4 && max_q >= 2);
    const bool use_m16_ring3 =
        (v4_mode == 1 && v4_ring3 && max_q >= 2 && max_q <= 16 &&
         muse_gqa16::gqa16_verify_m16_ring3_enabled());
    const bool use_2q4w =
        (v4_mode == 2) ||
        use_pipe4 ||
        use_m16_ring3 ||
        (v4_mode == 1 && max_q >= muse_gqa16::gqa16_verify_v4_min_q() &&
         window_left < 0 && muse_gqa16::gqa16_verify_v4_full_2q4w_enabled());
    if (use_2q4w) {
      muse_gqa16::gqa16_multiq_set_attrs_once();
      const dim3 v4_grid(1u, 2u, static_cast<unsigned>((max_q + 1) >> 1));
      const bool use_v4_pdl = (max_q <= 16) ? verify_pdl : multiq_pdl;
      if (v4_ring3 && v4_pipe4) {
        constexpr size_t v4_smem = sizeof(muse_gqa16::VerifyV4_2Q4W_Pipe4Storage);
        auto run_2q4w_p4 = [&](auto k_pdl, auto k_nopdl) -> int {
          if (use_v4_pdl) {
            return launch_pdl_blk(
                k_pdl, v4_grid, dim3(256u, 1u, 1u), v4_smem,
                q, kv, out, sk, bt, block_size, window_left, scale_log2,
                q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
                cuq, q_tok_stride, out_tok_stride, leaf_flag);
          }
          k_nopdl<<<v4_grid, dim3(256u, 1u, 1u), v4_smem, stream>>>(
              q, kv, out, sk, bt, block_size, window_left, scale_log2,
              q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
              cuq, q_tok_stride, out_tok_stride, leaf_flag);
          return static_cast<int>(cudaGetLastError());
        };
        if (kv_l2 == 2) {
          return run_2q4w_p4(muse_gqa16::gqa16_verify_v4_2q4w_pipe4_kernel<true, 2>,
                             muse_gqa16::gqa16_verify_v4_2q4w_pipe4_kernel<false, 2>);
        }
        if (kv_l2 == 1) {
          return run_2q4w_p4(muse_gqa16::gqa16_verify_v4_2q4w_pipe4_kernel<true, 1>,
                             muse_gqa16::gqa16_verify_v4_2q4w_pipe4_kernel<false, 1>);
        }
        return run_2q4w_p4(muse_gqa16::gqa16_verify_v4_2q4w_pipe4_kernel<true, 0>,
                           muse_gqa16::gqa16_verify_v4_2q4w_pipe4_kernel<false, 0>);
      }
      if (v4_ring3) {
        constexpr size_t v4_smem = sizeof(muse_gqa16::VerifyV4_2Q4W_Ring3Storage);
        auto run_2q4w_r3 = [&](auto k_pdl, auto k_nopdl) -> int {
          if (use_v4_pdl) {
            return launch_pdl_blk(
                k_pdl, v4_grid, dim3(256u, 1u, 1u), v4_smem,
                q, kv, out, sk, bt, block_size, window_left, scale_log2,
                q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
                cuq, q_tok_stride, out_tok_stride, leaf_flag);
          }
          k_nopdl<<<v4_grid, dim3(256u, 1u, 1u), v4_smem, stream>>>(
              q, kv, out, sk, bt, block_size, window_left, scale_log2,
              q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
              cuq, q_tok_stride, out_tok_stride, leaf_flag);
          return static_cast<int>(cudaGetLastError());
        };
        if (kv_l2 == 2) {
          return run_2q4w_r3(muse_gqa16::gqa16_verify_v4_2q4w_ring3_kernel<true, 2>,
                             muse_gqa16::gqa16_verify_v4_2q4w_ring3_kernel<false, 2>);
        }
        if (kv_l2 == 1) {
          return run_2q4w_r3(muse_gqa16::gqa16_verify_v4_2q4w_ring3_kernel<true, 1>,
                             muse_gqa16::gqa16_verify_v4_2q4w_ring3_kernel<false, 1>);
        }
        return run_2q4w_r3(muse_gqa16::gqa16_verify_v4_2q4w_ring3_kernel<true, 0>,
                           muse_gqa16::gqa16_verify_v4_2q4w_ring3_kernel<false, 0>);
      }
      constexpr size_t v4_smem = sizeof(muse_gqa16::VerifyV4_2Q4WStorage);
      if (use_v4_pdl) {
        return launch_pdl_blk(
            muse_gqa16::gqa16_verify_v4_2q4w_kernel<true>, v4_grid, dim3(256u, 1u, 1u), v4_smem,
            q, kv, out, sk, bt, block_size, window_left, scale_log2,
            q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
            cuq, q_tok_stride, out_tok_stride, leaf_flag);
      }
      muse_gqa16::gqa16_verify_v4_2q4w_kernel<false><<<v4_grid, dim3(256u, 1u, 1u), v4_smem, stream>>>(
          q, kv, out, sk, bt, block_size, window_left, scale_log2,
          q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
          cuq, q_tok_stride, out_tok_stride, leaf_flag);
      return static_cast<int>(cudaGetLastError());
    }
    if (v4_mode == 1 && max_q >= muse_gqa16::gqa16_verify_v4_min_q()) {
      muse_gqa16::gqa16_multiq_set_attrs_once();
      const dim3 v4_grid(1u, 2u, static_cast<unsigned>((max_q + 3) / 4));
      const bool use_v4_pdl = (max_q <= 16) ? verify_pdl : multiq_pdl;
      if (v4_ring3) {
        constexpr size_t v4_smem = sizeof(muse_gqa16::VerifyV4Ring3Storage);
        auto run_v4_r3 = [&](auto k_pdl, auto k_nopdl) -> int {
          if (use_v4_pdl) {
            return launch_pdl_blk(
                k_pdl, v4_grid, dim3(256u, 1u, 1u), v4_smem,
                q, kv, out, sk, bt, block_size, window_left, scale_log2,
                q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
                cuq, q_tok_stride, out_tok_stride, leaf_flag);
          }
          k_nopdl<<<v4_grid, dim3(256u, 1u, 1u), v4_smem, stream>>>(
              q, kv, out, sk, bt, block_size, window_left, scale_log2,
              q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
              cuq, q_tok_stride, out_tok_stride, leaf_flag);
          return static_cast<int>(cudaGetLastError());
        };
        if (kv_l2 == 2) {
          return run_v4_r3(muse_gqa16::gqa16_verify_v4_ring3_kernel<true, 2>,
                           muse_gqa16::gqa16_verify_v4_ring3_kernel<false, 2>);
        }
        if (kv_l2 == 1) {
          return run_v4_r3(muse_gqa16::gqa16_verify_v4_ring3_kernel<true, 1>,
                           muse_gqa16::gqa16_verify_v4_ring3_kernel<false, 1>);
        }
        return run_v4_r3(muse_gqa16::gqa16_verify_v4_ring3_kernel<true, 0>,
                         muse_gqa16::gqa16_verify_v4_ring3_kernel<false, 0>);
      }
      constexpr size_t v4_smem = sizeof(muse_gqa16::VerifyV4Storage);
      if (use_v4_pdl) {
        return launch_pdl_blk(
            muse_gqa16::gqa16_verify_v4_kernel<true>, v4_grid, dim3(256u, 1u, 1u), v4_smem,
            q, kv, out, sk, bt, block_size, window_left, scale_log2,
            q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
            cuq, q_tok_stride, out_tok_stride, leaf_flag);
      }
      muse_gqa16::gqa16_verify_v4_kernel<false><<<v4_grid, dim3(256u, 1u, 1u), v4_smem, stream>>>(
          q, kv, out, sk, bt, block_size, window_left, scale_log2,
          q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
          cuq, q_tok_stride, out_tok_stride, leaf_flag);
      return static_cast<int>(cudaGetLastError());
    }
    const int mq_mode = (max_q >= 2) ? muse_gqa16::gqa16_multiq_mode(max_q) : 0;
    if (mq_mode == 8) {
      muse_gqa16::gqa16_multiq_set_attrs_once();
      const dim3 mq_grid(1u, 2u, static_cast<unsigned>((max_q + 7) / 8));
      if (multiq_pdl) {
        return launch_pdl_blk(
            muse_gqa16::gqa16_verify_v3_kernel<8, true>, mq_grid, dim3(256u, 1u, 1u), 0u,
            q, kv, out, sk, bt, block_size, window_left, scale_log2,
            q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
            cuq, q_tok_stride, out_tok_stride, leaf_flag);
      }
      muse_gqa16::gqa16_verify_v3_kernel<8, false><<<mq_grid, dim3(256u, 1u, 1u), 0u, stream>>>(
          q, kv, out, sk, bt, block_size, window_left, scale_log2,
          q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
          cuq, q_tok_stride, out_tok_stride, leaf_flag);
      return static_cast<int>(cudaGetLastError());
    }
    if (mq_mode == 44) {
      muse_gqa16::gqa16_multiq_set_attrs_once();
      const dim3 mq_grid(1u, 2u, static_cast<unsigned>((max_q + 3) / 4));
      if (multiq_pdl) {
        return launch_pdl_blk(
            muse_gqa16::gqa16_verify_v3_kernel<4, true>, mq_grid, dim3(128u, 1u, 1u), 0u,
            q, kv, out, sk, bt, block_size, window_left, scale_log2,
            q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
            cuq, q_tok_stride, out_tok_stride, leaf_flag);
      }
      muse_gqa16::gqa16_verify_v3_kernel<4, false><<<mq_grid, dim3(128u, 1u, 1u), 0u, stream>>>(
          q, kv, out, sk, bt, block_size, window_left, scale_log2,
          q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
          cuq, q_tok_stride, out_tok_stride, leaf_flag);
      return static_cast<int>(cudaGetLastError());
    }
    if (max_q >= muse_gqa16_wg2::gqa16_wg2_min_q()) {
      muse_gqa16_wg2::gqa16_wg2_set_attrs_once();
      const dim3 wg2_grid(1u, 2u, static_cast<unsigned>((max_q + 1) >> 1));
      const dim3 wg2_block(256u, 1u, 1u);
      constexpr size_t wg2_smem = sizeof(muse_gqa16_wg2::Wg2Pipe128Storage);
      if (verify_pdl) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = wg2_grid;
        cfg.blockDim = wg2_block;
        cfg.dynamicSmemBytes = wg2_smem;
        cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        return static_cast<int>(cudaLaunchKernelEx(
            &cfg, muse_gqa16_wg2::gqa16_verify_wg2_kernel<true>,
            q, kv, out, sk, bt, block_size, window_left, scale_log2,
            q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
            cuq, q_tok_stride, out_tok_stride, leaf_flag));
      }
      muse_gqa16_wg2::gqa16_verify_wg2_kernel<false><<<wg2_grid, wg2_block, wg2_smem, stream>>>(
          q, kv, out, sk, bt, block_size, window_left, scale_log2,
          q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
          cuq, q_tok_stride, out_tok_stride, leaf_flag);
      return static_cast<int>(cudaGetLastError());
    }
    if (mq_mode == 4) {
      muse_gqa16::gqa16_multiq_set_attrs_once();
      const dim3 mq_grid(1u, 2u, static_cast<unsigned>((max_q + 3) / 4));
      const size_t dsmem = sizeof(muse_gqa16::MultiQPipe128Storage<4>);
      if (verify_pdl) {
        return launch_pdl(
            muse_gqa16::gqa16_fa2_block128_pipe_multiq_kernel<4, true>, mq_grid, dsmem,
            q, kv, out, sk, bt, block_size, window_left, scale_log2,
            q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
            cuq, q_tok_stride, out_tok_stride, leaf_flag);
      }
      muse_gqa16::gqa16_fa2_block128_pipe_multiq_kernel<4, false><<<mq_grid, dim3(128u, 1u, 1u), dsmem, stream>>>(
          q, kv, out, sk, bt, block_size, window_left, scale_log2,
          q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
          cuq, q_tok_stride, out_tok_stride, leaf_flag);
      return static_cast<int>(cudaGetLastError());
    }
    if (mq_mode == 2) {
      muse_gqa16::gqa16_multiq_set_attrs_once();
      const dim3 mq_grid(1u, 2u, static_cast<unsigned>((max_q + 1) / 2));
      const size_t dsmem = sizeof(muse_gqa16::MultiQPipe128Storage<2>);
      if (verify_pdl) {
        return launch_pdl(
            muse_gqa16::gqa16_fa2_block128_pipe_multiq_kernel<2, true>, mq_grid, dsmem,
            q, kv, out, sk, bt, block_size, window_left, scale_log2,
            q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
            cuq, q_tok_stride, out_tok_stride, leaf_flag);
      }
      muse_gqa16::gqa16_fa2_block128_pipe_multiq_kernel<2, false><<<mq_grid, dim3(128u, 1u, 1u), dsmem, stream>>>(
          q, kv, out, sk, bt, block_size, window_left, scale_log2,
          q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
          cuq, q_tok_stride, out_tok_stride, leaf_flag);
      return static_cast<int>(cudaGetLastError());
    }
    if (verify_pdl) {
      return launch_pdl(
          muse_gqa16::gqa16_fa2_block128_pipe_kernel<true, true>, grid, 0,
          q, kv, out, static_cast<float*>(nullptr), static_cast<float*>(nullptr),
          static_cast<int32_t*>(nullptr), sk, bt, block_size, window_left, 1,
          softmax_scale, scale_log2, q_head_stride, out_head_stride,
          stride_block, stride_slot, stride_kv_head, cuq, q_tok_stride, out_tok_stride, leaf_flag);
    }
    muse_gqa16::gqa16_fa2_block128_pipe_kernel<true, false><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
        q, kv, out, nullptr, nullptr, nullptr, sk, bt, block_size, window_left, 1,
        softmax_scale, scale_log2, q_head_stride, out_head_stride,
        stride_block, stride_slot, stride_kv_head, cuq, q_tok_stride, out_tok_stride, leaf_flag);
    return static_cast<int>(cudaGetLastError());
  }

  if (use_pipe128) {
    if (grid_splits == 4) {
      muse_gqa16::gqa16_1pass_dsplit_pipe_kernel<4, true><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
          q, kv, out, sk, bt, block_size, window_left, scale_log2,
          q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
          cuq, q_tok_stride, out_tok_stride);
    } else if (grid_splits == 8) {
      muse_gqa16::gqa16_1pass_dsplit_pipe_kernel<8, true><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
          q, kv, out, sk, bt, block_size, window_left, scale_log2,
          q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
          cuq, q_tok_stride, out_tok_stride);
    } else {
      muse_gqa16::gqa16_1pass_dsplit_pipe_kernel<16, true><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
          q, kv, out, sk, bt, block_size, window_left, scale_log2,
          q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
          cuq, q_tok_stride, out_tok_stride);
    }
  } else {
    if (grid_splits == 4) {
      muse_gqa16::gqa16_1pass_dsplit_kernel<128, 4, true><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
          q, kv, out, sk, bt, block_size, window_left, scale_log2,
          q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
          cuq, q_tok_stride, out_tok_stride);
    } else if (grid_splits == 8) {
      muse_gqa16::gqa16_1pass_dsplit_kernel<128, 8, true><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
          q, kv, out, sk, bt, block_size, window_left, scale_log2,
          q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
          cuq, q_tok_stride, out_tok_stride);
    } else {
      muse_gqa16::gqa16_1pass_dsplit_kernel<128, 16, true><<<grid, dim3(128u, 1u, 1u), 0, stream>>>(
          q, kv, out, sk, bt, block_size, window_left, scale_log2,
          q_head_stride, out_head_stride, stride_block, stride_slot, stride_kv_head,
          cuq, q_tok_stride, out_tok_stride);
    }
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int muse_gqa16_verify(
    const void* q_ptr,
    const void* kv_ptr,
    void* out_ptr,
    const void* seqused_k_ptr,
    const void* block_table_ptr,
    const void* cu_seqlens_q_ptr,
    int max_q,
    int block_size,
    int window_left,
    int grid_splits,
    int mode,
    float softmax_scale,
    int64_t q_tok_stride,
    int64_t q_head_stride,
    int64_t out_tok_stride,
    int64_t out_head_stride,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_kv_head,
    void* cuda_stream) {
  return muse_gqa16_verify_leaf(
      q_ptr, kv_ptr, out_ptr, seqused_k_ptr, block_table_ptr, cu_seqlens_q_ptr,
      max_q, block_size, window_left, grid_splits, mode, softmax_scale,
      q_tok_stride, q_head_stride, out_tok_stride, out_head_stride,
      stride_block, stride_slot, stride_kv_head, nullptr, cuda_stream);
}
