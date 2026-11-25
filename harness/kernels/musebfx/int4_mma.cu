// SPDX-License-Identifier: Apache-2.0
// muse_int4_mma: 4-bit asymmetric group-64 W4A16 weight-only Tensor-Core GEMM for the pinned DFlash drafter.
// Format (4.50 bits/weight, 0 wasted bits, 16-byte aligned):
//   Each 8x64 tile (8 output rows x 64 K cols) is 288 bytes (18 x uint4):
//     [0..127]   uint32_t q_lo[32] : for lane l=(r<<2)|t, 8 x 4-bit unsigned codes (s=0..1, 4 weights/k16)
//     [128..255] uint32_t q_hi[32] : for lane l=(r<<2)|t, 8 x 4-bit unsigned codes (s=2..3, 4 weights/k16)
//     [256..287] __nv_bfloat162 sz[8] : (scale, zero) for rows r=0..7 (w = q * scale + zero, q in 0..15)
//   Tiles are stored in [N/64, K/64, 8] order (2304 bytes = 144 x uint4 per 4-warp CTA per K=64 step).
//   Runs on SM121 with 6-stage plain cp.async.cg (no L2::cache_hint / createpolicy -> SM121 odd-UR safe),
//   1 barrier/step, 2x N-tile A reuse, 2 MiB TLB prefetch, native M<=16 m16n8k16, and zero-cat aux5 fc input.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <stdint.h>

namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * 32;                    // 128
constexpr int kNTilesPerWarp = 2;                                // 16 N-rows per warp -> 2x A reuse
constexpr int kNTilesPerCTA = kWarpsPerBlock * kNTilesPerWarp;   // 8
constexpr int kRowsPerCTA = kNTilesPerCTA * 8;                   // 64
constexpr int kTileBytes = 288;                                  // 18 * 16
constexpr int kStepWBytes = kNTilesPerCTA * kTileBytes;          // 2304 = 144 * 16
constexpr int kStepWU4 = kStepWBytes / 16;                       // 144
constexpr int kStages = 6;

__device__ __forceinline__ void int4_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void int4_mma_m16n8k16_full(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void int4_cp_async16_cg(uint32_t sd, const void* gs) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ void int4_cp_async16_ca(uint32_t sd, const void* gs) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ uint32_t int4_bf16x2_u32(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float int4_subwarp4_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}

// GPU packer: BF16 [N, K] (N%64==0, K%64==0) -> uint8 [N/64, K/64, 8, 288]
// Grid: (N/8, K/64), Block: 32 threads (1 warp per 8x64 tile).
// When kMseOpt is true, optimizes per-group (scale, zero) via warp-parallel clipping search
// + 5-iteration closed-form ALS minimizing exact BF16-dequantized MSE (monotonically <= min-max).
template <bool kMseOpt>
__global__ void int4_pack_kernel(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ out,
    int N, int K) {
  const int n_tile = blockIdx.x;        // 0 .. N/8 - 1
  const int kg = blockIdx.y;            // 0 .. K/64 - 1
  const int lane = threadIdx.x;         // 0 .. 31
  const int r = lane >> 2;              // 0 .. 7
  const int t = lane & 3;               // 0 .. 3
  const int row = n_tile * 8 + r;
  const int k_base = kg * 64;

  float vals[16];
  float rmin = 1e30f;
  float rmax = -1e30f;
  const __nv_bfloat16* wrow = W + static_cast<size_t>(row) * K + k_base;
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int kb = s * 16 + 2 * t;
    vals[4 * s + 0] = __bfloat162float(wrow[kb + 0]);
    vals[4 * s + 1] = __bfloat162float(wrow[kb + 1]);
    vals[4 * s + 2] = __bfloat162float(wrow[kb + 8]);
    vals[4 * s + 3] = __bfloat162float(wrow[kb + 9]);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    rmin = fminf(rmin, vals[i]);
    rmax = fmaxf(rmax, vals[i]);
  }
  rmin = fminf(rmin, __shfl_xor_sync(0xffffffffu, rmin, 1));
  rmin = fminf(rmin, __shfl_xor_sync(0xffffffffu, rmin, 2));
  rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, 1));
  rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, 2));

  float sc_mm = (rmax - rmin) * (1.0f / 15.0f);
  if (sc_mm < 1e-30f) sc_mm = 1e-30f;
  __nv_bfloat162 sz_bf = __floats2bfloat162_rn(sc_mm, rmin);

  if constexpr (kMseOpt) {
    float lsum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      lsum += vals[i];
    }
    const float sw = int4_subwarp4_sum(lsum);
    const float mu = sw * (1.0f / 64.0f);

    float lvar = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float d = vals[i] - mu;
      lvar = __fmaf_rn(d, d, lvar);
    }
    const float std_v = sqrtf(fmaxf(int4_subwarp4_sum(lvar) * (1.0f / 64.0f), 0.0f));

    auto eval_cand = [&](float sc_in, float zr_in, __nv_bfloat162& out_sz, float& out_mse) {
      if (!(sc_in > 1e-30f)) sc_in = 1e-30f;
      out_sz = __floats2bfloat162_rn(sc_in, zr_in);
      const float sc_q = __bfloat162float(out_sz.x);
      const float zr_q = __bfloat162float(out_sz.y);
      const float inv_sc = (sc_q > 1e-30f) ? (1.0f / sc_q) : 0.0f;
      float err = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        int q = __float2int_rn((vals[i] - zr_q) * inv_sc);
        if (q < 0) q = 0;
        if (q > 15) q = 15;
        const float w_hat = __bfloat162float(
            __float2bfloat16_rn(__fmaf_rn(static_cast<float>(q), sc_q, zr_q)));
        const float diff = vals[i] - w_hat;
        err = __fmaf_rn(diff, diff, err);
      }
      out_mse = int4_subwarp4_sum(err);
    };

    __nv_bfloat162 best_sz = sz_bf;
    float best_mse = 0.0f;
    eval_cand(sc_mm, rmin, best_sz, best_mse);

    __nv_bfloat162 try_sz;
    float try_mse;
    auto consider_interval = [&](float lo, float hi) {
      eval_cand((hi - lo) * (1.0f / 15.0f), lo, try_sz, try_mse);
      if (try_mse < best_mse) {
        best_mse = try_mse;
        best_sz = try_sz;
      }
    };

    const float mid = 0.5f * (rmin + rmax);
    const float half = 0.5f * (rmax - rmin);
    constexpr float kMidAlphas[4] = {0.94f, 0.88f, 0.82f, 0.75f};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float h = half * kMidAlphas[c];
      consider_interval(mid - h, mid + h);
    }

    constexpr float kStdCs[5] = {3.20f, 2.80f, 2.40f, 2.10f, 1.80f};
#pragma unroll
    for (int c = 0; c < 5; ++c) {
      const float d = std_v * kStdCs[c];
      consider_interval(fmaxf(rmin, mu - d), fminf(rmax, mu + d));
    }

    constexpr float kOneSideCs[2] = {2.60f, 2.20f};
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const float d = std_v * kOneSideCs[c];
      consider_interval(rmin, fminf(rmax, mu + d));
      consider_interval(fmaxf(rmin, mu - d), rmax);
    }

    // 5 iterations of exact closed-form ALS (1D linear regression q -> (scale, zero)).
    float cur_sc = __bfloat162float(best_sz.x);
    float cur_zr = __bfloat162float(best_sz.y);
#pragma unroll
    for (int it = 0; it < 5; ++it) {
      const float inv_cur = (cur_sc > 1e-30f) ? (1.0f / cur_sc) : 0.0f;
      float lsq = 0.0f;
      float lsqq = 0.0f;
      float lsqw = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        int q = __float2int_rn((vals[i] - cur_zr) * inv_cur);
        if (q < 0) q = 0;
        if (q > 15) q = 15;
        const float fq = static_cast<float>(q);
        lsq += fq;
        lsqq = __fmaf_rn(fq, fq, lsqq);
        lsqw = __fmaf_rn(fq, vals[i], lsqw);
      }
      const float sq = int4_subwarp4_sum(lsq);
      const float sqq = int4_subwarp4_sum(lsqq);
      const float sqw = int4_subwarp4_sum(lsqw);
      const float denom = 64.0f * sqq - sq * sq;
      if (denom > 0.5f) {
        const float cand_sc = (64.0f * sqw - sq * sw) / denom;
        const float cand_zr = (sw - cand_sc * sq) * (1.0f / 64.0f);
        if (cand_sc > 1e-30f && cand_sc < 1e30f && fabsf(cand_zr) < 1e30f) {
          cur_sc = cand_sc;
          cur_zr = cand_zr;
        }
      }
      eval_cand(cur_sc, cur_zr, try_sz, try_mse);
      if (try_mse < best_mse) {
        best_mse = try_mse;
        best_sz = try_sz;
      }
    }

    uint32_t sz_bits = *reinterpret_cast<const uint32_t*>(&best_sz);
    sz_bits = __shfl_sync(0xffffffffu, sz_bits, r << 2);
    sz_bf = *reinterpret_cast<const __nv_bfloat162*>(&sz_bits);
  }

  const float sc_q = __bfloat162float(sz_bf.x);
  const float zr_q = __bfloat162float(sz_bf.y);
  const float inv_sc = (sc_q > 1e-30f) ? (1.0f / sc_q) : 0.0f;

  uint32_t q_lo = 0u;
  uint32_t q_hi = 0u;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int q = __float2int_rn((vals[i] - zr_q) * inv_sc);
    if (q < 0) q = 0;
    if (q > 15) q = 15;
    q_lo |= (static_cast<uint32_t>(q) << (4 * i));
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int q = __float2int_rn((vals[8 + i] - zr_q) * inv_sc);
    if (q < 0) q = 0;
    if (q > 15) q = 15;
    q_hi |= (static_cast<uint32_t>(q) << (4 * i));
  }

  const int cta_n = n_tile >> 3;
  const int sub_tile = n_tile & 7;
  const int num_kg = K >> 6;
  const size_t tile_idx = (static_cast<size_t>(cta_n) * num_kg + kg) * kNTilesPerCTA + sub_tile;
  uint8_t* dst = out + tile_idx * kTileBytes;

  reinterpret_cast<uint32_t*>(dst)[lane] = q_lo;
  reinterpret_cast<uint32_t*>(dst + 128)[lane] = q_hi;
  if (t == 0) {
    reinterpret_cast<__nv_bfloat162*>(dst + 256)[r] = sz_bf;
  }
}

struct Aux5Desc {
  const __nv_bfloat16* p[5];
  int64_t stride[5];
  int kg_sub;
};

template <int kMaxM>
struct alignas(16) Int4StageSmem {
  uint8_t w[kStages][kStepWBytes];         // 6 * 2304 = 13824 B
  __nv_bfloat16 a[kStages][kMaxM][64];     // 6 * 1024 = 6144 B (M<=8) or 12288 B (M<=16)
};                                         // Total = 19.5 KiB (M<=8) or 25.5 KiB (M<=16), 3 CTAs/SM

template <bool kSplitK, bool kM16, bool kAux5>
__global__ void __launch_bounds__(128, 3) int4_mma_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    Aux5Desc aux5,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ C_fp32,
    int M, int N, int K, int split_k) {
  __shared__ Int4StageSmem<kM16 ? 16 : 8> smem;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;

  const int cta_n = blockIdx.x;
  const int sk_idx = kSplitK ? blockIdx.y : 0;
  const int kg_per_sk = kSplitK ? ((num_kg + split_k - 1) / split_k) : num_kg;
  const int kg_start = kSplitK ? (sk_idx * kg_per_sk) : 0;
  int kg_end = kSplitK ? (kg_start + kg_per_sk) : num_kg;
  if (kg_end > num_kg) kg_end = num_kg;
  const int my_kgs = kg_end - kg_start;
  if (my_kgs <= 0) return;

  const size_t cta_stride_bytes = static_cast<size_t>(num_kg) * kStepWBytes;
  const uint8_t* w_cta = W_packed + static_cast<size_t>(cta_n) * cta_stride_bytes;

  // Issue 2 MiB TLB prefetch from block (0, 0).
  if (cta_n == 0 && sk_idx == 0) {
    const size_t total_bytes = static_cast<size_t>(N / kRowsPerCTA) * cta_stride_bytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  auto issue_stage = [&](int kg, int st) {
    const uint8_t* w_src = w_cta + static_cast<size_t>(kg) * kStepWBytes;
    // 144 x 16B = 2304B across 128 threads: tid 0..127 load 0..127, tid 0..15 load 128..143.
    {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_cg(sd, w_src + tid * 16);
    }
    if (tid < (kStepWU4 - kThreads)) {
      const int idx = kThreads + tid;
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][idx * 16]));
      int4_cp_async16_cg(sd, w_src + idx * 16);
    }

    const __nv_bfloat16* a_base = nullptr;
    int64_t a_stride = K;
    int a_col_base = kg * 64;
    if constexpr (kAux5) {
      const int seg = kg / aux5.kg_sub;
      const int local_kg = kg - seg * aux5.kg_sub;
      a_base = aux5.p[seg];
      a_stride = aux5.stride[seg];
      a_col_base = local_kg * 64;
    } else {
      a_base = A;
    }

    if constexpr (kM16) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][row][col8 * 8]));
        int4_cp_async16_ca(sd, a_base + static_cast<size_t>(row) * a_stride + a_col_base + col8 * 8);
      }
    } else if (tid < 64) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][row][col8 * 8]));
        int4_cp_async16_ca(sd, a_base + static_cast<size_t>(row) * a_stride + a_col_base + col8 * 8);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kStages - 1;  // 5
  const int preload = (my_kgs < kPreload) ? my_kgs : kPreload;
  for (int i = 0; i < preload; ++i) {
    issue_stage(kg_start + i, i);
  }

  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);

  int st = 0;
  for (int i = 0; i < my_kgs; ++i) {
    const int rem = my_kgs - 1 - i;
    if (rem >= kPreload) {
      asm volatile("cp.async.wait_group 4;\n" ::: "memory");
    } else if (rem == 4) {
      asm volatile("cp.async.wait_group 4;\n" ::: "memory");
    } else if (rem == 3) {
      asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    } else if (rem == 2) {
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    } else if (rem == 1) {
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();

    const uint32_t* a_row_u32 = reinterpret_cast<const uint32_t*>(&smem.a[st][g][0]);
    uint32_t a0_r[4], a2_r[4];
    uint32_t a1_r[4], a3_r[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      a0_r[s] = (g < M) ? a_row_u32[s * 8 + t] : 0u;
      a2_r[s] = (g < M) ? a_row_u32[s * 8 + 4 + t] : 0u;
    }
    if constexpr (kM16) {
      const uint32_t* a_hi_u32 = reinterpret_cast<const uint32_t*>(&smem.a[st][g + 8][0]);
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        a1_r[s] = has_hi ? a_hi_u32[s * 8 + t] : 0u;
        a3_r[s] = has_hi ? a_hi_u32[s * 8 + 4 + t] : 0u;
      }
    }

    const uint8_t* wt0 = &smem.w[st][(warp * 2 + 0) * kTileBytes];
    const uint8_t* wt1 = &smem.w[st][(warp * 2 + 1) * kTileBytes];
    const uint32_t q0_lo = reinterpret_cast<const uint32_t*>(wt0)[lane];
    const uint32_t q0_hi = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
    const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 256)[g];
    const float sc0 = __bfloat162float(sz0.x);
    const float zr0 = __bfloat162float(sz0.y);

    const uint32_t q1_lo = reinterpret_cast<const uint32_t*>(wt1)[lane];
    const uint32_t q1_hi = reinterpret_cast<const uint32_t*>(wt1 + 128)[lane];
    const __nv_bfloat162 sz1 = reinterpret_cast<const __nv_bfloat162*>(wt1 + 256)[g];
    const float sc1 = __bfloat162float(sz1.x);
    const float zr1 = __bfloat162float(sz1.y);

    const int next_i = i + kPreload;
    if (next_i < my_kgs) {
      const int next_st = (st + kPreload >= kStages) ? (st + kPreload - kStages) : (st + kPreload);
      issue_stage(kg_start + next_i, next_st);
    }

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t w0 = (s < 2) ? (q0_lo >> (16 * s)) : (q0_hi >> (16 * (s - 2)));
      const uint32_t b0 = int4_bf16x2_u32(
          __fmaf_rn(static_cast<float>(w0 & 15u), sc0, zr0),
          __fmaf_rn(static_cast<float>((w0 >> 4) & 15u), sc0, zr0));
      const uint32_t b1 = int4_bf16x2_u32(
          __fmaf_rn(static_cast<float>((w0 >> 8) & 15u), sc0, zr0),
          __fmaf_rn(static_cast<float>((w0 >> 12) & 15u), sc0, zr0));
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0_r[s], a1_r[s], a2_r[s], a3_r[s], b0, b1);
      } else {
        int4_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0_r[s], a2_r[s], b0, b1);
      }

      const uint32_t w1 = (s < 2) ? (q1_lo >> (16 * s)) : (q1_hi >> (16 * (s - 2)));
      const uint32_t c0 = int4_bf16x2_u32(
          __fmaf_rn(static_cast<float>(w1 & 15u), sc1, zr1),
          __fmaf_rn(static_cast<float>((w1 >> 4) & 15u), sc1, zr1));
      const uint32_t c1 = int4_bf16x2_u32(
          __fmaf_rn(static_cast<float>((w1 >> 8) & 15u), sc1, zr1),
          __fmaf_rn(static_cast<float>((w1 >> 12) & 15u), sc1, zr1));
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0_r[s], a1_r[s], a2_r[s], a3_r[s], c0, c1);
      } else {
        int4_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0_r[s], a2_r[s], c0, c1);
      }
    }

    st = (st + 1 == kStages) ? 0 : (st + 1);
  }

  const int col0 = cta_n * kRowsPerCTA + (warp * 2 + 0) * 8 + 2 * t;
  const int col1 = cta_n * kRowsPerCTA + (warp * 2 + 1) * 8 + 2 * t;
  if (g < M) {
    if constexpr (!kSplitK) {
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) = int4_bf16x2_u32(d0_0, d1_0);
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col1) = int4_bf16x2_u32(d0_1, d1_1);
    } else {
      float* p0 = C_fp32 + (static_cast<size_t>(sk_idx) * M + g) * N + col0;
      float* p1 = C_fp32 + (static_cast<size_t>(sk_idx) * M + g) * N + col1;
      p0[0] = d0_0; p0[1] = d1_0;
      p1[0] = d0_1; p1[1] = d1_1;
    }
  }
  if constexpr (kM16) {
    if (has_hi) {
      const int ghi = g + 8;
      if constexpr (!kSplitK) {
        *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(ghi) * N + col0) = int4_bf16x2_u32(d2_0, d3_0);
        *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(ghi) * N + col1) = int4_bf16x2_u32(d2_1, d3_1);
      } else {
        float* p0 = C_fp32 + (static_cast<size_t>(sk_idx) * M + ghi) * N + col0;
        float* p1 = C_fp32 + (static_cast<size_t>(sk_idx) * M + ghi) * N + col1;
        p0[0] = d2_0; p0[1] = d3_0;
        p1[0] = d2_1; p1[1] = d3_1;
      }
    }
  }
}

__global__ void int4_reduce_splitk_kernel(
    const float* __restrict__ C_fp32,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int split_k) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = M * N;
  if (idx >= total) return;
  float sum = 0.0f;
  for (int s = 0; s < split_k; ++s) {
    sum += C_fp32[static_cast<size_t>(s) * total + idx];
  }
  C[idx] = __float2bfloat16_rn(sum);
}

// Persistent 144-CTA (3 CTAs/SM x 48 SMs) Stream-K W4A16 GEMM with continuous cross-tile cp.async.cg
// pipeline (zero pipeline drain across N-tile boundaries, zero tail-wave quantization, zero second
// reduce launch, lock-free atomicAdd + self-zeroing atomicExch, zero-div aux5 seg stepping, SM121 safe).
template <bool kM16, bool kAux5>
__global__ void __launch_bounds__(128, 3) int4_mma_streamk_kernel(
    const __nv_bfloat16* __restrict__ A,
    Aux5Desc aux5,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ sk_partials,  // [max_n_blocks, 16, 64] zero-initialized & self-zeroing
    int* __restrict__ sk_done,        // [max_n_blocks] zero-initialized & self-zeroing
    int M, int N, int K, int n_blocks, int grid_ctas, int pdl_mode) {
  __shared__ Int4StageSmem<kM16 ? 16 : 8> smem;
  __shared__ int s_is_last;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;
  const int total_iters = n_blocks * num_kg;

  const int cta_id = blockIdx.x;
  const int iter_start = static_cast<int>((static_cast<int64_t>(cta_id) * total_iters) / grid_ctas);
  const int iter_end = static_cast<int>((static_cast<int64_t>(cta_id + 1) * total_iters) / grid_ctas);
  const int my_iters = iter_end - iter_start;
  if (my_iters <= 0) return;

  if (cta_id == 0) {
    const size_t total_bytes = static_cast<size_t>(total_iters) * kStepWBytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  int cur_n = iter_start / num_kg;
  int cur_kg = iter_start - cur_n * num_kg;
  int seg_start_kg = cur_kg;

  int pre_iter = iter_start;
  int pre_kg = cur_kg;
  int pre_seg = kAux5 ? (cur_kg / aux5.kg_sub) : 0;
  int pre_local_kg = kAux5 ? (cur_kg - pre_seg * aux5.kg_sub) : 0;

  auto issue_w_only = [&](int st, int w_iter) {
    const uint8_t* w_src = W_packed + static_cast<size_t>(w_iter) * kStepWBytes;
    {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_cg(sd, w_src + tid * 16);
    }
    if (tid < (kStepWU4 - kThreads)) {
      const int idx = kThreads + tid;
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][idx * 16]));
      int4_cp_async16_cg(sd, w_src + idx * 16);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };
  auto load_a_sync = [&](int st, const __nv_bfloat16* a_base, int64_t a_stride, int a_col_base) {
    if constexpr (kM16) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        *reinterpret_cast<uint4*>(&smem.a[st][row][col8 * 8]) =
            __ldg(reinterpret_cast<const uint4*>(a_base + static_cast<size_t>(row) * a_stride + a_col_base + col8 * 8));
      }
    } else if (tid < 64) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        *reinterpret_cast<uint4*>(&smem.a[st][row][col8 * 8]) =
            __ldg(reinterpret_cast<const uint4*>(a_base + static_cast<size_t>(row) * a_stride + a_col_base + col8 * 8));
      }
    }
  };
  auto issue_next = [&](int st) {
    const uint8_t* w_src = W_packed + static_cast<size_t>(pre_iter) * kStepWBytes;
    {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_cg(sd, w_src + tid * 16);
    }
    if (tid < (kStepWU4 - kThreads)) {
      const int idx = kThreads + tid;
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][idx * 16]));
      int4_cp_async16_cg(sd, w_src + idx * 16);
    }

    const __nv_bfloat16* a_base = kAux5 ? aux5.p[pre_seg] : A;
    const int64_t a_stride = kAux5 ? aux5.stride[pre_seg] : static_cast<int64_t>(K);
    const int a_col_base = kAux5 ? (pre_local_kg * 64) : (pre_kg * 64);

    if constexpr (kM16) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][row][col8 * 8]));
        int4_cp_async16_ca(sd, a_base + static_cast<size_t>(row) * a_stride + a_col_base + col8 * 8);
      }
    } else if (tid < 64) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][row][col8 * 8]));
        int4_cp_async16_ca(sd, a_base + static_cast<size_t>(row) * a_stride + a_col_base + col8 * 8);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    ++pre_iter;
    if constexpr (kAux5) {
      if (++pre_local_kg == aux5.kg_sub) {
        pre_local_kg = 0;
        if (++pre_seg == 5) pre_seg = 0;
      }
    } else {
      if (++pre_kg == num_kg) pre_kg = 0;
    }
  };

  constexpr int kPreload = kStages - 1;  // 5
  const int preload = (my_iters < kPreload) ? my_iters : kPreload;
  if ((pdl_mode & 1) != 0) {
    for (int i = 0; i < preload; ++i) {
      issue_w_only(i, pre_iter + i);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = 0; i < preload; ++i) {
      const __nv_bfloat16* a_base = kAux5 ? aux5.p[pre_seg] : A;
      const int64_t a_stride = kAux5 ? aux5.stride[pre_seg] : static_cast<int64_t>(K);
      const int a_col_base = kAux5 ? (pre_local_kg * 64) : (pre_kg * 64);
      load_a_sync(i, a_base, a_stride, a_col_base);
      ++pre_iter;
      if constexpr (kAux5) {
        if (++pre_local_kg == aux5.kg_sub) {
          pre_local_kg = 0;
          if (++pre_seg == 5) pre_seg = 0;
        }
      } else {
        if (++pre_kg == num_kg) pre_kg = 0;
      }
    }
  } else {
    for (int i = 0; i < preload; ++i) {
      issue_next(i);
    }
  }

  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int local_c0 = (warp * 2 + 0) * 8 + 2 * t;
  const int local_c1 = (warp * 2 + 1) * 8 + 2 * t;
  const int trig_i = ((pdl_mode & 2) != 0) ? ((my_iters > 4) ? (my_iters - 4) : 0) : -1;

  int st = 0;
  for (int i = 0; i < my_iters; ++i) {
    if (i == trig_i) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    const int rem = my_iters - 1 - i;
    if (rem >= kPreload) {
      asm volatile("cp.async.wait_group 4;\n" ::: "memory");
    } else if (rem == 4) {
      asm volatile("cp.async.wait_group 4;\n" ::: "memory");
    } else if (rem == 3) {
      asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    } else if (rem == 2) {
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    } else if (rem == 1) {
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();

    const uint32_t* a_row_u32 = reinterpret_cast<const uint32_t*>(&smem.a[st][g][0]);
    uint32_t a0_r[4], a2_r[4];
    uint32_t a1_r[4], a3_r[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      a0_r[s] = (g < M) ? a_row_u32[s * 8 + t] : 0u;
      a2_r[s] = (g < M) ? a_row_u32[s * 8 + 4 + t] : 0u;
    }
    if constexpr (kM16) {
      const uint32_t* a_hi_u32 = reinterpret_cast<const uint32_t*>(&smem.a[st][g + 8][0]);
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        a1_r[s] = has_hi ? a_hi_u32[s * 8 + t] : 0u;
        a3_r[s] = has_hi ? a_hi_u32[s * 8 + 4 + t] : 0u;
      }
    }

    const uint8_t* wt0 = &smem.w[st][(warp * 2 + 0) * kTileBytes];
    const uint8_t* wt1 = &smem.w[st][(warp * 2 + 1) * kTileBytes];
    const uint32_t q0_lo = reinterpret_cast<const uint32_t*>(wt0)[lane];
    const uint32_t q0_hi = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
    const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 256)[g];
    const float sc0 = __bfloat162float(sz0.x);
    const float zr0 = __bfloat162float(sz0.y);

    const uint32_t q1_lo = reinterpret_cast<const uint32_t*>(wt1)[lane];
    const uint32_t q1_hi = reinterpret_cast<const uint32_t*>(wt1 + 128)[lane];
    const __nv_bfloat162 sz1 = reinterpret_cast<const __nv_bfloat162*>(wt1 + 256)[g];
    const float sc1 = __bfloat162float(sz1.x);
    const float zr1 = __bfloat162float(sz1.y);

    if (i + kPreload < my_iters) {
      const int next_st = (st + kPreload >= kStages) ? (st + kPreload - kStages) : (st + kPreload);
      issue_next(next_st);
    }

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t w0 = (s < 2) ? (q0_lo >> (16 * s)) : (q0_hi >> (16 * (s - 2)));
      const uint32_t b0 = int4_bf16x2_u32(
          __fmaf_rn(static_cast<float>(w0 & 15u), sc0, zr0),
          __fmaf_rn(static_cast<float>((w0 >> 4) & 15u), sc0, zr0));
      const uint32_t b1 = int4_bf16x2_u32(
          __fmaf_rn(static_cast<float>((w0 >> 8) & 15u), sc0, zr0),
          __fmaf_rn(static_cast<float>((w0 >> 12) & 15u), sc0, zr0));
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0_r[s], a1_r[s], a2_r[s], a3_r[s], b0, b1);
      } else {
        int4_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0_r[s], a2_r[s], b0, b1);
      }

      const uint32_t w1 = (s < 2) ? (q1_lo >> (16 * s)) : (q1_hi >> (16 * (s - 2)));
      const uint32_t c0 = int4_bf16x2_u32(
          __fmaf_rn(static_cast<float>(w1 & 15u), sc1, zr1),
          __fmaf_rn(static_cast<float>((w1 >> 4) & 15u), sc1, zr1));
      const uint32_t c1 = int4_bf16x2_u32(
          __fmaf_rn(static_cast<float>((w1 >> 8) & 15u), sc1, zr1),
          __fmaf_rn(static_cast<float>((w1 >> 12) & 15u), sc1, zr1));
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0_r[s], a1_r[s], a2_r[s], a3_r[s], c0, c1);
      } else {
        int4_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0_r[s], a2_r[s], c0, c1);
      }
    }

    st = (st + 1 == kStages) ? 0 : (st + 1);

    const bool tile_done = (cur_kg + 1 == num_kg) || (i + 1 == my_iters);
    if (tile_done) {
      const int seg_kgs = (cur_kg + 1) - seg_start_kg;
      const int col0 = cur_n * kRowsPerCTA + local_c0;
      const int col1 = cur_n * kRowsPerCTA + local_c1;
      if (seg_kgs == num_kg) {
        if (g < M) {
          *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) = int4_bf16x2_u32(d0_0, d1_0);
          *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col1) = int4_bf16x2_u32(d0_1, d1_1);
        }
        if constexpr (kM16) {
          if (has_hi) {
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * N + col0) = int4_bf16x2_u32(d2_0, d3_0);
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * N + col1) = int4_bf16x2_u32(d2_1, d3_1);
          }
        }
      } else {
        float* tile_p = sk_partials + static_cast<size_t>(cur_n) * (16 * 64);
        if (g < M) {
          float* row_p = tile_p + g * 64;
          atomicAdd(&row_p[local_c0 + 0], d0_0);
          atomicAdd(&row_p[local_c0 + 1], d1_0);
          atomicAdd(&row_p[local_c1 + 0], d0_1);
          atomicAdd(&row_p[local_c1 + 1], d1_1);
        }
        if constexpr (kM16) {
          if (has_hi) {
            float* row_p = tile_p + (g + 8) * 64;
            atomicAdd(&row_p[local_c0 + 0], d2_0);
            atomicAdd(&row_p[local_c0 + 1], d3_0);
            atomicAdd(&row_p[local_c1 + 0], d2_1);
            atomicAdd(&row_p[local_c1 + 1], d3_1);
          }
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) {
          s_is_last = (atomicAdd(&sk_done[cur_n], seg_kgs) + seg_kgs == num_kg) ? 1 : 0;
        }
        __syncthreads();
        if (s_is_last) {
          if (g < M) {
            float* row_p = tile_p + g * 64;
            const float s0_0 = atomicExch(&row_p[local_c0 + 0], 0.0f);
            const float s1_0 = atomicExch(&row_p[local_c0 + 1], 0.0f);
            const float s0_1 = atomicExch(&row_p[local_c1 + 0], 0.0f);
            const float s1_1 = atomicExch(&row_p[local_c1 + 1], 0.0f);
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) = int4_bf16x2_u32(s0_0, s1_0);
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col1) = int4_bf16x2_u32(s0_1, s1_1);
          }
          if constexpr (kM16) {
            if (has_hi) {
              float* row_p = tile_p + (g + 8) * 64;
              const float s2_0 = atomicExch(&row_p[local_c0 + 0], 0.0f);
              const float s3_0 = atomicExch(&row_p[local_c0 + 1], 0.0f);
              const float s2_1 = atomicExch(&row_p[local_c1 + 0], 0.0f);
              const float s3_1 = atomicExch(&row_p[local_c1 + 1], 0.0f);
              *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * N + col0) = int4_bf16x2_u32(s2_0, s3_0);
              *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * N + col1) = int4_bf16x2_u32(s2_1, s3_1);
            }
          }
          __syncthreads();
          if (tid == 0) {
            atomicExch(&sk_done[cur_n], 0);
          }
        }
      }
      d0_0 = d1_0 = d2_0 = d3_0 = 0.0f;
      d0_1 = d1_1 = d2_1 = d3_1 = 0.0f;
      ++cur_n;
      cur_kg = 0;
      seg_start_kg = 0;
    } else {
      ++cur_kg;
    }
  }
}

int int4_sk_grid() {
  static const int g = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_SK_GRID");
    if (!s || !*s) return 144;
    const int v = std::atoi(s);
    return (v >= 0 && v <= 576) ? v : 144;
  }();
  return g;
}

int int4_pdl_mode() {
  static const int mode = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_PDL");
    if (!s || !*s) return 3;
    return (std::atoi(s) != 0) ? 3 : 0;
  }();
  return mode;
}

template <typename... KArgs, typename... Args>
static inline void int4_launch_pdl(
    void (*kernel)(KArgs...), int grid, int block, cudaStream_t st, int pdl_mode, Args... args) {
  cudaLaunchConfig_t lcfg = {};
  lcfg.gridDim = dim3(static_cast<unsigned>(grid), 1u, 1u);
  lcfg.blockDim = dim3(static_cast<unsigned>(block), 1u, 1u);
  lcfg.dynamicSmemBytes = 0;
  lcfg.stream = st;
  cudaLaunchAttribute attr[1];
  if ((pdl_mode & 1) != 0) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    lcfg.attrs = attr;
    lcfg.numAttrs = 1;
  }
  cudaError_t err = cudaLaunchKernelEx(&lcfg, kernel, static_cast<KArgs>(args)...);
  TORCH_CHECK(err == cudaSuccess, "int4_mma_streamk_kernel launch failed: ", cudaGetErrorString(err));
}

void ensure_int4_sk_scratch(const torch::TensorOptions& opts, int n_blocks,
                            float*& partials_ptr, int*& done_ptr) {
  static torch::Tensor g_sk_partials;
  static torch::Tensor g_sk_done;
  static int g_sk_cap = 0;
  static uint32_t g_sk_slot = 0;
  if (!g_sk_partials.defined() || g_sk_partials.device() != opts.device() || g_sk_cap < n_blocks) {
    const int cap = (n_blocks > 1024) ? n_blocks : 1024;
    g_sk_partials = torch::zeros({2, static_cast<int64_t>(cap) * 16 * 64}, opts.dtype(torch::kFloat32));
    g_sk_done = torch::zeros({2, cap}, opts.dtype(torch::kInt32));
    g_sk_cap = cap;
  }
  const int slot = static_cast<int>(g_sk_slot & 1u);
  g_sk_slot ^= 1u;
  partials_ptr = g_sk_partials.data_ptr<float>() + static_cast<size_t>(slot) * static_cast<size_t>(g_sk_cap) * 16 * 64;
  done_ptr = g_sk_done.data_ptr<int>() + static_cast<size_t>(slot) * static_cast<size_t>(g_sk_cap);
}

torch::Tensor launch_int4_gemm(
    const __nv_bfloat16* A_ptr,
    Aux5Desc aux5,
    bool is_aux5,
    torch::Tensor W_packed,
    const torch::TensorOptions& opts,
    int M, int64_t N, int64_t K) {
  auto C = torch::empty({M, N}, opts);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n_blocks = static_cast<int>(N / kRowsPerCTA);
  const bool use_m16 = (M > 8);
  const int sk_grid = int4_sk_grid();
  if (is_aux5 && sk_grid > 0) {
    const int total_iters = n_blocks * static_cast<int>(K >> 6);
    const int grid = (sk_grid < total_iters) ? sk_grid : total_iters;
    const int pdl_mode = int4_pdl_mode();
    float* p_ptr = nullptr;
    int* d_ptr = nullptr;
    ensure_int4_sk_scratch(opts, n_blocks, p_ptr, d_ptr);
    if (is_aux5) {
      if (use_m16) {
        int4_launch_pdl(
            int4_mma_streamk_kernel<true, true>, grid, kThreads, st, pdl_mode,
            static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode);
      } else {
        int4_launch_pdl(
            int4_mma_streamk_kernel<false, true>, grid, kThreads, st, pdl_mode,
            static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode);
      }
    } else {
      if (use_m16) {
        int4_launch_pdl(
            int4_mma_streamk_kernel<true, false>, grid, kThreads, st, pdl_mode,
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode);
      } else {
        int4_launch_pdl(
            int4_mma_streamk_kernel<false, false>, grid, kThreads, st, pdl_mode,
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode);
      }
    }
    return C;
  }
  static torch::Tensor g_partials;
  static int64_t g_partials_cap = 0;
  int split_k = 1;
  if (n_blocks <= 128 && K >= 16384) {
    split_k = (K >= 28672) ? 4 : 3;
  }
  if (split_k == 1) {
    if (is_aux5) {
      if (use_m16) {
        int4_mma_gemm_kernel<false, true, true><<<n_blocks, kThreads, 0, st>>>(
            nullptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            nullptr, M, static_cast<int>(N), static_cast<int>(K), 1);
      } else {
        int4_mma_gemm_kernel<false, false, true><<<n_blocks, kThreads, 0, st>>>(
            nullptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            nullptr, M, static_cast<int>(N), static_cast<int>(K), 1);
      }
    } else {
      if (use_m16) {
        int4_mma_gemm_kernel<false, true, false><<<n_blocks, kThreads, 0, st>>>(
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            nullptr, M, static_cast<int>(N), static_cast<int>(K), 1);
      } else {
        int4_mma_gemm_kernel<false, false, false><<<n_blocks, kThreads, 0, st>>>(
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            nullptr, M, static_cast<int>(N), static_cast<int>(K), 1);
      }
    }
  } else {
    const int64_t need = static_cast<int64_t>(split_k) * M * N;
    const int64_t alloc = (need > 4 * 16 * 6656) ? need : (4 * 16 * 6656);
    if (!g_partials.defined() || g_partials.device() != W_packed.device() || g_partials_cap < need) {
      g_partials = torch::empty({alloc}, opts.dtype(torch::kFloat32));
      g_partials_cap = alloc;
    }
    if (is_aux5) {
      if (use_m16) {
        int4_mma_gemm_kernel<true, true, true><<<dim3(n_blocks, split_k, 1), kThreads, 0, st>>>(
            nullptr, aux5, W_packed.data_ptr<uint8_t>(),
            nullptr, g_partials.data_ptr<float>(),
            M, static_cast<int>(N), static_cast<int>(K), split_k);
      } else {
        int4_mma_gemm_kernel<true, false, true><<<dim3(n_blocks, split_k, 1), kThreads, 0, st>>>(
            nullptr, aux5, W_packed.data_ptr<uint8_t>(),
            nullptr, g_partials.data_ptr<float>(),
            M, static_cast<int>(N), static_cast<int>(K), split_k);
      }
    } else {
      if (use_m16) {
        int4_mma_gemm_kernel<true, true, false><<<dim3(n_blocks, split_k, 1), kThreads, 0, st>>>(
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            nullptr, g_partials.data_ptr<float>(),
            M, static_cast<int>(N), static_cast<int>(K), split_k);
      } else {
        int4_mma_gemm_kernel<true, false, false><<<dim3(n_blocks, split_k, 1), kThreads, 0, st>>>(
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            nullptr, g_partials.data_ptr<float>(),
            M, static_cast<int>(N), static_cast<int>(K), split_k);
      }
    }
    const int total = M * static_cast<int>(N);
    int4_reduce_splitk_kernel<<<(total + 255) / 256, 256, 0, st>>>(
        g_partials.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, static_cast<int>(N), split_k);
  }
  return C;
}

}  // namespace

torch::Tensor int4_pack(torch::Tensor W, bool mse_opt = true) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kBFloat16 && W.dim() == 2 && W.is_contiguous(),
              "int4_pack expects contiguous CUDA BF16 [N, K]");
  const int N = static_cast<int>(W.size(0));
  const int K = static_cast<int>(W.size(1));
  TORCH_CHECK(N % kRowsPerCTA == 0 && K % 64 == 0, "int4_pack requires N%64==0 and K%64==0");
  const int64_t total_bytes = static_cast<int64_t>(N / 8) * static_cast<int64_t>(K / 64) * kTileBytes;
  auto out = torch::empty({total_bytes}, W.options().dtype(torch::kUInt8));
  dim3 grid(N / 8, K / 64);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  if (mse_opt) {
    int4_pack_kernel<true><<<grid, 32, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()),
        out.data_ptr<uint8_t>(),
        N, K);
  } else {
    int4_pack_kernel<false><<<grid, 32, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()),
        out.data_ptr<uint8_t>(),
        N, K);
  }
  return out;
}

torch::Tensor int4_gemm(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous(),
              "int4_gemm expects contiguous CUDA BF16 [M, K]");
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && A.size(1) == K && N % kRowsPerCTA == 0 && K % 64 == 0,
              "int4_gemm shape mismatch");
  Aux5Desc dummy{};
  return launch_int4_gemm(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      dummy, false, W_packed, A.options(), M, N, K);
}

torch::Tensor int4_gemm_aux5(
    torch::Tensor h0, torch::Tensor h1, torch::Tensor h2, torch::Tensor h3, torch::Tensor h4,
    torch::Tensor W_packed, int64_t N, int64_t K) {
  const torch::Tensor hs[5] = {h0, h1, h2, h3, h4};
  const int M = static_cast<int>(h0.size(0));
  const int64_t k_sub = h0.size(1);
  TORCH_CHECK(M >= 1 && M <= 16 && k_sub > 0 && k_sub % 64 == 0 && K == 5 * k_sub && N % kRowsPerCTA == 0,
              "int4_gemm_aux5 shape mismatch");
  Aux5Desc desc{};
  desc.kg_sub = static_cast<int>(k_sub >> 6);
  for (int i = 0; i < 5; ++i) {
    TORCH_CHECK(hs[i].is_cuda() && hs[i].scalar_type() == torch::kBFloat16 && hs[i].dim() == 2 &&
                hs[i].size(0) == M && hs[i].size(1) == k_sub &&
                hs[i].stride(1) == 1 && (hs[i].stride(0) % 8 == 0) &&
                (reinterpret_cast<uintptr_t>(hs[i].data_ptr()) % 16 == 0),
                "int4_gemm_aux5 expects 16B-aligned CUDA BF16 [M, k_sub] tensors");
    desc.p[i] = reinterpret_cast<const __nv_bfloat16*>(hs[i].data_ptr());
    desc.stride[i] = hs[i].stride(0);
  }
  return launch_int4_gemm(nullptr, desc, true, W_packed, h0.options(), M, N, K);
}
