// SPDX-License-Identifier: Apache-2.0
// muse_int4_mma: 4-bit asymmetric group-64 W4A16 weight-only Tensor-Core GEMM for the pinned DFlash drafter.
// Format (4.50 bits/weight, 0 wasted bits, 16-byte aligned):
//   Each 8x64 tile (8 output rows x 64 K cols) is 288 bytes (18 x uint4):
//     [0..127]   uint32_t q_lo[32] : for lane l=(r<<2)|t, 8 x 4-bit unsigned codes (s=0..1, 4 weights/k16)
//     [128..255] uint32_t q_hi[32] : for lane l=(r<<2)|t, 8 x 4-bit unsigned codes (s=2..3, 4 weights/k16)
//     [256..287] __nv_bfloat162 sz[8] : (scale, zero) for rows r=0..7 (w = q * scale + zero, q in 0..15)
//   Tiles are stored in [N/64, K/64, 8] order (2304 bytes = 144 x uint4 per 4-warp CTA per K=64 step).
//   Runs on SM121 with 5-stage cp.async.cg.L2::256B (no L2::cache_hint / createpolicy -> SM121 odd-UR safe),
//   1 barrier/step, 2x N-tile A reuse, XOR-swizzled ldmatrix, BF16x2 HADD2+HFMA2 dequant, 4 CTAs/SM,
//   dual SM121 PDL, 2 MiB TLB prefetch, native M<=16 m16n8k16, and zero-cat aux5 fc input.
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
constexpr int kStages = 5;                                       // 21.25 KiB SMEM (M<=16) -> 4 CTAs/SM (192 CTAs)
constexpr int kWaveCTAs = 192;                                   // 4 CTAs/SM * 48 SMs on GB10

__device__ __host__ __forceinline__ size_t int4_wave_step_idx(
    int cta_n, int kg, int n_blocks, int num_kg) {
  const int wave = cta_n / kWaveCTAs;
  const int slot = cta_n - wave * kWaveCTAs;
  const int wave_base = wave * kWaveCTAs;
  const int w_size = (n_blocks - wave_base < kWaveCTAs) ? (n_blocks - wave_base) : kWaveCTAs;
  return static_cast<size_t>(wave_base) * num_kg + static_cast<size_t>(kg) * w_size + slot;
}

__device__ __host__ __forceinline__ size_t int4_streamk_step_idx(
    int cta_n, int kg, int n_blocks, int num_kg, int grid_ctas) {
  const int total_iters = n_blocks * num_kg;
  const int iter = cta_n * num_kg + kg;
  const int q = total_iters / grid_ctas;
  const int r = total_iters - q * grid_ctas;
  const int cut = r * (q + 1);
  int cta_id, step;
  if (iter < cut) {
    cta_id = iter / (q + 1);
    step = iter - cta_id * (q + 1);
  } else {
    const int rem = iter - cut;
    cta_id = r + rem / q;
    step = rem - (cta_id - r) * q;
  }
  return static_cast<size_t>(step) * grid_ctas + cta_id;
}

__host__ __forceinline__ bool int4_should_streamk(int n_blocks, int K, int sk_grid) {
  return (sk_grid > 0) && (n_blocks <= 128) && (K >= 28672);
}

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
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ void int4_cp_async16_ca_z(uint32_t sd, const void* gs, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(sd), "l"(gs), "r"(src_bytes));
}

__device__ __forceinline__ void int4_ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(smem_addr));
}

__device__ __forceinline__ void int4_ldmatrix_x2(
    uint32_t& r0, uint32_t& r1, uint32_t smem_addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(r0), "=r"(r1)
      : "r"(smem_addr));
}

__device__ __forceinline__ uint32_t int4_bf16x2_u32(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// Native BF16x2 4-bit dequant using Marlin-interleaved nibbles ((q0,q1) in [3:0]&[19:16], etc.):
// 0x41804180 | ((q << sh) & 0x00780078) == bf16x2(16+q0, 16+q1) exact -> HADD2(-16) -> HFMA2(sc, zr).
// Bit-exact with rn_bf16(fma(float(q), sc, zr)), 0 I2F, 0 F2FP.
template <bool kHiHalf>
__device__ __forceinline__ void int4_deq4_k16(
    uint32_t q32, __nv_bfloat162 sc2, __nv_bfloat162 zr2, uint32_t& b0, uint32_t& b1) {
  uint32_t t0, t1;
  if constexpr (!kHiHalf) {
    t0 = ((q32 << 3) & 0x00780078u) | 0x41804180u;
    t1 = ((q32 >> 1) & 0x00780078u) | 0x41804180u;
  } else {
    t0 = ((q32 >> 5) & 0x00780078u) | 0x41804180u;
    t1 = ((q32 >> 9) & 0x00780078u) | 0x41804180u;
  }
  const uint32_t neg16_u = 0xC180C180u;
  const __nv_bfloat162 neg16 = *reinterpret_cast<const __nv_bfloat162*>(&neg16_u);
  const __nv_bfloat162 q01 = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&t0), neg16);
  const __nv_bfloat162 q23 = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&t1), neg16);
  const __nv_bfloat162 w01 = __hfma2(q01, sc2, zr2);
  const __nv_bfloat162 w23 = __hfma2(q23, sc2, zr2);
  b0 = *reinterpret_cast<const uint32_t*>(&w01);
  b1 = *reinterpret_cast<const uint32_t*>(&w23);
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
    int N, int K, int ilv_mode, int sk_grid) {
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
    q_lo |= (static_cast<uint32_t>(q) << (((i >> 1) << 2) | ((i & 1) << 4)));
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int q = __float2int_rn((vals[8 + i] - zr_q) * inv_sc);
    if (q < 0) q = 0;
    if (q > 15) q = 15;
    q_hi |= (static_cast<uint32_t>(q) << (((i >> 1) << 2) | ((i & 1) << 4)));
  }

  const int cta_n = n_tile >> 3;
  const int sub_tile = n_tile & 7;
  const int n_blocks = N / kRowsPerCTA;
  const int num_kg = K >> 6;
  size_t step_slot;
  if (ilv_mode == 0) {
    step_slot = static_cast<size_t>(cta_n) * num_kg + kg;
  } else if (sk_grid > 0) {
    step_slot = int4_streamk_step_idx(cta_n, kg, n_blocks, num_kg, sk_grid);
  } else {
    step_slot = int4_wave_step_idx(cta_n, kg, n_blocks, num_kg);
  }
  const size_t tile_idx = step_slot * kNTilesPerCTA + sub_tile;
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
  uint8_t w[kStages][kStepWBytes];         // 5 * 2304 = 11520 B
  __nv_bfloat16 a[kStages][kMaxM][64];     // 5 * 1024 = 5120 B (M<=8) or 10240 B (M<=16)
};                                         // Total = 16.25 KiB (M<=8) or 21.25 KiB (M<=16), 4 CTAs/SM

template <bool kSplitK, bool kM16, bool kAux5>
__global__ void __launch_bounds__(128, 4) int4_mma_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    Aux5Desc aux5,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ C_fp32,
    int M, int N, int K, int split_k, int pdl_mode, int ilv_mode) {
  __shared__ Int4StageSmem<kM16 ? 16 : 8> smem;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;
  const int n_blocks = N / kRowsPerCTA;

  const int cta_n = blockIdx.x;
  const int sk_idx = kSplitK ? blockIdx.y : 0;
  const int kg_per_sk = kSplitK ? ((num_kg + split_k - 1) / split_k) : num_kg;
  const int kg_start = kSplitK ? (sk_idx * kg_per_sk) : 0;
  int kg_end = kSplitK ? (kg_start + kg_per_sk) : num_kg;
  if (kg_end > num_kg) kg_end = num_kg;
  const int my_kgs = kg_end - kg_start;
  if (my_kgs <= 0) return;

  const uint8_t* w_base;
  size_t w_kg_stride;
  if (ilv_mode != 0) {
    const int wave = cta_n / kWaveCTAs;
    const int slot = cta_n - wave * kWaveCTAs;
    const int wave_base = wave * kWaveCTAs;
    const int w_size = (n_blocks - wave_base < kWaveCTAs) ? (n_blocks - wave_base) : kWaveCTAs;
    w_base = W_packed + (static_cast<size_t>(wave_base) * num_kg + slot) * kStepWBytes;
    w_kg_stride = static_cast<size_t>(w_size) * kStepWBytes;
  } else {
    w_base = W_packed + static_cast<size_t>(cta_n) * num_kg * kStepWBytes;
    w_kg_stride = kStepWBytes;
  }

  // Issue 2 MiB TLB prefetch from block (0, 0) (immutable W_packed: safe before PDL wait).
  if (cta_n == 0 && sk_idx == 0) {
    const size_t total_bytes = static_cast<size_t>(n_blocks) * num_kg * kStepWBytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_swiz8 = a_col8 ^ (a_row & 7);
  const bool a_valid = a_row < M;

  auto issue_w_only = [&](int kg, int st) {
    const uint8_t* w_src = w_base + static_cast<size_t>(kg) * w_kg_stride;
    {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_cg(sd, w_src + tid * 16);
    }
    constexpr int kTailStart = kThreads - (kStepWU4 - kThreads);  // 112 (warp 3: no A-loads on M<=8)
    if (tid >= kTailStart) {
      const int idx = kThreads + (tid - kTailStart);
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][idx * 16]));
      int4_cp_async16_cg(sd, w_src + idx * 16);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  auto load_a_sync = [&](int kg, int st) {
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
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_swiz8 * 8]) =
          a_valid
              ? __ldg(reinterpret_cast<const uint4*>(
                    a_base + static_cast<size_t>(a_row) * a_stride + a_col_base + a_col8 * 8))
              : make_uint4(0u, 0u, 0u, 0u);
    } else if (tid < 64) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_swiz8 * 8]) =
          a_valid
              ? __ldg(reinterpret_cast<const uint4*>(
                    a_base + static_cast<size_t>(a_row) * a_stride + a_col_base + a_col8 * 8))
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  auto issue_stage = [&](int kg, int st) {
    const uint8_t* w_src = w_base + static_cast<size_t>(kg) * w_kg_stride;
    // 144 x 16B = 2304B across 128 threads: tid 0..127 load 0..127, tid 112..127 (warp 3) load 128..143.
    {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_cg(sd, w_src + tid * 16);
    }
    constexpr int kTailStart = kThreads - (kStepWU4 - kThreads);
    if (tid >= kTailStart) {
      const int idx = kThreads + (tid - kTailStart);
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
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid
                                    ? (a_base + static_cast<size_t>(a_row) * a_stride + a_col_base + a_col8 * 8)
                                    : a_base;
      int4_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid
                                    ? (a_base + static_cast<size_t>(a_row) * a_stride + a_col_base + a_col8 * 8)
                                    : a_base;
      int4_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kStages - 1;  // 4
  const int preload = (my_kgs < kPreload) ? my_kgs : kPreload;
  if ((pdl_mode & 1) != 0) {
    for (int i = 0; i < preload; ++i) {
      issue_w_only(kg_start + i, i);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = 0; i < preload; ++i) {
      load_a_sync(kg_start + i, i);
    }
  } else {
    for (int i = 0; i < preload; ++i) {
      issue_stage(kg_start + i, i);
    }
  }

  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int trig_i = ((pdl_mode & 2) != 0) ? ((my_kgs > 6) ? (my_kgs - 6) : 0) : -1;

  int st = 0;
  for (int i = 0; i < my_kgs; ++i) {
    if (i == trig_i) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    const int rem = my_kgs - 1 - i;
    if (rem >= 3) {
      asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    } else if (rem == 2) {
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    } else if (rem == 1) {
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();

    const int next_i = i + kPreload;
    if (next_i < my_kgs) {
      const int next_st = (st == 0) ? (kStages - 1) : (st - 1);
      issue_stage(kg_start + next_i, next_st);
    }

    const uint8_t* wt0 = &smem.w[st][(warp * 2 + 0) * kTileBytes];
    const uint8_t* wt1 = &smem.w[st][(warp * 2 + 1) * kTileBytes];
    const uint32_t q0_lo = reinterpret_cast<const uint32_t*>(wt0)[lane];
    const uint32_t q0_hi = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
    const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 256)[g];
    const __nv_bfloat162 sc0_2 = __bfloat162bfloat162(sz0.x);
    const __nv_bfloat162 zr0_2 = __bfloat162bfloat162(sz0.y);

    const uint32_t q1_lo = reinterpret_cast<const uint32_t*>(wt1)[lane];
    const uint32_t q1_hi = reinterpret_cast<const uint32_t*>(wt1 + 128)[lane];
    const __nv_bfloat162 sz1 = reinterpret_cast<const __nv_bfloat162*>(wt1 + 256)[g];
    const __nv_bfloat162 sc1_2 = __bfloat162bfloat162(sz1.x);
    const __nv_bfloat162 zr1_2 = __bfloat162bfloat162(sz1.y);

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        const int ldm_col8 = (2 * s + (lane >> 4)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 15][ldm_col8 * 8]));
        int4_ldmatrix_x4(a0, a1, a2, a3, a_addr);
      } else {
        const int ldm_col8 = (2 * s + ((lane >> 3) & 1)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 7][ldm_col8 * 8]));
        int4_ldmatrix_x2(a0, a2, a_addr);
      }
      uint32_t b0, b1, c0, c1;
      if ((s & 1) == 0) {
        int4_deq4_k16<false>((s < 2) ? q0_lo : q0_hi, sc0_2, zr0_2, b0, b1);
        int4_deq4_k16<false>((s < 2) ? q1_lo : q1_hi, sc1_2, zr1_2, c0, c1);
      } else {
        int4_deq4_k16<true>((s < 2) ? q0_lo : q0_hi, sc0_2, zr0_2, b0, b1);
        int4_deq4_k16<true>((s < 2) ? q1_lo : q1_hi, sc1_2, zr1_2, c0, c1);
      }
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0, a1, a2, a3, b0, b1);
        int4_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0, a1, a2, a3, c0, c1);
      } else {
        int4_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b0, b1);
        int4_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, c0, c1);
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

// Vectorized float4 -> bf16x4 split-K reduction with dual SM121 PDL (N % 64 == 0 => total % 4 == 0).
__global__ void int4_reduce_splitk_kernel(
    const float* __restrict__ C_fp32,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int split_k, int pdl_mode) {
  const int vidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int vtotal = (M * N) >> 2;
  if (vidx >= vtotal) return;
  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  const float4* src = reinterpret_cast<const float4*>(C_fp32);
  const size_t stride = static_cast<size_t>(vtotal);
  float4 acc = src[vidx];
  if (split_k == 2) {
    const float4 v1 = src[stride + vidx];
    acc.x += v1.x; acc.y += v1.y; acc.z += v1.z; acc.w += v1.w;
  } else if (split_k == 3) {
    const float4 v1 = src[stride + vidx];
    const float4 v2 = src[2 * stride + vidx];
    acc.x = (acc.x + v1.x) + v2.x; acc.y = (acc.y + v1.y) + v2.y;
    acc.z = (acc.z + v1.z) + v2.z; acc.w = (acc.w + v1.w) + v2.w;
  } else if (split_k == 4) {
    const float4 v1 = src[stride + vidx];
    const float4 v2 = src[2 * stride + vidx];
    const float4 v3 = src[3 * stride + vidx];
    acc.x = ((acc.x + v1.x) + v2.x) + v3.x; acc.y = ((acc.y + v1.y) + v2.y) + v3.y;
    acc.z = ((acc.z + v1.z) + v2.z) + v3.z; acc.w = ((acc.w + v1.w) + v2.w) + v3.w;
  } else {
    for (int s = 1; s < split_k; ++s) {
      const float4 vs = src[static_cast<size_t>(s) * stride + vidx];
      acc.x += vs.x; acc.y += vs.y; acc.z += vs.z; acc.w += vs.w;
    }
  }
  uint2 out_v;
  out_v.x = int4_bf16x2_u32(acc.x, acc.y);
  out_v.y = int4_bf16x2_u32(acc.z, acc.w);
  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
  reinterpret_cast<uint2*>(C)[vidx] = out_v;
}

// Persistent 192-CTA (4 CTAs/SM x 48 SMs) Stream-K W4A16 GEMM with continuous cross-tile cp.async.cg.L2::256B
// pipeline (zero pipeline drain across N-tile boundaries, zero tail-wave quantization, zero second
// reduce launch, lock-free atomicAdd + self-zeroing atomicExch, zero-div aux5 seg stepping, SM121 safe).
template <bool kM16, bool kAux5>
__global__ void __launch_bounds__(128, 4) int4_mma_streamk_kernel(
    const __nv_bfloat16* __restrict__ A,
    Aux5Desc aux5,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ sk_partials,  // [max_n_blocks, 16, 64] zero-initialized & self-zeroing
    int* __restrict__ sk_done,        // [max_n_blocks] zero-initialized & self-zeroing
    int M, int N, int K, int n_blocks, int grid_ctas, int pdl_mode, int ilv_mode) {
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
  const int q = total_iters / grid_ctas;
  const int r = total_iters - q * grid_ctas;
  const int my_iters = q + ((cta_id < r) ? 1 : 0);
  const int iter_start = (cta_id < r) ? (cta_id * (q + 1)) : (r * (q + 1) + (cta_id - r) * q);
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

  const uint8_t* w_ptr = (ilv_mode != 0)
                             ? (W_packed + static_cast<size_t>(cta_id) * kStepWBytes)
                             : (W_packed + static_cast<size_t>(iter_start) * kStepWBytes);
  const size_t w_step_bytes = static_cast<size_t>((ilv_mode != 0) ? grid_ctas : 1) * kStepWBytes;
  int pre_kg = cur_kg;
  int pre_seg = 0;
  int pre_local_kg = 0;
  if constexpr (kAux5) {
    pre_seg = cur_kg / aux5.kg_sub;
    pre_local_kg = cur_kg - pre_seg * aux5.kg_sub;
  }

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_swiz8 = a_col8 ^ (a_row & 7);
  const bool a_valid = a_row < M;

  auto issue_w_only = [&](int st, const uint8_t* w_src) {
    {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_cg(sd, w_src + tid * 16);
    }
    constexpr int kTailStart = kThreads - (kStepWU4 - kThreads);
    if (tid >= kTailStart) {
      const int idx = kThreads + (tid - kTailStart);
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][idx * 16]));
      int4_cp_async16_cg(sd, w_src + idx * 16);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };
  auto load_a_sync = [&](int st, const __nv_bfloat16* a_base, int64_t a_stride, int a_col_base) {
    if constexpr (kM16) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_swiz8 * 8]) =
          a_valid
              ? __ldg(reinterpret_cast<const uint4*>(
                    a_base + static_cast<size_t>(a_row) * a_stride + a_col_base + a_col8 * 8))
              : make_uint4(0u, 0u, 0u, 0u);
    } else if (tid < 64) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_swiz8 * 8]) =
          a_valid
              ? __ldg(reinterpret_cast<const uint4*>(
                    a_base + static_cast<size_t>(a_row) * a_stride + a_col_base + a_col8 * 8))
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  auto issue_next = [&](int st) {
    const uint8_t* w_src = w_ptr;
    {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_cg(sd, w_src + tid * 16);
    }
    constexpr int kTailStart = kThreads - (kStepWU4 - kThreads);
    if (tid >= kTailStart) {
      const int idx = kThreads + (tid - kTailStart);
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][idx * 16]));
      int4_cp_async16_cg(sd, w_src + idx * 16);
    }

    const __nv_bfloat16* a_base = A;
    int64_t a_stride = static_cast<int64_t>(K);
    int a_col_base = pre_kg * 64;
    if constexpr (kAux5) {
      a_base = aux5.p[pre_seg];
      a_stride = aux5.stride[pre_seg];
      a_col_base = pre_local_kg * 64;
    }

    if constexpr (kM16) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid
                                    ? (a_base + static_cast<size_t>(a_row) * a_stride + a_col_base + a_col8 * 8)
                                    : a_base;
      int4_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid
                                    ? (a_base + static_cast<size_t>(a_row) * a_stride + a_col_base + a_col8 * 8)
                                    : a_base;
      int4_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    w_ptr += w_step_bytes;
    if constexpr (kAux5) {
      if (++pre_local_kg == aux5.kg_sub) {
        pre_local_kg = 0;
        if (++pre_seg == 5) pre_seg = 0;
      }
    } else {
      if (++pre_kg == num_kg) pre_kg = 0;
    }
  };

  constexpr int kPreload = kStages - 1;  // 4
  const int preload = (my_iters < kPreload) ? my_iters : kPreload;
  if ((pdl_mode & 1) != 0) {
    for (int i = 0; i < preload; ++i) {
      issue_w_only(i, w_ptr + static_cast<size_t>(i) * w_step_bytes);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = 0; i < preload; ++i) {
      const __nv_bfloat16* a_base = A;
      int64_t a_stride = static_cast<int64_t>(K);
      int a_col_base = pre_kg * 64;
      if constexpr (kAux5) {
        a_base = aux5.p[pre_seg];
        a_stride = aux5.stride[pre_seg];
        a_col_base = pre_local_kg * 64;
      }
      load_a_sync(i, a_base, a_stride, a_col_base);
      w_ptr += w_step_bytes;
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
  const int trig_i = ((pdl_mode & 2) != 0) ? ((my_iters > 6) ? (my_iters - 6) : 0) : -1;

  int st = 0;
  for (int i = 0; i < my_iters; ++i) {
    if (i == trig_i) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    const int rem = my_iters - 1 - i;
    if (rem >= 3) {
      asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    } else if (rem == 2) {
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    } else if (rem == 1) {
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();

    if (i + kPreload < my_iters) {
      const int next_st = (st == 0) ? (kStages - 1) : (st - 1);
      issue_next(next_st);
    }

    const uint8_t* wt0 = &smem.w[st][(warp * 2 + 0) * kTileBytes];
    const uint8_t* wt1 = &smem.w[st][(warp * 2 + 1) * kTileBytes];
    const uint32_t q0_lo = reinterpret_cast<const uint32_t*>(wt0)[lane];
    const uint32_t q0_hi = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
    const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 256)[g];
    const __nv_bfloat162 sc0_2 = __bfloat162bfloat162(sz0.x);
    const __nv_bfloat162 zr0_2 = __bfloat162bfloat162(sz0.y);

    const uint32_t q1_lo = reinterpret_cast<const uint32_t*>(wt1)[lane];
    const uint32_t q1_hi = reinterpret_cast<const uint32_t*>(wt1 + 128)[lane];
    const __nv_bfloat162 sz1 = reinterpret_cast<const __nv_bfloat162*>(wt1 + 256)[g];
    const __nv_bfloat162 sc1_2 = __bfloat162bfloat162(sz1.x);
    const __nv_bfloat162 zr1_2 = __bfloat162bfloat162(sz1.y);

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        const int ldm_col8 = (2 * s + (lane >> 4)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 15][ldm_col8 * 8]));
        int4_ldmatrix_x4(a0, a1, a2, a3, a_addr);
      } else {
        const int ldm_col8 = (2 * s + ((lane >> 3) & 1)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 7][ldm_col8 * 8]));
        int4_ldmatrix_x2(a0, a2, a_addr);
      }
      uint32_t b0, b1, c0, c1;
      if ((s & 1) == 0) {
        int4_deq4_k16<false>((s < 2) ? q0_lo : q0_hi, sc0_2, zr0_2, b0, b1);
        int4_deq4_k16<false>((s < 2) ? q1_lo : q1_hi, sc1_2, zr1_2, c0, c1);
      } else {
        int4_deq4_k16<true>((s < 2) ? q0_lo : q0_hi, sc0_2, zr0_2, b0, b1);
        int4_deq4_k16<true>((s < 2) ? q1_lo : q1_hi, sc1_2, zr1_2, c0, c1);
      }
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0, a1, a2, a3, b0, b1);
        int4_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0, a1, a2, a3, c0, c1);
      } else {
        int4_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b0, b1);
        int4_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, c0, c1);
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

int int4_ilv_mode() {
  static const int mode = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_ILV");
    if (!s || !*s) return 1;
    return (std::atoi(s) != 0) ? 1 : 0;
  }();
  return mode;
}

int int4_sk_grid() {
  static const int g = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_SK_GRID");
    const int def = 144;  // 3 CTAs/SM beat 192 (627 vs 655 us in w49titan_12)
    if (!s || !*s) return def;
    const int v = std::atoi(s);
    return (v >= 0 && v <= 576) ? v : def;
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
    void (*kernel)(KArgs...), dim3 grid, int block, cudaStream_t st, int pdl_mode, Args... args) {
  cudaLaunchConfig_t lcfg = {};
  lcfg.gridDim = grid;
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
  TORCH_CHECK(err == cudaSuccess, "int4_mma kernel launch failed: ", cudaGetErrorString(err));
}

float* ensure_int4_splitk_partials(const torch::TensorOptions& opts, int64_t need) {
  static torch::Tensor g_partials;
  static int64_t g_partials_cap = 0;
  static uint32_t g_partials_slot = 0;
  const int64_t min_cap = 4 * 16 * 10240;
  const int64_t alloc = (need > min_cap) ? need : min_cap;
  if (!g_partials.defined() || g_partials.device() != opts.device() || g_partials_cap < need) {
    g_partials = torch::empty({2, alloc}, opts.dtype(torch::kFloat32));
    g_partials_cap = alloc;
  }
  const int slot = static_cast<int>(g_partials_slot & 1u);
  g_partials_slot ^= 1u;
  return g_partials.data_ptr<float>() + static_cast<size_t>(slot) * static_cast<size_t>(g_partials_cap);
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
  const int pdl_mode = int4_pdl_mode();
  const int ilv_mode = int4_ilv_mode();
  const int sk_grid = int4_sk_grid();
  if (int4_should_streamk(n_blocks, static_cast<int>(K), sk_grid)) {
    const int total_iters = n_blocks * static_cast<int>(K >> 6);
    const int grid = (sk_grid < total_iters) ? sk_grid : total_iters;
    float* p_ptr = nullptr;
    int* d_ptr = nullptr;
    ensure_int4_sk_scratch(opts, n_blocks, p_ptr, d_ptr);
    if (is_aux5) {
      if (use_m16) {
        int4_launch_pdl(
            int4_mma_streamk_kernel<true, true>, dim3(grid, 1, 1), kThreads, st, pdl_mode,
            static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode, ilv_mode);
      } else {
        int4_launch_pdl(
            int4_mma_streamk_kernel<false, true>, dim3(grid, 1, 1), kThreads, st, pdl_mode,
            static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode, ilv_mode);
      }
    } else {
      if (use_m16) {
        int4_launch_pdl(
            int4_mma_streamk_kernel<true, false>, dim3(grid, 1, 1), kThreads, st, pdl_mode,
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode, ilv_mode);
      } else {
        int4_launch_pdl(
            int4_mma_streamk_kernel<false, false>, dim3(grid, 1, 1), kThreads, st, pdl_mode,
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode, ilv_mode);
      }
    }
    return C;
  }
  int split_k = 1;
  if (n_blocks <= 128 && K >= 16384) {
    split_k = (K >= 28672) ? 4 : 3;
  }
  if (split_k == 1) {
    const dim3 grid(n_blocks, 1, 1);
    if (is_aux5) {
      if (use_m16) {
        int4_launch_pdl(
            int4_mma_gemm_kernel<false, true, true>, grid, kThreads, st, pdl_mode,
            static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode, ilv_mode);
      } else {
        int4_launch_pdl(
            int4_mma_gemm_kernel<false, false, true>, grid, kThreads, st, pdl_mode,
            static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode, ilv_mode);
      }
    } else {
      if (use_m16) {
        int4_launch_pdl(
            int4_mma_gemm_kernel<false, true, false>, grid, kThreads, st, pdl_mode,
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode, ilv_mode);
      } else {
        int4_launch_pdl(
            int4_mma_gemm_kernel<false, false, false>, grid, kThreads, st, pdl_mode,
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode, ilv_mode);
      }
    }
  } else {
    const int64_t need = static_cast<int64_t>(split_k) * M * N;
    float* partials_ptr = ensure_int4_splitk_partials(opts, need);
    const dim3 grid(n_blocks, split_k, 1);
    if (is_aux5) {
      if (use_m16) {
        int4_launch_pdl(
            int4_mma_gemm_kernel<true, true, true>, grid, kThreads, st, pdl_mode,
            static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
            static_cast<__nv_bfloat16*>(nullptr), partials_ptr,
            M, static_cast<int>(N), static_cast<int>(K), split_k, pdl_mode, ilv_mode);
      } else {
        int4_launch_pdl(
            int4_mma_gemm_kernel<true, false, true>, grid, kThreads, st, pdl_mode,
            static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
            static_cast<__nv_bfloat16*>(nullptr), partials_ptr,
            M, static_cast<int>(N), static_cast<int>(K), split_k, pdl_mode, ilv_mode);
      }
    } else {
      if (use_m16) {
        int4_launch_pdl(
            int4_mma_gemm_kernel<true, true, false>, grid, kThreads, st, pdl_mode,
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            static_cast<__nv_bfloat16*>(nullptr), partials_ptr,
            M, static_cast<int>(N), static_cast<int>(K), split_k, pdl_mode, ilv_mode);
      } else {
        int4_launch_pdl(
            int4_mma_gemm_kernel<true, false, false>, grid, kThreads, st, pdl_mode,
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            static_cast<__nv_bfloat16*>(nullptr), partials_ptr,
            M, static_cast<int>(N), static_cast<int>(K), split_k, pdl_mode, ilv_mode);
      }
    }
    const int vtotal = (M * static_cast<int>(N)) >> 2;
    const dim3 rgrid((vtotal + 255) / 256, 1, 1);
    int4_launch_pdl(
        int4_reduce_splitk_kernel, rgrid, 256, st, pdl_mode,
        static_cast<const float*>(partials_ptr),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, static_cast<int>(N), split_k, pdl_mode);
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
  const int n_blocks = N / kRowsPerCTA;
  const int ilv_mode = int4_ilv_mode();
  const int sk_raw = int4_sk_grid();
  const int total_iters = n_blocks * (K >> 6);
  const int sk_grid =
      int4_should_streamk(n_blocks, K, sk_raw) ? ((sk_raw < total_iters) ? sk_raw : total_iters) : 0;
  dim3 grid(N / 8, K / 64);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  if (mse_opt) {
    int4_pack_kernel<true><<<grid, 32, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()),
        out.data_ptr<uint8_t>(),
        N, K, ilv_mode, sk_grid);
  } else {
    int4_pack_kernel<false><<<grid, 32, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()),
        out.data_ptr<uint8_t>(),
        N, K, ilv_mode, sk_grid);
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
