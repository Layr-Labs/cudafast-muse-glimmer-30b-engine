// SPDX-License-Identifier: Apache-2.0
// muse_int3_mma: 3-bit asymmetric group-64 W3A16 weight-only Tensor-Core GEMM for the pinned DFlash drafter.
// Format (3.50 bits/weight, 0 wasted bits, 16-byte aligned):
//   Each 8x64 tile (8 output rows x 64 K cols) is 224 bytes (14 x uint4):
//     [0..127]   uint32_t lo2[32]  : for lane l=(r<<2)|t, 16 x 2-bit low bits (s=0..3, 4 weights/k16)
//     [128..191] uint16_t hi1[32]  : for lane l=(r<<2)|t, 16 x 1-bit high bits (s=0..3, 4 weights/k16)
//     [192..223] __nv_bfloat162 sz[8] : (scale, zero) for rows r=0..7 (w = q * scale + zero, q in 0..7)
//   Tiles are stored in [N/64, K/64, 8] order (1792 bytes = 112 x uint4 per 4-warp CTA per K=64 step).
//   Runs on SM121 with 6-stage cp.async (L2::evict_first), 1 barrier/step, 2x N-tile A reuse, and 2 MiB TLB prefetch.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <stdint.h>

namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * 32;    // 128
constexpr int kNTilesPerWarp = 2;                // 16 N-rows per warp -> 2x A reuse!
constexpr int kNTilesPerCTA = kWarpsPerBlock * kNTilesPerWarp;  // 8
constexpr int kRowsPerCTA = kNTilesPerCTA * 8;   // 64
constexpr int kTileBytes = 224;                  // 14 * 16
constexpr int kStepWBytes = kNTilesPerCTA * kTileBytes;  // 1792 = 112 * 16
constexpr int kStepWU4 = kStepWBytes / 16;       // 112
constexpr int kStages = 4;

template <bool kNf3>
__device__ __forceinline__ float int3_deq_f32(uint32_t q) {
  if constexpr (kNf3) {
    // Exact 8-level Lloyd-Max Normal(0,1) quantiles {0, 48, 83, 113, 142, 172, 207, 255}.
    // Byte 0 of 0x71533000u is 0x00, so PRMT with q in 0..7 auto-zeros bytes 1..3 in 1 cycle!
    const uint32_t b = __byte_perm(0x71533000u, 0xFFCFAC8Eu, q);
    return __uint_as_float(0x4B000000u | b) - 8388608.0f;
  } else {
    return __uint_as_float(0x4B000000u | q) - 8388608.0f;
  }
}

__device__ __forceinline__ int int3_quant_nf3(float u) {
  if (u < 127.5f) {
    if (u < 65.5f) return (u < 24.0f) ? 0 : 1;
    return (u < 98.0f) ? 2 : 3;
  } else {
    if (u < 189.5f) return (u < 157.0f) ? 4 : 5;
    return (u < 231.0f) ? 6 : 7;
  }
}

__device__ __forceinline__ void int3_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void int3_mma_m16n8k16_full(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint64_t int3_pol_ef() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

__device__ __forceinline__ void int3_cp_async16_ef(uint32_t sd, const void* gs, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(sd), "l"(gs), "l"(pol));
}

__device__ __forceinline__ void int3_cp_async16_cg(uint32_t sd, const void* gs) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ void int3_cp_async16_ca(uint32_t sd, const void* gs) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ uint32_t int3_bf16x2_u32(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float int3_subwarp4_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}

// GPU packer: BF16 [N, K] (N%64==0, K%64==0) -> uint8 [N/64, K/64, 8, 224]
// Grid: (N/8, K/64), Block: 32 threads (1 warp per 8x64 tile).
// When kMseOpt is true, optimizes per-group (scale, zero) via warp-parallel clipping search
// (symmetric midpoint, mean+-std, and one-sided clips) + 5-iteration closed-form ALS
// minimizing exact BF16-dequantized MSE (monotonically <= naive min-max on every group).
template <bool kMseOpt, bool kNf3 = true>
__global__ void int3_pack_kernel(
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

  constexpr float kInvMax = kNf3 ? (1.0f / 255.0f) : (1.0f / 7.0f);
  float sc_mm = (rmax - rmin) * kInvMax;
  if (sc_mm < 1e-30f) sc_mm = 1e-30f;
  __nv_bfloat162 sz_bf = __floats2bfloat162_rn(sc_mm, rmin);

  auto quant_val = [&](float v, float zr_q, float inv_sc) -> int {
    if constexpr (kNf3) {
      return int3_quant_nf3((v - zr_q) * inv_sc);
    } else {
      int q = __float2int_rn((v - zr_q) * inv_sc);
      if (q < 0) q = 0;
      if (q > 7) q = 7;
      return q;
    }
  };

  if constexpr (kMseOpt) {
    float lsum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      lsum += vals[i];
    }
    const float sw = int3_subwarp4_sum(lsum);
    const float mu = sw * (1.0f / 64.0f);

    float lvar = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float d = vals[i] - mu;
      lvar = __fmaf_rn(d, d, lvar);
    }
    const float std_v = sqrtf(fmaxf(int3_subwarp4_sum(lvar) * (1.0f / 64.0f), 0.0f));

    auto eval_cand = [&](float sc_in, float zr_in, __nv_bfloat162& out_sz, float& out_mse) {
      if (!(sc_in > 1e-30f)) sc_in = 1e-30f;
      out_sz = __floats2bfloat162_rn(sc_in, zr_in);
      const float sc_q = __bfloat162float(out_sz.x);
      const float zr_q = __bfloat162float(out_sz.y);
      const float inv_sc = (sc_q > 1e-30f) ? (1.0f / sc_q) : 0.0f;
      float err = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int q = quant_val(vals[i], zr_q, inv_sc);
        const float fq = int3_deq_f32<kNf3>(static_cast<uint32_t>(q));
        const float w_hat = __bfloat162float(
            __float2bfloat16_rn(__fmaf_rn(fq, sc_q, zr_q)));
        const float diff = vals[i] - w_hat;
        err = __fmaf_rn(diff, diff, err);
      }
      out_mse = int3_subwarp4_sum(err);
    };

    __nv_bfloat162 best_sz = sz_bf;
    float best_mse = 0.0f;
    eval_cand(sc_mm, rmin, best_sz, best_mse);

    __nv_bfloat162 try_sz;
    float try_mse;
    auto consider_interval = [&](float lo, float hi) {
      eval_cand((hi - lo) * kInvMax, lo, try_sz, try_mse);
      if (try_mse < best_mse) {
        best_mse = try_mse;
        best_sz = try_sz;
      }
    };

    // Symmetric midpoint shrinks (1.00 already evaluated as baseline min-max).
    const float mid = 0.5f * (rmin + rmax);
    const float half = 0.5f * (rmax - rmin);
    constexpr float kMidAlphas[5] = {0.96f, 0.92f, 0.84f, 0.76f, 0.68f};
#pragma unroll
    for (int c = 0; c < 5; ++c) {
      const float h = half * kMidAlphas[c];
      consider_interval(mid - h, mid + h);
    }

    // 6 mean +/- c*std clips (immune to heavy-tail outliers on either side).
    constexpr float kStdCs[6] = {2.90f, 2.60f, 2.30f, 2.00f, 1.75f, 1.50f};
#pragma unroll
    for (int c = 0; c < 6; ++c) {
      const float d = std_v * kStdCs[c];
      consider_interval(fmaxf(rmin, mu - d), fminf(rmax, mu + d));
    }

    // 4 one-sided clips (for asymmetric single-sided outlier groups).
    constexpr float kOneSideCs[2] = {2.20f, 1.80f};
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const float d = std_v * kOneSideCs[c];
      consider_interval(rmin, fminf(rmax, mu + d));
      consider_interval(fmaxf(rmin, mu - d), rmax);
    }

    // 5 iterations of exact closed-form ALS (1D linear regression on deq(q) -> (scale, zero)).
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
        const int q = quant_val(vals[i], cur_zr, inv_cur);
        const float fq = int3_deq_f32<kNf3>(static_cast<uint32_t>(q));
        lsq += fq;
        lsqq = __fmaf_rn(fq, fq, lsqq);
        lsqw = __fmaf_rn(fq, vals[i], lsqw);
      }
      const float sq = int3_subwarp4_sum(lsq);
      const float sqq = int3_subwarp4_sum(lsqq);
      const float sqw = int3_subwarp4_sum(lsqw);
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

    // Broadcast winning sz_bf from lane t==0 so all 4 threads pack with bit-identical (scale, zero).
    uint32_t sz_bits = *reinterpret_cast<const uint32_t*>(&best_sz);
    sz_bits = __shfl_sync(0xffffffffu, sz_bits, r << 2);
    sz_bf = *reinterpret_cast<const __nv_bfloat162*>(&sz_bits);
  }

  const float sc_q = __bfloat162float(sz_bf.x);
  const float zr_q = __bfloat162float(sz_bf.y);
  const float inv_sc = (sc_q > 1e-30f) ? (1.0f / sc_q) : 0.0f;

  uint32_t lo2 = 0u;
  uint32_t hi1 = 0u;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int q = quant_val(vals[i], zr_q, inv_sc);
    lo2 |= (static_cast<uint32_t>(q & 3) << (2 * i));
    hi1 |= (static_cast<uint32_t>((q >> 2) & 1) << i);
  }

  const int cta_n = n_tile >> 3;         // 0 .. N/64 - 1
  const int sub_tile = n_tile & 7;       // 0 .. 7
  const int num_kg = K >> 6;
  const size_t tile_idx = (static_cast<size_t>(cta_n) * num_kg + kg) * kNTilesPerCTA + sub_tile;
  uint8_t* dst = out + tile_idx * kTileBytes;

  reinterpret_cast<uint32_t*>(dst)[lane] = lo2;
  reinterpret_cast<uint16_t*>(dst + 128)[lane] = static_cast<uint16_t>(hi1);
  if (t == 0) {
    reinterpret_cast<__nv_bfloat162*>(dst + 192)[r] = sz_bf;
  }
}

// Paired gate_up packer: identical per-8-row independent (scale, zero) & MSE/ALS quantization
// (zero SNR loss vs int3_pack!), but interleaves gate tile (sub_tile = 2*w) and matching up tile
// (sub_tile = 2*w + 1) inside each CTA so one warp computes both gate and up in registers and
// writes SiLU(gate) * up directly without an intermediate [M, 2*I] BF16 round-trip.
template <bool kMseOpt, bool kNf3 = true>
__global__ void int3_pack_swiglu_kernel(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ out,
    int I, int K) {
  const int pair_tile = blockIdx.x;      // 0 .. 2*(I/8) - 1; even=gate, odd=up
  const int kg = blockIdx.y;             // 0 .. K/64 - 1
  const int lane = threadIdx.x;          // 0 .. 31
  const int r = lane >> 2;               // 0 .. 7
  const int t = lane & 3;                // 0 .. 3
  const int g_tile = pair_tile >> 1;     // 0 .. I/8 - 1
  const int is_up = pair_tile & 1;       // 0=gate, 1=up
  const int row = (is_up ? I : 0) + g_tile * 8 + r;
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

  constexpr float kInvMax = kNf3 ? (1.0f / 255.0f) : (1.0f / 7.0f);
  float sc_mm = (rmax - rmin) * kInvMax;
  if (sc_mm < 1e-30f) sc_mm = 1e-30f;
  __nv_bfloat162 sz_bf = __floats2bfloat162_rn(sc_mm, rmin);

  auto quant_val = [&](float v, float zr_q, float inv_sc) -> int {
    if constexpr (kNf3) {
      return int3_quant_nf3((v - zr_q) * inv_sc);
    } else {
      int q = __float2int_rn((v - zr_q) * inv_sc);
      if (q < 0) q = 0;
      if (q > 7) q = 7;
      return q;
    }
  };

  if constexpr (kMseOpt) {
    float lsum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) lsum += vals[i];
    const float sw = int3_subwarp4_sum(lsum);
    const float mu = sw * (1.0f / 64.0f);

    float lvar = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float d = vals[i] - mu;
      lvar = __fmaf_rn(d, d, lvar);
    }
    const float std_v = sqrtf(fmaxf(int3_subwarp4_sum(lvar) * (1.0f / 64.0f), 0.0f));

    auto eval_cand = [&](float sc_in, float zr_in, __nv_bfloat162& out_sz, float& out_mse) {
      if (!(sc_in > 1e-30f)) sc_in = 1e-30f;
      out_sz = __floats2bfloat162_rn(sc_in, zr_in);
      const float sc_q = __bfloat162float(out_sz.x);
      const float zr_q = __bfloat162float(out_sz.y);
      const float inv_sc = (sc_q > 1e-30f) ? (1.0f / sc_q) : 0.0f;
      float err = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int q = quant_val(vals[i], zr_q, inv_sc);
        const float fq = int3_deq_f32<kNf3>(static_cast<uint32_t>(q));
        const float w_hat = __bfloat162float(
            __float2bfloat16_rn(__fmaf_rn(fq, sc_q, zr_q)));
        const float diff = vals[i] - w_hat;
        err = __fmaf_rn(diff, diff, err);
      }
      out_mse = int3_subwarp4_sum(err);
    };

    __nv_bfloat162 best_sz = sz_bf;
    float best_mse = 0.0f;
    eval_cand(sc_mm, rmin, best_sz, best_mse);

    __nv_bfloat162 try_sz;
    float try_mse;
    auto consider_interval = [&](float lo, float hi) {
      eval_cand((hi - lo) * kInvMax, lo, try_sz, try_mse);
      if (try_mse < best_mse) {
        best_mse = try_mse;
        best_sz = try_sz;
      }
    };

    const float mid = 0.5f * (rmin + rmax);
    const float half = 0.5f * (rmax - rmin);
    constexpr float kMidAlphas[5] = {0.96f, 0.92f, 0.84f, 0.76f, 0.68f};
#pragma unroll
    for (int c = 0; c < 5; ++c) {
      const float h = half * kMidAlphas[c];
      consider_interval(mid - h, mid + h);
    }

    constexpr float kStdCs[6] = {2.90f, 2.60f, 2.30f, 2.00f, 1.75f, 1.50f};
#pragma unroll
    for (int c = 0; c < 6; ++c) {
      const float d = std_v * kStdCs[c];
      consider_interval(fmaxf(rmin, mu - d), fminf(rmax, mu + d));
    }

    constexpr float kOneSideCs[2] = {2.20f, 1.80f};
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const float d = std_v * kOneSideCs[c];
      consider_interval(rmin, fminf(rmax, mu + d));
      consider_interval(fmaxf(rmin, mu - d), rmax);
    }

    float cur_sc = __bfloat162float(best_sz.x);
    float cur_zr = __bfloat162float(best_sz.y);
#pragma unroll
    for (int it = 0; it < 5; ++it) {
      const float inv_cur = (cur_sc > 1e-30f) ? (1.0f / cur_sc) : 0.0f;
      float lsq = 0.0f, lsqq = 0.0f, lsqw = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int q = quant_val(vals[i], cur_zr, inv_cur);
        const float fq = int3_deq_f32<kNf3>(static_cast<uint32_t>(q));
        lsq += fq;
        lsqq = __fmaf_rn(fq, fq, lsqq);
        lsqw = __fmaf_rn(fq, vals[i], lsqw);
      }
      const float sq = int3_subwarp4_sum(lsq);
      const float sqq = int3_subwarp4_sum(lsqq);
      const float sqw = int3_subwarp4_sum(lsqw);
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

  uint32_t lo2 = 0u;
  uint32_t hi1 = 0u;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int q = quant_val(vals[i], zr_q, inv_sc);
    lo2 |= (static_cast<uint32_t>(q & 3) << (2 * i));
    hi1 |= (static_cast<uint32_t>((q >> 2) & 1) << i);
  }

  // cta_n in [0 .. I/32 - 1], warp_slot in [0 .. 3], sub_tile = warp_slot*2 + is_up.
  const int cta_n = g_tile >> 2;
  const int sub_tile = ((g_tile & 3) << 1) | is_up;
  const int num_kg = K >> 6;
  const size_t tile_idx = (static_cast<size_t>(cta_n) * num_kg + kg) * kNTilesPerCTA + sub_tile;
  uint8_t* dst = out + tile_idx * kTileBytes;

  reinterpret_cast<uint32_t*>(dst)[lane] = lo2;
  reinterpret_cast<uint16_t*>(dst + 128)[lane] = static_cast<uint16_t>(hi1);
  if (t == 0) {
    reinterpret_cast<__nv_bfloat162*>(dst + 192)[r] = sz_bf;
  }
}

template <int kMaxM>
struct alignas(16) Int3StageSmem {
  uint8_t w[kStages][kStepWBytes];         // 4 * 1792 = 7168 B
  __nv_bfloat16 a[kStages][kMaxM][72];     // 4 * 1152 = 4608 B (M<=8) or 9216 B (M<=16), bank-conflict-free 144B row stride
};                                         // Total = 11.5 KiB (M<=8) or 16.0 KiB (M<=16), 6 CTAs/SM

// Direct / split-K W3A16 Tensor-Core GEMM.
// kM16=false serves M=1..8 (11.5 KiB SMEM); kM16=true populates full m16n8k16 A rows 0..15 (16.0 KiB SMEM,
// plain cp.async.cg without L2::cache_hint to stay immune to SM121 odd-UR hazard) with zero extra weight reads.
template <bool kSplitK, bool kM16 = false, bool kNf3 = true>
__global__ void __launch_bounds__(128, 6) int3_mma_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ C_fp32,
    int M, int N, int K, int split_k, int pdl_mode) {
  __shared__ Int3StageSmem<kM16 ? 16 : 8> smem;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;
  uint64_t pol = 0ull;
  if constexpr (!kM16) {
    pol = int3_pol_ef();
  }

  const int cta_n = blockIdx.x;
  const int sk_idx = kSplitK ? blockIdx.y : 0;
  const int kg_per_sk = kSplitK ? ((num_kg + split_k - 1) / split_k) : num_kg;
  const int kg_start = kSplitK ? (sk_idx * kg_per_sk) : 0;
  int kg_end = kSplitK ? (kg_start + kg_per_sk) : num_kg;
  if (kg_end > num_kg) kg_end = num_kg;
  const int my_kgs = kg_end - kg_start;
  if (my_kgs <= 0) {
    if ((pdl_mode & 1) != 0) asm volatile("griddepcontrol.wait;" ::: "memory");
    if ((pdl_mode & 2) != 0) asm volatile("griddepcontrol.launch_dependents;");
    return;
  }

  const size_t cta_stride_bytes = static_cast<size_t>(num_kg) * kStepWBytes;
  const uint8_t* w_cta = W_packed + static_cast<size_t>(cta_n) * cta_stride_bytes;

  // Issue 2 MiB TLB prefetch from block (0, 0).
  if (cta_n == 0 && sk_idx == 0) {
    const size_t total_bytes = static_cast<size_t>(N / kRowsPerCTA) * cta_stride_bytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  auto issue_w_only = [&](int kg, int st) {
    if (tid < kStepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      if constexpr (kM16) {
        int3_cp_async16_cg(sd, w_cta + static_cast<size_t>(kg) * kStepWBytes + tid * 16);
      } else {
        int3_cp_async16_ef(sd, w_cta + static_cast<size_t>(kg) * kStepWBytes + tid * 16, pol);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  auto load_a_sync = [&](int kg, int st) {
    if constexpr (kM16) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        *reinterpret_cast<uint4*>(&smem.a[st][row][col8 * 8]) =
            __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(row) * K + kg * 64 + col8 * 8));
      }
    } else if (tid < 64) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        *reinterpret_cast<uint4*>(&smem.a[st][row][col8 * 8]) =
            __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(row) * K + kg * 64 + col8 * 8));
      }
    }
  };

  auto issue_stage = [&](int kg, int st) {
    if (tid < kStepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      if constexpr (kM16) {
        int3_cp_async16_cg(sd, w_cta + static_cast<size_t>(kg) * kStepWBytes + tid * 16);
      } else {
        int3_cp_async16_ef(sd, w_cta + static_cast<size_t>(kg) * kStepWBytes + tid * 16, pol);
      }
    }
    if constexpr (kM16) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][row][col8 * 8]));
        int3_cp_async16_ca(sd, A + static_cast<size_t>(row) * K + kg * 64 + col8 * 8);
      }
    } else if (tid < 64) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][row][col8 * 8]));
        int3_cp_async16_ca(sd, A + static_cast<size_t>(row) * K + kg * 64 + col8 * 8);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kStages - 1;  // 3
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
  const int trig_i = ((pdl_mode & 2) != 0) ? ((my_kgs > 4) ? (my_kgs - 4) : 0) : -1;

  int st = 0;
  for (int i = 0; i < my_kgs; ++i) {
    if (i == trig_i) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    const int rem = my_kgs - 1 - i;
    if (rem >= 2) {
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
    uint32_t lo2_0 = reinterpret_cast<const uint32_t*>(wt0)[lane];
    uint32_t hi1_0 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wt0 + 128)[lane]);
    const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 192)[g];
    const float sc0 = __bfloat162float(sz0.x);
    const float zr0 = __bfloat162float(sz0.y);

    uint32_t lo2_1 = reinterpret_cast<const uint32_t*>(wt1)[lane];
    uint32_t hi1_1 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wt1 + 128)[lane]);
    const __nv_bfloat162 sz1 = reinterpret_cast<const __nv_bfloat162*>(wt1 + 192)[g];
    const float sc1 = __bfloat162float(sz1.x);
    const float zr1 = __bfloat162float(sz1.y);

    const int next_i = i + kPreload;
    if (next_i < my_kgs) {
      const int next_st = (st + kPreload >= kStages) ? (st + kPreload - kStages) : (st + kPreload);
      issue_stage(kg_start + next_i, next_st);
    }

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t q0 = (lo2_0 & 3u) | ((hi1_0 & 1u) << 2);
      const uint32_t q1 = ((lo2_0 >> 2) & 3u) | (((hi1_0 >> 1) & 1u) << 2);
      const uint32_t q2 = ((lo2_0 >> 4) & 3u) | (((hi1_0 >> 2) & 1u) << 2);
      const uint32_t q3 = ((lo2_0 >> 6) & 3u) | (((hi1_0 >> 3) & 1u) << 2);
      lo2_0 >>= 8;
      hi1_0 >>= 4;
      const uint32_t b0 = int3_bf16x2_u32(__fmaf_rn(int3_deq_f32<kNf3>(q0), sc0, zr0),
                                          __fmaf_rn(int3_deq_f32<kNf3>(q1), sc0, zr0));
      const uint32_t b1 = int3_bf16x2_u32(__fmaf_rn(int3_deq_f32<kNf3>(q2), sc0, zr0),
                                          __fmaf_rn(int3_deq_f32<kNf3>(q3), sc0, zr0));
      if constexpr (kM16) {
        int3_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0_r[s], a1_r[s], a2_r[s], a3_r[s], b0, b1);
      } else {
        int3_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0_r[s], a2_r[s], b0, b1);
      }

      const uint32_t p0 = (lo2_1 & 3u) | ((hi1_1 & 1u) << 2);
      const uint32_t p1 = ((lo2_1 >> 2) & 3u) | (((hi1_1 >> 1) & 1u) << 2);
      const uint32_t p2 = ((lo2_1 >> 4) & 3u) | (((hi1_1 >> 2) & 1u) << 2);
      const uint32_t p3 = ((lo2_1 >> 6) & 3u) | (((hi1_1 >> 3) & 1u) << 2);
      lo2_1 >>= 8;
      hi1_1 >>= 4;
      const uint32_t c0 = int3_bf16x2_u32(__fmaf_rn(int3_deq_f32<kNf3>(p0), sc1, zr1),
                                          __fmaf_rn(int3_deq_f32<kNf3>(p1), sc1, zr1));
      const uint32_t c1 = int3_bf16x2_u32(__fmaf_rn(int3_deq_f32<kNf3>(p2), sc1, zr1),
                                          __fmaf_rn(int3_deq_f32<kNf3>(p3), sc1, zr1));
      if constexpr (kM16) {
        int3_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0_r[s], a1_r[s], a2_r[s], a3_r[s], c0, c1);
      } else {
        int3_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0_r[s], a2_r[s], c0, c1);
      }
    }

    st = (st + 1 == kStages) ? 0 : (st + 1);
  }

  const int col0 = cta_n * kRowsPerCTA + (warp * 2 + 0) * 8 + 2 * t;
  const int col1 = cta_n * kRowsPerCTA + (warp * 2 + 1) * 8 + 2 * t;
  if (g < M) {
    if constexpr (!kSplitK) {
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) = int3_bf16x2_u32(d0_0, d1_0);
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col1) = int3_bf16x2_u32(d0_1, d1_1);
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
        *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(ghi) * N + col0) = int3_bf16x2_u32(d2_0, d3_0);
        *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(ghi) * N + col1) = int3_bf16x2_u32(d2_1, d3_1);
      } else {
        float* p0 = C_fp32 + (static_cast<size_t>(sk_idx) * M + ghi) * N + col0;
        float* p1 = C_fp32 + (static_cast<size_t>(sk_idx) * M + ghi) * N + col1;
        p0[0] = d2_0; p0[1] = d3_0;
        p1[0] = d2_1; p1[1] = d3_1;
      }
    }
  }
}

__global__ void int3_reduce_splitk_kernel(
    const float* __restrict__ C_fp32,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int split_k, int pdl_mode) {
  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  const int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  const int total = M * N;
  if (idx4 >= total) {
    if ((pdl_mode & 2) != 0) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  for (int s = 0; s < split_k; ++s) {
    const float4 v = *reinterpret_cast<const float4*>(C_fp32 + static_cast<size_t>(s) * total + idx4);
    s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
  }
  uint2 out_u2;
  out_u2.x = int3_bf16x2_u32(s0, s1);
  out_u2.y = int3_bf16x2_u32(s2, s3);
  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
  *reinterpret_cast<uint2*>(C + idx4) = out_u2;
}

__device__ __forceinline__ float int3_swiglu_f32(float gate, float up) {
  return (gate / (1.0f + __expf(-gate))) * up;
}

// Fused paired gate_up + SiLU*Mul W3A16 GEMM (plain cp.async.cg, SM121 odd-UR safe, SM121 PDL enabled).
// Each CTA computes 32 intermediate outputs (4 warps x 8 cols): tile 2*w = gate, tile 2*w+1 = up.
template <bool kM16, bool kNf3 = true>
__global__ void __launch_bounds__(128, 6) int3_mma_gemm_swiglu_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    int M, int I, int K, int pdl_mode) {
  __shared__ Int3StageSmem<kM16 ? 16 : 8> smem;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;
  const int cta_n = blockIdx.x;

  const size_t cta_stride_bytes = static_cast<size_t>(num_kg) * kStepWBytes;
  const uint8_t* w_cta = W_packed + static_cast<size_t>(cta_n) * cta_stride_bytes;

  if (cta_n == 0) {
    const size_t total_bytes = static_cast<size_t>(I / 32) * cta_stride_bytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  auto issue_w_only = [&](int kg, int st) {
    if (tid < kStepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int3_cp_async16_cg(sd, w_cta + static_cast<size_t>(kg) * kStepWBytes + tid * 16);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  auto load_a_sync = [&](int kg, int st) {
    if constexpr (kM16) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        *reinterpret_cast<uint4*>(&smem.a[st][row][col8 * 8]) =
            __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(row) * K + kg * 64 + col8 * 8));
      }
    } else if (tid < 64) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        *reinterpret_cast<uint4*>(&smem.a[st][row][col8 * 8]) =
            __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(row) * K + kg * 64 + col8 * 8));
      }
    }
  };

  auto issue_stage = [&](int kg, int st) {
    if (tid < kStepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int3_cp_async16_cg(sd, w_cta + static_cast<size_t>(kg) * kStepWBytes + tid * 16);
    }
    if constexpr (kM16) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][row][col8 * 8]));
        int3_cp_async16_ca(sd, A + static_cast<size_t>(row) * K + kg * 64 + col8 * 8);
      }
    } else if (tid < 64) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][row][col8 * 8]));
        int3_cp_async16_ca(sd, A + static_cast<size_t>(row) * K + kg * 64 + col8 * 8);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kStages - 1;  // 3
  const int preload = (num_kg < kPreload) ? num_kg : kPreload;
  if ((pdl_mode & 1) != 0) {
    for (int i = 0; i < preload; ++i) {
      issue_w_only(i, i);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = 0; i < preload; ++i) {
      load_a_sync(i, i);
    }
  } else {
    for (int i = 0; i < preload; ++i) {
      issue_stage(i, i);
    }
  }

  float d0_g = 0.0f, d1_g = 0.0f, d2_g = 0.0f, d3_g = 0.0f;
  float d0_u = 0.0f, d1_u = 0.0f, d2_u = 0.0f, d3_u = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int trig_i = ((pdl_mode & 2) != 0) ? ((num_kg > 4) ? (num_kg - 4) : 0) : -1;

  int st = 0;
  for (int i = 0; i < num_kg; ++i) {
    if (i == trig_i) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    const int rem = num_kg - 1 - i;
    if (rem >= 2) {
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

    const uint8_t* wtg = &smem.w[st][(warp * 2 + 0) * kTileBytes];
    const uint8_t* wtu = &smem.w[st][(warp * 2 + 1) * kTileBytes];
    uint32_t lo2_g = reinterpret_cast<const uint32_t*>(wtg)[lane];
    uint32_t hi1_g = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wtg + 128)[lane]);
    const __nv_bfloat162 szg = reinterpret_cast<const __nv_bfloat162*>(wtg + 192)[g];
    const float scg = __bfloat162float(szg.x);
    const float zrg = __bfloat162float(szg.y);

    uint32_t lo2_u = reinterpret_cast<const uint32_t*>(wtu)[lane];
    uint32_t hi1_u = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wtu + 128)[lane]);
    const __nv_bfloat162 szu = reinterpret_cast<const __nv_bfloat162*>(wtu + 192)[g];
    const float scu = __bfloat162float(szu.x);
    const float zru = __bfloat162float(szu.y);

    const int next_i = i + kPreload;
    if (next_i < num_kg) {
      const int next_st = (st + kPreload >= kStages) ? (st + kPreload - kStages) : (st + kPreload);
      issue_stage(next_i, next_st);
    }

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t q0 = (lo2_g & 3u) | ((hi1_g & 1u) << 2);
      const uint32_t q1 = ((lo2_g >> 2) & 3u) | (((hi1_g >> 1) & 1u) << 2);
      const uint32_t q2 = ((lo2_g >> 4) & 3u) | (((hi1_g >> 2) & 1u) << 2);
      const uint32_t q3 = ((lo2_g >> 6) & 3u) | (((hi1_g >> 3) & 1u) << 2);
      lo2_g >>= 8;
      hi1_g >>= 4;
      const uint32_t b0 = int3_bf16x2_u32(__fmaf_rn(int3_deq_f32<kNf3>(q0), scg, zrg),
                                          __fmaf_rn(int3_deq_f32<kNf3>(q1), scg, zrg));
      const uint32_t b1 = int3_bf16x2_u32(__fmaf_rn(int3_deq_f32<kNf3>(q2), scg, zrg),
                                          __fmaf_rn(int3_deq_f32<kNf3>(q3), scg, zrg));
      if constexpr (kM16) {
        int3_mma_m16n8k16_full(d0_g, d1_g, d2_g, d3_g, a0_r[s], a1_r[s], a2_r[s], a3_r[s], b0, b1);
      } else {
        int3_mma_m16n8k16(d0_g, d1_g, d2_g, d3_g, a0_r[s], a2_r[s], b0, b1);
      }

      const uint32_t p0 = (lo2_u & 3u) | ((hi1_u & 1u) << 2);
      const uint32_t p1 = ((lo2_u >> 2) & 3u) | (((hi1_u >> 1) & 1u) << 2);
      const uint32_t p2 = ((lo2_u >> 4) & 3u) | (((hi1_u >> 2) & 1u) << 2);
      const uint32_t p3 = ((lo2_u >> 6) & 3u) | (((hi1_u >> 3) & 1u) << 2);
      lo2_u >>= 8;
      hi1_u >>= 4;
      const uint32_t c0 = int3_bf16x2_u32(__fmaf_rn(int3_deq_f32<kNf3>(p0), scu, zru),
                                          __fmaf_rn(int3_deq_f32<kNf3>(p1), scu, zru));
      const uint32_t c1 = int3_bf16x2_u32(__fmaf_rn(int3_deq_f32<kNf3>(p2), scu, zru),
                                          __fmaf_rn(int3_deq_f32<kNf3>(p3), scu, zru));
      if constexpr (kM16) {
        int3_mma_m16n8k16_full(d0_u, d1_u, d2_u, d3_u, a0_r[s], a1_r[s], a2_r[s], a3_r[s], c0, c1);
      } else {
        int3_mma_m16n8k16(d0_u, d1_u, d2_u, d3_u, a0_r[s], a2_r[s], c0, c1);
      }
    }

    st = (st + 1 == kStages) ? 0 : (st + 1);
  }

  const int col = cta_n * 32 + warp * 8 + 2 * t;
  if (g < M) {
    *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * I + col) =
        int3_bf16x2_u32(int3_swiglu_f32(d0_g, d0_u), int3_swiglu_f32(d1_g, d1_u));
  }
  if constexpr (kM16) {
    if (has_hi) {
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * I + col) =
          int3_bf16x2_u32(int3_swiglu_f32(d2_g, d2_u), int3_swiglu_f32(d3_g, d3_u));
    }
  }
}

// Persistent 144-CTA (3 CTAs/SM x 48 SMs) Stream-K W3A16 GEMM with continuous cross-tile cp.async.cg
// pipeline (zero pipeline drain across N-tile boundaries, zero tail-wave quantization, zero second
// reduce kernel launch, lock-free atomicAdd + self-zeroing atomicExch, SM121 odd-UR safe).
// kSwiglu=false: C is [M, N], n_blocks = N / 64.
// kSwiglu=true:  C is [M, I], n_blocks = I / 32 (tile 2*w = gate, tile 2*w+1 = up).
template <bool kM16, bool kSwiglu, bool kNf3 = true>
__global__ void __launch_bounds__(128, 6) int3_mma_streamk_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ sk_partials,  // [max_n_blocks, 16, 64] zero-initialized & self-zeroing
    int* __restrict__ sk_done,        // [max_n_blocks] zero-initialized & self-zeroing
    int M, int out_cols, int K, int n_blocks, int grid_ctas, int pdl_mode) {
  __shared__ Int3StageSmem<kM16 ? 16 : 8> smem;
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

  // Issue 2 MiB TLB prefetch from block 0 (immutable W_packed: safe before PDL wait).
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
  auto issue_w_only = [&](int st, int w_iter) {
    if (tid < kStepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int3_cp_async16_cg(sd, W_packed + static_cast<size_t>(w_iter) * kStepWBytes + tid * 16);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };
  auto load_a_sync = [&](int st, int a_kg) {
    if constexpr (kM16) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        *reinterpret_cast<uint4*>(&smem.a[st][row][col8 * 8]) =
            __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(row) * K + a_kg * 64 + col8 * 8));
      }
    } else if (tid < 64) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        *reinterpret_cast<uint4*>(&smem.a[st][row][col8 * 8]) =
            __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(row) * K + a_kg * 64 + col8 * 8));
      }
    }
  };
  auto issue_next = [&](int st) {
    if (tid < kStepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int3_cp_async16_cg(sd, W_packed + static_cast<size_t>(pre_iter) * kStepWBytes + tid * 16);
    }
    if constexpr (kM16) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][row][col8 * 8]));
        int3_cp_async16_ca(sd, A + static_cast<size_t>(row) * K + pre_kg * 64 + col8 * 8);
      }
    } else if (tid < 64) {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][row][col8 * 8]));
        int3_cp_async16_ca(sd, A + static_cast<size_t>(row) * K + pre_kg * 64 + col8 * 8);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    ++pre_iter;
    if (++pre_kg == num_kg) pre_kg = 0;
  };

  constexpr int kPreload = kStages - 1;  // 5
  const int preload = (my_iters < kPreload) ? my_iters : kPreload;
  if ((pdl_mode & 1) != 0) {
    for (int i = 0; i < preload; ++i) {
      issue_w_only(i, pre_iter + i);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = 0; i < preload; ++i) {
      load_a_sync(i, pre_kg);
      ++pre_iter;
      if (++pre_kg == num_kg) pre_kg = 0;
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
    if (rem >= 2) {
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
    uint32_t lo2_0 = reinterpret_cast<const uint32_t*>(wt0)[lane];
    uint32_t hi1_0 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wt0 + 128)[lane]);
    const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 192)[g];
    const float sc0 = __bfloat162float(sz0.x);
    const float zr0 = __bfloat162float(sz0.y);

    uint32_t lo2_1 = reinterpret_cast<const uint32_t*>(wt1)[lane];
    uint32_t hi1_1 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wt1 + 128)[lane]);
    const __nv_bfloat162 sz1 = reinterpret_cast<const __nv_bfloat162*>(wt1 + 192)[g];
    const float sc1 = __bfloat162float(sz1.x);
    const float zr1 = __bfloat162float(sz1.y);

    if (i + kPreload < my_iters) {
      const int next_st = (st + kPreload >= kStages) ? (st + kPreload - kStages) : (st + kPreload);
      issue_next(next_st);
    }

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t q0 = (lo2_0 & 3u) | ((hi1_0 & 1u) << 2);
      const uint32_t q1 = ((lo2_0 >> 2) & 3u) | (((hi1_0 >> 1) & 1u) << 2);
      const uint32_t q2 = ((lo2_0 >> 4) & 3u) | (((hi1_0 >> 2) & 1u) << 2);
      const uint32_t q3 = ((lo2_0 >> 6) & 3u) | (((hi1_0 >> 3) & 1u) << 2);
      lo2_0 >>= 8;
      hi1_0 >>= 4;
      const uint32_t b0 = int3_bf16x2_u32(__fmaf_rn(int3_deq_f32<kNf3>(q0), sc0, zr0),
                                          __fmaf_rn(int3_deq_f32<kNf3>(q1), sc0, zr0));
      const uint32_t b1 = int3_bf16x2_u32(__fmaf_rn(int3_deq_f32<kNf3>(q2), sc0, zr0),
                                          __fmaf_rn(int3_deq_f32<kNf3>(q3), sc0, zr0));
      if constexpr (kM16) {
        int3_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0_r[s], a1_r[s], a2_r[s], a3_r[s], b0, b1);
      } else {
        int3_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0_r[s], a2_r[s], b0, b1);
      }

      const uint32_t p0 = (lo2_1 & 3u) | ((hi1_1 & 1u) << 2);
      const uint32_t p1 = ((lo2_1 >> 2) & 3u) | (((hi1_1 >> 1) & 1u) << 2);
      const uint32_t p2 = ((lo2_1 >> 4) & 3u) | (((hi1_1 >> 2) & 1u) << 2);
      const uint32_t p3 = ((lo2_1 >> 6) & 3u) | (((hi1_1 >> 3) & 1u) << 2);
      lo2_1 >>= 8;
      hi1_1 >>= 4;
      const uint32_t c0 = int3_bf16x2_u32(__fmaf_rn(int3_deq_f32<kNf3>(p0), sc1, zr1),
                                          __fmaf_rn(int3_deq_f32<kNf3>(p1), sc1, zr1));
      const uint32_t c1 = int3_bf16x2_u32(__fmaf_rn(int3_deq_f32<kNf3>(p2), sc1, zr1),
                                          __fmaf_rn(int3_deq_f32<kNf3>(p3), sc1, zr1));
      if constexpr (kM16) {
        int3_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0_r[s], a1_r[s], a2_r[s], a3_r[s], c0, c1);
      } else {
        int3_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0_r[s], a2_r[s], c0, c1);
      }
    }

    st = (st + 1 == kStages) ? 0 : (st + 1);

    const bool tile_done = (cur_kg + 1 == num_kg) || (i + 1 == my_iters);
    if (tile_done) {
      const int seg_kgs = (cur_kg + 1) - seg_start_kg;
      if (seg_kgs == num_kg) {
        // Full tile owned by this CTA: direct BF16 store (zero partials, zero atomics).
        if constexpr (kSwiglu) {
          const int col = cur_n * 32 + warp * 8 + 2 * t;
          if (g < M) {
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * out_cols + col) =
                int3_bf16x2_u32(int3_swiglu_f32(d0_0, d0_1), int3_swiglu_f32(d1_0, d1_1));
          }
          if constexpr (kM16) {
            if (has_hi) {
              *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * out_cols + col) =
                  int3_bf16x2_u32(int3_swiglu_f32(d2_0, d2_1), int3_swiglu_f32(d3_0, d3_1));
            }
          }
        } else {
          const int col0 = cur_n * kRowsPerCTA + local_c0;
          const int col1 = cur_n * kRowsPerCTA + local_c1;
          if (g < M) {
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * out_cols + col0) = int3_bf16x2_u32(d0_0, d1_0);
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * out_cols + col1) = int3_bf16x2_u32(d0_1, d1_1);
          }
          if constexpr (kM16) {
            if (has_hi) {
              *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * out_cols + col0) = int3_bf16x2_u32(d2_0, d3_0);
              *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * out_cols + col1) = int3_bf16x2_u32(d2_1, d3_1);
            }
          }
        }
      } else {
        // Partial tile split across consecutive CTAs: lock-free FP32 atomicAdd + last-arriving self-zeroing store.
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
            if constexpr (kSwiglu) {
              const int col = cur_n * 32 + warp * 8 + 2 * t;
              *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * out_cols + col) =
                  int3_bf16x2_u32(int3_swiglu_f32(s0_0, s0_1), int3_swiglu_f32(s1_0, s1_1));
            } else {
              const int col0 = cur_n * kRowsPerCTA + local_c0;
              const int col1 = cur_n * kRowsPerCTA + local_c1;
              *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * out_cols + col0) = int3_bf16x2_u32(s0_0, s1_0);
              *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * out_cols + col1) = int3_bf16x2_u32(s0_1, s1_1);
            }
          }
          if constexpr (kM16) {
            if (has_hi) {
              float* row_p = tile_p + (g + 8) * 64;
              const float s2_0 = atomicExch(&row_p[local_c0 + 0], 0.0f);
              const float s3_0 = atomicExch(&row_p[local_c0 + 1], 0.0f);
              const float s2_1 = atomicExch(&row_p[local_c1 + 0], 0.0f);
              const float s3_1 = atomicExch(&row_p[local_c1 + 1], 0.0f);
              if constexpr (kSwiglu) {
                const int col = cur_n * 32 + warp * 8 + 2 * t;
                *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * out_cols + col) =
                    int3_bf16x2_u32(int3_swiglu_f32(s2_0, s2_1), int3_swiglu_f32(s3_0, s3_1));
              } else {
                const int col0 = cur_n * kRowsPerCTA + local_c0;
                const int col1 = cur_n * kRowsPerCTA + local_c1;
                *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * out_cols + col0) = int3_bf16x2_u32(s2_0, s3_0);
                *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * out_cols + col1) = int3_bf16x2_u32(s2_1, s3_1);
              }
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

int int3_sk_grid() {
  static const int g = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_SK_GRID");
    if (!s || !*s) return 0;
    const int v = std::atoi(s);
    return (v >= 0 && v <= 576) ? v : 0;
  }();
  return g;
}

bool int3_nf3_enabled() {
  static const bool en = []() -> bool {
    const char* s = std::getenv("MUSE_INT3_NF3");
    if (!s || !*s) return true;
    return std::atoi(s) != 0;
  }();
  return en;
}

int int3_pdl_mode() {
  static const int mode = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_PDL");
    if (!s || !*s) return 0;
    return (std::atoi(s) != 0) ? 3 : 0;
  }();
  return mode;
}

template <typename... KArgs, typename... Args>
static inline void int3_launch_pdl_dim3(
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
  TORCH_CHECK(err == cudaSuccess, "int3_mma launch failed: ", cudaGetErrorString(err));
}

template <typename... KArgs, typename... Args>
static inline void int3_launch_pdl(
    void (*kernel)(KArgs...), int grid, int block, cudaStream_t st, int pdl_mode, Args... args) {
  int3_launch_pdl_dim3(kernel, dim3(static_cast<unsigned>(grid), 1u, 1u), block, st, pdl_mode, args...);
}

void ensure_int3_sk_scratch(const torch::TensorOptions& opts, int n_blocks,
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

template <bool kNf3>
static void int3_set_carveout_nf3() {
  cudaFuncSetAttribute(int3_mma_gemm_kernel<false, false, kNf3>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_gemm_kernel<false, true, kNf3>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_gemm_kernel<true, false, kNf3>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_gemm_kernel<true, true, kNf3>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_gemm_swiglu_kernel<false, kNf3>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_gemm_swiglu_kernel<true, kNf3>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_streamk_kernel<false, false, kNf3>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_streamk_kernel<true, false, kNf3>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_streamk_kernel<false, true, kNf3>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_streamk_kernel<true, true, kNf3>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

static void int3_ensure_carveout() {
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    int3_set_carveout_nf3<false>();
    int3_set_carveout_nf3<true>();
  }
}

template <bool kNf3>
static torch::Tensor launch_int3_gemm_impl(
    torch::Tensor A, torch::Tensor W_packed, int M, int64_t N, int64_t K) {
  auto C = torch::empty({M, N}, A.options());
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n_blocks = static_cast<int>(N / kRowsPerCTA);
  const bool use_m16 = (M > 8);
  const int sk_grid = int3_sk_grid();
  const int pdl_mode = int3_pdl_mode();
  if (sk_grid > 0) {
    const int total_iters = n_blocks * static_cast<int>(K >> 6);
    const int grid = (sk_grid < total_iters) ? sk_grid : total_iters;
    float* p_ptr = nullptr;
    int* d_ptr = nullptr;
    ensure_int3_sk_scratch(A.options(), n_blocks, p_ptr, d_ptr);
    if (use_m16) {
      int3_launch_pdl(
          int3_mma_streamk_kernel<true, false, kNf3>, grid, kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode);
    } else {
      int3_launch_pdl(
          int3_mma_streamk_kernel<false, false, kNf3>, grid, kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode);
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
    if (use_m16) {
      int3_launch_pdl(
          int3_mma_gemm_kernel<false, true, kNf3>, n_blocks, kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode);
    } else {
      int3_launch_pdl(
          int3_mma_gemm_kernel<false, false, kNf3>, n_blocks, kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode);
    }
  } else {
    const int64_t need = static_cast<int64_t>(split_k) * M * N;
    const int64_t alloc = (need > 4 * 16 * 6656) ? need : (4 * 16 * 6656);
    if (!g_partials.defined() || g_partials.device() != A.device() || g_partials_cap < need) {
      g_partials = torch::empty({alloc}, A.options().dtype(torch::kFloat32));
      g_partials_cap = alloc;
    }
    const dim3 sk_dim(static_cast<unsigned>(n_blocks), static_cast<unsigned>(split_k), 1u);
    if (use_m16) {
      int3_launch_pdl_dim3(
          int3_mma_gemm_kernel<true, true, kNf3>, sk_dim, kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          static_cast<__nv_bfloat16*>(nullptr), g_partials.data_ptr<float>(),
          M, static_cast<int>(N), static_cast<int>(K), split_k, pdl_mode);
    } else {
      int3_launch_pdl_dim3(
          int3_mma_gemm_kernel<true, false, kNf3>, sk_dim, kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          static_cast<__nv_bfloat16*>(nullptr), g_partials.data_ptr<float>(),
          M, static_cast<int>(N), static_cast<int>(K), split_k, pdl_mode);
    }
    const int total4 = (M * static_cast<int>(N)) >> 2;
    int3_launch_pdl(
        int3_reduce_splitk_kernel, (total4 + 255) / 256, 256, st, pdl_mode,
        g_partials.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, static_cast<int>(N), split_k, pdl_mode);
  }
  return C;
}

template <bool kNf3>
static torch::Tensor launch_int3_gemm_swiglu_impl(
    torch::Tensor A, torch::Tensor W_packed, int M, int64_t I, int64_t K) {
  auto C = torch::empty({M, I}, A.options());
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n_blocks = static_cast<int>(I / 32);
  const int sk_grid = int3_sk_grid();
  const int pdl_mode = int3_pdl_mode();
  if (sk_grid > 0) {
    const int total_iters = n_blocks * static_cast<int>(K >> 6);
    const int grid = (sk_grid < total_iters) ? sk_grid : total_iters;
    float* p_ptr = nullptr;
    int* d_ptr = nullptr;
    ensure_int3_sk_scratch(A.options(), n_blocks, p_ptr, d_ptr);
    if (M > 8) {
      int3_launch_pdl(
          int3_mma_streamk_kernel<true, true, kNf3>, grid, kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, d_ptr, M, static_cast<int>(I), static_cast<int>(K), n_blocks, grid, pdl_mode);
    } else {
      int3_launch_pdl(
          int3_mma_streamk_kernel<false, true, kNf3>, grid, kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, d_ptr, M, static_cast<int>(I), static_cast<int>(K), n_blocks, grid, pdl_mode);
    }
    return C;
  }
  if (M > 8) {
    int3_launch_pdl(
        int3_mma_gemm_swiglu_kernel<true, kNf3>, n_blocks, kThreads, st, pdl_mode,
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, static_cast<int>(I), static_cast<int>(K), pdl_mode);
  } else {
    int3_launch_pdl(
        int3_mma_gemm_swiglu_kernel<false, kNf3>, n_blocks, kThreads, st, pdl_mode,
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, static_cast<int>(I), static_cast<int>(K), pdl_mode);
  }
  return C;
}

}  // namespace

torch::Tensor int3_pack(torch::Tensor W, bool mse_opt = true) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kBFloat16 && W.dim() == 2 && W.is_contiguous(),
              "int3_pack expects contiguous CUDA BF16 [N, K]");
  const int N = static_cast<int>(W.size(0));
  const int K = static_cast<int>(W.size(1));
  TORCH_CHECK(N % kRowsPerCTA == 0 && K % 64 == 0, "int3_pack requires N%64==0 and K%64==0");
  const int64_t total_bytes = static_cast<int64_t>(N / 8) * static_cast<int64_t>(K / 64) * kTileBytes;
  auto out = torch::empty({total_bytes}, W.options().dtype(torch::kUInt8));
  dim3 grid(N / 8, K / 64);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const bool nf3 = int3_nf3_enabled();
  if (nf3) {
    if (mse_opt) {
      int3_pack_kernel<true, true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K);
    } else {
      int3_pack_kernel<false, true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K);
    }
  } else {
    if (mse_opt) {
      int3_pack_kernel<true, false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K);
    } else {
      int3_pack_kernel<false, false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K);
    }
  }
  return out;
}

torch::Tensor int3_gemm(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous(),
              "int3_gemm expects contiguous CUDA BF16 [M, K]");
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && A.size(1) == K && N % kRowsPerCTA == 0 && K % 64 == 0,
              "int3_gemm shape mismatch");
  int3_ensure_carveout();
  if (int3_nf3_enabled()) {
    return launch_int3_gemm_impl<true>(A, W_packed, M, N, K);
  }
  return launch_int3_gemm_impl<false>(A, W_packed, M, N, K);
}

torch::Tensor int3_pack_swiglu(torch::Tensor W, bool mse_opt = true) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kBFloat16 && W.dim() == 2 && W.is_contiguous(),
              "int3_pack_swiglu expects contiguous CUDA BF16 [2*I, K]");
  const int N2 = static_cast<int>(W.size(0));
  const int K = static_cast<int>(W.size(1));
  TORCH_CHECK(N2 % 64 == 0 && K % 64 == 0, "int3_pack_swiglu requires (2*I)%64==0 and K%64==0");
  const int I = N2 / 2;
  const int64_t total_bytes = static_cast<int64_t>(N2 / 8) * static_cast<int64_t>(K / 64) * kTileBytes;
  auto out = torch::empty({total_bytes}, W.options().dtype(torch::kUInt8));
  dim3 grid(N2 / 8, K / 64);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const bool nf3 = int3_nf3_enabled();
  if (nf3) {
    if (mse_opt) {
      int3_pack_swiglu_kernel<true, true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), I, K);
    } else {
      int3_pack_swiglu_kernel<false, true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), I, K);
    }
  } else {
    if (mse_opt) {
      int3_pack_swiglu_kernel<true, false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), I, K);
    } else {
      int3_pack_swiglu_kernel<false, false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), I, K);
    }
  }
  return out;
}

torch::Tensor int3_gemm_swiglu(torch::Tensor A, torch::Tensor W_packed, int64_t I, int64_t K) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous(),
              "int3_gemm_swiglu expects contiguous CUDA BF16 [M, K]");
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && A.size(1) == K && I % 32 == 0 && K % 64 == 0,
              "int3_gemm_swiglu shape mismatch");
  int3_ensure_carveout();
  if (int3_nf3_enabled()) {
    return launch_int3_gemm_swiglu_impl<true>(A, W_packed, M, I, K);
  }
  return launch_int3_gemm_swiglu_impl<false>(A, W_packed, M, I, K);
}
