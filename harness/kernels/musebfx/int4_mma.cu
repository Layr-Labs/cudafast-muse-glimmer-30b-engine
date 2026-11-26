// SPDX-License-Identifier: Apache-2.0
// muse_int4_mma: 4-bit asymmetric group-64 W4A16 (exact 16-level Lloyd-Max NF4) Tensor-Core GEMM for DFlash.
// Format (4.50 bits/weight, 0 wasted bits, 16-byte aligned):
//   Each 8x64 tile (8 output rows x 64 K cols) is 288 bytes (18 x uint4):
//     [0..127]   uint32_t q_lo[32] : for lane l=(r<<2)|t, 8 x 4-bit unsigned codes (s=0..1, 4 weights/k16)
//     [128..255] uint32_t q_hi[32] : for lane l=(r<<2)|t, 8 x 4-bit unsigned codes (s=2..3, 4 weights/k16)
//     [256..287] __nv_bfloat162 sz[8] : (scale, zero) for rows r=0..7
//   Runs on SM121 with 5-stage cp.async (.L2::256B, 0-UR-hazard), 4 CTAs/SM, XOR-swizzled ldmatrix,
//   2x N-tile A reuse, 2 MiB TLB prefetch, zero-cat aux5 fc input, and fused paired INT4-NF4 SwiGLU.
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
constexpr int kStages = 5;

template <bool kNf4>
__device__ __forceinline__ float int4_deq_f32(uint32_t q) {
  if constexpr (kNf4) {
    // Exact 16-level Lloyd-Max Normal(0,1) quantiles:
    // {0, 31, 52, 69, 84, 97, 109, 122, 133, 146, 158, 171, 186, 203, 224, 255}.
    // Antisymmetry: level[15 - k] == 255 - level[k] == level[k] ^ 0xFF!
    // Byte 0 of 0x45341F00u is 0x00, so PRMT auto-zeros bytes 1..3 in 1 cycle.
    const uint32_t s8 = (q & 8u) ? 0xFFu : 0u;
    const uint32_t b = __byte_perm(0x45341F00u, 0x7A6D6154u, (q ^ s8) & 7u);
    return __uint_as_float((0x4B000000u | s8) ^ b) - 8388608.0f;
  } else {
    return __uint_as_float(0x4B000000u | q) - 8388608.0f;
  }
}

__device__ __forceinline__ int int4_quant_nf4(float u) {
  if (u < 127.5f) {
    if (u < 76.5f) {
      if (u < 41.5f) return (u < 15.5f) ? 0 : 1;
      return (u < 60.5f) ? 2 : 3;
    } else {
      if (u < 103.0f) return (u < 90.5f) ? 4 : 5;
      return (u < 115.5f) ? 6 : 7;
    }
  } else {
    if (u < 178.5f) {
      if (u < 152.0f) return (u < 139.5f) ? 8 : 9;
      return (u < 164.5f) ? 10 : 11;
    } else {
      if (u < 213.5f) return (u < 194.5f) ? 12 : 13;
      return (u < 239.5f) ? 14 : 15;
    }
  }
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

// Standard 16B L2-cached weight cp.async (100% immune to SM121 .L2::256B UR hazard at all stage counts).
__device__ __forceinline__ void int4_cp_async16_w(uint32_t sd, const void* gs) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ void int4_cp_async16_cg(uint32_t sd, const void* gs) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ void int4_cp_async16_ca(uint32_t sd, const void* gs) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ void int4_cp_async16_a_pred(uint32_t sd, const void* gs, int src_sz) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(sd), "l"(gs), "r"(src_sz));
}

__device__ __forceinline__ void int4_ldmatrix_x2(uint32_t& a0, uint32_t& a2, uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(a0), "=r"(a2)
      : "r"(addr));
}

__device__ __forceinline__ void int4_ldmatrix_x4(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
      : "r"(addr));
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

__device__ __forceinline__ float int4_swiglu_f32(float gate, float up) {
  return (gate / (1.0f + __expf(-gate))) * up;
}

template <bool kMseOpt, bool kNf4 = true>
__global__ void int4_pack_kernel(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ out,
    int N, int K) {
  const int n_tile = blockIdx.x;
  const int kg = blockIdx.y;
  const int lane = threadIdx.x;
  const int r = lane >> 2;
  const int t = lane & 3;
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

  constexpr float kInvMax = kNf4 ? (1.0f / 255.0f) : (1.0f / 15.0f);
  float sc_mm = (rmax - rmin) * kInvMax;
  if (sc_mm < 1e-30f) sc_mm = 1e-30f;
  __nv_bfloat162 sz_bf = __floats2bfloat162_rn(sc_mm, rmin);

  auto quant_val = [&](float v, float zr_q, float inv_sc) -> int {
    if constexpr (kNf4) {
      return int4_quant_nf4((v - zr_q) * inv_sc);
    } else {
      int q = __float2int_rn((v - zr_q) * inv_sc);
      if (q < 0) q = 0;
      if (q > 15) q = 15;
      return q;
    }
  };

  if constexpr (kMseOpt) {
    float lsum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) lsum += vals[i];
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
        const int q = quant_val(vals[i], zr_q, inv_sc);
        const float fq = int4_deq_f32<kNf4>(static_cast<uint32_t>(q));
        const float w_hat = __bfloat162float(
            __float2bfloat16_rn(__fmaf_rn(fq, sc_q, zr_q)));
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
      eval_cand((hi - lo) * kInvMax, lo, try_sz, try_mse);
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

    float cur_sc = __bfloat162float(best_sz.x);
    float cur_zr = __bfloat162float(best_sz.y);
#pragma unroll
    for (int it = 0; it < 5; ++it) {
      const float inv_cur = (cur_sc > 1e-30f) ? (1.0f / cur_sc) : 0.0f;
      float lsq = 0.0f, lsqq = 0.0f, lsqw = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int q = quant_val(vals[i], cur_zr, inv_cur);
        const float fq = int4_deq_f32<kNf4>(static_cast<uint32_t>(q));
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
    const int q = quant_val(vals[i], zr_q, inv_sc);
    q_lo |= (static_cast<uint32_t>(q) << (4 * i));
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = quant_val(vals[8 + i], zr_q, inv_sc);
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

template <bool kMseOpt, bool kNf4 = true>
__global__ void int4_pack_swiglu_kernel(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ out,
    int I, int K) {
  const int pair_tile = blockIdx.x;
  const int kg = blockIdx.y;
  const int lane = threadIdx.x;
  const int r = lane >> 2;
  const int t = lane & 3;
  const int g_tile = pair_tile >> 1;
  const int is_up = pair_tile & 1;
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

  constexpr float kInvMax = kNf4 ? (1.0f / 255.0f) : (1.0f / 15.0f);
  float sc_mm = (rmax - rmin) * kInvMax;
  if (sc_mm < 1e-30f) sc_mm = 1e-30f;
  __nv_bfloat162 sz_bf = __floats2bfloat162_rn(sc_mm, rmin);

  auto quant_val = [&](float v, float zr_q, float inv_sc) -> int {
    if constexpr (kNf4) {
      return int4_quant_nf4((v - zr_q) * inv_sc);
    } else {
      int q = __float2int_rn((v - zr_q) * inv_sc);
      if (q < 0) q = 0;
      if (q > 15) q = 15;
      return q;
    }
  };

  if constexpr (kMseOpt) {
    float lsum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) lsum += vals[i];
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
        const int q = quant_val(vals[i], zr_q, inv_sc);
        const float fq = int4_deq_f32<kNf4>(static_cast<uint32_t>(q));
        const float w_hat = __bfloat162float(
            __float2bfloat16_rn(__fmaf_rn(fq, sc_q, zr_q)));
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
      eval_cand((hi - lo) * kInvMax, lo, try_sz, try_mse);
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

    float cur_sc = __bfloat162float(best_sz.x);
    float cur_zr = __bfloat162float(best_sz.y);
#pragma unroll
    for (int it = 0; it < 5; ++it) {
      const float inv_cur = (cur_sc > 1e-30f) ? (1.0f / cur_sc) : 0.0f;
      float lsq = 0.0f, lsqq = 0.0f, lsqw = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int q = quant_val(vals[i], cur_zr, inv_cur);
        const float fq = int4_deq_f32<kNf4>(static_cast<uint32_t>(q));
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
    const int q = quant_val(vals[i], zr_q, inv_sc);
    q_lo |= (static_cast<uint32_t>(q) << (4 * i));
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = quant_val(vals[8 + i], zr_q, inv_sc);
    q_hi |= (static_cast<uint32_t>(q) << (4 * i));
  }

  const int cta_n = g_tile >> 2;
  const int sub_tile = ((g_tile & 3) << 1) | is_up;
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
  uint8_t w[kStages][kStepWBytes];         // 5 * 2304 = 11520 B
  __nv_bfloat16 a[kStages][kMaxM][64];     // 5 * 1024 = 5120 B (M<=8) or 10240 B (M<=16), XOR-swizzled
};                                         // Total = 16.25 KiB (M<=8) or 21.25 KiB (M<=16) -> 4 CTAs/SM

template <bool kSplitK, bool kM16, bool kAux5, bool kNf4 = true>
__global__ void __launch_bounds__(128, 4) int4_mma_gemm_kernel(
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

  if (cta_n == 0 && sk_idx == 0) {
    const size_t total_bytes = static_cast<size_t>(N / kRowsPerCTA) * cta_stride_bytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_phys_col8 = a_col8 ^ (a_row & 7);
  const int a_src_sz = (a_row < M) ? 16 : 0;
  const int a_safe_row = (a_row < M) ? a_row : 0;

  auto issue_stage = [&](int kg, int st) {
    const uint8_t* w_src = w_cta + static_cast<size_t>(kg) * kStepWBytes;
    {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_w(sd, w_src + tid * 16);
    }
    if (tid < (kStepWU4 - kThreads)) {
      const int idx = kThreads + tid;
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][idx * 16]));
      int4_cp_async16_w(sd, w_src + idx * 16);
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
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int4_cp_async16_a_pred(sd, a_base + static_cast<size_t>(a_safe_row) * a_stride + a_col_base + a_col8 * 8, a_src_sz);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int4_cp_async16_a_pred(sd, a_base + static_cast<size_t>(a_safe_row) * a_stride + a_col_base + a_col8 * 8, a_src_sz);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kStages - 1;  // 4
  const int preload = (my_kgs < kPreload) ? my_kgs : kPreload;
  for (int i = 0; i < preload; ++i) {
    issue_stage(kg_start + i, i);
  }

  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int ldm_row = kM16 ? (lane & 15) : (lane & 7);
  const int ldm_col_off = kM16 ? (lane >> 4) : ((lane >> 3) & 1);

  int st = 0;
  for (int i = 0; i < my_kgs; ++i) {
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

    uint32_t a0_r[4], a2_r[4];
    uint32_t a1_r[4], a3_r[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][ldm_row][pcol8 * 8]));
      if constexpr (kM16) {
        int4_ldmatrix_x4(a0_r[s], a1_r[s], a2_r[s], a3_r[s], addr);
      } else {
        int4_ldmatrix_x2(a0_r[s], a2_r[s], addr);
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
          __fmaf_rn(int4_deq_f32<kNf4>(w0 & 15u), sc0, zr0),
          __fmaf_rn(int4_deq_f32<kNf4>((w0 >> 4) & 15u), sc0, zr0));
      const uint32_t b1 = int4_bf16x2_u32(
          __fmaf_rn(int4_deq_f32<kNf4>((w0 >> 8) & 15u), sc0, zr0),
          __fmaf_rn(int4_deq_f32<kNf4>((w0 >> 12) & 15u), sc0, zr0));
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0_r[s], a1_r[s], a2_r[s], a3_r[s], b0, b1);
      } else {
        int4_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0_r[s], a2_r[s], b0, b1);
      }

      const uint32_t w1 = (s < 2) ? (q1_lo >> (16 * s)) : (q1_hi >> (16 * (s - 2)));
      const uint32_t c0 = int4_bf16x2_u32(
          __fmaf_rn(int4_deq_f32<kNf4>(w1 & 15u), sc1, zr1),
          __fmaf_rn(int4_deq_f32<kNf4>((w1 >> 4) & 15u), sc1, zr1));
      const uint32_t c1 = int4_bf16x2_u32(
          __fmaf_rn(int4_deq_f32<kNf4>((w1 >> 8) & 15u), sc1, zr1),
          __fmaf_rn(int4_deq_f32<kNf4>((w1 >> 12) & 15u), sc1, zr1));
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

constexpr int kN32Stages = 4;
constexpr int kN32StepWBytes = 4 * kTileBytes;      // 1152 = 72 * 16
constexpr int kN32StepWU4 = kN32StepWBytes / 16;    // 72

template <int kMaxM>
struct alignas(16) Int4N32StageSmem {
  uint8_t w[kN32Stages][kN32StepWBytes];   // 4 * 1152 = 4608 B
  __nv_bfloat16 a[kN32Stages][kMaxM][64];  // 4 * 1024 = 4096 B (M<=8) or 8192 B (M<=16), XOR-swizzled
};                                         // Total = 8.5 KiB (M<=8) or 12.5 KiB (M<=16) -> 6 CTAs/SM (288 CTAs)!

template <bool kSplitK, bool kM16, bool kAux5, bool kNf4 = true>
__global__ void __launch_bounds__(128, 6) int4_mma_gemm_n32_kernel(
    const __nv_bfloat16* __restrict__ A,
    Aux5Desc aux5,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ C_fp32,
    int M, int N, int K, int split_k) {
  __shared__ Int4N32StageSmem<kM16 ? 16 : 8> smem;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;

  const int half_cta = blockIdx.x;
  const int cta_n = half_cta >> 1;
  const int half_off = (half_cta & 1) * kN32StepWBytes;
  const int sk_idx = kSplitK ? blockIdx.y : 0;
  const int kg_per_sk = kSplitK ? ((num_kg + split_k - 1) / split_k) : num_kg;
  const int kg_start = kSplitK ? (sk_idx * kg_per_sk) : 0;
  int kg_end = kSplitK ? (kg_start + kg_per_sk) : num_kg;
  if (kg_end > num_kg) kg_end = num_kg;
  const int my_kgs = kg_end - kg_start;
  if (my_kgs <= 0) return;

  const size_t cta_stride_bytes = static_cast<size_t>(num_kg) * kStepWBytes;
  const uint8_t* w_cta = W_packed + static_cast<size_t>(cta_n) * cta_stride_bytes + half_off;

  if (half_cta == 0 && sk_idx == 0) {
    const size_t total_bytes = static_cast<size_t>(N / kRowsPerCTA) * cta_stride_bytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_phys_col8 = a_col8 ^ (a_row & 7);
  const int a_src_sz = (a_row < M) ? 16 : 0;
  const int a_safe_row = (a_row < M) ? a_row : 0;

  auto issue_stage = [&](int kg, int st) {
    if (tid < kN32StepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_w(sd, w_cta + static_cast<size_t>(kg) * kStepWBytes + tid * 16);
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
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int4_cp_async16_a_pred(sd, a_base + static_cast<size_t>(a_safe_row) * a_stride + a_col_base + a_col8 * 8, a_src_sz);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int4_cp_async16_a_pred(sd, a_base + static_cast<size_t>(a_safe_row) * a_stride + a_col_base + a_col8 * 8, a_src_sz);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kN32Stages - 1;  // 3
  const int preload = (my_kgs < kPreload) ? my_kgs : kPreload;
  for (int i = 0; i < preload; ++i) {
    issue_stage(kg_start + i, i);
  }

  float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int ldm_row = kM16 ? (lane & 15) : (lane & 7);
  const int ldm_col_off = kM16 ? (lane >> 4) : ((lane >> 3) & 1);

  int st = 0;
  for (int i = 0; i < my_kgs; ++i) {
    const int rem = my_kgs - 1 - i;
    if (rem >= 2) {
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    } else if (rem == 1) {
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();

    uint32_t a0_r[4], a2_r[4];
    uint32_t a1_r[4], a3_r[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][ldm_row][pcol8 * 8]));
      if constexpr (kM16) {
        int4_ldmatrix_x4(a0_r[s], a1_r[s], a2_r[s], a3_r[s], addr);
      } else {
        int4_ldmatrix_x2(a0_r[s], a2_r[s], addr);
      }
    }

    const uint8_t* wt0 = &smem.w[st][warp * kTileBytes];
    const uint32_t q0_lo = reinterpret_cast<const uint32_t*>(wt0)[lane];
    const uint32_t q0_hi = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
    const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 256)[g];
    const float sc0 = __bfloat162float(sz0.x);
    const float zr0 = __bfloat162float(sz0.y);

    const int next_i = i + kPreload;
    if (next_i < my_kgs) {
      const int next_st = (st + kPreload >= kN32Stages) ? (st + kPreload - kN32Stages) : (st + kPreload);
      issue_stage(kg_start + next_i, next_st);
    }

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t w0 = (s < 2) ? (q0_lo >> (16 * s)) : (q0_hi >> (16 * (s - 2)));
      const uint32_t b0 = int4_bf16x2_u32(
          __fmaf_rn(int4_deq_f32<kNf4>(w0 & 15u), sc0, zr0),
          __fmaf_rn(int4_deq_f32<kNf4>((w0 >> 4) & 15u), sc0, zr0));
      const uint32_t b1 = int4_bf16x2_u32(
          __fmaf_rn(int4_deq_f32<kNf4>((w0 >> 8) & 15u), sc0, zr0),
          __fmaf_rn(int4_deq_f32<kNf4>((w0 >> 12) & 15u), sc0, zr0));
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0, d1, d2, d3, a0_r[s], a1_r[s], a2_r[s], a3_r[s], b0, b1);
      } else {
        int4_mma_m16n8k16(d0, d1, d2, d3, a0_r[s], a2_r[s], b0, b1);
      }
    }

    st = (st + 1 == kN32Stages) ? 0 : (st + 1);
  }

  const int col0 = half_cta * 32 + warp * 8 + 2 * t;
  if (g < M) {
    if constexpr (!kSplitK) {
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) = int4_bf16x2_u32(d0, d1);
    } else {
      float* p0 = C_fp32 + (static_cast<size_t>(sk_idx) * M + g) * N + col0;
      p0[0] = d0; p0[1] = d1;
    }
  }
  if constexpr (kM16) {
    if (has_hi) {
      const int ghi = g + 8;
      if constexpr (!kSplitK) {
        *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(ghi) * N + col0) = int4_bf16x2_u32(d2, d3);
      } else {
        float* p0 = C_fp32 + (static_cast<size_t>(sk_idx) * M + ghi) * N + col0;
        p0[0] = d2; p0[1] = d3;
      }
    }
  }
}

__global__ void int4_reduce_splitk_kernel(
    const float* __restrict__ C_fp32,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int split_k) {
  const int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  const int total = M * N;
  if (idx4 >= total) return;
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  for (int s = 0; s < split_k; ++s) {
    const float4 v = *reinterpret_cast<const float4*>(C_fp32 + static_cast<size_t>(s) * total + idx4);
    s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
  }
  uint2 out_u2;
  out_u2.x = int4_bf16x2_u32(s0, s1);
  out_u2.y = int4_bf16x2_u32(s2, s3);
  *reinterpret_cast<uint2*>(C + idx4) = out_u2;
}

template <bool kM16, bool kNf4 = true>
__global__ void __launch_bounds__(128, 4) int4_mma_gemm_swiglu_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    int M, int I, int K) {
  __shared__ Int4StageSmem<kM16 ? 16 : 8> smem;
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

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_phys_col8 = a_col8 ^ (a_row & 7);
  const int a_src_sz = (a_row < M) ? 16 : 0;
  const int a_safe_row = (a_row < M) ? a_row : 0;

  auto issue_stage = [&](int kg, int st) {
    const uint8_t* w_src = w_cta + static_cast<size_t>(kg) * kStepWBytes;
    {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_w(sd, w_src + tid * 16);
    }
    if (tid < (kStepWU4 - kThreads)) {
      const int idx = kThreads + tid;
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][idx * 16]));
      int4_cp_async16_w(sd, w_src + idx * 16);
    }
    if constexpr (kM16) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int4_cp_async16_a_pred(sd, A + static_cast<size_t>(a_safe_row) * K + kg * 64 + a_col8 * 8, a_src_sz);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int4_cp_async16_a_pred(sd, A + static_cast<size_t>(a_safe_row) * K + kg * 64 + a_col8 * 8, a_src_sz);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kStages - 1;  // 4
  const int preload = (num_kg < kPreload) ? num_kg : kPreload;
  for (int i = 0; i < preload; ++i) {
    issue_stage(i, i);
  }

  float d0_g = 0.0f, d1_g = 0.0f, d2_g = 0.0f, d3_g = 0.0f;
  float d0_u = 0.0f, d1_u = 0.0f, d2_u = 0.0f, d3_u = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int ldm_row = kM16 ? (lane & 15) : (lane & 7);
  const int ldm_col_off = kM16 ? (lane >> 4) : ((lane >> 3) & 1);

  int st = 0;
  for (int i = 0; i < num_kg; ++i) {
    const int rem = num_kg - 1 - i;
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

    uint32_t a0_r[4], a2_r[4];
    uint32_t a1_r[4], a3_r[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][ldm_row][pcol8 * 8]));
      if constexpr (kM16) {
        int4_ldmatrix_x4(a0_r[s], a1_r[s], a2_r[s], a3_r[s], addr);
      } else {
        int4_ldmatrix_x2(a0_r[s], a2_r[s], addr);
      }
    }

    const uint8_t* wtg = &smem.w[st][(warp * 2 + 0) * kTileBytes];
    const uint8_t* wtu = &smem.w[st][(warp * 2 + 1) * kTileBytes];
    const uint32_t qg_lo = reinterpret_cast<const uint32_t*>(wtg)[lane];
    const uint32_t qg_hi = reinterpret_cast<const uint32_t*>(wtg + 128)[lane];
    const __nv_bfloat162 szg = reinterpret_cast<const __nv_bfloat162*>(wtg + 256)[g];
    const float scg = __bfloat162float(szg.x);
    const float zrg = __bfloat162float(szg.y);

    const uint32_t qu_lo = reinterpret_cast<const uint32_t*>(wtu)[lane];
    const uint32_t qu_hi = reinterpret_cast<const uint32_t*>(wtu + 128)[lane];
    const __nv_bfloat162 szu = reinterpret_cast<const __nv_bfloat162*>(wtu + 256)[g];
    const float scu = __bfloat162float(szu.x);
    const float zru = __bfloat162float(szu.y);

    const int next_i = i + kPreload;
    if (next_i < num_kg) {
      const int next_st = (st + kPreload >= kStages) ? (st + kPreload - kStages) : (st + kPreload);
      issue_stage(next_i, next_st);
    }

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t wg = (s < 2) ? (qg_lo >> (16 * s)) : (qg_hi >> (16 * (s - 2)));
      const uint32_t b0 = int4_bf16x2_u32(
          __fmaf_rn(int4_deq_f32<kNf4>(wg & 15u), scg, zrg),
          __fmaf_rn(int4_deq_f32<kNf4>((wg >> 4) & 15u), scg, zrg));
      const uint32_t b1 = int4_bf16x2_u32(
          __fmaf_rn(int4_deq_f32<kNf4>((wg >> 8) & 15u), scg, zrg),
          __fmaf_rn(int4_deq_f32<kNf4>((wg >> 12) & 15u), scg, zrg));
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0_g, d1_g, d2_g, d3_g, a0_r[s], a1_r[s], a2_r[s], a3_r[s], b0, b1);
      } else {
        int4_mma_m16n8k16(d0_g, d1_g, d2_g, d3_g, a0_r[s], a2_r[s], b0, b1);
      }

      const uint32_t wu = (s < 2) ? (qu_lo >> (16 * s)) : (qu_hi >> (16 * (s - 2)));
      const uint32_t c0 = int4_bf16x2_u32(
          __fmaf_rn(int4_deq_f32<kNf4>(wu & 15u), scu, zru),
          __fmaf_rn(int4_deq_f32<kNf4>((wu >> 4) & 15u), scu, zru));
      const uint32_t c1 = int4_bf16x2_u32(
          __fmaf_rn(int4_deq_f32<kNf4>((wu >> 8) & 15u), scu, zru),
          __fmaf_rn(int4_deq_f32<kNf4>((wu >> 12) & 15u), scu, zru));
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0_u, d1_u, d2_u, d3_u, a0_r[s], a1_r[s], a2_r[s], a3_r[s], c0, c1);
      } else {
        int4_mma_m16n8k16(d0_u, d1_u, d2_u, d3_u, a0_r[s], a2_r[s], c0, c1);
      }
    }

    st = (st + 1 == kStages) ? 0 : (st + 1);
  }

  const int col = cta_n * 32 + warp * 8 + 2 * t;
  if (g < M) {
    *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * I + col) =
        int4_bf16x2_u32(int4_swiglu_f32(d0_g, d0_u), int4_swiglu_f32(d1_g, d1_u));
  }
  if constexpr (kM16) {
    if (has_hi) {
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * I + col) =
          int4_bf16x2_u32(int4_swiglu_f32(d2_g, d2_u), int4_swiglu_f32(d3_g, d3_u));
    }
  }
}

// Exact 74c3b0bc 6-stage persistent Stream-K W4A16 GEMM for aux5 fc (preserves exact CTA schedule & 41-step trajectory).
constexpr int kSkStages = 6;
template <int kMaxM>
struct alignas(16) Int4SkStageSmem {
  uint8_t w[kSkStages][kStepWBytes];
  __nv_bfloat16 a[kSkStages][kMaxM][64];
};

template <bool kM16, bool kAux5, bool kNf4 = false>
__global__ void __launch_bounds__(128, 3) int4_mma_streamk_kernel(
    const __nv_bfloat16* __restrict__ A,
    Aux5Desc aux5,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ sk_partials,
    int* __restrict__ sk_done,
    int M, int N, int K, int n_blocks, int grid_ctas, int pdl_mode) {
  __shared__ Int4SkStageSmem<kM16 ? 16 : 8> smem;
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

  constexpr int kPreload = kSkStages - 1;  // 5
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
      const int next_st = (st + kPreload >= kSkStages) ? (st + kPreload - kSkStages) : (st + kPreload);
      issue_next(next_st);
    }

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t w0 = (s < 2) ? (q0_lo >> (16 * s)) : (q0_hi >> (16 * (s - 2)));
      const uint32_t b0 = int4_bf16x2_u32(
          __fmaf_rn(int4_deq_f32<kNf4>(w0 & 15u), sc0, zr0),
          __fmaf_rn(int4_deq_f32<kNf4>((w0 >> 4) & 15u), sc0, zr0));
      const uint32_t b1 = int4_bf16x2_u32(
          __fmaf_rn(int4_deq_f32<kNf4>((w0 >> 8) & 15u), sc0, zr0),
          __fmaf_rn(int4_deq_f32<kNf4>((w0 >> 12) & 15u), sc0, zr0));
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0_r[s], a1_r[s], a2_r[s], a3_r[s], b0, b1);
      } else {
        int4_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0_r[s], a2_r[s], b0, b1);
      }

      const uint32_t w1 = (s < 2) ? (q1_lo >> (16 * s)) : (q1_hi >> (16 * (s - 2)));
      const uint32_t c0 = int4_bf16x2_u32(
          __fmaf_rn(int4_deq_f32<kNf4>(w1 & 15u), sc1, zr1),
          __fmaf_rn(int4_deq_f32<kNf4>((w1 >> 4) & 15u), sc1, zr1));
      const uint32_t c1 = int4_bf16x2_u32(
          __fmaf_rn(int4_deq_f32<kNf4>((w1 >> 8) & 15u), sc1, zr1),
          __fmaf_rn(int4_deq_f32<kNf4>((w1 >> 12) & 15u), sc1, zr1));
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0_r[s], a1_r[s], a2_r[s], a3_r[s], c0, c1);
      } else {
        int4_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0_r[s], a2_r[s], c0, c1);
      }
    }

    st = (st + 1 == kSkStages) ? 0 : (st + 1);

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

static bool int4_bench_dyn_env() {
  static const bool dyn = (std::getenv("MUSE_BENCH_DYN_ENV") != nullptr);
  return dyn;
}

bool int4_nf4_for_shape(int K) {
  int mode = 0;
  if (int4_bench_dyn_env()) {
    const char* s = std::getenv("MUSE_INT4_NF4");
    mode = (!s || !*s) ? 0 : std::atoi(s);
  } else {
    static const int cached = []() -> int {
      const char* s = std::getenv("MUSE_INT4_NF4");
      if (!s || !*s) return 0;
      return std::atoi(s);
    }();
    mode = cached;
  }
  if (mode == 2) return (K <= 16384);
  return (mode != 0);
}

int int4_mode() {
  if (int4_bench_dyn_env()) {
    const char* s = std::getenv("MUSE_INT4_MODE");
    return (!s || !*s) ? 0 : std::atoi(s);
  }
  static const int m = []() -> int {
    const char* s = std::getenv("MUSE_INT4_MODE");
    if (!s || !*s) return 0;
    return std::atoi(s);
  }();
  return m;
}

int int4_fc_mode() {
  if (int4_bench_dyn_env()) {
    const char* s = std::getenv("MUSE_INT4_FC_MODE");
    return (!s || !*s) ? 0 : std::atoi(s);
  }
  static const int m = []() -> int {
    const char* s = std::getenv("MUSE_INT4_FC_MODE");
    if (!s || !*s) return 0;
    return std::atoi(s);
  }();
  return m;
}

int int4_sk_grid() {
  if (int4_bench_dyn_env()) {
    const char* s = std::getenv("MUSE_DRAFT_SK_GRID");
    if (!s || !*s) return 144;
    const int v = std::atoi(s);
    return (v >= 0 && v <= 576) ? v : 144;
  }
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

template <bool kNf4>
static void int4_set_carveout_nf4() {
  cudaFuncSetAttribute(int4_mma_gemm_kernel<false, false, false, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_kernel<false, true, false, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_kernel<false, false, true, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_kernel<false, true, true, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_kernel<true, false, false, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_kernel<true, true, false, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_kernel<true, false, true, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_kernel<true, true, true, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_n32_kernel<false, false, false, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_n32_kernel<false, true, false, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_n32_kernel<false, false, true, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_n32_kernel<false, true, true, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_n32_kernel<true, false, false, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_n32_kernel<true, true, false, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_n32_kernel<true, false, true, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_n32_kernel<true, true, true, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_swiglu_kernel<false, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_swiglu_kernel<true, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

static void int4_ensure_carveout() {
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    int4_set_carveout_nf4<false>();
    int4_set_carveout_nf4<true>();
  }
}

template <bool kNf4>
static torch::Tensor launch_int4_gemm_impl(
    const __nv_bfloat16* A_ptr,
    Aux5Desc aux5,
    bool is_aux5,
    torch::Tensor W_packed,
    const torch::TensorOptions& opts,
    int M, int64_t N, int64_t K) {
  int4_ensure_carveout();
  auto C = torch::empty({M, N}, opts);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n_blocks = static_cast<int>(N / kRowsPerCTA);
  const int n32_blocks = static_cast<int>(N / 32);
  const bool use_m16 = (M > 8);
  const int mode = int4_mode();
  const int fc_mode = int4_fc_mode();
  const int sk_grid = int4_sk_grid();

  if (is_aux5 && fc_mode == 1 && n32_blocks <= 288) {
    if (use_m16) {
      int4_mma_gemm_n32_kernel<false, true, true, kNf4><<<n32_blocks, kThreads, 0, st>>>(
          nullptr, aux5, W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          nullptr, M, static_cast<int>(N), static_cast<int>(K), 1);
    } else {
      int4_mma_gemm_n32_kernel<false, false, true, kNf4><<<n32_blocks, kThreads, 0, st>>>(
          nullptr, aux5, W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          nullptr, M, static_cast<int>(N), static_cast<int>(K), 1);
    }
    return C;
  }

  if (is_aux5 && fc_mode == 0 && sk_grid > 0) {
    const int total_iters = n_blocks * static_cast<int>(K >> 6);
    const int grid = (sk_grid < total_iters) ? sk_grid : total_iters;
    const int pdl_mode = int4_pdl_mode();
    float* p_ptr = nullptr;
    int* d_ptr = nullptr;
    ensure_int4_sk_scratch(opts, n_blocks, p_ptr, d_ptr);
    if (use_m16) {
      int4_launch_pdl(
          int4_mma_streamk_kernel<true, true, kNf4>, grid, kThreads, st, pdl_mode,
          static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode);
    } else {
      int4_launch_pdl(
          int4_mma_streamk_kernel<false, true, kNf4>, grid, kThreads, st, pdl_mode,
          static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode);
    }
    return C;
  }

  if (!is_aux5 && mode == 1 && n32_blocks <= 288) {
    if (use_m16) {
      int4_mma_gemm_n32_kernel<false, true, false, kNf4><<<n32_blocks, kThreads, 0, st>>>(
          A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          nullptr, M, static_cast<int>(N), static_cast<int>(K), 1);
    } else {
      int4_mma_gemm_n32_kernel<false, false, false, kNf4><<<n32_blocks, kThreads, 0, st>>>(
          A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          nullptr, M, static_cast<int>(N), static_cast<int>(K), 1);
    }
    return C;
  }

  static torch::Tensor g_partials;
  static int64_t g_partials_cap = 0;
  int split_k = 1;
  if (n_blocks <= 128 && K >= 16384) {
    split_k = (K >= 28672) ? 4 : 3;
  } else if (!is_aux5 && mode == 2 && n_blocks <= 128 && K >= 4096) {
    split_k = 2;
  } else if (is_aux5 && fc_mode == 2) {
    split_k = 2;
  }
  if (split_k == 1) {
    if (is_aux5) {
      if (use_m16) {
        int4_mma_gemm_kernel<false, true, true, kNf4><<<n_blocks, kThreads, 0, st>>>(
            nullptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            nullptr, M, static_cast<int>(N), static_cast<int>(K), 1);
      } else {
        int4_mma_gemm_kernel<false, false, true, kNf4><<<n_blocks, kThreads, 0, st>>>(
            nullptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            nullptr, M, static_cast<int>(N), static_cast<int>(K), 1);
      }
    } else {
      if (use_m16) {
        int4_mma_gemm_kernel<false, true, false, kNf4><<<n_blocks, kThreads, 0, st>>>(
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            nullptr, M, static_cast<int>(N), static_cast<int>(K), 1);
      } else {
        int4_mma_gemm_kernel<false, false, false, kNf4><<<n_blocks, kThreads, 0, st>>>(
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
        int4_mma_gemm_kernel<true, true, true, kNf4><<<dim3(n_blocks, split_k, 1), kThreads, 0, st>>>(
            nullptr, aux5, W_packed.data_ptr<uint8_t>(),
            nullptr, g_partials.data_ptr<float>(),
            M, static_cast<int>(N), static_cast<int>(K), split_k);
      } else {
        int4_mma_gemm_kernel<true, false, true, kNf4><<<dim3(n_blocks, split_k, 1), kThreads, 0, st>>>(
            nullptr, aux5, W_packed.data_ptr<uint8_t>(),
            nullptr, g_partials.data_ptr<float>(),
            M, static_cast<int>(N), static_cast<int>(K), split_k);
      }
    } else {
      if (use_m16) {
        int4_mma_gemm_kernel<true, true, false, kNf4><<<dim3(n_blocks, split_k, 1), kThreads, 0, st>>>(
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            nullptr, g_partials.data_ptr<float>(),
            M, static_cast<int>(N), static_cast<int>(K), split_k);
      } else {
        int4_mma_gemm_kernel<true, false, false, kNf4><<<dim3(n_blocks, split_k, 1), kThreads, 0, st>>>(
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            nullptr, g_partials.data_ptr<float>(),
            M, static_cast<int>(N), static_cast<int>(K), split_k);
      }
    }
    const int total4 = (M * static_cast<int>(N)) >> 2;
    int4_reduce_splitk_kernel<<<(total4 + 255) / 256, 256, 0, st>>>(
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
  if (int4_nf4_for_shape(K)) {
    if (mse_opt) {
      int4_pack_kernel<true, true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K);
    } else {
      int4_pack_kernel<false, true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K);
    }
  } else {
    if (mse_opt) {
      int4_pack_kernel<true, false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K);
    } else {
      int4_pack_kernel<false, false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K);
    }
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
  if (int4_nf4_for_shape(static_cast<int>(K))) {
    return launch_int4_gemm_impl<true>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()), dummy, false, W_packed, A.options(), M, N, K);
  }
  return launch_int4_gemm_impl<false>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()), dummy, false, W_packed, A.options(), M, N, K);
}

torch::Tensor int4_pack_swiglu(torch::Tensor W, bool mse_opt = true) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kBFloat16 && W.dim() == 2 && W.is_contiguous(),
              "int4_pack_swiglu expects contiguous CUDA BF16 [2*I, K]");
  const int N2 = static_cast<int>(W.size(0));
  const int K = static_cast<int>(W.size(1));
  TORCH_CHECK(N2 % 64 == 0 && K % 64 == 0, "int4_pack_swiglu requires (2*I)%64==0 and K%64==0");
  const int I = N2 / 2;
  const int64_t total_bytes = static_cast<int64_t>(N2 / 8) * static_cast<int64_t>(K / 64) * kTileBytes;
  auto out = torch::empty({total_bytes}, W.options().dtype(torch::kUInt8));
  dim3 grid(N2 / 8, K / 64);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  if (int4_nf4_for_shape(K)) {
    if (mse_opt) {
      int4_pack_swiglu_kernel<true, true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), I, K);
    } else {
      int4_pack_swiglu_kernel<false, true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), I, K);
    }
  } else {
    if (mse_opt) {
      int4_pack_swiglu_kernel<true, false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), I, K);
    } else {
      int4_pack_swiglu_kernel<false, false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), I, K);
    }
  }
  return out;
}

torch::Tensor int4_gemm_swiglu(torch::Tensor A, torch::Tensor W_packed, int64_t I, int64_t K) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous(),
              "int4_gemm_swiglu expects contiguous CUDA BF16 [M, K]");
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && A.size(1) == K && I % 32 == 0 && K % 64 == 0,
              "int4_gemm_swiglu shape mismatch");
  int4_ensure_carveout();
  auto C = torch::empty({M, I}, A.options());
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n_blocks = static_cast<int>(I / 32);
  if (int4_nf4_for_shape(static_cast<int>(K))) {
    if (M > 8) {
      int4_mma_gemm_swiglu_kernel<true, true><<<n_blocks, kThreads, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          M, static_cast<int>(I), static_cast<int>(K));
    } else {
      int4_mma_gemm_swiglu_kernel<false, true><<<n_blocks, kThreads, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          M, static_cast<int>(I), static_cast<int>(K));
    }
  } else {
    if (M > 8) {
      int4_mma_gemm_swiglu_kernel<true, false><<<n_blocks, kThreads, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          M, static_cast<int>(I), static_cast<int>(K));
    } else {
      int4_mma_gemm_swiglu_kernel<false, false><<<n_blocks, kThreads, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          M, static_cast<int>(I), static_cast<int>(K));
    }
  }
  return C;
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
  if (int4_nf4_for_shape(static_cast<int>(K))) {
    return launch_int4_gemm_impl<true>(nullptr, desc, true, W_packed, h0.options(), M, N, K);
  }
  return launch_int4_gemm_impl<false>(nullptr, desc, true, W_packed, h0.options(), M, N, K);
}
