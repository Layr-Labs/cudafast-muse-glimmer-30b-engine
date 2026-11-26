// SPDX-License-Identifier: Apache-2.0
// muse_nf8_mma: 8-level symmetric Gaussian Lloyd-Max (NF8) group-64 W3.25A16 Tensor-Core GEMM
// for the pinned DFlash drafter & reduced draft vocab head.
//
// Format (3.250 bits/weight exact, 0 wasted bits, 16-byte aligned):
//   8 symmetric N(0,1) Lloyd-Max centroids normalized to [-1, +1]:
//     |c| in {0.11376953, 0.3515625, 0.625, 1.0} (exact BF16: 0x3DE9, 0x3EB4, 0x3F20, 0x3F80),
//     sign bit in hi1 -> 8 levels (+/- c0..c3), zero-point = 0 (symmetric).
//   Each 8x64 tile (8 output rows x 64 K cols) is 208 bytes (13 x uint4):
//     [0..127]   uint32_t lo2[32] : for lane l=(r<<2)|t, 8 pairs of 2-bit mag indices (4 bits/pair)
//     [128..191] uint16_t hi1[32] : for lane l=(r<<2)|t, 8 pairs of 1-bit sign bits
//                                   (low byte: even weight signs for pairs 0..7; high byte: odd weight signs)
//     [192..207] __nv_bfloat16 sc[8] : 1 BF16 scale per row r=0..7
//   Per 4-warp CTA (64 N-rows, K=64): 8 x 208 = 1664 bytes = 104 x uint4.
//   Dequant per weight pair (b0/b1):
//     16-entry 64-byte SMEM BF16x2 magnitude-pair LUT + 1-op sign XOR + 1 HMUL2 (no HADD2, no zero-point).
//   5-stage cp.async.cg.L2::256B pipeline -> 18.6 KiB SMEM (M<=16) -> 4 CTAs/SM (192 CTAs on SM121).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <stdint.h>

namespace {

constexpr int kNf8Warps = 4;
constexpr int kNf8Threads = kNf8Warps * 32;            // 128
constexpr int kNf8NTilesPerWarp = 2;                   // 16 N-rows/warp -> 2x A reuse
constexpr int kNf8NTilesPerCTA = kNf8Warps * kNf8NTilesPerWarp;  // 8
constexpr int kNf8RowsPerCTA = kNf8NTilesPerCTA * 8;   // 64
constexpr int kNf8TileBytes = 208;                     // 13 * 16
constexpr int kNf8StepWBytes = kNf8NTilesPerCTA * kNf8TileBytes;  // 1664 = 104 * 16
constexpr int kNf8StepWU4 = kNf8StepWBytes / 16;       // 104
constexpr int kNf8Stages = 5;

// Exact BF16 representations of normalized 8-level symmetric N(0,1) Lloyd-Max magnitudes:
//   0.245094/2.151946 = 0.113894 -> BF16 0x3DE9 (0.11376953125)
//   0.756005/2.151946 = 0.351312 -> BF16 0x3EB4 (0.3515625)
//   1.343909/2.151946 = 0.624509 -> BF16 0x3F20 (0.625)
//   2.151946/2.151946 = 1.000000 -> BF16 0x3F80 (1.0)
__device__ __constant__ uint16_t kNf8MagBf16[4] = {0x3DE9u, 0x3EB4u, 0x3F20u, 0x3F80u};
__device__ __constant__ float kNf8MagF32[4] = {0.11376953125f, 0.3515625f, 0.625f, 1.0f};
__device__ __constant__ float kNf8Thresh[3] = {0.232666015625f, 0.48828125f, 0.8125f};

__device__ __forceinline__ void nf8_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void nf8_mma_m16n8k16_full(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void nf8_cp_async16_cg(uint32_t sd, const void* gs) {
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ void nf8_cp_async16_ca_z(uint32_t sd, const void* gs, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(sd), "l"(gs), "r"(src_bytes));
}

__device__ __forceinline__ void nf8_ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(smem_addr));
}

__device__ __forceinline__ void nf8_ldmatrix_x2(
    uint32_t& r0, uint32_t& r1, uint32_t smem_addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(r0), "=r"(r1)
      : "r"(smem_addr));
}

__device__ __forceinline__ uint32_t nf8_bf16x2_u32(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float nf8_subwarp4_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}

__device__ __forceinline__ void nf8_init_lut(uint32_t* __restrict__ lut16, int tid) {
  if (tid < 16) {
    const uint32_t lo = static_cast<uint32_t>(kNf8MagBf16[tid & 3]);
    const uint32_t hi = static_cast<uint32_t>(kNf8MagBf16[(tid >> 2) & 3]);
    lut16[tid] = lo | (hi << 16);
  }
}

__device__ __forceinline__ uint32_t nf8_unpack_hi32(uint16_t hi1) {
  const uint32_t u = static_cast<uint32_t>(hi1);
  return (u & 0x00FFu) | ((u & 0xFF00u) << 8);
}

// Dequant 4 weights (2 pairs: b0=(w0,w1), b1=(w2,w3)) for step s in 0..3.
// Pair 2*s uses lo2 bits [8*s+3 : 8*s] and hi32 bits (2*s, 16+2*s).
// Pair 2*s+1 uses lo2 bits [8*s+7 : 8*s+4] and hi32 bits (2*s+1, 16+2*s+1).
__device__ __forceinline__ void nf8_deq4_k16(
    const uint32_t* __restrict__ lut16,
    uint32_t lo2, uint32_t hi32, int s,
    __nv_bfloat162 sc2,
    uint32_t& b0, uint32_t& b1) {
  const uint32_t m0 = (lo2 >> (8 * s)) & 0xFu;
  const uint32_t m1 = (lo2 >> (8 * s + 4)) & 0xFu;
  const uint32_t sgn0 = (hi32 << (15 - 2 * s)) & 0x80008000u;
  const uint32_t sgn1 = (hi32 << (14 - 2 * s)) & 0x80008000u;
  const uint32_t q0_u = lut16[m0] ^ sgn0;
  const uint32_t q1_u = lut16[m1] ^ sgn1;
  const __nv_bfloat162 w01 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q0_u), sc2);
  const __nv_bfloat162 w23 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q1_u), sc2);
  b0 = *reinterpret_cast<const uint32_t*>(&w01);
  b1 = *reinterpret_cast<const uint32_t*>(&w23);
}

// Quantize a single normalized |x| = |w|/sc into mag index m in 0..3
__device__ __forceinline__ int nf8_mag_idx(float ax) {
  int m = 0;
  m += (ax > kNf8Thresh[0]);
  m += (ax > kNf8Thresh[1]);
  m += (ax > kNf8Thresh[2]);
  return m;
}

// GPU packer: BF16 [N, K] (N%64==0, K%64==0) -> uint8 [N/64, K/64, 8, 208]
// Grid: (N/8, K/64), Block: 32 threads (1 warp per 8x64 tile).
// Finds exact BF16 MSE-optimal scale per 64-weight group via multi-start closed-form ALS.
__global__ void nf8_pack_kernel(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ out,
    int N, int K,
    int wave_ilv) {
  const int n_tile = blockIdx.x;  // 0 .. N/8 - 1
  const int kg = blockIdx.y;      // 0 .. K/64 - 1
  const int lane = threadIdx.x;   // 0 .. 31
  const int r = lane >> 2;        // 0 .. 7
  const int t = lane & 3;         // 0 .. 3
  const int row = n_tile * 8 + r;
  const int k_base = kg * 64;

  float vals[16];
  float abs_max = 0.0f;
  float lsumsq = 0.0f;
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
    const float a = fabsf(vals[i]);
    abs_max = fmaxf(abs_max, a);
    lsumsq = __fmaf_rn(vals[i], vals[i], lsumsq);
  }
  abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffffu, abs_max, 1));
  abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffffu, abs_max, 2));
  const float rms = sqrtf(fmaxf(nf8_subwarp4_sum(lsumsq) * (1.0f / 64.0f), 0.0f));

  auto eval_and_refine = [&](float sc_init, __nv_bfloat16& best_sc, float& best_mse) {
    float sc = sc_init;
    if (!(sc > 1e-30f)) sc = 1e-30f;
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const float inv_sc = 1.0f / sc;
      float num = 0.0f, den = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const float a = fabsf(vals[i]);
        const float c = kNf8MagF32[nf8_mag_idx(a * inv_sc)];
        num = __fmaf_rn(a, c, num);
        den = __fmaf_rn(c, c, den);
      }
      num = nf8_subwarp4_sum(num);
      den = nf8_subwarp4_sum(den);
      if (den > 1e-30f) {
        const float sc_new = num / den;
        if (sc_new > 1e-30f) sc = sc_new;
      }
    }
    const __nv_bfloat16 sc_bf = __float2bfloat16_rn(sc);
    const float sc_q = __bfloat162float(sc_bf);
    const float inv_sc = (sc_q > 1e-30f) ? (1.0f / sc_q) : 0.0f;
    float err = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float a = fabsf(vals[i]);
      const float c = kNf8MagF32[nf8_mag_idx(a * inv_sc)];
      const float rec = __bfloat162float(__float2bfloat16_rn(c * sc_q));
      const float d = a - rec;
      err = __fmaf_rn(d, d, err);
    }
    err = nf8_subwarp4_sum(err);
    if (err < best_mse) {
      best_mse = err;
      best_sc = sc_bf;
    }
  };

  __nv_bfloat16 best_sc = __float2bfloat16_rn(fmaxf(abs_max, 1e-30f));
  float best_mse = 1e38f;
  eval_and_refine(fmaxf(abs_max, 1e-30f), best_sc, best_mse);
  if (rms > 1e-30f) {
    eval_and_refine(2.151946f * rms, best_sc, best_mse);
    eval_and_refine(1.85f * rms, best_sc, best_mse);
    eval_and_refine(2.50f * rms, best_sc, best_mse);
  }

  const float sc_q = __bfloat162float(best_sc);
  const float inv_sc = (sc_q > 1e-30f) ? (1.0f / sc_q) : 0.0f;

  uint32_t lo2 = 0u;
  uint32_t hi1 = 0u;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const float v0 = vals[2 * p + 0];
    const float v1 = vals[2 * p + 1];
    const uint32_t m0 = static_cast<uint32_t>(nf8_mag_idx(fabsf(v0) * inv_sc));
    const uint32_t m1 = static_cast<uint32_t>(nf8_mag_idx(fabsf(v1) * inv_sc));
    const uint32_t s0 = (v0 < 0.0f) ? 1u : 0u;
    const uint32_t s1 = (v1 < 0.0f) ? 1u : 0u;
    lo2 |= (m0 | (m1 << 2)) << (4 * p);
    hi1 |= (s0 << p) | (s1 << (8 + p));
  }

  const int cta_n = n_tile >> 3;
  const int sub_n = n_tile & 7;
  const int num_kg = K >> 6;
  const int num_cta = N >> 6;
  size_t tile_off;
  if (wave_ilv > 0) {
    const int wave = cta_n / wave_ilv;
    const int in_wave = cta_n - wave * wave_ilv;
    const int wave_sz = ((wave + 1) * wave_ilv <= num_cta) ? wave_ilv : (num_cta - wave * wave_ilv);
    const size_t wave_base = static_cast<size_t>(wave * wave_ilv) * num_kg * kNf8StepWBytes;
    tile_off = wave_base +
               (static_cast<size_t>(kg) * wave_sz + in_wave) * kNf8StepWBytes +
               static_cast<size_t>(sub_n) * kNf8TileBytes;
  } else {
    tile_off = (static_cast<size_t>(cta_n) * num_kg + kg) * kNf8StepWBytes +
               static_cast<size_t>(sub_n) * kNf8TileBytes;
  }
  uint8_t* dst = out + tile_off;
  reinterpret_cast<uint32_t*>(dst)[lane] = lo2;
  reinterpret_cast<uint16_t*>(dst + 128)[lane] = static_cast<uint16_t>(hi1);
  if (t == 0) {
    reinterpret_cast<__nv_bfloat16*>(dst + 192)[r] = best_sc;
  }
}

// Main NF8-g64 Tensor-Core GEMM kernel: C[M, N] = A[M, K] @ W[N, K]^T
template <bool kM16, bool kSplitK, int kWaveIlv>
__global__ void __launch_bounds__(kNf8Threads, 4) nf8_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ C_fp32,
    int M, int N, int K, int split_k, int pdl_mode) {
  constexpr int kAMRows = kM16 ? 16 : 8;
  struct __align__(16) Smem {
    uint8_t w[kNf8Stages][kNf8StepWBytes];
    __nv_bfloat16 a[kNf8Stages][kAMRows][64];
    uint32_t lut16[16];
  };
  __shared__ Smem smem;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;
  const int num_cta = N >> 6;

  const int cta_n = blockIdx.x;
  const int sk_idx = kSplitK ? blockIdx.y : 0;
  const int kg_per_sk = kSplitK ? ((num_kg + split_k - 1) / split_k) : num_kg;
  const int kg_start = kSplitK ? (sk_idx * kg_per_sk) : 0;
  int kg_end = kSplitK ? (kg_start + kg_per_sk) : num_kg;
  if (kg_end > num_kg) kg_end = num_kg;
  const int my_kgs = kg_end - kg_start;
  if (my_kgs <= 0) return;

  nf8_init_lut(smem.lut16, tid);

  const uint8_t* w_base;
  size_t w_kg_stride;
  if constexpr (kWaveIlv > 0) {
    const int wave = cta_n / kWaveIlv;
    const int in_wave = cta_n - wave * kWaveIlv;
    const int wave_sz = ((wave + 1) * kWaveIlv <= num_cta) ? kWaveIlv : (num_cta - wave * kWaveIlv);
    w_base = W_packed +
             static_cast<size_t>(wave * kWaveIlv) * num_kg * kNf8StepWBytes +
             static_cast<size_t>(in_wave) * kNf8StepWBytes;
    w_kg_stride = static_cast<size_t>(wave_sz) * kNf8StepWBytes;
  } else {
    w_base = W_packed + static_cast<size_t>(cta_n) * num_kg * kNf8StepWBytes;
    w_kg_stride = kNf8StepWBytes;
  }

  if (cta_n == 0 && sk_idx == 0) {
    const size_t total_bytes = static_cast<size_t>(num_cta) * num_kg * kNf8StepWBytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes;
         off += static_cast<size_t>(kNf8Threads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_swiz8 = a_col8 ^ (a_row & 7);
  const bool a_valid = a_row < M;

  auto issue_w_only = [&](int kg, int st) {
    if (tid < kNf8StepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      nf8_cp_async16_cg(sd, w_base + static_cast<size_t>(kg) * w_kg_stride + tid * 16);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  auto load_a_sync = [&](int kg, int st) {
    if constexpr (kM16) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_swiz8 * 8]) =
          a_valid
              ? __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8))
              : make_uint4(0u, 0u, 0u, 0u);
    } else if (tid < 64) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_swiz8 * 8]) =
          a_valid
              ? __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8))
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  auto issue_stage = [&](int kg, int st) {
    if (tid < kNf8StepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      nf8_cp_async16_cg(sd, w_base + static_cast<size_t>(kg) * w_kg_stride + tid * 16);
    }
    if constexpr (kM16) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid ? (A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8) : A;
      nf8_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid ? (A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8) : A;
      nf8_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kNf8Stages - 1;
  const int preload = (my_kgs < kPreload) ? my_kgs : kPreload;
  if ((pdl_mode & 1) != 0) {
    for (int i = 0; i < preload; ++i) issue_w_only(kg_start + i, i);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = 0; i < preload; ++i) load_a_sync(kg_start + i, i);
  } else {
    for (int i = 0; i < preload; ++i) issue_stage(kg_start + i, i);
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
    if (rem >= 3) asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    else if (rem == 2) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    else if (rem == 1) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    const int next_i = i + kPreload;
    if (next_i < my_kgs) {
      const int next_st = (st == 0) ? (kNf8Stages - 1) : (st - 1);
      issue_stage(kg_start + next_i, next_st);
    }

    const uint8_t* wt0 = &smem.w[st][(warp * 2 + 0) * kNf8TileBytes];
    const uint8_t* wt1 = &smem.w[st][(warp * 2 + 1) * kNf8TileBytes];
    const uint32_t lo0 = reinterpret_cast<const uint32_t*>(wt0)[lane];
    const uint32_t hi0 = nf8_unpack_hi32(reinterpret_cast<const uint16_t*>(wt0 + 128)[lane]);
    const __nv_bfloat16 sc0 = reinterpret_cast<const __nv_bfloat16*>(wt0 + 192)[g];
    const __nv_bfloat162 sc0_2 = __bfloat162bfloat162(sc0);

    const uint32_t lo1 = reinterpret_cast<const uint32_t*>(wt1)[lane];
    const uint32_t hi1 = nf8_unpack_hi32(reinterpret_cast<const uint16_t*>(wt1 + 128)[lane]);
    const __nv_bfloat16 sc1 = reinterpret_cast<const __nv_bfloat16*>(wt1 + 192)[g];
    const __nv_bfloat162 sc1_2 = __bfloat162bfloat162(sc1);

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        const int ldm_col8 = (2 * s + (lane >> 4)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 15][ldm_col8 * 8]));
        nf8_ldmatrix_x4(a0, a1, a2, a3, a_addr);
      } else {
        const int ldm_col8 = (2 * s + ((lane >> 3) & 1)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 7][ldm_col8 * 8]));
        nf8_ldmatrix_x2(a0, a2, a_addr);
      }
      uint32_t b0, b1, b2, b3;
      nf8_deq4_k16(smem.lut16, lo0, hi0, s, sc0_2, b0, b1);
      nf8_deq4_k16(smem.lut16, lo1, hi1, s, sc1_2, b2, b3);
      if constexpr (kM16) {
        nf8_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0, a1, a2, a3, b0, b1);
        nf8_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0, a1, a2, a3, b2, b3);
      } else {
        nf8_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b0, b1);
        nf8_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, b2, b3);
      }
    }
    st = (st + 1 == kNf8Stages) ? 0 : (st + 1);
  }

  const int col0 = cta_n * kNf8RowsPerCTA + (warp * 2 + 0) * 8 + 2 * t;
  const int col1 = cta_n * kNf8RowsPerCTA + (warp * 2 + 1) * 8 + 2 * t;
  if (g < M) {
    if constexpr (!kSplitK) {
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) = nf8_bf16x2_u32(d0_0, d1_0);
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col1) = nf8_bf16x2_u32(d0_1, d1_1);
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
        *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(ghi) * N + col0) = nf8_bf16x2_u32(d2_0, d3_0);
        *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(ghi) * N + col1) = nf8_bf16x2_u32(d2_1, d3_1);
      } else {
        float* p0 = C_fp32 + (static_cast<size_t>(sk_idx) * M + ghi) * N + col0;
        float* p1 = C_fp32 + (static_cast<size_t>(sk_idx) * M + ghi) * N + col1;
        p0[0] = d2_0; p0[1] = d3_0;
        p1[0] = d2_1; p1[1] = d3_1;
      }
    }
  }
}

__global__ void nf8_splitk_reduce_kernel(
    const float* __restrict__ partial,
    __nv_bfloat16* __restrict__ out,
    int total_elems, int split_k) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  float acc = 0.0f;
  for (int s = 0; s < split_k; ++s) {
    acc += partial[static_cast<size_t>(s) * total_elems + idx];
  }
  out[idx] = __float2bfloat16_rn(acc);
}

static int nf8_pdl_mode() {
  static int v = -1;
  if (v < 0) {
    const char* s = std::getenv("MUSE_DRAFT_PDL");
    v = (s && *s) ? std::atoi(s) : 3;
  }
  return v;
}

template <typename KernelFn, typename... Args>
static void nf8_launch_kernel(KernelFn fn, dim3 grid, dim3 block, cudaStream_t stream, int pdl_mode, Args... args) {
  if (pdl_mode != 0) {
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, fn, args...);
  } else {
    fn<<<grid, block, 0, stream>>>(args...);
  }
}

}  // namespace

torch::Tensor nf8_pack(torch::Tensor W, int64_t wave_ilv = 0) {
  TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kBFloat16 && W.dim() == 2,
              "nf8_pack expects 2D CUDA BF16 tensor [N, K]");
  auto Wc = W.contiguous();
  const int N = static_cast<int>(Wc.size(0));
  const int K = static_cast<int>(Wc.size(1));
  TORCH_CHECK(N % 64 == 0 && K % 64 == 0, "nf8_pack requires N%64==0 and K%64==0");
  TORCH_CHECK(wave_ilv == 0 || wave_ilv == 192, "nf8_pack supports wave_ilv in {0, 192}");

  const int64_t total_bytes = static_cast<int64_t>(N / 64) * (K / 64) * kNf8StepWBytes;
  auto out = torch::empty({total_bytes}, Wc.options().dtype(torch::kUInt8));
  dim3 grid(N / 8, K / 64);
  dim3 block(32);
  auto stream = at::cuda::getCurrentCUDAStream();
  nf8_pack_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(Wc.data_ptr<at::BFloat16>()),
      out.data_ptr<uint8_t>(), N, K, static_cast<int>(wave_ilv));
  return out;
}

torch::Tensor nf8_gemm(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K, int64_t wave_ilv = 0) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kBFloat16 && A.dim() == 2,
              "nf8_gemm expects 2D CUDA BF16 A[M, K]");
  TORCH_CHECK(W_packed.is_cuda() && W_packed.dtype() == torch::kUInt8,
              "nf8_gemm expects CUDA uint8 W_packed");
  auto Ac = A.contiguous();
  const int M = static_cast<int>(Ac.size(0));
  TORCH_CHECK(M >= 1 && M <= 16, "nf8_gemm expects 1 <= M <= 16");
  TORCH_CHECK(Ac.size(1) == K && (N % 64 == 0) && (K % 64 == 0), "nf8_gemm shape mismatch");

  auto C = torch::empty({M, N}, Ac.options());
  const int cta_n = static_cast<int>(N / kNf8RowsPerCTA);
  const int num_kg = static_cast<int>(K / 64);
  const int pdl = nf8_pdl_mode();
  auto stream = at::cuda::getCurrentCUDAStream();

  int split_k = 1;
  if (wave_ilv == 0 && cta_n < 192) {
    int want = (192 + cta_n - 1) / cta_n;
    if (want > 4) want = 4;
    if (want > num_kg / 8) want = num_kg / 8;
    if (want < 1) want = 1;
    split_k = want;
  }

  const auto* a_ptr = reinterpret_cast<const __nv_bfloat16*>(Ac.data_ptr<at::BFloat16>());
  const auto* w_ptr = W_packed.data_ptr<uint8_t>();
  auto* c_ptr = reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>());

  if (split_k > 1) {
    auto partial = torch::empty({split_k, M, N}, Ac.options().dtype(torch::kFloat32));
    float* p_ptr = partial.data_ptr<float>();
    dim3 grid(cta_n, split_k);
    dim3 block(kNf8Threads);
    if (M <= 8) {
      nf8_launch_kernel(nf8_gemm_kernel<false, true, 0>, grid, block, stream, pdl,
                        a_ptr, w_ptr, c_ptr, p_ptr, M, static_cast<int>(N), static_cast<int>(K), split_k, pdl);
    } else {
      nf8_launch_kernel(nf8_gemm_kernel<true, true, 0>, grid, block, stream, pdl,
                        a_ptr, w_ptr, c_ptr, p_ptr, M, static_cast<int>(N), static_cast<int>(K), split_k, pdl);
    }
    const int total = M * static_cast<int>(N);
    nf8_splitk_reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(p_ptr, c_ptr, total, split_k);
  } else {
    dim3 grid(cta_n, 1);
    dim3 block(kNf8Threads);
    if (wave_ilv == 192) {
      if (M <= 8) {
        nf8_launch_kernel(nf8_gemm_kernel<false, false, 192>, grid, block, stream, pdl,
                          a_ptr, w_ptr, c_ptr, static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl);
      } else {
        nf8_launch_kernel(nf8_gemm_kernel<true, false, 192>, grid, block, stream, pdl,
                          a_ptr, w_ptr, c_ptr, static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl);
      }
    } else {
      if (M <= 8) {
        nf8_launch_kernel(nf8_gemm_kernel<false, false, 0>, grid, block, stream, pdl,
                          a_ptr, w_ptr, c_ptr, static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl);
      } else {
        nf8_launch_kernel(nf8_gemm_kernel<true, false, 0>, grid, block, stream, pdl,
                          a_ptr, w_ptr, c_ptr, static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl);
      }
    }
  }
  return C;
}
