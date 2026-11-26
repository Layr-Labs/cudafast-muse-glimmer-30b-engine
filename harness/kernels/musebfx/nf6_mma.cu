// SPDX-License-Identifier: Apache-2.0
// muse_nf6_mma: 6-level symmetric Lloyd-Max Gaussian W3.0A16 Tensor-Core GEMM for the pinned DFlash drafter.
// Format (3.000 bits/weight EXACT, 0 wasted bits, 16-byte & 256-byte cache-line aligned):
//   6 symmetric N(0,1) Lloyd-Max centroids (stored as exact BF16):
//     {-1.8515625, -0.98046875, -0.318359375, +0.318359375, +0.98046875, +1.8515625}
//   Every MMA k16 quad (4 weights per lane) has 6^4 = 1,296 states -> fits in 11 bits (0..1295).
//   Each 8x64 tile (8 output rows x 64 K cols = 512 weights) is 192 bytes (12 x uint4):
//     [0..127]   uint32_t lo32[32] : for lane l=(r<<2)|t, c0[10:0] | c1[21:11] | c2_lo[31:22]
//     [128..191] uint16_t hi16[32] : c2_hi[0] | c3[11:1] | sc_nib[15:12] (4 bits of row r's BF16 scale per lane t)
//   8 tiles per 4-warp CTA = 1,536 bytes = 96 x uint4 = 6 x 256B cache lines exact (-14.3% vs INT3-g64 1,792 B).
//   Dequant: 1x LDS.64 from 10.1 KiB SMEM LUT (uint2 = bf16x4) + 2x HMUL2 (no HADD2 / zero-point!),
//   4-stage (M<=16) / 5-stage (M<=8) cp.async.cg.L2::256B, XOR-swizzled ldmatrix, 4 CTAs/SM (192 active CTAs),
//   dual SM121 PDL, and 2 MiB TLB prefetch.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <stdint.h>

namespace {

constexpr int kNf6WarpsPerBlock = 4;
constexpr int kNf6Threads = kNf6WarpsPerBlock * 32;            // 128
constexpr int kNf6NTilesPerWarp = 2;                           // 16 N-rows per warp -> 2x A reuse
constexpr int kNf6NTilesPerCTA = kNf6WarpsPerBlock * kNf6NTilesPerWarp;  // 8
constexpr int kNf6RowsPerCTA = kNf6NTilesPerCTA * 8;           // 64
constexpr int kNf6TileBytes = 192;                             // 12 * 16 = 3.000 bpw exact
constexpr int kNf6StepWBytes = kNf6NTilesPerCTA * kNf6TileBytes;  // 1536 = 96 * 16 = 6 * 256B
constexpr int kNf6StepWU4 = kNf6StepWBytes / 16;               // 96
constexpr int kNf6PairLutSize = 36;                            // 6^2 -> 144 B SMEM (fits 5 stages @ 4 CTAs/SM!)

// Exact BF16 bit patterns for 6-level N(0,1) Lloyd-Max centroids:
//   -1.8515625 (0xBFED), -0.98046875 (0xBF7B), -0.318359375 (0xBEA3),
//   +0.318359375 (0x3EA3), +0.98046875 (0x3F7B), +1.8515625 (0x3FED)
__device__ __constant__ uint16_t kNf6CentBf16[6] = {
    0xBFEDu, 0xBF7Bu, 0xBEA3u, 0x3EA3u, 0x3F7Bu, 0x3FEDu
};
__device__ __constant__ float kNf6CentF32[6] = {
    -1.8515625f, -0.98046875f, -0.318359375f, 0.318359375f, 0.98046875f, 1.8515625f
};
// Midpoints between consecutive centroids for fast nearest-centroid lookup:
//   -1.416015625, -0.6494140625, 0.0, +0.6494140625, +1.416015625
__device__ __forceinline__ int nf6_quantize_val(float vn) {
  int q = 0;
  if (vn >= 0.0f) {
    if (vn < 0.6494140625f) q = 3;
    else if (vn < 1.416015625f) q = 4;
    else q = 5;
  } else {
    if (vn > -0.6494140625f) q = 2;
    else if (vn > -1.416015625f) q = 1;
    else q = 0;
  }
  return q;
}

__device__ __forceinline__ void nf6_init_lut(uint32_t* lut, int tid) {
  if (tid < kNf6PairLutSize) {
    const int q0 = tid % 6;
    const int q1 = tid / 6;
    lut[tid] = static_cast<uint32_t>(kNf6CentBf16[q0]) | (static_cast<uint32_t>(kNf6CentBf16[q1]) << 16);
  }
}

__device__ __forceinline__ void nf6_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void nf6_mma_m16n8k16_full(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void nf6_cp_async16_cg(uint32_t sd, const void* gs) {
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ void nf6_cp_async16_ca_z(uint32_t sd, const void* gs, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(sd), "l"(gs), "r"(src_bytes));
}

__device__ __forceinline__ void nf6_ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(smem_addr));
}

__device__ __forceinline__ void nf6_ldmatrix_x2(
    uint32_t& r0, uint32_t& r1, uint32_t smem_addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(r0), "=r"(r1)
      : "r"(smem_addr));
}

__device__ __forceinline__ uint32_t nf6_bf16x2_u32(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float nf6_swiglu_f32(float gate, float up) {
  return (gate / (1.0f + __expf(-gate))) * up;
}

__device__ __forceinline__ float nf6_subwarp4_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}

__device__ __forceinline__ void nf6_deq_quad(
    const uint32_t* __restrict__ lut, uint32_t code, __nv_bfloat162 sc2, uint32_t& b0, uint32_t& b1) {
  // Exact division by 36 for all 0 <= code < 1296: floor(code * 1821 / 65536) == code / 36.
  const uint32_t hi = (code * 1821u) >> 16;
  const uint32_t lo = code - 36u * hi;
  const uint32_t u0 = lut[lo];
  const uint32_t u1 = lut[hi];
  const __nv_bfloat162 w01 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&u0), sc2);
  const __nv_bfloat162 w23 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&u1), sc2);
  b0 = *reinterpret_cast<const uint32_t*>(&w01);
  b1 = *reinterpret_cast<const uint32_t*>(&w23);
}

// Optimize per-64-group BF16 scale via RMS init + 6 closed-form Lloyd-Max ALS iterations
// minimizing exact BF16-dequantized MSE (w_hat = rn_bf16(bf16(cent[q]) * bf16(scale))).
__device__ __forceinline__ uint16_t nf6_fit_group_scale(const float (&vals)[16], int r) {
  float lsq = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    lsq = __fmaf_rn(vals[i], vals[i], lsq);
  }
  float rms = sqrtf(fmaxf(nf6_subwarp4_sum(lsq) * (1.0f / 64.0f), 0.0f));
  if (!(rms > 1e-30f)) rms = 1e-30f;

  auto eval_sc = [&](float sc_in, uint16_t& out_bits, float& out_mse) {
    if (!(sc_in > 1e-30f)) sc_in = 1e-30f;
    __nv_bfloat16 sc_bf = __float2bfloat16_rn(sc_in);
    out_bits = *reinterpret_cast<const uint16_t*>(&sc_bf);
    const float sc_q = __bfloat162float(sc_bf);
    const float inv_sc = (sc_q > 1e-30f) ? (1.0f / sc_q) : 0.0f;
    float err = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = nf6_quantize_val(vals[i] * inv_sc);
      const float w_hat = __bfloat162float(__float2bfloat16_rn(kNf6CentF32[q] * sc_q));
      const float d = vals[i] - w_hat;
      err = __fmaf_rn(d, d, err);
    }
    out_mse = nf6_subwarp4_sum(err);
  };

  uint16_t best_bits = 0u;
  float best_mse = 1e38f;
  float cur_sc = rms;
  eval_sc(cur_sc, best_bits, best_mse);

  // Try two initial scale multipliers around RMS to avoid local minima on heavy-tailed groups.
  constexpr float kInitMults[2] = {0.88f, 1.12f};
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    uint16_t tb;
    float tm;
    eval_sc(rms * kInitMults[m], tb, tm);
    if (tm < best_mse) {
      best_mse = tm;
      best_bits = tb;
      cur_sc = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&tb));
    }
  }

#pragma unroll
  for (int it = 0; it < 6; ++it) {
    const float inv_cur = (cur_sc > 1e-30f) ? (1.0f / cur_sc) : 0.0f;
    float lwc = 0.0f;
    float lcc = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = nf6_quantize_val(vals[i] * inv_cur);
      const float c = kNf6CentF32[q];
      lwc = __fmaf_rn(vals[i], c, lwc);
      lcc = __fmaf_rn(c, c, lcc);
    }
    const float swc = nf6_subwarp4_sum(lwc);
    const float scc = nf6_subwarp4_sum(lcc);
    if (scc > 1e-12f) {
      const float cand = swc / scc;
      if (cand > 1e-30f && cand < 1e30f) {
        cur_sc = cand;
        uint16_t tb;
        float tm;
        eval_sc(cur_sc, tb, tm);
        if (tm < best_mse) {
          best_mse = tm;
          best_bits = tb;
        }
      }
    }
  }
  uint32_t b32 = static_cast<uint32_t>(best_bits);
  b32 = __shfl_sync(0xffffffffu, b32, r << 2);
  return static_cast<uint16_t>(b32);
}

// GPU packer: BF16 [N, K] (N%64==0, K%64==0) -> uint8 [N/64, K/64, 8, 192]
__global__ void nf6_pack_kernel(
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
  const __nv_bfloat16* wrow = W + static_cast<size_t>(row) * K + k_base;
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int kb = s * 16 + 2 * t;
    vals[4 * s + 0] = __bfloat162float(wrow[kb + 0]);
    vals[4 * s + 1] = __bfloat162float(wrow[kb + 1]);
    vals[4 * s + 2] = __bfloat162float(wrow[kb + 8]);
    vals[4 * s + 3] = __bfloat162float(wrow[kb + 9]);
  }

  const uint16_t sc_bits = nf6_fit_group_scale(vals, r);
  const float sc_q = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&sc_bits));
  const float inv_sc = (sc_q > 1e-30f) ? (1.0f / sc_q) : 0.0f;

  uint32_t codes[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int q0 = nf6_quantize_val(vals[4 * s + 0] * inv_sc);
    const int q1 = nf6_quantize_val(vals[4 * s + 1] * inv_sc);
    const int q2 = nf6_quantize_val(vals[4 * s + 2] * inv_sc);
    const int q3 = nf6_quantize_val(vals[4 * s + 3] * inv_sc);
    codes[s] = static_cast<uint32_t>(q0 + 6 * (q1 + 6 * (q2 + 6 * q3)));
  }

  const uint32_t lo32 = codes[0] | (codes[1] << 11) | ((codes[2] & 0x3FFu) << 22);
  const uint32_t sc_nib = (static_cast<uint32_t>(sc_bits) >> (4 * t)) & 0xFu;
  const uint16_t hi16 = static_cast<uint16_t>((codes[2] >> 10) | (codes[3] << 1) | (sc_nib << 12));

  const int cta_n = n_tile >> 3;
  const int sub_tile = n_tile & 7;
  const int num_kg = K >> 6;
  const size_t tile_idx = (static_cast<size_t>(cta_n) * num_kg + kg) * kNf6NTilesPerCTA + sub_tile;
  uint8_t* dst = out + tile_idx * kNf6TileBytes;

  reinterpret_cast<uint32_t*>(dst)[lane] = lo32;
  reinterpret_cast<uint16_t*>(dst + 128)[lane] = hi16;
}

// Paired gate_up packer for fused SwiGLU: BF16 [2*I, K] -> uint8 [I/32, K/64, 8, 192]
__global__ void nf6_pack_swiglu_kernel(
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
  const __nv_bfloat16* wrow = W + static_cast<size_t>(row) * K + k_base;
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int kb = s * 16 + 2 * t;
    vals[4 * s + 0] = __bfloat162float(wrow[kb + 0]);
    vals[4 * s + 1] = __bfloat162float(wrow[kb + 1]);
    vals[4 * s + 2] = __bfloat162float(wrow[kb + 8]);
    vals[4 * s + 3] = __bfloat162float(wrow[kb + 9]);
  }

  const uint16_t sc_bits = nf6_fit_group_scale(vals, r);
  const float sc_q = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&sc_bits));
  const float inv_sc = (sc_q > 1e-30f) ? (1.0f / sc_q) : 0.0f;

  uint32_t codes[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int q0 = nf6_quantize_val(vals[4 * s + 0] * inv_sc);
    const int q1 = nf6_quantize_val(vals[4 * s + 1] * inv_sc);
    const int q2 = nf6_quantize_val(vals[4 * s + 2] * inv_sc);
    const int q3 = nf6_quantize_val(vals[4 * s + 3] * inv_sc);
    codes[s] = static_cast<uint32_t>(q0 + 6 * (q1 + 6 * (q2 + 6 * q3)));
  }

  const uint32_t lo32 = codes[0] | (codes[1] << 11) | ((codes[2] & 0x3FFu) << 22);
  const uint32_t sc_nib = (static_cast<uint32_t>(sc_bits) >> (4 * t)) & 0xFu;
  const uint16_t hi16 = static_cast<uint16_t>((codes[2] >> 10) | (codes[3] << 1) | (sc_nib << 12));

  const int cta_n = g_tile >> 2;
  const int sub_tile = ((g_tile & 3) << 1) | is_up;
  const int num_kg = K >> 6;
  const size_t tile_idx = (static_cast<size_t>(cta_n) * num_kg + kg) * kNf6NTilesPerCTA + sub_tile;
  uint8_t* dst = out + tile_idx * kNf6TileBytes;

  reinterpret_cast<uint32_t*>(dst)[lane] = lo32;
  reinterpret_cast<uint16_t*>(dst + 128)[lane] = hi16;
}

template <int kMaxM, int kStages>
struct alignas(16) Nf6StageSmem {
  uint8_t w[kStages][kNf6StepWBytes];            // 5*1536 = 7,680 B
  __nv_bfloat16 a[kStages][kMaxM][64];           // 5*2048 = 10,240 B (M<=16) or 5*1024 = 5,120 B (M<=8)
  uint32_t lut[kNf6PairLutSize];                 // 144 B -> Total = 18,064 B (M<=16), 4 CTAs/SM = 72 KiB!
};

__device__ __forceinline__ __nv_bfloat162 nf6_unpack_scale(uint32_t hi, int t) {
  uint32_t sn = (hi >> 12) << (4 * t);
  sn |= __shfl_xor_sync(0xffffffffu, sn, 1);
  sn |= __shfl_xor_sync(0xffffffffu, sn, 2);
  const uint32_t sc_u32 = sn | (sn << 16);
  return *reinterpret_cast<const __nv_bfloat162*>(&sc_u32);
}

// Direct / split-K NF6 W3.0A16 Tensor-Core GEMM (4 CTAs/SM, dual SM121 PDL, 2 MiB TLB prefetch).
template <bool kSplitK, bool kM16 = false>
__global__ void __launch_bounds__(128, 4) nf6_mma_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ C_fp32,
    int M, int N, int K, int split_k, int pdl_mode) {
  constexpr int kStages = 5;
  __shared__ Nf6StageSmem<kM16 ? 16 : 8, kStages> smem;
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

  // Populate 1,296-entry BF16x4 LUT in SMEM (~11 iters/thread, overlapped with initial weight prefetch).
  nf6_init_lut(smem.lut, tid);

  const size_t cta_stride_bytes = static_cast<size_t>(num_kg) * kNf6StepWBytes;
  const uint8_t* w_cta = W_packed + static_cast<size_t>(cta_n) * cta_stride_bytes;

  if (cta_n == 0 && sk_idx == 0) {
    const size_t total_bytes = static_cast<size_t>(N / kNf6RowsPerCTA) * cta_stride_bytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kNf6Threads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_swiz8 = a_col8 ^ (a_row & 7);
  const bool a_valid = a_row < M;

  auto issue_w_only = [&](int kg, int st) {
    if (tid < kNf6StepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      nf6_cp_async16_cg(sd, w_cta + static_cast<size_t>(kg) * kNf6StepWBytes + tid * 16);
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
    if (tid < kNf6StepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      nf6_cp_async16_cg(sd, w_cta + static_cast<size_t>(kg) * kNf6StepWBytes + tid * 16);
    }
    if constexpr (kM16) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid ? (A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8) : A;
      nf6_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid ? (A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8) : A;
      nf6_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kStages - 1;
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
    if constexpr (kStages == 5) {
      if (rem >= 3) asm volatile("cp.async.wait_group 3;\n" ::: "memory");
      else if (rem == 2) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
      else if (rem == 1) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
      else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    } else {
      if (rem >= 2) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
      else if (rem == 1) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
      else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();

    const int next_i = i + kPreload;
    if (next_i < my_kgs) {
      const int next_st = (st == 0) ? (kStages - 1) : (st - 1);
      issue_stage(kg_start + next_i, next_st);
    }

    const uint8_t* wt0 = &smem.w[st][(warp * 2 + 0) * kNf6TileBytes];
    const uint8_t* wt1 = &smem.w[st][(warp * 2 + 1) * kNf6TileBytes];
    const uint32_t lo0 = reinterpret_cast<const uint32_t*>(wt0)[lane];
    const uint32_t hi0 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wt0 + 128)[lane]);
    const __nv_bfloat162 sc0_2 = nf6_unpack_scale(hi0, t);
    const uint32_t c0_arr[4] = {
        lo0 & 0x7FFu, (lo0 >> 11) & 0x7FFu, (lo0 >> 22) | ((hi0 & 1u) << 10), (hi0 >> 1) & 0x7FFu
    };

    const uint32_t lo1 = reinterpret_cast<const uint32_t*>(wt1)[lane];
    const uint32_t hi1 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wt1 + 128)[lane]);
    const __nv_bfloat162 sc1_2 = nf6_unpack_scale(hi1, t);
    const uint32_t c1_arr[4] = {
        lo1 & 0x7FFu, (lo1 >> 11) & 0x7FFu, (lo1 >> 22) | ((hi1 & 1u) << 10), (hi1 >> 1) & 0x7FFu
    };

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        const int ldm_col8 = (2 * s + (lane >> 4)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 15][ldm_col8 * 8]));
        nf6_ldmatrix_x4(a0, a1, a2, a3, a_addr);
      } else {
        const int ldm_col8 = (2 * s + ((lane >> 3) & 1)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 7][ldm_col8 * 8]));
        nf6_ldmatrix_x2(a0, a2, a_addr);
      }
      uint32_t b0, b1, b2, b3;
      nf6_deq_quad(smem.lut, c0_arr[s], sc0_2, b0, b1);
      nf6_deq_quad(smem.lut, c1_arr[s], sc1_2, b2, b3);
      if constexpr (kM16) {
        nf6_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0, a1, a2, a3, b0, b1);
        nf6_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0, a1, a2, a3, b2, b3);
      } else {
        nf6_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b0, b1);
        nf6_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, b2, b3);
      }
    }
    st = (st + 1 == kStages) ? 0 : (st + 1);
  }

  const int col0 = cta_n * kNf6RowsPerCTA + (warp * 2 + 0) * 8 + 2 * t;
  const int col1 = cta_n * kNf6RowsPerCTA + (warp * 2 + 1) * 8 + 2 * t;
  if (g < M) {
    if constexpr (!kSplitK) {
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) = nf6_bf16x2_u32(d0_0, d1_0);
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col1) = nf6_bf16x2_u32(d0_1, d1_1);
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
        *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(ghi) * N + col0) = nf6_bf16x2_u32(d2_0, d3_0);
        *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(ghi) * N + col1) = nf6_bf16x2_u32(d2_1, d3_1);
      } else {
        float* p0 = C_fp32 + (static_cast<size_t>(sk_idx) * M + ghi) * N + col0;
        float* p1 = C_fp32 + (static_cast<size_t>(sk_idx) * M + ghi) * N + col1;
        p0[0] = d2_0; p0[1] = d3_0;
        p1[0] = d2_1; p1[1] = d3_1;
      }
    }
  }
}

__global__ void nf6_reduce_splitk_kernel(
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
  out_v.x = nf6_bf16x2_u32(acc.x, acc.y);
  out_v.y = nf6_bf16x2_u32(acc.z, acc.w);
  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
  reinterpret_cast<uint2*>(C)[vidx] = out_v;
}

// Fused paired gate_up + SiLU*Mul NF6 W3.0A16 GEMM (4 CTAs/SM, dual SM121 PDL, 2 MiB TLB prefetch).
template <bool kM16>
__global__ void __launch_bounds__(128, 4) nf6_mma_gemm_swiglu_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    int M, int I, int K, int pdl_mode) {
  constexpr int kStages = 5;
  __shared__ Nf6StageSmem<kM16 ? 16 : 8, kStages> smem;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;
  const int cta_n = blockIdx.x;

  nf6_init_lut(smem.lut, tid);

  const size_t cta_stride_bytes = static_cast<size_t>(num_kg) * kNf6StepWBytes;
  const uint8_t* w_cta = W_packed + static_cast<size_t>(cta_n) * cta_stride_bytes;

  if (cta_n == 0) {
    const size_t total_bytes = static_cast<size_t>(I / 32) * cta_stride_bytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kNf6Threads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_swiz8 = a_col8 ^ (a_row & 7);
  const bool a_valid = a_row < M;

  auto issue_w_only = [&](int kg, int st) {
    if (tid < kNf6StepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      nf6_cp_async16_cg(sd, w_cta + static_cast<size_t>(kg) * kNf6StepWBytes + tid * 16);
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
    if (tid < kNf6StepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      nf6_cp_async16_cg(sd, w_cta + static_cast<size_t>(kg) * kNf6StepWBytes + tid * 16);
    }
    if constexpr (kM16) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid ? (A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8) : A;
      nf6_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid ? (A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8) : A;
      nf6_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kStages - 1;
  const int preload = (num_kg < kPreload) ? num_kg : kPreload;
  if ((pdl_mode & 1) != 0) {
    for (int i = 0; i < preload; ++i) issue_w_only(i, i);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = 0; i < preload; ++i) load_a_sync(i, i);
  } else {
    for (int i = 0; i < preload; ++i) issue_stage(i, i);
  }

  float d0_g = 0.0f, d1_g = 0.0f, d2_g = 0.0f, d3_g = 0.0f;
  float d0_u = 0.0f, d1_u = 0.0f, d2_u = 0.0f, d3_u = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int trig_i = ((pdl_mode & 2) != 0) ? ((num_kg > 6) ? (num_kg - 6) : 0) : -1;

  int st = 0;
  for (int i = 0; i < num_kg; ++i) {
    if (i == trig_i) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    const int rem = num_kg - 1 - i;
    if constexpr (kStages == 5) {
      if (rem >= 3) asm volatile("cp.async.wait_group 3;\n" ::: "memory");
      else if (rem == 2) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
      else if (rem == 1) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
      else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    } else {
      if (rem >= 2) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
      else if (rem == 1) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
      else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();

    const int next_i = i + kPreload;
    if (next_i < num_kg) {
      const int next_st = (st == 0) ? (kStages - 1) : (st - 1);
      issue_stage(next_i, next_st);
    }

    const uint8_t* wtg = &smem.w[st][(warp * 2 + 0) * kNf6TileBytes];
    const uint8_t* wtu = &smem.w[st][(warp * 2 + 1) * kNf6TileBytes];
    const uint32_t log = reinterpret_cast<const uint32_t*>(wtg)[lane];
    const uint32_t hig = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wtg + 128)[lane]);
    const __nv_bfloat162 scg_2 = nf6_unpack_scale(hig, t);
    const uint32_t cg[4] = {
        log & 0x7FFu, (log >> 11) & 0x7FFu, (log >> 22) | ((hig & 1u) << 10), (hig >> 1) & 0x7FFu
    };

    const uint32_t lou = reinterpret_cast<const uint32_t*>(wtu)[lane];
    const uint32_t hiu = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wtu + 128)[lane]);
    const __nv_bfloat162 scu_2 = nf6_unpack_scale(hiu, t);
    const uint32_t cu[4] = {
        lou & 0x7FFu, (lou >> 11) & 0x7FFu, (lou >> 22) | ((hiu & 1u) << 10), (hiu >> 1) & 0x7FFu
    };

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        const int ldm_col8 = (2 * s + (lane >> 4)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 15][ldm_col8 * 8]));
        nf6_ldmatrix_x4(a0, a1, a2, a3, a_addr);
      } else {
        const int ldm_col8 = (2 * s + ((lane >> 3) & 1)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 7][ldm_col8 * 8]));
        nf6_ldmatrix_x2(a0, a2, a_addr);
      }
      uint32_t b0, b1, c0, c1;
      nf6_deq_quad(smem.lut, cg[s], scg_2, b0, b1);
      nf6_deq_quad(smem.lut, cu[s], scu_2, c0, c1);
      if constexpr (kM16) {
        nf6_mma_m16n8k16_full(d0_g, d1_g, d2_g, d3_g, a0, a1, a2, a3, b0, b1);
        nf6_mma_m16n8k16_full(d0_u, d1_u, d2_u, d3_u, a0, a1, a2, a3, c0, c1);
      } else {
        nf6_mma_m16n8k16(d0_g, d1_g, d2_g, d3_g, a0, a2, b0, b1);
        nf6_mma_m16n8k16(d0_u, d1_u, d2_u, d3_u, a0, a2, c0, c1);
      }
    }
    st = (st + 1 == kStages) ? 0 : (st + 1);
  }

  const int col = cta_n * 32 + warp * 8 + 2 * t;
  if (g < M) {
    *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * I + col) =
        nf6_bf16x2_u32(nf6_swiglu_f32(d0_g, d0_u), nf6_swiglu_f32(d1_g, d1_u));
  }
  if constexpr (kM16) {
    if (has_hi) {
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * I + col) =
          nf6_bf16x2_u32(nf6_swiglu_f32(d2_g, d2_u), nf6_swiglu_f32(d3_g, d3_u));
    }
  }
}

int nf6_pdl_mode() {
  static const int mode = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_PDL");
    if (!s || !*s) return 3;
    return (std::atoi(s) != 0) ? 3 : 0;
  }();
  return mode;
}

template <typename... KArgs, typename... Args>
static inline void nf6_launch_pdl(
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
  TORCH_CHECK(err == cudaSuccess, "nf6_mma kernel launch failed: ", cudaGetErrorString(err));
}

float* ensure_nf6_splitk_partials(const torch::TensorOptions& opts, int64_t need) {
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

}  // namespace

torch::Tensor nf6_pack(torch::Tensor W) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kBFloat16 && W.dim() == 2 && W.is_contiguous(),
              "nf6_pack expects contiguous CUDA BF16 [N, K]");
  const int N = static_cast<int>(W.size(0));
  const int K = static_cast<int>(W.size(1));
  TORCH_CHECK(N % kNf6RowsPerCTA == 0 && K % 64 == 0, "nf6_pack requires N%64==0 and K%64==0");
  const int64_t total_bytes = static_cast<int64_t>(N / 8) * static_cast<int64_t>(K / 64) * kNf6TileBytes;
  auto out = torch::empty({total_bytes}, W.options().dtype(torch::kUInt8));
  dim3 grid(N / 8, K / 64);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  nf6_pack_kernel<<<grid, 32, 0, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()),
      out.data_ptr<uint8_t>(),
      N, K);
  return out;
}

torch::Tensor nf6_gemm(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous(),
              "nf6_gemm expects contiguous CUDA BF16 [M, K]");
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && A.size(1) == K && N % kNf6RowsPerCTA == 0 && K % 64 == 0,
              "nf6_gemm shape mismatch");
  auto C = torch::empty({M, N}, A.options());
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n_blocks = static_cast<int>(N / kNf6RowsPerCTA);
  const bool use_m16 = (M > 8);
  const int pdl_mode = nf6_pdl_mode();
  int split_k = 1;
  if (n_blocks <= 128 && K >= 16384) {
    split_k = (K >= 28672) ? 4 : 3;
  }
  if (split_k == 1) {
    const dim3 grid(n_blocks, 1, 1);
    if (use_m16) {
      nf6_launch_pdl(
          nf6_mma_gemm_kernel<false, true>, grid, kNf6Threads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode);
    } else {
      nf6_launch_pdl(
          nf6_mma_gemm_kernel<false, false>, grid, kNf6Threads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode);
    }
  } else {
    const int64_t need = static_cast<int64_t>(split_k) * M * N;
    float* partials_ptr = ensure_nf6_splitk_partials(A.options(), need);
    const dim3 grid(n_blocks, split_k, 1);
    if (use_m16) {
      nf6_launch_pdl(
          nf6_mma_gemm_kernel<true, true>, grid, kNf6Threads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          static_cast<__nv_bfloat16*>(nullptr), partials_ptr,
          M, static_cast<int>(N), static_cast<int>(K), split_k, pdl_mode);
    } else {
      nf6_launch_pdl(
          nf6_mma_gemm_kernel<true, false>, grid, kNf6Threads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          static_cast<__nv_bfloat16*>(nullptr), partials_ptr,
          M, static_cast<int>(N), static_cast<int>(K), split_k, pdl_mode);
    }
    const int vtotal = (M * static_cast<int>(N)) >> 2;
    const dim3 rgrid((vtotal + 255) / 256, 1, 1);
    nf6_launch_pdl(
        nf6_reduce_splitk_kernel, rgrid, 256, st, pdl_mode,
        static_cast<const float*>(partials_ptr),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, static_cast<int>(N), split_k, pdl_mode);
  }
  return C;
}

torch::Tensor nf6_pack_swiglu(torch::Tensor W) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kBFloat16 && W.dim() == 2 && W.is_contiguous(),
              "nf6_pack_swiglu expects contiguous CUDA BF16 [2*I, K]");
  const int N2 = static_cast<int>(W.size(0));
  const int K = static_cast<int>(W.size(1));
  TORCH_CHECK(N2 % 64 == 0 && K % 64 == 0, "nf6_pack_swiglu requires (2*I)%64==0 and K%64==0");
  const int I = N2 / 2;
  const int64_t total_bytes = static_cast<int64_t>(N2 / 8) * static_cast<int64_t>(K / 64) * kNf6TileBytes;
  auto out = torch::empty({total_bytes}, W.options().dtype(torch::kUInt8));
  dim3 grid(N2 / 8, K / 64);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  nf6_pack_swiglu_kernel<<<grid, 32, 0, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()),
      out.data_ptr<uint8_t>(),
      I, K);
  return out;
}

torch::Tensor nf6_gemm_swiglu(torch::Tensor A, torch::Tensor W_packed, int64_t I, int64_t K) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous(),
              "nf6_gemm_swiglu expects contiguous CUDA BF16 [M, K]");
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && A.size(1) == K && I % 32 == 0 && K % 64 == 0,
              "nf6_gemm_swiglu shape mismatch");
  auto C = torch::empty({M, I}, A.options());
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n_blocks = static_cast<int>(I / 32);
  const int pdl_mode = nf6_pdl_mode();
  const dim3 grid(n_blocks, 1, 1);
  if (M > 8) {
    nf6_launch_pdl(
        nf6_mma_gemm_swiglu_kernel<true>, grid, kNf6Threads, st, pdl_mode,
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, static_cast<int>(I), static_cast<int>(K), pdl_mode);
  } else {
    nf6_launch_pdl(
        nf6_mma_gemm_swiglu_kernel<false>, grid, kNf6Threads, st, pdl_mode,
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, static_cast<int>(I), static_cast<int>(K), pdl_mode);
  }
  return C;
}
