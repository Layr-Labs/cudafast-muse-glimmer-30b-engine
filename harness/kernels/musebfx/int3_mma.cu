// SPDX-License-Identifier: Apache-2.0
// muse_int3_mma: 3-bit asymmetric group-64 W3A16 weight-only Tensor-Core GEMM for the pinned DFlash drafter.
// Format (3.50 bits/weight, 0 wasted bits, 16-byte aligned):
//   Each 8x64 tile (8 output rows x 64 K cols) is 224 bytes (14 x uint4):
//     [0..127]   uint32_t lo2[32]  : for lane l=(r<<2)|t, 16 x 2-bit low bits (s=0..3, 4 weights/k16)
//     [128..191] uint16_t hi1[32]  : for lane l=(r<<2)|t, 16 x 1-bit high bits (s=0..3, 4 weights/k16)
//     [192..223] __nv_bfloat162 sz[8] : (scale, zero) for rows r=0..7 (w = q * scale + zero, q in 0..7)
//   Tiles are stored in [N/64, K/64, 8] order (1792 bytes = 112 x uint4 per 4-warp CTA per K=64 step).
//   Runs on SM121 with 5-stage cp.async.cg.L2::256B (no L2::cache_hint / createpolicy -> SM121 odd-UR safe),
//   1 barrier/step, 2x N-tile A reuse, XOR-swizzled ldmatrix, BF16x2 HADD2+HFMA2 dequant, 4 CTAs/SM,
//   dual SM121 PDL, and 2 MiB TLB prefetch.
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
constexpr int kStages = 5;                       // 18.75 KiB SMEM (M<=16) -> 4 CTAs/SM (192 CTAs)

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

__device__ __forceinline__ void int3_cp_async16_cg(uint32_t sd, const void* gs) {
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ void int3_cp_async16_ca_z(uint32_t sd, const void* gs, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(sd), "l"(gs), "r"(src_bytes));
}

__device__ __forceinline__ void int3_ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(smem_addr));
}

__device__ __forceinline__ void int3_ldmatrix_x2(
    uint32_t& r0, uint32_t& r1, uint32_t smem_addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(r0), "=r"(r1)
      : "r"(smem_addr));
}

__device__ __forceinline__ float2 int3_ldcg_f2(const float2* ptr) {
  float2 v;
  asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint32_t int3_bf16x2_u32(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t int3_unpack_hi1(uint16_t hi1) {
  const uint32_t u = static_cast<uint32_t>(hi1);
  return (u & 0x00FFu) | ((u & 0xFF00u) << 8);
}

// Native BF16x2 3-bit dequant using pair-interleaved lo2/hi1:
// - kNf3=false (uniform): 0x41004100 | ((lo2<<4)&0x00300030) | ((hi32<<6)&0x00400040) == bf16x2(8+q0, 8+q1)
//   -> HADD2(-8.0) -> HFMA2(sc2, zr2): bit-exact with rn_bf16(fma(float(q), sc, zr)), 0 I2F, 0 F2FP.
// - kNf3=true (non-uniform NormalFloat3, exact BF16 centroids +-0.109375, +-0.3359375, +-0.609375, +-1.0):
//   register PRMT (__byte_perm) lookup of 2-bit magnitude + LOP3 sign bit from hi32 -> HFMA2(sc2, zr2).
//   Zero extra weight bytes, zero SMEM LUT loads, +0.34..+0.74 dB SQNR across Gaussian/Student-t/Laplace MLP weights.
template <bool kNf3>
__device__ __forceinline__ void int3_deq4_k16(
    uint32_t lo2, uint32_t hi32, __nv_bfloat162 sc2, __nv_bfloat162 zr2, uint32_t& b0, uint32_t& b1) {
  if constexpr (kNf3) {
    constexpr uint32_t kNf3Lut01 = 0x3EAC3DE0u;  // bf16(0.109375) | (bf16(0.3359375) << 16)
    constexpr uint32_t kNf3Lut23 = 0x3F803F1Cu;  // bf16(0.609375) | (bf16(1.000000) << 16)
    const uint32_t p0_lo = (lo2 & 3u) << 1;
    const uint32_t p0_hi = (lo2 >> 15) & 6u;
    const uint32_t sel0 = p0_lo | (p0_lo << 4) | (p0_hi << 8) | (p0_hi << 12) | 0x1010u;
    const uint32_t q01_u = __byte_perm(kNf3Lut01, kNf3Lut23, sel0) | ((hi32 << 15) & 0x80008000u);

    const uint32_t p1_lo = (lo2 >> 1) & 6u;
    const uint32_t p1_hi = (lo2 >> 17) & 6u;
    const uint32_t sel1 = p1_lo | (p1_lo << 4) | (p1_hi << 8) | (p1_hi << 12) | 0x1010u;
    const uint32_t q23_u = __byte_perm(kNf3Lut01, kNf3Lut23, sel1) | ((hi32 << 14) & 0x80008000u);

    const __nv_bfloat162 w01 = __hfma2(*reinterpret_cast<const __nv_bfloat162*>(&q01_u), sc2, zr2);
    const __nv_bfloat162 w23 = __hfma2(*reinterpret_cast<const __nv_bfloat162*>(&q23_u), sc2, zr2);
    b0 = *reinterpret_cast<const uint32_t*>(&w01);
    b1 = *reinterpret_cast<const uint32_t*>(&w23);
  } else {
    const uint32_t t0 = 0x41004100u | ((lo2 << 4) & 0x00300030u) | ((hi32 << 6) & 0x00400040u);
    const uint32_t t1 = 0x41004100u | ((lo2 << 2) & 0x00300030u) | ((hi32 << 5) & 0x00400040u);
    const uint32_t neg8_u = 0xC100C100u;
    const __nv_bfloat162 neg8 = *reinterpret_cast<const __nv_bfloat162*>(&neg8_u);
    const __nv_bfloat162 q01 = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&t0), neg8);
    const __nv_bfloat162 q23 = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&t1), neg8);
    const __nv_bfloat162 w01 = __hfma2(q01, sc2, zr2);
    const __nv_bfloat162 w23 = __hfma2(q23, sc2, zr2);
    b0 = *reinterpret_cast<const uint32_t*>(&w01);
    b1 = *reinterpret_cast<const uint32_t*>(&w23);
  }
}

__device__ __forceinline__ float int3_nf3_quant_val(float u, int& mag, int& sgn) {
  constexpr float kC0 = 0.109375f;
  constexpr float kC1 = 0.3359375f;
  constexpr float kC2 = 0.609375f;
  constexpr float kC3 = 1.000000f;
  const float au = fabsf(u);
  float cmag;
  if (au >= 0.47265625f) {
    if (au >= 0.8046875f) { mag = 3; cmag = kC3; }
    else { mag = 2; cmag = kC2; }
  } else {
    if (au >= 0.22265625f) { mag = 1; cmag = kC1; }
    else { mag = 0; cmag = kC0; }
  }
  sgn = (u < 0.0f) ? 1 : 0;
  return sgn ? -cmag : cmag;
}

__device__ __forceinline__ float int3_subwarp4_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}

// Shared group-64 MSE+ALS packer for both int3_pack_kernel and int3_pack_swiglu_kernel.
// kNf3=true packs non-uniform NormalFloat3 centroids (+-0.109375, +-0.3359375, +-0.609375, +-1.0);
// kNf3=false packs uniform 0..7 levels. Both minimize exact BF16-dequantized MSE via 5-iter closed-form ALS.
template <bool kMseOpt, bool kNf3>
__device__ __forceinline__ void int3_pack_group_16(
    const float vals[16],
    int r,
    uint32_t& lo2_out,
    uint32_t& hi1_out,
    __nv_bfloat162& sz_out) {
  float rmin = 1e30f;
  float rmax = -1e30f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    rmin = fminf(rmin, vals[i]);
    rmax = fmaxf(rmax, vals[i]);
  }
  rmin = fminf(rmin, __shfl_xor_sync(0xffffffffu, rmin, 1));
  rmin = fminf(rmin, __shfl_xor_sync(0xffffffffu, rmin, 2));
  rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, 1));
  rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, 2));

  const float mid = 0.5f * (rmin + rmax);
  const float half = 0.5f * (rmax - rmin);
  float sc_mm = kNf3 ? half : ((rmax - rmin) * (1.0f / 7.0f));
  const float zr_mm = kNf3 ? mid : rmin;
  if (sc_mm < 1e-30f) sc_mm = 1e-30f;
  __nv_bfloat162 sz_bf = __floats2bfloat162_rn(sc_mm, zr_mm);

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
        float c_val;
        if constexpr (kNf3) {
          int mag, sgn;
          c_val = int3_nf3_quant_val((vals[i] - zr_q) * inv_sc, mag, sgn);
        } else {
          int q = __float2int_rn((vals[i] - zr_q) * inv_sc);
          if (q < 0) q = 0;
          if (q > 7) q = 7;
          c_val = static_cast<float>(q);
        }
        const float w_hat = __bfloat162float(
            __float2bfloat16_rn(__fmaf_rn(c_val, sc_q, zr_q)));
        const float diff = vals[i] - w_hat;
        err = __fmaf_rn(diff, diff, err);
      }
      out_mse = int3_subwarp4_sum(err);
    };

    __nv_bfloat162 best_sz = sz_bf;
    float best_mse = 0.0f;
    eval_cand(sc_mm, zr_mm, best_sz, best_mse);

    __nv_bfloat162 try_sz;
    float try_mse;
    auto consider_interval = [&](float lo, float hi) {
      const float sc_c = kNf3 ? (0.5f * (hi - lo)) : ((hi - lo) * (1.0f / 7.0f));
      const float zr_c = kNf3 ? (0.5f * (lo + hi)) : lo;
      eval_cand(sc_c, zr_c, try_sz, try_mse);
      if (try_mse < best_mse) {
        best_mse = try_mse;
        best_sz = try_sz;
      }
    };

    constexpr float kMidAlphas[4] = {0.92f, 0.84f, 0.76f, 0.68f};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float h = half * kMidAlphas[c];
      consider_interval(mid - h, mid + h);
    }

    constexpr float kStdCs[6] = {3.00f, 2.60f, 2.30f, 2.00f, 1.75f, 1.50f};
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
    const float min_denom = kNf3 ? 1e-3f : 0.5f;
#pragma unroll
    for (int it = 0; it < 5; ++it) {
      const float inv_cur = (cur_sc > 1e-30f) ? (1.0f / cur_sc) : 0.0f;
      float lsq = 0.0f, lsqq = 0.0f, lsqw = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        float fq;
        if constexpr (kNf3) {
          int mag, sgn;
          fq = int3_nf3_quant_val((vals[i] - cur_zr) * inv_cur, mag, sgn);
        } else {
          int q = __float2int_rn((vals[i] - cur_zr) * inv_cur);
          if (q < 0) q = 0;
          if (q > 7) q = 7;
          fq = static_cast<float>(q);
        }
        lsq += fq;
        lsqq = __fmaf_rn(fq, fq, lsqq);
        lsqw = __fmaf_rn(fq, vals[i], lsqw);
      }
      const float sq = int3_subwarp4_sum(lsq);
      const float sqq = int3_subwarp4_sum(lsqq);
      const float sqw = int3_subwarp4_sum(lsqw);
      const float denom = 64.0f * sqq - sq * sq;
      if (denom > min_denom) {
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
    uint32_t q_lo, q_hi;
    if constexpr (kNf3) {
      int mag, sgn;
      (void)int3_nf3_quant_val((vals[i] - zr_q) * inv_sc, mag, sgn);
      q_lo = static_cast<uint32_t>(mag);
      q_hi = static_cast<uint32_t>(sgn);
    } else {
      int q = __float2int_rn((vals[i] - zr_q) * inv_sc);
      if (q < 0) q = 0;
      if (q > 7) q = 7;
      q_lo = static_cast<uint32_t>(q & 3);
      q_hi = static_cast<uint32_t>((q >> 2) & 1);
    }
    const int p = i >> 1;
    const int h = i & 1;
    lo2 |= (q_lo << (2 * p + 16 * h));
    hi1 |= (q_hi << (p + 8 * h));
  }
  lo2_out = lo2;
  hi1_out = hi1;
  sz_out = sz_bf;
}

template <bool kMseOpt, bool kNf3>
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
  const __nv_bfloat16* wrow = W + static_cast<size_t>(row) * K + k_base;
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int kb = s * 16 + 2 * t;
    vals[4 * s + 0] = __bfloat162float(wrow[kb + 0]);
    vals[4 * s + 1] = __bfloat162float(wrow[kb + 1]);
    vals[4 * s + 2] = __bfloat162float(wrow[kb + 8]);
    vals[4 * s + 3] = __bfloat162float(wrow[kb + 9]);
  }

  uint32_t lo2 = 0u, hi1 = 0u;
  __nv_bfloat162 sz_bf;
  int3_pack_group_16<kMseOpt, kNf3>(vals, r, lo2, hi1, sz_bf);

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
template <bool kMseOpt, bool kNf3>
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
  const __nv_bfloat16* wrow = W + static_cast<size_t>(row) * K + k_base;
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int kb = s * 16 + 2 * t;
    vals[4 * s + 0] = __bfloat162float(wrow[kb + 0]);
    vals[4 * s + 1] = __bfloat162float(wrow[kb + 1]);
    vals[4 * s + 2] = __bfloat162float(wrow[kb + 8]);
    vals[4 * s + 3] = __bfloat162float(wrow[kb + 9]);
  }

  uint32_t lo2 = 0u, hi1 = 0u;
  __nv_bfloat162 sz_bf;
  int3_pack_group_16<kMseOpt, kNf3>(vals, r, lo2, hi1, sz_bf);

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
  uint8_t w[kStages][kStepWBytes];         // 5 * 1792 = 8960 B
  __nv_bfloat16 a[kStages][kMaxM][64];     // 5 * 1024 = 5120 B (M<=8) or 10240 B (M<=16)
};                                         // Total = 13.75 KiB (M<=8) or 18.75 KiB (M<=16), 4 CTAs/SM

// Direct / split-K W3A16 Tensor-Core GEMM (cp.async.cg.L2::256B, XOR-swizzled ldmatrix,
// BF16x2 HADD2+HFMA2 dequant, dual SM121 PDL, 4 CTAs/SM, SM121 odd-UR safe).
template <bool kSplitK, bool kM16 = false, bool kNf3 = true>
__global__ void __launch_bounds__(128, 4) int3_mma_gemm_kernel(
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

  // Issue 2 MiB TLB prefetch from block (0, 0) (immutable W_packed: safe before PDL wait).
  if (cta_n == 0 && sk_idx == 0) {
    const size_t total_bytes = static_cast<size_t>(N / kRowsPerCTA) * cta_stride_bytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_swiz8 = a_col8 ^ (a_row & 7);
  const bool a_valid = a_row < M;

  auto issue_w_only = [&](int kg, int st) {
    if (tid < kStepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int3_cp_async16_cg(sd, w_cta + static_cast<size_t>(kg) * kStepWBytes + tid * 16);
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
    if (tid < kStepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int3_cp_async16_cg(sd, w_cta + static_cast<size_t>(kg) * kStepWBytes + tid * 16);
    }
    if constexpr (kM16) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid ? (A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8) : A;
      int3_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid ? (A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8) : A;
      int3_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
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
    uint32_t lo2_0 = reinterpret_cast<const uint32_t*>(wt0)[lane];
    uint32_t hi32_0 = int3_unpack_hi1(reinterpret_cast<const uint16_t*>(wt0 + 128)[lane]);
    const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 192)[g];
    const __nv_bfloat162 sc0_2 = __bfloat162bfloat162(sz0.x);
    const __nv_bfloat162 zr0_2 = __bfloat162bfloat162(sz0.y);

    uint32_t lo2_1 = reinterpret_cast<const uint32_t*>(wt1)[lane];
    uint32_t hi32_1 = int3_unpack_hi1(reinterpret_cast<const uint16_t*>(wt1 + 128)[lane]);
    const __nv_bfloat162 sz1 = reinterpret_cast<const __nv_bfloat162*>(wt1 + 192)[g];
    const __nv_bfloat162 sc1_2 = __bfloat162bfloat162(sz1.x);
    const __nv_bfloat162 zr1_2 = __bfloat162bfloat162(sz1.y);

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        const int ldm_col8 = (2 * s + (lane >> 4)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 15][ldm_col8 * 8]));
        int3_ldmatrix_x4(a0, a1, a2, a3, a_addr);
      } else {
        const int ldm_col8 = (2 * s + ((lane >> 3) & 1)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 7][ldm_col8 * 8]));
        int3_ldmatrix_x2(a0, a2, a_addr);
      }
      uint32_t b0, b1, c0, c1;
      int3_deq4_k16<kNf3>(lo2_0, hi32_0, sc0_2, zr0_2, b0, b1);
      lo2_0 >>= 4;
      hi32_0 >>= 2;
      int3_deq4_k16<kNf3>(lo2_1, hi32_1, sc1_2, zr1_2, c0, c1);
      lo2_1 >>= 4;
      hi32_1 >>= 2;
      if constexpr (kM16) {
        int3_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0, a1, a2, a3, b0, b1);
        int3_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0, a1, a2, a3, c0, c1);
      } else {
        int3_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b0, b1);
        int3_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, c0, c1);
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

// Vectorized float4 -> bf16x4 split-K reduction with dual SM121 PDL (N % 64 == 0 => total % 4 == 0).
__global__ void int3_reduce_splitk_kernel(
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
  out_v.x = int3_bf16x2_u32(acc.x, acc.y);
  out_v.y = int3_bf16x2_u32(acc.z, acc.w);
  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
  reinterpret_cast<uint2*>(C)[vidx] = out_v;
}

__device__ __forceinline__ float int3_swiglu_f32(float gate, float up) {
  return (gate / (1.0f + __expf(-gate))) * up;
}

// Fused paired gate_up + SiLU*Mul W3A16 GEMM (cp.async.cg.L2::256B, XOR-swizzled ldmatrix,
// BF16x2 HADD2+HFMA2 dequant, dual SM121 PDL, 4 CTAs/SM, SM121 odd-UR safe).
// Each CTA computes 32 intermediate outputs (4 warps x 8 cols): tile 2*w = gate, tile 2*w+1 = up.
template <bool kM16, bool kNf3 = true>
__global__ void __launch_bounds__(128, 4) int3_mma_gemm_swiglu_kernel(
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

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_swiz8 = a_col8 ^ (a_row & 7);
  const bool a_valid = a_row < M;

  auto issue_w_only = [&](int kg, int st) {
    if (tid < kStepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int3_cp_async16_cg(sd, w_cta + static_cast<size_t>(kg) * kStepWBytes + tid * 16);
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
    if (tid < kStepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int3_cp_async16_cg(sd, w_cta + static_cast<size_t>(kg) * kStepWBytes + tid * 16);
    }
    if constexpr (kM16) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid ? (A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8) : A;
      int3_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid ? (A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8) : A;
      int3_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kStages - 1;  // 4
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
  const int trig_i = ((pdl_mode & 2) != 0) ? ((num_kg > 6) ? (num_kg - 6) : 0) : -1;

  int st = 0;
  for (int i = 0; i < num_kg; ++i) {
    if (i == trig_i) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
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

    const int next_i = i + kPreload;
    if (next_i < num_kg) {
      const int next_st = (st == 0) ? (kStages - 1) : (st - 1);
      issue_stage(next_i, next_st);
    }

    const uint8_t* wtg = &smem.w[st][(warp * 2 + 0) * kTileBytes];
    const uint8_t* wtu = &smem.w[st][(warp * 2 + 1) * kTileBytes];
    uint32_t lo2_g = reinterpret_cast<const uint32_t*>(wtg)[lane];
    uint32_t hi32_g = int3_unpack_hi1(reinterpret_cast<const uint16_t*>(wtg + 128)[lane]);
    const __nv_bfloat162 szg = reinterpret_cast<const __nv_bfloat162*>(wtg + 192)[g];
    const __nv_bfloat162 scg_2 = __bfloat162bfloat162(szg.x);
    const __nv_bfloat162 zrg_2 = __bfloat162bfloat162(szg.y);

    uint32_t lo2_u = reinterpret_cast<const uint32_t*>(wtu)[lane];
    uint32_t hi32_u = int3_unpack_hi1(reinterpret_cast<const uint16_t*>(wtu + 128)[lane]);
    const __nv_bfloat162 szu = reinterpret_cast<const __nv_bfloat162*>(wtu + 192)[g];
    const __nv_bfloat162 scu_2 = __bfloat162bfloat162(szu.x);
    const __nv_bfloat162 zru_2 = __bfloat162bfloat162(szu.y);

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        const int ldm_col8 = (2 * s + (lane >> 4)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 15][ldm_col8 * 8]));
        int3_ldmatrix_x4(a0, a1, a2, a3, a_addr);
      } else {
        const int ldm_col8 = (2 * s + ((lane >> 3) & 1)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 7][ldm_col8 * 8]));
        int3_ldmatrix_x2(a0, a2, a_addr);
      }
      uint32_t b0, b1, c0, c1;
      int3_deq4_k16<kNf3>(lo2_g, hi32_g, scg_2, zrg_2, b0, b1);
      lo2_g >>= 4;
      hi32_g >>= 2;
      int3_deq4_k16<kNf3>(lo2_u, hi32_u, scu_2, zru_2, c0, c1);
      lo2_u >>= 4;
      hi32_u >>= 2;
      if constexpr (kM16) {
        int3_mma_m16n8k16_full(d0_g, d1_g, d2_g, d3_g, a0, a1, a2, a3, b0, b1);
        int3_mma_m16n8k16_full(d0_u, d1_u, d2_u, d3_u, a0, a1, a2, a3, c0, c1);
      } else {
        int3_mma_m16n8k16(d0_g, d1_g, d2_g, d3_g, a0, a2, b0, b1);
        int3_mma_m16n8k16(d0_u, d1_u, d2_u, d3_u, a0, a2, c0, c1);
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

// Persistent Stream-K W3A16 GEMM (cp.async.cg.L2::256B, XOR-swizzled ldmatrix, BF16x2 dequant,
// dual SM121 PDL, 4 CTAs/SM, SM121 odd-UR safe).
// kSwiglu=false: C is [M, N], n_blocks = N / 64.
// kSwiglu=true:  C is [M, I], n_blocks = I / 32 (tile 2*w = gate, tile 2*w+1 = up).
template <bool kM16, bool kSwiglu, bool kNf3 = true>
__global__ void __launch_bounds__(128, 4) int3_mma_streamk_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ sk_partials,    // [max_n_blocks, 16, 64] zero-initialized & self-zeroing (4+ parts)
    float* __restrict__ sk_multi_part,  // [max_n_blocks, 3, 16, 64] non-atomic 2/3-part partials (nullptr disables)
    int* __restrict__ sk_done,          // [max_n_blocks] zero-initialized & self-zeroing
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

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_swiz8 = a_col8 ^ (a_row & 7);
  const bool a_valid = a_row < M;

  auto issue_w_only = [&](int st, int w_iter) {
    if (tid < kStepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int3_cp_async16_cg(sd, W_packed + static_cast<size_t>(w_iter) * kStepWBytes + tid * 16);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };
  auto load_a_sync = [&](int st, int a_kg) {
    if constexpr (kM16) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_swiz8 * 8]) =
          a_valid
              ? __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(a_row) * K + a_kg * 64 + a_col8 * 8))
              : make_uint4(0u, 0u, 0u, 0u);
    } else if (tid < 64) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_swiz8 * 8]) =
          a_valid
              ? __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(a_row) * K + a_kg * 64 + a_col8 * 8))
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  auto issue_next = [&](int st) {
    if (tid < kStepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int3_cp_async16_cg(sd, W_packed + static_cast<size_t>(pre_iter) * kStepWBytes + tid * 16);
    }
    if constexpr (kM16) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid ? (A + static_cast<size_t>(a_row) * K + pre_kg * 64 + a_col8 * 8) : A;
      int3_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
      const __nv_bfloat16* gs = a_valid ? (A + static_cast<size_t>(a_row) * K + pre_kg * 64 + a_col8 * 8) : A;
      int3_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    ++pre_iter;
    if (++pre_kg == num_kg) pre_kg = 0;
  };

  constexpr int kPreload = kStages - 1;  // 4
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
    uint32_t lo2_0 = reinterpret_cast<const uint32_t*>(wt0)[lane];
    uint32_t hi32_0 = int3_unpack_hi1(reinterpret_cast<const uint16_t*>(wt0 + 128)[lane]);
    const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 192)[g];
    const __nv_bfloat162 sc0_2 = __bfloat162bfloat162(sz0.x);
    const __nv_bfloat162 zr0_2 = __bfloat162bfloat162(sz0.y);

    uint32_t lo2_1 = reinterpret_cast<const uint32_t*>(wt1)[lane];
    uint32_t hi32_1 = int3_unpack_hi1(reinterpret_cast<const uint16_t*>(wt1 + 128)[lane]);
    const __nv_bfloat162 sz1 = reinterpret_cast<const __nv_bfloat162*>(wt1 + 192)[g];
    const __nv_bfloat162 sc1_2 = __bfloat162bfloat162(sz1.x);
    const __nv_bfloat162 zr1_2 = __bfloat162bfloat162(sz1.y);

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        const int ldm_col8 = (2 * s + (lane >> 4)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 15][ldm_col8 * 8]));
        int3_ldmatrix_x4(a0, a1, a2, a3, a_addr);
      } else {
        const int ldm_col8 = (2 * s + ((lane >> 3) & 1)) ^ (lane & 7);
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][lane & 7][ldm_col8 * 8]));
        int3_ldmatrix_x2(a0, a2, a_addr);
      }
      uint32_t b0, b1, c0, c1;
      int3_deq4_k16<kNf3>(lo2_0, hi32_0, sc0_2, zr0_2, b0, b1);
      lo2_0 >>= 4;
      hi32_0 >>= 2;
      int3_deq4_k16<kNf3>(lo2_1, hi32_1, sc1_2, zr1_2, c0, c1);
      lo2_1 >>= 4;
      hi32_1 >>= 2;
      if constexpr (kM16) {
        int3_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0, a1, a2, a3, b0, b1);
        int3_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0, a1, a2, a3, c0, c1);
      } else {
        int3_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b0, b1);
        int3_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, c0, c1);
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
        const int tile_lo = cur_n * num_kg;
        const int tile_hi = tile_lo + num_kg;
        int num_parts = 0;
        int part_role = 0;
        if (sk_multi_part != nullptr) {
          if (seg_start_kg == 0 && cur_kg + 1 < num_kg) {
            if (cta_id + 1 < grid_ctas) {
              const int next_end = static_cast<int>((static_cast<int64_t>(cta_id + 2) * total_iters) / grid_ctas);
              if (next_end >= tile_hi) {
                num_parts = 2;
                part_role = 0;
              } else if (cta_id + 2 < grid_ctas) {
                const int next2_end = static_cast<int>((static_cast<int64_t>(cta_id + 3) * total_iters) / grid_ctas);
                if (next2_end >= tile_hi) {
                  num_parts = 3;
                  part_role = 0;
                }
              }
            }
          } else if (seg_start_kg > 0 && cur_kg + 1 < num_kg) {
            if (cta_id > 0 && cta_id + 1 < grid_ctas) {
              const int prev_start = static_cast<int>((static_cast<int64_t>(cta_id - 1) * total_iters) / grid_ctas);
              const int next_end = static_cast<int>((static_cast<int64_t>(cta_id + 2) * total_iters) / grid_ctas);
              if (prev_start <= tile_lo && next_end >= tile_hi) {
                num_parts = 3;
                part_role = 1;
              }
            }
          } else if (seg_start_kg > 0 && cur_kg + 1 == num_kg) {
            if (cta_id > 0) {
              const int prev_start = static_cast<int>((static_cast<int64_t>(cta_id - 1) * total_iters) / grid_ctas);
              if (prev_start <= tile_lo) {
                num_parts = 2;
                part_role = 1;
              } else if (cta_id >= 2) {
                const int prev2_start = static_cast<int>((static_cast<int64_t>(cta_id - 2) * total_iters) / grid_ctas);
                if (prev2_start <= tile_lo) {
                  num_parts = 3;
                  part_role = 2;
                }
              }
            }
          }
        }
        auto emit_out = [&](int row, float s0_0, float s1_0, float s0_1, float s1_1) {
          if constexpr (kSwiglu) {
            const int col = cur_n * 32 + warp * 8 + 2 * t;
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(row) * out_cols + col) =
                int3_bf16x2_u32(int3_swiglu_f32(s0_0, s0_1), int3_swiglu_f32(s1_0, s1_1));
          } else {
            const int col0 = cur_n * kRowsPerCTA + local_c0;
            const int col1 = cur_n * kRowsPerCTA + local_c1;
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(row) * out_cols + col0) = int3_bf16x2_u32(s0_0, s1_0);
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(row) * out_cols + col1) = int3_bf16x2_u32(s0_1, s1_1);
          }
        };
        if (num_parts >= 2) {
          // 2-part & 3-part non-atomic canonical Stream-K reduction: coalesced float2 stores +
          // 1 int32 arrival counter + L2-coherent float2 loads in canonical (slot0 + slot1) + slot2 order.
          float* tile_mp = sk_multi_part + static_cast<size_t>(cur_n) * (3 * 16 * 64);
          float* my_p = tile_mp + static_cast<size_t>(part_role) * (16 * 64);
          if (g < M) {
            float* row_p = my_p + g * 64;
            *reinterpret_cast<float2*>(&row_p[local_c0]) = make_float2(d0_0, d1_0);
            *reinterpret_cast<float2*>(&row_p[local_c1]) = make_float2(d0_1, d1_1);
          }
          if constexpr (kM16) {
            if (has_hi) {
              float* row_p = my_p + (g + 8) * 64;
              *reinterpret_cast<float2*>(&row_p[local_c0]) = make_float2(d2_0, d3_0);
              *reinterpret_cast<float2*>(&row_p[local_c1]) = make_float2(d2_1, d3_1);
            }
          }
          __threadfence();
          __syncthreads();
          if (tid == 0) {
            s_is_last = (atomicAdd(&sk_done[cur_n], 1) == (num_parts - 1)) ? 1 : 0;
          }
          __syncthreads();
          if (s_is_last) {
            if (num_parts == 2) {
              const float* peer_p = tile_mp + static_cast<size_t>(1 - part_role) * (16 * 64);
              if (g < M) {
                const float* row_p = peer_p + g * 64;
                const float2 p0 = int3_ldcg_f2(reinterpret_cast<const float2*>(&row_p[local_c0]));
                const float2 p1 = int3_ldcg_f2(reinterpret_cast<const float2*>(&row_p[local_c1]));
                emit_out(g, d0_0 + p0.x, d1_0 + p0.y, d0_1 + p1.x, d1_1 + p1.y);
              }
              if constexpr (kM16) {
                if (has_hi) {
                  const float* row_p = peer_p + (g + 8) * 64;
                  const float2 p0 = int3_ldcg_f2(reinterpret_cast<const float2*>(&row_p[local_c0]));
                  const float2 p1 = int3_ldcg_f2(reinterpret_cast<const float2*>(&row_p[local_c1]));
                  emit_out(g + 8, d2_0 + p0.x, d3_0 + p0.y, d2_1 + p1.x, d3_1 + p1.y);
                }
              }
            } else {
              const int peer_a = (part_role == 0) ? 1 : 0;
              const int peer_b = (part_role == 2) ? 1 : 2;
              const float* pa_base = tile_mp + static_cast<size_t>(peer_a) * (16 * 64);
              const float* pb_base = tile_mp + static_cast<size_t>(peer_b) * (16 * 64);
              auto sum3 = [&](float mx, float my, float2 va, float2 vb, float& ox, float& oy) {
                float x0, x1, x2, y0, y1, y2;
                if (part_role == 0) {
                  x0 = mx; x1 = va.x; x2 = vb.x;
                  y0 = my; y1 = va.y; y2 = vb.y;
                } else if (part_role == 1) {
                  x0 = va.x; x1 = mx; x2 = vb.x;
                  y0 = va.y; y1 = my; y2 = vb.y;
                } else {
                  x0 = va.x; x1 = vb.x; x2 = mx;
                  y0 = va.y; y1 = vb.y; y2 = my;
                }
                ox = (x0 + x1) + x2;
                oy = (y0 + y1) + y2;
              };
              if (g < M) {
                const float* row_a = pa_base + g * 64;
                const float* row_b = pb_base + g * 64;
                const float2 a0 = int3_ldcg_f2(reinterpret_cast<const float2*>(&row_a[local_c0]));
                const float2 b0 = int3_ldcg_f2(reinterpret_cast<const float2*>(&row_b[local_c0]));
                const float2 a1 = int3_ldcg_f2(reinterpret_cast<const float2*>(&row_a[local_c1]));
                const float2 b1 = int3_ldcg_f2(reinterpret_cast<const float2*>(&row_b[local_c1]));
                float s0_0, s1_0, s0_1, s1_1;
                sum3(d0_0, d1_0, a0, b0, s0_0, s1_0);
                sum3(d0_1, d1_1, a1, b1, s0_1, s1_1);
                emit_out(g, s0_0, s1_0, s0_1, s1_1);
              }
              if constexpr (kM16) {
                if (has_hi) {
                  const float* row_a = pa_base + (g + 8) * 64;
                  const float* row_b = pb_base + (g + 8) * 64;
                  const float2 a0 = int3_ldcg_f2(reinterpret_cast<const float2*>(&row_a[local_c0]));
                  const float2 b0 = int3_ldcg_f2(reinterpret_cast<const float2*>(&row_b[local_c0]));
                  const float2 a1 = int3_ldcg_f2(reinterpret_cast<const float2*>(&row_a[local_c1]));
                  const float2 b1 = int3_ldcg_f2(reinterpret_cast<const float2*>(&row_b[local_c1]));
                  float s2_0, s3_0, s2_1, s3_1;
                  sum3(d2_0, d3_0, a0, b0, s2_0, s3_0);
                  sum3(d2_1, d3_1, a1, b1, s2_1, s3_1);
                  emit_out(g + 8, s2_0, s3_0, s2_1, s3_1);
                }
              }
            }
            __syncthreads();
            if (tid == 0) {
              atomicExch(&sk_done[cur_n], 0);
            }
          }
        } else {
          // 4+ part fallback: lock-free FP32 atomicAdd + last-arriving self-zeroing store.
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
              emit_out(g, s0_0, s1_0, s0_1, s1_1);
            }
            if constexpr (kM16) {
              if (has_hi) {
                float* row_p = tile_p + (g + 8) * 64;
                const float s2_0 = atomicExch(&row_p[local_c0 + 0], 0.0f);
                const float s3_0 = atomicExch(&row_p[local_c0 + 1], 0.0f);
                const float s2_1 = atomicExch(&row_p[local_c1 + 0], 0.0f);
                const float s3_1 = atomicExch(&row_p[local_c1 + 1], 0.0f);
                emit_out(g + 8, s2_0, s3_0, s2_1, s3_1);
              }
            }
            __syncthreads();
            if (tid == 0) {
              atomicExch(&sk_done[cur_n], 0);
            }
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

int int3_pdl_mode() {
  static const int mode = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_PDL");
    if (!s || !*s) return 3;
    return (std::atoi(s) != 0) ? 3 : 0;
  }();
  return mode;
}

bool int3_nf3_enabled() {
  static const bool on = []() -> bool {
    const char* s = std::getenv("MUSE_INT3_NF3");
    if (!s || !*s) return true;
    return std::atoi(s) != 0;
  }();
  return on;
}

template <typename... KArgs, typename... Args>
static inline void int3_launch_pdl(
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
  TORCH_CHECK(err == cudaSuccess, "int3_mma kernel launch failed: ", cudaGetErrorString(err));
}

float* ensure_int3_splitk_partials(const torch::TensorOptions& opts, int64_t need) {
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

bool int3_two_part_red_enabled() {
  static const bool on = []() -> bool {
    const char* s = std::getenv("MUSE_INT3_TWO_PART_RED");
    if (!s || !*s) return true;
    return std::atoi(s) != 0;
  }();
  return on;
}

void ensure_int3_sk_scratch(const torch::TensorOptions& opts, int n_blocks,
                            float*& partials_ptr, float*& multi_part_ptr, int*& done_ptr) {
  static torch::Tensor g_sk_partials;
  static torch::Tensor g_sk_multi_part;
  static torch::Tensor g_sk_done;
  static int g_sk_cap = 0;
  static int g_sk_mp_cap = 0;
  static uint32_t g_sk_slot = 0;
  if (!g_sk_partials.defined() || g_sk_partials.device() != opts.device() || g_sk_cap < n_blocks) {
    const int cap = (n_blocks > 1024) ? n_blocks : 1024;
    g_sk_partials = torch::zeros({2, static_cast<int64_t>(cap) * 16 * 64}, opts.dtype(torch::kFloat32));
    g_sk_done = torch::zeros({2, cap}, opts.dtype(torch::kInt32));
    g_sk_cap = cap;
  }
  const bool use_mp = int3_two_part_red_enabled();
  if (use_mp && (!g_sk_multi_part.defined() || g_sk_multi_part.device() != opts.device() || g_sk_mp_cap < n_blocks)) {
    const int mp_cap = (n_blocks > 256) ? n_blocks : 256;
    g_sk_multi_part = torch::empty({2, static_cast<int64_t>(mp_cap) * 3 * 16 * 64}, opts.dtype(torch::kFloat32));
    g_sk_mp_cap = mp_cap;
  }
  const int slot = static_cast<int>(g_sk_slot & 1u);
  g_sk_slot ^= 1u;
  partials_ptr = g_sk_partials.data_ptr<float>() + static_cast<size_t>(slot) * static_cast<size_t>(g_sk_cap) * 16 * 64;
  multi_part_ptr = use_mp
      ? (g_sk_multi_part.data_ptr<float>() + static_cast<size_t>(slot) * static_cast<size_t>(g_sk_mp_cap) * 3 * 16 * 64)
      : nullptr;
  done_ptr = g_sk_done.data_ptr<int>() + static_cast<size_t>(slot) * static_cast<size_t>(g_sk_cap);
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

template <bool kNf3>
static torch::Tensor int3_gemm_impl(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K) {
  const int M = static_cast<int>(A.size(0));
  auto C = torch::empty({M, N}, A.options());
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n_blocks = static_cast<int>(N / kRowsPerCTA);
  const bool use_m16 = (M > 8);
  const int pdl_mode = int3_pdl_mode();
  const int sk_grid = int3_sk_grid();
  if (sk_grid > 0) {
    const int total_iters = n_blocks * static_cast<int>(K >> 6);
    const int grid = (sk_grid < total_iters) ? sk_grid : total_iters;
    float* p_ptr = nullptr;
    float* mp_ptr = nullptr;
    int* d_ptr = nullptr;
    ensure_int3_sk_scratch(A.options(), n_blocks, p_ptr, mp_ptr, d_ptr);
    if (use_m16) {
      int3_launch_pdl(
          int3_mma_streamk_kernel<true, false, kNf3>, dim3(grid, 1, 1), kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, mp_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode);
    } else {
      int3_launch_pdl(
          int3_mma_streamk_kernel<false, false, kNf3>, dim3(grid, 1, 1), kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, mp_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode);
    }
    return C;
  }
  int split_k = 1;
  if (n_blocks <= 128 && K >= 16384) {
    split_k = (K >= 28672) ? 4 : 3;
  }
  if (split_k == 1) {
    const dim3 grid(n_blocks, 1, 1);
    if (use_m16) {
      int3_launch_pdl(
          int3_mma_gemm_kernel<false, true, kNf3>, grid, kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode);
    } else {
      int3_launch_pdl(
          int3_mma_gemm_kernel<false, false, kNf3>, grid, kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode);
    }
  } else {
    const int64_t need = static_cast<int64_t>(split_k) * M * N;
    float* partials_ptr = ensure_int3_splitk_partials(A.options(), need);
    const dim3 grid(n_blocks, split_k, 1);
    if (use_m16) {
      int3_launch_pdl(
          int3_mma_gemm_kernel<true, true, kNf3>, grid, kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          static_cast<__nv_bfloat16*>(nullptr), partials_ptr,
          M, static_cast<int>(N), static_cast<int>(K), split_k, pdl_mode);
    } else {
      int3_launch_pdl(
          int3_mma_gemm_kernel<true, false, kNf3>, grid, kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          static_cast<__nv_bfloat16*>(nullptr), partials_ptr,
          M, static_cast<int>(N), static_cast<int>(K), split_k, pdl_mode);
    }
    const int vtotal = (M * static_cast<int>(N)) >> 2;
    const dim3 rgrid((vtotal + 255) / 256, 1, 1);
    int3_launch_pdl(
        int3_reduce_splitk_kernel, rgrid, 256, st, pdl_mode,
        static_cast<const float*>(partials_ptr),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, static_cast<int>(N), split_k, pdl_mode);
  }
  return C;
}

torch::Tensor int3_gemm(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous(),
              "int3_gemm expects contiguous CUDA BF16 [M, K]");
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && A.size(1) == K && N % kRowsPerCTA == 0 && K % 64 == 0,
              "int3_gemm shape mismatch");
  return int3_nf3_enabled() ? int3_gemm_impl<true>(A, W_packed, N, K)
                            : int3_gemm_impl<false>(A, W_packed, N, K);
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

template <bool kNf3>
static torch::Tensor int3_gemm_swiglu_impl(torch::Tensor A, torch::Tensor W_packed, int64_t I, int64_t K) {
  const int M = static_cast<int>(A.size(0));
  auto C = torch::empty({M, I}, A.options());
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n_blocks = static_cast<int>(I / 32);
  const int pdl_mode = int3_pdl_mode();
  const int sk_grid = int3_sk_grid();
  if (sk_grid > 0) {
    const int total_iters = n_blocks * static_cast<int>(K >> 6);
    const int grid = (sk_grid < total_iters) ? sk_grid : total_iters;
    float* p_ptr = nullptr;
    float* mp_ptr = nullptr;
    int* d_ptr = nullptr;
    ensure_int3_sk_scratch(A.options(), n_blocks, p_ptr, mp_ptr, d_ptr);
    if (M > 8) {
      int3_launch_pdl(
          int3_mma_streamk_kernel<true, true, kNf3>, dim3(grid, 1, 1), kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, mp_ptr, d_ptr, M, static_cast<int>(I), static_cast<int>(K), n_blocks, grid, pdl_mode);
    } else {
      int3_launch_pdl(
          int3_mma_streamk_kernel<false, true, kNf3>, dim3(grid, 1, 1), kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, mp_ptr, d_ptr, M, static_cast<int>(I), static_cast<int>(K), n_blocks, grid, pdl_mode);
    }
    return C;
  }
  const dim3 grid(n_blocks, 1, 1);
  if (M > 8) {
    int3_launch_pdl(
        int3_mma_gemm_swiglu_kernel<true, kNf3>, grid, kThreads, st, pdl_mode,
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, static_cast<int>(I), static_cast<int>(K), pdl_mode);
  } else {
    int3_launch_pdl(
        int3_mma_gemm_swiglu_kernel<false, kNf3>, grid, kThreads, st, pdl_mode,
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, static_cast<int>(I), static_cast<int>(K), pdl_mode);
  }
  return C;
}

torch::Tensor int3_gemm_swiglu(torch::Tensor A, torch::Tensor W_packed, int64_t I, int64_t K) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous(),
              "int3_gemm_swiglu expects contiguous CUDA BF16 [M, K]");
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && A.size(1) == K && I % 32 == 0 && K % 64 == 0,
              "int3_gemm_swiglu shape mismatch");
  return int3_nf3_enabled() ? int3_gemm_swiglu_impl<true>(A, W_packed, I, K)
                            : int3_gemm_swiglu_impl<false>(A, W_packed, I, K);
}
