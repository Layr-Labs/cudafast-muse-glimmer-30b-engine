// SPDX-License-Identifier: Apache-2.0
// muse_int4_mma: 4-bit asymmetric group-64 W4A16 (uniform INT4 BF16x2 HADD2+HFMA2 & optional NF4) Tensor-Core GEMM for DFlash.
// Format (4.50 bits/weight, 0 wasted bits, 16-byte aligned):
//   Each 8x64 tile (8 output rows x 64 K cols) is 288 bytes (18 x uint4):
//     [0..127]   uint32_t q_lo[32] : for lane l=(r<<2)|t, 8 x 4-bit unsigned codes (s=0..1, 4 weights/k16)
//     [128..255] uint32_t q_hi[32] : for lane l=(r<<2)|t, 8 x 4-bit unsigned codes (s=2..3, 4 weights/k16)
//     [256..287] __nv_bfloat162 sz[8] : (scale, zero) for rows r=0..7
//   Runs on SM121 with 5-stage cp.async.cg.L2::256B (no L2::cache_hint / createpolicy -> SM121 odd-UR safe),
//   4 CTAs/SM, XOR-swizzled ldmatrix, Marlin-interleaved BF16x2 HADD2+HFMA2 dequant (kNf4=false), dual SM121 PDL,
//   2x N-tile A reuse, 2 MiB TLB prefetch, zero-cat aux5 fc input, and whole-tile + deterministic non-atomic
//   2-part / 3-part float2 Stream-K reduction (citing e1f584e4-93df-4713-92f7-5f76154f82a0 &
//   51990a52-65ba-4d68-aa98-033064d52db5).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cstdlib>
#include <stdint.h>

void vq4d_clear_all_prerot();
void vq4d_reset_norm_rot_table();

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

template <bool kNf4>
__device__ __forceinline__ float int4_deq_f32(uint32_t q) {
  if constexpr (kNf4) {
    // Exact 16-level Lloyd-Max Normal(0,1) quantiles:
    // {0, 31, 52, 69, 84, 97, 109, 122, 133, 146, 158, 171, 186, 203, 224, 255}.
    // Antisymmetry: level[15 - k] == 255 - level[k] == level[k] ^ 0xFF.
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

// 16B L2::256B weight cp.async (no L2::cache_hint / createpolicy -> 100% SM121 odd-UR safe).
__device__ __forceinline__ void int4_cp_async16_w(uint32_t sd, const void* gs) {
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ uint64_t int4_pol_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

__device__ __forceinline__ void int4_mbar_init(uint64_t* mbar, int count) {
  for (int s = 0; s < count; ++s) {
    const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[s]));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b) : "memory");
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void int4_tma_bulk_w(
    void* smem_dst, const void* gmem_src, int bytes, uint64_t* mbar, uint64_t pol) {
  const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
      ::"r"(d), "l"(gmem_src), "r"(bytes), "r"(b), "l"(pol)
      : "memory");
}

__device__ __forceinline__ void int4_mbar_wait(uint64_t* mbar, uint32_t parity) {
  const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "{\n\t.reg .pred P1;\n\tINT4_MBAR_WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@!P1 bra INT4_MBAR_WAIT_%=;\n\t}"
      ::"r"(b), "r"(parity)
      : "memory");
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

__device__ __forceinline__ float2 int4_ldcg_f2(const float2* ptr) {
  float2 v;
  asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(ptr));
  return v;
}

__device__ __forceinline__ float4 int4_ldcg_f4(const float4* ptr) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(ptr));
  return v;
}

__host__ __device__ __forceinline__ size_t int4_n32_sk_step_slot(
    int cta_n32, int kg, int num_n32, int num_kg, int grid_ctas) {
  const int64_t total = static_cast<int64_t>(num_n32) * num_kg;
  const int64_t lin = static_cast<int64_t>(cta_n32) * num_kg + kg;
  const int c = static_cast<int>(((lin + 1) * grid_ctas - 1) / total);
  const int64_t iter_s = (static_cast<int64_t>(c) * total) / grid_ctas;
  const int64_t loc = lin - iter_s;
  const int64_t q = total / grid_ctas;
  if (loc < q) {
    return static_cast<size_t>(loc * grid_ctas + c);
  }
  return static_cast<size_t>(q * grid_ctas + (iter_s - static_cast<int64_t>(c) * q));
}

// Native BF16x2 4-bit dequant using Marlin-interleaved nibbles ((q0,q1) in [3:0]&[19:16], etc.):
// 0x41804180 | ((q << sh) & 0x00780078) == bf16x2(16+q0, 16+q1) exact -> HADD2(-16) -> HFMA2(sc, zr).
// Bit-exact with rn_bf16(fma(float(q), sc, zr)), 0 I2F, 0 F2FP.
template <bool kHiHalf>
__device__ __forceinline__ void int4_deq4_k16_bf16x2(
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

template <bool kNf4, bool kHiHalf>
__device__ __forceinline__ void int4_deq4_k16(
    uint32_t q32, __nv_bfloat162 sc2, __nv_bfloat162 zr2, float sc_f, float zr_f,
    uint32_t& b0, uint32_t& b1) {
  if constexpr (!kNf4) {
    int4_deq4_k16_bf16x2<kHiHalf>(q32, sc2, zr2, b0, b1);
  } else {
    const uint32_t w = kHiHalf ? (q32 >> 16) : q32;
    b0 = int4_bf16x2_u32(
        __fmaf_rn(int4_deq_f32<true>(w & 15u), sc_f, zr_f),
        __fmaf_rn(int4_deq_f32<true>((w >> 4) & 15u), sc_f, zr_f));
    b1 = int4_bf16x2_u32(
        __fmaf_rn(int4_deq_f32<true>((w >> 8) & 15u), sc_f, zr_f),
        __fmaf_rn(int4_deq_f32<true>((w >> 12) & 15u), sc_f, zr_f));
  }
}

__device__ __forceinline__ float int4_subwarp4_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}

__device__ __forceinline__ float int4_swiglu_f32(float gate, float up) {
  return (gate / (1.0f + __expf(-gate))) * up;
}

template <bool kMseOpt, bool kNf4 = false>
__device__ __forceinline__ void int4_quant_pack_row(
    const __nv_bfloat16* __restrict__ wrow, int r, int t,
    uint32_t& q_lo, uint32_t& q_hi, __nv_bfloat162& sz_bf) {
  float vals[16];
  float rmin = 1e30f;
  float rmax = -1e30f;
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
  sz_bf = __floats2bfloat162_rn(sc_mm, rmin);

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

  q_lo = 0u;
  q_hi = 0u;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = quant_val(vals[i], zr_q, inv_sc);
    if constexpr (!kNf4) {
      q_lo |= (static_cast<uint32_t>(q) << (((i >> 1) << 2) | ((i & 1) << 4)));
    } else {
      q_lo |= (static_cast<uint32_t>(q) << (4 * i));
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = quant_val(vals[8 + i], zr_q, inv_sc);
    if constexpr (!kNf4) {
      q_hi |= (static_cast<uint32_t>(q) << (((i >> 1) << 2) | ((i & 1) << 4)));
    } else {
      q_hi |= (static_cast<uint32_t>(q) << (4 * i));
    }
  }
}

template <bool kMseOpt, bool kNf4 = false>
__global__ void int4_pack_kernel(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ out,
    int N, int K, int k_major) {
  const int n_tile = blockIdx.x;
  const int kg = blockIdx.y;
  const int lane = threadIdx.x;
  const int r = lane >> 2;
  const int t = lane & 3;
  const int row = n_tile * 8 + r;
  const int k_base = kg * 64;

  uint32_t q_lo, q_hi;
  __nv_bfloat162 sz_bf;
  int4_quant_pack_row<kMseOpt, kNf4>(W + static_cast<size_t>(row) * K + k_base, r, t, q_lo, q_hi, sz_bf);

  const int cta_n = n_tile >> 3;
  const int sub_tile = n_tile & 7;
  const int num_kg = K >> 6;
  const int n_blocks = N / kRowsPerCTA;
  size_t dst_off = 0;
  if (k_major < 0) {
    const int sk_grid = -k_major;
    const int num_n32 = N >> 5;
    const int cta_n32 = n_tile >> 2;
    const int sub_n32 = n_tile & 3;
    const size_t step_slot = int4_n32_sk_step_slot(cta_n32, kg, num_n32, num_kg, sk_grid);
    dst_off = step_slot * (4 * kTileBytes) + static_cast<size_t>(sub_n32) * kTileBytes;
  } else if (k_major == 2) {
    const int num_n32 = N >> 5;
    const int cta_n32 = n_tile >> 2;
    const int sub_n32 = n_tile & 3;
    const int kg2 = kg >> 1;
    const int kh = kg & 1;
    dst_off = (static_cast<size_t>(kg2) * num_n32 + cta_n32) * (2 * 4 * kTileBytes) +
              static_cast<size_t>(kh) * (4 * kTileBytes) +
              static_cast<size_t>(sub_n32) * kTileBytes;
  } else {
    const size_t tile_idx = k_major
        ? ((static_cast<size_t>(kg) * n_blocks + cta_n) * kNTilesPerCTA + sub_tile)
        : ((static_cast<size_t>(cta_n) * num_kg + kg) * kNTilesPerCTA + sub_tile);
    dst_off = tile_idx * kTileBytes;
  }
  uint8_t* dst = out + dst_off;

  reinterpret_cast<uint32_t*>(dst)[lane] = q_lo;
  reinterpret_cast<uint32_t*>(dst + 128)[lane] = q_hi;
  if (t == 0) {
    reinterpret_cast<__nv_bfloat162*>(dst + 256)[r] = sz_bf;
  }
}

template <bool kMseOpt, bool kNf4 = false>
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

  uint32_t q_lo, q_hi;
  __nv_bfloat162 sz_bf;
  int4_quant_pack_row<kMseOpt, kNf4>(W + static_cast<size_t>(row) * K + k_base, r, t, q_lo, q_hi, sz_bf);

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

__device__ __forceinline__ void int4_aux5_select(
    const Aux5Desc& aux5, int seg, const __nv_bfloat16*& a_base, int64_t& a_stride) {
  if (seg < 2) {
    a_base = (seg == 0) ? aux5.p[0] : aux5.p[1];
    a_stride = (seg == 0) ? aux5.stride[0] : aux5.stride[1];
  } else if (seg == 2) {
    a_base = aux5.p[2];
    a_stride = aux5.stride[2];
  } else {
    a_base = (seg == 3) ? aux5.p[3] : aux5.p[4];
    a_stride = (seg == 3) ? aux5.stride[3] : aux5.stride[4];
  }
}

template <int kMaxM>
struct alignas(128) Int4StageSmem {
  alignas(128) uint8_t w[kStages][kStepWBytes];         // 5 * 2304 = 11520 B (18 * 128 B TMA-aligned)
  alignas(128) __nv_bfloat16 a[kStages][kMaxM][64];     // 5 * 1024 = 5120 B (M<=8) or 10240 B (M<=16), XOR-swizzled
  alignas(8) uint64_t mbar[kStages];                    // 40 B -> 21,888 B (M<=16) -> 4 CTAs/SM
};

template <bool kSplitK, bool kM16, bool kAux5, bool kNf4 = false>
__global__ void __launch_bounds__(128, 4) int4_mma_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    Aux5Desc aux5,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ C_fp32,
    int M, int N, int K, int split_k, int pdl_mode, int k_major, int use_tma) {
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

  if (use_tma) {
    if (tid == 0) {
      int4_mbar_init(smem.mbar, kStages);
    }
    __syncthreads();
  }
  const uint64_t w_pol = use_tma ? int4_pol_evict_first() : 0ull;

  const int n_blocks = N / kRowsPerCTA;
  const size_t cta_stride_bytes = static_cast<size_t>(num_kg) * kStepWBytes;
  const uint8_t* w_cta = k_major
      ? (W_packed + static_cast<size_t>(cta_n) * kStepWBytes)
      : (W_packed + static_cast<size_t>(cta_n) * cta_stride_bytes);
  const size_t w_kg_stride = k_major
      ? (static_cast<size_t>(n_blocks) * kStepWBytes)
      : static_cast<size_t>(kStepWBytes);

  if (cta_n == 0 && sk_idx == 0) {
    const size_t total_bytes = static_cast<size_t>(n_blocks) * cta_stride_bytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_phys_col8 = a_col8 ^ (a_row & 7);
  const bool a_valid = (a_row < M);
  const int a_src_sz = a_valid ? 16 : 0;
  const int a_safe_row = a_valid ? a_row : 0;

  auto issue_w_tma_only = [&](int kg, int st, bool need_fence) {
    if (tid == 0) {
      if (need_fence) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      }
      int4_tma_bulk_w(&smem.w[st][0], w_cta + static_cast<size_t>(kg) * w_kg_stride, kStepWBytes, &smem.mbar[st], w_pol);
    }
  };

  auto issue_a_only = [&](int kg, int st) {
    const __nv_bfloat16* a_base = nullptr;
    int64_t a_stride = K;
    int a_col_base = kg * 64;
    if constexpr (kAux5) {
      const int seg = kg / aux5.kg_sub;
      const int local_kg = kg - seg * aux5.kg_sub;
      int4_aux5_select(aux5, seg, a_base, a_stride);
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

  auto issue_w_only = [&](int kg, int st) {
    const uint8_t* w_src = w_cta + static_cast<size_t>(kg) * w_kg_stride;
    {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_w(sd, w_src + tid * 16);
    }
    if (tid < (kStepWU4 - kThreads)) {
      const int idx = kThreads + tid;
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][idx * 16]));
      int4_cp_async16_w(sd, w_src + idx * 16);
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
      int4_aux5_select(aux5, seg, a_base, a_stride);
      a_col_base = local_kg * 64;
    } else {
      a_base = A;
    }
    if constexpr (kM16) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_phys_col8 * 8]) =
          a_valid
              ? *reinterpret_cast<const uint4*>(
                    a_base + static_cast<size_t>(a_row) * a_stride + a_col_base + a_col8 * 8)
              : make_uint4(0u, 0u, 0u, 0u);
    } else if (tid < 64) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_phys_col8 * 8]) =
          a_valid
              ? *reinterpret_cast<const uint4*>(
                    a_base + static_cast<size_t>(a_row) * a_stride + a_col_base + a_col8 * 8)
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  auto issue_stage = [&](int kg, int st, bool need_fence) {
    if (use_tma) {
      issue_w_tma_only(kg, st, need_fence);
      issue_a_only(kg, st);
      return;
    }
    const uint8_t* w_src = w_cta + static_cast<size_t>(kg) * w_kg_stride;
    {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_w(sd, w_src + tid * 16);
    }
    if (tid < (kStepWU4 - kThreads)) {
      const int idx = kThreads + tid;
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][idx * 16]));
      int4_cp_async16_w(sd, w_src + idx * 16);
    }
    issue_a_only(kg, st);
  };

  constexpr int kPreload = kStages - 1;  // 4
  const int preload = (my_kgs < kPreload) ? my_kgs : kPreload;
  if ((pdl_mode & 1) != 0) {
    if (use_tma) {
      for (int i = 0; i < preload; ++i) {
        issue_w_tma_only(kg_start + i, i, false);
      }
      asm volatile("griddepcontrol.wait;" ::: "memory");
      for (int i = 0; i < preload; ++i) {
        issue_a_only(kg_start + i, i);
      }
    } else {
      for (int i = 0; i < preload; ++i) {
        issue_w_only(kg_start + i, i);
      }
      asm volatile("griddepcontrol.wait;" ::: "memory");
      for (int i = 0; i < preload; ++i) {
        load_a_sync(kg_start + i, i);
      }
    }
  } else {
    for (int i = 0; i < preload; ++i) {
      issue_stage(kg_start + i, i, false);
    }
  }

  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int ldm_row = kM16 ? (lane & 15) : (lane & 7);
  const int ldm_col_off = kM16 ? (lane >> 4) : ((lane >> 3) & 1);
  const int trig_i = ((pdl_mode & 2) != 0) ? ((my_kgs > 6) ? (my_kgs - 6) : 0) : -1;

  int st = 0;
  uint32_t mbar_phase = 0u;
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
    if (use_tma) {
      int4_mbar_wait(&smem.mbar[st], mbar_phase);
    }
    __syncthreads();

    const int next_i = i + kPreload;
    if (next_i < my_kgs) {
      const int next_st = (st == 0) ? (kStages - 1) : (st - 1);
      issue_stage(kg_start + next_i, next_st, true);
    }

    const uint8_t* wt0 = &smem.w[st][(warp * 2 + 0) * kTileBytes];
    const uint8_t* wt1 = &smem.w[st][(warp * 2 + 1) * kTileBytes];
    const uint32_t q0_lo = reinterpret_cast<const uint32_t*>(wt0)[lane];
    const uint32_t q0_hi = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
    const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 256)[g];
    const __nv_bfloat162 sc0_2 = __bfloat162bfloat162(sz0.x);
    const __nv_bfloat162 zr0_2 = __bfloat162bfloat162(sz0.y);
    const float sc0_f = kNf4 ? __bfloat162float(sz0.x) : 0.0f;
    const float zr0_f = kNf4 ? __bfloat162float(sz0.y) : 0.0f;

    const uint32_t q1_lo = reinterpret_cast<const uint32_t*>(wt1)[lane];
    const uint32_t q1_hi = reinterpret_cast<const uint32_t*>(wt1 + 128)[lane];
    const __nv_bfloat162 sz1 = reinterpret_cast<const __nv_bfloat162*>(wt1 + 256)[g];
    const __nv_bfloat162 sc1_2 = __bfloat162bfloat162(sz1.x);
    const __nv_bfloat162 zr1_2 = __bfloat162bfloat162(sz1.y);
    const float sc1_f = kNf4 ? __bfloat162float(sz1.x) : 0.0f;
    const float zr1_f = kNf4 ? __bfloat162float(sz1.y) : 0.0f;

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][ldm_row][pcol8 * 8]));
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        int4_ldmatrix_x4(a0, a1, a2, a3, addr);
      } else {
        int4_ldmatrix_x2(a0, a2, addr);
      }
      uint32_t b0, b1, c0, c1;
      if ((s & 1) == 0) {
        int4_deq4_k16<kNf4, false>((s < 2) ? q0_lo : q0_hi, sc0_2, zr0_2, sc0_f, zr0_f, b0, b1);
        int4_deq4_k16<kNf4, false>((s < 2) ? q1_lo : q1_hi, sc1_2, zr1_2, sc1_f, zr1_f, c0, c1);
      } else {
        int4_deq4_k16<kNf4, true>((s < 2) ? q0_lo : q0_hi, sc0_2, zr0_2, sc0_f, zr0_f, b0, b1);
        int4_deq4_k16<kNf4, true>((s < 2) ? q1_lo : q1_hi, sc1_2, zr1_2, sc1_f, zr1_f, c0, c1);
      }
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0, a1, a2, a3, b0, b1);
        int4_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0, a1, a2, a3, c0, c1);
      } else {
        int4_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b0, b1);
        int4_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, c0, c1);
      }
    }

    if (++st == kStages) {
      st = 0;
      mbar_phase ^= 1u;
    }
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
      *reinterpret_cast<float2*>(p0) = make_float2(d0_0, d1_0);
      *reinterpret_cast<float2*>(p1) = make_float2(d0_1, d1_1);
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
        *reinterpret_cast<float2*>(p0) = make_float2(d2_0, d3_0);
        *reinterpret_cast<float2*>(p1) = make_float2(d2_1, d3_1);
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
};                                         // Total = 8.5 KiB (M<=8) or 12.5 KiB (M<=16) -> 6 CTAs/SM (288 CTAs)

template <bool kSplitK, bool kM16, bool kAux5, bool kNf4 = false>
__global__ void __launch_bounds__(128, 6) int4_mma_gemm_n32_kernel(
    const __nv_bfloat16* __restrict__ A,
    Aux5Desc aux5,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ C_fp32,
    int M, int N, int K, int split_k, int pdl_mode) {
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
  const bool a_valid = (a_row < M);
  const int a_src_sz = a_valid ? 16 : 0;
  const int a_safe_row = a_valid ? a_row : 0;

  auto issue_w_only = [&](int kg, int st) {
    if (tid < kN32StepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_w(sd, w_cta + static_cast<size_t>(kg) * kStepWBytes + tid * 16);
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
      int4_aux5_select(aux5, seg, a_base, a_stride);
      a_col_base = local_kg * 64;
    } else {
      a_base = A;
    }
    if constexpr (kM16) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_phys_col8 * 8]) =
          a_valid
              ? *reinterpret_cast<const uint4*>(
                    a_base + static_cast<size_t>(a_row) * a_stride + a_col_base + a_col8 * 8)
              : make_uint4(0u, 0u, 0u, 0u);
    } else if (tid < 64) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_phys_col8 * 8]) =
          a_valid
              ? *reinterpret_cast<const uint4*>(
                    a_base + static_cast<size_t>(a_row) * a_stride + a_col_base + a_col8 * 8)
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

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
      int4_aux5_select(aux5, seg, a_base, a_stride);
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

  float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int ldm_row = kM16 ? (lane & 15) : (lane & 7);
  const int ldm_col_off = kM16 ? (lane >> 4) : ((lane >> 3) & 1);
  const int trig_i = ((pdl_mode & 2) != 0) ? ((my_kgs > 5) ? (my_kgs - 5) : 0) : -1;

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

    const int next_i = i + kPreload;
    if (next_i < my_kgs) {
      const int next_st = (st == 0) ? (kN32Stages - 1) : (st - 1);
      issue_stage(kg_start + next_i, next_st);
    }

    const uint8_t* wt0 = &smem.w[st][warp * kTileBytes];
    const uint32_t q0_lo = reinterpret_cast<const uint32_t*>(wt0)[lane];
    const uint32_t q0_hi = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
    const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 256)[g];
    const __nv_bfloat162 sc0_2 = __bfloat162bfloat162(sz0.x);
    const __nv_bfloat162 zr0_2 = __bfloat162bfloat162(sz0.y);
    const float sc0_f = kNf4 ? __bfloat162float(sz0.x) : 0.0f;
    const float zr0_f = kNf4 ? __bfloat162float(sz0.y) : 0.0f;

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][ldm_row][pcol8 * 8]));
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        int4_ldmatrix_x4(a0, a1, a2, a3, addr);
      } else {
        int4_ldmatrix_x2(a0, a2, addr);
      }
      uint32_t b0, b1;
      if ((s & 1) == 0) {
        int4_deq4_k16<kNf4, false>((s < 2) ? q0_lo : q0_hi, sc0_2, zr0_2, sc0_f, zr0_f, b0, b1);
      } else {
        int4_deq4_k16<kNf4, true>((s < 2) ? q0_lo : q0_hi, sc0_2, zr0_2, sc0_f, zr0_f, b0, b1);
      }
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
      } else {
        int4_mma_m16n8k16(d0, d1, d2, d3, a0, a2, b0, b1);
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
      *reinterpret_cast<float2*>(p0) = make_float2(d0, d1);
    }
  }
  if constexpr (kM16) {
    if (has_hi) {
      const int ghi = g + 8;
      if constexpr (!kSplitK) {
        *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(ghi) * N + col0) = int4_bf16x2_u32(d2, d3);
      } else {
        float* p0 = C_fp32 + (static_cast<size_t>(sk_idx) * M + ghi) * N + col0;
        *reinterpret_cast<float2*>(p0) = make_float2(d2, d3);
      }
    }
  }
}

constexpr int kN32PairTmaStages = 3;
constexpr int kN32PairStepWBytes = 2 * kN32StepWBytes;  // 2304 = 18 * 128 B

template <int kMaxM>
struct alignas(128) Int4N32PairTmaSmem {
  alignas(128) uint8_t w[kN32PairTmaStages][kN32PairStepWBytes];   // 3 * 2304 = 6912 B
  alignas(128) __nv_bfloat16 a[kN32PairTmaStages][2][kMaxM][64];   // 3 * 2048 = 6144 B (M<=8) or 12288 B (M<=16)
  alignas(8) uint64_t mbar[kN32PairTmaStages];                     // 24 B -> 13,184 B (7 CTAs/SM = 336!)
};

// Single-wave (<=336 CTAs at 7 CTAs/SM) BK=128 paired 2304B Hardware-TMA N32 kernel:
// 100.000% 0-ULP bit-identical to int4_mma_gemm_kernel, halving barriers and crushing o_proj/qkv/fkv tail waves!
template <bool kM16, bool kNf4>
__global__ void __launch_bounds__(128, kM16 ? 5 : 7) int4_mma_gemm_n32_pair_tma_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K, int pdl_mode) {
  __shared__ Int4N32PairTmaSmem<kM16 ? 16 : 8> smem;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_pairs = K >> 7;
  const int num_n32 = N >> 5;
  const int cta_n32 = blockIdx.x;

  if (tid == 0) {
    int4_mbar_init(smem.mbar, kN32PairTmaStages);
  }
  __syncthreads();

  const uint64_t w_pol = int4_pol_evict_first();
  const size_t w_pair_stride = static_cast<size_t>(num_n32) * kN32PairStepWBytes;
  const uint8_t* w_base = W_packed + static_cast<size_t>(cta_n32) * kN32PairStepWBytes;

  if (cta_n32 == 0) {
    const size_t total_bytes = static_cast<size_t>(num_pairs) * w_pair_stride;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  const int a_kh_m8 = tid >> 6;
  const int a_row = kM16 ? (tid >> 3) : ((tid >> 3) & 7);
  const int a_col8 = tid & 7;
  const int a_phys_col8 = a_col8 ^ (a_row & 7);
  const bool a_valid = (a_row < M);
  const int a_src_sz = a_valid ? 16 : 0;
  const int a_safe_row = a_valid ? a_row : 0;

  auto issue_w_tma = [&](int p, int st, bool need_fence) {
    if (tid == 0) {
      if (need_fence) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      }
      int4_tma_bulk_w(&smem.w[st][0], w_base + static_cast<size_t>(p) * w_pair_stride, kN32PairStepWBytes, &smem.mbar[st], w_pol);
    }
  };

  auto issue_a_only = [&](int p, int st) {
    const int k_base = p * 128;
    if constexpr (kM16) {
      const uint32_t sd0 = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][0][a_row][a_phys_col8 * 8]));
      const uint32_t sd1 = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][1][a_row][a_phys_col8 * 8]));
      int4_cp_async16_a_pred(sd0, A + static_cast<size_t>(a_safe_row) * K + k_base + a_col8 * 8, a_src_sz);
      int4_cp_async16_a_pred(sd1, A + static_cast<size_t>(a_safe_row) * K + k_base + 64 + a_col8 * 8, a_src_sz);
    } else {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_kh_m8][a_row][a_phys_col8 * 8]));
      int4_cp_async16_a_pred(sd, A + static_cast<size_t>(a_safe_row) * K + k_base + a_kh_m8 * 64 + a_col8 * 8, a_src_sz);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kN32PairTmaStages - 1;  // 2
  const int preload = (num_pairs < kPreload) ? num_pairs : kPreload;
  if ((pdl_mode & 1) != 0) {
    for (int i = 0; i < preload; ++i) {
      issue_w_tma(i, i, false);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = 0; i < preload; ++i) {
      issue_a_only(i, i);
    }
  } else {
    for (int i = 0; i < preload; ++i) {
      issue_w_tma(i, i, false);
      issue_a_only(i, i);
    }
  }

  float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int ldm_row = kM16 ? (lane & 15) : (lane & 7);
  const int ldm_col_off = kM16 ? (lane >> 4) : ((lane >> 3) & 1);
  const int trig_p = ((pdl_mode & 2) != 0) ? ((num_pairs > 3) ? (num_pairs - 3) : 0) : -1;

  int st = 0;
  uint32_t parity = 0u;
  for (int p = 0; p < num_pairs; ++p) {
    if (p == trig_p) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    int4_mbar_wait(&smem.mbar[st], parity);
    const int rem = num_pairs - 1 - p;
    if (rem >= 1) {
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();

    const int next_p = p + kPreload;
    if (next_p < num_pairs) {
      const int next_st = (st == 0) ? (kN32PairTmaStages - 1) : (st - 1);
      issue_w_tma(next_p, next_st, true);
      issue_a_only(next_p, next_st);
    }

#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {
      const uint8_t* wt0 = &smem.w[st][kh * kN32StepWBytes + warp * kTileBytes];
      const uint32_t q0_lo = reinterpret_cast<const uint32_t*>(wt0)[lane];
      const uint32_t q0_hi = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
      const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 256)[g];
      const __nv_bfloat162 sc0_2 = __bfloat162bfloat162(sz0.x);
      const __nv_bfloat162 zr0_2 = __bfloat162bfloat162(sz0.y);
      const float sc0_f = kNf4 ? __bfloat162float(sz0.x) : 0.0f;
      const float zr0_f = kNf4 ? __bfloat162float(sz0.y) : 0.0f;

#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
        const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][kh][ldm_row][pcol8 * 8]));
        uint32_t a0, a1, a2, a3;
        if constexpr (kM16) {
          int4_ldmatrix_x4(a0, a1, a2, a3, addr);
        } else {
          int4_ldmatrix_x2(a0, a2, addr);
        }
        uint32_t b0, b1;
        if ((s & 1) == 0) {
          int4_deq4_k16<kNf4, false>((s < 2) ? q0_lo : q0_hi, sc0_2, zr0_2, sc0_f, zr0_f, b0, b1);
        } else {
          int4_deq4_k16<kNf4, true>((s < 2) ? q0_lo : q0_hi, sc0_2, zr0_2, sc0_f, zr0_f, b0, b1);
        }
        if constexpr (kM16) {
          int4_mma_m16n8k16_full(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
        } else {
          int4_mma_m16n8k16(d0, d1, d2, d3, a0, a2, b0, b1);
        }
      }
    }

    if (++st == kN32PairTmaStages) {
      st = 0;
      parity ^= 1u;
    }
  }

  const int col0 = cta_n32 * 32 + warp * 8 + 2 * t;
  if (g < M) {
    *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) = int4_bf16x2_u32(d0, d1);
  }
  if constexpr (kM16) {
    if (has_hi) {
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * N + col0) = int4_bf16x2_u32(d2, d3);
    }
  }
}

constexpr int kN32TmaStages = 5;

template <int kMaxM>
struct alignas(128) Int4N32TmaSmem {
  alignas(128) uint8_t w[kN32TmaStages][kN32StepWBytes];   // 5 * 1152 = 5760 B (each stage 128B aligned!)
  alignas(128) __nv_bfloat16 a[kN32TmaStages][kMaxM][64];  // 5 * 1024 = 5120 B (M<=8) or 10240 B (M<=16)
  alignas(8) uint64_t mbar[kN32TmaStages];                 // 40 B -> 10.8 KiB (M<=8) or 15.8 KiB (M<=16) -> 5-6 CTAs/SM!
};

// 192/240-CTA (4/5 CTAs/SM) 5-stage 1152B Hardware-TMA + L2::evict_first + Dual-PDL N32 deterministic Stream-K kernel.
template <bool kM16, bool kNf4, bool kTma>
__global__ void __launch_bounds__(128, 5) int4_mma_n32_tma_det_sk_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ sk_partials,
    int* __restrict__ sk_done,
    int M, int N, int K, int grid_ctas, int pdl_mode) {
  __shared__ Int4N32TmaSmem<kM16 ? 16 : 8> smem;
  __shared__ int s_is_last;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;
  const int num_n32 = N >> 5;
  const int cta_id = blockIdx.x;
  const int total_iters = num_n32 * num_kg;
  const int q_iters = total_iters / grid_ctas;
  const int iter_start = static_cast<int>((static_cast<int64_t>(cta_id) * total_iters) / grid_ctas);
  const int iter_end = static_cast<int>((static_cast<int64_t>(cta_id + 1) * total_iters) / grid_ctas);
  const int my_iters = iter_end - iter_start;
  if (my_iters <= 0) return;

  if constexpr (kTma) {
    if (tid == 0) {
      int4_mbar_init(smem.mbar, kN32TmaStages);
    }
    __syncthreads();
  }

  const size_t total_bytes = static_cast<size_t>(total_iters) * kN32StepWBytes;
  if (cta_id == 0) {
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  auto step_w_ptr = [&](int idx) -> const uint8_t* {
    const size_t slot = (idx < q_iters)
        ? (static_cast<size_t>(idx) * grid_ctas + cta_id)
        : static_cast<size_t>(static_cast<int64_t>(q_iters) * grid_ctas +
                              (iter_start - static_cast<int64_t>(cta_id) * q_iters));
    return W_packed + slot * kN32StepWBytes;
  };

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_phys_col8 = a_col8 ^ (a_row & 7);
  const bool a_valid = (a_row < M);
  const int a_src_sz = a_valid ? 16 : 0;
  const int a_safe_row = a_valid ? a_row : 0;

  auto issue_w_only = [&](int idx, int st) {
    const uint8_t* wp = step_w_ptr(idx);
    if constexpr (kTma) {
      if (tid == 0) {
        const uint64_t pol = int4_pol_evict_first();
        int4_tma_bulk_w(smem.w[st], wp, kN32StepWBytes, &smem.mbar[st], pol);
      }
    } else {
      if (tid < kN32StepWU4) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
        int4_cp_async16_w(sd, wp + tid * 16);
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
  };

  auto load_a_sync = [&](int kg, int st) {
    if constexpr (kM16) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_phys_col8 * 8]) =
          a_valid
              ? *reinterpret_cast<const uint4*>(A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8)
              : make_uint4(0u, 0u, 0u, 0u);
    } else if (tid < 64) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_phys_col8 * 8]) =
          a_valid
              ? *reinterpret_cast<const uint4*>(A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8)
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  auto issue_stage = [&](int idx, int kg, int st, bool need_fence) {
    const uint8_t* wp = step_w_ptr(idx);
    if constexpr (kTma) {
      if (tid == 0) {
        if (need_fence) {
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        const uint64_t pol = int4_pol_evict_first();
        int4_tma_bulk_w(smem.w[st], wp, kN32StepWBytes, &smem.mbar[st], pol);
      }
    } else {
      if (tid < kN32StepWU4) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
        int4_cp_async16_w(sd, wp + tid * 16);
      }
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

  int cur_n = iter_start / num_kg;
  int cur_kg = iter_start - cur_n * num_kg;

  constexpr int kPreload = kN32TmaStages - 1;  // 4
  const int preload = (my_iters < kPreload) ? my_iters : kPreload;
  int next_kg = cur_kg;
  if ((pdl_mode & 1) != 0) {
    for (int p = 0; p < preload; ++p) {
      issue_w_only(p, p);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int p = 0; p < preload; ++p) {
      load_a_sync(next_kg, p);
      if (++next_kg == num_kg) next_kg = 0;
    }
  } else {
    for (int p = 0; p < preload; ++p) {
      issue_stage(p, next_kg, p, false);
      if (++next_kg == num_kg) next_kg = 0;
    }
  }

  float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int ldm_row = kM16 ? (lane & 15) : (lane & 7);
  const int ldm_col_off = kM16 ? (lane >> 4) : ((lane >> 3) & 1);
  const int trig_i = ((pdl_mode & 2) != 0) ? ((my_iters > 4) ? (my_iters - 4) : 0) : -1;

  int st = 0;
  uint32_t parity = 0u;
  for (int i = 0; i < my_iters; ++i) {
    if (i == trig_i) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    if constexpr (kTma) {
      int4_mbar_wait(&smem.mbar[st], parity);
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

    const int next_i = i + kPreload;
    if (next_i < my_iters) {
      const int next_st = (st == 0) ? (kN32TmaStages - 1) : (st - 1);
      issue_stage(next_i, next_kg, next_st, true);
      if (++next_kg == num_kg) next_kg = 0;
    }

    const uint8_t* wt0 = &smem.w[st][warp * kTileBytes];
    const uint32_t q0_lo = reinterpret_cast<const uint32_t*>(wt0)[lane];
    const uint32_t q0_hi = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
    const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 256)[g];
    const __nv_bfloat162 sc0_2 = __bfloat162bfloat162(sz0.x);
    const __nv_bfloat162 zr0_2 = __bfloat162bfloat162(sz0.y);
    const float sc0_f = kNf4 ? __bfloat162float(sz0.x) : 0.0f;
    const float zr0_f = kNf4 ? __bfloat162float(sz0.y) : 0.0f;

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][ldm_row][pcol8 * 8]));
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        int4_ldmatrix_x4(a0, a1, a2, a3, addr);
      } else {
        int4_ldmatrix_x2(a0, a2, addr);
      }
      uint32_t b0, b1;
      if ((s & 1) == 0) {
        int4_deq4_k16<kNf4, false>((s < 2) ? q0_lo : q0_hi, sc0_2, zr0_2, sc0_f, zr0_f, b0, b1);
      } else {
        int4_deq4_k16<kNf4, true>((s < 2) ? q0_lo : q0_hi, sc0_2, zr0_2, sc0_f, zr0_f, b0, b1);
      }
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
      } else {
        int4_mma_m16n8k16(d0, d1, d2, d3, a0, a2, b0, b1);
      }
    }

    if (++st == kN32TmaStages) {
      st = 0;
      parity ^= 1u;
    }

    const bool tile_done = (cur_kg + 1 == num_kg) || (i + 1 == my_iters);
    if (tile_done) {
      const int c_first = static_cast<int>(((static_cast<int64_t>(cur_n) * num_kg + 1) * grid_ctas - 1) / total_iters);
      const int c_last = static_cast<int>(((static_cast<int64_t>(cur_n + 1) * num_kg) * grid_ctas - 1) / total_iters);
      const int num_parts = c_last - c_first + 1;
      const int col0 = cur_n * 32 + warp * 8 + 2 * t;
      if (num_parts == 1) {
        if (g < M) {
          *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) = int4_bf16x2_u32(d0, d1);
        }
        if constexpr (kM16) {
          if (has_hi) {
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * N + col0) = int4_bf16x2_u32(d2, d3);
          }
        }
      } else {
        const int part_idx = cta_id - c_first;
        float4* p_tile = reinterpret_cast<float4*>(sk_partials) + static_cast<size_t>(cur_n) * 384;
        p_tile[part_idx * 128 + tid] = make_float4(d0, d1, d2, d3);
        __threadfence();
        __syncthreads();
        if (tid == 0) {
          s_is_last = (atomicAdd(&sk_done[cur_n], 1) + 1 == num_parts) ? 1 : 0;
        }
        __syncthreads();
        if (s_is_last) {
          if (tid == 0) {
            atomicExch(&sk_done[cur_n], 0);
          }
          float4 s = int4_ldcg_f4(&p_tile[tid]);
          const float4 p1 = int4_ldcg_f4(&p_tile[128 + tid]);
          s.x = __fadd_rn(s.x, p1.x);
          s.y = __fadd_rn(s.y, p1.y);
          s.z = __fadd_rn(s.z, p1.z);
          s.w = __fadd_rn(s.w, p1.w);
          if (num_parts == 3) {
            const float4 p2 = int4_ldcg_f4(&p_tile[256 + tid]);
            s.x = __fadd_rn(s.x, p2.x);
            s.y = __fadd_rn(s.y, p2.y);
            s.z = __fadd_rn(s.z, p2.z);
            s.w = __fadd_rn(s.w, p2.w);
          }
          if (g < M) {
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) = int4_bf16x2_u32(s.x, s.y);
          }
          if constexpr (kM16) {
            if (has_hi) {
              *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * N + col0) = int4_bf16x2_u32(s.z, s.w);
            }
          }
        }
      }
      d0 = d1 = d2 = d3 = 0.0f;
      ++cur_n;
      cur_kg = 0;
    } else {
      ++cur_kg;
    }
  }
}

static void ensure_int4_n32_sk_scratch(const torch::TensorOptions& opts, int num_n32,
                                       float*& partials_ptr, int*& done_ptr) {
  static torch::Tensor g_n32_partials;
  static torch::Tensor g_n32_done;
  static int g_n32_cap = 0;
  static uint32_t g_n32_slot = 0;
  if (!g_n32_partials.defined() || g_n32_partials.device() != opts.device() || g_n32_cap < num_n32) {
    const int cap = (num_n32 > 512) ? num_n32 : 512;
    g_n32_partials = torch::empty({2, static_cast<int64_t>(cap) * 384 * 4}, opts.dtype(torch::kFloat32));
    g_n32_done = torch::zeros({2, cap}, opts.dtype(torch::kInt32));
    g_n32_cap = cap;
  }
  const int slot = static_cast<int>(g_n32_slot & 1u);
  g_n32_slot ^= 1u;
  partials_ptr = g_n32_partials.data_ptr<float>() + static_cast<size_t>(slot) * static_cast<size_t>(g_n32_cap) * 1536;
  done_ptr = g_n32_done.data_ptr<int>() + static_cast<size_t>(slot) * static_cast<size_t>(g_n32_cap);
}

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

template <bool kM16, bool kNf4 = false>
__global__ void __launch_bounds__(128, 4) int4_mma_gemm_swiglu_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    int M, int I, int K, int pdl_mode) {
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
  const bool a_valid = (a_row < M);
  const int a_src_sz = a_valid ? 16 : 0;
  const int a_safe_row = a_valid ? a_row : 0;

  auto issue_w_only = [&](int kg, int st) {
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
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  auto load_a_sync = [&](int kg, int st) {
    if constexpr (kM16) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_phys_col8 * 8]) =
          a_valid
              ? *reinterpret_cast<const uint4*>(A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8)
              : make_uint4(0u, 0u, 0u, 0u);
    } else if (tid < 64) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_phys_col8 * 8]) =
          a_valid
              ? *reinterpret_cast<const uint4*>(A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8)
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

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
  const int ldm_row = kM16 ? (lane & 15) : (lane & 7);
  const int ldm_col_off = kM16 ? (lane >> 4) : ((lane >> 3) & 1);
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
    const uint32_t qg_lo = reinterpret_cast<const uint32_t*>(wtg)[lane];
    const uint32_t qg_hi = reinterpret_cast<const uint32_t*>(wtg + 128)[lane];
    const __nv_bfloat162 szg = reinterpret_cast<const __nv_bfloat162*>(wtg + 256)[g];
    const __nv_bfloat162 scg_2 = __bfloat162bfloat162(szg.x);
    const __nv_bfloat162 zrg_2 = __bfloat162bfloat162(szg.y);
    const float scg_f = kNf4 ? __bfloat162float(szg.x) : 0.0f;
    const float zrg_f = kNf4 ? __bfloat162float(szg.y) : 0.0f;

    const uint32_t qu_lo = reinterpret_cast<const uint32_t*>(wtu)[lane];
    const uint32_t qu_hi = reinterpret_cast<const uint32_t*>(wtu + 128)[lane];
    const __nv_bfloat162 szu = reinterpret_cast<const __nv_bfloat162*>(wtu + 256)[g];
    const __nv_bfloat162 scu_2 = __bfloat162bfloat162(szu.x);
    const __nv_bfloat162 zru_2 = __bfloat162bfloat162(szu.y);
    const float scu_f = kNf4 ? __bfloat162float(szu.x) : 0.0f;
    const float zru_f = kNf4 ? __bfloat162float(szu.y) : 0.0f;

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][ldm_row][pcol8 * 8]));
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        int4_ldmatrix_x4(a0, a1, a2, a3, addr);
      } else {
        int4_ldmatrix_x2(a0, a2, addr);
      }
      uint32_t b0, b1, c0, c1;
      if ((s & 1) == 0) {
        int4_deq4_k16<kNf4, false>((s < 2) ? qg_lo : qg_hi, scg_2, zrg_2, scg_f, zrg_f, b0, b1);
        int4_deq4_k16<kNf4, false>((s < 2) ? qu_lo : qu_hi, scu_2, zru_2, scu_f, zru_f, c0, c1);
      } else {
        int4_deq4_k16<kNf4, true>((s < 2) ? qg_lo : qg_hi, scg_2, zrg_2, scg_f, zrg_f, b0, b1);
        int4_deq4_k16<kNf4, true>((s < 2) ? qu_lo : qu_hi, scu_2, zru_2, scu_f, zru_f, c0, c1);
      }
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0_g, d1_g, d2_g, d3_g, a0, a1, a2, a3, b0, b1);
        int4_mma_m16n8k16_full(d0_u, d1_u, d2_u, d3_u, a0, a1, a2, a3, c0, c1);
      } else {
        int4_mma_m16n8k16(d0_g, d1_g, d2_g, d3_g, a0, a2, b0, b1);
        int4_mma_m16n8k16(d0_u, d1_u, d2_u, d3_u, a0, a2, c0, c1);
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

// Persistent 5-stage (4 CTAs/SM) XOR-swizzled ldmatrix + BF16x2 HADD2+HFMA2 Stream-K W4A16 GEMM.
// Supports:
//   - kWholeSplit=true  (e1f584e4): whole tiles bypass scratch; remaining tiles are bisected cleanly
//     into 2 halves with deterministic non-atomic float2 reduction (zero float atomicAdd).
//   - kWholeSplit=false (51990a52 + 3-part extension): balanced Stream-K across all CTAs with
//     deterministic non-atomic float2 reduction on ALL 2-part AND 3-part tiles (100% of tiles on
//     fc, fkv, qkv_proj, o_proj at grid=144/192), falling back to sk_partials only for >=4-part tiles.
template <bool kM16, bool kAux5, bool kWholeSplit, bool kNf4 = false>
__global__ void __launch_bounds__(128, 4) int4_mma_streamk_kernel(
    const __nv_bfloat16* __restrict__ A,
    Aux5Desc aux5,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ sk_partials,    // [max_n_blocks, 16, 64] zero-initialized & self-zeroing (>=4-part fallback)
    float* __restrict__ sk_multi_part,  // [max_n_blocks, 3, 16, 64] uninitialized non-atomic 2/3-part slots (nullptr disables)
    int* __restrict__ sk_done,          // [max_n_blocks] zero-initialized & self-zeroing
    int M, int N, int K, int n_blocks, int grid_ctas, int pdl_mode, int allow_3part, int use_tma) {
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
  int iter_start, iter_end;
  if constexpr (kWholeSplit) {
    const int whole_tiles = 2 * n_blocks - grid_ctas;
    if (cta_id < whole_tiles) {
      iter_start = cta_id * num_kg;
      iter_end = iter_start + num_kg;
    } else {
      const int split_cta = cta_id - whole_tiles;
      const int tile_start = (whole_tiles + (split_cta >> 1)) * num_kg;
      const int mid = tile_start + (num_kg >> 1);
      iter_start = (split_cta & 1) ? mid : tile_start;
      iter_end = (split_cta & 1) ? (tile_start + num_kg) : mid;
    }
  } else {
    iter_start = static_cast<int>((static_cast<int64_t>(cta_id) * total_iters) / grid_ctas);
    iter_end = static_cast<int>((static_cast<int64_t>(cta_id + 1) * total_iters) / grid_ctas);
  }
  const int my_iters = iter_end - iter_start;
  if (my_iters <= 0) return;

  if (use_tma) {
    if (tid == 0) {
      int4_mbar_init(smem.mbar, kStages);
    }
    __syncthreads();
  }
  const uint64_t w_pol = use_tma ? int4_pol_evict_first() : 0ull;

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

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_phys_col8 = a_col8 ^ (a_row & 7);
  const bool a_valid = (a_row < M);
  const int a_src_sz = a_valid ? 16 : 0;
  const int a_safe_row = a_valid ? a_row : 0;

  auto issue_w_tma_only = [&](int st, int w_iter, bool need_fence) {
    if (tid == 0) {
      if (need_fence) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      }
      int4_tma_bulk_w(&smem.w[st][0], W_packed + static_cast<size_t>(w_iter) * kStepWBytes, kStepWBytes, &smem.mbar[st], w_pol);
    }
  };

  auto issue_w_only = [&](int st, int w_iter) {
    const uint8_t* w_src = W_packed + static_cast<size_t>(w_iter) * kStepWBytes;
    {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_w(sd, w_src + tid * 16);
    }
    if (tid < (kStepWU4 - kThreads)) {
      const int idx = kThreads + tid;
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][idx * 16]));
      int4_cp_async16_w(sd, w_src + idx * 16);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  auto load_a_sync = [&](int st, const __nv_bfloat16* a_base, int64_t a_stride, int a_col_base) {
    if constexpr (kM16) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_phys_col8 * 8]) =
          a_valid
              ? *reinterpret_cast<const uint4*>(
                    a_base + static_cast<size_t>(a_row) * a_stride + a_col_base + a_col8 * 8)
              : make_uint4(0u, 0u, 0u, 0u);
    } else if (tid < 64) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_phys_col8 * 8]) =
          a_valid
              ? *reinterpret_cast<const uint4*>(
                    a_base + static_cast<size_t>(a_row) * a_stride + a_col_base + a_col8 * 8)
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  auto issue_a_next = [&](int st) {
    const __nv_bfloat16* a_base = A;
    int64_t a_stride = static_cast<int64_t>(K);
    if constexpr (kAux5) {
      int4_aux5_select(aux5, pre_seg, a_base, a_stride);
    }
    const int a_col_base = kAux5 ? (pre_local_kg * 64) : (pre_kg * 64);

    if constexpr (kM16) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int4_cp_async16_a_pred(sd, a_base + static_cast<size_t>(a_safe_row) * a_stride + a_col_base + a_col8 * 8, a_src_sz);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int4_cp_async16_a_pred(sd, a_base + static_cast<size_t>(a_safe_row) * a_stride + a_col_base + a_col8 * 8, a_src_sz);
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

  auto issue_next = [&](int st, bool need_fence) {
    if (use_tma) {
      issue_w_tma_only(st, pre_iter, need_fence);
      issue_a_next(st);
      return;
    }
    const uint8_t* w_src = W_packed + static_cast<size_t>(pre_iter) * kStepWBytes;
    {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int4_cp_async16_w(sd, w_src + tid * 16);
    }
    if (tid < (kStepWU4 - kThreads)) {
      const int idx = kThreads + tid;
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][idx * 16]));
      int4_cp_async16_w(sd, w_src + idx * 16);
    }
    issue_a_next(st);
  };

  constexpr int kPreload = kStages - 1;  // 4
  const int preload = (my_iters < kPreload) ? my_iters : kPreload;
  if ((pdl_mode & 1) != 0) {
    if (use_tma) {
      for (int i = 0; i < preload; ++i) {
        issue_w_tma_only(i, pre_iter + i, false);
      }
      asm volatile("griddepcontrol.wait;" ::: "memory");
      for (int i = 0; i < preload; ++i) {
        issue_a_next(i);
      }
    } else {
      for (int i = 0; i < preload; ++i) {
        issue_w_only(i, pre_iter + i);
      }
      asm volatile("griddepcontrol.wait;" ::: "memory");
      for (int i = 0; i < preload; ++i) {
        const __nv_bfloat16* a_base = A;
        int64_t a_stride = static_cast<int64_t>(K);
        if constexpr (kAux5) {
          int4_aux5_select(aux5, pre_seg, a_base, a_stride);
        }
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
    }
  } else {
    for (int i = 0; i < preload; ++i) {
      issue_next(i, false);
    }
  }

  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int ldm_row = kM16 ? (lane & 15) : (lane & 7);
  const int ldm_col_off = kM16 ? (lane >> 4) : ((lane >> 3) & 1);
  const int local_c0 = (warp * 2 + 0) * 8 + 2 * t;
  const int local_c1 = (warp * 2 + 1) * 8 + 2 * t;
  const int trig_i = ((pdl_mode & 2) != 0) ? ((my_iters > 6) ? (my_iters - 6) : 0) : -1;

  int st = 0;
  uint32_t mbar_phase = 0u;
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
    if (use_tma) {
      int4_mbar_wait(&smem.mbar[st], mbar_phase);
    }
    __syncthreads();

    if (i + kPreload < my_iters) {
      const int next_st = (st == 0) ? (kStages - 1) : (st - 1);
      issue_next(next_st, true);
    }

    const uint8_t* wt0 = &smem.w[st][(warp * 2 + 0) * kTileBytes];
    const uint8_t* wt1 = &smem.w[st][(warp * 2 + 1) * kTileBytes];
    const uint32_t q0_lo = reinterpret_cast<const uint32_t*>(wt0)[lane];
    const uint32_t q0_hi = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
    const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 256)[g];
    const __nv_bfloat162 sc0_2 = __bfloat162bfloat162(sz0.x);
    const __nv_bfloat162 zr0_2 = __bfloat162bfloat162(sz0.y);
    const float sc0_f = kNf4 ? __bfloat162float(sz0.x) : 0.0f;
    const float zr0_f = kNf4 ? __bfloat162float(sz0.y) : 0.0f;

    const uint32_t q1_lo = reinterpret_cast<const uint32_t*>(wt1)[lane];
    const uint32_t q1_hi = reinterpret_cast<const uint32_t*>(wt1 + 128)[lane];
    const __nv_bfloat162 sz1 = reinterpret_cast<const __nv_bfloat162*>(wt1 + 256)[g];
    const __nv_bfloat162 sc1_2 = __bfloat162bfloat162(sz1.x);
    const __nv_bfloat162 zr1_2 = __bfloat162bfloat162(sz1.y);
    const float sc1_f = kNf4 ? __bfloat162float(sz1.x) : 0.0f;
    const float zr1_f = kNf4 ? __bfloat162float(sz1.y) : 0.0f;

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][ldm_row][pcol8 * 8]));
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        int4_ldmatrix_x4(a0, a1, a2, a3, addr);
      } else {
        int4_ldmatrix_x2(a0, a2, addr);
      }
      uint32_t b0, b1, c0, c1;
      if ((s & 1) == 0) {
        int4_deq4_k16<kNf4, false>((s < 2) ? q0_lo : q0_hi, sc0_2, zr0_2, sc0_f, zr0_f, b0, b1);
        int4_deq4_k16<kNf4, false>((s < 2) ? q1_lo : q1_hi, sc1_2, zr1_2, sc1_f, zr1_f, c0, c1);
      } else {
        int4_deq4_k16<kNf4, true>((s < 2) ? q0_lo : q0_hi, sc0_2, zr0_2, sc0_f, zr0_f, b0, b1);
        int4_deq4_k16<kNf4, true>((s < 2) ? q1_lo : q1_hi, sc1_2, zr1_2, sc1_f, zr1_f, c0, c1);
      }
      if constexpr (kM16) {
        int4_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0, a1, a2, a3, b0, b1);
        int4_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0, a1, a2, a3, c0, c1);
      } else {
        int4_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b0, b1);
        int4_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, c0, c1);
      }
    }

    if (++st == kStages) {
      st = 0;
      mbar_phase ^= 1u;
    }

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
        int num_parts = 0;
        int part_role = 0;
        if constexpr (kWholeSplit) {
          num_parts = 2;
          part_role = (seg_start_kg != 0) ? 1 : 0;
        } else if (sk_multi_part != nullptr && total_iters >= grid_ctas) {
          const int tile_lo = cur_n * num_kg;
          const int tile_hi = tile_lo + num_kg;
          const int max_parts = allow_3part ? 4 : 2;
          int first_c = cta_id;
          while (first_c > 0 && (cta_id - first_c) < max_parts &&
                 static_cast<int>((static_cast<int64_t>(first_c) * total_iters) / grid_ctas) > tile_lo) {
            --first_c;
          }
          const bool lo_ok =
              (static_cast<int>((static_cast<int64_t>(first_c) * total_iters) / grid_ctas) <= tile_lo);
          int last_c = cta_id;
          while (last_c + 1 < grid_ctas && (last_c - first_c + 1) <= max_parts &&
                 static_cast<int>((static_cast<int64_t>(last_c + 1) * total_iters) / grid_ctas) < tile_hi) {
            ++last_c;
          }
          const bool hi_ok =
              (static_cast<int>((static_cast<int64_t>(last_c + 1) * total_iters) / grid_ctas) >= tile_hi);
          const int np = last_c - first_c + 1;
          if (lo_ok && hi_ok && np >= 2 && np <= max_parts) {
            num_parts = np;
            part_role = cta_id - first_c;
          }
        }

        if (num_parts >= 2 && num_parts <= 4) {
          // Deterministic non-atomic float2 reduction (2-part, 3-part & 4-part):
          // coalesced float2 stores + 1 int32 arrival counter + L2-coherent ld.global.cg.v2.f32 loads.
          // 2-part: IEEE-754 float+ is commutative => bit-deterministic.
          // 3/4-part: strict fixed-order (((v0 + v1) + v2) + v3) => 100% bit-deterministic across all 64 fc tiles!
          float* tile_mp = sk_multi_part + static_cast<size_t>(cur_n) * (4 * 16 * 64);
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
            s_is_last = (atomicAdd(&sk_done[cur_n], 1) + 1 == num_parts) ? 1 : 0;
          }
          __syncthreads();
          if (s_is_last) {
            if (num_parts == 2) {
              const float* peer_p = tile_mp + static_cast<size_t>(1 - part_role) * (16 * 64);
              if (g < M) {
                const float* row_p = peer_p + g * 64;
                const float2 p0 = int4_ldcg_f2(reinterpret_cast<const float2*>(&row_p[local_c0]));
                const float2 p1 = int4_ldcg_f2(reinterpret_cast<const float2*>(&row_p[local_c1]));
                *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) =
                    int4_bf16x2_u32(__fadd_rn(d0_0, p0.x), __fadd_rn(d1_0, p0.y));
                *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col1) =
                    int4_bf16x2_u32(__fadd_rn(d0_1, p1.x), __fadd_rn(d1_1, p1.y));
              }
              if constexpr (kM16) {
                if (has_hi) {
                  const float* row_p = peer_p + (g + 8) * 64;
                  const float2 p0 = int4_ldcg_f2(reinterpret_cast<const float2*>(&row_p[local_c0]));
                  const float2 p1 = int4_ldcg_f2(reinterpret_cast<const float2*>(&row_p[local_c1]));
                  *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * N + col0) =
                      int4_bf16x2_u32(__fadd_rn(d2_0, p0.x), __fadd_rn(d3_0, p0.y));
                  *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * N + col1) =
                      int4_bf16x2_u32(__fadd_rn(d2_1, p1.x), __fadd_rn(d3_1, p1.y));
                }
              }
            } else {
              auto sum_k_pair = [&](int row_off, int col_off, float m0, float m1) -> uint32_t {
                float sx = 0.0f;
                float sy = 0.0f;
                for (int r = 0; r < num_parts; ++r) {
                  float rx, ry;
                  if (r == part_role) {
                    rx = m0;
                    ry = m1;
                  } else {
                    const float2 pv = int4_ldcg_f2(reinterpret_cast<const float2*>(
                        tile_mp + static_cast<size_t>(r) * (16 * 64) + row_off * 64 + col_off));
                    rx = pv.x;
                    ry = pv.y;
                  }
                  sx = (r == 0) ? rx : __fadd_rn(sx, rx);
                  sy = (r == 0) ? ry : __fadd_rn(sy, ry);
                }
                return int4_bf16x2_u32(sx, sy);
              };
              if (g < M) {
                *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) =
                    sum_k_pair(g, local_c0, d0_0, d1_0);
                *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col1) =
                    sum_k_pair(g, local_c1, d0_1, d1_1);
              }
              if constexpr (kM16) {
                if (has_hi) {
                  const int ghi = g + 8;
                  *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(ghi) * N + col0) =
                      sum_k_pair(ghi, local_c0, d2_0, d3_0);
                  *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(ghi) * N + col1) =
                      sum_k_pair(ghi, local_c1, d2_1, d3_1);
                }
              }
            }
            __syncthreads();
            if (tid == 0) {
              atomicExch(&sk_done[cur_n], 0);
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

static int int4_read_env_int(const char* name, int def_val) {
  const char* s = std::getenv(name);
  if (!s || !*s) return def_val;
  return std::atoi(s);
}

bool int4_nf4_for_shape(int K) {
  int mode = 0;
  if (int4_bench_dyn_env()) {
    mode = int4_read_env_int("MUSE_INT4_NF4", 0);
  } else {
    static const int cached = int4_read_env_int("MUSE_INT4_NF4", 0);
    mode = cached;
  }
  if (mode == 2) return (K <= 16384);
  return (mode != 0);
}

int int4_mode() {
  if (int4_bench_dyn_env()) return int4_read_env_int("MUSE_INT4_MODE", 0);
  static const int m = int4_read_env_int("MUSE_INT4_MODE", 0);
  return m;
}

int int4_fc_mode() {
  if (int4_bench_dyn_env()) return int4_read_env_int("MUSE_INT4_FC_MODE", 0);
  static const int m = int4_read_env_int("MUSE_INT4_FC_MODE", 0);
  return m;
}

int int4_sk_grid() {
  auto parse = []() -> int {
    const int v = int4_read_env_int("MUSE_DRAFT_SK_GRID", 192);
    return (v >= 0 && v <= 576) ? v : 192;
  };
  if (int4_bench_dyn_env()) return parse();
  static const int g = parse();
  return g;
}

// 0=off, 1=2-part non-atomic only, 2=2-part & 3-part non-atomic (default: 2 -> 100% non-atomic on all drafter INT4 shapes)
int int4_two_part_red_mode() {
  if (int4_bench_dyn_env()) return int4_read_env_int("MUSE_INT4_TWO_PART_RED", 2);
  static const int m = int4_read_env_int("MUSE_INT4_TWO_PART_RED", 2);
  return m;
}

// Query/attention INT4 Stream-K mode (for qkv_proj, o_proj, fkv):
//   0 = direct 5-stage XOR-ldmatrix + BF16x2 + dual-PDL (+ K-major wave-interleaving when MUSE_INT4_ILV=1)
//   1 = whole-tile + exact 2-part bisect (e1f584e4)
//   2 = balanced Stream-K + 2/3-part non-atomic float2
//   3 = smart hybrid
int int4_query_mode() {
  if (int4_bench_dyn_env()) return int4_read_env_int("MUSE_INT4_QUERY_MODE", 0);
  static const int m = int4_read_env_int("MUSE_INT4_QUERY_MODE", 0);
  return m;
}

int int4_ilv_mode() {
  if (int4_bench_dyn_env()) return int4_read_env_int("MUSE_INT4_ILV", 1);
  static const int m = int4_read_env_int("MUSE_INT4_ILV", 1);
  return m;
}

int int4_n32_sk_grid_for_shape(int N, int K) {
  const int num_n32 = N >> 5;
  const int num_kg = K >> 6;
  if (num_n32 <= 0 || num_n32 > 384 || K > 8192 || num_kg < 2) return 0;
  auto eval = [&](int en, int cfg_g) -> int {
    if (en == 0 || int4_mode() != 0 || int4_query_mode() != 0) return 0;
    const int g = (num_n32 == 192 || num_n32 == 240) ? num_n32 : cfg_g;
    if (g <= 0 || (g % 48) != 0 || g > 240) return 0;
    const int min_iters = static_cast<int>((static_cast<int64_t>(num_n32) * num_kg) / g);
    if (2 * min_iters < num_kg) return 0;
    return g;
  };
  if (int4_bench_dyn_env()) {
    return eval(int4_read_env_int("MUSE_INT4_N32_SK", 0),
                int4_read_env_int("MUSE_INT4_N32_SK_GRID", 240));
  }
  static const int cached_en = int4_read_env_int("MUSE_INT4_N32_SK", 0);
  static const int cached_g = int4_read_env_int("MUSE_INT4_N32_SK_GRID", 240);
  return eval(cached_en, cached_g);
}

int int4_tma_enabled();

bool int4_n32_pair_tma_for_shape(int N, int K) {
  const int num_n32 = N >> 5;
  const int n_blocks = N / kRowsPerCTA;
  if (num_n32 <= 0 || num_n32 > 336 || (K & 127) != 0 || K > 8192) return false;
  auto eval = [&](int en, int qkv_en) -> bool {
    if ((n_blocks % 48) == 0 && (qkv_en == 0 || num_n32 > 240)) return false;
    return (en != 0) && (int4_mode() == 0) && (int4_query_mode() == 0) &&
           (int4_tma_enabled() != 0) && (int4_n32_sk_grid_for_shape(N, K) == 0);
  };
  if (int4_bench_dyn_env()) {
    return eval(int4_read_env_int("MUSE_INT4_N32_PAIR_TMA", 1),
                int4_read_env_int("MUSE_INT4_N32_PAIR_QKV", 1));
  }
  static const int cached_en = int4_read_env_int("MUSE_INT4_N32_PAIR_TMA", 1);
  static const int cached_qkv = int4_read_env_int("MUSE_INT4_N32_PAIR_QKV", 1);
  return eval(cached_en, cached_qkv);
}

bool int4_k_major_for_shape(int N, int K) {
  return (int4_ilv_mode() != 0) && (int4_mode() == 0) && (int4_query_mode() == 0) &&
         !int4_n32_pair_tma_for_shape(N, K) &&
         ((N / kRowsPerCTA) <= 128) && (K <= 8192);
}

int int4_query_grid() {
  auto parse = []() -> int {
    const int v = int4_read_env_int("MUSE_INT4_QUERY_GRID", 192);
    return (v >= 0 && v <= 576) ? v : 192;
  };
  if (int4_bench_dyn_env()) return parse();
  static const int g = parse();
  return g;
}

int int4_tma_enabled() {
  if (int4_bench_dyn_env()) return int4_read_env_int("MUSE_INT4_TMA", 1);
  static const int en = int4_read_env_int("MUSE_INT4_TMA", 1);
  return en;
}

int int4_pdl_mode() {
  if (int4_bench_dyn_env()) {
    const int v = int4_read_env_int("MUSE_DRAFT_PDL", 3);
    return (v != 0) ? 3 : 0;
  }
  static const int mode = (int4_read_env_int("MUSE_DRAFT_PDL", 3) != 0) ? 3 : 0;
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
                            float*& partials_ptr, float*& multi_part_ptr, int*& done_ptr,
                            int red_mode) {
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
  const bool use_mp = (red_mode != 0);
  if (use_mp && (!g_sk_multi_part.defined() || g_sk_multi_part.device() != opts.device() || g_sk_mp_cap < n_blocks)) {
    const int mp_cap = (n_blocks > 256) ? n_blocks : 256;
    g_sk_multi_part = torch::empty({2, static_cast<int64_t>(mp_cap) * 4 * 16 * 64}, opts.dtype(torch::kFloat32));
    g_sk_mp_cap = mp_cap;
  }
  const int slot = static_cast<int>(g_sk_slot & 1u);
  g_sk_slot ^= 1u;
  partials_ptr = g_sk_partials.data_ptr<float>() + static_cast<size_t>(slot) * static_cast<size_t>(g_sk_cap) * 16 * 64;
  multi_part_ptr = use_mp
      ? (g_sk_multi_part.data_ptr<float>() + static_cast<size_t>(slot) * static_cast<size_t>(g_sk_mp_cap) * 4 * 16 * 64)
      : nullptr;
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
  cudaFuncSetAttribute(int4_mma_streamk_kernel<false, true, false, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_streamk_kernel<true, true, false, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_streamk_kernel<false, false, false, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_streamk_kernel<true, false, false, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_streamk_kernel<false, false, true, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_streamk_kernel<true, false, true, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_n32_tma_det_sk_kernel<false, kNf4, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_n32_tma_det_sk_kernel<true, kNf4, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_n32_tma_det_sk_kernel<false, kNf4, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_n32_tma_det_sk_kernel<true, kNf4, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_n32_pair_tma_kernel<false, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int4_mma_gemm_n32_pair_tma_kernel<true, kNf4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
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
    int M, int64_t N, int64_t K,
    bool query_streamk = false) {
  int4_ensure_carveout();
  auto C = torch::empty({M, N}, opts);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n_blocks = static_cast<int>(N / kRowsPerCTA);
  const int n32_blocks = static_cast<int>(N / 32);
  const int num_kg = static_cast<int>(K >> 6);
  const bool use_m16 = (M > 8);
  const int mode = int4_mode();
  const int fc_mode = int4_fc_mode();
  const int sk_grid = int4_sk_grid();
  const int pdl_mode = int4_pdl_mode();
  const int red_mode = int4_two_part_red_mode();
  const int allow_3part = (red_mode >= 2) ? 1 : 0;
  const int use_tma =
      (int4_tma_enabled() != 0 && ((reinterpret_cast<uintptr_t>(W_packed.data_ptr()) & 15u) == 0u)) ? 1 : 0;

  if (is_aux5 && fc_mode == 1 && n32_blocks <= 288) {
    if (use_m16) {
      int4_launch_pdl(
          int4_mma_gemm_n32_kernel<false, true, true, kNf4>, dim3(n32_blocks, 1, 1), kThreads, st, pdl_mode,
          static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode);
    } else {
      int4_launch_pdl(
          int4_mma_gemm_n32_kernel<false, false, true, kNf4>, dim3(n32_blocks, 1, 1), kThreads, st, pdl_mode,
          static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode);
    }
    return C;
  }

  if (is_aux5 && fc_mode == 0 && sk_grid > 0) {
    const int total_iters = n_blocks * num_kg;
    const int grid = (sk_grid < total_iters) ? sk_grid : total_iters;
    float* p_ptr = nullptr;
    float* mp_ptr = nullptr;
    int* d_ptr = nullptr;
    ensure_int4_sk_scratch(opts, n_blocks, p_ptr, mp_ptr, d_ptr, red_mode);
    if (use_m16) {
      int4_launch_pdl(
          int4_mma_streamk_kernel<true, true, false, kNf4>, dim3(grid, 1, 1), kThreads, st, pdl_mode,
          static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, mp_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode, allow_3part, use_tma);
    } else {
      int4_launch_pdl(
          int4_mma_streamk_kernel<false, true, false, kNf4>, dim3(grid, 1, 1), kThreads, st, pdl_mode,
          static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, mp_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode, allow_3part, use_tma);
    }
    return C;
  }

  if (!is_aux5) {
    const int n32_sk = int4_n32_sk_grid_for_shape(static_cast<int>(N), static_cast<int>(K));
    if (n32_sk > 0) {
      float* p_ptr = nullptr;
      int* d_ptr = nullptr;
      ensure_int4_n32_sk_scratch(opts, n32_blocks, p_ptr, d_ptr);
      if (use_tma) {
        if (use_m16) {
          int4_launch_pdl(
              int4_mma_n32_tma_det_sk_kernel<true, kNf4, true>, dim3(n32_sk, 1, 1), kThreads, st, pdl_mode,
              A_ptr, W_packed.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
              p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n32_sk, pdl_mode);
        } else {
          int4_launch_pdl(
              int4_mma_n32_tma_det_sk_kernel<false, kNf4, true>, dim3(n32_sk, 1, 1), kThreads, st, pdl_mode,
              A_ptr, W_packed.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
              p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n32_sk, pdl_mode);
        }
      } else {
        if (use_m16) {
          int4_launch_pdl(
              int4_mma_n32_tma_det_sk_kernel<true, kNf4, false>, dim3(n32_sk, 1, 1), kThreads, st, pdl_mode,
              A_ptr, W_packed.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
              p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n32_sk, pdl_mode);
        } else {
          int4_launch_pdl(
              int4_mma_n32_tma_det_sk_kernel<false, kNf4, false>, dim3(n32_sk, 1, 1), kThreads, st, pdl_mode,
              A_ptr, W_packed.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
              p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n32_sk, pdl_mode);
        }
      }
      return C;
    }
  }

  if (!is_aux5 && use_tma && int4_n32_pair_tma_for_shape(static_cast<int>(N), static_cast<int>(K))) {
    if (use_m16) {
      int4_launch_pdl(
          int4_mma_gemm_n32_pair_tma_kernel<true, kNf4>, dim3(n32_blocks, 1, 1), kThreads, st, pdl_mode,
          A_ptr, W_packed.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          M, static_cast<int>(N), static_cast<int>(K), pdl_mode);
    } else {
      int4_launch_pdl(
          int4_mma_gemm_n32_pair_tma_kernel<false, kNf4>, dim3(n32_blocks, 1, 1), kThreads, st, pdl_mode,
          A_ptr, W_packed.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          M, static_cast<int>(N), static_cast<int>(K), pdl_mode);
    }
    return C;
  }

  const int k_major = (!is_aux5 && int4_k_major_for_shape(static_cast<int>(N), static_cast<int>(K))) ? 1 : 0;

  // Non-aux5 query/context attention projections (qkv_proj, o_proj, fkv):
  // whole-tile + deterministic non-atomic 2-part bisect (e1f584e4) or balanced 2/3-part Stream-K.
  if (!is_aux5 && !k_major && query_streamk && mode == 0 && num_kg >= 2 && n_blocks > 0) {
    const int qmode = int4_query_mode();
    const int qgrid_cfg = int4_query_grid();
    if (qmode > 0 && qgrid_cfg > 0 && n_blocks < qgrid_cfg) {
      const int total_iters = n_blocks * num_kg;
      bool use_whole = false;
      int grid = 0;
      if (qmode == 1) {
        use_whole = true;
        grid = std::min(qgrid_cfg, 2 * n_blocks);
      } else if (qmode == 2) {
        use_whole = false;
        grid = std::min(qgrid_cfg, total_iters);
      } else {
        const int g_ws = std::min(qgrid_cfg, 2 * n_blocks);
        const int whole_tiles = 2 * n_blocks - g_ws;
        if (whole_tiles == 0 || (whole_tiles % 48 == 0 && (g_ws - whole_tiles) % 48 == 0)) {
          use_whole = true;
          grid = g_ws;
        } else {
          use_whole = false;
          grid = std::min(qgrid_cfg, total_iters);
        }
      }
      if (grid > n_blocks) {
        float* p_ptr = nullptr;
        float* mp_ptr = nullptr;
        int* d_ptr = nullptr;
        ensure_int4_sk_scratch(opts, n_blocks, p_ptr, mp_ptr, d_ptr, use_whole ? std::max(red_mode, 1) : red_mode);
        if (use_whole) {
          if (use_m16) {
            int4_launch_pdl(
                int4_mma_streamk_kernel<true, false, true, kNf4>, dim3(grid, 1, 1), kThreads, st, pdl_mode,
                A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
                reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
                p_ptr, mp_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode, allow_3part, use_tma);
          } else {
            int4_launch_pdl(
                int4_mma_streamk_kernel<false, false, true, kNf4>, dim3(grid, 1, 1), kThreads, st, pdl_mode,
                A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
                reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
                p_ptr, mp_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode, allow_3part, use_tma);
          }
        } else {
          if (use_m16) {
            int4_launch_pdl(
                int4_mma_streamk_kernel<true, false, false, kNf4>, dim3(grid, 1, 1), kThreads, st, pdl_mode,
                A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
                reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
                p_ptr, mp_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode, allow_3part, use_tma);
          } else {
            int4_launch_pdl(
                int4_mma_streamk_kernel<false, false, false, kNf4>, dim3(grid, 1, 1), kThreads, st, pdl_mode,
                A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
                reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
                p_ptr, mp_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid, pdl_mode, allow_3part, use_tma);
          }
        }
        return C;
      }
    }
  }

  if (!is_aux5 && mode == 1 && n32_blocks <= 288) {
    if (use_m16) {
      int4_launch_pdl(
          int4_mma_gemm_n32_kernel<false, true, false, kNf4>, dim3(n32_blocks, 1, 1), kThreads, st, pdl_mode,
          A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode);
    } else {
      int4_launch_pdl(
          int4_mma_gemm_n32_kernel<false, false, false, kNf4>, dim3(n32_blocks, 1, 1), kThreads, st, pdl_mode,
          A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode);
    }
    return C;
  }

  int split_k = 1;
  if (n_blocks <= 128 && K >= 16384) {
    split_k = (K >= 28672) ? 4 : 3;
  } else if (!is_aux5 && mode == 2 && n_blocks <= 128 && K >= 4096) {
    split_k = 2;
  } else if (is_aux5 && fc_mode == 2) {
    split_k = 2;
  }
  if (split_k == 1) {
    const dim3 grid(n_blocks, 1, 1);
    if (is_aux5) {
      if (use_m16) {
        int4_launch_pdl(
            int4_mma_gemm_kernel<false, true, true, kNf4>, grid, kThreads, st, pdl_mode,
            static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode, 0, use_tma);
      } else {
        int4_launch_pdl(
            int4_mma_gemm_kernel<false, false, true, kNf4>, grid, kThreads, st, pdl_mode,
            static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode, 0, use_tma);
      }
    } else {
      if (use_m16) {
        int4_launch_pdl(
            int4_mma_gemm_kernel<false, true, false, kNf4>, grid, kThreads, st, pdl_mode,
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode, k_major, use_tma);
      } else {
        int4_launch_pdl(
            int4_mma_gemm_kernel<false, false, false, kNf4>, grid, kThreads, st, pdl_mode,
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl_mode, k_major, use_tma);
      }
    }
  } else {
    const int64_t need = static_cast<int64_t>(split_k) * M * N;
    float* partials_ptr = ensure_int4_splitk_partials(opts, need);
    const dim3 grid(n_blocks, split_k, 1);
    if (is_aux5) {
      if (use_m16) {
        int4_launch_pdl(
            int4_mma_gemm_kernel<true, true, true, kNf4>, grid, kThreads, st, pdl_mode,
            static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
            static_cast<__nv_bfloat16*>(nullptr), partials_ptr,
            M, static_cast<int>(N), static_cast<int>(K), split_k, pdl_mode, 0, use_tma);
      } else {
        int4_launch_pdl(
            int4_mma_gemm_kernel<true, false, true, kNf4>, grid, kThreads, st, pdl_mode,
            static_cast<const __nv_bfloat16*>(nullptr), aux5, W_packed.data_ptr<uint8_t>(),
            static_cast<__nv_bfloat16*>(nullptr), partials_ptr,
            M, static_cast<int>(N), static_cast<int>(K), split_k, pdl_mode, 0, use_tma);
      }
    } else {
      if (use_m16) {
        int4_launch_pdl(
            int4_mma_gemm_kernel<true, true, false, kNf4>, grid, kThreads, st, pdl_mode,
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            static_cast<__nv_bfloat16*>(nullptr), partials_ptr,
            M, static_cast<int>(N), static_cast<int>(K), split_k, pdl_mode, k_major, use_tma);
      } else {
        int4_launch_pdl(
            int4_mma_gemm_kernel<true, false, false, kNf4>, grid, kThreads, st, pdl_mode,
            A_ptr, aux5, W_packed.data_ptr<uint8_t>(),
            static_cast<__nv_bfloat16*>(nullptr), partials_ptr,
            M, static_cast<int>(N), static_cast<int>(K), split_k, pdl_mode, k_major, use_tma);
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
  vq4d_reset_norm_rot_table();
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kBFloat16 && W.dim() == 2 && W.is_contiguous(),
              "int4_pack expects contiguous CUDA BF16 [N, K]");
  const int N = static_cast<int>(W.size(0));
  const int K = static_cast<int>(W.size(1));
  TORCH_CHECK(N % kRowsPerCTA == 0 && K % 64 == 0, "int4_pack requires N%64==0 and K%64==0");
  const int64_t total_bytes = static_cast<int64_t>(N / 8) * static_cast<int64_t>(K / 64) * kTileBytes;
  auto out = torch::empty({total_bytes}, W.options().dtype(torch::kUInt8));
  dim3 grid(N / 8, K / 64);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n32_sk = int4_n32_sk_grid_for_shape(N, K);
  const int k_major = (n32_sk > 0) ? (-n32_sk) : (int4_n32_pair_tma_for_shape(N, K) ? 2 : (int4_k_major_for_shape(N, K) ? 1 : 0));
  if (int4_nf4_for_shape(K)) {
    if (mse_opt) {
      int4_pack_kernel<true, true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K, k_major);
    } else {
      int4_pack_kernel<false, true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K, k_major);
    }
  } else {
    if (mse_opt) {
      int4_pack_kernel<true, false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K, k_major);
    } else {
      int4_pack_kernel<false, false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K, k_major);
    }
  }
  return out;
}

torch::Tensor int4_gemm(
    torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K, bool query_streamk = false) {
  vq4d_clear_all_prerot();
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous(),
              "int4_gemm expects contiguous CUDA BF16 [M, K]");
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && A.size(1) == K && N % kRowsPerCTA == 0 && K % 64 == 0,
              "int4_gemm shape mismatch");
  Aux5Desc dummy{};
  if (int4_nf4_for_shape(static_cast<int>(K))) {
    return launch_int4_gemm_impl<true>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()), dummy, false, W_packed, A.options(), M, N, K, query_streamk);
  }
  return launch_int4_gemm_impl<false>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()), dummy, false, W_packed, A.options(), M, N, K, query_streamk);
}

torch::Tensor int4_pack_swiglu(torch::Tensor W, bool mse_opt = true) {
  vq4d_reset_norm_rot_table();
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
  vq4d_clear_all_prerot();
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous(),
              "int4_gemm_swiglu expects contiguous CUDA BF16 [M, K]");
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && A.size(1) == K && I % 32 == 0 && K % 64 == 0,
              "int4_gemm_swiglu shape mismatch");
  int4_ensure_carveout();
  auto C = torch::empty({M, I}, A.options());
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n_blocks = static_cast<int>(I / 32);
  const int pdl_mode = int4_pdl_mode();
  if (int4_nf4_for_shape(static_cast<int>(K))) {
    if (M > 8) {
      int4_launch_pdl(
          int4_mma_gemm_swiglu_kernel<true, true>, dim3(n_blocks, 1, 1), kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          M, static_cast<int>(I), static_cast<int>(K), pdl_mode);
    } else {
      int4_launch_pdl(
          int4_mma_gemm_swiglu_kernel<false, true>, dim3(n_blocks, 1, 1), kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          M, static_cast<int>(I), static_cast<int>(K), pdl_mode);
    }
  } else {
    if (M > 8) {
      int4_launch_pdl(
          int4_mma_gemm_swiglu_kernel<true, false>, dim3(n_blocks, 1, 1), kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          M, static_cast<int>(I), static_cast<int>(K), pdl_mode);
    } else {
      int4_launch_pdl(
          int4_mma_gemm_swiglu_kernel<false, false>, dim3(n_blocks, 1, 1), kThreads, st, pdl_mode,
          reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          M, static_cast<int>(I), static_cast<int>(K), pdl_mode);
    }
  }
  return C;
}

torch::Tensor int4_gemm_aux5(
    torch::Tensor h0, torch::Tensor h1, torch::Tensor h2, torch::Tensor h3, torch::Tensor h4,
    torch::Tensor W_packed, int64_t N, int64_t K) {
  vq4d_clear_all_prerot();
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
    return launch_int4_gemm_impl<true>(nullptr, desc, true, W_packed, h0.options(), M, N, K, false);
  }
  return launch_int4_gemm_impl<false>(nullptr, desc, true, W_packed, h0.options(), M, N, K, false);
}
