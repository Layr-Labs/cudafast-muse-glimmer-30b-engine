// SPDX-License-Identifier: Apache-2.0
// muse_int3_mma: 3-bit asymmetric group-64 W3A16 (exact 8-level Lloyd-Max NF3) Tensor-Core GEMM for DFlash.
// Format (3.50 bits/weight, 0 wasted bits, 16-byte aligned):
//   Each 8x64 tile (8 output rows x 64 K cols) is 224 bytes (14 x uint4):
//     [0..127]   uint32_t pk_lo[32] : for lane l=(r<<2)|t, w01 | (w23 << 24) (6-bit pair indices)
//     [128..191] uint16_t pk_hi[32] : for lane l=(r<<2)|t, static_cast<uint16_t>(w23 >> 8)
//     [192..223] __nv_bfloat162 sz[8] : (scale, zero) for rows r=0..7
//   Runs on SM121 with 5-stage cp.async.cg (zero .L2::256B UR hazard), wave-interleaved DRAM layout,
//   256B SMEM BF16x2 LUT + __hfma2 dequant, XOR-swizzled ldmatrix A, and SM121 PDL.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * 32;                    // 128
constexpr int kNTilesPerWarp = 2;                                // 16 N-rows per warp -> 2x A reuse
constexpr int kNTilesPerCTA = kWarpsPerBlock * kNTilesPerWarp;   // 8
constexpr int kRowsPerCTA = kNTilesPerCTA * 8;                   // 64
constexpr int kTileBytes = 224;                                  // 14 * 16 (asym 3.50 bpw)
constexpr int kStepWBytes = kNTilesPerCTA * kTileBytes;          // 1792 = 112 * 16
constexpr int kStepWU4 = kStepWBytes / 16;                       // 112
constexpr int kN32StepWBytes = 4 * kTileBytes;                   // 896 = 56 * 16
constexpr int kN32StepWU4 = kN32StepWBytes / 16;                 // 56
constexpr int kSymTileBytes = 200;                               // 192 B (3-bit) + 8 B (UE4M4 scales) = 3.125 bpw
constexpr int kSymStepWBytes = kNTilesPerCTA * kSymTileBytes;    // 1600 = 100 * 16 (64B aligned, TMA-valid)
constexpr int kSymN32StepWBytes = 4 * kSymTileBytes;             // 800 = 50 * 16 (32B aligned, TMA-valid)
constexpr int kSymFooterBytes = 16;                              // [magic, base_shifted, 0, 0]
constexpr int kSymH64FooterBytes = 32;                           // distinct size for H64 + sym NF3
constexpr uint32_t kSymMagic = 0x53594D33u;                      // "SYM3"
constexpr uint32_t kSymH64Magic = 0x48363433u;                   // "H643"
constexpr int kStages = 4;
constexpr int kN32Stages = 5;
constexpr int kSwgStages = 5;
constexpr int kDefaultSwgWaveIlv = 192;                          // 48 SMs * 4 CTAs/SM (optimal for Q=16 & M=6)

static bool int3_bench_dyn_env() {
  static const bool dyn = (std::getenv("MUSE_BENCH_DYN_ENV") != nullptr);
  return dyn;
}

static int int3_pdl_mode() {
  if (int3_bench_dyn_env()) {
    const char* s = std::getenv("MUSE_DRAFT_PDL");
    if (!s || !*s) return 3;
    return std::atoi(s);
  }
  static const int mode = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_PDL");
    if (!s || !*s) return 3;
    return (std::atoi(s) != 0) ? 3 : 0;
  }();
  return mode;
}

static bool int3_tma_enabled() {
  if (int3_bench_dyn_env()) {
    const char* s = std::getenv("MUSE_DRAFT_TMA");
    return (!s || !*s) ? true : (std::atoi(s) != 0);
  }
  static const bool en = []() -> bool {
    const char* s = std::getenv("MUSE_DRAFT_TMA");
    return (!s || !*s) ? true : (std::atoi(s) != 0);
  }();
  return en;
}

static bool int3_n32_tma_enabled() {
  if (int3_bench_dyn_env()) {
    const char* s = std::getenv("MUSE_DRAFT_TMA_N32");
    return (!s || !*s) ? false : (std::atoi(s) != 0);
  }
  static const bool en = []() -> bool {
    const char* s = std::getenv("MUSE_DRAFT_TMA_N32");
    return (!s || !*s) ? false : (std::atoi(s) != 0);
  }();
  return en;
}

__device__ __forceinline__ uint64_t int3_pol_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

__device__ __forceinline__ void int3_mbar_init(uint64_t* mbar, int count) {
  for (int s = 0; s < count; ++s) {
    const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[s]));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b) : "memory");
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void int3_tma_bulk_w(
    void* smem_dst, const void* gmem_src, int bytes, uint64_t* mbar, uint64_t pol) {
  const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
      ::"r"(d), "l"(gmem_src), "r"(bytes), "r"(b), "l"(pol)
      : "memory");
}

__device__ __forceinline__ void int3_mbar_wait(uint64_t* mbar, uint32_t parity) {
  const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "{\n\t.reg .pred P1;\n\tINT3_MBAR_WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@!P1 bra INT3_MBAR_WAIT_%=;\n\t}"
      ::"r"(b), "r"(parity)
      : "memory");
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

template <bool kNf3>
__device__ __forceinline__ float int3_deq_f32(uint32_t q) {
  if constexpr (kNf3) {
    // Exact 8-level Lloyd-Max Normal(0,1) quantiles {0, 48, 83, 113, 142, 172, 207, 255}.
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

// Standard 16B L2-cached weight cp.async (100% immune to SM121 .L2::256B UR hazard at all stage counts).
__device__ __forceinline__ void int3_cp_async16_w(uint32_t sd, const void* gs) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ void int3_cp_async16_a_pred(uint32_t sd, const void* gs, int src_sz) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(sd), "l"(gs), "r"(src_sz));
}

__device__ __forceinline__ void int3_ldmatrix_x2(uint32_t& a0, uint32_t& a2, uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(a0), "=r"(a2)
      : "r"(addr));
}

__device__ __forceinline__ void int3_ldmatrix_x4(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
      : "r"(addr));
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

__device__ __forceinline__ float int3_rad_sign(int k) {
  uint32_t x = static_cast<uint32_t>(k) * 0x9E3779B9u + 0x7F4A7C15u;
  x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
  x = (x ^ (x >> 13)) * 0xC2B2AE35u;
  x ^= (x >> 16);
  return (x & 1u) ? -1.0f : 1.0f;
}

__device__ __forceinline__ void int3_warp_fwht64(float& v0, float& v1, int lane) {
  {
    const float s = v0 + v1;
    const float d = v0 - v1;
    v0 = s;
    v1 = d;
  }
#pragma unroll
  for (int mask = 1; mask <= 16; mask <<= 1) {
    const float o0 = __shfl_xor_sync(0xffffffffu, v0, mask);
    const float o1 = __shfl_xor_sync(0xffffffffu, v1, mask);
    const bool hi = (lane & mask) != 0;
    v0 = hi ? (o0 - v0) : (v0 + o0);
    v1 = hi ? (o1 - v1) : (v1 + o1);
  }
  v0 *= 0.125f;
  v1 *= 0.125f;
}

__device__ __forceinline__ float int3_sym_nf3_f32(uint32_t q) {
  static constexpr float kSymNf3[8] = {
      -2.1519927f, -1.3439093f, -0.7560053f, -0.2450942f,
      +0.2450942f, +0.7560053f, +1.3439093f, +2.1519927f};
  return kSymNf3[q & 7u];
}

__device__ __forceinline__ int int3_quant_sym_nf3(float u) {
  if (u < 0.0f) {
    if (u < -1.0499510f) return (u < -1.7479510f) ? 0 : 1;
    return (u < -0.5005498f) ? 2 : 3;
  } else {
    if (u < 1.0499510f) return (u < 0.5005498f) ? 4 : 5;
    return (u < 1.7479510f) ? 6 : 7;
  }
}

__device__ __forceinline__ uint8_t int3_encode_ue4m4(float sc, uint32_t base_shifted) {
  if (!(sc > 0.0f)) return 0u;
  const __nv_bfloat16 b = __float2bfloat16_rn(sc);
  uint32_t u = static_cast<uint32_t>(__bfloat16_as_ushort(b));
  u = (u + 4u) & ~7u;
  if (u <= base_shifted) return 0u;
  uint32_t code = (u - base_shifted) >> 3;
  if (code > 255u) code = 255u;
  return static_cast<uint8_t>(code);
}

__device__ __forceinline__ __nv_bfloat16 int3_decode_ue4m4(uint8_t code, uint32_t base_shifted) {
  const uint16_t u = static_cast<uint16_t>((static_cast<uint32_t>(code) << 3) + base_shifted);
  return __ushort_as_bfloat16(u);
}

template <bool kNf3, bool kSym = false>
__device__ __forceinline__ void int3_init_lut(uint32_t* __restrict__ lut64, int tid) {
  if (tid < 64) {
    if constexpr (kSym) {
      lut64[tid] = int3_bf16x2_u32(
          int3_sym_nf3_f32(static_cast<uint32_t>(tid & 7)),
          int3_sym_nf3_f32(static_cast<uint32_t>((tid >> 3) & 7)));
    } else {
      lut64[tid] = int3_bf16x2_u32(
          int3_deq_f32<kNf3>(static_cast<uint32_t>(tid & 7)),
          int3_deq_f32<kNf3>(static_cast<uint32_t>((tid >> 3) & 7)));
    }
  }
}

template <bool kSym = false>
__device__ __forceinline__ void int3_load_tile_words(
    const uint8_t* __restrict__ wt, int lane, int g, uint32_t base_shifted,
    uint32_t& w01, uint32_t& w23, __nv_bfloat162& sc2, __nv_bfloat162& zr2) {
  const uint32_t pk_lo = reinterpret_cast<const uint32_t*>(wt)[lane];
  const uint32_t pk_hi = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wt + 128)[lane]);
  w01 = pk_lo;
  w23 = __funnelshift_r(pk_lo, pk_hi, 24);
  if constexpr (kSym) {
    sc2 = __bfloat162bfloat162(int3_decode_ue4m4(wt[192 + g], base_shifted));
  } else {
    const __nv_bfloat162 sz = reinterpret_cast<const __nv_bfloat162*>(wt + 192)[g];
    sc2 = __bfloat162bfloat162(sz.x);
    zr2 = __bfloat162bfloat162(sz.y);
  }
}

template <bool kSym = false>
__device__ __forceinline__ void int3_deq4_k16(
    const uint32_t* __restrict__ lut64,
    uint32_t w01, uint32_t w23, int s,
    __nv_bfloat162 sc2, __nv_bfloat162 zr2,
    uint32_t& b0, uint32_t& b1) {
  const uint32_t word = (s < 2) ? w01 : w23;
  const int sh = (s & 1) * 12;
  const uint32_t q01 = lut64[(word >> sh) & 63u];
  const uint32_t q23 = lut64[(word >> (sh + 6)) & 63u];
  if constexpr (kSym) {
    const __nv_bfloat162 v01 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q01), sc2);
    const __nv_bfloat162 v23 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q23), sc2);
    b0 = *reinterpret_cast<const uint32_t*>(&v01);
    b1 = *reinterpret_cast<const uint32_t*>(&v23);
  } else {
    const __nv_bfloat162 v01 = __hfma2(*reinterpret_cast<const __nv_bfloat162*>(&q01), sc2, zr2);
    const __nv_bfloat162 v23 = __hfma2(*reinterpret_cast<const __nv_bfloat162*>(&q23), sc2, zr2);
    b0 = *reinterpret_cast<const uint32_t*>(&v01);
    b1 = *reinterpret_cast<const uint32_t*>(&v23);
  }
}

template <bool kMseOpt, bool kNf3>
__device__ __forceinline__ void int3_quant_pack_row(
    const __nv_bfloat16* __restrict__ wrow, int r, int t,
    uint32_t& pk_lo, uint16_t& pk_hi, __nv_bfloat162& sz_bf) {
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

  constexpr float kInvMax = kNf3 ? (1.0f / 255.0f) : (1.0f / 7.0f);
  float sc_mm = (rmax - rmin) * kInvMax;
  if (sc_mm < 1e-30f) sc_mm = 1e-30f;
  sz_bf = __floats2bfloat162_rn(sc_mm, rmin);

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

  uint32_t w01 = 0u;
  uint32_t w23 = 0u;
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const uint32_t q0 = static_cast<uint32_t>(quant_val(vals[2 * p + 0], zr_q, inv_sc));
    const uint32_t q1 = static_cast<uint32_t>(quant_val(vals[2 * p + 1], zr_q, inv_sc));
    w01 |= ((q0 | (q1 << 3)) << (6 * p));
  }
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const uint32_t q0 = static_cast<uint32_t>(quant_val(vals[8 + 2 * p + 0], zr_q, inv_sc));
    const uint32_t q1 = static_cast<uint32_t>(quant_val(vals[8 + 2 * p + 1], zr_q, inv_sc));
    w23 |= ((q0 | (q1 << 3)) << (6 * p));
  }
  pk_lo = w01 | (w23 << 24);
  pk_hi = static_cast<uint16_t>(w23 >> 8);
}

template <bool kMseOpt, bool kNf3 = true>
__global__ void int3_pack_kernel(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ out,
    int N, int K, int n32_ilv) {
  const int n_tile = blockIdx.x;
  const int kg = blockIdx.y;
  const int lane = threadIdx.x;
  const int r = lane >> 2;
  const int t = lane & 3;
  const int row = n_tile * 8 + r;
  const int k_base = kg * 64;

  uint32_t pk_lo;
  uint16_t pk_hi;
  __nv_bfloat162 sz_bf;
  int3_quant_pack_row<kMseOpt, kNf3>(
      W + static_cast<size_t>(row) * K + k_base, r, t, pk_lo, pk_hi, sz_bf);

  const int num_kg = K >> 6;
  size_t tile_off = 0;
  if (n32_ilv > 0) {
    const int cta_n = n_tile >> 2;
    const int sub_n = n_tile & 3;
    const int num_cta = N >> 5;
    const int wave = cta_n / n32_ilv;
    const int in_wave = cta_n - wave * n32_ilv;
    const int wave_sz = ((wave + 1) * n32_ilv <= num_cta) ? n32_ilv : (num_cta - wave * n32_ilv);
    tile_off = (static_cast<size_t>(wave * n32_ilv) * num_kg +
                static_cast<size_t>(kg) * wave_sz + in_wave) * kN32StepWBytes +
               static_cast<size_t>(sub_n) * kTileBytes;
  } else {
    const int cta_n = n_tile >> 3;
    const int sub_tile = n_tile & 7;
    const size_t tile_idx = (static_cast<size_t>(cta_n) * num_kg + kg) * kNTilesPerCTA + sub_tile;
    tile_off = tile_idx * kTileBytes;
  }

  uint8_t* dst = out + tile_off;
  reinterpret_cast<uint32_t*>(dst)[lane] = pk_lo;
  reinterpret_cast<uint16_t*>(dst + 128)[lane] = pk_hi;
  if (t == 0) {
    reinterpret_cast<__nv_bfloat162*>(dst + 192)[r] = sz_bf;
  }
}

template <bool kMseOpt, bool kNf3 = true>
__global__ void int3_pack_swiglu_kernel(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ out,
    int I, int K, int swg_ilv) {
  const int pair_tile = blockIdx.x;
  const int kg = blockIdx.y;
  const int lane = threadIdx.x;
  const int r = lane >> 2;
  const int t = lane & 3;
  const int g_tile = pair_tile >> 1;
  const int is_up = pair_tile & 1;
  const int row = (is_up ? I : 0) + g_tile * 8 + r;
  const int k_base = kg * 64;

  uint32_t pk_lo;
  uint16_t pk_hi;
  __nv_bfloat162 sz_bf;
  int3_quant_pack_row<kMseOpt, kNf3>(
      W + static_cast<size_t>(row) * K + k_base, r, t, pk_lo, pk_hi, sz_bf);

  const int cta_n = g_tile >> 2;
  const int sub_tile = ((g_tile & 3) << 1) | is_up;
  const int num_kg = K >> 6;
  const int num_cta = I >> 5;
  const int wave = cta_n / swg_ilv;
  const int in_wave = cta_n - wave * swg_ilv;
  const int wave_sz = ((wave + 1) * swg_ilv <= num_cta) ? swg_ilv : (num_cta - wave * swg_ilv);
  const size_t tile_off = (static_cast<size_t>(wave * swg_ilv) * num_kg +
                           static_cast<size_t>(kg) * wave_sz + in_wave) * kStepWBytes +
                          static_cast<size_t>(sub_tile) * kTileBytes;

  uint8_t* dst = out + tile_off;
  reinterpret_cast<uint32_t*>(dst)[lane] = pk_lo;
  reinterpret_cast<uint16_t*>(dst + 128)[lane] = pk_hi;
  if (t == 0) {
    reinterpret_cast<__nv_bfloat162*>(dst + 192)[r] = sz_bf;
  }
}

template <bool kMseOpt>
__device__ __forceinline__ void int3_quant_pack_row_sym(
    const __nv_bfloat16* __restrict__ wrow, int r, int t, uint32_t base_shifted,
    uint32_t& pk_lo, uint16_t& pk_hi, uint8_t& sc_code) {
  float vals[16];
  float lss = 0.0f;
  float lamax = 0.0f;
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
    const float v = vals[i];
    lss = __fmaf_rn(v, v, lss);
    lamax = fmaxf(lamax, fabsf(v));
  }
  const float rms = sqrtf(fmaxf(int3_subwarp4_sum(lss) * (1.0f / 64.0f), 0.0f));
  lamax = fmaxf(lamax, __shfl_xor_sync(0xffffffffu, lamax, 1));
  lamax = fmaxf(lamax, __shfl_xor_sync(0xffffffffu, lamax, 2));

  uint8_t best_code = int3_encode_ue4m4(rms > 1e-30f ? rms : (lamax * (1.0f / 2.1519927f)), base_shifted);

  if constexpr (kMseOpt) {
    auto eval_code = [&](uint8_t code) -> float {
      const float sc_q = __bfloat162float(int3_decode_ue4m4(code, base_shifted));
      const float inv_sc = (sc_q > 1e-30f) ? (1.0f / sc_q) : 0.0f;
      float err = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int q = int3_quant_sym_nf3(vals[i] * inv_sc);
        const float fq = int3_sym_nf3_f32(static_cast<uint32_t>(q));
        const float w_hat = __bfloat162float(__float2bfloat16_rn(fq * sc_q));
        const float diff = vals[i] - w_hat;
        err = __fmaf_rn(diff, diff, err);
      }
      return int3_subwarp4_sum(err);
    };

    float best_mse = eval_code(best_code);
    auto consider_code = [&](uint8_t c) {
      const float m = eval_code(c);
      if (m < best_mse) {
        best_mse = m;
        best_code = c;
      }
    };

    constexpr float kRmsAlphas[8] = {0.72f, 0.80f, 0.88f, 0.94f, 1.06f, 1.14f, 1.24f, 1.36f};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      consider_code(int3_encode_ue4m4(rms * kRmsAlphas[k], base_shifted));
    }
    constexpr float kAmaxAlphas[3] = {1.00f / 2.1519927f, 0.88f / 2.1519927f, 0.76f / 2.1519927f};
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      consider_code(int3_encode_ue4m4(lamax * kAmaxAlphas[k], base_shifted));
    }

    float cur_sc = __bfloat162float(int3_decode_ue4m4(best_code, base_shifted));
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const float inv_cur = (cur_sc > 1e-30f) ? (1.0f / cur_sc) : 0.0f;
      float lsqq = 0.0f, lsqw = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int q = int3_quant_sym_nf3(vals[i] * inv_cur);
        const float fq = int3_sym_nf3_f32(static_cast<uint32_t>(q));
        lsqq = __fmaf_rn(fq, fq, lsqq);
        lsqw = __fmaf_rn(fq, vals[i], lsqw);
      }
      const float sqq = int3_subwarp4_sum(lsqq);
      const float sqw = int3_subwarp4_sum(lsqw);
      if (sqq > 1e-12f && sqw > 0.0f) {
        cur_sc = sqw / sqq;
        const uint8_t c0 = int3_encode_ue4m4(cur_sc, base_shifted);
        consider_code(c0);
        if (c0 > 0u) consider_code(static_cast<uint8_t>(c0 - 1u));
        if (c0 < 255u) consider_code(static_cast<uint8_t>(c0 + 1u));
      }
    }
  }

  const uint32_t code_u32 = __shfl_sync(0xffffffffu, static_cast<uint32_t>(best_code), r << 2);
  sc_code = static_cast<uint8_t>(code_u32);
  const float sc_q = __bfloat162float(int3_decode_ue4m4(sc_code, base_shifted));
  const float inv_sc = (sc_q > 1e-30f) ? (1.0f / sc_q) : 0.0f;

  uint32_t w01 = 0u;
  uint32_t w23 = 0u;
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const uint32_t q0 = static_cast<uint32_t>(int3_quant_sym_nf3(vals[2 * p + 0] * inv_sc));
    const uint32_t q1 = static_cast<uint32_t>(int3_quant_sym_nf3(vals[2 * p + 1] * inv_sc));
    w01 |= ((q0 | (q1 << 3)) << (6 * p));
  }
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const uint32_t q0 = static_cast<uint32_t>(int3_quant_sym_nf3(vals[8 + 2 * p + 0] * inv_sc));
    const uint32_t q1 = static_cast<uint32_t>(int3_quant_sym_nf3(vals[8 + 2 * p + 1] * inv_sc));
    w23 |= ((q0 | (q1 << 3)) << (6 * p));
  }
  pk_lo = w01 | (w23 << 24);
  pk_hi = static_cast<uint16_t>(w23 >> 8);
}

template <bool kMseOpt>
__global__ void int3_pack_sym_kernel(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ out,
    int N, int K, int n32_ilv, uint32_t base_shifted, size_t payload_bytes) {
  const int n_tile = blockIdx.x;
  const int kg = blockIdx.y;
  const int lane = threadIdx.x;
  const int r = lane >> 2;
  const int t = lane & 3;
  const int row = n_tile * 8 + r;
  const int k_base = kg * 64;

  uint32_t pk_lo;
  uint16_t pk_hi;
  uint8_t sc_code;
  int3_quant_pack_row_sym<kMseOpt>(
      W + static_cast<size_t>(row) * K + k_base, r, t, base_shifted, pk_lo, pk_hi, sc_code);

  const int num_kg = K >> 6;
  size_t tile_off = 0;
  if (n32_ilv > 0) {
    const int cta_n = n_tile >> 2;
    const int sub_n = n_tile & 3;
    const int num_cta = N >> 5;
    const int wave = cta_n / n32_ilv;
    const int in_wave = cta_n - wave * n32_ilv;
    const int wave_sz = ((wave + 1) * n32_ilv <= num_cta) ? n32_ilv : (num_cta - wave * n32_ilv);
    if ((num_kg & 1) == 0) {
      const int num_kg2 = num_kg >> 1;
      const int kg2 = kg >> 1;
      const int h = kg & 1;
      tile_off = (static_cast<size_t>(wave * n32_ilv) * num_kg2 +
                  static_cast<size_t>(kg2) * wave_sz + in_wave) * (2 * kSymN32StepWBytes) +
                 static_cast<size_t>(h) * kSymN32StepWBytes +
                 static_cast<size_t>(sub_n) * kSymTileBytes;
    } else {
      tile_off = (static_cast<size_t>(wave * n32_ilv) * num_kg +
                  static_cast<size_t>(kg) * wave_sz + in_wave) * kSymN32StepWBytes +
                 static_cast<size_t>(sub_n) * kSymTileBytes;
    }
  } else {
    const int cta_n = n_tile >> 3;
    const int sub_tile = n_tile & 7;
    const size_t tile_idx = (static_cast<size_t>(cta_n) * num_kg + kg) * kNTilesPerCTA + sub_tile;
    tile_off = tile_idx * kSymTileBytes;
  }

  uint8_t* dst = out + tile_off;
  reinterpret_cast<uint32_t*>(dst)[lane] = pk_lo;
  reinterpret_cast<uint16_t*>(dst + 128)[lane] = pk_hi;
  if (t == 0) {
    dst[192 + r] = sc_code;
  }
  if (n_tile == 0 && kg == 0 && lane == 0) {
    uint32_t* ftr = reinterpret_cast<uint32_t*>(out + payload_bytes);
    ftr[0] = kSymMagic;
    ftr[1] = base_shifted;
    ftr[2] = 0u;
    ftr[3] = 0u;
  }
}

// 256-thread (8-warp) H64+Rademacher + UE4M4 sym NF3-g64 packer for linear layers (3.125 bpw, ~15.5 dB).
template <bool kMseOpt>
__global__ void __launch_bounds__(256) int3_pack_sym_h64_kernel(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ out,
    int N, int K, int n32_ilv, uint32_t base_shifted, size_t payload_bytes) {
  __shared__ __nv_bfloat16 s_w[8][64];
  const int n_tile = blockIdx.x;
  const int kg = blockIdx.y;
  const int tid = threadIdx.x;
  const int rw = tid >> 5;
  const int lw = tid & 31;
  const int k_base = kg * 64;

  const __nv_bfloat16* wrow = W + static_cast<size_t>(n_tile * 8 + rw) * K + k_base;
  const int col0 = k_base + 2 * lw;
  float v0 = __bfloat162float(wrow[2 * lw + 0]) * int3_rad_sign(col0 + 0);
  float v1 = __bfloat162float(wrow[2 * lw + 1]) * int3_rad_sign(col0 + 1);
  int3_warp_fwht64(v0, v1, lw);
  s_w[rw][2 * lw + 0] = __float2bfloat16_rn(v0);
  s_w[rw][2 * lw + 1] = __float2bfloat16_rn(v1);
  __syncthreads();

  if (tid < 32) {
    const int lane = tid;
    const int r = lane >> 2;
    const int t = lane & 3;
    uint32_t pk_lo;
    uint16_t pk_hi;
    uint8_t sc_code;
    int3_quant_pack_row_sym<kMseOpt>(&s_w[r][0], r, t, base_shifted, pk_lo, pk_hi, sc_code);

    const int num_kg = K >> 6;
    size_t tile_off = 0;
    if (n32_ilv > 0) {
      const int cta_n = n_tile >> 2;
      const int sub_n = n_tile & 3;
      const int num_cta = N >> 5;
      const int wave = cta_n / n32_ilv;
      const int in_wave = cta_n - wave * n32_ilv;
      const int wave_sz = ((wave + 1) * n32_ilv <= num_cta) ? n32_ilv : (num_cta - wave * n32_ilv);
      if ((num_kg & 1) == 0) {
        const int num_kg2 = num_kg >> 1;
        const int kg2 = kg >> 1;
        const int h = kg & 1;
        tile_off = (static_cast<size_t>(wave * n32_ilv) * num_kg2 +
                    static_cast<size_t>(kg2) * wave_sz + in_wave) * (2 * kSymN32StepWBytes) +
                   static_cast<size_t>(h) * kSymN32StepWBytes +
                   static_cast<size_t>(sub_n) * kSymTileBytes;
      } else {
        tile_off = (static_cast<size_t>(wave * n32_ilv) * num_kg +
                    static_cast<size_t>(kg) * wave_sz + in_wave) * kSymN32StepWBytes +
                   static_cast<size_t>(sub_n) * kSymTileBytes;
      }
    } else {
      const int cta_n = n_tile >> 3;
      const int sub_tile = n_tile & 7;
      const size_t tile_idx = (static_cast<size_t>(cta_n) * num_kg + kg) * kNTilesPerCTA + sub_tile;
      tile_off = tile_idx * kSymTileBytes;
    }

    uint8_t* dst = out + tile_off;
    reinterpret_cast<uint32_t*>(dst)[lane] = pk_lo;
    reinterpret_cast<uint16_t*>(dst + 128)[lane] = pk_hi;
    if (t == 0) {
      dst[192 + r] = sc_code;
    }
    if (n_tile == 0 && kg == 0 && lane == 0) {
      uint32_t* ftr = reinterpret_cast<uint32_t*>(out + payload_bytes);
      ftr[0] = kSymH64Magic;
      ftr[1] = base_shifted;
      ftr[2] = 0u;
      ftr[3] = 0u;
      ftr[4] = 0u;
      ftr[5] = 0u;
      ftr[6] = 0u;
      ftr[7] = 0u;
    }
  }
}

// 256-thread (8-warp) H64+Rademacher + UE4M4 sym NF3-g64 packer for fused SwiGLU (3.125 bpw, ~15.5 dB).
template <bool kMseOpt>
__global__ void __launch_bounds__(256) int3_pack_swiglu_sym_h64_kernel(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ out,
    int I, int K, int swg_ilv, uint32_t base_shifted, size_t payload_bytes) {
  __shared__ __nv_bfloat16 s_w[8][64];
  const int pair_tile = blockIdx.x;
  const int kg = blockIdx.y;
  const int tid = threadIdx.x;
  const int rw = tid >> 5;
  const int lw = tid & 31;
  const int g_tile = pair_tile >> 1;
  const int is_up = pair_tile & 1;
  const int row = (is_up ? I : 0) + g_tile * 8 + rw;
  const int k_base = kg * 64;

  const __nv_bfloat16* wrow = W + static_cast<size_t>(row) * K + k_base;
  const int col0 = k_base + 2 * lw;
  float v0 = __bfloat162float(wrow[2 * lw + 0]) * int3_rad_sign(col0 + 0);
  float v1 = __bfloat162float(wrow[2 * lw + 1]) * int3_rad_sign(col0 + 1);
  int3_warp_fwht64(v0, v1, lw);
  s_w[rw][2 * lw + 0] = __float2bfloat16_rn(v0);
  s_w[rw][2 * lw + 1] = __float2bfloat16_rn(v1);
  __syncthreads();

  if (tid < 32) {
    const int lane = tid;
    const int r = lane >> 2;
    const int t = lane & 3;
    uint32_t pk_lo;
    uint16_t pk_hi;
    uint8_t sc_code;
    int3_quant_pack_row_sym<kMseOpt>(&s_w[r][0], r, t, base_shifted, pk_lo, pk_hi, sc_code);

    const int cta_n = g_tile >> 2;
    const int sub_tile = ((g_tile & 3) << 1) | is_up;
    const int num_kg = K >> 6;
    const int num_cta = I >> 5;
    const int wave = cta_n / swg_ilv;
    const int in_wave = cta_n - wave * swg_ilv;
    const int wave_sz = ((wave + 1) * swg_ilv <= num_cta) ? swg_ilv : (num_cta - wave * swg_ilv);
    const size_t tile_off = (static_cast<size_t>(wave * swg_ilv) * num_kg +
                             static_cast<size_t>(kg) * wave_sz + in_wave) * kSymStepWBytes +
                            static_cast<size_t>(sub_tile) * kSymTileBytes;

    uint8_t* dst = out + tile_off;
    reinterpret_cast<uint32_t*>(dst)[lane] = pk_lo;
    reinterpret_cast<uint16_t*>(dst + 128)[lane] = pk_hi;
    if (t == 0) {
      dst[192 + r] = sc_code;
    }
    if (pair_tile == 0 && kg == 0 && lane == 0) {
      uint32_t* ftr = reinterpret_cast<uint32_t*>(out + payload_bytes);
      ftr[0] = kSymH64Magic;
      ftr[1] = base_shifted;
      ftr[2] = 0u;
      ftr[3] = 0u;
      ftr[4] = 0u;
      ftr[5] = 0u;
      ftr[6] = 0u;
      ftr[7] = 0u;
    }
  }
}

template <bool kMseOpt>
__global__ void int3_pack_swiglu_sym_kernel(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ out,
    int I, int K, int swg_ilv, uint32_t base_shifted, size_t payload_bytes) {
  const int pair_tile = blockIdx.x;
  const int kg = blockIdx.y;
  const int lane = threadIdx.x;
  const int r = lane >> 2;
  const int t = lane & 3;
  const int g_tile = pair_tile >> 1;
  const int is_up = pair_tile & 1;
  const int row = (is_up ? I : 0) + g_tile * 8 + r;
  const int k_base = kg * 64;

  uint32_t pk_lo;
  uint16_t pk_hi;
  uint8_t sc_code;
  int3_quant_pack_row_sym<kMseOpt>(
      W + static_cast<size_t>(row) * K + k_base, r, t, base_shifted, pk_lo, pk_hi, sc_code);

  const int cta_n = g_tile >> 2;
  const int sub_tile = ((g_tile & 3) << 1) | is_up;
  const int num_kg = K >> 6;
  const int num_cta = I >> 5;
  const int wave = cta_n / swg_ilv;
  const int in_wave = cta_n - wave * swg_ilv;
  const int wave_sz = ((wave + 1) * swg_ilv <= num_cta) ? swg_ilv : (num_cta - wave * swg_ilv);
  const size_t tile_off = (static_cast<size_t>(wave * swg_ilv) * num_kg +
                           static_cast<size_t>(kg) * wave_sz + in_wave) * kSymStepWBytes +
                          static_cast<size_t>(sub_tile) * kSymTileBytes;

  uint8_t* dst = out + tile_off;
  reinterpret_cast<uint32_t*>(dst)[lane] = pk_lo;
  reinterpret_cast<uint16_t*>(dst + 128)[lane] = pk_hi;
  if (t == 0) {
    dst[192 + r] = sc_code;
  }
  if (pair_tile == 0 && kg == 0 && lane == 0) {
    uint32_t* ftr = reinterpret_cast<uint32_t*>(out + payload_bytes);
    ftr[0] = kSymMagic;
    ftr[1] = base_shifted;
    ftr[2] = 0u;
    ftr[3] = 0u;
  }
}

template <int kMaxM, int kStepWB = kStepWBytes>
struct alignas(128) Int3StageSmem {
  alignas(128) uint8_t w[kStages][kStepWB];             // 4 * 1792 = 7168 B (asym) or 4 * 1600 = 6400 B (sym)
  alignas(128) __nv_bfloat16 a[kStages][kMaxM][64];     // 4 * 1024 = 4096 B (M<=8) or 8192 B (M<=16), XOR-swizzled
  alignas(8) uint32_t lut64[64];                        // 256 B
  alignas(8) uint64_t mbar[kStages];                    // 32 B
};

template <int kMaxM, int kN32StepWB = kN32StepWBytes>
struct alignas(128) Int3N32StageSmem {
  alignas(128) uint8_t w[kN32Stages][kN32StepWB];       // 5 * 896 = 4480 B (asym) or 5 * 800 = 4000 B (sym)
  alignas(128) __nv_bfloat16 a[kN32Stages][kMaxM][64];  // 5 * 1024 = 5120 B (M<=8) or 10240 B (M<=16), XOR-swizzled
  alignas(8) uint32_t lut64[64];                        // 256 B
  alignas(8) uint64_t mbar[kN32Stages];                 // 40 B -> 9.2-9.7 KiB (M<=8) or 14.2-14.7 KiB (M<=16) -> 6 CTAs/SM!
};

template <bool kSplitK, bool kM16 = false, bool kNf3 = true, bool kTma = false, bool kSym = false>
__global__ void __launch_bounds__(128, kM16 ? 5 : 6) int3_mma_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ C_fp32,
    int M, int N, int K, int split_k, int pdl_mode) {
  constexpr int kTileB = kSym ? kSymTileBytes : kTileBytes;
  constexpr int kStepWB = kSym ? kSymStepWBytes : kStepWBytes;
  constexpr int kStepU4 = kStepWB / 16;
  __shared__ Int3StageSmem<kM16 ? 16 : 8, kStepWB> smem;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;

  int3_init_lut<kNf3, kSym>(smem.lut64, tid);

  const int cta_n = blockIdx.x;
  const int sk_idx = kSplitK ? blockIdx.y : 0;
  const int kg_per_sk = kSplitK ? ((num_kg + split_k - 1) / split_k) : num_kg;
  const int kg_start = kSplitK ? (sk_idx * kg_per_sk) : 0;
  int kg_end = kSplitK ? (kg_start + kg_per_sk) : num_kg;
  if (kg_end > num_kg) kg_end = num_kg;
  const int my_kgs = kg_end - kg_start;
  if (my_kgs <= 0) return;

  if constexpr (kTma) {
    if (tid == 0) {
      int3_mbar_init(smem.mbar, kStages);
    }
    __syncthreads();
  }

  const size_t cta_stride_bytes = static_cast<size_t>(num_kg) * kStepWB;
  const size_t total_bytes = static_cast<size_t>(N / kRowsPerCTA) * cta_stride_bytes;
  const uint32_t base_shifted =
      kSym ? __ldg(reinterpret_cast<const uint32_t*>(W_packed + total_bytes) + 1) : 0u;
  const uint8_t* w_cta = W_packed + static_cast<size_t>(cta_n) * cta_stride_bytes;

  if (cta_n == 0 && sk_idx == 0) {
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_phys_col8 = a_col8 ^ (a_row & 7);
  const int a_src_sz = (a_row < M) ? 16 : 0;
  const int a_safe_row = (a_row < M) ? a_row : 0;

  auto issue_stage = [&](int kg, int st, bool need_fence) {
    if constexpr (kTma) {
      if (tid == 0) {
        if (need_fence) {
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        const uint64_t pol = int3_pol_evict_first();
        int3_tma_bulk_w(
            smem.w[st], w_cta + static_cast<size_t>(kg) * kStepWB, kStepWB, &smem.mbar[st], pol);
      }
    } else {
      if (tid < kStepU4) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
        int3_cp_async16_w(sd, w_cta + static_cast<size_t>(kg) * kStepWB + tid * 16);
      }
    }
    if constexpr (kM16) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int3_cp_async16_a_pred(sd, A + static_cast<size_t>(a_safe_row) * K + kg * 64 + a_col8 * 8, a_src_sz);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int3_cp_async16_a_pred(sd, A + static_cast<size_t>(a_safe_row) * K + kg * 64 + a_col8 * 8, a_src_sz);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kStages - 1;  // 3
  const int preload = (my_kgs < kPreload) ? my_kgs : kPreload;
  for (int i = 0; i < preload; ++i) {
    issue_stage(kg_start + i, i, false);
  }

  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int ldm_row = kM16 ? (lane & 15) : (lane & 7);
  const int ldm_col_off = kM16 ? (lane >> 4) : ((lane >> 3) & 1);
  const int trig_i = (!kSplitK && ((pdl_mode & 2) != 0)) ? ((my_kgs > 4) ? (my_kgs - 4) : 0) : -1;

  int st = 0;
  uint32_t parity = 0u;
  for (int i = 0; i < my_kgs; ++i) {
    if (i == trig_i) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    if constexpr (kTma) {
      int3_mbar_wait(&smem.mbar[st], parity);
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
      const int next_st = (st == 0) ? (kStages - 1) : (st - 1);
      issue_stage(kg_start + next_i, next_st, true);
    }

    uint32_t w01_0, w23_0, w01_1, w23_1;
    __nv_bfloat162 sc0, zr0, sc1, zr1;
    int3_load_tile_words<kSym>(&smem.w[st][(warp * 2 + 0) * kTileB], lane, g, base_shifted, w01_0, w23_0, sc0, zr0);
    int3_load_tile_words<kSym>(&smem.w[st][(warp * 2 + 1) * kTileB], lane, g, base_shifted, w01_1, w23_1, sc1, zr1);

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][ldm_row][pcol8 * 8]));
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        int3_ldmatrix_x4(a0, a1, a2, a3, addr);
      } else {
        int3_ldmatrix_x2(a0, a2, addr);
      }
      uint32_t b0, b1, c0, c1;
      int3_deq4_k16<kSym>(smem.lut64, w01_0, w23_0, s, sc0, zr0, b0, b1);
      int3_deq4_k16<kSym>(smem.lut64, w01_1, w23_1, s, sc1, zr1, c0, c1);
      if constexpr (kM16) {
        int3_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0, a1, a2, a3, b0, b1);
        int3_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0, a1, a2, a3, c0, c1);
      } else {
        int3_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b0, b1);
        int3_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, c0, c1);
      }
    }

    if (++st == kStages) {
      st = 0;
      parity ^= 1u;
    }
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

// 32-row/CTA single-wave kernel (14.2-14.7 KiB SMEM -> 6 CTAs/SM = 288 CTAs/wave, 5-stage TMA/cp.async, wave-ilv, PDL).
template <bool kSplitK, bool kM16 = false, bool kNf3 = true, bool kTma = false, bool kSym = false>
__global__ void __launch_bounds__(128, 6) int3_mma_gemm_n32_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ C_fp32,
    int M, int N, int K, int split_k, int n32_ilv, int pdl_mode) {
  constexpr int kTileB = kSym ? kSymTileBytes : kTileBytes;
  constexpr int kN32StepWB = kSym ? kSymN32StepWBytes : kN32StepWBytes;
  constexpr int kN32StepU4 = kN32StepWB / 16;
  __shared__ Int3N32StageSmem<kM16 ? 16 : 8, kN32StepWB> smem;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;
  const int num_cta = N >> 5;

  int3_init_lut<kNf3, kSym>(smem.lut64, tid);

  const int cta_n = blockIdx.x;
  const int sk_idx = kSplitK ? blockIdx.y : 0;
  const int kg_per_sk = kSplitK ? ((num_kg + split_k - 1) / split_k) : num_kg;
  const int kg_start = kSplitK ? (sk_idx * kg_per_sk) : 0;
  int kg_end = kSplitK ? (kg_start + kg_per_sk) : num_kg;
  if (kg_end > num_kg) kg_end = num_kg;
  const int my_kgs = kg_end - kg_start;
  if (my_kgs <= 0) return;

  if constexpr (kTma) {
    if (tid == 0) {
      int3_mbar_init(smem.mbar, kN32Stages);
    }
    __syncthreads();
  }

  const int wave = cta_n / n32_ilv;
  const int in_wave = cta_n - wave * n32_ilv;
  const int wave_sz = ((wave + 1) * n32_ilv <= num_cta) ? n32_ilv : (num_cta - wave * n32_ilv);
  const size_t total_bytes = static_cast<size_t>(num_cta) * num_kg * kN32StepWB;
  const uint32_t base_shifted =
      kSym ? __ldg(reinterpret_cast<const uint32_t*>(W_packed + total_bytes) + 1) : 0u;
  const uint8_t* w_base = W_packed +
                          (static_cast<size_t>(wave * n32_ilv) * num_kg + in_wave) * kN32StepWB;
  const size_t w_kg_stride = static_cast<size_t>(wave_sz) * kN32StepWB;

  if (cta_n == 0 && sk_idx == 0) {
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
    if constexpr (kTma) {
      if (tid == 0) {
        const uint64_t pol = int3_pol_evict_first();
        int3_tma_bulk_w(
            smem.w[st], w_base + static_cast<size_t>(kg) * w_kg_stride, kN32StepWB, &smem.mbar[st], pol);
      }
    } else {
      if (tid < kN32StepU4) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
        int3_cp_async16_w(sd, w_base + static_cast<size_t>(kg) * w_kg_stride + tid * 16);
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
  };

  auto load_a_sync = [&](int kg, int st) {
    if constexpr (kM16) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_phys_col8 * 8]) =
          a_valid
              ? __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8))
              : make_uint4(0u, 0u, 0u, 0u);
    } else if (tid < 64) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_phys_col8 * 8]) =
          a_valid
              ? __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8))
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  auto issue_stage = [&](int kg, int st, bool need_fence) {
    if constexpr (kTma) {
      if (tid == 0) {
        if (need_fence) {
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        const uint64_t pol = int3_pol_evict_first();
        int3_tma_bulk_w(
            smem.w[st], w_base + static_cast<size_t>(kg) * w_kg_stride, kN32StepWB, &smem.mbar[st], pol);
      }
    } else {
      if (tid < kN32StepU4) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
        int3_cp_async16_w(sd, w_base + static_cast<size_t>(kg) * w_kg_stride + tid * 16);
      }
    }
    if constexpr (kM16) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int3_cp_async16_a_pred(sd, A + static_cast<size_t>(a_safe_row) * K + kg * 64 + a_col8 * 8, a_src_sz);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int3_cp_async16_a_pred(sd, A + static_cast<size_t>(a_safe_row) * K + kg * 64 + a_col8 * 8, a_src_sz);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kN32Stages - 1;  // 4
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
      issue_stage(kg_start + i, i, false);
    }
  }

  float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int ldm_row = kM16 ? (lane & 15) : (lane & 7);
  const int ldm_col_off = kM16 ? (lane >> 4) : ((lane >> 3) & 1);
  const int trig_i = (!kSplitK && ((pdl_mode & 2) != 0)) ? ((my_kgs > 4) ? (my_kgs - 4) : 0) : -1;

  int st = 0;
  uint32_t parity = 0u;
  for (int i = 0; i < my_kgs; ++i) {
    if (i == trig_i) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    if constexpr (kTma) {
      int3_mbar_wait(&smem.mbar[st], parity);
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
      const int next_st = (st == 0) ? (kN32Stages - 1) : (st - 1);
      issue_stage(kg_start + next_i, next_st, true);
    }

    uint32_t w01_0, w23_0;
    __nv_bfloat162 sc0, zr0;
    int3_load_tile_words<kSym>(&smem.w[st][warp * kTileB], lane, g, base_shifted, w01_0, w23_0, sc0, zr0);

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][ldm_row][pcol8 * 8]));
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        int3_ldmatrix_x4(a0, a1, a2, a3, addr);
      } else {
        int3_ldmatrix_x2(a0, a2, addr);
      }
      uint32_t b0, b1;
      int3_deq4_k16<kSym>(smem.lut64, w01_0, w23_0, s, sc0, zr0, b0, b1);
      if constexpr (kM16) {
        int3_mma_m16n8k16_full(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
      } else {
        int3_mma_m16n8k16(d0, d1, d2, d3, a0, a2, b0, b1);
      }
    }

    if (++st == kN32Stages) {
      st = 0;
      parity ^= 1u;
    }
  }

  const int col0 = cta_n * 32 + warp * 8 + 2 * t;
  if (g < M) {
    if constexpr (!kSplitK) {
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) = int3_bf16x2_u32(d0, d1);
    } else {
      float* p0 = C_fp32 + (static_cast<size_t>(sk_idx) * M + g) * N + col0;
      p0[0] = d0; p0[1] = d1;
    }
  }
  if constexpr (kM16) {
    if (has_hi) {
      const int ghi = g + 8;
      if constexpr (!kSplitK) {
        *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(ghi) * N + col0) = int3_bf16x2_u32(d2, d3);
      } else {
        float* p0 = C_fp32 + (static_cast<size_t>(sk_idx) * M + ghi) * N + col0;
        p0[0] = d2; p0[1] = d3;
      }
    }
  }
}

__global__ void int3_reduce_splitk_kernel(
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
  out_u2.x = int3_bf16x2_u32(s0, s1);
  out_u2.y = int3_bf16x2_u32(s2, s3);
  *reinterpret_cast<uint2*>(C + idx4) = out_u2;
}

__device__ __forceinline__ float int3_swiglu_f32(float gate, float up) {
  return (gate / (1.0f + __expf(-gate))) * up;
}

template <int kMaxM>
struct alignas(128) Int3N32PairSmem {
  alignas(128) uint8_t w[3][2 * kSymN32StepWBytes];     // 3 * 1600 B = 4800 B (exact 64B TMA-aligned!)
  alignas(128) __nv_bfloat16 a[2][2][kMaxM][64];        // 2 stages x BK=128 (4 KiB M<=8, 8 KiB M<=16)
  alignas(8) uint32_t lut64[64];                        // 256 B
  alignas(8) uint64_t mbar[3];                          // 24 B -> 9.2 KiB (M<=8) / 13.3 KiB (M<=16) -> 6 CTAs/SM!
};

// BK=128 paired 32-row/CTA sym NF3-g64 kernel (1600B Hardware-TMA + L2::evict_first, 6 CTAs/SM = 288 >= 208).
template <bool kM16, bool kTma>
__global__ void __launch_bounds__(128, 6) int3_mma_gemm_n32_pair_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K, int n32_ilv, int pdl_mode) {
  constexpr int kTileB = kSymTileBytes;
  constexpr int kN32StepWB = kSymN32StepWBytes;
  constexpr int kPairStepWB = 2 * kSymN32StepWBytes;    // 1600 B = 25 * 64 B
  constexpr int kPairStepU4 = kPairStepWB / 16;         // 100 (< 128)
  __shared__ Int3N32PairSmem<kM16 ? 16 : 8> smem;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_pairs = K >> 7;
  const int num_cta = N >> 5;
  const int cta_n = blockIdx.x;

  int3_init_lut<true, true>(smem.lut64, tid);

  if constexpr (kTma) {
    if (tid == 0) {
      int3_mbar_init(smem.mbar, 3);
    }
    __syncthreads();
  }

  const int wave = cta_n / n32_ilv;
  const int in_wave = cta_n - wave * n32_ilv;
  const int wave_sz = ((wave + 1) * n32_ilv <= num_cta) ? n32_ilv : (num_cta - wave * n32_ilv);
  const size_t total_bytes = static_cast<size_t>(num_cta) * num_pairs * kPairStepWB;
  const uint32_t base_shifted = __ldg(reinterpret_cast<const uint32_t*>(W_packed + total_bytes) + 1);
  const uint8_t* w_base = W_packed +
                          (static_cast<size_t>(wave * n32_ilv) * num_pairs + in_wave) * kPairStepWB;
  const size_t w_pair_stride = static_cast<size_t>(wave_sz) * kPairStepWB;

  if (cta_n == 0) {
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  const int a_row = kM16 ? (tid >> 3) : ((tid & 63) >> 3);
  const int a_col8 = tid & 7;
  const int a_phys_col8 = a_col8 ^ (a_row & 7);
  const bool a_valid = (a_row < M);
  const int a_src_sz = a_valid ? 16 : 0;
  const int a_safe_row = a_valid ? a_row : 0;
  const int a_half_m8 = tid >> 6;

  auto load_a_sync = [&](int p, int st_a) {
    const int k_base = p * 128;
    if constexpr (kM16) {
      *reinterpret_cast<uint4*>(&smem.a[st_a][0][a_row][a_phys_col8 * 8]) =
          a_valid ? __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(a_row) * K + k_base + a_col8 * 8))
                  : make_uint4(0u, 0u, 0u, 0u);
      *reinterpret_cast<uint4*>(&smem.a[st_a][1][a_row][a_phys_col8 * 8]) =
          a_valid ? __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(a_row) * K + k_base + 64 + a_col8 * 8))
                  : make_uint4(0u, 0u, 0u, 0u);
    } else {
      *reinterpret_cast<uint4*>(&smem.a[st_a][a_half_m8][a_row][a_phys_col8 * 8]) =
          a_valid ? __ldg(reinterpret_cast<const uint4*>(
                        A + static_cast<size_t>(a_row) * K + k_base + a_half_m8 * 64 + a_col8 * 8))
                  : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  auto issue_a_async = [&](int p, int st_a) {
    const int k_base = p * 128;
    if constexpr (kM16) {
      const uint32_t sd0 = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st_a][0][a_row][a_phys_col8 * 8]));
      const uint32_t sd1 = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st_a][1][a_row][a_phys_col8 * 8]));
      int3_cp_async16_a_pred(sd0, A + static_cast<size_t>(a_safe_row) * K + k_base + a_col8 * 8, a_src_sz);
      int3_cp_async16_a_pred(sd1, A + static_cast<size_t>(a_safe_row) * K + k_base + 64 + a_col8 * 8, a_src_sz);
    } else {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st_a][a_half_m8][a_row][a_phys_col8 * 8]));
      int3_cp_async16_a_pred(sd, A + static_cast<size_t>(a_safe_row) * K + k_base + a_half_m8 * 64 + a_col8 * 8, a_src_sz);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  auto issue_w_tma = [&](int p, int st_w, bool need_fence) {
    if (tid == 0) {
      if (need_fence) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      }
      const uint64_t pol = int3_pol_evict_first();
      int3_tma_bulk_w(smem.w[st_w], w_base + static_cast<size_t>(p) * w_pair_stride, kPairStepWB, &smem.mbar[st_w], pol);
    }
  };

  auto issue_w_cpasync = [&](int p, int st_w) {
    if (tid < kPairStepU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st_w][tid * 16]));
      int3_cp_async16_w(sd, w_base + static_cast<size_t>(p) * w_pair_stride + tid * 16);
    }
  };

  float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int ldm_row = kM16 ? (lane & 15) : (lane & 7);
  const int ldm_col_off = kM16 ? (lane >> 4) : ((lane >> 3) & 1);
  const int trig_p = ((pdl_mode & 2) != 0) ? ((num_pairs > 2) ? (num_pairs - 2) : 0) : -1;

  if constexpr (kTma) {
    const int w_pre = (num_pairs < 2) ? num_pairs : 2;
    for (int i = 0; i < w_pre; ++i) {
      issue_w_tma(i, i, false);
    }
    if ((pdl_mode & 1) != 0) {
      asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    load_a_sync(0, 0);
    __syncthreads();
    if (num_pairs > 1) {
      issue_a_async(1, 1);
    }

    int st_w = 0;
    uint32_t par_w = 0u;
    for (int p = 0; p < num_pairs; ++p) {
      if (p == trig_p) {
        asm volatile("griddepcontrol.launch_dependents;");
      }
      const int st_a = p & 1;
      int3_mbar_wait(&smem.mbar[st_w], par_w);
      if (p > 0) {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
      }
      __syncthreads();

      if (p + 2 < num_pairs) {
        const int next_st_w = (st_w == 0) ? 2 : (st_w - 1);
        issue_w_tma(p + 2, next_st_w, true);
      }
      if (p >= 1 && p + 1 < num_pairs) {
        issue_a_async(p + 1, st_a ^ 1);
      }

#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t w01_0, w23_0;
        __nv_bfloat162 sc0, zr0;
        int3_load_tile_words<true>(
            &smem.w[st_w][h * kN32StepWB + warp * kTileB], lane, g, base_shifted, w01_0, w23_0, sc0, zr0);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
          const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st_a][h][ldm_row][pcol8 * 8]));
          uint32_t a0, a1, a2, a3;
          if constexpr (kM16) {
            int3_ldmatrix_x4(a0, a1, a2, a3, addr);
          } else {
            int3_ldmatrix_x2(a0, a2, addr);
          }
          uint32_t b0, b1;
          int3_deq4_k16<true>(smem.lut64, w01_0, w23_0, s, sc0, zr0, b0, b1);
          if constexpr (kM16) {
            int3_mma_m16n8k16_full(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
          } else {
            int3_mma_m16n8k16(d0, d1, d2, d3, a0, a2, b0, b1);
          }
        }
      }

      if (++st_w == 3) {
        st_w = 0;
        par_w ^= 1u;
      }
    }
  } else {
    issue_w_cpasync(0, 0);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    if ((pdl_mode & 1) != 0) {
      asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    load_a_sync(0, 0);

    for (int p = 0; p < num_pairs; ++p) {
      if (p == trig_p) {
        asm volatile("griddepcontrol.launch_dependents;");
      }
      const int st = p & 1;
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
      __syncthreads();
      if (p + 1 < num_pairs) {
        issue_w_cpasync(p + 1, st ^ 1);
        issue_a_async(p + 1, st ^ 1);
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t w01_0, w23_0;
        __nv_bfloat162 sc0, zr0;
        int3_load_tile_words<true>(
            &smem.w[st][h * kN32StepWB + warp * kTileB], lane, g, base_shifted, w01_0, w23_0, sc0, zr0);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
          const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][h][ldm_row][pcol8 * 8]));
          uint32_t a0, a1, a2, a3;
          if constexpr (kM16) {
            int3_ldmatrix_x4(a0, a1, a2, a3, addr);
          } else {
            int3_ldmatrix_x2(a0, a2, addr);
          }
          uint32_t b0, b1;
          int3_deq4_k16<true>(smem.lut64, w01_0, w23_0, s, sc0, zr0, b0, b1);
          if constexpr (kM16) {
            int3_mma_m16n8k16_full(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
          } else {
            int3_mma_m16n8k16(d0, d1, d2, d3, a0, a2, b0, b1);
          }
        }
      }
    }
  }

  const int col0 = cta_n * 32 + warp * 8 + 2 * t;
  if (g < M) {
    *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) = int3_bf16x2_u32(d0, d1);
  }
  if constexpr (kM16) {
    if (has_hi) {
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * N + col0) = int3_bf16x2_u32(d2, d3);
    }
  }
}

template <int kMaxM, int kStepWB = kStepWBytes>
struct alignas(128) Int3SwgStageSmem {
  alignas(128) uint8_t w[kSwgStages][kStepWB];          // 5 * 1792 = 8960 B (asym) or 5 * 1600 = 8000 B (sym)
  alignas(128) __nv_bfloat16 a[kSwgStages][kMaxM][64];  // 5 * 1024 = 5120 B (M<=8) or 10240 B (M<=16), XOR-swizzled
  alignas(8) uint32_t lut64[64];                        // 256 B
  alignas(8) uint64_t mbar[kSwgStages];                 // 40 B -> 13.1-14.0 KiB (M<=8) or 18.1-19.0 KiB (M<=16)
};

template <bool kM16, bool kNf3 = true, bool kTma = false, bool kSym = false>
__global__ void __launch_bounds__(128, kM16 ? 4 : 6) int3_mma_gemm_swiglu_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    int M, int I, int K, int swg_ilv, int pdl_mode) {
  constexpr int kTileB = kSym ? kSymTileBytes : kTileBytes;
  constexpr int kStepWB = kSym ? kSymStepWBytes : kStepWBytes;
  constexpr int kStepU4 = kStepWB / 16;
  __shared__ Int3SwgStageSmem<kM16 ? 16 : 8, kStepWB> smem;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;
  const int num_cta = I >> 5;
  const int cta_n = blockIdx.x;

  int3_init_lut<kNf3, kSym>(smem.lut64, tid);

  if constexpr (kTma) {
    if (tid == 0) {
      int3_mbar_init(smem.mbar, kSwgStages);
    }
    __syncthreads();
  }

  const int wave = cta_n / swg_ilv;
  const int in_wave = cta_n - wave * swg_ilv;
  const int wave_sz = ((wave + 1) * swg_ilv <= num_cta) ? swg_ilv : (num_cta - wave * swg_ilv);
  const size_t total_bytes = static_cast<size_t>(num_cta) * num_kg * kStepWB;
  const uint32_t base_shifted =
      kSym ? __ldg(reinterpret_cast<const uint32_t*>(W_packed + total_bytes) + 1) : 0u;
  const uint8_t* w_base = W_packed +
                          (static_cast<size_t>(wave * swg_ilv) * num_kg + in_wave) * kStepWB;
  const size_t w_kg_stride = static_cast<size_t>(wave_sz) * kStepWB;

  if (cta_n == 0) {
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
    if constexpr (kTma) {
      if (tid == 0) {
        const uint64_t pol = int3_pol_evict_first();
        int3_tma_bulk_w(
            smem.w[st], w_base + static_cast<size_t>(kg) * w_kg_stride, kStepWB, &smem.mbar[st], pol);
      }
    } else {
      if (tid < kStepU4) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
        int3_cp_async16_w(sd, w_base + static_cast<size_t>(kg) * w_kg_stride + tid * 16);
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
  };

  auto load_a_sync = [&](int kg, int st) {
    if constexpr (kM16) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_phys_col8 * 8]) =
          a_valid
              ? __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8))
              : make_uint4(0u, 0u, 0u, 0u);
    } else if (tid < 64) {
      *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_phys_col8 * 8]) =
          a_valid
              ? __ldg(reinterpret_cast<const uint4*>(A + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8))
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  auto issue_stage = [&](int kg, int st, bool need_fence) {
    if constexpr (kTma) {
      if (tid == 0) {
        if (need_fence) {
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        const uint64_t pol = int3_pol_evict_first();
        int3_tma_bulk_w(
            smem.w[st], w_base + static_cast<size_t>(kg) * w_kg_stride, kStepWB, &smem.mbar[st], pol);
      }
    } else {
      if (tid < kStepU4) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
        int3_cp_async16_w(sd, w_base + static_cast<size_t>(kg) * w_kg_stride + tid * 16);
      }
    }
    if constexpr (kM16) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int3_cp_async16_a_pred(sd, A + static_cast<size_t>(a_safe_row) * K + kg * 64 + a_col8 * 8, a_src_sz);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int3_cp_async16_a_pred(sd, A + static_cast<size_t>(a_safe_row) * K + kg * 64 + a_col8 * 8, a_src_sz);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  constexpr int kPreload = kSwgStages - 1;  // 4
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
      issue_stage(i, i, false);
    }
  }

  float d0_g = 0.0f, d1_g = 0.0f, d2_g = 0.0f, d3_g = 0.0f;
  float d0_u = 0.0f, d1_u = 0.0f, d2_u = 0.0f, d3_u = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int ldm_row = kM16 ? (lane & 15) : (lane & 7);
  const int ldm_col_off = kM16 ? (lane >> 4) : ((lane >> 3) & 1);
  const int trig_i = ((pdl_mode & 2) != 0) ? ((num_kg > 4) ? (num_kg - 4) : 0) : -1;

  int st = 0;
  uint32_t parity = 0u;
  for (int i = 0; i < num_kg; ++i) {
    if (i == trig_i) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    if constexpr (kTma) {
      int3_mbar_wait(&smem.mbar[st], parity);
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
      const int next_st = (st == 0) ? (kSwgStages - 1) : (st - 1);
      issue_stage(next_i, next_st, true);
    }

    uint32_t w01_g, w23_g, w01_u, w23_u;
    __nv_bfloat162 scg, zrg, scu, zru;
    int3_load_tile_words<kSym>(&smem.w[st][(warp * 2 + 0) * kTileB], lane, g, base_shifted, w01_g, w23_g, scg, zrg);
    int3_load_tile_words<kSym>(&smem.w[st][(warp * 2 + 1) * kTileB], lane, g, base_shifted, w01_u, w23_u, scu, zru);

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][ldm_row][pcol8 * 8]));
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        int3_ldmatrix_x4(a0, a1, a2, a3, addr);
      } else {
        int3_ldmatrix_x2(a0, a2, addr);
      }
      uint32_t b0, b1, c0, c1;
      int3_deq4_k16<kSym>(smem.lut64, w01_g, w23_g, s, scg, zrg, b0, b1);
      int3_deq4_k16<kSym>(smem.lut64, w01_u, w23_u, s, scu, zru, c0, c1);
      if constexpr (kM16) {
        int3_mma_m16n8k16_full(d0_g, d1_g, d2_g, d3_g, a0, a1, a2, a3, b0, b1);
        int3_mma_m16n8k16_full(d0_u, d1_u, d2_u, d3_u, a0, a1, a2, a3, c0, c1);
      } else {
        int3_mma_m16n8k16(d0_g, d1_g, d2_g, d3_g, a0, a2, b0, b1);
        int3_mma_m16n8k16(d0_u, d1_u, d2_u, d3_u, a0, a2, c0, c1);
      }
    }

    if (++st == kSwgStages) {
      st = 0;
      parity ^= 1u;
    }
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

// 4-stage persistent Stream-K W3A16 GEMM + SwiGLU.
template <bool kM16, bool kSwiglu, bool kNf3 = true, bool kTma = false, bool kSym = false>
__global__ void __launch_bounds__(128, kM16 ? 5 : 6) int3_mma_streamk_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ sk_partials,
    int* __restrict__ sk_done,
    int M, int out_cols, int K, int n_blocks, int grid_ctas) {
  constexpr int kTileB = kSym ? kSymTileBytes : kTileBytes;
  constexpr int kStepWB = kSym ? kSymStepWBytes : kStepWBytes;
  constexpr int kStepU4 = kStepWB / 16;
  __shared__ Int3StageSmem<kM16 ? 16 : 8, kStepWB> smem;
  __shared__ int s_is_last;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;
  const int total_iters = n_blocks * num_kg;

  int3_init_lut<kNf3, kSym>(smem.lut64, tid);

  const int cta_id = blockIdx.x;
  const int iter_start = static_cast<int>((static_cast<int64_t>(cta_id) * total_iters) / grid_ctas);
  const int iter_end = static_cast<int>((static_cast<int64_t>(cta_id + 1) * total_iters) / grid_ctas);
  const int my_iters = iter_end - iter_start;
  if (my_iters <= 0) return;

  if constexpr (kTma) {
    if (tid == 0) {
      int3_mbar_init(smem.mbar, kStages);
    }
    __syncthreads();
  }

  const size_t total_bytes = static_cast<size_t>(total_iters) * kStepWB;
  const uint32_t base_shifted =
      kSym ? __ldg(reinterpret_cast<const uint32_t*>(W_packed + total_bytes) + 1) : 0u;

  if (cta_id == 0) {
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
  const int a_phys_col8 = a_col8 ^ (a_row & 7);
  const int a_src_sz = (a_row < M) ? 16 : 0;
  const int a_safe_row = (a_row < M) ? a_row : 0;

  auto issue_next = [&](int st, bool need_fence) {
    if constexpr (kTma) {
      if (tid == 0) {
        if (need_fence) {
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        const uint64_t pol = int3_pol_evict_first();
        int3_tma_bulk_w(
            smem.w[st], W_packed + static_cast<size_t>(pre_iter) * kStepWB, kStepWB, &smem.mbar[st], pol);
      }
    } else {
      if (tid < kStepU4) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
        int3_cp_async16_w(sd, W_packed + static_cast<size_t>(pre_iter) * kStepWB + tid * 16);
      }
    }
    if constexpr (kM16) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int3_cp_async16_a_pred(sd, A + static_cast<size_t>(a_safe_row) * K + pre_kg * 64 + a_col8 * 8, a_src_sz);
    } else if (tid < 64) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][a_row][a_phys_col8 * 8]));
      int3_cp_async16_a_pred(sd, A + static_cast<size_t>(a_safe_row) * K + pre_kg * 64 + a_col8 * 8, a_src_sz);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    ++pre_iter;
    if (++pre_kg == num_kg) pre_kg = 0;
  };

  constexpr int kPreload = kStages - 1;
  const int preload = (my_iters < kPreload) ? my_iters : kPreload;
  for (int i = 0; i < preload; ++i) {
    issue_next(i, false);
  }

  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;
  const bool has_hi = kM16 && ((g + 8) < M);
  const int ldm_row = kM16 ? (lane & 15) : (lane & 7);
  const int ldm_col_off = kM16 ? (lane >> 4) : ((lane >> 3) & 1);
  const int local_c0 = (warp * 2 + 0) * 8 + 2 * t;
  const int local_c1 = (warp * 2 + 1) * 8 + 2 * t;

  int st = 0;
  uint32_t parity = 0u;
  for (int i = 0; i < my_iters; ++i) {
    if constexpr (kTma) {
      int3_mbar_wait(&smem.mbar[st], parity);
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

    if (i + kPreload < my_iters) {
      const int next_st = (st == 0) ? (kStages - 1) : (st - 1);
      issue_next(next_st, true);
    }

    uint32_t w01_0, w23_0, w01_1, w23_1;
    __nv_bfloat162 sc0, zr0, sc1, zr1;
    int3_load_tile_words<kSym>(&smem.w[st][(warp * 2 + 0) * kTileB], lane, g, base_shifted, w01_0, w23_0, sc0, zr0);
    int3_load_tile_words<kSym>(&smem.w[st][(warp * 2 + 1) * kTileB], lane, g, base_shifted, w01_1, w23_1, sc1, zr1);

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int pcol8 = (2 * s + ldm_col_off) ^ (ldm_row & 7);
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][ldm_row][pcol8 * 8]));
      uint32_t a0, a1, a2, a3;
      if constexpr (kM16) {
        int3_ldmatrix_x4(a0, a1, a2, a3, addr);
      } else {
        int3_ldmatrix_x2(a0, a2, addr);
      }
      uint32_t b0, b1, c0, c1;
      int3_deq4_k16<kSym>(smem.lut64, w01_0, w23_0, s, sc0, zr0, b0, b1);
      int3_deq4_k16<kSym>(smem.lut64, w01_1, w23_1, s, sc1, zr1, c0, c1);
      if constexpr (kM16) {
        int3_mma_m16n8k16_full(d0_0, d1_0, d2_0, d3_0, a0, a1, a2, a3, b0, b1);
        int3_mma_m16n8k16_full(d0_1, d1_1, d2_1, d3_1, a0, a1, a2, a3, c0, c1);
      } else {
        int3_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b0, b1);
        int3_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, c0, c1);
      }
    }

    if (++st == kStages) {
      st = 0;
      parity ^= 1u;
    }

    const bool tile_done = (cur_kg + 1 == num_kg) || (i + 1 == my_iters);
    if (tile_done) {
      const int seg_kgs = (cur_kg + 1) - seg_start_kg;
      if (seg_kgs == num_kg) {
        if (g < M) {
          if constexpr (kSwiglu) {
            const int col = cur_n * 32 + warp * 8 + 2 * t;
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * out_cols + col) =
                int3_bf16x2_u32(int3_swiglu_f32(d0_0, d0_1), int3_swiglu_f32(d1_0, d1_1));
          } else {
            const int col0 = cur_n * kRowsPerCTA + local_c0;
            const int col1 = cur_n * kRowsPerCTA + local_c1;
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * out_cols + col0) = int3_bf16x2_u32(d0_0, d1_0);
            *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * out_cols + col1) = int3_bf16x2_u32(d0_1, d1_1);
          }
        }
        if constexpr (kM16) {
          if (has_hi) {
            if constexpr (kSwiglu) {
              const int col = cur_n * 32 + warp * 8 + 2 * t;
              *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * out_cols + col) =
                  int3_bf16x2_u32(int3_swiglu_f32(d2_0, d2_1), int3_swiglu_f32(d3_0, d3_1));
            } else {
              const int col0 = cur_n * kRowsPerCTA + local_c0;
              const int col1 = cur_n * kRowsPerCTA + local_c1;
              *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * out_cols + col0) = int3_bf16x2_u32(d2_0, d3_0);
              *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g + 8) * out_cols + col1) = int3_bf16x2_u32(d2_1, d3_1);
            }
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

bool int3_nf3_enabled() {
  if (int3_bench_dyn_env()) {
    const char* s = std::getenv("MUSE_INT3_NF3");
    return (!s || !*s) ? true : (std::atoi(s) != 0);
  }
  static const bool en = []() -> bool {
    const char* s = std::getenv("MUSE_INT3_NF3");
    if (!s || !*s) return true;
    return std::atoi(s) != 0;
  }();
  return en;
}

int int3_mode() {
  if (int3_bench_dyn_env()) {
    const char* s = std::getenv("MUSE_INT3_MODE");
    return (!s || !*s) ? 1 : std::atoi(s);
  }
  static const int m = []() -> int {
    const char* s = std::getenv("MUSE_INT3_MODE");
    if (!s || !*s) return 1;
    return std::atoi(s);
  }();
  return m;
}

static int int3_n32_wave_ilv(int N, int mode) {
  const int num_cta = N >> 5;
  if (!(mode == 1 && num_cta <= 288)) return 0;
  if (int3_bench_dyn_env()) {
    const char* s = std::getenv("MUSE_INT3_N32_ILV");
    const int v = (!s || !*s) ? num_cta : std::atoi(s);
    return (v > 0 && v <= num_cta) ? v : num_cta;
  }
  static const int cfg = []() -> int {
    const char* s = std::getenv("MUSE_INT3_N32_ILV");
    return (!s || !*s) ? 0 : std::atoi(s);
  }();
  return (cfg > 0 && cfg <= num_cta) ? cfg : num_cta;
}

static int int3_swg_wave_ilv(int I, int mode) {
  if (mode >= 3) return 1;
  const int num_cta = I >> 5;
  if (int3_bench_dyn_env()) {
    const char* s = std::getenv("MUSE_INT3_SWG_ILV");
    int v = (!s || !*s) ? kDefaultSwgWaveIlv : std::atoi(s);
    if (v <= 0) v = kDefaultSwgWaveIlv;
    return (v <= num_cta) ? v : num_cta;
  }
  static const int cfg = []() -> int {
    const char* s = std::getenv("MUSE_INT3_SWG_ILV");
    const int v = (!s || !*s) ? kDefaultSwgWaveIlv : std::atoi(s);
    return (v > 0) ? v : kDefaultSwgWaveIlv;
  }();
  return (cfg <= num_cta) ? cfg : num_cta;
}

template <bool kNf3, bool kTma, bool kSym>
static void int3_set_carveout_impl() {
  cudaFuncSetAttribute(int3_mma_gemm_kernel<false, false, kNf3, kTma, kSym>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_gemm_kernel<false, true, kNf3, kTma, kSym>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_gemm_kernel<true, false, kNf3, kTma, kSym>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_gemm_kernel<true, true, kNf3, kTma, kSym>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_gemm_n32_kernel<false, false, kNf3, kTma, kSym>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_gemm_n32_kernel<false, true, kNf3, kTma, kSym>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_gemm_n32_kernel<true, false, kNf3, kTma, kSym>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_gemm_n32_kernel<true, true, kNf3, kTma, kSym>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_gemm_swiglu_kernel<false, kNf3, kTma, kSym>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_gemm_swiglu_kernel<true, kNf3, kTma, kSym>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_streamk_kernel<false, false, kNf3, kTma, kSym>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_streamk_kernel<true, false, kNf3, kTma, kSym>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_streamk_kernel<false, true, kNf3, kTma, kSym>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(int3_mma_streamk_kernel<true, true, kNf3, kTma, kSym>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

static void int3_ensure_carveout() {
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    int3_set_carveout_impl<false, false, false>();
    int3_set_carveout_impl<true, false, false>();
    int3_set_carveout_impl<false, true, false>();
    int3_set_carveout_impl<true, true, false>();
    int3_set_carveout_impl<true, false, true>();
    int3_set_carveout_impl<true, true, true>();
    cudaFuncSetAttribute(int3_mma_gemm_n32_pair_kernel<false, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(int3_mma_gemm_n32_pair_kernel<true, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(int3_mma_gemm_n32_pair_kernel<false, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(int3_mma_gemm_n32_pair_kernel<true, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  }
}

static uint32_t int3_compute_base_shifted(const torch::Tensor& W, bool h64 = false) {
  const float max_abs = W.abs().max().item<float>();
  const float hi = (max_abs > 1e-12f) ? max_abs : 1e-12f;
  uint32_t fbits = 0u;
  std::memcpy(&fbits, &hi, sizeof(uint32_t));
  const int exp8 = static_cast<int>((fbits >> 23) & 0xFFu);
  int base_exp = exp8 - (h64 ? 14 : 15);
  if (base_exp < 1) base_exp = 1;
  if (base_exp > 239) base_exp = 239;
  return static_cast<uint32_t>(base_exp) << 7;
}

template <bool kNf3, bool kTma, bool kSym>
static torch::Tensor launch_int3_gemm_impl(
    torch::Tensor A, const __nv_bfloat16* A_ptr, torch::Tensor W_packed, int M, int64_t N, int64_t K) {
  auto C = torch::empty({M, N}, A.options());
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n_blocks = static_cast<int>(N / kRowsPerCTA);
  const int n32_blocks = static_cast<int>(N / 32);
  const bool use_m16 = (M > 8);
  const int mode = int3_mode();
  const int pdl = int3_pdl_mode();

  if (mode >= 2) {
    const int total_iters = n_blocks * static_cast<int>(K >> 6);
    const int grid = (288 < total_iters) ? 288 : total_iters;
    float* p_ptr = nullptr;
    int* d_ptr = nullptr;
    ensure_int3_sk_scratch(A.options(), n_blocks, p_ptr, d_ptr);
    if (use_m16) {
      int3_mma_streamk_kernel<true, false, kNf3, kTma, kSym><<<grid, kThreads, 0, st>>>(
          A_ptr,
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid);
    } else {
      int3_mma_streamk_kernel<false, false, kNf3, kTma, kSym><<<grid, kThreads, 0, st>>>(
          A_ptr,
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, d_ptr, M, static_cast<int>(N), static_cast<int>(K), n_blocks, grid);
    }
    return C;
  }

  const int n32_ilv = int3_n32_wave_ilv(static_cast<int>(N), mode);
  if (n32_ilv > 0) {
    if constexpr (kSym) {
      if ((K & 127) == 0) {
        if (use_m16) {
          int3_launch_pdl(
              int3_mma_gemm_n32_pair_kernel<true, kTma>,
              dim3(static_cast<unsigned>(n32_blocks), 1u, 1u), kThreads, st, pdl,
              A_ptr, W_packed.data_ptr<uint8_t>(),
              reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
              M, static_cast<int>(N), static_cast<int>(K), n32_ilv, pdl);
        } else {
          int3_launch_pdl(
              int3_mma_gemm_n32_pair_kernel<false, kTma>,
              dim3(static_cast<unsigned>(n32_blocks), 1u, 1u), kThreads, st, pdl,
              A_ptr, W_packed.data_ptr<uint8_t>(),
              reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
              M, static_cast<int>(N), static_cast<int>(K), n32_ilv, pdl);
        }
        return C;
      }
    }
    if (use_m16) {
      int3_launch_pdl(
          int3_mma_gemm_n32_kernel<false, true, kNf3, kTma, kSym>,
          dim3(static_cast<unsigned>(n32_blocks), 1u, 1u), kThreads, st, pdl,
          A_ptr,
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, n32_ilv, pdl);
    } else {
      int3_launch_pdl(
          int3_mma_gemm_n32_kernel<false, false, kNf3, kTma, kSym>,
          dim3(static_cast<unsigned>(n32_blocks), 1u, 1u), kThreads, st, pdl,
          A_ptr,
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, n32_ilv, pdl);
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
          int3_mma_gemm_kernel<false, true, kNf3, kTma, kSym>,
          dim3(static_cast<unsigned>(n_blocks), 1u, 1u), kThreads, st, pdl,
          A_ptr,
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl);
    } else {
      int3_launch_pdl(
          int3_mma_gemm_kernel<false, false, kNf3, kTma, kSym>,
          dim3(static_cast<unsigned>(n_blocks), 1u, 1u), kThreads, st, pdl,
          A_ptr,
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          static_cast<float*>(nullptr), M, static_cast<int>(N), static_cast<int>(K), 1, pdl);
    }
  } else {
    const int64_t need = static_cast<int64_t>(split_k) * M * N;
    const int64_t alloc = (need > 4 * 16 * 6656) ? need : (4 * 16 * 6656);
    if (!g_partials.defined() || g_partials.device() != A.device() || g_partials_cap < need) {
      g_partials = torch::empty({alloc}, A.options().dtype(torch::kFloat32));
      g_partials_cap = alloc;
    }
    if (use_m16) {
      int3_launch_pdl(
          int3_mma_gemm_kernel<true, true, kNf3, kTma, kSym>,
          dim3(static_cast<unsigned>(n_blocks), static_cast<unsigned>(split_k), 1u), kThreads, st, pdl & 1,
          A_ptr,
          W_packed.data_ptr<uint8_t>(),
          static_cast<__nv_bfloat16*>(nullptr), g_partials.data_ptr<float>(),
          M, static_cast<int>(N), static_cast<int>(K), split_k, pdl & 1);
    } else {
      int3_launch_pdl(
          int3_mma_gemm_kernel<true, false, kNf3, kTma, kSym>,
          dim3(static_cast<unsigned>(n_blocks), static_cast<unsigned>(split_k), 1u), kThreads, st, pdl & 1,
          A_ptr,
          W_packed.data_ptr<uint8_t>(),
          static_cast<__nv_bfloat16*>(nullptr), g_partials.data_ptr<float>(),
          M, static_cast<int>(N), static_cast<int>(K), split_k, pdl & 1);
    }
    const int total4 = (M * static_cast<int>(N)) >> 2;
    int3_reduce_splitk_kernel<<<(total4 + 255) / 256, 256, 0, st>>>(
        g_partials.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, static_cast<int>(N), split_k);
  }
  return C;
}

template <bool kNf3, bool kTma, bool kSym>
static torch::Tensor launch_int3_gemm_swiglu_impl(
    torch::Tensor A, const __nv_bfloat16* A_ptr, torch::Tensor W_packed, int M, int64_t I, int64_t K) {
  auto C = torch::empty({M, I}, A.options());
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n_blocks = static_cast<int>(I / 32);
  const int mode = int3_mode();
  const int pdl = int3_pdl_mode();
  if (mode >= 3) {
    const int sk_target = (mode == 4) ? 576 : 288;
    const int total_iters = n_blocks * static_cast<int>(K >> 6);
    const int grid = (sk_target < total_iters) ? sk_target : total_iters;
    float* p_ptr = nullptr;
    int* d_ptr = nullptr;
    ensure_int3_sk_scratch(A.options(), n_blocks, p_ptr, d_ptr);
    if (M > 8) {
      int3_mma_streamk_kernel<true, true, kNf3, kTma, kSym><<<grid, kThreads, 0, st>>>(
          A_ptr,
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, d_ptr, M, static_cast<int>(I), static_cast<int>(K), n_blocks, grid);
    } else {
      int3_mma_streamk_kernel<false, true, kNf3, kTma, kSym><<<grid, kThreads, 0, st>>>(
          A_ptr,
          W_packed.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
          p_ptr, d_ptr, M, static_cast<int>(I), static_cast<int>(K), n_blocks, grid);
    }
    return C;
  }
  const int swg_ilv = int3_swg_wave_ilv(static_cast<int>(I), mode);
  if (M > 8) {
    int3_launch_pdl(
        int3_mma_gemm_swiglu_kernel<true, kNf3, kTma, kSym>,
        dim3(static_cast<unsigned>(n_blocks), 1u, 1u), kThreads, st, pdl,
        A_ptr,
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, static_cast<int>(I), static_cast<int>(K), swg_ilv, pdl);
  } else {
    int3_launch_pdl(
        int3_mma_gemm_swiglu_kernel<false, kNf3, kTma, kSym>,
        dim3(static_cast<unsigned>(n_blocks), 1u, 1u), kThreads, st, pdl,
        A_ptr,
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, static_cast<int>(I), static_cast<int>(K), swg_ilv, pdl);
  }
  return C;
}

}  // namespace

const __nv_bfloat16* vq4d_rot_a_ptr(torch::Tensor A, int M, int64_t K, cudaStream_t st, int pdl);
torch::Tensor vq4d_gemm(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K);
torch::Tensor vq4d_gemm_swiglu(torch::Tensor A, torch::Tensor W_packed, int64_t I, int64_t K);

torch::Tensor int3_pack(torch::Tensor W, bool mse_opt = true, bool sym = false, bool h64 = false) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kBFloat16 && W.dim() == 2 && W.is_contiguous(),
              "int3_pack expects contiguous CUDA BF16 [N, K]");
  const int N = static_cast<int>(W.size(0));
  const int K = static_cast<int>(W.size(1));
  TORCH_CHECK(N % kRowsPerCTA == 0 && K % 64 == 0, "int3_pack requires N%64==0 and K%64==0");
  dim3 grid(N / 8, K / 64);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n32_ilv = int3_n32_wave_ilv(N, int3_mode());
  if (sym || h64) {
    const int64_t payload_bytes = static_cast<int64_t>(N / 8) * static_cast<int64_t>(K / 64) * kSymTileBytes;
    const int64_t ftr_bytes = h64 ? kSymH64FooterBytes : kSymFooterBytes;
    auto out = torch::empty({payload_bytes + ftr_bytes}, W.options().dtype(torch::kUInt8));
    const uint32_t base_shifted = int3_compute_base_shifted(W, h64);
    if (h64) {
      if (mse_opt) {
        int3_pack_sym_h64_kernel<true><<<grid, 256, 0, st>>>(
            reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(),
            N, K, n32_ilv, base_shifted, static_cast<size_t>(payload_bytes));
      } else {
        int3_pack_sym_h64_kernel<false><<<grid, 256, 0, st>>>(
            reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(),
            N, K, n32_ilv, base_shifted, static_cast<size_t>(payload_bytes));
      }
    } else if (mse_opt) {
      int3_pack_sym_kernel<true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(),
          N, K, n32_ilv, base_shifted, static_cast<size_t>(payload_bytes));
    } else {
      int3_pack_sym_kernel<false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(),
          N, K, n32_ilv, base_shifted, static_cast<size_t>(payload_bytes));
    }
    return out;
  }
  const int64_t total_bytes = static_cast<int64_t>(N / 8) * static_cast<int64_t>(K / 64) * kTileBytes;
  auto out = torch::empty({total_bytes}, W.options().dtype(torch::kUInt8));
  if (int3_nf3_enabled()) {
    if (mse_opt) {
      int3_pack_kernel<true, true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K, n32_ilv);
    } else {
      int3_pack_kernel<false, true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K, n32_ilv);
    }
  } else {
    if (mse_opt) {
      int3_pack_kernel<true, false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K, n32_ilv);
    } else {
      int3_pack_kernel<false, false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), N, K, n32_ilv);
    }
  }
  return out;
}

torch::Tensor int3_gemm(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K) {
  const int64_t num_tiles = static_cast<int64_t>(N / 8) * static_cast<int64_t>(K / 64);
  const int64_t vq9_bytes = num_tiles * 152 + 16;
  const int64_t vq10_bytes = num_tiles * 168 + 16;
  const int64_t wb = W_packed.numel();
  if (wb == vq9_bytes || wb == vq9_bytes + 2048 || wb == vq10_bytes || wb == vq10_bytes + 4096) {
    return vq4d_gemm(A, W_packed, N, K);
  }
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous(),
              "int3_gemm expects contiguous CUDA BF16 [M, K]");
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && A.size(1) == K && N % kRowsPerCTA == 0 && K % 64 == 0,
              "int3_gemm shape mismatch");
  int3_ensure_carveout();
  const bool aligned16 = ((reinterpret_cast<uintptr_t>(W_packed.data_ptr()) & 15u) == 0u);
  const int64_t sym_payload = num_tiles * kSymTileBytes;
  const int64_t sym_bytes = sym_payload + kSymFooterBytes;
  const int64_t sym_h64_bytes = sym_payload + kSymH64FooterBytes;
  if (W_packed.numel() == sym_bytes || W_packed.numel() == sym_h64_bytes) {
    const bool use_pair_tma = ((K & 127) == 0 && int3_n32_wave_ilv(static_cast<int>(N), int3_mode()) > 0)
                                  ? int3_tma_enabled()
                                  : int3_n32_tma_enabled();
    const bool use_tma = use_pair_tma && aligned16;
    const __nv_bfloat16* a_ptr = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr());
    if (W_packed.numel() == sym_h64_bytes) {
      auto st = c10::cuda::getCurrentCUDAStream().stream();
      a_ptr = vq4d_rot_a_ptr(A, M, K, st, int3_pdl_mode());
    }
    return use_tma ? launch_int3_gemm_impl<true, true, true>(A, a_ptr, W_packed, M, N, K)
                   : launch_int3_gemm_impl<true, false, true>(A, a_ptr, W_packed, M, N, K);
  }
  const bool use_tma = int3_n32_tma_enabled() && aligned16;
  const __nv_bfloat16* a_ptr = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr());
  if (int3_nf3_enabled()) {
    return use_tma ? launch_int3_gemm_impl<true, true, false>(A, a_ptr, W_packed, M, N, K)
                   : launch_int3_gemm_impl<true, false, false>(A, a_ptr, W_packed, M, N, K);
  }
  return use_tma ? launch_int3_gemm_impl<false, true, false>(A, a_ptr, W_packed, M, N, K)
                 : launch_int3_gemm_impl<false, false, false>(A, a_ptr, W_packed, M, N, K);
}

torch::Tensor int3_pack_swiglu(torch::Tensor W, bool mse_opt = true, bool sym = false, bool h64 = false) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kBFloat16 && W.dim() == 2 && W.is_contiguous(),
              "int3_pack_swiglu expects contiguous CUDA BF16 [2*I, K]");
  const int N2 = static_cast<int>(W.size(0));
  const int K = static_cast<int>(W.size(1));
  TORCH_CHECK(N2 % 64 == 0 && K % 64 == 0, "int3_pack_swiglu requires (2*I)%64==0 and K%64==0");
  const int I = N2 / 2;
  dim3 grid(N2 / 8, K / 64);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int swg_ilv = int3_swg_wave_ilv(I, int3_mode());
  if (sym || h64) {
    const int64_t payload_bytes = static_cast<int64_t>(N2 / 8) * static_cast<int64_t>(K / 64) * kSymTileBytes;
    const int64_t ftr_bytes = h64 ? kSymH64FooterBytes : kSymFooterBytes;
    auto out = torch::empty({payload_bytes + ftr_bytes}, W.options().dtype(torch::kUInt8));
    const uint32_t base_shifted = int3_compute_base_shifted(W, h64);
    if (h64) {
      if (mse_opt) {
        int3_pack_swiglu_sym_h64_kernel<true><<<grid, 256, 0, st>>>(
            reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(),
            I, K, swg_ilv, base_shifted, static_cast<size_t>(payload_bytes));
      } else {
        int3_pack_swiglu_sym_h64_kernel<false><<<grid, 256, 0, st>>>(
            reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(),
            I, K, swg_ilv, base_shifted, static_cast<size_t>(payload_bytes));
      }
    } else if (mse_opt) {
      int3_pack_swiglu_sym_kernel<true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(),
          I, K, swg_ilv, base_shifted, static_cast<size_t>(payload_bytes));
    } else {
      int3_pack_swiglu_sym_kernel<false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(),
          I, K, swg_ilv, base_shifted, static_cast<size_t>(payload_bytes));
    }
    return out;
  }
  const int64_t total_bytes = static_cast<int64_t>(N2 / 8) * static_cast<int64_t>(K / 64) * kTileBytes;
  auto out = torch::empty({total_bytes}, W.options().dtype(torch::kUInt8));
  if (int3_nf3_enabled()) {
    if (mse_opt) {
      int3_pack_swiglu_kernel<true, true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), I, K, swg_ilv);
    } else {
      int3_pack_swiglu_kernel<false, true><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), I, K, swg_ilv);
    }
  } else {
    if (mse_opt) {
      int3_pack_swiglu_kernel<true, false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), I, K, swg_ilv);
    } else {
      int3_pack_swiglu_kernel<false, false><<<grid, 32, 0, st>>>(
          reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), out.data_ptr<uint8_t>(), I, K, swg_ilv);
    }
  }
  return out;
}

torch::Tensor int3_gemm_swiglu(torch::Tensor A, torch::Tensor W_packed, int64_t I, int64_t K) {
  const int64_t num_tiles = static_cast<int64_t>((2 * I) / 8) * static_cast<int64_t>(K / 64);
  const int64_t vq9_bytes = num_tiles * 152 + 16;
  const int64_t vq10_bytes = num_tiles * 168 + 16;
  const int64_t wb = W_packed.numel();
  if (wb == vq9_bytes || wb == vq9_bytes + 2048 || wb == vq10_bytes || wb == vq10_bytes + 4096) {
    return vq4d_gemm_swiglu(A, W_packed, I, K);
  }
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous(),
              "int3_gemm_swiglu expects contiguous CUDA BF16 [M, K]");
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && A.size(1) == K && I % 32 == 0 && K % 64 == 0,
              "int3_gemm_swiglu shape mismatch");
  int3_ensure_carveout();
  const bool use_tma = int3_tma_enabled() && ((reinterpret_cast<uintptr_t>(W_packed.data_ptr()) & 15u) == 0u);
  const int64_t sym_payload = num_tiles * kSymTileBytes;
  const int64_t sym_bytes = sym_payload + kSymFooterBytes;
  const int64_t sym_h64_bytes = sym_payload + kSymH64FooterBytes;
  if (W_packed.numel() == sym_bytes || W_packed.numel() == sym_h64_bytes) {
    const __nv_bfloat16* a_ptr = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr());
    if (W_packed.numel() == sym_h64_bytes) {
      auto st = c10::cuda::getCurrentCUDAStream().stream();
      a_ptr = vq4d_rot_a_ptr(A, M, K, st, int3_pdl_mode());
    }
    return use_tma ? launch_int3_gemm_swiglu_impl<true, true, true>(A, a_ptr, W_packed, M, I, K)
                   : launch_int3_gemm_swiglu_impl<true, false, true>(A, a_ptr, W_packed, M, I, K);
  }
  const __nv_bfloat16* a_ptr = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr());
  if (int3_nf3_enabled()) {
    return use_tma ? launch_int3_gemm_swiglu_impl<true, true, false>(A, a_ptr, W_packed, M, I, K)
                   : launch_int3_gemm_swiglu_impl<true, false, false>(A, a_ptr, W_packed, M, I, K);
  }
  return use_tma ? launch_int3_gemm_swiglu_impl<false, true, false>(A, a_ptr, W_packed, M, I, K)
                 : launch_int3_gemm_swiglu_impl<false, false, false>(A, a_ptr, W_packed, M, I, K);
}
