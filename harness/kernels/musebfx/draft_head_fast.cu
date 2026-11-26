// SPDX-License-Identifier: Apache-2.0
// draft_head_fast: 2-stage compressed draft LM head for greedy DFlash on GB10 (SM121).
//
// Replaces the 330 MB NVFP4 static draft head GEMM (1.384 ms/cyc at V_static=98,304, K=6,656)
// with a 2-stage exact-rescore pipeline (~0.58 ms/cyc at 1.625 bpw, ~0.70 ms/cyc at 2.00 bpw):
//   Stage 0 (boot, ~15 ms on GPU): Mean-centers and column-whitens the V_static NVFP4 rows directly
//     from raw NVFP4 weights (zero BF16 full-matrix allocation), disperses rows via a coprime
//     bijection across 192 persistent CTAs (eliminating frequent-token clumping), computes weighted
//     least-squares optimal per-row scale & threshold in logit space, and packs into wave-interleaved
//     8x64 MMA tiles:
//       - mode=158 (1.625 bpw, ~133 MB): ternary {-1, 0, +1}, 16 trits in 26 bits/thread (104 B/tile),
//         decoded via 2x exact umulhi(/81) + 81-entry 648 B SMEM BF16x2 LUT.
//       - mode=2   (2.000 bpw, ~164 MB): odd-integer 2-bit {-3, -1, +1, +3}, 16 weights in 32 bits/thread
//         (128 B/tile), decoded via 0-LUT Marlin-interleaved LOP3 + single HFMA2(t, 2.0, -19.0).
//   Stage 1 (inference, 192 persistent CTAs, 4 CTAs/SM): 6-stage cp.async.cg.L2::256B + ldmatrix.x2
//     + m16n8k16 BF16 Tensor-Core GEMM with fused in-CTA top-8 shortlist extraction (never writes
//     [M, V_static] logits to DRAM; outputs only 72 KB of per-CTA top-8 into L2 cache).
//   Stage 1b (inference, M blocks, ~3 us): Exact 3-pass 8-bit MSB radix select reducing 192*8=1536
//     candidates per row down to top K_cand (128..256) unique target token IDs in L2.
//   Stage 2 (inference, 16*M blocks + M reduce blocks, ~25 us): Exact NVFP4 rescore of all K_cand
//     static shortlist candidates PLUS all dynamic prompt-tail OOV rows (`tail_ids[0..tail_count-1]`)
//     against raw `(full_raw_w, full_raw_s, full_global_scale)` with exact lowest-original-ID
//     tie-breaking and full top-1 / top-2 / margin / tree top-4/6/8 + tree_vals contract.
//
// 100% CUDA-graph safe (fixed-address per-device scratch, zero dynamic allocations, zero syncs).
// Target lm_head (202,048 rows) is completely untouched.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <stdint.h>
#include <tuple>

namespace {

constexpr int kMaxM = 8;
constexpr int kWarpsPerCTA = 4;
constexpr int kThreads = kWarpsPerCTA * 32;                    // 128
constexpr int kNTilesPerWarp = 2;                              // 16 N-rows/warp -> 2x A reuse
constexpr int kNTilesPerCTA = kWarpsPerCTA * kNTilesPerWarp;   // 8
constexpr int kRowsPerCTA = kNTilesPerCTA * 8;                 // 64
constexpr int kWaveCTAs = 192;                                 // 4 CTAs/SM * 48 SMs on GB10
constexpr int kStages = 6;                                     // 6-stage cp.async pipeline (<=15 KiB SMEM)
constexpr int kCtaTopK = 8;                                    // per-CTA shortlist per row m
constexpr int kMaxCand = 256;                                  // max Stage-2 shortlist per row m
constexpr int kS2CTAsPerRow = 16;                              // 96 CTAs at M=6 (2 CTAs/SM)
constexpr int kS2Warps = 8;
constexpr int kS2MaxGroups = 512;                              // K <= 8192 (16 KiB SMEM)

constexpr int kTileBytes158 = 104;                             // 26 * uint32_t (32 * 26 bits)
constexpr int kStepWBytes158 = kNTilesPerCTA * kTileBytes158;  // 832 B = 52 * uint4
constexpr int kStepWU4_158 = kStepWBytes158 / 16;              // 52

constexpr int kTileBytes2 = 128;                               // 32 * uint32_t (32 * 32 bits)
constexpr int kStepWBytes2 = kNTilesPerCTA * kTileBytes2;      // 1024 B = 64 * uint4
constexpr int kStepWU4_2 = kStepWBytes2 / 16;                  // 64

int dhf_pdl_mode() {
  static const int mode = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_PDL");
    if (!s || !*s) return 3;
    return (std::atoi(s) != 0) ? 3 : 0;
  }();
  return mode;
}

template <typename... KArgs, typename... Args>
static inline void dhf_launch_pdl(
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
  TORCH_CHECK(err == cudaSuccess, "draft_head_fast kernel launch failed: ", cudaGetErrorString(err));
}

__device__ __host__ __forceinline__ size_t dhf_wave_step_idx(
    int cta_n, int kg, int n_blocks, int num_kg) {
  const int wave = cta_n / kWaveCTAs;
  const int slot = cta_n - wave * kWaveCTAs;
  const int wave_base = wave * kWaveCTAs;
  const int w_size = (n_blocks - wave_base < kWaveCTAs) ? (n_blocks - wave_base) : kWaveCTAs;
  return static_cast<size_t>(wave_base) * num_kg + static_cast<size_t>(kg) * w_size + slot;
}

__device__ __forceinline__ float dhf_e2m1_x2_to_f32(uint32_t nib) {
  const uint32_t mag_u = (0xC8643210u >> ((nib & 7u) << 2)) & 0xFu;
  const float mag = __uint_as_float(0x41800000u | (mag_u << 19)) - 16.0f;
  return __uint_as_float(__float_as_uint(mag) ^ ((nib & 8u) << 28));
}

__device__ __forceinline__ float dhf_fp8_e4m3_to_f32(uint8_t bits) {
  __nv_fp8_e4m3 v;
  v.__x = bits;
  return static_cast<float>(v);
}

__device__ __forceinline__ uint32_t dhf_bf16x2_u32(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void dhf_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void dhf_cp_async16_cg(uint32_t sd, const void* gs) {
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ void dhf_cp_async16_ca_z(uint32_t sd, const void* gs, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(sd), "l"(gs), "r"(src_bytes));
}

__device__ __forceinline__ void dhf_ldmatrix_x2(uint32_t& r0, uint32_t& r1, uint32_t smem_addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(r0), "=r"(r1)
      : "r"(smem_addr));
}

__device__ __forceinline__ bool dhf_better(float v1, int i1, float v2, int i2) {
  return (v1 > v2) || (v1 == v2 && i1 < i2);
}

__device__ __forceinline__ void dhf_top2_insert(
    float& v1, int& i1, float& v2, int& i2, float cand_v, int cand_i) {
  if (cand_i == i1 || cand_i == i2) return;
  if (dhf_better(cand_v, cand_i, v1, i1)) {
    v2 = v1; i2 = i1; v1 = cand_v; i1 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v2, i2)) {
    v2 = cand_v; i2 = cand_i;
  }
}

__device__ __forceinline__ void dhf_top4_insert(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float cand_v, int cand_i) {
  if (cand_i == i1 || cand_i == i2 || cand_i == i3 || cand_i == i4) return;
  if (dhf_better(cand_v, cand_i, v1, i1)) {
    v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = cand_v; i1 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v2, i2)) {
    v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = cand_v; i2 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v3, i3)) {
    v4 = v3; i4 = i3; v3 = cand_v; i3 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v4, i4)) {
    v4 = cand_v; i4 = cand_i;
  }
}

__device__ __forceinline__ void dhf_top6_insert(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float& v5, int& i5, float& v6, int& i6,
    float cand_v, int cand_i) {
  if (cand_i == i1 || cand_i == i2 || cand_i == i3 ||
      cand_i == i4 || cand_i == i5 || cand_i == i6) return;
  if (dhf_better(cand_v, cand_i, v1, i1)) {
    v6 = v5; i6 = i5; v5 = v4; i5 = i4; v4 = v3; i4 = i3;
    v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = cand_v; i1 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v2, i2)) {
    v6 = v5; i6 = i5; v5 = v4; i5 = i4; v4 = v3; i4 = i3;
    v3 = v2; i3 = i2; v2 = cand_v; i2 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v3, i3)) {
    v6 = v5; i6 = i5; v5 = v4; i5 = i4; v4 = v3; i4 = i3; v3 = cand_v; i3 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v4, i4)) {
    v6 = v5; i6 = i5; v5 = v4; i5 = i4; v4 = cand_v; i4 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v5, i5)) {
    v6 = v5; i6 = i5; v5 = cand_v; i5 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v6, i6)) {
    v6 = cand_v; i6 = cand_i;
  }
}

__device__ __forceinline__ void dhf_top8_insert(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float& v5, int& i5, float& v6, int& i6,
    float& v7, int& i7, float& v8, int& i8,
    float cand_v, int cand_i) {
  if (cand_i == i1 || cand_i == i2 || cand_i == i3 || cand_i == i4 ||
      cand_i == i5 || cand_i == i6 || cand_i == i7 || cand_i == i8) return;
  if (dhf_better(cand_v, cand_i, v1, i1)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4;
    v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = cand_v; i1 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v2, i2)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4;
    v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = cand_v; i2 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v3, i3)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4;
    v4 = v3; i4 = i3; v3 = cand_v; i3 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v4, i4)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4;
    v4 = cand_v; i4 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v5, i5)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = cand_v; i5 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v6, i6)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = cand_v; i6 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v7, i7)) {
    v8 = v7; i8 = i7; v7 = cand_v; i7 = cand_i;
  } else if (dhf_better(cand_v, cand_i, v8, i8)) {
    v8 = cand_v; i8 = cand_i;
  }
}

__device__ __forceinline__ void dhf_top8_merge(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float& v5, int& i5, float& v6, int& i6,
    float& v7, int& i7, float& v8, int& i8,
    float ov1, int oi1, float ov2, int oi2,
    float ov3, int oi3, float ov4, int oi4,
    float ov5, int oi5, float ov6, int oi6,
    float ov7, int oi7, float ov8, int oi8) {
  if (ov1 >= v8) {
    dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov1, oi1);
    if (ov2 >= v8) {
      dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov2, oi2);
      if (ov3 >= v8) {
        dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov3, oi3);
        if (ov4 >= v8) {
          dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov4, oi4);
          if (ov5 >= v8) {
            dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov5, oi5);
            if (ov6 >= v8) {
              dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov6, oi6);
              if (ov7 >= v8) {
                dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov7, oi7);
                if (ov8 >= v8) {
                  dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov8, oi8);
                }
              }
            }
          }
        }
      }
    }
  }
}

__device__ __forceinline__ void dhf_top6_merge(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float& v5, int& i5, float& v6, int& i6,
    float ov1, int oi1, float ov2, int oi2,
    float ov3, int oi3, float ov4, int oi4,
    float ov5, int oi5, float ov6, int oi6) {
  if (ov1 >= v6) {
    dhf_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov1, oi1);
    if (ov2 >= v6) {
      dhf_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov2, oi2);
      if (ov3 >= v6) {
        dhf_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov3, oi3);
        if (ov4 >= v6) {
          dhf_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov4, oi4);
          if (ov5 >= v6) {
            dhf_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov5, oi5);
            if (ov6 >= v6) {
              dhf_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov6, oi6);
            }
          }
        }
      }
    }
  }
}

__device__ __forceinline__ void dhf_top4_merge(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float ov1, int oi1, float ov2, int oi2,
    float ov3, int oi3, float ov4, int oi4) {
  if (ov1 >= v4) {
    dhf_top4_insert(v1, i1, v2, i2, v3, i3, v4, i4, ov1, oi1);
    if (ov2 >= v4) {
      dhf_top4_insert(v1, i1, v2, i2, v3, i3, v4, i4, ov2, oi2);
      if (ov3 >= v4) {
        dhf_top4_insert(v1, i1, v2, i2, v3, i3, v4, i4, ov3, oi3);
        if (ov4 >= v4) {
          dhf_top4_insert(v1, i1, v2, i2, v3, i3, v4, i4, ov4, oi4);
        }
      }
    }
  }
}

__device__ __forceinline__ float dhf_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  return v;
}

// ============================================================================
// Boot-time GPU Packer Kernels (~15 ms total for 98,304 x 6,656)
// ============================================================================

// Pass 1: Compute per-column mean and std across all V_static rows directly from raw NVFP4.
// Grid: num_g = K / 16 blocks, Block: 256 threads (16 threads per group column? No: 256 threads
// stride over V_static rows, each thread accumulates 16 columns in registers -> 1 block reduction).
__global__ void dhf_col_stats_kernel(
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    const float* __restrict__ global_scale,
    const int32_t* __restrict__ packed_tok_ids,
    float* __restrict__ col_mean,
    float* __restrict__ col_std,
    int V_static, int K) {
  const int g = blockIdx.x;  // 0 .. K/16 - 1
  const int tid = threadIdx.x;
  const int K2 = K >> 1;
  const int num_g = K >> 4;
  const float gs_half = __ldg(global_scale) * 0.5f;

  float s1[16];
  float s2[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    s1[j] = 0.0f;
    s2[j] = 0.0f;
  }

  for (int r = tid; r < V_static; r += blockDim.x) {
    const int tok = __ldg(&packed_tok_ids[r]);
    const float sc = dhf_fp8_e4m3_to_f32(__ldg(&raw_s[static_cast<size_t>(tok) * num_g + g])) * gs_half;
    const uint64_t w8 = __ldg(reinterpret_cast<const uint64_t*>(
        raw_w + static_cast<size_t>(tok) * K2 + g * 8));
    uint32_t wlo = static_cast<uint32_t>(w8);
    uint32_t whi = static_cast<uint32_t>(w8 >> 32);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v0 = dhf_e2m1_x2_to_f32(wlo & 0xFu) * sc;
      const float v1 = dhf_e2m1_x2_to_f32((wlo >> 4) & 0xFu) * sc;
      s1[2 * j + 0] += v0; s2[2 * j + 0] = __fmaf_rn(v0, v0, s2[2 * j + 0]);
      s1[2 * j + 1] += v1; s2[2 * j + 1] = __fmaf_rn(v1, v1, s2[2 * j + 1]);
      wlo >>= 8;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v0 = dhf_e2m1_x2_to_f32(whi & 0xFu) * sc;
      const float v1 = dhf_e2m1_x2_to_f32((whi >> 4) & 0xFu) * sc;
      s1[8 + 2 * j + 0] += v0; s2[8 + 2 * j + 0] = __fmaf_rn(v0, v0, s2[8 + 2 * j + 0]);
      s1[8 + 2 * j + 1] += v1; s2[8 + 2 * j + 1] = __fmaf_rn(v1, v1, s2[8 + 2 * j + 1]);
      whi >>= 8;
    }
  }

  __shared__ float sm_s1[8][16];
  __shared__ float sm_s2[8][16];
  const int warp = tid >> 5;
  const int lane = tid & 31;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const float w1 = dhf_warp_sum(s1[j]);
    const float w2 = dhf_warp_sum(s2[j]);
    if (lane == 0) {
      sm_s1[warp][j] = w1;
      sm_s2[warp][j] = w2;
    }
  }
  __syncthreads();

  if (tid < 16) {
    float tot1 = 0.0f;
    float tot2 = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      tot1 += sm_s1[w][tid];
      tot2 += sm_s2[w][tid];
    }
    const float inv_v = 1.0f / static_cast<float>(V_static);
    const float mu = tot1 * inv_v;
    const float var = fmaxf(tot2 * inv_v - mu * mu, 1e-12f);
    col_mean[g * 16 + tid] = mu;
    col_std[g * 16 + tid] = sqrtf(var);
  }
}

// Pass 2: Compute weighted-LS optimal per-row scale & threshold in centered logit space.
// Grid: V_static / 8 blocks, Block: 256 threads (1 warp per packed row).
template <int kMode>
__global__ void dhf_row_scales_kernel(
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    const float* __restrict__ global_scale,
    const int32_t* __restrict__ packed_tok_ids,
    const float* __restrict__ col_mean,
    const float* __restrict__ col_std,
    float* __restrict__ row_scales,
    float* __restrict__ row_thresh,
    int V_static, int K) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp_id >= V_static) return;

  const int tok = __ldg(&packed_tok_ids[warp_id]);
  const int K2 = K >> 1;
  const int num_g = K >> 4;
  const float gs_half = __ldg(global_scale) * 0.5f;
  const uint8_t* w_row = raw_w + static_cast<size_t>(tok) * K2;
  const uint8_t* s_row = raw_s + static_cast<size_t>(tok) * num_g;

  // Step 1: Compute RMS of whitened centered row u[k] = (w[k] - mu[k]) / cs[k].
  float u2 = 0.0f;
  for (int g = lane; g < num_g; g += 32) {
    const float sc = dhf_fp8_e4m3_to_f32(__ldg(&s_row[g])) * gs_half;
    const uint64_t w8 = __ldg(reinterpret_cast<const uint64_t*>(w_row + g * 8));
    uint32_t wlo = static_cast<uint32_t>(w8);
    uint32_t whi = static_cast<uint32_t>(w8 >> 32);
    const int kb = g << 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v0 = dhf_e2m1_x2_to_f32(wlo & 0xFu) * sc;
      const float v1 = dhf_e2m1_x2_to_f32((wlo >> 4) & 0xFu) * sc;
      const float u0 = (v0 - __ldg(&col_mean[kb + 2 * j + 0])) / __ldg(&col_std[kb + 2 * j + 0]);
      const float u1 = (v1 - __ldg(&col_mean[kb + 2 * j + 1])) / __ldg(&col_std[kb + 2 * j + 1]);
      u2 = __fmaf_rn(u0, u0, u2);
      u2 = __fmaf_rn(u1, u1, u2);
      wlo >>= 8;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v0 = dhf_e2m1_x2_to_f32(whi & 0xFu) * sc;
      const float v1 = dhf_e2m1_x2_to_f32((whi >> 4) & 0xFu) * sc;
      const float u0 = (v0 - __ldg(&col_mean[kb + 8 + 2 * j + 0])) / __ldg(&col_std[kb + 8 + 2 * j + 0]);
      const float u1 = (v1 - __ldg(&col_mean[kb + 8 + 2 * j + 1])) / __ldg(&col_std[kb + 8 + 2 * j + 1]);
      u2 = __fmaf_rn(u0, u0, u2);
      u2 = __fmaf_rn(u1, u1, u2);
      whi >>= 8;
    }
  }
  const float std_u = sqrtf(fmaxf(dhf_warp_sum(u2) / static_cast<float>(K), 1e-12f));
  float thresh = (kMode == 158) ? (0.6120f * std_u) : (0.9957f * std_u);
  float alpha = std_u;

  // Step 2: 2-iteration weighted least-squares refinement in true centered logit space.
#pragma unroll
  for (int it = 0; it < 2; ++it) {
    float num = 0.0f;
    float den = 0.0f;
    for (int g = lane; g < num_g; g += 32) {
      const float sc = dhf_fp8_e4m3_to_f32(__ldg(&s_row[g])) * gs_half;
      const uint64_t w8 = __ldg(reinterpret_cast<const uint64_t*>(w_row + g * 8));
      uint32_t wlo = static_cast<uint32_t>(w8);
      uint32_t whi = static_cast<uint32_t>(w8 >> 32);
      const int kb = g << 4;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t ww = (h == 0) ? wlo : whi;
        const int ko = kb + (h << 3);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float v0 = dhf_e2m1_x2_to_f32(ww & 0xFu) * sc;
          const float v1 = dhf_e2m1_x2_to_f32((ww >> 4) & 0xFu) * sc;
          const float c0 = __ldg(&col_std[ko + 2 * j + 0]);
          const float c1 = __ldg(&col_std[ko + 2 * j + 1]);
          const float wc0 = v0 - __ldg(&col_mean[ko + 2 * j + 0]);
          const float wc1 = v1 - __ldg(&col_mean[ko + 2 * j + 1]);
          const float u0 = wc0 / c0;
          const float u1 = wc1 / c1;
          float q0, q1;
          if constexpr (kMode == 158) {
            q0 = (u0 > thresh) ? 1.0f : ((u0 < -thresh) ? -1.0f : 0.0f);
            q1 = (u1 > thresh) ? 1.0f : ((u1 < -thresh) ? -1.0f : 0.0f);
          } else {
            q0 = (u0 >= thresh) ? 3.0f : ((u0 >= 0.0f) ? 1.0f : ((u0 > -thresh) ? -1.0f : -3.0f));
            q1 = (u1 >= thresh) ? 3.0f : ((u1 >= 0.0f) ? 1.0f : ((u1 > -thresh) ? -1.0f : -3.0f));
          }
          num = __fmaf_rn(wc0 * c0, q0, num);
          num = __fmaf_rn(wc1 * c1, q1, num);
          den = __fmaf_rn(c0 * c0, q0 * q0, den);
          den = __fmaf_rn(c1 * c1, q1 * q1, den);
          ww >>= 8;
        }
      }
    }
    const float g_num = dhf_warp_sum(num);
    const float g_den = dhf_warp_sum(den);
    if (g_den > 1e-20f) {
      alpha = g_num / g_den;
      if (it == 0) {
        thresh = (kMode == 158) ? (0.5f * fabsf(alpha)) : (2.0f * fabsf(alpha));
      }
    }
  }

  if (lane == 0) {
    row_scales[warp_id] = alpha;
    row_thresh[warp_id] = thresh;
  }
}

// Helper: decode single NVFP4 element at (tok, k) directly from raw_w/raw_s.
__device__ __forceinline__ float dhf_decode_nvfp4_elem(
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    float gs_half, int tok, int k, int K2, int num_g) {
  const int g = k >> 4;
  const float sc = dhf_fp8_e4m3_to_f32(__ldg(&raw_s[static_cast<size_t>(tok) * num_g + g])) * gs_half;
  const uint8_t byte = __ldg(&raw_w[static_cast<size_t>(tok) * K2 + (k >> 1)]);
  const uint32_t nib = (k & 1) ? (byte >> 4) : (byte & 0xFu);
  return dhf_e2m1_x2_to_f32(nib) * sc;
}

// Pass 3: Pack wave-interleaved 8x64 tiles for mode=158 (104 B/tile) or mode=2 (128 B/tile).
// Grid: (V_static / 8, K / 64), Block: 32 threads (1 warp per 8x64 tile).
template <int kMode>
__global__ void dhf_pack_tiles_kernel(
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    const float* __restrict__ global_scale,
    const int32_t* __restrict__ packed_tok_ids,
    const float* __restrict__ col_mean,
    const float* __restrict__ col_std,
    const float* __restrict__ row_thresh,
    uint8_t* __restrict__ out,
    int V_static, int K) {
  const int n_tile = blockIdx.x;  // 0 .. V_static/8 - 1
  const int kg = blockIdx.y;      // 0 .. K/64 - 1
  const int lane = threadIdx.x;   // 0 .. 31
  const int r = lane >> 2;        // 0 .. 7
  const int t = lane & 3;         // 0 .. 3
  const int pack_row = n_tile * 8 + r;
  const int tok = __ldg(&packed_tok_ids[pack_row]);
  const float thresh = __ldg(&row_thresh[pack_row]);
  const float gs_half = __ldg(global_scale) * 0.5f;
  const int K2 = K >> 1;
  const int num_g = K >> 4;
  const int k_base = kg * 64;

  const int cta_n = n_tile >> 3;
  const int sub_tile = n_tile & 7;
  const int n_blocks = V_static / kRowsPerCTA;
  const int num_kg = K >> 6;
  const size_t step_slot = dhf_wave_step_idx(cta_n, kg, n_blocks, num_kg);
  const size_t tile_idx = step_slot * kNTilesPerCTA + sub_tile;

  if constexpr (kMode == 158) {
    uint32_t code[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int kb = k_base + s * 16 + 2 * t;
      const int ks[4] = {kb + 0, kb + 1, kb + 8, kb + 9};
      uint32_t tr[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int kk = ks[j];
        const float w = dhf_decode_nvfp4_elem(raw_w, raw_s, gs_half, tok, kk, K2, num_g);
        const float u = (w - __ldg(&col_mean[kk])) / __ldg(&col_std[kk]);
        tr[j] = (u > thresh) ? 2u : ((u < -thresh) ? 0u : 1u);
      }
      code[s] = tr[0] + 3u * tr[1] + 9u * tr[2] + 27u * tr[3];
    }
    const uint32_t c_lo = code[0] + 81u * code[1];
    const uint32_t c_hi = code[2] + 81u * code[3];
    const uint32_t c26 = c_lo | (c_hi << 13);

    // Pack 32 x 26-bit values into 26 x uint32_t (104 bytes) via full-warp __shfl_sync.
    const int w_safe = (lane < 26) ? lane : 0;
    const int bit_start = w_safe * 32;
    const int l0 = bit_start / 26;
    const int o0 = bit_start - 26 * l0;
    const int n0 = 26 - o0;  // 1..26
    const int l1 = (l0 + 1 < 32) ? (l0 + 1) : 31;
    const int l2 = (l0 + 2 < 32) ? (l0 + 2) : 31;
    const uint32_t v0 = __shfl_sync(0xffffffffu, c26, l0);
    const uint32_t v1 = __shfl_sync(0xffffffffu, c26, l1);
    const uint32_t v2 = __shfl_sync(0xffffffffu, c26, l2);
    uint32_t word = (v0 >> o0) | (v1 << n0);
    if (n0 < 6) {
      word |= (v2 << (n0 + 26));
    }
    if (lane < 26) {
      uint32_t* dst = reinterpret_cast<uint32_t*>(out + tile_idx * kTileBytes158);
      dst[lane] = word;
    }
  } else {
    uint32_t q32 = 0u;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int kb = k_base + s * 16 + 2 * t;
      const int ks[4] = {kb + 0, kb + 1, kb + 8, kb + 9};
      uint32_t q[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int kk = ks[j];
        const float w = dhf_decode_nvfp4_elem(raw_w, raw_s, gs_half, tok, kk, K2, num_g);
        const float u = (w - __ldg(&col_mean[kk])) / __ldg(&col_std[kk]);
        q[j] = (u >= thresh) ? 3u : ((u >= 0.0f) ? 2u : ((u > -thresh) ? 1u : 0u));
      }
      // Marlin-interleaved 2-bit layout:
      //   q[0] at [4*s+1 : 4*s], q[2] at [4*s+3 : 4*s+2] (low 16)
      //   q[1] at [16+4*s+1 : 16+4*s], q[3] at [16+4*s+3 : 16+4*s+2] (high 16)
      q32 |= (q[0] << (4 * s)) | (q[2] << (4 * s + 2)) |
             (q[1] << (16 + 4 * s)) | (q[3] << (16 + 4 * s + 2));
    }
    uint32_t* dst = reinterpret_cast<uint32_t*>(out + tile_idx * kTileBytes2);
    dst[lane] = q32;
  }
}

// ============================================================================
// Inference Kernels (100% CUDA-Graph Safe, PDL-Chained)
// ============================================================================

// Kernel 0: Scale input A [M, K] by col_scales [K] -> A_scaled [M, K] (L2-hot, ~2 us).
__global__ void dhf_prep_a_kernel(
    const __nv_bfloat16* __restrict__ A,
    const float* __restrict__ col_scales,
    __nv_bfloat16* __restrict__ A_scaled,
    int M, int K, int pdl_mode) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const __nv_bfloat16* a_in = A + static_cast<size_t>(m) * K;
  __nv_bfloat16* a_out = A_scaled + static_cast<size_t>(m) * K;
  const int k2 = K >> 1;
  for (int i = threadIdx.x; i < k2; i += blockDim.x) {
    const __nv_bfloat162 av = reinterpret_cast<const __nv_bfloat162*>(a_in)[i];
    const float2 af = __bfloat1622float2(av);
    const float2 cs = reinterpret_cast<const float2*>(col_scales)[i];
    reinterpret_cast<__nv_bfloat162*>(a_out)[i] = __floats2bfloat162_rn(af.x * cs.x, af.y * cs.y);
  }
  if ((pdl_mode & 2) != 0 && threadIdx.x == 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
}

// Native BF16x2 2-bit odd-integer dequant: q in {0,1,2,3} -> {-3,-1,+1,+3} via 1 LOP3 + 1 HFMA2.
template <int kStep>
__device__ __forceinline__ void dhf_deq2_k16(uint32_t q32, uint32_t& b0, uint32_t& b1) {
  uint32_t t0, t1;
  if constexpr (kStep == 0) {
    t0 = ((q32 << 4) & 0x00300030u) | 0x41004100u;
    t1 = ((q32 << 2) & 0x00300030u) | 0x41004100u;
  } else if constexpr (kStep == 1) {
    t0 = (q32 & 0x00300030u) | 0x41004100u;
    t1 = ((q32 >> 2) & 0x00300030u) | 0x41004100u;
  } else if constexpr (kStep == 2) {
    t0 = ((q32 >> 4) & 0x00300030u) | 0x41004100u;
    t1 = ((q32 >> 6) & 0x00300030u) | 0x41004100u;
  } else {
    t0 = ((q32 >> 8) & 0x00300030u) | 0x41004100u;
    t1 = ((q32 >> 10) & 0x00300030u) | 0x41004100u;
  }
  const uint32_t two_u = 0x40004000u;    // bf16x2(2.0, 2.0)
  const uint32_t neg19_u = 0xC198C198u;  // bf16x2(-19.0, -19.0)
  const __nv_bfloat162 two = *reinterpret_cast<const __nv_bfloat162*>(&two_u);
  const __nv_bfloat162 neg19 = *reinterpret_cast<const __nv_bfloat162*>(&neg19_u);
  const __nv_bfloat162 w0 = __hfma2(*reinterpret_cast<const __nv_bfloat162*>(&t0), two, neg19);
  const __nv_bfloat162 w1 = __hfma2(*reinterpret_cast<const __nv_bfloat162*>(&t1), two, neg19);
  b0 = *reinterpret_cast<const uint32_t*>(&w0);
  b1 = *reinterpret_cast<const uint32_t*>(&w1);
}

template <int kStepWBytes>
struct alignas(16) DhfStageSmem {
  uint8_t w[kStages][kStepWBytes];
  __nv_bfloat16 a[kStages][kMaxM][64];
  uint2 lut[81];
  float s_scores[kMaxM][kRowsPerCTA];
  int32_t s_toks[kRowsPerCTA];
  float s_topv[kMaxM][kCtaTopK];
  int32_t s_topi[kMaxM][kCtaTopK];
};

// Kernel 1: Persistent 192-CTA fused Stage-1 compressed MMA + in-CTA top-8 shortlist extraction.
template <int kMode>
__global__ void __launch_bounds__(128, 4) dhf_stage1_kernel(
    const __nv_bfloat16* __restrict__ A_scaled,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ row_scales,
    const int32_t* __restrict__ packed_tok_ids,
    float* __restrict__ cta_vals,
    int32_t* __restrict__ cta_ids,
    int M, int V_static, int K, int pdl_mode) {
  constexpr int kTileBytes = (kMode == 158) ? kTileBytes158 : kTileBytes2;
  constexpr int kStepWBytes = (kMode == 158) ? kStepWBytes158 : kStepWBytes2;
  constexpr int kStepWU4 = (kMode == 158) ? kStepWU4_158 : kStepWU4_2;
  __shared__ DhfStageSmem<kStepWBytes> smem;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;
  const int n_blocks = V_static / kRowsPerCTA;
  const int bid = blockIdx.x;
  const int grid_ctas = gridDim.x;

  // Populate 81-entry 4-trit BF16x2 LUT in SMEM (mode 158 only) & init persistent top-8.
  if constexpr (kMode == 158) {
    if (tid < 81) {
      const int c = tid;
      const int t0 = c % 3;
      const int t1 = (c / 3) % 3;
      const int t2 = (c / 9) % 3;
      const int t3 = c / 27;
      smem.lut[c] = make_uint2(
          dhf_bf16x2_u32(static_cast<float>(t0 - 1), static_cast<float>(t1 - 1)),
          dhf_bf16x2_u32(static_cast<float>(t2 - 1), static_cast<float>(t3 - 1)));
    }
  }
  if (tid < kMaxM * kCtaTopK) {
    smem.s_topv[tid >> 3][tid & 7] = -1e38f;
    smem.s_topi[tid >> 3][tid & 7] = -1;
  }

  // 2 MiB TLB prefetch from block 0 (immutable W_packed: safe before PDL wait).
  if (bid == 0) {
    const size_t total_bytes = static_cast<size_t>(n_blocks) * num_kg * kStepWBytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes;
         off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  // Thread mapping for single-pass cp.async (tid 0..63 -> A, tid 64..64+kStepWU4-1 -> W).
  const int a_row = tid >> 3;
  const int a_col8 = tid & 7;
  const int a_swiz8 = a_col8 ^ (a_row & 7);
  const bool a_valid = (tid < 64) && (a_row < M);
  const int w_u4_idx = tid - 64;
  const bool w_loader = (tid >= 64) && (w_u4_idx < kStepWU4);

  // Precompute 26-bit unpack indices for mode=158.
  const int bit158 = lane * 26;
  const int w_idx158 = bit158 >> 5;
  const int b_off158 = bit158 & 31;
  const bool need_hi158 = (b_off158 > 6);

  // Precompute ldmatrix SMEM offsets for the 4 k16 steps.
  uint32_t ldm_off[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int ldm_col8 = (2 * s + ((lane >> 3) & 1)) ^ (lane & 7);
    ldm_off[s] = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem.a[0][lane & 7][ldm_col8 * 8]) -
        __cvta_generic_to_shared(&smem.a[0][0][0]));
  }

  constexpr int kPreload = kStages - 1;  // 5
  const int preload = (num_kg < kPreload) ? num_kg : kPreload;

  for (int cta_n = bid; cta_n < n_blocks; cta_n += grid_ctas) {
    const bool is_first_wave = (cta_n == bid);
    const bool is_last_wave = (cta_n + grid_ctas >= n_blocks);

    const int wave = cta_n / kWaveCTAs;
    const int slot = cta_n - wave * kWaveCTAs;
    const int wave_base = wave * kWaveCTAs;
    const int w_size = (n_blocks - wave_base < kWaveCTAs) ? (n_blocks - wave_base) : kWaveCTAs;
    const uint8_t* w_base =
        W_packed + (static_cast<size_t>(wave_base) * num_kg + slot) * kStepWBytes;
    const size_t w_kg_stride = static_cast<size_t>(w_size) * kStepWBytes;

    auto issue_w_only = [&](int kg, int st) {
      if (w_loader) {
        const uint8_t* w_src = w_base + static_cast<size_t>(kg) * w_kg_stride;
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][w_u4_idx * 16]));
        dhf_cp_async16_cg(sd, w_src + w_u4_idx * 16);
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    };

    auto load_a_sync = [&](int kg, int st) {
      if (tid < 64) {
        *reinterpret_cast<uint4*>(&smem.a[st][a_row][a_swiz8 * 8]) =
            a_valid
                ? __ldg(reinterpret_cast<const uint4*>(
                      A_scaled + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8))
                : make_uint4(0u, 0u, 0u, 0u);
      }
    };

    auto issue_stage = [&](int kg, int st) {
      if (w_loader) {
        const uint8_t* w_src = w_base + static_cast<size_t>(kg) * w_kg_stride;
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][w_u4_idx * 16]));
        dhf_cp_async16_cg(sd, w_src + w_u4_idx * 16);
      } else if (tid < 64) {
        const uint32_t sd = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.a[st][a_row][a_swiz8 * 8]));
        const __nv_bfloat16* gs =
            a_valid ? (A_scaled + static_cast<size_t>(a_row) * K + kg * 64 + a_col8 * 8) : A_scaled;
        dhf_cp_async16_ca_z(sd, gs, a_valid ? 16 : 0);
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    };

    if (is_first_wave && ((pdl_mode & 1) != 0)) {
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

    float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
    float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;
    const int trig_i = (is_last_wave && ((pdl_mode & 2) != 0))
                           ? ((num_kg > 6) ? (num_kg - 6) : 0)
                           : -1;

    int st = 0;
    for (int i = 0; i < num_kg; ++i) {
      if (i == trig_i) {
        asm volatile("griddepcontrol.launch_dependents;");
      }
      const int rem = num_kg - 1 - i;
      if (rem >= 4) {
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

      const int next_i = i + kPreload;
      if (next_i < num_kg) {
        const int next_st = (st == 0) ? (kStages - 1) : (st - 1);
        issue_stage(next_i, next_st);
      }

      const uint8_t* wt0 = &smem.w[st][(warp * 2 + 0) * kTileBytes];
      const uint8_t* wt1 = &smem.w[st][(warp * 2 + 1) * kTileBytes];
      const uint32_t a_stage_base = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][0][0]));

      if constexpr (kMode == 158) {
        const uint32_t* u0 = reinterpret_cast<const uint32_t*>(wt0);
        const uint32_t* u1 = reinterpret_cast<const uint32_t*>(wt1);
        const uint32_t c26_0 = __funnelshift_r(u0[w_idx158], need_hi158 ? u0[w_idx158 + 1] : 0u, b_off158) & 0x03FFFFFFu;
        const uint32_t c26_1 = __funnelshift_r(u1[w_idx158], need_hi158 ? u1[w_idx158 + 1] : 0u, b_off158) & 0x03FFFFFFu;

        const uint32_t lo0 = c26_0 & 0x1FFFu;
        const uint32_t hi0 = c26_0 >> 13;
        const uint32_t q0_1 = __umulhi(lo0, 0x03291620u);
        const uint32_t q0_0 = lo0 - q0_1 * 81u;
        const uint32_t q0_3 = __umulhi(hi0, 0x03291620u);
        const uint32_t q0_2 = hi0 - q0_3 * 81u;

        const uint32_t lo1 = c26_1 & 0x1FFFu;
        const uint32_t hi1 = c26_1 >> 13;
        const uint32_t q1_1 = __umulhi(lo1, 0x03291620u);
        const uint32_t q1_0 = lo1 - q1_1 * 81u;
        const uint32_t q1_3 = __umulhi(hi1, 0x03291620u);
        const uint32_t q1_2 = hi1 - q1_3 * 81u;

        const uint32_t idx0[4] = {q0_0, q0_1, q0_2, q0_3};
        const uint32_t idx1[4] = {q1_0, q1_1, q1_2, q1_3};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          uint32_t a0, a2;
          dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[s]);
          const uint2 b = smem.lut[idx0[s]];
          const uint2 c = smem.lut[idx1[s]];
          dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b.x, b.y);
          dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, c.x, c.y);
        }
      } else {
        const uint32_t q32_0 = reinterpret_cast<const uint32_t*>(wt0)[lane];
        const uint32_t q32_1 = reinterpret_cast<const uint32_t*>(wt1)[lane];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          uint32_t a0, a2;
          dhf_ldmatrix_x2(a0, a2, a_stage_base + ldm_off[s]);
          uint32_t b0, b1, c0, c1;
          if constexpr (true) {
            if (s == 0) {
              dhf_deq2_k16<0>(q32_0, b0, b1); dhf_deq2_k16<0>(q32_1, c0, c1);
            } else if (s == 1) {
              dhf_deq2_k16<1>(q32_0, b0, b1); dhf_deq2_k16<1>(q32_1, c0, c1);
            } else if (s == 2) {
              dhf_deq2_k16<2>(q32_0, b0, b1); dhf_deq2_k16<2>(q32_1, c0, c1);
            } else {
              dhf_deq2_k16<3>(q32_0, b0, b1); dhf_deq2_k16<3>(q32_1, c0, c1);
            }
          }
          dhf_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2, b0, b1);
          dhf_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2, c0, c1);
        }
      }

      st = (st + 1 == kStages) ? 0 : (st + 1);
    }

    // Apply per-row optimal scales and stage 64 scores into SMEM.
    const int row_base = cta_n * kRowsPerCTA;
    if (tid < kRowsPerCTA) {
      smem.s_toks[tid] = __ldg(&packed_tok_ids[row_base + tid]);
    }
    if (g < M) {
      const int c0 = (warp * 2 + 0) * 8 + 2 * t;
      const int c1 = (warp * 2 + 1) * 8 + 2 * t;
      smem.s_scores[g][c0 + 0] = d0_0 * __ldg(&row_scales[row_base + c0 + 0]);
      smem.s_scores[g][c0 + 1] = d1_0 * __ldg(&row_scales[row_base + c0 + 1]);
      smem.s_scores[g][c1 + 0] = d0_1 * __ldg(&row_scales[row_base + c1 + 0]);
      smem.s_scores[g][c1 + 1] = d1_1 * __ldg(&row_scales[row_base + c1 + 1]);
    }
    __syncthreads();

    // Update persistent per-CTA top-8 for each row m (4 warps handle m = warp, warp + 4).
    for (int m = warp; m < M; m += kWarpsPerCTA) {
      const float thresh = smem.s_topv[m][7];
      const float v0 = smem.s_scores[m][lane];
      const float v1 = smem.s_scores[m][lane + 32];
      uint32_t m0 = __ballot_sync(0xffffffffu, v0 >= thresh);
      uint32_t m1 = __ballot_sync(0xffffffffu, v1 >= thresh);
      if (lane == 0 && ((m0 | m1) != 0u)) {
        float tv1 = smem.s_topv[m][0], tv2 = smem.s_topv[m][1];
        float tv3 = smem.s_topv[m][2], tv4 = smem.s_topv[m][3];
        float tv5 = smem.s_topv[m][4], tv6 = smem.s_topv[m][5];
        float tv7 = smem.s_topv[m][6], tv8 = smem.s_topv[m][7];
        int ti1 = smem.s_topi[m][0], ti2 = smem.s_topi[m][1];
        int ti3 = smem.s_topi[m][2], ti4 = smem.s_topi[m][3];
        int ti5 = smem.s_topi[m][4], ti6 = smem.s_topi[m][5];
        int ti7 = smem.s_topi[m][6], ti8 = smem.s_topi[m][7];
        while (m0 != 0u) {
          const int j = __ffs(m0) - 1;
          m0 &= m0 - 1u;
          dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                          tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                          smem.s_scores[m][j], smem.s_toks[j]);
        }
        while (m1 != 0u) {
          const int j = 32 + __ffs(m1) - 1;
          m1 &= m1 - 1u;
          dhf_top8_insert(tv1, ti1, tv2, ti2, tv3, ti3, tv4, ti4,
                          tv5, ti5, tv6, ti6, tv7, ti7, tv8, ti8,
                          smem.s_scores[m][j], smem.s_toks[j]);
        }
        smem.s_topv[m][0] = tv1; smem.s_topi[m][0] = ti1;
        smem.s_topv[m][1] = tv2; smem.s_topi[m][1] = ti2;
        smem.s_topv[m][2] = tv3; smem.s_topi[m][2] = ti3;
        smem.s_topv[m][3] = tv4; smem.s_topi[m][3] = ti4;
        smem.s_topv[m][4] = tv5; smem.s_topi[m][4] = ti5;
        smem.s_topv[m][5] = tv6; smem.s_topi[m][5] = ti6;
        smem.s_topv[m][6] = tv7; smem.s_topi[m][6] = ti7;
        smem.s_topv[m][7] = tv8; smem.s_topi[m][7] = ti8;
      }
    }
    __syncthreads();
  }

  // Emit per-CTA top-8 to L2-resident scratch.
  const int row_stride = grid_ctas * kCtaTopK;
  if (tid < M * kCtaTopK) {
    const int m = tid >> 3;
    const int k = tid & 7;
    cta_vals[static_cast<size_t>(m) * row_stride + bid * kCtaTopK + k] = smem.s_topv[m][k];
    cta_ids[static_cast<size_t>(m) * row_stride + bid * kCtaTopK + k] = smem.s_topi[m][k];
  }
}

__device__ __forceinline__ uint32_t dhf_f32_to_key(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

// Kernel 2: Exact 3-pass 8-bit MSB radix select from N_in (<=1536) CTA shortlist -> K_cand (<=256).
// Grid: M blocks, Block: 256 threads (~3 us, L2-resident).
__global__ void dhf_select_topk_kernel(
    const float* __restrict__ cta_vals,
    const int32_t* __restrict__ cta_ids,
    int32_t* __restrict__ cand_ids,
    int N_in, int K_cand, int pdl_mode) {
  const int m = blockIdx.x;
  const int tid = threadIdx.x;

  __shared__ uint32_t s_keys[kWaveCTAs * kCtaTopK];
  __shared__ int32_t s_ids[kWaveCTAs * kCtaTopK];
  __shared__ uint32_t s_hist[256];
  __shared__ uint32_t s_prefix;
  __shared__ uint32_t s_rem_k;
  __shared__ uint32_t s_out_cnt;

  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  const float* v_in = cta_vals + static_cast<size_t>(m) * N_in;
  const int32_t* i_in = cta_ids + static_cast<size_t>(m) * N_in;
  int32_t* out = cand_ids + static_cast<size_t>(m) * K_cand;

  for (int i = tid; i < N_in; i += 256) {
    float v = v_in[i];
    int32_t id = i_in[i];
    if (isnan(v) || id < 0) {
      v = -1e38f;
      id = -1;
    }
    s_keys[i] = dhf_f32_to_key(v);
    s_ids[i] = id;
  }
  if (tid == 0) {
    s_prefix = 0u;
    s_rem_k = static_cast<uint32_t>((K_cand < N_in) ? K_cand : N_in);
    s_out_cnt = 0u;
  }
  __syncthreads();

#pragma unroll
  for (int pass = 0; pass < 3; ++pass) {
    const uint32_t shift = static_cast<uint32_t>(24 - pass * 8);
    const uint32_t mask = (pass == 0) ? 0u : ((pass == 1) ? 0xFF000000u : 0xFFFF0000u);
    const uint32_t pref = s_prefix;
    s_hist[tid] = 0u;
    __syncthreads();

    for (int i = tid; i < N_in; i += 256) {
      const uint32_t k = s_keys[i];
      if ((k & mask) == pref) {
        atomicAdd(&s_hist[(k >> shift) & 0xFFu], 1u);
      }
    }
    __syncthreads();

    if (tid == 0) {
      uint32_t acc = 0u;
      const uint32_t need = s_rem_k;
      uint32_t chosen = 0u;
      for (int b = 255; b >= 0; --b) {
        const uint32_t c = s_hist[b];
        if (acc + c >= need) {
          chosen = static_cast<uint32_t>(b);
          s_rem_k = need - acc;
          break;
        }
        acc += c;
      }
      s_prefix = pref | (chosen << shift);
    }
    __syncthreads();
  }

  const uint32_t thresh = s_prefix;  // exact top-24-bit threshold (bits 31..8)
  // Emit strictly greater items first.
  for (int i = tid; i < N_in; i += 256) {
    if ((s_keys[i] & 0xFFFFFF00u) > thresh && s_ids[i] >= 0) {
      const uint32_t slot = atomicAdd(&s_out_cnt, 1u);
      if (slot < static_cast<uint32_t>(K_cand)) {
        out[slot] = s_ids[i];
      }
    }
  }
  __syncthreads();

  // Fill remaining slots up to K_cand with items matching the 24-bit threshold bucket.
  for (int i = tid; i < N_in; i += 256) {
    if ((s_keys[i] & 0xFFFFFF00u) == thresh && s_ids[i] >= 0) {
      const uint32_t slot = atomicAdd(&s_out_cnt, 1u);
      if (slot < static_cast<uint32_t>(K_cand)) {
        out[slot] = s_ids[i];
      }
    }
  }
  __syncthreads();

  const uint32_t final_cnt = s_out_cnt;
  for (int i = static_cast<int>(final_cnt) + tid; i < K_cand; i += 256) {
    out[i] = -1;
  }
  if ((pdl_mode & 2) != 0 && tid == 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
}

// Exact warp-level NVFP4 dot product (bit-identical to draft_vocab_tail.cu's dvt_warp_nvfp4_dot).
__device__ __forceinline__ float dhf_nvfp4_group_dot(
    uint32_t wlo, uint32_t whi, const uint4& a0, const uint4& a1) {
  const __nv_bfloat162* ap0 = reinterpret_cast<const __nv_bfloat162*>(&a0);
  const __nv_bfloat162* ap1 = reinterpret_cast<const __nv_bfloat162*>(&a1);
  float gsum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 af = __bfloat1622float2(ap0[i]);
    gsum = __fmaf_rn(dhf_e2m1_x2_to_f32(wlo & 0xFu), af.x, gsum);
    gsum = __fmaf_rn(dhf_e2m1_x2_to_f32((wlo >> 4) & 0xFu), af.y, gsum);
    wlo >>= 8;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 af = __bfloat1622float2(ap1[i]);
    gsum = __fmaf_rn(dhf_e2m1_x2_to_f32(whi & 0xFu), af.x, gsum);
    gsum = __fmaf_rn(dhf_e2m1_x2_to_f32((whi >> 4) & 0xFu), af.y, gsum);
    whi >>= 8;
  }
  return gsum;
}

__device__ __forceinline__ float dhf_warp_nvfp4_dot(
    const __nv_bfloat16* __restrict__ a_row,
    const uint4* __restrict__ s_a0,
    const uint4* __restrict__ s_a1,
    bool use_smem_a,
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    int v, int K, int num_g, int lane, float total_scale) {
  const uint8_t* w_row = raw_w + static_cast<size_t>(v) * (K >> 1);
  const uint8_t* s_row = raw_s + static_cast<size_t>(v) * num_g;
  float acc = 0.0f;
  if (((num_g & 1) == 0) && ((reinterpret_cast<uintptr_t>(w_row) & 15u) == 0u) &&
      ((reinterpret_cast<uintptr_t>(s_row) & 1u) == 0u)) {
    const int num_g2 = num_g >> 1;
    const uint4* w16_row = reinterpret_cast<const uint4*>(w_row);
    const uint16_t* s2_row = reinterpret_cast<const uint16_t*>(s_row);
    for (int g2 = lane; g2 < num_g2; g2 += 32) {
      const uint16_t s2 = __ldg(&s2_row[g2]);
      const uint4 w16 = __ldg(&w16_row[g2]);
      const int g0 = g2 << 1;
      const int g1 = g0 | 1;
      const float sc0 = dhf_fp8_e4m3_to_f32(static_cast<uint8_t>(s2 & 0xFFu)) * 0.5f;
      const float sc1 = dhf_fp8_e4m3_to_f32(static_cast<uint8_t>(s2 >> 8)) * 0.5f;
      const uint4 a00 = use_smem_a ? s_a0[g0] : __ldg(reinterpret_cast<const uint4*>(a_row) + g0 * 2 + 0);
      const uint4 a01 = use_smem_a ? s_a1[g0] : __ldg(reinterpret_cast<const uint4*>(a_row) + g0 * 2 + 1);
      const uint4 a10 = use_smem_a ? s_a0[g1] : __ldg(reinterpret_cast<const uint4*>(a_row) + g1 * 2 + 0);
      const uint4 a11 = use_smem_a ? s_a1[g1] : __ldg(reinterpret_cast<const uint4*>(a_row) + g1 * 2 + 1);
      acc = __fmaf_rn(dhf_nvfp4_group_dot(w16.x, w16.y, a00, a01), sc0, acc);
      acc = __fmaf_rn(dhf_nvfp4_group_dot(w16.z, w16.w, a10, a11), sc1, acc);
    }
  } else {
    for (int g = lane; g < num_g; g += 32) {
      const float sc_half = dhf_fp8_e4m3_to_f32(__ldg(&s_row[g])) * 0.5f;
      const uint64_t w8 = __ldg(reinterpret_cast<const uint64_t*>(w_row + g * 8));
      const uint4 a0 = use_smem_a ? s_a0[g] : __ldg(reinterpret_cast<const uint4*>(a_row) + g * 2 + 0);
      const uint4 a1 = use_smem_a ? s_a1[g] : __ldg(reinterpret_cast<const uint4*>(a_row) + g * 2 + 1);
      acc = __fmaf_rn(
          dhf_nvfp4_group_dot(static_cast<uint32_t>(w8), static_cast<uint32_t>(w8 >> 32), a0, a1),
          sc_half, acc);
    }
  }
  acc = dhf_warp_sum(acc);
  return __bfloat162float(__float2bfloat16_rn(acc * total_scale));
}

// Kernel 3: Stage 2 exact NVFP4 rescore of K_cand shortlist + prompt tail.
// Grid: (16, M), Block: 256 threads (8 warps, 2 CTAs/SM).
// Always tracks top-8 per warp/block (negligible cost on <= 32 candidates/CTA), feeding Kernel 4.
__global__ void __launch_bounds__(256, 2) dhf_stage2_rescore_kernel(
    const __nv_bfloat16* __restrict__ A,
    const int32_t* __restrict__ cand_ids,
    const int32_t* __restrict__ tail_ids,
    const int32_t* __restrict__ tail_count,
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    const float* __restrict__ global_scale,
    float* __restrict__ s2_vals,
    int32_t* __restrict__ s2_ids,
    int K_cand, int K, int max_tail, float logit_scale, int pdl_mode) {
  const int bid = blockIdx.x;
  const int m = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int num_g = K >> 4;
  const bool use_smem_a = (num_g <= kS2MaxGroups);

  __shared__ uint4 s_a0[kS2MaxGroups];
  __shared__ uint4 s_a1[kS2MaxGroups];
  __shared__ float wv[kS2Warps][8];
  __shared__ int32_t wi[kS2Warps][8];

  const __nv_bfloat16* a_row = A + static_cast<size_t>(m) * K;
  if (use_smem_a) {
    const uint4* a_u4 = reinterpret_cast<const uint4*>(a_row);
    for (int g = tid; g < num_g; g += 256) {
      s_a0[g] = __ldg(&a_u4[g * 2 + 0]);
      s_a1[g] = __ldg(&a_u4[g * 2 + 1]);
    }
    __syncthreads();
  }

  float v1 = -1e38f, v2 = -1e38f, v3 = -1e38f, v4 = -1e38f;
  float v5 = -1e38f, v6 = -1e38f, v7 = -1e38f, v8 = -1e38f;
  int i1 = 0x7fffffff, i2 = 0x7fffffff, i3 = 0x7fffffff, i4 = 0x7fffffff;
  int i5 = 0x7fffffff, i6 = 0x7fffffff, i7 = 0x7fffffff, i8 = 0x7fffffff;

  const float total_scale = __ldg(global_scale) * logit_scale;
  const int stride_w = kS2CTAsPerRow * kS2Warps;  // 128

  // Phase A (before PDL wait): score dynamic prompt-tail OOV rows (already ready in memory!).
  int tcnt = __ldg(tail_count);
  if (tcnt < 0) tcnt = 0;
  if (tcnt > max_tail) tcnt = max_tail;
  for (int idx = bid * kS2Warps + warp; idx < tcnt; idx += stride_w) {
    const int tok = __ldg(&tail_ids[idx]);
    if (tok >= 0) {
      const float val = dhf_warp_nvfp4_dot(
          a_row, s_a0, s_a1, use_smem_a, raw_w, raw_s, tok, K, num_g, lane, total_scale);
      if (lane == 0) {
        dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, val, tok);
      }
    }
  }

  // Wait for Kernel 2 (cand_ids) via SM121 PDL.
  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  // Phase B: score static shortlist candidates against exact raw NVFP4 weights.
  const int32_t* my_cands = cand_ids + static_cast<size_t>(m) * K_cand;
  for (int idx = bid * kS2Warps + warp; idx < K_cand; idx += stride_w) {
    const int tok = __ldg(&my_cands[idx]);
    if (tok >= 0) {
      const float val = dhf_warp_nvfp4_dot(
          a_row, s_a0, s_a1, use_smem_a, raw_w, raw_s, tok, K, num_g, lane, total_scale);
      if (lane == 0) {
        dhf_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, val, tok);
      }
    }
  }

  if (lane == 0) {
    wv[warp][0] = v1; wi[warp][0] = i1;
    wv[warp][1] = v2; wi[warp][1] = i2;
    wv[warp][2] = v3; wi[warp][2] = i3;
    wv[warp][3] = v4; wi[warp][3] = i4;
    wv[warp][4] = v5; wi[warp][4] = i5;
    wv[warp][5] = v6; wi[warp][5] = i6;
    wv[warp][6] = v7; wi[warp][6] = i7;
    wv[warp][7] = v8; wi[warp][7] = i8;
  }
  __syncthreads();

  if (tid == 0) {
    if ((pdl_mode & 2) != 0) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    float bv1 = wv[0][0], bv2 = wv[0][1], bv3 = wv[0][2], bv4 = wv[0][3];
    float bv5 = wv[0][4], bv6 = wv[0][5], bv7 = wv[0][6], bv8 = wv[0][7];
    int bi1 = wi[0][0], bi2 = wi[0][1], bi3 = wi[0][2], bi4 = wi[0][3];
    int bi5 = wi[0][4], bi6 = wi[0][5], bi7 = wi[0][6], bi8 = wi[0][7];
#pragma unroll
    for (int w = 1; w < kS2Warps; ++w) {
      dhf_top8_merge(
          bv1, bi1, bv2, bi2, bv3, bi3, bv4, bi4, bv5, bi5, bv6, bi6, bv7, bi7, bv8, bi8,
          wv[w][0], wi[w][0], wv[w][1], wi[w][1], wv[w][2], wi[w][2], wv[w][3], wi[w][3],
          wv[w][4], wi[w][4], wv[w][5], wi[w][5], wv[w][6], wi[w][6], wv[w][7], wi[w][7]);
    }
    const size_t base = (static_cast<size_t>(m) * kS2CTAsPerRow + bid) * 8;
    s2_vals[base + 0] = bv1; s2_ids[base + 0] = bi1;
    s2_vals[base + 1] = bv2; s2_ids[base + 1] = bi2;
    s2_vals[base + 2] = bv3; s2_ids[base + 2] = bi3;
    s2_vals[base + 3] = bv4; s2_ids[base + 3] = bi4;
    s2_vals[base + 4] = bv5; s2_ids[base + 4] = bi5;
    s2_vals[base + 5] = bv6; s2_ids[base + 5] = bi6;
    s2_vals[base + 6] = bv7; s2_ids[base + 6] = bi7;
    s2_vals[base + 7] = bv8; s2_ids[base + 7] = bi8;
  }
}

// Kernel 4: Final 16-block merge per row m -> exact output contract matching draft_vocab_tail_argmax.
__global__ void dhf_final_reduce_kernel(
    const float* __restrict__ s2_vals,
    const int32_t* __restrict__ s2_ids,
    int64_t* __restrict__ out,
    int64_t* __restrict__ out_top2_row0,
    int top2_rows,
    float* __restrict__ out_margins,
    int margin_rows,
    int64_t* __restrict__ out_tree_top,
    float* __restrict__ out_tree_vals,
    int tree_top_rows,
    int tree_k,
    int M,
    int pdl_mode) {
  const int m = blockIdx.x;
  const int lane = threadIdx.x;
  if (m >= M) return;

  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  float v1 = -1e38f, v2 = -1e38f, v3 = -1e38f, v4 = -1e38f;
  float v5 = -1e38f, v6 = -1e38f, v7 = -1e38f, v8 = -1e38f;
  int i1 = 0x7fffffff, i2 = 0x7fffffff, i3 = 0x7fffffff, i4 = 0x7fffffff;
  int i5 = 0x7fffffff, i6 = 0x7fffffff, i7 = 0x7fffffff, i8 = 0x7fffffff;

  if (lane < kS2CTAsPerRow) {
    const size_t base = (static_cast<size_t>(m) * kS2CTAsPerRow + lane) * 8;
    v1 = s2_vals[base + 0]; i1 = s2_ids[base + 0];
    v2 = s2_vals[base + 1]; i2 = s2_ids[base + 1];
    v3 = s2_vals[base + 2]; i3 = s2_ids[base + 2];
    v4 = s2_vals[base + 3]; i4 = s2_ids[base + 3];
    v5 = s2_vals[base + 4]; i5 = s2_ids[base + 4];
    v6 = s2_vals[base + 5]; i6 = s2_ids[base + 5];
    v7 = s2_vals[base + 6]; i7 = s2_ids[base + 6];
    v8 = s2_vals[base + 7]; i8 = s2_ids[base + 7];
  }

#pragma unroll
  for (int o = 8; o > 0; o >>= 1) {
    const float ov1 = __shfl_down_sync(0xffffffffu, v1, o);
    const int oi1 = __shfl_down_sync(0xffffffffu, i1, o);
    const float ov2 = __shfl_down_sync(0xffffffffu, v2, o);
    const int oi2 = __shfl_down_sync(0xffffffffu, i2, o);
    const float ov3 = __shfl_down_sync(0xffffffffu, v3, o);
    const int oi3 = __shfl_down_sync(0xffffffffu, i3, o);
    const float ov4 = __shfl_down_sync(0xffffffffu, v4, o);
    const int oi4 = __shfl_down_sync(0xffffffffu, i4, o);
    const float ov5 = __shfl_down_sync(0xffffffffu, v5, o);
    const int oi5 = __shfl_down_sync(0xffffffffu, i5, o);
    const float ov6 = __shfl_down_sync(0xffffffffu, v6, o);
    const int oi6 = __shfl_down_sync(0xffffffffu, i6, o);
    const float ov7 = __shfl_down_sync(0xffffffffu, v7, o);
    const int oi7 = __shfl_down_sync(0xffffffffu, i7, o);
    const float ov8 = __shfl_down_sync(0xffffffffu, v8, o);
    const int oi8 = __shfl_down_sync(0xffffffffu, i8, o);
    dhf_top8_merge(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8,
                   ov1, oi1, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6, ov7, oi7, ov8, oi8);
  }

  if (lane == 0) {
    out[m] = static_cast<int64_t>(i1);
    if (out_top2_row0 != nullptr && m < top2_rows) {
      out_top2_row0[2 * m + 0] = static_cast<int64_t>(i1);
      out_top2_row0[2 * m + 1] = (i2 == 0x7fffffff) ? static_cast<int64_t>(i1) : static_cast<int64_t>(i2);
    }
    if (out_margins != nullptr && m < margin_rows) {
      out_margins[m] = (i2 == 0x7fffffff) ? 100.0f : (v1 - v2);
    }
    if (out_tree_top != nullptr && m < tree_top_rows) {
      if (tree_k >= 8) {
        out_tree_top[8 * m + 0] = static_cast<int64_t>(i1);
        out_tree_top[8 * m + 1] = (i2 == 0x7fffffff) ? -1 : static_cast<int64_t>(i2);
        out_tree_top[8 * m + 2] = (i3 == 0x7fffffff) ? -1 : static_cast<int64_t>(i3);
        out_tree_top[8 * m + 3] = (i4 == 0x7fffffff) ? -1 : static_cast<int64_t>(i4);
        out_tree_top[8 * m + 4] = (i5 == 0x7fffffff) ? -1 : static_cast<int64_t>(i5);
        out_tree_top[8 * m + 5] = (i6 == 0x7fffffff) ? -1 : static_cast<int64_t>(i6);
        out_tree_top[8 * m + 6] = (i7 == 0x7fffffff) ? -1 : static_cast<int64_t>(i7);
        out_tree_top[8 * m + 7] = (i8 == 0x7fffffff) ? -1 : static_cast<int64_t>(i8);
        if (out_tree_vals != nullptr) {
          out_tree_vals[8 * m + 0] = (i1 == 0x7fffffff) ? -1e30f : v1;
          out_tree_vals[8 * m + 1] = (i2 == 0x7fffffff) ? -1e30f : v2;
          out_tree_vals[8 * m + 2] = (i3 == 0x7fffffff) ? -1e30f : v3;
          out_tree_vals[8 * m + 3] = (i4 == 0x7fffffff) ? -1e30f : v4;
          out_tree_vals[8 * m + 4] = (i5 == 0x7fffffff) ? -1e30f : v5;
          out_tree_vals[8 * m + 5] = (i6 == 0x7fffffff) ? -1e30f : v6;
          out_tree_vals[8 * m + 6] = (i7 == 0x7fffffff) ? -1e30f : v7;
          out_tree_vals[8 * m + 7] = (i8 == 0x7fffffff) ? -1e30f : v8;
        }
      } else if (tree_k >= 6) {
        out_tree_top[6 * m + 0] = static_cast<int64_t>(i1);
        out_tree_top[6 * m + 1] = (i2 == 0x7fffffff) ? -1 : static_cast<int64_t>(i2);
        out_tree_top[6 * m + 2] = (i3 == 0x7fffffff) ? -1 : static_cast<int64_t>(i3);
        out_tree_top[6 * m + 3] = (i4 == 0x7fffffff) ? -1 : static_cast<int64_t>(i4);
        out_tree_top[6 * m + 4] = (i5 == 0x7fffffff) ? -1 : static_cast<int64_t>(i5);
        out_tree_top[6 * m + 5] = (i6 == 0x7fffffff) ? -1 : static_cast<int64_t>(i6);
        if (out_tree_vals != nullptr) {
          out_tree_vals[6 * m + 0] = (i1 == 0x7fffffff) ? -1e30f : v1;
          out_tree_vals[6 * m + 1] = (i2 == 0x7fffffff) ? -1e30f : v2;
          out_tree_vals[6 * m + 2] = (i3 == 0x7fffffff) ? -1e30f : v3;
          out_tree_vals[6 * m + 3] = (i4 == 0x7fffffff) ? -1e30f : v4;
          out_tree_vals[6 * m + 4] = (i5 == 0x7fffffff) ? -1e30f : v5;
          out_tree_vals[6 * m + 5] = (i6 == 0x7fffffff) ? -1e30f : v6;
        }
      } else {
        out_tree_top[4 * m + 0] = static_cast<int64_t>(i1);
        out_tree_top[4 * m + 1] = (i2 == 0x7fffffff) ? -1 : static_cast<int64_t>(i2);
        out_tree_top[4 * m + 2] = (i3 == 0x7fffffff) ? -1 : static_cast<int64_t>(i3);
        out_tree_top[4 * m + 3] = (i4 == 0x7fffffff) ? -1 : static_cast<int64_t>(i4);
      }
    }
  }
}

struct DHeadFastScratch {
  torch::Tensor a_scaled;
  torch::Tensor cta_vals;
  torch::Tensor cta_ids;
  torch::Tensor cand_ids;
  torch::Tensor s2_vals;
  torch::Tensor s2_ids;
  int cap_k = 0;
  c10::DeviceIndex dev = -1;
};

DHeadFastScratch& get_dhf_scratch(torch::Device device, int K) {
  static DHeadFastScratch sc;
  if (!sc.cta_vals.defined() || sc.dev != device.index() || sc.cap_k < K) {
    auto opt_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    sc.a_scaled = torch::empty({kMaxM, K}, opt_bf16);
    sc.cta_vals = torch::empty({kMaxM, kWaveCTAs * kCtaTopK}, opt_f32);
    sc.cta_ids = torch::empty({kMaxM, kWaveCTAs * kCtaTopK}, opt_i32);
    sc.cand_ids = torch::empty({kMaxM, kMaxCand}, opt_i32);
    sc.s2_vals = torch::empty({kMaxM, kS2CTAsPerRow, 8}, opt_f32);
    sc.s2_ids = torch::empty({kMaxM, kS2CTAsPerRow, 8}, opt_i32);
    sc.cap_k = K;
    sc.dev = device.index();
  }
  return sc;
}

int pick_coprime_stride(int V_static) {
  static constexpr int kPrimes[] = {1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061};
  for (int p : kPrimes) {
    if (std::gcd(p, V_static) == 1) return p;
  }
  for (int p = 1063; ; p += 2) {
    if (std::gcd(p, V_static) == 1) return p;
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> dhead_fast_pack(
    torch::Tensor raw_w,
    torch::Tensor raw_s,
    torch::Tensor global_scale,
    torch::Tensor vocab_map,
    int64_t mode) {
  TORCH_CHECK(raw_w.is_cuda() && raw_w.scalar_type() == torch::kUInt8 && raw_w.dim() == 2 && raw_w.is_contiguous());
  TORCH_CHECK(raw_s.is_cuda() && raw_s.dim() == 2 && raw_s.is_contiguous());
  TORCH_CHECK(global_scale.is_cuda() && global_scale.scalar_type() == torch::kFloat32);
  TORCH_CHECK(vocab_map.is_cuda() && vocab_map.scalar_type() == torch::kInt64 && vocab_map.dim() == 1);
  TORCH_CHECK(mode == 158 || mode == 2, "dhead_fast_pack: mode must be 158 or 2, got ", mode);

  const int V_static = static_cast<int>(vocab_map.numel());
  const int K = static_cast<int>(raw_w.size(1) * 2);
  TORCH_CHECK(V_static >= 64 && (V_static % kRowsPerCTA) == 0,
              "dhead_fast_pack: V_static must be multiple of 64, got ", V_static);
  TORCH_CHECK(K >= 64 && (K % 64) == 0, "dhead_fast_pack: K must be multiple of 64, got ", K);

  auto dev = raw_w.device();
  auto opt_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
  auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);

  // Coprime bijection disperses high-frequency BPE tokens uniformly across all 192 CTAs.
  const int stride = pick_coprime_stride(V_static);
  auto perm = (torch::arange(V_static, vocab_map.options()) * stride).remainder(V_static);
  auto packed_tok_ids = vocab_map.index_select(0, perm).to(torch::kInt32).contiguous();

  auto col_mean = torch::empty({K}, opt_f32);
  auto col_std = torch::empty({K}, opt_f32);
  auto row_scales = torch::empty({V_static}, opt_f32);
  auto row_thresh = torch::empty({V_static}, opt_f32);

  const int64_t tile_bytes = (mode == 158) ? kTileBytes158 : kTileBytes2;
  const int64_t num_tiles = static_cast<int64_t>(V_static / 8) * (K / 64);
  auto blob = torch::empty({num_tiles * tile_bytes}, opt_u8);

  auto st = c10::cuda::getCurrentCUDAStream().stream();

  // Pass 1: column mean & std
  dhf_col_stats_kernel<<<K >> 4, 256, 0, st>>>(
      raw_w.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
      global_scale.data_ptr<float>(),
      packed_tok_ids.data_ptr<int32_t>(),
      col_mean.data_ptr<float>(),
      col_std.data_ptr<float>(),
      V_static, K);

  // Pass 2: weighted-LS optimal row scales & thresholds
  const int row_blocks = (V_static + 7) / 8;
  if (mode == 158) {
    dhf_row_scales_kernel<158><<<row_blocks, 256, 0, st>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        packed_tok_ids.data_ptr<int32_t>(),
        col_mean.data_ptr<float>(),
        col_std.data_ptr<float>(),
        row_scales.data_ptr<float>(),
        row_thresh.data_ptr<float>(),
        V_static, K);
  } else {
    dhf_row_scales_kernel<2><<<row_blocks, 256, 0, st>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        packed_tok_ids.data_ptr<int32_t>(),
        col_mean.data_ptr<float>(),
        col_std.data_ptr<float>(),
        row_scales.data_ptr<float>(),
        row_thresh.data_ptr<float>(),
        V_static, K);
  }

  // Pass 3: pack wave-interleaved 8x64 tiles
  dim3 pack_grid(V_static / 8, K / 64);
  if (mode == 158) {
    dhf_pack_tiles_kernel<158><<<pack_grid, 32, 0, st>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        packed_tok_ids.data_ptr<int32_t>(),
        col_mean.data_ptr<float>(),
        col_std.data_ptr<float>(),
        row_thresh.data_ptr<float>(),
        blob.data_ptr<uint8_t>(),
        V_static, K);
  } else {
    dhf_pack_tiles_kernel<2><<<pack_grid, 32, 0, st>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        packed_tok_ids.data_ptr<int32_t>(),
        col_mean.data_ptr<float>(),
        col_std.data_ptr<float>(),
        row_thresh.data_ptr<float>(),
        blob.data_ptr<uint8_t>(),
        V_static, K);
  }

  return std::make_tuple(blob, row_scales, col_std, packed_tok_ids);
}

torch::Tensor dhead_fast_argmax(
    torch::Tensor A,
    torch::Tensor blob,
    torch::Tensor row_scales,
    torch::Tensor col_scales,
    torch::Tensor packed_tok_ids,
    torch::Tensor tail_ids,
    torch::Tensor tail_count,
    torch::Tensor raw_w,
    torch::Tensor raw_s,
    torch::Tensor global_scale,
    double logit_scale,
    int64_t mode,
    int64_t topk,
    c10::optional<torch::Tensor> out_top2_row0 = c10::nullopt,
    c10::optional<torch::Tensor> out_margins = c10::nullopt,
    c10::optional<torch::Tensor> out_tree_top = c10::nullopt,
    c10::optional<torch::Tensor> out_tree_vals = c10::nullopt) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous());
  TORCH_CHECK(blob.is_cuda() && blob.scalar_type() == torch::kUInt8 && blob.is_contiguous());
  TORCH_CHECK(mode == 158 || mode == 2);
  const int M = static_cast<int>(A.size(0));
  const int K = static_cast<int>(A.size(1));
  const int V_static = static_cast<int>(packed_tok_ids.numel());
  const int max_tail = static_cast<int>(tail_ids.numel());
  TORCH_CHECK(M >= 1 && M <= kMaxM && (K % 64) == 0 && (V_static % kRowsPerCTA) == 0);
  TORCH_CHECK(K == raw_w.size(1) * 2);

  int64_t* top2_ptr = nullptr;
  int top2_rows = 0;
  if (out_top2_row0.has_value() && out_top2_row0.value().defined() && out_top2_row0.value().numel() >= 2) {
    TORCH_CHECK(out_top2_row0.value().is_cuda() && out_top2_row0.value().scalar_type() == torch::kInt64);
    top2_ptr = out_top2_row0.value().data_ptr<int64_t>();
    top2_rows = (out_top2_row0.value().numel() >= 4 && M >= 2) ? 2 : 1;
  }
  float* margin_ptr = nullptr;
  int margin_rows = 0;
  if (out_margins.has_value() && out_margins.value().defined() && out_margins.value().numel() >= 1) {
    TORCH_CHECK(out_margins.value().is_cuda() && out_margins.value().scalar_type() == torch::kFloat32);
    margin_ptr = out_margins.value().data_ptr<float>();
    margin_rows = std::min(static_cast<int>(out_margins.value().numel()), top2_rows);
  }
  int64_t* tree_top_ptr = nullptr;
  int tree_top_rows = 0;
  int tree_k = 0;
  if (out_tree_top.has_value() && out_tree_top.value().defined() && out_tree_top.value().numel() >= 4) {
    TORCH_CHECK(out_tree_top.value().is_cuda() && out_tree_top.value().scalar_type() == torch::kInt64);
    tree_top_ptr = out_tree_top.value().data_ptr<int64_t>();
    if (out_tree_top.value().numel() >= 48) {
      tree_k = 8;
      tree_top_rows = std::min(6, std::min(M, static_cast<int>(out_tree_top.value().numel() / 8)));
    } else if (out_tree_top.value().numel() >= 36) {
      tree_k = 6;
      tree_top_rows = std::min(6, std::min(M, static_cast<int>(out_tree_top.value().numel() / 6)));
    } else {
      tree_k = 4;
      tree_top_rows = std::min(5, std::min(M, static_cast<int>(out_tree_top.value().numel() / 4)));
    }
  }
  float* tree_vals_ptr = nullptr;
  if (tree_k >= 6 && tree_top_rows > 0 && out_tree_vals.has_value() && out_tree_vals.value().defined() &&
      out_tree_vals.value().numel() >= tree_top_rows * tree_k) {
    TORCH_CHECK(out_tree_vals.value().is_cuda() && out_tree_vals.value().scalar_type() == torch::kFloat32);
    tree_vals_ptr = out_tree_vals.value().data_ptr<float>();
  }

  auto& sc = get_dhf_scratch(A.device(), K);
  auto out = torch::empty({M}, A.options().dtype(torch::kInt64));
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int pdl_mode = dhf_pdl_mode();

  const int n_blocks = V_static / kRowsPerCTA;
  const int grid_ctas = std::min(kWaveCTAs, n_blocks);
  const int N_in = grid_ctas * kCtaTopK;
  int K_cand = static_cast<int>(topk);
  if (K_cand < 16) K_cand = 16;
  if (K_cand > kMaxCand) K_cand = kMaxCand;
  if (K_cand > N_in) K_cand = N_in;

  // Kernel 0: Scale A by col_scales (does not wait on PDL; triggers Stage 1).
  dhf_launch_pdl(
      &dhf_prep_a_kernel, dim3(M, 1, 1), 256, st, (pdl_mode & 2),
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      col_scales.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(sc.a_scaled.data_ptr()),
      M, K, (pdl_mode & 2));

  // Kernel 1: Persistent Stage 1 compressed MMA + in-CTA top-8 shortlist.
  if (mode == 158) {
    dhf_launch_pdl(
        &dhf_stage1_kernel<158>, dim3(grid_ctas, 1, 1), kThreads, st, pdl_mode,
        reinterpret_cast<const __nv_bfloat16*>(sc.a_scaled.data_ptr()),
        blob.data_ptr<uint8_t>(),
        row_scales.data_ptr<float>(),
        packed_tok_ids.data_ptr<int32_t>(),
        sc.cta_vals.data_ptr<float>(),
        sc.cta_ids.data_ptr<int32_t>(),
        M, V_static, K, pdl_mode);
  } else {
    dhf_launch_pdl(
        &dhf_stage1_kernel<2>, dim3(grid_ctas, 1, 1), kThreads, st, pdl_mode,
        reinterpret_cast<const __nv_bfloat16*>(sc.a_scaled.data_ptr()),
        blob.data_ptr<uint8_t>(),
        row_scales.data_ptr<float>(),
        packed_tok_ids.data_ptr<int32_t>(),
        sc.cta_vals.data_ptr<float>(),
        sc.cta_ids.data_ptr<int32_t>(),
        M, V_static, K, pdl_mode);
  }

  // Kernel 2: Exact 3-pass radix select -> top K_cand target token IDs per row m.
  dhf_launch_pdl(
      &dhf_select_topk_kernel, dim3(M, 1, 1), 256, st, pdl_mode,
      static_cast<const float*>(sc.cta_vals.data_ptr<float>()),
      static_cast<const int32_t*>(sc.cta_ids.data_ptr<int32_t>()),
      sc.cand_ids.data_ptr<int32_t>(),
      N_in, K_cand, pdl_mode);

  // Kernel 3: Stage 2 exact NVFP4 rescore (prompt tail before PDL wait + shortlist after).
  dhf_launch_pdl(
      &dhf_stage2_rescore_kernel, dim3(kS2CTAsPerRow, M, 1), 256, st, pdl_mode,
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      static_cast<const int32_t*>(sc.cand_ids.data_ptr<int32_t>()),
      tail_ids.data_ptr<int32_t>(),
      tail_count.data_ptr<int32_t>(),
      raw_w.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
      global_scale.data_ptr<float>(),
      sc.s2_vals.data_ptr<float>(),
      sc.s2_ids.data_ptr<int32_t>(),
      K_cand, K, max_tail, static_cast<float>(logit_scale), pdl_mode);

  // Kernel 4: Final exact top-1 / top-2 / margin / tree top-4/6/8 reduction.
  dhf_launch_pdl(
      &dhf_final_reduce_kernel, dim3(M, 1, 1), 32, st, (pdl_mode & 1),
      static_cast<const float*>(sc.s2_vals.data_ptr<float>()),
      static_cast<const int32_t*>(sc.s2_ids.data_ptr<int32_t>()),
      out.data_ptr<int64_t>(),
      top2_ptr, top2_rows,
      margin_ptr, margin_rows,
      tree_top_ptr, tree_vals_ptr, tree_top_rows, tree_k,
      M, (pdl_mode & 1));

  return out;
}
