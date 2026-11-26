// SPDX-License-Identifier: Apache-2.0
// draft_vocab_tail: Dynamic per-prompt OOV token tail + fused exact-NVFP4 argmax for DFlash.
//
// Pairs with a reduced-vocabulary exact-NVFP4 static head (`V_static` rows, e.g. 16k..64k via
// decode6 TMA4-ILV):
//   1. `draft_vocab_update_tail`: 1-CTA (~3 us, zero CPU-GPU sync) GPU kernel that resets on
//      `positions[0] == 0` (new prompt) and appends any unique token ID not in `in_static_vocab`
//      into a fixed-address `[512]` GPU buffer (`tail_ids`, `tail_count`).
//   2. `draft_vocab_tail_argmax`: 16-CTA-per-row fused kernel (~20 us) that reduces
//      `static_logits [M, V_static]` (from L2 cache, mapping via `vocab_map`) AND scores all active
//      `tail_ids[0..tail_count-1]` rows directly against raw NVFP4 `(full_raw_w, full_raw_s,
//      full_global_scale)` using zero-bank-conflict split uint4 SMEM `s_a0/s_a1` and register E2M1
//      LUT (`0xC8643210u`), taking exact max with lowest-original-ID tie-breaking. 100% CUDA-graph safe.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <stdint.h>

namespace {

constexpr int kMaxTail = 512;
constexpr int kMaxM = 8;
constexpr int kS2CTAsPerRow = 16;  // 96 CTAs at M=6 (2 CTAs/SM on GB10)
constexpr int kS2Warps = 8;
constexpr int kS2MaxGroups = 512;  // covers K <= 8192 (16 KiB SMEM)

int dvt_pdl_mode() {
  static const int mode = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_PDL");
    if (!s || !*s) return 3;
    return (std::atoi(s) != 0) ? 3 : 0;
  }();
  return mode;
}

bool dvt_fast_enabled() {
  static const bool enabled = []() -> bool {
    const char* s = std::getenv("MUSE_DVT_FAST");
    if (!s || !*s) return true;
    return std::atoi(s) != 0;
  }();
  return enabled;
}

template <typename... KArgs, typename... Args>
static inline void dvt_launch_pdl(
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
  TORCH_CHECK(err == cudaSuccess, "draft_vocab kernel launch failed: ", cudaGetErrorString(err));
}

// Zero-I2F exact 2x E2M1 magnitude decode ({0,1,2,3,4,6,8,12}) via FP32 16.0f bias + sign XOR.
__device__ __forceinline__ float dvt_e2m1_x2_to_f32(uint32_t nib) {
  const uint32_t mag_u = (0xC8643210u >> ((nib & 7u) << 2)) & 0xFu;
  const float mag = __uint_as_float(0x41800000u | (mag_u << 19)) - 16.0f;
  return __uint_as_float(__float_as_uint(mag) ^ ((nib & 8u) << 28));
}

__device__ __forceinline__ float dvt_fp8_e4m3_to_f32(uint8_t bits) {
  __nv_fp8_e4m3 v;
  v.__x = bits;
  return static_cast<float>(v);
}

__device__ __forceinline__ bool dvt_better(float v1, int i1, float v2, int i2) {
  return (v1 > v2) || (v1 == v2 && i1 < i2);
}

__device__ __forceinline__ void dvt_top2_insert(
    float& v1, int& i1, float& v2, int& i2, float cand_v, int cand_i) {
  if (cand_i == i1 || cand_i == i2) return;
  if (dvt_better(cand_v, cand_i, v1, i1)) {
    v2 = v1;
    i2 = i1;
    v1 = cand_v;
    i1 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v2, i2)) {
    v2 = cand_v;
    i2 = cand_i;
  }
}

__device__ __forceinline__ void dvt_top4_insert(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float cand_v, int cand_i) {
  if (cand_i == i1 || cand_i == i2 || cand_i == i3 || cand_i == i4) return;
  if (dvt_better(cand_v, cand_i, v1, i1)) {
    v4 = v3; i4 = i3;
    v3 = v2; i3 = i2;
    v2 = v1; i2 = i1;
    v1 = cand_v; i1 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v2, i2)) {
    v4 = v3; i4 = i3;
    v3 = v2; i3 = i2;
    v2 = cand_v; i2 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v3, i3)) {
    v4 = v3; i4 = i3;
    v3 = cand_v; i3 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v4, i4)) {
    v4 = cand_v; i4 = cand_i;
  }
}

__device__ __forceinline__ void dvt_top6_insert(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float& v5, int& i5, float& v6, int& i6,
    float cand_v, int cand_i) {
  if (cand_i == i1 || cand_i == i2 || cand_i == i3 ||
      cand_i == i4 || cand_i == i5 || cand_i == i6) return;
  if (dvt_better(cand_v, cand_i, v1, i1)) {
    v6 = v5; i6 = i5;
    v5 = v4; i5 = i4;
    v4 = v3; i4 = i3;
    v3 = v2; i3 = i2;
    v2 = v1; i2 = i1;
    v1 = cand_v; i1 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v2, i2)) {
    v6 = v5; i6 = i5;
    v5 = v4; i5 = i4;
    v4 = v3; i4 = i3;
    v3 = v2; i3 = i2;
    v2 = cand_v; i2 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v3, i3)) {
    v6 = v5; i6 = i5;
    v5 = v4; i5 = i4;
    v4 = v3; i4 = i3;
    v3 = cand_v; i3 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v4, i4)) {
    v6 = v5; i6 = i5;
    v5 = v4; i5 = i4;
    v4 = cand_v; i4 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v5, i5)) {
    v6 = v5; i6 = i5;
    v5 = cand_v; i5 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v6, i6)) {
    v6 = cand_v; i6 = cand_i;
  }
}

__device__ __forceinline__ void dvt_top6_merge(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float& v5, int& i5, float& v6, int& i6,
    float ov1, int oi1, float ov2, int oi2,
    float ov3, int oi3, float ov4, int oi4,
    float ov5, int oi5, float ov6, int oi6) {
  if (ov1 >= v6) {
    dvt_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov1, oi1);
    if (ov2 >= v6) {
      dvt_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov2, oi2);
      if (ov3 >= v6) {
        dvt_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov3, oi3);
        if (ov4 >= v6) {
          dvt_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov4, oi4);
          if (ov5 >= v6) {
            dvt_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov5, oi5);
            if (ov6 >= v6) {
              dvt_top6_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, ov6, oi6);
            }
          }
        }
      }
    }
  }
}

__device__ __forceinline__ void dvt_top8_insert(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float& v5, int& i5, float& v6, int& i6,
    float& v7, int& i7, float& v8, int& i8,
    float cand_v, int cand_i) {
  if (cand_i == i1 || cand_i == i2 || cand_i == i3 || cand_i == i4 ||
      cand_i == i5 || cand_i == i6 || cand_i == i7 || cand_i == i8) return;
  if (dvt_better(cand_v, cand_i, v1, i1)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4;
    v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = cand_v; i1 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v2, i2)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4;
    v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = cand_v; i2 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v3, i3)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4;
    v4 = v3; i4 = i3; v3 = cand_v; i3 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v4, i4)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4;
    v4 = cand_v; i4 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v5, i5)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = cand_v; i5 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v6, i6)) {
    v8 = v7; i8 = i7; v7 = v6; i7 = i6; v6 = cand_v; i6 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v7, i7)) {
    v8 = v7; i8 = i7; v7 = cand_v; i7 = cand_i;
  } else if (dvt_better(cand_v, cand_i, v8, i8)) {
    v8 = cand_v; i8 = cand_i;
  }
}

__device__ __forceinline__ void dvt_top8_merge(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float& v5, int& i5, float& v6, int& i6,
    float& v7, int& i7, float& v8, int& i8,
    float ov1, int oi1, float ov2, int oi2,
    float ov3, int oi3, float ov4, int oi4,
    float ov5, int oi5, float ov6, int oi6,
    float ov7, int oi7, float ov8, int oi8) {
  if (ov1 >= v8) {
    dvt_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov1, oi1);
    if (ov2 >= v8) {
      dvt_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov2, oi2);
      if (ov3 >= v8) {
        dvt_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov3, oi3);
        if (ov4 >= v8) {
          dvt_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov4, oi4);
          if (ov5 >= v8) {
            dvt_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov5, oi5);
            if (ov6 >= v8) {
              dvt_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov6, oi6);
              if (ov7 >= v8) {
                dvt_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov7, oi7);
                if (ov8 >= v8) {
                  dvt_top8_insert(v1, i1, v2, i2, v3, i3, v4, i4, v5, i5, v6, i6, v7, i7, v8, i8, ov8, oi8);
                }
              }
            }
          }
        }
      }
    }
  }
}

__device__ __forceinline__ void dvt_top4_merge(
    float& v1, int& i1, float& v2, int& i2,
    float& v3, int& i3, float& v4, int& i4,
    float ov1, int oi1, float ov2, int oi2,
    float ov3, int oi3, float ov4, int oi4) {
  if (ov1 >= v4) {
    dvt_top4_insert(v1, i1, v2, i2, v3, i3, v4, i4, ov1, oi1);
    if (ov2 >= v4) {
      dvt_top4_insert(v1, i1, v2, i2, v3, i3, v4, i4, ov2, oi2);
      if (ov3 >= v4) {
        dvt_top4_insert(v1, i1, v2, i2, v3, i3, v4, i4, ov3, oi3);
        if (ov4 >= v4) {
          dvt_top4_insert(v1, i1, v2, i2, v3, i3, v4, i4, ov4, oi4);
        }
      }
    }
  }
}

// 1-CTA GPU tail updater (~3 us, zero CPU-GPU sync).
// Automatically resets when positions[0] == 0 (start of a new prompt) and appends any unique
// OOV token IDs (in_static_vocab[tok] == 0) from accepted input_ids[0..num_tokens-num_rejected-1]
// PLUS extra_ids[0..num_extra-1] (newly generated bonus token, 0-step lag) into tail_ids[0..max_tail-1].
template <typename IdT>
__global__ void draft_vocab_update_tail_kernel(
    const IdT* __restrict__ input_ids,
    const int64_t* __restrict__ positions,
    const uint8_t* __restrict__ in_static_vocab,
    int32_t* __restrict__ tail_ids,
    int32_t* __restrict__ tail_count,
    int32_t* __restrict__ tail_seen,
    const int32_t* __restrict__ num_rejected,
    const IdT* __restrict__ extra_ids,
    int num_tokens,
    int num_extra,
    int vocab_size,
    int max_tail) {
  __shared__ int s_reset;
  __shared__ int s_old_cnt;
  __shared__ int s_eff_tokens;
  const int tid = threadIdx.x;

  if (tid == 0) {
    s_reset = (num_tokens > 0 && positions[0] == 0) ? 1 : 0;
    int oc = tail_count[0];
    if (oc < 0) oc = 0;
    if (oc > max_tail) oc = max_tail;
    s_old_cnt = oc;
    int eff = num_tokens;
    if (!s_reset && num_rejected != nullptr) {
      int nr = num_rejected[0];
      if (nr > 0) {
        eff = (nr < num_tokens) ? (num_tokens - nr) : 0;
      }
    }
    s_eff_tokens = eff;
  }
  __syncthreads();

  if (s_reset) {
    for (int i = tid; i < s_old_cnt; i += blockDim.x) {
      const int prev_tok = tail_ids[i];
      if (prev_tok >= 0 && prev_tok < vocab_size) {
        tail_seen[prev_tok] = 0;
      }
    }
    for (int i = tid; i < max_tail; i += blockDim.x) {
      tail_ids[i] = -1;
    }
    if (tid == 0) {
      tail_count[0] = 0;
    }
    __syncthreads();
  }

  auto try_add = [&](int64_t raw_tok) {
    if (raw_tok >= 0 && raw_tok < vocab_size) {
      const int tok = static_cast<int>(raw_tok);
      if (in_static_vocab[tok] == 0) {
        if (atomicExch(&tail_seen[tok], 1) == 0) {
          const int slot = atomicAdd(tail_count, 1);
          if (slot < max_tail) {
            tail_ids[slot] = tok;
          } else {
            // Did not fit in tail_ids -> release seen mark so future reset stays clean.
            tail_seen[tok] = 0;
          }
        }
      }
    }
  };

  const int eff_tokens = s_eff_tokens;
  for (int i = tid; i < eff_tokens; i += blockDim.x) {
    try_add(static_cast<int64_t>(input_ids[i]));
  }
  if (extra_ids != nullptr && num_extra > 0) {
    for (int i = tid; i < num_extra; i += blockDim.x) {
      try_add(static_cast<int64_t>(extra_ids[i]));
    }
  }
  __syncthreads();

  if (tid == 0 && tail_count[0] > max_tail) {
    tail_count[0] = max_tail;
  }
}

// Helper: 1 warp computes exact NVFP4 dot product for row `v` against SMEM/global `a_row`.
// Vectorized 2-group (uint4 weights + uint16 scales) loop when num_g is even (e.g. K=6656 -> 416).
__device__ __forceinline__ float dvt_nvfp4_group_dot(
    uint32_t wlo, uint32_t whi, const uint4& a0, const uint4& a1) {
  const __nv_bfloat162* ap0 = reinterpret_cast<const __nv_bfloat162*>(&a0);
  const __nv_bfloat162* ap1 = reinterpret_cast<const __nv_bfloat162*>(&a1);
  float gsum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 af = __bfloat1622float2(ap0[i]);
    gsum = __fmaf_rn(dvt_e2m1_x2_to_f32(wlo & 0xFu), af.x, gsum);
    gsum = __fmaf_rn(dvt_e2m1_x2_to_f32((wlo >> 4) & 0xFu), af.y, gsum);
    wlo >>= 8;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 af = __bfloat1622float2(ap1[i]);
    gsum = __fmaf_rn(dvt_e2m1_x2_to_f32(whi & 0xFu), af.x, gsum);
    gsum = __fmaf_rn(dvt_e2m1_x2_to_f32((whi >> 4) & 0xFu), af.y, gsum);
    whi >>= 8;
  }
  return gsum;
}

__device__ __forceinline__ float dvt_warp_nvfp4_dot(
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
      const float sc0 = dvt_fp8_e4m3_to_f32(static_cast<uint8_t>(s2 & 0xFFu)) * 0.5f;
      const float sc1 = dvt_fp8_e4m3_to_f32(static_cast<uint8_t>(s2 >> 8)) * 0.5f;
      const uint4 a00 = use_smem_a ? s_a0[g0] : __ldg(reinterpret_cast<const uint4*>(a_row) + g0 * 2 + 0);
      const uint4 a01 = use_smem_a ? s_a1[g0] : __ldg(reinterpret_cast<const uint4*>(a_row) + g0 * 2 + 1);
      const uint4 a10 = use_smem_a ? s_a0[g1] : __ldg(reinterpret_cast<const uint4*>(a_row) + g1 * 2 + 0);
      const uint4 a11 = use_smem_a ? s_a1[g1] : __ldg(reinterpret_cast<const uint4*>(a_row) + g1 * 2 + 1);
      acc = __fmaf_rn(dvt_nvfp4_group_dot(w16.x, w16.y, a00, a01), sc0, acc);
      acc = __fmaf_rn(dvt_nvfp4_group_dot(w16.z, w16.w, a10, a11), sc1, acc);
    }
  } else {
    for (int g = lane; g < num_g; g += 32) {
      const float sc_half = dvt_fp8_e4m3_to_f32(__ldg(&s_row[g])) * 0.5f;
      const uint64_t w8 = __ldg(reinterpret_cast<const uint64_t*>(w_row + g * 8));
      const uint4 a0 = use_smem_a ? s_a0[g] : __ldg(reinterpret_cast<const uint4*>(a_row) + g * 2 + 0);
      const uint4 a1 = use_smem_a ? s_a1[g] : __ldg(reinterpret_cast<const uint4*>(a_row) + g * 2 + 1);
      acc = __fmaf_rn(
          dvt_nvfp4_group_dot(static_cast<uint32_t>(w8), static_cast<uint32_t>(w8 >> 32), a0, a1),
          sc_half, acc);
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    acc += __shfl_xor_sync(0xffffffffu, acc, o);
  }
  // Match BF16 static_logits precision + scale.
  return __bfloat162float(__float2bfloat16_rn(acc * total_scale));
}

// Fused static-logits reduction + dynamic prompt-tail exact-NVFP4 scoring.
// Grid: (16, M), Block: 256 threads (8 warps).
// Under SM121 PDL (pdl_mode & 1), Phase 2 (prompt-tail OOV scoring against A/raw_w) runs BEFORE
// griddepcontrol.wait, overlapping with the tail of draft_head; Phase 1 then runs vectorized uint4
// (8 BF16/load) scan of static_logits with exact 8-wide BF16 __hmax2 skip and templated TopK merge.
// TopK = 8 or 6: compile-time specialized (eliminates 5-way runtime branches, fits <=128 regs/thread
// for 2 CTAs/SM = 1 wave at M=6); TopK = 0: generic fallback for all other shapes/modes.
template <int TopK>
__global__ void __launch_bounds__(256, 2) draft_vocab_tail_argmax_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ static_logits,
    const int64_t* __restrict__ vocab_map,
    const int32_t* __restrict__ tail_ids,
    const int32_t* __restrict__ tail_count,
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    const float* __restrict__ global_scale,
    float* __restrict__ out_vals,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_vals2,
    int32_t* __restrict__ out_ids2,
    float* __restrict__ out_vals3,
    int32_t* __restrict__ out_ids3,
    float* __restrict__ out_vals4,
    int32_t* __restrict__ out_ids4,
    float* __restrict__ out_vals5,
    int32_t* __restrict__ out_ids5,
    float* __restrict__ out_vals6,
    int32_t* __restrict__ out_ids6,
    float* __restrict__ out_vals7,
    int32_t* __restrict__ out_ids7,
    float* __restrict__ out_vals8,
    int32_t* __restrict__ out_ids8,
    int top2_max_rows,
    int tree_top_rows,
    int tree_k,
    int V_static,
    int K,
    int max_tail,
    float logit_scale,
    int pdl_mode) {
  __shared__ uint4 s_a0[kS2MaxGroups];  // 8192 B (stride 16 B -> 0 LDS.128 bank conflicts)
  __shared__ uint4 s_a1[kS2MaxGroups];  // 8192 B
  __shared__ float s_vals[kS2Warps];
  __shared__ int32_t s_ids[kS2Warps];
  __shared__ float s_vals2[kS2Warps];
  __shared__ int32_t s_ids2[kS2Warps];
  __shared__ float s_vals3[kS2Warps];
  __shared__ int32_t s_ids3[kS2Warps];
  __shared__ float s_vals4[kS2Warps];
  __shared__ int32_t s_ids4[kS2Warps];
  __shared__ float s_vals5[kS2Warps];
  __shared__ int32_t s_ids5[kS2Warps];
  __shared__ float s_vals6[kS2Warps];
  __shared__ int32_t s_ids6[kS2Warps];
  __shared__ float s_vals7[kS2Warps];
  __shared__ int32_t s_ids7[kS2Warps];
  __shared__ float s_vals8[kS2Warps];
  __shared__ int32_t s_ids8[kS2Warps];

  const int cta_c = blockIdx.x;  // 0 .. 15
  const int m = blockIdx.y;      // 0 .. M-1
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool want_top8 = (TopK == 8) || (TopK == 0 && tree_k >= 8 && m < tree_top_rows &&
                          out_vals3 != nullptr && out_ids3 != nullptr &&
                          out_vals4 != nullptr && out_ids4 != nullptr &&
                          out_vals5 != nullptr && out_ids5 != nullptr &&
                          out_vals6 != nullptr && out_ids6 != nullptr &&
                          out_vals7 != nullptr && out_ids7 != nullptr &&
                          out_vals8 != nullptr && out_ids8 != nullptr);
  const bool want_top6 = (TopK == 6) || (TopK == 0 && !want_top8 && tree_k >= 6 && m < tree_top_rows &&
                          out_vals3 != nullptr && out_ids3 != nullptr &&
                          out_vals4 != nullptr && out_ids4 != nullptr &&
                          out_vals5 != nullptr && out_ids5 != nullptr &&
                          out_vals6 != nullptr && out_ids6 != nullptr);
  const bool want_top4 = (TopK == 0 && !want_top8 && !want_top6 && m < 3 && m < tree_top_rows &&
                          out_vals3 != nullptr && out_ids3 != nullptr &&
                          out_vals4 != nullptr && out_ids4 != nullptr);
  const bool want_top2 = (TopK == 0 && !want_top8 && !want_top6 && !want_top4 && m < top2_max_rows &&
                          out_vals2 != nullptr && out_ids2 != nullptr);

  float best_val = -1e30f;
  int best_id = 0x7fffffff;
  float best2_val = -1e30f;
  int best2_id = 0x7fffffff;
  float best3_val = -1e30f;
  int best3_id = 0x7fffffff;
  float best4_val = -1e30f;
  int best4_id = 0x7fffffff;
  float best5_val = -1e30f;
  int best5_id = 0x7fffffff;
  float best6_val = -1e30f;
  int best6_id = 0x7fffffff;
  float best7_val = -1e30f;
  int best7_id = 0x7fffffff;
  float best8_val = -1e30f;
  int best8_id = 0x7fffffff;

  // Phase 2 FIRST (independent of draft_head's static_logits!): score dynamic prompt-tail OOV tokens
  // via exact NVFP4, overlapping completely with draft_head's tail under SM121 PDL.
  int n_tail = tail_count[0];
  if (n_tail < 0) n_tail = 0;
  if (n_tail > max_tail) n_tail = max_tail;

  if (n_tail > 0) {
    const int num_g = K >> 4;
    const __nv_bfloat16* a_row = A + static_cast<size_t>(m) * K;
    const bool use_smem_a = (num_g <= kS2MaxGroups);
    if (use_smem_a) {
      const uint4* src_u4 = reinterpret_cast<const uint4*>(a_row);
      for (int g = tid; g < num_g; g += kS2Warps * 32) {
        s_a0[g] = __ldg(src_u4 + g * 2 + 0);
        s_a1[g] = __ldg(src_u4 + g * 2 + 1);
      }
      __syncthreads();
    }
    const float total_scale = __ldg(global_scale) * logit_scale;
    const int stride_warps = kS2CTAsPerRow * kS2Warps;  // 128 warps
    for (int ti = cta_c * kS2Warps + warp; ti < n_tail; ti += stride_warps) {
      const int tok = __ldg(&tail_ids[ti]);
      if (tok >= 0) {
        const float val = dvt_warp_nvfp4_dot(
            a_row, s_a0, s_a1, use_smem_a, raw_w, raw_s, tok, K, num_g, lane, total_scale);
        if (lane == 0) {
          if constexpr (TopK == 8) {
            if (val >= best8_val) {
              dvt_top8_insert(best_val, best_id, best2_val, best2_id,
                              best3_val, best3_id, best4_val, best4_id,
                              best5_val, best5_id, best6_val, best6_id,
                              best7_val, best7_id, best8_val, best8_id, val, tok);
            }
          } else if constexpr (TopK == 6) {
            if (val >= best6_val) {
              dvt_top6_insert(best_val, best_id, best2_val, best2_id,
                              best3_val, best3_id, best4_val, best4_id,
                              best5_val, best5_id, best6_val, best6_id, val, tok);
            }
          } else {
            if (want_top8) {
              dvt_top8_insert(best_val, best_id, best2_val, best2_id,
                              best3_val, best3_id, best4_val, best4_id,
                              best5_val, best5_id, best6_val, best6_id,
                              best7_val, best7_id, best8_val, best8_id, val, tok);
            } else if (want_top6) {
              dvt_top6_insert(best_val, best_id, best2_val, best2_id,
                              best3_val, best3_id, best4_val, best4_id,
                              best5_val, best5_id, best6_val, best6_id, val, tok);
            } else if (want_top4) {
              dvt_top4_insert(best_val, best_id, best2_val, best2_id,
                              best3_val, best3_id, best4_val, best4_id, val, tok);
            } else if (want_top2) {
              dvt_top2_insert(best_val, best_id, best2_val, best2_id, val, tok);
            } else if (dvt_better(val, tok, best_val, best_id)) {
              best_val = val;
              best_id = tok;
            }
          }
        }
      }
    }
  }

  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  // Phase 1: Reduce static_logits[m, chunk] from L2 cache via vectorized uint4 (8 BF16/load).
  const __nv_bfloat16* sl_row = static_logits + static_cast<size_t>(m) * V_static;
  const bool vec8_ok = ((V_static & 7) == 0) && ((reinterpret_cast<uintptr_t>(sl_row) & 15u) == 0u);
  const int chunk = vec8_ok ? ((((V_static + kS2CTAsPerRow - 1) / kS2CTAsPerRow) + 7) & ~7)
                            : ((V_static + kS2CTAsPerRow - 1) / kS2CTAsPerRow);
  const int j_start = cta_c * chunk;
  const int j_end = (j_start + chunk < V_static) ? (j_start + chunk) : V_static;

  auto consider_static = [&](float v, int j_idx) {
    if constexpr (TopK == 8) {
      if (v >= best8_val) {
        const int orig_id = static_cast<int>(__ldg(&vocab_map[j_idx]));
        dvt_top8_insert(best_val, best_id, best2_val, best2_id,
                        best3_val, best3_id, best4_val, best4_id,
                        best5_val, best5_id, best6_val, best6_id,
                        best7_val, best7_id, best8_val, best8_id, v, orig_id);
      }
    } else if constexpr (TopK == 6) {
      if (v >= best6_val) {
        const int orig_id = static_cast<int>(__ldg(&vocab_map[j_idx]));
        dvt_top6_insert(best_val, best_id, best2_val, best2_id,
                        best3_val, best3_id, best4_val, best4_id,
                        best5_val, best5_id, best6_val, best6_id, v, orig_id);
      }
    } else {
      if (want_top8) {
        if (v >= best8_val) {
          const int orig_id = static_cast<int>(__ldg(&vocab_map[j_idx]));
          dvt_top8_insert(best_val, best_id, best2_val, best2_id,
                          best3_val, best3_id, best4_val, best4_id,
                          best5_val, best5_id, best6_val, best6_id,
                          best7_val, best7_id, best8_val, best8_id, v, orig_id);
        }
      } else if (want_top6) {
        if (v >= best6_val) {
          const int orig_id = static_cast<int>(__ldg(&vocab_map[j_idx]));
          dvt_top6_insert(best_val, best_id, best2_val, best2_id,
                          best3_val, best3_id, best4_val, best4_id,
                          best5_val, best5_id, best6_val, best6_id, v, orig_id);
        }
      } else if (want_top4) {
        if (v >= best4_val) {
          const int orig_id = static_cast<int>(__ldg(&vocab_map[j_idx]));
          dvt_top4_insert(best_val, best_id, best2_val, best2_id,
                          best3_val, best3_id, best4_val, best4_id, v, orig_id);
        }
      } else if (want_top2) {
        if (v >= best2_val) {
          const int orig_id = static_cast<int>(__ldg(&vocab_map[j_idx]));
          dvt_top2_insert(best_val, best_id, best2_val, best2_id, v, orig_id);
        }
      } else if (v >= best_val) {
        const int orig_id = static_cast<int>(__ldg(&vocab_map[j_idx]));
        if (dvt_better(v, orig_id, best_val, best_id)) {
          best_val = v;
          best_id = orig_id;
        }
      }
    }
  };

  if (vec8_ok) {
    const uint4* sl_u4 = reinterpret_cast<const uint4*>(sl_row);
    const int j8_start = j_start >> 3;
    const int j8_end = j_end >> 3;
    for (int j8 = j8_start + tid; j8 < j8_end; j8 += blockDim.x) {
      const uint4 v4 = __ldg(&sl_u4[j8]);
      const __nv_bfloat162* vp = reinterpret_cast<const __nv_bfloat162*>(&v4);
      const int j_base = j8 << 3;
      if constexpr (TopK == 8 || TopK == 6) {
        // Exact 3-instruction BF16 8-wide max skip (100% bit-exact incl. NaN/Inf/+-0/ties).
        const __nv_bfloat162 m01 = __hmax2(vp[0], vp[1]);
        const __nv_bfloat162 m23 = __hmax2(vp[2], vp[3]);
        const __nv_bfloat162 m4 = __hmax2(m01, m23);
        const float2 mf = __bfloat1622float2(m4);
        const float max8 = fmaxf(mf.x, mf.y);
        const float thresh = (TopK == 8) ? best8_val : best6_val;
        if (max8 >= thresh) {
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            const float2 f = __bfloat1622float2(vp[p]);
            consider_static(f.x, j_base + 2 * p + 0);
            consider_static(f.y, j_base + 2 * p + 1);
          }
        }
      } else {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const float2 f = __bfloat1622float2(vp[p]);
          consider_static(f.x, j_base + 2 * p + 0);
          consider_static(f.y, j_base + 2 * p + 1);
        }
      }
    }
  } else {
    for (int j = j_start + tid; j < j_end; j += blockDim.x) {
      const float v = __bfloat162float(__ldg(&sl_row[j]));
      consider_static(v, j);
    }
  }

  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }

  // Warp reduce merged Phase 1 + Phase 2 -> lane 0 holds warp's best candidates.
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    if constexpr (TopK == 8) {
      const float ov = __shfl_down_sync(0xffffffffu, best_val, o);
      const int oi = __shfl_down_sync(0xffffffffu, best_id, o);
      const float ov2 = __shfl_down_sync(0xffffffffu, best2_val, o);
      const int oi2 = __shfl_down_sync(0xffffffffu, best2_id, o);
      const float ov3 = __shfl_down_sync(0xffffffffu, best3_val, o);
      const int oi3 = __shfl_down_sync(0xffffffffu, best3_id, o);
      const float ov4 = __shfl_down_sync(0xffffffffu, best4_val, o);
      const int oi4 = __shfl_down_sync(0xffffffffu, best4_id, o);
      const float ov5 = __shfl_down_sync(0xffffffffu, best5_val, o);
      const int oi5 = __shfl_down_sync(0xffffffffu, best5_id, o);
      const float ov6 = __shfl_down_sync(0xffffffffu, best6_val, o);
      const int oi6 = __shfl_down_sync(0xffffffffu, best6_id, o);
      const float ov7 = __shfl_down_sync(0xffffffffu, best7_val, o);
      const int oi7 = __shfl_down_sync(0xffffffffu, best7_id, o);
      const float ov8 = __shfl_down_sync(0xffffffffu, best8_val, o);
      const int oi8 = __shfl_down_sync(0xffffffffu, best8_id, o);
      if (lane < o) {
        dvt_top8_merge(best_val, best_id, best2_val, best2_id, best3_val, best3_id,
                       best4_val, best4_id, best5_val, best5_id, best6_val, best6_id,
                       best7_val, best7_id, best8_val, best8_id,
                       ov, oi, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6, ov7, oi7, ov8, oi8);
      }
    } else if constexpr (TopK == 6) {
      const float ov = __shfl_down_sync(0xffffffffu, best_val, o);
      const int oi = __shfl_down_sync(0xffffffffu, best_id, o);
      const float ov2 = __shfl_down_sync(0xffffffffu, best2_val, o);
      const int oi2 = __shfl_down_sync(0xffffffffu, best2_id, o);
      const float ov3 = __shfl_down_sync(0xffffffffu, best3_val, o);
      const int oi3 = __shfl_down_sync(0xffffffffu, best3_id, o);
      const float ov4 = __shfl_down_sync(0xffffffffu, best4_val, o);
      const int oi4 = __shfl_down_sync(0xffffffffu, best4_id, o);
      const float ov5 = __shfl_down_sync(0xffffffffu, best5_val, o);
      const int oi5 = __shfl_down_sync(0xffffffffu, best5_id, o);
      const float ov6 = __shfl_down_sync(0xffffffffu, best6_val, o);
      const int oi6 = __shfl_down_sync(0xffffffffu, best6_id, o);
      if (lane < o) {
        dvt_top6_merge(best_val, best_id, best2_val, best2_id, best3_val, best3_id,
                       best4_val, best4_id, best5_val, best5_id, best6_val, best6_id,
                       ov, oi, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6);
      }
    } else {
      const float ov = __shfl_xor_sync(0xffffffffu, best_val, o);
      const int oi = __shfl_xor_sync(0xffffffffu, best_id, o);
      if (want_top8) {
        const float ov2 = __shfl_xor_sync(0xffffffffu, best2_val, o);
        const int oi2 = __shfl_xor_sync(0xffffffffu, best2_id, o);
        const float ov3 = __shfl_xor_sync(0xffffffffu, best3_val, o);
        const int oi3 = __shfl_xor_sync(0xffffffffu, best3_id, o);
        const float ov4 = __shfl_xor_sync(0xffffffffu, best4_val, o);
        const int oi4 = __shfl_xor_sync(0xffffffffu, best4_id, o);
        const float ov5 = __shfl_xor_sync(0xffffffffu, best5_val, o);
        const int oi5 = __shfl_xor_sync(0xffffffffu, best5_id, o);
        const float ov6 = __shfl_xor_sync(0xffffffffu, best6_val, o);
        const int oi6 = __shfl_xor_sync(0xffffffffu, best6_id, o);
        const float ov7 = __shfl_xor_sync(0xffffffffu, best7_val, o);
        const int oi7 = __shfl_xor_sync(0xffffffffu, best7_id, o);
        const float ov8 = __shfl_xor_sync(0xffffffffu, best8_val, o);
        const int oi8 = __shfl_xor_sync(0xffffffffu, best8_id, o);
        dvt_top8_merge(best_val, best_id, best2_val, best2_id, best3_val, best3_id,
                       best4_val, best4_id, best5_val, best5_id, best6_val, best6_id,
                       best7_val, best7_id, best8_val, best8_id,
                       ov, oi, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6, ov7, oi7, ov8, oi8);
      } else if (want_top6) {
        const float ov2 = __shfl_xor_sync(0xffffffffu, best2_val, o);
        const int oi2 = __shfl_xor_sync(0xffffffffu, best2_id, o);
        const float ov3 = __shfl_xor_sync(0xffffffffu, best3_val, o);
        const int oi3 = __shfl_xor_sync(0xffffffffu, best3_id, o);
        const float ov4 = __shfl_xor_sync(0xffffffffu, best4_val, o);
        const int oi4 = __shfl_xor_sync(0xffffffffu, best4_id, o);
        const float ov5 = __shfl_xor_sync(0xffffffffu, best5_val, o);
        const int oi5 = __shfl_xor_sync(0xffffffffu, best5_id, o);
        const float ov6 = __shfl_xor_sync(0xffffffffu, best6_val, o);
        const int oi6 = __shfl_xor_sync(0xffffffffu, best6_id, o);
        dvt_top6_merge(best_val, best_id, best2_val, best2_id, best3_val, best3_id,
                       best4_val, best4_id, best5_val, best5_id, best6_val, best6_id,
                       ov, oi, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6);
      } else if (want_top4) {
        const float ov2 = __shfl_xor_sync(0xffffffffu, best2_val, o);
        const int oi2 = __shfl_xor_sync(0xffffffffu, best2_id, o);
        const float ov3 = __shfl_xor_sync(0xffffffffu, best3_val, o);
        const int oi3 = __shfl_xor_sync(0xffffffffu, best3_id, o);
        const float ov4 = __shfl_xor_sync(0xffffffffu, best4_val, o);
        const int oi4 = __shfl_xor_sync(0xffffffffu, best4_id, o);
        dvt_top4_merge(best_val, best_id, best2_val, best2_id, best3_val, best3_id, best4_val, best4_id,
                       ov, oi, ov2, oi2, ov3, oi3, ov4, oi4);
      } else if (want_top2) {
        const float ov2 = __shfl_xor_sync(0xffffffffu, best2_val, o);
        const int oi2 = __shfl_xor_sync(0xffffffffu, best2_id, o);
        if (ov >= best2_val) {
          dvt_top2_insert(best_val, best_id, best2_val, best2_id, ov, oi);
          if (ov2 >= best2_val) {
            dvt_top2_insert(best_val, best_id, best2_val, best2_id, ov2, oi2);
          }
        }
      } else if (dvt_better(ov, oi, best_val, best_id)) {
        best_val = ov;
        best_id = oi;
      }
    }
  }

  if (lane == 0) {
    s_vals[warp] = best_val;
    s_ids[warp] = best_id;
    if constexpr (TopK == 8) {
      s_vals2[warp] = best2_val; s_ids2[warp] = best2_id;
      s_vals3[warp] = best3_val; s_ids3[warp] = best3_id;
      s_vals4[warp] = best4_val; s_ids4[warp] = best4_id;
      s_vals5[warp] = best5_val; s_ids5[warp] = best5_id;
      s_vals6[warp] = best6_val; s_ids6[warp] = best6_id;
      s_vals7[warp] = best7_val; s_ids7[warp] = best7_id;
      s_vals8[warp] = best8_val; s_ids8[warp] = best8_id;
    } else if constexpr (TopK == 6) {
      s_vals2[warp] = best2_val; s_ids2[warp] = best2_id;
      s_vals3[warp] = best3_val; s_ids3[warp] = best3_id;
      s_vals4[warp] = best4_val; s_ids4[warp] = best4_id;
      s_vals5[warp] = best5_val; s_ids5[warp] = best5_id;
      s_vals6[warp] = best6_val; s_ids6[warp] = best6_id;
    } else {
      if (want_top8 || want_top6 || want_top4 || want_top2) {
        s_vals2[warp] = best2_val;
        s_ids2[warp] = best2_id;
      }
      if (want_top8 || want_top6 || want_top4) {
        s_vals3[warp] = best3_val;
        s_ids3[warp] = best3_id;
        s_vals4[warp] = best4_val;
        s_ids4[warp] = best4_id;
      }
      if (want_top8 || want_top6) {
        s_vals5[warp] = best5_val;
        s_ids5[warp] = best5_id;
        s_vals6[warp] = best6_val;
        s_ids6[warp] = best6_id;
      }
      if (want_top8) {
        s_vals7[warp] = best7_val;
        s_ids7[warp] = best7_id;
        s_vals8[warp] = best8_val;
        s_ids8[warp] = best8_id;
      }
    }
  }
  __syncthreads();

  if (warp == 0 && lane < kS2Warps) {
    float bv = s_vals[lane];
    int bi = s_ids[lane];
    float bv2 = (TopK == 8 || TopK == 6 || want_top8 || want_top6 || want_top4 || want_top2) ? s_vals2[lane] : -1e30f;
    int bi2 = (TopK == 8 || TopK == 6 || want_top8 || want_top6 || want_top4 || want_top2) ? s_ids2[lane] : 0x7fffffff;
    float bv3 = (TopK == 8 || TopK == 6 || want_top8 || want_top6 || want_top4) ? s_vals3[lane] : -1e30f;
    int bi3 = (TopK == 8 || TopK == 6 || want_top8 || want_top6 || want_top4) ? s_ids3[lane] : 0x7fffffff;
    float bv4 = (TopK == 8 || TopK == 6 || want_top8 || want_top6 || want_top4) ? s_vals4[lane] : -1e30f;
    int bi4 = (TopK == 8 || TopK == 6 || want_top8 || want_top6 || want_top4) ? s_ids4[lane] : 0x7fffffff;
    float bv5 = (TopK == 8 || TopK == 6 || want_top8 || want_top6) ? s_vals5[lane] : -1e30f;
    int bi5 = (TopK == 8 || TopK == 6 || want_top8 || want_top6) ? s_ids5[lane] : 0x7fffffff;
    float bv6 = (TopK == 8 || TopK == 6 || want_top8 || want_top6) ? s_vals6[lane] : -1e30f;
    int bi6 = (TopK == 8 || TopK == 6 || want_top8 || want_top6) ? s_ids6[lane] : 0x7fffffff;
    float bv7 = (TopK == 8 || want_top8) ? s_vals7[lane] : -1e30f;
    int bi7 = (TopK == 8 || want_top8) ? s_ids7[lane] : 0x7fffffff;
    float bv8 = (TopK == 8 || want_top8) ? s_vals8[lane] : -1e30f;
    int bi8 = (TopK == 8 || want_top8) ? s_ids8[lane] : 0x7fffffff;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
      if constexpr (TopK == 8) {
        const float ov = __shfl_down_sync(0xffu, bv, o);
        const int oi = __shfl_down_sync(0xffu, bi, o);
        const float ov2 = __shfl_down_sync(0xffu, bv2, o);
        const int oi2 = __shfl_down_sync(0xffu, bi2, o);
        const float ov3 = __shfl_down_sync(0xffu, bv3, o);
        const int oi3 = __shfl_down_sync(0xffu, bi3, o);
        const float ov4 = __shfl_down_sync(0xffu, bv4, o);
        const int oi4 = __shfl_down_sync(0xffu, bi4, o);
        const float ov5 = __shfl_down_sync(0xffu, bv5, o);
        const int oi5 = __shfl_down_sync(0xffu, bi5, o);
        const float ov6 = __shfl_down_sync(0xffu, bv6, o);
        const int oi6 = __shfl_down_sync(0xffu, bi6, o);
        const float ov7 = __shfl_down_sync(0xffu, bv7, o);
        const int oi7 = __shfl_down_sync(0xffu, bi7, o);
        const float ov8 = __shfl_down_sync(0xffu, bv8, o);
        const int oi8 = __shfl_down_sync(0xffu, bi8, o);
        if (lane < o) {
          dvt_top8_merge(bv, bi, bv2, bi2, bv3, bi3, bv4, bi4, bv5, bi5, bv6, bi6, bv7, bi7, bv8, bi8,
                         ov, oi, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6, ov7, oi7, ov8, oi8);
        }
      } else if constexpr (TopK == 6) {
        const float ov = __shfl_down_sync(0xffu, bv, o);
        const int oi = __shfl_down_sync(0xffu, bi, o);
        const float ov2 = __shfl_down_sync(0xffu, bv2, o);
        const int oi2 = __shfl_down_sync(0xffu, bi2, o);
        const float ov3 = __shfl_down_sync(0xffu, bv3, o);
        const int oi3 = __shfl_down_sync(0xffu, bi3, o);
        const float ov4 = __shfl_down_sync(0xffu, bv4, o);
        const int oi4 = __shfl_down_sync(0xffu, bi4, o);
        const float ov5 = __shfl_down_sync(0xffu, bv5, o);
        const int oi5 = __shfl_down_sync(0xffu, bi5, o);
        const float ov6 = __shfl_down_sync(0xffu, bv6, o);
        const int oi6 = __shfl_down_sync(0xffu, bi6, o);
        if (lane < o) {
          dvt_top6_merge(bv, bi, bv2, bi2, bv3, bi3, bv4, bi4, bv5, bi5, bv6, bi6,
                         ov, oi, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6);
        }
      } else {
        const float ov = __shfl_xor_sync(0xffu, bv, o);
        const int oi = __shfl_xor_sync(0xffu, bi, o);
        if (want_top8) {
          const float ov2 = __shfl_xor_sync(0xffu, bv2, o);
          const int oi2 = __shfl_xor_sync(0xffu, bi2, o);
          const float ov3 = __shfl_xor_sync(0xffu, bv3, o);
          const int oi3 = __shfl_xor_sync(0xffu, bi3, o);
          const float ov4 = __shfl_xor_sync(0xffu, bv4, o);
          const int oi4 = __shfl_xor_sync(0xffu, bi4, o);
          const float ov5 = __shfl_xor_sync(0xffu, bv5, o);
          const int oi5 = __shfl_xor_sync(0xffu, bi5, o);
          const float ov6 = __shfl_xor_sync(0xffu, bv6, o);
          const int oi6 = __shfl_xor_sync(0xffu, bi6, o);
          const float ov7 = __shfl_xor_sync(0xffu, bv7, o);
          const int oi7 = __shfl_xor_sync(0xffu, bi7, o);
          const float ov8 = __shfl_xor_sync(0xffu, bv8, o);
          const int oi8 = __shfl_xor_sync(0xffu, bi8, o);
          dvt_top8_merge(bv, bi, bv2, bi2, bv3, bi3, bv4, bi4, bv5, bi5, bv6, bi6, bv7, bi7, bv8, bi8,
                         ov, oi, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6, ov7, oi7, ov8, oi8);
        } else if (want_top6) {
          const float ov2 = __shfl_xor_sync(0xffu, bv2, o);
          const int oi2 = __shfl_xor_sync(0xffu, bi2, o);
          const float ov3 = __shfl_xor_sync(0xffu, bv3, o);
          const int oi3 = __shfl_xor_sync(0xffu, bi3, o);
          const float ov4 = __shfl_xor_sync(0xffu, bv4, o);
          const int oi4 = __shfl_xor_sync(0xffu, bi4, o);
          const float ov5 = __shfl_xor_sync(0xffu, bv5, o);
          const int oi5 = __shfl_xor_sync(0xffu, bi5, o);
          const float ov6 = __shfl_xor_sync(0xffu, bv6, o);
          const int oi6 = __shfl_xor_sync(0xffu, bi6, o);
          dvt_top6_merge(bv, bi, bv2, bi2, bv3, bi3, bv4, bi4, bv5, bi5, bv6, bi6,
                         ov, oi, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6);
        } else if (want_top4) {
          const float ov2 = __shfl_xor_sync(0xffu, bv2, o);
          const int oi2 = __shfl_xor_sync(0xffu, bi2, o);
          const float ov3 = __shfl_xor_sync(0xffu, bv3, o);
          const int oi3 = __shfl_xor_sync(0xffu, bi3, o);
          const float ov4 = __shfl_xor_sync(0xffu, bv4, o);
          const int oi4 = __shfl_xor_sync(0xffu, bi4, o);
          dvt_top4_merge(bv, bi, bv2, bi2, bv3, bi3, bv4, bi4, ov, oi, ov2, oi2, ov3, oi3, ov4, oi4);
        } else if (want_top2) {
          const float ov2 = __shfl_xor_sync(0xffu, bv2, o);
          const int oi2 = __shfl_xor_sync(0xffu, bi2, o);
          if (ov >= bv2) {
            dvt_top2_insert(bv, bi, bv2, bi2, ov, oi);
            if (ov2 >= bv2) {
              dvt_top2_insert(bv, bi, bv2, bi2, ov2, oi2);
            }
          }
        } else if (dvt_better(ov, oi, bv, bi)) {
          bv = ov;
          bi = oi;
        }
      }
    }
    if (lane == 0) {
      out_vals[m * kS2CTAsPerRow + cta_c] = bv;
      out_ids[m * kS2CTAsPerRow + cta_c] = bi;
      if constexpr (TopK == 8) {
        out_vals2[m * kS2CTAsPerRow + cta_c] = bv2; out_ids2[m * kS2CTAsPerRow + cta_c] = bi2;
        out_vals3[m * kS2CTAsPerRow + cta_c] = bv3; out_ids3[m * kS2CTAsPerRow + cta_c] = bi3;
        out_vals4[m * kS2CTAsPerRow + cta_c] = bv4; out_ids4[m * kS2CTAsPerRow + cta_c] = bi4;
        out_vals5[m * kS2CTAsPerRow + cta_c] = bv5; out_ids5[m * kS2CTAsPerRow + cta_c] = bi5;
        out_vals6[m * kS2CTAsPerRow + cta_c] = bv6; out_ids6[m * kS2CTAsPerRow + cta_c] = bi6;
        out_vals7[m * kS2CTAsPerRow + cta_c] = bv7; out_ids7[m * kS2CTAsPerRow + cta_c] = bi7;
        out_vals8[m * kS2CTAsPerRow + cta_c] = bv8; out_ids8[m * kS2CTAsPerRow + cta_c] = bi8;
      } else if constexpr (TopK == 6) {
        out_vals2[m * kS2CTAsPerRow + cta_c] = bv2; out_ids2[m * kS2CTAsPerRow + cta_c] = bi2;
        out_vals3[m * kS2CTAsPerRow + cta_c] = bv3; out_ids3[m * kS2CTAsPerRow + cta_c] = bi3;
        out_vals4[m * kS2CTAsPerRow + cta_c] = bv4; out_ids4[m * kS2CTAsPerRow + cta_c] = bi4;
        out_vals5[m * kS2CTAsPerRow + cta_c] = bv5; out_ids5[m * kS2CTAsPerRow + cta_c] = bi5;
        out_vals6[m * kS2CTAsPerRow + cta_c] = bv6; out_ids6[m * kS2CTAsPerRow + cta_c] = bi6;
      } else {
        if (want_top8 || want_top6 || want_top4 || want_top2) {
          out_vals2[m * kS2CTAsPerRow + cta_c] = bv2;
          out_ids2[m * kS2CTAsPerRow + cta_c] = bi2;
        }
        if (want_top8 || want_top6 || want_top4) {
          out_vals3[m * kS2CTAsPerRow + cta_c] = bv3;
          out_ids3[m * kS2CTAsPerRow + cta_c] = bi3;
          out_vals4[m * kS2CTAsPerRow + cta_c] = bv4;
          out_ids4[m * kS2CTAsPerRow + cta_c] = bi4;
        }
        if (want_top8 || want_top6) {
          out_vals5[m * kS2CTAsPerRow + cta_c] = bv5;
          out_ids5[m * kS2CTAsPerRow + cta_c] = bi5;
          out_vals6[m * kS2CTAsPerRow + cta_c] = bv6;
          out_ids6[m * kS2CTAsPerRow + cta_c] = bi6;
        }
        if (want_top8) {
          out_vals7[m * kS2CTAsPerRow + cta_c] = bv7;
          out_ids7[m * kS2CTAsPerRow + cta_c] = bi7;
          out_vals8[m * kS2CTAsPerRow + cta_c] = bv8;
          out_ids8[m * kS2CTAsPerRow + cta_c] = bi8;
        }
      }
    }
  }
}

template <int TopK>
__global__ void draft_vocab_final_reduce_kernel(
    const float* __restrict__ in_vals,
    const int32_t* __restrict__ in_ids,
    const float* __restrict__ in_vals2,
    const int32_t* __restrict__ in_ids2,
    const float* __restrict__ in_vals3,
    const int32_t* __restrict__ in_ids3,
    const float* __restrict__ in_vals4,
    const int32_t* __restrict__ in_ids4,
    const float* __restrict__ in_vals5,
    const int32_t* __restrict__ in_ids5,
    const float* __restrict__ in_vals6,
    const int32_t* __restrict__ in_ids6,
    const float* __restrict__ in_vals7,
    const int32_t* __restrict__ in_ids7,
    const float* __restrict__ in_vals8,
    const int32_t* __restrict__ in_ids8,
    int64_t* __restrict__ out_tokens,
    int64_t* __restrict__ out_top2,
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
  if (lane >= kS2CTAsPerRow) return;
  const bool want_top8 = (TopK == 8) || (TopK == 0 && tree_k >= 8 && m < tree_top_rows && out_tree_top != nullptr &&
                          in_vals3 != nullptr && in_ids3 != nullptr &&
                          in_vals4 != nullptr && in_ids4 != nullptr &&
                          in_vals5 != nullptr && in_ids5 != nullptr &&
                          in_vals6 != nullptr && in_ids6 != nullptr &&
                          in_vals7 != nullptr && in_ids7 != nullptr &&
                          in_vals8 != nullptr && in_ids8 != nullptr);
  const bool want_top6 = (TopK == 6) || (TopK == 0 && !want_top8 && tree_k >= 6 && m < tree_top_rows && out_tree_top != nullptr &&
                          in_vals3 != nullptr && in_ids3 != nullptr &&
                          in_vals4 != nullptr && in_ids4 != nullptr &&
                          in_vals5 != nullptr && in_ids5 != nullptr &&
                          in_vals6 != nullptr && in_ids6 != nullptr);
  const bool want_top4 = (TopK == 0 && !want_top8 && !want_top6 && m < 3 && m < tree_top_rows && out_tree_top != nullptr &&
                          in_vals3 != nullptr && in_ids3 != nullptr &&
                          in_vals4 != nullptr && in_ids4 != nullptr);
  const bool want_top2 = (TopK == 0 && !want_top8 && !want_top6 && !want_top4 &&
                          ((m < top2_rows && out_top2 != nullptr) ||
                           (m < tree_top_rows && out_tree_top != nullptr)) &&
                          in_vals2 != nullptr && in_ids2 != nullptr);
  float bv = in_vals[m * kS2CTAsPerRow + lane];
  int bi = in_ids[m * kS2CTAsPerRow + lane];
  float bv2 = (TopK == 8 || TopK == 6 || want_top8 || want_top6 || want_top4 || want_top2) ? in_vals2[m * kS2CTAsPerRow + lane] : -1e30f;
  int bi2 = (TopK == 8 || TopK == 6 || want_top8 || want_top6 || want_top4 || want_top2) ? in_ids2[m * kS2CTAsPerRow + lane] : 0x7fffffff;
  float bv3 = (TopK == 8 || TopK == 6 || want_top8 || want_top6 || want_top4) ? in_vals3[m * kS2CTAsPerRow + lane] : -1e30f;
  int bi3 = (TopK == 8 || TopK == 6 || want_top8 || want_top6 || want_top4) ? in_ids3[m * kS2CTAsPerRow + lane] : 0x7fffffff;
  float bv4 = (TopK == 8 || TopK == 6 || want_top8 || want_top6 || want_top4) ? in_vals4[m * kS2CTAsPerRow + lane] : -1e30f;
  int bi4 = (TopK == 8 || TopK == 6 || want_top8 || want_top6 || want_top4) ? in_ids4[m * kS2CTAsPerRow + lane] : 0x7fffffff;
  float bv5 = (TopK == 8 || TopK == 6 || want_top8 || want_top6) ? in_vals5[m * kS2CTAsPerRow + lane] : -1e30f;
  int bi5 = (TopK == 8 || TopK == 6 || want_top8 || want_top6) ? in_ids5[m * kS2CTAsPerRow + lane] : 0x7fffffff;
  float bv6 = (TopK == 8 || TopK == 6 || want_top8 || want_top6) ? in_vals6[m * kS2CTAsPerRow + lane] : -1e30f;
  int bi6 = (TopK == 8 || TopK == 6 || want_top8 || want_top6) ? in_ids6[m * kS2CTAsPerRow + lane] : 0x7fffffff;
  float bv7 = (TopK == 8 || want_top8) ? in_vals7[m * kS2CTAsPerRow + lane] : -1e30f;
  int bi7 = (TopK == 8 || want_top8) ? in_ids7[m * kS2CTAsPerRow + lane] : 0x7fffffff;
  float bv8 = (TopK == 8 || want_top8) ? in_vals8[m * kS2CTAsPerRow + lane] : -1e30f;
  int bi8 = (TopK == 8 || want_top8) ? in_ids8[m * kS2CTAsPerRow + lane] : 0x7fffffff;
#pragma unroll
  for (int o = kS2CTAsPerRow / 2; o > 0; o >>= 1) {
    if constexpr (TopK == 8) {
      const float ov = __shfl_down_sync(0xffffu, bv, o);
      const int oi = __shfl_down_sync(0xffffu, bi, o);
      const float ov2 = __shfl_down_sync(0xffffu, bv2, o);
      const int oi2 = __shfl_down_sync(0xffffu, bi2, o);
      const float ov3 = __shfl_down_sync(0xffffu, bv3, o);
      const int oi3 = __shfl_down_sync(0xffffu, bi3, o);
      const float ov4 = __shfl_down_sync(0xffffu, bv4, o);
      const int oi4 = __shfl_down_sync(0xffffu, bi4, o);
      const float ov5 = __shfl_down_sync(0xffffu, bv5, o);
      const int oi5 = __shfl_down_sync(0xffffu, bi5, o);
      const float ov6 = __shfl_down_sync(0xffffu, bv6, o);
      const int oi6 = __shfl_down_sync(0xffffu, bi6, o);
      const float ov7 = __shfl_down_sync(0xffffu, bv7, o);
      const int oi7 = __shfl_down_sync(0xffffu, bi7, o);
      const float ov8 = __shfl_down_sync(0xffffu, bv8, o);
      const int oi8 = __shfl_down_sync(0xffffu, bi8, o);
      if (lane < o) {
        dvt_top8_merge(bv, bi, bv2, bi2, bv3, bi3, bv4, bi4, bv5, bi5, bv6, bi6, bv7, bi7, bv8, bi8,
                       ov, oi, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6, ov7, oi7, ov8, oi8);
      }
    } else if constexpr (TopK == 6) {
      const float ov = __shfl_down_sync(0xffffu, bv, o);
      const int oi = __shfl_down_sync(0xffffu, bi, o);
      const float ov2 = __shfl_down_sync(0xffffu, bv2, o);
      const int oi2 = __shfl_down_sync(0xffffu, bi2, o);
      const float ov3 = __shfl_down_sync(0xffffu, bv3, o);
      const int oi3 = __shfl_down_sync(0xffffu, bi3, o);
      const float ov4 = __shfl_down_sync(0xffffu, bv4, o);
      const int oi4 = __shfl_down_sync(0xffffu, bi4, o);
      const float ov5 = __shfl_down_sync(0xffffu, bv5, o);
      const int oi5 = __shfl_down_sync(0xffffu, bi5, o);
      const float ov6 = __shfl_down_sync(0xffffu, bv6, o);
      const int oi6 = __shfl_down_sync(0xffffu, bi6, o);
      if (lane < o) {
        dvt_top6_merge(bv, bi, bv2, bi2, bv3, bi3, bv4, bi4, bv5, bi5, bv6, bi6,
                       ov, oi, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6);
      }
    } else {
      const float ov = __shfl_xor_sync(0xffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffu, bi, o);
      if (want_top8) {
        const float ov2 = __shfl_xor_sync(0xffffu, bv2, o);
        const int oi2 = __shfl_xor_sync(0xffffu, bi2, o);
        const float ov3 = __shfl_xor_sync(0xffffu, bv3, o);
        const int oi3 = __shfl_xor_sync(0xffffu, bi3, o);
        const float ov4 = __shfl_xor_sync(0xffffu, bv4, o);
        const int oi4 = __shfl_xor_sync(0xffffu, bi4, o);
        const float ov5 = __shfl_xor_sync(0xffffu, bv5, o);
        const int oi5 = __shfl_xor_sync(0xffffu, bi5, o);
        const float ov6 = __shfl_xor_sync(0xffffu, bv6, o);
        const int oi6 = __shfl_xor_sync(0xffffu, bi6, o);
        const float ov7 = __shfl_xor_sync(0xffffu, bv7, o);
        const int oi7 = __shfl_xor_sync(0xffffu, bi7, o);
        const float ov8 = __shfl_xor_sync(0xffffu, bv8, o);
        const int oi8 = __shfl_xor_sync(0xffffu, bi8, o);
        dvt_top8_merge(bv, bi, bv2, bi2, bv3, bi3, bv4, bi4, bv5, bi5, bv6, bi6, bv7, bi7, bv8, bi8,
                       ov, oi, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6, ov7, oi7, ov8, oi8);
      } else if (want_top6) {
        const float ov2 = __shfl_xor_sync(0xffffu, bv2, o);
        const int oi2 = __shfl_xor_sync(0xffffu, bi2, o);
        const float ov3 = __shfl_xor_sync(0xffffu, bv3, o);
        const int oi3 = __shfl_xor_sync(0xffffu, bi3, o);
        const float ov4 = __shfl_xor_sync(0xffffu, bv4, o);
        const int oi4 = __shfl_xor_sync(0xffffu, bi4, o);
        const float ov5 = __shfl_xor_sync(0xffffu, bv5, o);
        const int oi5 = __shfl_xor_sync(0xffffu, bi5, o);
        const float ov6 = __shfl_xor_sync(0xffffu, bv6, o);
        const int oi6 = __shfl_xor_sync(0xffffu, bi6, o);
        dvt_top6_merge(bv, bi, bv2, bi2, bv3, bi3, bv4, bi4, bv5, bi5, bv6, bi6,
                       ov, oi, ov2, oi2, ov3, oi3, ov4, oi4, ov5, oi5, ov6, oi6);
      } else if (want_top4) {
        const float ov2 = __shfl_xor_sync(0xffffu, bv2, o);
        const int oi2 = __shfl_xor_sync(0xffffu, bi2, o);
        const float ov3 = __shfl_xor_sync(0xffffu, bv3, o);
        const int oi3 = __shfl_xor_sync(0xffffu, bi3, o);
        const float ov4 = __shfl_xor_sync(0xffffu, bv4, o);
        const int oi4 = __shfl_xor_sync(0xffffu, bi4, o);
        dvt_top4_merge(bv, bi, bv2, bi2, bv3, bi3, bv4, bi4, ov, oi, ov2, oi2, ov3, oi3, ov4, oi4);
      } else if (want_top2) {
        const float ov2 = __shfl_xor_sync(0xffffu, bv2, o);
        const int oi2 = __shfl_xor_sync(0xffffu, bi2, o);
        if (ov >= bv2) {
          dvt_top2_insert(bv, bi, bv2, bi2, ov, oi);
          if (ov2 >= bv2) {
            dvt_top2_insert(bv, bi, bv2, bi2, ov2, oi2);
          }
        }
      } else if (dvt_better(ov, oi, bv, bi)) {
        bv = ov;
        bi = oi;
      }
    }
  }
  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
  if (lane == 0) {
    out_tokens[m] = static_cast<int64_t>(bi);
    if ((TopK == 8 || TopK == 6 || want_top8 || want_top6 || want_top4 || want_top2) && out_top2 != nullptr && m < top2_rows) {
      out_top2[2 * m + 0] = static_cast<int64_t>(bi);
      out_top2[2 * m + 1] = (bi2 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi2);
      if (out_margins != nullptr && m < margin_rows) {
        out_margins[m] = (bi2 == 0x7fffffff) ? 1e30f : (bv - bv2);
      }
    }
    if constexpr (TopK == 8) {
      out_tree_top[8 * m + 0] = static_cast<int64_t>(bi);
      out_tree_top[8 * m + 1] = (bi2 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi2);
      out_tree_top[8 * m + 2] = (bi3 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi3);
      out_tree_top[8 * m + 3] = (bi4 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi4);
      out_tree_top[8 * m + 4] = (bi5 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi5);
      out_tree_top[8 * m + 5] = (bi6 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi6);
      out_tree_top[8 * m + 6] = (bi7 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi7);
      out_tree_top[8 * m + 7] = (bi8 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi8);
      if (out_tree_vals != nullptr) {
        out_tree_vals[8 * m + 0] = (bi == 0x7fffffff) ? -1e30f : bv;
        out_tree_vals[8 * m + 1] = (bi2 == 0x7fffffff) ? -1e30f : bv2;
        out_tree_vals[8 * m + 2] = (bi3 == 0x7fffffff) ? -1e30f : bv3;
        out_tree_vals[8 * m + 3] = (bi4 == 0x7fffffff) ? -1e30f : bv4;
        out_tree_vals[8 * m + 4] = (bi5 == 0x7fffffff) ? -1e30f : bv5;
        out_tree_vals[8 * m + 5] = (bi6 == 0x7fffffff) ? -1e30f : bv6;
        out_tree_vals[8 * m + 6] = (bi7 == 0x7fffffff) ? -1e30f : bv7;
        out_tree_vals[8 * m + 7] = (bi8 == 0x7fffffff) ? -1e30f : bv8;
      }
    } else if constexpr (TopK == 6) {
      out_tree_top[6 * m + 0] = static_cast<int64_t>(bi);
      out_tree_top[6 * m + 1] = (bi2 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi2);
      out_tree_top[6 * m + 2] = (bi3 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi3);
      out_tree_top[6 * m + 3] = (bi4 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi4);
      out_tree_top[6 * m + 4] = (bi5 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi5);
      out_tree_top[6 * m + 5] = (bi6 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi6);
      if (out_tree_vals != nullptr) {
        out_tree_vals[6 * m + 0] = (bi == 0x7fffffff) ? -1e30f : bv;
        out_tree_vals[6 * m + 1] = (bi2 == 0x7fffffff) ? -1e30f : bv2;
        out_tree_vals[6 * m + 2] = (bi3 == 0x7fffffff) ? -1e30f : bv3;
        out_tree_vals[6 * m + 3] = (bi4 == 0x7fffffff) ? -1e30f : bv4;
        out_tree_vals[6 * m + 4] = (bi5 == 0x7fffffff) ? -1e30f : bv5;
        out_tree_vals[6 * m + 5] = (bi6 == 0x7fffffff) ? -1e30f : bv6;
      }
    } else if (out_tree_top != nullptr && m < tree_top_rows) {
      if (tree_k >= 8) {
        out_tree_top[8 * m + 0] = static_cast<int64_t>(bi);
        out_tree_top[8 * m + 1] = (bi2 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi2);
        out_tree_top[8 * m + 2] = (want_top8 && bi3 != 0x7fffffff) ? static_cast<int64_t>(bi3) : -1;
        out_tree_top[8 * m + 3] = (want_top8 && bi4 != 0x7fffffff) ? static_cast<int64_t>(bi4) : -1;
        out_tree_top[8 * m + 4] = (want_top8 && bi5 != 0x7fffffff) ? static_cast<int64_t>(bi5) : -1;
        out_tree_top[8 * m + 5] = (want_top8 && bi6 != 0x7fffffff) ? static_cast<int64_t>(bi6) : -1;
        out_tree_top[8 * m + 6] = (want_top8 && bi7 != 0x7fffffff) ? static_cast<int64_t>(bi7) : -1;
        out_tree_top[8 * m + 7] = (want_top8 && bi8 != 0x7fffffff) ? static_cast<int64_t>(bi8) : -1;
        if (out_tree_vals != nullptr) {
          out_tree_vals[8 * m + 0] = (bi == 0x7fffffff) ? -1e30f : bv;
          out_tree_vals[8 * m + 1] = (bi2 == 0x7fffffff) ? -1e30f : bv2;
          out_tree_vals[8 * m + 2] = (want_top8 && bi3 != 0x7fffffff) ? bv3 : -1e30f;
          out_tree_vals[8 * m + 3] = (want_top8 && bi4 != 0x7fffffff) ? bv4 : -1e30f;
          out_tree_vals[8 * m + 4] = (want_top8 && bi5 != 0x7fffffff) ? bv5 : -1e30f;
          out_tree_vals[8 * m + 5] = (want_top8 && bi6 != 0x7fffffff) ? bv6 : -1e30f;
          out_tree_vals[8 * m + 6] = (want_top8 && bi7 != 0x7fffffff) ? bv7 : -1e30f;
          out_tree_vals[8 * m + 7] = (want_top8 && bi8 != 0x7fffffff) ? bv8 : -1e30f;
        }
      } else if (tree_k >= 6) {
        out_tree_top[6 * m + 0] = static_cast<int64_t>(bi);
        out_tree_top[6 * m + 1] = (bi2 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi2);
        out_tree_top[6 * m + 2] = (want_top6 && bi3 != 0x7fffffff) ? static_cast<int64_t>(bi3) : -1;
        out_tree_top[6 * m + 3] = (want_top6 && bi4 != 0x7fffffff) ? static_cast<int64_t>(bi4) : -1;
        out_tree_top[6 * m + 4] = (want_top6 && bi5 != 0x7fffffff) ? static_cast<int64_t>(bi5) : -1;
        out_tree_top[6 * m + 5] = (want_top6 && bi6 != 0x7fffffff) ? static_cast<int64_t>(bi6) : -1;
        if (out_tree_vals != nullptr) {
          out_tree_vals[6 * m + 0] = (bi == 0x7fffffff) ? -1e30f : bv;
          out_tree_vals[6 * m + 1] = (bi2 == 0x7fffffff) ? -1e30f : bv2;
          out_tree_vals[6 * m + 2] = (want_top6 && bi3 != 0x7fffffff) ? bv3 : -1e30f;
          out_tree_vals[6 * m + 3] = (want_top6 && bi4 != 0x7fffffff) ? bv4 : -1e30f;
          out_tree_vals[6 * m + 4] = (want_top6 && bi5 != 0x7fffffff) ? bv5 : -1e30f;
          out_tree_vals[6 * m + 5] = (want_top6 && bi6 != 0x7fffffff) ? bv6 : -1e30f;
        }
      } else {
        out_tree_top[4 * m + 0] = static_cast<int64_t>(bi);
        out_tree_top[4 * m + 1] = (bi2 == 0x7fffffff) ? -1 : static_cast<int64_t>(bi2);
        out_tree_top[4 * m + 2] = (want_top4 && bi3 != 0x7fffffff) ? static_cast<int64_t>(bi3) : -1;
        out_tree_top[4 * m + 3] = (want_top4 && bi4 != 0x7fffffff) ? static_cast<int64_t>(bi4) : -1;
      }
    }
  }
}

// Diagnostic / load-time self-check helper: score `row_ids [R]` exact NVFP4 rows for `A [M, K]`.
__global__ void draft_vocab_score_rows_kernel(
    const __nv_bfloat16* __restrict__ A,
    const int32_t* __restrict__ row_ids,
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    const float* __restrict__ global_scale,
    float* __restrict__ out_scores,
    int M, int R, int K, float logit_scale) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp_id >= M * R) return;
  const int m = warp_id / R;
  const int r = warp_id - m * R;
  const int tok = row_ids[r];
  const float total_scale = __ldg(global_scale) * logit_scale;
  const float val = dvt_warp_nvfp4_dot(
      A + static_cast<size_t>(m) * K, nullptr, nullptr, false,
      raw_w, raw_s, tok, K, K >> 4, lane, total_scale);
  if (lane == 0) {
    out_scores[m * R + r] = val;
  }
}

struct DraftVocabScratch {
  torch::Tensor s2_vals;
  torch::Tensor s2_ids;
  torch::Tensor s2_vals2;
  torch::Tensor s2_ids2;
  torch::Tensor s2_vals3;
  torch::Tensor s2_ids3;
  torch::Tensor s2_vals4;
  torch::Tensor s2_ids4;
  torch::Tensor s2_vals5;
  torch::Tensor s2_ids5;
  torch::Tensor s2_vals6;
  torch::Tensor s2_ids6;
  torch::Tensor s2_vals7;
  torch::Tensor s2_ids7;
  torch::Tensor s2_vals8;
  torch::Tensor s2_ids8;
  c10::DeviceIndex dev = -1;
};

DraftVocabScratch& get_dvt_scratch(torch::Device device) {
  static DraftVocabScratch sc;
  if (!sc.s2_vals.defined() || sc.dev != device.index()) {
    auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    sc.s2_vals = torch::empty({kMaxM, kS2CTAsPerRow}, opt_f32);
    sc.s2_ids = torch::empty({kMaxM, kS2CTAsPerRow}, opt_i32);
    sc.s2_vals2 = torch::empty({6, kS2CTAsPerRow}, opt_f32);
    sc.s2_ids2 = torch::empty({6, kS2CTAsPerRow}, opt_i32);
    sc.s2_vals3 = torch::empty({6, kS2CTAsPerRow}, opt_f32);
    sc.s2_ids3 = torch::empty({6, kS2CTAsPerRow}, opt_i32);
    sc.s2_vals4 = torch::empty({6, kS2CTAsPerRow}, opt_f32);
    sc.s2_ids4 = torch::empty({6, kS2CTAsPerRow}, opt_i32);
    sc.s2_vals5 = torch::empty({6, kS2CTAsPerRow}, opt_f32);
    sc.s2_ids5 = torch::empty({6, kS2CTAsPerRow}, opt_i32);
    sc.s2_vals6 = torch::empty({6, kS2CTAsPerRow}, opt_f32);
    sc.s2_ids6 = torch::empty({6, kS2CTAsPerRow}, opt_i32);
    sc.s2_vals7 = torch::empty({6, kS2CTAsPerRow}, opt_f32);
    sc.s2_ids7 = torch::empty({6, kS2CTAsPerRow}, opt_i32);
    sc.s2_vals8 = torch::empty({6, kS2CTAsPerRow}, opt_f32);
    sc.s2_ids8 = torch::empty({6, kS2CTAsPerRow}, opt_i32);
    sc.dev = device.index();
  }
  return sc;
}

}  // namespace

void draft_vocab_update_tail(
    torch::Tensor input_ids,
    torch::Tensor positions,
    torch::Tensor in_static_vocab,
    torch::Tensor tail_ids,
    torch::Tensor tail_count,
    torch::Tensor tail_seen,
    c10::optional<torch::Tensor> num_rejected = c10::nullopt,
    c10::optional<torch::Tensor> extra_ids = c10::nullopt) {
  TORCH_CHECK(input_ids.is_cuda() && positions.is_cuda() && in_static_vocab.is_cuda());
  TORCH_CHECK(tail_ids.is_cuda() && tail_count.is_cuda() && tail_seen.is_cuda());
  const int num_tokens = static_cast<int>(input_ids.numel());
  if (num_tokens <= 0) return;
  const int vocab_size = static_cast<int>(in_static_vocab.numel());
  const int max_tail = static_cast<int>(tail_ids.numel());
  TORCH_CHECK(positions.numel() >= num_tokens && positions.scalar_type() == torch::kInt64);
  auto ids_c = input_ids.is_contiguous() ? input_ids : input_ids.contiguous();
  auto pos_c = positions.is_contiguous() ? positions : positions.contiguous();
  const int32_t* nr_ptr = nullptr;
  if (num_rejected.has_value() && num_rejected.value().defined() && num_rejected.value().numel() >= 1) {
    TORCH_CHECK(num_rejected.value().is_cuda() && num_rejected.value().scalar_type() == torch::kInt32);
    nr_ptr = num_rejected.value().data_ptr<int32_t>();
  }
  torch::Tensor extra_c;
  int num_extra = 0;
  if (extra_ids.has_value() && extra_ids.value().defined() && extra_ids.value().numel() > 0) {
    TORCH_CHECK(extra_ids.value().is_cuda());
    extra_c = extra_ids.value();
    if (extra_c.scalar_type() != ids_c.scalar_type()) {
      extra_c = extra_c.to(ids_c.scalar_type());
    }
    if (!extra_c.is_contiguous()) {
      extra_c = extra_c.contiguous();
    }
    num_extra = static_cast<int>(extra_c.numel());
  }
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  if (ids_c.scalar_type() == torch::kInt32) {
    const int32_t* ex_ptr = (num_extra > 0) ? extra_c.data_ptr<int32_t>() : nullptr;
    draft_vocab_update_tail_kernel<int32_t><<<1, 256, 0, st>>>(
        ids_c.data_ptr<int32_t>(), pos_c.data_ptr<int64_t>(),
        in_static_vocab.data_ptr<uint8_t>(), tail_ids.data_ptr<int32_t>(),
        tail_count.data_ptr<int32_t>(), tail_seen.data_ptr<int32_t>(),
        nr_ptr, ex_ptr, num_tokens, num_extra, vocab_size, max_tail);
  } else {
    TORCH_CHECK(ids_c.scalar_type() == torch::kInt64);
    const int64_t* ex_ptr = (num_extra > 0) ? extra_c.data_ptr<int64_t>() : nullptr;
    draft_vocab_update_tail_kernel<int64_t><<<1, 256, 0, st>>>(
        ids_c.data_ptr<int64_t>(), pos_c.data_ptr<int64_t>(),
        in_static_vocab.data_ptr<uint8_t>(), tail_ids.data_ptr<int32_t>(),
        tail_count.data_ptr<int32_t>(), tail_seen.data_ptr<int32_t>(),
        nr_ptr, ex_ptr, num_tokens, num_extra, vocab_size, max_tail);
  }
}

torch::Tensor draft_vocab_tail_argmax(
    torch::Tensor A,
    torch::Tensor static_logits,
    torch::Tensor vocab_map,
    torch::Tensor tail_ids,
    torch::Tensor tail_count,
    torch::Tensor raw_w,
    torch::Tensor raw_s,
    torch::Tensor global_scale,
    double logit_scale,
    c10::optional<torch::Tensor> out_top2_row0 = c10::nullopt,
    c10::optional<torch::Tensor> out_margins = c10::nullopt,
    c10::optional<torch::Tensor> out_tree_top = c10::nullopt,
    c10::optional<torch::Tensor> out_tree_vals = c10::nullopt) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous());
  TORCH_CHECK(static_logits.is_cuda() && static_logits.scalar_type() == torch::kBFloat16 &&
              static_logits.dim() == 2 && static_logits.is_contiguous());
  TORCH_CHECK(vocab_map.is_cuda() && vocab_map.scalar_type() == torch::kInt64);
  const int M = static_cast<int>(A.size(0));
  const int K = static_cast<int>(A.size(1));
  const int V_static = static_cast<int>(static_logits.size(1));
  const int max_tail = static_cast<int>(tail_ids.numel());
  TORCH_CHECK(M >= 1 && M <= kMaxM && static_logits.size(0) == M && vocab_map.numel() >= V_static);
  TORCH_CHECK(K == raw_w.size(1) * 2 && (K % 16) == 0);

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

  auto& sc = get_dvt_scratch(A.device());
  auto out = torch::empty({M}, A.options().dtype(torch::kInt64));
  auto st = c10::cuda::getCurrentCUDAStream().stream();

  const int top2_max_rows = std::max(top2_rows, tree_top_rows);
  float* v2_ptr = (top2_max_rows > 0) ? sc.s2_vals2.data_ptr<float>() : nullptr;
  int32_t* i2_ptr = (top2_max_rows > 0) ? sc.s2_ids2.data_ptr<int32_t>() : nullptr;
  float* v3_ptr = (tree_top_ptr != nullptr) ? sc.s2_vals3.data_ptr<float>() : nullptr;
  int32_t* i3_ptr = (tree_top_ptr != nullptr) ? sc.s2_ids3.data_ptr<int32_t>() : nullptr;
  float* v4_ptr = (tree_top_ptr != nullptr) ? sc.s2_vals4.data_ptr<float>() : nullptr;
  int32_t* i4_ptr = (tree_top_ptr != nullptr) ? sc.s2_ids4.data_ptr<int32_t>() : nullptr;
  float* v5_ptr = (tree_top_ptr != nullptr && tree_k >= 6) ? sc.s2_vals5.data_ptr<float>() : nullptr;
  int32_t* i5_ptr = (tree_top_ptr != nullptr && tree_k >= 6) ? sc.s2_ids5.data_ptr<int32_t>() : nullptr;
  float* v6_ptr = (tree_top_ptr != nullptr && tree_k >= 6) ? sc.s2_vals6.data_ptr<float>() : nullptr;
  int32_t* i6_ptr = (tree_top_ptr != nullptr && tree_k >= 6) ? sc.s2_ids6.data_ptr<int32_t>() : nullptr;
  float* v7_ptr = (tree_top_ptr != nullptr && tree_k >= 8) ? sc.s2_vals7.data_ptr<float>() : nullptr;
  int32_t* i7_ptr = (tree_top_ptr != nullptr && tree_k >= 8) ? sc.s2_ids7.data_ptr<int32_t>() : nullptr;
  float* v8_ptr = (tree_top_ptr != nullptr && tree_k >= 8) ? sc.s2_vals8.data_ptr<float>() : nullptr;
  int32_t* i8_ptr = (tree_top_ptr != nullptr && tree_k >= 8) ? sc.s2_ids8.data_ptr<int32_t>() : nullptr;

  const int pdl_mode = dvt_pdl_mode();
  const bool use_fast = dvt_fast_enabled();
  const bool fast_top8 = (use_fast && tree_k == 8 && tree_top_rows >= M &&
                          tree_top_ptr != nullptr && v8_ptr != nullptr && i8_ptr != nullptr);
  const bool fast_top6 = (!fast_top8 && use_fast && tree_k == 6 && tree_top_rows >= M &&
                          tree_top_ptr != nullptr && v6_ptr != nullptr && i6_ptr != nullptr);
  auto launch_both = [&](auto stage1_kernel, auto stage2_kernel) {
    dvt_launch_pdl(
        stage1_kernel, dim3(kS2CTAsPerRow, M), 256, st, pdl_mode,
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(static_logits.data_ptr()),
        vocab_map.data_ptr<int64_t>(),
        tail_ids.data_ptr<int32_t>(),
        tail_count.data_ptr<int32_t>(),
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        sc.s2_vals.data_ptr<float>(),
        sc.s2_ids.data_ptr<int32_t>(),
        v2_ptr,
        i2_ptr,
        v3_ptr,
        i3_ptr,
        v4_ptr,
        i4_ptr,
        v5_ptr,
        i5_ptr,
        v6_ptr,
        i6_ptr,
        v7_ptr,
        i7_ptr,
        v8_ptr,
        i8_ptr,
        top2_max_rows,
        tree_top_rows,
        tree_k,
        V_static, K, max_tail, static_cast<float>(logit_scale), pdl_mode);

    dvt_launch_pdl(
        stage2_kernel, dim3(M, 1, 1), 32, st, pdl_mode,
        static_cast<const float*>(sc.s2_vals.data_ptr<float>()),
        static_cast<const int32_t*>(sc.s2_ids.data_ptr<int32_t>()),
        static_cast<const float*>(v2_ptr), static_cast<const int32_t*>(i2_ptr),
        static_cast<const float*>(v3_ptr), static_cast<const int32_t*>(i3_ptr),
        static_cast<const float*>(v4_ptr), static_cast<const int32_t*>(i4_ptr),
        static_cast<const float*>(v5_ptr), static_cast<const int32_t*>(i5_ptr),
        static_cast<const float*>(v6_ptr), static_cast<const int32_t*>(i6_ptr),
        static_cast<const float*>(v7_ptr), static_cast<const int32_t*>(i7_ptr),
        static_cast<const float*>(v8_ptr), static_cast<const int32_t*>(i8_ptr),
        out.data_ptr<int64_t>(), top2_ptr, top2_rows,
        margin_ptr, margin_rows, tree_top_ptr, tree_vals_ptr, tree_top_rows, tree_k, M, pdl_mode);
  };
  if (fast_top8) {
    launch_both(&draft_vocab_tail_argmax_kernel<8>, &draft_vocab_final_reduce_kernel<8>);
  } else if (fast_top6) {
    launch_both(&draft_vocab_tail_argmax_kernel<6>, &draft_vocab_final_reduce_kernel<6>);
  } else {
    launch_both(&draft_vocab_tail_argmax_kernel<0>, &draft_vocab_final_reduce_kernel<0>);
  }
  return out;
}

torch::Tensor draft_vocab_score_rows(
    torch::Tensor A,
    torch::Tensor row_ids,
    torch::Tensor raw_w,
    torch::Tensor raw_s,
    torch::Tensor global_scale,
    double logit_scale) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous());
  TORCH_CHECK(row_ids.is_cuda() && row_ids.scalar_type() == torch::kInt32 && row_ids.is_contiguous());
  const int M = static_cast<int>(A.size(0));
  const int K = static_cast<int>(A.size(1));
  const int R = static_cast<int>(row_ids.numel());
  auto out = torch::empty({M, R}, A.options().dtype(torch::kFloat32));
  if (M == 0 || R == 0) return out;
  const int blocks = (M * R + 7) / 8;
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  draft_vocab_score_rows_kernel<<<blocks, 256, 0, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      row_ids.data_ptr<int32_t>(),
      raw_w.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
      global_scale.data_ptr<float>(),
      out.data_ptr<float>(), M, R, K, static_cast<float>(logit_scale));
  return out;
}
