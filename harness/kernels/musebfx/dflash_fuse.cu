// SPDX-License-Identifier: Apache-2.0
// dflash_fuse: 0-ULP fused context-KV post-GEMM kernel for the pinned DFlash drafter on SM121 (GB10).
//
// `dflash_kv_post`: fuses context-KV permute/split + per-layer K RMSNorm + positions.repeat +
// RoPE + L-layer paged KV-cache scatter (9 launches -> 1 launch per speculative cycle).
// 100% CUDA-graph safe (all descriptors passed by value, zero host sync) and verified at load
// against stock vLLM ops before admission.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <stdint.h>
#include <tuple>
#include <vector>

__nv_bfloat16* vq4d_alloc_norm_prerot(
    const void* src_ptr, const void* weight_ptr, int M, int64_t K, const torch::TensorOptions& opts);
void vq4d_clear_all_prerot();

namespace {

static bool dflash_norm_fuse_rot_enabled() {
  static const bool dyn = (std::getenv("MUSE_BENCH_DYN_ENV") != nullptr);
  if (dyn) {
    const char* s = std::getenv("MUSE_DFLASH_NORM_FUSE_ROT");
    return (!s || !*s) ? false : (std::atoi(s) != 0);
  }
  static const bool en = []() -> bool {
    const char* s = std::getenv("MUSE_DFLASH_NORM_FUSE_ROT");
    return (!s || !*s) ? false : (std::atoi(s) != 0);
  }();
  return en;
}

__device__ __forceinline__ float dflash_rad_sign(int k) {
  uint32_t x = static_cast<uint32_t>(k) * 0x9E3779B9u + 0x7F4A7C15u;
  x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
  x = (x ^ (x >> 13)) * 0xC2B2AE35u;
  x ^= (x >> 16);
  return (x & 1u) ? -1.0f : 1.0f;
}

__device__ __forceinline__ void dflash_warp_fwht64(float& v0, float& v1, int lane) {
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

constexpr int kHeadDim = 128;
constexpr int kMaxLayers = 8;

static int dflash_fuse_pdl_mode() {
  static const int mode = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_PDL");
    if (!s || !*s) return 3;
    return (std::atoi(s) != 0) ? 3 : 0;
  }();
  return mode;
}

static bool dflash_carveout100_enabled() {
  static const bool en = []() -> bool {
    const char* s = std::getenv("MUSE_DRAFTER_CARVEOUT100");
    return !s || !*s || std::atoi(s) != 0;
  }();
  return en;
}

static inline void dflash_set_carveout100_once(const void* fn) {
  if (!dflash_carveout100_enabled()) return;
  static const void* seen[32] = {};
  static int n_seen = 0;
  for (int i = 0; i < n_seen; ++i) {
    if (seen[i] == fn) return;
  }
  if (n_seen < 32) seen[n_seen++] = fn;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  cudaThreadExchangeStreamCaptureMode(&mode);
  cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaThreadExchangeStreamCaptureMode(&mode);
}

template <typename... KArgs, typename... Args>
static inline void dflash_launch_pdl(
    void (*kernel)(KArgs...), int blocks, int threads, cudaStream_t st, int pdl_mode, Args... args) {
  dflash_set_carveout100_once(reinterpret_cast<const void*>(kernel));
  cudaLaunchConfig_t lcfg = {};
  lcfg.gridDim = dim3(static_cast<unsigned>(blocks), 1u, 1u);
  lcfg.blockDim = dim3(static_cast<unsigned>(threads), 1u, 1u);
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
  TORCH_CHECK(err == cudaSuccess, "dflash_fuse launch failed: ", cudaGetErrorString(err));
}

struct RopeCS {
  float c0, c1, c2, c3;
  float s0, s1, s2, s3;
};

template <bool kNeox, bool kCacheFP32>
__device__ __forceinline__ RopeCS preload_rope_cs_hd128(
    const void* __restrict__ cos_sin_cache,
    int64_t pos,
    int cache_rows,
    int lane) {
  RopeCS cs = {};
  if (pos < 0) pos += cache_rows;
  if constexpr (kNeox) {
    const int d_lo = (lane & 15) * 4;
    if constexpr (kCacheFP32) {
      const float* row = reinterpret_cast<const float*>(cos_sin_cache) + pos * 128;
      const float4 cv = __ldg(reinterpret_cast<const float4*>(row + d_lo));
      const float4 sv = __ldg(reinterpret_cast<const float4*>(row + 64 + d_lo));
      cs.c0 = cv.x; cs.c1 = cv.y; cs.c2 = cv.z; cs.c3 = cv.w;
      cs.s0 = sv.x; cs.s1 = sv.y; cs.s2 = sv.z; cs.s3 = sv.w;
    } else {
      const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache) + pos * 128;
      const uint2 cv = __ldg(reinterpret_cast<const uint2*>(row + d_lo));
      const uint2 sv = __ldg(reinterpret_cast<const uint2*>(row + 64 + d_lo));
      const __nv_bfloat16* cb = reinterpret_cast<const __nv_bfloat16*>(&cv);
      const __nv_bfloat16* sb = reinterpret_cast<const __nv_bfloat16*>(&sv);
      cs.c0 = __bfloat162float(cb[0]); cs.c1 = __bfloat162float(cb[1]);
      cs.c2 = __bfloat162float(cb[2]); cs.c3 = __bfloat162float(cb[3]);
      cs.s0 = __bfloat162float(sb[0]); cs.s1 = __bfloat162float(sb[1]);
      cs.s2 = __bfloat162float(sb[2]); cs.s3 = __bfloat162float(sb[3]);
    }
  } else {
    const int d_lo = lane * 2;
    if constexpr (kCacheFP32) {
      const float* row = reinterpret_cast<const float*>(cos_sin_cache) + pos * 128;
      cs.c0 = __ldg(row + d_lo + 0); cs.c1 = __ldg(row + d_lo + 1);
      cs.s0 = __ldg(row + 64 + d_lo + 0); cs.s1 = __ldg(row + 64 + d_lo + 1);
    } else {
      const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache) + pos * 128;
      cs.c0 = __bfloat162float(__ldg(row + d_lo + 0));
      cs.c1 = __bfloat162float(__ldg(row + d_lo + 1));
      cs.s0 = __bfloat162float(__ldg(row + 64 + d_lo + 0));
      cs.s1 = __bfloat162float(__ldg(row + 64 + d_lo + 1));
    }
  }
  return cs;
}

template <bool kNeox>
__device__ __forceinline__ void apply_rmsnorm_rope_hd128_preloaded(
    const __nv_bfloat16* __restrict__ in_ptr,
    uint2 wraw,
    const RopeCS& cs,
    float eps,
    int lane,
    uint2& out_v) {
  const uint2 raw = *reinterpret_cast<const uint2*>(in_ptr + lane * 4);
  const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&raw);
  const float x0 = __bfloat162float(vb[0]);
  const float x1 = __bfloat162float(vb[1]);
  const float x2 = __bfloat162float(vb[2]);
  const float x3 = __bfloat162float(vb[3]);

  float ss = __fmaf_rn(x0, x0, 0.0f);
  ss = __fmaf_rn(x1, x1, ss);
  ss = __fmaf_rn(x2, x2, ss);
  ss = __fmaf_rn(x3, x3, ss);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, o);
  }
  const float inv_rms = rsqrtf(ss * (1.0f / 128.0f) + eps);

  const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wraw);
  // Round to BF16 between RMSNorm and RoPE to match ops.rms_norm -> ops.rotary_embedding verbatim.
  const float y0 = __bfloat162float(__float2bfloat16_rn(x0 * inv_rms * __bfloat162float(wb[0])));
  const float y1 = __bfloat162float(__float2bfloat16_rn(x1 * inv_rms * __bfloat162float(wb[1])));
  const float y2 = __bfloat162float(__float2bfloat16_rn(x2 * inv_rms * __bfloat162float(wb[2])));
  const float y3 = __bfloat162float(__float2bfloat16_rn(x3 * inv_rms * __bfloat162float(wb[3])));

  float o0, o1, o2, o3;
  if constexpr (kNeox) {
    const float p0 = __shfl_xor_sync(0xffffffffu, y0, 16);
    const float p1 = __shfl_xor_sync(0xffffffffu, y1, 16);
    const float p2 = __shfl_xor_sync(0xffffffffu, y2, 16);
    const float p3 = __shfl_xor_sync(0xffffffffu, y3, 16);
    if (lane < 16) {
      o0 = __fsub_rn(__fmul_rn(y0, cs.c0), __fmul_rn(p0, cs.s0));
      o1 = __fsub_rn(__fmul_rn(y1, cs.c1), __fmul_rn(p1, cs.s1));
      o2 = __fsub_rn(__fmul_rn(y2, cs.c2), __fmul_rn(p2, cs.s2));
      o3 = __fsub_rn(__fmul_rn(y3, cs.c3), __fmul_rn(p3, cs.s3));
    } else {
      o0 = __fadd_rn(__fmul_rn(y0, cs.c0), __fmul_rn(p0, cs.s0));
      o1 = __fadd_rn(__fmul_rn(y1, cs.c1), __fmul_rn(p1, cs.s1));
      o2 = __fadd_rn(__fmul_rn(y2, cs.c2), __fmul_rn(p2, cs.s2));
      o3 = __fadd_rn(__fmul_rn(y3, cs.c3), __fmul_rn(p3, cs.s3));
    }
  } else {
    o0 = __fsub_rn(__fmul_rn(y0, cs.c0), __fmul_rn(y1, cs.s0));
    o1 = __fadd_rn(__fmul_rn(y1, cs.c0), __fmul_rn(y0, cs.s0));
    o2 = __fsub_rn(__fmul_rn(y2, cs.c1), __fmul_rn(y3, cs.s1));
    o3 = __fadd_rn(__fmul_rn(y3, cs.c1), __fmul_rn(y2, cs.s1));
  }

  __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&out_v);
  ob[0] = __float2bfloat16_rn(o0);
  ob[1] = __float2bfloat16_rn(o1);
  ob[2] = __float2bfloat16_rn(o2);
  ob[3] = __float2bfloat16_rn(o3);
}

struct KvPostDesc {
  __nv_bfloat16* k_cache[kMaxLayers];
  __nv_bfloat16* v_cache[kMaxLayers];
  const int64_t* slots[kMaxLayers];
  int64_t k_sb[kMaxLayers], k_sn[kMaxLayers], k_sh[kMaxLayers];
  int64_t v_sb[kMaxLayers], v_sn[kMaxLayers], v_sh[kMaxLayers];
};

template <bool kNeox, bool kCacheFP32>
__global__ void dflash_kv_post_kernel(
    const __nv_bfloat16* __restrict__ all_kv_flat,  // [M, L, 2, nkv, 128]
    const int64_t* __restrict__ positions,          // [M]
    const __nv_bfloat16* __restrict__ k_norm_w,     // [L, 128]
    const void* __restrict__ cos_sin_cache,         // [cache_rows, 128]
    KvPostDesc desc,
    int M, int L, int nkv, int block_size, int cache_rows, float eps, int pdl_mode) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int total = M * L * nkv;
  if (warp_id >= total) return;
  const int lane = threadIdx.x & 31;
  const int h = warp_id % nkv;
  const int ml = warp_id / nkv;
  const int l = ml % L;
  const int m = ml / L;

  // Preload all predecessor-independent inputs (norm weight, slot, pos, cos/sin) before PDL wait
  // so launch + metadata/LUT fetches fully overlap the predecessor fkv GEMM tail.
  const __nv_bfloat16* nw = k_norm_w + static_cast<size_t>(l) * kHeadDim;
  const uint2 wraw = __ldg(reinterpret_cast<const uint2*>(nw + lane * 4));
  const int64_t slot = __ldg(desc.slots[l] + m);
  const int64_t pos = __ldg(positions + m);
  const RopeCS cs = (slot >= 0) ? preload_rope_cs_hd128<kNeox, kCacheFP32>(cos_sin_cache, pos, cache_rows, lane)
                                : RopeCS{};
  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
  if (slot < 0) return;

  const int64_t block_idx = (block_size == 64) ? (slot >> 6) : ((block_size == 16) ? (slot >> 4) : (slot / block_size));
  const int64_t block_off = (block_size == 64) ? (slot & 63) : ((block_size == 16) ? (slot & 15) : (slot - block_idx * block_size));

  const int kv_stride = nkv * kHeadDim;
  const size_t base_ml = static_cast<size_t>(m * L + l) * (2 * kv_stride);
  const __nv_bfloat16* k_in = all_kv_flat + base_ml + h * kHeadDim;
  const __nv_bfloat16* v_in = all_kv_flat + base_ml + kv_stride + h * kHeadDim;

  uint2 k_out_v;
  apply_rmsnorm_rope_hd128_preloaded<kNeox>(k_in, wraw, cs, eps, lane, k_out_v);
  const uint2 v_out_v = *reinterpret_cast<const uint2*>(v_in + lane * 4);

  __nv_bfloat16* k_dst = desc.k_cache[l] + block_idx * desc.k_sb[l] + block_off * desc.k_sn[l] + h * desc.k_sh[l];
  __nv_bfloat16* v_dst = desc.v_cache[l] + block_idx * desc.v_sb[l] + block_off * desc.v_sn[l] + h * desc.v_sh[l];
  *reinterpret_cast<uint2*>(k_dst + lane * 4) = k_out_v;
  *reinterpret_cast<uint2*>(v_dst + lane * 4) = v_out_v;
}

// Optional paged KV-cache scatter fused into dflash_qkv_norm_rope (one
// launch per drafter layer instead of qkv_norm_rope + reshape_and_cache_flash).
struct QkvCacheDesc {
  __nv_bfloat16* k_cache;
  __nv_bfloat16* v_cache;
  const int64_t* slots;
  int64_t k_sb, k_sn, k_sh;
  int64_t v_sb, v_sn, v_sh;
  int block_size;
  int enabled;
};

template <bool kNeox, bool kCacheFP32>
__global__ void dflash_qkv_norm_rope_kernel(
    const __nv_bfloat16* __restrict__ qkv,         // [M, (nq + 2*nkv) * 128]
    const int64_t* __restrict__ positions,         // [M]
    const __nv_bfloat16* __restrict__ q_norm_w,    // [128]
    const __nv_bfloat16* __restrict__ k_norm_w,    // [128]
    const void* __restrict__ cos_sin_cache,        // [cache_rows, 128]
    __nv_bfloat16* __restrict__ q_out,             // [M, nq * 128]
    __nv_bfloat16* __restrict__ k_out,             // [M, nkv * 128]
    QkvCacheDesc cdesc,
    int M, int nq, int nkv, int cache_rows, float eps, int pdl_mode) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int total_h = nq + nkv;
  if (warp_id >= M * total_h) return;
  const int lane = threadIdx.x & 31;
  const int h = warp_id % total_h;
  const int m = warp_id / total_h;

  // Preload all predecessor-independent inputs (norm weight, pos, cos/sin, slot) before PDL wait.
  const __nv_bfloat16* nw = (h < nq) ? q_norm_w : k_norm_w;
  const uint2 wraw = __ldg(reinterpret_cast<const uint2*>(nw + lane * 4));
  const int64_t pos = __ldg(positions + m);
  const RopeCS cs = preload_rope_cs_hd128<kNeox, kCacheFP32>(cos_sin_cache, pos, cache_rows, lane);
  const int64_t slot = (h >= nq && cdesc.enabled) ? __ldg(cdesc.slots + m) : -1;
  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }

  const size_t qkv_stride = static_cast<size_t>(nq + 2 * nkv) * kHeadDim;
  const __nv_bfloat16* row_in = qkv + static_cast<size_t>(m) * qkv_stride;

  uint2 out_v;
  if (h < nq) {
    apply_rmsnorm_rope_hd128_preloaded<kNeox>(
        row_in + static_cast<size_t>(h) * kHeadDim,
        wraw, cs, eps, lane, out_v);
    __nv_bfloat16* dst = q_out + (static_cast<size_t>(m) * nq + h) * kHeadDim;
    *reinterpret_cast<uint2*>(dst + lane * 4) = out_v;
  } else {
    const int kh = h - nq;
    apply_rmsnorm_rope_hd128_preloaded<kNeox>(
        row_in + static_cast<size_t>(nq + kh) * kHeadDim,
        wraw, cs, eps, lane, out_v);
    uint2 v_out_v = {};
    if (cdesc.enabled) {
      v_out_v = *reinterpret_cast<const uint2*>(
          row_in + static_cast<size_t>(nq + nkv + kh) * kHeadDim + lane * 4);
    }
    // k_out is always written so the returned K stays valid for any
    // consumer (and for the unfused cache update if the skip is not taken).
    __nv_bfloat16* dst = k_out + (static_cast<size_t>(m) * nkv + kh) * kHeadDim;
    *reinterpret_cast<uint2*>(dst + lane * 4) = out_v;
    if (cdesc.enabled && slot >= 0) {
      const int64_t b = (cdesc.block_size == 64) ? (slot >> 6)
                        : ((cdesc.block_size == 16) ? (slot >> 4) : (slot / cdesc.block_size));
      const int64_t o = (cdesc.block_size == 64) ? (slot & 63)
                        : ((cdesc.block_size == 16) ? (slot & 15) : (slot - b * cdesc.block_size));
      __nv_bfloat16* k_dst = cdesc.k_cache + b * cdesc.k_sb + o * cdesc.k_sn + kh * cdesc.k_sh;
      __nv_bfloat16* v_dst = cdesc.v_cache + b * cdesc.v_sb + o * cdesc.v_sn + kh * cdesc.v_sh;
      *reinterpret_cast<uint2*>(k_dst + lane * 4) = out_v;
      *reinterpret_cast<uint2*>(v_dst + lane * 4) = v_out_v;
    }
  }
}

template <typename LastSampT, typename TailIdT>
__global__ void __launch_bounds__(256) dflash_prepare_inputs_kernel(
    int32_t* out_input_ids,
    int64_t* out_query_positions,
    int32_t* out_query_start_loc,
    int32_t* out_seq_lens,
    int64_t* out_query_slot_mapping,
    int64_t* out_context_positions,
    int64_t* out_context_slot_mapping,
    int64_t* out_sample_indices,
    int64_t* out_sample_pos,
    int32_t* out_sample_idx_mapping,
    float* out_temperature,
    int64_t* out_seeds,
    const int64_t* target_positions,
    const int32_t* target_query_start_loc,
    const int32_t* idx_mapping,
    LastSampT* last_sampled,
    const int32_t* next_prefill_tokens,
    const int32_t* num_sampled,
    const int32_t* num_rejected,
    const float* in_temperature,
    const int64_t* in_seeds,
    const int32_t* block_table,
    int64_t block_table_stride,
    int32_t parallel_drafting_token_id,
    int64_t block_size,
    int num_query_per_req,
    int num_speculative_steps,
    int max_num_reqs,
    int max_num_tokens,
    int64_t max_model_len,
    int64_t pad_slot_id,
    const TailIdT* target_input_ids,
    int tail_num_tokens,
    const uint8_t* __restrict__ in_static_vocab,
    int32_t* tail_ids,
    int32_t* tail_count,
    int32_t* tail_seen,
    int vocab_size,
    int max_tail,
    int include_bonus,
    int32_t* hist,
    int32_t* hist_len,
    int hist_cap,
    const int64_t* post_sampled_tokens,
    int32_t* post_num_computed_tokens,
    int32_t* post_all_token_ids,
    int64_t post_all_tok_stride,
    int32_t* post_total_len,
    int32_t* post_output_bin_counts,
    int64_t post_bin_stride,
    int pdl_mode) {
  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  __shared__ int32_t s_req_state_idx;
  __shared__ int32_t s_ctx_start;
  __shared__ int32_t s_num_ctx;
  __shared__ int32_t s_valid_tail_len;
  __shared__ int32_t s_bonus_token;
  __shared__ int64_t s_last_valid_pos;
  __shared__ int32_t s_ns;
  __shared__ int s_tail_reset;
  __shared__ int s_tail_old_cnt;
  __shared__ int s_hist_write;
  __shared__ int64_t s_hist_need_len;

  const int tid = threadIdx.x;
  if (tid == 0) {
    const int32_t req_state_idx = idx_mapping[0];
    const int32_t ctx_start = target_query_start_loc[0];
    const int32_t ctx_end = target_query_start_loc[1];
    const int32_t num_ctx = ctx_end - ctx_start;
    const int32_t nr = num_rejected[0];
    const int32_t valid_ctx_end = ctx_end - nr;
    const int32_t ns = num_sampled[0];
    if (post_sampled_tokens != nullptr && req_state_idx >= 0) {
      const int32_t tl_old = post_total_len[req_state_idx];
      if (ns > 0) {
        const int64_t last_tok = post_sampled_tokens[ns - 1];
        last_sampled[req_state_idx] = static_cast<LastSampT>(last_tok);
        post_total_len[req_state_idx] = tl_old + ns;
      }
      int32_t* all_row = post_all_token_ids + static_cast<int64_t>(req_state_idx) * post_all_tok_stride + tl_old;
      int32_t* bin_row = (post_output_bin_counts != nullptr)
          ? (post_output_bin_counts + static_cast<int64_t>(req_state_idx) * post_bin_stride)
          : nullptr;
      for (int i = 0; i < ns; ++i) {
        const int64_t tok_i = post_sampled_tokens[i];
        all_row[i] = static_cast<int32_t>(tok_i);
        if (bin_row != nullptr) {
          bin_row[tok_i] += 1;
        }
      }
      const int32_t computed_delta = num_ctx - nr;
      if (computed_delta != 0) {
        post_num_computed_tokens[req_state_idx] += computed_delta;
      }
    }
    const int32_t bonus_token = (ns > 0)
        ? static_cast<int32_t>(last_sampled[req_state_idx])
        : static_cast<int32_t>(next_prefill_tokens[req_state_idx]);
    const int64_t last_valid_pos = target_positions[valid_ctx_end - 1];

    int32_t valid_end = valid_ctx_end;
    if (valid_end > tail_num_tokens) valid_end = tail_num_tokens;
    int32_t valid_tail_len = valid_end - ctx_start;
    if (valid_tail_len < 0) valid_tail_len = 0;

    s_req_state_idx = req_state_idx;
    s_ctx_start = ctx_start;
    s_num_ctx = num_ctx;
    s_valid_tail_len = valid_tail_len;
    s_bonus_token = bonus_token;
    s_last_valid_pos = last_valid_pos;
    s_ns = ns;

    out_query_start_loc[0] = 0;
    const int64_t raw_sl = last_valid_pos + 1 + num_query_per_req;
    out_seq_lens[0] = static_cast<int32_t>((raw_sl < max_model_len) ? raw_sl : max_model_len);
    out_temperature[req_state_idx] = in_temperature[req_state_idx];
    out_seeds[req_state_idx] = in_seeds[req_state_idx];

    if (in_static_vocab != nullptr && tail_num_tokens > 0) {
      s_tail_reset = (target_positions[ctx_start] == 0) ? 1 : 0;
      int oc = tail_count[0];
      if (oc < 0) oc = 0;
      if (oc > max_tail) oc = max_tail;
      s_tail_old_cnt = oc;
    } else {
      s_tail_reset = 0;
      s_tail_old_cnt = 0;
    }

    if (hist != nullptr && hist_len != nullptr && target_input_ids != nullptr &&
        tail_num_tokens > 0 && valid_tail_len > 0) {
      const int64_t first_pos = target_positions[ctx_start];
      const int64_t need_len = last_valid_pos + 1 + ((ns > 0) ? 1 : 0);
      const int32_t cur_hlen = hist_len[0];
      // Write on prompt start (first_pos == 0) OR when native_leaf_commit_tree did not already
      // advance hist_len to need_len (preserves exact tree-hit tokens without overwriting them!).
      if (first_pos == 0 || (cur_hlen >= first_pos && cur_hlen < need_len)) {
        s_hist_write = 1;
        s_hist_need_len = need_len;
      } else {
        s_hist_write = 0;
        s_hist_need_len = 0;
      }
    } else {
      s_hist_write = 0;
      s_hist_need_len = 0;
    }
  }
  __syncthreads();

  const int32_t req_state_idx = s_req_state_idx;
  const int32_t ctx_start = s_ctx_start;
  const int32_t num_ctx = s_num_ctx;
  const int32_t valid_tail_len = s_valid_tail_len;
  const int32_t bonus_token = s_bonus_token;
  const int64_t last_valid_pos = s_last_valid_pos;

  for (int j = tid; j < num_ctx; j += 256) {
    const int64_t ctx_pos = target_positions[ctx_start + j];
    int64_t ctx_blk = (block_size == 16) ? (ctx_pos >> 4) : (ctx_pos / block_size);
    if (ctx_blk > block_table_stride - 1) ctx_blk = block_table_stride - 1;
    const int64_t ctx_bid = static_cast<int64_t>(block_table[ctx_blk]);
    const int64_t ctx_off = (block_size == 16) ? (ctx_pos & 15) : (ctx_pos % block_size);
    out_context_positions[ctx_start + j] = ctx_pos;
    out_context_slot_mapping[ctx_start + j] = ctx_bid * block_size + ctx_off;
  }

  for (int q = tid; q < num_query_per_req; q += 256) {
    const int64_t query_pos = last_valid_pos + 1 + q;
    const int32_t input_id = (q == 0) ? bonus_token : parallel_drafting_token_id;
    int64_t q_blk = (block_size == 16) ? (query_pos >> 4) : (query_pos / block_size);
    if (q_blk > block_table_stride - 1) q_blk = block_table_stride - 1;
    const int64_t q_bid = static_cast<int64_t>(block_table[q_blk]);
    const int64_t q_off = (block_size == 16) ? (query_pos & 15) : (query_pos % block_size);
    out_input_ids[q] = input_id;
    out_query_positions[q] = (query_pos < max_model_len - 1) ? query_pos : (max_model_len - 1);
    out_query_slot_mapping[q] = (query_pos < max_model_len) ? (q_bid * block_size + q_off) : pad_slot_id;
    if (q >= 1 && q < 1 + num_speculative_steps) {
      const int s = q - 1;
      out_sample_indices[s] = q;
      out_sample_pos[s] = query_pos;
      out_sample_idx_mapping[s] = req_state_idx;
    }
  }

  for (int i = 1 + tid; i <= max_num_reqs; i += 256) {
    out_query_start_loc[i] = num_query_per_req;
  }
  for (int i = 1 + tid; i < max_num_reqs; i += 256) {
    out_seq_lens[i] = 0;
  }
  const int pad_end = max_num_reqs * num_speculative_steps;
  for (int i = num_speculative_steps + tid; i < pad_end; i += 256) {
    out_sample_indices[i] = 0;
    out_sample_pos[i] = 0;
    out_sample_idx_mapping[i] = -1;
  }
  const int q_pad_end = (max_num_tokens < max_num_reqs * num_query_per_req)
                            ? max_num_tokens
                            : (max_num_reqs * num_query_per_req);
  for (int i = num_query_per_req + tid; i < q_pad_end; i += 256) {
    out_query_slot_mapping[i] = pad_slot_id;
  }

  if (in_static_vocab != nullptr && tail_num_tokens > 0) {
    if (s_tail_reset) {
      for (int i = tid; i < s_tail_old_cnt; i += 256) {
        const int prev_tok = tail_ids[i];
        if (prev_tok >= 0 && prev_tok < vocab_size) {
          tail_seen[prev_tok] = 0;
        }
      }
      for (int i = tid; i < max_tail; i += 256) {
        tail_ids[i] = -1;
      }
      if (tid == 0) {
        tail_count[0] = 0;
      }
      __syncthreads();
    }
    const int scan_end = valid_tail_len + (include_bonus ? 1 : 0);
    if (tid < 32) {
      const uint32_t lane_lt = (1u << tid) - 1u;
      int cur_cnt = tail_count[0];
      if (cur_cnt < 0) cur_cnt = 0;
      for (int base = 0; base < scan_end; base += 32) {
        const int i = base + tid;
        int tok = -1;
        bool cand = false;
        if (i < scan_end) {
          const int64_t raw_tok = (i < valid_tail_len)
              ? static_cast<int64_t>(target_input_ids[ctx_start + i])
              : static_cast<int64_t>(bonus_token);
          if (raw_tok >= 0 && raw_tok < vocab_size) {
            tok = static_cast<int>(raw_tok);
            cand = (in_static_vocab[tok] == 0);
          }
        }
        bool admit = false;
        if (cand && atomicExch(&tail_seen[tok], 1) == 0) {
          admit = true;
        }
        const uint32_t msk = __ballot_sync(0xffffffffu, admit);
        const int slot = cur_cnt + __popc(msk & lane_lt);
        if (admit) {
          if (slot < max_tail) {
            tail_ids[slot] = tok;
          } else {
            tail_seen[tok] = 0;
          }
        }
        cur_cnt += __popc(msk);
      }
      if (tid == 0) {
        tail_count[0] = (cur_cnt > max_tail) ? max_tail : cur_cnt;
      }
    }
    __syncthreads();
  }

  if (s_hist_write) {
    for (int i = tid; i < valid_tail_len; i += 256) {
      const int64_t pos = target_positions[ctx_start + i];
      if (pos >= 0 && pos < hist_cap) {
        hist[pos] = static_cast<int32_t>(target_input_ids[ctx_start + i]);
      }
    }
    if (tid == 0) {
      if (s_ns > 0) {
        const int64_t bpos = last_valid_pos + 1;
        if (bpos >= 0 && bpos < hist_cap) {
          hist[bpos] = bonus_token;
        }
      }
      hist_len[0] = (s_hist_need_len <= hist_cap) ? static_cast<int32_t>(s_hist_need_len) : -1;
    }
  }
  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
}

template <bool kFusedAdd, bool kPreloadRes = false, bool kEmitRot = false>
__global__ void __launch_bounds__(256) dflash_rms_norm_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual_in,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ residual_out,
    __nv_bfloat16* __restrict__ rot_out,
    int K,
    float eps,
    int pdl_mode) {
  __shared__ alignas(16) __nv_bfloat16 s_norm[kEmitRot ? 8192 : 8];
  __shared__ float s_warp[8];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int k8 = K >> 3;

  // Preload immutable norm weight (and predecessor-independent residual_in when fused) before
  // PDL wait so both fetches overlap the predecessor GEMM tail, then fire launch_dependents
  // immediately after wait so the successor GEMM prologue overlaps our reduction + writeback.
  const uint4* w_u4 = reinterpret_cast<const uint4*>(weight);
  const uint4* r_u4 = kFusedAdd ? reinterpret_cast<const uint4*>(residual_in + static_cast<size_t>(row) * K) : nullptr;
  uint4 w_reg[4];
  uint4 r_reg[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int idx8 = tid + c * 256;
    if (idx8 < k8) {
      w_reg[c] = __ldg(w_u4 + idx8);
      if constexpr (kFusedAdd && kPreloadRes) {
        r_reg[c] = r_u4[idx8];
      }
    }
  }
  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }

  const uint4* x_u4 = reinterpret_cast<const uint4*>(input + static_cast<size_t>(row) * K);
  uint4 h_reg[4];
  float ss = 0.0f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int idx8 = tid + c * 256;
    if (idx8 < k8) {
      uint4 xv = x_u4[idx8];
      if constexpr (kFusedAdd) {
        const uint4 rv = kPreloadRes ? r_reg[c] : r_u4[idx8];
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
        const __nv_bfloat162* rp = reinterpret_cast<const __nv_bfloat162*>(&rv);
        __nv_bfloat162* hp = reinterpret_cast<__nv_bfloat162*>(&h_reg[c]);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const float2 xf = __bfloat1622float2(xp[p]);
          const float2 rf = __bfloat1622float2(rp[p]);
          const __nv_bfloat162 hv = __floats2bfloat162_rn(xf.x + rf.x, xf.y + rf.y);
          hp[p] = hv;
          const float2 hf = __bfloat1622float2(hv);
          ss = __fmaf_rn(hf.x, hf.x, ss);
          ss = __fmaf_rn(hf.y, hf.y, ss);
        }
      } else {
        h_reg[c] = xv;
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const float2 xf = __bfloat1622float2(xp[p]);
          ss = __fmaf_rn(xf.x, xf.x, ss);
          ss = __fmaf_rn(xf.y, xf.y, ss);
        }
      }
    }
  }

#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, o);
  }
  if (lane == 0) s_warp[warp] = ss;
  __syncthreads();
  float tot = s_warp[lane & 7];
#pragma unroll
  for (int o = 4; o > 0; o >>= 1) {
    tot += __shfl_xor_sync(0xffffffffu, tot, o);
  }
  tot = __shfl_sync(0xffffffffu, tot, 0);
  const float inv_rms = rsqrtf(tot / static_cast<float>(K) + eps);

  uint4 out_reg[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int idx8 = tid + c * 256;
    if (idx8 < k8) {
      const __nv_bfloat162* hp = reinterpret_cast<const __nv_bfloat162*>(&h_reg[c]);
      const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&w_reg[c]);
      __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&out_reg[c]);
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const float2 hf = __bfloat1622float2(hp[p]);
        const float2 wf = __bfloat1622float2(wp[p]);
        op[p] = __floats2bfloat162_rn(hf.x * inv_rms * wf.x, hf.y * inv_rms * wf.y);
      }
    }
  }

  uint4* out_u4 = reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * K);
  uint4* rout_u4 = kFusedAdd ? reinterpret_cast<uint4*>(residual_out + static_cast<size_t>(row) * K) : nullptr;
  uint4* sn_u4 = reinterpret_cast<uint4*>(s_norm);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int idx8 = tid + c * 256;
    if (idx8 < k8) {
      out_u4[idx8] = out_reg[c];
      if constexpr (kEmitRot) {
        sn_u4[idx8] = out_reg[c];
      }
      if constexpr (kFusedAdd) {
        if (rout_u4 != nullptr) {
          rout_u4[idx8] = h_reg[c];
        }
      }
    }
  }
  if constexpr (kEmitRot) {
    __syncthreads();
    const int num_kg = K >> 6;
    __nv_bfloat16* rot_row = rot_out + static_cast<size_t>(row) * K;
    for (int kg = warp; kg < num_kg; kg += 8) {
      const int col0 = kg * 64 + 2 * lane;
      const __nv_bfloat162 raw = *reinterpret_cast<const __nv_bfloat162*>(&s_norm[col0]);
      const float2 rf = __bfloat1622float2(raw);
      float v0 = rf.x * dflash_rad_sign(col0 + 0);
      float v1 = rf.y * dflash_rad_sign(col0 + 1);
      dflash_warp_fwht64(v0, v1, lane);
      *reinterpret_cast<__nv_bfloat162*>(rot_row + col0) = __floats2bfloat162_rn(v0, v1);
    }
  }
}

}  // namespace

void dflash_kv_post(
    torch::Tensor all_kv_flat,
    torch::Tensor positions,
    torch::Tensor k_norm_w,
    torch::Tensor cos_sin_cache,
    const std::vector<torch::Tensor>& k_caches,
    const std::vector<torch::Tensor>& v_caches,
    const std::vector<torch::Tensor>& slot_mappings,
    int64_t nkv,
    int64_t block_size,
    double eps,
    bool is_neox) {
  const int L = static_cast<int>(k_caches.size());
  TORCH_CHECK(L >= 1 && L <= kMaxLayers &&
              static_cast<int>(v_caches.size()) == L &&
              static_cast<int>(slot_mappings.size()) == L);
  TORCH_CHECK(all_kv_flat.is_cuda() && all_kv_flat.scalar_type() == torch::kBFloat16 &&
              all_kv_flat.dim() == 2 && all_kv_flat.is_contiguous());
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 && positions.is_contiguous());
  TORCH_CHECK(k_norm_w.is_cuda() && k_norm_w.scalar_type() == torch::kBFloat16 &&
              k_norm_w.dim() == 2 && k_norm_w.size(0) == L && k_norm_w.size(1) == kHeadDim &&
              k_norm_w.is_contiguous());
  TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == kHeadDim &&
              cos_sin_cache.is_contiguous());
  const int M = static_cast<int>(all_kv_flat.size(0));
  TORCH_CHECK(all_kv_flat.size(1) == L * 2 * nkv * kHeadDim && positions.numel() >= M);
  if (M == 0) return;

  KvPostDesc desc{};
  for (int l = 0; l < L; ++l) {
    const auto& kc = k_caches[l];
    const auto& vc = v_caches[l];
    const auto& sm = slot_mappings[l];
    TORCH_CHECK(kc.is_cuda() && kc.scalar_type() == torch::kBFloat16 && kc.dim() == 4 && kc.stride(3) == 1);
    TORCH_CHECK(vc.is_cuda() && vc.scalar_type() == torch::kBFloat16 && vc.dim() == 4 && vc.stride(3) == 1);
    TORCH_CHECK(sm.is_cuda() && sm.scalar_type() == torch::kInt64 && sm.is_contiguous() && sm.numel() >= M);
    desc.k_cache[l] = reinterpret_cast<__nv_bfloat16*>(kc.data_ptr());
    desc.v_cache[l] = reinterpret_cast<__nv_bfloat16*>(vc.data_ptr());
    desc.slots[l] = sm.data_ptr<int64_t>();
    desc.k_sb[l] = kc.stride(0); desc.k_sn[l] = kc.stride(1); desc.k_sh[l] = kc.stride(2);
    desc.v_sb[l] = vc.stride(0); desc.v_sn[l] = vc.stride(1); desc.v_sh[l] = vc.stride(2);
  }

  const int total_warps = M * L * static_cast<int>(nkv);
  const int blocks = (total_warps + 3) / 4;
  const int cache_rows = static_cast<int>(cos_sin_cache.size(0));
  const bool cache_f32 = (cos_sin_cache.scalar_type() == torch::kFloat32);
  TORCH_CHECK(cache_f32 || cos_sin_cache.scalar_type() == torch::kBFloat16);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int pdl_mode = dflash_fuse_pdl_mode();

#define LAUNCH_KVP(NEOX, FP32) \
  dflash_launch_pdl( \
      dflash_kv_post_kernel<NEOX, FP32>, blocks, 128, st, pdl_mode, \
      reinterpret_cast<const __nv_bfloat16*>(all_kv_flat.data_ptr()), \
      positions.data_ptr<int64_t>(), \
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w.data_ptr()), \
      static_cast<const void*>(cos_sin_cache.data_ptr()), \
      desc, M, L, static_cast<int>(nkv), static_cast<int>(block_size), cache_rows, static_cast<float>(eps), pdl_mode)

  if (is_neox) {
    if (cache_f32) { LAUNCH_KVP(true, true); }
    else { LAUNCH_KVP(true, false); }
  } else {
    if (cache_f32) { LAUNCH_KVP(false, true); }
    else { LAUNCH_KVP(false, false); }
  }
#undef LAUNCH_KVP
}

std::tuple<torch::Tensor, torch::Tensor> dflash_qkv_norm_rope(
    torch::Tensor qkv,
    torch::Tensor positions,
    torch::Tensor q_norm_w,
    torch::Tensor k_norm_w,
    torch::Tensor cos_sin_cache,
    int64_t nq,
    int64_t nkv,
    double eps,
    bool is_neox,
    c10::optional<torch::Tensor> slot_mapping,
    c10::optional<torch::Tensor> k_cache,
    c10::optional<torch::Tensor> v_cache,
    int64_t block_size) {
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == torch::kBFloat16 &&
              qkv.dim() == 2 && qkv.is_contiguous());
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 && positions.is_contiguous());
  TORCH_CHECK(q_norm_w.is_cuda() && q_norm_w.scalar_type() == torch::kBFloat16 &&
              q_norm_w.numel() == kHeadDim && q_norm_w.is_contiguous());
  TORCH_CHECK(k_norm_w.is_cuda() && k_norm_w.scalar_type() == torch::kBFloat16 &&
              k_norm_w.numel() == kHeadDim && k_norm_w.is_contiguous());
  TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.dim() == 2 &&
              cos_sin_cache.size(1) == kHeadDim && cos_sin_cache.is_contiguous());
  const int M = static_cast<int>(qkv.size(0));
  TORCH_CHECK(qkv.size(1) == (nq + 2 * nkv) * kHeadDim && positions.numel() >= M);

  auto q_out = torch::empty({M, nq * kHeadDim}, qkv.options());
  auto k_out = torch::empty({M, nkv * kHeadDim}, qkv.options());
  if (M == 0) return {q_out, k_out};

  QkvCacheDesc cdesc{};
  if (slot_mapping.has_value() && slot_mapping->defined() &&
      k_cache.has_value() && k_cache->defined() &&
      v_cache.has_value() && v_cache->defined()) {
    TORCH_CHECK(slot_mapping->is_cuda() && slot_mapping->scalar_type() == torch::kInt64 &&
                slot_mapping->is_contiguous() && slot_mapping->numel() >= M);
    TORCH_CHECK(k_cache->is_cuda() && k_cache->scalar_type() == torch::kBFloat16 &&
                k_cache->dim() == 4 && k_cache->stride(3) == 1 &&
                k_cache->size(2) == nkv && k_cache->size(3) == kHeadDim);
    TORCH_CHECK(v_cache->is_cuda() && v_cache->scalar_type() == torch::kBFloat16 &&
                v_cache->dim() == 4 && v_cache->stride(3) == 1 &&
                v_cache->size(2) == nkv && v_cache->size(3) == kHeadDim);
    TORCH_CHECK(block_size > 0 && k_cache->size(1) == block_size && v_cache->size(1) == block_size);
    cdesc.k_cache = reinterpret_cast<__nv_bfloat16*>(k_cache->data_ptr());
    cdesc.v_cache = reinterpret_cast<__nv_bfloat16*>(v_cache->data_ptr());
    cdesc.slots = slot_mapping->data_ptr<int64_t>();
    cdesc.k_sb = k_cache->stride(0); cdesc.k_sn = k_cache->stride(1); cdesc.k_sh = k_cache->stride(2);
    cdesc.v_sb = v_cache->stride(0); cdesc.v_sn = v_cache->stride(1); cdesc.v_sh = v_cache->stride(2);
    cdesc.block_size = static_cast<int>(block_size);
    cdesc.enabled = 1;
  }

  const int total_warps = M * static_cast<int>(nq + nkv);
  const int blocks = (total_warps + 3) / 4;
  const int cache_rows = static_cast<int>(cos_sin_cache.size(0));
  const bool cache_f32 = (cos_sin_cache.scalar_type() == torch::kFloat32);
  TORCH_CHECK(cache_f32 || cos_sin_cache.scalar_type() == torch::kBFloat16);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int pdl_mode = dflash_fuse_pdl_mode();

#define LAUNCH_QKVNR(NEOX, FP32) \
  dflash_launch_pdl( \
      dflash_qkv_norm_rope_kernel<NEOX, FP32>, blocks, 128, st, pdl_mode, \
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), \
      positions.data_ptr<int64_t>(), \
      reinterpret_cast<const __nv_bfloat16*>(q_norm_w.data_ptr()), \
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w.data_ptr()), \
      static_cast<const void*>(cos_sin_cache.data_ptr()), \
      reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr()), \
      reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr()), \
      cdesc, M, static_cast<int>(nq), static_cast<int>(nkv), cache_rows, static_cast<float>(eps), pdl_mode)

  if (is_neox) {
    if (cache_f32) { LAUNCH_QKVNR(true, true); }
    else { LAUNCH_QKVNR(true, false); }
  } else {
    if (cache_f32) { LAUNCH_QKVNR(false, true); }
    else { LAUNCH_QKVNR(false, false); }
  }
#undef LAUNCH_QKVNR
  return {q_out, k_out};
}

void dflash_prepare_inputs(
    torch::Tensor out_input_ids,
    torch::Tensor out_query_positions,
    torch::Tensor out_query_start_loc,
    torch::Tensor out_seq_lens,
    torch::Tensor out_query_slot_mapping,
    torch::Tensor out_context_positions,
    torch::Tensor out_context_slot_mapping,
    torch::Tensor out_sample_indices,
    torch::Tensor out_sample_pos,
    torch::Tensor out_sample_idx_mapping,
    torch::Tensor out_temperature,
    torch::Tensor out_seeds,
    torch::Tensor target_positions,
    torch::Tensor target_query_start_loc,
    torch::Tensor idx_mapping,
    torch::Tensor last_sampled,
    torch::Tensor next_prefill_tokens,
    torch::Tensor num_sampled,
    torch::Tensor num_rejected,
    torch::Tensor in_temperature,
    torch::Tensor in_seeds,
    torch::Tensor block_table,
    int64_t block_table_stride,
    int64_t parallel_drafting_token_id,
    int64_t block_size,
    int64_t num_query_per_req,
    int64_t num_speculative_steps,
    int64_t max_num_reqs,
    int64_t max_num_tokens,
    int64_t max_model_len,
    int64_t pad_slot_id,
    c10::optional<torch::Tensor> target_input_ids,
    int64_t tail_num_tokens,
    c10::optional<torch::Tensor> in_static_vocab,
    c10::optional<torch::Tensor> tail_ids,
    c10::optional<torch::Tensor> tail_count,
    c10::optional<torch::Tensor> tail_seen,
    c10::optional<torch::Tensor> hist,
    c10::optional<torch::Tensor> hist_len,
    c10::optional<torch::Tensor> post_sampled_tokens,
    c10::optional<torch::Tensor> post_num_computed_tokens,
    c10::optional<torch::Tensor> post_all_token_ids,
    c10::optional<torch::Tensor> post_total_len,
    c10::optional<torch::Tensor> post_output_bin_counts) {
  TORCH_CHECK(out_input_ids.scalar_type() == torch::kInt32 && out_query_positions.scalar_type() == torch::kInt64 &&
                  out_query_start_loc.scalar_type() == torch::kInt32 && out_seq_lens.scalar_type() == torch::kInt32 &&
                  out_query_slot_mapping.scalar_type() == torch::kInt64 &&
                  out_context_positions.scalar_type() == torch::kInt64 &&
                  out_context_slot_mapping.scalar_type() == torch::kInt64 &&
                  out_sample_indices.scalar_type() == torch::kInt64 &&
                  out_sample_pos.scalar_type() == torch::kInt64 &&
                  out_sample_idx_mapping.scalar_type() == torch::kInt32 &&
                  out_temperature.scalar_type() == torch::kFloat32 && out_seeds.scalar_type() == torch::kInt64,
              "dflash_prepare_inputs bad output dtypes");
  TORCH_CHECK(target_positions.scalar_type() == torch::kInt64 &&
                  target_query_start_loc.scalar_type() == torch::kInt32 &&
                  idx_mapping.scalar_type() == torch::kInt32 &&
                  num_sampled.scalar_type() == torch::kInt32 &&
                  num_rejected.scalar_type() == torch::kInt32 &&
                  in_temperature.scalar_type() == torch::kFloat32 &&
                  in_seeds.scalar_type() == torch::kInt64 &&
                  block_table.scalar_type() == torch::kInt32,
              "dflash_prepare_inputs bad input dtypes");

  const uint8_t* static_vocab_ptr = nullptr;
  int32_t* tail_ids_ptr = nullptr;
  int32_t* tail_count_ptr = nullptr;
  int32_t* tail_seen_ptr = nullptr;
  int vocab_size = 0;
  int max_tail = 0;
  int eff_tail_tokens = 0;
  bool tail_i64 = false;
  const void* target_ids_raw = nullptr;

  int32_t* hist_ptr = nullptr;
  int32_t* hist_len_ptr = nullptr;
  int hist_cap = 0;

  if (target_input_ids.has_value() && target_input_ids->defined() && tail_num_tokens > 0) {
    TORCH_CHECK(target_input_ids->is_contiguous(), "dflash_prepare_inputs non-contiguous target_input_ids");
    tail_i64 = (target_input_ids->scalar_type() == torch::kInt64);
    TORCH_CHECK(tail_i64 || target_input_ids->scalar_type() == torch::kInt32,
                "dflash_prepare_inputs bad target_input_ids dtype");
    target_ids_raw = target_input_ids->data_ptr();
    eff_tail_tokens = static_cast<int>(tail_num_tokens);
    if (in_static_vocab.has_value() && in_static_vocab->defined() &&
        tail_ids.has_value() && tail_ids->defined() &&
        tail_count.has_value() && tail_count->defined() &&
        tail_seen.has_value() && tail_seen->defined()) {
      static_vocab_ptr = in_static_vocab->data_ptr<uint8_t>();
      tail_ids_ptr = tail_ids->data_ptr<int32_t>();
      tail_count_ptr = tail_count->data_ptr<int32_t>();
      tail_seen_ptr = tail_seen->data_ptr<int32_t>();
      vocab_size = static_cast<int>(in_static_vocab->numel());
      max_tail = static_cast<int>(tail_ids->numel());
    }
    if (hist.has_value() && hist->defined() && hist->numel() > 0 &&
        hist_len.has_value() && hist_len->defined() && hist_len->numel() >= 1) {
      TORCH_CHECK(hist->is_cuda() && hist->scalar_type() == torch::kInt32 && hist->is_contiguous());
      TORCH_CHECK(hist_len->is_cuda() && hist_len->scalar_type() == torch::kInt32);
      hist_ptr = hist->data_ptr<int32_t>();
      hist_len_ptr = hist_len->data_ptr<int32_t>();
      hist_cap = static_cast<int>(hist->numel());
    }
  }

  const bool ls_i64 = (last_sampled.scalar_type() == torch::kInt64);
  TORCH_CHECK(ls_i64 || last_sampled.scalar_type() == torch::kInt32, "bad last_sampled dtype");
  TORCH_CHECK(next_prefill_tokens.scalar_type() == torch::kInt32, "bad next_prefill_tokens dtype");

  const int64_t* post_samp_ptr = nullptr;
  int32_t* post_ncomp_ptr = nullptr;
  int32_t* post_all_ptr = nullptr;
  int64_t post_all_stride = 0;
  int32_t* post_tlen_ptr = nullptr;
  int32_t* post_bin_ptr = nullptr;
  int64_t post_bin_stride = 0;
  if (post_sampled_tokens.has_value() && post_sampled_tokens->defined()) {
    TORCH_CHECK(post_sampled_tokens->is_cuda() && post_sampled_tokens->scalar_type() == torch::kInt64 &&
                    post_sampled_tokens->stride(-1) == 1,
                "dflash_prepare_inputs bad post_sampled_tokens");
    TORCH_CHECK(post_num_computed_tokens.has_value() && post_num_computed_tokens->defined() &&
                    post_num_computed_tokens->scalar_type() == torch::kInt32 &&
                    post_num_computed_tokens->is_contiguous(),
                "dflash_prepare_inputs bad post_num_computed_tokens");
    TORCH_CHECK(post_all_token_ids.has_value() && post_all_token_ids->defined() &&
                    post_all_token_ids->scalar_type() == torch::kInt32 &&
                    post_all_token_ids->stride(-1) == 1,
                "dflash_prepare_inputs bad post_all_token_ids");
    TORCH_CHECK(post_total_len.has_value() && post_total_len->defined() &&
                    post_total_len->scalar_type() == torch::kInt32 &&
                    post_total_len->is_contiguous(),
                "dflash_prepare_inputs bad post_total_len");
    post_samp_ptr = post_sampled_tokens->data_ptr<int64_t>();
    post_ncomp_ptr = post_num_computed_tokens->data_ptr<int32_t>();
    post_all_ptr = post_all_token_ids->data_ptr<int32_t>();
    post_all_stride = post_all_token_ids->stride(0);
    post_tlen_ptr = post_total_len->data_ptr<int32_t>();
    if (post_output_bin_counts.has_value() && post_output_bin_counts->defined()) {
      TORCH_CHECK(post_output_bin_counts->scalar_type() == torch::kInt32 &&
                      post_output_bin_counts->stride(-1) == 1,
                  "dflash_prepare_inputs bad post_output_bin_counts");
      post_bin_ptr = post_output_bin_counts->data_ptr<int32_t>();
      post_bin_stride = post_output_bin_counts->stride(0);
    }
  }

  static const int include_bonus = []() -> int {
    const char* s = std::getenv("MUSE_TAIL_ZERO_LAG");
    return (!s || !*s || std::atoi(s) != 0) ? 1 : 0;
  }();
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int pdl = dflash_fuse_pdl_mode();
#define LAUNCH_PREP(LST, TIDT) \
  dflash_launch_pdl( \
      dflash_prepare_inputs_kernel<LST, TIDT>, 1, 256, st, pdl, \
      out_input_ids.data_ptr<int32_t>(), \
      out_query_positions.data_ptr<int64_t>(), \
      out_query_start_loc.data_ptr<int32_t>(), \
      out_seq_lens.data_ptr<int32_t>(), \
      out_query_slot_mapping.data_ptr<int64_t>(), \
      out_context_positions.data_ptr<int64_t>(), \
      out_context_slot_mapping.data_ptr<int64_t>(), \
      out_sample_indices.data_ptr<int64_t>(), \
      out_sample_pos.data_ptr<int64_t>(), \
      out_sample_idx_mapping.data_ptr<int32_t>(), \
      out_temperature.data_ptr<float>(), \
      out_seeds.data_ptr<int64_t>(), \
      target_positions.data_ptr<int64_t>(), \
      target_query_start_loc.data_ptr<int32_t>(), \
      idx_mapping.data_ptr<int32_t>(), \
      last_sampled.data_ptr<LST>(), \
      next_prefill_tokens.data_ptr<int32_t>(), \
      num_sampled.data_ptr<int32_t>(), \
      num_rejected.data_ptr<int32_t>(), \
      in_temperature.data_ptr<float>(), \
      in_seeds.data_ptr<int64_t>(), \
      block_table.data_ptr<int32_t>(), \
      block_table_stride, \
      static_cast<int32_t>(parallel_drafting_token_id), \
      block_size, \
      static_cast<int>(num_query_per_req), \
      static_cast<int>(num_speculative_steps), \
      static_cast<int>(max_num_reqs), \
      static_cast<int>(max_num_tokens), \
      max_model_len, \
      pad_slot_id, \
      reinterpret_cast<const TIDT*>(target_ids_raw), \
      eff_tail_tokens, \
      static_vocab_ptr, \
      tail_ids_ptr, \
      tail_count_ptr, \
      tail_seen_ptr, \
      vocab_size, \
      max_tail, \
      include_bonus, \
      hist_ptr, \
      hist_len_ptr, \
      hist_cap, \
      post_samp_ptr, \
      post_ncomp_ptr, \
      post_all_ptr, \
      post_all_stride, \
      post_tlen_ptr, \
      post_bin_ptr, \
      post_bin_stride, \
      pdl)

  if (ls_i64) {
    if (tail_i64) { LAUNCH_PREP(int64_t, int64_t); }
    else { LAUNCH_PREP(int64_t, int32_t); }
  } else {
    if (tail_i64) { LAUNCH_PREP(int32_t, int64_t); }
    else { LAUNCH_PREP(int32_t, int32_t); }
  }
#undef LAUNCH_PREP
  vq4d_clear_all_prerot();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor dflash_rms_norm(torch::Tensor input, torch::Tensor weight, double eps) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kBFloat16 && input.dim() == 2 && input.is_contiguous(),
              "dflash_rms_norm expects contiguous CUDA BF16 [M, K]");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == torch::kBFloat16 && weight.dim() == 1 && weight.is_contiguous(),
              "dflash_rms_norm expects contiguous CUDA BF16 [K] weight");
  const int M = static_cast<int>(input.size(0));
  const int K = static_cast<int>(input.size(1));
  TORCH_CHECK(weight.size(0) == K && K > 0 && K <= 8192 && (K % 8) == 0,
              "dflash_rms_norm expects K<=8192 and K%8==0");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(input.data_ptr()) & 15u) == 0u &&
              (reinterpret_cast<uintptr_t>(weight.data_ptr()) & 15u) == 0u,
              "dflash_rms_norm expects 16B-aligned tensors");
  auto out = torch::empty_like(input);
  if (M == 0) {
    vq4d_clear_all_prerot();
    return out;
  }
  static const bool selective_rot = []() -> bool {
    const char* s = std::getenv("MUSE_DFLASH_NORM_SELECTIVE");
    return (!s || !*s) ? true : (std::atoi(s) != 0);
  }();
  const void* w_tag = selective_rot ? weight.data_ptr() : nullptr;
  __nv_bfloat16* rot_out =
      (M <= 16 && (K & 63) == 0 && dflash_norm_fuse_rot_enabled())
          ? vq4d_alloc_norm_prerot(out.data_ptr(), w_tag, M, K, out.options())
          : nullptr;
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int pdl = dflash_fuse_pdl_mode();
  auto k = (rot_out != nullptr) ? dflash_rms_norm_kernel<false, false, true>
                                : dflash_rms_norm_kernel<false, false, false>;
  dflash_launch_pdl(
      k, M, 256, st, pdl,
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      static_cast<const __nv_bfloat16*>(nullptr),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<__nv_bfloat16*>(nullptr),
      rot_out,
      K, static_cast<float>(eps), pdl);
  return out;
}

std::tuple<torch::Tensor, torch::Tensor> dflash_fused_add_rms_norm(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kBFloat16 && input.dim() == 2 && input.is_contiguous(),
              "dflash_fused_add_rms_norm expects contiguous CUDA BF16 [M, K] input");
  TORCH_CHECK(residual.is_cuda() && residual.scalar_type() == torch::kBFloat16 && residual.sizes() == input.sizes() &&
              residual.is_contiguous(),
              "dflash_fused_add_rms_norm expects contiguous CUDA BF16 [M, K] residual");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == torch::kBFloat16 && weight.dim() == 1 && weight.is_contiguous(),
              "dflash_fused_add_rms_norm expects contiguous CUDA BF16 [K] weight");
  const int M = static_cast<int>(input.size(0));
  const int K = static_cast<int>(input.size(1));
  TORCH_CHECK(weight.size(0) == K && K > 0 && K <= 8192 && (K % 8) == 0,
              "dflash_fused_add_rms_norm expects K<=8192 and K%8==0");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(input.data_ptr()) & 15u) == 0u &&
              (reinterpret_cast<uintptr_t>(residual.data_ptr()) & 15u) == 0u &&
              (reinterpret_cast<uintptr_t>(weight.data_ptr()) & 15u) == 0u,
              "dflash_fused_add_rms_norm expects 16B-aligned tensors");
  auto out = torch::empty_like(input);
  auto res_out = torch::empty_like(residual);
  if (M == 0) {
    vq4d_clear_all_prerot();
    return std::make_tuple(out, res_out);
  }
  static const bool selective_rot = []() -> bool {
    const char* s = std::getenv("MUSE_DFLASH_NORM_SELECTIVE");
    return (!s || !*s) ? true : (std::atoi(s) != 0);
  }();
  const void* w_tag = selective_rot ? weight.data_ptr() : nullptr;
  __nv_bfloat16* rot_out =
      (M <= 16 && (K & 63) == 0 && dflash_norm_fuse_rot_enabled())
          ? vq4d_alloc_norm_prerot(out.data_ptr(), w_tag, M, K, out.options())
          : nullptr;
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int pdl = dflash_fuse_pdl_mode();
  const bool preload_res = (input.data_ptr() != residual.data_ptr());
  auto k = (rot_out != nullptr)
               ? (preload_res ? dflash_rms_norm_kernel<true, true, true> : dflash_rms_norm_kernel<true, false, true>)
               : (preload_res ? dflash_rms_norm_kernel<true, true, false> : dflash_rms_norm_kernel<true, false, false>);
  dflash_launch_pdl(
      k, M, 256, st, pdl,
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(res_out.data_ptr()),
      rot_out,
      K, static_cast<float>(eps), pdl);
  return std::make_tuple(out, res_out);
}
