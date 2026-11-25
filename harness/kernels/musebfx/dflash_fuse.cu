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
#include <stdint.h>
#include <vector>

namespace {

constexpr int kHeadDim = 128;
constexpr int kMaxLayers = 8;

template <bool kNeox, bool kCacheFP32>
__device__ __forceinline__ void apply_rmsnorm_rope_hd128(
    const __nv_bfloat16* __restrict__ in_ptr,
    const __nv_bfloat16* __restrict__ norm_w,
    const void* __restrict__ cos_sin_cache,
    int64_t pos,
    int cache_rows,
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

  const uint2 wraw = *reinterpret_cast<const uint2*>(norm_w + lane * 4);
  const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wraw);
  // Round to BF16 between RMSNorm and RoPE to match ops.rms_norm -> ops.rotary_embedding verbatim.
  const float y0 = __bfloat162float(__float2bfloat16_rn(x0 * inv_rms * __bfloat162float(wb[0])));
  const float y1 = __bfloat162float(__float2bfloat16_rn(x1 * inv_rms * __bfloat162float(wb[1])));
  const float y2 = __bfloat162float(__float2bfloat16_rn(x2 * inv_rms * __bfloat162float(wb[2])));
  const float y3 = __bfloat162float(__float2bfloat16_rn(x3 * inv_rms * __bfloat162float(wb[3])));

  if (pos < 0) pos += cache_rows;
  float o0, o1, o2, o3;

  if constexpr (kNeox) {
    const float p0 = __shfl_xor_sync(0xffffffffu, y0, 16);
    const float p1 = __shfl_xor_sync(0xffffffffu, y1, 16);
    const float p2 = __shfl_xor_sync(0xffffffffu, y2, 16);
    const float p3 = __shfl_xor_sync(0xffffffffu, y3, 16);
    const int d_lo = (lane & 15) * 4;
    float c0, c1, c2, c3, s0, s1, s2, s3;
    if constexpr (kCacheFP32) {
      const float* row = reinterpret_cast<const float*>(cos_sin_cache) + pos * 128;
      const float4 cv = *reinterpret_cast<const float4*>(row + d_lo);
      const float4 sv = *reinterpret_cast<const float4*>(row + 64 + d_lo);
      c0 = cv.x; c1 = cv.y; c2 = cv.z; c3 = cv.w;
      s0 = sv.x; s1 = sv.y; s2 = sv.z; s3 = sv.w;
    } else {
      const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache) + pos * 128;
      const uint2 cv = *reinterpret_cast<const uint2*>(row + d_lo);
      const uint2 sv = *reinterpret_cast<const uint2*>(row + 64 + d_lo);
      const __nv_bfloat16* cb = reinterpret_cast<const __nv_bfloat16*>(&cv);
      const __nv_bfloat16* sb = reinterpret_cast<const __nv_bfloat16*>(&sv);
      c0 = __bfloat162float(cb[0]); c1 = __bfloat162float(cb[1]);
      c2 = __bfloat162float(cb[2]); c3 = __bfloat162float(cb[3]);
      s0 = __bfloat162float(sb[0]); s1 = __bfloat162float(sb[1]);
      s2 = __bfloat162float(sb[2]); s3 = __bfloat162float(sb[3]);
    }
    if (lane < 16) {
      o0 = __fmaf_rn(y0, c0, -__fmul_rn(p0, s0));
      o1 = __fmaf_rn(y1, c1, -__fmul_rn(p1, s1));
      o2 = __fmaf_rn(y2, c2, -__fmul_rn(p2, s2));
      o3 = __fmaf_rn(y3, c3, -__fmul_rn(p3, s3));
    } else {
      o0 = __fmaf_rn(y0, c0, __fmul_rn(p0, s0));
      o1 = __fmaf_rn(y1, c1, __fmul_rn(p1, s1));
      o2 = __fmaf_rn(y2, c2, __fmul_rn(p2, s2));
      o3 = __fmaf_rn(y3, c3, __fmul_rn(p3, s3));
    }
  } else {
    const int d_lo = lane * 2;
    float c0, c1, s0, s1;
    if constexpr (kCacheFP32) {
      const float* row = reinterpret_cast<const float*>(cos_sin_cache) + pos * 128;
      c0 = row[d_lo + 0]; c1 = row[d_lo + 1];
      s0 = row[64 + d_lo + 0]; s1 = row[64 + d_lo + 1];
    } else {
      const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache) + pos * 128;
      c0 = __bfloat162float(row[d_lo + 0]); c1 = __bfloat162float(row[d_lo + 1]);
      s0 = __bfloat162float(row[64 + d_lo + 0]); s1 = __bfloat162float(row[64 + d_lo + 1]);
    }
    o0 = __fmaf_rn(y0, c0, -__fmul_rn(y1, s0));
    o1 = __fmaf_rn(y1, c0,  __fmul_rn(y0, s0));
    o2 = __fmaf_rn(y2, c1, -__fmul_rn(y3, s1));
    o3 = __fmaf_rn(y3, c1,  __fmul_rn(y2, s1));
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
    int M, int L, int nkv, int block_size, int cache_rows, float eps) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int total = M * L * nkv;
  if (warp_id >= total) return;
  const int lane = threadIdx.x & 31;
  const int h = warp_id % nkv;
  const int ml = warp_id / nkv;
  const int l = ml % L;
  const int m = ml / L;

  const int64_t slot = desc.slots[l][m];
  if (slot < 0) return;
  const int64_t block_idx = slot / block_size;
  const int64_t block_off = slot - block_idx * block_size;

  const int kv_stride = nkv * kHeadDim;
  const size_t base_ml = static_cast<size_t>(m * L + l) * (2 * kv_stride);
  const __nv_bfloat16* k_in = all_kv_flat + base_ml + h * kHeadDim;
  const __nv_bfloat16* v_in = all_kv_flat + base_ml + kv_stride + h * kHeadDim;
  const __nv_bfloat16* nw = k_norm_w + static_cast<size_t>(l) * kHeadDim;

  uint2 k_out_v;
  apply_rmsnorm_rope_hd128<kNeox, kCacheFP32>(k_in, nw, cos_sin_cache, positions[m], cache_rows, eps, lane, k_out_v);
  const uint2 v_out_v = *reinterpret_cast<const uint2*>(v_in + lane * 4);

  __nv_bfloat16* k_dst = desc.k_cache[l] + block_idx * desc.k_sb[l] + block_off * desc.k_sn[l] + h * desc.k_sh[l];
  __nv_bfloat16* v_dst = desc.v_cache[l] + block_idx * desc.v_sb[l] + block_off * desc.v_sn[l] + h * desc.v_sh[l];
  *reinterpret_cast<uint2*>(k_dst + lane * 4) = k_out_v;
  *reinterpret_cast<uint2*>(v_dst + lane * 4) = v_out_v;
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

#define LAUNCH_KVP(NEOX, FP32) \
  dflash_kv_post_kernel<NEOX, FP32><<<blocks, 128, 0, st>>>( \
      reinterpret_cast<const __nv_bfloat16*>(all_kv_flat.data_ptr()), \
      positions.data_ptr<int64_t>(), \
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w.data_ptr()), \
      cos_sin_cache.data_ptr(), \
      desc, M, L, static_cast<int>(nkv), static_cast<int>(block_size), cache_rows, static_cast<float>(eps))

  if (is_neox) {
    if (cache_f32) { LAUNCH_KVP(true, true); }
    else { LAUNCH_KVP(true, false); }
  } else {
    if (cache_f32) { LAUNCH_KVP(false, true); }
    else { LAUNCH_KVP(false, false); }
  }
#undef LAUNCH_KVP
}
