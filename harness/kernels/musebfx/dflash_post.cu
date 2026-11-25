// SPDX-License-Identifier: Apache-2.0
// DFlash drafter context-KV & per-layer query QKV fused kernels (0-ULP bitwise exact).
// 1. dflash_kv_post: fuses 9 eager context-KV launches (permute+contiguous, 5-layer grouped
//    K RMSNorm, positions repeat, NEOX RoPE, and 5 paged BF16 KV-cache scatter writes) into 1 launch.
// 2. dflash_qkv_norm_rope_cache: fuses per-layer query qkv split + weighted Q RMSNorm + weighted
//    K RMSNorm + NEOX RoPE + paged BF16 KV-cache scatter write into 1 launch inside the FULL graph.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>
#include <math.h>
#include <vector>

namespace {

struct DFlashKVPtrs {
  __nv_bfloat16* p0;
  __nv_bfloat16* p1;
  __nv_bfloat16* p2;
  __nv_bfloat16* p3;
  __nv_bfloat16* p4;
};

template <bool CacheFP32>
__device__ __forceinline__ uint2 dflash_rms_rope_lane(
    uint2 xraw,
    uint2 wraw,
    const void* __restrict__ cos_sin_cache,
    int64_t pos,
    int cache_rows,
    int lane,
    float eps) {
  const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xraw);
  const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wraw);
  const float x0 = __bfloat162float(xb[0]), x1 = __bfloat162float(xb[1]);
  const float x2 = __bfloat162float(xb[2]), x3 = __bfloat162float(xb[3]);

  float ss = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float r = rsqrtf(ss * (1.0f / 128.0f) + eps);

  // Match vLLM ops.rms_norm -> BF16 intermediate before ops.rotary_embedding
  float y0 = __bfloat162float(__float2bfloat16_rn(x0 * r * __bfloat162float(wb[0])));
  float y1 = __bfloat162float(__float2bfloat16_rn(x1 * r * __bfloat162float(wb[1])));
  float y2 = __bfloat162float(__float2bfloat16_rn(x2 * r * __bfloat162float(wb[2])));
  float y3 = __bfloat162float(__float2bfloat16_rn(x3 * r * __bfloat162float(wb[3])));

  const float p0 = __shfl_xor_sync(0xffffffffu, y0, 16);
  const float p1 = __shfl_xor_sync(0xffffffffu, y1, 16);
  const float p2 = __shfl_xor_sync(0xffffffffu, y2, 16);
  const float p3 = __shfl_xor_sync(0xffffffffu, y3, 16);

  if (pos < 0) pos += cache_rows;
  const int d_lo = (lane & 15) * 4;
  float c0, c1, c2, c3, s0, s1, s2, s3;
  if constexpr (CacheFP32) {
    const float* row = reinterpret_cast<const float*>(cos_sin_cache) + pos * 128;
    const float4 cv = __ldg(reinterpret_cast<const float4*>(row + d_lo));
    const float4 sv = __ldg(reinterpret_cast<const float4*>(row + 64 + d_lo));
    c0 = cv.x; c1 = cv.y; c2 = cv.z; c3 = cv.w;
    s0 = sv.x; s1 = sv.y; s2 = sv.z; s3 = sv.w;
  } else {
    const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache) + pos * 128;
    const uint2 cv = __ldg(reinterpret_cast<const uint2*>(row + d_lo));
    const uint2 sv = __ldg(reinterpret_cast<const uint2*>(row + 64 + d_lo));
    const __nv_bfloat16* cb = reinterpret_cast<const __nv_bfloat16*>(&cv);
    const __nv_bfloat16* sb = reinterpret_cast<const __nv_bfloat16*>(&sv);
    c0 = __bfloat162float(cb[0]); c1 = __bfloat162float(cb[1]);
    c2 = __bfloat162float(cb[2]); c3 = __bfloat162float(cb[3]);
    s0 = __bfloat162float(sb[0]); s1 = __bfloat162float(sb[1]);
    s2 = __bfloat162float(sb[2]); s3 = __bfloat162float(sb[3]);
  }

  float o0, o1, o2, o3;
  if (lane < 16) {
    o0 = __fsub_rn(__fmul_rn(y0, c0), __fmul_rn(p0, s0));
    o1 = __fsub_rn(__fmul_rn(y1, c1), __fmul_rn(p1, s1));
    o2 = __fsub_rn(__fmul_rn(y2, c2), __fmul_rn(p2, s2));
    o3 = __fsub_rn(__fmul_rn(y3, c3), __fmul_rn(p3, s3));
  } else {
    o0 = __fadd_rn(__fmul_rn(y0, c0), __fmul_rn(p0, s0));
    o1 = __fadd_rn(__fmul_rn(y1, c1), __fmul_rn(p1, s1));
    o2 = __fadd_rn(__fmul_rn(y2, c2), __fmul_rn(p2, s2));
    o3 = __fadd_rn(__fmul_rn(y3, c3), __fmul_rn(p3, s3));
  }

  uint2 out_u2;
  __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&out_u2);
  ob[0] = __float2bfloat16_rn(o0);
  ob[1] = __float2bfloat16_rn(o1);
  ob[2] = __float2bfloat16_rn(o2);
  ob[3] = __float2bfloat16_rn(o3);
  return out_u2;
}

template <bool CacheFP32>
__global__ void __launch_bounds__(256) dflash_kv_post_kernel(
    const __nv_bfloat16* __restrict__ all_kv_flat,
    const __nv_bfloat16* __restrict__ k_norm_weights,  // [L, 128]
    const void* __restrict__ cos_sin_cache,            // [max_pos, 128]
    const int64_t* __restrict__ positions,             // [num_ctx]
    const int64_t* __restrict__ slot_mapping,          // [num_ctx] or nullptr
    DFlashKVPtrs kv_caches,
    int num_ctx, int num_layers, int num_kv_heads, int cache_rows,
    int block_size, int64_t stride_block, int64_t stride_kv_head, int64_t stride_slot, float eps) {
  const int ctx_idx = blockIdx.x;
  const int layer_idx = blockIdx.y;
  const int kv_head = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (ctx_idx >= num_ctx || layer_idx >= num_layers || kv_head >= num_kv_heads) return;

  const size_t pair_base = (((size_t)ctx_idx * num_layers + layer_idx) * 2) * ((size_t)num_kv_heads * 128);
  const __nv_bfloat16* k_src = all_kv_flat + pair_base + (size_t)kv_head * 128 + lane * 4;
  const __nv_bfloat16* v_src = k_src + (size_t)num_kv_heads * 128;

  const uint2 kraw = __ldg(reinterpret_cast<const uint2*>(k_src));
  const uint2 vraw = __ldg(reinterpret_cast<const uint2*>(v_src));
  const uint2 wraw = __ldg(reinterpret_cast<const uint2*>(k_norm_weights + (size_t)layer_idx * 128 + lane * 4));
  const int64_t pos = __ldg(positions + ctx_idx);

  const uint2 k_out = dflash_rms_rope_lane<CacheFP32>(kraw, wraw, cos_sin_cache, pos, cache_rows, lane, eps);

  if (slot_mapping == nullptr) return;
  const int64_t slot = __ldg(slot_mapping + ctx_idx);
  if (slot < 0) return;

  const int64_t blk = (block_size == 16) ? (slot >> 4) : (slot / block_size);
  const int64_t off = (block_size == 16) ? (slot & 15) : (slot - blk * block_size);
  __nv_bfloat16* kv_base = (layer_idx == 0) ? kv_caches.p0
                         : (layer_idx == 1) ? kv_caches.p1
                         : (layer_idx == 2) ? kv_caches.p2
                         : (layer_idx == 3) ? kv_caches.p3
                                            : kv_caches.p4;
  __nv_bfloat16* dst = kv_base + blk * stride_block + kv_head * stride_kv_head + off * stride_slot + lane * 4;
  *reinterpret_cast<uint2*>(dst) = k_out;
  *reinterpret_cast<uint2*>(dst + 128) = vraw;
}

// Grid: (10, M), Block: 128 threads (4 warps/block -> 40 warps/token = 32 Q heads + 8 K/V heads).
template <bool CacheFP32>
__global__ void __launch_bounds__(128) dflash_qkv_norm_rope_cache_kernel(
    const __nv_bfloat16* __restrict__ qkv,             // [M, 6144]
    const __nv_bfloat16* __restrict__ q_norm_weight,   // [128]
    const __nv_bfloat16* __restrict__ k_norm_weight,   // [128]
    const void* __restrict__ cos_sin_cache,            // [max_pos, 128]
    const int64_t* __restrict__ positions,             // [M]
    const int64_t* __restrict__ slot_mapping,          // [M] or nullptr
    __nv_bfloat16* __restrict__ q_out,                 // [M, 4096]
    __nv_bfloat16* __restrict__ k_out,                 // [M, 1024] or nullptr
    __nv_bfloat16* __restrict__ kv_cache,              // paged BF16 cache or nullptr
    int M, int cache_rows, int block_size,
    int64_t stride_block, int64_t stride_kv_head, int64_t stride_slot, int64_t v_offset, float eps) {
  const int tok_idx = blockIdx.y;
  const int h = blockIdx.x * 4 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (tok_idx >= M || h >= 40) return;

  const int64_t pos = __ldg(positions + tok_idx);
  const __nv_bfloat16* qkv_row = qkv + static_cast<size_t>(tok_idx) * 6144;

  if (h < 32) {
    const uint2 qraw = __ldg(reinterpret_cast<const uint2*>(qkv_row + h * 128 + lane * 4));
    const uint2 qw = __ldg(reinterpret_cast<const uint2*>(q_norm_weight + lane * 4));
    const uint2 q_rot = dflash_rms_rope_lane<CacheFP32>(qraw, qw, cos_sin_cache, pos, cache_rows, lane, eps);
    *reinterpret_cast<uint2*>(q_out + static_cast<size_t>(tok_idx) * 4096 + h * 128 + lane * 4) = q_rot;
  } else {
    const int kh = h - 32;
    const uint2 kraw = __ldg(reinterpret_cast<const uint2*>(qkv_row + 4096 + kh * 128 + lane * 4));
    const uint2 vraw = __ldg(reinterpret_cast<const uint2*>(qkv_row + 5120 + kh * 128 + lane * 4));
    const uint2 kw = __ldg(reinterpret_cast<const uint2*>(k_norm_weight + lane * 4));
    const uint2 k_rot = dflash_rms_rope_lane<CacheFP32>(kraw, kw, cos_sin_cache, pos, cache_rows, lane, eps);
    if (k_out != nullptr) {
      *reinterpret_cast<uint2*>(k_out + static_cast<size_t>(tok_idx) * 1024 + kh * 128 + lane * 4) = k_rot;
    }
    if (slot_mapping != nullptr && kv_cache != nullptr) {
      const int64_t slot = __ldg(slot_mapping + tok_idx);
      if (slot >= 0) {
        const int64_t blk = (block_size == 16) ? (slot >> 4) : (slot / block_size);
        const int64_t off = (block_size == 16) ? (slot & 15) : (slot - blk * block_size);
        __nv_bfloat16* dst = kv_cache + blk * stride_block + kh * stride_kv_head + off * stride_slot + lane * 4;
        *reinterpret_cast<uint2*>(dst) = k_rot;
        *reinterpret_cast<uint2*>(dst + v_offset) = vraw;
      }
    }
  }
}

}  // namespace

void dflash_kv_post(
    torch::Tensor all_kv_flat, torch::Tensor k_norm_weights, torch::Tensor cos_sin_cache,
    torch::Tensor positions, c10::optional<torch::Tensor> slot_mapping,
    const std::vector<int64_t>& kv_ptrs, int64_t block_size,
    int64_t stride_block, int64_t stride_kv_head, int64_t stride_slot, double eps) {
  const int num_ctx = static_cast<int>(all_kv_flat.size(0));
  const int num_layers = static_cast<int>(k_norm_weights.size(0));
  TORCH_CHECK(num_ctx >= 0 && num_layers == 5, "DFlash kv_post requires 5 layers");
  if (num_ctx == 0) return;
  const int num_kv_heads = static_cast<int>(all_kv_flat.size(1) / (num_layers * 2 * 128));
  TORCH_CHECK(num_kv_heads == 8 && all_kv_flat.is_contiguous() && k_norm_weights.is_contiguous() &&
                  cos_sin_cache.is_contiguous(),
              "bad kv_post shapes");
  DFlashKVPtrs ptrs = {};
  const int64_t* slot_ptr = nullptr;
  if (slot_mapping.has_value() && slot_mapping->defined()) {
    TORCH_CHECK(slot_mapping->is_contiguous() && slot_mapping->scalar_type() == torch::kInt64 &&
                    static_cast<int>(kv_ptrs.size()) == 5,
                "bad slot_mapping");
    slot_ptr = slot_mapping->data_ptr<int64_t>();
    ptrs.p0 = reinterpret_cast<__nv_bfloat16*>(kv_ptrs[0]);
    ptrs.p1 = reinterpret_cast<__nv_bfloat16*>(kv_ptrs[1]);
    ptrs.p2 = reinterpret_cast<__nv_bfloat16*>(kv_ptrs[2]);
    ptrs.p3 = reinterpret_cast<__nv_bfloat16*>(kv_ptrs[3]);
    ptrs.p4 = reinterpret_cast<__nv_bfloat16*>(kv_ptrs[4]);
  }
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  dim3 grid(num_ctx, num_layers);
  const int cache_rows = static_cast<int>(cos_sin_cache.size(0));
  if (cos_sin_cache.scalar_type() == torch::kFloat) {
    dflash_kv_post_kernel<true><<<grid, 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(all_kv_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_norm_weights.data_ptr()),
        cos_sin_cache.data_ptr(), positions.data_ptr<int64_t>(), slot_ptr, ptrs,
        num_ctx, num_layers, num_kv_heads, cache_rows, static_cast<int>(block_size),
        stride_block, stride_kv_head, stride_slot, static_cast<float>(eps));
  } else {
    dflash_kv_post_kernel<false><<<grid, 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(all_kv_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_norm_weights.data_ptr()),
        cos_sin_cache.data_ptr(), positions.data_ptr<int64_t>(), slot_ptr, ptrs,
        num_ctx, num_layers, num_kv_heads, cache_rows, static_cast<int>(block_size),
        stride_block, stride_kv_head, stride_slot, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dflash_qkv_norm_rope_cache(
    torch::Tensor qkv,
    torch::Tensor q_norm_weight,
    torch::Tensor k_norm_weight,
    torch::Tensor cos_sin_cache,
    torch::Tensor positions,
    torch::Tensor q_out,
    c10::optional<torch::Tensor> k_out,
    c10::optional<torch::Tensor> slot_mapping,
    int64_t kv_ptr,
    int64_t block_size,
    int64_t stride_block,
    int64_t stride_kv_head,
    int64_t stride_slot,
    int64_t v_offset,
    double eps) {
  const int M = static_cast<int>(qkv.size(0));
  if (M == 0) return;
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == torch::kBFloat16 && qkv.dim() == 2 &&
                  qkv.size(1) == 6144 && qkv.is_contiguous(),
              "dflash_qkv_norm_rope_cache bad qkv");
  TORCH_CHECK(q_norm_weight.numel() == 128 && k_norm_weight.numel() == 128 &&
                  q_norm_weight.is_contiguous() && k_norm_weight.is_contiguous(),
              "dflash_qkv_norm_rope_cache bad norm weights");
  TORCH_CHECK(q_out.is_cuda() && q_out.scalar_type() == torch::kBFloat16 && q_out.is_contiguous() &&
                  q_out.numel() == M * 4096,
              "dflash_qkv_norm_rope_cache bad q_out");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 &&
                  positions.is_contiguous() && positions.numel() >= M,
              "dflash_qkv_norm_rope_cache bad positions");

  __nv_bfloat16* k_out_ptr = nullptr;
  if (k_out.has_value() && k_out->defined()) {
    TORCH_CHECK(k_out->is_cuda() && k_out->scalar_type() == torch::kBFloat16 &&
                    k_out->is_contiguous() && k_out->numel() == M * 1024,
                "dflash_qkv_norm_rope_cache bad k_out");
    k_out_ptr = reinterpret_cast<__nv_bfloat16*>(k_out->data_ptr());
  }

  const int64_t* slot_ptr = nullptr;
  __nv_bfloat16* kv_cache_ptr = nullptr;
  if (slot_mapping.has_value() && slot_mapping->defined() && kv_ptr != 0) {
    TORCH_CHECK(slot_mapping->is_cuda() && slot_mapping->scalar_type() == torch::kInt64 &&
                    slot_mapping->is_contiguous() && slot_mapping->numel() >= M,
                "dflash_qkv_norm_rope_cache bad slot_mapping");
    slot_ptr = slot_mapping->data_ptr<int64_t>();
    kv_cache_ptr = reinterpret_cast<__nv_bfloat16*>(kv_ptr);
  }

  auto st = c10::cuda::getCurrentCUDAStream().stream();
  dim3 grid(10, M);
  const int cache_rows = static_cast<int>(cos_sin_cache.size(0));
  if (cos_sin_cache.scalar_type() == torch::kFloat) {
    dflash_qkv_norm_rope_cache_kernel<true><<<grid, 128, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_norm_weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_norm_weight.data_ptr()),
        cos_sin_cache.data_ptr(), positions.data_ptr<int64_t>(), slot_ptr,
        reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr()), k_out_ptr, kv_cache_ptr,
        M, cache_rows, static_cast<int>(block_size),
        stride_block, stride_kv_head, stride_slot, v_offset, static_cast<float>(eps));
  } else {
    dflash_qkv_norm_rope_cache_kernel<false><<<grid, 128, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_norm_weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_norm_weight.data_ptr()),
        cos_sin_cache.data_ptr(), positions.data_ptr<int64_t>(), slot_ptr,
        reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr()), k_out_ptr, kv_cache_ptr,
        M, cache_rows, static_cast<int>(block_size),
        stride_block, stride_kv_head, stride_slot, v_offset, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
