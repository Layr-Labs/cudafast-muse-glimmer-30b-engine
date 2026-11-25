// SPDX-License-Identifier: Apache-2.0
// DFlash drafter propose() & context-KV fused post-processing kernels (100% original).
// 1. dflash_kv_post: fuses 9 eager context-KV launches (permute+contiguous, 5-layer grouped
//    K RMSNorm, positions repeat, NEOX RoPE, and 5 NHD/HND paged BF16 KV-cache scatter writes)
//    into 1 kernel launch (0-ULP bitwise exact vs vLLM eager path).
// 2. dflash_argmax_bf16: 48-CTA 2-stage lowest-index-tie [M<=8, V] BF16 greedy argmax directly
//    into draft_tokens (int64).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>
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

  // Offset inside all_kv_flat [num_ctx, L, 2, nkv, 128]
  const size_t pair_base = (((size_t)ctx_idx * num_layers + layer_idx) * 2) * ((size_t)num_kv_heads * 128);
  const __nv_bfloat16* k_src = all_kv_flat + pair_base + (size_t)kv_head * 128 + lane * 4;
  const __nv_bfloat16* v_src = k_src + (size_t)num_kv_heads * 128;

  const uint2 kraw = __ldg(reinterpret_cast<const uint2*>(k_src));
  const uint2 vraw = __ldg(reinterpret_cast<const uint2*>(v_src));
  const uint2 wraw = __ldg(reinterpret_cast<const uint2*>(k_norm_weights + (size_t)layer_idx * 128 + lane * 4));

  const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kraw);
  const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wraw);
  const float x0 = __bfloat162float(kb[0]), x1 = __bfloat162float(kb[1]);
  const float x2 = __bfloat162float(kb[2]), x3 = __bfloat162float(kb[3]);

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

  int64_t pos = __ldg(positions + ctx_idx);
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

  if (slot_mapping == nullptr) return;
  const int64_t slot = __ldg(slot_mapping + ctx_idx);
  if (slot < 0) return;

  uint2 k_out;
  __nv_bfloat16* kob = reinterpret_cast<__nv_bfloat16*>(&k_out);
  kob[0] = __float2bfloat16_rn(o0);
  kob[1] = __float2bfloat16_rn(o1);
  kob[2] = __float2bfloat16_rn(o2);
  kob[3] = __float2bfloat16_rn(o3);

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

__global__ void __launch_bounds__(256) dflash_argmax_bf16_stage1(
    const __nv_bfloat16* __restrict__ logits, float* __restrict__ part_val, int32_t* __restrict__ part_idx,
    int M, int V) {
  __shared__ float s_val[8];
  __shared__ int32_t s_idx[8];
  const int chunk = blockIdx.x;
  const int row = blockIdx.y;
  if (row >= M) return;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const __nv_bfloat16* rptr = logits + (size_t)row * V;
  const int V8 = V >> 3;
  const int c_start = (V8 * chunk) / gridDim.x;
  const int c_end = (V8 * (chunk + 1)) / gridDim.x;

  float best_v = -INFINITY;
  int32_t best_i = 0x7fffffff;

  const uint4* r4 = reinterpret_cast<const uint4*>(rptr);
  for (int i8 = c_start + tid; i8 < c_end; i8 += blockDim.x) {
    const uint4 u = __ldg(r4 + i8);
    const __nv_bfloat16* vals = reinterpret_cast<const __nv_bfloat16*>(&u);
    const int base = i8 << 3;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float v = __bfloat162float(vals[j]);
      const int idx = base + j;
      if (v > best_v || (v == best_v && idx < best_i)) {
        best_v = v;
        best_i = idx;
      }
    }
  }
  if (chunk == gridDim.x - 1) {
    for (int idx = (V8 << 3) + tid; idx < V; idx += blockDim.x) {
      const float v = __bfloat162float(__ldg(rptr + idx));
      if (v > best_v || (v == best_v && idx < best_i)) {
        best_v = v;
        best_i = idx;
      }
    }
  }

#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best_v, o);
    const int32_t oi = __shfl_xor_sync(0xffffffffu, best_i, o);
    if (ov > best_v || (ov == best_v && oi < best_i)) {
      best_v = ov;
      best_i = oi;
    }
  }
  if (lane == 0) {
    s_val[warp] = best_v;
    s_idx[warp] = best_i;
  }
  __syncthreads();
  if (tid < 8) {
    best_v = s_val[tid];
    best_i = s_idx[tid];
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffu, best_v, o);
      const int32_t oi = __shfl_xor_sync(0xffu, best_i, o);
      if (ov > best_v || (ov == best_v && oi < best_i)) {
        best_v = ov;
        best_i = oi;
      }
    }
    if (tid == 0) {
      const int out_off = row * gridDim.x + chunk;
      part_val[out_off] = best_v;
      part_idx[out_off] = best_i;
    }
  }
}

__global__ void __launch_bounds__(256) dflash_argmax_bf16_stage2(
    const float* __restrict__ part_val, const int32_t* __restrict__ part_idx, int64_t* __restrict__ out_tok,
    int M, int chunks) {
  const int row = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (row >= M) return;
  float best_v = -INFINITY;
  int32_t best_i = 0x7fffffff;
  if (lane < chunks) {
    best_v = part_val[row * chunks + lane];
    best_i = part_idx[row * chunks + lane];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best_v, o);
    const int32_t oi = __shfl_xor_sync(0xffffffffu, best_i, o);
    if (ov > best_v || (ov == best_v && oi < best_i)) {
      best_v = ov;
      best_i = oi;
    }
  }
  if (lane == 0) {
    out_tok[row] = (int64_t)best_i;
  }
}

}  // namespace

void dflash_kv_post(
    torch::Tensor all_kv_flat, torch::Tensor k_norm_weights, torch::Tensor cos_sin_cache,
    torch::Tensor positions, c10::optional<torch::Tensor> slot_mapping,
    const std::vector<int64_t>& kv_ptrs, int64_t block_size,
    int64_t stride_block, int64_t stride_kv_head, int64_t stride_slot, double eps) {
  const int num_ctx = (int)all_kv_flat.size(0);
  const int num_layers = (int)k_norm_weights.size(0);
  TORCH_CHECK(num_ctx >= 0 && num_layers == 5, "DFlash kv_post requires 5 layers");
  if (num_ctx == 0) return;
  const int num_kv_heads = (int)(all_kv_flat.size(1) / (num_layers * 2 * 128));
  TORCH_CHECK(num_kv_heads == 8 && all_kv_flat.is_contiguous() && k_norm_weights.is_contiguous() &&
                  cos_sin_cache.is_contiguous(),
              "bad kv_post shapes");
  DFlashKVPtrs ptrs = {};
  const int64_t* slot_ptr = nullptr;
  if (slot_mapping.has_value() && slot_mapping->defined()) {
    TORCH_CHECK(slot_mapping->is_contiguous() && slot_mapping->scalar_type() == torch::kInt64 &&
                    (int)kv_ptrs.size() == 5,
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
  const int cache_rows = (int)cos_sin_cache.size(0);
  if (cos_sin_cache.scalar_type() == torch::kFloat) {
    dflash_kv_post_kernel<true><<<grid, 256, 0, st>>>(
        (const __nv_bfloat16*)all_kv_flat.data_ptr(), (const __nv_bfloat16*)k_norm_weights.data_ptr(),
        cos_sin_cache.data_ptr(), positions.data_ptr<int64_t>(), slot_ptr, ptrs,
        num_ctx, num_layers, num_kv_heads, cache_rows, (int)block_size,
        stride_block, stride_kv_head, stride_slot, (float)eps);
  } else {
    dflash_kv_post_kernel<false><<<grid, 256, 0, st>>>(
        (const __nv_bfloat16*)all_kv_flat.data_ptr(), (const __nv_bfloat16*)k_norm_weights.data_ptr(),
        cos_sin_cache.data_ptr(), positions.data_ptr<int64_t>(), slot_ptr, ptrs,
        num_ctx, num_layers, num_kv_heads, cache_rows, (int)block_size,
        stride_block, stride_kv_head, stride_slot, (float)eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dflash_argmax_bf16(torch::Tensor logits, torch::Tensor scratch_val, torch::Tensor scratch_idx,
                        torch::Tensor out_tok) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2 &&
                  logits.is_contiguous(),
              "bad logits");
  const int M = (int)logits.size(0), V = (int)logits.size(1);
  TORCH_CHECK((V & 7) == 0 && (reinterpret_cast<uintptr_t>(logits.data_ptr()) & 15u) == 0u,
              "dflash_argmax_bf16 requires V%8==0 and 16B alignment");
  TORCH_CHECK(M >= 1 && M <= 8 && V > 0 && out_tok.is_cuda() && out_tok.scalar_type() == torch::kInt64 &&
                  out_tok.is_contiguous() && out_tok.numel() == M,
              "bad out_tok");
  const int chunks = std::min(32, std::max(1, 48 / M));
  TORCH_CHECK(scratch_val.numel() >= M * chunks && scratch_idx.numel() >= M * chunks, "bad scratch");
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  dim3 grid(chunks, M);
  dflash_argmax_bf16_stage1<<<grid, 256, 0, st>>>(
      (const __nv_bfloat16*)logits.data_ptr(), scratch_val.data_ptr<float>(), scratch_idx.data_ptr<int32_t>(), M, V);
  dflash_argmax_bf16_stage2<<<1, M * 32, 0, st>>>(
      scratch_val.data_ptr<float>(), scratch_idx.data_ptr<int32_t>(), out_tok.data_ptr<int64_t>(), M, chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
