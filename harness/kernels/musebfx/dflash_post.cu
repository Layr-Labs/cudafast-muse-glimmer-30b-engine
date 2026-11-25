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

namespace {

template <typename LastSampT, typename NextPrefT, typename BtT, typename TailIdT>
__global__ void __launch_bounds__(256) dflash_prepare_inputs_kernel(
    int32_t* __restrict__ out_input_ids,
    int64_t* __restrict__ out_query_positions,
    int32_t* __restrict__ out_query_start_loc,
    int32_t* __restrict__ out_seq_lens,
    int64_t* __restrict__ out_query_slot_mapping,
    int64_t* __restrict__ out_context_positions,
    int64_t* __restrict__ out_context_slot_mapping,
    int64_t* __restrict__ out_sample_indices,
    int64_t* __restrict__ out_sample_pos,
    int32_t* __restrict__ out_sample_idx_mapping,
    float* __restrict__ out_temperature,
    int64_t* __restrict__ out_seeds,
    const int64_t* __restrict__ target_positions,
    const int32_t* __restrict__ target_query_start_loc,
    const int32_t* __restrict__ idx_mapping,
    const LastSampT* __restrict__ last_sampled,
    const NextPrefT* __restrict__ next_prefill_tokens,
    const int32_t* __restrict__ num_sampled,
    const int32_t* __restrict__ num_rejected,
    const float* __restrict__ in_temperature,
    const int64_t* __restrict__ in_seeds,
    const BtT* __restrict__ block_table,
    int64_t block_table_stride,
    int32_t parallel_drafting_token_id,
    int64_t block_size,
    int num_query_per_req,
    int num_speculative_steps,
    int max_num_reqs,
    int max_num_tokens,
    int64_t max_model_len,
    int64_t pad_slot_id,
    // Optional fused prompt-tail OOV update (enabled when in_static_vocab != nullptr)
    const TailIdT* __restrict__ target_input_ids,
    int tail_num_tokens,
    const uint8_t* __restrict__ in_static_vocab,
    int32_t* __restrict__ tail_ids,
    int32_t* __restrict__ tail_count,
    int32_t* __restrict__ tail_seen,
    int vocab_size,
    int max_tail) {
  __shared__ int32_t s_req_state_idx;
  __shared__ int32_t s_ctx_start;
  __shared__ int32_t s_num_ctx;
  __shared__ int32_t s_bonus_token;
  __shared__ int64_t s_last_valid_pos;
  __shared__ int s_tail_reset;
  __shared__ int s_tail_old_cnt;

  const int tid = threadIdx.x;
  if (tid == 0) {
    const int32_t req_state_idx = idx_mapping[0];
    const int32_t ctx_start = target_query_start_loc[0];
    const int32_t ctx_end = target_query_start_loc[1];
    const int32_t num_ctx = ctx_end - ctx_start;
    const int32_t nr = num_rejected[0];
    const int32_t valid_ctx_end = ctx_end - nr;
    const int32_t ns = num_sampled[0];
    const int32_t bonus_token = (ns > 0)
        ? static_cast<int32_t>(last_sampled[req_state_idx])
        : static_cast<int32_t>(next_prefill_tokens[req_state_idx]);
    const int64_t last_valid_pos = target_positions[valid_ctx_end - 1];

    s_req_state_idx = req_state_idx;
    s_ctx_start = ctx_start;
    s_num_ctx = num_ctx;
    s_bonus_token = bonus_token;
    s_last_valid_pos = last_valid_pos;

    out_query_start_loc[0] = 0;
    out_seq_lens[0] = static_cast<int32_t>(last_valid_pos + 1 + num_query_per_req);
    out_temperature[req_state_idx] = in_temperature[req_state_idx];
    out_seeds[req_state_idx] = in_seeds[req_state_idx];

    if (in_static_vocab != nullptr && tail_num_tokens > 0) {
      s_tail_reset = (target_positions[0] == 0) ? 1 : 0;
      int oc = tail_count[0];
      if (oc < 0) oc = 0;
      if (oc > max_tail) oc = max_tail;
      s_tail_old_cnt = oc;
    } else {
      s_tail_reset = 0;
      s_tail_old_cnt = 0;
    }
  }
  __syncthreads();

  const int32_t req_state_idx = s_req_state_idx;
  const int32_t ctx_start = s_ctx_start;
  const int32_t num_ctx = s_num_ctx;
  const int32_t bonus_token = s_bonus_token;
  const int64_t last_valid_pos = s_last_valid_pos;

  // 1. Context positions & slots
  for (int j = tid; j < num_ctx; j += 256) {
    const int64_t ctx_pos = target_positions[ctx_start + j];
    int64_t ctx_blk = (block_size == 16) ? (ctx_pos >> 4) : (ctx_pos / block_size);
    if (ctx_blk > block_table_stride - 1) ctx_blk = block_table_stride - 1;
    const int64_t ctx_bid = static_cast<int64_t>(block_table[ctx_blk]);
    const int64_t ctx_off = (block_size == 16) ? (ctx_pos & 15) : (ctx_pos % block_size);
    out_context_positions[ctx_start + j] = ctx_pos;
    out_context_slot_mapping[ctx_start + j] = ctx_bid * block_size + ctx_off;
  }

  // 2. Query positions, input_ids, slots, and sample metadata
  for (int q = tid; q < num_query_per_req; q += 256) {
    const int64_t query_pos = last_valid_pos + 1 + q;
    const int32_t input_id = (q == 0) ? bonus_token : parallel_drafting_token_id;
    int64_t q_blk = (block_size == 16) ? (query_pos >> 4) : (query_pos / block_size);
    if (q_blk > block_table_stride - 1) q_blk = block_table_stride - 1;
    const int64_t q_bid = static_cast<int64_t>(block_table[q_blk]);
    const int64_t q_off = (block_size == 16) ? (query_pos & 15) : (query_pos % block_size);
    out_input_ids[q] = input_id;
    out_query_positions[q] = (query_pos < max_model_len - 1) ? query_pos : (max_model_len - 1);
    out_query_slot_mapping[q] = q_bid * block_size + q_off;
    if (q >= 1 && q < 1 + num_speculative_steps) {
      const int s = q - 1;
      out_sample_indices[s] = q;
      out_sample_pos[s] = query_pos;
      out_sample_idx_mapping[s] = req_state_idx;
    }
  }

  // 3. CUDA-graph multi-req padding (0 iterations when max_num_reqs == 1)
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

  // 4. Optional fused prompt-tail OOV update
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
    for (int i = tid; i < tail_num_tokens; i += 256) {
      const int64_t raw_tok = static_cast<int64_t>(target_input_ids[i]);
      if (raw_tok >= 0 && raw_tok < vocab_size) {
        const int tok = static_cast<int>(raw_tok);
        if (in_static_vocab[tok] == 0) {
          if (atomicExch(&tail_seen[tok], 1) == 0) {
            const int slot = atomicAdd(tail_count, 1);
            if (slot < max_tail) {
              tail_ids[slot] = tok;
            } else {
              tail_seen[tok] = 0;
            }
          }
        }
      }
    }
    __syncthreads();
    if (tid == 0 && tail_count[0] > max_tail) {
      tail_count[0] = max_tail;
    }
  }
}

}  // namespace

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
    c10::optional<torch::Tensor> tail_seen) {
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

  if (target_input_ids.has_value() && target_input_ids->defined() &&
      in_static_vocab.has_value() && in_static_vocab->defined() &&
      tail_ids.has_value() && tail_ids->defined() &&
      tail_count.has_value() && tail_count->defined() &&
      tail_seen.has_value() && tail_seen->defined() && tail_num_tokens > 0) {
    TORCH_CHECK(target_input_ids->is_contiguous(), "dflash_prepare_inputs non-contiguous target_input_ids");
    tail_i64 = (target_input_ids->scalar_type() == torch::kInt64);
    TORCH_CHECK(tail_i64 || target_input_ids->scalar_type() == torch::kInt32,
                "dflash_prepare_inputs bad target_input_ids dtype");
    target_ids_raw = target_input_ids->data_ptr();
    static_vocab_ptr = in_static_vocab->data_ptr<uint8_t>();
    tail_ids_ptr = tail_ids->data_ptr<int32_t>();
    tail_count_ptr = tail_count->data_ptr<int32_t>();
    tail_seen_ptr = tail_seen->data_ptr<int32_t>();
    vocab_size = static_cast<int>(in_static_vocab->numel());
    max_tail = static_cast<int>(tail_ids->numel());
    eff_tail_tokens = static_cast<int>(tail_num_tokens);
  }

  const bool ls_i64 = (last_sampled.scalar_type() == torch::kInt64);
  TORCH_CHECK(ls_i64 || last_sampled.scalar_type() == torch::kInt32, "bad last_sampled dtype");
  TORCH_CHECK(next_prefill_tokens.scalar_type() == torch::kInt32, "bad next_prefill_tokens dtype");

  auto st = c10::cuda::getCurrentCUDAStream().stream();
#define LAUNCH_PREP(LST, TIDT) \
  dflash_prepare_inputs_kernel<LST, int32_t, int32_t, TIDT><<<1, 256, 0, st>>>( \
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
      max_tail)

  if (ls_i64) {
    if (tail_i64) { LAUNCH_PREP(int64_t, int64_t); }
    else { LAUNCH_PREP(int64_t, int32_t); }
  } else {
    if (tail_i64) { LAUNCH_PREP(int32_t, int64_t); }
    else { LAUNCH_PREP(int32_t, int32_t); }
  }
#undef LAUNCH_PREP
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
