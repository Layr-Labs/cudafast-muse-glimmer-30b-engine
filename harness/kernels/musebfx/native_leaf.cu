// SPDX-License-Identifier: Apache-2.0
// native_leaf: Zero-overhead fused GPU bookkeeping kernels for native 1-leaf DFlash verify.
//
// Pairs with the 5-line KV-index remap inside gqa16_fa2_block128_pipe_kernel<true>:
//   - Row 6 carries `alt` (drafter's 2nd-best token for position p+1) when prev_ns <= low_thresh.
//   - positions[6] is set to positions[1] (p+1) so qk_norm_rope_kvcache applies RoPE(p+1)
//     and writes alt's K/V into slot p+6 (never read by causal rows 0..5).
//   - gqa16_fa2_block128_pipe_kernel remaps tok==(p+1) -> (p+6) with seqlen_k=(p+2),
//     producing 0-ULP bit-identical attention output to serial/row-1 evaluation.
//   - On a leaf hit (num_sampled == 1 && sampled[0] == alt): commits [alt, u], copies
//     hidden_states[6] -> [1] (+ aux[0..4][6] -> [1]), and copies slot(p+6) -> slot(p+1)
//     across all 52 target layers in a single 52-CTA x 64-thread launch (~4 us).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__global__ void native_leaf_setup_kernel(
    int32_t* __restrict__ leaf_flag,
    const int32_t* __restrict__ prev_ns,
    const int64_t* __restrict__ alt_ids,
    int32_t* __restrict__ input_ids,
    int64_t* __restrict__ positions,
    int64_t* __restrict__ saved_alt_tok,
    int low_thresh,
    int selftest_mode) {
  const int64_t a0 = alt_ids[0];
  const int64_t a1 = alt_ids[1];
  const int64_t d1 = static_cast<int64_t>(input_ids[1]);
  // Activate iff prior verify committed <= low_thresh AND alt_ids came from the same draft
  // step that produced d1 (a0 == d1) with a valid distinct second choice a1.
  const bool valid_alt = selftest_mode ? true : (a1 >= 0 && a1 != d1 && a0 == d1);
  const bool mode = (prev_ns[0] <= low_thresh) && valid_alt;
  leaf_flag[0] = mode ? 1 : 0;
  if (mode) {
    const int64_t leaf_tok = selftest_mode ? d1 : a1;
    input_ids[6] = static_cast<int32_t>(leaf_tok);
    positions[6] = positions[1];
    saved_alt_tok[0] = leaf_tok;
  } else {
    saved_alt_tok[0] = -1;
  }
}

__global__ void native_leaf_commit_kernel(
    int32_t* __restrict__ leaf_flag,
    int32_t* __restrict__ prev_ns,
    int32_t* __restrict__ hit_flag,
    const int64_t* __restrict__ saved_alt_tok,
    const int64_t* __restrict__ leaf_u_tok,
    int64_t* __restrict__ sampled_tokens,
    int32_t* __restrict__ num_sampled,
    int32_t* __restrict__ num_rejected,
    int64_t* __restrict__ stats,
    int no_hit_mode) {
  const int mode = leaf_flag[0];
  leaf_flag[0] = 0;  // Clear immediately so subsequent non-leaf or M!=7 steps see 0.
  int32_t ns = num_sampled[0];
  int32_t nr = num_rejected[0];
  int32_t is_hit = 0;
  int32_t is_clamp = 0;

  if (mode != 0) {
    if (ns == 7) {
      // Row 6 was alt (not d6), so 6 "accepted" drafts clamp to 5 accepted + 1 bonus = 6.
      ns = 6;
      nr += 1;
      is_clamp = 1;
    } else if (ns == 1 && no_hit_mode == 0) {
      const int64_t t0_tok = sampled_tokens[0];
      const int64_t alt_tok = saved_alt_tok[0];
      if (alt_tok >= 0 && t0_tok == alt_tok) {
        sampled_tokens[1] = leaf_u_tok[0];
        ns = 2;
        nr -= 1;
        is_hit = 1;
      }
    }
    num_sampled[0] = ns;
    num_rejected[0] = nr;
  }

  hit_flag[0] = is_hit;
  prev_ns[0] = ns;

  if (stats != nullptr) {
    stats[0] += 1;
    if (mode != 0) stats[1] += 1;
    if (is_hit != 0) stats[2] += 1;
    if (is_clamp != 0) stats[3] += 1;
  }
}

// Copies row 6 -> row 1 across up to 8 target hidden/aux tensors in one launch.
// Grid: (num_tensors), Block: 256 threads. Early-exits in ~1 us when hit_flag[0] == 0.
__global__ void native_leaf_fix_hidden_kernel(
    const int32_t* __restrict__ hit_flag,
    const uint64_t* __restrict__ tensor_ptrs,
    int num_tensors,
    int64_t row_stride_u4,
    int64_t copy_u4) {
  if (hit_flag[0] == 0) return;
  const int b = blockIdx.x;
  if (b >= num_tensors) return;
  uint4* base = reinterpret_cast<uint4*>(static_cast<uintptr_t>(tensor_ptrs[b]));
  uint4* dst = base + row_stride_u4;          // row 1
  const uint4* src = base + 6 * row_stride_u4;  // row 6
  for (int64_t i = threadIdx.x; i < copy_u4; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// Copies KV cache slot(p+6) -> slot(p+1) for both KV heads (K+V = 256 bf16/head = 1024 B/layer)
// across all target GQA-16 layers in a single launch.
// Grid: (num_layers), Block: 64 threads (1 uint4 per thread -> 2 heads x 32 uint4 = 1024 B).
__global__ void native_leaf_fix_kv_all_layers_kernel(
    const int32_t* __restrict__ hit_flag,
    const int64_t* __restrict__ slot_mappings,
    int64_t slot_stride,
    const uint64_t* __restrict__ kv_ptrs,
    const int64_t* __restrict__ stride_block,
    const int64_t* __restrict__ stride_head,
    const int64_t* __restrict__ stride_slot,
    const int32_t* __restrict__ block_size,
    const int32_t* __restrict__ layer_gid,
    int num_layers) {
  if (hit_flag[0] == 0) return;
  const int L = blockIdx.x;
  if (L >= num_layers) return;
  const int gi = layer_gid[L];
  const int64_t s_dst = slot_mappings[static_cast<int64_t>(gi) * slot_stride + 1];
  const int64_t s_src = slot_mappings[static_cast<int64_t>(gi) * slot_stride + 6];
  if (s_dst < 0 || s_src < 0) return;

  const int64_t bs = static_cast<int64_t>(block_size[L]);
  const int64_t blk_dst = s_dst / bs;
  const int64_t off_dst = s_dst - blk_dst * bs;
  const int64_t blk_src = s_src / bs;
  const int64_t off_src = s_src - blk_src * bs;

  const int t = threadIdx.x;
  const int h = t >> 5;     // 0 or 1 (2 KV heads)
  const int vec = t & 31;   // 0..31 (32 * 16B = 512B = 256 bf16 = 128 K + 128 V)

  __nv_bfloat16* kv = reinterpret_cast<__nv_bfloat16*>(static_cast<uintptr_t>(kv_ptrs[L]));
  const int64_t sB = stride_block[L];
  const int64_t sH = stride_head[L];
  const int64_t sS = stride_slot[L];

  uint4* dst = reinterpret_cast<uint4*>(kv + blk_dst * sB + h * sH + off_dst * sS);
  const uint4* src = reinterpret_cast<const uint4*>(kv + blk_src * sB + h * sH + off_src * sS);
  dst[vec] = src[vec];
}

}  // namespace

void native_leaf_setup(
    torch::Tensor leaf_flag,
    torch::Tensor prev_ns,
    torch::Tensor alt_ids,
    torch::Tensor input_ids,
    torch::Tensor positions,
    torch::Tensor saved_alt_tok,
    int64_t low_thresh,
    bool selftest_mode) {
  TORCH_CHECK(leaf_flag.is_cuda() && leaf_flag.scalar_type() == torch::kInt32 && leaf_flag.numel() >= 1);
  TORCH_CHECK(prev_ns.is_cuda() && prev_ns.scalar_type() == torch::kInt32 && prev_ns.numel() >= 1);
  TORCH_CHECK(alt_ids.is_cuda() && alt_ids.scalar_type() == torch::kInt64 && alt_ids.numel() >= 2);
  TORCH_CHECK(input_ids.is_cuda() && input_ids.scalar_type() == torch::kInt32 && input_ids.numel() >= 7);
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 && positions.numel() >= 7);
  TORCH_CHECK(saved_alt_tok.is_cuda() && saved_alt_tok.scalar_type() == torch::kInt64 && saved_alt_tok.numel() >= 1);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  native_leaf_setup_kernel<<<1, 1, 0, st>>>(
      leaf_flag.data_ptr<int32_t>(),
      prev_ns.data_ptr<int32_t>(),
      alt_ids.data_ptr<int64_t>(),
      input_ids.data_ptr<int32_t>(),
      positions.data_ptr<int64_t>(),
      saved_alt_tok.data_ptr<int64_t>(),
      static_cast<int>(low_thresh),
      selftest_mode ? 1 : 0);
}

void native_leaf_commit(
    torch::Tensor leaf_flag,
    torch::Tensor prev_ns,
    torch::Tensor hit_flag,
    torch::Tensor saved_alt_tok,
    torch::Tensor leaf_u_tok,
    torch::Tensor sampled_tokens,
    torch::Tensor num_sampled,
    torch::Tensor num_rejected,
    c10::optional<torch::Tensor> stats,
    bool no_hit_mode) {
  TORCH_CHECK(leaf_flag.is_cuda() && leaf_flag.scalar_type() == torch::kInt32);
  TORCH_CHECK(prev_ns.is_cuda() && prev_ns.scalar_type() == torch::kInt32);
  TORCH_CHECK(hit_flag.is_cuda() && hit_flag.scalar_type() == torch::kInt32);
  TORCH_CHECK(saved_alt_tok.is_cuda() && saved_alt_tok.scalar_type() == torch::kInt64);
  TORCH_CHECK(leaf_u_tok.is_cuda() && leaf_u_tok.scalar_type() == torch::kInt64);
  TORCH_CHECK(sampled_tokens.is_cuda() && sampled_tokens.scalar_type() == torch::kInt64 && sampled_tokens.numel() >= 2);
  TORCH_CHECK(num_sampled.is_cuda() && num_sampled.scalar_type() == torch::kInt32);
  TORCH_CHECK(num_rejected.is_cuda() && num_rejected.scalar_type() == torch::kInt32);
  int64_t* stats_ptr = nullptr;
  if (stats.has_value() && stats.value().defined() && stats.value().numel() >= 4) {
    TORCH_CHECK(stats.value().is_cuda() && stats.value().scalar_type() == torch::kInt64);
    stats_ptr = stats.value().data_ptr<int64_t>();
  }
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  native_leaf_commit_kernel<<<1, 1, 0, st>>>(
      leaf_flag.data_ptr<int32_t>(),
      prev_ns.data_ptr<int32_t>(),
      hit_flag.data_ptr<int32_t>(),
      saved_alt_tok.data_ptr<int64_t>(),
      leaf_u_tok.data_ptr<int64_t>(),
      sampled_tokens.data_ptr<int64_t>(),
      num_sampled.data_ptr<int32_t>(),
      num_rejected.data_ptr<int32_t>(),
      stats_ptr,
      no_hit_mode ? 1 : 0);
}

void native_leaf_fix_hidden(
    torch::Tensor hit_flag,
    torch::Tensor tensor_ptrs,
    int64_t row_stride_bytes,
    int64_t copy_bytes) {
  TORCH_CHECK(hit_flag.is_cuda() && hit_flag.scalar_type() == torch::kInt32);
  TORCH_CHECK(tensor_ptrs.is_cuda() && tensor_ptrs.scalar_type() == torch::kInt64);
  const int num_tensors = static_cast<int>(tensor_ptrs.numel());
  if (num_tensors <= 0 || copy_bytes <= 0) return;
  TORCH_CHECK((row_stride_bytes & 15) == 0 && (copy_bytes & 15) == 0,
              "native_leaf_fix_hidden requires 16-byte aligned row stride and copy size");
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  native_leaf_fix_hidden_kernel<<<num_tensors, 256, 0, st>>>(
      hit_flag.data_ptr<int32_t>(),
      reinterpret_cast<const uint64_t*>(tensor_ptrs.data_ptr<int64_t>()),
      num_tensors,
      row_stride_bytes >> 4,
      copy_bytes >> 4);
}

void native_leaf_fix_kv(
    torch::Tensor hit_flag,
    torch::Tensor slot_mappings,
    torch::Tensor kv_ptrs,
    torch::Tensor stride_block,
    torch::Tensor stride_head,
    torch::Tensor stride_slot,
    torch::Tensor block_size,
    torch::Tensor layer_gid) {
  TORCH_CHECK(hit_flag.is_cuda() && hit_flag.scalar_type() == torch::kInt32);
  TORCH_CHECK(slot_mappings.is_cuda() && slot_mappings.scalar_type() == torch::kInt64 &&
              slot_mappings.dim() == 2 && slot_mappings.size(1) >= 7 && slot_mappings.stride(1) == 1);
  const int num_layers = static_cast<int>(kv_ptrs.numel());
  if (num_layers <= 0) return;
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  native_leaf_fix_kv_all_layers_kernel<<<num_layers, 64, 0, st>>>(
      hit_flag.data_ptr<int32_t>(),
      slot_mappings.data_ptr<int64_t>(),
      slot_mappings.stride(0),
      reinterpret_cast<const uint64_t*>(kv_ptrs.data_ptr<int64_t>()),
      stride_block.data_ptr<int64_t>(),
      stride_head.data_ptr<int64_t>(),
      stride_slot.data_ptr<int64_t>(),
      block_size.data_ptr<int32_t>(),
      layer_gid.data_ptr<int32_t>(),
      num_layers);
}
