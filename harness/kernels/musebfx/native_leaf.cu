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

__global__ void native_leaf_prepare_m8_kernel(
    int64_t* __restrict__ positions,
    int32_t* __restrict__ seq_lens) {
  positions[7] = positions[6] + 1;
  seq_lens[0] = seq_lens[0] + 1;
}

__global__ void native_leaf_prepare_tree_kernel(
    int64_t* __restrict__ positions,
    int32_t* __restrict__ seq_lens,
    int tree_m) {
  const int64_t p6 = positions[6];
  for (int r = 7; r < tree_m; ++r) {
    positions[r] = p6 + static_cast<int64_t>(r - 6);
  }
  seq_lens[0] = seq_lens[0] + (tree_m - 7);
}

// 1-level multi-leaf tree setup (rows 7..tree_m-1), launched as 1 warp (<<<1, 32>>>, <2 us):
//   - dynamic_mode==1 (default, tree_vals!=null, tree_k>=6): provably optimal additive scorer
//     over all 30 candidates (k in 1..6, rank in 1..5 = #2..#6):
//       e_{k,r} = exp((z_{k,r} - z_{k,0}) / T), Z_k = sum_{r=0..5} e_{k,r} + c_tail * e_{k,5}
//       A_0 = 1, A_k = A_{k-1} * (gamma / Z_k), score(k, r) = A_{k-1} * (gamma * e_{k,r} / Z_k)
//     Uses (tree_m - 7) warp-max passes to place the exact top-(tree_m-7) leaves into rows 7..tree_m-1.
//   - dynamic_mode==0 (or selftest_mode!=0): static Candidate B topology:
//     M=12 (5 leaves): (d1#2, d1#3, d2#2, d2#3, d3#2) -> k={1,1,2,2,3}, rank={1,2,1,2,1}
//     M=16 (9 leaves): + (d1#4, d3#3, d4#2, d5#2)     -> k={1,3,4,5},   rank={3,2,1,1}
__global__ void native_leaf_setup_tree_kernel(
    int32_t* __restrict__ leaf_flag,
    const int64_t* __restrict__ tree_top,
    const float* __restrict__ tree_vals,
    const int32_t* __restrict__ req_acc_state,
    int32_t* __restrict__ input_ids,
    int64_t* __restrict__ positions,
    int64_t* __restrict__ saved_leaf_toks,
    int32_t* __restrict__ saved_leaf_k,
    int tree_m,
    int tree_k,
    int selftest_mode,
    int dynamic_mode,
    float inv_temp,
    float tail_c,
    float gamma) {
  const int tid = threadIdx.x;
  const bool high_acc = (req_acc_state != nullptr && req_acc_state[2] != 0);

  // Chain rows 0..6: plain causal lengths p+1 .. p+7, no KV remap.
  if (tid < 7) {
    leaf_flag[1 + 3 * tid + 0] = tree_m - 1 - tid;
    leaf_flag[1 + 3 * tid + 1] = -1;
    leaf_flag[1 + 3 * tid + 2] = -1;
  }

  const int num_leaves = tree_m - 7;
  if (dynamic_mode == 0 || selftest_mode != 0 || tree_vals == nullptr || tree_k < 6) {
    if (tid == 0) {
      constexpr int kLeafK[9] = {1, 1, 2, 2, 3, 1, 3, 4, 5};
      constexpr int kLeafRank[9] = {1, 2, 1, 2, 1, 3, 2, 1, 1};
      bool any_active = false;
      for (int li = 0; li < num_leaves; ++li) {
        const int r = 7 + li;
        const int k = kLeafK[li];
        const int rank = kLeafRank[li];
        const int64_t dk = static_cast<int64_t>(input_ids[k]);
        int64_t cand = -1;
        bool valid = false;
        if (!high_acc) {
          if (selftest_mode != 0) {
            cand = dk;
            valid = true;
          } else {
            const int64_t top1 = tree_top[(k - 1) * tree_k + 0];
            cand = tree_top[(k - 1) * tree_k + rank];
            valid = (top1 == dk && cand >= 0 && cand != dk);
          }
        }
        if (valid) {
          any_active = true;
          input_ids[r] = static_cast<int32_t>(cand);
          positions[r] = positions[k];
          saved_leaf_toks[r] = cand;
          saved_leaf_k[r] = k;
          leaf_flag[1 + 3 * r + 0] = tree_m - (k + 1);
          leaf_flag[1 + 3 * r + 1] = tree_m - k;
          leaf_flag[1 + 3 * r + 2] = tree_m - r;
        } else {
          input_ids[r] = input_ids[0];
          saved_leaf_toks[r] = -1;
          saved_leaf_k[r] = 0;
          leaf_flag[1 + 3 * r + 0] = tree_m - 1 - r;
          leaf_flag[1 + 3 * r + 1] = -1;
          leaf_flag[1 + 3 * r + 2] = -1;
        }
      }
      leaf_flag[0] = any_active ? -1 : 0;
    }
    return;
  }

  // Dynamic per-step GPU leaf allocator across 30 candidates (k in 1..6, rank in 1..5).
  __shared__ float s_Z[6];
  __shared__ float s_A[6];
  __shared__ float s_z0[6];
  __shared__ int64_t s_dk[6];
  __shared__ uint8_t s_valid_row[6];
  __shared__ int s_any_active;

  if (tid < 6) {
    const int k = tid + 1;
    const int64_t dk = static_cast<int64_t>(input_ids[k]);
    s_dk[tid] = dk;
    const int base = tid * tree_k;
    const int64_t top1 = tree_top[base + 0];
    const float z0 = tree_vals[base + 0];
    const bool row_ok = (!high_acc) && (top1 == dk) && (z0 > -1e20f);
    s_valid_row[tid] = row_ok ? 1 : 0;
    s_z0[tid] = z0;
    if (row_ok) {
      float sum_e = 1.0f;
      float last_e = 0.0f;
#pragma unroll
      for (int rk = 1; rk < 6; ++rk) {
        const int64_t tok_r = tree_top[base + rk];
        const float zr = tree_vals[base + rk];
        if (tok_r >= 0 && zr > -1e20f) {
          const float er = __expf((zr - z0) * inv_temp);
          sum_e += er;
          last_e = er;
        }
      }
      s_Z[tid] = sum_e + tail_c * last_e;
    } else {
      s_Z[tid] = 1e30f;
    }
  }
  __syncwarp();

  if (tid == 0) {
    float a = 1.0f;
    for (int j = 0; j < 6; ++j) {
      s_A[j] = a;
      if (s_valid_row[j]) {
        const float p1 = gamma / s_Z[j];
        a *= (p1 < 1.0f ? p1 : 1.0f);
      } else {
        a = 0.0f;
      }
    }
    s_any_active = 0;
  }
  __syncwarp();

  float my_score = -1.0f;
  int my_k = 0;
  int64_t my_cand = -1;
  if (tid < 30) {
    const int row_idx = tid / 5;                  // 0..5 -> depth k = row_idx + 1
    const int rank = 1 + (tid - row_idx * 5);     // 1..5 -> candidate rank #2..#6
    if (s_valid_row[row_idx] && s_A[row_idx] > 0.0f) {
      const int base = row_idx * tree_k;
      const int64_t cand = tree_top[base + rank];
      const float zr = tree_vals[base + rank];
      const int64_t dk = s_dk[row_idx];
      bool ok = (cand >= 0 && cand != dk && zr > -1e20f);
      if (ok) {
        for (int pr = 1; pr < rank; ++pr) {
          if (tree_top[base + pr] == cand) {
            ok = false;
            break;
          }
        }
      }
      if (ok) {
        const float pr = gamma * __expf((zr - s_z0[row_idx]) * inv_temp) / s_Z[row_idx];
        my_score = s_A[row_idx] * pr;
        my_k = row_idx + 1;
        my_cand = cand;
      }
    }
  }

  for (int li = 0; li < num_leaves; ++li) {
    const int r = 7 + li;
    float best_s = my_score;
    int best_lane = tid;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float os = __shfl_xor_sync(0xffffffffu, best_s, o);
      const int ol = __shfl_xor_sync(0xffffffffu, best_lane, o);
      if (os > best_s || (os == best_s && ol < best_lane)) {
        best_s = os;
        best_lane = ol;
      }
    }
    if (best_s > 0.0f) {
      if (tid == best_lane) {
        input_ids[r] = static_cast<int32_t>(my_cand);
        positions[r] = positions[my_k];
        saved_leaf_toks[r] = my_cand;
        saved_leaf_k[r] = my_k;
        leaf_flag[1 + 3 * r + 0] = tree_m - (my_k + 1);
        leaf_flag[1 + 3 * r + 1] = tree_m - my_k;
        leaf_flag[1 + 3 * r + 2] = tree_m - r;
        s_any_active = 1;
        my_score = -1.0f;
      }
    } else if (tid == 0) {
      input_ids[r] = input_ids[0];
      saved_leaf_toks[r] = -1;
      saved_leaf_k[r] = 0;
      leaf_flag[1 + 3 * r + 0] = tree_m - 1 - r;
      leaf_flag[1 + 3 * r + 1] = -1;
      leaf_flag[1 + 3 * r + 2] = -1;
    }
    __syncwarp();
  }

  if (tid == 0) {
    leaf_flag[0] = s_any_active ? -1 : 0;
  }
}

__global__ void native_leaf_commit_tree_kernel(
    int32_t* __restrict__ leaf_flag,
    int32_t* __restrict__ prev_ns,
    int32_t* __restrict__ hit_flag,
    int32_t* __restrict__ hit_src_row,
    const int64_t* __restrict__ saved_leaf_toks,
    const int32_t* __restrict__ saved_leaf_k,
    const int64_t* __restrict__ leaf_u_toks,
    int64_t* __restrict__ sampled_tokens,
    int32_t* __restrict__ num_sampled,
    int32_t* __restrict__ num_rejected,
    int64_t* __restrict__ stats,
    int stats_len,
    int32_t* __restrict__ req_acc_state,
    int tree_m,
    int no_hit_mode) {
  const int mode = leaf_flag[0];
  leaf_flag[0] = 0;  // Clear immediately so non-tree steps see 0.
  int32_t ns = num_sampled[0];
  int32_t nr = num_rejected[0];
  int32_t hit_dst = 0;
  int32_t hit_src = 0;

  // Supports hits at ALL draft depths k = 1..6 (including k=6 promoting ns=6 -> 7, nr=0).
  if (mode < 0 && ns >= 1 && ns <= 6 && no_hit_mode == 0) {
    const int64_t tok_k = sampled_tokens[ns - 1];
    for (int r = 7; r < tree_m; ++r) {
      if (saved_leaf_k[r] == ns && saved_leaf_toks[r] >= 0 && tok_k == saved_leaf_toks[r]) {
        sampled_tokens[ns] = leaf_u_toks[r - 7];
        hit_dst = ns;
        hit_src = r;
        ns += 1;
        nr -= 1;
        num_sampled[0] = ns;
        num_rejected[0] = nr;
        break;
      }
    }
  }

  hit_flag[0] = hit_dst;
  hit_src_row[0] = hit_src;
  prev_ns[0] = ns;

  if (req_acc_state != nullptr) {
    const int32_t r = req_acc_state[0] + 1;
    const int32_t c = req_acc_state[1] + ns;
    req_acc_state[0] = r;
    req_acc_state[1] = c;
    // Dynamic high-acceptance pause (un-pauses if mean drops <= 4.0); value 2 = hard latch from ceiling clamp.
    if (req_acc_state[2] != 2) {
      req_acc_state[2] = (r >= 4 && c > 4 * r) ? 1 : 0;
    }
  }

  if (stats != nullptr && stats_len >= 4) {
    stats[0] += 1;
    if (mode < 0) stats[1] += 1;
    if (hit_dst > 0) {
      stats[2] += 1;
      if (stats_len >= 10 && hit_dst >= 1 && hit_dst <= 6) {
        stats[3 + hit_dst] += 1;  // stats[4..9] = hits at depth k=1..6
      }
      if (stats_len >= 19 && hit_src >= 7 && hit_src < 16) {
        stats[10 + (hit_src - 7)] += 1;  // stats[10..18] = hits by leaf slot 0..8
      }
    }
  }
}

__global__ void native_leaf_setup_kernel(
    int32_t* __restrict__ leaf_flag,
    const int32_t* __restrict__ prev_ns,
    const int64_t* __restrict__ alt_ids,
    int alt_ids_len,
    const float* __restrict__ alt_margins,
    const int32_t* __restrict__ req_acc_state,
    int32_t* __restrict__ input_ids,
    int64_t* __restrict__ positions,
    int64_t* __restrict__ saved_alt_tok,
    int low_thresh,
    int selftest_mode,
    int m8_mode,
    int policy,
    float margin_alpha) {
  const int64_t a0 = alt_ids[0];
  const int64_t a1 = alt_ids[1];
  const int64_t d1 = static_cast<int64_t>(input_ids[1]);

  if (m8_mode != 0) {
    const bool high_acc = (req_acc_state != nullptr && req_acc_state[2] != 0);
    int k = 0;
    int64_t leaf_tok = -1;
    if (!high_acc) {
      if (selftest_mode != 0) {
        k = (selftest_mode == 2) ? 2 : 1;
        leaf_tok = static_cast<int64_t>(input_ids[k]);
      } else {
        const bool v1 = (a1 >= 0 && a1 != d1 && a0 == d1);
        const int64_t d2 = static_cast<int64_t>(input_ids[2]);
        const bool has_d2 = (alt_ids_len >= 4);
        const int64_t b0 = has_d2 ? alt_ids[2] : -1;
        const int64_t b1 = has_d2 ? alt_ids[3] : -1;
        const bool v2 = (has_d2 && b1 >= 0 && b1 != d2 && b0 == d2);
        if (policy == 1 && v2) {
          if (!v1) {
            k = 2;
            leaf_tok = b1;
          } else if (alt_margins != nullptr && (alt_margins[1] * margin_alpha) < alt_margins[0]) {
            k = 2;
            leaf_tok = b1;
          } else {
            k = 1;
            leaf_tok = a1;
          }
        } else if (v1) {
          k = 1;
          leaf_tok = a1;
        }
      }
    }
    leaf_flag[0] = k;
    if (k != 0) {
      input_ids[7] = static_cast<int32_t>(leaf_tok);
      positions[7] = positions[k];
      saved_alt_tok[0] = leaf_tok;
    } else {
      // Keep row 7 valid for embedding lookup; positions[7] remains p+7.
      input_ids[7] = input_ids[0];
      saved_alt_tok[0] = -1;
    }
    return;
  }

  // Legacy M=7 path (replaces row 6 when prev_ns <= low_thresh).
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
    int stats_len,
    int32_t* __restrict__ req_acc_state,
    int no_hit_mode,
    int m8_mode) {
  const int mode = leaf_flag[0];
  leaf_flag[0] = 0;  // Clear immediately so subsequent non-leaf or M!=7/8 steps see 0.
  int32_t ns = num_sampled[0];
  int32_t nr = num_rejected[0];
  int32_t is_hit = 0;
  int32_t is_clamp = 0;

  if (m8_mode != 0) {
    // M=8 zero-clamp: rows 0..6 carry the full chain [t0, d1..d6] -> NEVER clamp ns==7!
    if ((mode == 1 || mode == 2) && ns == mode && no_hit_mode == 0) {
      const int64_t tok_k = sampled_tokens[mode - 1];
      const int64_t alt_tok = saved_alt_tok[0];
      if (alt_tok >= 0 && tok_k == alt_tok) {
        sampled_tokens[mode] = leaf_u_tok[0];
        ns = mode + 1;
        nr -= 1;
        is_hit = mode;
        num_sampled[0] = ns;
        num_rejected[0] = nr;
      }
    }
  } else if (mode != 0) {
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

  if (req_acc_state != nullptr) {
    const int32_t r = req_acc_state[0] + 1;
    const int32_t c = req_acc_state[1] + ns;
    req_acc_state[0] = r;
    req_acc_state[1] = c;
    // Dynamic high-acceptance pause (un-pauses if mean drops <= 4.0); value 2 = hard latch from ceiling clamp.
    if (req_acc_state[2] != 2) {
      req_acc_state[2] = (r >= 4 && c > 4 * r) ? 1 : 0;
    }
  }

  if (stats != nullptr && stats_len >= 4) {
    stats[0] += 1;
    if (mode != 0) stats[1] += 1;
    if (is_hit != 0) stats[2] += 1;
    if (is_clamp != 0) stats[3] += 1;
    if (stats_len >= 6) {
      if (mode == 2) stats[4] += 1;
      if (is_hit == 2) stats[5] += 1;
    }
  }
}

// Copies row `src_row` (or `src_row_ptr[0]` if non-null) -> row `dst_row = hit_flag[0]` across up to 8 tensors.
// Grid: (num_tensors), Block: 256 threads. Early-exits in ~1 us when hit_flag[0] == 0.
__global__ void native_leaf_fix_hidden_kernel(
    const int32_t* __restrict__ hit_flag,
    const int32_t* __restrict__ src_row_ptr,
    const uint64_t* __restrict__ tensor_ptrs,
    int num_tensors,
    int64_t row_stride_u4,
    int64_t copy_u4,
    int src_row) {
  const int dst_row = hit_flag[0];
  if (dst_row <= 0) return;
  const int eff_src = (src_row_ptr != nullptr) ? src_row_ptr[0] : src_row;
  if (eff_src <= 0) return;
  const int b = blockIdx.x;
  if (b >= num_tensors) return;
  uint4* base = reinterpret_cast<uint4*>(static_cast<uintptr_t>(tensor_ptrs[b]));
  uint4* dst = base + static_cast<int64_t>(dst_row) * row_stride_u4;
  const uint4* src = base + static_cast<int64_t>(eff_src) * row_stride_u4;
  for (int64_t i = threadIdx.x; i < copy_u4; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// Copies KV cache slot(p+src_row) -> slot(p+dst_row) for both KV heads (K+V = 1024 B/layer)
// across all target GQA-16 layers in a single launch.
// Grid: (num_layers), Block: 64 threads (1 uint4 per thread -> 2 heads x 32 uint4 = 1024 B).
__global__ void native_leaf_fix_kv_all_layers_kernel(
    const int32_t* __restrict__ hit_flag,
    const int32_t* __restrict__ src_row_ptr,
    const int64_t* __restrict__ slot_mappings,
    int64_t slot_stride,
    const uint64_t* __restrict__ kv_ptrs,
    const int64_t* __restrict__ stride_block,
    const int64_t* __restrict__ stride_head,
    const int64_t* __restrict__ stride_slot,
    const int32_t* __restrict__ block_size,
    const int32_t* __restrict__ layer_gid,
    int num_layers,
    int src_row) {
  const int dst_row = hit_flag[0];
  if (dst_row <= 0) return;
  const int eff_src = (src_row_ptr != nullptr) ? src_row_ptr[0] : src_row;
  if (eff_src <= 0) return;
  const int L = blockIdx.x;
  if (L >= num_layers) return;
  const int gi = layer_gid[L];
  const int64_t s_dst = slot_mappings[static_cast<int64_t>(gi) * slot_stride + dst_row];
  const int64_t s_src = slot_mappings[static_cast<int64_t>(gi) * slot_stride + eff_src];
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

void native_leaf_prepare_m8(
    torch::Tensor positions,
    torch::Tensor seq_lens) {
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 && positions.numel() >= 8);
  TORCH_CHECK(seq_lens.is_cuda() && seq_lens.scalar_type() == torch::kInt32 && seq_lens.numel() >= 1);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  native_leaf_prepare_m8_kernel<<<1, 1, 0, st>>>(
      positions.data_ptr<int64_t>(),
      seq_lens.data_ptr<int32_t>());
}

void native_leaf_prepare_tree(
    torch::Tensor positions,
    torch::Tensor seq_lens,
    int64_t tree_m) {
  TORCH_CHECK(tree_m >= 8 && tree_m <= 16, "tree_m must be in 8..16");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 && positions.numel() >= tree_m);
  TORCH_CHECK(seq_lens.is_cuda() && seq_lens.scalar_type() == torch::kInt32 && seq_lens.numel() >= 1);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  native_leaf_prepare_tree_kernel<<<1, 1, 0, st>>>(
      positions.data_ptr<int64_t>(),
      seq_lens.data_ptr<int32_t>(),
      static_cast<int>(tree_m));
}

void native_leaf_setup_tree(
    torch::Tensor leaf_flag,
    torch::Tensor tree_top,
    torch::Tensor input_ids,
    torch::Tensor positions,
    torch::Tensor saved_leaf_toks,
    torch::Tensor saved_leaf_k,
    int64_t tree_m,
    int64_t selftest_mode,
    c10::optional<torch::Tensor> req_acc_state,
    c10::optional<torch::Tensor> tree_vals,
    bool dynamic_mode,
    double temp,
    double tail_c,
    double gamma) {
  TORCH_CHECK(tree_m >= 8 && tree_m <= 16, "tree_m must be in 8..16");
  TORCH_CHECK(leaf_flag.is_cuda() && leaf_flag.scalar_type() == torch::kInt32 && leaf_flag.numel() >= 1 + tree_m * 3);
  TORCH_CHECK(tree_top.is_cuda() && tree_top.scalar_type() == torch::kInt64 && tree_top.numel() >= 20);
  TORCH_CHECK(input_ids.is_cuda() && input_ids.scalar_type() == torch::kInt32 && input_ids.numel() >= tree_m);
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 && positions.numel() >= tree_m);
  TORCH_CHECK(saved_leaf_toks.is_cuda() && saved_leaf_toks.scalar_type() == torch::kInt64 &&
              saved_leaf_toks.numel() >= tree_m);
  TORCH_CHECK(saved_leaf_k.is_cuda() && saved_leaf_k.scalar_type() == torch::kInt32 &&
              saved_leaf_k.numel() >= tree_m);
  const int tree_k = (tree_top.numel() >= 36) ? 6 : 4;
  const int32_t* acc_ptr = nullptr;
  if (req_acc_state.has_value() && req_acc_state.value().defined() && req_acc_state.value().numel() >= 3) {
    TORCH_CHECK(req_acc_state.value().is_cuda() && req_acc_state.value().scalar_type() == torch::kInt32);
    acc_ptr = req_acc_state.value().data_ptr<int32_t>();
  }
  const float* vals_ptr = nullptr;
  if (tree_vals.has_value() && tree_vals.value().defined() && tree_vals.value().numel() >= 36) {
    TORCH_CHECK(tree_vals.value().is_cuda() && tree_vals.value().scalar_type() == torch::kFloat32);
    vals_ptr = tree_vals.value().data_ptr<float>();
  }
  const float eff_temp = (temp > 1e-3) ? static_cast<float>(temp) : 1.35f;
  const float eff_tail = (tail_c >= 0.0) ? static_cast<float>(tail_c) : 4.0f;
  const float eff_gamma = (gamma > 0.0 && gamma <= 1.0) ? static_cast<float>(gamma) : 0.92f;
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  native_leaf_setup_tree_kernel<<<1, 32, 0, st>>>(
      leaf_flag.data_ptr<int32_t>(),
      tree_top.data_ptr<int64_t>(),
      vals_ptr,
      acc_ptr,
      input_ids.data_ptr<int32_t>(),
      positions.data_ptr<int64_t>(),
      saved_leaf_toks.data_ptr<int64_t>(),
      saved_leaf_k.data_ptr<int32_t>(),
      static_cast<int>(tree_m),
      tree_k,
      static_cast<int>(selftest_mode),
      dynamic_mode ? 1 : 0,
      1.0f / eff_temp,
      eff_tail,
      eff_gamma);
}

void native_leaf_commit_tree(
    torch::Tensor leaf_flag,
    torch::Tensor prev_ns,
    torch::Tensor hit_flag,
    torch::Tensor hit_src_row,
    torch::Tensor saved_leaf_toks,
    torch::Tensor saved_leaf_k,
    torch::Tensor leaf_u_toks,
    torch::Tensor sampled_tokens,
    torch::Tensor num_sampled,
    torch::Tensor num_rejected,
    int64_t tree_m,
    c10::optional<torch::Tensor> stats,
    bool no_hit_mode,
    c10::optional<torch::Tensor> req_acc_state) {
  TORCH_CHECK(tree_m >= 8 && tree_m <= 16, "tree_m must be in 8..16");
  TORCH_CHECK(leaf_flag.is_cuda() && leaf_flag.scalar_type() == torch::kInt32);
  TORCH_CHECK(prev_ns.is_cuda() && prev_ns.scalar_type() == torch::kInt32);
  TORCH_CHECK(hit_flag.is_cuda() && hit_flag.scalar_type() == torch::kInt32);
  TORCH_CHECK(hit_src_row.is_cuda() && hit_src_row.scalar_type() == torch::kInt32);
  TORCH_CHECK(saved_leaf_toks.is_cuda() && saved_leaf_toks.scalar_type() == torch::kInt64 &&
              saved_leaf_toks.numel() >= tree_m);
  TORCH_CHECK(saved_leaf_k.is_cuda() && saved_leaf_k.scalar_type() == torch::kInt32 &&
              saved_leaf_k.numel() >= tree_m);
  TORCH_CHECK(leaf_u_toks.is_cuda() && leaf_u_toks.scalar_type() == torch::kInt64 &&
              leaf_u_toks.numel() >= (tree_m - 7));
  TORCH_CHECK(sampled_tokens.is_cuda() && sampled_tokens.scalar_type() == torch::kInt64 &&
              sampled_tokens.numel() >= 7);
  TORCH_CHECK(num_sampled.is_cuda() && num_sampled.scalar_type() == torch::kInt32);
  TORCH_CHECK(num_rejected.is_cuda() && num_rejected.scalar_type() == torch::kInt32);
  int64_t* stats_ptr = nullptr;
  int stats_len = 0;
  if (stats.has_value() && stats.value().defined() && stats.value().numel() >= 4) {
    TORCH_CHECK(stats.value().is_cuda() && stats.value().scalar_type() == torch::kInt64);
    stats_ptr = stats.value().data_ptr<int64_t>();
    stats_len = static_cast<int>(stats.value().numel());
  }
  int32_t* acc_ptr = nullptr;
  if (req_acc_state.has_value() && req_acc_state.value().defined() && req_acc_state.value().numel() >= 3) {
    TORCH_CHECK(req_acc_state.value().is_cuda() && req_acc_state.value().scalar_type() == torch::kInt32);
    acc_ptr = req_acc_state.value().data_ptr<int32_t>();
  }
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  native_leaf_commit_tree_kernel<<<1, 1, 0, st>>>(
      leaf_flag.data_ptr<int32_t>(),
      prev_ns.data_ptr<int32_t>(),
      hit_flag.data_ptr<int32_t>(),
      hit_src_row.data_ptr<int32_t>(),
      saved_leaf_toks.data_ptr<int64_t>(),
      saved_leaf_k.data_ptr<int32_t>(),
      leaf_u_toks.data_ptr<int64_t>(),
      sampled_tokens.data_ptr<int64_t>(),
      num_sampled.data_ptr<int32_t>(),
      num_rejected.data_ptr<int32_t>(),
      stats_ptr,
      stats_len,
      acc_ptr,
      static_cast<int>(tree_m),
      no_hit_mode ? 1 : 0);
}

void native_leaf_setup(
    torch::Tensor leaf_flag,
    torch::Tensor prev_ns,
    torch::Tensor alt_ids,
    torch::Tensor input_ids,
    torch::Tensor positions,
    torch::Tensor saved_alt_tok,
    int64_t low_thresh,
    int64_t selftest_mode,
    c10::optional<torch::Tensor> alt_margins,
    c10::optional<torch::Tensor> req_acc_state,
    bool m8_mode,
    int64_t policy,
    double margin_alpha) {
  TORCH_CHECK(leaf_flag.is_cuda() && leaf_flag.scalar_type() == torch::kInt32 && leaf_flag.numel() >= 1);
  TORCH_CHECK(prev_ns.is_cuda() && prev_ns.scalar_type() == torch::kInt32 && prev_ns.numel() >= 1);
  TORCH_CHECK(alt_ids.is_cuda() && alt_ids.scalar_type() == torch::kInt64 && alt_ids.numel() >= 2);
  const int64_t min_rows = m8_mode ? 8 : 7;
  TORCH_CHECK(input_ids.is_cuda() && input_ids.scalar_type() == torch::kInt32 && input_ids.numel() >= min_rows);
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 && positions.numel() >= min_rows);
  TORCH_CHECK(saved_alt_tok.is_cuda() && saved_alt_tok.scalar_type() == torch::kInt64 && saved_alt_tok.numel() >= 1);
  const float* margins_ptr = nullptr;
  if (alt_margins.has_value() && alt_margins.value().defined() && alt_margins.value().numel() >= 2) {
    TORCH_CHECK(alt_margins.value().is_cuda() && alt_margins.value().scalar_type() == torch::kFloat32);
    margins_ptr = alt_margins.value().data_ptr<float>();
  }
  const int32_t* acc_ptr = nullptr;
  if (req_acc_state.has_value() && req_acc_state.value().defined() && req_acc_state.value().numel() >= 3) {
    TORCH_CHECK(req_acc_state.value().is_cuda() && req_acc_state.value().scalar_type() == torch::kInt32);
    acc_ptr = req_acc_state.value().data_ptr<int32_t>();
  }
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  native_leaf_setup_kernel<<<1, 1, 0, st>>>(
      leaf_flag.data_ptr<int32_t>(),
      prev_ns.data_ptr<int32_t>(),
      alt_ids.data_ptr<int64_t>(),
      static_cast<int>(alt_ids.numel()),
      margins_ptr,
      acc_ptr,
      input_ids.data_ptr<int32_t>(),
      positions.data_ptr<int64_t>(),
      saved_alt_tok.data_ptr<int64_t>(),
      static_cast<int>(low_thresh),
      static_cast<int>(selftest_mode),
      m8_mode ? 1 : 0,
      static_cast<int>(policy),
      static_cast<float>(margin_alpha));
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
    bool no_hit_mode,
    c10::optional<torch::Tensor> req_acc_state,
    bool m8_mode) {
  TORCH_CHECK(leaf_flag.is_cuda() && leaf_flag.scalar_type() == torch::kInt32);
  TORCH_CHECK(prev_ns.is_cuda() && prev_ns.scalar_type() == torch::kInt32);
  TORCH_CHECK(hit_flag.is_cuda() && hit_flag.scalar_type() == torch::kInt32);
  TORCH_CHECK(saved_alt_tok.is_cuda() && saved_alt_tok.scalar_type() == torch::kInt64);
  TORCH_CHECK(leaf_u_tok.is_cuda() && leaf_u_tok.scalar_type() == torch::kInt64);
  TORCH_CHECK(sampled_tokens.is_cuda() && sampled_tokens.scalar_type() == torch::kInt64 &&
              sampled_tokens.numel() >= (m8_mode ? 3 : 2));
  TORCH_CHECK(num_sampled.is_cuda() && num_sampled.scalar_type() == torch::kInt32);
  TORCH_CHECK(num_rejected.is_cuda() && num_rejected.scalar_type() == torch::kInt32);
  int64_t* stats_ptr = nullptr;
  int stats_len = 0;
  if (stats.has_value() && stats.value().defined() && stats.value().numel() >= 4) {
    TORCH_CHECK(stats.value().is_cuda() && stats.value().scalar_type() == torch::kInt64);
    stats_ptr = stats.value().data_ptr<int64_t>();
    stats_len = static_cast<int>(stats.value().numel());
  }
  int32_t* acc_ptr = nullptr;
  if (req_acc_state.has_value() && req_acc_state.value().defined() && req_acc_state.value().numel() >= 3) {
    TORCH_CHECK(req_acc_state.value().is_cuda() && req_acc_state.value().scalar_type() == torch::kInt32);
    acc_ptr = req_acc_state.value().data_ptr<int32_t>();
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
      stats_len,
      acc_ptr,
      no_hit_mode ? 1 : 0,
      m8_mode ? 1 : 0);
}

void native_leaf_fix_hidden(
    torch::Tensor hit_flag,
    torch::Tensor tensor_ptrs,
    int64_t row_stride_bytes,
    int64_t copy_bytes,
    int64_t src_row,
    c10::optional<torch::Tensor> src_row_buf) {
  TORCH_CHECK(hit_flag.is_cuda() && hit_flag.scalar_type() == torch::kInt32);
  TORCH_CHECK(tensor_ptrs.is_cuda() && tensor_ptrs.scalar_type() == torch::kInt64);
  const int num_tensors = static_cast<int>(tensor_ptrs.numel());
  if (num_tensors <= 0 || copy_bytes <= 0) return;
  TORCH_CHECK((row_stride_bytes & 15) == 0 && (copy_bytes & 15) == 0,
              "native_leaf_fix_hidden requires 16-byte aligned row stride and copy size");
  const int32_t* src_ptr = nullptr;
  if (src_row_buf.has_value() && src_row_buf.value().defined() && src_row_buf.value().numel() >= 1) {
    TORCH_CHECK(src_row_buf.value().is_cuda() && src_row_buf.value().scalar_type() == torch::kInt32);
    src_ptr = src_row_buf.value().data_ptr<int32_t>();
  }
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  native_leaf_fix_hidden_kernel<<<num_tensors, 256, 0, st>>>(
      hit_flag.data_ptr<int32_t>(),
      src_ptr,
      reinterpret_cast<const uint64_t*>(tensor_ptrs.data_ptr<int64_t>()),
      num_tensors,
      row_stride_bytes >> 4,
      copy_bytes >> 4,
      static_cast<int>(src_row));
}

void native_leaf_fix_kv(
    torch::Tensor hit_flag,
    torch::Tensor slot_mappings,
    torch::Tensor kv_ptrs,
    torch::Tensor stride_block,
    torch::Tensor stride_head,
    torch::Tensor stride_slot,
    torch::Tensor block_size,
    torch::Tensor layer_gid,
    int64_t src_row,
    c10::optional<torch::Tensor> src_row_buf) {
  TORCH_CHECK(hit_flag.is_cuda() && hit_flag.scalar_type() == torch::kInt32);
  TORCH_CHECK(slot_mappings.is_cuda() && slot_mappings.scalar_type() == torch::kInt64 &&
              slot_mappings.dim() == 2 && slot_mappings.size(1) > src_row && slot_mappings.stride(1) == 1);
  const int num_layers = static_cast<int>(kv_ptrs.numel());
  if (num_layers <= 0) return;
  const int32_t* src_ptr = nullptr;
  if (src_row_buf.has_value() && src_row_buf.value().defined() && src_row_buf.value().numel() >= 1) {
    TORCH_CHECK(src_row_buf.value().is_cuda() && src_row_buf.value().scalar_type() == torch::kInt32);
    src_ptr = src_row_buf.value().data_ptr<int32_t>();
  }
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  native_leaf_fix_kv_all_layers_kernel<<<num_layers, 64, 0, st>>>(
      hit_flag.data_ptr<int32_t>(),
      src_ptr,
      slot_mappings.data_ptr<int64_t>(),
      slot_mappings.stride(0),
      reinterpret_cast<const uint64_t*>(kv_ptrs.data_ptr<int64_t>()),
      stride_block.data_ptr<int64_t>(),
      stride_head.data_ptr<int64_t>(),
      stride_slot.data_ptr<int64_t>(),
      block_size.data_ptr<int32_t>(),
      layer_gid.data_ptr<int32_t>(),
      num_layers,
      static_cast<int>(src_row));
}
