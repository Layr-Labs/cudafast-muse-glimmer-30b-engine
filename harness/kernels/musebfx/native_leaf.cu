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
#include <cstdlib>
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

// Calibrated path-probability scorer for Q=16 hybrid INT4/INT3 (5-fold CV validated on 25 real prompts / 737 steps).
// Pre-miss:  p(j,r) = exp(a_j + w_sm*sm_c + w_z0*z0_c + c_r + beta*(z_r - z_0)) / (1 + sum_{r<6} exp(...))
// Post-miss: p_post(j,r) = exp(a_j + a2 + w_z0p*z0_c + c_r + beta_post*(z_r - z_0)) / (1 + sum_{r<6} exp(...))
// L3+ after L2 sub hit resyncs with drafter top-1 (uses p1[j], empirical 63.3%); L1 also spawns d_{k+1}#2 via p2post (14.2%).
__constant__ float kCalA[6] = {1.675690f, 1.258784f, 1.070196f, 1.171448f, 1.102070f, 0.731610f};
__constant__ float kCalC[8] = {0.000000f, -0.402942f, -0.763639f, -0.817452f, -1.416669f, -1.642223f, -1.750000f, -1.900000f};
#define CAL_A2 (-1.614419f)
#define CAL_BETA 0.851144f
#define CAL_BETA_POST 0.326972f
#define CAL_W_SM_PRE 0.472339f
#define CAL_W_Z0_PRE 0.201693f
#define CAL_W_Z0_POST 0.178773f
#define CAL_SHIFT 0.0349f

// Arbitrary-depth (1..6-level) dynamic multi-leaf tree setup (rows 7..tree_m-1, tree_m <= 32), launched as 1 warp (<<<1, 32>>>, <3 us):
//   - Bit-packed remap inside int32[97] (1 + 32*3):
//       leaf_flag[1 + 3*r + 0] = seq_sub  (seqlen_k_eff = seqlen_k - seq_sub)
//       leaf_flag[1 + 3*r + 1] = base_sub (-1 for causal rows; else tree_m - k0 of root ancestor)
//       leaf_flag[1 + 3*r + 2] = remap_packed (5-bit slot j = physical row r_j in 0..31 for tok = p + k0 + j, j=0..5)
//   - dynamic_mode==1 (default, tree_vals!=null, tree_k>=6): greedy EV allocator across 1-level candidates
//     plus dynamically spawned arbitrary-depth (2..max_depth <= 6) zero-extra-drafter children:
//       - Substitution continuation (`sub`, kind=1): aligned d_{k+1}#1 -> d_{k+2}#1 -> ... -> d_6#1
//       - Shift recovery (`shf`, kind=2): off-by-1 d_k#1 -> d_{k+1}#1 -> ... -> d_5#1
//       - Post-miss top-2 (`alt2`, kind=3) and for tree_m > 16 also top-8 L1 + `alt3` (kind=4) + L2 `alt2`.
//     When tree_m <= 16, uses exact top-6 (eff_k=6, non-rich) preserving bit-identical Wave 34/37 M=16 behavior.
//   - selftest_mode!=0: places a full deep chain (6-level on M=16, 5-level on M=12) mirroring d1..d6.
__global__ void native_leaf_setup_tree_kernel(
    int32_t* __restrict__ leaf_flag,
    const int64_t* __restrict__ tree_top,
    const float* __restrict__ tree_vals,
    const int32_t* __restrict__ req_acc_state,
    int32_t* __restrict__ input_ids,
    int64_t* __restrict__ positions,
    int64_t* __restrict__ saved_leaf_toks,
    int32_t* __restrict__ saved_leaf_k,
    int32_t* __restrict__ saved_leaf_parent,
    int tree_m,
    int tree_k,
    int selftest_mode,
    int dynamic_mode,
    float inv_temp,
    float tail_c,
    float gamma,
    int max_depth,
    float sub_coef,
    float shf_coef,
    float cont_decay,
    int adapt_gamma,
    int calib_mode) {
  const int tid = threadIdx.x;
  const bool high_acc = (req_acc_state != nullptr && req_acc_state[2] != 0);
  const int eff_max_depth = (max_depth < 1) ? 1 : ((max_depth > 6) ? 6 : max_depth);
  // Online request-adaptive gamma_eff (used only when calib_mode == 0).
  float eff_gamma = gamma;
  if (calib_mode == 0 && adapt_gamma != 0 && req_acc_state != nullptr) {
    const int32_t rr = req_acc_state[0];
    const int32_t cc = req_acc_state[1];
    if (rr >= 3 && cc > rr) {
      const float mean_ns = static_cast<float>(cc) / static_cast<float>(rr);
      const float chain_ns = (mean_ns > 1.45f) ? (mean_ns - 0.45f) : 1.0f;
      const float p_emp = (chain_ns - 1.0f) / chain_ns;
      const float g_adapt = 0.5f * gamma + 0.5f * (p_emp + 0.18f);
      if (g_adapt < eff_gamma) {
        eff_gamma = (g_adapt < 0.78f) ? 0.78f : g_adapt;
      }
    }
  }

  // Chain rows 0..6: plain causal lengths p+1 .. p+7, no KV remap.
  if (tid < 7) {
    leaf_flag[1 + 3 * tid + 0] = tree_m - 1 - tid;
    leaf_flag[1 + 3 * tid + 1] = -1;
    leaf_flag[1 + 3 * tid + 2] = 0;
    if (saved_leaf_parent != nullptr) {
      saved_leaf_parent[tid] = -1;
    }
  }

  const int num_leaves = tree_m - 7;
  if (dynamic_mode == 0 || selftest_mode != 0 || tree_vals == nullptr || tree_k < 6) {
    if (tid == 0) {
      constexpr int kLeafK[9] = {1, 1, 2, 2, 3, 1, 3, 4, 5};
      constexpr int kLeafRank[9] = {1, 2, 1, 2, 1, 3, 2, 1, 1};
      int row_base_sub[32];
      int row_depth[32];
      uint32_t row_packed[32];
      bool any_active = false;
      for (int li = 0; li < num_leaves; ++li) {
        const int r = 7 + li;
        int k = (li < 9) ? kLeafK[li] : 1;
        int par = -1;
        int64_t cand = -1;
        bool valid = false;
        if (!high_acc) {
          if (selftest_mode != 0) {
            // Deep chain selftest: rows 7..12 mirror d1..d6 (depth 1..6), rows 13..tree_m-1 mirror d1..d6 (depth 1).
            if (li < 6) {
              k = li + 1;
              par = (li == 0) ? -1 : (r - 1);
            } else {
              k = ((li - 6) % 6) + 1;
              par = -1;
            }
            cand = static_cast<int64_t>(input_ids[k]);
            valid = true;
          } else if (li < 9) {
            const int rank = kLeafRank[li];
            const int64_t dk = static_cast<int64_t>(input_ids[k]);
            const int64_t top1 = tree_top[(k - 1) * tree_k + 0];
            cand = tree_top[(k - 1) * tree_k + rank];
            valid = (top1 == dk && cand >= 0 && cand != dk);
          }
        }
        if (valid) {
          any_active = true;
          int b_sub = 0;
          int dp = 1;
          uint32_t pkd = 0u;
          if (par < 0) {
            b_sub = tree_m - k;
            dp = 1;
            pkd = static_cast<uint32_t>(r) & 0x1Fu;
          } else {
            b_sub = row_base_sub[par];
            dp = row_depth[par] + 1;
            pkd = row_packed[par] | ((static_cast<uint32_t>(r) & 0x1Fu) << (5 * (dp - 1)));
          }
          row_base_sub[r] = b_sub;
          row_depth[r] = dp;
          row_packed[r] = pkd;
          input_ids[r] = static_cast<int32_t>(cand);
          positions[r] = positions[k];
          saved_leaf_toks[r] = cand;
          saved_leaf_k[r] = k;
          if (saved_leaf_parent != nullptr) {
            saved_leaf_parent[r] = par;
          }
          leaf_flag[1 + 3 * r + 0] = tree_m - (k + 1);
          leaf_flag[1 + 3 * r + 1] = b_sub;
          leaf_flag[1 + 3 * r + 2] = static_cast<int32_t>(pkd);
        } else {
          row_base_sub[r] = -1;
          row_depth[r] = 0;
          row_packed[r] = 0u;
          input_ids[r] = input_ids[0];
          saved_leaf_toks[r] = -1;
          saved_leaf_k[r] = 0;
          if (saved_leaf_parent != nullptr) {
            saved_leaf_parent[r] = -1;
          }
          leaf_flag[1 + 3 * r + 0] = tree_m - 1 - r;
          leaf_flag[1 + 3 * r + 1] = -1;
          leaf_flag[1 + 3 * r + 2] = 0;
        }
      }
      leaf_flag[0] = any_active ? -1 : 0;
    }
    return;
  }

  // Dynamic per-step GPU arbitrary-depth (1..6-level) tree allocator (1 warp, up to 128 candidates).
  const bool wide_rich = (tree_m > 16);
  const int eff_k = (wide_rich && tree_k >= 8) ? 8 : 6;
  const int init_cands = 6 * (eff_k - 1);

  __shared__ float s_Z[6];
  __shared__ float s_A[6];
  __shared__ float s_p1[6];
  __shared__ float s_p1post[6];
  __shared__ float s_p2post[6];
  __shared__ float s_p3post[6];
  __shared__ float s_pr[6][8];
  __shared__ float s_z0[6];
  __shared__ float s_margin[6];
  __shared__ int64_t s_dk[6];
  __shared__ uint8_t s_valid_row[6];
  __shared__ int s_any_active;

  __shared__ float s_cand_score[128];
  __shared__ int64_t s_cand_tok[128];
  __shared__ int8_t s_cand_k[128];
  __shared__ int8_t s_cand_par[128];
  __shared__ int8_t s_cand_depth[128];
  __shared__ int8_t s_cand_kind[128];  // 0=1-level, 1=sub, 2=shf, 3=alt2, 4=alt3
  __shared__ int s_num_cands;

  __shared__ int s_row_base_sub[32];
  __shared__ uint32_t s_row_packed[32];

  // Pass 1: validate rows & gather per-position top-1 margin z0 - z1 for step-level difficulty context.
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
    const int64_t top2 = tree_top[base + 1];
    const float z1 = tree_vals[base + 1];
    s_margin[tid] = (row_ok && top2 >= 0 && z1 > -1e20f) ? (z0 - z1) : 2.5f;
  }
  __syncwarp();

  if (tid < 6) {
    const int base = tid * tree_k;
    const float z0 = s_z0[tid];
    const bool row_ok = (s_valid_row[tid] != 0);
    if (row_ok) {
      if (calib_mode != 0) {
        float sm_c = (s_margin[0] + s_margin[1] + s_margin[2] +
                      s_margin[3] + s_margin[4] + s_margin[5]) * (1.0f / 6.0f) - 2.5f;
        if (sm_c < -2.0f) sm_c = -2.0f;
        if (sm_c > 2.5f) sm_c = 2.5f;
        float z0_c = z0 - 36.0f;
        if (z0_c < -10.0f) z0_c = -10.0f;
        if (z0_c > 6.0f) z0_c = 6.0f;
        const float a_pre = kCalA[tid] + CAL_W_SM_PRE * sm_c + CAL_W_Z0_PRE * z0_c;
        const float a_post = kCalA[tid] + CAL_A2 + CAL_W_Z0_POST * z0_c;
        float e0[8], e1[8];
        float den0 = 1.0f, den1 = 1.0f;
        for (int rk = 0; rk < eff_k; ++rk) {
          const int64_t tok_r = tree_top[base + rk];
          const float zr = tree_vals[base + rk];
          if (rk == 0 || (tok_r >= 0 && zr > -1e20f)) {
            const float dz = zr - z0;
            const float g0 = kCalC[rk] + CAL_BETA * dz;
            const float g1 = kCalC[rk] + CAL_BETA_POST * dz;
            e0[rk] = __expf(a_pre + g0);
            e1[rk] = __expf(a_post + g1);
          } else {
            e0[rk] = 0.0f;
            e1[rk] = 0.0f;
          }
          den0 += e0[rk];
          den1 += e1[rk];
        }
        for (int rk = 0; rk < eff_k; ++rk) {
          s_pr[tid][rk] = e0[rk] / den0;
        }
        s_p1post[tid] = e1[0] / den1;
        s_p2post[tid] = e1[1] / den1;
        s_p3post[tid] = e1[2] / den1;
        s_Z[tid] = den0;
        s_p1[tid] = s_pr[tid][0];
      } else {
        float sum_e = 1.0f;
        float last_e = 0.0f;
        for (int rk = 1; rk < eff_k; ++rk) {
          const int64_t tok_r = tree_top[base + rk];
          const float zr = tree_vals[base + rk];
          if (tok_r >= 0 && zr > -1e20f) {
            const float er = __expf((zr - z0) * inv_temp);
            sum_e += er;
            last_e = er;
          }
        }
        const float z_tot = sum_e + tail_c * last_e;
        s_Z[tid] = z_tot;
        const float p1 = eff_gamma / z_tot;
        s_p1[tid] = (p1 < 1.0f) ? p1 : 1.0f;
        s_p1post[tid] = 0.0f;
        s_p2post[tid] = 0.0f;
        s_p3post[tid] = 0.0f;
      }
    } else {
      s_Z[tid] = 1e30f;
      s_p1[tid] = 0.0f;
      s_p1post[tid] = 0.0f;
      s_p2post[tid] = 0.0f;
      s_p3post[tid] = 0.0f;
    }
  }
  __syncwarp();

  if (tid == 0) {
    float a = 1.0f;
    for (int j = 0; j < 6; ++j) {
      s_A[j] = a;
      if (s_valid_row[j]) {
        a *= s_p1[j];
      } else {
        a = 0.0f;
      }
    }
    s_any_active = 0;
    s_num_cands = init_cands;
  }
  __syncwarp();

  // Initialize 1-level candidates in slots 0..init_cands-1 (30 for K=6, 42 for K=8), clear all 128 slots.
  s_cand_score[tid] = -1.0f;
  s_cand_score[tid + 32] = -1.0f;
  s_cand_score[tid + 64] = -1.0f;
  s_cand_score[tid + 96] = -1.0f;
  const int cands_per_row = eff_k - 1;
  for (int cid = tid; cid < init_cands; cid += 32) {
    const int row_idx = cid / cands_per_row;                  // 0..5 -> depth k = row_idx + 1
    const int rank = 1 + (cid - row_idx * cands_per_row);     // 1..eff_k-1 -> candidate rank #2..#eff_k
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
        const float pr = (calib_mode != 0)
            ? s_pr[row_idx][rank]
            : (eff_gamma * __expf((zr - s_z0[row_idx]) * inv_temp) / s_Z[row_idx]);
        s_cand_score[cid] = s_A[row_idx] * pr;
        s_cand_tok[cid] = cand;
        s_cand_k[cid] = static_cast<int8_t>(row_idx + 1);
        s_cand_par[cid] = -1;
        s_cand_depth[cid] = 1;
        s_cand_kind[cid] = 0;
      }
    }
  }
  __syncwarp();

  for (int li = 0; li < num_leaves; ++li) {
    const int r = 7 + li;
    const int nc = s_num_cands;
    float best_s = -1.0f;
    int best_idx = tid;
#pragma unroll
    for (int off = 0; off < 128; off += 32) {
      const int idx = tid + off;
      if (idx < nc) {
        const float sv = s_cand_score[idx];
        if (sv > best_s || (sv == best_s && idx < best_idx)) {
          best_s = sv;
          best_idx = idx;
        }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float os = __shfl_xor_sync(0xffffffffu, best_s, o);
      const int oi = __shfl_xor_sync(0xffffffffu, best_idx, o);
      if (os > best_s || (os == best_s && oi < best_idx)) {
        best_s = os;
        best_idx = oi;
      }
    }
    if (tid == 0) {
      if (best_s > 0.0f) {
        const int64_t tok = s_cand_tok[best_idx];
        const int k_r = static_cast<int>(s_cand_k[best_idx]);
        const int par = static_cast<int>(s_cand_par[best_idx]);
        const int dp = static_cast<int>(s_cand_depth[best_idx]);
        const int kind = static_cast<int>(s_cand_kind[best_idx]);
        s_cand_score[best_idx] = -1.0f;

        int b_sub = 0;
        uint32_t pkd = 0u;
        if (par < 0) {
          b_sub = tree_m - k_r;
          pkd = static_cast<uint32_t>(r) & 0x1Fu;
        } else {
          b_sub = s_row_base_sub[par];
          pkd = s_row_packed[par] | ((static_cast<uint32_t>(r) & 0x1Fu) << (5 * (dp - 1)));
        }
        s_row_base_sub[r] = b_sub;
        s_row_packed[r] = pkd;

        input_ids[r] = static_cast<int32_t>(tok);
        positions[r] = positions[k_r];
        saved_leaf_toks[r] = tok;
        saved_leaf_k[r] = k_r;
        if (saved_leaf_parent != nullptr) {
          saved_leaf_parent[r] = par;
        }
        leaf_flag[1 + 3 * r + 0] = tree_m - (k_r + 1);
        leaf_flag[1 + 3 * r + 1] = b_sub;
        leaf_flag[1 + 3 * r + 2] = static_cast<int32_t>(pkd);
        s_any_active = 1;

        // Dynamically spawn depth+1 zero-extra-drafter children (up to eff_max_depth <= 6, k_child <= 6).
        if (li + 1 < num_leaves && dp < eff_max_depth && k_r < 6 && saved_leaf_parent != nullptr) {
          const int k_child = k_r + 1;
          int64_t sub_tok_spawned = -1;
          int64_t alt2_tok_spawned = -1;
          int64_t alt3_tok_spawned = -1;
          // 1) Aligned substitution continuation (`sub`, kind=1): d_{k_r+1}#1 = s_dk[k_r].
          //    After L2 sub hit (kind==1), context has resynced with drafter top-1 -> use s_p1[k_r] (empirical 63.3%).
          if ((kind == 0 || kind == 1 || kind == 3 || kind == 4) && s_valid_row[k_r] && s_num_cands < 128) {
            const float p_cont = (kind == 1) ? s_p1[k_r] : s_p1post[k_r];
            const float child_s = (calib_mode != 0)
                ? (best_s * p_cont)
                : (best_s * ((kind == 0) ? sub_coef : cont_decay) * s_p1[k_r]);
            const int64_t sub_tok = s_dk[k_r];
            if (child_s > 0.0f && sub_tok >= 0) {
              const int slot = s_num_cands++;
              s_cand_score[slot] = child_s;
              s_cand_tok[slot] = sub_tok;
              s_cand_k[slot] = static_cast<int8_t>(k_child);
              s_cand_par[slot] = static_cast<int8_t>(r);
              s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
              s_cand_kind[slot] = 1;
              sub_tok_spawned = sub_tok;
            }
          }
          // 2) Post-miss top-2 continuation (`alt2`, kind=3): d_{k_r+1}#2 = tree_top[k_r*tree_k + 1] (empirical 14.2%).
          //    In wide_rich mode (tree_m > 16), also allow L2 alt2 branching from kind==1 or kind==3.
          if (calib_mode != 0 && (kind == 0 || (wide_rich && (kind == 1 || kind == 3))) &&
              s_valid_row[k_r] && s_num_cands < 128) {
            const int64_t alt2_tok = tree_top[k_r * tree_k + 1];
            if (alt2_tok >= 0 && alt2_tok != sub_tok_spawned) {
              const float child_s = best_s * s_p2post[k_r];
              if (child_s > 0.0f) {
                const int slot = s_num_cands++;
                s_cand_score[slot] = child_s;
                s_cand_tok[slot] = alt2_tok;
                s_cand_k[slot] = static_cast<int8_t>(k_child);
                s_cand_par[slot] = static_cast<int8_t>(r);
                s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
                s_cand_kind[slot] = 3;
                alt2_tok_spawned = alt2_tok;
              }
            }
          }
          // 3) Post-miss top-3 continuation (`alt3`, kind=4, wide_rich tree_m > 16 only; includes L3 after sub/alt2).
          if (wide_rich && calib_mode != 0 && (kind == 0 || kind == 1 || kind == 3) &&
              s_valid_row[k_r] && s_num_cands < 128) {
            const int64_t alt3_tok = tree_top[k_r * tree_k + 2];
            if (alt3_tok >= 0 && alt3_tok != sub_tok_spawned && alt3_tok != alt2_tok_spawned) {
              const float child_s = best_s * s_p3post[k_r];
              if (child_s > 0.0f) {
                const int slot = s_num_cands++;
                s_cand_score[slot] = child_s;
                s_cand_tok[slot] = alt3_tok;
                s_cand_k[slot] = static_cast<int8_t>(k_child);
                s_cand_par[slot] = static_cast<int8_t>(r);
                s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
                s_cand_kind[slot] = 4;
                alt3_tok_spawned = alt3_tok;
              }
            }
          }
          // 4) Off-by-1 shift recovery (`shf`, kind=2): d_{k_r}#1 = s_dk[k_r - 1].
          //    Skipped in wide_rich (0/196 empirical hits when top-8 + alt2/alt3 are active; M=16 unchanged).
          if (!wide_rich && (kind == 0 || kind == 2) && s_valid_row[k_r - 1] && s_num_cands < 128) {
            const int64_t shf_tok = s_dk[k_r - 1];
            if (shf_tok >= 0 && shf_tok != sub_tok_spawned && shf_tok != alt2_tok_spawned &&
                shf_tok != alt3_tok_spawned) {
              const float child_s = (calib_mode != 0)
                  ? ((kind == 0) ? (best_s * CAL_SHIFT) : (best_s * s_p1[k_r - 1]))
                  : ((kind == 0) ? (best_s * shf_coef) : (best_s * cont_decay * s_p1[k_r - 1]));
              if (child_s > 0.0f) {
                const int slot = s_num_cands++;
                s_cand_score[slot] = child_s;
                s_cand_tok[slot] = shf_tok;
                s_cand_k[slot] = static_cast<int8_t>(k_child);
                s_cand_par[slot] = static_cast<int8_t>(r);
                s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
                s_cand_kind[slot] = 2;
              }
            }
          }
        }
      } else {
        s_row_base_sub[r] = -1;
        s_row_packed[r] = 0u;
        input_ids[r] = input_ids[0];
        saved_leaf_toks[r] = -1;
        saved_leaf_k[r] = 0;
        if (saved_leaf_parent != nullptr) {
          saved_leaf_parent[r] = -1;
        }
        leaf_flag[1 + 3 * r + 0] = tree_m - 1 - r;
        leaf_flag[1 + 3 * r + 1] = -1;
        leaf_flag[1 + 3 * r + 2] = 0;
      }
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
    int hit_flag_len,
    int32_t* __restrict__ hit_src_row,
    const int64_t* __restrict__ saved_leaf_toks,
    const int32_t* __restrict__ saved_leaf_k,
    const int32_t* __restrict__ saved_leaf_parent,
    const int64_t* __restrict__ leaf_u_toks,
    int64_t* __restrict__ sampled_tokens,
    int32_t* __restrict__ num_sampled,
    int32_t* __restrict__ num_rejected,
    int64_t* __restrict__ stats,
    int stats_len,
    int32_t* __restrict__ req_acc_state,
    int tree_m,
    int no_hit_mode,
    float high_acc_cpc) {
  const int mode = leaf_flag[0];
  leaf_flag[0] = 0;  // Clear immediately so non-tree steps see 0.
  int32_t ns = num_sampled[0];
  int32_t nr = num_rejected[0];
  int32_t hit_dst0 = 0;
  int32_t hit_src0 = 0;
  int32_t hit_len = 0;

  // Supports arbitrary-depth (1..6-level) commits up to ns == 7 (nr == 0) strictly preserving ns + nr == 7.
  if (mode < 0 && ns >= 1 && ns <= 6 && no_hit_mode == 0) {
    int32_t cur_parent = -1;
    for (int stage = 0; stage < 6 && ns <= 6; ++stage) {
      const int64_t tok_k = sampled_tokens[ns - 1];
      int matched_r = -1;
      for (int r = 7; r < tree_m; ++r) {
        const int32_t par = (saved_leaf_parent != nullptr) ? saved_leaf_parent[r] : -1;
        if (par == cur_parent && saved_leaf_k[r] == ns &&
            saved_leaf_toks[r] >= 0 && tok_k == saved_leaf_toks[r]) {
          matched_r = r;
          break;
        }
      }
      if (matched_r < 0) break;
      const int32_t dst_r = ns;
      sampled_tokens[dst_r] = leaf_u_toks[matched_r - 7];
      if (hit_len == 0) {
        hit_dst0 = dst_r;
        hit_src0 = matched_r;
      }
      if (hit_flag_len >= 2 + 2 * (hit_len + 1)) {
        hit_flag[2 + 2 * hit_len] = dst_r;
        hit_flag[3 + 2 * hit_len] = matched_r;
      }
      hit_len += 1;
      ns += 1;
      nr -= 1;
      cur_parent = matched_r;
    }
    if (hit_len > 0) {
      num_sampled[0] = ns;
      num_rejected[0] = nr;
    }
  }

  hit_flag[0] = hit_dst0;
  if (hit_flag_len >= 2) {
    hit_flag[1] = hit_len;
  }
  hit_src_row[0] = hit_src0;
  prev_ns[0] = ns;

  if (req_acc_state != nullptr) {
    const int32_t r = req_acc_state[0] + 1;
    const int32_t c = req_acc_state[1] + ns;
    req_acc_state[0] = r;
    req_acc_state[1] = c;
    // On M=16 verify, pausing leaves saves 0.00 ms because FULL(16) always runs; keep all 9 leaves
    // active unless committed_token_ceiling hard-latches req_acc_state[2]==2 (or MUSE_TREE_HIGH_ACC_CPC > 0).
    if (req_acc_state[2] != 2) {
      req_acc_state[2] =
          (high_acc_cpc > 0.0f && r >= 4 && static_cast<float>(c) > high_acc_cpc * static_cast<float>(r)) ? 1 : 0;
    }
  }

  if (stats != nullptr && stats_len >= 4) {
    stats[0] += 1;
    if (mode < 0) stats[1] += 1;
    if (hit_len > 0) {
      stats[2] += 1;
      if (hit_len >= 2) stats[3] += 1;
      if (stats_len >= 10 && hit_dst0 >= 1 && hit_dst0 <= 6) {
        stats[3 + hit_dst0] += 1;  // stats[4..9] = initial 1-level hit depth k=1..6
      }
      if (stats_len >= 19 && hit_src0 >= 7 && hit_src0 < 16) {
        stats[10 + (hit_src0 - 7)] += 1;  // stats[10..18] = winning initial leaf slot 0..8
      }
      if (stats_len >= 32) {
        const int hl = (hit_len <= 6) ? hit_len : 6;
        stats[18 + hl] += 1;                 // stats[19..24] = hit_len histogram 1..6
        stats[25] += hit_len;                // total extra tokens committed by tree
        if (hit_len > 1) {
          stats[26] += (hit_len - 1);        // deep bonus tokens (stage >= 1)
          stats[29] += 1;                    // reached depth >= 2
          if (hit_len >= 3) stats[30] += 1;  // reached depth >= 3
          if (hit_len >= 4) stats[31] += 1;  // reached depth >= 4
        }
        if (hit_len > stats[28]) stats[28] = hit_len;  // max hit_len seen
      }
    }
    if (stats_len >= 32) {
      stats[27] += ns;  // total committed tokens
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
    int hit_flag_len,
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
  if (hit_flag_len >= 2) {
    hit_flag[1] = 0;
  }
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

// Copies winning tree row(s) src -> dst across up to 8 tensors in ONE launch.
// Supports arbitrary-depth 1..6 multi-node copy when hit_flag_len >= 14 && hit_flag[1] > 0;
// falls back to single-row legacy M=7/8 copy otherwise. Early-exits in ~1 us when hit_flag[0] == 0.
__global__ void native_leaf_fix_hidden_kernel(
    const int32_t* __restrict__ hit_flag,
    int hit_flag_len,
    const int32_t* __restrict__ src_row_ptr,
    const uint64_t* __restrict__ tensor_ptrs,
    int num_tensors,
    int64_t row_stride_u4,
    int64_t copy_u4,
    int src_row) {
  const int dst0 = hit_flag[0];
  if (dst0 <= 0) return;
  const int b = blockIdx.x;
  if (b >= num_tensors) return;
  uint4* base = reinterpret_cast<uint4*>(static_cast<uintptr_t>(tensor_ptrs[b]));

  if (hit_flag_len >= 14 && hit_flag[1] > 0) {
    int h_len = hit_flag[1];
    if (h_len > 6) h_len = 6;
    for (int h = 0; h < h_len; ++h) {
      const int dst_row = hit_flag[2 + 2 * h];
      const int eff_src = hit_flag[3 + 2 * h];
      if (dst_row <= 0 || eff_src <= 0) continue;
      uint4* dst = base + static_cast<int64_t>(dst_row) * row_stride_u4;
      const uint4* src = base + static_cast<int64_t>(eff_src) * row_stride_u4;
      for (int64_t i = threadIdx.x; i < copy_u4; i += blockDim.x) {
        dst[i] = src[i];
      }
    }
    return;
  }

  const int eff_src = (src_row_ptr != nullptr) ? src_row_ptr[0] : src_row;
  if (eff_src <= 0) return;
  uint4* dst = base + static_cast<int64_t>(dst0) * row_stride_u4;
  const uint4* src = base + static_cast<int64_t>(eff_src) * row_stride_u4;
  for (int64_t i = threadIdx.x; i < copy_u4; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// Copies winning KV cache slot(p+src_r) -> slot(p+dst_r) for both KV heads (K+V = 1024 B/layer)
// across all target GQA-16 layers in a single launch (supports 1..6 multi-node deep tree hits).
// Grid: (num_layers), Block: 64 threads (1 uint4 per thread -> 2 heads x 32 uint4 = 1024 B).
__global__ void native_leaf_fix_kv_all_layers_kernel(
    const int32_t* __restrict__ hit_flag,
    int hit_flag_len,
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
  const int dst0 = hit_flag[0];
  if (dst0 <= 0) return;
  const int L = blockIdx.x;
  if (L >= num_layers) return;

  const int gi = layer_gid[L];
  const int64_t gi_off = static_cast<int64_t>(gi) * slot_stride;
  const int64_t bs = static_cast<int64_t>(block_size[L]);
  const int t = threadIdx.x;
  const int h = t >> 5;     // 0 or 1 (2 KV heads)
  const int vec = t & 31;   // 0..31 (32 * 16B = 512B = 256 bf16 = 128 K + 128 V)
  __nv_bfloat16* kv = reinterpret_cast<__nv_bfloat16*>(static_cast<uintptr_t>(kv_ptrs[L]));
  const int64_t sB = stride_block[L];
  const int64_t sH = stride_head[L];
  const int64_t sS = stride_slot[L];

  if (hit_flag_len >= 14 && hit_flag[1] > 0) {
    int h_len = hit_flag[1];
    if (h_len > 6) h_len = 6;
    for (int k = 0; k < h_len; ++k) {
      const int dst_row = hit_flag[2 + 2 * k];
      const int eff_src = hit_flag[3 + 2 * k];
      if (dst_row <= 0 || eff_src <= 0) continue;
      const int64_t s_dst = slot_mappings[gi_off + dst_row];
      const int64_t s_src = slot_mappings[gi_off + eff_src];
      if (s_dst < 0 || s_src < 0) continue;
      const int64_t blk_dst = s_dst / bs;
      const int64_t off_dst = s_dst - blk_dst * bs;
      const int64_t blk_src = s_src / bs;
      const int64_t off_src = s_src - blk_src * bs;
      uint4* dst = reinterpret_cast<uint4*>(kv + blk_dst * sB + h * sH + off_dst * sS);
      const uint4* src = reinterpret_cast<const uint4*>(kv + blk_src * sB + h * sH + off_src * sS);
      dst[vec] = src[vec];
    }
    return;
  }

  const int eff_src = (src_row_ptr != nullptr) ? src_row_ptr[0] : src_row;
  if (eff_src <= 0) return;
  const int64_t s_dst = slot_mappings[gi_off + dst0];
  const int64_t s_src = slot_mappings[gi_off + eff_src];
  if (s_dst < 0 || s_src < 0) return;
  const int64_t blk_dst = s_dst / bs;
  const int64_t off_dst = s_dst - blk_dst * bs;
  const int64_t blk_src = s_src / bs;
  const int64_t off_src = s_src - blk_src * bs;
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
  TORCH_CHECK(tree_m >= 8 && tree_m <= 32, "tree_m must be in 8..32");
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
    double gamma,
    c10::optional<torch::Tensor> saved_leaf_parent,
    int64_t max_depth,
    double sub_coef,
    double shf_coef,
    double cont_decay) {
  TORCH_CHECK(tree_m >= 8 && tree_m <= 32, "tree_m must be in 8..32");
  TORCH_CHECK(leaf_flag.is_cuda() && leaf_flag.scalar_type() == torch::kInt32 && leaf_flag.numel() >= 1 + tree_m * 3);
  TORCH_CHECK(tree_top.is_cuda() && tree_top.scalar_type() == torch::kInt64 && tree_top.numel() >= 20);
  TORCH_CHECK(input_ids.is_cuda() && input_ids.scalar_type() == torch::kInt32 && input_ids.numel() >= tree_m);
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 && positions.numel() >= tree_m);
  TORCH_CHECK(saved_leaf_toks.is_cuda() && saved_leaf_toks.scalar_type() == torch::kInt64 &&
              saved_leaf_toks.numel() >= tree_m);
  TORCH_CHECK(saved_leaf_k.is_cuda() && saved_leaf_k.scalar_type() == torch::kInt32 &&
              saved_leaf_k.numel() >= tree_m);
  const int tree_k = (tree_top.numel() >= 48) ? 8 : ((tree_top.numel() >= 36) ? 6 : 4);
  const int32_t* acc_ptr = nullptr;
  if (req_acc_state.has_value() && req_acc_state.value().defined() && req_acc_state.value().numel() >= 3) {
    TORCH_CHECK(req_acc_state.value().is_cuda() && req_acc_state.value().scalar_type() == torch::kInt32);
    acc_ptr = req_acc_state.value().data_ptr<int32_t>();
  }
  const float* vals_ptr = nullptr;
  if (tree_vals.has_value() && tree_vals.value().defined() && tree_vals.value().numel() >= 6 * tree_k) {
    TORCH_CHECK(tree_vals.value().is_cuda() && tree_vals.value().scalar_type() == torch::kFloat32);
    vals_ptr = tree_vals.value().data_ptr<float>();
  }
  int32_t* parent_ptr = nullptr;
  if (saved_leaf_parent.has_value() && saved_leaf_parent.value().defined() &&
      saved_leaf_parent.value().numel() >= tree_m) {
    TORCH_CHECK(saved_leaf_parent.value().is_cuda() && saved_leaf_parent.value().scalar_type() == torch::kInt32);
    parent_ptr = saved_leaf_parent.value().data_ptr<int32_t>();
  }
  const float eff_temp = (temp > 1e-3) ? static_cast<float>(temp) : 1.35f;
  const float eff_tail = (tail_c >= 0.0) ? static_cast<float>(tail_c) : 4.0f;
  const float eff_gamma = (gamma > 0.0 && gamma <= 1.0) ? static_cast<float>(gamma) : 0.92f;
  const int eff_depth = (max_depth >= 1 && max_depth <= 6) ? static_cast<int>(max_depth) : 6;
  const float eff_sub = (sub_coef >= 0.0) ? static_cast<float>(sub_coef) : 0.55f;
  const float eff_shf = (shf_coef >= 0.0) ? static_cast<float>(shf_coef) : 0.20f;
  const float eff_cont = (cont_decay >= 0.0) ? static_cast<float>(cont_decay) : 0.88f;
  static const int adapt_gamma_env = []() -> int {
    const char* s = std::getenv("MUSE_TREE_ADAPT_GAMMA");
    return (!s || !*s || std::strtol(s, nullptr, 10) != 0) ? 1 : 0;
  }();
  static const int calib_env = []() -> int {
    const char* s = std::getenv("MUSE_TREE_CALIB");
    return (!s || !*s || std::strtol(s, nullptr, 10) != 0) ? 1 : 0;
  }();
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
      parent_ptr,
      static_cast<int>(tree_m),
      tree_k,
      static_cast<int>(selftest_mode),
      dynamic_mode ? 1 : 0,
      1.0f / eff_temp,
      eff_tail,
      eff_gamma,
      eff_depth,
      eff_sub,
      eff_shf,
      eff_cont,
      adapt_gamma_env,
      calib_env);
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
    c10::optional<torch::Tensor> req_acc_state,
    c10::optional<torch::Tensor> saved_leaf_parent) {
  TORCH_CHECK(tree_m >= 8 && tree_m <= 32, "tree_m must be in 8..32");
  TORCH_CHECK(leaf_flag.is_cuda() && leaf_flag.scalar_type() == torch::kInt32);
  TORCH_CHECK(prev_ns.is_cuda() && prev_ns.scalar_type() == torch::kInt32);
  TORCH_CHECK(hit_flag.is_cuda() && hit_flag.scalar_type() == torch::kInt32 && hit_flag.numel() >= 1);
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
  const int32_t* parent_ptr = nullptr;
  if (saved_leaf_parent.has_value() && saved_leaf_parent.value().defined() &&
      saved_leaf_parent.value().numel() >= tree_m) {
    TORCH_CHECK(saved_leaf_parent.value().is_cuda() && saved_leaf_parent.value().scalar_type() == torch::kInt32);
    parent_ptr = saved_leaf_parent.value().data_ptr<int32_t>();
  }
  static const float high_acc_cpc_env = []() -> float {
    const char* s = std::getenv("MUSE_TREE_HIGH_ACC_CPC");
    if (!s || !*s) return 0.0f;
    const double v = std::strtod(s, nullptr);
    return (v > 0.0) ? static_cast<float>(v) : 0.0f;
  }();
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  native_leaf_commit_tree_kernel<<<1, 1, 0, st>>>(
      leaf_flag.data_ptr<int32_t>(),
      prev_ns.data_ptr<int32_t>(),
      hit_flag.data_ptr<int32_t>(),
      static_cast<int>(hit_flag.numel()),
      hit_src_row.data_ptr<int32_t>(),
      saved_leaf_toks.data_ptr<int64_t>(),
      saved_leaf_k.data_ptr<int32_t>(),
      parent_ptr,
      leaf_u_toks.data_ptr<int64_t>(),
      sampled_tokens.data_ptr<int64_t>(),
      num_sampled.data_ptr<int32_t>(),
      num_rejected.data_ptr<int32_t>(),
      stats_ptr,
      stats_len,
      acc_ptr,
      static_cast<int>(tree_m),
      no_hit_mode ? 1 : 0,
      high_acc_cpc_env);
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
      static_cast<int>(hit_flag.numel()),
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
  TORCH_CHECK(hit_flag.is_cuda() && hit_flag.scalar_type() == torch::kInt32 && hit_flag.numel() >= 1);
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
      static_cast<int>(hit_flag.numel()),
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
  TORCH_CHECK(hit_flag.is_cuda() && hit_flag.scalar_type() == torch::kInt32 && hit_flag.numel() >= 1);
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
      static_cast<int>(hit_flag.numel()),
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
