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
__constant__ float kCalC[6] = {0.000000f, -0.402942f, -0.763639f, -0.817452f, -1.416669f, -1.642223f};
#define CAL_A2 (-1.614419f)
#define CAL_BETA 0.851144f
#define CAL_BETA_POST 0.326972f
#define CAL_W_SM_PRE 0.472339f
#define CAL_W_Z0_PRE 0.201693f
#define CAL_W_Z0_POST 0.178773f
#define CAL_SHIFT 0.0349f

struct CalibKnobs {
  float a_shift;
  float a2;
  float beta;
  float beta_post;
  float w_sm_pre;
  float w_z0_pre;
  float w_z0_post;
  float z0_center;
  float shf;
  float alt2_scale;
};

constexpr int kPlWindowMax = 4096;

struct PlKnobs {
  int mode;             // 0=off, 1=enabled
  int min_n;            // min suffix match len for standalone OOV copy root at j=0
  int mid_min_n;        // min suffix match len for standalone OOV copy root at j>0
  int miss_gate;        // 0=ungated, 1=multiply by (1 - s_p1[j]), 2=multiply by p_oov[j], 3=max(p_oov[j], oov_floor)
  int in_top6_floor;    // 1=allow calibrated (1 - s_p1[j]) * t6_root_p floor for in-top-6 copy candidates
  int cont_from_df_n;   // min match len to spawn pl_cont child from a DFlash candidate
  int max_dp_4;         // max copy-chain depth for 4-gram root (root + continuations)
  int max_dp_5;         // max copy-chain depth for 5-gram root
  int max_dp_6;         // max copy-chain depth for 6+-gram root
  float mid_scale;      // multiplier for j>0 standalone OOV copy roots
  float oov_floor;      // min gate floor for miss_gate==3 (prevents overconfident DFlash crushing OOV 4-gram+ copies)
  float root_p[7];      // OOV copy root score weight for match len 0..6
  float t6_root_p[7];   // in-top-6 copy root score floor weight for match len 0..6
  float cont_p[7];      // copy continuation probability for match len 0..6
  float boost_alt[7];   // additive log-odds boost to DFlash top-6 alt (r=1..5) for match len 0..6
  float boost_d1[7];    // additive log-odds boost to DFlash top-1 (r=0) for match len 0..6
};

struct BgEbKnobs {
  // Smoothed Markov prompt-history bigram/trigram coherence + branch-conditioned unique OOV copy
  int bg_mode;          // 0=off, 1=enabled
  int bg_sub;           // 1=also check bigram coherence on resynced kind==1 (sub) with stricter sub_min_p
  int bg_off;           // 1=chain bigram coherence & unique-OOV across off-chain depth>=2 (kind==3,6)
  int chp_bg2;          // 1=attach history hpos on strong repeated bigram (cnt>=2 && p_bg>=0.45) in addition to trigram
  float bg_alpha;       // smoothing pseudocount for p_bg = bg_cnt / (alt_cnt + alpha)
  float bg_min_p;       // min p_bg for bigram boost on off-chain parents
  float sub_min_p;      // stricter min p_bg for kind==1 (sub) parents
  float tri_min_p;      // min p_bg to trust trigram [anc, alt, r'] boost
  float f2;             // base bigram log-odds boost
  float w2;             // slope * p_bg bigram log-odds boost
  float f3;             // base trigram log-odds boost
  float w3;             // slope * p_bg trigram log-odds boost
  // Branch-conditioned unique OOV successor (cnt>=2 deterministic next token in s_h)
  int uniq_mode;        // 0=off, 1=enabled
  float uniq_p2;        // score multiplier when cnt == 2
  float uniq_p3;        // score multiplier when cnt >= 3
  // Shift-bigram calibration for shf (kind=2)
  int shf_cal;          // 0=off (use ck.shf), 1=use shf_hi if (alt, dk[j]) in history else shf_lo
  float shf_lo;
  float shf_hi;
  // Downward-bounded online Empirical-Bayes a_shift from req_acc_state [rounds, committed]
  int eb_mode;          // 0=off, 1=enabled
  int eb_min_r;
  float eb_p0;
  float eb_tau;
  float eb_scale;
  float eb_lo;
  float eb_hi;
};

constexpr int kTrActiveCap = 512;

struct TrKnobs {
  // Wave-63 GPU Target-Logit Token Recycling (16-row exact 30B argmax log + monomorphic suffix roots/continuations)
  int mode;              // 0=off, 1=enabled
  int cont_mode;         // 1=spawn recycled 30B continuations under selected leaves
  int det_sh_root;       // 1=enable monomorphic trigram(cnt>=2)/bigram(cnt>=3) OOV copy roots from s_h
  int max_tr_dp;         // max tree depth for recycled continuations
  float b_t6_tri;        // log-odds boost to DFlash top-6 alt on recycled trigram match
  float b_t6_bg;         // log-odds boost to DFlash top-6 alt on recycled bigram match
  float b_t6_near;       // log-odds boost to DFlash top-6 alt on rare near-frontier recycled unigram
  float oov_tri_p;       // standalone OOV root score weight on monomorphic (cnt>=2) recycled trigram
  float oov_a1_bg_p;     // standalone OOV root score weight on fresh age==1 near-pos recycled bigram
  float oov_a1_pos2_p;   // standalone OOV root score weight on unanimous (cnt>=2) age==1 near-pos recycled unigram
  float oov_fresh_p;     // standalone OOV root score weight on fresh (anc_u1==0) near-frontier recycled unigram
  float det_tri3_root_p; // standalone OOV root score weight on monomorphic (cnt>=2) s_h trigram
  float det_bg2_root_p;  // standalone OOV root score weight on monomorphic (cnt>=3) s_h bigram
  float ch_bg_t6;        // child continuation prob on monomorphic (cnt>=2) tier-0 bigram in DFlash top-6
  float ch_bg_oov;       // child continuation prob on monomorphic (cnt>=2) tier-0 OOV bigram
  float ch_bg1_t6;       // child continuation prob on conflict-free (cnt>=1) tier-0 bigram + DFlash top-6 agreement
  float ch_near_t6;      // child continuation prob on conflict-free near-frontier recycled unigram in DFlash top-6
  float ch_near_oov;     // child continuation prob on conflict-free near-frontier OOV recycled unigram
  float ch_t6_agree;     // child continuation prob when monomorphic (cnt>=2, tier0>=1) 30B & DFlash top-6 agree
};

// Arbitrary-depth (1..6-level) dynamic multi-leaf tree setup (rows 7..tree_m-1), launched as <<<1, 256>>> (~3.5 us):
//   - Bit-packed remap inside int32[49] (1 + 16*3):
//       leaf_flag[1 + 3*r + 0] = seq_sub  (seqlen_k_eff = seqlen_k - seq_sub)
//       leaf_flag[1 + 3*r + 1] = base_sub (-1 for causal rows; else tree_m - k0 of root ancestor)
//       leaf_flag[1 + 3*r + 2] = remap_packed (nibble j = physical row r_j in 0..15 for tok = p + k0 + j, j=0..5)
//   - Fused SMEM history scan (kPlWindowMax=4096) over committed token history `hist[0..p0]` +
//     per-request 30B Target Recycling Log `tr_log`:
//       1) n-gram + 30B target-recycle confirmation/boosting of DFlash top-6 candidates (r=0..5) at depths j=0..5,
//       2) standalone OOV prompt-lookup & monomorphic 30B recycled roots + multi-step continuation chains.
//   - selftest_mode!=0: places a full deep chain (6-level on M=16, 5-level on M=12) mirroring d1..d6.
__global__ void __launch_bounds__(256) native_leaf_setup_tree_kernel(
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
    int calib_mode,
    CalibKnobs ck,
    const int32_t* __restrict__ hist,
    const int32_t* __restrict__ hist_len,
    int hist_cap,
    int64_t* __restrict__ saved_pos0,
    int64_t* __restrict__ stats,
    int stats_len,
    PlKnobs pl,
    BgEbKnobs bg,
    TrKnobs tr,
    int32_t* __restrict__ saved_row_ctx,
    const int32_t* __restrict__ tr_log,
    int32_t* __restrict__ tr_state,
    int tr_cap) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (tid == 0 && saved_pos0 != nullptr) {
    saved_pos0[0] = positions[0];
  }
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
      int row_base_sub[16];
      int row_depth[16];
      uint32_t row_packed[16];
      bool any_active = false;
      for (int li = 0; li < num_leaves; ++li) {
        const int r = 7 + li;
        int k = kLeafK[li];
        int par = -1;
        int64_t cand = -1;
        bool valid = false;
        if (!high_acc) {
          if (selftest_mode != 0) {
            // Deep chain selftest: rows 7..12 mirror d1..d6 (depth 1..6), rows 13..15 mirror d1..d3 (depth 1).
            if (li < 6) {
              k = li + 1;
              par = (li == 0) ? -1 : (r - 1);
            } else {
              k = (li - 6) + 1;
              par = -1;
            }
            cand = static_cast<int64_t>(input_ids[k]);
            valid = true;
          } else {
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
            pkd = static_cast<uint32_t>(r) & 0xFu;
          } else {
            b_sub = row_base_sub[par];
            dp = row_depth[par] + 1;
            pkd = row_packed[par] | ((static_cast<uint32_t>(r) & 0xFu) << (4 * (dp - 1)));
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
      if (saved_row_ctx != nullptr) {
        saved_row_ctx[3] = -1;
      }
    }
    return;
  }

  // Dynamic per-step GPU arbitrary-depth (1..6-level) tree allocator + fused prompt lookup (up to 96 candidates).
  __shared__ int32_t s_h[kPlWindowMax];
  __shared__ int32_t s_ctx[22];
  __shared__ int32_t s_top[6][6];
  __shared__ uint32_t s_w_pl_key[8][6];
  __shared__ uint32_t s_w_cand_key[8][6][6];
  __shared__ uint32_t s_pl_key[6];
  __shared__ uint32_t s_cand_key[6][6];
  // Wave-62 smoothed Markov bigram/trigram coherence & zero-register atomic unique-successor tables (~2.1 KiB SMEM).
  __shared__ uint32_t s_alt_cnt[5][6];
  __shared__ uint32_t s_bg_cnt[5][6][6];
  __shared__ uint32_t s_bg3_mask[5][6];
  __shared__ uint32_t s_bg_key[5][6][6];
  __shared__ uint32_t s_shf_bg[5][5];
  __shared__ int32_t s_uniq_min[5][5];
  __shared__ int32_t s_uniq_max[5][5];
  __shared__ int32_t s_uniq_hp[5][5];
  // Wave-63 anchor monomorphism & L2-resident 30B Target Recycling active buffer (~12.8 KiB SMEM).
  __shared__ uint32_t s_anc_u1_cnt[6];
  __shared__ uint32_t s_anc_bg2_cnt[6];
  __shared__ int32_t s_anc_bg2_min[6];
  __shared__ int32_t s_anc_bg2_max[6];
  __shared__ int32_t s_anc_bg2_hp[6];
  __shared__ uint32_t s_anc_tri3_cnt[6];
  __shared__ int32_t s_anc_tri3_min[6];
  __shared__ int32_t s_anc_tri3_max[6];
  __shared__ int32_t s_anc_tri3_hp[6];
  __shared__ int s_tr_active_n;
  __shared__ int s_tr_cur_step;
  __shared__ int32_t s_tr_cur[kTrActiveCap];
  __shared__ int32_t s_tr_p1[kTrActiveCap];
  __shared__ int32_t s_tr_p2[kTrActiveCap];
  __shared__ int32_t s_tr_u[kTrActiveCap];
  __shared__ int32_t s_tr_st[kTrActiveCap];
  __shared__ int32_t s_tr_pos_tier[kTrActiveCap];
  __shared__ float s_tr_b_rk[6][6];
  __shared__ int32_t s_tr_oov_tok[6];
  __shared__ float s_tr_oov_w[6];
  __shared__ int32_t s_leaf_tok_r[16];
  __shared__ int32_t s_leaf_prv_r[16];
  __shared__ int32_t s_leaf_prv2_r[16];
  __shared__ int64_t s_tr_ch_tok;
  __shared__ float s_tr_ch_p;
  __shared__ int s_hist_ok;
  __shared__ int s_win;
  __shared__ int s_h_base;

  __shared__ float s_Z[6];
  __shared__ float s_A[6];
  __shared__ float s_p1[6];
  __shared__ float s_p_oov[6];
  __shared__ float s_p1post[6];
  __shared__ float s_p2post[6];
  __shared__ float s_pr[6][6];
  __shared__ float s_e1[6][6];
  __shared__ float s_den1[6];
  __shared__ float s_z0[6];
  __shared__ float s_margin[6];
  __shared__ int64_t s_dk[6];
  __shared__ uint8_t s_valid_row[6];
  __shared__ int s_any_active;

  __shared__ float s_cand_score[96];
  __shared__ int64_t s_cand_tok[96];
  __shared__ int8_t s_cand_k[96];
  __shared__ int8_t s_cand_par[96];
  __shared__ int8_t s_cand_depth[96];
  __shared__ int8_t s_cand_kind[96];  // 0=1-level, 1=sub, 2=shift, 3=alt2, 4=pl_root, 5=pl_cont, 6=coh_cont, 7=tr
  __shared__ int16_t s_cand_hpos[96]; // index in s_h matching s_cand_tok (-1 if none)
  __shared__ int8_t s_cand_ml[96];    // history match length 0..16
  __shared__ int8_t s_cand_rk[96];    // rank 0..5 in row (k-1), or -1 if non-top6
  __shared__ int s_num_cands;

  __shared__ int s_row_base_sub[16];
  __shared__ uint32_t s_row_packed[16];

  // Pass 1: validate rows, gather context tokens & top-6 IDs, and verify fail-closed history + recycle sync.
  if (tid == 0) {
    const int64_t p0 = positions[0];
    const bool h_ok = (!high_acc && hist != nullptr && hist_len != nullptr &&
                       p0 >= 1 && p0 < hist_cap &&
                       hist_len[0] == static_cast<int32_t>(p0 + 1) &&
                       hist[p0] == input_ids[0]);
    s_hist_ok = h_ok ? 1 : 0;
    if (h_ok) {
      const int total_h = static_cast<int>(p0 + 1);
      const int win = (total_h < kPlWindowMax) ? total_h : kPlWindowMax;
      s_win = win;
      s_h_base = total_h - win;
    } else {
      s_win = 0;
      s_h_base = 0;
    }
    s_tr_active_n = 0;
    int cur_tr_step = 0;
    if (tr.mode != 0 && tr_log != nullptr && tr_state != nullptr && tr_cap > 0) {
      if (!h_ok || tr_state[0] < 0 || (tr_state[0] > 0 && p0 <= static_cast<int64_t>(tr_state[2]))) {
        // Fail-closed zero-leak invariant: reset immediately if history invalid or p0 did not advance.
        tr_state[0] = 0;
        tr_state[1] = 0;
        tr_state[2] = -1;
      }
      cur_tr_step = tr_state[1];
    }
    s_tr_cur_step = cur_tr_step;
    if (stats != nullptr && stats_len >= 34 && hist != nullptr && hist_len != nullptr && !high_acc) {
      if (h_ok) stats[32] += 1;
      else stats[33] += 1;
    }
  }
  if (tid < 15) {
    s_ctx[tid] = -2;
  }
  if (tid < 16) {
    s_leaf_tok_r[tid] = -1;
    s_leaf_prv_r[tid] = -1;
    s_leaf_prv2_r[tid] = -1;
  }
  if (tid < 6) {
    const int k = tid + 1;
    const int32_t dk32 = input_ids[k];
    const int64_t dk = static_cast<int64_t>(dk32);
    s_dk[tid] = dk;
    s_ctx[16 + tid] = dk32;
    const int base = tid * tree_k;
    const int64_t top1 = tree_top[base + 0];
    const float z0 = tree_vals[base + 0];
    const bool row_ok = (!high_acc) && (top1 == dk) && (z0 > -1e20f);
    s_valid_row[tid] = row_ok ? 1 : 0;
    s_z0[tid] = z0;
    const int64_t top2 = tree_top[base + 1];
    const float z1 = tree_vals[base + 1];
    s_margin[tid] = (row_ok && top2 >= 0 && z1 > -1e20f) ? (z0 - z1) : 2.5f;
    s_anc_u1_cnt[tid] = 0u;
    s_anc_bg2_cnt[tid] = 0u;
    s_anc_bg2_min[tid] = 2147483647;
    s_anc_bg2_max[tid] = -1;
    s_anc_bg2_hp[tid] = -1;
    s_anc_tri3_cnt[tid] = 0u;
    s_anc_tri3_min[tid] = 2147483647;
    s_anc_tri3_max[tid] = -1;
    s_anc_tri3_hp[tid] = -1;
    s_tr_oov_tok[tid] = -1;
    s_tr_oov_w[tid] = 0.0f;
#pragma unroll
    for (int r = 0; r < 6; ++r) {
      const int64_t tv = tree_top[base + r];
      s_top[tid][r] = (tv >= 0) ? static_cast<int32_t>(tv) : -1;
      s_tr_b_rk[tid][r] = 0.0f;
    }
  } else if (tid == 6) {
    s_ctx[15] = input_ids[0];
  }
  if (tid < 180) {
    const int j = tid / 36;
    const int rem = tid - j * 36;
    const int r = rem / 6;
    const int rp = rem - r * 6;
    s_bg_cnt[j][r][rp] = 0u;
    s_bg_key[j][r][rp] = 0u;
  }
  if (tid < 30) {
    const int j = tid / 6;
    const int r = tid - j * 6;
    s_alt_cnt[j][r] = 0u;
    s_bg3_mask[j][r] = 0u;
  }
  if (tid < 25) {
    const int j = tid / 5;
    const int rm = tid - j * 5;
    s_shf_bg[j][rm] = 0u;
    s_uniq_min[j][rm] = 2147483647;
    s_uniq_max[j][rm] = -1;
    s_uniq_hp[j][rm] = -1;
  }
  __syncthreads();

  // Pass 2: collaborative 256-thread SMEM scan of history window for multi-depth suffix matches,
  // Markov bigram/trigram co-occurrence counts, shift-bigram flags, unique OOV successors, and 30B recycle log.
  const bool scan_bg = (bg.bg_mode != 0 || bg.uniq_mode != 0 || bg.shf_cal != 0);
  const bool scan_anc = (pl.mode != 0 || tr.mode != 0 || tr.det_sh_root != 0);
  if ((scan_anc || scan_bg) && s_hist_ok != 0 && s_win >= 2) {
    const int win = s_win;
    const int h_base = s_h_base;
    for (int i = tid; i < win; i += 256) {
      s_h[i] = hist[h_base + i];
    }
    __syncthreads();
    if (tid < 15) {
      const int hi = win - 16 + tid;
      s_ctx[tid] = (hi >= 0) ? s_h[hi] : -2;
    }
    __syncthreads();

    uint32_t pl_k[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    uint32_t u1_c[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    uint32_t c_k[6][6];
#pragma unroll
    for (int j = 0; j < 6; ++j) {
#pragma unroll
      for (int r = 0; r < 6; ++r) {
        c_k[j][r] = 0u;
      }
    }

    const int32_t anchors[6] = {s_ctx[15], s_ctx[16], s_ctx[17], s_ctx[18], s_ctx[19], s_ctx[20]};
    for (int i = tid; i < win - 1; i += 256) {
      const int32_t h_i = s_h[i];
      const int32_t nxt = s_h[i + 1];
      if (nxt < 0) continue;
      if (scan_anc) {
#pragma unroll
        for (int j = 0; j < 6; ++j) {
          if (h_i == anchors[j]) {
            ++u1_c[j];
            int ml = 1;
            while (ml < 16 && i - ml >= 0 && s_h[i - ml] == s_ctx[15 + j - ml]) {
              ++ml;
            }
            if (tr.mode != 0 || tr.det_sh_root != 0) {
              if (ml >= 2) {
                atomicAdd(&s_anc_bg2_cnt[j], 1u);
                atomicMin(&s_anc_bg2_min[j], nxt);
                atomicMax(&s_anc_bg2_max[j], nxt);
                atomicMax(&s_anc_bg2_hp[j], i + 1);
              }
              if (ml >= 3) {
                atomicAdd(&s_anc_tri3_cnt[j], 1u);
                atomicMin(&s_anc_tri3_min[j], nxt);
                atomicMax(&s_anc_tri3_max[j], nxt);
                atomicMax(&s_anc_tri3_hp[j], i + 1);
              }
            }
            if (pl.mode != 0) {
              const uint32_t key = (static_cast<uint32_t>(ml) << 16) | static_cast<uint32_t>(i);
              if (nxt != s_ctx[16 + j] && key > pl_k[j]) {
                pl_k[j] = key;
              }
#pragma unroll
              for (int r = 0; r < 6; ++r) {
                if (nxt == s_top[j][r] && key > c_k[j][r]) {
                  c_k[j][r] = key;
                }
              }
            }
          }
        }
      }
      if (scan_bg) {
        const int r_start = (bg.bg_sub != 0 && bg.bg_mode != 0) ? 0 : 1;
#pragma unroll
        for (int j = 0; j < 5; ++j) {
          const int32_t anc_j = anchors[j];
          const int32_t dk_j = s_ctx[16 + j];
          for (int r = r_start; r < 6; ++r) {
            const int32_t tok_r = s_top[j][r];
            if (tok_r >= 0 && h_i == tok_r) {
              atomicAdd(&s_alt_cnt[j][r], 1u);
              if (r >= 1) {
                if (bg.uniq_mode != 0) {
                  atomicMin(&s_uniq_min[j][r - 1], nxt);
                  atomicMax(&s_uniq_max[j][r - 1], nxt);
                  atomicMax(&s_uniq_hp[j][r - 1], i + 1);
                }
                if (bg.shf_cal != 0 && nxt == dk_j) {
                  s_shf_bg[j][r - 1] = 1u;
                }
              }
              if (bg.bg_mode != 0) {
                bool tri_checked = false;
                int ml_b = 1;
#pragma unroll
                for (int rp = 0; rp < 6; ++rp) {
                  if (nxt == s_top[j + 1][rp]) {
                    atomicAdd(&s_bg_cnt[j][r][rp], 1u);
                    if (!tri_checked) {
                      tri_checked = true;
                      if (i >= 1 && s_h[i - 1] == anc_j) {
                        ml_b = 2;
                        while (ml_b < 16 && i - ml_b >= 0 &&
                               s_h[i - ml_b] == s_ctx[15 + j - (ml_b - 1)]) {
                          ++ml_b;
                        }
                      }
                    }
                    if (ml_b >= 2) {
                      atomicOr(&s_bg3_mask[j][r], 1u << rp);
                    }
                    const uint32_t kv = (static_cast<uint32_t>(ml_b) << 16) | static_cast<uint32_t>(i + 1);
                    atomicMax(&s_bg_key[j][r][rp], kv);
                  }
                }
              }
            }
          }
        }
      }
    }

    // Pass 2b: collaborative 256-thread anti-echo compaction of active 30B recycle log entries into SMEM.
    if (tr.mode != 0 && tr_log != nullptr && tr_state != nullptr && tr_cap > 0) {
      int n_log = tr_state[0];
      if (n_log > tr_cap) n_log = tr_cap;
      const int cur_st = s_tr_cur_step;
      for (int idx = n_log - 1 - tid; idx >= 0; idx -= 256) {
        const int base8 = idx * 8;
        const int32_t st_t = tr_log[base8 + 5];
        const int32_t tier_t = tr_log[base8 + 7];
        if (tier_t == 0 || (cur_st - st_t) == 1) {
          const int dst = atomicAdd(&s_tr_active_n, 1);
          if (dst < kTrActiveCap) {
            s_tr_cur[dst] = tr_log[base8 + 1];
            s_tr_p1[dst] = tr_log[base8 + 2];
            s_tr_p2[dst] = tr_log[base8 + 3];
            s_tr_u[dst] = tr_log[base8 + 4];
            s_tr_st[dst] = st_t;
            s_tr_pos_tier[dst] = (tr_log[base8 + 6] << 1) | (tier_t & 1);
          }
        }
      }
    }

#pragma unroll
    for (int j = 0; j < 6; ++j) {
      const uint32_t wm = __reduce_max_sync(0xffffffffu, pl_k[j]);
      const uint32_t wu = __reduce_add_sync(0xffffffffu, u1_c[j]);
      if (lane == 0) {
        s_w_pl_key[warp][j] = wm;
        if (wu > 0u) atomicAdd(&s_anc_u1_cnt[j], wu);
      }
#pragma unroll
      for (int r = 0; r < 6; ++r) {
        const uint32_t cwm = __reduce_max_sync(0xffffffffu, c_k[j][r]);
        if (lane == 0) s_w_cand_key[warp][j][r] = cwm;
      }
    }
    __syncthreads();

    if (warp != 0) return;
    if (lane == 0 && s_tr_active_n > kTrActiveCap) {
      s_tr_active_n = kTrActiveCap;
    }
    if (lane < 8) {
#pragma unroll
      for (int j = 0; j < 6; ++j) {
        const uint32_t fm = __reduce_max_sync(0xffu, s_w_pl_key[lane][j]);
        if (lane == 0) s_pl_key[j] = fm;
#pragma unroll
        for (int r = 0; r < 6; ++r) {
          const uint32_t cfm = __reduce_max_sync(0xffu, s_w_cand_key[lane][j][r]);
          if (lane == 0) s_cand_key[j][r] = cfm;
        }
      }
    }
    __syncwarp();
  } else {
    if (warp != 0) return;
    if (lane < 6) {
      s_pl_key[lane] = 0u;
#pragma unroll
      for (int r = 0; r < 6; ++r) {
        s_cand_key[lane][r] = 0u;
      }
    }
    __syncwarp();
  }

  // Pass 2c (warp 0): hardware Redux reduction of 30B Target Recycling features across anchors j=0..5 (~0.5 us).
  if (calib_mode != 0 && tr.mode != 0 && s_hist_ok != 0 && s_tr_active_n > 0) {
    const int n_act = s_tr_active_n;
    const int cur_st = s_tr_cur_step;
    const int p0_i32 = static_cast<int>(positions[0]);
    for (int j = 0; j < 6; ++j) {
      if (!s_valid_row[j]) continue;
      const int32_t cur = s_ctx[15 + j];
      const int32_t prv1 = s_ctx[14 + j];
      const int32_t prv2 = s_ctx[13 + j];
      const int pos_j = p0_i32 + j;

      int u_cnt = 0, u_min = 2147483647, u_max = -1;
      int bg_c = 0, bg_min = 2147483647, bg_max = -1;
      int tri_c = 0, tri_min = 2147483647, tri_max = -1;
      int near_c = 0, near_min = 2147483647, near_max = -1;
      int a1_pos_c = 0, a1_pos_min = 2147483647, a1_pos_max = -1;
      int a1_bg_c = 0, a1_bg_min = 2147483647, a1_bg_max = -1;
      int tot_a1 = 0, a1_v1 = 0, a1_v2 = 0, a1_v3 = 0, a1_v4 = 0, a1_v5 = 0;
      const int32_t t1 = s_top[j][1], t2 = s_top[j][2], t3 = s_top[j][3], t4 = s_top[j][4], t5 = s_top[j][5];

      for (int idx = lane; idx < n_act; idx += 32) {
        if (s_tr_cur[idx] != cur) continue;
        const int32_t u_t = s_tr_u[idx];
        const int32_t p1_t = s_tr_p1[idx];
        const int32_t p2_t = s_tr_p2[idx];
        const int age = cur_st - s_tr_st[idx];
        const int pt = s_tr_pos_tier[idx];
        const int pos_t = pt >> 1;
        const int tier_t = pt & 1;
        const int dist = (pos_j >= pos_t) ? (pos_j - pos_t) : (pos_t - pos_j);

        u_cnt += 1;
        if (u_t < u_min) u_min = u_t;
        if (u_t > u_max) u_max = u_t;

        if (age == 1 && dist <= 8) {
          const int w = (tier_t == 0) ? 2 : 1;
          tot_a1 += w;
          if (u_t == t1) a1_v1 += w;
          if (u_t == t2) a1_v2 += w;
          if (u_t == t3) a1_v3 += w;
          if (u_t == t4) a1_v4 += w;
          if (u_t == t5) a1_v5 += w;
        }
        if (age == 1 && dist <= 3 && u_t != cur) {
          a1_pos_c += 1;
          if (u_t < a1_pos_min) a1_pos_min = u_t;
          if (u_t > a1_pos_max) a1_pos_max = u_t;
          if (prv1 >= 0 && p1_t == prv1) {
            a1_bg_c += 1;
            if (u_t < a1_bg_min) a1_bg_min = u_t;
            if (u_t > a1_bg_max) a1_bg_max = u_t;
          }
        }
        if (age <= 2 && dist <= 12) {
          near_c += 1;
          if (u_t < near_min) near_min = u_t;
          if (u_t > near_max) near_max = u_t;
        }
        if (prv1 >= 0 && p1_t == prv1) {
          bg_c += 1;
          if (u_t < bg_min) bg_min = u_t;
          if (u_t > bg_max) bg_max = u_t;
          if (prv2 >= 0 && p2_t == prv2) {
            tri_c += 1;
            if (u_t < tri_min) tri_min = u_t;
            if (u_t > tri_max) tri_max = u_t;
          }
        }
      }

      u_cnt = __reduce_add_sync(0xffffffffu, u_cnt);
      if (u_cnt > 0) {
        u_min = __reduce_min_sync(0xffffffffu, u_min);
        u_max = __reduce_max_sync(0xffffffffu, u_max);
        bg_c = __reduce_add_sync(0xffffffffu, bg_c);
        bg_min = __reduce_min_sync(0xffffffffu, bg_min);
        bg_max = __reduce_max_sync(0xffffffffu, bg_max);
        tri_c = __reduce_add_sync(0xffffffffu, tri_c);
        tri_min = __reduce_min_sync(0xffffffffu, tri_min);
        tri_max = __reduce_max_sync(0xffffffffu, tri_max);
        near_c = __reduce_add_sync(0xffffffffu, near_c);
        near_min = __reduce_min_sync(0xffffffffu, near_min);
        near_max = __reduce_max_sync(0xffffffffu, near_max);
        a1_pos_c = __reduce_add_sync(0xffffffffu, a1_pos_c);
        a1_pos_min = __reduce_min_sync(0xffffffffu, a1_pos_min);
        a1_pos_max = __reduce_max_sync(0xffffffffu, a1_pos_max);
        a1_bg_c = __reduce_add_sync(0xffffffffu, a1_bg_c);
        a1_bg_min = __reduce_min_sync(0xffffffffu, a1_bg_min);
        a1_bg_max = __reduce_max_sync(0xffffffffu, a1_bg_max);
        tot_a1 = __reduce_add_sync(0xffffffffu, tot_a1);
        a1_v1 = __reduce_add_sync(0xffffffffu, a1_v1);
        a1_v2 = __reduce_add_sync(0xffffffffu, a1_v2);
        a1_v3 = __reduce_add_sync(0xffffffffu, a1_v3);
        a1_v4 = __reduce_add_sync(0xffffffffu, a1_v4);
        a1_v5 = __reduce_add_sync(0xffffffffu, a1_v5);

        if (lane == 0) {
          const int a1_v[6] = {0, a1_v1, a1_v2, a1_v3, a1_v4, a1_v5};
          const uint32_t anc_u1 = s_anc_u1_cnt[j];
          for (int rk = 1; rk < 6; ++rk) {
            const int32_t cand = s_top[j][rk];
            if (cand < 0) continue;
            float b = 0.0f;
            if (tri_c >= 1 && tri_min == tri_max && tri_min == cand && tr.b_t6_tri > b) b = tr.b_t6_tri;
            if (bg_c >= 1 && bg_min == bg_max && bg_min == cand && tr.b_t6_bg > b) b = tr.b_t6_bg;
            if (near_c >= 1 && near_min == near_max && near_min == cand && u_min == u_max && anc_u1 <= 3u) {
              if (tr.b_t6_near > b) b = tr.b_t6_near;
            } else if (tot_a1 >= 2 && a1_v[rk] * 3 >= tot_a1 * 2 && anc_u1 <= 3u) {
              if (tr.b_t6_near > b) b = tr.b_t6_near;
            }
            s_tr_b_rk[j][rk] = b;
          }
          const int32_t dk32 = static_cast<int32_t>(s_dk[j]);
          if (tri_c >= 2 && tri_min == tri_max && tri_min != dk32 && tr.oov_tri_p > 0.0f) {
            s_tr_oov_tok[j] = tri_min;
            s_tr_oov_w[j] = tr.oov_tri_p;
          } else if (a1_bg_c >= 1 && a1_bg_min == a1_bg_max && a1_bg_min != dk32 &&
                     s_anc_bg2_cnt[j] == 0u && tr.oov_a1_bg_p > 0.0f) {
            s_tr_oov_tok[j] = a1_bg_min;
            s_tr_oov_w[j] = tr.oov_a1_bg_p;
          } else if (a1_pos_c >= 2 && a1_pos_min == a1_pos_max && a1_pos_min != dk32 &&
                     anc_u1 <= 4u && tr.oov_a1_pos2_p > 0.0f) {
            s_tr_oov_tok[j] = a1_pos_min;
            s_tr_oov_w[j] = tr.oov_a1_pos2_p;
          } else if (near_c >= 1 && near_min == near_max && u_min == u_max && anc_u1 == 0u &&
                     near_min != cur && near_min != dk32 && tr.oov_fresh_p > 0.0f) {
            s_tr_oov_tok[j] = near_min;
            s_tr_oov_w[j] = tr.oov_fresh_p;
          }
        }
      }
    }
    __syncwarp();
  }

  // Pass 3 (warp 0 only): calibrated probabilities + online Empirical-Bayes + n-gram & Markov bigram 9-leaf EV tree.
  if (tid < 6) {
    const int base = tid * tree_k;
    const float z0 = s_z0[tid];
    const bool row_ok = (s_valid_row[tid] != 0);
    if (row_ok) {
      if (calib_mode != 0) {
        float eb_delta_a = 0.0f;
        if (bg.eb_mode != 0 && req_acc_state != nullptr) {
          const int32_t rr = req_acc_state[0];
          const int32_t cc = req_acc_state[1];
          if (rr >= bg.eb_min_r && cc > 0) {
            const float mean_ns = static_cast<float>(cc) / static_cast<float>(rr);
            const float den_m = (mean_ns - 0.35f > 0.25f) ? (mean_ns - 0.35f) : 0.25f;
            float p_hat = (mean_ns - 1.35f) / den_m;
            if (p_hat < 0.35f) p_hat = 0.35f;
            if (p_hat > 0.92f) p_hat = 0.92f;
            const float p0 = bg.eb_p0;
            const float p_post = (static_cast<float>(rr) * p_hat + bg.eb_tau * p0) /
                                 (static_cast<float>(rr) + bg.eb_tau);
            const float da = (__logf(p_post / (1.0f - p_post)) - __logf(p0 / (1.0f - p0))) * bg.eb_scale;
            eb_delta_a = (da < bg.eb_lo) ? bg.eb_lo : ((da > bg.eb_hi) ? bg.eb_hi : da);
          }
        }
        float sm_c = (s_margin[0] + s_margin[1] + s_margin[2] +
                      s_margin[3] + s_margin[4] + s_margin[5]) * (1.0f / 6.0f) - 2.5f;
        if (sm_c < -2.0f) sm_c = -2.0f;
        if (sm_c > 2.5f) sm_c = 2.5f;
        float z0_c = z0 - ck.z0_center;
        if (z0_c < -10.0f) z0_c = -10.0f;
        if (z0_c > 6.0f) z0_c = 6.0f;
        const float a_base = kCalA[tid] + ck.a_shift + eb_delta_a;
        const float a_pre = a_base + ck.w_sm_pre * sm_c + ck.w_z0_pre * z0_c;
        const float a_post = a_base + ck.a2 + ck.w_z0_post * z0_c;
        float e0[6], e1[6];
        float den0 = 1.0f, den1 = 1.0f;
#pragma unroll
        for (int rk = 0; rk < 6; ++rk) {
          const int64_t tok_r = tree_top[base + rk];
          const float zr = tree_vals[base + rk];
          if (rk == 0 || (tok_r >= 0 && zr > -1e20f)) {
            const float dz = zr - z0;
            float b_pl = 0.0f;
            if (pl.mode != 0 && s_hist_ok != 0) {
              const int raw_ml = static_cast<int>(s_cand_key[tid][rk] >> 16);
              if (raw_ml >= 1) {
                const int ml = (raw_ml > 6) ? 6 : raw_ml;
                b_pl = (rk == 0) ? pl.boost_d1[ml] : pl.boost_alt[ml];
              }
            }
            const float b_tr = s_tr_b_rk[tid][rk];
            const float b_eff = (b_pl > b_tr) ? b_pl : b_tr;
            const float g0 = kCalC[rk] + ck.beta * dz + b_eff;
            const float g1 = kCalC[rk] + ck.beta_post * dz;
            e0[rk] = __expf(a_pre + g0);
            e1[rk] = __expf(a_post + g1);
          } else {
            e0[rk] = 0.0f;
            e1[rk] = 0.0f;
          }
          den0 += e0[rk];
          den1 += e1[rk];
        }
#pragma unroll
        for (int rk = 0; rk < 6; ++rk) {
          s_pr[tid][rk] = e0[rk] / den0;
          s_e1[tid][rk] = e1[rk];
        }
        s_den1[tid] = den1;
        s_p1post[tid] = e1[0] / den1;
        s_p2post[tid] = (e1[1] / den1) * ck.alt2_scale;
        s_Z[tid] = den0;
        s_p1[tid] = s_pr[tid][0];
        s_p_oov[tid] = 1.0f / den0;
      } else {
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
        const float z_tot = sum_e + tail_c * last_e;
        s_Z[tid] = z_tot;
        const float p1 = eff_gamma / z_tot;
        s_p1[tid] = (p1 < 1.0f) ? p1 : 1.0f;
        s_p_oov[tid] = 1.0f - s_p1[tid];
        s_p1post[tid] = 0.0f;
        s_p2post[tid] = 0.0f;
        s_den1[tid] = 1.0f;
#pragma unroll
        for (int rk = 0; rk < 6; ++rk) {
          s_e1[tid][rk] = 0.0f;
        }
      }
    } else {
      s_Z[tid] = 1e30f;
      s_p1[tid] = 0.0f;
      s_p_oov[tid] = 0.0f;
      s_p1post[tid] = 0.0f;
      s_p2post[tid] = 0.0f;
      s_den1[tid] = 1.0f;
#pragma unroll
      for (int rk = 0; rk < 6; ++rk) {
        s_e1[tid][rk] = 0.0f;
      }
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
    s_num_cands = 30;
  }
  __syncwarp();

  // Initialize 30 1-level DFlash candidates in slots 0..29, clear slots 30..95.
  s_cand_score[tid] = -1.0f;
  s_cand_score[tid + 32] = -1.0f;
  s_cand_score[tid + 64] = -1.0f;
  s_cand_hpos[tid] = -1;
  s_cand_hpos[tid + 32] = -1;
  s_cand_hpos[tid + 64] = -1;
  s_cand_ml[tid] = 0;
  s_cand_ml[tid + 32] = 0;
  s_cand_ml[tid + 64] = 0;
  s_cand_rk[tid] = -1;
  s_cand_rk[tid + 32] = -1;
  s_cand_rk[tid + 64] = -1;
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
        const float pr = (calib_mode != 0)
            ? s_pr[row_idx][rank]
            : (eff_gamma * __expf((zr - s_z0[row_idx]) * inv_temp) / s_Z[row_idx]);
        s_cand_score[tid] = s_A[row_idx] * pr;
        s_cand_tok[tid] = cand;
        s_cand_k[tid] = static_cast<int8_t>(row_idx + 1);
        s_cand_par[tid] = -1;
        s_cand_depth[tid] = 1;
        s_cand_kind[tid] = 0;
        s_cand_rk[tid] = static_cast<int8_t>(rank);
        if (pl.mode != 0 && s_hist_ok != 0) {
          const uint32_t ck_j = s_cand_key[row_idx][rank];
          if (ck_j != 0u) {
            s_cand_hpos[tid] = static_cast<int16_t>((ck_j & 0xFFFFu) + 1u);
            s_cand_ml[tid] = static_cast<int8_t>(ck_j >> 16);
          }
        }
      }
    }
  }
  __syncwarp();

  // Inject standalone OOV prompt-lookup & monomorphic 30B recycled roots (and floor in-top-6 copy candidates).
  if (tid == 0 && (pl.mode != 0 || tr.mode != 0 || tr.det_sh_root != 0) && s_hist_ok != 0 && s_win >= 2) {
    bool any_pl_root = false;
    bool any_boost = false;
    bool any_tr_boost = false;
    bool any_tr_root = false;
    for (int j = 0; j < 6; ++j) {
      if (!s_valid_row[j] || !(s_A[j] > 0.0f)) continue;
      for (int rk = 0; rk < 6; ++rk) {
        if (pl.mode != 0) {
          const int raw_cml = static_cast<int>(s_cand_key[j][rk] >> 16);
          if (raw_cml >= 1) {
            const int cml = (raw_cml > 6) ? 6 : raw_cml;
            const float bv = (rk == 0) ? pl.boost_d1[cml] : pl.boost_alt[cml];
            if (bv != 0.0f) any_boost = true;
          }
        }
        if (s_tr_b_rk[j][rk] > 0.0f) any_tr_boost = true;
      }
      if (pl.mode != 0 || tr.det_sh_root != 0) {
        const uint32_t pk = s_pl_key[j];
        const int raw_ml = static_cast<int>(pk >> 16);
        const int ml = (raw_ml > 6) ? 6 : raw_ml;
        const int min_req = (j == 0) ? pl.min_n : pl.mid_min_n;
        int eff_ml = raw_ml;
        int hpos = (pk != 0u) ? (static_cast<int>(pk & 0xFFFFu) + 1) : -1;
        float rw = (pl.mode != 0 && raw_ml >= min_req && ml >= 1) ? pl.root_p[ml] : 0.0f;
        float t6w = (pl.mode != 0 && raw_ml >= min_req && ml >= 1)
            ? ((pl.t6_root_p[ml] > 0.0f) ? pl.t6_root_p[ml] : pl.root_p[ml])
            : 0.0f;
        if (rw == 0.0f && tr.det_sh_root != 0) {
          const int32_t dk32 = static_cast<int32_t>(s_dk[j]);
          if (s_anc_tri3_cnt[j] >= 2u && s_anc_tri3_min[j] >= 0 &&
              s_anc_tri3_min[j] == s_anc_tri3_max[j] && s_anc_tri3_min[j] != dk32 &&
              tr.det_tri3_root_p > 0.0f) {
            hpos = s_anc_tri3_hp[j];
            rw = tr.det_tri3_root_p;
            t6w = rw;
            eff_ml = 3;
          } else if (s_anc_bg2_cnt[j] >= 3u && s_anc_bg2_min[j] >= 0 &&
                     s_anc_bg2_min[j] == s_anc_bg2_max[j] && s_anc_bg2_min[j] != dk32 &&
                     tr.det_bg2_root_p > 0.0f) {
            hpos = s_anc_bg2_hp[j];
            rw = tr.det_bg2_root_p;
            t6w = rw;
            eff_ml = 3;
          }
        }
        if (rw > 0.0f && hpos >= 0 && hpos < s_win) {
          const int64_t tok = static_cast<int64_t>(s_h[hpos]);
          if (tok >= 0 && tok != s_dk[j]) {
            const float gate = (pl.miss_gate == 1)
                ? (1.0f - s_p1[j])
                : ((pl.miss_gate == 2)
                       ? s_p_oov[j]
                       : ((pl.miss_gate == 3)
                              ? ((s_p_oov[j] > pl.oov_floor) ? s_p_oov[j] : pl.oov_floor)
                              : 1.0f));
            const float j_scale = (j == 0) ? 1.0f : pl.mid_scale;
            const float pl_s = s_A[j] * gate * rw * j_scale;
            int ex_slot = -1;
            for (int r = 0; r < 5; ++r) {
              const int sl = j * 5 + r;
              if (s_cand_score[sl] > -0.5f && s_cand_tok[sl] == tok) {
                ex_slot = sl;
                break;
              }
            }
            if (ex_slot >= 0) {
              if (pl.in_top6_floor != 0) {
                const float t6_s = s_A[j] * (1.0f - s_p1[j]) * t6w * j_scale;
                const float fl_s = (t6_s > pl_s) ? t6_s : pl_s;
                if (fl_s > s_cand_score[ex_slot]) {
                  s_cand_score[ex_slot] = fl_s;
                  any_pl_root = true;
                }
              }
              if (s_cand_hpos[ex_slot] < 0 || eff_ml > static_cast<int>(s_cand_ml[ex_slot])) {
                s_cand_hpos[ex_slot] = static_cast<int16_t>(hpos);
                s_cand_ml[ex_slot] = static_cast<int8_t>(eff_ml);
              }
            } else if (pl_s > 0.0f && s_num_cands < 96) {
              const int slot = s_num_cands++;
              s_cand_score[slot] = pl_s;
              s_cand_tok[slot] = tok;
              s_cand_k[slot] = static_cast<int8_t>(j + 1);
              s_cand_par[slot] = -1;
              s_cand_depth[slot] = 1;
              s_cand_kind[slot] = 4;
              s_cand_hpos[slot] = static_cast<int16_t>(hpos);
              s_cand_ml[slot] = static_cast<int8_t>(eff_ml);
              s_cand_rk[slot] = -1;
              any_pl_root = true;
            }
          }
        }
      }
    }
    if (calib_mode != 0 && tr.mode != 0 && s_tr_active_n > 0) {
      for (int j = 0; j < 6; ++j) {
        if (!s_valid_row[j] || !(s_A[j] > 0.0f)) continue;
        if (j >= 1 && s_ctx[15 + j] == s_ctx[14 + j]) continue;
        const int64_t otok = static_cast<int64_t>(s_tr_oov_tok[j]);
        const float ow = s_tr_oov_w[j];
        if (otok >= 0 && ow > 0.0f && otok != s_dk[j]) {
          const float gate = (s_p_oov[j] > pl.oov_floor) ? s_p_oov[j] : pl.oov_floor;
          const float tr_s = s_A[j] * gate * ow;
          int ex_slot = -1;
          for (int ic = 0; ic < s_num_cands; ++ic) {
            if (s_cand_score[ic] > -0.5f && static_cast<int>(s_cand_k[ic]) == j + 1 &&
                static_cast<int>(s_cand_par[ic]) == -1 && s_cand_tok[ic] == otok) {
              ex_slot = ic;
              break;
            }
          }
          if (ex_slot >= 0) {
            if (tr_s > s_cand_score[ex_slot]) {
              s_cand_score[ex_slot] = tr_s;
              any_tr_root = true;
            }
          } else if (tr_s > 0.0f && s_num_cands < 96) {
            const int slot = s_num_cands++;
            s_cand_score[slot] = tr_s;
            s_cand_tok[slot] = otok;
            s_cand_k[slot] = static_cast<int8_t>(j + 1);
            s_cand_par[slot] = -1;
            s_cand_depth[slot] = 1;
            s_cand_kind[slot] = 7;
            s_cand_hpos[slot] = -1;
            s_cand_ml[slot] = 0;
            s_cand_rk[slot] = -1;
            any_tr_root = true;
          }
        }
      }
    }
    if (stats != nullptr && stats_len >= 36) {
      if (any_pl_root) stats[34] += 1;
      if (any_boost) stats[35] += 1;
      if (stats_len >= 40) {
        if (any_tr_boost) stats[36] += 1;
        if (any_tr_root) stats[37] += 1;
      }
    }
  }
  __syncwarp();

  for (int li = 0; li < num_leaves; ++li) {
    const int r = 7 + li;
    const int nc = s_num_cands;
    float best_s = (tid < nc) ? s_cand_score[tid] : -1.0f;
    int best_idx = tid;
    const int idx2 = tid + 32;
    if (idx2 < nc) {
      const float s2 = s_cand_score[idx2];
      if (s2 > best_s) {
        best_s = s2;
        best_idx = idx2;
      }
    }
    const int idx3 = tid + 64;
    if (idx3 < nc) {
      const float s3 = s_cand_score[idx3];
      if (s3 > best_s) {
        best_s = s3;
        best_idx = idx3;
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
    // Collaborative warp-0 30B Target Recycling continuation query for selected leaf (when active).
    if (best_s > 0.0f) {
      if (tid == 0) {
        const int32_t tok32 = static_cast<int32_t>(s_cand_tok[best_idx]);
        const int k_r = static_cast<int>(s_cand_k[best_idx]);
        const int par = static_cast<int>(s_cand_par[best_idx]);
        s_leaf_tok_r[r] = tok32;
        s_leaf_prv_r[r] = (par < 0) ? s_ctx[15 + k_r - 1] : s_leaf_tok_r[par];
        s_leaf_prv2_r[r] = (par < 0) ? s_ctx[15 + k_r - 2] : s_leaf_prv_r[par];
        s_tr_ch_tok = -1;
        s_tr_ch_p = 0.0f;
      }
      if (li + 1 < num_leaves && calib_mode != 0 && tr.mode != 0 && tr.cont_mode != 0 &&
          s_hist_ok != 0 && s_tr_active_n > 0 && saved_leaf_parent != nullptr) {
        __syncwarp();
        const int dp_w = static_cast<int>(s_cand_depth[best_idx]);
        const int k_w = static_cast<int>(s_cand_k[best_idx]);
        if (dp_w < eff_max_depth && dp_w < tr.max_tr_dp && k_w < 6) {
          const int32_t tok32 = s_leaf_tok_r[r];
          const int32_t pv_r = s_leaf_prv_r[r];
          const int32_t pv2_r = s_leaf_prv2_r[r];
          const int n_act = s_tr_active_n;
          const int cur_st = s_tr_cur_step;
          const int pos_leaf = static_cast<int>(positions[0]) + k_w;

          int u_tot_c = 0, t0_u_c = 0, u_min_c = 2147483647, u_max_c = -1;
          int bg_t0_c = 0, bg_min_c = 2147483647, bg_max_c = -1;
          int tri_t0_c = 0, tri_min_c = 2147483647, tri_max_c = -1;
          int near_c = 0;

          for (int idx = lane; idx < n_act; idx += 32) {
            if (s_tr_cur[idx] != tok32) continue;
            const int32_t u_t = s_tr_u[idx];
            const int32_t p1_t = s_tr_p1[idx];
            const int32_t p2_t = s_tr_p2[idx];
            const int age = cur_st - s_tr_st[idx];
            const int pt = s_tr_pos_tier[idx];
            const int pos_t = pt >> 1;
            const int tier_t = pt & 1;
            const int dist = (pos_leaf >= pos_t) ? (pos_leaf - pos_t) : (pos_t - pos_leaf);

            u_tot_c += 1;
            if (tier_t == 0) t0_u_c += 1;
            if (u_t < u_min_c) u_min_c = u_t;
            if (u_t > u_max_c) u_max_c = u_t;
            if (age <= 2 && dist <= 12) near_c += 1;
            if (pv_r >= 0 && p1_t == pv_r && tier_t == 0) {
              bg_t0_c += 1;
              if (u_t < bg_min_c) bg_min_c = u_t;
              if (u_t > bg_max_c) bg_max_c = u_t;
              if (pv2_r >= 0 && p2_t == pv2_r) {
                tri_t0_c += 1;
                if (u_t < tri_min_c) tri_min_c = u_t;
                if (u_t > tri_max_c) tri_max_c = u_t;
              }
            }
          }

          u_tot_c = __reduce_add_sync(0xffffffffu, u_tot_c);
          if (u_tot_c > 0) {
            t0_u_c = __reduce_add_sync(0xffffffffu, t0_u_c);
            u_min_c = __reduce_min_sync(0xffffffffu, u_min_c);
            u_max_c = __reduce_max_sync(0xffffffffu, u_max_c);
            near_c = __reduce_add_sync(0xffffffffu, near_c);
            bg_t0_c = __reduce_add_sync(0xffffffffu, bg_t0_c);
            bg_min_c = __reduce_min_sync(0xffffffffu, bg_min_c);
            bg_max_c = __reduce_max_sync(0xffffffffu, bg_max_c);
            tri_t0_c = __reduce_add_sync(0xffffffffu, tri_t0_c);
            tri_min_c = __reduce_min_sync(0xffffffffu, tri_min_c);
            tri_max_c = __reduce_max_sync(0xffffffffu, tri_max_c);

            int sh_occ = 0;
            bool sh_conflict = false;
            bool sh_bg_conflict = false;
            const bool is_tri_mono = (tri_t0_c >= 2 && tri_min_c == tri_max_c);
            if (!is_tri_mono && ((bg_t0_c >= 1 && bg_min_c == bg_max_c) ||
                                 (u_min_c == u_max_c && u_min_c != tok32))) {
              int my_occ = 0;
              int my_conf = 0;
              int my_bg_conf = 0;
              const int win_m1 = s_win - 1;
              for (int hi = lane; hi < win_m1; hi += 32) {
                if (s_h[hi] == tok32) {
                  my_occ += 1;
                  const int32_t nxt_h = s_h[hi + 1];
                  if (nxt_h != u_min_c) my_conf = 1;
                  if (hi >= 1 && s_h[hi - 1] == pv_r && nxt_h != bg_min_c) my_bg_conf = 1;
                }
              }
              sh_occ = __reduce_add_sync(0xffffffffu, my_occ);
              sh_conflict = (__ballot_sync(0xffffffffu, my_conf) != 0u);
              sh_bg_conflict = (__ballot_sync(0xffffffffu, my_bg_conf) != 0u);
            }

            if (lane == 0) {
              const int j_ch = k_w;
              auto in_ch_top6 = [&](int32_t cand_c) -> bool {
                if (!s_valid_row[j_ch]) return false;
#pragma unroll
                for (int rp = 0; rp < 6; ++rp) {
                  if (s_top[j_ch][rp] == cand_c) return true;
                }
                return false;
              };
              if (is_tri_mono) {
                s_tr_ch_tok = static_cast<int64_t>(tri_min_c);
                s_tr_ch_p = in_ch_top6(tri_min_c) ? tr.ch_bg_t6 : tr.ch_bg_oov;
              } else if (bg_t0_c >= 2 && bg_min_c == bg_max_c &&
                         (!sh_bg_conflict || (near_c >= 2 && in_ch_top6(bg_min_c)))) {
                s_tr_ch_tok = static_cast<int64_t>(bg_min_c);
                s_tr_ch_p = in_ch_top6(bg_min_c) ? tr.ch_bg_t6 : tr.ch_bg_oov;
              } else if (bg_t0_c >= 1 && bg_min_c == bg_max_c && !sh_bg_conflict &&
                         tr.ch_bg1_t6 > 0.0f && in_ch_top6(bg_min_c)) {
                s_tr_ch_tok = static_cast<int64_t>(bg_min_c);
                s_tr_ch_p = tr.ch_bg1_t6;
              } else if (u_min_c == u_max_c && u_min_c != tok32) {
                const bool in_t6 = in_ch_top6(u_min_c);
                if (near_c >= 1 && !sh_conflict && (sh_occ == 0 || (t0_u_c >= 2 && sh_occ <= 3))) {
                  s_tr_ch_tok = static_cast<int64_t>(u_min_c);
                  s_tr_ch_p = in_t6 ? tr.ch_near_t6 : tr.ch_near_oov;
                } else if (in_t6 && t0_u_c >= 1 && u_tot_c >= 2 && sh_occ <= 3) {
                  s_tr_ch_tok = static_cast<int64_t>(u_min_c);
                  s_tr_ch_p = tr.ch_t6_agree;
                }
              }
            }
          }
        }
        __syncwarp();
      }
    }
    if (tid == 0) {
      if (best_s > 0.0f) {
        const int64_t tok = s_cand_tok[best_idx];
        const int k_r = static_cast<int>(s_cand_k[best_idx]);
        const int par = static_cast<int>(s_cand_par[best_idx]);
        const int dp = static_cast<int>(s_cand_depth[best_idx]);
        const int kind = static_cast<int>(s_cand_kind[best_idx]);
        const int hpos = static_cast<int>(s_cand_hpos[best_idx]);
        const int ml = static_cast<int>(s_cand_ml[best_idx]);
        const int rk_in_row = static_cast<int>(s_cand_rk[best_idx]);
        s_cand_score[best_idx] = -1.0f;

        int b_sub = 0;
        uint32_t pkd = 0u;
        if (par < 0) {
          b_sub = tree_m - k_r;
          pkd = static_cast<uint32_t>(r) & 0xFu;
        } else {
          b_sub = s_row_base_sub[par];
          pkd = s_row_packed[par] | ((static_cast<uint32_t>(r) & 0xFu) << (4 * (dp - 1)));
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
          const int j_par = k_r - 1;  // 0..4
          const int j_ch = k_r;       // 1..5
          int64_t spawned[10];
          int n_spawned = 0;
          auto already_spawned = [&](int64_t t) -> bool {
            for (int si = 0; si < n_spawned; ++si) {
              if (spawned[si] == t) return true;
            }
            return false;
          };
          const int root_ml = (kind == 5) ? (ml - (dp - 1)) : ml;
          const int max_pl_dp = (root_ml <= 4) ? pl.max_dp_4 : ((root_ml == 5) ? pl.max_dp_5 : pl.max_dp_6);
          const bool can_pl_cont = (pl.mode != 0 && s_hist_ok != 0 && hpos >= 0 && hpos + 1 < s_win &&
                                    dp < max_pl_dp && (kind == 4 || kind == 5 || ml >= pl.cont_from_df_n));
          const int64_t pl_next_tok = can_pl_cont ? static_cast<int64_t>(s_h[hpos + 1]) : -1;
          const int next_ml = (ml < 16) ? (ml + 1) : 16;
          const int next_ml_b = (next_ml > 6) ? 6 : next_ml;
          const float pl_cp = (can_pl_cont && next_ml_b >= 1) ? pl.cont_p[next_ml_b] : 0.0f;
          const int64_t tr_ch_tok = s_tr_ch_tok;
          const float tr_ch_p = s_tr_ch_p;
          if (tr_ch_tok >= 0 && tr_ch_p > 0.0f && stats != nullptr && stats_len >= 40) {
            stats[38] += 1;
          }

          // Compute smoothed Markov bigram/trigram post-miss distribution when history coherence matches.
          const bool allow_bg_kind =
              (kind == 0) || (bg.bg_sub != 0 && kind == 1) || (bg.bg_off != 0 && (kind == 3 || kind == 6));
          const bool use_bg = (calib_mode != 0 && bg.bg_mode != 0 && s_hist_ok != 0 &&
                               s_valid_row[j_ch] != 0 && rk_in_row >= 0 && allow_bg_kind);
          const uint32_t ac = use_bg ? s_alt_cnt[j_par][rk_in_row] : 0u;
          const uint32_t m3 = use_bg ? s_bg3_mask[j_par][rk_in_row] : 0u;
          float p_post_vec[6];
          uint32_t bg_mask = 0u;
          bool has_p_post_vec = false;
          if (use_bg && ac > 0u) {
            const float inv_den = 1.0f / (static_cast<float>(ac) + bg.bg_alpha);
            const float min_p = (kind == 1) ? bg.sub_min_p : bg.bg_min_p;
            float boosts[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int rp = 0; rp < 6; ++rp) {
              const uint32_t cnt_rp = s_bg_cnt[j_par][rk_in_row][rp];
              if (cnt_rp > 0u) {
                const float p_bg = static_cast<float>(cnt_rp) * inv_den;
                const bool is3 = ((m3 & (1u << rp)) != 0u) && (p_bg >= bg.tri_min_p);
                float bv = 0.0f;
                if (p_bg >= min_p) bv += bg.f2 + bg.w2 * p_bg;
                if (is3) bv += bg.f3 + bg.w3 * p_bg;
                if (bv > 0.0f) {
                  boosts[rp] = bv;
                  bg_mask |= (1u << rp);
                }
              }
            }
            if (bg_mask != 0u) {
              float e1_mod[6];
              float den1_mod = 1.0f;
#pragma unroll
              for (int rp = 0; rp < 6; ++rp) {
                float ev = s_e1[j_ch][rp];
                if (ev > 0.0f && boosts[rp] > 0.0f) {
                  ev *= __expf(boosts[rp]);
                }
                e1_mod[rp] = ev;
                den1_mod += ev;
              }
              const float inv_d1 = 1.0f / den1_mod;
#pragma unroll
              for (int rp = 0; rp < 6; ++rp) {
                p_post_vec[rp] = e1_mod[rp] * inv_d1;
              }
              has_p_post_vec = true;
            }
          }

          auto resolve_bg_hpos = [&](int rp, bool agrees_pl, int16_t& out_hp, int8_t& out_ml) {
            if (agrees_pl) {
              out_hp = static_cast<int16_t>(hpos + 1);
              out_ml = static_cast<int8_t>(next_ml);
              return;
            }
            out_hp = -1;
            out_ml = 0;
            if (!use_bg || ac == 0u) return;
            const uint32_t cnt_rp = s_bg_cnt[j_par][rk_in_row][rp];
            if (cnt_rp == 0u) return;
            const float p_bg = static_cast<float>(cnt_rp) / (static_cast<float>(ac) + bg.bg_alpha);
            const bool is3 = ((m3 & (1u << rp)) != 0u) && (p_bg >= bg.tri_min_p);
            const bool ok_hp = is3 || (bg.chp_bg2 != 0 && cnt_rp >= 2u && p_bg >= 0.45f);
            if (ok_hp) {
              const uint32_t bk = s_bg_key[j_par][rk_in_row][rp];
              if (bk != 0u) {
                const int bk_hp = static_cast<int>(bk & 0xFFFFu);
                const int bk_ml = static_cast<int>(bk >> 16);
                out_hp = static_cast<int16_t>(bk_hp);
                out_ml = static_cast<int8_t>((bk_ml < 3) ? 3 : bk_ml);
              }
            }
          };

          // 1) Aligned substitution continuation (`sub`, kind=1): d_{k_r+1}#1 = s_dk[k_r].
          if ((kind == 0 || kind == 1 || kind == 3 || kind == 4 || kind == 5 || kind == 6 || kind == 7) &&
              s_valid_row[j_ch] && s_num_cands < 96) {
            const int64_t sub_tok = s_dk[j_ch];
            const bool agrees_pl = (can_pl_cont && pl_next_tok == sub_tok && pl_cp > 0.0f);
            const bool agrees_tr = (tr_ch_tok == sub_tok && tr_ch_p > 0.0f);
            float p_cont;
            if (kind == 1) {
              p_cont = s_p1[j_ch];
              if (has_p_post_vec && ((bg_mask & 1u) != 0u) && p_post_vec[0] > p_cont) {
                p_cont = p_post_vec[0];
              }
            } else {
              p_cont = has_p_post_vec ? p_post_vec[0] : s_p1post[j_ch];
            }
            if (agrees_pl && pl_cp > p_cont) p_cont = pl_cp;
            if (agrees_tr && tr_ch_p > p_cont) p_cont = tr_ch_p;
            const float child_s = (calib_mode != 0)
                ? (best_s * p_cont)
                : (best_s * ((kind == 0) ? sub_coef : cont_decay) * s_p1[j_ch]);
            if (child_s > 0.0f && sub_tok >= 0) {
              int16_t ch_hp = -1;
              int8_t ch_ml = 0;
              resolve_bg_hpos(0, agrees_pl, ch_hp, ch_ml);
              const int slot = s_num_cands++;
              s_cand_score[slot] = child_s;
              s_cand_tok[slot] = sub_tok;
              s_cand_k[slot] = static_cast<int8_t>(k_child);
              s_cand_par[slot] = static_cast<int8_t>(r);
              s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
              s_cand_kind[slot] = 1;
              s_cand_hpos[slot] = ch_hp;
              s_cand_ml[slot] = ch_ml;
              s_cand_rk[slot] = 0;
              spawned[n_spawned++] = sub_tok;
            }
          }
          // 2) Post-miss top-2 (`alt2`, kind=3) + Markov-coherent / 30B-recycled r'=2..5 (`coh_cont`, kind=6).
          if (calib_mode != 0 && s_valid_row[j_ch]) {
            for (int rp = 1; rp < 6; ++rp) {
              if (s_num_cands >= 96) break;
              const int64_t ctok = tree_top[j_ch * tree_k + rp];
              if (ctok < 0 || already_spawned(ctok)) continue;
              const bool is_bg_hit = ((bg_mask & (1u << rp)) != 0u);
              const bool agrees_tr = (tr_ch_tok == ctok && tr_ch_p > 0.0f);
              if (!(((kind == 0 || kind == 4 || kind == 7) && rp == 1) || is_bg_hit || agrees_tr)) continue;
              const bool agrees_pl = (can_pl_cont && pl_next_tok == ctok && pl_cp > 0.0f);
              float p_c = has_p_post_vec
                  ? ((rp == 1) ? (p_post_vec[1] * ck.alt2_scale) : p_post_vec[rp])
                  : ((rp == 1) ? s_p2post[j_ch] : (s_e1[j_ch][rp] / s_den1[j_ch]));
              if (agrees_pl && pl_cp > p_c) p_c = pl_cp;
              if (agrees_tr && tr_ch_p > p_c) p_c = tr_ch_p;
              const float child_s = best_s * p_c;
              if (child_s > 0.0f) {
                int16_t ch_hp = -1;
                int8_t ch_ml = 0;
                resolve_bg_hpos(rp, agrees_pl, ch_hp, ch_ml);
                const int slot = s_num_cands++;
                s_cand_score[slot] = child_s;
                s_cand_tok[slot] = ctok;
                s_cand_k[slot] = static_cast<int8_t>(k_child);
                s_cand_par[slot] = static_cast<int8_t>(r);
                s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
                s_cand_kind[slot] = (rp == 1) ? 3 : 6;
                s_cand_hpos[slot] = ch_hp;
                s_cand_ml[slot] = ch_ml;
                s_cand_rk[slot] = static_cast<int8_t>(rp);
                spawned[n_spawned++] = ctok;
              }
            }
          }
          // 3) Off-by-1 shift recovery (`shf`, kind=2): d_{k_r}#1 = s_dk[k_r - 1].
          if ((kind == 0 || kind == 2 || kind == 4 || kind == 7) && s_valid_row[k_r - 1] && s_num_cands < 96) {
            const int64_t shf_tok = s_dk[k_r - 1];
            if (shf_tok >= 0 && !already_spawned(shf_tok)) {
              const bool agrees_pl = (can_pl_cont && pl_next_tok == shf_tok && pl_cp > 0.0f);
              const bool agrees_tr = (tr_ch_tok == shf_tok && tr_ch_p > 0.0f);
              float p_shf;
              if (kind == 0 || kind == 4 || kind == 7) {
                if (bg.shf_cal != 0 && kind == 0 && rk_in_row >= 1) {
                  p_shf = (s_hist_ok != 0 && s_shf_bg[j_par][rk_in_row - 1] != 0u) ? bg.shf_hi : bg.shf_lo;
                } else {
                  p_shf = ck.shf;
                }
              } else {
                p_shf = s_p1[k_r - 1];
              }
              if (agrees_pl && pl_cp > p_shf) p_shf = pl_cp;
              if (agrees_tr && tr_ch_p > p_shf) p_shf = tr_ch_p;
              const float child_s = (calib_mode != 0)
                  ? (best_s * p_shf)
                  : ((kind == 0) ? (best_s * shf_coef) : (best_s * cont_decay * s_p1[k_r - 1]));
              if (child_s > 0.0f) {
                const int slot = s_num_cands++;
                s_cand_score[slot] = child_s;
                s_cand_tok[slot] = shf_tok;
                s_cand_k[slot] = static_cast<int8_t>(k_child);
                s_cand_par[slot] = static_cast<int8_t>(r);
                s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
                s_cand_kind[slot] = 2;
                s_cand_hpos[slot] = agrees_pl ? static_cast<int16_t>(hpos + 1) : -1;
                s_cand_ml[slot] = agrees_pl ? static_cast<int8_t>(next_ml) : 0;
                s_cand_rk[slot] = -1;
                spawned[n_spawned++] = shf_tok;
              }
            }
          }
          // 4) Prompt-lookup copy continuation (`pl_cont`, kind=5): s_h[hpos + 1].
          if (can_pl_cont && pl_next_tok >= 0 && pl_cp > 0.0f &&
              !already_spawned(pl_next_tok) && s_num_cands < 96) {
            const float p_use = (tr_ch_tok == pl_next_tok && tr_ch_p > pl_cp) ? tr_ch_p : pl_cp;
            const float child_s = best_s * p_use;
            if (child_s > 0.0f) {
              const int slot = s_num_cands++;
              s_cand_score[slot] = child_s;
              s_cand_tok[slot] = pl_next_tok;
              s_cand_k[slot] = static_cast<int8_t>(k_child);
              s_cand_par[slot] = static_cast<int8_t>(r);
              s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
              s_cand_kind[slot] = 5;
              s_cand_hpos[slot] = static_cast<int16_t>(hpos + 1);
              s_cand_ml[slot] = static_cast<int8_t>(next_ml);
              s_cand_rk[slot] = -1;
              spawned[n_spawned++] = pl_next_tok;
            }
          }
          // 5) Branch-conditioned unique OOV successor (deterministic nxt occurring >=2 times in s_h).
          if (calib_mode != 0 && bg.uniq_mode != 0 && s_hist_ok != 0 && rk_in_row >= 1 &&
              (kind == 0 || (bg.bg_off != 0 && (kind == 3 || kind == 6))) && s_num_cands < 96) {
            const uint32_t uc = s_alt_cnt[j_par][rk_in_row];
            const int32_t umin = s_uniq_min[j_par][rk_in_row - 1];
            const int32_t umax = s_uniq_max[j_par][rk_in_row - 1];
            if (uc >= 2u && umin >= 0 && umin == umax) {
              const int64_t utok = static_cast<int64_t>(umin);
              if (!already_spawned(utok)) {
                float p_u = (uc == 2u) ? bg.uniq_p2 : bg.uniq_p3;
                if (tr_ch_tok == utok && tr_ch_p > p_u) p_u = tr_ch_p;
                const float child_s = best_s * p_u;
                if (child_s > 0.0f) {
                  const int slot = s_num_cands++;
                  s_cand_score[slot] = child_s;
                  s_cand_tok[slot] = utok;
                  s_cand_k[slot] = static_cast<int8_t>(k_child);
                  s_cand_par[slot] = static_cast<int8_t>(r);
                  s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
                  s_cand_kind[slot] = 5;
                  s_cand_hpos[slot] = static_cast<int16_t>(s_uniq_hp[j_par][rk_in_row - 1]);
                  s_cand_ml[slot] = (uc == 2u) ? 3 : 4;
                  s_cand_rk[slot] = -1;
                  spawned[n_spawned++] = utok;
                }
              }
            }
          }
          // 6) Pure 30B Target-Recycled OOV continuation (`tr_cont`, kind=7).
          if (calib_mode != 0 && tr_ch_tok >= 0 && tr_ch_p > 0.0f &&
              !already_spawned(tr_ch_tok) && s_num_cands < 96) {
            const float child_s = best_s * tr_ch_p;
            if (child_s > 0.0f) {
              const int slot = s_num_cands++;
              s_cand_score[slot] = child_s;
              s_cand_tok[slot] = tr_ch_tok;
              s_cand_k[slot] = static_cast<int8_t>(k_child);
              s_cand_par[slot] = static_cast<int8_t>(r);
              s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
              s_cand_kind[slot] = 7;
              s_cand_hpos[slot] = -1;
              s_cand_ml[slot] = 0;
              s_cand_rk[slot] = -1;
              spawned[n_spawned++] = tr_ch_tok;
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

  // Record per-row (cur, prv1, prv2, pos_r) for 16-row 30B Target Recycling commit in 1 parallel cycle.
  if (tid < tree_m && saved_row_ctx != nullptr) {
    const int p0_i32 = static_cast<int>(positions[0]);
    const int32_t p_m1 = (s_hist_ok != 0 && s_ctx[14] >= 0) ? s_ctx[14] : -1;
    const int32_t p_m2 = (s_hist_ok != 0 && s_ctx[13] >= 0) ? s_ctx[13] : -1;
    int32_t cur_r = -1, prv1_r = -1, prv2_r = -1, pos_r = -1;
    if (!high_acc) {
      if (tid < 7) {
        cur_r = input_ids[tid];
        prv1_r = (tid == 0) ? p_m1 : input_ids[tid - 1];
        prv2_r = (tid == 0) ? p_m2 : ((tid == 1) ? p_m1 : input_ids[tid - 2]);
        pos_r = p0_i32 + tid;
      } else if (s_leaf_tok_r[tid] >= 0 && saved_leaf_k[tid] > 0) {
        cur_r = s_leaf_tok_r[tid];
        prv1_r = (s_leaf_prv_r[tid] >= 0) ? s_leaf_prv_r[tid] : -1;
        prv2_r = (s_leaf_prv2_r[tid] >= 0) ? s_leaf_prv2_r[tid] : -1;
        pos_r = p0_i32 + saved_leaf_k[tid];
      }
    }
    saved_row_ctx[tid * 4 + 0] = cur_r;
    saved_row_ctx[tid * 4 + 1] = prv1_r;
    saved_row_ctx[tid * 4 + 2] = prv2_r;
    saved_row_ctx[tid * 4 + 3] = pos_r;
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
    float high_acc_cpc,
    int32_t* __restrict__ hist,
    int32_t* __restrict__ hist_len,
    int hist_cap,
    const int64_t* __restrict__ saved_pos0,
    int32_t* __restrict__ saved_row_ctx,
    int32_t* __restrict__ tr_log,
    int32_t* __restrict__ tr_state,
    int tr_cap) {
  const int mode = leaf_flag[0];
  leaf_flag[0] = 0;  // Clear immediately so non-tree steps see 0.
  int32_t ns = num_sampled[0];
  int32_t nr = num_rejected[0];
  const int32_t raw_ns = ns;
  int32_t raw_u[16];
  const bool can_rec_tr = (no_hit_mode == 0 && tr_log != nullptr && tr_state != nullptr &&
                           saved_row_ctx != nullptr && tr_cap > 0);
  if (can_rec_tr) {
    for (int r = 0; r < 7 && r < tree_m; ++r) {
      raw_u[r] = static_cast<int32_t>(sampled_tokens[r]);
    }
    for (int r = 7; r < tree_m; ++r) {
      raw_u[r] = static_cast<int32_t>(leaf_u_toks[r - 7]);
    }
  }
  uint32_t matched_mask = 0u;
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
      matched_mask |= (1u << matched_r);
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

  // Append exact committed tokens (including tree leaf hits!) directly to token history.
  if (hist != nullptr && hist_len != nullptr && saved_pos0 != nullptr) {
    const int64_t p0 = saved_pos0[0];
    if (p0 >= 0 && hist_len[0] == static_cast<int32_t>(p0 + 1)) {
      const int64_t new_len = p0 + 1 + ns;
      if (new_len <= hist_cap) {
        for (int i = 0; i < ns; ++i) {
          hist[p0 + 1 + i] = static_cast<int32_t>(sampled_tokens[i]);
        }
        hist_len[0] = static_cast<int32_t>(new_len);
      } else {
        hist_len[0] = -1;
      }
    }
  }

  // Append all active 30B-verified (cur, prv1, prv2) -> raw_u[r] pairs to per-request tr_log.
  if (can_rec_tr && saved_pos0 != nullptr) {
    const int64_t p0 = saved_pos0[0];
    const int32_t ctx_p0 = saved_row_ctx[3];
    saved_row_ctx[3] = -1;  // Consume once so non-tree steps cannot re-read stale row contexts.
    if (p0 >= 0 && ctx_p0 == static_cast<int32_t>(p0) && saved_row_ctx[0] >= 0) {
      int32_t len = tr_state[0];
      int32_t step_id = tr_state[1];
      if (len < 0 || (len > 0 && p0 <= static_cast<int64_t>(tr_state[2]))) {
        len = 0;
        step_id = 0;
      }
      uint32_t exact_mask = 0u;
      for (int r = 0; r < 7 && r < tree_m; ++r) {
        if (r < raw_ns) exact_mask |= (1u << r);
      }
      for (int r = 7; r < tree_m; ++r) {
        if (saved_row_ctx[r * 4 + 0] < 0) continue;
        const int32_t par = (saved_leaf_parent != nullptr) ? saved_leaf_parent[r] : -1;
        const int32_t k = saved_leaf_k[r];
        bool ex = false;
        if (par < 0) {
          ex = (k > 0 && k <= raw_ns);
        } else if (par >= 7 && par < r) {
          ex = ((exact_mask & (1u << par)) != 0u) && ((matched_mask & (1u << par)) != 0u);
        }
        if (ex) exact_mask |= (1u << r);
      }
      int appended = 0;
      for (int r = 0; r < tree_m; ++r) {
        const int32_t cur_t = saved_row_ctx[r * 4 + 0];
        const int32_t u_t = raw_u[r];
        if (cur_t < 0 || u_t < 0) continue;
        if (len >= tr_cap) break;
        const int base8 = len * 8;
        tr_log[base8 + 0] = r;
        tr_log[base8 + 1] = cur_t;
        tr_log[base8 + 2] = saved_row_ctx[r * 4 + 1];
        tr_log[base8 + 3] = saved_row_ctx[r * 4 + 2];
        tr_log[base8 + 4] = u_t;
        tr_log[base8 + 5] = step_id;
        tr_log[base8 + 6] = (r == 0) ? ctx_p0 : saved_row_ctx[r * 4 + 3];
        tr_log[base8 + 7] = ((exact_mask & (1u << r)) != 0u) ? 0 : 1;
        len += 1;
        appended += 1;
      }
      tr_state[0] = len;
      tr_state[1] = step_id + 1;
      tr_state[2] = static_cast<int32_t>(p0);
      if (stats != nullptr && stats_len >= 40) {
        stats[39] += appended;
      }
    }
  }

  if (req_acc_state != nullptr) {
    const int32_t r = req_acc_state[0] + 1;
    const int32_t c = req_acc_state[1] + ns;
    req_acc_state[0] = r;
    req_acc_state[1] = c;
    // On M=16 verify, pausing leaves saves 0.00 ms because FULL(16) always runs; keep all 9 leaves
    // active whenever depth==6 returns (never permanently latch off after a ceiling-clamped step).
    req_acc_state[2] =
        (high_acc_cpc > 0.0f && r >= 4 && static_cast<float>(c) > high_acc_cpc * static_cast<float>(r)) ? 1 : 0;
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
    // Dynamic high-acceptance pause (un-pauses whenever mean drops <= 4.0).
    req_acc_state[2] = (r >= 4 && c > 4 * r) ? 1 : 0;
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
    double gamma,
    c10::optional<torch::Tensor> saved_leaf_parent,
    int64_t max_depth,
    double sub_coef,
    double shf_coef,
    double cont_decay,
    c10::optional<torch::Tensor> hist,
    c10::optional<torch::Tensor> hist_len,
    c10::optional<torch::Tensor> saved_pos0,
    c10::optional<torch::Tensor> stats,
    c10::optional<torch::Tensor> saved_row_ctx,
    c10::optional<torch::Tensor> tr_log,
    c10::optional<torch::Tensor> tr_state) {
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
  static const CalibKnobs ck_env = []() -> CalibKnobs {
    auto env_f = [](const char* name, float def) -> float {
      const char* s = std::getenv(name);
      return (s && *s) ? static_cast<float>(std::strtod(s, nullptr)) : def;
    };
    return CalibKnobs{
        env_f("MUSE_CAL_A_SHIFT", -0.18f),
        env_f("MUSE_CAL_A2", CAL_A2),
        env_f("MUSE_CAL_BETA", CAL_BETA),
        env_f("MUSE_CAL_BETA_POST", CAL_BETA_POST),
        env_f("MUSE_CAL_W_SM", CAL_W_SM_PRE),
        env_f("MUSE_CAL_W_Z0", CAL_W_Z0_PRE),
        env_f("MUSE_CAL_W_Z0_POST", CAL_W_Z0_POST),
        env_f("MUSE_CAL_Z0_CENTER", 36.0f),
        env_f("MUSE_CAL_SHIFT", CAL_SHIFT),
        env_f("MUSE_CAL_ALT2", 1.20f),
    };
  }();
  const int32_t* hist_ptr = nullptr;
  const int32_t* hist_len_ptr = nullptr;
  int hist_cap = 0;
  if (hist.has_value() && hist.value().defined() && hist.value().numel() > 0 &&
      hist_len.has_value() && hist_len.value().defined() && hist_len.value().numel() >= 1) {
    TORCH_CHECK(hist.value().is_cuda() && hist.value().scalar_type() == torch::kInt32 && hist.value().is_contiguous());
    TORCH_CHECK(hist_len.value().is_cuda() && hist_len.value().scalar_type() == torch::kInt32);
    hist_ptr = hist.value().data_ptr<int32_t>();
    hist_len_ptr = hist_len.value().data_ptr<int32_t>();
    hist_cap = static_cast<int>(hist.value().numel());
  }
  int64_t* saved_pos0_ptr = nullptr;
  if (saved_pos0.has_value() && saved_pos0.value().defined() && saved_pos0.value().numel() >= 1) {
    TORCH_CHECK(saved_pos0.value().is_cuda() && saved_pos0.value().scalar_type() == torch::kInt64);
    saved_pos0_ptr = saved_pos0.value().data_ptr<int64_t>();
  }
  int64_t* stats_ptr = nullptr;
  int stats_len = 0;
  if (stats.has_value() && stats.value().defined() && stats.value().numel() >= 4) {
    TORCH_CHECK(stats.value().is_cuda() && stats.value().scalar_type() == torch::kInt64);
    stats_ptr = stats.value().data_ptr<int64_t>();
    stats_len = static_cast<int>(stats.value().numel());
  }
  static const PlKnobs pl_env = []() -> PlKnobs {
    auto env_i = [](const char* name, int def) -> int {
      const char* s = std::getenv(name);
      return (s && *s) ? static_cast<int>(std::strtol(s, nullptr, 10)) : def;
    };
    auto env_f = [](const char* name, float def) -> float {
      const char* s = std::getenv(name);
      return (s && *s) ? static_cast<float>(std::strtod(s, nullptr)) : def;
    };
    PlKnobs p{};
    p.mode = env_i("MUSE_PL_MODE", 1);
    p.min_n = env_i("MUSE_PL_MIN_N", 4);
    p.mid_min_n = env_i("MUSE_PL_MID_MIN_N", 4);
    p.miss_gate = env_i("MUSE_PL_MISS_GATE", 3);
    p.in_top6_floor = env_i("MUSE_PL_IN_TOP6_FLOOR", 1);
    p.cont_from_df_n = env_i("MUSE_PL_CONT_FROM_DF_N", 3);
    p.max_dp_4 = env_i("MUSE_PL_MAX_DP4", 3);
    p.max_dp_5 = env_i("MUSE_PL_MAX_DP5", 4);
    p.max_dp_6 = env_i("MUSE_PL_MAX_DP6", 5);
    p.mid_scale = env_f("MUSE_PL_MID_SCALE", 1.0f);
    p.oov_floor = env_f("MUSE_PL_OOV_FLOOR", 0.12f);
    p.root_p[0] = 0.0f;
    p.root_p[1] = env_f("MUSE_PL_ROOT_P1", 0.0f);
    p.root_p[2] = env_f("MUSE_PL_ROOT_P2", 0.0f);
    p.root_p[3] = env_f("MUSE_PL_ROOT_P3", 0.0f);
    p.root_p[4] = env_f("MUSE_PL_ROOT_P4", 0.50f);
    p.root_p[5] = env_f("MUSE_PL_ROOT_P5", 0.85f);
    p.root_p[6] = env_f("MUSE_PL_ROOT_P6", 1.00f);
    p.t6_root_p[0] = 0.0f;
    p.t6_root_p[1] = env_f("MUSE_PL_T6_ROOT_P1", 0.0f);
    p.t6_root_p[2] = env_f("MUSE_PL_T6_ROOT_P2", 0.0f);
    p.t6_root_p[3] = env_f("MUSE_PL_T6_ROOT_P3", 0.0f);
    p.t6_root_p[4] = env_f("MUSE_PL_T6_ROOT_P4", 0.50f);
    p.t6_root_p[5] = env_f("MUSE_PL_T6_ROOT_P5", 0.70f);
    p.t6_root_p[6] = env_f("MUSE_PL_T6_ROOT_P6", 0.80f);
    p.cont_p[0] = 0.0f;
    p.cont_p[1] = env_f("MUSE_PL_CONT_P1", 0.0f);
    p.cont_p[2] = env_f("MUSE_PL_CONT_P2", 0.0f);
    p.cont_p[3] = env_f("MUSE_PL_CONT_P3", 0.35f);
    p.cont_p[4] = env_f("MUSE_PL_CONT_P4", 0.78f);
    p.cont_p[5] = env_f("MUSE_PL_CONT_P5", 0.84f);
    p.cont_p[6] = env_f("MUSE_PL_CONT_P6", 0.88f);
    p.boost_alt[0] = 0.0f;
    p.boost_alt[1] = env_f("MUSE_PL_BOOST_ALT1", 0.0f);
    p.boost_alt[2] = env_f("MUSE_PL_BOOST_ALT2", 0.0f);
    p.boost_alt[3] = env_f("MUSE_PL_BOOST_ALT3", 0.45f);
    p.boost_alt[4] = env_f("MUSE_PL_BOOST_ALT4", 0.75f);
    p.boost_alt[5] = env_f("MUSE_PL_BOOST_ALT5", 0.95f);
    p.boost_alt[6] = env_f("MUSE_PL_BOOST_ALT6", 1.15f);
    p.boost_d1[0] = 0.0f;
    p.boost_d1[1] = env_f("MUSE_PL_BOOST_D1_1", 0.0f);
    p.boost_d1[2] = env_f("MUSE_PL_BOOST_D1_2", 0.0f);
    p.boost_d1[3] = env_f("MUSE_PL_BOOST_D1_3", 0.0f);
    p.boost_d1[4] = env_f("MUSE_PL_BOOST_D1_4", 0.55f);
    p.boost_d1[5] = env_f("MUSE_PL_BOOST_D1_5", 0.75f);
    p.boost_d1[6] = env_f("MUSE_PL_BOOST_D1_6", 0.90f);
    return p;
  }();
  static const BgEbKnobs bg_env = []() -> BgEbKnobs {
    auto env_i = [](const char* name, int def) -> int {
      const char* s = std::getenv(name);
      return (s && *s) ? static_cast<int>(std::strtol(s, nullptr, 10)) : def;
    };
    auto env_f = [](const char* name, float def) -> float {
      const char* s = std::getenv(name);
      return (s && *s) ? static_cast<float>(std::strtod(s, nullptr)) : def;
    };
    BgEbKnobs b{};
    b.bg_mode = env_i("MUSE_BG_MODE", 1);
    b.bg_sub = env_i("MUSE_BG_SUB", 1);
    b.bg_off = env_i("MUSE_BG_OFF", 1);
    b.chp_bg2 = env_i("MUSE_BG_CHP_BG2", 1);
    b.bg_alpha = env_f("MUSE_BG_ALPHA", 4.0f);
    b.bg_min_p = env_f("MUSE_BG_MIN_P", 0.10f);
    b.sub_min_p = env_f("MUSE_BG_SUB_MIN_P", 0.25f);
    b.tri_min_p = env_f("MUSE_BG_TRI_MIN_P", 0.06f);
    b.f2 = env_f("MUSE_BG_F2", 0.50f);
    b.w2 = env_f("MUSE_BG_W2", 3.00f);
    b.f3 = env_f("MUSE_BG_F3", 1.00f);
    b.w3 = env_f("MUSE_BG_W3", 1.50f);
    b.uniq_mode = env_i("MUSE_UNIQ_MODE", 1);
    b.uniq_p2 = env_f("MUSE_UNIQ_P2", 0.42f);
    b.uniq_p3 = env_f("MUSE_UNIQ_P3", 0.70f);
    b.shf_cal = env_i("MUSE_SHF_CAL", 1);
    b.shf_lo = env_f("MUSE_SHF_LO", 0.015f);
    b.shf_hi = env_f("MUSE_SHF_HI", 0.11f);
    b.eb_mode = env_i("MUSE_EB_MODE", 1);
    b.eb_min_r = env_i("MUSE_EB_MIN_R", 4);
    b.eb_p0 = env_f("MUSE_EB_P0", 0.72f);
    b.eb_tau = env_f("MUSE_EB_TAU", 6.0f);
    b.eb_scale = env_f("MUSE_EB_SCALE", 1.0f);
    b.eb_lo = env_f("MUSE_EB_LO", -0.50f);
    b.eb_hi = env_f("MUSE_EB_HI", 0.0f);
    return b;
  }();
  static const TrKnobs tr_env = []() -> TrKnobs {
    auto env_i = [](const char* name, int def) -> int {
      const char* s = std::getenv(name);
      return (s && *s) ? static_cast<int>(std::strtol(s, nullptr, 10)) : def;
    };
    auto env_f = [](const char* name, float def) -> float {
      const char* s = std::getenv(name);
      return (s && *s) ? static_cast<float>(std::strtod(s, nullptr)) : def;
    };
    TrKnobs t{};
    t.mode = env_i("MUSE_TR_MODE", 1);
    t.cont_mode = env_i("MUSE_TR_CONT_MODE", 1);
    t.det_sh_root = env_i("MUSE_DET_SH_ROOT", 1);
    t.max_tr_dp = env_i("MUSE_TR_MAX_DP", 5);
    t.b_t6_tri = env_f("MUSE_TR_B_T6_TRI", 1.40f);
    t.b_t6_bg = env_f("MUSE_TR_B_T6_BG", 0.95f);
    t.b_t6_near = env_f("MUSE_TR_B_T6_NEAR", 0.95f);
    t.oov_tri_p = env_f("MUSE_TR_OOV_TRI_P", 0.60f);
    t.oov_a1_bg_p = env_f("MUSE_TR_OOV_A1_BG_P", 0.55f);
    t.oov_a1_pos2_p = env_f("MUSE_TR_OOV_A1_POS2_P", 0.48f);
    t.oov_fresh_p = env_f("MUSE_TR_OOV_FRESH_P", 0.48f);
    t.det_tri3_root_p = env_f("MUSE_DET_TRI3_ROOT_P", 0.65f);
    t.det_bg2_root_p = env_f("MUSE_DET_BG2_ROOT_P", 0.48f);
    t.ch_bg_t6 = env_f("MUSE_TR_CH_BG_T6", 0.86f);
    t.ch_bg_oov = env_f("MUSE_TR_CH_BG_OOV", 0.70f);
    t.ch_bg1_t6 = env_f("MUSE_TR_CH_BG1_T6", 0.78f);
    t.ch_near_t6 = env_f("MUSE_TR_CH_NEAR_T6", 0.82f);
    t.ch_near_oov = env_f("MUSE_TR_CH_NEAR_OOV", 0.62f);
    t.ch_t6_agree = env_f("MUSE_TR_CH_T6_AGREE", 0.72f);
    return t;
  }();
  int32_t* row_ctx_ptr = nullptr;
  if (saved_row_ctx.has_value() && saved_row_ctx.value().defined() &&
      saved_row_ctx.value().numel() >= tree_m * 4) {
    TORCH_CHECK(saved_row_ctx.value().is_cuda() && saved_row_ctx.value().scalar_type() == torch::kInt32 &&
                saved_row_ctx.value().is_contiguous());
    row_ctx_ptr = saved_row_ctx.value().data_ptr<int32_t>();
  }
  const int32_t* tr_log_ptr = nullptr;
  int32_t* tr_state_ptr = nullptr;
  int tr_cap = 0;
  if (tr_log.has_value() && tr_log.value().defined() && tr_log.value().numel() >= 8 &&
      tr_state.has_value() && tr_state.value().defined() && tr_state.value().numel() >= 4) {
    TORCH_CHECK(tr_log.value().is_cuda() && tr_log.value().scalar_type() == torch::kInt32 &&
                tr_log.value().is_contiguous());
    TORCH_CHECK(tr_state.value().is_cuda() && tr_state.value().scalar_type() == torch::kInt32 &&
                tr_state.value().is_contiguous());
    tr_log_ptr = tr_log.value().data_ptr<int32_t>();
    tr_state_ptr = tr_state.value().data_ptr<int32_t>();
    tr_cap = static_cast<int>(tr_log.value().numel() / 8);
  }
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  native_leaf_setup_tree_kernel<<<1, 256, 0, st>>>(
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
      calib_env,
      ck_env,
      hist_ptr,
      hist_len_ptr,
      hist_cap,
      saved_pos0_ptr,
      stats_ptr,
      stats_len,
      pl_env,
      bg_env,
      tr_env,
      row_ctx_ptr,
      tr_log_ptr,
      tr_state_ptr,
      tr_cap);
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
    c10::optional<torch::Tensor> saved_leaf_parent,
    c10::optional<torch::Tensor> hist,
    c10::optional<torch::Tensor> hist_len,
    c10::optional<torch::Tensor> saved_pos0,
    c10::optional<torch::Tensor> saved_row_ctx,
    c10::optional<torch::Tensor> tr_log,
    c10::optional<torch::Tensor> tr_state) {
  TORCH_CHECK(tree_m >= 8 && tree_m <= 16, "tree_m must be in 8..16");
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
  int32_t* hist_ptr = nullptr;
  int32_t* hist_len_ptr = nullptr;
  int hist_cap = 0;
  if (hist.has_value() && hist.value().defined() && hist.value().numel() > 0 &&
      hist_len.has_value() && hist_len.value().defined() && hist_len.value().numel() >= 1) {
    TORCH_CHECK(hist.value().is_cuda() && hist.value().scalar_type() == torch::kInt32 && hist.value().is_contiguous());
    TORCH_CHECK(hist_len.value().is_cuda() && hist_len.value().scalar_type() == torch::kInt32);
    hist_ptr = hist.value().data_ptr<int32_t>();
    hist_len_ptr = hist_len.value().data_ptr<int32_t>();
    hist_cap = static_cast<int>(hist.value().numel());
  }
  const int64_t* saved_pos0_ptr = nullptr;
  if (saved_pos0.has_value() && saved_pos0.value().defined() && saved_pos0.value().numel() >= 1) {
    TORCH_CHECK(saved_pos0.value().is_cuda() && saved_pos0.value().scalar_type() == torch::kInt64);
    saved_pos0_ptr = saved_pos0.value().data_ptr<int64_t>();
  }
  int32_t* row_ctx_ptr = nullptr;
  if (saved_row_ctx.has_value() && saved_row_ctx.value().defined() &&
      saved_row_ctx.value().numel() >= tree_m * 4) {
    TORCH_CHECK(saved_row_ctx.value().is_cuda() && saved_row_ctx.value().scalar_type() == torch::kInt32 &&
                saved_row_ctx.value().is_contiguous());
    row_ctx_ptr = saved_row_ctx.value().data_ptr<int32_t>();
  }
  int32_t* tr_log_ptr = nullptr;
  int32_t* tr_state_ptr = nullptr;
  int tr_cap = 0;
  if (tr_log.has_value() && tr_log.value().defined() && tr_log.value().numel() >= 8 &&
      tr_state.has_value() && tr_state.value().defined() && tr_state.value().numel() >= 4) {
    TORCH_CHECK(tr_log.value().is_cuda() && tr_log.value().scalar_type() == torch::kInt32 &&
                tr_log.value().is_contiguous());
    TORCH_CHECK(tr_state.value().is_cuda() && tr_state.value().scalar_type() == torch::kInt32 &&
                tr_state.value().is_contiguous());
    tr_log_ptr = tr_log.value().data_ptr<int32_t>();
    tr_state_ptr = tr_state.value().data_ptr<int32_t>();
    tr_cap = static_cast<int>(tr_log.value().numel() / 8);
  }
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
      high_acc_cpc_env,
      hist_ptr,
      hist_len_ptr,
      hist_cap,
      saved_pos0_ptr,
      row_ctx_ptr,
      tr_log_ptr,
      tr_state_ptr,
      tr_cap);
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
