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
#include <cuda_device_runtime_api.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdint.h>
#include <unordered_map>

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
constexpr int kMaxTreeK = 8;

__constant__ float kCalA[6] = {1.675690f, 1.258784f, 1.070196f, 1.171448f, 1.102070f, 0.731610f};
__constant__ float kCalC[8] = {
    0.000000f, -0.402942f, -0.763639f, -0.817452f,
    -1.416669f, -1.642223f, -1.860000f, -2.060000f};
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
  int m32_wide;         // 1=M=32 tiered copy-chain depths (dp=3/4/6; default 1)
  int sh_branch_cont;   // 1=256-thread branch-conditioned s_h continuation scan (default 1)
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
  // Downward-bounded online Empirical-Bayes a_shift from req_acc_state:
  //   [rounds, committed_ns, high_acc, raw_acc_drafts, raw_trials]
  int eb_mode;          // 0=off, 1=enabled
  int eb_raw;           // 1=use unbiased raw-chain geometric MLE req_acc_state[3]/req_acc_state[4] (default 1)
  int eb_min_r;
  float eb_p0;
  float eb_tau;
  float eb_scale;
  float eb_lo;
  float eb_hi;
};

__device__ __forceinline__ float compute_eb_delta_a(
    const BgEbKnobs& bg,
    const int32_t* req_acc_state,
    int req_acc_state_len) {
  if (bg.eb_mode == 0 || req_acc_state == nullptr) return 0.0f;
  const int32_t rr = req_acc_state[0];
  const int32_t cc = req_acc_state[1];
  if (rr < bg.eb_min_r || cc <= 0) return 0.0f;
  float p_hat = 0.0f;
  if (bg.eb_raw != 0 && req_acc_state_len >= 5 && req_acc_state[4] > 0) {
    p_hat = static_cast<float>(req_acc_state[3]) / static_cast<float>(req_acc_state[4]);
  } else {
    const float mean_ns = static_cast<float>(cc) / static_cast<float>(rr);
    const float den_m = (mean_ns - 0.35f > 0.25f) ? (mean_ns - 0.35f) : 0.25f;
    p_hat = (mean_ns - 1.35f) / den_m;
  }
  if (p_hat < 0.35f) p_hat = 0.35f;
  if (p_hat > 0.92f) p_hat = 0.92f;
  const float p0 = bg.eb_p0;
  const float p_post = (static_cast<float>(rr) * p_hat + bg.eb_tau * p0) /
                       (static_cast<float>(rr) + bg.eb_tau);
  const float da = (__logf(p_post / (1.0f - p_post)) - __logf(p0 / (1.0f - p0))) * bg.eb_scale;
  return (da < bg.eb_lo) ? bg.eb_lo : ((da > bg.eb_hi) ? bg.eb_hi : da);
}

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
  float b_d1_tri;        // log-odds boost to DFlash top-1 (rk==0) on monomorphic 30B recycled trigram match
  float b_d1_bg;         // log-odds boost to DFlash top-1 (rk==0) on tier-0 confirmed 30B recycled bigram match
  float ch_a1_bg_t6;     // child continuation prob on fresh age==1 (dist<=8) 30B bigram in DFlash top-K
  float ch_a1_bg_oov;    // child continuation prob on fresh age==1 (dist<=8) 30B OOV bigram
  int m32_wide;          // 1=un-throttle TR depth & conservative det_bg2/tri1 thresholds for tree_m>16 (default 1)
};

struct AntiChimeraKnobs {
  int ac_mode;           // 0=off, 1=0.00ms transition-coherence chimera penalty enabled
  int min_u;             // min history count N(u) for forward contradiction penalty
  float fwd_lambda;      // forward penalty weight when N(u)>=min_u && N(u,v)==0
  float fwd_alpha;       // log smoothing for N(u)
  float fwd_cap;         // max forward log-odds penalty
  int min_v;             // min incoming history count N_in(v) for exclusive-owner penalty
  float rev_lambda;      // reverse/exclusive-owner penalty when N(u_ref,v)>0 && N(u,v)==0
  float rev_alpha;       // smoothing for N_in(v)
  float rev_cap;         // max reverse log-odds penalty
  float ent_lambda;      // entropy-coupling chimera discount on unverified off-chain sub (rp==0)
  float lh_copy_w;       // 1-step copy-chain lookahead tie-breaker weight (non-compounding)
  float stutter_scale;   // multiplier on unconfirmed [tok -> tok] stutter sub continuation
  float skip_root_bg;    // left-shift (d_{j+2}) root weight on M>16 when (anc_j -> d_{j+2}) in history
  float skip_lh_w;       // lookahead tie-breaker weight for left-shift root matching d_{j+2}
  float skip_coef;       // left-shift child continuation scale * s_p1[k_r + 1]
  float shf_unseen_scale;// insertion-shift unseen probability scale * s_p1[k_r - 1] on M>16
};

struct CascadeKnobs {
  int mode;             // 0=off, 1=enabled
  int denoise_mode;     // 1=enable Action 2 (main-chain prefix denoising at m in {2,3}; default 0)
  int force_all;        // 1=diagnostic force trigger on every eligible step
  int max_b;            // max bifurcation slot b in 0..max_b (default 3 -> slots 0..3)
  int max_r;            // max runner-up rank r in 1..max_r (default 2 -> ranks #2..#3)
  float min_p;          // min effective reach probability for branch (default 0.09)
  float min_qrel;       // min scale-invariant within-top6 runner-up share q_rel(b,r) (default 0.15)
  float min_ev;         // min expected extra token EV to trigger Pass 2 (default 0.22)
  float branch_gain;    // expected extra tokens per remaining slot (5 - b) (default 0.58)
  float denoise_min_a;  // min prefix reachability A_m to trigger denoise (default 0.50)
  float denoise_max_p1; // max p_1(m) fog threshold to trigger denoise (default 0.60)
  float denoise_gain;   // expected extra tokens per remaining slot (6 - m) * fog (default 0.44)
  float ddtree_w;       // DDTree local margin weight in [0, 1] (default 0.35)
  float lookahead_w;    // 1-step lookahead priority tie-breaker for kind==8 nodes (default 0.30)
  float backup_w;       // Pass-1 backup branch floor weight on mode==2 divergence (default 0.45)
};

struct CascadePrepArgs {
  int32_t* input_ids;
  const int64_t* query_positions;
  const int64_t* draft_tokens;
  const int64_t* tree_top;
  const float* tree_vals;
  const int64_t* alt_ids;
  const float* alt_margins;
  const int32_t* req_acc_state;
  int req_acc_state_len;
  int32_t* cascade_meta;
  int64_t* p1_save_draft;
  int64_t* p1_save_top;
  float* p1_save_vals;
  int64_t* p1_save_alt;
  float* p1_save_margins;
  int64_t* stats;
  int stats_len;
  int32_t mask_token_id;
  int calib_mode;
  CalibKnobs ck;
  BgEbKnobs bg;
  CascadeKnobs csk;
  cudaGraphConditionalHandle cond_handle;
  int has_cond_handle;
};

__global__ void __launch_bounds__(64) dflash_cascade_gate_prep_kernel(CascadePrepArgs a) {
  const int tid = threadIdx.x;
  if (tid < 36) {
    a.p1_save_top[tid] = a.tree_top[tid];
    a.p1_save_vals[tid] = a.tree_vals[tid];
  }
  if (tid < 6) {
    a.p1_save_draft[tid] = a.draft_tokens[tid];
  }
  if (tid < 4 && a.alt_ids != nullptr && a.p1_save_alt != nullptr) {
    a.p1_save_alt[tid] = a.alt_ids[tid];
  }
  if (tid < 2 && a.alt_margins != nullptr && a.p1_save_margins != nullptr) {
    a.p1_save_margins[tid] = a.alt_margins[tid];
  }
  __syncthreads();

  const bool high_acc = (a.req_acc_state != nullptr && a.req_acc_state[2] != 0);
  if (a.csk.mode == 0 || high_acc) {
    if (tid == 0) {
      a.cascade_meta[0] = 0;
      if (a.has_cond_handle != 0) {
        cudaGraphSetConditional(a.cond_handle, 0u);
      }
    }
    return;
  }

  __shared__ float s_z0[6];
  __shared__ float s_margin[6];
  __shared__ uint8_t s_valid[6];
  __shared__ float s_pr[6][6];
  __shared__ float s_qrel[6][6];

  if (tid < 6) {
    const int base = tid * 6;
    const int64_t dk = a.p1_save_draft[tid];
    const int64_t top1 = a.p1_save_top[base + 0];
    const float z0 = a.p1_save_vals[base + 0];
    const bool ok = (dk >= 0 && dk < 202048 && top1 == dk && z0 > -1e20f);
    s_valid[tid] = ok ? 1 : 0;
    s_z0[tid] = z0;
    const int64_t top2 = a.p1_save_top[base + 1];
    const float z1 = a.p1_save_vals[base + 1];
    s_margin[tid] = (ok && top2 >= 0 && z1 > -1e20f) ? (z0 - z1) : 2.5f;
  }
  __syncthreads();

  if (tid < 6) {
    const int base = tid * 6;
    if (s_valid[tid] != 0) {
      const float eb_delta_a = compute_eb_delta_a(a.bg, a.req_acc_state, a.req_acc_state_len);
      const float sm_mean = (s_margin[0] + s_margin[1] + s_margin[2] +
                             s_margin[3] + s_margin[4] + s_margin[5]) * (1.0f / 6.0f);
      float sm_c = ((a.csk.ddtree_w > 0.0f)
                        ? (a.csk.ddtree_w * s_margin[tid] + (1.0f - a.csk.ddtree_w) * sm_mean)
                        : sm_mean) - 2.5f;
      if (sm_c < -2.0f) sm_c = -2.0f;
      if (sm_c > 2.5f) sm_c = 2.5f;
      float z0_c = s_z0[tid] - a.ck.z0_center;
      if (z0_c < -10.0f) z0_c = -10.0f;
      if (z0_c > 6.0f) z0_c = 6.0f;
      const float a_pre = kCalA[tid] + a.ck.a_shift + eb_delta_a +
                          a.ck.w_sm_pre * sm_c + a.ck.w_z0_pre * z0_c;
      float e0[6];
      float erel[6];
      float den0 = 1.0f;
      float den_rel = 0.0f;
#pragma unroll
      for (int rk = 0; rk < 6; ++rk) {
        const int64_t tok_r = a.p1_save_top[base + rk];
        const float zr = a.p1_save_vals[base + rk];
        if (rk == 0 || (tok_r >= 0 && tok_r < 202048 && zr > -1e20f)) {
          const float g = kCalC[rk] + a.ck.beta * (zr - s_z0[tid]);
          erel[rk] = __expf(g);
          e0[rk] = __expf(a_pre + g);
        } else {
          erel[rk] = 0.0f;
          e0[rk] = 0.0f;
        }
        den_rel += erel[rk];
        den0 += e0[rk];
      }
      const float inv_rel = (den_rel > 0.0f) ? (1.0f / den_rel) : 0.0f;
#pragma unroll
      for (int rk = 0; rk < 6; ++rk) {
        s_pr[tid][rk] = e0[rk] / den0;
        s_qrel[tid][rk] = erel[rk] * inv_rel;
      }
    } else {
#pragma unroll
      for (int rk = 0; rk < 6; ++rk) {
        s_pr[tid][rk] = 0.0f;
        s_qrel[tid][rk] = 0.0f;
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
    if (a.stats != nullptr && a.stats_len >= 48) {
      a.stats[40] += 1;
    }
    float A[6];
    float A_rel[6];
    float acc = 1.0f;
    float acc_rel = 1.0f;
    for (int j = 0; j < 6; ++j) {
      A[j] = acc;
      A_rel[j] = acc_rel;
      if (s_valid[j] != 0) {
        acc *= s_pr[j][0];
        const float p0_eff = (s_pr[j][0] > 0.80f * s_qrel[j][0]) ? s_pr[j][0] : (0.80f * s_qrel[j][0]);
        acc_rel *= p0_eff;
      } else {
        acc = 0.0f;
        acc_rel = 0.0f;
      }
    }

    float best_ev = -1.0f;
    int best_mode = 0;
    int best_b = 0;
    int best_r = 1;
    int64_t best_tok = -1;

    const int max_b = (a.csk.max_b < 0) ? 0 : ((a.csk.max_b > 4) ? 4 : a.csk.max_b);
    const int max_r = (a.csk.max_r < 1) ? 1 : ((a.csk.max_r > 5) ? 5 : a.csk.max_r);

    // Action 1: Scale-invariant Multi-Future Branch (b in 0..max_b, r in 1..max_r).
    for (int b = 0; b <= max_b; ++b) {
      if (s_valid[b] == 0 || !(A_rel[b] >= 0.18f)) break;
      const int64_t dk_b = a.p1_save_draft[b];
      for (int r = 1; r <= max_r; ++r) {
        const int64_t alt_tok = a.p1_save_top[b * 6 + r];
        if (alt_tok < 0 || alt_tok >= 202048 || alt_tok == dk_b) continue;
        bool dup = false;
        for (int pr = 1; pr < r; ++pr) {
          if (a.p1_save_top[b * 6 + pr] == alt_tok) {
            dup = true;
            break;
          }
        }
        if (dup) continue;
        const float qr = s_qrel[b][r];
        if (qr < a.csk.min_qrel) continue;
        const float pr_eff = (s_pr[b][r] > 0.78f * qr) ? s_pr[b][r] : (0.78f * qr);
        const float p_reach = A_rel[b] * pr_eff;
        if (p_reach >= a.csk.min_p) {
          const float ev = p_reach * (a.csk.branch_gain * static_cast<float>(5 - b));
          if (ev > best_ev) {
            best_ev = ev;
            best_mode = 1;
            best_b = b;
            best_r = r;
            best_tok = alt_tok;
          }
        }
      }
    }

    // Action 2: Iterative Main-Chain Prefix Denoising at m in {2, 3}
    if (a.csk.denoise_mode != 0) {
      for (int m = 2; m <= 3; ++m) {
        if (s_valid[m] != 0 && A[m] >= a.csk.denoise_min_a && s_pr[m][0] <= a.csk.denoise_max_p1) {
          const float p1_next = (m + 1 < 6 && s_valid[m + 1] != 0) ? s_pr[m + 1][0] : s_pr[m][0];
          const float fog = 0.60f * (1.0f - s_pr[m][0]) + 0.40f * (1.0f - p1_next);
          const float ev = A[m] * fog * (a.csk.denoise_gain * static_cast<float>(6 - m));
          if (ev > best_ev) {
            best_ev = ev;
            best_mode = 2;
            best_b = m - 1;
            best_r = 0;
            best_tok = a.p1_save_draft[m - 1];
          }
        }
      }
    }

    if (a.csk.force_all != 0 && best_mode == 0 && s_valid[0] != 0 && a.p1_save_top[1] >= 0 &&
        a.p1_save_top[1] < 202048 && a.p1_save_top[1] != a.p1_save_draft[0]) {
      best_mode = 1;
      best_b = 0;
      best_r = 1;
      best_tok = a.p1_save_top[1];
      best_ev = 1.0f;
    }

    if (best_mode != 0 && best_tok >= 0 && best_tok < 202048 &&
        (best_ev >= a.csk.min_ev || a.csk.force_all != 0)) {
      for (int j = 0; j < best_b; ++j) {
        a.input_ids[1 + j] = static_cast<int32_t>(a.p1_save_draft[j]);
      }
      a.input_ids[1 + best_b] = static_cast<int32_t>(best_tok);
      for (int j = best_b + 1; j < 15; ++j) {
        a.input_ids[1 + j] = a.mask_token_id;
      }
      a.cascade_meta[0] = best_mode;
      a.cascade_meta[1] = best_b;
      a.cascade_meta[2] = best_r;
      a.cascade_meta[3] = static_cast<int32_t>(best_tok);
      a.cascade_meta[4] = static_cast<int32_t>(a.query_positions[0]);
      if (a.stats != nullptr && a.stats_len >= 48) {
        a.stats[(best_mode == 1) ? 41 : 42] += 1;
      }
      if (a.has_cond_handle != 0) {
        cudaGraphSetConditional(a.cond_handle, 1u);
      }
    } else {
      a.cascade_meta[0] = 0;
      if (a.has_cond_handle != 0) {
        cudaGraphSetConditional(a.cond_handle, 0u);
      }
    }
  }
}

__global__ void __launch_bounds__(64) dflash_cascade_post_restore_kernel(
    int32_t* __restrict__ input_ids,
    int64_t* __restrict__ draft_tokens,
    int64_t* __restrict__ tree_top,
    float* __restrict__ tree_vals,
    int64_t* __restrict__ alt_ids,
    float* __restrict__ alt_margins,
    const int32_t* __restrict__ cascade_meta,
    int64_t* __restrict__ cascade_toks,
    float* __restrict__ cascade_vals,
    const int64_t* __restrict__ p1_save_draft,
    const int64_t* __restrict__ p1_save_top,
    const float* __restrict__ p1_save_vals,
    const int64_t* __restrict__ p1_save_alt,
    const float* __restrict__ p1_save_margins,
    int32_t mask_token_id) {
  const int tid = threadIdx.x;
  if (tid < 15) {
    input_ids[1 + tid] = mask_token_id;
  }
  const int mode = cascade_meta[0];
  const int b_star = cascade_meta[1];
  if (mode == 1) {
    // Future-B branch: save Pass-2 top-6 into cascade_toks/vals and restore Pass-1 main chain.
    if (tid < 36) {
      cascade_toks[tid] = tree_top[tid];
      cascade_vals[tid] = tree_vals[tid];
    }
    __syncthreads();
    if (tid < 36) {
      tree_top[tid] = p1_save_top[tid];
      tree_vals[tid] = p1_save_vals[tid];
    }
    if (tid < 6) {
      draft_tokens[tid] = p1_save_draft[tid];
    }
    if (tid < 4 && alt_ids != nullptr && p1_save_alt != nullptr) {
      alt_ids[tid] = p1_save_alt[tid];
    }
    if (tid < 2 && alt_margins != nullptr && p1_save_margins != nullptr) {
      alt_margins[tid] = p1_save_margins[tid];
    }
  } else if (mode == 2) {
    // Main-chain prefix denoising at m = b_star + 1: save Pass-1 into cascade_toks/vals as backup,
    // restore prefix 0..m-1 from Pass-1, and keep Pass-2's denoised suffix m..5 on main chain!
    const int m = (b_star + 1 < 1) ? 1 : ((b_star + 1 > 5) ? 5 : (b_star + 1));
    if (tid < 36) {
      cascade_toks[tid] = p1_save_top[tid];
      cascade_vals[tid] = p1_save_vals[tid];
    }
    __syncthreads();
    if (tid < 6 * m) {
      tree_top[tid] = p1_save_top[tid];
      tree_vals[tid] = p1_save_vals[tid];
    }
    if (tid < m) {
      draft_tokens[tid] = p1_save_draft[tid];
    }
    if (tid < 4 && (tid / 2) < m && alt_ids != nullptr && p1_save_alt != nullptr) {
      alt_ids[tid] = p1_save_alt[tid];
    }
    if (tid < 2 && tid < m && alt_margins != nullptr && p1_save_margins != nullptr) {
      alt_margins[tid] = p1_save_margins[tid];
    }
  }
}

// Arbitrary-depth (1..6-level) dynamic multi-leaf tree setup (rows 7..tree_m-1), launched as <<<1, 256>>> (~3.5 us):
//   - Bit-packed remap inside int32[49] (1 + 16*3):
//       leaf_flag[1 + 3*r + 0] = seq_sub  (seqlen_k_eff = seqlen_k - seq_sub)
//       leaf_flag[1 + 3*r + 1] = base_sub (-1 for causal rows; else tree_m - k0 of root ancestor)
//       leaf_flag[1 + 3*r + 2] = remap_packed (5-bit slot j = physical row r_j in 0..31 for tok = p + k0 + j, j=0..5)
//   - Fused SMEM history scan (kPlWindowMax=4096) over committed token history `hist[0..p0]` +
//     per-request 30B Target Recycling Log `tr_log`:
//       1) n-gram + 30B target-recycle confirmation/boosting of DFlash top-6 candidates (r=0..5) at depths j=0..5,
//       2) standalone OOV prompt-lookup & monomorphic 30B recycled roots + multi-step continuation chains.
//   - selftest_mode!=0: places a full deep chain (6-level on M=16, 5-level on M=12) mirroring d1..d6.
static int leaf_pdl_mode() {
  static const int mode = []() -> int {
    const char* s = std::getenv("MUSE_LEAF_PDL");
    if (!s || !*s) return 3;
    return (std::atoi(s) != 0) ? 3 : 0;
  }();
  return mode;
}

template <typename... KArgs, typename... Args>
static inline void leaf_launch_pdl(
    void (*kernel)(KArgs...), unsigned blocks, unsigned threads, cudaStream_t st, int pdl_mode, Args... args) {
  cudaLaunchConfig_t lcfg = {};
  lcfg.gridDim = dim3(blocks, 1u, 1u);
  lcfg.blockDim = dim3(threads, 1u, 1u);
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
  TORCH_CHECK(err == cudaSuccess, "native_leaf launch failed: ", cudaGetErrorString(err));
}

__global__ void __launch_bounds__(256) native_leaf_setup_tree_kernel(
    int32_t* leaf_flag,
    const int64_t* tree_top,
    const float* tree_vals,
    const int32_t* req_acc_state,
    int32_t* input_ids,
    int64_t* positions,
    int64_t* saved_leaf_toks,
    int32_t* saved_leaf_k,
    int32_t* saved_leaf_parent,
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
    const int32_t* hist,
    const int32_t* hist_len,
    int hist_cap,
    int64_t* saved_pos0,
    int64_t* stats,
    int stats_len,
    PlKnobs pl,
    BgEbKnobs bg,
    TrKnobs tr,
    int32_t* saved_row_ctx,
    const int32_t* tr_log,
    int32_t* tr_state,
    int tr_cap,
    int vocab_size,
    AntiChimeraKnobs ac,
    int32_t* cascade_meta,
    const int64_t* cascade_toks,
    const float* cascade_vals,
    CascadeKnobs csk,
    int req_acc_state_len,
    int pdl_mode) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
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

  __shared__ int s_row_base_sub[32];
  __shared__ int s_row_depth[32];
  __shared__ uint32_t s_row_packed[32];

  const int num_leaves = tree_m - 7;
  if (dynamic_mode == 0 || selftest_mode != 0 || tree_vals == nullptr || tree_k < 6) {
    if (tid == 0) {
      if (cascade_meta != nullptr) {
        cascade_meta[0] = 0;
        cascade_meta[5] = 0;
        cascade_meta[6] = 0;
      }
      constexpr int kLeafK[9] = {1, 1, 2, 2, 3, 1, 3, 4, 5};
      constexpr int kLeafRank[9] = {1, 2, 1, 2, 1, 3, 2, 1, 1};
      bool any_active = false;
      for (int li = 0; li < num_leaves; ++li) {
        const int r = 7 + li;
        int k = (li < 9) ? kLeafK[li] : 1;
        int par = -1;
        int64_t cand = -1;
        bool valid = false;
        if (!high_acc) {
          if (selftest_mode != 0) {
            // Deep chain selftest: rows 7..12 mirror d1..d6 (depth 1..6), rows 13..31 mirror d1..d6 (depth 1).
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
            b_sub = s_row_base_sub[par];
            dp = s_row_depth[par] + 1;
            pkd = s_row_packed[par] | ((static_cast<uint32_t>(r) & 0x1Fu) << (5 * (dp - 1)));
          }
          s_row_base_sub[r] = b_sub;
          s_row_depth[r] = dp;
          s_row_packed[r] = pkd;
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
          s_row_base_sub[r] = -1;
          s_row_depth[r] = 0;
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
      leaf_flag[0] = any_active ? -1 : 0;
      if (saved_row_ctx != nullptr) {
        saved_row_ctx[3] = -1;
      }
    }
    if ((pdl_mode & 2) != 0) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  const int tk = (tree_k >= 8) ? 8 : 6;

  // Dynamic per-step GPU arbitrary-depth (1..6-level) tree allocator + fused prompt lookup (up to 192 candidates).
  __shared__ int32_t s_h[kPlWindowMax];
  __shared__ int32_t s_ctx[22];
  __shared__ int32_t s_top[6][kMaxTreeK];
  __shared__ uint32_t s_w_pl_key[8][6];
  __shared__ uint32_t s_w_cand_key[8][6][kMaxTreeK];
  __shared__ uint32_t s_pl_key[6];
  __shared__ uint32_t s_cand_key[6][kMaxTreeK];
  // Wave-62/69 smoothed Markov bigram/trigram coherence + 0.00ms Anti-Chimera incoming counts (~3.4 KiB SMEM).
  __shared__ uint32_t s_alt_cnt[5][kMaxTreeK];
  __shared__ uint32_t s_in_cnt[5][kMaxTreeK];
  __shared__ uint32_t s_bg_cnt[5][kMaxTreeK][kMaxTreeK];
  __shared__ uint32_t s_bg3_mask[5][kMaxTreeK];
  __shared__ uint32_t s_bg_key[5][kMaxTreeK][kMaxTreeK];
  __shared__ uint32_t s_shf_bg[5][kMaxTreeK];
  __shared__ uint32_t s_skip_bg[5];
  __shared__ int32_t s_uniq_min[5][kMaxTreeK];
  __shared__ int32_t s_uniq_max[5][kMaxTreeK];
  __shared__ int32_t s_uniq_hp[5][kMaxTreeK];
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
  __shared__ float s_tr_b_rk[6][kMaxTreeK];
  __shared__ int32_t s_tr_oov_tok[6];
  __shared__ float s_tr_oov_w[6];
  __shared__ int32_t s_leaf_tok_r[32];
  __shared__ int32_t s_leaf_prv_r[32];
  __shared__ int32_t s_leaf_prv2_r[32];
  __shared__ int64_t s_tr_ch_tok;
  __shared__ float s_tr_ch_p;
  __shared__ int64_t s_sh_ch_tok;
  __shared__ float s_sh_ch_p;
  __shared__ int s_sh_ch_hp;
  __shared__ int s_sh_ch_ml;
  __shared__ float s_sel_s;
  __shared__ int s_sel_idx;
  __shared__ float s_sel_prob;
  __shared__ int s_sel_hpos;
  __shared__ int s_sel_ml;
  __shared__ uint32_t s_w_tri_key[8];
  __shared__ int s_w_sh[10][8];
  __shared__ int s_w_tr[17][8];
  __shared__ int s_hist_ok;
  __shared__ int s_win;
  __shared__ int s_h_base;

  __shared__ float s_Z[6];
  __shared__ float s_A[6];
  __shared__ float s_p1[6];
  __shared__ float s_p_oov[6];
  __shared__ float s_p1post[6];
  __shared__ float s_p2post[6];
  __shared__ float s_pr[6][kMaxTreeK];
  __shared__ float s_e0[6][kMaxTreeK];
  __shared__ float s_e1[6][kMaxTreeK];
  __shared__ float s_p_post_vec[kMaxTreeK];
  __shared__ int32_t s_spawned[16];
  __shared__ float s_den1[6];
  __shared__ float s_z0[6];
  __shared__ float s_margin[6];
  __shared__ int64_t s_dk[6];
  __shared__ uint8_t s_valid_row[6];
  __shared__ int s_any_active;

  constexpr int kCandCap = 192;
  __shared__ float s_cand_score[kCandCap];
  __shared__ float s_cand_prob[kCandCap];
  __shared__ int64_t s_cand_tok[kCandCap];
  __shared__ int8_t s_cand_k[kCandCap];
  __shared__ int8_t s_cand_par[kCandCap];
  __shared__ int8_t s_cand_depth[kCandCap];
  __shared__ int8_t s_cand_kind[kCandCap];  // 0=1-lvl, 1=sub, 2=shf, 3=alt2, 4=pl_root, 5=pl_cont, 6=coh_cont, 7=tr, 8=casc_chain, 9=casc_alt, 10=skip-1
  __shared__ int16_t s_cand_hpos[kCandCap]; // index in s_h matching s_cand_tok (-1 if none)
  __shared__ int8_t s_cand_ml[kCandCap];    // history match length 0..16
  __shared__ int8_t s_cand_rk[kCandCap];    // rank 0..7 in row (k-1), or -1 if non-topK
  __shared__ int s_num_cands;
  __shared__ int s_casc_mode;
  __shared__ int s_casc_b;
  __shared__ int s_casc_r;
  __shared__ int64_t s_casc_top[6][2];
  __shared__ float s_casc_p1[6];
  __shared__ float s_casc_p2[6];

  // Pass 1: validate rows, gather context tokens & top-6 IDs, and verify fail-closed history + recycle + cascade sync.
  if (tid == 0) {
    const int64_t p0 = positions[0];
    int cm = 0, cb = 0, cr = 0;
    if (cascade_meta != nullptr) {
      const int raw_m = cascade_meta[0];
      const int raw_b = cascade_meta[1];
      const int raw_r = cascade_meta[2];
      const int32_t raw_tok = cascade_meta[3];
      const int32_t raw_pos0 = cascade_meta[4];
      cascade_meta[0] = 0;
      cascade_meta[5] = 0;
      cascade_meta[6] = 0;
      if (!high_acc && csk.mode != 0 && cascade_toks != nullptr && cascade_vals != nullptr &&
          (raw_m == 1 || raw_m == 2) && raw_b >= 0 && raw_b <= 4 &&
          raw_tok >= 0 && raw_tok < vocab_size &&
          static_cast<int32_t>(p0) == raw_pos0) {
        if (raw_m == 1 && raw_r >= 1 && raw_r <= 5 &&
            tree_top[raw_b * tree_k + raw_r] == static_cast<int64_t>(raw_tok)) {
          cm = 1;
          cb = raw_b;
          cr = raw_r;
        } else if (raw_m == 2 && raw_r == 0 && raw_b + 1 <= 5) {
          cm = 2;
          cb = raw_b;
          cr = 0;
        }
      }
    }
    s_casc_mode = cm;
    s_casc_b = cb;
    s_casc_r = cr;
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
  if (tid < 32) {
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
    for (int r = 0; r < kMaxTreeK; ++r) {
      const int64_t tv = (r < tk) ? tree_top[base + r] : -1;
      s_top[tid][r] = (tv >= 0) ? static_cast<int32_t>(tv) : -1;
      s_tr_b_rk[tid][r] = 0.0f;
    }
  } else if (tid == 6) {
    s_ctx[15] = input_ids[0];
  }
  for (int idx = tid; idx < 5 * kMaxTreeK * kMaxTreeK; idx += 256) {
    const int j = idx / (kMaxTreeK * kMaxTreeK);
    const int rem = idx - j * (kMaxTreeK * kMaxTreeK);
    const int r = rem / kMaxTreeK;
    const int rp = rem - r * kMaxTreeK;
    s_bg_cnt[j][r][rp] = 0u;
    s_bg_key[j][r][rp] = 0u;
  }
  if (tid < 5 * kMaxTreeK) {
    const int j = tid / kMaxTreeK;
    const int r = tid - j * kMaxTreeK;
    s_alt_cnt[j][r] = 0u;
    s_in_cnt[j][r] = 0u;
    s_bg3_mask[j][r] = 0u;
    s_shf_bg[j][r] = 0u;
    s_uniq_min[j][r] = 2147483647;
    s_uniq_max[j][r] = -1;
    s_uniq_hp[j][r] = -1;
  }
  if (tid < 5) {
    s_skip_bg[tid] = 0u;
  }
  __syncthreads();

  // Pass 2: collaborative 256-thread SMEM scan of history window for multi-depth suffix matches,
  // Markov bigram/trigram co-occurrence counts, Anti-Chimera incoming counts, shift-bigram flags,
  // unique OOV successors, and 30B recycle log.
  const bool scan_bg = (bg.bg_mode != 0 || bg.uniq_mode != 0 || bg.shf_cal != 0 || ac.ac_mode != 0);
  const bool scan_anc = (pl.mode != 0 || tr.mode != 0 || tr.det_sh_root != 0 || ac.skip_root_bg > 0.0f);
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
    uint32_t c_k[6][kMaxTreeK];
#pragma unroll
    for (int j = 0; j < 6; ++j) {
#pragma unroll
      for (int r = 0; r < kMaxTreeK; ++r) {
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
            if (j < 5 && ac.skip_root_bg > 0.0f && nxt == s_ctx[16 + j + 1]) {
              s_skip_bg[j] = 1u;
            }
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
              for (int r = 0; r < kMaxTreeK; ++r) {
                if (r < tk && nxt == s_top[j][r] && key > c_k[j][r]) {
                  c_k[j][r] = key;
                }
              }
            }
          }
        }
      }
      if (scan_bg) {
        if (ac.ac_mode != 0) {
#pragma unroll
          for (int j = 0; j < 5; ++j) {
#pragma unroll
            for (int rp = 0; rp < kMaxTreeK; ++rp) {
              if (rp < tk && nxt == s_top[j + 1][rp]) {
                atomicAdd(&s_in_cnt[j][rp], 1u);
              }
            }
          }
        }
        const int r_start = (ac.ac_mode != 0 || (bg.bg_sub != 0 && bg.bg_mode != 0)) ? 0 : 1;
#pragma unroll
        for (int j = 0; j < 5; ++j) {
          const int32_t anc_j = anchors[j];
          const int32_t dk_j = s_ctx[16 + j];
          for (int r = r_start; r < tk; ++r) {
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
              if (bg.bg_mode != 0 || ac.ac_mode != 0) {
                bool tri_checked = false;
                int ml_b = 1;
#pragma unroll
                for (int rp = 0; rp < kMaxTreeK; ++rp) {
                  if (rp < tk && nxt == s_top[j + 1][rp]) {
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

    // Pass 2b: warp-0 deterministic (__ballot_sync + __popc, zero atomicAdd) anti-echo compaction of active 30B recycle log entries into SMEM.
    if (warp == 0 && tr.mode != 0 && tr_log != nullptr && tr_state != nullptr && tr_cap > 0) {
      int n_log = tr_state[0];
      if (n_log > tr_cap) n_log = tr_cap;
      const int cur_st = s_tr_cur_step;
      const uint32_t lane_lt = (1u << lane) - 1u;
      int out_n = 0;
      for (int base_idx = n_log - 1; base_idx >= 0 && out_n < kTrActiveCap; base_idx -= 32) {
        const int idx = base_idx - lane;
        bool keep = false;
        int base8 = 0;
        int32_t st_t = 0, tier_t = 0, u_val = -1;
        if (idx >= 0) {
          base8 = idx * 8;
          st_t = tr_log[base8 + 5];
          tier_t = tr_log[base8 + 7];
          u_val = tr_log[base8 + 4];
          keep = ((tier_t == 0 || (cur_st - st_t) == 1) && u_val >= 0 && u_val < vocab_size);
        }
        const uint32_t msk = __ballot_sync(0xffffffffu, keep);
        const int dst = out_n + __popc(msk & lane_lt);
        if (keep && dst < kTrActiveCap) {
          s_tr_cur[dst] = tr_log[base8 + 1];
          s_tr_p1[dst] = tr_log[base8 + 2];
          s_tr_p2[dst] = tr_log[base8 + 3];
          s_tr_u[dst] = u_val;
          s_tr_st[dst] = st_t;
          s_tr_pos_tier[dst] = (tr_log[base8 + 6] << 1) | (tier_t & 1);
        }
        out_n += __popc(msk);
      }
      if (lane == 0) {
        s_tr_active_n = (out_n > kTrActiveCap) ? kTrActiveCap : out_n;
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
      for (int r = 0; r < kMaxTreeK; ++r) {
        const uint32_t cwm = __reduce_max_sync(0xffffffffu, c_k[j][r]);
        if (lane == 0) s_w_cand_key[warp][j][r] = cwm;
      }
    }
    __syncthreads();

    if (warp == 0) {
      if (lane == 0 && s_tr_active_n > kTrActiveCap) {
        s_tr_active_n = kTrActiveCap;
      }
      if (lane < 8) {
#pragma unroll
        for (int j = 0; j < 6; ++j) {
          const uint32_t fm = __reduce_max_sync(0xffu, s_w_pl_key[lane][j]);
          if (lane == 0) s_pl_key[j] = fm;
#pragma unroll
          for (int r = 0; r < kMaxTreeK; ++r) {
            const uint32_t cfm = __reduce_max_sync(0xffu, s_w_cand_key[lane][j][r]);
            if (lane == 0) s_cand_key[j][r] = cfm;
          }
        }
      }
      __syncwarp();
    }
  } else {
    if (warp == 0) {
      if (lane < 6) {
        s_pl_key[lane] = 0u;
#pragma unroll
        for (int r = 0; r < kMaxTreeK; ++r) {
          s_cand_key[lane][r] = 0u;
        }
      }
      __syncwarp();
    }
  }

  const bool pl_w32 = (tree_m > 16 && pl.m32_wide != 0);
  const bool tr_w32 = (tree_m > 16 && tr.m32_wide != 0);
  auto cand_lh = [&](int k_pos, int hpos_v, int ml_v) -> float {
    if (ac.lh_copy_w > 0.0f && k_pos < 6 && pl.mode != 0 && s_hist_ok != 0 &&
        hpos_v >= 0 && hpos_v + 1 < s_win && ml_v >= pl.cont_from_df_n) {
      const int nml = (ml_v + 1 > 6) ? 6 : (ml_v + 1);
      return 1.0f + ac.lh_copy_w * pl.cont_p[nml];
    }
    return 1.0f;
  };

  if (warp == 0) {
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
      int bg_c = 0, bg_t0_c = 0, bg_min = 2147483647, bg_max = -1;
      int tri_c = 0, tri_min = 2147483647, tri_max = -1;
      int near_c = 0, near_min = 2147483647, near_max = -1;
      int a1_pos_c = 0, a1_pos_min = 2147483647, a1_pos_max = -1;
      int a1_bg_c = 0, a1_bg_min = 2147483647, a1_bg_max = -1;
      int tot_a1 = 0, a1_v1 = 0, a1_v2 = 0, a1_v3 = 0, a1_v4 = 0, a1_v5 = 0, a1_v6 = 0, a1_v7 = 0;
      const int32_t t1 = s_top[j][1], t2 = s_top[j][2], t3 = s_top[j][3], t4 = s_top[j][4];
      const int32_t t5 = s_top[j][5], t6 = s_top[j][6], t7 = s_top[j][7];

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
          if (u_t == t6) a1_v6 += w;
          if (u_t == t7) a1_v7 += w;
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
          if (tier_t == 0) bg_t0_c += 1;
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
        bg_t0_c = __reduce_add_sync(0xffffffffu, bg_t0_c);
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
        a1_v6 = __reduce_add_sync(0xffffffffu, a1_v6);
        a1_v7 = __reduce_add_sync(0xffffffffu, a1_v7);

        if (lane == 0) {
          const uint32_t anc_u1 = s_anc_u1_cnt[j];
          const int32_t dk0 = s_top[j][0];
          if (dk0 >= 0) {
            float b0 = 0.0f;
            if (tri_c >= 1 && tri_min == tri_max && tri_min == dk0 && tr.b_d1_tri > b0) b0 = tr.b_d1_tri;
            if (bg_t0_c >= 1 && bg_min == bg_max && bg_min == dk0 && tr.b_d1_bg > b0) b0 = tr.b_d1_bg;
            s_tr_b_rk[j][0] = b0;
          }
#pragma unroll
          for (int rk = 1; rk < kMaxTreeK; ++rk) {
            if (rk < tk) {
              const int32_t cand = s_top[j][rk];
              if (cand >= 0) {
                const int a1_vr = (rk == 1) ? a1_v1 : ((rk == 2) ? a1_v2 : ((rk == 3) ? a1_v3 :
                                  ((rk == 4) ? a1_v4 : ((rk == 5) ? a1_v5 : ((rk == 6) ? a1_v6 : a1_v7)))));
                float b = 0.0f;
                if (tri_c >= 1 && tri_min == tri_max && tri_min == cand && tr.b_t6_tri > b) b = tr.b_t6_tri;
                if (bg_c >= 1 && bg_min == bg_max && bg_min == cand && tr.b_t6_bg > b) b = tr.b_t6_bg;
                if (near_c >= 1 && near_min == near_max && near_min == cand && u_min == u_max && anc_u1 <= 3u) {
                  if (tr.b_t6_near > b) b = tr.b_t6_near;
                } else if (tot_a1 >= 2 && a1_vr * 3 >= tot_a1 * 2 && anc_u1 <= 3u) {
                  if (tr.b_t6_near > b) b = tr.b_t6_near;
                }
                s_tr_b_rk[j][rk] = b;
              }
            }
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

  // Pass 3 (warp 0 only): calibrated probabilities + online Empirical-Bayes + DDTree (0 stack spill).
  if (tid < 6) {
    const int base = tid * tree_k;
    const float z0 = s_z0[tid];
    const bool row_ok = (s_valid_row[tid] != 0);
    const float eb_delta_a = (calib_mode != 0) ? compute_eb_delta_a(bg, req_acc_state, req_acc_state_len) : 0.0f;
    const float sm_mean = (s_margin[0] + s_margin[1] + s_margin[2] +
                           s_margin[3] + s_margin[4] + s_margin[5]) * (1.0f / 6.0f);
    if (row_ok) {
      if (calib_mode != 0) {
        float sm_c = ((csk.ddtree_w > 0.0f)
                          ? (csk.ddtree_w * s_margin[tid] + (1.0f - csk.ddtree_w) * sm_mean)
                          : sm_mean) - 2.5f;
        if (sm_c < -2.0f) sm_c = -2.0f;
        if (sm_c > 2.5f) sm_c = 2.5f;
        float z0_c = z0 - ck.z0_center;
        if (z0_c < -10.0f) z0_c = -10.0f;
        if (z0_c > 6.0f) z0_c = 6.0f;
        const float a_base = kCalA[tid] + ck.a_shift + eb_delta_a;
        const float a_pre = a_base + ck.w_sm_pre * sm_c + ck.w_z0_pre * z0_c;
        const float a_post = a_base + ck.a2 + ck.w_z0_post * z0_c;
        float den0 = 1.0f, den1 = 1.0f;
#pragma unroll
        for (int rk = 0; rk < kMaxTreeK; ++rk) {
          float v0 = 0.0f;
          float v1 = 0.0f;
          if (rk < tk) {
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
              v0 = __expf(a_pre + g0);
              v1 = __expf(a_post + g1);
            }
          }
          s_e0[tid][rk] = v0;
          s_e1[tid][rk] = v1;
          den0 += v0;
          den1 += v1;
        }
#pragma unroll
        for (int rk = 0; rk < kMaxTreeK; ++rk) {
          s_pr[tid][rk] = s_e0[tid][rk] / den0;
        }
        s_den1[tid] = den1;
        s_p1post[tid] = s_e1[tid][0] / den1;
        s_p2post[tid] = (s_e1[tid][1] / den1) * ck.alt2_scale;
        s_Z[tid] = den0;
        s_p1[tid] = s_pr[tid][0];
        s_p_oov[tid] = 1.0f / den0;
      } else {
        float sum_e = 1.0f;
        float last_e = 0.0f;
#pragma unroll
        for (int rk = 1; rk < kMaxTreeK; ++rk) {
          if (rk < tk) {
            const int64_t tok_r = tree_top[base + rk];
            const float zr = tree_vals[base + rk];
            if (tok_r >= 0 && zr > -1e20f) {
              const float er = __expf((zr - z0) * inv_temp);
              sum_e += er;
              last_e = er;
            }
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
        for (int rk = 0; rk < kMaxTreeK; ++rk) {
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
      for (int rk = 0; rk < kMaxTreeK; ++rk) {
        s_e1[tid][rk] = 0.0f;
      }
    }

    // Calibrate Pass-2 coherent Future-B (mode==1) or Pass-1 saved backup (mode==2) logits (0 stack arrays).
    s_casc_top[tid][0] = -1;
    s_casc_top[tid][1] = -1;
    s_casc_p1[tid] = 0.0f;
    s_casc_p2[tid] = 0.0f;
    if (s_casc_mode != 0 && cascade_toks != nullptr && cascade_vals != nullptr) {
      const int cbase = tid * 6;
      const int64_t ct0 = cascade_toks[cbase + 0];
      const float cz0 = cascade_vals[cbase + 0];
      if (ct0 >= 0 && ct0 < static_cast<int64_t>(vocab_size) && cz0 > -1e20f) {
        s_casc_top[tid][0] = ct0;
        const int64_t ct1 = cascade_toks[cbase + 1];
        const float cz1 = cascade_vals[cbase + 1];
        if (ct1 >= 0 && ct1 < static_cast<int64_t>(vocab_size) && ct1 != ct0 && cz1 > -1e20f) {
          s_casc_top[tid][1] = ct1;
        }
        const float cm_loc = (ct1 >= 0 && cz1 > -1e20f) ? (cz0 - cz1) : 2.5f;
        float c_sm = ((csk.ddtree_w > 0.0f)
                          ? (csk.ddtree_w * cm_loc + (1.0f - csk.ddtree_w) * sm_mean)
                          : sm_mean) - 2.5f;
        if (c_sm < -2.0f) c_sm = -2.0f;
        if (c_sm > 2.5f) c_sm = 2.5f;
        float c_z0 = cz0 - ck.z0_center;
        if (c_z0 < -10.0f) c_z0 = -10.0f;
        if (c_z0 > 6.0f) c_z0 = 6.0f;
        const int rel_j = (s_casc_mode == 1 && tid > s_casc_b) ? (tid - (s_casc_b + 1)) : tid;
        const float ca_pre = kCalA[rel_j] + ck.a_shift + eb_delta_a +
                             ck.w_sm_pre * c_sm + ck.w_z0_pre * c_z0;
        float ce0 = 0.0f, ce1 = 0.0f, cerel0 = 0.0f, cerel1 = 0.0f;
        float cden = 1.0f;
        float cden_rel = 0.0f;
#pragma unroll
        for (int rk = 0; rk < 6; ++rk) {
          const int64_t ctr = cascade_toks[cbase + rk];
          const float czr = cascade_vals[cbase + rk];
          float vr_rel = 0.0f, vr_e = 0.0f;
          if (rk == 0 || (ctr >= 0 && ctr < static_cast<int64_t>(vocab_size) && czr > -1e20f)) {
            const float cg = kCalC[rk] + ck.beta * (czr - cz0);
            vr_rel = __expf(cg);
            vr_e = __expf(ca_pre + cg);
          }
          if (rk == 0) { cerel0 = vr_rel; ce0 = vr_e; }
          else if (rk == 1) { cerel1 = vr_rel; ce1 = vr_e; }
          cden_rel += vr_rel;
          cden += vr_e;
        }
        const float cinv_rel = (cden_rel > 0.0f) ? (1.0f / cden_rel) : 0.0f;
        const float cp1_raw = ce0 / cden;
        const float cp2_raw = (s_casc_top[tid][1] >= 0) ? (ce1 / cden) : 0.0f;
        if (s_casc_mode == 1 && tid > s_casc_b) {
          const float cq1 = cerel0 * cinv_rel;
          const float cq2 = (s_casc_top[tid][1] >= 0) ? (cerel1 * cinv_rel) : 0.0f;
          s_casc_p1[tid] = (cp1_raw > 0.78f * cq1) ? cp1_raw : (0.78f * cq1);
          s_casc_p2[tid] = (cp2_raw > 0.72f * cq2) ? cp2_raw : (0.72f * cq2);
        } else {
          s_casc_p1[tid] = cp1_raw;
          s_casc_p2[tid] = cp2_raw;
        }
      }
    }
  }
  __syncwarp();

  const int num_1lvl = 6 * (tk - 1);
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
    s_num_cands = num_1lvl;
  }
  __syncwarp();

  // Clear all kCandCap slots across warp 0, then populate 6*(tk-1) 1-level DFlash roots.
#pragma unroll
  for (int off = 0; off < kCandCap; off += 32) {
    s_cand_score[tid + off] = -1.0f;
    s_cand_prob[tid + off] = -1.0f;
    s_cand_hpos[tid + off] = -1;
    s_cand_ml[tid + off] = 0;
    s_cand_rk[tid + off] = -1;
  }
  for (int ci = tid; ci < num_1lvl; ci += 32) {
    const int row_idx = ci / (tk - 1);                  // 0..5 -> depth k = row_idx + 1
    const int rank = 1 + (ci - row_idx * (tk - 1));     // 1..(tk-1) -> candidate rank #2..#tk
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
        const float cprob = s_A[row_idx] * pr;
        int hp_i = -1;
        int ml_i = 0;
        if (pl.mode != 0 && s_hist_ok != 0) {
          const uint32_t ck_j = s_cand_key[row_idx][rank];
          if (ck_j != 0u) {
            hp_i = static_cast<int>((ck_j & 0xFFFFu) + 1u);
            ml_i = static_cast<int>(ck_j >> 16);
            s_cand_hpos[ci] = static_cast<int16_t>(hp_i);
            s_cand_ml[ci] = static_cast<int8_t>(ml_i);
          }
        }
        float lh = cand_lh(row_idx + 1, hp_i, ml_i);
        if (ac.skip_lh_w > 0.0f && row_idx < 5 && s_valid_row[row_idx + 1] != 0 && cand == s_dk[row_idx + 1]) {
          lh *= (1.0f + ac.skip_lh_w * s_p1[row_idx + 1]);
        }
        s_cand_prob[ci] = cprob;
        s_cand_score[ci] = cprob * lh;
        s_cand_tok[ci] = cand;
        s_cand_k[ci] = static_cast<int8_t>(row_idx + 1);
        s_cand_par[ci] = -1;
        s_cand_depth[ci] = 1;
        s_cand_kind[ci] = 0;
        s_cand_rk[ci] = static_cast<int8_t>(rank);
      }
    }
  }
  __syncwarp();

  // Inject standalone OOV prompt-lookup, left-shift (M>16), & monomorphic 30B recycled roots.
  if (tid == 0 && (pl.mode != 0 || tr.mode != 0 || tr.det_sh_root != 0 || ac.skip_root_bg > 0.0f) &&
      s_hist_ok != 0 && s_win >= 2) {
    bool any_pl_root = false;
    bool any_boost = false;
    bool any_tr_boost = false;
    bool any_tr_root = false;
    for (int j = 0; j < 6; ++j) {
      if (!s_valid_row[j] || !(s_A[j] > 0.0f)) continue;
      for (int rk = 0; rk < tk; ++rk) {
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
          } else if (tr_w32 && s_anc_bg2_cnt[j] >= 2u && s_anc_bg2_min[j] >= 0 &&
                     s_anc_bg2_min[j] == s_anc_bg2_max[j] && s_anc_bg2_min[j] != dk32 &&
                     tr.det_bg2_root_p > 0.0f) {
            hpos = s_anc_bg2_hp[j];
            rw = 0.22f;
            t6w = 0.22f;
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
            for (int r = 0; r < tk - 1; ++r) {
              const int sl = j * (tk - 1) + r;
              if (s_cand_prob[sl] > -0.5f && s_cand_tok[sl] == tok) {
                ex_slot = sl;
                break;
              }
            }
            const float lh = cand_lh(j + 1, hpos, eff_ml);
            if (ex_slot >= 0) {
              if (s_cand_hpos[ex_slot] < 0 || eff_ml > static_cast<int>(s_cand_ml[ex_slot])) {
                s_cand_hpos[ex_slot] = static_cast<int16_t>(hpos);
                s_cand_ml[ex_slot] = static_cast<int8_t>(eff_ml);
              }
              if (pl.in_top6_floor != 0) {
                const float t6_s = s_A[j] * (1.0f - s_p1[j]) * t6w * j_scale;
                const float fl_s = (t6_s > pl_s) ? t6_s : pl_s;
                if (fl_s > s_cand_prob[ex_slot]) {
                  s_cand_prob[ex_slot] = fl_s;
                  any_pl_root = true;
                }
              }
              const float upd_sc = s_cand_prob[ex_slot] *
                  cand_lh(j + 1, static_cast<int>(s_cand_hpos[ex_slot]), static_cast<int>(s_cand_ml[ex_slot]));
              if (upd_sc > s_cand_score[ex_slot]) {
                s_cand_score[ex_slot] = upd_sc;
              }
            } else if (pl_s > 0.0f && s_num_cands < kCandCap) {
              const int slot = s_num_cands++;
              s_cand_prob[slot] = pl_s;
              s_cand_score[slot] = pl_s * lh;
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
      // Left-shift (skip-1) history-confirmed root on 25-leaf wide tree (M>16).
      if (tree_m > 16 && calib_mode != 0 && j <= 3 && s_valid_row[j + 1] != 0 &&
          ac.skip_root_bg > 0.0f && s_skip_bg[j] != 0u) {
        const int64_t sk_tok = s_dk[j + 1];
        if (sk_tok >= 0 && sk_tok != s_dk[j] && s_num_cands < kCandCap) {
          int ex_sk = -1;
          for (int ic = 0; ic < s_num_cands; ++ic) {
            if (s_cand_prob[ic] > -0.5f && static_cast<int>(s_cand_k[ic]) == j + 1 &&
                static_cast<int>(s_cand_par[ic]) == -1 && s_cand_tok[ic] == sk_tok) {
              ex_sk = ic;
              break;
            }
          }
          const float sk_p = s_A[j] * (1.0f - s_p1[j]) * ac.skip_root_bg;
          const float sk_lh = 1.0f + ac.skip_lh_w * s_p1[j + 1];
          if (ex_sk < 0 && sk_p > 0.0f) {
            const int slot = s_num_cands++;
            s_cand_prob[slot] = sk_p;
            s_cand_score[slot] = sk_p * sk_lh;
            s_cand_tok[slot] = sk_tok;
            s_cand_k[slot] = static_cast<int8_t>(j + 1);
            s_cand_par[slot] = -1;
            s_cand_depth[slot] = 1;
            s_cand_kind[slot] = 10;
            s_cand_hpos[slot] = -1;
            s_cand_ml[slot] = 0;
            s_cand_rk[slot] = -1;
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
            if (s_cand_prob[ic] > -0.5f && static_cast<int>(s_cand_k[ic]) == j + 1 &&
                static_cast<int>(s_cand_par[ic]) == -1 && s_cand_tok[ic] == otok) {
              ex_slot = ic;
              break;
            }
          }
          if (ex_slot >= 0) {
            if (tr_s > s_cand_prob[ex_slot]) {
              s_cand_prob[ex_slot] = tr_s;
              const float upd_sc = tr_s * cand_lh(
                  j + 1, static_cast<int>(s_cand_hpos[ex_slot]), static_cast<int>(s_cand_ml[ex_slot]));
              if (upd_sc > s_cand_score[ex_slot]) s_cand_score[ex_slot] = upd_sc;
              any_tr_root = true;
            }
          } else if (tr_s > 0.0f && s_num_cands < kCandCap) {
            const int slot = s_num_cands++;
            s_cand_score[slot] = tr_s;
            s_cand_prob[slot] = tr_s;
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
  // Seed Multi-Future Cascade root (mode==1 Future-B branch, or mode==2 first diverging Pass-1 backup).
  if (tid == 0 && s_casc_mode != 0) {
    if (s_casc_mode == 1) {
      const int ex = s_casc_b * (tk - 1) + (s_casc_r - 1);
      if (ex >= 0 && ex < num_1lvl && s_cand_prob[ex] > 0.0f) {
        const int b_base = s_casc_b * tree_k;
        const float bz0 = tree_vals[b_base + 0];
        float bden_rel = 0.0f;
        float ber_star = 0.0f;
        for (int rk = 0; rk < tk; ++rk) {
          const int64_t btok = tree_top[b_base + rk];
          const float bzr = tree_vals[b_base + rk];
          if (rk == 0 || (btok >= 0 && bzr > -1e20f)) {
            const float ev = __expf(kCalC[rk] + ck.beta * (bzr - bz0));
            bden_rel += ev;
            if (rk == s_casc_r) ber_star = ev;
          }
        }
        if (bden_rel > 0.0f) {
          const float b_floor = s_A[s_casc_b] * (0.72f * (ber_star / bden_rel));
          if (b_floor > s_cand_prob[ex]) {
            s_cand_prob[ex] = b_floor;
          }
        }
        s_cand_kind[ex] = 8;
        const float lh = (s_casc_b + 1 < 6) ? (1.0f + csk.lookahead_w * s_casc_p1[s_casc_b + 1]) : 1.0f;
        s_cand_score[ex] = s_cand_prob[ex] * lh;
      }
    } else if (s_casc_mode == 2) {
      int m_div = -1;
      for (int mj = s_casc_b + 1; mj < 6; ++mj) {
        if (s_valid_row[mj] && s_A[mj] > 0.0f && s_casc_top[mj][0] >= 0 && s_casc_top[mj][0] != s_dk[mj]) {
          m_div = mj;
          break;
        }
      }
      if (m_div >= 1) {
        if (cascade_meta != nullptr) {
          cascade_meta[6] = m_div + 1;
        }
        if (stats != nullptr && stats_len >= 48) {
          stats[46] += 1;
        }
        const int64_t old_tok = s_casc_top[m_div][0];
        const float p_bk = s_A[m_div] * csk.backup_w * (1.0f - s_p1[m_div]) * s_casc_p1[m_div];
        const float lh = (m_div + 1 < 6) ? (1.0f + csk.lookahead_w * s_casc_p1[m_div + 1]) : 1.0f;
        int ex_slot = -1;
        for (int r = 0; r < tk - 1; ++r) {
          const int sl = m_div * (tk - 1) + r;
          if (s_cand_prob[sl] > -0.5f && s_cand_tok[sl] == old_tok) {
            ex_slot = sl;
            break;
          }
        }
        if (ex_slot >= 0) {
          const float eff_p = (p_bk > s_cand_prob[ex_slot]) ? p_bk : s_cand_prob[ex_slot];
          s_cand_prob[ex_slot] = eff_p;
          s_cand_score[ex_slot] = eff_p * lh;
          s_cand_kind[ex_slot] = 8;
        } else if (p_bk > 0.0f && s_num_cands < kCandCap) {
          const int slot = s_num_cands++;
          s_cand_prob[slot] = p_bk;
          s_cand_score[slot] = p_bk * lh;
          s_cand_tok[slot] = old_tok;
          s_cand_k[slot] = static_cast<int8_t>(m_div + 1);
          s_cand_par[slot] = -1;
          s_cand_depth[slot] = 1;
          s_cand_kind[slot] = 8;
          s_cand_hpos[slot] = -1;
          s_cand_ml[slot] = 0;
          s_cand_rk[slot] = -1;
        }
      }
    }
  }
  __syncwarp();
  }  // end if (warp == 0) pre-loop setup
  __syncthreads();

  uint32_t casc_row_mask = 0u;
  const int eff_max_tr_dp = tr_w32 ? ((tr.max_tr_dp < 6) ? 6 : tr.max_tr_dp) : tr.max_tr_dp;
  for (int li = 0; li < num_leaves; ++li) {
    const int r = 7 + li;
    if (warp == 0) {
      const int nc = s_num_cands;
      float best_s_w = (tid < nc) ? s_cand_score[tid] : -1.0f;
      int best_idx_w = tid;
#pragma unroll
      for (int off = 32; off < kCandCap; off += 32) {
        const int idx_o = tid + off;
        if (idx_o < nc) {
          const float so = s_cand_score[idx_o];
          if (so > best_s_w) {
            best_s_w = so;
            best_idx_w = idx_o;
          }
        }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float os = __shfl_xor_sync(0xffffffffu, best_s_w, o);
        const int oi = __shfl_xor_sync(0xffffffffu, best_idx_w, o);
        if (os > best_s_w || (os == best_s_w && oi < best_idx_w)) {
          best_s_w = os;
          best_idx_w = oi;
        }
      }
      if (tid == 0) {
        s_sel_s = best_s_w;
        s_sel_idx = best_idx_w;
        s_sel_prob = (best_s_w > 0.0f) ? s_cand_prob[best_idx_w] : 0.0f;
        s_sel_hpos = (best_s_w > 0.0f) ? static_cast<int>(s_cand_hpos[best_idx_w]) : -1;
        s_sel_ml = (best_s_w > 0.0f) ? static_cast<int>(s_cand_ml[best_idx_w]) : 0;
        if (best_s_w > 0.0f) {
          const int32_t tok32 = static_cast<int32_t>(s_cand_tok[best_idx_w]);
          const int k_r = static_cast<int>(s_cand_k[best_idx_w]);
          const int par = static_cast<int>(s_cand_par[best_idx_w]);
          s_leaf_tok_r[r] = tok32;
          s_leaf_prv_r[r] = (par < 0) ? s_ctx[15 + k_r - 1] : s_leaf_tok_r[par];
          s_leaf_prv2_r[r] = (par < 0) ? s_ctx[15 + k_r - 2] : s_leaf_prv_r[par];
          s_tr_ch_tok = -1;
          s_tr_ch_p = 0.0f;
          s_sh_ch_tok = -1;
          s_sh_ch_p = 0.0f;
          s_sh_ch_hp = -1;
          s_sh_ch_ml = 0;
        }
      }
    }
    // Barrier #1: broadcasts s_sel_* and s_leaf_*_r[r] to all 256 threads.
    __syncthreads();

    const float best_s = s_sel_s;
    const int best_idx = s_sel_idx;
    const float best_prob_w = s_sel_prob;
    const int sel_hpos_w = s_sel_hpos;
    const int sel_ml_w = s_sel_ml;
    const bool can_scan = (best_s > 0.0f && li + 1 < num_leaves && calib_mode != 0 &&
                           s_hist_ok != 0 && saved_leaf_parent != nullptr);
    const int dp_w = can_scan ? static_cast<int>(s_cand_depth[best_idx]) : 99;
    const int k_w = can_scan ? static_cast<int>(s_cand_k[best_idx]) : 99;
    const bool want_sh_scan =
        can_scan && pl.sh_branch_cont != 0 && s_win >= 2 && dp_w < eff_max_depth && k_w < 6 &&
        best_prob_w >= 0.030f && (sel_hpos_w < 0 || sel_ml_w < 3);
    const bool want_tr_scan =
        can_scan && tr.mode != 0 && tr.cont_mode != 0 && s_tr_active_n > 0 &&
        dp_w < eff_max_depth && dp_w < eff_max_tr_dp && k_w < 6 && best_prob_w >= 0.015f;

    // Fused 256-thread (8-warp) s_h + 30B Target Recycling scan sharing ONE __syncthreads() barrier.
    if (want_sh_scan) {
      const int32_t tok32 = s_leaf_tok_r[r];
      const int32_t pv_r = s_leaf_prv_r[r];
      const int32_t pv2_r = s_leaf_prv2_r[r];
      const int win_m1 = s_win - 1;

      uint32_t best_tri_key = 0u;
      int u_c = 0, u_min = 2147483647, u_max = -1;
      int bg_c = 0, bg_min = 2147483647, bg_max = -1, bg_hp = -1;
      int tri_c = 0, tri_min = 2147483647, tri_max = -1;

      for (int hi = tid; hi < win_m1; hi += 256) {
        if (s_h[hi] != tok32) continue;
        const int32_t nxt = s_h[hi + 1];
        if (nxt < 0) continue;
        u_c += 1;
        if (nxt < u_min) u_min = nxt;
        if (nxt > u_max) u_max = nxt;

        if (pv_r >= 0 && hi >= 1 && s_h[hi - 1] == pv_r) {
          bg_c += 1;
          if (nxt < bg_min) bg_min = nxt;
          if (nxt > bg_max) bg_max = nxt;
          if (hi + 1 > bg_hp) bg_hp = hi + 1;

          if (pv2_r >= 0 && hi >= 2 && s_h[hi - 2] == pv2_r) {
            tri_c += 1;
            if (nxt < tri_min) tri_min = nxt;
            if (nxt > tri_max) tri_max = nxt;
            const uint32_t tk_key = (3u << 16) | static_cast<uint32_t>(hi);
            if (tk_key > best_tri_key) best_tri_key = tk_key;
          }
        }
      }

      const uint32_t w_tk = __reduce_max_sync(0xffffffffu, best_tri_key);
      const int w_uc = __reduce_add_sync(0xffffffffu, u_c);
      const int w_umin = __reduce_min_sync(0xffffffffu, u_min);
      const int w_umax = __reduce_max_sync(0xffffffffu, u_max);
      const int w_bgc = __reduce_add_sync(0xffffffffu, bg_c);
      const int w_bgmin = __reduce_min_sync(0xffffffffu, bg_min);
      const int w_bgmax = __reduce_max_sync(0xffffffffu, bg_max);
      const int w_bghp = __reduce_max_sync(0xffffffffu, bg_hp);
      const int w_tc = __reduce_add_sync(0xffffffffu, tri_c);
      const int w_tmin = __reduce_min_sync(0xffffffffu, tri_min);
      const int w_tmax = __reduce_max_sync(0xffffffffu, tri_max);
      if (lane == 0) {
        s_w_tri_key[warp] = w_tk;
        s_w_sh[0][warp] = w_uc;
        s_w_sh[1][warp] = w_umin;
        s_w_sh[2][warp] = w_umax;
        s_w_sh[3][warp] = w_bgc;
        s_w_sh[4][warp] = w_bgmin;
        s_w_sh[5][warp] = w_bgmax;
        s_w_sh[6][warp] = w_bghp;
        s_w_sh[7][warp] = w_tc;
        s_w_sh[8][warp] = w_tmin;
        s_w_sh[9][warp] = w_tmax;
      }
    }

    if (want_tr_scan) {
      const int32_t tok32 = s_leaf_tok_r[r];
      const int32_t pv_r = s_leaf_prv_r[r];
      const int32_t pv2_r = s_leaf_prv2_r[r];
      const int n_act = s_tr_active_n;
      const int cur_st = s_tr_cur_step;
      const int pos_leaf = static_cast<int>(positions[0]) + k_w;

      int u_tot_c = 0, t0_u_c = 0, u_min_c = 2147483647, u_max_c = -1;
      int bg_t0_c = 0, bg_min_c = 2147483647, bg_max_c = -1;
      int tri_t0_c = 0, tri_min_c = 2147483647, tri_max_c = -1;
      int a1_bg_c = 0, a1_bg_min = 2147483647, a1_bg_max = -1;
      int a1_tri_c = 0, a1_tri_min = 2147483647, a1_tri_max = -1;
      int near_c = 0;

      for (int idx = tid; idx < n_act; idx += 256) {
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
        if (pv_r >= 0 && p1_t == pv_r) {
          if (tier_t == 0) {
            bg_t0_c += 1;
            if (u_t < bg_min_c) bg_min_c = u_t;
            if (u_t > bg_max_c) bg_max_c = u_t;
            if (pv2_r >= 0 && p2_t == pv2_r) {
              tri_t0_c += 1;
              if (u_t < tri_min_c) tri_min_c = u_t;
              if (u_t > tri_max_c) tri_max_c = u_t;
            }
          }
          if (age == 1 && dist <= 8) {
            a1_bg_c += 1;
            if (u_t < a1_bg_min) a1_bg_min = u_t;
            if (u_t > a1_bg_max) a1_bg_max = u_t;
            if (pv2_r >= 0 && p2_t == pv2_r) {
              a1_tri_c += 1;
              if (u_t < a1_tri_min) a1_tri_min = u_t;
              if (u_t > a1_tri_max) a1_tri_max = u_t;
            }
          }
        }
      }

      const int w_utot = __reduce_add_sync(0xffffffffu, u_tot_c);
      const int w_t0u = __reduce_add_sync(0xffffffffu, t0_u_c);
      const int w_umin = __reduce_min_sync(0xffffffffu, u_min_c);
      const int w_umax = __reduce_max_sync(0xffffffffu, u_max_c);
      const int w_near = __reduce_add_sync(0xffffffffu, near_c);
      const int w_bgc = __reduce_add_sync(0xffffffffu, bg_t0_c);
      const int w_bgmin = __reduce_min_sync(0xffffffffu, bg_min_c);
      const int w_bgmax = __reduce_max_sync(0xffffffffu, bg_max_c);
      const int w_tric = __reduce_add_sync(0xffffffffu, tri_t0_c);
      const int w_trimin = __reduce_min_sync(0xffffffffu, tri_min_c);
      const int w_trimax = __reduce_max_sync(0xffffffffu, tri_max_c);
      const int w_a1bgc = __reduce_add_sync(0xffffffffu, a1_bg_c);
      const int w_a1bgmin = __reduce_min_sync(0xffffffffu, a1_bg_min);
      const int w_a1bgmax = __reduce_max_sync(0xffffffffu, a1_bg_max);
      const int w_a1tric = __reduce_add_sync(0xffffffffu, a1_tri_c);
      const int w_a1trimin = __reduce_min_sync(0xffffffffu, a1_tri_min);
      const int w_a1trimax = __reduce_max_sync(0xffffffffu, a1_tri_max);
      if (lane == 0) {
        s_w_tr[0][warp] = w_utot;
        s_w_tr[1][warp] = w_t0u;
        s_w_tr[2][warp] = w_umin;
        s_w_tr[3][warp] = w_umax;
        s_w_tr[4][warp] = w_near;
        s_w_tr[5][warp] = w_bgc;
        s_w_tr[6][warp] = w_bgmin;
        s_w_tr[7][warp] = w_bgmax;
        s_w_tr[8][warp] = w_tric;
        s_w_tr[9][warp] = w_trimin;
        s_w_tr[10][warp] = w_trimax;
        s_w_tr[11][warp] = w_a1bgc;
        s_w_tr[12][warp] = w_a1bgmin;
        s_w_tr[13][warp] = w_a1bgmax;
        s_w_tr[14][warp] = w_a1tric;
        s_w_tr[15][warp] = w_a1trimin;
        s_w_tr[16][warp] = w_a1trimax;
      }
    }

    // Barrier #2 (final barrier of leaf iteration): all 256 threads finished reading s_sel_* and writing s_w_sh/s_w_tr.
    __syncthreads();

    if (warp == 0) {
      int sh_u_c = 0, sh_u_min = 2147483647, sh_u_max = -1;
      int sh_bg_c = 0, sh_bg_min = 2147483647, sh_bg_max = -1;
      int sh_tri_c = 0, sh_tri_min = 2147483647, sh_tri_max = -1;

      if (want_sh_scan && lane < 8) {
        const uint32_t g_tri_key = __reduce_max_sync(0xffu, s_w_tri_key[lane]);
        const int g_u_c = __reduce_add_sync(0xffu, s_w_sh[0][lane]);
        const int g_u_min = __reduce_min_sync(0xffu, s_w_sh[1][lane]);
        const int g_u_max = __reduce_max_sync(0xffu, s_w_sh[2][lane]);
        const int g_bg_c = __reduce_add_sync(0xffu, s_w_sh[3][lane]);
        const int g_bg_min = __reduce_min_sync(0xffu, s_w_sh[4][lane]);
        const int g_bg_max = __reduce_max_sync(0xffu, s_w_sh[5][lane]);
        const int g_bg_hp = __reduce_max_sync(0xffu, s_w_sh[6][lane]);
        const int g_tri_c = __reduce_add_sync(0xffu, s_w_sh[7][lane]);
        const int g_tri_min = __reduce_min_sync(0xffu, s_w_sh[8][lane]);
        const int g_tri_max = __reduce_max_sync(0xffu, s_w_sh[9][lane]);
        if (lane == 0) {
          sh_u_c = g_u_c;
          sh_u_min = g_u_min;
          sh_u_max = g_u_max;
          sh_bg_c = g_bg_c;
          sh_bg_min = g_bg_min;
          sh_bg_max = g_bg_max;
          sh_tri_c = g_tri_c;
          sh_tri_min = g_tri_min;
          sh_tri_max = g_tri_max;
          const int j_ch = k_w;
          auto in_ch_topK = [&](int32_t cand_c) -> bool {
            if (!s_valid_row[j_ch]) return false;
#pragma unroll
            for (int rp = 0; rp < kMaxTreeK; ++rp) {
              if (rp < tk && s_top[j_ch][rp] == cand_c) return true;
            }
            return false;
          };
          // Anti-eviction M=32/M=16 rules: strictly monomorphic trigrams, no runaway hpos chaining on 1-shot bigrams/unigrams.
          if (g_tri_c >= 1 && g_tri_min == g_tri_max && g_tri_min >= 0 && g_tri_key != 0u) {
            const int hi_t = static_cast<int>(g_tri_key & 0xFFFFu);
            const bool in_tk = in_ch_topK(g_tri_min);
            s_sh_ch_tok = static_cast<int64_t>(g_tri_min);
            s_sh_ch_hp = hi_t + 1;
            s_sh_ch_ml = 3;
            s_sh_ch_p = (g_tri_c >= 2) ? (in_tk ? 0.72f : 0.58f) : (in_tk ? 0.62f : 0.45f);
          } else if (g_bg_c >= 2 && g_bg_min == g_bg_max && g_bg_min >= 0) {
            const bool in_tk = in_ch_topK(g_bg_min);
            s_sh_ch_tok = static_cast<int64_t>(g_bg_min);
            s_sh_ch_hp = g_bg_hp;
            s_sh_ch_ml = 2;
            s_sh_ch_p = in_tk ? 0.65f : 0.45f;
          } else if (g_bg_c == 1 && g_bg_min == g_bg_max && g_bg_min >= 0 && in_ch_topK(g_bg_min)) {
            s_sh_ch_tok = static_cast<int64_t>(g_bg_min);
            s_sh_ch_hp = -1;
            s_sh_ch_ml = 0;
            s_sh_ch_p = 0.48f;
          } else if (bg.uniq_mode != 0 && g_u_c >= 2 && g_u_min == g_u_max && g_u_min >= 0) {
            s_sh_ch_tok = static_cast<int64_t>(g_u_min);
            s_sh_ch_hp = -1;
            s_sh_ch_ml = 0;
            s_sh_ch_p = (g_u_c == 2) ? bg.uniq_p2 : bg.uniq_p3;
          }
        }
      }

      if (want_tr_scan) {
        int g_utot = 0, g_t0u = 0, g_umin = 2147483647, g_umax = -1, g_near = 0;
        int g_bgc = 0, g_bgmin = 2147483647, g_bgmax = -1;
        int g_tric = 0, g_trimin = 2147483647, g_trimax = -1;
        int g_a1_bgc = 0, g_a1_bgmin = 2147483647, g_a1_bgmax = -1;
        int g_a1_tric = 0, g_a1_trimin = 2147483647, g_a1_trimax = -1;
        if (lane < 8) {
          g_utot = __reduce_add_sync(0xffu, s_w_tr[0][lane]);
          g_t0u = __reduce_add_sync(0xffu, s_w_tr[1][lane]);
          g_umin = __reduce_min_sync(0xffu, s_w_tr[2][lane]);
          g_umax = __reduce_max_sync(0xffu, s_w_tr[3][lane]);
          g_near = __reduce_add_sync(0xffu, s_w_tr[4][lane]);
          g_bgc = __reduce_add_sync(0xffu, s_w_tr[5][lane]);
          g_bgmin = __reduce_min_sync(0xffu, s_w_tr[6][lane]);
          g_bgmax = __reduce_max_sync(0xffu, s_w_tr[7][lane]);
          g_tric = __reduce_add_sync(0xffu, s_w_tr[8][lane]);
          g_trimin = __reduce_min_sync(0xffu, s_w_tr[9][lane]);
          g_trimax = __reduce_max_sync(0xffu, s_w_tr[10][lane]);
          g_a1_bgc = __reduce_add_sync(0xffu, s_w_tr[11][lane]);
          g_a1_bgmin = __reduce_min_sync(0xffu, s_w_tr[12][lane]);
          g_a1_bgmax = __reduce_max_sync(0xffu, s_w_tr[13][lane]);
          g_a1_tric = __reduce_add_sync(0xffu, s_w_tr[14][lane]);
          g_a1_trimin = __reduce_min_sync(0xffu, s_w_tr[15][lane]);
          g_a1_trimax = __reduce_max_sync(0xffu, s_w_tr[16][lane]);
        }
        g_utot = __shfl_sync(0xffffffffu, g_utot, 0);
        if (g_utot > 0) {
          g_umin = __shfl_sync(0xffffffffu, g_umin, 0);
          g_umax = __shfl_sync(0xffffffffu, g_umax, 0);
          g_bgc = __shfl_sync(0xffffffffu, g_bgc, 0);
          g_bgmin = __shfl_sync(0xffffffffu, g_bgmin, 0);
          g_bgmax = __shfl_sync(0xffffffffu, g_bgmax, 0);
          g_tric = __shfl_sync(0xffffffffu, g_tric, 0);
          g_trimin = __shfl_sync(0xffffffffu, g_trimin, 0);
          g_trimax = __shfl_sync(0xffffffffu, g_trimax, 0);
          g_a1_bgc = __shfl_sync(0xffffffffu, g_a1_bgc, 0);
          g_a1_bgmin = __shfl_sync(0xffffffffu, g_a1_bgmin, 0);
          g_a1_bgmax = __shfl_sync(0xffffffffu, g_a1_bgmax, 0);

          const int32_t tok32 = s_leaf_tok_r[r];
          const int32_t pv_r = s_leaf_prv_r[r];
          const int32_t pv2_r = s_leaf_prv2_r[r];
          int sh_occ = 0;
          bool sh_conflict = false;
          bool sh_bg_conflict = false;
          bool sh_a1_conflict = false;
          bool sh_tri_conflict = false;
          const bool is_tri_mono = (g_tric >= 2 && g_trimin == g_trimax);
          const bool is_tri1_cand = (tr_w32 && g_tric >= 1 && g_trimin == g_trimax);
          if (want_sh_scan) {
            if (lane == 0) {
              sh_occ = sh_u_c;
              sh_conflict = (sh_u_c > 0 && (sh_u_min != g_umin || sh_u_max != g_umin));
              sh_bg_conflict = (sh_bg_c > 0 && (sh_bg_min != g_bgmin || sh_bg_max != g_bgmin));
              sh_a1_conflict = (g_a1_bgc >= 1 && g_a1_bgmin == g_a1_bgmax) &&
                               (sh_bg_c > 0 && (sh_bg_min != g_a1_bgmin || sh_bg_max != g_a1_bgmin));
              sh_tri_conflict = (sh_tri_c > 0 && (sh_tri_min != g_trimin || sh_tri_max != g_trimin));
            }
          } else if (!is_tri_mono && (is_tri1_cand ||
                                      (g_bgc >= 1 && g_bgmin == g_bgmax) ||
                                      (g_a1_bgc >= 1 && g_a1_bgmin == g_a1_bgmax) ||
                                      (g_umin == g_umax && g_umin != tok32))) {
            int my_occ = 0;
            int my_conf = 0;
            int my_bg_conf = 0;
            int my_a1_conf = 0;
            int my_tri_conf = 0;
            const int win_m1 = s_win - 1;
            for (int hi = lane; hi < win_m1; hi += 32) {
              if (s_h[hi] == tok32) {
                my_occ += 1;
                const int32_t nxt_h = s_h[hi + 1];
                if (nxt_h != g_umin) my_conf = 1;
                if (pv_r >= 0 && hi >= 1 && s_h[hi - 1] == pv_r) {
                  if (nxt_h != g_bgmin) my_bg_conf = 1;
                  if (nxt_h != g_a1_bgmin) my_a1_conf = 1;
                  if (pv2_r >= 0 && hi >= 2 && s_h[hi - 2] == pv2_r && nxt_h != g_trimin) {
                    my_tri_conf = 1;
                  }
                }
              }
            }
            sh_occ = __reduce_add_sync(0xffffffffu, my_occ);
            sh_conflict = (__ballot_sync(0xffffffffu, my_conf) != 0u);
            sh_bg_conflict = (__ballot_sync(0xffffffffu, my_bg_conf) != 0u);
            sh_a1_conflict = (__ballot_sync(0xffffffffu, my_a1_conf) != 0u);
            sh_tri_conflict = (__ballot_sync(0xffffffffu, my_tri_conf) != 0u);
          }

          if (lane == 0) {
            const int j_ch = k_w;
            auto in_ch_topK = [&](int32_t cand_c) -> bool {
              if (!s_valid_row[j_ch]) return false;
#pragma unroll
              for (int rp = 0; rp < kMaxTreeK; ++rp) {
                if (rp < tk && s_top[j_ch][rp] == cand_c) return true;
              }
              return false;
            };
            if (is_tri_mono) {
              s_tr_ch_tok = static_cast<int64_t>(g_trimin);
              s_tr_ch_p = in_ch_topK(g_trimin) ? tr.ch_bg_t6 : tr.ch_bg_oov;
            } else if (is_tri1_cand && !sh_tri_conflict) {
              s_tr_ch_tok = static_cast<int64_t>(g_trimin);
              s_tr_ch_p = in_ch_topK(g_trimin) ? 0.68f : 0.45f;
            } else if (g_bgc >= 2 && g_bgmin == g_bgmax &&
                       (!sh_bg_conflict || (g_near >= 2 && in_ch_topK(g_bgmin)))) {
              s_tr_ch_tok = static_cast<int64_t>(g_bgmin);
              s_tr_ch_p = in_ch_topK(g_bgmin) ? tr.ch_bg_t6 : tr.ch_bg_oov;
            } else if (g_a1_tric >= 1 && g_a1_trimin == g_a1_trimax && g_a1_trimin != tok32 &&
                       !sh_a1_conflict && tr.ch_a1_bg_t6 > 0.0f) {
              s_tr_ch_tok = static_cast<int64_t>(g_a1_trimin);
              s_tr_ch_p = in_ch_topK(g_a1_trimin) ? tr.ch_bg_t6 : tr.ch_a1_bg_oov;
            } else if (g_bgc >= 1 && g_bgmin == g_bgmax && !sh_bg_conflict &&
                       tr.ch_bg1_t6 > 0.0f && in_ch_topK(g_bgmin)) {
              s_tr_ch_tok = static_cast<int64_t>(g_bgmin);
              s_tr_ch_p = tr.ch_bg1_t6;
            } else if (g_a1_bgc >= 1 && g_a1_bgmin == g_a1_bgmax && g_a1_bgmin != tok32 &&
                       !sh_a1_conflict && tr.ch_a1_bg_t6 > 0.0f &&
                       (g_a1_bgc >= 2 || in_ch_topK(g_a1_bgmin))) {
              s_tr_ch_tok = static_cast<int64_t>(g_a1_bgmin);
              s_tr_ch_p = in_ch_topK(g_a1_bgmin) ? tr.ch_a1_bg_t6 : tr.ch_a1_bg_oov;
            } else if (g_umin == g_umax && g_umin != tok32) {
              const bool in_tk = in_ch_topK(g_umin);
              if (g_near >= 1 && !sh_conflict && (sh_occ == 0 || (g_t0u >= 2 && sh_occ <= 3))) {
                s_tr_ch_tok = static_cast<int64_t>(g_umin);
                s_tr_ch_p = in_tk ? tr.ch_near_t6 : tr.ch_near_oov;
              } else if (in_tk && g_t0u >= 1 && g_utot >= 2 && sh_occ <= 3) {
                s_tr_ch_tok = static_cast<int64_t>(g_umin);
                s_tr_ch_p = tr.ch_t6_agree;
              }
            }
          }
        }
      }

      if (tid == 0) {
      const float best_prob = (best_s > 0.0f) ? s_cand_prob[best_idx] : -1.0f;
      if (best_s > 0.0f) {
        s_cand_score[best_idx] = -1.0f;
        s_cand_prob[best_idx] = -1.0f;
      }
      const int64_t sel_tok = (best_s > 0.0f) ? s_cand_tok[best_idx] : -1;
      if (best_s > 0.0f && sel_tok >= 0 && sel_tok < static_cast<int64_t>(vocab_size)) {
        const int64_t tok = sel_tok;
        const int k_r = static_cast<int>(s_cand_k[best_idx]);
        const int par = static_cast<int>(s_cand_par[best_idx]);
        const int dp = static_cast<int>(s_cand_depth[best_idx]);
        const int kind = static_cast<int>(s_cand_kind[best_idx]);
        const int hpos = static_cast<int>(s_cand_hpos[best_idx]);
        const int ml = static_cast<int>(s_cand_ml[best_idx]);
        const int rk_in_row = static_cast<int>(s_cand_rk[best_idx]);
        if (kind == 8 || kind == 9) {
          casc_row_mask |= (1u << r);
          if (stats != nullptr && stats_len >= 48) {
            stats[43] += 1;
          }
        }

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
        s_row_depth[r] = dp;
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

        // Dynamically spawn depth+1 children (up to eff_max_depth <= 6, k_child <= 6, 0 stack spill).
        if (li + 1 < num_leaves && dp < eff_max_depth && k_r < 6 && saved_leaf_parent != nullptr) {
          const int k_child = k_r + 1;
          const int j_par = k_r - 1;  // 0..4
          const int j_ch = k_r;       // 1..5
          int n_spawned = 0;
          auto already_spawned = [&](int64_t t) -> bool {
            const int32_t t32 = static_cast<int32_t>(t);
            for (int si = 0; si < n_spawned; ++si) {
              if (s_spawned[si] == t32) return true;
            }
            return false;
          };
          auto record_spawn = [&](int64_t t) {
            if (n_spawned < 16) s_spawned[n_spawned++] = static_cast<int32_t>(t);
          };
          const int root_ml = (kind == 5) ? (ml - (dp - 1)) : ml;
          int max_pl_dp;
          if (pl_w32) {
            const int dp4 = (pl.max_dp_4 < 4) ? 4 : pl.max_dp_4;
            const int dp5 = (pl.max_dp_5 < 6) ? 6 : pl.max_dp_5;
            const int dp6 = (pl.max_dp_6 < 6) ? 6 : pl.max_dp_6;
            max_pl_dp = (root_ml <= 3) ? 3 : ((root_ml == 4) ? dp4 : ((root_ml == 5) ? dp5 : dp6));
          } else {
            max_pl_dp = (root_ml <= 4) ? pl.max_dp_4 : ((root_ml == 5) ? pl.max_dp_5 : pl.max_dp_6);
          }
          const bool can_pl_cont = (pl.mode != 0 && s_hist_ok != 0 && hpos >= 0 && hpos + 1 < s_win &&
                                    dp < max_pl_dp && (kind == 4 || kind == 5 || ml >= pl.cont_from_df_n));
          const int64_t pl_next_tok = can_pl_cont ? static_cast<int64_t>(s_h[hpos + 1]) : -1;
          const int next_ml = (ml < 16) ? (ml + 1) : 16;
          const int next_ml_b = (next_ml > 6) ? 6 : next_ml;
          const float pl_cp = (can_pl_cont && next_ml_b >= 1) ? pl.cont_p[next_ml_b] : 0.0f;
          const int64_t tr_ch_tok = s_tr_ch_tok;
          const float tr_ch_p = s_tr_ch_p;
          const int64_t sh_ch_tok = s_sh_ch_tok;
          const float sh_ch_p = s_sh_ch_p;
          const int sh_ch_hp = s_sh_ch_hp;
          const int sh_ch_ml = s_sh_ch_ml;
          if (tr_ch_tok >= 0 && tr_ch_p > 0.0f && stats != nullptr && stats_len >= 40) {
            stats[38] += 1;
          }

          // Compute smoothed Markov bigram/trigram + 0.00ms Anti-Chimera post-miss distribution (SMEM scratch).
          const bool allow_bg_kind =
              (kind == 0) || (bg.bg_sub != 0 && kind == 1) || (bg.bg_off != 0 && (kind == 3 || kind == 6));
          const bool use_bg = (calib_mode != 0 && (bg.bg_mode != 0 || ac.ac_mode != 0) && s_hist_ok != 0 &&
                               s_valid_row[j_ch] != 0 && rk_in_row >= 0 && allow_bg_kind);
          const uint32_t ac_u = use_bg ? s_alt_cnt[j_par][rk_in_row] : 0u;
          const uint32_t m3 = use_bg ? s_bg3_mask[j_par][rk_in_row] : 0u;
          uint32_t bg_mask = 0u;
          bool has_p_post_vec = false;
          if (calib_mode != 0 && s_valid_row[j_ch] != 0 && (kind < 8 || kind == 10)) {
            bool any_delta = false;
            float den1_mod = 1.0f;
            const float inv_den = (ac_u > 0u) ? (1.0f / (static_cast<float>(ac_u) + bg.bg_alpha)) : 0.0f;
            const float min_p = (kind == 1) ? bg.sub_min_p : bg.bg_min_p;
            const float fwd_pen = (ac.ac_mode != 0 && use_bg && static_cast<int>(ac_u) >= ac.min_u)
                ? fminf(ac.fwd_cap, ac.fwd_lambda * __logf(1.0f + static_cast<float>(ac_u) / ac.fwd_alpha))
                : 0.0f;
#pragma unroll
            for (int rp = 0; rp < kMaxTreeK; ++rp) {
              float ev = 0.0f;
              if (rp < tk) {
                ev = s_e1[j_ch][rp];
                float dv = 0.0f;
                const uint32_t cnt_rp = (use_bg && ac_u > 0u) ? s_bg_cnt[j_par][rk_in_row][rp] : 0u;
                if (bg.bg_mode != 0 && cnt_rp > 0u) {
                  const float p_bg = static_cast<float>(cnt_rp) * inv_den;
                  const bool is3 = ((m3 & (1u << rp)) != 0u) && (p_bg >= bg.tri_min_p);
                  if (p_bg >= min_p) dv += bg.f2 + bg.w2 * p_bg;
                  if (is3) dv += bg.f3 + bg.w3 * p_bg;
                  if (dv > 0.0f) bg_mask |= (1u << rp);
                } else if (ac.ac_mode != 0 && cnt_rp == 0u) {
                  if (fwd_pen > 0.0f) {
                    dv -= (kind == 1) ? (0.5f * fwd_pen) : fwd_pen;
                  }
                  if (use_bg && kind != 1 && rk_in_row >= 1) {
                    const uint32_t vin = s_in_cnt[j_par][rp];
                    const uint32_t own_cnt = s_bg_cnt[j_par][0][rp];
                    if (static_cast<int>(vin) >= ac.min_v && own_cnt > 0u) {
                      const float ofrac = static_cast<float>(own_cnt) / (static_cast<float>(vin) + ac.rev_alpha);
                      dv -= fminf(ac.rev_cap, ac.rev_lambda * ofrac);
                    }
                  }
                  if (rp == 0 && kind == 0 && ac.ent_lambda > 0.0f &&
                      s_margin[j_par] < 2.0f && s_p1[j_ch] < 0.75f) {
                    dv -= ac.ent_lambda * (1.0f - s_p1[j_ch]) * (1.0f - 0.5f * s_margin[j_par]);
                  }
                }
                if (dv != 0.0f) {
                  any_delta = true;
                  if (ev > 0.0f) ev *= __expf(dv);
                }
              }
              s_p_post_vec[rp] = ev;
              den1_mod += ev;
            }
            if (any_delta) {
              const float inv_d1 = 1.0f / den1_mod;
#pragma unroll
              for (int rp = 0; rp < kMaxTreeK; ++rp) {
                s_p_post_vec[rp] *= inv_d1;
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
            if (!use_bg || ac_u == 0u) return;
            const uint32_t cnt_rp = s_bg_cnt[j_par][rk_in_row][rp];
            if (cnt_rp == 0u) return;
            const float p_bg = static_cast<float>(cnt_rp) / (static_cast<float>(ac_u) + bg.bg_alpha);
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

          // 0) Multi-Future Cascade coherent continuation (`casc_chain` kind=8, `casc_alt` kind=9).
          if ((kind == 8 || kind == 9) && s_casc_mode != 0 && s_num_cands < kCandCap) {
            const int64_t c1 = s_casc_top[j_ch][0];
            if (c1 >= 0 && c1 < static_cast<int64_t>(vocab_size) && !already_spawned(c1)) {
              const bool agrees_pl1 = (can_pl_cont && pl_next_tok == c1 && pl_cp > 0.0f);
              const bool agrees_tr1 = (tr_ch_tok == c1 && tr_ch_p > 0.0f);
              float p_c1 = (kind == 8) ? s_casc_p1[j_ch] : (s_casc_p1[j_ch] * 0.55f);
              if (agrees_pl1 && pl_cp > p_c1) p_c1 = pl_cp;
              if (agrees_tr1 && tr_ch_p > p_c1) p_c1 = tr_ch_p;
              const float ch_prob1 = best_prob * p_c1;
              if (ch_prob1 > 0.0f) {
                const float lh1 = (kind == 8 && j_ch + 1 < 6 && li + 2 < num_leaves)
                    ? (1.0f + csk.lookahead_w * s_casc_p1[j_ch + 1])
                    : 1.0f;
                const int slot = s_num_cands++;
                s_cand_prob[slot] = ch_prob1;
                s_cand_score[slot] = ch_prob1 * lh1;
                s_cand_tok[slot] = c1;
                s_cand_k[slot] = static_cast<int8_t>(k_child);
                s_cand_par[slot] = static_cast<int8_t>(r);
                s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
                s_cand_kind[slot] = 8;
                s_cand_hpos[slot] = agrees_pl1 ? static_cast<int16_t>(hpos + 1) : -1;
                s_cand_ml[slot] = agrees_pl1 ? static_cast<int8_t>(next_ml) : 0;
                s_cand_rk[slot] = -1;
                record_spawn(c1);
              }
            }
            if (kind == 8 && s_num_cands < kCandCap) {
              const int64_t c2 = s_casc_top[j_ch][1];
              if (c2 >= 0 && c2 < static_cast<int64_t>(vocab_size) && !already_spawned(c2)) {
                const bool agrees_pl2 = (can_pl_cont && pl_next_tok == c2 && pl_cp > 0.0f);
                const bool agrees_tr2 = (tr_ch_tok == c2 && tr_ch_p > 0.0f);
                float p_c2 = s_casc_p2[j_ch];
                if (agrees_pl2 && pl_cp > p_c2) p_c2 = pl_cp;
                if (agrees_tr2 && tr_ch_p > p_c2) p_c2 = tr_ch_p;
                const float ch_prob2 = best_prob * p_c2;
                if (ch_prob2 > 0.0f) {
                  const int slot = s_num_cands++;
                  s_cand_prob[slot] = ch_prob2;
                  s_cand_score[slot] = ch_prob2;
                  s_cand_tok[slot] = c2;
                  s_cand_k[slot] = static_cast<int8_t>(k_child);
                  s_cand_par[slot] = static_cast<int8_t>(r);
                  s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
                  s_cand_kind[slot] = 9;
                  s_cand_hpos[slot] = agrees_pl2 ? static_cast<int16_t>(hpos + 1) : -1;
                  s_cand_ml[slot] = agrees_pl2 ? static_cast<int8_t>(next_ml) : 0;
                  s_cand_rk[slot] = -1;
                  record_spawn(c2);
                }
              }
            }
          }

          // 1) Aligned substitution continuation (`sub`, kind=1): d_{k_r+1}#1 = s_dk[k_r] (with anti-stutter).
          if ((kind == 0 || kind == 1 || kind == 3 || kind == 4 || kind == 5 || kind == 6 || kind == 7 || kind == 10) &&
              s_valid_row[j_ch] && s_num_cands < kCandCap) {
            const int64_t sub_tok = s_dk[j_ch];
            const bool agrees_pl = (can_pl_cont && pl_next_tok == sub_tok && pl_cp > 0.0f);
            const bool agrees_tr = (tr_ch_tok == sub_tok && tr_ch_p > 0.0f);
            const bool agrees_sh = (sh_ch_tok == sub_tok && sh_ch_p > 0.0f);
            float p_cont;
            if (kind == 1) {
              p_cont = s_p1[j_ch];
              if (has_p_post_vec && ((bg_mask & 1u) != 0u) && s_p_post_vec[0] > p_cont) {
                p_cont = s_p_post_vec[0];
              }
            } else {
              p_cont = has_p_post_vec ? s_p_post_vec[0] : s_p1post[j_ch];
            }
            if (agrees_pl && pl_cp > p_cont) p_cont = pl_cp;
            if (agrees_tr && tr_ch_p > p_cont) p_cont = tr_ch_p;
            if (agrees_sh && sh_ch_p > p_cont) p_cont = sh_ch_p;
            if (sub_tok == tok && !agrees_pl && !agrees_tr && !agrees_sh && ((bg_mask & 1u) == 0u)) {
              p_cont *= ac.stutter_scale;
            }
            const float child_p = (calib_mode != 0)
                ? (best_prob * p_cont)
                : (best_prob * ((kind == 0) ? sub_coef : cont_decay) * s_p1[j_ch]);
            if (child_p > 0.0f && sub_tok >= 0 && !already_spawned(sub_tok)) {
              int16_t ch_hp = -1;
              int8_t ch_ml = 0;
              resolve_bg_hpos(0, agrees_pl, ch_hp, ch_ml);
              if (agrees_sh && sh_ch_hp >= 0 && (ch_hp < 0 || sh_ch_ml > static_cast<int>(ch_ml))) {
                ch_hp = static_cast<int16_t>(sh_ch_hp);
                ch_ml = static_cast<int8_t>(sh_ch_ml);
              }
              const int slot = s_num_cands++;
              s_cand_prob[slot] = child_p;
              s_cand_score[slot] = child_p * cand_lh(k_child, static_cast<int>(ch_hp), static_cast<int>(ch_ml));
              s_cand_tok[slot] = sub_tok;
              s_cand_k[slot] = static_cast<int8_t>(k_child);
              s_cand_par[slot] = static_cast<int8_t>(r);
              s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
              s_cand_kind[slot] = 1;
              s_cand_hpos[slot] = ch_hp;
              s_cand_ml[slot] = ch_ml;
              s_cand_rk[slot] = 0;
              record_spawn(sub_tok);
            }
          }
          // 1b) Left-shift / deletion recovery (`skip-1`, kind=10): s_dk[k_r + 1] when tok == s_dk[k_r] or kind==10.
          if (calib_mode != 0 && ac.skip_coef > 0.0f && k_r + 1 < 6 && s_valid_row[k_r + 1] != 0 &&
              (tok == s_dk[k_r] || kind == 10) && s_num_cands < kCandCap) {
            const int64_t skip_tok = s_dk[k_r + 1];
            if (skip_tok >= 0 && !already_spawned(skip_tok)) {
              const bool agrees_pl = (can_pl_cont && pl_next_tok == skip_tok && pl_cp > 0.0f);
              const bool agrees_tr = (tr_ch_tok == skip_tok && tr_ch_p > 0.0f);
              const bool agrees_sh = (sh_ch_tok == skip_tok && sh_ch_p > 0.0f);
              float p_skip = ((kind == 10) ? cont_decay : ac.skip_coef) * s_p1[k_r + 1];
              if (agrees_pl && pl_cp > p_skip) p_skip = pl_cp;
              if (agrees_tr && tr_ch_p > p_skip) p_skip = tr_ch_p;
              if (agrees_sh && sh_ch_p > p_skip) p_skip = sh_ch_p;
              const float child_p = best_prob * p_skip;
              if (child_p > 0.0f) {
                int ch_hp_i = agrees_pl ? (hpos + 1) : -1;
                int ch_ml_i = agrees_pl ? next_ml : 0;
                if (agrees_sh && sh_ch_hp >= 0 && (ch_hp_i < 0 || sh_ch_ml > ch_ml_i)) {
                  ch_hp_i = sh_ch_hp;
                  ch_ml_i = sh_ch_ml;
                }
                float sk_lh = cand_lh(k_child, ch_hp_i, ch_ml_i);
                if (ac.skip_lh_w > 0.0f && k_r + 2 < 6 && s_valid_row[k_r + 2] != 0) {
                  sk_lh *= (1.0f + ac.skip_lh_w * s_p1[k_r + 2]);
                }
                const int slot = s_num_cands++;
                s_cand_prob[slot] = child_p;
                s_cand_score[slot] = child_p * sk_lh;
                s_cand_tok[slot] = skip_tok;
                s_cand_k[slot] = static_cast<int8_t>(k_child);
                s_cand_par[slot] = static_cast<int8_t>(r);
                s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
                s_cand_kind[slot] = 10;
                s_cand_hpos[slot] = static_cast<int16_t>(ch_hp_i);
                s_cand_ml[slot] = static_cast<int8_t>(ch_ml_i);
                s_cand_rk[slot] = -1;
                record_spawn(skip_tok);
              }
            }
          }
          // 2) Post-miss top-2 (`alt2`, kind=3) + Markov-coherent / 30B-recycled / s_h-confirmed / wide r'=2..tk-1 (`coh_cont`, kind=6).
          if (calib_mode != 0 && s_valid_row[j_ch] && (kind < 8 || kind == 10)) {
            for (int rp = 1; rp < tk; ++rp) {
              if (s_num_cands >= kCandCap) break;
              const int64_t ctok = tree_top[j_ch * tree_k + rp];
              if (ctok < 0 || already_spawned(ctok)) continue;
              const bool is_bg_hit = ((bg_mask & (1u << rp)) != 0u);
              const bool agrees_tr = (tr_ch_tok == ctok && tr_ch_p > 0.0f);
              const bool agrees_sh = (sh_ch_tok == ctok && sh_ch_p > 0.0f);
              const bool allow_wide = (tree_m > 16) &&
                  (((kind == 0 || kind == 1 || kind == 3 || kind == 4 || kind == 6 || kind == 7 || kind == 10) &&
                    (rp == 1 || rp == 2)) ||
                   (rp == 3 && kind == 0 && k_r <= 2 && s_margin[j_ch] < 1.8f));
              if (!(((kind == 0 || kind == 4 || kind == 7) && rp == 1) || allow_wide || is_bg_hit || agrees_tr || agrees_sh)) {
                continue;
              }
              const bool agrees_pl = (can_pl_cont && pl_next_tok == ctok && pl_cp > 0.0f);
              float p_c = has_p_post_vec
                  ? ((rp == 1) ? (s_p_post_vec[1] * ck.alt2_scale) : s_p_post_vec[rp])
                  : ((rp == 1) ? s_p2post[j_ch] : (s_e1[j_ch][rp] / s_den1[j_ch]));
              if (agrees_pl && pl_cp > p_c) p_c = pl_cp;
              if (agrees_tr && tr_ch_p > p_c) p_c = tr_ch_p;
              if (agrees_sh && sh_ch_p > p_c) p_c = sh_ch_p;
              const float child_p = best_prob * p_c;
              if (child_p > 0.0f) {
                int16_t ch_hp = -1;
                int8_t ch_ml = 0;
                resolve_bg_hpos(rp, agrees_pl, ch_hp, ch_ml);
                if (agrees_sh && sh_ch_hp >= 0 && (ch_hp < 0 || sh_ch_ml > static_cast<int>(ch_ml))) {
                  ch_hp = static_cast<int16_t>(sh_ch_hp);
                  ch_ml = static_cast<int8_t>(sh_ch_ml);
                }
                const int slot = s_num_cands++;
                s_cand_prob[slot] = child_p;
                s_cand_score[slot] = child_p * cand_lh(k_child, static_cast<int>(ch_hp), static_cast<int>(ch_ml));
                s_cand_tok[slot] = ctok;
                s_cand_k[slot] = static_cast<int8_t>(k_child);
                s_cand_par[slot] = static_cast<int8_t>(r);
                s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
                s_cand_kind[slot] = (rp == 1) ? 3 : 6;
                s_cand_hpos[slot] = ch_hp;
                s_cand_ml[slot] = ch_ml;
                s_cand_rk[slot] = static_cast<int8_t>(rp);
                record_spawn(ctok);
              }
            }
          }
          // 3) Off-by-1 shift recovery (`shf`, kind=2): d_{k_r}#1 = s_dk[k_r - 1].
          if ((kind == 0 || kind == 2 || kind == 4 || kind == 7) && s_valid_row[k_r - 1] && s_num_cands < kCandCap) {
            const int64_t shf_tok = s_dk[k_r - 1];
            if (shf_tok >= 0 && !already_spawned(shf_tok)) {
              const bool agrees_pl = (can_pl_cont && pl_next_tok == shf_tok && pl_cp > 0.0f);
              const bool agrees_tr = (tr_ch_tok == shf_tok && tr_ch_p > 0.0f);
              const bool agrees_sh = (sh_ch_tok == shf_tok && sh_ch_p > 0.0f);
              float p_shf;
              if (kind == 0 || kind == 4 || kind == 7) {
                if (bg.shf_cal != 0 && kind == 0 && rk_in_row >= 1) {
                  const float unseen_shf = (tree_m > 16 && ac.shf_unseen_scale > 0.0f)
                      ? fmaxf(bg.shf_lo, ac.shf_unseen_scale * s_p1[k_r - 1])
                      : bg.shf_lo;
                  p_shf = (s_hist_ok != 0 && s_shf_bg[j_par][rk_in_row - 1] != 0u) ? bg.shf_hi : unseen_shf;
                } else {
                  p_shf = ck.shf;
                }
              } else {
                p_shf = s_p1[k_r - 1];
              }
              if (agrees_pl && pl_cp > p_shf) p_shf = pl_cp;
              if (agrees_tr && tr_ch_p > p_shf) p_shf = tr_ch_p;
              if (agrees_sh && sh_ch_p > p_shf) p_shf = sh_ch_p;
              const float child_p = (calib_mode != 0)
                  ? (best_prob * p_shf)
                  : ((kind == 0) ? (best_prob * shf_coef) : (best_prob * cont_decay * s_p1[k_r - 1]));
              if (child_p > 0.0f) {
                int ch_hp_i = agrees_pl ? (hpos + 1) : -1;
                int ch_ml_i = agrees_pl ? next_ml : 0;
                if (agrees_sh && sh_ch_hp >= 0 && (ch_hp_i < 0 || sh_ch_ml > ch_ml_i)) {
                  ch_hp_i = sh_ch_hp;
                  ch_ml_i = sh_ch_ml;
                }
                const int slot = s_num_cands++;
                s_cand_prob[slot] = child_p;
                s_cand_score[slot] = child_p * cand_lh(k_child, ch_hp_i, ch_ml_i);
                s_cand_tok[slot] = shf_tok;
                s_cand_k[slot] = static_cast<int8_t>(k_child);
                s_cand_par[slot] = static_cast<int8_t>(r);
                s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
                s_cand_kind[slot] = 2;
                s_cand_hpos[slot] = static_cast<int16_t>(ch_hp_i);
                s_cand_ml[slot] = static_cast<int8_t>(ch_ml_i);
                s_cand_rk[slot] = -1;
                record_spawn(shf_tok);
              }
            }
          }
          // 4) Prompt-lookup copy continuation (`pl_cont`, kind=5): s_h[hpos + 1].
          if (can_pl_cont && pl_next_tok >= 0 && pl_cp > 0.0f &&
              !already_spawned(pl_next_tok) && s_num_cands < kCandCap) {
            float p_use = (tr_ch_tok == pl_next_tok && tr_ch_p > pl_cp) ? tr_ch_p : pl_cp;
            if (sh_ch_tok == pl_next_tok && sh_ch_p > p_use) p_use = sh_ch_p;
            const float child_p = best_prob * p_use;
            if (child_p > 0.0f) {
              const int slot = s_num_cands++;
              s_cand_prob[slot] = child_p;
              s_cand_score[slot] = child_p * cand_lh(k_child, hpos + 1, next_ml);
              s_cand_tok[slot] = pl_next_tok;
              s_cand_k[slot] = static_cast<int8_t>(k_child);
              s_cand_par[slot] = static_cast<int8_t>(r);
              s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
              s_cand_kind[slot] = 5;
              s_cand_hpos[slot] = static_cast<int16_t>(hpos + 1);
              s_cand_ml[slot] = static_cast<int8_t>(next_ml);
              s_cand_rk[slot] = -1;
              record_spawn(pl_next_tok);
            }
          }
          // 5) Branch-conditioned unique OOV successor (deterministic nxt occurring >=2 times in s_h).
          if (calib_mode != 0 && bg.uniq_mode != 0 && s_hist_ok != 0 && rk_in_row >= 1 &&
              (kind == 0 || (bg.bg_off != 0 && (kind == 3 || kind == 6))) && s_num_cands < kCandCap) {
            const uint32_t uc = s_alt_cnt[j_par][rk_in_row];
            const int32_t umin = s_uniq_min[j_par][rk_in_row - 1];
            const int32_t umax = s_uniq_max[j_par][rk_in_row - 1];
            if (uc >= 2u && umin >= 0 && umin == umax) {
              const int64_t utok = static_cast<int64_t>(umin);
              if (!already_spawned(utok)) {
                float p_u = (uc == 2u) ? bg.uniq_p2 : bg.uniq_p3;
                if (tr_ch_tok == utok && tr_ch_p > p_u) p_u = tr_ch_p;
                if (sh_ch_tok == utok && sh_ch_p > p_u) p_u = sh_ch_p;
                const float child_p = best_prob * p_u;
                if (child_p > 0.0f) {
                  const int hp_u = static_cast<int>(s_uniq_hp[j_par][rk_in_row - 1]);
                  const int ml_u = (uc == 2u) ? 3 : 4;
                  const int slot = s_num_cands++;
                  s_cand_prob[slot] = child_p;
                  s_cand_score[slot] = child_p * cand_lh(k_child, hp_u, ml_u);
                  s_cand_tok[slot] = utok;
                  s_cand_k[slot] = static_cast<int8_t>(k_child);
                  s_cand_par[slot] = static_cast<int8_t>(r);
                  s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
                  s_cand_kind[slot] = 5;
                  s_cand_hpos[slot] = static_cast<int16_t>(hp_u);
                  s_cand_ml[slot] = static_cast<int8_t>(ml_u);
                  s_cand_rk[slot] = -1;
                  record_spawn(utok);
                }
              }
            }
          }
          // 5b) Fused 256-thread branch-conditioned s_h OOV continuation (`mono_tri` / `bg2_mono` / `u2_mono`).
          if (calib_mode != 0 && sh_ch_tok >= 0 && sh_ch_p > 0.0f &&
              !already_spawned(sh_ch_tok) && s_num_cands < kCandCap) {
            const float p_use = (tr_ch_tok == sh_ch_tok && tr_ch_p > sh_ch_p) ? tr_ch_p : sh_ch_p;
            const float child_p = best_prob * p_use;
            if (child_p > 0.0f) {
              const int slot = s_num_cands++;
              s_cand_prob[slot] = child_p;
              s_cand_score[slot] = child_p * cand_lh(k_child, sh_ch_hp, sh_ch_ml);
              s_cand_tok[slot] = sh_ch_tok;
              s_cand_k[slot] = static_cast<int8_t>(k_child);
              s_cand_par[slot] = static_cast<int8_t>(r);
              s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
              s_cand_kind[slot] = 5;
              s_cand_hpos[slot] = static_cast<int16_t>(sh_ch_hp);
              s_cand_ml[slot] = static_cast<int8_t>(sh_ch_ml);
              s_cand_rk[slot] = -1;
              record_spawn(sh_ch_tok);
            }
          }
          // 6) Pure 30B Target-Recycled OOV continuation (`tr_cont`, kind=7).
          if (calib_mode != 0 && tr_ch_tok >= 0 && tr_ch_p > 0.0f &&
              !already_spawned(tr_ch_tok) && s_num_cands < kCandCap) {
            const float child_p = best_prob * tr_ch_p;
            if (child_p > 0.0f) {
              const int slot = s_num_cands++;
              s_cand_prob[slot] = child_p;
              s_cand_score[slot] = child_p;
              s_cand_tok[slot] = tr_ch_tok;
              s_cand_k[slot] = static_cast<int8_t>(k_child);
              s_cand_par[slot] = static_cast<int8_t>(r);
              s_cand_depth[slot] = static_cast<int8_t>(dp + 1);
              s_cand_kind[slot] = 7;
              s_cand_hpos[slot] = -1;
              s_cand_ml[slot] = 0;
              s_cand_rk[slot] = -1;
              record_spawn(tr_ch_tok);
            }
          }
        }
      } else {
        s_row_base_sub[r] = -1;
        s_row_depth[r] = 0;
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
      }  // end if (tid == 0)
      __syncwarp();
    }  // end if (warp == 0)
  }

  // Record per-row (cur, prv1, prv2, pos_r) for up to 32-row 30B Target Recycling commit in 1 parallel cycle.
  if (warp == 0) {
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
      reinterpret_cast<int4*>(saved_row_ctx)[tid] = make_int4(cur_r, prv1_r, prv2_r, pos_r);
    }

    if (tid == 0) {
      if (cascade_meta != nullptr) {
        cascade_meta[5] = static_cast<int32_t>(casc_row_mask);
      }
      leaf_flag[0] = s_any_active ? -1 : 0;
    }
    if ((pdl_mode & 2) != 0) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
  }
}

__global__ void __launch_bounds__(32) native_leaf_commit_tree_kernel(
    int32_t* leaf_flag,
    int32_t* prev_ns,
    int32_t* hit_flag,
    int hit_flag_len,
    int32_t* hit_src_row,
    const int64_t* saved_leaf_toks,
    const int32_t* saved_leaf_k,
    const int32_t* saved_leaf_parent,
    const int64_t* leaf_u_toks,
    int64_t* sampled_tokens,
    int32_t* num_sampled,
    int32_t* num_rejected,
    int64_t* stats,
    int stats_len,
    int32_t* req_acc_state,
    int tree_m,
    int no_hit_mode,
    float high_acc_cpc,
    int32_t* hist,
    int32_t* hist_len,
    int hist_cap,
    const int64_t* saved_pos0,
    int32_t* saved_row_ctx,
    int32_t* tr_log,
    int32_t* tr_state,
    int tr_cap,
    int vocab_size,
    int chain_all_valid,
    int32_t* cascade_meta,
    const uint8_t* __restrict__ in_static_vocab,
    int32_t* tail_ids,
    int32_t* tail_count,
    int32_t* tail_seen,
    int max_tail,
    int oov_preadmit_mode,
    int req_acc_state_len,
    uint8_t* hit_cond_u8,
    int pdl_mode) {
  const int lane = threadIdx.x;
  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  __shared__ uint32_t s_exact_mask;
  __shared__ uint32_t s_matched_mask;
  __shared__ int32_t s_eff_raw_ns;
  __shared__ int32_t s_final_ns;
  __shared__ int32_t s_len0;
  __shared__ int32_t s_step_id;
  __shared__ int32_t s_ctx_p0;
  __shared__ int32_t s_do_tr;
  __shared__ int32_t s_do_oov;

  const int32_t init_ns = num_sampled[0];
  int32_t u_lane = -1;
  if (lane < tree_m) {
    if (no_hit_mode == 0) {
      if (lane < 7) {
        const int chain_lim = (chain_all_valid != 0) ? 7 : ((init_ns < 0) ? 0 : ((init_ns < 7) ? init_ns : 7));
        if (lane < chain_lim) {
          const int64_t v = sampled_tokens[lane];
          u_lane = (v >= 0 && v < static_cast<int64_t>(vocab_size)) ? static_cast<int32_t>(v) : -1;
        }
      } else {
        const int64_t v = leaf_u_toks[lane - 7];
        u_lane = (v >= 0 && v < static_cast<int64_t>(vocab_size)) ? static_cast<int32_t>(v) : -1;
      }
    }
  }
  int32_t lpar = -1;
  int32_t lk = 0;
  int64_t ltok = -1;
  if (lane >= 7 && lane < tree_m) {
    lpar = (saved_leaf_parent != nullptr) ? saved_leaf_parent[lane] : -1;
    lk = saved_leaf_k[lane];
    ltok = saved_leaf_toks[lane];
  }

  const int mode_w = leaf_flag[0];
  int32_t ns_w = init_ns;
  int32_t nr_w = num_rejected[0];
  uint32_t matched_mask_w = 0u;
  int32_t hit_dst0_w = 0;
  int32_t hit_src0_w = 0;
  int32_t hit_len_w = 0;
  int32_t casc_hit_cnt_w = 0;
  if (mode_w < 0 && ns_w >= 1 && ns_w <= 6 && no_hit_mode == 0) {
    uint32_t casc_row_mask_w = 0u;
    if (lane == 0 && cascade_meta != nullptr) {
      casc_row_mask_w = static_cast<uint32_t>(cascade_meta[5]);
    }
    casc_row_mask_w = __shfl_sync(0xffffffffu, casc_row_mask_w, 0);
    int32_t cur_parent = -1;
    int32_t u_w = 0;
    for (int stage = 0; stage < 6 && ns_w <= 6; ++stage) {
      const int64_t tok_k = (hit_len_w == 0) ? sampled_tokens[ns_w - 1] : static_cast<int64_t>(u_w);
      const bool pred = (lane >= 7) && (lpar == cur_parent) && (lk == ns_w) &&
                        (ltok >= 0) && (ltok == tok_k);
      const uint32_t match = __ballot_sync(0xffffffffu, pred);
      if (match == 0u) break;
      const int winner = __ffs(match) - 1;
      const int32_t u_w_new = __shfl_sync(0xffffffffu, u_lane, winner);
      if (u_w_new < 0) break;
      matched_mask_w |= (1u << winner);
      if (lane == 0) {
        const int32_t dst_r = ns_w;
        sampled_tokens[dst_r] = static_cast<int64_t>(u_w_new);
        if (hit_len_w == 0) {
          hit_dst0_w = dst_r;
          hit_src0_w = winner;
        }
        if (hit_flag_len >= 2 + 2 * (hit_len_w + 1)) {
          hit_flag[2 + 2 * hit_len_w] = dst_r;
          hit_flag[3 + 2 * hit_len_w] = winner;
        }
      }
      if ((casc_row_mask_w & (1u << winner)) != 0u) {
        casc_hit_cnt_w += 1;
      }
      hit_len_w += 1;
      ns_w += 1;
      nr_w -= 1;
      cur_parent = winner;
      u_w = u_w_new;
    }
  }
  int32_t casc_m_div1_l0 = 0;
  if (lane == 0) {
    leaf_flag[0] = 0;  // Clear immediately so non-tree steps see 0.
    if (cascade_meta != nullptr) {
      casc_m_div1_l0 = cascade_meta[6];
      cascade_meta[5] = 0;
      cascade_meta[6] = 0;
    }
    if (hit_len_w > 0) {
      num_sampled[0] = ns_w;
      num_rejected[0] = nr_w;
    }
  }
  __syncwarp();
  if (lane == 0) {
    const int32_t ns = ns_w;
    const int32_t raw_ns = init_ns;
    const int32_t casc_hit_cnt = casc_hit_cnt_w;
    const uint32_t matched_mask = matched_mask_w;
    const int32_t hit_dst0 = hit_dst0_w;
    const int32_t hit_src0 = hit_src0_w;
    const int32_t hit_len = hit_len_w;
    const int32_t casc_m_div1 = casc_m_div1_l0;
    const int mode = mode_w;

    hit_flag[0] = hit_dst0;
    if (hit_flag_len >= 2) {
      hit_flag[1] = hit_len;
    }
    hit_src_row[0] = hit_src0;
    if (hit_cond_u8 != nullptr) {
      *hit_cond_u8 = (hit_dst0 > 0) ? 1u : 0u;
    }
    prev_ns[0] = ns;
    s_eff_raw_ns = raw_ns;
    s_final_ns = ns;
    s_matched_mask = matched_mask;
    s_exact_mask = 0u;
    s_do_oov = (no_hit_mode == 0 && oov_preadmit_mode > 0 &&
                in_static_vocab != nullptr && tail_ids != nullptr &&
                tail_count != nullptr && tail_seen != nullptr && max_tail > 0 &&
                saved_pos0 != nullptr && saved_pos0[0] >= 1)
        ? 1
        : 0;

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

    int32_t do_tr = 0;
    if (no_hit_mode == 0 && tr_log != nullptr && tr_state != nullptr &&
        saved_row_ctx != nullptr && tr_cap > 0 && saved_pos0 != nullptr) {
      const int64_t p0 = saved_pos0[0];
      const int32_t ctx_p0 = saved_row_ctx[3];
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
        s_exact_mask = exact_mask;
        s_len0 = len;
        s_step_id = step_id;
        s_ctx_p0 = ctx_p0;
        do_tr = 1;
      } else {
        saved_row_ctx[3] = -1;
      }
    }
    s_do_tr = do_tr;

    if (req_acc_state != nullptr) {
      const int32_t r = req_acc_state[0] + 1;
      const int32_t c = req_acc_state[1] + ns;
      req_acc_state[0] = r;
      req_acc_state[1] = c;
      req_acc_state[2] =
          (high_acc_cpc > 0.0f && r >= 4 && static_cast<float>(c) > high_acc_cpc * static_cast<float>(r)) ? 1 : 0;
      if (req_acc_state_len >= 5) {
        const int32_t raw_acc = (raw_ns > 1) ? ((raw_ns - 1 <= 6) ? (raw_ns - 1) : 6) : 0;
        const int32_t raw_tr = (raw_ns >= 6) ? 6 : ((raw_ns >= 1) ? raw_ns : 1);
        req_acc_state[3] += raw_acc;
        req_acc_state[4] += raw_tr;
      }
    }

    if (stats != nullptr && stats_len >= 4) {
      stats[0] += 1;
      if (mode < 0) stats[1] += 1;
      if (hit_len > 0) {
        stats[2] += 1;
        if (hit_len >= 2) stats[3] += 1;
        if (stats_len >= 10 && hit_dst0 >= 1 && hit_dst0 <= 6) {
          stats[3 + hit_dst0] += 1;
        }
        if (stats_len >= 19 && hit_src0 >= 7 && hit_src0 < 16) {
          stats[10 + (hit_src0 - 7)] += 1;
        }
        if (stats_len >= 32) {
          const int hl = (hit_len <= 6) ? hit_len : 6;
          stats[18 + hl] += 1;
          stats[25] += hit_len;
          if (hit_len > 1) {
            stats[26] += (hit_len - 1);
            stats[29] += 1;
            if (hit_len >= 3) stats[30] += 1;
            if (hit_len >= 4) stats[31] += 1;
          }
          if (hit_len > stats[28]) stats[28] = hit_len;
        }
      }
      if (stats_len >= 32) {
        stats[27] += ns;
      }
      if (stats_len >= 48) {
        if (casc_hit_cnt > 0) {
          stats[44] += 1;
          stats[45] += casc_hit_cnt;
        }
        if (casc_m_div1 > 0 && raw_ns >= casc_m_div1 + 1) {
          stats[47] += (raw_ns - casc_m_div1);
        }
      }
    }
  }
  __syncwarp();

  // Warp-deterministic (__ballot_sync + __popc, zero atomicAdd) 32-row OOV pre-admission:
  // Pass A admits committed sampled_tokens[0..ns-1] (including tree hits);
  // Pass B pre-admits 30B greedy cone predictions across ALL 32 verified rows into tail_ids[512].
  if (s_do_oov != 0) {
    auto admit_warp_tok = [&](int32_t tok_a) {
      int won = 0;
      if (tok_a >= 0 && tok_a < vocab_size && in_static_vocab[tok_a] == 0) {
        if (atomicExch(&tail_seen[tok_a], 1) == 0) {
          won = 1;
        }
      }
      const uint32_t won_mask = __ballot_sync(0xffffffffu, won);
      if (won_mask != 0u) {
        int base_cnt = 0;
        if (lane == 0) {
          base_cnt = tail_count[0];
          if (base_cnt < 0) base_cnt = 0;
        }
        base_cnt = __shfl_sync(0xffffffffu, base_cnt, 0);
        const int off = __popc(won_mask & ((1u << lane) - 1u));
        const int slot = base_cnt + off;
        if (won) {
          if (slot < max_tail) {
            tail_ids[slot] = tok_a;
          } else {
            tail_seen[tok_a] = 0;
          }
        }
        if (lane == 0) {
          const int total_won = __popc(won_mask);
          const int new_cnt = (base_cnt + total_won <= max_tail) ? (base_cnt + total_won) : max_tail;
          tail_count[0] = new_cnt;
        }
      }
    };

    int32_t tok_commit = -1;
    if (lane < s_final_ns) {
      const int64_t sv = sampled_tokens[lane];
      if (sv >= 0 && sv < static_cast<int64_t>(vocab_size)) {
        tok_commit = static_cast<int32_t>(sv);
      }
    }
    admit_warp_tok(tok_commit);

    if (oov_preadmit_mode >= 2 && chain_all_valid != 0) {
      int32_t tok_cone = -1;
      if (lane < tree_m) {
        const bool row_ok = (oov_preadmit_mode >= 3)
            ? (lane <= 6 || saved_leaf_k[lane] > 0)
            : (((s_exact_mask & (1u << lane)) != 0u) || lane <= s_eff_raw_ns ||
               ((s_matched_mask & (1u << lane)) != 0u));
        if (row_ok) {
          tok_cone = u_lane;
        }
      }
      admit_warp_tok(tok_cone);
    }
  }

  // Warp-parallel 32-lane coalesced int4 append to tr_log (~1.5 us vs ~35 us scalar).
  if (s_do_tr != 0) {
    int4 ctx4 = make_int4(-1, -1, -1, -1);
    int32_t u_t = -1;
    if (lane < tree_m) {
      ctx4 = reinterpret_cast<const int4*>(saved_row_ctx)[lane];
      u_t = u_lane;
    }
    const bool valid = (lane < tree_m && ctx4.x >= 0 && ctx4.x < vocab_size && u_t >= 0 && u_t < vocab_size);
    const uint32_t vmask = __ballot_sync(0xffffffffu, valid);
    const int off = __popc(vmask & ((1u << lane) - 1u));
    const int dst = s_len0 + off;
    if (valid && dst < tr_cap) {
      int4* tr4 = reinterpret_cast<int4*>(tr_log);
      const int32_t pos_val = (lane == 0) ? s_ctx_p0 : ctx4.w;
      const int32_t tier_val = ((s_exact_mask & (1u << lane)) != 0u) ? 0 : 1;
      tr4[dst * 2 + 0] = make_int4(lane, ctx4.x, ctx4.y, ctx4.z);
      tr4[dst * 2 + 1] = make_int4(u_t, s_step_id, pos_val, tier_val);
    }
    if (lane == 0) {
      const int total_v = __popc(vmask);
      const int new_len = (s_len0 + total_v <= tr_cap) ? (s_len0 + total_v) : tr_cap;
      const int appended = (new_len > s_len0) ? (new_len - s_len0) : 0;
      saved_row_ctx[3] = -1;
      tr_state[0] = new_len;
      tr_state[1] = s_step_id + 1;
      tr_state[2] = s_ctx_p0;
      if (stats != nullptr && stats_len >= 40) {
        stats[39] += appended;
      }
    }
  }
  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
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
    const int32_t* hit_flag,
    int hit_flag_len,
    const int32_t* src_row_ptr,
    const uint64_t* __restrict__ tensor_ptrs,
    int num_tensors,
    int64_t row_stride_u4,
    int64_t copy_u4,
    int src_row,
    int pdl_mode) {
  const int b = blockIdx.x;
  uint4* base = (b < num_tensors)
      ? reinterpret_cast<uint4*>(static_cast<uintptr_t>(tensor_ptrs[b]))
      : nullptr;
  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  const int dst0 = hit_flag[0];
  if (dst0 <= 0 || b >= num_tensors) {
    if ((pdl_mode & 2) != 0) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

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
    if ((pdl_mode & 2) != 0) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  const int eff_src = (src_row_ptr != nullptr) ? src_row_ptr[0] : src_row;
  if (eff_src > 0) {
    uint4* dst = base + static_cast<int64_t>(dst0) * row_stride_u4;
    const uint4* src = base + static_cast<int64_t>(eff_src) * row_stride_u4;
    for (int64_t i = threadIdx.x; i < copy_u4; i += blockDim.x) {
      dst[i] = src[i];
    }
  }
  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
}

// Copies winning KV cache slot(p+src_r) -> slot(p+dst_r) for both KV heads (K+V = 1024 B/layer)
// across all target GQA-16 layers in a single launch (supports 1..6 multi-node deep tree hits).
// Grid: (num_layers), Block: 64 threads (1 uint4 per thread -> 2 heads x 32 uint4 = 1024 B).
__global__ void native_leaf_fix_kv_all_layers_kernel(
    const int32_t* hit_flag,
    int hit_flag_len,
    const int32_t* src_row_ptr,
    const int64_t* slot_mappings,
    int64_t slot_stride,
    const uint64_t* __restrict__ kv_ptrs,
    const int64_t* __restrict__ stride_block,
    const int64_t* __restrict__ stride_head,
    const int64_t* __restrict__ stride_slot,
    const int32_t* __restrict__ block_size,
    const int32_t* __restrict__ layer_gid,
    int num_layers,
    int src_row,
    int pdl_mode) {
  const int L = blockIdx.x;
  const int gi = (L < num_layers) ? layer_gid[L] : 0;
  const int64_t gi_off = static_cast<int64_t>(gi) * slot_stride;
  const int64_t bs = (L < num_layers) ? static_cast<int64_t>(block_size[L]) : 16;
  const int t = threadIdx.x;
  const int h = t >> 5;     // 0 or 1 (2 KV heads)
  const int vec = t & 31;   // 0..31 (32 * 16B = 512B = 256 bf16 = 128 K + 128 V)
  __nv_bfloat16* kv = (L < num_layers)
      ? reinterpret_cast<__nv_bfloat16*>(static_cast<uintptr_t>(kv_ptrs[L]))
      : nullptr;
  const int64_t sB = (L < num_layers) ? stride_block[L] : 0;
  const int64_t sH = (L < num_layers) ? stride_head[L] : 0;
  const int64_t sS = (L < num_layers) ? stride_slot[L] : 0;

  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  const int dst0 = hit_flag[0];
  if (dst0 <= 0 || L >= num_layers) {
    if ((pdl_mode & 2) != 0) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

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
    if ((pdl_mode & 2) != 0) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    return;
  }

  const int eff_src = (src_row_ptr != nullptr) ? src_row_ptr[0] : src_row;
  if (eff_src > 0) {
    const int64_t s_dst = slot_mappings[gi_off + dst0];
    const int64_t s_src = slot_mappings[gi_off + eff_src];
    if (s_dst >= 0 && s_src >= 0) {
      const int64_t blk_dst = s_dst / bs;
      const int64_t off_dst = s_dst - blk_dst * bs;
      const int64_t blk_src = s_src / bs;
      const int64_t off_src = s_src - blk_src * bs;
      uint4* dst = reinterpret_cast<uint4*>(kv + blk_dst * sB + h * sH + off_dst * sS);
      const uint4* src = reinterpret_cast<const uint4*>(kv + blk_src * sB + h * sH + off_src * sS);
      dst[vec] = src[vec];
    }
  }
  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
}

static const CalibKnobs& get_calib_knobs_env() {
  static const CalibKnobs ck = []() -> CalibKnobs {
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
  return ck;
}

static const BgEbKnobs& get_bgeb_knobs_env() {
  static const BgEbKnobs b = []() -> BgEbKnobs {
    auto env_i = [](const char* name, int def) -> int {
      const char* s = std::getenv(name);
      return (s && *s) ? static_cast<int>(std::strtol(s, nullptr, 10)) : def;
    };
    auto env_f = [](const char* name, float def) -> float {
      const char* s = std::getenv(name);
      return (s && *s) ? static_cast<float>(std::strtod(s, nullptr)) : def;
    };
    BgEbKnobs v{};
    v.bg_mode = env_i("MUSE_BG_MODE", 1);
    v.bg_sub = env_i("MUSE_BG_SUB", 1);
    v.bg_off = env_i("MUSE_BG_OFF", 1);
    v.chp_bg2 = env_i("MUSE_BG_CHP_BG2", 1);
    v.bg_alpha = env_f("MUSE_BG_ALPHA", 4.0f);
    v.bg_min_p = env_f("MUSE_BG_MIN_P", 0.10f);
    v.sub_min_p = env_f("MUSE_BG_SUB_MIN_P", 0.25f);
    v.tri_min_p = env_f("MUSE_BG_TRI_MIN_P", 0.06f);
    v.f2 = env_f("MUSE_BG_F2", 0.50f);
    v.w2 = env_f("MUSE_BG_W2", 3.00f);
    v.f3 = env_f("MUSE_BG_F3", 1.00f);
    v.w3 = env_f("MUSE_BG_W3", 1.50f);
    v.uniq_mode = env_i("MUSE_UNIQ_MODE", 1);
    v.uniq_p2 = env_f("MUSE_UNIQ_P2", 0.42f);
    v.uniq_p3 = env_f("MUSE_UNIQ_P3", 0.70f);
    v.shf_cal = env_i("MUSE_SHF_CAL", 1);
    v.shf_lo = env_f("MUSE_SHF_LO", 0.015f);
    v.shf_hi = env_f("MUSE_SHF_HI", 0.11f);
    v.eb_mode = env_i("MUSE_EB_MODE", 1);
    v.eb_raw = env_i("MUSE_EB_RAW", 1);
    v.eb_min_r = env_i("MUSE_EB_MIN_R", 4);
    v.eb_p0 = env_f("MUSE_EB_P0", 0.72f);
    v.eb_tau = env_f("MUSE_EB_TAU", 6.0f);
    v.eb_scale = env_f("MUSE_EB_SCALE", 1.0f);
    v.eb_lo = env_f("MUSE_EB_LO", -0.50f);
    v.eb_hi = env_f("MUSE_EB_HI", 0.0f);
    return v;
  }();
  return b;
}

static const AntiChimeraKnobs& get_ac_knobs_env() {
  static const AntiChimeraKnobs a = []() -> AntiChimeraKnobs {
    auto env_i = [](const char* name, int def) -> int {
      const char* s = std::getenv(name);
      return (s && *s) ? static_cast<int>(std::strtol(s, nullptr, 10)) : def;
    };
    auto env_f = [](const char* name, float def) -> float {
      const char* s = std::getenv(name);
      return (s && *s) ? static_cast<float>(std::strtod(s, nullptr)) : def;
    };
    AntiChimeraKnobs v{};
    v.ac_mode = env_i("MUSE_AC_MODE", 1);
    v.min_u = env_i("MUSE_AC_MIN_U", 4);
    v.fwd_lambda = env_f("MUSE_AC_FWD_LAMBDA", 0.45f);
    v.fwd_alpha = env_f("MUSE_AC_FWD_ALPHA", 3.0f);
    v.fwd_cap = env_f("MUSE_AC_FWD_CAP", 1.10f);
    v.min_v = env_i("MUSE_AC_MIN_V", 3);
    v.rev_lambda = env_f("MUSE_AC_REV_LAMBDA", 0.60f);
    v.rev_alpha = env_f("MUSE_AC_REV_ALPHA", 2.0f);
    v.rev_cap = env_f("MUSE_AC_REV_CAP", 0.90f);
    v.ent_lambda = env_f("MUSE_AC_ENT_LAMBDA", 0.35f);
    v.lh_copy_w = env_f("MUSE_LH_COPY_W", 0.18f);
    v.stutter_scale = env_f("MUSE_STUTTER_SCALE", 0.08f);
    v.skip_root_bg = env_f("MUSE_SKIP_ROOT_BG", 0.30f);
    v.skip_lh_w = env_f("MUSE_SKIP_LH_W", 0.22f);
    v.skip_coef = env_f("MUSE_SKIP_COEF", 0.58f);
    v.shf_unseen_scale = env_f("MUSE_SHF_UNSEEN_SCALE", 0.065f);
    return v;
  }();
  return a;
}

static const CascadeKnobs& get_cascade_knobs_env() {
  static const CascadeKnobs c = []() -> CascadeKnobs {
    auto env_i = [](const char* name, int def) -> int {
      const char* s = std::getenv(name);
      return (s && *s) ? static_cast<int>(std::strtol(s, nullptr, 10)) : def;
    };
    auto env_f = [](const char* name, float def) -> float {
      const char* s = std::getenv(name);
      return (s && *s) ? static_cast<float>(std::strtod(s, nullptr)) : def;
    };
    CascadeKnobs v{};
    v.mode = env_i("MUSE_DFLASH_CASCADE", 1);
    v.denoise_mode = env_i("MUSE_CASCADE_DENOISE", 0);
    v.force_all = env_i("MUSE_CASCADE_FORCE_ALL", 0);
    v.max_b = env_i("MUSE_CASCADE_MAX_B", 3);
    v.max_r = env_i("MUSE_CASCADE_MAX_R", 2);
    v.min_p = env_f("MUSE_CASCADE_MIN_P", 0.09f);
    v.min_qrel = env_f("MUSE_CASCADE_MIN_QREL", 0.15f);
    v.min_ev = env_f("MUSE_CASCADE_MIN_EV", 0.22f);
    v.branch_gain = env_f("MUSE_CASCADE_BRANCH_GAIN", 0.58f);
    v.denoise_min_a = env_f("MUSE_CASCADE_DENOISE_MIN_A", 0.50f);
    v.denoise_max_p1 = env_f("MUSE_CASCADE_DENOISE_MAX_P1", 0.60f);
    v.denoise_gain = env_f("MUSE_CASCADE_DENOISE_GAIN", 0.44f);
    v.ddtree_w = env_f("MUSE_DDTREE_W", 0.35f);
    v.lookahead_w = env_f("MUSE_CASCADE_LH", 0.30f);
    v.backup_w = env_f("MUSE_CASCADE_BACKUP_W", 0.45f);
    return v;
  }();
  return c;
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
    double cont_decay,
    c10::optional<torch::Tensor> hist,
    c10::optional<torch::Tensor> hist_len,
    c10::optional<torch::Tensor> saved_pos0,
    c10::optional<torch::Tensor> stats,
    c10::optional<torch::Tensor> saved_row_ctx,
    c10::optional<torch::Tensor> tr_log,
    c10::optional<torch::Tensor> tr_state,
    int64_t vocab_size,
    c10::optional<torch::Tensor> cascade_meta,
    c10::optional<torch::Tensor> cascade_toks,
    c10::optional<torch::Tensor> cascade_vals) {
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
  int acc_len = 0;
  if (req_acc_state.has_value() && req_acc_state.value().defined() && req_acc_state.value().numel() >= 3) {
    TORCH_CHECK(req_acc_state.value().is_cuda() && req_acc_state.value().scalar_type() == torch::kInt32);
    acc_ptr = req_acc_state.value().data_ptr<int32_t>();
    acc_len = static_cast<int>(req_acc_state.value().numel());
  }
  const float* vals_ptr = nullptr;
  if (tree_vals.has_value() && tree_vals.value().defined() &&
      tree_vals.value().numel() >= static_cast<int64_t>(6 * tree_k)) {
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
    p.m32_wide = env_i("MUSE_PL_M32_WIDE", 1);
    p.sh_branch_cont = env_i("MUSE_SH_BRANCH_CONT", 1);
    return p;
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
    t.b_d1_tri = env_f("MUSE_TR_B_D1_TRI", 0.90f);
    t.b_d1_bg = env_f("MUSE_TR_B_D1_BG", 0.60f);
    t.ch_a1_bg_t6 = env_f("MUSE_TR_CH_A1_BG_T6", 0.80f);
    t.ch_a1_bg_oov = env_f("MUSE_TR_CH_A1_BG_OOV", 0.58f);
    t.m32_wide = env_i("MUSE_TR_M32_WIDE", 1);
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
  int32_t* casc_meta_ptr = nullptr;
  if (cascade_meta.has_value() && cascade_meta.value().defined() && cascade_meta.value().numel() >= 8) {
    TORCH_CHECK(cascade_meta.value().is_cuda() && cascade_meta.value().scalar_type() == torch::kInt32);
    casc_meta_ptr = cascade_meta.value().data_ptr<int32_t>();
  }
  const int64_t* casc_toks_ptr = nullptr;
  if (cascade_toks.has_value() && cascade_toks.value().defined() && cascade_toks.value().numel() >= 36) {
    TORCH_CHECK(cascade_toks.value().is_cuda() && cascade_toks.value().scalar_type() == torch::kInt64);
    casc_toks_ptr = cascade_toks.value().data_ptr<int64_t>();
  }
  const float* casc_vals_ptr = nullptr;
  if (cascade_vals.has_value() && cascade_vals.value().defined() && cascade_vals.value().numel() >= 36) {
    TORCH_CHECK(cascade_vals.value().is_cuda() && cascade_vals.value().scalar_type() == torch::kFloat32);
    casc_vals_ptr = cascade_vals.value().data_ptr<float>();
  }
  const CalibKnobs& ck_env = get_calib_knobs_env();
  const BgEbKnobs& bg_env = get_bgeb_knobs_env();
  const AntiChimeraKnobs& ac_env = get_ac_knobs_env();
  const CascadeKnobs& csk_env = get_cascade_knobs_env();
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int pdl = leaf_pdl_mode();
  leaf_launch_pdl(
      native_leaf_setup_tree_kernel, 1u, 256u, st, pdl,
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
      tr_cap,
      static_cast<int>(vocab_size > 0 ? vocab_size : 202048),
      ac_env,
      casc_meta_ptr,
      casc_toks_ptr,
      casc_vals_ptr,
      csk_env,
      acc_len,
      pdl);
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
    c10::optional<torch::Tensor> tr_state,
    int64_t vocab_size,
    bool chain_all_valid,
    c10::optional<torch::Tensor> cascade_meta,
    c10::optional<torch::Tensor> in_static_vocab,
    c10::optional<torch::Tensor> tail_ids,
    c10::optional<torch::Tensor> tail_count,
    c10::optional<torch::Tensor> tail_seen,
    c10::optional<torch::Tensor> hit_cond) {
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
  int acc_len = 0;
  if (req_acc_state.has_value() && req_acc_state.value().defined() && req_acc_state.value().numel() >= 3) {
    TORCH_CHECK(req_acc_state.value().is_cuda() && req_acc_state.value().scalar_type() == torch::kInt32);
    acc_ptr = req_acc_state.value().data_ptr<int32_t>();
    acc_len = static_cast<int>(req_acc_state.value().numel());
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
  int32_t* casc_meta_ptr = nullptr;
  if (cascade_meta.has_value() && cascade_meta.value().defined() && cascade_meta.value().numel() >= 8) {
    TORCH_CHECK(cascade_meta.value().is_cuda() && cascade_meta.value().scalar_type() == torch::kInt32);
    casc_meta_ptr = cascade_meta.value().data_ptr<int32_t>();
  }
  const int eff_vocab = static_cast<int>(vocab_size > 0 ? vocab_size : 202048);
  static const int oov_preadmit_env = []() -> int {
    const char* s = std::getenv("MUSE_OOV_PREADMIT");
    return (!s || !*s) ? 2 : static_cast<int>(std::strtol(s, nullptr, 10));
  }();
  const uint8_t* in_sv_ptr = nullptr;
  int32_t* t_ids_ptr = nullptr;
  int32_t* t_cnt_ptr = nullptr;
  int32_t* t_seen_ptr = nullptr;
  int max_tail = 0;
  if (oov_preadmit_env > 0 &&
      in_static_vocab.has_value() && in_static_vocab.value().defined() &&
      tail_ids.has_value() && tail_ids.value().defined() &&
      tail_count.has_value() && tail_count.value().defined() &&
      tail_seen.has_value() && tail_seen.value().defined() &&
      in_static_vocab.value().numel() >= eff_vocab &&
      tail_seen.value().numel() >= eff_vocab &&
      tail_ids.value().numel() > 0 &&
      tail_count.value().numel() >= 1) {
    TORCH_CHECK(in_static_vocab.value().is_cuda() && in_static_vocab.value().scalar_type() == torch::kUInt8 &&
                in_static_vocab.value().is_contiguous());
    TORCH_CHECK(tail_ids.value().is_cuda() && tail_ids.value().scalar_type() == torch::kInt32 &&
                tail_ids.value().is_contiguous());
    TORCH_CHECK(tail_count.value().is_cuda() && tail_count.value().scalar_type() == torch::kInt32);
    TORCH_CHECK(tail_seen.value().is_cuda() && tail_seen.value().scalar_type() == torch::kInt32 &&
                tail_seen.value().is_contiguous());
    in_sv_ptr = in_static_vocab.value().data_ptr<uint8_t>();
    t_ids_ptr = tail_ids.value().data_ptr<int32_t>();
    t_cnt_ptr = tail_count.value().data_ptr<int32_t>();
    t_seen_ptr = tail_seen.value().data_ptr<int32_t>();
    max_tail = static_cast<int>(tail_ids.value().numel());
  }
  uint8_t* hit_cond_ptr = nullptr;
  if (hit_cond.has_value() && hit_cond.value().defined() && hit_cond.value().numel() >= 1) {
    TORCH_CHECK(hit_cond.value().is_cuda() &&
                (hit_cond.value().scalar_type() == torch::kUInt8 ||
                 hit_cond.value().scalar_type() == torch::kBool));
    hit_cond_ptr = reinterpret_cast<uint8_t*>(hit_cond.value().data_ptr());
  }
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int pdl = leaf_pdl_mode();
  leaf_launch_pdl(
      native_leaf_commit_tree_kernel, 1u, 32u, st, pdl,
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
      tr_cap,
      eff_vocab,
      chain_all_valid ? 1 : 0,
      casc_meta_ptr,
      in_sv_ptr,
      t_ids_ptr,
      t_cnt_ptr,
      t_seen_ptr,
      max_tail,
      oov_preadmit_env,
      acc_len,
      hit_cond_ptr,
      pdl);
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
  const int pdl = leaf_pdl_mode();
  leaf_launch_pdl(
      native_leaf_fix_hidden_kernel, static_cast<unsigned>(num_tensors), 256u, st, pdl,
      hit_flag.data_ptr<int32_t>(),
      static_cast<int>(hit_flag.numel()),
      src_ptr,
      reinterpret_cast<const uint64_t*>(tensor_ptrs.data_ptr<int64_t>()),
      num_tensors,
      row_stride_bytes >> 4,
      copy_bytes >> 4,
      static_cast<int>(src_row),
      pdl);
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
  const int pdl = leaf_pdl_mode();
  leaf_launch_pdl(
      native_leaf_fix_kv_all_layers_kernel, static_cast<unsigned>(num_layers), 64u, st, pdl,
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
      static_cast<int>(src_row),
      pdl);
}

namespace {

struct CondGraphEntry {
  cudaGraph_t outer_graph = nullptr;
  cudaGraphExec_t outer_exec = nullptr;
};

std::mutex g_cond_mu;
int64_t g_next_cloned_id = 1;
int64_t g_next_exec_id = 1;
std::unordered_map<int64_t, cudaGraph_t> g_cloned_graphs;
std::unordered_map<int64_t, CondGraphEntry> g_cond_execs;

CascadePrepArgs make_cascade_prep_args(
    torch::Tensor input_ids,
    torch::Tensor query_positions,
    torch::Tensor draft_tokens,
    torch::Tensor tree_top,
    torch::Tensor tree_vals,
    c10::optional<torch::Tensor> alt_ids,
    c10::optional<torch::Tensor> alt_margins,
    c10::optional<torch::Tensor> req_acc_state,
    torch::Tensor cascade_meta,
    torch::Tensor p1_save_draft,
    torch::Tensor p1_save_top,
    torch::Tensor p1_save_vals,
    c10::optional<torch::Tensor> p1_save_alt,
    c10::optional<torch::Tensor> p1_save_margins,
    c10::optional<torch::Tensor> stats,
    int64_t mask_token_id) {
  TORCH_CHECK(input_ids.is_cuda() && input_ids.scalar_type() == torch::kInt32 && input_ids.numel() >= 16);
  TORCH_CHECK(query_positions.is_cuda() && query_positions.scalar_type() == torch::kInt64 && query_positions.numel() >= 1);
  TORCH_CHECK(draft_tokens.is_cuda() && draft_tokens.scalar_type() == torch::kInt64 && draft_tokens.numel() >= 6);
  TORCH_CHECK(tree_top.is_cuda() && tree_top.scalar_type() == torch::kInt64 && tree_top.numel() >= 36);
  TORCH_CHECK(tree_vals.is_cuda() && tree_vals.scalar_type() == torch::kFloat32 && tree_vals.numel() >= 36);
  TORCH_CHECK(cascade_meta.is_cuda() && cascade_meta.scalar_type() == torch::kInt32 && cascade_meta.numel() >= 8);
  TORCH_CHECK(p1_save_draft.is_cuda() && p1_save_draft.scalar_type() == torch::kInt64 && p1_save_draft.numel() >= 6);
  TORCH_CHECK(p1_save_top.is_cuda() && p1_save_top.scalar_type() == torch::kInt64 && p1_save_top.numel() >= 36);
  TORCH_CHECK(p1_save_vals.is_cuda() && p1_save_vals.scalar_type() == torch::kFloat32 && p1_save_vals.numel() >= 36);

  CascadePrepArgs a{};
  a.input_ids = input_ids.data_ptr<int32_t>();
  a.query_positions = query_positions.data_ptr<int64_t>();
  a.draft_tokens = draft_tokens.data_ptr<int64_t>();
  a.tree_top = tree_top.data_ptr<int64_t>();
  a.tree_vals = tree_vals.data_ptr<float>();
  a.alt_ids = (alt_ids.has_value() && alt_ids.value().defined() && alt_ids.value().numel() >= 4)
      ? alt_ids.value().data_ptr<int64_t>()
      : nullptr;
  a.alt_margins = (alt_margins.has_value() && alt_margins.value().defined() && alt_margins.value().numel() >= 2)
      ? alt_margins.value().data_ptr<float>()
      : nullptr;
  if (req_acc_state.has_value() && req_acc_state.value().defined() && req_acc_state.value().numel() >= 3) {
    a.req_acc_state = req_acc_state.value().data_ptr<int32_t>();
    a.req_acc_state_len = static_cast<int>(req_acc_state.value().numel());
  } else {
    a.req_acc_state = nullptr;
    a.req_acc_state_len = 0;
  }
  a.cascade_meta = cascade_meta.data_ptr<int32_t>();
  a.p1_save_draft = p1_save_draft.data_ptr<int64_t>();
  a.p1_save_top = p1_save_top.data_ptr<int64_t>();
  a.p1_save_vals = p1_save_vals.data_ptr<float>();
  a.p1_save_alt = (p1_save_alt.has_value() && p1_save_alt.value().defined() && p1_save_alt.value().numel() >= 4)
      ? p1_save_alt.value().data_ptr<int64_t>()
      : nullptr;
  a.p1_save_margins = (p1_save_margins.has_value() && p1_save_margins.value().defined() && p1_save_margins.value().numel() >= 2)
      ? p1_save_margins.value().data_ptr<float>()
      : nullptr;
  if (stats.has_value() && stats.value().defined() && stats.value().numel() >= 4) {
    a.stats = stats.value().data_ptr<int64_t>();
    a.stats_len = static_cast<int>(stats.value().numel());
  } else {
    a.stats = nullptr;
    a.stats_len = 0;
  }
  a.mask_token_id = static_cast<int32_t>(mask_token_id);
  static const int calib_env = []() -> int {
    const char* s = std::getenv("MUSE_TREE_CALIB");
    return (!s || !*s || std::strtol(s, nullptr, 10) != 0) ? 1 : 0;
  }();
  a.calib_mode = calib_env;
  a.ck = get_calib_knobs_env();
  a.bg = get_bgeb_knobs_env();
  a.csk = get_cascade_knobs_env();
  a.cond_handle = cudaGraphConditionalHandle{};
  a.has_cond_handle = 0;
  return a;
}

}  // namespace

void dflash_cascade_gate_prep(
    torch::Tensor input_ids,
    torch::Tensor query_positions,
    torch::Tensor draft_tokens,
    torch::Tensor tree_top,
    torch::Tensor tree_vals,
    torch::Tensor cascade_meta,
    torch::Tensor p1_save_draft,
    torch::Tensor p1_save_top,
    torch::Tensor p1_save_vals,
    int64_t mask_token_id,
    c10::optional<torch::Tensor> alt_ids,
    c10::optional<torch::Tensor> alt_margins,
    c10::optional<torch::Tensor> p1_save_alt,
    c10::optional<torch::Tensor> p1_save_margins,
    c10::optional<torch::Tensor> req_acc_state,
    c10::optional<torch::Tensor> stats) {
  CascadePrepArgs args = make_cascade_prep_args(
      input_ids, query_positions, draft_tokens, tree_top, tree_vals,
      alt_ids, alt_margins, req_acc_state, cascade_meta,
      p1_save_draft, p1_save_top, p1_save_vals, p1_save_alt, p1_save_margins,
      stats, mask_token_id);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  dflash_cascade_gate_prep_kernel<<<1, 64, 0, st>>>(args);
}

void dflash_cascade_post_restore(
    torch::Tensor input_ids,
    torch::Tensor draft_tokens,
    torch::Tensor tree_top,
    torch::Tensor tree_vals,
    torch::Tensor cascade_meta,
    torch::Tensor cascade_toks,
    torch::Tensor cascade_vals,
    torch::Tensor p1_save_draft,
    torch::Tensor p1_save_top,
    torch::Tensor p1_save_vals,
    int64_t mask_token_id,
    c10::optional<torch::Tensor> alt_ids,
    c10::optional<torch::Tensor> alt_margins,
    c10::optional<torch::Tensor> p1_save_alt,
    c10::optional<torch::Tensor> p1_save_margins) {
  TORCH_CHECK(input_ids.is_cuda() && input_ids.scalar_type() == torch::kInt32 && input_ids.numel() >= 16);
  TORCH_CHECK(draft_tokens.is_cuda() && draft_tokens.scalar_type() == torch::kInt64 && draft_tokens.numel() >= 6);
  TORCH_CHECK(tree_top.is_cuda() && tree_top.scalar_type() == torch::kInt64 && tree_top.numel() >= 36);
  TORCH_CHECK(tree_vals.is_cuda() && tree_vals.scalar_type() == torch::kFloat32 && tree_vals.numel() >= 36);
  TORCH_CHECK(cascade_meta.is_cuda() && cascade_meta.scalar_type() == torch::kInt32 && cascade_meta.numel() >= 8);
  TORCH_CHECK(cascade_toks.is_cuda() && cascade_toks.scalar_type() == torch::kInt64 && cascade_toks.numel() >= 36);
  TORCH_CHECK(cascade_vals.is_cuda() && cascade_vals.scalar_type() == torch::kFloat32 && cascade_vals.numel() >= 36);
  TORCH_CHECK(p1_save_draft.is_cuda() && p1_save_draft.scalar_type() == torch::kInt64 && p1_save_draft.numel() >= 6);
  TORCH_CHECK(p1_save_top.is_cuda() && p1_save_top.scalar_type() == torch::kInt64 && p1_save_top.numel() >= 36);
  TORCH_CHECK(p1_save_vals.is_cuda() && p1_save_vals.scalar_type() == torch::kFloat32 && p1_save_vals.numel() >= 36);

  int64_t* alt_ptr = (alt_ids.has_value() && alt_ids.value().defined() && alt_ids.value().numel() >= 4)
      ? alt_ids.value().data_ptr<int64_t>()
      : nullptr;
  float* marg_ptr = (alt_margins.has_value() && alt_margins.value().defined() && alt_margins.value().numel() >= 2)
      ? alt_margins.value().data_ptr<float>()
      : nullptr;
  const int64_t* s_alt_ptr = (p1_save_alt.has_value() && p1_save_alt.value().defined() && p1_save_alt.value().numel() >= 4)
      ? p1_save_alt.value().data_ptr<int64_t>()
      : nullptr;
  const float* s_marg_ptr = (p1_save_margins.has_value() && p1_save_margins.value().defined() && p1_save_margins.value().numel() >= 2)
      ? p1_save_margins.value().data_ptr<float>()
      : nullptr;

  auto st = c10::cuda::getCurrentCUDAStream().stream();
  dflash_cascade_post_restore_kernel<<<1, 64, 0, st>>>(
      input_ids.data_ptr<int32_t>(),
      draft_tokens.data_ptr<int64_t>(),
      tree_top.data_ptr<int64_t>(),
      tree_vals.data_ptr<float>(),
      alt_ptr,
      marg_ptr,
      cascade_meta.data_ptr<int32_t>(),
      cascade_toks.data_ptr<int64_t>(),
      cascade_vals.data_ptr<float>(),
      p1_save_draft.data_ptr<int64_t>(),
      p1_save_top.data_ptr<int64_t>(),
      p1_save_vals.data_ptr<float>(),
      s_alt_ptr,
      s_marg_ptr,
      static_cast<int32_t>(mask_token_id));
}

int64_t dflash_cascade_clone_capturing_graph() {
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  unsigned long long cap_id = 0;
  cudaGraph_t raw_graph = nullptr;
  cudaError_t err = cudaStreamGetCaptureInfo(st, &status, &cap_id, &raw_graph, nullptr, nullptr, nullptr);
  TORCH_CHECK(err == cudaSuccess && status == cudaStreamCaptureStatusActive && raw_graph != nullptr,
              "dflash_cascade_clone_capturing_graph requires an active capturing stream: ",
              cudaGetErrorString(err));
  cudaGraph_t cloned = nullptr;
  err = cudaGraphClone(&cloned, raw_graph);
  TORCH_CHECK(err == cudaSuccess && cloned != nullptr,
              "cudaGraphClone failed: ", cudaGetErrorString(err));
  std::lock_guard<std::mutex> lk(g_cond_mu);
  const int64_t id = g_next_cloned_id++;
  g_cloned_graphs[id] = cloned;
  return id;
}

int64_t dflash_cascade_build_cond_graph(
    int64_t graph_handle,
    bool is_raw_cuda_graph_ptr,
    torch::Tensor input_ids,
    torch::Tensor query_positions,
    torch::Tensor draft_tokens,
    torch::Tensor tree_top,
    torch::Tensor tree_vals,
    torch::Tensor cascade_meta,
    torch::Tensor p1_save_draft,
    torch::Tensor p1_save_top,
    torch::Tensor p1_save_vals,
    int64_t mask_token_id,
    c10::optional<torch::Tensor> alt_ids,
    c10::optional<torch::Tensor> alt_margins,
    c10::optional<torch::Tensor> p1_save_alt,
    c10::optional<torch::Tensor> p1_save_margins,
    c10::optional<torch::Tensor> req_acc_state,
    c10::optional<torch::Tensor> stats) {
  cudaGraph_t src_graph = nullptr;
  cudaGraph_t owned_clone = nullptr;
  if (is_raw_cuda_graph_ptr) {
    src_graph = reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(graph_handle));
    TORCH_CHECK(src_graph != nullptr, "Null raw cudaGraph_t pointer");
  } else {
    std::lock_guard<std::mutex> lk(g_cond_mu);
    auto it = g_cloned_graphs.find(graph_handle);
    TORCH_CHECK(it != g_cloned_graphs.end(), "Unknown cloned graph handle: ", graph_handle);
    src_graph = it->second;
    owned_clone = src_graph;
    g_cloned_graphs.erase(it);
  }

  CascadePrepArgs prep_args = make_cascade_prep_args(
      input_ids, query_positions, draft_tokens, tree_top, tree_vals,
      alt_ids, alt_margins, req_acc_state, cascade_meta,
      p1_save_draft, p1_save_top, p1_save_vals, p1_save_alt, p1_save_margins,
      stats, mask_token_id);

  cudaGraph_t outer_graph = nullptr;
  cudaError_t err = cudaGraphCreate(&outer_graph, 0);
  TORCH_CHECK(err == cudaSuccess && outer_graph != nullptr,
              "cudaGraphCreate failed: ", cudaGetErrorString(err));

  cudaGraphConditionalHandle cond_handle{};
  err = cudaGraphConditionalHandleCreate(&cond_handle, outer_graph, 0u, cudaGraphCondAssignDefault);
  if (err != cudaSuccess) {
    cudaGraphDestroy(outer_graph);
    if (owned_clone != nullptr) cudaGraphDestroy(owned_clone);
    TORCH_CHECK(false, "cudaGraphConditionalHandleCreate failed: ", cudaGetErrorString(err));
  }
  prep_args.cond_handle = cond_handle;
  prep_args.has_cond_handle = 1;

  cudaKernelNodeParams kparams{};
  kparams.func = reinterpret_cast<void*>(dflash_cascade_gate_prep_kernel);
  kparams.gridDim = dim3(1, 1, 1);
  kparams.blockDim = dim3(64, 1, 1);
  kparams.sharedMemBytes = 0;
  void* kargs[1] = {&prep_args};
  kparams.kernelParams = kargs;
  kparams.extra = nullptr;

  cudaGraphNode_t prep_node = nullptr;
  err = cudaGraphAddKernelNode(&prep_node, outer_graph, nullptr, 0, &kparams);
  if (err != cudaSuccess) {
    cudaGraphDestroy(outer_graph);
    if (owned_clone != nullptr) cudaGraphDestroy(owned_clone);
    TORCH_CHECK(false, "cudaGraphAddKernelNode(prep) failed: ", cudaGetErrorString(err));
  }

  cudaGraphNodeParams cparams{};
  std::memset(&cparams, 0, sizeof(cparams));
  cparams.type = cudaGraphNodeTypeConditional;
  cparams.conditional.handle = cond_handle;
  cparams.conditional.type = cudaGraphCondTypeIf;
  cparams.conditional.size = 1;

  cudaGraphNode_t cond_node = nullptr;
  err = cudaGraphAddNode(&cond_node, outer_graph, &prep_node, nullptr, 1, &cparams);
  if (err != cudaSuccess) {
    cudaGraphDestroy(outer_graph);
    if (owned_clone != nullptr) cudaGraphDestroy(owned_clone);
    TORCH_CHECK(false, "cudaGraphAddNode(conditional) failed: ", cudaGetErrorString(err));
  }

  cudaGraph_t body_graph = cparams.conditional.phGraph_out[0];
  cudaGraphNode_t child_node = nullptr;
  err = cudaGraphAddChildGraphNode(&child_node, body_graph, nullptr, 0, src_graph);
  if (owned_clone != nullptr) {
    cudaGraphDestroy(owned_clone);
    owned_clone = nullptr;
  }
  if (err != cudaSuccess) {
    cudaGraphDestroy(outer_graph);
    TORCH_CHECK(false, "cudaGraphAddChildGraphNode(body) failed: ", cudaGetErrorString(err));
  }

  cudaGraphExec_t outer_exec = nullptr;
  err = cudaGraphInstantiate(&outer_exec, outer_graph, 0);
  if (err != cudaSuccess) {
    cudaGraphDestroy(outer_graph);
    TORCH_CHECK(false, "cudaGraphInstantiate(cond) failed: ", cudaGetErrorString(err));
  }
  {
    auto st_up = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t up_err = cudaGraphUpload(outer_exec, st_up);
    if (up_err != cudaSuccess) cudaGetLastError();
  }

  std::lock_guard<std::mutex> lk(g_cond_mu);
  const int64_t exec_id = g_next_exec_id++;
  g_cond_execs[exec_id] = CondGraphEntry{outer_graph, outer_exec};
  return exec_id;
}

namespace {

struct AdaptiveVerifyGateArgs {
  const float* tree_vals;
  const int32_t* req_acc_state;
  int tree_k;
  int min_rounds;
  float wide_cpc;
  float mid_cpc;
  float early_sm;
  float early_m0;
  float skip_wide_sm;
  float skip_wide_m0;
  float burst_wide_sm;
  float burst_wide_m0;
  int force_mode;  // 0=adaptive, 1=always M=16, 2=always M=32
  cudaGraphConditionalHandle h_m16;
  cudaGraphConditionalHandle h_m32;
};

__global__ void __launch_bounds__(32) native_leaf_choose_m_cond_kernel(AdaptiveVerifyGateArgs a) {
  if (threadIdx.x != 0) return;
  bool use_m32 = false;
  if (a.force_mode == 2) {
    use_m32 = true;
  } else if (a.force_mode == 1) {
    use_m32 = false;
  } else {
    const int tk = (a.tree_k >= 8) ? 8 : 6;
    float min_m03 = 2.5f;
    float sm4_sum = 0.0f;
    if (a.tree_vals != nullptr) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float z0 = a.tree_vals[j * tk + 0];
        const float z1 = a.tree_vals[j * tk + 1];
        const float mj = (z0 > -1e20f && z1 > -1e20f) ? (z0 - z1) : 2.5f;
        if (j == 0 || mj < min_m03) min_m03 = mj;
        sm4_sum += mj;
      }
    } else {
      sm4_sum = 10.0f;
    }
    const float sm4 = 0.25f * sm4_sum;
    const int32_t rounds = (a.req_acc_state != nullptr) ? a.req_acc_state[0] : 0;
    const int32_t committed = (a.req_acc_state != nullptr) ? a.req_acc_state[1] : 0;
    if (rounds <= 0) {
      use_m32 = (a.min_rounds <= 0) ? true : (sm4 < a.early_sm && min_m03 < a.early_m0);
    } else if (rounds < a.min_rounds) {
      use_m32 = (sm4 < a.early_sm && min_m03 < a.early_m0);
    } else {
      const float cpc = static_cast<float>(committed) / static_cast<float>(rounds);
      if (cpc < a.wide_cpc) {
        use_m32 = !(sm4 >= a.skip_wide_sm && min_m03 >= a.skip_wide_m0);
      } else if (cpc < a.mid_cpc) {
        use_m32 = (sm4 < a.burst_wide_sm && min_m03 < a.burst_wide_m0);
      } else {
        use_m32 = false;
      }
    }
  }
  cudaGraphSetConditional(a.h_m32, use_m32 ? 1u : 0u);
  cudaGraphSetConditional(a.h_m16, use_m32 ? 0u : 1u);
}

struct PacedL2PrefetchArgs {
  uint64_t ptr0;
  uint64_t span0;
  uint32_t hot0;
  uint64_t ptr1;
  uint64_t span1;
  uint32_t hot1;
  uint32_t step_ns;
};

struct CompoundL2PfConfig {
  uint64_t fc_ptr = 0;
  uint64_t fc_span = 0;
  uint64_t fkv_ptr = 0;
  uint64_t fkv_span = 0;
  uint64_t l0_qkv_ptr = 0;
  uint64_t l0_qkv_span = 0;
  uint64_t l0_gate_ptr = 0;
  uint64_t l0_gate_span = 0;
};
static CompoundL2PfConfig g_l2pf_cfg{};

__device__ __forceinline__ void muse_bulk_pf_16k(const uint8_t* p, uint32_t n) {
  uint64_t pol;
  asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;\n"
               :: "l"(p), "r"(n), "l"(pol) : "memory");
}

__global__ void __launch_bounds__(64) muse_paced_l2_prefetch_kernel(PacedL2PrefetchArgs a) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t nthr = gridDim.x * blockDim.x;
  // Phase 1: Warm 2 MiB GMMU TLB pages across full weight spans (~1 us).
  constexpr uint64_t kPage = 1ull << 21;
  const uint32_t pages0 = static_cast<uint32_t>((a.span0 + kPage - 1ull) >> 21);
  const uint32_t pages1 = static_cast<uint32_t>((a.span1 + kPage - 1ull) >> 21);
  for (uint32_t pg = tid; pg < pages0 + pages1; pg += nthr) {
    const uint8_t* p = (pg < pages0)
        ? (reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a.ptr0)) + (static_cast<uint64_t>(pg) << 21))
        : (reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a.ptr1)) + (static_cast<uint64_t>(pg - pages0) << 21));
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
  }
  // Phase 2: %globaltimer-paced 16 KiB bulk L2::evict_last prefetch on hot Wave-0 prefix.
  constexpr uint32_t kPiece = 16384u;
  const uint32_t h0 = (a.ptr0 != 0 && (a.ptr0 & 15ull) == 0) ? (a.hot0 & ~15u) : 0u;
  const uint32_t h1 = (a.ptr1 != 0 && (a.ptr1 & 15ull) == 0) ? (a.hot1 & ~15u) : 0u;
  const uint32_t c0 = (h0 + kPiece - 1u) / kPiece;
  const uint32_t c1 = (h1 + kPiece - 1u) / kPiece;
  const uint32_t total_c = c0 + c1;
  uint64_t t_start = 0;
  if (a.step_ns > 0 && total_c > 0) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_start));
  }
  uint32_t round_idx = 0;
  for (uint32_t idx = tid; idx < total_c; idx += nthr, ++round_idx) {
    if (a.step_ns > 0 && round_idx > 0) {
      const uint64_t target = t_start + static_cast<uint64_t>(round_idx) * static_cast<uint64_t>(a.step_ns);
      uint64_t now = 0;
      for (int spin = 0; spin < 64; ++spin) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if (now >= target) break;
      }
    }
    if (idx < c0) {
      const uint32_t off = idx * kPiece;
      const uint32_t len = (h0 - off < kPiece) ? (h0 - off) : kPiece;
      if (len >= 16u) {
        muse_bulk_pf_16k(reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a.ptr0)) + off, len & ~15u);
      }
    } else {
      const uint32_t off = (idx - c0) * kPiece;
      const uint32_t len = (h1 - off < kPiece) ? (h1 - off) : kPiece;
      if (len >= 16u) {
        muse_bulk_pf_16k(reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a.ptr1)) + off, len & ~15u);
      }
    }
  }
}

static bool chain_child_graphs(cudaGraph_t body, const std::vector<int64_t>& ptrs) {
  cudaGraphNode_t prev = nullptr;
  for (int64_t raw : ptrs) {
    if (raw == 0) continue;
    cudaGraph_t cg = reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(raw));
    cudaGraphNode_t cur = nullptr;
    cudaError_t err = cudaGraphAddChildGraphNode(
        &cur, body, (prev != nullptr) ? &prev : nullptr, (prev != nullptr) ? 1 : 0, cg);
    if (err != cudaSuccess) return false;
    prev = cur;
  }
  return true;
}

static bool chain_verify_branch_with_l2pf(
    cudaGraph_t body,
    const std::vector<int64_t>& ptrs,
    bool enable_l2pf,
    const PacedL2PrefetchArgs& l0_args,
    const PacedL2PrefetchArgs& fc_args) {
  if (!enable_l2pf || ptrs.size() != 4) {
    return chain_child_graphs(body, ptrs);
  }
  cudaGraph_t cg_pre = reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(ptrs[0]));
  cudaGraph_t cg_full = reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(ptrs[1]));
  cudaGraph_t cg_lmp = reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(ptrs[2]));
  cudaGraph_t cg_cf = reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(ptrs[3]));

  cudaGraphNode_t pre_node = nullptr;
  if (cudaGraphAddChildGraphNode(&pre_node, body, nullptr, 0, cg_pre) != cudaSuccess) return false;

  cudaGraphNode_t full_deps[2] = {pre_node, nullptr};
  size_t n_full_deps = 1;
  if (l0_args.ptr0 != 0 || l0_args.ptr1 != 0) {
    PacedL2PrefetchArgs l0_local = l0_args;
    cudaKernelNodeParams kp{};
    kp.func = reinterpret_cast<void*>(muse_paced_l2_prefetch_kernel);
    kp.gridDim = dim3(2, 1, 1);
    kp.blockDim = dim3(64, 1, 1);
    kp.sharedMemBytes = 0;
    void* kargs[1] = {&l0_local};
    kp.kernelParams = kargs;
    cudaGraphNode_t l0_pf_node = nullptr;
    if (cudaGraphAddKernelNode(&l0_pf_node, body, nullptr, 0, &kp) != cudaSuccess) return false;
    full_deps[1] = l0_pf_node;
    n_full_deps = 2;
  }

  cudaGraphNode_t full_node = nullptr;
  if (cudaGraphAddChildGraphNode(&full_node, body, full_deps, n_full_deps, cg_full) != cudaSuccess) return false;

  cudaGraphNode_t lmp_node = nullptr;
  if (cudaGraphAddChildGraphNode(&lmp_node, body, &full_node, 1, cg_lmp) != cudaSuccess) return false;

  cudaGraphNode_t cf_node = nullptr;
  if (cudaGraphAddChildGraphNode(&cf_node, body, &lmp_node, 1, cg_cf) != cudaSuccess) return false;

  if (fc_args.ptr0 != 0 || fc_args.ptr1 != 0) {
    PacedL2PrefetchArgs fc_local = fc_args;
    cudaKernelNodeParams kp{};
    kp.func = reinterpret_cast<void*>(muse_paced_l2_prefetch_kernel);
    kp.gridDim = dim3(2, 1, 1);
    kp.blockDim = dim3(64, 1, 1);
    kp.sharedMemBytes = 0;
    void* kargs[1] = {&fc_local};
    kp.kernelParams = kargs;
    cudaGraphNode_t fc_pf_node = nullptr;
    if (cudaGraphAddKernelNode(&fc_pf_node, body, &lmp_node, 1, &kp) != cudaSuccess) return false;
  }
  return true;
}

struct StoreHostOutArgs {
  const int64_t* sampled_tokens;
  const int32_t* num_sampled;
  int64_t* uva_dst;
};

__global__ void __launch_bounds__(32) native_leaf_store_host_out_kernel(StoreHostOutArgs a) {
  const int lane = threadIdx.x & 31;
  if (lane < 7) {
    a.uva_dst[lane] = a.sampled_tokens[lane];
  } else if (lane == 7) {
    a.uva_dst[7] = static_cast<int64_t>(a.num_sampled[0]);
  }
  __threadfence_system();
}

}  // namespace

int64_t native_leaf_build_adaptive_verify_graph(
    std::vector<int64_t> m16_graphs,
    std::vector<int64_t> m32_graphs,
    torch::Tensor tree_vals,
    torch::Tensor req_acc_state,
    int64_t tree_k,
    int64_t min_rounds,
    double wide_cpc,
    double mid_cpc,
    int64_t force_mode) {
  TORCH_CHECK(!m16_graphs.empty() && !m32_graphs.empty(), "empty m16/m32 graph list");
  TORCH_CHECK(tree_vals.is_cuda() && tree_vals.scalar_type() == torch::kFloat32);
  TORCH_CHECK(req_acc_state.is_cuda() && req_acc_state.scalar_type() == torch::kInt32);

  auto env_f = [](const char* name, float def) -> float {
    const char* s = std::getenv(name);
    return (s && *s) ? static_cast<float>(std::strtod(s, nullptr)) : def;
  };

  cudaGraph_t outer_graph = nullptr;
  cudaError_t err = cudaGraphCreate(&outer_graph, 0);
  TORCH_CHECK(err == cudaSuccess && outer_graph != nullptr,
              "cudaGraphCreate(adaptive) failed: ", cudaGetErrorString(err));

  cudaGraphConditionalHandle h_m16{}, h_m32{};
  if (cudaGraphConditionalHandleCreate(&h_m16, outer_graph, 1u, cudaGraphCondAssignDefault) != cudaSuccess ||
      cudaGraphConditionalHandleCreate(&h_m32, outer_graph, 0u, cudaGraphCondAssignDefault) != cudaSuccess) {
    cudaGraphDestroy(outer_graph);
    TORCH_CHECK(false, "cudaGraphConditionalHandleCreate(adaptive) failed");
  }

  AdaptiveVerifyGateArgs gargs{};
  gargs.tree_vals = tree_vals.data_ptr<float>();
  gargs.req_acc_state = req_acc_state.data_ptr<int32_t>();
  gargs.tree_k = static_cast<int>(tree_k);
  gargs.min_rounds = static_cast<int>(min_rounds);
  gargs.wide_cpc = static_cast<float>(wide_cpc);
  gargs.mid_cpc = static_cast<float>(mid_cpc);
  gargs.early_sm = env_f("MUSE_ADAPT_EARLY_SM", 1.35f);
  gargs.early_m0 = env_f("MUSE_ADAPT_EARLY_M0", 1.40f);
  gargs.skip_wide_sm = env_f("MUSE_ADAPT_SKIP_WIDE_SM", 99.0f);
  gargs.skip_wide_m0 = env_f("MUSE_ADAPT_SKIP_WIDE_M0", 99.0f);
  gargs.burst_wide_sm = env_f("MUSE_ADAPT_BURST_WIDE_SM", 1.10f);
  gargs.burst_wide_m0 = env_f("MUSE_ADAPT_BURST_WIDE_M0", 0.95f);
  gargs.force_mode = static_cast<int>(force_mode);
  gargs.h_m16 = h_m16;
  gargs.h_m32 = h_m32;

  cudaKernelNodeParams kparams{};
  kparams.func = reinterpret_cast<void*>(native_leaf_choose_m_cond_kernel);
  kparams.gridDim = dim3(1, 1, 1);
  kparams.blockDim = dim3(32, 1, 1);
  kparams.sharedMemBytes = 0;
  void* kargs[1] = {&gargs};
  kparams.kernelParams = kargs;

  cudaGraphNode_t gate_node = nullptr;
  if (cudaGraphAddKernelNode(&gate_node, outer_graph, nullptr, 0, &kparams) != cudaSuccess) {
    cudaGraphDestroy(outer_graph);
    TORCH_CHECK(false, "cudaGraphAddKernelNode(gate) failed");
  }

  cudaGraphNodeParams cp16{};
  cp16.type = cudaGraphNodeTypeConditional;
  cp16.conditional.handle = h_m16;
  cp16.conditional.type = cudaGraphCondTypeIf;
  cp16.conditional.size = 1;
  cudaGraphNode_t cond16_node = nullptr;
  if (cudaGraphAddNode(&cond16_node, outer_graph, &gate_node, nullptr, 1, &cp16) != cudaSuccess ||
      !chain_child_graphs(cp16.conditional.phGraph_out[0], m16_graphs)) {
    cudaGraphDestroy(outer_graph);
    TORCH_CHECK(false, "cudaGraphAddNode(cond16) failed");
  }

  cudaGraphNodeParams cp32{};
  cp32.type = cudaGraphNodeTypeConditional;
  cp32.conditional.handle = h_m32;
  cp32.conditional.type = cudaGraphCondTypeIf;
  cp32.conditional.size = 1;
  cudaGraphNode_t cond32_node = nullptr;
  if (cudaGraphAddNode(&cond32_node, outer_graph, &cond16_node, nullptr, 1, &cp32) != cudaSuccess ||
      !chain_child_graphs(cp32.conditional.phGraph_out[0], m32_graphs)) {
    cudaGraphDestroy(outer_graph);
    TORCH_CHECK(false, "cudaGraphAddNode(cond32) failed");
  }

  cudaGraphExec_t outer_exec = nullptr;
  err = cudaGraphInstantiate(&outer_exec, outer_graph, 0);
  if (err != cudaSuccess) {
    cudaGraphDestroy(outer_graph);
    TORCH_CHECK(false, "cudaGraphInstantiate(adaptive) failed: ", cudaGetErrorString(err));
  }
  {
    auto st_up = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t up_err = cudaGraphUpload(outer_exec, st_up);
    if (up_err != cudaSuccess) cudaGetLastError();
  }

  std::lock_guard<std::mutex> lk(g_cond_mu);
  const int64_t exec_id = g_next_exec_id++;
  g_cond_execs[exec_id] = CondGraphEntry{outer_graph, outer_exec};
  return exec_id;
}

int64_t native_leaf_build_compound_step_graph(
    std::vector<int64_t> m16_graphs,
    std::vector<int64_t> m32_graphs,
    std::vector<int64_t> tail_graphs,
    torch::Tensor sampled_tokens,
    torch::Tensor num_sampled,
    torch::Tensor host_out_row,
    int64_t raw_cuda_event,
    torch::Tensor tree_vals,
    torch::Tensor req_acc_state,
    int64_t tree_k,
    int64_t min_rounds,
    double wide_cpc,
    double mid_cpc,
    int64_t force_mode) {
  TORCH_CHECK(!m16_graphs.empty() && !m32_graphs.empty() && !tail_graphs.empty(),
              "empty m16/m32/tail graph list");
  TORCH_CHECK(sampled_tokens.is_cuda() && sampled_tokens.scalar_type() == torch::kInt64 &&
              sampled_tokens.numel() >= 7);
  TORCH_CHECK(num_sampled.is_cuda() && num_sampled.scalar_type() == torch::kInt32 &&
              num_sampled.numel() >= 1);
  TORCH_CHECK(!host_out_row.is_cuda() && host_out_row.scalar_type() == torch::kInt64 &&
              host_out_row.is_contiguous() && host_out_row.numel() >= 8,
              "host_out_row must be contiguous CPU int64[>=8] pinned tensor");
  TORCH_CHECK(raw_cuda_event != 0, "raw_cuda_event must be materialized before capture");
  TORCH_CHECK(tree_vals.is_cuda() && tree_vals.scalar_type() == torch::kFloat32);
  TORCH_CHECK(req_acc_state.is_cuda() && req_acc_state.scalar_type() == torch::kInt32);

  void* uva_raw = nullptr;
  cudaError_t herr = cudaHostGetDevicePointer(&uva_raw, host_out_row.data_ptr<int64_t>(), 0);
  TORCH_CHECK(herr == cudaSuccess && uva_raw != nullptr,
              "cudaHostGetDevicePointer(compound) failed: ", cudaGetErrorString(herr));

  auto env_f = [](const char* name, float def) -> float {
    const char* s = std::getenv(name);
    return (s && *s) ? static_cast<float>(std::strtod(s, nullptr)) : def;
  };

  cudaGraph_t outer_graph = nullptr;
  cudaError_t err = cudaGraphCreate(&outer_graph, 0);
  TORCH_CHECK(err == cudaSuccess && outer_graph != nullptr,
              "cudaGraphCreate(compound) failed: ", cudaGetErrorString(err));

  cudaGraphConditionalHandle h_m16{}, h_m32{};
  if (cudaGraphConditionalHandleCreate(&h_m16, outer_graph, 1u, cudaGraphCondAssignDefault) != cudaSuccess ||
      cudaGraphConditionalHandleCreate(&h_m32, outer_graph, 0u, cudaGraphCondAssignDefault) != cudaSuccess) {
    cudaGraphDestroy(outer_graph);
    TORCH_CHECK(false, "cudaGraphConditionalHandleCreate(compound) failed");
  }

  AdaptiveVerifyGateArgs gargs{};
  gargs.tree_vals = tree_vals.data_ptr<float>();
  gargs.req_acc_state = req_acc_state.data_ptr<int32_t>();
  gargs.tree_k = static_cast<int>(tree_k);
  gargs.min_rounds = static_cast<int>(min_rounds);
  gargs.wide_cpc = static_cast<float>(wide_cpc);
  gargs.mid_cpc = static_cast<float>(mid_cpc);
  gargs.early_sm = env_f("MUSE_ADAPT_EARLY_SM", 1.35f);
  gargs.early_m0 = env_f("MUSE_ADAPT_EARLY_M0", 1.40f);
  gargs.skip_wide_sm = env_f("MUSE_ADAPT_SKIP_WIDE_SM", 99.0f);
  gargs.skip_wide_m0 = env_f("MUSE_ADAPT_SKIP_WIDE_M0", 99.0f);
  gargs.burst_wide_sm = env_f("MUSE_ADAPT_BURST_WIDE_SM", 1.10f);
  gargs.burst_wide_m0 = env_f("MUSE_ADAPT_BURST_WIDE_M0", 0.95f);
  gargs.force_mode = static_cast<int>(force_mode);
  gargs.h_m16 = h_m16;
  gargs.h_m32 = h_m32;

  cudaKernelNodeParams kparams{};
  kparams.func = reinterpret_cast<void*>(native_leaf_choose_m_cond_kernel);
  kparams.gridDim = dim3(1, 1, 1);
  kparams.blockDim = dim3(32, 1, 1);
  kparams.sharedMemBytes = 0;
  void* kargs[1] = {&gargs};
  kparams.kernelParams = kargs;

  cudaGraphNode_t gate_node = nullptr;
  if (cudaGraphAddKernelNode(&gate_node, outer_graph, nullptr, 0, &kparams) != cudaSuccess) {
    cudaGraphDestroy(outer_graph);
    TORCH_CHECK(false, "cudaGraphAddKernelNode(compound gate) failed");
  }

  cudaGraphNodeParams cp16{};
  cp16.type = cudaGraphNodeTypeConditional;
  cp16.conditional.handle = h_m16;
  cp16.conditional.type = cudaGraphCondTypeIf;
  cp16.conditional.size = 1;
  const char* l2pf_env = std::getenv("MUSE_COMPOUND_L2PF");
  const bool enable_l2pf = (l2pf_env && l2pf_env[0] == '1');
  const uint32_t l0_kb = static_cast<uint32_t>(env_f("MUSE_L2PF_L0_KB", 1536.0f)) * 1024u;
  const uint32_t fc_kb = static_cast<uint32_t>(env_f("MUSE_L2PF_FC_KB", 2048.0f)) * 1024u;
  const uint32_t pf_step_ns = static_cast<uint32_t>(env_f("MUSE_L2PF_STEP_NS", 80.0f));
  PacedL2PrefetchArgs l0_pf{};
  l0_pf.ptr0 = g_l2pf_cfg.l0_qkv_ptr;
  l0_pf.span0 = g_l2pf_cfg.l0_qkv_span;
  l0_pf.hot0 = (g_l2pf_cfg.l0_qkv_span > 0 && l0_kb < g_l2pf_cfg.l0_qkv_span)
      ? l0_kb : static_cast<uint32_t>(g_l2pf_cfg.l0_qkv_span);
  l0_pf.ptr1 = g_l2pf_cfg.l0_gate_ptr;
  l0_pf.span1 = g_l2pf_cfg.l0_gate_span;
  l0_pf.hot1 = (g_l2pf_cfg.l0_gate_span > 0 && (l0_kb / 2u) < g_l2pf_cfg.l0_gate_span)
      ? (l0_kb / 2u) : static_cast<uint32_t>(g_l2pf_cfg.l0_gate_span);
  l0_pf.step_ns = pf_step_ns;

  PacedL2PrefetchArgs fc_pf{};
  fc_pf.ptr0 = g_l2pf_cfg.fc_ptr;
  fc_pf.span0 = g_l2pf_cfg.fc_span;
  fc_pf.hot0 = (g_l2pf_cfg.fc_span > 0 && fc_kb < g_l2pf_cfg.fc_span)
      ? fc_kb : static_cast<uint32_t>(g_l2pf_cfg.fc_span);
  fc_pf.ptr1 = g_l2pf_cfg.fkv_ptr;
  fc_pf.span1 = g_l2pf_cfg.fkv_span;
  fc_pf.hot1 = (g_l2pf_cfg.fkv_span > 0 && 524288u < g_l2pf_cfg.fkv_span)
      ? 524288u : static_cast<uint32_t>(g_l2pf_cfg.fkv_span);
  fc_pf.step_ns = pf_step_ns;

  cudaGraphNode_t cond16_node = nullptr;
  if (cudaGraphAddNode(&cond16_node, outer_graph, &gate_node, nullptr, 1, &cp16) != cudaSuccess ||
      !chain_verify_branch_with_l2pf(cp16.conditional.phGraph_out[0], m16_graphs, enable_l2pf, l0_pf, fc_pf)) {
    cudaGraphDestroy(outer_graph);
    TORCH_CHECK(false, "cudaGraphAddNode(compound cond16) failed");
  }

  cudaGraphNodeParams cp32{};
  cp32.type = cudaGraphNodeTypeConditional;
  cp32.conditional.handle = h_m32;
  cp32.conditional.type = cudaGraphCondTypeIf;
  cp32.conditional.size = 1;
  cudaGraphNode_t cond32_node = nullptr;
  if (cudaGraphAddNode(&cond32_node, outer_graph, &cond16_node, nullptr, 1, &cp32) != cudaSuccess ||
      !chain_verify_branch_with_l2pf(cp32.conditional.phGraph_out[0], m32_graphs, enable_l2pf, l0_pf, fc_pf)) {
    cudaGraphDestroy(outer_graph);
    TORCH_CHECK(false, "cudaGraphAddNode(compound cond32) failed");
  }

  StoreHostOutArgs sh_args{};
  sh_args.sampled_tokens = sampled_tokens.data_ptr<int64_t>();
  sh_args.num_sampled = num_sampled.data_ptr<int32_t>();
  sh_args.uva_dst = reinterpret_cast<int64_t*>(uva_raw);

  cudaKernelNodeParams sparams{};
  sparams.func = reinterpret_cast<void*>(native_leaf_store_host_out_kernel);
  sparams.gridDim = dim3(1, 1, 1);
  sparams.blockDim = dim3(32, 1, 1);
  sparams.sharedMemBytes = 0;
  void* sargs[1] = {&sh_args};
  sparams.kernelParams = sargs;

  cudaGraphNode_t store_node = nullptr;
  if (cudaGraphAddKernelNode(&store_node, outer_graph, &cond32_node, 1, &sparams) != cudaSuccess) {
    cudaGraphDestroy(outer_graph);
    TORCH_CHECK(false, "cudaGraphAddKernelNode(store_host_out) failed");
  }

  cudaEvent_t cev = reinterpret_cast<cudaEvent_t>(static_cast<uintptr_t>(raw_cuda_event));
  cudaGraphNode_t ev_node = nullptr;
  if (cudaGraphAddEventRecordNode(&ev_node, outer_graph, &store_node, 1, cev) != cudaSuccess) {
    cudaGraphDestroy(outer_graph);
    TORCH_CHECK(false, "cudaGraphAddEventRecordNode(copy_event) failed");
  }

  static const bool fork_host_out = []() -> bool {
    const char* s = std::getenv("MUSE_COMPOUND_FORK_HOST_OUT");
    return s && *s && std::atoi(s) != 0;
  }();
  cudaGraphNode_t tail_root = fork_host_out ? cond32_node : ev_node;

  if (tail_graphs.size() == 4 && tail_graphs[0] != 0 && tail_graphs[1] != 0 &&
      tail_graphs[2] != 0 && tail_graphs[3] != 0) {
    // 2-branch Diamond DAG:
    //   Arm A: tail_root -> post_prep_node (tail_graphs[0])
    //   Arm B: cond32_node -> ctx_proj_node (tail_graphs[1]: fc + hidden_norm + fkv)
    //   Join:  {post_prep_node, ctx_proj_node} -> ctx_kv_post_node (tail_graphs[2]) -> query_node (tail_graphs[3])
    cudaGraph_t cg_pp = reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(tail_graphs[0]));
    cudaGraph_t cg_proj = reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(tail_graphs[1]));
    cudaGraph_t cg_kvp = reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(tail_graphs[2]));
    cudaGraph_t cg_qry = reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(tail_graphs[3]));

    cudaGraphNode_t pp_node = nullptr;
    cudaGraphNode_t proj_node = nullptr;
    if (cudaGraphAddChildGraphNode(&pp_node, outer_graph, &tail_root, 1, cg_pp) != cudaSuccess ||
        cudaGraphAddChildGraphNode(&proj_node, outer_graph, &cond32_node, 1, cg_proj) != cudaSuccess) {
      cudaGraphDestroy(outer_graph);
      TORCH_CHECK(false, "cudaGraphAddChildGraphNode(diamond arm A/B) failed");
    }
    cudaGraphNode_t join_deps[2] = {pp_node, proj_node};
    cudaGraphNode_t kvp_node = nullptr;
    cudaGraphNode_t qry_node = nullptr;
    if (cudaGraphAddChildGraphNode(&kvp_node, outer_graph, join_deps, 2, cg_kvp) != cudaSuccess ||
        cudaGraphAddChildGraphNode(&qry_node, outer_graph, &kvp_node, 1, cg_qry) != cudaSuccess) {
      cudaGraphDestroy(outer_graph);
      TORCH_CHECK(false, "cudaGraphAddChildGraphNode(diamond join) failed");
    }
  } else {
    cudaGraphNode_t prev = tail_root;
    for (int64_t raw : tail_graphs) {
      if (raw == 0) continue;
      cudaGraph_t cg = reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(raw));
      cudaGraphNode_t cur = nullptr;
      if (cudaGraphAddChildGraphNode(&cur, outer_graph, &prev, 1, cg) != cudaSuccess) {
        cudaGraphDestroy(outer_graph);
        TORCH_CHECK(false, "cudaGraphAddChildGraphNode(tail) failed");
      }
      prev = cur;
    }
  }

  cudaGraphExec_t outer_exec = nullptr;
  err = cudaGraphInstantiate(&outer_exec, outer_graph, 0);
  if (err != cudaSuccess) {
    cudaGraphDestroy(outer_graph);
    TORCH_CHECK(false, "cudaGraphInstantiate(compound) failed: ", cudaGetErrorString(err));
  }
  {
    auto st_up = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t up_err = cudaGraphUpload(outer_exec, st_up);
    if (up_err != cudaSuccess) cudaGetLastError();
  }

  std::lock_guard<std::mutex> lk(g_cond_mu);
  const int64_t exec_id = g_next_exec_id++;
  g_cond_execs[exec_id] = CondGraphEntry{outer_graph, outer_exec};
  return exec_id;
}

void cuda_graph_upload(int64_t raw_exec_ptr) {
  if (raw_exec_ptr == 0) return;
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  cudaError_t err = cudaGraphUpload(
      reinterpret_cast<cudaGraphExec_t>(static_cast<uintptr_t>(raw_exec_ptr)), st);
  if (err != cudaSuccess) {
    cudaGetLastError();
  }
}

void native_leaf_set_compound_l2pf_ranges(
    int64_t fc_ptr, int64_t fc_span,
    int64_t fkv_ptr, int64_t fkv_span,
    int64_t l0_qkv_ptr, int64_t l0_qkv_span,
    int64_t l0_gate_ptr, int64_t l0_gate_span) {
  g_l2pf_cfg.fc_ptr = static_cast<uint64_t>(fc_ptr);
  g_l2pf_cfg.fc_span = static_cast<uint64_t>(fc_span > 0 ? fc_span : 0);
  g_l2pf_cfg.fkv_ptr = static_cast<uint64_t>(fkv_ptr);
  g_l2pf_cfg.fkv_span = static_cast<uint64_t>(fkv_span > 0 ? fkv_span : 0);
  g_l2pf_cfg.l0_qkv_ptr = static_cast<uint64_t>(l0_qkv_ptr);
  g_l2pf_cfg.l0_qkv_span = static_cast<uint64_t>(l0_qkv_span > 0 ? l0_qkv_span : 0);
  g_l2pf_cfg.l0_gate_ptr = static_cast<uint64_t>(l0_gate_ptr);
  g_l2pf_cfg.l0_gate_span = static_cast<uint64_t>(l0_gate_span > 0 ? l0_gate_span : 0);
}

void dflash_cascade_launch_cond_graph(int64_t exec_id) {
  cudaGraphExec_t exec = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_cond_mu);
    auto it = g_cond_execs.find(exec_id);
    TORCH_CHECK(it != g_cond_execs.end(), "Unknown conditional graph exec_id: ", exec_id);
    exec = it->second.outer_exec;
  }
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  cudaError_t err = cudaGraphLaunch(exec, st);
  TORCH_CHECK(err == cudaSuccess, "cudaGraphLaunch(cond) failed: ", cudaGetErrorString(err));
}
