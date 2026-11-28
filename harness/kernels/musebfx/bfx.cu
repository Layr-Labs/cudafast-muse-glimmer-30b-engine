// musebfx: lossless 12-bit coded BF16 weights for small-M (<=8) GEMM on GB10.
// Weight w[n,k] (bf16) is stored as lo[n,k] = sign<<7 | mantissa7 and a 4-bit code c[n,k]:
//   c in 1..15 -> exponent = base[n] + c ; c == 0 -> contributes 0 (zeros and out-of-window
//   values; every nonzero out-of-window value is in the per-row escape list (CSR), added exactly).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <tuple>
#include <vector>

constexpr int CH = 512;   // k chunk
constexpr int RPW = 2;    // rows per warp
constexpr int WARPS = 8;

template <int M>
__global__ void __launch_bounds__(256) bfx_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ lo,
    const uint8_t* __restrict__ nib, const int* __restrict__ base, const int* __restrict__ rowptr,
    const int* __restrict__ ecol, const __nv_bfloat16* __restrict__ eval, __nv_bfloat16* __restrict__ out, int N, int K) {
  __shared__ float xs[M][CH];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int n0 = (blockIdx.x * WARPS + warp) * RPW;
  float acc[RPW][M];
#pragma unroll
  for (int r = 0; r < RPW; r++)
#pragma unroll
    for (int m = 0; m < M; m++) acc[r][m] = 0.f;
  int bs[RPW];
#pragma unroll
  for (int r = 0; r < RPW; r++) bs[r] = (n0 + r < N) ? base[n0 + r] : 0;
  for (int kc = 0; kc < K; kc += CH) {
    // issue weight loads first (latency overlap with staging)
    uint4 L[RPW]; uint2 C[RPW];
#pragma unroll
    for (int r = 0; r < RPW; r++) {
      const int n = n0 + r;
      if (n < N) {
        L[r] = __ldg(reinterpret_cast<const uint4*>(lo + (size_t)n * K + kc + lane * 16));
        C[r] = __ldg(reinterpret_cast<const uint2*>(nib + (size_t)n * (K / 2) + kc / 2 + lane * 8));
      } else { L[r] = make_uint4(0, 0, 0, 0); C[r] = make_uint2(0, 0); }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < M * CH; i += 256) {
      const int m = i / CH, k = i % CH;
      xs[m][(k & 15) * 32 + (k >> 4)] = __bfloat162float(x[(size_t)m * K + kc + k]);
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < RPW; r++) {
      const uint32_t lw[4] = {L[r].x, L[r].y, L[r].z, L[r].w};
      const uint64_t cw = (uint64_t)C[r].x | ((uint64_t)C[r].y << 32);
      const uint32_t b = (uint32_t)bs[r];
#pragma unroll
      for (int i = 0; i < 16; i++) {
        const uint32_t byte = (lw[i >> 2] >> ((i & 3) * 8)) & 0xFF;
        const uint32_t c = (uint32_t)(cw >> (4 * i)) & 0xF;
        const uint32_t bits = c ? (((byte & 0x80u) << 8) | ((b + c) << 7) | (byte & 0x7Fu)) : 0u;
        const float w = __uint_as_float(bits << 16);
#pragma unroll
        for (int m = 0; m < M; m++) acc[r][m] = fmaf(w, xs[m][i * 32 + lane], acc[r][m]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RPW; r++) {
    const int n = n0 + r;
    if (n >= N) continue;
    for (int e = rowptr[n] + lane; e < rowptr[n + 1]; e += 32) {
      const int k = ecol[e]; const float v = __bfloat162float(eval[e]);
#pragma unroll
      for (int m = 0; m < M; m++) acc[r][m] = fmaf(v, __bfloat162float(x[(size_t)m * K + k]), acc[r][m]);
    }
#pragma unroll
    for (int m = 0; m < M; m++) {
      float a = acc[r][m];
#pragma unroll
      for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0) out[(size_t)m * N + n] = __float2bfloat16_rn(a);
    }
  }
}

torch::Tensor bfx_gemm(torch::Tensor x, torch::Tensor lo, torch::Tensor nib, torch::Tensor base, torch::Tensor rowptr,
                       torch::Tensor ecol, torch::Tensor eval) {
  const int M = x.size(0), K = x.size(1), N = lo.size(0);
  TORCH_CHECK(M >= 1 && M <= 8 && K % CH == 0 && lo.size(1) == K && x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
  auto out = torch::empty({M, N}, x.options());
  dim3 grid((N + WARPS * RPW - 1) / (WARPS * RPW));
  auto st = c10::cuda::getCurrentCUDAStream().stream();
#define L_(MM) case MM: bfx_kernel<MM><<<grid, 256, 0, st>>>((const __nv_bfloat16*)x.data_ptr(), lo.data_ptr<uint8_t>(), \
    nib.data_ptr<uint8_t>(), base.data_ptr<int>(), rowptr.data_ptr<int>(), ecol.data_ptr<int>(), \
    (const __nv_bfloat16*)eval.data_ptr(), (__nv_bfloat16*)out.data_ptr(), N, K); break;
  switch (M) { L_(1) L_(2) L_(3) L_(4) L_(5) L_(6) L_(7) L_(8) }
#undef L_
  return out;
}

#include <tuple>
torch::Tensor bf16_rans_pack(torch::Tensor W);
torch::Tensor bf16_rans_unpack(torch::Tensor blob, int64_t N, int64_t K);
torch::Tensor bf16_rans_gemm(torch::Tensor x, torch::Tensor blob, int64_t N, int64_t K, int64_t base_exp);
void bf16_rans_gemm_out(torch::Tensor x, torch::Tensor blob, torch::Tensor out, int64_t N, int64_t K, int64_t base_exp);
std::tuple<torch::Tensor, torch::Tensor> bf16_rans_gemm_dual(
    torch::Tensor x, torch::Tensor blob, int64_t N1, int64_t N2, int64_t K, int64_t base_exp);
void bf16_rans_gemm_dual_out(
    torch::Tensor x, torch::Tensor blob, torch::Tensor out1, torch::Tensor out2,
    int64_t N1, int64_t N2, int64_t K, int64_t base_exp);
int64_t bf16_rans_base(torch::Tensor blob);
torch::Tensor int3_pack(torch::Tensor W, bool mse_opt = true, bool sym = false, bool h64 = false);
torch::Tensor int3_gemm(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K);
torch::Tensor int3_pack_swiglu(torch::Tensor W, bool mse_opt = true, bool sym = false, bool h64 = false);
torch::Tensor int3_gemm_swiglu(torch::Tensor A, torch::Tensor W_packed, int64_t I, int64_t K);
torch::Tensor vq4d_pack(torch::Tensor W, int64_t bits = 10);
torch::Tensor vq4d_gemm(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K);
torch::Tensor vq4d_pack_swiglu(torch::Tensor W, int64_t bits = 10);
torch::Tensor vq4d_gemm_swiglu(torch::Tensor A, torch::Tensor W_packed, int64_t I, int64_t K);
torch::Tensor int4_pack(torch::Tensor W, bool mse_opt = true);
torch::Tensor int4_gemm(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K, bool query_streamk = false);
torch::Tensor int4_pack_swiglu(torch::Tensor W, bool mse_opt = true);
torch::Tensor int4_gemm_swiglu(torch::Tensor A, torch::Tensor W_packed, int64_t I, int64_t K);
torch::Tensor int4_gemm_aux5(
    torch::Tensor h0, torch::Tensor h1, torch::Tensor h2, torch::Tensor h3, torch::Tensor h4,
    torch::Tensor W_packed, int64_t N, int64_t K);
torch::Tensor dflash_rms_norm(torch::Tensor input, torch::Tensor weight, double eps);
std::tuple<torch::Tensor, torch::Tensor> dflash_fused_add_rms_norm(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps);
std::tuple<torch::Tensor, torch::Tensor> dflash_embed_rms_norm(
    torch::Tensor input_ids, torch::Tensor embed_weight, torch::Tensor norm_weight, double eps);
void dflash_kv_post(
    torch::Tensor all_kv_flat, torch::Tensor positions, torch::Tensor k_norm_w,
    torch::Tensor cos_sin_cache,
    const std::vector<torch::Tensor>& k_caches,
    const std::vector<torch::Tensor>& v_caches,
    const std::vector<torch::Tensor>& slot_mappings,
    int64_t nkv, int64_t block_size, double eps, bool is_neox);
std::tuple<torch::Tensor, torch::Tensor> dflash_qkv_norm_rope(
    torch::Tensor qkv, torch::Tensor positions, torch::Tensor q_norm_w,
    torch::Tensor k_norm_w, torch::Tensor cos_sin_cache,
    int64_t nq, int64_t nkv, double eps, bool is_neox,
    c10::optional<torch::Tensor> slot_mapping = c10::nullopt,
    c10::optional<torch::Tensor> k_cache = c10::nullopt,
    c10::optional<torch::Tensor> v_cache = c10::nullopt,
    int64_t block_size = 16);
void dflash_draft_attn(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor out,
    torch::Tensor seqused_k, torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    double softmax_scale, int64_t window_left = -1, int64_t window_right = -1,
    int64_t max_q = 16, c10::optional<torch::Tensor> sinks = c10::nullopt);
void dflash_prepare_inputs(
    torch::Tensor out_input_ids, torch::Tensor out_query_positions,
    torch::Tensor out_query_start_loc, torch::Tensor out_seq_lens,
    torch::Tensor out_query_slot_mapping, torch::Tensor out_context_positions,
    torch::Tensor out_context_slot_mapping, torch::Tensor out_sample_indices,
    torch::Tensor out_sample_pos, torch::Tensor out_sample_idx_mapping,
    torch::Tensor out_temperature, torch::Tensor out_seeds,
    torch::Tensor target_positions, torch::Tensor target_query_start_loc,
    torch::Tensor idx_mapping, torch::Tensor last_sampled,
    torch::Tensor next_prefill_tokens, torch::Tensor num_sampled,
    torch::Tensor num_rejected, torch::Tensor in_temperature,
    torch::Tensor in_seeds, torch::Tensor block_table,
    int64_t block_table_stride, int64_t parallel_drafting_token_id,
    int64_t block_size, int64_t num_query_per_req,
    int64_t num_speculative_steps, int64_t max_num_reqs,
    int64_t max_num_tokens, int64_t max_model_len, int64_t pad_slot_id,
    c10::optional<torch::Tensor> target_input_ids = c10::nullopt,
    int64_t tail_num_tokens = 0,
    c10::optional<torch::Tensor> in_static_vocab = c10::nullopt,
    c10::optional<torch::Tensor> tail_ids = c10::nullopt,
    c10::optional<torch::Tensor> tail_count = c10::nullopt,
    c10::optional<torch::Tensor> tail_seen = c10::nullopt,
    c10::optional<torch::Tensor> hist = c10::nullopt,
    c10::optional<torch::Tensor> hist_len = c10::nullopt,
    c10::optional<torch::Tensor> post_sampled_tokens = c10::nullopt,
    c10::optional<torch::Tensor> post_num_computed_tokens = c10::nullopt,
    c10::optional<torch::Tensor> post_all_token_ids = c10::nullopt,
    c10::optional<torch::Tensor> post_total_len = c10::nullopt,
    c10::optional<torch::Tensor> post_output_bin_counts = c10::nullopt);
void draft_vocab_update_tail(
    torch::Tensor input_ids, torch::Tensor positions, torch::Tensor in_static_vocab,
    torch::Tensor tail_ids, torch::Tensor tail_count, torch::Tensor tail_seen,
    c10::optional<torch::Tensor> num_rejected = c10::nullopt,
    c10::optional<torch::Tensor> extra_ids = c10::nullopt);
torch::Tensor draft_vocab_tail_argmax(
    torch::Tensor A, torch::Tensor static_logits, torch::Tensor vocab_map,
    torch::Tensor tail_ids, torch::Tensor tail_count, torch::Tensor raw_w,
    torch::Tensor raw_s, torch::Tensor global_scale, double logit_scale,
    c10::optional<torch::Tensor> out_top2_row0 = c10::nullopt,
    c10::optional<torch::Tensor> out_margins = c10::nullopt,
    c10::optional<torch::Tensor> out_tree_top = c10::nullopt,
    c10::optional<torch::Tensor> out_tree_vals = c10::nullopt,
    bool vocab_map_sorted = false);
torch::Tensor draft_vocab_score_rows(
    torch::Tensor A, torch::Tensor row_ids, torch::Tensor raw_w,
    torch::Tensor raw_s, torch::Tensor global_scale, double logit_scale);
torch::Tensor draft_head_pack(
    torch::Tensor raw_w, torch::Tensor raw_s, torch::Tensor global_scale,
    torch::Tensor vocab_map, int64_t grid_ctas = 96, int64_t mode = 0);
torch::Tensor draft_head_topc(
    torch::Tensor A, torch::Tensor W_packed, torch::Tensor vocab_map,
    int64_t N, int64_t K, int64_t grid_ctas = 96, int64_t mode = 0, int64_t rescore_k = 0);
torch::Tensor draft_head_gemm_full(
    torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K,
    int64_t grid_ctas = 96, int64_t mode = 0);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> dhead_fast_pack(
    torch::Tensor raw_w, torch::Tensor raw_s, torch::Tensor global_scale,
    torch::Tensor vocab_map, int64_t mode = 150);
torch::Tensor dhead_fast_argmax(
    torch::Tensor A, torch::Tensor blob, torch::Tensor row_scales,
    torch::Tensor col_scales, torch::Tensor packed_tok_ids,
    torch::Tensor tail_ids, torch::Tensor tail_count, torch::Tensor raw_w,
    torch::Tensor raw_s, torch::Tensor global_scale, double logit_scale = 1.0,
    int64_t mode = 150, int64_t topk = 192,
    c10::optional<torch::Tensor> out_top2_row0 = c10::nullopt,
    c10::optional<torch::Tensor> out_margins = c10::nullopt,
    c10::optional<torch::Tensor> out_tree_top = c10::nullopt,
    c10::optional<torch::Tensor> out_tree_vals = c10::nullopt);
void native_leaf_prepare_m8(
    torch::Tensor positions, torch::Tensor seq_lens);
void native_leaf_prepare_tree(
    torch::Tensor positions, torch::Tensor seq_lens, int64_t tree_m);
void native_leaf_setup_tree(
    torch::Tensor leaf_flag, torch::Tensor tree_top,
    torch::Tensor input_ids, torch::Tensor positions,
    torch::Tensor saved_leaf_toks, torch::Tensor saved_leaf_k,
    int64_t tree_m = 12, int64_t selftest_mode = 0,
    c10::optional<torch::Tensor> req_acc_state = c10::nullopt,
    c10::optional<torch::Tensor> tree_vals = c10::nullopt,
    bool dynamic_mode = true,
    double temp = 1.35,
    double tail_c = 4.0,
    double gamma = 0.92,
    c10::optional<torch::Tensor> saved_leaf_parent = c10::nullopt,
    int64_t max_depth = 6,
    double sub_coef = 0.55,
    double shf_coef = 0.20,
    double cont_decay = 0.88,
    c10::optional<torch::Tensor> hist = c10::nullopt,
    c10::optional<torch::Tensor> hist_len = c10::nullopt,
    c10::optional<torch::Tensor> saved_pos0 = c10::nullopt,
    c10::optional<torch::Tensor> stats = c10::nullopt,
    c10::optional<torch::Tensor> saved_row_ctx = c10::nullopt,
    c10::optional<torch::Tensor> tr_log = c10::nullopt,
    c10::optional<torch::Tensor> tr_state = c10::nullopt,
    int64_t vocab_size = 202048,
    c10::optional<torch::Tensor> cascade_meta = c10::nullopt,
    c10::optional<torch::Tensor> cascade_toks = c10::nullopt,
    c10::optional<torch::Tensor> cascade_vals = c10::nullopt);
void native_leaf_commit_tree(
    torch::Tensor leaf_flag, torch::Tensor prev_ns,
    torch::Tensor hit_flag, torch::Tensor hit_src_row,
    torch::Tensor saved_leaf_toks, torch::Tensor saved_leaf_k,
    torch::Tensor leaf_u_toks, torch::Tensor sampled_tokens,
    torch::Tensor num_sampled, torch::Tensor num_rejected,
    int64_t tree_m = 12,
    c10::optional<torch::Tensor> stats = c10::nullopt, bool no_hit_mode = false,
    c10::optional<torch::Tensor> req_acc_state = c10::nullopt,
    c10::optional<torch::Tensor> saved_leaf_parent = c10::nullopt,
    c10::optional<torch::Tensor> hist = c10::nullopt,
    c10::optional<torch::Tensor> hist_len = c10::nullopt,
    c10::optional<torch::Tensor> saved_pos0 = c10::nullopt,
    c10::optional<torch::Tensor> saved_row_ctx = c10::nullopt,
    c10::optional<torch::Tensor> tr_log = c10::nullopt,
    c10::optional<torch::Tensor> tr_state = c10::nullopt,
    int64_t vocab_size = 202048,
    bool chain_all_valid = false,
    c10::optional<torch::Tensor> cascade_meta = c10::nullopt,
    c10::optional<torch::Tensor> in_static_vocab = c10::nullopt,
    c10::optional<torch::Tensor> tail_ids = c10::nullopt,
    c10::optional<torch::Tensor> tail_count = c10::nullopt,
    c10::optional<torch::Tensor> tail_seen = c10::nullopt,
    c10::optional<torch::Tensor> hit_cond = c10::nullopt);
void cuda_graph_upload(int64_t raw_exec_ptr);
void native_leaf_set_compound_l2pf_ranges(
    int64_t fc_ptr, int64_t fc_span,
    int64_t fkv_ptr, int64_t fkv_span,
    int64_t l0_qkv_ptr, int64_t l0_qkv_span,
    int64_t l0_gate_ptr, int64_t l0_gate_span);
int64_t native_leaf_build_adaptive_verify_graph(
    std::vector<int64_t> m16_graphs,
    std::vector<int64_t> m32_graphs,
    torch::Tensor tree_vals,
    torch::Tensor req_acc_state,
    int64_t tree_k = 8,
    int64_t min_rounds = 3,
    double wide_cpc = 3.25,
    double mid_cpc = 3.55,
    int64_t force_mode = 0);
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
    int64_t tree_k = 8,
    int64_t min_rounds = 3,
    double wide_cpc = 3.25,
    double mid_cpc = 3.55,
    int64_t force_mode = 0);
void dflash_cascade_gate_prep(
    torch::Tensor input_ids, torch::Tensor query_positions,
    torch::Tensor draft_tokens, torch::Tensor tree_top, torch::Tensor tree_vals,
    torch::Tensor cascade_meta, torch::Tensor p1_save_draft,
    torch::Tensor p1_save_top, torch::Tensor p1_save_vals,
    int64_t mask_token_id = 201818,
    c10::optional<torch::Tensor> alt_ids = c10::nullopt,
    c10::optional<torch::Tensor> alt_margins = c10::nullopt,
    c10::optional<torch::Tensor> p1_save_alt = c10::nullopt,
    c10::optional<torch::Tensor> p1_save_margins = c10::nullopt,
    c10::optional<torch::Tensor> req_acc_state = c10::nullopt,
    c10::optional<torch::Tensor> stats = c10::nullopt);
void dflash_cascade_post_restore(
    torch::Tensor input_ids, torch::Tensor draft_tokens,
    torch::Tensor tree_top, torch::Tensor tree_vals,
    torch::Tensor cascade_meta, torch::Tensor cascade_toks, torch::Tensor cascade_vals,
    torch::Tensor p1_save_draft, torch::Tensor p1_save_top, torch::Tensor p1_save_vals,
    int64_t mask_token_id = 201818,
    c10::optional<torch::Tensor> alt_ids = c10::nullopt,
    c10::optional<torch::Tensor> alt_margins = c10::nullopt,
    c10::optional<torch::Tensor> p1_save_alt = c10::nullopt,
    c10::optional<torch::Tensor> p1_save_margins = c10::nullopt);
int64_t dflash_cascade_clone_capturing_graph();
int64_t dflash_cascade_build_cond_graph(
    int64_t graph_handle, bool is_raw_cuda_graph_ptr,
    torch::Tensor input_ids, torch::Tensor query_positions,
    torch::Tensor draft_tokens, torch::Tensor tree_top, torch::Tensor tree_vals,
    torch::Tensor cascade_meta, torch::Tensor p1_save_draft,
    torch::Tensor p1_save_top, torch::Tensor p1_save_vals,
    int64_t mask_token_id = 201818,
    c10::optional<torch::Tensor> alt_ids = c10::nullopt,
    c10::optional<torch::Tensor> alt_margins = c10::nullopt,
    c10::optional<torch::Tensor> p1_save_alt = c10::nullopt,
    c10::optional<torch::Tensor> p1_save_margins = c10::nullopt,
    c10::optional<torch::Tensor> req_acc_state = c10::nullopt,
    c10::optional<torch::Tensor> stats = c10::nullopt);
void dflash_cascade_launch_cond_graph(int64_t exec_id);
void native_leaf_setup(
    torch::Tensor leaf_flag, torch::Tensor prev_ns, torch::Tensor alt_ids,
    torch::Tensor input_ids, torch::Tensor positions, torch::Tensor saved_alt_tok,
    int64_t low_thresh = 3, int64_t selftest_mode = 0,
    c10::optional<torch::Tensor> alt_margins = c10::nullopt,
    c10::optional<torch::Tensor> req_acc_state = c10::nullopt,
    bool m8_mode = false, int64_t policy = 0, double margin_alpha = 1.0);
void native_leaf_commit(
    torch::Tensor leaf_flag, torch::Tensor prev_ns, torch::Tensor hit_flag,
    torch::Tensor saved_alt_tok, torch::Tensor leaf_u_tok, torch::Tensor sampled_tokens,
    torch::Tensor num_sampled, torch::Tensor num_rejected,
    c10::optional<torch::Tensor> stats = c10::nullopt, bool no_hit_mode = false,
    c10::optional<torch::Tensor> req_acc_state = c10::nullopt, bool m8_mode = false);
void native_leaf_fix_hidden(
    torch::Tensor hit_flag, torch::Tensor tensor_ptrs,
    int64_t row_stride_bytes, int64_t copy_bytes, int64_t src_row = 6,
    c10::optional<torch::Tensor> src_row_buf = c10::nullopt);
void native_leaf_fix_kv(
    torch::Tensor hit_flag, torch::Tensor slot_mappings,
    torch::Tensor kv_ptrs, torch::Tensor stride_block, torch::Tensor stride_head,
    torch::Tensor stride_slot, torch::Tensor block_size, torch::Tensor layer_gid,
    int64_t src_row = 6,
    c10::optional<torch::Tensor> src_row_buf = c10::nullopt);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &bfx_gemm);
  m.def("rans_pack", &bf16_rans_pack);
  m.def("rans_unpack", &bf16_rans_unpack);
  m.def("rans_gemm", &bf16_rans_gemm);
  m.def("rans_gemm_out", &bf16_rans_gemm_out);
  m.def("rans_gemm_dual", &bf16_rans_gemm_dual);
  m.def("rans_gemm_dual_out", &bf16_rans_gemm_dual_out);
  m.def("rans_base", &bf16_rans_base);
  m.def("int3_pack", &int3_pack, pybind11::arg("W"), pybind11::arg("mse_opt") = true, pybind11::arg("sym") = false, pybind11::arg("h64") = false);
  m.def("int3_gemm", &int3_gemm);
  m.def("int3_pack_swiglu", &int3_pack_swiglu, pybind11::arg("W"), pybind11::arg("mse_opt") = true, pybind11::arg("sym") = false, pybind11::arg("h64") = false);
  m.def("int3_gemm_swiglu", &int3_gemm_swiglu);
  m.def("vq4d_pack", &vq4d_pack, pybind11::arg("W"), pybind11::arg("bits") = 10);
  m.def("vq4d_gemm", &vq4d_gemm);
  m.def("vq4d_pack_swiglu", &vq4d_pack_swiglu, pybind11::arg("W"), pybind11::arg("bits") = 10);
  m.def("vq4d_gemm_swiglu", &vq4d_gemm_swiglu);
  m.def("int4_pack", &int4_pack, pybind11::arg("W"), pybind11::arg("mse_opt") = true);
  m.def("int4_gemm", &int4_gemm, pybind11::arg("A"), pybind11::arg("W_packed"),
        pybind11::arg("N"), pybind11::arg("K"), pybind11::arg("query_streamk") = false);
  m.def("int4_pack_swiglu", &int4_pack_swiglu, pybind11::arg("W"), pybind11::arg("mse_opt") = true);
  m.def("int4_gemm_swiglu", &int4_gemm_swiglu);
  m.def("int4_gemm_aux5", &int4_gemm_aux5);
  m.def("dflash_rms_norm", &dflash_rms_norm);
  m.def("dflash_fused_add_rms_norm", &dflash_fused_add_rms_norm);
  m.def("dflash_embed_rms_norm", &dflash_embed_rms_norm);
  m.def("dflash_kv_post", &dflash_kv_post);
  m.def("dflash_qkv_norm_rope", &dflash_qkv_norm_rope,
        pybind11::arg("qkv"), pybind11::arg("positions"),
        pybind11::arg("q_norm_w"), pybind11::arg("k_norm_w"),
        pybind11::arg("cos_sin_cache"), pybind11::arg("nq"),
        pybind11::arg("nkv"), pybind11::arg("eps"), pybind11::arg("is_neox"),
        pybind11::arg("slot_mapping") = c10::nullopt,
        pybind11::arg("k_cache") = c10::nullopt,
        pybind11::arg("v_cache") = c10::nullopt,
        pybind11::arg("block_size") = 16);
  m.def("dflash_prepare_inputs", &dflash_prepare_inputs,
        pybind11::arg("out_input_ids"), pybind11::arg("out_query_positions"),
        pybind11::arg("out_query_start_loc"), pybind11::arg("out_seq_lens"),
        pybind11::arg("out_query_slot_mapping"), pybind11::arg("out_context_positions"),
        pybind11::arg("out_context_slot_mapping"), pybind11::arg("out_sample_indices"),
        pybind11::arg("out_sample_pos"), pybind11::arg("out_sample_idx_mapping"),
        pybind11::arg("out_temperature"), pybind11::arg("out_seeds"),
        pybind11::arg("target_positions"), pybind11::arg("target_query_start_loc"),
        pybind11::arg("idx_mapping"), pybind11::arg("last_sampled"),
        pybind11::arg("next_prefill_tokens"), pybind11::arg("num_sampled"),
        pybind11::arg("num_rejected"), pybind11::arg("in_temperature"),
        pybind11::arg("in_seeds"), pybind11::arg("block_table"),
        pybind11::arg("block_table_stride"), pybind11::arg("parallel_drafting_token_id"),
        pybind11::arg("block_size"), pybind11::arg("num_query_per_req"),
        pybind11::arg("num_speculative_steps"), pybind11::arg("max_num_reqs"),
        pybind11::arg("max_num_tokens"), pybind11::arg("max_model_len"),
        pybind11::arg("pad_slot_id"),
        pybind11::arg("target_input_ids") = c10::nullopt,
        pybind11::arg("tail_num_tokens") = 0,
        pybind11::arg("in_static_vocab") = c10::nullopt,
        pybind11::arg("tail_ids") = c10::nullopt,
        pybind11::arg("tail_count") = c10::nullopt,
        pybind11::arg("tail_seen") = c10::nullopt,
        pybind11::arg("hist") = c10::nullopt,
        pybind11::arg("hist_len") = c10::nullopt,
        pybind11::arg("post_sampled_tokens") = c10::nullopt,
        pybind11::arg("post_num_computed_tokens") = c10::nullopt,
        pybind11::arg("post_all_token_ids") = c10::nullopt,
        pybind11::arg("post_total_len") = c10::nullopt,
        pybind11::arg("post_output_bin_counts") = c10::nullopt);
  m.def("draft_vocab_update_tail", &draft_vocab_update_tail,
        pybind11::arg("input_ids"), pybind11::arg("positions"), pybind11::arg("in_static_vocab"),
        pybind11::arg("tail_ids"), pybind11::arg("tail_count"), pybind11::arg("tail_seen"),
        pybind11::arg("num_rejected") = c10::nullopt,
        pybind11::arg("extra_ids") = c10::nullopt);
  m.def("draft_vocab_tail_argmax", &draft_vocab_tail_argmax,
        pybind11::arg("A"), pybind11::arg("static_logits"), pybind11::arg("vocab_map"),
        pybind11::arg("tail_ids"), pybind11::arg("tail_count"), pybind11::arg("raw_w"),
        pybind11::arg("raw_s"), pybind11::arg("global_scale"), pybind11::arg("logit_scale") = 1.0,
        pybind11::arg("out_top2_row0") = c10::nullopt,
        pybind11::arg("out_margins") = c10::nullopt,
        pybind11::arg("out_tree_top") = c10::nullopt,
        pybind11::arg("out_tree_vals") = c10::nullopt,
        pybind11::arg("vocab_map_sorted") = false);
  m.def("draft_vocab_score_rows", &draft_vocab_score_rows,
        pybind11::arg("A"), pybind11::arg("row_ids"), pybind11::arg("raw_w"),
        pybind11::arg("raw_s"), pybind11::arg("global_scale"), pybind11::arg("logit_scale") = 1.0);
  m.def("draft_head_pack", &draft_head_pack,
        pybind11::arg("raw_w"), pybind11::arg("raw_s"), pybind11::arg("global_scale"),
        pybind11::arg("vocab_map"), pybind11::arg("grid_ctas") = 96, pybind11::arg("mode") = 0);
  m.def("draft_head_topc", &draft_head_topc,
        pybind11::arg("A"), pybind11::arg("W_packed"), pybind11::arg("vocab_map"),
        pybind11::arg("N"), pybind11::arg("K"),
        pybind11::arg("grid_ctas") = 96, pybind11::arg("mode") = 0, pybind11::arg("rescore_k") = 0);
  m.def("draft_head_gemm_full", &draft_head_gemm_full,
        pybind11::arg("A"), pybind11::arg("W_packed"), pybind11::arg("N"), pybind11::arg("K"),
        pybind11::arg("grid_ctas") = 96, pybind11::arg("mode") = 0);
  m.def("dhead_fast_pack", &dhead_fast_pack,
        pybind11::arg("raw_w"), pybind11::arg("raw_s"), pybind11::arg("global_scale"),
        pybind11::arg("vocab_map"), pybind11::arg("mode") = 150);
  m.def("dhead_fast_argmax", &dhead_fast_argmax,
        pybind11::arg("A"), pybind11::arg("blob"), pybind11::arg("row_scales"),
        pybind11::arg("col_scales"), pybind11::arg("packed_tok_ids"),
        pybind11::arg("tail_ids"), pybind11::arg("tail_count"), pybind11::arg("raw_w"),
        pybind11::arg("raw_s"), pybind11::arg("global_scale"), pybind11::arg("logit_scale") = 1.0,
        pybind11::arg("mode") = 150, pybind11::arg("topk") = 192,
        pybind11::arg("out_top2_row0") = c10::nullopt,
        pybind11::arg("out_margins") = c10::nullopt,
        pybind11::arg("out_tree_top") = c10::nullopt,
        pybind11::arg("out_tree_vals") = c10::nullopt);
  m.def("native_leaf_prepare_m8", &native_leaf_prepare_m8);
  m.def("native_leaf_prepare_tree", &native_leaf_prepare_tree,
        pybind11::arg("positions"), pybind11::arg("seq_lens"), pybind11::arg("tree_m") = 12);
  m.def("native_leaf_setup_tree", &native_leaf_setup_tree,
        pybind11::arg("leaf_flag"), pybind11::arg("tree_top"),
        pybind11::arg("input_ids"), pybind11::arg("positions"),
        pybind11::arg("saved_leaf_toks"), pybind11::arg("saved_leaf_k"),
        pybind11::arg("tree_m") = 12, pybind11::arg("selftest_mode") = 0,
        pybind11::arg("req_acc_state") = c10::nullopt,
        pybind11::arg("tree_vals") = c10::nullopt,
        pybind11::arg("dynamic_mode") = true,
        pybind11::arg("temp") = 1.35,
        pybind11::arg("tail_c") = 4.0,
        pybind11::arg("gamma") = 0.92,
        pybind11::arg("saved_leaf_parent") = c10::nullopt,
        pybind11::arg("max_depth") = 6,
        pybind11::arg("sub_coef") = 0.55,
        pybind11::arg("shf_coef") = 0.20,
        pybind11::arg("cont_decay") = 0.88,
        pybind11::arg("hist") = c10::nullopt,
        pybind11::arg("hist_len") = c10::nullopt,
        pybind11::arg("saved_pos0") = c10::nullopt,
        pybind11::arg("stats") = c10::nullopt,
        pybind11::arg("saved_row_ctx") = c10::nullopt,
        pybind11::arg("tr_log") = c10::nullopt,
        pybind11::arg("tr_state") = c10::nullopt,
        pybind11::arg("vocab_size") = 202048,
        pybind11::arg("cascade_meta") = c10::nullopt,
        pybind11::arg("cascade_toks") = c10::nullopt,
        pybind11::arg("cascade_vals") = c10::nullopt);
  m.def("native_leaf_commit_tree", &native_leaf_commit_tree,
        pybind11::arg("leaf_flag"), pybind11::arg("prev_ns"),
        pybind11::arg("hit_flag"), pybind11::arg("hit_src_row"),
        pybind11::arg("saved_leaf_toks"), pybind11::arg("saved_leaf_k"),
        pybind11::arg("leaf_u_toks"), pybind11::arg("sampled_tokens"),
        pybind11::arg("num_sampled"), pybind11::arg("num_rejected"),
        pybind11::arg("tree_m") = 12,
        pybind11::arg("stats") = c10::nullopt, pybind11::arg("no_hit_mode") = false,
        pybind11::arg("req_acc_state") = c10::nullopt,
        pybind11::arg("saved_leaf_parent") = c10::nullopt,
        pybind11::arg("hist") = c10::nullopt,
        pybind11::arg("hist_len") = c10::nullopt,
        pybind11::arg("saved_pos0") = c10::nullopt,
        pybind11::arg("saved_row_ctx") = c10::nullopt,
        pybind11::arg("tr_log") = c10::nullopt,
        pybind11::arg("tr_state") = c10::nullopt,
        pybind11::arg("vocab_size") = 202048,
        pybind11::arg("chain_all_valid") = false,
        pybind11::arg("cascade_meta") = c10::nullopt,
        pybind11::arg("in_static_vocab") = c10::nullopt,
        pybind11::arg("tail_ids") = c10::nullopt,
        pybind11::arg("tail_count") = c10::nullopt,
        pybind11::arg("tail_seen") = c10::nullopt,
        pybind11::arg("hit_cond") = c10::nullopt);
  m.def("cuda_graph_upload", &cuda_graph_upload, pybind11::arg("raw_exec_ptr"));
  m.def("native_leaf_set_compound_l2pf_ranges", &native_leaf_set_compound_l2pf_ranges,
        pybind11::arg("fc_ptr") = 0, pybind11::arg("fc_span") = 0,
        pybind11::arg("fkv_ptr") = 0, pybind11::arg("fkv_span") = 0,
        pybind11::arg("l0_qkv_ptr") = 0, pybind11::arg("l0_qkv_span") = 0,
        pybind11::arg("l0_gate_ptr") = 0, pybind11::arg("l0_gate_span") = 0);
  m.def("native_leaf_build_adaptive_verify_graph", &native_leaf_build_adaptive_verify_graph,
        pybind11::arg("m16_graphs"), pybind11::arg("m32_graphs"),
        pybind11::arg("tree_vals"), pybind11::arg("req_acc_state"),
        pybind11::arg("tree_k") = 8, pybind11::arg("min_rounds") = 3,
        pybind11::arg("wide_cpc") = 3.25, pybind11::arg("mid_cpc") = 3.55,
        pybind11::arg("force_mode") = 0);
  m.def("native_leaf_build_compound_step_graph", &native_leaf_build_compound_step_graph,
        pybind11::arg("m16_graphs"), pybind11::arg("m32_graphs"), pybind11::arg("tail_graphs"),
        pybind11::arg("sampled_tokens"), pybind11::arg("num_sampled"),
        pybind11::arg("host_out_row"), pybind11::arg("raw_cuda_event"),
        pybind11::arg("tree_vals"), pybind11::arg("req_acc_state"),
        pybind11::arg("tree_k") = 8, pybind11::arg("min_rounds") = 3,
        pybind11::arg("wide_cpc") = 3.25, pybind11::arg("mid_cpc") = 3.55,
        pybind11::arg("force_mode") = 0);
  m.def("dflash_cascade_gate_prep", &dflash_cascade_gate_prep,
        pybind11::arg("input_ids"), pybind11::arg("query_positions"),
        pybind11::arg("draft_tokens"), pybind11::arg("tree_top"), pybind11::arg("tree_vals"),
        pybind11::arg("cascade_meta"), pybind11::arg("p1_save_draft"),
        pybind11::arg("p1_save_top"), pybind11::arg("p1_save_vals"),
        pybind11::arg("mask_token_id") = 201818,
        pybind11::arg("alt_ids") = c10::nullopt,
        pybind11::arg("alt_margins") = c10::nullopt,
        pybind11::arg("p1_save_alt") = c10::nullopt,
        pybind11::arg("p1_save_margins") = c10::nullopt,
        pybind11::arg("req_acc_state") = c10::nullopt,
        pybind11::arg("stats") = c10::nullopt);
  m.def("dflash_cascade_post_restore", &dflash_cascade_post_restore,
        pybind11::arg("input_ids"), pybind11::arg("draft_tokens"),
        pybind11::arg("tree_top"), pybind11::arg("tree_vals"),
        pybind11::arg("cascade_meta"), pybind11::arg("cascade_toks"), pybind11::arg("cascade_vals"),
        pybind11::arg("p1_save_draft"), pybind11::arg("p1_save_top"), pybind11::arg("p1_save_vals"),
        pybind11::arg("mask_token_id") = 201818,
        pybind11::arg("alt_ids") = c10::nullopt,
        pybind11::arg("alt_margins") = c10::nullopt,
        pybind11::arg("p1_save_alt") = c10::nullopt,
        pybind11::arg("p1_save_margins") = c10::nullopt);
  m.def("dflash_cascade_clone_capturing_graph", &dflash_cascade_clone_capturing_graph);
  m.def("dflash_cascade_build_cond_graph", &dflash_cascade_build_cond_graph,
        pybind11::arg("graph_handle"), pybind11::arg("is_raw_cuda_graph_ptr"),
        pybind11::arg("input_ids"), pybind11::arg("query_positions"),
        pybind11::arg("draft_tokens"), pybind11::arg("tree_top"), pybind11::arg("tree_vals"),
        pybind11::arg("cascade_meta"), pybind11::arg("p1_save_draft"),
        pybind11::arg("p1_save_top"), pybind11::arg("p1_save_vals"),
        pybind11::arg("mask_token_id") = 201818,
        pybind11::arg("alt_ids") = c10::nullopt,
        pybind11::arg("alt_margins") = c10::nullopt,
        pybind11::arg("p1_save_alt") = c10::nullopt,
        pybind11::arg("p1_save_margins") = c10::nullopt,
        pybind11::arg("req_acc_state") = c10::nullopt,
        pybind11::arg("stats") = c10::nullopt);
  m.def("dflash_cascade_launch_cond_graph", &dflash_cascade_launch_cond_graph,
        pybind11::arg("exec_id"));
  m.def("native_leaf_setup", &native_leaf_setup,
        pybind11::arg("leaf_flag"), pybind11::arg("prev_ns"), pybind11::arg("alt_ids"),
        pybind11::arg("input_ids"), pybind11::arg("positions"), pybind11::arg("saved_alt_tok"),
        pybind11::arg("low_thresh") = 3, pybind11::arg("selftest_mode") = 0,
        pybind11::arg("alt_margins") = c10::nullopt,
        pybind11::arg("req_acc_state") = c10::nullopt,
        pybind11::arg("m8_mode") = false,
        pybind11::arg("policy") = 0,
        pybind11::arg("margin_alpha") = 1.0);
  m.def("native_leaf_commit", &native_leaf_commit,
        pybind11::arg("leaf_flag"), pybind11::arg("prev_ns"), pybind11::arg("hit_flag"),
        pybind11::arg("saved_alt_tok"), pybind11::arg("leaf_u_tok"), pybind11::arg("sampled_tokens"),
        pybind11::arg("num_sampled"), pybind11::arg("num_rejected"),
        pybind11::arg("stats") = c10::nullopt, pybind11::arg("no_hit_mode") = false,
        pybind11::arg("req_acc_state") = c10::nullopt, pybind11::arg("m8_mode") = false);
  m.def("native_leaf_fix_hidden", &native_leaf_fix_hidden,
        pybind11::arg("hit_flag"), pybind11::arg("tensor_ptrs"),
        pybind11::arg("row_stride_bytes"), pybind11::arg("copy_bytes"),
        pybind11::arg("src_row") = 6,
        pybind11::arg("src_row_buf") = c10::nullopt);
  m.def("native_leaf_fix_kv", &native_leaf_fix_kv,
        pybind11::arg("hit_flag"), pybind11::arg("slot_mappings"),
        pybind11::arg("kv_ptrs"), pybind11::arg("stride_block"), pybind11::arg("stride_head"),
        pybind11::arg("stride_slot"), pybind11::arg("block_size"), pybind11::arg("layer_gid"),
        pybind11::arg("src_row") = 6,
        pybind11::arg("src_row_buf") = c10::nullopt);
  m.def("dflash_draft_attn", &dflash_draft_attn,
        pybind11::arg("q"), pybind11::arg("k_cache"), pybind11::arg("v_cache"),
        pybind11::arg("out"), pybind11::arg("seqused_k"), pybind11::arg("block_table"),
        pybind11::arg("cu_seqlens_q"), pybind11::arg("softmax_scale"),
        pybind11::arg("window_left") = -1, pybind11::arg("window_right") = -1,
        pybind11::arg("max_q") = 16, pybind11::arg("sinks") = c10::nullopt);
}
