// musebfx: lossless 12-bit coded BF16 weights for small-M (<=8) GEMM on GB10.
// Weight w[n,k] (bf16) is stored as lo[n,k] = sign<<7 | mantissa7 and a 4-bit code c[n,k]:
//   c in 1..15 -> exponent = base[n] + c ; c == 0 -> contributes 0 (zeros and out-of-window
//   values; every nonzero out-of-window value is in the per-row escape list (CSR), added exactly).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <stdint.h>

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

torch::Tensor bf16_rans_pack(torch::Tensor W);
torch::Tensor bf16_rans_unpack(torch::Tensor blob, int64_t N, int64_t K);
torch::Tensor bf16_rans_gemm(torch::Tensor x, torch::Tensor blob, int64_t N, int64_t K, int64_t base_exp);
void bf16_rans_gemm_out(torch::Tensor x, torch::Tensor blob, torch::Tensor out, int64_t N, int64_t K, int64_t base_exp);
int64_t bf16_rans_base(torch::Tensor blob);
torch::Tensor int3_pack(torch::Tensor W, bool mse_opt = true);
torch::Tensor int3_gemm(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K);
void draft_vocab_update_tail(
    torch::Tensor input_ids, torch::Tensor positions, torch::Tensor in_static_vocab,
    torch::Tensor tail_ids, torch::Tensor tail_count, torch::Tensor tail_seen);
torch::Tensor draft_vocab_tail_argmax(
    torch::Tensor A, torch::Tensor static_logits, torch::Tensor vocab_map,
    torch::Tensor tail_ids, torch::Tensor tail_count, torch::Tensor raw_w,
    torch::Tensor raw_s, torch::Tensor global_scale, double logit_scale,
    c10::optional<torch::Tensor> out_top2_row0 = c10::nullopt,
    c10::optional<torch::Tensor> out_margins = c10::nullopt,
    c10::optional<torch::Tensor> out_tree_top = c10::nullopt,
    c10::optional<torch::Tensor> out_tree_vals = c10::nullopt);
torch::Tensor draft_vocab_score_rows(
    torch::Tensor A, torch::Tensor row_ids, torch::Tensor raw_w,
    torch::Tensor raw_s, torch::Tensor global_scale, double logit_scale);
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
    double gamma = 0.92);
void native_leaf_commit_tree(
    torch::Tensor leaf_flag, torch::Tensor prev_ns,
    torch::Tensor hit_flag, torch::Tensor hit_src_row,
    torch::Tensor saved_leaf_toks, torch::Tensor saved_leaf_k,
    torch::Tensor leaf_u_toks, torch::Tensor sampled_tokens,
    torch::Tensor num_sampled, torch::Tensor num_rejected,
    int64_t tree_m = 12,
    c10::optional<torch::Tensor> stats = c10::nullopt, bool no_hit_mode = false,
    c10::optional<torch::Tensor> req_acc_state = c10::nullopt);
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
  m.def("rans_base", &bf16_rans_base);
  m.def("int3_pack", &int3_pack, pybind11::arg("W"), pybind11::arg("mse_opt") = true);
  m.def("int3_gemm", &int3_gemm);
  m.def("draft_vocab_update_tail", &draft_vocab_update_tail);
  m.def("draft_vocab_tail_argmax", &draft_vocab_tail_argmax,
        pybind11::arg("A"), pybind11::arg("static_logits"), pybind11::arg("vocab_map"),
        pybind11::arg("tail_ids"), pybind11::arg("tail_count"), pybind11::arg("raw_w"),
        pybind11::arg("raw_s"), pybind11::arg("global_scale"), pybind11::arg("logit_scale") = 1.0,
        pybind11::arg("out_top2_row0") = c10::nullopt,
        pybind11::arg("out_margins") = c10::nullopt,
        pybind11::arg("out_tree_top") = c10::nullopt,
        pybind11::arg("out_tree_vals") = c10::nullopt);
  m.def("draft_vocab_score_rows", &draft_vocab_score_rows,
        pybind11::arg("A"), pybind11::arg("row_ids"), pybind11::arg("raw_w"),
        pybind11::arg("raw_s"), pybind11::arg("global_scale"), pybind11::arg("logit_scale") = 1.0);
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
        pybind11::arg("gamma") = 0.92);
  m.def("native_leaf_commit_tree", &native_leaf_commit_tree,
        pybind11::arg("leaf_flag"), pybind11::arg("prev_ns"),
        pybind11::arg("hit_flag"), pybind11::arg("hit_src_row"),
        pybind11::arg("saved_leaf_toks"), pybind11::arg("saved_leaf_k"),
        pybind11::arg("leaf_u_toks"), pybind11::arg("sampled_tokens"),
        pybind11::arg("num_sampled"), pybind11::arg("num_rejected"),
        pybind11::arg("tree_m") = 12,
        pybind11::arg("stats") = c10::nullopt, pybind11::arg("no_hit_mode") = false,
        pybind11::arg("req_acc_state") = c10::nullopt);
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
}
