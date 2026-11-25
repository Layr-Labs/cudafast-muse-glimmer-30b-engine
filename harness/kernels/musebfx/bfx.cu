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
torch::Tensor int3_pack(torch::Tensor W);
torch::Tensor int3_gemm(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K);
void int3_gemm_out(torch::Tensor A, torch::Tensor W_packed, torch::Tensor C, int64_t N, int64_t K);
void int3_gemm_aux5_out(
    torch::Tensor x0, torch::Tensor x1, torch::Tensor x2, torch::Tensor x3, torch::Tensor x4,
    torch::Tensor W_packed, torch::Tensor C, int64_t N, int64_t K_each);
torch::Tensor int4_pack(torch::Tensor W);
torch::Tensor int4_gemm(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K);
void int4_gemm_out(torch::Tensor A, torch::Tensor W_packed, torch::Tensor C, int64_t N, int64_t K);
void int4_gemm_aux5_out(
    torch::Tensor x0, torch::Tensor x1, torch::Tensor x2, torch::Tensor x3, torch::Tensor x4,
    torch::Tensor W_packed, torch::Tensor C, int64_t N, int64_t K_each);
void dflash_kv_post(
    torch::Tensor all_kv_flat, torch::Tensor k_norm_weights, torch::Tensor cos_sin_cache,
    torch::Tensor positions, c10::optional<torch::Tensor> slot_mapping,
    const std::vector<int64_t>& kv_ptrs, int64_t block_size,
    int64_t stride_block, int64_t stride_kv_head, int64_t stride_slot, double eps);
void dflash_argmax_bf16(torch::Tensor logits, torch::Tensor scratch_val, torch::Tensor scratch_idx,
                        torch::Tensor out_tok);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &bfx_gemm);
  m.def("rans_pack", &bf16_rans_pack);
  m.def("rans_unpack", &bf16_rans_unpack);
  m.def("rans_gemm", &bf16_rans_gemm);
  m.def("rans_gemm_out", &bf16_rans_gemm_out);
  m.def("rans_base", &bf16_rans_base);
  m.def("int3_pack", &int3_pack);
  m.def("int3_gemm", &int3_gemm);
  m.def("int3_gemm_out", &int3_gemm_out);
  m.def("int3_gemm_aux5_out", &int3_gemm_aux5_out);
  m.def("int4_pack", &int4_pack);
  m.def("int4_gemm", &int4_gemm);
  m.def("int4_gemm_out", &int4_gemm_out);
  m.def("int4_gemm_aux5_out", &int4_gemm_aux5_out);
  m.def("dflash_kv_post", &dflash_kv_post);
  m.def("dflash_argmax_bf16", &dflash_argmax_bf16);
}
