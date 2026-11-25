// musebfx: lossless 12-bit coded BF16 weights for small-M (<=8) GEMM on GB10.
// Weight w[n,k] (bf16) is stored as lo[n,k] = sign<<7 | mantissa7 and a 4-bit code c[n,k]:
//   c in 1..15 -> exponent = base[n] + c ; c == 0 -> contributes 0 (zeros and out-of-window
//   values; every nonzero out-of-window value is in the per-row escape list (CSR), added exactly).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cstdlib>
#include <algorithm>

constexpr int CH = 512;   // k chunk
constexpr int RPW = 2;    // rows per warp
constexpr int WARPS = 8;

template <int M>
__global__ void __launch_bounds__(256) bfx_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ lo,
    const uint8_t* __restrict__ nib, const int* __restrict__ base, const int* __restrict__ rowptr,
    const int* __restrict__ ecol, const __nv_bfloat16* __restrict__ eval, __nv_bfloat16* __restrict__ out, int N, int K, float* __restrict__ out32, int kper) {
  __shared__ float xs[M][CH];
  const int kb = blockIdx.y * kper, ke = min(K, kb + kper);
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
  for (int kc = kb; kc < ke; kc += CH) {
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
      const int k = ecol[e];
      if (k < kb || k >= ke) continue;
      const float v = __bfloat162float(eval[e]);
#pragma unroll
      for (int m = 0; m < M; m++) acc[r][m] = fmaf(v, __bfloat162float(x[(size_t)m * K + k]), acc[r][m]);
    }
#pragma unroll
    for (int m = 0; m < M; m++) {
      float a = acc[r][m];
#pragma unroll
      for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0) {
        if (out32) atomicAdd(out32 + (size_t)m * N + n, a);
        else out[(size_t)m * N + n] = __float2bfloat16_rn(a);
      }
    }
  }
}

__global__ void bfx_cvt(const float* __restrict__ a, __nv_bfloat16* __restrict__ o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = __float2bfloat16_rn(a[i]);
}

torch::Tensor bfx_gemm(torch::Tensor x, torch::Tensor lo, torch::Tensor nib, torch::Tensor base, torch::Tensor rowptr,
                       torch::Tensor ecol, torch::Tensor eval) {
  const int M = x.size(0), K = x.size(1), N = lo.size(0);
  TORCH_CHECK(M >= 1 && M <= 8 && K % CH == 0 && lo.size(1) == K && x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
  auto out = torch::empty({M, N}, x.options());
  const int blocks = (N + WARPS * RPW - 1) / (WARPS * RPW);
  static const int split_env = [] { const char* e = std::getenv("MUSEBFX_SPLIT"); return e ? atoi(e) : -1; }();
  const int nch = K / CH;
  int S = split_env;
  if (S < 0) S = (blocks >= 1536) ? 1 : std::min(nch, (1536 + blocks - 1) / blocks);
  S = std::max(1, std::min(S, nch));
  const int kper = ((nch + S - 1) / S) * CH;
  S = (K + kper - 1) / kper;
  dim3 grid(blocks, S);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  torch::Tensor ws;
  float* o32 = nullptr;
  if (S > 1) {
    ws = torch::zeros({M, N}, x.options().dtype(torch::kFloat));
    o32 = ws.data_ptr<float>();
  }
#define L_(MM) case MM: bfx_kernel<MM><<<grid, 256, 0, st>>>((const __nv_bfloat16*)x.data_ptr(), lo.data_ptr<uint8_t>(), \
    nib.data_ptr<uint8_t>(), base.data_ptr<int>(), rowptr.data_ptr<int>(), ecol.data_ptr<int>(), \
    (const __nv_bfloat16*)eval.data_ptr(), (__nv_bfloat16*)out.data_ptr(), N, K, o32, kper); break;
  switch (M) { L_(1) L_(2) L_(3) L_(4) L_(5) L_(6) L_(7) L_(8) }
#undef L_
  if (S > 1) bfx_cvt<<<(M * N + 255) / 256, 256, 0, st>>>(o32, (__nv_bfloat16*)out.data_ptr(), M * N);
  return out;
}


// ---- 4-bit drafter weights: NF4 code per weight, bf16 absmax per group of 32 (4.5 bits/weight) ----
template <int M, int R>
__global__ void __launch_bounds__(256) nf4_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, const float* __restrict__ lut, __nv_bfloat16* __restrict__ out, int N, int K,
    float* __restrict__ out32, int kper) {
  __shared__ float xs[M][CH];
  __shared__ float lt[16];
  if (threadIdx.x < 16) lt[threadIdx.x] = lut[threadIdx.x];
  const int kb = blockIdx.y * kper, ke = min(K, kb + kper);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int n0 = (blockIdx.x * WARPS + warp) * R;
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int m = 0; m < M; m++) acc[r][m] = 0.f;
  const int G = K / 32;
  for (int kc = kb; kc < ke; kc += CH) {
    uint2 C[R]; float S[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
      const int n = n0 + r;
      if (n < N) {
        C[r] = __ldg(reinterpret_cast<const uint2*>(codes + (size_t)n * (K / 2) + kc / 2 + lane * 8));
        S[r] = __bfloat162float(scales[(size_t)n * G + (kc + lane * 16) / 32]);
      } else { C[r] = make_uint2(0, 0); S[r] = 0.f; }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < M * CH; i += 256) {
      const int m = i / CH, k = i % CH;
      xs[m][(k & 15) * 32 + (k >> 4)] = __bfloat162float(x[(size_t)m * K + kc + k]);
    }
    __syncthreads();
    uint64_t cw[R];
#pragma unroll
    for (int r = 0; r < R; r++) cw[r] = (uint64_t)C[r].x | ((uint64_t)C[r].y << 32);
#pragma unroll
    for (int i = 0; i < 16; i++) {
      float xv[M];
#pragma unroll
      for (int m = 0; m < M; m++) xv[m] = xs[m][i * 32 + lane];
#pragma unroll
      for (int r = 0; r < R; r++) {
        const float w = lt[(cw[r] >> (4 * i)) & 0xF] * S[r];
#pragma unroll
        for (int m = 0; m < M; m++) acc[r][m] = fmaf(w, xv[m], acc[r][m]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int n = n0 + r;
    if (n >= N) continue;
#pragma unroll
    for (int m = 0; m < M; m++) {
      float a = acc[r][m];
#pragma unroll
      for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0) {
        if (out32) atomicAdd(out32 + (size_t)m * N + n, a);
        else out[(size_t)m * N + n] = __float2bfloat16_rn(a);
      }
    }
  }
}

torch::Tensor nf4_gemm(torch::Tensor x, torch::Tensor codes, torch::Tensor scales, torch::Tensor lut) {
  const int M = x.size(0), K = x.size(1), N = codes.size(0);
  TORCH_CHECK(M >= 1 && M <= 8 && K % CH == 0 && codes.size(1) * 2 == K && x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
  auto out = torch::empty({M, N}, x.options());
  static const int r4 = [] { const char* e = std::getenv("MUSENF4_R"); return e ? atoi(e) : 4; }();
  const int blocks = (N + WARPS * r4 - 1) / (WARPS * r4);
  const int nch = K / CH;
  int S = (blocks >= 1536) ? 1 : std::min(nch, (1536 + blocks - 1) / blocks);
  S = std::max(1, std::min(S, nch));
  const int kper = ((nch + S - 1) / S) * CH;
  S = (K + kper - 1) / kper;
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  torch::Tensor ws; float* o32 = nullptr;
  if (S > 1) { ws = torch::zeros({M, N}, x.options().dtype(torch::kFloat)); o32 = ws.data_ptr<float>(); }
  dim3 grid(blocks, S);
#define L4R_(MM, RR) nf4_kernel<MM, RR><<<grid, 256, 0, st>>>((const __nv_bfloat16*)x.data_ptr(), codes.data_ptr<uint8_t>(), \
    (const __nv_bfloat16*)scales.data_ptr(), lut.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), N, K, o32, kper)
#define L4_(MM) case MM: if (r4 == 4) L4R_(MM, 4); else if (r4 == 8) L4R_(MM, 8); else L4R_(MM, 2); break;
  switch (M) { L4_(1) L4_(2) L4_(3) L4_(4) L4_(5) L4_(6) L4_(7) L4_(8) }
#undef L4_
#undef L4R_
  if (S > 1) bfx_cvt<<<(M * N + 255) / 256, 256, 0, st>>>(o32, (__nv_bfloat16*)out.data_ptr(), M * N);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &bfx_gemm); m.def("gemm4", &nf4_gemm); }
