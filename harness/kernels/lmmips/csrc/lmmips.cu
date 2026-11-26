// lmmips: H2 exact Cauchy-Schwarz / rank-32 subspace branch-and-bound for the target lm_head argmax (MUSE_LM_MIPS).
//
// Load time (harness/kernels/lmmips/py/__init__.py) builds, from the dense kernel's own effective weights W (row v =
// global_scale * dequantized NVFP4 row, recovered through the dense GEMV), any V in R^{K x 32} (randomized range
// finder) and per vocab row v:  c^_v = fp16(V^T w_v),  rho_v = w_v - V c^_v  (fp64),
//     d_v = ||V^T rho_v|| + 34 u ||c^_v||   r_v = ||rho_v||   e_v = delta_v + 1e-10 ||c^_v||   (fp32, rounded up)
// where delta_v bounds ||W_eff,v - W~_v|| (0 for rows whose probe values decode uniquely).
// For a query x (bf16) with q = fp32(V^T x) (fp64 accumulation), y = x - V q, every real dot product obeys
//     x . w_v = (V^T x) . c^_v + q . (V^T rho_v) + y . rho_v
//            <= fl32(q . c^_v) + ||q|| d_v + ||y|| r_v + ||x|| e_v          (Cauchy-Schwarz, no orthonormality needed)
// and the dense kernel's fp32 value z_v = fl(fl_TC(sum x_k b_vk) * g) is within err_i = coef ||x|| max_v ||w_v|| of
// it. A tile T is skipped for row i iff  U_{i,T} < zlo(F_i) - eps_i,  eps_i = err_i + eps_floor (>= 0.10), where
// F_i = softcapped value of the seed winner and zlo(F) = the largest fp32 z whose bf16 rounding (RNE) has every LUT
// value < F. A skipped column is then strictly below F_i <= the dense maximum: it can neither win nor tie, so the
// lowest-index argmax over the kept tiles equals the dense argmax bit for bit.
//
// Per call (all on the caller's stream, fixed buffers, no host sync, CUDA-graph capturable):
//   lmmips_project  -> rowbuf       lmmips_bound -> U[16][tiles]      lmmips_seed -> seed tile mask
//   [masked lm_head over seed tiles: tok1/val1]   lmmips_thresh -> pass-2 tile mask
//   [masked lm_head over pass-2 tiles: tok2/val2] lmmips_merge -> out tokens
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

constexpr int kR = 32;
constexpr int kMaxM = 16;
constexpr int kRowF = 40;  // rowbuf floats per row: q[0..31], qn, yn, xn, eps, finite, pad
constexpr int kTile = 128;

__device__ __forceinline__ float up_f(double v) { return __double2float_ru(v); }

// One CTA per row. fp64 accumulation: q = V^T x, ||x||^2; then ||x - V q32||^2 with q32 the rounded q actually used.
__global__ void __launch_bounds__(256) lmmips_project_kernel(
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ vmat,  // vmat [K][32]
    int k, float wmax, float err_coef, float eps_floor, float* __restrict__ rowbuf) {
  const int row = blockIdx.x;
  const __nv_bfloat16* xr = x + static_cast<size_t>(row) * k;
  __shared__ double sh_red[8][kR + 1];
  __shared__ float sh_q[kR];
  __shared__ double sh_tot[2];
  __shared__ int sh_bad;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) sh_bad = 0;
  __syncthreads();
  double acc[kR];
#pragma unroll
  for (int j = 0; j < kR; ++j) acc[j] = 0.0;
  double xx = 0.0;
  int bad = 0;
  for (int kk = tid; kk < k; kk += 256) {
    const float xf = __bfloat162float(xr[kk]);
    if (!isfinite(xf)) bad = 1;
    const double xd = static_cast<double>(xf);
    xx = fma(xd, xd, xx);
    const float4* vr = reinterpret_cast<const float4*>(vmat + static_cast<size_t>(kk) * kR);
#pragma unroll
    for (int j4 = 0; j4 < kR / 4; ++j4) {
      const float4 v = __ldg(vr + j4);
      acc[4 * j4 + 0] = fma(xd, static_cast<double>(v.x), acc[4 * j4 + 0]);
      acc[4 * j4 + 1] = fma(xd, static_cast<double>(v.y), acc[4 * j4 + 1]);
      acc[4 * j4 + 2] = fma(xd, static_cast<double>(v.z), acc[4 * j4 + 2]);
      acc[4 * j4 + 3] = fma(xd, static_cast<double>(v.w), acc[4 * j4 + 3]);
    }
  }
  if (bad) atomicOr(&sh_bad, 1);
#pragma unroll
  for (int j = 0; j < kR; ++j) {
    double v = acc[j];
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) sh_red[warp][j] = v;
  }
  {
    double v = xx;
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) sh_red[warp][kR] = v;
  }
  __syncthreads();
  if (tid <= kR) {
    double v = 0.0;
    for (int w = 0; w < 8; ++w) v += sh_red[w][tid];
    if (tid < kR) {
      sh_q[tid] = static_cast<float>(v);  // q32: the value every later bound uses
    } else {
      sh_tot[0] = v;
    }
  }
  __syncthreads();
  double yy = 0.0;
  for (int kk = tid; kk < k; kk += 256) {
    const double xd = static_cast<double>(__bfloat162float(xr[kk]));
    const float* vr = vmat + static_cast<size_t>(kk) * kR;
    double proj = 0.0;
#pragma unroll
    for (int j = 0; j < kR; ++j) proj = fma(static_cast<double>(__ldg(vr + j)), static_cast<double>(sh_q[j]), proj);
    const double y = xd - proj;
    yy = fma(y, y, yy);
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) yy += __shfl_xor_sync(0xffffffffu, yy, off);
  __syncthreads();
  if (lane == 0) sh_red[warp][0] = yy;
  __syncthreads();
  if (tid == 0) {
    double ysum = 0.0;
    for (int w = 0; w < 8; ++w) ysum += sh_red[w][0];
    double qq = 0.0;
    for (int j = 0; j < kR; ++j) qq += static_cast<double>(sh_q[j]) * static_cast<double>(sh_q[j]);
    // fp64 sums of K <= 16384 non-negative terms: relative error < 2e-12; 1e-9 slack covers it with margin.
    const double g = 1.0 + 1e-9;
    const float qn = up_f(sqrt(qq) * g);
    const float yn = up_f(sqrt(ysum) * g);
    const float xn = up_f(sqrt(sh_tot[0]) * g);
    float* rb = rowbuf + row * kRowF;
    for (int j = 0; j < kR; ++j) rb[j] = sh_q[j];
    rb[kR + 0] = qn;
    rb[kR + 1] = yn;
    rb[kR + 2] = xn;
    const double err = static_cast<double>(err_coef) * static_cast<double>(xn) * static_cast<double>(wmax);
    float eps = up_f(err + static_cast<double>(eps_floor));
    const bool finite = (sh_bad == 0) && isfinite(eps) && isfinite(qn) && isfinite(yn) && isfinite(xn);
    if (!finite) eps = __int_as_float(0x7f800000);
    rb[kR + 3] = eps;
    rb[kR + 4] = finite ? 1.0f : 0.0f;
  }
}

// One CTA (128 threads) per 128-row tile; thread t owns vocab row 128 T + t.
// U[i][T] = max_v fl32(q_i . c^_v) + qn_i d_v + yn_i r_v + xn_i e_v  (padded rows -> -inf).
__global__ void __launch_bounds__(kTile) lmmips_bound_kernel(
    const __half* __restrict__ chat, const float4* __restrict__ aux, const float* __restrict__ rowbuf,
    int m, int org_vocab, int n_tiles, float* __restrict__ ubuf) {
  __shared__ float sh_rb[kMaxM * kRowF];
  __shared__ float sh_max[kTile / 32][kMaxM];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < m * kRowF; i += kTile) sh_rb[i] = rowbuf[i];
  __syncthreads();
  const int tile = blockIdx.x;
  const int v = tile * kTile + tid;
  float c[kR];
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  const bool live = v < org_vocab;
  if (live) {
    const uint4* cp = reinterpret_cast<const uint4*>(chat + static_cast<size_t>(v) * kR);
#pragma unroll
    for (int j8 = 0; j8 < kR / 8; ++j8) {
      const uint4 u = __ldg(cp + j8);
      const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        const __half2 hv = *reinterpret_cast<const __half2*>(&w[h]);
        const float2 f = __half22float2(hv);
        c[8 * j8 + 2 * h] = f.x;
        c[8 * j8 + 2 * h + 1] = f.y;
      }
    }
    a = __ldg(aux + v);
  }
  for (int i = 0; i < m; ++i) {
    const float* rb = sh_rb + i * kRowF;
    float u = -__int_as_float(0x7f800000);
    if (live) {
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < kR; ++j) s = fmaf(rb[j], c[j], s);
      u = s + rb[kR + 0] * a.x + rb[kR + 1] * a.y + rb[kR + 2] * a.z;
      if (u != u) u = __int_as_float(0x7f800000);  // NaN bound -> keep the tile
    }
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) u = fmaxf(u, __shfl_xor_sync(0xffffffffu, u, off));
    if (lane == 0) sh_max[warp][i] = u;
  }
  __syncthreads();
  if (tid < m) {
    float u = sh_max[0][tid];
#pragma unroll
    for (int w = 1; w < kTile / 32; ++w) u = fmaxf(u, sh_max[w][tid]);
    ubuf[tid * n_tiles + tile] = u;
  }
}

// One CTA, one warp per row: the top-`topk` tiles of each row by U plus tiles [0, hot_tiles) form the seed.
// Any non-finite row -> empty seed (pass 2 then evaluates every tile, i.e. runs the dense kernel exactly).
__global__ void __launch_bounds__(512) lmmips_seed_kernel(
    const float* __restrict__ ubuf, const float* __restrict__ rowbuf, int m, int n_tiles, int hot_tiles, int topk,
    uint32_t* __restrict__ seed_mask, int n_words) {
  __shared__ uint32_t sh_mask[64];
  __shared__ int sh_bad;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid < 64) sh_mask[tid] = 0u;
  if (tid == 0) sh_bad = 0;
  __syncthreads();
  if (tid < m && rowbuf[tid * kRowF + kR + 4] == 0.0f) atomicOr(&sh_bad, 1);
  for (int t = tid; t < hot_tiles && t < n_tiles; t += blockDim.x) atomicOr(&sh_mask[t >> 5], 1u << (t & 31));
  if (warp < m) {
    const float* ur = ubuf + warp * n_tiles;
    int chosen[4] = {-1, -1, -1, -1};
    for (int p = 0; p < topk && p < 4; ++p) {
      float best = -__int_as_float(0x7f800000);
      int bi = 0x7fffffff;
      for (int t = lane; t < n_tiles; t += 32) {
        if (t == chosen[0] || t == chosen[1] || t == chosen[2] || t == chosen[3]) continue;
        const float u = ur[t];
        if (u > best || (u == best && t < bi)) {
          best = u;
          bi = t;
        }
      }
#pragma unroll
      for (int off = 16; off >= 1; off >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, off);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ob > best || (ob == best && oi < bi)) {
          best = ob;
          bi = oi;
        }
      }
      if (bi >= n_tiles) break;
      chosen[p] = bi;
      if (lane == 0) atomicOr(&sh_mask[bi >> 5], 1u << (bi & 31));
    }
  }
  __syncthreads();
  if (tid < n_words) seed_mask[tid] = sh_bad ? 0u : sh_mask[tid];
}

// Thread per tile. keep(T) = OR_i !(U[i][T] < thr_i), thr_i = RD(zlo(F_i) - eps_i); pass-2 mask = keep & ~seed.
// Rows whose seed value is NaN / -inf get thr = -inf (keep every tile).
__global__ void __launch_bounds__(256) lmmips_thresh_kernel(
    const float* __restrict__ ubuf, const float* __restrict__ rowbuf, const float* __restrict__ val1,
    const float* __restrict__ zlo, const uint32_t* __restrict__ seed_mask, int m, int n_tiles, int n_words,
    uint32_t* __restrict__ mask2, unsigned long long* __restrict__ stats) {
  __shared__ float sh_thr[kMaxM];
  const int tid = threadIdx.x;
  if (tid < m) {
    const float f = val1[tid];
    const float eps = rowbuf[tid * kRowF + kR + 3];
    float thr = -__int_as_float(0x7f800000);
    if (f == f && f != -__int_as_float(0x7f800000)) {
      const uint32_t code = __float_as_uint(f) >> 16;  // val1 is a widened bf16: low half is zero
      thr = __fsub_rd(__ldg(zlo + code), eps);
      if (thr != thr) thr = -__int_as_float(0x7f800000);
    }
    sh_thr[tid] = thr;
  }
  __syncthreads();
  const int t = blockIdx.x * blockDim.x + tid;
  bool keep = false;
  if (t < n_tiles) {
    for (int i = 0; i < m; ++i) {
      if (!(ubuf[i * n_tiles + t] < sh_thr[i])) {
        keep = true;
        break;
      }
    }
    if ((__ldg(seed_mask + (t >> 5)) >> (t & 31)) & 1u) keep = false;
  }
  const uint32_t word = __ballot_sync(0xffffffffu, keep);
  const int w = t >> 5;
  if ((tid & 31) == 0 && w < n_words) {
    mask2[w] = word;
    if (stats != nullptr) atomicAdd(stats, static_cast<unsigned long long>(__popc(word)));
  }
  if (stats != nullptr && blockIdx.x == 0 && tid == 0) atomicAdd(stats + 1, 1ull);
}

// out[i] = lowest-index argmax over the union of the two disjoint passes (pass 1 wins only when strictly better or
// equal with a lower index, so NaN / empty passes resolve exactly as the single dense pass).
__global__ void lmmips_merge_kernel(const int64_t* __restrict__ tok1, const float* __restrict__ val1,
                                    const int64_t* __restrict__ tok2, const float* __restrict__ val2, int m,
                                    int64_t* __restrict__ out) {
  const int i = threadIdx.x;
  if (i < m) {
    const float v1 = val1[i], v2 = val2[i];
    const int64_t t1 = tok1[i], t2 = tok2[i];
    out[i] = (v1 > v2 || (v1 == v2 && t1 < t2)) ? t1 : t2;
  }
}

}  // namespace

extern "C" int lmmips_project(const void* x, const void* vmat, int k, int m, float wmax, float err_coef,
                              float eps_floor, void* rowbuf, cudaStream_t stream) {
  if (m < 1 || m > kMaxM || k <= 0) return cudaErrorInvalidValue;
  lmmips_project_kernel<<<m, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x), static_cast<const float*>(vmat),
                                               k, wmax, err_coef, eps_floor, static_cast<float*>(rowbuf));
  return cudaGetLastError();
}

extern "C" int lmmips_bound(const void* chat, const void* aux, const void* rowbuf, int m, int org_vocab, int n_tiles,
                            void* ubuf, cudaStream_t stream) {
  if (m < 1 || m > kMaxM || n_tiles <= 0) return cudaErrorInvalidValue;
  lmmips_bound_kernel<<<n_tiles, kTile, 0, stream>>>(
      static_cast<const __half*>(chat), static_cast<const float4*>(aux), static_cast<const float*>(rowbuf), m,
      org_vocab, n_tiles, static_cast<float*>(ubuf));
  return cudaGetLastError();
}

extern "C" int lmmips_seed(const void* ubuf, const void* rowbuf, int m, int n_tiles, int hot_tiles, int topk,
                           void* seed_mask, int n_words, cudaStream_t stream) {
  if (m < 1 || m > kMaxM || n_words > 64 || n_words * 32 < n_tiles) return cudaErrorInvalidValue;
  lmmips_seed_kernel<<<1, 512, 0, stream>>>(static_cast<const float*>(ubuf), static_cast<const float*>(rowbuf), m,
                                            n_tiles, hot_tiles, topk, static_cast<uint32_t*>(seed_mask), n_words);
  return cudaGetLastError();
}

extern "C" int lmmips_thresh(const void* ubuf, const void* rowbuf, const void* val1, const void* zlo,
                             const void* seed_mask, int m, int n_tiles, int n_words, void* mask2, void* stats,
                             cudaStream_t stream) {
  if (m < 1 || m > kMaxM || n_words * 32 < n_tiles) return cudaErrorInvalidValue;
  const int grid = (n_words * 32 + 255) / 256;
  lmmips_thresh_kernel<<<grid, 256, 0, stream>>>(
      static_cast<const float*>(ubuf), static_cast<const float*>(rowbuf), static_cast<const float*>(val1),
      static_cast<const float*>(zlo), static_cast<const uint32_t*>(seed_mask), m, n_tiles, n_words,
      static_cast<uint32_t*>(mask2), static_cast<unsigned long long*>(stats));
  return cudaGetLastError();
}

extern "C" int lmmips_merge(const void* tok1, const void* val1, const void* tok2, const void* val2, int m, void* out,
                            cudaStream_t stream) {
  if (m < 1 || m > kMaxM) return cudaErrorInvalidValue;
  lmmips_merge_kernel<<<1, 32, 0, stream>>>(static_cast<const int64_t*>(tok1), static_cast<const float*>(val1),
                                            static_cast<const int64_t*>(tok2), static_cast<const float*>(val2), m,
                                            static_cast<int64_t*>(out));
  return cudaGetLastError();
}
