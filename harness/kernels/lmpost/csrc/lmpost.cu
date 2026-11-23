// SPDX-License-Identifier: Apache-2.0
// lmpost: bit-exact BF16 logit scaling + softcapping via a 64K PyTorch-derived LUT,
// and fused 1-pass softcap+argmax for serial M=1 decode without logprobs.
//
// Idea 33: Monotonic order-preserving BF16 key (zero 128 KB LUT reads in Pass 1) +
// Blackwell hardware redux.sync.max/min.u32 + exact equivalence-class tie verification
// (handles compressed/saturated tanh plateaus, -0.0 == +0.0 ties, and NaN skipping at 0-ULP).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace lmpost {

constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;
constexpr int kArgmaxGrid = 128;

__device__ __forceinline__ bool is_better(float va, int ia, float vb, int ib) {
  return (va > vb) || (va == vb && ia < ib);
}

__device__ __forceinline__ void pick_better(float& bv, int& bi, float v, int i) {
  if (is_better(v, i, bv, bi)) {
    bv = v;
    bi = i;
  }
}

__device__ __forceinline__ void block_reduce_max(float& bv, int& bi, float* sv, int* si, int warp, int lane) {
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    float ov = __shfl_down_sync(0xffffffffu, bv, off);
    int oi = __shfl_down_sync(0xffffffffu, bi, off);
    pick_better(bv, bi, ov, oi);
  }
  if (lane == 0) {
    sv[warp] = bv;
    si[warp] = bi;
  }
  __syncthreads();
  if (warp == 0) {
    bv = (lane < kWarps) ? sv[lane] : -INFINITY;
    bi = (lane < kWarps) ? si[lane] : 0x7fffffff;
#pragma unroll
    for (int off = 4; off >= 1; off >>= 1) {
      float ov = __shfl_down_sync(0xffffffffu, bv, off);
      int oi = __shfl_down_sync(0xffffffffu, bi, off);
      pick_better(bv, bi, ov, oi);
    }
  }
}

// Map raw BF16 bit-pattern u to a 16-bit unsigned sortable key:
// - NaN ((u & 0x7fffu) > 0x7f80u) -> 0u (always loses to any non-NaN key >= 0x007fu)
// - -Inf (0xff80u) .. -min_subnormal (0x8001u) -> 0x007fu .. 0x7ffeu
// - -0.0 (0x8000u) and +0.0 (0x0000u) -> 0x8000u (canonicalized so -0.0 == +0.0 ties go to lowest idx)
// - +min_subnormal (0x0001u) .. +Inf (0x7f80u) -> 0x8001u .. 0xff80u
__device__ __forceinline__ uint32_t bf16_to_okey(uint32_t u) {
  uint32_t abs_u = u & 0x7fffu;
  uint32_t sign_mask = 0x8000u | ((u >> 15) * 0x7fffu);
  uint32_t key = u ^ sign_mask;
  key = (key == 0x7fffu) ? 0x8000u : key;
  return (abs_u > 0x7f80u) ? 0u : key;
}

__device__ __forceinline__ uint16_t okey_to_bf16(uint32_t key) {
  return (uint16_t)(key ^ ((key >= 0x8000u) ? 0x8000u : 0xffffu));
}

__device__ __forceinline__ uint32_t warp_reduce_max_u32(uint32_t v) {
  uint32_t out;
  asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(out) : "r"(v));
  return out;
}

__device__ __forceinline__ uint32_t warp_reduce_min_u32(uint32_t v) {
  uint32_t out;
  asm("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(out) : "r"(v));
  return out;
}

__device__ __forceinline__ void block_reduce_okey(uint32_t& bk, uint32_t& bi, uint32_t* sk, uint32_t* si, int warp,
                                                  int lane) {
  uint32_t wmax = warp_reduce_max_u32(bk);
  uint32_t cand = (bk == wmax) ? bi : 0x7fffffffu;
  uint32_t wmin = warp_reduce_min_u32(cand);
  if (lane == 0) {
    sk[warp] = wmax;
    si[warp] = wmin;
  }
  __syncthreads();
  if (warp == 0) {
    bk = (lane < kWarps) ? sk[lane] : 0u;
    bi = (lane < kWarps) ? si[lane] : 0x7fffffffu;
    wmax = warp_reduce_max_u32(bk);
    cand = (bk == wmax) ? bi : 0x7fffffffu;
    bi = warp_reduce_min_u32(cand);
    bk = wmax;
  }
}

__global__ void softcap_inplace_vec8(__nv_bfloat16* __restrict__ logits, const uint16_t* __restrict__ lut,
                                     int n_vec) {
  uint4* vec = reinterpret_cast<uint4*>(logits);
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += stride) {
    uint4 v = vec[i];
    uint16_t* p = reinterpret_cast<uint16_t*>(&v);
#pragma unroll
    for (int k = 0; k < 8; ++k) p[k] = __ldg(lut + p[k]);
    vec[i] = v;
  }
}

__global__ void softcap_inplace_scalar(__nv_bfloat16* __restrict__ logits, const uint16_t* __restrict__ lut,
                                       int start, int total) {
  uint16_t* raw = reinterpret_cast<uint16_t*>(logits);
  const int stride = gridDim.x * blockDim.x;
  for (int i = start + blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
    raw[i] = __ldg(lut + raw[i]);
  }
}

__device__ __forceinline__ float lut_to_float(const uint16_t* lut, uint16_t code) {
  uint16_t m = __ldg(lut + code);
  __nv_bfloat16 bf = *reinterpret_cast<const __nv_bfloat16*>(&m);
  return __bfloat162float(bf);
}

template <bool Aligned>
__global__ void __launch_bounds__(kBlock)
softcap_argmax_lut_kernel(const __nv_bfloat16* __restrict__ logits, const uint16_t* __restrict__ lut,
                          int64_t* __restrict__ out_token, float* __restrict__ scratch_val,
                          int* __restrict__ scratch_idx, int* __restrict__ done_cnt, int n) {
  __shared__ float sv[kWarps];
  __shared__ int si[kWarps];
  __shared__ int am_last;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int gtid = blockIdx.x * kBlock + tid;
  const int stride = gridDim.x * kBlock;

  float bv = -INFINITY;
  int bi = 0x7fffffff;

  int vec_end = 0;
  if constexpr (Aligned) {
    const int n_vec = n >> 3;
    vec_end = n_vec << 3;
    const uint4* vec = reinterpret_cast<const uint4*>(logits);
    for (int i = gtid; i < n_vec; i += stride) {
      uint4 v = vec[i];
      const uint16_t* p = reinterpret_cast<const uint16_t*>(&v);
      const int base_idx = i << 3;
#pragma unroll
      for (int k = 0; k < 8; ++k) pick_better(bv, bi, lut_to_float(lut, p[k]), base_idx + k);
    }
  }
  const uint16_t* raw = reinterpret_cast<const uint16_t*>(logits);
  for (int i = vec_end + gtid; i < n; i += stride) {
    pick_better(bv, bi, lut_to_float(lut, raw[i]), i);
  }

  block_reduce_max(bv, bi, sv, si, warp, lane);
  if (tid == 0) {
    scratch_val[blockIdx.x] = bv;
    scratch_idx[blockIdx.x] = bi;
    __threadfence();
    am_last = (atomicAdd(done_cnt, 1) == (int)gridDim.x - 1) ? 1 : 0;
  }
  __syncthreads();
  if (am_last) {
    bv = (tid < (int)gridDim.x) ? ((const volatile float*)scratch_val)[tid] : -INFINITY;
    bi = (tid < (int)gridDim.x) ? ((const volatile int*)scratch_idx)[tid] : 0x7fffffff;
    block_reduce_max(bv, bi, sv, si, warp, lane);
    if (tid == 0) {
      *done_cnt = 0;
      *out_token = (int64_t)bi;
    }
  }
}

template <bool Aligned>
__global__ void __launch_bounds__(kBlock)
softcap_argmax_kernel(const __nv_bfloat16* __restrict__ logits, const uint16_t* __restrict__ lut,
                      int64_t* __restrict__ out_token, float* __restrict__ scratch_val,
                      int* __restrict__ scratch_idx, int* __restrict__ done_cnt, int n) {
  __shared__ uint32_t sk[kWarps];
  __shared__ uint32_t si[kWarps];
  __shared__ int am_last;
  __shared__ int sh_mode;
  __shared__ float sh_fy_max;
  __shared__ uint32_t sh_gidx;
  __shared__ uint32_t sh_tie_cnt;
  __shared__ int sh_tie_blks[kArgmaxGrid];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int gtid = blockIdx.x * kBlock + tid;
  const int stride = gridDim.x * kBlock;

  uint32_t best_key = 0u;
  uint32_t best_idx = 0x7fffffffu;

  int vec_end = 0;
  if constexpr (Aligned) {
    const int n_vec = n >> 3;
    vec_end = n_vec << 3;
    const uint4* vec = reinterpret_cast<const uint4*>(logits);
    for (int i = gtid; i < n_vec; i += stride) {
      uint4 v = vec[i];
      const uint16_t* p = reinterpret_cast<const uint16_t*>(&v);
      const uint32_t base_idx = (uint32_t)(i << 3);
      uint32_t vmax = 0u;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        uint32_t key = bf16_to_okey((uint32_t)p[k]);
        if (key != 0u) {
          uint32_t packed = (key << 3) | (7u - (uint32_t)k);
          if (packed > vmax) vmax = packed;
        }
      }
      uint32_t vkey = vmax >> 3;
      if (vkey > best_key) {
        best_key = vkey;
        best_idx = base_idx + (7u - (vmax & 7u));
      }
    }
  }
  const uint16_t* raw = reinterpret_cast<const uint16_t*>(logits);
  for (int i = vec_end + gtid; i < n; i += stride) {
    uint32_t key = bf16_to_okey((uint32_t)raw[i]);
    if (key > best_key) {
      best_key = key;
      best_idx = (uint32_t)i;
    }
  }

  block_reduce_okey(best_key, best_idx, sk, si, warp, lane);
  if (tid == 0) {
    reinterpret_cast<uint32_t*>(scratch_val)[blockIdx.x] = best_key;
    scratch_idx[blockIdx.x] = (int)best_idx;
    __threadfence();
    am_last = (atomicAdd(done_cnt, 1) == (int)gridDim.x - 1) ? 1 : 0;
  }
  __syncthreads();

  if (am_last) {
    const uint32_t blk_key = (tid < (int)gridDim.x) ? ((const volatile uint32_t*)scratch_val)[tid] : 0u;
    const uint32_t blk_idx = (tid < (int)gridDim.x) ? (uint32_t)((const volatile int*)scratch_idx)[tid] : 0x7fffffffu;

    uint32_t gkey = blk_key;
    uint32_t gidx = blk_idx;
    block_reduce_okey(gkey, gidx, sk, si, warp, lane);

    if (tid == 0) {
      sh_gidx = gidx;
      sh_tie_cnt = 0u;
      if (gkey == 0u) {
        sh_mode = 0;
      } else {
        float fy_max = lut_to_float(lut, okey_to_bf16(gkey));
        sh_fy_max = fy_max;

        float f_neg = lut_to_float(lut, 0xbf80u);
        float f_zero = lut_to_float(lut, 0x0000u);
        float f_pos = lut_to_float(lut, 0x3f80u);
        if (isnan(fy_max) || !(f_neg <= f_zero && f_zero <= f_pos && f_neg < f_pos)) {
          sh_mode = 2;
        } else if (gkey == 0x007fu) {
          sh_mode = 0;
        } else {
          uint32_t key_below = (gkey == 0x8000u) ? 0x7ffeu : (gkey - 1u);
          float fy_below = lut_to_float(lut, okey_to_bf16(key_below));
          if (fy_below == fy_max) {
            sh_mode = 1;
          } else if (fy_below > fy_max) {
            sh_mode = 2;
          } else {
            sh_mode = 0;
          }
        }
      }
    }
    __syncthreads();

    if (sh_mode == 0) {
      if (tid == 0) {
        *done_cnt = 0;
        *out_token = (int64_t)(int)sh_gidx;
      }
      return;
    }

    if (sh_mode == 1) {
      const float fy_max = sh_fy_max;
      float blk_fy = (blk_key > 0u) ? lut_to_float(lut, okey_to_bf16(blk_key)) : -INFINITY;
      bool bad_blk = (tid < (int)gridDim.x && blk_key > 0u) && (isnan(blk_fy) || blk_fy > fy_max);
      if (__syncthreads_or(bad_blk)) {
        if (tid == 0) sh_mode = 2;
        __syncthreads();
      } else {
        if (tid < (int)gridDim.x && blk_key > 0u && blk_fy == fy_max) {
          uint32_t slot = atomicAdd(&sh_tie_cnt, 1u);
          sh_tie_blks[slot] = tid;
        }
        __syncthreads();

        uint32_t best_tie_idx = sh_gidx;
        const int tie_cnt = (int)sh_tie_cnt;
        const int n_vec = Aligned ? (n >> 3) : 0;
        const int tail_start = n_vec << 3;
        const uint4* vec = reinterpret_cast<const uint4*>(logits);

        for (int t = 0; t < tie_cnt; ++t) {
          const int b = sh_tie_blks[t];
          if constexpr (Aligned) {
            for (int i = b * kBlock + tid; i < n_vec; i += stride) {
              const uint32_t base_idx = (uint32_t)(i << 3);
              if (base_idx >= best_tie_idx) break;
              uint4 v = vec[i];
              const uint16_t* p = reinterpret_cast<const uint16_t*>(&v);
#pragma unroll
              for (int k = 0; k < 8; ++k) {
                uint32_t idx = base_idx + (uint32_t)k;
                if (idx < best_tie_idx && lut_to_float(lut, p[k]) == fy_max) {
                  best_tie_idx = idx;
                  break;
                }
              }
            }
          }
          for (int i = tail_start + b * kBlock + tid; i < n; i += stride) {
            if ((uint32_t)i >= best_tie_idx) break;
            if (lut_to_float(lut, raw[i]) == fy_max) {
              best_tie_idx = (uint32_t)i;
              break;
            }
          }
        }

        uint32_t wmin = warp_reduce_min_u32(best_tie_idx);
        if (lane == 0) si[warp] = wmin;
        __syncthreads();
        if (warp == 0) {
          uint32_t bi = (lane < kWarps) ? si[lane] : 0x7fffffffu;
          bi = warp_reduce_min_u32(bi);
          if (tid == 0) {
            *done_cnt = 0;
            *out_token = (int64_t)(int)bi;
          }
        }
        return;
      }
    }

    // Fallback (sh_mode == 2): exact LUT reduction across [0, n) inside the last block.
    __syncthreads();
    {
      float bv = -INFINITY;
      int bi = 0x7fffffff;
      int fb_vec_end = 0;
      if constexpr (Aligned) {
        const int n_vec = n >> 3;
        fb_vec_end = n_vec << 3;
        const uint4* vec = reinterpret_cast<const uint4*>(logits);
        for (int i = tid; i < n_vec; i += kBlock) {
          uint4 v = vec[i];
          const uint16_t* p = reinterpret_cast<const uint16_t*>(&v);
          const int base_idx = i << 3;
#pragma unroll
          for (int k = 0; k < 8; ++k) pick_better(bv, bi, lut_to_float(lut, p[k]), base_idx + k);
        }
      }
      for (int i = fb_vec_end + tid; i < n; i += kBlock) {
        pick_better(bv, bi, lut_to_float(lut, raw[i]), i);
      }
      block_reduce_max(bv, bi, reinterpret_cast<float*>(sk), reinterpret_cast<int*>(si), warp, lane);
      if (tid == 0) {
        *done_cnt = 0;
        *out_token = (int64_t)bi;
      }
    }
  }
}

}  // namespace lmpost

extern "C" int lmpost_init() {
  cudaFuncAttributes attr;
  cudaError_t rc;
  rc = cudaFuncGetAttributes(&attr, lmpost::softcap_inplace_vec8);
  if (rc != cudaSuccess) return (int)rc;
  rc = cudaFuncGetAttributes(&attr, lmpost::softcap_inplace_scalar);
  if (rc != cudaSuccess) return (int)rc;
  rc = cudaFuncGetAttributes(&attr, lmpost::softcap_argmax_kernel<true>);
  if (rc != cudaSuccess) return (int)rc;
  rc = cudaFuncGetAttributes(&attr, lmpost::softcap_argmax_kernel<false>);
  if (rc != cudaSuccess) return (int)rc;
  rc = cudaFuncGetAttributes(&attr, lmpost::softcap_argmax_lut_kernel<true>);
  if (rc != cudaSuccess) return (int)rc;
  return (int)cudaFuncGetAttributes(&attr, lmpost::softcap_argmax_lut_kernel<false>);
}

extern "C" int lmpost_softcap_inplace(void* logits, const void* lut, int total_elems, void* stream) {
  using namespace lmpost;
  if (total_elems <= 0) return (int)cudaErrorInvalidValue;
  cudaStream_t st = (cudaStream_t)stream;
  const uint16_t* lut_p = (const uint16_t*)lut;
  __nv_bfloat16* log_p = (__nv_bfloat16*)logits;
  if (((uintptr_t)logits & 15) == 0 && total_elems >= 8) {
    const int n_vec = total_elems >> 3;
    int grid = (n_vec + kBlock - 1) / kBlock;
    if (grid > 512) grid = 512;
    softcap_inplace_vec8<<<grid, kBlock, 0, st>>>(log_p, lut_p, n_vec);
    const int tail_start = n_vec << 3;
    if (tail_start < total_elems) {
      softcap_inplace_scalar<<<1, kBlock, 0, st>>>(log_p, lut_p, tail_start, total_elems);
    }
  } else {
    int grid = (total_elems + kBlock - 1) / kBlock;
    if (grid > 512) grid = 512;
    softcap_inplace_scalar<<<grid, kBlock, 0, st>>>(log_p, lut_p, 0, total_elems);
  }
  return (int)cudaGetLastError();
}

extern "C" int lmpost_softcap_argmax(const void* logits, const void* lut, void* out_token, void* scratch_val,
                                     void* scratch_idx, void* done_cnt, int n, void* stream) {
  using namespace lmpost;
  if (n <= 0) return (int)cudaErrorInvalidValue;
  cudaStream_t st = (cudaStream_t)stream;
  if (((uintptr_t)logits & 15) == 0 && n >= 8) {
    softcap_argmax_kernel<true><<<kArgmaxGrid, kBlock, 0, st>>>(
        (const __nv_bfloat16*)logits, (const uint16_t*)lut, (int64_t*)out_token, (float*)scratch_val,
        (int*)scratch_idx, (int*)done_cnt, n);
  } else {
    softcap_argmax_kernel<false><<<kArgmaxGrid, kBlock, 0, st>>>(
        (const __nv_bfloat16*)logits, (const uint16_t*)lut, (int64_t*)out_token, (float*)scratch_val,
        (int*)scratch_idx, (int*)done_cnt, n);
  }
  return (int)cudaGetLastError();
}

extern "C" int lmpost_softcap_argmax_lut(const void* logits, const void* lut, void* out_token, void* scratch_val,
                                         void* scratch_idx, void* done_cnt, int n, void* stream) {
  using namespace lmpost;
  if (n <= 0) return (int)cudaErrorInvalidValue;
  cudaStream_t st = (cudaStream_t)stream;
  if (((uintptr_t)logits & 15) == 0 && n >= 8) {
    softcap_argmax_lut_kernel<true><<<kArgmaxGrid, kBlock, 0, st>>>(
        (const __nv_bfloat16*)logits, (const uint16_t*)lut, (int64_t*)out_token, (float*)scratch_val,
        (int*)scratch_idx, (int*)done_cnt, n);
  } else {
    softcap_argmax_lut_kernel<false><<<kArgmaxGrid, kBlock, 0, st>>>(
        (const __nv_bfloat16*)logits, (const uint16_t*)lut, (int64_t*)out_token, (float*)scratch_val,
        (int*)scratch_idx, (int*)done_cnt, n);
  }
  return (int)cudaGetLastError();
}
