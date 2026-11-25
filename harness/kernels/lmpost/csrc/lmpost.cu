// SPDX-License-Identifier: Apache-2.0
// lmpost: bit-exact BF16 logit scaling + softcapping via a 64K PyTorch-derived LUT,
// and fused 1-pass softcap+argmax for serial M=1 decode without logprobs.
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
softcap_argmax_kernel(const __nv_bfloat16* __restrict__ logits, const uint16_t* __restrict__ lut,
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
__global__ void softcap_inplace_strided_kernel(
    __nv_bfloat16* __restrict__ logits,
    const uint16_t* __restrict__ lut,
    int cols,
    int row_stride) {
  __nv_bfloat16* row = logits + static_cast<size_t>(blockIdx.y) * static_cast<size_t>(row_stride);
  const int gtid = blockIdx.x * kBlock + threadIdx.x;
  const int stride = gridDim.x * kBlock;
  int vec_end = 0;
  if constexpr (Aligned) {
    const int n_vec = cols >> 3;
    vec_end = n_vec << 3;
    uint4* vec = reinterpret_cast<uint4*>(row);
    for (int i = gtid; i < n_vec; i += stride) {
      uint4 v = vec[i];
      uint16_t* p = reinterpret_cast<uint16_t*>(&v);
#pragma unroll
      for (int k = 0; k < 8; ++k) p[k] = __ldg(lut + p[k]);
      vec[i] = v;
    }
  }
  uint16_t* raw = reinterpret_cast<uint16_t*>(row);
  for (int i = vec_end + gtid; i < cols; i += stride) {
    raw[i] = __ldg(lut + raw[i]);
  }
}

template <bool Aligned>
__global__ void __launch_bounds__(kBlock)
softcap_argmax_rows_strided_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const uint16_t* __restrict__ lut,
    int* __restrict__ out_argmax_i32,
    const void* __restrict__ draft_ids,
    int draft_ids_64,
    int64_t* __restrict__ sampled_out,
    int* __restrict__ num_sampled_out,
    int* __restrict__ num_rejected_out,
    float* __restrict__ scratch_val,
    int* __restrict__ scratch_idx,
    int* __restrict__ done_cnt,
    int rows,
    int n,
    int row_stride) {
  __shared__ float sv[kWarps];
  __shared__ int si[kWarps];
  __shared__ int am_last;

  const int r = blockIdx.y;
  const __nv_bfloat16* row_logits =
      logits + static_cast<size_t>(r) * static_cast<size_t>(row_stride);
  float* r_sv = scratch_val + r * kArgmaxGrid;
  int* r_si = scratch_idx + r * kArgmaxGrid;
  int* r_dc = done_cnt + r;

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
    const uint4* vec = reinterpret_cast<const uint4*>(row_logits);
    for (int i = gtid; i < n_vec; i += stride) {
      uint4 v = vec[i];
      const uint16_t* p = reinterpret_cast<const uint16_t*>(&v);
      const int base_idx = i << 3;
#pragma unroll
      for (int k = 0; k < 8; ++k) pick_better(bv, bi, lut_to_float(lut, p[k]), base_idx + k);
    }
  }
  const uint16_t* raw = reinterpret_cast<const uint16_t*>(row_logits);
  for (int i = vec_end + gtid; i < n; i += stride) {
    pick_better(bv, bi, lut_to_float(lut, raw[i]), i);
  }

  block_reduce_max(bv, bi, sv, si, warp, lane);
  if (tid == 0) {
    r_sv[blockIdx.x] = bv;
    r_si[blockIdx.x] = bi;
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
    am_last = (atomicAdd(r_dc, 1) == static_cast<int>(gridDim.x) - 1) ? 1 : 0;
  }
  __syncthreads();
  if (am_last) {
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
    bv = (tid < static_cast<int>(gridDim.x)) ? ((const volatile float*)r_sv)[tid] : -INFINITY;
    bi = (tid < static_cast<int>(gridDim.x)) ? ((const volatile int*)r_si)[tid] : 0x7fffffff;
    block_reduce_max(bv, bi, sv, si, warp, lane);
    if (tid == 0) {
      *r_dc = 0;
      out_argmax_i32[r] = bi;
      if (sampled_out != nullptr) {
        asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
        int* rows_dc = done_cnt + rows;
        if (atomicAdd(rows_dc, 1) == rows - 1) {
          asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
          *rows_dc = 0;
          int accepted_len = 0;
          bool accepted = true;
          const volatile int* row_am = static_cast<const volatile int*>(out_argmax_i32);
          for (int i = 0; i < rows - 1; ++i) {
            const int64_t t_tok = static_cast<int64_t>(row_am[i]);
            const int64_t d_tok = draft_ids_64
                                      ? static_cast<const int64_t*>(draft_ids)[i + 1]
                                      : static_cast<int64_t>(static_cast<const int32_t*>(draft_ids)[i + 1]);
            if (accepted) {
              if (t_tok == d_tok) {
                sampled_out[i] = d_tok;
                accepted_len += 1;
              } else {
                sampled_out[i] = t_tok;
                accepted = false;
              }
            }
          }
          if (accepted) {
            sampled_out[rows - 1] = static_cast<int64_t>(row_am[rows - 1]);
          }
          const int n_samp = accepted_len + 1;
          *num_sampled_out = n_samp;
          *num_rejected_out = rows - n_samp;
        }
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
  rc = cudaFuncGetAttributes(&attr, lmpost::softcap_inplace_strided_kernel<true>);
  if (rc != cudaSuccess) return (int)rc;
  rc = cudaFuncGetAttributes(&attr, lmpost::softcap_inplace_strided_kernel<false>);
  if (rc != cudaSuccess) return (int)rc;
  rc = cudaFuncGetAttributes(&attr, lmpost::softcap_argmax_kernel<true>);
  if (rc != cudaSuccess) return (int)rc;
  rc = cudaFuncGetAttributes(&attr, lmpost::softcap_argmax_kernel<false>);
  if (rc != cudaSuccess) return (int)rc;
  rc = cudaFuncGetAttributes(&attr, lmpost::softcap_argmax_rows_strided_kernel<true>);
  if (rc != cudaSuccess) return (int)rc;
  return (int)cudaFuncGetAttributes(&attr, lmpost::softcap_argmax_rows_strided_kernel<false>);
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

extern "C" int lmpost_softcap_inplace_strided(
    void* logits, const void* lut, int rows, int cols, int row_stride, void* stream) {
  using namespace lmpost;
  if (rows <= 0 || cols <= 0 || row_stride < cols) return (int)cudaErrorInvalidValue;
  cudaStream_t st = (cudaStream_t)stream;
  const bool aligned = (((uintptr_t)logits & 15) == 0) && ((row_stride & 7) == 0) && (cols >= 8);
  const int n_units = aligned ? (cols >> 3) : cols;
  int grid_x = (n_units + kBlock - 1) / kBlock;
  if (grid_x < 1) grid_x = 1;
  if (grid_x > 256) grid_x = 256;
  dim3 grid(grid_x, rows);
  if (aligned) {
    softcap_inplace_strided_kernel<true><<<grid, kBlock, 0, st>>>(
        (__nv_bfloat16*)logits, (const uint16_t*)lut, cols, row_stride);
  } else {
    softcap_inplace_strided_kernel<false><<<grid, kBlock, 0, st>>>(
        (__nv_bfloat16*)logits, (const uint16_t*)lut, cols, row_stride);
  }
  return (int)cudaGetLastError();
}

extern "C" int lmpost_argmax_rows_strided(
    const void* logits, const void* lut, void* out_argmax_i32,
    void* scratch_val, void* scratch_idx, void* done_cnt,
    int rows, int vocab_size, int row_stride, void* stream) {
  using namespace lmpost;
  if (rows <= 0 || rows > 16 || vocab_size <= 0 || row_stride < vocab_size) {
    return (int)cudaErrorInvalidValue;
  }
  cudaStream_t st = (cudaStream_t)stream;
  dim3 grid(kArgmaxGrid, rows);
  const bool aligned = (((uintptr_t)logits & 15) == 0) && ((row_stride & 7) == 0) && (vocab_size >= 8);
  if (aligned) {
    softcap_argmax_rows_strided_kernel<true><<<grid, kBlock, 0, st>>>(
        (const __nv_bfloat16*)logits, (const uint16_t*)lut, (int*)out_argmax_i32,
        nullptr, 0, nullptr, nullptr, nullptr,
        (float*)scratch_val, (int*)scratch_idx, (int*)done_cnt,
        rows, vocab_size, row_stride);
  } else {
    softcap_argmax_rows_strided_kernel<false><<<grid, kBlock, 0, st>>>(
        (const __nv_bfloat16*)logits, (const uint16_t*)lut, (int*)out_argmax_i32,
        nullptr, 0, nullptr, nullptr, nullptr,
        (float*)scratch_val, (int*)scratch_idx, (int*)done_cnt,
        rows, vocab_size, row_stride);
  }
  return (int)cudaGetLastError();
}

extern "C" int lmpost_greedy_verify_strided(
    const void* logits, const void* lut, const void* draft_ids, int draft_ids_64,
    void* out_argmax_i32, void* sampled_out, void* num_sampled_out, void* num_rejected_out,
    void* scratch_val, void* scratch_idx, void* done_cnt,
    int rows, int vocab_size, int row_stride, void* stream) {
  using namespace lmpost;
  if (rows < 2 || rows > 16 || vocab_size <= 0 || row_stride < vocab_size) {
    return (int)cudaErrorInvalidValue;
  }
  cudaStream_t st = (cudaStream_t)stream;
  dim3 grid(kArgmaxGrid, rows);
  const bool aligned = (((uintptr_t)logits & 15) == 0) && ((row_stride & 7) == 0) && (vocab_size >= 8);
  if (aligned) {
    softcap_argmax_rows_strided_kernel<true><<<grid, kBlock, 0, st>>>(
        (const __nv_bfloat16*)logits, (const uint16_t*)lut, (int*)out_argmax_i32,
        draft_ids, draft_ids_64, (int64_t*)sampled_out, (int*)num_sampled_out, (int*)num_rejected_out,
        (float*)scratch_val, (int*)scratch_idx, (int*)done_cnt,
        rows, vocab_size, row_stride);
  } else {
    softcap_argmax_rows_strided_kernel<false><<<grid, kBlock, 0, st>>>(
        (const __nv_bfloat16*)logits, (const uint16_t*)lut, (int*)out_argmax_i32,
        draft_ids, draft_ids_64, (int64_t*)sampled_out, (int*)num_sampled_out, (int*)num_rejected_out,
        (float*)scratch_val, (int*)scratch_idx, (int*)done_cnt,
        rows, vocab_size, row_stride);
  }
  return (int)cudaGetLastError();
}
