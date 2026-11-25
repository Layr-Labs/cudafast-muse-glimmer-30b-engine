// SPDX-License-Identifier: Apache-2.0
// lmpost: bit-exact BF16 logit scaling + softcapping via a 64K PyTorch-derived LUT,
// and fused 1-pass softcap+argmax for serial M=1 decode without logprobs.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdlib>
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

static bool lmpost_pdl_enabled() {
  static const bool on = []() {
    const char* env = std::getenv("MUSE_LMPOST_PDL");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  return on;
}

template <bool kPdl>
__global__ void __launch_bounds__(32) greedy_verify_kernel(
    const int64_t* __restrict__ target_argmax,
    const int32_t* __restrict__ input_ids,
    const int64_t* __restrict__ logits_indices,
    int32_t* __restrict__ num_sampled_out,
    int32_t* __restrict__ num_rejected_out,
    int m) {
  int64_t draft_toks[15];
  if constexpr (kPdl) {
    if (threadIdx.x == 0) {
      for (int i = 0; i < m - 1; ++i) {
        const int64_t idx = logits_indices ? logits_indices[i + 1] : static_cast<int64_t>(i + 1);
        draft_toks[i] = static_cast<int64_t>(input_ids[idx]);
      }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  if (threadIdx.x != 0) return;
  int accepted = 0;
  for (int i = 0; i < m - 1; ++i) {
    const int64_t draft_tok = kPdl
        ? draft_toks[i]
        : static_cast<int64_t>(input_ids[logits_indices ? logits_indices[i + 1] : static_cast<int64_t>(i + 1)]);
    if (target_argmax[i] == draft_tok) {
      ++accepted;
    } else {
      break;
    }
  }
  const int32_t ns = static_cast<int32_t>(accepted + 1);
  num_sampled_out[0] = ns;
  num_rejected_out[0] = static_cast<int32_t>(m) - ns;
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
  rc = cudaFuncGetAttributes(&attr, lmpost::greedy_verify_kernel<true>);
  if (rc != cudaSuccess) return (int)rc;
  return (int)cudaFuncGetAttributes(&attr, lmpost::greedy_verify_kernel<false>);
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

extern "C" int lmpost_greedy_verify(const void* target_argmax, const void* input_ids,
                                    const void* logits_indices, void* num_sampled_out,
                                    void* num_rejected_out, int m, void* stream) {
  using namespace lmpost;
  if (m < 2 || m > 16 || target_argmax == nullptr || input_ids == nullptr ||
      num_sampled_out == nullptr || num_rejected_out == nullptr) {
    return (int)cudaErrorInvalidValue;
  }
  cudaStream_t st = (cudaStream_t)stream;
  if (lmpost_pdl_enabled()) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1, 1, 1);
    cfg.blockDim = dim3(32, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return (int)cudaLaunchKernelEx(
        &cfg, greedy_verify_kernel<true>,
        (const int64_t*)target_argmax, (const int32_t*)input_ids,
        (const int64_t*)logits_indices, (int32_t*)num_sampled_out,
        (int32_t*)num_rejected_out, m);
  }
  greedy_verify_kernel<false><<<1, 32, 0, st>>>(
      (const int64_t*)target_argmax, (const int32_t*)input_ids,
      (const int64_t*)logits_indices, (int32_t*)num_sampled_out,
      (int32_t*)num_rejected_out, m);
  return (int)cudaGetLastError();
}
