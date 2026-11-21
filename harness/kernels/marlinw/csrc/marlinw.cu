// marlinw: the Marlin W4A16 NVFP4 kernel (bf16 activations, fe2m1f weights,
// fe4m3fn group-16 scales, fp32 global scale, fp32 global reduce) with a host
// supplied work list instead of the in-kernel striped partition. Each work item
// is one slice of the partition that ops.marlin_gemm itself would use for the
// same rows, so every output element goes through the same instructions in the
// same order; only which block runs a slice, and when, changes.
//
// The kernel body is marlin_template.h of vLLM v0.27.1 with the partition code
// replaced (marlinw_template.h). Plain C ABI for ctypes; no torch linkage.

#define MARLIN_NAMESPACE_NAME marlinw

#include <cuda_runtime.h>

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"
#include "marlinw_template.h"

namespace marlinw {

// thread_m_blocks 4 (64-row chunks), thread_n 256, thread_k 64, 256 threads,
// 4 stages, group_blocks 1: the configuration ops.marlin_gemm selects for a
// launch of a multiple of 64 rows when {64, 256, 256} is valid for (N, K).
constexpr auto kKernel =
    Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
           vllm::kFE4M3fn.id(), 256, 4, 16, 4, false, 4, 1, false>;
constexpr int kThreads = 256;

}  // namespace marlinw

extern "C" {

// Sets the dynamic shared memory limit of the kernel on the current device.
// Must run outside CUDA graph capture. Returns 0 or a cudaError_t.
int marlinw_init(int smem) {
  cudaError_t e = cudaFuncSetAttribute(
      marlinw::kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  return static_cast<int>(e);
}

int marlinw_regs() {
  cudaFuncAttributes attr;
  if (cudaFuncGetAttributes(&attr, marlinw::kKernel) != cudaSuccess) return -1;
  return attr.numRegs;
}

// A: bf16 [M, lda] (row stride lda elements), B: Marlin-packed fe2m1f
// [K/16, 2N] int32, C: bf16 [M, N], C_tmp: fp32 scratch of n_slots * 64 * 256,
// scales: fe4m3fn [K/16, N] (S0E5M3, Marlin order), global_scale: fp32 [1],
// locks: int32 [n_slots], zero on entry and left zero on exit.
// work / work_ptr: see marlinw_template.h. grid = number of blocks.
int marlinw_gemm(const void* A, const void* B, void* C, void* C_tmp,
                 const void* scales, const void* global_scale, int prob_m,
                 int prob_n, int prob_k, int lda, void* locks,
                 const void* work, const void* work_ptr, int grid, int smem,
                 void* stream) {
  if (prob_m <= 0 || prob_m % 64 || prob_n % 256 || prob_k % 64 || lda % 8 ||
      grid <= 0)
    return -2;
  marlinw::kKernel<<<grid, marlinw::kThreads, smem,
                     static_cast<cudaStream_t>(stream)>>>(
      static_cast<const int4*>(A), static_cast<const int4*>(B),
      static_cast<int4*>(C), static_cast<int4*>(C_tmp), nullptr, nullptr,
      static_cast<const int4*>(scales),
      static_cast<const float*>(global_scale), nullptr, nullptr,
      prob_k / 16, prob_m, prob_n, prob_k, lda, static_cast<int*>(locks),
      false, false, true, smem, static_cast<const int4*>(work),
      static_cast<const int*>(work_ptr));
  return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
