// marlint: the Marlin W4A16 NVFP4 kernel of ops.marlin_gemm for launches of
// 1 to 8 rows (bf16 activations, fe2m1f weights, fe4m3fn group-16 scales,
// fp32 global scale, fp32 global reduce), reading the weight codes and scales
// from a tile-major copy of the Marlin-prepared tensors.
//
// The kernel body is marlin_template.h of vLLM v0.27.1 (marlint_template.h).
// With MARLINT_TILE_MAJOR=1 the only change is the global address of each
// B and scale load (row stride of one column tile instead of the full row,
// tile offset of prob_k / 16 rows); partition, loads per stage, shared-memory
// layout, dequantization, mma order, reductions and epilogue are the stock
// code, instantiated with the stock template arguments. Plain C ABI for
// ctypes; no torch linkage.

#ifndef MARLINT_TILE_MAJOR
  #define MARLINT_TILE_MAJOR 1
#endif
#ifndef MARLINT_NS
  #define MARLINT_NS marlint
#endif
#define MARLIN_NAMESPACE_NAME MARLINT_NS

#include <cuda_runtime.h>

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"
#include "marlint_template.h"

namespace MARLINT_NS {

// The configurations marlin_mm selects for a launch of 1..8 rows
// (thread_m_blocks 1, m_block_size_8, 4 stages, group_blocks 1):
// cfg 0: thread_k 128, thread_n 128, 256 threads;
// cfg 1: thread_k 128, thread_n 64, 128 threads.
constexpr auto kKernel0 =
    Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
           vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 4, 1, false>;
constexpr auto kKernel1 =
    Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
           vllm::kFE4M3fn.id(), 128, 1, 4, 8, true, 4, 1, false>;

}  // namespace MARLINT_NS

#define MARLINT_CAT2(a, b) a##b
#define MARLINT_CAT(a, b) MARLINT_CAT2(a, b)
#define MARLINT_FN(name) MARLINT_CAT(MARLINT_NS, name)

extern "C" {

int MARLINT_FN(_tile_major)() { return MARLINT_TILE_MAJOR; }

// Sets the dynamic shared memory limit of both kernels on the current device.
// Must run outside CUDA graph capture. Returns 0 or a cudaError_t.
int MARLINT_FN(_init)(int smem) {
  cudaError_t e = cudaFuncSetAttribute(
      MARLINT_NS::kKernel0, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return static_cast<int>(e);
  e = cudaFuncSetAttribute(MARLINT_NS::kKernel1,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  return static_cast<int>(e);
}

int MARLINT_FN(_regs)(int cfg) {
  cudaFuncAttributes attr;
  cudaError_t e = cfg == 0 ? cudaFuncGetAttributes(&attr, MARLINT_NS::kKernel0)
                           : cudaFuncGetAttributes(&attr, MARLINT_NS::kKernel1);
  return e == cudaSuccess ? attr.numRegs : -1;
}

// One launch exactly as marlin_mm issues it for prob_m <= 8 rows:
// A: bf16 [M, lda]; B: fe2m1f codes, tile-major (MARLINT_TILE_MAJOR) or
// [K/16, 2N] int32; C: bf16 [M, N]; C_tmp: fp32 scratch of
// sms * 64 * 256 floats; scales: fe4m3fn, same order as B; global_scale:
// fp32 [1]; locks: the Marlin workspace (int32, zero on entry, left zero).
// grid = number of SMs, smem = the device's opt-in shared memory per block.
int MARLINT_FN(_gemm)(int cfg, const void* A, const void* B, void* C,
                      void* C_tmp, const void* scales, const void* global_scale,
                      int prob_m, int prob_n, int prob_k, int lda, void* locks,
                      int grid, int smem, void* stream) {
  const int tn = cfg == 0 ? 128 : 64;
  if (prob_m <= 0 || prob_m > 8 || prob_n % tn || prob_k % 128 || lda % 8 ||
      grid <= 0 || (cfg != 0 && cfg != 1))
    return -2;
  auto kernel = cfg == 0 ? MARLINT_NS::kKernel0 : MARLINT_NS::kKernel1;
  const int threads = cfg == 0 ? 256 : 128;
  kernel<<<grid, threads, smem, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const int4*>(A), static_cast<const int4*>(B),
      static_cast<int4*>(C), static_cast<int4*>(C_tmp), nullptr, nullptr,
      static_cast<const int4*>(scales),
      static_cast<const float*>(global_scale), nullptr, nullptr,
      prob_k / 16, prob_m, prob_n, prob_k, lda, static_cast<int*>(locks),
      false, false, true, smem);
  return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
