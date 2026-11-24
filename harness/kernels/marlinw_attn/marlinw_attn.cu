// SPDX-License-Identifier: Apache-2.0
// The narrow MarlinW kernel of marlinw_down (same namespace and template
// arguments, hence the same device code: 64x128 tiles, 128 threads, three
// stages, two blocks per SM) with the problem shape and the grid given at run
// time. marlinw_attn.py hands it the stock partition of a 512-row
// ops.marlin_gemm call of an NVFP4 attention projection: every slice keeps its
// chunk, K range, slice_idx and slice_count, so each output element gets the
// same K16 MMA chains, the same two-warp FP32 sum, the same FP32 split-K
// reduction order and the same BF16 epilogue as the stock call.
#define MARLIN_NAMESPACE_NAME marlinw_down
#include <cuda_runtime.h>
#include "marlinw_template.h"

namespace {
constexpr int kThreads = 128;
constexpr int kShared = 45056;
constexpr auto kKernel =
    marlinw_down::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                        vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                        kThreads, 4, 8, 4, false, 3, 1, false>;
}  // namespace

extern "C" int marlinw_attn_init(int* active_blocks) {
  auto status = cudaFuncSetAttribute(
      kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_blocks, kKernel, kThreads, kShared);
}

extern "C" int marlinw_attn_gemm(
    const int4* input, const int4* weight, int4* output, int4* partial,
    const int4* scales, const float* global_scale, int* locks,
    const int4* work, const int* work_ptr, int rows, int columns, int inner,
    int row_stride, int grid, cudaStream_t stream) {
  if (grid <= 0 || (rows != 512 && rows != 64) || columns % 128 || inner % 64 ||
      row_stride < inner || row_stride % 8)
    return cudaErrorInvalidValue;
  kKernel<<<grid, kThreads, kShared, stream>>>(
      input, weight, output, partial, nullptr, nullptr, scales, global_scale,
      nullptr, nullptr, inner / 16, rows, columns, inner, row_stride, locks,
      false, false, true, kShared, work, work_ptr);
  return cudaGetLastError();
}
