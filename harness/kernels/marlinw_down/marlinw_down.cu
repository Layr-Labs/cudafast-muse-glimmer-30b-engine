// SPDX-License-Identifier: Apache-2.0
// The MarlinW work list keeps every stock split and places both slices of a
// narrow output tile in one block. Its block never waits for another block.
#define MARLIN_NAMESPACE_NAME marlinw_down
#include <cuda_runtime.h>
#include "marlinw_template.h"

namespace {
constexpr int kRows = 512;
constexpr int kColumns = 6656;
constexpr int kInner = 19968;
constexpr int kBlocks = (kRows / 64) * (kColumns / 128);
constexpr int kThreads = 128;
constexpr int kShared = 45056;
constexpr auto kKernel =
    marlinw_down::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                        vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                        kThreads, 4, 8, 4, false, 3, 1, false>;
}  // namespace

extern "C" int marlinw_down_init(int* active_blocks) {
  auto status = cudaFuncSetAttribute(
      kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_blocks, kKernel, kThreads, kShared);
}

extern "C" int marlinw_down_gemm(
    const int4* input, const int4* weight, int4* output, int4* partial,
    const int4* scales, const float* global_scale, int* locks,
    const int4* work, const int* work_ptr, int row_stride, cudaStream_t stream) {
  kKernel<<<kBlocks, kThreads, kShared, stream>>>(
      input, weight, output, partial, nullptr, nullptr, scales, global_scale,
      nullptr, nullptr, kInner / 16, kRows, kColumns, kInner, row_stride, locks,
      false, false, true, kShared, work, work_ptr);
  return cudaGetLastError();
}
