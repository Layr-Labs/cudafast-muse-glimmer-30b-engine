// SPDX-License-Identifier: Apache-2.0
// Narrow gate/up GEMM with the stock BF16 SiLU-and-mul epilogue.
#include <cuda_runtime.h>

#define MARLIN_NAMESPACE_NAME gb10_narrow_fused
#include "../marlinw/csrc/silu_mul.cuh"
#include "marlin/marlin_template.h"

extern "C" int marlin_narrow_fused(
    const int4* input, const int4* weight, int4* output,
    const int4* scales, const float* global_scale, int4* partial, int* locks,
    cudaStream_t stream, int rows, int columns, int inner) {
  constexpr int threads = 128;
  constexpr int tile_columns = 128;
  constexpr int gate_int4 = (2 * 8 + 1) * 16 * 4;
  constexpr int shared =
      (3 * 64 * 64 * 2 + 64 * (128 + 8) * 2 + 3 * 4 * 128 * 2) / 16;
  constexpr bool gate_shared = false;
  constexpr int total_shared = (shared + (gate_shared ? gate_int4 : 0)) * 16;
  auto kernel = gb10_narrow_fused::Marlin<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), threads, 4, 8, 4, false, 3, 1, false, true, true, gate_shared>;
  auto status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, total_shared);
  if (status != cudaSuccess) return status;
  // One CTA owns both halves of a gate/up tile. No inter-CTA activation
  // synchronization or activation-sized scratch is needed.
  kernel<<<(rows / 64) * (columns / (2 * tile_columns)), threads,
           total_shared, stream>>>(
      input, weight, output, partial, nullptr, nullptr, scales, global_scale,
      nullptr, nullptr, inner / 16, rows, columns, inner, inner, locks,
      false, false, true, total_shared);
  return cudaGetLastError();
}
