// SPDX-License-Identifier: Apache-2.0
// The headers in marlin/ are from vLLM v0.27.1, commit 6e448d0ea9bf.
// marlin_template.h reorders the complete output tiles in init_part1_slice.
#include <cuda_runtime.h>

#define MARLIN_NAMESPACE_NAME gb10_prefill
#include "marlin/marlin_template.h"

extern "C" int marlin_prefill(
    const int4* input, const int4* weight, int4* output,
    const int4* scales, const float* global_scale, int4* partial, int* locks,
    cudaStream_t stream, int rows, int columns, int inner, int sms,
    int shared_bytes) {
  using namespace vllm;
  auto launch = [&](auto kernel, int blocks, int threads, int shared) {
    auto status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (status != cudaSuccess) return status;
    kernel<<<blocks, threads, shared, stream>>>(
        input, weight, output, partial, nullptr, nullptr, scales, global_scale,
        nullptr, nullptr, inner / 16, rows, columns, inner, inner, locks,
        false, false, true, shared);
    return cudaGetLastError();
  };
  const int stock_tiles = (rows / 64) * (columns / 256);
  if (columns > inner && stock_tiles % sms == 0) {
    // Marlin uses a whole K reduction for every stock tile in this case.
    // Halving N and the warp count preserves both K accumulation groups.
    constexpr int shared = 3 * 64 * 64 * 2 + 64 * (128 + 8) * 2 + 3 * 4 * 128 * 2;
    return launch(gb10_prefill::Marlin<
                      kBFloat16.id(), kFE2M1f.id(), kBFloat16.id(), kFE4M3fn.id(),
                      128, 4, 8, 4, false, 3, 1, false, true>,
                  stock_tiles * 2, 128, shared);
  }
  return launch(gb10_prefill::Marlin<
                    kBFloat16.id(), kFE2M1f.id(), kBFloat16.id(), kFE4M3fn.id(),
                    256, 4, 16, 4, false, 4, 1, false>,
                sms, 256, shared_bytes);
}
