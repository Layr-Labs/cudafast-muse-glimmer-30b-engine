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
  auto kernel = gb10_prefill::Marlin<
      kBFloat16.id(), kFE2M1f.id(), kBFloat16.id(), kFE4M3fn.id(),
      256, 4, 16, 4, false, 4, 1, false>;
  auto status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  if (status != cudaSuccess) return status;
  kernel<<<sms, 256, shared_bytes, stream>>>(
      input, weight, output, partial, nullptr, nullptr, scales, global_scale,
      nullptr, nullptr, inner / 16, rows, columns, inner, inner, locks,
      false, false, true, shared_bytes);
  return cudaGetLastError();
}
