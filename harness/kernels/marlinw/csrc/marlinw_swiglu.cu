// SPDX-License-Identifier: Apache-2.0
// A complete-K 64x128 Marlin tile produces 64 paired gate/up columns and
// applies the parent projection's BF16 conversion before the activation.
#define MARLINW_SWIGLU
#define MARLIN_NAMESPACE_NAME marlinw_swiglu
#include <cuda_runtime.h>
#include "marlinw_template.h"

namespace {
using namespace vllm;
constexpr auto kernel =
    marlinw_swiglu::Marlin<kBFloat16.id(), kFE2M1f.id(), kBFloat16.id(),
                           kFE4M3fn.id(), 128, 4, 8, 4, false, 3, 1, false>;
constexpr int shared = 3 * 64 * 64 * 2 + 64 * (128 + 8) * 2 + 3 * 4 * 128 * 2;
}

extern "C" int swiglu_prepare() {
  auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  swiglu_initialize<<<256, 256>>>();
  return cudaDeviceSynchronize();
}

extern "C" int swiglu_gemm(const void* x, const void* w, const void* scales,
                           const void* global, void* out, int stride, cudaStream_t stream) {
  constexpr int m = 512, n = 39936, k = 6656;
  if (stride < k || stride % 8) return cudaErrorInvalidValue;
#ifdef MARLINW_SINGLE_SLAB
  constexpr int rows = 64;
#else
  constexpr int rows = m;
#endif
  kernel<<<(rows / 64) * (n / 128), 128, shared, stream>>>(
      static_cast<const int4*>(x), static_cast<const int4*>(w),
      static_cast<int4*>(out), nullptr, nullptr, nullptr, static_cast<const int4*>(scales),
      static_cast<const float*>(global), nullptr, nullptr, k / 16, m, n, k, stride,
      nullptr, false, false, true, shared, nullptr, nullptr);
  return cudaGetLastError();
}
