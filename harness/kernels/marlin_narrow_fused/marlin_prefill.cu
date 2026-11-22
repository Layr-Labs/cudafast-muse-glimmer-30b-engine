// SPDX-License-Identifier: Apache-2.0
// Narrow gate/up GEMM with the stock BF16 SiLU-and-mul epilogue.
#include <cuda_runtime.h>

#define MARLIN_NAMESPACE_NAME gb10_narrow_fused
#include "silu_mul.cuh"
#include "marlin/marlin_template.h"

namespace marlinw {

// The parent package's standalone activation, for rows outside the fused
// geometry. Each vector handles eight BF16 values in a contiguous [gate|up]
// row; the output is written once after the same FP32 activation/multiply.
__global__ void SiluMul(const int4* __restrict__ in, int4* __restrict__ out,
                        int m, int d8, int ld_in8) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(m) * d8) return;
  int r = static_cast<int>(i / d8), c = static_cast<int>(i % d8);
  int4 g = in[static_cast<int64_t>(r) * ld_in8 + c];
  int4 u = in[static_cast<int64_t>(r) * ld_in8 + d8 + c];
  int4 o;
  o.x = silu_mul_bf16x2(g.x, u.x);
  o.y = silu_mul_bf16x2(g.y, u.y);
  o.z = silu_mul_bf16x2(g.z, u.z);
  o.w = silu_mul_bf16x2(g.w, u.w);
  out[i] = o;
}

}  // namespace marlinw

extern "C" int marlin_narrow_fused_silu(
    const int4* input, int4* output, int rows, int columns,
    int input_row_stride, cudaStream_t stream) {
  if (rows < 0 || columns <= 0 || columns % 8 || input_row_stride % 8 ||
      input_row_stride < 2 * columns)
    return -2;
  if (rows == 0) return cudaSuccess;
  int64_t n = static_cast<int64_t>(rows) * (columns / 8);
  constexpr int threads = 256;
  marlinw::SiluMul<<<(n + threads - 1) / threads, threads, 0, stream>>>(
      input, output, rows, columns / 8, input_row_stride / 8);
  return static_cast<int>(cudaGetLastError());
}

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
