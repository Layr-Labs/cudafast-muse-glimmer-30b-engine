// SPDX-License-Identifier: Apache-2.0
// The fixed gate/up grid of gemv6.cu (muse_marlin6_gate_up, tiled layout) reading rANS-coded block scales
// (marlin_template.h, MUSE_SCALE_RANS): per CTA slice, 8 warps x 32 lanes decode the stage's 1,024 one-byte
// scales one stage ahead into the shared slot the uncompressed kernel fills by cp.async, and the consumers read
// raw scale bytes exactly as the uncompressed kernel does. The arithmetic is untouched; the loader admits a layer
// only after a bitwise comparison with the six-bit kernel. Its own namespace keeps every symbol distinct.
// coded = [decode table, 4,096 B][336 CTA headers, 3,328 B each][stage blocks]; see encode_gate_up_scales.
#define MARLIN_NAMESPACE_NAME muse_marlin6_srans
#define MUSE_SCALE_RANS
#include "marlin/marlin_template.h"

extern "C" int muse_marlin6_gate_up_srans(const int4* input, const int4* weight, int4* output,
                                          int4* partials, const int4* coded, const int4* palette,
                                          const float* global_scale, int* locks, int n, int k, int blocks,
                                          int device, cudaStream_t stream) {
  int shared = 0;
  auto status = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (status != cudaSuccess) return status;
  if (n != 39936 || k != 6656 || blocks != 48) return cudaErrorInvalidValue;
  const auto* base = reinterpret_cast<const unsigned char*>(coded);
  const auto* lut = reinterpret_cast<const int*>(base);
  const auto* headers = reinterpret_cast<const int4*>(base + 4096);
  const auto* var = reinterpret_cast<const int4*>(base + 4096 + 336 * 3328);
  auto kernel = muse_marlin6_srans::Marlin<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 4, 1, false, true, true>;
  status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  kernel<<<336, 256, shared, stream>>>(input, weight, output, partials, nullptr, nullptr, headers, global_scale,
                                       var, lut, k / 16, 1, n, k, k, locks, false, false, true, shared, palette);
  return cudaGetLastError();
}
