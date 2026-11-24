// SPDX-License-Identifier: Apache-2.0
// The one-row 48-CTA stream-K decode GEMV of gemv6.cu (muse_marlin6_gemv, tiled layout) reading rANS-coded block
// scales for the 6656 x 19968 down projection (marlin_template.h, MUSE_SCALE_RANS + MUSE_SCALE_RANS_SK). Each CTA
// owns 169 consecutive k-stages of the tile-major stage order (52 column tiles x 156 k-stages = 48 x 169), which
// cross at most one column-slice boundary; its 8 warps x 32 lanes, its block stream and its scale-slot ring run
// through both slices by global stage index and decode each stage's 1,024 one-byte scales ahead into the shared
// slot the consumers read, so the arithmetic is untouched. The loader admits a layer only after a bitwise
// comparison with the six-bit kernel. Its own namespace keeps every symbol distinct.
// coded = [decode table, 4,096 B][48 CTA headers, 3,840 B each][stage blocks]; see encode_down_scales.
#define MARLIN_NAMESPACE_NAME muse_marlin6_srans_sk
#define MUSE_SCALE_RANS
#define MUSE_SCALE_RANS_SK
#include "marlin/marlin_template.h"

extern "C" int muse_marlin6_down_srans(const int4* input, const int4* weight, int4* output, int4* partials,
                                       const int4* coded, const int4* palette, const float* global_scale,
                                       int* locks, int n, int k, int blocks, int device, cudaStream_t stream) {
  int shared = 0;
  auto status = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (status != cudaSuccess) return status;
  if (n != 6656 || k != 19968 || blocks != 48) return cudaErrorInvalidValue;
  const auto* base = reinterpret_cast<const unsigned char*>(coded);
  const auto* lut = reinterpret_cast<const int*>(base);
  const auto* headers = reinterpret_cast<const int4*>(base + 4096);
  const auto* var = reinterpret_cast<const int4*>(base + 4096 + 48 * 3840);
  auto kernel = muse_marlin6_srans_sk::Marlin<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 4, 1, false, true, false>;
  status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  kernel<<<blocks, 256, shared, stream>>>(input, weight, output, partials, nullptr, nullptr, headers, global_scale,
                                          var, lut, k / 16, 1, n, k, k, locks, false, false, true, shared, palette, 0,
                                          nullptr, nullptr, nullptr, 0, nullptr);
  return cudaGetLastError();
}
