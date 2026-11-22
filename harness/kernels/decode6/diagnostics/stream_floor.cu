// SPDX-License-Identifier: Apache-2.0
// Standalone bandwidth diagnostic, not linked into the inference engine.
#include <cuda_runtime.h>
#include <cuda.h>

namespace {

__device__ __forceinline__ uint4 load(const uint4* src) {
  uint4 v;
  asm volatile("ld.global.cg.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(src));
  return v;
}

template<int lanes>
__global__ void stream(const uint4* __restrict__ payload, unsigned* __restrict__ sums,
                       size_t chunk) {
  // Contiguous stripe per CTA. Four independent accumulators let the
  // requests overlap, while the checksum makes every loaded word observable.
  const uint4* begin = payload + size_t(blockIdx.x) * chunk;
  unsigned sx = 0, sy = 0, sz = 0, sw = 0;
  for (size_t i = threadIdx.x; i < chunk; i += lanes) {
    const auto v = load(begin + i);
    sx ^= v.x;
    sy ^= v.y;
    sz ^= v.z;
    sw ^= v.w;
  }
  sums[size_t(blockIdx.x) * lanes + threadIdx.x] = sx ^ sy ^ sz ^ sw;
}

}  // namespace

extern "C" int stream_floor(const void* payload, void* sums, size_t bytes, int blocks,
                            int lanes, cudaStream_t on_stream) {
  if (blocks < 1 || bytes == 0 || bytes % (sizeof(uint4) * blocks) ||
      reinterpret_cast<size_t>(payload) % alignof(uint4))
    return cudaErrorInvalidValue;
  const size_t chunk = bytes / (sizeof(uint4) * blocks);
  const auto* src = static_cast<const uint4*>(payload);
  auto* dst = static_cast<unsigned*>(sums);
  switch (lanes) {
    case 128: stream<128><<<blocks, 128, 0, on_stream>>>(src, dst, chunk); break;
    case 256: stream<256><<<blocks, 256, 0, on_stream>>>(src, dst, chunk); break;
    case 512: stream<512><<<blocks, 512, 0, on_stream>>>(src, dst, chunk); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
