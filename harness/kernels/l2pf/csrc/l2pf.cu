// l2pf: queue bulk L2 prefetches of byte ranges and return.
//
// Launched on a side stream, the prefetches run while latency-bound kernels on
// the main stream leave DRAM idle, so a later bandwidth-bound kernel that reads
// the same bytes finds them in L2. The only memory instruction in these kernels
// is cp.async.bulk.prefetch.L2, which moves data from global memory into L2 and
// has no destination operand: they write nothing and change no value. Besides
// the prefetched ranges they read only their own descriptor table.
//
// A bulk prefetch of up to 64 KB is honoured; larger requests are cut into
// 64 KB pieces. Every range must be 16-byte aligned with a size that is a
// multiple of 16.
#include <cuda_runtime.h>
#include <cstdint>

namespace l2pf {

constexpr uint32_t kMaxPiece = 1u << 16;

__device__ __forceinline__ void bulk_prefetch(const uint8_t* p, uint32_t n) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(n)
               : "memory");
}

__global__ void prefetch_kernel(const uint8_t* __restrict__ p, size_t bytes, uint32_t chunk) {
  const size_t nthr = (size_t)gridDim.x * blockDim.x;
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t off = tid * chunk; off < bytes; off += nthr * chunk) {
    const size_t n = bytes - off < chunk ? bytes - off : chunk;
    const uint32_t m = (uint32_t)(n & ~(size_t)15);
    if (m) bulk_prefetch(p + off, m);
  }
}

// Descriptor j is three 64-bit words: base address, pitch in bytes, and
// (segment bytes << 32 | segment count). It names `count` segments of `seg`
// bytes at base + r * pitch. Segments are dealt round-robin over all threads.
__global__ void prefetch_desc_kernel(const unsigned long long* __restrict__ d, int nd) {
  const uint32_t nthr = gridDim.x * blockDim.x;
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t first = 0;  // index of the thread that takes row 0 of descriptor j
  for (int j = 0; j < nd; ++j) {
    const unsigned long long base = d[3 * j];
    const unsigned long long pitch = d[3 * j + 1];
    const unsigned long long sc = d[3 * j + 2];
    const uint32_t seg = (uint32_t)(sc >> 32);
    const uint32_t cnt = (uint32_t)(sc & 0xffffffffull);
    uint32_t r = (tid + nthr - first) % nthr;
    for (; r < cnt; r += nthr) bulk_prefetch((const uint8_t*)(base + r * pitch), seg);
    first = (first + cnt) % nthr;
  }
}

}  // namespace l2pf

extern "C" {

// 0 on success; -1 for a misaligned range or chunk; otherwise a CUDA error code.
int l2pf_prefetch(const void* p, size_t bytes, int chunk, int blocks, void* stream) {
  if (reinterpret_cast<uintptr_t>(p) % 16 || chunk < 16 || chunk % 16 ||
      (uint32_t)chunk > l2pf::kMaxPiece || blocks < 1)
    return -1;
  l2pf::prefetch_kernel<<<blocks, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      (const uint8_t*)p, bytes, (uint32_t)chunk);
  return (int)cudaGetLastError();
}

// The descriptors are checked on the host when the table is built (alignment,
// segment size <= 64 KB, every segment inside one weight tensor); here only the
// launch arguments are.
int l2pf_prefetch_desc(const void* desc, int ndesc, int blocks, int threads, void* stream) {
  if (!desc || ndesc < 1 || blocks < 1 || threads < 32 || threads > 1024 ||
      reinterpret_cast<uintptr_t>(desc) % 8)
    return -1;
  l2pf::prefetch_desc_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      (const unsigned long long*)desc, ndesc);
  return (int)cudaGetLastError();
}

}  // extern "C"
