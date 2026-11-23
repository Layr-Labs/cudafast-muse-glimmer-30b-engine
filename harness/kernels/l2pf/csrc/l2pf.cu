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
#include <cstdlib>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <sys/prctl.h>
#include <time.h>
#include <atomic>
#endif

namespace l2pf {

constexpr uint32_t kMaxPiece = 1u << 16;

__device__ __forceinline__ uint64_t policy_evict_last() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

template <bool EvictLast>
__device__ __forceinline__ void bulk_prefetch(const uint8_t* p, uint32_t n, uint64_t pol) {
  if constexpr (EvictLast) {
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;\n" ::"l"(p), "r"(n), "l"(pol)
                 : "memory");
  } else {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(n)
                 : "memory");
  }
}

template <bool EvictLast>
__global__ void prefetch_kernel(const uint8_t* __restrict__ p, size_t bytes, uint32_t chunk) {
  const uint64_t pol = EvictLast ? policy_evict_last() : 0ull;
  const size_t nthr = (size_t)gridDim.x * blockDim.x;
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t off = tid * chunk; off < bytes; off += nthr * chunk) {
    const size_t n = bytes - off < chunk ? bytes - off : chunk;
    const uint32_t m = (uint32_t)(n & ~(size_t)15);
    if (m) bulk_prefetch<EvictLast>(p + off, m, pol);
  }
}

// Descriptor j is three 64-bit words: base address, pitch in bytes, and
// (segment bytes << 32 | segment count). It names `count` segments of `seg`
// bytes at base + r * pitch. Segments are dealt round-robin over all threads.
template <bool EvictLast>
__global__ void prefetch_desc_kernel(const unsigned long long* __restrict__ d, int nd) {
  const uint64_t pol = EvictLast ? policy_evict_last() : 0ull;
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
    for (; r < cnt; r += nthr) bulk_prefetch<EvictLast>((const uint8_t*)(base + r * pitch), seg, pol);
    first = (first + cnt) % nthr;
  }
}

__global__ void stream_read_kernel(const uint4* __restrict__ p, size_t n, unsigned* sink) {
  uint64_t pol;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  unsigned acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    uint4 v;
    asm volatile(
        "ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p + i), "l"(pol));
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u && sink) *sink = acc;
}

inline bool use_evict_last() {
  static const bool enabled = []() {
    const char* v = std::getenv("L2PF_EVICT_LAST");
    return !v || v[0] != '0' || v[1] != '\0';
  }();
  return enabled;
}

}  // namespace l2pf

extern "C" {

// 0 on success; -1 for a misaligned range or chunk; otherwise a CUDA error code.
int l2pf_prefetch(const void* p, size_t bytes, int chunk, int blocks, void* stream) {
  if (reinterpret_cast<uintptr_t>(p) % 16 || chunk < 16 || chunk % 16 ||
      (uint32_t)chunk > l2pf::kMaxPiece || blocks < 1)
    return -1;
  auto st = reinterpret_cast<cudaStream_t>(stream);
  if (l2pf::use_evict_last()) {
    l2pf::prefetch_kernel<true><<<blocks, 128, 0, st>>>((const uint8_t*)p, bytes, (uint32_t)chunk);
  } else {
    l2pf::prefetch_kernel<false><<<blocks, 128, 0, st>>>((const uint8_t*)p, bytes, (uint32_t)chunk);
  }
  return (int)cudaGetLastError();
}

// The descriptors are checked on the host when the table is built (alignment,
// segment size <= 64 KB, every segment inside one weight tensor); here only the
// launch arguments are.
int l2pf_prefetch_desc(const void* desc, int ndesc, int blocks, int threads, void* stream) {
  if (!desc || ndesc < 1 || blocks < 1 || threads < 32 || threads > 1024 ||
      reinterpret_cast<uintptr_t>(desc) % 8)
    return -1;
  auto st = reinterpret_cast<cudaStream_t>(stream);
  if (l2pf::use_evict_last()) {
    l2pf::prefetch_desc_kernel<true><<<blocks, threads, 0, st>>>(
        (const unsigned long long*)desc, ndesc);
  } else {
    l2pf::prefetch_desc_kernel<false><<<blocks, threads, 0, st>>>(
        (const unsigned long long*)desc, ndesc);
  }
  return (int)cudaGetLastError();
}

// Pure DRAM streaming read probe; never modifies p.
// 0 on success; -1 for invalid arguments or misaligned pointer; otherwise a CUDA error code.
int l2pf_stream_read(const void* p, size_t bytes, unsigned* sink, int blocks, void* stream) {
  if (!p || !sink || reinterpret_cast<uintptr_t>(p) % 16 || bytes < 16 || blocks < 1)
    return -1;
  l2pf::stream_read_kernel<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      (const uint4*)p, bytes >> 4, sink);
  return (int)cudaGetLastError();
}

#ifdef __linux__
static std::atomic<bool> g_tickle_started{false};

// Micro-attribution: The pure-C SCHED_IDLE 1 ms nanosleep tickler to prevent ARM LPI-3 deep-idle
// entry on pinned high-capacity cores is inspired by @zereraz (Yukon submission 77795739).
static void* l2pf_tickle_worker(void* arg) {
  int cpu = static_cast<int>(reinterpret_cast<intptr_t>(arg));
  if (cpu >= 0 && cpu < CPU_SETSIZE) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    (void)sched_setaffinity(0, sizeof(set), &set);
  }
  (void)prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
  struct sched_param sp{};
  sp.sched_priority = 0;
  (void)sched_setscheduler(0, SCHED_IDLE, &sp);
  struct timespec req{};
  req.tv_sec = 0;
  req.tv_nsec = 1000000L;  // 1 ms
  for (;;) {
    (void)nanosleep(&req, nullptr);
  }
  return nullptr;
}

int l2pf_start_idle_tickle(const int* cpus, int count) {
  if (!cpus || count <= 0 || count > 64) return 0;
  bool expected = false;
  if (!g_tickle_started.compare_exchange_strong(expected, true)) return 0;
  int started = 0;
  for (int i = 0; i < count; ++i) {
    pthread_t th;
    if (pthread_create(&th, nullptr, l2pf_tickle_worker,
                       reinterpret_cast<void*>(static_cast<intptr_t>(cpus[i]))) == 0) {
      pthread_detach(th);
      ++started;
    }
  }
  return started;
}
#else
int l2pf_start_idle_tickle(const int*, int) { return 0; }
#endif

}  // extern "C"
