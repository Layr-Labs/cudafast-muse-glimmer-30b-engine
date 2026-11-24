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

__device__ __forceinline__ void bulk_prefetch_normal(const uint8_t* p, uint32_t n) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n"
               :: "l"(p), "r"(n) : "memory");
}

__device__ __forceinline__ void bulk_prefetch_evict_last(const uint8_t* p, uint32_t n) {
  uint64_t pol;
  asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;\n"
               :: "l"(p), "r"(n), "l"(pol) : "memory");
}

__global__ void prefetch_kernel(const uint8_t* __restrict__ p, size_t bytes, uint32_t chunk) {
  const size_t nthr = (size_t)gridDim.x * blockDim.x;
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t off = tid * chunk; off < bytes; off += nthr * chunk) {
    const size_t n = bytes - off < chunk ? bytes - off : chunk;
    const uint32_t m = (uint32_t)(n & ~(size_t)15);
    if (m) bulk_prefetch_normal(p + off, m);
  }
}

// Descriptor j is three 64-bit words: base address, pitch in bytes, and
// (raw_seg << 32 | segment count), where bit 31 of raw_seg requests L2::evict_last
// and bits 0..30 hold segment bytes. Segments are dealt round-robin over all threads.
__global__ void prefetch_desc_kernel(const unsigned long long* __restrict__ d, int nd) {
  const uint32_t nthr = gridDim.x * blockDim.x;
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t first = 0;  // index of the thread that takes row 0 of descriptor j
  for (int j = 0; j < nd; ++j) {
    const unsigned long long base = d[3 * j];
    const unsigned long long pitch = d[3 * j + 1];
    const unsigned long long sc = d[3 * j + 2];
    const uint32_t raw_seg = (uint32_t)(sc >> 32);
    const uint32_t seg = raw_seg & 0x7fffffffu;
    const bool evict_last = (raw_seg >> 31) != 0;
    const uint32_t cnt = (uint32_t)(sc & 0xffffffffull);
    uint32_t r = (tid + nthr - first) % nthr;
    if (evict_last) {
      for (; r < cnt; r += nthr) bulk_prefetch_evict_last((const uint8_t*)(base + r * pitch), seg);
    } else {
      for (; r < cnt; r += nthr) bulk_prefetch_normal((const uint8_t*)(base + r * pitch), seg);
    }
    first = (first + cnt) % nthr;
  }
}

// CTA-first spread variant: segment r goes to CTA r % gridDim.x so bulk prefetches
// spread across SMs instead of concentrating on CTA 0.
__global__ void prefetch_desc_spread_kernel(const unsigned long long* __restrict__ d, int nd) {
  const uint32_t nthr = gridDim.x * blockDim.x;
  const uint32_t tid = threadIdx.x * gridDim.x + blockIdx.x;
  uint32_t first = 0;
  for (int j = 0; j < nd; ++j) {
    const unsigned long long base = d[3 * j];
    const unsigned long long pitch = d[3 * j + 1];
    const unsigned long long sc = d[3 * j + 2];
    const uint32_t raw_seg = (uint32_t)(sc >> 32);
    const uint32_t seg = raw_seg & 0x7fffffffu;
    const bool evict_last = (raw_seg >> 31) != 0;
    const uint32_t cnt = (uint32_t)(sc & 0xffffffffull);
    uint32_t r = (tid + nthr - first) % nthr;
    if (evict_last) {
      for (; r < cnt; r += nthr) bulk_prefetch_evict_last((const uint8_t*)(base + r * pitch), seg);
    } else {
      for (; r < cnt; r += nthr) bulk_prefetch_normal((const uint8_t*)(base + r * pitch), seg);
    }
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
int l2pf_prefetch_desc_spread(const void* desc, int ndesc, int blocks, int threads, void* stream) {
  if (!desc || ndesc < 1 || blocks < 1 || threads < 32 || threads > 1024 ||
      reinterpret_cast<uintptr_t>(desc) % 8)
    return -1;
  l2pf::prefetch_desc_spread_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      (const unsigned long long*)desc, ndesc);
  return (int)cudaGetLastError();
}

int l2pf_prefetch_desc(const void* desc, int ndesc, int blocks, int threads, void* stream) {
  if (!desc || ndesc < 1 || blocks < 1 || threads < 32 || threads > 1024 ||
      reinterpret_cast<uintptr_t>(desc) % 8)
    return -1;
  l2pf::prefetch_desc_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      (const unsigned long long*)desc, ndesc);
  return (int)cudaGetLastError();
}

int l2pf_set_persisting_window(const void* base_ptr, size_t num_bytes, size_t limit_bytes, void* stream) {
  if (limit_bytes > 0) {
    cudaError_t e = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, limit_bytes);
    if (e != cudaSuccess) return (int)e;
  }
  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow.base_ptr = const_cast<void*>(base_ptr);
  attr.accessPolicyWindow.num_bytes = num_bytes;
  attr.accessPolicyWindow.hitRatio = (num_bytes > 0) ? 1.0f : 0.0f;
  attr.accessPolicyWindow.hitProp = (num_bytes > 0) ? cudaAccessPropertyPersisting : cudaAccessPropertyNormal;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cudaError_t se = cudaStreamSetAttribute(reinterpret_cast<cudaStream_t>(stream),
                                          cudaStreamAttributeAccessPolicyWindow, &attr);
  if (limit_bytes == 0) {
    (void)cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
  }
  return (int)se;
}

int l2pf_clear_stream_persisting_window(void* stream) {
  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow.base_ptr = nullptr;
  attr.accessPolicyWindow.num_bytes = 0;
  attr.accessPolicyWindow.hitRatio = 0.0f;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  cudaError_t se = cudaStreamSetAttribute(reinterpret_cast<cudaStream_t>(stream),
                                          cudaStreamAttributeAccessPolicyWindow, &attr);
  return (int)se;
}

int l2pf_set_persisting_window_ex(const void* base_ptr, size_t num_bytes, size_t limit_bytes,
                                  int miss_streaming, void* stream) {
  if (limit_bytes > 0) {
    cudaError_t e = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, limit_bytes);
    if (e != cudaSuccess) return (int)e;
  }
  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow.base_ptr = const_cast<void*>(base_ptr);
  attr.accessPolicyWindow.num_bytes = num_bytes;
  attr.accessPolicyWindow.hitRatio = (num_bytes > 0) ? 1.0f : 0.0f;
  attr.accessPolicyWindow.hitProp = (num_bytes > 0) ? cudaAccessPropertyPersisting : cudaAccessPropertyNormal;
  attr.accessPolicyWindow.missProp =
      (num_bytes > 0 && miss_streaming) ? cudaAccessPropertyStreaming : cudaAccessPropertyNormal;
  cudaError_t se = cudaStreamSetAttribute(reinterpret_cast<cudaStream_t>(stream),
                                          cudaStreamAttributeAccessPolicyWindow, &attr);
  return (int)se;
}

int l2pf_set_max_l2_fetch_granularity(int bytes) {
  if (bytes != 32 && bytes != 64 && bytes != 128) return -1;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
  cudaError_t e = cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, static_cast<size_t>(bytes));
  return (int)e;
#else
  return -1;
#endif
}

#ifdef __linux__
static std::atomic<bool> g_tickle_started{false};
static std::atomic<long> g_tickle_ns{1000000L};
static std::atomic<long> g_prefill_tickle_ns{-1L};

static long resolve_prefill_tickle_ns() {
  long v = g_prefill_tickle_ns.load(std::memory_order_relaxed);
  if (v > 0L) return v;
  long ns = 1000000L;  // 1 ms default in both prefill and decode to prevent ARM LPI-3 entry
  const char* env = std::getenv("L2PF_PREFILL_TICKLE_MS");
  if (env && *env) {
    char* end = nullptr;
    long ms = std::strtol(env, &end, 10);
    if (end != env && ms >= 1L) {
      if (ms > 999L) ms = 999L;
      ns = ms * 1000000L;
    }
  }
  g_prefill_tickle_ns.store(ns, std::memory_order_relaxed);
  return ns;
}

int l2pf_set_tickle_phase(int is_decode) {
  const long ns = is_decode ? 1000000L : resolve_prefill_tickle_ns();
  g_tickle_ns.store(ns, std::memory_order_relaxed);
  return 0;
}

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
  for (;;) {
    struct timespec req{};
    req.tv_sec = 0;
    req.tv_nsec = g_tickle_ns.load(std::memory_order_relaxed);
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
int l2pf_set_tickle_phase(int) { return 0; }
int l2pf_start_idle_tickle(const int*, int) { return 0; }
#endif

}  // extern "C"
