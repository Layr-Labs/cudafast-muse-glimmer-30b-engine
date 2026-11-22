// marlinw: the Marlin W4A16 NVFP4 kernel (bf16 activations, fe2m1f weights,
// fe4m3fn group-16 scales, fp32 global scale, fp32 global reduce) with a host
// supplied work list instead of the in-kernel striped partition. Each work item
// is one slice of the partition that ops.marlin_gemm itself would use for the
// same rows, so every output element goes through the same instructions in the
// same order; only which block runs a slice, and when, changes.
//
// The kernel body is marlin_template.h of vLLM v0.27.1 with the partition code
// replaced (marlinw_template.h). Plain C ABI for ctypes; no torch linkage.

#define MARLIN_NAMESPACE_NAME marlinw

#include <cuda_runtime.h>

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"
#include "silu_mul.cuh"
#include "marlinw_template.h"
#include "marlinw_sched.h"

// MarlinS configurations (marlinw_sched.h) built into the library, by index:
// X(index, threads, thread_n_blocks, warp_j, stages, min_blocks, dq_ahead).
// 0: CTA tile 64 x 128, 4 warps of 64 x 64, 3 stages, 2 CTAs per SM.
// 1: CTA tile 64 x 256, 8 warps of 64 x 64, 5 stages, 1 CTA per SM, the next
//    k16 step's codes dequantized during the current step's mma.
#define MARLINW_SCHED_LIST(X) \
  X(0, 128, 8, 4, 3, 2, false) X(1, 256, 16, 4, 5, 1, true)

namespace marlinw {

template <int threads, int thread_n_blocks, int warp_j, int stages,
          int min_blocks, bool dq_ahead>
struct Sched {
  static constexpr auto plain =
      MarlinS<threads, thread_n_blocks, warp_j, stages, min_blocks, dq_ahead,
              false>;
  static constexpr auto silu =
      MarlinS<threads, thread_n_blocks, warp_j, stages, min_blocks, dq_ahead,
              true>;
  static constexpr int smem =
      marlin_s_smem<threads, thread_n_blocks, warp_j, stages>();
};

#define MARLINW_SCHED_TYPE(i, t, n, j, s, b, d) using Sched##i = Sched<t, n, j, s, b, d>;
MARLINW_SCHED_LIST(MARLINW_SCHED_TYPE)
#undef MARLINW_SCHED_TYPE

// thread_m_blocks 4 (64-row chunks), thread_n 256, thread_k 64, 256 threads,
// 4 stages, group_blocks 1: the configuration ops.marlin_gemm selects for a
// launch of a multiple of 64 rows when {64, 256, 256} is valid for (N, K).
constexpr auto kKernel =
    Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
           vllm::kFE4M3fn.id(), 256, 4, 16, 4, false, 4, 1, false>;
// The same kernel writing silu(gate) * up of a [gate | up] weight (C is
// [M, N / 2]); see silu_mul in marlinw_template.h.
constexpr auto kKernelSiluMul =
    Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
           vllm::kFE4M3fn.id(), 256, 4, 16, 4, false, 4, 1, false, true>;
constexpr int kThreads = 256;

// out[r, c] = silu(in[r, c]) * in[r, d + c] for bf16 in [m, 2d] (row stride
// ld_in elements) and out [m, d], 8 elements per thread; d % 8 == 0.
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

extern "C" {

// Sets the dynamic shared memory limit of the kernel on the current device.
// Must run outside CUDA graph capture. Returns 0 or a cudaError_t.
int marlinw_init(int smem) {
  cudaError_t e = cudaFuncSetAttribute(
      marlinw::kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return static_cast<int>(e);
  e = cudaFuncSetAttribute(marlinw::kKernelSiluMul,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  return static_cast<int>(e);
}

int marlinw_regs(int silu_mul) {
  cudaFuncAttributes attr;
  if (cudaFuncGetAttributes(&attr, silu_mul ? marlinw::kKernelSiluMul
                                            : marlinw::kKernel) != cudaSuccess)
    return -1;
  return attr.numRegs;
}

// A: bf16 [M, lda] (row stride lda elements), B: Marlin-packed fe2m1f
// [K/16, 2N] int32, C: bf16 [M, N] (silu_mul: [M, N / 2]), C_tmp: fp32 scratch
// of n_slots * 64 * 256, scales: fe4m3fn [K/16, N] (S0E5M3, Marlin order),
// global_scale: fp32 [1], locks: int32 [n_slots], zero on entry and left zero
// on exit. work / work_ptr: see marlinw_template.h. grid = number of blocks.
// silu_mul != 0: the work list must hold whole-K slices only, each gate tile
// (col < N / 512) followed later in the same block's list by the up tile of
// the same chunk at col + N / 512.
int marlinw_gemm(const void* A, const void* B, void* C, void* C_tmp,
                 const void* scales, const void* global_scale, int prob_m,
                 int prob_n, int prob_k, int lda, void* locks,
                 const void* work, const void* work_ptr, int grid, int smem,
                 int silu_mul, void* stream) {
  if (prob_m <= 0 || prob_m % 64 || prob_n % 256 || prob_k % 64 || lda % 8 ||
      grid <= 0 || (silu_mul && prob_n % 512))
    return -2;
  auto kernel = silu_mul ? marlinw::kKernelSiluMul : marlinw::kKernel;
  kernel<<<grid, marlinw::kThreads, smem, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const int4*>(A), static_cast<const int4*>(B),
      static_cast<int4*>(C), static_cast<int4*>(C_tmp), nullptr, nullptr,
      static_cast<const int4*>(scales),
      static_cast<const float*>(global_scale), nullptr, nullptr,
      prob_k / 16, prob_m, prob_n, prob_k, lda, static_cast<int*>(locks),
      false, false, true, smem, static_cast<const int4*>(work),
      static_cast<const int*>(work_ptr));
  return static_cast<int>(cudaGetLastError());
}

// MarlinS configurations. marlinw_sched_info(v, out[8]) fills: threads,
// tile_n (columns per CTA), min_blocks (CTAs per SM the plan's grid assumes),
// dynamic shared memory, and the CTAs per SM that fit for the plain and the
// SiLU-and-mul kernel, their register counts. Returns -1 for an unknown
// index. Run marlinw_sched_init(v) first (outside graph capture): it sets the
// shared memory limit that the occupancy query and the launch need.
int marlinw_sched_count() {
  int n = 0;
#define MARLINW_SCHED_COUNT(i, t, nb, j, s, b, d) n++;
  MARLINW_SCHED_LIST(MARLINW_SCHED_COUNT)
#undef MARLINW_SCHED_COUNT
  return n;
}

int marlinw_sched_init(int v) {
#define MARLINW_SCHED_INIT(i, t, nb, j, s, b, d)                              \
  if (v == i) {                                                               \
    cudaError_t e = cudaFuncSetAttribute(                                     \
        marlinw::Sched##i::plain,                                             \
        cudaFuncAttributeMaxDynamicSharedMemorySize, marlinw::Sched##i::smem); \
    if (e == cudaSuccess)                                                     \
      e = cudaFuncSetAttribute(marlinw::Sched##i::silu,                       \
                               cudaFuncAttributeMaxDynamicSharedMemorySize,   \
                               marlinw::Sched##i::smem);                      \
    return static_cast<int>(e);                                               \
  }
  MARLINW_SCHED_LIST(MARLINW_SCHED_INIT)
#undef MARLINW_SCHED_INIT
  return -1;
}

int marlinw_sched_info(int v, int* out) {
#define MARLINW_SCHED_INFO(i, t, nb, j, s, b, d)                              \
  if (v == i) {                                                               \
    using S = marlinw::Sched##i;                                              \
    int o0 = 0, o1 = 0;                                                       \
    cudaFuncAttributes a0, a1;                                                \
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o0, S::plain, t,       \
                                                      S::smem) != cudaSuccess || \
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o1, S::silu, t,        \
                                                      S::smem) != cudaSuccess || \
        cudaFuncGetAttributes(&a0, S::plain) != cudaSuccess ||                \
        cudaFuncGetAttributes(&a1, S::silu) != cudaSuccess)                   \
      return -2;                                                              \
    out[0] = t;                                                               \
    out[1] = 16 * nb;                                                         \
    out[2] = b;                                                               \
    out[3] = S::smem;                                                         \
    out[4] = o0;                                                              \
    out[5] = o1;                                                              \
    out[6] = a0.numRegs;                                                      \
    out[7] = a1.numRegs;                                                      \
    return 0;                                                                 \
  }
  MARLINW_SCHED_LIST(MARLINW_SCHED_INFO)
#undef MARLINW_SCHED_INFO
  return -1;
}

// marlinw_gemm with MarlinS configuration v: the work list must come from the
// planner for that configuration's tile_n and grid (min_blocks * SMs).
int marlinw_sched_gemm(int v, const void* A, const void* B, void* C,
                       void* C_tmp, const void* scales,
                       const void* global_scale, int prob_m, int prob_n,
                       int prob_k, int lda, void* locks, const void* work,
                       const void* work_ptr, int grid, int silu_mul,
                       void* stream) {
#define MARLINW_SCHED_GEMM(i, t, nb, j, s, b, d)                              \
  if (v == i) {                                                               \
    using S = marlinw::Sched##i;                                              \
    if (prob_m <= 0 || prob_m % 64 || prob_n % (16 * nb) || prob_k % 64 ||    \
        lda % 8 || grid <= 0 || (silu_mul && prob_n % (32 * nb)))             \
      return -2;                                                              \
    auto kernel = silu_mul ? S::silu : S::plain;                              \
    kernel<<<grid, t, S::smem, static_cast<cudaStream_t>(stream)>>>(          \
        static_cast<const int4*>(A), static_cast<const int4*>(B),             \
        static_cast<int4*>(C), static_cast<int4*>(C_tmp),                     \
        static_cast<const int4*>(scales),                                     \
        static_cast<const float*>(global_scale), prob_m, prob_n, prob_k, lda, \
        static_cast<int*>(locks), static_cast<const int4*>(work),             \
        static_cast<const int*>(work_ptr));                                   \
    return static_cast<int>(cudaGetLastError());                              \
  }
  MARLINW_SCHED_LIST(MARLINW_SCHED_GEMM)
#undef MARLINW_SCHED_GEMM
  return -1;
}

// out [m, d] = silu(in[:, :d]) * in[:, d:] (bf16, row stride ld_in elements),
// the same arithmetic as the silu_mul epilogue above.
int marlinw_silu_mul(const void* in, void* out, int m, int d, int ld_in,
                     void* stream) {
  if (m < 0 || d <= 0 || d % 8 || ld_in % 8 || ld_in < 2 * d) return -2;
  if (m == 0) return 0;
  int64_t n = static_cast<int64_t>(m) * (d / 8);
  int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  marlinw::SiluMul<<<static_cast<unsigned>(blocks), threads, 0,
                     static_cast<cudaStream_t>(stream)>>>(
      static_cast<const int4*>(in), static_cast<int4*>(out), m, d / 8,
      ld_in / 8);
  return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
