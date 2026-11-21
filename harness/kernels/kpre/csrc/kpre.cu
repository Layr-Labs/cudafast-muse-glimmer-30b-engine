// kpre: vLLM v0.27.1 Marlin NVFP4 W4A16 GEMM for prefill (M a multiple of 64, one
// launch), with one change: the part-1 tiles (whole-K, one CTA each, no cross-CTA
// reduce) are visited column-major over the 64-row M slabs instead of slab-major,
// so a round of CTAs shares each weight tile through L2 instead of re-reading the
// whole weight from DRAM once per slab. The kernel body, the tile config, the
// part-1/part-2 split and every part-2 slice are stock (marlin/marlin_template.h,
// vendored from csrc/libtorch_stable/quantization/marlin, patch under KPRE_REMAP),
// so each output element is produced by the same instructions in the same order.
//
// C ABI (ctypes): kpre_gemm(...) returns 0 on launch, >0 when the call is outside the
// guard (caller runs stock Marlin), <0 on CUDA error.
#define MARLIN_NAMESPACE_NAME kpre_marlin
#ifndef KPRE_STOCK_ORDER  // test-only build: stock slab-major order
#define KPRE_REMAP 1
#endif
#include "kernel.h"
#include "marlin_template.h"

namespace kpre_marlin {
// The exact stock instantiation chosen by marlin_mm for NVFP4 bf16 at
// thread_m_blocks 4: large_batch config {thread_k 64, thread_n 256, 256 threads}.
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                                vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 256,
                                4, 16, 4, false, 4, 1, false>(MARLIN_KERNEL_PARAMS);
}  // namespace kpre_marlin

namespace {
using namespace kpre_marlin;
auto kKernel = Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
                      vllm::kFE4M3fn.id(), 256, 4, 16, 4, false, 4, 1, false>;

// Same arithmetic as marlin.cu get_kernel_cache_size / get_scales_cache_size for
// this case (no act order, no zp, 16-bit A, group_size 16, stages 4).
int cache_size(int tb_k, int tb_n, int thread_m_blocks) {
  const int stages = 4, pack_factor = 8, group_size = 16;
  int tb_m = thread_m_blocks * 16;
  int sh_a_size = stages * (tb_m * tb_k) * 2;
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);
  int tb_groups = div_ceil(tb_k, group_size);
  int sh_s_size = tb_groups * tb_n * 2 * stages;
  return tmp_size + sh_a_size + sh_s_size;
}
}  // namespace

extern "C" {

int kpre_init() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return -1;
  int smem = 0;
  cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (cudaFuncSetAttribute(kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) !=
      cudaSuccess)
    return -2;
  return 0;
}

// Returns 1 when stock marlin_mm would run exactly this kernel in exactly one launch
// with prob_m_split == m (so kpre reproduces it), else 0.
int kpre_eligible(int m, int n, int k, int sms) {
  if (m <= 64 || m % 64 != 0) return 0;  // thread_m_blocks 4 and parallel > 1
  int max_par = n <= 4096 ? 16 * 8 : 16;
  if (m / 64 > max_par) return 0;  // stock would split M over several launches
  if (k % 64 != 0 || n % 256 != 0) return 0;
  int dev = 0, smem = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  // determine_exec_config: first valid large-batch config must be {64, 256, 256}.
  if (cache_size(64, 256, 4) > smem - 512) return 0;
  // marlin_mm override to {128, 64, 128} when the tile count is small.
  if (n / 256 * (m / 64) * 4 <= sms) return 0;
  return 1;
}

int kpre_gemm(const void* A, long long lda, int m, const void* B, const void* S,
              const void* GS, void* C, void* C_tmp, void* locks, int n, int k, int sms,
              void* stream) {
  if (!kpre_eligible(m, n, k, sms)) return 1;
  if (lda % 8 != 0 || ((unsigned long long)A) % 16 != 0) return 2;
  int dev = 0, smem = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  const int num_groups = k / 16;
  kKernel<<<sms, 256, smem, (cudaStream_t)stream>>>(
      (const int4*)A, (const int4*)B, (int4*)C, (int4*)C_tmp, (const int4*)nullptr,
      (const float*)nullptr, (const int4*)S, (const float*)GS, (const int4*)nullptr,
      (const int*)nullptr, num_groups, m, n, k, (int)lda, (int*)locks,
      /*has_bias=*/false, /*use_atomic_add=*/false, /*use_fp32_reduce=*/true, smem);
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : -(int)e;
}

}  // extern "C"
