#define MARLIN_NAMESPACE_NAME muse_marlin
#include "marlin/marlin_template.h"

namespace {

template <int NBlocks, int Threads>
void launch(const void* a, const void* weight, const void* scales,
            const float* global_scale, void* output, void* scratch, int* locks,
            int n, int k, int sms, bool expanded, cudaStream_t stream) {
  constexpr int n_tile = 16 * NBlocks;
  constexpr int smem = NBlocks == 8 ? 45056 : 26624;
  const int n_tiles = n / n_tile;
  int tail_tiles = n_tiles;
  if (n_tiles > sms) {
    tail_tiles = n_tiles % sms;
    if (tail_tiles * 3 <= sms) tail_tiles += sms;
  }
  const int blocks = expanded ? n_tiles - tail_tiles + sms : sms;
  auto kernel = muse_marlin::Marlin<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), Threads, 1, NBlocks, 8, true, 4, 1, false>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  kernel<<<blocks, Threads, smem, stream>>>(
      static_cast<const int4*>(a), static_cast<const int4*>(weight),
      static_cast<int4*>(output), static_cast<int4*>(scratch), nullptr, nullptr,
      static_cast<const int4*>(scales), global_scale, nullptr, nullptr, k / 16,
      1, n, k, k, locks, false, false, true, smem, sms, expanded);
}

}  // namespace

void launch_muse_marlin(const void* a, const void* weight, const void* scales,
                        const float* global_scale, void* output, void* scratch,
                        int* locks, int n, int k, int sms, bool expanded,
                        cudaStream_t stream) {
  // This is the stock selector's first valid tile for these dimensions.
  if (n % 128 == 0 && (n / 128) * 4 > sms) {
    launch<8, 256>(a, weight, scales, global_scale, output, scratch, locks, n, k,
                   sms, expanded, stream);
  } else {
    launch<4, 128>(a, weight, scales, global_scale, output, scratch, locks, n, k,
                   sms, expanded, stream);
  }
}
