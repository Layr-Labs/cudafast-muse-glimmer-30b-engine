// SPDX-License-Identifier: Apache-2.0
// The board's M = 1 tiled-layout NVFP4 decode GEMV (harness/kernels/decode/gemv.cu) with the group-16 FP8 block scales
// read as 6-bit codes into a 64-entry per-layer palette (marlin_template.h, template flag scale6). Same template
// arguments otherwise, so the same instructions on the same operands in the same order: the same output bits.
#define MARLIN_NAMESPACE_NAME muse_marlin6
#include <cstdlib>
#include "marlin/marlin_template.h"

namespace {
static size_t decode6_attn_tpf_stride() {
  static size_t stride = []() -> size_t {
    const char* env = std::getenv("MUSE_DECODE6_TLB_PREFETCH");
    if (!env || !*env) return 0u;
    long kb = std::strtol(env, nullptr, 10);
    return kb > 0 ? (static_cast<size_t>(kb) << 10) : 0u;
  }();
  return stride;
}

template <int TileN, int Threads, bool GateUpGrid = false>
cudaError_t launch(const int4* input, const int4* weight, int4* output,
                   int4* partials, const int4* scales6, const int4* palette, const float* global_scale,
                   int* locks, int n, int k, int blocks, int shared, cudaStream_t stream) {
    auto kernel = muse_marlin6::Marlin<
        vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
        vllm::kFE4M3fn.id(), Threads, 1, TileN / 16, 8, true, 4, 1, false, true, GateUpGrid>;
    auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (status != cudaSuccess) return status;
    const int grid = GateUpGrid ? 336 : blocks;
    kernel<<<grid, Threads, shared, stream>>>(
        input, weight, output, partials, nullptr, nullptr, scales6, global_scale,
        nullptr, nullptr, k / 16, 1, n, k, k, locks, false, false, true, shared, palette,
        decode6_attn_tpf_stride());
    return cudaGetLastError();
}
}  // namespace

extern "C" int muse_marlin6_gemv(const int4* input, const int4* weight, int4* output,
                                 int4* partials, const int4* scales6, const int4* palette,
                                 const float* global_scale, int* locks, int n, int k, int blocks,
                                 int device, cudaStream_t stream) {
    int shared = 0;
    auto status = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status != cudaSuccess) return status;
    if (n % 128 == 0)
        return launch<128, 256>(input, weight, output, partials, scales6, palette, global_scale,
                                locks, n, k, blocks, shared, stream);
    return launch<64, 128>(input, weight, output, partials, scales6, palette, global_scale,
                           locks, n, k, blocks, shared, stream);
}

extern "C" int muse_marlin6_gate_up(const int4* input, const int4* weight, int4* output,
                                 int4* partials, const int4* scales6, const int4* palette,
                                 const float* global_scale, int* locks, int n, int k, int blocks,
                                 int device, cudaStream_t stream) {
    int shared = 0;
    auto status = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status != cudaSuccess) return status;
    if (n != 39936 || k != 6656 || blocks != 48) return cudaErrorInvalidValue;
    return launch<128, 256, true>(input, weight, output, partials, scales6, palette, global_scale,
                                  locks, n, k, blocks, shared, stream);
}
