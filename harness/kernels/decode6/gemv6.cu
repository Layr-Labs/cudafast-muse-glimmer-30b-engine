// SPDX-License-Identifier: Apache-2.0
// The board's M = 1 tiled-layout NVFP4 decode GEMV (harness/kernels/decode/gemv.cu) with the group-16 FP8 block scales
// read as 6-bit codes into a 64-entry per-layer palette (marlin_template.h, template flag scale6). Same template
// arguments otherwise, so the same instructions on the same operands in the same order: the same output bits.
#define MARLIN_NAMESPACE_NAME muse_marlin6
#include "marlin/marlin_template.h"

namespace {
template <int TileN, int Threads, bool GateUpGrid = false>
cudaError_t launch(const int4* input, const int4* weight, int4* output,
                   int4* partials, const int4* scales6, const int4* palette, const float* global_scale,
                   int* locks, int n, int k, int blocks, int device, cudaStream_t stream) {
    auto kernel = muse_marlin6::Marlin<
        vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
        vllm::kFE4M3fn.id(), Threads, 1, TileN / 16, 8, true, 4, 1, false, true, GateUpGrid>;
    static int cached_shared[32] = {};
    static bool configured[32][3] = {};
    int shared = 0;
    const int tidx = GateUpGrid ? 2 : ((TileN == 128) ? 0 : 1);
    if (device >= 0 && device < 32) {
        if (cached_shared[device] == 0) {
            auto status = cudaDeviceGetAttribute(&cached_shared[device], cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
            if (status != cudaSuccess) return status;
        }
        shared = cached_shared[device];
        if (!configured[device][tidx]) {
            auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
            if (status != cudaSuccess) return status;
            configured[device][tidx] = true;
        }
    } else {
        auto status = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (status != cudaSuccess) return status;
        status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
        if (status != cudaSuccess) return status;
    }
    const int grid = GateUpGrid ? 336 : blocks;
    kernel<<<grid, Threads, shared, stream>>>(
        input, weight, output, partials, nullptr, nullptr, scales6, global_scale,
        nullptr, nullptr, k / 16, 1, n, k, k, locks, false, false, true, shared, palette);
    return cudaGetLastError();
}
}  // namespace

extern "C" int muse_marlin6_gemv(const int4* input, const int4* weight, int4* output,
                                 int4* partials, const int4* scales6, const int4* palette,
                                 const float* global_scale, int* locks, int n, int k, int blocks,
                                 int device, cudaStream_t stream) {
    if (n % 128 == 0 && (n / 128) * 4 > blocks)
        return launch<128, 256>(input, weight, output, partials, scales6, palette, global_scale,
                                locks, n, k, blocks, device, stream);
    return launch<64, 128>(input, weight, output, partials, scales6, palette, global_scale,
                           locks, n, k, blocks, device, stream);
}

extern "C" int muse_marlin6_gate_up(const int4* input, const int4* weight, int4* output,
                                    int4* partials, const int4* scales6, const int4* palette,
                                    const float* global_scale, int* locks, int n, int k, int blocks,
                                    int device, cudaStream_t stream) {
    if (n != 39936 || k != 6656 || blocks != 48) return cudaErrorInvalidValue;
    return launch<128, 256, true>(input, weight, output, partials, scales6, palette, global_scale,
                                  locks, n, k, blocks, device, stream);
}
