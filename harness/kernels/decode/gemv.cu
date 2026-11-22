// SPDX-License-Identifier: Apache-2.0
#define MARLIN_NAMESPACE_NAME muse_marlin
#include "marlin/marlin_template.h"

namespace {
template <int TileN, int Threads>
cudaError_t launch(const int4* input, const int4* weight, int4* output,
                   int4* partials, const int4* scales, const float* global_scale,
                   int* locks, int n, int k, int blocks, int device, cudaStream_t stream) {
    auto kernel = muse_marlin::Marlin<
        vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
        vllm::kFE4M3fn.id(), Threads, 1, TileN / 16, 8, true, 4, 1, false>;
    static int cached_shared[32] = {};
    static bool configured[32][2] = {};
    int shared = 0;
    const int tidx = (TileN == 128) ? 0 : 1;
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
    kernel<<<blocks, Threads, shared, stream>>>(
        input, weight, output, partials, nullptr, nullptr, scales, global_scale,
        nullptr, nullptr, k / 16, 1, n, k, k, locks, false, false, true, shared);
    return cudaGetLastError();
}
}  // namespace

extern "C" int muse_marlin_gemv(const int4* input, const int4* weight, int4* output,
                                int4* partials, const int4* scales, const float* global_scale,
                                int* locks, int n, int k, int blocks, int device, cudaStream_t stream) {
    if (n % 128 == 0 && (n / 128) * 4 > blocks)
        return launch<128, 256>(input, weight, output, partials, scales, global_scale,
                                locks, n, k, blocks, device, stream);
    return launch<64, 128>(input, weight, output, partials, scales, global_scale,
                           locks, n, k, blocks, device, stream);
}
