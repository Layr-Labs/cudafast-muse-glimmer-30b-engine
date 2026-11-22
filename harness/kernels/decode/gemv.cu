// SPDX-License-Identifier: Apache-2.0
#define MARLIN_NAMESPACE_NAME muse_marlin
#include <cstdlib>
#include "marlin/marlin_template.h"

namespace {
template <int TileN, int Threads>
cudaError_t launch(const int4* input, const int4* weight, int4* output,
                   int4* partials, const int4* scales, const float* global_scale,
                   int* locks, int n, int k, int blocks, int shared, cudaStream_t stream) {
    auto kernel = muse_marlin::Marlin<
        vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
        vllm::kFE4M3fn.id(), Threads, 1, TileN / 16, 8, true, 4, 1, false>;
    auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (status != cudaSuccess) return status;
    // Programmatic dependent launch: the kernel prefetches weights before its
    // griddepcontrol.wait, so it may start while the previous kernel drains.
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks);
    cfg.blockDim = dim3(Threads);
    cfg.dynamicSmemBytes = shared;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = getenv("MUSE_PDL_OFF") ? 0 : 1;
    return cudaLaunchKernelEx(&cfg, kernel,
        input, weight, output, partials, (const int4*)nullptr, (const float*)nullptr, scales, global_scale,
        (const int4*)nullptr, (const int*)nullptr, k / 16, 1, n, k, k, locks, false, false, true, shared);
}
}  // namespace

extern "C" int muse_marlin_gemv(const int4* input, const int4* weight, int4* output,
                                int4* partials, const int4* scales, const float* global_scale,
                                int* locks, int n, int k, int blocks, int device, cudaStream_t stream) {
    int shared = 0;
    auto status = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status != cudaSuccess) return status;
    if (n % 128 == 0)
        return launch<128, 256>(input, weight, output, partials, scales, global_scale,
                                locks, n, k, blocks, shared, stream);
    return launch<64, 128>(input, weight, output, partials, scales, global_scale,
                           locks, n, k, blocks, shared, stream);
}
