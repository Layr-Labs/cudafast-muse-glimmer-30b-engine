// SPDX-License-Identifier: Apache-2.0
#define MARLIN_NAMESPACE_NAME muse_marlin
#include "marlin/marlin_template.h"

namespace {
template <int TileN, int Threads, int ThreadMBlocks, bool MBlock8, bool MultiM>
cudaError_t launch(const int4* input, const int4* weight, int4* output,
                   int4* partials, const int4* scales, const float* global_scale,
                   int* locks, int m, int n, int k, int blocks, int device, cudaStream_t stream) {
    auto kernel = muse_marlin::Marlin<
        vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
        vllm::kFE4M3fn.id(), Threads, ThreadMBlocks, TileN / 16, 8, MBlock8, 4, 1, false, MultiM>;
    // Exact SMEM (stages=4, thread_k_blocks=8):
    //   TileN=128: sh_b(32768)+sh_s(4096)+sh_a(8192|16384|32768) = 44|52|68 KiB -> 48|56|72 KiB
    //   TileN=64:  sh_b(16384)+sh_s(2048)+sh_a(8192|16384|32768) = 26|34|50 KiB -> 32|40|56 KiB
    constexpr int shared = (TileN == 128)
        ? (MBlock8 ? (48 * 1024) : (ThreadMBlocks == 2 ? (72 * 1024) : (56 * 1024)))
        : (MBlock8 ? (32 * 1024) : (ThreadMBlocks == 2 ? (56 * 1024) : (40 * 1024)));
    static bool configured[16] = {};
    const int dev_idx = (device >= 0 && device < 16) ? device : 0;
    if (!configured[dev_idx]) {
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        cudaThreadExchangeStreamCaptureMode(&mode);
        auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
        if (status == cudaSuccess && shared <= 62 * 1024) {
            (void)cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
        }
        cudaThreadExchangeStreamCaptureMode(&mode);
        if (status != cudaSuccess) return status;
        configured[dev_idx] = true;
    }
    kernel<<<blocks, Threads, shared, stream>>>(
        input, weight, output, partials, nullptr, nullptr, scales, global_scale,
        nullptr, nullptr, k / 16, m, n, k, k, locks, false, false, true, shared);
    return cudaGetLastError();
}

template <int TileN, int Threads>
cudaError_t launch_by_m(const int4* input, const int4* weight, int4* output,
                        int4* partials, const int4* scales, const float* global_scale,
                        int* locks, int m, int n, int k, int blocks, int device, cudaStream_t stream) {
    if (m == 1)
        return launch<TileN, Threads, 1, true, false>(
            input, weight, output, partials, scales, global_scale, locks, 1, n, k, blocks, device, stream);
    if (m <= 16)
        return launch<TileN, Threads, 1, false, true>(
            input, weight, output, partials, scales, global_scale, locks, m, n, k, blocks, device, stream);
    return launch<TileN, Threads, 2, false, true>(
        input, weight, output, partials, scales, global_scale, locks, m, n, k, blocks, device, stream);
}
}  // namespace

extern "C" int muse_marlin_gemv_m(const int4* input, const int4* weight, int4* output,
                                  int4* partials, const int4* scales, const float* global_scale,
                                  int* locks, int m, int n, int k, int blocks, int device,
                                  cudaStream_t stream) {
    if (m < 1 || m > 32 || (n % 64) != 0 || (k % 128) != 0 || blocks <= 0)
        return cudaErrorInvalidValue;
    if (n % 128 == 0)
        return launch_by_m<128, 256>(input, weight, output, partials, scales, global_scale,
                                     locks, m, n, k, blocks, device, stream);
    return launch_by_m<64, 128>(input, weight, output, partials, scales, global_scale,
                                locks, m, n, k, blocks, device, stream);
}

extern "C" int muse_marlin_gemv(const int4* input, const int4* weight, int4* output,
                                int4* partials, const int4* scales, const float* global_scale,
                                int* locks, int n, int k, int blocks, int device, cudaStream_t stream) {
    return muse_marlin_gemv_m(input, weight, output, partials, scales, global_scale,
                              locks, 1, n, k, blocks, device, stream);
}
