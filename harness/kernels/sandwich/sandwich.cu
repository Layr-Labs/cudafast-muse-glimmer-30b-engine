#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace {
constexpr int Columns = 6656;
constexpr int Threads = 256;
constexpr int Vectors = Columns / 4;
constexpr int Tiles = (Vectors + Threads - 1) / Threads;
constexpr int CacheLineBytes = 128;
constexpr int RowBytes = Columns * 2;
constexpr int RowCacheLines = RowBytes / CacheLineBytes;
static_assert(RowBytes % CacheLineBytes == 0,
              "Row byte size must be an exact multiple of 128 B for L2 discard");
static_assert(RowCacheLines <= Threads,
              "One CTA thread per cache line must cover an entire row");

__device__ __forceinline__ void l2_discard(const void* p) {
    asm volatile("discard.global.L2 [%0], 128;" :: "l"(p) : "memory");
}

__device__ float bf16(uint2 packed, int channel) {
    const unsigned bits = channel < 2 ? packed.x : packed.y;
    return __uint_as_float((channel & 1) ? bits & 0xffff0000 : bits << 16);
}

__device__ unsigned pack_bf16(float low, float high) {
    unsigned result;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(result) : "f"(low), "f"(high));
    return result;
}

__device__ float full_divide(float value, float divisor) {
    float result;
    asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(value), "f"(divisor));
    return result;
}

__device__ float inverse_rms(float total, float eps) {
    // ptxas contracts the archived div.full by this constant and the epsilon
    // addition to an FFMA, even though the emitted PTX lists both operations.
    const float variance = __fmaf_rn(total, 1.f / Columns, eps);
    float result;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(variance));
    return result;
}

__device__ float reduce(const float (&partial)[4], float* warp_totals, unsigned thread) {
    // Match Inductor's [1,4] elements per lane, then XOR within a warp and
    // across the eight warps. Each channel carries its own 1024-strided chain.
    float sum = __fadd_rn(__fadd_rn(__fadd_rn(partial[0], partial[1]), partial[2]), partial[3]);
    #pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, delta));
    }
    if (thread % 32 == 0) {
        warp_totals[thread / 32] = sum;
    }
    __syncthreads();
    sum = warp_totals[thread % 8];
    #pragma unroll
    for (int delta = 4; delta > 0; delta /= 2) {
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, delta, 8));
    }
    sum = __shfl_sync(0xffffffff, sum, 0, 8);
    __syncthreads();
    return sum;
}

__device__ float apply_norm(float value, float inverse, float weight) {
    return __fmul_rn(__fmul_rn(value, inverse), __fadd_rn(weight, 1.f));
}

__device__ float bounded_fp8(float value, float scale_inv) {
    float scaled = __fmul_rn(value, scale_inv);
    if (scaled < -448.f) scaled = -448.f;
    if (scaled > 448.f) scaled = 448.f;
    return scaled;
}

__device__ unsigned pack_fp8(const float (&values)[4], float scale_inv) {
    unsigned short low, high;
    float x0 = bounded_fp8(values[0], scale_inv);
    float x1 = bounded_fp8(values[1], scale_inv);
    float x2 = bounded_fp8(values[2], scale_inv);
    float x3 = bounded_fp8(values[3], scale_inv);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(low) : "f"(x0), "f"(x1));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(high) : "f"(x2), "f"(x3));
    return static_cast<unsigned>(low) | (static_cast<unsigned>(high) << 16);
}

template <bool Quantize, bool Shared, bool DiscardDown>
__global__ __launch_bounds__(Threads) void sandwich(
    uint2* post_attn, const uint2* down, const uint2* residual,
    const float* previous_sum, const uint2* post_attn_weight,
    const uint2* post_ff_weight, const uint2* next_pre_attn_weight,
    const float* scale, uint2* hidden, unsigned* quantized) {
    __shared__ float warp_totals[8];
    extern __shared__ uint2 shared_row[];
    uint2 saved[Tiles];
    const unsigned thread = threadIdx.x;
    const size_t base = blockIdx.x * Vectors;
    float partial[4] = {};
    #pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        const unsigned offset = tile * Threads + thread;
        if (offset < Vectors) {
            const uint2 values = __ldcs(&down[base + offset]);
            if constexpr (Shared) {
                shared_row[offset] = values;
            } else {
                saved[tile] = values;
            }
            #pragma unroll
            for (int channel = 0; channel < 4; ++channel) {
                const float value = bf16(values, channel);
                partial[channel] = __fmaf_rn(value, value, partial[channel]);
            }
        }
    }
    const float down_inverse = inverse_rms(reduce(partial, warp_totals, thread), 1.e-8f);
    const float attn_inverse = inverse_rms(previous_sum[blockIdx.x], 1.e-8f);
    if constexpr (DiscardDown) {
        if (thread < RowCacheLines) {
            l2_discard(reinterpret_cast<const char*>(down + base) + thread * CacheLineBytes);
        }
    }
    #pragma unroll
    for (int channel = 0; channel < 4; ++channel) partial[channel] = 0.f;
    #pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        const unsigned offset = tile * Threads + thread;
        if (offset < Vectors) {
            uint2 d;
            if constexpr (Shared) {
                d = shared_row[offset];
            } else {
                d = saved[tile];
            }
            const uint2 a = post_attn[base + offset];
            const uint2 r = __ldcs(&residual[base + offset]);
            const uint2 wa = post_attn_weight[offset];
            const uint2 wd = post_ff_weight[offset];
            float values[4];
            #pragma unroll
            for (int channel = 0; channel < 4; ++channel) {
                const float attention = apply_norm(bf16(a, channel), attn_inverse, bf16(wa, channel));
                const float feed_forward = __fmul_rn(bf16(d, channel), down_inverse);
                const float attention_sum = __fadd_rn(bf16(r, channel), attention);
                const float value = __fmaf_rn(feed_forward, __fadd_rn(bf16(wd, channel), 1.f), attention_sum);
                values[channel] = value;
                partial[channel] = __fmaf_rn(value, value, partial[channel]);
            }
            const uint2 rounded = make_uint2(pack_bf16(values[0], values[1]),
                                             pack_bf16(values[2], values[3]));
            post_attn[base + offset] = rounded;
            if constexpr (Shared) {
                shared_row[offset] = rounded;
            } else {
                saved[tile] = rounded;
            }
        }
    }
    const float input_inverse = inverse_rms(reduce(partial, warp_totals, thread), 1.e-5f);
    const float scale_inv = Quantize ? full_divide(1.f, *scale) : 1.f;
    #pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        const unsigned offset = tile * Threads + thread;
        if (offset < Vectors) {
            uint2 r;
            if constexpr (Shared) {
                r = shared_row[offset];
            } else {
                r = saved[tile];
            }
            const uint2 w = next_pre_attn_weight[offset];
            float values[4];
            #pragma unroll
            for (int channel = 0; channel < 4; ++channel) {
                values[channel] = apply_norm(bf16(r, channel), input_inverse, bf16(w, channel));
            }
            hidden[base + offset] = make_uint2(pack_bf16(values[0], values[1]),
                                               pack_bf16(values[2], values[3]));
            if constexpr (Quantize) {
                quantized[base + offset] = pack_fp8(values, scale_inv);
            }
        }
    }
}

bool span_disjoint(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    const uintptr_t a0 = reinterpret_cast<uintptr_t>(a);
    const uintptr_t b0 = reinterpret_cast<uintptr_t>(b);
    return a0 <= b0 ? (b0 - a0 >= a_bytes) : (a0 - b0 >= b_bytes);
}

template <bool Quantize>
bool can_discard_down(const void* const* args, int rows) {
    if (rows <= 0) return false;
    const void* down = args[1];
    if ((reinterpret_cast<uintptr_t>(down) & (CacheLineBytes - 1)) != 0) return false;
    const size_t act_bytes = static_cast<size_t>(rows) * RowBytes;
    const size_t vec_bytes = RowBytes;
    const bool disjoint_common =
        span_disjoint(down, act_bytes, args[0], act_bytes) &&
        span_disjoint(down, act_bytes, args[2], act_bytes) &&
        span_disjoint(down, act_bytes, args[3], static_cast<size_t>(rows) * sizeof(float)) &&
        span_disjoint(down, act_bytes, args[4], vec_bytes) &&
        span_disjoint(down, act_bytes, args[5], vec_bytes) &&
        span_disjoint(down, act_bytes, args[6], vec_bytes) &&
        span_disjoint(down, act_bytes, args[Quantize ? 8 : 7], act_bytes);
    if (!disjoint_common) return false;
    if constexpr (Quantize) {
        return span_disjoint(down, act_bytes, args[7], sizeof(float)) &&
               span_disjoint(down, act_bytes, args[9], static_cast<size_t>(rows) * Columns);
    }
    return true;
}

template <bool Quantize, bool Shared>
void launch(const void* const* args, int rows, cudaStream_t stream) {
    auto invoke = [&](auto discard_tag) {
        constexpr bool DiscardDown = decltype(discard_tag)::value;
        sandwich<Quantize, Shared, DiscardDown><<<rows, Threads, Shared ? Vectors * sizeof(uint2) : 0, stream>>>(
            static_cast<uint2*>(const_cast<void*>(args[0])), static_cast<const uint2*>(args[1]),
            static_cast<const uint2*>(args[2]), static_cast<const float*>(args[3]),
            static_cast<const uint2*>(args[4]), static_cast<const uint2*>(args[5]),
            static_cast<const uint2*>(args[6]), Quantize ? static_cast<const float*>(args[7]) : nullptr,
            static_cast<uint2*>(const_cast<void*>(args[Quantize ? 8 : 7])),
            Quantize ? static_cast<unsigned*>(const_cast<void*>(args[9])) : nullptr);
    };
    if constexpr (Shared) {
        if (can_discard_down<Quantize>(args, rows)) {
            invoke(std::true_type{});
            return;
        }
    }
    invoke(std::false_type{});
}
}

extern "C" int muse_sandwich(const void* const* args, int rows, bool quantize, bool shared,
                              cudaStream_t stream) {
    if (quantize) {
        if (shared) launch<true, true>(args, rows, stream);
        else launch<true, false>(args, rows, stream);
    } else {
        if (shared) launch<false, true>(args, rows, stream);
        else launch<false, false>(args, rows, stream);
    }
    return cudaGetLastError();
}
