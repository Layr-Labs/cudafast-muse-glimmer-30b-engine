#include <cuda_runtime.h>

namespace {
constexpr int Columns = 6656;
constexpr int Threads = 256;
constexpr int Vectors = Columns / 4;
constexpr int Tiles = (Vectors + Threads - 1) / Threads;

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

// Gate (prefill3): also emit the FP8 input of the attention output-gate projection, i.e. what the compiled
// requant kernel (triton_poi_fused__to_copy_clamp_mul_reciprocal_0) computes from `hidden`: the bf16-rounded
// value times div.full(1, gate_scale), clamped to +-448, cvt.rn.satfinite.e4m3. With WriteHidden false the bf16
// `hidden` row is not stored (its only reader in the FP8-attention layers is that requant).
__device__ __forceinline__ void store_evict_last(unsigned* address, unsigned value) {
    // L2 eviction priority only (no effect on the value): the folded gate operand is read by the gate GEMM after the
    // qkv GEMM has streamed its weights through L2.
    unsigned long long policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;" :: "l"(address), "r"(value), "l"(policy) : "memory");
}

__device__ __forceinline__ void store2_evict_last(uint2* address, uint2 value) {
    unsigned long long policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    asm volatile("st.global.L2::cache_hint.v2.b32 [%0], {%1, %2}, %3;"
                 :: "l"(address), "r"(value.x), "r"(value.y), "l"(policy) : "memory");
}

template <bool Quantize, bool Shared, bool Gate = false, bool WriteHidden = true, bool GateEvictLast = false,
          bool ResidualEvictLast = false>
__global__ __launch_bounds__(Threads) void sandwich(
    uint2* post_attn, const uint2* down, const uint2* residual,
    const float* previous_sum, const uint2* post_attn_weight,
    const uint2* post_ff_weight, const uint2* next_pre_attn_weight,
    const float* scale, uint2* hidden, unsigned* quantized,
    const float* gate_scale = nullptr, unsigned* gate_quantized = nullptr) {
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
            const uint2 values = down[base + offset];
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
            const uint2 r = residual[base + offset];
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
            if constexpr (ResidualEvictLast) {
                store2_evict_last(post_attn + base + offset, rounded);
            } else {
                post_attn[base + offset] = rounded;
            }
            if constexpr (Shared) {
                shared_row[offset] = rounded;
            } else {
                saved[tile] = rounded;
            }
        }
    }
    const float input_inverse = inverse_rms(reduce(partial, warp_totals, thread), 1.e-5f);
    const float scale_inv = Quantize ? full_divide(1.f, *scale) : 1.f;
    const float gate_inv = Gate ? full_divide(1.f, *gate_scale) : 1.f;
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
            const uint2 rounded_hidden = make_uint2(pack_bf16(values[0], values[1]),
                                                    pack_bf16(values[2], values[3]));
            if constexpr (WriteHidden) {
                hidden[base + offset] = rounded_hidden;
            }
            if constexpr (Quantize) {
                quantized[base + offset] = pack_fp8(values, scale_inv);
            }
            if constexpr (Gate) {
                float gate_values[4];
                #pragma unroll
                for (int channel = 0; channel < 4; ++channel) {
                    gate_values[channel] = bf16(rounded_hidden, channel);
                }
                if constexpr (GateEvictLast) {
                    store_evict_last(gate_quantized + base + offset, pack_fp8(gate_values, gate_inv));
                } else {
                    gate_quantized[base + offset] = pack_fp8(gate_values, gate_inv);
                }
            }
        }
    }
}

template <bool Quantize, bool Shared>
void launch(const void* const* args, int rows, cudaStream_t stream) {
    sandwich<Quantize, Shared><<<rows, Threads, Shared ? Vectors * sizeof(uint2) : 0, stream>>>(
        static_cast<uint2*>(const_cast<void*>(args[0])), static_cast<const uint2*>(args[1]),
        static_cast<const uint2*>(args[2]), static_cast<const float*>(args[3]),
        static_cast<const uint2*>(args[4]), static_cast<const uint2*>(args[5]),
        static_cast<const uint2*>(args[6]), Quantize ? static_cast<const float*>(args[7]) : nullptr,
        static_cast<uint2*>(const_cast<void*>(args[Quantize ? 8 : 7])),
        Quantize ? static_cast<unsigned*>(const_cast<void*>(args[9])) : nullptr);
}
}

template <bool Shared, bool WriteHidden, bool EvictLast = false>
void launch_gate(const void* const* args, int rows, cudaStream_t stream) {
    sandwich<true, Shared, true, WriteHidden, EvictLast><<<rows, Threads, Shared ? Vectors * sizeof(uint2) : 0, stream>>>(
        static_cast<uint2*>(const_cast<void*>(args[0])), static_cast<const uint2*>(args[1]),
        static_cast<const uint2*>(args[2]), static_cast<const float*>(args[3]),
        static_cast<const uint2*>(args[4]), static_cast<const uint2*>(args[5]),
        static_cast<const uint2*>(args[6]), static_cast<const float*>(args[7]),
        static_cast<uint2*>(const_cast<void*>(args[8])), static_cast<unsigned*>(const_cast<void*>(args[9])),
        static_cast<const float*>(args[10]), static_cast<unsigned*>(const_cast<void*>(args[11])));
}

// The gate quantization formula alone on n bf16 bit patterns (the admission check runs it on all 65,536 and
// compares with the compiled requant kernel's output for the same patterns and the layer's real scale).
__global__ void gate_formula(const unsigned short* bits, const float* gate_scale, unsigned char* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gate_inv = full_divide(1.f, *gate_scale);
    float v[4] = {__uint_as_float(static_cast<unsigned>(bits[i]) << 16), 0.f, 0.f, 0.f};
    out[i] = static_cast<unsigned char>(pack_fp8(v, gate_inv) & 0xff);
}

// args: the ten pointers of the quantizing sandwich, then gate_scale (float[1]) and gate_quantized (fp8 rows).
extern "C" int muse_sandwich_gate(const void* const* args, int rows, bool shared, bool write_hidden,
                                   cudaStream_t stream) {
    if (shared) {
        if (write_hidden) launch_gate<true, true>(args, rows, stream);
        else launch_gate<true, false>(args, rows, stream);
    } else {
        if (write_hidden) launch_gate<false, true>(args, rows, stream);
        else launch_gate<false, false>(args, rows, stream);
    }
    return cudaGetLastError();
}

// The same with the gate operand stored at L2 evict_last priority (shared-row staging, bf16 row not stored).
extern "C" int muse_sandwich_gate_evict_last(const void* const* args, int rows, cudaStream_t stream) {
    launch_gate<true, false, true>(args, rows, stream);
    return cudaGetLastError();
}

// Cache-hint variants for the 512-row prefill (values identical to the plain kernels; only L2 eviction priority of
// the stores changes). mode 0: non-quantizing sandwich (args as muse_sandwich, quantize false); mode 1: folding
// sandwich without the bf16 row (args as muse_sandwich_gate). gate_el / residual_el: store at evict_last priority.
template <bool GateEL, bool ResEL>
int launch_hint(const void* const* args, int rows, int mode, cudaStream_t stream) {
    const size_t smem = Vectors * sizeof(uint2);
    if (mode == 0) {
        sandwich<false, true, false, true, false, ResEL><<<rows, Threads, smem, stream>>>(
            static_cast<uint2*>(const_cast<void*>(args[0])), static_cast<const uint2*>(args[1]),
            static_cast<const uint2*>(args[2]), static_cast<const float*>(args[3]),
            static_cast<const uint2*>(args[4]), static_cast<const uint2*>(args[5]),
            static_cast<const uint2*>(args[6]), nullptr, static_cast<uint2*>(const_cast<void*>(args[7])), nullptr);
    } else {
        sandwich<true, true, true, false, GateEL, ResEL><<<rows, Threads, smem, stream>>>(
            static_cast<uint2*>(const_cast<void*>(args[0])), static_cast<const uint2*>(args[1]),
            static_cast<const uint2*>(args[2]), static_cast<const float*>(args[3]),
            static_cast<const uint2*>(args[4]), static_cast<const uint2*>(args[5]),
            static_cast<const uint2*>(args[6]), static_cast<const float*>(args[7]),
            static_cast<uint2*>(const_cast<void*>(args[8])), static_cast<unsigned*>(const_cast<void*>(args[9])),
            static_cast<const float*>(args[10]), static_cast<unsigned*>(const_cast<void*>(args[11])));
    }
    return cudaGetLastError();
}

extern "C" int muse_sandwich_hint(const void* const* args, int rows, int mode, int gate_el, int residual_el,
                                  cudaStream_t stream) {
    if (mode == 0) {
        return residual_el ? launch_hint<false, true>(args, rows, 0, stream)
                           : launch_hint<false, false>(args, rows, 0, stream);
    }
    if (gate_el) {
        return residual_el ? launch_hint<true, true>(args, rows, 1, stream)
                           : launch_hint<true, false>(args, rows, 1, stream);
    }
    return residual_el ? launch_hint<false, true>(args, rows, 1, stream)
                       : launch_hint<false, false>(args, rows, 1, stream);
}

extern "C" int muse_sandwich_gate_formula(const void* bits, const void* gate_scale, void* out, int n,
                                           cudaStream_t stream) {
    gate_formula<<<(n + 255) / 256, 256, 0, stream>>>(static_cast<const unsigned short*>(bits),
                                                       static_cast<const float*>(gate_scale),
                                                       static_cast<unsigned char*>(out), n);
    return cudaGetLastError();
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
