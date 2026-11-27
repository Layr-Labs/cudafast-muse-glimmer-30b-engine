#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

namespace {
constexpr int Columns = 6656;
constexpr int Threads = 256;
constexpr int Vectors = Columns / 4;
constexpr int Tiles = (Vectors + Threads - 1) / Threads;

static bool g_prefill_cs = true;
static int g_sandwich_pdl_override = -1;

static bool sandwich_pdl_enabled() {
    if (g_sandwich_pdl_override >= 0) return g_sandwich_pdl_override != 0;
    static const bool env_on = []() {
        const char* env = std::getenv("MUSE_SANDWICH_PDL");
        return !env || !*env || std::strtol(env, nullptr, 10) != 0;
    }();
    return env_on;
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;"); }

template <typename... KArgs, typename... Args>
static cudaError_t launch_ex_pdl(void (*kernel)(KArgs...), int rows, int threads, size_t smem, cudaStream_t stream,
                                 Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(rows, 1, 1);
    cfg.blockDim = dim3(threads, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...);
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

__device__ __forceinline__ float reduce_1bar(const float (&partial)[4], float* warp_totals, unsigned thread) {
    float sum = __fadd_rn(__fadd_rn(__fadd_rn(partial[0], partial[1]), partial[2]), partial[3]);
    #pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, delta));
    }
    if ((thread & 31u) == 0u) {
        warp_totals[thread >> 5] = sum;
    }
    __syncthreads();
    sum = warp_totals[thread & 7u];
    #pragma unroll
    for (int delta = 4; delta > 0; delta /= 2) {
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, delta, 8));
    }
    return __shfl_sync(0xffffffff, sum, 0, 8);
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

__device__ __forceinline__ float bounded_fp8(float value, float scale_inv) {
    const float scaled = __fmul_rn(value, scale_inv);
    float r;
    asm("min.NaN.f32 %0, %1, %1;" : "=f"(r) : "f"(scaled));
    return r;
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

__device__ __forceinline__ uint2 load2_cs(const uint2* address) {
    uint2 r;
    asm("ld.global.cs.v2.u32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(address));
    return r;
}

__device__ __forceinline__ uint4 load4_cs(const uint4* address) {
    uint4 r;
    asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(address));
    return r;
}

__device__ __forceinline__ void discard_l2_128(const void* ptr) {
    asm volatile("discard.global.L2 [%0], 128;" :: "l"(ptr) : "memory");
}

template <bool kCs>
__device__ __forceinline__ uint2 ld2_v4(const uint2* row_base, unsigned offset) {
    return kCs ? load2_cs(row_base + offset) : row_base[offset];
}
__device__ __forceinline__ uint2 ldg2_v4(const uint2* base, unsigned offset) {
    return __ldg(base + offset);
}
__device__ __forceinline__ void st2_v4(uint2* row_base, unsigned offset, uint2 val) {
    row_base[offset] = val;
}
__device__ __forceinline__ void st2_v4_evict_last(uint2* row_base, unsigned offset, uint2 val) {
    unsigned long long policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    asm volatile("st.global.L2::cache_hint.v2.b32 [%0], {%1, %2}, %3;"
                 :: "l"(row_base + offset), "r"(val.x), "r"(val.y), "l"(policy) : "memory");
}
__device__ __forceinline__ void st1_v2(unsigned* row_base, unsigned offset, unsigned val) {
    row_base[offset] = val;
}
__device__ __forceinline__ void st1_v2_evict_last(unsigned* row_base, unsigned offset, unsigned val) {
    unsigned long long policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;"
                 :: "l"(row_base + offset), "r"(val), "l"(policy) : "memory");
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
          bool ResidualEvictLast = false, bool CsLoads = false, bool Pdl = false>
__global__ __launch_bounds__(Threads) void sandwich(
    uint2* post_attn, const uint2* down, const uint2* residual,
    const float* previous_sum, const uint2* post_attn_weight,
    const uint2* post_ff_weight, const uint2* next_pre_attn_weight,
    const float* scale, uint2* hidden, unsigned* quantized,
    const float* gate_scale = nullptr, unsigned* gate_quantized = nullptr) {
    constexpr bool UseCs = Shared && CsLoads;
    constexpr bool UsePdl = Shared && Pdl;
    __shared__ float warp_totals[8];
    extern __shared__ uint2 shared_row[];
    uint2 saved[Tiles];
    const unsigned thread = threadIdx.x;
    const size_t base = blockIdx.x * Vectors;
    float h1_pre[Tiles][4];
    float scale_inv_pre = 1.f;
    float gate_inv_pre = 1.f;
    if constexpr (UsePdl) {
        // Preload predecessor-independent inputs (post_attn, residual, weights, scales) and precompute
        // attention_sum BEFORE waiting for down_proj completion; exact same FP32 ops & rounding order.
        const float prev_s_pre = UseCs ? __ldg(previous_sum + blockIdx.x) : previous_sum[blockIdx.x];
        const float attn_inv_pre = inverse_rms(prev_s_pre, 1.e-8f);
        if constexpr (Quantize) scale_inv_pre = full_divide(1.f, UseCs ? __ldg(scale) : *scale);
        if constexpr (Gate) gate_inv_pre = full_divide(1.f, UseCs ? __ldg(gate_scale) : *gate_scale);
        #pragma unroll
        for (int tile = 0; tile < Tiles; ++tile) {
            const unsigned offset = tile * Threads + thread;
            if (offset < Vectors) {
                const uint2 a = UseCs ? load2_cs(post_attn + base + offset) : post_attn[base + offset];
                uint2 r, wa;
                if constexpr (UseCs) {
                    r = load2_cs(residual + base + offset);
                    wa = __ldg(post_attn_weight + offset);
                } else {
                    r = residual[base + offset];
                    wa = post_attn_weight[offset];
                }
                #pragma unroll
                for (int channel = 0; channel < 4; ++channel) {
                    const float attention = apply_norm(bf16(a, channel), attn_inv_pre, bf16(wa, channel));
                    h1_pre[tile][channel] = __fadd_rn(bf16(r, channel), attention);
                }
            }
        }
        pdl_wait();
        pdl_launch_dependents();
    }
    float partial[4] = {};
    #pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        const unsigned offset = tile * Threads + thread;
        if (offset < Vectors) {
            uint2 values;
            if constexpr (UseCs) {
                values = load2_cs(down + base + offset);
            } else {
                values = down[base + offset];
            }
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
    float attn_inverse = 0.f;
    if constexpr (!UsePdl) {
        const float prev_s = UseCs ? __ldg(previous_sum + blockIdx.x) : previous_sum[blockIdx.x];
        attn_inverse = inverse_rms(prev_s, 1.e-8f);
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
            float values[4];
            if constexpr (UsePdl) {
                const uint2 wd = UseCs ? __ldg(post_ff_weight + offset) : post_ff_weight[offset];
                #pragma unroll
                for (int channel = 0; channel < 4; ++channel) {
                    const float feed_forward = __fmul_rn(bf16(d, channel), down_inverse);
                    const float value = __fmaf_rn(feed_forward, __fadd_rn(bf16(wd, channel), 1.f), h1_pre[tile][channel]);
                    values[channel] = value;
                    partial[channel] = __fmaf_rn(value, value, partial[channel]);
                }
            } else {
                const uint2 a = post_attn[base + offset];
                uint2 r, wa, wd;
                if constexpr (UseCs) {
                    r = load2_cs(residual + base + offset);
                    wa = __ldg(post_attn_weight + offset);
                    wd = __ldg(post_ff_weight + offset);
                } else {
                    r = residual[base + offset];
                    wa = post_attn_weight[offset];
                    wd = post_ff_weight[offset];
                }
                #pragma unroll
                for (int channel = 0; channel < 4; ++channel) {
                    const float attention = apply_norm(bf16(a, channel), attn_inverse, bf16(wa, channel));
                    const float feed_forward = __fmul_rn(bf16(d, channel), down_inverse);
                    const float attention_sum = __fadd_rn(bf16(r, channel), attention);
                    const float value = __fmaf_rn(feed_forward, __fadd_rn(bf16(wd, channel), 1.f), attention_sum);
                    values[channel] = value;
                    partial[channel] = __fmaf_rn(value, value, partial[channel]);
                }
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
    const float scale_inv = UsePdl ? scale_inv_pre
                                   : (Quantize ? full_divide(1.f, UseCs ? __ldg(scale) : *scale) : 1.f);
    const float gate_inv = UsePdl ? gate_inv_pre
                                  : (Gate ? full_divide(1.f, UseCs ? __ldg(gate_scale) : *gate_scale) : 1.f);
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
            const uint2 w = UseCs ? __ldg(next_pre_attn_weight + offset) : next_pre_attn_weight[offset];
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

// High-bandwidth 16B uint4 coalesced M=512 prefill sandwich + inline discard.global.L2 on dead `down` (6.82 MB).
// Strict 0-ULP identical FP32/BF16/FP8 arithmetic & reduction order to sandwich<...>.
// Safe for tile 6 because active threads [0..127] are 4 full warps; requires 16B alignment on all row pointers
// (and 128B alignment on `down` when DiscardDown is true; 6656*2 = 13312 = 104 * 128 B per row).
template <bool Quantize, bool Gate, bool WriteHidden, bool GateEvictLast,
          bool ResidualEvictLast, bool CsLoads, bool Pdl, bool DiscardDown>
__global__ __launch_bounds__(Threads) void sandwich_v4(
    uint2* __restrict__ post_attn, const uint2* __restrict__ down, const uint2* __restrict__ residual,
    const float* __restrict__ previous_sum, const uint2* __restrict__ post_attn_weight,
    const uint2* __restrict__ post_ff_weight, const uint2* __restrict__ next_pre_attn_weight,
    const float* __restrict__ scale, uint2* __restrict__ hidden, unsigned* __restrict__ quantized,
    const float* __restrict__ gate_scale = nullptr, unsigned* __restrict__ gate_quantized = nullptr,
    const char* __restrict__ dead_ptr = nullptr, size_t dead_lines = 0) {
    __shared__ float warp_totals[2][8];
    uint2 saved[Tiles];
    const unsigned thread = threadIdx.x;
    const size_t base = static_cast<size_t>(blockIdx.x) * Vectors;
    uint2* const pa_row = post_attn + base;
    const uint2* const dn_row = down + base;
    const uint2* const rs_row = residual + base;
    float h1_pre[Pdl ? Tiles : 1][4];
    float scale_inv_pre = 1.f;
    float gate_inv_pre = 1.f;
    if constexpr (Pdl) {
        const float prev_s_pre = __ldg(previous_sum + blockIdx.x);
        const float attn_inv_pre = inverse_rms(prev_s_pre, 1.e-8f);
        if constexpr (Quantize) scale_inv_pre = full_divide(1.f, __ldg(scale));
        if constexpr (Gate) gate_inv_pre = full_divide(1.f, __ldg(gate_scale));
        #pragma unroll
        for (int tile = 0; tile < Tiles; ++tile) {
            const unsigned offset = tile * Threads + thread;
            if (offset < Vectors) {
                const uint2 a = ld2_v4<CsLoads>(pa_row, offset);
                const uint2 r = ld2_v4<CsLoads>(rs_row, offset);
                const uint2 wa = ldg2_v4(post_attn_weight, offset);
                #pragma unroll
                for (int channel = 0; channel < 4; ++channel) {
                    const float attention = apply_norm(bf16(a, channel), attn_inv_pre, bf16(wa, channel));
                    h1_pre[tile][channel] = __fadd_rn(bf16(r, channel), attention);
                }
            }
        }
        pdl_wait();
        pdl_launch_dependents();
    }
    float partial[4] = {};
    #pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        const unsigned offset = tile * Threads + thread;
        if (offset < Vectors) {
            const uint2 values = ld2_v4<CsLoads>(dn_row, offset);
            saved[tile] = values;
            #pragma unroll
            for (int channel = 0; channel < 4; ++channel) {
                const float value = bf16(values, channel);
                partial[channel] = __fmaf_rn(value, value, partial[channel]);
            }
        }
    }
    const float down_inverse = inverse_rms(reduce_1bar(partial, warp_totals[0], thread), 1.e-8f);
    if constexpr (DiscardDown) {
        // reduce() already synchronized the CTA; all 104 cache lines (13312 B) of dn_row are now dead.
        if (thread < 104u) {
            discard_l2_128(reinterpret_cast<const char*>(dn_row) + (thread << 7));
        }
        if (dead_ptr) {
            const size_t gid = static_cast<size_t>(blockIdx.x) * Threads + thread;
            const size_t gstride = static_cast<size_t>(gridDim.x) * Threads;
            for (size_t i = gid; i < dead_lines; i += gstride) {
                discard_l2_128(dead_ptr + (i << 7));
            }
        }
    }
    float attn_inverse = 0.f;
    if constexpr (!Pdl) {
        attn_inverse = inverse_rms(__ldg(previous_sum + blockIdx.x), 1.e-8f);
    }
    #pragma unroll
    for (int channel = 0; channel < 4; ++channel) partial[channel] = 0.f;
    #pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        const unsigned offset = tile * Threads + thread;
        if (offset < Vectors) {
            const uint2 d = saved[tile];
            float values[4];
            if constexpr (Pdl) {
                const uint2 wd = ldg2_v4(post_ff_weight, offset);
                #pragma unroll
                for (int channel = 0; channel < 4; ++channel) {
                    const float feed_forward = __fmul_rn(bf16(d, channel), down_inverse);
                    const float value = __fmaf_rn(feed_forward, __fadd_rn(bf16(wd, channel), 1.f), h1_pre[tile][channel]);
                    values[channel] = value;
                    partial[channel] = __fmaf_rn(value, value, partial[channel]);
                }
            } else {
                const uint2 a = ld2_v4<CsLoads>(pa_row, offset);
                const uint2 r = ld2_v4<CsLoads>(rs_row, offset);
                const uint2 wa = ldg2_v4(post_attn_weight, offset);
                const uint2 wd = ldg2_v4(post_ff_weight, offset);
                #pragma unroll
                for (int channel = 0; channel < 4; ++channel) {
                    const float attention = apply_norm(bf16(a, channel), attn_inverse, bf16(wa, channel));
                    const float feed_forward = __fmul_rn(bf16(d, channel), down_inverse);
                    const float attention_sum = __fadd_rn(bf16(r, channel), attention);
                    const float value = __fmaf_rn(feed_forward, __fadd_rn(bf16(wd, channel), 1.f), attention_sum);
                    values[channel] = value;
                    partial[channel] = __fmaf_rn(value, value, partial[channel]);
                }
            }
            const uint2 rounded = make_uint2(pack_bf16(values[0], values[1]),
                                             pack_bf16(values[2], values[3]));
            if constexpr (ResidualEvictLast) {
                st2_v4_evict_last(pa_row, offset, rounded);
            } else {
                st2_v4(pa_row, offset, rounded);
            }
            saved[tile] = rounded;
        }
    }
    const float input_inverse = inverse_rms(reduce_1bar(partial, warp_totals[1], thread), 1.e-5f);
    const float scale_inv = Pdl ? scale_inv_pre
                                : (Quantize ? full_divide(1.f, __ldg(scale)) : 1.f);
    const float gate_inv = Pdl ? gate_inv_pre
                               : (Gate ? full_divide(1.f, __ldg(gate_scale)) : 1.f);
    uint2* const hid_row = WriteHidden ? (hidden + base) : nullptr;
    unsigned* const q_row = Quantize ? (quantized + base) : nullptr;
    unsigned* const g_row = Gate ? (gate_quantized + base) : nullptr;
    #pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        const unsigned offset = tile * Threads + thread;
        if (offset < Vectors) {
            const uint2 r = saved[tile];
            const uint2 w = ldg2_v4(next_pre_attn_weight, offset);
            float values[4];
            #pragma unroll
            for (int channel = 0; channel < 4; ++channel) {
                values[channel] = apply_norm(bf16(r, channel), input_inverse, bf16(w, channel));
            }
            const uint2 rounded_hidden = make_uint2(pack_bf16(values[0], values[1]),
                                                    pack_bf16(values[2], values[3]));
            if constexpr (WriteHidden) {
                st2_v4(hid_row, offset, rounded_hidden);
            }
            if constexpr (Quantize) {
                st1_v2(q_row, offset, pack_fp8(values, scale_inv));
            }
            if constexpr (Gate) {
                float gate_values[4];
                #pragma unroll
                for (int channel = 0; channel < 4; ++channel) {
                    gate_values[channel] = bf16(rounded_hidden, channel);
                }
                if constexpr (GateEvictLast) {
                    st1_v2_evict_last(g_row, offset, pack_fp8(gate_values, gate_inv));
                } else {
                    st1_v2(g_row, offset, pack_fp8(gate_values, gate_inv));
                }
            }
        }
    }
}

static int g_sandwich_v4_override = -1;
static bool sandwich_v4_enabled() {
    if (g_sandwich_v4_override >= 0) return g_sandwich_v4_override != 0;
    static const bool env_on = []() {
        const char* env = std::getenv("MUSE_SANDWICH_V4");
        return !env || !*env || std::strtol(env, nullptr, 10) != 0;
    }();
    return env_on;
}

static int g_sandwich_discard_override = -1;
static bool sandwich_discard_enabled() {
    if (g_sandwich_discard_override >= 0) return g_sandwich_discard_override != 0;
    static const bool env_on = []() {
        const char* env = std::getenv("MUSE_SANDWICH_DISCARD_L2");
        return !env || !*env || std::strtol(env, nullptr, 10) != 0;
    }();
    return env_on;
}

static inline bool aligned16_row_ptrs(const void* a0, const void* a1, const void* a2,
                                      const void* a4, const void* a5, const void* a6,
                                      const void* a8, const void* a9, const void* a11) {
    uintptr_t u = reinterpret_cast<uintptr_t>(a0) | reinterpret_cast<uintptr_t>(a1)
                | reinterpret_cast<uintptr_t>(a2) | reinterpret_cast<uintptr_t>(a4)
                | reinterpret_cast<uintptr_t>(a5) | reinterpret_cast<uintptr_t>(a6);
    if (a8) u |= reinterpret_cast<uintptr_t>(a8);
    if ((u & 15u) != 0u) return false;
    if (a9 && (reinterpret_cast<uintptr_t>(a9) & 7u) != 0u) return false;
    if (a11 && (reinterpret_cast<uintptr_t>(a11) & 7u) != 0u) return false;
    return true;
}

static const void* g_next_dead_ptr = nullptr;
static size_t g_next_dead_bytes = 0;

static inline void align_dead_128(const void* ptr, size_t nbytes, const char*& out_p, size_t& out_lines) {
    out_p = nullptr;
    out_lines = 0;
    if (!ptr || nbytes < 128u) return;
    const uintptr_t p0 = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t s = (p0 + 127u) & ~uintptr_t(127u);
    const uintptr_t e = (p0 + nbytes) & ~uintptr_t(127u);
    if (e > s) {
        out_p = reinterpret_cast<const char*>(s);
        out_lines = static_cast<size_t>((e - s) >> 7);
    }
}

template <bool Quantize, bool Gate, bool WriteHidden, bool GateEvictLast, bool ResidualEvictLast, bool CsLoads>
static inline cudaError_t dispatch_v4(
    int rows, cudaStream_t stream,
    uint2* a0, const uint2* a1, const uint2* a2, const float* a3,
    const uint2* a4, const uint2* a5, const uint2* a6,
    const float* a7, uint2* a8, unsigned* a9,
    const float* a10, unsigned* a11) {
    const void* raw_dp = g_next_dead_ptr;
    const size_t raw_db = g_next_dead_bytes;
    g_next_dead_ptr = nullptr;
    g_next_dead_bytes = 0;
    const char* dp = nullptr;
    size_t dl = 0;
    const bool use_pdl = sandwich_pdl_enabled();
    const bool disc = sandwich_discard_enabled() && ((reinterpret_cast<uintptr_t>(a1) & 127u) == 0u);
    if (disc && rows == 512) align_dead_128(raw_dp, raw_db, dp, dl);
    if (use_pdl) {
        if (disc) return launch_ex_pdl(sandwich_v4<Quantize, Gate, WriteHidden, GateEvictLast, ResidualEvictLast, CsLoads, true, true>, rows, Threads, 0, stream, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, dp, dl);
        return launch_ex_pdl(sandwich_v4<Quantize, Gate, WriteHidden, GateEvictLast, ResidualEvictLast, CsLoads, true, false>, rows, Threads, 0, stream, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, static_cast<const char*>(nullptr), size_t(0));
    }
    if (disc) sandwich_v4<Quantize, Gate, WriteHidden, GateEvictLast, ResidualEvictLast, CsLoads, false, true><<<rows, Threads, 0, stream>>>(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, dp, dl);
    else      sandwich_v4<Quantize, Gate, WriteHidden, GateEvictLast, ResidualEvictLast, CsLoads, false, false><<<rows, Threads, 0, stream>>>(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, nullptr, 0);
    return cudaGetLastError();
}

template <bool Quantize, bool Shared, bool CsLoads = false>
void launch(const void* const* args, int rows, cudaStream_t stream) {
    auto* a0 = static_cast<uint2*>(const_cast<void*>(args[0]));
    const auto* a1 = static_cast<const uint2*>(args[1]);
    const auto* a2 = static_cast<const uint2*>(args[2]);
    const auto* a3 = static_cast<const float*>(args[3]);
    const auto* a4 = static_cast<const uint2*>(args[4]);
    const auto* a5 = static_cast<const uint2*>(args[5]);
    const auto* a6 = static_cast<const uint2*>(args[6]);
    const auto* a7 = Quantize ? static_cast<const float*>(args[7]) : nullptr;
    auto* a8 = static_cast<uint2*>(const_cast<void*>(args[Quantize ? 8 : 7]));
    auto* a9 = Quantize ? static_cast<unsigned*>(const_cast<void*>(args[9])) : nullptr;
    if constexpr (Shared) {
        if (sandwich_v4_enabled() && aligned16_row_ptrs(a0, a1, a2, a4, a5, a6, a8, a9, nullptr)) {
            (void)dispatch_v4<Quantize, false, true, false, false, CsLoads>(
                rows, stream, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, nullptr, nullptr);
            return;
        }
        if (sandwich_pdl_enabled()) {
            launch_ex_pdl(sandwich<Quantize, true, false, true, false, false, CsLoads, true>, rows, Threads,
                          Vectors * sizeof(uint2), stream, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9,
                          static_cast<const float*>(nullptr), static_cast<unsigned*>(nullptr));
            return;
        }
    }
    sandwich<Quantize, Shared, false, true, false, false, CsLoads, false><<<rows, Threads, Shared ? Vectors * sizeof(uint2) : 0, stream>>>(
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
}
}

extern "C" void muse_sandwich_set_prefill_cs(int enable) {
    g_prefill_cs = (enable != 0);
}

extern "C" int muse_sandwich_set_pdl(int enable) {
    const int old = sandwich_pdl_enabled() ? 1 : 0;
    g_sandwich_pdl_override = (enable < 0) ? -1 : (enable != 0 ? 1 : 0);
    return old;
}

extern "C" int muse_sandwich_set_v4(int enable) {
    const int old = sandwich_v4_enabled() ? 1 : 0;
    g_sandwich_v4_override = (enable < 0) ? -1 : (enable != 0 ? 1 : 0);
    return old;
}

extern "C" void muse_sandwich_set_next_dead(const void* ptr, size_t nbytes) {
    g_next_dead_ptr = ptr;
    g_next_dead_bytes = nbytes;
}

extern "C" int muse_sandwich_set_discard(int enable) {
    const int old = sandwich_discard_enabled() ? 1 : 0;
    g_sandwich_discard_override = (enable < 0) ? -1 : (enable != 0 ? 1 : 0);
    return old;
}

template <bool Shared, bool WriteHidden, bool EvictLast = false, bool CsLoads = false>
void launch_gate(const void* const* args, int rows, cudaStream_t stream) {
    auto* a0 = static_cast<uint2*>(const_cast<void*>(args[0]));
    const auto* a1 = static_cast<const uint2*>(args[1]);
    const auto* a2 = static_cast<const uint2*>(args[2]);
    const auto* a3 = static_cast<const float*>(args[3]);
    const auto* a4 = static_cast<const uint2*>(args[4]);
    const auto* a5 = static_cast<const uint2*>(args[5]);
    const auto* a6 = static_cast<const uint2*>(args[6]);
    const auto* a7 = static_cast<const float*>(args[7]);
    auto* a8 = static_cast<uint2*>(const_cast<void*>(args[8]));
    auto* a9 = static_cast<unsigned*>(const_cast<void*>(args[9]));
    const auto* a10 = static_cast<const float*>(args[10]);
    auto* a11 = static_cast<unsigned*>(const_cast<void*>(args[11]));
    if constexpr (Shared) {
        if (sandwich_v4_enabled() && aligned16_row_ptrs(a0, a1, a2, a4, a5, a6, WriteHidden ? a8 : nullptr, a9, a11)) {
            (void)dispatch_v4<true, true, WriteHidden, EvictLast, false, CsLoads>(
                rows, stream, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            return;
        }
        if (sandwich_pdl_enabled()) {
            launch_ex_pdl(sandwich<true, true, true, WriteHidden, EvictLast, false, CsLoads, true>, rows, Threads,
                          Vectors * sizeof(uint2), stream, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            return;
        }
    }
    sandwich<true, Shared, true, WriteHidden, EvictLast, false, CsLoads, false><<<rows, Threads, Shared ? Vectors * sizeof(uint2) : 0, stream>>>(
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
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
        if (g_prefill_cs) {
            if (write_hidden) launch_gate<true, true, false, true>(args, rows, stream);
            else launch_gate<true, false, false, true>(args, rows, stream);
        } else {
            if (write_hidden) launch_gate<true, true, false, false>(args, rows, stream);
            else launch_gate<true, false, false, false>(args, rows, stream);
        }
    } else {
        if (write_hidden) launch_gate<false, true>(args, rows, stream);
        else launch_gate<false, false>(args, rows, stream);
    }
    return cudaGetLastError();
}

// The same with the gate operand stored at L2 evict_last priority (shared-row staging, bf16 row not stored).
extern "C" int muse_sandwich_gate_evict_last(const void* const* args, int rows, cudaStream_t stream) {
    if (g_prefill_cs) launch_gate<true, false, true, true>(args, rows, stream);
    else launch_gate<true, false, true, false>(args, rows, stream);
    return cudaGetLastError();
}

// Cache-hint variants for the 512-row prefill (values identical to the plain kernels; only L2 eviction priority of
// the stores changes). mode 0: non-quantizing sandwich (args as muse_sandwich, quantize false); mode 1: folding
// sandwich without the bf16 row (args as muse_sandwich_gate). gate_el / residual_el: store at evict_last priority.
template <bool GateEL, bool ResEL, bool CsLoads>
int launch_hint(const void* const* args, int rows, int mode, cudaStream_t stream) {
    const size_t smem = Vectors * sizeof(uint2);
    const bool use_pdl = sandwich_pdl_enabled();
    if (mode == 0) {
        auto* a0 = static_cast<uint2*>(const_cast<void*>(args[0]));
        const auto* a1 = static_cast<const uint2*>(args[1]);
        const auto* a2 = static_cast<const uint2*>(args[2]);
        const auto* a3 = static_cast<const float*>(args[3]);
        const auto* a4 = static_cast<const uint2*>(args[4]);
        const auto* a5 = static_cast<const uint2*>(args[5]);
        const auto* a6 = static_cast<const uint2*>(args[6]);
        auto* a7 = static_cast<uint2*>(const_cast<void*>(args[7]));
        if (sandwich_v4_enabled() && aligned16_row_ptrs(a0, a1, a2, a4, a5, a6, a7, nullptr, nullptr)) {
            return static_cast<int>(dispatch_v4<false, false, true, false, ResEL, CsLoads>(
                rows, stream, a0, a1, a2, a3, a4, a5, a6, nullptr, a7, nullptr, nullptr, nullptr));
        }
        if (use_pdl) {
            return static_cast<int>(launch_ex_pdl(
                sandwich<false, true, false, true, false, ResEL, CsLoads, true>, rows, Threads, smem, stream,
                a0, a1, a2, a3, a4, a5, a6, static_cast<const float*>(nullptr), a7,
                static_cast<unsigned*>(nullptr), static_cast<const float*>(nullptr), static_cast<unsigned*>(nullptr)));
        }
        sandwich<false, true, false, true, false, ResEL, CsLoads, false><<<rows, Threads, smem, stream>>>(
            a0, a1, a2, a3, a4, a5, a6, nullptr, a7, nullptr);
    } else {
        auto* a0 = static_cast<uint2*>(const_cast<void*>(args[0]));
        const auto* a1 = static_cast<const uint2*>(args[1]);
        const auto* a2 = static_cast<const uint2*>(args[2]);
        const auto* a3 = static_cast<const float*>(args[3]);
        const auto* a4 = static_cast<const uint2*>(args[4]);
        const auto* a5 = static_cast<const uint2*>(args[5]);
        const auto* a6 = static_cast<const uint2*>(args[6]);
        const auto* a7 = static_cast<const float*>(args[7]);
        auto* a8 = static_cast<uint2*>(const_cast<void*>(args[8]));
        auto* a9 = static_cast<unsigned*>(const_cast<void*>(args[9]));
        const auto* a10 = static_cast<const float*>(args[10]);
        auto* a11 = static_cast<unsigned*>(const_cast<void*>(args[11]));
        if (sandwich_v4_enabled() && aligned16_row_ptrs(a0, a1, a2, a4, a5, a6, nullptr, a9, a11)) {
            return static_cast<int>(dispatch_v4<true, true, false, GateEL, ResEL, CsLoads>(
                rows, stream, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11));
        }
        if (use_pdl) {
            return static_cast<int>(launch_ex_pdl(
                sandwich<true, true, true, false, GateEL, ResEL, CsLoads, true>, rows, Threads, smem, stream,
                a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11));
        }
        sandwich<true, true, true, false, GateEL, ResEL, CsLoads, false><<<rows, Threads, smem, stream>>>(
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
    }
    return cudaGetLastError();
}

extern "C" int muse_sandwich_hint(const void* const* args, int rows, int mode, int gate_el, int residual_el,
                                  cudaStream_t stream) {
    if (mode == 0) {
        if (g_prefill_cs) {
            return residual_el ? launch_hint<false, true, true>(args, rows, 0, stream)
                               : launch_hint<false, false, true>(args, rows, 0, stream);
        }
        return residual_el ? launch_hint<false, true, false>(args, rows, 0, stream)
                           : launch_hint<false, false, false>(args, rows, 0, stream);
    }
    if (g_prefill_cs) {
        if (gate_el) {
            return residual_el ? launch_hint<true, true, true>(args, rows, 1, stream)
                               : launch_hint<true, false, true>(args, rows, 1, stream);
        }
        return residual_el ? launch_hint<false, true, true>(args, rows, 1, stream)
                           : launch_hint<false, false, true>(args, rows, 1, stream);
    }
    if (gate_el) {
        return residual_el ? launch_hint<true, true, false>(args, rows, 1, stream)
                           : launch_hint<true, false, false>(args, rows, 1, stream);
    }
    return residual_el ? launch_hint<false, true, false>(args, rows, 1, stream)
                       : launch_hint<false, false, false>(args, rows, 1, stream);
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
        if (shared) {
            if (g_prefill_cs) launch<true, true, true>(args, rows, stream);
            else launch<true, true, false>(args, rows, stream);
        } else {
            launch<true, false>(args, rows, stream);
        }
    } else {
        if (shared) {
            if (g_prefill_cs) launch<false, true, true>(args, rows, stream);
            else launch<false, true, false>(args, rows, stream);
        } else {
            launch<false, false>(args, rows, stream);
        }
    }
    return cudaGetLastError();
}
