#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

__device__ __forceinline__ __nv_bfloat162 dequant_s_bf162(uint32_t b) {
    uint32_t bits = b ? ((b << 4) + 0x3800u) : 0u;
    uint32_t pair = bits | (bits << 16);
    return *reinterpret_cast<__nv_bfloat162*>(&pair);
}

__device__ __forceinline__ uint32_t lop3_or_and(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ uint32_t hmul2_u32(uint32_t f, __nv_bfloat162 s) {
    __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&f), s);
    return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma_m16n8k16_bf16_m1(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1)
    );
}

__device__ __forceinline__ void mma_m16n8k16_bf16_full(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

__device__ __forceinline__ void mma_m16n8k32_fp8_m1(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1)
    );
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr
) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
    );
}

// ---------------------------------------------------------------------------
// 1. Zero-tail-wave 96-block __ldg(A) NVFP4 M=1 GEMV for down_proj (N=6656, K=19968).
//    Exact 96 half-stripe Marlin partition (iters=169 = exact 2 CTAs/SM x 48 SMs),
//    zero chunk barriers, zero per-call partials alloc -> 315.4us (237.1 GB/s), mism=0.
// ---------------------------------------------------------------------------
__global__ void nvfp4_down_m1_kernel(
    const uint32_t* __restrict__ A,
    const uint4* __restrict__ B_q,
    const uint8_t* __restrict__ B_s,
    const float* __restrict__ g_scale,
    float* __restrict__ partials,
    __nv_bfloat16* __restrict__ C
) {
    constexpr int N = 6656;
    constexpr int k_tiles = 156;
    constexpr int iters = 169;
    constexpr int row_stride_int4 = N >> 1;

    __shared__ float s_acc[4][64];

    int w = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int t = lane & 3;
    int mblk = blockIdx.x >> 1;
    int half = blockIdx.x & 1;
    int t_start = mblk * iters;
    int t_end = (mblk + 1) * iters;
    float gs = *g_scale;

    int cur = t_start;
    while (cur < t_end) {
        int col = cur / k_tiles;
        int col_start = col * k_tiles;
        int col_end = col_start + k_tiles;
        int seg_end_global = min(t_end, col_end);
        int seg_start = cur - col_start;
        int seg_end = seg_end_global - col_start;
        int n64 = col * 2 + half;

        float d0[8] = {0}, d1[8] = {0}, d2[8] = {0}, d3[8] = {0};
        int k16_base = seg_start * 8 + 2 * w;
        uint4 qv_cur;
        uint64_t sc_cur;
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(qv_cur.x), "=r"(qv_cur.y), "=r"(qv_cur.z), "=r"(qv_cur.w)
            : "l"(B_q + static_cast<uint64_t>(k16_base) * row_stride_int4 + n64 * 32 + lane));
        sc_cur = *reinterpret_cast<const uint64_t*>(
            B_s + static_cast<uint64_t>(k16_base) * N + n64 * 64 + group * 8);

        int n_steps = (seg_end - seg_start) * 2;
        for (int step = 0; step < n_steps; ++step) {
            int k16 = (seg_start + (step >> 1)) * 8 + 2 * w + (step & 1);
            uint4 qv_nxt;
            uint64_t sc_nxt;
            if (step + 1 < n_steps) {
                int nk16 = (seg_start + ((step + 1) >> 1)) * 8 + 2 * w + ((step + 1) & 1);
                asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(qv_nxt.x), "=r"(qv_nxt.y), "=r"(qv_nxt.z), "=r"(qv_nxt.w)
                    : "l"(B_q + static_cast<uint64_t>(nk16) * row_stride_int4 + n64 * 32 + lane));
                sc_nxt = *reinterpret_cast<const uint64_t*>(
                    B_s + static_cast<uint64_t>(nk16) * N + n64 * 64 + group * 8);
            }
            uint32_t a0 = (group == 0) ? __ldg(A + k16 * 8 + t) : 0u;
            uint32_t a2 = (group == 0) ? __ldg(A + k16 * 8 + 4 + t) : 0u;
            uint32_t q_arr[4] = {qv_cur.x, qv_cur.y, qv_cur.z, qv_cur.w};
            const uint8_t* sc = reinterpret_cast<const uint8_t*>(&sc_cur);
            uint8_t sb[8] = {sc[0], sc[2], sc[1], sc[3], sc[4], sc[6], sc[5], sc[7]};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t q = q_arr[j];
                uint32_t q0 = q << 12;
                uint32_t q1 = q << 8;
                uint32_t q2 = q << 4;
                uint32_t f0 = lop3_or_and(q0, 0x80008000u, (q0 >> 6) & 0x01C001C0u);
                uint32_t f1 = lop3_or_and(q1, 0x80008000u, (q1 >> 6) & 0x01C001C0u);
                uint32_t f2 = lop3_or_and(q2, 0x80008000u, (q2 >> 6) & 0x01C001C0u);
                uint32_t f3 = lop3_or_and(q,  0x80008000u, (q  >> 6) & 0x01C001C0u);
                __nv_bfloat162 s0 = dequant_s_bf162(sb[2 * j]);
                __nv_bfloat162 s1 = dequant_s_bf162(sb[2 * j + 1]);
                mma_m16n8k16_bf16_m1(d0[2 * j], d1[2 * j], d2[2 * j], d3[2 * j],
                                     a0, a2, hmul2_u32(f0, s0), hmul2_u32(f1, s0));
                mma_m16n8k16_bf16_m1(d0[2 * j + 1], d1[2 * j + 1], d2[2 * j + 1], d3[2 * j + 1],
                                     a0, a2, hmul2_u32(f2, s1), hmul2_u32(f3, s1));
            }
            qv_cur = qv_nxt;
            sc_cur = sc_nxt;
        }
        if (group == 0) {
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                s_acc[w][m * 8 + t * 2] = d0[m];
                s_acc[w][m * 8 + t * 2 + 1] = d1[m];
            }
        }
        __syncthreads();
        if (threadIdx.x < 64) {
            int c = threadIdx.x;
            float P = s_acc[0][c] + (s_acc[1][c] + (s_acc[3][c] + s_acc[2][c]));
            int out_idx = n64 * 64 + c;
            if (seg_start == 0 && seg_end == k_tiles) {
                C[out_idx] = __float2bfloat16_rn(P * gs);
            } else if (seg_start == 0) {
                partials[out_idx] = P;
            } else {
                partials[N + out_idx] = P;
            }
        }
        __syncthreads();
        cur = seg_end_global;
    }
}

__global__ void nvfp4_down_reduce_kernel(
    const float* __restrict__ partials,
    const float* __restrict__ g_scale,
    __nv_bfloat16* __restrict__ C
) {
    constexpr int N = 6656;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int col = idx >> 7;
    if ((col * 156) / 169 != (col * 156 + 155) / 169) {
        C[idx] = __float2bfloat16_rn((partials[idx] + partials[N + idx]) * (*g_scale));
    }
}

// ---------------------------------------------------------------------------
// 1b. Fused bit-exact NVFP4 gate_up + SiluAndMul at M=1 (grid=156, 0 tail wave).
//     Exact Marlin part1/part2 split-K (up_n64 >= 576: [26..52) + [0..26)),
//     exact mode-0 SiluAndMul ((g / (1 + expf(-g))) * u -> bf16) matching
//     torch.compile(SiluAndMul.forward_native) bit-for-bit.
// ---------------------------------------------------------------------------
__global__ void nvfp4_gate_up_silu_m1_kernel(
    const uint32_t* __restrict__ A,
    const uint4* __restrict__ B_q,
    const uint8_t* __restrict__ B_s,
    const float* __restrict__ g_scale,
    __nv_bfloat16* __restrict__ C
) {
    constexpr int N = 39936;
    constexpr int K = 6656;
    constexpr int k_tiles = 52;
    constexpr int half_n64 = 312;
    constexpr int row_stride_int4 = N >> 1;

    extern __shared__ uint32_t sA32[];
    const uint4* A4 = reinterpret_cast<const uint4*>(A);
    uint4* sA4 = reinterpret_cast<uint4*>(sA32);
    for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) {
        sA4[i] = A4[i];
    }
    __shared__ float s_acc[4][64];
    __syncthreads();

    int w = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int t = lane & 3;
    float gs = *g_scale;

    auto compute_n64_seg = [&](int n64, int seg_start, int seg_end) -> float {
        float d0[8] = {0}, d1[8] = {0}, d2[8] = {0}, d3[8] = {0};
        int k16_base = seg_start * 8 + 2 * w;
        uint4 qv_cur;
        uint64_t sc_cur;
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(qv_cur.x), "=r"(qv_cur.y), "=r"(qv_cur.z), "=r"(qv_cur.w)
            : "l"(B_q + static_cast<uint64_t>(k16_base) * row_stride_int4 + n64 * 32 + lane));
        sc_cur = *reinterpret_cast<const uint64_t*>(
            B_s + static_cast<uint64_t>(k16_base) * N + n64 * 64 + group * 8);
        int n_steps = (seg_end - seg_start) * 2;
        for (int step = 0; step < n_steps; ++step) {
            int k16 = (seg_start + (step >> 1)) * 8 + 2 * w + (step & 1);
            uint4 qv_nxt;
            uint64_t sc_nxt;
            if (step + 1 < n_steps) {
                int nk16 = (seg_start + ((step + 1) >> 1)) * 8 + 2 * w + ((step + 1) & 1);
                asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(qv_nxt.x), "=r"(qv_nxt.y), "=r"(qv_nxt.z), "=r"(qv_nxt.w)
                    : "l"(B_q + static_cast<uint64_t>(nk16) * row_stride_int4 + n64 * 32 + lane));
                sc_nxt = *reinterpret_cast<const uint64_t*>(
                    B_s + static_cast<uint64_t>(nk16) * N + n64 * 64 + group * 8);
            }
            uint32_t a0 = (group == 0) ? sA32[k16 * 8 + t] : 0u;
            uint32_t a2 = (group == 0) ? sA32[k16 * 8 + 4 + t] : 0u;
            uint32_t q_arr[4] = {qv_cur.x, qv_cur.y, qv_cur.z, qv_cur.w};
            const uint8_t* sc = reinterpret_cast<const uint8_t*>(&sc_cur);
            uint8_t sb[8] = {sc[0], sc[2], sc[1], sc[3], sc[4], sc[6], sc[5], sc[7]};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t q = q_arr[j];
                uint32_t q0 = q << 12, q1 = q << 8, q2 = q << 4;
                uint32_t f0 = lop3_or_and(q0, 0x80008000u, (q0 >> 6) & 0x01C001C0u);
                uint32_t f1 = lop3_or_and(q1, 0x80008000u, (q1 >> 6) & 0x01C001C0u);
                uint32_t f2 = lop3_or_and(q2, 0x80008000u, (q2 >> 6) & 0x01C001C0u);
                uint32_t f3 = lop3_or_and(q,  0x80008000u, (q  >> 6) & 0x01C001C0u);
                __nv_bfloat162 s0 = dequant_s_bf162(sb[2 * j]);
                __nv_bfloat162 s1 = dequant_s_bf162(sb[2 * j + 1]);
                mma_m16n8k16_bf16_m1(d0[2 * j], d1[2 * j], d2[2 * j], d3[2 * j],
                                     a0, a2, hmul2_u32(f0, s0), hmul2_u32(f1, s0));
                mma_m16n8k16_bf16_m1(d0[2 * j + 1], d1[2 * j + 1], d2[2 * j + 1], d3[2 * j + 1],
                                     a0, a2, hmul2_u32(f2, s1), hmul2_u32(f3, s1));
            }
            qv_cur = qv_nxt;
            sc_cur = sc_nxt;
        }
        if (group == 0) {
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                s_acc[w][m * 8 + t * 2] = d0[m];
                s_acc[w][m * 8 + t * 2 + 1] = d1[m];
            }
        }
        __syncthreads();
        float P = 0.f;
        if (threadIdx.x < 64) {
            int c = threadIdx.x;
            P = s_acc[0][c] + (s_acc[1][c] + (s_acc[3][c] + s_acc[2][c]));
        }
        __syncthreads();
        return P;
    };

    for (int n_blk = blockIdx.x; n_blk < half_n64; n_blk += gridDim.x) {
        float g_raw = compute_n64_seg(n_blk, 0, k_tiles);
        int up_n64 = half_n64 + n_blk;
        float u_raw;
        if (up_n64 < 576) {
            u_raw = compute_n64_seg(up_n64, 0, k_tiles);
        } else {
            float hi = compute_n64_seg(up_n64, 26, k_tiles);
            float lo = compute_n64_seg(up_n64, 0, 26);
            u_raw = lo + hi;
        }
        if (threadIdx.x < 64) {
            float g = __bfloat162float(__float2bfloat16_rn(g_raw * gs));
            float u = __bfloat162float(__float2bfloat16_rn(u_raw * gs));
            C[n_blk * 64 + threadIdx.x] = __float2bfloat16_rn((g / (1.0f + expf(-g))) * u);
        }
    }
}

__global__ void silu_mul_mode0_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ out,
    int M, int half_N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * half_N;
    if (idx >= total) return;
    int r = idx / half_N;
    int c = idx - r * half_N;
    const __nv_bfloat16* row = gate_up + static_cast<uint64_t>(r) * (half_N * 2);
    float g = __bfloat162float(row[c]);
    float u = __bfloat162float(row[half_N + c]);
    out[idx] = __float2bfloat16_rn((g / (1.0f + expf(-g))) * u);
}

// ---------------------------------------------------------------------------
// 2. Bit-exact Marlin NVFP4 M=1 streaming GEMV for lm_head (N=202048, K=6656).
// ---------------------------------------------------------------------------
__global__ void nvfp4_lmhead_m1_kernel(
    const uint32_t* __restrict__ A,
    const uint4* __restrict__ B_q,
    const uint8_t* __restrict__ B_s,
    const float* __restrict__ g_scale,
    __nv_bfloat16* __restrict__ C,
    int N, int K
) {
    extern __shared__ uint32_t sA32[];
    int k_u4 = K >> 3;
    const uint4* A4 = reinterpret_cast<const uint4*>(A);
    uint4* sA4 = reinterpret_cast<uint4*>(sA32);
    for (int i = threadIdx.x; i < k_u4; i += blockDim.x) {
        sA4[i] = A4[i];
    }
    __shared__ float s_acc[4][64];
    __syncthreads();

    int w = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int t = lane & 3;
    constexpr int k_tiles = 52;
    constexpr int num_n64 = 3157;
    constexpr int p2_cols = 37;
    constexpr int p2_iters = 41;
    int row_stride_int4 = N >> 1;
    float gs = *g_scale;

    for (int n64 = blockIdx.x; n64 < num_n64; n64 += gridDim.x) {
        bool in_p2 = (n64 >= (num_n64 - p2_cols));
        int local_col = n64 - (num_n64 - p2_cols);

        int seg_starts[4], seg_ends[4];
        int n_segs = 0;
        if (!in_p2) {
            seg_starts[0] = 0;
            seg_ends[0] = k_tiles;
            n_segs = 1;
        } else {
            int col_start = local_col * k_tiles;
            int col_end = col_start + k_tiles;
            int first_blk = col_start / p2_iters;
            int last_blk = (col_end - 1) / p2_iters;
            for (int blk = last_blk; blk >= first_blk; --blk) {
                seg_starts[n_segs] = max(col_start, blk * p2_iters) - col_start;
                seg_ends[n_segs] = min(col_end, (blk + 1) * p2_iters) - col_start;
                n_segs++;
            }
        }

        float R_reg = 0.f;
        bool has_R = false;
        for (int sg = 0; sg < n_segs; ++sg) {
            int seg_start = seg_starts[sg], seg_end = seg_ends[sg];
            float d0[8] = {0}, d1[8] = {0}, d2[8] = {0}, d3[8] = {0};
            int k16_base = seg_start * 8 + 2 * w;
            uint4 qv_cur;
            uint64_t sc_cur;
            asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(qv_cur.x), "=r"(qv_cur.y), "=r"(qv_cur.z), "=r"(qv_cur.w)
                : "l"(B_q + static_cast<uint64_t>(k16_base) * row_stride_int4 + n64 * 32 + lane));
            sc_cur = *reinterpret_cast<const uint64_t*>(
                B_s + static_cast<uint64_t>(k16_base) * N + n64 * 64 + group * 8);

            int n_steps = (seg_end - seg_start) * 2;
            for (int step = 0; step < n_steps; ++step) {
                int kt = seg_start + (step >> 1);
                int sub = step & 1;
                int k16 = kt * 8 + 2 * w + sub;
                uint4 qv_nxt;
                uint64_t sc_nxt;
                if (step + 1 < n_steps) {
                    int nkt = seg_start + ((step + 1) >> 1);
                    int nsub = (step + 1) & 1;
                    int nk16 = nkt * 8 + 2 * w + nsub;
                    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(qv_nxt.x), "=r"(qv_nxt.y), "=r"(qv_nxt.z), "=r"(qv_nxt.w)
                        : "l"(B_q + static_cast<uint64_t>(nk16) * row_stride_int4 + n64 * 32 + lane));
                    sc_nxt = *reinterpret_cast<const uint64_t*>(
                        B_s + static_cast<uint64_t>(nk16) * N + n64 * 64 + group * 8);
                }
                uint32_t a0 = (group == 0) ? sA32[k16 * 8 + t] : 0u;
                uint32_t a2 = (group == 0) ? sA32[k16 * 8 + 4 + t] : 0u;
                uint32_t q_arr[4] = {qv_cur.x, qv_cur.y, qv_cur.z, qv_cur.w};
                const uint8_t* sc = reinterpret_cast<const uint8_t*>(&sc_cur);
                uint8_t sb[8] = {sc[0], sc[2], sc[1], sc[3], sc[4], sc[6], sc[5], sc[7]};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t q = q_arr[j];
                    uint32_t f0 = ((q & 0x00080008u) << 12) | ((q & 0x00070007u) << 6);
                    uint32_t f1 = ((q & 0x00800080u) << 8)  | ((q & 0x00700070u) << 2);
                    uint32_t f2 = ((q & 0x08000800u) << 4)  | ((q & 0x07000700u) >> 2);
                    uint32_t f3 = ((q & 0x80008000u))       | ((q & 0x70007000u) >> 6);
                    __nv_bfloat162 s0 = dequant_s_bf162(sb[2 * j]);
                    __nv_bfloat162 s1 = dequant_s_bf162(sb[2 * j + 1]);
                    mma_m16n8k16_bf16_m1(d0[2 * j], d1[2 * j], d2[2 * j], d3[2 * j],
                                         a0, a2, hmul2_u32(f0, s0), hmul2_u32(f1, s0));
                    mma_m16n8k16_bf16_m1(d0[2 * j + 1], d1[2 * j + 1], d2[2 * j + 1], d3[2 * j + 1],
                                         a0, a2, hmul2_u32(f2, s1), hmul2_u32(f3, s1));
                }
                qv_cur = qv_nxt;
                sc_cur = sc_nxt;
            }
            if (group == 0) {
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    s_acc[w][m * 8 + t * 2] = d0[m];
                    s_acc[w][m * 8 + t * 2 + 1] = d1[m];
                }
            }
            __syncthreads();
            if (threadIdx.x < 64) {
                int c = threadIdx.x;
                float P = s_acc[0][c] + (s_acc[1][c] + (s_acc[3][c] + s_acc[2][c]));
                R_reg = has_R ? (P + R_reg) : P;
                has_R = true;
            }
            __syncthreads();
        }
        if (threadIdx.x < 64) {
            C[n64 * 64 + threadIdx.x] = __float2bfloat16_rn(R_reg * gs);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Bit-exact column-major L2-resident SM121 NVFP4 Prefill GEMM (BM=64, BN=128, BK=64)
//    4 warps (2 along K, 2 along N), 3-stage cp.async pipeline (38.4 KB SMEM ->
//    2 CTAs/SM hiding barrier & cp.async latency), 1 __syncthreads per K-stage,
//    XOR-swizzled sA + ldmatrix.x4 + lop3 FP4 dequant. Zero extra weight RAM.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(128, 2) nvfp4_prefill_bm64_bn128_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint4* __restrict__ B_q,
    const uint8_t* __restrict__ B_s,
    const float* __restrict__ g_scale,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    constexpr int STAGES = 3;
    int m_tiles = M >> 6;
    int m_blk = blockIdx.x % m_tiles;
    int n_blk = blockIdx.x / m_tiles;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int group = lane >> 2;
    int t = lane & 3;
    int w_k = warp_id >> 1;
    int w_n = warp_id & 1;
    int n64 = n_blk * 2 + w_n;

    extern __shared__ char smem_raw[];
    uint4* sA = reinterpret_cast<uint4*>(smem_raw);
    uint4* sBq = reinterpret_cast<uint4*>(smem_raw + STAGES * 8192);
    uint8_t* sBs = reinterpret_cast<uint8_t*>(smem_raw + STAGES * 12288);

    int k_stages = K >> 6;
    int row_stride_int4 = N >> 1;
    const uint4* A_base4 = reinterpret_cast<const uint4*>(A + static_cast<uint64_t>(m_blk * 64) * K);
    int k_u4 = K >> 3;

    auto issue_stage = [&](int ks, int st) {
        uint4* sA_st = sA + st * 512;
        int k_off_u4 = ks * 8;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 3;
            int c8 = idx & 7;
            cp_async_16(sA_st + r * 8 + (c8 ^ (r & 7)),
                        A_base4 + static_cast<uint64_t>(r) * k_u4 + k_off_u4 + c8);
        }
        uint4* sBq_st = sBq + st * 256;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid + i * 128;
            int k16_local = idx >> 6;
            int col64 = idx & 63;
            int k16 = ks * 4 + k16_local;
            cp_async_16(sBq_st + idx,
                        B_q + static_cast<uint64_t>(k16) * row_stride_int4 + n_blk * 64 + col64);
        }
        if (tid < 32) {
            int k16_local = tid >> 3;
            int c16 = tid & 7;
            int k16 = ks * 4 + k16_local;
            cp_async_16(sBs + st * 512 + k16_local * 128 + c16 * 16,
                        B_s + static_cast<uint64_t>(k16) * N + n_blk * 128 + c16 * 16);
        }
        asm volatile("cp.async.commit_group;\n");
    };

    #pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < k_stages) issue_stage(st, st);
    }

    float d0[4][8] = {{0}}, d1[4][8] = {{0}}, d2[4][8] = {{0}}, d3[4][8] = {{0}};
    int lm_row = lane & 15;
    int lm_col_hi = lane >> 4;
    if (k_stages >= 2) {
        asm volatile("cp.async.wait_group 1;\n");
    } else {
        asm volatile("cp.async.wait_group 0;\n");
    }
    __syncthreads();

    for (int ks = 0; ks < k_stages; ++ks) {
        int st = ks % STAGES;
        const uint4* sA_st = sA + st * 512;
        const uint4* sBq_st = sBq + st * 256;
        const uint8_t* sBs_st = sBs + st * 512;

        int next_ks = ks + (STAGES - 1);
        if (next_ks < k_stages) {
            issue_stage(next_ks, next_ks % STAGES);
        }

        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            int k16_local = w_k * 2 + sub;
            uint4 qv = sBq_st[k16_local * 64 + w_n * 32 + lane];
            uint2 sc2 = *reinterpret_cast<const uint2*>(
                sBs_st + k16_local * 128 + w_n * 64 + group * 8);

            uint32_t a0[4], a1[4], a2[4], a3[4];
            int lm_col = k16_local * 2 + lm_col_hi;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r = mi * 16 + lm_row;
                ldmatrix_x4(a0[mi], a1[mi], a2[mi], a3[mi],
                            sA_st + r * 8 + (lm_col ^ (r & 7)));
            }

            __nv_bfloat162 s_arr[8] = {
                dequant_s_bf162(sc2.x & 0xFFu),
                dequant_s_bf162((sc2.x >> 16) & 0xFFu),
                dequant_s_bf162((sc2.x >> 8) & 0xFFu),
                dequant_s_bf162((sc2.x >> 24) & 0xFFu),
                dequant_s_bf162(sc2.y & 0xFFu),
                dequant_s_bf162((sc2.y >> 16) & 0xFFu),
                dequant_s_bf162((sc2.y >> 8) & 0xFFu),
                dequant_s_bf162((sc2.y >> 24) & 0xFFu),
            };
            uint32_t q_arr[4] = {qv.x, qv.y, qv.z, qv.w};

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t q = q_arr[j];
                uint32_t q0 = q << 12;
                uint32_t q1 = q << 8;
                uint32_t q2 = q << 4;
                uint32_t f0 = lop3_or_and(q0, 0x80008000u, (q0 >> 6) & 0x01C001C0u);
                uint32_t f1 = lop3_or_and(q1, 0x80008000u, (q1 >> 6) & 0x01C001C0u);
                uint32_t f2 = lop3_or_and(q2, 0x80008000u, (q2 >> 6) & 0x01C001C0u);
                uint32_t f3 = lop3_or_and(q,  0x80008000u, (q  >> 6) & 0x01C001C0u);
                uint32_t b00 = hmul2_u32(f0, s_arr[2 * j]);
                uint32_t b01 = hmul2_u32(f1, s_arr[2 * j]);
                uint32_t b10 = hmul2_u32(f2, s_arr[2 * j + 1]);
                uint32_t b11 = hmul2_u32(f3, s_arr[2 * j + 1]);
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    mma_m16n8k16_bf16_full(d0[mi][2 * j], d1[mi][2 * j], d2[mi][2 * j], d3[mi][2 * j],
                                           a0[mi], a1[mi], a2[mi], a3[mi], b00, b01);
                    mma_m16n8k16_bf16_full(d0[mi][2 * j + 1], d1[mi][2 * j + 1],
                                           d2[mi][2 * j + 1], d3[mi][2 * j + 1],
                                           a0[mi], a1[mi], a2[mi], a3[mi], b10, b11);
                }
            }
        }
        if (ks + 3 < k_stages) {
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();
    }

    float* s_red = reinterpret_cast<float*>(smem_raw);
    float gs = *g_scale;
    __nv_bfloat16* C_base = C + static_cast<uint64_t>(m_blk * 64) * N + n64 * 64;
    if (w_k == 1) {
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            int r0 = mi * 16 + group;
            int r1 = r0 + 8;
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                int c = m * 8 + t * 2;
                s_red[(w_n * 64 + r0) * 64 + c] = d0[mi][m];
                s_red[(w_n * 64 + r0) * 64 + c + 1] = d1[mi][m];
                s_red[(w_n * 64 + r1) * 64 + c] = d2[mi][m];
                s_red[(w_n * 64 + r1) * 64 + c + 1] = d3[mi][m];
            }
        }
    }
    __syncthreads();
    if (w_k == 0) {
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            int r0 = mi * 16 + group;
            int r1 = r0 + 8;
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                int c = m * 8 + t * 2;
                __nv_bfloat162 p0 = __floats2bfloat162_rn(
                    (d0[mi][m] + s_red[(w_n * 64 + r0) * 64 + c]) * gs,
                    (d1[mi][m] + s_red[(w_n * 64 + r0) * 64 + c + 1]) * gs);
                __nv_bfloat162 p1 = __floats2bfloat162_rn(
                    (d2[mi][m] + s_red[(w_n * 64 + r1) * 64 + c]) * gs,
                    (d3[mi][m] + s_red[(w_n * 64 + r1) * 64 + c + 1]) * gs);
                *reinterpret_cast<__nv_bfloat162*>(C_base + static_cast<uint64_t>(r0) * N + c) = p0;
                *reinterpret_cast<__nv_bfloat162*>(C_base + static_cast<uint64_t>(r1) * N + c) = p1;
            }
        }
    }
}

__global__ void build_silu_lut_kernel(float* __restrict__ lut) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536) {
        uint16_t bits = static_cast<uint16_t>(i);
        float g = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&bits));
        lut[i] = __fdividef(g, 1.0f + expf(-g));
    }
}

const float* get_silu_lut_ptr(at::Device dev) {
    static at::Tensor s_lut;
    if (!s_lut.defined() || s_lut.device() != dev) {
        s_lut = at::empty({65536}, at::TensorOptions().dtype(at::kFloat).device(dev));
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        build_silu_lut_kernel<<<256, 256, 0, stream>>>(s_lut.data_ptr<float>());
    }
    return s_lut.data_ptr<float>();
}

// ---------------------------------------------------------------------------
// 3b. Fused bit-exact NVFP4 gate_up + SiluAndMul (+ optional FP8 E4M3 quant)
//     prefill GEMM (BM=64, BN=128) with exact 65536-entry BF16->FP32 SiLU LUT.
//     w_n=0 computes gate[n_blk], w_n=1 computes up[n_blk] on standard Marlin
//     layout (zero weight permutation!). 4-warp cooperative epilogue + coalesced
//     stores; zero intermediate gate_up DRAM traffic, 0 spills, 2 CTAs/SM.
// ---------------------------------------------------------------------------
template <bool OUT_FP8>
__global__ void __launch_bounds__(128, 2) nvfp4_gate_up_silu_m512_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint4* __restrict__ B_q,
    const uint8_t* __restrict__ B_s,
    const float* __restrict__ g_scale,
    const float* __restrict__ silu_lut,
    const float* __restrict__ out_scale_a,
    void* __restrict__ C_raw,
    int M, int half_N, int K
) {
    constexpr int STAGES = 3;
    int m_tiles = M >> 6;
    int m_blk = blockIdx.x % m_tiles;
    int n_blk = blockIdx.x / m_tiles;
    int half_n64 = half_N >> 6;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int group = lane >> 2;
    int t = lane & 3;
    int w_k = warp_id >> 1;
    int w_n = warp_id & 1;

    extern __shared__ char smem_raw[];
    uint4* sA = reinterpret_cast<uint4*>(smem_raw);
    uint4* sBq = reinterpret_cast<uint4*>(smem_raw + STAGES * 8192);
    uint8_t* sBs = reinterpret_cast<uint8_t*>(smem_raw + STAGES * 12288);

    int k_stages = K >> 6;
    int N = half_N * 2;
    int row_stride_int4 = N >> 1;
    const uint4* A_base4 = reinterpret_cast<const uint4*>(A + static_cast<uint64_t>(m_blk * 64) * K);
    int k_u4 = K >> 3;

    auto issue_stage = [&](int ks, int st) {
        uint4* sA_st = sA + st * 512;
        int k_off_u4 = ks * 8;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 3;
            int c8 = idx & 7;
            cp_async_16(sA_st + r * 8 + (c8 ^ (r & 7)),
                        A_base4 + static_cast<uint64_t>(r) * k_u4 + k_off_u4 + c8);
        }
        uint4* sBq_st = sBq + st * 256;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid + i * 128;
            int k16_local = idx >> 6;
            int col64 = idx & 63;
            int is_up = col64 >> 5;
            int c32 = col64 & 31;
            int k16 = ks * 4 + k16_local;
            int src_n64 = is_up * half_n64 + n_blk;
            cp_async_16(sBq_st + idx,
                        B_q + static_cast<uint64_t>(k16) * row_stride_int4 + src_n64 * 32 + c32);
        }
        if (tid < 32) {
            int k16_local = tid >> 3;
            int c16 = tid & 7;
            int is_up = c16 >> 2;
            int c4 = c16 & 3;
            int k16 = ks * 4 + k16_local;
            int src_n64 = is_up * half_n64 + n_blk;
            cp_async_16(sBs + st * 512 + k16_local * 128 + c16 * 16,
                        B_s + static_cast<uint64_t>(k16) * N + src_n64 * 64 + c4 * 16);
        }
        asm volatile("cp.async.commit_group;\n");
    };

    #pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < k_stages) issue_stage(st, st);
    }

    float d0[4][8] = {{0}}, d1[4][8] = {{0}}, d2[4][8] = {{0}}, d3[4][8] = {{0}};
    int lm_row = lane & 15;
    int lm_col_hi = lane >> 4;
    if (k_stages >= 2) {
        asm volatile("cp.async.wait_group 1;\n");
    } else {
        asm volatile("cp.async.wait_group 0;\n");
    }
    __syncthreads();

    for (int ks = 0; ks < k_stages; ++ks) {
        int st = ks % STAGES;
        const uint4* sA_st = sA + st * 512;
        const uint4* sBq_st = sBq + st * 256;
        const uint8_t* sBs_st = sBs + st * 512;
        int next_ks = ks + (STAGES - 1);
        if (next_ks < k_stages) issue_stage(next_ks, next_ks % STAGES);

        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            int k16_local = w_k * 2 + sub;
            uint4 qv = sBq_st[k16_local * 64 + w_n * 32 + lane];
            uint2 sc2 = *reinterpret_cast<const uint2*>(
                sBs_st + k16_local * 128 + w_n * 64 + group * 8);
            uint32_t a0[4], a1[4], a2[4], a3[4];
            int lm_col = k16_local * 2 + lm_col_hi;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r = mi * 16 + lm_row;
                ldmatrix_x4(a0[mi], a1[mi], a2[mi], a3[mi],
                            sA_st + r * 8 + (lm_col ^ (r & 7)));
            }
            __nv_bfloat162 s_arr[8] = {
                dequant_s_bf162(sc2.x & 0xFFu), dequant_s_bf162((sc2.x >> 16) & 0xFFu),
                dequant_s_bf162((sc2.x >> 8) & 0xFFu), dequant_s_bf162((sc2.x >> 24) & 0xFFu),
                dequant_s_bf162(sc2.y & 0xFFu), dequant_s_bf162((sc2.y >> 16) & 0xFFu),
                dequant_s_bf162((sc2.y >> 8) & 0xFFu), dequant_s_bf162((sc2.y >> 24) & 0xFFu),
            };
            uint32_t q_arr[4] = {qv.x, qv.y, qv.z, qv.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t q = q_arr[j];
                uint32_t q0 = q << 12, q1 = q << 8, q2 = q << 4;
                uint32_t f0 = lop3_or_and(q0, 0x80008000u, (q0 >> 6) & 0x01C001C0u);
                uint32_t f1 = lop3_or_and(q1, 0x80008000u, (q1 >> 6) & 0x01C001C0u);
                uint32_t f2 = lop3_or_and(q2, 0x80008000u, (q2 >> 6) & 0x01C001C0u);
                uint32_t f3 = lop3_or_and(q,  0x80008000u, (q  >> 6) & 0x01C001C0u);
                uint32_t b00 = hmul2_u32(f0, s_arr[2 * j]), b01 = hmul2_u32(f1, s_arr[2 * j]);
                uint32_t b10 = hmul2_u32(f2, s_arr[2 * j + 1]), b11 = hmul2_u32(f3, s_arr[2 * j + 1]);
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    mma_m16n8k16_bf16_full(d0[mi][2 * j], d1[mi][2 * j], d2[mi][2 * j], d3[mi][2 * j],
                                           a0[mi], a1[mi], a2[mi], a3[mi], b00, b01);
                    mma_m16n8k16_bf16_full(d0[mi][2 * j + 1], d1[mi][2 * j + 1], d2[mi][2 * j + 1], d3[mi][2 * j + 1],
                                           a0[mi], a1[mi], a2[mi], a3[mi], b10, b11);
                }
            }
        }
        if (ks + 3 < k_stages) {
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();
    }

    float* s_red = reinterpret_cast<float*>(smem_raw);
    uint32_t* s_red_u32 = reinterpret_cast<uint32_t*>(smem_raw);
    float gs = *g_scale;
    if (w_k == 1) {
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            int r0 = mi * 16 + group, r1 = r0 + 8;
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                int c = m * 8 + t * 2;
                s_red[(w_n * 64 + r0) * 64 + c]     = d0[mi][m];
                s_red[(w_n * 64 + r0) * 64 + c + 1] = d1[mi][m];
                s_red[(w_n * 64 + r1) * 64 + c]     = d2[mi][m];
                s_red[(w_n * 64 + r1) * 64 + c + 1] = d3[mi][m];
            }
        }
    }
    __syncthreads();
    if (w_k == 0) {
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            int r0 = mi * 16 + group, r1 = r0 + 8;
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                int c = m * 8 + t * 2;
                int idx0 = (w_n * 64 + r0) * 64 + c;
                int idx1 = (w_n * 64 + r1) * 64 + c;
                __nv_bfloat16 b0 = __float2bfloat16_rn((d0[mi][m] + s_red[idx0]) * gs);
                __nv_bfloat16 b1 = __float2bfloat16_rn((d1[mi][m] + s_red[idx0 + 1]) * gs);
                __nv_bfloat16 b2 = __float2bfloat16_rn((d2[mi][m] + s_red[idx1]) * gs);
                __nv_bfloat16 b3 = __float2bfloat16_rn((d3[mi][m] + s_red[idx1 + 1]) * gs);
                if (w_n == 0) {
                    s_red_u32[idx0]     = static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&b0));
                    s_red_u32[idx0 + 1] = static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&b1));
                    s_red_u32[idx1]     = static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&b2));
                    s_red_u32[idx1 + 1] = static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&b3));
                } else {
                    s_red[idx0]     = __bfloat162float(b0);
                    s_red[idx0 + 1] = __bfloat162float(b1);
                    s_red[idx1]     = __bfloat162float(b2);
                    s_red[idx1 + 1] = __bfloat162float(b3);
                }
            }
        }
    }
    __syncthreads();
    if constexpr (OUT_FP8) {
        float inv_sa = __fdividef(1.0f, *out_scale_a);
        uint8_t* C_base = reinterpret_cast<uint8_t*>(C_raw) + static_cast<uint64_t>(m_blk * 64) * half_N + n_blk * 64;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 3;
            int c8 = (idx & 7) << 3;
            uint8_t out8[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                uint32_t gb = s_red_u32[r * 64 + c8 + k];
                float sg = __ldg(silu_lut + gb);
                float u = s_red[(64 + r) * 64 + c8 + k];
                float v = (sg * u) * inv_sa;
                v = fminf(fmaxf(v, -448.0f), 448.0f);
                __nv_fp8_e4m3 q(v);
                out8[k] = *reinterpret_cast<uint8_t*>(&q);
            }
            *reinterpret_cast<uint2*>(C_base + static_cast<uint64_t>(r) * half_N + c8) =
                *reinterpret_cast<const uint2*>(out8);
        }
    } else {
        __nv_bfloat16* C_base = reinterpret_cast<__nv_bfloat16*>(C_raw) + static_cast<uint64_t>(m_blk * 64) * half_N + n_blk * 64;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 3;
            int c8 = (idx & 7) << 3;
            __nv_bfloat16 out8[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                uint32_t gb = s_red_u32[r * 64 + c8 + k];
                float sg = __ldg(silu_lut + gb);
                float u = s_red[(64 + r) * 64 + c8 + k];
                out8[k] = __float2bfloat16_rn(sg * u);
            }
            *reinterpret_cast<uint4*>(C_base + static_cast<uint64_t>(r) * half_N + c8) =
                *reinterpret_cast<const uint4*>(out8);
        }
    }
}

// ---------------------------------------------------------------------------
// 3c. Bit-exact LPT-sorted non-persistent tasklist NVFP4 prefill GEMM for
//     down_proj (N=6656, K=19968, M=512). Matches Marlin part1/part2 split-K
//     bit-for-bit across all 40 NVFP4 down layers at 2 CTAs/SM, 0 spills.
// ---------------------------------------------------------------------------
struct TileTask {
    int16_t m_blk;
    int16_t n_blk;
    int16_t s0;
    int16_t s1;
    int16_t part_slot;
    int16_t pad;
};

struct ReduceTask {
    int16_t m_blk;
    int16_t n_blk;
    int16_t p_start;
    int16_t n_parts;
};

__global__ void __launch_bounds__(128, 2) nvfp4_bn128_tasklist_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint4* __restrict__ B_q,
    const uint8_t* __restrict__ B_s,
    const float* __restrict__ g_scale,
    const TileTask* __restrict__ tasks,
    float* __restrict__ partials,
    __nv_bfloat16* __restrict__ C,
    int N, int K
) {
    constexpr int STAGES = 3;
    TileTask tk = tasks[blockIdx.x];
    int m_blk = tk.m_blk, n_blk = tk.n_blk, s0 = tk.s0, s1 = tk.s1, part_slot = tk.part_slot;
    int slen = s1 - s0;
    int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31, group = lane >> 2, t = lane & 3;
    int w_k = warp_id >> 1, w_n = warp_id & 1, n64 = n_blk * 2 + w_n;

    extern __shared__ char smem_raw[];
    uint4* sA = reinterpret_cast<uint4*>(smem_raw);
    uint4* sBq = reinterpret_cast<uint4*>(smem_raw + STAGES * 8192);
    uint8_t* sBs = reinterpret_cast<uint8_t*>(smem_raw + STAGES * 12288);

    int row_stride_int4 = N >> 1, k_u4 = K >> 3;
    const uint4* A_base4 = reinterpret_cast<const uint4*>(A + static_cast<uint64_t>(m_blk * 64) * K);

    auto issue_stage = [&](int ks, int st) {
        uint4* sA_st = sA + st * 512;
        int k_off = ks * 8;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 3, c8 = idx & 7;
            cp_async_16(sA_st + r * 8 + (c8 ^ (r & 7)),
                        A_base4 + static_cast<uint64_t>(r) * k_u4 + k_off + c8);
        }
        uint4* sBq_st = sBq + st * 256;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid + i * 128;
            int k16l = idx >> 6, col = idx & 63;
            cp_async_16(sBq_st + idx,
                        B_q + static_cast<uint64_t>(ks * 4 + k16l) * row_stride_int4 + n_blk * 64 + col);
        }
        if (tid < 32) {
            int k16l = tid >> 3, c16 = tid & 7;
            cp_async_16(sBs + st * 512 + k16l * 128 + c16 * 16,
                        B_s + static_cast<uint64_t>(ks * 4 + k16l) * N + n_blk * 128 + c16 * 16);
        }
        asm volatile("cp.async.commit_group;\n");
    };

    #pragma unroll
    for (int st = 0; st < 2; ++st) {
        if (st < slen) issue_stage(s0 + st, st);
    }

    float d0[4][8] = {{0}}, d1[4][8] = {{0}}, d2[4][8] = {{0}}, d3[4][8] = {{0}};
    int lm_row = lane & 15, lm_col_hi = lane >> 4;
    if (slen >= 2) {
        asm volatile("cp.async.wait_group 1;\n");
    } else {
        asm volatile("cp.async.wait_group 0;\n");
    }
    __syncthreads();

    for (int lk = 0; lk < slen; ++lk) {
        int st = lk % 3;
        const uint4* sA_st = sA + st * 512;
        const uint4* sBq_st = sBq + st * 256;
        const uint8_t* sBs_st = sBs + st * 512;
        if (lk + 2 < slen) issue_stage(s0 + lk + 2, (lk + 2) % 3);

        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            int k16l = w_k * 2 + sub;
            uint4 qv = sBq_st[k16l * 64 + w_n * 32 + lane];
            uint2 sc2 = *reinterpret_cast<const uint2*>(sBs_st + k16l * 128 + w_n * 64 + group * 8);
            uint32_t a0[4], a1[4], a2[4], a3[4];
            int lm_col = k16l * 2 + lm_col_hi;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r = mi * 16 + lm_row;
                ldmatrix_x4(a0[mi], a1[mi], a2[mi], a3[mi], sA_st + r * 8 + (lm_col ^ (r & 7)));
            }
            uint32_t qa[4] = {qv.x, qv.y, qv.z, qv.w};
            __nv_bfloat162 sa[8] = {
                dequant_s_bf162(sc2.x & 0xFFu), dequant_s_bf162((sc2.x >> 16) & 0xFFu),
                dequant_s_bf162((sc2.x >> 8) & 0xFFu), dequant_s_bf162((sc2.x >> 24) & 0xFFu),
                dequant_s_bf162(sc2.y & 0xFFu), dequant_s_bf162((sc2.y >> 16) & 0xFFu),
                dequant_s_bf162((sc2.y >> 8) & 0xFFu), dequant_s_bf162((sc2.y >> 24) & 0xFFu),
            };
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t q = qa[j];
                uint32_t q0 = q << 12, q1 = q << 8, q2 = q << 4;
                uint32_t f0 = lop3_or_and(q0, 0x80008000u, (q0 >> 6) & 0x01C001C0u);
                uint32_t f1 = lop3_or_and(q1, 0x80008000u, (q1 >> 6) & 0x01C001C0u);
                uint32_t f2 = lop3_or_and(q2, 0x80008000u, (q2 >> 6) & 0x01C001C0u);
                uint32_t f3 = lop3_or_and(q,  0x80008000u, (q  >> 6) & 0x01C001C0u);
                uint32_t b00 = hmul2_u32(f0, sa[2 * j]), b01 = hmul2_u32(f1, sa[2 * j]);
                uint32_t b10 = hmul2_u32(f2, sa[2 * j + 1]), b11 = hmul2_u32(f3, sa[2 * j + 1]);
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    mma_m16n8k16_bf16_full(d0[mi][2 * j], d1[mi][2 * j], d2[mi][2 * j], d3[mi][2 * j],
                                           a0[mi], a1[mi], a2[mi], a3[mi], b00, b01);
                    mma_m16n8k16_bf16_full(d0[mi][2 * j + 1], d1[mi][2 * j + 1], d2[mi][2 * j + 1], d3[mi][2 * j + 1],
                                           a0[mi], a1[mi], a2[mi], a3[mi], b10, b11);
                }
            }
        }
        if (lk + 3 < slen) {
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();
    }

    float* s_red = reinterpret_cast<float*>(smem_raw);
    if (w_k == 1) {
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            int r0 = mi * 16 + group, r1 = r0 + 8;
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                int c = m * 8 + t * 2;
                s_red[(w_n * 64 + r0) * 64 + c]     = d0[mi][m];
                s_red[(w_n * 64 + r0) * 64 + c + 1] = d1[mi][m];
                s_red[(w_n * 64 + r1) * 64 + c]     = d2[mi][m];
                s_red[(w_n * 64 + r1) * 64 + c + 1] = d3[mi][m];
            }
        }
    }
    __syncthreads();
    if (w_k == 0) {
        if (part_slot < 0) {
            float gs = *g_scale;
            __nv_bfloat16* C_base = C + static_cast<uint64_t>(m_blk * 64) * N + n64 * 64;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r0 = mi * 16 + group, r1 = r0 + 8;
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    int c = m * 8 + t * 2;
                    *reinterpret_cast<__nv_bfloat162*>(C_base + static_cast<uint64_t>(r0) * N + c) =
                        __floats2bfloat162_rn((d0[mi][m] + s_red[(w_n * 64 + r0) * 64 + c]) * gs,
                                              (d1[mi][m] + s_red[(w_n * 64 + r0) * 64 + c + 1]) * gs);
                    *reinterpret_cast<__nv_bfloat162*>(C_base + static_cast<uint64_t>(r1) * N + c) =
                        __floats2bfloat162_rn((d2[mi][m] + s_red[(w_n * 64 + r1) * 64 + c]) * gs,
                                              (d3[mi][m] + s_red[(w_n * 64 + r1) * 64 + c + 1]) * gs);
                }
            }
        } else {
            float* P_base = partials + static_cast<uint64_t>(part_slot) * 8192 + w_n * 64;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r0 = mi * 16 + group, r1 = r0 + 8;
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    int c = m * 8 + t * 2;
                    P_base[r0 * 128 + c]     = d0[mi][m] + s_red[(w_n * 64 + r0) * 64 + c];
                    P_base[r0 * 128 + c + 1] = d1[mi][m] + s_red[(w_n * 64 + r0) * 64 + c + 1];
                    P_base[r1 * 128 + c]     = d2[mi][m] + s_red[(w_n * 64 + r1) * 64 + c];
                    P_base[r1 * 128 + c + 1] = d3[mi][m] + s_red[(w_n * 64 + r1) * 64 + c + 1];
                }
            }
        }
    }
}

__global__ void nvfp4_reduce_tasklist_kernel(
    const ReduceTask* __restrict__ rtasks,
    const float* __restrict__ partials,
    const float* __restrict__ g_scale,
    __nv_bfloat16* __restrict__ C,
    int N
) {
    ReduceTask rt = rtasks[blockIdx.x];
    float gs = *g_scale;
    __nv_bfloat16* C_base = C + static_cast<uint64_t>(rt.m_blk * 64) * N + rt.n_blk * 128;
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        int r = i >> 6;
        int c2 = (i & 63) << 1;
        float v0 = partials[static_cast<uint64_t>(rt.p_start + rt.n_parts - 1) * 8192 + r * 128 + c2];
        float v1 = partials[static_cast<uint64_t>(rt.p_start + rt.n_parts - 1) * 8192 + r * 128 + c2 + 1];
        for (int k = rt.n_parts - 2; k >= 0; --k) {
            v0 = partials[static_cast<uint64_t>(rt.p_start + k) * 8192 + r * 128 + c2] + v0;
            v1 = partials[static_cast<uint64_t>(rt.p_start + k) * 8192 + r * 128 + c2 + 1] + v1;
        }
        *reinterpret_cast<__nv_bfloat162*>(C_base + static_cast<uint64_t>(r) * N + c2) =
            __floats2bfloat162_rn(v0 * gs, v1 * gs);
    }
}

// ---------------------------------------------------------------------------
// 4. Bit-exact FP8 w8a8 M=1 repacked MMA streaming GEMV (all 180 FP8 layers).
//    Load-time lossless repack [N/8, K/64, 32, uint4] -> 512B coalesced per warp.
//    Matches CUTLASS SM120 mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
//    bit-for-bit for both per-tensor and per-channel weight scales.
// ---------------------------------------------------------------------------
__global__ void repack_fp8_mma_kernel(
    const uint8_t* __restrict__ B,
    uint4* __restrict__ B_pack,
    int N, int K
) {
    int n_tile = blockIdx.x;
    int k_tiles = K >> 6;
    int lane = threadIdx.x;
    int group = lane >> 2;
    int t = lane & 3;
    int row = (n_tile << 3) + group;
    const uint32_t* Brow = reinterpret_cast<const uint32_t*>(B + static_cast<uint64_t>(row) * K);
    uint4* dst = B_pack + (static_cast<uint64_t>(n_tile) * k_tiles) * 32;
    for (int kt = 0; kt < k_tiles; ++kt) {
        int k32 = kt << 4;
        uint4 v;
        v.x = Brow[k32 + t];
        v.y = Brow[k32 + 4 + t];
        v.z = Brow[k32 + 8 + t];
        v.w = Brow[k32 + 12 + t];
        dst[static_cast<uint64_t>(kt) * 32 + lane] = v;
    }
}

__device__ __forceinline__ uint8_t quant_bf16_to_e4m3(__nv_bfloat16 x, float inv_scale) {
    float v = __bfloat162float(x) * inv_scale;
    v = fminf(fmaxf(v, -448.0f), 448.0f);
    __nv_fp8_e4m3 q(v);
    return *reinterpret_cast<uint8_t*>(&q);
}

template <bool SB_IS_VEC, bool FUSE_BF16_QUANT, int COLS_PER_WARP = 1>
__global__ void fp8_mma_repacked_m1_kernel(
    const void* __restrict__ A_raw,
    const uint4* __restrict__ B_pack,
    __nv_bfloat16* __restrict__ C,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    int N, int K
) {
    extern __shared__ uint32_t sA32[];
    float sa = *scale_a;
    if constexpr (FUSE_BF16_QUANT) {
        float inv_sa = __fdividef(1.0f, sa);
        const uint4* A4 = reinterpret_cast<const uint4*>(A_raw);
        uint2* sA2 = reinterpret_cast<uint2*>(sA32);
        for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) {
            uint4 v = A4[i];
            const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&v);
            uint8_t q[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                q[k] = quant_bf16_to_e4m3(p[k], inv_sa);
            }
            sA2[i] = *reinterpret_cast<const uint2*>(q);
        }
    } else {
        const uint4* A4 = reinterpret_cast<const uint4*>(A_raw);
        uint4* sA4 = reinterpret_cast<uint4*>(sA32);
        for (int i = threadIdx.x; i < (K >> 4); i += blockDim.x) {
            sA4[i] = A4[i];
        }
    }
    __syncthreads();

    float sb_scalar = SB_IS_VEC ? 1.f : (*scale_b);
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int t = lane & 3;
    int warps_per_block = blockDim.x >> 5;
    int total_warps = gridDim.x * warps_per_block;
    int global_warp = blockIdx.x * warps_per_block + warp_id;

    int num_n8 = N >> 3;
    int k_tiles = K >> 6;

    if constexpr (COLS_PER_WARP == 1) {
        for (int nt = global_warp; nt < num_n8; nt += total_warps) {
            const uint4* Bp = B_pack + (static_cast<uint64_t>(nt) * k_tiles) * 32 + lane;
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

            uint4 v_cur;
            asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(v_cur.x), "=r"(v_cur.y), "=r"(v_cur.z), "=r"(v_cur.w)
                : "l"(Bp));

            #pragma unroll 4
            for (int kt = 0; kt < k_tiles - 1; ++kt) {
                uint4 v_nxt;
                asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(v_nxt.x), "=r"(v_nxt.y), "=r"(v_nxt.z), "=r"(v_nxt.w)
                    : "l"(Bp + (kt + 1) * 32));
                int k32 = kt << 4;
                uint32_t a0 = sA32[k32 + t];
                uint32_t a1 = sA32[k32 + 4 + t];
                uint32_t a2 = sA32[k32 + 8 + t];
                uint32_t a3 = sA32[k32 + 12 + t];
                mma_m16n8k32_fp8_m1(d0, d1, d2, d3, a0, a1, v_cur.x, v_cur.y);
                mma_m16n8k32_fp8_m1(d0, d1, d2, d3, a2, a3, v_cur.z, v_cur.w);
                v_cur = v_nxt;
            }
            {
                int k32 = (k_tiles - 1) << 4;
                uint32_t a0 = sA32[k32 + t];
                uint32_t a1 = sA32[k32 + 4 + t];
                uint32_t a2 = sA32[k32 + 8 + t];
                uint32_t a3 = sA32[k32 + 12 + t];
                mma_m16n8k32_fp8_m1(d0, d1, d2, d3, a0, a1, v_cur.x, v_cur.y);
                mma_m16n8k32_fp8_m1(d0, d1, d2, d3, a2, a3, v_cur.z, v_cur.w);
            }

            if (group == 0) {
                int col = (nt << 3) + (t << 1);
                if constexpr (SB_IS_VEC) {
                    C[col] = __float2bfloat16_rn((d0 * scale_b[col]) * sa);
                    C[col + 1] = __float2bfloat16_rn((d1 * scale_b[col + 1]) * sa);
                } else {
                    C[col] = __float2bfloat16_rn((d0 * sb_scalar) * sa);
                    C[col + 1] = __float2bfloat16_rn((d1 * sb_scalar) * sa);
                }
            }
        }
    } else {
        int num_pairs = num_n8 >> 1;
        for (int np = global_warp; np < num_pairs; np += total_warps) {
            const uint4* Bp0 = B_pack + (static_cast<uint64_t>(np * 2) * k_tiles) * 32 + lane;
            const uint4* Bp1 = B_pack + (static_cast<uint64_t>(np * 2 + 1) * k_tiles) * 32 + lane;
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;

            uint4 v0, v1;
            asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(v0.x), "=r"(v0.y), "=r"(v0.z), "=r"(v0.w)
                : "l"(Bp0));
            asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(v1.x), "=r"(v1.y), "=r"(v1.z), "=r"(v1.w)
                : "l"(Bp1));

            #pragma unroll 4
            for (int kt = 0; kt < k_tiles - 1; ++kt) {
                uint4 n0, n1;
                asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(n0.x), "=r"(n0.y), "=r"(n0.z), "=r"(n0.w)
                    : "l"(Bp0 + (kt + 1) * 32));
                asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(n1.x), "=r"(n1.y), "=r"(n1.z), "=r"(n1.w)
                    : "l"(Bp1 + (kt + 1) * 32));
                int k32 = kt << 4;
                uint32_t a0 = sA32[k32 + t];
                uint32_t a1 = sA32[k32 + 4 + t];
                uint32_t a2 = sA32[k32 + 8 + t];
                uint32_t a3 = sA32[k32 + 12 + t];
                mma_m16n8k32_fp8_m1(d0, d1, d2, d3, a0, a1, v0.x, v0.y);
                mma_m16n8k32_fp8_m1(e0, e1, e2, e3, a0, a1, v1.x, v1.y);
                mma_m16n8k32_fp8_m1(d0, d1, d2, d3, a2, a3, v0.z, v0.w);
                mma_m16n8k32_fp8_m1(e0, e1, e2, e3, a2, a3, v1.z, v1.w);
                v0 = n0;
                v1 = n1;
            }
            {
                int k32 = (k_tiles - 1) << 4;
                uint32_t a0 = sA32[k32 + t];
                uint32_t a1 = sA32[k32 + 4 + t];
                uint32_t a2 = sA32[k32 + 8 + t];
                uint32_t a3 = sA32[k32 + 12 + t];
                mma_m16n8k32_fp8_m1(d0, d1, d2, d3, a0, a1, v0.x, v0.y);
                mma_m16n8k32_fp8_m1(e0, e1, e2, e3, a0, a1, v1.x, v1.y);
                mma_m16n8k32_fp8_m1(d0, d1, d2, d3, a2, a3, v0.z, v0.w);
                mma_m16n8k32_fp8_m1(e0, e1, e2, e3, a2, a3, v1.z, v1.w);
            }

            if (group == 0) {
                int col0 = ((np * 2) << 3) + (t << 1);
                int col1 = ((np * 2 + 1) << 3) + (t << 1);
                if constexpr (SB_IS_VEC) {
                    C[col0] = __float2bfloat16_rn((d0 * scale_b[col0]) * sa);
                    C[col0 + 1] = __float2bfloat16_rn((d1 * scale_b[col0 + 1]) * sa);
                    C[col1] = __float2bfloat16_rn((e0 * scale_b[col1]) * sa);
                    C[col1 + 1] = __float2bfloat16_rn((e1 * scale_b[col1 + 1]) * sa);
                } else {
                    C[col0] = __float2bfloat16_rn((d0 * sb_scalar) * sa);
                    C[col0 + 1] = __float2bfloat16_rn((d1 * sb_scalar) * sa);
                    C[col1] = __float2bfloat16_rn((e0 * sb_scalar) * sa);
                    C[col1 + 1] = __float2bfloat16_rn((e1 * sb_scalar) * sa);
                }
            }
        }
    }
}

at::Tensor nvfp4_down_m1(
    const at::Tensor& A,
    const at::Tensor& B_q,
    const at::Tensor& B_s,
    const at::Tensor& g_scale,
    at::Tensor partials
) {
    constexpr int N = 6656;
    constexpr int K = 19968;
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1 && A.size(1) == K);
    auto C = at::empty({1, N}, A.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    nvfp4_down_m1_kernel<<<96, 128, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
        g_scale.data_ptr<float>(),
        partials.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr())
    );
    nvfp4_down_reduce_kernel<<<(N + 255) / 256, 256, 0, stream>>>(
        partials.data_ptr<float>(),
        g_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr())
    );
    return C;
}

at::Tensor nvfp4_gate_up_silu_m512(
    const at::Tensor& A,
    const at::Tensor& B_q,
    const at::Tensor& B_s,
    const at::Tensor& g_scale,
    int64_t half_N
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int hn = static_cast<int>(half_N);
    TORCH_CHECK(M % 64 == 0 && hn % 64 == 0 && K % 64 == 0);
    auto C = at::empty({M, hn}, A.options());
    int grid = (M / 64) * (hn / 64);
    constexpr int smem = 3 * (8192 + 4096 + 512);
    const float* lut = get_silu_lut_ptr(A.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    nvfp4_gate_up_silu_m512_kernel<false><<<grid, 128, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
        g_scale.data_ptr<float>(),
        lut,
        nullptr,
        C.data_ptr(),
        M, hn, K
    );
    return C;
}

at::Tensor nvfp4_gate_up_silu_fp8quant_m512(
    const at::Tensor& A,
    const at::Tensor& B_q,
    const at::Tensor& B_s,
    const at::Tensor& g_scale,
    const at::Tensor& out_scale_a,
    int64_t half_N
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int hn = static_cast<int>(half_N);
    TORCH_CHECK(M % 64 == 0 && hn % 64 == 0 && K % 64 == 0);
    auto C = at::empty({M, hn}, A.options().dtype(at::kFloat8_e4m3fn));
    int grid = (M / 64) * (hn / 64);
    constexpr int smem = 3 * (8192 + 4096 + 512);
    const float* lut = get_silu_lut_ptr(A.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    nvfp4_gate_up_silu_m512_kernel<true><<<grid, 128, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
        g_scale.data_ptr<float>(),
        lut,
        out_scale_a.data_ptr<float>(),
        C.data_ptr(),
        M, hn, K
    );
    return C;
}

at::Tensor nvfp4_gate_up_silu_m1(
    const at::Tensor& A,
    const at::Tensor& B_q,
    const at::Tensor& B_s,
    const at::Tensor& g_scale
) {
    constexpr int half_N = 19968;
    constexpr int K = 6656;
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1 && A.size(1) == K);
    auto C = at::empty({1, half_N}, A.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    nvfp4_gate_up_silu_m1_kernel<<<156, 128, K * 2, stream>>>(
        reinterpret_cast<const uint32_t*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
        g_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr())
    );
    return C;
}

at::Tensor silu_mul_mode0(const at::Tensor& gate_up) {
    TORCH_CHECK(gate_up.is_cuda() && gate_up.is_contiguous() && gate_up.dtype() == at::kBFloat16);
    int M = static_cast<int>(gate_up.size(0));
    int half_N = static_cast<int>(gate_up.size(1) / 2);
    auto C = at::empty({M, half_N}, gate_up.options());
    int total = M * half_N;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    silu_mul_mode0_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, half_N
    );
    return C;
}

at::Tensor nvfp4_down_m512_lpt(
    const at::Tensor& A,
    const at::Tensor& B_q,
    const at::Tensor& B_s,
    const at::Tensor& g_scale,
    const at::Tensor& task_list,
    const at::Tensor& reduce_list,
    at::Tensor partials,
    int64_t N_in
) {
    int N = static_cast<int>(N_in);
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    auto C = at::empty({M, N}, A.options());
    constexpr int smem = 3 * (8192 + 4096 + 512);
    int n_tasks = static_cast<int>(task_list.size(0) / 6);
    int n_rtasks = static_cast<int>(reduce_list.size(0) / 4);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    nvfp4_bn128_tasklist_kernel<<<n_tasks, 128, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
        g_scale.data_ptr<float>(),
        reinterpret_cast<const TileTask*>(task_list.data_ptr()),
        partials.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        N, K
    );
    if (n_rtasks > 0) {
        nvfp4_reduce_tasklist_kernel<<<n_rtasks, 256, 0, stream>>>(
            reinterpret_cast<const ReduceTask*>(reduce_list.data_ptr()),
            partials.data_ptr<float>(),
            g_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            N
        );
    }
    return C;
}

at::Tensor nvfp4_lmhead_m1(
    const at::Tensor& A,
    const at::Tensor& B_q,
    const at::Tensor& B_s,
    const at::Tensor& g_scale
) {
    constexpr int N = 202048;
    constexpr int K = 6656;
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1 && A.size(1) == K);
    auto C = at::empty({1, N}, A.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    nvfp4_lmhead_m1_kernel<<<192, 128, K * 2, stream>>>(
        reinterpret_cast<const uint32_t*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
        g_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        N, K
    );
    return C;
}

at::Tensor nvfp4_prefill_bm64_bn128(
    const at::Tensor& A,
    const at::Tensor& B_q,
    const at::Tensor& B_s,
    const at::Tensor& g_scale,
    int64_t N
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int n_int = static_cast<int>(N);
    TORCH_CHECK(M % 64 == 0 && n_int % 128 == 0 && K % 64 == 0);
    auto C = at::empty({M, n_int}, A.options());
    int grid = (M / 64) * (n_int / 128);
    constexpr int smem = 3 * (8192 + 4096 + 512);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    nvfp4_prefill_bm64_bn128_kernel<<<grid, 128, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
        g_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, n_int, K
    );
    return C;
}

at::Tensor repack_fp8_mma(const at::Tensor& B_colmajor) {
    TORCH_CHECK(B_colmajor.is_cuda() && B_colmajor.dim() == 2);
    TORCH_CHECK(B_colmajor.stride(0) == 1, "Expected column-major [K, N] FP8 weight");
    int K = static_cast<int>(B_colmajor.size(0));
    int N = static_cast<int>(B_colmajor.size(1));
    TORCH_CHECK(N % 8 == 0 && K % 64 == 0);
    auto B_pack = at::empty(
        {static_cast<int64_t>(N >> 3) * (K >> 6) * 32 * 4},
        B_colmajor.options().dtype(at::kInt)
    );
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    repack_fp8_mma_kernel<<<N >> 3, 32, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(B_colmajor.data_ptr()),
        reinterpret_cast<uint4*>(B_pack.data_ptr()),
        N, K
    );
    return B_pack;
}

at::Tensor fp8_mma_repacked_m1(
    const at::Tensor& A,
    const at::Tensor& B_pack,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    int64_t N
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dim() == 2 && A.size(0) == 1);
    int K = static_cast<int>(A.size(1));
    int n_int = static_cast<int>(N);
    TORCH_CHECK(n_int % 8 == 0 && K % 64 == 0);
    auto C = at::empty({1, n_int}, A.options().dtype(at::kBFloat16));
    int grid = 128;
    int blk = 256;
    int cols = 1;
    if (n_int >= 32768) {
        // gate_up FP8 (39936x6656): 2-col register blocking, 241.8 GB/s
        grid = 192;
        blk = 256;
        cols = 2;
    } else if (K >= 16384) {
        // down FP8 (6656x19968): 244.5 GB/s
        grid = 128;
        blk = 256;
        cols = 1;
    } else if (n_int == 4096) {
        // gate FP8 (4096x6656): 2-col register blocking, 238.7 GB/s
        grid = 96;
        blk = 128;
        cols = 2;
    } else if (n_int == 4608) {
        // qkv FP8 (4608x6656): 237.1 GB/s
        grid = 96;
        blk = 256;
        cols = 1;
    } else if (K <= 4096) {
        // oproj FP8 (6656x4096): 240.8 GB/s
        grid = 128;
        blk = 256;
        cols = 1;
    }
    if ((n_int >> 3) % 2 != 0) cols = 1;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bool fuse_bf16 = (A.dtype() == at::kBFloat16);
    bool vec_sb = (scale_b.numel() > 1);
    auto launch = [&](auto sb_tag, auto fuse_tag, auto cols_tag) {
        constexpr bool SB = decltype(sb_tag)::value;
        constexpr bool FB = decltype(fuse_tag)::value;
        constexpr int CW = decltype(cols_tag)::value;
        fp8_mma_repacked_m1_kernel<SB, FB, CW><<<grid, blk, K, stream>>>(
            A.data_ptr(),
            reinterpret_cast<const uint4*>(B_pack.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            n_int, K
        );
    };
    if (cols == 2) {
        if (fuse_bf16) {
            if (vec_sb) launch(std::true_type{}, std::true_type{}, std::integral_constant<int, 2>{});
            else        launch(std::false_type{}, std::true_type{}, std::integral_constant<int, 2>{});
        } else {
            if (vec_sb) launch(std::true_type{}, std::false_type{}, std::integral_constant<int, 2>{});
            else        launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 2>{});
        }
    } else {
        if (fuse_bf16) {
            if (vec_sb) launch(std::true_type{}, std::true_type{}, std::integral_constant<int, 1>{});
            else        launch(std::false_type{}, std::true_type{}, std::integral_constant<int, 1>{});
        } else {
            if (vec_sb) launch(std::true_type{}, std::false_type{}, std::integral_constant<int, 1>{});
            else        launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 1>{});
        }
    }
    return C;
}

// ---------------------------------------------------------------------------
// 4b. Original 32-lane warp-interleaved 512B-aligned lossless FP8 exponent-pair
//     rANS compression & M=1 streaming MMA GEMV (117/117 FP8 layers).
//     - Sign|mantissa nibbles packed as coalesced uint2 (256B per k=64 tile).
//     - Exponent pairs (256 symbols) rANS-coded with 12-bit CDF / 16-bit renorm,
//       interleaved across 32 lanes in exact forward consumption order so every
//       DRAM read is a 100% coalesced 512B burst into a 1KB per-warp SMEM ring.
//     - Executes exact same mma.sync.aligned.m16n8k32 K-order & epilogue as
//       CUTLASS SM120 -> 0 bit mismatches.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t get_lanemask_lt() {
    uint32_t m;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
    return m;
}

__device__ __forceinline__ uint32_t pack_fp8_sm_pair(uint32_t a, uint32_t b) {
    return (a & 0x87878787u)
         | ((b & 0x07070707u) << 4)
         | ((b & 0x80808080u) >> 4);
}

__device__ __forceinline__ uint32_t fp8_exp_pair_lo(uint32_t w) {
    return ((w >> 3) & 0xFu) | (((w >> 11) & 0xFu) << 4);
}

__device__ __forceinline__ uint32_t fp8_exp_pair_hi(uint32_t w) {
    return ((w >> 19) & 0xFu) | (((w >> 27) & 0xFu) << 4);
}

__global__ void fp8_hist_exp_pair_kernel(
    const uint32_t* __restrict__ B_u32,
    int64_t num_u32,
    unsigned long long* __restrict__ g_hist
) {
    __shared__ uint32_t s_hist[256];
    if (threadIdx.x < 256) s_hist[threadIdx.x] = 0u;
    __syncthreads();

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = idx; i < num_u32; i += stride) {
        uint32_t w = B_u32[i];
        atomicAdd(&s_hist[fp8_exp_pair_lo(w)], 1u);
        atomicAdd(&s_hist[fp8_exp_pair_hi(w)], 1u);
    }
    __syncthreads();
    if (threadIdx.x < 256 && s_hist[threadIdx.x] > 0u) {
        atomicAdd(&g_hist[threadIdx.x], static_cast<unsigned long long>(s_hist[threadIdx.x]));
    }
}

__global__ void fp8_rans_measure_sm_kernel(
    const uint8_t* __restrict__ B,
    uint4* __restrict__ sm_pack,
    const uint32_t* __restrict__ d_freq,
    const uint32_t* __restrict__ d_cdf,
    int32_t* __restrict__ tile_u16,
    int64_t* __restrict__ tile_chunks,
    int N, int K
) {
    int nt = blockIdx.x;
    int k_tiles = K >> 6;
    int k_pairs = k_tiles >> 1;
    int lane = threadIdx.x;
    int group = lane >> 2;
    int t = lane & 3;
    int row = (nt << 3) + group;
    const uint32_t* Brow = reinterpret_cast<const uint32_t*>(B + static_cast<uint64_t>(row) * K);
    uint4* sm_dst = sm_pack + (static_cast<uint64_t>(nt) * k_pairs) * 32;

    uint32_t st0 = 65536u, st1 = 65536u, st2 = 65536u, st3 = 65536u;
    uint32_t total_u16 = 0u;

    #define RANS_MEAS_SYM(st_var, sym_val) do { \
        uint32_t _s = (sym_val); \
        uint32_t _f = d_freq[_s]; \
        uint32_t _c = d_cdf[_s]; \
        bool _em = ((st_var) >= (_f << 21)); \
        uint32_t _m = __ballot_sync(0xffffffffu, _em); \
        total_u16 += static_cast<uint32_t>(__popc(_m)); \
        if (_em) (st_var) >>= 16; \
        (st_var) = (((st_var) / _f) << 11) + ((st_var) % _f) + _c; \
    } while (0)

    for (int kp = k_pairs - 1; kp >= 0; --kp) {
        int k_odd = (kp * 2 + 1) << 4;
        uint32_t o0 = Brow[k_odd + t];
        uint32_t o1 = Brow[k_odd + 4 + t];
        uint32_t o2 = Brow[k_odd + 8 + t];
        uint32_t o3 = Brow[k_odd + 12 + t];
        RANS_MEAS_SYM(st3, fp8_exp_pair_hi(o3));
        RANS_MEAS_SYM(st2, fp8_exp_pair_hi(o2));
        RANS_MEAS_SYM(st1, fp8_exp_pair_hi(o1));
        RANS_MEAS_SYM(st0, fp8_exp_pair_hi(o0));
        RANS_MEAS_SYM(st3, fp8_exp_pair_lo(o3));
        RANS_MEAS_SYM(st2, fp8_exp_pair_lo(o2));
        RANS_MEAS_SYM(st1, fp8_exp_pair_lo(o1));
        RANS_MEAS_SYM(st0, fp8_exp_pair_lo(o0));

        int k_evn = (kp * 2) << 4;
        uint32_t e0 = Brow[k_evn + t];
        uint32_t e1 = Brow[k_evn + 4 + t];
        uint32_t e2 = Brow[k_evn + 8 + t];
        uint32_t e3 = Brow[k_evn + 12 + t];
        RANS_MEAS_SYM(st3, fp8_exp_pair_hi(e3));
        RANS_MEAS_SYM(st2, fp8_exp_pair_hi(e2));
        RANS_MEAS_SYM(st1, fp8_exp_pair_hi(e1));
        RANS_MEAS_SYM(st0, fp8_exp_pair_hi(e0));
        RANS_MEAS_SYM(st3, fp8_exp_pair_lo(e3));
        RANS_MEAS_SYM(st2, fp8_exp_pair_lo(e2));
        RANS_MEAS_SYM(st1, fp8_exp_pair_lo(e1));
        RANS_MEAS_SYM(st0, fp8_exp_pair_lo(e0));

        sm_dst[static_cast<uint64_t>(kp) * 32 + lane] = make_uint4(
            pack_fp8_sm_pair(e0, e1),
            pack_fp8_sm_pair(e2, e3),
            pack_fp8_sm_pair(o0, o1),
            pack_fp8_sm_pair(o2, o3)
        );
    }
    #undef RANS_MEAS_SYM

    if (lane == 0) {
        tile_u16[nt] = static_cast<int32_t>(total_u16);
        int64_t chunks = (static_cast<int64_t>(total_u16) + 255) / 256;
        tile_chunks[nt] = chunks > 0 ? chunks : 1;
    }
}

__global__ void fp8_rans_encode_kernel(
    const uint8_t* __restrict__ B,
    const uint32_t* __restrict__ d_freq,
    const uint32_t* __restrict__ d_cdf,
    const int32_t* __restrict__ tile_u16,
    const int64_t* __restrict__ tile_chunk_off,
    uint16_t* __restrict__ exp_stream,
    uint32_t* __restrict__ init_states,
    int N, int K
) {
    int nt = blockIdx.x;
    int k_tiles = K >> 6;
    int lane = threadIdx.x;
    int group = lane >> 2;
    int t = lane & 3;
    int row = (nt << 3) + group;
    const uint32_t* Brow = reinterpret_cast<const uint32_t*>(B + static_cast<uint64_t>(row) * K);
    uint32_t lmask_lt = get_lanemask_lt();

    uint32_t st0 = 65536u, st1 = 65536u, st2 = 65536u, st3 = 65536u;
    int64_t ptr = tile_chunk_off[nt] * 256 + static_cast<int64_t>(tile_u16[nt]);

    #define RANS_ENC_SYM(st_var, sym_val) do { \
        uint32_t _s = (sym_val); \
        uint32_t _f = d_freq[_s]; \
        uint32_t _c = d_cdf[_s]; \
        bool _em = ((st_var) >= (_f << 21)); \
        uint32_t _m = __ballot_sync(0xffffffffu, _em); \
        ptr -= __popc(_m); \
        if (_em) { \
            uint32_t _rk = static_cast<uint32_t>(__popc(_m & lmask_lt)); \
            exp_stream[ptr + _rk] = static_cast<uint16_t>((st_var) & 0xFFFFu); \
            (st_var) >>= 16; \
        } \
        (st_var) = (((st_var) / _f) << 11) + ((st_var) % _f) + _c; \
    } while (0)

    for (int kt = k_tiles - 1; kt >= 0; --kt) {
        int k32 = kt << 4;
        uint32_t w0 = Brow[k32 + t];
        uint32_t w1 = Brow[k32 + 4 + t];
        uint32_t w2 = Brow[k32 + 8 + t];
        uint32_t w3 = Brow[k32 + 12 + t];
        RANS_ENC_SYM(st3, fp8_exp_pair_hi(w3));
        RANS_ENC_SYM(st2, fp8_exp_pair_hi(w2));
        RANS_ENC_SYM(st1, fp8_exp_pair_hi(w1));
        RANS_ENC_SYM(st0, fp8_exp_pair_hi(w0));
        RANS_ENC_SYM(st3, fp8_exp_pair_lo(w3));
        RANS_ENC_SYM(st2, fp8_exp_pair_lo(w2));
        RANS_ENC_SYM(st1, fp8_exp_pair_lo(w1));
        RANS_ENC_SYM(st0, fp8_exp_pair_lo(w0));
    }
    #undef RANS_ENC_SYM

    uint64_t sb = static_cast<uint64_t>(nt) * 128;
    init_states[sb + lane] = st0;
    init_states[sb + 32 + lane] = st1;
    init_states[sb + 64 + lane] = st2;
    init_states[sb + 96 + lane] = st3;
}

template <bool SB_IS_VEC, bool FUSE_BF16_QUANT, bool DEEP_PIPE>
__global__ void fp8_rans_m1_kernel(
    const void* __restrict__ A_raw,
    const uint4* __restrict__ sm_pack,
    const uint4* __restrict__ exp_stream_u4,
    const int64_t* __restrict__ tile_chunk_off,
    const uint32_t* __restrict__ init_states,
    const uint32_t* __restrict__ d_lut,
    __nv_bfloat16* __restrict__ C,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    int N, int K
) {
    extern __shared__ uint8_t smem_rans_raw[];
    uint32_t* sA32 = reinterpret_cast<uint32_t*>(smem_rans_raw);
    uint32_t* sLut = reinterpret_cast<uint32_t*>(smem_rans_raw + K);
    uint16_t* sRing = reinterpret_cast<uint16_t*>(smem_rans_raw + K + 8192);

    float sa = *scale_a;
    if constexpr (FUSE_BF16_QUANT) {
        float inv_sa = __fdividef(1.0f, sa);
        const uint4* A4 = reinterpret_cast<const uint4*>(A_raw);
        uint2* sA2 = reinterpret_cast<uint2*>(sA32);
        for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) {
            uint4 v = A4[i];
            const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&v);
            uint8_t q[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                q[k] = quant_bf16_to_e4m3(p[k], inv_sa);
            }
            sA2[i] = *reinterpret_cast<const uint2*>(q);
        }
    } else {
        const uint4* A4 = reinterpret_cast<const uint4*>(A_raw);
        uint4* sA4 = reinterpret_cast<uint4*>(sA32);
        for (int i = threadIdx.x; i < (K >> 4); i += blockDim.x) {
            sA4[i] = A4[i];
        }
    }
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
        sLut[i] = d_lut[i];
    }
    __syncthreads();

    float sb_scalar = SB_IS_VEC ? 1.f : (*scale_b);
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int t = lane & 3;
    int warps_per_block = blockDim.x >> 5;
    int total_warps = gridDim.x * warps_per_block;
    int global_warp = blockIdx.x * warps_per_block + warp_id;
    uint32_t lmask_lt = get_lanemask_lt();

    uint16_t* wbuf = sRing + warp_id * 512;
    uint4* wbuf4 = reinterpret_cast<uint4*>(wbuf);

    int num_n8 = N >> 3;
    int k_pairs = K >> 7;

    #define RANS_DEC_QUAD(o0, o1, o2, o3) do { \
        uint32_t _e0 = sLut[st0 & 2047u]; \
        uint32_t _e1 = sLut[st1 & 2047u]; \
        uint32_t _e2 = sLut[st2 & 2047u]; \
        uint32_t _e3 = sLut[st3 & 2047u]; \
        st0 = ((_e0 >> 11) & 2047u) * (st0 >> 11) + (_e0 & 2047u); \
        st1 = ((_e1 >> 11) & 2047u) * (st1 >> 11) + (_e1 & 2047u); \
        st2 = ((_e2 >> 11) & 2047u) * (st2 >> 11) + (_e2 & 2047u); \
        st3 = ((_e3 >> 11) & 2047u) * (st3 >> 11) + (_e3 & 2047u); \
        bool _n0 = (st0 < 65536u); \
        bool _n1 = (st1 < 65536u); \
        bool _n2 = (st2 < 65536u); \
        bool _n3 = (st3 < 65536u); \
        uint32_t _m0 = __ballot_sync(0xffffffffu, _n0); \
        uint32_t _m1 = __ballot_sync(0xffffffffu, _n1); \
        uint32_t _m2 = __ballot_sync(0xffffffffu, _n2); \
        uint32_t _m3 = __ballot_sync(0xffffffffu, _n3); \
        int _p1 = pos + __popc(_m0); \
        int _p2 = _p1 + __popc(_m1); \
        int _p3 = _p2 + __popc(_m2); \
        if (_n0) st0 = (st0 << 16) | static_cast<uint32_t>(wbuf[(pos + __popc(_m0 & lmask_lt)) & 511]); \
        if (_n1) st1 = (st1 << 16) | static_cast<uint32_t>(wbuf[(_p1 + __popc(_m1 & lmask_lt)) & 511]); \
        if (_n2) st2 = (st2 << 16) | static_cast<uint32_t>(wbuf[(_p2 + __popc(_m2 & lmask_lt)) & 511]); \
        if (_n3) st3 = (st3 << 16) | static_cast<uint32_t>(wbuf[(_p3 + __popc(_m3 & lmask_lt)) & 511]); \
        pos = _p3 + __popc(_m3); \
        o0 = _e0; o1 = _e1; o2 = _e2; o3 = _e3; \
    } while (0)

    #define RANS_REFILL_CHECK() do { \
        if (pos >= refill_at) { \
            int slot = ((refill_at >> 8) - 1) & 1; \
            __syncwarp(); \
            wbuf4[slot * 32 + lane] = c_pref0; \
            __syncwarp(); \
            if constexpr (DEEP_PIPE) { \
                c_pref0 = c_pref1; \
                if (chunk_idx < tile_chunks) { \
                    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" \
                        : "=r"(c_pref1.x), "=r"(c_pref1.y), "=r"(c_pref1.z), "=r"(c_pref1.w) \
                        : "l"(es4 + static_cast<uint64_t>(chunk_idx) * 32)); \
                } \
            } else { \
                if (chunk_idx < tile_chunks) { \
                    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" \
                        : "=r"(c_pref0.x), "=r"(c_pref0.y), "=r"(c_pref0.z), "=r"(c_pref0.w) \
                        : "l"(es4 + static_cast<uint64_t>(chunk_idx) * 32)); \
                } \
            } \
            chunk_idx++; \
            refill_at += 256; \
        } \
    } while (0)

    #define RANS_DO_KT(sm_xy_a, sm_xy_b, k32_off) do { \
        uint32_t _l0, _l1, _l2, _l3, _h0, _h1, _h2, _h3; \
        RANS_DEC_QUAD(_l0, _l1, _l2, _l3); \
        RANS_DEC_QUAD(_h0, _h1, _h2, _h3); \
        { \
            uint32_t _a0 = sA32[(k32_off) + t]; \
            uint32_t _a1 = sA32[(k32_off) + 4 + t]; \
            uint32_t _b0 = ((sm_xy_a) & 0x87878787u) \
                         | ((_l0 >> 21) & 0x00000078u) | ((_l0 >> 17) & 0x00007800u) \
                         | ((_h0 >> 5)  & 0x00780000u) | ((_h0 >> 1)  & 0x78000000u); \
            uint32_t _b1 = ((((sm_xy_a) & 0x70707070u) >> 4) | (((sm_xy_a) & 0x08080808u) << 4)) \
                         | ((_l1 >> 21) & 0x00000078u) | ((_l1 >> 17) & 0x00007800u) \
                         | ((_h1 >> 5)  & 0x00780000u) | ((_h1 >> 1)  & 0x78000000u); \
            mma_m16n8k32_fp8_m1(d0, d1, d2, d3, _a0, _a1, _b0, _b1); \
        } \
        { \
            uint32_t _a2 = sA32[(k32_off) + 8 + t]; \
            uint32_t _a3 = sA32[(k32_off) + 12 + t]; \
            uint32_t _b2 = ((sm_xy_b) & 0x87878787u) \
                         | ((_l2 >> 21) & 0x00000078u) | ((_l2 >> 17) & 0x00007800u) \
                         | ((_h2 >> 5)  & 0x00780000u) | ((_h2 >> 1)  & 0x78000000u); \
            uint32_t _b3 = ((((sm_xy_b) & 0x70707070u) >> 4) | (((sm_xy_b) & 0x08080808u) << 4)) \
                         | ((_l3 >> 21) & 0x00000078u) | ((_l3 >> 17) & 0x00007800u) \
                         | ((_h3 >> 5)  & 0x00780000u) | ((_h3 >> 1)  & 0x78000000u); \
            mma_m16n8k32_fp8_m1(d0, d1, d2, d3, _a2, _a3, _b2, _b3); \
        } \
        RANS_REFILL_CHECK(); \
    } while (0)

    for (int nt = global_warp; nt < num_n8; nt += total_warps) {
        const uint4* Sm = sm_pack + (static_cast<uint64_t>(nt) * k_pairs) * 32 + lane;
        int64_t c_start = tile_chunk_off[nt];
        int tile_chunks = static_cast<int>(tile_chunk_off[nt + 1] - c_start);
        const uint4* es4 = exp_stream_u4 + c_start * 32 + lane;

        uint64_t sb = static_cast<uint64_t>(nt) * 128;
        uint32_t st0 = init_states[sb + lane];
        uint32_t st1 = init_states[sb + 32 + lane];
        uint32_t st2 = init_states[sb + 64 + lane];
        uint32_t st3 = init_states[sb + 96 + lane];
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

        uint4 c0, c1, c_pref0, c_pref1;
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(c0.x), "=r"(c0.y), "=r"(c0.z), "=r"(c0.w) : "l"(es4));
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(c1.x), "=r"(c1.y), "=r"(c1.z), "=r"(c1.w) : "l"(es4 + 32));
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(c_pref0.x), "=r"(c_pref0.y), "=r"(c_pref0.z), "=r"(c_pref0.w) : "l"(es4 + 64));
        if constexpr (DEEP_PIPE) {
            asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(c_pref1.x), "=r"(c_pref1.y), "=r"(c_pref1.z), "=r"(c_pref1.w) : "l"(es4 + 96));
        }

        wbuf4[lane] = c0;
        wbuf4[32 + lane] = c1;
        __syncwarp();

        int chunk_idx = DEEP_PIPE ? 4 : 3;
        int pos = 0;
        int refill_at = 256;

        if constexpr (DEEP_PIPE) {
            uint4 sm0, sm1, sm2, sm3;
            asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(sm0.x), "=r"(sm0.y), "=r"(sm0.z), "=r"(sm0.w) : "l"(Sm));
            if (1 < k_pairs) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(sm1.x), "=r"(sm1.y), "=r"(sm1.z), "=r"(sm1.w) : "l"(Sm + 32));
            }
            if (2 < k_pairs) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(sm2.x), "=r"(sm2.y), "=r"(sm2.z), "=r"(sm2.w) : "l"(Sm + 64));
            }
            if (3 < k_pairs) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(sm3.x), "=r"(sm3.y), "=r"(sm3.z), "=r"(sm3.w) : "l"(Sm + 96));
            }
            for (int kp = 0; kp < k_pairs; ++kp) {
                uint4 sm_new;
                if (kp + 4 < k_pairs) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(sm_new.x), "=r"(sm_new.y), "=r"(sm_new.z), "=r"(sm_new.w)
                        : "l"(Sm + (kp + 4) * 32));
                }
                int k32_e = (kp * 2) << 4;
                int k32_o = (kp * 2 + 1) << 4;
                RANS_DO_KT(sm0.x, sm0.y, k32_e);
                RANS_DO_KT(sm0.z, sm0.w, k32_o);
                sm0 = sm1;
                sm1 = sm2;
                sm2 = sm3;
                sm3 = sm_new;
            }
        } else {
            uint4 sm_cur;
            asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(sm_cur.x), "=r"(sm_cur.y), "=r"(sm_cur.z), "=r"(sm_cur.w) : "l"(Sm));
            for (int kp = 0; kp < k_pairs; ++kp) {
                uint4 sm_nxt;
                if (kp + 1 < k_pairs) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(sm_nxt.x), "=r"(sm_nxt.y), "=r"(sm_nxt.z), "=r"(sm_nxt.w)
                        : "l"(Sm + (kp + 1) * 32));
                }
                int k32_e = (kp * 2) << 4;
                int k32_o = (kp * 2 + 1) << 4;
                RANS_DO_KT(sm_cur.x, sm_cur.y, k32_e);
                RANS_DO_KT(sm_cur.z, sm_cur.w, k32_o);
                sm_cur = sm_nxt;
            }
        }

        if (group == 0) {
            int col = (nt << 3) + (t << 1);
            if constexpr (SB_IS_VEC) {
                C[col] = __float2bfloat16_rn((d0 * scale_b[col]) * sa);
                C[col + 1] = __float2bfloat16_rn((d1 * scale_b[col + 1]) * sa);
            } else {
                C[col] = __float2bfloat16_rn((d0 * sb_scalar) * sa);
                C[col + 1] = __float2bfloat16_rn((d1 * sb_scalar) * sa);
            }
        }
    }
    #undef RANS_DO_KT
    #undef RANS_REFILL_CHECK
    #undef RANS_DEC_QUAD
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> repack_fp8_rans(
    const at::Tensor& B_colmajor
) {
    TORCH_CHECK(B_colmajor.is_cuda() && B_colmajor.dim() == 2);
    TORCH_CHECK(B_colmajor.stride(0) == 1, "Expected column-major [K, N] FP8 weight");
    int K = static_cast<int>(B_colmajor.size(0));
    int N = static_cast<int>(B_colmajor.size(1));
    TORCH_CHECK(N % 8 == 0 && K % 128 == 0);
    int num_n8 = N >> 3;
    int k_pairs = K >> 7;
    auto dev_opts = B_colmajor.options();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 1. Histogram of 256 exponent-pair symbols on GPU.
    auto g_hist = at::zeros({256}, dev_opts.dtype(at::kLong));
    int64_t num_u32 = (static_cast<int64_t>(K) * N) >> 2;
    int hist_blocks = static_cast<int>(std::min<int64_t>((num_u32 + 255) / 256, 2048));
    fp8_hist_exp_pair_kernel<<<hist_blocks, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(B_colmajor.data_ptr()),
        num_u32,
        reinterpret_cast<unsigned long long*>(g_hist.data_ptr<int64_t>())
    );
    auto h_hist_t = g_hist.to(at::kCPU);
    const int64_t* h_hist = h_hist_t.data_ptr<int64_t>();

    // 2. Quantize frequencies to sum to 2048 (11-bit CDF -> 8 KB SMEM LUT, ~0.83 ratio).
    constexpr int M_PROB = 2048;
    int64_t total = 0;
    for (int i = 0; i < 256; ++i) total += h_hist[i];
    std::vector<uint32_t> freq(256, 0u);
    std::vector<double> exact(256, 0.0);
    std::vector<double> rem(256, -1.0);
    int sum_f = 0;
    for (int i = 0; i < 256; ++i) {
        if (h_hist[i] > 0) {
            double e = static_cast<double>(h_hist[i]) * static_cast<double>(M_PROB) / static_cast<double>(total);
            exact[i] = e;
            int f = static_cast<int>(std::floor(e));
            if (f < 1) f = 1;
            if (f > M_PROB - 1) f = M_PROB - 1;
            freq[i] = static_cast<uint32_t>(f);
            rem[i] = e - static_cast<double>(f);
            sum_f += f;
        }
    }
    while (sum_f < M_PROB) {
        int best = -1;
        double best_r = -1e9;
        for (int i = 0; i < 256; ++i) {
            if (h_hist[i] > 0 && freq[i] < static_cast<uint32_t>(M_PROB - 1) && rem[i] > best_r) {
                best_r = rem[i];
                best = i;
            }
        }
        if (best < 0) {
            for (int i = 0; i < 256; ++i) {
                if (freq[i] == 0u) { best = i; break; }
            }
        }
        freq[best] += 1u;
        rem[best] = -1e9;
        sum_f += 1;
    }
    while (sum_f > M_PROB) {
        int best = -1;
        double worst_excess = -1e9;
        for (int i = 0; i < 256; ++i) {
            if (freq[i] > 1u) {
                double ex = static_cast<double>(freq[i]) - exact[i];
                if (ex > worst_excess) {
                    worst_excess = ex;
                    best = i;
                }
            }
        }
        freq[best] -= 1u;
        sum_f -= 1;
    }

    std::vector<uint32_t> cdf(256, 0u);
    std::vector<uint32_t> lut(M_PROB, 0u);
    uint32_t acc = 0u;
    for (int s = 0; s < 256; ++s) {
        cdf[s] = acc;
        for (uint32_t j = 0; j < freq[s]; ++j) {
            lut[acc + j] = (static_cast<uint32_t>(s) << 24) | (freq[s] << 11) | j;
        }
        acc += freq[s];
    }

    auto d_freq = at::from_blob(freq.data(), {256}, at::kInt).to(B_colmajor.device());
    auto d_cdf  = at::from_blob(cdf.data(), {256}, at::kInt).to(B_colmajor.device());
    auto d_lut  = at::from_blob(lut.data(), {M_PROB}, at::kInt).to(B_colmajor.device());

    // 3. Measure per-tile u16 counts and pack raw sign|mantissa as 512B uint4 pairs.
    auto sm_pack = at::empty({static_cast<int64_t>(num_n8) * k_pairs * 32 * 4}, dev_opts.dtype(at::kInt));
    auto tile_u16 = at::empty({num_n8}, dev_opts.dtype(at::kInt));
    auto tile_chunks = at::empty({num_n8}, dev_opts.dtype(at::kLong));
    fp8_rans_measure_sm_kernel<<<num_n8, 32, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(B_colmajor.data_ptr()),
        reinterpret_cast<uint4*>(sm_pack.data_ptr()),
        reinterpret_cast<const uint32_t*>(d_freq.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(d_cdf.data_ptr<int32_t>()),
        tile_u16.data_ptr<int32_t>(),
        tile_chunks.data_ptr<int64_t>(),
        N, K
    );

    // 4. Prefix-sum 512B chunk offsets (+ 3-chunk safety tail for 2-chunk lookahead).
    auto tile_chunk_off = at::zeros({num_n8 + 1}, dev_opts.dtype(at::kLong));
    tile_chunk_off.narrow(0, 1, num_n8).copy_(at::cumsum(tile_chunks, 0));
    int64_t total_chunks = tile_chunk_off[num_n8].item<int64_t>();

    auto exp_stream = at::zeros({(total_chunks + 6) * 256}, dev_opts.dtype(at::kShort));
    auto init_states = at::empty({num_n8, 128}, dev_opts.dtype(at::kInt));

    // 5. Encode exact 4-way ILP warp-interleaved forward-consumable rANS streams on GPU.
    fp8_rans_encode_kernel<<<num_n8, 32, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(B_colmajor.data_ptr()),
        reinterpret_cast<const uint32_t*>(d_freq.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(d_cdf.data_ptr<int32_t>()),
        tile_u16.data_ptr<int32_t>(),
        tile_chunk_off.data_ptr<int64_t>(),
        reinterpret_cast<uint16_t*>(exp_stream.data_ptr<int16_t>()),
        reinterpret_cast<uint32_t*>(init_states.data_ptr<int32_t>()),
        N, K
    );

    return std::make_tuple(sm_pack, exp_stream, tile_chunk_off, init_states, d_lut);
}

at::Tensor fp8_rans_m1(
    const at::Tensor& A,
    const at::Tensor& sm_pack,
    const at::Tensor& exp_stream,
    const at::Tensor& tile_chunk_off,
    const at::Tensor& init_states,
    const at::Tensor& lut,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    int64_t N
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dim() == 2 && A.size(0) == 1);
    int K = static_cast<int>(A.size(1));
    int n_int = static_cast<int>(N);
    TORCH_CHECK(n_int % 8 == 0 && K % 128 == 0);
    auto C = at::empty({1, n_int}, A.options().dtype(at::kBFloat16));
    // Exact 48-SM balanced single-wave grids with 1.0 tile/warp (except gate_up 4.0 tiles/warp).
    int grid = 96;
    int blk = 288;
    if (n_int >= 32768) {
        // gate_up FP8 (39936x6656): 4992 tiles -> 96*13=1248 warps = exact 4 tiles/warp, 2 CTAs/SM -> 0 tail wave!
        grid = 96;
        blk = 416;
    } else if (K >= 16384) {
        // down FP8 (6656x19968): 832 tiles -> 96*9=864 warps = 1 tile/warp, 2 CTAs/SM -> 1 wave!
        grid = 96;
        blk = 288;
    } else if (n_int == 4096) {
        // gate FP8 (4096x6656): 512 tiles -> 48*11=528 warps = 1 tile/warp, 1 CTA/SM -> 1 wave!
        grid = 48;
        blk = 352;
    } else if (n_int == 4608) {
        // qkv FP8 (4608x6656): 576 tiles -> 48*12=576 warps = 1 tile/warp, 1 CTA/SM -> 1 wave!
        grid = 48;
        blk = 384;
    } else {
        // oproj FP8 (6656x4096): 832 tiles -> 96*9=864 warps = 1 tile/warp, 2 CTAs/SM -> 1 wave!
        grid = 96;
        blk = 288;
    }
    int smem = K + 8192 + (blk >> 5) * 1024;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bool fuse_bf16 = (A.dtype() == at::kBFloat16);
    bool vec_sb = (scale_b.numel() > 1);
    bool deep_pipe = (n_int < 32768);
    static bool smem_attr_set = false;
    if (!smem_attr_set) {
        cudaFuncSetAttribute(fp8_rans_m1_kernel<false, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(fp8_rans_m1_kernel<true, false, false>,  cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(fp8_rans_m1_kernel<false, true, false>,  cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(fp8_rans_m1_kernel<true, true, false>,   cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(fp8_rans_m1_kernel<false, false, true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(fp8_rans_m1_kernel<true, false, true>,   cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(fp8_rans_m1_kernel<false, true, true>,   cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(fp8_rans_m1_kernel<true, true, true>,    cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        smem_attr_set = true;
    }
    auto launch = [&](auto sb_tag, auto fuse_tag, auto dp_tag) {
        constexpr bool SB = decltype(sb_tag)::value;
        constexpr bool FB = decltype(fuse_tag)::value;
        constexpr bool DP = decltype(dp_tag)::value;
        fp8_rans_m1_kernel<SB, FB, DP><<<grid, blk, smem, stream>>>(
            A.data_ptr(),
            reinterpret_cast<const uint4*>(sm_pack.data_ptr()),
            reinterpret_cast<const uint4*>(exp_stream.data_ptr()),
            tile_chunk_off.data_ptr<int64_t>(),
            reinterpret_cast<const uint32_t*>(init_states.data_ptr()),
            reinterpret_cast<const uint32_t*>(lut.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            n_int, K
        );
    };
    auto dispatch_dp = [&](auto sb_tag, auto fuse_tag) {
        if (deep_pipe) launch(sb_tag, fuse_tag, std::true_type{});
        else           launch(sb_tag, fuse_tag, std::false_type{});
    };
    if (fuse_bf16) {
        if (vec_sb) dispatch_dp(std::true_type{}, std::true_type{});
        else        dispatch_dp(std::false_type{}, std::true_type{});
    } else {
        if (vec_sb) dispatch_dp(std::true_type{}, std::false_type{});
        else        dispatch_dp(std::false_type{}, std::false_type{});
    }
    return C;
}
//    - init_decode_loop: initializes M=1 static graph buffers & slot_mappings
//    - greedy_argmax_step_update: exact float32 argmax (lowest-index tie-break,
//      matching Triton tl.max) + writes out_tokens[step], updates input_ids,
//      positions, seq_lens, and slot_mappings in a single 1-block CUDA kernel.
// ---------------------------------------------------------------------------
__global__ void init_decode_loop_kernel(
    int32_t first_token,
    int64_t start_pos,
    int32_t count,
    int32_t* __restrict__ step_idx,
    int32_t* __restrict__ step_limit,
    int32_t* __restrict__ input_ids,
    int64_t* __restrict__ positions,
    int32_t* __restrict__ seq_lens,
    int32_t* __restrict__ query_start_loc,
    const uint64_t* __restrict__ input_block_table_ptrs,
    const int32_t* __restrict__ block_sizes,
    int64_t* __restrict__ slot_mappings,
    int64_t slot_stride,
    int num_groups
) {
    if (threadIdx.x == 0) {
        step_idx[0] = 0;
        step_limit[0] = count;
        input_ids[0] = first_token;
        positions[0] = start_pos;
        seq_lens[0] = static_cast<int32_t>(start_pos + 1);
        query_start_loc[0] = 0;
        query_start_loc[1] = 1;
        for (int g = 0; g < num_groups; ++g) {
            int bs = block_sizes[g];
            int64_t blk_idx = start_pos / bs;
            int64_t blk_off = start_pos % bs;
            const int32_t* bt = reinterpret_cast<const int32_t*>(input_block_table_ptrs[g]);
            slot_mappings[g * slot_stride] = static_cast<int64_t>(bt[blk_idx]) * bs + blk_off;
        }
    }
}

__global__ void greedy_argmax_step_update_kernel(
    const __nv_bfloat16* __restrict__ logits,
    int vocab_size,
    int32_t* __restrict__ out_tokens,
    int32_t* __restrict__ step_idx,
    const int32_t* __restrict__ step_limit,
    int32_t* __restrict__ input_ids,
    int64_t* __restrict__ positions,
    int32_t* __restrict__ seq_lens,
    const uint64_t* __restrict__ input_block_table_ptrs,
    const int32_t* __restrict__ block_sizes,
    int64_t* __restrict__ slot_mappings,
    int64_t slot_stride,
    int num_groups
) {
    int tid = threadIdx.x;
    float best_val = -3.402823466e+38f;
    int32_t best_idx = 0;

    int vec_elems = vocab_size & ~7;
    const uint4* logits4 = reinterpret_cast<const uint4*>(logits);
    int n_u4 = vec_elems >> 3;
    for (int i = tid; i < n_u4; i += 1024) {
        uint4 v = logits4[i];
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&v);
        int base = i << 3;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float fv = __bfloat162float(p[k]);
            int idx = base + k;
            if (fv > best_val || (fv == best_val && idx < best_idx)) {
                best_val = fv;
                best_idx = idx;
            }
        }
    }
    for (int idx = vec_elems + tid; idx < vocab_size; idx += 1024) {
        float fv = __bfloat162float(logits[idx]);
        if (fv > best_val || (fv == best_val && idx < best_idx)) {
            best_val = fv;
            best_idx = idx;
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, best_val, offset);
        int32_t oi = __shfl_down_sync(0xffffffffu, best_idx, offset);
        if (ov > best_val || (ov == best_val && oi < best_idx)) {
            best_val = ov;
            best_idx = oi;
        }
    }

    __shared__ float s_val[32];
    __shared__ int32_t s_idx[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) {
        s_val[warp] = best_val;
        s_idx[warp] = best_idx;
    }
    __syncthreads();

    if (warp == 0) {
        best_val = s_val[lane];
        best_idx = s_idx[lane];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, best_val, offset);
            int32_t oi = __shfl_down_sync(0xffffffffu, best_idx, offset);
            if (ov > best_val || (ov == best_val && oi < best_idx)) {
                best_val = ov;
                best_idx = oi;
            }
        }
        if (lane == 0) {
            int32_t tok = best_idx;
            int32_t step = step_idx[0];
            out_tokens[step] = tok;
            int32_t next_step = step + 1;
            step_idx[0] = next_step;
            input_ids[0] = tok;
            if (next_step < step_limit[0]) {
                int64_t new_pos = positions[0] + 1;
                positions[0] = new_pos;
                seq_lens[0] = static_cast<int32_t>(new_pos + 1);
                for (int g = 0; g < num_groups; ++g) {
                    int bs = block_sizes[g];
                    int64_t blk_idx = new_pos / bs;
                    int64_t blk_off = new_pos % bs;
                    const int32_t* bt = reinterpret_cast<const int32_t*>(input_block_table_ptrs[g]);
                    slot_mappings[g * slot_stride] = static_cast<int64_t>(bt[blk_idx]) * bs + blk_off;
                }
            }
        }
    }
}

void init_decode_loop(
    int64_t first_token,
    int64_t start_pos,
    int64_t count,
    at::Tensor step_idx,
    at::Tensor step_limit,
    at::Tensor input_ids,
    at::Tensor positions,
    at::Tensor seq_lens,
    at::Tensor query_start_loc,
    const at::Tensor& input_block_table_ptrs,
    const at::Tensor& block_sizes,
    at::Tensor slot_mappings
) {
    int num_groups = static_cast<int>(block_sizes.size(0));
    int64_t slot_stride = slot_mappings.stride(0);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    init_decode_loop_kernel<<<1, 1, 0, stream>>>(
        static_cast<int32_t>(first_token),
        start_pos,
        static_cast<int32_t>(count),
        step_idx.data_ptr<int32_t>(),
        step_limit.data_ptr<int32_t>(),
        input_ids.data_ptr<int32_t>(),
        positions.data_ptr<int64_t>(),
        seq_lens.data_ptr<int32_t>(),
        query_start_loc.data_ptr<int32_t>(),
        reinterpret_cast<const uint64_t*>(input_block_table_ptrs.data_ptr()),
        block_sizes.data_ptr<int32_t>(),
        slot_mappings.data_ptr<int64_t>(),
        slot_stride,
        num_groups
    );
}

void greedy_argmax_step_update(
    const at::Tensor& logits,
    at::Tensor out_tokens,
    at::Tensor step_idx,
    const at::Tensor& step_limit,
    at::Tensor input_ids,
    at::Tensor positions,
    at::Tensor seq_lens,
    const at::Tensor& input_block_table_ptrs,
    const at::Tensor& block_sizes,
    at::Tensor slot_mappings
) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous() && logits.dtype() == at::kBFloat16);
    int vocab_size = static_cast<int>(logits.numel());
    int num_groups = static_cast<int>(block_sizes.size(0));
    int64_t slot_stride = slot_mappings.stride(0);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    greedy_argmax_step_update_kernel<<<1, 1024, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
        vocab_size,
        out_tokens.data_ptr<int32_t>(),
        step_idx.data_ptr<int32_t>(),
        step_limit.data_ptr<int32_t>(),
        input_ids.data_ptr<int32_t>(),
        positions.data_ptr<int64_t>(),
        seq_lens.data_ptr<int32_t>(),
        reinterpret_cast<const uint64_t*>(input_block_table_ptrs.data_ptr()),
        block_sizes.data_ptr<int32_t>(),
        slot_mappings.data_ptr<int64_t>(),
        slot_stride,
        num_groups
    );
}

// ---------------------------------------------------------------------------
// 5b. Fused nvfp4_lmhead_m1 + output_multiplier + softcap + argmax + step update.
//     Eliminates the 404 KB logits DRAM buffer and 5 post-logit kernels per step.
// ---------------------------------------------------------------------------
__global__ void nvfp4_lmhead_m1_fused_argmax_kernel(
    const uint32_t* __restrict__ A,
    const uint4* __restrict__ B_q,
    const uint8_t* __restrict__ B_s,
    const float* __restrict__ g_scale,
    float output_multiplier,
    float softcap,
    float* __restrict__ blk_max_val,
    int32_t* __restrict__ blk_max_idx
) {
    constexpr int N = 202048;
    constexpr int K = 6656;
    constexpr int k_tiles = 52;
    constexpr int num_n64 = 3157;
    constexpr int p2_cols = 37;
    constexpr int p2_iters = 41;
    constexpr int row_stride_int4 = N >> 1;

    extern __shared__ uint32_t sA32[];
    const uint4* A4 = reinterpret_cast<const uint4*>(A);
    uint4* sA4 = reinterpret_cast<uint4*>(sA32);
    for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) {
        sA4[i] = A4[i];
    }
    __shared__ float s_acc[4][64];
    __syncthreads();

    int w = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int t = lane & 3;
    float gs = *g_scale;
    float my_best_val = -3.402823466e+38f;
    int32_t my_best_idx = 0;

    for (int n64 = blockIdx.x; n64 < num_n64; n64 += gridDim.x) {
        bool in_p2 = (n64 >= (num_n64 - p2_cols));
        int local_col = n64 - (num_n64 - p2_cols);
        int seg_s[4], seg_e[4], n_segs = 0;
        if (!in_p2) {
            seg_s[0] = 0;
            seg_e[0] = k_tiles;
            n_segs = 1;
        } else {
            int cs = local_col * k_tiles;
            int ce = cs + k_tiles;
            int b0 = cs / p2_iters;
            int b1 = (ce - 1) / p2_iters;
            for (int b = b1; b >= b0; --b) {
                seg_s[n_segs] = max(cs, b * p2_iters) - cs;
                seg_e[n_segs] = min(ce, (b + 1) * p2_iters) - cs;
                n_segs++;
            }
        }
        float R_reg = 0.f;
        bool has_R = false;
        for (int sg = 0; sg < n_segs; ++sg) {
            int seg_start = seg_s[sg], seg_end = seg_e[sg];
            float d0[8] = {0}, d1[8] = {0}, d2[8] = {0}, d3[8] = {0};
            int k16_base = seg_start * 8 + 2 * w;
            uint4 qv_cur;
            uint64_t sc_cur;
            asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(qv_cur.x), "=r"(qv_cur.y), "=r"(qv_cur.z), "=r"(qv_cur.w)
                : "l"(B_q + static_cast<uint64_t>(k16_base) * row_stride_int4 + n64 * 32 + lane));
            sc_cur = *reinterpret_cast<const uint64_t*>(
                B_s + static_cast<uint64_t>(k16_base) * N + n64 * 64 + group * 8);
            int n_steps = (seg_end - seg_start) * 2;
            for (int step = 0; step < n_steps; ++step) {
                int k16 = (seg_start + (step >> 1)) * 8 + 2 * w + (step & 1);
                uint4 qv_nxt;
                uint64_t sc_nxt;
                if (step + 1 < n_steps) {
                    int nk16 = (seg_start + ((step + 1) >> 1)) * 8 + 2 * w + ((step + 1) & 1);
                    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(qv_nxt.x), "=r"(qv_nxt.y), "=r"(qv_nxt.z), "=r"(qv_nxt.w)
                        : "l"(B_q + static_cast<uint64_t>(nk16) * row_stride_int4 + n64 * 32 + lane));
                    sc_nxt = *reinterpret_cast<const uint64_t*>(
                        B_s + static_cast<uint64_t>(nk16) * N + n64 * 64 + group * 8);
                }
                uint32_t a0 = (group == 0) ? sA32[k16 * 8 + t] : 0u;
                uint32_t a2 = (group == 0) ? sA32[k16 * 8 + 4 + t] : 0u;
                uint32_t q_arr[4] = {qv_cur.x, qv_cur.y, qv_cur.z, qv_cur.w};
                const uint8_t* sc = reinterpret_cast<const uint8_t*>(&sc_cur);
                uint8_t sb[8] = {sc[0], sc[2], sc[1], sc[3], sc[4], sc[6], sc[5], sc[7]};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t q = q_arr[j];
                    uint32_t q0 = q << 12, q1 = q << 8, q2 = q << 4;
                    uint32_t f0 = lop3_or_and(q0, 0x80008000u, (q0 >> 6) & 0x01C001C0u);
                    uint32_t f1 = lop3_or_and(q1, 0x80008000u, (q1 >> 6) & 0x01C001C0u);
                    uint32_t f2 = lop3_or_and(q2, 0x80008000u, (q2 >> 6) & 0x01C001C0u);
                    uint32_t f3 = lop3_or_and(q,  0x80008000u, (q  >> 6) & 0x01C001C0u);
                    __nv_bfloat162 s0 = dequant_s_bf162(sb[2 * j]);
                    __nv_bfloat162 s1 = dequant_s_bf162(sb[2 * j + 1]);
                    mma_m16n8k16_bf16_m1(d0[2 * j], d1[2 * j], d2[2 * j], d3[2 * j],
                                         a0, a2, hmul2_u32(f0, s0), hmul2_u32(f1, s0));
                    mma_m16n8k16_bf16_m1(d0[2 * j + 1], d1[2 * j + 1], d2[2 * j + 1], d3[2 * j + 1],
                                         a0, a2, hmul2_u32(f2, s1), hmul2_u32(f3, s1));
                }
                qv_cur = qv_nxt;
                sc_cur = sc_nxt;
            }
            if (group == 0) {
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    s_acc[w][m * 8 + t * 2] = d0[m];
                    s_acc[w][m * 8 + t * 2 + 1] = d1[m];
                }
            }
            __syncthreads();
            if (threadIdx.x < 64) {
                int c = threadIdx.x;
                float P = s_acc[0][c] + (s_acc[1][c] + (s_acc[3][c] + s_acc[2][c]));
                R_reg = has_R ? (P + R_reg) : P;
                has_R = true;
            }
            __syncthreads();
        }
        if (threadIdx.x < 64) {
            float v = __bfloat162float(__float2bfloat16_rn(R_reg * gs));
            v = __bfloat162float(__float2bfloat16_rn(v * output_multiplier));
            if (softcap > 0.f) {
                float d = __bfloat162float(__float2bfloat16_rn(v / softcap));
                float th = __bfloat162float(__float2bfloat16_rn(tanhf(d)));
                v = __bfloat162float(__float2bfloat16_rn(softcap * th));
            }
            int32_t idx = n64 * 64 + threadIdx.x;
            if (v > my_best_val || (v == my_best_val && idx < my_best_idx)) {
                my_best_val = v;
                my_best_idx = idx;
            }
        }
    }
    if (threadIdx.x < 64) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, my_best_val, off);
            int32_t oi = __shfl_down_sync(0xffffffffu, my_best_idx, off);
            if (ov > my_best_val || (ov == my_best_val && oi < my_best_idx)) {
                my_best_val = ov;
                my_best_idx = oi;
            }
        }
        if (lane == 0) {
            s_acc[0][w] = my_best_val;
            reinterpret_cast<int32_t*>(s_acc[1])[w] = my_best_idx;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v0 = s_acc[0][0], v1 = s_acc[0][1];
        int32_t i0 = reinterpret_cast<int32_t*>(s_acc[1])[0];
        int32_t i1 = reinterpret_cast<int32_t*>(s_acc[1])[1];
        if (v1 > v0 || (v1 == v0 && i1 < i0)) {
            v0 = v1;
            i0 = i1;
        }
        blk_max_val[blockIdx.x] = v0;
        blk_max_idx[blockIdx.x] = i0;
    }
}

__global__ void finalize_192_argmax_step_update_kernel(
    const float* __restrict__ blk_val,
    const int32_t* __restrict__ blk_idx,
    int32_t* __restrict__ out_tokens,
    int32_t* __restrict__ step_idx,
    const int32_t* __restrict__ step_limit,
    int32_t* __restrict__ input_ids,
    int64_t* __restrict__ positions,
    int32_t* __restrict__ seq_lens,
    const uint64_t* __restrict__ input_block_table_ptrs,
    const int32_t* __restrict__ block_sizes,
    int64_t* __restrict__ slot_mappings,
    int64_t slot_stride,
    int num_groups
) {
    int tid = threadIdx.x;
    float v = (tid < 192) ? blk_val[tid] : -3.402823466e+38f;
    int32_t idx = (tid < 192) ? blk_idx[tid] : 0;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, v, off);
        int32_t oi = __shfl_down_sync(0xffffffffu, idx, off);
        if (ov > v || (ov == v && oi < idx)) {
            v = ov;
            idx = oi;
        }
    }
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    if ((tid & 31) == 0) {
        sv[tid >> 5] = v;
        si[tid >> 5] = idx;
    }
    __syncthreads();
    if (tid == 0) {
        float bv = sv[0];
        int32_t bi = si[0];
        for (int i = 1; i < 6; ++i) {
            if (sv[i] > bv || (sv[i] == bv && si[i] < bi)) {
                bv = sv[i];
                bi = si[i];
            }
        }
        int32_t step = step_idx[0];
        out_tokens[step] = bi;
        int32_t next_step = step + 1;
        step_idx[0] = next_step;
        input_ids[0] = bi;
        if (next_step < step_limit[0]) {
            int64_t new_pos = positions[0] + 1;
            positions[0] = new_pos;
            seq_lens[0] = static_cast<int32_t>(new_pos + 1);
            for (int g = 0; g < num_groups; ++g) {
                int bs = block_sizes[g];
                int64_t blk_i = new_pos / bs;
                int64_t blk_o = new_pos % bs;
                const int32_t* bt = reinterpret_cast<const int32_t*>(input_block_table_ptrs[g]);
                slot_mappings[g * slot_stride] = static_cast<int64_t>(bt[blk_i]) * bs + blk_o;
            }
        }
    }
}

void nvfp4_lmhead_m1_argmax_step_update(
    const at::Tensor& A,
    const at::Tensor& B_q,
    const at::Tensor& B_s,
    const at::Tensor& g_scale,
    double output_multiplier,
    double softcap,
    at::Tensor blk_max_val,
    at::Tensor blk_max_idx,
    at::Tensor out_tokens,
    at::Tensor step_idx,
    const at::Tensor& step_limit,
    at::Tensor input_ids,
    at::Tensor positions,
    at::Tensor seq_lens,
    const at::Tensor& input_block_table_ptrs,
    const at::Tensor& block_sizes,
    at::Tensor slot_mappings
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1 && A.size(1) == 6656);
    int num_groups = static_cast<int>(block_sizes.size(0));
    int64_t slot_stride = slot_mappings.stride(0);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    nvfp4_lmhead_m1_fused_argmax_kernel<<<192, 128, 6656 * 2, stream>>>(
        reinterpret_cast<const uint32_t*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
        g_scale.data_ptr<float>(),
        static_cast<float>(output_multiplier),
        static_cast<float>(softcap),
        blk_max_val.data_ptr<float>(),
        blk_max_idx.data_ptr<int32_t>()
    );
    finalize_192_argmax_step_update_kernel<<<1, 192, 0, stream>>>(
        blk_max_val.data_ptr<float>(),
        blk_max_idx.data_ptr<int32_t>(),
        out_tokens.data_ptr<int32_t>(),
        step_idx.data_ptr<int32_t>(),
        step_limit.data_ptr<int32_t>(),
        input_ids.data_ptr<int32_t>(),
        positions.data_ptr<int64_t>(),
        seq_lens.data_ptr<int32_t>(),
        reinterpret_cast<const uint64_t*>(input_block_table_ptrs.data_ptr()),
        block_sizes.data_ptr<int32_t>(),
        slot_mappings.data_ptr<int64_t>(),
        slot_stride,
        num_groups
    );
}

// ---------------------------------------------------------------------------
// 6. Tile-Major Contiguous NVFP4 M=1 Repacker & Adaptive 5b/6b/8b GEMV Suite
//    Repacks Marlin [K/16, N*2] weights into [num_n64, k_pairs, 4_warps, 128]
//    contiguous uint4 tiles with lossless per-n64-tile adaptive scale packing:
//      - mode 5 (4b+1b, 10 uint4/block = 37.5% savings, ~96% of tiles)
//      - mode 6 (4b+2b, 12 uint4/block = 25.0% savings,  ~4% of tiles)
//      - mode 8 (8b fallback, 16 uint4/block, <0.01% of tiles)
//    Preserves exact Marlin rev=1 split-K reduction order for all 6 shapes.
// ---------------------------------------------------------------------------
__global__ void analyze_nvfp4_scales_kernel(
    const uint8_t* __restrict__ B_s,
    uint8_t* __restrict__ B_base,
    uint8_t* __restrict__ tile_s_mode,
    int N, int K
) {
    int n64 = blockIdx.x;
    int k_pairs = K >> 8;
    int w = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int g = lane >> 2;
    int sub = lane & 3;
    uint32_t warp_max_rng = 0;

    for (int kp = 0; kp < k_pairs; ++kp) {
        int kt = (kp << 1) + (sub >> 1);
        int k16 = (kt << 3) + (w << 1) + (sub & 1);
        const uint8_t* p = B_s + static_cast<uint64_t>(k16) * N + n64 * 64 + g * 8;
        uint32_t mn = 255u, mx = 0u;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t v = p[i];
            mn = min(mn, v);
            mx = max(mx, v);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            mn = min(mn, __shfl_xor_sync(0xffffffffu, mn, off));
            mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        }
        if (lane == 0) {
            B_base[(static_cast<uint64_t>(n64) * k_pairs + kp) * 4 + w] = static_cast<uint8_t>(mn);
            warp_max_rng = max(warp_max_rng, mx - mn);
        }
    }
    __shared__ uint32_t s_rng[4];
    if (lane == 0) s_rng[w] = warp_max_rng;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t r = max(max(s_rng[0], s_rng[1]), max(s_rng[2], s_rng[3]));
        tile_s_mode[n64] = (r <= 31u) ? 5 : ((r <= 63u) ? 6 : 8);
    }
}

__global__ void pack_nvfp4_m1_kernel(
    const uint4* __restrict__ B_q,
    const uint8_t* __restrict__ B_s,
    const uint8_t* __restrict__ B_base,
    const int64_t* __restrict__ tile_s_off,
    const uint8_t* __restrict__ tile_s_mode,
    uint4* __restrict__ Bq_tm,
    uint4* __restrict__ Bs_tm,
    int N, int K
) {
    int n64 = blockIdx.x;
    int k_pairs = K >> 8;
    int row_stride_int4 = N >> 1;
    int w = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    uint8_t mode = tile_s_mode[n64];
    int sc_u4 = (mode == 5) ? 10 : ((mode == 6) ? 12 : 16);
    int64_t base_s_off = tile_s_off[n64];

    for (int kp = 0; kp < k_pairs; ++kp) {
        uint4* q_dst = Bq_tm + ((static_cast<uint64_t>(n64) * k_pairs + kp) * 4 + w) * 128;
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int kt = (kp << 1) + (sub >> 1);
            int k16 = (kt << 3) + (w << 1) + (sub & 1);
            q_dst[sub * 32 + lane] = B_q[static_cast<uint64_t>(k16) * row_stride_int4 + n64 * 32 + lane];
        }
        if (lane < 8) {
            int g = lane;
            uint32_t base = B_base[(static_cast<uint64_t>(n64) * k_pairs + kp) * 4 + w];
            uint4* s_dst = Bs_tm + base_s_off + (static_cast<int64_t>(kp) * 4 + w) * sc_u4;
            uint32_t lo_w[4] = {0, 0, 0, 0};
            uint32_t hi_w0 = 0, hi_w1 = 0;
            uint32_t s8_lo[4] = {0, 0, 0, 0}, s8_hi[4] = {0, 0, 0, 0};

            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int kt = (kp << 1) + (sub >> 1);
                int k16 = (kt << 3) + (w << 1) + (sub & 1);
                const uint8_t* p = B_s + static_cast<uint64_t>(k16) * N + n64 * 64 + g * 8;
                uint32_t e[4] = {p[0], p[1], p[4], p[5]};
                uint32_t o[4] = {p[2], p[3], p[6], p[7]};
                if (mode == 8) {
                    s8_lo[sub] = e[0] | (e[1] << 8) | (e[2] << 16) | (e[3] << 24);
                    s8_hi[sub] = o[0] | (o[1] << 8) | (o[2] << 16) | (o[3] << 24);
                } else {
                    uint32_t lw = 0;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        uint32_t de = e[k] - base;
                        uint32_t do_ = o[k] - base;
                        lw |= ((de & 0xFu) | ((do_ & 0xFu) << 4)) << (k * 8);
                        if (mode == 5) {
                            hi_w0 |= ((((de >> 4) & 1u) << (sub * 2)) | (((do_ >> 4) & 1u) << (sub * 2 + 1))) << (k * 8);
                        } else {
                            uint32_t bits = (((de >> 4) & 3u) << ((sub & 1) * 4)) | (((do_ >> 4) & 3u) << ((sub & 1) * 4 + 2));
                            if (sub < 2) hi_w0 |= bits << (k * 8);
                            else hi_w1 |= bits << (k * 8);
                        }
                    }
                    lo_w[sub] = lw;
                }
            }
            if (mode == 8) {
                s_dst[g] = make_uint4(s8_lo[0], s8_hi[0], s8_lo[1], s8_hi[1]);
                s_dst[8 + g] = make_uint4(s8_lo[2], s8_hi[2], s8_lo[3], s8_hi[3]);
            } else if (mode == 6) {
                s_dst[g] = make_uint4(lo_w[0], lo_w[1], lo_w[2], lo_w[3]);
                uint32_t* hi_p = reinterpret_cast<uint32_t*>(s_dst + 8);
                hi_p[g * 2 + 0] = hi_w0;
                hi_p[g * 2 + 1] = hi_w1;
            } else {
                s_dst[g] = make_uint4(lo_w[0], lo_w[1], lo_w[2], lo_w[3]);
                uint32_t* hi_p = reinterpret_cast<uint32_t*>(s_dst + 8);
                hi_p[g] = hi_w0;
            }
        }
    }
}

// Pure compile-time specialized tile-major NVFP4 M=1 segment accumulator.
// SCALE_BITS is constexpr (5, 6, or 8) -> zero runtime mode branches, zero
// __noinline__ ABI stack spills, 100% pipelined across even & odd split-K
// boundaries, and per-tile B_base cached once in SMEM (0 B_base LDG in K-loop).
template <int SCALE_BITS, bool USE_LDG_A>
__device__ __forceinline__ float compute_n64_seg_tm_ct(
    const uint32_t* __restrict__ A,
    const uint32_t* __restrict__ sA32,
    const uint4* __restrict__ Bq_tm,
    const uint4* __restrict__ Bs_tm,
    const uint8_t* __restrict__ s_base,
    int64_t tile_s_off_val,
    int n64, int k_pairs, int seg_start, int seg_end,
    float (&s_acc)[4][64], uint4 (&s_sc)[4][16]
) {
    int w = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int t = lane & 3;
    constexpr int SC_U4 = (SCALE_BITS == 5) ? 10 : ((SCALE_BITS == 6) ? 12 : 16);

    float d0[8] = {0}, d1[8] = {0}, d2[8] = {0}, d3[8] = {0};
    const uint4* bq_tile = Bq_tm + static_cast<uint64_t>(n64) * k_pairs * 4 * 128;
    const uint4* bs_tile = Bs_tm + tile_s_off_val;

    auto load_q = [&](const uint4* p, uint4& q) {
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(q.x), "=r"(q.y), "=r"(q.z), "=r"(q.w) : "l"(p));
    };

    auto run_sub = [&](int kp, int sub, const uint4& qv, uint32_t sc_lo, uint32_t sc_hi) {
        int kt = (kp << 1) + (sub >> 1);
        int k16 = (kt << 3) + (w << 1) + (sub & 1);
        uint32_t a0, a2;
        if constexpr (USE_LDG_A) {
            a0 = (group == 0) ? __ldg(A + k16 * 8 + t) : 0u;
            a2 = (group == 0) ? __ldg(A + k16 * 8 + 4 + t) : 0u;
        } else {
            a0 = (group == 0) ? sA32[k16 * 8 + t] : 0u;
            a2 = (group == 0) ? sA32[k16 * 8 + 4 + t] : 0u;
        }
        uint32_t q_arr[4] = {qv.x, qv.y, qv.z, qv.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t q = q_arr[j];
            uint32_t q0 = q << 12, q1 = q << 8, q2 = q << 4;
            uint32_t f0 = lop3_or_and(q0, 0x80008000u, (q0 >> 6) & 0x01C001C0u);
            uint32_t f1 = lop3_or_and(q1, 0x80008000u, (q1 >> 6) & 0x01C001C0u);
            uint32_t f2 = lop3_or_and(q2, 0x80008000u, (q2 >> 6) & 0x01C001C0u);
            uint32_t f3 = lop3_or_and(q,  0x80008000u, (q  >> 6) & 0x01C001C0u);
            __nv_bfloat162 s0 = dequant_s_bf162((sc_lo >> (j * 8)) & 0xFFu);
            __nv_bfloat162 s1 = dequant_s_bf162((sc_hi >> (j * 8)) & 0xFFu);
            mma_m16n8k16_bf16_m1(d0[2 * j], d1[2 * j], d2[2 * j], d3[2 * j],
                                 a0, a2, hmul2_u32(f0, s0), hmul2_u32(f1, s0));
            mma_m16n8k16_bf16_m1(d0[2 * j + 1], d1[2 * j + 1], d2[2 * j + 1], d3[2 * j + 1],
                                 a0, a2, hmul2_u32(f2, s1), hmul2_u32(f3, s1));
        }
    };

    auto run_half_kp = [&](int kp, int sub_lo) {
        uint4 sc_tmp;
        uint32_t base_cur = (SCALE_BITS != 8) ? (static_cast<uint32_t>(s_base[kp * 4 + w]) * 0x01010101u) : 0u;
        if (lane < SC_U4) load_q(bs_tile + (kp * 4 + w) * SC_U4 + lane, sc_tmp);
        if (lane < SC_U4) s_sc[w][lane] = sc_tmp;
        __syncwarp();
        uint4 lo4 = s_sc[w][group];
        uint4 s8_hi = (SCALE_BITS == 8) ? s_sc[w][8 + group] : make_uint4(0, 0, 0, 0);
        const uint32_t* hi_p = reinterpret_cast<const uint32_t*>(&s_sc[w][8]);
        uint32_t hi_w0 = (SCALE_BITS == 6) ? hi_p[group * 2 + 0] : ((SCALE_BITS == 5) ? hi_p[group] : 0u);
        uint32_t hi_w1 = (SCALE_BITS == 6) ? hi_p[group * 2 + 1] : 0u;
        uint32_t lo_arr[4] = {lo4.x, lo4.y, lo4.z, lo4.w};
        uint32_t s8_arr[8] = {lo4.x, lo4.y, lo4.z, lo4.w, s8_hi.x, s8_hi.y, s8_hi.z, s8_hi.w};
        const uint4* qp_cur = bq_tile + (kp * 4 + w) * 128;
        uint4 q0, q1;
        load_q(qp_cur + sub_lo * 32 + lane, q0);
        load_q(qp_cur + (sub_lo + 1) * 32 + lane, q1);
        #pragma unroll
        for (int idx = 0; idx < 2; ++idx) {
            int sub = sub_lo + idx;
            uint32_t sc_lo, sc_hi;
            if constexpr (SCALE_BITS == 5) {
                uint32_t lw = lo_arr[sub];
                uint32_t e_hi = (sub <= 2) ? ((hi_w0 << (4 - sub * 2)) & 0x10101010u) : ((hi_w0 >> (sub * 2 - 4)) & 0x10101010u);
                uint32_t o_hi = (sub <= 1) ? ((hi_w0 << (3 - sub * 2)) & 0x10101010u) : ((hi_w0 >> (sub * 2 - 3)) & 0x10101010u);
                sc_lo = __vadd4((lw & 0x0F0F0F0Fu) | e_hi, base_cur);
                sc_hi = __vadd4(((lw >> 4) & 0x0F0F0F0Fu) | o_hi, base_cur);
            } else if constexpr (SCALE_BITS == 6) {
                uint32_t lw = lo_arr[sub];
                uint32_t hw = (sub < 2) ? hi_w0 : hi_w1;
                uint32_t e_hi = (sub & 1) ? (hw & 0x30303030u) : ((hw << 4) & 0x30303030u);
                uint32_t o_hi = (sub & 1) ? ((hw >> 2) & 0x30303030u) : ((hw << 2) & 0x30303030u);
                sc_lo = __vadd4((lw & 0x0F0F0F0Fu) | e_hi, base_cur);
                sc_hi = __vadd4(((lw >> 4) & 0x0F0F0F0Fu) | o_hi, base_cur);
            } else {
                sc_lo = s8_arr[sub * 2 + 0];
                sc_hi = s8_arr[sub * 2 + 1];
            }
            run_sub(kp, sub, idx == 0 ? q0 : q1, sc_lo, sc_hi);
        }
    };

    int cur_kt = seg_start;
    if ((cur_kt & 1) && cur_kt < seg_end) {
        run_half_kp(cur_kt >> 1, 2);
        cur_kt += 1;
    }

    int kp_start = cur_kt >> 1;
    int kp_end = seg_end >> 1;
    if (kp_start < kp_end) {
        uint4 q_cur[2];
        uint4 sc_cur;
        {
            const uint4* qp = bq_tile + (kp_start * 4 + w) * 128;
            load_q(qp + lane, q_cur[0]);
            load_q(qp + 32 + lane, q_cur[1]);
            if (lane < SC_U4) load_q(bs_tile + (kp_start * 4 + w) * SC_U4 + lane, sc_cur);
        }
        for (int kp = kp_start; kp < kp_end; ++kp) {
            if (lane < SC_U4) s_sc[w][lane] = sc_cur;
            __syncwarp();
            uint4 lo4 = s_sc[w][group];
            uint4 s8_hi = (SCALE_BITS == 8) ? s_sc[w][8 + group] : make_uint4(0, 0, 0, 0);
            const uint32_t* hi_p = reinterpret_cast<const uint32_t*>(&s_sc[w][8]);
            uint32_t hi_w0 = (SCALE_BITS == 6) ? hi_p[group * 2 + 0] : ((SCALE_BITS == 5) ? hi_p[group] : 0u);
            uint32_t hi_w1 = (SCALE_BITS == 6) ? hi_p[group * 2 + 1] : 0u;
            uint32_t base_cur = (SCALE_BITS != 8) ? (static_cast<uint32_t>(s_base[kp * 4 + w]) * 0x01010101u) : 0u;

            uint4 sc_nxt;
            if (kp + 1 < kp_end) {
                if (lane < SC_U4) load_q(bs_tile + ((kp + 1) * 4 + w) * SC_U4 + lane, sc_nxt);
            }
            const uint4* qp_cur = bq_tile + (kp * 4 + w) * 128;
            const uint4* qp_nxt = bq_tile + ((kp + 1) * 4 + w) * 128;
            uint32_t lo_arr[4] = {lo4.x, lo4.y, lo4.z, lo4.w};
            uint32_t s8_arr[8] = {lo4.x, lo4.y, lo4.z, lo4.w, s8_hi.x, s8_hi.y, s8_hi.z, s8_hi.w};

            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                uint4 qv = q_cur[sub & 1];
                if (sub < 2) {
                    load_q(qp_cur + (sub + 2) * 32 + lane, q_cur[sub & 1]);
                } else if (kp + 1 < kp_end) {
                    load_q(qp_nxt + (sub - 2) * 32 + lane, q_cur[sub & 1]);
                }
                uint32_t sc_lo, sc_hi;
                if constexpr (SCALE_BITS == 5) {
                    uint32_t lw = lo_arr[sub];
                    uint32_t e_hi = (sub <= 2) ? ((hi_w0 << (4 - sub * 2)) & 0x10101010u) : ((hi_w0 >> (sub * 2 - 4)) & 0x10101010u);
                    uint32_t o_hi = (sub <= 1) ? ((hi_w0 << (3 - sub * 2)) & 0x10101010u) : ((hi_w0 >> (sub * 2 - 3)) & 0x10101010u);
                    sc_lo = __vadd4((lw & 0x0F0F0F0Fu) | e_hi, base_cur);
                    sc_hi = __vadd4(((lw >> 4) & 0x0F0F0F0Fu) | o_hi, base_cur);
                } else if constexpr (SCALE_BITS == 6) {
                    uint32_t lw = lo_arr[sub];
                    uint32_t hw = (sub < 2) ? hi_w0 : hi_w1;
                    uint32_t e_hi = (sub & 1) ? (hw & 0x30303030u) : ((hw << 4) & 0x30303030u);
                    uint32_t o_hi = (sub & 1) ? ((hw >> 2) & 0x30303030u) : ((hw << 2) & 0x30303030u);
                    sc_lo = __vadd4((lw & 0x0F0F0F0Fu) | e_hi, base_cur);
                    sc_hi = __vadd4(((lw >> 4) & 0x0F0F0F0Fu) | o_hi, base_cur);
                } else {
                    sc_lo = s8_arr[sub * 2 + 0];
                    sc_hi = s8_arr[sub * 2 + 1];
                }
                run_sub(kp, sub, qv, sc_lo, sc_hi);
            }
            sc_cur = sc_nxt;
        }
    }

    if ((seg_end & 1) && (kp_end * 2) < seg_end) {
        run_half_kp(kp_end, 0);
    }

    if (group == 0) {
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            s_acc[w][m * 8 + t * 2] = d0[m];
            s_acc[w][m * 8 + t * 2 + 1] = d1[m];
        }
    }
    __syncthreads();
    float P = 0.f;
    if (threadIdx.x < 64) {
        int c = threadIdx.x;
        P = s_acc[0][c] + (s_acc[1][c] + (s_acc[3][c] + s_acc[2][c]));
    }
    __syncthreads();
    return P;
}

template <int SCALE_BITS, bool USE_LDG_A>
__global__ void nvfp4_tm_linear_m1_kernel(
    const uint32_t* __restrict__ A,
    const uint4* __restrict__ Bq_tm,
    const uint4* __restrict__ Bs_tm,
    const uint8_t* __restrict__ B_base,
    const int64_t* __restrict__ tile_s_off,
    const uint8_t* __restrict__ tile_s_mode,
    const float* __restrict__ g_scale,
    __nv_bfloat16* __restrict__ C,
    int N, int K, int n_grp, int p1_cols, int p2_iters
) {
    (void)tile_s_mode;
    extern __shared__ uint32_t sA32[];
    if constexpr (!USE_LDG_A) {
        const uint4* A4 = reinterpret_cast<const uint4*>(A);
        uint4* sA4 = reinterpret_cast<uint4*>(sA32);
        for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) {
            sA4[i] = A4[i];
        }
    }
    __shared__ float s_acc[4][64];
    __shared__ uint4 s_sc[4][16];
    __shared__ uint8_t s_base[320]; // holds up to k_pairs=80 x 4 warps (K<=20480)
    __syncthreads();

    int num_n64 = N >> 6;
    int k_tiles = K >> 7;
    int k_pairs = k_tiles >> 1;
    int n_base_bytes = k_pairs * 4;
    float gs = *g_scale;

    for (int n64 = blockIdx.x; n64 < num_n64; n64 += gridDim.x) {
        if constexpr (SCALE_BITS != 8) {
            const uint8_t* bb = B_base + static_cast<uint64_t>(n64) * n_base_bytes;
            for (int i = threadIdx.x; i < n_base_bytes; i += blockDim.x) {
                s_base[i] = bb[i];
            }
            __syncthreads();
        }
        int col = (n_grp == 128) ? (n64 >> 1) : n64;
        int64_t soff = tile_s_off[n64];

        int seg_s[8], seg_e[8], n_segs = 0;
        if (col < p1_cols) {
            seg_s[0] = 0; seg_e[0] = k_tiles; n_segs = 1;
        } else {
            int local_col = col - p1_cols;
            int cs = local_col * k_tiles;
            int ce = cs + k_tiles;
            int b0 = cs / p2_iters;
            int b1 = (ce - 1) / p2_iters;
            for (int b = b1; b >= b0; --b) {
                seg_s[n_segs] = max(cs, b * p2_iters) - cs;
                seg_e[n_segs] = min(ce, (b + 1) * p2_iters) - cs;
                n_segs++;
            }
        }
        float R_reg = 0.f;
        bool has_R = false;
        for (int sg = 0; sg < n_segs; ++sg) {
            float P = compute_n64_seg_tm_ct<SCALE_BITS, USE_LDG_A>(
                A, sA32, Bq_tm, Bs_tm, s_base, soff, n64, k_pairs, seg_s[sg], seg_e[sg], s_acc, s_sc);
            if (threadIdx.x < 64) {
                R_reg = has_R ? (P + R_reg) : P;
                has_R = true;
            }
        }
        if (threadIdx.x < 64) {
            C[n64 * 64 + threadIdx.x] = __float2bfloat16_rn(R_reg * gs);
        }
    }
}

template <int SCALE_BITS>
__global__ void nvfp4_tm_gate_up_silu_m1_kernel(
    const uint32_t* __restrict__ A,
    const uint4* __restrict__ Bq_tm,
    const uint4* __restrict__ Bs_tm,
    const uint8_t* __restrict__ B_base,
    const int64_t* __restrict__ tile_s_off,
    const uint8_t* __restrict__ tile_s_mode,
    const float* __restrict__ g_scale,
    __nv_bfloat16* __restrict__ C
) {
    (void)tile_s_mode;
    constexpr int K = 6656;
    constexpr int k_pairs = 26;
    constexpr int half_n64 = 312;
    constexpr int n_base_bytes = k_pairs * 4;

    extern __shared__ uint32_t sA32[];
    const uint4* A4 = reinterpret_cast<const uint4*>(A);
    uint4* sA4 = reinterpret_cast<uint4*>(sA32);
    for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) {
        sA4[i] = A4[i];
    }
    __shared__ float s_acc[4][64];
    __shared__ uint4 s_sc[4][16];
    __shared__ uint8_t s_base[128];
    __syncthreads();

    float gs = *g_scale;
    for (int n_blk = blockIdx.x; n_blk < half_n64; n_blk += gridDim.x) {
        int up_n64 = half_n64 + n_blk;
        float vals[2] = {0.f, 0.f};
        int tiles[2] = {n_blk, up_n64};
        for (int h = 0; h < 2; ++h) {
            int tn = tiles[h];
            if constexpr (SCALE_BITS != 8) {
                const uint8_t* bb = B_base + static_cast<uint64_t>(tn) * n_base_bytes;
                if (threadIdx.x < n_base_bytes) s_base[threadIdx.x] = bb[threadIdx.x];
                __syncthreads();
            }
            int64_t soff = tile_s_off[tn];
            int seg_s[2], seg_e[2], n_segs;
            if (h == 1 && tn >= 576) {
                seg_s[0] = 26; seg_e[0] = 52;
                seg_s[1] = 0;  seg_e[1] = 26;
                n_segs = 2;
            } else {
                seg_s[0] = 0; seg_e[0] = 52;
                n_segs = 1;
            }
            float R = 0.f;
            bool has_R = false;
            for (int sg = 0; sg < n_segs; ++sg) {
                float P = compute_n64_seg_tm_ct<SCALE_BITS, false>(
                    A, sA32, Bq_tm, Bs_tm, s_base, soff, tn, k_pairs, seg_s[sg], seg_e[sg], s_acc, s_sc);
                if (threadIdx.x < 64) {
                    R = has_R ? (P + R) : P;
                    has_R = true;
                }
            }
            vals[h] = R;
        }
        if (threadIdx.x < 64) {
            float g = __bfloat162float(__float2bfloat16_rn(vals[0] * gs));
            float u = __bfloat162float(__float2bfloat16_rn(vals[1] * gs));
            C[n_blk * 64 + threadIdx.x] = __float2bfloat16_rn((g / (1.0f + expf(-g))) * u);
        }
    }
}

template <int SCALE_BITS>
__global__ void nvfp4_tm_lmhead_m1_fused_argmax_kernel(
    const uint32_t* __restrict__ A,
    const uint4* __restrict__ Bq_tm,
    const uint4* __restrict__ Bs_tm,
    const uint8_t* __restrict__ B_base,
    const int64_t* __restrict__ tile_s_off,
    const uint8_t* __restrict__ tile_s_mode,
    const float* __restrict__ g_scale,
    float output_multiplier,
    float softcap,
    float* __restrict__ blk_max_val,
    int32_t* __restrict__ blk_max_idx
) {
    (void)tile_s_mode;
    constexpr int K = 6656;
    constexpr int k_tiles = 52;
    constexpr int k_pairs = 26;
    constexpr int num_n64 = 3157;
    constexpr int p1_cols = 3120;
    constexpr int p2_iters = 41;
    constexpr int n_base_bytes = k_pairs * 4;

    extern __shared__ uint32_t sA32[];
    const uint4* A4 = reinterpret_cast<const uint4*>(A);
    uint4* sA4 = reinterpret_cast<uint4*>(sA32);
    for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) {
        sA4[i] = A4[i];
    }
    __shared__ float s_acc[4][64];
    __shared__ uint4 s_sc[4][16];
    __shared__ uint8_t s_base[128];
    __syncthreads();

    int w = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    float gs = *g_scale;
    float my_best_val = -3.402823466e+38f;
    int32_t my_best_idx = 0;

    for (int n64 = blockIdx.x; n64 < num_n64; n64 += gridDim.x) {
        if constexpr (SCALE_BITS != 8) {
            const uint8_t* bb = B_base + static_cast<uint64_t>(n64) * n_base_bytes;
            if (threadIdx.x < n_base_bytes) s_base[threadIdx.x] = bb[threadIdx.x];
            __syncthreads();
        }
        int64_t soff = tile_s_off[n64];
        int seg_s[4], seg_e[4], n_segs = 0;
        if (n64 < p1_cols) {
            seg_s[0] = 0; seg_e[0] = k_tiles; n_segs = 1;
        } else {
            int local_col = n64 - p1_cols;
            int cs = local_col * k_tiles;
            int ce = cs + k_tiles;
            int b0 = cs / p2_iters;
            int b1 = (ce - 1) / p2_iters;
            for (int b = b1; b >= b0; --b) {
                seg_s[n_segs] = max(cs, b * p2_iters) - cs;
                seg_e[n_segs] = min(ce, (b + 1) * p2_iters) - cs;
                n_segs++;
            }
        }
        float R_reg = 0.f;
        bool has_R = false;
        for (int sg = 0; sg < n_segs; ++sg) {
            float P = compute_n64_seg_tm_ct<SCALE_BITS, false>(
                A, sA32, Bq_tm, Bs_tm, s_base, soff, n64, k_pairs, seg_s[sg], seg_e[sg], s_acc, s_sc);
            if (threadIdx.x < 64) {
                R_reg = has_R ? (P + R_reg) : P;
                has_R = true;
            }
        }
        if (threadIdx.x < 64) {
            float v = __bfloat162float(__float2bfloat16_rn(R_reg * gs));
            v = __bfloat162float(__float2bfloat16_rn(v * output_multiplier));
            if (softcap > 0.f) {
                float d = __bfloat162float(__float2bfloat16_rn(v / softcap));
                float th = __bfloat162float(__float2bfloat16_rn(tanhf(d)));
                v = __bfloat162float(__float2bfloat16_rn(softcap * th));
            }
            int32_t idx = n64 * 64 + threadIdx.x;
            if (v > my_best_val || (v == my_best_val && idx < my_best_idx)) {
                my_best_val = v;
                my_best_idx = idx;
            }
        }
    }
    if (threadIdx.x < 64) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, my_best_val, off);
            int32_t oi = __shfl_down_sync(0xffffffffu, my_best_idx, off);
            if (ov > my_best_val || (ov == my_best_val && oi < my_best_idx)) {
                my_best_val = ov;
                my_best_idx = oi;
            }
        }
        if (lane == 0) {
            s_acc[0][w] = my_best_val;
            reinterpret_cast<int32_t*>(s_acc[1])[w] = my_best_idx;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v0 = s_acc[0][0], v1 = s_acc[0][1];
        int32_t i0 = reinterpret_cast<int32_t*>(s_acc[1])[0];
        int32_t i1 = reinterpret_cast<int32_t*>(s_acc[1])[1];
        if (v1 > v0 || (v1 == v0 && i1 < i0)) {
            v0 = v1;
            i0 = i1;
        }
        blk_max_val[blockIdx.x] = v0;
        blk_max_idx[blockIdx.x] = i0;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> repack_nvfp4_m1(
    const at::Tensor& B_q,
    const at::Tensor& B_s,
    int64_t N_in,
    int64_t K_in
) {
    int N = static_cast<int>(N_in);
    int K = static_cast<int>(K_in);
    TORCH_CHECK(B_q.is_cuda() && B_s.is_cuda() && N % 64 == 0 && K % 256 == 0);
    int num_n64 = N >> 6;
    int k_pairs = K >> 8;
    auto opts_u8 = B_q.options().dtype(at::kByte);
    auto opts_i32 = B_q.options().dtype(at::kInt);

    auto B_base = at::empty({static_cast<int64_t>(num_n64) * k_pairs * 4}, opts_u8);
    auto tile_s_mode = at::empty({num_n64}, opts_u8);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    analyze_nvfp4_scales_kernel<<<num_n64, 128, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
        B_base.data_ptr<uint8_t>(),
        tile_s_mode.data_ptr<uint8_t>(),
        N, K
    );

    // Promote whole layer to a single uniform bit-width (max across tiles: 5, 6, or 8)
    // so decode runs in ONE single balanced full-grid kernel with zero fallback passes.
    auto mode_cpu = tile_s_mode.to(at::kCPU);
    uint8_t* m_ptr = mode_cpu.data_ptr<uint8_t>();
    uint8_t layer_mode = 5;
    for (int i = 0; i < num_n64; ++i) {
        if (m_ptr[i] > layer_mode) layer_mode = m_ptr[i];
    }
    for (int i = 0; i < num_n64; ++i) m_ptr[i] = layer_mode;
    tile_s_mode.copy_(mode_cpu);

    auto off_cpu = at::empty({num_n64}, at::TensorOptions().dtype(at::kLong));
    int64_t* o_ptr = off_cpu.data_ptr<int64_t>();
    int sc_u4 = (layer_mode == 5) ? 10 : ((layer_mode == 6) ? 12 : 16);
    int64_t total_u4 = 0;
    for (int i = 0; i < num_n64; ++i) {
        o_ptr[i] = total_u4;
        total_u4 += static_cast<int64_t>(k_pairs) * 4 * sc_u4;
    }
    auto tile_s_off = off_cpu.to(B_q.device(), /*non_blocking=*/false);
    auto Bq_tm = at::empty({static_cast<int64_t>(num_n64) * k_pairs * 4 * 128 * 4}, opts_i32);
    auto Bs_tm = at::empty({total_u4 * 4}, opts_i32);

    pack_nvfp4_m1_kernel<<<num_n64, 128, 0, stream>>>(
        reinterpret_cast<const uint4*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
        B_base.data_ptr<uint8_t>(),
        tile_s_off.data_ptr<int64_t>(),
        tile_s_mode.data_ptr<uint8_t>(),
        reinterpret_cast<uint4*>(Bq_tm.data_ptr()),
        reinterpret_cast<uint4*>(Bs_tm.data_ptr()),
        N, K
    );
    return std::make_tuple(Bq_tm, Bs_tm, B_base, tile_s_off, tile_s_mode);
}

at::Tensor nvfp4_tm_linear_m1(
    const at::Tensor& A,
    const at::Tensor& Bq_tm,
    const at::Tensor& Bs_tm,
    const at::Tensor& B_base,
    const at::Tensor& tile_s_off,
    const at::Tensor& tile_s_mode,
    const at::Tensor& g_scale,
    int64_t N_in,
    int64_t n_grp_in,
    int64_t p1_cols_in,
    int64_t p2_iters_in,
    int64_t grid_in,
    int64_t mode_mask
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1);
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(N_in);
    auto C = at::empty({1, N}, A.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int grid = static_cast<int>(grid_in);
    if (K > 8192) {
        if (mode_mask & 1) {
            nvfp4_tm_linear_m1_kernel<5, true><<<grid, 128, 0, stream>>>(
                reinterpret_cast<const uint32_t*>(A.data_ptr()),
                reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
                reinterpret_cast<const uint4*>(Bs_tm.data_ptr()),
                B_base.data_ptr<uint8_t>(),
                tile_s_off.data_ptr<int64_t>(),
                tile_s_mode.data_ptr<uint8_t>(),
                g_scale.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
                N, K, static_cast<int>(n_grp_in), static_cast<int>(p1_cols_in), static_cast<int>(p2_iters_in)
            );
        } else if (mode_mask & 2) {
            nvfp4_tm_linear_m1_kernel<6, true><<<grid, 128, 0, stream>>>(
                reinterpret_cast<const uint32_t*>(A.data_ptr()),
                reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
                reinterpret_cast<const uint4*>(Bs_tm.data_ptr()),
                B_base.data_ptr<uint8_t>(),
                tile_s_off.data_ptr<int64_t>(),
                tile_s_mode.data_ptr<uint8_t>(),
                g_scale.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
                N, K, static_cast<int>(n_grp_in), static_cast<int>(p1_cols_in), static_cast<int>(p2_iters_in)
            );
        } else {
            nvfp4_tm_linear_m1_kernel<8, true><<<grid, 128, 0, stream>>>(
                reinterpret_cast<const uint32_t*>(A.data_ptr()),
                reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
                reinterpret_cast<const uint4*>(Bs_tm.data_ptr()),
                B_base.data_ptr<uint8_t>(),
                tile_s_off.data_ptr<int64_t>(),
                tile_s_mode.data_ptr<uint8_t>(),
                g_scale.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
                N, K, static_cast<int>(n_grp_in), static_cast<int>(p1_cols_in), static_cast<int>(p2_iters_in)
            );
        }
    } else {
        if (mode_mask & 1) {
            nvfp4_tm_linear_m1_kernel<5, false><<<grid, 128, K * 2, stream>>>(
                reinterpret_cast<const uint32_t*>(A.data_ptr()),
                reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
                reinterpret_cast<const uint4*>(Bs_tm.data_ptr()),
                B_base.data_ptr<uint8_t>(),
                tile_s_off.data_ptr<int64_t>(),
                tile_s_mode.data_ptr<uint8_t>(),
                g_scale.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
                N, K, static_cast<int>(n_grp_in), static_cast<int>(p1_cols_in), static_cast<int>(p2_iters_in)
            );
        } else if (mode_mask & 2) {
            nvfp4_tm_linear_m1_kernel<6, false><<<grid, 128, K * 2, stream>>>(
                reinterpret_cast<const uint32_t*>(A.data_ptr()),
                reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
                reinterpret_cast<const uint4*>(Bs_tm.data_ptr()),
                B_base.data_ptr<uint8_t>(),
                tile_s_off.data_ptr<int64_t>(),
                tile_s_mode.data_ptr<uint8_t>(),
                g_scale.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
                N, K, static_cast<int>(n_grp_in), static_cast<int>(p1_cols_in), static_cast<int>(p2_iters_in)
            );
        } else {
            nvfp4_tm_linear_m1_kernel<8, false><<<grid, 128, K * 2, stream>>>(
                reinterpret_cast<const uint32_t*>(A.data_ptr()),
                reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
                reinterpret_cast<const uint4*>(Bs_tm.data_ptr()),
                B_base.data_ptr<uint8_t>(),
                tile_s_off.data_ptr<int64_t>(),
                tile_s_mode.data_ptr<uint8_t>(),
                g_scale.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
                N, K, static_cast<int>(n_grp_in), static_cast<int>(p1_cols_in), static_cast<int>(p2_iters_in)
            );
        }
    }
    return C;
}

at::Tensor nvfp4_tm_gate_up_silu_m1(
    const at::Tensor& A,
    const at::Tensor& Bq_tm,
    const at::Tensor& Bs_tm,
    const at::Tensor& B_base,
    const at::Tensor& tile_s_off,
    const at::Tensor& tile_s_mode,
    const at::Tensor& g_scale,
    int64_t mode_mask
) {
    constexpr int half_N = 19968;
    constexpr int K = 6656;
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1 && A.size(1) == K);
    auto C = at::empty({1, half_N}, A.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (mode_mask & 1) {
        nvfp4_tm_gate_up_silu_m1_kernel<5><<<156, 128, K * 2, stream>>>(
            reinterpret_cast<const uint32_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
            reinterpret_cast<const uint4*>(Bs_tm.data_ptr()),
            B_base.data_ptr<uint8_t>(),
            tile_s_off.data_ptr<int64_t>(),
            tile_s_mode.data_ptr<uint8_t>(),
            g_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr())
        );
    } else if (mode_mask & 2) {
        nvfp4_tm_gate_up_silu_m1_kernel<6><<<156, 128, K * 2, stream>>>(
            reinterpret_cast<const uint32_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
            reinterpret_cast<const uint4*>(Bs_tm.data_ptr()),
            B_base.data_ptr<uint8_t>(),
            tile_s_off.data_ptr<int64_t>(),
            tile_s_mode.data_ptr<uint8_t>(),
            g_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr())
        );
    } else {
        nvfp4_tm_gate_up_silu_m1_kernel<8><<<156, 128, K * 2, stream>>>(
            reinterpret_cast<const uint32_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
            reinterpret_cast<const uint4*>(Bs_tm.data_ptr()),
            B_base.data_ptr<uint8_t>(),
            tile_s_off.data_ptr<int64_t>(),
            tile_s_mode.data_ptr<uint8_t>(),
            g_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr())
        );
    }
    return C;
}

void nvfp4_tm_lmhead_m1_argmax_step_update(
    const at::Tensor& A,
    const at::Tensor& Bq_tm,
    const at::Tensor& Bs_tm,
    const at::Tensor& B_base,
    const at::Tensor& tile_s_off,
    const at::Tensor& tile_s_mode,
    const at::Tensor& g_scale,
    double output_multiplier,
    double softcap,
    at::Tensor blk_max_val,
    at::Tensor blk_max_idx,
    at::Tensor out_tokens,
    at::Tensor step_idx,
    const at::Tensor& step_limit,
    at::Tensor input_ids,
    at::Tensor positions,
    at::Tensor seq_lens,
    const at::Tensor& input_block_table_ptrs,
    const at::Tensor& block_sizes,
    at::Tensor slot_mappings,
    int64_t mode_mask
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1 && A.size(1) == 6656);
    int num_groups = static_cast<int>(block_sizes.size(0));
    int64_t slot_stride = slot_mappings.stride(0);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (mode_mask & 1) {
        nvfp4_tm_lmhead_m1_fused_argmax_kernel<5><<<192, 128, 6656 * 2, stream>>>(
            reinterpret_cast<const uint32_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
            reinterpret_cast<const uint4*>(Bs_tm.data_ptr()),
            B_base.data_ptr<uint8_t>(),
            tile_s_off.data_ptr<int64_t>(),
            tile_s_mode.data_ptr<uint8_t>(),
            g_scale.data_ptr<float>(),
            static_cast<float>(output_multiplier),
            static_cast<float>(softcap),
            blk_max_val.data_ptr<float>(),
            blk_max_idx.data_ptr<int32_t>()
        );
    } else if (mode_mask & 2) {
        nvfp4_tm_lmhead_m1_fused_argmax_kernel<6><<<192, 128, 6656 * 2, stream>>>(
            reinterpret_cast<const uint32_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
            reinterpret_cast<const uint4*>(Bs_tm.data_ptr()),
            B_base.data_ptr<uint8_t>(),
            tile_s_off.data_ptr<int64_t>(),
            tile_s_mode.data_ptr<uint8_t>(),
            g_scale.data_ptr<float>(),
            static_cast<float>(output_multiplier),
            static_cast<float>(softcap),
            blk_max_val.data_ptr<float>(),
            blk_max_idx.data_ptr<int32_t>()
        );
    } else {
        nvfp4_tm_lmhead_m1_fused_argmax_kernel<8><<<192, 128, 6656 * 2, stream>>>(
            reinterpret_cast<const uint32_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
            reinterpret_cast<const uint4*>(Bs_tm.data_ptr()),
            B_base.data_ptr<uint8_t>(),
            tile_s_off.data_ptr<int64_t>(),
            tile_s_mode.data_ptr<uint8_t>(),
            g_scale.data_ptr<float>(),
            static_cast<float>(output_multiplier),
            static_cast<float>(softcap),
            blk_max_val.data_ptr<float>(),
            blk_max_idx.data_ptr<int32_t>()
        );
    }
    finalize_192_argmax_step_update_kernel<<<1, 192, 0, stream>>>(
        blk_max_val.data_ptr<float>(),
        blk_max_idx.data_ptr<int32_t>(),
        out_tokens.data_ptr<int32_t>(),
        step_idx.data_ptr<int32_t>(),
        step_limit.data_ptr<int32_t>(),
        input_ids.data_ptr<int32_t>(),
        positions.data_ptr<int64_t>(),
        seq_lens.data_ptr<int32_t>(),
        reinterpret_cast<const uint64_t*>(input_block_table_ptrs.data_ptr()),
        block_sizes.data_ptr<int32_t>(),
        slot_mappings.data_ptr<int64_t>(),
        slot_stride,
        num_groups
    );
}

// ---------------------------------------------------------------------------
// 7. Original 32-lane Warp-Interleaved Lossless NVFP4 Block-Scale rANS Suite
//    - Compresses 8-bit E4M3 NVFP4 block scales (~3.815 bits/scale entropy) via
//      10-bit (4 KB SMEM LUT) 32-lane interleaved rANS into 256B-aligned chunks.
//    - Repacks Bq_tm per-warp in exact Marlin rev=1 split-K execution order so
//      both Bq_tm and sc_stream walk strictly linearly across all segments.
//    - Covers all 132 NVFP4 layers (gate_up, down, qkv, gate, oproj, lm_head)
//      with 0 bit mismatches vs Marlin.
// ---------------------------------------------------------------------------
__global__ void nvfp4_sc_hist_kernel(
    const uint32_t* __restrict__ Bs_u32,
    int64_t num_u32,
    unsigned long long* __restrict__ g_hist
) {
    __shared__ uint32_t s_hist[256];
    if (threadIdx.x < 256) s_hist[threadIdx.x] = 0u;
    __syncthreads();
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = idx; i < num_u32; i += stride) {
        uint32_t w = Bs_u32[i];
        atomicAdd(&s_hist[w & 0xFFu], 1u);
        atomicAdd(&s_hist[(w >> 8) & 0xFFu], 1u);
        atomicAdd(&s_hist[(w >> 16) & 0xFFu], 1u);
        atomicAdd(&s_hist[(w >> 24) & 0xFFu], 1u);
    }
    __syncthreads();
    if (threadIdx.x < 256 && s_hist[threadIdx.x] > 0u) {
        atomicAdd(&g_hist[threadIdx.x], static_cast<unsigned long long>(s_hist[threadIdx.x]));
    }
}

__global__ void nvfp4_sc_rans_measure_bq_kernel(
    const uint4* __restrict__ B_q,
    const uint8_t* __restrict__ B_s,
    const uint32_t* __restrict__ d_freq,
    const uint32_t* __restrict__ d_cdf,
    uint4* __restrict__ Bq_tm,
    int32_t* __restrict__ stream_u16,
    int64_t* __restrict__ stream_chunks,
    int N, int K, int n_grp, int p1_cols, int p2_iters
) {
    int sid = blockIdx.x;
    int n64 = sid >> 2;
    int w = sid & 3;
    int lane = threadIdx.x;
    int g = lane >> 2;
    int part = lane & 3;
    int sub_s = part >> 1;
    int is_odd = part & 1;
    int k_tiles = K >> 7;
    int row_stride_int4 = N >> 1;

    int col = (n_grp == 128) ? (n64 >> 1) : n64;
    int cs = (col - p1_cols) * k_tiles;
    int b_lo = (col < p1_cols) ? 0 : (cs / p2_iters);
    int b_hi = (col < p1_cols) ? 0 : ((cs + k_tiles - 1) / p2_iters);

    // 1. Pack Bq_tm per-warp strictly sequentially in execution order.
    uint4* q_dst = Bq_tm + static_cast<uint64_t>(sid) * k_tiles * 64;
    int kt_ord = 0;
    for (int b = b_hi; b >= b_lo; --b) {
        int seg_s = (col < p1_cols) ? 0 : (max(cs, b * p2_iters) - cs);
        int seg_e = (col < p1_cols) ? k_tiles : (min(cs + k_tiles, (b + 1) * p2_iters) - cs);
        for (int kt = seg_s; kt < seg_e; ++kt, ++kt_ord) {
            int k16_0 = kt * 8 + w * 2;
            int k16_1 = k16_0 + 1;
            q_dst[kt_ord * 64 + lane] = B_q[static_cast<uint64_t>(k16_0) * row_stride_int4 + n64 * 32 + lane];
            q_dst[kt_ord * 64 + 32 + lane] = B_q[static_cast<uint64_t>(k16_1) * row_stride_int4 + n64 * 32 + lane];
        }
    }

    // 2. Measure exact 4-way ILP warp-interleaved 10-bit rANS u16 length in reverse order.
    uint32_t st0 = 65536u, st1 = 65536u, st2 = 65536u, st3 = 65536u;
    uint32_t total_u16 = 0u;
    #define SC_MEAS_SYM4(st_v, sym_val) do { \
        uint32_t _s = (sym_val); \
        uint32_t _f = d_freq[_s]; \
        uint32_t _c = d_cdf[_s]; \
        bool _em = ((st_v) >= (_f << 22)); \
        uint32_t _m = __ballot_sync(0xffffffffu, _em); \
        total_u16 += static_cast<uint32_t>(__popc(_m)); \
        if (_em) (st_v) >>= 16; \
        (st_v) = (((st_v) / _f) << 10) + ((st_v) % _f) + _c; \
    } while (0)

    for (int b = b_lo; b <= b_hi; ++b) {
        int seg_s = (col < p1_cols) ? 0 : (max(cs, b * p2_iters) - cs);
        int seg_e = (col < p1_cols) ? k_tiles : (min(cs + k_tiles, (b + 1) * p2_iters) - cs);
        for (int kt = seg_e - 1; kt >= seg_s; --kt) {
            int k16 = kt * 8 + w * 2 + sub_s;
            const uint8_t* p = B_s + static_cast<uint64_t>(k16) * N + n64 * 64 + g * 8;
            uint32_t s0 = is_odd ? p[2] : p[0];
            uint32_t s1 = is_odd ? p[3] : p[1];
            uint32_t s2 = is_odd ? p[6] : p[4];
            uint32_t s3 = is_odd ? p[7] : p[5];
            SC_MEAS_SYM4(st3, s3);
            SC_MEAS_SYM4(st2, s2);
            SC_MEAS_SYM4(st1, s1);
            SC_MEAS_SYM4(st0, s0);
        }
    }
    #undef SC_MEAS_SYM4

    if (lane == 0) {
        stream_u16[sid] = static_cast<int32_t>(total_u16);
        int64_t ch = (static_cast<int64_t>(total_u16) + 255) / 256;
        stream_chunks[sid] = ch > 0 ? ch : 1;
    }
}

__global__ void nvfp4_sc_rans_encode_kernel(
    const uint8_t* __restrict__ B_s,
    const uint32_t* __restrict__ d_freq,
    const uint32_t* __restrict__ d_cdf,
    const int32_t* __restrict__ stream_u16,
    const int64_t* __restrict__ stream_chunk_off,
    uint16_t* __restrict__ sc_stream,
    uint32_t* __restrict__ init_states,
    int N, int K, int n_grp, int p1_cols, int p2_iters
) {
    int sid = blockIdx.x;
    int n64 = sid >> 2;
    int w = sid & 3;
    int lane = threadIdx.x;
    int g = lane >> 2;
    int part = lane & 3;
    int sub_s = part >> 1;
    int is_odd = part & 1;
    int k_tiles = K >> 7;
    uint32_t lmask_lt = get_lanemask_lt();

    int col = (n_grp == 128) ? (n64 >> 1) : n64;
    int cs = (col - p1_cols) * k_tiles;
    int b_lo = (col < p1_cols) ? 0 : (cs / p2_iters);
    int b_hi = (col < p1_cols) ? 0 : ((cs + k_tiles - 1) / p2_iters);

    uint32_t st0 = 65536u, st1 = 65536u, st2 = 65536u, st3 = 65536u;
    int64_t ptr = stream_chunk_off[sid] * 256 + static_cast<int64_t>(stream_u16[sid]);

    #define SC_ENC_SYM4(st_v, sym_val) do { \
        uint32_t _s = (sym_val); \
        uint32_t _f = d_freq[_s]; \
        uint32_t _c = d_cdf[_s]; \
        bool _em = ((st_v) >= (_f << 22)); \
        uint32_t _m = __ballot_sync(0xffffffffu, _em); \
        ptr -= __popc(_m); \
        if (_em) { \
            uint32_t _rk = static_cast<uint32_t>(__popc(_m & lmask_lt)); \
            sc_stream[ptr + _rk] = static_cast<uint16_t>((st_v) & 0xFFFFu); \
            (st_v) >>= 16; \
        } \
        (st_v) = (((st_v) / _f) << 10) + ((st_v) % _f) + _c; \
    } while (0)

    for (int b = b_lo; b <= b_hi; ++b) {
        int seg_s = (col < p1_cols) ? 0 : (max(cs, b * p2_iters) - cs);
        int seg_e = (col < p1_cols) ? k_tiles : (min(cs + k_tiles, (b + 1) * p2_iters) - cs);
        for (int kt = seg_e - 1; kt >= seg_s; --kt) {
            int k16 = kt * 8 + w * 2 + sub_s;
            const uint8_t* p = B_s + static_cast<uint64_t>(k16) * N + n64 * 64 + g * 8;
            uint32_t s0 = is_odd ? p[2] : p[0];
            uint32_t s1 = is_odd ? p[3] : p[1];
            uint32_t s2 = is_odd ? p[6] : p[4];
            uint32_t s3 = is_odd ? p[7] : p[5];
            SC_ENC_SYM4(st3, s3);
            SC_ENC_SYM4(st2, s2);
            SC_ENC_SYM4(st1, s1);
            SC_ENC_SYM4(st0, s0);
        }
    }
    #undef SC_ENC_SYM4

    uint64_t sb = static_cast<uint64_t>(sid) * 128;
    init_states[sb + lane] = st0;
    init_states[sb + 32 + lane] = st1;
    init_states[sb + 64 + lane] = st2;
    init_states[sb + 96 + lane] = st3;
}

template <bool USE_LDG_A>
__device__ __forceinline__ float compute_n64_tile_rans(
    const uint32_t* __restrict__ A,
    const uint32_t* __restrict__ sA32,
    const uint4* __restrict__ Bq_tm,
    const uint4* __restrict__ sc_stream_u4,
    const int64_t* __restrict__ stream_chunk_off,
    const uint32_t* __restrict__ init_states,
    const uint32_t* __restrict__ sLut,
    int n64, int k_tiles, int n_grp, int p1_cols, int p2_iters,
    float (&s_acc)[4][64], uint4 (&s_sc)[4][8], uint16_t (&s_ring)[4][512]
) {
    int w = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int t = lane & 3;
    uint32_t lmask_lt = get_lanemask_lt();

    int col = (n_grp == 128) ? (n64 >> 1) : n64;
    int cs = (col - p1_cols) * k_tiles;
    int b_lo = (col < p1_cols) ? 0 : (cs / p2_iters);
    int b_hi = (col < p1_cols) ? 0 : ((cs + k_tiles - 1) / p2_iters);

    int sid = n64 * 4 + w;
    const uint4* bq_w = Bq_tm + static_cast<uint64_t>(sid) * k_tiles * 64;
    int64_t c_start = stream_chunk_off[sid];
    int s_chunks = static_cast<int>(stream_chunk_off[sid + 1] - c_start);
    const uint4* es4 = sc_stream_u4 + c_start * 32 + lane;
    uint64_t sb = static_cast<uint64_t>(sid) * 128;
    uint32_t st0 = init_states[sb + lane];
    uint32_t st1 = init_states[sb + 32 + lane];
    uint32_t st2 = init_states[sb + 64 + lane];
    uint32_t st3 = init_states[sb + 96 + lane];

    uint16_t* wbuf = s_ring[w];
    uint4* wbuf4 = reinterpret_cast<uint4*>(wbuf);
    uint32_t* s_sc32 = reinterpret_cast<uint32_t*>(s_sc[w]);

    uint4 c0, c1, c_pref;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(c0.x), "=r"(c0.y), "=r"(c0.z), "=r"(c0.w) : "l"(es4));
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(c1.x), "=r"(c1.y), "=r"(c1.z), "=r"(c1.w) : "l"(es4 + 32));
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(c_pref.x), "=r"(c_pref.y), "=r"(c_pref.z), "=r"(c_pref.w) : "l"(es4 + 64));
    wbuf4[lane] = c0;
    wbuf4[32 + lane] = c1;
    __syncwarp();

    int chunk_idx = 3;
    int pos = 0;
    int refill_at = 256;
    int kt_ord = 0;

    auto load_q = [&](const uint4* p, uint4& q) {
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(q.x), "=r"(q.y), "=r"(q.z), "=r"(q.w) : "l"(p));
    };

    float R_reg = 0.f;
    bool has_R = false;

    for (int b = b_hi; b >= b_lo; --b) {
        int seg_start = (col < p1_cols) ? 0 : (max(cs, b * p2_iters) - cs);
        int seg_end   = (col < p1_cols) ? k_tiles : (min(cs + k_tiles, (b + 1) * p2_iters) - cs);
        float d0[8] = {0}, d1[8] = {0}, d2[8] = {0}, d3[8] = {0};

        uint4 q0, q1;
        load_q(bq_w + kt_ord * 64 + lane, q0);
        load_q(bq_w + kt_ord * 64 + 32 + lane, q1);

        auto run_sub = [&](int kt, int sub, const uint4& qv, uint32_t sc_lo, uint32_t sc_hi) {
            int k16 = (kt << 3) + (w << 1) + sub;
            uint32_t a0, a2;
            if constexpr (USE_LDG_A) {
                a0 = (group == 0) ? __ldg(A + k16 * 8 + t) : 0u;
                a2 = (group == 0) ? __ldg(A + k16 * 8 + 4 + t) : 0u;
            } else {
                a0 = (group == 0) ? sA32[k16 * 8 + t] : 0u;
                a2 = (group == 0) ? sA32[k16 * 8 + 4 + t] : 0u;
            }
            uint32_t q_arr[4] = {qv.x, qv.y, qv.z, qv.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t q = q_arr[j];
                uint32_t q0_ = q << 12, q1_ = q << 8, q2_ = q << 4;
                uint32_t f0 = lop3_or_and(q0_, 0x80008000u, (q0_ >> 6) & 0x01C001C0u);
                uint32_t f1 = lop3_or_and(q1_, 0x80008000u, (q1_ >> 6) & 0x01C001C0u);
                uint32_t f2 = lop3_or_and(q2_, 0x80008000u, (q2_ >> 6) & 0x01C001C0u);
                uint32_t f3 = lop3_or_and(q,   0x80008000u, (q   >> 6) & 0x01C001C0u);
                __nv_bfloat162 s0 = dequant_s_bf162((sc_lo >> (j * 8)) & 0xFFu);
                __nv_bfloat162 s1 = dequant_s_bf162((sc_hi >> (j * 8)) & 0xFFu);
                mma_m16n8k16_bf16_m1(d0[2 * j], d1[2 * j], d2[2 * j], d3[2 * j],
                                     a0, a2, hmul2_u32(f0, s0), hmul2_u32(f1, s0));
                mma_m16n8k16_bf16_m1(d0[2 * j + 1], d1[2 * j + 1], d2[2 * j + 1], d3[2 * j + 1],
                                     a0, a2, hmul2_u32(f2, s1), hmul2_u32(f3, s1));
            }
        };

        for (int kt = seg_start; kt < seg_end; ++kt, ++kt_ord) {
            uint32_t e0 = sLut[st0 & 1023u];
            uint32_t e1 = sLut[st1 & 1023u];
            uint32_t e2 = sLut[st2 & 1023u];
            uint32_t e3 = sLut[st3 & 1023u];
            st0 = ((e0 >> 10) & 1023u) * (st0 >> 10) + (e0 & 1023u);
            st1 = ((e1 >> 10) & 1023u) * (st1 >> 10) + (e1 & 1023u);
            st2 = ((e2 >> 10) & 1023u) * (st2 >> 10) + (e2 & 1023u);
            st3 = ((e3 >> 10) & 1023u) * (st3 >> 10) + (e3 & 1023u);
            bool n0 = (st0 < 65536u), n1 = (st1 < 65536u), n2 = (st2 < 65536u), n3 = (st3 < 65536u);
            uint32_t m0 = __ballot_sync(0xffffffffu, n0);
            uint32_t m1 = __ballot_sync(0xffffffffu, n1);
            uint32_t m2 = __ballot_sync(0xffffffffu, n2);
            uint32_t m3 = __ballot_sync(0xffffffffu, n3);
            int p1 = pos + __popc(m0);
            int p2 = p1 + __popc(m1);
            int p3 = p2 + __popc(m2);
            if (n0) st0 = (st0 << 16) | static_cast<uint32_t>(wbuf[(pos + __popc(m0 & lmask_lt)) & 511]);
            if (n1) st1 = (st1 << 16) | static_cast<uint32_t>(wbuf[(p1  + __popc(m1 & lmask_lt)) & 511]);
            if (n2) st2 = (st2 << 16) | static_cast<uint32_t>(wbuf[(p2  + __popc(m2 & lmask_lt)) & 511]);
            if (n3) st3 = (st3 << 16) | static_cast<uint32_t>(wbuf[(p3  + __popc(m3 & lmask_lt)) & 511]);
            pos = p3 + __popc(m3);

            s_sc32[lane] = (e0 >> 24) | ((e1 >> 16) & 0xFF00u) | ((e2 >> 8) & 0xFF0000u) | (e3 & 0xFF000000u);
            __syncwarp();
            uint4 sc_kt = s_sc[w][group];

            if (pos >= refill_at) {
                int slot = ((refill_at >> 8) - 1) & 1;
                __syncwarp();
                wbuf4[slot * 32 + lane] = c_pref;
                __syncwarp();
                if (chunk_idx < s_chunks) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(c_pref.x), "=r"(c_pref.y), "=r"(c_pref.z), "=r"(c_pref.w)
                        : "l"(es4 + static_cast<uint64_t>(chunk_idx) * 32));
                }
                chunk_idx++;
                refill_at += 256;
            }

            uint4 qn0, qn1;
            if (kt + 1 < seg_end) {
                load_q(bq_w + (kt_ord + 1) * 64 + lane, qn0);
            }
            run_sub(kt, 0, q0, sc_kt.x, sc_kt.y);
            if (kt + 1 < seg_end) {
                load_q(bq_w + (kt_ord + 1) * 64 + 32 + lane, qn1);
            }
            run_sub(kt, 1, q1, sc_kt.z, sc_kt.w);
            q0 = qn0;
            q1 = qn1;
        }

        if (group == 0) {
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                s_acc[w][m * 8 + t * 2] = d0[m];
                s_acc[w][m * 8 + t * 2 + 1] = d1[m];
            }
        }
        __syncthreads();
        if (threadIdx.x < 64) {
            int c = threadIdx.x;
            float P = s_acc[0][c] + (s_acc[1][c] + (s_acc[3][c] + s_acc[2][c]));
            R_reg = has_R ? (P + R_reg) : P;
            has_R = true;
        }
        __syncthreads();
    }
    return R_reg;
}

template <bool USE_LDG_A>
__global__ void nvfp4_rans_linear_m1_kernel(
    const uint32_t* __restrict__ A,
    const uint4* __restrict__ Bq_tm,
    const uint4* __restrict__ sc_stream_u4,
    const int64_t* __restrict__ stream_chunk_off,
    const uint32_t* __restrict__ init_states,
    const uint32_t* __restrict__ d_lut,
    const float* __restrict__ g_scale,
    __nv_bfloat16* __restrict__ C,
    int N, int K, int n_grp, int p1_cols, int p2_iters
) {
    extern __shared__ uint32_t sA32[];
    if constexpr (!USE_LDG_A) {
        const uint4* A4 = reinterpret_cast<const uint4*>(A);
        uint4* sA4 = reinterpret_cast<uint4*>(sA32);
        for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) {
            sA4[i] = A4[i];
        }
    }
    __shared__ uint32_t sLut[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        sLut[i] = d_lut[i];
    }
    __shared__ float s_acc[4][64];
    __shared__ uint4 s_sc[4][8];
    __shared__ uint16_t s_ring[4][512];
    __syncthreads();

    int num_n64 = N >> 6;
    int k_tiles = K >> 7;
    float gs = *g_scale;

    for (int n64 = blockIdx.x; n64 < num_n64; n64 += gridDim.x) {
        float R_reg = compute_n64_tile_rans<USE_LDG_A>(
            A, sA32, Bq_tm, sc_stream_u4, stream_chunk_off, init_states, sLut,
            n64, k_tiles, n_grp, p1_cols, p2_iters, s_acc, s_sc, s_ring
        );
        if (threadIdx.x < 64) {
            C[n64 * 64 + threadIdx.x] = __float2bfloat16_rn(R_reg * gs);
        }
    }
}

template <bool OUT_FP8>
__global__ void nvfp4_rans_gate_up_silu_m1_kernel(
    const uint32_t* __restrict__ A,
    const uint4* __restrict__ Bq_tm,
    const uint4* __restrict__ sc_stream_u4,
    const int64_t* __restrict__ stream_chunk_off,
    const uint32_t* __restrict__ init_states,
    const uint32_t* __restrict__ d_lut,
    const float* __restrict__ g_scale,
    const float* __restrict__ out_scale_a,
    void* __restrict__ C_raw
) {
    constexpr int K = 6656;
    constexpr int k_tiles = 52;
    constexpr int half_n64 = 312;

    extern __shared__ uint32_t sA32[];
    const uint4* A4 = reinterpret_cast<const uint4*>(A);
    uint4* sA4 = reinterpret_cast<uint4*>(sA32);
    for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) {
        sA4[i] = A4[i];
    }
    __shared__ uint32_t sLut[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        sLut[i] = d_lut[i];
    }
    __shared__ float s_acc[4][64];
    __shared__ uint4 s_sc[4][8];
    __shared__ uint16_t s_ring[4][512];
    __syncthreads();

    float gs = *g_scale;
    float inv_sa = 1.0f;
    if constexpr (OUT_FP8) {
        inv_sa = __fdividef(1.0f, *out_scale_a);
    }
    for (int n_blk = blockIdx.x; n_blk < half_n64; n_blk += gridDim.x) {
        float g_val = compute_n64_tile_rans<false>(
            A, sA32, Bq_tm, sc_stream_u4, stream_chunk_off, init_states, sLut,
            n_blk, k_tiles, 64, 576, 26, s_acc, s_sc, s_ring
        );
        float u_val = compute_n64_tile_rans<false>(
            A, sA32, Bq_tm, sc_stream_u4, stream_chunk_off, init_states, sLut,
            half_n64 + n_blk, k_tiles, 64, 576, 26, s_acc, s_sc, s_ring
        );
        if (threadIdx.x < 64) {
            float g = __bfloat162float(__float2bfloat16_rn(g_val * gs));
            float u = __bfloat162float(__float2bfloat16_rn(u_val * gs));
            float prod = __fdividef(g, 1.0f + expf(-g)) * u;
            if constexpr (OUT_FP8) {
                float v = fminf(fmaxf(prod * inv_sa, -448.0f), 448.0f);
                __nv_fp8_e4m3 q(v);
                reinterpret_cast<uint8_t*>(C_raw)[n_blk * 64 + threadIdx.x] =
                    *reinterpret_cast<uint8_t*>(&q);
            } else {
                reinterpret_cast<__nv_bfloat16*>(C_raw)[n_blk * 64 + threadIdx.x] =
                    __float2bfloat16_rn(prod);
            }
        }
    }
}

// Zero-tail-wave 1248-warp (96 CTAs x 13 warps = 26 warps/SM on all 48 SMs)
// paired-(16 gate + 16 up) 4-way ILP rANS gate_up_silu M=1:
//   - Exact 1.000 pair/warp across all 1248 warps (0 SM tail wave, 0 warp tail wave)
//   - Zero cross-warp __syncthreads() in K-loop, zero s_acc/s_sc SMEM (__shfl_sync scales)
//   - Preserves exact Marlin w=0..3 & rev=1 split-K FP32 reduction order (mism = 0)
__device__ __forceinline__ uint32_t pick_u4_comp(const uint4& v, int jq) {
    return (jq == 0) ? v.x : ((jq == 1) ? v.y : ((jq == 2) ? v.z : v.w));
}

__global__ void nvfp4_sc_rans_gu_w16_measure_bq_kernel(
    const uint4* __restrict__ B_q,
    const uint8_t* __restrict__ B_s,
    const uint32_t* __restrict__ d_freq,
    const uint32_t* __restrict__ d_cdf,
    uint4* __restrict__ Bq_tm,
    int32_t* __restrict__ stream_u16,
    int64_t* __restrict__ stream_chunks
) {
    constexpr int N = 39936;
    constexpr int row_stride_int4 = N >> 1;
    int p16 = blockIdx.x; // 0 .. 1247
    int n_blk = p16 >> 2; // 0 .. 311
    int j_q = p16 & 3;    // 0 .. 3
    int gate_n64 = n_blk;
    int up_n64 = 312 + n_blk;
    bool split_up = (n_blk >= 264);
    int j_off = ((j_q & 2) << 1) | (j_q & 1);

    int lane = threadIdx.x;
    int group = lane >> 2;
    int t = lane & 3;

    uint4* q_dst = Bq_tm + static_cast<uint64_t>(p16) * 104 * 64;
    for (int wp = 0; wp < 4; ++wp) {
        int w = 3 - wp;
        for (int kp2 = 0; kp2 < 26; ++kp2) {
            int ord = wp * 26 + kp2;
            int kp2_u = (!split_up) ? kp2 : ((kp2 < 13) ? (kp2 + 13) : (kp2 - 13));
            uint32_t qg[4], qu[4];
            #pragma unroll
            for (int s4 = 0; s4 < 4; ++s4) {
                int k16_g = (kp2   * 2 + (s4 >> 1)) * 8 + w * 2 + (s4 & 1);
                int k16_u = (kp2_u * 2 + (s4 >> 1)) * 8 + w * 2 + (s4 & 1);
                uint4 vg = B_q[static_cast<uint64_t>(k16_g) * row_stride_int4 + gate_n64 * 32 + lane];
                uint4 vu = B_q[static_cast<uint64_t>(k16_u) * row_stride_int4 + up_n64   * 32 + lane];
                qg[s4] = pick_u4_comp(vg, j_q);
                qu[s4] = pick_u4_comp(vu, j_q);
            }
            q_dst[ord * 64 + lane]      = make_uint4(qg[0], qg[1], qg[2], qg[3]);
            q_dst[ord * 64 + 32 + lane] = make_uint4(qu[0], qu[1], qu[2], qu[3]);
        }
    }

    uint32_t st0 = 65536u, st1 = 65536u, st2 = 65536u, st3 = 65536u;
    uint32_t total_u16 = 0u;
    #define GU16_MEAS_SYM(st_v, sym_val) do { \
        uint32_t _s = (sym_val); \
        uint32_t _f = d_freq[_s]; \
        uint32_t _c = d_cdf[_s]; \
        bool _em = ((st_v) >= (_f << 22)); \
        uint32_t _m = __ballot_sync(0xffffffffu, _em); \
        total_u16 += static_cast<uint32_t>(__popc(_m)); \
        if (_em) (st_v) >>= 16; \
        (st_v) = (((st_v) / _f) << 10) + ((st_v) % _f) + _c; \
    } while (0)

    for (int wp = 3; wp >= 0; --wp) {
        int w = 3 - wp;
        for (int kp2 = 25; kp2 >= 0; --kp2) {
            int kp2_u = (!split_up) ? kp2 : ((kp2 < 13) ? (kp2 + 13) : (kp2 - 13));
            int k16_g = (kp2   * 2 + (t >> 1)) * 8 + w * 2 + (t & 1);
            int k16_u = (kp2_u * 2 + (t >> 1)) * 8 + w * 2 + (t & 1);
            const uint8_t* pg = B_s + static_cast<uint64_t>(k16_g) * N + gate_n64 * 64 + group * 8 + j_off;
            const uint8_t* pu = B_s + static_cast<uint64_t>(k16_u) * N + up_n64   * 64 + group * 8 + j_off;
            uint32_t s0 = pg[0], s1 = pg[2], s2 = pu[0], s3 = pu[2];
            GU16_MEAS_SYM(st3, s3);
            GU16_MEAS_SYM(st2, s2);
            GU16_MEAS_SYM(st1, s1);
            GU16_MEAS_SYM(st0, s0);
        }
    }
    #undef GU16_MEAS_SYM

    if (lane == 0) {
        stream_u16[p16] = static_cast<int32_t>(total_u16);
        int64_t ch = (static_cast<int64_t>(total_u16) + 255) / 256;
        stream_chunks[p16] = ch > 0 ? ch : 1;
    }
}

__global__ void nvfp4_sc_rans_gu_w16_encode_kernel(
    const uint8_t* __restrict__ B_s,
    const uint32_t* __restrict__ d_freq,
    const uint32_t* __restrict__ d_cdf,
    const int32_t* __restrict__ stream_u16,
    const int64_t* __restrict__ stream_chunk_off,
    uint16_t* __restrict__ sc_stream,
    uint32_t* __restrict__ init_states
) {
    constexpr int N = 39936;
    int p16 = blockIdx.x;
    int n_blk = p16 >> 2;
    int j_q = p16 & 3;
    int gate_n64 = n_blk;
    int up_n64 = 312 + n_blk;
    bool split_up = (n_blk >= 264);
    int j_off = ((j_q & 2) << 1) | (j_q & 1);

    int lane = threadIdx.x;
    int group = lane >> 2;
    int t = lane & 3;
    uint32_t lmask_lt = get_lanemask_lt();

    uint32_t st0 = 65536u, st1 = 65536u, st2 = 65536u, st3 = 65536u;
    int64_t ptr = stream_chunk_off[p16] * 256 + static_cast<int64_t>(stream_u16[p16]);

    #define GU16_ENC_SYM(st_v, sym_val) do { \
        uint32_t _s = (sym_val); \
        uint32_t _f = d_freq[_s]; \
        uint32_t _c = d_cdf[_s]; \
        bool _em = ((st_v) >= (_f << 22)); \
        uint32_t _m = __ballot_sync(0xffffffffu, _em); \
        ptr -= __popc(_m); \
        if (_em) { \
            uint32_t _rk = static_cast<uint32_t>(__popc(_m & lmask_lt)); \
            sc_stream[ptr + _rk] = static_cast<uint16_t>((st_v) & 0xFFFFu); \
            (st_v) >>= 16; \
        } \
        (st_v) = (((st_v) / _f) << 10) + ((st_v) % _f) + _c; \
    } while (0)

    for (int wp = 3; wp >= 0; --wp) {
        int w = 3 - wp;
        for (int kp2 = 25; kp2 >= 0; --kp2) {
            int kp2_u = (!split_up) ? kp2 : ((kp2 < 13) ? (kp2 + 13) : (kp2 - 13));
            int k16_g = (kp2   * 2 + (t >> 1)) * 8 + w * 2 + (t & 1);
            int k16_u = (kp2_u * 2 + (t >> 1)) * 8 + w * 2 + (t & 1);
            const uint8_t* pg = B_s + static_cast<uint64_t>(k16_g) * N + gate_n64 * 64 + group * 8 + j_off;
            const uint8_t* pu = B_s + static_cast<uint64_t>(k16_u) * N + up_n64   * 64 + group * 8 + j_off;
            uint32_t s0 = pg[0], s1 = pg[2], s2 = pu[0], s3 = pu[2];
            GU16_ENC_SYM(st3, s3);
            GU16_ENC_SYM(st2, s2);
            GU16_ENC_SYM(st1, s1);
            GU16_ENC_SYM(st0, s0);
        }
    }
    #undef GU16_ENC_SYM

    uint64_t sb = static_cast<uint64_t>(p16) * 128;
    init_states[sb + lane] = st0;
    init_states[sb + 32 + lane] = st1;
    init_states[sb + 64 + lane] = st2;
    init_states[sb + 96 + lane] = st3;
}

template <bool OUT_FP8>
__global__ void __launch_bounds__(416, 2) nvfp4_rans_gate_up_silu_w16_m1_kernel(
    const uint32_t* __restrict__ A,
    const uint4* __restrict__ Bq_tm,
    const uint4* __restrict__ sc_stream_u4,
    const int64_t* __restrict__ stream_chunk_off,
    const uint32_t* __restrict__ init_states,
    const uint32_t* __restrict__ d_lut,
    const float* __restrict__ g_scale,
    const float* __restrict__ out_scale_a,
    void* __restrict__ C_raw
) {
    constexpr int K = 6656;
    extern __shared__ uint8_t smem_gu16[];
    uint32_t* sA32 = reinterpret_cast<uint32_t*>(smem_gu16);
    uint32_t* sLut = reinterpret_cast<uint32_t*>(smem_gu16 + K * 2);
    uint16_t* sRing = reinterpret_cast<uint16_t*>(smem_gu16 + K * 2 + 4096);
    float* sSave = reinterpret_cast<float*>(smem_gu16 + K * 2 + 4096 + 13 * 1024);

    const uint4* A4 = reinterpret_cast<const uint4*>(A);
    uint4* sA4 = reinterpret_cast<uint4*>(sA32);
    for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) {
        sA4[i] = A4[i];
    }
    const uint4* lut4 = reinterpret_cast<const uint4*>(d_lut);
    uint4* sLut4 = reinterpret_cast<uint4*>(sLut);
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        sLut4[i] = lut4[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x >> 5; // 0 .. 12
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int t = lane & 3;
    uint32_t lmask_lt = get_lanemask_lt();

    int p16 = blockIdx.x * 13 + warp_id; // 0 .. 1247
    int n_blk = p16 >> 2;
    bool split_up = (n_blk >= 264);

    const uint4* bq_w = Bq_tm + static_cast<uint64_t>(p16) * 104 * 64;
    int64_t c_start = stream_chunk_off[p16];
    int s_chunks = static_cast<int>(stream_chunk_off[p16 + 1] - c_start);
    const uint4* es4 = sc_stream_u4 + c_start * 32 + lane;
    uint64_t sb = static_cast<uint64_t>(p16) * 128;
    uint32_t st0 = init_states[sb + lane];
    uint32_t st1 = init_states[sb + 32 + lane];
    uint32_t st2 = init_states[sb + 64 + lane];
    uint32_t st3 = init_states[sb + 96 + lane];

    uint16_t* wbuf = sRing + warp_id * 512;
    uint4* wbuf4 = reinterpret_cast<uint4*>(wbuf);

    auto load_q = [&](const uint4* p, uint4& q) {
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(q.x), "=r"(q.y), "=r"(q.z), "=r"(q.w) : "l"(p));
    };

    uint4 c0, c1, c_pref;
    load_q(es4, c0);
    load_q(es4 + 32, c1);
    load_q(es4 + 64, c_pref);

    uint4 qg_cur, qu_cur;
    load_q(bq_w + lane, qg_cur);
    load_q(bq_w + 32 + lane, qu_cur);

    wbuf4[lane] = c0;
    wbuf4[32 + lane] = c1;
    __syncwarp();

    int chunk_idx = 3;
    int pos = 0;
    int refill_at = 256;
    float* my_save = sSave + (warp_id * 4 + t) * 12;

    #pragma unroll 1
    for (int wp = 0; wp < 4; ++wp) {
        int w = 3 - wp;
        float gd0[2] = {0.f, 0.f}, gd1[2] = {0.f, 0.f}, gd2[2] = {0.f, 0.f}, gd3[2] = {0.f, 0.f};
        float ud0[2] = {0.f, 0.f}, ud1[2] = {0.f, 0.f}, ud2[2] = {0.f, 0.f}, ud3[2] = {0.f, 0.f};

        #pragma unroll 1
        for (int kp2 = 0; kp2 < 26; ++kp2) {
            int ord = wp * 26 + kp2;
            uint4 qg_nxt, qu_nxt;
            if (ord + 1 < 104) {
                load_q(bq_w + (ord + 1) * 64 + lane, qg_nxt);
                load_q(bq_w + (ord + 1) * 64 + 32 + lane, qu_nxt);
            }

            uint32_t e0 = sLut[st0 & 1023u];
            uint32_t e1 = sLut[st1 & 1023u];
            uint32_t e2 = sLut[st2 & 1023u];
            uint32_t e3 = sLut[st3 & 1023u];
            st0 = ((e0 >> 10) & 1023u) * (st0 >> 10) + (e0 & 1023u);
            st1 = ((e1 >> 10) & 1023u) * (st1 >> 10) + (e1 & 1023u);
            st2 = ((e2 >> 10) & 1023u) * (st2 >> 10) + (e2 & 1023u);
            st3 = ((e3 >> 10) & 1023u) * (st3 >> 10) + (e3 & 1023u);
            bool n0 = (st0 < 65536u), n1 = (st1 < 65536u), n2 = (st2 < 65536u), n3 = (st3 < 65536u);
            uint32_t m0 = __ballot_sync(0xffffffffu, n0);
            uint32_t m1 = __ballot_sync(0xffffffffu, n1);
            uint32_t m2 = __ballot_sync(0xffffffffu, n2);
            uint32_t m3 = __ballot_sync(0xffffffffu, n3);
            int p1 = pos + __popc(m0);
            int p2 = p1 + __popc(m1);
            int p3 = p2 + __popc(m2);
            if (n0) st0 = (st0 << 16) | static_cast<uint32_t>(wbuf[(pos + __popc(m0 & lmask_lt)) & 511]);
            if (n1) st1 = (st1 << 16) | static_cast<uint32_t>(wbuf[(p1  + __popc(m1 & lmask_lt)) & 511]);
            if (n2) st2 = (st2 << 16) | static_cast<uint32_t>(wbuf[(p2  + __popc(m2 & lmask_lt)) & 511]);
            if (n3) st3 = (st3 << 16) | static_cast<uint32_t>(wbuf[(p3  + __popc(m3 & lmask_lt)) & 511]);
            pos = p3 + __popc(m3);

            uint32_t my_sc = (e0 >> 24) | ((e1 >> 16) & 0xFF00u) | ((e2 >> 8) & 0xFF0000u) | (e3 & 0xFF000000u);

            if (pos >= refill_at) {
                int slot = ((refill_at >> 8) - 1) & 1;
                __syncwarp();
                wbuf4[slot * 32 + lane] = c_pref;
                __syncwarp();
                if (chunk_idx < s_chunks) {
                    load_q(es4 + static_cast<uint64_t>(chunk_idx) * 32, c_pref);
                }
                chunk_idx++;
                refill_at += 256;
            }

            int kp2_u = (!split_up) ? kp2 : ((kp2 < 13) ? (kp2 + 13) : (kp2 - 13));
            uint32_t qg_arr[4] = {qg_cur.x, qg_cur.y, qg_cur.z, qg_cur.w};
            uint32_t qu_arr[4] = {qu_cur.x, qu_cur.y, qu_cur.z, qu_cur.w};

            #pragma unroll
            for (int s4 = 0; s4 < 4; ++s4) {
                uint32_t sc4 = __shfl_sync(0xffffffffu, my_sc, (lane & 28) | s4);
                int k16_g = (kp2 * 2 + (s4 >> 1)) * 8 + w * 2 + (s4 & 1);
                uint32_t a0_g = (group == 0) ? sA32[k16_g * 8 + t] : 0u;
                uint32_t a2_g = (group == 0) ? sA32[k16_g * 8 + 4 + t] : 0u;
                {
                    uint32_t q = qg_arr[s4];
                    uint32_t q0_ = q << 12, q1_ = q << 8, q2_ = q << 4;
                    uint32_t f0 = lop3_or_and(q0_, 0x80008000u, (q0_ >> 6) & 0x01C001C0u);
                    uint32_t f1 = lop3_or_and(q1_, 0x80008000u, (q1_ >> 6) & 0x01C001C0u);
                    uint32_t f2 = lop3_or_and(q2_, 0x80008000u, (q2_ >> 6) & 0x01C001C0u);
                    uint32_t f3 = lop3_or_and(q,   0x80008000u, (q   >> 6) & 0x01C001C0u);
                    __nv_bfloat162 sg0 = dequant_s_bf162(sc4 & 0xFFu);
                    __nv_bfloat162 sg1 = dequant_s_bf162((sc4 >> 8) & 0xFFu);
                    mma_m16n8k16_bf16_m1(gd0[0], gd1[0], gd2[0], gd3[0],
                                         a0_g, a2_g, hmul2_u32(f0, sg0), hmul2_u32(f1, sg0));
                    mma_m16n8k16_bf16_m1(gd0[1], gd1[1], gd2[1], gd3[1],
                                         a0_g, a2_g, hmul2_u32(f2, sg1), hmul2_u32(f3, sg1));
                }
                int k16_u = (kp2_u * 2 + (s4 >> 1)) * 8 + w * 2 + (s4 & 1);
                uint32_t a0_u = (!split_up) ? a0_g : ((group == 0) ? sA32[k16_u * 8 + t] : 0u);
                uint32_t a2_u = (!split_up) ? a2_g : ((group == 0) ? sA32[k16_u * 8 + 4 + t] : 0u);
                {
                    uint32_t q = qu_arr[s4];
                    uint32_t q0_ = q << 12, q1_ = q << 8, q2_ = q << 4;
                    uint32_t f0 = lop3_or_and(q0_, 0x80008000u, (q0_ >> 6) & 0x01C001C0u);
                    uint32_t f1 = lop3_or_and(q1_, 0x80008000u, (q1_ >> 6) & 0x01C001C0u);
                    uint32_t f2 = lop3_or_and(q2_, 0x80008000u, (q2_ >> 6) & 0x01C001C0u);
                    uint32_t f3 = lop3_or_and(q,   0x80008000u, (q   >> 6) & 0x01C001C0u);
                    __nv_bfloat162 su0 = dequant_s_bf162((sc4 >> 16) & 0xFFu);
                    __nv_bfloat162 su1 = dequant_s_bf162(sc4 >> 24);
                    mma_m16n8k16_bf16_m1(ud0[0], ud1[0], ud2[0], ud3[0],
                                         a0_u, a2_u, hmul2_u32(f0, su0), hmul2_u32(f1, su0));
                    mma_m16n8k16_bf16_m1(ud0[1], ud1[1], ud2[1], ud3[1],
                                         a0_u, a2_u, hmul2_u32(f2, su1), hmul2_u32(f3, su1));
                }
            }

            qg_cur = qg_nxt;
            qu_cur = qu_nxt;

            if (split_up && kp2 == 12) {
                if (group == 0) {
                    float puA[4] = {ud0[0], ud1[0], ud0[1], ud1[1]};
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        float prev = (w == 3) ? 0.f : my_save[4 + m];
                        my_save[4 + m] = (w == 3) ? puA[m] : ((w == 2) ? (prev + puA[m]) : (puA[m] + prev));
                    }
                }
                ud0[0] = 0.f; ud1[0] = 0.f; ud2[0] = 0.f; ud3[0] = 0.f;
                ud0[1] = 0.f; ud1[1] = 0.f; ud2[1] = 0.f; ud3[1] = 0.f;
            }
        }

        if (group == 0) {
            float pg[4] = {gd0[0], gd1[0], gd0[1], gd1[1]};
            float pu[4] = {ud0[0], ud1[0], ud0[1], ud1[1]};
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                float prev_g = (w == 3) ? 0.f : my_save[0 + m];
                float prev_u = (w == 3) ? 0.f : my_save[8 + m];
                my_save[0 + m] = (w == 3) ? pg[m] : ((w == 2) ? (prev_g + pg[m]) : (pg[m] + prev_g));
                my_save[8 + m] = (w == 3) ? pu[m] : ((w == 2) ? (prev_u + pu[m]) : (pu[m] + prev_u));
            }
        }
    }

    if (group == 0) {
        float gs = *g_scale;
        float inv_sa = 1.0f;
        if constexpr (OUT_FP8) {
            inv_sa = __fdividef(1.0f, *out_scale_a);
        }
        int base_col = p16 * 16;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            int col = base_col + h * 8 + t * 2;
            float gr0 = my_save[h * 2 + 0];
            float gr1 = my_save[h * 2 + 1];
            float ur0 = split_up ? (my_save[8 + h * 2 + 0] + my_save[4 + h * 2 + 0]) : my_save[8 + h * 2 + 0];
            float ur1 = split_up ? (my_save[8 + h * 2 + 1] + my_save[4 + h * 2 + 1]) : my_save[8 + h * 2 + 1];
            float g0 = __bfloat162float(__float2bfloat16_rn(gr0 * gs));
            float g1 = __bfloat162float(__float2bfloat16_rn(gr1 * gs));
            float u0 = __bfloat162float(__float2bfloat16_rn(ur0 * gs));
            float u1 = __bfloat162float(__float2bfloat16_rn(ur1 * gs));
            float pr0 = __fdividef(g0, 1.0f + expf(-g0)) * u0;
            float pr1 = __fdividef(g1, 1.0f + expf(-g1)) * u1;
            if constexpr (OUT_FP8) {
                __nv_fp8_e4m3 q0(fminf(fmaxf(pr0 * inv_sa, -448.0f), 448.0f));
                __nv_fp8_e4m3 q1(fminf(fmaxf(pr1 * inv_sa, -448.0f), 448.0f));
                uint16_t pair = static_cast<uint16_t>(*reinterpret_cast<uint8_t*>(&q0))
                              | (static_cast<uint16_t>(*reinterpret_cast<uint8_t*>(&q1)) << 8);
                *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(C_raw) + col) = pair;
            } else {
                __nv_bfloat16 b0 = __float2bfloat16_rn(pr0);
                __nv_bfloat16 b1 = __float2bfloat16_rn(pr1);
                uint32_t pair = static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&b0))
                              | (static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&b1)) << 16);
                *reinterpret_cast<uint32_t*>(reinterpret_cast<__nv_bfloat16*>(C_raw) + col) = pair;
            }
        }
    }
}

__global__ void nvfp4_rans_lmhead_m1_fused_argmax_kernel(
    const uint32_t* __restrict__ A,
    const uint4* __restrict__ Bq_tm,
    const uint4* __restrict__ sc_stream_u4,
    const int64_t* __restrict__ stream_chunk_off,
    const uint32_t* __restrict__ init_states,
    const uint32_t* __restrict__ d_lut,
    const float* __restrict__ g_scale,
    float output_multiplier,
    float softcap,
    float* __restrict__ blk_max_val,
    int32_t* __restrict__ blk_max_idx
) {
    constexpr int K = 6656;
    constexpr int k_tiles = 52;
    constexpr int num_n64 = 3157;

    extern __shared__ uint32_t sA32[];
    const uint4* A4 = reinterpret_cast<const uint4*>(A);
    uint4* sA4 = reinterpret_cast<uint4*>(sA32);
    for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) {
        sA4[i] = A4[i];
    }
    __shared__ uint32_t sLut[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        sLut[i] = d_lut[i];
    }
    __shared__ float s_acc[4][64];
    __shared__ uint4 s_sc[4][8];
    __shared__ uint16_t s_ring[4][512];
    __syncthreads();

    int w = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    float gs = *g_scale;
    float my_best_val = -3.402823466e+38f;
    int32_t my_best_idx = 0;

    for (int n64 = blockIdx.x; n64 < num_n64; n64 += gridDim.x) {
        float R_reg = compute_n64_tile_rans<false>(
            A, sA32, Bq_tm, sc_stream_u4, stream_chunk_off, init_states, sLut,
            n64, k_tiles, 64, 3120, 41, s_acc, s_sc, s_ring
        );
        if (threadIdx.x < 64) {
            float v = __bfloat162float(__float2bfloat16_rn(R_reg * gs));
            v = __bfloat162float(__float2bfloat16_rn(v * output_multiplier));
            if (softcap > 0.f) {
                float d = __bfloat162float(__float2bfloat16_rn(v / softcap));
                float th = __bfloat162float(__float2bfloat16_rn(tanhf(d)));
                v = __bfloat162float(__float2bfloat16_rn(softcap * th));
            }
            int32_t idx = n64 * 64 + threadIdx.x;
            if (v > my_best_val || (v == my_best_val && idx < my_best_idx)) {
                my_best_val = v;
                my_best_idx = idx;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, my_best_val, offset);
        int32_t oi = __shfl_down_sync(0xffffffffu, my_best_idx, offset);
        if (ov > my_best_val || (ov == my_best_val && oi < my_best_idx)) {
            my_best_val = ov;
            my_best_idx = oi;
        }
    }
    __shared__ float s_bval[4];
    __shared__ int32_t s_bidx[4];
    if (lane == 0) {
        s_bval[w] = my_best_val;
        s_bidx[w] = my_best_idx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float bv = s_bval[0];
        int32_t bi = s_bidx[0];
        if (s_bval[1] > bv || (s_bval[1] == bv && s_bidx[1] < bi)) {
            bv = s_bval[1];
            bi = s_bidx[1];
        }
        blk_max_val[blockIdx.x] = bv;
        blk_max_idx[blockIdx.x] = bi;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> repack_nvfp4_rans(
    const at::Tensor& B_q,
    const at::Tensor& B_s,
    int64_t N_in,
    int64_t K_in,
    int64_t n_grp,
    int64_t p1_cols,
    int64_t p2_iters
) {
    int N = static_cast<int>(N_in);
    int K = static_cast<int>(K_in);
    TORCH_CHECK(B_q.is_cuda() && B_s.is_cuda());
    TORCH_CHECK(N % 64 == 0 && K % 128 == 0);
    int num_n64 = N >> 6;
    int num_streams = num_n64 * 4;
    int k_tiles = K >> 7;
    auto dev_opts = B_q.options();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 1. Histogram of 256 scale byte values on GPU.
    auto g_hist = at::zeros({256}, dev_opts.dtype(at::kLong));
    int64_t num_u32 = (static_cast<int64_t>(K >> 4) * N) >> 2;
    int hist_blocks = static_cast<int>(std::min<int64_t>((num_u32 + 255) / 256, 2048));
    nvfp4_sc_hist_kernel<<<hist_blocks, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(B_s.data_ptr()),
        num_u32,
        reinterpret_cast<unsigned long long*>(g_hist.data_ptr<int64_t>())
    );
    auto h_hist_t = g_hist.to(at::kCPU);
    const int64_t* h_hist = h_hist_t.data_ptr<int64_t>();

    // 2. Quantize frequencies to 10-bit M_PROB=1024 (4 KB SMEM LUT).
    constexpr int M_PROB = 1024;
    int64_t total = 0;
    for (int i = 0; i < 256; ++i) total += h_hist[i];
    std::vector<uint32_t> freq(256, 0u);
    std::vector<double> exact(256, 0.0);
    std::vector<double> rem(256, -1.0);
    int sum_f = 0;
    for (int i = 0; i < 256; ++i) {
        if (h_hist[i] > 0) {
            double e = static_cast<double>(h_hist[i]) * static_cast<double>(M_PROB) / static_cast<double>(total);
            exact[i] = e;
            int f = static_cast<int>(std::floor(e));
            if (f < 1) f = 1;
            if (f > M_PROB - 1) f = M_PROB - 1;
            freq[i] = static_cast<uint32_t>(f);
            rem[i] = e - static_cast<double>(f);
            sum_f += f;
        }
    }
    while (sum_f < M_PROB) {
        int best = -1;
        double best_r = -1e9;
        for (int i = 0; i < 256; ++i) {
            if (h_hist[i] > 0 && freq[i] < static_cast<uint32_t>(M_PROB - 1) && rem[i] > best_r) {
                best_r = rem[i];
                best = i;
            }
        }
        if (best < 0) {
            for (int i = 0; i < 256; ++i) {
                if (freq[i] == 0u) { best = i; break; }
            }
        }
        freq[best] += 1u;
        rem[best] = -1e9;
        sum_f += 1;
    }
    while (sum_f > M_PROB) {
        int best = -1;
        double worst_excess = -1e9;
        for (int i = 0; i < 256; ++i) {
            if (freq[i] > 1u) {
                double ex = static_cast<double>(freq[i]) - exact[i];
                if (ex > worst_excess) {
                    worst_excess = ex;
                    best = i;
                }
            }
        }
        freq[best] -= 1u;
        sum_f -= 1;
    }

    std::vector<uint32_t> cdf(256, 0u);
    std::vector<uint32_t> lut(M_PROB, 0u);
    uint32_t acc = 0u;
    for (int s = 0; s < 256; ++s) {
        cdf[s] = acc;
        for (uint32_t j = 0; j < freq[s]; ++j) {
            lut[acc + j] = (static_cast<uint32_t>(s) << 24) | (freq[s] << 10) | j;
        }
        acc += freq[s];
    }

    auto d_freq = at::from_blob(freq.data(), {256}, at::kInt).to(B_q.device());
    auto d_cdf  = at::from_blob(cdf.data(), {256}, at::kInt).to(B_q.device());
    auto d_lut  = at::from_blob(lut.data(), {M_PROB}, at::kInt).to(B_q.device());

    bool use_gu_w16 = (n_grp == 64 && N == 39936 && K == 6656 && p1_cols == 576 && p2_iters == 26);
    int bq_steps = k_tiles;
    if (use_gu_w16) {
        num_streams = 1248;
        bq_steps = 104;
    }

    // 3. Repack Bq_tm in exact warp-sequential segment order and measure scale streams.
    auto Bq_tm = at::empty({static_cast<int64_t>(num_streams) * bq_steps * 64 * 4}, dev_opts.dtype(at::kInt));
    auto stream_u16 = at::empty({num_streams}, dev_opts.dtype(at::kInt));
    auto stream_chunks = at::empty({num_streams}, dev_opts.dtype(at::kLong));
    if (use_gu_w16) {
        nvfp4_sc_rans_gu_w16_measure_bq_kernel<<<num_streams, 32, 0, stream>>>(
            reinterpret_cast<const uint4*>(B_q.data_ptr()),
            reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
            reinterpret_cast<const uint32_t*>(d_freq.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(d_cdf.data_ptr<int32_t>()),
            reinterpret_cast<uint4*>(Bq_tm.data_ptr()),
            stream_u16.data_ptr<int32_t>(),
            stream_chunks.data_ptr<int64_t>()
        );
    } else {
        nvfp4_sc_rans_measure_bq_kernel<<<num_streams, 32, 0, stream>>>(
            reinterpret_cast<const uint4*>(B_q.data_ptr()),
            reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
            reinterpret_cast<const uint32_t*>(d_freq.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(d_cdf.data_ptr<int32_t>()),
            reinterpret_cast<uint4*>(Bq_tm.data_ptr()),
            stream_u16.data_ptr<int32_t>(),
            stream_chunks.data_ptr<int64_t>(),
            N, K, static_cast<int>(n_grp), static_cast<int>(p1_cols), static_cast<int>(p2_iters)
        );
    }

    // 4. Prefix-sum 512B chunk offsets (+ 4-chunk safety tail for lookahead).
    auto stream_chunk_off = at::zeros({num_streams + 1}, dev_opts.dtype(at::kLong));
    stream_chunk_off.narrow(0, 1, num_streams).copy_(at::cumsum(stream_chunks, 0));
    int64_t total_chunks = stream_chunk_off[num_streams].item<int64_t>();

    auto sc_stream = at::zeros({(total_chunks + 4) * 256}, dev_opts.dtype(at::kShort));
    auto init_states = at::empty({num_streams, 128}, dev_opts.dtype(at::kInt));

    // 5. Encode exact forward-consumable 4-way ILP 32-lane interleaved NVFP4 scale rANS streams.
    if (use_gu_w16) {
        nvfp4_sc_rans_gu_w16_encode_kernel<<<num_streams, 32, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
            reinterpret_cast<const uint32_t*>(d_freq.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(d_cdf.data_ptr<int32_t>()),
            stream_u16.data_ptr<int32_t>(),
            stream_chunk_off.data_ptr<int64_t>(),
            reinterpret_cast<uint16_t*>(sc_stream.data_ptr<int16_t>()),
            reinterpret_cast<uint32_t*>(init_states.data_ptr<int32_t>())
        );
    } else {
        nvfp4_sc_rans_encode_kernel<<<num_streams, 32, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
            reinterpret_cast<const uint32_t*>(d_freq.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(d_cdf.data_ptr<int32_t>()),
            stream_u16.data_ptr<int32_t>(),
            stream_chunk_off.data_ptr<int64_t>(),
            reinterpret_cast<uint16_t*>(sc_stream.data_ptr<int16_t>()),
            reinterpret_cast<uint32_t*>(init_states.data_ptr<int32_t>()),
            N, K, static_cast<int>(n_grp), static_cast<int>(p1_cols), static_cast<int>(p2_iters)
        );
    }

    // Pre-initialize the 65536-entry BF16->FP32 SiLU LUT at load time (never during graph capture).
    (void)get_silu_lut_ptr(B_q.device());

    return std::make_tuple(Bq_tm, sc_stream, stream_chunk_off, init_states, d_lut);
}

at::Tensor nvfp4_rans_linear_m1(
    const at::Tensor& A,
    const at::Tensor& Bq_tm,
    const at::Tensor& sc_stream,
    const at::Tensor& stream_chunk_off,
    const at::Tensor& init_states,
    const at::Tensor& lut,
    const at::Tensor& g_scale,
    int64_t N_in,
    int64_t n_grp,
    int64_t p1_cols,
    int64_t p2_iters,
    int64_t grid_in
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1);
    int N = static_cast<int>(N_in);
    int K = static_cast<int>(A.size(1));
    auto C = at::empty({1, N}, A.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int grid = static_cast<int>(grid_in);
    bool use_ldg = (K > 16384);
    int smem = use_ldg ? 0 : (K * 2);
    if (use_ldg) {
        nvfp4_rans_linear_m1_kernel<true><<<grid, 128, smem, stream>>>(
            reinterpret_cast<const uint32_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
            reinterpret_cast<const uint4*>(sc_stream.data_ptr()),
            stream_chunk_off.data_ptr<int64_t>(),
            reinterpret_cast<const uint32_t*>(init_states.data_ptr()),
            reinterpret_cast<const uint32_t*>(lut.data_ptr()),
            g_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            N, K, static_cast<int>(n_grp), static_cast<int>(p1_cols), static_cast<int>(p2_iters)
        );
    } else {
        nvfp4_rans_linear_m1_kernel<false><<<grid, 128, smem, stream>>>(
            reinterpret_cast<const uint32_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
            reinterpret_cast<const uint4*>(sc_stream.data_ptr()),
            stream_chunk_off.data_ptr<int64_t>(),
            reinterpret_cast<const uint32_t*>(init_states.data_ptr()),
            reinterpret_cast<const uint32_t*>(lut.data_ptr()),
            g_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            N, K, static_cast<int>(n_grp), static_cast<int>(p1_cols), static_cast<int>(p2_iters)
        );
    }
    return C;
}

// ---------------------------------------------------------------------------
// 7c. Zero-Tail-Wave 192-CTA Split-K NVFP4 Block-Scale rANS GEMV (M=1).
//     Executes the exact 48 Marlin blocks x 4 n32-subtiles = 192 CTAs (exact
//     4 CTAs/SM = 16 warps/SM on 48 SMs, 0 tail wave!) with 2-way ILP rANS
//     (st0, st1), STACK=0 scalar segment iteration, and fused single-launch
//     L2-coherent (.cg) atomic reduction (zero 2nd kernel launch, mism = 0).
// ---------------------------------------------------------------------------
__global__ void nvfp4_sc_rans_splitk192_measure_bq_kernel(
    const uint4* __restrict__ B_q,
    const uint8_t* __restrict__ B_s,
    const uint32_t* __restrict__ d_freq,
    const uint32_t* __restrict__ d_cdf,
    uint2* __restrict__ Bq_tm,
    int32_t* __restrict__ stream_u16,
    int64_t* __restrict__ stream_chunks,
    int N, int K, int iters
) {
    int sid = blockIdx.x;        // 0..767
    int cta = sid >> 2;          // 0..191
    int w = sid & 3;
    int mblk = cta >> 2;         // 0..47
    int half64 = (cta >> 1) & 1; // 0..1
    int sub32 = cta & 1;         // 0..1
    int lane = threadIdx.x;
    int g = lane >> 2;
    int part = lane & 3;
    int sub_s = part >> 1;
    int is_odd = part & 1;
    int k_tiles = K >> 7;
    int num_cols = N >> 7;
    int row_stride_int4 = N >> 1;
    int total_kt = num_cols * k_tiles;
    int t_start = mblk * iters;
    int t_end = min((mblk + 1) * iters, total_kt);

    uint2* q_dst = Bq_tm + static_cast<uint64_t>(sid) * iters * 64;
    int kt_ord = 0;
    for (int cur = t_start; cur < t_end; ) {
        int col = cur / k_tiles;
        int col_start = col * k_tiles;
        int seg_end_g = min(t_end, col_start + k_tiles);
        int seg_s = cur - col_start;
        int seg_e = seg_end_g - col_start;
        int n64 = col * 2 + half64;
        for (int kt = seg_s; kt < seg_e; ++kt, ++kt_ord) {
            int k16_0 = kt * 8 + w * 2;
            int k16_1 = k16_0 + 1;
            uint4 v0 = B_q[static_cast<uint64_t>(k16_0) * row_stride_int4 + n64 * 32 + lane];
            uint4 v1 = B_q[static_cast<uint64_t>(k16_1) * row_stride_int4 + n64 * 32 + lane];
            q_dst[kt_ord * 64 + lane]      = sub32 ? make_uint2(v0.z, v0.w) : make_uint2(v0.x, v0.y);
            q_dst[kt_ord * 64 + 32 + lane] = sub32 ? make_uint2(v1.z, v1.w) : make_uint2(v1.x, v1.y);
        }
        cur = seg_end_g;
    }

    uint32_t st0 = 65536u, st1 = 65536u;
    uint32_t total_u16 = 0u;
    #define SK_MEAS_SYM2(st_v, sym_val) do { \
        uint32_t _s = (sym_val); \
        uint32_t _f = d_freq[_s]; \
        uint32_t _c = d_cdf[_s]; \
        bool _em = ((st_v) >= (_f << 22)); \
        uint32_t _m = __ballot_sync(0xffffffffu, _em); \
        total_u16 += static_cast<uint32_t>(__popc(_m)); \
        if (_em) (st_v) >>= 16; \
        (st_v) = (((st_v) / _f) << 10) + ((st_v) % _f) + _c; \
    } while (0)

    for (int cur_end = t_end; cur_end > t_start; ) {
        int col = (cur_end - 1) / k_tiles;
        int col_start = col * k_tiles;
        int seg_start_g = max(t_start, col_start);
        int seg_s = seg_start_g - col_start;
        int seg_e = cur_end - col_start;
        int n64 = col * 2 + half64;
        for (int kt = seg_e - 1; kt >= seg_s; --kt) {
            int k16 = kt * 8 + w * 2 + sub_s;
            const uint8_t* p = B_s + static_cast<uint64_t>(k16) * N + n64 * 64 + g * 8 + sub32 * 4;
            uint32_t s0 = is_odd ? p[2] : p[0];
            uint32_t s1 = is_odd ? p[3] : p[1];
            SK_MEAS_SYM2(st1, s1);
            SK_MEAS_SYM2(st0, s0);
        }
        cur_end = seg_start_g;
    }
    #undef SK_MEAS_SYM2

    if (lane == 0) {
        stream_u16[sid] = static_cast<int32_t>(total_u16);
        int64_t ch = (static_cast<int64_t>(total_u16) + 127) / 128;
        stream_chunks[sid] = ch > 0 ? ch : 1;
    }
}

__global__ void nvfp4_sc_rans_splitk192_encode_kernel(
    const uint8_t* __restrict__ B_s,
    const uint32_t* __restrict__ d_freq,
    const uint32_t* __restrict__ d_cdf,
    const int32_t* __restrict__ stream_u16,
    const int64_t* __restrict__ stream_chunk_off,
    uint16_t* __restrict__ sc_stream,
    uint32_t* __restrict__ init_states,
    int N, int K, int iters
) {
    int sid = blockIdx.x;
    int cta = sid >> 2;
    int w = sid & 3;
    int mblk = cta >> 2;
    int half64 = (cta >> 1) & 1;
    int sub32 = cta & 1;
    int lane = threadIdx.x;
    int g = lane >> 2;
    int part = lane & 3;
    int sub_s = part >> 1;
    int is_odd = part & 1;
    int k_tiles = K >> 7;
    int num_cols = N >> 7;
    uint32_t lmask_lt = get_lanemask_lt();
    int total_kt = num_cols * k_tiles;
    int t_start = mblk * iters;
    int t_end = min((mblk + 1) * iters, total_kt);

    uint32_t st0 = 65536u, st1 = 65536u;
    int64_t ptr = stream_chunk_off[sid] * 128 + static_cast<int64_t>(stream_u16[sid]);

    #define SK_ENC_SYM2(st_v, sym_val) do { \
        uint32_t _s = (sym_val); \
        uint32_t _f = d_freq[_s]; \
        uint32_t _c = d_cdf[_s]; \
        bool _em = ((st_v) >= (_f << 22)); \
        uint32_t _m = __ballot_sync(0xffffffffu, _em); \
        ptr -= __popc(_m); \
        if (_em) { \
            uint32_t _rk = static_cast<uint32_t>(__popc(_m & lmask_lt)); \
            sc_stream[ptr + _rk] = static_cast<uint16_t>((st_v) & 0xFFFFu); \
            (st_v) >>= 16; \
        } \
        (st_v) = (((st_v) / _f) << 10) + ((st_v) % _f) + _c; \
    } while (0)

    for (int cur_end = t_end; cur_end > t_start; ) {
        int col = (cur_end - 1) / k_tiles;
        int col_start = col * k_tiles;
        int seg_start_g = max(t_start, col_start);
        int seg_s = seg_start_g - col_start;
        int seg_e = cur_end - col_start;
        int n64 = col * 2 + half64;
        for (int kt = seg_e - 1; kt >= seg_s; --kt) {
            int k16 = kt * 8 + w * 2 + sub_s;
            const uint8_t* p = B_s + static_cast<uint64_t>(k16) * N + n64 * 64 + g * 8 + sub32 * 4;
            uint32_t s0 = is_odd ? p[2] : p[0];
            uint32_t s1 = is_odd ? p[3] : p[1];
            SK_ENC_SYM2(st1, s1);
            SK_ENC_SYM2(st0, s0);
        }
        cur_end = seg_start_g;
    }
    #undef SK_ENC_SYM2

    uint64_t sb = static_cast<uint64_t>(sid) * 64;
    init_states[sb + lane] = st0;
    init_states[sb + 32 + lane] = st1;
}

template <bool USE_LDG_A>
__global__ void __launch_bounds__(128, 4) nvfp4_rans_splitk192_m1_kernel(
    const uint32_t* __restrict__ A,
    const uint2* __restrict__ Bq_tm,
    const uint2* __restrict__ sc_stream_u2,
    const int64_t* __restrict__ stream_chunk_off,
    const uint32_t* __restrict__ init_states,
    const uint32_t* __restrict__ d_lut,
    const float* __restrict__ g_scale,
    float* __restrict__ partials,
    __nv_bfloat16* __restrict__ C,
    int N, int K, int iters
) {
    extern __shared__ uint32_t sA32[];
    if constexpr (!USE_LDG_A) {
        const uint4* A4 = reinterpret_cast<const uint4*>(A);
        uint4* sA4 = reinterpret_cast<uint4*>(sA32);
        for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) {
            sA4[i] = A4[i];
        }
    }
    __shared__ uint32_t sLut[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        sLut[i] = d_lut[i];
    }
    __shared__ float s_acc[4][32];
    __shared__ uint2 s_sc[4][8];
    __shared__ uint16_t s_ring[4][256];
    __syncthreads();

    int cta = blockIdx.x;
    int mblk = cta >> 2;
    int half64 = (cta >> 1) & 1;
    int sub32 = cta & 1;
    int k_tiles = K >> 7;
    int num_cols = N >> 7;
    int total_kt = num_cols * k_tiles;
    int t_start = mblk * iters;
    int t_end = min((mblk + 1) * iters, total_kt);
    if (t_start >= t_end) return;

    int w = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int t = lane & 3;
    uint32_t lmask_lt = get_lanemask_lt();

    int sid = cta * 4 + w;
    const uint2* bq_w = Bq_tm + static_cast<uint64_t>(sid) * iters * 64;
    int64_t c_start = stream_chunk_off[sid];
    int s_chunks = static_cast<int>(stream_chunk_off[sid + 1] - c_start);
    const uint2* es2 = sc_stream_u2 + c_start * 32 + lane;
    uint64_t sb = static_cast<uint64_t>(sid) * 64;
    uint32_t st0 = init_states[sb + lane];
    uint32_t st1 = init_states[sb + 32 + lane];

    uint16_t* wbuf = s_ring[w];
    uint2* wbuf2 = reinterpret_cast<uint2*>(wbuf);
    uint16_t* s_sc16 = reinterpret_cast<uint16_t*>(s_sc[w]);

    uint2 c0, c1, c_pref;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];"
        : "=r"(c0.x), "=r"(c0.y) : "l"(es2));
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];"
        : "=r"(c1.x), "=r"(c1.y) : "l"(es2 + 32));
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];"
        : "=r"(c_pref.x), "=r"(c_pref.y) : "l"(es2 + 64));
    wbuf2[lane] = c0;
    wbuf2[32 + lane] = c1;
    __syncwarp();

    int chunk_idx = 3;
    int pos = 0;
    int refill_at = 128;
    int kt_ord = 0;
    float gs = *g_scale;
    int32_t* done_cnt = reinterpret_cast<int32_t*>(partials + 3 * 6656);

    auto load_q2 = [&](const uint2* p, uint2& q) {
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];"
            : "=r"(q.x), "=r"(q.y) : "l"(p));
    };

    auto run_sub = [&](int kt, int sub, const uint2& qv, uint32_t sc_sub,
                       float (&d0)[4], float (&d1)[4], float (&d2)[4], float (&d3)[4]) {
        int k16 = (kt << 3) + (w << 1) + sub;
        uint32_t a0, a2;
        if constexpr (USE_LDG_A) {
            a0 = (group == 0) ? __ldg(A + k16 * 8 + t) : 0u;
            a2 = (group == 0) ? __ldg(A + k16 * 8 + 4 + t) : 0u;
        } else {
            a0 = (group == 0) ? sA32[k16 * 8 + t] : 0u;
            a2 = (group == 0) ? sA32[k16 * 8 + 4 + t] : 0u;
        }
        uint32_t q_arr[2] = {qv.x, qv.y};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            uint32_t q = q_arr[j];
            uint32_t q0_ = q << 12, q1_ = q << 8, q2_ = q << 4;
            uint32_t f0 = lop3_or_and(q0_, 0x80008000u, (q0_ >> 6) & 0x01C001C0u);
            uint32_t f1 = lop3_or_and(q1_, 0x80008000u, (q1_ >> 6) & 0x01C001C0u);
            uint32_t f2 = lop3_or_and(q2_, 0x80008000u, (q2_ >> 6) & 0x01C001C0u);
            uint32_t f3 = lop3_or_and(q,   0x80008000u, (q   >> 6) & 0x01C001C0u);
            __nv_bfloat162 s0 = dequant_s_bf162((sc_sub >> (j * 8)) & 0xFFu);
            __nv_bfloat162 s1 = dequant_s_bf162((sc_sub >> (16 + j * 8)) & 0xFFu);
            mma_m16n8k16_bf16_m1(d0[2 * j], d1[2 * j], d2[2 * j], d3[2 * j],
                                 a0, a2, hmul2_u32(f0, s0), hmul2_u32(f1, s0));
            mma_m16n8k16_bf16_m1(d0[2 * j + 1], d1[2 * j + 1], d2[2 * j + 1], d3[2 * j + 1],
                                 a0, a2, hmul2_u32(f2, s1), hmul2_u32(f3, s1));
        }
    };

    for (int cur = t_start; cur < t_end; ) {
        int col = cur / k_tiles;
        int col_start = col * k_tiles;
        int seg_end_g = min(t_end, col_start + k_tiles);
        int seg_start = cur - col_start;
        int seg_end = seg_end_g - col_start;
        cur = seg_end_g;

        int n64 = col * 2 + half64;
        float d0[4] = {0}, d1[4] = {0}, d2[4] = {0}, d3[4] = {0};

        uint2 q0, q1;
        load_q2(bq_w + kt_ord * 64 + lane, q0);
        load_q2(bq_w + kt_ord * 64 + 32 + lane, q1);

        for (int kt = seg_start; kt < seg_end; ++kt, ++kt_ord) {
            uint32_t e0 = sLut[st0 & 1023u];
            uint32_t e1 = sLut[st1 & 1023u];
            st0 = ((e0 >> 10) & 1023u) * (st0 >> 10) + (e0 & 1023u);
            st1 = ((e1 >> 10) & 1023u) * (st1 >> 10) + (e1 & 1023u);
            bool n0 = (st0 < 65536u), n1 = (st1 < 65536u);
            uint32_t m0 = __ballot_sync(0xffffffffu, n0);
            uint32_t m1 = __ballot_sync(0xffffffffu, n1);
            int p1 = pos + __popc(m0);
            if (n0) st0 = (st0 << 16) | static_cast<uint32_t>(wbuf[(pos + __popc(m0 & lmask_lt)) & 255]);
            if (n1) st1 = (st1 << 16) | static_cast<uint32_t>(wbuf[(p1  + __popc(m1 & lmask_lt)) & 255]);
            pos = p1 + __popc(m1);

            s_sc16[lane] = static_cast<uint16_t>((e0 >> 24) | ((e1 >> 16) & 0xFF00u));
            __syncwarp();
            uint2 sc_kt = s_sc[w][group];

            if (pos >= refill_at) {
                int slot = ((refill_at >> 7) - 1) & 1;
                __syncwarp();
                wbuf2[slot * 32 + lane] = c_pref;
                __syncwarp();
                if (chunk_idx < s_chunks) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];"
                        : "=r"(c_pref.x), "=r"(c_pref.y)
                        : "l"(es2 + static_cast<uint64_t>(chunk_idx) * 32));
                }
                chunk_idx++;
                refill_at += 128;
            }

            uint2 qn0, qn1;
            if (kt + 1 < seg_end) {
                load_q2(bq_w + (kt_ord + 1) * 64 + lane, qn0);
            }
            run_sub(kt, 0, q0, sc_kt.x, d0, d1, d2, d3);
            if (kt + 1 < seg_end) {
                load_q2(bq_w + (kt_ord + 1) * 64 + 32 + lane, qn1);
            }
            run_sub(kt, 1, q1, sc_kt.y, d0, d1, d2, d3);
            q0 = qn0;
            q1 = qn1;
        }

        if (group == 0) {
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                s_acc[w][m * 8 + t * 2] = d0[m];
                s_acc[w][m * 8 + t * 2 + 1] = d1[m];
            }
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            int c = threadIdx.x;
            float P = s_acc[0][c] + (s_acc[1][c] + (s_acc[3][c] + s_acc[2][c]));
            int out_idx = n64 * 64 + sub32 * 32 + c;
            if (seg_start == 0 && seg_end == k_tiles) {
                C[out_idx] = __float2bfloat16_rn(P * gs);
            } else {
                int b0 = col_start / iters;
                int b1 = (col_start + k_tiles - 1) / iters;
                int n_splits = b1 - b0 + 1;
                int slot = mblk - b0;
                asm volatile("st.global.cg.f32 [%0], %1;" :: "l"(partials + slot * N + out_idx), "f"(P));
                __threadfence();
                int idx32 = n64 * 2 + sub32;
                int prev_done = 0;
                if (c == 0) {
                    prev_done = atomicAdd(done_cnt + idx32, 1);
                }
                prev_done = __shfl_sync(0xffffffffu, prev_done, 0);
                if (prev_done == n_splits - 1) {
                    float p0, p1;
                    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(p0) : "l"(partials + out_idx));
                    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(p1) : "l"(partials + N + out_idx));
                    float sum = p0 + p1;
                    if (n_splits == 3) {
                        float p2;
                        asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(p2) : "l"(partials + 2 * N + out_idx));
                        sum = p0 + (p1 + p2);
                    }
                    C[out_idx] = __float2bfloat16_rn(sum * gs);
                    if (c == 0) {
                        atomicExch(done_cnt + idx32, 0);
                    }
                }
            }
        }
        __syncthreads();
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> repack_nvfp4_rans_splitk96(
    const at::Tensor& B_q,
    const at::Tensor& B_s,
    int64_t N_in,
    int64_t K_in,
    int64_t iters_in
) {
    int N = static_cast<int>(N_in);
    int K = static_cast<int>(K_in);
    int iters = static_cast<int>(iters_in);
    TORCH_CHECK(B_q.is_cuda() && B_s.is_cuda());
    TORCH_CHECK(N % 128 == 0 && K % 128 == 0);
    constexpr int num_streams = 192 * 4;
    auto dev_opts = B_q.options();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto g_hist = at::zeros({256}, dev_opts.dtype(at::kLong));
    int64_t num_u32 = (static_cast<int64_t>(K >> 4) * N) >> 2;
    int hist_blocks = static_cast<int>(std::min<int64_t>((num_u32 + 255) / 256, 2048));
    nvfp4_sc_hist_kernel<<<hist_blocks, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(B_s.data_ptr()),
        num_u32,
        reinterpret_cast<unsigned long long*>(g_hist.data_ptr<int64_t>())
    );
    auto h_hist_t = g_hist.to(at::kCPU);
    const int64_t* h_hist = h_hist_t.data_ptr<int64_t>();

    constexpr int M_PROB = 1024;
    int64_t total = 0;
    for (int i = 0; i < 256; ++i) total += h_hist[i];
    std::vector<uint32_t> freq(256, 0u);
    std::vector<double> exact(256, 0.0);
    std::vector<double> rem(256, -1.0);
    int sum_f = 0;
    for (int i = 0; i < 256; ++i) {
        if (h_hist[i] > 0) {
            double e = static_cast<double>(h_hist[i]) * static_cast<double>(M_PROB) / static_cast<double>(total);
            exact[i] = e;
            int f = static_cast<int>(std::floor(e));
            if (f < 1) f = 1;
            if (f > M_PROB - 1) f = M_PROB - 1;
            freq[i] = static_cast<uint32_t>(f);
            rem[i] = e - static_cast<double>(f);
            sum_f += f;
        }
    }
    while (sum_f < M_PROB) {
        int best = -1;
        double best_r = -1e9;
        for (int i = 0; i < 256; ++i) {
            if (h_hist[i] > 0 && freq[i] < static_cast<uint32_t>(M_PROB - 1) && rem[i] > best_r) {
                best_r = rem[i];
                best = i;
            }
        }
        if (best < 0) {
            for (int i = 0; i < 256; ++i) {
                if (freq[i] == 0u) { best = i; break; }
            }
        }
        freq[best] += 1u;
        rem[best] = -1e9;
        sum_f += 1;
    }
    while (sum_f > M_PROB) {
        int best = -1;
        double worst_excess = -1e9;
        for (int i = 0; i < 256; ++i) {
            if (freq[i] > 1u) {
                double ex = static_cast<double>(freq[i]) - exact[i];
                if (ex > worst_excess) {
                    worst_excess = ex;
                    best = i;
                }
            }
        }
        freq[best] -= 1u;
        sum_f -= 1;
    }

    std::vector<uint32_t> cdf(256, 0u);
    std::vector<uint32_t> lut(M_PROB, 0u);
    uint32_t acc = 0u;
    for (int s = 0; s < 256; ++s) {
        cdf[s] = acc;
        for (uint32_t j = 0; j < freq[s]; ++j) {
            lut[acc + j] = (static_cast<uint32_t>(s) << 24) | (freq[s] << 10) | j;
        }
        acc += freq[s];
    }

    auto d_freq = at::from_blob(freq.data(), {256}, at::kInt).to(B_q.device());
    auto d_cdf  = at::from_blob(cdf.data(), {256}, at::kInt).to(B_q.device());
    auto d_lut  = at::from_blob(lut.data(), {M_PROB}, at::kInt).to(B_q.device());

    auto Bq_tm = at::empty({static_cast<int64_t>(num_streams) * iters * 64 * 2}, dev_opts.dtype(at::kInt));
    auto stream_u16 = at::empty({num_streams}, dev_opts.dtype(at::kInt));
    auto stream_chunks = at::empty({num_streams}, dev_opts.dtype(at::kLong));
    nvfp4_sc_rans_splitk192_measure_bq_kernel<<<num_streams, 32, 0, stream>>>(
        reinterpret_cast<const uint4*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
        reinterpret_cast<const uint32_t*>(d_freq.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(d_cdf.data_ptr<int32_t>()),
        reinterpret_cast<uint2*>(Bq_tm.data_ptr()),
        stream_u16.data_ptr<int32_t>(),
        stream_chunks.data_ptr<int64_t>(),
        N, K, iters
    );

    auto stream_chunk_off = at::zeros({num_streams + 1}, dev_opts.dtype(at::kLong));
    stream_chunk_off.narrow(0, 1, num_streams).copy_(at::cumsum(stream_chunks, 0));
    int64_t total_chunks = stream_chunk_off[num_streams].item<int64_t>();

    auto sc_stream = at::zeros({(total_chunks + 3) * 128}, dev_opts.dtype(at::kShort));
    auto init_states = at::empty({num_streams, 64}, dev_opts.dtype(at::kInt));

    nvfp4_sc_rans_splitk192_encode_kernel<<<num_streams, 32, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
        reinterpret_cast<const uint32_t*>(d_freq.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(d_cdf.data_ptr<int32_t>()),
        stream_u16.data_ptr<int32_t>(),
        stream_chunk_off.data_ptr<int64_t>(),
        reinterpret_cast<uint16_t*>(sc_stream.data_ptr<int16_t>()),
        reinterpret_cast<uint32_t*>(init_states.data_ptr<int32_t>()),
        N, K, iters
    );

    return std::make_tuple(Bq_tm, sc_stream, stream_chunk_off, init_states, d_lut);
}

at::Tensor nvfp4_rans_splitk96_m1(
    const at::Tensor& A,
    const at::Tensor& Bq_tm,
    const at::Tensor& sc_stream,
    const at::Tensor& stream_chunk_off,
    const at::Tensor& init_states,
    const at::Tensor& lut,
    const at::Tensor& g_scale,
    at::Tensor partials,
    int64_t N_in,
    int64_t iters_in
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1);
    int N = static_cast<int>(N_in);
    int K = static_cast<int>(A.size(1));
    int iters = static_cast<int>(iters_in);
    auto C = at::empty({1, N}, A.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bool use_ldg = (K > 16384);
    int smem = use_ldg ? 0 : (K * 2);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(nvfp4_rans_splitk192_m1_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 20480);
        attr_set = true;
    }
    if (use_ldg) {
        nvfp4_rans_splitk192_m1_kernel<true><<<192, 128, smem, stream>>>(
            reinterpret_cast<const uint32_t*>(A.data_ptr()),
            reinterpret_cast<const uint2*>(Bq_tm.data_ptr()),
            reinterpret_cast<const uint2*>(sc_stream.data_ptr()),
            stream_chunk_off.data_ptr<int64_t>(),
            reinterpret_cast<const uint32_t*>(init_states.data_ptr()),
            reinterpret_cast<const uint32_t*>(lut.data_ptr()),
            g_scale.data_ptr<float>(),
            partials.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            N, K, iters
        );
    } else {
        nvfp4_rans_splitk192_m1_kernel<false><<<192, 128, smem, stream>>>(
            reinterpret_cast<const uint32_t*>(A.data_ptr()),
            reinterpret_cast<const uint2*>(Bq_tm.data_ptr()),
            reinterpret_cast<const uint2*>(sc_stream.data_ptr()),
            stream_chunk_off.data_ptr<int64_t>(),
            reinterpret_cast<const uint32_t*>(init_states.data_ptr()),
            reinterpret_cast<const uint32_t*>(lut.data_ptr()),
            g_scale.data_ptr<float>(),
            partials.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            N, K, iters
        );
    }
    return C;
}

at::Tensor nvfp4_rans_gate_up_silu_m1(
    const at::Tensor& A,
    const at::Tensor& Bq_tm,
    const at::Tensor& sc_stream,
    const at::Tensor& stream_chunk_off,
    const at::Tensor& init_states,
    const at::Tensor& lut,
    const at::Tensor& g_scale
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1 && A.size(1) == 6656);
    auto C = at::empty({1, 19968}, A.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (init_states.size(0) == 1248) {
        constexpr int smem = 6656 * 2 + 4096 + 13 * 1024 + 2560;
        nvfp4_rans_gate_up_silu_w16_m1_kernel<false><<<96, 416, smem, stream>>>(
            reinterpret_cast<const uint32_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
            reinterpret_cast<const uint4*>(sc_stream.data_ptr()),
            stream_chunk_off.data_ptr<int64_t>(),
            reinterpret_cast<const uint32_t*>(init_states.data_ptr()),
            reinterpret_cast<const uint32_t*>(lut.data_ptr()),
            g_scale.data_ptr<float>(),
            nullptr,
            C.data_ptr()
        );
    } else {
        nvfp4_rans_gate_up_silu_m1_kernel<false><<<156, 128, 6656 * 2, stream>>>(
            reinterpret_cast<const uint32_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
            reinterpret_cast<const uint4*>(sc_stream.data_ptr()),
            stream_chunk_off.data_ptr<int64_t>(),
            reinterpret_cast<const uint32_t*>(init_states.data_ptr()),
            reinterpret_cast<const uint32_t*>(lut.data_ptr()),
            g_scale.data_ptr<float>(),
            nullptr,
            C.data_ptr()
        );
    }
    return C;
}

at::Tensor nvfp4_rans_gate_up_silu_fp8quant_m1(
    const at::Tensor& A,
    const at::Tensor& Bq_tm,
    const at::Tensor& sc_stream,
    const at::Tensor& stream_chunk_off,
    const at::Tensor& init_states,
    const at::Tensor& lut,
    const at::Tensor& g_scale,
    const at::Tensor& out_scale_a
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1 && A.size(1) == 6656);
    auto C = at::empty({1, 19968}, A.options().dtype(at::kFloat8_e4m3fn));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (init_states.size(0) == 1248) {
        constexpr int smem = 6656 * 2 + 4096 + 13 * 1024 + 2560;
        nvfp4_rans_gate_up_silu_w16_m1_kernel<true><<<96, 416, smem, stream>>>(
            reinterpret_cast<const uint32_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
            reinterpret_cast<const uint4*>(sc_stream.data_ptr()),
            stream_chunk_off.data_ptr<int64_t>(),
            reinterpret_cast<const uint32_t*>(init_states.data_ptr()),
            reinterpret_cast<const uint32_t*>(lut.data_ptr()),
            g_scale.data_ptr<float>(),
            out_scale_a.data_ptr<float>(),
            C.data_ptr()
        );
    } else {
        nvfp4_rans_gate_up_silu_m1_kernel<true><<<156, 128, 6656 * 2, stream>>>(
            reinterpret_cast<const uint32_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
            reinterpret_cast<const uint4*>(sc_stream.data_ptr()),
            stream_chunk_off.data_ptr<int64_t>(),
            reinterpret_cast<const uint32_t*>(init_states.data_ptr()),
            reinterpret_cast<const uint32_t*>(lut.data_ptr()),
            g_scale.data_ptr<float>(),
            out_scale_a.data_ptr<float>(),
            C.data_ptr()
        );
    }
    return C;
}

__global__ void quant_fp8_triton_exact_kernel(
    const uint4* __restrict__ x4,
    const float* __restrict__ scale_a,
    uint2* __restrict__ out2,
    int n_vec8
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vec8) return;
    float inv_sa = __fdividef(1.0f, *scale_a);
    uint4 v = x4[i];
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&v);
    uint8_t q[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        q[k] = quant_bf16_to_e4m3(p[k], inv_sa);
    }
    out2[i] = *reinterpret_cast<const uint2*>(q);
}

at::Tensor quant_fp8_triton_exact(
    const at::Tensor& x,
    const at::Tensor& scale_a
) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == at::kBFloat16);
    TORCH_CHECK((x.numel() & 7) == 0);
    auto out = at::empty_like(x, x.options().dtype(at::kFloat8_e4m3fn));
    int n_vec8 = static_cast<int>(x.numel() >> 3);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    quant_fp8_triton_exact_kernel<<<(n_vec8 + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(x.data_ptr()),
        scale_a.data_ptr<float>(),
        reinterpret_cast<uint2*>(out.data_ptr()),
        n_vec8
    );
    return out;
}

__global__ void finalize_192_argmax_only_kernel(
    const float* __restrict__ blk_val,
    const int32_t* __restrict__ blk_idx,
    int32_t* __restrict__ out_token
) {
    int tid = threadIdx.x;
    float v = (tid < 192) ? blk_val[tid] : -3.402823466e+38f;
    int32_t idx = (tid < 192) ? blk_idx[tid] : 0;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, v, off);
        int32_t oi = __shfl_down_sync(0xffffffffu, idx, off);
        if (ov > v || (ov == v && oi < idx)) {
            v = ov;
            idx = oi;
        }
    }
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    if ((tid & 31) == 0) {
        sv[tid >> 5] = v;
        si[tid >> 5] = idx;
    }
    __syncthreads();
    if (tid == 0) {
        float bv = sv[0];
        int32_t bi = si[0];
        #pragma unroll
        for (int w = 1; w < 6; ++w) {
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) {
                bv = sv[w];
                bi = si[w];
            }
        }
        out_token[0] = bi;
    }
}

void nvfp4_rans_lmhead_m1_argmax_only(
    const at::Tensor& A,
    const at::Tensor& Bq_tm,
    const at::Tensor& sc_stream,
    const at::Tensor& stream_chunk_off,
    const at::Tensor& init_states,
    const at::Tensor& lut,
    const at::Tensor& g_scale,
    double output_multiplier,
    double softcap,
    at::Tensor blk_max_val,
    at::Tensor blk_max_idx,
    at::Tensor out_token
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1 && A.size(1) == 6656);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    nvfp4_rans_lmhead_m1_fused_argmax_kernel<<<192, 128, 6656 * 2, stream>>>(
        reinterpret_cast<const uint32_t*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
        reinterpret_cast<const uint4*>(sc_stream.data_ptr()),
        stream_chunk_off.data_ptr<int64_t>(),
        reinterpret_cast<const uint32_t*>(init_states.data_ptr()),
        reinterpret_cast<const uint32_t*>(lut.data_ptr()),
        g_scale.data_ptr<float>(),
        static_cast<float>(output_multiplier),
        static_cast<float>(softcap),
        blk_max_val.data_ptr<float>(),
        blk_max_idx.data_ptr<int32_t>()
    );
    finalize_192_argmax_only_kernel<<<1, 192, 0, stream>>>(
        blk_max_val.data_ptr<float>(),
        blk_max_idx.data_ptr<int32_t>(),
        out_token.data_ptr<int32_t>()
    );
}

void nvfp4_rans_lmhead_m1_argmax_step_update(
    const at::Tensor& A,
    const at::Tensor& Bq_tm,
    const at::Tensor& sc_stream,
    const at::Tensor& stream_chunk_off,
    const at::Tensor& init_states,
    const at::Tensor& lut,
    const at::Tensor& g_scale,
    double output_multiplier,
    double softcap,
    at::Tensor blk_max_val,
    at::Tensor blk_max_idx,
    at::Tensor out_tokens,
    at::Tensor step_idx,
    const at::Tensor& step_limit,
    at::Tensor input_ids,
    at::Tensor positions,
    at::Tensor seq_lens,
    const at::Tensor& input_block_table_ptrs,
    const at::Tensor& block_sizes,
    at::Tensor slot_mappings
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1 && A.size(1) == 6656);
    int num_groups = static_cast<int>(block_sizes.size(0));
    int64_t slot_stride = slot_mappings.stride(0);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    nvfp4_rans_lmhead_m1_fused_argmax_kernel<<<192, 128, 6656 * 2, stream>>>(
        reinterpret_cast<const uint32_t*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(Bq_tm.data_ptr()),
        reinterpret_cast<const uint4*>(sc_stream.data_ptr()),
        stream_chunk_off.data_ptr<int64_t>(),
        reinterpret_cast<const uint32_t*>(init_states.data_ptr()),
        reinterpret_cast<const uint32_t*>(lut.data_ptr()),
        g_scale.data_ptr<float>(),
        static_cast<float>(output_multiplier),
        static_cast<float>(softcap),
        blk_max_val.data_ptr<float>(),
        blk_max_idx.data_ptr<int32_t>()
    );
    finalize_192_argmax_step_update_kernel<<<1, 192, 0, stream>>>(
        blk_max_val.data_ptr<float>(),
        blk_max_idx.data_ptr<int32_t>(),
        out_tokens.data_ptr<int32_t>(),
        step_idx.data_ptr<int32_t>(),
        step_limit.data_ptr<int32_t>(),
        input_ids.data_ptr<int32_t>(),
        positions.data_ptr<int64_t>(),
        seq_lens.data_ptr<int32_t>(),
        reinterpret_cast<const uint64_t*>(input_block_table_ptrs.data_ptr()),
        block_sizes.data_ptr<int32_t>(),
        slot_mappings.data_ptr<int64_t>(),
        slot_stride,
        num_groups
    );
}

// ---------------------------------------------------------------------------
// 8. Original 32-lane Warp-Interleaved Lossless BF16 Exponent rANS GEMV
//    (12 BF16 attention layers: 6x qkv_proj + 6x output_gate_proj, 695 MB ->
//    ~459 MB, ~0.66x lossless compression).
//    - Sign|mantissa (8 bits) packed as coalesced 512B uint4 per k=64 tile.
//    - 8-bit BF16 exponent (~2.57 bits entropy) rANS-coded (10-bit, 4 KB LUT,
//      4-way ILP, 512B-aligned chunks).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t bf16_sm_byte(uint16_t u) {
    return static_cast<uint8_t>((u & 0x7Fu) | ((u >> 8) & 0x80u));
}

__device__ __forceinline__ uint32_t bf16_exp_byte(uint16_t u) {
    return static_cast<uint32_t>((u >> 7) & 0xFFu);
}

__device__ __forceinline__ float bf16_reconstruct_f32(uint32_t sm_byte, uint32_t e_entry) {
    uint32_t bits = ((sm_byte & 0x80u) << 24)
                  | ((e_entry & 0xFF000000u) >> 1)
                  | ((sm_byte & 0x7Fu) << 16);
    return __uint_as_float(bits);
}

__global__ void bf16_exp_hist_kernel(
    const uint32_t* __restrict__ B_u32,
    int64_t num_u32,
    unsigned long long* __restrict__ g_hist
) {
    __shared__ uint32_t s_hist[256];
    if (threadIdx.x < 256) s_hist[threadIdx.x] = 0u;
    __syncthreads();
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = idx; i < num_u32; i += stride) {
        uint32_t w = B_u32[i];
        atomicAdd(&s_hist[(w >> 7) & 0xFFu], 1u);
        atomicAdd(&s_hist[(w >> 23) & 0xFFu], 1u);
    }
    __syncthreads();
    if (threadIdx.x < 256 && s_hist[threadIdx.x] > 0u) {
        atomicAdd(&g_hist[threadIdx.x], static_cast<unsigned long long>(s_hist[threadIdx.x]));
    }
}

__global__ void bf16_rans_measure_sm_kernel(
    const uint16_t* __restrict__ B,
    uint4* __restrict__ sm_pack,
    const uint32_t* __restrict__ d_freq,
    const uint32_t* __restrict__ d_cdf,
    int32_t* __restrict__ tile_u16,
    int64_t* __restrict__ tile_chunks,
    int N, int K
) {
    // Bit-exact cuBLAS gemvx lane_stride=32 layout: 8 rows per warp stream,
    // k stepped by 512 (16 elements per lane per step, k_steps = K/512).
    int nt = blockIdx.x;
    int k_steps = K >> 9;
    int lane = threadIdx.x;
    uint4* sm_dst = sm_pack + (static_cast<uint64_t>(nt) * 8 * k_steps) * 32;

    uint32_t st0 = 65536u, st1 = 65536u, st2 = 65536u, st3 = 65536u;
    uint32_t total_u16 = 0u;

    #define BF_MEAS_SYM(st_v, sym_val) do { \
        uint32_t _s = (sym_val); \
        uint32_t _f = d_freq[_s]; \
        uint32_t _c = d_cdf[_s]; \
        bool _em = ((st_v) >= (_f << 22)); \
        uint32_t _m = __ballot_sync(0xffffffffu, _em); \
        total_u16 += static_cast<uint32_t>(__popc(_m)); \
        if (_em) (st_v) >>= 16; \
        (st_v) = (((st_v) / _f) << 10) + ((st_v) % _f) + _c; \
    } while (0)

    for (int r = 7; r >= 0; --r) {
        const uint16_t* Brow = B + static_cast<uint64_t>(nt * 8 + r) * K;
        for (int ks = k_steps - 1; ks >= 0; --ks) {
            int kb = ks << 9;
            uint16_t u[16];
            #pragma unroll
            for (int m = 0; m < 16; ++m) {
                u[m] = Brow[kb + m * 32 + lane];
            }
            uint8_t smb[16];
            #pragma unroll
            for (int m = 0; m < 16; ++m) smb[m] = bf16_sm_byte(u[m]);
            sm_dst[static_cast<uint64_t>(r * k_steps + ks) * 32 + lane] = *reinterpret_cast<const uint4*>(smb);

            for (int q = 3; q >= 0; --q) {
                BF_MEAS_SYM(st3, bf16_exp_byte(u[q * 4 + 3]));
                BF_MEAS_SYM(st2, bf16_exp_byte(u[q * 4 + 2]));
                BF_MEAS_SYM(st1, bf16_exp_byte(u[q * 4 + 1]));
                BF_MEAS_SYM(st0, bf16_exp_byte(u[q * 4 + 0]));
            }
        }
    }
    #undef BF_MEAS_SYM

    if (lane == 0) {
        tile_u16[nt] = static_cast<int32_t>(total_u16);
        int64_t ch = (static_cast<int64_t>(total_u16) + 255) / 256;
        tile_chunks[nt] = ch > 0 ? ch : 1;
    }
}

__global__ void bf16_rans_encode_kernel(
    const uint16_t* __restrict__ B,
    const uint32_t* __restrict__ d_freq,
    const uint32_t* __restrict__ d_cdf,
    const int32_t* __restrict__ tile_u16,
    const int64_t* __restrict__ tile_chunk_off,
    uint16_t* __restrict__ exp_stream,
    uint32_t* __restrict__ init_states,
    int N, int K
) {
    int nt = blockIdx.x;
    int k_steps = K >> 9;
    int lane = threadIdx.x;
    uint32_t lmask_lt = get_lanemask_lt();

    uint32_t st0 = 65536u, st1 = 65536u, st2 = 65536u, st3 = 65536u;
    int64_t ptr = tile_chunk_off[nt] * 256 + static_cast<int64_t>(tile_u16[nt]);

    #define BF_ENC_SYM(st_v, sym_val) do { \
        uint32_t _s = (sym_val); \
        uint32_t _f = d_freq[_s]; \
        uint32_t _c = d_cdf[_s]; \
        bool _em = ((st_v) >= (_f << 22)); \
        uint32_t _m = __ballot_sync(0xffffffffu, _em); \
        ptr -= __popc(_m); \
        if (_em) { \
            uint32_t _rk = static_cast<uint32_t>(__popc(_m & lmask_lt)); \
            exp_stream[ptr + _rk] = static_cast<uint16_t>((st_v) & 0xFFFFu); \
            (st_v) >>= 16; \
        } \
        (st_v) = (((st_v) / _f) << 10) + ((st_v) % _f) + _c; \
    } while (0)

    for (int r = 7; r >= 0; --r) {
        const uint16_t* Brow = B + static_cast<uint64_t>(nt * 8 + r) * K;
        for (int ks = k_steps - 1; ks >= 0; --ks) {
            int kb = ks << 9;
            uint16_t u[16];
            #pragma unroll
            for (int m = 0; m < 16; ++m) {
                u[m] = Brow[kb + m * 32 + lane];
            }
            for (int q = 3; q >= 0; --q) {
                BF_ENC_SYM(st3, bf16_exp_byte(u[q * 4 + 3]));
                BF_ENC_SYM(st2, bf16_exp_byte(u[q * 4 + 2]));
                BF_ENC_SYM(st1, bf16_exp_byte(u[q * 4 + 1]));
                BF_ENC_SYM(st0, bf16_exp_byte(u[q * 4 + 0]));
            }
        }
    }
    #undef BF_ENC_SYM

    uint64_t sb = static_cast<uint64_t>(nt) * 128;
    init_states[sb + lane] = st0;
    init_states[sb + 32 + lane] = st1;
    init_states[sb + 64 + lane] = st2;
    init_states[sb + 96 + lane] = st3;
}

__global__ void bf16_rans_m1_kernel(
    const uint16_t* __restrict__ A,
    const uint4* __restrict__ sm_pack,
    const uint4* __restrict__ exp_stream_u4,
    const int64_t* __restrict__ tile_chunk_off,
    const uint32_t* __restrict__ init_states,
    const uint32_t* __restrict__ d_lut,
    __nv_bfloat16* __restrict__ C,
    int N, int K
) {
    extern __shared__ uint8_t smem_bf_raw[];
    uint16_t* sA = reinterpret_cast<uint16_t*>(smem_bf_raw);
    uint32_t* sLut = reinterpret_cast<uint32_t*>(smem_bf_raw + K * 2);
    uint16_t* sRing = reinterpret_cast<uint16_t*>(smem_bf_raw + K * 2 + 4096);

    const uint4* A4 = reinterpret_cast<const uint4*>(A);
    uint4* sA4 = reinterpret_cast<uint4*>(sA);
    for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) {
        sA4[i] = A4[i];
    }
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        sLut[i] = d_lut[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int total_warps = gridDim.x * warps_per_block;
    int global_warp = blockIdx.x * warps_per_block + warp_id;
    uint32_t lmask_lt = get_lanemask_lt();

    uint16_t* wbuf = sRing + warp_id * 512;
    uint4* wbuf4 = reinterpret_cast<uint4*>(wbuf);

    int num_n8 = N >> 3;
    int k_steps = K >> 9;
    int total_steps = 8 * k_steps;

    #define BF_DEC_QUAD(o0, o1, o2, o3) do { \
        uint32_t _e0 = sLut[st0 & 1023u]; \
        uint32_t _e1 = sLut[st1 & 1023u]; \
        uint32_t _e2 = sLut[st2 & 1023u]; \
        uint32_t _e3 = sLut[st3 & 1023u]; \
        st0 = ((_e0 >> 10) & 1023u) * (st0 >> 10) + (_e0 & 1023u); \
        st1 = ((_e1 >> 10) & 1023u) * (st1 >> 10) + (_e1 & 1023u); \
        st2 = ((_e2 >> 10) & 1023u) * (st2 >> 10) + (_e2 & 1023u); \
        st3 = ((_e3 >> 10) & 1023u) * (st3 >> 10) + (_e3 & 1023u); \
        bool _n0 = (st0 < 65536u); \
        bool _n1 = (st1 < 65536u); \
        bool _n2 = (st2 < 65536u); \
        bool _n3 = (st3 < 65536u); \
        uint32_t _m0 = __ballot_sync(0xffffffffu, _n0); \
        uint32_t _m1 = __ballot_sync(0xffffffffu, _n1); \
        uint32_t _m2 = __ballot_sync(0xffffffffu, _n2); \
        uint32_t _m3 = __ballot_sync(0xffffffffu, _n3); \
        int _p1 = pos + __popc(_m0); \
        int _p2 = _p1 + __popc(_m1); \
        int _p3 = _p2 + __popc(_m2); \
        if (_n0) st0 = (st0 << 16) | static_cast<uint32_t>(wbuf[(pos + __popc(_m0 & lmask_lt)) & 511]); \
        if (_n1) st1 = (st1 << 16) | static_cast<uint32_t>(wbuf[(_p1 + __popc(_m1 & lmask_lt)) & 511]); \
        if (_n2) st2 = (st2 << 16) | static_cast<uint32_t>(wbuf[(_p2 + __popc(_m2 & lmask_lt)) & 511]); \
        if (_n3) st3 = (st3 << 16) | static_cast<uint32_t>(wbuf[(_p3 + __popc(_m3 & lmask_lt)) & 511]); \
        pos = _p3 + __popc(_m3); \
        o0 = _e0; o1 = _e1; o2 = _e2; o3 = _e3; \
    } while (0)

    #define BF_REFILL() do { \
        if (pos >= refill_at) { \
            int slot = ((refill_at >> 8) - 1) & 1; \
            __syncwarp(); \
            wbuf4[slot * 32 + lane] = c_pref0; \
            __syncwarp(); \
            c_pref0 = c_pref1; \
            if (chunk_idx < tile_chunks) { \
                asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" \
                    : "=r"(c_pref1.x), "=r"(c_pref1.y), "=r"(c_pref1.z), "=r"(c_pref1.w) \
                    : "l"(es4 + static_cast<uint64_t>(chunk_idx) * 32)); \
            } \
            chunk_idx++; \
            refill_at += 256; \
        } \
    } while (0)

    #define BF_ACC4(sm_word, kb_off) do { \
        uint32_t _q0, _q1, _q2, _q3; \
        BF_DEC_QUAD(_q0, _q1, _q2, _q3); \
        float _a0 = __uint_as_float(static_cast<uint32_t>(sA[(kb_off) + 0 * 32 + lane]) << 16); \
        float _a1 = __uint_as_float(static_cast<uint32_t>(sA[(kb_off) + 1 * 32 + lane]) << 16); \
        float _a2 = __uint_as_float(static_cast<uint32_t>(sA[(kb_off) + 2 * 32 + lane]) << 16); \
        float _a3 = __uint_as_float(static_cast<uint32_t>(sA[(kb_off) + 3 * 32 + lane]) << 16); \
        acc = __fadd_rn(acc, __fmul_rn(_a0, bf16_reconstruct_f32((sm_word) & 0xFFu, _q0))); \
        acc = __fadd_rn(acc, __fmul_rn(_a1, bf16_reconstruct_f32(((sm_word) >> 8) & 0xFFu, _q1))); \
        acc = __fadd_rn(acc, __fmul_rn(_a2, bf16_reconstruct_f32(((sm_word) >> 16) & 0xFFu, _q2))); \
        acc = __fadd_rn(acc, __fmul_rn(_a3, bf16_reconstruct_f32(((sm_word) >> 24) & 0xFFu, _q3))); \
    } while (0)

    for (int nt = global_warp; nt < num_n8; nt += total_warps) {
        const uint4* Sm = sm_pack + (static_cast<uint64_t>(nt) * total_steps) * 32 + lane;
        int64_t c_start = tile_chunk_off[nt];
        int tile_chunks = static_cast<int>(tile_chunk_off[nt + 1] - c_start);
        const uint4* es4 = exp_stream_u4 + c_start * 32 + lane;

        uint64_t sb = static_cast<uint64_t>(nt) * 128;
        uint32_t st0 = init_states[sb + lane];
        uint32_t st1 = init_states[sb + 32 + lane];
        uint32_t st2 = init_states[sb + 64 + lane];
        uint32_t st3 = init_states[sb + 96 + lane];

        uint4 c0, c1, c_pref0, c_pref1;
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(c0.x), "=r"(c0.y), "=r"(c0.z), "=r"(c0.w) : "l"(es4));
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(c1.x), "=r"(c1.y), "=r"(c1.z), "=r"(c1.w) : "l"(es4 + 32));
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(c_pref0.x), "=r"(c_pref0.y), "=r"(c_pref0.z), "=r"(c_pref0.w) : "l"(es4 + 64));
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(c_pref1.x), "=r"(c_pref1.y), "=r"(c_pref1.z), "=r"(c_pref1.w) : "l"(es4 + 96));

        uint4 sm0, sm1, sm2, sm3;
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(sm0.x), "=r"(sm0.y), "=r"(sm0.z), "=r"(sm0.w) : "l"(Sm));
        if (1 < total_steps) {
            asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(sm1.x), "=r"(sm1.y), "=r"(sm1.z), "=r"(sm1.w) : "l"(Sm + 32));
        }
        if (2 < total_steps) {
            asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(sm2.x), "=r"(sm2.y), "=r"(sm2.z), "=r"(sm2.w) : "l"(Sm + 64));
        }
        if (3 < total_steps) {
            asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(sm3.x), "=r"(sm3.y), "=r"(sm3.z), "=r"(sm3.w) : "l"(Sm + 96));
        }

        wbuf4[lane] = c0;
        wbuf4[32 + lane] = c1;
        __syncwarp();

        int chunk_idx = 4;
        int pos = 0;
        int refill_at = 256;
        int step_idx = 0;

        for (int r = 0; r < 8; ++r) {
            float acc = 0.f;
            for (int ks = 0; ks < k_steps; ++ks, ++step_idx) {
                uint4 sm_new;
                if (step_idx + 4 < total_steps) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(sm_new.x), "=r"(sm_new.y), "=r"(sm_new.z), "=r"(sm_new.w)
                        : "l"(Sm + (step_idx + 4) * 32));
                }
                int kb = ks << 9;
                BF_ACC4(sm0.x, kb);
                BF_ACC4(sm0.y, kb + 128);
                BF_REFILL();
                BF_ACC4(sm0.z, kb + 256);
                BF_ACC4(sm0.w, kb + 384);
                BF_REFILL();
                sm0 = sm1;
                sm1 = sm2;
                sm2 = sm3;
                sm3 = sm_new;
            }
            // Exact binary tree reduction across 32 lanes (matches cuBLAS gemvx bit-for-bit!)
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, off));
            }
            if (lane == 0) {
                C[nt * 8 + r] = __float2bfloat16_rn(acc);
            }
        }
    }
    #undef BF_ACC4
    #undef BF_REFILL
    #undef BF_DEC_QUAD
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> repack_bf16_rans(
    const at::Tensor& B_rowmajor
) {
    TORCH_CHECK(B_rowmajor.is_cuda() && B_rowmajor.is_contiguous() && B_rowmajor.dtype() == at::kBFloat16);
    TORCH_CHECK(B_rowmajor.dim() == 2);
    int N = static_cast<int>(B_rowmajor.size(0));
    int K = static_cast<int>(B_rowmajor.size(1));
    TORCH_CHECK(N % 8 == 0 && K % 512 == 0);
    int num_n8 = N >> 3;
    int k_steps = K >> 9;
    auto dev_opts = B_rowmajor.options();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto g_hist = at::zeros({256}, dev_opts.dtype(at::kLong));
    int64_t num_u32 = (static_cast<int64_t>(N) * K) >> 1;
    int hist_blocks = static_cast<int>(std::min<int64_t>((num_u32 + 255) / 256, 2048));
    bf16_exp_hist_kernel<<<hist_blocks, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(B_rowmajor.data_ptr()),
        num_u32,
        reinterpret_cast<unsigned long long*>(g_hist.data_ptr<int64_t>())
    );
    auto h_hist_t = g_hist.to(at::kCPU);
    const int64_t* h_hist = h_hist_t.data_ptr<int64_t>();

    constexpr int M_PROB = 1024;
    int64_t total = 0;
    for (int i = 0; i < 256; ++i) total += h_hist[i];
    std::vector<uint32_t> freq(256, 0u);
    std::vector<double> exact(256, 0.0);
    std::vector<double> rem(256, -1.0);
    int sum_f = 0;
    for (int i = 0; i < 256; ++i) {
        if (h_hist[i] > 0) {
            double e = static_cast<double>(h_hist[i]) * static_cast<double>(M_PROB) / static_cast<double>(total);
            exact[i] = e;
            int f = static_cast<int>(std::floor(e));
            if (f < 1) f = 1;
            if (f > M_PROB - 1) f = M_PROB - 1;
            freq[i] = static_cast<uint32_t>(f);
            rem[i] = e - static_cast<double>(f);
            sum_f += f;
        }
    }
    while (sum_f < M_PROB) {
        int best = -1;
        double best_r = -1e9;
        for (int i = 0; i < 256; ++i) {
            if (h_hist[i] > 0 && freq[i] < static_cast<uint32_t>(M_PROB - 1) && rem[i] > best_r) {
                best_r = rem[i];
                best = i;
            }
        }
        if (best < 0) {
            for (int i = 0; i < 256; ++i) {
                if (freq[i] == 0u) { best = i; break; }
            }
        }
        freq[best] += 1u;
        rem[best] = -1e9;
        sum_f += 1;
    }
    while (sum_f > M_PROB) {
        int best = -1;
        double worst_excess = -1e9;
        for (int i = 0; i < 256; ++i) {
            if (freq[i] > 1u) {
                double ex = static_cast<double>(freq[i]) - exact[i];
                if (ex > worst_excess) {
                    worst_excess = ex;
                    best = i;
                }
            }
        }
        freq[best] -= 1u;
        sum_f -= 1;
    }

    std::vector<uint32_t> cdf(256, 0u);
    std::vector<uint32_t> lut(M_PROB, 0u);
    uint32_t acc = 0u;
    for (int s = 0; s < 256; ++s) {
        cdf[s] = acc;
        for (uint32_t j = 0; j < freq[s]; ++j) {
            lut[acc + j] = (static_cast<uint32_t>(s) << 24) | (freq[s] << 10) | j;
        }
        acc += freq[s];
    }

    auto d_freq = at::from_blob(freq.data(), {256}, at::kInt).to(B_rowmajor.device());
    auto d_cdf  = at::from_blob(cdf.data(), {256}, at::kInt).to(B_rowmajor.device());
    auto d_lut  = at::from_blob(lut.data(), {M_PROB}, at::kInt).to(B_rowmajor.device());

    auto sm_pack = at::empty({static_cast<int64_t>(num_n8) * 8 * k_steps * 32 * 4}, dev_opts.dtype(at::kInt));
    auto tile_u16 = at::empty({num_n8}, dev_opts.dtype(at::kInt));
    auto tile_chunks = at::empty({num_n8}, dev_opts.dtype(at::kLong));
    bf16_rans_measure_sm_kernel<<<num_n8, 32, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(B_rowmajor.data_ptr()),
        reinterpret_cast<uint4*>(sm_pack.data_ptr()),
        reinterpret_cast<const uint32_t*>(d_freq.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(d_cdf.data_ptr<int32_t>()),
        tile_u16.data_ptr<int32_t>(),
        tile_chunks.data_ptr<int64_t>(),
        N, K
    );

    auto tile_chunk_off = at::zeros({num_n8 + 1}, dev_opts.dtype(at::kLong));
    tile_chunk_off.narrow(0, 1, num_n8).copy_(at::cumsum(tile_chunks, 0));
    int64_t total_chunks = tile_chunk_off[num_n8].item<int64_t>();

    auto exp_stream = at::zeros({(total_chunks + 6) * 256}, dev_opts.dtype(at::kShort));
    auto init_states = at::empty({num_n8, 128}, dev_opts.dtype(at::kInt));

    bf16_rans_encode_kernel<<<num_n8, 32, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(B_rowmajor.data_ptr()),
        reinterpret_cast<const uint32_t*>(d_freq.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(d_cdf.data_ptr<int32_t>()),
        tile_u16.data_ptr<int32_t>(),
        tile_chunk_off.data_ptr<int64_t>(),
        reinterpret_cast<uint16_t*>(exp_stream.data_ptr<int16_t>()),
        reinterpret_cast<uint32_t*>(init_states.data_ptr<int32_t>()),
        N, K
    );

    return std::make_tuple(sm_pack, exp_stream, tile_chunk_off, init_states, d_lut);
}

at::Tensor bf16_rans_m1(
    const at::Tensor& A,
    const at::Tensor& sm_pack,
    const at::Tensor& exp_stream,
    const at::Tensor& tile_chunk_off,
    const at::Tensor& init_states,
    const at::Tensor& lut,
    int64_t N_in
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1);
    int N = static_cast<int>(N_in);
    int K = static_cast<int>(A.size(1));
    TORCH_CHECK(N % 8 == 0 && K % 512 == 0);
    auto C = at::empty({1, N}, A.options());
    int grid = 48;
    int blk = (N == 4608) ? 384 : 352;
    if (N != 4608 && N != 4096) {
        grid = 96;
        blk = 256;
    }
    int smem = K * 2 + 4096 + (blk >> 5) * 1024;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    static bool bf_attr_set = false;
    if (!bf_attr_set) {
        cudaFuncSetAttribute(bf16_rans_m1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        bf_attr_set = true;
    }
    bf16_rans_m1_kernel<<<grid, blk, smem, stream>>>(
        reinterpret_cast<const uint16_t*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(sm_pack.data_ptr()),
        reinterpret_cast<const uint4*>(exp_stream.data_ptr()),
        tile_chunk_off.data_ptr<int64_t>(),
        reinterpret_cast<const uint32_t*>(init_states.data_ptr()),
        reinterpret_cast<const uint32_t*>(lut.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        N, K
    );
    return C;
}

// ---------------------------------------------------------------------------
// 9. Tile-Contiguous M=512 Prefill Kernel Suite (NVFP4 + FP8, SM121)
//    - Pre-expanded BF16 B_s16 + cp.async.ca (sA) / cp.async.cg (sB) NVFP4 GEMMs
//    - Pre-swizzled [N/128, K/64, 512] uint4 FP8 GEMMs + fused gate_up_silu
//    - 0 stack, 0 spills, mism = 0 vs Marlin W4A16 and CUTLASS SM120
// ---------------------------------------------------------------------------
namespace m512_contig {

__device__ __forceinline__ __nv_bfloat162 dup_lo_bf16(uint32_t v) {
    uint32_t p;
    asm("prmt.b32 %0, %1, %1, 0x1010;" : "=r"(p) : "r"(v));
    return *reinterpret_cast<__nv_bfloat162*>(&p);
}

__device__ __forceinline__ __nv_bfloat162 dup_hi_bf16(uint32_t v) {
    uint32_t p;
    asm("prmt.b32 %0, %1, %1, 0x3232;" : "=r"(p) : "r"(v));
    return *reinterpret_cast<__nv_bfloat162*>(&p);
}

__device__ __forceinline__ void dequant_fp4_pair(
    uint32_t q, __nv_bfloat162 s0, __nv_bfloat162 s1,
    uint32_t& b00, uint32_t& b01, uint32_t& b10, uint32_t& b11
) {
    uint32_t f0 = lop3_or_and(q << 12, 0x80008000u, (q & 0x00070007u) << 6);
    uint32_t f1 = lop3_or_and(q << 8,  0x80008000u, (q & 0x00700070u) << 2);
    uint32_t f2 = lop3_or_and(q << 4,  0x80008000u, (q & 0x07000700u) >> 2);
    uint32_t f3 = lop3_or_and(q,       0x80008000u, (q & 0x70007000u) >> 6);
    b00 = hmul2_u32(f0, s0);
    b01 = hmul2_u32(f1, s0);
    b10 = hmul2_u32(f2, s1);
    b11 = hmul2_u32(f3, s1);
}

__device__ __forceinline__ void mma_m16n8k32_fp8_full(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

__device__ __forceinline__ void cp_async_cg16(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_ca16(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}

template <bool IS_PAIRED>
__global__ void repack_nvfp4_tile_contig_kernel(
    const uint4* __restrict__ B_q,
    const uint8_t* __restrict__ B_s,
    uint4* __restrict__ B_pack,
    int N, int K
) {
    int k_stages = K >> 6;
    int n_blk = blockIdx.x;
    int ks = blockIdx.y;
    int tid = threadIdx.x;
    int half_n64 = (N >> 1) >> 6;
    int row_stride_int4 = N >> 1;
    uint4* dst = B_pack + (static_cast<uint64_t>(n_blk) * k_stages + ks) * 320;

    for (int idx = tid; idx < 256; idx += blockDim.x) {
        int k16_local = idx >> 6;
        int col64 = idx & 63;
        int k16 = ks * 4 + k16_local;
        if constexpr (IS_PAIRED) {
            int is_up = col64 >> 5;
            int c32 = col64 & 31;
            dst[idx] = B_q[static_cast<uint64_t>(k16) * row_stride_int4 + (is_up * half_n64 + n_blk) * 32 + c32];
        } else {
            dst[idx] = B_q[static_cast<uint64_t>(k16) * row_stride_int4 + n_blk * 64 + col64];
        }
    }
    if (tid < 64) {
        int k16_local = tid >> 4;
        int c16 = tid & 15;
        int k16 = ks * 4 + k16_local;
        const uint8_t* p;
        if constexpr (IS_PAIRED) {
            int is_up = c16 >> 3;
            int c8 = c16 & 7;
            p = B_s + static_cast<uint64_t>(k16) * N + (is_up * half_n64 + n_blk) * 64 + c8 * 8;
        } else {
            p = B_s + static_cast<uint64_t>(k16) * N + n_blk * 128 + c16 * 8;
        }
        auto cvt = [](uint8_t b) -> uint32_t { return b ? ((static_cast<uint32_t>(b) << 4) + 0x3800u) : 0u; };
        uint4 out;
        out.x = cvt(p[0]) | (cvt(p[2]) << 16);
        out.y = cvt(p[1]) | (cvt(p[3]) << 16);
        out.z = cvt(p[4]) | (cvt(p[6]) << 16);
        out.w = cvt(p[5]) | (cvt(p[7]) << 16);
        dst[256 + tid] = out;
    }
}

template <bool OUT_FP8>
__global__ void __launch_bounds__(128, 2) nvfp4_gate_up_silu_contig_m512_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint4* __restrict__ B_pack,
    const float* __restrict__ g_scale,
    const float* __restrict__ silu_lut,
    const float* __restrict__ out_scale_a,
    void* __restrict__ C_raw,
    int M, int half_N, int K
) {
    constexpr int STAGES = 3;
    int m_tiles = M >> 6;
    int m_blk = blockIdx.x % m_tiles;
    int n_blk = blockIdx.x / m_tiles;
    int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31, group = lane >> 2, t = lane & 3;
    int w_k = warp_id >> 1, w_n = warp_id & 1;

    extern __shared__ char smem_raw[];
    uint4* sA = reinterpret_cast<uint4*>(smem_raw);
    uint4* sB = reinterpret_cast<uint4*>(smem_raw + STAGES * 8192);

    int k_stages = K >> 6, k_u4 = K >> 3;
    const uint4* A_base4 = reinterpret_cast<const uint4*>(A + static_cast<uint64_t>(m_blk * 64) * K);
    const uint4* B_tile = B_pack + (static_cast<uint64_t>(n_blk) * k_stages) * 320;

    auto issue_stage = [&](int ks, int st) {
        uint4* sA_st = sA + st * 512;
        int k_off_u4 = ks * 8;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 3, c8 = idx & 7;
            cp_async_ca16(sA_st + r * 8 + (c8 ^ (r & 7)),
                          A_base4 + static_cast<uint64_t>(r) * k_u4 + k_off_u4 + c8);
        }
        uint4* sB_st = sB + st * 320;
        const uint4* gB_st = B_tile + static_cast<uint64_t>(ks) * 320;
        cp_async_cg16(sB_st + tid,       gB_st + tid);
        cp_async_cg16(sB_st + tid + 128, gB_st + tid + 128);
        if (tid < 64) cp_async_cg16(sB_st + tid + 256, gB_st + tid + 256);
        asm volatile("cp.async.commit_group;\n");
    };

    #pragma unroll
    for (int st = 0; st < 2; ++st) {
        if (st < k_stages) issue_stage(st, st);
    }

    float d0[4][8] = {{0}}, d1[4][8] = {{0}}, d2[4][8] = {{0}}, d3[4][8] = {{0}};
    int lm_row = lane & 15, lm_col_hi = lane >> 4;
    if (k_stages >= 2) asm volatile("cp.async.wait_group 1;\n");
    else asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    for (int ks = 0; ks < k_stages; ++ks) {
        int st = ks % 3;
        const uint4* sA_st = sA + st * 512;
        const uint4* sBq_st = sB + st * 320;
        const uint4* sBs_st = sBq_st + 256;
        if (ks + 2 < k_stages) issue_stage(ks + 2, (ks + 2) % 3);

        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            int k16_local = w_k * 2 + sub;
            uint4 qv = sBq_st[k16_local * 64 + w_n * 32 + lane];
            uint4 sc4 = sBs_st[k16_local * 16 + w_n * 8 + group];
            uint32_t a0[4], a1[4], a2[4], a3[4];
            int lm_col = k16_local * 2 + lm_col_hi;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r = mi * 16 + lm_row;
                ldmatrix_x4(a0[mi], a1[mi], a2[mi], a3[mi], sA_st + r * 8 + (lm_col ^ (r & 7)));
            }
            uint32_t q_arr[4] = {qv.x, qv.y, qv.z, qv.w};
            uint32_t sp[4] = {sc4.x, sc4.y, sc4.z, sc4.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t b00, b01, b10, b11;
                dequant_fp4_pair(q_arr[j], dup_lo_bf16(sp[j]), dup_hi_bf16(sp[j]), b00, b01, b10, b11);
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    mma_m16n8k16_bf16_full(d0[mi][2 * j], d1[mi][2 * j], d2[mi][2 * j], d3[mi][2 * j],
                                           a0[mi], a1[mi], a2[mi], a3[mi], b00, b01);
                    mma_m16n8k16_bf16_full(d0[mi][2 * j + 1], d1[mi][2 * j + 1], d2[mi][2 * j + 1], d3[mi][2 * j + 1],
                                           a0[mi], a1[mi], a2[mi], a3[mi], b10, b11);
                }
            }
        }
        if (ks + 3 < k_stages) asm volatile("cp.async.wait_group 1;\n");
        else asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
    }

    float* s_red = reinterpret_cast<float*>(smem_raw);
    uint32_t* s_red_u32 = reinterpret_cast<uint32_t*>(smem_raw);
    float gs = *g_scale;
    if (w_k == 1) {
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            int r0 = mi * 16 + group, r1 = r0 + 8;
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                int c = m * 8 + t * 2;
                s_red[(w_n * 64 + r0) * 64 + c]     = d0[mi][m];
                s_red[(w_n * 64 + r0) * 64 + c + 1] = d1[mi][m];
                s_red[(w_n * 64 + r1) * 64 + c]     = d2[mi][m];
                s_red[(w_n * 64 + r1) * 64 + c + 1] = d3[mi][m];
            }
        }
    }
    __syncthreads();
    if (w_k == 0) {
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            int r0 = mi * 16 + group, r1 = r0 + 8;
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                int c = m * 8 + t * 2;
                int idx0 = (w_n * 64 + r0) * 64 + c;
                int idx1 = (w_n * 64 + r1) * 64 + c;
                __nv_bfloat16 b0 = __float2bfloat16_rn((d0[mi][m] + s_red[idx0]) * gs);
                __nv_bfloat16 b1 = __float2bfloat16_rn((d1[mi][m] + s_red[idx0 + 1]) * gs);
                __nv_bfloat16 b2 = __float2bfloat16_rn((d2[mi][m] + s_red[idx1]) * gs);
                __nv_bfloat16 b3 = __float2bfloat16_rn((d3[mi][m] + s_red[idx1 + 1]) * gs);
                if (w_n == 0) {
                    s_red_u32[idx0]     = static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&b0));
                    s_red_u32[idx0 + 1] = static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&b1));
                    s_red_u32[idx1]     = static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&b2));
                    s_red_u32[idx1 + 1] = static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&b3));
                } else {
                    s_red[idx0]     = __bfloat162float(b0);
                    s_red[idx0 + 1] = __bfloat162float(b1);
                    s_red[idx1]     = __bfloat162float(b2);
                    s_red[idx1 + 1] = __bfloat162float(b3);
                }
            }
        }
    }
    __syncthreads();
    if constexpr (OUT_FP8) {
        float inv_sa = __fdividef(1.0f, *out_scale_a);
        uint8_t* C_base = reinterpret_cast<uint8_t*>(C_raw) + static_cast<uint64_t>(m_blk * 64) * half_N + n_blk * 64;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 3, c8 = (idx & 7) << 3;
            uint8_t out8[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float sg = __ldg(silu_lut + s_red_u32[r * 64 + c8 + k]);
                float u = s_red[(64 + r) * 64 + c8 + k];
                float v = fminf(fmaxf((sg * u) * inv_sa, -448.0f), 448.0f);
                __nv_fp8_e4m3 q(v);
                out8[k] = *reinterpret_cast<uint8_t*>(&q);
            }
            *reinterpret_cast<uint2*>(C_base + static_cast<uint64_t>(r) * half_N + c8) =
                *reinterpret_cast<const uint2*>(out8);
        }
    } else {
        __nv_bfloat16* C_base = reinterpret_cast<__nv_bfloat16*>(C_raw) + static_cast<uint64_t>(m_blk * 64) * half_N + n_blk * 64;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 3, c8 = (idx & 7) << 3;
            __nv_bfloat16 out8[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float sg = __ldg(silu_lut + s_red_u32[r * 64 + c8 + k]);
                float u = s_red[(64 + r) * 64 + c8 + k];
                out8[k] = __float2bfloat16_rn(sg * u);
            }
            *reinterpret_cast<uint4*>(C_base + static_cast<uint64_t>(r) * half_N + c8) =
                *reinterpret_cast<const uint4*>(out8);
        }
    }
}

__global__ void __launch_bounds__(128, 2) nvfp4_bn128_tasklist_contig_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint4* __restrict__ B_pack,
    const float* __restrict__ g_scale,
    const TileTask* __restrict__ tasks,
    float* __restrict__ partials,
    __nv_bfloat16* __restrict__ C,
    int N, int K
) {
    constexpr int STAGES = 3;
    TileTask tk = tasks[blockIdx.x];
    int m_blk = tk.m_blk, n_blk = tk.n_blk, s0 = tk.s0, s1 = tk.s1, part_slot = tk.part_slot;
    int slen = s1 - s0;
    int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31, group = lane >> 2, t = lane & 3;
    int w_k = warp_id >> 1, w_n = warp_id & 1, n64 = n_blk * 2 + w_n;

    extern __shared__ char smem_raw[];
    uint4* sA = reinterpret_cast<uint4*>(smem_raw);
    uint4* sB = reinterpret_cast<uint4*>(smem_raw + STAGES * 8192);

    int k_stages = K >> 6, k_u4 = K >> 3;
    const uint4* A_base4 = reinterpret_cast<const uint4*>(A + static_cast<uint64_t>(m_blk * 64) * K);
    const uint4* B_tile = B_pack + (static_cast<uint64_t>(n_blk) * k_stages) * 320;

    auto issue_stage = [&](int ks, int st) {
        uint4* sA_st = sA + st * 512;
        int k_off = ks * 8;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 3, c8 = idx & 7;
            cp_async_ca16(sA_st + r * 8 + (c8 ^ (r & 7)),
                          A_base4 + static_cast<uint64_t>(r) * k_u4 + k_off + c8);
        }
        uint4* sB_st = sB + st * 320;
        const uint4* gB_st = B_tile + static_cast<uint64_t>(ks) * 320;
        cp_async_cg16(sB_st + tid,       gB_st + tid);
        cp_async_cg16(sB_st + tid + 128, gB_st + tid + 128);
        if (tid < 64) cp_async_cg16(sB_st + tid + 256, gB_st + tid + 256);
        asm volatile("cp.async.commit_group;\n");
    };

    #pragma unroll
    for (int st = 0; st < 2; ++st) {
        if (st < slen) issue_stage(s0 + st, st);
    }

    float d0[4][8] = {{0}}, d1[4][8] = {{0}}, d2[4][8] = {{0}}, d3[4][8] = {{0}};
    int lm_row = lane & 15, lm_col_hi = lane >> 4;
    if (slen >= 2) asm volatile("cp.async.wait_group 1;\n");
    else asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    for (int lk = 0; lk < slen; ++lk) {
        int st = lk % 3;
        const uint4* sA_st = sA + st * 512;
        const uint4* sBq_st = sB + st * 320;
        const uint4* sBs_st = sBq_st + 256;
        if (lk + 2 < slen) issue_stage(s0 + lk + 2, (lk + 2) % 3);

        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            int k16l = w_k * 2 + sub;
            uint4 qv = sBq_st[k16l * 64 + w_n * 32 + lane];
            uint4 sc4 = sBs_st[k16l * 16 + w_n * 8 + group];
            uint32_t a0[4], a1[4], a2[4], a3[4];
            int lm_col = k16l * 2 + lm_col_hi;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r = mi * 16 + lm_row;
                ldmatrix_x4(a0[mi], a1[mi], a2[mi], a3[mi], sA_st + r * 8 + (lm_col ^ (r & 7)));
            }
            uint32_t qa[4] = {qv.x, qv.y, qv.z, qv.w};
            uint32_t sp[4] = {sc4.x, sc4.y, sc4.z, sc4.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t b00, b01, b10, b11;
                dequant_fp4_pair(qa[j], dup_lo_bf16(sp[j]), dup_hi_bf16(sp[j]), b00, b01, b10, b11);
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    mma_m16n8k16_bf16_full(d0[mi][2 * j], d1[mi][2 * j], d2[mi][2 * j], d3[mi][2 * j],
                                           a0[mi], a1[mi], a2[mi], a3[mi], b00, b01);
                    mma_m16n8k16_bf16_full(d0[mi][2 * j + 1], d1[mi][2 * j + 1], d2[mi][2 * j + 1], d3[mi][2 * j + 1],
                                           a0[mi], a1[mi], a2[mi], a3[mi], b10, b11);
                }
            }
        }
        if (lk + 3 < slen) asm volatile("cp.async.wait_group 1;\n");
        else asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
    }

    float* s_red = reinterpret_cast<float*>(smem_raw);
    if (w_k == 1) {
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            int r0 = mi * 16 + group, r1 = r0 + 8;
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                int c = m * 8 + t * 2;
                s_red[(w_n * 64 + r0) * 64 + c]     = d0[mi][m];
                s_red[(w_n * 64 + r0) * 64 + c + 1] = d1[mi][m];
                s_red[(w_n * 64 + r1) * 64 + c]     = d2[mi][m];
                s_red[(w_n * 64 + r1) * 64 + c + 1] = d3[mi][m];
            }
        }
    }
    __syncthreads();
    if (w_k == 0) {
        if (part_slot < 0) {
            float gs = *g_scale;
            __nv_bfloat16* C_base = C + static_cast<uint64_t>(m_blk * 64) * N + n64 * 64;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r0 = mi * 16 + group, r1 = r0 + 8;
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    int c = m * 8 + t * 2;
                    *reinterpret_cast<__nv_bfloat162*>(C_base + static_cast<uint64_t>(r0) * N + c) =
                        __floats2bfloat162_rn((d0[mi][m] + s_red[(w_n * 64 + r0) * 64 + c]) * gs,
                                              (d1[mi][m] + s_red[(w_n * 64 + r0) * 64 + c + 1]) * gs);
                    *reinterpret_cast<__nv_bfloat162*>(C_base + static_cast<uint64_t>(r1) * N + c) =
                        __floats2bfloat162_rn((d2[mi][m] + s_red[(w_n * 64 + r1) * 64 + c]) * gs,
                                              (d3[mi][m] + s_red[(w_n * 64 + r1) * 64 + c + 1]) * gs);
                }
            }
        } else {
            float* P_base = partials + static_cast<uint64_t>(part_slot) * 8192 + w_n * 64;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r0 = mi * 16 + group, r1 = r0 + 8;
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    int c = m * 8 + t * 2;
                    P_base[r0 * 128 + c]     = d0[mi][m] + s_red[(w_n * 64 + r0) * 64 + c];
                    P_base[r0 * 128 + c + 1] = d1[mi][m] + s_red[(w_n * 64 + r0) * 64 + c + 1];
                    P_base[r1 * 128 + c]     = d2[mi][m] + s_red[(w_n * 64 + r1) * 64 + c];
                    P_base[r1 * 128 + c + 1] = d3[mi][m] + s_red[(w_n * 64 + r1) * 64 + c + 1];
                }
            }
        }
    }
}

__global__ void __launch_bounds__(128, 2) nvfp4_prefill_contig_bm64_bn128_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint4* __restrict__ B_pack,
    const float* __restrict__ g_scale,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    constexpr int STAGES = 3;
    int m_tiles = M >> 6;
    int m_blk = blockIdx.x % m_tiles;
    int n_blk = blockIdx.x / m_tiles;
    int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31, group = lane >> 2, t = lane & 3;
    int w_k = warp_id >> 1, w_n = warp_id & 1;

    extern __shared__ char smem_raw[];
    uint4* sA = reinterpret_cast<uint4*>(smem_raw);
    uint4* sB = reinterpret_cast<uint4*>(smem_raw + STAGES * 8192);

    int k_stages = K >> 6, k_u4 = K >> 3;
    const uint4* A_base4 = reinterpret_cast<const uint4*>(A + static_cast<uint64_t>(m_blk * 64) * K);
    const uint4* B_tile = B_pack + (static_cast<uint64_t>(n_blk) * k_stages) * 320;

    auto issue_stage = [&](int ks, int st) {
        uint4* sA_st = sA + st * 512;
        int k_off_u4 = ks * 8;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 3, c8 = idx & 7;
            cp_async_ca16(sA_st + r * 8 + (c8 ^ (r & 7)),
                          A_base4 + static_cast<uint64_t>(r) * k_u4 + k_off_u4 + c8);
        }
        uint4* sB_st = sB + st * 320;
        const uint4* gB_st = B_tile + static_cast<uint64_t>(ks) * 320;
        cp_async_cg16(sB_st + tid,       gB_st + tid);
        cp_async_cg16(sB_st + tid + 128, gB_st + tid + 128);
        if (tid < 64) cp_async_cg16(sB_st + tid + 256, gB_st + tid + 256);
        asm volatile("cp.async.commit_group;\n");
    };

    #pragma unroll
    for (int st = 0; st < 2; ++st) {
        if (st < k_stages) issue_stage(st, st);
    }

    float d0[4][8] = {{0}}, d1[4][8] = {{0}}, d2[4][8] = {{0}}, d3[4][8] = {{0}};
    int lm_row = lane & 15, lm_col_hi = lane >> 4;
    if (k_stages >= 2) asm volatile("cp.async.wait_group 1;\n");
    else asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    for (int ks = 0; ks < k_stages; ++ks) {
        int st = ks % 3;
        const uint4* sA_st = sA + st * 512;
        const uint4* sBq_st = sB + st * 320;
        const uint4* sBs_st = sBq_st + 256;
        if (ks + 2 < k_stages) issue_stage(ks + 2, (ks + 2) % 3);

        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            int k16_local = w_k * 2 + sub;
            uint4 qv = sBq_st[k16_local * 64 + w_n * 32 + lane];
            uint4 sc4 = sBs_st[k16_local * 16 + w_n * 8 + group];
            uint32_t a0[4], a1[4], a2[4], a3[4];
            int lm_col = k16_local * 2 + lm_col_hi;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r = mi * 16 + lm_row;
                ldmatrix_x4(a0[mi], a1[mi], a2[mi], a3[mi], sA_st + r * 8 + (lm_col ^ (r & 7)));
            }
            uint32_t q_arr[4] = {qv.x, qv.y, qv.z, qv.w};
            uint32_t sp[4] = {sc4.x, sc4.y, sc4.z, sc4.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t b00, b01, b10, b11;
                dequant_fp4_pair(q_arr[j], dup_lo_bf16(sp[j]), dup_hi_bf16(sp[j]), b00, b01, b10, b11);
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    mma_m16n8k16_bf16_full(d0[mi][2 * j], d1[mi][2 * j], d2[mi][2 * j], d3[mi][2 * j],
                                           a0[mi], a1[mi], a2[mi], a3[mi], b00, b01);
                    mma_m16n8k16_bf16_full(d0[mi][2 * j + 1], d1[mi][2 * j + 1], d2[mi][2 * j + 1], d3[mi][2 * j + 1],
                                           a0[mi], a1[mi], a2[mi], a3[mi], b10, b11);
                }
            }
        }
        if (ks + 3 < k_stages) asm volatile("cp.async.wait_group 1;\n");
        else asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
    }

    float* s_red = reinterpret_cast<float*>(smem_raw);
    float gs = *g_scale;
    if (w_k == 1) {
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            int r0 = mi * 16 + group, r1 = r0 + 8;
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                int c = m * 8 + t * 2;
                s_red[(w_n * 64 + r0) * 64 + c]     = d0[mi][m];
                s_red[(w_n * 64 + r0) * 64 + c + 1] = d1[mi][m];
                s_red[(w_n * 64 + r1) * 64 + c]     = d2[mi][m];
                s_red[(w_n * 64 + r1) * 64 + c + 1] = d3[mi][m];
            }
        }
    }
    __syncthreads();
    if (w_k == 0) {
        __nv_bfloat16* C_base = C + static_cast<uint64_t>(m_blk * 64) * N + n_blk * 128 + w_n * 64;
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            int r0 = mi * 16 + group, r1 = r0 + 8;
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                int c = m * 8 + t * 2;
                int idx0 = (w_n * 64 + r0) * 64 + c, idx1 = (w_n * 64 + r1) * 64 + c;
                *reinterpret_cast<__nv_bfloat162*>(C_base + static_cast<uint64_t>(r0) * N + c) =
                    __floats2bfloat162_rn((d0[mi][m] + s_red[idx0]) * gs, (d1[mi][m] + s_red[idx0 + 1]) * gs);
                *reinterpret_cast<__nv_bfloat162*>(C_base + static_cast<uint64_t>(r1) * N + c) =
                    __floats2bfloat162_rn((d2[mi][m] + s_red[idx1]) * gs, (d3[mi][m] + s_red[idx1 + 1]) * gs);
            }
        }
    }
}

template <bool IS_PAIRED>
__global__ void repack_fp8_tile_contig_kernel(
    const uint4* __restrict__ B,
    uint4* __restrict__ B_pack,
    int N, int K
) {
    int n_blk = blockIdx.x, ks = blockIdx.y;
    int k_stages = K >> 6, k_u4 = K >> 4, half_N = N >> 1;
    uint4* dst = B_pack + (static_cast<uint64_t>(n_blk) * k_stages + ks) * 512;
    for (int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        int r = idx >> 2, c4 = idx & 3;
        int swz = c4 ^ ((r >> 1) & 3);
        int src_row;
        if constexpr (IS_PAIRED) {
            src_row = (r < 64) ? (n_blk * 64 + r) : (half_N + n_blk * 64 + (r - 64));
        } else {
            src_row = n_blk * 128 + r;
        }
        dst[r * 4 + swz] = B[static_cast<uint64_t>(src_row) * k_u4 + ks * 4 + c4];
    }
}

template <bool SB_IS_VEC, bool OUT_FP8>
__global__ void __launch_bounds__(128, 2) fp8_gate_up_silu_contig_m512_kernel(
    const uint8_t* __restrict__ A,
    const uint4* __restrict__ B_pack,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    const float* __restrict__ silu_lut,
    const float* __restrict__ out_scale_a,
    void* __restrict__ C_raw,
    int M, int half_N, int K
) {
    constexpr int STAGES = 3;
    int m_tiles = M >> 7;
    int m_blk = blockIdx.x % m_tiles;
    int n_blk = blockIdx.x / m_tiles;
    int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31, group = lane >> 2, t = lane & 3;
    int w_m = warp_id >> 1, w_n = warp_id & 1;

    extern __shared__ char smem_raw[];
    uint4* sA = reinterpret_cast<uint4*>(smem_raw);
    uint4* sB = reinterpret_cast<uint4*>(smem_raw + STAGES * 8192);

    int k_stages = K >> 6, k_u4 = K >> 4;
    const uint4* A_base4 = reinterpret_cast<const uint4*>(A + static_cast<uint64_t>(m_blk * 128) * K);
    const uint4* B_tile = B_pack + (static_cast<uint64_t>(n_blk) * k_stages) * 512;

    auto issue_stage = [&](int ks, int st) {
        uint4* sA_st = sA + st * 512;
        uint4* sB_st = sB + st * 512;
        int koff = ks * 4;
        const uint4* gB_st = B_tile + static_cast<uint64_t>(ks) * 512;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 2, c4 = idx & 3;
            cp_async_cg16(sA_st + r * 4 + (c4 ^ ((r >> 1) & 3)),
                          A_base4 + static_cast<uint64_t>(r) * k_u4 + koff + c4);
            cp_async_cg16(sB_st + idx, gB_st + idx);
        }
        asm volatile("cp.async.commit_group;\n");
    };

    #pragma unroll
    for (int st = 0; st < 2; ++st) {
        if (st < k_stages) issue_stage(st, st);
    }

    float d0[4][8] = {{0}}, d1[4][8] = {{0}}, d2[4][8] = {{0}}, d3[4][8] = {{0}};
    int lm_row = lane & 15, lm_col_hi = lane >> 4;
    if (k_stages >= 2) asm volatile("cp.async.wait_group 1;\n");
    else asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    for (int ks = 0; ks < k_stages; ++ks) {
        int st = ks % 3;
        const uint4* sA_st = sA + st * 512;
        const uint4* sB_st = sB + st * 512;
        if (ks + 2 < k_stages) issue_stage(ks + 2, (ks + 2) % 3);

        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t a0[4], a1[4], a2[4], a3[4];
            int c_idx = kk * 2 + lm_col_hi;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r = w_m * 64 + mi * 16 + lm_row;
                ldmatrix_x4(a0[mi], a1[mi], a2[mi], a3[mi], sA_st + r * 4 + (c_idx ^ ((r >> 1) & 3)));
            }
            uint32_t b0, b1, b2, b3;
            {
                int r = w_n * 64 + lm_row;
                ldmatrix_x4(b0, b1, b2, b3, sB_st + r * 4 + (c_idx ^ ((r >> 1) & 3)));
            }
            #pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
                uint32_t nb0 = 0, nb1 = 0, nb2 = 0, nb3 = 0;
                if (nj + 1 < 4) {
                    int nr = w_n * 64 + (nj + 1) * 16 + lm_row;
                    ldmatrix_x4(nb0, nb1, nb2, nb3, sB_st + nr * 4 + (c_idx ^ ((nr >> 1) & 3)));
                }
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    mma_m16n8k32_fp8_full(d0[mi][2 * nj], d1[mi][2 * nj], d2[mi][2 * nj], d3[mi][2 * nj],
                                          a0[mi], a1[mi], a2[mi], a3[mi], b0, b2);
                    mma_m16n8k32_fp8_full(d0[mi][2 * nj + 1], d1[mi][2 * nj + 1], d2[mi][2 * nj + 1], d3[mi][2 * nj + 1],
                                          a0[mi], a1[mi], a2[mi], a3[mi], b1, b3);
                }
                b0 = nb0; b1 = nb1; b2 = nb2; b3 = nb3;
            }
        }
        if (ks + 3 < k_stages) asm volatile("cp.async.wait_group 1;\n");
        else asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
    }

    uint16_t* sG = reinterpret_cast<uint16_t*>(smem_raw);
    __nv_bfloat16* sU = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    float sa = *scale_a;
    float sb_scalar = SB_IS_VEC ? 1.f : (*scale_b);
    int col_base = w_n * half_N + n_blk * 64;

    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        int r0 = w_m * 64 + mi * 16 + group, r1 = r0 + 8;
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            int c = m * 8 + t * 2;
            float sb0 = SB_IS_VEC ? scale_b[col_base + c]     : sb_scalar;
            float sb1 = SB_IS_VEC ? scale_b[col_base + c + 1] : sb_scalar;
            __nv_bfloat16 b0 = __float2bfloat16_rn((d0[mi][m] * sb0) * sa);
            __nv_bfloat16 b1 = __float2bfloat16_rn((d1[mi][m] * sb1) * sa);
            __nv_bfloat16 b2 = __float2bfloat16_rn((d2[mi][m] * sb0) * sa);
            __nv_bfloat16 b3 = __float2bfloat16_rn((d3[mi][m] * sb1) * sa);
            if (w_n == 0) {
                sG[r0 * 64 + c]     = *reinterpret_cast<uint16_t*>(&b0);
                sG[r0 * 64 + c + 1] = *reinterpret_cast<uint16_t*>(&b1);
                sG[r1 * 64 + c]     = *reinterpret_cast<uint16_t*>(&b2);
                sG[r1 * 64 + c + 1] = *reinterpret_cast<uint16_t*>(&b3);
            } else {
                sU[r0 * 64 + c]     = b0;
                sU[r0 * 64 + c + 1] = b1;
                sU[r1 * 64 + c]     = b2;
                sU[r1 * 64 + c + 1] = b3;
            }
        }
    }
    __syncthreads();

    if constexpr (OUT_FP8) {
        float inv_sa = __fdividef(1.0f, *out_scale_a);
        uint8_t* C_base = reinterpret_cast<uint8_t*>(C_raw) + static_cast<uint64_t>(m_blk * 128) * half_N + n_blk * 64;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 3, c8 = (idx & 7) << 3;
            uint8_t out8[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float sg = __ldg(silu_lut + sG[r * 64 + c8 + k]);
                float u = __bfloat162float(sU[r * 64 + c8 + k]);
                float v = fminf(fmaxf((sg * u) * inv_sa, -448.0f), 448.0f);
                __nv_fp8_e4m3 q(v);
                out8[k] = *reinterpret_cast<uint8_t*>(&q);
            }
            *reinterpret_cast<uint2*>(C_base + static_cast<uint64_t>(r) * half_N + c8) =
                *reinterpret_cast<const uint2*>(out8);
        }
    } else {
        __nv_bfloat16* C_base = reinterpret_cast<__nv_bfloat16*>(C_raw) + static_cast<uint64_t>(m_blk * 128) * half_N + n_blk * 64;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 3, c8 = (idx & 7) << 3;
            __nv_bfloat16 out8[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float sg = __ldg(silu_lut + sG[r * 64 + c8 + k]);
                float u = __bfloat162float(sU[r * 64 + c8 + k]);
                out8[k] = __float2bfloat16_rn(sg * u);
            }
            *reinterpret_cast<uint4*>(C_base + static_cast<uint64_t>(r) * half_N + c8) =
                *reinterpret_cast<const uint4*>(out8);
        }
    }
}

template <bool SB_IS_VEC>
__global__ void __launch_bounds__(128, 2) fp8_linear_contig_m512_kernel(
    const uint8_t* __restrict__ A,
    const uint4* __restrict__ B_pack,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    constexpr int STAGES = 3;
    int m_tiles = M >> 7;
    int m_blk = blockIdx.x % m_tiles;
    int n_blk = blockIdx.x / m_tiles;
    int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31, group = lane >> 2, t = lane & 3;
    int w_m = warp_id >> 1, w_n = warp_id & 1;

    extern __shared__ char smem_raw[];
    uint4* sA = reinterpret_cast<uint4*>(smem_raw);
    uint4* sB = reinterpret_cast<uint4*>(smem_raw + STAGES * 8192);

    int k_stages = K >> 6, k_u4 = K >> 4;
    const uint4* A_base4 = reinterpret_cast<const uint4*>(A + static_cast<uint64_t>(m_blk * 128) * K);
    const uint4* B_tile = B_pack + (static_cast<uint64_t>(n_blk) * k_stages) * 512;

    auto issue_stage = [&](int ks, int st) {
        uint4* sA_st = sA + st * 512;
        uint4* sB_st = sB + st * 512;
        int koff = ks * 4;
        const uint4* gB_st = B_tile + static_cast<uint64_t>(ks) * 512;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 2, c4 = idx & 3;
            cp_async_cg16(sA_st + r * 4 + (c4 ^ ((r >> 1) & 3)),
                          A_base4 + static_cast<uint64_t>(r) * k_u4 + koff + c4);
            cp_async_cg16(sB_st + idx, gB_st + idx);
        }
        asm volatile("cp.async.commit_group;\n");
    };

    #pragma unroll
    for (int st = 0; st < 2; ++st) {
        if (st < k_stages) issue_stage(st, st);
    }

    float d0[4][8] = {{0}}, d1[4][8] = {{0}}, d2[4][8] = {{0}}, d3[4][8] = {{0}};
    int lm_row = lane & 15, lm_col_hi = lane >> 4;
    if (k_stages >= 2) asm volatile("cp.async.wait_group 1;\n");
    else asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    for (int ks = 0; ks < k_stages; ++ks) {
        int st = ks % 3;
        const uint4* sA_st = sA + st * 512;
        const uint4* sB_st = sB + st * 512;
        if (ks + 2 < k_stages) issue_stage(ks + 2, (ks + 2) % 3);

        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t a0[4], a1[4], a2[4], a3[4];
            int c_idx = kk * 2 + lm_col_hi;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r = w_m * 64 + mi * 16 + lm_row;
                ldmatrix_x4(a0[mi], a1[mi], a2[mi], a3[mi], sA_st + r * 4 + (c_idx ^ ((r >> 1) & 3)));
            }
            uint32_t b0, b1, b2, b3;
            {
                int r = w_n * 64 + lm_row;
                ldmatrix_x4(b0, b1, b2, b3, sB_st + r * 4 + (c_idx ^ ((r >> 1) & 3)));
            }
            #pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
                uint32_t nb0 = 0, nb1 = 0, nb2 = 0, nb3 = 0;
                if (nj + 1 < 4) {
                    int nr = w_n * 64 + (nj + 1) * 16 + lm_row;
                    ldmatrix_x4(nb0, nb1, nb2, nb3, sB_st + nr * 4 + (c_idx ^ ((nr >> 1) & 3)));
                }
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    mma_m16n8k32_fp8_full(d0[mi][2 * nj], d1[mi][2 * nj], d2[mi][2 * nj], d3[mi][2 * nj],
                                          a0[mi], a1[mi], a2[mi], a3[mi], b0, b2);
                    mma_m16n8k32_fp8_full(d0[mi][2 * nj + 1], d1[mi][2 * nj + 1], d2[mi][2 * nj + 1], d3[mi][2 * nj + 1],
                                          a0[mi], a1[mi], a2[mi], a3[mi], b1, b3);
                }
                b0 = nb0; b1 = nb1; b2 = nb2; b3 = nb3;
            }
        }
        if (ks + 3 < k_stages) asm volatile("cp.async.wait_group 1;\n");
        else asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
    }

    float sa = *scale_a;
    float sb_scalar = SB_IS_VEC ? 1.f : (*scale_b);
    __nv_bfloat16* C_base = C + static_cast<uint64_t>(m_blk * 128 + w_m * 64) * N + n_blk * 128 + w_n * 64;
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        int r0 = mi * 16 + group, r1 = r0 + 8;
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            int c = m * 8 + t * 2;
            float sb0 = SB_IS_VEC ? scale_b[n_blk * 128 + w_n * 64 + c]     : sb_scalar;
            float sb1 = SB_IS_VEC ? scale_b[n_blk * 128 + w_n * 64 + c + 1] : sb_scalar;
            *reinterpret_cast<__nv_bfloat162*>(C_base + static_cast<uint64_t>(r0) * N + c) =
                __floats2bfloat162_rn((d0[mi][m] * sb0) * sa, (d1[mi][m] * sb1) * sa);
            *reinterpret_cast<__nv_bfloat162*>(C_base + static_cast<uint64_t>(r1) * N + c) =
                __floats2bfloat162_rn((d2[mi][m] * sb0) * sa, (d3[mi][m] * sb1) * sa);
        }
    }
}

}  // namespace m512_contig

__global__ void silu_mul_fp8quant_mode0_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    const float* __restrict__ silu_lut,
    const float* __restrict__ out_scale_a,
    uint8_t* __restrict__ out,
    int M, int half_N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * half_N;
    if (idx >= total) return;
    int r = idx / half_N;
    int c = idx - r * half_N;
    const __nv_bfloat16* row = gate_up + static_cast<int64_t>(r) * (half_N * 2);
    uint16_t gb = *reinterpret_cast<const uint16_t*>(&row[c]);
    float sg = __ldg(silu_lut + gb);
    float u = __bfloat162float(row[half_N + c]);
    float v = fminf(fmaxf((sg * u) * __fdividef(1.0f, *out_scale_a), -448.0f), 448.0f);
    __nv_fp8_e4m3 q(v);
    out[idx] = *reinterpret_cast<uint8_t*>(&q);
}

at::Tensor silu_mul_fp8quant_mode0(const at::Tensor& gate_up, const at::Tensor& out_scale_a) {
    TORCH_CHECK(gate_up.is_cuda() && gate_up.is_contiguous() && gate_up.dtype() == at::kBFloat16);
    int M = static_cast<int>(gate_up.size(0));
    int half_N = static_cast<int>(gate_up.size(1) / 2);
    auto C = at::empty({M, half_N}, gate_up.options().dtype(at::kFloat8_e4m3fn));
    int total = M * half_N;
    const float* lut = get_silu_lut_ptr(gate_up.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    silu_mul_fp8quant_mode0_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr()),
        lut,
        out_scale_a.data_ptr<float>(),
        reinterpret_cast<uint8_t*>(C.data_ptr()),
        M, half_N
    );
    return C;
}

at::Tensor repack_nvfp4_m512_contig(
    const at::Tensor& B_q,
    const at::Tensor& B_s,
    int64_t N_in,
    int64_t K_in,
    bool is_paired
) {
    int N = static_cast<int>(N_in);
    int K = static_cast<int>(K_in);
    TORCH_CHECK(B_q.is_cuda() && B_s.is_cuda());
    TORCH_CHECK(N % 128 == 0 && K % 64 == 0);
    auto B_pack = at::empty(
        {static_cast<int64_t>(N / 128) * (K / 64) * 320 * 4},
        B_q.options().dtype(at::kInt)
    );
    (void)get_silu_lut_ptr(B_q.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    dim3 grid(N / 128, K / 64);
    if (is_paired) {
        m512_contig::repack_nvfp4_tile_contig_kernel<true><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const uint4*>(B_q.data_ptr()),
            reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
            reinterpret_cast<uint4*>(B_pack.data_ptr()),
            N, K
        );
    } else {
        m512_contig::repack_nvfp4_tile_contig_kernel<false><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const uint4*>(B_q.data_ptr()),
            reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
            reinterpret_cast<uint4*>(B_pack.data_ptr()),
            N, K
        );
    }
    return B_pack;
}

at::Tensor nvfp4_gate_up_silu_contig_m512(
    const at::Tensor& A,
    const at::Tensor& B_pack,
    const at::Tensor& g_scale,
    int64_t half_N
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int hn = static_cast<int>(half_N);
    TORCH_CHECK(M % 64 == 0 && hn % 64 == 0 && K % 64 == 0);
    auto C = at::empty({M, hn}, A.options());
    int grid = (M / 64) * (hn / 64);
    const float* lut = get_silu_lut_ptr(A.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    m512_contig::nvfp4_gate_up_silu_contig_m512_kernel<false><<<grid, 128, 39936, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(B_pack.data_ptr()),
        g_scale.data_ptr<float>(),
        lut,
        nullptr,
        C.data_ptr(),
        M, hn, K
    );
    return C;
}

at::Tensor nvfp4_gate_up_silu_fp8quant_contig_m512(
    const at::Tensor& A,
    const at::Tensor& B_pack,
    const at::Tensor& g_scale,
    const at::Tensor& out_scale_a,
    int64_t half_N
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int hn = static_cast<int>(half_N);
    TORCH_CHECK(M % 64 == 0 && hn % 64 == 0 && K % 64 == 0);
    auto C = at::empty({M, hn}, A.options().dtype(at::kFloat8_e4m3fn));
    int grid = (M / 64) * (hn / 64);
    const float* lut = get_silu_lut_ptr(A.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    m512_contig::nvfp4_gate_up_silu_contig_m512_kernel<true><<<grid, 128, 39936, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(B_pack.data_ptr()),
        g_scale.data_ptr<float>(),
        lut,
        out_scale_a.data_ptr<float>(),
        C.data_ptr(),
        M, hn, K
    );
    return C;
}

at::Tensor nvfp4_lpt_contig_m512(
    const at::Tensor& A,
    const at::Tensor& B_pack,
    const at::Tensor& g_scale,
    const at::Tensor& task_list,
    const at::Tensor& reduce_list,
    at::Tensor partials,
    int64_t N_in
) {
    int N = static_cast<int>(N_in);
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    auto C = at::empty({M, N}, A.options());
    int n_tasks = static_cast<int>(task_list.size(0) / 6);
    int n_rtasks = static_cast<int>(reduce_list.size(0) / 4);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    m512_contig::nvfp4_bn128_tasklist_contig_kernel<<<n_tasks, 128, 39936, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(B_pack.data_ptr()),
        g_scale.data_ptr<float>(),
        reinterpret_cast<const TileTask*>(task_list.data_ptr()),
        partials.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        N, K
    );
    if (n_rtasks > 0) {
        nvfp4_reduce_tasklist_kernel<<<n_rtasks, 256, 0, stream>>>(
            reinterpret_cast<const ReduceTask*>(reduce_list.data_ptr()),
            partials.data_ptr<float>(),
            g_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            N
        );
    }
    return C;
}

at::Tensor nvfp4_prefill_contig_bm64_bn128(
    const at::Tensor& A,
    const at::Tensor& B_pack,
    const at::Tensor& g_scale,
    int64_t N_in
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int n_int = static_cast<int>(N_in);
    TORCH_CHECK(M % 64 == 0 && n_int % 128 == 0 && K % 64 == 0);
    auto C = at::empty({M, n_int}, A.options());
    int grid = (M / 64) * (n_int / 128);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    m512_contig::nvfp4_prefill_contig_bm64_bn128_kernel<<<grid, 128, 39936, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(B_pack.data_ptr()),
        g_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, n_int, K
    );
    return C;
}

at::Tensor repack_fp8_m512_contig(const at::Tensor& B_colmajor, bool is_paired) {
    TORCH_CHECK(B_colmajor.is_cuda() && B_colmajor.dim() == 2);
    TORCH_CHECK(B_colmajor.stride(0) == 1, "Expected column-major [K, N] FP8 weight");
    int K = static_cast<int>(B_colmajor.size(0));
    int N = static_cast<int>(B_colmajor.size(1));
    TORCH_CHECK(N % 128 == 0 && K % 64 == 0);
    auto B_pack = at::empty(
        {static_cast<int64_t>(N / 128) * (K / 64) * 512 * 4},
        B_colmajor.options().dtype(at::kInt)
    );
    (void)get_silu_lut_ptr(B_colmajor.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    dim3 grid(N / 128, K / 64);
    if (is_paired) {
        m512_contig::repack_fp8_tile_contig_kernel<true><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const uint4*>(B_colmajor.data_ptr()),
            reinterpret_cast<uint4*>(B_pack.data_ptr()),
            N, K
        );
    } else {
        m512_contig::repack_fp8_tile_contig_kernel<false><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const uint4*>(B_colmajor.data_ptr()),
            reinterpret_cast<uint4*>(B_pack.data_ptr()),
            N, K
        );
    }
    return B_pack;
}

static void ensure_fp8_m512_smem_attrs() {
    static bool done = false;
    if (!done) {
        cudaFuncSetAttribute(m512_contig::fp8_gate_up_silu_contig_m512_kernel<false, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
        cudaFuncSetAttribute(m512_contig::fp8_gate_up_silu_contig_m512_kernel<true, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
        cudaFuncSetAttribute(m512_contig::fp8_gate_up_silu_contig_m512_kernel<false, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
        cudaFuncSetAttribute(m512_contig::fp8_gate_up_silu_contig_m512_kernel<true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
        cudaFuncSetAttribute(m512_contig::fp8_linear_contig_m512_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
        cudaFuncSetAttribute(m512_contig::fp8_linear_contig_m512_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
        done = true;
    }
}

at::Tensor fp8_gate_up_silu_contig_m512(
    const at::Tensor& A,
    const at::Tensor& B_pack,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    int64_t half_N
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kFloat8_e4m3fn);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int hn = static_cast<int>(half_N);
    TORCH_CHECK(M % 128 == 0 && hn % 64 == 0 && K % 64 == 0);
    ensure_fp8_m512_smem_attrs();
    auto C = at::empty({M, hn}, A.options().dtype(at::kBFloat16));
    int grid = (M / 128) * (hn / 64);
    const float* lut = get_silu_lut_ptr(A.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (scale_b.numel() > 1) {
        m512_contig::fp8_gate_up_silu_contig_m512_kernel<true, false><<<grid, 128, 49152, stream>>>(
            reinterpret_cast<const uint8_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(B_pack.data_ptr()),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            lut,
            nullptr,
            C.data_ptr(),
            M, hn, K
        );
    } else {
        m512_contig::fp8_gate_up_silu_contig_m512_kernel<false, false><<<grid, 128, 49152, stream>>>(
            reinterpret_cast<const uint8_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(B_pack.data_ptr()),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            lut,
            nullptr,
            C.data_ptr(),
            M, hn, K
        );
    }
    return C;
}

at::Tensor fp8_gate_up_silu_fp8quant_contig_m512(
    const at::Tensor& A,
    const at::Tensor& B_pack,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const at::Tensor& out_scale_a,
    int64_t half_N
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kFloat8_e4m3fn);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int hn = static_cast<int>(half_N);
    TORCH_CHECK(M % 128 == 0 && hn % 64 == 0 && K % 64 == 0);
    ensure_fp8_m512_smem_attrs();
    auto C = at::empty({M, hn}, A.options());
    int grid = (M / 128) * (hn / 64);
    const float* lut = get_silu_lut_ptr(A.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (scale_b.numel() > 1) {
        m512_contig::fp8_gate_up_silu_contig_m512_kernel<true, true><<<grid, 128, 49152, stream>>>(
            reinterpret_cast<const uint8_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(B_pack.data_ptr()),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            lut,
            out_scale_a.data_ptr<float>(),
            C.data_ptr(),
            M, hn, K
        );
    } else {
        m512_contig::fp8_gate_up_silu_contig_m512_kernel<false, true><<<grid, 128, 49152, stream>>>(
            reinterpret_cast<const uint8_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(B_pack.data_ptr()),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            lut,
            out_scale_a.data_ptr<float>(),
            C.data_ptr(),
            M, hn, K
        );
    }
    return C;
}

at::Tensor fp8_linear_contig_m512(
    const at::Tensor& A,
    const at::Tensor& B_pack,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    int64_t N_in
) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kFloat8_e4m3fn);
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(N_in);
    TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 64 == 0);
    ensure_fp8_m512_smem_attrs();
    auto C = at::empty({M, N}, A.options().dtype(at::kBFloat16));
    int grid = (M / 128) * (N / 128);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (scale_b.numel() > 1) {
        m512_contig::fp8_linear_contig_m512_kernel<true><<<grid, 128, 49152, stream>>>(
            reinterpret_cast<const uint8_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(B_pack.data_ptr()),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            M, N, K
        );
    } else {
        m512_contig::fp8_linear_contig_m512_kernel<false><<<grid, 128, 49152, stream>>>(
            reinterpret_cast<const uint8_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(B_pack.data_ptr()),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            M, N, K
        );
    }
    return C;
}

}  // namespace

TORCH_LIBRARY(muse_fast, m) {
    m.def("nvfp4_down_m1(Tensor A, Tensor B_q, Tensor B_s, Tensor g_scale, Tensor(a!) partials) -> Tensor");
    m.def("nvfp4_gate_up_silu_m1(Tensor A, Tensor B_q, Tensor B_s, Tensor g_scale) -> Tensor");
    m.def("silu_mul_mode0(Tensor gate_up) -> Tensor");
    m.def("nvfp4_lmhead_m1(Tensor A, Tensor B_q, Tensor B_s, Tensor g_scale) -> Tensor");
    m.def("nvfp4_prefill_bm64_bn128(Tensor A, Tensor B_q, Tensor B_s, Tensor g_scale, int N) -> Tensor");
    m.def("nvfp4_gate_up_silu_m512(Tensor A, Tensor B_q, Tensor B_s, Tensor g_scale, int half_N) -> Tensor");
    m.def("nvfp4_gate_up_silu_fp8quant_m512(Tensor A, Tensor B_q, Tensor B_s, Tensor g_scale, Tensor out_scale_a, int half_N) -> Tensor");
    m.def("nvfp4_down_m512_lpt(Tensor A, Tensor B_q, Tensor B_s, Tensor g_scale, Tensor task_list, Tensor reduce_list, Tensor(a!) partials, int N) -> Tensor");
    m.def("repack_fp8_mma(Tensor B_colmajor) -> Tensor");
    m.def("fp8_mma_repacked_m1(Tensor A, Tensor B_pack, Tensor scale_a, Tensor scale_b, int N) -> Tensor");
    m.def("repack_fp8_rans(Tensor B_colmajor) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("fp8_rans_m1(Tensor A, Tensor sm_pack, Tensor exp_stream, Tensor tile_chunk_off, Tensor init_states, Tensor lut, Tensor scale_a, Tensor scale_b, int N) -> Tensor");
    m.def("init_decode_loop(int first_token, int start_pos, int count, Tensor(a!) step_idx, Tensor(b!) step_limit, Tensor(c!) input_ids, Tensor(d!) positions, Tensor(e!) seq_lens, Tensor(f!) query_start_loc, Tensor input_block_table_ptrs, Tensor block_sizes, Tensor(g!) slot_mappings) -> ()");
    m.def("greedy_argmax_step_update(Tensor logits, Tensor(a!) out_tokens, Tensor(b!) step_idx, Tensor step_limit, Tensor(c!) input_ids, Tensor(d!) positions, Tensor(e!) seq_lens, Tensor input_block_table_ptrs, Tensor block_sizes, Tensor(f!) slot_mappings) -> ()");
    m.def("nvfp4_lmhead_m1_argmax_step_update(Tensor A, Tensor B_q, Tensor B_s, Tensor g_scale, float output_multiplier, float softcap, Tensor(a!) blk_max_val, Tensor(b!) blk_max_idx, Tensor(c!) out_tokens, Tensor(d!) step_idx, Tensor step_limit, Tensor(e!) input_ids, Tensor(f!) positions, Tensor(g!) seq_lens, Tensor input_block_table_ptrs, Tensor block_sizes, Tensor(h!) slot_mappings) -> ()");
    m.def("repack_nvfp4_m1(Tensor B_q, Tensor B_s, int N, int K) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("nvfp4_tm_linear_m1(Tensor A, Tensor Bq_tm, Tensor Bs_tm, Tensor B_base, Tensor tile_s_off, Tensor tile_s_mode, Tensor g_scale, int N, int n_grp, int p1_cols, int p2_iters, int grid, int mode_mask) -> Tensor");
    m.def("nvfp4_tm_gate_up_silu_m1(Tensor A, Tensor Bq_tm, Tensor Bs_tm, Tensor B_base, Tensor tile_s_off, Tensor tile_s_mode, Tensor g_scale, int mode_mask) -> Tensor");
    m.def("nvfp4_tm_lmhead_m1_argmax_step_update(Tensor A, Tensor Bq_tm, Tensor Bs_tm, Tensor B_base, Tensor tile_s_off, Tensor tile_s_mode, Tensor g_scale, float output_multiplier, float softcap, Tensor(a!) blk_max_val, Tensor(b!) blk_max_idx, Tensor(c!) out_tokens, Tensor(d!) step_idx, Tensor step_limit, Tensor(e!) input_ids, Tensor(f!) positions, Tensor(g!) seq_lens, Tensor input_block_table_ptrs, Tensor block_sizes, Tensor(h!) slot_mappings, int mode_mask) -> ()");
    m.def("repack_nvfp4_rans(Tensor B_q, Tensor B_s, int N, int K, int n_grp, int p1_cols, int p2_iters) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("nvfp4_rans_linear_m1(Tensor A, Tensor Bq_tm, Tensor sc_stream, Tensor stream_chunk_off, Tensor init_states, Tensor lut, Tensor g_scale, int N, int n_grp, int p1_cols, int p2_iters, int grid) -> Tensor");
    m.def("repack_nvfp4_rans_splitk96(Tensor B_q, Tensor B_s, int N, int K, int iters) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("nvfp4_rans_splitk96_m1(Tensor A, Tensor Bq_tm, Tensor sc_stream, Tensor stream_chunk_off, Tensor init_states, Tensor lut, Tensor g_scale, Tensor(a!) partials, int N, int iters) -> Tensor");
    m.def("nvfp4_rans_gate_up_silu_m1(Tensor A, Tensor Bq_tm, Tensor sc_stream, Tensor stream_chunk_off, Tensor init_states, Tensor lut, Tensor g_scale) -> Tensor");
    m.def("nvfp4_rans_gate_up_silu_fp8quant_m1(Tensor A, Tensor Bq_tm, Tensor sc_stream, Tensor stream_chunk_off, Tensor init_states, Tensor lut, Tensor g_scale, Tensor out_scale_a) -> Tensor");
    m.def("nvfp4_rans_lmhead_m1_argmax_step_update(Tensor A, Tensor Bq_tm, Tensor sc_stream, Tensor stream_chunk_off, Tensor init_states, Tensor lut, Tensor g_scale, float output_multiplier, float softcap, Tensor(a!) blk_max_val, Tensor(b!) blk_max_idx, Tensor(c!) out_tokens, Tensor(d!) step_idx, Tensor step_limit, Tensor(e!) input_ids, Tensor(f!) positions, Tensor(g!) seq_lens, Tensor input_block_table_ptrs, Tensor block_sizes, Tensor(h!) slot_mappings) -> ()");
    m.def("nvfp4_rans_lmhead_m1_argmax_only(Tensor A, Tensor Bq_tm, Tensor sc_stream, Tensor stream_chunk_off, Tensor init_states, Tensor lut, Tensor g_scale, float output_multiplier, float softcap, Tensor(a!) blk_max_val, Tensor(b!) blk_max_idx, Tensor(c!) out_token) -> ()");
    m.def("quant_fp8_triton_exact(Tensor x, Tensor scale_a) -> Tensor");
    m.def("repack_bf16_rans(Tensor B_rowmajor) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("bf16_rans_m1(Tensor A, Tensor sm_pack, Tensor exp_stream, Tensor tile_chunk_off, Tensor init_states, Tensor lut, int N) -> Tensor");
    m.def("silu_mul_fp8quant_mode0(Tensor gate_up, Tensor out_scale_a) -> Tensor");
    m.def("repack_nvfp4_m512_contig(Tensor B_q, Tensor B_s, int N, int K, bool is_paired) -> Tensor");
    m.def("nvfp4_gate_up_silu_contig_m512(Tensor A, Tensor B_pack, Tensor g_scale, int half_N) -> Tensor");
    m.def("nvfp4_gate_up_silu_fp8quant_contig_m512(Tensor A, Tensor B_pack, Tensor g_scale, Tensor out_scale_a, int half_N) -> Tensor");
    m.def("nvfp4_lpt_contig_m512(Tensor A, Tensor B_pack, Tensor g_scale, Tensor task_list, Tensor reduce_list, Tensor(a!) partials, int N) -> Tensor");
    m.def("nvfp4_prefill_contig_bm64_bn128(Tensor A, Tensor B_pack, Tensor g_scale, int N) -> Tensor");
    m.def("repack_fp8_m512_contig(Tensor B_colmajor, bool is_paired) -> Tensor");
    m.def("fp8_gate_up_silu_contig_m512(Tensor A, Tensor B_pack, Tensor scale_a, Tensor scale_b, int half_N) -> Tensor");
    m.def("fp8_gate_up_silu_fp8quant_contig_m512(Tensor A, Tensor B_pack, Tensor scale_a, Tensor scale_b, Tensor out_scale_a, int half_N) -> Tensor");
    m.def("fp8_linear_contig_m512(Tensor A, Tensor B_pack, Tensor scale_a, Tensor scale_b, int N) -> Tensor");
}

TORCH_LIBRARY_IMPL(muse_fast, CUDA, m) {
    m.impl("nvfp4_down_m1", nvfp4_down_m1);
    m.impl("nvfp4_gate_up_silu_m1", nvfp4_gate_up_silu_m1);
    m.impl("silu_mul_mode0", silu_mul_mode0);
    m.impl("nvfp4_lmhead_m1", nvfp4_lmhead_m1);
    m.impl("nvfp4_prefill_bm64_bn128", nvfp4_prefill_bm64_bn128);
    m.impl("nvfp4_gate_up_silu_m512", nvfp4_gate_up_silu_m512);
    m.impl("nvfp4_gate_up_silu_fp8quant_m512", nvfp4_gate_up_silu_fp8quant_m512);
    m.impl("nvfp4_down_m512_lpt", nvfp4_down_m512_lpt);
    m.impl("repack_fp8_mma", repack_fp8_mma);
    m.impl("fp8_mma_repacked_m1", fp8_mma_repacked_m1);
    m.impl("repack_fp8_rans", repack_fp8_rans);
    m.impl("fp8_rans_m1", fp8_rans_m1);
    m.impl("init_decode_loop", init_decode_loop);
    m.impl("greedy_argmax_step_update", greedy_argmax_step_update);
    m.impl("nvfp4_lmhead_m1_argmax_step_update", nvfp4_lmhead_m1_argmax_step_update);
    m.impl("repack_nvfp4_m1", repack_nvfp4_m1);
    m.impl("nvfp4_tm_linear_m1", nvfp4_tm_linear_m1);
    m.impl("nvfp4_tm_gate_up_silu_m1", nvfp4_tm_gate_up_silu_m1);
    m.impl("nvfp4_tm_lmhead_m1_argmax_step_update", nvfp4_tm_lmhead_m1_argmax_step_update);
    m.impl("repack_nvfp4_rans", repack_nvfp4_rans);
    m.impl("nvfp4_rans_linear_m1", nvfp4_rans_linear_m1);
    m.impl("repack_nvfp4_rans_splitk96", repack_nvfp4_rans_splitk96);
    m.impl("nvfp4_rans_splitk96_m1", nvfp4_rans_splitk96_m1);
    m.impl("nvfp4_rans_gate_up_silu_m1", nvfp4_rans_gate_up_silu_m1);
    m.impl("nvfp4_rans_gate_up_silu_fp8quant_m1", nvfp4_rans_gate_up_silu_fp8quant_m1);
    m.impl("nvfp4_rans_lmhead_m1_argmax_step_update", nvfp4_rans_lmhead_m1_argmax_step_update);
    m.impl("nvfp4_rans_lmhead_m1_argmax_only", nvfp4_rans_lmhead_m1_argmax_only);
    m.impl("quant_fp8_triton_exact", quant_fp8_triton_exact);
    m.impl("repack_bf16_rans", repack_bf16_rans);
    m.impl("bf16_rans_m1", bf16_rans_m1);
    m.impl("silu_mul_fp8quant_mode0", silu_mul_fp8quant_mode0);
    m.impl("repack_nvfp4_m512_contig", repack_nvfp4_m512_contig);
    m.impl("nvfp4_gate_up_silu_contig_m512", nvfp4_gate_up_silu_contig_m512);
    m.impl("nvfp4_gate_up_silu_fp8quant_contig_m512", nvfp4_gate_up_silu_fp8quant_contig_m512);
    m.impl("nvfp4_lpt_contig_m512", nvfp4_lpt_contig_m512);
    m.impl("nvfp4_prefill_contig_bm64_bn128", nvfp4_prefill_contig_bm64_bn128);
    m.impl("repack_fp8_m512_contig", repack_fp8_m512_contig);
    m.impl("fp8_gate_up_silu_contig_m512", fp8_gate_up_silu_contig_m512);
    m.impl("fp8_gate_up_silu_fp8quant_contig_m512", fp8_gate_up_silu_fp8quant_contig_m512);
    m.impl("fp8_linear_contig_m512", fp8_linear_contig_m512);
}
