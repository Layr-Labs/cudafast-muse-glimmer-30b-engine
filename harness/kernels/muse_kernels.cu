#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <cstdint>

namespace {

__device__ __forceinline__ __nv_bfloat162 dequant_s_bf162(uint32_t b) {
    uint32_t bits = b ? ((b << 4) + 0x3800u) : 0u;
    uint32_t pair = bits | (bits << 16);
    return *reinterpret_cast<__nv_bfloat162*>(&pair);
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
// 1. High-occupancy chunked-SMEM NVFP4 M=1 GEMV for down_proj (N=6656, K=19968).
//    Exact 96 half-stripe Marlin partition (iters=169), 3 KB SMEM -> 4+ CTAs/SM.
// ---------------------------------------------------------------------------
__global__ void nvfp4_down_m1_kernel(
    const uint32_t* __restrict__ A,
    const uint4* __restrict__ B_q,
    const uint8_t* __restrict__ B_s,
    const float* __restrict__ g_scale,
    float* __restrict__ partials,
    __nv_bfloat16* __restrict__ C,
    int N, int K
) {
    __shared__ uint32_t sA[512];  // 8 k128-tiles = 1024 bf16 = 2 KB
    __shared__ float s_acc[4][64];

    int w = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int group = lane >> 2;
    int t = lane & 3;

    int mblk = blockIdx.x >> 1;
    int half = blockIdx.x & 1;
    constexpr int k_tiles = 156;
    constexpr int iters = 169;
    int t_start = mblk * iters;
    int t_end = (mblk + 1) * iters;
    int row_stride_int4 = N >> 1;
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

        for (int c_start = seg_start; c_start < seg_end; c_start += 8) {
            int c_end = min(c_start + 8, seg_end);
            int c_len_u4 = (c_end - c_start) * 16;
            const uint4* A_src = reinterpret_cast<const uint4*>(A + c_start * 64);
            uint4* sA4 = reinterpret_cast<uint4*>(sA);
            for (int i = threadIdx.x; i < c_len_u4; i += blockDim.x) {
                sA4[i] = A_src[i];
            }
            __syncthreads();

            int k16_base = c_start * 8 + 2 * w;
            uint4 qv_cur;
            uint64_t sc_cur;
            asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(qv_cur.x), "=r"(qv_cur.y), "=r"(qv_cur.z), "=r"(qv_cur.w)
                : "l"(B_q + static_cast<uint64_t>(k16_base) * row_stride_int4 + n64 * 32 + lane));
            sc_cur = *reinterpret_cast<const uint64_t*>(
                B_s + static_cast<uint64_t>(k16_base) * N + n64 * 64 + group * 8);

            int n_steps = (c_end - c_start) * 2;
            for (int step = 0; step < n_steps; ++step) {
                int local_kt = step >> 1;
                int sub = step & 1;
                int local_k16 = local_kt * 8 + 2 * w + sub;
                uint4 qv_nxt;
                uint64_t sc_nxt;
                if (step + 1 < n_steps) {
                    int nkt = c_start + ((step + 1) >> 1);
                    int nsub = (step + 1) & 1;
                    int nk16 = nkt * 8 + 2 * w + nsub;
                    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(qv_nxt.x), "=r"(qv_nxt.y), "=r"(qv_nxt.z), "=r"(qv_nxt.w)
                        : "l"(B_q + static_cast<uint64_t>(nk16) * row_stride_int4 + n64 * 32 + lane));
                    sc_nxt = *reinterpret_cast<const uint64_t*>(
                        B_s + static_cast<uint64_t>(nk16) * N + n64 * 64 + group * 8);
                }
                uint32_t a0 = (group == 0) ? sA[local_k16 * 8 + t] : 0u;
                uint32_t a2 = (group == 0) ? sA[local_k16 * 8 + 4 + t] : 0u;
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
            __syncthreads();
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
    __nv_bfloat16* __restrict__ C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int col = idx >> 7;
    int col_start = col * 156;
    int col_end = col_start + 156;
    if (col_start / 169 != (col_end - 1) / 169) {
        C[idx] = __float2bfloat16_rn((partials[idx] + partials[N + idx]) * (*g_scale));
    }
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
// 3. Bit-exact column-major L2-resident SM121 NVFP4 Prefill GEMM (BM=64, BN=256, BK=64)
//    4-stage cp.async pipeline with XOR-swizzled sA and ldmatrix.x4.
//    Zero extra weight memory; streams each n256 column from DRAM once via L2 reuse.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 1) nvfp4_prefill_bm64_bn256_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint4* __restrict__ B_q,
    const uint8_t* __restrict__ B_s,
    const float* __restrict__ g_scale,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    constexpr int STAGES = 4;
    int m_tiles = M >> 6;
    int m_blk = blockIdx.x % m_tiles;
    int n_blk = blockIdx.x / m_tiles;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int group = lane >> 2;
    int t = lane & 3;
    int w_k = warp_id >> 2;
    int w_n = warp_id & 3;
    int n64 = n_blk * 4 + w_n;

    extern __shared__ char smem_raw[];
    uint4* sA = reinterpret_cast<uint4*>(smem_raw);
    uint4* sBq = reinterpret_cast<uint4*>(smem_raw + STAGES * 8192);
    uint8_t* sBs = reinterpret_cast<uint8_t*>(smem_raw + STAGES * 16384);

    int k_stages = K >> 6;
    int row_stride_int4 = N >> 1;
    const uint4* A_base4 = reinterpret_cast<const uint4*>(A + static_cast<uint64_t>(m_blk * 64) * K);
    int k_u4 = K >> 3;

    auto issue_stage = [&](int ks, int st) {
        uint4* sA_st = sA + st * 512;
        int k_off_u4 = ks * 8;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid + i * 256;
            int r = idx >> 3;
            int c8 = idx & 7;
            cp_async_16(sA_st + r * 8 + (c8 ^ (r & 7)),
                        A_base4 + static_cast<uint64_t>(r) * k_u4 + k_off_u4 + c8);
        }
        uint4* sBq_st = sBq + st * 512;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid + i * 256;
            int k16_local = idx >> 7;
            int col128 = idx & 127;
            int k16 = ks * 4 + k16_local;
            cp_async_16(sBq_st + idx,
                        B_q + static_cast<uint64_t>(k16) * row_stride_int4 + n_blk * 128 + col128);
        }
        if (tid < 64) {
            int k16_local = tid >> 4;
            int c16 = tid & 15;
            int k16 = ks * 4 + k16_local;
            cp_async_16(sBs + st * 1024 + k16_local * 256 + c16 * 16,
                        B_s + static_cast<uint64_t>(k16) * N + n_blk * 256 + c16 * 16);
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

    for (int ks = 0; ks < k_stages; ++ks) {
        int next_ks = ks + (STAGES - 1);
        if (next_ks < k_stages) {
            issue_stage(next_ks, next_ks % STAGES);
            asm volatile("cp.async.wait_group 2;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        int st = ks % STAGES;
        const uint4* sA_st = sA + st * 512;
        const uint4* sBq_st = sBq + st * 512;
        const uint8_t* sBs_st = sBs + st * 1024;

        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            int k16_local = w_k * 2 + sub;
            uint4 qv = sBq_st[k16_local * 128 + w_n * 32 + lane];
            uint2 sc2 = *reinterpret_cast<const uint2*>(
                sBs_st + k16_local * 256 + w_n * 64 + group * 8);

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
                uint32_t f0 = ((q & 0x00080008u) << 12) | ((q & 0x00070007u) << 6);
                uint32_t f1 = ((q & 0x00800080u) << 8)  | ((q & 0x00700070u) << 2);
                uint32_t f2 = ((q & 0x08000800u) << 4)  | ((q & 0x07000700u) >> 2);
                uint32_t f3 = ((q & 0x80008000u))       | ((q & 0x70007000u) >> 6);
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

template <bool SB_IS_VEC>
__global__ void fp8_mma_repacked_m1_kernel(
    const uint8_t* __restrict__ A,
    const uint4* __restrict__ B_pack,
    __nv_bfloat16* __restrict__ C,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    int N, int K
) {
    extern __shared__ uint32_t sA32[];
    const uint4* A4 = reinterpret_cast<const uint4*>(A);
    uint4* sA4 = reinterpret_cast<uint4*>(sA32);
    for (int i = threadIdx.x; i < (K >> 4); i += blockDim.x) {
        sA4[i] = A4[i];
    }
    __syncthreads();

    float sa = *scale_a;
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
}

at::Tensor nvfp4_down_m1(
    const at::Tensor& A,
    const at::Tensor& B_q,
    const at::Tensor& B_s,
    const at::Tensor& g_scale
) {
    constexpr int N = 6656;
    constexpr int K = 19968;
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 1 && A.size(1) == K);
    auto C = at::empty({1, N}, A.options());
    auto partials = at::empty({2, N}, A.options().dtype(at::kFloat));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    nvfp4_down_m1_kernel<<<96, 128, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(A.data_ptr()),
        reinterpret_cast<const uint4*>(B_q.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_s.data_ptr()),
        g_scale.data_ptr<float>(),
        partials.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        N, K
    );
    nvfp4_down_reduce_kernel<<<(N + 255) / 256, 256, 0, stream>>>(
        partials.data_ptr<float>(),
        g_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        N
    );
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

at::Tensor nvfp4_prefill_bm64_bn256(
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
    TORCH_CHECK(M % 64 == 0 && n_int % 256 == 0 && K % 64 == 0);
    auto C = at::empty({M, n_int}, A.options());
    int grid = (M / 64) * (n_int / 256);
    constexpr int smem = 4 * (8192 + 8192 + 1024);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            nvfp4_prefill_bm64_bn256_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem
        );
        attr_set = true;
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    nvfp4_prefill_bm64_bn256_kernel<<<grid, 256, smem, stream>>>(
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
    int grid = 144;
    int blk = 128;
    if (n_int >= 32768) {
        grid = 192;
        blk = 128;
    } else if (K <= 4096 || n_int == 4096) {
        grid = 144;
        blk = 256;
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (scale_b.numel() > 1) {
        fp8_mma_repacked_m1_kernel<true><<<grid, blk, K, stream>>>(
            reinterpret_cast<const uint8_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(B_pack.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            n_int, K
        );
    } else {
        fp8_mma_repacked_m1_kernel<false><<<grid, blk, K, stream>>>(
            reinterpret_cast<const uint8_t*>(A.data_ptr()),
            reinterpret_cast<const uint4*>(B_pack.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
            scale_a.data_ptr<float>(),
            scale_b.data_ptr<float>(),
            n_int, K
        );
    }
    return C;
}

}  // namespace

TORCH_LIBRARY(muse_fast, m) {
    m.def("nvfp4_down_m1(Tensor A, Tensor B_q, Tensor B_s, Tensor g_scale) -> Tensor");
    m.def("nvfp4_lmhead_m1(Tensor A, Tensor B_q, Tensor B_s, Tensor g_scale) -> Tensor");
    m.def("nvfp4_prefill_bm64_bn256(Tensor A, Tensor B_q, Tensor B_s, Tensor g_scale, int N) -> Tensor");
    m.def("repack_fp8_mma(Tensor B_colmajor) -> Tensor");
    m.def("fp8_mma_repacked_m1(Tensor A, Tensor B_pack, Tensor scale_a, Tensor scale_b, int N) -> Tensor");
}

TORCH_LIBRARY_IMPL(muse_fast, CUDA, m) {
    m.impl("nvfp4_down_m1", nvfp4_down_m1);
    m.impl("nvfp4_lmhead_m1", nvfp4_lmhead_m1);
    m.impl("nvfp4_prefill_bm64_bn256", nvfp4_prefill_bm64_bn256);
    m.impl("repack_fp8_mma", repack_fp8_mma);
    m.impl("fp8_mma_repacked_m1", fp8_mma_repacked_m1);
}
