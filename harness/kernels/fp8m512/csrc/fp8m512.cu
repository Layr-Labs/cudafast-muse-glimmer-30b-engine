// fp8m512: 512-row FP8 (e4m3 x e4m3 -> bf16) GEMM for the prompt pass of the FP8 down
// projections, bitwise equal to vLLM's cutlass_scaled_mm (checked per layer at load).
//
// Source: public Yukon submission 98cd9fcd-35ae-4820-a97c-e61177a5cdd3, commit
// e3f124f6e127f9e7f2986aef48303262ee78672e, harness/kernels/muse_kernels.cu, namespace
// m512_contig (ldmatrix_x4, mma_m16n8k32_fp8_full, cp_async_cg16,
// repack_fp8_tile_contig_kernel, fp8_linear_contig_m512_kernel). The device code is that
// file's text, unchanged; only the C entry points at the end are new (the source wraps the
// kernels in a torch extension).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace m512_contig {

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

extern "C" int fp8m512_init() {
  auto e = cudaFuncSetAttribute(m512_contig::fp8_linear_contig_m512_kernel<false>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
  if (e != cudaSuccess) return e;
  return cudaFuncSetAttribute(m512_contig::fp8_linear_contig_m512_kernel<true>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
}

// B: e4m3 weight as vLLM holds it, column-major [K, N] (N rows of K bytes in memory);
// B_pack: N / 128 * K / 64 * 512 uint4.
extern "C" int fp8m512_repack(const void* B, void* B_pack, int N, int K, cudaStream_t stream) {
  if (N % 128 || K % 64) return cudaErrorInvalidValue;
  dim3 grid(N / 128, K / 64);
  m512_contig::repack_fp8_tile_contig_kernel<false><<<grid, 128, 0, stream>>>(
      static_cast<const uint4*>(B), static_cast<uint4*>(B_pack), N, K);
  return cudaGetLastError();
}

// A: e4m3 [M, K] contiguous, C: bf16 [M, N]; scale_a fp32 [1]; scale_b fp32 [1] or [N].
extern "C" int fp8m512_gemm(const void* A, const void* B_pack, const float* scale_a,
                            const float* scale_b, void* C, int M, int N, int K, int sb_is_vec,
                            cudaStream_t stream) {
  if (M % 128 || N % 128 || K % 64) return cudaErrorInvalidValue;
  const int grid = (M / 128) * (N / 128);
  if (sb_is_vec)
    m512_contig::fp8_linear_contig_m512_kernel<true><<<grid, 128, 49152, stream>>>(
        static_cast<const uint8_t*>(A), static_cast<const uint4*>(B_pack), scale_a, scale_b,
        static_cast<__nv_bfloat16*>(C), M, N, K);
  else
    m512_contig::fp8_linear_contig_m512_kernel<false><<<grid, 128, 49152, stream>>>(
        static_cast<const uint8_t*>(A), static_cast<const uint4*>(B_pack), scale_a, scale_b,
        static_cast<__nv_bfloat16*>(C), M, N, K);
  return cudaGetLastError();
}
