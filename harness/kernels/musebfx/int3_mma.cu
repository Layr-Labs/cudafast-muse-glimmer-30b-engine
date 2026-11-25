// SPDX-License-Identifier: Apache-2.0
// muse_int3_mma: 3-bit asymmetric group-64 W3A16 weight-only Tensor-Core GEMM for the pinned DFlash drafter.
// Format (3.50 bits/weight, 0 wasted bits, 16-byte aligned):
//   Each 8x64 tile (8 output rows x 64 K cols) is 224 bytes (14 x uint4):
//     [0..127]   uint32_t lo2[32]  : for lane l=(r<<2)|t, 16 x 2-bit low bits (s=0..3, 4 weights/k16)
//     [128..191] uint16_t hi1[32]  : for lane l=(r<<2)|t, 16 x 1-bit high bits (s=0..3, 4 weights/k16)
//     [192..223] __nv_bfloat162 sz[8] : (scale, zero) for rows r=0..7 (w = q * scale + zero, q in 0..7)
//   Tiles are stored in [N/64, K/64, 8] order (1792 bytes = 112 x uint4 per 4-warp CTA per K=64 step).
//   Runs on SM121 with 6-stage cp.async (L2::evict_first), 1 barrier/step, 2x N-tile A reuse, and 2 MiB TLB prefetch.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>

namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * 32;    // 128
constexpr int kNTilesPerWarp = 2;                // 16 N-rows per warp -> 2x A reuse!
constexpr int kNTilesPerCTA = kWarpsPerBlock * kNTilesPerWarp;  // 8
constexpr int kRowsPerCTA = kNTilesPerCTA * 8;   // 64
constexpr int kTileBytes = 224;                  // 14 * 16
constexpr int kStepWBytes = kNTilesPerCTA * kTileBytes;  // 1792 = 112 * 16
constexpr int kStepWU4 = kStepWBytes / 16;       // 112
constexpr int kStages = 6;

__device__ __forceinline__ void int3_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint64_t int3_pol_ef() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

__device__ __forceinline__ void int3_cp_async16_ef(uint32_t sd, const void* gs, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(sd), "l"(gs), "l"(pol));
}

__device__ __forceinline__ void int3_cp_async16_ca(uint32_t sd, const void* gs) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(sd), "l"(gs));
}

__device__ __forceinline__ uint32_t int3_bf16x2_u32(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// GPU packer: BF16 [N, K] (N%64==0, K%64==0) -> uint8 [N/64, K/64, 8, 224]
// Grid: (N/8, K/64), Block: 32 threads (1 warp per 8x64 tile).
__global__ void int3_pack_kernel(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ out,
    int N, int K) {
  const int n_tile = blockIdx.x;        // 0 .. N/8 - 1
  const int kg = blockIdx.y;            // 0 .. K/64 - 1
  const int lane = threadIdx.x;         // 0 .. 31
  const int r = lane >> 2;              // 0 .. 7
  const int t = lane & 3;               // 0 .. 3
  const int row = n_tile * 8 + r;
  const int k_base = kg * 64;

  float vals[16];
  float rmin = 1e30f;
  float rmax = -1e30f;
  const __nv_bfloat16* wrow = W + static_cast<size_t>(row) * K + k_base;
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int kb = s * 16 + 2 * t;
    vals[4 * s + 0] = __bfloat162float(wrow[kb + 0]);
    vals[4 * s + 1] = __bfloat162float(wrow[kb + 1]);
    vals[4 * s + 2] = __bfloat162float(wrow[kb + 8]);
    vals[4 * s + 3] = __bfloat162float(wrow[kb + 9]);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    rmin = fminf(rmin, vals[i]);
    rmax = fmaxf(rmax, vals[i]);
  }
  rmin = fminf(rmin, __shfl_xor_sync(0xffffffffu, rmin, 1));
  rmin = fminf(rmin, __shfl_xor_sync(0xffffffffu, rmin, 2));
  rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, 1));
  rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, 2));

  float sc = (rmax - rmin) * (1.0f / 7.0f);
  if (sc < 1e-30f) sc = 1e-30f;
  const __nv_bfloat162 sz_bf = __floats2bfloat162_rn(sc, rmin);
  const float sc_q = __bfloat162float(sz_bf.x);
  const float zr_q = __bfloat162float(sz_bf.y);
  const float inv_sc = (sc_q > 1e-30f) ? (1.0f / sc_q) : 0.0f;

  uint32_t lo2 = 0u;
  uint32_t hi1 = 0u;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    int q = __float2int_rn((vals[i] - zr_q) * inv_sc);
    if (q < 0) q = 0;
    if (q > 7) q = 7;
    lo2 |= (static_cast<uint32_t>(q & 3) << (2 * i));
    hi1 |= (static_cast<uint32_t>((q >> 2) & 1) << i);
  }

  const int cta_n = n_tile >> 3;         // 0 .. N/64 - 1
  const int sub_tile = n_tile & 7;       // 0 .. 7
  const int num_kg = K >> 6;
  const size_t tile_idx = (static_cast<size_t>(cta_n) * num_kg + kg) * kNTilesPerCTA + sub_tile;
  uint8_t* dst = out + tile_idx * kTileBytes;

  reinterpret_cast<uint32_t*>(dst)[lane] = lo2;
  reinterpret_cast<uint16_t*>(dst + 128)[lane] = static_cast<uint16_t>(hi1);
  if (t == 0) {
    reinterpret_cast<__nv_bfloat162*>(dst + 192)[r] = sz_bf;
  }
}

struct alignas(16) Int3StageSmem {
  uint8_t w[kStages][kStepWBytes];     // 6 * 1792 = 10752 B
  __nv_bfloat16 a[kStages][16][72];    // 6 * 2304 = 13824 B (bank-conflict-free 144B row stride)
};                                     // Total = 24576 B (24.0 KiB)

struct Int3Aux5Ptrs {
  const __nv_bfloat16* p0;
  const __nv_bfloat16* p1;
  const __nv_bfloat16* p2;
  const __nv_bfloat16* p3;
  const __nv_bfloat16* p4;
  int kg_each;
};

__device__ __forceinline__ const __nv_bfloat16* int3_aux5_ptr(const Int3Aux5Ptrs& a, int ci) {
  return (ci == 0) ? a.p0 : (ci == 1) ? a.p1 : (ci == 2) ? a.p2 : (ci == 3) ? a.p3 : a.p4;
}

// Persistent 48-CTA / 96-CTA or direct GEMM with optional split-K (for small N/64 like down/fc).
// When kAux5 is true, reads 5 contiguous [M, K_each] BF16 activation slices directly without torch.cat.
template <bool kSplitK, bool kAux5 = false>
__global__ void __launch_bounds__(128, 3) int3_mma_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    Int3Aux5Ptrs aux5,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ C_fp32,
    int M, int N, int K, int split_k) {
  __shared__ Int3StageSmem smem;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int num_kg = K >> 6;
  const uint64_t pol = int3_pol_ef();

  const int cta_n = blockIdx.x;
  const int sk_idx = kSplitK ? blockIdx.y : 0;
  const int kg_per_sk = kSplitK ? ((num_kg + split_k - 1) / split_k) : num_kg;
  const int kg_start = kSplitK ? (sk_idx * kg_per_sk) : 0;
  int kg_end = kSplitK ? (kg_start + kg_per_sk) : num_kg;
  if (kg_end > num_kg) kg_end = num_kg;
  const int my_kgs = kg_end - kg_start;
  if (my_kgs <= 0) return;

  const size_t cta_stride_bytes = static_cast<size_t>(num_kg) * kStepWBytes;
  const uint8_t* w_cta = W_packed + static_cast<size_t>(cta_n) * cta_stride_bytes;

  // Issue 2 MiB TLB prefetch from block (0, 0).
  if (cta_n == 0 && sk_idx == 0) {
    const size_t total_bytes = static_cast<size_t>(N / kRowsPerCTA) * cta_stride_bytes;
    for (size_t off = static_cast<size_t>(tid) << 21; off < total_bytes; off += static_cast<size_t>(kThreads) << 21) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(W_packed + off));
    }
  }

  auto issue_stage = [&](int kg, int st) {
    if (tid < kStepWU4) {
      const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.w[st][tid * 16]));
      int3_cp_async16_ef(sd, w_cta + static_cast<size_t>(kg) * kStepWBytes + tid * 16, pol);
    }
    {
      const int row = tid >> 3;
      const int col8 = tid & 7;
      if (row < M) {
        const uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.a[st][row][col8 * 8]));
        if constexpr (kAux5) {
          // Fast exact branchless chunk lookup for 5 chunks (avoids 32-bit integer division in hot loop).
          const int ke = aux5.kg_each;
          int ci = 0;
          int kg_loc = kg;
          if (kg_loc >= ke) { kg_loc -= ke; ci = 1; }
          if (kg_loc >= ke) { kg_loc -= ke; ci = 2; }
          if (kg_loc >= ke) { kg_loc -= ke; ci = 3; }
          if (kg_loc >= ke) { kg_loc -= ke; ci = 4; }
          const int k_each = ke << 6;
          int3_cp_async16_ca(sd, int3_aux5_ptr(aux5, ci) + static_cast<size_t>(row) * k_each + kg_loc * 64 + col8 * 8);
        } else {
          int3_cp_async16_ca(sd, A + static_cast<size_t>(row) * K + kg * 64 + col8 * 8);
        }
      }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  // Fill up to kStages - 1 = 5 stages in flight.
  constexpr int kPreload = kStages - 1;  // 5
  const int preload = (my_kgs < kPreload) ? my_kgs : kPreload;
  for (int i = 0; i < preload; ++i) {
    issue_stage(kg_start + i, i);
  }

  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;

  int st = 0;
  for (int i = 0; i < my_kgs; ++i) {
    const int rem = my_kgs - 1 - i;
    if (rem >= kPreload) {
      asm volatile("cp.async.wait_group 4;\n" ::: "memory");
    } else if (rem == 4) {
      asm volatile("cp.async.wait_group 4;\n" ::: "memory");
    } else if (rem == 3) {
      asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    } else if (rem == 2) {
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    } else if (rem == 1) {
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();

    // Load A + both W tiles from stage `st`, then immediately issue stage `i + 5` into free slot!
    const int g1 = g + 8;
    const uint32_t* a_row0_u32 = reinterpret_cast<const uint32_t*>(&smem.a[st][g][0]);
    const uint32_t* a_row1_u32 = reinterpret_cast<const uint32_t*>(&smem.a[st][g1][0]);
    uint32_t a0_r[4], a1_r[4], a2_r[4], a3_r[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      a0_r[s] = (g < M) ? a_row0_u32[s * 8 + t] : 0u;
      a2_r[s] = (g < M) ? a_row0_u32[s * 8 + 4 + t] : 0u;
      a1_r[s] = (g1 < M) ? a_row1_u32[s * 8 + t] : 0u;
      a3_r[s] = (g1 < M) ? a_row1_u32[s * 8 + 4 + t] : 0u;
    }

    const uint8_t* wt0 = &smem.w[st][(warp * 2 + 0) * kTileBytes];
    const uint8_t* wt1 = &smem.w[st][(warp * 2 + 1) * kTileBytes];
    uint32_t lo2_0 = reinterpret_cast<const uint32_t*>(wt0)[lane];
    uint32_t hi1_0 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wt0 + 128)[lane]);
    const __nv_bfloat162 sz0 = reinterpret_cast<const __nv_bfloat162*>(wt0 + 192)[g];
    const float sc0 = __bfloat162float(sz0.x);
    const float zr0 = __bfloat162float(sz0.y);

    uint32_t lo2_1 = reinterpret_cast<const uint32_t*>(wt1)[lane];
    uint32_t hi1_1 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wt1 + 128)[lane]);
    const __nv_bfloat162 sz1 = reinterpret_cast<const __nv_bfloat162*>(wt1 + 192)[g];
    const float sc1 = __bfloat162float(sz1.x);
    const float zr1 = __bfloat162float(sz1.y);

    // Next write slot is (st + kPreload) % kStages = (st + 5) % 6 (last read 1 iter ago, before __syncthreads()!).
    const int next_i = i + kPreload;
    if (next_i < my_kgs) {
      const int next_st = (st + kPreload >= kStages) ? (st + kPreload - kStages) : (st + kPreload);
      issue_stage(kg_start + next_i, next_st);
    }

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t q0 = (lo2_0 & 3u) | ((hi1_0 & 1u) << 2);
      const uint32_t q1 = ((lo2_0 >> 2) & 3u) | (((hi1_0 >> 1) & 1u) << 2);
      const uint32_t q2 = ((lo2_0 >> 4) & 3u) | (((hi1_0 >> 2) & 1u) << 2);
      const uint32_t q3 = ((lo2_0 >> 6) & 3u) | (((hi1_0 >> 3) & 1u) << 2);
      lo2_0 >>= 8;
      hi1_0 >>= 4;
      const uint32_t b0 = int3_bf16x2_u32(__fmaf_rn(static_cast<float>(q0), sc0, zr0),
                                          __fmaf_rn(static_cast<float>(q1), sc0, zr0));
      const uint32_t b1 = int3_bf16x2_u32(__fmaf_rn(static_cast<float>(q2), sc0, zr0),
                                          __fmaf_rn(static_cast<float>(q3), sc0, zr0));
      int3_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0_r[s], a1_r[s], a2_r[s], a3_r[s], b0, b1);

      const uint32_t p0 = (lo2_1 & 3u) | ((hi1_1 & 1u) << 2);
      const uint32_t p1 = ((lo2_1 >> 2) & 3u) | (((hi1_1 >> 1) & 1u) << 2);
      const uint32_t p2 = ((lo2_1 >> 4) & 3u) | (((hi1_1 >> 2) & 1u) << 2);
      const uint32_t p3 = ((lo2_1 >> 6) & 3u) | (((hi1_1 >> 3) & 1u) << 2);
      lo2_1 >>= 8;
      hi1_1 >>= 4;
      const uint32_t c0 = int3_bf16x2_u32(__fmaf_rn(static_cast<float>(p0), sc1, zr1),
                                          __fmaf_rn(static_cast<float>(p1), sc1, zr1));
      const uint32_t c1 = int3_bf16x2_u32(__fmaf_rn(static_cast<float>(p2), sc1, zr1),
                                          __fmaf_rn(static_cast<float>(p3), sc1, zr1));
      int3_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0_r[s], a1_r[s], a2_r[s], a3_r[s], c0, c1);
    }

    st = (st + 1 == kStages) ? 0 : (st + 1);
  }

  const int col0 = cta_n * kRowsPerCTA + (warp * 2 + 0) * 8 + 2 * t;
  const int col1 = cta_n * kRowsPerCTA + (warp * 2 + 1) * 8 + 2 * t;
  if (g < M) {
    if constexpr (!kSplitK) {
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col0) = int3_bf16x2_u32(d0_0, d1_0);
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g) * N + col1) = int3_bf16x2_u32(d0_1, d1_1);
    } else {
      float* p0 = C_fp32 + (static_cast<size_t>(sk_idx) * M + g) * N + col0;
      float* p1 = C_fp32 + (static_cast<size_t>(sk_idx) * M + g) * N + col1;
      p0[0] = d0_0; p0[1] = d1_0;
      p1[0] = d0_1; p1[1] = d1_1;
    }
  }
  if (g + 8 < M) {
    const int g1 = g + 8;
    if constexpr (!kSplitK) {
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g1) * N + col0) = int3_bf16x2_u32(d2_0, d3_0);
      *reinterpret_cast<uint32_t*>(C + static_cast<size_t>(g1) * N + col1) = int3_bf16x2_u32(d2_1, d3_1);
    } else {
      float* p0 = C_fp32 + (static_cast<size_t>(sk_idx) * M + g1) * N + col0;
      float* p1 = C_fp32 + (static_cast<size_t>(sk_idx) * M + g1) * N + col1;
      p0[0] = d2_0; p0[1] = d3_0;
      p1[0] = d2_1; p1[1] = d3_1;
    }
  }
}

__global__ void int3_reduce_splitk_kernel(
    const float* __restrict__ C_fp32,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int split_k) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = M * N;
  if (idx >= total) return;
  float sum = 0.0f;
  for (int s = 0; s < split_k; ++s) {
    sum += C_fp32[static_cast<size_t>(s) * total + idx];
  }
  C[idx] = __float2bfloat16_rn(sum);
}

}  // namespace

torch::Tensor int3_pack(torch::Tensor W) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kBFloat16 && W.dim() == 2 && W.is_contiguous(),
              "int3_pack expects contiguous CUDA BF16 [N, K]");
  const int N = static_cast<int>(W.size(0));
  const int K = static_cast<int>(W.size(1));
  TORCH_CHECK(N % kRowsPerCTA == 0 && K % 64 == 0, "int3_pack requires N%64==0 and K%64==0");
  const int64_t total_bytes = static_cast<int64_t>(N / 8) * static_cast<int64_t>(K / 64) * kTileBytes;
  auto out = torch::empty({total_bytes}, W.options().dtype(torch::kUInt8));
  dim3 grid(N / 8, K / 64);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  int3_pack_kernel<<<grid, 32, 0, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()),
      out.data_ptr<uint8_t>(),
      N, K);
  return out;
}

static void int3_gemm_launch(
    const __nv_bfloat16* a_ptr,
    Int3Aux5Ptrs aux5,
    bool is_aux5,
    const uint8_t* w_ptr,
    __nv_bfloat16* c_ptr,
    const torch::TensorOptions& opts,
    int M, int N, int K) {
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int n_blocks = N / kRowsPerCTA;
  static torch::Tensor g_partials;
  static int64_t g_partials_cap = 0;
  int split_k = 1;
  if (n_blocks <= 128 && K >= 16384) {
    split_k = (K >= 28672) ? 4 : 3;
  }
  if (split_k == 1) {
    if (is_aux5) {
      int3_mma_gemm_kernel<false, true><<<n_blocks, kThreads, 0, st>>>(
          nullptr, aux5, w_ptr, c_ptr, nullptr, M, N, K, 1);
    } else {
      int3_mma_gemm_kernel<false, false><<<n_blocks, kThreads, 0, st>>>(
          a_ptr, aux5, w_ptr, c_ptr, nullptr, M, N, K, 1);
    }
  } else {
    const int64_t need = static_cast<int64_t>(split_k) * M * N;
    const int64_t alloc = (need > 4 * 16 * 6656) ? need : (4 * 16 * 6656);
    if (!g_partials.defined() || g_partials.device() != opts.device() || g_partials_cap < need) {
      g_partials = torch::empty({alloc}, opts.dtype(torch::kFloat32));
      g_partials_cap = alloc;
    }
    if (is_aux5) {
      int3_mma_gemm_kernel<true, true><<<dim3(n_blocks, split_k, 1), kThreads, 0, st>>>(
          nullptr, aux5, w_ptr, nullptr, g_partials.data_ptr<float>(), M, N, K, split_k);
    } else {
      int3_mma_gemm_kernel<true, false><<<dim3(n_blocks, split_k, 1), kThreads, 0, st>>>(
          a_ptr, aux5, w_ptr, nullptr, g_partials.data_ptr<float>(), M, N, K, split_k);
    }
    const int total = M * N;
    int3_reduce_splitk_kernel<<<(total + 255) / 256, 256, 0, st>>>(
        g_partials.data_ptr<float>(), c_ptr, M, N, split_k);
  }
}

torch::Tensor int3_gemm(torch::Tensor A, torch::Tensor W_packed, int64_t N, int64_t K) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous(),
              "int3_gemm expects contiguous CUDA BF16 [M, K]");
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && A.size(1) == K && N % kRowsPerCTA == 0 && K % 64 == 0,
              "int3_gemm shape mismatch");
  auto C = torch::empty({M, N}, A.options());
  Int3Aux5Ptrs dummy = {};
  int3_gemm_launch(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      dummy, false,
      W_packed.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
      A.options(), M, static_cast<int>(N), static_cast<int>(K));
  return C;
}

void int3_gemm_out(torch::Tensor A, torch::Tensor W_packed, torch::Tensor C, int64_t N, int64_t K) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous(),
              "int3_gemm_out expects contiguous CUDA BF16 [M, K]");
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= 16 && A.size(1) == K && N % kRowsPerCTA == 0 && K % 64 == 0,
              "int3_gemm_out shape mismatch");
  TORCH_CHECK(C.is_cuda() && C.scalar_type() == torch::kBFloat16 && C.is_contiguous() && C.numel() == M * N,
              "int3_gemm_out bad C");
  Int3Aux5Ptrs dummy = {};
  int3_gemm_launch(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      dummy, false,
      W_packed.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
      A.options(), M, static_cast<int>(N), static_cast<int>(K));
}

void int3_gemm_aux5_out(
    torch::Tensor x0, torch::Tensor x1, torch::Tensor x2, torch::Tensor x3, torch::Tensor x4,
    torch::Tensor W_packed, torch::Tensor C, int64_t N, int64_t K_each) {
  const int64_t K = K_each * 5;
  TORCH_CHECK(N % kRowsPerCTA == 0 && K_each % 64 == 0, "int3_gemm_aux5_out requires N%64==0 and K_each%64==0");
  const torch::Tensor* xs[5] = {&x0, &x1, &x2, &x3, &x4};
  const int M = static_cast<int>(x0.size(0));
  TORCH_CHECK(M >= 1 && M <= 16, "int3_gemm_aux5_out requires 1 <= M <= 16");
  Int3Aux5Ptrs aux5 = {};
  aux5.kg_each = static_cast<int>(K_each >> 6);
  for (int i = 0; i < 5; ++i) {
    TORCH_CHECK(xs[i]->is_cuda() && xs[i]->scalar_type() == torch::kBFloat16 && xs[i]->dim() == 2 &&
                xs[i]->size(0) == M && xs[i]->size(1) == K_each && xs[i]->is_contiguous(),
                "int3_gemm_aux5_out bad aux tensor");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(xs[i]->data_ptr()) & 15u) == 0u,
                "int3_gemm_aux5_out requires 16B alignment");
  }
  aux5.p0 = reinterpret_cast<const __nv_bfloat16*>(x0.data_ptr());
  aux5.p1 = reinterpret_cast<const __nv_bfloat16*>(x1.data_ptr());
  aux5.p2 = reinterpret_cast<const __nv_bfloat16*>(x2.data_ptr());
  aux5.p3 = reinterpret_cast<const __nv_bfloat16*>(x3.data_ptr());
  aux5.p4 = reinterpret_cast<const __nv_bfloat16*>(x4.data_ptr());
  TORCH_CHECK(C.is_cuda() && C.scalar_type() == torch::kBFloat16 && C.is_contiguous() && C.numel() == M * N,
              "int3_gemm_aux5_out bad C");
  int3_gemm_launch(
      nullptr, aux5, true,
      W_packed.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
      x0.options(), M, static_cast<int>(N), static_cast<int>(K));
}
