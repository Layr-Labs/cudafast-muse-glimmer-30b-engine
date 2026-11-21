// kmarlin: a Marlin-exact NVFP4 W4A16 GEMM for M = 1..8 bf16 rows (used at M = 1).
//
// Port of "marlinx" by dadon (Nemotron CUDA track, submission 100549d4), extended to
// Marlin's {thread_k 128, thread_n 64, 128 threads} configuration (lm_head 6656 -> 202048,
// where N % 128 != 0) and given switchable, value-neutral memory hints (L2 evict_first
// policy, L2 look-ahead prefetch, programmatic dependent launch).
//
// What vLLM v0.27.1's dense Marlin computes (csrc/libtorch_stable/quantization/marlin,
// MarlinNvFp4LinearKernel path, use_fp32_reduce = true, use_atomic_add = false, M <= 8 so
// thread_m_blocks = 1 and m_block_size_8), per output element (token t, feature n):
//   * The thread config is {128, 128, 256} when N % 128 == 0 and K % 128 == 0, else
//     {128, 64, 128} (when N % 64 == 0); tile_n = 128 or 64 features per Marlin column.
//   * Marlin's stripe partition over grid = #SMs cuts each column's k-tiles (128 K each)
//     into segments; each segment is reduced inside one CTA.
//   * Inside a segment, 4 warp rows w = 0..3 each chain mma.m16n8k16.bf16 (weights as the A
//     operand, "mma_trans") over the k16 blocks 8*tau + 2*w and 8*tau + 2*w + 1, tau
//     ascending, from +0. A 64-feature group is one Marlin warp column in both configs.
//   * CTA partial P = acc0 + (acc1 + (acc3 + acc2))  (thread_block_reduce).
//   * Segments fold bottom-up in fp32: R = P_bottom, R = P + R  (global_reduce_fp32).
//   * out = bf16_rn(R * global_scale)  (write_result).
// The weight dequant (bit placement, bf16 scale multiply) and the operand placement are
// Marlin's. Only the schedule differs: one CTA of 4 warps owns one 64-feature unit for its
// whole K, streams it through a register ring, and folds segments in shared memory.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace kmarlin {

struct Params {
  const __nv_bfloat16* A;  // [M, lda] bf16 activations
  long long lda;           // elements
  int M;                   // 1..8
  const uint4* B;          // Marlin-packed fp4 weights, [K/16][N/2] int4
  const uint2* S;          // Marlin-processed e4m3 scales, [K/16][N] bytes
  const float* GS;         // Marlin-processed fp32 global scale (1 value)
  __nv_bfloat16* C;        // [M, ldc] bf16 output
  long long ldc;           // elements
  int N, K;
  int units_per_tile;      // tile_n / 64 (2 for Marlin's 128-wide config, 1 for 64-wide)
  int n_tiles, k_tiles;    // N / tile_n, K / 128
  int part1_cols;          // Marlin columns reduced by a single block
  int iters;               // Marlin stripe length (k-tiles per block) in part 2
};

__device__ __forceinline__ uint64_t evict_first_policy() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}

// Weight stream: read once per call; bypass L1, optionally mark evict-first in L2.
template <bool EVICT>
__device__ __forceinline__ uint4 ld_w_v4(const uint4* p, uint64_t pol) {
  uint4 r;
  if constexpr (EVICT) {
    asm volatile(
        "ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
        : "l"(p), "l"(pol));
  } else {
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
  }
  return r;
}

template <bool EVICT>
__device__ __forceinline__ uint2 ld_s_v2(const uint2* p, uint64_t pol) {
  uint2 r;
  if constexpr (EVICT) {
    asm volatile("ld.global.nc.L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;\n"
                 : "=r"(r.x), "=r"(r.y)
                 : "l"(p), "l"(pol));
  } else {
    asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  }
  return r;
}

__device__ __forceinline__ uint32_t ld_nc_u32(const uint32_t* p) {
  uint32_t r;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}

__device__ __forceinline__ void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p));
}

// PDL: wait for the preceding kernel (and its memory) before the first dependent read.
__device__ __forceinline__ void grid_dependency_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

__device__ __forceinline__ __nv_bfloat162 as_bf162(uint32_t v) {
  return *reinterpret_cast<__nv_bfloat162*>(&v);
}

__device__ __forceinline__ uint32_t as_u32(__nv_bfloat162 v) {
  return *reinterpret_cast<uint32_t*>(&v);
}

// Marlin dequant<nv_bfloat162, kFE2M1f, dequant_skip_flop = true> (dequant.h).
__device__ __forceinline__ void dequant_fp4(uint32_t q, uint32_t (&out)[2]) {
  uint32_t o1 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
  q <<= 4;
  uint32_t o2 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
  out[1] = o1;  // reverse indexing as in Marlin
  out[0] = o2;
}

// Marlin dequant_fp8_scales<nv_bfloat162, kFE4M3fn> (dequant.h; the 0x80008000 literal is
// unsigned there too, so the shift is logical).
__device__ __forceinline__ void dequant_scales(uint32_t q, uint32_t (&out)[2]) {
  uint32_t o1 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
  q <<= 8;
  uint32_t o2 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
  out[1] = o1;
  out[0] = o2;
}

// Marlin mma_trans<bf16>: weights (two n8 tiles) as the A operand, the activations of 8
// tokens as the B operand.
__device__ __forceinline__ void mma_trans(const uint32_t (&a)[2], const uint32_t (&b)[2],
                                          const uint32_t (&b2)[2], float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]), "f"(c[0]),
        "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// One k16 block of one warp: Marlin's matmul() body for m_block_size_8, group_blocks 1.
__device__ __forceinline__ void block_mma(const uint4& bq, const uint2& sq, const uint32_t (&a)[2],
                                          float (&acc)[4][4]) {
  uint32_t fs[4];
  {
    uint32_t t[2];
    dequant_scales(sq.x, t);
    fs[0] = t[0];
    fs[1] = t[1];
    dequant_scales(sq.y, t);
    fs[2] = t[0];
    fs[3] = t[1];
  }
  const uint32_t q[4] = {bq.x, bq.y, bq.z, bq.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t b0[2], b1[2];
    dequant_fp4(q[j] << 8, b0);  // b_quant_0 = b_quant_1 << 8
    dequant_fp4(q[j], b1);       // b_quant_1
    const __nv_bfloat162 s = as_bf162(fs[j]);
    const __nv_bfloat162 sx = __bfloat162bfloat162(s.x);
    const __nv_bfloat162 sy = __bfloat162bfloat162(s.y);
    b0[0] = as_u32(__hmul2(as_bf162(b0[0]), sx));
    b0[1] = as_u32(__hmul2(as_bf162(b0[1]), sx));
    b1[0] = as_u32(__hmul2(as_bf162(b1[0]), sy));
    b1[1] = as_u32(__hmul2(as_bf162(b1[1]), sy));
    mma_trans(a, b0, b1, acc[j]);
  }
}

// Walks one column's k16 blocks for one warp chain: segments bottom-up (highest K
// first), tiles ascending inside a segment, k in {0, 1}.
struct Cursor {
  int tau, k, lo, hi;
};

__device__ __forceinline__ void cursor_init(Cursor& c, int first_cut, int k_tiles, int iters) {
  c.hi = k_tiles;
  c.lo = first_cut >= k_tiles ? 0 : first_cut + ((k_tiles - 1 - first_cut) / iters) * iters;
  c.tau = c.lo;
  c.k = 0;
}

// Advances past the current block; true if that block closed its segment.
__device__ __forceinline__ bool cursor_next(Cursor& c, int iters) {
  c.k ^= 1;
  if (c.k) return false;
  if (++c.tau < c.hi) return false;
  c.hi = c.lo;
  c.lo = c.lo - iters > 0 ? c.lo - iters : 0;
  c.tau = c.lo;
  return true;
}

// Segment end: Marlin's thread_block_reduce (P = acc0 + (acc1 + (acc3 + acc2))) and
// global_reduce_fp32 (R = P + R, bottom segment first). red is double-buffered by segment
// parity, so one barrier per segment suffices.
__device__ __forceinline__ void close_segment(float (&red)[2][3][16][32], int par, int w, int lane,
                                              float (&acc)[4][4], float (&R)[4][4], bool first_seg) {
  if (w) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) red[par][w - 1][4 * j + e][lane] = acc[j][e];
  }
  __syncthreads();
  if (!w) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float a1 = red[par][0][4 * j + e][lane];
        const float a2 = red[par][1][4 * j + e][lane];
        const float a3 = red[par][2][4 * j + e][lane];
        const float pp = acc[j][e] + (a1 + (a3 + a2));
        R[j][e] = first_seg ? pp : pp + R[j][e];
      }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;
}

// Marlin write_result: bf16_rn(R * global_scale), rows < M only (warp 0).
__device__ __forceinline__ void write_out(const Params& p, int unit, int lane,
                                          const float (&R)[4][4]) {
  const float g = __ldg(p.GS);
  const int t0 = 2 * (lane & 3);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const long long f = 64LL * unit + 16 * j + (lane >> 2);
    const __nv_bfloat16 v0 = __float2bfloat16_rn(R[j][0] * g);
    const __nv_bfloat16 v1 = __float2bfloat16_rn(R[j][1] * g);
    const __nv_bfloat16 v2 = __float2bfloat16_rn(R[j][2] * g);
    const __nv_bfloat16 v3 = __float2bfloat16_rn(R[j][3] * g);
    if (t0 < p.M) {
      p.C[t0 * p.ldc + f] = v0;
      p.C[t0 * p.ldc + f + 8] = v2;
    }
    if (t0 + 1 < p.M) {
      p.C[(t0 + 1) * p.ldc + f] = v1;
      p.C[(t0 + 1) * p.ldc + f + 8] = v3;
    }
  }
}

// First internal segment cut (k-tiles) of Marlin column `col`, k_tiles if none.
__device__ __forceinline__ int column_first_cut(const Params& p, int col) {
  if (col < p.part1_cols) return p.k_tiles;
  const int l0 = (col - p.part1_cols) * p.k_tiles;
  const int cut = (l0 / p.iters + 1) * p.iters - l0;
  return cut < p.k_tiles ? cut : p.k_tiles;
}

// One 64-feature unit per CTA of 4 warps; warp w runs Marlin's warp-row chain w through
// a register ring of P k16 blocks. PF > 0: prefetch.global.L2 the weight line PF blocks
// beyond the ring. PDL: weights fill the ring before griddepcontrol.wait, activations after.
template <int P, bool EVICT, int PF, bool PDL>
__global__ void __launch_bounds__(128) gemm_kernel(const Params p) {
  __shared__ float red[2][3][16][32];
  const int unit = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;  // Marlin warp row (chain)
  const int col = unit / p.units_per_tile;

  const int first_cut = column_first_cut(p, col);

  const long long nB = p.N >> 1;  // int4 per k16 row
  const long long nS = p.N >> 3;  // uint2 per k16 row
  const uint4* Bp = p.B + 32LL * unit + lane;
  const uint2* Sp = p.S + 8LL * unit + (lane >> 2);
  const int tok = lane >> 2;
  const bool arow = tok < p.M;
  const uint32_t* Ap =
      reinterpret_cast<const uint32_t*>(p.A + (arow ? (long long)tok * p.lda : 0)) + (lane & 3);
  uint64_t pol = 0;
  if constexpr (EVICT) pol = evict_first_policy();

  uint4 rb[P];
  uint2 rs[P];
  uint32_t ra[P][2];

  Cursor ld, cs, pf;
  cursor_init(ld, first_cut, p.k_tiles, p.iters);
  cs = ld;
  const int total = 2 * p.k_tiles;

  if constexpr (PDL) {
    Cursor a = ld;
#pragma unroll
    for (int s = 0; s < P; ++s) {
      if (s < total) {
        const int kb = 8 * ld.tau + 2 * w + ld.k;
        rb[s] = ld_w_v4<EVICT>(Bp + kb * nB, pol);
        rs[s] = ld_s_v2<EVICT>(Sp + kb * nS, pol);
        cursor_next(ld, p.iters);
      }
    }
    grid_dependency_wait();
#pragma unroll
    for (int s = 0; s < P; ++s) {
      if (s < total) {
        const int kb = 8 * a.tau + 2 * w + a.k;
        ra[s][0] = arow ? ld_nc_u32(Ap + 8 * kb) : 0u;
        ra[s][1] = arow ? ld_nc_u32(Ap + 8 * kb + 4) : 0u;
        cursor_next(a, p.iters);
      }
    }
  } else {
#pragma unroll
    for (int s = 0; s < P; ++s) {
      if (s < total) {
        const int kb = 8 * ld.tau + 2 * w + ld.k;
        rb[s] = ld_w_v4<EVICT>(Bp + kb * nB, pol);
        rs[s] = ld_s_v2<EVICT>(Sp + kb * nS, pol);
        ra[s][0] = arow ? ld_nc_u32(Ap + 8 * kb) : 0u;
        ra[s][1] = arow ? ld_nc_u32(Ap + 8 * kb + 4) : 0u;
        cursor_next(ld, p.iters);
      }
    }
  }
  if constexpr (PF > 0) {
    pf = ld;
#pragma unroll
    for (int s = 0; s < PF; ++s) {
      if (P + s < total) {
        prefetch_l2(Bp + (8 * pf.tau + 2 * w + pf.k) * nB);
        cursor_next(pf, p.iters);
      }
    }
  }

  float acc[4][4];
  float R[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      acc[j][e] = 0.0f;
      R[j][e] = 0.0f;
    }
  bool first_seg = true;
  int par = 0;

  for (int base = 0; base < total; base += P) {
#pragma unroll
    for (int s = 0; s < P; ++s) {
      const int i = base + s;
      if (i < total) {
        block_mma(rb[s], rs[s], ra[s], acc);
        const bool seg_end = cursor_next(cs, p.iters);
        if (i + P < total) {
          const int kb = 8 * ld.tau + 2 * w + ld.k;
          rb[s] = ld_w_v4<EVICT>(Bp + kb * nB, pol);
          rs[s] = ld_s_v2<EVICT>(Sp + kb * nS, pol);
          ra[s][0] = arow ? ld_nc_u32(Ap + 8 * kb) : 0u;
          ra[s][1] = arow ? ld_nc_u32(Ap + 8 * kb + 4) : 0u;
          cursor_next(ld, p.iters);
        }
        if constexpr (PF > 0) {
          if (i + P + PF < total) {
            prefetch_l2(Bp + (8 * pf.tau + 2 * w + pf.k) * nB);
            cursor_next(pf, p.iters);
          }
        }
        if (seg_end) {  // uniform across the CTA: every chain has the same length
          close_segment(red, par, w, lane, acc, R, first_seg);
          first_seg = false;
          par ^= 1;
        }
      }
    }
  }
  if (!w) write_out(p, unit, lane, R);
}

template <int P, bool EVICT, int PF, bool PDL>
static int launch(const Params& p, cudaStream_t stream) {
  const int units = p.n_tiles * p.units_per_tile;
  if constexpr (PDL) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(units);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return (int)cudaLaunchKernelEx(&cfg, gemm_kernel<P, EVICT, PF, PDL>, p);
  } else {
    gemm_kernel<P, EVICT, PF, PDL><<<units, 128, 0, stream>>>(p);
    return (int)cudaGetLastError();
  }
}

// Variant table: bit 0 = evict_first, bit 1 = L2 prefetch (distance 8), bit 2 = PDL.
using LaunchFn = int (*)(const Params&, cudaStream_t);
using AttrFn = const void*;

#define KM_VARIANTS(X)   \
  X(0, 8, false, 0, false) \
  X(1, 8, true, 0, false)  \
  X(2, 8, false, 8, false) \
  X(3, 8, true, 8, false)  \
  X(4, 8, false, 0, true)  \
  X(5, 8, true, 0, true)   \
  X(6, 8, false, 8, true)  \
  X(7, 8, true, 8, true)

static LaunchFn launch_fn(int v) {
  switch (v) {
#define KM_CASE(id, P, E, F, D) \
  case id:                      \
    return &launch<P, E, F, D>;
    KM_VARIANTS(KM_CASE)
#undef KM_CASE
    default:
      return nullptr;
  }
}

static const void* kernel_fn(int v) {
  switch (v) {
#define KM_CASE(id, P, E, F, D) \
  case id:                      \
    return reinterpret_cast<const void*>(&gemm_kernel<P, E, F, D>);
    KM_VARIANTS(KM_CASE)
#undef KM_CASE
    default:
      return nullptr;
  }
}

}  // namespace kmarlin

extern "C" {

// Loads every kernel variant into the context (call outside CUDA graph capture).
// Returns 0 or a cudaError_t value.
int kmarlin_init(void) {
  for (int v = 0; v < 8; ++v) {
    cudaFuncAttributes attr;
    cudaError_t e = cudaFuncGetAttributes(&attr, kmarlin::kernel_fn(v));
    if (e != cudaSuccess) return (int)e;
  }
  return 0;
}

// Registers per thread of a variant, or -1.
int kmarlin_regs(int variant) {
  const void* f = kmarlin::kernel_fn(variant);
  if (!f) return -1;
  cudaFuncAttributes attr;
  return cudaFuncGetAttributes(&attr, f) == cudaSuccess ? attr.numRegs : -1;
}

// Marlin's thread_n for this problem at thread_m_blocks = 1 (small-batch config order
// {128,128,256}, {64,128,128}, {128,64,128}; marlin.cu overrides to {128,64,128} when
// N/128*4 <= sms), or -1 if this kernel does not reproduce it.
int kmarlin_marlin_tile_n(int N, int K, int sms) {
  if (K % 128 == 0 && N % 128 == 0) {
    if ((N / 128) * 4 <= sms) return 64;  // the {128, 64, 128} override
    return 128;
  }
  if (K % 128 == 0 && N % 64 == 0) return 64;  // {64,128,*} needs N % 128: skipped by Marlin
  return -1;
}

// out[M, N] = Marlin NVFP4 GEMM of A[M, K]. sms = the SM count Marlin uses as its grid
// (cudaDevAttrMultiProcessorCount). Returns 0, a cudaError_t value, or a negative code
// for unsupported arguments.
int kmarlin_gemm(const void* A, long long lda, int M, const void* B, const void* S,
                 const void* GS, void* C, long long ldc, int N, int K, int sms, int variant,
                 void* stream) {
  if (M < 1 || M > 8) return -1;
  if (N <= 0 || K <= 0) return -2;
  if (sms <= 0) return -3;
  if (lda % 8 || (reinterpret_cast<uintptr_t>(A) & 15)) return -4;
  const int tile_n = kmarlin_marlin_tile_n(N, K, sms);
  if (tile_n < 0) return -2;
  kmarlin::LaunchFn fn = kmarlin::launch_fn(variant);
  if (!fn) return -5;
  kmarlin::Params p;
  p.A = static_cast<const __nv_bfloat16*>(A);
  p.lda = lda;
  p.M = M;
  p.B = static_cast<const uint4*>(B);
  p.S = static_cast<const uint2*>(S);
  p.GS = static_cast<const float*>(GS);
  p.C = static_cast<__nv_bfloat16*>(C);
  p.ldc = ldc;
  p.N = N;
  p.K = K;
  p.units_per_tile = tile_n / 64;
  p.n_tiles = N / tile_n;
  p.k_tiles = K / 128;
  // Marlin's stripe partition with parallel = 1 and grid = sms (marlin_template.h).
  int part2 = p.n_tiles;
  if (p.n_tiles > sms) {
    part2 = p.n_tiles % sms;
    if (part2 * 3 <= sms) part2 += sms;
  }
  p.part1_cols = p.n_tiles - part2;
  p.iters = (p.k_tiles * part2 + sms - 1) / sms;
  // group_blocks (1) < thread_k_blocks (8): Marlin does not round iters.
  return fn(p, static_cast<cudaStream_t>(stream));
}

}  // extern "C"
