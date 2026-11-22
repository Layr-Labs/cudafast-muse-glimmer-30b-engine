// SPDX-License-Identifier: Apache-2.0
// fp8ws: zero-VRAM dual-TMA M=512 prefill FP8 GEMM for MuseGlimmer-30B-NVFP4,
// C[512,N] = A[512,K] x W[N,K]^T, BF16 out, across all 5 MuseGlimmer FP8 shapes:
// (4608,6656), (4096,6656), (6656,4096), (39936,6656), (6656,19968).
//
// Arithmetic: one ascending-K chain of mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
// from +0.0f (no split-K), then bf16_rn(fmul(fmul(acc, scale_b), scale_a)).
// Dual 2-D TMA (tmA on A[M,K], tmB directly on column-major B[K,N] == row-major W[N,K])
// with SWIZZLE_128B: zero extra weight VRAM and zero load-time packing.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace fp8ws {
constexpr int M = 512;
constexpr int KT = 128, ST = 2;
constexpr int WM = 4, WN = 2, NW = 8;
constexpr int BN = WN * NW * 8;        // 128
constexpr int NWARP = WM * WN, NT = NWARP * 32;
constexpr int WSZ = BN * KT;           // 16384 B

template <int N, int K>
struct Shape {
  static constexpr int ST = (N <= 4608) ? 3 : 2;
  static constexpr int MW = (N <= 4608) ? 2 : 4;
  static constexpr int BM = WM * MW * 16;          // 128 for N<=4608, 256 otherwise
  static constexpr int NB = M / BM;                // 4 bands for BM=128, 2 for BM=256
  static constexpr int ASZ = BM * KT;              // 16384 or 32768 B
  static constexpr int STB = ASZ + WSZ;            // 32768 or 49152 B
  static constexpr int SMEM = ST * STB + 1024;     // 99328 B <= 99 KiB
  static_assert(M % BM == 0 && STB % 1024 == 0, "stage");
  static_assert(N % BN == 0 && K % KT == 0, "shape");
  static constexpr int NTN = N / BN, NTILES = NB * NTN, NKT = K / KT;
};

__host__ __device__ constexpr int swz(int r, int c) { return c ^ (r & 7); }

__device__ __forceinline__ void mbar_init(uint32_t a, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, int tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, int parity) {
  asm volatile(
      "{\n .reg .pred p;\n W_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra W_%=;\n}\n" ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      ::"r"(dst), "l"(map), "r"(x), "r"(y), "r"(bar) : "memory");
}
__device__ __forceinline__ void ldsm4(uint32_t a, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma(const uint32_t (&a)[4], uint32_t b0, uint32_t b1, float (&c)[4]) {
  asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ uint32_t cvt2(float lo, float hi, float sa, float sb) {
  float x = __fmul_rn(__fmul_rn(lo, sb), sa);
  float y = __fmul_rn(__fmul_rn(hi, sb), sa);
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(x), "f"(y));
  return r;
}
__device__ __forceinline__ uint32_t selp(uint32_t a, uint32_t b, bool c) {
  uint32_t r;
  asm("{\n .reg .pred p;\n setp.ne.b32 p, %3, 0;\n selp.b32 %0, %1, %2, p;\n}"
      : "=r"(r) : "r"(a), "r"(b), "r"((uint32_t)c));
  return r;
}
__device__ __forceinline__ uint32_t sel4(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, int i) {
  const uint32_t lo = selp(v1, v0, i & 1), hi = selp(v3, v2, i & 1);
  return selp(hi, lo, i & 2);
}

struct Params {
  const float* sa;
  const float* sb;
  __nv_bfloat16* out;
};

struct ParamsDual {
  const float* sa;
  const float* sb0;
  const float* sb1;
  __nv_bfloat16* out0;
  __nv_bfloat16* out1;
};

template <int N0, int N1, int K>
struct ShapeDual {
  static constexpr int MW = 2;
  static constexpr int BM = WM * MW * 16;          // 128
  static constexpr int NB = M / BM;                // 4 bands
  static constexpr int ASZ = BM * KT;              // 16384 B
  static constexpr int STB = ASZ + WSZ;            // 32768 B
  static constexpr int SMEM = ST * STB + 1024;     // 66560 B <= 99 KiB
  static_assert(M % BM == 0 && STB % 1024 == 0, "stage");
  static_assert(N0 % BN == 0 && N1 % BN == 0 && K % KT == 0, "shape");
  static constexpr int NTN0 = N0 / BN, NTN1 = N1 / BN, NTN = NTN0 + NTN1;
  static constexpr int NTILES = NB * NTN, NKT = K / KT;
};

template <int N>
__device__ __forceinline__ void epilogue(const float (&acc)[NW][4], __nv_bfloat16* out, int row0, int col0,
                                         int lane, float sa, float sb) {
  const int q = lane & 3, rq = lane >> 2;
#pragma unroll
  for (int pp = 0; pp < 4; ++pp) {
    const int j = 2 * pp;
    const uint32_t v0 = cvt2(acc[j][0], acc[j][1], sa, sb);
    const uint32_t v1 = cvt2(acc[j][2], acc[j][3], sa, sb);
    const uint32_t v2 = cvt2(acc[j + 1][0], acc[j + 1][1], sa, sb);
    const uint32_t v3 = cvt2(acc[j + 1][2], acc[j + 1][3], sa, sb);
    const uint32_t r1 = __shfl_xor_sync(0xffffffffu, sel4(v0, v1, v2, v3, q ^ 1), 1);
    const uint32_t r2 = __shfl_xor_sync(0xffffffffu, sel4(v0, v1, v2, v3, q ^ 2), 2);
    const uint32_t r3 = __shfl_xor_sync(0xffffffffu, sel4(v0, v1, v2, v3, q ^ 3), 3);
    const uint32_t own = sel4(v0, v1, v2, v3, q);
    uint4 w;
    w.x = sel4(own, r1, r2, r3, 0 ^ q);
    w.y = sel4(own, r1, r2, r3, 1 ^ q);
    w.z = sel4(own, r1, r2, r3, 2 ^ q);
    w.w = sel4(own, r1, r2, r3, 3 ^ q);
    const int row = row0 + rq + (q & 1) * 8, col = col0 + (j + (q >> 1)) * 8;
    *reinterpret_cast<uint4*>(out + (size_t)row * N + col) = w;
  }
}

template <int MW>
__device__ __forceinline__ void consume_stage(uint32_t abase, uint32_t wbase, int wm, int wn, int ar, int ach,
                                              int br, int bch, float (&acc)[MW][NW][4]) {
#pragma unroll
  for (int ks = 0; ks < KT / 32; ++ks) {
    uint32_t a[MW][4], b[NW][2];
#pragma unroll
    for (int i = 0; i < MW; ++i) {
      const int r = wm * MW * 16 + i * 16 + ar;
      ldsm4(abase + r * KT + swz(r, 2 * ks + ach) * 16, a[i]);
    }
#pragma unroll
    for (int j = 0; j < NW / 2; ++j) {
      const int r = wn * NW * 8 + j * 16 + br;
      uint32_t qd[4];
      ldsm4(wbase + r * KT + swz(r, 2 * ks + bch) * 16, qd);
      b[2 * j][0] = qd[0];
      b[2 * j][1] = qd[1];
      b[2 * j + 1][0] = qd[2];
      b[2 * j + 1][1] = qd[3];
    }
#pragma unroll
    for (int i = 0; i < MW; ++i)
#pragma unroll
      for (int j = 0; j < NW; ++j) mma(a[i], b[j][0], b[j][1], acc[i][j]);
  }
}

template <int N, int K>
__global__ void __launch_bounds__(NT, 1)
gemm(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB, const Params p) {
  using S = Shape<N, K>;
  constexpr int ST = S::ST;
  constexpr int MW = S::MW, BM = S::BM, NB = S::NB, ASZ = S::ASZ, STB = S::STB;
  constexpr int NTILES = S::NTILES, NKT = S::NKT;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST;
  const int grid = gridDim.x;
  if ((int)blockIdx.x >= NTILES) return;
  const int nmine = (NTILES - blockIdx.x + grid - 1) / grid;
  const int total = nmine * NKT;
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  auto produce = [&](int gl) {
    const int s = gl % ST, n = gl / ST;
    if (gl >= ST) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
    const int ti = gl / NKT, kt = gl - ti * NKT;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
    mbar_expect_tx(bar, STB);
    tma2d(dst, &tmA, kt * KT, band * BM, bar);
    tma2d(dst + ASZ, &tmB, kt * KT, nt * BN, bar);
  };
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
    for (int s = 0; s < ST - 1 && s < total; ++s) produce(s);
  }
  const int wm = warp % WM, wn = warp / WM;
  const float sa = __ldg(p.sa), sb = __ldg(p.sb);
  float acc[MW][NW][4];
#pragma unroll
  for (int i = 0; i < MW; ++i)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  int kt = 0, ti = 0;
  for (int g = 0; g < total; ++g) {
    const int s = g % ST, n = g / ST;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const int rb0 = band * BM + wm * MW * 16;
    if (tid == 0 && g + ST - 1 < total) produce(g + ST - 1);
    mbar_wait(full0 + 8 * s, n & 1);
    const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
    consume_stage<MW>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * s);
    if (++kt == NKT) {
      kt = 0;
      ++ti;
      const int col0 = nt * BN + wn * NW * 8;
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        epilogue<N>(acc[i], p.out, rb0 + i * 16, col0, lane, sa, sb);
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
      }
    }
  }
}

template <int N0, int N1, int K>
__global__ void __launch_bounds__(NT, 1)
gemm_dual(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB0,
          const __grid_constant__ CUtensorMap tmB1, const ParamsDual p) {
  using S = ShapeDual<N0, N1, K>;
  constexpr int MW = S::MW, BM = S::BM, NB = S::NB, ASZ = S::ASZ, STB = S::STB;
  constexpr int NTN0 = S::NTN0, NTILES = S::NTILES, NKT = S::NKT;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST;
  const int grid = gridDim.x;
  if ((int)blockIdx.x >= NTILES) return;
  const int nmine = (NTILES - blockIdx.x + grid - 1) / grid;
  const int total = nmine * NKT;
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  auto produce = [&](int gl) {
    const int s = gl % ST, n = gl / ST;
    if (gl >= ST) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
    const int ti = gl / NKT, kt = gl - ti * NKT;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
    mbar_expect_tx(bar, STB);
    tma2d(dst, &tmA, kt * KT, band * BM, bar);
    if (nt < NTN0) {
      tma2d(dst + ASZ, &tmB0, kt * KT, nt * BN, bar);
    } else {
      tma2d(dst + ASZ, &tmB1, kt * KT, (nt - NTN0) * BN, bar);
    }
  };
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB0) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB1) : "memory");
    produce(0);
  }
  const int wm = warp % WM, wn = warp / WM;
  const float sa = __ldg(p.sa), sb0 = __ldg(p.sb0), sb1 = __ldg(p.sb1);
  float acc[MW][NW][4];
#pragma unroll
  for (int i = 0; i < MW; ++i)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  int kt = 0, ti = 0;
  for (int g = 0; g < total; ++g) {
    const int s = g % ST, n = g / ST;
    const int t = blockIdx.x + ti * grid, nt = t / NB, band = t - nt * NB;
    const int rb0 = band * BM + wm * MW * 16;
    if (tid == 0 && g + 1 < total) produce(g + 1);
    mbar_wait(full0 + 8 * s, n & 1);
    const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
    consume_stage<MW>(abase, wbase, wm, wn, ar, ach, br, bch, acc);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * s);
    if (++kt == NKT) {
      kt = 0;
      ++ti;
      const bool is1 = nt >= NTN0;
      const int col0 = (is1 ? (nt - NTN0) : nt) * BN + wn * NW * 8;
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        if (is1) {
          epilogue<N1>(acc[i], p.out1, rb0 + i * 16, col0, lane, sa, sb1);
        } else {
          epilogue<N0>(acc[i], p.out0, rb0 + i * 16, col0, lane, sa, sb0);
        }
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
      }
    }
  }
}

typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                             const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                             CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn encode = nullptr;
static int num_sms = 0;

static bool encode_tma2d(CUtensorMap* tm, const void* ptr, int k, int rows, int box_rows) {
  cuuint64_t dims[2] = {(cuuint64_t)k, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)k};
  cuuint32_t box[2] = {(cuuint32_t)KT, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1, 1};
  return encode(tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims, strides, box, es,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <int N, int K>
static cudaError_t launch_shape(const CUtensorMap& tmA, const CUtensorMap& tmB, const Params& p, cudaStream_t st) {
  using S = Shape<N, K>;
  constexpr int T = S::NTILES;
  gemm<N, K><<<num_sms < T ? num_sms : T, NT, S::SMEM, st>>>(tmA, tmB, p);
  return cudaGetLastError();
}

template <int N0, int N1, int K>
static cudaError_t launch_dual(const CUtensorMap& tmA, const CUtensorMap& tmB0, const CUtensorMap& tmB1,
                               const ParamsDual& p, cudaStream_t st) {
  using S = ShapeDual<N0, N1, K>;
  constexpr int T = S::NTILES;
  gemm_dual<N0, N1, K><<<num_sms < T ? num_sms : T, NT, S::SMEM, st>>>(tmA, tmB0, tmB1, p);
  return cudaGetLastError();
}
}  // namespace fp8ws

extern "C" int fp8ws_init() {
  using namespace fp8ws;
  if (!encode) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
    if (e != cudaSuccess || !fn || q != cudaDriverEntryPointSuccess) return e != cudaSuccess ? (int)e : -2;
    encode = (EncodeFn)fn;
  }
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e == cudaSuccess) e = cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
#define FP8WS_ATTR(N, K)                                                                                       \
  if (e == cudaSuccess)                                                                                        \
    e = cudaFuncSetAttribute(gemm<N, K>, cudaFuncAttributeMaxDynamicSharedMemorySize, Shape<N, K>::SMEM)
  FP8WS_ATTR(4608, 6656);
  FP8WS_ATTR(4096, 6656);
  FP8WS_ATTR(6656, 4096);
  FP8WS_ATTR(39936, 6656);
  FP8WS_ATTR(6656, 19968);
#undef FP8WS_ATTR
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(gemm_dual<4608, 4096, 6656>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             ShapeDual<4608, 4096, 6656>::SMEM);
  }
  return (int)e;
}

extern "C" int fp8ws_run(const void* a, const void* b, const void* sa, const void* sb, void* c, int m, int n,
                         int k, void* stream) {
  using namespace fp8ws;
  if (m != M || !encode || num_sms <= 0) return (int)cudaErrorInvalidValue;
  if (((uintptr_t)a & 15) || ((uintptr_t)b & 15) || ((uintptr_t)c & 15)) return (int)cudaErrorInvalidValue;
  const int bm = (n <= 4608) ? 128 : 256;
  CUtensorMap tmA, tmB;
  if (!encode_tma2d(&tmA, a, k, M, bm)) return -3;
  if (!encode_tma2d(&tmB, b, k, n, BN)) return -4;
  Params p{(const float*)sa, (const float*)sb, (__nv_bfloat16*)c};
  cudaStream_t st = (cudaStream_t)stream;
  if (n == 4608 && k == 6656) return (int)launch_shape<4608, 6656>(tmA, tmB, p, st);
  if (n == 4096 && k == 6656) return (int)launch_shape<4096, 6656>(tmA, tmB, p, st);
  if (n == 6656 && k == 4096) return (int)launch_shape<6656, 4096>(tmA, tmB, p, st);
  if (n == 39936 && k == 6656) return (int)launch_shape<39936, 6656>(tmA, tmB, p, st);
  if (n == 6656 && k == 19968) return (int)launch_shape<6656, 19968>(tmA, tmB, p, st);
  return (int)cudaErrorInvalidValue;
}

extern "C" int fp8ws_run_dual(const void* a, const void* b0, const void* b1, const void* sa, const void* sb0,
                              const void* sb1, void* c0, void* c1, int m, int n0, int n1, int k, void* stream) {
  using namespace fp8ws;
  if (m != M || n0 != 4608 || n1 != 4096 || k != 6656 || !encode || num_sms <= 0)
    return (int)cudaErrorInvalidValue;
  if (((uintptr_t)a & 15) || ((uintptr_t)b0 & 15) || ((uintptr_t)b1 & 15) ||
      ((uintptr_t)c0 & 15) || ((uintptr_t)c1 & 15))
    return (int)cudaErrorInvalidValue;
  CUtensorMap tmA, tmB0, tmB1;
  if (!encode_tma2d(&tmA, a, k, M, 128)) return -3;
  if (!encode_tma2d(&tmB0, b0, k, n0, BN)) return -4;
  if (!encode_tma2d(&tmB1, b1, k, n1, BN)) return -5;
  ParamsDual p{(const float*)sa, (const float*)sb0, (const float*)sb1, (__nv_bfloat16*)c0, (__nv_bfloat16*)c1};
  return (int)launch_dual<4608, 4096, 6656>(tmA, tmB0, tmB1, p, (cudaStream_t)stream);
}
