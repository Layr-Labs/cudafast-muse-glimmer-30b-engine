// SPDX-License-Identifier: Apache-2.0
// kfp8: CUTLASS-exact FP8 w8a8 (per-tensor scales) GEMV/GEMM for 1 <= M <= 16 rows.
//
// It computes what vLLM v0.27.1's ops.cutlass_scaled_mm computes on sm_120/sm_121 for
// M <= 16 (scaled_mm_sm120_fp8_dispatch.cuh, config M16: tile 16x64x128, pingpong, one CTA
// per output tile walking all of K, no split-K), from the SAME tensors (no weight repack):
//   1. the original E4M3 bytes of A [M,K] (row-major) and B [K,N] (column-major) enter native
//      mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 with the standard
//      fragment layout, the 32 K values of each MMA being K[32s, 32s+32);
//   2. every output starts at +0.0f and follows ONE ascending-K chain s = 0..K/32-1, the
//      accumulator fed back as C each step (no split-K, no cross-warp reduction);
//   3. epilogue = ScaledEpilogue: (acc * scale_b) * scale_a in fp32 RN, then RN to bf16.
// What differs from CUTLASS is only the schedule: one warp owns 8 output columns (one n8
// MMA tile) for the whole K, loads straight from global memory through a register ring,
// and the grid has N/8 warps, so the whole GEMV is one wave on 48 (GB10) or 170 (5090) SMs.
// Lineage: the single ascending-K m16n8k32 chain and the epilogue order are those of
// ivanfioravanti's fp8small (Nemotron CUDA track, submission 31d30b53); the PDL launch follows
// its PDL variant in submission 0852152c (kshitij-hash). This version reads the checkpoint
// weight in place (no pack), covers M <= 16, and adds optional L2 hints. No fast math.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace kfp8 {

struct P {
  const uint8_t* a;   // [M, lda] e4m3
  const uint8_t* b;   // [K, N] column-major e4m3: column n is K contiguous bytes at b + n*K
  const float* sa;    // scalar
  const float* sb;    // scalar
  __nv_bfloat16* c;   // [M, ldc] bf16
  long long lda, ldc;
  int m, n;
};

// Variant bits (all give the same values; only memory hints / launch differ).
enum : int {
  V_EVICT_FIRST = 1,   // weight loads carry an L2::evict_first cache policy
  V_L1_NOALLOC = 2,    // weight loads use L1::no_allocate
  V_PREFETCH = 4,      // prefetch.global.L2 of the weight row D steps ahead
  V_PDL = 8,           // programmatic dependent launch: weight prologue before griddepcontrol.wait
  V_R8 = 16,           // 8-deep ring instead of 16
};

__device__ __forceinline__ uint64_t evict_first_policy() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

template <int V>
__device__ __forceinline__ uint32_t ldb(const uint8_t* p, uint64_t pol) {
  uint32_t v;
  if constexpr ((V & V_EVICT_FIRST) && (V & V_L1_NOALLOC)) {
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.u32 %0, [%1], %2;"
                 : "=r"(v) : "l"(p), "l"(pol));
  } else if constexpr (V & V_EVICT_FIRST) {
    asm volatile("ld.global.nc.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(v) : "l"(p), "l"(pol));
  } else if constexpr (V & V_L1_NOALLOC) {
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
  } else {
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  }
  return v;
}

// Activation word: an ordinary (coherent) global load. Under PDL it is issued only after
// griddepcontrol.wait, whose "memory" clobber keeps it there.
__device__ __forceinline__ uint32_t lda32(const uint8_t* p) {
  uint32_t v;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void mma(const uint32_t (&a)[4], const uint32_t (&b)[2], float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

constexpr int W = 4;          // warps per CTA
constexpr int PF_DIST = 32;   // prefetch distance in K steps (32 B each) = 1 KiB ahead

// K: reduction length (multiple of 32*R). M16: rows 8..15 may be live.
template <int K, bool M16, int V>
__global__ void __launch_bounds__(W * 32) gemv(P p) {
  constexpr int R = (V & V_R8) ? 8 : 16;
  constexpr int S = K / 32;
  static_assert(K % 32 == 0 && S % R == 0, "K must be a multiple of 32*R");
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = blockIdx.x * W + warp;      // 8-column group
  if (g * 8 >= p.n) return;                 // no block-wide sync below: safe
  const int r = lane >> 2, q = lane & 3;
  // B fragment of m16n8k32 (col): lane holds column n0 + r, K bytes 4q..4q+3 and 16+4q..+3.
  const uint8_t* wrow = p.b + (long long)(g * 8 + r) * K + 4 * q;
  // A fragment (row): rows r and r+8, the same K bytes.
  const bool r0 = r < p.m, r1 = M16 && (r + 8 < p.m);
  const uint8_t* arow0 = p.a + (long long)r * p.lda + 4 * q;
  const uint8_t* arow1 = p.a + (long long)(r + 8) * p.lda + 4 * q;
  uint64_t pol = 0;
  if constexpr (V & V_EVICT_FIRST) pol = evict_first_policy();

  float c[4] = {0.f, 0.f, 0.f, 0.f};
  uint32_t a[R][4], b[R][2];

  auto load_b = [&](int j, int s) {
    b[j][0] = ldb<V>(wrow + 32 * s, pol);
    b[j][1] = ldb<V>(wrow + 32 * s + 16, pol);
  };
  auto load_a = [&](int j, int s) {
    a[j][0] = r0 ? lda32(arow0 + 32 * s) : 0u;
    a[j][2] = r0 ? lda32(arow0 + 32 * s + 16) : 0u;
    a[j][1] = r1 ? lda32(arow1 + 32 * s) : 0u;
    a[j][3] = r1 ? lda32(arow1 + 32 * s + 16) : 0u;
  };
  auto prefetch = [&](int s) {
    if constexpr (V & V_PREFETCH) {
      // one 128-byte line (4 steps) of this lane's column, issued by q == 0 every 4th step
      if (q == 0 && (s & 3) == 0 && s + PF_DIST < S)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(wrow + 32 * (s + PF_DIST)));
    }
  };

#pragma unroll
  for (int j = 0; j < R; ++j) load_b(j, j);
  if constexpr (V & V_PDL) pdl_wait();
#pragma unroll
  for (int j = 0; j < R; ++j) load_a(j, j);

#pragma unroll 1
  for (int base = 0; base < S; base += R) {
#pragma unroll
    for (int j = 0; j < R; ++j) {
      const int s = base + j;
      mma(a[j], b[j], c);
      prefetch(s);
      if (s + R < S) {
        load_b(j, s + R);
        load_a(j, s + R);
      }
    }
  }
  if constexpr (V & V_PDL) pdl_launch_dependents();

  const float sa = __ldg(p.sa), sb = __ldg(p.sb);
  const int col = g * 8 + 2 * q;
  // C fragment: c0,c1 = row r, cols 2q,2q+1; c2,c3 = row r+8.
  if (r0) {
    __nv_bfloat162 v;
    v.x = __float2bfloat16_rn(__fmul_rn(__fmul_rn(c[0], sb), sa));
    v.y = __float2bfloat16_rn(__fmul_rn(__fmul_rn(c[1], sb), sa));
    *reinterpret_cast<__nv_bfloat162*>(p.c + (long long)r * p.ldc + col) = v;
  }
  if (r1) {
    __nv_bfloat162 v;
    v.x = __float2bfloat16_rn(__fmul_rn(__fmul_rn(c[2], sb), sa));
    v.y = __float2bfloat16_rn(__fmul_rn(__fmul_rn(c[3], sb), sa));
    *reinterpret_cast<__nv_bfloat162*>(p.c + (long long)(r + 8) * p.ldc + col) = v;
  }
}

template <int K, bool M16, int V>
cudaError_t launch(const P& p, cudaStream_t st) {
  const int groups = p.n / 8;
  dim3 grid((groups + W - 1) / W), block(W * 32);
  if constexpr (V & V_PDL) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaError_t rc = cudaLaunchKernelEx(&cfg, gemv<K, M16, V>, p);
    if (rc != cudaSuccess) {
      cudaGetLastError();
      return rc;
    }
  } else {
    gemv<K, M16, V><<<grid, block, 0, st>>>(p);
  }
  return cudaGetLastError();
}

template <int K, int V>
cudaError_t launch_k(const P& p, cudaStream_t st) {
  return p.m <= 8 ? launch<K, false, V>(p, st) : launch<K, true, V>(p, st);
}

template <int V>
cudaError_t launch_v(const P& p, int k, cudaStream_t st) {
  switch (k) {
    case 6656: return launch_k<6656, V>(p, st);
    case 4096: return launch_k<4096, V>(p, st);
    default: return cudaErrorInvalidValue;
  }
}

template <int V>
cudaError_t attrs_v(int k, int m16, cudaFuncAttributes* a) {
  switch (k) {
    case 6656: return m16 ? cudaFuncGetAttributes(a, gemv<6656, true, V>) : cudaFuncGetAttributes(a, gemv<6656, false, V>);
    case 4096: return m16 ? cudaFuncGetAttributes(a, gemv<4096, true, V>) : cudaFuncGetAttributes(a, gemv<4096, false, V>);
    default: return cudaErrorInvalidValue;
  }
}

// The variants compiled in. KFP8_ALL_VARIANTS adds the research set.
#ifdef KFP8_ALL_VARIANTS
#define KFP8_VARIANTS(X) X(0) X(1) X(2) X(3) X(4) X(5) X(8) X(9) X(13) X(16) X(17) X(20) X(24) X(25)
#else
#define KFP8_VARIANTS(X) X(0) X(8) X(16) X(24)
#endif

}  // namespace kfp8

extern "C" {

// Returns 1 when (k, variant) is built in.
int kfp8_supported(int k, int variant) {
  if (k != 6656 && k != 4096) return 0;
  switch (variant) {
#define KFP8_CASE(v) case v: return 1;
    KFP8_VARIANTS(KFP8_CASE)
#undef KFP8_CASE
    default: return 0;
  }
}

// Loads every built kernel into the current context (call outside graph capture).
// Returns 0, or a cudaError_t.
int kfp8_init() {
  cudaFuncAttributes a;
  const int ks[2] = {6656, 4096};
  for (int i = 0; i < 2; ++i)
    for (int m16 = 0; m16 < 2; ++m16) {
      cudaError_t rc = cudaSuccess;
#define KFP8_INIT(v) if (rc == cudaSuccess) rc = kfp8::attrs_v<v>(ks[i], m16, &a);
      KFP8_VARIANTS(KFP8_INIT)
#undef KFP8_INIT
      if (rc != cudaSuccess) return (int)rc;
    }
  return 0;
}

// Registers per thread of one instantiation (diagnostics), or -1.
int kfp8_regs(int k, int m16, int variant) {
  cudaFuncAttributes a;
  cudaError_t rc = cudaErrorInvalidValue;
  switch (variant) {
#define KFP8_REGS(v) case v: rc = kfp8::attrs_v<v>(k, m16, &a); break;
    KFP8_VARIANTS(KFP8_REGS)
#undef KFP8_REGS
    default: break;
  }
  return rc == cudaSuccess ? a.numRegs : -1;
}

// out[M, N] (bf16, row stride ldc) = bf16_rn(((A[M,K] . B[K,N]) * sb) * sa).
// Preconditions (checked by the caller): 1 <= m <= 16, n % 8 == 0, K in {6656, 4096},
// a, b, c 16-byte aligned, lda % 16 == 0, ldc % 2 == 0, b column-major with ld K.
int kfp8_gemm(const void* a, long long lda, const void* b, const void* sa, const void* sb,
              void* c, long long ldc, int m, int n, int k, int variant, void* stream) {
  if (m < 1 || m > 16 || n < 8 || n % 8 != 0) return (int)cudaErrorInvalidValue;
  kfp8::P p{(const uint8_t*)a, (const uint8_t*)b, (const float*)sa, (const float*)sb,
            (__nv_bfloat16*)c, lda, ldc, m, n};
  cudaStream_t st = (cudaStream_t)stream;
  switch (variant) {
#define KFP8_RUN(v) case v: return (int)kfp8::launch_v<v>(p, k, st);
    KFP8_VARIANTS(KFP8_RUN)
#undef KFP8_RUN
    default: return (int)cudaErrorInvalidValue;
  }
}

}  // extern "C"
