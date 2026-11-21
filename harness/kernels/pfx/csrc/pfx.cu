// pfx: a Marlin-exact NVFP4 W4A16 GEMM for the 512-row prompt pass of Muse Glimmer 30B.
//
// What stock Marlin computes there (vLLM v0.27.1 marlin_mm / marlin_template.h): one launch of
// Marlin<bf16, fe2m1f, bf16, fe4m3fn, 256 threads, thread_m_blocks 4, thread_n_blocks 16,
// thread_k_blocks 4, stages 4, group_blocks 1> over grid = #SMs (48), `parallel` = 8 sub-problems
// of 64 rows, fp32 global reduce, no atomics. Per output element (row r, feature n):
//   * The tile (64-row block of r, 256-feature column of n) is cut by Marlin's stripe partition
//     into k-tile (64 K) segments; one Marlin CTA reduces one segment.
//   * Inside a segment, k-warp group g = 0, 1 (warps 4g..4g+3) runs one fp32 chain from +0 of
//     mma.m16n8k16.row.col.f32.bf16.bf16.f32 (activations as A, dequantized weights as B) over the
//     k16 blocks 4t + 2g, 4t + 2g + 1 for the segment's tiles t ascending.
//   * P = acc_g0 + acc_g1 (thread_block_reduce).
//   * global_reduce_fp32 in lock order: R = P of the highest-K segment, then R = P + R for each
//     segment going down in K.
//   * out = bf16_rn(R * global_scale) (write_result).
// The host transcribes the partition (py/__init__.py marlin_units) into a unit table: one unit is
// one (64-row block, 256-feature column) tile with its segments in fold order. A persistent CTA
// with Marlin's own warp roles reduces a whole unit (all its segments, the fold in a per-CTA fp32
// scratch), so there are no cross-CTA locks, and the units run column-major: the 8 row blocks of a
// column are in flight together and the weights stream from DRAM once instead of 8 times.
// Dequant, scale multiply, MMA operands and every fp32 add are Marlin's (as harness/kernels in the
// previous track's marlinx/shx, which were verified bit-exact against stock Marlin).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace pfx {

constexpr int kThreads = 256;
constexpr int kStages = 4;
constexpr int kTileN = 256;                       // Marlin thread_n
constexpr int kRows = 64;                         // Marlin m_block_size (thread_m_blocks 4)
constexpr int kAStage = kRows * 64 * 2;           // 64 rows x 64 k bf16 = 8192 B
constexpr int kBStage = 4 * 128 * 16;             // 4 k16 rows x 128 int4 = 8192 B
constexpr int kSStage = 4 * kTileN;               // 4 k16 rows x 256 scale bytes = 1024 B
constexpr int kStageBytes = kAStage + kBStage + kSStage;
constexpr int kRedBytes = 16 * kTileN * 4;        // one m16 block of the tile in fp32
constexpr int kOutStride = kTileN + 8;            // bf16 row stride of the staged output (bank shift)
constexpr int kSmemBytes = kStages * kStageBytes + kRedBytes;
constexpr int kMaxSeg = 4;
constexpr int kUnitInts = 4 + 2 * kMaxSeg;        // row0, col, nseg, 0, lo[4], hi[4] (k-tiles)
constexpr int kScratchFloats = 128 * 128;         // R of one unit: 128 threads x 128 floats

struct Params {
  const __nv_bfloat16* A;  // [M, lda]
  long long lda;
  const uint4* B;          // Marlin fp4 weight, [K/16][N/2] int4
  const uint8_t* S;        // Marlin-processed e4m3 scales, [K/16][N] bytes
  const float* GS;         // Marlin-processed fp32 global scale
  __nv_bfloat16* C;        // [M, ldc]
  long long ldc;
  int N, K;
  const int* units;
  int n_units;
  float* scratch;          // [grid][kScratchFloats]
};

__device__ __forceinline__ __nv_bfloat162 as_bf162(uint32_t v) { return *reinterpret_cast<__nv_bfloat162*>(&v); }
__device__ __forceinline__ uint32_t as_u32(__nv_bfloat162 v) { return *reinterpret_cast<uint32_t*>(&v); }

// Marlin dequant<nv_bfloat162, kFE2M1f, dequant_skip_flop = true>.
__device__ __forceinline__ void dequant_fp4(uint32_t q, uint32_t (&out)[2]) {
  uint32_t o1 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
  q <<= 4;
  uint32_t o2 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
  out[1] = o1;
  out[0] = o2;
}

// Marlin dequant_fp8_scales<nv_bfloat162, kFE4M3fn>.
__device__ __forceinline__ void dequant_scales(uint32_t q, uint32_t (&out)[2]) {
  uint32_t o1 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
  q <<= 8;
  uint32_t o2 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
  out[1] = o1;
  out[0] = o2;
}

// Marlin mma<bf16>: activations as A, weights (one n8 tile) as B.
__device__ __forceinline__ void mma(const uint32_t (&a)[4], const uint32_t (&b)[2], float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
        "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// One k16 block (kl = 0..3 inside the stage) of one warp: 64 rows (4 m16) x 64 features (8 n8),
// exactly Marlin's matmul(k) for this warp.
__device__ __forceinline__ void block_mma(uint32_t sA, uint32_t sB, uint32_t sS, int kl, int nw, int lane,
                                          float (&acc)[4][4][2][4]) {
  uint32_t a[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = 16 * i + (lane & 15);
    const int c = 2 * kl + (lane >> 4);
    ldsm4(a[i], sA + r * 128 + ((c ^ (r & 7)) << 4));
  }
  uint32_t q[4], s0, s1;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3])
               : "r"(sB + (kl * 128 + 32 * nw + lane) * 16));
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(s0), "=r"(s1) : "r"(sS + kl * 256 + 64 * nw + 8 * (lane >> 2)));
  uint32_t fs[4];
  {
    uint32_t t[2];
    dequant_scales(s0, t);
    fs[0] = t[0];
    fs[1] = t[1];
    dequant_scales(s1, t);
    fs[2] = t[0];
    fs[3] = t[1];
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t b0[2], b1[2];
    dequant_fp4(q[j] << 8, b0);  // b_quant_0 = b_quant_1 << 8
    dequant_fp4(q[j], b1);
    const __nv_bfloat162 s = as_bf162(fs[j]);
    const __nv_bfloat162 sx = __bfloat162bfloat162(s.x);
    const __nv_bfloat162 sy = __bfloat162bfloat162(s.y);
    b0[0] = as_u32(__hmul2(as_bf162(b0[0]), sx));
    b0[1] = as_u32(__hmul2(as_bf162(b0[1]), sx));
    b1[0] = as_u32(__hmul2(as_bf162(b1[0]), sy));
    b1[1] = as_u32(__hmul2(as_bf162(b1[1]), sy));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      mma(a[i], b0, acc[i][j][0]);
      mma(a[i], b1, acc[i][j][1]);
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1) gemm_kernel(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nw = warp & 3;  // 64-feature slab of the tile
  const int g = warp >> 2;  // k-warp group
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  float* red = reinterpret_cast<float*>(smem + kStages * kStageBytes);
  const uint32_t sred = sbase + kStages * kStageBytes;
  float* scr = p.scratch + static_cast<long long>(blockIdx.x) * kScratchFloats + (tid & 127);
  const float gsc = __ldg(p.GS);
  const long long nB = p.N >> 1;  // int4 per k16 row

  for (int u = blockIdx.x; u < p.n_units; u += gridDim.x) {
    const int* U = p.units + kUnitInts * u;
    const int row0 = U[0], col = U[1], nseg = U[2];
    int lo[kMaxSeg], len[kMaxSeg];
    int nst = 0;
#pragma unroll
    for (int s = 0; s < kMaxSeg; ++s) {
      lo[s] = U[4 + s];
      len[s] = s < nseg ? U[4 + kMaxSeg + s] - U[4 + s] : 0;
      nst += len[s];
    }
    const uint4* Bu = p.B + 128 * col;
    const uint8_t* Su = p.S + 256 * col;
    const __nv_bfloat16* Au = p.A + static_cast<long long>(row0) * p.lda;

    auto kt_of = [&](int st) -> int {  // k-tile of stage st: segments in fold order, tiles ascending
      int kt = 0;
#pragma unroll
      for (int s = 0; s < kMaxSeg; ++s) {
        if (st >= 0 && st < len[s]) kt = lo[s] + st;
        st -= len[s];
      }
      return kt;
    };
    auto load = [&](int st) {
      const int kt = kt_of(st);
      const uint32_t slot = sbase + (st % kStages) * kStageBytes;
#pragma unroll
      for (int i = 0; i < 2; ++i) {  // A: 64 rows x 8 chunks of 8 k
        const int idx = tid + kThreads * i;
        const int r = idx >> 3, c = idx & 7;
        cp16(slot + r * 128 + ((c ^ (r & 7)) << 4), Au + static_cast<long long>(r) * p.lda + 64 * kt + 8 * c);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {  // B: 4 k16 rows x 128 int4
        const int idx = tid + kThreads * i;
        const int rr = idx >> 7, c = idx & 127;
        cp16(slot + kAStage + idx * 16, Bu + static_cast<long long>(4 * kt + rr) * nB + c);
      }
      if (tid < 64) {  // S: 4 k16 rows x 256 bytes
        const int rr = tid >> 4, c = tid & 15;
        cp16(slot + kAStage + kBStage + rr * 256 + 16 * c, Su + static_cast<long long>(4 * kt + rr) * p.N + 16 * c);
      }
    };

    __syncthreads();  // every warp is done with the previous unit's stages and red
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
      if (s < nst) load(s);
      cp_commit();
    }

    float acc[4][4][2][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[i][j][h][e] = 0.0f;

    int seg = 0, seg_end = len[0];
    for (int st = 0; st < nst; ++st) {
      cp_wait<kStages - 2>();
      __syncthreads();
      if (st + kStages - 1 < nst) load(st + kStages - 1);
      cp_commit();
      const uint32_t slot = sbase + (st % kStages) * kStageBytes;
#pragma unroll
      for (int k = 0; k < 2; ++k)
        block_mma(slot, slot + kAStage, slot + kAStage + kBStage, 2 * g + k, nw, lane, acc);
      if (st + 1 == seg_end) {
        // thread_block_reduce: P = acc_g0 + acc_g1, kept by group 0, one m16 block at a time
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (g == 1) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
              for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int e = 0; e < 4; ++e) red[((j * 2 + h) * 4 + e) * 128 + (tid & 127)] = acc[i][j][h][e];
          }
          __syncthreads();
          if (g == 0) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
              for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                  acc[i][j][h][e] = __fadd_rn(acc[i][j][h][e], red[((j * 2 + h) * 4 + e) * 128 + (tid & 127)]);
          }
          __syncthreads();
        }
        // global_reduce_fp32 in fold order: R = P (first), then R = P + R
        const bool first = seg == 0, last = seg == nseg - 1;
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                const int x = (((i * 4 + j) * 2 + h) * 4 + e) * 128;
                float v = acc[i][j][h][e];
                if (g == 0) {
                  if (!first) v = __fadd_rn(v, scr[x]);
                  if (!last) {
                    scr[x] = v;
                    v = 0.0f;
                  }
                } else {
                  v = 0.0f;
                }
                acc[i][j][h][e] = v;  // group 0 keeps R of the last segment for the epilogue
              }
        ++seg;
        if (seg < kMaxSeg) seg_end += len[seg];
      }
    }
    cp_wait<0>();

    // write_result: bf16_rn(R * global_scale), staged through red for 16-byte row stores
    __nv_bfloat16* Cu = p.C + static_cast<long long>(row0) * p.ldc + kTileN * col;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (g == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int f = 64 * nw + 16 * j + 8 * h + 2 * (lane & 3);
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              const int r = (lane >> 2) + 8 * hh;
              __nv_bfloat162 v;
              v.x = __float2bfloat16_rn(__fmul_rn(acc[i][j][h][2 * hh], gsc));
              v.y = __float2bfloat16_rn(__fmul_rn(acc[i][j][h][2 * hh + 1], gsc));
              asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(sred + (r * kOutStride + f) * 2), "r"(as_u32(v)));
            }
          }
      }
      __syncthreads();
#pragma unroll
      for (int t = 0; t < 2; ++t) {  // 16 rows x 32 int4
        const int idx = tid + kThreads * t;
        const int r = idx >> 5, c = idx & 31;
        uint4 v;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "r"(sred + (r * kOutStride + 8 * c) * 2));
        *reinterpret_cast<uint4*>(Cu + static_cast<long long>(16 * i + r) * p.ldc + 8 * c) = v;
      }
      __syncthreads();
    }
  }
}

}  // namespace pfx

extern "C" {

int pfx_version() { return 1; }

int pfx_smem_bytes() { return pfx::kSmemBytes; }

int pfx_scratch_floats(int grid) { return grid * pfx::kScratchFloats; }

int pfx_unit_ints() { return pfx::kUnitInts; }

int pfx_init() {
  cudaError_t e = cudaFuncSetAttribute(pfx::gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pfx::kSmemBytes);
  return static_cast<int>(e);
}

int pfx_attrs(int* out) {
  cudaFuncAttributes a;
  cudaError_t e = cudaFuncGetAttributes(&a, pfx::gemm_kernel);
  if (e != cudaSuccess) return static_cast<int>(e);
  out[0] = a.numRegs;
  out[1] = static_cast<int>(a.sharedSizeBytes);
  out[2] = a.maxThreadsPerBlock;
  out[3] = static_cast<int>(a.localSizeBytes);
  return 0;
}

// C[M, N] = Marlin NVFP4 GEMM of A[M, K] for the units of py/__init__.py marlin_units.
int pfx_gemm(const void* A, long long lda, const void* B, const void* S, const void* GS, void* C, long long ldc,
             int N, int K, const void* units, int n_units, void* scratch, int grid, void* stream) {
  if (N % 256 || K % 64 || lda % 8 || ldc % 8) return -2;
  if ((reinterpret_cast<uintptr_t>(A) & 15) || (reinterpret_cast<uintptr_t>(C) & 15)) return -4;
  pfx::Params p;
  p.A = static_cast<const __nv_bfloat16*>(A);
  p.lda = lda;
  p.B = static_cast<const uint4*>(B);
  p.S = static_cast<const uint8_t*>(S);
  p.GS = static_cast<const float*>(GS);
  p.C = static_cast<__nv_bfloat16*>(C);
  p.ldc = ldc;
  p.N = N;
  p.K = K;
  p.units = static_cast<const int*>(units);
  p.n_units = n_units;
  p.scratch = static_cast<float*>(scratch);
  pfx::gemm_kernel<<<grid, pfx::kThreads, pfx::kSmemBytes, static_cast<cudaStream_t>(stream)>>>(p);
  return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
