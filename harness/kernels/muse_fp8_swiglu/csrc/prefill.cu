// SPDX-License-Identifier: Apache-2.0
// Scaled accumulator and BF16 conversion follow the G2 scaled-MM epilogue.
// Interleave consecutive gate and up columns. Two four-bit transport lanes
// carry one unmodified E4M3 output byte through the TMA store.
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdlib>
#include <stdint.h>
#include "swiglu.cuh"
template<class Fragment> struct PairSwiGLU {
    CUTLASS_DEVICE auto operator()(Fragment const& halves, Fragment const& scale) const {
        constexpr int Num=Fragment::kElements;
        static_assert(Num>=2 && Num%2==0, "Each register fragment must contain an even adjacent N pair");
        cutlass::Array<cutlass::float_e2m1_t,Num> out;
        #pragma unroll
        for (int i=0;i<Num;i+=2) {
            const float silu=__ldg(bf16_silu + (__float_as_uint(halves[i])>>16));
            float inverse;
            asm("div.full.f32 %0, %1, %2;" : "=f"(inverse) : "f"(1.0f), "f"(scale[i]));
            const float q=__fmul_rn(__fmul_rn(silu,halves[i+1]),inverse);
            const float clamped=q < -448.f ? -448.f : (q>448.f ? 448.f : q);
            unsigned value=__nv_cvt_float_to_fp8(clamped,__NV_SATFINITE,__NV_E4M3);
            out[i]=cutlass::float_e2m1_t::bitcast(value&15);
            out[i+1]=cutlass::float_e2m1_t::bitcast(value>>4);
        }
        return out;
    }
};
extern "C" int init_silu(cudaStream_t stream){swiglu_initialize<<<256,256,0,stream>>>();return int(cudaGetLastError());}

using namespace cute;

template <class TileShape, class EpiTile, class KernelSchedule>
struct Fp8Gemm {
  using ElementAB = cutlass::float_e4m3_t; using Half = cutlass::bfloat16_t; using ElementD = cutlass::float_e2m1_t; using ElementAcc = float;
  using LayoutA = cutlass::layout::RowMajor; using LayoutB = cutlass::layout::ColumnMajor; using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ClusterShape = Shape<_1, _1, _1>;
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using ScaleA = cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>;
  using ScaleB = cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>;
  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, Half, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using HalfOut = cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVT0>;
  using Pack = cutlass::epilogue::fusion::Sm90Compute<PairSwiGLU, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT = cutlass::epilogue::fusion::Sm90EVT<Pack, HalfOut, ScaleA>;
  // With 4-bit raw transport, a single 1-KiB epilogue stage leaves enough SMEM for the regular mainloop depth.
  using EpiPolicy = cutlass::epilogue::Sm120TmaWarpSpecialized<1, 1, 4, false, false>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::detail::Sm120TmaBuilderImpl<
      TileShape, EpiTile, ElementAcc, float, void, LayoutD, AlignD, ElementD, LayoutD, AlignD,
      EVT, EpiPolicy>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementAB, LayoutA, AlignAB, ElementAB, LayoutB, AlignAB,
      ElementAcc, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<3>,
      KernelSchedule, void>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;


  static int run(int M, int N, int K, const void* A, const void* B,
                 const float* sa, const float* sb, const float* down, void* D,
                 void* workspace, size_t capacity, int device, int sms, cudaStream_t stream) {
    using StrideA = typename GemmKernel::StrideA; using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC; using StrideD = typename GemmKernel::StrideD;
    StrideA sA=cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M,K,1));
    StrideB sB=cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N,K,1));
    StrideC sC=cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M,N,1));
    StrideD sD=cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M,N,1));
    typename HalfOut::Arguments half{{{}, {sa}, {}}, {{{}, {sb}, {}}, {}, {}}, {}};
    typename EVT::Arguments evt{half,{{}, {down}, {}},{}};
    cutlass::KernelHardwareInfo hw;
    hw.device_id=device;
    hw.sm_count=sms;
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K,1},
      {static_cast<ElementAB const*>(A),sA,static_cast<ElementAB const*>(B),sB},
      {evt,nullptr,sC,static_cast<ElementD*>(D),sD},hw};
    auto status=Gemm::can_implement(args);
    if(status!=cutlass::Status::kSuccess) return 100+int(status);
    if(Gemm::get_workspace_size(args)>capacity) return -2;
    Gemm op;
    status=op.initialize(args,workspace,stream);
    if(status!=cutlass::Status::kSuccess) return 200+int(status);
    return int(op.run(stream));
  }
};
using Schedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Main = Fp8Gemm<Shape<_128, _128, _128>, Shape<_64, _32>, Schedule>;
static_assert(Main::GemmKernel::SharedStorageSize == 101376);
extern "C" int prefill(int m, int n, int k, const void* a,const void* b,
  const float* sa,const float* sb,const float* down,void* d,void* workspace,size_t capacity,int device,int sms,cudaStream_t stream){
  return Main::run(m,n,k,a,b,sa,sb,down,d,workspace,capacity,device,sms,stream);
}

extern "C" size_t shared_bytes() { return Main::GemmKernel::SharedStorageSize; }

// Wave-31b: Zero-VRAM Dual-TMA 256x128 (13 exact 48-SM waves, AlongM) M=512 FP8 SwiGLU kernel.
namespace swiglu_ws {
constexpr int M_DIM = 512, N_DIM = 39936, OUT_COLS = 19968, K_DIM = 6656, KT = 128;
constexpr int WM = 4, NW = 8, BN = 128, NWARP = 8, NT = 256, WSZ = BN * KT, SMEM_MAX = 99328;

__device__ int ws_silu_diffs;
__global__ void ws_silu_check() {
  const unsigned word = 256u * blockIdx.x + threadIdx.x;
  const float direct = swiglu_silu(__uint_as_float(word << 16));
  if (__float_as_uint(direct) != __float_as_uint(bf16_silu[word])) atomicAdd(&ws_silu_diffs, 1);
}

__host__ __device__ constexpr int swz(int r, int c) { return c ^ (r & 7); }
__device__ __forceinline__ void mbar_init(uint32_t a, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, int tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx_only(uint32_t a, int tx) {
  asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(tx) : "memory");
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;"); }
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, int parity) {
  asm volatile(
      "{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n"
      ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ uint64_t tma_pol_el() {
  uint64_t p; asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ uint64_t tma_pol_ef() {
  uint64_t p; asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ void tma2d_hint(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;"
      ::"r"(dst), "l"(map), "r"(x), "r"(y), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void ldsm4(uint32_t a, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_fp8(const uint32_t (&a)[4], uint32_t b0, uint32_t b1, float (&c)[4]) {
  asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
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

__device__ __forceinline__ float eval_pair_scaled(float g_acc, float u_acc, float sa, float sb, float inv_down) {
  const float g_f = __fmul_rn(__fmul_rn(g_acc, sb), sa);
  const float u_f = __fmul_rn(__fmul_rn(u_acc, sb), sa);
  uint32_t packed_bf16;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(packed_bf16) : "f"(g_f), "f"(u_f));
  const float g_bf = __uint_as_float(packed_bf16 << 16);
  const float u_bf = __uint_as_float(packed_bf16 & 0xffff0000u);
  const float silu = swiglu_silu(g_bf);
  return __fmul_rn(__fmul_rn(silu, u_bf), inv_down);
}

__device__ __forceinline__ void epilogue_swiglu_fp8(const float (&acc)[NW][4], uint8_t* out, int row0, int out_col0,
                                                    int lane, float sa, float sb, float inv_down) {
  const int q = lane & 3, rq = lane >> 2;
  uint32_t* out32 = reinterpret_cast<uint32_t*>(out);
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int j0 = 4 * half;
    uint32_t b_lo[4], b_hi[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const float q_lo = eval_pair_scaled(acc[j0 + m][0], acc[j0 + m][1], sa, sb, inv_down);
      const float q_hi = eval_pair_scaled(acc[j0 + m][2], acc[j0 + m][3], sa, sb, inv_down);
      uint16_t conv;
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(conv) : "f"(q_hi), "f"(q_lo));
      b_lo[m] = static_cast<uint32_t>(conv & 0xffu);
      b_hi[m] = static_cast<uint32_t>(conv >> 8);
    }
    auto store_row = [&](int r_off, const uint32_t (&bv)[4]) {
      const uint32_t s1 = __shfl_xor_sync(0xffffffffu, sel4(bv[0], bv[1], bv[2], bv[3], q ^ 1), 1);
      const uint32_t s2 = __shfl_xor_sync(0xffffffffu, sel4(bv[0], bv[1], bv[2], bv[3], q ^ 2), 2);
      const uint32_t s3 = __shfl_xor_sync(0xffffffffu, sel4(bv[0], bv[1], bv[2], bv[3], q ^ 3), 3);
      const uint32_t own = sel4(bv[0], bv[1], bv[2], bv[3], q);
      const uint32_t b0 = sel4(own, s1, s2, s3, 0 ^ q);
      const uint32_t b1 = sel4(own, s1, s2, s3, 1 ^ q);
      const uint32_t b2 = sel4(own, s1, s2, s3, 2 ^ q);
      const uint32_t b3 = sel4(own, s1, s2, s3, 3 ^ q);
      const uint32_t word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
      out32[((size_t)(row0 + rq + r_off) * OUT_COLS + out_col0) / 4 + 4 * half + q] = word;
    };
    store_row(0, b_lo);
    store_row(8, b_hi);
  }
}

template <bool kPdl = false>
__global__ void __launch_bounds__(NT, 1)
swiglu_ws_256_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                     const float* sa_p, const float* sb_p, const float* down_p, uint8_t* out) {
  constexpr int BM_T = 256, ST_T = 2, MW = 4, NB = 2, ASZ = BM_T * KT, STB = ASZ + WSZ;
  constexpr int NTN = N_DIM / BN, NTILES = NB * NTN, NKT = K_DIM / KT;
  static_assert(NTILES == 13 * 48, "13 exact waves on 48 SMs");
  constexpr int total = 13 * NKT;
  constexpr int kPdlTrig = total - 12;
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST_T];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST_T;
  if (tid == 0) {
    for (int s = 0; s < ST_T; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  auto produce = [&](int gl) {
    const int s = gl & 1, n = gl >> 1;
    if (gl >= ST_T) mbar_wait(empty0 + 8 * s, (n - 1) & 1);
    const int ti = gl / NKT, kt = gl - ti * NKT;
    const int t = b + ti * 48, nt = t >> 1, band = t & 1;
    const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
    if constexpr (kPdl) {
      if (gl == 1) {
        mbar_expect_tx(bar, ASZ);
        tma2d_hint(dst, &tmA, kt * KT, band * BM_T, bar, tma_pol_el());
        return;
      }
    }
    mbar_expect_tx(bar, STB);
    tma2d_hint(dst, &tmA, kt * KT, band * BM_T, bar, tma_pol_el());
    tma2d_hint(dst + ASZ, &tmB, kt * KT, nt * BN, bar, tma_pol_ef());
  };
  const float sa = __ldg(sa_p), sb = __ldg(sb_p);
  float inv_down;
  asm("div.full.f32 %0, 0f3F800000, %1;" : "=f"(inv_down) : "f"(__ldg(down_p)));
  if constexpr (kPdl) {
    const int nt0 = b >> 1, band0 = b & 1;
    if (tid == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
      mbar_expect_tx_only(full0, WSZ);
      tma2d_hint(sbase + ASZ, &tmB, 0, nt0 * BN, full0, tma_pol_ef());
      mbar_expect_tx_only(full0 + 8, WSZ);
      tma2d_hint(sbase + STB + ASZ, &tmB, KT, nt0 * BN, full0 + 8, tma_pol_ef());
    }
    pdl_wait();
    if (tid == 0) {
      mbar_expect_tx(full0, ASZ);
      tma2d_hint(sbase, &tmA, 0, band0 * BM_T, full0, tma_pol_el());
    }
  } else if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
    produce(0);
  }
  const int wm = warp % WM, wn = warp / WM;
  float acc[MW][NW][4] = {};
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  int kt = 0, ti = 0;
  for (int g = 0; g < total; ++g) {
    if constexpr (kPdl) {
      if (g == kPdlTrig) pdl_launch_dependents();
    }
    const int s = g & 1, n = g >> 1;
    const int t = b + ti * 48, nt = t >> 1, band = t & 1;
    const int rb0 = band * BM_T + wm * MW * 16;
    if (tid == 0 && g + 1 < total) produce(g + 1);
    mbar_wait(full0 + 8 * s, n & 1);
    const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
#pragma unroll
    for (int ks = 0; ks < KT / 32; ++ks) {
      uint32_t a_reg[MW][4], b_reg[NW][2];
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        const int r = wm * MW * 16 + i * 16 + ar;
        ldsm4(abase + r * KT + swz(r, 2 * ks + ach) * 16, a_reg[i]);
      }
#pragma unroll
      for (int j = 0; j < NW / 2; ++j) {
        const int r = wn * NW * 8 + j * 16 + br;
        uint32_t qd[4];
        ldsm4(wbase + r * KT + swz(r, 2 * ks + bch) * 16, qd);
        b_reg[2 * j][0] = qd[0]; b_reg[2 * j][1] = qd[1];
        b_reg[2 * j + 1][0] = qd[2]; b_reg[2 * j + 1][1] = qd[3];
      }
#pragma unroll
      for (int i = 0; i < MW; ++i)
#pragma unroll
        for (int j = 0; j < NW; ++j) mma_fp8(a_reg[i], b_reg[j][0], b_reg[j][1], acc[i][j]);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * s);
    if (++kt == NKT) {
      kt = 0; ++ti;
      const int out_col0 = (nt * BN + wn * NW * 8) / 2;
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        epilogue_swiglu_fp8(acc[i], out, rb0 + i * 16, out_col0, lane, sa, sb, inv_down);
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
      }
    }
  }
}


// 9-warp WS ST=3 BM=128 (26 exact 48-SM waves, 4-SM lockstep L2 B sharing, consume_stage_pipe, PDL)
constexpr int NT_WS9 = (NWARP + 1) * 32;

__device__ __forceinline__ void consume_stage_pipe_sw(
    uint32_t abase, uint32_t wbase, int wm, int wn, int ar, int ach, int br, int bch, float (&acc)[2][NW][4]) {
  auto load_a_i = [&](int ks, int i, uint32_t (&a)[2][4]) {
    const int r = wm * 32 + i * 16 + ar;
    ldsm4(abase + r * KT + swz(r, 2 * ks + ach) * 16, a[i]);
  };
  auto load_b_j = [&](int ks, int j, uint32_t (&b)[NW][2]) {
    const int r = wn * NW * 8 + j * 16 + br;
    uint32_t qd[4];
    ldsm4(wbase + r * KT + swz(r, 2 * ks + bch) * 16, qd);
    b[2 * j][0] = qd[0];
    b[2 * j][1] = qd[1];
    b[2 * j + 1][0] = qd[2];
    b[2 * j + 1][1] = qd[3];
  };
  uint32_t a[2][2][4], b[2][NW][2];
#pragma unroll
  for (int i = 0; i < 2; ++i) load_a_i(0, i, a[0]);
#pragma unroll
  for (int j = 0; j < NW / 2; ++j) load_b_j(0, j, b[0]);
#pragma unroll
  for (int ks = 0; ks < KT / 32; ++ks) {
    const int cur = ks & 1, nxt = cur ^ 1;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < NW; ++j) {
        mma_fp8(a[cur][i], b[cur][j][0], b[cur][j][1], acc[i][j]);
        if (ks + 1 < KT / 32) {
          if (i == 0 && (j == 0 || j == 2)) {
            load_a_i(ks + 1, j >> 1, a[nxt]);
          } else if (i == 1 && (j & 1) == 0) {
            load_b_j(ks + 1, j >> 1, b[nxt]);
          }
        }
      }
    }
  }
}

template <bool kPdl = false>
__global__ void __launch_bounds__(NT_WS9, 1)
swiglu_ws9_st3_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                      const float* sa_p, const float* sb_p, const float* down_p, uint8_t* out) {
  constexpr int BM_T = 128, ST_T = 3, ASZ = BM_T * KT, STB = ASZ + WSZ;
  constexpr int NTN = N_DIM / BN, NTILES = 4 * NTN, NKT = K_DIM / KT;
  static_assert(NTILES == 26 * 48, "26 exact waves on 48 SMs");
  extern __shared__ uint8_t sm_raw[];
  __shared__ __align__(8) unsigned long long bars[2 * ST_T];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  const uint32_t sbase = ((uint32_t)__cvta_generic_to_shared(sm_raw) + 1023u) & ~1023u;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(&bars[0]), empty0 = full0 + 8 * ST_T;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < ST_T; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NWARP); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();

  if (warp == NWARP) {
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
      int s = 0, par = 0, g = 0;
      const uint64_t pol_el = tma_pol_el(), pol_ef = tma_pol_ef();
      if constexpr (kPdl) {
        const int nt0 = b >> 2;
#pragma unroll
        for (int ps = 0; ps < ST_T; ++ps) {
          const uint32_t bar = full0 + 8 * ps, dst = sbase + ps * STB;
          mbar_expect_tx_only(bar, WSZ);
          tma2d_hint(dst + ASZ, &tmB, ps * KT, nt0 * BN, bar, pol_ef);
        }
        pdl_wait();
      }
      for (int ti = 0; ti < 26; ++ti) {
        const int t = b + ti * 48, nt = t >> 2, band = t & 3;
        const int row0 = band * BM_T, col0 = nt * BN;
        for (int kt = 0; kt < NKT; ++kt) {
          if (g >= ST_T) {
            mbar_wait(empty0 + 8 * s, par ^ 1);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          }
          const uint32_t bar = full0 + 8 * s, dst = sbase + s * STB;
          if constexpr (kPdl) {
            if (g < ST_T) {
              mbar_expect_tx(bar, ASZ);
              tma2d_hint(dst, &tmA, kt * KT, row0, bar, pol_el);
              ++g;
              if (++s == ST_T) { s = 0; par ^= 1; }
              continue;
            }
          }
          mbar_expect_tx(bar, STB);
          tma2d_hint(dst, &tmA, kt * KT, row0, bar, pol_el);
          tma2d_hint(dst + ASZ, &tmB, kt * KT, col0, bar, pol_ef);
          ++g;
          if (++s == ST_T) { s = 0; par ^= 1; }
        }
      }
    }
    return;
  }

  const int wm = warp % WM, wn = warp / WM;
  const float sa = __ldg(sa_p), sb = __ldg(sb_p);
  float inv_down;
  asm("div.full.f32 %0, 0f3F800000, %1;" : "=f"(inv_down) : "f"(__ldg(down_p)));
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ach = lane >> 4;
  const int br = (lane & 7) + (lane >> 4) * 8, bch = (lane >> 3) & 1;
  float acc[2][NW][4] = {};
  int s = 0, par = 0;
  for (int ti = 0; ti < 26; ++ti) {
    const int t = b + ti * 48, nt = t >> 2, band = t & 3;
    for (int kt = 0; kt < NKT; ++kt) {
      if constexpr (kPdl) { if (ti == 25 && kt == NKT - 12) pdl_launch_dependents(); }
      mbar_wait(full0 + 8 * s, par);
      const uint32_t abase = sbase + s * STB, wbase = abase + ASZ;
      consume_stage_pipe_sw(abase, wbase, wm, wn, ar, ach, br, bch, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
      if (++s == ST_T) { s = 0; par ^= 1; }
    }
    const int rb0 = band * BM_T + wm * 32;
    const int out_col0 = (nt * BN + wn * NW * 8) / 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      epilogue_swiglu_fp8(acc[i], out, rb0 + i * 16, out_col0, lane, sa, sb, inv_down);
#pragma unroll
      for (int j = 0; j < NW; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
    }
  }
}

static int g_ws_pdl_override = -1;
static bool ws_pdl_enabled() {
  if (g_ws_pdl_override >= 0) return g_ws_pdl_override != 0;
  static const bool env_on = []() {
    const char* env = std::getenv("MUSE_FP8_SWIGLU_PDL");
    return env && *env && std::strtol(env, nullptr, 10) != 0;
  }();
  return env_on;
}

typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                             const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                             CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn encode = nullptr;

static bool encode_tma2d(CUtensorMap* tm, const void* ptr, int k, int rows, int box_rows) {
  cuuint64_t dims[2] = {(cuuint64_t)k, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)k};
  cuuint32_t box[2] = {(cuuint32_t)KT, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1, 1};
  return encode(tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims, strides, box, es,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
}  // namespace swiglu_ws

#include "fp8_gemm_2cta.cuh"

extern "C" int swiglu_ws_set_pdl(int on) {
  const int old = swiglu_ws::ws_pdl_enabled() ? 1 : 0;
  swiglu_ws::g_ws_pdl_override = (on < 0) ? -1 : (on != 0 ? 1 : 0);
  return old;
}

extern "C" int swiglu_ws_query_occ(int mode) {
  using namespace swiglu_ws;
  int nb = 0;
  switch (mode) {
    case 0: cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, swiglu_ws9_st3_kernel<false>, NT_WS9, SMEM_MAX); break;
    case 2: cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, swiglu_ws8_256x128_st2_kernel<false, false, true>, NT, SMEM_MAX); break;
    case 3: cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, swiglu_ws9_st3_lut_kernel<false, false, true>, NT_WS9, SMEM_MAX); break;
  }
  return nb;
}

extern "C" int swiglu_ws_init(cudaStream_t stream) {
  using namespace swiglu_ws;
  if (!encode) {
    void* fn = nullptr; cudaDriverEntryPointQueryResult q;
    cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
    if (e != cudaSuccess || !fn || q != cudaDriverEntryPointSuccess) return e != cudaSuccess ? int(e) : -2;
    encode = (EncodeFn)fn;
  }
  auto set_smem = [](auto k, int bytes) -> cudaError_t {
    return cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  };
  cudaError_t e = set_smem(swiglu_ws9_st3_kernel<true>, SMEM_MAX);
  if (e == cudaSuccess) e = set_smem(swiglu_ws9_st3_kernel<false>, SMEM_MAX);
  if (e == cudaSuccess) e = set_smem(swiglu_ws8_256x128_st2_kernel<false, false, true>, SMEM_MAX);
  if (e == cudaSuccess) e = set_smem(swiglu_ws8_256x128_st2_kernel<true, false, true>, SMEM_MAX);
  if (e == cudaSuccess) e = set_smem(swiglu_ws8_256x128_st2_kernel<true, true, true>, SMEM_MAX);
  if (e == cudaSuccess) e = set_smem(swiglu_ws9_st3_lut_kernel<false, false, true>, SMEM_MAX);
  if (e == cudaSuccess) e = set_smem(swiglu_ws9_st3_lut_kernel<true, false, true>, SMEM_MAX);
  if (e == cudaSuccess) e = set_smem(swiglu_ws9_st3_lut_kernel<true, true, true>, SMEM_MAX);
  if (e == cudaSuccess) e = set_smem(fp8_6656_ws8_256_st2_st4_kernel<19968, false>, SMEM_MAX);
  if (e == cudaSuccess) e = set_smem(fp8_6656_ws8_256_st2_st4_kernel<19968, true>, SMEM_MAX);
  if (e == cudaSuccess) e = set_smem(fp8_dual_a_ws8_256_kernel<false>, SMEM_MAX);
  if (e == cudaSuccess) e = set_smem(fp8_dual_a_ws8_256_kernel<true>, SMEM_MAX);
  if (e == cudaSuccess) e = set_smem(bf16_dual_ws8_256_kernel<false>, SMEM_MAX);
  if (e == cudaSuccess) e = set_smem(bf16_dual_ws8_256_kernel<true>, SMEM_MAX);
  if (e != cudaSuccess) return int(e);
  int diffs = 0;
  e = cudaMemcpyToSymbolAsync(ws_silu_diffs, &diffs, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  if (e != cudaSuccess) return int(e);
  ws_silu_check<<<256, 256, 0, stream>>>();
  e = cudaMemcpyFromSymbolAsync(&diffs, ws_silu_diffs, sizeof(int), 0, cudaMemcpyDeviceToHost, stream);
  if (e != cudaSuccess) return int(e);
  e = cudaStreamSynchronize(stream);
  if (e != cudaSuccess) return int(e);
  return diffs == 0 ? 0 : -5;
}

extern "C" int swiglu_ws_run_mode(int mode, int pdl, const void* a, const void* b, const float* sa, const float* sb,
                                  const float* down, void* d, cudaStream_t stream) {
  using namespace swiglu_ws;
  if (!encode || (((uintptr_t)a | (uintptr_t)b | (uintptr_t)d) & 15)) return -1;
  uint8_t* out = static_cast<uint8_t*>(d);
  CUtensorMap tmA, tmB;
  switch (mode) {
    case 0:
      if (!encode_tma2d(&tmA, a, K_DIM, M_DIM, 128) || !encode_tma2d(&tmB, b, K_DIM, N_DIM, 128)) return -3;
      if (pdl) return int(launch_pdl_ex(swiglu_ws9_st3_kernel<true>, 48, NT_WS9, SMEM_MAX, stream, tmA, tmB, sa, sb, down, out));
      swiglu_ws9_st3_kernel<false><<<48, NT_WS9, SMEM_MAX, stream>>>(tmA, tmB, sa, sb, down, out);
      return int(cudaGetLastError());
    case 2:
      if (!encode_tma2d(&tmA, a, K_DIM, M_DIM, 256) || !encode_tma2d(&tmB, b, K_DIM, N_DIM, 128)) return -3;
      if (pdl >= 2) return int(launch_pdl_ex(swiglu_ws8_256x128_st2_kernel<true, true, true>, 48, NT, SMEM_MAX, stream, tmA, tmB, sa, sb, down, out));
      if (pdl == 1) return int(launch_pdl_ex(swiglu_ws8_256x128_st2_kernel<true, false, true>, 48, NT, SMEM_MAX, stream, tmA, tmB, sa, sb, down, out));
      swiglu_ws8_256x128_st2_kernel<false, false, true><<<48, NT, SMEM_MAX, stream>>>(tmA, tmB, sa, sb, down, out);
      return int(cudaGetLastError());
    case 3:
      if (!encode_tma2d(&tmA, a, K_DIM, M_DIM, 128) || !encode_tma2d(&tmB, b, K_DIM, N_DIM, 128)) return -3;
      if (pdl >= 2) return int(launch_pdl_ex(swiglu_ws9_st3_lut_kernel<true, true, true>, 48, NT_WS9, SMEM_MAX, stream, tmA, tmB, sa, sb, down, out));
      if (pdl == 1) return int(launch_pdl_ex(swiglu_ws9_st3_lut_kernel<true, false, true>, 48, NT_WS9, SMEM_MAX, stream, tmA, tmB, sa, sb, down, out));
      swiglu_ws9_st3_lut_kernel<false, false, true><<<48, NT_WS9, SMEM_MAX, stream>>>(tmA, tmB, sa, sb, down, out);
      return int(cudaGetLastError());
  }
  return -2;
}

static int g_gemm_pdl_override = -1;
static int gemm_pdl_enabled() {
  if (g_gemm_pdl_override >= 0) return g_gemm_pdl_override;
  static const int env_on = []() {
    const char* env = std::getenv("FP8WS_PDL");
    return (!env || !*env || std::strtol(env, nullptr, 10) != 0) ? 1 : 0;
  }();
  return env_on;
}

extern "C" int muse_fp8_set_gemm_pdl(int on) {
  const int old = gemm_pdl_enabled();
  g_gemm_pdl_override = (on < 0) ? -1 : (on != 0 ? 1 : 0);
  return old;
}

extern "C" int muse_fp8_gemm_6656(int mode, int pdl, const void* a, const void* b, const float* sa, const float* sb,
                                  void* c, int k, cudaStream_t stream) {
  using namespace swiglu_ws;
  if (!encode || k != 19968 || (((uintptr_t)a | (uintptr_t)b | (uintptr_t)c) & 15)) return -1;
  __nv_bfloat16* out = static_cast<__nv_bfloat16*>(c);
  CUtensorMap tmA256, tmA128, tmB128, tmB48, tmB32;
  if (!encode_tma2d(&tmA256, a, k, M_DIM, 256) || !encode_tma2d(&tmA128, a, k, M_DIM, 128)) return -3;
  if (!encode_tma2d(&tmB128, b, k, 6656, 128) || !encode_tma2d(&tmB48, b, k, 6656, 48) ||
      !encode_tma2d(&tmB32, b, k, 6656, 32)) return -4;
  const int use_pdl = (pdl < 0) ? gemm_pdl_enabled() : pdl;
  if (use_pdl) return int(launch_pdl_ex(fp8_6656_ws8_256_st2_st4_kernel<19968, true>, 48, NT, SMEM_MAX, stream, tmA256, tmA128, tmB128, tmB48, tmB32, sa, sb, out));
  fp8_6656_ws8_256_st2_st4_kernel<19968, false><<<48, NT, SMEM_MAX, stream>>>(tmA256, tmA128, tmB128, tmB48, tmB32, sa, sb, out);
  return int(cudaGetLastError());
}

extern "C" int muse_fp8_dual_a(int mode, int pdl, const void* a0, const void* a1, const void* b0, const void* b1,
                               const float* sa0, const float* sa1, const float* sb0, const float* sb1,
                               void* c0, void* c1, cudaStream_t stream) {
  using namespace swiglu_ws;
  if (!encode || (((uintptr_t)a0 | (uintptr_t)a1 | (uintptr_t)b0 | (uintptr_t)b1 | (uintptr_t)c0 | (uintptr_t)c1) & 15)) return -1;
  __nv_bfloat16* o0 = static_cast<__nv_bfloat16*>(c0);
  __nv_bfloat16* o1 = static_cast<__nv_bfloat16*>(c1);
  CUtensorMap tmA0, tmA1, tmB0, tmB1_128, tmB1_112, tmB1_96;
  if (!encode_tma2d(&tmA0, a0, 6656, M_DIM, 256) || !encode_tma2d(&tmA1, a1, 6656, M_DIM, 256)) return -3;
  if (!encode_tma2d(&tmB0, b0, 6656, 4608, 128)) return -4;
  if (!encode_tma2d(&tmB1_128, b1, 6656, 4096, 128) || !encode_tma2d(&tmB1_112, b1, 6656, 4096, 112) ||
      !encode_tma2d(&tmB1_96, b1, 6656, 4096, 96)) return -5;
  const int use_pdl = (pdl < 0) ? gemm_pdl_enabled() : pdl;
  if (use_pdl) return int(launch_pdl_ex(fp8_dual_a_ws8_256_kernel<true>, 48, NT, SMEM_MAX, stream, tmA0, tmA1, tmB0, tmB1_128, tmB1_112, tmB1_96, sa0, sa1, sb0, sb1, o0, o1));
  fp8_dual_a_ws8_256_kernel<false><<<48, NT, SMEM_MAX, stream>>>(tmA0, tmA1, tmB0, tmB1_128, tmB1_112, tmB1_96, sa0, sa1, sb0, sb1, o0, o1);
  return int(cudaGetLastError());
}

extern "C" int muse_bf16_dual_stacked(int mode, int pdl, const void* a, const void* w_cat, void* c0, void* c1, cudaStream_t stream) {
  using namespace swiglu_ws;
  if (!encode || (((uintptr_t)a | (uintptr_t)w_cat | (uintptr_t)c0 | (uintptr_t)c1) & 15)) return -1;
  CUtensorMap tmA256, tmB_128, tmB_112, tmB_96;
  if (!encode_tma2d(&tmA256, a, 2 * 6656, M_DIM, 256)) return -3;
  if (!encode_tma2d(&tmB_128, w_cat, 2 * 6656, 8704, 128) || !encode_tma2d(&tmB_112, w_cat, 2 * 6656, 8704, 112) ||
      !encode_tma2d(&tmB_96, w_cat, 2 * 6656, 8704, 96)) return -4;
  __nv_bfloat16* o0 = static_cast<__nv_bfloat16*>(c0);
  __nv_bfloat16* o1 = static_cast<__nv_bfloat16*>(c1);
  const int use_pdl = (pdl < 0) ? gemm_pdl_enabled() : pdl;
  if (use_pdl) return int(launch_pdl_ex(bf16_dual_ws8_256_kernel<true>, 48, NT, SMEM_MAX, stream, tmA256, tmB_128, tmB_112, tmB_96, o0, o1));
  bf16_dual_ws8_256_kernel<false><<<48, NT, SMEM_MAX, stream>>>(tmA256, tmB_128, tmB_112, tmB_96, o0, o1);
  return int(cudaGetLastError());
}

extern "C" int swiglu_ws_run(const void* a, const void* b, const float* sa, const float* sb, const float* down,
                             void* d, cudaStream_t stream) {
  using namespace swiglu_ws;
  static const int sw_mode = []() {
    const char* env = std::getenv("MUSE_FP8_SWIGLU_MODE");
    return (env && *env) ? int(std::strtol(env, nullptr, 10)) : 3;
  }();
  static const int sw_pdl_def = []() {
    const char* env = std::getenv("MUSE_FP8_SWIGLU_PDL");
    return (!env || !*env) ? 2 : int(std::strtol(env, nullptr, 10));
  }();
  const int pdl = (g_ws_pdl_override >= 0) ? (g_ws_pdl_override != 0 ? 2 : 0) : sw_pdl_def;
  return swiglu_ws_run_mode(sw_mode, pdl, a, b, sa, sb, down, d, stream);
}

