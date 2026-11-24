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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
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

template <class TileShape, class EpiTile, class KernelSchedule, int Stages>
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
      cutlass::gemm::collective::StageCount<Stages>,
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
// CONFIG TABLE for the 512-row gate/up projection (N = 2 x 19968 interleaved, K = 6656). This kernel is the fastest
// FP8 GEMM in the tree (173 TFLOPS) and is the rate the rest of prefill is measured against, so the sweep is about
// whether it has any headroom left at all rather than about fixing a known problem. Every variant keeps the same
// SM120 mainloop, the same SM120_16x8x32_TN atoms and the same k order, so the bits do not move; only the tile and
// the pipeline depth change. Footprints (shared_bytes_cfg): 0 -> 101376 B (2 CTAs/SM), 1 -> 137728 B (1 CTA/SM),
// 2 -> 92160 B (2 CTAs/SM), 3 -> 133120 B (1 CTA/SM).
// 0: the shipping kernel, 128x128x128 at 3 stages.
using Main = Fp8Gemm<Shape<_128, _128, _128>, Shape<_64, _32>, Schedule, 3>;
static_assert(Main::GemmKernel::SharedStorageSize == 101376);
// 1: 128x128x128 at 4 stages -- one CTA per SM, so the 1248-CTA grid becomes 26 exact waves (1248/48) instead of
//    13 waves at 2 CTAs/SM where the last wave is 93% full. Tests depth against occupancy at the shipping tile.
using Alt1 = Fp8Gemm<Shape<_128, _128, _128>, Shape<_64, _32>, Schedule, 4>;
// 2: 128x192x128 at 2 stages -- 832 CTAs (208 N-tiles), 8.67 waves at 2 CTAs/SM. Fewer, fatter CTAs at the same
//    occupancy: the direction that helped the down projection's wave efficiency, applied here.
using Alt2 = Fp8Gemm<Shape<_128, _192, _128>, Shape<_64, _32>, Schedule, 2>;
// 3: 128x192x128 at 3 stages -- the same tile at one CTA per SM, as the depth control for Alt2.
using Alt3 = Fp8Gemm<Shape<_128, _192, _128>, Shape<_64, _32>, Schedule, 3>;

template <class K>
static inline int run_cfg(int m, int n, int k, const void* a, const void* b, const float* sa, const float* sb,
                          const float* down, void* d, void* workspace, size_t capacity, int device, int sms,
                          cudaStream_t stream) {
  return K::run(m, n, k, a, b, sa, sb, down, d, workspace, capacity, device, sms, stream);
}

extern "C" int prefill(int m, int n, int k, const void* a,const void* b,
  const float* sa,const float* sb,const float* down,void* d,void* workspace,size_t capacity,int device,int sms,cudaStream_t stream){
  return Main::run(m,n,k,a,b,sa,sb,down,d,workspace,capacity,device,sms,stream);
}

// Config-selectable entry point, for sweeps only; the scored path keeps using prefill() above.
extern "C" int prefill_cfg(int cfg, int m, int n, int k, const void* a, const void* b, const float* sa,
                           const float* sb, const float* down, void* d, void* workspace, size_t capacity, int device,
                           int sms, cudaStream_t stream) {
  switch (cfg) {
    case 0: return run_cfg<Main>(m, n, k, a, b, sa, sb, down, d, workspace, capacity, device, sms, stream);
    case 1: return run_cfg<Alt1>(m, n, k, a, b, sa, sb, down, d, workspace, capacity, device, sms, stream);
    case 2: return run_cfg<Alt2>(m, n, k, a, b, sa, sb, down, d, workspace, capacity, device, sms, stream);
    case 3: return run_cfg<Alt3>(m, n, k, a, b, sa, sb, down, d, workspace, capacity, device, sms, stream);
  }
  return -1;
}

extern "C" size_t shared_bytes() { return Main::GemmKernel::SharedStorageSize; }

extern "C" size_t shared_bytes_cfg(int cfg) {
  switch (cfg) {
    case 0: return Main::GemmKernel::SharedStorageSize;
    case 1: return Alt1::GemmKernel::SharedStorageSize;
    case 2: return Alt2::GemmKernel::SharedStorageSize;
    case 3: return Alt3::GemmKernel::SharedStorageSize;
  }
  return 0;
}
