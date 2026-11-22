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
// Micro-attribution: Interleaving gate and up columns so each register fragment pairs adjacent
// values and packs one E4M3 output byte across two 4-bit transport lanes is inspired by @kaankolcu
// (Yukon submission 1c021643); the dual-TMA fp8ws fallback and SM120 VRAM budget orchestration
// around it are custom-implemented here.
template<class Fragment> struct PairSwiGLU {
    CUTLASS_DEVICE auto operator()(Fragment const& halves, Fragment const& scale) const {
        constexpr int Num=Fragment::kElements;
        static_assert(Num>=2 && Num%2==0, "Each register fragment must contain an even adjacent N pair");
        cutlass::Array<cutlass::float_e2m1_t,Num> out;
        float inverse;
        asm("div.full.f32 %0, %1, %2;" : "=f"(inverse) : "f"(1.0f), "f"(scale[0]));
        #pragma unroll
        for (int i=0;i<Num;i+=2) {
            const float silu=swiglu_silu(halves[i]);
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
