// fp8x_ext.cu (glimmer_fp8_ext): CUTLASS 4.5 sm120 FP8 (e4m3 x e4m3 -> fp32 acc -> bf16) GEMMs with vLLM's ScaledEpilogue arithmetic
// D = bf16(scale_a * fp32(scale_b * acc)) for per-tensor scales, in several tile configs, to find a faster EXACT decode kernel.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
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
using namespace cute;

template <class TileShape, class EpiTile, class KernelSchedule>
struct Fp8Gemm {
  using ElementAB = cutlass::float_e4m3_t; using ElementD = cutlass::bfloat16_t; using ElementAcc = float;
  using LayoutA = cutlass::layout::RowMajor; using LayoutB = cutlass::layout::ColumnMajor; using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ClusterShape = Shape<_1, _1, _1>;
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using ScaleA = cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>;
  using ScaleB = cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>;
  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT = cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVT0>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape, EpiTile,
      ElementAcc, float, void, LayoutD, AlignD, ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto, EVT>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementAB, LayoutA, AlignAB, ElementAB, LayoutB, AlignAB,
      ElementAcc, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule, void>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static torch::Tensor run(torch::Tensor a, torch::Tensor b, torch::Tensor sa, torch::Tensor sb) {
    // a: [M, K] e4m3 row-major; b: [K, N] e4m3 column-major (b.stride(0) == 1); sa, sb: float32 scalars on device
    int M = a.size(0), K = a.size(1), N = b.size(1);
    auto d = torch::empty({M, N}, a.options().dtype(at::kBFloat16));
    using StrideA = typename GemmKernel::StrideA; using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC; using StrideD = typename GemmKernel::StrideD;
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    typename EVT::Arguments evt{
        {{}, {static_cast<float const*>(sa.data_ptr())}, {}},                                   // ScaleA
        {{{}, {static_cast<float const*>(sb.data_ptr())}, {}}, {}, {}},                          // EVT0 = (ScaleB, Accum, Compute0)
        {}};                                                                                     // Compute1
    cutlass::KernelHardwareInfo hw;
    hw.device_id = a.get_device();
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        {static_cast<ElementAB const*>(a.data_ptr()), sA, static_cast<ElementAB const*>(b.data_ptr()), sB},
        {evt, nullptr, sC, static_cast<ElementD*>(d.data_ptr()), sD}, hw};
    Gemm op;
    TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "fp8x: cannot implement");
    size_t ws = Gemm::get_workspace_size(args);
    auto w = torch::empty({(int64_t)std::max<size_t>(ws, 1)}, a.options().dtype(at::kByte));
    auto stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
    TORCH_CHECK(op.initialize(args, w.data_ptr(), stream) == cutlass::Status::kSuccess, "fp8x: init");
    TORCH_CHECK(op.run(stream) == cutlass::Status::kSuccess, "fp8x: run");
    return d;
  }
};
using PP = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using G0 = Fp8Gemm<Shape<_16, _64, _128>, Shape<_16, _32>, PP>;   // == vLLM sm120_fp8_config_M16
using G2 = Fp8Gemm<Shape<_16, _32, _128>, Shape<_16, _32>, PP>;
using G512 = Fp8Gemm<Shape<_16, _32, _512>, Shape<_16, _32>, PP>;
torch::Tensor fp8x_gemm(torch::Tensor a, torch::Tensor b, torch::Tensor sa, torch::Tensor sb, int64_t cfg) {
  const at::cuda::OptionalCUDAGuard guard(a.device());
  switch (cfg) {
    case 0: return G0::run(a, b, sa, sb);
    case 2: return G2::run(a, b, sa, sb);
    case 512: return G512::run(a, b, sa, sb);
  }
  TORCH_CHECK(false, "fp8x: bad cfg");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &fp8x_gemm); }
