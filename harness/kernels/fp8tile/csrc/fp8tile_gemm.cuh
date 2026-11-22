// SPDX-License-Identifier: Apache-2.0
// fp8tile: vLLM v0.27.1's SM120 FP8 w8a8 CUTLASS GEMM for M <= 16 with other
// output tile widths.
//
// vLLM's cutlass_scaled_mm uses, for M <= 16 on SM120, the CUTLASS 3.x
// collective-builder GEMM with TileShape 16x64x128, KernelTmaWarpSpecializedPingpong,
// cluster 1x1x1, EpilogueTile 16x32, the default persistent tile scheduler and
// the ScaledEpilogue visitor tree out = bf16_rn(scale_a * fp32_rn(scale_b * acc)).
// This file instantiates the same template with the same element types,
// layouts, alignments, schedule and epilogue tree, and changes only the tile:
//
//   * the SM120 pingpong builder uses a 2x2x1 warp layout (no K split) and a
//     permutation tile of min(TileM,128) x min(TileN,32) x 32, so a 16x32 tile
//     has exactly the tiled MMA of the 16x64 tile, and every tile shape runs the
//     same mma.sync m16n8k32 e4m3 instruction;
//   * the mainloop clears the accumulators and, for every k tile in ascending
//     order and every k32 block of it in ascending order, issues one MMA into
//     each output element's fp32 accumulator; the N tile width only decides how
//     many output columns a CTA owns, and a longer K tile (256, 512) is the same
//     ascending sequence of k32 blocks (a layer is admitted only when the tile
//     divides N and K, so no residue path is involved);
//   * no split-K (persistent scheduler), and the epilogue is element-wise.
//
// So every output element is computed by the same instruction sequence in the
// same order as in the stock kernel. The library is still admitted per layer
// only after a bit-for-bit comparison with the stock kernel at load time.

#pragma once

#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace fp8tile {

using namespace cute;

template <class TileShape, class EpilogueTile>
struct Gemm {
  using ElementAB = cutlass::float_e4m3_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementAcc = float;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ClusterShape = Shape<_1, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

  // ScaledEpilogue with per-tensor scales: Compute0 = scale_b * acc (fp32 RN),
  // Compute1 = scale_a * Compute0 -> bf16 (RN).
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using ScaleA = cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>;
  using ScaleB = cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>;
  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute = cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      EpilogueTile, ElementAcc, ElementCompute, void, LayoutC, AlignmentCD, ElementD, LayoutD,
      AlignmentCD, EpilogueSchedule, EVTCompute>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementAB, LayoutA, AlignmentAB,
      ElementAB, LayoutB, AlignmentAB, ElementAcc, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule, void>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                          CollectiveMainloop, CollectiveEpilogue,
                                                          void>;
  using Op = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static typename GemmKernel::Arguments args(const void* a, const void* b, const float* sa,
                                             const float* sb, void* d, int m, int n, int k) {
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;
    StrideA a_stride = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    StrideB b_stride = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    StrideC c_stride = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
    StrideD d_stride = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
    typename EVTCompute0::Arguments evt0{{{}, {sb}, {}}, {}, {}};
    typename EVTCompute::Arguments evt{{{}, {sa}, {}}, evt0, {}};
    cutlass::KernelHardwareInfo hw_info;
    return typename GemmKernel::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {static_cast<const ElementAB*>(a), a_stride, static_cast<const ElementAB*>(b), b_stride},
        {evt, static_cast<const ElementD*>(d), c_stride, static_cast<ElementD*>(d), d_stride},
        hw_info};
  }

  static int workspace(int m, int n, int k, size_t* bytes) {
    auto a = args(nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k);
    *bytes = Op::get_workspace_size(a);
    return 0;
  }

  static int run(const void* a, const void* b, const float* sa, const float* sb, void* d, int m,
                 int n, int k, void* ws, cudaStream_t st) {
    auto ar = args(a, b, sa, sb, d, m, n, k);
    Op op;
    if (op.can_implement(ar) != cutlass::Status::kSuccess) return -20;
    if (op.run(ar, ws, st) != cutlass::Status::kSuccess) return -21;
    return (int)cudaGetLastError();
  }
};

// Configurations: TileShape (M x N x K) and EpilogueTile. (16-wide N tiles do not
// build with this CUTLASS: the epilogue TiledCopy has too few values per thread.)
template <int I> struct Config;
template <> struct Config<0> { using G = Gemm<Shape<_16, _64, _128>, Shape<_16, _32>>; };  // vLLM's M16
template <> struct Config<1> { using G = Gemm<Shape<_16, _32, _128>, Shape<_16, _32>>; };
template <> struct Config<3> { using G = Gemm<Shape<_16, _32, _256>, Shape<_16, _32>>; };
template <> struct Config<4> { using G = Gemm<Shape<_16, _64, _256>, Shape<_16, _32>>; };
template <> struct Config<6> { using G = Gemm<Shape<_16, _32, _512>, Shape<_16, _32>>; };
template <> struct Config<7> { using G = Gemm<Shape<_16, _64, _512>, Shape<_16, _32>>; };

}  // namespace fp8tile
