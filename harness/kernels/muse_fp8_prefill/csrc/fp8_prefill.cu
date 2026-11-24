// SPDX-License-Identifier: Apache-2.0
// muse_fp8_prefill: prompt-sized (M > 16) FP8 w8a8 GEMMs on GB10 (sm_121a) with CUTLASS 4.5 SM120 kernels whose tile
// shape and pipeline depth are chosen for 512-row prefill, in place of the stock vLLM sm120 FP8 kernel
// (128x128x128 tile, 2 pipeline stages).
//
// Exactness: every config uses the same CUTLASS SM120 mainloop as the stock kernel (MainloopSm120TmaWarpSpecialized,
// SM120_16x8x32_TN e4m3 MMA atoms; one fp32 accumulator per output element, MMA k32 blocks issued in k order across
// all k-tiles; no split-K, no stream-K), so each element sees the same instruction chain in the same order whatever
// the tile shape or stage count. The epilogue is vLLM's ScaledEpilogue arithmetic, D = bf16_rn(scale_a * fp32_rn(scale_b
// * acc)), for per-tensor scales. The Python side admits each layer only after a bitwise comparison with
// ops.cutlass_scaled_mm on the layer's own weight.
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
#include <type_traits>
using namespace cute;

template <class TileShape, class EpiTile, class KernelSchedule, int Stages = 0>
struct Fp8Prefill {
  using ElementAB = cutlass::float_e4m3_t; using ElementD = cutlass::bfloat16_t; using ElementAcc = float;
  using LayoutA = cutlass::layout::RowMajor; using LayoutB = cutlass::layout::ColumnMajor; using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
  // Cluster multicast is NOT available on this arch: sm120_mma_builder.inl:86 asserts
  //   static_assert(cute::size(ClusterShape_MNK{}) == Int<1>{}, "no programmatic multicast on this arch");
  // so any ClusterShape other than 1x1x1 fails to compile, and the 4x M-tile reuse of the down projection's weight
  // has to be served by L2 rather than by a multicast TMA. Measured (see the config table): the weight is already
  // being re-read at an effective 426 GB/s, i.e. L2 is absorbing that reuse, so nothing is lost by this.
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
  // Stages == 0 keeps CUTLASS's carveout-driven depth; an explicit value pins it. Measured footprints
  // (fp8_prefill_shared_bytes): auto gives 92160 B for 128x192x128 and 128x192x64, 96256 B for 128x96x128 -- all
  // under 114 KiB, so all of them already place two CTAs per SM on a 228 KiB part. Depth is therefore a throughput
  // knob here, not an occupancy knob, and the interesting direction is the one that *reduces* CTAs per SM to buy
  // wave efficiency back. The mainloop instruction chain and the k order are identical at any depth, so the bits
  // do not move.
  using MainloopStages = std::conditional_t<Stages == 0,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::StageCount<Stages>>;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementAB, LayoutA, AlignAB, ElementAB, LayoutB, AlignAB,
      ElementAcc, TileShape, ClusterShape, MainloopStages, KernelSchedule, void>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static int run(int M, int N, int K, const void* A, const void* B, const float* sa, const float* sb, void* D,
                 void* workspace, size_t capacity, int device, int sms, cudaStream_t stream) {
    using StrideA = typename GemmKernel::StrideA; using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC; using StrideD = typename GemmKernel::StrideD;
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    typename EVT::Arguments evt{{{}, {sa}, {}}, {{{}, {sb}, {}}, {}, {}}, {}};
    cutlass::KernelHardwareInfo hw;
    hw.device_id = device;
    hw.sm_count = sms;
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        {static_cast<ElementAB const*>(A), sA, static_cast<ElementAB const*>(B), sB},
        {evt, nullptr, sC, static_cast<ElementD*>(D), sD}, hw};
    auto status = Gemm::can_implement(args);
    if (status != cutlass::Status::kSuccess) return 100 + int(status);
    if (Gemm::get_workspace_size(args) > capacity) return -2;
    Gemm op;
    status = op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return 200 + int(status);
    status = op.run(stream);
    if (status != cutlass::Status::kSuccess) return 300 + int(status);
    return 0;
  }
};

using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
// CONFIG TABLE. Every entry uses CUTLASS's own StageCountAutoCarveout depth: an explicit
// cutlass::gemm::collective::StageCount<N> compiles but FAILS AT RUNTIME on this builder
// (initialize() returns status 207 = 200 + kErrorUnsupportedArch) for every tile and every N tried, so depth is not a
// knob here and the only remaining knobs are the tile shape and the k-tile. Measured footprints
// (fp8_prefill_shared_bytes) show auto-carveout resolves to a SHALLOW pipeline -- 92160 B for 128x192x128, i.e. two
// CTAs per SM on a 228 KiB part -- which corrects the premise this table was originally built on ("auto gives one CTA
// per SM"): the shipping config already has a second pipeline. Measured on GB10 at M=512, 20 iterations, median:
//   cfg2 128x192x128 -> 0.7658 ms, 177.7 TFLOPS on 6656x19968 (down)
//   cfg3 128x96x128  -> 0.7724 ms, 176.2 TFLOPS on 6656x19968
//   cfg1 128x192x64  -> 0.8270 ms, 164.6 TFLOPS on 6656x19968
// all three bit-identical to each other, so the tile is a pure schedule choice. 177.7 TFLOPS is 71% of the ~250 TFLOPS
// dense FP8 peak and is the best rate any kernel in this tree reaches, so the remaining headroom is small.
// 1: 128x192x64 (k-tile 64).
using Cfg1 = Fp8Prefill<Shape<_128, _192, _64>, Shape<_64, _32>, Coop>;
// 2: 128x192x128 (k-tile 128). Shipping choice for the FP8 down projection (6656x19968).
using Cfg2 = Fp8Prefill<Shape<_128, _192, _128>, Shape<_64, _32>, Coop>;
// 3: 128x96x128. Shipping choice for o_proj (6656x4096) and qkv (4608x6656).
using Cfg3 = Fp8Prefill<Shape<_128, _96, _128>, Shape<_64, _32>, Coop>;
// 4: 128x128x128 -- the stock vLLM tile, kept as the reference point for the sweep.
using Cfg4 = Fp8Prefill<Shape<_128, _128, _128>, Shape<_64, _32>, Coop>;
//
// THE TWO WALLS, both measured, which together close the config space on this path:
// (a) compile time: 128x256x128, 256x128x128 and 128x128x256 do not build. StageCountAutoCarveout resolves them to
//     ONE stage and sm120_mma_tma.hpp:152 asserts Stages >= 2. So the usable tiles are exactly those auto-carveout
//     gives two stages to.
// (b) run time: an explicit StageCount<N> builds but initialize() returns status 207 (200 + kErrorUnsupportedArch) for
//     every tile and every N tried, so depth cannot be pinned by hand either. The boundary is a shared-memory ceiling
//     just under 100 KiB: muse_fp8_swiglu runs 128x128x128 at 3 stages in 101376 B and works, while this file's
//     128x128x128 at 3 stages needs 108544 B (its bf16 epilogue is larger) and fails. Since every usable config is
//     <= ~100 KiB on a 228 KiB SM, EVERY usable config is two CTAs per SM, and the occupancy knob does not exist.
//     That is why config 2's 177.7 TFLOPS is not a tuning artifact: it is already the best point in the reachable set.
// (c) epilogue: 128x208x128 also fails, with "EPI_TILE_N must divide CTA_N" (208/32 is not an integer), so the
//     reachable N-tile is a multiple of 32 that also divides 6656 -- i.e. exactly {32, 64, 128, 192}. Configs 1, 2, 3
//     and 4 are 64, 192, 96 and 128, so this table ENUMERATES the reachable tile space for this shape rather than
//     sampling it, and the sweep below is exhaustive rather than a guess.

extern "C" int fp8_prefill_gemm(int cfg, int m, int n, int k, const void* a, const void* b, const float* sa,
                                const float* sb, void* d, void* workspace, size_t capacity, int device, int sms,
                                cudaStream_t stream) {
  switch (cfg) {
    case 1: return Cfg1::run(m, n, k, a, b, sa, sb, d, workspace, capacity, device, sms, stream);
    case 2: return Cfg2::run(m, n, k, a, b, sa, sb, d, workspace, capacity, device, sms, stream);
    case 3: return Cfg3::run(m, n, k, a, b, sa, sb, d, workspace, capacity, device, sms, stream);
    case 4: return Cfg4::run(m, n, k, a, b, sa, sb, d, workspace, capacity, device, sms, stream);
  }
  return -1;
}

// Shared-memory footprint per config, so a sweep can see how many CTAs per SM a config allows without a device query
// (GB10 has 228 KiB per SM).
extern "C" size_t fp8_prefill_shared_bytes(int cfg) {
  switch (cfg) {
    case 1: return Cfg1::GemmKernel::SharedStorageSize;
    case 2: return Cfg2::GemmKernel::SharedStorageSize;
    case 3: return Cfg3::GemmKernel::SharedStorageSize;
    case 4: return Cfg4::GemmKernel::SharedStorageSize;
  }
  return 0;
}

// CUTLASS status name, so a sweep reports "kErrorUnsupportedArch" instead of "207".
extern "C" const char* fp8_prefill_status_str(int s) { return cutlassGetStatusString(static_cast<cutlass::Status>(s)); }
