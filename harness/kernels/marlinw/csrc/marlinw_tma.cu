// SPDX-License-Identifier: Apache-2.0
// 128-row NVFP4 W4A16 prompt GEMM tiles (M = 512) whose operands are brought into shared memory by TMA.
//
// Derived from e-man07's marlinw_down128 and marlinw_swiglu2 (128-row x 128-column tiles, each weight fragment
// dequantized once per 128 rows). Tiles, operand layouts in registers, per-element arithmetic and both epilogues
// are theirs: fe2m1 code -> bf16, times the bf16 group scale (__hmul2), chain A0 over k16 blocks {4t, 4t+1}
// (warps kw = 0) and A1 over {4t+2, 4t+3} (warps kw = 1) of every 64-deep k-tile in ascending t, each from +0 with
// mma.sync.m16n8k16 (bf16 x bf16 -> fp32), P = A0 + A1, stock K cuts folded through the fp32 scratch (down),
// bf16_rn(P * global_scale), and for gate/up the compiled SwiGLU product.
//
// What differs:
//  - How the operands reach shared memory. In the 128-row kernels every MMA warp issues six cp.async per k-tile
//    with their address arithmetic. Here one thread starts three (down) or five (gate/up) TMA bulk tensor copies
//    per k-tile (activations: a 128 B x rows box with SWIZZLE_128B, which is Marlin's xor layout; codes and scales:
//    plain boxes), stages are handed over with mbarriers (complete_tx), and the MMA warps keep ldmatrix, dequant
//    and HMMA. TMA mechanics follow SP5D's fp8ws.
//  - The fe2m1 -> bf16 placement of each 8 codes is written as mask, multiply, mask: the same four registers bit
//    for bit, three fewer integer instructions.
//  - Down projection, the 16 tiles left over after four full waves (208 tiles on 48 SMs): e-man07 splits them by
//    rows into 48 CTAs of 3/3/2 m-blocks, which run far below the tensor-core rate. Here they are split in K into
//    three full 128x128 pieces per tile (work list: marlinw_down128.tma_work_lists); a later piece continues from
//    the accumulator registers its predecessor stored, so every chain is the whole tile's chain, HMMA for HMMA.
//  - Down projection, K cuts: the hook that saves and clears the accumulators at a stock cut runs between loop
//    segments instead of inside the k-tile loop (where ptxas turned it into ~200 predicated instructions that
//    every k-tile issued).
//  - Gate/up epilogue: SiLU is evaluated with the instruction sequence that fills e-man07's 64K-entry table, on
//    the same BF16-rounded gate, instead of being read from the table; swiglu_prepare compares both for every BF16
//    pattern (FP32 bit for bit) and refuses the library on any difference.
// Measured on GB10 against e-man07's 128-row kernels (bit-exact on real layers): gate/up -2.2% in clock cycles per
// tile, down -4% per call.
//
// MARLINW_TMA_GATEUP selects the fused gate/up + SwiGLU kernel (N = 2 x 19968, K = 6656); default is the down
// projection (N = 6656, K = 19968, work list of marlinw_down128.tma_work_lists).
#ifdef MARLINW_TMA_GATEUP
#define MARLINW_SWIGLU
#ifdef MARLINW_TMA_FP8
#define MARLINW_SWIGLU_FP8
#define MARLIN_NAMESPACE_NAME marlinw_tma_gateup_fp8
#else
#define MARLIN_NAMESPACE_NAME marlinw_tma_gateup
#endif
#else
#define MARLIN_NAMESPACE_NAME marlinw_tma_down
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <type_traits>
#include "marlinw_template.h"


namespace MARLIN_NAMESPACE_NAME {

constexpr int kThreads = 256;
constexpr int kStages = 4;  // power of two
constexpr int kDist = 3;    // k-tiles copied ahead of the one being multiplied (<= kStages - 1)
static_assert((kStages & (kStages - 1)) == 0 && kDist >= 1 && kDist < kStages, "stages");
constexpr int kABytes = 128 * 128;  // 128 rows x 64 k (bf16), 128 B per row, swizzled
constexpr int kBBytes = 4096;       // 4 k16 rows x 128 columns of fe2m1 codes
constexpr int kSBytes = 512;        // 4 k16 rows x 128 fe4m3 scales
constexpr int kStageBytes = 21 * 1024;
// The epilogues stage their reduction in the same window: 64 KiB (down) / 34 KiB (gate/up) from its start.
constexpr int kEpiBytes = 128 * 68 * 4;
template <int kTilesPerCta>
constexpr int kEpiExtraFor() {
  return (kTilesPerCta > 1 && kEpiBytes > kStageBytes) ? (kEpiBytes - kStageBytes) : 0;
}
template <int kTilesPerCta>
constexpr int kSharedFor() {
  constexpr int sb = kStages * kStageBytes + kEpiExtraFor<kTilesPerCta>() + 1024 + 128;
  return sb > 66 * 1024 ? sb : 66 * 1024;
}
constexpr int kShared = kSharedFor<1>();
static_assert(kSharedFor<1>() <= 101376 && kSharedFor<26>() <= 101376, "shared window");
constexpr auto kBF = vllm::kBFloat16.id();

using T = MarlinScalarType<kBF>;
using FragA = T::FragA;
using FragB = T::FragB;

__device__ __forceinline__ void mma16(const FragA& a, const FragB& b, float* c) {
  const uint32_t* ap = reinterpret_cast<const uint32_t*>(&a);
  const uint32_t* bp = reinterpret_cast<const uint32_t*>(&b);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(ap[0]), "r"(ap[1]), "r"(ap[2]), "r"(ap[3]), "r"(bp[0]), "r"(bp[1]));
}
__device__ __forceinline__ void ldsm4(FragA& a, uint32_t smem) {
  uint32_t* r = reinterpret_cast<uint32_t*>(&a);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem));
}
__device__ __forceinline__ void mbar_init(uint32_t a, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx_only(uint32_t a, int bytes) {
  asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;"); }
__device__ __forceinline__ void mbar_wait(uint32_t a, int parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ uint64_t tma_pol_el() {
  uint64_t p;
  asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t tma_pol_ef() {
  uint64_t p;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void tma2d_hint(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar,
                                           uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;"
      ::"r"(dst), "l"(map), "r"(x), "r"(y), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      ::"r"(dst), "l"(map), "r"(x), "r"(y), "r"(bar) : "memory");
}

// The k loop of one CTA: rows [16 MB] x 128 columns x `iters` 64-deep k-tiles, two accumulator sets per element
// (one per kw warp row). produce(g, dst, bar) (thread 0 only) starts the TMA copies of k-tile g into the stage at
// shared address dst, completing on mbarrier bar; on_tile(t) runs before k-tile t. b_rd / s_rd: this thread's
// int4 / int offsets of its codes and scales inside a stage, to which 4 b_row / s_row per k16 row are added.
// A CTA may run several tiles back to back (kTilesPerCta): q0 is the position of this tile's first k-tile in the CTA's
// k-tile stream of length `total`, and the copies keep running across the tile boundary.
template <int MB, bool kPdl = false, bool kPdlProd = false, int kEpiExtraVal = 0, bool kSingleFa = false,
          int kPdlLead = 8, class Produce, class OnTile>
__device__ __forceinline__ void k_loop(
    float (&acc)[MB][2][2][4], const int iters, const int b_rd, const int b_row, const int s_rd, const int s_row,
    Produce&& produce, OnTile&& on_tile, const int q0 = 0, const int total_or_0 = 0,
    const float4* const resume = nullptr, const int t0 = 0) {
  const int total = total_or_0 ? total_or_0 : iters;
  extern __shared__ int4 sh[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int kw = warp >> 2;  // k16 pair of each k-tile: chain A0 / A1
  // Stages start on 1 KiB boundaries of the shared window (TMA swizzles by shared address).
  const uint32_t sbase = (static_cast<uint32_t>(__cvta_generic_to_shared(sh)) + 1023u) & ~1023u;
  const int4* shb = reinterpret_cast<const int4*>(
      reinterpret_cast<const char*>(sh) + (sbase - static_cast<uint32_t>(__cvta_generic_to_shared(sh))));
  // ldmatrix: m-block j of k-half kk sits at 16 (a_rd[kk] + 128 j) (the swizzle term does not depend on j)
  const uint32_t a_rd0 = 16 * (8 * (lane & 15) + (((lane >> 4) + 4 * kw) ^ (lane & 7)));
  const uint32_t a_rd1 = 16 * (8 * (lane & 15) + (((lane >> 4) + 4 * kw + 2) ^ (lane & 7)));
  const uint32_t full_bar = sbase + kStages * kStageBytes + kEpiExtraVal;
  const uint32_t empty_bar = full_bar + 8 * kStages;
  if (q0 == 0) {
    if (tid == 0) {
#pragma unroll
      for (int i = 0; i < kStages; i++) {
        mbar_init(full_bar + 8 * i, 1);
        mbar_init(empty_bar + 8 * i, kThreads / 32);
      }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
  }
  // Thread 0: k-tile g goes to stage g mod kStages once every thread has finished with that stage's previous
  // occupant (k-tile g - kStages).
  auto start_copy = [&](const int g) {
    const int sg = g & (kStages - 1);
    if (g >= t0 + kStages) mbar_wait(empty_bar + 8 * sg, ((g - kStages) / kStages) & 1);
    produce(g, sbase + sg * kStageBytes, full_bar + 8 * sg, 0);
  };
  if constexpr (kPdl) {
    if (q0 == 0) {
      if (tid == 0) {
#pragma unroll
        for (int g = 0; g < kDist; g++) {
          if (t0 + g < total) {
            const int sg = (t0 + g) & (kStages - 1);
            produce(t0 + g, sbase + sg * kStageBytes, full_bar + 8 * sg, 1);
          }
        }
      }
      pdl_wait();
      if (tid == 0) {
#pragma unroll
        for (int g = 0; g < kDist; g++) {
          if (t0 + g < total) {
            const int sg = (t0 + g) & (kStages - 1);
            produce(t0 + g, sbase + sg * kStageBytes, full_bar + 8 * sg, 2);
          }
        }
      }
    }
  } else {
    if (tid == 0 && q0 == 0) {
#pragma unroll
      for (int g = 0; g < kDist; g++)
        if (t0 + g < total) start_copy(t0 + g);
    }
  }

  FragA fa[kSingleFa ? 1 : 2][MB];
  int2 fbq[2];
  int fsq[2];
  FragB bb[2][2][2];  // dequantized B operands [k-half][32-column quarter][n8 block]

  auto load_bs = [&](const int o, const int stage) {
    const int4* sa = shb + (kStageBytes / 16) * stage;
    const int g = 2 * kw + o;
    fbq[o] = *reinterpret_cast<const int2*>(reinterpret_cast<const int*>(sa) + b_rd + b_row * g);
    fsq[o] = reinterpret_cast<const int*>(sa)[s_rd + s_row * g];
  };
  auto dequant_to = [&](const int o) {
    nv_bfloat162 sc[2];
    dequant_fp8_scales<nv_bfloat162, vllm::kFE4M3fn.id()>(fsq[o], sc);
#pragma unroll
    for (int jj = 0; jj < 2; jj++) {
      FragB& b0 = bb[o][jj][0];
      FragB& b1 = bb[o][jj][1];
      const int bq1 = jj == 0 ? fbq[o].x : fbq[o].y;
      // The four registers of dequant<.., kFE2M1f, true>(bq1 << 8) and (bq1), bit for bit: per register the two
      // nibbles are isolated, copied to the sign position and to the exponent/mantissa position by one multiply
      // (the copies cannot overlap) and masked: bit 15 <- sign, bits 8:6 <- e1 e0 m, the same in the upper half.
      {
        const unsigned u = static_cast<unsigned>(bq1);
        reinterpret_cast<unsigned*>(&b1)[1] = (u & 0x80008000u) | ((u >> 6) & 0x01c001c0u);
        reinterpret_cast<unsigned*>(&b1)[0] = (((u & 0x0f000f00u) >> 2) * 65u) & 0x81c081c0u;
        reinterpret_cast<unsigned*>(&b0)[1] = ((u & 0x00f000f0u) * 260u) & 0x81c081c0u;
        reinterpret_cast<unsigned*>(&b0)[0] = ((u & 0x000f000fu) * 4160u) & 0x81c081c0u;
      }
      const nv_bfloat162 s = sc[jj];
      const nv_bfloat162 s0 = __bfloat162bfloat162(reinterpret_cast<const nv_bfloat16*>(&s)[0]);
      const nv_bfloat162 s1 = __bfloat162bfloat162(reinterpret_cast<const nv_bfloat16*>(&s)[1]);
      b0[0] = __hmul2(b0[0], s0);
      b0[1] = __hmul2(b0[1], s0);
      b1[0] = __hmul2(b1[0], s1);
      b1[1] = __hmul2(b1[1], s1);
    }
  };

  if (resume) {  // a later K piece of a tile: both chains continue from the state the previous piece left
#pragma unroll
    for (int e = 0; e < MB * 4; e++) reinterpret_cast<float4*>(acc)[e] = resume[e * kThreads + tid];
  } else {
#pragma unroll
    for (int i = 0; i < MB * 16; i++) reinterpret_cast<float*>(acc)[i] = 0.0f;
  }
  mbar_wait(full_bar + 8 * (q0 & (kStages - 1)), (q0 / kStages) & 1);
#pragma unroll
  for (int j = 0; j < MB; j++) ldsm4(fa[0][j], sbase + kStageBytes * (q0 & (kStages - 1)) + a_rd0 + 2048 * j);
  load_bs(0, q0 & (kStages - 1));
  dequant_to(0);

  // on_tile(t) returns the k-tile before which it wants to run next: the loop over the k-tiles in between carries
  // none of its code (compiled into the loop it becomes predicated instructions that every k-tile pays for).
#pragma unroll 1
  for (int t = t0; t < iters;) {
  const int stop = on_tile(t);
#pragma unroll 1
  for (; t < stop; t++) {
    if constexpr (kPdlProd) {
      if (t == ((iters - kPdlLead > t0) ? (iters - kPdlLead) : t0) && q0 + iters == total) pdl_launch_dependents();
    }
    const int q = q0 + t;
    const int s = q & (kStages - 1);
    const int ns = (q + 1) & (kStages - 1);
    if (tid == 0 && q + kDist < total) start_copy(q + kDist);
    // k-half 0 of k-tile t (operands already in registers); meanwhile k-half 1 of the same stage is loaded.
    load_bs(1, s);
    dequant_to(1);
    if constexpr (kSingleFa) {
      const uint32_t sa = sbase + kStageBytes * s + a_rd1;
#pragma unroll
      for (int i = 0; i < MB; i++) {
        mma16(fa[0][i], bb[0][0][0], acc[i][0][0]);
        mma16(fa[0][i], bb[0][0][1], acc[i][0][1]);
        mma16(fa[0][i], bb[0][1][0], acc[i][1][0]);
        mma16(fa[0][i], bb[0][1][1], acc[i][1][1]);
        ldsm4(fa[0][i], sa + 2048 * i);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_bar + 8 * s);
      const bool has_next = (t + 1 < iters);
      if (has_next) {
        mbar_wait(full_bar + 8 * ns, ((q + 1) / kStages) & 1);
        load_bs(0, ns);
        dequant_to(0);
      }
      {
        const uint32_t sa0 = sbase + kStageBytes * ns + a_rd0;
#pragma unroll
        for (int i = 0; i < MB; i++) {
          mma16(fa[0][i], bb[1][0][0], acc[i][0][0]);
          mma16(fa[0][i], bb[1][0][1], acc[i][0][1]);
          mma16(fa[0][i], bb[1][1][0], acc[i][1][0]);
          mma16(fa[0][i], bb[1][1][1], acc[i][1][1]);
          if (has_next) ldsm4(fa[0][i], sa0 + 2048 * i);
        }
      }
    } else {
      {
        const uint32_t sa = sbase + kStageBytes * s + a_rd1;
#pragma unroll
        for (int i = 0; i < MB; i++) {
          mma16(fa[0][i], bb[0][0][0], acc[i][0][0]);
          mma16(fa[0][i], bb[0][0][1], acc[i][0][1]);
          ldsm4(fa[1][i], sa + 2048 * i);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_bar + 8 * s);  // warp-aggregated empty signal
#pragma unroll
      for (int i = 0; i < MB; i++) {
        mma16(fa[0][i], bb[0][1][0], acc[i][1][0]);
        mma16(fa[0][i], bb[0][1][1], acc[i][1][1]);
      }
      // k-half 1 of k-tile t; meanwhile k-half 0 of k-tile t + 1 is loaded.
      if (t + 1 < iters) mbar_wait(full_bar + 8 * ns, ((q + 1) / kStages) & 1);
      load_bs(0, ns);
      dequant_to(0);
      {
        const uint32_t sa = sbase + kStageBytes * ns + a_rd0;
#pragma unroll
        for (int i = 0; i < MB; i++) {
          mma16(fa[1][i], bb[1][0][0], acc[i][0][0]);
          mma16(fa[1][i], bb[1][0][1], acc[i][0][1]);
          ldsm4(fa[0][i], sa + 2048 * i);
        }
      }
#pragma unroll
      for (int i = 0; i < MB; i++) {
        mma16(fa[1][i], bb[1][1][0], acc[i][1][0]);
        mma16(fa[1][i], bb[1][1][1], acc[i][1][1]);
      }
    }
  }
  }
}

__device__ __forceinline__ uint32_t selp_u32(uint32_t a, uint32_t b, bool c) {
  uint32_t r;
  asm("{\n .reg .pred p;\n setp.ne.b32 p, %3, 0;\n selp.b32 %0, %1, %2, p;\n}"
      : "=r"(r) : "r"(a), "r"(b), "r"((uint32_t)c));
  return r;
}
__device__ __forceinline__ uint32_t sel4_u32(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, int i) {
  const uint32_t lo = selp_u32(v1, v0, i & 1), hi = selp_u32(v3, v2, i & 1);
  return selp_u32(hi, lo, i & 2);
}

#ifndef MARLINW_TMA_GATEUP
// ---------------------------------------------------------------------------------------------------------------
// Down projection. One CTA = one work item of marlinw_down128: w[2b] = {row0, col128, mb, slot},
// w[2b+1] = {cut_a, cut_b, k_tiles, 0}; cut_a / cut_b are the k-tile at which the stock partition restarts the
// chains of the 64-row chunk containing row0 / the next chunk (-1: none).
constexpr int kColumns = 6656;
constexpr int kInner = 19968;

template <int V>
struct IntTag {
  static constexpr int value = V;
};

template <int MB, bool kLater, int kCols, bool kShflEpi = true, int kPdlMode = 0, int kPdlLead = 8>  // kPdlMode: 0=off, 1=prod-only, 2=dual
__device__ __forceinline__ void down_tile(
    const CUtensorMap* tmA, const CUtensorMap* tmB, const CUtensorMap* tmS, int4* __restrict__ C0,
    float4* __restrict__ scratch, const float gs, const int row0, const int col, const int slot, const int cut_a,
    const int cut_b, const int k_tiles
    , const int piece, volatile int* const flags, const int cut_a2 = -1, const int cut_b2 = -1
    ) {
  constexpr bool kTwoCuts = (kCols == 4096);
  extern __shared__ int4 sh[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int kw = warp >> 2;
  const int nb = (warp >> 1) & 1;  // 64-column half of the tile
  const int sh2 = warp & 1;        // 32-column half of that

  float acc[MB][2][2][4];
  const int n_first = 4 - ((row0 >> 4) & 3);  // m-blocks in the chunk containing row0
  float4* my_scratch = scratch + ((size_t)slot * 256 + tid) * 32;
  float4* my_scratch2 = scratch + ((size_t)(slot + 1) * 256 + tid) * 32;
  // K pieces. piece = k0 | role << 10 | index << 12: this CTA runs k-tiles [k0, k0 + k_tiles) of its tile (k0 a
  // multiple of 8). role 0: the whole tile. 1: first piece (chains from +0, state stored at the end). 2: middle piece
  // (state loaded, state stored). 3: last piece (state loaded, epilogue). The state is every thread's accumulator
  // registers (scratch slot `index`), so the chains of a split tile are the chains of the whole tile, HMMA for HMMA.
  // A later piece waits until its predecessor has published the state: flags[index] goes 0 -> 1 (first piece
  // done) -> 2 (middle piece done) -> 0 (taken by the last piece).
  const int k0 = (kCols == 6656 && kLater) ? (piece & 1023) : 0;
  const int k1 = k0 + k_tiles;
  if constexpr (kCols == 6656 && kLater) {
    if (tid == 0 && (piece & 2048)) {
      volatile int* const flag = flags + (piece >> 12);
      const int before = ((piece >> 10) & 3) - 1;
      while (*flag != before) {
      }
      __threadfence();  // acquire: the predecessor's state (stored before its release fence + flag) is visible
      if (before == 2) *flag = 0;
    }
  }
  // Runs before k-tile t; returns the k-tile before which it has to run again (the next cut, or the end).
  auto cuts = [&](const int t) {
    if (t == cut_a || t == cut_b || (kTwoCuts && (t == cut_a2 || t == cut_b2))) {
      __syncthreads();
#pragma unroll
      for (int i = 0; i < MB; i++) {
        const int ci = i < n_first ? cut_a : cut_b;
        if (ci == t) {
#pragma unroll
          for (int e = 0; e < 4; e++) my_scratch[i * 4 + e] = reinterpret_cast<float4*>(acc[i])[e];
#pragma unroll
          for (int e = 0; e < 16; e++) reinterpret_cast<float*>(acc[i])[e] = 0.0f;
        } else if constexpr (kTwoCuts) {
          const int ci2 = i < n_first ? cut_a2 : cut_b2;
          if (ci2 == t) {
#pragma unroll
            for (int e = 0; e < 4; e++) my_scratch2[i * 4 + e] = reinterpret_cast<float4*>(acc[i])[e];
#pragma unroll
            for (int e = 0; e < 16; e++) reinterpret_cast<float*>(acc[i])[e] = 0.0f;
          }
        }
      }
      __syncthreads();
    }
    int stop = k1;
    if (cut_a > t && cut_a < stop) stop = cut_a;
    if (cut_b > t && cut_b < stop) stop = cut_b;
    if constexpr (kTwoCuts) {
      if (cut_a2 > t && cut_a2 < stop) stop = cut_a2;
      if (cut_b2 > t && cut_b2 < stop) stop = cut_b2;
    }
    return stop;
  };
  auto produce = [&](const int g, const uint32_t dst, const uint32_t bar, const int mode) {
    if (mode == 0) {
      mbar_expect_tx(bar, MB * 16 * 128 + kBBytes + kSBytes);
      tma2d_hint(dst, tmA, 128 * g, row0, bar, tma_pol_el());                         // bytes, rows
      if constexpr (kCols == 6656) {
        tma2d_hint(dst + kABytes, tmB, 256 * col, 4 * g, bar, tma_pol_ef());          // uint32, k16 rows
        tma2d_hint(dst + kABytes + kBBytes, tmS, 128 * col, 4 * g, bar, tma_pol_ef());// bytes, k16 rows
      } else {
        tma2d(dst + kABytes, tmB, 256 * col, 4 * g, bar);
        tma2d(dst + kABytes + kBBytes, tmS, 128 * col, 4 * g, bar);
      }
    } else if (mode == 1) {
      mbar_expect_tx_only(bar, kBBytes + kSBytes);
      if constexpr (kCols == 6656) {
        tma2d_hint(dst + kABytes, tmB, 256 * col, 4 * g, bar, tma_pol_ef());
        tma2d_hint(dst + kABytes + kBBytes, tmS, 128 * col, 4 * g, bar, tma_pol_ef());
      } else {
        tma2d(dst + kABytes, tmB, 256 * col, 4 * g, bar);
        tma2d(dst + kABytes + kBBytes, tmS, 128 * col, 4 * g, bar);
      }
    } else {
      mbar_expect_tx(bar, MB * 16 * 128);
      tma2d_hint(dst, tmA, 128 * g, row0, bar, tma_pol_el());
    }
  };
  // codes: int4 (64 g + 32 nb + lane), int2 sh2 of it; scales: int 32 g + 2 (8 nb + lane / 4) + sh2
  k_loop<MB, (kPdlMode == 2), (kPdlMode != 0), 0, false, kPdlLead>(
      acc, k1, kABytes / 4 + 4 * (32 * nb + lane) + 2 * sh2, 256,
      (kABytes + kBBytes) / 4 + 2 * (8 * nb + (lane >> 2)) + sh2, 32, produce, cuts, 0, 0,
      kLater ? scratch + (size_t)(piece >> 12) * (kThreads * 32) : nullptr, k0);
  __syncthreads();
  if constexpr (kCols == 6656) {
    const int role = (piece >> 10) & 3;
    if (kLater ? (role == 2) : (role == 1)) {
      float4* const hand = scratch + (size_t)(piece >> 12) * (kThreads * 32);
#pragma unroll
      for (int e = 0; e < MB * 4; e++) hand[e * kThreads + tid] = reinterpret_cast<float4*>(acc)[e];
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        flags[piece >> 12] = role;
        __threadfence();
      }
      return;
    }
  }

  // P = A0 + A1, fold of the K cuts, bf16_rn(P * global_scale): as marlinw_down128.
  // For MB == 8 (all TMA down shapes), both k-warp halves (all 256 threads) execute the reduction,
  // cut fold, and BF16 global store in parallel (kw=0 owns i=0..3, kw=1 owns i=4..7) with strictly
  // compile-time acc[] indices (0 register spills) and exact A0 + A1 accumulation order (0-ULP).
  float4* red = reinterpret_cast<float4*>(sh);
  if constexpr (MB == 8) {
    const int t128 = tid & 127;
    if (kw == 0) {
#pragma unroll
      for (int k = 0; k < 16; k++) red[k * 128 + t128] = reinterpret_cast<float4*>(&acc[4])[k];
    } else {
#pragma unroll
      for (int k = 0; k < 16; k++) red[(16 + k) * 128 + t128] = reinterpret_cast<float4*>(&acc[0])[k];
    }
    __syncthreads();
    auto finish_half = [&](auto base_tag) {
      constexpr int kBase = decltype(base_tag)::value;
      if constexpr (kBase == 0) {
#pragma unroll
        for (int k = 0; k < 16; k++) {
          const float4 r = red[(16 + k) * 128 + t128];
          float* c = reinterpret_cast<float*>(&acc[0]) + 4 * k;
          c[0] += r.x; c[1] += r.y; c[2] += r.z; c[3] += r.w;
        }
      } else {
#pragma unroll
        for (int k = 0; k < 16; k++) {
          const float4 r = red[k * 128 + t128];
          float* c = reinterpret_cast<float*>(&acc[4]) + 4 * k;
          c[0] = r.x + c[0]; c[1] = r.y + c[1]; c[2] = r.z + c[2]; c[3] = r.w + c[3];
        }
      }
      if (cut_a >= 0 || cut_b >= 0) {
        const float4* s0 = scratch + ((size_t)slot * 256 + t128) * 32;
        const float4* s1 = scratch + ((size_t)slot * 256 + t128 + 128) * 32;
        const float4* s0_2 = scratch + ((size_t)(slot + 1) * 256 + t128) * 32;
        const float4* s1_2 = scratch + ((size_t)(slot + 1) * 256 + t128 + 128) * 32;
#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
          const int i = kBase + ii;
          const int ci = i < n_first ? cut_a : cut_b;
          if (ci >= 0) {
            if constexpr (kTwoCuts) {
              const int ci2 = i < n_first ? cut_a2 : cut_b2;
              if (ci2 >= 0) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                  const float4 x = s0[i * 4 + e], y = s1[i * 4 + e];
                  const float4 x2 = s0_2[i * 4 + e], y2 = s1_2[i * 4 + e];
                  float* c = reinterpret_cast<float*>(acc[kBase + ii]) + 4 * e;
                  c[0] = (x.x + y.x) + ((x2.x + y2.x) + c[0]);
                  c[1] = (x.y + y.y) + ((x2.y + y2.y) + c[1]);
                  c[2] = (x.z + y.z) + ((x2.z + y2.z) + c[2]);
                  c[3] = (x.w + y.w) + ((x2.w + y2.w) + c[3]);
                }
              } else {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                  const float4 x = s0[i * 4 + e], y = s1[i * 4 + e];
                  float* c = reinterpret_cast<float*>(acc[kBase + ii]) + 4 * e;
                  c[0] = (x.x + y.x) + c[0];
                  c[1] = (x.y + y.y) + c[1];
                  c[2] = (x.z + y.z) + c[2];
                  c[3] = (x.w + y.w) + c[3];
                }
              }
            } else {
#pragma unroll
              for (int e = 0; e < 4; e++) {
                const float4 x = s0[i * 4 + e], y = s1[i * 4 + e];
                float* c = reinterpret_cast<float*>(acc[kBase + ii]) + 4 * e;
                c[0] = (x.x + y.x) + c[0];
                c[1] = (x.y + y.y) + c[1];
                c[2] = (x.z + y.z) + c[2];
                c[3] = (x.w + y.w) + c[3];
              }
            }
          }
        }
      }
      __nv_bfloat16* C = reinterpret_cast<__nv_bfloat16*>(C0);
      if constexpr (kShflEpi) {
        const int q = lane & 3;
        const int r0 = row0 + (lane >> 2);
        const int base_col0 = 128 * col + 64 * nb + 32 * sh2;
#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
          const int i = kBase + ii;
#pragma unroll
          for (int jj = 0; jj < 2; jj++) {
            const float* c = &acc[kBase + ii][jj][0][0];
            const nv_bfloat162 lo0 = __halves2bfloat162(__float2bfloat16(c[0] * gs), __float2bfloat16(c[1] * gs));
            const nv_bfloat162 hi0 = __halves2bfloat162(__float2bfloat16(c[2] * gs), __float2bfloat16(c[3] * gs));
            const nv_bfloat162 lo1 = __halves2bfloat162(__float2bfloat16(c[4] * gs), __float2bfloat16(c[5] * gs));
            const nv_bfloat162 hi1 = __halves2bfloat162(__float2bfloat16(c[6] * gs), __float2bfloat16(c[7] * gs));
            const uint32_t v0 = *reinterpret_cast<const uint32_t*>(&lo0);
            const uint32_t v1 = *reinterpret_cast<const uint32_t*>(&hi0);
            const uint32_t v2 = *reinterpret_cast<const uint32_t*>(&lo1);
            const uint32_t v3 = *reinterpret_cast<const uint32_t*>(&hi1);
            const uint32_t r1 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 1), 1);
            const uint32_t r2 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 2), 2);
            const uint32_t r3 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 3), 3);
            const uint32_t own = sel4_u32(v0, v1, v2, v3, q);
            uint4 w;
            w.x = sel4_u32(own, r1, r2, r3, 0 ^ q);
            w.y = sel4_u32(own, r1, r2, r3, 1 ^ q);
            w.z = sel4_u32(own, r1, r2, r3, 2 ^ q);
            w.w = sel4_u32(own, r1, r2, r3, 3 ^ q);
            const int r_out = r0 + 16 * i + (q & 1) * 8;
            const int c_out = base_col0 + 16 * jj + (q >> 1) * 8;
            *reinterpret_cast<uint4*>(&C[(size_t)r_out * kCols + c_out]) = w;
          }
        }
      } else {
        const int r0 = row0 + (lane >> 2);
        const int col0 = 128 * col + 64 * nb + 32 * sh2 + 2 * (lane & 3);
#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
          const int i = kBase + ii;
#pragma unroll
          for (int jj = 0; jj < 2; jj++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
              const float* c = acc[kBase + ii][jj][h];
              const nv_bfloat162 lo = __halves2bfloat162(__float2bfloat16(c[0] * gs), __float2bfloat16(c[1] * gs));
              const nv_bfloat162 hi = __halves2bfloat162(__float2bfloat16(c[2] * gs), __float2bfloat16(c[3] * gs));
              const int cc = col0 + 16 * jj + 8 * h;
              *reinterpret_cast<nv_bfloat162*>(&C[(size_t)(r0 + 16 * i) * kCols + cc]) = lo;
              *reinterpret_cast<nv_bfloat162*>(&C[(size_t)(r0 + 16 * i + 8) * kCols + cc]) = hi;
            }
        }
      }
    };
    if (kw == 0) finish_half(IntTag<0>{});
    else finish_half(IntTag<4>{});
  } else {
    if (kw == 1) {
#pragma unroll
      for (int k = 0; k < MB * 4; k++) red[k * 128 + (tid - 128)] = reinterpret_cast<float4*>(acc)[k];
    }
    __syncthreads();
    if (kw == 0) {
#pragma unroll
      for (int k = 0; k < MB * 4; k++) {
        const float4 r = red[k * 128 + tid];
        float* c = reinterpret_cast<float*>(acc) + 4 * k;
        c[0] += r.x; c[1] += r.y; c[2] += r.z; c[3] += r.w;
      }
      if (cut_a >= 0 || cut_b >= 0) {
        const float4* s0 = scratch + ((size_t)slot * 256 + tid) * 32;
        const float4* s1 = scratch + ((size_t)slot * 256 + tid + 128) * 32;
        const float4* s0_2 = scratch + ((size_t)(slot + 1) * 256 + tid) * 32;
        const float4* s1_2 = scratch + ((size_t)(slot + 1) * 256 + tid + 128) * 32;
#pragma unroll
        for (int i = 0; i < MB; i++) {
          const int ci = i < n_first ? cut_a : cut_b;
          if (ci >= 0) {
            if constexpr (kTwoCuts) {
              const int ci2 = i < n_first ? cut_a2 : cut_b2;
              if (ci2 >= 0) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                  const float4 x = s0[i * 4 + e], y = s1[i * 4 + e];
                  const float4 x2 = s0_2[i * 4 + e], y2 = s1_2[i * 4 + e];
                  float* c = reinterpret_cast<float*>(acc[i]) + 4 * e;
                  c[0] = (x.x + y.x) + ((x2.x + y2.x) + c[0]);
                  c[1] = (x.y + y.y) + ((x2.y + y2.y) + c[1]);
                  c[2] = (x.z + y.z) + ((x2.z + y2.z) + c[2]);
                  c[3] = (x.w + y.w) + ((x2.w + y2.w) + c[3]);
                }
              } else {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                  const float4 x = s0[i * 4 + e], y = s1[i * 4 + e];
                  float* c = reinterpret_cast<float*>(acc[i]) + 4 * e;
                  c[0] = (x.x + y.x) + c[0];
                  c[1] = (x.y + y.y) + c[1];
                  c[2] = (x.z + y.z) + c[2];
                  c[3] = (x.w + y.w) + c[3];
                }
              }
            } else {
#pragma unroll
              for (int e = 0; e < 4; e++) {
                const float4 x = s0[i * 4 + e], y = s1[i * 4 + e];
                float* c = reinterpret_cast<float*>(acc[i]) + 4 * e;
                c[0] = (x.x + y.x) + c[0];
                c[1] = (x.y + y.y) + c[1];
                c[2] = (x.z + y.z) + c[2];
                c[3] = (x.w + y.w) + c[3];
              }
            }
          }
        }
      }
      __nv_bfloat16* C = reinterpret_cast<__nv_bfloat16*>(C0);
      if constexpr (kShflEpi) {
        const int q = lane & 3;
        const int r0 = row0 + (lane >> 2);
        const int base_col0 = 128 * col + 64 * nb + 32 * sh2;
#pragma unroll
        for (int i = 0; i < MB; i++) {
#pragma unroll
          for (int jj = 0; jj < 2; jj++) {
            const float* c = &acc[i][jj][0][0];
            const nv_bfloat162 lo0 = __halves2bfloat162(__float2bfloat16(c[0] * gs), __float2bfloat16(c[1] * gs));
            const nv_bfloat162 hi0 = __halves2bfloat162(__float2bfloat16(c[2] * gs), __float2bfloat16(c[3] * gs));
            const nv_bfloat162 lo1 = __halves2bfloat162(__float2bfloat16(c[4] * gs), __float2bfloat16(c[5] * gs));
            const nv_bfloat162 hi1 = __halves2bfloat162(__float2bfloat16(c[6] * gs), __float2bfloat16(c[7] * gs));
            const uint32_t v0 = *reinterpret_cast<const uint32_t*>(&lo0);
            const uint32_t v1 = *reinterpret_cast<const uint32_t*>(&hi0);
            const uint32_t v2 = *reinterpret_cast<const uint32_t*>(&lo1);
            const uint32_t v3 = *reinterpret_cast<const uint32_t*>(&hi1);
            const uint32_t r1 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 1), 1);
            const uint32_t r2 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 2), 2);
            const uint32_t r3 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 3), 3);
            const uint32_t own = sel4_u32(v0, v1, v2, v3, q);
            uint4 w;
            w.x = sel4_u32(own, r1, r2, r3, 0 ^ q);
            w.y = sel4_u32(own, r1, r2, r3, 1 ^ q);
            w.z = sel4_u32(own, r1, r2, r3, 2 ^ q);
            w.w = sel4_u32(own, r1, r2, r3, 3 ^ q);
            const int r_out = r0 + 16 * i + (q & 1) * 8;
            const int c_out = base_col0 + 16 * jj + (q >> 1) * 8;
            *reinterpret_cast<uint4*>(&C[(size_t)r_out * kCols + c_out]) = w;
          }
        }
      } else {
        const int r0 = row0 + (lane >> 2);
        const int col0 = 128 * col + 64 * nb + 32 * sh2 + 2 * (lane & 3);
#pragma unroll
        for (int i = 0; i < MB; i++)
#pragma unroll
          for (int jj = 0; jj < 2; jj++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
              const float* c = acc[i][jj][h];
              const nv_bfloat162 lo = __halves2bfloat162(__float2bfloat16(c[0] * gs), __float2bfloat16(c[1] * gs));
              const nv_bfloat162 hi = __halves2bfloat162(__float2bfloat16(c[2] * gs), __float2bfloat16(c[3] * gs));
              const int cc = col0 + 16 * jj + 8 * h;
              *reinterpret_cast<nv_bfloat162*>(&C[(size_t)(r0 + 16 * i) * kCols + cc]) = lo;
              *reinterpret_cast<nv_bfloat162*>(&C[(size_t)(r0 + 16 * i + 8) * kCols + cc]) = hi;
            }
      }
    }
  }
}


template <int kCols, bool kShflEpi = true, int kPdlMode = 0>
__global__ void __launch_bounds__(kThreads, 1) down(
    const __grid_constant__ CUtensorMap tmA8, const __grid_constant__ CUtensorMap tmB,
    const __grid_constant__ CUtensorMap tmS, int4* __restrict__ C0, float4* __restrict__ scratch,
    const float* __restrict__ gs_ptr, int4* __restrict__ work,
    const uint8_t* __restrict__ w_raw, size_t w_bytes
    ) {
  if (threadIdx.x == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA8) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmS) : "memory");
  }
  if (blockIdx.x == 0 && w_raw != nullptr) {
    constexpr size_t kTpf = 2048u << 10;
    for (size_t off = static_cast<size_t>(threadIdx.x) * kTpf; off < w_bytes;
         off += static_cast<size_t>(kThreads) * kTpf) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(w_raw + off));
    }
  }
  const int4 w0 = work[2 * blockIdx.x];
  const int4 w1 = work[2 * blockIdx.x + 1];
  const float gs = gs_ptr[0];
  const int cut_a2 = (kCols == 4096) ? static_cast<int>(static_cast<int16_t>(w0.z & 0xFFFF)) : -1;
  const int cut_b2 = (kCols == 4096) ? static_cast<int>(static_cast<int16_t>((w0.z >> 16) & 0xFFFF)) : -1;
  if constexpr (kCols == 6656) {
    if (w1.w & 2048)
      down_tile<8, true, kCols, kShflEpi, kPdlMode>(&tmA8, &tmB, &tmS, C0, scratch, gs, w0.x, w0.y, w0.w, w1.x, w1.y, w1.z, w1.w,
                         reinterpret_cast<volatile int*>(work + 2 * gridDim.x), cut_a2, cut_b2);
    else
      down_tile<8, false, kCols, kShflEpi, kPdlMode>(&tmA8, &tmB, &tmS, C0, scratch, gs, w0.x, w0.y, w0.w, w1.x, w1.y, w1.z, w1.w,
                          reinterpret_cast<volatile int*>(work + 2 * gridDim.x), cut_a2, cut_b2);
  } else {
    down_tile<8, false, kCols, kShflEpi, kPdlMode>(&tmA8, &tmB, &tmS, C0, scratch, gs, w0.x, w0.y, w0.w, w1.x, w1.y, w1.z, w1.w,
                        reinterpret_cast<volatile int*>(work + 2 * gridDim.x), cut_a2, cut_b2);
  }
}

// Fused 4-stage TMA kernel for paired NVFP4 attention projections: gate (4096x6656, 2-cut in pair=3) +
// qkv (4608x6656, 0 K-cuts). One 512x6656 BF16 activation matrix tmA8 is shared across all 272 column-major CTAs
// (128 gate first, 144 qkv last -> 4x weight L2 reuse, 0 handover scratch, 16 free SMs on Wave 6 for Dual PDL
// qk_norm_rope overlap with kPdlLead=24, and L2-hot qkv output for qk_norm_rope; 0-ULP bit-exact with stock Marlin).
// Clean 2-branch dispatch (bit 30 of w1.w selects gate 4096 vs qkv 4608; no split-K / kLater codegen).
template <bool kShflEpi = true, int kPdlMode = 0>
__global__ void __launch_bounds__(kThreads, 1) down_dual(
    const __grid_constant__ CUtensorMap tmA8,
    const __grid_constant__ CUtensorMap tmB_q, const __grid_constant__ CUtensorMap tmS_q,
    const __grid_constant__ CUtensorMap tmB_g, const __grid_constant__ CUtensorMap tmS_g,
    int4* __restrict__ Cq, int4* __restrict__ Cg, float4* __restrict__ scratch,
    const float* __restrict__ gs_q_ptr, const float* __restrict__ gs_g_ptr,
    int4* __restrict__ work,
    const uint8_t* __restrict__ wq_raw, size_t wq_bytes,
    const uint8_t* __restrict__ wg_raw, size_t wg_bytes) {
  if (blockIdx.x == 0) {
    constexpr size_t kTpf = 2048u << 10;
    if (wq_raw != nullptr) {
      for (size_t off = static_cast<size_t>(threadIdx.x) * kTpf; off < wq_bytes;
           off += static_cast<size_t>(kThreads) * kTpf) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(wq_raw + off));
      }
    }
    if (wg_raw != nullptr) {
      for (size_t off = static_cast<size_t>(threadIdx.x) * kTpf; off < wg_bytes;
           off += static_cast<size_t>(kThreads) * kTpf) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(wg_raw + off));
      }
    }
  }
  const int4 w0 = work[2 * blockIdx.x];
  const int4 w1 = work[2 * blockIdx.x + 1];
  if (w1.w & (1 << 30)) {
    if (threadIdx.x == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA8) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_g) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmS_g) : "memory");
    }
    const int cut_a2 = static_cast<int>(static_cast<int16_t>(w0.z & 0xFFFF));
    const int cut_b2 = static_cast<int>(static_cast<int16_t>((w0.z >> 16) & 0xFFFF));
    down_tile<8, false, 4096, kShflEpi, kPdlMode, 24>(
        &tmA8, &tmB_g, &tmS_g, Cg, scratch, gs_g_ptr[0], w0.x, w0.y, w0.w, w1.x, w1.y, w1.z, 0,
        nullptr, cut_a2, cut_b2);
  } else {
    if (threadIdx.x == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA8) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB_q) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(&tmS_q) : "memory");
    }
    down_tile<8, false, 4608, kShflEpi, kPdlMode, 24>(
        &tmA8, &tmB_q, &tmS_q, Cq, scratch, gs_q_ptr[0], w0.x, w0.y, 0, -1, -1, w1.z, 0,
        nullptr, -1, -1);
  }
}
#else
// ---------------------------------------------------------------------------------------------------------------
// Fused gate/up + SwiGLU. blockIdx = 4 * column tile + 128-row chunk; a column tile is 64 gate and the matching 64
// up columns. Epilogue as marlinw_swiglu2.
constexpr int kTiles = 6656 / 64;
constexpr int kFused = 39936;

// The epilogue evaluates SiLU with the instruction sequence that fills the table of the 128-row kernels
// (swiglu_silu, on the gate already rounded to BF16) instead of reading the table. silu_check runs both for every
// BF16 pattern and counts the patterns whose FP32 results differ in any bit; swiglu_prepare refuses the library
// unless that count is 0.
__device__ int silu_differences;
__global__ void silu_check() {
  const unsigned word = 256u * blockIdx.x + threadIdx.x;
  const float direct = swiglu_silu(__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(word))));
  if (__float_as_uint(direct) != __float_as_uint(bf16_silu[word])) atomicAdd(&silu_differences, 1);
}

template <int kPdlMode = 0, int kTilesPerCta = 1, bool kCsOut = false, bool kPolEf = false>
__global__ void __launch_bounds__(kThreads, 1) gateup(
    const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
    const __grid_constant__ CUtensorMap tmS, char* __restrict__ out,
    const float* __restrict__ global_scale_ptr,
    const uint8_t* __restrict__ w_raw
#ifdef MARLINW_TMA_FP8
    , const float* __restrict__ down_input_scale
#endif
    ) {
  constexpr int kEpiExtraVal = kEpiExtraFor<kTilesPerCta>();
  extern __shared__ int4 sh[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int kw = warp >> 2, cw = (warp >> 1) & 1, jh = warp & 1;
  const int chunk = blockIdx.x & 3, col0 = (blockIdx.x >> 2) * kTilesPerCta;
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmS) : "memory");
  }
  if (blockIdx.x == 0 && w_raw != nullptr) {
    constexpr size_t kTpf = 2048u << 10;
    constexpr size_t kWBytes = (static_cast<size_t>(kFused) * 6656u) >> 1;
    for (size_t off = static_cast<size_t>(tid) * kTpf; off < kWBytes;
         off += static_cast<size_t>(kThreads) * kTpf) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(w_raw + off));
    }
  }
  const float gs = global_scale_ptr[0];
  float c[8][2][2][4];
  // Stage: [activations 16 KiB][gate codes 4 x 512 B][up codes 4 x 512 B][gate scales 4 x 64 B][up scales 4 x 64 B]
  // q: position in this CTA's k-tile stream = kTiles * (tile within the CTA) + k-tile.
  auto tma_bs = [&](uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar) {
    if constexpr (kPolEf) {
      tma2d_hint(dst, map, x, y, bar, tma_pol_ef());
    } else {
      tma2d(dst, map, x, y, bar);
    }
  };
  auto produce = [&](const int q, const uint32_t dst, const uint32_t bar, const int mode) {
    const int ti = (kTilesPerCta == 1) ? 0 : (q / kTiles);
    const int g = (kTilesPerCta == 1) ? q : (q - ti * kTiles);
    const int slice_col = col0 + ti;
    if (mode == 0) {
      mbar_expect_tx(bar, kABytes + kBBytes + kSBytes);
      tma2d_hint(dst, &tmA, 128 * g, 128 * chunk, bar, tma_pol_el());                          // bytes, rows
      tma_bs(dst + kABytes, &tmB, 128 * slice_col, 4 * g, bar);                                // uint32, k16 rows
      tma_bs(dst + kABytes + kBBytes / 2, &tmB, 128 * slice_col + kFused, 4 * g, bar);
      tma_bs(dst + kABytes + kBBytes, &tmS, 64 * slice_col, 4 * g, bar);                       // bytes, k16 rows
      tma_bs(dst + kABytes + kBBytes + kSBytes / 2, &tmS, 64 * slice_col + kFused / 2, 4 * g, bar);
    } else if (mode == 1) {
      mbar_expect_tx_only(bar, kBBytes + kSBytes);
      tma_bs(dst + kABytes, &tmB, 128 * slice_col, 4 * g, bar);
      tma_bs(dst + kABytes + kBBytes / 2, &tmB, 128 * slice_col + kFused, 4 * g, bar);
      tma_bs(dst + kABytes + kBBytes, &tmS, 64 * slice_col, 4 * g, bar);
      tma_bs(dst + kABytes + kBBytes + kSBytes / 2, &tmS, 64 * slice_col + kFused / 2, 4 * g, bar);
    } else {
      mbar_expect_tx(bar, kABytes);
      tma2d_hint(dst, &tmA, 128 * g, 128 * chunk, bar, tma_pol_el());
    }
  };
  // codes: int4 (128 cw + 32 g + lane), int2 jh of it; scales: int 64 cw + 16 g + 2 (lane / 4) + jh
  // Epilogue staging: the start of the window, or (several tiles per CTA) the last stage plus the bytes after it.
  int4* const epi = kTilesPerCta == 1 ? sh : reinterpret_cast<int4*>(
      reinterpret_cast<char*>(sh) + (((static_cast<uint32_t>(__cvta_generic_to_shared(sh)) + 1023u) & ~1023u) -
                                     static_cast<uint32_t>(__cvta_generic_to_shared(sh))) + (kStages - 1) * kStageBytes);
#pragma unroll 1
  for (int ti = 0; ti < kTilesPerCta; ti++) {
  const int slice_col = col0 + ti;
  k_loop<8, (kPdlMode == 2), (kPdlMode != 0), kEpiExtraVal>(
      c, kTiles, kABytes / 4 + 4 * (128 * cw + lane) + 2 * jh, 128,
      (kABytes + kBBytes) / 4 + 64 * cw + 2 * (lane >> 2) + jh, 16, produce, [](int) { return kTiles; }, ti * kTiles,
      kTilesPerCta * kTiles);
  __syncthreads();

  // Reduce the two k-warps: P = A0 (kw 0) + A1 (kw 1).
  int4* red = epi;
  const int t128 = tid & 127;
  if constexpr (kTilesPerCta == 1) {
    if (kw == 1) {
#pragma unroll
      for (int q = 0; q < 32; q++) red[q * 128 + t128] = reinterpret_cast<int4*>(c)[q];
    }
    __syncthreads();
    if (kw == 0) {
#pragma unroll
      for (int q = 0; q < 32; q++) {
        const int4 v = red[q * 128 + t128];
        float* cf = reinterpret_cast<float*>(c) + 4 * q;
        cf[0] += __int_as_float(v.x);
        cf[1] += __int_as_float(v.y);
        cf[2] += __int_as_float(v.z);
        cf[3] += __int_as_float(v.w);
      }
    }
    __syncthreads();
  } else {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      if (kw == 1) {
#pragma unroll
        for (int q = 0; q < 16; q++) red[q * 128 + t128] = reinterpret_cast<int4*>(&c[4 * half])[q];
      }
      __syncthreads();
      if (kw == 0) {
#pragma unroll
        for (int q = 0; q < 16; q++) {
          const int4 v = red[q * 128 + t128];
          float* cf = reinterpret_cast<float*>(&c[4 * half]) + 4 * q;
          cf[0] += __int_as_float(v.x);
          cf[1] += __int_as_float(v.y);
          cf[2] += __int_as_float(v.z);
          cf[3] += __int_as_float(v.w);
        }
      }
      __syncthreads();
    }
  }

  if (kw == 0) {
    // words (bf16x2) in rows of 68 words
    uint32_t* shw = reinterpret_cast<uint32_t*>(epi);
    int wr = 68 * (lane >> 2) + (lane & 3) + 32 * cw;
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
      for (int jl = 0; jl < 2; jl++) {
        const int j = 2 * jh + jl;
        auto put = [&](int idx, float c0, float c1) {
          __nv_bfloat162 r = __halves2bfloat162(__float2bfloat16_rn(c0 * gs), __float2bfloat16_rn(c1 * gs));
          shw[idx] = *reinterpret_cast<uint32_t*>(&r);
        };
        const int w = wr + 8 * j;
        put(w, c[i][jl][0][0], c[i][jl][0][1]);
        put(w + 68 * 8, c[i][jl][0][2], c[i][jl][0][3]);
        put(w + 4, c[i][jl][1][0], c[i][jl][1][1]);
        put(w + 68 * 8 + 4, c[i][jl][1][2], c[i][jl][1][3]);
      }
      wr += 16 * 68;
    }
  }
  __syncthreads();
  {
    const uint32_t* shw = reinterpret_cast<const uint32_t*>(epi);
#ifdef MARLINW_TMA_FP8
    float inverse;
    asm("div.full.f32 %0, 0f3F800000, %1;" : "=f"(inverse) : "f"(*down_input_scale));
    uint16_t* o8 = reinterpret_cast<uint16_t*>(out);
#pragma unroll 4
    for (int row = warp; row < 128; row += 8) {
      uint32_t gates = shw[row * 68 + lane], ups = shw[row * 68 + lane + 32];
      float scaled[2];
#pragma unroll
      for (int e = 0; e < 2; e++) {
        auto gate = reinterpret_cast<const __nv_bfloat16*>(&gates)[e];
        auto up = reinterpret_cast<const __nv_bfloat16*>(&ups)[e];
        const float silu = swiglu_silu(__bfloat162float(gate));
        const float product = __fmul_rn(silu, __bfloat162float(up));
        const float q = __fmul_rn(product, inverse);
        scaled[e] = q < -448.0f ? -448.0f : (q > 448.0f ? 448.0f : q);
      }
      uint16_t converted;
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(converted) : "f"(scaled[1]), "f"(scaled[0]));
      uint16_t* dst_p = &o8[(size_t)(chunk * 128 + row) * (kFused / 4) + slice_col * 32 + lane];
      if constexpr (kCsOut) {
        asm volatile("st.global.cs.u16 [%0], %1;" :: "l"(dst_p), "h"(converted) : "memory");
      } else {
        *dst_p = converted;
      }
    }
#else
    uint32_t* o = reinterpret_cast<uint32_t*>(out);
#pragma unroll 4
    for (int row = warp; row < 128; row += 8) {
      uint32_t gates = shw[row * 68 + lane], ups = shw[row * 68 + lane + 32];
      uint32_t prod;
#pragma unroll
      for (int e = 0; e < 2; e++) {
        auto gate = reinterpret_cast<const __nv_bfloat16*>(&gates)[e];
        auto up = reinterpret_cast<const __nv_bfloat16*>(&ups)[e];
        // swiglu_product with the table entry replaced by its definition (silu_check: equal for all patterns)
        reinterpret_cast<__nv_bfloat16*>(&prod)[e] =
            __float2bfloat16_rn(__fmul_rn(swiglu_silu(__bfloat162float(gate)), __bfloat162float(up)));
      }
      uint32_t* dst_p = &o[(size_t)(chunk * 128 + row) * (kFused / 4) + slice_col * 32 + lane];
      if constexpr (kCsOut) {
        asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(dst_p), "r"(prod) : "memory");
      } else {
        *dst_p = prod;
      }
    }
#endif
  }
  if constexpr (kTilesPerCta > 1) {
    // Ensure all warps finish reading `epi` (which aliases stage 3) before warp 0 issues stage 3 TMA for ti + 1.
    if (ti + 1 < kTilesPerCta) __syncthreads();
  }
  }  // tiles of this CTA
}

// Wave-54: 8-warp zero-spill interleaved [g16 | u16] 3-TMA gate/up + double-buf fa[2][8] + Dual PDL + 8-warp reg-only SwiGLU.
// - 8 warps (256 threads) keeps SM121's 255-reg hardware cap (avoiding 9-warp's 168-reg scheduler-0 spill cliff!).
// - Double-buffered fa[2][8] (kSingleFa=false) avoids fa[8]'s +10.8% LDSM WAR stall inside the HMMA stream.
// - Dual PDL (kPdlMode=2) preloads all 3 stages of tmB+tmS for Tile 0 BEFORE pdl_wait(), hiding post_attn_norm (-1.7 ms!).
// - Contiguous wave-strided slice_col = col_b + ti * kColStride preserves linear DRAM/L2/TLB traversal.
// - 8-warp register-only SwiGLU epilogue (kEpi8w=true): kw==0 reduces & owns m-blocks 0..3 in c[0..3], kw==1 reduces &
//   owns m-blocks 4..7 moved into c[0..3] (exact A0+A1 FP32 order, zero shw, zero dynamic c indexing, zero spills!).
//   Unconditional __syncthreads() eliminates ptxas ti==T-1 epilogue peeling (240 MUFU -> 64 MUFU = 3.75x smaller epilogue),
//   runs all 8 warps (2 warps/sub-partition) in parallel, saves 1 barrier/wave, and frees c[4..7] (64 regs) before SwiGLU!
template <int kPdlMode = 2, int kTilesPerCta = 26, bool kCsOut = true, bool kPolEf = false,
          bool kSingleFa = false, bool kEpi8w = true, bool kLutSilu = true>
__global__ void __launch_bounds__(kThreads, 1) gateup_ilv(
    const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
    const __grid_constant__ CUtensorMap tmS, char* __restrict__ out,
    const float* __restrict__ global_scale_ptr,
    const uint8_t* __restrict__ w_raw
#ifdef MARLINW_TMA_FP8
    , const float* __restrict__ down_input_scale
#endif
    ) {
  constexpr int kEpiExtraVal = kEpiExtraFor<kTilesPerCta>();
  constexpr int kColStride = (kFused / 128) / kTilesPerCta;  // 12 for T=26, 24 for T=13, 156 for T=2, 312 for T=1
  extern __shared__ int4 sh[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int kw = warp >> 2, nb = (warp >> 1) & 1, sh2 = warp & 1;
  const int chunk = blockIdx.x & 3, col_b = blockIdx.x >> 2;
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmA) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmB) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmS) : "memory");
  }
  if (blockIdx.x == 0 && w_raw != nullptr) {
    constexpr size_t kTpf = 2048u << 10;
    constexpr size_t kWBytes = (static_cast<size_t>(kFused) * 6656u) >> 1;
    for (size_t off = static_cast<size_t>(tid) * kTpf; off < kWBytes;
         off += static_cast<size_t>(kThreads) * kTpf) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(w_raw + off));
    }
  }
  const float gs = global_scale_ptr[0];
#ifdef MARLINW_TMA_FP8
  float inverse;
  asm("div.full.f32 %0, 0f3F800000, %1;" : "=f"(inverse) : "f"(*down_input_scale));
#endif
  float c[8][2][2][4];
  auto tma_bs = [&](uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar) {
    if constexpr (kPolEf) {
      tma2d_hint(dst, map, x, y, bar, tma_pol_ef());
    } else {
      tma2d(dst, map, x, y, bar);
    }
  };
  int4* const epi = kTilesPerCta == 1 ? sh : reinterpret_cast<int4*>(
      reinterpret_cast<char*>(sh) + (((static_cast<uint32_t>(__cvta_generic_to_shared(sh)) + 1023u) & ~1023u) -
                                     static_cast<uint32_t>(__cvta_generic_to_shared(sh))) + (kStages - 1) * kStageBytes);
#pragma unroll 1
  for (int ti = 0; ti < kTilesPerCta; ti++) {
  const int slice_col = col_b + ti * kColStride;
  const int q_base = ti * kTiles;
  auto produce = [&](const int q, const uint32_t dst, const uint32_t bar, const int mode) {
    int g = (kTilesPerCta == 1) ? q : (q - q_base);
    int sc = slice_col;
    if constexpr (kTilesPerCta > 1) {
      if (g >= kTiles) {
        g -= kTiles;
        sc += kColStride;
      }
    }
    if (mode == 0) {
      mbar_expect_tx(bar, kABytes + kBBytes + kSBytes);
      tma2d_hint(dst, &tmA, 128 * g, 128 * chunk, bar, tma_pol_el());
      tma_bs(dst + kABytes, &tmB, 256 * sc, 4 * g, bar);
      tma_bs(dst + kABytes + kBBytes, &tmS, 128 * sc, 4 * g, bar);
    } else if (mode == 1) {
      mbar_expect_tx_only(bar, kBBytes + kSBytes);
      tma_bs(dst + kABytes, &tmB, 256 * sc, 4 * g, bar);
      tma_bs(dst + kABytes + kBBytes, &tmS, 128 * sc, 4 * g, bar);
    } else {
      mbar_expect_tx(bar, kABytes);
      tma2d_hint(dst, &tmA, 128 * g, 128 * chunk, bar, tma_pol_el());
    }
  };
  k_loop<8, (kPdlMode == 2), (kPdlMode != 0), kEpiExtraVal, kSingleFa>(
      c, kTiles, kABytes / 4 + 4 * (32 * nb + lane) + 2 * sh2, 256,
      (kABytes + kBBytes) / 4 + 2 * (8 * nb + (lane >> 2)) + sh2, 32, produce, [](int) { return kTiles; },
      q_base, kTilesPerCta * kTiles);
  __syncthreads();

  int4* red = epi;
  const int t128 = tid & 127;
  if constexpr (kEpi8w) {
    if constexpr (kTilesPerCta == 1) {
      if (kw == 1) {
#pragma unroll
        for (int q = 0; q < 16; q++) red[q * 128 + t128] = reinterpret_cast<int4*>(&c[0])[q];
      } else {
#pragma unroll
        for (int q = 0; q < 16; q++) red[(16 + q) * 128 + t128] = reinterpret_cast<int4*>(&c[4])[q];
      }
      __syncthreads();
      if (kw == 0) {
#pragma unroll
        for (int q = 0; q < 16; q++) {
          const int4 v = red[q * 128 + t128];
          float* cf = reinterpret_cast<float*>(&c[0]) + 4 * q;
          cf[0] += __int_as_float(v.x);
          cf[1] += __int_as_float(v.y);
          cf[2] += __int_as_float(v.z);
          cf[3] += __int_as_float(v.w);
        }
      } else {
#pragma unroll
        for (int q = 0; q < 16; q++) {
          const int4 v = red[(16 + q) * 128 + t128];
          const float* c4 = reinterpret_cast<const float*>(&c[4]) + 4 * q;
          float* c0 = reinterpret_cast<float*>(&c[0]) + 4 * q;
          c0[0] = __int_as_float(v.x) + c4[0];
          c0[1] = __int_as_float(v.y) + c4[1];
          c0[2] = __int_as_float(v.z) + c4[2];
          c0[3] = __int_as_float(v.w) + c4[3];
        }
      }
    } else {
      if (kw == 1) {
#pragma unroll
        for (int q = 0; q < 16; q++) red[q * 128 + t128] = reinterpret_cast<int4*>(&c[0])[q];
      }
      __syncthreads();
      if (kw == 0) {
#pragma unroll
        for (int q = 0; q < 16; q++) {
          const int4 v = red[q * 128 + t128];
          float* cf = reinterpret_cast<float*>(&c[0]) + 4 * q;
          cf[0] += __int_as_float(v.x);
          cf[1] += __int_as_float(v.y);
          cf[2] += __int_as_float(v.z);
          cf[3] += __int_as_float(v.w);
          red[q * 128 + t128] = reinterpret_cast<int4*>(&c[4])[q];
        }
      }
      __syncthreads();
      if (kw == 1) {
#pragma unroll
        for (int q = 0; q < 16; q++) {
          const int4 v = red[q * 128 + t128];
          const float* c4 = reinterpret_cast<const float*>(&c[4]) + 4 * q;
          float* c0 = reinterpret_cast<float*>(&c[0]) + 4 * q;
          c0[0] = __int_as_float(v.x) + c4[0];
          c0[1] = __int_as_float(v.y) + c4[1];
          c0[2] = __int_as_float(v.z) + c4[2];
          c0[3] = __int_as_float(v.w) + c4[3];
        }
      }
      // Unconditional barrier prevents ptxas ti==T-1 epilogue peeling and frees epi (stage 3) before ti+1.
      __syncthreads();
    }
  } else {
    if constexpr (kTilesPerCta == 1) {
      if (kw == 1) {
#pragma unroll
        for (int q = 0; q < 32; q++) red[q * 128 + t128] = reinterpret_cast<int4*>(c)[q];
      }
      __syncthreads();
      if (kw == 0) {
#pragma unroll
        for (int q = 0; q < 32; q++) {
          const int4 v = red[q * 128 + t128];
          float* cf = reinterpret_cast<float*>(c) + 4 * q;
          cf[0] += __int_as_float(v.x);
          cf[1] += __int_as_float(v.y);
          cf[2] += __int_as_float(v.z);
          cf[3] += __int_as_float(v.w);
        }
      }
    } else {
#pragma unroll
      for (int half = 0; half < 2; half++) {
        if (kw == 1) {
#pragma unroll
          for (int q = 0; q < 16; q++) red[q * 128 + t128] = reinterpret_cast<int4*>(&c[4 * half])[q];
        }
        __syncthreads();
        if (kw == 0) {
#pragma unroll
          for (int q = 0; q < 16; q++) {
            const int4 v = red[q * 128 + t128];
            float* cf = reinterpret_cast<float*>(&c[4 * half]) + 4 * q;
            cf[0] += __int_as_float(v.x);
            cf[1] += __int_as_float(v.y);
            cf[2] += __int_as_float(v.z);
            cf[3] += __int_as_float(v.w);
          }
        }
        // Unconditional barrier eliminates ptxas ti==T-1 epilogue duplication (240 -> 128 MUFU).
        __syncthreads();
      }
    }
  }

  // `epi` (stage 3) is now free. With kEpi8w=true, all 8 warps run 4 iterations (32 silus each) in parallel
  // with c[0..3] holding rows 0..63 (kw==0) and 64..127 (kw==1).
  if (kEpi8w || kw == 0) {
    constexpr int kMBlocks = kEpi8w ? 4 : 8;
    const int q = lane & 3;
    const int r0 = 128 * chunk + (kEpi8w ? 64 * kw : 0) + (lane >> 2);
    const int out_col0 = 64 * slice_col + 16 * (2 * nb + sh2);
    auto eval_silu = [&](__nv_bfloat16 bg) -> float {
      if constexpr (kLutSilu) return __ldg(bf16_silu + __bfloat16_as_ushort(bg));
      else                    return swiglu_silu(__bfloat162float(bg));
    };
#ifdef MARLINW_TMA_FP8
    uint8_t* out_u8 = reinterpret_cast<uint8_t*>(out);
    auto eval_fp8_pair = [&](float g0, float u0, float g1, float u1) -> uint32_t {
      const __nv_bfloat16 bg0 = __float2bfloat16_rn(g0 * gs);
      const __nv_bfloat16 bu0 = __float2bfloat16_rn(u0 * gs);
      const __nv_bfloat16 bg1 = __float2bfloat16_rn(g1 * gs);
      const __nv_bfloat16 bu1 = __float2bfloat16_rn(u1 * gs);
      const float q0 = __fmul_rn(__fmul_rn(eval_silu(bg0), __bfloat162float(bu0)), inverse);
      const float q1 = __fmul_rn(__fmul_rn(eval_silu(bg1), __bfloat162float(bu1)), inverse);
      const float s0 = q0 < -448.0f ? -448.0f : (q0 > 448.0f ? 448.0f : q0);
      const float s1 = q1 < -448.0f ? -448.0f : (q1 > 448.0f ? 448.0f : q1);
      uint16_t conv;
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(conv) : "f"(s1), "f"(s0));
      return static_cast<uint32_t>(conv);
    };
#pragma unroll
    for (int i = 0; i < kMBlocks; i++) {
      const uint32_t v0 = eval_fp8_pair(c[i][0][0][0], c[i][1][0][0], c[i][0][0][1], c[i][1][0][1]);
      const uint32_t v1 = eval_fp8_pair(c[i][0][0][2], c[i][1][0][2], c[i][0][0][3], c[i][1][0][3]);
      const uint32_t v2 = eval_fp8_pair(c[i][0][1][0], c[i][1][1][0], c[i][0][1][1], c[i][1][1][1]);
      const uint32_t v3 = eval_fp8_pair(c[i][0][1][2], c[i][1][1][2], c[i][0][1][3], c[i][1][1][3]);
      const uint32_t r1 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 1), 1);
      const uint32_t r2 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 2), 2);
      const uint32_t r3 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 3), 3);
      const uint32_t own = sel4_u32(v0, v1, v2, v3, q);
      const uint32_t x0 = sel4_u32(own, r1, r2, r3, 0 ^ q);
      const uint32_t x1 = sel4_u32(own, r1, r2, r3, 1 ^ q);
      const uint32_t x2 = sel4_u32(own, r1, r2, r3, 2 ^ q);
      const uint32_t x3 = sel4_u32(own, r1, r2, r3, 3 ^ q);
      const uint32_t w0 = x0 | (x1 << 16);
      const uint32_t w1 = x2 | (x3 << 16);
      const int r_out = r0 + 16 * i + (q & 1) * 8;
      const int c_out = out_col0 + (q >> 1) * 8;
      uint2* dst_p = reinterpret_cast<uint2*>(&out_u8[(size_t)r_out * (kFused / 2) + c_out]);
      if constexpr (kCsOut) {
        asm volatile("st.global.cs.v2.u32 [%0], {%1,%2};" :: "l"(dst_p), "r"(w0), "r"(w1) : "memory");
      } else {
        *dst_p = make_uint2(w0, w1);
      }
    }
#else
    __nv_bfloat16* out_bf16 = reinterpret_cast<__nv_bfloat16*>(out);
    auto eval_bf16_pair = [&](float g0, float u0, float g1, float u1) -> uint32_t {
      const __nv_bfloat16 bg0 = __float2bfloat16_rn(g0 * gs);
      const __nv_bfloat16 bu0 = __float2bfloat16_rn(u0 * gs);
      const __nv_bfloat16 bg1 = __float2bfloat16_rn(g1 * gs);
      const __nv_bfloat16 bu1 = __float2bfloat16_rn(u1 * gs);
      const __nv_bfloat16 p0 = __float2bfloat16_rn(__fmul_rn(eval_silu(bg0), __bfloat162float(bu0)));
      const __nv_bfloat16 p1 = __float2bfloat16_rn(__fmul_rn(eval_silu(bg1), __bfloat162float(bu1)));
      const __nv_bfloat162 r = __halves2bfloat162(p0, p1);
      return *reinterpret_cast<const uint32_t*>(&r);
    };
#pragma unroll
    for (int i = 0; i < kMBlocks; i++) {
      const uint32_t v0 = eval_bf16_pair(c[i][0][0][0], c[i][1][0][0], c[i][0][0][1], c[i][1][0][1]);
      const uint32_t v1 = eval_bf16_pair(c[i][0][0][2], c[i][1][0][2], c[i][0][0][3], c[i][1][0][3]);
      const uint32_t v2 = eval_bf16_pair(c[i][0][1][0], c[i][1][1][0], c[i][0][1][1], c[i][1][1][1]);
      const uint32_t v3 = eval_bf16_pair(c[i][0][1][2], c[i][1][1][2], c[i][0][1][3], c[i][1][1][3]);
      const uint32_t r1 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 1), 1);
      const uint32_t r2 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 2), 2);
      const uint32_t r3 = __shfl_xor_sync(0xffffffffu, sel4_u32(v0, v1, v2, v3, q ^ 3), 3);
      const uint32_t own = sel4_u32(v0, v1, v2, v3, q);
      uint4 w;
      w.x = sel4_u32(own, r1, r2, r3, 0 ^ q);
      w.y = sel4_u32(own, r1, r2, r3, 1 ^ q);
      w.z = sel4_u32(own, r1, r2, r3, 2 ^ q);
      w.w = sel4_u32(own, r1, r2, r3, 3 ^ q);
      const int r_out = r0 + 16 * i + (q & 1) * 8;
      const int c_out = out_col0 + (q >> 1) * 8;
      uint4* dst_p = reinterpret_cast<uint4*>(&out_bf16[(size_t)r_out * (kFused / 2) + c_out]);
      if constexpr (kCsOut) {
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
                     :: "l"(dst_p), "r"(w.x), "r"(w.y), "r"(w.z), "r"(w.w) : "memory");
      } else {
        *dst_p = w;
      }
    }
#endif
  }
  }  // tiles of this CTA
}
#endif

// ---- host side: tensor maps --------------------------------------------------------------------------------------
typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                             const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                             CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn encode = nullptr;

static cudaError_t find_encode() {
  if (encode) return cudaSuccess;
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult q;
  cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
  if (e != cudaSuccess) return e;
  if (!fn || q != cudaDriverEntryPointSuccess) return cudaErrorNotSupported;
  encode = (EncodeFn)fn;
  return cudaSuccess;
}
// A 2-D map over `rows` rows of `width` elements of `esize` bytes, `stride` bytes apart; box = bw x bh elements.
static bool map2d(CUtensorMap* tm, const void* ptr, int esize, uint64_t width, uint64_t rows, uint64_t stride,
                  uint32_t bw, uint32_t bh, bool swizzle) {
  const cuuint64_t dims[2] = {width, rows};
  const cuuint64_t strides[1] = {stride};
  const cuuint32_t box[2] = {bw, bh};
  const cuuint32_t es[2] = {1, 1};
  const CUtensorMapDataType dt = esize == 4 ? CU_TENSOR_MAP_DATA_TYPE_UINT32 : CU_TENSOR_MAP_DATA_TYPE_UINT8;
  return encode(tm, dt, 2, const_cast<void*>(ptr), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

static int g_tma_pdl_override = -1;
static int tma_pdl_mode() {
  if (g_tma_pdl_override >= 0) return g_tma_pdl_override;
  static const int env_mode = []() -> int {
#ifndef MARLINW_TMA_GATEUP
    constexpr int kDefault = 2;  // Dual SM121 PDL: preload 3 stages of tmB+tmS behind pdl_wait() (0 spills, same regs)
#else
    constexpr int kDefault = 1;  // legacy gateup fallback (gateup_ilv uses gateup_ilv_pdl_mode() = 2)
#endif
    const char* env = std::getenv("MARLINW_TMA_PDL");
    if (!env || !*env) return kDefault;
    long v = std::strtol(env, nullptr, 10);
    return (v == 0 || v == 1 || v == 2) ? static_cast<int>(v) : kDefault;
  }();
  return env_mode;
}

template <typename Kernel, typename... Args>
static cudaError_t launch_ex_pdl(Kernel kernel, int blocks, int threads, size_t smem, cudaStream_t st,
                                 const Args&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks, 1, 1);
  cfg.blockDim = dim3(threads, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace MARLIN_NAMESPACE_NAME

extern "C" int marlinw_tma_set_pdl(int mode) {
  const int old = MARLIN_NAMESPACE_NAME::tma_pdl_mode();
  MARLIN_NAMESPACE_NAME::g_tma_pdl_override = (mode == 0 || mode == 1 || mode == 2) ? mode : -1;
  return old;
}

#ifndef MARLINW_TMA_GATEUP
// Same C ABI as libmarlinw_down128.so, plus multi-shape TMA entrypoints (qkv 4608x6656, o_proj 6656x4096, gate 4096x6656).
static bool g_shfl_epi = true;

extern "C" void marlinw_down128_set_shfl_epi(int enable) {
  g_shfl_epi = (enable != 0);
}

template <int kCols>
static int init_down_shape(int* active_blocks) {
  using namespace MARLIN_NAMESPACE_NAME;
  auto status = find_encode();
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(down<kCols, true, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(down<kCols, false, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(down<kCols, true, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(down<kCols, false, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(down<kCols, true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(down<kCols, false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  int occ_shfl = 0, occ_stock = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_shfl, down<kCols, true, 1>, kThreads, kShared);
  if (status != cudaSuccess) return status;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_stock, down<kCols, false, 1>, kThreads, kShared);
  if (status != cudaSuccess) return status;
  *active_blocks = occ_shfl < occ_stock ? occ_shfl : occ_stock;
  return cudaSuccess;
}

template <int kCols>
static int launch_down_shape(
    const int4* input, const int4* weight, int4* output, float4* scratch, const int4* scales,
    const float* global_scale, const int4* work, int grid, int k, int row_stride, cudaStream_t stream) {
  using namespace MARLIN_NAMESPACE_NAME;
  if (!encode || row_stride < k || row_stride % 8 || k % 64 != 0) return cudaErrorInvalidValue;
  if (((uintptr_t)input | (uintptr_t)weight | (uintptr_t)scales) & 15) return cudaErrorInvalidValue;
  CUtensorMap tmA8, tmB, tmS;
  if (!map2d(&tmA8, input, 1, 2 * (uint64_t)k, 512, 2 * (uint64_t)row_stride, 128, 128, true) ||
      !map2d(&tmB, weight, 4, 2 * (uint64_t)kCols, (uint64_t)k / 16, 8 * (uint64_t)kCols, 256, 4, false) ||
      !map2d(&tmS, scales, 1, (uint64_t)kCols, (uint64_t)k / 16, (uint64_t)kCols, 128, 4, false))
    return cudaErrorInvalidValue;
  const int pm = tma_pdl_mode();
  const size_t w_bytes = (static_cast<size_t>(kCols) * static_cast<size_t>(k)) >> 1;
  const uint8_t* w_raw = reinterpret_cast<const uint8_t*>(weight);
  if (pm == 2) {
    auto kernel = g_shfl_epi ? down<kCols, true, 2> : down<kCols, false, 2>;
    return launch_ex_pdl(kernel, grid, kThreads, kShared, stream, tmA8, tmB, tmS, output, scratch, global_scale,
                         const_cast<int4*>(work), w_raw, w_bytes);
  }
  auto kernel = (pm == 1) ? (g_shfl_epi ? down<kCols, true, 1> : down<kCols, false, 1>)
                          : (g_shfl_epi ? down<kCols, true, 0> : down<kCols, false, 0>);
  kernel<<<grid, kThreads, kShared, stream>>>(tmA8, tmB, tmS, output, scratch, global_scale,
                                              const_cast<int4*>(work), w_raw, w_bytes);
  return cudaGetLastError();
}

extern "C" int marlinw_down128_init(int* active_blocks) {
  return init_down_shape<6656>(active_blocks);
}
extern "C" int marlinw_down128_main(
    const int4* input, const int4* weight, int4* output, float4* scratch, const int4* scales,
    const float* global_scale, const int4* work, int grid, int row_stride, cudaStream_t stream) {
  return launch_down_shape<6656>(input, weight, output, scratch, scales, global_scale, work, grid, 19968,
                                 row_stride, stream);
}
extern "C" int marlinw_down128_shape_tma_init(int n, int k, int* active_blocks) {
  if (k % 64 != 0) return cudaErrorInvalidValue;
  if (n == 6656) return init_down_shape<6656>(active_blocks);
  if (n == 4608) return init_down_shape<4608>(active_blocks);
  if (n == 4096) return init_down_shape<4096>(active_blocks);
  return cudaErrorInvalidValue;
}
extern "C" int marlinw_down128_shape_tma_main(
    const int4* input, const int4* weight, int4* output, float4* scratch, const int4* scales,
    const float* global_scale, const int4* work, int grid, int n, int k, int row_stride, cudaStream_t stream) {
  if (n == 6656)
    return launch_down_shape<6656>(input, weight, output, scratch, scales, global_scale, work, grid, k, row_stride,
                                   stream);
  if (n == 4608)
    return launch_down_shape<4608>(input, weight, output, scratch, scales, global_scale, work, grid, k, row_stride,
                                   stream);
  if (n == 4096)
    return launch_down_shape<4096>(input, weight, output, scratch, scales, global_scale, work, grid, k, row_stride,
                                   stream);
  return cudaErrorInvalidValue;
}

static int init_down_dual(int* active_blocks) {
  using namespace MARLIN_NAMESPACE_NAME;
  auto status = find_encode();
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(down_dual<true, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(down_dual<false, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(down_dual<true, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(down_dual<false, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(down_dual<true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(down_dual<false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (status != cudaSuccess) return status;
  int occ_s1 = 0, occ_n1 = 0, occ_s2 = 0, occ_n2 = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_s1, down_dual<true, 1>, kThreads, kShared);
  if (status != cudaSuccess) return status;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_n1, down_dual<false, 1>, kThreads, kShared);
  if (status != cudaSuccess) return status;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_s2, down_dual<true, 2>, kThreads, kShared);
  if (status != cudaSuccess) return status;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_n2, down_dual<false, 2>, kThreads, kShared);
  if (status != cudaSuccess) return status;
  int occ = occ_s1 < occ_n1 ? occ_s1 : occ_n1;
  if (occ_s2 < occ) occ = occ_s2;
  if (occ_n2 < occ) occ = occ_n2;
  *active_blocks = occ;
  return cudaSuccess;
}

extern "C" int marlinw_down128_dual_tma_init(int* active_blocks) {
  return init_down_dual(active_blocks);
}

extern "C" int marlinw_down128_dual_tma_main(
    const int4* input,
    const int4* w_q, const int4* s_q, const float* gs_q, int4* out_q,
    const int4* w_g, const int4* s_g, const float* gs_g, int4* out_g,
    float4* scratch, const int4* work, int grid, int row_stride, cudaStream_t stream) {
  using namespace MARLIN_NAMESPACE_NAME;
  constexpr int kK = 6656;
  constexpr int kNq = 4608;
  constexpr int kNg = 4096;
  if (!encode || row_stride < kK || row_stride % 8) return cudaErrorInvalidValue;
  if (((uintptr_t)input | (uintptr_t)w_q | (uintptr_t)s_q | (uintptr_t)w_g | (uintptr_t)s_g) & 15)
    return cudaErrorInvalidValue;
  CUtensorMap tmA8, tmB_q, tmS_q, tmB_g, tmS_g;
  if (!map2d(&tmA8, input, 1, 2 * (uint64_t)kK, 512, 2 * (uint64_t)row_stride, 128, 128, true) ||
      !map2d(&tmB_q, w_q, 4, 2 * (uint64_t)kNq, (uint64_t)kK / 16, 8 * (uint64_t)kNq, 256, 4, false) ||
      !map2d(&tmS_q, s_q, 1, (uint64_t)kNq, (uint64_t)kK / 16, (uint64_t)kNq, 128, 4, false) ||
      !map2d(&tmB_g, w_g, 4, 2 * (uint64_t)kNg, (uint64_t)kK / 16, 8 * (uint64_t)kNg, 256, 4, false) ||
      !map2d(&tmS_g, s_g, 1, (uint64_t)kNg, (uint64_t)kK / 16, (uint64_t)kNg, 128, 4, false))
    return cudaErrorInvalidValue;
  const int pm = tma_pdl_mode();
  constexpr size_t wq_bytes = (static_cast<size_t>(kNq) * static_cast<size_t>(kK)) >> 1;
  constexpr size_t wg_bytes = (static_cast<size_t>(kNg) * static_cast<size_t>(kK)) >> 1;
  const uint8_t* wq_raw = reinterpret_cast<const uint8_t*>(w_q);
  const uint8_t* wg_raw = reinterpret_cast<const uint8_t*>(w_g);
  if (pm == 2) {
    auto kernel = g_shfl_epi ? down_dual<true, 2> : down_dual<false, 2>;
    return launch_ex_pdl(kernel, grid, kThreads, kShared, stream, tmA8, tmB_q, tmS_q, tmB_g, tmS_g,
                         out_q, out_g, scratch, gs_q, gs_g, const_cast<int4*>(work),
                         wq_raw, wq_bytes, wg_raw, wg_bytes);
  }
  auto kernel = (pm == 1) ? (g_shfl_epi ? down_dual<true, 1> : down_dual<false, 1>)
                          : (g_shfl_epi ? down_dual<true, 0> : down_dual<false, 0>);
  kernel<<<grid, kThreads, kShared, stream>>>(
      tmA8, tmB_q, tmS_q, tmB_g, tmS_g, out_q, out_g, scratch, gs_q, gs_g, const_cast<int4*>(work),
      wq_raw, wq_bytes, wg_raw, wg_bytes);
  return cudaGetLastError();
}
#else
namespace MARLIN_NAMESPACE_NAME {
static int gateup_tiles_per_cta() {
  static const int t = []() -> int {
#ifdef MARLINW_TMA_FP8
    const char* env = std::getenv("MARLINW_GATEUP_FP8_TILES");
    constexpr int kDefault = 1;  // 239 regs, 0 spills; avoids 255-reg penalty of T>1
#else
    const char* env = std::getenv("MARLINW_GATEUP_TILES");
    constexpr int kDefault = 1;  // 239 regs, 0 spills; avoids warm +1.6 ms 255-reg penalty of T=26
#endif
    if (!env || !*env) return kDefault;
    long v = std::strtol(env, nullptr, 10);
    if (v == 1 || v == 2 || v == 13 || v == 26) return static_cast<int>(v);
    return kDefault;
  }();
  return t;
}
static bool gateup_cs_out() {
  static const bool cs = []() -> bool {
    const char* env = std::getenv("MARLINW_GATEUP_CS_OUT");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  return cs;
}
static bool gateup_b_ef() {
  static const bool ef = []() -> bool {
#ifdef MARLINW_TMA_FP8
    const char* env = std::getenv("MARLINW_GATEUP_FP8_B_EF");
    constexpr bool kDefault = false;  // normal L2 policy on 4-chunk shared tmB/tmS (A stays evict_last)
#else
    const char* env = std::getenv("MARLINW_GATEUP_B_EF");
    constexpr bool kDefault = false;  // normal L2 policy on 4-chunk shared tmB/tmS (-4.5 ms vs B_EF=1 at T=26)
#endif
    if (!env || !*env) return kDefault;
    return std::strtol(env, nullptr, 10) != 0;
  }();
  return ef;
}
static int gateup_ilv_tiles_per_cta() {
  static const int t = []() -> int {
    const char* env = std::getenv("MARLINW_GATEUP_ILV_TILES");
    if (!env || !*env) return 26;  // persistent 48 CTAs with double-buf fa[2][8] + 8w reg-only SwiGLU
    long v = std::strtol(env, nullptr, 10);
    if (v == 1 || v == 2 || v == 13 || v == 26) return static_cast<int>(v);
    return 26;
  }();
  return t;
}
static bool gateup_ilv_b_ef() {
  static const bool ef = []() -> bool {
    const char* env = std::getenv("MARLINW_GATEUP_ILV_B_EF");
    if (!env || !*env) return false;
    return std::strtol(env, nullptr, 10) != 0;
  }();
  return ef;
}
static bool gateup_ilv_epi8w() {
  static const bool on = []() -> bool {
    const char* env = std::getenv("MARLINW_GATEUP_ILV_EPI8W");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  return on;
}
static bool gateup_ilv_lut_silu() {
  static const bool on = []() -> bool {
    const char* env = std::getenv("MARLINW_GATEUP_LUT_SILU");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  return on;
}
static int gateup_ilv_pdl_mode() {
  if (g_tma_pdl_override >= 0) return g_tma_pdl_override;
  static const int mode = []() -> int {
    const char* env = std::getenv("MARLINW_GATEUP_ILV_PDL");
    if (!env || !*env) return 2;  // Dual SM121 PDL: preload 3 stages of tmB+tmS behind pdl_wait()
    long v = std::strtol(env, nullptr, 10);
    return (v == 0 || v == 1 || v == 2) ? static_cast<int>(v) : 2;
  }();
  return mode;
}
static cudaError_t set_gateup_attrs() {
  cudaError_t s = cudaSuccess;
  auto set1 = [&](auto fn, int smem) {
    if (s == cudaSuccess) s = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  };
  set1(gateup<0, 1, false, true>, kSharedFor<1>());
  set1(gateup<1, 1, false, false>, kSharedFor<1>());
  set1(gateup<1, 1, false, true>, kSharedFor<1>());
  set1(gateup<1, 1, true, false>, kSharedFor<1>());
  set1(gateup<2, 1, false, true>, kSharedFor<1>());
  set1(gateup<1, 2, false, false>, kSharedFor<2>());
  set1(gateup<1, 2, false, true>, kSharedFor<2>());
  set1(gateup<1, 13, false, false>, kSharedFor<13>());
  set1(gateup<1, 13, false, true>, kSharedFor<13>());
  set1(gateup<1, 13, true, false>, kSharedFor<13>());
  set1(gateup<1, 26, false, false>, kSharedFor<26>());
  set1(gateup<1, 26, false, true>, kSharedFor<26>());
  set1(gateup<1, 26, true, false>, kSharedFor<26>());
  // Wave-54 gateup_ilv: double-buffered fa[2][8] (kSingleFa=false) + 8-warp reg-only SwiGLU (kEpi8w=true)
  set1(gateup_ilv<0, 26, true, false, false, true>, kSharedFor<26>());
  set1(gateup_ilv<1, 26, true, false, false, true>, kSharedFor<26>());
  set1(gateup_ilv<2, 26, true, false, false, true>, kSharedFor<26>());
  set1(gateup_ilv<2, 26, true, false, false, true, false>, kSharedFor<26>());
  set1(gateup_ilv<2, 26, true, true, false, true>, kSharedFor<26>());
  set1(gateup_ilv<2, 13, true, false, false, true>, kSharedFor<13>());
  set1(gateup_ilv<2, 2, true, false, false, true>, kSharedFor<2>());
  set1(gateup_ilv<2, 1, true, false, false, true>, kSharedFor<1>());
  set1(gateup_ilv<1, 1, true, false, false, true>, kSharedFor<1>());
  // 4-warp de-duplicated epilogue A/B variants (MARLINW_GATEUP_ILV_EPI8W=0)
  set1(gateup_ilv<2, 26, true, false, false, false>, kSharedFor<26>());
  set1(gateup_ilv<1, 26, true, false, false, false>, kSharedFor<26>());
  set1(gateup_ilv<1, 1, true, false, false, false>, kSharedFor<1>());
  return s;
}
template <typename... Extra>
static cudaError_t dispatch_gateup_ilv(int pm, int tpc, bool cs, bool ef, bool epi8w, bool lut_silu, cudaStream_t stream,
                                       const CUtensorMap& tmA, const CUtensorMap& tmB, const CUtensorMap& tmS,
                                       char* out, const float* global, const uint8_t* w_raw, Extra... extra) {
  constexpr int m = 512, n = kFused;
  (void)cs;
  auto launch = [&](auto tag_pm, auto tag_tpc, auto tag_ef, auto tag_e8) -> cudaError_t {
    constexpr int kP = decltype(tag_pm)::value;
    constexpr int kT = decltype(tag_tpc)::value;
    constexpr bool kE = decltype(tag_ef)::value;
    constexpr bool kE8 = decltype(tag_e8)::value;
    static_assert((n / 128) % kT == 0, "tiles per CTA");
    constexpr int grid = (m / 128) * (n / 128) / kT;
    if constexpr (kP == 2) {
      return launch_ex_pdl(gateup_ilv<2, kT, true, kE, false, kE8, true>, grid, kThreads, kSharedFor<kT>(), stream,
                           tmA, tmB, tmS, out, global, w_raw, extra...);
    } else {
      gateup_ilv<kP, kT, true, kE, false, kE8, true><<<grid, kThreads, kSharedFor<kT>(), stream>>>(
          tmA, tmB, tmS, out, global, w_raw, extra...);
      return cudaGetLastError();
    }
  };
  if (!epi8w) {
    if (tpc == 26) {
      return (pm == 2)
                 ? launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 26>{}, std::false_type{}, std::false_type{})
                 : launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 26>{}, std::false_type{}, std::false_type{});
    }
    return launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{}, std::false_type{}, std::false_type{});
  }
  if (tpc == 26) {
    if (pm == 0) return launch(std::integral_constant<int, 0>{}, std::integral_constant<int, 26>{}, std::false_type{}, std::true_type{});
    if (pm == 1) return launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 26>{}, std::false_type{}, std::true_type{});
    if (!ef && !lut_silu) {
      constexpr int grid = (m / 128) * (n / 128) / 26;
      return launch_ex_pdl(gateup_ilv<2, 26, true, false, false, true, false>, grid, kThreads, kSharedFor<26>(), stream,
                           tmA, tmB, tmS, out, global, w_raw, extra...);
    }
    return ef ? launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 26>{}, std::true_type{}, std::true_type{})
              : launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 26>{}, std::false_type{}, std::true_type{});
  }
  if (tpc == 13) return launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 13>{}, std::false_type{}, std::true_type{});
  if (tpc == 2)  return launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{}, std::false_type{}, std::true_type{});
  return (pm == 2)
             ? launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{}, std::false_type{}, std::true_type{})
             : launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{}, std::false_type{}, std::true_type{});
}
template <typename... Extra>
static cudaError_t dispatch_gateup(int pm, int tpc, bool cs, bool ef, cudaStream_t stream,
                                   const CUtensorMap& tmA, const CUtensorMap& tmB, const CUtensorMap& tmS,
                                   char* out, const float* global, const uint8_t* w_raw, Extra... extra) {
  constexpr int m = 512, n = kFused;
  if (pm == 2) {
    constexpr int grid = (m / 128) * (n / 128);
    return launch_ex_pdl(gateup<2, 1, false, true>, grid, kThreads, kSharedFor<1>(), stream,
                         tmA, tmB, tmS, out, global, w_raw, extra...);
  }
  if (pm == 0) {
    constexpr int grid = (m / 128) * (n / 128);
    gateup<0, 1, false, true><<<grid, kThreads, kSharedFor<1>(), stream>>>(
        tmA, tmB, tmS, out, global, w_raw, extra...);
    return cudaGetLastError();
  }
  auto run = [&](auto tag_tpc, auto tag_cs, auto tag_ef) -> cudaError_t {
    constexpr int kT = decltype(tag_tpc)::value;
    constexpr bool kC = decltype(tag_cs)::value;
    constexpr bool kE = decltype(tag_ef)::value;
    static_assert((n / 128) % kT == 0, "tiles per CTA");
    constexpr int grid = (m / 128) * (n / 128) / kT;
    gateup<1, kT, kC, kE><<<grid, kThreads, kSharedFor<kT>(), stream>>>(
        tmA, tmB, tmS, out, global, w_raw, extra...);
    return cudaGetLastError();
  };
  if (tpc == 26) {
    if (cs) return run(std::integral_constant<int, 26>{}, std::true_type{}, std::false_type{});
    return ef ? run(std::integral_constant<int, 26>{}, std::false_type{}, std::true_type{})
              : run(std::integral_constant<int, 26>{}, std::false_type{}, std::false_type{});
  }
  if (tpc == 13) {
    if (cs) return run(std::integral_constant<int, 13>{}, std::true_type{}, std::false_type{});
    return ef ? run(std::integral_constant<int, 13>{}, std::false_type{}, std::true_type{})
              : run(std::integral_constant<int, 13>{}, std::false_type{}, std::false_type{});
  }
  if (tpc == 2) {
    return ef ? run(std::integral_constant<int, 2>{}, std::false_type{}, std::true_type{})
              : run(std::integral_constant<int, 2>{}, std::false_type{}, std::false_type{});
  }
  if (cs) return run(std::integral_constant<int, 1>{}, std::true_type{}, std::false_type{});
  return ef ? run(std::integral_constant<int, 1>{}, std::false_type{}, std::true_type{})
            : run(std::integral_constant<int, 1>{}, std::false_type{}, std::false_type{});
}
}  // namespace MARLIN_NAMESPACE_NAME
#ifdef MARLINW_TMA_FP8
extern "C" int swiglu_fp8_prepare() {
  using namespace MARLIN_NAMESPACE_NAME;
  auto status = find_encode();
  if (status != cudaSuccess) return status;
  status = set_gateup_attrs();
  if (status != cudaSuccess) return status;
  swiglu_initialize<<<256, 256>>>();
  int differences = 0;
  status = cudaMemcpyToSymbol(silu_differences, &differences, sizeof(int));
  if (status != cudaSuccess) return status;
  silu_check<<<256, 256>>>();
  status = cudaMemcpyFromSymbol(&differences, silu_differences, sizeof(int));
  if (status != cudaSuccess) return status;
  if (differences) return cudaErrorUnknown;
  return cudaDeviceSynchronize();
}
extern "C" int swiglu_fp8_gemm(const void* x, const void* w, const void* scales, const void* global,
                               const float* down_input_scale, void* out, int stride, cudaStream_t stream) {
  using namespace MARLIN_NAMESPACE_NAME;
  constexpr int m = 512, n = kFused, k = 6656;
  if (!encode || stride < k || stride % 8) return cudaErrorInvalidValue;
  if (((uintptr_t)x | (uintptr_t)w | (uintptr_t)scales) & 15) return cudaErrorInvalidValue;
  CUtensorMap tmA, tmB, tmS;
  if (!map2d(&tmA, x, 1, 2 * k, m, 2 * (uint64_t)stride, 128, 128, true) ||
      !map2d(&tmB, w, 4, 2 * n, k / 16, 8 * n, 128, 4, false) ||
      !map2d(&tmS, scales, 1, n, k / 16, n, 64, 4, false))
    return cudaErrorInvalidValue;
  return dispatch_gateup(tma_pdl_mode(), gateup_tiles_per_cta(), gateup_cs_out(), gateup_b_ef(), stream,
                         tmA, tmB, tmS, static_cast<char*>(out), static_cast<const float*>(global),
                         reinterpret_cast<const uint8_t*>(w), down_input_scale);
}
extern "C" int swiglu_fp8_gemm_ilv(const void* x, const void* w_ilv, const void* scales_ilv, const void* global,
                                   const float* down_input_scale, void* out, int stride, cudaStream_t stream) {
  using namespace MARLIN_NAMESPACE_NAME;
  constexpr int m = 512, n = kFused, k = 6656;
  if (!encode || stride < k || stride % 8) return cudaErrorInvalidValue;
  if (((uintptr_t)x | (uintptr_t)w_ilv | (uintptr_t)scales_ilv) & 15) return cudaErrorInvalidValue;
  CUtensorMap tmA, tmB, tmS;
  if (!map2d(&tmA, x, 1, 2 * k, m, 2 * (uint64_t)stride, 128, 128, true) ||
      !map2d(&tmB, w_ilv, 4, 2 * n, k / 16, 8 * n, 256, 4, false) ||
      !map2d(&tmS, scales_ilv, 1, n, k / 16, n, 128, 4, false))
    return cudaErrorInvalidValue;
  return dispatch_gateup_ilv(gateup_ilv_pdl_mode(), gateup_ilv_tiles_per_cta(), gateup_cs_out(), gateup_ilv_b_ef(),
                             gateup_ilv_epi8w(), gateup_ilv_lut_silu(), stream, tmA, tmB, tmS, static_cast<char*>(out),
                             static_cast<const float*>(global), reinterpret_cast<const uint8_t*>(w_ilv),
                             down_input_scale);
}
#else
// Same C ABI as libmarlinw_swiglu2.so.
extern "C" int swiglu_prepare() {
  using namespace MARLIN_NAMESPACE_NAME;
  auto status = find_encode();
  if (status != cudaSuccess) return status;
  status = set_gateup_attrs();
  if (status != cudaSuccess) return status;
  swiglu_initialize<<<256, 256>>>();
  int differences = 0;
  status = cudaMemcpyToSymbol(silu_differences, &differences, sizeof(int));
  if (status != cudaSuccess) return status;
  silu_check<<<256, 256>>>();
  status = cudaMemcpyFromSymbol(&differences, silu_differences, sizeof(int));
  if (status != cudaSuccess) return status;
  if (differences) return cudaErrorUnknown;
  return cudaDeviceSynchronize();
}
extern "C" int swiglu_gemm(const void* x, const void* w, const void* scales, const void* global, void* out,
                           int stride, cudaStream_t stream) {
  using namespace MARLIN_NAMESPACE_NAME;
  constexpr int m = 512, n = kFused, k = 6656;
  if (!encode || stride < k || stride % 8) return cudaErrorInvalidValue;
  if (((uintptr_t)x | (uintptr_t)w | (uintptr_t)scales) & 15) return cudaErrorInvalidValue;
  CUtensorMap tmA, tmB, tmS;
  if (!map2d(&tmA, x, 1, 2 * k, m, 2 * (uint64_t)stride, 128, 128, true) ||
      !map2d(&tmB, w, 4, 2 * n, k / 16, 8 * n, 128, 4, false) ||
      !map2d(&tmS, scales, 1, n, k / 16, n, 64, 4, false))
    return cudaErrorInvalidValue;
  return dispatch_gateup(tma_pdl_mode(), gateup_tiles_per_cta(), gateup_cs_out(), gateup_b_ef(), stream,
                         tmA, tmB, tmS, static_cast<char*>(out), static_cast<const float*>(global),
                         reinterpret_cast<const uint8_t*>(w));
}
extern "C" int swiglu_gemm_ilv(const void* x, const void* w_ilv, const void* scales_ilv, const void* global, void* out,
                               int stride, cudaStream_t stream) {
  using namespace MARLIN_NAMESPACE_NAME;
  constexpr int m = 512, n = kFused, k = 6656;
  if (!encode || stride < k || stride % 8) return cudaErrorInvalidValue;
  if (((uintptr_t)x | (uintptr_t)w_ilv | (uintptr_t)scales_ilv) & 15) return cudaErrorInvalidValue;
  CUtensorMap tmA, tmB, tmS;
  if (!map2d(&tmA, x, 1, 2 * k, m, 2 * (uint64_t)stride, 128, 128, true) ||
      !map2d(&tmB, w_ilv, 4, 2 * n, k / 16, 8 * n, 256, 4, false) ||
      !map2d(&tmS, scales_ilv, 1, n, k / 16, n, 128, 4, false))
    return cudaErrorInvalidValue;
  return dispatch_gateup_ilv(gateup_ilv_pdl_mode(), gateup_ilv_tiles_per_cta(), gateup_cs_out(), gateup_ilv_b_ef(),
                             gateup_ilv_epi8w(), gateup_ilv_lut_silu(), stream, tmA, tmB, tmS, static_cast<char*>(out),
                             static_cast<const float*>(global), reinterpret_cast<const uint8_t*>(w_ilv));
}
#endif
#endif
