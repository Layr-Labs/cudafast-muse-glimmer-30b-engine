// SPDX-License-Identifier: Apache-2.0
// 4-stage TMA super-block rANS block-scale M = 1 NVFP4 decode GEMV.
// Uses hardware TMA cp.async.bulk with 2 alternating super-block slots (4 k-stages per super-block),
// wave/stage-interleaved var super-blocks, and continuous cross-slice cp.async pipelines for
// gate_up (336 CTAs x 256T), down/qkv/o_proj (48 CTAs x 256T), GateV48 (24 CTAs x 256T), and lm_head (48 CTAs x 128T).
#define MARLIN_NAMESPACE_NAME muse_marlin6_tma4
#define MUSE_SCALE_TMA4
#define MUSE_B_EVICT_FIRST
#include <cstdlib>
#include "marlin/marlin_template.h"

extern "C" int muse_marlin6_get_gate_blocks();

namespace {

static size_t decode6_tpf_stride() {
  static size_t stride = []() -> size_t {
    const char* env = std::getenv("MUSE_DECODE6_TLB_PREFETCH");
    if (!env || !*env) return 0u;
    long kb = std::strtol(env, nullptr, 10);
    return kb > 0 ? (static_cast<size_t>(kb) << 10) : 0u;
  }();
  return stride;
}

template <int Threads>
__global__ void tma4_pass1_kernel(
    const uint8_t* __restrict__ sym_tiled, int k_tiles,
    const int* __restrict__ stage_col_k, const int* __restrict__ blk_stage_off,
    const uint32_t* __restrict__ blk_sb_off, const uint32_t* __restrict__ sb_gsb_map,
    const uint32_t* __restrict__ enc_f, const uint32_t* __restrict__ enc_c,
    uint32_t* __restrict__ sb_wc) {
  constexpr int kWarps = Threads / 32;
  constexpr int kStageBytes = Threads * 4;
  __shared__ uint32_t sf[64], sc[64];
  if (threadIdx.x < 64) {
    sf[threadIdx.x] = enc_f[threadIdx.x];
    sc[threadIdx.x] = enc_c[threadIdx.x];
  }
  __syncthreads();

  const int b = blockIdx.x;
  const int s0 = blk_stage_off[b];
  const int s1 = blk_stage_off[b + 1];
  const int num_sbs = static_cast<int>(blk_sb_off[b + 1] - blk_sb_off[b]);
  const uint32_t sb_base = blk_sb_off[b];
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  uint32_t x = 65536u;

  for (int sb = num_sbs - 1; sb >= 0; --sb) {
    const int st_lo = s0 + sb * 4;
    const int st_hi = (st_lo + 4 < s1) ? (st_lo + 4) : s1;
    uint32_t wtot = 0;
    for (int s = st_hi - 1; s >= st_lo; --s) {
      const int col = stage_col_k[2 * s];
      const int kt = stage_col_k[2 * s + 1];
      const uint8_t* raw = sym_tiled + (static_cast<size_t>(col) * k_tiles + kt) * kStageBytes + 128 * w + lane;
#pragma unroll
      for (int m = 3; m >= 0; --m) {
        const uint32_t sy = raw[32 * m];
        const uint32_t fs = sf[sy], cs = sc[sy];
        const bool emit = static_cast<uint64_t>(x) >= (static_cast<uint64_t>(fs) << 22);
        wtot += __popc(__ballot_sync(0xffffffffu, emit));
        if (emit) x >>= 16;
        x = ((x / fs) << 10) + (x % fs) + cs;
      }
    }
    if (lane == 0) sb_wc[sb_gsb_map[sb_base + sb] * kWarps + w] = wtot;
  }
}

template <int Threads>
__global__ void tma4_pass2_kernel(
    const uint8_t* __restrict__ sym_tiled, int k_tiles,
    const int* __restrict__ stage_col_k, const int* __restrict__ blk_stage_off,
    const uint32_t* __restrict__ blk_sb_off, const uint32_t* __restrict__ sb_gsb_map,
    const uint32_t* __restrict__ enc_f, const uint32_t* __restrict__ enc_c,
    const uint32_t* __restrict__ sb_wc, const uint32_t* __restrict__ off16,
    uint32_t* __restrict__ states_out, unsigned char* __restrict__ var_out) {
  constexpr int kWarps = Threads / 32;
  constexpr int kStageBytes = Threads * 4;
  __shared__ uint32_t sf[64], sc[64];
  if (threadIdx.x < 64) {
    sf[threadIdx.x] = enc_f[threadIdx.x];
    sc[threadIdx.x] = enc_c[threadIdx.x];
  }
  __syncthreads();

  const int b = blockIdx.x;
  const int s0 = blk_stage_off[b];
  const int s1 = blk_stage_off[b + 1];
  const int num_sbs = static_cast<int>(blk_sb_off[b + 1] - blk_sb_off[b]);
  const uint32_t sb_base = blk_sb_off[b];
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t lt = (1u << lane) - 1u;
  uint32_t x = 65536u;

  for (int sb = num_sbs - 1; sb >= 0; --sb) {
    const uint32_t gsb = sb_gsb_map[sb_base + sb];
    uint16_t* sb_ptr = reinterpret_cast<uint16_t*>(var_out + static_cast<size_t>(off16[gsb]) * 16);
    const uint32_t my_wc = sb_wc[gsb * kWarps + w];
    if (lane == 0) sb_ptr[w] = static_cast<uint16_t>(my_wc);
    uint32_t wpos = kWarps;
#pragma unroll
    for (int i = 0; i < kWarps - 1; ++i) {
      if (i < w) wpos += sb_wc[gsb * kWarps + i];
    }
    wpos += my_wc;

    const int st_lo = s0 + sb * 4;
    const int st_hi = (st_lo + 4 < s1) ? (st_lo + 4) : s1;
    for (int s = st_hi - 1; s >= st_lo; --s) {
      const int col = stage_col_k[2 * s];
      const int kt = stage_col_k[2 * s + 1];
      const uint8_t* raw = sym_tiled + (static_cast<size_t>(col) * k_tiles + kt) * kStageBytes + 128 * w + lane;
#pragma unroll
      for (int m = 3; m >= 0; --m) {
        const uint32_t sy = raw[32 * m];
        const uint32_t fs = sf[sy], cs = sc[sy];
        const bool emit = static_cast<uint64_t>(x) >= (static_cast<uint64_t>(fs) << 22);
        const uint32_t msk = __ballot_sync(0xffffffffu, emit);
        wpos -= __popc(msk);
        if (emit) {
          sb_ptr[wpos + __popc(msk & lt)] = static_cast<uint16_t>(x & 0xFFFFu);
          x >>= 16;
        }
        x = ((x / fs) << 10) + (x % fs) + cs;
      }
    }
  }
  states_out[static_cast<size_t>(b) * Threads + threadIdx.x] = x;
}

static int decode6_tma4_stages(const char* env_name, int def_val) {
  const char* env = std::getenv(env_name);
  if (!env || !*env) return def_val;
  long v = std::strtol(env, nullptr, 10);
  return (v == 3 || v == 4 || v == 8) ? static_cast<int>(v) : def_val;
}

static bool decode6_tma4_ovlp() {
  static const bool ovlp = []() {
    const char* env = std::getenv("MUSE_TMA4_OVLP");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  return ovlp;
}

static bool decode6_tma4_pal16() {
  static const bool pal16 = []() {
    const char* env = std::getenv("MUSE_TMA4_PAL16");
    return !env || !*env || std::strtol(env, nullptr, 10) != 0;
  }();
  return pal16;
}

static bool decode6_tma4_pal16_lm(bool is_pad128) {
  if (is_pad128) {
    static const bool pal16_lm128 = []() {
      const char* env = std::getenv("MUSE_TMA4_PAL16_LM");
      return !env || !*env || std::strtol(env, nullptr, 10) != 0;
    }();
    return pal16_lm128;
  }
  static const bool pal16_lm = []() {
    const char* env = std::getenv("MUSE_TMA4_PAL16_LM");
    return env && *env && std::strtol(env, nullptr, 10) != 0;
  }();
  return pal16_lm;
}

static int decode6_lm128_stages(int blocks) {
  static const int env_st = []() -> int {
    const char* env = std::getenv("MUSE_LM128_STAGES");
    if (env && *env) {
      long v = std::strtol(env, nullptr, 10);
      if (v == 3 || v == 4) return static_cast<int>(v);
    }
    return 3;
  }();
  return (blocks == 96) ? 3 : env_st;
}

static int g_tma4_pdl_override = -1;

static int decode6_tma4_pdl() {
  if (g_tma4_pdl_override >= 0) return g_tma4_pdl_override;
  static const int mode = []() -> int {
    const char* env = std::getenv("MUSE_TMA4_PDL");
    if (!env || !*env) return 2;  // default ON: full dual-sided SM121 PDL (all edges; 0=off, 1=role-smart)
    long v = std::strtol(env, nullptr, 10);
    return (v == 0 || v == 1 || v == 2) ? static_cast<int>(v) : 2;
  }();
  return mode;
}

template <int TileN, int Threads, bool GateUpGrid, bool DownIlv, bool GateV48,
          bool GateUpSwiglu, bool FuseLmPost, bool FuseMulSigmoid,
          int Stages, bool Split2, int FixedShape,
          bool GateUpSwigluFp8, bool TmaOvlp, bool HalfSplit, bool FastPal16,
          bool MultiM = false>
int launch_tma4_impl(const int4* input, const int4* weight, int4* output, int4* partials,
                     const int4* coded, const int4* palette, const float* global_scale,
                     int* locks, int n, int k, int grid, int device, cudaStream_t stream,
                     int* pair_locks, const uint16_t* lmpost_lut,
                     int64_t* lmpost_out, int org_vocab_size,
                     const int4* input_aux, const float* fp8_scale,
                     uint8_t* fp8_out, int prob_m = 1) {
  auto kernel = muse_marlin6_tma4::Marlin<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), Threads, 1, TileN / 16, 8, true, Stages, 1, false, true,
      GateUpGrid, DownIlv, GateV48, GateUpSwiglu, FuseLmPost, FuseMulSigmoid, Split2,
      FixedShape, GateUpSwigluFp8, TmaOvlp, HalfSplit, FastPal16, MultiM>;
  static int cached_shared[16] = {0};
  int shared = (device >= 0 && device < 16) ? cached_shared[device] : 0;
  if (shared == 0) {
    if constexpr (MultiM) {
      auto status = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
      if (status != cudaSuccess) return status;
      status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
      if (status != cudaSuccess) return status;
    } else if constexpr (HalfSplit) {
      shared = FuseMulSigmoid ? (43 * 1024) : (34 * 1024);
      auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
      if (status != cudaSuccess) return status;
    } else if constexpr (Stages == 3 && TileN == 128 && Threads == 256) {
      shared = (FixedShape == 3) ? (FastPal16 ? (49 * 1024) : (45 * 1024))
                                 : (FastPal16 ? (48 * 1024) : (45 * 1024));
      auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
      if (status != cudaSuccess) return status;
    } else if constexpr (Stages == 4 && TileN == 128 && Threads == 256 && DownIlv && !FuseMulSigmoid && (FastPal16 || TmaOvlp)) {
      shared = (FixedShape == 3) ? (FastPal16 ? (58 * 1024) : (55 * 1024))
                                 : (FastPal16 ? (57 * 1024) : 53504);
      auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
      if (status != cudaSuccess) return status;
    } else if constexpr (Split2 && TileN == 64 && Threads == 128) {
      shared = 42 * 1024;
      auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
      if (status != cudaSuccess) return status;
    } else {
      auto status = cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
      if (status != cudaSuccess) return status;
      status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
      if (status != cudaSuccess) return status;
    }
    if (device >= 0 && device < 16) cached_shared[device] = shared;
  }
  size_t total_sbs = 0;
  size_t max_sbs = 0;
  if constexpr (GateUpGrid) {
    total_sbs = 288 * 13 + 48 * 7;  // 4080
    max_sbs = 0;
  } else {
    struct Entry { int n, k, grid; size_t total_sbs, max_sbs; };
    static Entry cache[8] = {
        {-1, -1, -1, 0, 0}, {-1, -1, -1, 0, 0}, {-1, -1, -1, 0, 0}, {-1, -1, -1, 0, 0},
        {-1, -1, -1, 0, 0}, {-1, -1, -1, 0, 0}, {-1, -1, -1, 0, 0}, {-1, -1, -1, 0, 0},
    };
    static int next_slot = 0;
    bool hit = false;
    for (int i = 0; i < 8; ++i) {
      if (cache[i].n == n && cache[i].k == k && cache[i].grid == grid) {
        total_sbs = cache[i].total_sbs;
        max_sbs = cache[i].max_sbs;
        hit = true;
        break;
      }
    }
    if (!hit) {
      if constexpr (FixedShape == 3) {
        const int n_tiles = 1579;
        const int k_tiles = 52;
        const int full_p = n_tiles / grid;
        const int rem_c = n_tiles - full_p * grid;
        for (int b = 0; b < grid; ++b) {
          const int it = (full_p + (b < rem_c ? 1 : 0)) * k_tiles;
          const size_t sbs = static_cast<size_t>((it + 3) / 4);
          total_sbs += sbs;
          if (sbs > max_sbs) max_sbs = sbs;
        }
      } else {
        const int vgrid = (GateV48 || Split2 || HalfSplit) ? 48 : grid;
        const int n_tiles = HalfSplit ? (n / 128) : (n / TileN);
        const int k_tiles = k / 128;
        int part2_mn_tiles = n_tiles;
        int part1_mn_iters = 0;
        if (n_tiles > vgrid) {
          part2_mn_tiles = n_tiles % vgrid;
          if (part2_mn_tiles * 3 <= vgrid) part2_mn_tiles += vgrid;
          part1_mn_iters = (n_tiles - part2_mn_tiles) / vgrid;
        }
        const int p2_iters = (k_tiles * part2_mn_tiles + vgrid - 1) / vgrid;
        const int total_p2 = k_tiles * part2_mn_tiles;
        const int lo_p1 = (part1_mn_iters + 1) / 2;
        const int hi_p1 = part1_mn_iters - lo_p1;
        for (int b = 0; b < grid; ++b) {
          int it = 0;
          if constexpr (GateV48) {
            const int v1 = 47 - b;
            const int rem1 = total_p2 - p2_iters * v1;
            const int p2_1 = rem1 > 0 ? (rem1 < p2_iters ? rem1 : p2_iters) : 0;
            int p2_2 = 0;
            if (v1 >= grid) {
              const int v2 = v1 - grid;
              const int rem2 = total_p2 - p2_iters * v2;
              p2_2 = rem2 > 0 ? (rem2 < p2_iters ? rem2 : p2_iters) : 0;
            }
            it = p2_1 + p2_2;
          } else if constexpr (Split2) {
            if (b < 48) {
              it = lo_p1 * k_tiles;
            } else {
              const int vb = b - 48;
              const int rem = total_p2 - p2_iters * vb;
              const int p2 = rem > 0 ? (rem < p2_iters ? rem : p2_iters) : 0;
              it = hi_p1 * k_tiles + p2;
            }
          } else if constexpr (HalfSplit) {
            if (grid == 96) {
              const int vb = b >> 1;
              const int rem = total_p2 - p2_iters * vb;
              const int p2 = rem > 0 ? (rem < p2_iters ? rem : p2_iters) : 0;
              it = part1_mn_iters * k_tiles + p2;
            } else {
              const int rem = total_p2 - p2_iters * b;
              const int p2 = rem > 0 ? (rem < p2_iters ? rem : p2_iters) : 0;
              it = 2 * (part1_mn_iters * k_tiles + p2);
            }
          } else {
            const int rem = total_p2 - p2_iters * b;
            const int p2 = rem > 0 ? (rem < p2_iters ? rem : p2_iters) : 0;
            it = part1_mn_iters * k_tiles + p2;
          }
          const size_t sbs = static_cast<size_t>((it + 3) / 4);
          total_sbs += sbs;
          if (sbs > max_sbs) max_sbs = sbs;
        }
      }
      cache[next_slot] = {n, k, grid, total_sbs, max_sbs};
      next_slot = (next_slot + 1) & 7;
    }
  }
  const size_t meta_bytes = ((static_cast<size_t>(4 + grid) + max_sbs + 3) / 4) * 16;
  const size_t off16_bytes = ((total_sbs + 1) * 4 + 15) / 16 * 16;
  const size_t states_off = 4096 + meta_bytes + off16_bytes;
  const size_t var_off = states_off + static_cast<size_t>(grid) * Threads * 4;
  const auto* base = reinterpret_cast<const unsigned char*>(coded);
  const auto* lut_and_meta = reinterpret_cast<const int*>(base);
  const auto* states = reinterpret_cast<const int4*>(base + states_off);
  const auto* var = reinterpret_cast<const int4*>(base + var_off);
  const float* a_sc_arg = GateUpSwigluFp8 ? fp8_scale : nullptr;
  int64_t* lm_out_arg = GateUpSwigluFp8 ? reinterpret_cast<int64_t*>(fp8_out) : lmpost_out;
  const int pdl_cfg = decode6_tma4_pdl();
  int pdl_mode = 0;
  if (pdl_cfg == 1) {
    // Role-smart SM121 PDL: o_proj / down_proj trigger the following 1-CTA norm during their tail;
    // gate_up / qkv consume the preceding 1-CTA norm via PDL (preload LUT/states/sb0/B before wait).
    if constexpr (FuseMulSigmoid) {
      pdl_mode = 2;
    } else if (n == 6656 && k == 19968) {
      pdl_mode = 2;
    } else if constexpr (DownIlv && !FuseLmPost) {
      if ((n == 39936 && k == 6656) || (n == 4608 && k == 6656)) {
        pdl_mode = 1;
      }
    }
  } else if (pdl_cfg == 2) {
    if constexpr (FuseMulSigmoid) {
      pdl_mode = 2;
    } else if constexpr (DownIlv && !FuseLmPost) {
      pdl_mode = 3;
    } else if constexpr (DownIlv && FuseLmPost) {
      static const int lm_pdl = []() -> int {
        const char* env = std::getenv("MUSE_LM_HEAD_PDL");
        if (!env || !*env) return 3;
        long v = std::strtol(env, nullptr, 10);
        return (v >= 0 && v <= 3) ? static_cast<int>(v) : 3;
      }();
      pdl_mode = lm_pdl;
    }
  }
  if ((pdl_mode & 1) != 0) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid, 1, 1);
    cfg.blockDim = dim3(Threads, 1, 1);
    cfg.dynamicSmemBytes = static_cast<size_t>(shared);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return static_cast<int>(cudaLaunchKernelEx(
        &cfg, kernel,
        input, weight, output, partials, static_cast<const int4*>(nullptr), a_sc_arg, states, global_scale,
        var, lut_and_meta, k / 16, prob_m, n, k, k, locks, false, false, true, shared, palette,
        decode6_tpf_stride(), pair_locks, lmpost_lut, lm_out_arg, org_vocab_size, input_aux, pdl_mode));
  }
  kernel<<<grid, Threads, shared, stream>>>(
      input, weight, output, partials, nullptr, a_sc_arg, states, global_scale,
      var, lut_and_meta, k / 16, prob_m, n, k, k, locks, false, false, true, shared, palette,
      decode6_tpf_stride(), pair_locks, lmpost_lut, lm_out_arg, org_vocab_size, input_aux, pdl_mode);
  return cudaGetLastError();
}

template <int TileN, int Threads, bool GateUpGrid, bool DownIlv = false, bool GateV48 = false,
          bool GateUpSwiglu = false, bool FuseLmPost = false, bool FuseMulSigmoid = false,
          int Stages = 4, bool Split2 = false, int FixedShape = 0,
          bool GateUpSwigluFp8 = false, bool HalfSplit = false>
int launch_tma4(const int4* input, const int4* weight, int4* output, int4* partials,
                const int4* coded, const int4* palette, const float* global_scale,
                int* locks, int n, int k, int grid, int device, cudaStream_t stream,
                int* pair_locks = nullptr, const uint16_t* lmpost_lut = nullptr,
                int64_t* lmpost_out = nullptr, int org_vocab_size = 0,
                const int4* input_aux = nullptr, const float* fp8_scale = nullptr,
                uint8_t* fp8_out = nullptr) {
  const bool ovlp = decode6_tma4_ovlp();
  const bool pal16 = !GateUpGrid && (FuseLmPost ? decode6_tma4_pal16_lm(FixedShape == 3) : decode6_tma4_pal16());
  if (ovlp) {
    if constexpr (!GateUpGrid) {
      if (pal16) {
        return launch_tma4_impl<TileN, Threads, GateUpGrid, DownIlv, GateV48, GateUpSwiglu,
                                FuseLmPost, FuseMulSigmoid, Stages, Split2, FixedShape,
                                GateUpSwigluFp8, true, HalfSplit, true>(
            input, weight, output, partials, coded, palette, global_scale, locks,
            n, k, grid, device, stream, pair_locks, lmpost_lut, lmpost_out,
            org_vocab_size, input_aux, fp8_scale, fp8_out);
      }
    }
    return launch_tma4_impl<TileN, Threads, GateUpGrid, DownIlv, GateV48, GateUpSwiglu,
                            FuseLmPost, FuseMulSigmoid, Stages, Split2, FixedShape,
                            GateUpSwigluFp8, true, HalfSplit, false>(
        input, weight, output, partials, coded, palette, global_scale, locks,
        n, k, grid, device, stream, pair_locks, lmpost_lut, lmpost_out,
        org_vocab_size, input_aux, fp8_scale, fp8_out);
  }
  if constexpr (!GateUpGrid) {
    if (pal16) {
      return launch_tma4_impl<TileN, Threads, GateUpGrid, DownIlv, GateV48, GateUpSwiglu,
                              FuseLmPost, FuseMulSigmoid, Stages, Split2, FixedShape,
                              GateUpSwigluFp8, false, HalfSplit, true>(
          input, weight, output, partials, coded, palette, global_scale, locks,
          n, k, grid, device, stream, pair_locks, lmpost_lut, lmpost_out,
          org_vocab_size, input_aux, fp8_scale, fp8_out);
    }
  }
  return launch_tma4_impl<TileN, Threads, GateUpGrid, DownIlv, GateV48, GateUpSwiglu,
                          FuseLmPost, FuseMulSigmoid, Stages, Split2, FixedShape,
                          GateUpSwigluFp8, false, HalfSplit, false>(
      input, weight, output, partials, coded, palette, global_scale, locks,
      n, k, grid, device, stream, pair_locks, lmpost_lut, lmpost_out,
      org_vocab_size, input_aux, fp8_scale, fp8_out);
}

}  // namespace

extern "C" int muse_marlin6_set_pdl(int mode) {
  const int prev = decode6_tma4_pdl();
  g_tma4_pdl_override = (mode == 0 || mode == 1 || mode == 2) ? mode : -1;
  return prev;
}

extern "C" int muse_marlin6_gate_up_tma4(const int4* input, const int4* weight, int4* output,
                                         int4* partials, const int4* coded, const int4* palette,
                                         const float* global_scale, int* locks, int n, int k, int blocks,
                                         int device, cudaStream_t stream) {
  if (n != 39936 || k != 6656 || blocks != 48) return cudaErrorInvalidValue;
  return launch_tma4<128, 256, true>(input, weight, output, partials, coded, palette,
                                     global_scale, locks, n, k, 336, device, stream);
}

extern "C" int muse_marlin6_gate_up_tma4_swiglu(
    const int4* input, const int4* weight, int4* output, int4* partials,
    const int4* coded, const int4* palette, const float* global_scale,
    int* locks, int* pair_locks, int n, int k, int blocks, int device, cudaStream_t stream) {
  if (n != 39936 || k != 6656 || blocks != 48)
    return cudaErrorInvalidValue;
  return launch_tma4<128, 256, true, false, false, true, false>(
      input, weight, output, partials, coded, palette, global_scale, locks,
      n, k, 336, device, stream, pair_locks);
}

extern "C" int muse_marlin6_gate_up_tma4_swiglu_fp8(
    const int4* input, const int4* weight, int4* scratch_out, int4* partials,
    const int4* coded, const int4* palette, const float* global_scale,
    int* locks, const float* fp8_scale, uint8_t* fp8_out,
    int n, int k, int blocks, int device, cudaStream_t stream) {
  if (n != 39936 || k != 6656 || blocks != 48 || fp8_scale == nullptr || fp8_out == nullptr)
    return cudaErrorInvalidValue;
  return launch_tma4<128, 256, true, false, false, true, false, false, 4, false, 0, true>(
      input, weight, scratch_out, partials, coded, palette, global_scale, locks,
      n, k, 336, device, stream, nullptr, nullptr, nullptr, 0, nullptr, fp8_scale, fp8_out);
}

extern "C" int muse_marlin6_sk_tma4_ilv_swiglu(
    const int4* input, const int4* weight, int4* output, int4* partials,
    const int4* coded, const int4* palette, const float* global_scale,
    int* locks, int n, int k, int blocks, int device, cudaStream_t stream) {
  if (n != 39936 || k != 6656 || blocks != 48)
    return cudaErrorInvalidValue;
  static const int gu_st = decode6_tma4_stages("MUSE_TMA4_STAGES", 4);
  if (gu_st == 3)
    return launch_tma4<128, 256, false, true, false, true, false, false, 3>(
        input, weight, output, partials, coded, palette, global_scale, locks,
        n, k, 48, device, stream);
  if (gu_st == 8)
    return launch_tma4<128, 256, false, true, false, true, false, false, 8>(
        input, weight, output, partials, coded, palette, global_scale, locks,
        n, k, 48, device, stream);
  return launch_tma4<128, 256, false, true, false, true, false, false, 4>(
      input, weight, output, partials, coded, palette, global_scale, locks,
      n, k, 48, device, stream);
}

extern "C" int muse_marlin6_sk_tma4_ilv_swiglu_m(
    const int4* input, const int4* weight, int4* output, int4* partials,
    const int4* coded, const int4* palette, const float* global_scale,
    int* locks, int m, int n, int k, int blocks, int device, cudaStream_t stream) {
  if (m < 1 || m > 8 || n != 39936 || k != 6656 || blocks != 48)
    return cudaErrorInvalidValue;
  static const int gu_st = decode6_tma4_stages("MUSE_TMA4_SWIGLU_M_STAGES", 3);  // S3 NVFP4TUNE: 3-stage multi-M gate_up
  if (gu_st == 3)
    return launch_tma4_impl<128, 256, false, true, false, true, false, false, 3, false, 0, false, true, false, true, true>(
        input, weight, output, partials, coded, palette, global_scale, locks,
        n, k, 48, device, stream, nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, m);
  return launch_tma4_impl<128, 256, false, true, false, true, false, false, 4, false, 0, false, true, false, true, true>(
      input, weight, output, partials, coded, palette, global_scale, locks,
      n, k, 48, device, stream, nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, m);
}

extern "C" int muse_marlin6_sk_tma4_ilv_swiglu_fp8(
    const int4* input, const int4* weight, int4* scratch_out, int4* partials,
    const int4* coded, const int4* palette, const float* global_scale,
    int* locks, const float* fp8_scale, uint8_t* fp8_out,
    int n, int k, int blocks, int device, cudaStream_t stream) {
  if (n != 39936 || k != 6656 || blocks != 48 || fp8_scale == nullptr || fp8_out == nullptr)
    return cudaErrorInvalidValue;
  static const int gu_st = decode6_tma4_stages("MUSE_TMA4_STAGES", 4);
  if (gu_st == 3)
    return launch_tma4<128, 256, false, true, false, true, false, false, 3, false, 0, true>(
        input, weight, scratch_out, partials, coded, palette, global_scale, locks,
        n, k, 48, device, stream, nullptr, nullptr, nullptr, 0, nullptr, fp8_scale, fp8_out);
  if (gu_st == 8)
    return launch_tma4<128, 256, false, true, false, true, false, false, 8, false, 0, true>(
        input, weight, scratch_out, partials, coded, palette, global_scale, locks,
        n, k, 48, device, stream, nullptr, nullptr, nullptr, 0, nullptr, fp8_scale, fp8_out);
  return launch_tma4<128, 256, false, true, false, true, false, false, 4, false, 0, true>(
      input, weight, scratch_out, partials, coded, palette, global_scale, locks,
      n, k, 48, device, stream, nullptr, nullptr, nullptr, 0, nullptr, fp8_scale, fp8_out);
}

extern "C" int muse_marlin6_sk_tma4_lmpost(
    const int4* input, const int4* weight, int4* scratch, int4* partials,
    const int4* coded, const int4* palette, const float* global_scale, int* locks,
    const uint16_t* lmpost_lut, int64_t* lmpost_out, int org_vocab_size,
    int n, int k, int blocks, int device, cudaStream_t stream) {
  if (k % 128 != 0 || n % 128 == 0 || n % 64 != 0 || blocks != 48 ||
      org_vocab_size <= 0 || org_vocab_size > n ||
      lmpost_lut == nullptr || lmpost_out == nullptr) {
    return cudaErrorInvalidValue;
  }
  return launch_tma4<64, 128, false, false, false, false, true>(
      input, weight, scratch, partials, coded, palette, global_scale, locks,
      n, k, 48, device, stream, nullptr, lmpost_lut, lmpost_out, org_vocab_size);
}

extern "C" int muse_marlin6_sk_tma4_ilv_lmpost(
    const int4* input, const int4* weight, int4* scratch, int4* partials,
    const int4* coded, const int4* palette, const float* global_scale, int* locks,
    const uint16_t* lmpost_lut, int64_t* lmpost_out, int org_vocab_size,
    int n, int k, int blocks, int device, cudaStream_t stream) {
  if ((n != 202048 && n != 202112) || k != 6656 || (blocks != 48 && blocks != 96) ||
      org_vocab_size <= 0 || org_vocab_size > n ||
      lmpost_lut == nullptr || lmpost_out == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (n == 202112) {
    const int st = decode6_lm128_stages(blocks);
    if (st == 3) {
      return launch_tma4<128, 256, false, true, false, false, true, false, 3, false, 3>(
          input, weight, scratch, partials, coded, palette, global_scale, locks,
          n, k, blocks, device, stream, nullptr, lmpost_lut, lmpost_out, org_vocab_size);
    }
    return launch_tma4<128, 256, false, true, false, false, true, false, 4, false, 3>(
        input, weight, scratch, partials, coded, palette, global_scale, locks,
        n, k, blocks, device, stream, nullptr, lmpost_lut, lmpost_out, org_vocab_size);
  }
  if (blocks == 96) {
    return launch_tma4<64, 128, false, true, false, false, true, false, 4, true, 2>(
        input, weight, scratch, partials, coded, palette, global_scale, locks,
        n, k, 96, device, stream, nullptr, lmpost_lut, lmpost_out, org_vocab_size);
  }
  return launch_tma4<64, 128, false, true, false, false, true, false, 4, false, 2>(
      input, weight, scratch, partials, coded, palette, global_scale, locks,
      n, k, 48, device, stream, nullptr, lmpost_lut, lmpost_out, org_vocab_size);
}

extern "C" int muse_marlin6_sk_tma4(const int4* input, const int4* weight, int4* output,
                                    int4* partials, const int4* coded, const int4* palette,
                                    const float* global_scale, int* locks, int n, int k, int blocks,
                                    int device, cudaStream_t stream) {
  if (k % 128 != 0 || blocks != 48) return cudaErrorInvalidValue;
  if (n == 4096 && k == 6656) {
    const int gb = muse_marlin6_get_gate_blocks();
    if (gb != 48) {
      return launch_tma4<128, 256, false, false, true>(input, weight, output, partials, coded, palette,
                                                       global_scale, locks, n, k, gb, device, stream);
    }
  }
  if (n % 128 == 0)
    return launch_tma4<128, 256, false, false, false>(input, weight, output, partials, coded, palette,
                                                      global_scale, locks, n, k, 48, device, stream);
  if (n % 64 == 0)
    return launch_tma4<64, 128, false, false, false>(input, weight, output, partials, coded, palette,
                                                     global_scale, locks, n, k, 48, device, stream);
  return cudaErrorInvalidValue;
}

extern "C" int muse_marlin6_sk_tma4_ilv(const int4* input, const int4* weight, int4* output,
                                        int4* partials, const int4* coded, const int4* palette,
                                        const float* global_scale, int* locks, int n, int k, int blocks,
                                        int device, cudaStream_t stream) {
  if (n == 202112 && k == 6656 && (blocks == 48 || blocks == 96)) {
    const int st = decode6_lm128_stages(blocks);
    if (st == 3) {
      return launch_tma4<128, 256, false, true, false, false, false, false, 3, false, 3>(
          input, weight, output, partials, coded, palette, global_scale, locks, n, k, blocks, device, stream);
    }
    return launch_tma4<128, 256, false, true, false, false, false, false, 4, false, 3>(
        input, weight, output, partials, coded, palette, global_scale, locks, n, k, blocks, device, stream);
  }
  if (n == 202048 && k == 6656 && (blocks == 48 || blocks == 96)) {
    if (blocks == 96) {
      return launch_tma4<64, 128, false, true, false, false, false, false, 4, true>(
          input, weight, output, partials, coded, palette, global_scale, locks, n, k, 96, device, stream);
    }
    return launch_tma4<64, 128, false, true, false, false, false, false, 4, false>(
        input, weight, output, partials, coded, palette, global_scale, locks, n, k, 48, device, stream);
  }
  if (k % 128 != 0 || n % 128 != 0 || blocks != 48) return cudaErrorInvalidValue;
  static const int ilv_st = decode6_tma4_stages("MUSE_TMA4_STAGES", 4);
  if (n == 4096 && k == 6656) {
    const int gb = muse_marlin6_get_gate_blocks();
    if (gb == 24) {
      if (ilv_st == 8)
        return launch_tma4<128, 256, false, true, true, false, false, false, 8>(
            input, weight, output, partials, coded, palette, global_scale, locks, n, k, 24, device, stream);
      return launch_tma4<128, 256, false, true, true, false, false, false, 4>(
          input, weight, output, partials, coded, palette, global_scale, locks, n, k, 24, device, stream);
    }
    if (gb != 48) return cudaErrorInvalidValue;
    static const bool gate_s3 = []() {
      const char* env = std::getenv("MUSE_NV6_GATE_S3");
      return !env || !*env || std::strtol(env, nullptr, 10) != 0;
    }();
    if (gate_s3) {
      return launch_tma4<128, 256, false, true, false, false, false, false, 3>(
          input, weight, output, partials, coded, palette, global_scale, locks, n, k, 48, device, stream);
    }
  } else if (!((n == 39936 && k == 6656) || (n == 6656 && k == 19968) ||
               (n == 4608 && k == 6656) || (n == 6656 && k == 4096))) {
    return cudaErrorInvalidValue;
  }
  if (n == 6656 && k == 19968) {
    static const bool spec_dn = []() {
      const char* env = std::getenv("MUSE_DOWN_ILV_SPEC");
      return !env || !*env || std::strtol(env, nullptr, 10) != 0;
    }();
    if (spec_dn) {
      if (ilv_st == 3)
        return launch_tma4<128, 256, false, true, false, false, false, false, 3, false, 1>(
            input, weight, output, partials, coded, palette, global_scale, locks, n, k, 48, device, stream);
      if (ilv_st == 8)
        return launch_tma4<128, 256, false, true, false, false, false, false, 8, false, 1>(
            input, weight, output, partials, coded, palette, global_scale, locks, n, k, 48, device, stream);
      return launch_tma4<128, 256, false, true, false, false, false, false, 4, false, 1>(
          input, weight, output, partials, coded, palette, global_scale, locks, n, k, 48, device, stream);
    }
  }
  if (ilv_st == 3)
    return launch_tma4<128, 256, false, true, false, false, false, false, 3>(
        input, weight, output, partials, coded, palette, global_scale, locks, n, k, 48, device, stream);
  if (ilv_st == 8)
    return launch_tma4<128, 256, false, true, false, false, false, false, 8>(
        input, weight, output, partials, coded, palette, global_scale, locks, n, k, 48, device, stream);
  return launch_tma4<128, 256, false, true, false, false, false, false, 4>(
      input, weight, output, partials, coded, palette, global_scale, locks, n, k, 48, device, stream);
}

extern "C" int muse_marlin6_sk_tma4_ilv_m(const int4* input, const int4* weight, int4* output,
                                          int4* partials, const int4* coded, const int4* palette,
                                          const float* global_scale, int* locks, int m, int n, int k,
                                          int blocks, int device, cudaStream_t stream) {
  if (m < 1 || m > 8) return cudaErrorInvalidValue;
  if (n == 202112 && k == 6656 && (blocks == 48 || blocks == 96)) {
    const int st = decode6_lm128_stages(blocks);
    if (st == 3) {
      return launch_tma4_impl<128, 256, false, true, false, false, false, false, 3, false, 3, false, true, false, true, true>(
          input, weight, output, partials, coded, palette, global_scale, locks, n, k, blocks, device, stream,
          nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, m);
    }
    return launch_tma4_impl<128, 256, false, true, false, false, false, false, 4, false, 3, false, true, false, true, true>(
        input, weight, output, partials, coded, palette, global_scale, locks, n, k, blocks, device, stream,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, m);
  }
  if (n == 202048 && k == 6656 && (blocks == 48 || blocks == 96)) {
    if (blocks == 96) {
      return launch_tma4_impl<64, 128, false, true, false, false, false, false, 4, true, 0, false, true, false, true, true>(
          input, weight, output, partials, coded, palette, global_scale, locks, n, k, 96, device, stream,
          nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, m);
    }
    return launch_tma4_impl<64, 128, false, true, false, false, false, false, 4, false, 0, false, true, false, true, true>(
        input, weight, output, partials, coded, palette, global_scale, locks, n, k, 48, device, stream,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, m);
  }
  if (k % 128 != 0 || n % 128 != 0 || blocks != 48) return cudaErrorInvalidValue;
  if (n == 4096 && k == 6656) {
    const int gb = muse_marlin6_get_gate_blocks();
    if (gb == 24) {
      return launch_tma4_impl<128, 256, false, true, true, false, false, false, 4, false, 0, false, true, false, true, true>(
          input, weight, output, partials, coded, palette, global_scale, locks, n, k, 24, device, stream,
          nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, m);
    }
    return launch_tma4_impl<128, 256, false, true, false, false, false, false, 3, false, 0, false, true, false, true, true>(
        input, weight, output, partials, coded, palette, global_scale, locks, n, k, 48, device, stream,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, m);
  }
  if (n == 6656 && k == 19968) {
    return launch_tma4_impl<128, 256, false, true, false, false, false, false, 3, false, 1, false, true, false, true, true>(
        input, weight, output, partials, coded, palette, global_scale, locks, n, k, 48, device, stream,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, m);
  }
  if (n == 39936 && k == 6656) {
    return launch_tma4_impl<128, 256, false, true, false, false, false, false, 3, false, 0, false, true, false, true, true>(
        input, weight, output, partials, coded, palette, global_scale, locks, n, k, 48, device, stream,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, m);
  }
  return launch_tma4_impl<128, 256, false, true, false, false, false, false, 4, false, 0, false, true, false, true, true>(
      input, weight, output, partials, coded, palette, global_scale, locks, n, k, 48, device, stream,
      nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, m);
}

extern "C" int muse_marlin6_sk_tma4_gated(
    const int4* gate, const int4* attn, const int4* weight, int4* output,
    int4* partials, const int4* coded, const int4* palette, const float* global_scale,
    int* locks, int n, int k, int blocks, int device, cudaStream_t stream) {
  if (n != 6656 || k != 4096 || blocks != 48 || gate == nullptr || attn == nullptr)
    return cudaErrorInvalidValue;
  return launch_tma4<128, 256, false, false, false, false, false, true>(
      gate, weight, output, partials, coded, palette, global_scale, locks,
      n, k, 48, device, stream, nullptr, nullptr, nullptr, 0, attn);
}

extern "C" int muse_marlin6_sk_tma4_ilv_gated(
    const int4* gate, const int4* attn, const int4* weight, int4* output,
    int4* partials, const int4* coded, const int4* palette, const float* global_scale,
    int* locks, int n, int k, int blocks, int device, cudaStream_t stream) {
  if (n != 6656 || k != 4096 || blocks != 48 || gate == nullptr || attn == nullptr)
    return cudaErrorInvalidValue;
  return launch_tma4<128, 256, false, true, false, false, false, true>(
      gate, weight, output, partials, coded, palette, global_scale, locks,
      n, k, 48, device, stream, nullptr, nullptr, nullptr, 0, attn);
}

extern "C" int muse_tma4_pass1(const uint8_t* sym_tiled, int k_tiles,
                               const int* stage_col_k, const int* blk_stage_off,
                               const uint32_t* blk_sb_off, const uint32_t* sb_gsb_map,
                               const uint32_t* enc_f, const uint32_t* enc_c,
                               uint32_t* sb_wc, int num_blks, int threads,
                               cudaStream_t stream) {
  if (threads == 256)
    tma4_pass1_kernel<256><<<num_blks, 256, 0, stream>>>(
        sym_tiled, k_tiles, stage_col_k, blk_stage_off, blk_sb_off, sb_gsb_map, enc_f, enc_c, sb_wc);
  else if (threads == 128)
    tma4_pass1_kernel<128><<<num_blks, 128, 0, stream>>>(
        sym_tiled, k_tiles, stage_col_k, blk_stage_off, blk_sb_off, sb_gsb_map, enc_f, enc_c, sb_wc);
  else
    return cudaErrorInvalidValue;
  return cudaError_t(cudaGetLastError());
}

extern "C" int muse_tma4_pass2(const uint8_t* sym_tiled, int k_tiles,
                               const int* stage_col_k, const int* blk_stage_off,
                               const uint32_t* blk_sb_off, const uint32_t* sb_gsb_map,
                               const uint32_t* enc_f, const uint32_t* enc_c,
                               const uint32_t* sb_wc, const uint32_t* off16,
                               uint32_t* states_out, unsigned char* var_out,
                               int num_blks, int threads, cudaStream_t stream) {
  if (threads == 256)
    tma4_pass2_kernel<256><<<num_blks, 256, 0, stream>>>(
        sym_tiled, k_tiles, stage_col_k, blk_stage_off, blk_sb_off, sb_gsb_map, enc_f, enc_c,
        sb_wc, off16, states_out, var_out);
  else if (threads == 128)
    tma4_pass2_kernel<128><<<num_blks, 128, 0, stream>>>(
        sym_tiled, k_tiles, stage_col_k, blk_stage_off, blk_sb_off, sb_gsb_map, enc_f, enc_c,
        sb_wc, off16, states_out, var_out);
  else
    return cudaErrorInvalidValue;
  return cudaError_t(cudaGetLastError());
}
