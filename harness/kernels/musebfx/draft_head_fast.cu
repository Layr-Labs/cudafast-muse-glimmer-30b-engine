#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <stdint.h>

namespace {

constexpr int kDhWarps = 4;
constexpr int kDhThreads = kDhWarps * 32;
constexpr int kDhSubTilesPerCTA = kDhWarps * 2;
constexpr int kDhRowsPerCTA = 64;
constexpr int kDhBK = 128;
constexpr int kDhStages = 4;
constexpr int kDhMaxM = 8;
constexpr int kDhMaxCandPool = 512;
constexpr int kDhSentinelCols = 32;

constexpr int kNf3SubTileBytes = 400;
constexpr int kNf3StepWBytes = kDhSubTilesPerCTA * kNf3SubTileBytes;
constexpr int kVq25SubTileBytes = 336;
constexpr int kVq25StepWBytes = kDhSubTilesPerCTA * kVq25SubTileBytes;

static bool dh_dyn_env() {
  static const bool dyn = []() {
    const char* s = std::getenv("MUSE_DHEAD_DYN_ENV");
    return s && *s && std::atoi(s) != 0;
  }();
  return dyn;
}

static int dh_pdl_mode() {
  auto parse = []() -> int {
    const char* s = std::getenv("MUSE_DRAFT_PDL");
    if (!s || !*s) return 3;
    return (std::atoi(s) != 0) ? 3 : 0;
  };
  if (dh_dyn_env()) return parse();
  static const int cached = parse();
  return cached;
}

static int dh_rescore_k_cfg() {
  auto parse = []() -> int {
    const char* s = std::getenv("MUSE_DHEAD_RESCORE_K");
    if (!s || !*s) return 64;
    const int v = std::atoi(s);
    return std::max(16, std::min(kDhMaxCandPool, v));
  };
  if (dh_dyn_env()) return parse();
  static const int cached = parse();
  return cached;
}

static uint32_t dh_perm_stride(int N) {
  const uint32_t un = static_cast<uint32_t>(N);
  uint32_t p = static_cast<uint32_t>(((static_cast<uint64_t>(un) * 2654435761ull) >> 32) | 1ull);
  if (p < 3u) p = 3u;
  while (std::gcd(p, un) != 1u) {
    p += 2u;
  }
  return p;
}

template <typename... KArgs, typename... Args>
static inline void dh_launch_pdl(
    void (*kernel)(KArgs...), dim3 grid, int block, cudaStream_t st, int pdl_mode, Args... args) {
  cudaLaunchConfig_t lcfg = {};
  lcfg.gridDim = grid;
  lcfg.blockDim = dim3(static_cast<unsigned>(block), 1u, 1u);
  lcfg.dynamicSmemBytes = 0;
  lcfg.stream = st;
  cudaLaunchAttribute attr[1];
  if ((pdl_mode & 1) != 0) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    lcfg.attrs = attr;
    lcfg.numAttrs = 1;
  }
  cudaError_t err = cudaLaunchKernelEx(&lcfg, kernel, static_cast<KArgs>(args)...);
  TORCH_CHECK(err == cudaSuccess, "draft_head_fast launch failed: ", cudaGetErrorString(err));
}

__device__ __forceinline__ float dh_rad_sign(int k) {
  uint32_t x = static_cast<uint32_t>(k) * 0x9E3779B9u + 0x7F4A7C15u;
  x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
  x = (x ^ (x >> 13)) * 0xC2B2AE35u;
  x ^= (x >> 16);
  return (x & 1u) ? -1.0f : 1.0f;
}

__device__ __forceinline__ void dh_warp_fwht64(float& v0, float& v1, int lane) {
  {
    const float s = v0 + v1;
    const float d = v0 - v1;
    v0 = s;
    v1 = d;
  }
#pragma unroll
  for (int mask = 1; mask <= 16; mask <<= 1) {
    const float o0 = __shfl_xor_sync(0xffffffffu, v0, mask);
    const float o1 = __shfl_xor_sync(0xffffffffu, v1, mask);
    const bool hi = (lane & mask) != 0;
    v0 = hi ? (o0 - v0) : (v0 + o0);
    v1 = hi ? (o1 - v1) : (v1 + o1);
  }
  v0 *= 0.125f;
  v1 *= 0.125f;
}

__device__ __forceinline__ float dh_e2m1_x2_to_f32(uint32_t nib) {
  const uint32_t mag_u = (0xC8643210u >> ((nib & 7u) << 2)) & 0xFu;
  const float mag = __uint_as_float(0x41800000u | (mag_u << 19)) - 16.0f;
  return __uint_as_float(__float_as_uint(mag) ^ ((nib & 8u) << 28));
}

__device__ __forceinline__ float dh_fp8_e4m3_to_f32(uint8_t bits) {
  __nv_fp8_e4m3 v;
  v.__x = bits;
  return static_cast<float>(v);
}

__constant__ float kDhNf3Cb[8] = {
    -2.1519927f, -1.3439093f, -0.7560053f, -0.2450942f,
    +0.2450942f, +0.7560053f, +1.3439093f, +2.1519927f,
};

__constant__ float2 kDhVq25Cb[32] = {
    {+0.336695f, +0.235905f}, {-0.221841f, +0.311697f}, {-0.410038f, -0.229322f}, {+0.157453f, -0.307426f},
    {+0.937875f, +0.342424f}, {+0.388188f, +0.849478f}, {-0.343749f, +0.863302f}, {-0.875162f, +0.305411f},
    {-1.020643f, -0.387199f}, {-0.401048f, -0.826984f}, {+0.334300f, -0.888789f}, {+0.824240f, -0.310415f},
    {+1.575516f, +0.022458f}, {+1.214165f, +0.978337f}, {+0.421903f, +1.533850f}, {-0.424192f, +1.515543f},
    {-1.186799f, +0.947133f}, {-1.643692f, +0.012782f}, {-1.203122f, -1.066539f}, {-0.311558f, -1.495596f},
    {+0.576900f, -1.490208f}, {+1.226518f, -0.875182f}, {+2.221101f, +0.807127f}, {+1.339640f, +1.921301f},
    {-0.006492f, +2.464531f}, {-1.308323f, +1.928008f}, {-2.202196f, +0.801181f}, {-2.229151f, -0.889488f},
    {-1.152257f, -2.075637f}, {+0.200840f, -2.368705f}, {+1.524886f, -1.809442f}, {+2.240060f, -0.632572f},
};

__device__ __forceinline__ uint32_t dh_quant_nf3_sym(float u) {
  if (u < 0.0f) {
    if (u < -0.999951f) return (u < -1.747951f) ? 0u : 1u;
    return (u < -0.500550f) ? 2u : 3u;
  } else {
    if (u < +0.999951f) return (u < +0.500550f) ? 4u : 5u;
    return (u < +1.747951f) ? 6u : 7u;
  }
}

__device__ __forceinline__ uint32_t dh_quant_vq25(float u0, float u1) {
  uint32_t best_i = 0u;
  float best_d = 1e30f;
#pragma unroll
  for (uint32_t i = 0; i < 32u; ++i) {
    const float2 c = kDhVq25Cb[i];
    const float dx = u0 - c.x;
    const float dy = u1 - c.y;
    const float d = __fmaf_rn(dx, dx, dy * dy);
    if (d < best_d) {
      best_d = d;
      best_i = i;
    }
  }
  return best_i;
}

__device__ __forceinline__ uint32_t dh_bf16x2_u32(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float dh_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  return v;
}

__device__ __forceinline__ uint32_t dh_f32_to_ord_u32(float f) {
  const uint32_t u = __float_as_uint(f);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) return 0u;
  const uint32_t k = (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
  return (k == 0u) ? 1u : k;
}

__device__ __forceinline__ void dh_top4_insert_u64(uint64_t (&top)[4], uint64_t cand) {
  if (cand <= top[3]) return;
  if (cand > top[0]) {
    top[3] = top[2]; top[2] = top[1]; top[1] = top[0]; top[0] = cand;
  } else if (cand > top[1]) {
    top[3] = top[2]; top[2] = top[1]; top[1] = cand;
  } else if (cand > top[2]) {
    top[3] = top[2]; top[2] = cand;
  } else {
    top[3] = cand;
  }
}

__device__ __forceinline__ uint64_t dh_warp_max_u64(uint64_t v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const uint32_t hi = __shfl_xor_sync(0xffffffffu, static_cast<uint32_t>(v >> 32), o);
    const uint32_t lo = __shfl_xor_sync(0xffffffffu, static_cast<uint32_t>(v), o);
    const uint64_t ov = (static_cast<uint64_t>(hi) << 32) | lo;
    if (ov > v) v = ov;
  }
  return v;
}

__host__ __device__ __forceinline__ size_t dh_wave_tile_offset(
    int cta_tile, int kg, int num_cta_tiles, int num_kg, int grid_ctas, size_t step_bytes) {
  const int super_iter = cta_tile / grid_ctas;
  const int cta_id = cta_tile - super_iter * grid_ctas;
  const int rem = num_cta_tiles - super_iter * grid_ctas;
  const int active = (rem < grid_ctas) ? rem : grid_ctas;
  return (static_cast<size_t>(super_iter * grid_ctas) * num_kg +
          static_cast<size_t>(kg) * active + cta_id) * step_bytes;
}

template <bool kVq25>
__global__ void __launch_bounds__(256) draft_head_pack_kernel(
    const uint8_t* __restrict__ raw_w,
    const uint8_t* __restrict__ raw_s,
    const float* __restrict__ global_scale,
    const int64_t* __restrict__ vocab_map,
    uint8_t* __restrict__ out,
    int N, int K, int grid_ctas, uint32_t perm_stride) {
  __shared__ float s_w[8][128];
  __shared__ __nv_bfloat16 s_sc[8];

  const int sub_n = blockIdx.x;
  const int kg = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  const int static_row = sub_n * 8 + warp;
  const uint32_t perm_row = static_cast<uint32_t>(
      (static_cast<uint64_t>(static_row) * perm_stride) % static_cast<uint32_t>(N));
  const int64_t orig_id = vocab_map[perm_row];
  const int k_base = kg * 128;
  const int num_g16 = K >> 4;
  const float gs = *global_scale;

  const uint8_t* w_row = raw_w + static_cast<size_t>(orig_id) * (K >> 1);
  const uint8_t* s_row = raw_s + static_cast<size_t>(orig_id) * num_g16;

  float rw[4];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int col0 = k_base + h * 64 + 2 * lane;
    const uint8_t packed_byte = w_row[col0 >> 1];
    const float sc_half = dh_fp8_e4m3_to_f32(s_row[col0 >> 4]) * 0.5f * gs;
    float v0 = dh_e2m1_x2_to_f32(packed_byte & 0xFu) * sc_half * dh_rad_sign(col0 + 0);
    float v1 = dh_e2m1_x2_to_f32(packed_byte >> 4) * sc_half * dh_rad_sign(col0 + 1);
    dh_warp_fwht64(v0, v1, lane);
    rw[2 * h + 0] = v0;
    rw[2 * h + 1] = v1;
    s_w[warp][h * 64 + 2 * lane + 0] = v0;
    s_w[warp][h * 64 + 2 * lane + 1] = v1;
  }

  float lsq = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) lsq = __fmaf_rn(rw[i], rw[i], lsq);
  const float rms = sqrtf(fmaxf(dh_warp_sum(lsq) * (1.0f / 128.0f), 1e-30f));

  __nv_bfloat16 best_sc_bf = __float2bfloat16_rn(rms);
  float best_err = 1e30f;

  auto eval_sc = [&](float sc_cand) {
    const __nv_bfloat16 sc_b = __float2bfloat16_rn(fmaxf(sc_cand, 1e-30f));
    const float sc_f = __bfloat162float(sc_b);
    const float inv_sc = (sc_f > 1e-30f) ? (1.0f / sc_f) : 0.0f;
    float lerr = 0.0f;
    if constexpr (!kVq25) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t q = dh_quant_nf3_sym(rw[i] * inv_sc);
        const float q_bf = __bfloat162float(__float2bfloat16_rn(kDhNf3Cb[q]));
        const float wh = __bfloat162float(__float2bfloat16_rn(q_bf * sc_f));
        const float d = rw[i] - wh;
        lerr = __fmaf_rn(d, d, lerr);
      }
    } else {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t q = dh_quant_vq25(rw[2 * h + 0] * inv_sc, rw[2 * h + 1] * inv_sc);
        const float2 c = kDhVq25Cb[q];
        const float c0 = __bfloat162float(__float2bfloat16_rn(c.x));
        const float c1 = __bfloat162float(__float2bfloat16_rn(c.y));
        const float wh0 = __bfloat162float(__float2bfloat16_rn(c0 * sc_f));
        const float wh1 = __bfloat162float(__float2bfloat16_rn(c1 * sc_f));
        const float d0 = rw[2 * h + 0] - wh0;
        const float d1 = rw[2 * h + 1] - wh1;
        lerr = __fmaf_rn(d0, d0, __fmaf_rn(d1, d1, lerr));
      }
    }
    const float tot_err = dh_warp_sum(lerr);
    if (tot_err < best_err) {
      best_err = tot_err;
      best_sc_bf = sc_b;
    }
  };

  constexpr float kInitMults[5] = {0.85f, 0.95f, 1.02f, 1.10f, 1.20f};
#pragma unroll
  for (int m = 0; m < 5; ++m) {
    eval_sc(rms * kInitMults[m]);
  }

  float cur_sc = __bfloat162float(best_sc_bf);
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const float inv_sc = (cur_sc > 1e-30f) ? (1.0f / cur_sc) : 0.0f;
    float l_wq = 0.0f, l_qq = 0.0f;
    if constexpr (!kVq25) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t q = dh_quant_nf3_sym(rw[i] * inv_sc);
        const float qf = __bfloat162float(__float2bfloat16_rn(kDhNf3Cb[q]));
        l_wq = __fmaf_rn(rw[i], qf, l_wq);
        l_qq = __fmaf_rn(qf, qf, l_qq);
      }
    } else {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t q = dh_quant_vq25(rw[2 * h + 0] * inv_sc, rw[2 * h + 1] * inv_sc);
        const float2 c = kDhVq25Cb[q];
        const float c0 = __bfloat162float(__float2bfloat16_rn(c.x));
        const float c1 = __bfloat162float(__float2bfloat16_rn(c.y));
        l_wq = __fmaf_rn(rw[2 * h + 0], c0, __fmaf_rn(rw[2 * h + 1], c1, l_wq));
        l_qq = __fmaf_rn(c0, c0, __fmaf_rn(c1, c1, l_qq));
      }
    }
    const float swq = dh_warp_sum(l_wq);
    const float sqq = dh_warp_sum(l_qq);
    if (sqq > 1e-20f) {
      cur_sc = fmaxf(swq / sqq, 1e-30f);
      eval_sc(cur_sc);
      cur_sc = __bfloat162float(best_sc_bf);
    }
  }

  if (lane == 0) {
    s_sc[warp] = best_sc_bf;
  }
  __syncthreads();

  if (warp != 0) return;

  const int r = lane >> 2;
  const int t = lane & 3;
  const float sc_f = __bfloat162float(s_sc[r]);
  const float inv_sc = (sc_f > 1e-30f) ? (1.0f / sc_f) : 0.0f;

  const int cta_tile = sub_n >> 3;
  const int sub_in_cta = sub_n & 7;
  const int num_cta_tiles = N / kDhRowsPerCTA;
  const int num_kg = K / kDhBK;
  constexpr size_t kStepBytes = kVq25 ? kVq25StepWBytes : kNf3StepWBytes;
  constexpr size_t kSubBytes = kVq25 ? kVq25SubTileBytes : kNf3SubTileBytes;

  uint8_t* dst = out + dh_wave_tile_offset(cta_tile, kg, num_cta_tiles, num_kg, grid_ctas, kStepBytes) +
                 static_cast<size_t>(sub_in_cta) * kSubBytes;

  if constexpr (!kVq25) {
    uint32_t c12[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int kb = s * 16 + 2 * t;
      const uint32_t q0 = dh_quant_nf3_sym(s_w[r][kb + 0] * inv_sc);
      const uint32_t q1 = dh_quant_nf3_sym(s_w[r][kb + 1] * inv_sc);
      const uint32_t q2 = dh_quant_nf3_sym(s_w[r][kb + 8] * inv_sc);
      const uint32_t q3 = dh_quant_nf3_sym(s_w[r][kb + 9] * inv_sc);
      c12[s] = q0 | (q1 << 3) | (q2 << 6) | (q3 << 9);
    }
    const uint32_t w0 = c12[0] | (c12[1] << 12) | (c12[2] << 24);
    const uint32_t w1 = (c12[2] >> 8) | (c12[3] << 4) | (c12[4] << 16) | (c12[5] << 28);
    const uint32_t w2 = (c12[5] >> 4) | (c12[6] << 8) | (c12[7] << 20);
    reinterpret_cast<uint32_t*>(dst + 0)[lane] = w0;
    reinterpret_cast<uint32_t*>(dst + 128)[lane] = w1;
    reinterpret_cast<uint32_t*>(dst + 256)[lane] = w2;
    if (t == 0) {
      reinterpret_cast<__nv_bfloat16*>(dst + 384)[r] = s_sc[r];
    }
  } else {
    uint32_t c10[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int kb = s * 16 + 2 * t;
      const uint32_t p01 = dh_quant_vq25(s_w[r][kb + 0] * inv_sc, s_w[r][kb + 1] * inv_sc);
      const uint32_t p23 = dh_quant_vq25(s_w[r][kb + 8] * inv_sc, s_w[r][kb + 9] * inv_sc);
      c10[s] = p01 | (p23 << 5);
    }
    const uint32_t w0 = c10[0] | (c10[1] << 10) | (c10[2] << 20) | (c10[3] << 30);
    const uint32_t w1 = (c10[3] >> 2) | (c10[4] << 8) | (c10[5] << 18) | (c10[6] << 28);
    const uint16_t w2 = static_cast<uint16_t>((c10[6] >> 4) | (c10[7] << 6));
    reinterpret_cast<uint32_t*>(dst + 0)[lane] = w0;
    reinterpret_cast<uint32_t*>(dst + 128)[lane] = w1;
    reinterpret_cast<uint16_t*>(dst + 256)[lane] = w2;
    if (t == 0) {
      reinterpret_cast<__nv_bfloat16*>(dst + 320)[r] = s_sc[r];
    }
  }
}

__global__ void __launch_bounds__(256) dhead_rotate_a_kernel(
    const __nv_bfloat16* __restrict__ A,
    __nv_bfloat16* __restrict__ A_rot,
    int M, int K, int pdl_mode) {
  const int blk64 = blockIdx.x;
  const int m = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int k_base = blk64 * 64;
  const int col8 = lane >> 2;
  const int phys_col8 = col8 ^ m;
  const int elem_off = phys_col8 * 8 + (lane & 3) * 2;
  __nv_bfloat16* dst_row = A_rot + (static_cast<size_t>(blk64) * 8 + m) * 64;

  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  if (m < M) {
    const int col0 = k_base + 2 * lane;
    const __nv_bfloat162 raw2 = *reinterpret_cast<const __nv_bfloat162*>(A + static_cast<size_t>(m) * K + col0);
    const float2 rf = __bfloat1622float2(raw2);
    float v0 = rf.x * dh_rad_sign(col0 + 0);
    float v1 = rf.y * dh_rad_sign(col0 + 1);
    dh_warp_fwht64(v0, v1, lane);
    *reinterpret_cast<__nv_bfloat162*>(dst_row + elem_off) = __floats2bfloat162_rn(v0, v1);
  } else {
    *reinterpret_cast<uint32_t*>(dst_row + elem_off) = 0u;
  }

  if ((pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
}

__device__ __forceinline__ void dh_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void dh_ldmatrix_x2(uint32_t& a0, uint32_t& a2, uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(a0), "=r"(a2)
      : "r"(addr));
}

__device__ __forceinline__ uint64_t dh_pol_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

template <bool kVq25>
struct alignas(128) DhSmem {
  static constexpr int kStepW = kVq25 ? kVq25StepWBytes : kNf3StepWBytes;
  union {
    struct {
      alignas(128) uint8_t w[kDhStages][kStepW];
      alignas(128) __nv_bfloat16 a[kDhStages][2][8][64];
    } pipe;
    uint64_t epilogue_top[kDhMaxM][64];
  };
  uint32_t lut[64];
  uint64_t mbar[kDhStages];
};

template <bool kVq25, int kTopPerCta, bool kWriteFullLogits = false>
__global__ void __launch_bounds__(128, 3) draft_head_fast_kernel(
    const __nv_bfloat16* __restrict__ A_rot,
    const uint8_t* __restrict__ W_packed,
    const int64_t* __restrict__ vocab_map,
    uint64_t* __restrict__ cand_out,
    __nv_bfloat16* __restrict__ full_logits_out,
    int M, int N, int K, int grid_ctas, uint32_t perm_stride, int pdl_mode) {
  __shared__ DhSmem<kVq25> smem;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

  if constexpr (!kVq25) {
    if (tid < 64) {
      smem.lut[tid] = dh_bf16x2_u32(kDhNf3Cb[tid & 7], kDhNf3Cb[(tid >> 3) & 7]);
    }
  } else {
    if (tid < 32) {
      const float2 c = kDhVq25Cb[tid];
      smem.lut[tid] = dh_bf16x2_u32(c.x, c.y);
    }
  }

  const int cta_id = blockIdx.x;
  const int num_cta_tiles = N / kDhRowsPerCTA;
  const int num_kg = K / kDhBK;
  int my_tiles = 0;
  if (cta_id < num_cta_tiles) {
    my_tiles = (num_cta_tiles - 1 - cta_id) / grid_ctas + 1;
  }
  const int total_steps = my_tiles * num_kg;

  constexpr size_t kStepW = DhSmem<kVq25>::kStepW;
  constexpr size_t kSubW = kVq25 ? kVq25SubTileBytes : kNf3SubTileBytes;

  auto w_stage_ptr = [&](int step) -> const uint8_t* {
    const int super_iter = step / num_kg;
    const int kg = step - super_iter * num_kg;
    const int cta_tile = super_iter * grid_ctas + cta_id;
    return W_packed + dh_wave_tile_offset(cta_tile, kg, num_cta_tiles, num_kg, grid_ctas, kStepW);
  };

  auto a_stage_ptr = [&](int kg) -> const __nv_bfloat16* {
    return A_rot + static_cast<size_t>(kg) * 1024;
  };

  auto mbar_addr = [&](int st) -> uint32_t {
    return static_cast<uint32_t>(__cvta_generic_to_shared(&smem.mbar[st]));
  };

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kDhStages; ++s) {
      const uint32_t b = mbar_addr(s);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  constexpr int kPreload = kDhStages - 1;
  const int preload = (total_steps < kPreload) ? total_steps : kPreload;

  if (tid == 0) {
    const uint64_t pol = dh_pol_evict_first();
    for (int p = 0; p < preload; ++p) {
      const uint32_t b = mbar_addr(p);
      const uint32_t dst_w = static_cast<uint32_t>(__cvta_generic_to_shared(smem.pipe.w[p]));
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                   ::"r"(b), "r"(static_cast<int>(kStepW))
                   : "memory");
      asm volatile(
          "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
          ::"r"(dst_w), "l"(w_stage_ptr(p)), "r"(static_cast<int>(kStepW)), "r"(b), "l"(pol)
          : "memory");
    }
  }
  if ((pdl_mode & 1) != 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  for (int p = 0; p < preload; ++p) {
    const int kg = p % num_kg;
    reinterpret_cast<uint4*>(smem.pipe.a[p])[tid] =
        __ldg(&reinterpret_cast<const uint4*>(a_stage_ptr(kg))[tid]);
  }
  __syncthreads();

  uint64_t my_top[4] = {0ull, 0ull, 0ull, 0ull};
  float d0_0 = 0.0f, d1_0 = 0.0f, d2_0 = 0.0f, d3_0 = 0.0f;
  float d0_1 = 0.0f, d1_1 = 0.0f, d2_1 = 0.0f, d3_1 = 0.0f;

  const int ldm_row = lane & 7;
  const int ldm_col_off = (lane >> 3) & 1;
  const int trig_step = ((pdl_mode & 2) != 0 && total_steps > 0)
                            ? ((total_steps > 8) ? (total_steps - 8) : 0)
                            : -1;

  int super_iter = 0;
  int kg = 0;
  for (int step = 0; step < total_steps; ++step) {
    if (step == trig_step) {
      asm volatile("griddepcontrol.launch_dependents;");
    }
    const int st = step & (kDhStages - 1);
    const int next_step = step + kPreload;

    if (next_step < total_steps && tid == 0) {
      const int nst = next_step & (kDhStages - 1);
      const int nkg = next_step % num_kg;
      const uint32_t b = mbar_addr(nst);
      const uint32_t dst_w = static_cast<uint32_t>(__cvta_generic_to_shared(smem.pipe.w[nst]));
      const uint32_t dst_a = static_cast<uint32_t>(__cvta_generic_to_shared(smem.pipe.a[nst]));
      const uint64_t pol = dh_pol_evict_first();
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                   ::"r"(b), "r"(static_cast<int>(kStepW + 2048))
                   : "memory");
      asm volatile(
          "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
          ::"r"(dst_w), "l"(w_stage_ptr(next_step)), "r"(static_cast<int>(kStepW)), "r"(b), "l"(pol)
          : "memory");
      asm volatile(
          "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
          ::"r"(dst_a), "l"(a_stage_ptr(nkg)), "r"(2048), "r"(b)
          : "memory");
    }

    const uint32_t cur_bar = mbar_addr(st);
    const uint32_t parity = static_cast<uint32_t>((step >> 2) & 1);
    asm volatile(
        "{\n\t.reg .pred P1;\n\tDH_MBAR_WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
        "@!P1 bra DH_MBAR_WAIT_%=;\n\t}"
        ::"r"(cur_bar), "r"(parity)
        : "memory");

    const uint8_t* wt0 = &smem.pipe.w[st][(warp * 2 + 0) * kSubW];
    const uint8_t* wt1 = &smem.pipe.w[st][(warp * 2 + 1) * kSubW];

    if constexpr (!kVq25) {
      const uint32_t w0_0 = reinterpret_cast<const uint32_t*>(wt0 + 0)[lane];
      const uint32_t w1_0 = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
      const uint32_t w2_0 = reinterpret_cast<const uint32_t*>(wt0 + 256)[lane];
      const __nv_bfloat162 sc0 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16*>(wt0 + 384)[g]);

      const uint32_t w0_1 = reinterpret_cast<const uint32_t*>(wt1 + 0)[lane];
      const uint32_t w1_1 = reinterpret_cast<const uint32_t*>(wt1 + 128)[lane];
      const uint32_t w2_1 = reinterpret_cast<const uint32_t*>(wt1 + 256)[lane];
      const __nv_bfloat162 sc1 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16*>(wt1 + 384)[g]);

      const uint32_t c0_arr[8] = {
          w0_0 & 0xFFFu,
          (w0_0 >> 12) & 0xFFFu,
          __funnelshift_r(w0_0, w1_0, 24) & 0xFFFu,
          (w1_0 >> 4) & 0xFFFu,
          (w1_0 >> 16) & 0xFFFu,
          __funnelshift_r(w1_0, w2_0, 28) & 0xFFFu,
          (w2_0 >> 8) & 0xFFFu,
          w2_0 >> 20,
      };
      const uint32_t c1_arr[8] = {
          w0_1 & 0xFFFu,
          (w0_1 >> 12) & 0xFFFu,
          __funnelshift_r(w0_1, w1_1, 24) & 0xFFFu,
          (w1_1 >> 4) & 0xFFFu,
          (w1_1 >> 16) & 0xFFFu,
          __funnelshift_r(w1_1, w2_1, 28) & 0xFFFu,
          (w2_1 >> 8) & 0xFFFu,
          w2_1 >> 20,
      };

#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int h = s >> 2;
        const int s4 = s & 3;
        const int pcol8 = (2 * s4 + ldm_col_off) ^ ldm_row;
        const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.pipe.a[st][h][ldm_row][pcol8 * 8]));
        uint32_t a0, a2;
        dh_ldmatrix_x2(a0, a2, addr);

        const uint32_t q01_0 = smem.lut[c0_arr[s] & 63u];
        const uint32_t q23_0 = smem.lut[c0_arr[s] >> 6];
        const __nv_bfloat162 b01_0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q01_0), sc0);
        const __nv_bfloat162 b23_0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q23_0), sc0);

        const uint32_t q01_1 = smem.lut[c1_arr[s] & 63u];
        const uint32_t q23_1 = smem.lut[c1_arr[s] >> 6];
        const __nv_bfloat162 b01_1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q01_1), sc1);
        const __nv_bfloat162 b23_1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q23_1), sc1);

        dh_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2,
                        *reinterpret_cast<const uint32_t*>(&b01_0),
                        *reinterpret_cast<const uint32_t*>(&b23_0));
        dh_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2,
                        *reinterpret_cast<const uint32_t*>(&b01_1),
                        *reinterpret_cast<const uint32_t*>(&b23_1));
      }
    } else {
      const uint32_t w0_0 = reinterpret_cast<const uint32_t*>(wt0 + 0)[lane];
      const uint32_t w1_0 = reinterpret_cast<const uint32_t*>(wt0 + 128)[lane];
      const uint32_t w2_0 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wt0 + 256)[lane]);
      const __nv_bfloat162 sc0 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16*>(wt0 + 320)[g]);

      const uint32_t w0_1 = reinterpret_cast<const uint32_t*>(wt1 + 0)[lane];
      const uint32_t w1_1 = reinterpret_cast<const uint32_t*>(wt1 + 128)[lane];
      const uint32_t w2_1 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(wt1 + 256)[lane]);
      const __nv_bfloat162 sc1 = __bfloat162bfloat162(reinterpret_cast<const __nv_bfloat16*>(wt1 + 320)[g]);

      const uint32_t c0_arr[8] = {
          w0_0 & 0x3FFu,
          (w0_0 >> 10) & 0x3FFu,
          (w0_0 >> 20) & 0x3FFu,
          __funnelshift_r(w0_0, w1_0, 30) & 0x3FFu,
          (w1_0 >> 8) & 0x3FFu,
          (w1_0 >> 18) & 0x3FFu,
          __funnelshift_r(w1_0, w2_0, 28) & 0x3FFu,
          (w2_0 >> 6) & 0x3FFu,
      };
      const uint32_t c1_arr[8] = {
          w0_1 & 0x3FFu,
          (w0_1 >> 10) & 0x3FFu,
          (w0_1 >> 20) & 0x3FFu,
          __funnelshift_r(w0_1, w1_1, 30) & 0x3FFu,
          (w1_1 >> 8) & 0x3FFu,
          (w1_1 >> 18) & 0x3FFu,
          __funnelshift_r(w1_1, w2_1, 28) & 0x3FFu,
          (w2_1 >> 6) & 0x3FFu,
      };

#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int h = s >> 2;
        const int s4 = s & 3;
        const int pcol8 = (2 * s4 + ldm_col_off) ^ ldm_row;
        const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.pipe.a[st][h][ldm_row][pcol8 * 8]));
        uint32_t a0, a2;
        dh_ldmatrix_x2(a0, a2, addr);

        const uint32_t q01_0 = smem.lut[c0_arr[s] & 31u];
        const uint32_t q23_0 = smem.lut[c0_arr[s] >> 5];
        const __nv_bfloat162 b01_0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q01_0), sc0);
        const __nv_bfloat162 b23_0 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q23_0), sc0);

        const uint32_t q01_1 = smem.lut[c1_arr[s] & 31u];
        const uint32_t q23_1 = smem.lut[c1_arr[s] >> 5];
        const __nv_bfloat162 b01_1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q01_1), sc1);
        const __nv_bfloat162 b23_1 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q23_1), sc1);

        dh_mma_m16n8k16(d0_0, d1_0, d2_0, d3_0, a0, a2,
                        *reinterpret_cast<const uint32_t*>(&b01_0),
                        *reinterpret_cast<const uint32_t*>(&b23_0));
        dh_mma_m16n8k16(d0_1, d1_1, d2_1, d3_1, a0, a2,
                        *reinterpret_cast<const uint32_t*>(&b01_1),
                        *reinterpret_cast<const uint32_t*>(&b23_1));
      }
    }

    if (kg + 1 == num_kg) {
      const int cta_tile = super_iter * grid_ctas + cta_id;
      const uint32_t col0 = static_cast<uint32_t>(cta_tile * kDhRowsPerCTA + (warp * 2 + 0) * 8 + 2 * t);
      const uint32_t col1 = static_cast<uint32_t>(cta_tile * kDhRowsPerCTA + (warp * 2 + 1) * 8 + 2 * t);
      if (g < M) {
        if constexpr (kWriteFullLogits) {
          const uint32_t un = static_cast<uint32_t>(N);
          const uint32_t p00 = static_cast<uint32_t>((static_cast<uint64_t>(col0 + 0u) * perm_stride) % un);
          const uint32_t p01 = static_cast<uint32_t>((static_cast<uint64_t>(col0 + 1u) * perm_stride) % un);
          const uint32_t p10 = static_cast<uint32_t>((static_cast<uint64_t>(col1 + 0u) * perm_stride) % un);
          const uint32_t p11 = static_cast<uint32_t>((static_cast<uint64_t>(col1 + 1u) * perm_stride) % un);
          __nv_bfloat16* row_out = full_logits_out + static_cast<size_t>(g) * N;
          row_out[p00] = __float2bfloat16_rn(d0_0);
          row_out[p01] = __float2bfloat16_rn(d1_0);
          row_out[p10] = __float2bfloat16_rn(d0_1);
          row_out[p11] = __float2bfloat16_rn(d1_1);
        } else {
          dh_top4_insert_u64(my_top, (static_cast<uint64_t>(dh_f32_to_ord_u32(d0_0)) << 32) | (col0 + 0u));
          dh_top4_insert_u64(my_top, (static_cast<uint64_t>(dh_f32_to_ord_u32(d1_0)) << 32) | (col0 + 1u));
          dh_top4_insert_u64(my_top, (static_cast<uint64_t>(dh_f32_to_ord_u32(d0_1)) << 32) | (col1 + 0u));
          dh_top4_insert_u64(my_top, (static_cast<uint64_t>(dh_f32_to_ord_u32(d1_1)) << 32) | (col1 + 1u));
        }
      }
      d0_0 = d1_0 = d2_0 = d3_0 = 0.0f;
      d0_1 = d1_1 = d2_1 = d3_1 = 0.0f;
      ++super_iter;
      kg = 0;
    } else {
      ++kg;
    }

    __syncthreads();
  }

  if (total_steps == 0 && (pdl_mode & 2) != 0) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
  if constexpr (kWriteFullLogits) {
    return;
  }

  __syncthreads();
  if (g < M) {
    const int base_idx = warp * 16 + t * 4;
    smem.epilogue_top[g][base_idx + 0] = my_top[0];
    smem.epilogue_top[g][base_idx + 1] = my_top[1];
    smem.epilogue_top[g][base_idx + 2] = my_top[2];
    smem.epilogue_top[g][base_idx + 3] = my_top[3];
  }
  __syncthreads();

  const int num_cands = grid_ctas * kTopPerCta;
  const uint32_t un = static_cast<uint32_t>(N);
  for (int m = warp; m < M; m += kDhWarps) {
    const uint64_t u0 = smem.epilogue_top[m][2 * lane + 0];
    const uint64_t u1 = smem.epilogue_top[m][2 * lane + 1];
    uint64_t cur = (u0 >= u1) ? u0 : u1;
    uint64_t nxt = (u0 >= u1) ? u1 : u0;
    uint64_t picked = 0ull;

#pragma unroll
    for (int r = 0; r < kTopPerCta; ++r) {
      const uint64_t best = dh_warp_max_u64(cur);
      if (lane == r) picked = best;
      if (best != 0ull) {
        const uint32_t msk = __ballot_sync(0xffffffffu, cur == best);
        const int win = __ffs(msk) - 1;
        if (lane == win) {
          cur = nxt;
          nxt = 0ull;
        }
      }
    }

    if (lane < kTopPerCta) {
      uint64_t out_val = 0ull;
      if (picked != 0ull) {
        const uint32_t scol = static_cast<uint32_t>(picked & 0xffffffffull);
        const uint32_t perm_col = static_cast<uint32_t>((static_cast<uint64_t>(scol) * perm_stride) % un);
        const uint32_t orig_id = static_cast<uint32_t>(__ldg(&vocab_map[perm_col]));
        out_val = (picked & 0xffffffff00000000ull) | (0xffffffffu - orig_id);
      }
      cand_out[static_cast<size_t>(m) * num_cands + cta_id * kTopPerCta + lane] = out_val;
    }
  }
}

struct DraftHeadFastState {
  torch::Tensor a_rot;
  torch::Tensor cand_buf;
  torch::Tensor sentinel;
  int64_t a_rot_elems = 0;
  int last_num_cands = 384;
  int last_rescore_k = 64;
  c10::DeviceIndex dev = -1;
};

DraftHeadFastState& get_dh_state(torch::Device device, int K) {
  static DraftHeadFastState st;
  const int64_t need_arot = static_cast<int64_t>(K / 64) * 8 * 64;
  if (!st.sentinel.defined() || st.dev != device.index() || st.a_rot_elems < need_arot) {
    auto opt_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    st.a_rot = torch::zeros({need_arot}, opt_bf16);
    st.cand_buf = torch::zeros({kDhMaxM, kDhMaxCandPool}, opt_i64);
    st.sentinel = torch::zeros({kDhMaxM, kDhSentinelCols}, opt_bf16);
    st.a_rot_elems = need_arot;
    st.dev = device.index();
  }
  return st;
}

template <bool kVq25>
static void set_dh_carveout_one() {
  cudaFuncSetAttribute(draft_head_fast_kernel<kVq25, 8, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(draft_head_fast_kernel<kVq25, 4, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(draft_head_fast_kernel<kVq25, 2, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(draft_head_fast_kernel<kVq25, 4, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

static void ensure_dh_carveout() {
  static bool done = false;
  if (!done) {
    done = true;
    set_dh_carveout_one<false>();
    set_dh_carveout_one<true>();
  }
}

}

bool draft_head_try_get_cands(
    const torch::Tensor& static_logits,
    const uint64_t*& out_cand_ptr,
    int& out_num_cands,
    int& out_rescore_k) {
  if (!static_logits.is_cuda() || static_logits.size(1) != kDhSentinelCols) {
    return false;
  }
  auto& st = get_dh_state(static_logits.device(), 6656);
  if (!st.sentinel.defined() || static_logits.data_ptr() != st.sentinel.data_ptr()) {
    return false;
  }
  out_cand_ptr = reinterpret_cast<const uint64_t*>(st.cand_buf.data_ptr());
  out_num_cands = st.last_num_cands;
  out_rescore_k = st.last_rescore_k;
  return true;
}

torch::Tensor draft_head_pack(
    torch::Tensor raw_w,
    torch::Tensor raw_s,
    torch::Tensor global_scale,
    torch::Tensor vocab_map,
    int64_t grid_ctas = 96,
    int64_t mode = 0) {
  TORCH_CHECK(raw_w.is_cuda() && raw_w.scalar_type() == torch::kUInt8 && raw_w.dim() == 2 && raw_w.is_contiguous());
  TORCH_CHECK(raw_s.is_cuda() && raw_s.dim() == 2 && raw_s.is_contiguous());
  TORCH_CHECK(vocab_map.is_cuda() && vocab_map.scalar_type() == torch::kInt64 && vocab_map.is_contiguous());
  const int N = static_cast<int>(vocab_map.numel());
  const int K = static_cast<int>(raw_w.size(1) * 2);
  TORCH_CHECK(N >= 64 && (N % kDhRowsPerCTA) == 0 && K >= 128 && (K % kDhBK) == 0);
  TORCH_CHECK(grid_ctas == 48 || grid_ctas == 96 || grid_ctas == 192,
              "draft_head_pack grid_ctas must be 48, 96, or 192");

  (void)get_dh_state(raw_w.device(), K);
  ensure_dh_carveout();

  const uint32_t pstride = dh_perm_stride(N);
  const bool vq25 = (mode == 1);
  const int64_t sub_bytes = vq25 ? kVq25SubTileBytes : kNf3SubTileBytes;
  const int64_t total_bytes = static_cast<int64_t>(N / 8) * static_cast<int64_t>(K / kDhBK) * sub_bytes;
  auto out = torch::empty({total_bytes}, raw_w.options().dtype(torch::kUInt8));

  dim3 grid(static_cast<unsigned>(N / 8), static_cast<unsigned>(K / kDhBK), 1u);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  if (!vq25) {
    draft_head_pack_kernel<false><<<grid, 256, 0, stream>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        vocab_map.data_ptr<int64_t>(),
        out.data_ptr<uint8_t>(),
        N, K, static_cast<int>(grid_ctas), pstride);
  } else {
    draft_head_pack_kernel<true><<<grid, 256, 0, stream>>>(
        raw_w.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(raw_s.data_ptr()),
        global_scale.data_ptr<float>(),
        vocab_map.data_ptr<int64_t>(),
        out.data_ptr<uint8_t>(),
        N, K, static_cast<int>(grid_ctas), pstride);
  }
  return out;
}

torch::Tensor draft_head_topc(
    torch::Tensor A,
    torch::Tensor W_packed,
    torch::Tensor vocab_map,
    int64_t N,
    int64_t K,
    int64_t grid_ctas = 96,
    int64_t mode = 0,
    int64_t rescore_k = 0) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous());
  const int M = static_cast<int>(A.size(0));
  TORCH_CHECK(M >= 1 && M <= kDhMaxM && A.size(1) == K);
  TORCH_CHECK(grid_ctas == 48 || grid_ctas == 96 || grid_ctas == 192);

  ensure_dh_carveout();
  auto& st = get_dh_state(A.device(), static_cast<int>(K));
  const int top_per_cta = (grid_ctas == 48) ? 8 : ((grid_ctas == 96) ? 4 : 2);
  st.last_num_cands = static_cast<int>(grid_ctas) * top_per_cta;
  st.last_rescore_k = (rescore_k > 0) ? static_cast<int>(rescore_k) : dh_rescore_k_cfg();

  const int pdl = dh_pdl_mode();
  const bool vq25 = (mode == 1);
  const uint32_t pstride = dh_perm_stride(static_cast<int>(N));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  dh_launch_pdl(
      dhead_rotate_a_kernel,
      dim3(static_cast<unsigned>(K / 64), 1u, 1u), 256, stream, pdl,
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(st.a_rot.data_ptr()),
      M, static_cast<int>(K), pdl);

  auto launch_main = [&](auto vq_tag, auto top_tag) {
    constexpr bool KVQ = decltype(vq_tag)::value;
    constexpr int KTOP = decltype(top_tag)::value;
    dh_launch_pdl(
        draft_head_fast_kernel<KVQ, KTOP, false>,
        dim3(static_cast<unsigned>(grid_ctas), 1u, 1u), kDhThreads, stream, pdl,
        reinterpret_cast<const __nv_bfloat16*>(st.a_rot.data_ptr()),
        W_packed.data_ptr<uint8_t>(),
        vocab_map.data_ptr<int64_t>(),
        reinterpret_cast<uint64_t*>(st.cand_buf.data_ptr()),
        static_cast<__nv_bfloat16*>(nullptr),
        M, static_cast<int>(N), static_cast<int>(K), static_cast<int>(grid_ctas), pstride, pdl);
  };

  if (!vq25) {
    if (top_per_cta == 8) launch_main(std::false_type{}, std::integral_constant<int, 8>{});
    else if (top_per_cta == 4) launch_main(std::false_type{}, std::integral_constant<int, 4>{});
    else launch_main(std::false_type{}, std::integral_constant<int, 2>{});
  } else {
    if (top_per_cta == 8) launch_main(std::true_type{}, std::integral_constant<int, 8>{});
    else if (top_per_cta == 4) launch_main(std::true_type{}, std::integral_constant<int, 4>{});
    else launch_main(std::true_type{}, std::integral_constant<int, 2>{});
  }

  return st.sentinel.narrow(0, 0, M);
}

torch::Tensor draft_head_gemm_full(
    torch::Tensor A,
    torch::Tensor W_packed,
    int64_t N,
    int64_t K,
    int64_t grid_ctas = 96,
    int64_t mode = 0) {
  TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kBFloat16 && A.dim() == 2 && A.is_contiguous());
  const int total_m = static_cast<int>(A.size(0));
  TORCH_CHECK(total_m >= 1 && A.size(1) == K);
  TORCH_CHECK(grid_ctas == 48 || grid_ctas == 96 || grid_ctas == 192);

  ensure_dh_carveout();
  auto& st = get_dh_state(A.device(), static_cast<int>(K));
  auto out = torch::empty({total_m, N}, A.options());
  const int pdl = dh_pdl_mode();
  const bool vq25 = (mode == 1);
  const uint32_t pstride = dh_perm_stride(static_cast<int>(N));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  for (int m0 = 0; m0 < total_m; m0 += kDhMaxM) {
    const int cm = std::min(kDhMaxM, total_m - m0);
    dh_launch_pdl(
        dhead_rotate_a_kernel,
        dim3(static_cast<unsigned>(K / 64), 1u, 1u), 256, stream, pdl,
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()) + static_cast<size_t>(m0) * K,
        reinterpret_cast<__nv_bfloat16*>(st.a_rot.data_ptr()),
        cm, static_cast<int>(K), pdl);
    if (!vq25) {
      dh_launch_pdl(
          draft_head_fast_kernel<false, 4, true>,
          dim3(static_cast<unsigned>(grid_ctas), 1u, 1u), kDhThreads, stream, pdl,
          reinterpret_cast<const __nv_bfloat16*>(st.a_rot.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          static_cast<const int64_t*>(nullptr),
          static_cast<uint64_t*>(nullptr),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()) + static_cast<size_t>(m0) * N,
          cm, static_cast<int>(N), static_cast<int>(K), static_cast<int>(grid_ctas), pstride, pdl);
    } else {
      dh_launch_pdl(
          draft_head_fast_kernel<true, 4, true>,
          dim3(static_cast<unsigned>(grid_ctas), 1u, 1u), kDhThreads, stream, pdl,
          reinterpret_cast<const __nv_bfloat16*>(st.a_rot.data_ptr()),
          W_packed.data_ptr<uint8_t>(),
          static_cast<const int64_t*>(nullptr),
          static_cast<uint64_t*>(nullptr),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()) + static_cast<size_t>(m0) * N,
          cm, static_cast<int>(N), static_cast<int>(K), static_cast<int>(grid_ctas), pstride, pdl);
    }
  }
  return out;
}
