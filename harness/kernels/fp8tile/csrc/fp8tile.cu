// SPDX-License-Identifier: Apache-2.0
// fp8tile C API: dispatch to the configurations this library was built with
// (FP8TILE_HAVE_<i>); see fp8tile_gemm.cuh for what they compute.
#include <cuda_runtime.h>
#include <stddef.h>

#define FP8TILE_DECL(I)                                                                       \
  extern "C" int fp8tile_ws_##I(int m, int n, int k, size_t* bytes);                          \
  extern "C" int fp8tile_run_##I(const void* a, const void* b, const float* sa, const float* sb, \
                                 void* d, int m, int n, int k, void* ws, cudaStream_t st);
#ifdef FP8TILE_HAVE_0
FP8TILE_DECL(0)
#endif
#ifdef FP8TILE_HAVE_1
FP8TILE_DECL(1)
#endif
#ifdef FP8TILE_HAVE_2
FP8TILE_DECL(2)
#endif
#ifdef FP8TILE_HAVE_3
FP8TILE_DECL(3)
#endif
#ifdef FP8TILE_HAVE_4
FP8TILE_DECL(4)
#endif
#ifdef FP8TILE_HAVE_5
FP8TILE_DECL(5)
#endif
#ifdef FP8TILE_HAVE_6
FP8TILE_DECL(6)
#endif
#ifdef FP8TILE_HAVE_7
FP8TILE_DECL(7)
#endif

extern "C" {

// Bit i set: configuration i is built.
int fp8tile_configs(void) {
  int mask = 0;
#ifdef FP8TILE_HAVE_0
  mask |= 1 << 0;
#endif
#ifdef FP8TILE_HAVE_1
  mask |= 1 << 1;
#endif
#ifdef FP8TILE_HAVE_2
  mask |= 1 << 2;
#endif
#ifdef FP8TILE_HAVE_3
  mask |= 1 << 3;
#endif
#ifdef FP8TILE_HAVE_4
  mask |= 1 << 4;
#endif
#ifdef FP8TILE_HAVE_5
  mask |= 1 << 5;
#endif
#ifdef FP8TILE_HAVE_6
  mask |= 1 << 6;
#endif
#ifdef FP8TILE_HAVE_7
  mask |= 1 << 7;
#endif
  return mask;
}

int fp8tile_workspace(int cfg, int m, int n, int k, size_t* bytes) {
#define FP8TILE_WS(I) \
  if (cfg == I) return fp8tile_ws_##I(m, n, k, bytes);
#ifdef FP8TILE_HAVE_0
  FP8TILE_WS(0)
#endif
#ifdef FP8TILE_HAVE_1
  FP8TILE_WS(1)
#endif
#ifdef FP8TILE_HAVE_2
  FP8TILE_WS(2)
#endif
#ifdef FP8TILE_HAVE_3
  FP8TILE_WS(3)
#endif
#ifdef FP8TILE_HAVE_4
  FP8TILE_WS(4)
#endif
#ifdef FP8TILE_HAVE_5
  FP8TILE_WS(5)
#endif
#ifdef FP8TILE_HAVE_6
  FP8TILE_WS(6)
#endif
#ifdef FP8TILE_HAVE_7
  FP8TILE_WS(7)
#endif
  return -10;
}

// d[m, n] (row-major bf16) = bf16_rn(sa * fp32_rn(sb * (a[m, k] . b[k, n]))) with a
// row-major e4m3, b column-major e4m3 (the [n, k] row-major bytes), sa and sb
// device pointers to fp32 scalars, ws a device workspace of fp8tile_workspace
// bytes. Returns 0, a negative code or a cudaError_t value.
int fp8tile_gemm(int cfg, const void* a, const void* b, const void* sa, const void* sb, void* d,
                 int m, int n, int k, void* ws, void* stream) {
  if (m < 1 || m > 16 || n <= 0 || k <= 0 || n % 16 || k % 16) return -1;
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const float* fa = static_cast<const float*>(sa);
  const float* fb = static_cast<const float*>(sb);
#define FP8TILE_RUN(I) \
  if (cfg == I) return fp8tile_run_##I(a, b, fa, fb, d, m, n, k, ws, st);
#ifdef FP8TILE_HAVE_0
  FP8TILE_RUN(0)
#endif
#ifdef FP8TILE_HAVE_1
  FP8TILE_RUN(1)
#endif
#ifdef FP8TILE_HAVE_2
  FP8TILE_RUN(2)
#endif
#ifdef FP8TILE_HAVE_3
  FP8TILE_RUN(3)
#endif
#ifdef FP8TILE_HAVE_4
  FP8TILE_RUN(4)
#endif
#ifdef FP8TILE_HAVE_5
  FP8TILE_RUN(5)
#endif
#ifdef FP8TILE_HAVE_6
  FP8TILE_RUN(6)
#endif
#ifdef FP8TILE_HAVE_7
  FP8TILE_RUN(7)
#endif
  return -10;
}

}  // extern "C"
