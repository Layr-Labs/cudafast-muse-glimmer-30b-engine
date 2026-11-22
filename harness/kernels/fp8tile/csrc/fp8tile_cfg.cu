// SPDX-License-Identifier: Apache-2.0
// One fp8tile configuration (FP8TILE_CFG = index into Config<>), compiled as
// its own object so that the configurations build in parallel.
#include "fp8tile_gemm.cuh"

#ifndef FP8TILE_CFG
#error "FP8TILE_CFG must be defined"
#endif

#define FP8TILE_CAT2(a, b) a##b
#define FP8TILE_CAT(a, b) FP8TILE_CAT2(a, b)

extern "C" int FP8TILE_CAT(fp8tile_ws_, FP8TILE_CFG)(int m, int n, int k, size_t* bytes) {
  return fp8tile::Config<FP8TILE_CFG>::G::workspace(m, n, k, bytes);
}

extern "C" int FP8TILE_CAT(fp8tile_run_, FP8TILE_CFG)(const void* a, const void* b,
                                                       const float* sa, const float* sb, void* d,
                                                       int m, int n, int k, void* ws,
                                                       cudaStream_t st) {
  return fp8tile::Config<FP8TILE_CFG>::G::run(a, b, sa, sb, d, m, n, k, ws, st);
}
