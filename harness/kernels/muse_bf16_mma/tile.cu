// SPDX-License-Identifier: Apache-2.0
// Weight-only, *lossless* BF16 transport. Each warp computes 16 output columns
// and up to 8 input rows. Advance one BF16 MMA k16 at a time in dense order.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr unsigned kFull = 0xffffffffu;
// Keep sign + low 11 bits in each 16-bit half. The four holes in each half
// hold a third of another BF16 pair, whose exponent is also stored as a delta.
constexpr uint32_t kKeep = 0x87ff87ffu;
__device__ __forceinline__ uint32_t pair(const uint16_t* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ void dense_fragment(const uint16_t* w, int k, int nt,
                                               int kt, int lane, uint32_t (&v)[4]) {
    const uint16_t* q = w + (nt * 16 + (lane >> 2)) * k + 16 * kt + 2 * (lane & 3);
    v[0] = pair(q);     v[1] = pair(q + 8*k);
    v[2] = pair(q + 8); v[3] = pair(q + 8*k + 8);
}
__device__ __forceinline__ int pid(int nt, int kt, int tk) {
    return ((nt / 8) * tk + kt) * 8 + (nt & 7);
}
__global__ void pack(const uint16_t* __restrict__ w, uint32_t* __restrict__ payload,
                     int* __restrict__ meta, int n, int k) {
    int lane = threadIdx.x & 31, tk = k/16;
    int logical = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    if (logical >= (n/16)*tk) return;
    int nt = logical/tk, kt = logical%tk;
    uint32_t v[4]; dense_fragment(w, k, nt, kt, lane, v);
    int hi = 0, lo = 255;
    #pragma unroll
    for (int i=0; i<4; ++i) {
        int a = (v[i] >> 7) & 255, b = (v[i] >> 23) & 255;
        hi = max(hi, max(a,b)); lo = min(lo, min(a,b));
    }
    #pragma unroll
    for (int j=16; j; j>>=1) {
        hi = max(hi, __shfl_xor_sync(kFull,hi,j));
        lo = min(lo, __shfl_xor_sync(kFull,lo,j));
    }
    if (lane == 0) meta[logical] = hi-lo < 16 ? lo : -1;
    if (hi-lo >= 16) return;  // Decode reads the original tile, not payload.
    uint32_t base = (lo << 7)*0x00010001u;
    #pragma unroll
    for (int i=0; i<4; ++i) v[i] -= base; // No borrow between halves.
    uint32_t* p = payload + pid(nt,kt,tk)*96 + lane;
    p[0]  = (v[0]&kKeep) | ((v[3]&0x000f000fu)<<11);
    p[32] = (v[1]&kKeep) | ((v[3]&0x00f000f0u)<<7);
    p[64] = (v[2]&kKeep) | ((v[3]&0x07000700u)<<3) | ((v[3]&0x80008000u)>>1);
}
__device__ __forceinline__ void fragment(const uint16_t* w, const uint32_t* payload,
        const int* meta, int k, int nt, int kt, int lane, uint32_t (&a)[4]) {
    int tk = k/16, e = meta[nt*tk + kt];
    if (e < 0) { dense_fragment(w,k,nt,kt,lane,a); return; }
    const uint32_t* p = payload + pid(nt,kt,tk)*96 + lane;
    uint32_t p0=p[0], p1=p[32], p2=p[64], base=(e<<7)*0x00010001u;
    a[0]=(p0&kKeep)+base; a[1]=(p1&kKeep)+base; a[2]=(p2&kKeep)+base;
    a[3]=(((p0>>11)&0x000f000fu) | ((p1>>7)&0x00f000f0u) |
          ((p2>>3)&0x07000700u) | ((p2<<1)&0x80008000u))+base;
}
__global__ void verify(const uint16_t* w, const uint32_t* payload, const int* meta,
                       int n, int k, int* bad) {
    int lane=threadIdx.x&31, tk=k/16, tile=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if (tile >= (n/16)*tk) return;
    uint32_t ref[4], got[4];
    dense_fragment(w,k,tile/tk,tile%tk,lane,ref);
    fragment(w,payload,meta,k,tile/tk,tile%tk,lane,got);
    if (got[0]!=ref[0] || got[1]!=ref[1] || got[2]!=ref[2] || got[3]!=ref[3]) atomicOr(bad,1);
}
__global__ void project(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
        const uint32_t* __restrict__ payload, const int* __restrict__ meta,
        __nv_bfloat16* __restrict__ y, int m, int n, int k) {
    int lane=threadIdx.x&31, nt=(blockIdx.x*blockDim.x+threadIdx.x)>>5, tk=k/16;
    if (nt >= n/16) return;
    float d[4]={}; int j=lane>>2, c=2*(lane&3);
    const uint16_t* xp=x+(j<m?j:0)*k+c;
    for (int kt=0; kt<tk; ++kt) {
        uint32_t a[4]; fragment(w,payload,meta,k,nt,kt,lane,a);
        uint32_t b0=j<m?pair(xp+16*kt):0, b1=j<m?pair(xp+16*kt+8):0;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    #pragma unroll
    for (int i=0; i<4; ++i) {
        int row=c+(i&1);
        if (row<m) y[row*n + nt*16 + j + 8*(i>>1)] = __float2bfloat16_rn(d[i]);
    }
}
}
extern "C" int muse_bmma_prepare(const uint16_t* w, uint32_t* payload, int* meta,
        int* bad, int n, int k, cudaStream_t st) {
    if (n!=39936 || k!=6656 || !w || !payload || !meta || !bad) return cudaErrorInvalidValue;
    int blocks=(n/16)*(k/16)/8;
    pack<<<blocks,256,0,st>>>(w,payload,meta,n,k);
    cudaError_t e=cudaGetLastError(); if (e!=cudaSuccess) return e;
    verify<<<blocks,256,0,st>>>(w,payload,meta,n,k,bad);
    return cudaGetLastError();
}
extern "C" int muse_bmma_run(const uint16_t* x, const uint16_t* w, const uint32_t* payload,
        const int* meta, __nv_bfloat16* y, int m, int n, int k, cudaStream_t st) {
    if (m<2 || m>8 || n!=39936 || k!=6656 || !x || !w || !payload || !meta || !y) return cudaErrorInvalidValue;
    project<<<n/128,256,0,st>>>(x,w,payload,meta,y,m,n,k);
    return cudaGetLastError();
}
