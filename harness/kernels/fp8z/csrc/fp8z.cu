// SPDX-License-Identifier: Apache-2.0
// fp8z: the CUTLASS-exact FP8 w8a8 GEMM of 1..8 rows (decode) for any N % 32 == 0,
// K % 128 == 0, reading a lossless 7-bit coding of the weight bytes.
//
// Derived from fp8c (harness/kernels/fp8c of our nemotron engine, MIT): the same
// code, layout and register decoder, generalized from two compile-time shapes to
// runtime N and K (more than 32 chunks per tile: the per-(chunk, warp) exception
// offsets are read from the tile header per chunk instead of from one lane each).
//
// Arithmetic (as vLLM's sm120 CUTLASS ScaledEpilogue, per-tensor scales): every
// output is one chain of native e4m3 mma.sync m16n8k32 over K in ascending order
// from +0 (CUTLASS does not split K at these sizes), then fp32 (acc * scale_b),
// then bf16_rn(that * scale_a). The decoder rebuilds the exact B-fragment words
// (verified for every byte at load time) so the MMA operands are the stock ones.
//
// Code (per output column): an 8-exponent window [e0, e0+7]. A byte s|eeee|mmm
// with e in the window is the 7-bit code s|(e-e0)|mmm; one code (ESC, the least
// frequent one of the column) marks an escape whose original byte is stored raw
// in the tile's exception area. One region per 32-column tile (= one CTA), in
// the order the CTA consumes it:
//   [header: u16 offsets per (chunk, warp) + final total; u16 e0|esc<<8 per column at HDR-64]
//   [C chunks x 3584 B: planes A (4 x u32/lane), B (2 x u32/lane), D (1 x u32/lane)]
//   [exception bytes, ordered (chunk, warp, lane, word, byte)]
// A chunk is 4 K-steps (128 K). Per lane it holds the 8 B-fragment words
// R0..R7 (R[2s+h] = step s, half h) as 7 words: byte j of word i<7 is
// code(R_i byte j) | bit i of code(R_7 byte j) << 7.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace fp8z {
constexpr unsigned FULL = 0xffffffffu;
constexpr int MAIN = 3584;  // bytes per (tile, chunk)
__host__ __device__ __forceinline__ int hdr_bytes(int c) { return (((4 * c + 1) * 2 + 64) + 127) / 128 * 128; }
struct P {const uint8_t *a,*w;const float *sa,*sb;__nv_bfloat16 *c;int m,n,k,tile_bytes;};
struct Slot {uint32_t c[7];uint32_t x;uint32_t q0;};

__device__ __forceinline__ uint4 ld4(const uint8_t* p) {
 uint4 v;
 asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
  : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p));
 return v;
}
__device__ __forceinline__ uint2 ld2(const uint8_t* p) {
 uint2 v;
 asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x),"=r"(v.y) : "l"(p));
 return v;
}
__device__ __forceinline__ uint32_t ld1(const uint8_t* p) {
 uint32_t v;
 asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
 return v;
}
__device__ __forceinline__ uint32_t bfi8(uint32_t b, uint32_t w, uint32_t pos) {
 uint32_t d;
 asm("bfi.b32 %0, %1, %2, %3, 8;\n" : "=r"(d) : "r"(b), "r"(w), "r"(pos));
 return d;
}
// Main planes of chunk c (no dependence on the header).
__device__ __forceinline__ void load_main(const uint8_t* tb,int hdr,int warp,int lane,int c,Slot& s) {
 const uint8_t* mp=tb+hdr+c*MAIN;
 uint4 A=ld4(mp+warp*512+lane*16);
 uint2 B=ld2(mp+2048+warp*256+lane*8);
 uint32_t D=ld1(mp+3072+warp*128+lane*4);
 s.c[0]=A.x;s.c[1]=A.y;s.c[2]=A.z;s.c[3]=A.w;s.c[4]=B.x;s.c[5]=B.y;s.c[6]=D;
}
// Exception window of chunk c: lane l holds bytes [wb+4l, wb+4l+4) of the tile's
// exception area, wb = off & ~3, for the words that hold this warp-chunk's bytes.
// off/end: this warp-chunk's escape range, from the header (a broadcast load).
__device__ __forceinline__ void load_x(const uint8_t* tb,int exc,int warp,int lane,int c,Slot& s) {
 const uint16_t* tab=(const uint16_t*)tb;
 uint32_t off=__ldg(tab+c*4+warp),end=__ldg(tab+c*4+warp+1);
 uint32_t q0=off&3u;
 s.q0=q0;
 s.x=(4u*lane<q0+end-off)?ld1(tb+exc+(off-q0)+4*lane):0u;
}
__device__ __forceinline__ uint32_t lanemask_lt() {
 uint32_t m;
 asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
 return m;
}
// Rebuild the 8 B-fragment words of this lane for one chunk (fp8c's decoder).
// Escapes of a warp-chunk are stored lane by lane, each lane's in (word, byte)
// order; q = window-relative index of this lane's first one (exclusive prefix
// of the per-lane counts, from bit-plane ballots). Fast path: two shuffled window
// words hold this lane's bytes q..q+7-(q&3), enough unless (q&3)+cnt > 8 in some
// lane (rare: then a shuffle per escape).
__device__ __forceinline__ void decode(const Slot& s,uint32_t e0x,uint32_t esc4,int lane,uint32_t (&w)[8]) {
 uint32_t cw[8],em[8],pc[8],hi=0,cnt=0;
 #pragma unroll
 for(int i=0;i<7;++i){cw[i]=s.c[i];hi|=(s.c[i]>>(7-i))&(0x01010101u<<i);}
 cw[7]=hi;
 #pragma unroll
 for(int i=0;i<8;++i){
  uint32_t x=(cw[i]^esc4)&0x7f7f7f7fu;  // bytes <= 0x7f: x+0x7f carries into bit 7 iff x != 0
  em[i]=~(x+0x7f7f7f7fu)&0x80808080u;
  pc[i]=__popc(em[i]);
  cnt+=pc[i];
  // code s|e'|mmm (bits 6..0) -> s|e'+e0|mmm; e'+e0 <= 15, so no byte carries.
  w[i]=((cw[i]&0x3f3f3f3fu)+e0x)|((cw[i]<<1)&0x80808080u);
 }
 uint32_t lt=lanemask_lt();
 uint32_t pre=(uint32_t)__popc(__ballot_sync(FULL,cnt&1u)&lt)
  +((uint32_t)__popc(__ballot_sync(FULL,cnt&2u)&lt)<<1)
  +((uint32_t)__popc(__ballot_sync(FULL,cnt&4u)&lt)<<2);
 uint32_t q=s.q0+pre;  // exact when no lane has more than 7 escapes
 uint32_t v0=__shfl_sync(FULL,s.x,(q>>2)&31u),v1=__shfl_sync(FULL,s.x,((q>>2)+1u)&31u);
 uint32_t sel=(q&3u)*0x1111u,em2[8],sl[8],two=0;
 #pragma unroll
 for(int i=0;i<8;++i){
  em2[i]=em[i]&(em[i]-1u);
  two|=em2[i];
  uint32_t fm=((em[i]^em2[i])>>7)*0xffu;
  w[i]=(w[i]&~fm)|(__byte_perm(v0,v1,sel)&fm);
  sl[i]=sel+0x1111u;
  sel+=pc[i]*0x1111u;
 }
 bool slow=cnt>7u||(q&3u)+cnt>8u;
 if(__any_sync(FULL,two)){
  uint32_t three=0;
  #pragma unroll
  for(int i=0;i<8;++i){
   three|=em2[i]&(em2[i]-1u);
   uint32_t fm=(em2[i]>>7)*0xffu;
   w[i]=(w[i]&~fm)|(__byte_perm(v0,v1,sl[i])&fm);
  }
  slow|=three!=0u;
 }
 if(__any_sync(FULL,slow)){
  pre=0;
  #pragma unroll
  for(int b=0;b<6;++b)pre+=(uint32_t)__popc(__ballot_sync(FULL,(cnt>>b)&1u)&lt)<<b;
  q=s.q0+pre;
  #pragma unroll
  for(int i=0;i<8;++i){
   uint32_t mk=em[i];
   while(__any_sync(FULL,mk)){
    uint32_t v=__shfl_sync(FULL,s.x,(q>>2)&31u);
    if(mk){
     w[i]=bfi8(__byte_perm(v,0u,q&3u),w[i],(uint32_t)__ffs(mk)-8u);
     mk&=mk-1u;++q;
    }
   }
  }
 }
}
// Activation words (rows 0..7) of the 4 steps of chunk c; rows 8..15 are zero for M<=8.
__device__ __forceinline__ void load_a(const P&p,int lane,int c,uint32_t (&a)[4][2]) {
 int row=lane>>2;
 #pragma unroll
 for(int s=0;s<4;++s){
  int kk=(c*4+s)*32+(lane&3)*4;
  a[s][0]=row<p.m?__ldg((const uint32_t*)(p.a+(size_t)row*p.k+kk)):0u;
  a[s][1]=row<p.m?__ldg((const uint32_t*)(p.a+(size_t)row*p.k+kk+16)):0u;
 }
}
__device__ __forceinline__ void mma(uint32_t a0,uint32_t a2,uint32_t b0,uint32_t b1,float(&c)[4]) {
 asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
 :"=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3])
 :"r"(a0),"r"(0u),"r"(a2),"r"(0u),"r"(b0),"r"(b1),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}
// RC chunks (4*RC K-steps) in flight per warp; MINB = minimum resident CTAs
// per SM asked of the register allocator (more CTAs per SM, fewer registers);
// WPC = warps per CTA (4: one CTA per 32-column tile; 2 or 1: a tile's warps
// in 2 or 4 CTAs, for finer load balance). Each warp's work is the same.
template<int RC,int MINB,int WPC>
__global__ __launch_bounds__(32*WPC,MINB) void gemm(P p) {
 const int C=p.k>>7,HDR=hdr_bytes(C),EXC=HDR+C*MAIN;
 constexpr int CPT=4/WPC;  // CTAs per tile
 const int tile=blockIdx.x/CPT;
 int lane=threadIdx.x%32,warp=(blockIdx.x%CPT)*WPC+threadIdx.x/32;
 const uint8_t* tb=p.w+(size_t)tile*p.tile_bytes;
 uint32_t prm=__ldg((const uint16_t*)(tb+HDR-64)+warp*8+(lane>>2));  // this lane's column
 uint32_t e0x=((prm&0xffu)<<3)*0x01010101u,esc4=((prm>>8)&0xffu)*0x01010101u;
 Slot s[RC];uint32_t a[RC][4][2];float acc[4]={0.f,0.f,0.f,0.f};
 #pragma unroll
 for(int j=0;j<RC;++j)if(j<C)load_main(tb,HDR,warp,lane,j,s[j]);
 #pragma unroll
 for(int j=0;j<RC;++j)if(j<C)load_x(tb,EXC,warp,lane,j,s[j]);
 #pragma unroll
 for(int j=0;j<RC;++j)if(j<C)load_a(p,lane,j,a[j]);
 // Software pipeline: wc holds chunk c decoded; the MMAs of chunk c are issued
 // before the decode of chunk c+1, so the two latency chains overlap. Slot
 // (c mod RC) of the ring is refilled with chunk c+RC right after its decode.
 uint32_t wc[8];
 decode(s[0],e0x,esc4,lane,wc);
 if(RC<C){load_main(tb,HDR,warp,lane,RC,s[0]);load_x(tb,EXC,warp,lane,RC,s[0]);}
 for(int cb=0;cb<C;cb+=RC){
  #pragma unroll
  for(int j=0;j<RC;++j){
   int c=cb+j;
   if(c<C){
    #pragma unroll
    for(int st=0;st<4;++st)mma(a[j][st][0],a[j][st][1],wc[2*st],wc[2*st+1],acc);
    if(c+RC<C)load_a(p,lane,c+RC,a[j]);
    if(c+1<C){
     uint32_t wn[8];
     decode(s[(j+1)%RC],e0x,esc4,lane,wn);
     if(c+1+RC<C){
      load_main(tb,HDR,warp,lane,c+1+RC,s[(j+1)%RC]);
      load_x(tb,EXC,warp,lane,c+1+RC,s[(j+1)%RC]);
     }
     #pragma unroll
     for(int i=0;i<8;++i)wc[i]=wn[i];
    }
   }
  }
 }
 float sa=__ldg(p.sa),sb=__ldg(p.sb);int row=lane>>2,col=(tile*4+warp)*8+(lane&3)*2;
 if(row<p.m){
  p.c[(size_t)row*p.n+col]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(acc[0],sb),sa));
  p.c[(size_t)row*p.n+col+1]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(acc[1],sb),sa));
 }
}
// Load-time proof: the same loads and decoder, storing the rebuilt words back in
// the original weight layout W[N][K] (row = output column) instead of an MMA.
__global__ __launch_bounds__(128) void decode_kernel(const uint8_t* buf,int tile_bytes,int k,uint32_t* out) {
 const int C=k>>7,HDR=hdr_bytes(C),EXC=HDR+C*MAIN;
 int lane=threadIdx.x%32,warp=threadIdx.x/32;
 const uint8_t* tb=buf+(size_t)blockIdx.x*tile_bytes;
 uint32_t prm=__ldg((const uint16_t*)(tb+HDR-64)+warp*8+(lane>>2));
 uint32_t e0x=((prm&0xffu)<<3)*0x01010101u,esc4=((prm>>8)&0xffu)*0x01010101u;
 size_t col=(size_t)blockIdx.x*32+warp*8+(lane>>2);
 for(int c=0;c<C;++c){
  Slot s;uint32_t w[8];
  load_main(tb,HDR,warp,lane,c,s);
  load_x(tb,EXC,warp,lane,c,s);
  decode(s,e0x,esc4,lane,w);
  #pragma unroll
  for(int st=0;st<4;++st){
   size_t kk=(size_t)(c*4+st)*32+(lane&3)*4;
   out[(col*k+kk)/4]=w[2*st];
   out[(col*k+kk+16)/4]=w[2*st+1];
  }
 }
}
template<int RC,int MINB,int WPC=4>
cudaError_t launch(const P&p,cudaStream_t stream) {
 gemm<RC,MINB,WPC><<<p.n/32*(4/WPC),32*WPC,0,stream>>>(p);
 return cudaGetLastError();
}
}  // namespace fp8z

// Variant <ring depth in chunks, minimum CTAs per SM, warps per CTA>: 2 = <2,4,4>,
// 3 = <3,2,4>, 33 = <3,4,4>, 4 = <4,2,4>, 6 = <6,2,4>, 32 = <3,8,2>, 31 = <3,16,1>,
// 21 = <2,16,1>.
extern "C" int fp8z_run(const void*a,const void*w,const void*sa,const void*sb,void*c,int m,int n,int k,
                        int tile_bytes,int ring,void*stream) {
 using namespace fp8z;
 if(m<1||m>8||n<=0||n%32||k<=0||k%128)return (int)cudaErrorInvalidValue;
 P p{(const uint8_t*)a,(const uint8_t*)w,(const float*)sa,(const float*)sb,(__nv_bfloat16*)c,m,n,k,tile_bytes};
 cudaStream_t st=(cudaStream_t)stream;
 switch(ring){
  case 2: return (int)launch<2,4>(p,st);
  case 3: return (int)launch<3,2>(p,st);
  case 33: return (int)launch<3,4>(p,st);
  case 4: return (int)launch<4,2>(p,st);
  case 6: return (int)launch<6,2>(p,st);
  case 32: return (int)launch<3,8,2>(p,st);
  case 31: return (int)launch<3,16,1>(p,st);
  case 21: return (int)launch<2,16,1>(p,st);
  default: return (int)cudaErrorInvalidValue;
 }
}
extern "C" int fp8z_decode(const void*buf,int tile_bytes,void*out,int n,int k,void*stream) {
 using namespace fp8z;
 if(n<=0||n%32||k<=0||k%128)return (int)cudaErrorInvalidValue;
 decode_kernel<<<n/32,128,0,(cudaStream_t)stream>>>((const uint8_t*)buf,tile_bytes,k,(uint32_t*)out);
 return (int)cudaGetLastError();
}
extern "C" int fp8z_header_bytes(int k) { return fp8z::hdr_bytes(k / 128); }
extern "C" int fp8z_init() {
 cudaFuncAttributes attr;cudaError_t rc;
 rc=cudaFuncGetAttributes(&attr,fp8z::gemm<2,4,4>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8z::gemm<3,2,4>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8z::gemm<3,4,4>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8z::gemm<4,2,4>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8z::gemm<6,2,4>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8z::gemm<3,8,2>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8z::gemm<3,16,1>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8z::gemm<2,16,1>);if(rc!=cudaSuccess)return (int)rc;
 return (int)cudaFuncGetAttributes(&attr,fp8z::decode_kernel);
}
