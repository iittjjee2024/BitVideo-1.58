// ===========================================================================
//  cuda_helpers.h -- BitVideo-1.58
//
//  Architecture feature detection, error handling, and the low level device
//  primitives shared by every kernel in this extension:
//
//    * CUDA_CHECK / CUDA_CHECK_LAST        loud, actionable diagnostics
//    * arch capability macros              cp.async, ldmatrix, mma.s8, DP4A
//    * 128-bit vector load / store         LDG.128 / STS.128 / LDS.128
//    * cp.async wrappers                   ca / cg variants + commit/wait
//    * warp primitives                     shuffle reductions, ballot helpers
//    * DP4A wrappers                       signed/unsigned 4-way INT8 dot
//    * XOR swizzle                         bank-conflict-free shared layouts
//    * numeric conversion                  int32 -> fp16 / bf16 / fp32
//
//  Everything is header-only and `__forceinline__` so nvcc can fold the
//  address arithmetic into the memory instructions.
//
//  References
//  ----------
//  [1] CUDA C++ Programming Guide, ch. 7.27 (Asynchronous Data Copies)
//  [2] PTX ISA 8.5, ch. 9.7.9 (Warp Level Matrix Multiply-Accumulate)
//  [3] CUDA C++ Best Practices Guide, ch. 9 (Memory Optimizations)
//  [4] Nsight Compute Kernel Profiling Guide (stall reason taxonomy)
// ===========================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>

// ---------------------------------------------------------------------------
//  Toolkit sanity check
// ---------------------------------------------------------------------------
#if defined(CUDART_VERSION) && (CUDART_VERSION < 12000)
#error "BitVideo-1.58 requires CUDA 12.0 or newer."
#endif

// ---------------------------------------------------------------------------
//  Host/device decorators
// ---------------------------------------------------------------------------
#define BV_HD          __host__ __device__
#define BV_DI          __device__ __forceinline__
#define BV_HDI         __host__ __device__ __forceinline__
#define BV_GLOBAL      __global__

// ---------------------------------------------------------------------------
//  Architecture capability macros
//
//  These are evaluated in DEVICE compilation passes only (__CUDA_ARCH__ is
//  undefined on the host pass).  Host code must query the runtime instead --
//  see bit_gemm_launch.cu :: bitvideo_device_info().
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__)
#  define BV_ARCH __CUDA_ARCH__
#else
#  define BV_ARCH 0
#endif

// DP4A: 4-way INT8 dot product with INT32 accumulate.  sm_61+.
#define BV_HAS_DP4A            (BV_ARCH >= 610)
// cp.async (LDGSTS): global -> shared bypassing registers.  sm_80+.
#define BV_HAS_CP_ASYNC        (BV_ARCH >= 800)
// ldmatrix: cooperative shared -> register matrix fragment load.  sm_75+.
#define BV_HAS_LDMATRIX        (BV_ARCH >= 750)
// mma.sync m16n8k32 with s8 operands and s32 accumulate.  sm_80+.
#define BV_HAS_MMA_S8_K32      (BV_ARCH >= 800)
// Asynchronous barriers (mbarrier) for multi-stage pipelines.  sm_80+.
#define BV_HAS_MBARRIER        (BV_ARCH >= 800)
// Distributed shared memory / cluster launch.  sm_90+.
#define BV_HAS_CLUSTER         (BV_ARCH >= 900)
// 5th-gen tensor cores (tcgen05.mma).  DATACENTER Blackwell only (sm_100).
// Consumer Blackwell (sm_120, RTX 50xx) does NOT expose tcgen05 -- it keeps
// the warp-level mma.sync issue model.  Confirmed by CUTLASS #2800 / #3044.
#define BV_HAS_TCGEN05         (BV_ARCH >= 1000 && BV_ARCH < 1200)

// Shared memory capacity we are willing to opt into per block, by arch.
// (Hardware maxima: sm_80 163KB, sm_86/89 99KB, sm_90 227KB, sm_120 99KB.)
#if   BV_ARCH >= 1200
#  define BV_MAX_DYN_SMEM  (99u * 1024u)
#elif BV_ARCH >= 1000
#  define BV_MAX_DYN_SMEM  (227u * 1024u)
#elif BV_ARCH >= 900
#  define BV_MAX_DYN_SMEM  (227u * 1024u)
#elif BV_ARCH >= 860
#  define BV_MAX_DYN_SMEM  (99u * 1024u)
#elif BV_ARCH >= 800
#  define BV_MAX_DYN_SMEM  (163u * 1024u)
#else
#  define BV_MAX_DYN_SMEM  (48u * 1024u)
#endif

#define BV_WARP_SIZE       32
#define BV_FULL_MASK       0xffffffffu

// ---------------------------------------------------------------------------
//  Error handling
// ---------------------------------------------------------------------------
namespace bitvideo {

/// Thrown by CUDA_CHECK when BITVIDEO_THROW_ON_CUDA_ERROR is enabled (default).
class CudaError : public std::runtime_error {
 public:
  CudaError(cudaError_t code, const char* expr, const char* file, int line)
      : std::runtime_error(build(code, expr, file, line)), code_(code) {}
  cudaError_t code() const noexcept { return code_; }

 private:
  static std::string build(cudaError_t code, const char* expr, const char* file, int line) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "[bitvideo][CUDA] %s (%d: %s)\n"
                  "  expression : %s\n"
                  "  location   : %s:%d\n"
                  "  hint       : run with CUDA_LAUNCH_BLOCKING=1 and "
                  "compute-sanitizer for an exact fault site.",
                  cudaGetErrorName(code), static_cast<int>(code), cudaGetErrorString(code), expr,
                  file, line);
    return std::string(buf);
  }
  cudaError_t code_;
};

inline void cuda_check_impl(cudaError_t code, const char* expr, const char* file, int line) {
  if (code != cudaSuccess) {
#if defined(BITVIDEO_NO_EXCEPTIONS)
    std::fprintf(stderr, "[bitvideo][CUDA][FATAL] %s at %s:%d -> %s\n", expr, file, line,
                 cudaGetErrorString(code));
    std::abort();
#else
    throw CudaError(code, expr, file, line);
#endif
  }
}

}  // namespace bitvideo

/// Wrap every CUDA runtime call.  Never silently ignore a status code.
#define CUDA_CHECK(expr) ::bitvideo::cuda_check_impl((expr), #expr, __FILE__, __LINE__)

/// Drain the sticky error slot after a kernel launch.
#define CUDA_CHECK_LAST() \
  ::bitvideo::cuda_check_impl(cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__)

/// Synchronising variant, only active in debug builds (it serialises the GPU).
#if defined(BITVIDEO_DEBUG)
#  define CUDA_CHECK_SYNC()                                                     \
    do {                                                                        \
      CUDA_CHECK_LAST();                                                        \
      ::bitvideo::cuda_check_impl(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", __FILE__, \
                                  __LINE__);                                    \
    } while (0)
#else
#  define CUDA_CHECK_SYNC() CUDA_CHECK_LAST()
#endif

/// Device-side assertion that compiles away in release builds.
#if defined(BITVIDEO_DEBUG)
#  define BV_DEV_ASSERT(cond, msg)                                                 \
    do {                                                                           \
      if (!(cond)) {                                                               \
        std::printf("[bitvideo][assert] %s  (block %d thread %d)  %s:%d\n", (msg), \
                    blockIdx.x, threadIdx.x, __FILE__, __LINE__);                  \
        __trap();                                                                  \
      }                                                                            \
    } while (0)
#else
#  define BV_DEV_ASSERT(cond, msg) ((void)0)
#endif

namespace bitvideo {

// ---------------------------------------------------------------------------
//  Integer utilities (constexpr, usable in template arguments)
// ---------------------------------------------------------------------------
BV_HDI constexpr int div_ceil(int a, int b) { return a / b + static_cast<int>(a % b != 0); }
BV_HDI constexpr int64_t div_ceil64(int64_t a, int64_t b) {
  return a / b + static_cast<int64_t>(a % b != 0);
}
BV_HDI constexpr int round_up(int a, int b) { return div_ceil(a, b) * b; }
BV_HDI constexpr bool is_pow2(int x) { return x > 0 && (x & (x - 1)) == 0; }
BV_HDI constexpr int log2_ct(int x) { return x <= 1 ? 0 : 1 + log2_ct(x >> 1); }

template <typename T>
BV_HDI constexpr T bv_min(T a, T b) { return a < b ? a : b; }
template <typename T>
BV_HDI constexpr T bv_max(T a, T b) { return a > b ? a : b; }

// ---------------------------------------------------------------------------
//  128-bit vector types and aligned access
//
//  Rationale: a single LDG.128 moves 16B per thread; a full warp therefore
//  moves 512B == 4 sectors of 128B, which is the widest transaction the L1/L2
//  path can service in one go.  Using anything narrower for the weight stream
//  leaves DRAM bandwidth on the table -- and the ternary weight stream *is*
//  the bottleneck for batch-1 decode.
// ---------------------------------------------------------------------------
struct alignas(16) Vec128 {
  uint32_t x, y, z, w;
};
static_assert(sizeof(Vec128) == 16, "Vec128 must be 16 bytes");

/// Non-coherent 128-bit global load (compiles to LDG.E.128.CONSTANT).
BV_DI Vec128 ldg128(const void* __restrict__ p) {
#if defined(__CUDA_ARCH__)
  Vec128 r;
  const uint4 v = __ldg(reinterpret_cast<const uint4*>(p));
  r.x = v.x; r.y = v.y; r.z = v.z; r.w = v.w;
  return r;
#else
  Vec128 r;
  std::memcpy(&r, p, sizeof(r));
  return r;
#endif
}

/// Streaming 128-bit global load that bypasses L1 (`ld.global.nc.L2::...`).
/// Useful for the weight stream, which has zero reuse in the GEMV kernel.
BV_DI uint4 ldg128_stream(const void* __restrict__ p) {
#if defined(__CUDA_ARCH__) && BV_ARCH >= 800
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p)
               : "memory");
  return r;
#elif defined(__CUDA_ARCH__)
  return __ldg(reinterpret_cast<const uint4*>(p));
#else
  uint4 r; std::memcpy(&r, p, sizeof(r)); return r;
#endif
}

BV_DI void st128(void* __restrict__ p, const uint4& v) {
  *reinterpret_cast<uint4*>(p) = v;
}

BV_DI uint4 lds128(const void* p) { return *reinterpret_cast<const uint4*>(p); }

BV_HDI bool is_aligned(const void* p, size_t bytes) {
  return (reinterpret_cast<uintptr_t>(p) & (bytes - 1)) == 0;
}

// ---------------------------------------------------------------------------
//  Generic -> shared address conversion (needed by cp.async / ldmatrix)
// ---------------------------------------------------------------------------
BV_DI uint32_t smem_addr(const void* p) {
#if defined(__CUDA_ARCH__)
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
#else
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
#endif
}

// ---------------------------------------------------------------------------
//  cp.async  (Ampere "LDGSTS": global -> shared, no register round trip)
//
//  ".ca" caches in L1 (use when the tile is re-read by other blocks),
//  ".cg" bypasses L1 and only allocates in L2 (use for streaming tiles).
//  Only 4/8/16-byte transfers are legal; 16B is what we always use.
//
//  `src_bytes` implements the zero-fill predication trick: when the K tail is
//  ragged we pass src_bytes=0 and the hardware writes zeros instead of
//  reading out of bounds.  Zeros are the identity for our accumulation, so no
//  epilogue masking is required -- this removes every boundary branch from
//  the main loop.
// ---------------------------------------------------------------------------
template <int Bytes = 16, bool BypassL1 = true>
BV_DI void cp_async(void* smem_dst, const void* gmem_src, bool predicate = true) {
  static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp.async supports 4/8/16 bytes");
#if BV_HAS_CP_ASYNC
  const uint32_t dst = smem_addr(smem_dst);
  const int src_bytes = predicate ? Bytes : 0;
  if (BypassL1 && Bytes == 16) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(dst), "l"(gmem_src),
                 "n"(Bytes), "r"(src_bytes));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(dst), "l"(gmem_src),
                 "n"(Bytes), "r"(src_bytes));
  }
#else
  // Pre-Ampere / host fallback: synchronous copy with explicit zero fill.
  if (predicate) {
    if (Bytes == 16) {
      *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
    } else if (Bytes == 8) {
      *reinterpret_cast<uint2*>(smem_dst) = *reinterpret_cast<const uint2*>(gmem_src);
    } else {
      *reinterpret_cast<uint32_t*>(smem_dst) = *reinterpret_cast<const uint32_t*>(gmem_src);
    }
  } else {
    if (Bytes == 16) {
      *reinterpret_cast<uint4*>(smem_dst) = make_uint4(0u, 0u, 0u, 0u);
    } else if (Bytes == 8) {
      *reinterpret_cast<uint2*>(smem_dst) = make_uint2(0u, 0u);
    } else {
      *reinterpret_cast<uint32_t*>(smem_dst) = 0u;
    }
  }
#endif
}

BV_DI void cp_async_commit() {
#if BV_HAS_CP_ASYNC
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

/// Wait until at most `N` cp.async groups are still in flight.
template <int N>
BV_DI void cp_async_wait_group() {
#if BV_HAS_CP_ASYNC
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

BV_DI void cp_async_wait_all() {
#if BV_HAS_CP_ASYNC
  asm volatile("cp.async.wait_all;\n" ::);
#endif
}

/// Fence that keeps the compiler from hoisting shared reads above the wait.
BV_DI void smem_fence() {
#if defined(__CUDA_ARCH__)
  asm volatile("" ::: "memory");
#endif
}

// ---------------------------------------------------------------------------
//  ldmatrix : cooperative 8x8x16bit shared -> register fragment load
//  (used by the MMA path to feed A fragments without bank conflicts)
// ---------------------------------------------------------------------------
BV_DI void ldmatrix_x4(uint32_t (&d)[4], const void* smem_ptr) {
#if BV_HAS_LDMATRIX
  const uint32_t a = smem_addr(smem_ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a));
#else
  const uint4 v = lds128(smem_ptr);
  d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
#endif
}

BV_DI void ldmatrix_x2(uint32_t (&d)[2], const void* smem_ptr) {
#if BV_HAS_LDMATRIX
  const uint32_t a = smem_addr(smem_ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(a));
#else
  const uint2 v = *reinterpret_cast<const uint2*>(smem_ptr);
  d[0] = v.x; d[1] = v.y;
#endif
}

// ---------------------------------------------------------------------------
//  DP4A : d = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w + c   (INT8 x INT8 -> INT32)
//
//  One instruction, four MACs, issued on the INT32 pipe.  Available since
//  sm_61 and *not* deprecated on Blackwell.  For ternary weights the operand
//  `a` is produced by the PRMT decode in bit_utils.h, so the whole 4-tap dot
//  product costs ~6 ALU ops + 1 DP4A.
// ---------------------------------------------------------------------------
BV_DI int32_t dp4a(uint32_t a, uint32_t b, int32_t c) {
#if BV_HAS_DP4A
  int32_t d;
  asm volatile("dp4a.s32.s32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
#else
  const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
  const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
  int32_t d = c;
  for (int i = 0; i < 4; ++i) d += static_cast<int32_t>(pa[i]) * static_cast<int32_t>(pb[i]);
  return d;
#endif
}

/// Unsigned-by-signed variant (used when one side is a 0/1 selection mask).
BV_DI int32_t dp4a_us(uint32_t a_unsigned, uint32_t b_signed, int32_t c) {
#if BV_HAS_DP4A
  int32_t d;
  asm volatile("dp4a.u32.s32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a_unsigned), "r"(b_signed), "r"(c));
  return d;
#else
  const uint8_t* pa = reinterpret_cast<const uint8_t*>(&a_unsigned);
  const int8_t* pb = reinterpret_cast<const int8_t*>(&b_signed);
  int32_t d = c;
  for (int i = 0; i < 4; ++i) d += static_cast<int32_t>(pa[i]) * static_cast<int32_t>(pb[i]);
  return d;
#endif
}

// ---------------------------------------------------------------------------
//  mma.sync wrappers  (INT8 tensor cores, sm_80 .. sm_120)
//
//  D[16x8](s32) = A[16xK](s8) * B[Kx8](s8) + C[16x8](s32)
//
//  Fragment -> lane mapping, with  g = lane >> 2  and  t = lane & 3
//  (PTX ISA 8.5, tables "Multiplicand A ... .m16n8k32"):
//
//    A (row major, 16xK):  4 regs for K=32, 2 regs for K=16
//        reg0 -> row g    , cols 4t   .. 4t+3
//        reg1 -> row g+8  , cols 4t   .. 4t+3
//        reg2 -> row g    , cols 4t+16.. 4t+19     (K=32 only)
//        reg3 -> row g+8  , cols 4t+16.. 4t+19     (K=32 only)
//    B (col major, Kx8):   2 regs for K=32, 1 reg for K=16
//        reg0 -> col g    , rows 4t   .. 4t+3
//        reg1 -> col g    , rows 4t+16.. 4t+19     (K=32 only)
//    C/D (16x8 s32):       4 regs
//        reg0 -> (g  , 2t) reg1 -> (g  , 2t+1)
//        reg2 -> (g+8, 2t) reg3 -> (g+8, 2t+1)
// ---------------------------------------------------------------------------
BV_DI void mma_m16n8k32_s8(int32_t (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
                           const int32_t (&c)[4]) {
#if BV_HAS_MMA_S8_K32
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]),
        "r"(c[2]), "r"(c[3]));
#elif defined(__CUDA_ARCH__)
  // Scalar emulation keeps pre-sm80 device builds testable.
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  int32_t acc[4] = {c[0], c[1], c[2], c[3]};
  for (int kk = 0; kk < 32; ++kk) {
    const int areg = ((kk >> 4) << 1);
    const int abyte = (kk & 15) - 4 * t;
    if (abyte < 0 || abyte > 3) continue;
    const int8_t a_lo = reinterpret_cast<const int8_t*>(&a[areg])[abyte];
    const int8_t a_hi = reinterpret_cast<const int8_t*>(&a[areg + 1])[abyte];
    const int8_t bv = reinterpret_cast<const int8_t*>(&b[kk >> 4])[abyte];
    acc[0] += static_cast<int32_t>(a_lo) * static_cast<int32_t>(bv);
    acc[2] += static_cast<int32_t>(a_hi) * static_cast<int32_t>(bv);
  }
  d[0] = acc[0]; d[1] = acc[1]; d[2] = acc[2]; d[3] = acc[3];
  (void)g;
#else
  d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
  (void)a; (void)b;
#endif
}

// ---------------------------------------------------------------------------
//  Warp level reductions
//
//  __shfl_xor_sync with a butterfly pattern gives every lane the result in
//  log2(32)=5 steps with no shared memory and no __syncthreads().  The
//  `_down` variant is one instruction cheaper per step when only lane 0 needs
//  the answer, but the butterfly avoids a broadcast afterwards.
// ---------------------------------------------------------------------------
template <int Width = BV_WARP_SIZE>
BV_DI int32_t warp_reduce_sum_i32(int32_t v, uint32_t mask = BV_FULL_MASK) {
#if defined(__CUDA_ARCH__)
#pragma unroll
  for (int off = Width >> 1; off > 0; off >>= 1) v += __shfl_xor_sync(mask, v, off, Width);
#else
  (void)mask;
#endif
  return v;
}

template <int Width = BV_WARP_SIZE>
BV_DI float warp_reduce_sum_f32(float v, uint32_t mask = BV_FULL_MASK) {
#if defined(__CUDA_ARCH__)
#pragma unroll
  for (int off = Width >> 1; off > 0; off >>= 1) v += __shfl_xor_sync(mask, v, off, Width);
#else
  (void)mask;
#endif
  return v;
}

template <int Width = BV_WARP_SIZE>
BV_DI float warp_reduce_max_f32(float v, uint32_t mask = BV_FULL_MASK) {
#if defined(__CUDA_ARCH__)
#pragma unroll
  for (int off = Width >> 1; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(mask, v, off, Width));
#else
  (void)mask;
#endif
  return v;
}

template <int Width = BV_WARP_SIZE>
BV_DI int32_t warp_reduce_max_i32(int32_t v, uint32_t mask = BV_FULL_MASK) {
#if defined(__CUDA_ARCH__)
#pragma unroll
  for (int off = Width >> 1; off > 0; off >>= 1)
    v = bv_max(v, __shfl_xor_sync(mask, v, off, Width));
#else
  (void)mask;
#endif
  return v;
}

/// Reduce across a whole block using one shared slot per warp.
/// `smem` must hold at least `blockDim.x / 32` elements.
template <typename T>
BV_DI T block_reduce_sum(T v, T* smem) {
#if defined(__CUDA_ARCH__)
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(BV_FULL_MASK, v, off, 32);
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  T total = T(0);
  if (threadIdx.x < 32) {
    T x = (threadIdx.x < static_cast<unsigned>(nwarps)) ? smem[threadIdx.x] : T(0);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) x += __shfl_xor_sync(BV_FULL_MASK, x, off, 32);
    total = x;
    if (threadIdx.x == 0) smem[0] = total;
  }
  __syncthreads();
  return smem[0];
#else
  (void)smem;
  return v;
#endif
}

// ---------------------------------------------------------------------------
//  Shared memory XOR swizzle
//
//  Shared memory has 32 banks of 4B.  A row-major tile whose row pitch is a
//  multiple of 32 words makes every thread in a warp hit the same bank when
//  striding down a column -> 32-way conflict.  Two standard cures:
//
//    (a) padding      pitch += 1 word.  Simple, wastes a little smem, breaks
//                     128-bit alignment (so it cannot be combined with
//                     LDS.128 / ldmatrix).
//    (b) XOR swizzle  permute the *offset* inside a 128B row so that column
//                     walks spread across banks while keeping every 16B
//                     chunk 16B-aligned.  This is what CUTLASS does and what
//                     we use on the MMA path.
//
//  swizzle<3,3,3>() == CUTLASS Swizzle<3,3,3>: xor bits [6:4] with [9:7],
//  i.e. permute eight 16B chunks within a 128B row.
// ---------------------------------------------------------------------------
template <int BBits, int MBase, int SShift>
BV_HDI int swizzle_offset(int offset) {
  constexpr int bit_msk = ((1 << BBits) - 1) << (MBase + SShift);
  return offset ^ ((offset & bit_msk) >> SShift);
}

/// Pad a row pitch (in 4B words) so that consecutive rows land on different
/// bank groups while remaining 16B aligned (pitch stays a multiple of 4).
BV_HDI constexpr int pad_pitch_words(int words) {
  // Multiples of 32 words are the pathological case; shift by 4 words (16B)
  // which keeps uint4 alignment and rotates the bank group by 4.
  return (words % 32 == 0) ? words + 4 : round_up(words, 4);
}

// ---------------------------------------------------------------------------
//  Output conversion helpers
// ---------------------------------------------------------------------------
template <typename T>
struct OutTraits;

template <>
struct OutTraits<float> {
  BV_DI static float from_float(float v) { return v; }
  BV_DI static float zero() { return 0.f; }
};
template <>
struct OutTraits<__half> {
  BV_DI static __half from_float(float v) { return __float2half_rn(v); }
  BV_DI static __half zero() { return __float2half_rn(0.f); }
};
template <>
struct OutTraits<__nv_bfloat16> {
  BV_DI static __nv_bfloat16 from_float(float v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __float2bfloat16_rn(v);
#else
    return __float2bfloat16(v);
#endif
  }
  BV_DI static __nv_bfloat16 zero() { return __float2bfloat16(0.f); }
};

// ---------------------------------------------------------------------------
//  Grid swizzling (threadblock rasterisation order)
//
//  Launching a 2D grid in linear order makes all concurrently resident CTAs
//  read the same rows of A and stream disjoint columns of B -> poor L2 reuse
//  and, on multi-partition memory systems, partition camping.  Remapping the
//  linear block id into column-major groups of `GroupN` gives each wave of
//  CTAs a compact 2D footprint, which is the single cheapest L2 hit-rate
//  optimisation available (CUTLASS calls this "threadblock swizzle").
// ---------------------------------------------------------------------------
BV_DI void grid_swizzle(int linear_id, int tiles_m, int tiles_n, int group_m, int& tile_m,
                        int& tile_n) {
  const int blocks_per_group = group_m * tiles_n;
  const int group_id = linear_id / blocks_per_group;
  const int first_m = group_id * group_m;
  const int group_size_m = bv_min(tiles_m - first_m, group_m);
  const int idx_in_group = linear_id - group_id * blocks_per_group;
  tile_m = first_m + (idx_in_group % group_size_m);
  tile_n = idx_in_group / group_size_m;
}

// ---------------------------------------------------------------------------
//  Host-side timing helper (used by the C++ benchmark and the gtest suite)
// ---------------------------------------------------------------------------
class CudaTimer {
 public:
  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  CudaTimer(const CudaTimer&) = delete;
  CudaTimer& operator=(const CudaTimer&) = delete;

  void start(cudaStream_t s = nullptr) { CUDA_CHECK(cudaEventRecord(start_, s)); }
  float stop(cudaStream_t s = nullptr) {
    CUDA_CHECK(cudaEventRecord(stop_, s));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_{}, stop_{};
};

}  // namespace bitvideo
