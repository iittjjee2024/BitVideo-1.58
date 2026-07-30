// ===========================================================================
//  int8_gemm.cu -- BitVideo-1.58
//
//  Dynamic activation quantisation and the dense INT8xINT8 fallback GEMM.
//
//  Quantisation modes:
//      0  per tensor             scale[1]
//      1  per token / row        scale[M]       (BitNet default)
//      2  per 128-element group  scale[M, ceil(K/128)]
//
//  GEMM computes exact INT32 accumulation using DP4A.  A 32x32 CTA tile is
//  split into 2x2 register microtiles (one per thread), while A and B use a
//  two-stage cp.async pipeline through shared memory.  This path handles the
//  quality-critical INT8 K/V projections and serves as an independent oracle
//  for the ternary kernel tests.
// ===========================================================================
#include "bit_gemm_kernel.h"
#include "bit_utils.h"
#include "cuda_helpers.h"

#include <cuda_runtime.h>

#include <climits>
#include <cfloat>
#include <cmath>
#include <cstdint>

namespace bitvideo {
namespace {

constexpr int kQuantThreads = 256;
constexpr int kQuantGroup = 128;
constexpr int kMaxQuantBlocks = 4096;

BV_DI float sanitize_quant_input(float value) {
  if (isnan(value)) return 0.0f;
  if (isinf(value)) return copysignf(FLT_MAX, value);
  return value;
}

template <typename T>
BV_DI float in_to_float(T v);
template <>
BV_DI float in_to_float<float>(float v) { return sanitize_quant_input(v); }
template <>
BV_DI float in_to_float<__half>(__half v) {
  return sanitize_quant_input(__half2float(v));
}
template <>
BV_DI float in_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return sanitize_quant_input(__bfloat162float(v));
}

BV_DI float out_to_float(float v) { return v; }
BV_DI float out_to_float(__half v) { return __half2float(v); }
BV_DI float out_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

inline BitStatus launch_result() {
  return cudaGetLastError() == cudaSuccess ? BV_OK : BV_ERR_LAUNCH_FAILED;
}

inline int quant_grid(int64_t work) {
  const int64_t blocks = div_ceil64(work, kQuantThreads);
  return static_cast<int>(bv_max<int64_t>(1, bv_min<int64_t>(kMaxQuantBlocks, blocks)));
}

template <typename T>
__global__ void quantize_rows_kernel(const T* __restrict__ x, int m, int k, int ldx, float eps,
                                     int8_t* __restrict__ q, int ldq,
                                     float* __restrict__ scales) {
  const int row = blockIdx.x;
  if (row >= m) return;
  float local = 0.0f;
  for (int col = threadIdx.x; col < k; col += blockDim.x)
    local = fmaxf(local, fabsf(in_to_float(x[static_cast<int64_t>(row) * ldx + col])));

  __shared__ float smem[kQuantThreads];
  smem[threadIdx.x] = local;
  __syncthreads();
#pragma unroll
  for (int s = kQuantThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();
  }

  const float scale = fmaxf(smem[0], eps) / 127.0f;
  const float inv = 1.0f / scale;
  if (threadIdx.x == 0) scales[row] = scale;
  for (int col = threadIdx.x; col < k; col += blockDim.x)
    q[static_cast<int64_t>(row) * ldq + col] =
        quantize_int8_rn(in_to_float(x[static_cast<int64_t>(row) * ldx + col]), inv);
}

template <typename T>
__global__ void quantize_groups_kernel(const T* __restrict__ x, int m, int k, int ldx, float eps,
                                       int groups, int8_t* __restrict__ q, int ldq,
                                       float* __restrict__ scales) {
  const int group_linear = blockIdx.x;
  const int row = group_linear / groups;
  const int group = group_linear - row * groups;
  if (row >= m) return;
  const int begin = group * kQuantGroup;
  const int end = bv_min(k, begin + kQuantGroup);

  float local = 0.0f;
  for (int col = begin + threadIdx.x; col < end; col += blockDim.x)
    local = fmaxf(local, fabsf(in_to_float(x[static_cast<int64_t>(row) * ldx + col])));

  __shared__ float smem[kQuantThreads];
  smem[threadIdx.x] = local;
  __syncthreads();
#pragma unroll
  for (int s = kQuantThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();
  }

  const float scale = fmaxf(smem[0], eps) / 127.0f;
  const float inv = 1.0f / scale;
  if (threadIdx.x == 0) scales[static_cast<int64_t>(row) * groups + group] = scale;
  for (int col = begin + threadIdx.x; col < end; col += blockDim.x)
    q[static_cast<int64_t>(row) * ldq + col] =
        quantize_int8_rn(in_to_float(x[static_cast<int64_t>(row) * ldx + col]), inv);
}

template <typename T>
__global__ void tensor_absmax_partial_kernel(const T* __restrict__ x, int m, int k, int ldx,
                                             float* __restrict__ maximum) {
  const int64_t total = static_cast<int64_t>(m) * k;
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  float local = 0.0f;
  for (int64_t idx = first; idx < total; idx += stride) {
    const int row = static_cast<int>(idx / k);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * k);
    local = fmaxf(local, fabsf(in_to_float(x[static_cast<int64_t>(row) * ldx + col])));
  }

  __shared__ float smem[kQuantThreads];
  smem[threadIdx.x] = local;
  __syncthreads();
#pragma unroll
  for (int s = kQuantThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicMax(reinterpret_cast<int*>(maximum), __float_as_int(smem[0]));
}

__global__ void tensor_scale_finalize_kernel(float* scale, float eps) {
  if (threadIdx.x == 0) scale[0] = fmaxf(scale[0], eps) / 127.0f;
}

template <typename T>
__global__ void quantize_tensor_kernel(const T* __restrict__ x, int m, int k, int ldx,
                                       const float* __restrict__ scale, int8_t* __restrict__ q,
                                       int ldq) {
  const int64_t total = static_cast<int64_t>(m) * k;
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const float inv = 1.0f / scale[0];
  for (int64_t idx = first; idx < total; idx += stride) {
    const int row = static_cast<int>(idx / k);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * k);
    q[static_cast<int64_t>(row) * ldq + col] =
        quantize_int8_rn(in_to_float(x[static_cast<int64_t>(row) * ldx + col]), inv);
  }
}

template <typename T>
BitStatus launch_quantize(const T* x, int m, int k, int ldx, int granularity, float eps,
                          int8_t* q, int ldq, float* scales, cudaStream_t stream) {
  if (granularity == 1) {
    quantize_rows_kernel<T><<<m, kQuantThreads, 0, stream>>>(x, m, k, ldx, eps, q, ldq, scales);
    return launch_result();
  }
  if (granularity == 2) {
    const int groups = div_ceil(k, kQuantGroup);
    quantize_groups_kernel<T><<<m * groups, kQuantThreads, 0, stream>>>(
        x, m, k, ldx, eps, groups, q, ldq, scales);
    return launch_result();
  }

  cudaError_t e = cudaMemsetAsync(scales, 0, sizeof(float), stream);
  if (e != cudaSuccess) return BV_ERR_LAUNCH_FAILED;
  const int64_t total = static_cast<int64_t>(m) * k;
  const int blocks = quant_grid(div_ceil64(total, 8));
  tensor_absmax_partial_kernel<T><<<blocks, kQuantThreads, 0, stream>>>(x, m, k, ldx, scales);
  if (launch_result() != BV_OK) return BV_ERR_LAUNCH_FAILED;
  tensor_scale_finalize_kernel<<<1, 1, 0, stream>>>(scales, eps);
  if (launch_result() != BV_OK) return BV_ERR_LAUNCH_FAILED;
  quantize_tensor_kernel<T><<<quant_grid(total), kQuantThreads, 0, stream>>>(
      x, m, k, ldx, scales, q, ldq);
  return launch_result();
}

// ---------------------------------------------------------------------------
//  Dense INT8 GEMM
// ---------------------------------------------------------------------------
constexpr int kInt8BM = 32;
constexpr int kInt8BN = 32;
constexpr int kInt8BK = 128;
constexpr int kInt8Stages = 2;
constexpr int kInt8Pitch = kInt8BK + 16;
constexpr int kInt8StageBytes = kInt8BM * kInt8Pitch;

BV_DI void wait_int8_oldest(int pending) {
  if (pending >= 2) cp_async_wait_group<1>();
  else cp_async_wait_group<0>();
  smem_fence();
}

BV_DI void issue_int8_stage(const int8_t* __restrict__ a, const int8_t* __restrict__ b,
                            int m, int n, int k, int lda, int ldb, bool b_row_major,
                            int tile_m, int tile_n, int k_tile, int stage,
                            uint8_t* __restrict__ sm_a, uint8_t* __restrict__ sm_b) {
  uint8_t* as = sm_a + stage * kInt8StageBytes;
  uint8_t* bs = sm_b + stage * kInt8StageBytes;
  const int m0 = tile_m * kInt8BM;
  const int n0 = tile_n * kInt8BN;
  const int k0 = k_tile * kInt8BK;
  constexpr int vecs = kInt8BM * (kInt8BK / 16);

  for (int v = threadIdx.x; v < vecs; v += blockDim.x) {
    const int row = v / (kInt8BK / 16);
    const int kk = (v - row * (kInt8BK / 16)) * 16;

    // A row-major [M,K].
    uint8_t* adst = as + row * kInt8Pitch + kk;
    const int gm = m0 + row;
    const int gk = k0 + kk;
    const bool afull = gm < m && gk + 16 <= k;
    const int8_t* asrc = afull ? a + static_cast<int64_t>(gm) * lda + gk : a;
    if (afull && is_aligned(asrc, 16)) {
      cp_async<16, true>(adst, asrc, true);
    } else {
      uint4 value = make_uint4(0u, 0u, 0u, 0u);
      if (gm < m && gk < k) {
        uint32_t words[4] = {0u, 0u, 0u, 0u};
        const int rem = bv_min(16, k - gk);
        const int8_t* src = a + static_cast<int64_t>(gm) * lda + gk;
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (j < rem)
            words[j >> 2] |= static_cast<uint32_t>(static_cast<uint8_t>(src[j])) << (8 * (j & 3));
        value = make_uint4(words[0], words[1], words[2], words[3]);
      }
      *reinterpret_cast<uint4*>(adst) = value;
    }

    // B is staged as [BN,BK], regardless of its external orientation.
    uint8_t* bdst = bs + row * kInt8Pitch + kk;
    const int gn = n0 + row;
    if (b_row_major) {
      const bool bfull = gn < n && gk + 16 <= k;
      const int8_t* bsrc = bfull ? b + static_cast<int64_t>(gn) * ldb + gk : b;
      if (bfull && is_aligned(bsrc, 16)) {
        cp_async<16, true>(bdst, bsrc, true);
      } else {
        uint4 value = make_uint4(0u, 0u, 0u, 0u);
        if (gn < n && gk < k) {
          uint32_t words[4] = {0u, 0u, 0u, 0u};
          const int rem = bv_min(16, k - gk);
          const int8_t* src = b + static_cast<int64_t>(gn) * ldb + gk;
#pragma unroll
          for (int j = 0; j < 16; ++j)
            if (j < rem)
              words[j >> 2] |= static_cast<uint32_t>(static_cast<uint8_t>(src[j])) << (8 * (j & 3));
          value = make_uint4(words[0], words[1], words[2], words[3]);
        }
        *reinterpret_cast<uint4*>(bdst) = value;
      }
    } else {
      // External B is [K,N] row-major: transpose the 16-element vector while
      // staging.  This path is naturally scalar; the hot BitLinear path uses
      // b_row_major=true and therefore stays on cp.async.
      uint32_t words[4] = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int gkj = gk + j;
        const int8_t value = (gn < n && gkj < k) ? b[static_cast<int64_t>(gkj) * ldb + gn] : 0;
        words[j >> 2] |= static_cast<uint32_t>(static_cast<uint8_t>(value)) << (8 * (j & 3));
      }
      *reinterpret_cast<uint4*>(bdst) = make_uint4(words[0], words[1], words[2], words[3]);
    }
  }
  cp_async_commit();
}

template <typename OutT>
__launch_bounds__(256, 2) __global__
void int8_dp4a_gemm_kernel(const int8_t* __restrict__ a, const int8_t* __restrict__ b,
                           int m, int n, int k, int lda, int ldb,
                           const float* __restrict__ a_scale, BitActScaleMode a_scale_mode,
                           const float* __restrict__ b_scale, BitScaleMode b_scale_mode,
                           const float* __restrict__ bias, float alpha, float beta,
                           OutT* __restrict__ c, int ldc, bool b_row_major,
                           int tiles_m, int tiles_n) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint8_t* sm_a = smem;
  uint8_t* sm_b = smem + kInt8Stages * kInt8StageBytes;

  int tile_m, tile_n;
  grid_swizzle(static_cast<int>(blockIdx.x), tiles_m, tiles_n, 8, tile_m, tile_n);

  const int tid = threadIdx.x;
  const int lr0 = tid >> 4;       // 0..15
  const int lc0 = tid & 15;       // 0..15
  int32_t acc00 = 0, acc01 = 0, acc10 = 0, acc11 = 0;

  const int k_tiles = div_ceil(k, kInt8BK);
  const int preload = bv_min(kInt8Stages, k_tiles);
  for (int s = 0; s < preload; ++s)
    issue_int8_stage(a, b, m, n, k, lda, ldb, b_row_major, tile_m, tile_n, s, s,
                     sm_a, sm_b);
  wait_int8_oldest(preload);
  __syncthreads();

  for (int kt = 0; kt < k_tiles; ++kt) {
    const int stage = kt & 1;
    const uint8_t* as = sm_a + stage * kInt8StageBytes;
    const uint8_t* bs = sm_b + stage * kInt8StageBytes;
#pragma unroll
    for (int q = 0; q < kInt8BK / 4; ++q) {
      const uint32_t ar0 = *reinterpret_cast<const uint32_t*>(as + lr0 * kInt8Pitch + q * 4);
      const uint32_t ar1 = *reinterpret_cast<const uint32_t*>(as + (lr0 + 16) * kInt8Pitch + q * 4);
      const uint32_t br0 = *reinterpret_cast<const uint32_t*>(bs + lc0 * kInt8Pitch + q * 4);
      const uint32_t br1 = *reinterpret_cast<const uint32_t*>(bs + (lc0 + 16) * kInt8Pitch + q * 4);
      acc00 = dp4a(ar0, br0, acc00);
      acc01 = dp4a(ar0, br1, acc01);
      acc10 = dp4a(ar1, br0, acc10);
      acc11 = dp4a(ar1, br1, acc11);
    }

    __syncthreads();
    const int future = kt + kInt8Stages;
    if (future < k_tiles)
      issue_int8_stage(a, b, m, n, k, lda, ldb, b_row_major, tile_m, tile_n, future,
                       stage, sm_a, sm_b);
    const int remaining = k_tiles - kt - 1;
    if (remaining > 0) {
      wait_int8_oldest(bv_min(kInt8Stages, remaining));
      __syncthreads();
    }
  }

  const int rows[2] = {tile_m * kInt8BM + lr0, tile_m * kInt8BM + lr0 + 16};
  const int cols[2] = {tile_n * kInt8BN + lc0, tile_n * kInt8BN + lc0 + 16};
  const int32_t values[2][2] = {{acc00, acc01}, {acc10, acc11}};
#pragma unroll
  for (int ri = 0; ri < 2; ++ri) {
    if (rows[ri] >= m) continue;
    const float sa = a_scale
                         ? (a_scale_mode == BV_ACT_SCALE_PER_TOKEN ? a_scale[rows[ri]] : a_scale[0])
                         : 1.0f;
#pragma unroll
    for (int ci = 0; ci < 2; ++ci) {
      if (cols[ci] >= n) continue;
      const float sb = b_scale
                           ? (b_scale_mode == BV_SCALE_PER_CHANNEL ? b_scale[cols[ci]] : b_scale[0])
                           : 1.0f;
      const float bi = bias ? bias[cols[ci]] : 0.0f;
      OutT* dst = c + static_cast<int64_t>(rows[ri]) * ldc + cols[ci];
      float value = fmaf(alpha * sa * sb, static_cast<float>(values[ri][ci]), bi);
      if (beta != 0.0f) value = fmaf(beta, out_to_float(*dst), value);
      *dst = OutTraits<OutT>::from_float(value);
    }
  }
}

template <typename OutT>
BitStatus launch_int8_gemm_typed(const int8_t* a, const int8_t* b, int m, int n, int k,
                                 int lda, int ldb, const float* a_scale,
                                 BitActScaleMode a_scale_mode, const float* b_scale,
                                 BitScaleMode b_scale_mode, const float* bias, float alpha,
                                 float beta, void* c, int ldc, bool b_is_nk,
                                 cudaStream_t stream) {
  constexpr int smem = 2 * kInt8Stages * kInt8StageBytes;
  auto kernel = int8_dp4a_gemm_kernel<OutT>;
  if (smem > 48 * 1024) {
    const cudaError_t e = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return BV_ERR_LAUNCH_FAILED;
  }
  const int tm = div_ceil(m, kInt8BM);
  const int tn = div_ceil(n, kInt8BN);
  kernel<<<tm * tn, 256, smem, stream>>>(
      a, b, m, n, k, lda, ldb, a_scale, a_scale_mode, b_scale, b_scale_mode, bias, alpha,
      beta, reinterpret_cast<OutT*>(c), ldc, b_is_nk, tm, tn);
  return launch_result();
}

}  // namespace

BitStatus bit_quantize_activations(const void* x, BitOutDtype x_dtype, int m, int k, int ldx,
                                   int granularity, float eps, int8_t* q_out, int ldq,
                                   float* scale_out, cudaStream_t stream) {
  if (!x || !q_out || !scale_out) return BV_ERR_NULL_POINTER;
  if (m <= 0 || k <= 0 || ldx < k || ldq < k || eps <= 0.0f) return BV_ERR_BAD_SHAPE;
  if (granularity < 0 || granularity > 2) return BV_ERR_INVALID_ARGUMENT;
  if (granularity == 2 &&
      static_cast<int64_t>(m) * div_ceil(k, kQuantGroup) > INT32_MAX)
    return BV_ERR_BAD_SHAPE;
  switch (x_dtype) {
    case BV_OUT_FP32:
      return launch_quantize(reinterpret_cast<const float*>(x), m, k, ldx, granularity, eps,
                             q_out, ldq, scale_out, stream);
    case BV_OUT_FP16:
      return launch_quantize(reinterpret_cast<const __half*>(x), m, k, ldx, granularity, eps,
                             q_out, ldq, scale_out, stream);
    case BV_OUT_BF16:
      return launch_quantize(reinterpret_cast<const __nv_bfloat16*>(x), m, k, ldx, granularity,
                             eps, q_out, ldq, scale_out, stream);
    default: return BV_ERR_UNSUPPORTED_DTYPE;
  }
}

BitStatus bit_int8_gemm(const int8_t* a, const int8_t* b, int m, int n, int k, int lda,
                        int ldb, const float* a_scale, BitActScaleMode a_scale_mode,
                        const float* b_scale, BitScaleMode b_scale_mode, const float* bias,
                        float alpha, float beta, void* c, BitOutDtype c_dtype, int ldc,
                        bool b_is_nk, cudaStream_t stream) {
  if (!a || !b || !c) return BV_ERR_NULL_POINTER;
  if (m <= 0 || n <= 0 || k <= 0 || lda < k || ldc < n) return BV_ERR_BAD_SHAPE;
  // Worst-case signed INT8 product is (-128)*(-128) == 16384.
  if (k > INT32_MAX / 16384) return BV_ERR_BAD_SHAPE;
  if ((b_is_nk && ldb < k) || (!b_is_nk && ldb < n)) return BV_ERR_BAD_SHAPE;
  if (a_scale_mode != BV_ACT_SCALE_PER_TENSOR && a_scale_mode != BV_ACT_SCALE_PER_TOKEN)
    return BV_ERR_INVALID_ARGUMENT;
  if (b_scale_mode != BV_SCALE_PER_TENSOR && b_scale_mode != BV_SCALE_PER_CHANNEL)
    return BV_ERR_INVALID_ARGUMENT;
  const int64_t tm = div_ceil(m, kInt8BM);
  const int64_t tn = div_ceil(n, kInt8BN);
  if (tm * tn > INT32_MAX) return BV_ERR_BAD_SHAPE;
  switch (c_dtype) {
    case BV_OUT_FP32:
      return launch_int8_gemm_typed<float>(a, b, m, n, k, lda, ldb, a_scale, a_scale_mode,
                                           b_scale, b_scale_mode, bias, alpha, beta, c, ldc,
                                           b_is_nk, stream);
    case BV_OUT_FP16:
      return launch_int8_gemm_typed<__half>(a, b, m, n, k, lda, ldb, a_scale, a_scale_mode,
                                            b_scale, b_scale_mode, bias, alpha, beta, c, ldc,
                                            b_is_nk, stream);
    case BV_OUT_BF16:
      return launch_int8_gemm_typed<__nv_bfloat16>(
          a, b, m, n, k, lda, ldb, a_scale, a_scale_mode, b_scale, b_scale_mode, bias, alpha,
          beta, c, ldc, b_is_nk, stream);
    default: return BV_ERR_UNSUPPORTED_DTYPE;
  }
}

}  // namespace bitvideo
