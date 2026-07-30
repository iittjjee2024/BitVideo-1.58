// ===========================================================================
//  ternary_gemm.cu -- BitVideo-1.58
//
//  Device-side weight preprocessing:
//    1. absmean reduction (per tensor or per output channel)
//    2. BitNet b1.58 ternarisation with a gamma/2 dead zone
//    3. direct 2-bit packing into any supported inference layout
//    4. packed-layout conversion, unpacking, and distribution statistics
//
//  Preprocessing is intentionally separate from the hot inference kernels.
//  Every destination uint32 is produced by exactly one thread, so there are
//  no atomics in packing and no races even for the fragment-interleaved MMA
//  layout.  Padding codes are written as 00 (zero), making all K/N tails safe
//  for branch-free GEMM reduction loops.
// ===========================================================================
#include "bit_gemm_kernel.h"
#include "bit_utils.h"
#include "cuda_helpers.h"

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <cstdio>

namespace bitvideo {
namespace {

constexpr int kPackThreads = 256;
constexpr int kMaxReductionBlocks = 4096;

template <typename T>
BV_DI float weight_to_float(T v);

template <>
BV_DI float weight_to_float<float>(float v) { return v; }
template <>
BV_DI float weight_to_float<__half>(__half v) { return __half2float(v); }
template <>
BV_DI float weight_to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

/// Inverse of code_addr(): map a destination word/slot back to logical (n,k).
/// This lets one thread construct a whole word without atomics for every
/// layout, including MMA_INTERLEAVED where consecutive K values are permuted.
BV_DI void logical_from_word_slot(BitLayout layout, int64_t word, int slot, int64_t Np,
                                  int64_t Kp, int64_t& n, int64_t& k) {
  const int64_t kw = packed_words(Kp);
  switch (layout) {
    case BV_LAYOUT_ROW_MAJOR: {
      n = word / kw;
      const int64_t kword = word - n * kw;
      k = kword * BV_CODES_PER_U32 + slot;
      break;
    }
    case BV_LAYOUT_COLUMN_MAJOR: {
      const int64_t kword = word / Np;
      n = word - kword * Np;
      k = kword * BV_CODES_PER_U32 + slot;
      break;
    }
    case BV_LAYOUT_BLOCKED_N64: {
      const int64_t nin = word % BV_BLOCK_N;
      const int64_t q = word / BV_BLOCK_N;
      const int64_t kword = q % kw;
      const int64_t nb = q / kw;
      n = nb * BV_BLOCK_N + nin;
      k = kword * BV_CODES_PER_U32 + slot;
      break;
    }
    case BV_LAYOUT_MMA_INTERLEAVED: {
      const int64_t byte_offset = word * 4 + slot / BV_CODES_PER_BYTE;
      const int j = slot % BV_CODES_PER_BYTE;
      const int64_t tile = byte_offset / BV_MMA_TILE_BYTES;
      const int in_tile = static_cast<int>(byte_offset - tile * BV_MMA_TILE_BYTES);
      const int lane = in_tile / BV_MMA_BYTES_PER_LANE;
      const int byte_in_lane = in_tile % BV_MMA_BYTES_PER_LANE;
      const int64_t k_blocks = Kp / BV_MMA_K_BLOCK;
      const int64_t n_tile = tile / k_blocks;
      const int64_t k_block = tile - n_tile * k_blocks;
      const int g = lane >> 2;
      const int t = lane & 3;
      const int sub = byte_in_lane >> 1;
      const int half = byte_in_lane & 1;
      n = n_tile * BV_MMA_N_TILE + g;
      k = k_block * BV_MMA_K_BLOCK + sub * BV_MMA_K_TILE + half * 16 + t * 4 + j;
      break;
    }
    default:
      n = 0;
      k = 0;
      break;
  }
}

BV_HDI bool valid_layout_value(BitLayout layout) {
  return layout >= BV_LAYOUT_ROW_MAJOR && layout <= BV_LAYOUT_MMA_INTERLEAVED;
}

BV_HDI size_t packed_alignment(BitLayout layout) {
  return (layout == BV_LAYOUT_BLOCKED_N64 || layout == BV_LAYOUT_MMA_INTERLEAVED) ? 16u : 4u;
}

bool byte_ranges_overlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
  const uintptr_t ap = reinterpret_cast<uintptr_t>(a);
  const uintptr_t bp = reinterpret_cast<uintptr_t>(b);
  return ap <= bp ? (bp - ap < a_bytes) : (ap - bp < b_bytes);
}

BitStatus validate_layout_shape(BitLayout layout, int n, int k, int np, int kp) {
  if (!valid_layout_value(layout)) return BV_ERR_UNSUPPORTED_LAYOUT;
  if (n < 0 || k < 0 || np < n || kp < k) return BV_ERR_BAD_SHAPE;
  int nm = 1, km = 16;
  layout_alignment(layout, nm, km);
  if (np % nm != 0 || kp % km != 0) return BV_ERR_BAD_SHAPE;
  return BV_OK;
}

inline BitStatus last_launch_status() {
  return cudaGetLastError() == cudaSuccess ? BV_OK : BV_ERR_LAUNCH_FAILED;
}

template <typename T>
__global__ void absmean_per_channel_kernel(const T* __restrict__ w, int n, int k, int ldw,
                                           float eps, float* __restrict__ gamma) {
  const int row = blockIdx.x;
  if (row >= n) return;
  double local = 0.0;
  for (int col = threadIdx.x; col < k; col += blockDim.x)
    local += static_cast<double>(
        fabsf(weight_to_float(w[static_cast<int64_t>(row) * ldw + col])));

  __shared__ double smem[kPackThreads];
  smem[threadIdx.x] = local;
  __syncthreads();
#pragma unroll
  for (int stride = kPackThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    gamma[row] =
        fmaxf(static_cast<float>(smem[0] / static_cast<double>(k)), eps);
}

template <typename T>
__global__ void absmean_tensor_partial_kernel(const T* __restrict__ w, int n, int k, int ldw,
                                              double* __restrict__ partials) {
  const int64_t total = static_cast<int64_t>(n) * k;
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  double local = 0.0;
  for (int64_t idx = first; idx < total; idx += stride) {
    const int row = static_cast<int>(idx / k);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * k);
    local += static_cast<double>(fabsf(weight_to_float(w[static_cast<int64_t>(row) * ldw + col])));
  }

  __shared__ double smem[kPackThreads];
  smem[threadIdx.x] = local;
  __syncthreads();
#pragma unroll
  for (int s = kPackThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = smem[0];
}

__global__ void absmean_finalize_kernel(const double* __restrict__ partials, int blocks,
                                        int64_t count, float eps, float* __restrict__ gamma) {
  double local = 0.0;
  for (int i = threadIdx.x; i < blocks; i += blockDim.x) local += partials[i];
  __shared__ double smem[kPackThreads];
  smem[threadIdx.x] = local;
  __syncthreads();
#pragma unroll
  for (int s = kPackThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    gamma[0] = fmaxf(static_cast<float>(smem[0] / static_cast<double>(count)), eps);
}

template <typename T>
__global__ void ternarize_pack_words_kernel(const T* __restrict__ w, int n, int k, int ldw,
                                            BitLayout layout, BitScaleMode scale_mode,
                                            const float* __restrict__ gamma, int np, int kp,
                                            int64_t words, uint32_t* __restrict__ out) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t word = first; word < words; word += stride) {
    uint32_t packed = 0u;
#pragma unroll
    for (int slot = 0; slot < BV_CODES_PER_U32; ++slot) {
      int64_t row, col;
      logical_from_word_slot(layout, word, slot, np, kp, row, col);
      uint32_t code = BV_CODE_ZERO;
      if (row < n && col < k) {
        const float g = scale_mode == BV_SCALE_PER_CHANNEL ? gamma[row] : gamma[0];
        const float value = weight_to_float(w[row * static_cast<int64_t>(ldw) + col]);
        code = encode_ternary(ternarize_with_threshold(value, 0.5f * g));
      }
      packed |= code << (2 * slot);
    }
    out[word] = packed;
  }
}

__global__ void pack_int8_words_kernel(const int8_t* __restrict__ w, int n, int k, int ldw,
                                       BitLayout layout, int np, int kp, int64_t words,
                                       uint32_t* __restrict__ out) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t word = first; word < words; word += stride) {
    uint32_t packed = 0u;
#pragma unroll
    for (int slot = 0; slot < BV_CODES_PER_U32; ++slot) {
      int64_t row, col;
      logical_from_word_slot(layout, word, slot, np, kp, row, col);
      const int value = (row < n && col < k) ? static_cast<int>(w[row * static_cast<int64_t>(ldw) + col]) : 0;
      packed |= encode_ternary(value) << (2 * slot);
    }
    out[word] = packed;
  }
}

__global__ void unpack_int8_kernel(const uint32_t* __restrict__ packed, BitLayout layout, int n,
                                   int k, int np, int kp, int8_t* __restrict__ out, int ldw) {
  const int64_t total = static_cast<int64_t>(n) * k;
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = first; idx < total; idx += stride) {
    const int row = static_cast<int>(idx / k);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * k);
    out[static_cast<int64_t>(row) * ldw + col] =
        static_cast<int8_t>(decode_ternary(load_code(packed, layout, row, col, np, kp)));
  }
}

__global__ void convert_words_kernel(const uint32_t* __restrict__ src, BitLayout src_layout,
                                     int src_np, int src_kp, uint32_t* __restrict__ dst,
                                     BitLayout dst_layout, int n, int k, int dst_np, int dst_kp,
                                     int64_t dst_words) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t word = first; word < dst_words; word += stride) {
    uint32_t result = 0u;
#pragma unroll
    for (int slot = 0; slot < BV_CODES_PER_U32; ++slot) {
      int64_t row, col;
      logical_from_word_slot(dst_layout, word, slot, dst_np, dst_kp, row, col);
      const uint32_t code = (row < n && col < k)
                                ? load_code(src, src_layout, row, col, src_np, src_kp)
                                : BV_CODE_ZERO;
      result |= code << (2 * slot);
    }
    dst[word] = result;
  }
}

__global__ void packed_stats_kernel(const uint32_t* __restrict__ data, int64_t words,
                                    unsigned long long* __restrict__ counts) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  unsigned long long z = 0, p = 0, m = 0;
  for (int64_t i = first; i < words; i += stride) {
    const uint32_t word = data[i];
#pragma unroll
    for (int j = 0; j < BV_CODES_PER_U32; ++j) {
      const uint32_t c = get_code(word, j);
      p += static_cast<unsigned long long>(c == BV_CODE_POS);
      m += static_cast<unsigned long long>(c == BV_CODE_NEG);
      z += static_cast<unsigned long long>(c == BV_CODE_ZERO || c == BV_CODE_RSVD);
    }
  }
  if (z) atomicAdd(counts + 0, z);
  if (p) atomicAdd(counts + 1, p);
  if (m) atomicAdd(counts + 2, m);
}

inline int bounded_grid(int64_t work) {
  const int64_t blocks = div_ceil64(work, kPackThreads);
  return static_cast<int>(bv_max<int64_t>(1, bv_min<int64_t>(kMaxReductionBlocks, blocks)));
}

template <typename T>
BitStatus launch_absmean(const T* w, int n, int k, int ldw, BitScaleMode mode, float eps,
                         float* gamma, cudaStream_t stream) {
  if (mode == BV_SCALE_PER_CHANNEL) {
    absmean_per_channel_kernel<T><<<n, kPackThreads, 0, stream>>>(w, n, k, ldw, eps, gamma);
    return last_launch_status();
  }
  const int64_t count = static_cast<int64_t>(n) * k;
  const int blocks = bounded_grid(div_ceil64(count, 8));
  double* partials = nullptr;
  cudaError_t e = cudaMallocAsync(reinterpret_cast<void**>(&partials),
                                  static_cast<size_t>(blocks) * sizeof(double), stream);
  if (e != cudaSuccess) return BV_ERR_LAUNCH_FAILED;
  absmean_tensor_partial_kernel<T><<<blocks, kPackThreads, 0, stream>>>(
      w, n, k, ldw, partials);
  if (last_launch_status() != BV_OK) {
    cudaFreeAsync(partials, stream);
    return BV_ERR_LAUNCH_FAILED;
  }
  absmean_finalize_kernel<<<1, kPackThreads, 0, stream>>>(partials, blocks, count, eps, gamma);
  const BitStatus status = last_launch_status();
  e = cudaFreeAsync(partials, stream);
  return status == BV_OK && e == cudaSuccess ? BV_OK : BV_ERR_LAUNCH_FAILED;
}

template <typename T>
BitStatus launch_ternary_pack(const T* w, int n, int k, int ldw, BitLayout layout,
                              BitScaleMode mode, const float* gamma, int np, int kp,
                              uint32_t* out, cudaStream_t stream) {
  const int64_t words = layout_num_words(layout, np, kp);
  ternarize_pack_words_kernel<T><<<bounded_grid(words), kPackThreads, 0, stream>>>(
      w, n, k, ldw, layout, mode, gamma, np, kp, words, out);
  return last_launch_status();
}

}  // namespace

BitStatus bit_ternarize_and_pack(const void* w, BitOutDtype w_dtype, int n, int k, int ldw,
                                 BitLayout layout, BitScaleMode scale_mode, float eps,
                                 uint32_t* w_packed, float* gamma_out, int n_padded,
                                 int k_padded, cudaStream_t stream) {
  if (!w || !w_packed || !gamma_out) return BV_ERR_NULL_POINTER;
  if (n <= 0 || k <= 0 || ldw < k || eps <= 0.0f) return BV_ERR_BAD_SHAPE;
  if (scale_mode != BV_SCALE_PER_TENSOR && scale_mode != BV_SCALE_PER_CHANNEL)
    return BV_ERR_INVALID_ARGUMENT;
  BitStatus s = validate_layout_shape(layout, n, k, n_padded, k_padded);
  if (s != BV_OK) return s;
  if (!is_aligned(w_packed, packed_alignment(layout)) || !is_aligned(gamma_out, 4))
    return BV_ERR_BAD_ALIGNMENT;

#define BV_ABSMEAN_AND_PACK(TYPE)                                                               \
  do {                                                                                           \
    const TYPE* typed = reinterpret_cast<const TYPE*>(w);                                        \
    s = launch_absmean<TYPE>(typed, n, k, ldw, scale_mode, eps, gamma_out, stream);              \
    if (s == BV_OK)                                                                              \
      s = launch_ternary_pack<TYPE>(typed, n, k, ldw, layout, scale_mode, gamma_out, n_padded,   \
                                    k_padded, w_packed, stream);                                  \
  } while (0)

  switch (w_dtype) {
    case BV_OUT_FP32: BV_ABSMEAN_AND_PACK(float); break;
    case BV_OUT_FP16: BV_ABSMEAN_AND_PACK(__half); break;
    case BV_OUT_BF16: BV_ABSMEAN_AND_PACK(__nv_bfloat16); break;
    default: s = BV_ERR_UNSUPPORTED_DTYPE; break;
  }
#undef BV_ABSMEAN_AND_PACK
  return s;
}

BitStatus bit_pack_int8_ternary(const int8_t* w, int n, int k, int ldw, BitLayout layout,
                                uint32_t* w_packed, int n_padded, int k_padded,
                                cudaStream_t stream) {
  if (!w || !w_packed) return BV_ERR_NULL_POINTER;
  if (n <= 0 || k <= 0 || ldw < k) return BV_ERR_BAD_SHAPE;
  BitStatus s = validate_layout_shape(layout, n, k, n_padded, k_padded);
  if (s != BV_OK) return s;
  if (!is_aligned(w_packed, packed_alignment(layout))) return BV_ERR_BAD_ALIGNMENT;
  const int64_t words = layout_num_words(layout, n_padded, k_padded);
  pack_int8_words_kernel<<<bounded_grid(words), kPackThreads, 0, stream>>>(
      w, n, k, ldw, layout, n_padded, k_padded, words, w_packed);
  return last_launch_status();
}

BitStatus bit_unpack_to_int8(const uint32_t* w_packed, BitLayout layout, int n, int k,
                             int n_padded, int k_padded, int8_t* w_out, int ldw,
                             cudaStream_t stream) {
  if (!w_packed || !w_out) return BV_ERR_NULL_POINTER;
  if (n <= 0 || k <= 0 || ldw < k) return BV_ERR_BAD_SHAPE;
  BitStatus s = validate_layout_shape(layout, n, k, n_padded, k_padded);
  if (s != BV_OK) return s;
  if (!is_aligned(w_packed, 4)) return BV_ERR_BAD_ALIGNMENT;
  const int64_t total = static_cast<int64_t>(n) * k;
  unpack_int8_kernel<<<bounded_grid(total), kPackThreads, 0, stream>>>(
      w_packed, layout, n, k, n_padded, k_padded, w_out, ldw);
  return last_launch_status();
}

BitStatus bit_convert_layout(const uint32_t* src, BitLayout src_layout, uint32_t* dst,
                             BitLayout dst_layout, int n, int k, int n_padded_src,
                             int k_padded_src, int n_padded_dst, int k_padded_dst,
                             cudaStream_t stream) {
  if (!src || !dst) return BV_ERR_NULL_POINTER;
  if (n <= 0 || k <= 0) return BV_ERR_BAD_SHAPE;
  BitStatus s = validate_layout_shape(src_layout, n, k, n_padded_src, k_padded_src);
  if (s != BV_OK) return s;
  s = validate_layout_shape(dst_layout, n, k, n_padded_dst, k_padded_dst);
  if (s != BV_OK) return s;
  if (!is_aligned(src, 4) || !is_aligned(dst, packed_alignment(dst_layout)))
    return BV_ERR_BAD_ALIGNMENT;

  const int64_t src_words = layout_num_words(src_layout, n_padded_src, k_padded_src);
  const int64_t dst_words = layout_num_words(dst_layout, n_padded_dst, k_padded_dst);
  if (src == dst && src_layout == dst_layout && n_padded_src == n_padded_dst &&
      k_padded_src == k_padded_dst)
    return BV_OK;
  const size_t src_bytes = static_cast<size_t>(src_words) * sizeof(uint32_t);
  const size_t dst_bytes = static_cast<size_t>(dst_words) * sizeof(uint32_t);
  if (byte_ranges_overlap(src, src_bytes, dst, dst_bytes)) return BV_ERR_INVALID_ARGUMENT;

  convert_words_kernel<<<bounded_grid(dst_words), kPackThreads, 0, stream>>>(
      src, src_layout, n_padded_src, k_padded_src, dst, dst_layout, n, k, n_padded_dst,
      k_padded_dst, dst_words);
  return last_launch_status();
}

BitStatus bit_packed_stats(const uint32_t* w_packed, int64_t num_words, int64_t* counts,
                           cudaStream_t stream) {
  if (!w_packed || !counts) return BV_ERR_NULL_POINTER;
  if (num_words < 0 || num_words > INT64_MAX / BV_CODES_PER_U32) return BV_ERR_BAD_SHAPE;
  if (!is_aligned(w_packed, 4) || !is_aligned(counts, alignof(int64_t)))
    return BV_ERR_BAD_ALIGNMENT;
  cudaError_t e = cudaMemsetAsync(counts, 0, 3 * sizeof(int64_t), stream);
  if (e != cudaSuccess) return BV_ERR_LAUNCH_FAILED;
  if (num_words == 0) return BV_OK;
  packed_stats_kernel<<<bounded_grid(num_words), kPackThreads, 0, stream>>>(
      w_packed, num_words, reinterpret_cast<unsigned long long*>(counts));
  return last_launch_status();
}

}  // namespace bitvideo
