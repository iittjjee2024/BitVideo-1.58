// ===========================================================================
//  bit_packing.cpp -- CPU reference/offline packer for BitVideo-1.58
// ===========================================================================
#include "bit_packing.h"

#include "bit_utils.h"
#include "cuda_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace bitvideo {
namespace {

bool valid_layout(BitLayout layout) {
  return layout >= BV_LAYOUT_ROW_MAJOR && layout <= BV_LAYOUT_MMA_INTERLEAVED;
}

size_t required_alignment(BitLayout layout) {
  return (layout == BV_LAYOUT_BLOCKED_N64 || layout == BV_LAYOUT_MMA_INTERLEAVED) ? 16u : 4u;
}

bool checked_word_bytes(int64_t words, size_t* bytes) {
  if (!bytes || words < 0 ||
      static_cast<uint64_t>(words) >
          std::numeric_limits<size_t>::max() / sizeof(uint32_t))
    return false;
  *bytes = static_cast<size_t>(words) * sizeof(uint32_t);
  return true;
}

BitStatus validate_shape(BitLayout layout, int n, int k, int np, int kp) {
  if (!valid_layout(layout)) return BV_ERR_UNSUPPORTED_LAYOUT;
  if (n <= 0 || k <= 0 || np < n || kp < k) return BV_ERR_BAD_SHAPE;
  int nm = 1, km = 16;
  layout_alignment(layout, nm, km);
  return (np % nm == 0 && kp % km == 0) ? BV_OK : BV_ERR_BAD_SHAPE;
}

bool ranges_overlap(const void* a, size_t an, const void* b, size_t bn) {
  const uintptr_t ap = reinterpret_cast<uintptr_t>(a);
  const uintptr_t bp = reinterpret_cast<uintptr_t>(b);
  return ap <= bp ? bp - ap < an : ap - bp < bn;
}

}  // namespace

BitStatus bit_host_padded_extents(BitLayout layout, int n, int k, int* n_padded,
                                  int* k_padded) {
  if (!n_padded || !k_padded) return BV_ERR_NULL_POINTER;
  if (!valid_layout(layout)) return BV_ERR_UNSUPPORTED_LAYOUT;
  if (n <= 0 || k <= 0) return BV_ERR_BAD_SHAPE;
  int nm = 1, km = 16;
  layout_alignment(layout, nm, km);
  const int64_t np = div_ceil64(static_cast<int64_t>(n), nm) * nm;
  const int64_t kp = div_ceil64(static_cast<int64_t>(k), km) * km;
  if (np > std::numeric_limits<int>::max() ||
      kp > std::numeric_limits<int>::max())
    return BV_ERR_BAD_SHAPE;
  *n_padded = static_cast<int>(np);
  *k_padded = static_cast<int>(kp);
  return BV_OK;
}

BitStatus bit_pack_host_f32(const float* w, int n, int k, int ldw, BitLayout layout,
                            BitScaleMode scale_mode, float eps, uint32_t* packed,
                            float* gamma, int n_padded, int k_padded) {
  if (!w || !packed || !gamma) return BV_ERR_NULL_POINTER;
  if (ldw < k || !std::isfinite(eps) || eps <= 0.0f) return BV_ERR_BAD_SHAPE;
  if (scale_mode != BV_SCALE_PER_TENSOR && scale_mode != BV_SCALE_PER_CHANNEL)
    return BV_ERR_INVALID_ARGUMENT;
  BitStatus status = validate_shape(layout, n, k, n_padded, k_padded);
  if (status != BV_OK) return status;
  if (!is_aligned(packed, required_alignment(layout)) || !is_aligned(gamma, 4))
    return BV_ERR_BAD_ALIGNMENT;

  if (scale_mode == BV_SCALE_PER_CHANNEL) {
    for (int row = 0; row < n; ++row) {
      double sum = 0.0;
      for (int col = 0; col < k; ++col)
        sum += std::abs(static_cast<double>(w[static_cast<int64_t>(row) * ldw + col]));
      gamma[row] = std::fmax(static_cast<float>(sum / static_cast<double>(k)), eps);
    }
  } else {
    double sum = 0.0;
    for (int row = 0; row < n; ++row)
      for (int col = 0; col < k; ++col)
        sum += std::abs(static_cast<double>(w[static_cast<int64_t>(row) * ldw + col]));
    gamma[0] = std::fmax(
        static_cast<float>(sum / static_cast<double>(static_cast<int64_t>(n) * k)), eps);
  }

  const int64_t words = layout_num_words(layout, n_padded, k_padded);
  size_t packed_bytes = 0;
  if (!checked_word_bytes(words, &packed_bytes)) return BV_ERR_BAD_SHAPE;
  std::memset(packed, 0, packed_bytes);

  for (int row = 0; row < n; ++row) {
    const float threshold = 0.5f *
        (scale_mode == BV_SCALE_PER_CHANNEL ? gamma[row] : gamma[0]);
    for (int col = 0; col < k; ++col) {
      const float value = w[static_cast<int64_t>(row) * ldw + col];
      const uint32_t code = encode_ternary(ternarize_with_threshold(value, threshold));
      store_code(packed, layout, row, col, n_padded, k_padded, code);
    }
  }
  return BV_OK;
}

BitStatus bit_pack_host_int8(const int8_t* w, int n, int k, int ldw, BitLayout layout,
                             uint32_t* packed, int n_padded, int k_padded) {
  if (!w || !packed) return BV_ERR_NULL_POINTER;
  if (ldw < k) return BV_ERR_BAD_SHAPE;
  BitStatus status = validate_shape(layout, n, k, n_padded, k_padded);
  if (status != BV_OK) return status;
  if (!is_aligned(packed, required_alignment(layout))) return BV_ERR_BAD_ALIGNMENT;
  const int64_t words = layout_num_words(layout, n_padded, k_padded);
  size_t packed_bytes = 0;
  if (!checked_word_bytes(words, &packed_bytes)) return BV_ERR_BAD_SHAPE;
  std::memset(packed, 0, packed_bytes);
  for (int row = 0; row < n; ++row)
    for (int col = 0; col < k; ++col)
      store_code(packed, layout, row, col, n_padded, k_padded,
                 encode_ternary(w[static_cast<int64_t>(row) * ldw + col]));
  return BV_OK;
}

BitStatus bit_unpack_host_int8(const uint32_t* packed, BitLayout layout, int n, int k,
                               int n_padded, int k_padded, int8_t* out, int ldout) {
  if (!packed || !out) return BV_ERR_NULL_POINTER;
  if (ldout < k) return BV_ERR_BAD_SHAPE;
  BitStatus status = validate_shape(layout, n, k, n_padded, k_padded);
  if (status != BV_OK) return status;
  if (!is_aligned(packed, 4)) return BV_ERR_BAD_ALIGNMENT;
  for (int row = 0; row < n; ++row)
    for (int col = 0; col < k; ++col)
      out[static_cast<int64_t>(row) * ldout + col] = static_cast<int8_t>(
          decode_ternary(load_code(packed, layout, row, col, n_padded, k_padded)));
  return BV_OK;
}

BitStatus bit_convert_layout_host(const uint32_t* src, BitLayout src_layout,
                                  int src_n_padded, int src_k_padded, uint32_t* dst,
                                  BitLayout dst_layout, int dst_n_padded, int dst_k_padded,
                                  int n, int k) {
  if (!src || !dst) return BV_ERR_NULL_POINTER;
  BitStatus status = validate_shape(src_layout, n, k, src_n_padded, src_k_padded);
  if (status != BV_OK) return status;
  status = validate_shape(dst_layout, n, k, dst_n_padded, dst_k_padded);
  if (status != BV_OK) return status;
  if (!is_aligned(src, 4) || !is_aligned(dst, required_alignment(dst_layout)))
    return BV_ERR_BAD_ALIGNMENT;

  const int64_t src_words = layout_num_words(src_layout, src_n_padded, src_k_padded);
  const int64_t dst_words = layout_num_words(dst_layout, dst_n_padded, dst_k_padded);
  size_t src_bytes = 0, dst_bytes = 0;
  if (!checked_word_bytes(src_words, &src_bytes) ||
      !checked_word_bytes(dst_words, &dst_bytes))
    return BV_ERR_BAD_SHAPE;
  if (src == dst && src_layout == dst_layout && src_n_padded == dst_n_padded &&
      src_k_padded == dst_k_padded)
    return BV_OK;
  if (ranges_overlap(src, src_bytes, dst, dst_bytes)) return BV_ERR_INVALID_ARGUMENT;
  std::memset(dst, 0, dst_bytes);

  for (int row = 0; row < n; ++row)
    for (int col = 0; col < k; ++col)
      store_code(dst, dst_layout, row, col, dst_n_padded, dst_k_padded,
                 load_code(src, src_layout, row, col, src_n_padded, src_k_padded));
  return BV_OK;
}

}  // namespace bitvideo
