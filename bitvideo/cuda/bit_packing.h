// ===========================================================================
//  bit_packing.h -- CPU reference/offline packer for BitVideo-1.58
//
//  This interface has no PyTorch dependency.  It is used by the ATen bridge,
//  GoogleTest, model-conversion tools, and non-CUDA deployment runtimes.
// ===========================================================================
#pragma once

#include "bit_gemm_kernel.h"

#include <cstdint>

namespace bitvideo {

/// Compute the minimum padded extents required by a packed layout.
BitStatus bit_host_padded_extents(BitLayout layout, int n, int k, int* n_padded,
                                  int* k_padded);

/// Absmean-ternarise and pack an FP32 row-major [N,K] matrix.
BitStatus bit_pack_host_f32(const float* w, int n, int k, int ldw, BitLayout layout,
                            BitScaleMode scale_mode, float eps, uint32_t* packed,
                            float* gamma, int n_padded, int k_padded);

/// Pack an already ternary INT8 row-major [N,K] matrix.  Values are mapped by
/// sign: negative -> -1, zero -> 0, positive -> +1.
BitStatus bit_pack_host_int8(const int8_t* w, int n, int k, int ldw, BitLayout layout,
                             uint32_t* packed, int n_padded, int k_padded);

/// Unpack to a row-major INT8 [N,K] matrix.
BitStatus bit_unpack_host_int8(const uint32_t* packed, BitLayout layout, int n, int k,
                               int n_padded, int k_padded, int8_t* out, int ldout);

/// Convert any packed layout to any other without materialising an unpacked
/// matrix.  Source and destination ranges must not overlap unless this is an
/// exact no-op conversion.
BitStatus bit_convert_layout_host(const uint32_t* src, BitLayout src_layout,
                                  int src_n_padded, int src_k_padded, uint32_t* dst,
                                  BitLayout dst_layout, int dst_n_padded, int dst_k_padded,
                                  int n, int k);

}  // namespace bitvideo
