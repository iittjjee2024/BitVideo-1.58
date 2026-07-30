// ===========================================================================
//  torch_bindings.h -- declarations shared by bit_gemm.cpp and kernels.cpp
// ===========================================================================
#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <tuple>

namespace bitvideo::torch_ext {

using PackedResult = std::tuple<at::Tensor, at::Tensor, int64_t, int64_t>;
using ConvertedResult = std::tuple<at::Tensor, int64_t, int64_t>;
using QuantizedResult = std::tuple<at::Tensor, at::Tensor>;

PackedResult pack_ternary_weights(const at::Tensor& weight, int64_t layout,
                                  int64_t scale_mode, double eps);

std::tuple<at::Tensor, int64_t, int64_t> pack_ternary_int8(
    const at::Tensor& weight, int64_t layout);

at::Tensor unpack_ternary_weights(const at::Tensor& packed, int64_t n, int64_t k,
                                  int64_t n_padded, int64_t k_padded, int64_t layout);

ConvertedResult convert_ternary_layout(const at::Tensor& packed, int64_t n, int64_t k,
                                       int64_t src_n_padded, int64_t src_k_padded,
                                       int64_t src_layout, int64_t dst_layout);

QuantizedResult quantize_activations(const at::Tensor& x, int64_t granularity,
                                     double eps);

at::Tensor bit_linear_forward(const at::Tensor& x, const at::Tensor& packed_weight,
                              const at::Tensor& weight_scale, int64_t out_features,
                              int64_t in_features, int64_t n_padded, int64_t k_padded,
                              int64_t layout, const c10::optional<at::Tensor>& bias,
                              int64_t out_dtype, int64_t weight_scale_mode,
                              int64_t activation_granularity, double alpha,
                              int64_t variant, int64_t split_k, bool autotune);

}  // namespace bitvideo::torch_ext
