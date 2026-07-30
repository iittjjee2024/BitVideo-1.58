// ===========================================================================
//  bit_gemm.cpp -- ATen bridge for BitVideo-1.58
//
//  Public Python operations:
//    pack_ternary_weights   CPU or CUDA offline absmean packer
//    pack_ternary_int8      pack an already ternary tensor
//    unpack_ternary_weights CPU or CUDA reference unpacker
//    convert_ternary_layout CPU or CUDA layout conversion
//    quantize_activations   fused CUDA dynamic INT8 quantizer
//    bit_linear_forward     fused quantize -> W1.58A8 GEMV/GEMM -> dequant
// ===========================================================================
#include "torch_bindings.h"

#include "bit_gemm_kernel.h"
#include "bit_packing.h"
#include "bit_utils.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace bitvideo::torch_ext {
namespace {

int checked_int(int64_t value, const char* name, bool positive = true) {
  if (positive)
    TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(), name,
                " must be in [1, INT_MAX], got ", value);
  else
    TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), name,
                " must be in [0, INT_MAX], got ", value);
  return static_cast<int>(value);
}

BitLayout checked_layout(int64_t value) {
  TORCH_CHECK(value >= BV_LAYOUT_ROW_MAJOR && value <= BV_LAYOUT_MMA_INTERLEAVED,
              "layout must be 0(row), 1(column), 2(blocked-N64), or 3(MMA-interleaved); got ",
              value);
  return static_cast<BitLayout>(value);
}

BitScaleMode checked_weight_scale_mode(int64_t value) {
  TORCH_CHECK(value == BV_SCALE_PER_TENSOR || value == BV_SCALE_PER_CHANNEL,
              "weight_scale_mode must be 0 (per-tensor) or 1 (per-channel); got ", value);
  return static_cast<BitScaleMode>(value);
}

float checked_positive_float(double value, const char* name) {
  TORCH_CHECK(std::isfinite(value) && value > 0.0, name,
              " must be finite and positive; got ", value);
  const float converted = static_cast<float>(value);
  TORCH_CHECK(std::isfinite(converted) && converted > 0.0f, name,
              " must be representable as a positive float32 value; got ", value);
  return converted;
}

BitKernelVariant checked_variant(int64_t value) {
  TORCH_CHECK(value >= std::numeric_limits<int32_t>::min() &&
                  value <= std::numeric_limits<int32_t>::max(),
              "variant is outside the int32 enum range: ", value);
  const auto variant = static_cast<BitKernelVariant>(value);
  TORCH_CHECK(variant == BV_KERNEL_AUTO ||
                  bit_variant_min_arch(variant) != std::numeric_limits<int>::max(),
              "unknown kernel variant: ", value);
  return variant;
}

BitOutDtype scalar_to_bit_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat: return BV_OUT_FP32;
    case at::kHalf: return BV_OUT_FP16;
    case at::kBFloat16: return BV_OUT_BF16;
    default:
      TORCH_CHECK(false, "expected float32, float16, or bfloat16 tensor, got ", dtype);
  }
  return BV_OUT_FP32;  // Unreachable: TORCH_CHECK above always throws.
}

at::ScalarType bit_to_scalar_dtype(int64_t dtype) {
  switch (dtype) {
    case BV_OUT_FP32: return at::kFloat;
    case BV_OUT_FP16: return at::kHalf;
    case BV_OUT_BF16: return at::kBFloat16;
    default:
      TORCH_CHECK(false, "out_dtype must be 0(float32), 1(float16), or 2(bfloat16); got ",
                  dtype);
  }
  return at::kFloat;  // Unreachable: TORCH_CHECK above always throws.
}

void check_floating_input(const at::Tensor& x, const char* name) {
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                  x.scalar_type() == at::kBFloat16,
              name, " must be float32, float16, or bfloat16; got ", x.scalar_type());
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

void check_packed_tensor(const at::Tensor& packed) {
  TORCH_CHECK(packed.scalar_type() == at::kInt,
              "packed_weight must use torch.int32 storage (four bytes per packed word)");
  TORCH_CHECK(packed.dim() == 1 && packed.is_contiguous(),
              "packed_weight must be a contiguous one-dimensional tensor");
}

void check_status(BitStatus status, const char* operation) {
  TORCH_CHECK(status == BV_OK, operation, " failed: ", bit_status_string(status),
              " (status=", static_cast<int>(status), ")");
}

cudaStream_t current_stream_for(const at::Tensor& tensor) {
  return c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

int64_t expected_words(BitLayout layout, int np, int kp) {
  const int64_t words = layout_num_words(layout, np, kp);
  TORCH_CHECK(words >= 0, "packed word count overflow");
  return words;
}

void check_same_cuda_device(const at::Tensor& reference, const at::Tensor& other,
                            const char* name) {
  TORCH_CHECK(other.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(other.device() == reference.device(), name, " must be on ", reference.device(),
              ", got ", other.device());
}

}  // namespace

PackedResult pack_ternary_weights(const at::Tensor& weight, int64_t layout_value,
                                  int64_t scale_mode_value, double eps_value) {
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [out_features, in_features]");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  check_floating_input(weight, "weight");
  const float eps = checked_positive_float(eps_value, "eps");

  const int n = checked_int(weight.size(0), "out_features");
  const int k = checked_int(weight.size(1), "in_features");
  const BitLayout layout = checked_layout(layout_value);
  const BitScaleMode scale_mode = checked_weight_scale_mode(scale_mode_value);
  int np = 0, kp = 0;
  check_status(bit_host_padded_extents(layout, n, k, &np, &kp), "padded extent calculation");
  const int64_t words = expected_words(layout, np, kp);
  const int64_t scale_count = scale_mode == BV_SCALE_PER_CHANNEL ? n : 1;

  auto packed = at::empty({words}, weight.options().dtype(at::kInt));
  auto gamma = at::empty({scale_count}, weight.options().dtype(at::kFloat));

  if (weight.is_cuda()) {
    c10::cuda::CUDAGuard guard(weight.device());
    const BitStatus status = bit_ternarize_and_pack(
        weight.data_ptr(), scalar_to_bit_dtype(weight.scalar_type()), n, k, k, layout,
        scale_mode, eps, reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
        gamma.data_ptr<float>(), np, kp, current_stream_for(weight));
    check_status(status, "CUDA ternary packing");
  } else {
    // The standalone host packer consumes FP32.  Converting once offline is
    // preferable to duplicating half/bfloat host conversion semantics.
    const at::Tensor fp32 = weight.scalar_type() == at::kFloat
                                ? weight
                                : weight.to(at::kFloat).contiguous();
    const BitStatus status = bit_pack_host_f32(
        fp32.data_ptr<float>(), n, k, k, layout, scale_mode, eps,
        reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()), gamma.data_ptr<float>(), np,
        kp);
    check_status(status, "CPU ternary packing");
  }
  return {packed, gamma, np, kp};
}

std::tuple<at::Tensor, int64_t, int64_t> pack_ternary_int8(const at::Tensor& weight,
                                                           int64_t layout_value) {
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [out_features, in_features]");
  TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be torch.int8");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  const int n = checked_int(weight.size(0), "out_features");
  const int k = checked_int(weight.size(1), "in_features");
  const BitLayout layout = checked_layout(layout_value);
  int np = 0, kp = 0;
  check_status(bit_host_padded_extents(layout, n, k, &np, &kp), "padded extent calculation");
  auto packed = at::empty({expected_words(layout, np, kp)}, weight.options().dtype(at::kInt));

  if (weight.is_cuda()) {
    c10::cuda::CUDAGuard guard(weight.device());
    check_status(bit_pack_int8_ternary(
                     weight.data_ptr<int8_t>(), n, k, k, layout,
                     reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()), np, kp,
                     current_stream_for(weight)),
                 "CUDA INT8 ternary packing");
  } else {
    check_status(bit_pack_host_int8(
                     weight.data_ptr<int8_t>(), n, k, k, layout,
                     reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()), np, kp),
                 "CPU INT8 ternary packing");
  }
  return {packed, np, kp};
}

at::Tensor unpack_ternary_weights(const at::Tensor& packed, int64_t n_value, int64_t k_value,
                                  int64_t np_value, int64_t kp_value,
                                  int64_t layout_value) {
  check_packed_tensor(packed);
  const int n = checked_int(n_value, "out_features");
  const int k = checked_int(k_value, "in_features");
  const int np = checked_int(np_value, "n_padded");
  const int kp = checked_int(kp_value, "k_padded");
  const BitLayout layout = checked_layout(layout_value);
  TORCH_CHECK(packed.numel() == expected_words(layout, np, kp),
              "packed_weight has ", packed.numel(), " words, expected ",
              expected_words(layout, np, kp));
  auto out = at::empty({n, k}, packed.options().dtype(at::kChar));

  if (packed.is_cuda()) {
    c10::cuda::CUDAGuard guard(packed.device());
    check_status(bit_unpack_to_int8(
                     reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>()), layout, n,
                     k, np, kp, out.data_ptr<int8_t>(), k, current_stream_for(packed)),
                 "CUDA ternary unpacking");
  } else {
    check_status(bit_unpack_host_int8(
                     reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>()), layout, n,
                     k, np, kp, out.data_ptr<int8_t>(), k),
                 "CPU ternary unpacking");
  }
  return out;
}

ConvertedResult convert_ternary_layout(const at::Tensor& packed, int64_t n_value,
                                       int64_t k_value, int64_t src_np_value,
                                       int64_t src_kp_value, int64_t src_layout_value,
                                       int64_t dst_layout_value) {
  check_packed_tensor(packed);
  const int n = checked_int(n_value, "out_features");
  const int k = checked_int(k_value, "in_features");
  const int src_np = checked_int(src_np_value, "src_n_padded");
  const int src_kp = checked_int(src_kp_value, "src_k_padded");
  const BitLayout src_layout = checked_layout(src_layout_value);
  const BitLayout dst_layout = checked_layout(dst_layout_value);
  TORCH_CHECK(packed.numel() == expected_words(src_layout, src_np, src_kp),
              "source packed word count does not match its metadata");
  int dst_np = 0, dst_kp = 0;
  check_status(bit_host_padded_extents(dst_layout, n, k, &dst_np, &dst_kp),
               "destination padded extent calculation");
  auto dst = at::empty({expected_words(dst_layout, dst_np, dst_kp)}, packed.options());

  if (packed.is_cuda()) {
    c10::cuda::CUDAGuard guard(packed.device());
    check_status(bit_convert_layout(
                     reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>()), src_layout,
                     reinterpret_cast<uint32_t*>(dst.data_ptr<int32_t>()), dst_layout, n, k,
                     src_np, src_kp, dst_np, dst_kp, current_stream_for(packed)),
                 "CUDA packed-layout conversion");
  } else {
    check_status(bit_convert_layout_host(
                     reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>()), src_layout,
                     src_np, src_kp, reinterpret_cast<uint32_t*>(dst.data_ptr<int32_t>()),
                     dst_layout, dst_np, dst_kp, n, k),
                 "CPU packed-layout conversion");
  }
  return {dst, dst_np, dst_kp};
}

QuantizedResult quantize_activations(const at::Tensor& x, int64_t granularity_value,
                                     double eps_value) {
  TORCH_CHECK(x.is_cuda(), "the compiled activation quantizer requires a CUDA tensor");
  TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
  check_floating_input(x, "x");
  const float eps = checked_positive_float(eps_value, "eps");
  TORCH_CHECK(granularity_value >= 0 && granularity_value <= 2,
              "granularity must be 0(tensor), 1(token), or 2(group-128)");
  const int k = checked_int(x.size(-1), "x.size(-1)");
  TORCH_CHECK(x.numel() % k == 0, "x cannot be flattened into rows of K elements");
  const int m = checked_int(x.numel() / k, "flattened token count", false);
  const int granularity = static_cast<int>(granularity_value);
  const int groups = div_ceil(k, 128);
  const int64_t scale_count =
      granularity == 0 ? 1 : (granularity == 1 ? m : static_cast<int64_t>(m) * groups);
  TORCH_CHECK(granularity != 2 || scale_count <= std::numeric_limits<int32_t>::max(),
              "group-128 scale count exceeds INT32_MAX: ", scale_count);

  c10::cuda::CUDAGuard guard(x.device());
  // q: [..., K], exactly x.sizes(); scales: [1], [M], or [M, ceil(K/128)].
  auto q = at::empty(x.sizes(), x.options().dtype(at::kChar));
  if (m == 0) {
    // The empty per-tensor reduction has max(abs(x)) == 0 by definition, so
    // its usable dequantization scale is eps / 127. Other modes have no rows.
    auto scales = granularity == 0
                      ? at::full({1}, eps / 127.0f, x.options().dtype(at::kFloat))
                      : at::empty({scale_count}, x.options().dtype(at::kFloat));
    if (granularity == 2) scales = scales.view({0, groups});
    return {q, scales};
  }

  auto scales = at::empty({scale_count}, x.options().dtype(at::kFloat));
  check_status(bit_quantize_activations(
                   x.data_ptr(), scalar_to_bit_dtype(x.scalar_type()), m, k, k, granularity,
                   eps, q.data_ptr<int8_t>(), k, scales.data_ptr<float>(),
                   current_stream_for(x)),
               "CUDA activation quantization");
  if (granularity == 2) scales = scales.view({m, groups});
  return {q, scales};
}

at::Tensor bit_linear_forward(const at::Tensor& x, const at::Tensor& packed_weight,
                              const at::Tensor& weight_scale, int64_t out_features_value,
                              int64_t in_features_value, int64_t np_value, int64_t kp_value,
                              int64_t layout_value, const c10::optional<at::Tensor>& bias,
                              int64_t out_dtype_value, int64_t weight_scale_mode_value,
                              int64_t activation_granularity_value, double alpha_value,
                              int64_t variant_value, int64_t split_k_value, bool autotune) {
  TORCH_CHECK(x.is_cuda(), "bit_linear_forward requires CUDA input");
  TORCH_CHECK(x.dim() >= 1, "x must have shape [..., in_features]");
  check_floating_input(x, "x");
  check_packed_tensor(packed_weight);
  check_same_cuda_device(x, packed_weight, "packed_weight");
  check_same_cuda_device(x, weight_scale, "weight_scale");
  TORCH_CHECK(weight_scale.scalar_type() == at::kFloat && weight_scale.is_contiguous(),
              "weight_scale must be contiguous float32");
  TORCH_CHECK(weight_scale.dim() == 1, "weight_scale must be one-dimensional");

  const int n = checked_int(out_features_value, "out_features");
  const int k = checked_int(in_features_value, "in_features");
  const int np = checked_int(np_value, "n_padded");
  const int kp = checked_int(kp_value, "k_padded");
  const BitLayout layout = checked_layout(layout_value);
  const BitScaleMode weight_scale_mode = checked_weight_scale_mode(weight_scale_mode_value);
  TORCH_CHECK(x.size(-1) == k, "x.size(-1) must equal in_features (", k, "), got ",
              x.size(-1));
  TORCH_CHECK(packed_weight.numel() == expected_words(layout, np, kp),
              "packed_weight word count does not match layout metadata");
  TORCH_CHECK(weight_scale.numel() == (weight_scale_mode == BV_SCALE_PER_CHANNEL ? n : 1),
              "weight_scale has wrong length for its scale mode");
  TORCH_CHECK(activation_granularity_value == 0 || activation_granularity_value == 1,
              "bit_linear_forward supports per-tensor (0) or per-token (1) activation scales; "
              "group-128 quantization requires a groupwise GEMM epilogue");
  TORCH_CHECK(std::isfinite(alpha_value), "alpha must be finite");
  const float alpha = static_cast<float>(alpha_value);
  TORCH_CHECK(std::isfinite(alpha), "alpha exceeds the finite float32 range: ", alpha_value);
  const int split_k = checked_int(split_k_value, "split_k", false);
  BitKernelVariant variant = checked_variant(variant_value);

  const int64_t m64 = x.numel() / k;
  TORCH_CHECK(m64 <= std::numeric_limits<int>::max(), "flattened token count exceeds INT_MAX");
  const int m = static_cast<int>(m64);

  const float* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined()) {
    check_same_cuda_device(x, *bias, "bias");
    TORCH_CHECK(bias->scalar_type() == at::kFloat && bias->is_contiguous() &&
                    bias->dim() == 1 && bias->numel() == n,
                "bias must be contiguous float32 with shape [out_features]");
    bias_ptr = bias->data_ptr<float>();
  }

  // output: [..., N], preserving every leading dimension of x.
  std::vector<int64_t> output_shape = x.sizes().vec();
  output_shape.back() = n;
  const at::ScalarType output_scalar = bit_to_scalar_dtype(out_dtype_value);
  auto output_options = x.options().dtype(output_scalar);
  if (m == 0) return at::empty(output_shape, output_options);

  c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = current_stream_for(x);
  auto q = at::empty(x.sizes(), x.options().dtype(at::kChar));
  const int activation_granularity = static_cast<int>(activation_granularity_value);
  const int64_t activation_scale_count = activation_granularity == 0 ? 1 : m;
  auto activation_scale = at::empty({activation_scale_count}, x.options().dtype(at::kFloat));
  check_status(bit_quantize_activations(
                   x.data_ptr(), scalar_to_bit_dtype(x.scalar_type()), m, k, k,
                   activation_granularity, 1.0e-5f, q.data_ptr<int8_t>(), k,
                   activation_scale.data_ptr<float>(), stream),
               "activation quantization");

  auto output_2d = at::empty({m, n}, output_options);
  BitGemmProblem problem{};
  problem.m = m;
  problem.n = n;
  problem.k = k;
  problem.x = q.data_ptr<int8_t>();
  problem.w_packed =
      reinterpret_cast<const uint32_t*>(packed_weight.data_ptr<int32_t>());
  problem.layout = layout;
  problem.n_padded = np;
  problem.k_padded = kp;
  problem.act_scale = activation_scale.data_ptr<float>();
  problem.act_scale_mode = activation_granularity == 0 ? BV_ACT_SCALE_PER_TENSOR
                                                        : BV_ACT_SCALE_PER_TOKEN;
  problem.w_scale = weight_scale.data_ptr<float>();
  problem.w_scale_mode = weight_scale_mode;
  problem.bias = bias_ptr;
  problem.alpha = alpha;
  problem.beta = 0.0f;
  problem.y = output_2d.data_ptr();
  problem.out_dtype = scalar_to_bit_dtype(output_scalar);
  problem.ldx = k;
  problem.ldy = n;
  problem.split_k = split_k;
  problem.deterministic = true;

  char reason[512];
  const BitStatus validation = bit_gemm_validate(problem, reason, sizeof(reason));
  TORCH_CHECK(validation == BV_OK, "invalid BitLinear problem: ", reason, " (",
              bit_status_string(validation), ")");

  size_t workspace_bytes = bit_gemm_workspace_size(problem, variant);
  if (autotune && layout == BV_LAYOUT_ROW_MAJOR && m <= 8) {
    const size_t split_bytes = bit_gemm_workspace_size(problem, BV_KERNEL_GEMV_SPLITK);
    workspace_bytes = std::max(workspace_bytes, split_bytes);
  }
  TORCH_CHECK(workspace_bytes != std::numeric_limits<size_t>::max(),
              "BitLinear workspace size overflow");
  TORCH_CHECK(workspace_bytes <=
                  static_cast<size_t>(std::numeric_limits<int64_t>::max()),
              "BitLinear workspace cannot be represented as a tensor length: ",
              workspace_bytes, " bytes");
  // workspace: [workspace_bytes] raw uint8 scratch for deterministic split-K.
  auto workspace = workspace_bytes > 0
                       ? at::empty({static_cast<int64_t>(workspace_bytes)},
                                   x.options().dtype(at::kByte))
                       : at::Tensor();
  void* workspace_ptr =
      workspace_bytes > 0 ? static_cast<void*>(workspace.data_ptr<uint8_t>()) : nullptr;

  if (autotune) {
    BitKernelVariant winner = BV_KERNEL_AUTO;
    float winner_ms = 0.0f;
    check_status(bit_gemm_autotune(problem, workspace_ptr, workspace_bytes, 2, 10, stream,
                                   &winner, &winner_ms),
                 "BitLinear autotuning");
    variant = winner;
  }
  check_status(bit_gemm_launch(problem, variant, workspace_ptr, workspace_bytes, stream),
               "BitLinear CUDA launch");
  return output_2d.view(output_shape);
}

}  // namespace bitvideo::torch_ext
