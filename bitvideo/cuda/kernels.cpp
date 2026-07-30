// ===========================================================================
//  kernels.cpp -- PyBind module definition for bitvideo._C
// ===========================================================================
#include "torch_bindings.h"

#include "bit_gemm_kernel.h"
#include "bit_utils.h"

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <pybind11/stl.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

int checked_positive_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(), name,
              " must be in [1, INT_MAX], got ", value);
  return static_cast<int>(value);
}

bitvideo::BitLayout checked_layout(int64_t value) {
  TORCH_CHECK(value >= bitvideo::BV_LAYOUT_ROW_MAJOR &&
                  value <= bitvideo::BV_LAYOUT_MMA_INTERLEAVED,
              "layout must be 0(row), 1(column), 2(blocked-N64), or "
              "3(MMA-interleaved); got ",
              value);
  return static_cast<bitvideo::BitLayout>(value);
}

bitvideo::BitOutDtype checked_out_dtype(int64_t value) {
  TORCH_CHECK(value >= bitvideo::BV_OUT_FP32 && value <= bitvideo::BV_OUT_BF16,
              "out_dtype must be 0(float32), 1(float16), or 2(bfloat16); got ", value);
  return static_cast<bitvideo::BitOutDtype>(value);
}

bitvideo::BitKernelVariant checked_variant(int64_t value) {
  TORCH_CHECK(value >= std::numeric_limits<int32_t>::min() &&
                  value <= std::numeric_limits<int32_t>::max(),
              "variant is outside the int32 enum range: ", value);
  const auto variant = static_cast<bitvideo::BitKernelVariant>(value);
  TORCH_CHECK(variant == bitvideo::BV_KERNEL_AUTO ||
                  bitvideo::bit_variant_min_arch(variant) !=
                      std::numeric_limits<int>::max(),
              "unknown kernel variant: ", value);
  return variant;
}

py::dict device_info_dict(int device) {
  bitvideo::BitDeviceInfo info{};
  const bitvideo::BitStatus status = bitvideo::bit_device_info(device, &info);
  TORCH_CHECK(status == bitvideo::BV_OK, "device_info failed: ",
              bitvideo::bit_status_string(status));
  py::dict result;
  result["device"] = info.device;
  result["name"] = std::string(info.name);
  result["compute_capability"] = info.cc;
  result["sm_count"] = info.sm_count;
  result["max_threads_per_sm"] = info.max_threads_per_sm;
  result["max_grid_x"] = info.max_grid_x;
  result["max_smem_per_block_optin"] = info.max_smem_per_block_optin;
  result["max_smem_per_sm"] = info.max_smem_per_sm;
  result["registers_per_sm"] = info.regs_per_sm;
  result["l2_cache_bytes"] = info.l2_cache_bytes;
  result["core_clock_khz"] = info.core_clock_khz;
  result["dram_bandwidth_gbps"] = info.dram_bandwidth_gbps;
  result["has_cp_async"] = info.has_cp_async;
  result["has_mma_s8"] = info.has_mma_s8;
  result["cooperative_launch"] = info.cooperative_launch;
  result["memory_pools_supported"] = info.memory_pools_supported;
  return result;
}

py::dict kernel_stats_dict(const at::Tensor& x, int64_t out_features, int64_t in_features,
                           int64_t n_padded, int64_t k_padded, int64_t layout,
                           int64_t out_dtype, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
  const int n = checked_positive_int(out_features, "out_features");
  const int k = checked_positive_int(in_features, "in_features");
  const int np = checked_positive_int(n_padded, "n_padded");
  const int kp = checked_positive_int(k_padded, "k_padded");
  const auto packed_layout = checked_layout(layout);
  const auto output_dtype = checked_out_dtype(out_dtype);
  const auto kernel_variant = checked_variant(variant);
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) == k,
              "x must have shape [..., in_features]");
  TORCH_CHECK(x.numel() % k == 0, "x cannot be flattened into rows of K elements");
  const int64_t m64 = x.numel() / k;
  TORCH_CHECK(m64 > 0 && m64 <= std::numeric_limits<int>::max(),
              "flattened token count must be in [1, INT_MAX], got ", m64);
  TORCH_CHECK(np >= n && kp >= k,
              "padded extents must not be smaller than logical extents");
  int n_multiple = 1, k_multiple = 16;
  bitvideo::layout_alignment(packed_layout, n_multiple, k_multiple);
  TORCH_CHECK(np % n_multiple == 0 && kp % k_multiple == 0,
              "layout requires n_padded to be a multiple of ", n_multiple,
              " and k_padded to be a multiple of ", k_multiple);

  // This API models geometry only and never dereferences operands. The real x
  // address is retained solely so AUTO selection uses its actual alignment.
  bitvideo::BitGemmProblem p{};
  p.m = static_cast<int>(m64);
  p.n = n;
  p.k = k;
  p.n_padded = np;
  p.k_padded = kp;
  p.layout = packed_layout;
  p.out_dtype = output_dtype;
  p.x = reinterpret_cast<const int8_t*>(x.data_ptr());
  p.ldx = k;
  p.ldy = n;

  c10::cuda::CUDAGuard guard(x.device());
  bitvideo::BitKernelStats stats{};
  const bitvideo::BitStatus status =
      bitvideo::bit_gemm_kernel_stats(p, kernel_variant, &stats);
  TORCH_CHECK(status == bitvideo::BV_OK, "kernel_stats failed: ",
              bitvideo::bit_status_string(status));
  py::dict result;
  result["variant"] = static_cast<int>(stats.variant);
  result["variant_name"] = bitvideo::bit_variant_string(stats.variant);
  result["block_m"] = stats.block_m;
  result["block_n"] = stats.block_n;
  result["block_k"] = stats.block_k;
  result["warp_m"] = stats.warp_m;
  result["warp_n"] = stats.warp_n;
  result["threads"] = stats.threads;
  result["stages"] = stats.stages;
  result["registers_per_thread_estimate"] = stats.regs_per_thread;
  result["shared_memory_bytes"] = stats.smem_bytes;
  result["active_blocks_per_sm_estimate"] = stats.max_active_blocks_per_sm;
  result["theoretical_occupancy"] = stats.theoretical_occupancy;
  return result;
}

std::vector<std::string> variant_names() {
  const bitvideo::BitKernelVariant variants[] = {
      bitvideo::BV_KERNEL_AUTO,
      bitvideo::BV_KERNEL_GEMV_1WARP,
      bitvideo::BV_KERNEL_GEMV_SPLITK,
      bitvideo::BV_KERNEL_GEMV_WIDE,
      bitvideo::BV_KERNEL_GEMM_DP4A_64x64,
      bitvideo::BV_KERNEL_GEMM_DP4A_128x64,
      bitvideo::BV_KERNEL_GEMM_DP4A_128x128,
      bitvideo::BV_KERNEL_GEMM_DP4A_64x128,
      bitvideo::BV_KERNEL_GEMM_MMA_64x64,
      bitvideo::BV_KERNEL_GEMM_MMA_128x64,
      bitvideo::BV_KERNEL_GEMM_MMA_128x128,
      bitvideo::BV_KERNEL_GEMM_MMA_64x128,
  };
  std::vector<std::string> names;
  names.reserve(sizeof(variants) / sizeof(variants[0]));
  for (const auto v : variants) names.emplace_back(bitvideo::bit_variant_string(v));
  return names;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "BitVideo-1.58 W1.58A8 CUDA kernels";
  m.attr("__version__") = "1.58.0";
  m.attr("has_cuda_kernels") = true;

  m.def("pack_ternary_weights", &bitvideo::torch_ext::pack_ternary_weights,
        py::arg("weight"), py::arg("layout") = static_cast<int>(bitvideo::BV_LAYOUT_ROW_MAJOR),
        py::arg("scale_mode") = static_cast<int>(bitvideo::BV_SCALE_PER_TENSOR),
        py::arg("eps") = 1.0e-5,
        "Absmean-ternarize and pack a contiguous [N,K] weight tensor.");
  m.def("pack_ternary_int8", &bitvideo::torch_ext::pack_ternary_int8,
        py::arg("weight"), py::arg("layout") = static_cast<int>(bitvideo::BV_LAYOUT_ROW_MAJOR),
        "Pack a contiguous INT8 tensor whose values are interpreted by sign.");
  m.def("unpack_ternary_weights", &bitvideo::torch_ext::unpack_ternary_weights,
        py::arg("packed"), py::arg("out_features"), py::arg("in_features"),
        py::arg("n_padded"), py::arg("k_padded"), py::arg("layout"),
        "Unpack 2-bit weights to an INT8 {-1,0,+1} matrix.");
  m.def("convert_ternary_layout", &bitvideo::torch_ext::convert_ternary_layout,
        py::arg("packed"), py::arg("out_features"), py::arg("in_features"),
        py::arg("src_n_padded"), py::arg("src_k_padded"), py::arg("src_layout"),
        py::arg("dst_layout"), "Convert a packed tensor between inference layouts.");
  m.def("quantize_activations", &bitvideo::torch_ext::quantize_activations,
        py::arg("x"), py::arg("granularity") = 1, py::arg("eps") = 1.0e-5,
        "Dynamic symmetric INT8 activation quantization.");
  m.def("bit_linear_forward", &bitvideo::torch_ext::bit_linear_forward,
        py::arg("x"), py::arg("packed_weight"), py::arg("weight_scale"),
        py::arg("out_features"), py::arg("in_features"), py::arg("n_padded"),
        py::arg("k_padded"), py::arg("layout"), py::arg("bias") = py::none(),
        py::arg("out_dtype") = static_cast<int>(bitvideo::BV_OUT_FP16),
        py::arg("weight_scale_mode") =
            static_cast<int>(bitvideo::BV_SCALE_PER_TENSOR),
        py::arg("activation_granularity") = 1, py::arg("alpha") = 1.0,
        py::arg("variant") = static_cast<int>(bitvideo::BV_KERNEL_AUTO),
        py::arg("split_k") = 0, py::arg("autotune") = false,
        "Run fused dynamic-A8 / packed-W1.58 linear inference.");

  m.def("device_info", &device_info_dict, py::arg("device") = -1);
  m.def("kernel_stats", &kernel_stats_dict, py::arg("x"), py::arg("out_features"),
        py::arg("in_features"), py::arg("n_padded"), py::arg("k_padded"),
        py::arg("layout"), py::arg("out_dtype"), py::arg("variant") = 0);
  m.def("variant_names", &variant_names);
  m.def("clear_autotune_cache", &bitvideo::bit_gemm_autotune_reset);

  m.attr("LAYOUT_ROW_MAJOR") = static_cast<int>(bitvideo::BV_LAYOUT_ROW_MAJOR);
  m.attr("LAYOUT_COLUMN_MAJOR") = static_cast<int>(bitvideo::BV_LAYOUT_COLUMN_MAJOR);
  m.attr("LAYOUT_BLOCKED_N64") = static_cast<int>(bitvideo::BV_LAYOUT_BLOCKED_N64);
  m.attr("LAYOUT_MMA_INTERLEAVED") =
      static_cast<int>(bitvideo::BV_LAYOUT_MMA_INTERLEAVED);
  m.attr("SCALE_PER_TENSOR") = static_cast<int>(bitvideo::BV_SCALE_PER_TENSOR);
  m.attr("SCALE_PER_CHANNEL") = static_cast<int>(bitvideo::BV_SCALE_PER_CHANNEL);
  m.attr("OUT_FP32") = static_cast<int>(bitvideo::BV_OUT_FP32);
  m.attr("OUT_FP16") = static_cast<int>(bitvideo::BV_OUT_FP16);
  m.attr("OUT_BF16") = static_cast<int>(bitvideo::BV_OUT_BF16);
}
