// ===========================================================================
//  bit_gemm_launch.cu -- BitVideo-1.58
//
//  Runtime policy layer for the W1.58A8 kernels:
//    * strict argument/layout/alignment/overflow validation
//    * cached device capability discovery (sm80 through sm120)
//    * architecture- and shape-aware tile selection
//    * deterministic split-K workspace planning
//    * measured autotuning with stream-safe scratch output and a process cache
//    * analytic occupancy, traffic, arithmetic-intensity and roofline models
//
//  This file owns policy only.  Kernel bodies and the low-level launch shims
//  live in bit_gemm_kernel.cu, keeping the hot translation unit independent
//  of STL and mutex code.
// ===========================================================================
#include "bit_gemm_kernel.h"
#include "bit_utils.h"
#include "cuda_helpers.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace bitvideo {

// Kernel launch shims implemented in bit_gemm_kernel.cu.
namespace detail {
BitStatus launch_gemv_kernel(const BitGemmProblem& p, BitKernelVariant variant, int split_k,
                             void* workspace, cudaStream_t stream);
BitStatus launch_dp4a_kernel(const BitGemmProblem& p, BitKernelVariant variant,
                             cudaStream_t stream);
BitStatus launch_mma_kernel(const BitGemmProblem& p, BitKernelVariant variant,
                            cudaStream_t stream);
}  // namespace detail

namespace {

constexpr int kMinSupportedCc = 80;
constexpr int kGemvKPerWarpStep = 2048;
constexpr int kMaxSplitK = 64;
constexpr int kMaxBlocksPerSm = 32;

std::mutex g_device_mutex;
std::unordered_map<int, BitDeviceInfo> g_device_cache;

struct TuneKey {
  int device;
  int cc;
  int m;
  int n;
  int k;
  int n_padded;
  int k_padded;
  int ldx;
  int ldy;
  int split_k;
  int x_alignment;
  int epilogue_flags;
  int layout;
  int dtype;

  bool operator==(const TuneKey& other) const noexcept {
    return device == other.device && cc == other.cc && m == other.m && n == other.n &&
           k == other.k && n_padded == other.n_padded && k_padded == other.k_padded &&
           ldx == other.ldx && ldy == other.ldy && split_k == other.split_k &&
           x_alignment == other.x_alignment && epilogue_flags == other.epilogue_flags &&
           layout == other.layout && dtype == other.dtype;
  }
};

struct TuneKeyHash {
  size_t operator()(const TuneKey& x) const noexcept {
    size_t h = 1469598103934665603ull;
    const int values[] = {x.device, x.cc, x.m, x.n, x.k, x.n_padded, x.k_padded,
                          x.ldx, x.ldy, x.split_k, x.x_alignment, x.epilogue_flags,
                          x.layout, x.dtype};
    for (int v : values) {
      h ^= static_cast<size_t>(static_cast<uint32_t>(v));
      h *= 1099511628211ull;
    }
    return h;
  }
};

std::mutex g_tune_mutex;
std::unordered_map<TuneKey, BitKernelVariant, TuneKeyHash> g_tune_cache;

inline TuneKey make_tune_key(const BitGemmProblem& p, const BitDeviceInfo& dev) {
  int flags = p.beta != 0.0f ? 1 : 0;
  flags |= p.bias ? 2 : 0;
  flags |= p.act_scale ? 4 : 0;
  flags |= p.w_scale ? 8 : 0;
  flags |= static_cast<int>(p.act_scale_mode) << 4;
  flags |= static_cast<int>(p.w_scale_mode) << 6;
  return TuneKey{dev.device,
                 dev.cc,
                 p.m,
                 p.n,
                 p.k,
                 p.n_padded,
                 p.k_padded,
                 p.ldx_or_default(),
                 p.ldy_or_default(),
                 p.split_k,
                 static_cast<int>(reinterpret_cast<uintptr_t>(p.x) & 15u),
                 flags,
                 static_cast<int>(p.layout),
                 static_cast<int>(p.out_dtype)};
}

inline size_t dtype_bytes(BitOutDtype dtype) {
  switch (dtype) {
    case BV_OUT_FP32: return 4;
    case BV_OUT_FP16:
    case BV_OUT_BF16: return 2;
    default: return 0;
  }
}

inline bool valid_layout(BitLayout layout) {
  return layout >= BV_LAYOUT_ROW_MAJOR && layout <= BV_LAYOUT_MMA_INTERLEAVED;
}

inline bool valid_dtype(BitOutDtype dtype) {
  return dtype >= BV_OUT_FP32 && dtype <= BV_OUT_BF16;
}

inline bool valid_variant(BitKernelVariant v) {
  switch (v) {
    case BV_KERNEL_AUTO:
    case BV_KERNEL_GEMV_1WARP:
    case BV_KERNEL_GEMV_SPLITK:
    case BV_KERNEL_GEMV_WIDE:
    case BV_KERNEL_GEMM_DP4A_64x64:
    case BV_KERNEL_GEMM_DP4A_128x64:
    case BV_KERNEL_GEMM_DP4A_128x128:
    case BV_KERNEL_GEMM_DP4A_64x128:
    case BV_KERNEL_GEMM_MMA_64x64:
    case BV_KERNEL_GEMM_MMA_128x64:
    case BV_KERNEL_GEMM_MMA_128x128:
    case BV_KERNEL_GEMM_MMA_64x128: return true;
    default: return false;
  }
}

inline bool is_gemv_variant(BitKernelVariant v) {
  return v == BV_KERNEL_GEMV_1WARP || v == BV_KERNEL_GEMV_SPLITK ||
         v == BV_KERNEL_GEMV_WIDE;
}

inline bool is_dp4a_variant(BitKernelVariant v) {
  return v >= BV_KERNEL_GEMM_DP4A_64x64 && v <= BV_KERNEL_GEMM_DP4A_64x128;
}

inline bool is_mma_variant(BitKernelVariant v) {
  return v >= BV_KERNEL_GEMM_MMA_64x64 && v <= BV_KERNEL_GEMM_MMA_64x128;
}

inline size_t packed_required_alignment(BitLayout layout) {
  return (layout == BV_LAYOUT_BLOCKED_N64 || layout == BV_LAYOUT_MMA_INTERLEAVED) ? 16u : 4u;
}

inline void set_reason(char* why, int why_len, const char* text) {
  if (why && why_len > 0) std::snprintf(why, static_cast<size_t>(why_len), "%s", text);
}

template <typename... Args>
void set_reasonf(char* why, int why_len, const char* fmt, Args... args) {
  if (why && why_len > 0)
    std::snprintf(why, static_cast<size_t>(why_len), fmt, args...);
}

inline bool checked_mul_size(size_t a, size_t b, size_t& out) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
  out = a * b;
  return true;
}

inline bool checked_add_size(size_t a, size_t b, size_t& out) {
  if (b > std::numeric_limits<size_t>::max() - a) return false;
  out = a + b;
  return true;
}

int effective_split_k(const BitGemmProblem& p, BitKernelVariant variant,
                      const BitDeviceInfo& dev) {
  if (variant != BV_KERNEL_GEMV_SPLITK) return 1;
  const int k_steps = std::max(1, div_ceil(p.k, kGemvKPerWarpStep));
  if (p.split_k == 1 || k_steps < 2) return 1;
  if (p.split_k > 1) return p.split_k;  // strict validation guarantees <= k_steps and <= 64

  const int blocks_x = std::max(1, div_ceil(p.n, 8 * 2));
  const int target_ctas = std::max(dev.sm_count, 1) * 2;
  const int desired = std::max(2, div_ceil(target_ctas, blocks_x));
  return std::min({desired, k_steps, kMaxSplitK});
}

bool variant_matches_problem(const BitGemmProblem& p, BitKernelVariant v,
                             const BitDeviceInfo& dev) {
  if (!valid_variant(v) || v == BV_KERNEL_AUTO || bit_variant_min_arch(v) > dev.cc)
    return false;
  if (is_gemv_variant(v)) {
    if (p.layout != BV_LAYOUT_ROW_MAJOR || p.m < 1 || p.m > 8) return false;
    if (p.split_k > 1) return v == BV_KERNEL_GEMV_SPLITK;
    if (p.split_k == 1) return v != BV_KERNEL_GEMV_SPLITK;
    if (v == BV_KERNEL_GEMV_SPLITK)
      return div_ceil(p.k, kGemvKPerWarpStep) >= 2;
    return true;
  }
  if (is_dp4a_variant(v)) return p.layout == BV_LAYOUT_BLOCKED_N64;
  if (is_mma_variant(v)) return p.layout == BV_LAYOUT_MMA_INTERLEAVED && dev.has_mma_s8;
  return false;
}

BitKernelVariant heuristic_variant(const BitGemmProblem& p, const BitDeviceInfo& dev) {
  if (p.layout == BV_LAYOUT_ROW_MAJOR && p.m <= 8) {
    const int one_warp_blocks = div_ceil(p.n, 8);
    const int k_steps = div_ceil(p.k, kGemvKPerWarpStep);
    if (p.split_k > 1) return BV_KERNEL_GEMV_SPLITK;
    // Split K only when N does not expose enough independent output CTAs and
    // K is long enough to give every slice useful work.
    if (p.split_k != 1 && one_warp_blocks < dev.sm_count && k_steps >= 2)
      return BV_KERNEL_GEMV_SPLITK;
    // Wide ILP amortises activation loads for very small M once N already
    // provides abundant CTA-level parallelism.
    if (p.m <= 2 && p.n >= 4096 && one_warp_blocks >= 2 * dev.sm_count)
      return BV_KERNEL_GEMV_WIDE;
    return BV_KERNEL_GEMV_1WARP;
  }

  const bool big_m = p.m >= 96;
  const bool big_n = p.n >= 96;
  if (p.layout == BV_LAYOUT_BLOCKED_N64) {
    if (big_m && big_n) return BV_KERNEL_GEMM_DP4A_128x128;
    if (big_m) return BV_KERNEL_GEMM_DP4A_128x64;
    if (big_n) return BV_KERNEL_GEMM_DP4A_64x128;
    return BV_KERNEL_GEMM_DP4A_64x64;
  }
  if (p.layout == BV_LAYOUT_MMA_INTERLEAVED && dev.has_mma_s8) {
    if (big_m && big_n) return BV_KERNEL_GEMM_MMA_128x128;
    if (big_m) return BV_KERNEL_GEMM_MMA_128x64;
    if (big_n) return BV_KERNEL_GEMM_MMA_64x128;
    return BV_KERNEL_GEMM_MMA_64x64;
  }
  return BV_KERNEL_AUTO;
}

BitStatus launch_variant_unchecked(const BitGemmProblem& p, BitKernelVariant variant,
                                   void* workspace, size_t workspace_bytes,
                                   cudaStream_t stream) {
  BitDeviceInfo dev;
  BitStatus status = bit_device_info(-1, &dev);
  if (status != BV_OK) return status;
  const size_t required = bit_gemm_workspace_size(p, variant);
  if (required == std::numeric_limits<size_t>::max()) return BV_ERR_WORKSPACE_TOO_SMALL;
  if (required > workspace_bytes || (required > 0 && workspace == nullptr))
    return BV_ERR_WORKSPACE_TOO_SMALL;
  if (required > 0 && !is_aligned(workspace, alignof(int32_t)))
    return BV_ERR_BAD_ALIGNMENT;

  if (is_gemv_variant(variant)) {
    const int split = effective_split_k(p, variant, dev);
    return detail::launch_gemv_kernel(p, variant, split, workspace, stream);
  }
  if (is_dp4a_variant(variant)) return detail::launch_dp4a_kernel(p, variant, stream);
  if (is_mma_variant(variant)) return detail::launch_mma_kernel(p, variant, stream);
  return BV_ERR_NO_VARIANT;
}

std::vector<BitKernelVariant> candidates_for(const BitGemmProblem& p,
                                             const BitDeviceInfo& dev) {
  std::vector<BitKernelVariant> result;
  if (p.layout == BV_LAYOUT_ROW_MAJOR && p.m <= 8) {
    result = {BV_KERNEL_GEMV_1WARP, BV_KERNEL_GEMV_WIDE};
    if (p.split_k != 1 && div_ceil(p.k, kGemvKPerWarpStep) >= 2)
      result.push_back(BV_KERNEL_GEMV_SPLITK);
  } else if (p.layout == BV_LAYOUT_BLOCKED_N64) {
    result = {BV_KERNEL_GEMM_DP4A_64x64, BV_KERNEL_GEMM_DP4A_128x64,
              BV_KERNEL_GEMM_DP4A_64x128, BV_KERNEL_GEMM_DP4A_128x128};
  } else if (p.layout == BV_LAYOUT_MMA_INTERLEAVED && dev.has_mma_s8) {
    result = {BV_KERNEL_GEMM_MMA_64x64, BV_KERNEL_GEMM_MMA_128x64,
              BV_KERNEL_GEMM_MMA_64x128, BV_KERNEL_GEMM_MMA_128x128};
  }
  result.erase(std::remove_if(result.begin(), result.end(), [&](BitKernelVariant v) {
                 return !variant_matches_problem(p, v, dev);
               }),
               result.end());
  return result;
}

}  // namespace

// ===========================================================================
//  Public metadata helpers
// ===========================================================================
const char* bit_status_string(BitStatus s) {
  switch (s) {
    case BV_OK: return "success";
    case BV_ERR_NULL_POINTER: return "null pointer";
    case BV_ERR_BAD_SHAPE: return "invalid shape, stride, or padded extent";
    case BV_ERR_BAD_ALIGNMENT: return "pointer alignment does not satisfy kernel layout";
    case BV_ERR_UNSUPPORTED_DTYPE: return "unsupported data type";
    case BV_ERR_UNSUPPORTED_LAYOUT: return "unsupported packed weight layout";
    case BV_ERR_UNSUPPORTED_ARCH: return "GPU architecture is unsupported";
    case BV_ERR_WORKSPACE_TOO_SMALL: return "workspace is null or too small";
    case BV_ERR_LAUNCH_FAILED: return "CUDA kernel launch failed";
    case BV_ERR_NO_VARIANT: return "no kernel variant supports this shape/layout";
    case BV_ERR_INVALID_ARGUMENT: return "invalid argument";
    default: return "unknown BitVideo status";
  }
}

const char* bit_dtype_string(BitOutDtype d) {
  switch (d) {
    case BV_OUT_FP32: return "float32";
    case BV_OUT_FP16: return "float16";
    case BV_OUT_BF16: return "bfloat16";
    default: return "unknown";
  }
}

const char* bit_variant_string(BitKernelVariant v) {
  switch (v) {
    case BV_KERNEL_AUTO: return "auto";
    case BV_KERNEL_GEMV_1WARP: return "gemv_1warp";
    case BV_KERNEL_GEMV_SPLITK: return "gemv_splitk";
    case BV_KERNEL_GEMV_WIDE: return "gemv_wide";
    case BV_KERNEL_GEMM_DP4A_64x64: return "gemm_dp4a_64x64";
    case BV_KERNEL_GEMM_DP4A_128x64: return "gemm_dp4a_128x64";
    case BV_KERNEL_GEMM_DP4A_128x128: return "gemm_dp4a_128x128";
    case BV_KERNEL_GEMM_DP4A_64x128: return "gemm_dp4a_64x128";
    case BV_KERNEL_GEMM_MMA_64x64: return "gemm_mma_64x64";
    case BV_KERNEL_GEMM_MMA_128x64: return "gemm_mma_128x64";
    case BV_KERNEL_GEMM_MMA_128x128: return "gemm_mma_128x128";
    case BV_KERNEL_GEMM_MMA_64x128: return "gemm_mma_64x128";
    default: return "unknown";
  }
}

bool bit_variant_requires_mma_layout(BitKernelVariant v) { return is_mma_variant(v); }
bool bit_variant_requires_blocked_layout(BitKernelVariant v) { return is_dp4a_variant(v); }

BitLayout bit_variant_preferred_layout(BitKernelVariant v) {
  if (is_mma_variant(v)) return BV_LAYOUT_MMA_INTERLEAVED;
  if (is_dp4a_variant(v)) return BV_LAYOUT_BLOCKED_N64;
  return BV_LAYOUT_ROW_MAJOR;
}

int bit_variant_min_arch(BitKernelVariant v) {
  if (v == BV_KERNEL_AUTO) return 0;
  if (!valid_variant(v)) return std::numeric_limits<int>::max();
  // The distributed package intentionally supports CUDA architectures 8.0+;
  // DP4A itself exists earlier, but cp.async pipelines do not.
  return 80;
}

// ===========================================================================
//  Device discovery
// ===========================================================================
BitStatus bit_device_info(int device, BitDeviceInfo* out) {
  if (!out) return BV_ERR_NULL_POINTER;
  if (device < 0) {
    if (cudaGetDevice(&device) != cudaSuccess) return BV_ERR_LAUNCH_FAILED;
  }

  {
    std::lock_guard<std::mutex> lock(g_device_mutex);
    const auto it = g_device_cache.find(device);
    if (it != g_device_cache.end()) {
      *out = it->second;
      return BV_OK;
    }
  }

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return BV_ERR_LAUNCH_FAILED;

  BitDeviceInfo info{};
  info.device = device;
  info.cc = prop.major * 10 + prop.minor;
  info.sm_count = prop.multiProcessorCount;
  info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  info.max_grid_x = prop.maxGridSize[0];
  info.max_smem_per_sm = static_cast<int>(prop.sharedMemPerMultiprocessor);
  info.regs_per_sm = prop.regsPerMultiprocessor;
  info.l2_cache_bytes = prop.l2CacheSize;
  info.core_clock_khz = prop.clockRate;
  info.warp_size = prop.warpSize;
  info.has_cp_async = info.cc >= 80;
  info.has_mma_s8 = info.cc >= 80;
  std::snprintf(info.name, sizeof(info.name), "%s", prop.name);

  int value = 0;
  if (cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) ==
      cudaSuccess)
    info.max_smem_per_block_optin = value;
  else
    info.max_smem_per_block_optin = static_cast<int>(prop.sharedMemPerBlock);
  if (cudaDeviceGetAttribute(&value, cudaDevAttrCooperativeLaunch, device) == cudaSuccess)
    info.cooperative_launch = value != 0;
  if (cudaDeviceGetAttribute(&value, cudaDevAttrMemoryPoolsSupported, device) == cudaSuccess)
    info.memory_pools_supported = value != 0;

  const double memory_hz = static_cast<double>(prop.memoryClockRate) * 1000.0;
  const double bytes_per_edge = static_cast<double>(prop.memoryBusWidth) / 8.0;
  info.dram_bandwidth_gbps = 2.0 * memory_hz * bytes_per_edge / 1.0e9;

  {
    std::lock_guard<std::mutex> lock(g_device_mutex);
    g_device_cache[device] = info;
  }
  *out = info;
  return BV_OK;
}

// ===========================================================================
//  Validation, workspace, selection and launch
// ===========================================================================
BitStatus bit_gemm_validate(const BitGemmProblem& p, char* why, int why_len) {
  if (why && why_len > 0) why[0] = '\0';
  if (!p.x || !p.w_packed || !p.y) {
    set_reason(why, why_len, "x, w_packed and y must all be non-null");
    return BV_ERR_NULL_POINTER;
  }
  if (p.m <= 0 || p.n <= 0 || p.k <= 0) {
    set_reasonf(why, why_len, "M, N and K must be positive (got %d x %d x %d)", p.m, p.n,
                p.k);
    return BV_ERR_BAD_SHAPE;
  }
  if (p.k > INT32_MAX / 128) {
    set_reason(why, why_len, "K can overflow an INT32 ternary accumulator");
    return BV_ERR_BAD_SHAPE;
  }
  if (p.ldx < 0 || p.ldy < 0) {
    set_reason(why, why_len, "negative leading dimensions are invalid; use zero for default");
    return BV_ERR_BAD_SHAPE;
  }
  if (p.ldx_or_default() < p.k || p.ldy_or_default() < p.n) {
    set_reasonf(why, why_len, "leading dimensions are too small (ldx=%d, ldy=%d)",
                p.ldx_or_default(), p.ldy_or_default());
    return BV_ERR_BAD_SHAPE;
  }
  if (!valid_layout(p.layout)) {
    set_reason(why, why_len, "packed weight layout enum is invalid");
    return BV_ERR_UNSUPPORTED_LAYOUT;
  }
  if (!valid_dtype(p.out_dtype)) {
    set_reason(why, why_len, "output dtype must be FP16, BF16 or FP32");
    return BV_ERR_UNSUPPORTED_DTYPE;
  }
  if (p.n_padded < p.n || p.k_padded < p.k) {
    set_reasonf(why, why_len, "padded extents (%d,%d) are smaller than logical (%d,%d)",
                p.n_padded, p.k_padded, p.n, p.k);
    return BV_ERR_BAD_SHAPE;
  }
  int n_multiple = 1, k_multiple = 16;
  layout_alignment(p.layout, n_multiple, k_multiple);
  if (p.n_padded % n_multiple != 0 || p.k_padded % k_multiple != 0) {
    set_reasonf(why, why_len, "layout requires N padded to %d and K padded to %d", n_multiple,
                k_multiple);
    return BV_ERR_BAD_SHAPE;
  }
  if (!is_aligned(p.w_packed, packed_required_alignment(p.layout))) {
    set_reasonf(why, why_len, "packed %s weights require %zu-byte base alignment",
                p.layout == BV_LAYOUT_MMA_INTERLEAVED ? "MMA" : "blocked/row",
                packed_required_alignment(p.layout));
    return BV_ERR_BAD_ALIGNMENT;
  }
  const size_t out_align = dtype_bytes(p.out_dtype);
  if (!is_aligned(p.y, out_align)) {
    set_reasonf(why, why_len, "output pointer requires %zu-byte alignment", out_align);
    return BV_ERR_BAD_ALIGNMENT;
  }
  if ((p.act_scale && !is_aligned(p.act_scale, 4)) ||
      (p.w_scale && !is_aligned(p.w_scale, 4)) || (p.bias && !is_aligned(p.bias, 4))) {
    set_reason(why, why_len, "scale and bias pointers require 4-byte alignment");
    return BV_ERR_BAD_ALIGNMENT;
  }
  if (p.act_scale_mode != BV_ACT_SCALE_PER_TENSOR &&
      p.act_scale_mode != BV_ACT_SCALE_PER_TOKEN) {
    set_reason(why, why_len, "invalid activation scale mode");
    return BV_ERR_INVALID_ARGUMENT;
  }
  if (p.w_scale_mode != BV_SCALE_PER_TENSOR && p.w_scale_mode != BV_SCALE_PER_CHANNEL) {
    set_reason(why, why_len, "invalid weight scale mode");
    return BV_ERR_INVALID_ARGUMENT;
  }
  if (!std::isfinite(p.alpha) || !std::isfinite(p.beta)) {
    set_reason(why, why_len, "alpha and beta must be finite");
    return BV_ERR_INVALID_ARGUMENT;
  }
  if (p.split_k < 0 || p.split_k > kMaxSplitK) {
    set_reasonf(why, why_len, "split_k must be in [0,%d]", kMaxSplitK);
    return BV_ERR_INVALID_ARGUMENT;
  }
  const int useful_k_slices = div_ceil(p.k, kGemvKPerWarpStep);
  if (p.split_k > 1 && p.split_k > useful_k_slices) {
    set_reasonf(why, why_len, "split_k=%d exceeds the %d useful K slices", p.split_k,
                useful_k_slices);
    return BV_ERR_INVALID_ARGUMENT;
  }

  BitDeviceInfo dev;
  const BitStatus device_status = bit_device_info(-1, &dev);
  if (device_status != BV_OK) {
    set_reason(why, why_len, "could not query the active CUDA device");
    return device_status;
  }
  if (dev.cc < kMinSupportedCc) {
    set_reasonf(why, why_len, "compute capability sm_%d is unsupported; sm_80+ is required",
                dev.cc);
    return BV_ERR_UNSUPPORTED_ARCH;
  }
  const int64_t grid_blocks = p.layout == BV_LAYOUT_ROW_MAJOR
                                  ? div_ceil64(p.n, 8)
                                  : div_ceil64(p.m, 64) * div_ceil64(p.n, 64);
  if (grid_blocks > dev.max_grid_x) {
    set_reasonf(why, why_len, "problem requires %lld CTAs in grid.x; device limit is %d",
                static_cast<long long>(grid_blocks), dev.max_grid_x);
    return BV_ERR_BAD_SHAPE;
  }
  return BV_OK;
}

size_t bit_gemm_workspace_size(const BitGemmProblem& p, BitKernelVariant variant) {
  if (p.m <= 0 || p.n <= 0 || p.k <= 0 || !valid_variant(variant))
    return std::numeric_limits<size_t>::max();
  BitDeviceInfo dev;
  if (bit_device_info(-1, &dev) != BV_OK) return std::numeric_limits<size_t>::max();
  if (variant == BV_KERNEL_AUTO) variant = bit_gemm_select_variant(p, dev);
  if (!is_gemv_variant(variant)) return 0;
  const int split = effective_split_k(p, variant, dev);
  if (split <= 1) return 0;
  size_t elements = 0, bytes = 0;
  if (!checked_mul_size(static_cast<size_t>(p.m), static_cast<size_t>(p.n), elements) ||
      !checked_mul_size(elements, static_cast<size_t>(split), elements) ||
      !checked_mul_size(elements, sizeof(int32_t), bytes))
    return std::numeric_limits<size_t>::max();
  return bytes;
}

BitKernelVariant bit_gemm_select_variant(const BitGemmProblem& p, const BitDeviceInfo& dev) {
  const TuneKey key = make_tune_key(p, dev);
  {
    std::lock_guard<std::mutex> lock(g_tune_mutex);
    const auto it = g_tune_cache.find(key);
    if (it != g_tune_cache.end() && variant_matches_problem(p, it->second, dev))
      return it->second;
  }
  return heuristic_variant(p, dev);
}

BitStatus bit_gemm_launch(const BitGemmProblem& p, BitKernelVariant variant, void* workspace,
                          size_t workspace_bytes, cudaStream_t stream) {
  char why[512];
  BitStatus status = bit_gemm_validate(p, why, sizeof(why));
  if (status != BV_OK) return status;

  BitDeviceInfo dev;
  status = bit_device_info(-1, &dev);
  if (status != BV_OK) return status;
  if (variant == BV_KERNEL_AUTO) variant = bit_gemm_select_variant(p, dev);
  if (!variant_matches_problem(p, variant, dev)) return BV_ERR_NO_VARIANT;
  return launch_variant_unchecked(p, variant, workspace, workspace_bytes, stream);
}

// ===========================================================================
//  Measured autotuner
// ===========================================================================
BitStatus bit_gemm_autotune(const BitGemmProblem& p, void* workspace, size_t workspace_bytes,
                            int warmup, int iters, cudaStream_t stream,
                            BitKernelVariant* best_out, float* best_ms_out) {
  if (!best_out || !best_ms_out) return BV_ERR_NULL_POINTER;
  *best_out = BV_KERNEL_AUTO;
  *best_ms_out = std::numeric_limits<float>::infinity();
  if (warmup < 0 || iters <= 0) return BV_ERR_INVALID_ARGUMENT;

  BitStatus status = bit_gemm_validate(p, nullptr, 0);
  if (status != BV_OK) return status;
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess) return BV_ERR_LAUNCH_FAILED;
  if (capture != cudaStreamCaptureStatusNone) return BV_ERR_INVALID_ARGUMENT;

  BitDeviceInfo dev;
  status = bit_device_info(-1, &dev);
  if (status != BV_OK) return status;
  const std::vector<BitKernelVariant> candidates = candidates_for(p, dev);
  if (candidates.empty()) return BV_ERR_NO_VARIANT;

  size_t last_row_offset = 0, output_elements = 0, output_bytes = 0;
  if (!checked_mul_size(static_cast<size_t>(p.m - 1),
                        static_cast<size_t>(p.ldy_or_default()), last_row_offset) ||
      !checked_add_size(last_row_offset, static_cast<size_t>(p.n), output_elements) ||
      !checked_mul_size(output_elements, dtype_bytes(p.out_dtype), output_bytes))
    return BV_ERR_BAD_SHAPE;

  void* scratch_y = nullptr;
  const bool async_alloc = dev.memory_pools_supported;
  const cudaError_t alloc_status = async_alloc ? cudaMallocAsync(&scratch_y, output_bytes, stream)
                                                : cudaMalloc(&scratch_y, output_bytes);
  if (alloc_status != cudaSuccess) return BV_ERR_LAUNCH_FAILED;
  BitGemmProblem bench = p;
  bench.y = scratch_y;

  cudaEvent_t begin = nullptr, end = nullptr;
  if (cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess) {
    if (begin) cudaEventDestroy(begin);
    if (end) cudaEventDestroy(end);
    if (async_alloc) cudaFreeAsync(scratch_y, stream);
    else cudaFree(scratch_y);
    return BV_ERR_LAUNCH_FAILED;
  }

  BitKernelVariant winner = BV_KERNEL_AUTO;
  float winner_ms = std::numeric_limits<float>::infinity();
  for (BitKernelVariant candidate : candidates) {
    const size_t need = bit_gemm_workspace_size(bench, candidate);
    if (need == std::numeric_limits<size_t>::max() || need > workspace_bytes ||
        (need > 0 && workspace == nullptr))
      continue;
    if (bench.beta != 0.0f && cudaMemsetAsync(scratch_y, 0, output_bytes, stream) != cudaSuccess)
      continue;

    bool failed = false;
    for (int i = 0; i < warmup; ++i) {
      if (launch_variant_unchecked(bench, candidate, workspace, workspace_bytes, stream) != BV_OK) {
        failed = true;
        break;
      }
    }
    if (failed) continue;

    if (cudaEventRecord(begin, stream) != cudaSuccess) continue;
    for (int i = 0; i < iters; ++i) {
      if (launch_variant_unchecked(bench, candidate, workspace, workspace_bytes, stream) != BV_OK) {
        failed = true;
        break;
      }
    }
    if (failed || cudaEventRecord(end, stream) != cudaSuccess) continue;
    if (cudaEventSynchronize(end) != cudaSuccess) continue;
    float elapsed = 0.0f;
    if (cudaEventElapsedTime(&elapsed, begin, end) != cudaSuccess) continue;
    const float mean = elapsed / static_cast<float>(iters);
    if (mean < winner_ms) {
      winner_ms = mean;
      winner = candidate;
    }
  }

  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  const cudaError_t free_status = async_alloc ? cudaFreeAsync(scratch_y, stream)
                                                : cudaFree(scratch_y);
  if (free_status != cudaSuccess) return BV_ERR_LAUNCH_FAILED;
  if (winner == BV_KERNEL_AUTO) return BV_ERR_NO_VARIANT;

  {
    std::lock_guard<std::mutex> lock(g_tune_mutex);
    g_tune_cache[make_tune_key(p, dev)] = winner;
  }
  *best_out = winner;
  *best_ms_out = winner_ms;
  return BV_OK;
}

void bit_gemm_autotune_reset() {
  std::lock_guard<std::mutex> lock(g_tune_mutex);
  g_tune_cache.clear();
}

// ===========================================================================
//  Kernel-resource and roofline models
// ===========================================================================
BitStatus bit_gemm_kernel_stats(const BitGemmProblem& p, BitKernelVariant variant,
                                BitKernelStats* out) {
  if (!out) return BV_ERR_NULL_POINTER;
  BitDeviceInfo dev;
  BitStatus status = bit_device_info(-1, &dev);
  if (status != BV_OK) return status;
  if (variant == BV_KERNEL_AUTO) variant = bit_gemm_select_variant(p, dev);
  if (!variant_matches_problem(p, variant, dev)) return BV_ERR_NO_VARIANT;

  BitKernelStats s{};
  s.variant = variant;
  if (is_gemv_variant(variant)) {
    s.block_m = std::min(p.m, 8);
    s.block_n = variant == BV_KERNEL_GEMV_WIDE ? 32 : (variant == BV_KERNEL_GEMV_SPLITK ? 16 : 8);
    s.block_k = kGemvKPerWarpStep;
    s.warp_m = s.block_m;
    s.warp_n = variant == BV_KERNEL_GEMV_WIDE ? 4 : (variant == BV_KERNEL_GEMV_SPLITK ? 2 : 1);
    s.threads = 256;
    s.stages = 1;
    s.regs_per_thread = variant == BV_KERNEL_GEMV_WIDE ? 72 : 56;
  } else {
    switch (variant) {
      case BV_KERNEL_GEMM_DP4A_64x64:
        s.block_m = 64; s.block_n = 64; s.block_k = 128; s.threads = 256; s.stages = 2;
        s.regs_per_thread = 48; break;
      case BV_KERNEL_GEMM_DP4A_128x64:
        s.block_m = 128; s.block_n = 64; s.block_k = 128; s.threads = 256; s.stages = 3;
        s.regs_per_thread = 72; break;
      case BV_KERNEL_GEMM_DP4A_64x128:
        s.block_m = 64; s.block_n = 128; s.block_k = 128; s.threads = 256; s.stages = 3;
        s.regs_per_thread = 72; break;
      case BV_KERNEL_GEMM_DP4A_128x128:
        s.block_m = 128; s.block_n = 128; s.block_k = 128; s.threads = 512; s.stages = 3;
        s.regs_per_thread = 72; break;
      case BV_KERNEL_GEMM_MMA_64x64:
        s.block_m = 64; s.block_n = 64; s.block_k = 256; s.threads = 256; s.stages = 3;
        s.regs_per_thread = 64; break;
      case BV_KERNEL_GEMM_MMA_128x64:
        s.block_m = 128; s.block_n = 64; s.block_k = 256; s.threads = 256; s.stages = 2;
        s.regs_per_thread = 80; break;
      case BV_KERNEL_GEMM_MMA_64x128:
        s.block_m = 64; s.block_n = 128; s.block_k = 256; s.threads = 256; s.stages = 3;
        s.regs_per_thread = 80; break;
      case BV_KERNEL_GEMM_MMA_128x128:
        s.block_m = 128; s.block_n = 128; s.block_k = 256; s.threads = 512; s.stages = 2;
        s.regs_per_thread = 80; break;
      default: return BV_ERR_NO_VARIANT;
    }
    if (is_mma_variant(variant)) {
      s.warp_m = variant == BV_KERNEL_GEMM_MMA_64x64 ? 16 : 32;
      s.warp_n = 32;
    } else {
      s.warp_m = 16;
      s.warp_n = 16;
    }
    if (is_dp4a_variant(variant)) {
      const int a_stage = s.block_m * (s.block_k + 16);
      const int w_stage = (s.block_k / 16) * (s.block_n + 4) * 4;
      s.smem_bytes = s.stages * (a_stage + w_stage);
    } else {
      s.smem_bytes = s.stages * s.block_m * (s.block_k + 16);
    }
  }

  int active = kMaxBlocksPerSm;
  active = std::min(active, dev.max_threads_per_sm / s.threads);
  if (s.smem_bytes > 0)
    active = std::min(active, dev.max_smem_per_sm / s.smem_bytes);
  if (s.regs_per_thread > 0) {
    const int regs_per_block = s.regs_per_thread * s.threads;
    active = std::min(active, dev.regs_per_sm / regs_per_block);
  }
  if (s.smem_bytes > dev.max_smem_per_block_optin) active = 0;
  s.max_active_blocks_per_sm = std::max(0, active);
  s.theoretical_occupancy = dev.max_threads_per_sm > 0
                                ? std::min(1.0f, static_cast<float>(active * s.threads) /
                                                     static_cast<float>(dev.max_threads_per_sm))
                                : 0.0f;
  *out = s;
  return BV_OK;
}

BitTrafficModel bit_gemm_traffic_model(const BitGemmProblem& p, const BitDeviceInfo& dev,
                                       BitKernelVariant variant) {
  BitTrafficModel model{};
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess || current_device != dev.device) return model;
  if (variant == BV_KERNEL_AUTO) variant = bit_gemm_select_variant(p, dev);
  BitKernelStats stats{};
  if (bit_gemm_kernel_stats(p, variant, &stats) != BV_OK) return model;

  const double packed_once =
      static_cast<double>(layout_num_words(p.layout, p.n_padded, p.k_padded)) * 4.0;
  const double x_once = static_cast<double>(p.m) * p.k;
  if (is_gemv_variant(variant)) {
    const int channels_per_warp = variant == BV_KERNEL_GEMV_WIDE ? 4 :
                                  (variant == BV_KERNEL_GEMV_SPLITK ? 2 : 1);
    model.weight_bytes = packed_once;
    model.activation_bytes = x_once * std::ceil(static_cast<double>(p.n) / channels_per_warp);
  } else {
    const int m_tiles = div_ceil(p.m, stats.block_m);
    const int n_tiles = div_ceil(p.n, stats.block_n);
    model.weight_bytes = packed_once * m_tiles;
    model.activation_bytes = x_once * n_tiles;
  }

  model.output_bytes = static_cast<double>(p.m) * p.n * dtype_bytes(p.out_dtype);
  if (p.beta != 0.0f) model.output_bytes *= 2.0;  // read + write
  double workspace_traffic = 0.0;
  if (variant == BV_KERNEL_GEMV_SPLITK) {
    const int split = effective_split_k(p, variant, dev);
    // Each slice writes an INT32 partial and the reduction kernel reads it.
    workspace_traffic = 2.0 * split * static_cast<double>(p.m) * p.n * sizeof(int32_t);
  }
  model.total_dram_bytes =
      model.weight_bytes + model.activation_bytes + model.output_bytes + workspace_traffic;
  model.macs = static_cast<double>(p.m) * p.n * p.k;
  if (model.total_dram_bytes > 0.0)
    model.arithmetic_intensity = model.macs / model.total_dram_bytes;
  if (dev.dram_bandwidth_gbps > 0.0)
    model.roofline_ms_bw = model.total_dram_bytes / (dev.dram_bandwidth_gbps * 1.0e9) * 1.0e3;

  // Conservative architecture model: DP4A sustains roughly 128 INT8 MACs
  // per SM-cycle; mma.sync sustains 1024 on Ampere and 2048 on Ada+.
  const double mac_per_sm_cycle = is_mma_variant(variant) ? (dev.cc >= 89 ? 2048.0 : 1024.0)
                                                          : 128.0;
  const double peak_macs = mac_per_sm_cycle * dev.sm_count * dev.core_clock_khz * 1000.0;
  if (peak_macs > 0.0) model.roofline_ms_math = model.macs / peak_macs * 1.0e3;
  return model;
}

}  // namespace bitvideo
