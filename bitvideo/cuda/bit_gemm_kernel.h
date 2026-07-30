// ===========================================================================
//  bit_gemm_kernel.h -- BitVideo-1.58
//
//  Public interface of the W1.58A8 kernel library.  Deliberately free of any
//  PyTorch / ATen types so the kernels can be linked into TensorRT plugins,
//  ggml-style runtimes, or the standalone C++ benchmark.
//
//  MATH
//  ----
//  Given
//      X  : [M, K] INT8      activations, symmetric per-token (or per-tensor)
//      W  : [N, K] ternary   weights in {-1, 0, +1}, packed 2 bits/weight
//      s_a: [M] or [1]       activation dequant scale  (x_real ~= q * s_a)
//      s_w: [N] or [1]       weight scale gamma        (w_real ~= c * s_w)
//      b  : [N] or null      bias in the output dtype's compute precision
//
//  we compute, with an INT32 accumulator and an FP32 epilogue,
//
//      acc[m,n] = sum_k  X[m,k] * W[n,k]                      (INT32, exact)
//      Y[m,n]   = alpha * s_a[m] * s_w[n] * acc[m,n]
//               + beta  * Y_in[m,n]
//               + b[n]
//
//  `beta != 0` fuses a residual add; `alpha` folds any extra global constant
//  (e.g. 1/127 if the caller keeps the activation scale in "absmax" form).
//  There is NO floating point multiply inside the reduction loop -- the only
//  FP work is the per-output epilogue, which costs O(M*N) instead of O(M*N*K).
//
//  KERNEL FAMILY
//  -------------
//      GEMV  (M <= 8) : DRAM-bandwidth bound.  One warp per output channel
//                       group, split-K across CTAs, warp-shuffle reduction.
//      GEMM  (M >  8) : compute bound.  Threadblock/warp/register tiling,
//                       cp.async multi-stage pipeline.  Two math back-ends:
//                         * DP4A  -- INT32 pipe, no layout constraints
//                         * MMA   -- INT8 tensor cores (mma.m16n8k32.s8)
//
//  ERROR MODEL
//  -----------
//  Launchers return `BitStatus`.  They never throw, never print, and never
//  call cudaDeviceSynchronize().  Argument validation happens here (cheap,
//  host side) so the PyTorch bridge can turn a status into a TORCH_CHECK
//  message with full context.
// ===========================================================================
#pragma once

#include "bit_utils.h"
#include "cuda_helpers.h"

#include <cstdint>

namespace bitvideo {

// ---------------------------------------------------------------------------
//  Status codes
// ---------------------------------------------------------------------------
enum BitStatus : int32_t {
  BV_OK = 0,
  BV_ERR_NULL_POINTER = 1,
  BV_ERR_BAD_SHAPE = 2,
  BV_ERR_BAD_ALIGNMENT = 3,
  BV_ERR_UNSUPPORTED_DTYPE = 4,
  BV_ERR_UNSUPPORTED_LAYOUT = 5,
  BV_ERR_UNSUPPORTED_ARCH = 6,
  BV_ERR_WORKSPACE_TOO_SMALL = 7,
  BV_ERR_LAUNCH_FAILED = 8,
  BV_ERR_NO_VARIANT = 9,
  BV_ERR_INVALID_ARGUMENT = 10,
};

const char* bit_status_string(BitStatus s);

// ---------------------------------------------------------------------------
//  Output element type
// ---------------------------------------------------------------------------
enum BitOutDtype : int32_t {
  BV_OUT_FP32 = 0,
  BV_OUT_FP16 = 1,
  BV_OUT_BF16 = 2,
};

const char* bit_dtype_string(BitOutDtype d);

/// How the weight scale vector should be interpreted.
enum BitScaleMode : int32_t {
  BV_SCALE_PER_TENSOR = 0,  // s_w points at a single float
  BV_SCALE_PER_CHANNEL = 1, // s_w points at N floats (one per output channel)
};

/// How the activation scale vector should be interpreted.
enum BitActScaleMode : int32_t {
  BV_ACT_SCALE_PER_TENSOR = 0,  // s_a points at a single float
  BV_ACT_SCALE_PER_TOKEN = 1,   // s_a points at M floats (one per row of X)
};

// ---------------------------------------------------------------------------
//  Kernel variants
//
//  Naming: <family>_<mathpipe>_<BM>x<BN>[_sK]
//  The autotuner picks one of these; a caller may pin a variant for
//  reproducible benchmarking or for CUDA-graph capture.
// ---------------------------------------------------------------------------
enum BitKernelVariant : int32_t {
  BV_KERNEL_AUTO = 0,

  // ---- GEMV family (M <= 8) -------------------------------------------
  BV_KERNEL_GEMV_1WARP = 10,   // 1 warp per output channel, no split-K
  BV_KERNEL_GEMV_SPLITK = 11,  // split-K + deterministic two-pass reduce
  BV_KERNEL_GEMV_WIDE = 12,    // 4 output channels per warp, wider ILP

  // ---- GEMM, DP4A math -------------------------------------------------
  BV_KERNEL_GEMM_DP4A_64x64 = 20,
  BV_KERNEL_GEMM_DP4A_128x64 = 21,
  BV_KERNEL_GEMM_DP4A_128x128 = 22,
  BV_KERNEL_GEMM_DP4A_64x128 = 23,

  // ---- GEMM, INT8 tensor core math ------------------------------------
  BV_KERNEL_GEMM_MMA_64x64 = 30,
  BV_KERNEL_GEMM_MMA_128x64 = 31,
  BV_KERNEL_GEMM_MMA_128x128 = 32,
  BV_KERNEL_GEMM_MMA_64x128 = 33,

  BV_KERNEL_VARIANT_COUNT = 34,
};

const char* bit_variant_string(BitKernelVariant v);

/// True when the variant needs BV_LAYOUT_MMA_INTERLEAVED weights.
bool bit_variant_requires_mma_layout(BitKernelVariant v);
/// True when the variant needs BV_LAYOUT_BLOCKED_N64 weights.
bool bit_variant_requires_blocked_layout(BitKernelVariant v);
/// Preferred weight layout for a variant.
BitLayout bit_variant_preferred_layout(BitKernelVariant v);
/// Minimum compute capability (major*10+minor) required by a variant.
int bit_variant_min_arch(BitKernelVariant v);

// ---------------------------------------------------------------------------
//  Device description (cached; safe to call on every launch)
// ---------------------------------------------------------------------------
struct BitDeviceInfo {
  int device = -1;
  int cc = 0;                    // major*10 + minor, e.g. 86, 89, 90, 120
  int sm_count = 0;
  int max_threads_per_sm = 0;
  int max_grid_x = 0;
  int max_smem_per_block_optin = 0;  // bytes, after cudaFuncSetAttribute
  int max_smem_per_sm = 0;
  int regs_per_sm = 0;
  int l2_cache_bytes = 0;
  int core_clock_khz = 0;
  int warp_size = 32;
  double dram_bandwidth_gbps = 0.0;  // theoretical peak from clock * bus width
  bool has_cp_async = false;
  bool has_mma_s8 = false;
  bool cooperative_launch = false;
  bool memory_pools_supported = false;
  char name[128] = {0};
};

/// Query (and memoise) the properties of `device` (-1 = current device).
BitStatus bit_device_info(int device, BitDeviceInfo* out);

// ---------------------------------------------------------------------------
//  Problem description
// ---------------------------------------------------------------------------
struct BitGemmProblem {
  // ---- shapes ----------------------------------------------------------
  int m = 0;  // tokens (batch * sequence)
  int n = 0;  // output channels
  int k = 0;  // input channels (reduction length)

  // ---- operands --------------------------------------------------------
  const int8_t* x = nullptr;        // [m, ldx] INT8, row major
  const uint32_t* w_packed = nullptr;  // packed ternary weights
  BitLayout layout = BV_LAYOUT_ROW_MAJOR;

  // Padded logical extents of the packed weight buffer.  For ROW_MAJOR these
  // equal (n, k); for the tiled layouts they are (n, k) rounded up to the
  // layout alignment.  Rows/columns in the padding are guaranteed zero.
  int n_padded = 0;
  int k_padded = 0;

  // ---- scales / epilogue ----------------------------------------------
  const float* act_scale = nullptr;  // [m] or [1]
  BitActScaleMode act_scale_mode = BV_ACT_SCALE_PER_TOKEN;
  const float* w_scale = nullptr;    // [n] or [1]
  BitScaleMode w_scale_mode = BV_SCALE_PER_TENSOR;
  const float* bias = nullptr;       // [n] or null, FP32
  float alpha = 1.0f;
  float beta = 0.0f;

  // ---- output ----------------------------------------------------------
  void* y = nullptr;  // [m, ldy]
  BitOutDtype out_dtype = BV_OUT_FP16;

  // ---- strides (elements, not bytes) ----------------------------------
  int ldx = 0;  // default: k
  int ldy = 0;  // default: n

  // ---- execution knobs -------------------------------------------------
  int split_k = 0;      // 0 = let the heuristic decide, 1 = disabled
  bool deterministic = true;  // split-K uses a two-pass reduce instead of atomics

  BV_HDI int ldx_or_default() const { return ldx == 0 ? k : ldx; }
  BV_HDI int ldy_or_default() const { return ldy == 0 ? n : ldy; }
};

// ---------------------------------------------------------------------------
//  Launch API
// ---------------------------------------------------------------------------

/// Validate a problem without launching anything.  Fills `why` (optional,
/// >= 256 chars) with a human readable explanation on failure.
BitStatus bit_gemm_validate(const BitGemmProblem& p, char* why, int why_len);

/// Bytes of scratch required by `variant` for `p`.  0 means "no workspace".
size_t bit_gemm_workspace_size(const BitGemmProblem& p, BitKernelVariant variant);

/// Heuristic (and, when enabled, measured) variant selection.
BitKernelVariant bit_gemm_select_variant(const BitGemmProblem& p, const BitDeviceInfo& dev);

/// Launch.  `variant == BV_KERNEL_AUTO` runs the selector first.
BitStatus bit_gemm_launch(const BitGemmProblem& p, BitKernelVariant variant, void* workspace,
                          size_t workspace_bytes, cudaStream_t stream);

/// Measure every legal variant once and cache the winner for this
/// (arch, M, N, K, dtype) key.  Subsequent BV_KERNEL_AUTO launches reuse it.
BitStatus bit_gemm_autotune(const BitGemmProblem& p, void* workspace, size_t workspace_bytes,
                            int warmup, int iters, cudaStream_t stream,
                            BitKernelVariant* best_out, float* best_ms_out);

/// Clear the autotune cache (used by the benchmark harness).
void bit_gemm_autotune_reset();

// ---------------------------------------------------------------------------
//  Packing / layout conversion (implemented in ternary_gemm.cu)
// ---------------------------------------------------------------------------

/// Per-output-channel or per-tensor absmean of |W|, then ternarise+pack in a
/// single pass.  `w` is FP32/FP16/BF16 [n, k] row major.
///
///   gamma  = mean(|W|)            (per tensor or per row, see mode)
///   thresh = 0.5 * gamma
///   code   = +1 if w >  thresh, -1 if w < -thresh, else 0
///
/// `gamma_out` receives the scale(s) actually used (1 or n floats).
BitStatus bit_ternarize_and_pack(const void* w, BitOutDtype w_dtype, int n, int k, int ldw,
                                 BitLayout layout, BitScaleMode scale_mode, float eps,
                                 uint32_t* w_packed, float* gamma_out, int n_padded, int k_padded,
                                 cudaStream_t stream);

/// Pack an already-ternarised INT8 tensor (values in {-1,0,1}).
BitStatus bit_pack_int8_ternary(const int8_t* w, int n, int k, int ldw, BitLayout layout,
                                uint32_t* w_packed, int n_padded, int k_padded,
                                cudaStream_t stream);

/// Unpack back to INT8 {-1,0,1} -- used by tests and by the ONNX exporter.
BitStatus bit_unpack_to_int8(const uint32_t* w_packed, BitLayout layout, int n, int k, int n_padded,
                            int k_padded, int8_t* w_out, int ldw, cudaStream_t stream);

/// Convert between any two packed layouts (device to device).
BitStatus bit_convert_layout(const uint32_t* src, BitLayout src_layout, uint32_t* dst,
                             BitLayout dst_layout, int n, int k, int n_padded_src, int k_padded_src,
                             int n_padded_dst, int k_padded_dst, cudaStream_t stream);

/// Sparsity / distribution statistics over a packed buffer.
/// `counts` receives {num_zero, num_pos, num_neg} as int64.
BitStatus bit_packed_stats(const uint32_t* w_packed, int64_t num_words, int64_t* counts,
                          cudaStream_t stream);

// ---------------------------------------------------------------------------
//  Activation quantisation (implemented in int8_gemm.cu)
// ---------------------------------------------------------------------------

/// Dynamic symmetric INT8 quantisation of `x` [m, k].
///   scale[m] = max(|x[m,:]|) / 127
///   q[m,k]   = round(x[m,k] / scale[m])   clamped to [-127, 127]
/// `granularity`: 0 = per tensor, 1 = per token (row), 2 = per 128-wide group.
BitStatus bit_quantize_activations(const void* x, BitOutDtype x_dtype, int m, int k, int ldx,
                                   int granularity, float eps, int8_t* q_out, int ldq,
                                   float* scale_out, cudaStream_t stream);

/// Plain INT8 x INT8 -> INT32 GEMM with the same FP epilogue.  Used for the
/// cross-attention K/V projections that stay 8-bit, and as the numerical
/// reference for the ternary path.  `a_scale_mode` selects a scalar or one
/// scale per row; `b_scale_mode` selects a scalar or one scale per output
/// channel.  Groupwise activation scales are intentionally not accepted by
/// this API because they require a separately scaled accumulator per K group.
BitStatus bit_int8_gemm(const int8_t* a, const int8_t* b, int m, int n, int k, int lda, int ldb,
                        const float* a_scale, BitActScaleMode a_scale_mode,
                        const float* b_scale, BitScaleMode b_scale_mode, const float* bias,
                        float alpha, float beta, void* c, BitOutDtype c_dtype, int ldc,
                        bool b_is_nk, cudaStream_t stream);

// ---------------------------------------------------------------------------
//  Introspection helpers used by the benchmark report generator
// ---------------------------------------------------------------------------
struct BitKernelStats {
  BitKernelVariant variant = BV_KERNEL_AUTO;
  int block_m = 0, block_n = 0, block_k = 0;
  int warp_m = 0, warp_n = 0;
  int threads = 0;
  int stages = 0;
  int regs_per_thread = 0;      // conservative compiled-kernel estimate
  int smem_bytes = 0;
  int max_active_blocks_per_sm = 0;  // resource-model estimate
  float theoretical_occupancy = 0.f;
};

/// Query the compiled attributes + occupancy of a variant for this problem.
BitStatus bit_gemm_kernel_stats(const BitGemmProblem& p, BitKernelVariant variant,
                                BitKernelStats* out);

/// Analytic traffic model, used by docs/BENCHMARK_REPORT.md.
struct BitTrafficModel {
  double weight_bytes = 0;      // packed weights read from DRAM (compulsory)
  double activation_bytes = 0;  // X reads (may be amplified by split-N)
  double output_bytes = 0;      // Y writes
  double total_dram_bytes = 0;
  double macs = 0;              // multiply-accumulates
  double arithmetic_intensity = 0;  // MAC / byte
  double roofline_ms_bw = 0;    // lower bound from bandwidth
  double roofline_ms_math = 0;  // lower bound from INT8/DP4A peak
};

BitTrafficModel bit_gemm_traffic_model(const BitGemmProblem& p, const BitDeviceInfo& dev,
                                       BitKernelVariant variant);

}  // namespace bitvideo
