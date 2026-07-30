// ===========================================================================
//  GoogleTest coverage for the standalone BitVideo CUDA library
// ===========================================================================
#include "bit_gemm_kernel.h"
#include "bit_packing.h"
#include "bit_utils.h"
#include "cuda_helpers.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bitvideo {
namespace {

struct AlignedWords {
  explicit AlignedWords(int64_t word_count)
      : words(static_cast<size_t>((word_count + 3) / 4)), count(word_count) {}
  uint32_t* data() { return reinterpret_cast<uint32_t*>(words.data()); }
  const uint32_t* data() const { return reinterpret_cast<const uint32_t*>(words.data()); }
  std::vector<uint4> words;
  int64_t count;
};

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    const cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T));
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
  }
  ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  size_t size() const { return count_; }
 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

bool cuda_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template <typename T>
void copy_to_device(DeviceBuffer<T>& dst, const std::vector<T>& src) {
  ASSERT_GE(dst.size(), src.size());
  ASSERT_EQ(cudaMemcpy(dst.get(), src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice),
            cudaSuccess);
}

std::vector<int8_t> make_weights(int n, int k) {
  std::vector<int8_t> w(static_cast<size_t>(n) * k);
  for (int row = 0; row < n; ++row)
    for (int col = 0; col < k; ++col)
      w[static_cast<size_t>(row) * k + col] = static_cast<int8_t>((row * 17 + col * 7) % 3 - 1);
  return w;
}

std::vector<int8_t> make_activations(int m, int k) {
  std::vector<int8_t> x(static_cast<size_t>(m) * k);
  for (int row = 0; row < m; ++row)
    for (int col = 0; col < k; ++col)
      x[static_cast<size_t>(row) * k + col] = static_cast<int8_t>((row * 11 + col * 5) % 13 - 6);
  return x;
}

std::vector<float> reference(const std::vector<int8_t>& x, const std::vector<int8_t>& w,
                             int m, int n, int k) {
  std::vector<float> out(static_cast<size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row)
    for (int col = 0; col < n; ++col) {
      int32_t sum = 0;
      for (int kk = 0; kk < k; ++kk)
        sum += static_cast<int32_t>(x[static_cast<size_t>(row) * k + kk]) *
               static_cast<int32_t>(w[static_cast<size_t>(col) * k + kk]);
      out[static_cast<size_t>(row) * n + col] = static_cast<float>(sum);
    }
  return out;
}

void expect_equal(const std::vector<float>& actual, const std::vector<float>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) EXPECT_FLOAT_EQ(actual[i], expected[i]) << "at " << i;
}

void run_gpu_problem(int m, int n, int k, BitLayout layout, BitKernelVariant variant) {
  if (!cuda_available()) GTEST_SKIP() << "CUDA device unavailable";
  const std::vector<int8_t> x = make_activations(m, k);
  const std::vector<int8_t> w = make_weights(n, k);
  int np = 0, kp = 0;
  ASSERT_EQ(bit_host_padded_extents(layout, n, k, &np, &kp), BV_OK);
  const int64_t word_count = layout_num_words(layout, np, kp);
  AlignedWords packed_host(word_count);
  ASSERT_EQ(bit_pack_host_int8(w.data(), n, k, k, layout, packed_host.data(), np, kp), BV_OK);

  DeviceBuffer<int8_t> dx(x.size());
  DeviceBuffer<uint32_t> dw(static_cast<size_t>(word_count));
  DeviceBuffer<float> dscale(1);
  DeviceBuffer<float> dy(static_cast<size_t>(m) * n);
  ASSERT_EQ(cudaMemcpy(dx.get(), x.data(), x.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dw.get(), packed_host.data(), static_cast<size_t>(word_count) * 4,
                       cudaMemcpyHostToDevice), cudaSuccess);
  const float one = 1.0f;
  ASSERT_EQ(cudaMemcpy(dscale.get(), &one, sizeof(one), cudaMemcpyHostToDevice), cudaSuccess);

  BitGemmProblem p{};
  p.m = m; p.n = n; p.k = k;
  p.x = dx.get(); p.w_packed = dw.get(); p.layout = layout;
  p.n_padded = np; p.k_padded = kp;
  p.act_scale = dscale.get(); p.act_scale_mode = BV_ACT_SCALE_PER_TENSOR;
  p.w_scale = dscale.get(); p.w_scale_mode = BV_SCALE_PER_TENSOR;
  p.y = dy.get(); p.out_dtype = BV_OUT_FP32;
  p.ldx = k; p.ldy = n; p.split_k = 1;

  char why[512];
  ASSERT_EQ(bit_gemm_validate(p, why, sizeof(why)), BV_OK) << why;
  ASSERT_EQ(bit_gemm_launch(p, variant, nullptr, 0, nullptr), BV_OK);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> actual(static_cast<size_t>(m) * n);
  ASSERT_EQ(cudaMemcpy(actual.data(), dy.get(), actual.size() * sizeof(float),
                       cudaMemcpyDeviceToHost), cudaSuccess);
  expect_equal(actual, reference(x, w, m, n, k));
}

TEST(TernaryCodec, EveryByteDecodesCorrectly) {
  for (uint32_t byte = 0; byte < 256; ++byte) {
    const uint32_t decoded = decode4_int8_ref(byte);
    for (int j = 0; j < 4; ++j) {
      const int8_t got = reinterpret_cast<const int8_t*>(&decoded)[j];
      const int8_t expected = static_cast<int8_t>(decode_ternary((byte >> (2 * j)) & 3u));
      EXPECT_EQ(got, expected) << "byte=" << byte << " slot=" << j;
    }
  }
}

TEST(TernaryCodec, EncodeDecodeRoundTrip) {
  EXPECT_EQ(decode_ternary(encode_ternary(-7)), -1);
  EXPECT_EQ(decode_ternary(encode_ternary(0)), 0);
  EXPECT_EQ(decode_ternary(encode_ternary(9)), 1);
  EXPECT_EQ(decode_ternary(BV_CODE_RSVD), 0);
}

TEST(HostPacking, OddShapeRoundTripEveryLayout) {
  constexpr int n = 67, k = 37;
  const auto input = make_weights(n, k);
  for (BitLayout layout : {BV_LAYOUT_ROW_MAJOR, BV_LAYOUT_COLUMN_MAJOR,
                           BV_LAYOUT_BLOCKED_N64, BV_LAYOUT_MMA_INTERLEAVED}) {
    int np = 0, kp = 0;
    ASSERT_EQ(bit_host_padded_extents(layout, n, k, &np, &kp), BV_OK);
    AlignedWords packed(layout_num_words(layout, np, kp));
    ASSERT_EQ(bit_pack_host_int8(input.data(), n, k, k, layout, packed.data(), np, kp), BV_OK);
    std::vector<int8_t> output(static_cast<size_t>(n) * k);
    ASSERT_EQ(bit_unpack_host_int8(packed.data(), layout, n, k, np, kp, output.data(), k), BV_OK);
    EXPECT_EQ(output, input) << "layout=" << static_cast<int>(layout);
  }
}

TEST(HostPacking, AbsmeanAndDeadZone) {
  const std::vector<float> w = {-4.0f, -0.1f, 0.1f, 4.0f,
                                -2.0f, -1.0f, 1.0f, 2.0f};
  int np = 0, kp = 0;
  ASSERT_EQ(bit_host_padded_extents(BV_LAYOUT_ROW_MAJOR, 2, 4, &np, &kp), BV_OK);
  AlignedWords packed(layout_num_words(BV_LAYOUT_ROW_MAJOR, np, kp));
  std::array<float, 2> gamma{};
  ASSERT_EQ(bit_pack_host_f32(w.data(), 2, 4, 4, BV_LAYOUT_ROW_MAJOR,
                              BV_SCALE_PER_CHANNEL, 1e-5f, packed.data(), gamma.data(), np, kp),
            BV_OK);
  EXPECT_FLOAT_EQ(gamma[0], 2.05f);
  EXPECT_FLOAT_EQ(gamma[1], 1.5f);
  std::vector<int8_t> q(8);
  ASSERT_EQ(bit_unpack_host_int8(packed.data(), BV_LAYOUT_ROW_MAJOR, 2, 4, np, kp,
                                 q.data(), 4), BV_OK);
  const std::vector<int8_t> expected = {-1, 0, 0, 1, -1, -1, 1, 1};
  EXPECT_EQ(q, expected);
}

TEST(HostPacking, NonFiniteWeightsHaveDeterministicSemantics) {
  const float eps = 1.0e-5f;
  const std::vector<float> w = {
      std::numeric_limits<float>::quiet_NaN(), 4.0f, -4.0f, 0.0f};
  int np = 0, kp = 0;
  ASSERT_EQ(bit_host_padded_extents(BV_LAYOUT_ROW_MAJOR, 1, 4, &np, &kp), BV_OK);
  AlignedWords packed(layout_num_words(BV_LAYOUT_ROW_MAJOR, np, kp));
  float gamma = 0.0f;
  ASSERT_EQ(bit_pack_host_f32(w.data(), 1, 4, 4, BV_LAYOUT_ROW_MAJOR,
                              BV_SCALE_PER_TENSOR, eps, packed.data(), &gamma, np, kp),
            BV_OK);
  EXPECT_FLOAT_EQ(gamma, eps);
  std::vector<int8_t> q(4);
  ASSERT_EQ(bit_unpack_host_int8(packed.data(), BV_LAYOUT_ROW_MAJOR, 1, 4, np, kp,
                                 q.data(), 4), BV_OK);
  EXPECT_EQ(q, (std::vector<int8_t>{0, 1, -1, 0}));

  const float nan_eps = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(bit_pack_host_f32(w.data(), 1, 4, 4, BV_LAYOUT_ROW_MAJOR,
                              BV_SCALE_PER_TENSOR, nan_eps, packed.data(), &gamma, np, kp),
            BV_ERR_BAD_SHAPE);
}

TEST(HostPacking, ConversionPreservesCodes) {
  constexpr int n = 19, k = 71;
  const auto input = make_weights(n, k);
  int rnp = 0, rkp = 0;
  ASSERT_EQ(bit_host_padded_extents(BV_LAYOUT_ROW_MAJOR, n, k, &rnp, &rkp), BV_OK);
  AlignedWords row(layout_num_words(BV_LAYOUT_ROW_MAJOR, rnp, rkp));
  ASSERT_EQ(bit_pack_host_int8(input.data(), n, k, k, BV_LAYOUT_ROW_MAJOR,
                               row.data(), rnp, rkp), BV_OK);
  for (BitLayout layout : {BV_LAYOUT_COLUMN_MAJOR, BV_LAYOUT_BLOCKED_N64,
                           BV_LAYOUT_MMA_INTERLEAVED}) {
    int np = 0, kp = 0;
    ASSERT_EQ(bit_host_padded_extents(layout, n, k, &np, &kp), BV_OK);
    AlignedWords converted(layout_num_words(layout, np, kp));
    ASSERT_EQ(bit_convert_layout_host(row.data(), BV_LAYOUT_ROW_MAJOR, rnp, rkp,
                                      converted.data(), layout, np, kp, n, k), BV_OK);
    std::vector<int8_t> output(static_cast<size_t>(n) * k);
    ASSERT_EQ(bit_unpack_host_int8(converted.data(), layout, n, k, np, kp,
                                   output.data(), k), BV_OK);
    EXPECT_EQ(output, input);
  }
}

TEST(Validation, RejectsInvalidShapesAndExtentOverflow) {
  int np = 0, kp = 0;
  EXPECT_EQ(bit_host_padded_extents(BV_LAYOUT_ROW_MAJOR, 0, 16, &np, &kp), BV_ERR_BAD_SHAPE);
  EXPECT_EQ(bit_host_padded_extents(BV_LAYOUT_BLOCKED_N64,
                                    std::numeric_limits<int>::max(), 16, &np, &kp),
            BV_ERR_BAD_SHAPE);
  EXPECT_STREQ(bit_status_string(BV_ERR_BAD_ALIGNMENT),
               "pointer alignment does not satisfy kernel layout");
}

TEST(CudaKernels, GemvOddShape) {
  run_gpu_problem(1, 19, 73, BV_LAYOUT_ROW_MAJOR, BV_KERNEL_GEMV_1WARP);
}

TEST(CudaKernels, Dp4aGemmOddShape) {
  run_gpu_problem(17, 35, 77, BV_LAYOUT_BLOCKED_N64, BV_KERNEL_GEMM_DP4A_64x64);
}

TEST(CudaKernels, TensorCoreGemmOddShape) {
  if (!cuda_available()) GTEST_SKIP() << "CUDA device unavailable";
  BitDeviceInfo info{};
  ASSERT_EQ(bit_device_info(-1, &info), BV_OK);
  if (!info.has_mma_s8) GTEST_SKIP() << "INT8 MMA unavailable";
  run_gpu_problem(17, 19, 77, BV_LAYOUT_MMA_INTERLEAVED, BV_KERNEL_GEMM_MMA_64x64);
}

}  // namespace
}  // namespace bitvideo
