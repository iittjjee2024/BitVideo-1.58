// ===========================================================================
//  bit_gemm_kernel.cu -- BitVideo-1.58
//
//  The W1.58A8 kernels.  Three families, one dispatch entry point.
//
//  ============================ MEMORY HIERARCHY ============================
//
//     DRAM  (packed ternary W, 2 bits/weight; INT8 X)
//       |                                    <-- LDG.128 / cp.async.cg (bypass L1)
//       v
//     L2    (X is tiny and stays resident; W streams through)
//       |
//       v
//     SMEM  (X tiles only -- W is NEVER unpacked into shared memory)
//       |                                    <-- LDS.32, conflict-free by padding
//       v
//     REGS  (PRMT decode: 2-bit codes -> int8x4)
//       |
//       v
//     ALU   (DP4A on the INT32 pipe)  or  TENSOR CORES (mma.m16n8k32.s8)
//
//  Why W never touches shared memory:
//    * the packed tile is 4x smaller than the INT8 it would decode to, so
//      staging the *decoded* form would quadruple the shared footprint and
//      cut occupancy by the same factor;
//    * in the GEMV the weight stream has zero reuse, so a shared bounce is
//      pure added latency;
//    * in the MMA GEMM the interleaved layout makes a lane's whole B operand
//      one 128-bit global load, which is already the optimal access.
//    (The DP4A GEMM does stage the still-*packed* words in shared memory --
//     8x smaller than an INT8 stage -- because there the reuse factor across
//     the M direction is 8x and L1 alone would thrash.)
//
//  ============================ GEMV (M <= 8) ===============================
//
//   grid.x -> output-channel groups        grid.y -> split-K slices
//
//   CTA = WARPS warps.  Each warp owns N_PER_WARP output channels.
//
//        warp                lane 0        lane 1              lane 31
//        channel n  ->  W[n, 0..63]   W[n, 64..127]  ...   W[n, 1984..2047]
//                       X[:, 0..63]   X[:, 64..127]  ...   X[:, 1984..2047]
//                            |             |                    |
//                        16 x DP4A     16 x DP4A            16 x DP4A
//                            \             |                    /
//                             `----- __shfl_xor_sync x5 -------'
//                                        acc[m][n]
//
//   Both the weight and the activation stream are read with LDG.128, so a
//   warp moves 512 B per instruction from each -- the widest transaction the
//   memory pipe can issue.  There is no shared memory and no __syncthreads()
//   anywhere in the GEMV, hence no barrier stalls at all.
//
//  ============================ GEMM (M > 8) ================================
//
//   Threadblock tile  BM x BN,  reduction tile BK
//   Warp tile         WM x WN
//   Register tile     TM x TN  (DP4A)  /  (WM/16) x (WN/8) MMAs (tensor core)
//
//     k ->                     BK
//          +-----------------------------------+
//     BM   |   X tile (INT8) staged in SMEM    |   cp.async, 2-3 stages
//          +-----------------------------------+
//
//          +-----------------------------------+
//     BN   |   W tile (2-bit) SMEM (DP4A path) |   cp.async
//          |   or  global -> regs (MMA path)   |   LDG.128
//          +-----------------------------------+
//
//   Pipeline (STAGES = 2 shown):
//
//     iter t:   wait_group<0> | sync | issue tile t+1 | compute tile t
//                                       \____ overlapped ____/
//
// ===========================================================================
#include "bit_gemm_kernel.h"
#include "bit_utils.h"
#include "cuda_helpers.h"

#include <climits>
#include <cuda_runtime.h>

namespace bitvideo {
namespace detail {

// ===========================================================================
//  Output / epilogue helpers
// ===========================================================================
BV_DI float to_float(float v) { return v; }
BV_DI float to_float(__half v) { return __half2float(v); }
BV_DI float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

/// Fused epilogue:  y = alpha*s_a*s_w*acc + beta*y + bias
template <typename OutT>
BV_DI void epilogue_store(OutT* __restrict__ p, int32_t acc, float scale, float bias, float beta) {
  float v = fmaf(scale, static_cast<float>(acc), bias);
  if (beta != 0.0f) v = fmaf(beta, to_float(*p), v);
  *p = OutTraits<OutT>::from_float(v);
}

BV_DI float act_scale_of(const float* __restrict__ s, BitActScaleMode mode, int m) {
  if (s == nullptr) return 1.0f;
  return (mode == BV_ACT_SCALE_PER_TOKEN) ? s[m] : s[0];
}

BV_DI float w_scale_of(const float* __restrict__ s, BitScaleMode mode, int n) {
  if (s == nullptr) return 1.0f;
  return (mode == BV_SCALE_PER_CHANNEL) ? s[n] : s[0];
}

// ===========================================================================
//  GEMV  --  batch 1..8, pure memory-bandwidth play
// ===========================================================================
//
//  Template parameters
//    M_MAX        compile-time upper bound on the number of tokens (1,2,4,8)
//    KPL          ternary codes handled by one lane per step (32 or 64)
//    N_PER_WARP   output channels per warp (ILP knob: 1, 2 or 4)
//    WARPS        warps per CTA
//    SPLITK       write INT32 partials instead of running the epilogue
//
//  Register budget (M_MAX=1, KPL=64, N_PER_WARP=2):
//      16 x uint32   X fragment (64 INT8 activations)
//       4 x uint32   packed weights (one uint4 per channel)
//      16 x uint32   decoded int8x4 (transient, reused across tokens)
//       2 x int32    accumulators
//     ~14            addresses / loop state
//     ------------------------------------------------------------------
//     ~56 registers  ->  8 CTAs/SM at 256 threads  ->  100% occupancy
// ---------------------------------------------------------------------------
template <int M_MAX, int KPL, int N_PER_WARP, int WARPS, bool SPLITK, typename OutT>
__launch_bounds__(WARPS * 32, 2) __global__
void bit_gemv_kernel(const int8_t* __restrict__ x, const uint32_t* __restrict__ w, int M, int N,
                     int K, int ldx, int ldy, int64_t k_words,
                     const float* __restrict__ act_scale, BitActScaleMode act_mode,
                     const float* __restrict__ w_scale, BitScaleMode w_mode,
                     const float* __restrict__ bias, float alpha, float beta,
                     void* __restrict__ y, int32_t* __restrict__ partials, int split_k) {
  static_assert(KPL == 32 || KPL == 64, "KPL must be 32 or 64");
  static_assert(M_MAX >= 1 && M_MAX <= 8, "M_MAX must be in [1, 8]");
  static_assert(N_PER_WARP >= 1 && N_PER_WARP <= 4, "N_PER_WARP must be in [1, 4]");

  constexpr int WORDS_PER_LANE = KPL / BV_CODES_PER_U32;  // 2 or 4 uint32
  constexpr int QUADS_PER_LANE = KPL / 4;                 // 8 or 16 int8x4 groups
  constexpr int U4_PER_LANE = KPL / 16;                   // uint4 loads of X per token
  constexpr int K_PER_WARP_STEP = KPL * 32;               // 1024 or 2048 codes

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const int n_base = (blockIdx.x * WARPS + warp) * N_PER_WARP;

  // ---- K slice owned by this CTA (split-K across grid.y) ----------------
  const int slice = blockIdx.y;
  const int64_t steps_total = div_ceil64(K, K_PER_WARP_STEP);
  const int64_t steps_per_slice = div_ceil64(steps_total, split_k);
  const int64_t step_begin = static_cast<int64_t>(slice) * steps_per_slice;
  const int64_t step_end = bv_min<int64_t>(steps_total, step_begin + steps_per_slice);

  int32_t acc[M_MAX][N_PER_WARP];
#pragma unroll
  for (int m = 0; m < M_MAX; ++m)
#pragma unroll
    for (int c = 0; c < N_PER_WARP; ++c) acc[m][c] = 0;

  const int m_active = bv_min(M, M_MAX);

  for (int64_t step = step_begin; step < step_end; ++step) {
    const int64_t k0 = step * K_PER_WARP_STEP + static_cast<int64_t>(lane) * KPL;
    const int64_t kw_off = k0 / BV_CODES_PER_U32;

    // ---------------- activations: global -> registers ------------------
    uint32_t xf[M_MAX][QUADS_PER_LANE];
#pragma unroll
    for (int m = 0; m < M_MAX; ++m) {
#pragma unroll
      for (int u = 0; u < U4_PER_LANE; ++u) {
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        const int64_t koff = k0 + 16 * u;
        if (m < m_active && koff < K) {
          const int8_t* src = x + static_cast<int64_t>(m) * ldx + koff;
          if (koff + 16 <= K && is_aligned(src, 16)) {
            v = ldg128_stream(src);
          } else {
            // Ragged or unaligned row: byte gather with zero fill.  This keeps
            // odd K and arbitrary legal ldx values correct without weakening
            // the 128-bit fast path used by aligned transformer dimensions.
            const int rem = bv_min(16, static_cast<int>(K - koff));
            uint32_t tmp[4] = {0u, 0u, 0u, 0u};
#pragma unroll
            for (int b = 0; b < 16; ++b) {
              if (b < rem)
                tmp[b >> 2] |= static_cast<uint32_t>(static_cast<uint8_t>(src[b])) << (8 * (b & 3));
            }
            v = make_uint4(tmp[0], tmp[1], tmp[2], tmp[3]);
          }
        }
        xf[m][4 * u + 0] = v.x;
        xf[m][4 * u + 1] = v.y;
        xf[m][4 * u + 2] = v.z;
        xf[m][4 * u + 3] = v.w;
      }
    }

    // ---------------- weights: global -> registers -> PRMT decode -------
#pragma unroll
    for (int c = 0; c < N_PER_WARP; ++c) {
      const int n = n_base + c;
      if (n >= N) continue;

      uint32_t packed[WORDS_PER_LANE];
#pragma unroll
      for (int i = 0; i < WORDS_PER_LANE; ++i) packed[i] = 0u;

      if (kw_off + WORDS_PER_LANE <= k_words) {
        const uint32_t* src = w + static_cast<int64_t>(n) * k_words + kw_off;
        if constexpr (WORDS_PER_LANE == 4) {
          if (is_aligned(src, 16)) {
            const uint4 v = ldg128_stream(src);
            packed[0] = v.x; packed[1] = v.y; packed[2] = v.z; packed[3] = v.w;
          } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) packed[i] = __ldg(src + i);
          }
        } else {
          if (is_aligned(src, 8)) {
            const uint2 v = *reinterpret_cast<const uint2*>(src);
            packed[0] = v.x; packed[1] = v.y;
          } else {
            packed[0] = __ldg(src);
            packed[1] = __ldg(src + 1);
          }
        }
      } else {
        // Tail words (only possible when K is not a multiple of KPL*32).
#pragma unroll
        for (int i = 0; i < WORDS_PER_LANE; ++i) {
          const int64_t widx = kw_off + i;
          if (widx < k_words) packed[i] = __ldg(w + static_cast<int64_t>(n) * k_words + widx);
        }
      }

      // Decode ONCE per weight word; reuse across every token row.
#pragma unroll
      for (int wi = 0; wi < WORDS_PER_LANE; ++wi) {
        uint32_t dec[4];
        decode16_int8(packed[wi], dec);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int quad = wi * 4 + q;
#pragma unroll
          for (int m = 0; m < M_MAX; ++m) acc[m][c] = dp4a(dec[q], xf[m][quad], acc[m][c]);
        }
      }
    }
  }

  // ---------------- warp reduction (butterfly, 5 shuffle steps) ---------
#pragma unroll
  for (int m = 0; m < M_MAX; ++m)
#pragma unroll
    for (int c = 0; c < N_PER_WARP; ++c) acc[m][c] = warp_reduce_sum_i32<32>(acc[m][c]);

  if (lane != 0) return;

  if constexpr (SPLITK) {
    // partials[slice][m][n] -- INT32, so the cross-slice sum is bitwise
    // reproducible no matter what order the slices complete in.
    int32_t* dst = partials + static_cast<int64_t>(slice) * M * N;
#pragma unroll
    for (int c = 0; c < N_PER_WARP; ++c) {
      const int n = n_base + c;
      if (n >= N) continue;
      for (int m = 0; m < m_active; ++m) dst[static_cast<int64_t>(m) * N + n] = acc[m][c];
    }
  } else {
    OutT* out = reinterpret_cast<OutT*>(y);
#pragma unroll
    for (int c = 0; c < N_PER_WARP; ++c) {
      const int n = n_base + c;
      if (n >= N) continue;
      const float sw = w_scale_of(w_scale, w_mode, n);
      const float b = bias ? bias[n] : 0.0f;
      for (int m = 0; m < m_active; ++m) {
        const float sa = act_scale_of(act_scale, act_mode, m);
        epilogue_store<OutT>(out + static_cast<int64_t>(m) * ldy + n, acc[m][c], alpha * sa * sw, b,
                             beta);
      }
    }
  }
}

// ---------------------------------------------------------------------------
//  Split-K reduction + epilogue.  One thread per output element.
// ---------------------------------------------------------------------------
template <typename OutT>
__global__ void bit_splitk_reduce_kernel(const int32_t* __restrict__ partials, int M, int N,
                                         int split_k, int ldy,
                                         const float* __restrict__ act_scale,
                                         BitActScaleMode act_mode,
                                         const float* __restrict__ w_scale, BitScaleMode w_mode,
                                         const float* __restrict__ bias, float alpha, float beta,
                                         void* __restrict__ y) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(M) * N;
  if (idx >= total) return;

  const int m = static_cast<int>(idx / N);
  const int n = static_cast<int>(idx - static_cast<int64_t>(m) * N);

  int32_t sum = 0;
  const int64_t stride = static_cast<int64_t>(M) * N;
#pragma unroll 4
  for (int s = 0; s < split_k; ++s) sum += partials[s * stride + idx];

  const float sa = act_scale_of(act_scale, act_mode, m);
  const float sw = w_scale_of(w_scale, w_mode, n);
  const float b = bias ? bias[n] : 0.0f;
  epilogue_store<OutT>(reinterpret_cast<OutT*>(y) + static_cast<int64_t>(m) * ldy + n, sum,
                       alpha * sa * sw, b, beta);
}

// ===========================================================================
//  Shared-memory pipeline helpers
// ===========================================================================

/// Wait until the oldest pending cp.async group is complete.  `pending` is at
/// most three in the kernels below, so every wait count remains an immediate
/// as required by PTX.  Keeping the younger groups in flight is what overlaps
/// the next tile's DRAM latency with the current tile's DP4A/MMA instructions.
template <int STAGES>
BV_DI void wait_oldest_group(int pending) {
  static_assert(STAGES == 2 || STAGES == 3, "BitVideo pipelines use two or three stages");
  if constexpr (STAGES == 3) {
    if (pending >= 3) cp_async_wait_group<2>();
    else if (pending == 2) cp_async_wait_group<1>();
    else cp_async_wait_group<0>();
  } else {
    if (pending >= 2) cp_async_wait_group<1>();
    else cp_async_wait_group<0>();
  }
  smem_fence();
}

// ===========================================================================
//  DP4A GEMM -- blocked-N64 packed layout
// ===========================================================================
//
//  Each warp owns one or more 16x16 output tiles.  A lane owns a 2x4 register
//  micro-tile:
//
//        lane = 4*r + c, r in [0,7], c in [0,3]
//        rows = {2r, 2r+1}, columns = {4c, 4c+1, 4c+2, 4c+3}
//
//  For every group of four K values the lane loads two INT8x4 A registers,
//  decodes four packed W bytes into four INT8x4 registers, then issues eight
//  DP4As.  There is no FP operation in this loop.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES>
BV_DI void issue_dp4a_stage(const int8_t* __restrict__ x, const uint32_t* __restrict__ w,
                            int M, int K, int ldx, int Np, int Kp, int tile_m, int tile_n,
                            int k_tile, int stage, uint8_t* __restrict__ sm_a,
                            uint32_t* __restrict__ sm_w) {
  constexpr int A_PITCH = BK + 16;                 // bytes; rotates banks by 4 words
  constexpr int W_PITCH = BN + 4;                  // uint32; preserves LDG.128 alignment
  constexpr int A_VECS = BM * (BK / 16);
  constexpr int W_VECS = (BK / 16) * (BN / 4);
  constexpr int A_STAGE_BYTES = BM * A_PITCH;
  constexpr int W_STAGE_WORDS = (BK / 16) * W_PITCH;

  uint8_t* a_stage = sm_a + stage * A_STAGE_BYTES;
  uint32_t* w_stage = sm_w + stage * W_STAGE_WORDS;
  const int m0 = tile_m * BM;
  const int n0 = tile_n * BN;
  const int k0 = k_tile * BK;

  // A: [BM, BK] INT8.  One cp.async per 16-byte vector.
  for (int vec = threadIdx.x; vec < A_VECS; vec += blockDim.x) {
    const int lm = vec / (BK / 16);
    const int lk = (vec - lm * (BK / 16)) * 16;
    const int gm = m0 + lm;
    const int gk = k0 + lk;
    uint8_t* dst = a_stage + lm * A_PITCH + lk;
    const bool full = gm < M && gk + 16 <= K;
    const int8_t* src = full ? x + static_cast<int64_t>(gm) * ldx + gk : x;
    if (full && is_aligned(src, 16)) {
      cp_async<16, true>(dst, src, true);
    } else {
      // Odd K/stride tail.  This path is correctness-first and does not affect
      // aligned transformer dimensions (all production hidden sizes are /16).
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (gm < M && gk < K) {
        uint32_t words[4] = {0u, 0u, 0u, 0u};
        const int rem = bv_min(16, K - gk);
        const int8_t* row = x + static_cast<int64_t>(gm) * ldx + gk;
#pragma unroll
        for (int b = 0; b < 16; ++b)
          if (b < rem)
            words[b >> 2] |= static_cast<uint32_t>(static_cast<uint8_t>(row[b])) << (8 * (b & 3));
        v = make_uint4(words[0], words[1], words[2], words[3]);
      }
      *reinterpret_cast<uint4*>(dst) = v;
    }
  }

  // W: [(BK/16), BN] packed uint32.  BLOCKED_N64 stores four adjacent
  // channels contiguously, so every copy below is a naturally aligned 16B
  // transaction.  The padding rows/channels were zeroed by the packer.
  const int64_t k_words = packed_words(Kp);
  for (int vec = threadIdx.x; vec < W_VECS; vec += blockDim.x) {
    const int lkw = vec / (BN / 4);
    const int ln = (vec - lkw * (BN / 4)) * 4;
    const int gn = n0 + ln;
    const int64_t gkw = static_cast<int64_t>(k0 / 16) + lkw;
    uint32_t* dst = w_stage + lkw * W_PITCH + ln;
    const bool valid = gn + 3 < Np && gkw < k_words;
    const uint32_t* src = w;
    if (valid) {
      const int64_t nb = gn / BV_BLOCK_N;
      const int64_t nin = gn % BV_BLOCK_N;
      src = w + (nb * k_words + gkw) * BV_BLOCK_N + nin;
    }
    cp_async<16, true>(dst, src, valid);
  }
  cp_async_commit();
}

template <int BM, int BN, int BK, int STAGES, int WARPS, typename OutT>
__launch_bounds__(WARPS * 32, 1) __global__
void bit_gemm_dp4a_kernel(const int8_t* __restrict__ x, const uint32_t* __restrict__ w,
                          int M, int N, int K, int Np, int Kp, int ldx, int ldy,
                          const float* __restrict__ act_scale, BitActScaleMode act_mode,
                          const float* __restrict__ w_scale, BitScaleMode w_mode,
                          const float* __restrict__ bias, float alpha, float beta,
                          void* __restrict__ y, int tiles_m, int tiles_n) {
  static_assert(BM % 16 == 0 && BN % 16 == 0 && BK % 16 == 0, "DP4A tile alignment");
  constexpr int WARP_TILES = (BM / 16) * (BN / 16);
  static_assert(WARP_TILES % WARPS == 0, "warp tiles must divide evenly");
  constexpr int TILES_PER_WARP = WARP_TILES / WARPS;
  constexpr int A_PITCH = BK + 16;
  constexpr int W_PITCH = BN + 4;
  constexpr int A_STAGE_BYTES = BM * A_PITCH;
  constexpr int W_STAGE_WORDS = (BK / 16) * W_PITCH;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  uint8_t* sm_a = smem_raw;
  const size_t all_a_bytes = static_cast<size_t>(STAGES) * A_STAGE_BYTES;
  uint32_t* sm_w = reinterpret_cast<uint32_t*>(smem_raw + round_up(static_cast<int>(all_a_bytes), 16));

  int tile_m, tile_n;
  grid_swizzle(static_cast<int>(blockIdx.x), tiles_m, tiles_n, 8, tile_m, tile_n);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int lane_m = (lane >> 2) * 2;
  const int lane_n = (lane & 3) * 4;

  int32_t acc[TILES_PER_WARP][2][4];
#pragma unroll
  for (int wt = 0; wt < TILES_PER_WARP; ++wt)
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[wt][r][c] = 0;

  const int k_tiles = div_ceil(Kp, BK);
  const int preload = bv_min(STAGES, k_tiles);
  for (int s = 0; s < preload; ++s)
    issue_dp4a_stage<BM, BN, BK, STAGES>(x, w, M, K, ldx, Np, Kp, tile_m, tile_n, s, s,
                                         sm_a, sm_w);
  wait_oldest_group<STAGES>(preload);
  __syncthreads();

  for (int kt = 0; kt < k_tiles; ++kt) {
    const int stage = kt % STAGES;
    const uint8_t* a_stage = sm_a + stage * A_STAGE_BYTES;
    const uint32_t* w_stage = sm_w + stage * W_STAGE_WORDS;

#pragma unroll
    for (int q = 0; q < BK / 4; ++q) {
      const int lkw = q >> 2;
      const int shift = (q & 3) * 8;
#pragma unroll
      for (int wt = 0; wt < TILES_PER_WARP; ++wt) {
        const int warp_tile = warp + wt * WARPS;
        const int wm = warp_tile % (BM / 16);
        const int wn = warp_tile / (BM / 16);
        const int lm0 = wm * 16 + lane_m;
        const int ln0 = wn * 16 + lane_n;

        const uint32_t xa0 = *reinterpret_cast<const uint32_t*>(a_stage + lm0 * A_PITCH + q * 4);
        const uint32_t xa1 = *reinterpret_cast<const uint32_t*>(a_stage + (lm0 + 1) * A_PITCH + q * 4);

        uint32_t wb[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint32_t packed = w_stage[lkw * W_PITCH + ln0 + c];
          wb[c] = decode4_int8((packed >> shift) & 0xffu);
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          acc[wt][0][c] = dp4a(xa0, wb[c], acc[wt][0][c]);
          acc[wt][1][c] = dp4a(xa1, wb[c], acc[wt][1][c]);
        }
      }
    }

    // No thread may overwrite a stage until every warp has consumed it.
    __syncthreads();
    const int future = kt + STAGES;
    if (future < k_tiles)
      issue_dp4a_stage<BM, BN, BK, STAGES>(x, w, M, K, ldx, Np, Kp, tile_m, tile_n,
                                           future, stage, sm_a, sm_w);
    const int remaining = k_tiles - (kt + 1);
    if (remaining > 0) {
      wait_oldest_group<STAGES>(bv_min(STAGES, remaining));
      __syncthreads();
    }
  }

  // Fused scale/bias/residual epilogue.  This is the only FP arithmetic in
  // the kernel and is O(MN), while the integer reduction is O(MNK).
  OutT* out = reinterpret_cast<OutT*>(y);
#pragma unroll
  for (int wt = 0; wt < TILES_PER_WARP; ++wt) {
    const int warp_tile = warp + wt * WARPS;
    const int wm = warp_tile % (BM / 16);
    const int wn = warp_tile / (BM / 16);
    const int gm0 = tile_m * BM + wm * 16 + lane_m;
    const int gn0 = tile_n * BN + wn * 16 + lane_n;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int gm = gm0 + r;
      if (gm >= M) continue;
      const float sa = act_scale_of(act_scale, act_mode, gm);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int gn = gn0 + c;
        if (gn >= N) continue;
        const float sw = w_scale_of(w_scale, w_mode, gn);
        const float b = bias ? bias[gn] : 0.0f;
        epilogue_store<OutT>(out + static_cast<int64_t>(gm) * ldy + gn, acc[wt][r][c],
                             alpha * sa * sw, b, beta);
      }
    }
  }
}

// ===========================================================================
//  Tensor-core GEMM -- mma.sync.m16n8k32.row.col.s32.s8.s8.s32
// ===========================================================================
//
//  W stays packed all the way to the lane registers.  In the
//  MMA_INTERLEAVED layout one lane loads 16 bytes containing its B fragments
//  for an entire K=256 block (8 MMAs).  Every packed byte becomes one int8x4
//  B register through the PRMT codec; it is consumed immediately by MMA and
//  never written to shared memory.
// ---------------------------------------------------------------------------
template <int BM, int BK, int STAGES>
BV_DI void issue_mma_a_stage(const int8_t* __restrict__ x, int M, int K, int ldx, int tile_m,
                             int k_block, int stage, uint8_t* __restrict__ sm_a) {
  constexpr int A_PITCH = BK + 16;
  constexpr int A_VECS = BM * (BK / 16);
  constexpr int A_STAGE_BYTES = BM * A_PITCH;
  uint8_t* a_stage = sm_a + stage * A_STAGE_BYTES;
  const int m0 = tile_m * BM;
  const int k0 = k_block * BK;

  for (int vec = threadIdx.x; vec < A_VECS; vec += blockDim.x) {
    const int lm = vec / (BK / 16);
    const int lk = (vec - lm * (BK / 16)) * 16;
    const int gm = m0 + lm;
    const int gk = k0 + lk;
    uint8_t* dst = a_stage + lm * A_PITCH + lk;
    const bool full = gm < M && gk + 16 <= K;
    const int8_t* src = full ? x + static_cast<int64_t>(gm) * ldx + gk : x;
    if (full && is_aligned(src, 16)) {
      cp_async<16, true>(dst, src, true);
    } else {
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (gm < M && gk < K) {
        uint32_t words[4] = {0u, 0u, 0u, 0u};
        const int rem = bv_min(16, K - gk);
        const int8_t* row = x + static_cast<int64_t>(gm) * ldx + gk;
#pragma unroll
        for (int b = 0; b < 16; ++b)
          if (b < rem)
            words[b >> 2] |= static_cast<uint32_t>(static_cast<uint8_t>(row[b])) << (8 * (b & 3));
        v = make_uint4(words[0], words[1], words[2], words[3]);
      }
      *reinterpret_cast<uint4*>(dst) = v;
    }
  }
  cp_async_commit();
}

BV_DI uint32_t byte_from_u4(const uint4& v, int byte_index) {
  const uint32_t word = byte_index < 4 ? v.x : (byte_index < 8 ? v.y : (byte_index < 12 ? v.z : v.w));
  return (word >> (8 * (byte_index & 3))) & 0xffu;
}

template <int BM, int BN, int STAGES, int WARPS_M, int WARPS_N, typename OutT>
__launch_bounds__(WARPS_M * WARPS_N * 32, 1) __global__
void bit_gemm_mma_kernel(const int8_t* __restrict__ x, const uint32_t* __restrict__ w,
                         int M, int N, int K, int Np, int Kp, int ldx, int ldy,
                         const float* __restrict__ act_scale, BitActScaleMode act_mode,
                         const float* __restrict__ w_scale, BitScaleMode w_mode,
                         const float* __restrict__ bias, float alpha, float beta,
                         void* __restrict__ y, int tiles_m, int tiles_n) {
  constexpr int BK = BV_MMA_K_BLOCK;  // one interleaved lane vector = K256
  constexpr int A_PITCH = BK + 16;
  constexpr int A_STAGE_BYTES = BM * A_PITCH;
  constexpr int WARPS = WARPS_M * WARPS_N;
  constexpr int WM = BM / WARPS_M;
  constexpr int WN = BN / WARPS_N;
  constexpr int MMA_M_TILES = WM / 16;
  constexpr int MMA_N_TILES = WN / 8;
  static_assert(BM % WARPS_M == 0 && BN % WARPS_N == 0, "warp partition");
  static_assert(WM % 16 == 0 && WN % 8 == 0, "mma tile partition");
  static_assert(WARPS <= 16, "CUDA blocks support at most 32 warps; we cap at 16");

  extern __shared__ __align__(16) unsigned char smem_raw[];
  uint8_t* sm_a = smem_raw;

  int tile_m, tile_n;
  grid_swizzle(static_cast<int>(blockIdx.x), tiles_m, tiles_n, 8, tile_m, tile_n);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int warp_m = warp % WARPS_M;
  const int warp_n = warp / WARPS_M;
  const int lm_warp = warp_m * WM;
  const int ln_warp = warp_n * WN;

  int32_t acc[MMA_M_TILES][MMA_N_TILES][4];
#pragma unroll
  for (int mi = 0; mi < MMA_M_TILES; ++mi)
#pragma unroll
    for (int ni = 0; ni < MMA_N_TILES; ++ni)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[mi][ni][r] = 0;

  const int k_blocks = Kp / BK;
  const int preload = bv_min(STAGES, k_blocks);
  for (int s = 0; s < preload; ++s)
    issue_mma_a_stage<BM, BK, STAGES>(x, M, K, ldx, tile_m, s, s, sm_a);
  wait_oldest_group<STAGES>(preload);
  __syncthreads();

  const uint8_t* w_bytes = reinterpret_cast<const uint8_t*>(w);

  for (int kb = 0; kb < k_blocks; ++kb) {
    const int stage = kb % STAGES;
    const uint8_t* a_stage = sm_a + stage * A_STAGE_BYTES;

    // N-subtile outermost: its 16-byte lane vector is loaded exactly once,
    // then all eight packed bytes/pairs are consumed by eight MMAs.
#pragma unroll
    for (int ni = 0; ni < MMA_N_TILES; ++ni) {
      const int gn0 = tile_n * BN + ln_warp + ni * 8;
      uint4 packed_lane = make_uint4(0u, 0u, 0u, 0u);
      if (gn0 < Np) {
        const int gn_tile = gn0 / BV_MMA_N_TILE;
        const int64_t packed_tile = static_cast<int64_t>(gn_tile) * k_blocks + kb;
        const uint8_t* lane_src =
            w_bytes + packed_tile * BV_MMA_TILE_BYTES + lane * BV_MMA_BYTES_PER_LANE;
        packed_lane = ldg128_stream(lane_src);
      }

#pragma unroll
      for (int sub = 0; sub < BK / 32; ++sub) {
        uint32_t bfrag[2];
        bfrag[0] = decode4_int8(byte_from_u4(packed_lane, 2 * sub));
        bfrag[1] = decode4_int8(byte_from_u4(packed_lane, 2 * sub + 1));

#pragma unroll
        for (int mi = 0; mi < MMA_M_TILES; ++mi) {
          const int lm0 = lm_warp + mi * 16;
          const int lk0 = sub * 32 + t * 4;
          uint32_t afrag[4];
          afrag[0] = *reinterpret_cast<const uint32_t*>(a_stage + (lm0 + g) * A_PITCH + lk0);
          afrag[1] = *reinterpret_cast<const uint32_t*>(a_stage + (lm0 + g + 8) * A_PITCH + lk0);
          afrag[2] = *reinterpret_cast<const uint32_t*>(a_stage + (lm0 + g) * A_PITCH + lk0 + 16);
          afrag[3] = *reinterpret_cast<const uint32_t*>(a_stage + (lm0 + g + 8) * A_PITCH + lk0 + 16);
          int32_t next[4];
          mma_m16n8k32_s8(next, afrag, bfrag, acc[mi][ni]);
#pragma unroll
          for (int r = 0; r < 4; ++r) acc[mi][ni][r] = next[r];
        }
      }
    }

    __syncthreads();
    const int future = kb + STAGES;
    if (future < k_blocks)
      issue_mma_a_stage<BM, BK, STAGES>(x, M, K, ldx, tile_m, future, stage, sm_a);
    const int remaining = k_blocks - (kb + 1);
    if (remaining > 0) {
      wait_oldest_group<STAGES>(bv_min(STAGES, remaining));
      __syncthreads();
    }
  }

  OutT* out = reinterpret_cast<OutT*>(y);
#pragma unroll
  for (int mi = 0; mi < MMA_M_TILES; ++mi) {
#pragma unroll
    for (int ni = 0; ni < MMA_N_TILES; ++ni) {
      const int gm[4] = {
          tile_m * BM + lm_warp + mi * 16 + g,
          tile_m * BM + lm_warp + mi * 16 + g,
          tile_m * BM + lm_warp + mi * 16 + g + 8,
          tile_m * BM + lm_warp + mi * 16 + g + 8};
      const int gn[4] = {
          tile_n * BN + ln_warp + ni * 8 + 2 * t,
          tile_n * BN + ln_warp + ni * 8 + 2 * t + 1,
          tile_n * BN + ln_warp + ni * 8 + 2 * t,
          tile_n * BN + ln_warp + ni * 8 + 2 * t + 1};
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (gm[r] >= M || gn[r] >= N) continue;
        const float sa = act_scale_of(act_scale, act_mode, gm[r]);
        const float sw = w_scale_of(w_scale, w_mode, gn[r]);
        const float b = bias ? bias[gn[r]] : 0.0f;
        epilogue_store<OutT>(out + static_cast<int64_t>(gm[r]) * ldy + gn[r], acc[mi][ni][r],
                             alpha * sa * sw, b, beta);
      }
    }
  }
}

// ===========================================================================
//  Host launch shims (called by bit_gemm_launch.cu)
// ===========================================================================
namespace {

inline BitStatus launch_status() {
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? BV_OK : BV_ERR_LAUNCH_FAILED;
}

template <typename Kernel>
BitStatus opt_in_smem(Kernel kernel, int bytes) {
  if (bytes <= 48 * 1024) return BV_OK;
  const cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  return e == cudaSuccess ? BV_OK : BV_ERR_LAUNCH_FAILED;
}

template <typename OutT, int MMAX, int NPER, bool SPLIT>
BitStatus launch_gemv_shape(const BitGemmProblem& p, int split_k, int32_t* partials,
                            cudaStream_t stream) {
  constexpr int WARPS = 8;
  constexpr int KPL = 64;
  const dim3 block(WARPS * 32);
  const dim3 grid(div_ceil(p.n, WARPS * NPER), split_k);
  bit_gemv_kernel<MMAX, KPL, NPER, WARPS, SPLIT, OutT>
      <<<grid, block, 0, stream>>>(p.x, p.w_packed, p.m, p.n, p.k, p.ldx_or_default(),
                                   p.ldy_or_default(), packed_words(p.k_padded), p.act_scale,
                                   p.act_scale_mode, p.w_scale, p.w_scale_mode, p.bias, p.alpha,
                                   p.beta, p.y, partials, split_k);
  return launch_status();
}

template <typename OutT, int NPER, bool SPLIT>
BitStatus launch_gemv_mbucket(const BitGemmProblem& p, int split_k, int32_t* partials,
                              cudaStream_t stream) {
  if (p.m == 1) return launch_gemv_shape<OutT, 1, NPER, SPLIT>(p, split_k, partials, stream);
  if (p.m <= 2) return launch_gemv_shape<OutT, 2, NPER, SPLIT>(p, split_k, partials, stream);
  if (p.m <= 4) return launch_gemv_shape<OutT, 4, NPER, SPLIT>(p, split_k, partials, stream);
  return launch_gemv_shape<OutT, 8, NPER, SPLIT>(p, split_k, partials, stream);
}

template <typename OutT>
BitStatus launch_split_reduce(const BitGemmProblem& p, int split_k, const int32_t* partials,
                              cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(p.m) * p.n;
  bit_splitk_reduce_kernel<OutT><<<div_ceil64(count, 256), 256, 0, stream>>>(
      partials, p.m, p.n, split_k, p.ldy_or_default(), p.act_scale, p.act_scale_mode,
      p.w_scale, p.w_scale_mode, p.bias, p.alpha, p.beta, p.y);
  return launch_status();
}

template <int BM, int BN, int BK, int STAGES, int WARPS, typename OutT>
BitStatus launch_dp4a_shape(const BitGemmProblem& p, cudaStream_t stream) {
  constexpr int A_STAGE_BYTES = BM * (BK + 16);
  constexpr int W_STAGE_WORDS = (BK / 16) * (BN + 4);
  constexpr int SMEM = round_up(STAGES * A_STAGE_BYTES, 16) + STAGES * W_STAGE_WORDS * 4;
  auto kernel = bit_gemm_dp4a_kernel<BM, BN, BK, STAGES, WARPS, OutT>;
  BitStatus status = opt_in_smem(kernel, SMEM);
  if (status != BV_OK) return status;
  const int tm = div_ceil(p.m, BM);
  const int tn = div_ceil(p.n, BN);
  const int64_t blocks = static_cast<int64_t>(tm) * tn;
  if (blocks > INT_MAX) return BV_ERR_BAD_SHAPE;
  kernel<<<static_cast<unsigned>(blocks), WARPS * 32, SMEM, stream>>>(
      p.x, p.w_packed, p.m, p.n, p.k, p.n_padded, p.k_padded, p.ldx_or_default(),
      p.ldy_or_default(), p.act_scale, p.act_scale_mode, p.w_scale, p.w_scale_mode, p.bias,
      p.alpha, p.beta, p.y, tm, tn);
  return launch_status();
}

template <int BM, int BN, int STAGES, int WARPS_M, int WARPS_N, typename OutT>
BitStatus launch_mma_shape(const BitGemmProblem& p, cudaStream_t stream) {
  constexpr int SMEM = STAGES * BM * (BV_MMA_K_BLOCK + 16);
  auto kernel = bit_gemm_mma_kernel<BM, BN, STAGES, WARPS_M, WARPS_N, OutT>;
  BitStatus status = opt_in_smem(kernel, SMEM);
  if (status != BV_OK) return status;
  const int tm = div_ceil(p.m, BM);
  const int tn = div_ceil(p.n, BN);
  const int64_t blocks = static_cast<int64_t>(tm) * tn;
  if (blocks > INT_MAX) return BV_ERR_BAD_SHAPE;
  kernel<<<static_cast<unsigned>(blocks), WARPS_M * WARPS_N * 32, SMEM, stream>>>(
      p.x, p.w_packed, p.m, p.n, p.k, p.n_padded, p.k_padded, p.ldx_or_default(),
      p.ldy_or_default(), p.act_scale, p.act_scale_mode, p.w_scale, p.w_scale_mode, p.bias,
      p.alpha, p.beta, p.y, tm, tn);
  return launch_status();
}

}  // namespace

BitStatus launch_gemv_kernel(const BitGemmProblem& p, BitKernelVariant variant, int split_k,
                             void* workspace, cudaStream_t stream) {
  const bool split = split_k > 1;
  if (split && workspace == nullptr) return BV_ERR_WORKSPACE_TOO_SMALL;
  int32_t* partials = split ? reinterpret_cast<int32_t*>(workspace) : nullptr;
  BitStatus s = BV_ERR_UNSUPPORTED_DTYPE;

#define BV_LAUNCH_GEMV_FOR_TYPE(OUT_T)                                                        \
  do {                                                                                         \
    if (split)                                                                                 \
      s = launch_gemv_mbucket<OUT_T, 2, true>(p, split_k, partials, stream);                   \
    else if (variant == BV_KERNEL_GEMV_WIDE)                                                   \
      s = launch_gemv_mbucket<OUT_T, 4, false>(p, 1, nullptr, stream);                         \
    else                                                                                        \
      s = launch_gemv_mbucket<OUT_T, 1, false>(p, 1, nullptr, stream);                         \
    if (s == BV_OK && split) s = launch_split_reduce<OUT_T>(p, split_k, partials, stream);     \
  } while (0)

  switch (p.out_dtype) {
    case BV_OUT_FP32: BV_LAUNCH_GEMV_FOR_TYPE(float); break;
    case BV_OUT_FP16: BV_LAUNCH_GEMV_FOR_TYPE(__half); break;
    case BV_OUT_BF16: BV_LAUNCH_GEMV_FOR_TYPE(__nv_bfloat16); break;
    default: s = BV_ERR_UNSUPPORTED_DTYPE; break;
  }
#undef BV_LAUNCH_GEMV_FOR_TYPE
  return s;
}

template <typename OutT>
BitStatus launch_dp4a_variant_typed(const BitGemmProblem& p, BitKernelVariant v,
                                    cudaStream_t stream) {
  constexpr int BK = 128;
  switch (v) {
    case BV_KERNEL_GEMM_DP4A_64x64:
      return launch_dp4a_shape<64, 64, BK, 2, 8, OutT>(p, stream);
    case BV_KERNEL_GEMM_DP4A_128x64:
      return launch_dp4a_shape<128, 64, BK, 3, 8, OutT>(p, stream);
    case BV_KERNEL_GEMM_DP4A_64x128:
      return launch_dp4a_shape<64, 128, BK, 3, 8, OutT>(p, stream);
    case BV_KERNEL_GEMM_DP4A_128x128:
      return launch_dp4a_shape<128, 128, BK, 3, 16, OutT>(p, stream);
    default: return BV_ERR_NO_VARIANT;
  }
}

BitStatus launch_dp4a_kernel(const BitGemmProblem& p, BitKernelVariant v, cudaStream_t stream) {
  switch (p.out_dtype) {
    case BV_OUT_FP32: return launch_dp4a_variant_typed<float>(p, v, stream);
    case BV_OUT_FP16: return launch_dp4a_variant_typed<__half>(p, v, stream);
    case BV_OUT_BF16: return launch_dp4a_variant_typed<__nv_bfloat16>(p, v, stream);
    default: return BV_ERR_UNSUPPORTED_DTYPE;
  }
}

template <typename OutT>
BitStatus launch_mma_variant_typed(const BitGemmProblem& p, BitKernelVariant v,
                                   cudaStream_t stream) {
  switch (v) {
    case BV_KERNEL_GEMM_MMA_64x64:
      return launch_mma_shape<64, 64, 3, 4, 2, OutT>(p, stream);
    case BV_KERNEL_GEMM_MMA_128x64:
      return launch_mma_shape<128, 64, 2, 4, 2, OutT>(p, stream);
    case BV_KERNEL_GEMM_MMA_64x128:
      return launch_mma_shape<64, 128, 3, 2, 4, OutT>(p, stream);
    case BV_KERNEL_GEMM_MMA_128x128:
      return launch_mma_shape<128, 128, 2, 4, 4, OutT>(p, stream);
    default: return BV_ERR_NO_VARIANT;
  }
}

BitStatus launch_mma_kernel(const BitGemmProblem& p, BitKernelVariant v, cudaStream_t stream) {
  switch (p.out_dtype) {
    case BV_OUT_FP32: return launch_mma_variant_typed<float>(p, v, stream);
    case BV_OUT_FP16: return launch_mma_variant_typed<__half>(p, v, stream);
    case BV_OUT_BF16: return launch_mma_variant_typed<__nv_bfloat16>(p, v, stream);
    default: return BV_ERR_UNSUPPORTED_DTYPE;
  }
}

}  // namespace detail
}  // namespace bitvideo
