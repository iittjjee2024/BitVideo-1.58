// ===========================================================================
//  bit_utils.h -- BitVideo-1.58
//
//  The ternary (1.58-bit) weight codec, the four supported packed memory
//  layouts, and the register-only decode paths.
//
//  ---------------------------------------------------------------------
//  1. ENCODING
//  ---------------------------------------------------------------------
//  Two bits per weight, exactly as specified by the project requirements:
//
//        code | value | int8 payload
//        -----+-------+--------------
//        0b00 |   0   | 0x00
//        0b01 |  +1   | 0x01
//        0b10 |  -1   | 0xFF
//        0b11 | rsvd  | 0x00      (treated as zero; never emitted by the packer)
//
//  16 weights per uint32, 4 weights per uint8.  Code j of a uint32 occupies
//  bits [2j+1 : 2j], so byte b of the word holds weights 4b .. 4b+3 with the
//  lowest k in the least significant bit pair.  That ordering is chosen so a
//  decoded int8x4 register lines up byte-for-byte with a little-endian 32-bit
//  load of four consecutive INT8 activations -- which is what DP4A and the
//  mma.s8 B-fragment both want.
//
//  Storage: 2 bits/weight = 16x smaller than FP32, 8x smaller than FP16.
//  (log2(3) = 1.585 bits is the information-theoretic floor; 2 bits is the
//  aligned, decode-free-ish practical choice.  See "future improvements" in
//  docs/RESEARCH_REPORT.md for the 5-trits-in-8-bits base-3 variant.)
//
//  ---------------------------------------------------------------------
//  2. REGISTER DECODE  (the interesting part)
//  ---------------------------------------------------------------------
//  Requirement: unpack ONLY in registers, using AND / SHIFT / XOR /
//  predication -- no branches, no floating point, never unpack into shared
//  memory (that would triple the shared footprint and re-introduce the
//  bandwidth we just saved).
//
//  Naive decode of 4 weights costs ~12 ALU ops (4x shift, 4x and, 4x select).
//  We do it in 6 by turning the problem into a byte permute:
//
//      byte  b = c3 c2 c1 c0                      (4 x 2-bit codes)
//
//      (a) spread the 2-bit codes into 4-bit nibbles
//              s = (b | (b << 4)) & 0x0F0F        ; codes 1,0 -> byte0
//                                                 ; codes 3,2 -> byte1
//              s = (s | (s << 2)) & 0x3333        ; one code per nibble
//
//      (b) use the nibbles as PRMT byte selectors into a 4-entry LUT
//              LUT = 0x00FF0100  (byte0=0x00, byte1=0x01, byte2=0xFF, byte3=0x00)
//              r   = prmt.b32(LUT, LUT, s)        ; == __byte_perm
//
//      -> r is an int8x4 register holding {w0,w1,w2,w3} as signed bytes.
//
//  Cost: 2 shifts + 2 ORs + 2 ANDs + 1 PRMT = 7 instructions for 4 weights
//  (1.75 IPW), and the AND immediates fold into LOP3 so ptxas typically emits
//  5.  Compare with the alternative of storing pre-spread nibbles in memory:
//  that halves the ALU cost but doubles the weight traffic, and the kernel is
//  DRAM bound, so it is strictly worse.
//
//  Why not "two DP4A with 0/1 masks"?  You can compute
//      acc = dp4a(pos_mask, x, 0) - dp4a(neg_mask, x, 0)
//  which skips the LUT entirely, but it doubles the number of DP4A/MMA
//  issues.  On the tensor-core path that halves peak throughput, so we only
//  keep it as a reference/verification path (`ternary_masks_from_byte`).
//
//  ---------------------------------------------------------------------
//  3. LAYOUTS
//  ---------------------------------------------------------------------
//  See `BitLayout` below.  Recommendation and measurements are in
//  docs/CUDA_OPTIMIZATION.md; the short version:
//
//    GEMV (batch 1)  -> ROW_MAJOR      lanes walk K, warp reads 128B/lane-group
//    GEMM DP4A       -> BLOCKED_N64    CTA weight tile is fully contiguous
//    GEMM tensor core-> MMA_INTERLEAVED lane's mma B-fragment is one LDG.128
//    reference/bench -> COLUMN_MAJOR   deliberately bad, kept for the ablation
// ===========================================================================
#pragma once

#include "cuda_helpers.h"

#include <cstdint>

namespace bitvideo {

// ---------------------------------------------------------------------------
//  Codec constants
// ---------------------------------------------------------------------------
enum : uint32_t {
  BV_CODE_ZERO = 0u,  // 0b00 ->  0
  BV_CODE_POS = 1u,   // 0b01 -> +1
  BV_CODE_NEG = 2u,   // 0b10 -> -1
  BV_CODE_RSVD = 3u,  // 0b11 -> reserved, decoded as 0
};

/// Weights per storage unit.
enum : int {
  BV_CODES_PER_BYTE = 4,
  BV_CODES_PER_U32 = 16,
  BV_BITS_PER_CODE = 2,
};

/// PRMT lookup table: byte i is the int8 payload of code i.
///   byte0 = 0x00 (code 00 ->  0)
///   byte1 = 0x01 (code 01 -> +1)
///   byte2 = 0xFF (code 10 -> -1)
///   byte3 = 0x00 (code 11 -> reserved)
static constexpr uint32_t BV_TERNARY_LUT = 0x00FF0100u;

/// Same LUT with the sign flipped -- used by the transposed/negated epilogue
/// and by the "subtract" formulation of the reference path.
static constexpr uint32_t BV_TERNARY_LUT_NEG = 0x000100FFu;

// ---------------------------------------------------------------------------
//  Scalar encode / decode (host and device, used by the packer and by tests)
// ---------------------------------------------------------------------------
BV_HDI uint32_t encode_ternary(int v) {
  // v is expected in {-1,0,+1}; anything else is clamped by sign.
  return (v > 0) ? BV_CODE_POS : ((v < 0) ? BV_CODE_NEG : BV_CODE_ZERO);
}

BV_HDI int decode_ternary(uint32_t code) {
  // 00 -> 0, 01 -> +1, 10 -> -1, 11 -> 0
  const int is_pos = static_cast<int>((code & 3u) == BV_CODE_POS);
  const int is_neg = static_cast<int>((code & 3u) == BV_CODE_NEG);
  return is_pos - is_neg;
}

/// Extract code j (0..15) from a packed uint32.
BV_HDI uint32_t get_code(uint32_t word, int j) { return (word >> (2 * j)) & 3u; }

/// Insert code (0..3) at position j (0..15) of a packed uint32.
BV_HDI uint32_t set_code(uint32_t word, int j, uint32_t code) {
  const uint32_t shift = static_cast<uint32_t>(2 * j);
  return (word & ~(3u << shift)) | ((code & 3u) << shift);
}

/// Number of uint32 words needed to hold `k` ternary weights.
BV_HDI constexpr int64_t packed_words(int64_t k) { return (k + BV_CODES_PER_U32 - 1) / BV_CODES_PER_U32; }

/// Number of bytes needed to hold `k` ternary weights (uint32 granularity).
BV_HDI constexpr int64_t packed_bytes(int64_t k) { return packed_words(k) * 4; }

// ---------------------------------------------------------------------------
//  Register decode: 4 codes (one byte) -> int8x4
// ---------------------------------------------------------------------------

/// Spread four 2-bit codes held in the low byte of `code8` into four nibbles.
///   in : 0000 0000 0000 0000 0000 0000 c3c3 c2c2c1c1 c0c0   (low 8 bits used)
///   out: 0000 0000 0000 0000 00c3 00c2 00c1 00c0
/// Precondition: bits [31:8] of `code8` must be zero.
BV_HDI uint32_t spread_codes_to_nibbles(uint32_t code8) {
  uint32_t s = (code8 | (code8 << 4)) & 0x0F0Fu;  // codes {1,0} -> byte0, {3,2} -> byte1
  s = (s | (s << 2)) & 0x3333u;                   // one code per nibble
  return s;
}

/// Decode 4 ternary codes (low byte of `code8`) into a signed int8x4 register.
/// Result byte i == int8 value of code i, ready for DP4A / mma.s8.
BV_DI uint32_t decode4_int8(uint32_t code8) {
  const uint32_t sel = spread_codes_to_nibbles(code8 & 0xFFu);
#if defined(__CUDA_ARCH__)
  uint32_t r;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(BV_TERNARY_LUT), "r"(0u), "r"(sel));
  return r;
#else
  uint32_t r = 0;
  for (int i = 0; i < 4; ++i) {
    const uint32_t nib = (sel >> (4 * i)) & 0xFu;
    const uint32_t byte = (BV_TERNARY_LUT >> (8 * nib)) & 0xFFu;
    r |= byte << (8 * i);
  }
  return r;
#endif
}

/// Host-visible (non-inline-asm) mirror of decode4_int8 for CPU reference code.
BV_HDI uint32_t decode4_int8_ref(uint32_t code8) {
  const uint32_t sel = spread_codes_to_nibbles(code8 & 0xFFu);
  uint32_t r = 0;
  for (int i = 0; i < 4; ++i) {
    const uint32_t nib = (sel >> (4 * i)) & 0xFu;
    const uint32_t byte = (BV_TERNARY_LUT >> (8 * nib)) & 0xFFu;
    r |= byte << (8 * i);
  }
  return r;
}

/// Decode a full packed uint32 (16 codes) into four int8x4 registers.
/// out[b] holds weights 4b .. 4b+3.
BV_DI void decode16_int8(uint32_t word, uint32_t (&out)[4]) {
#pragma unroll
  for (int b = 0; b < 4; ++b) out[b] = decode4_int8((word >> (8 * b)) & 0xFFu);
}

/// Decode a 128-bit packed chunk (64 codes) into sixteen int8x4 registers.
BV_DI void decode64_int8(const uint4& w, uint32_t (&out)[16]) {
  const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int b = 0; b < 4; ++b) out[4 * i + b] = decode4_int8((words[i] >> (8 * b)) & 0xFFu);
  }
}

// ---------------------------------------------------------------------------
//  Alternative decode: 0/1 selection masks (reference + ablation path)
// ---------------------------------------------------------------------------
/// Produce two int8x4 registers whose bytes are 1 where the weight is +1 /
/// -1 respectively and 0 elsewhere.  Enables the branch-free
///     acc = dp4a_us(pos, x, acc);  acc = -dp4a_us(neg, x, -acc);
/// formulation, and is also the cheapest way to *count* non-zeros.
BV_DI void ternary_masks_from_byte(uint32_t code8, uint32_t& pos, uint32_t& neg) {
  const uint32_t b = code8 & 0xFFu;
  // Bit 0 of each code -> "+1" flag; bit 1 -> "-1" flag.
  const uint32_t p = b & 0x55u;
  const uint32_t n = (b >> 1) & 0x55u;
  // Spread bits {0,2,4,6} into bytes {0,8,16,24}: multiply by
  // (1 + 2^6 + 2^12 + 2^18) then mask.  No cross-term lands on a kept bit.
  pos = (p * 0x00041041u) & 0x01010101u;
  neg = (n * 0x00041041u) & 0x01010101u;
}

/// Population count of non-zero weights in a packed word -- used by the
/// sparsity statistics kernel and by the packer's validation pass.
BV_HDI int count_nonzero_codes(uint32_t word) {
  const uint32_t lo = word & 0x55555555u;         // bit0 of every code
  const uint32_t hi = (word >> 1) & 0x55555555u;  // bit1 of every code
  const uint32_t nz = lo | hi;                    // code != 00
#if defined(__CUDA_ARCH__)
  return __popc(nz);
#else
  int c = 0;
  uint32_t v = nz;
  while (v) { v &= (v - 1); ++c; }
  return c;
#endif
}

/// Sum of the ternary values in a packed word (needed for the
/// zero-point / bias correction term when activations are asymmetric).
BV_HDI int sum_codes(uint32_t word) {
  const uint32_t lo = word & 0x55555555u;
  const uint32_t hi = (word >> 1) & 0x55555555u;
  const uint32_t pos = lo & ~hi;  // 01
  const uint32_t neg = hi & ~lo;  // 10
#if defined(__CUDA_ARCH__)
  return __popc(pos) - __popc(neg);
#else
  int cp = 0, cn = 0;
  uint32_t v = pos;
  while (v) { v &= (v - 1); ++cp; }
  v = neg;
  while (v) { v &= (v - 1); ++cn; }
  return cp - cn;
#endif
}

// ===========================================================================
//  LAYOUTS
// ===========================================================================
/// Packed weight memory layouts.  All of them store the same codes; they
/// differ only in *where*.
enum BitLayout : int32_t {
  /// [N][K/16] uint32, row (output-channel) major.
  ///  + trivially convertible, human readable, cache friendly for GEMV
  ///  + one warp assigned to one output row walks K contiguously
  ///  - a GEMM CTA reading BN rows issues BN separate short bursts
  BV_LAYOUT_ROW_MAJOR = 0,

  /// [K/16][N] uint32, K major.  Kept only for the ablation study: a GEMV
  /// warp now strides by N*4 bytes per step, which destroys spatial locality.
  BV_LAYOUT_COLUMN_MAJOR = 1,

  /// [N/64][K/16][64] uint32.  N is blocked by 64 so that, for a fixed k
  /// word, the 64 output channels of a block are contiguous (256 B).  A CTA
  /// weight tile is therefore one contiguous region and the K loop is a pure
  /// sequential stream -> peak DRAM efficiency for the DP4A GEMM.
  BV_LAYOUT_BLOCKED_N64 = 2,

  /// Fragment-interleaved layout for `mma.sync.m16n8k32.s8`.
  ///
  ///   tile = (n_tile of 8 output channels) x (k_block of 256 reductions)
  ///   each tile stores 32 lanes x 16 B = 512 B, fully contiguous
  ///   lane L = 4*g + t  (g = n within tile, t = k quarter)
  ///   byte  = 2*sub_tile + half   (sub_tile = k32 index, half = k16 index)
  ///
  /// A lane's entire B-operand for eight consecutive MMAs is ONE LDG.128.
  /// This is the ternary analogue of Marlin's permuted INT4 layout.
  BV_LAYOUT_MMA_INTERLEAVED = 3,
};

/// Tile geometry of BV_LAYOUT_MMA_INTERLEAVED.
enum : int {
  BV_MMA_N_TILE = 8,     // mma N
  BV_MMA_K_TILE = 32,    // mma K
  BV_MMA_K_BLOCK = 256,  // 8 MMAs worth of K per 128-bit lane load
  BV_MMA_LANES = 32,
  BV_MMA_BYTES_PER_LANE = 16,
  BV_MMA_TILE_BYTES = BV_MMA_LANES * BV_MMA_BYTES_PER_LANE,  // 512
};

/// Tile geometry of BV_LAYOUT_BLOCKED_N64.
enum : int { BV_BLOCK_N = 64 };

/// Alignment requirements per layout, expressed as (n_multiple, k_multiple).
BV_HDI void layout_alignment(BitLayout layout, int& n_mult, int& k_mult) {
  switch (layout) {
    case BV_LAYOUT_ROW_MAJOR:      n_mult = 1;             k_mult = 16;               break;
    case BV_LAYOUT_COLUMN_MAJOR:   n_mult = 1;             k_mult = 16;               break;
    case BV_LAYOUT_BLOCKED_N64:    n_mult = BV_BLOCK_N;    k_mult = 16;               break;
    case BV_LAYOUT_MMA_INTERLEAVED:n_mult = BV_MMA_N_TILE; k_mult = BV_MMA_K_BLOCK;   break;
    default:                       n_mult = 1;             k_mult = 16;               break;
  }
}

/// Total number of uint32 words in a packed buffer for the given layout.
/// `n` and `k` must already be padded to `layout_alignment`.
BV_HDI int64_t layout_num_words(BitLayout layout, int64_t n, int64_t k) {
  switch (layout) {
    case BV_LAYOUT_ROW_MAJOR:
    case BV_LAYOUT_COLUMN_MAJOR:
    case BV_LAYOUT_BLOCKED_N64:
      return n * packed_words(k);
    case BV_LAYOUT_MMA_INTERLEAVED:
      return (n / BV_MMA_N_TILE) * (k / BV_MMA_K_BLOCK) * (BV_MMA_TILE_BYTES / 4);
    default:
      return n * packed_words(k);
  }
}

// ---------------------------------------------------------------------------
//  Address arithmetic: where does weight (n, k) live?
//
//  Returns the *word* index and the *code slot* (0..15) inside that word.
//  These functions are the single source of truth shared by the host packer,
//  the device packer, and the gtest reference -- if they are right, every
//  layout is right.
// ---------------------------------------------------------------------------
struct CodeAddr {
  int64_t word;  // index into the uint32 buffer
  int slot;      // 0..15, code position inside the word
};

BV_HDI CodeAddr code_addr_row_major(int64_t n, int64_t k, int64_t /*N*/, int64_t K) {
  const int64_t kw = packed_words(K);
  CodeAddr a;
  a.word = n * kw + (k / BV_CODES_PER_U32);
  a.slot = static_cast<int>(k % BV_CODES_PER_U32);
  return a;
}

BV_HDI CodeAddr code_addr_column_major(int64_t n, int64_t k, int64_t N, int64_t /*K*/) {
  CodeAddr a;
  a.word = (k / BV_CODES_PER_U32) * N + n;
  a.slot = static_cast<int>(k % BV_CODES_PER_U32);
  return a;
}

BV_HDI CodeAddr code_addr_blocked_n64(int64_t n, int64_t k, int64_t /*N*/, int64_t K) {
  const int64_t kw = packed_words(K);
  const int64_t nb = n / BV_BLOCK_N;
  const int64_t nin = n % BV_BLOCK_N;
  CodeAddr a;
  a.word = (nb * kw + (k / BV_CODES_PER_U32)) * BV_BLOCK_N + nin;
  a.slot = static_cast<int>(k % BV_CODES_PER_U32);
  return a;
}

BV_HDI CodeAddr code_addr_mma_interleaved(int64_t n, int64_t k, int64_t /*N*/, int64_t K) {
  const int64_t k_blocks = K / BV_MMA_K_BLOCK;
  const int64_t n_tile = n / BV_MMA_N_TILE;
  const int g = static_cast<int>(n % BV_MMA_N_TILE);  // B column inside the mma tile

  const int64_t k_block = k / BV_MMA_K_BLOCK;
  const int kk = static_cast<int>(k % BV_MMA_K_BLOCK);  // 0..255
  const int sub_tile = kk / BV_MMA_K_TILE;              // 0..7  (which mma)
  const int kkk = kk % BV_MMA_K_TILE;                   // 0..31 (K inside mma)
  const int half = kkk / 16;                            // 0 -> reg0, 1 -> reg1
  const int pos = kkk % 16;                             // 0..15
  const int t = pos / 4;                                // lane quarter
  const int j = pos % 4;                                // byte lane inside reg

  const int lane = 4 * g + t;
  const int byte_in_lane = 2 * sub_tile + half;  // 0..15

  const int64_t tile_index = n_tile * k_blocks + k_block;  // N-major: sequential K stream
  const int64_t byte_offset =
      tile_index * BV_MMA_TILE_BYTES + static_cast<int64_t>(lane) * BV_MMA_BYTES_PER_LANE +
      byte_in_lane;

  CodeAddr a;
  a.word = byte_offset / 4;
  a.slot = static_cast<int>((byte_offset % 4) * BV_CODES_PER_BYTE + j);
  return a;
}

BV_HDI CodeAddr code_addr(BitLayout layout, int64_t n, int64_t k, int64_t N, int64_t K) {
  switch (layout) {
    case BV_LAYOUT_ROW_MAJOR:       return code_addr_row_major(n, k, N, K);
    case BV_LAYOUT_COLUMN_MAJOR:    return code_addr_column_major(n, k, N, K);
    case BV_LAYOUT_BLOCKED_N64:     return code_addr_blocked_n64(n, k, N, K);
    case BV_LAYOUT_MMA_INTERLEAVED: return code_addr_mma_interleaved(n, k, N, K);
    default:                        return code_addr_row_major(n, k, N, K);
  }
}

// ---------------------------------------------------------------------------
//  Generic (slow, correctness-oriented) accessors used by packers and tests
// ---------------------------------------------------------------------------
BV_HDI void store_code(uint32_t* buf, BitLayout layout, int64_t n, int64_t k, int64_t N, int64_t K,
                       uint32_t code) {
  const CodeAddr a = code_addr(layout, n, k, N, K);
  buf[a.word] = set_code(buf[a.word], a.slot, code);
}

BV_HDI uint32_t load_code(const uint32_t* buf, BitLayout layout, int64_t n, int64_t k, int64_t N,
                          int64_t K) {
  const CodeAddr a = code_addr(layout, n, k, N, K);
  return get_code(buf[a.word], a.slot);
}

// ---------------------------------------------------------------------------
//  Activation quantisation helpers (shared by kernels and the reference)
// ---------------------------------------------------------------------------
/// Symmetric absmax INT8 quantiser: q = round(x * (127 / amax)), clamped.
BV_DI int8_t quantize_int8_rn(float x, float inv_scale) {
  const float s = x * inv_scale;
#if defined(__CUDA_ARCH__)
  int q = __float2int_rn(s);
#else
  int q = static_cast<int>(s < 0.f ? s - 0.5f : s + 0.5f);
#endif
  q = bv_max(-127, bv_min(127, q));
  return static_cast<int8_t>(q);
}

/// Pack four quantised activations into one int8x4 register.
BV_DI uint32_t pack_int8x4(int8_t a, int8_t b, int8_t c, int8_t d) {
  return (static_cast<uint32_t>(static_cast<uint8_t>(a))) |
         (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24);
}

/// Ternarisation threshold from the BitNet b1.58 absmean rule.
///   gamma = mean(|W|);  W_q = clamp(round(W / gamma), -1, 1)
/// which is equivalent to a dead zone of  |W| < gamma/2  ->  0.
BV_HDI float ternary_threshold_from_absmean(float absmean, float eps = 1e-5f) {
  const float g = absmean > eps ? absmean : eps;
  return 0.5f * g;
}

BV_HDI int ternarize_with_threshold(float w, float thresh) {
  const int pos = static_cast<int>(w > thresh);
  const int neg = static_cast<int>(w < -thresh);
  return pos - neg;
}

}  // namespace bitvideo
