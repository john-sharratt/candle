#pragma once

// =============================================================================
// BLOCK TYPE DEFINITIONS
// =============================================================================
// Central definition of all block types used throughout the kernel system.
// Includes both dtype blocks (F32, F16, BF16, FP8) and quantized blocks
// (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, K-quants, AWQ).
//
// NOT included here (stay in their original locations):
//   - Compact/repacked blocks (block_c_*) in block_compact.cuh
//   - Temporary repack blocks (block_q*_t) in repack_gemx.cuh
//
// This file should be included wherever block types are needed.
// =============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// =============================================================================
// CONSTANTS
// =============================================================================

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 32
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// K-quant super-block size
#ifdef GGML_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#ifndef QK_K
#define QK_K 256
#endif
#ifndef K_SCALE_SIZE
#define K_SCALE_SIZE 12
#endif
#endif

// =============================================================================
// DTYPE BLOCKS (F32, F16, BF16, FP8)
// =============================================================================
// 32 elements per block, matching warp size for efficient processing

constexpr int DTYPE_BLOCK_SIZE = CHUNK_SIZE;  // 32 elements

struct block_f32 {
    float data[DTYPE_BLOCK_SIZE];
    static constexpr int QK = DTYPE_BLOCK_SIZE;
    static constexpr int BYTES = sizeof(float) * QK;
};
static_assert(sizeof(block_f32) == 128, "block_f32 must be 128 bytes");

struct block_f16 {
    __half data[DTYPE_BLOCK_SIZE];
    static constexpr int QK = DTYPE_BLOCK_SIZE;
    static constexpr int BYTES = sizeof(__half) * QK;
};
static_assert(sizeof(block_f16) == 64, "block_f16 must be 64 bytes");

struct block_bf16 {
    __nv_bfloat16 data[DTYPE_BLOCK_SIZE];
    static constexpr int QK = DTYPE_BLOCK_SIZE;
    static constexpr int BYTES = sizeof(__nv_bfloat16) * QK;
};
static_assert(sizeof(block_bf16) == 64, "block_bf16 must be 64 bytes");

struct block_fp8_e4m3 {
    __nv_fp8_e4m3 data[DTYPE_BLOCK_SIZE];
    static constexpr int QK = DTYPE_BLOCK_SIZE;
    static constexpr int BYTES = sizeof(__nv_fp8_e4m3) * QK;
};
static_assert(sizeof(block_fp8_e4m3) == 32, "block_fp8_e4m3 must be 32 bytes");

// =============================================================================
// SIMPLE QUANT BLOCKS (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
// =============================================================================
// GGML-style 32-element blocks with scale (and optional minimum/sum)

// Q4_0: 4-bit quantization with per-block scale
#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
typedef struct {
    half    d;              // delta (scale)
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;

// Q4_0 K/128 constants
#define QK4_0_KTILE 128
#define QR4_0_KTILE 2
#define QI4_0_KTILE 32
#define VDR_Q4_0_KTILE 2

// Q4_1: 4-bit quantization with per-block scale and minimum
#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
typedef struct {
    half2   dm;             // dm.x = delta, dm.y = min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;

// Q4_1 K/128 constants
#define QK4_1_KTILE 128
#define QR4_1_KTILE 2
#define QI4_1_KTILE 32
#define VDR_Q4_1_KTILE 2

// Q5_0: 5-bit quantization with per-block scale
#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
typedef struct {
    half d;                 // delta (scale)
    uint8_t qh[4];          // 5th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants (low 4 bits)
} block_q5_0;

// Q5_0 K/128 constants
#define QK5_0_KTILE 128
#define QR5_0_KTILE 2
#define QI5_0_KTILE 32
#define VDR_Q5_0_KTILE 2

// Q5_1: 5-bit quantization with per-block scale and minimum
#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
typedef struct {
    half2 dm;               // dm.x = delta, dm.y = min
    uint8_t qh[4];          // 5th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants (low 4 bits)
} block_q5_1;

// Q5_1 K/128 constants
#define QK5_1_KTILE 128
#define QR5_1_KTILE 2
#define QI5_1_KTILE 32
#define VDR_Q5_1_KTILE 2

// Q8_0: 8-bit quantization with per-block scale
#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
typedef struct {
    half    d;              // delta (scale)
    int8_t  qs[QK8_0];      // quants
} block_q8_0;

// Q8_0 K/128 constants
#define QK8_0_KTILE 128
#define QR8_0_KTILE 1
#define QI8_0_KTILE 16
#define VDR_Q8_0_KTILE 1

// Q8_1: 8-bit quantization with per-block scale and sum
#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
typedef struct {
    half2  ds;              // ds.x = scale, ds.y = sum
    int8_t qs[QK8_1];       // quants
} block_q8_1;

// Q8_1 K/128 constants
#define QK8_1_KTILE 128
#define QR8_1_KTILE 1
#define QI8_1_KTILE 16
#define VDR_Q8_1_KTILE 1

// Q8A128: q8 ACTIVATION block — 128-wide, contiguous int8 + ONE per-128 {scale,sum}
// header up front. The activation-ordered twin of block_c_q8_1_k128 (the interleaved
// weight form), but qs is contiguous and 16-byte aligned for a single wide cp.async and
// the standard A-fragment load. The 16-byte header holds the tile's single {scale, sum} at
// ds[0] (per-128); ds[1..3] are alignment pad, exactly aligning qs (144 = 16 + 128, both
// 16-aligned). ds[0].x = scale (amax/127, = s_a), ds[0].y = sum (Σx, the raw float sum
// feeding the asymmetric-weight min term).
#define QK8A128 128
typedef struct __align__(16) {
    half2  ds[4];           // ds[0] = the tile's {scale, sum} (per-128); ds[1..3] = align pad
    int8_t qs[QK8A128];     // contiguous, 16-byte aligned
} block_q8a128;
static_assert(sizeof(block_q8a128) == 144, "block_q8a128 must be 144 bytes");

// Q8A1024 — flat-grouped, position-independent packing of the q8a128 activation
// data. The activation tensor is a flat sequence of 128-element tiles in
// token-major order (flat_tile = token*tiles_per_row + k_tile). Eight consecutive
// tiles share one self-contained 1152-byte block (9 cache lines):
//
//   [ 1024 B : 8 × int8 qs[128] ]   slot s at +s*128    (8 lines, each 32-aligned)
//   [  128 B : 8 × half2 ds[4]  ]   slot s at +1024+s*16 (1 line; only ds[0] live, +12 B pad)
//
//   block = flat_tile >> 3 ;  slot = flat_tile & 7
//
// De-interleaving the int8 quants from the scales makes every qs tile a clean
// 128-byte run → 100% cp.async sector efficiency (vs 49% for the old 144-byte
// AoS block whose 16-byte ds header knocked qs off the 32-byte sector grid).
// The packing depends ONLY on flat_tile, so it is position-independent: the
// per-tensor (rows,cols) split is irrelevant and the n-only unified dispatch
// produces byte-identical output to the typed path. Numerics are the per-128
// {amax/127, Σx} (fp16) → bit-exact across the two dispatch paths.
#define Q8A1024_BLK_BYTES 1152
#define Q8A1024_META_OFF  1024

// byte offset of tile `flat_tile`'s int8 qs[128] within the q8a1024 buffer.
__host__ __device__ __forceinline__ int64_t q8a1024_qs_off(int64_t flat_tile) {
    return (flat_tile >> 3) * Q8A1024_BLK_BYTES + (flat_tile & 7) * 128;
}
// byte offset of tile `flat_tile`'s ds slot (ds[0] = the per-128 {scale, sum}) within the buffer.
__host__ __device__ __forceinline__ int64_t q8a1024_ds_off(int64_t flat_tile) {
    return (flat_tile >> 3) * Q8A1024_BLK_BYTES + Q8A1024_META_OFF + (flat_tile & 7) * 16;
}
// number of 1152-byte blocks needed to hold `total_tiles` 128-element tiles.
__host__ __device__ __forceinline__ int64_t q8a1024_num_blocks(int64_t total_tiles) {
    return (total_tiles + 7) >> 3;
}

// Q4_KS: 4-bit with attention-sink sub-block scaling (20 bytes)
// Sub-block A: elements 0-3 (attention sinks), fine scale sa
// Sub-block B: elements 4-31, fine scale sb
// Nibble packing: qs[k] = elem[k] | (elem[k+16] << 4), quants biased by +8
#define QK_Q4_KS 32
typedef struct {
    half    d;              // coarse scale: d = amax_all / 7.0
    uint8_t sa;             // fine scale A: sa = round(amax_A / amax_all * 255)
    uint8_t sb;             // fine scale B: sb = round(amax_B / amax_all * 255)
    uint8_t qs[QK_Q4_KS / 2]; // 32 x 4-bit quants (nibble-packed, biased +8)
} block_q4_ks;
static_assert(sizeof(block_q4_ks) == 20, "block_q4_ks size");

// Q8_KS: 8-bit with attention-sink sub-block scaling (36 bytes)
#define QK_Q8_KS 32
typedef struct {
    half    d;              // coarse scale: d = amax_all / 127.0
    uint8_t sa;             // fine scale A: sa = round(amax_A / amax_all * 255)
    uint8_t sb;             // fine scale B: sb = round(amax_B / amax_all * 255)
    int8_t  qs[QK_Q8_KS];  // 32 x 8-bit signed quants
} block_q8_ks;
static_assert(sizeof(block_q8_ks) == 36, "block_q8_ks size");

// Q2_0: 2-bit symmetric with per-block scale (10 bytes)
// 4 quants per byte (2 bits each), decode: d * (q - 1.5), d = amax / 1.5
#define QK2_0 32
typedef struct {
    half    d;              // scale: d = amax / 1.5
    uint8_t qs[QK2_0 / 4]; // 32 x 2-bit quants, 4 per byte [8 bytes]
} block_q2_0;
static_assert(sizeof(block_q2_0) == 10, "block_q2_0 size");

// Q3_0: 3-bit symmetric with per-block scale (14 bytes)
// Low 2 bits in qs (4 per byte), high bit in qh (8 per byte)
// decode: d * (q - 3.5), d = amax / 3.5
#define QK3_0 32
typedef struct {
    half    d;              // scale: d = amax / 3.5
    uint8_t qh[4];          // high (3rd) bit of each quant [4 bytes]
    uint8_t qs[QK3_0 / 4]; // low 2 bits, 4 quants per byte [8 bytes]
} block_q3_0;
static_assert(sizeof(block_q3_0) == 14, "block_q3_0 size");

// Q0: Constant block — all 32 elements reconstruct to the same INT8 value (1 byte)
// Quantize: block mean rounded to [-127, 127]. Dequant: centroid / 127.0.
// Range provided by palette outer scale; no per-block scale stored.
#define QK_Q0 32
typedef struct {
    int8_t centroid;         // INT8 constant: round(mean * 127), range [-127, 127]
} block_q0;
static_assert(sizeof(block_q0) == 1, "block_q0 size");

// Q0_V: Parametric-curve quantization (2 bytes per block, 0.50 BPE).
//
// Each block is fully self-contained — no group header, no slot-level state.
// The 16 bits decompose into three orthogonal indexes that pick a curve and
// the (centroid, scale) pair used to reconstruct it. All three are looked
// up in constant-memory tables (see q0_v_tables.cuh).
//
// Per-block layout (2 bytes total):
//   byte 0 (lo):  bits[7:0] = curve_idx     (8-bit, indexes 256-entry curve_table)
//   byte 1 (hi):  bits[4:0] = scale_idx     (5-bit, indexes 32-entry scale_table)
//                 bits[7:5] = centroid_idx  (3-bit, indexes 8 entries within
//                                            centroid_table[scale_idx])
//
// Reconstruction (outer-normalised):
//   x[e] = centroid_table[scale_idx][centroid_idx] / 32767
//        + (scale_table  [scale_idx]                / 65535)
//        * (curve_table  [curve_idx][e]             /   127)
#define QK_Q0_V 32
typedef struct {
    uint8_t lo;   // [7:0] = curve_idx
    uint8_t hi;   // [4:0] = scale_idx, [7:5] = centroid_idx
} block_q0_v;
static_assert(sizeof(block_q0_v) == 2, "block_q0_v size");

// Q0_X: Flat block with one outlier escape (2 bytes, 0.50 BPE)
// 32 elements share `bulk_anchor` (INT8 constant). One element — at
// `outlier_idx` — gets a coarse offset of `outlier_delta * Q0_X_S_OUTLIER`
// added to the anchor. Targets blocks that are nearly constant with one
// anomalous spike (attention sinks, content boundaries on V).
//
// Encode: bulk_anchor = round(mean(x)*127); outlier_idx = argmax(|x_i - bulk|);
//         outlier_delta = clamp(round(residual / S_OUTLIER), -4, 3).
// Decode: v_i8 = bulk_anchor + (i==outlier_idx ? outlier_delta * S_OUTLIER : 0)
//         clamped to [-127, 127]; x = v_i8 / 127 / outer_scale.
//
// Layout (2 bytes total):
//   byte 0: int8_t bulk_anchor          (full INT8 range, [-127..127])
//   byte 1: bits[4:0] = outlier_idx     (5 bits, 0..31)
//           bits[7:5] = outlier_delta   (signed 3-bit, two's complement, [-4..3])
//
// Q0_X_S_OUTLIER is a build-time constant (default 32 → delta range
// [-128..96] in INT8 units). Tune to match calibrated outlier magnitude.
#define QK_Q0_X 32
#define Q0_X_S_OUTLIER 32
typedef struct {
    int8_t  bulk_anchor;
    uint8_t outlier_packed;  // [4:0]=outlier_idx, [7:5]=outlier_delta (signed 3-bit)
} block_q0_x;
static_assert(sizeof(block_q0_x) == 2, "block_q0_x size");

// Q0_M2: Two-constant block with 8-bit quartet mask (3 bytes, 0.75 BPE)
// 32 elements split into 8 quartets of 4; each quartet bit selects c0 or c1.
// qmask bit i: 0 → quartet i uses c0, 1 → quartet i uses c1.
// Constants stored as INT8, dequant via centroid / 127.0.
#define QK_Q0_M2 32
typedef struct {
    int8_t  centroid[2];     // [0]=c0, [1]=c1
    uint8_t qmask;           // 8-bit mask: bit i → quartet i
} block_q0_m2;
static_assert(sizeof(block_q0_m2) == 3, "block_q0_m2 size");

// Q0_M4: Four-constant block with 32-bit pair mask (8 bytes, 2.00 BPE)
// 32 elements split into 16 pairs of 2; each 2-bit field selects one of 4 constants.
// qmask bits [2i+1:2i]: pair i constant index (0-3).
// Constants stored as INT8, dequant via centroid / 127.0.
#define QK_Q0_M4 32
typedef struct {
    int8_t   centroid[4];    // [0]=c0, [1]=c1, [2]=c2, [3]=c3
    uint32_t qmask;          // 32-bit mask: bits [2i+1:2i] → pair i
} block_q0_m4;
static_assert(sizeof(block_q0_m4) == 8, "block_q0_m4 size");

// Q1_S: 1-bit symmetric with INT8 scale (5 bytes)
// 32 sign bits in qs[4], scale as INT8 (round(mean_abs * 127))
// decode: sign_bit ? +scale/127 : -scale/127
#define QK1_S 32
typedef struct {
    int8_t  scale;          // INT8 encoded mean(|x|): round(mean_abs * 127), range [0,127]
    uint8_t qs[4];          // 32 x 1-bit signs, 8 per byte
} block_q1_s;
static_assert(sizeof(block_q1_s) == 5, "block_q1_s size");

// Q1_A: 1-bit asymmetric with separate amplitude per sign (6 bytes, 1.50 BPE)
// 32 sign bits in qs[4]; scale_pos and scale_neg are INT8-encoded means of the
// positive (incl. zero) and negative subsets respectively.
//   sign_bit = 1 → x = +scale_pos / 127
//   sign_bit = 0 → x = -scale_neg / 127
// Branchless decode via select; no warp divergence.
#define QK1_A 32
typedef struct {
    int8_t  scale_pos;      // round(mean(x_i for x_i >= 0) * 127), range [0,127]
    int8_t  scale_neg;      // round(mean(|x_i| for x_i <  0) * 127), range [0,127]
    uint8_t qs[4];          // 32 x 1-bit signs, 8 per byte (bit set = positive/zero)
} block_q1_a;
static_assert(sizeof(block_q1_a) == 6, "block_q1_a size");

// Q2_S: 2-bit symmetric with INT8 scale (9 bytes)
// 32 x 2-bit values in qs[8], scale as INT8 (round(amax/1.5 * 127))
// decode: d * (q - 1.5)  where q in [0,3], d = scale/127.0
#define QK2_S 32
typedef struct {
    int8_t  scale;          // INT8 encoded d = amax/1.5: round(d * 127), range [0,85]
    uint8_t qs[8];          // 32 x 2-bit quants, 4 per byte
} block_q2_s;
static_assert(sizeof(block_q2_s) == 9, "block_q2_s size");

// Q2_A: 2-bit asymmetric with INT8 scale + INT8 bias (10 bytes)
// 32 x 2-bit values in qs[8], scale and bias as INT8
// decode: q * (scale/127) + (bias/127)  where q in [0,3]
#define QK2_A 32
typedef struct {
    int8_t  scale;          // INT8 encoded delta = (max-min)/3: round(delta * 127), range [0,85]
    int8_t  bias;           // INT8 encoded min value: round(min * 127), range [-127,127]
    uint8_t qs[8];          // 32 x 2-bit quants, 4 per byte
} block_q2_a;
static_assert(sizeof(block_q2_a) == 10, "block_q2_a size");

// Q2_1: 2-bit asymmetric with F16 scale + F16 min (12 bytes)
// Like Q4_1 but 2-bit instead of 4-bit
// decode: q * d + m  where q in [0,3]
#define QK2_1 32
typedef struct {
    __half2 dm;             // dm.x = scale (delta), dm.y = min
    uint8_t qs[8];          // 32 x 2-bit quants, 4 per byte
} block_q2_1;
static_assert(sizeof(block_q2_1) == 12, "block_q2_1 size");

// Q3_1: 3-bit asymmetric with F16 scale + F16 min (16 bytes)
// Like Q3_0 but asymmetric with min, using 3-bit unsigned [0..7]
// Low 2 bits in qs[8], high bit in qh[4]
// decode: q * d + m  where q in [0,7]
#define QK3_1 32
typedef struct {
    __half2 dm;             // dm.x = scale (delta), dm.y = min
    uint8_t qh[4];          // high (3rd) bit of each quant [4 bytes]
    uint8_t qs[8];          // low 2 bits, 4 quants per byte [8 bytes]
} block_q3_1;
static_assert(sizeof(block_q3_1) == 16, "block_q3_1 size");

// P2: 2-bit palette index — arena routing metadata, NOT a quant.
// Each 2-bit value selects one of 4 per-head arenas for that head_dim position.
// One index covers all 32 tokens in the chunk for that head_dim slot.
// Packed 4 head_dim positions per byte (u8). head_dim must be a multiple of 4.
// Per head per chunk: head_dim / 4 bytes.
#define QK_P2 4
typedef struct {
    uint8_t packed;         // 4 × 2-bit palette indices (1 byte)
} block_p2;
static_assert(sizeof(block_p2) == 1, "block_p2 size");

// R16: Raw F16 with reserved Q-capture space (128 bytes)
// 32 x F16 primary values (64 bytes) + 32 x uint16 Q space (64 bytes)
#define QK_R16 32
typedef struct {
    half    d[QK_R16];      // 32 x F16 values (64 bytes)
    uint16_t q[QK_R16];    // 32 x reserved Q space (64 bytes)
} block_r16;
static_assert(sizeof(block_r16) == 128, "block_r16 size");

// =============================================================================
// K-QUANT BLOCKS (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K)
// =============================================================================
// Super-block quantization with 256 elements per block (QK_K)

// Aliases for K-quant block sizes
#define QK2_K QK_K
#define QK3_K QK_K
#define QK4_K QK_K
#define QK5_K QK_K
#define QK6_K QK_K

// Q2_K: 2-bit quantization with super-block scales
#define QR2_K 4
#define QI2_K (QK_K / (4 * QR2_K))
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    half2 dm;                // super-block scale for quantized scales/mins
} block_q2_K;

// Q3_K: 3-bit quantization with super-block scales
#define QR3_K 4
#define QI3_K (QK_K / (4*QR3_K))
typedef struct {
    uint8_t hmask[QK_K/8];    // quants - high bit
    uint8_t qs[QK_K/4];       // quants - low 2 bits
#ifdef GGML_QKK_64
    uint8_t scales[2];        // scales, quantized with 8 bits
#else
    uint8_t scales[K_SCALE_SIZE]; // scales, quantized with 6 bits
#endif
    half d;                   // super-block scale
} block_q3_K;

// Q4_K: 4-bit quantization with super-block scales
#define QR4_K 2
#define QI4_K (QK_K / (4*QR4_K))
#ifdef GGML_QKK_64
typedef struct {
    half    dm[2];            // super-block scales/mins
    uint8_t scales[2];        // 4-bit block scales/mins
    uint8_t qs[QK_K/2];       // 4-bit quants
} block_q4_K;
#else
typedef struct {
    half2 dm;                 // super-block scale for quantized scales/mins
    uint8_t scales[3*QK_K/64]; // scales, quantized with 6 bits
    uint8_t qs[QK_K/2];       // 4-bit quants
} block_q4_K;
#endif

// Q5_K: 5-bit quantization with super-block scales
#define QR5_K 2
#define QI5_K (QK_K / (4*QR5_K))
#ifdef GGML_QKK_64
typedef struct {
    half d;                   // super-block scale
    int8_t scales[QK_K/16];   // block scales
    uint8_t qh[QK_K/8];       // quants, high bit
    uint8_t qs[QK_K/2];       // quants, low 4 bits
} block_q5_K;
#else
typedef struct {
    half2 dm;                 // super-block scale for quantized scales/mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];       // quants, high bit
    uint8_t qs[QK_K/2];       // quants, low 4 bits
} block_q5_K;
#endif

// Q6_K: 6-bit quantization with super-block scales
#define QR6_K 2
#define QI6_K (QK_K / (4*QR6_K))
typedef struct {
    uint8_t ql[QK_K/2];       // quants, lower 4 bits
    uint8_t qh[QK_K/4];       // quants, upper 2 bits
    int8_t  scales[QK_K/16];  // scales
    half    d;                // delta
} block_q6_K;

// Q8_K: 8-bit quantization with super-block scales (256 elements per block)
// Used primarily as vec_dot partner for other K-quants (Q2K-Q6K)
// Note: d is float32, not half like other K-quants
#define QR8_K 1
#define QI8_K (QK_K / (4 * QR8_K))
typedef struct {
    float d;                  // delta (scale) - note: f32 not f16!
    int8_t qs[QK_K];          // quants (256 int8 values)
    int16_t bsums[QK_K/16];   // sum of quants in groups of 16
} block_q8_K;

// Q8_K K/128 constants
#define QK8_K_KTILE 128
#define QR8_K_KTILE 1
#define QI8_K_KTILE 16
#define VDR_Q8_K_KTILE 1

// =============================================================================
// AWQ BLOCKS (Activation-Aware Weight Quantization)
// =============================================================================
// AWQ uses 4-bit asymmetric quantization: w = scale * (q - zero)

// Q_AWQ with group size 128: 128 × 4-bit + 1 scale + 1 zero
#define QK_Q_AWQ 128
#define QR_Q_AWQ 2
#define QI_Q_AWQ (QK_Q_AWQ / (4 * QR_Q_AWQ))
// No `block_q_awq` struct: it existed only for the AWQ matmul path, which is
// gone (`QCudaStorage::fwd` refuses an AWQ weight). The KV arena reads AWQ
// through `block_c_q_awq_k128` in `quantized/block_compact.cuh`, which is a
// different — and correct — layout.

// Q_AWQ K/128 constants
#define QK_Q_AWQ_KTILE 128
#define QR_Q_AWQ_KTILE 2
#define QI_Q_AWQ_KTILE 16
#define VDR_Q_AWQ_KTILE 1

// Q_AWQ_G64 with group size 64: 64 × 4-bit + 1 scale + 1 zero per group
#define QK_Q_AWQ_G64 64
#define QR_Q_AWQ_G64 2
#define QI_Q_AWQ_G64 (QK_Q_AWQ_G64 / (4 * QR_Q_AWQ_G64))
// No `block_q_awq_g64` struct, for the same reason — and this one was the more
// wrong of the two: it declared 64 elements with a single scale/zero where the
// Rust `BlockQAWQ_G64` has 128 with `scales[2]` / `zeros[2]`.

// Q_AWQ_G64 K/128 constants
#define QK_Q_AWQ_G64_KTILE 128
#define QR_Q_AWQ_G64_KTILE 2
#define QI_Q_AWQ_G64_KTILE 16
#define VDR_Q_AWQ_G64_KTILE 1
