// SPDX-License-Identifier: MIT
// Q0_V — Per-Block Parametric-Curve Quantization: codebook tables.
//
// K and V are calibrated SEPARATELY: each side gets its own codebook fitted
// to its own statistics. Kernels select between them at compile time via
// `template <bool IS_K>`. Per side:
//
//   1. q0_v_curve_base2_<side>[4][64]         — the four base curves (i8,
//                                               [-127, +127]), each stored
//                                               twice in a row. 256 B. The
//                                               encoder reads the same rows
//                                               as floats (base2f, 1 KB) and
//                                               each base's Σ base² (energy).
//   2. q0_v_scale_table_bits_<side>[32]       — 32-entry scale codebook stored
//                                               as f16 BIT PATTERNS (u16). The
//                                               stored value is `scale_norm /
//                                               127`, so the curve's `/127`
//                                               normalisation is pre-baked into
//                                               scale. 64 B.
//   3. q0_v_centroid_table_bits_<side>[32][16] — per-scale centroid codebook,
//                                               f16 bit patterns (u16) in
//                                               [-1, +1]. 1 KB.
//
// The 128 curves a block can name are signed rotations of the four bases:
//
//   curve[b·16 + p][e] = ±base[b & 3][(e + 2p) mod 32],   − for bucket b ≥ 4
//
// The full 128-curve tables are `CURVE_TABLE_K` / `CURVE_TABLE_V` in
// candle-core's k_quants.rs, the reference codec's; the structure is pinned by
// candle-core/tests/q0_v_curve_structure.rs, and the decode and encode oracle
// tests hold every GPU path to the reference, bit for bit.
//
// Every table lives in plain global memory, read through the read-only cache:
// the decoder's lanes index them per block — a warp's lanes decode different
// blocks — and so do the encoder's (one scale entry, one centroid entry, one
// (base, phase) pair per lane). Constant memory serves one address per cycle,
// so divergent indices would serialise.
//
// Per-block 2-byte layout (16 bits, little-endian):
//   bits[0..6]   = curve_idx
//   bits[7..11]  = scale_idx
//   bits[12..15] = centroid_idx
//
// Decode for element e (single FMA, ZERO constant multiplications):
//   curve     = ±base2[(curve_idx >> 4) & 3][e + 2·(curve_idx & 15)]
//   scale     = __half2float(__ushort_as_half(q0_v_scale_table_bits_<side>[scale_idx]))
//   centroid  = __half2float(__ushort_as_half(q0_v_centroid_table_bits_<side>[scale_idx][centroid_idx]))
//   x[e]      = __fmaf_rn(scale, curve, centroid)
//
// where `scale` already has the curve's `/127` factor folded in. The element
// loaders divide by `outer_scale` to recover the original signed value.

#pragma once

#include <stdint.h>
#include <cuda_fp16.h>

// =============================================================================
// K-SIDE TABLES
// =============================================================================

__device__ static const uint16_t q0_v_scale_table_bits_k[32] = {
        0x01D0,0x0F3A,0x10B7,0x11BB,0x12BE,0x13B5,0x144C,0x14B4,
        0x1514,0x1570,0x15C8,0x161C,0x1670,0x16C4,0x1719,0x1770,
        0x17CB,0x1815,0x1846,0x187B,0x18B3,0x18F1,0x1935,0x1981,
        0x19D7,0x1A3B,0x1AB3,0x1B49,0x1C0D,0x1CAE,0x1DB6,0x2008
};

__device__ static const uint16_t q0_v_centroid_table_bits_k[32][16] = {
      /* scale_idx =  0  (8253 src blocks) */ { 0xA86D,0xA86D,0x9D30,0x9D30,0x99F0,0x99F0,0x9380,0x9380,0x1400,0x1400,0x1980,0x1980,0x1CD0,0x1CD0,0x26B2,0x26B2 },
      /* scale_idx =  1  (44790 src blocks) */ { 0xACA1,0xACA1,0xA2E4,0xA2E4,0x9FE0,0x9FE0,0x9B20,0x9B20,0x11C0,0x11C0,0x1D70,0x1D70,0x21D0,0x21D0,0x2CE8,0x2CE8 },
      /* scale_idx =  2  (37123 src blocks) */ { 0xAE1E,0xAE1E,0xA452,0xA452,0xA114,0xA114,0x9C40,0x9C40,0x1680,0x1680,0x1FF0,0x1FF0,0x23D0,0x23D0,0x2D1F,0x2D1F },
      /* scale_idx =  3  (35842 src blocks) */ { 0xAF54,0xAF54,0xA4F6,0xA4F6,0xA180,0xA180,0x9C20,0x9C20,0x1930,0x1930,0x20F8,0x20F8,0x24B0,0x24B0,0x2FB0,0x2FB0 },
      /* scale_idx =  4  (36114 src blocks) */ { 0xB0C9,0xB0C9,0xA5DC,0xA5DC,0xA248,0xA248,0x9C58,0x9C58,0x1AC0,0x1AC0,0x21D0,0x21D0,0x25A6,0x25A6,0x3012,0x3012 },
      /* scale_idx =  5  (35808 src blocks) */ { 0xB258,0xB258,0xA698,0xA698,0xA2DC,0xA2DC,0x9C70,0x9C70,0x1C68,0x1C68,0x22E4,0x22E4,0x26A2,0x26A2,0x3028,0x3028 },
      /* scale_idx =  6  (36158 src blocks) */ { 0xB152,0xB152,0xA780,0xA780,0xA3A4,0xA3A4,0x9CA0,0x9CA0,0x1D70,0x1D70,0x23E0,0x23E0,0x2796,0x2796,0x3050,0x3050 },
      /* scale_idx =  7  (36233 src blocks) */ { 0xB167,0xB167,0xA830,0xA830,0xA43A,0xA43A,0x9D10,0x9D10,0x1DC8,0x1DC8,0x2464,0x2464,0x2828,0x2828,0x33A3,0x33A3 },
      /* scale_idx =  8  (36056 src blocks) */ { 0xB194,0xB194,0xA874,0xA874,0xA494,0xA494,0x9DA0,0x9DA0,0x1E70,0x1E70,0x24BA,0x24BA,0x2879,0x2879,0x3283,0x3283 },
      /* scale_idx =  9  (36005 src blocks) */ { 0xB15D,0xB15D,0xA8D2,0xA8D2,0xA4EC,0xA4EC,0x9DE8,0x9DE8,0x1E98,0x1E98,0x2516,0x2516,0x28D3,0x28D3,0x31D8,0x31D8 },
      /* scale_idx = 10  (36228 src blocks) */ { 0xB1F8,0xB1F8,0xA90E,0xA90E,0xA544,0xA544,0x9E80,0x9E80,0x1F18,0x1F18,0x2554,0x2554,0x2910,0x2910,0x31F4,0x31F4 },
      /* scale_idx = 11  (36307 src blocks) */ { 0xB220,0xB220,0xA96C,0xA96C,0xA5AE,0xA5AE,0x9F38,0x9F38,0x1F30,0x1F30,0x259E,0x259E,0x295E,0x295E,0x3220,0x3220 },
      /* scale_idx = 12  (35970 src blocks) */ { 0xB2F0,0xB2F0,0xA9B8,0xA9B8,0xA610,0xA610,0x9FD0,0x9FD0,0x1FB8,0x1FB8,0x2604,0x2604,0x299D,0x299D,0x3281,0x3281 },
      /* scale_idx = 13  (36282 src blocks) */ { 0xB2EA,0xB2EA,0xA9F4,0xA9F4,0xA66A,0xA66A,0x9FF0,0x9FF0,0x1FF0,0x1FF0,0x2634,0x2634,0x29D9,0x29D9,0x334A,0x334A },
      /* scale_idx = 14  (36018 src blocks) */ { 0xB2CF,0xB2CF,0xAA13,0xAA13,0xA682,0xA682,0xA03C,0xA03C,0x200C,0x200C,0x2652,0x2652,0x2A07,0x2A07,0x333F,0x333F },
      /* scale_idx = 15  (35878 src blocks) */ { 0xB441,0xB441,0xAA6D,0xAA6D,0xA6FE,0xA6FE,0xA0A8,0xA0A8,0x1FE0,0x1FE0,0x2680,0x2680,0x2A2A,0x2A2A,0x33A0,0x33A0 },
      /* scale_idx = 16  (36336 src blocks) */ { 0xB452,0xB452,0xAA9F,0xAA9F,0xA724,0xA724,0xA0B4,0xA0B4,0x2044,0x2044,0x26D6,0x26D6,0x2A82,0x2A82,0x3358,0x3358 },
      /* scale_idx = 17  (36077 src blocks) */ { 0xB477,0xB477,0xAAE4,0xAAE4,0xA770,0xA770,0xA10C,0xA10C,0x1FD0,0x1FD0,0x26F6,0x26F6,0x2AB7,0x2AB7,0x32C5,0x32C5 },
      /* scale_idx = 18  (36093 src blocks) */ { 0xB4D9,0xB4D9,0xAB0F,0xAB0F,0xA7AC,0xA7AC,0xA138,0xA138,0x2054,0x2054,0x2740,0x2740,0x2AEA,0x2AEA,0x3368,0x3368 },
      /* scale_idx = 19  (36184 src blocks) */ { 0xB532,0xB532,0xAB65,0xAB65,0xA7FA,0xA7FA,0xA134,0xA134,0x20A4,0x20A4,0x2784,0x2784,0x2B21,0x2B21,0x34AB,0x34AB },
      /* scale_idx = 20  (36038 src blocks) */ { 0xB4D7,0xB4D7,0xABC2,0xABC2,0xA816,0xA816,0xA160,0xA160,0x20B4,0x20B4,0x27D0,0x27D0,0x2B7C,0x2B7C,0x3495,0x3495 },
      /* scale_idx = 21  (36245 src blocks) */ { 0xB5E6,0xB5E6,0xAC04,0xAC04,0xA830,0xA830,0xA158,0xA158,0x212C,0x212C,0x2826,0x2826,0x2BDD,0x2BDD,0x342C,0x342C },
      /* scale_idx = 22  (36260 src blocks) */ { 0xB53A,0xB53A,0xAC2E,0xAC2E,0xA86E,0xA86E,0xA1F4,0xA1F4,0x2118,0x2118,0x283B,0x283B,0x2C13,0x2C13,0x34A4,0x34A4 },
      /* scale_idx = 23  (35892 src blocks) */ { 0xB510,0xB510,0xAC6B,0xAC6B,0xA8B6,0xA8B6,0xA240,0xA240,0x2170,0x2170,0x2879,0x2879,0x2C38,0x2C38,0x34C9,0x34C9 },
      /* scale_idx = 24  (36229 src blocks) */ { 0xB598,0xB598,0xACA8,0xACA8,0xA8EA,0xA8EA,0xA2B4,0xA2B4,0x2198,0x2198,0x28AF,0x28AF,0x2C7F,0x2C7F,0x34ED,0x34ED },
      /* scale_idx = 25  (36251 src blocks) */ { 0xB593,0xB593,0xAD19,0xAD19,0xA95A,0xA95A,0xA364,0xA364,0x2198,0x2198,0x28D8,0x28D8,0x2CB2,0x2CB2,0x3601,0x3601 },
      /* scale_idx = 26  (36332 src blocks) */ { 0xB591,0xB591,0xAD7D,0xAD7D,0xA9F2,0xA9F2,0xA42E,0xA42E,0x2210,0x2210,0x291F,0x291F,0x2D0D,0x2D0D,0x35E3,0x35E3 },
      /* scale_idx = 27  (36591 src blocks) */ { 0xB675,0xB675,0xADD5,0xADD5,0xAA53,0xAA53,0xA494,0xA494,0x21F8,0x21F8,0x297D,0x297D,0x2D75,0x2D75,0x360A,0x360A },
      /* scale_idx = 28  (36833 src blocks) */ { 0xB6DC,0xB6DC,0xAE72,0xAE72,0xAAF1,0xAAF1,0xA4CC,0xA4CC,0x2364,0x2364,0x2A71,0x2A71,0x2E3A,0x2E3A,0x3639,0x3639 },
      /* scale_idx = 29  (35066 src blocks) */ { 0xB722,0xB722,0xB00D,0xB00D,0xAC2E,0xAC2E,0xA54A,0xA54A,0x254C,0x254C,0x2C43,0x2C43,0x3012,0x3012,0x3746,0x3746 },
      /* scale_idx = 30  (47819 src blocks) */ { 0xB6D2,0xB6D2,0xB278,0xB278,0xB0AF,0xB0AF,0xA9D6,0xA9D6,0x2ABE,0x2ABE,0x3096,0x3096,0x3263,0x3263,0x3649,0x3649 },
      /* scale_idx = 31  (4897 src blocks) */ { 0xB55E,0xB55E,0xAE9C,0xAE9C,0xABF9,0xABF9,0xA2AC,0xA2AC,0x28AA,0x28AA,0x2CBE,0x2CBE,0x2F2C,0x2F2C,0x33CC,0x33CC },
};

// =============================================================================
// V-SIDE TABLES
// =============================================================================

__device__ static const uint16_t q0_v_scale_table_bits_v[32] = {
        0x0575,0x168E,0x17CE,0x1850,0x18A3,0x18EA,0x1929,0x1964,
        0x199B,0x19CF,0x1A02,0x1A34,0x1A65,0x1A95,0x1AC6,0x1AF7,
        0x1B29,0x1B5D,0x1B92,0x1BC8,0x1C01,0x1C1F,0x1C3F,0x1C62,
        0x1C88,0x1CB3,0x1CE4,0x1D1E,0x1D67,0x1DCA,0x1E69,0x2008
};

__device__ static const uint16_t q0_v_centroid_table_bits_v[32][16] = {
      /* scale_idx =  0  (7635 src blocks) */ { 0xA9F3,0xA9F3,0x9FE8,0x9FE8,0x9C08,0x9C08,0x9500,0x9500,0x1340,0x1340,0x1B10,0x1B10,0x1F40,0x1F40,0x2C53,0x2C53 },
      /* scale_idx =  1  (43800 src blocks) */ { 0xB09E,0xB09E,0xA5C2,0xA5C2,0xA19C,0xA19C,0x9A40,0x9A40,0x1BF0,0x1BF0,0x21F4,0x21F4,0x25E8,0x25E8,0x31D1,0x31D1 },
      /* scale_idx =  2  (37297 src blocks) */ { 0xB10F,0xB10F,0xA7B6,0xA7B6,0xA3AC,0xA3AC,0x9C88,0x9C88,0x1D50,0x1D50,0x2402,0x2402,0x27C4,0x27C4,0x3220,0x3220 },
      /* scale_idx =  3  (36644 src blocks) */ { 0xB41E,0xB41E,0xA833,0xA833,0xA454,0xA454,0x9D50,0x9D50,0x1D78,0x1D78,0x245E,0x245E,0x2844,0x2844,0x311D,0x311D },
      /* scale_idx =  4  (36307 src blocks) */ { 0xB26F,0xB26F,0xA88A,0xA88A,0xA4B2,0xA4B2,0x9E20,0x9E20,0x1DA0,0x1DA0,0x2498,0x2498,0x2871,0x2871,0x319C,0x319C },
      /* scale_idx =  5  (36217 src blocks) */ { 0xB33F,0xB33F,0xA8AD,0xA8AD,0xA4BA,0xA4BA,0x9DC8,0x9DC8,0x1E68,0x1E68,0x24F2,0x24F2,0x28B4,0x28B4,0x3214,0x3214 },
      /* scale_idx =  6  (36480 src blocks) */ { 0xB2D5,0xB2D5,0xA8F9,0xA8F9,0xA534,0xA534,0x9EA8,0x9EA8,0x1E40,0x1E40,0x2516,0x2516,0x28EE,0x28EE,0x32A6,0x32A6 },
      /* scale_idx =  7  (35974 src blocks) */ { 0xB39B,0xB39B,0xA924,0xA924,0xA55A,0xA55A,0x9EE0,0x9EE0,0x1E88,0x1E88,0x2546,0x2546,0x2934,0x2934,0x3329,0x3329 },
      /* scale_idx =  8  (36106 src blocks) */ { 0xB487,0xB487,0xA948,0xA948,0xA57A,0xA57A,0x9EE8,0x9EE8,0x1ED0,0x1ED0,0x2560,0x2560,0x2933,0x2933,0x3372,0x3372 },
      /* scale_idx =  9  (36212 src blocks) */ { 0xB3FB,0xB3FB,0xA974,0xA974,0xA5B2,0xA5B2,0x9F28,0x9F28,0x1EF0,0x1EF0,0x2592,0x2592,0x2967,0x2967,0x328C,0x328C },
      /* scale_idx = 10  (35935 src blocks) */ { 0xB460,0xB460,0xA9A7,0xA9A7,0xA5CA,0xA5CA,0x9EF8,0x9EF8,0x1F50,0x1F50,0x25D0,0x25D0,0x29A7,0x29A7,0x3350,0x3350 },
      /* scale_idx = 11  (36276 src blocks) */ { 0xB3CC,0xB3CC,0xA9B1,0xA9B1,0xA5EA,0xA5EA,0x9FA8,0x9FA8,0x1F20,0x1F20,0x25DE,0x25DE,0x29B9,0x29B9,0x3482,0x3482 },
      /* scale_idx = 12  (35923 src blocks) */ { 0xB473,0xB473,0xA9E1,0xA9E1,0xA624,0xA624,0x9FE0,0x9FE0,0x1F90,0x1F90,0x2604,0x2604,0x29CF,0x29CF,0x345B,0x345B },
      /* scale_idx = 13  (36137 src blocks) */ { 0xB359,0xB359,0xAA09,0xAA09,0xA626,0xA626,0x9FF0,0x9FF0,0x1FB8,0x1FB8,0x2628,0x2628,0x29FC,0x29FC,0x347A,0x347A },
      /* scale_idx = 14  (36011 src blocks) */ { 0xB4AA,0xB4AA,0xAA25,0xAA25,0xA666,0xA666,0xA034,0xA034,0x1FC8,0x1FC8,0x263A,0x263A,0x2A16,0x2A16,0x345F,0x345F },
      /* scale_idx = 15  (36307 src blocks) */ { 0xB4AB,0xB4AB,0xAA47,0xAA47,0xA6A0,0xA6A0,0xA054,0xA054,0x1F68,0x1F68,0x2640,0x2640,0x2A18,0x2A18,0x33E8,0x33E8 },
      /* scale_idx = 16  (35915 src blocks) */ { 0xB437,0xB437,0xAA53,0xAA53,0xA69E,0xA69E,0xA064,0xA064,0x2000,0x2000,0x2676,0x2676,0x2A38,0x2A38,0x34BF,0x34BF },
      /* scale_idx = 17  (36203 src blocks) */ { 0xB4DB,0xB4DB,0xAA76,0xAA76,0xA6C8,0xA6C8,0xA080,0xA080,0x1F80,0x1F80,0x268C,0x268C,0x2A7A,0x2A7A,0x34E7,0x34E7 },
      /* scale_idx = 18  (35969 src blocks) */ { 0xB478,0xB478,0xAAA2,0xAAA2,0xA6D2,0xA6D2,0xA054,0xA054,0x2038,0x2038,0x26D0,0x26D0,0x2A95,0x2A95,0x3564,0x3564 },
      /* scale_idx = 19  (36317 src blocks) */ { 0xB4E4,0xB4E4,0xAAC1,0xAAC1,0xA704,0xA704,0xA094,0xA094,0x201C,0x201C,0x26C0,0x26C0,0x2AA6,0x2AA6,0x34E3,0x34E3 },
      /* scale_idx = 20  (35929 src blocks) */ { 0xB4AA,0xB4AA,0xAAD5,0xAAD5,0xA700,0xA700,0xA074,0xA074,0x2034,0x2034,0x26EA,0x26EA,0x2AD4,0x2AD4,0x35EA,0x35EA },
      /* scale_idx = 21  (36227 src blocks) */ { 0xB4C6,0xB4C6,0xAAF8,0xAAF8,0xA760,0xA760,0xA0E0,0xA0E0,0x2044,0x2044,0x26E8,0x26E8,0x2ACB,0x2ACB,0x3516,0x3516 },
      /* scale_idx = 22  (36215 src blocks) */ { 0xB48E,0xB48E,0xAB21,0xAB21,0xA774,0xA774,0xA0D8,0xA0D8,0x2068,0x2068,0x2732,0x2732,0x2B00,0x2B00,0x34F1,0x34F1 },
      /* scale_idx = 23  (36199 src blocks) */ { 0xB51A,0xB51A,0xAB47,0xAB47,0xA786,0xA786,0xA0BC,0xA0BC,0x2084,0x2084,0x2754,0x2754,0x2B3F,0x2B3F,0x34A4,0x34A4 },
      /* scale_idx = 24  (36183 src blocks) */ { 0xB4D3,0xB4D3,0xAB64,0xAB64,0xA7A8,0xA7A8,0xA108,0xA108,0x2084,0x2084,0x275C,0x275C,0x2B4D,0x2B4D,0x3484,0x3484 },
      /* scale_idx = 25  (36006 src blocks) */ { 0xB4F6,0xB4F6,0xAB82,0xAB82,0xA7A2,0xA7A2,0xA0C4,0xA0C4,0x20D0,0x20D0,0x27BC,0x27BC,0x2B7A,0x2B7A,0x350C,0x350C },
      /* scale_idx = 26  (36250 src blocks) */ { 0xB4BC,0xB4BC,0xABB0,0xABB0,0xA808,0xA808,0xA138,0xA138,0x2094,0x2094,0x27BA,0x27BA,0x2B93,0x2B93,0x3555,0x3555 },
      /* scale_idx = 27  (36361 src blocks) */ { 0xB51A,0xB51A,0xAC10,0xAC10,0xA834,0xA834,0xA15C,0xA15C,0x20E8,0x20E8,0x2812,0x2812,0x2BEA,0x2BEA,0x35C0,0x35C0 },
      /* scale_idx = 28  (36642 src blocks) */ { 0xB560,0xB560,0xAC3C,0xAC3C,0xA851,0xA851,0xA1B4,0xA1B4,0x2120,0x2120,0x282C,0x282C,0x2C27,0x2C27,0x3548,0x3548 },
      /* scale_idx = 29  (37210 src blocks) */ { 0xB4D0,0xB4D0,0xAC78,0xAC78,0xA886,0xA886,0xA188,0xA188,0x21C8,0x21C8,0x2883,0x2883,0x2C6F,0x2C6F,0x34C8,0x34C8 },
      /* scale_idx = 30  (38828 src blocks) */ { 0xB4DC,0xB4DC,0xAD2F,0xAD2F,0xA921,0xA921,0xA27C,0xA27C,0x21D4,0x21D4,0x2902,0x2902,0x2D19,0x2D19,0x3436,0x3436 },
      /* scale_idx = 31  (12493 src blocks) */ { 0xB457,0xB457,0xABCD,0xABCD,0xA844,0xA844,0xA0F8,0xA0F8,0x21E0,0x21E0,0x285B,0x285B,0x2BD9,0x2BD9,0x3435,0x3435 },
};

// =============================================================================
// BASE CURVES — the four per side, each stored twice
// =============================================================================
// Each base is stored twice in a row, so the rotation's `mod 32` disappears:
// element e of phase p is byte e + 2p of the 64-byte row, and e + 2p ≤ 61. A
// side's whole curve codebook is 256 bytes — two cache lines — so a warp
// decoding 32 different curves touches at most two lines. 16-byte aligned so
// a row reads as aligned 32-bit words. Base b is curve slot 16·b of the
// reference's 128-curve table.
//
// Each base is spelled once, below, and three tables are built from that one
// spelling: the doubled i8 rows the decoder reads, the same rows as floats
// for the encoder's correlations (an integer literal converts to float
// exactly, so no per-element conversion is left for the search to do), and
// each base's energy Σ base², summed at compile time.
#define Q0_V_BASE_K0  127, 126, 122, 116, 108,  98,  86,  75,  63,  51,  40,  30,  21,  14,   8,   4, \
                        2,   1,   0,   0,   0,  -1,  -2,  -4,  -8, -13, -20, -29, -39, -50, -62, -74
#define Q0_V_BASE_K1 -127, -54, -57, -52, -53, -52, -50, -43,   8,   8,   9,  10,  11,  10,  10,   7, \
                       -3,  -4,  -3,  -5,  -6,  -7,  -7,   6,  49,  51,  52,  53,  53,  53,  55,  41
#define Q0_V_BASE_K2  127,  37,  33,  40,  45,  34,   9,  -2,  -4,  -6,  -8,  -8,  -6,  -9, -17, -22, \
                      -23, -22, -20, -18, -18, -19, -20, -22, -22, -23, -24, -25, -27, -19,  21,  43
#define Q0_V_BASE_K3 -127, -84, -82, -83,  32,  33,  32,  32,  31,  31,  31,  32,  27,  28,  28,  28, \
                       28,  28,  28,  27,  23,  23,  23,  24,  24,  25,  24,  25, -79, -81, -83, -83
#define Q0_V_BASE_V0  127,  28,  15,   5,   1,  -2,  -5,  -7,  -9,  -9,  -8,  -9,  -9,  -9,  -9,  -9, \
                       -9,  -9,  -9,  -9,  -9,  -9,  -9, -10,  -9,  -7,  -6,  -3,   0,   4,   9,  15
#define Q0_V_BASE_V1 -127, -13,  -6,   6,   8,   8,   8,   6,   5,   3,   3,   2,   2,   1,   1,   0, \
                        0,   2,   1,   1,   2,   3,   4,   6,   8,   7,   9,  10,  11,  11,   9,   4
#define Q0_V_BASE_V2  127, -42,  -3, -10, -10,  -5,  -3,  -2,   1,  -2,   0,   1,  -1,  -1,  -1,  -2, \
                       -1,  -2,  -1,   0,   1,   1,   4,   4,   4,   1,  -1,  -3,  -8,  -9, -12,  -9
#define Q0_V_BASE_V3 -127, -36, -13,  -2,   7,  11,  16,  19,  23,  22,  21,  22,  20,  20,  18,  18, \
                       19,  16,  14,  11,  10,   6,   4,   0,  -4,  -7, -11, -15, -19, -24, -28, -31

#define Q0_V_TWICE(...) { __VA_ARGS__, __VA_ARGS__ }
__device__ __align__(16) static const int8_t q0_v_curve_base2_k[4][64] = {
    Q0_V_TWICE(Q0_V_BASE_K0), Q0_V_TWICE(Q0_V_BASE_K1), Q0_V_TWICE(Q0_V_BASE_K2), Q0_V_TWICE(Q0_V_BASE_K3),
};
__device__ __align__(16) static const int8_t q0_v_curve_base2_v[4][64] = {
    Q0_V_TWICE(Q0_V_BASE_V0), Q0_V_TWICE(Q0_V_BASE_V1), Q0_V_TWICE(Q0_V_BASE_V2), Q0_V_TWICE(Q0_V_BASE_V3),
};
__device__ __align__(16) static const float q0_v_curve_base2f_k[4][64] = {
    Q0_V_TWICE(Q0_V_BASE_K0), Q0_V_TWICE(Q0_V_BASE_K1), Q0_V_TWICE(Q0_V_BASE_K2), Q0_V_TWICE(Q0_V_BASE_K3),
};
__device__ __align__(16) static const float q0_v_curve_base2f_v[4][64] = {
    Q0_V_TWICE(Q0_V_BASE_V0), Q0_V_TWICE(Q0_V_BASE_V1), Q0_V_TWICE(Q0_V_BASE_V2), Q0_V_TWICE(Q0_V_BASE_V3),
};
#undef Q0_V_TWICE

template <int... V> struct Q0VSumSq { static constexpr int value = ((V * V) + ...); };
__device__ static const int q0_v_base_energy_k[4] = {
    Q0VSumSq<Q0_V_BASE_K0>::value, Q0VSumSq<Q0_V_BASE_K1>::value,
    Q0VSumSq<Q0_V_BASE_K2>::value, Q0VSumSq<Q0_V_BASE_K3>::value,
};
__device__ static const int q0_v_base_energy_v[4] = {
    Q0VSumSq<Q0_V_BASE_V0>::value, Q0VSumSq<Q0_V_BASE_V1>::value,
    Q0VSumSq<Q0_V_BASE_V2>::value, Q0VSumSq<Q0_V_BASE_V3>::value,
};
#undef Q0_V_BASE_K0
#undef Q0_V_BASE_K1
#undef Q0_V_BASE_K2
#undef Q0_V_BASE_K3
#undef Q0_V_BASE_V0
#undef Q0_V_BASE_V1
#undef Q0_V_BASE_V2
#undef Q0_V_BASE_V3

// =============================================================================
// TABLE ACCESSORS  (shared by encoder and decoder)
// =============================================================================
namespace q0_v_detail {

// The scale / centroid view the shared encoder prologue reads
// (`compute_target_and_indices`). `Q0VTablesStatic<IS_K>` is empty and folds
// to direct global loads; `Q0VTablesRuntime` below carries device pointers.
template <bool IS_K> struct Q0VTablesStatic;
template <> struct Q0VTablesStatic<true> {
    __device__ __forceinline__ uint16_t scale_bits(int i)    const { return __ldg(&q0_v_scale_table_bits_k[i]); }
    __device__ __forceinline__ uint16_t centroid_bits(int s, int c) const { return __ldg(&q0_v_centroid_table_bits_k[s][c]); }
};
template <> struct Q0VTablesStatic<false> {
    __device__ __forceinline__ uint16_t scale_bits(int i)    const { return __ldg(&q0_v_scale_table_bits_v[i]); }
    __device__ __forceinline__ uint16_t centroid_bits(int s, int c) const { return __ldg(&q0_v_centroid_table_bits_v[s][c]); }
};

// A codebook supplied at launch, for the curve-selection diagnostic that
// swaps the codebook without recompiling. It carries the full 128-curve
// table and the peak-bin permutation its hierarchical search reads:
//   const int8_t* curve(int slot)
//   uint16_t      scale_bits(int i)
//   uint16_t      centroid_bits(int scale_idx, int cent_idx)
//   uint8_t       peak_idx(int i)
//   uint16_t      peak_off(int bin)
struct Q0VTablesRuntime {
    const int8_t*   curve_table_flat;          // [128 * 32] i8
    const uint16_t* scale_table_bits;           // [32] f16 bits
    const uint16_t* centroid_table_bits_flat;   // [32 * 16] f16 bits, scale-major
    const uint8_t*  peak_curve_indices;         // [128]
    const uint16_t* peak_bin_offsets;           // [33]

    __device__ __forceinline__ const int8_t* curve(int slot) const {
        return curve_table_flat + (size_t)slot * 32;
    }
    __device__ __forceinline__ uint16_t scale_bits(int i) const { return scale_table_bits[i]; }
    __device__ __forceinline__ uint16_t centroid_bits(int s, int c) const {
        return centroid_table_bits_flat[(size_t)s * 16 + c];
    }
    __device__ __forceinline__ uint8_t  peak_idx(int i)   const { return peak_curve_indices[i]; }
    __device__ __forceinline__ uint16_t peak_off(int bin) const { return peak_bin_offsets[bin]; }
};

// -----------------------------------------------------------------------------
// The decoder's view of a side's codebook: the doubled base curves and the
// scale / centroid tables, every read through the read-only cache (`__ldg`).
// -----------------------------------------------------------------------------
template <bool IS_K> struct Q0VDecodeTables;
template <> struct Q0VDecodeTables<true> {
    static __device__ __forceinline__ const int8_t* base2(int b) { return q0_v_curve_base2_k[b]; }
    static __device__ __forceinline__ const float* base2f(int b) { return q0_v_curve_base2f_k[b]; }
    static __device__ __forceinline__ int energy(int b) { return __ldg(&q0_v_base_energy_k[b]); }
    static __device__ __forceinline__ uint16_t scale_bits(int s) { return __ldg(&q0_v_scale_table_bits_k[s]); }
    static __device__ __forceinline__ uint16_t centroid_bits(int s, int c) { return __ldg(&q0_v_centroid_table_bits_k[s][c]); }
};
template <> struct Q0VDecodeTables<false> {
    static __device__ __forceinline__ const int8_t* base2(int b) { return q0_v_curve_base2_v[b]; }
    static __device__ __forceinline__ const float* base2f(int b) { return q0_v_curve_base2f_v[b]; }
    static __device__ __forceinline__ int energy(int b) { return __ldg(&q0_v_base_energy_v[b]); }
    static __device__ __forceinline__ uint16_t scale_bits(int s) { return __ldg(&q0_v_scale_table_bits_v[s]); }
    static __device__ __forceinline__ uint16_t centroid_bits(int s, int c) { return __ldg(&q0_v_centroid_table_bits_v[s][c]); }
};

/// The block's 16-bit code: curve_idx = bits[0..6], scale_idx = bits[7..11],
/// centroid_idx = bits[12..15].
__device__ __forceinline__ int q0_v_curve_idx(uint32_t bits)    { return (int)(bits & 0x7Fu); }
__device__ __forceinline__ int q0_v_scale_idx(uint32_t bits)    { return (int)((bits >> 7) & 0x1Fu); }
__device__ __forceinline__ int q0_v_centroid_idx(uint32_t bits) { return (int)((bits >> 12) & 0x0Fu); }

/// Byte `e` of the returned row is curve element e of `curve_idx`, BEFORE the
/// bucket's sign: base b & 3 of the doubled table, offset by the phase's 2p.
template <bool IS_K>
__device__ __forceinline__ const int8_t* q0_v_curve_row(int curve_idx) {
    return Q0VDecodeTables<IS_K>::base2((curve_idx >> 4) & 3) + 2 * (curve_idx & 15);
}

/// Buckets 4–7 (curve_idx bit 6) are the negations of buckets 0–3.
__device__ __forceinline__ bool q0_v_curve_negated(int curve_idx) { return (curve_idx & 0x40) != 0; }

template <bool IS_K>
__device__ __forceinline__ float q0_v_scale(uint32_t bits) {
    return __half2float(__ushort_as_half(Q0VDecodeTables<IS_K>::scale_bits(q0_v_scale_idx(bits))));
}

template <bool IS_K>
__device__ __forceinline__ float q0_v_centroid(uint32_t bits) {
    return __half2float(__ushort_as_half(
        Q0VDecodeTables<IS_K>::centroid_bits(q0_v_scale_idx(bits), q0_v_centroid_idx(bits))));
}

}  // namespace q0_v_detail
