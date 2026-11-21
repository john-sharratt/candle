#pragma once

// NOTE: Do NOT include impl/common.cuh here - it creates a circular dependency
// because impl/common.cuh -> kernel.cuh -> loaders.cuh -> loader/*.cuh need block_c_* types.
// The block_compact type trait specializations are defined in loaders.cuh instead.

// =============================================================================
// COMPACTED BLOCK STRUCTURES (block_c_*) - SCALES EMBEDDED
// =============================================================================
// These structures represent quantized blocks with scales EMBEDDED at the end
// of each K/128 block for use with batched matmul and GEMX kernels.
// This enables:
//   1. Better memory coalescing (scales accessed in separate pattern)
//   2. Potential tensor core usage with repacked weights
//   3. Reduced memory traffic when scales are cached/reused
//
// LAYOUT MODE:
// - K/64 (suffix _k64): 64 elements per struct, scales embedded at end
//
// Naming: 
//   block_c_<format>_k64 = K/64 layout, scales embedded at end of the block
//
// K/64 LAYOUT (scales embedded at end of each block):
// ┌───────────────────┬─────────────────┬──────────────────────────────┐
// │ Format            │ Weights         │ Embedded Scales (per 64)      │
// ├───────────────────┼─────────────────┼──────────────────────────────┤
// │ Q4_0              │ 64 × 4-bit      │ 1 × half2 (d0, d1)           │
// │ Q4_1              │ 64 × 4-bit      │ 2 × half2 (d0,m0)(d1,m1)     │
// │ Q5_0              │ 64 × 5-bit      │ 1 × half2 (d0, d1)           │
// │ Q5_1              │ 64 × 5-bit      │ 2 × half2 (d0,m0)(d1,m1)     │
// │ Q8_0              │ 64 × 8-bit      │ 1 × half2 (d0, d1)           │
// │ Q2_K              │ 64 × 2-bit      │ 4 × half2 (per 16 elements)  │
// │ Q3_K              │ 64 × 3-bit      │ 4 × half2 (per 16 elements)  │
// │ Q4_K              │ 64 × 4-bit      │ 2 × half2 (per 32 elements)  │
// │ Q5_K              │ 64 × 5-bit      │ 2 × half2 (per 32 elements)  │
// │ Q6_K              │ 64 × 6-bit      │ 4 × half (per 16 elements)   │
// └───────────────────┴─────────────────┴──────────────────────────────┘
// =============================================================================

// =============================================================================
// GGML QUANTIZATION BLOCK MEMORY LAYOUTS - AUTHORITATIVE REFERENCE
// =============================================================================
// 
// ⚠️  CRITICAL: These layouts are NOT always intuitive! Each format has unique
//     memory organization inherited from GGML. Always verify against GGML's
//     dequantize_block_* functions in ggml-quants.c before writing extractors.
//
// -----------------------------------------------------------------------------
// Q4_0: 4-bit symmetric quantization, 32 elements per block
// -----------------------------------------------------------------------------
// Struct: { half d; uint8_t qs[16]; }  // 18 bytes
// 
// qs layout: SEQUENTIAL nibbles
//   qs[i] low nibble  = element 2*i
//   qs[i] high nibble = element 2*i + 1
//
// Dequant: value = d * (nibble - 8)
//
// -----------------------------------------------------------------------------
// Q4_1: 4-bit asymmetric quantization, 32 elements per block  
// -----------------------------------------------------------------------------
// Struct: { half2 dm; uint8_t qs[16]; }  // 20 bytes (dm.x=d, dm.y=m)
//
// qs layout: SEQUENTIAL nibbles (same as Q4_0)
//   qs[i] low nibble  = element 2*i
//   qs[i] high nibble = element 2*i + 1
//
// Dequant: value = d * nibble + m
//
// -----------------------------------------------------------------------------
// Q5_0: 5-bit symmetric quantization, 32 elements per block
// -----------------------------------------------------------------------------
// Struct: { half d; uint8_t qh[4]; uint8_t qs[16]; }  // 22 bytes
//
// qs layout: SEQUENTIAL nibbles (same as Q4_0)
// qh layout: SEQUENTIAL bits - bit i of qh[i/8] = high bit of element i
//   qh[i/8] bit (i%8) = 5th bit for element i
//
// Dequant: value = d * ((nibble | (highbit << 4)) - 16)
//
// -----------------------------------------------------------------------------
// Q5_1: 5-bit asymmetric quantization, 32 elements per block
// -----------------------------------------------------------------------------
// Struct: { half2 dm; uint8_t qh[4]; uint8_t qs[16]; }  // 24 bytes
//
// qs layout: SEQUENTIAL nibbles (same as Q4_0)
// qh layout: SEQUENTIAL bits (same as Q5_0)
//
// Dequant: value = d * (nibble | (highbit << 4)) + m
//
// -----------------------------------------------------------------------------
// Q8_0: 8-bit symmetric quantization, 32 elements per block
// -----------------------------------------------------------------------------
// Struct: { half d; int8_t qs[32]; }  // 34 bytes
//
// qs layout: SEQUENTIAL - qs[i] = element i directly
//
// Dequant: value = d * qs[i]
//
// =============================================================================
// K-QUANTS (256 elements per super-block) - MORE COMPLEX LAYOUTS!
// =============================================================================
//
// -----------------------------------------------------------------------------
// Q2_K: 2-bit quantization, 256 elements per super-block
// -----------------------------------------------------------------------------
// Struct: { uint8_t scales[16]; uint8_t qs[64]; half2 dm; }  // 84 bytes
//
// qs layout: SEQUENTIAL 2-bit values
//   qs[i] contains 4 elements: bits [1:0], [3:2], [5:4], [7:6]
//   Element 4*i+j = (qs[i] >> (2*j)) & 3
//
// scales: 16 × 4-bit scale/min pairs for 16 sub-blocks of 16 elements each
//
// -----------------------------------------------------------------------------
// Q3_K: 3-bit quantization, 256 elements per super-block
// -----------------------------------------------------------------------------
// Struct: { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; half d; }  // 110 bytes
//
// qs layout: SEQUENTIAL 2-bit low parts
//   Low 2 bits: (qs[i] >> (2*j)) & 3 for element 4*i+j
//
// hmask layout: HIGH BIT INTERLEAVED
//   hmask[i] bit j = high bit for element i + 32*j
//   (NOT sequential like Q5_0!)
//
// scales: 12 bytes encoding 16 × 6-bit scales (complex packing)
//
// -----------------------------------------------------------------------------
// Q4_K: 4-bit quantization, 256 elements per super-block
// -----------------------------------------------------------------------------
// Struct: { half2 dm; uint8_t scales[12]; uint8_t qs[128]; }  // 144 bytes
//
// qs layout: BLOCK INTERLEAVED (32 low, 32 high pattern)
//   Elements 0-31:    qs[0-31] LOW nibbles
//   Elements 32-63:   qs[0-31] HIGH nibbles  
//   Elements 64-95:   qs[32-63] LOW nibbles
//   Elements 96-127:  qs[32-63] HIGH nibbles
//   Elements 128-159: qs[64-95] LOW nibbles
//   Elements 160-191: qs[64-95] HIGH nibbles
//   Elements 192-223: qs[96-127] LOW nibbles
//   Elements 224-255: qs[96-127] HIGH nibbles
//
// Formula: element e → qs[block_64*32 + (e&31)], nibble = (e&32) ? high : low
//   where block_64 = e / 64
//
// scales: 12 bytes encoding 8 × (6-bit scale, 6-bit min) pairs
//
// -----------------------------------------------------------------------------
// Q5_K: 5-bit quantization, 256 elements per super-block
// -----------------------------------------------------------------------------
// Struct: { half2 dm; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; }  // 176 bytes
//
// qs layout: BLOCK INTERLEAVED (same as Q4_K!)
//   Elements 0-31:    qs[0-31] LOW nibbles
//   Elements 32-63:   qs[0-31] HIGH nibbles
//   Elements 64-95:   qs[32-63] LOW nibbles
//   Elements 96-127:  qs[32-63] HIGH nibbles
//   Elements 128-159: qs[64-95] LOW nibbles
//   Elements 160-191: qs[64-95] HIGH nibbles
//   Elements 192-223: qs[96-127] LOW nibbles
//   Elements 224-255: qs[96-127] HIGH nibbles
//
// qh layout: TRANSPOSED BIT MATRIX (NOT sequential!)
//   qh[i] bit j = high bit for element (32*j + i)
//   Formula: element e → qh[e % 32] bit (e / 32)
//
// ⚠️  Q5_K qh is DIFFERENT from Q5_0 qh! Q5_0 is sequential, Q5_K is transposed.
//
// scales: 12 bytes encoding 8 × (6-bit scale, 6-bit min) pairs (same as Q4_K)
//
// -----------------------------------------------------------------------------
// Q6_K: 6-bit symmetric quantization, 256 elements per super-block
// -----------------------------------------------------------------------------
// Struct: { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; }  // 210 bytes
//
// ql layout: HALF INTERLEAVED
//   Elements 0-31:   ql[0-31] LOW nibbles
//   Elements 32-63:  ql[32-63] LOW nibbles
//   Elements 64-95:  ql[0-31] HIGH nibbles
//   Elements 96-127: ql[32-63] HIGH nibbles
//   Elements 128-159: ql[64-95] LOW nibbles
//   Elements 160-191: ql[96-127] LOW nibbles
//   Elements 192-223: ql[64-95] HIGH nibbles
//   Elements 224-255: ql[96-127] HIGH nibbles
//
// qh layout: CRUMB INTERLEAVED (2 bits per element)
//   qh[i] contains crumbs for elements i, 32+i, 64+i, 96+i at bits [1:0], [3:2], [5:4], [7:6]
//   Formula: element e → qh[e % 32] bits ((e/32)*2 +: 2)
//
// scales: 16 × int8_t scales for 16 sub-blocks of 16 elements each
//
// =============================================================================
// EXTRACTION HELPER FORMULAS (copy-paste ready)
// =============================================================================
//
// Q4_K/Q5_K qs (block interleaved):
//   block_64 = elem / 64
//   in_block = elem % 64
//   is_high = (in_block >= 32)
//   byte_idx = block_64 * 32 + (in_block & 31)
//   nibble = is_high ? (qs[byte_idx] >> 4) : (qs[byte_idx] & 0xF)
//
// Q5_K qh (transposed bits):
//   byte_idx = elem % 32
//   bit_idx = elem / 32
//   highbit = (qh[byte_idx] >> bit_idx) & 1
//
// Q6_K ql (half interleaved):
//   half_128 = elem / 128
//   in_half = elem % 128
//   is_high = (in_half >= 64)
//   byte_idx = half_128 * 64 + (in_half & 63) - (is_high ? 32 : 0)
//   nibble = is_high ? (ql[byte_idx] >> 4) : (ql[byte_idx] & 0xF)
//
// Q6_K qh (crumb interleaved):
//   byte_idx = elem % 32
//   crumb_idx = elem / 32
//   crumb = (qh[byte_idx] >> (crumb_idx * 2)) & 3
//
// =============================================================================

#include <cuda_fp16.h>
#include <stdint.h>

#include "block_compact/k128_block_structs.cuh"

#include "block_compact/ko_chunks_awq_and_tile_traits.cuh"

#include "block_compact/qtype_enum_and_helpers.cuh"
