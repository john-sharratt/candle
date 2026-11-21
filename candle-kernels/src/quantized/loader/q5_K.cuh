#pragma once

// =============================================================================
// Q5_K LOADER - K/128 FORMAT WITH EMBEDDED SCALES
// =============================================================================
//
// This loader processes Q5_K weights in K/128 format [K/128, N].
// Each block contains 128 × 5-bit weights plus embedded half2 scales.
//
// STORAGE FORMAT
// --------------
// Q5_K uses the same 12-byte K-tile structure as Q6_K for alignment:
//   - ql_x: 4B (nibbles 0-7, 4 bits each)
//   - ql_y: 4B (nibbles 8-15, 4 bits each)
//   - qh:   4B (16 × 1-bit high bits, PADDED from 2B to 4B with zeros)
//
// The original Q5_K K-tile is 10 bytes (8B qs + 2B qh), but 10-byte stride
// causes severe memory coalescing issues. By padding qh to 4B (with upper
// 16 bits = 0), we get 12-byte aligned tiles and can reuse Q6_K's efficient
// extraction code with the same 0x30303030 masks (upper crumb bits are 0).
//
// LAYOUT
// ------
// Weights: [K/128, N] block_c_q5_K_k128
// Scales:  embedded half2[2] per 32 elements
//
// DEQUANTIZATION
// --------------
// 5-bit value: q5 = (ql_nibble & 0xF) | ((qh_bit & 0x1) << 4)
// Since qh is padded with zeros, the Q6_K crumb extraction (qh & 0x3) << 4
// produces the same result as (qh & 0x1) << 4 for Q5_K data.
// Symmetric quantization: value = scale * (q5 - 16)
//
// OPTIMIZATIONS
// -------------
// - 12-byte aligned K-tiles: proper memory coalescing (vs 10B unaligned)
// - Reuses Q6_K extraction logic: same masks work due to zero padding
// - float4 vector loads/stores: 4× fewer memory transactions
// - half2/bf162 arithmetic: 2× throughput for FP16/BF16 Y inputs
// - Extract-and-consume: process 4 elements at a time to reduce register pressure
//
// =============================================================================

#include "../impl/common.cuh"
#include "../block_compact.cuh"
#include "../math.cuh"
#include "scale_types.cuh"
#include <cuda_fp8.h>

#include "q5_K/ktile_loader.cuh"

// =============================================================================
// GEMX DEQUANT TRAITS - Q5_K
// =============================================================================
//
// K-tile: 12B = {ql_x, ql_y, qh} containing 16 × 5-bit elements.
// ql_x/ql_y contain 8 nibbles each in LOP3-ready layout.
// qh contains 1-bit high bits (padded to crumb format).
// Dequant: q5 = (ql_nibble | (qh_bit << 4)), then apply affine scale.
//
// LOP3-READY: ql_x/ql_y nibbles are in LOP3-ready layout:
//   ql & 0x000f000f          → (n0, n1)
//   (ql >> 4) & 0x000f000f   → (n4, n5)  
//   (ql >> 8) & 0x000f000f   → (n2, n3)
//   (ql >> 12) & 0x000f000f  → (n6, n7)
//
// =============================================================================

#include "gemx_dequant.cuh"

#include "q5_K/gemx_dequant.cuh"

