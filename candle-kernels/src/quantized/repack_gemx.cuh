#pragma once

// =============================================================================
// GEMX WEIGHT REPACKING KERNELS (K/128)
// =============================================================================
// Reorders quantized weights to GEMX format for tensor core kernels.
// K/128 format: 16 threads × 8 elements per thread = 128 elements per block.
//
// Key transformations:
// 1. Extract scales from GGML superblocks and embed into K/128 output
// 2. Permute quants to thread-major layout (each thread owns contiguous elements)
// 3. Pack bits efficiently for warp-level tensor core consumption
//
// For each Q type (all output K/128 blocks):
// - Q4_0: 4×32-elem GGML blocks → 1×128-elem K/128 block with 4 scales
// - Q4_1: 4×32-elem GGML blocks → 1×128-elem K/128 block with 4 dm pairs
// - Q5_0: 4×32-elem GGML blocks → 1×128-elem K/128 block with 4 scales + high bits
// - Q5_1: 4×32-elem GGML blocks → 1×128-elem K/128 block with 4 dm pairs + high bits
// - Q8_0: 4×32-elem GGML blocks → 1×128-elem K/128 block with 4 scales
// - Q2_K: 1×256-elem / 2 = 128-elem slice → K/128 block with 8 dm pairs
// - Q3_K: 1×256-elem / 2 = 128-elem slice → K/128 block with 8 dm pairs
// - Q4_K: 1×256-elem / 2 = 128-elem slice → K/128 block with 4 dm pairs
// - Q5_K: 1×256-elem / 2 = 128-elem slice → K/128 block with 4 dm pairs + high bits
// - Q6_K: 1×256-elem / 2 = 128-elem slice → K/128 block with 8 scales
//
// ⚠️  MEMORY LAYOUT REFERENCE: See block_compact.cuh for K/128 struct definitions.
// =============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "block_compact.cuh"

#include "repack_gemx/gemx_constants_and_permutations.cuh"

#include "repack_gemx/simple_format_repacks.cuh"

#include "repack_gemx/kquant_awq_repacks.cuh"
