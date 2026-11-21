#pragma once

// =============================================================================
// Q4_K LOADER - K/128 FORMAT WITH EMBEDDED SCALES
// =============================================================================
//
// This loader processes Q4_K weights in K/128 format [K/128, N].
// Each block contains 128 × 4-bit weights plus embedded half2 scales.
//
// LAYOUT
// ------
// Weights: [K/128, N] block_c_q4_K_k128
// Scales:  embedded half2[4] per 32 elements (4 groups of 32)
//
// THREAD MAPPING & WARP UTILIZATION
// ---------------------------------
// Each K/128 block requires 16 threads (each thread handles 8 elements).
// The kernel runs with 128 threads (4 warps), processing 8 K-blocks in parallel:
//
//   Threads 0-15   → K-block 0, lane = threadIdx.x & 15 → 0-15
//   Threads 16-31  → K-block 1, lane = threadIdx.x & 15 → 0-15
//   Threads 32-47  → K-block 2, lane = threadIdx.x & 15 → 0-15
//   ...            → ...       ...
//   Threads 112-127→ K-block 7, lane = threadIdx.x & 15 → 0-15
//
// IMPORTANT: get_lane() = threadIdx.x & 15 does NOT mean 50% warp utilization!
// All 128 threads are active - the masking is just for indexing within each
// 16-thread group that handles one K/128 block. This achieves 100% utilization.
//
// OPTIMIZATIONS
// -------------
// - Scales stored as half2: no conversion for half/bf16 paths
// - Batch nibble extraction: separate lo/hi nibbles, then byte extraction
// - float4 vector loads/stores: 4× fewer memory transactions
// - half2/bf162 arithmetic: 2× throughput for FP16/BF16 Y inputs
// - Register-pressure optimized: process in groups, immediate consumption
//
// =============================================================================

#include "../impl/common.cuh"
#include "../block_compact.cuh"
#include "../math.cuh"
#include "scale_types.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "q4_K/k_tile_loader.cuh"

// =============================================================================
// GEMX DEQUANT TRAITS - Q4_K (4-bit K-quant with scale and min)
// =============================================================================
// Include the base gemx_dequant infrastructure
#include "gemx_dequant.cuh"

#include "q4_K/mma_dequant.cuh"



