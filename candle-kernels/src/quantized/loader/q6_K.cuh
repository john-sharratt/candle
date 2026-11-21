#pragma once

// =============================================================================
// Q6_K LOADER - K/128 FORMAT WITH EMBEDDED SCALES (COMPACT 112-BYTE LAYOUT)
// =============================================================================
//
// This loader processes Q6_K weights in K/128 format [K/128, N].
// Each block contains 128 × 6-bit weights plus embedded half scales.
//
// COMPACT 112-BYTE LAYOUT (12.5% smaller than 128-byte)
// -----------------------------------------------------
// Bytes 0-63:   ql[16]    - 16 ints, one per thread (PERFECT coalescing!)
// Bytes 64-95:  qh[8]     - 8 ints, packed as (qh_lo | qh_hi << 16)
// Bytes 96-111: scales[8] - 8 halfs, one per thread-pair
//
// DEQUANTIZATION
// --------------
// 6-bit value: q6 = (ql_nibble & 0xF) | ((qh_crumb & 0x3) << 4)
// Symmetric quantization: value = scale * (q6 - 32)
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
// - K-tile permutation: {0,4,1,5,2,6,3,7,8,12,9,13,10,14,11,15} for coalescing
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

#include "q6_K/q6k_ktile_loader.cuh"

#include "gemx_dequant.cuh"

#include "q6_K/gemx_dequant_traits_q6.cuh"

