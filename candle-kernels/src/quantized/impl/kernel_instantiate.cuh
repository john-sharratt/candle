#pragma once

// =============================================================================
// KERNEL INSTANTIATION MACROS
// =============================================================================
// Generates extern "C" kernel entry points for a specific quantization format
// and act_t combination. Creates batch-specialized variants:
//
// GEMV kernels (batch 1-15):
//   _s1 - _s8:      BATCH_TILE=1-8, single to octet batch
//   _s8_xf:         BATCH_TILE=8, batch-fast grid layout
//
// Batch >= 16: handled by GEMX tensor core kernels (separate compilation)
//
// Greedy decomposition:
//   batch = (n_s8 × 8) + (n_s7 × 7) + ... + (n_s1 × 1)
//
// Usage:
//   #include "kernel_instantiate.cuh"
//   #include "common.cuh"
//   #include "../loader/{format}.cuh"
//   INSTANTIATE_KERNELS(name, qk, qi, block_type, vdr, act_t, dst_t)
// =============================================================================

// =============================================================================
// REGISTER-ONLY KERNELS (memory-bound, batch 1-8)
// No TC variants - at small batch, memory bandwidth is the bottleneck,
// tensor cores provide no benefit over CUDA cores.
// =============================================================================

// Single-batch specialist (BATCH_TILE=1)
// Decode fast path: no batch loops, minimal registers, max occupancy.
// Note: Scales are embedded in K/128 blocks - no external scales parameter.
#define INSTANTIATE_KERNEL_S1(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_VSMALL name##_s1( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int batch_size) { \
    quantized_matmul<qk, qi, block_type, vdr, act_t, dst_t, 1>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size); \
}

// Pair-batch specialist (BATCH_TILE=2)
// Fixed 2-batch with full unroll. Weights loaded once, reused 2×.
#define INSTANTIATE_KERNEL_S2(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_VSMALL name##_s2( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int batch_size) { \
    quantized_matmul<qk, qi, block_type, vdr, act_t, dst_t, 2>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size); \
}

// Triple-batch specialist (BATCH_TILE=3)
// Fixed 3-batch with full unroll. Weights loaded once, reused 3×.
#define INSTANTIATE_KERNEL_S3(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_VSMALL name##_s3( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int batch_size) { \
    quantized_matmul<qk, qi, block_type, vdr, act_t, dst_t, 3>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size); \
}

// Quad-batch specialist (BATCH_TILE=4)
// Fixed 4-batch with full unroll. Weights loaded once, reused 4×.
#define INSTANTIATE_KERNEL_S4(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_SMALL name##_s4( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int batch_size) { \
    quantized_matmul<qk, qi, block_type, vdr, act_t, dst_t, 4>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size); \
}

// Penta-batch specialist (BATCH_TILE=5)
// Fixed 5-batch with full unroll. Weights loaded once, reused 5×.
#define INSTANTIATE_KERNEL_S5(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_SMALL name##_s5( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int batch_size) { \
    quantized_matmul<qk, qi, block_type, vdr, act_t, dst_t, 5>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size); \
}

// Hexa-batch specialist (BATCH_TILE=6)
// Fixed 6-batch with full unroll. Weights loaded once, reused 6×.
#define INSTANTIATE_KERNEL_S6(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_SMALL name##_s6( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int batch_size) { \
    quantized_matmul<qk, qi, block_type, vdr, act_t, dst_t, 6>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size); \
}

// Septa-batch specialist (BATCH_TILE=7)
// Fixed 7-batch with full unroll. Weights loaded once, reused 7×.
#define INSTANTIATE_KERNEL_S7(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_SMALL name##_s7( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int batch_size) { \
    quantized_matmul<qk, qi, block_type, vdr, act_t, dst_t, 7>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size); \
}

// Octet-batch specialist (BATCH_TILE=8)
// Fixed 8-batch with full unroll. Weights loaded once, reused 8×.
#define INSTANTIATE_KERNEL_S8(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_SMALL name##_s8( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int batch_size) { \
    quantized_matmul<qk, qi, block_type, vdr, act_t, dst_t, 8, GRID_LAYOUT_ROW_FAST>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size); \
}

// 32-batch specialist (BATCH_TILE=32)
// Hybrid path: smem weights + batch groups of 8. 4× weight reuse per smem load.
// Better amortization of smem load cost than _s16 (2×).
#define INSTANTIATE_KERNEL_S32(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_VEC name##_s32( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int batch_size) { \
    quantized_matmul<qk, qi, block_type, vdr, act_t, dst_t, 32>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size); \
}

// 48-batch specialist (BATCH_TILE=48)
// Hybrid path: smem weights + batch groups of 8. 6× weight reuse per smem load.
#define INSTANTIATE_KERNEL_S48(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_VEC name##_s48( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int batch_size) { \
    quantized_matmul<qk, qi, block_type, vdr, act_t, dst_t, 48>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size); \
}

// =============================================================================
// SHARED MEMORY KERNELS (transitional regime, batch 33-255)
// =============================================================================

// Medium bulk specialist (BATCH_TILE=64) - CUDA core path
// Hybrid path: smem weights + batch groups of 8. 8× weight reuse per smem load.
#define INSTANTIATE_KERNEL_S64(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_VEC name##_s64( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int batch_size) { \
    quantized_matmul<qk, qi, block_type, vdr, act_t, dst_t, 64>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size); \
}

// Large bulk specialist (BATCH_TILE=128) - CUDA core path
// Smem weight tiles, single buffer, requires ≥48KB smem.
// Available on: Turing GTX, Pascal with 48KB.
#define INSTANTIATE_KERNEL_S128(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_VEC name##_s128( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int batch_size) { \
    quantized_matmul<qk, qi, block_type, vdr, act_t, dst_t, 128>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size); \
}

// =============================================================================
// L2 CHUNKED KERNELS - REMOVED (TODO: re-implement with GEMX pipeline)
// =============================================================================

// =============================================================================
// MAIN MACRO: Instantiate all kernels for a QTYPE/YTYPE pair
// =============================================================================
// Generates kernels including batch-fast (_xf) variants for L2 optimization:
//   Register-only: _s1-_s7, _s8, _s8_xf
//   Iterator kernels: _s1_iter, _s2_iter, _s3_iter, _s4_iter (internal batch loop)
//   Batch >= 16: handled by GEMX tensor core kernels (separate compilation)
//
// _xf variants use grid(batch_tiles, row_blocks) for better X (weights) L2 caching
// when batch_tiles > 1. Dispatcher chooses based on batch size and model size.
//
// _iter variants use internal batch iteration for L2 weight reuse across batches.
// Single kernel launch processes all batches, weights stay in L2 cache.

// =============================================================================
// ITERATOR KERNELS (L2 weight reuse across batch iterations)
// =============================================================================
// s2_iter: BATCH_TILE=2 iteration kernels for large batches.
// - Uses ~60 registers (high occupancy ~100%)
// - Good latency hiding with many warps in flight
// - s2_iter2: 4 batches, s2_iter3: 6 batches, ..., s2_iter8: 16 batches
// Note: s2_iter1 removed (unused in dispatch tables)
//
// Dispatcher uses s2_iter8 in a loop for batch > 16.

#define INSTANTIATE_KERNEL_S2_ITER2(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_ITER name##_s2_iter2( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int total_batches) { \
    quantized_matmul_iter<qk, qi, block_type, vdr, act_t, dst_t, 2, 2>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, total_batches); \
}

#define INSTANTIATE_KERNEL_S2_ITER3(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_ITER name##_s2_iter3( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int total_batches) { \
    quantized_matmul_iter<qk, qi, block_type, vdr, act_t, dst_t, 2, 3>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, total_batches); \
}

#define INSTANTIATE_KERNEL_S2_ITER4(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_ITER name##_s2_iter4( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int total_batches) { \
    quantized_matmul_iter<qk, qi, block_type, vdr, act_t, dst_t, 2, 4>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, total_batches); \
}

#define INSTANTIATE_KERNEL_S2_ITER5(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_ITER name##_s2_iter5( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int total_batches) { \
    quantized_matmul_iter<qk, qi, block_type, vdr, act_t, dst_t, 2, 5>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, total_batches); \
}

#define INSTANTIATE_KERNEL_S2_ITER6(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_ITER name##_s2_iter6( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int total_batches) { \
    quantized_matmul_iter<qk, qi, block_type, vdr, act_t, dst_t, 2, 6>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, total_batches); \
}

#define INSTANTIATE_KERNEL_S2_ITER7(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_ITER name##_s2_iter7( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int total_batches) { \
    quantized_matmul_iter<qk, qi, block_type, vdr, act_t, dst_t, 2, 7>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, total_batches); \
}

#define INSTANTIATE_KERNEL_S2_ITER8(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_ITER name##_s2_iter8( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int total_batches) { \
    quantized_matmul_iter<qk, qi, block_type, vdr, act_t, dst_t, 2, 8>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, total_batches); \
}

// s3_iter: BATCH_TILE=3 iteration kernels for batches divisible by 3.
// - Handles odd batch counts that s2_iter can't cover cleanly (9, etc.)
// Note: Only s3_iter3 is used (for batch=9, 18, 25, 41, 57)
//       s3_iter1/2/4/5 removed (unused in dispatch tables)

#define INSTANTIATE_KERNEL_S3_ITER3(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_ITER name##_s3_iter3( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int total_batches) { \
    quantized_matmul_iter<qk, qi, block_type, vdr, act_t, dst_t, 3, 3>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, total_batches); \
}

// =============================================================================
// UNIFIED TC KERNELS - Compile-time dispatch for tc16+tcN combinations
// =============================================================================
// TC16 KERNELS - Batch 3-31 with hierarchical grid tiling
// =============================================================================
// 16 separate kernels, one for each REMAINDER_BATCH (0-15).
// Each kernel contains tc16 + one specific tcN, no runtime switch.
// Eliminates L2 cache thrashing by keeping weights hot across all batch tiles.
// R=0: just tc16 with grid.y tiling
// R=1-15: tc16 + tcR (remainder handlers)
//
// HIERARCHICAL GRID (kernel_cache_design.md):
//   x = batch tiles (L1 scope)
//   y = row tiles (L2 scope)
//   z = wave index: row_group + batch_group × num_row_groups
// row_groups parameter enables decode: row_group = z % row_groups

#define INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, N) \
extern "C" __global__ void LAUNCH_BOUNDS_TC16 name##_tc16_##N( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int batch_size, const int row_groups) { \
    quantized_matmul_tc16_entry<qk, qi, block_type, vdr, act_t, dst_t, N>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size, row_groups); \
}

// Generate all 16 TC16 kernels (tc16_0=pure tc16, tc16_1-15=tc16+tcN)
#define INSTANTIATE_KERNEL_TC16(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 0) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 1) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 2) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 3) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 4) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 5) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 6) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 7) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 8) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 9) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 10) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 11) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 12) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 13) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 14) \
    INSTANTIATE_KERNEL_TC16_N(name, qk, qi, block_type, vdr, act_t, dst_t, 15)

// =============================================================================
// TC32 KERNELS - Greedy dispatch for tc32+tc16+tcN combinations
// =============================================================================
// 16 separate kernels, one for each REMAINDER_BATCH (0-15).
// R = batch_size % 16 (greedy decomposition computed internally)
// R=0: tc32 tiles + optional tc16 (e.g., batch 48 = tc32 + tc16)
// R=1-15: tc32 tiles + optional tc16 + tcR (e.g., batch 49 = tc32 + tc16 + tc1)
//
// HIERARCHICAL GRID (kernel_cache_design.md):
//   x = batch tiles (L1 scope)
//   y = row tiles (L2 scope)
//   z = wave index: row_group + batch_group × num_row_groups
// row_groups parameter enables decode: row_group = z % row_groups

#define INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, N) \
extern "C" __global__ void LAUNCH_BOUNDS_TC32 name##_tc32_##N( \
    const void * __restrict__ vx, const act_t * __restrict__ vy, dst_t * __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, \
    const int nrows_dst, const int batch_size, const int row_groups) { \
    quantized_matmul_tc32_entry<qk, qi, block_type, vdr, act_t, dst_t, N>( \
        vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size, row_groups); \
}

// Generate all 16 TC32 kernels (tc32_0 through tc32_15)
#define INSTANTIATE_KERNEL_TC32(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 0) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 1) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 2) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 3) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 4) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 5) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 6) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 7) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 8) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 9) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 10) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 11) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 12) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 13) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 14) \
    INSTANTIATE_KERNEL_TC32_N(name, qk, qi, block_type, vdr, act_t, dst_t, 15)

// All kernels - consolidated to tc16/tc32 kernels
// Reduced from 94 to 48 kernels:
//   - 8 GEMV (s1-s8)
//   - 8 Iterator (s2_iter2-8, s3_iter3)
//   - 16 tc16 (tc16_0-tc16_15 for batch 3-31)
//   - 16 tc32 (tc32_0-tc32_15 for batch 32+)
#define INSTANTIATE_KERNELS_BASE(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S1(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S2(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S3(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S4(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S5(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S6(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S7(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S8(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_TC16(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_TC32(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S2_ITER2(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S2_ITER3(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S2_ITER4(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S2_ITER5(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S2_ITER6(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S2_ITER7(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S2_ITER8(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNEL_S3_ITER3(name, qk, qi, block_type, vdr, act_t, dst_t)

// All kernels: 48 kernels per (qtype, ytype):
//   - 8 GEMV (s1-s8)
//   - 8 Iterator (s2_iter2-8, s3_iter3)
//   - 16 tc16 (tc16_0-tc16_15 for batch 3-31)
//   - 16 tc32 (tc32_0-tc32_15 for batch 32+)
//
// MoE grouped dispatch is handled at the dispatcher level by looping over
// sorted expert groups and calling run_quantized_matmul per expert, giving
// each expert full greedy decomposition with proper weight reuse.
#define INSTANTIATE_KERNELS(name, qk, qi, block_type, vdr, act_t, dst_t) \
    INSTANTIATE_KERNELS_BASE(name, qk, qi, block_type, vdr, act_t, dst_t)
