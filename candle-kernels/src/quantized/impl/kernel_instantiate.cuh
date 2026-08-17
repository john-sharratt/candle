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
// REGISTER-ONLY GEMV KERNELS (batch 1-8) — pre-SM80 fallback only
// These CUDA-core kernels are NOT used on tensor-core hardware: benchmarking
// found the TC kernels (tc16_N) faster than the GEMV path in every measured
// case, even single-token decode. On SM80+ the dispatcher routes all batch
// sizes to tensor cores; these remain solely for GPUs without tensor cores.
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

// =============================================================================
// GROUPED KERNEL - all MoE experts in one launch (grid = total_tiles × row_tiles)
// =============================================================================
// Single launch over a device-side (tile → expert, batch-slice) table; collapses
// the per-expert segment loop into one kernel that also fills the SMs across
// experts. See grouped_tc::quantized_matmul_grouped_entry in kernel.cuh.
#define INSTANTIATE_KERNEL_GROUPED(name, qk, qi, block_type, vdr, act_t, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_TC16 name##_grouped( \
    const uint64_t* __restrict__ weight_ptrs, \
    const int* __restrict__ tile_expert, \
    const int* __restrict__ tile_b_start, \
    const int* __restrict__ tile_b_cnt, \
    const act_t* __restrict__ vy, dst_t* __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int y_stride, const int dst_stride) { \
    grouped_tc::quantized_matmul_grouped_entry<qk, qi, block_type, vdr, act_t, dst_t>( \
        weight_ptrs, tile_expert, tile_b_start, tile_b_cnt, \
        vy, dst, ncols_x, nrows_x, y_stride, dst_stride); \
}

// INT8 kernels — q8a128 activations × quantized weights on the INT8 m16n8k32
// tensor core (grouped_tc_int8). Invoked explicitly for every format with an int8
// weight-unpack (all 14), NOT from INSTANTIATE_KERNELS_BASE.
//
// The dense and grouped entries are emitted by separate macros because they are
// consumed at different widths. A dense projection (q/k/v, o_proj, router,
// lm_head) is read back at the model's activation dtype, so it stores narrow and
// carries one entry per output dtype. The grouped (MoE) result feeds straight
// into the SwiGLU requantisation, where the wider F32 store is load-bearing, so
// it has exactly one entry.
//
//   name##_dense    — regular QMatMul: one weight, implicit tile schedule
//                     (blockIdx.x → the Bm=16 batch slice). Launched from
//                     run_quantized_matmul on ytype==3.
//   name##_dense_m2 — the same, N_SUB=2 (Bm=32): the weight chunk's dequant is
//                     reused across 2 token sub-tiles per block, halving the
//                     weight re-reads. The large-M (prefill) regime; the host
//                     launches it with grid.x = ceil(total_batch / 32).
//   name##_grouped  — MoE: device (tile→expert, batch-slice) table, single launch
//                     over all experts. Same ABI as INSTANTIATE_KERNEL_GROUPED.
#define INSTANTIATE_KERNEL_DENSE_INT8(name, qk, qi, block_type, vdr, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_TC16 name##_dense( \
    const void* __restrict__ weights, \
    const block_q8a128* __restrict__ vy, dst_t* __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int total_batch, \
    const int y_stride, const int dst_stride) { \
    grouped_tc::quantized_matmul_dense_entry_int8<qk, qi, block_type, vdr, dst_t, 1>( \
        reinterpret_cast<const block_compact_t<block_type>*>(weights), \
        vy, dst, ncols_x, nrows_x, total_batch, y_stride, dst_stride); \
}

#define INSTANTIATE_KERNEL_DENSE_INT8_M2(name, qk, qi, block_type, vdr, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_TC16 name##_dense_m2( \
    const void* __restrict__ weights, \
    const block_q8a128* __restrict__ vy, dst_t* __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int total_batch, \
    const int y_stride, const int dst_stride) { \
    grouped_tc::quantized_matmul_dense_entry_int8<qk, qi, block_type, vdr, dst_t, 2>( \
        reinterpret_cast<const block_compact_t<block_type>*>(weights), \
        vy, dst, ncols_x, nrows_x, total_batch, y_stride, dst_stride); \
}

// The three narrowed dense entries for one format: `base` names the format's int8
// kernel family without its output-dtype tag (e.g. q4_k_int8), so this emits
// base##_f16_dense, base##_bf16_dense and base##_f32_dense.
#define INSTANTIATE_KERNEL_DENSE_INT8_ALL(base, qk, qi, block_type, vdr) \
    INSTANTIATE_KERNEL_DENSE_INT8(base##_f16, qk, qi, block_type, vdr, half) \
    INSTANTIATE_KERNEL_DENSE_INT8(base##_bf16, qk, qi, block_type, vdr, __nv_bfloat16) \
    INSTANTIATE_KERNEL_DENSE_INT8(base##_f32, qk, qi, block_type, vdr, float)

#define INSTANTIATE_KERNEL_DENSE_INT8_M2_ALL(base, qk, qi, block_type, vdr) \
    INSTANTIATE_KERNEL_DENSE_INT8_M2(base##_f16, qk, qi, block_type, vdr, half) \
    INSTANTIATE_KERNEL_DENSE_INT8_M2(base##_bf16, qk, qi, block_type, vdr, __nv_bfloat16) \
    INSTANTIATE_KERNEL_DENSE_INT8_M2(base##_f32, qk, qi, block_type, vdr, float)

#define INSTANTIATE_KERNEL_GROUPED_INT8(name, qk, qi, block_type, vdr, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_TC16 name##_grouped( \
    const uint64_t* __restrict__ weight_ptrs, \
    const int* __restrict__ tile_expert, \
    const int* __restrict__ tile_b_start, \
    const int* __restrict__ tile_b_cnt, \
    const block_q8a128* __restrict__ vy, dst_t* __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int y_stride, const int dst_stride) { \
    /* N_SUB=2: mode-2 Bm=32 weight-reuse. Tiles are built ≤32 tokens/expert (cuda.rs); */ \
    /* a ≤16-token tile runs one sub-tile and writes nothing for the empty one. */ \
    grouped_tc::quantized_matmul_grouped_entry<qk, qi, block_type, vdr, block_q8a128, dst_t, 2>( \
        weight_ptrs, tile_expert, tile_b_start, tile_b_cnt, \
        vy, dst, ncols_x, nrows_x, y_stride, dst_stride); \
}

/* Wide-Bm grouped twins for the PREFILL regime (many rows per expert): each
 * weight chunk's load + dequant is reused across 4 / 8 token sub-tiles, cutting
 * the per-32-row weight re-streaming that dominates the routed GEMM once
 * rows-per-expert clears the tile width (decode stays on the N_SUB=2 form —
 * wide tiles there would MMA mostly zero-padding for the same weight traffic).
 * Bit-identical per output row to the N_SUB=2 kernel: the K-loop accumulation
 * order is unchanged; the sub-tile split only regroups which tokens share a
 * block. Relaxed launch bounds: the wide tiles hold 4·N_SUB accumulators per
 * thread and 2×(16·N_SUB)×KI8_STRIDE activation smem, so the TC16 10-block
 * register budget (~51/thread) would spill them.
 * The host picks the mode from rows-per-expert (cuda.rs
 * grouped_matmul_gemx_q8a128) and sizes the tile tables to 16·N_SUB. */
#define INSTANTIATE_KERNEL_GROUPED_INT8_M4(name, qk, qi, block_type, vdr, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_ITER name##_grouped_m4( \
    const uint64_t* __restrict__ weight_ptrs, \
    const int* __restrict__ tile_expert, \
    const int* __restrict__ tile_b_start, \
    const int* __restrict__ tile_b_cnt, \
    const block_q8a128* __restrict__ vy, dst_t* __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int y_stride, const int dst_stride) { \
    grouped_tc::quantized_matmul_grouped_entry<qk, qi, block_type, vdr, block_q8a128, dst_t, 4>( \
        weight_ptrs, tile_expert, tile_b_start, tile_b_cnt, \
        vy, dst, ncols_x, nrows_x, y_stride, dst_stride); \
}

#define INSTANTIATE_KERNEL_GROUPED_INT8_M8(name, qk, qi, block_type, vdr, dst_t) \
extern "C" __global__ void LAUNCH_BOUNDS_VSMALL name##_grouped_m8( \
    const uint64_t* __restrict__ weight_ptrs, \
    const int* __restrict__ tile_expert, \
    const int* __restrict__ tile_b_start, \
    const int* __restrict__ tile_b_cnt, \
    const block_q8a128* __restrict__ vy, dst_t* __restrict__ dst, \
    const int ncols_x, const int nrows_x, const int y_stride, const int dst_stride) { \
    grouped_tc::quantized_matmul_grouped_entry<qk, qi, block_type, vdr, block_q8a128, dst_t, 8>( \
        weight_ptrs, tile_expert, tile_b_start, tile_b_cnt, \
        vy, dst, ncols_x, nrows_x, y_stride, dst_stride); \
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
    INSTANTIATE_KERNEL_GROUPED(name, qk, qi, block_type, vdr, act_t, dst_t) \
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
