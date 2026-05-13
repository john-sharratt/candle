#pragma once

// =============================================================================
// TILE PROCESSING KERNELS FOR BATCHED QUANTIZED MATVEC (NON-TC PATH)
// =============================================================================
// This file contains tile processing functions for the CUDA-core (non-TC) path.
// These functions use vec_dot loaders for scalar FMA-based computation.
//
// FOR TENSOR CORE PATH: See pipeline.cuh which has its own compute_tile()
// that uses MMA instructions on dequantized data in smem.
//
// REGISTER PATH (process_tile_register, process_tile_register_partial)
//   - Weights loaded from global memory per-row
//   - Uses INLINE loader for weight reuse across batch elements
//   - No smem for weights, minimal footprint
//
// LOADER TYPES:
// -------------
// INLINE (_inline): Store packed quants + scale, dequantize in dot_y()
//   - Lower register pressure, used by register path
//
// K/128 FORMAT (ALL TYPES)
// -------------------------
// All quant types now use K/128 blocks with embedded scales and NUM_PARTS=1:
//   - 16 threads per K/128 block
//   - Each thread handles 8 elements (128 total)
//   - Single load_part<0>() + dot_y<0>() call per block
//
// =============================================================================

#include "math.cuh"
#include "impl/common.cuh"  // Block type definitions
#include "loaders.cuh"  // vec_dot_loader_for<>, loader_num_parts_v

// ============================================================================
// SPLIT-LOAD DOT PRODUCT HELPERS
// ============================================================================
// These helper templates dispatch to the loader's load_part<N>/dot_y<N> methods.
// With NUM_PARTS=1 for all K/128 formats, this results in a single iteration:
//   load_part<0>(x, row, kbx, num_rows)
//   dot_y<0>(y) -> accumulated
// ============================================================================

// ============================================================================
// RECURSIVE DOT LOOP HELPER
// ============================================================================
// Matches the DequantLoop pattern in dequantize.cuh
// Processes parts 0 to NumParts-1 using load_part<N>/dot_y<N> interface
// Must be defined BEFORE split_load_dot specializations that use it.
//
// GEMX [K, N] layout indexing:
//   - row: output row index (0 to num_rows-1)
//   - kbx: block column index (0 to blocks_per_row-1)
//   - num_rows: total number of rows (for column-major index computation)

template <typename loader_t, typename block_c_t, typename act_t, typename acc_t, int N, int NumParts>
struct DotLoop {
    __device__ __forceinline__ static acc_t compute(
        loader_t& loader,
        const block_c_t* __restrict__ x,
        const int row,
        const int kbx,
        const int num_rows,
        const act_t* __restrict__ y
    ) {
        loader.template load_part<N>(x, row, kbx, num_rows);
        // Call dot_y<N, acc_t> with explicit acc_t for proper type handling
        acc_t contrib = loader.template dot_y<N, acc_t>(y);
        return contrib + DotLoop<loader_t, block_c_t, act_t, acc_t, N + 1, NumParts>::compute(
            loader, x, row, kbx, num_rows, y);
    }
};

// Base case: N == NumParts, stop recursion
template <typename loader_t, typename block_c_t, typename act_t, typename acc_t, int NumParts>
struct DotLoop<loader_t, block_c_t, act_t, acc_t, NumParts, NumParts> {
    __device__ __forceinline__ static acc_t compute(
        loader_t&, const block_c_t*, int, int, int, const act_t*
    ) {
        return acc_zero<acc_t>(); // Zero contribution
    }
};

// Primary template (default case - uses DotLoop for any NUM_PARTS)
// All NUM_PARTS values (1, 2, 4, 8) use the same recursive DotLoop pattern
template <typename block_q_t, typename loader_t, typename block_c_t, typename act_t, typename acc_t, int NUM_PARTS = loader_num_parts_v<block_q_t>>
struct split_load_dot {
    __device__ __forceinline__ static acc_t compute(
        loader_t& loader,
        const block_c_t* __restrict__ x,
        const int row,
        const int kbx,
        const int num_rows,
        const act_t* __restrict__ y
    ) {
        return DotLoop<loader_t, block_c_t, act_t, acc_t, 0, NUM_PARTS>::compute(
            loader, x, row, kbx, num_rows, y);
    }
};

// Convenience function for the common case
template <typename block_q_t, typename acc_t, typename loader_t, typename block_c_t, typename act_t>
__device__ __forceinline__ acc_t compute_split_dot(
    loader_t& loader,
    const block_c_t* __restrict__ x,
    const int row,
    const int kbx,
    const int num_rows,
    const act_t* __restrict__ y
) {
    return split_load_dot<block_q_t, loader_t, block_c_t, act_t, acc_t>::compute(
        loader, x, row, kbx, num_rows, y);
}

// ============================================================================
// REGISTER PATH: Full tile (compile-time iteration count)
// ============================================================================
// Weights loaded from global memory. Loader stores decoded weights in registers
// and reuses them across all BATCH_TILE batch elements.
// Greedy decomposition guarantees batch_count == BATCH_TILE (no partial batches).
//
// GEMX [K, N] layout: weights are stored column-major with embedded scales.
// Best for small batch tiles (1-8) where smem overhead isn't worthwhile.
//
// PASS-BY-VALUE: RegArray passed by value to avoid address-taking
// which forces stack allocation. Returns the modified RegArray.
//
// Y POINTER COMPUTATION:
// Instead of pre-computed y_tiles pointers, we take base pointer + strides:
//   y_base[b] = vy[b * y_stride_per_row]  (base for batch b)
//   tile_y_offset = tile_start * y_stride_per_block  (already added by caller)
//   Final access: y_base[b] + ky_local
//
// For iter kernel, batch_y_offset includes (iter * BATCH_TILE * y_stride_per_row).
template <int qk, int qi, typename block_q_t, int vdr, typename act_t,
          typename acc_t, int nwarps, int TILE_BLOCKS, int BATCH_TILE, int ROWS_PER_PHASE>
__device__ __forceinline__ RegArray<acc_t, BATCH_TILE, ROWS_PER_PHASE>
process_tile_register(
    const block_compact_t<block_q_t> * __restrict__ x, // Global weights (compacted blocks with embedded scales)
    const act_t * __restrict__ vy,                    // Base activation pointer
    const int y_stride_per_row,                       // Stride between batch elements in Y
    const int batch_y_offset,                         // Y offset for batch 0 (includes iter offset for iter kernel)
    const int tile_y_offset,                          // Y offset for this tile (tile_start * y_stride_per_block)
    RegArray<acc_t, BATCH_TILE, ROWS_PER_PHASE> tmp,  // BY VALUE - accumulator array
    const int tile_start,                             // First block index of this tile
    const int row0,                                   // First row of this phase
    const int num_rows,                               // Total rows (for GEMX indexing)
    const int tid)                                    // Thread ID within block
{
    using loader_t = typename vec_dot_loader_for<block_q_t, vdr, act_t>::type;
    using block_c_t = block_compact_t<block_q_t>;   // Compacted block type
    
    constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;
    
    const int thread_offset = tid / (qi / vdr);
    
    constexpr int iters_per_tile = (TILE_BLOCKS + blocks_per_iter - 1) / blocks_per_iter;
    constexpr int y_stride_per_block = std::is_same_v<act_t, block_q8_1> ? (qk / QK8_1) : qk;
    
    // Compute at compile time whether kbx_local < TILE_BLOCKS is always true.
    // This allows nvcc to fully unroll when no runtime branch is needed.
    // max_thread_offset = (total_threads - 1) / threads_per_block
    // max_kbx_local = max_thread_offset + (iters_per_tile - 1) * blocks_per_iter
    constexpr int threads_per_block_element = qi / vdr;
    constexpr int max_thread_offset = (nwarps * WARP_SIZE - 1) / threads_per_block_element;
    constexpr int max_kbx_local = max_thread_offset + (iters_per_tile - 1) * blocks_per_iter;
    constexpr bool always_in_bounds = max_kbx_local < TILE_BLOCKS;
    
    loader_t loader;
    
    // Use explicit unroll count. Combined with if constexpr removing the branch,
    // this should enable full unrolling.
    #pragma unroll
    for (int iter = 0; iter < iters_per_tile; ++iter) {
        const int kbx_local = thread_offset + iter * blocks_per_iter;
        
        // Use if constexpr to eliminate branch when provably always in bounds.
        // This enables full loop unrolling for small batch kernels (s1, s2, etc.)
        if constexpr (always_in_bounds) {
            const int kbx = tile_start + kbx_local;
            const int ky_local = kbx_local * y_stride_per_block;
            
            // ROWS unroll: s1=4, s2=2, s3+=1 for balance of ILP vs register pressure
            #pragma unroll ((4 / BATCH_TILE) > 0 ? (4 / BATCH_TILE) : 1)
            for (int i = 0; i < ROWS_PER_PHASE; ++i) {
                const int row = row0 + i;
                
                // BATCH unroll: always full for weight reuse
                #pragma unroll
                for (int b = 0; b < BATCH_TILE; ++b) {
                    // Y pointer: base + batch offset + tile offset + local K offset
                    const act_t* y_ptr = &vy[batch_y_offset + b * y_stride_per_row + tile_y_offset + ky_local];
                    acc_t dot = compute_split_dot<block_q_t, acc_t>(
                        loader, x,
                        row, kbx, num_rows, y_ptr);
                    accumulate(tmp(b, i), dot);
                }
            }
        } else {
            if (kbx_local < TILE_BLOCKS) {
                const int kbx = tile_start + kbx_local;
                const int ky_local = kbx_local * y_stride_per_block;
                
                #pragma unroll ((4 / BATCH_TILE) > 0 ? (4 / BATCH_TILE) : 1)
                for (int i = 0; i < ROWS_PER_PHASE; ++i) {
                    const int row = row0 + i;
                    
                    #pragma unroll
                    for (int b = 0; b < BATCH_TILE; ++b) {
                        const act_t* y_ptr = &vy[batch_y_offset + b * y_stride_per_row + tile_y_offset + ky_local];
                        acc_t dot = compute_split_dot<block_q_t, acc_t>(
                            loader, x,
                            row, kbx, num_rows, y_ptr);
                        accumulate(tmp(b, i), dot);
                    }
                }
            }
        }
    }
    return tmp;
}

// ============================================================================
// REGISTER PATH: Partial tile (runtime iteration count)
// ============================================================================
// Same as process_tile_register but with runtime tile_blocks for remainder.
// NOTE: No TILE_BLOCKS template param - not needed since iteration is runtime.
// Greedy decomposition guarantees batch_count == BATCH_TILE (no partial batches).
//
// GEMX [K, N] layout: weights are stored column-major with embedded scales.
//
// PASS-BY-VALUE: RegArray passed by value to avoid address-taking.
template <int qk, int qi, typename block_q_t, int vdr, typename act_t,
          typename acc_t, int nwarps, int BATCH_TILE, int ROWS_PER_PHASE>
__device__ __forceinline__ RegArray<acc_t, BATCH_TILE, ROWS_PER_PHASE>
process_tile_register_partial(
    const block_compact_t<block_q_t> * __restrict__ x, // Global weights (compacted blocks with embedded scales)
    const act_t * __restrict__ vy,                    // Base activation pointer
    const int y_stride_per_row,                       // Stride between batch elements in Y
    const int batch_y_offset,                         // Y offset for batch 0 (includes iter offset for iter kernel)
    const int tile_y_offset,                          // Y offset for this tile (tile_start * y_stride_per_block)
    RegArray<acc_t, BATCH_TILE, ROWS_PER_PHASE> tmp,  // BY VALUE - accumulator array
    const int tile_start,                             // First block index of this tile
    const int tile_blocks,                            // Actual blocks in this tile
    const int row0,                                   // First row of this phase
    const int num_rows,                               // Total rows (for GEMX indexing)
    const int tid)                                    // Thread ID within block
{
    using loader_t = typename vec_dot_loader_for<block_q_t, vdr, act_t>::type;
    using block_c_t = block_compact_t<block_q_t>;   // Compacted block type
    
    constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;
    
    const int thread_offset = tid / (qi / vdr);
    
    constexpr int y_stride_per_block = std::is_same_v<act_t, block_q8_1> ? (qk / QK8_1) : qk;
    
    loader_t loader;
    
    #pragma unroll
    for (int kbx_local = thread_offset; kbx_local < tile_blocks; kbx_local += blocks_per_iter) {
        const int kbx = tile_start + kbx_local;
        const int ky_local = kbx_local * y_stride_per_block;
        
        // ROWS unroll: s1=4, s2=2, s3+=1 for balance of ILP vs register pressure
        #pragma unroll ((4 / BATCH_TILE) > 0 ? (4 / BATCH_TILE) : 1)
        for (int i = 0; i < ROWS_PER_PHASE; ++i) {
            const int row = row0 + i;
            
            // BATCH unroll: always full for weight reuse
            #pragma unroll
            for (int b = 0; b < BATCH_TILE; ++b) {
                const act_t* y_ptr = &vy[batch_y_offset + b * y_stride_per_row + tile_y_offset + ky_local];
                acc_t dot = compute_split_dot<block_q_t, acc_t>(
                    loader, x,
                    row, kbx, num_rows, y_ptr);
                accumulate(tmp(b, i), dot);
            }
        }
    }
    return tmp;
}