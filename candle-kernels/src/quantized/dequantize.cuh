#pragma once

// =============================================================================
// DEQUANTIZE KERNELS FOR REPACKED (K/128) QUANTIZED TENSORS
// =============================================================================
// Dequantizes repacked quantized blocks back to float32 using the actual
// matmul loaders. This ensures we test the exact same element mapping logic
// that the matmul kernel uses.
//
// Each loader's dequant<N>() method writes dequantized values to the output
// buffer at the same positions that dot_y() would read from Y.
//
// Output: float32 tensor with same shape as original (nrows × ncols)
//
// K/128 FORMAT:
// -------------
// All quant types use K/128 blocks with embedded scales (NUM_PARTS=1):
//   - 16 threads per K/128 block
//   - Each thread handles 8 elements (128 total)
//   - Scales embedded inline with quantized data
//
// Pattern for all types:
//   load_part<N>(x, row, kbx, num_rows)  // Load data into loader fields
//   dequant<N>(out)                       // Dequantize from loader fields to out
//
// The unified dequantize_impl<> template uses loader_num_parts trait (=1)
// to dispatch a single load/dequant iteration per block.
// =============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// Include common types first - defines block types, QK constants, etc.
#include "impl/common.cuh"

// Include all loaders - dequantize_impl needs the loader traits for each block type
// This is a "fat" header but dequantize.cu is the only consumer and it needs
// all loaders anyway (single compilation unit for all dequant kernels).
#include "loaders.cuh"
#include "loader/q4_0.cuh"
#include "loader/q4_1.cuh"
#include "loader/q5_0.cuh"
#include "loader/q5_1.cuh"
#include "loader/q8_0.cuh"
#include "loader/q8_1.cuh"
#include "loader/q2_K.cuh"
#include "loader/q3_K.cuh"
#include "loader/q4_K.cuh"
#include "loader/q5_K.cuh"
#include "loader/q6_K.cuh"

// =============================================================================
// BLOCK SIZE CONSTANTS
// =============================================================================

#ifndef QK_K
#define QK_K 256
#endif

#ifndef QK4_0
#define QK4_0 32
#endif

#ifndef QK4_1
#define QK4_1 32
#endif

#ifndef QK5_0
#define QK5_0 32
#endif

#ifndef QK5_1
#define QK5_1 32
#endif

#ifndef QK8_0
#define QK8_0 32
#endif

// Note: loader_num_parts trait is defined in loaders.cuh

// =============================================================================
// RECURSIVE LOOP HELPER
// =============================================================================
// Processes parts 0 to NumParts-1 using the unified load_part<N>/dequant<N> interface
//
// For GEMX [K/block, N] layout:
//   - row: output row index (0 to num_rows-1)
//   - kbx: block column index (0 to blocks_per_row-1)  
//   - num_rows: total number of rows (for column-major index computation)
//
// Weight block index: kbx * num_rows + row
// Scale index: (kbx * scales_per_block + offset) * num_rows + row
// Note: K/128 blocks have embedded scales - no external scales parameter needed.

template <typename BlockC, typename Loader, int N, int NumParts>
struct DequantLoop {
    __device__ static void run(Loader& loader, const BlockC* x,
                               float* out, int row, int kbx, int num_rows) {
        // Load part N (scales are embedded in K/128 blocks)
        loader.template load_part<N>(x, row, kbx, num_rows);
        // Dequantize part N (data already loaded into loader fields)
        loader.template dequant<N>(out);
        // Continue to next part
        DequantLoop<BlockC, Loader, N + 1, NumParts>::run(loader, x, out, row, kbx, num_rows);
    }
};

// Base case: N == NumParts, stop recursion
template <typename BlockC, typename Loader, int NumParts>
struct DequantLoop<BlockC, Loader, NumParts, NumParts> {
    __device__ static void run(Loader&, const BlockC*, float*, int, int, int) {
        // Done - all parts processed
    }
};

// =============================================================================
// UNIFIED DEQUANTIZE IMPLEMENTATION
// =============================================================================
// Single template using recursive loop over all parts
// Note: K/128 blocks have embedded scales - no external scales parameter needed.

template <typename block_q_t, typename block_q_c_t, int QK, int qi, int vdr>
__device__ void dequantize_impl(
    const block_q_c_t* __restrict__ x,
    float* __restrict__ out,
    int nrows,
    int ncols
) {
    constexpr int num_parts = loader_num_parts_v<block_q_t>;
    constexpr int threads_per_block = qi / vdr;
    
    const int blocks_per_row = ncols / QK;
    const int total_blocks = nrows * blocks_per_row;
    const int total_threads = total_blocks * threads_per_block;
    
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid >= total_threads) return;
    
    const int block_idx = gtid / threads_per_block;
    
    const int row = block_idx / blocks_per_row;
    const int kbx = block_idx % blocks_per_row;  // column block index
    
    float* block_out = out + row * ncols + kbx * QK;
    
    using loader_t = typename vec_dot_loader_for<block_q_t, vdr, float>::type;
    loader_t loader;
    
    // Recursive loop processes parts 0 through num_parts-1
    // Pass (row, kbx, nrows) for Marlin [K/block, N] layout indexing
    // Scales are embedded in K/128 blocks - no external scales parameter
    DequantLoop<block_q_c_t, loader_t, 0, num_parts>::run(
        loader, x, block_out, row, kbx, nrows);
}

// =============================================================================
// CONCRETE IMPLEMENTATIONS FOR EACH QUANT TYPE
// =============================================================================
// All formats now use the unified dequantize_impl<> template with recursive loop
// Input is compacted block type with embedded scales (K/128 format).

// Q4_0 - K/128 FORMAT (NUM_PARTS=1, vdr=2)
// Threads handle K/128 blocks with embedded scales.
// 16 threads per block: qi=32, vdr=2 → threads_per_block = 32/2 = 16
__device__ void dequantize_q4_0_impl(
    const block_c_q4_0* __restrict__ x,
    float* __restrict__ out,
    int nrows, int ncols
) {
    // K/128: QK=128, qi=32 (for 16 threads), vdr=2
    dequantize_impl<block_q4_0, block_c_q4_0, 128, 32, 2>(x, out, nrows, ncols);
}

// Q4_1 - K/128 FORMAT (NUM_PARTS=1, vdr=2)
// 16 threads per block: qi=32, vdr=2 → threads_per_block = 32/2 = 16
__device__ void dequantize_q4_1_impl(
    const block_c_q4_1* __restrict__ x,
    float* __restrict__ out,
    int nrows, int ncols
) {
    dequantize_impl<block_q4_1, block_c_q4_1, 128, 32, 2>(x, out, nrows, ncols);
}

// Q5_0 - K/128 FORMAT (NUM_PARTS=1, vdr=2)
// 16 threads per block: qi=32, vdr=2 → threads_per_block = 32/2 = 16
__device__ void dequantize_q5_0_impl(
    const block_c_q5_0* __restrict__ x,
    float* __restrict__ out,
    int nrows, int ncols
) {
    dequantize_impl<block_q5_0, block_c_q5_0, 128, 32, 2>(x, out, nrows, ncols);
}

// Q5_1 - K/128 FORMAT (NUM_PARTS=1, vdr=2)
// 16 threads per block: qi=32, vdr=2 → threads_per_block = 32/2 = 16
__device__ void dequantize_q5_1_impl(
    const block_c_q5_1* __restrict__ x,
    float* __restrict__ out,
    int nrows, int ncols
) {
    dequantize_impl<block_q5_1, block_c_q5_1, 128, 32, 2>(x, out, nrows, ncols);
}

// Q8_0 - K/128 FORMAT (NUM_PARTS=1, vdr=1)
// 16 threads per block: qi=16, vdr=1 → threads_per_block = 16/1 = 16
__device__ void dequantize_q8_0_impl(
    const block_c_q8_0* __restrict__ x,
    float* __restrict__ out,
    int nrows, int ncols
) {
    dequantize_impl<block_q8_0, block_c_q8_0, 128, 16, 1>(x, out, nrows, ncols);
}

// Q2_K - K/128 FORMAT (NUM_PARTS=1, vdr=1)
__device__ void dequantize_q2_K_impl(
    const block_c_q2_K* __restrict__ x,
    float* __restrict__ out,
    int nrows, int ncols
) {
    dequantize_impl<block_q2_K, block_c_q2_K, 128, QI2_K, 1>(x, out, nrows, ncols);
}

// Q3_K - K/128 FORMAT (NUM_PARTS=1, vdr=1)
__device__ void dequantize_q3_K_impl(
    const block_c_q3_K* __restrict__ x,
    float* __restrict__ out,
    int nrows, int ncols
) {
    dequantize_impl<block_q3_K, block_c_q3_K, 128, QI3_K, 1>(x, out, nrows, ncols);
}

// Q4_K - K/128 FORMAT (NUM_PARTS=1, vdr=2)
// Uses specialized loader for K/128 layout with embedded scales.
__device__ void dequantize_q4_K_impl(
    const block_c_q4_K* __restrict__ x,
    float* __restrict__ out,
    int nrows, int ncols
) {
    dequantize_impl<block_q4_K, block_c_q4_K, 128, QI4_K, 2>(x, out, nrows, ncols);
}

// Q5_K - K/128 FORMAT (NUM_PARTS=1, vdr=2)
__device__ void dequantize_q5_K_impl(
    const block_c_q5_K* __restrict__ x,
    float* __restrict__ out,
    int nrows, int ncols
) {
    dequantize_impl<block_q5_K, block_c_q5_K, 128, QI5_K, 2>(x, out, nrows, ncols);
}

// Q6_K - K/128 FORMAT (NUM_PARTS=1, vdr=2)
__device__ void dequantize_q6_K_impl(
    const block_c_q6_K* __restrict__ x,
    float* __restrict__ out,
    int nrows, int ncols
) {
    dequantize_impl<block_q6_K, block_c_q6_K, 128, QI6_K, 2>(x, out, nrows, ncols);
}
