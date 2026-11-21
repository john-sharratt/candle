#pragma once

// =============================================================================
// BATCHED QUANTIZED MATRIX-VECTOR MULTIPLY KERNEL
// =============================================================================
// 
// Computes: dst[batch, row] = sum_k( X[row, k] * Y[batch, k] )
//   - X: quantized weight matrix [nrows_x, ncols_x] in TC-repacked column-major format
//   - Y: activation vectors [batch_size, ncols_x] (float/half/bf16/fp8 or q8_1)
//   - dst: output [batch_size, nrows_x]
//
// KERNEL VARIANTS (greedy decomposition):
//   _s1: BATCH_TILE=1  - single vector, minimal registers
//   _s2: BATCH_TILE=2  - 2 vectors batched
//   _s3: BATCH_TILE=3  - 3 vectors batched
//   _s4: BATCH_TILE=4  - 4 vectors batched (sweet spot for register usage)
//   _s5-_s8: BATCH_TILE=5-8 - larger batches, may spill due to 128 reg limit
//
// PASS-BY-VALUE OPTIMIZATION:
//   Accumulators and pointer arrays are wrapped in structs (RegArray, YTiles)
//   and passed by value. This prevents nvcc from forcing stack allocation
//   when addresses would otherwise be taken for pointer parameters.
//   Result: _s1-_s4 kernels achieve ZERO stack spills.
//
// =============================================================================

// -----------------------------------------------------------------------------
// RegArray: Accumulator wrapper for pass-by-value semantics
// -----------------------------------------------------------------------------
// Without this wrapper, `acc_t tmp[B][R]` passed to functions forces nvcc to
// allocate stack space even when registers are available. By wrapping in a
// struct and passing/returning by value, the compiler keeps data in registers.
//
// Usage: RegArray<half, 4, 16> tmp = {};  // 4 batches × 16 rows
//        tmp(b, r) = value;               // access element
//        tmp = process_tile(..., tmp);    // pass by value, return modified
// -----------------------------------------------------------------------------
template<typename T, int B, int R>
struct RegArray {
    T data[B][R];
    __device__ __forceinline__ T& operator()(int b, int r) { return data[b][r]; }
    __device__ __forceinline__ const T& operator()(int b, int r) const { return data[b][r]; }
};

// -----------------------------------------------------------------------------
// YTiles: Pointer array wrapper for pass-by-value semantics  
// -----------------------------------------------------------------------------
// Same principle as RegArray but for the Y tile pointer array. Passed by value
// to callees that only read (don't modify) the pointers.
// -----------------------------------------------------------------------------
template<typename T, int B>
struct YTiles {
    T data[B];
    __device__ __forceinline__ T& operator[](int b) { return data[b]; }
    __device__ __forceinline__ const T& operator[](int b) const { return data[b]; }
};

// =============================================================================
// INCLUDES
// =============================================================================
#include <cassert>
#include "types.cuh"
#include "math.cuh"
#include "loaders.cuh"
#include "process_tile.cuh"
#include "loader/gemx_dequant.cuh"  // gemx_dequant_traits base template
#include "mma_dequant.cuh"
#include "../dequant/dequant.cuh"
#include "../mma/mma_wrappers.cuh"  // fused_attn INT8 MMA wrappers + frag loaders (grouped_tc int8 path)
#include "../fast_exp.cuh"           // the SiLU the fused-activation loader applies
#include "../quantize/q8a128_tile.cuh" // the one q8a128 quantization, for that loader
#include "moe_live.cuh"                 // the grouped entry's workers over a live expert table
#include "../moe_read_ahead.cuh"        // the gate launch's read-ahead items
#include "splitk_segs.cuh"              // the split-K dense entry's weight segments
#include "pdl.cuh"                      // programmatic dependent launch: prefetch, then wait

#include "kernel/batched_matvec_kernels.cuh"


// =============================================================================
// S16_TC: TENSOR CORE KERNEL FOR BATCH >= 16
// =============================================================================
//
// OPERATION: dst[batch, row] = Σ_k Y[batch, k] × W[row, k]
//
// MMA m16n8k16: C[m,n] += A[m,k] × B[k,n]
//   A = activations [16 batch, 16 K]  - from shared memory  
//   B = weights [16 K, 8 rows]        - dequantized via loader traits
//   C = output [16 batch, 8 rows]     - accumulated in registers
//
// ARCHITECTURE:
//   - 128 threads = 4 warps
//   - Each warp handles one m16n8 output tile (16 batches × 8 rows)
//   - 4 warps process 4 n8 tiles = 32 rows per block
//   - Weight layout: K-major [K/128, N] - each K/128 block is 80 bytes
//
// THREAD MAPPING (per warp, m16n8k16):
//   Lane 0-3:   Row 0, K_GROUP 0-3  (K positions 0-15)
//   Lane 4-7:   Row 1, K_GROUP 0-3
//   ...
//   Lane 28-31: Row 7, K_GROUP 0-3
//
//   Each thread holds 4 weight elements for one row at one K/16 position.
//   The loader's dequant_for_mma_k16 extracts these with LOP3 dequant.
//
// =============================================================================

// =============================================================================
// TC_COMMON: SHARED CONSTANTS AND FUNCTIONS FOR TENSOR CORE KERNELS
// =============================================================================
namespace tc_common {

#include "kernel/tc_common_helpers.cuh"

} // namespace tc_common

#include "kernel/s16_tc_kernel.cuh"

#include "kernel/moe_live_device_helpers.cuh"

namespace grouped_tc {
#include "kernel/grouped_tc_loaders.cuh"

#include "kernel/int8_matmul_entries.cuh"

#include "kernel/moe_live_worker_and_entry.cuh"

} // namespace grouped_tc

// The decode-width dense entry that walks K across the warps of one block (grouped_tc).
#include "dense_narrow.cuh"

#include "kernel/sn_tc_and_tc16_dispatch.cuh"


// =============================================================================
// S32_TC: TENSOR CORE KERNEL FOR BATCH_TILE=32
// =============================================================================
//
// OPERATION: dst[batch, row] = Σ_k Y[batch, k] × W[row, k]
//
// This kernel processes 32 batches per tile using 2 sequential MMA m16n8k16
// operations (batches 0-15, then batches 16-31). This is a direct fork of
// s16_tc that can be customized for larger batch tiles.
//
// Key differences from s16_tc:
//   - BATCH_TILE = 32 (2 × 16 MMA tiles in M dimension)
//   - Loads 32 batches of activations per K tile
//   - Runs 2 sets of MMA operations per K iteration
//
// Grid layout: (row_blocks, batch_tiles) where batch_tiles = ceil(batch/32)
// Block: 128 threads (4 warps × 32 threads)
// Shared memory: 32×136 for activations, 32×block_bytes for weights
// =============================================================================

namespace s32_tc {

#include "kernel/s32_tc_kernel.cuh"

} // namespace s32_tc


// =============================================================================
// UNIFIED TC32 KERNEL DISPATCHER
// =============================================================================
// Single kernel entry point for batch 32+ with greedy internal decomposition:
//   - tc32 tiles handle multiples of 32
//   - tc16 tile handles overflow >= 16 (if remainder_32 >= 16)
//   - tcR handles final remainder 0-15
//
// HIERARCHICAL GRID (kernel_cache_design.md):
//   x = batch tiles (L1 scope) - consecutive blocks share weights in L1
//   y = row tiles (L2 scope) - wave fills all SMs sharing activations
//   z = wave index: row_group + batch_group × num_row_groups
//
// HAS_REMAINDER: whether R = batch_size % 16 is nonzero
//   - false: Only tc32+tc16 tiles (no tcR remainder)
//   - true: tc32+tc16 tiles + one tcR remainder tile
//
// SMEM OPTIMIZATION: Declares unified SMEM once (tc32 size = 32×136) and passes
// to _impl functions to avoid NVCC summing SMEM from multiple __shared__ declarations.
// tc16 and tcN use the first 16 rows of smem_A, tc32 uses all 32.
// =============================================================================

namespace tc32 {

using namespace s32_tc;

// Each unified kernel has tc32 + optional tc16 + (when HAS_REMAINDER) one tcR
// tile whose R = batch_size % 16 is read at runtime.
// HIERARCHICAL GRID: z = row_group + batch_group × num_row_groups (rows inner, batches outer)
template <typename block_c_t, typename compute_t, typename act_t, typename output_t, bool HAS_REMAINDER>
__device__ void dispatch_tc32_tc16_tcN(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride,
    int batch_size, int row_groups)
{
    // Unified SMEM declaration - tc32 size covers tc32, tc16, and tcN paths
    constexpr int K128_BYTES = sizeof(block_c_t);
    __shared__ compute_t smem_A[BATCH_TILE_32][K_STRIDE];  // 32 × 136 (tc32 size)
    __shared__ uint8_t smem_W_flat[N_TILE * K128_BYTES];   // 32 × block_bytes
    
    // =========================================================================
    // HIERARCHICAL GRID DECODE
    // =========================================================================
    // z = row_group + batch_group × num_row_groups (rows inner, batches outer)
    const int row_group = blockIdx.z % row_groups;
    const int batch_group = blockIdx.z / row_groups;
    
    // Decode actual tile indices from hierarchical grid
    const int row_tile_idx = row_group * gridDim.y + blockIdx.y;
    const int batch_tile_idx = batch_group * gridDim.x + blockIdx.x;
    
    // Greedy decomposition: total batch tiles
    const int tc32_tiles = batch_size / 32;
    const int remainder_32 = batch_size % 32;
    const int has_tc16 = (remainder_32 >= 16) ? 1 : 0;
    const int total_batch_tiles = tc32_tiles + has_tc16 + (HAS_REMAINDER ? 1 : 0);
    const int row_tiles = (nrows + N_TILE - 1) / N_TILE;
    
    // Early exit if out of bounds
    if (row_tile_idx >= row_tiles || batch_tile_idx >= total_batch_tiles) return;
    
    // Dispatch based on batch_tile_idx position
    if (batch_tile_idx < tc32_tiles) {
        // tc32 tile: handles 32 batches
        const int batch_start = batch_tile_idx * 32;
        s32_tc::tc32_kernel_impl<block_c_t, compute_t, act_t, output_t>(
            weights, activations, dst, ncols, nrows, y_stride, dst_stride, 
            batch_size, smem_A, smem_W_flat, batch_start, batch_tile_idx, 1,
            row_tile_idx);
    } else if (has_tc16 && batch_tile_idx == tc32_tiles) {
        // tc16 tile: handles 16 batches
        // Uses first 16 rows of smem_A
        const int batch_start = tc32_tiles * 32;
        s16_tc::tc16_kernel_impl<block_c_t, compute_t, act_t, output_t>(
            weights, activations, dst, ncols, nrows, y_stride, dst_stride,
            batch_size, smem_A, smem_W_flat, batch_start, batch_tile_idx, 1,
            row_tile_idx);
    } else {
        // tcR remainder tile (only with a remainder)
        if constexpr (HAS_REMAINDER) {
            const int batch_start = tc32_tiles * 32 + has_tc16 * 16;
            sN_tc::tcN_kernel_impl<block_c_t, compute_t, act_t, output_t>(
                weights, activations, dst, ncols, nrows, y_stride, dst_stride,
                smem_A, smem_W_flat, batch_size - batch_start, batch_start,
                row_tile_idx);
        }
    }
}

} // namespace tc32


// =============================================================================
// quantized_matmul_tc32_entry - TC32 kernel entry point
// =============================================================================
// Handles all TC for batch 32+ with greedy internal decomposition:
//   - tc32 tiles handle multiples of 32
//   - tc16 tile handles remainder >= 16 (decomposes to 16 + final_remainder)
//   - tcR handles final remainder 0-15
//
// HAS_REMAINDER: whether R = batch_size % 16 is nonzero
//   - false: Only tc32+tc16 tiles (e.g., batch 48 = tc32 + tc16)
//   - true: tc32+tc16 tiles + tcR (e.g., batch 49 = tc32 + tc16 + tc1)
//
// HIERARCHICAL GRID (kernel_cache_design.md):
//   x = batch tiles (L1 scope) - consecutive blocks share weights in L1
//   y = row tiles (L2 scope) - wave fills all SMs sharing activations
//   z = wave index: row_group + batch_group × num_row_groups
// row_groups parameter enables decode: row_group = z % row_groups
// =============================================================================
template <int qk, int qi, typename block_q_t, int vdr,
          typename act_t, typename output_t, bool HAS_REMAINDER>
static __device__ void quantized_matmul_tc32_entry(
    const void * __restrict__ vx,
    const act_t * __restrict__ vy,
    output_t * __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int nrows_y,
    const int nrows_dst,
    const int batch_size,
    const int row_groups)
{
    using compute_t = std::conditional_t<
        std::is_same_v<act_t, float>, half,
        std::conditional_t<
            std::is_same_v<act_t, half>, half,
            std::conditional_t<
                std::is_same_v<act_t, __nv_bfloat16>, __nv_bfloat16,
                std::conditional_t<
                    std::is_same_v<act_t, __nv_fp8_e4m3>, __nv_fp8_e4m3,
                    half
                >
            >
        >
    >;
    
    using block_c_t = block_compact_t<block_q_t>;
    
    const auto* weights = reinterpret_cast<const block_c_t*>(vx);
    
    // Hierarchical grid decode computed internally
    tc32::dispatch_tc32_tc16_tcN<block_c_t, compute_t, act_t, output_t, HAS_REMAINDER>(
        weights, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size, row_groups);
}

