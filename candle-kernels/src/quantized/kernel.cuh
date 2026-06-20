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

// =============================================================================
// FORWARD DECLARATIONS - Optimized standalone dequant functions
// =============================================================================
// These are defined in their respective loader/*.cuh files but need forward
// declaration here since kernel.cuh is included before the specific loaders.
// The impl/*.cu files include both kernel.cuh and loader/*.cuh.
template <typename compute_t>
__device__ __forceinline__ void dequant_q8_0_for_4x_mma_k16_runtime(
    const uint8_t* __restrict__ smem_rows,
    int half_idx,
    int lane,
    uint32_t* frag_b
);

// =============================================================================
// GRID LAYOUT MODES
// =============================================================================
// Controls how blockIdx dimensions map to row blocks vs batch tiles.
// Choice affects L2 cache locality for weights (X) vs activations (Y).
// -----------------------------------------------------------------------------
constexpr int GRID_LAYOUT_ROW_FAST   = 0;  // blockIdx.x=row, blockIdx.y=batch
constexpr int GRID_LAYOUT_BATCH_FAST = 1;  // blockIdx.x=batch, blockIdx.y=row

// =============================================================================
// KERNEL CONFIGURATION CONSTANTS
// =============================================================================
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int KERNEL_NWARPS         = 1;   // AMD RDNA: 1 warp per block
    constexpr int KERNEL_ROWS_PER_BLOCK = 1;   // AMD RDNA: 1 row per block
#else
    constexpr int KERNEL_NWARPS         = 4;   // NVIDIA: 4 warps = 128 threads
    constexpr int KERNEL_ROWS_PER_BLOCK = 16;  // NVIDIA: 16 output rows per block
#endif

constexpr int KERNEL_ROWS_PER_PHASE = 16;  // Rows processed per reduction phase
constexpr int KERNEL_NUM_PHASES     = KERNEL_ROWS_PER_BLOCK / KERNEL_ROWS_PER_PHASE;
constexpr int KERNEL_NUM_THREADS    = KERNEL_NWARPS * WARP_SIZE;


// =============================================================================
// phase_reduce_to_smem - Warp reduction and store to shared memory
// =============================================================================
//
// Each thread holds partial sums in `tmp` for BATCH_TILE batches × ROWS_PER_PHASE rows.
// This function:
//   1. Performs warp-level reduction (shuffle-add across 32 lanes)
//   2. Lane 0 of each warp writes reduced values to shared memory
//
// Vectorized path (half/bf16): Packs pairs into half2/bf162 for 2× throughput.
// Scalar path (float/fp8): Standard reduction, one element at a time.
//
// PARAMETERS:
//   tmp        - Accumulator array passed BY VALUE (no address taken)
//   tmp_shared - Shared memory [BATCH_TILE][rows_per_cuda_block][nwarps]
//   phase      - Which phase (0..NUM_PHASES-1) determines smem row offset
// -----------------------------------------------------------------------------
template <typename acc_t, typename output_t, int BATCH_TILE, int ROWS_PER_PHASE,
          int rows_per_cuda_block, int nwarps>
__device__ __forceinline__ void phase_reduce_to_smem(
    RegArray<acc_t, BATCH_TILE, ROWS_PER_PHASE> tmp,
    acc_t (* __restrict__ tmp_shared)[rows_per_cuda_block][nwarps],
    const int phase)
{
    // === HALF PRECISION PATH (vectorized 2×) ===
    if constexpr (std::is_same_v<acc_t, __half>) {
        static_assert(ROWS_PER_PHASE % 2 == 0, "ROWS_PER_PHASE must be even");
        constexpr int NUM_PAIRS = ROWS_PER_PHASE / 2;
        
        #pragma unroll (64 / BATCH_TILE)
        for (int b = 0; b < BATCH_TILE; ++b) {
            __half2 reduced[NUM_PAIRS];
            
            // Pack pairs and reduce across warp
            #pragma unroll
            for (int i = 0; i < NUM_PAIRS; ++i) {
                __half2 pair = __halves2half2(tmp(b, i*2), tmp(b, i*2+1));
                reduced[i] = warp_reduce_sum_t<__half2>(pair);
            }
            
            // Lane 0 writes to smem
            if (threadIdx.x == 0) {
                #pragma unroll
                for (int i = 0; i < NUM_PAIRS; ++i) {
                    tmp_shared[b][phase * ROWS_PER_PHASE + i*2][threadIdx.y] = __low2half(reduced[i]);
                    tmp_shared[b][phase * ROWS_PER_PHASE + i*2+1][threadIdx.y] = __high2half(reduced[i]);
                }
            }
        }
    }
    // === BF16 PRECISION PATH (vectorized 2×) ===
    else if constexpr (std::is_same_v<acc_t, __nv_bfloat16>) {
        static_assert(ROWS_PER_PHASE % 2 == 0, "ROWS_PER_PHASE must be even");
        constexpr int NUM_PAIRS = ROWS_PER_PHASE / 2;
        
        #pragma unroll (64 / BATCH_TILE)
        for (int b = 0; b < BATCH_TILE; ++b) {
            __nv_bfloat162 reduced[NUM_PAIRS];
            
            // Pack pairs and reduce across warp
            #pragma unroll
            for (int i = 0; i < NUM_PAIRS; ++i) {
                __nv_bfloat162 pair = __halves2bfloat162(tmp(b, i*2), tmp(b, i*2+1));
                reduced[i] = warp_reduce_sum_t<__nv_bfloat162>(pair);
            }
            
            // Lane 0 writes to smem
            if (threadIdx.x == 0) {
                #pragma unroll
                for (int i = 0; i < NUM_PAIRS; ++i) {
                    tmp_shared[b][phase * ROWS_PER_PHASE + i*2][threadIdx.y] = __low2bfloat16(reduced[i]);
                    tmp_shared[b][phase * ROWS_PER_PHASE + i*2+1][threadIdx.y] = __high2bfloat16(reduced[i]);
                }
            }
        }
    }
    // === SCALAR PATH (float, fp8) ===
    else {
        #pragma unroll (64 / BATCH_TILE)
        for (int b = 0; b < BATCH_TILE; ++b) {
            acc_t reduced[ROWS_PER_PHASE];
            
            // Reduce each row across warp
            #pragma unroll
            for (int i = 0; i < ROWS_PER_PHASE; ++i) {
                reduced[i] = warp_reduce_sum_t<acc_t>(tmp(b, i));
            }
            
            // Lane 0 writes to smem
            if (threadIdx.x == 0) {
                #pragma unroll
                for (int i = 0; i < ROWS_PER_PHASE; ++i) {
                    tmp_shared[b][phase * ROWS_PER_PHASE + i][threadIdx.y] = reduced[i];
                }
            }
        }
    }
}


// =============================================================================
// final_output - Sum across warps and write final results
// =============================================================================
//
// After all phases complete, shared memory holds per-warp partial sums.
// This function sums across warps and writes to global memory.
//
// Thread mapping: Each thread handles one (batch, row) output element.
//   - tid % rows_per_cuda_block → row within block
//   - tid / rows_per_cuda_block → batch index within tile
//
// PARAMETERS:
//   tmp_shared  - Shared memory with per-warp sums
//   dst         - Global output buffer [batch_size, nrows_dst]
//   batch_start - First batch index for this tile
//   row0        - First row index for this block
//   nrows_dst   - Total output rows (for bounds check)
//   tid         - Thread ID within block (0..127)
// -----------------------------------------------------------------------------
template <typename acc_t, typename output_t, int BATCH_TILE, int rows_per_cuda_block, int nwarps>
__device__ __forceinline__ void final_output(
    acc_t (* __restrict__ tmp_shared)[rows_per_cuda_block][nwarps],
    output_t * __restrict__ dst,
    const int batch_start,
    const int row0,
    const int nrows_dst,
    const int tid)
{
    // Map thread to (batch, row) output element
    const int row_idx = tid % rows_per_cuda_block;
    const int b = (BATCH_TILE == 1) ? 0 : (tid / rows_per_cuda_block);
    
    // Sum contributions from all warps
    acc_t sum = tmp_shared[b][row_idx][0];
    #pragma unroll
    for (int w = 1; w < nwarps; ++w) {
        accumulate(sum, tmp_shared[b][row_idx][w]);
    }
    
    // Bounds check and type conversion
    const bool valid = (tid < BATCH_TILE * rows_per_cuda_block) && (row0 + row_idx < nrows_dst);
    if (valid) {
        // Convert accumulator to output type
        float result_f = acc_to_float(sum);
        output_t result;
        if constexpr (std::is_same_v<output_t, float>) {
            result = result_f;
        } else if constexpr (std::is_same_v<output_t, __half>) {
            result = __float2half(result_f);
        } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
            result = __float2bfloat16(result_f);
        } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
            result = from_f32<__nv_fp8_e4m3>(result_f);
        }
        
        // Write to global memory: dst[batch_start + b, row0 + row_idx]
        dst[(batch_start + b) * nrows_dst + row0 + row_idx] = result;
    }
}


// =============================================================================
// quantized_matmul_register_path - Main computation kernel
// =============================================================================
//
// Implements batched quantized matvec using register-only accumulation.
// Weights are loaded from global memory; only reduction uses shared memory.
//
// EXECUTION FLOW:
//   1. Initialize Y batch pointers (one per batch element)
//   2. For each phase (typically 1 phase = 16 rows):
//      a. Zero-initialize accumulator RegArray
//      b. Process all K tiles, accumulating dot products
//      c. Warp-reduce and store to smem
//   3. Sync threads
//   4. Sum across warps and write final output
//
// PASS-BY-VALUE PATTERN:
//   - RegArray<acc_t, BATCH_TILE, ROWS_PER_PHASE> passed to process_tile_*
//   - Modified copy returned: tmp = process_tile_register(..., tmp)
//   - Prevents nvcc from allocating stack for accumulator array
//
// PARAMETERS:
//   x               - Quantized weights [nrows_x, blocks_per_row_x] (column-major TC format)
//   vy              - Activations [batch_size, y_stride_per_row]
//   dst             - Output [batch_size, nrows_dst]
//   blocks_per_row_x- Number of quantized blocks per weight row
//   nrows_x         - Number of weight rows
//   nrows_dst       - Number of output rows (may differ due to padding)
//   y_stride_per_row- Stride between batch elements in Y
//   batch_start     - First batch index for this tile
// -----------------------------------------------------------------------------
template <int qk, int qi, typename block_q_t, int vdr, typename act_t,
          typename acc_t, typename output_t, int TILE_K_BLOCKS, int BATCH_TILE,
          int GRID_LAYOUT = GRID_LAYOUT_ROW_FAST>
__device__ void quantized_matmul_register_path(
    const block_q_t * __restrict__ x,
    const act_t * __restrict__ vy,
    output_t * __restrict__ dst,
    const int blocks_per_row_x,
    const int nrows_x,
    const int nrows_dst,
    const int y_stride_per_row,
    const int batch_start)
{
    // Kernel configuration constants
    constexpr int nwarps = KERNEL_NWARPS;
    constexpr int rows_per_cuda_block = KERNEL_ROWS_PER_BLOCK;
    constexpr int ROWS_PER_PHASE = KERNEL_ROWS_PER_PHASE;
    constexpr int NUM_PHASES = KERNEL_NUM_PHASES;
    
    // Grid layout determines which blockIdx dimension is rows vs batches
    constexpr bool batch_fast = (GRID_LAYOUT == GRID_LAYOUT_BATCH_FAST);
    const int row_block_idx = batch_fast ? blockIdx.y : blockIdx.x;
    
    // Thread and row indices
    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block * row_block_idx;
    
    // Shared memory for reduction: [batch][row][warp]
    __shared__ acc_t tmp_shared[BATCH_TILE][rows_per_cuda_block][nwarps];
    
    // Tile counts for K dimension
    const int num_full_tiles = blocks_per_row_x / TILE_K_BLOCKS;
    const int remainder_blocks = blocks_per_row_x % TILE_K_BLOCKS;
    
    // Weight pointer (compacted blocks with embedded scales)
    using block_c_t = block_compact_t<block_q_t>;
    const block_c_t * x_c = reinterpret_cast<const block_c_t *>(x);
    
    // Y activation setup
    constexpr bool is_q8_y = std::is_same_v<act_t, block_q8_1>;
    constexpr int y_tile_stride = is_q8_y ? (TILE_K_BLOCKS * (qk / QK8_1)) : (TILE_K_BLOCKS * qk);
    constexpr int y_stride_per_block = is_q8_y ? (qk / QK8_1) : qk;
    
    // Batch Y offset: first batch starts at batch_start * y_stride_per_row
    const int batch_y_offset = batch_start * y_stride_per_row;
    
    // === PHASE LOOP: Process ROWS_PER_PHASE rows at a time ===
    #pragma unroll 1
    for (int phase = 0; phase < NUM_PHASES; ++phase) {
        const int phase_row0 = row0 + phase * ROWS_PER_PHASE;
        
        // Zero-initialize accumulator (pass-by-value wrapper)
        RegArray<acc_t, BATCH_TILE, ROWS_PER_PHASE> tmp = {};
        #pragma unroll (64 / BATCH_TILE)
        for (int b = 0; b < BATCH_TILE; ++b) {
            #pragma unroll
            for (int i = 0; i < ROWS_PER_PHASE; ++i) {
                tmp(b, i) = acc_zero<acc_t>();
            }
        }
        
        // === FULL TILE LOOP: Fixed-size tiles for compile-time unrolling ===
        #pragma unroll 1
        for (int tile = 0; tile < num_full_tiles; ++tile) {
            const int tile_start = tile * TILE_K_BLOCKS;
            const int tile_y_offset = tile * y_tile_stride;
            
            // Process tile and get updated accumulator (pass-by-value)
            tmp = process_tile_register<qk, qi, block_q_t, vdr, act_t,
                                acc_t, nwarps, TILE_K_BLOCKS, BATCH_TILE, ROWS_PER_PHASE>(
                x_c, vy, y_stride_per_row, batch_y_offset, tile_y_offset,
                tmp, tile_start, phase_row0, nrows_x, tid);
        }
        
        // === REMAINDER TILE: Handle leftover blocks ===
        if (remainder_blocks > 0) {
            const int tile_start = num_full_tiles * TILE_K_BLOCKS;
            const int tile_y_offset = num_full_tiles * y_tile_stride;
            tmp = process_tile_register_partial<qk, qi, block_q_t, vdr, act_t,
                                acc_t, nwarps, BATCH_TILE, ROWS_PER_PHASE>(
                x_c, vy, y_stride_per_row, batch_y_offset, tile_y_offset,
                tmp, tile_start, remainder_blocks, phase_row0, nrows_x, tid);
        }
        
        // Warp-reduce and store to shared memory
        phase_reduce_to_smem<acc_t, output_t, BATCH_TILE, ROWS_PER_PHASE, rows_per_cuda_block, nwarps>(
            tmp, tmp_shared, phase);
    }
    
    // Synchronize before reading smem for final reduction
    __syncthreads();
    
    // Sum across warps and write final output
    final_output<acc_t, output_t, BATCH_TILE, rows_per_cuda_block, nwarps>(
        tmp_shared, dst, batch_start, row0, nrows_dst, tid);
}


// =============================================================================
// quantized_matmul - Top-level kernel entry point
// =============================================================================
//
// Called from dispatcher with specific BATCH_TILE instantiation (_s1 through _s8).
// Performs setup and dispatches to quantized_matmul_register_path.
//
// GREEDY DECOMPOSITION:
//   The dispatcher breaks batch_size into exact multiples of BATCH_TILE.
//   Example: batch_size=13 → _s8(8) + _s4(4) + _s1(1)
//   This kernel always processes exactly BATCH_TILE batches (no partial tiles).
//
// TEMPLATE PARAMETERS:
//   qk         - Quantization block size (e.g., 128 for K/128 format)
//   qi         - Number of int32s per block for index calculation
//   block_q_t  - Quantized block type (block_q4_0, block_q4_K, etc.)
//   vdr        - Vector dot ratio (elements per thread per iteration)
//   act_t      - Activation type (float, __half, __nv_bfloat16, block_q8_1)
//   output_t   - Output type (float, __half, __nv_bfloat16, __nv_fp8_e4m3)
//   BATCH_TILE - Number of batches processed per block (1-8)
//   GRID_LAYOUT- Row-fast (0) or batch-fast (1) grid mapping
// -----------------------------------------------------------------------------
template <int qk, int qi, typename block_q_t, int vdr,
          typename act_t, typename output_t = float, int BATCH_TILE = 8,
          int GRID_LAYOUT = GRID_LAYOUT_ROW_FAST>
static __device__ __noinline__ void quantized_matmul(
    const void * __restrict__ vx,
    const act_t * __restrict__ vy,
    output_t * __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int nrows_y,
    const int nrows_dst, 
    const int batch_size)
{
    // Compute which batch tile this block processes
    constexpr bool batch_fast = (GRID_LAYOUT == GRID_LAYOUT_BATCH_FAST);
    const int batch_tile_idx = batch_fast ? blockIdx.x : blockIdx.y;
    const int batch_start = batch_tile_idx * BATCH_TILE;
    
    // Bounds check (defensive - dispatcher should never create OOB blocks)
    if (batch_start >= batch_size) return;
    
    // Verify greedy decomposition invariant
    assert(batch_start + BATCH_TILE <= batch_size && 
           "Greedy decomposition violated: incomplete batch tile");
    
    // =========================================================================
    // CUDA CORE PATH (register-only)
    // =========================================================================
    // TC path for batch >= 32 uses quantized_matmul_tc instead.
    
    // Derive accumulator type from output type
    typedef typename accumulator_type<output_t>::type acc_t;
    
    // Tile size for K dimension
    // tile_k_blocks_v computes blocks in GGML units (output_count elements per block)
    // The kernel uses qk elements per block, so scale appropriately
    constexpr int GGML_OUTPUT_COUNT = dequant_sizes<block_q_t>::output_count;
    constexpr int TILE_K_BLOCKS_GGML = tile_k_blocks_v<block_q_t, output_t, false>;
    // Scale to K128 block units when qk < GGML_OUTPUT_COUNT (K/128 format)
    // For qk=128, GGML_OUTPUT_COUNT=256: ratio = 2, TILE_K_BLOCKS *= 2
    // For qk=256 (GGML format): ratio = 1, no scaling
    constexpr int TILE_K_BLOCKS = TILE_K_BLOCKS_GGML * (GGML_OUTPUT_COUNT / qk);
    
    // Weight and activation layout
    const int blocks_per_row_x = ncols_x / qk;
    constexpr bool is_q8_y = std::is_same_v<act_t, block_q8_1>;
    const int y_stride_per_row = is_q8_y ? (nrows_y / QK8_1) : nrows_y;
    
    const block_q_t * x = (const block_q_t *) vx;
    
    // Dispatch to register path (scales embedded in blocks)
    quantized_matmul_register_path<qk, qi, block_q_t, vdr, act_t, acc_t, output_t, 
                                   TILE_K_BLOCKS, BATCH_TILE, GRID_LAYOUT>(
        x, vy, dst, blocks_per_row_x, nrows_x, nrows_dst, y_stride_per_row, batch_start);
}


// =============================================================================
// quantized_matmul_iter - BATCH_TILE=2 iteration kernel for large batches
// =============================================================================
//
// DESIGN PHILOSOPHY:
// For batch > 8, use BATCH_TILE=2 with NUM_ITERS iterations:
//   - High occupancy (~100%) due to low register usage (~60 regs)
//   - Many warps in flight = good latency hiding
//   - s2_iter4 processes 8 batches total (4 iterations of 2)
//
// REGISTER USAGE:
//   - Uses ~60 registers (BATCH_TILE=2)
//   - High occupancy allows many concurrent warps
//   - Good for latency-bound workloads
//
// DISPATCH:
//   - batch > 8: loop of s2_iter4(8)
//   - batch 1-8: use s1-s8 directly
// -----------------------------------------------------------------------------
template <int qk, int qi, typename block_q_t, int vdr,
          typename act_t, typename output_t = float,
          int BATCH_TILE = 2, int NUM_ITERS = 4>
static __device__ void quantized_matmul_iter(
    const void * __restrict__ vx,
    const act_t * __restrict__ vy,
    output_t * __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int nrows_y,
    const int nrows_dst, 
    const int total_batches)
{
    // Total batches processed per block = BATCH_TILE * NUM_ITERS (e.g., 8 for s2_iter4)
    constexpr int BATCHES_PER_BLOCK = BATCH_TILE * NUM_ITERS;
    
    // Kernel configuration constants
    constexpr int nwarps = KERNEL_NWARPS;                       // 4 warps = 128 threads
    constexpr int rows_per_cuda_block = KERNEL_ROWS_PER_BLOCK;  // 16 rows
    constexpr int ROWS_PER_PHASE = KERNEL_ROWS_PER_PHASE;       // 16 rows per phase
    constexpr int NUM_PHASES = KERNEL_NUM_PHASES;               // 1 phase
    
    // Grid layout: blockIdx.x = row blocks, blockIdx.y = batch tiles
    const int row_block_idx = blockIdx.x;
    const int batch_tile_idx = blockIdx.y;
    
    // Base batch index for this block
    const int batch_base = batch_tile_idx * BATCHES_PER_BLOCK;
    
    // Bounds check: skip if this batch tile is out of range
    if (batch_base >= total_batches) return;
    
    // Thread and row indices
    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block * row_block_idx;
    
    // Derive accumulator type from output type
    typedef typename accumulator_type<output_t>::type acc_t;
    
    // Tile size for K dimension
    constexpr int GGML_OUTPUT_COUNT = dequant_sizes<block_q_t>::output_count;
    constexpr int TILE_K_BLOCKS_GGML = tile_k_blocks_v<block_q_t, output_t, false>;
    constexpr int TILE_K_BLOCKS = TILE_K_BLOCKS_GGML * (GGML_OUTPUT_COUNT / qk);
    
    // Weight and activation layout
    const int blocks_per_row_x = ncols_x / qk;
    constexpr bool is_q8_y = std::is_same_v<act_t, block_q8_1>;
    const int y_stride_per_row = is_q8_y ? (nrows_y / QK8_1) : nrows_y;
    
    const block_q_t * x = (const block_q_t *) vx;
    
    // Tile counts for K dimension
    const int num_full_tiles = blocks_per_row_x / TILE_K_BLOCKS;
    const int remainder_blocks = blocks_per_row_x % TILE_K_BLOCKS;
    
    // Weight pointer (compacted blocks with embedded scales)
    using block_c_t = block_compact_t<block_q_t>;
    const block_c_t * x_c = reinterpret_cast<const block_c_t *>(x);
    
    // Shared memory for reduction: [NUM_ITERS iterations][BATCH_TILE batches][16 rows][4 warps]
    // Separate buffer per iteration eliminates inter-iteration __syncthreads()
    __shared__ acc_t tmp_shared[NUM_ITERS][BATCH_TILE][rows_per_cuda_block][nwarps];
    
    // Y activation setup - all offsets computed from base pointer + strides
    constexpr int y_tile_stride = is_q8_y ? (TILE_K_BLOCKS * (qk / QK8_1)) : (TILE_K_BLOCKS * qk);
    
    // Stride to advance Y by BATCH_TILE batches between iterations
    const int iter_y_stride = BATCH_TILE * y_stride_per_row;
    
    // Base Y offset for this batch tile (batch_base * y_stride_per_row)
    const int base_y_offset = batch_base * y_stride_per_row;
    
    // =========================================================================
    // PHASE 1: Process all iterations, write to per-iteration smem buffers
    // iter is compile-time constant after #pragma unroll, so all offsets fold
    // =========================================================================
    #pragma unroll
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        // Batch Y offset for this iteration: base + iter * BATCH_TILE * y_stride_per_row
        const int batch_y_offset = base_y_offset + iter * iter_y_stride;
        
        // =====================================================================
        // PHASE LOOP: Process ROWS_PER_PHASE rows at a time
        // =====================================================================
        #pragma unroll 1
        for (int phase = 0; phase < NUM_PHASES; ++phase) {
            const int phase_row0 = row0 + phase * ROWS_PER_PHASE;
            
            // Zero-initialize accumulator (pass-by-value wrapper)
            RegArray<acc_t, BATCH_TILE, ROWS_PER_PHASE> tmp = {};
            #pragma unroll
            for (int b = 0; b < BATCH_TILE; ++b) {
                #pragma unroll
                for (int i = 0; i < ROWS_PER_PHASE; ++i) {
                    tmp(b, i) = acc_zero<acc_t>();
                }
            }
            
            // =================================================================
            // TILE LOOP: Process K tiles
            // =================================================================
            #pragma unroll 1
            for (int tile = 0; tile < num_full_tiles; ++tile) {
                const int tile_start = tile * TILE_K_BLOCKS;
                const int tile_y_offset = tile * y_tile_stride;
                
                tmp = process_tile_register<qk, qi, block_q_t, vdr, act_t,
                                    acc_t, nwarps, TILE_K_BLOCKS, BATCH_TILE, ROWS_PER_PHASE>(
                    x_c, vy, y_stride_per_row, batch_y_offset, tile_y_offset,
                    tmp, tile_start, phase_row0, nrows_x, tid);
            }
            
            // Handle remainder blocks
            if (remainder_blocks > 0) {
                const int tile_start = num_full_tiles * TILE_K_BLOCKS;
                const int tile_y_offset = num_full_tiles * y_tile_stride;
                tmp = process_tile_register_partial<qk, qi, block_q_t, vdr, act_t,
                                    acc_t, nwarps, BATCH_TILE, ROWS_PER_PHASE>(
                    x_c, vy, y_stride_per_row, batch_y_offset, tile_y_offset,
                    tmp, tile_start, remainder_blocks, phase_row0, nrows_x, tid);
            }
            
            // Warp-reduce and store to shared memory (iteration-specific buffer)
            phase_reduce_to_smem<acc_t, output_t, BATCH_TILE, ROWS_PER_PHASE, rows_per_cuda_block, nwarps>(
                tmp, tmp_shared[iter], phase);
        }
    }
    
    // =========================================================================
    // SINGLE SYNC: Wait for all iterations to finish smem writes
    // =========================================================================
    __syncthreads();
    
    // =========================================================================
    // PHASE 2: Finalize all iterations - read smem and write to global memory
    // =========================================================================
    #pragma unroll
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        // batch_start is global batch index: batch_base + iter * BATCH_TILE
        const int batch_start = batch_base + iter * BATCH_TILE;
        final_output<acc_t, output_t, BATCH_TILE, rows_per_cuda_block, nwarps>(
            tmp_shared[iter], dst, batch_start, row0, nrows_dst, tid);
    }
}


// =============================================================================
// final_output_tc - Final output for TC path with partial batch support
// =============================================================================
//
// Same as final_output but handles partial batch tiles (last tile may have
// fewer than BATCH_TILE batches).
// -----------------------------------------------------------------------------
template <typename acc_t, typename output_t, int BATCH_TILE, int rows_per_cuda_block, int nwarps>
__device__ __forceinline__ void final_output_tc(
    acc_t (* __restrict__ tmp_shared)[rows_per_cuda_block][nwarps],
    output_t * __restrict__ dst,
    const int batch_start,
    const int row0,
    const int nrows_dst,
    const int batches_this_tile,
    const int tid)
{
    // Map thread to (batch, row) output element
    const int row_idx = tid % rows_per_cuda_block;
    const int b = tid / rows_per_cuda_block;
    
    // Sum contributions from all warps
    acc_t sum = tmp_shared[b][row_idx][0];
    #pragma unroll
    for (int w = 1; w < nwarps; ++w) {
        accumulate(sum, tmp_shared[b][row_idx][w]);
    }
    
    // Bounds check: valid batch AND valid row AND within actual batches for this tile
    const bool valid = (b < batches_this_tile) && 
                       (tid < BATCH_TILE * rows_per_cuda_block) && 
                       (row0 + row_idx < nrows_dst);
    if (valid) {
        // Convert accumulator to output type
        float result_f = acc_to_float(sum);
        output_t result;
        if constexpr (std::is_same_v<output_t, float>) {
            result = result_f;
        } else if constexpr (std::is_same_v<output_t, __half>) {
            result = __float2half(result_f);
        } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
            result = __float2bfloat16(result_f);
        } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
            result = from_f32<__nv_fp8_e4m3>(result_f);
        }
        
        // Write to global memory: dst[batch_start + b, row0 + row_idx]
        dst[(batch_start + b) * nrows_dst + row0 + row_idx] = result;
    }
}


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

// Configuration constants shared by s16_tc and sN_tc
constexpr int BATCH_TILE = 16;      // 16 batches per MMA (M dimension) — FP16 path
// INT8 path token tile. N_SUBTILES_I8 = BATCH_TILE_I8/16 m16 token sub-tiles run per
// weight load, the weight k-block staged ONCE in smem and reused across them. At 16
// (single sub-tile) this is the plain per-tile schedule; at 32 it trades doubled
// activation smem for halved weight-L2 re-reads — a net win only when the weight bytes
// dominate (measured: Q8 only; light formats regress on the occupancy hit). Only the int8
// impl/entries use this constant directly; the dense grid (dispatcher.cu) and the MoE
// tiler (cuda.rs grouped_matmul_gemx_q8a128) carry the matching token count as literals —
// keep all three in sync. FP16 stays at BATCH_TILE = 16.
constexpr int BATCH_TILE_I8 = 16;
constexpr int N_SUBTILES_I8 = BATCH_TILE_I8 / 16;  // m16 token sub-tiles per block
constexpr int N_TILE = 32;          // 32 rows per block (4 warps × 8 rows)
constexpr int K_TILE = 128;         // K/128 block size
// INT8 activation smem row stride: K_TILE + 16B pad. The pad shifts each row's bank
// by 4 (stride 144B → (144/4)%32 = 4) so the 8 rows ldmatrix reads per tile land in
// distinct banks — conflict-free, the int8 analogue of the FP path's K_STRIDE pad.
constexpr int KI8_STRIDE = K_TILE + 16;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int MMA_ITERS = K_TILE / MMA_K;  // 8 iterations per K/128
constexpr int HALF_MMA_ITERS = MMA_ITERS / 2;  // 4 iterations (for 2x k16 batching)
constexpr int NUM_THREADS = 128;
constexpr int WARP_SIZE_TC = 32;
constexpr int NUM_WARPS = 4;
constexpr int K_PAD = 8;
constexpr int K_STRIDE = K_TILE + K_PAD;

// =============================================================================
// LDMATRIX CONFIGURATION
// =============================================================================
// ldmatrix.sync.aligned.m8n8.x4 is available on sm_75+ (Turing and later)
// It loads 4 × 8×8 matrices (128 bytes total) in one instruction, matching
// the fragment layout expected by mma.m16n8k16.
//
// Benefits:
//   - Single instruction loads all 4 fragments (vs 4 separate loads)
//   - Hardware-optimized shared memory access pattern
//   - Better latency hiding
//
// Requirements:
//   - Shared memory must be 16-byte aligned
//   - Each lane provides address for its portion of the 8×8 tile
//   - Data must be contiguous in 16-byte chunks
// =============================================================================

// Feature detection for ldmatrix — SM80 floor guarantees SM75+ (Turing+)
#define TC_USE_LDMATRIX 1

// -----------------------------------------------------------------------------
// Load FragA using ldmatrix (sm_75+)
// ldmatrix.sync.aligned.m8n8.x4 loads four 8×8 matrices from shared memory
// directly into the register layout expected by mma.m16n8k16.
//
// For mma.m16n8k16 with row-major A[16,16]:
//   - ldmatrix loads 4 × 8×8 tiles arranged as:
//     [0][2]    where each tile is 8×8
//     [1][3]
//   - Each lane in the warp provides one address
//   - The hardware collects 8 addresses per 8×8 tile
// -----------------------------------------------------------------------------
template <typename compute_t>
__device__ __forceinline__ void load_frag_a_ldmatrix(
    uint32_t frag_a[4],
    const compute_t smem_A[BATCH_TILE][K_STRIDE],
    int k_start, int lane)
{
    // Use ldmatrix for 16-bit types (half and bfloat16) — SM80+ always has ldmatrix
    if constexpr (std::is_same_v<compute_t, half> || std::is_same_v<compute_t, __nv_bfloat16>) {
        // ldmatrix addressing for m8n8.x4:
        // - Lanes 0-7: provide addresses for tile 0 (rows 0-7, K 0-7)
        // - Lanes 8-15: provide addresses for tile 1 (rows 8-15, K 0-7)
        // - Lanes 16-23: provide addresses for tile 2 (rows 0-7, K 8-15)
        // - Lanes 24-31: provide addresses for tile 3 (rows 8-15, K 8-15)
        //
        // Each lane provides address to one row of its 8×8 tile
        const int tile_idx = lane / 8;      // Which of the 4 tiles (0-3)
        const int row_in_tile = lane % 8;   // Row within the 8×8 tile (0-7)
        
        // Tile layout for m16n8k16:
        //   Tile 0: rows 0-7,  K k_start..k_start+7
        //   Tile 1: rows 8-15, K k_start..k_start+7
        //   Tile 2: rows 0-7,  K k_start+8..k_start+15
        //   Tile 3: rows 8-15, K k_start+8..k_start+15
        const int m_offset = (tile_idx & 1) * 8;  // 0 for tiles 0,2; 8 for tiles 1,3
        const int k_offset = (tile_idx >> 1) * 8; // 0 for tiles 0,1; 8 for tiles 2,3
        
        // Address of this lane's row in shared memory
        // Each row has 8 half values (16 bytes) contiguous in K dimension
        const uint32_t addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem_A[m_offset + row_in_tile][k_start + k_offset]));
        
        // ldmatrix.sync.aligned.m8n8.x4 loads 4 matrices, output to frag_a[0-3]
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(frag_a[0]), "=r"(frag_a[1]), "=r"(frag_a[2]), "=r"(frag_a[3])
            : "r"(addr)
        );
        return;
    }
    // Fallback for non-ldmatrix path or non-half types
    // (compile-time eliminated when ldmatrix path is taken)
    const int groupID = lane / 4;
    const int threadID_in_group = lane % 4;
    const int k_col = k_start + threadID_in_group * 2;
    
    if constexpr (std::is_same_v<compute_t, half>) {
        half2* f = reinterpret_cast<half2*>(frag_a);
        f[0] = *reinterpret_cast<const half2*>(&smem_A[groupID][k_col]);
        f[1] = *reinterpret_cast<const half2*>(&smem_A[groupID + 8][k_col]);
        f[2] = *reinterpret_cast<const half2*>(&smem_A[groupID][k_col + 8]);
        f[3] = *reinterpret_cast<const half2*>(&smem_A[groupID + 8][k_col + 8]);
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        __nv_bfloat162* f = reinterpret_cast<__nv_bfloat162*>(frag_a);
        f[0] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID][k_col]);
        f[1] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID + 8][k_col]);
        f[2] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID][k_col + 8]);
        f[3] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID + 8][k_col + 8]);
    } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
        half2* f = reinterpret_cast<half2*>(frag_a);
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        __nv_fp8x2_storage_t fp8_01 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID][k_col]);
        __nv_fp8x2_storage_t fp8_23 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID + 8][k_col]);
        __nv_fp8x2_storage_t fp8_45 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID][k_col + 8]);
        __nv_fp8x2_storage_t fp8_67 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID + 8][k_col + 8]);
        
        __half2_raw h01 = __nv_cvt_fp8x2_to_halfraw2(fp8_01, __NV_E4M3);
        __half2_raw h23 = __nv_cvt_fp8x2_to_halfraw2(fp8_23, __NV_E4M3);
        __half2_raw h45 = __nv_cvt_fp8x2_to_halfraw2(fp8_45, __NV_E4M3);
        __half2_raw h67 = __nv_cvt_fp8x2_to_halfraw2(fp8_67, __NV_E4M3);
        
        f[0] = *reinterpret_cast<half2*>(&h01);
        f[1] = *reinterpret_cast<half2*>(&h23);
        f[2] = *reinterpret_cast<half2*>(&h45);
        f[3] = *reinterpret_cast<half2*>(&h67);
        #else
        f[0] = __float22half2_rn(make_float2(float(smem_A[groupID][k_col]), float(smem_A[groupID][k_col + 1])));
        f[1] = __float22half2_rn(make_float2(float(smem_A[groupID + 8][k_col]), float(smem_A[groupID + 8][k_col + 1])));
        f[2] = __float22half2_rn(make_float2(float(smem_A[groupID][k_col + 8]), float(smem_A[groupID][k_col + 8 + 1])));
        f[3] = __float22half2_rn(make_float2(float(smem_A[groupID + 8][k_col + 8]), float(smem_A[groupID + 8][k_col + 8 + 1])));
        #endif
    }
}

// -----------------------------------------------------------------------------
// Load FragA: shared memory → registers for MMA (original non-ldmatrix version)
// For mma.m16n8k16 row-major A with f16/bf16, PTX specifies:
//   groupID           = lane / 4  (range 0-7)
//   threadID_in_group = lane % 4  (range 0-3)
//
//   frag[0] (a0,a1): M=groupID,   K=threadID_in_group*2+{0,1}
//   frag[1] (a2,a3): M=groupID+8, K=threadID_in_group*2+{0,1}
//   frag[2] (a4,a5): M=groupID,   K=threadID_in_group*2+8+{0,1}
//   frag[3] (a6,a7): M=groupID+8, K=threadID_in_group*2+8+{0,1}
// -----------------------------------------------------------------------------
template <typename compute_t>
__device__ __forceinline__ void load_frag_a(
    uint32_t frag_a[4],
    const compute_t smem_A[BATCH_TILE][K_STRIDE],
    int k_start, int lane)
{
    // Use ldmatrix for 16-bit types (half and bfloat16) — always available on SM80+
    if constexpr (std::is_same_v<compute_t, half> || std::is_same_v<compute_t, __nv_bfloat16>) {
        load_frag_a_ldmatrix(frag_a, smem_A, k_start, lane);
        return;
    }
    
    const int groupID = lane / 4;           // 0-7
    const int threadID_in_group = lane % 4; // 0-3
    const int k_col = k_start + threadID_in_group * 2;
    
    if constexpr (std::is_same_v<compute_t, half>) {
        half2* f = reinterpret_cast<half2*>(frag_a);
        f[0] = *reinterpret_cast<const half2*>(&smem_A[groupID][k_col]);
        f[1] = *reinterpret_cast<const half2*>(&smem_A[groupID + 8][k_col]);
        f[2] = *reinterpret_cast<const half2*>(&smem_A[groupID][k_col + 8]);
        f[3] = *reinterpret_cast<const half2*>(&smem_A[groupID + 8][k_col + 8]);
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        __nv_bfloat162* f = reinterpret_cast<__nv_bfloat162*>(frag_a);
        f[0] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID][k_col]);
        f[1] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID + 8][k_col]);
        f[2] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID][k_col + 8]);
        f[3] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[groupID + 8][k_col + 8]);
    } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
        half2* f = reinterpret_cast<half2*>(frag_a);
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        __nv_fp8x2_storage_t fp8_01 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID][k_col]);
        __nv_fp8x2_storage_t fp8_23 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID + 8][k_col]);
        __nv_fp8x2_storage_t fp8_45 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID][k_col + 8]);
        __nv_fp8x2_storage_t fp8_67 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[groupID + 8][k_col + 8]);
        
        __half2_raw h01 = __nv_cvt_fp8x2_to_halfraw2(fp8_01, __NV_E4M3);
        __half2_raw h23 = __nv_cvt_fp8x2_to_halfraw2(fp8_23, __NV_E4M3);
        __half2_raw h45 = __nv_cvt_fp8x2_to_halfraw2(fp8_45, __NV_E4M3);
        __half2_raw h67 = __nv_cvt_fp8x2_to_halfraw2(fp8_67, __NV_E4M3);
        
        f[0] = *reinterpret_cast<half2*>(&h01);
        f[1] = *reinterpret_cast<half2*>(&h23);
        f[2] = *reinterpret_cast<half2*>(&h45);
        f[3] = *reinterpret_cast<half2*>(&h67);
        #else
        f[0] = __float22half2_rn(make_float2(float(smem_A[groupID][k_col]), float(smem_A[groupID][k_col + 1])));
        f[1] = __float22half2_rn(make_float2(float(smem_A[groupID + 8][k_col]), float(smem_A[groupID + 8][k_col + 1])));
        f[2] = __float22half2_rn(make_float2(float(smem_A[groupID][k_col + 8]), float(smem_A[groupID][k_col + 8 + 1])));
        f[3] = __float22half2_rn(make_float2(float(smem_A[groupID + 8][k_col + 8]), float(smem_A[groupID + 8][k_col + 8 + 1])));
        #endif
    }
}

// -----------------------------------------------------------------------------
// Load weights: global → shared memory using cp.async (all 32 rows per block)
// Weight layout is K-major: weights[k_blk * nrows + row]
// Uses 16-byte async copies for better memory bandwidth utilization
// All 128 threads cooperatively load all 32 rows for better coalescing
// -----------------------------------------------------------------------------
template <typename block_c_t>
__device__ __forceinline__ void load_weights_async_coop(
    uint8_t* smem_W_flat,  // Flat buffer: 32 rows × block_bytes bytes
    const block_c_t* __restrict__ weights,
    int row0, int k_blk, int nrows, int k_blocks, int tid)
{
    constexpr int block_bytes = sizeof(block_c_t);
    constexpr int int4s_per_block = block_bytes / 16;  // 16-byte chunks per block
    constexpr int total_int4s = N_TILE * int4s_per_block;  // 32 rows × int4s_per_block
    constexpr int loads_per_thread = (total_int4s + NUM_THREADS - 1) / NUM_THREADS;
    
    const block_c_t* row_base = &weights[k_blk * nrows + row0];
    
    #pragma unroll
    for (int i = 0; i < loads_per_thread; ++i) {
        const int int4_idx = tid + i * NUM_THREADS;
        
        if (int4_idx < total_int4s) {
            const int row_local = int4_idx / int4s_per_block;
            const int chunk_in_row = int4_idx % int4s_per_block;
            
            uint8_t* dst = smem_W_flat + row_local * block_bytes + chunk_in_row * 16;
            const uint8_t* src = reinterpret_cast<const uint8_t*>(&row_base[row_local]) + chunk_in_row * 16;
            
            // ...existing code...
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                   "l"(src)
            );
        }
    }
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// 16-byte cg cp.async (global → shared). The q8a128 qs is 16-aligned within its
// block, so each thread's chunk is a single wide async copy.
__device__ __forceinline__ void cp_async_cg16(void* sdst, const void* gsrc) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sdst))), "l"(gsrc));
}

// 16-byte ca cp.async (global → shared) — caches the line in L1 as well as L2.
// Used for activations: each expert-token-tile's activation is re-read by every
// row-tile block (grid.y), so keeping it L1-resident turns those re-reads into L1
// hits instead of L2 traffic. (The .cg variant bypasses L1 and made the int8 path
// L2-bandwidth-bound while the LDG-based FP path stayed DRAM-bound.) Weights keep
// .cg: they are streamed with little intra-SM reuse and would only thrash L1.
__device__ __forceinline__ void cp_async_ca16(void* sdst, const void* gsrc) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sdst))), "l"(gsrc));
}

// 4-byte ca cp.async (global → shared) — for the per-token activation (scale,sum) half2.
__device__ __forceinline__ void cp_async_ca4(void* sdst, const void* gsrc) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sdst))), "l"(gsrc));
}

// Legacy sync load for fallback (pre-Ampere)
template <typename block_c_t>
__device__ __forceinline__ void load_weights_warp(
    uint8_t* smem_W,  // 8 × block_bytes bytes per warp
    const block_c_t* __restrict__ weights,
    int row_base, int k_blk, int nrows, int k_blocks, int lane)
{
    constexpr int block_bytes = sizeof(block_c_t);
    constexpr int ints_per_block = block_bytes / sizeof(int);
    constexpr int total_ints = 8 * ints_per_block;
    constexpr int loads_per_thread = (total_ints + WARP_SIZE_TC - 1) / WARP_SIZE_TC;
    
    int* dst = reinterpret_cast<int*>(smem_W);
    
    #pragma unroll
    for (int i = 0; i < loads_per_thread; ++i) {
        const int int_idx = lane + i * WARP_SIZE_TC;
        
        if (int_idx < total_ints) {
            const int row_local = int_idx / ints_per_block;
            const int word_in_row = int_idx % ints_per_block;
            const int global_row = row_base + row_local;
            
            int val = 0;
            if (global_row < nrows && row_local < 8) {
                const block_c_t* blk = &weights[k_blk * nrows + global_row];
                val = blk->data[word_in_row];
            }
            dst[int_idx] = val;
        }
    }
}

// -----------------------------------------------------------------------------
// MMA instruction wrapper for m16n8k16
// -----------------------------------------------------------------------------
template <typename compute_t>
__device__ __forceinline__ void mma_m16n8k16(
    const uint32_t* a, const uint32_t* b, float* c)
{
    if constexpr (std::is_same_v<compute_t, half>) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
    } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
        // FP8 uses f16 MMA for k16 - native FP8 MMA is k32 only
        // Activations and weights should be in f16 format for this path
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
    }
}

// -----------------------------------------------------------------------------
// MMA instruction wrapper for m16n8k32
// -----------------------------------------------------------------------------
// Processes 32 K elements at once.
//
// For FP8 on SM89+: Uses native mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
//   - a[4]: 4 × uint32, each holding 4 packed fp8 = 16 fp8 total
//   - b[2]: 2 × uint32, each holding 4 packed fp8 = 8 fp8 total
//   Note: a_lo[0..1] and a_hi[0..1] are used as the 4-register A operand
//         b[0..1] are used as the 2-register B operand
//
// For F16/BF16 (and FP8 pre-SM89 fallback): Uses two consecutive m16n8k16 MMAs
//   - a_lo[4]: First k16 slice (4 × half2 = 8 f16)
//   - a_hi[4]: Second k16 slice (4 × half2 = 8 f16)
//   - b[4]: b[0..1] for first k16, b[2..3] for second k16
//
// Parameters:
//   a_lo[4], a_hi[4]: A operands (interpretation depends on compute_t and arch)
//   b[4]:             B operand
//   c[4]:             Accumulator (4 × float, accumulated in place)
// -----------------------------------------------------------------------------
template <typename compute_t>
__device__ __forceinline__ void mma_m16n8k32(
    const uint32_t* a_lo, const uint32_t* a_hi,
    const uint32_t* b, float* c)
{
    if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        // Native FP8 m16n8k32 on SM89+ (Ada/Hopper)
        // A operand: 4 registers with 4 fp8 each = 16 fp8
        // B operand: 2 registers with 4 fp8 each = 8 fp8
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_hi[0]), "r"(a_hi[1]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
#else
        // Pre-SM89 fallback: use 2× f16 k16 MMAs (assumes f16 format input)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]),
              "r"(b[2]), "r"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
#endif
    } else if constexpr (std::is_same_v<compute_t, half>) {
        // First k16
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
        // Second k16
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]),
              "r"(b[2]), "r"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        // First k16
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
        // Second k16
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]),
              "r"(b[2]), "r"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
    }
}

// -----------------------------------------------------------------------------
// MMA instruction wrapper for m16n8k64 (INT4 / sub-byte types)
// -----------------------------------------------------------------------------
// Processes 64 K elements at once.
//
// For INT4 on SM80+: Uses native mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
//   - a[4]: 4 × uint32, each holding 8 packed int4 = 32 int4 per thread
//   - b[2]: 2 × uint32, each holding 8 packed int4 = 16 int4 per thread
//   - c[4]: 4 × int32 accumulator
//
// For other types: Falls back to 4× m16n8k16 (f16) or 2× m16n8k32 (fp8)
//   - Requires input data converted to appropriate format
//
// Parameters:
//   a[4]: A operand (4 registers)
//   b[4]: B operand (4 registers for fallback, only b[0..1] used for native int4)
//   c[4]: Accumulator (4 × int32 for int4, 4 × float for fallback)
// -----------------------------------------------------------------------------
template <typename compute_t>
__device__ __forceinline__ void mma_m16n8k64(
    const uint32_t* a,
    const uint32_t* b,
    int32_t* c)
{
    // Native INT4 m16n8k64 MMA — always available on SM80+
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

// Overload with float accumulator for when we need to fall back to f16/fp8 paths
// This handles cases where int4 weights are dequantized to floating point
template <typename compute_t>
__device__ __forceinline__ void mma_m16n8k64(
    const uint32_t* a,
    const uint32_t* b,
    float* c)
{
    if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
        // 2× m16n8k32 for FP8 (64 elements = 2 × 32)
        mma_m16n8k32<compute_t>(a, a + 2, b, c);        // First k32: a[0..1], a[2..3]
        mma_m16n8k32<compute_t>(a + 4, a + 6, b + 2, c); // Second k32: would need more regs
        // Note: This simplified version assumes caller provides 8 A regs for 2× k32
    } else if constexpr (std::is_same_v<compute_t, half>) {
        // 4× m16n8k16 for F16 (64 elements = 4 × 16)
        // Requires a[0..3] for k0-15, a[4..7] for k16-31, etc.
        // b[0..1] for each k16 slice
        uint32_t a_slice[4];
        
        // k=0..15
        a_slice[0] = a[0]; a_slice[1] = a[1]; a_slice[2] = a[2]; a_slice[3] = a[3];
        mma_m16n8k16<compute_t>(a_slice, b, c);
        
        // k=16..31
        a_slice[0] = a[4]; a_slice[1] = a[5]; a_slice[2] = a[6]; a_slice[3] = a[7];
        mma_m16n8k16<compute_t>(a_slice, b + 2, c);
        
        // k=32..47
        a_slice[0] = a[8]; a_slice[1] = a[9]; a_slice[2] = a[10]; a_slice[3] = a[11];
        mma_m16n8k16<compute_t>(a_slice, b + 4, c);
        
        // k=48..63
        a_slice[0] = a[12]; a_slice[1] = a[13]; a_slice[2] = a[14]; a_slice[3] = a[15];
        mma_m16n8k16<compute_t>(a_slice, b + 6, c);
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        // 4× m16n8k16 for BF16
        uint32_t a_slice[4];
        
        a_slice[0] = a[0]; a_slice[1] = a[1]; a_slice[2] = a[2]; a_slice[3] = a[3];
        mma_m16n8k16<compute_t>(a_slice, b, c);
        
        a_slice[0] = a[4]; a_slice[1] = a[5]; a_slice[2] = a[6]; a_slice[3] = a[7];
        mma_m16n8k16<compute_t>(a_slice, b + 2, c);
        
        a_slice[0] = a[8]; a_slice[1] = a[9]; a_slice[2] = a[10]; a_slice[3] = a[11];
        mma_m16n8k16<compute_t>(a_slice, b + 4, c);
        
        a_slice[0] = a[12]; a_slice[1] = a[13]; a_slice[2] = a[14]; a_slice[3] = a[15];
        mma_m16n8k16<compute_t>(a_slice, b + 6, c);
    }
}

// -----------------------------------------------------------------------------
// Dequant wrapper for 4× m16n8k16 MMA: prepares frag_b[8] for half K/128 tile
// -----------------------------------------------------------------------------
// Processes 4 k16 slices (half of K/128). Better occupancy than 8x.
// half_idx: 0 = k16 slices 0-3, 1 = k16 slices 4-7 (template parameter)
// Explicit if constexpr dispatch ensures direct calls without indirection.
template <typename block_c_t, typename compute_t, int half_idx>
__device__ __forceinline__ void dequant_weights_4x_k16(
    const uint8_t* smem_rows, int lane,
    uint32_t frag_b[8])
{
    if constexpr (std::is_same_v<block_c_t, block_c_q8_0>) {
        dequant_q8_0_for_4x_mma_k16_runtime<compute_t, half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q4_0>) {
        gemx_dequant_traits<block_c_q4_0, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q4_1>) {
        gemx_dequant_traits<block_c_q4_1, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q5_0>) {
        gemx_dequant_traits<block_c_q5_0, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q5_1>) {
        gemx_dequant_traits<block_c_q5_1, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q2_K>) {
        gemx_dequant_traits<block_c_q2_K, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q3_K>) {
        gemx_dequant_traits<block_c_q3_K, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q4_K>) {
        gemx_dequant_traits<block_c_q4_K, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q5_K>) {
        gemx_dequant_traits<block_c_q5_K, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else if constexpr (std::is_same_v<block_c_t, block_c_q6_K>) {
        gemx_dequant_traits<block_c_q6_K, compute_t, half>::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    } else {
        // Fallback for any unlisted types
        using Traits = gemx_dequant_traits<block_c_t, compute_t, half>;
        Traits::template dequant_for_4x_mma_k16_runtime<half_idx>(smem_rows, lane, frag_b);
    }
}

// -----------------------------------------------------------------------------
// Output write helper with type dispatch
// -----------------------------------------------------------------------------
// Output writes use .cs (streaming) cache hint to avoid polluting L1/L2
// Output data is written once and never re-read, so streaming eviction is optimal
template <typename output_t>
__device__ __forceinline__ void write_output(
    output_t* dst, int dst_stride, int b, int row, float val)
{
    // Use .cs cache hint (streaming, bypass L2) for all output types
    if constexpr (std::is_same_v<output_t, float>) {
        asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(&dst[b * dst_stride + row]), "f"(val));
    } else if constexpr (std::is_same_v<output_t, half>) {
        asm volatile("st.global.cs.u16 [%0], %1;" :: "l"(&dst[b * dst_stride + row]), "h"(__half_as_ushort(__float2half(val))));
    } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
        asm volatile("st.global.cs.u16 [%0], %1;" :: "l"(&dst[b * dst_stride + row]), "h"(__bfloat16_as_ushort(__float2bfloat16(val))));
    } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
        unsigned int raw_byte = static_cast<unsigned int>(*reinterpret_cast<const unsigned char*>(&from_f32<__nv_fp8_e4m3>(val)));
        asm volatile("st.global.cs.u8 [%0], %1;" :: "l"(&dst[b * dst_stride + row]), "r"(raw_byte));
    }
}

} // namespace tc_common

namespace s16_tc {

// Import common constants
using namespace tc_common;

// -----------------------------------------------------------------------------
// Load activations: global → shared memory (cooperative across all threads)
// Vectorized paths for half and float inputs for better memory bandwidth
// Note: Assumes batch_size % BATCH_TILE == 0 and ncols % K_TILE == 0
//       (true for all common LLM dimensions) - allows branchless loading
// -----------------------------------------------------------------------------
template <typename compute_t, typename act_t>
__device__ __forceinline__ void load_activations(
    compute_t smem_A[BATCH_TILE][K_STRIDE],
    const act_t* __restrict__ vy,
    int batch_start, int k_offset, int y_stride, int tid)
{
    // === VECTORIZED HALF → HALF PATH ===
    // 128 threads × 2 iterations × 8 elements = 2048 elements
    // Each thread loads 16 consecutive bytes (8 halfs) per iteration
    // Use .cg cache hint: bypass L1, cache in L2 for cross-SM sharing
    if constexpr (std::is_same_v<act_t, half> && std::is_same_v<compute_t, half>) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 8;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            const int gb = batch_start + b;
            const int gk = k_offset + k;
            
            int4 data = *reinterpret_cast<const int4*>(&vy[gb * y_stride + gk]);
            *reinterpret_cast<int4*>(&smem_A[b][k]) = data;
        }
    }
    // === VECTORIZED FLOAT → HALF PATH ===
    // 128 threads × 4 iterations × 4 elements = 2048 elements
    // Each thread loads float4 (16 bytes) and converts to 4 halfs
    // Use .cg cache hint: bypass L1, cache in L2 for cross-SM sharing
    else if constexpr (std::is_same_v<act_t, float> && std::is_same_v<compute_t, half>) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 4;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            const int gb = batch_start + b;
            const int gk = k_offset + k;
            
            float4 f4 = *reinterpret_cast<const float4*>(&vy[gb * y_stride + gk]);
            
            // Convert and store as 2 half2
            half2 h0 = __floats2half2_rn(f4.x, f4.y);
            half2 h1 = __floats2half2_rn(f4.z, f4.w);
            *reinterpret_cast<half2*>(&smem_A[b][k]) = h0;
            *reinterpret_cast<half2*>(&smem_A[b][k + 2]) = h1;
        }
    }
    // === VECTORIZED BF16 → BF16 PATH ===
    // Same as half path - 16 bytes = 8 bf16 elements
    // Use .cg cache hint: bypass L1, cache in L2 for cross-SM sharing
    else if constexpr (std::is_same_v<act_t, __nv_bfloat16> && std::is_same_v<compute_t, __nv_bfloat16>) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 8;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            const int gb = batch_start + b;
            const int gk = k_offset + k;
            
            int4 data = *reinterpret_cast<const int4*>(&vy[gb * y_stride + gk]);
            *reinterpret_cast<int4*>(&smem_A[b][k]) = data;
        }
    }
    // === SCALAR FALLBACK PATH (FP8, mixed types) ===
    else {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx = tid + i * NUM_THREADS;
            const int b = idx / K_TILE;
            const int k = idx % K_TILE;
            const int gb = batch_start + b;
            const int gk = k_offset + k;
            
            float val;
            if constexpr (std::is_same_v<act_t, float>) {
                val = vy[gb * y_stride + gk];
            } else if constexpr (std::is_same_v<act_t, half>) {
                val = __half2float(vy[gb * y_stride + gk]);
            } else if constexpr (std::is_same_v<act_t, __nv_bfloat16>) {
                val = __bfloat162float(vy[gb * y_stride + gk]);
            } else if constexpr (std::is_same_v<act_t, __nv_fp8_e4m3>) {
                val = to_f32(vy[gb * y_stride + gk]);
            }
            
            if constexpr (std::is_same_v<compute_t, half>) {
                smem_A[b][k] = __float2half(val);
            } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                smem_A[b][k] = __float2bfloat16(val);
            } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
                smem_A[b][k] = __nv_fp8_e4m3(__float2half(val));
            }
        }
    }
}

// -----------------------------------------------------------------------------
// TC kernel implementation: takes SMEM as parameters for unified dispatch
// This avoids SMEM summing when called from dispatch functions
// -----------------------------------------------------------------------------
template <typename block_c_t, typename compute_t, typename act_t, typename output_t>
__device__ void tc16_kernel_impl(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride, int batch_size,
    compute_t smem_A[][K_STRIDE],
    uint8_t* smem_W_flat,
    int batch_offset = 0,
    int batch_tile_idx_in = 0,   // Actual batch tile index (not blockIdx.x)
    int total_batch_tiles = 0,   // Total batch tiles (for bounds check)
    int row_tile_idx = -1)       // -1 = use blockIdx.y (legacy), >= 0 = hierarchical decode
{
    // Hierarchical grid decode: use passed row_tile_idx if >= 0, else blockIdx.y
    const int n_block = (row_tile_idx >= 0) ? row_tile_idx : (int)blockIdx.y;
    const int tid = threadIdx.y * WARP_SIZE_TC + threadIdx.x;
    const int warp_id = tid / WARP_SIZE_TC;
    const int lane = tid % WARP_SIZE_TC;
    
    const int row0 = n_block * N_TILE;
    // For hierarchical grid: batch_offset already contains the base batch
    const int batch0 = batch_offset;
    
    // Bounds check: row must be valid and batch must be in range
    if (row0 >= nrows || batch0 >= batch_size) return;
    
    constexpr int K128_BYTES = sizeof(block_c_t);
    
    const int warp_row_base = row0 + warp_id * 8;
    uint8_t* smem_W = smem_W_flat + warp_id * 8 * K128_BYTES;
    
    float frag_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    const int k_blocks = ncols / K_TILE;
    
    for (int k_blk = 0; k_blk < k_blocks; ++k_blk) {
        load_activations<compute_t, act_t>(smem_A, activations, batch0, k_blk * K_TILE, 
                                            y_stride, tid);
        
        // cp.async: SM80+ always has async copy
        load_weights_async_coop<block_c_t>(smem_W_flat, weights, row0, 
                                            k_blk, nrows, k_blocks, tid);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
        
        // Process K/128: first half (k16 slices 0-3)
        {
            uint32_t frag_b[8];
            dequant_weights_4x_k16<block_c_t, compute_t, 0>(smem_W, lane, frag_b);
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t frag_a[4];
                load_frag_a<compute_t>(frag_a, smem_A, i * MMA_K, lane);
                mma_m16n8k16<compute_t>(frag_a, frag_b + i * 2, frag_c);
            }
        }
        // Second half (k16 slices 4-7)
        {
            uint32_t frag_b[8];
            dequant_weights_4x_k16<block_c_t, compute_t, 1>(smem_W, lane, frag_b);
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t frag_a[4];
                load_frag_a<compute_t>(frag_a, smem_A, (4 + i) * MMA_K, lane);
                mma_m16n8k16<compute_t>(frag_a, frag_b + i * 2, frag_c);
            }
        }
        
        __syncthreads();
    }
    
    const int groupID = lane / 4;
    const int threadID_in_group = lane % 4;
    const int out_row = warp_row_base + threadID_in_group * 2;
    const int b0 = batch0 + groupID;
    const int b1 = batch0 + groupID + 8;
    
    if (b0 < batch_size) {
        if constexpr (std::is_same_v<output_t, float>) {
            *reinterpret_cast<float2*>(&dst[b0 * dst_stride + out_row]) = make_float2(frag_c[0], frag_c[1]);
        } else if constexpr (std::is_same_v<output_t, half>) {
            *reinterpret_cast<half2*>(&dst[b0 * dst_stride + out_row]) = __floats2half2_rn(frag_c[0], frag_c[1]);
        } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
            *reinterpret_cast<__nv_bfloat162*>(&dst[b0 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c[0], frag_c[1]);
        } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
            dst[b0 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c[0]);
            dst[b0 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c[1]);
        }
    }
    if (b1 < batch_size) {
        if constexpr (std::is_same_v<output_t, float>) {
            *reinterpret_cast<float2*>(&dst[b1 * dst_stride + out_row]) = make_float2(frag_c[2], frag_c[3]);
        } else if constexpr (std::is_same_v<output_t, half>) {
            *reinterpret_cast<half2*>(&dst[b1 * dst_stride + out_row]) = __floats2half2_rn(frag_c[2], frag_c[3]);
        } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
            *reinterpret_cast<__nv_bfloat162*>(&dst[b1 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c[2], frag_c[3]);
        } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
            dst[b1 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c[2]);
            dst[b1 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c[3]);
        }
    }
}

// -----------------------------------------------------------------------------
// Main TC kernel: standalone wrapper that declares SMEM and calls impl
// Uses cp.async for overlapped memory transfers with cooperative loading
//
// batch_offset: Global batch index where this segment starts (for output indexing)
// tile_offset:  blockIdx.y value where this segment starts (for local tile calculation)
// local_batch_tiles: Number of batch tiles this segment should process (for bounds)
//                    Default 0 means use gridDim.y (standalone mode)
// -----------------------------------------------------------------------------
template <typename block_c_t, typename compute_t, typename act_t, typename output_t>
__device__ void tc16_kernel(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride, int batch_size,
    int batch_offset = 0,
    int tile_offset = 0,
    int local_batch_tiles = 0)
{
    assert(nrows % N_TILE == 0 && "TC kernel requires nrows to be a multiple of 32");
    assert(ncols % K_TILE == 0 && "TC kernel requires ncols to be a multiple of 128");
    
    constexpr int K128_BYTES = sizeof(block_c_t);
    
    __shared__ compute_t smem_A[BATCH_TILE][K_STRIDE];
    __shared__ uint8_t smem_W_flat[N_TILE * K128_BYTES];
    
    tc16_kernel_impl<block_c_t, compute_t, act_t, output_t>(
        weights, activations, dst, ncols, nrows, y_stride, dst_stride, batch_size,
        smem_A, smem_W_flat, batch_offset, tile_offset, local_batch_tiles);
}

} // namespace s16_tc

// =============================================================================
// GROUPED TENSOR-CORE GEMM — all MoE experts in a SINGLE kernel launch
// =============================================================================
// The per-expert path dispatches N experts as N separate launches (one per
// segment). At small tokens-per-expert that is both launch-bound AND occupancy-
// starved: each expert's matmul fills only ceil(N/32) row-blocks, so a handful
// of blocks idle most of the GPU. This kernel processes ALL experts' tiles in
// one launch — grid = (total_tiles, row_tiles). A device-side table maps
// blockIdx.x → the owning expert's weight pointer and its [b_start,b_start+b_cnt)
// slice of the stacked activations/output (b_cnt ≤ 16, one MMA M-tile). The
// combined expert tiles fill the SMs, so occupancy and launch count improve
// together. Reuses the tc16 MMA inner loop verbatim.
//
// TILE WIDTH: this kernel only emits 16-wide (tc16-style) MMA tiles — an expert
// with M tokens becomes ceil(M/16) tiles. That is optimal for MoE decode, where
// tokens-per-expert is small (1 single-session, ~4-16 wave-batched). For very
// large M-per-expert (>= ~32) the per-weight path would escalate to the 32-wide
// tc32 kernel, which re-reads each weight block half as often; so a single
// expert carrying hundreds of tokens (a regime MoE decode does not hit) would
// run marginally faster through the per-segment path than through this one. A
// 32-wide grouped tile variant would close that gap if such a workload appears.
//
// PRECONDITION: nrows (output features) must be a multiple of N_TILE (32) — the
// row-tile writes a full 32-row block with no partial-row guard, same as the
// tc16 path. All MoE expert dims (768 / 2048) satisfy this; grouped_matmul_gemx
// enforces it host-side.
namespace grouped_tc {
using namespace tc_common;

// Runtime-batch activation load: BATCH_TILE×K_TILE into smem, rows >= b_cnt
// zeroed (M-dim padding). Mirrors s16_tc::load_activations but bounds at runtime
// so a tile may carry 1..16 tokens without reading past its expert's slice.
template <typename compute_t, typename act_t>
__device__ __forceinline__ void load_activations_runtime(
    compute_t smem_A[BATCH_TILE][K_STRIDE],
    const act_t* __restrict__ vy,
    int b_start, int b_cnt, int k_offset, int y_stride, int tid)
{
    if constexpr ((std::is_same_v<act_t, half> && std::is_same_v<compute_t, half>) ||
                  (std::is_same_v<act_t, __nv_bfloat16> &&
                   std::is_same_v<compute_t, __nv_bfloat16>)) {
        constexpr int4 ZERO4 = {0, 0, 0, 0};
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 8;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            int4 data = (b < b_cnt)
                ? *reinterpret_cast<const int4*>(&vy[(b_start + b) * y_stride + k_offset + k])
                : ZERO4;
            *reinterpret_cast<int4*>(&smem_A[b][k]) = data;
        }
    } else if constexpr (std::is_same_v<act_t, float> && std::is_same_v<compute_t, half>) {
        const half2 ZERO2 = {__float2half(0.f), __float2half(0.f)};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 4;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            half2 h0 = ZERO2, h1 = ZERO2;
            if (b < b_cnt) {
                float4 f4 = *reinterpret_cast<const float4*>(
                    &vy[(b_start + b) * y_stride + k_offset + k]);
                h0 = __floats2half2_rn(f4.x, f4.y);
                h1 = __floats2half2_rn(f4.z, f4.w);
            }
            *reinterpret_cast<half2*>(&smem_A[b][k]) = h0;
            *reinterpret_cast<half2*>(&smem_A[b][k + 2]) = h1;
        }
    } else if constexpr (std::is_same_v<act_t, float> &&
                         std::is_same_v<compute_t, __nv_bfloat16>) {
        const __nv_bfloat162 ZERO2 = {__float2bfloat16(0.f), __float2bfloat16(0.f)};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 4;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            __nv_bfloat162 h0 = ZERO2, h1 = ZERO2;
            if (b < b_cnt) {
                float4 f4 = *reinterpret_cast<const float4*>(
                    &vy[(b_start + b) * y_stride + k_offset + k]);
                h0 = __floats2bfloat162_rn(f4.x, f4.y);
                h1 = __floats2bfloat162_rn(f4.z, f4.w);
            }
            *reinterpret_cast<__nv_bfloat162*>(&smem_A[b][k]) = h0;
            *reinterpret_cast<__nv_bfloat162*>(&smem_A[b][k + 2]) = h1;
        }
    } else {
        // Scalar fallback (fp8 / mixed); zero rows beyond b_cnt.
        constexpr int TOTAL = BATCH_TILE * K_TILE;
        for (int idx = tid; idx < TOTAL; idx += NUM_THREADS) {
            const int b = idx / K_TILE;
            const int k = idx % K_TILE;
            float val = 0.f;
            if (b < b_cnt) {
                const int off = (b_start + b) * y_stride + k_offset + k;
                if constexpr (std::is_same_v<act_t, float>) val = vy[off];
                else if constexpr (std::is_same_v<act_t, half>) val = __half2float(vy[off]);
                else if constexpr (std::is_same_v<act_t, __nv_bfloat16>) val = __bfloat162float(vy[off]);
                else if constexpr (std::is_same_v<act_t, __nv_fp8_e4m3>) val = to_f32(vy[off]);
            }
            if constexpr (std::is_same_v<compute_t, half>) smem_A[b][k] = __float2half(val);
            else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) smem_A[b][k] = __float2bfloat16(val);
            else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) smem_A[b][k] = __nv_fp8_e4m3(__float2half(val));
        }
    }
}

// Shared tile-output store: the accumulated frag_c[4] → dst for the two token
// rows (b0, b1) this lane owns, at columns out_row / out_row+1. The (batch,row)
// mapping is identical for the FP and INT8 grouped impls, so both call this.
template <typename output_t>
__device__ __forceinline__ void store_tile_output(
    output_t* __restrict__ dst, const float frag_c[4],
    int dst_stride, int warp_row_base, int b_start, int b_cnt, int lane)
{
    const int groupID = lane / 4;
    const int threadID_in_group = lane % 4;
    const int out_row = warp_row_base + threadID_in_group * 2;
    const int bend = b_start + b_cnt;
    const int b0 = b_start + groupID;
    const int b1 = b_start + groupID + 8;

    if (b0 < bend) {
        if constexpr (std::is_same_v<output_t, float>) {
            *reinterpret_cast<float2*>(&dst[b0 * dst_stride + out_row]) = make_float2(frag_c[0], frag_c[1]);
        } else if constexpr (std::is_same_v<output_t, half>) {
            *reinterpret_cast<half2*>(&dst[b0 * dst_stride + out_row]) = __floats2half2_rn(frag_c[0], frag_c[1]);
        } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
            *reinterpret_cast<__nv_bfloat162*>(&dst[b0 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c[0], frag_c[1]);
        } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
            dst[b0 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c[0]);
            dst[b0 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c[1]);
        }
    }
    if (b1 < bend) {
        if constexpr (std::is_same_v<output_t, float>) {
            *reinterpret_cast<float2*>(&dst[b1 * dst_stride + out_row]) = make_float2(frag_c[2], frag_c[3]);
        } else if constexpr (std::is_same_v<output_t, half>) {
            *reinterpret_cast<half2*>(&dst[b1 * dst_stride + out_row]) = __floats2half2_rn(frag_c[2], frag_c[3]);
        } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
            *reinterpret_cast<__nv_bfloat162*>(&dst[b1 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c[2], frag_c[3]);
        } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
            dst[b1 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c[2]);
            dst[b1 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c[3]);
        }
    }
}

// One (expert-tile, row-tile): MMA over K for up to 16 tokens × 32 rows.
// Identical inner loop to s16_tc::tc16_kernel_impl, but the weight pointer is
// the per-expert pointer and the batch slice is [b_start, b_start+b_cnt).
template <typename block_c_t, typename compute_t, typename act_t, typename output_t>
__device__ void grouped_matmul_impl(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride,
    int b_start, int b_cnt, int row_tile_idx,
    compute_t smem_A[BATCH_TILE][K_STRIDE], uint8_t* smem_W_flat)
{
    const int tid = threadIdx.y * WARP_SIZE_TC + threadIdx.x;
    const int warp_id = tid / WARP_SIZE_TC;
    const int lane = tid % WARP_SIZE_TC;
    const int row0 = row_tile_idx * N_TILE;
    if (row0 >= nrows || b_cnt <= 0) return;

    constexpr int K128_BYTES = sizeof(block_c_t);
    const int warp_row_base = row0 + warp_id * 8;
    uint8_t* smem_W = smem_W_flat + warp_id * 8 * K128_BYTES;

    float frag_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int k_blocks = ncols / K_TILE;

    for (int k_blk = 0; k_blk < k_blocks; ++k_blk) {
        load_activations_runtime<compute_t, act_t>(
            smem_A, activations, b_start, b_cnt, k_blk * K_TILE, y_stride, tid);
        load_weights_async_coop<block_c_t>(smem_W_flat, weights, row0, k_blk, nrows, k_blocks, tid);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        {
            uint32_t frag_b[8];
            dequant_weights_4x_k16<block_c_t, compute_t, 0>(smem_W, lane, frag_b);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t frag_a[4];
                load_frag_a<compute_t>(frag_a, smem_A, i * MMA_K, lane);
                mma_m16n8k16<compute_t>(frag_a, frag_b + i * 2, frag_c);
            }
        }
        {
            uint32_t frag_b[8];
            dequant_weights_4x_k16<block_c_t, compute_t, 1>(smem_W, lane, frag_b);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t frag_a[4];
                load_frag_a<compute_t>(frag_a, smem_A, (4 + i) * MMA_K, lane);
                mma_m16n8k16<compute_t>(frag_a, frag_b + i * 2, frag_c);
            }
        }
        __syncthreads();
    }

    store_tile_output<output_t>(dst, frag_c, dst_stride, warp_row_base, b_start, b_cnt, lane);
}

// =============================================================================
// INT8 q8a128 PATH — same grid / entry / store machinery as the FP grouped impl
// above; only the activation load, weight unpack, and MMA+fold differ. The
// contraction runs on the INT8 m16n8k32 tensor core: q8a128 activations (raw int8
// qs) × quantized weights (raw integers), int32 accumulate, deferred-scale fold to
// F32 with the per-sub {scale, min}. Per-16-scale K-quants split the k32 MMA in two
// (Q6_K/Q3_K symmetric; Q2_K affine + an all-ones MMA for per-16 activation sums).
// See docs/q8_matmul_pipeline.md.
// =============================================================================

// Activation tile load: global → shared via cp.async (.ca, L1-resident — the tile is re-read
// by every grid.y row-tile). Issues the async copies ONLY; the caller commits them in the same
// cp.async group as the weight chunk and waits later, so the activation load pipelines a tile
// ahead of compute (its latency hides under the prior tile's MMA). qs: 128 threads × 16 B
// (token = tid>>3, kchunk = (tid&7)*16); ds: one (scale,sum) half2 per token. Rows ≥ b_cnt are
// zero-padded with a plain smem store (no global read, not part of the cp.async group).
template <int N_SUB = 1>
__device__ __forceinline__ void load_q8a128_activations(
    const block_q8a128* __restrict__ act, int b_start, int b_cnt, int k_blk,
    int tiles_per_row, int tid,
    int8_t smem_A_i8[N_SUB * 16][KI8_STRIDE],
    half2 smem_A_ds[N_SUB * 16])
{
    // q8a1024 flat-grouped activation: flat_tile = token*tiles_per_row + k_blk locates
    // the tile's de-interleaved qs (128-aligned line) and ds within its super-block.
    const uint8_t* abytes = reinterpret_cast<const uint8_t*>(act);
    const int kchunk = (tid & 7) * 16;
    #pragma unroll
    for (int thalf = 0; thalf < N_SUB; ++thalf) {
        const int token = (tid >> 3) + thalf * 16;
        if (token < b_cnt) {
            const int64_t flat = (int64_t)(b_start + token) * tiles_per_row + k_blk;
            cp_async_ca16(&smem_A_i8[token][kchunk], abytes + q8a1024_qs_off(flat) + kchunk);
        } else {
            *reinterpret_cast<int4*>(&smem_A_i8[token][kchunk]) = make_int4(0, 0, 0, 0);
        }
    }
    // ds: one (scale, sum) per token (per-128). One thread per token; threads past the tile idle.
    // 128 threads cover up to 128 tokens, so one pass handles N_SUB*16 (N_SUB ≤ 8).
    {
        const int t2 = tid;   // token
        if (t2 < N_SUB * 16) {
            if (t2 < b_cnt) {
                const int64_t flat = (int64_t)(b_start + t2) * tiles_per_row + k_blk;
                cp_async_ca4(&smem_A_ds[t2], abytes + q8a1024_ds_off(flat));   // (scale,sum) fp16
            } else {
                smem_A_ds[t2] = make_half2(__float2half(1.f), __float2half(0.f));
            }
        }
    }
}

// One (expert-tile, row-tile) on the INT8 tensor core. Same single-stage load loop
// and store as grouped_matmul_impl: load_q8a128_activations + a single per-warp weight
// slot, in 8-row chunks. Each warp streams its row-group's chunks through ONE slot — the
// dequant drains the loaded chunk to registers, freeing the slot for the next prefetch, so
// no ring/ping-pong is needed (the double-buffered activations provide the load/compute
// overlap). RING_I8 stays at 1 and sizes the single weight slot in shared memory.
constexpr int RING_I8 = 1;

// Load this warp's own k1024 weight chunk for `k_blk` into `slot_ptr` with its 32 lanes.
// `weights` is an array of k1024 chunks laid out [k_blk][row-group of 8]; the chunk carries
// its quants AND its inline scales, so this single contiguous coalesced cp.async brings the
// whole chunk in (no separate scale stream). Caller commits the group.
template <typename block_c_t>
__device__ __forceinline__ void load_warp_chunk_int8(
    uint8_t* slot_ptr, const block_c_t* __restrict__ weights, int k_blk,
    int warp_row_base, int nrows, int lane)
{
    constexpr int N16 = int8_chunk_bytes<block_c_t>::value / 16;   // 16B units in the chunk
    const uint8_t* sbytes;
    if constexpr (is_scale_separate<block_c_t>::value) {
        // KO k1024: one 8-row chunk, indexed [k_blk][row-group of 8].
        sbytes = reinterpret_cast<const uint8_t*>(&weights[(int64_t)k_blk * (nrows / 8) + warp_row_base / 8]);
    } else {
        // Legacy non-KO: 8 contiguous per-row blocks at [k_blk][row].
        sbytes = reinterpret_cast<const uint8_t*>(&weights[(int64_t)k_blk * nrows + warp_row_base]);
    }
    for (int i = lane; i < N16; i += WARP_SIZE_TC) {
        cp_async_cg16(slot_ptr + i * 16, sbytes + i * 16);   // quants (+ inline scales for KO)
    }
}

// load_warp_chunk_int8 weight stage, then the int8 MMA + deferred fold per sub.
// N_SUB = m16 token sub-tiles per block. Mode-1 = 1 (Bm 16); mode-2 = larger Bm so each
// weight chunk's dequant is reused across N_SUB token sub-tiles (fewer weight re-reads).
template <int qk, int qi, typename block_q_t, int vdr, typename output_t, int N_SUB = 1>
__device__ void grouped_matmul_impl_int8(
    const block_compact_t<block_q_t>* __restrict__ weights,
    const block_q8a128* __restrict__ act,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride,
    int b_start, int b_cnt, int row_tile_idx,
    int8_t smem_A_i8[][N_SUB * 16][KI8_STRIDE],
    half2 smem_A_ds[][N_SUB * 16],
    uint8_t* smem_W_flat)
{
    using block_c_t = block_compact_t<block_q_t>;
    const int tid = threadIdx.y * WARP_SIZE_TC + threadIdx.x;
    const int warp_id = tid / WARP_SIZE_TC;
    const int lane = tid % WARP_SIZE_TC;
    const int row0 = row_tile_idx * N_TILE;
    if (row0 >= nrows || b_cnt <= 0) return;

    const int warp_row_base = row0 + warp_id * 8;
    const int tiles_per_row = ncols / K_TILE;  // block_q8a128 tiles per activation row

    // N_SUB m16 token sub-tiles per block: frag_c[t*4 .. t*4+3] = tokens [t*16, t*16+16).
    // The weight (and its dequant) is shared across them — dequanted once per k_blk, reused.
    float frag_c[4 * N_SUB] = {};
    const int k_blocks = ncols / K_TILE;

    const int groupID  = lane >> 2;
    const int threadID = lane & 3;

    // Software-pipelined load. Per-warp weight slot (RING_I8=1, reused — the dequant drains it to
    // registers, freeing it for the next prefetch) + double-buffered activations (ping-pong). Each
    // iteration prefetches tile k+1's weight chunk AND activation tile in ONE cp.async group, so
    // their loads run during tile k's MMA; the activation barrier then waits only for warp skew,
    // not load latency. No WAR barrier — the next tile lands in the other activation buffer.
    // (Single-buffering mode-2's larger activation was measured to regress 7–14% at M=4096 — the
    // intra-block overlap beats the occupancy it would buy back, so both modes double-buffer.)
    constexpr int CB = int8_chunk_bytes<block_c_t>::value;   // one 8-row chunk
    uint8_t* my_slot = smem_W_flat + warp_id * CB;

    // Prologue: tile 0 — weight chunk 0 + activation tile 0 (buffer 0), one cp.async group.
    if (k_blocks > 0) {
        load_warp_chunk_int8<block_c_t>(my_slot, weights, 0, warp_row_base, nrows, lane);
        load_q8a128_activations<N_SUB>(act, b_start, b_cnt, 0, tiles_per_row, tid,
                                       smem_A_i8[0], smem_A_ds[0]);
        cp_async_commit();
    }

    for (int k_blk = 0; k_blk < k_blocks; ++k_blk) {
        const int ab = k_blk & 1;        // this tile's activation buffer
        const int nb = (k_blk + 1) & 1;  // next tile's buffer (held tile k-1, already consumed)
        cp_async_wait_group<0>();        // tile k_blk (weight + activation) resident

        // Inline scales + up-front dequant: read this warp's chunk into registers. Intra-warp
        // (no barrier needed); frees the weight slot for the next prefetch. The (scale,min) live
        // in blk.dm[row]; this thread owns rows (threadID*2, threadID*2+1).
        const block_c_t* blk = reinterpret_cast<const block_c_t*>(my_slot);
        const int rl = threadID * 2;
        // dm[rl] and dm[rl+1] are adjacent half2 (8 B, rl*4 is 8-aligned) → ONE int2 LDS.64.
        const int2 dd = *reinterpret_cast<const int2*>(&blk->dm[rl]);
        const float2 d0 = __half22float2(*reinterpret_cast<const half2*>(&dd.x));  // (d, m) row rl
        const float2 d1 = __half22float2(*reinterpret_cast<const half2*>(&dd.y));  // (d, m) row rl+1
        // Lane-major dequant: this lane's 4 subs are stored contiguously, so each stream is
        // pulled in ONE wide LDS (ql int4, plus crumb/hi for Q5/Q6, or 2 int4 for Q8) instead
        // of 4 per-sub loads — fewer MIO instructions, still bank-conflict-free.
        uint32_t b_frags[4][2];
        gemx_dequant_traits<block_c_t, half, half>::dequant_all_subs_int8(my_slot, lane, b_frags);

        __syncthreads();  // RAW: tile k activation visible to all warps; also guarantees tile
                          // k-1's MMA (which read buffer nb) finished before we prefetch into nb.

        // Prefetch tile k+1: weight chunk into the freed slot + activation into buffer nb, ONE
        // cp.async group. Both overlap the MMA below (registers + buffer ab only). WAR-safe:
        // slot freed by the dequant above; buffer nb consumed before the barrier above.
        if (k_blk + 1 < k_blocks) {
            load_warp_chunk_int8<block_c_t>(my_slot, weights, k_blk + 1, warp_row_base, nrows, lane);
            load_q8a128_activations<N_SUB>(act, b_start, b_cnt, k_blk + 1, tiles_per_row, tid,
                                           smem_A_i8[nb], smem_A_ds[nb]);
            cp_async_commit();
        }

        // Per-128 collapse, per token sub-tile. The dequanted weight (b_frags, d0/d1) is REUSED
        // across all N_SUB sub-tiles — that's the mode-2 win (one weight dequant amortized over
        // N_SUB·16 tokens). Each sub-tile t has its own activation (smem at t*16) and output
        // frag_c[t*4..]. Two accumulators break the C-dependency chain per sub-tile.
        #pragma unroll
        for (int t = 0; t < N_SUB; ++t) {
            const float2 a0 = __half22float2(smem_A_ds[ab][t * 16 + groupID]);      // token-half A
            const float2 a1 = __half22float2(smem_A_ds[ab][t * 16 + groupID + 8]);  // token-half B
            int32_t C0[4] = {0, 0, 0, 0};
            int32_t C1[4] = {0, 0, 0, 0};
            #pragma unroll
            for (int sub = 0; sub < 4; sub += 2) {
                uint32_t a0f[4], a1f[4];
                fused_attn::load_a_frag_m16k32_ldmatrix(a0f, &smem_A_i8[ab][t * 16][sub * 32], KI8_STRIDE, lane);
                fused_attn::load_a_frag_m16k32_ldmatrix(a1f, &smem_A_i8[ab][t * 16][(sub + 1) * 32], KI8_STRIDE, lane);
                fused_attn::mma_int8_m16n8k32(C0, a0f, b_frags[sub], C0);
                fused_attn::mma_int8_m16n8k32(C1, a1f, b_frags[sub + 1], C1);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) C0[i] += C1[i];
            frag_c[t * 4 + 0] += d0.x * a0.x * (float)C0[0] + d0.y * a0.y;
            frag_c[t * 4 + 1] += d1.x * a0.x * (float)C0[1] + d1.y * a0.y;
            frag_c[t * 4 + 2] += d0.x * a1.x * (float)C0[2] + d0.y * a1.y;
            frag_c[t * 4 + 3] += d1.x * a1.x * (float)C0[3] + d1.y * a1.y;
        }
    }

    // One store per sub-tile: tile t → tokens [b_start+t*16, +16). store_tile_output
    // clamps via the token count, so partial/empty tiles write nothing.
    #pragma unroll
    for (int t = 0; t < N_SUB; ++t) {
        const int rem = b_cnt - t * 16;
        const int cnt = rem < 0 ? 0 : (rem > 16 ? 16 : rem);
        store_tile_output<output_t>(dst, &frag_c[t * 4], dst_stride, warp_row_base,
                                    b_start + t * 16, cnt, lane);
    }
}

// INT8 dense entry (regular non-MoE QMatMul): one weight, implicit tile schedule —
// blockIdx.x → the ≤16-token batch slice, blockIdx.y → the 32-row tile. Launched
// from run_quantized_matmul on ytype==3.
template <int qk, int qi, typename block_q_t, int vdr, typename output_t, int N_SUB = 1>
static __device__ void quantized_matmul_dense_entry_int8(
    const block_compact_t<block_q_t>* __restrict__ weights,
    const block_q8a128* __restrict__ act,
    output_t* __restrict__ dst,
    int ncols_x, int nrows_x, int total_batch, int y_stride, int dst_stride)
{
    constexpr int BATCH = N_SUB * 16;   // tokens per block (mode-1: 16, mode-2: 16·N_SUB)
    const int b_start = blockIdx.x * BATCH;
    const int b_cnt = min(BATCH, total_batch - b_start);
    const int row_tile_idx = blockIdx.y;

    __shared__ __align__(16) int8_t smem_A_i8[2][BATCH][KI8_STRIDE];   // double-buffered
    __shared__ __align__(16) half2 smem_A_ds[2][BATCH];
    __shared__ uint8_t smem_W_flat[(N_TILE / 8) * RING_I8 * int8_chunk_bytes<block_compact_t<block_q_t>>::value];

    // The int8 impl is KO-only (inline-scale k1024 chunks). Only instantiate it for KO; for
    // any non-KO type the call is discarded so the kernel is a no-op (never dispatched to).
    if constexpr (is_scale_separate<block_compact_t<block_q_t>>::value) {
        grouped_matmul_impl_int8<qk, qi, block_q_t, vdr, output_t, N_SUB>(
            weights, act, dst, ncols_x, nrows_x, y_stride, dst_stride,
            b_start, b_cnt, row_tile_idx, smem_A_i8, smem_A_ds, smem_W_flat);
    }
}

// Grouped entry: decode (expert, batch-slice) from device tables and run one tile.
//   grid = (total_tiles, row_tiles), block = 128 threads (4 warps × 32).
template <int qk, int qi, typename block_q_t, int vdr, typename act_t, typename output_t, int N_SUB = 1>
static __device__ void quantized_matmul_grouped_entry(
    const uint64_t* __restrict__ weight_ptrs,  // [num_experts] device weight pointers
    const int* __restrict__ tile_expert,       // [total_tiles] owning expert
    const int* __restrict__ tile_b_start,      // [total_tiles] stacked batch start
    const int* __restrict__ tile_b_cnt,        // [total_tiles] tokens in tile (1..16·N_SUB)
    const act_t* __restrict__ vy,
    output_t* __restrict__ dst,
    int ncols_x, int nrows_x, int y_stride, int dst_stride)
{
    using block_c_t = block_compact_t<block_q_t>;

    const int tile = blockIdx.x;
    const int expert = tile_expert[tile];
    const block_c_t* weights =
        reinterpret_cast<const block_c_t*>(static_cast<uintptr_t>(weight_ptrs[expert]));
    const int b_start = tile_b_start[tile];
    const int b_cnt = tile_b_cnt[tile];
    const int row_tile_idx = blockIdx.y;

    // Same decode / grid / store for every activation type; only the smem layout and
    // the per-tile compute differ. q8a128 → INT8 m16n8k32; FP → FP16 m16n8k16. N_SUB
    // scales the int8 token tile (mode-1: 16, mode-2: 32 weight-reuse); the impl sweeps
    // the once-loaded weight over each 16-row sub-tile, partial sub-tiles writing nothing.
    if constexpr (std::is_same_v<act_t, block_q8a128>) {
        constexpr int BATCH_I8 = N_SUB * 16;
        __shared__ __align__(16) int8_t smem_A_i8[2][BATCH_I8][KI8_STRIDE];   // double-buffered
        __shared__ __align__(16) half2 smem_A_ds[2][BATCH_I8];
        __shared__ uint8_t smem_W_flat[(N_TILE / 8) * RING_I8 * int8_chunk_bytes<block_c_t>::value];
        // KO-only int8 impl (inline-scale k1024). Non-KO → discarded (no-op kernel).
        if constexpr (is_scale_separate<block_c_t>::value) {
            grouped_matmul_impl_int8<qk, qi, block_q_t, vdr, output_t, N_SUB>(
                weights, vy, dst, ncols_x, nrows_x, y_stride, dst_stride,
                b_start, b_cnt, row_tile_idx, smem_A_i8, smem_A_ds, smem_W_flat);
        }
    } else {
        using compute_t = std::conditional_t<
            std::is_same_v<act_t, float>, half,
            std::conditional_t<
                std::is_same_v<act_t, half>, half,
                std::conditional_t<
                    std::is_same_v<act_t, __nv_bfloat16>, __nv_bfloat16,
                    std::conditional_t<
                        std::is_same_v<act_t, __nv_fp8_e4m3>, __nv_fp8_e4m3, half>>>>;
        __shared__ compute_t smem_A[BATCH_TILE][K_STRIDE];
        __shared__ uint8_t smem_W_flat[N_TILE * sizeof(block_c_t)];
        grouped_matmul_impl<block_c_t, compute_t, act_t, output_t>(
            weights, vy, dst, ncols_x, nrows_x, y_stride, dst_stride,
            b_start, b_cnt, row_tile_idx, smem_A, smem_W_flat);
    }
}

} // namespace grouped_tc

// =============================================================================
// SN_TC: TENSOR CORE KERNEL FOR BATCH 5-15 (RUNTIME PADDED MMA)
// =============================================================================
//
// OPERATION: dst[batch, row] = Σ_k Y[batch, k] × W[row, k]
//
// This kernel handles batch sizes 5-15 (never 1-4, never >= 16) by:
//   1. Loading actual batch data into rows 0..(batch_size-1)
//   2. Zero-padding rows batch_size..15 in the MMA activation fragment
//   3. Running the same MMA m16n8k16 as s16_tc
//   4. Only writing valid outputs (batch < batch_size)
//
// OPTIMIZATIONS FOR BATCH 5-15:
//   - Vectorized activation loading with simple `b < batch_size` predicate
//   - Batches 0-4 are ALWAYS valid (no check needed in hot paths)
//   - b0 (0-7): Only groupID 5-7 need batch_size check
//   - b1 (8-15): Always need check since batch_size < 16
//   - Single batch block (no gridDim.y loop)
//
// MMA m16n8k16: C[m,n] += A[m,k] × B[k,n]
//   A = activations [16 batch, 16 K]  - rows >= batch_size are zeroed
//   B = weights [16 K, 8 rows]        - same as s16_tc
//   C = output [16 batch, 8 rows]     - only write valid batches
//
// =============================================================================

namespace sN_tc {

// Import common constants and functions
using namespace tc_common;

// -----------------------------------------------------------------------------
// Load activations with VECTORIZED loads and zero-padding for batch 5-15
// 
// OPTIMIZATION: Uses int4 (16-byte) vectorized loads with simple b < batch_size
// predicate. Since batch_size is 5-15:
//   - Iteration 0: batches 0-7 × K_TILE - batches 0-4 always valid
//   - Iteration 1: batches 8-15 × K_TILE - all may need zero-padding
// BATCH_SIZE is compile-time constant for branch elimination
// batch_offset is applied to get global indices for memory reads
// -----------------------------------------------------------------------------
template <typename compute_t, typename act_t, int BATCH_SIZE>
__device__ __forceinline__ void load_activations_padded(
    compute_t smem_A[BATCH_TILE][K_STRIDE],
    const act_t* __restrict__ vy,
    int k_offset, int y_stride, int tid, int batch_offset = 0)
{
    // === VECTORIZED HALF → HALF PATH (most common) ===
    // 128 threads × 2 iterations × 8 elements = 2048 elements
    // Each thread loads 16 bytes (8 halfs) per iteration
    // Use .cg cache hint: bypass L1, cache in L2 for cross-SM sharing
    if constexpr (std::is_same_v<act_t, half> && std::is_same_v<compute_t, half>) {
        constexpr int4 ZERO4 = {0, 0, 0, 0};
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 8;
            const int b = elem_idx / K_TILE;   // local batch 0-15
            const int k = elem_idx % K_TILE;   // K offset
            const int gk = k_offset + k;
            const int gb = batch_offset + b;   // global batch index
            
            // Compile-time predicate: valid if b < BATCH_SIZE
            int4 data;
            if constexpr (BATCH_SIZE >= 16) {
                // All batches valid, no predicate needed
                data = *reinterpret_cast<const int4*>(&vy[gb * y_stride + gk]);
            } else {
                // Check at compile time which iterations need predicate
                // For BATCH_SIZE=8: b<8 always valid (iter 0), b>=8 always zero (iter 1)
                if (b < BATCH_SIZE) {
                    data = *reinterpret_cast<const int4*>(&vy[gb * y_stride + gk]);
                } else {
                    data = ZERO4;  // Zero-padding for unused batches
                }
            }
            *reinterpret_cast<int4*>(&smem_A[b][k]) = data;
        }
    }
    // === VECTORIZED BF16 → BF16 PATH ===
    // Use .cg cache hint: bypass L1, cache in L2 for cross-SM sharing
    else if constexpr (std::is_same_v<act_t, __nv_bfloat16> && std::is_same_v<compute_t, __nv_bfloat16>) {
        constexpr int4 ZERO4 = {0, 0, 0, 0};
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 8;
            const int b = elem_idx / K_TILE;   // local batch
            const int k = elem_idx % K_TILE;
            const int gk = k_offset + k;
            const int gb = batch_offset + b;   // global batch index
            
            int4 data;
            if constexpr (BATCH_SIZE >= 16) {
                data = *reinterpret_cast<const int4*>(&vy[gb * y_stride + gk]);
            } else {
                if (b < BATCH_SIZE) {
                    data = *reinterpret_cast<const int4*>(&vy[gb * y_stride + gk]);
                } else {
                    data = ZERO4;
                }
            }
            *reinterpret_cast<int4*>(&smem_A[b][k]) = data;
        }
    }
    // === VECTORIZED FLOAT → HALF PATH ===
    // 128 threads × 4 iterations × 4 elements = 2048 elements
    // Use .cg cache hint: bypass L1, cache in L2 for cross-SM sharing
    else if constexpr (std::is_same_v<act_t, float> && std::is_same_v<compute_t, half>) {
        const half2 ZERO2 = {__float2half(0.0f), __float2half(0.0f)};
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 4;
            const int b = elem_idx / K_TILE;   // local batch
            const int k = elem_idx % K_TILE;
            const int gk = k_offset + k;
            const int gb = batch_offset + b;   // global batch index
            
            half2 h0, h1;
            if constexpr (BATCH_SIZE >= 16) {
                float4 f4 = *reinterpret_cast<const float4*>(&vy[gb * y_stride + gk]);
                h0 = __floats2half2_rn(f4.x, f4.y);
                h1 = __floats2half2_rn(f4.z, f4.w);
            } else {
                if (b < BATCH_SIZE) {
                    float4 f4 = *reinterpret_cast<const float4*>(&vy[gb * y_stride + gk]);
                    h0 = __floats2half2_rn(f4.x, f4.y);
                    h1 = __floats2half2_rn(f4.z, f4.w);
                } else {
                    h0 = ZERO2;
                    h1 = ZERO2;
                }
            }
            *reinterpret_cast<half2*>(&smem_A[b][k]) = h0;
            *reinterpret_cast<half2*>(&smem_A[b][k + 2]) = h1;
        }
    }
    // === VECTORIZED FLOAT → BF16 PATH ===
    else if constexpr (std::is_same_v<act_t, float> && std::is_same_v<compute_t, __nv_bfloat16>) {
        const __nv_bfloat162 ZERO2 = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 4;
            const int b = elem_idx / K_TILE;   // local batch
            const int k = elem_idx % K_TILE;
            const int gk = k_offset + k;
            const int gb = batch_offset + b;   // global batch index
            
            __nv_bfloat162 h0, h1;
            if constexpr (BATCH_SIZE >= 16) {
                float4 f4 = *reinterpret_cast<const float4*>(&vy[gb * y_stride + gk]);
                h0 = __floats2bfloat162_rn(f4.x, f4.y);
                h1 = __floats2bfloat162_rn(f4.z, f4.w);
            } else {
                if (b < BATCH_SIZE) {
                    float4 f4 = *reinterpret_cast<const float4*>(&vy[gb * y_stride + gk]);
                    h0 = __floats2bfloat162_rn(f4.x, f4.y);
                    h1 = __floats2bfloat162_rn(f4.z, f4.w);
                } else {
                    h0 = ZERO2;
                    h1 = ZERO2;
                }
            }
            *reinterpret_cast<__nv_bfloat162*>(&smem_A[b][k]) = h0;
            *reinterpret_cast<__nv_bfloat162*>(&smem_A[b][k + 2]) = h1;
        }
    }
    // === FP8 PATH - vectorized uint32 loads ===
    else if constexpr (std::is_same_v<act_t, __nv_fp8_e4m3>) {
        // FP8: 128 threads × 4 iterations × 4 elements = 2048 elements
        // Each thread loads 4 bytes (4 FP8) per iteration
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 4;
            const int b = elem_idx / K_TILE;   // local batch
            const int k = elem_idx % K_TILE;
            const int gk = k_offset + k;
            const int gb = batch_offset + b;   // global batch index
            
            uint32_t data;
            if constexpr (BATCH_SIZE >= 16) {
                data = *reinterpret_cast<const uint32_t*>(&vy[gb * y_stride + gk]);
            } else {
                if (b < BATCH_SIZE) {
                    data = *reinterpret_cast<const uint32_t*>(&vy[gb * y_stride + gk]);
                } else {
                    data = 0;
                }
            }
            *reinterpret_cast<uint32_t*>(&smem_A[b][k]) = data;
        }
    }
    // === FALLBACK: Scalar path for other type combinations ===
    else {
        constexpr int TOTAL_ELEMENTS = BATCH_TILE * K_TILE;
        constexpr int ELEMS_PER_THREAD = (TOTAL_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;
        
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            const int idx = tid + i * NUM_THREADS;
            if (idx < TOTAL_ELEMENTS) {
                const int b = idx / K_TILE;    // local batch
                const int k = idx % K_TILE;
                const int gk = k_offset + k;
                const int gb = batch_offset + b;  // global batch index
                
                compute_t val;
                if (b < BATCH_SIZE) {
                    float f;
                    if constexpr (std::is_same_v<act_t, half>) {
                        f = __half2float(vy[gb * y_stride + gk]);
                    } else if constexpr (std::is_same_v<act_t, __nv_bfloat16>) {
                        f = __bfloat162float(vy[gb * y_stride + gk]);
                    } else {
                        f = static_cast<float>(vy[gb * y_stride + gk]);
                    }
                    if constexpr (std::is_same_v<compute_t, half>) {
                        val = __float2half(f);
                    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                        val = __float2bfloat16(f);
                    }
                } else {
                    if constexpr (std::is_same_v<compute_t, half>) {
                        val = __float2half(0.0f);
                    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                        val = __float2bfloat16(0.0f);
                    }
                }
                smem_A[b][k] = val;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Main TC kernel for batch 5-15
// OPTIMIZATIONS vs s16_tc:
//   1. No gridDim.y - single batch block handles all batches
//   2. Vectorized activation loading with simple batch predicate
//   3. Output writes: b0 (0-4) unconditional, b0 (5-7) + b1 (8-15) conditional
// BATCH_SIZE is a compile-time constant for optimal branch elimination
//
// batch_offset: Added for unified dispatch - this segment's batches start at batch_offset
//               Activations pointer should already be offset, but output needs this
// row_tile_idx: -1 = use blockIdx.y (legacy), >= 0 = hierarchical decode
// -----------------------------------------------------------------------------
template <typename block_c_t, typename compute_t, typename act_t, typename output_t, int BATCH_SIZE>
__device__ void tcN_kernel_impl(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride,
    compute_t smem_A[][K_STRIDE],
    uint8_t* smem_W_flat,
    int batch_offset = 0,
    int row_tile_idx = -1)  // -1 = use blockIdx.y (legacy), >= 0 = hierarchical decode
{
    static_assert(BATCH_SIZE >= 1 && BATCH_SIZE <= 15, "sN_tc kernel requires batch 1-15");
    
    // Hierarchical grid decode: use passed row_tile_idx if >= 0, else blockIdx.y
    const int n_block = (row_tile_idx >= 0) ? row_tile_idx : (int)blockIdx.y;
    const int tid = threadIdx.y * WARP_SIZE_TC + threadIdx.x;
    const int warp_id = tid / WARP_SIZE_TC;
    const int lane = tid % WARP_SIZE_TC;
    
    const int row0 = n_block * N_TILE;
    
    if (row0 >= nrows) return;
    
    constexpr int K128_BYTES = sizeof(block_c_t);
    
    const int warp_row_base = row0 + warp_id * 8;
    uint8_t* smem_W = smem_W_flat + warp_id * 8 * K128_BYTES;
    
    float frag_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int k_blocks = ncols / K_TILE;

    for (int k_blk = 0; k_blk < k_blocks; ++k_blk) {
        load_activations_padded<compute_t, act_t, BATCH_SIZE>(smem_A, activations,
                                                               k_blk * K_TILE, y_stride, tid, batch_offset);

        // cp.async: SM80+ always has async copy
        load_weights_async_coop<block_c_t>(smem_W_flat, weights, row0,
                                            k_blk, nrows, k_blocks, tid);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // Process K/128: first half (k16 slices 0-3)
        {
            uint32_t frag_b[8];
            dequant_weights_4x_k16<block_c_t, compute_t, 0>(smem_W, lane, frag_b);

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t frag_a[4];
                load_frag_a<compute_t>(frag_a, smem_A, i * MMA_K, lane);
                mma_m16n8k16<compute_t>(frag_a, frag_b + i * 2, frag_c);
            }

        }
        // Second half (k16 slices 4-7)
        {
            uint32_t frag_b[8];
            dequant_weights_4x_k16<block_c_t, compute_t, 1>(smem_W, lane, frag_b);
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t frag_a[4];
                load_frag_a<compute_t>(frag_a, smem_A, (4 + i) * MMA_K, lane);
                mma_m16n8k16<compute_t>(frag_a, frag_b + i * 2, frag_c);
            }
        }

        __syncthreads();
    }

    const int groupID = lane / 4;
    const int threadID_in_group = lane % 4;
    const int out_row = warp_row_base + threadID_in_group * 2;
    const int b0_local = groupID;
    const int b1_local = groupID + 8;
    const int b0 = batch_offset + b0_local;
    const int b1 = batch_offset + b1_local;

    if constexpr (BATCH_SIZE >= 8) {
        if constexpr (std::is_same_v<output_t, float>) {
            *reinterpret_cast<float2*>(&dst[b0 * dst_stride + out_row]) = make_float2(frag_c[0], frag_c[1]);
        } else if constexpr (std::is_same_v<output_t, half>) {
            *reinterpret_cast<half2*>(&dst[b0 * dst_stride + out_row]) = __floats2half2_rn(frag_c[0], frag_c[1]);
        } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
            *reinterpret_cast<__nv_bfloat162*>(&dst[b0 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c[0], frag_c[1]);
        } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
            dst[b0 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c[0]);
            dst[b0 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c[1]);
        }
    } else {
        if (b0_local < BATCH_SIZE) {
            if constexpr (std::is_same_v<output_t, float>) {
                *reinterpret_cast<float2*>(&dst[b0 * dst_stride + out_row]) = make_float2(frag_c[0], frag_c[1]);
            } else if constexpr (std::is_same_v<output_t, half>) {
                *reinterpret_cast<half2*>(&dst[b0 * dst_stride + out_row]) = __floats2half2_rn(frag_c[0], frag_c[1]);
            } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
                *reinterpret_cast<__nv_bfloat162*>(&dst[b0 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c[0], frag_c[1]);
            } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
                dst[b0 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c[0]);
                dst[b0 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c[1]);
            }
        }
    }
    
    if constexpr (BATCH_SIZE > 8) {
        if (b1_local < BATCH_SIZE) {
            if constexpr (std::is_same_v<output_t, float>) {
                *reinterpret_cast<float2*>(&dst[b1 * dst_stride + out_row]) = make_float2(frag_c[2], frag_c[3]);
            } else if constexpr (std::is_same_v<output_t, half>) {
                *reinterpret_cast<half2*>(&dst[b1 * dst_stride + out_row]) = __floats2half2_rn(frag_c[2], frag_c[3]);
            } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
                *reinterpret_cast<__nv_bfloat162*>(&dst[b1 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c[2], frag_c[3]);
            } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
                dst[b1 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c[2]);
                dst[b1 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c[3]);
            }
        }
    }
}

// Standalone wrapper that declares SMEM
template <typename block_c_t, typename compute_t, typename act_t, typename output_t, int BATCH_SIZE>
__device__ void tcN_kernel(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride,
    int batch_offset = 0)
{
    assert(nrows % N_TILE == 0 && "TC kernel requires nrows to be a multiple of 32");
    
    constexpr int K128_BYTES = sizeof(block_c_t);
    
    __shared__ compute_t smem_A[BATCH_TILE][K_STRIDE];
    __shared__ uint8_t smem_W_flat[N_TILE * K128_BYTES];
    
    tcN_kernel_impl<block_c_t, compute_t, act_t, output_t, BATCH_SIZE>(
        weights, activations, dst, ncols, nrows, y_stride, dst_stride,
        smem_A, smem_W_flat, batch_offset);
}

} // namespace sN_tc


// =============================================================================
// UNIFIED TC KERNEL DISPATCHER
// =============================================================================
// Single kernel entry point that dispatches to existing tc_kernel implementations
// based on blockIdx.y and the segment plan passed as kernel arguments.
//
// This enables a single kernel launch for multi-segment TC work (e.g., tc12+tc5
// for batch 17), keeping weights hot in L2 cache across all batch tiles.
//
// The dispatcher forks as early as possible based on blockIdx.y:
// - Segment 0: blockIdx.y in [0, seg1_tile_start)
// - Segment 1: blockIdx.y in [seg1_tile_start, total_tiles)
//
// Each segment calls the appropriate existing optimized tc_kernel with
// the batch_offset parameter to handle global batch indexing.
// =============================================================================

namespace tc16 {

using namespace tc_common;

// =============================================================================
// UNIFIED TC KERNEL - Handles all TC for batch 1-31 (and grid.y tiling for 32+)
// =============================================================================
// REMAINDER_BATCH: compile-time constant 0-15
//   - R=0: just tc16 with grid.y tiling (replaces standalone s16_tc)
//   - R=1-15: tc16 tiles + tcR remainder (replaces tc16+tcN two-launch)
//
// HIERARCHICAL GRID (kernel_cache_design.md):
//   x = batch tiles (L1 scope) - consecutive blocks share weights in L1
//   y = row tiles (L2 scope) - wave fills all SMs sharing activations
//   z = wave index: row_group + batch_group × num_row_groups
//
// Decode: row_tile = row_group * gridDim.y + blockIdx.y
//         batch_tile = batch_group * gridDim.x + blockIdx.x
//
// SMEM OPTIMIZATION: Declares unified SMEM once and passes to _impl functions
// to avoid NVCC summing SMEM from multiple __shared__ declarations.
// =============================================================================
template <typename block_c_t, typename compute_t, typename act_t, typename output_t, int REMAINDER_BATCH>
__device__ void dispatch_tc16_tcN(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride,
    int batch_size, int row_groups)
{
    static_assert(REMAINDER_BATCH >= 0 && REMAINDER_BATCH <= 15, 
                  "REMAINDER_BATCH must be 0-15");
    
    // Unified SMEM declaration - tc16 size covers both tc16 and tcN paths
    constexpr int K128_BYTES = sizeof(block_c_t);
    __shared__ compute_t smem_A[BATCH_TILE][K_STRIDE];  // 16 × 136
    __shared__ uint8_t smem_W_flat[N_TILE * K128_BYTES];  // 32 × block_bytes
    
    // =========================================================================
    // HIERARCHICAL GRID DECODE
    // =========================================================================
    // z = row_group + batch_group × num_row_groups (rows inner, batches outer)
    const int row_group = blockIdx.z % row_groups;
    const int batch_group = blockIdx.z / row_groups;
    
    // Decode actual tile indices from hierarchical grid
    const int row_tile_idx = row_group * gridDim.y + blockIdx.y;
    const int batch_tile_idx = batch_group * gridDim.x + blockIdx.x;
    
    // Total batch tiles for bounds checking
    const int tc16_tiles = batch_size / 16;
    const int total_batch_tiles = tc16_tiles + (REMAINDER_BATCH > 0 ? 1 : 0);
    const int row_tiles = (nrows + N_TILE - 1) / N_TILE;
    
    // Early exit if out of bounds
    if (row_tile_idx >= row_tiles || batch_tile_idx >= total_batch_tiles) return;
    
    // Compute tiling from batch_size
    const int remainder_start = tc16_tiles * 16;
    
    if constexpr (REMAINDER_BATCH == 0) {
        // R=0: All tiles are tc16 (grid.y = tc16_tiles)
        s16_tc::tc16_kernel_impl<block_c_t, compute_t, act_t, output_t>(
            weights, activations, dst, ncols, nrows, y_stride, dst_stride, 
            batch_size, smem_A, smem_W_flat, 0, batch_tile_idx, tc16_tiles,
            row_tile_idx);
    } else {
        // R=1-15: tc16 tiles + one tcR remainder tile
        if (batch_tile_idx >= tc16_tiles) {
            // Remainder tile: tcN kernel
            sN_tc::tcN_kernel_impl<block_c_t, compute_t, act_t, output_t, REMAINDER_BATCH>(
                weights, activations, dst, ncols, nrows, y_stride, dst_stride,
                smem_A, smem_W_flat, remainder_start, row_tile_idx);
        } else {
            // TC16 tiles
            s16_tc::tc16_kernel_impl<block_c_t, compute_t, act_t, output_t>(
                weights, activations, dst, ncols, nrows, y_stride, dst_stride, 
                batch_size, smem_A, smem_W_flat, 0, batch_tile_idx, tc16_tiles,
                row_tile_idx);
        }
    }
}

} // namespace tc16


// =============================================================================
// quantized_matmul_tc16_entry - TC16 kernel entry point
// =============================================================================
// Handles all TC for batch 1+ with automatic tiling:
//   - batch 1-15: single tcR tile (R = batch_size)
//   - batch 16: single tc16 tile
//   - batch 17-31: tc16 + tcR (R = batch_size % 16)
//   - batch 32+: tc16 with grid.y tiling + optional tcR remainder
//
// REMAINDER_BATCH: compile-time constant 0-15 (R = batch_size % 16)
//   - R=0: Only tc16 tiles (no remainder)
//   - R=1-15: tc16 tiles + one tcR remainder tile
//
// HIERARCHICAL GRID (kernel_cache_design.md):
//   x = batch tiles (L1 scope) - consecutive blocks share weights in L1
//   y = row tiles (L2 scope) - wave fills all SMs sharing activations
//   z = wave index: row_group + batch_group × num_row_groups
// row_groups parameter enables decode: row_group = z % row_groups
// =============================================================================
template <int qk, int qi, typename block_q_t, int vdr,
          typename act_t, typename output_t = float, int REMAINDER_BATCH = 0>
static __device__ void quantized_matmul_tc16_entry(
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
    static_assert(REMAINDER_BATCH >= 0 && REMAINDER_BATCH <= 15, 
                  "REMAINDER_BATCH must be 0-15");
    
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
    
    // Compile-time dispatch - hierarchical grid decode computed internally
    tc16::dispatch_tc16_tcN<block_c_t, compute_t, act_t, output_t, REMAINDER_BATCH>(
        weights, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size, row_groups);
}


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

// Configuration constants for s32_tc
constexpr int BATCH_TILE_32 = 32;       // 32 batches per tile (2 × 16 MMA)
constexpr int N_TILE = 32;              // 32 rows per block (same as s16_tc)
constexpr int K_TILE = 128;             // K/128 block size
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int MMA_ITERS = K_TILE / MMA_K;  // 8 iterations per K/128
constexpr int HALF_MMA_ITERS = MMA_ITERS / 2;  // 4 iterations (for 2x k16 batching)
constexpr int NUM_THREADS = 128;
constexpr int WARP_SIZE_TC = 32;
constexpr int NUM_WARPS = 4;
constexpr int K_PAD = 8;
constexpr int K_STRIDE = K_TILE + K_PAD;

// Import shared helpers from tc_common
using tc_common::load_frag_a;
using tc_common::load_weights_async_coop;
using tc_common::load_weights_warp;
using tc_common::mma_m16n8k16;
using tc_common::mma_m16n8k32;
using tc_common::dequant_weights_4x_k16;
using tc_common::cp_async_commit;
using tc_common::cp_async_wait_all;
using tc_common::write_output;

// -----------------------------------------------------------------------------
// Load activations for 32 batches: global → shared memory (cooperative)
// Vectorized paths for half and float inputs for better memory bandwidth
// Use .cg cache hint: bypass L1, cache in L2 for cross-SM sharing
// Note: Assumes batch_size % 32 == 0 and ncols % K_TILE == 0
// -----------------------------------------------------------------------------
template <typename compute_t, typename act_t>
__device__ __forceinline__ void load_activations_32(
    compute_t smem_A[BATCH_TILE_32][K_STRIDE],
    const act_t* __restrict__ vy,
    int batch_start, int k_offset, int y_stride, int tid)
{
    // === VECTORIZED HALF → HALF PATH ===
    // 128 threads × 4 iterations × 8 elements = 4096 elements (32 batches × 128 K)
    if constexpr (std::is_same_v<act_t, half> && std::is_same_v<compute_t, half>) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 8;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            const int gb = batch_start + b;
            const int gk = k_offset + k;
            
            int4 data = *reinterpret_cast<const int4*>(&vy[gb * y_stride + gk]);
            *reinterpret_cast<int4*>(&smem_A[b][k]) = data;
        }
    }
    // === VECTORIZED FLOAT → HALF PATH ===
    else if constexpr (std::is_same_v<act_t, float> && std::is_same_v<compute_t, half>) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 4;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            const int gb = batch_start + b;
            const int gk = k_offset + k;
            
            float4 f4 = *reinterpret_cast<const float4*>(&vy[gb * y_stride + gk]);
            half2 h0 = __floats2half2_rn(f4.x, f4.y);
            half2 h1 = __floats2half2_rn(f4.z, f4.w);
            *reinterpret_cast<half2*>(&smem_A[b][k]) = h0;
            *reinterpret_cast<half2*>(&smem_A[b][k + 2]) = h1;
        }
    }
    // === VECTORIZED BF16 → BF16 PATH ===
    else if constexpr (std::is_same_v<act_t, __nv_bfloat16> && std::is_same_v<compute_t, __nv_bfloat16>) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int elem_idx = (tid + i * NUM_THREADS) * 8;
            const int b = elem_idx / K_TILE;
            const int k = elem_idx % K_TILE;
            const int gb = batch_start + b;
            const int gk = k_offset + k;
            
            int4 data = *reinterpret_cast<const int4*>(&vy[gb * y_stride + gk]);
            *reinterpret_cast<int4*>(&smem_A[b][k]) = data;
        }
    }
    // === SCALAR FALLBACK PATH ===
    else {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int idx = tid + i * NUM_THREADS;
            const int b = idx / K_TILE;
            const int k = idx % K_TILE;
            if (b < BATCH_TILE_32) {
                const int gb = batch_start + b;
                const int gk = k_offset + k;
                
                float val;
                if constexpr (std::is_same_v<act_t, float>) {
                    val = vy[gb * y_stride + gk];
                } else if constexpr (std::is_same_v<act_t, half>) {
                    val = __half2float(vy[gb * y_stride + gk]);
                } else if constexpr (std::is_same_v<act_t, __nv_bfloat16>) {
                    val = __bfloat162float(vy[gb * y_stride + gk]);
                } else if constexpr (std::is_same_v<act_t, __nv_fp8_e4m3>) {
                    val = to_f32(vy[gb * y_stride + gk]);
                }
                
                if constexpr (std::is_same_v<compute_t, half>) {
                    smem_A[b][k] = __float2half(val);
                } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                    smem_A[b][k] = __float2bfloat16(val);
                } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
                    smem_A[b][k] = __nv_fp8_e4m3(__float2half(val));
                }
            }
        }
    }
}

// Load FragA for s32_tc: load from 32-batch shared memory using ldmatrix
// batch_half: 0 for batches 0-15, 1 for batches 16-31
template <typename compute_t>
__device__ __forceinline__ void load_frag_a_16(
    uint32_t frag_a[4],
    const compute_t smem_A[BATCH_TILE_32][K_STRIDE],
    int k_start, int lane, int batch_half)
{
    const int batch_offset = batch_half * 16;  // 0 or 16
    
    // Use ldmatrix for 16-bit types (half and bfloat16) — always available on SM80+
    if constexpr (std::is_same_v<compute_t, half> || std::is_same_v<compute_t, __nv_bfloat16>) {
        // ldmatrix addressing for m8n8.x4 with batch offset
        const int tile_idx = lane / 8;
        const int row_in_tile = lane % 8;
        const int m_offset = (tile_idx & 1) * 8;
        const int k_offset = (tile_idx >> 1) * 8;
        
        const uint32_t addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem_A[batch_offset + m_offset + row_in_tile][k_start + k_offset]));
        
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(frag_a[0]), "=r"(frag_a[1]), "=r"(frag_a[2]), "=r"(frag_a[3])
            : "r"(addr)
        );
        return;
    }
    
    // Fallback for non-ldmatrix path or non-16-bit types
    const int groupID = lane / 4;
    const int threadID_in_group = lane % 4;
    const int k_col = k_start + threadID_in_group * 2;
    
    if constexpr (std::is_same_v<compute_t, half>) {
        half2* f = reinterpret_cast<half2*>(frag_a);
        f[0] = *reinterpret_cast<const half2*>(&smem_A[batch_offset + groupID][k_col]);
        f[1] = *reinterpret_cast<const half2*>(&smem_A[batch_offset + groupID + 8][k_col]);
        f[2] = *reinterpret_cast<const half2*>(&smem_A[batch_offset + groupID][k_col + 8]);
        f[3] = *reinterpret_cast<const half2*>(&smem_A[batch_offset + groupID + 8][k_col + 8]);
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        __nv_bfloat162* f = reinterpret_cast<__nv_bfloat162*>(frag_a);
        f[0] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[batch_offset + groupID][k_col]);
        f[1] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[batch_offset + groupID + 8][k_col]);
        f[2] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[batch_offset + groupID][k_col + 8]);
        f[3] = *reinterpret_cast<const __nv_bfloat162*>(&smem_A[batch_offset + groupID + 8][k_col + 8]);
    } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
        half2* f = reinterpret_cast<half2*>(frag_a);
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        __nv_fp8x2_storage_t fp8_01 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[batch_offset + groupID][k_col]);
        __nv_fp8x2_storage_t fp8_23 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[batch_offset + groupID + 8][k_col]);
        __nv_fp8x2_storage_t fp8_45 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[batch_offset + groupID][k_col + 8]);
        __nv_fp8x2_storage_t fp8_67 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&smem_A[batch_offset + groupID + 8][k_col + 8]);
        
        __half2_raw h01 = __nv_cvt_fp8x2_to_halfraw2(fp8_01, __NV_E4M3);
        __half2_raw h23 = __nv_cvt_fp8x2_to_halfraw2(fp8_23, __NV_E4M3);
        __half2_raw h45 = __nv_cvt_fp8x2_to_halfraw2(fp8_45, __NV_E4M3);
        __half2_raw h67 = __nv_cvt_fp8x2_to_halfraw2(fp8_67, __NV_E4M3);
        
        f[0] = *reinterpret_cast<half2*>(&h01);
        f[1] = *reinterpret_cast<half2*>(&h23);
        f[2] = *reinterpret_cast<half2*>(&h45);
        f[3] = *reinterpret_cast<half2*>(&h67);
        #else
        f[0] = __float22half2_rn(make_float2(float(smem_A[batch_offset + groupID][k_col]), float(smem_A[batch_offset + groupID][k_col + 1])));
        f[1] = __float22half2_rn(make_float2(float(smem_A[batch_offset + groupID + 8][k_col]), float(smem_A[batch_offset + groupID + 8][k_col + 1])));
        f[2] = __float22half2_rn(make_float2(float(smem_A[batch_offset + groupID][k_col + 8]), float(smem_A[batch_offset + groupID][k_col + 8 + 1])));
        f[3] = __float22half2_rn(make_float2(float(smem_A[batch_offset + groupID + 8][k_col + 8]), float(smem_A[batch_offset + groupID + 8][k_col + 8 + 1])));
        #endif
    }
}

// -----------------------------------------------------------------------------
// Load FragA for m16n8k32: mirrors mma_m16n8k32 operand layout
// -----------------------------------------------------------------------------
// For FP8 on SM89+: Loads 16 fp8 values packed into a_lo[0..1] and a_hi[0..1]
//   - a_lo[0], a_lo[1], a_hi[0], a_hi[1] form the 4-register A operand
//   - Each register holds 4 packed fp8 values with INTERLEAVED ROW layout
//   - a_lo[2..3] and a_hi[2..3] are unused
//
// CRITICAL: m16n8k32 A operand has INTERLEAVED row layout per PTX ISA:
//   Each 32-bit register packs 4 FP8 bytes as:
//     byte[0] = A[row0, col_base]      (groupID)
//     byte[1] = A[row0, col_base+1]    (groupID)
//     byte[2] = A[row1, col_base]      (groupID+8, interleaved!)
//     byte[3] = A[row1, col_base+1]    (groupID+8, interleaved!)
//
//   Register-to-column mapping (tid = threadID_in_group = lane % 4):
//     a_lo[0]: cols tid*4+0,  tid*4+1   (K offset 0..15 region)
//     a_lo[1]: cols tid*4+2,  tid*4+3   (K offset 0..15 region)
//     a_hi[0]: cols tid*4+16, tid*4+17  (K offset 16..31 region)
//     a_hi[1]: cols tid*4+18, tid*4+19  (K offset 16..31 region)
//
// For F16/BF16 (and FP8 pre-SM89): Loads two k16 slices
//   - a_lo[4]: First k16 slice (k_start to k_start+15)
//   - a_hi[4]: Second k16 slice (k_start+16 to k_start+31)
//
// batch_half: 0 for batches 0-15, 1 for batches 16-31
// -----------------------------------------------------------------------------
template <typename compute_t>
__device__ __forceinline__ void load_frag_a_32(
    uint32_t frag_a_lo[4],
    uint32_t frag_a_hi[4],
    const compute_t smem_A[BATCH_TILE_32][K_STRIDE],
    int k_start, int lane, int batch_half)
{
    const int batch_offset = batch_half * 16;
    
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
        // Native FP8 m16n8k32: pack 16 fp8 into 4 registers with INTERLEAVED layout
        const int groupID = lane / 4;           // 0-7, selects row pair
        const int tid = lane % 4;               // 0-3, selects K-column group
        
        const int row0 = batch_offset + groupID;      // Rows 0-7 (or 16-23)
        const int row1 = batch_offset + groupID + 8;  // Rows 8-15 (or 24-31)
        
        // Column bases for the 4 registers:
        //   a_lo[0]: cols tid*4+0, tid*4+1     (within first K16)
        //   a_lo[1]: cols tid*4+2, tid*4+3     (within first K16)
        //   a_hi[0]: cols tid*4+16, tid*4+17   (within second K16)
        //   a_hi[1]: cols tid*4+18, tid*4+19   (within second K16)
        const int c0  = k_start + tid * 4;        // cols 0,1 for a_lo[0]
        const int c2  = k_start + tid * 4 + 2;    // cols 2,3 for a_lo[1]
        const int c16 = k_start + tid * 4 + 16;   // cols 16,17 for a_hi[0]
        const int c18 = k_start + tid * 4 + 18;   // cols 18,19 for a_hi[1]
        
        // Pack with INTERLEAVED row layout: {row0_col0, row0_col1, row1_col0, row1_col1}
        // This is required by PTX ISA for m16n8k32.e4m3 A operand
        // Optimized: load 2 bytes at a time (uint16) and combine, halving memory ops
        auto pack_interleaved = [&](int col_base) -> uint32_t {
            // Load 2 consecutive FP8 from row0 as uint16 (cols col_base, col_base+1)
            uint32_t lo = *reinterpret_cast<const uint16_t*>(&smem_A[row0][col_base]);
            // Load 2 consecutive FP8 from row1 as uint16 (cols col_base, col_base+1)
            uint32_t hi = *reinterpret_cast<const uint16_t*>(&smem_A[row1][col_base]);
            // Combine: lo in bits[0:15], hi in bits[16:31]
            return lo | (hi << 16);
        };
        
        frag_a_lo[0] = pack_interleaved(c0);   // rows {0,8} × cols {tid*4+0, tid*4+1}
        frag_a_lo[1] = pack_interleaved(c2);   // rows {0,8} × cols {tid*4+2, tid*4+3}
        frag_a_hi[0] = pack_interleaved(c16);  // rows {0,8} × cols {tid*4+16, tid*4+17}
        frag_a_hi[1] = pack_interleaved(c18);  // rows {0,8} × cols {tid*4+18, tid*4+19}
        // frag_a_lo[2..3] and frag_a_hi[2..3] unused in native FP8 path
        return;
    }
#endif
    
    // F16/BF16/pre-SM89 FP8: Load two separate k16 slices
    load_frag_a_16<compute_t>(frag_a_lo, smem_A, k_start, lane, batch_half);
    load_frag_a_16<compute_t>(frag_a_hi, smem_A, k_start + 16, lane, batch_half);
}

// -----------------------------------------------------------------------------
// Load FragA for m16n8k64 (INT4): mirrors mma_m16n8k64_s4/u4 operand layout
// -----------------------------------------------------------------------------
// Loads 64 int4 values (32 per thread distributed across warp) packed into 4 registers.
// Each register holds 8 packed int4 values (4 bits each × 8 = 32 bits).
//
// For INT4 activation data stored as packed nibbles:
//   - smem_A contains int4 values packed 2 per byte (low nibble first)
//   - K_STRIDE should be 64 (or multiple) to align with k64 operations
//
// MMA A operand layout for m16n8k64.s4:
//   - 4 registers per thread
//   - Each register holds 8 int4 values
//   - Thread (lane) distribution follows standard m16n8 pattern
//
// Parameters:
//   frag_a[4]:  Output A fragment (4 × uint32, each with 8 packed int4)
//   smem_A:     Shared memory with packed int4 activations (2 per byte)
//   k_start:    Starting K column (should be multiple of 64)
//   lane:       Thread lane ID (0-31)
//   batch_half: 0 for batches 0-15, 1 for batches 16-31
// -----------------------------------------------------------------------------
template <typename int4_storage_t>  // uint8_t for packed storage, or custom type
__device__ __forceinline__ void load_frag_a_64(
    uint32_t frag_a[4],
    const int4_storage_t smem_A[BATCH_TILE_32][K_STRIDE / 2],  // K/2 because 2 int4 per byte
    int k_start, int lane, int batch_half)
{
    // INT4 MMA fragment load — always available on SM80+
    const int batch_offset = batch_half * 16;
    
    // For m16n8k64 with int4:
    // Each thread loads 32 int4 values = 16 bytes = 4 × uint32
    // Thread mapping follows m16n8 pattern:
    //   groupID (0-7) selects which pair of rows
    //   threadID_in_group (0-3) selects K position within 64-element span
    const int groupID = lane / 4;
    const int threadID_in_group = lane % 4;
    
    // Each thread handles elements at k_start + threadID_in_group * 16 (stride of 16 int4 = 8 bytes)
    // Load 8 int4 per register from two rows (groupID and groupID+8)
    // k position: threadID_in_group * 16 gives offset in int4 units
    //             threadID_in_group * 8 gives offset in byte units (2 int4 per byte)
    const int k_byte_offset = (k_start / 2) + threadID_in_group * 8;
    
    // Load 4 bytes (8 int4) per register, from rows based on groupID
    // Register 0: row=groupID,     k=[0..7]   relative to thread's k position
    // Register 1: row=groupID+8,   k=[0..7]
    // Register 2: row=groupID,     k=[32..39] (next 32 k positions)
    // Register 3: row=groupID+8,   k=[32..39]
    frag_a[0] = *reinterpret_cast<const uint32_t*>(&smem_A[batch_offset + groupID][k_byte_offset]);
    frag_a[1] = *reinterpret_cast<const uint32_t*>(&smem_A[batch_offset + groupID + 8][k_byte_offset]);
    frag_a[2] = *reinterpret_cast<const uint32_t*>(&smem_A[batch_offset + groupID][k_byte_offset + 16]);  // +16 bytes = +32 int4
    frag_a[3] = *reinterpret_cast<const uint32_t*>(&smem_A[batch_offset + groupID + 8][k_byte_offset + 16]);
}

// -----------------------------------------------------------------------------
// TC kernel implementation for batch tile = 32 - takes SMEM as parameters
// batch_tile_idx: Decoded batch tile index (hierarchical or simple grid)
// row_tile_idx: Decoded row tile index (-1 = use blockIdx.y for legacy)
// -----------------------------------------------------------------------------
template <typename block_c_t, typename compute_t, typename act_t, typename output_t>
__device__ void tc32_kernel_impl(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride, int batch_size,
    compute_t smem_A[][K_STRIDE],
    uint8_t* smem_W_flat,
    int batch_offset = 0,
    int batch_tile_idx_in = 0,   // Actual batch tile index (not blockIdx.x)
    int total_batch_tiles = 0,   // Total batch tiles (for bounds check)
    int row_tile_idx = -1)       // -1 = use blockIdx.y (legacy), >= 0 = hierarchical decode
{
    // Hierarchical grid decode: use passed row_tile_idx if >= 0, else blockIdx.y
    const int n_block = (row_tile_idx >= 0) ? row_tile_idx : (int)blockIdx.y;
    const int tid = threadIdx.y * WARP_SIZE_TC + threadIdx.x;
    const int warp_id = tid / WARP_SIZE_TC;
    const int lane = tid % WARP_SIZE_TC;
    
    const int row0 = n_block * N_TILE;
    // For hierarchical grid: batch_offset already contains the base batch
    const int batch0 = batch_offset;
    
    // Bounds check: row must be valid and batch must be in range
    if (row0 >= nrows || batch0 >= batch_size) return;
    
    constexpr int K128_BYTES = sizeof(block_c_t);
    
    const int warp_row_base = row0 + warp_id * 8;
    uint8_t* smem_W = smem_W_flat + warp_id * 8 * K128_BYTES;
    
    float frag_c_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float frag_c_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    const int k_blocks = ncols / K_TILE;
    
    for (int k_blk = 0; k_blk < k_blocks; ++k_blk) {
        load_activations_32<compute_t, act_t>(smem_A, activations, batch0, k_blk * K_TILE, 
                                               y_stride, tid);
        
        // cp.async: SM80+ always has async copy
        load_weights_async_coop<block_c_t>(smem_W_flat, weights, row0, 
                                            k_blk, nrows, k_blocks, tid);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
        
        // Process K/128 tile - use k32 path for FP8 SM89+, k16 path otherwise
        uint32_t frag_a_lo[4], frag_a_hi[4];
        
        // Non-FP8 or pre-SM89: Use standard k16 path
        // First half: k16 slices 0-3
        {
            uint32_t frag_b[8];
            dequant_weights_4x_k16<block_c_t, compute_t, 0>(smem_W, lane, frag_b);
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int k_start = i * MMA_K;
                
                load_frag_a_16<compute_t>(frag_a_lo, smem_A, k_start, lane, 0);
                mma_m16n8k16<compute_t>(frag_a_lo, frag_b + i * 2, frag_c_lo);
                
                load_frag_a_16<compute_t>(frag_a_hi, smem_A, k_start, lane, 1);
                mma_m16n8k16<compute_t>(frag_a_hi, frag_b + i * 2, frag_c_hi);
            }
        }
        
        // Second half: k16 slices 4-7
        {
            uint32_t frag_b[8];
            dequant_weights_4x_k16<block_c_t, compute_t, 1>(smem_W, lane, frag_b);
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int k_start = (4 + i) * MMA_K;
                
                load_frag_a_16<compute_t>(frag_a_lo, smem_A, k_start, lane, 0);
                mma_m16n8k16<compute_t>(frag_a_lo, frag_b + i * 2, frag_c_lo);
                
                load_frag_a_16<compute_t>(frag_a_hi, smem_A, k_start, lane, 1);
                mma_m16n8k16<compute_t>(frag_a_hi, frag_b + i * 2, frag_c_hi);
            }
        }
        
        __syncthreads();
    }
    
    const int groupID = lane / 4;
    const int threadID_in_group = lane % 4;
    const int out_row = warp_row_base + threadID_in_group * 2;
    
    // Batches 0-15
    {
        const int b0 = batch0 + groupID;
        const int b1 = batch0 + groupID + 8;
        
        if (b0 < batch_size) {
            if constexpr (std::is_same_v<output_t, float>) {
                *reinterpret_cast<float2*>(&dst[b0 * dst_stride + out_row]) = make_float2(frag_c_lo[0], frag_c_lo[1]);
            } else if constexpr (std::is_same_v<output_t, half>) {
                *reinterpret_cast<half2*>(&dst[b0 * dst_stride + out_row]) = __floats2half2_rn(frag_c_lo[0], frag_c_lo[1]);
            } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
                *reinterpret_cast<__nv_bfloat162*>(&dst[b0 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c_lo[0], frag_c_lo[1]);
            } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
                dst[b0 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c_lo[0]);
                dst[b0 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c_lo[1]);
            }
        }
        if (b1 < batch_size) {
            if constexpr (std::is_same_v<output_t, float>) {
                *reinterpret_cast<float2*>(&dst[b1 * dst_stride + out_row]) = make_float2(frag_c_lo[2], frag_c_lo[3]);
            } else if constexpr (std::is_same_v<output_t, half>) {
                *reinterpret_cast<half2*>(&dst[b1 * dst_stride + out_row]) = __floats2half2_rn(frag_c_lo[2], frag_c_lo[3]);
            } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
                *reinterpret_cast<__nv_bfloat162*>(&dst[b1 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c_lo[2], frag_c_lo[3]);
            } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
                dst[b1 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c_lo[2]);
                dst[b1 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c_lo[3]);
            }
        }
    }
    
    // Batches 16-31
    {
        const int b0 = batch0 + 16 + groupID;
        const int b1 = batch0 + 16 + groupID + 8;
        
        if (b0 < batch_size) {
            if constexpr (std::is_same_v<output_t, float>) {
                *reinterpret_cast<float2*>(&dst[b0 * dst_stride + out_row]) = make_float2(frag_c_hi[0], frag_c_hi[1]);
            } else if constexpr (std::is_same_v<output_t, half>) {
                *reinterpret_cast<half2*>(&dst[b0 * dst_stride + out_row]) = __floats2half2_rn(frag_c_hi[0], frag_c_hi[1]);
            } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
                *reinterpret_cast<__nv_bfloat162*>(&dst[b0 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c_hi[0], frag_c_hi[1]);
            } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
                dst[b0 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c_hi[0]);
                dst[b0 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c_hi[1]);
            }
        }
        if (b1 < batch_size) {
            if constexpr (std::is_same_v<output_t, float>) {
                *reinterpret_cast<float2*>(&dst[b1 * dst_stride + out_row]) = make_float2(frag_c_hi[2], frag_c_hi[3]);
            } else if constexpr (std::is_same_v<output_t, half>) {
                *reinterpret_cast<half2*>(&dst[b1 * dst_stride + out_row]) = __floats2half2_rn(frag_c_hi[2], frag_c_hi[3]);
            } else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) {
                *reinterpret_cast<__nv_bfloat162*>(&dst[b1 * dst_stride + out_row]) = __floats2bfloat162_rn(frag_c_hi[2], frag_c_hi[3]);
            } else if constexpr (std::is_same_v<output_t, __nv_fp8_e4m3>) {
                dst[b1 * dst_stride + out_row] = from_f32<__nv_fp8_e4m3>(frag_c_hi[2]);
                dst[b1 * dst_stride + out_row + 1] = from_f32<__nv_fp8_e4m3>(frag_c_hi[3]);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Main TC kernel for batch tile = 32 - standalone wrapper
// -----------------------------------------------------------------------------
template <typename block_c_t, typename compute_t, typename act_t, typename output_t>
__device__ void tc32_kernel(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride, int batch_size,
    int batch_offset = 0,
    int tile_offset = 0,
    int local_batch_tiles = 0)
{
    assert(nrows % N_TILE == 0 && "TC kernel requires nrows to be a multiple of 32");
    assert(ncols % K_TILE == 0 && "TC kernel requires ncols to be a multiple of 128");
    
    constexpr int K128_BYTES = sizeof(block_c_t);
    
    __shared__ compute_t smem_A[BATCH_TILE_32][K_STRIDE];
    __shared__ uint8_t smem_W_flat[N_TILE * K128_BYTES];
    
    tc32_kernel_impl<block_c_t, compute_t, act_t, output_t>(
        weights, activations, dst, ncols, nrows, y_stride, dst_stride, batch_size,
        smem_A, smem_W_flat, batch_offset, tile_offset, local_batch_tiles);
}

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
// REMAINDER_BATCH: compile-time constant 0-15 (R = batch_size % 16)
//   - R=0: Only tc32+tc16 tiles (no tcR remainder)
//   - R=1-15: tc32+tc16 tiles + one tcR remainder tile
//
// SMEM OPTIMIZATION: Declares unified SMEM once (tc32 size = 32×136) and passes
// to _impl functions to avoid NVCC summing SMEM from multiple __shared__ declarations.
// tc16 and tcN use the first 16 rows of smem_A, tc32 uses all 32.
// =============================================================================

namespace tc32 {

using namespace s32_tc;

// Compile-time dispatch: each unified kernel has tc32 + optional tc16 + one specific tcN (0-15)
// HIERARCHICAL GRID: z = row_group + batch_group × num_row_groups (rows inner, batches outer)
template <typename block_c_t, typename compute_t, typename act_t, typename output_t, int REMAINDER_BATCH>
__device__ void dispatch_tc32_tc16_tcN(
    const block_c_t* __restrict__ weights,
    const act_t* __restrict__ activations,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride,
    int batch_size, int row_groups)
{
    static_assert(REMAINDER_BATCH >= 0 && REMAINDER_BATCH <= 15, 
                  "REMAINDER_BATCH must be 0-15");
    
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
    const int total_batch_tiles = tc32_tiles + has_tc16 + (REMAINDER_BATCH > 0 ? 1 : 0);
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
        // tcR remainder tile (only if REMAINDER_BATCH > 0)
        if constexpr (REMAINDER_BATCH > 0) {
            const int batch_start = tc32_tiles * 32 + has_tc16 * 16;
            sN_tc::tcN_kernel_impl<block_c_t, compute_t, act_t, output_t, REMAINDER_BATCH>(
                weights, activations, dst, ncols, nrows, y_stride, dst_stride,
                smem_A, smem_W_flat, batch_start, row_tile_idx);
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
// REMAINDER_BATCH: compile-time constant 0-15 (R = batch_size % 16)
//   - R=0: Only tc32+tc16 tiles (e.g., batch 48 = tc32 + tc16)
//   - R=1-15: tc32+tc16 tiles + tcR (e.g., batch 49 = tc32 + tc16 + tc1)
//
// HIERARCHICAL GRID (kernel_cache_design.md):
//   x = batch tiles (L1 scope) - consecutive blocks share weights in L1
//   y = row tiles (L2 scope) - wave fills all SMs sharing activations
//   z = wave index: row_group + batch_group × num_row_groups
// row_groups parameter enables decode: row_group = z % row_groups
// =============================================================================
template <int qk, int qi, typename block_q_t, int vdr,
          typename act_t, typename output_t = float, int REMAINDER_BATCH = 0>
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
    static_assert(REMAINDER_BATCH >= 0 && REMAINDER_BATCH <= 15, 
                  "REMAINDER_BATCH must be 0-15");
    
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
    
    // Compile-time dispatch - hierarchical grid decode computed internally
    tc32::dispatch_tc32_tc16_tcN<block_c_t, compute_t, act_t, output_t, REMAINDER_BATCH>(
        weights, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batch_size, row_groups);
}

