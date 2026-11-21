// The batched quantized matrix-vector kernels and their configuration
// constants.
//
// Holds the forward declaration of `dequant_q8_0_for_4x_mma_k16_runtime`, the
// grid layout modes and kernel configuration constants, the cross-warp
// reduction `phase_reduce_to_smem` with `final_output`, and the
// `quantized_matmul_register_path`, `quantized_matmul`, `quantized_matmul_iter`
// and `final_output_tc` kernels.

// =============================================================================
// FORWARD DECLARATIONS - Optimized standalone dequant functions
// =============================================================================
// These are defined in their respective loader/*.cuh files but need forward
// declaration here since kernel.cuh is included before the specific loaders.
// The impl/*.cu files include both kernel.cuh and loader/*.cuh.
template <typename compute_t, int half_idx>
__device__ __forceinline__ void dequant_q8_0_for_4x_mma_k16_runtime(
    const uint8_t* __restrict__ smem_rows,
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
    //
    // A real loop. Each iteration is a whole K walk of its own (the tile loop
    // below is not unrolled), so iterations never overlap one another and
    // unrolling them bought only folded offsets — while pasting the K walk,
    // the partial tile and the reduce NUM_ITERS times into the kernel.
    // =========================================================================
    #pragma unroll 1
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
    // (a real loop, as phase 1)
    // =========================================================================
    #pragma unroll 1
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
