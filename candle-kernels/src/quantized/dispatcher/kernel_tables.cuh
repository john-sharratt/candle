// The kernel variant declarations, the kernel lookup structures, and the
// table-driven launch helpers.
//
// `DECLARE_KERNEL_VARIANTS` forward-declares the kernel variants for each
// quantized type. `kernel_set_t` holds the kernel lookup structures, and
// `launch_kernel_by_type` launches through them with a table-driven helper.

// =============================================================================
// Forward declarations of all kernel variants
// =============================================================================

// Macro to declare all batch-specialized variants for a kernel
// Note: These are __global__ CUDA kernels, must be launched with cudaLaunchKernel
// Simplified kernel signatures: no extra tiling params for unified kernels
#define DECLARE_KERNEL_VARIANTS(name) \
    extern "C" __global__ void name##_s1( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_s2( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_s3( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_s4( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_s5( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_s6( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_s7( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_s8( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_0( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_r( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_0( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_r( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_s2_iter2( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_s2_iter3( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_s2_iter4( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_s2_iter5( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_s2_iter6( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_s2_iter7( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_s2_iter8( \
        const void*, const void*, void*, \
        int, int, int, int, int); \
    extern "C" __global__ void name##_s3_iter3( \
        const void*, const void*, void*, \
        int, int, int, int, int)

// Q4_0 variants
DECLARE_KERNEL_VARIANTS(q4_0_q8);
DECLARE_KERNEL_VARIANTS(q4_0_f16);
DECLARE_KERNEL_VARIANTS(q4_0_bf16);
DECLARE_KERNEL_VARIANTS(q4_0_f32);

// Q4_1 variants
DECLARE_KERNEL_VARIANTS(q4_1_q8);
DECLARE_KERNEL_VARIANTS(q4_1_f16);
DECLARE_KERNEL_VARIANTS(q4_1_bf16);
DECLARE_KERNEL_VARIANTS(q4_1_f32);

// Q5_0 variants
DECLARE_KERNEL_VARIANTS(q5_0_q8);
DECLARE_KERNEL_VARIANTS(q5_0_f16);
DECLARE_KERNEL_VARIANTS(q5_0_bf16);
DECLARE_KERNEL_VARIANTS(q5_0_f32);

// Q5_1 variants
DECLARE_KERNEL_VARIANTS(q5_1_q8);
DECLARE_KERNEL_VARIANTS(q5_1_f16);
DECLARE_KERNEL_VARIANTS(q5_1_bf16);
DECLARE_KERNEL_VARIANTS(q5_1_f32);

// Q8_0 variants
DECLARE_KERNEL_VARIANTS(q8_0_q8);
DECLARE_KERNEL_VARIANTS(q8_0_f16);
DECLARE_KERNEL_VARIANTS(q8_0_bf16);
DECLARE_KERNEL_VARIANTS(q8_0_f32);

// Q2_K variants
DECLARE_KERNEL_VARIANTS(q2_K_q8);
DECLARE_KERNEL_VARIANTS(q2_K_f16);
DECLARE_KERNEL_VARIANTS(q2_K_bf16);
DECLARE_KERNEL_VARIANTS(q2_K_f32);

// Q3_K variants
DECLARE_KERNEL_VARIANTS(q3_k_q8);
DECLARE_KERNEL_VARIANTS(q3_k_f16);
DECLARE_KERNEL_VARIANTS(q3_k_bf16);
DECLARE_KERNEL_VARIANTS(q3_k_f32);

// Q4_K variants
DECLARE_KERNEL_VARIANTS(q4_k_q8);
DECLARE_KERNEL_VARIANTS(q4_k_f16);
DECLARE_KERNEL_VARIANTS(q4_k_bf16);
DECLARE_KERNEL_VARIANTS(q4_k_f32);

// Q5_K variants
DECLARE_KERNEL_VARIANTS(q5_k_q8);
DECLARE_KERNEL_VARIANTS(q5_k_f16);
DECLARE_KERNEL_VARIANTS(q5_k_bf16);
DECLARE_KERNEL_VARIANTS(q5_k_f32);

// Q6_K variants
DECLARE_KERNEL_VARIANTS(q6_k_q8);
DECLARE_KERNEL_VARIANTS(q6_k_f16);
DECLARE_KERNEL_VARIANTS(q6_k_bf16);
DECLARE_KERNEL_VARIANTS(q6_k_f32);

// Q8_1 variants
DECLARE_KERNEL_VARIANTS(q8_1_f16);
DECLARE_KERNEL_VARIANTS(q8_1_bf16);
DECLARE_KERNEL_VARIANTS(q8_1_f32);

// Q8_K variants
DECLARE_KERNEL_VARIANTS(q8_k_f16);
DECLARE_KERNEL_VARIANTS(q8_k_bf16);
DECLARE_KERNEL_VARIANTS(q8_k_f32);

// Q_AWQ variants
DECLARE_KERNEL_VARIANTS(q_awq_f16);
DECLARE_KERNEL_VARIANTS(q_awq_bf16);
DECLARE_KERNEL_VARIANTS(q_awq_f32);

// Q_AWQ_G64 variants
DECLARE_KERNEL_VARIANTS(q_awq_g64_f16);
DECLARE_KERNEL_VARIANTS(q_awq_g64_bf16);
DECLARE_KERNEL_VARIANTS(q_awq_g64_f32);

#undef DECLARE_KERNEL_VARIANTS

// =============================================================================
// Kernel lookup structures
// =============================================================================

// Standard kernels: batch-specialized variants per qtype/ytype/use_tc
// These are stored as void* because they're __global__ function addresses
// that need to be launched with cudaLaunchKernel, not called directly.
// Greedy decomposition guarantees full batch utilization for all kernels.
//
// Tensor core variant:
//   - tc16_0 / tc16_r: BATCH_TILE=16 with grid.y for batch tiling, TC path for
//     batch 1-31 — without / with a remainder tile
//   - tc32_0 / tc32_r: BATCH_TILE=32 with greedy decomposition, TC path for
//     batch 32+ — without / with a remainder tile
//
// Iterator variants (_iter suffix = internal batch loop for L2 weight reuse):
//   - s2_iter2-8: BATCH_TILE=2, NUM_ITERS=2-8, processes 4/6/8/10/12/14/16 batches
//   - s3_iter3: BATCH_TILE=3, NUM_ITERS=3, processes 9 batches
//     Uses ~85 registers, high occupancy (~100%), good latency hiding
struct kernel_set_t {
    void* s1;         // BATCH_TILE=1, single-context decode
    void* s2;         // BATCH_TILE=2, pair batch
    void* s3;         // BATCH_TILE=3, triple batch
    void* s4;         // BATCH_TILE=4, quad batch
    void* s5;         // BATCH_TILE=5, penta batch
    void* s6;         // BATCH_TILE=6, hexa batch
    void* s7;         // BATCH_TILE=7, septa batch
    void* s8;         // BATCH_TILE=8, octet batch
    // TC16 kernels. R = batch_size % 16, taken by the kernel from batch_size.
    // tc16_0: pure tc16 grid.y tiling (R = 0); tc16_r: tc16 + one tcR
    // remainder tile (R = 1-15).
    void* tc16_0;
    void* tc16_r;
    // TC32 kernels - greedy tc32 + tc16 decomposition computed internally.
    // tc32_0: no remainder (R = 0); tc32_r: plus one tcR tile (R = 1-15).
    void* tc32_0;
    void* tc32_r;
    // Iterator kernels
    void* s2_iter2;   // BATCH_TILE=2, NUM_ITERS=2 (4 batches)
    void* s2_iter3;   // BATCH_TILE=2, NUM_ITERS=3 (6 batches)
    void* s2_iter4;   // BATCH_TILE=2, NUM_ITERS=4 (8 batches)
    void* s2_iter5;   // BATCH_TILE=2, NUM_ITERS=5 (10 batches)
    void* s2_iter6;   // BATCH_TILE=2, NUM_ITERS=6 (12 batches)
    void* s2_iter7;   // BATCH_TILE=2, NUM_ITERS=7 (14 batches)
    void* s2_iter8;   // BATCH_TILE=2, NUM_ITERS=8 (16 batches)
    void* s3_iter3;   // BATCH_TILE=3, NUM_ITERS=3 (9 batches)
};

// GEMV kernels (s1-s8 register-only path, all use CUDA cores)
// TC kernels: tc16_0 / tc16_r + tc32_0 / tc32_r
#define KERNEL_SET(name) { \
    (void*)name##_s1, (void*)name##_s2, (void*)name##_s3, (void*)name##_s4, \
    (void*)name##_s5, (void*)name##_s6, (void*)name##_s7, (void*)name##_s8, \
    (void*)name##_tc16_0, (void*)name##_tc16_r, \
    (void*)name##_tc32_0, (void*)name##_tc32_r, \
    (void*)name##_s2_iter2, \
    (void*)name##_s2_iter3, (void*)name##_s2_iter4, \
    (void*)name##_s2_iter5, (void*)name##_s2_iter6, \
    (void*)name##_s2_iter7, (void*)name##_s2_iter8, \
    (void*)name##_s3_iter3 \
}

// Same as KERNEL_SET
#define KERNEL_SET_TC(name) KERNEL_SET(name)

// =============================================================================
// Kernel launch helpers
// =============================================================================
// Use cudaLaunchKernel to properly launch __global__ kernels with grid/block config

// Launch a standard (non-chunked) kernel
// Block dimensions: 32 threads per warp × 4 warps = 128 threads
// The kernel uses threadIdx.y as warp index and threadIdx.x as lane index
// grid_layout parameter controls grid dimension ordering (for compute_grid only)
// The kernel itself has grid_layout as a compile-time template parameter
// smem_size: shared memory bytes to allocate (0 for register-only kernels)
inline cudaError_t launch_kernel(
    void* kernel_fn,
    int batch_tile,
    const void* vx, const void* vy, void* dst,
    int ncols_x, int nrows_x, int nrows_y, int nrows_dst, int batch_size,
    int grid_layout,
    size_t smem_size,
    cudaStream_t stream
) {
    dim3 grid = compute_grid(nrows_x, batch_size, batch_tile, grid_layout);
    dim3 block(WARP_SIZE, 4, 1);  // 32 threads/warp × 4 warps = 128 threads
    
    // For kernels requiring dynamic shared memory,
    // set the max dynamic shared memory attribute before launch
    if (smem_size > 0) {
        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    
    void* args[] = { 
        (void*)&vx, (void*)&vy, (void*)&dst,
        (void*)&ncols_x, (void*)&nrows_x, (void*)&nrows_y, 
        (void*)&nrows_dst, (void*)&batch_size
    };
    
    return cudaLaunchKernel(kernel_fn, grid, block, args, smem_size, stream);
}

// Launch an iterator kernel (internal batch loop, L2 weight reuse)
// Grid: (row_blocks, batch_tiles) where each block processes BATCHES_PER_BLOCK batches
// For s2_iter4: BATCHES_PER_BLOCK = 8, so batch_tiles = ceil(total_batches / 8)
inline cudaError_t launch_kernel_iter(
    void* kernel_fn, 
    int batches_per_block,  // BATCH_TILE * NUM_ITERS (e.g., 8 for s2_iter4)
    const void* vx, const void* vy, void* dst,
    int ncols_x, int nrows_x, int nrows_y, int nrows_dst, int total_batches,
    cudaStream_t stream
) {
    // Grid: row blocks × batch tiles
    int row_blocks = (nrows_x + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int batch_tiles = (total_batches + batches_per_block - 1) / batches_per_block;
    dim3 grid(row_blocks, batch_tiles, 1);
    dim3 block(WARP_SIZE, 4, 1);  // 32 threads/warp × 4 warps = 128 threads
    
    void* args[] = { 
        (void*)&vx, (void*)&vy, (void*)&dst,
        (void*)&ncols_x, (void*)&nrows_x, (void*)&nrows_y, 
        (void*)&nrows_dst, (void*)&total_batches
    };

    return cudaLaunchKernel(kernel_fn, grid, block, args, 0, stream);
}

// TC kernel row tile size (4 warps × 8 rows/warp = 32 rows per block)
constexpr int TC_ROWS_PER_BLOCK = 32;

// Launch a tensor core kernel with grid.y batch tiling (single launch for all batches)
// Grid: (row_blocks, batch_tiles) where each block processes BATCH_TILE=16 batches
// batch_size is passed to kernel for bounds checking (last tile may be partial)
inline cudaError_t launch_kernel_tc(
    void* kernel_fn, 
    int batch_tile,  // BATCH_TILE (16 for s16_tc)
    const void* vx, const void* vy, void* dst,
    int ncols_x, int nrows_x, int nrows_y, int nrows_dst, int batch_size,
    cudaStream_t stream
) {
    // Grid: row blocks × batch tiles
    // TC kernel uses N_TILE=32 (4 warps × 8 rows), not ROWS_PER_BLOCK=16
    int row_blocks = (nrows_x + TC_ROWS_PER_BLOCK - 1) / TC_ROWS_PER_BLOCK;
    int batch_tiles = (batch_size + batch_tile - 1) / batch_tile;
    dim3 grid(row_blocks, batch_tiles, 1);
    dim3 block(WARP_SIZE, 4, 1);  // 32 threads/warp × 4 warps = 128 threads
    
    void* args[] = { 
        (void*)&vx, (void*)&vy, (void*)&dst,
        (void*)&ncols_x, (void*)&nrows_x, (void*)&nrows_y, 
        (void*)&nrows_dst, (void*)&batch_size
    };

    return cudaLaunchKernel(kernel_fn, grid, block, args, 0, stream);
}

// =============================================================================
// Launch TC16 kernel with HIERARCHICAL GRID for L1/L2 cache optimization
// =============================================================================
// Same hierarchical design as launch_tc32 (see kernel_cache_design.md):
//
// GRID DIMENSIONS:
//   x = batch tiles (L1 scope): min(batch_tiles, blocks_per_sm)
//   y = row tiles (L2 scope): fills wave with L2 capacity constraint
//   z = wave index: row_group + batch_group × num_row_groups
//
// L2 CAPACITY CONSTRAINT:
//   y_from_l2 = l2_usable / (gridDim.x × TILE_BATCH × K × sizeof(half))
//   Ensures activation working set fits in L2 for maximum reuse
//
// Parameters:
//   has_remainder: batch_size % 16 != 0 — picks tc16_r over tc16_0; the
//     kernel derives R itself from batch_size
//   batch_size: total batches to process
inline cudaError_t launch_tc16(
    const kernel_set_t& kset,
    bool has_remainder,
    const void* vx, const void* vy, void* dst,
    int ncols_x, int nrows_x, int nrows_y, int nrows_dst, int batch_size,
    cudaStream_t stream
) {
    void* kernel_fn = has_remainder ? kset.tc16_r : kset.tc16_0;

    // Compute total batch tiles
    const int tc16_tiles = batch_size / 16;
    const int total_batch_tiles = tc16_tiles + (has_remainder ? 1 : 0);
    const int row_tiles = (nrows_x + TC_ROWS_PER_BLOCK - 1) / TC_ROWS_PER_BLOCK;
    
    // =========================================================================
    // HIERARCHICAL GRID SIZING with L2 CAPACITY CONSTRAINT
    // =========================================================================
    // Use compute_grid_config for unified grid sizing logic
    // TILE_BATCH=16 for tc16 kernels
    GridConfig cfg = compute_grid_config(row_tiles, total_batch_tiles, 16, ncols_x);

    dim3 block(WARP_SIZE, 4, 1);  // 32 threads/warp × 4 warps = 128 threads

    // Pass hierarchical params to kernel for decode
    void* args[] = {
        (void*)&vx, (void*)&vy, (void*)&dst,
        (void*)&ncols_x, (void*)&nrows_x, (void*)&nrows_y,
        (void*)&nrows_dst, (void*)&batch_size,
        (void*)&cfg.row_groups  // For z-decode: row_group = z % row_groups
    };

    return cudaLaunchKernel(kernel_fn, cfg.grid, block, args, 0, stream);
}

// Launch TC32 kernel with HIERARCHICAL GRID for L1/L2 cache optimization
// =============================================================================
// Implements the Grid-as-Cache-Hierarchy design from kernel_cache_design.md:
//
// GRID DIMENSIONS:
//   x = batch tiles (L1 scope): Same SM gets consecutive x blocks, sharing weights in L1
//   y = row tiles within wave (L2 scope): Different SMs share activations in L2
//   z = wave index: Sequential waves for row_groups × batch_groups
//
// Z-STRIDE ORDERING (rows inner, batches outer):
//   z = row_group + batch_group × num_row_groups
//   Consecutive z values process SAME activations (different rows) → L2 persistence
//
// L2 CAPACITY CONSTRAINT:
//   y_from_l2 = l2_usable / (gridDim.x × TILE_BATCH × K × sizeof(half))
//   Ensures activation working set fits in L2 for maximum reuse
//
// The kernel decodes: row_group = z % num_row_groups, batch_group = z / num_row_groups
// =============================================================================

inline cudaError_t launch_tc32(
    const kernel_set_t& kset,
    bool has_remainder,  // batch_size % 16 != 0 — picks tc32_r over tc32_0
    const void* vx, const void* vy, void* dst,
    int ncols_x, int nrows_x, int nrows_y, int nrows_dst, int batch_size,
    cudaStream_t stream
) {
    void* kernel_fn = has_remainder ? kset.tc32_r : kset.tc32_0;

    // Compute total batch tiles (greedy decomposition)
    const int tc32_tiles = batch_size / 32;
    const int remainder_32 = batch_size % 32;
    const int has_tc16 = (remainder_32 >= 16) ? 1 : 0;
    const int has_tcR = has_remainder ? 1 : 0;
    const int total_batch_tiles = tc32_tiles + has_tc16 + has_tcR;
    
    const int row_tiles = (nrows_x + TC_ROWS_PER_BLOCK - 1) / TC_ROWS_PER_BLOCK;
    
    // =========================================================================
    // HIERARCHICAL GRID SIZING with L2 CAPACITY CONSTRAINT
    // =========================================================================
    // Use compute_grid_config for unified grid sizing logic
    // TILE_BATCH=32 for tc32 kernels
    GridConfig cfg = compute_grid_config(row_tiles, total_batch_tiles, 32, ncols_x);
    
    dim3 block(WARP_SIZE, 4, 1);  // 32 threads/warp × 4 warps = 128 threads
    
    // Pass hierarchical params to kernel for decode
    void* args[] = { 
        (void*)&vx, (void*)&vy, (void*)&dst,
        (void*)&ncols_x, (void*)&nrows_x, (void*)&nrows_y, 
        (void*)&nrows_dst, (void*)&batch_size,
        (void*)&cfg.row_groups  // For z-decode: row_group = z % row_groups
    };

    return cudaLaunchKernel(kernel_fn, cfg.grid, block, args, 0, stream);
}


// =============================================================================
// TABLE-DRIVEN KERNEL LAUNCH HELPER
// =============================================================================
// Launches a GEMV kernel based on kernel_type_t from dispatch_table.cuh
// TC kernels (tc16, tc32) are handled by separate launch functions

inline cudaError_t launch_kernel_by_type(
    const kernel_set_t& kset,
    kernel_type_t kt,
    int batches,
    const void* vx, const void* vy, void* dst,
    int ncols_x, int nrows_x, int nrows_y, int nrows_dst,
    cudaStream_t stream
) {
    // Simple sN kernels (DRAM-friendly, N batches per block)
    switch (kt) {
        case K_S1: return launch_kernel(kset.s1, 1, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST, 0, stream);
        case K_S2: return launch_kernel(kset.s2, 2, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST, 0, stream);
        case K_S3: return launch_kernel(kset.s3, 3, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST, 0, stream);
        case K_S4: return launch_kernel(kset.s4, 4, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST, 0, stream);
        case K_S5: return launch_kernel(kset.s5, 5, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST, 0, stream);
        case K_S6: return launch_kernel(kset.s6, 6, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST, 0, stream);
        case K_S7: return launch_kernel(kset.s7, 7, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST, 0, stream);
        case K_S8: return launch_kernel(kset.s8, 8, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST, 0, stream);

        // Iter kernels (L2-friendly, high occupancy with internal loop)
        case K_S2_ITER2: return launch_kernel_iter(kset.s2_iter2, 4, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, stream);
        case K_S2_ITER3: return launch_kernel_iter(kset.s2_iter3, 6, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, stream);
        case K_S2_ITER4: return launch_kernel_iter(kset.s2_iter4, 8, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, stream);
        case K_S2_ITER5: return launch_kernel_iter(kset.s2_iter5, 10, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, stream);
        case K_S2_ITER6: return launch_kernel_iter(kset.s2_iter6, 12, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, stream);
        case K_S2_ITER7: return launch_kernel_iter(kset.s2_iter7, 14, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, stream);
        case K_S2_ITER8: return launch_kernel_iter(kset.s2_iter8, 16, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, stream);

        case K_S3_ITER3: return launch_kernel_iter(kset.s3_iter3, 9, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, stream);

        // Note: TC kernels (K_TC16_*, K_TC32_*) are launched
        // via launch_tc16() and launch_tc32() in the TC path,
        // not through this function. They shouldn't appear here.

        case K_NONE:
        default:
            return cudaSuccess;
    }
}
