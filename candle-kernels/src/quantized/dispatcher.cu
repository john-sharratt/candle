// =============================================================================
// QUANTIZED MATMUL DISPATCHER WITH GREEDY BATCH DECOMPOSITION
// =============================================================================
// Provides a single extern "C" entry point (run_quantized_matmul) that dispatches
// to the appropriate typed kernel based on qtype and ytype parameters. 
//
// qtype: 0-9 (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
// ytype: 0-2 (F16, BF16, F32)
//   Note: F32 (ytype=2) only supported for Q4_K currently. Used for validation.
//
// Tensor core usage is determined automatically from SM version:
//   - SM80+ (Ampere/Ada): TC enabled for F16/BF16/F32
//
// Dispatch paths (SM80+):
//   - TC16 path (batch 1-31): tc16_N kernels with internal tiling
//   - TC32 path (batch 32+):  tc32_N kernels with greedy decomposition
//
// Tensor cores are used for ALL batch sizes >= 1, including batch 1-2.
// Benchmarking found the TC kernels faster than the CUDA-core GEMV (s1..s8)
// kernels in every measured case — even single-token decode — so there is no
// CUDA-core fast path on TC-capable hardware. The s1..s8 GEMV kernels remain
// only as the fallback for pre-SM80 GPUs without tensor cores.
//
// At small batch a single weight's TC grid under-fills the SMs. For MoE the
// grouped path (run_grouped_quantized_matmul) fixes this by running all experts
// in one launch so their tiles together fill the machine; the per-weight path
// here just accepts the lower occupancy.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cstdio>
#include <cstring>

// Dispatch table for deterministic kernel selection and reporting
#include "block_compact.cuh"  // qtype_to_matmul_kernel_index + QTYPE_* constants
#include "dispatch_table.cuh"
#include "matmul_status.cuh"  // QMM_* launcher status codes

// =============================================================================
// L2 CACHE FLUSH UTILITY
// =============================================================================
// Provides a fast way to evict L2 cache contents for realistic benchmarking.
// This simulates real LLaMA inference where different matrices alternate.

// Static buffer for L2 flush (allocated once, reused)
static __device__ char* g_flush_buffer = nullptr;
static size_t g_flush_buffer_size = 0;

// Simple kernel to read through buffer and evict L2 contents
__global__ void flush_l2_kernel(const char* __restrict__ buffer, size_t size, volatile int* dummy) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    int sum = 0;
    for (size_t i = idx; i < size; i += stride) {
        sum += buffer[i];
    }
    // Prevent optimization - write to global memory
    if (idx == 0) {
        *dummy = sum;
    }
}

// Host function to flush L2 cache
// buffer: Pre-allocated device buffer larger than L2 cache
// size: Size of buffer in bytes (should be >= 2x L2 cache size)
extern "C" void flush_l2_cache(const void* buffer, size_t size) {
    // Use enough threads to saturate memory bandwidth
    const int threads = 256;
    const int blocks = (size + threads * 16 - 1) / (threads * 16);  // 16 bytes per thread min
    
    // Allocate a tiny dummy output to prevent optimization
    static int* d_dummy = nullptr;
    if (d_dummy == nullptr) {
        cudaMalloc(&d_dummy, sizeof(int));
    }
    
    flush_l2_kernel<<<blocks, threads>>>(static_cast<const char*>(buffer), size, d_dummy);
    cudaDeviceSynchronize();
}

// =============================================================================
// KERNEL LAUNCH CONFIGURATION
// =============================================================================
// Grid dimensions: (row_blocks, batch_tiles)
//   - row_blocks = ceil(nrows_x / rows_per_cuda_block)
//   - batch_tiles = ceil(batch_size / BATCH_TILE)
// Block dimensions: num_threads = 4 warps × 32 = 128 threads
// 
// rows_per_cuda_block = 16 (constant, matches kernel.cuh)
// BATCH_TILE varies by kernel variant: 1, 8, or 64

constexpr int ROWS_PER_BLOCK = 16;
constexpr int THREADS_PER_BLOCK = 128;
constexpr int WARP_SIZE = 32;  // threads per warp

// =============================================================================
// CACHED DEVICE PROPERTIES (via shared device_caps.cuh)
// =============================================================================
// Populated once per device via get_device_caps(), then reused everywhere.
// This avoids repeated cudaGetDeviceProperties() calls which are expensive.
#include "device_caps.cuh"

static int g_cached_blocks_per_sm = 8;      // Max concurrent blocks per SM (occupancy)

// =============================================================================
// GRID-AS-CACHE-HIERARCHY CONFIGURATION
// =============================================================================
// Implements hierarchical L1/L2 cache optimization for TC kernels:
//   gridDim.x = batch tiles (L1 scope): Same-SM blocks share weights in L1
//   gridDim.y = row tiles (L2 scope): All blocks share activations in L2
//   gridDim.z = wave overflow (sequential processing)
//
// L2 capacity constraint ensures activation working set fits in L2:
//   act_per_x_group = gridDim.x × TILE_BATCH × K × sizeof(half)
//   y_from_l2 = (L2_size × 0.70) / act_per_x_group
//
// Z-stride encoding (rows inner for L2 activation persistence):
//   z = row_group + batch_group × num_row_groups

struct GridConfig {
    dim3 grid;          // Final grid dimensions (x, y, z)
    int row_groups;     // Number of row groups for z-decode
    int batch_groups;   // Number of batch groups for z-decode
    int wave_size;      // Blocks per wave (SM count × blocks/SM)
};

// Compute hierarchical grid configuration with L2 capacity constraint
// Parameters:
//   row_tiles: Total row tiles to process
//   batch_tiles: Total batch tiles to process
//   tile_batch: Batch size per tile (16 or 32)
//   K: Hidden dimension (ncols_x)
__host__ inline GridConfig compute_grid_config(
    int row_tiles, int batch_tiles, int tile_batch, int K
) {
    GridConfig cfg;
    const auto& caps = get_device_caps();
    cfg.wave_size = caps.sm_count * g_cached_blocks_per_sm;

    // L2-usable size (90% of L2)
    const size_t l2_usable = (caps.l2_cache_size * 90) / 100;
    const size_t tile_size = (size_t)tile_batch * K * sizeof(half);
    // Total number of tiles that fit in L2
    const double A = (tile_size > 0) ? (double)l2_usable / (double)tile_size : 1.0;

    // Square-root-based sizing for grid.x and grid.y
    int grid_x = (int)ceil(sqrt(A));
    grid_x = max(1, min(grid_x, batch_tiles));
    // Cap grid.x by occupancy
    grid_x = min(grid_x, cfg.wave_size);

    int grid_y = (int)max(1.0, floor(A / grid_x));
    grid_y = min(grid_y, row_tiles);
    // Cap grid.y by occupancy
    if (grid_x > 0) grid_y = min(grid_y, cfg.wave_size / grid_x);

    // Row and batch groups for z-dimension
    cfg.row_groups = (row_tiles + grid_y - 1) / grid_y;
    cfg.batch_groups = (batch_tiles + grid_x - 1) / grid_x;

    // grid.z: Total waves (rows inner, batches outer for L2 persistence)
    const int grid_z = cfg.row_groups * cfg.batch_groups;

    cfg.grid = dim3(grid_x, grid_y, grid_z);
    return cfg;
}

// Grid layout modes for L2 cache optimization:
//   GRID_LAYOUT_ROW_FAST (0): blockIdx.x=row, blockIdx.y=batch
//     - Y (activations) stay in L2 across row blocks
//     - Best when batch_tiles=1 (most decode scenarios)
//   GRID_LAYOUT_BATCH_FAST (1): blockIdx.x=batch, blockIdx.y=row  
//     - X (weights) stay in L2 across batch tiles
//     - Best when batch_tiles>1 and weights don't fit in L2
constexpr int GRID_LAYOUT_ROW_FAST = 0;
constexpr int GRID_LAYOUT_BATCH_FAST = 1;

// Helper to compute grid dimensions based on layout
__host__ inline dim3 compute_grid(int nrows_x, int batch_size, int batch_tile, int grid_layout) {
    int row_blocks = (nrows_x + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int batch_tiles = (batch_size + batch_tile - 1) / batch_tile;
    if (grid_layout == GRID_LAYOUT_BATCH_FAST) {
        return dim3(batch_tiles, row_blocks);  // batch varies fastest
    } else {
        return dim3(row_blocks, batch_tiles);  // row varies fastest (default)
    }
}

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
    extern "C" __global__ void name##_tc16_1( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_2( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_3( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_4( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_5( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_6( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_7( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_8( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_9( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_10( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_11( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_12( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_13( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_14( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc16_15( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_0( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_1( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_2( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_3( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_4( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_5( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_6( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_7( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_8( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_9( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_10( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_11( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_12( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_13( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_14( \
        const void*, const void*, void*, \
        int, int, int, int, int, int); \
    extern "C" __global__ void name##_tc32_15( \
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
//   - tc16_N: BATCH_TILE=16 with grid.y for batch tiling, TC path for batch 3-31
//   - tc32_N: BATCH_TILE=32 with greedy decomposition, TC path for batch 32+
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
    // TC16 kernels - compile-time dispatch for tc16+tcN (0-15)
    // R=0: pure tc16 grid.y tiling
    // R=1-15: tc16 + tcR remainder
    void* tc16_0;
    void* tc16_1;
    void* tc16_2;
    void* tc16_3;
    void* tc16_4;
    void* tc16_5;
    void* tc16_6;
    void* tc16_7;
    void* tc16_8;
    void* tc16_9;
    void* tc16_10;
    void* tc16_11;
    void* tc16_12;
    void* tc16_13;
    void* tc16_14;
    void* tc16_15;
    // TC32 kernels - greedy dispatch for tc32+tc16+tcN (0-15)
    // R = batch_size % 16, greedy decomposition computed internally
    // R=0: tc32 + tc16 (no remainder)
    // R=1-15: tc32 + tc16 + tcR
    void* tc32_0;
    void* tc32_1;
    void* tc32_2;
    void* tc32_3;
    void* tc32_4;
    void* tc32_5;
    void* tc32_6;
    void* tc32_7;
    void* tc32_8;
    void* tc32_9;
    void* tc32_10;
    void* tc32_11;
    void* tc32_12;
    void* tc32_13;
    void* tc32_14;
    void* tc32_15;
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
// TC kernels: tc16_(0-15) + tc32_(0-15)
#define KERNEL_SET(name) { \
    (void*)name##_s1, (void*)name##_s2, (void*)name##_s3, (void*)name##_s4, \
    (void*)name##_s5, (void*)name##_s6, (void*)name##_s7, (void*)name##_s8, \
    (void*)name##_tc16_0, (void*)name##_tc16_1, (void*)name##_tc16_2, \
    (void*)name##_tc16_3, (void*)name##_tc16_4, (void*)name##_tc16_5, \
    (void*)name##_tc16_6, (void*)name##_tc16_7, (void*)name##_tc16_8, \
    (void*)name##_tc16_9, (void*)name##_tc16_10, (void*)name##_tc16_11, \
    (void*)name##_tc16_12, (void*)name##_tc16_13, (void*)name##_tc16_14, \
    (void*)name##_tc16_15, \
    (void*)name##_tc32_0, (void*)name##_tc32_1, (void*)name##_tc32_2, \
    (void*)name##_tc32_3, (void*)name##_tc32_4, (void*)name##_tc32_5, \
    (void*)name##_tc32_6, (void*)name##_tc32_7, (void*)name##_tc32_8, \
    (void*)name##_tc32_9, (void*)name##_tc32_10, (void*)name##_tc32_11, \
    (void*)name##_tc32_12, (void*)name##_tc32_13, (void*)name##_tc32_14, \
    (void*)name##_tc32_15, \
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
inline void launch_kernel(
    void* kernel_fn, 
    int batch_tile,
    const void* vx, const void* vy, void* dst,
    int ncols_x, int nrows_x, int nrows_y, int nrows_dst, int batch_size,
    int grid_layout,
    size_t smem_size = 0
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
    
    cudaLaunchKernel(kernel_fn, grid, block, args, smem_size, nullptr);
}

// Launch an iterator kernel (internal batch loop, L2 weight reuse)
// Grid: (row_blocks, batch_tiles) where each block processes BATCHES_PER_BLOCK batches
// For s2_iter4: BATCHES_PER_BLOCK = 8, so batch_tiles = ceil(total_batches / 8)
inline void launch_kernel_iter(
    void* kernel_fn, 
    int batches_per_block,  // BATCH_TILE * NUM_ITERS (e.g., 8 for s2_iter4)
    const void* vx, const void* vy, void* dst,
    int ncols_x, int nrows_x, int nrows_y, int nrows_dst, int total_batches
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
    
    cudaLaunchKernel(kernel_fn, grid, block, args, 0, nullptr);
}

// TC kernel row tile size (4 warps × 8 rows/warp = 32 rows per block)
constexpr int TC_ROWS_PER_BLOCK = 32;

// Launch a tensor core kernel with grid.y batch tiling (single launch for all batches)
// Grid: (row_blocks, batch_tiles) where each block processes BATCH_TILE=16 batches
// batch_size is passed to kernel for bounds checking (last tile may be partial)
inline void launch_kernel_tc(
    void* kernel_fn, 
    int batch_tile,  // BATCH_TILE (16 for s16_tc)
    const void* vx, const void* vy, void* dst,
    int ncols_x, int nrows_x, int nrows_y, int nrows_dst, int batch_size
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
    
    cudaLaunchKernel(kernel_fn, grid, block, args, 0, nullptr);
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
//   remainder: 0-15, selects which tc16 kernel (R = batch_size % 16)
//   batch_size: total batches to process
inline void launch_tc16(
    const kernel_set_t& kset,
    int remainder,  // 0-15, selects tc16_0..tc16_15
    const void* vx, const void* vy, void* dst,
    int ncols_x, int nrows_x, int nrows_y, int nrows_dst, int batch_size
) {
    // Select the appropriate tc16 kernel based on remainder
    void* kernel_fn = nullptr;
    switch (remainder) {
        case 0:  kernel_fn = kset.tc16_0; break;
        case 1:  kernel_fn = kset.tc16_1; break;
        case 2:  kernel_fn = kset.tc16_2; break;
        case 3:  kernel_fn = kset.tc16_3; break;
        case 4:  kernel_fn = kset.tc16_4; break;
        case 5:  kernel_fn = kset.tc16_5; break;
        case 6:  kernel_fn = kset.tc16_6; break;
        case 7:  kernel_fn = kset.tc16_7; break;
        case 8:  kernel_fn = kset.tc16_8; break;
        case 9:  kernel_fn = kset.tc16_9; break;
        case 10: kernel_fn = kset.tc16_10; break;
        case 11: kernel_fn = kset.tc16_11; break;
        case 12: kernel_fn = kset.tc16_12; break;
        case 13: kernel_fn = kset.tc16_13; break;
        case 14: kernel_fn = kset.tc16_14; break;
        case 15: kernel_fn = kset.tc16_15; break;
        default: return;  // Invalid remainder, should never happen
    }
    
    // Compute total batch tiles
    const int tc16_tiles = batch_size / 16;
    const int total_batch_tiles = tc16_tiles + (remainder > 0 ? 1 : 0);
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

    cudaLaunchKernel(kernel_fn, cfg.grid, block, args, 0, nullptr);
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

inline void launch_tc32(
    const kernel_set_t& kset,
    int remainder,  // 0-15, selects tc32_0..tc32_15
    const void* vx, const void* vy, void* dst,
    int ncols_x, int nrows_x, int nrows_y, int nrows_dst, int batch_size
) {
    // Select the appropriate tc32 kernel based on remainder
    void* kernel_fn = nullptr;
    switch (remainder) {
        case 0:  kernel_fn = kset.tc32_0; break;
        case 1:  kernel_fn = kset.tc32_1; break;
        case 2:  kernel_fn = kset.tc32_2; break;
        case 3:  kernel_fn = kset.tc32_3; break;
        case 4:  kernel_fn = kset.tc32_4; break;
        case 5:  kernel_fn = kset.tc32_5; break;
        case 6:  kernel_fn = kset.tc32_6; break;
        case 7:  kernel_fn = kset.tc32_7; break;
        case 8:  kernel_fn = kset.tc32_8; break;
        case 9:  kernel_fn = kset.tc32_9; break;
        case 10: kernel_fn = kset.tc32_10; break;
        case 11: kernel_fn = kset.tc32_11; break;
        case 12: kernel_fn = kset.tc32_12; break;
        case 13: kernel_fn = kset.tc32_13; break;
        case 14: kernel_fn = kset.tc32_14; break;
        case 15: kernel_fn = kset.tc32_15; break;
        default: return;  // Invalid remainder, should never happen
    }
    
    // Compute total batch tiles (greedy decomposition)
    const int tc32_tiles = batch_size / 32;
    const int remainder_32 = batch_size % 32;
    const int has_tc16 = (remainder_32 >= 16) ? 1 : 0;
    const int has_tcR = (remainder > 0) ? 1 : 0;
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
    
    cudaLaunchKernel(kernel_fn, cfg.grid, block, args, 0, nullptr);
}


// =============================================================================
// TABLE-DRIVEN KERNEL LAUNCH HELPER
// =============================================================================
// Launches a GEMV kernel based on kernel_type_t from dispatch_table.cuh
// TC kernels (tc16, tc32) are handled by separate launch functions

inline void launch_kernel_by_type(
    const kernel_set_t& kset,
    kernel_type_t kt,
    int batches,
    const void* vx, const void* vy, void* dst,
    int ncols_x, int nrows_x, int nrows_y, int nrows_dst
) {
    // Simple sN kernels (DRAM-friendly, N batches per block)
    switch (kt) {
        case K_S1: launch_kernel(kset.s1, 1, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST); return;
        case K_S2: launch_kernel(kset.s2, 2, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST); return;
        case K_S3: launch_kernel(kset.s3, 3, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST); return;
        case K_S4: launch_kernel(kset.s4, 4, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST); return;
        case K_S5: launch_kernel(kset.s5, 5, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST); return;
        case K_S6: launch_kernel(kset.s6, 6, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST); return;
        case K_S7: launch_kernel(kset.s7, 7, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST); return;
        case K_S8: launch_kernel(kset.s8, 8, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches, GRID_LAYOUT_ROW_FAST); return;
        
        // Iter kernels (L2-friendly, high occupancy with internal loop)
        case K_S2_ITER2: launch_kernel_iter(kset.s2_iter2, 4, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches); return;
        case K_S2_ITER3: launch_kernel_iter(kset.s2_iter3, 6, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches); return;
        case K_S2_ITER4: launch_kernel_iter(kset.s2_iter4, 8, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches); return;
        case K_S2_ITER5: launch_kernel_iter(kset.s2_iter5, 10, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches); return;
        case K_S2_ITER6: launch_kernel_iter(kset.s2_iter6, 12, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches); return;
        case K_S2_ITER7: launch_kernel_iter(kset.s2_iter7, 14, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches); return;
        case K_S2_ITER8: launch_kernel_iter(kset.s2_iter8, 16, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches); return;
        
        case K_S3_ITER3: launch_kernel_iter(kset.s3_iter3, 9, vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst, batches); return;
        
        // Note: TC kernels (K_TC16_*, K_TC32_*) are launched
        // via launch_tc16() and launch_tc32() in the TC path,
        // not through this function. They shouldn't appear here.
        
        case K_NONE:
        default:
            return;
    }
}


// =============================================================================
// Unified dispatcher function with TABLE-DRIVEN batch decomposition
// =============================================================================
// qtype: 0=Q4_0, 1=Q4_1, 2=Q5_0, 3=Q5_1, 4=Q8_0, 5=Q2_K, 6=Q3_K, 7=Q4_K, 8=Q5_K, 9=Q6_K
// ytype: 0=F16, 1=BF16, 2=F32
//
// Dispatch flow (SM80+):
// 1. TC path for ALL batch >= 1 (tc16 for 1-31, tc32 for 32+). TC beat the
//    CUDA-core GEMV kernels in every measured case, batch 1-2 included.
// 2. CUDA-core GEMV (s1..s8) only on pre-SM80 GPUs without tensor cores.
// (MoE callers should prefer run_grouped_quantized_matmul, which fuses all
//  experts into one launch instead of looping this per-expert.)

// Segment descriptor: one per expert (MoE) or one total (non-MoE)
typedef struct {
    const void* weights;     // Device pointer to quantized weight data
    int32_t     batch_count; // Batches in this segment (greedy boundary)
} vx_segment_t;

// q8a128 INT8 DENSE kernels — the regular (non-MoE) QMatMul. Same INT8 m16n8k32
// core as the grouped kernels, single weight, implicit tile schedule (blockIdx.x →
// batch slice). Launched from run_quantized_matmul on ytype==3. Row ordering
// matches qtype_to_matmul_kernel_index (same as grouped_kernels_int8).
//
// Three output dtypes per format: the accumulator is F32 in registers either way,
// so the tag only names the width of the final store. A dense projection is read
// back at the model's activation dtype, and storing narrow there is what removes
// the separate cast launch (and halves the bytes the result occupies on the wave
// span). The grouped/MoE table stays F32-only — see grouped_kernels_int8.
#define DECL_DENSE_INT8(name) \
    extern "C" __global__ void name(const void*, const void*, void*, int, int, int, int, int);

#define DECL_DENSE_INT8_ALL(base) \
    DECL_DENSE_INT8(base##_f16_dense) \
    DECL_DENSE_INT8(base##_bf16_dense) \
    DECL_DENSE_INT8(base##_f32_dense)

#define DECL_DENSE_INT8_M2_ALL(base) \
    DECL_DENSE_INT8(base##_f16_dense_m2) \
    DECL_DENSE_INT8(base##_bf16_dense_m2) \
    DECL_DENSE_INT8(base##_f32_dense_m2)

DECL_DENSE_INT8_ALL(q4_0_int8)
DECL_DENSE_INT8_ALL(q4_1_int8)
DECL_DENSE_INT8_ALL(q5_0_int8)
DECL_DENSE_INT8_ALL(q5_1_int8)
DECL_DENSE_INT8_ALL(q8_0_int8)
DECL_DENSE_INT8_ALL(q2_k_int8)
DECL_DENSE_INT8_ALL(q3_k_int8)
DECL_DENSE_INT8_ALL(q4_k_int8)
DECL_DENSE_INT8_ALL(q5_k_int8)
DECL_DENSE_INT8_ALL(q6_k_int8)
DECL_DENSE_INT8_ALL(q8_1_int8)
DECL_DENSE_INT8_ALL(q8_k_int8)
DECL_DENSE_INT8_ALL(q_awq_int8)
DECL_DENSE_INT8_ALL(q_awq_g64_int8)
// KO byte-permuted twins (rows 14-17).
DECL_DENSE_INT8_ALL(q4_ko_int8)
DECL_DENSE_INT8_ALL(q5_ko_int8)
DECL_DENSE_INT8_ALL(q6_ko_int8)
DECL_DENSE_INT8_ALL(q8_ko_int8)
// MXFP4_KO exponent-collapse twin (row 18).
DECL_DENSE_INT8_ALL(mxfp4_ko_int8)
// Q2_KO 2-bit affine twin (row 19).
DECL_DENSE_INT8_ALL(q2_ko_int8)

// q8a128 mode-1 → mode-2 (Bm=32 weight-reuse) crossover. The DENSE crossover is decided in Rust
// (a weight-aware closed-form fit, see q8a128_dense_use_mode2) and passed in as `force_mode2`,
// since the optimal point depends on weight bytes vs L2, not token count alone.
// Mode-2 exists for the KO formats only — they are the ones a prefill wave runs.
DECL_DENSE_INT8_M2_ALL(q4_ko_int8)
DECL_DENSE_INT8_M2_ALL(q5_ko_int8)
DECL_DENSE_INT8_M2_ALL(q6_ko_int8)
DECL_DENSE_INT8_M2_ALL(q8_ko_int8)
DECL_DENSE_INT8_M2_ALL(mxfp4_ko_int8)
DECL_DENSE_INT8_M2_ALL(q2_ko_int8)

#undef DECL_DENSE_INT8_M2_ALL
#undef DECL_DENSE_INT8_ALL
#undef DECL_DENSE_INT8

// One row per output dtype, in OutDType order (0=F16, 1=BF16, 2=F32).
#define DENSE_INT8_ROW(tag) { \
    (void*)q4_0_int8_##tag##_dense,      /* 0   q4_0 */ \
    (void*)q4_1_int8_##tag##_dense,      /* 1   q4_1 */ \
    (void*)q5_0_int8_##tag##_dense,      /* 2   q5_0 */ \
    (void*)q5_1_int8_##tag##_dense,      /* 3   q5_1 */ \
    (void*)q8_0_int8_##tag##_dense,      /* 4   q8_0 */ \
    (void*)q2_k_int8_##tag##_dense,      /* 5   q2_K */ \
    (void*)q3_k_int8_##tag##_dense,      /* 6   q3_K */ \
    (void*)q4_k_int8_##tag##_dense,      /* 7   q4_K */ \
    (void*)q5_k_int8_##tag##_dense,      /* 8   q5_K */ \
    (void*)q6_k_int8_##tag##_dense,      /* 9   q6_K */ \
    (void*)q8_1_int8_##tag##_dense,      /* 10  q8_1 */ \
    (void*)q8_k_int8_##tag##_dense,      /* 11  q8_K */ \
    (void*)q_awq_int8_##tag##_dense,     /* 12  q_awq */ \
    (void*)q_awq_g64_int8_##tag##_dense, /* 13  q_awq_g64 */ \
    (void*)q4_ko_int8_##tag##_dense,     /* 14  q4_KO */ \
    (void*)q5_ko_int8_##tag##_dense,     /* 15  q5_KO */ \
    (void*)q6_ko_int8_##tag##_dense,     /* 16  q6_KO */ \
    (void*)q8_ko_int8_##tag##_dense,     /* 17  q8_KO */ \
    (void*)mxfp4_ko_int8_##tag##_dense,  /* 18  mxfp4_KO */ \
    (void*)q2_ko_int8_##tag##_dense,     /* 19  q2_KO */ \
}

// Indexed by [out_dtype][kernel_row].
static void* dense_kernels_int8[3][20] = {
    DENSE_INT8_ROW(f16),
    DENSE_INT8_ROW(bf16),
    DENSE_INT8_ROW(f32),
};
#undef DENSE_INT8_ROW

// Indexed by [out_dtype][kernel_row - 14]:
// Q4_KO=14, Q5_KO=15, Q6_KO=16, Q8_KO=17, MXFP4_KO=18, Q2_KO=19.
#define DENSE_INT8_M2_ROW(tag) { \
    (void*)q4_ko_int8_##tag##_dense_m2, \
    (void*)q5_ko_int8_##tag##_dense_m2, \
    (void*)q6_ko_int8_##tag##_dense_m2, \
    (void*)q8_ko_int8_##tag##_dense_m2, \
    (void*)mxfp4_ko_int8_##tag##_dense_m2, \
    (void*)q2_ko_int8_##tag##_dense_m2, \
}

static void* dense_kernels_int8_m2[3][6] = {
    DENSE_INT8_M2_ROW(f16),
    DENSE_INT8_M2_ROW(bf16),
    DENSE_INT8_M2_ROW(f32),
};
#undef DENSE_INT8_M2_ROW

extern "C" int run_quantized_matmul(
    const vx_segment_t* segments,
    int32_t num_segments,
    const void* vy,
    void* dst,
    int32_t ncols_x,
    int32_t nrows_x,
    int32_t nrows_y,
    int32_t nrows_dst,
    int32_t qtype,
    int32_t ytype,
    size_t weight_bytes,  // Weight tensor size in bytes for L2 cache decision (FP path)
    int32_t force_mode2,  // int8 dense tiling: 0 = mode-1 (Bm=16), 1 = mode-2 (Bm=32 reuse). Rust decides.
    int32_t out_dtype     // int8 dense store width: 0 = F16, 1 = BF16, 2 = F32. FP path ignores it
                          // (there the output dtype is the activation dtype).
) {
    // Lookup table for kernel sets: [qtype][ytype][use_tc]
    // ytype: 0=F16, 1=BF16, 2=F32
    // use_tc: 0 = CUDA cores, 1 = tensor cores (for s64)
    static kernel_set_t kernels[14][3][2] = {
        // Q4_0 (qtype=0)
        {{ KERNEL_SET(q4_0_f16), KERNEL_SET_TC(q4_0_f16) },
         { KERNEL_SET(q4_0_bf16), KERNEL_SET_TC(q4_0_bf16) },
         { KERNEL_SET(q4_0_f32), KERNEL_SET_TC(q4_0_f32) }},
        // Q4_1 (qtype=1)
        {{ KERNEL_SET(q4_1_f16), KERNEL_SET_TC(q4_1_f16) },
         { KERNEL_SET(q4_1_bf16), KERNEL_SET_TC(q4_1_bf16) },
         { KERNEL_SET(q4_1_f32), KERNEL_SET_TC(q4_1_f32) }},
        // Q5_0 (qtype=2)
        {{ KERNEL_SET(q5_0_f16), KERNEL_SET_TC(q5_0_f16) },
         { KERNEL_SET(q5_0_bf16), KERNEL_SET_TC(q5_0_bf16) },
         { KERNEL_SET(q5_0_f32), KERNEL_SET_TC(q5_0_f32) }},
        // Q5_1 (qtype=3)
        {{ KERNEL_SET(q5_1_f16), KERNEL_SET_TC(q5_1_f16) },
         { KERNEL_SET(q5_1_bf16), KERNEL_SET_TC(q5_1_bf16) },
         { KERNEL_SET(q5_1_f32), KERNEL_SET_TC(q5_1_f32) }},
        // Q8_0 (qtype=4)
        {{ KERNEL_SET(q8_0_f16), KERNEL_SET_TC(q8_0_f16) },
         { KERNEL_SET(q8_0_bf16), KERNEL_SET_TC(q8_0_bf16) },
         { KERNEL_SET(q8_0_f32), KERNEL_SET_TC(q8_0_f32) }},
        // Q2_K (qtype=5)
        {{ KERNEL_SET(q2_K_f16), KERNEL_SET_TC(q2_K_f16) },
         { KERNEL_SET(q2_K_bf16), KERNEL_SET_TC(q2_K_bf16) },
         { KERNEL_SET(q2_K_f32), KERNEL_SET_TC(q2_K_f32) }},
        // Q3_K (qtype=6)
        {{ KERNEL_SET(q3_k_f16), KERNEL_SET_TC(q3_k_f16) },
         { KERNEL_SET(q3_k_bf16), KERNEL_SET_TC(q3_k_bf16) },
         { KERNEL_SET(q3_k_f32), KERNEL_SET_TC(q3_k_f32) }},
        // Q4_K (qtype=7)
        {{ KERNEL_SET(q4_k_f16), KERNEL_SET_TC(q4_k_f16) },
         { KERNEL_SET(q4_k_bf16), KERNEL_SET_TC(q4_k_bf16) },
         { KERNEL_SET(q4_k_f32), KERNEL_SET_TC(q4_k_f32) }},
        // Q5_K (qtype=8)
        {{ KERNEL_SET(q5_k_f16), KERNEL_SET_TC(q5_k_f16) },
         { KERNEL_SET(q5_k_bf16), KERNEL_SET_TC(q5_k_bf16) },
         { KERNEL_SET(q5_k_f32), KERNEL_SET_TC(q5_k_f32) }},
        // Q6_K (qtype=9)
        {{ KERNEL_SET(q6_k_f16), KERNEL_SET_TC(q6_k_f16) },
         { KERNEL_SET(q6_k_bf16), KERNEL_SET_TC(q6_k_bf16) },
         { KERNEL_SET(q6_k_f32), KERNEL_SET_TC(q6_k_f32) }},
        // Q8_1 (qtype=10)
        {{ KERNEL_SET(q8_1_f16), KERNEL_SET_TC(q8_1_f16) },
         { KERNEL_SET(q8_1_bf16), KERNEL_SET_TC(q8_1_bf16) },
         { KERNEL_SET(q8_1_f32), KERNEL_SET_TC(q8_1_f32) }},
        // Q8_K (qtype=11)
        {{ KERNEL_SET(q8_k_f16), KERNEL_SET_TC(q8_k_f16) },
         { KERNEL_SET(q8_k_bf16), KERNEL_SET_TC(q8_k_bf16) },
         { KERNEL_SET(q8_k_f32), KERNEL_SET_TC(q8_k_f32) }},
        // Q_AWQ (qtype=12)
        {{ KERNEL_SET(q_awq_f16), KERNEL_SET_TC(q_awq_f16) },
         { KERNEL_SET(q_awq_bf16), KERNEL_SET_TC(q_awq_bf16) },
         { KERNEL_SET(q_awq_f32), KERNEL_SET_TC(q_awq_f32) }},
        // Q_AWQ_G64 (qtype=13)
        {{ KERNEL_SET(q_awq_g64_f16), KERNEL_SET_TC(q_awq_g64_f16) },
         { KERNEL_SET(q_awq_g64_bf16), KERNEL_SET_TC(q_awq_g64_bf16) },
         { KERNEL_SET(q_awq_g64_f32), KERNEL_SET_TC(q_awq_g64_f32) }},
    };

    // Map GgmlDType-aligned qtype values to the kernel-table row (0..13).
    // `qtype_to_matmul_kernel_index` lives in block_compact.cuh and returns
    // -1 for any format that has no matmul kernel.
    int kernel_row = qtype_to_matmul_kernel_index(qtype);
    if (kernel_row < 0) {
        return QMM_BAD_QTYPE;
    }

    // q8a128 INT8 path: `vy` is a block_q8a128 buffer, the single weight is
    // segments[0].weights (non-MoE → one segment), the output stores at `out_dtype`.
    // Same INT8 m16n8k32 core as the grouped path. TC-only. nrows_y is the batch M;
    // nrows_x is N. ytype 3 = Q8A128: the ONE int8 activation type. The mode (mode-1
    // Bm=16 vs mode-2 Bm=32 weight-reuse) is a kernel/tiling property the dispatcher
    // picks from the token count — the q8a1024 activation layout is mode-independent.
    if (ytype == 3) {
        if (num_segments < 1) {
            return QMM_NO_SEGMENTS;
        }
        if (out_dtype < 0 || out_dtype > 2) {
            return QMM_BAD_OUT_DTYPE;
        }
        const void* weights = segments[0].weights;
        const int total_batch = nrows_y;                  // M
        const int y_stride = ncols_x;                     // unused by the int8 kernel (ABI)
        const int dst_stride = nrows_x;                   // N
        const bool mode2 = (force_mode2 != 0);  // weight-reuse crossover decided in Rust
        void* kfn;
        int batch_div;
        if (mode2 && kernel_row >= 14 && kernel_row <= 19) {
            kfn = dense_kernels_int8_m2[out_dtype][kernel_row - 14];  // Bm=32 weight-reuse variant
            batch_div = 32;                               // Bm = 32 (mode-2, N_SUB=2)
        } else {
            kfn = dense_kernels_int8[out_dtype][kernel_row];
            batch_div = 16;                               // BATCH_TILE_I8 = 16 (mode-1)
        }
        if (kfn == nullptr) {
            return QMM_NO_KERNEL;
        }
        const int batch_tiles = (total_batch + batch_div - 1) / batch_div;
        const int row_tiles = (nrows_x + 31) / 32;        // N_TILE = 32
        dim3 grid(batch_tiles, row_tiles, 1);
        dim3 block(WARP_SIZE, 4, 1);                       // 128 threads (4 warps × 32)
        void* args[] = {
            (void*)&weights, (void*)&vy, (void*)&dst,
            (void*)&ncols_x, (void*)&nrows_x, (void*)&total_batch,
            (void*)&y_stride, (void*)&dst_stride,
        };
        cudaLaunchKernel(kfn, grid, block, args, 0, nullptr);
        return QMM_OK;
    }

    if (ytype < 0 || ytype > 2) {
        return QMM_BAD_YTYPE;
    }
    // KO byte-permuted formats (rows >= 14) have INT8 kernels only; the FP kernel
    // table `kernels` is sized to the 14 base formats. KO reaches here only on a
    // misrouted FP call — reject rather than index out of bounds.
    if (kernel_row >= 14) {
        return QMM_BAD_QTYPE;
    }

    // Cache device properties (queried once per device) via shared header
    const auto& caps = get_device_caps();

    // Determine tensor core usage from SM version
    // TC path uses MMA instructions - all dtypes can use TC!
    // ytype: 0=F16, 1=BF16, 2=F32
    // SM80+ (Ampere/Ada): TC enabled for F16/BF16/F32 (ytype 0, 1, 2)
    //   F32 uses FP16 MMA but accumulates in FP32 for precision
    bool use_tc = false;
    if (ytype == 0 || ytype == 1 || ytype == 2) {  // F16, BF16, or F32
        use_tc = (caps.sm_version >= 800);
    }

    // Select kernel set from lookup table
    // tc_idx: 0 = CUDA cores, 1 = tensor cores (s16_tc variant)
    int tc_idx = use_tc ? 1 : 0;
    const kernel_set_t& kset = kernels[kernel_row][ytype][tc_idx];
    
    // Runtime validation: check that at least s1 kernel is linked (catches missing .cu files)
    // This is a one-time low-cost check that prevents silent NaN from unlinked kernels
    if (kset.s1 == nullptr) {
        // Kernel not compiled for this (format, activation dtype).
        return QMM_NO_KERNEL;
    }
    
    // Calculate Y element size for pointer arithmetic
    // ytype: 0=F16 (2 bytes), 1=BF16 (2 bytes), 2=F32 (4 bytes)
    static const int y_elem_sizes[3] = { 2, 2, 4 };
    const int y_elem_size = y_elem_sizes[ytype];
    
    // Calculate dst element size (same as Y type)
    // F16/BF16 = 2 bytes, F32 = 4 bytes
    const int dst_elem_size = y_elem_sizes[ytype];
    
    // =========================================================================
    // L2 CACHE DECISION
    // =========================================================================
    // L2 cache threshold: 80% of L2 (leave room for activations)
    size_t l2_threshold = (caps.l2_cache_size * 80) / 100;
    
    // Dispatch strategy selection:
    // - L2-cached weights: Use iter kernels (high occupancy hides L2 latency)
    // - DRAM-bound weights: Use greedy s8+sN (more weight reuse per launch)
    bool use_l2_path = (weight_bytes < l2_threshold);
    
    // Variables for potential future L2 policy use (currently disabled)
    cudaStream_t stream = nullptr;  // Using default stream
    (void)stream;  // Suppress unused warning
    
    // =========================================================================
    // SEGMENT LOOP: dispatch each segment (expert) independently
    // =========================================================================
    // Each segment gets full greedy decomposition (TC → k1/k2/k3 → bulk → remainder).
    // Host-serial, GPU-async: each cudaLaunchKernel returns immediately.
    // Single-segment (non-MoE) is 1-iteration — same codepath, zero branching.
    
    int batch_offset = 0;
    
    for (int seg = 0; seg < num_segments; ++seg) {
        const void* vx = segments[seg].weights;
        int seg_batch = segments[seg].batch_count;
        if (seg_batch == 0) continue;
        
        const void* vy_slice = static_cast<const char*>(vy) + (size_t)batch_offset * nrows_y * y_elem_size;
        void* dst_slice = static_cast<char*>(dst) + (size_t)batch_offset * nrows_dst * dst_elem_size;
        
        int remaining = seg_batch;
        
        // =================================================================
        // TC PATH (plan-driven)
        // =================================================================
        dispatch_plan_t plan = get_dispatch_plan(remaining, use_l2_path, use_tc);
        
        if (is_tc32_kernel(plan.tc_kernel) && plan.tc_batch > 0) {
            int remainder = get_tc32_remainder(plan.tc_kernel);
            launch_tc32(kset, remainder, vx, vy_slice, dst_slice,
                       ncols_x, nrows_x, nrows_y, nrows_dst, plan.tc_batch);
            remaining = 0;
        } else if (is_tc16_kernel(plan.tc_kernel) && plan.tc_batch > 0) {
            int remainder = get_tc16_remainder(plan.tc_kernel);
            launch_tc16(kset, remainder, vx, vy_slice, dst_slice,
                       ncols_x, nrows_x, nrows_y, nrows_dst, plan.tc_batch);
            remaining = 0;
        }
        
        // =================================================================
        // GEMV KERNELS from plan (remainder after TC, or full batch if no TC)
        // =================================================================
        // Note: vx is the same weight pointer for all launches in this segment.
        // Only vy_slice/dst_slice advance (different batch data, same weights).
        
        // k1
        if (plan.k1 != K_NONE && plan.b1 > 0 && remaining > 0) {
            int actual_b1 = (plan.b1 <= remaining) ? plan.b1 : remaining;
            launch_kernel_by_type(kset, plan.k1, actual_b1, vx, vy_slice, dst_slice,
                                 ncols_x, nrows_x, nrows_y, nrows_dst);
            remaining -= actual_b1;
            vy_slice = static_cast<const char*>(vy_slice) + actual_b1 * nrows_y * y_elem_size;
            dst_slice = static_cast<char*>(dst_slice) + actual_b1 * nrows_dst * dst_elem_size;
        }
        
        // k2
        if (plan.k2 != K_NONE && plan.b2 > 0 && remaining > 0) {
            int actual_b2 = (plan.b2 <= remaining) ? plan.b2 : remaining;
            launch_kernel_by_type(kset, plan.k2, actual_b2, vx, vy_slice, dst_slice,
                                 ncols_x, nrows_x, nrows_y, nrows_dst);
            remaining -= actual_b2;
            vy_slice = static_cast<const char*>(vy_slice) + actual_b2 * nrows_y * y_elem_size;
            dst_slice = static_cast<char*>(dst_slice) + actual_b2 * nrows_dst * dst_elem_size;
        }
        
        // k3 (L2 path 3-kernel cases)
        if (plan.k3 != K_NONE && plan.b3 > 0 && remaining > 0) {
            int actual_b3 = (plan.b3 <= remaining) ? plan.b3 : remaining;
            launch_kernel_by_type(kset, plan.k3, actual_b3, vx, vy_slice, dst_slice,
                                 ncols_x, nrows_x, nrows_y, nrows_dst);
            remaining -= actual_b3;
            vy_slice = static_cast<const char*>(vy_slice) + actual_b3 * nrows_y * y_elem_size;
            dst_slice = static_cast<char*>(dst_slice) + actual_b3 * nrows_dst * dst_elem_size;
        }
        
        // =================================================================
        // BULK CATCH: Reduce any remaining >= 8 using s8 with grid.y
        // =================================================================
        while (remaining >= 8) {
            int bulk = (remaining / 8) * 8;
            launch_kernel(kset.s8, 8, vx, vy_slice, dst_slice,
                         ncols_x, nrows_x, nrows_y, nrows_dst, bulk, GRID_LAYOUT_ROW_FAST);
            remaining -= bulk;
            vy_slice = static_cast<const char*>(vy_slice) + bulk * nrows_y * y_elem_size;
            dst_slice = static_cast<char*>(dst_slice) + bulk * nrows_dst * dst_elem_size;
        }
        
        // =================================================================
        // FINAL REMAINDER: Handle remaining 1-7 batches
        // =================================================================
        if (remaining >= 1 && remaining <= 7) {
            void* remainder_kernels[] = {
                nullptr, kset.s1, kset.s2, kset.s3, kset.s4, kset.s5, kset.s6, kset.s7
            };
            launch_kernel(remainder_kernels[remaining], remaining, vx, vy_slice, dst_slice,
                         ncols_x, nrows_x, nrows_y, nrows_dst, remaining, GRID_LAYOUT_ROW_FAST);
        }
        
        batch_offset += seg_batch;
    }
    return QMM_OK;
}

// =============================================================================
// GROUPED MATMUL — all MoE experts in a single launch
// =============================================================================
// One kernel over a device-side (tile → expert, batch-slice) table. Collapses
// the per-expert segment loop above into a single cudaLaunchKernel whose grid
// (total_tiles × row_tiles) spans all experts at once, fixing both the launch
// count and the per-expert occupancy starvation. The tile tables + weight
// pointer array are built and uploaded by the caller (candle-core).

#define DECLARE_GROUPED(name) \
    extern "C" __global__ void name##_grouped( \
        const void*, const void*, const void*, const void*, \
        const void*, void*, int, int, int, int);
#define DECLARE_GROUPED3(base) \
    DECLARE_GROUPED(base##_f16) DECLARE_GROUPED(base##_bf16) DECLARE_GROUPED(base##_f32)

DECLARE_GROUPED3(q4_0)  DECLARE_GROUPED3(q4_1)  DECLARE_GROUPED3(q5_0)
DECLARE_GROUPED3(q5_1)  DECLARE_GROUPED3(q8_0)  DECLARE_GROUPED3(q2_K)
DECLARE_GROUPED3(q3_k)  DECLARE_GROUPED3(q4_k)  DECLARE_GROUPED3(q5_k)
DECLARE_GROUPED3(q6_k)  DECLARE_GROUPED3(q8_1)  DECLARE_GROUPED3(q8_k)
DECLARE_GROUPED3(q_awq) DECLARE_GROUPED3(q_awq_g64)

#undef DECLARE_GROUPED3
#undef DECLARE_GROUPED

#define GROUPED_ROW(base) \
    { (void*)base##_f16_grouped, (void*)base##_bf16_grouped, (void*)base##_f32_grouped }

// [qtype_kernel_row][ytype] — same ordering as the kernels[14][3] table above.
static void* grouped_kernels[14][3] = {
    GROUPED_ROW(q4_0),  GROUPED_ROW(q4_1),  GROUPED_ROW(q5_0),  GROUPED_ROW(q5_1),
    GROUPED_ROW(q8_0),  GROUPED_ROW(q2_K),  GROUPED_ROW(q3_k),  GROUPED_ROW(q4_k),
    GROUPED_ROW(q5_k),  GROUPED_ROW(q6_k),  GROUPED_ROW(q8_1),  GROUPED_ROW(q8_k),
    GROUPED_ROW(q_awq), GROUPED_ROW(q_awq_g64),
};

#undef GROUPED_ROW

// q8a128 TC path — INT8-MMA: q8a128 activations (raw int8 qs) are multiplied
// against Q4_K weights (raw 4-bit nibbles) on the m16n8k32 int8 tensor core, with
// the deferred-scale fold (d_w·s_a·C + m_w·Σx) applied post-MMA and F32 output.
// See grouped_tc_int8 in kernel.cuh. Same launch ABI as the FP grouped kernels
// (one block_q8a128 activation pointer). Registered for ytype==3 →
// grouped_kernels_int8[row].
extern "C" __global__ void q4_k_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q8_0_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q4_0_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q4_1_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q5_0_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q5_1_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q5_k_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q6_k_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q3_k_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q2_k_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q8_1_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q8_k_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q_awq_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q_awq_g64_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
// KO byte-permuted twins (rows 14-17).
extern "C" __global__ void q4_ko_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q5_ko_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q6_ko_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
extern "C" __global__ void q8_ko_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
// MXFP4_KO exponent-collapse twin (row 18).
extern "C" __global__ void mxfp4_ko_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);
// Q2_KO 2-bit affine twin (row 19).
extern "C" __global__ void q2_ko_int8_f32_grouped(
    const void*, const void*, const void*, const void*, const void*,
    void*, int, int, int, int);

// Wide-Bm (mode-4 / mode-8) grouped twins — KO rows only (14-19): the int8
// impl is KO-exclusive, and the wide tiles exist for the routed-expert
// PREFILL regime where those are the only formats in play. The host requests
// a wide mode only for KO dtypes (cuda.rs), so the null rows are never asked
// for; a null lookup returns without launching rather than mis-launching.
#define DECLARE_GROUPED_WIDE(name) \
    extern "C" __global__ void name##_grouped_m4( \
        const void*, const void*, const void*, const void*, const void*, \
        void*, int, int, int, int); \
    extern "C" __global__ void name##_grouped_m8( \
        const void*, const void*, const void*, const void*, const void*, \
        void*, int, int, int, int);
DECLARE_GROUPED_WIDE(q4_ko_int8_f32)
DECLARE_GROUPED_WIDE(q5_ko_int8_f32)
DECLARE_GROUPED_WIDE(q6_ko_int8_f32)
DECLARE_GROUPED_WIDE(q8_ko_int8_f32)
DECLARE_GROUPED_WIDE(mxfp4_ko_int8_f32)
DECLARE_GROUPED_WIDE(q2_ko_int8_f32)
#undef DECLARE_GROUPED_WIDE

static void* grouped_kernels_int8_m4[20] = {
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    (void*)q4_ko_int8_f32_grouped_m4,    // 14  q4_KO
    (void*)q5_ko_int8_f32_grouped_m4,    // 15  q5_KO
    (void*)q6_ko_int8_f32_grouped_m4,    // 16  q6_KO
    (void*)q8_ko_int8_f32_grouped_m4,    // 17  q8_KO
    (void*)mxfp4_ko_int8_f32_grouped_m4, // 18  mxfp4_KO
    (void*)q2_ko_int8_f32_grouped_m4,    // 19  q2_KO
};
static void* grouped_kernels_int8_m8[20] = {
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    (void*)q4_ko_int8_f32_grouped_m8,    // 14  q4_KO
    (void*)q5_ko_int8_f32_grouped_m8,    // 15  q5_KO
    (void*)q6_ko_int8_f32_grouped_m8,    // 16  q6_KO
    (void*)q8_ko_int8_f32_grouped_m8,    // 17  q8_KO
    (void*)mxfp4_ko_int8_f32_grouped_m8, // 18  mxfp4_KO
    (void*)q2_ko_int8_f32_grouped_m8,    // 19  q2_KO
};

// [qtype_kernel_row] — same row ordering as grouped_kernels above.
static void* grouped_kernels_int8[20] = {
    (void*)q4_0_int8_f32_grouped,      // 0   q4_0
    (void*)q4_1_int8_f32_grouped,      // 1   q4_1
    (void*)q5_0_int8_f32_grouped,      // 2   q5_0
    (void*)q5_1_int8_f32_grouped,      // 3   q5_1
    (void*)q8_0_int8_f32_grouped,      // 4   q8_0
    (void*)q2_k_int8_f32_grouped,      // 5   q2_K
    (void*)q3_k_int8_f32_grouped,      // 6   q3_K
    (void*)q4_k_int8_f32_grouped,      // 7   q4_K
    (void*)q5_k_int8_f32_grouped,      // 8   q5_K
    (void*)q6_k_int8_f32_grouped,      // 9   q6_K
    (void*)q8_1_int8_f32_grouped,      // 10  q8_1
    (void*)q8_k_int8_f32_grouped,      // 11  q8_K
    (void*)q_awq_int8_f32_grouped,     // 12  q_awq
    (void*)q_awq_g64_int8_f32_grouped, // 13  q_awq_g64
    (void*)q4_ko_int8_f32_grouped,     // 14  q4_KO
    (void*)q5_ko_int8_f32_grouped,     // 15  q5_KO
    (void*)q6_ko_int8_f32_grouped,     // 16  q6_KO
    (void*)q8_ko_int8_f32_grouped,     // 17  q8_KO
    (void*)mxfp4_ko_int8_f32_grouped,  // 18  mxfp4_KO
    (void*)q2_ko_int8_f32_grouped,     // 19  q2_KO
};

/// Single-launch grouped matmul over all expert tiles.
///
/// All pointer arguments are DEVICE pointers, prepared by the caller:
/// - weight_ptrs:  uint64_t[num_experts] — each expert's K/128 weight pointer
/// - tile_expert:  int[num_tiles] — owning expert id per tile
/// - tile_b_start: int[num_tiles] — stacked-batch start row per tile
/// - tile_b_cnt:   int[num_tiles] — tokens in the tile (1..16)
/// - vy:           stacked activations [total_batch, K]
/// - dst:          stacked output [total_batch, N]
///
/// ncols_x = K, nrows_x = N, y_stride = K, dst_stride = N.
extern "C" void run_grouped_quantized_matmul(
    const void* weight_ptrs,
    const void* tile_expert,
    const void* tile_b_start,
    const void* tile_b_cnt,
    const void* vy,
    void* dst,
    int32_t ncols_x,
    int32_t nrows_x,
    int32_t y_stride,
    int32_t dst_stride,
    int32_t num_tiles,
    int32_t qtype,
    int32_t ytype,
    int32_t n_sub)  // int8 token-tile width / 16: 2 (Bm 32), 4 (Bm 64), 8 (Bm 128)
{
    int kernel_row = qtype_to_matmul_kernel_index(qtype);
    if (kernel_row < 0 || ytype < 0 || ytype > 3 || num_tiles <= 0) {
        return;
    }
    // KO byte-permuted formats (rows >= 14) have INT8 grouped kernels only; the FP
    // `grouped_kernels` table is sized to the 14 base formats. Reject a misrouted
    // FP (ytype != 3) call on a KO row before indexing out of bounds.
    if (ytype != 3 && kernel_row >= 14) {
        return;
    }
    // Activation input selected by `ytype`, same 10-arg launch ABI either way:
    //   0/1/2 (F16/BF16/F32) → FP activations → FP16-MMA grouped kernel.
    //   3 (q8a128)           → q8 activations (`vy` is a block_q8a128 buffer) →
    //                          INT8-MMA grouped kernel (raw int8 × Q4_K nibbles,
    //                          deferred-scale fold, F32 output). TC-only — callers
    //                          only pass q8a128 when tensor cores exist.
    // `n_sub` selects the int8 tile width — the caller sized the tile tables
    // to 16·n_sub, so a mode without a kernel MUST refuse rather than launch a
    // narrower kernel that would drop the tiles' upper rows.
    void* kfn;
    if (ytype == 3) {
        switch (n_sub) {
            case 2: kfn = grouped_kernels_int8[kernel_row]; break;
            case 4: kfn = grouped_kernels_int8_m4[kernel_row]; break;
            case 8: kfn = grouped_kernels_int8_m8[kernel_row]; break;
            default: return;
        }
    } else {
        kfn = grouped_kernels[kernel_row][ytype];
    }
    if (kfn == nullptr) {
        return;
    }

    const int row_tiles = (nrows_x + 31) / 32;  // N_TILE = 32
    dim3 grid(num_tiles, row_tiles, 1);
    dim3 block(WARP_SIZE, 4, 1);  // 128 threads (4 warps × 32)

    void* args[] = {
        (void*)&weight_ptrs, (void*)&tile_expert, (void*)&tile_b_start, (void*)&tile_b_cnt,
        (void*)&vy, (void*)&dst,
        (void*)&ncols_x, (void*)&nrows_x, (void*)&y_stride, (void*)&dst_stride,
    };
    cudaLaunchKernel(kfn, grid, block, args, 0, nullptr);
}

// =============================================================================
// DISPATCH INFO QUERY (for benchmark reporting)
// =============================================================================
// Returns a string describing which kernels will be used for a given configuration.
// Format: "kernel1(batch1)+kernel2(batch2)" or "kernel1(batch1)" for single launches
// Examples: "s2i8(16)", "s2i4(8)+s3(3)", "[tc32(32)]+s8(8)"
//
// This function uses get_dispatch_plan() to ensure reporting matches actual dispatch.

extern "C" int32_t get_dispatch_info(
    int32_t batch_size,
    size_t weight_bytes,
    char* buffer,
    int32_t buffer_len
) {
    if (buffer == nullptr || buffer_len < 32) {
        return -1;
    }
    
    // Query L2 cache size and SM version
    int device;
    cudaGetDevice(&device);
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, device);
    int sm_major;
    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
    
    // Use 80% of L2 as threshold (same as dispatcher)
    size_t l2_threshold = (size_t)(l2_cache_size * 0.8);
    bool use_l2_path = (weight_bytes <= l2_threshold);
    bool use_tc = (sm_major >= 8);  // Ampere+ for tensor cores
    
    int pos = 0;
    int remaining = batch_size;
    
    // =========================================================================
    // PLAN-BASED REPORTING: Uses same logic as dispatcher
    // =========================================================================
    
    // For batch >= 64, we need bulk reduction first
    // L2 path: s2_iter8 (16 batches/call), DRAM path: s8 (8 batches/call)
    if (remaining >= DISPATCH_TABLE_SIZE) {
        // Get plan for the full batch - TC handles up front
        dispatch_plan_t plan = get_dispatch_plan(remaining, use_l2_path, use_tc);
        
        // Report TC kernel if used - TC32 handles everything internally
        if (plan.tc_kernel != K_NONE && plan.tc_batch > 0) {
            pos += snprintf(buffer + pos, buffer_len - pos, "%s(%d)", 
                           kernel_type_name(plan.tc_kernel), plan.tc_batch);
            remaining = 0;  // TC32 kernel handles all batches internally
        }
        
        // Report bulk reduction for remaining >= 64 (only if no TC)
        if (remaining >= DISPATCH_TABLE_SIZE) {
            if (use_l2_path) {
                int bulk = (remaining / 16) * 16;
                pos += snprintf(buffer + pos, buffer_len - pos, "s2i8(%d)", bulk);
                remaining -= bulk;
            } else {
                int bulk = (remaining / 8) * 8;
                pos += snprintf(buffer + pos, buffer_len - pos, "s8(%d)", bulk);
                remaining -= bulk;
            }
            if (remaining > 0 && pos < buffer_len - 1) {
                buffer[pos++] = '+';
            }
        }
    }
    
    // =========================================================================
    // TABLE LOOKUP for remainder (0-63)
    // =========================================================================
    if (remaining > 0 && remaining < DISPATCH_TABLE_SIZE) {
        dispatch_plan_t plan = get_dispatch_plan(remaining, use_l2_path, use_tc);
        
        // TC kernel handles everything internally (tc16 tiles + remainder)
        if (plan.tc_kernel != K_NONE && plan.tc_batch > 0) {
            pos += snprintf(buffer + pos, buffer_len - pos, "%s(%d)", 
                           kernel_type_name(plan.tc_kernel), plan.tc_batch);
            remaining = 0;  // TC kernel handles all batches, no GEMV remainder
        } else {
            // GEMV kernels (only when TC not used)
            // First GEMV kernel
            if (plan.k1 != K_NONE && plan.b1 > 0) {
                pos += snprintf(buffer + pos, buffer_len - pos, "%s(%d)", 
                               kernel_type_name(plan.k1), plan.b1);
                remaining -= plan.b1;
            }
            
            // Second GEMV kernel
            if (plan.k2 != K_NONE && plan.b2 > 0 && pos < buffer_len - 1) {
                buffer[pos++] = '+';
                pos += snprintf(buffer + pos, buffer_len - pos, "%s(%d)", 
                               kernel_type_name(plan.k2), plan.b2);
                remaining -= plan.b2;
            }
            
            // Third GEMV kernel (L2 path 3-kernel cases)
            if (plan.k3 != K_NONE && plan.b3 > 0 && pos < buffer_len - 1) {
                buffer[pos++] = '+';
                pos += snprintf(buffer + pos, buffer_len - pos, "%s(%d)", 
                               kernel_type_name(plan.k3), plan.b3);
                remaining -= plan.b3;
            }
            
            // Check for 3-kernel case (L2 path remainder) - fallback if not in plan
            if (use_l2_path && remaining > 0) {
                int l2_rem = get_l2_remainder(batch_size % 64);
                if (l2_rem > 0 && remaining == l2_rem && pos < buffer_len - 1) {
                    buffer[pos++] = '+';
                    pos += snprintf(buffer + pos, buffer_len - pos, "s3(%d)", l2_rem);
                    remaining = 0;
                }
            }
        }
    }
    
    // Final remainder (shouldn't happen with good tables)
    if (remaining > 0 && remaining <= 7 && pos < buffer_len - 8) {
        if (pos > 0 && buffer[pos-1] != '+') {
            buffer[pos++] = '+';
        }
        pos += snprintf(buffer + pos, buffer_len - pos, "s%d(%d)", remaining, remaining);
    }
    
    buffer[pos] = '\0';
    return pos;
}
