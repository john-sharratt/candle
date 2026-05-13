#pragma once

// =============================================================================
// TYPES AND TRAITS FOR QUANTIZED KERNELS
// =============================================================================
// This file provides:
// 1. accumulator_type: Maps output_t → appropriate accumulator type
// 2. tile_k_config: Dynamic tile sizing based on quant block and MMA alignment
// 3. pipeline_config: Dynamic smem/buffer configuration based on available memory
// 4. Helper aliases for easy access to computed constants
//
// The pipeline_config infrastructure computes optimal tile sizes dynamically
// based on shared memory constraints, quant block sizes, and MMA K dimensions.
// This replaces the previous hardcoded tile_k_blocks_trait and can_double_buffer_trait.
// =============================================================================

// Forward declaration for dequant_sizes (defined in impl/common.cuh before including this)
template <typename block_t> struct dequant_sizes;

// Accumulator type trait: maps output_t → appropriate accumulator type
// ALWAYS use float for accumulation to prevent precision loss over 256+ additions.
// Only the final output conversion uses the target precision.
// This adds ~10% register pressure but ensures correctness.
template <typename T> struct accumulator_type;

template <> struct accumulator_type<float> {
    using type = float;
};

template <> struct accumulator_type<__half> {
    using type = float;  // Float accumulator prevents precision loss over many additions
};

template <> struct accumulator_type<__nv_bfloat16> {
    using type = float;  // Float accumulator prevents precision loss over many additions
};

template <> struct accumulator_type<__nv_fp8_e4m3> {
    using type = float;  // Float accumulator prevents precision loss over many additions
};

template <> struct accumulator_type<__nv_fp8_e5m2> {
    using type = float;  // Float accumulator prevents precision loss over many additions
};

// Convenience alias
template <typename T>
using accumulator_t = typename accumulator_type<T>::type;

// =============================================================================
// COMPILE-TIME MATH UTILITIES
// =============================================================================

// Compile-time GCD (Greatest Common Divisor)
constexpr int ct_gcd(int a, int b) {
    return b == 0 ? a : ct_gcd(b, a % b);
}

// Compile-time LCM (Least Common Multiple)
constexpr int ct_lcm(int a, int b) {
    return a / ct_gcd(a, b) * b;
}

// =============================================================================
// SHARED MEMORY BUDGET CONSTANTS
// =============================================================================
// Total shared memory available based on USE_TC:
// - USE_TC=false: Older GPUs (Turing, Pascal) - 48 KB
// - USE_TC=true:  SM80+ (Ampere, Ada, Hopper) - 100 KB (conservative for RTX 3090/4090)

constexpr int SMEM_BYTES_SCALAR = 48 * 1024;   // 48 KB for non-TC path
constexpr int SMEM_BYTES_TC = 100 * 1024;      // 100 KB for TC path (SM80+)

// Reserved shared memory (before pipeline buffers):
// - Cross-warp reduction: (nwarps-1) * rows_per_cuda_block * WARP_SIZE * sizeof(float)
//   With nwarps=4, rows=16: 3 * 16 * 32 * 4 = 6144 bytes
// - Y vector tile (TILE_K elements as FP8/FP16): ~256-2048 bytes depending on type
// - Miscellaneous alignment padding: ~256 bytes
constexpr int SMEM_RESERVED_BYTES = 8 * 1024;  // 8 KB reserved

// Cross-warp reduction tmp_shared: (nwarps-1) * rows * WARP_SIZE * sizeof(acc_t)
// For FP8 path (acc_t=float): 3 * 16 * 32 * 4 = 6144 bytes
// For BF16 path (acc_t=bf16): 3 * 16 * 32 * 2 = 3072 bytes
// Use max (6144) to be safe
constexpr int TMP_SHARED_BYTES = 6144;

// Available for dequant pipeline buffers
// Note: TC path uses 4 buffers (2×X + 2×Y), plus tmp_shared for cross-warp reduction
constexpr int SMEM_PIPELINE_SCALAR = SMEM_BYTES_SCALAR - SMEM_RESERVED_BYTES;  // 40 KB
// TC path: 100KB extended smem - tmp_shared - some padding
// The launcher will call cudaFuncSetAttribute to enable extended smem
constexpr int SMEM_PIPELINE_TC = 100 * 1024 - TMP_SHARED_BYTES - 2 * 1024;  // ~92 KB for 4-buffer pipeline

// Alignment constants
constexpr int CP_ASYNC_ALIGN = 16;  // cp.async requires 16-byte alignment
constexpr int MMA_K_FP8 = 32;       // FP8 tensor core K dimension
constexpr int MMA_K_FP16 = 16;      // FP16/BF16 tensor core K dimension

// =============================================================================
// TILE K CONFIGURATION
// =============================================================================
// Computes the base tile_k (in elements) that satisfies three alignment constraints:
//   1. output_count (elements per dequant) - thread work granularity
//   2. MMA_K (32 for FP8) - tensor core K dimension
//   3. CP_ASYNC_ALIGN (16 bytes) - cp.async alignment requirement
//
// The relationship: for tile_k elements, we consume (tile_k / output_count) * input_stride bytes
// So the byte consumption must also be 16-byte aligned.

template <typename block_t, int MMA_K = MMA_K_FP8>
struct tile_k_config {
    static constexpr int input_stride = dequant_sizes<block_t>::input_stride;
    static constexpr int output_count = dequant_sizes<block_t>::output_count;
    
    // Step 1: base_tile_k must be multiple of both output_count and MMA_K
    static constexpr int lcm_tile_k = ct_lcm(output_count, MMA_K);
    
    // Step 2: Check byte consumption at lcm_tile_k
    // bytes_consumed = (tile_k / output_count) * input_stride
    static constexpr int blocks_at_lcm = lcm_tile_k / output_count;
    static constexpr int bytes_at_lcm = blocks_at_lcm * input_stride;
    
    // Step 3: Scale up if bytes_at_lcm is not 16-byte aligned
    // We need: (scale * bytes_at_lcm) % 16 == 0
    // Minimum scale = 16 / gcd(bytes_at_lcm, 16)
    static constexpr int byte_alignment_scale = 
        (bytes_at_lcm % CP_ASYNC_ALIGN == 0) ? 1 
        : CP_ASYNC_ALIGN / ct_gcd(bytes_at_lcm, CP_ASYNC_ALIGN);
    
    // Base tile_k: minimum aligned tile size (in elements)
    static constexpr int base_value = lcm_tile_k * byte_alignment_scale;
    
    // Base derived quantities
    static constexpr int base_blocks_per_tile = base_value / output_count;
    static constexpr int base_bytes_per_tile = base_blocks_per_tile * input_stride;
    
    // Static assertions for tile_k_config invariants
    static_assert(base_value > 0, "base_value must be positive");
    static_assert(base_value % output_count == 0, "base_value must be multiple of output_count");
    static_assert(base_value % MMA_K == 0, "base_value must be multiple of MMA_K");
    static_assert(base_bytes_per_tile % CP_ASYNC_ALIGN == 0, "base_bytes must be 16-byte aligned for cp.async");
    static_assert(base_blocks_per_tile > 0, "must have at least 1 block per tile");
};

// Helper aliases for base tile values (before multiplier)
template <typename block_t, int MMA_K = MMA_K_FP8>
constexpr int tile_base_k_v = tile_k_config<block_t, MMA_K>::base_value;

template <typename block_t, int MMA_K = MMA_K_FP8>
constexpr int tile_base_blocks_v = tile_k_config<block_t, MMA_K>::base_blocks_per_tile;

template <typename block_t, int MMA_K = MMA_K_FP8>
constexpr int tile_base_bytes_v = tile_k_config<block_t, MMA_K>::base_bytes_per_tile;

// Dequantize smem bytes: bytes of smem consumed per dequantize call
// This is sizeof(output_t) * output_count (elements produced per call)
template <typename block_t, typename output_t>
constexpr int dequant_smem_bytes_v = sizeof(output_t) * dequant_sizes<block_t>::output_count;

// Base tile smem bytes (one buffer, multiplier=1)
template <typename block_t, typename output_t, int MMA_K = MMA_K_FP8>
constexpr int tile_base_smem_bytes_v = sizeof(output_t) * tile_base_k_v<block_t, MMA_K>;

// =============================================================================
// PIPELINE BUFFER CONFIGURATION
// =============================================================================
// Determine tile_multiplier and buffering based on available smem.
// Double buffering allows overlap of dequantize with MMA compute.
// We maximize tile_multiplier while still fitting num_buffers in smem.
//
// 4-BUFFER LAYOUT (for MMA path):
//   smem_Y[2][TILE_K]                    - double-buffered dequantized Y
//   smem_X[2][rows_per_cuda_block][TILE_K] - double-buffered dequantized X
//
// Pipeline: DEQUANT(X,Y) → buf[i] || COMPUTE from buf[1-i]

// Maximum tile_k in elements - keeps tiles practical for typical K dimensions.
// With 4 warps × 32 threads = 128 threads, and 4 concurrent dequants per iteration,
// 2048 elements = ~8 iterations for Q4_0 (64 elem/dequant) or ~2 for K-quants (256).
// This balances latency hiding with memory efficiency across K dimensions 4K-16K.
static constexpr int MAX_TILE_K_ELEMENTS = 2048;

// rows_per_cuda_block for TC path (must match kernel's rows_per_cuda_block)
static constexpr int TC_ROWS_PER_BLOCK = 16;

template <typename block_t, typename output_t, bool USE_TC, int MMA_K = MMA_K_FP8>
struct pipeline_config {
    static constexpr int smem_available = USE_TC ? SMEM_PIPELINE_TC : SMEM_PIPELINE_SCALAR;
    static constexpr int base_tile_k = tile_base_k_v<block_t, MMA_K>;
    
    // Per-buffer smem for Y: TILE_K elements × sizeof(output_t)
    static constexpr int base_smem_y = base_tile_k * sizeof(output_t);
    
    // Per-buffer smem for X: rows × TILE_K elements × sizeof(output_t)
    static constexpr int base_smem_x = TC_ROWS_PER_BLOCK * base_tile_k * sizeof(output_t);
    
    // Total per-buffer (one X + one Y) at multiplier=1
    static constexpr int base_smem_per_buffer = base_smem_x + base_smem_y;
    
    // For TC path, we need 2 buffers (A/B flip). Check if even multiplier=1 fits.
    static constexpr bool can_double_buffer = USE_TC && (2 * base_smem_per_buffer <= smem_available);
    
    // TC path uses double buffering if possible, scalar path uses 1 for tile sizing math
    // If TC requested but can't double buffer, we'll fall back to scalar at dispatch
    static constexpr int num_buffers = can_double_buffer ? 2 : 1;
    
    // Multiplier limited by smem: smem_available / (num_buffers * base_smem_per_buffer)
    // For scalar path, we don't use smem buffers so this is unlimited (use size limit)
    static constexpr int smem_limited_multiplier = 
        USE_TC ? (smem_available / (num_buffers * base_smem_per_buffer)) : MAX_TILE_K_ELEMENTS;
    
    // Multiplier limited by max tile_k: MAX_TILE_K_ELEMENTS / base_tile_k
    static constexpr int size_limited_multiplier = MAX_TILE_K_ELEMENTS / base_tile_k;
    
    // Take the minimum of both limits, ensure at least 1
    static constexpr int raw_multiplier = (smem_limited_multiplier < size_limited_multiplier)
                                         ? smem_limited_multiplier : size_limited_multiplier;
    static constexpr int tile_multiplier = (raw_multiplier > 0) ? raw_multiplier : 1;
    
    // Final tile sizes with multiplier applied
    static constexpr int tile_k = tile_base_k_v<block_t, MMA_K> * tile_multiplier;
    static constexpr int tile_blocks = tile_base_blocks_v<block_t, MMA_K> * tile_multiplier;
    static constexpr int tile_bytes = tile_base_bytes_v<block_t, MMA_K> * tile_multiplier;
    
    // Smem sizes with multiplier
    static constexpr int tile_smem_y = base_smem_y * tile_multiplier;
    static constexpr int tile_smem_x = base_smem_x * tile_multiplier;
    static constexpr int tile_smem_per_buffer = tile_smem_x + tile_smem_y;
    
    // Legacy alias for compatibility
    static constexpr int tile_smem_bytes = tile_smem_y;  // Just Y for backward compat
    
    // Total smem used for pipeline buffers (0 for scalar path since it doesn't use smem)
    static constexpr int pipeline_smem_bytes = USE_TC ? (num_buffers * tile_smem_per_buffer) : 0;
    
    // Static assertions for pipeline_config invariants
    static_assert(tile_multiplier >= 1, "tile_multiplier must be at least 1");
    static_assert(tile_k > 0, "tile_k must be positive");
    static_assert(tile_k <= MAX_TILE_K_ELEMENTS, "tile_k exceeds MAX_TILE_K_ELEMENTS cap");
    static_assert(tile_blocks > 0, "tile_blocks must be positive");
    // Only assert smem fits when we can actually double buffer (will fall back to scalar otherwise)
    static_assert(!can_double_buffer || pipeline_smem_bytes <= smem_available, "Pipeline buffers exceed available smem");
    static_assert(base_smem_per_buffer > 0, "base_smem_per_buffer must be positive");
    static_assert(num_buffers >= 1, "need at least 1 buffer");
};

// =============================================================================
// HELPER ALIASES FOR PIPELINE CONFIG
// =============================================================================
// These provide convenient access to pipeline_config computed values

template <typename block_t, typename output_t, bool USE_TC, int MMA_K = MMA_K_FP8>
constexpr int tile_multiplier_v = pipeline_config<block_t, output_t, USE_TC, MMA_K>::tile_multiplier;

template <typename block_t, typename output_t, bool USE_TC, int MMA_K = MMA_K_FP8>
constexpr int tile_k_v = pipeline_config<block_t, output_t, USE_TC, MMA_K>::tile_k;

template <typename block_t, typename output_t, bool USE_TC, int MMA_K = MMA_K_FP8>
constexpr int tile_k_blocks_v = pipeline_config<block_t, output_t, USE_TC, MMA_K>::tile_blocks;

template <typename block_t, typename output_t, bool USE_TC, int MMA_K = MMA_K_FP8>
constexpr int tile_k_bytes_v = pipeline_config<block_t, output_t, USE_TC, MMA_K>::tile_bytes;

template <typename block_t, typename output_t, bool USE_TC, int MMA_K = MMA_K_FP8>
constexpr int tile_k_smem_bytes_v = pipeline_config<block_t, output_t, USE_TC, MMA_K>::tile_smem_bytes;

template <typename block_t, typename output_t, bool USE_TC, int MMA_K = MMA_K_FP8>
constexpr int pipeline_num_buffers_v = pipeline_config<block_t, output_t, USE_TC, MMA_K>::num_buffers;

template <typename block_t, typename output_t, bool USE_TC, int MMA_K = MMA_K_FP8>
constexpr int pipeline_smem_bytes_v = pipeline_config<block_t, output_t, USE_TC, MMA_K>::pipeline_smem_bytes;

template <typename block_t, typename output_t, bool USE_TC, int MMA_K = MMA_K_FP8>
constexpr bool can_double_buffer_v = pipeline_config<block_t, output_t, USE_TC, MMA_K>::can_double_buffer;
