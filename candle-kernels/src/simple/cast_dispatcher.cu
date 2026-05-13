// =============================================================================
// CAST OPERATIONS DISPATCHER (Regular + In-Place)
// =============================================================================
// Provides extern "C" entry points for:
// 1. run_cast: Regular cast (separate input/output buffers)
// 2. run_cast_mut: In-place cast (single buffer, cooperative launch support)
//
// Dtype order: f32=0, f64=1, u8=2, u32=3, i64=4, f16=5, bf16=6, f8_e4m3=7
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// =============================================================================
// Configuration Constants
// =============================================================================

// Block sizes optimized for different workloads
// 256 threads provides good occupancy across GPU architectures while allowing
// sufficient registers per thread for complex type conversions (especially FP8)
constexpr int CAST_BLOCK_SIZE = 256;          // Regular cast
constexpr int CAST_MUT_BLOCK_SIZE = 256;      // In-place cast

// Small tensor threshold - use single block fast path
// Below this size, cooperative launch overhead exceeds parallel speedup.
// Tuned based on typical kernel launch latency (~5-10us) vs compute.
// With 256 threads, 2048 elements gives each thread ~8 elements to process,
// providing enough work to amortize __syncthreads() overhead.
constexpr size_t SMALL_TENSOR_THRESHOLD = 2048;

// =============================================================================
// Function pointer type for in-place casts
// =============================================================================

// In-place cast signature
using cast_mut_fn_t = void (*)(size_t numel, void* buf);

// =============================================================================
// PART 1: REGULAR CAST FORWARD DECLARATIONS
// =============================================================================
// These are __global__ kernels defined in cast.cu

// Identity casts
extern "C" __global__ void cast_u8_u8(size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void cast_u32_u32(size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void cast_i64_i64(size_t, size_t, const size_t*, const int64_t*, int64_t*);
extern "C" __global__ void cast_f32_f32(size_t, size_t, const size_t*, const float*, float*);
extern "C" __global__ void cast_f64_f64(size_t, size_t, const size_t*, const double*, double*);
extern "C" __global__ void cast_f16_f16(size_t, size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void cast_bf16_bf16(size_t, size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void cast_f8_e4m3_f8_e4m3(size_t, size_t, const size_t*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);

// From u8
extern "C" __global__ void cast_u8_u32(size_t, size_t, const size_t*, const uint8_t*, uint32_t*);
extern "C" __global__ void cast_u8_i64(size_t, size_t, const size_t*, const uint8_t*, int64_t*);
extern "C" __global__ void cast_u8_f32(size_t, size_t, const size_t*, const uint8_t*, float*);
extern "C" __global__ void cast_u8_f64(size_t, size_t, const size_t*, const uint8_t*, double*);
extern "C" __global__ void cast_u8_f16(size_t, size_t, const size_t*, const uint8_t*, __half*);
extern "C" __global__ void cast_u8_bf16(size_t, size_t, const size_t*, const uint8_t*, __nv_bfloat16*);
extern "C" __global__ void cast_u8_f8_e4m3(size_t, size_t, const size_t*, const uint8_t*, __nv_fp8_e4m3*);

// From u32
extern "C" __global__ void cast_u32_u8(size_t, size_t, const size_t*, const uint32_t*, uint8_t*);
extern "C" __global__ void cast_u32_i64(size_t, size_t, const size_t*, const uint32_t*, int64_t*);
extern "C" __global__ void cast_u32_f32(size_t, size_t, const size_t*, const uint32_t*, float*);
extern "C" __global__ void cast_u32_f64(size_t, size_t, const size_t*, const uint32_t*, double*);
extern "C" __global__ void cast_u32_f16(size_t, size_t, const size_t*, const uint32_t*, __half*);
extern "C" __global__ void cast_u32_bf16(size_t, size_t, const size_t*, const uint32_t*, __nv_bfloat16*);

// From i64
extern "C" __global__ void cast_i64_u8(size_t, size_t, const size_t*, const int64_t*, uint8_t*);
extern "C" __global__ void cast_i64_u32(size_t, size_t, const size_t*, const int64_t*, uint32_t*);
extern "C" __global__ void cast_i64_f32(size_t, size_t, const size_t*, const int64_t*, float*);
extern "C" __global__ void cast_i64_f64(size_t, size_t, const size_t*, const int64_t*, double*);
extern "C" __global__ void cast_i64_f16(size_t, size_t, const size_t*, const int64_t*, __half*);
extern "C" __global__ void cast_i64_bf16(size_t, size_t, const size_t*, const int64_t*, __nv_bfloat16*);

// From f32
extern "C" __global__ void cast_f32_u8(size_t, size_t, const size_t*, const float*, uint8_t*);
extern "C" __global__ void cast_f32_u32(size_t, size_t, const size_t*, const float*, uint32_t*);
extern "C" __global__ void cast_f32_i64(size_t, size_t, const size_t*, const float*, int64_t*);
extern "C" __global__ void cast_f32_f64(size_t, size_t, const size_t*, const float*, double*);
extern "C" __global__ void cast_f32_f16(size_t, size_t, const size_t*, const float*, __half*);
extern "C" __global__ void cast_f32_bf16(size_t, size_t, const size_t*, const float*, __nv_bfloat16*);
extern "C" __global__ void cast_f32_f8_e4m3(size_t, size_t, const size_t*, const float*, __nv_fp8_e4m3*);

// From f64
extern "C" __global__ void cast_f64_u8(size_t, size_t, const size_t*, const double*, uint8_t*);
extern "C" __global__ void cast_f64_u32(size_t, size_t, const size_t*, const double*, uint32_t*);
extern "C" __global__ void cast_f64_i64(size_t, size_t, const size_t*, const double*, int64_t*);
extern "C" __global__ void cast_f64_f32(size_t, size_t, const size_t*, const double*, float*);
extern "C" __global__ void cast_f64_f16(size_t, size_t, const size_t*, const double*, __half*);
extern "C" __global__ void cast_f64_bf16(size_t, size_t, const size_t*, const double*, __nv_bfloat16*);
extern "C" __global__ void cast_f64_f8_e4m3(size_t, size_t, const size_t*, const double*, __nv_fp8_e4m3*);

// From f16
extern "C" __global__ void cast_f16_u8(size_t, size_t, const size_t*, const __half*, uint8_t*);
extern "C" __global__ void cast_f16_u32(size_t, size_t, const size_t*, const __half*, uint32_t*);
extern "C" __global__ void cast_f16_f32(size_t, size_t, const size_t*, const __half*, float*);
extern "C" __global__ void cast_f16_f64(size_t, size_t, const size_t*, const __half*, double*);
extern "C" __global__ void cast_f16_bf16(size_t, size_t, const size_t*, const __half*, __nv_bfloat16*);
extern "C" __global__ void cast_f16_f8_e4m3(size_t, size_t, const size_t*, const __half*, __nv_fp8_e4m3*);

// From bf16
extern "C" __global__ void cast_bf16_u8(size_t, size_t, const size_t*, const __nv_bfloat16*, uint8_t*);
extern "C" __global__ void cast_bf16_u32(size_t, size_t, const size_t*, const __nv_bfloat16*, uint32_t*);
extern "C" __global__ void cast_bf16_f32(size_t, size_t, const size_t*, const __nv_bfloat16*, float*);
extern "C" __global__ void cast_bf16_f64(size_t, size_t, const size_t*, const __nv_bfloat16*, double*);
extern "C" __global__ void cast_bf16_f16(size_t, size_t, const size_t*, const __nv_bfloat16*, __half*);
extern "C" __global__ void cast_bf16_f8_e4m3(size_t, size_t, const size_t*, const __nv_bfloat16*, __nv_fp8_e4m3*);

// From f8_e4m3
extern "C" __global__ void cast_f8_e4m3_u8(size_t, size_t, const size_t*, const __nv_fp8_e4m3*, uint8_t*);
extern "C" __global__ void cast_f8_e4m3_u32(size_t, size_t, const size_t*, const __nv_fp8_e4m3*, uint32_t*);
extern "C" __global__ void cast_f8_e4m3_i64(size_t, size_t, const size_t*, const __nv_fp8_e4m3*, int64_t*);
extern "C" __global__ void cast_f8_e4m3_f32(size_t, size_t, const size_t*, const __nv_fp8_e4m3*, float*);
extern "C" __global__ void cast_f8_e4m3_f64(size_t, size_t, const size_t*, const __nv_fp8_e4m3*, double*);
extern "C" __global__ void cast_f8_e4m3_f16(size_t, size_t, const size_t*, const __nv_fp8_e4m3*, __half*);
extern "C" __global__ void cast_f8_e4m3_bf16(size_t, size_t, const size_t*, const __nv_fp8_e4m3*, __nv_bfloat16*);

// To f8_e4m3 (additional)
extern "C" __global__ void cast_u32_f8_e4m3(size_t, size_t, const size_t*, const uint32_t*, __nv_fp8_e4m3*);
extern "C" __global__ void cast_i64_f8_e4m3(size_t, size_t, const size_t*, const int64_t*, __nv_fp8_e4m3*);

// =============================================================================
// PART 2: IN-PLACE CAST FORWARD DECLARATIONS
// =============================================================================

// --- Identity casts ---
extern "C" __global__ void cast_mut_u8_u8(size_t, void*);
extern "C" __global__ void cast_mut_u32_u32(size_t, void*);
extern "C" __global__ void cast_mut_i64_i64(size_t, void*);
extern "C" __global__ void cast_mut_f32_f32(size_t, void*);
extern "C" __global__ void cast_mut_f64_f64(size_t, void*);
extern "C" __global__ void cast_mut_f16_f16(size_t, void*);
extern "C" __global__ void cast_mut_bf16_bf16(size_t, void*);
extern "C" __global__ void cast_mut_f8_e4m3_f8_e4m3(size_t, void*);

// --- From u8 ---
extern "C" __global__ void cast_mut_u8_u32(size_t, void*);
extern "C" __global__ void cast_mut_u8_i64(size_t, void*);
extern "C" __global__ void cast_mut_u8_f32(size_t, void*);
extern "C" __global__ void cast_mut_u8_f64(size_t, void*);
extern "C" __global__ void cast_mut_u8_f16(size_t, void*);
extern "C" __global__ void cast_mut_u8_bf16(size_t, void*);
extern "C" __global__ void cast_mut_u8_f8_e4m3(size_t, void*);

// --- From u32 ---
extern "C" __global__ void cast_mut_u32_u8(size_t, void*);
extern "C" __global__ void cast_mut_u32_i64(size_t, void*);
extern "C" __global__ void cast_mut_u32_f32(size_t, void*);
extern "C" __global__ void cast_mut_u32_f64(size_t, void*);
extern "C" __global__ void cast_mut_u32_f16(size_t, void*);
extern "C" __global__ void cast_mut_u32_bf16(size_t, void*);
extern "C" __global__ void cast_mut_u32_f8_e4m3(size_t, void*);

// --- From i64 ---
extern "C" __global__ void cast_mut_i64_u8(size_t, void*);
extern "C" __global__ void cast_mut_i64_u32(size_t, void*);
extern "C" __global__ void cast_mut_i64_f32(size_t, void*);
extern "C" __global__ void cast_mut_i64_f64(size_t, void*);
extern "C" __global__ void cast_mut_i64_f16(size_t, void*);
extern "C" __global__ void cast_mut_i64_bf16(size_t, void*);

// --- From f32 ---
extern "C" __global__ void cast_mut_f32_u8(size_t, void*);
extern "C" __global__ void cast_mut_f32_u32(size_t, void*);
extern "C" __global__ void cast_mut_f32_i64(size_t, void*);
extern "C" __global__ void cast_mut_f32_f64(size_t, void*);
extern "C" __global__ void cast_mut_f32_f16(size_t, void*);
extern "C" __global__ void cast_mut_f32_bf16(size_t, void*);
extern "C" __global__ void cast_mut_f32_f8_e4m3(size_t, void*);

// --- From f64 ---
extern "C" __global__ void cast_mut_f64_u8(size_t, void*);
extern "C" __global__ void cast_mut_f64_u32(size_t, void*);
extern "C" __global__ void cast_mut_f64_i64(size_t, void*);
extern "C" __global__ void cast_mut_f64_f32(size_t, void*);
extern "C" __global__ void cast_mut_f64_f16(size_t, void*);
extern "C" __global__ void cast_mut_f64_bf16(size_t, void*);
extern "C" __global__ void cast_mut_f64_f8_e4m3(size_t, void*);

// --- From f16 ---
extern "C" __global__ void cast_mut_f16_u8(size_t, void*);
extern "C" __global__ void cast_mut_f16_u32(size_t, void*);
extern "C" __global__ void cast_mut_f16_i64(size_t, void*);
extern "C" __global__ void cast_mut_f16_f32(size_t, void*);
extern "C" __global__ void cast_mut_f16_f64(size_t, void*);
extern "C" __global__ void cast_mut_f16_bf16(size_t, void*);
extern "C" __global__ void cast_mut_f16_f8_e4m3(size_t, void*);

// --- From bf16 ---
extern "C" __global__ void cast_mut_bf16_u8(size_t, void*);
extern "C" __global__ void cast_mut_bf16_u32(size_t, void*);
extern "C" __global__ void cast_mut_bf16_i64(size_t, void*);
extern "C" __global__ void cast_mut_bf16_f16(size_t, void*);
extern "C" __global__ void cast_mut_bf16_f32(size_t, void*);
extern "C" __global__ void cast_mut_bf16_f64(size_t, void*);
extern "C" __global__ void cast_mut_bf16_f8_e4m3(size_t, void*);

// --- From f8_e4m3 ---
extern "C" __global__ void cast_mut_f8_e4m3_u8(size_t, void*);
extern "C" __global__ void cast_mut_f8_e4m3_u32(size_t, void*);
extern "C" __global__ void cast_mut_f8_e4m3_i64(size_t, void*);
extern "C" __global__ void cast_mut_f8_e4m3_f16(size_t, void*);
extern "C" __global__ void cast_mut_f8_e4m3_bf16(size_t, void*);
extern "C" __global__ void cast_mut_f8_e4m3_f32(size_t, void*);
extern "C" __global__ void cast_mut_f8_e4m3_f64(size_t, void*);

// =============================================================================
// PART 3: COOPERATIVE GRID KERNEL FORWARD DECLARATIONS (_coop suffix)
// =============================================================================

// --- Identity casts ---
extern "C" __global__ void cast_mut_u8_u8_coop(size_t, void*);
extern "C" __global__ void cast_mut_u32_u32_coop(size_t, void*);
extern "C" __global__ void cast_mut_i64_i64_coop(size_t, void*);
extern "C" __global__ void cast_mut_f32_f32_coop(size_t, void*);
extern "C" __global__ void cast_mut_f64_f64_coop(size_t, void*);
extern "C" __global__ void cast_mut_f16_f16_coop(size_t, void*);
extern "C" __global__ void cast_mut_bf16_bf16_coop(size_t, void*);
extern "C" __global__ void cast_mut_f8_e4m3_f8_e4m3_coop(size_t, void*);

// --- From u8 ---
extern "C" __global__ void cast_mut_u8_u32_coop(size_t, void*);
extern "C" __global__ void cast_mut_u8_i64_coop(size_t, void*);
extern "C" __global__ void cast_mut_u8_f32_coop(size_t, void*);
extern "C" __global__ void cast_mut_u8_f64_coop(size_t, void*);
extern "C" __global__ void cast_mut_u8_f16_coop(size_t, void*);
extern "C" __global__ void cast_mut_u8_bf16_coop(size_t, void*);
extern "C" __global__ void cast_mut_u8_f8_e4m3_coop(size_t, void*);

// --- From u32 ---
extern "C" __global__ void cast_mut_u32_u8_coop(size_t, void*);
extern "C" __global__ void cast_mut_u32_i64_coop(size_t, void*);
extern "C" __global__ void cast_mut_u32_f32_coop(size_t, void*);
extern "C" __global__ void cast_mut_u32_f64_coop(size_t, void*);
extern "C" __global__ void cast_mut_u32_f16_coop(size_t, void*);
extern "C" __global__ void cast_mut_u32_bf16_coop(size_t, void*);
extern "C" __global__ void cast_mut_u32_f8_e4m3_coop(size_t, void*);

// --- From i64 ---
extern "C" __global__ void cast_mut_i64_u8_coop(size_t, void*);
extern "C" __global__ void cast_mut_i64_u32_coop(size_t, void*);
extern "C" __global__ void cast_mut_i64_f32_coop(size_t, void*);
extern "C" __global__ void cast_mut_i64_f64_coop(size_t, void*);
extern "C" __global__ void cast_mut_i64_f16_coop(size_t, void*);
extern "C" __global__ void cast_mut_i64_bf16_coop(size_t, void*);

// --- From f32 ---
extern "C" __global__ void cast_mut_f32_u8_coop(size_t, void*);
extern "C" __global__ void cast_mut_f32_u32_coop(size_t, void*);
extern "C" __global__ void cast_mut_f32_i64_coop(size_t, void*);
extern "C" __global__ void cast_mut_f32_f64_coop(size_t, void*);
extern "C" __global__ void cast_mut_f32_f16_coop(size_t, void*);
extern "C" __global__ void cast_mut_f32_bf16_coop(size_t, void*);
extern "C" __global__ void cast_mut_f32_f8_e4m3_coop(size_t, void*);

// --- From f64 ---
extern "C" __global__ void cast_mut_f64_u8_coop(size_t, void*);
extern "C" __global__ void cast_mut_f64_u32_coop(size_t, void*);
extern "C" __global__ void cast_mut_f64_i64_coop(size_t, void*);
extern "C" __global__ void cast_mut_f64_f32_coop(size_t, void*);
extern "C" __global__ void cast_mut_f64_f16_coop(size_t, void*);
extern "C" __global__ void cast_mut_f64_bf16_coop(size_t, void*);
extern "C" __global__ void cast_mut_f64_f8_e4m3_coop(size_t, void*);

// --- From f16 ---
extern "C" __global__ void cast_mut_f16_u8_coop(size_t, void*);
extern "C" __global__ void cast_mut_f16_u32_coop(size_t, void*);
extern "C" __global__ void cast_mut_f16_i64_coop(size_t, void*);
extern "C" __global__ void cast_mut_f16_f32_coop(size_t, void*);
extern "C" __global__ void cast_mut_f16_f64_coop(size_t, void*);
extern "C" __global__ void cast_mut_f16_bf16_coop(size_t, void*);
extern "C" __global__ void cast_mut_f16_f8_e4m3_coop(size_t, void*);

// --- From bf16 ---
extern "C" __global__ void cast_mut_bf16_u8_coop(size_t, void*);
extern "C" __global__ void cast_mut_bf16_u32_coop(size_t, void*);
extern "C" __global__ void cast_mut_bf16_i64_coop(size_t, void*);
extern "C" __global__ void cast_mut_bf16_f16_coop(size_t, void*);
extern "C" __global__ void cast_mut_bf16_f32_coop(size_t, void*);
extern "C" __global__ void cast_mut_bf16_f64_coop(size_t, void*);
extern "C" __global__ void cast_mut_bf16_f8_e4m3_coop(size_t, void*);

// --- From f8_e4m3 ---
extern "C" __global__ void cast_mut_f8_e4m3_u8_coop(size_t, void*);
extern "C" __global__ void cast_mut_f8_e4m3_u32_coop(size_t, void*);
extern "C" __global__ void cast_mut_f8_e4m3_i64_coop(size_t, void*);
extern "C" __global__ void cast_mut_f8_e4m3_f16_coop(size_t, void*);
extern "C" __global__ void cast_mut_f8_e4m3_bf16_coop(size_t, void*);
extern "C" __global__ void cast_mut_f8_e4m3_f32_coop(size_t, void*);
extern "C" __global__ void cast_mut_f8_e4m3_f64_coop(size_t, void*);

// =============================================================================
// PART 4: KERNEL PAIR STRUCTURE FOR IN-PLACE CAST
// =============================================================================

struct cast_mut_kernel_pair {
    cast_mut_fn_t fallback;  // Single-block version
    cast_mut_fn_t coop;      // Cooperative grid version
};

// =============================================================================
// PART 5: HELPER FUNCTIONS
// =============================================================================

// Check if cooperative launch is supported on current device (cached)
__host__ static bool supports_cooperative_launch() {
    static int cached_result = -1;  // -1 = not checked, 0 = no, 1 = yes
    
    if (cached_result >= 0) {
        return cached_result == 1;
    }
    
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) {
        cached_result = 0;
        return false;
    }
    
    int supports_coop = 0;
    if (cudaDeviceGetAttribute(&supports_coop, cudaDevAttrCooperativeLaunch, device) != cudaSuccess) {
        cached_result = 0;
        return false;
    }
    
    cached_result = supports_coop ? 1 : 0;
    return cached_result == 1;
}

// Get optimal block count for cooperative launch
__host__ static int get_coop_block_count(cast_mut_fn_t kernel) {
    int device;
    cudaGetDevice(&device);
    
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, CAST_MUT_BLOCK_SIZE, 0);
    
    return num_sms * max_blocks_per_sm;
}

// Launch in-place cast kernel with cooperative or fallback mode
__host__ static void launch_cast_mut(
    const cast_mut_kernel_pair& kernels,
    size_t numel,
    void* buf
) {
    if (kernels.fallback == nullptr) {
        return;  // Unsupported conversion
    }
    
    // Fast path for small tensors - single block is more efficient
    if (numel <= SMALL_TENSOR_THRESHOLD) {
        kernels.fallback<<<1, CAST_MUT_BLOCK_SIZE>>>(numel, buf);
        return;
    }
    
    // Try cooperative launch if supported and kernel available
    if (kernels.coop != nullptr && supports_cooperative_launch()) {
        int num_blocks = get_coop_block_count(kernels.coop);
        
        // Ensure we don't launch more blocks than needed
        int blocks_needed = (numel + CAST_MUT_BLOCK_SIZE - 1) / CAST_MUT_BLOCK_SIZE;
        num_blocks = min(num_blocks, blocks_needed);
        num_blocks = max(num_blocks, 1);
        
        void* args[] = { &numel, &buf };
        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)kernels.coop,
            dim3(num_blocks),
            dim3(CAST_MUT_BLOCK_SIZE),
            args,
            0,  // shared memory
            0   // stream (default)
        );
        
        if (err == cudaSuccess) {
            return;  // Cooperative launch succeeded
        }
        // Fall through to single-block fallback
    }
    
    // Single-block fallback
    kernels.fallback<<<1, CAST_MUT_BLOCK_SIZE>>>(numel, buf);
}

// =============================================================================
// PART 6: LOOKUP TABLES (for cast_mut only)
// =============================================================================

// Note: Regular cast uses explicit switch-case dispatch in run_cast()
// due to CUDA kernel launch syntax requirements.
// Only cast_mut uses lookup tables since it launches kernels differently.

// In-place cast lookup table: [src_dtype][dst_dtype]
static const cast_mut_kernel_pair CAST_MUT_KERNELS[8][8] = {
    // From f32 (src_dtype=0)
    {
        {cast_mut_f32_f32, cast_mut_f32_f32_coop},           // -> f32
        {cast_mut_f32_f64, cast_mut_f32_f64_coop},           // -> f64
        {cast_mut_f32_u8, cast_mut_f32_u8_coop},             // -> u8
        {cast_mut_f32_u32, cast_mut_f32_u32_coop},           // -> u32
        {cast_mut_f32_i64, cast_mut_f32_i64_coop},           // -> i64
        {cast_mut_f32_f16, cast_mut_f32_f16_coop},           // -> f16
        {cast_mut_f32_bf16, cast_mut_f32_bf16_coop},         // -> bf16
        {cast_mut_f32_f8_e4m3, cast_mut_f32_f8_e4m3_coop},   // -> f8_e4m3
    },
    // From f64 (src_dtype=1)
    {
        {cast_mut_f64_f32, cast_mut_f64_f32_coop},           // -> f32
        {cast_mut_f64_f64, cast_mut_f64_f64_coop},           // -> f64
        {cast_mut_f64_u8, cast_mut_f64_u8_coop},             // -> u8
        {cast_mut_f64_u32, cast_mut_f64_u32_coop},           // -> u32
        {cast_mut_f64_i64, cast_mut_f64_i64_coop},           // -> i64
        {cast_mut_f64_f16, cast_mut_f64_f16_coop},           // -> f16
        {cast_mut_f64_bf16, cast_mut_f64_bf16_coop},         // -> bf16
        {cast_mut_f64_f8_e4m3, cast_mut_f64_f8_e4m3_coop},   // -> f8_e4m3
    },
    // From u8 (src_dtype=2)
    {
        {cast_mut_u8_f32, cast_mut_u8_f32_coop},             // -> f32
        {cast_mut_u8_f64, cast_mut_u8_f64_coop},             // -> f64
        {cast_mut_u8_u8, cast_mut_u8_u8_coop},               // -> u8
        {cast_mut_u8_u32, cast_mut_u8_u32_coop},             // -> u32
        {cast_mut_u8_i64, cast_mut_u8_i64_coop},             // -> i64
        {cast_mut_u8_f16, cast_mut_u8_f16_coop},             // -> f16
        {cast_mut_u8_bf16, cast_mut_u8_bf16_coop},           // -> bf16
        {cast_mut_u8_f8_e4m3, cast_mut_u8_f8_e4m3_coop},     // -> f8_e4m3
    },
    // From u32 (src_dtype=3)
    {
        {cast_mut_u32_f32, cast_mut_u32_f32_coop},           // -> f32
        {cast_mut_u32_f64, cast_mut_u32_f64_coop},           // -> f64
        {cast_mut_u32_u8, cast_mut_u32_u8_coop},             // -> u8
        {cast_mut_u32_u32, cast_mut_u32_u32_coop},           // -> u32
        {cast_mut_u32_i64, cast_mut_u32_i64_coop},           // -> i64
        {cast_mut_u32_f16, cast_mut_u32_f16_coop},           // -> f16
        {cast_mut_u32_bf16, cast_mut_u32_bf16_coop},         // -> bf16
        {cast_mut_u32_f8_e4m3, cast_mut_u32_f8_e4m3_coop},   // -> f8_e4m3
    },
    // From i64 (src_dtype=4)
    {
        {cast_mut_i64_f32, cast_mut_i64_f32_coop},           // -> f32
        {cast_mut_i64_f64, cast_mut_i64_f64_coop},           // -> f64
        {cast_mut_i64_u8, cast_mut_i64_u8_coop},             // -> u8
        {cast_mut_i64_u32, cast_mut_i64_u32_coop},           // -> u32
        {cast_mut_i64_i64, cast_mut_i64_i64_coop},           // -> i64
        {cast_mut_i64_f16, cast_mut_i64_f16_coop},           // -> f16
        {cast_mut_i64_bf16, cast_mut_i64_bf16_coop},         // -> bf16
        {nullptr, nullptr},                                   // -> f8_e4m3 (not supported)
    },
    // From f16 (src_dtype=5)
    {
        {cast_mut_f16_f32, cast_mut_f16_f32_coop},           // -> f32
        {cast_mut_f16_f64, cast_mut_f16_f64_coop},           // -> f64
        {cast_mut_f16_u8, cast_mut_f16_u8_coop},             // -> u8
        {cast_mut_f16_u32, cast_mut_f16_u32_coop},           // -> u32
        {cast_mut_f16_i64, cast_mut_f16_i64_coop},           // -> i64
        {cast_mut_f16_f16, cast_mut_f16_f16_coop},           // -> f16
        {cast_mut_f16_bf16, cast_mut_f16_bf16_coop},         // -> bf16
        {cast_mut_f16_f8_e4m3, cast_mut_f16_f8_e4m3_coop},   // -> f8_e4m3
    },
    // From bf16 (src_dtype=6)
    {
        {cast_mut_bf16_f32, cast_mut_bf16_f32_coop},         // -> f32
        {cast_mut_bf16_f64, cast_mut_bf16_f64_coop},         // -> f64
        {cast_mut_bf16_u8, cast_mut_bf16_u8_coop},           // -> u8
        {cast_mut_bf16_u32, cast_mut_bf16_u32_coop},         // -> u32
        {cast_mut_bf16_i64, cast_mut_bf16_i64_coop},         // -> i64
        {cast_mut_bf16_f16, cast_mut_bf16_f16_coop},         // -> f16
        {cast_mut_bf16_bf16, cast_mut_bf16_bf16_coop},       // -> bf16
        {cast_mut_bf16_f8_e4m3, cast_mut_bf16_f8_e4m3_coop}, // -> f8_e4m3
    },
    // From f8_e4m3 (src_dtype=7)
    {
        {cast_mut_f8_e4m3_f32, cast_mut_f8_e4m3_f32_coop},   // -> f32
        {cast_mut_f8_e4m3_f64, cast_mut_f8_e4m3_f64_coop},   // -> f64
        {cast_mut_f8_e4m3_u8, cast_mut_f8_e4m3_u8_coop},     // -> u8
        {cast_mut_f8_e4m3_u32, cast_mut_f8_e4m3_u32_coop},   // -> u32
        {cast_mut_f8_e4m3_i64, cast_mut_f8_e4m3_i64_coop},   // -> i64
        {cast_mut_f8_e4m3_f16, cast_mut_f8_e4m3_f16_coop},   // -> f16
        {cast_mut_f8_e4m3_bf16, cast_mut_f8_e4m3_bf16_coop}, // -> bf16
        {cast_mut_f8_e4m3_f8_e4m3, cast_mut_f8_e4m3_f8_e4m3_coop}, // -> f8_e4m3
    },
};

// =============================================================================
// PART 7: DISPATCHER FUNCTIONS
// =============================================================================

// Block size for kernel launches
constexpr int CAST_BLOCK_SIZE_LAUNCH = 256;

// Helper to compute grid size
inline int cast_grid_size(size_t numel) {
    return (numel + CAST_BLOCK_SIZE_LAUNCH - 1) / CAST_BLOCK_SIZE_LAUNCH;
}

// Regular cast dispatcher - uses switch statements for proper kernel launches
// dtype order: f32=0, f64=1, u8=2, u32=3, i64=4, f16=5, bf16=6, f8_e4m3=7
extern "C" void run_cast(
    int32_t src_dtype,
    int32_t dst_dtype,
    size_t numel,
    size_t num_dims,
    const size_t* info,
    const void* inp,
    void* out
) {
    int grid = cast_grid_size(numel);
    
    // Identity casts
    if (src_dtype == dst_dtype) {
        switch (src_dtype) {
            case 0: cast_f32_f32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const float*)inp, (float*)out); break;
            case 1: cast_f64_f64<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const double*)inp, (double*)out); break;
            case 2: cast_u8_u8<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint8_t*)inp, (uint8_t*)out); break;
            case 3: cast_u32_u32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint32_t*)inp, (uint32_t*)out); break;
            case 4: cast_i64_i64<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const int64_t*)inp, (int64_t*)out); break;
            case 5: cast_f16_f16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __half*)inp, (__half*)out); break;
            case 6: cast_bf16_bf16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, (__nv_bfloat16*)out); break;
            case 7: cast_f8_e4m3_f8_e4m3<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_fp8_e4m3*)inp, (__nv_fp8_e4m3*)out); break;
        }
        return;
    }
    
    // From f32
    if (src_dtype == 0) {
        switch (dst_dtype) {
            case 1: cast_f32_f64<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const float*)inp, (double*)out); break;
            case 2: cast_f32_u8<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const float*)inp, (uint8_t*)out); break;
            case 3: cast_f32_u32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const float*)inp, (uint32_t*)out); break;
            case 4: cast_f32_i64<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const float*)inp, (int64_t*)out); break;
            case 5: cast_f32_f16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const float*)inp, (__half*)out); break;
            case 6: cast_f32_bf16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const float*)inp, (__nv_bfloat16*)out); break;
            case 7: cast_f32_f8_e4m3<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const float*)inp, (__nv_fp8_e4m3*)out); break;
        }
        return;
    }
    // From f64
    if (src_dtype == 1) {
        switch (dst_dtype) {
            case 0: cast_f64_f32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const double*)inp, (float*)out); break;
            case 2: cast_f64_u8<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const double*)inp, (uint8_t*)out); break;
            case 3: cast_f64_u32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const double*)inp, (uint32_t*)out); break;
            case 4: cast_f64_i64<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const double*)inp, (int64_t*)out); break;
            case 5: cast_f64_f16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const double*)inp, (__half*)out); break;
            case 6: cast_f64_bf16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const double*)inp, (__nv_bfloat16*)out); break;
            case 7: cast_f64_f8_e4m3<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const double*)inp, (__nv_fp8_e4m3*)out); break;
        }
        return;
    }
    // From u8
    if (src_dtype == 2) {
        switch (dst_dtype) {
            case 0: cast_u8_f32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint8_t*)inp, (float*)out); break;
            case 1: cast_u8_f64<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint8_t*)inp, (double*)out); break;
            case 3: cast_u8_u32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint8_t*)inp, (uint32_t*)out); break;
            case 4: cast_u8_i64<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint8_t*)inp, (int64_t*)out); break;
            case 5: cast_u8_f16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint8_t*)inp, (__half*)out); break;
            case 6: cast_u8_bf16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint8_t*)inp, (__nv_bfloat16*)out); break;
            case 7: cast_u8_f8_e4m3<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint8_t*)inp, (__nv_fp8_e4m3*)out); break;
        }
        return;
    }
    // From u32
    if (src_dtype == 3) {
        switch (dst_dtype) {
            case 0: cast_u32_f32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint32_t*)inp, (float*)out); break;
            case 1: cast_u32_f64<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint32_t*)inp, (double*)out); break;
            case 2: cast_u32_u8<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint32_t*)inp, (uint8_t*)out); break;
            case 4: cast_u32_i64<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint32_t*)inp, (int64_t*)out); break;
            case 5: cast_u32_f16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint32_t*)inp, (__half*)out); break;
            case 6: cast_u32_bf16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint32_t*)inp, (__nv_bfloat16*)out); break;
            case 7: cast_u32_f8_e4m3<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const uint32_t*)inp, (__nv_fp8_e4m3*)out); break;
        }
        return;
    }
    // From i64
    if (src_dtype == 4) {
        switch (dst_dtype) {
            case 0: cast_i64_f32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const int64_t*)inp, (float*)out); break;
            case 1: cast_i64_f64<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const int64_t*)inp, (double*)out); break;
            case 2: cast_i64_u8<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const int64_t*)inp, (uint8_t*)out); break;
            case 3: cast_i64_u32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const int64_t*)inp, (uint32_t*)out); break;
            case 5: cast_i64_f16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const int64_t*)inp, (__half*)out); break;
            case 6: cast_i64_bf16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const int64_t*)inp, (__nv_bfloat16*)out); break;
            case 7: cast_i64_f8_e4m3<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const int64_t*)inp, (__nv_fp8_e4m3*)out); break;
        }
        return;
    }
    // From f16
    if (src_dtype == 5) {
        switch (dst_dtype) {
            case 0: cast_f16_f32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __half*)inp, (float*)out); break;
            case 1: cast_f16_f64<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __half*)inp, (double*)out); break;
            case 2: cast_f16_u8<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __half*)inp, (uint8_t*)out); break;
            case 3: cast_f16_u32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __half*)inp, (uint32_t*)out); break;
            case 6: cast_f16_bf16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __half*)inp, (__nv_bfloat16*)out); break;
            case 7: cast_f16_f8_e4m3<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __half*)inp, (__nv_fp8_e4m3*)out); break;
        }
        return;
    }
    // From bf16
    if (src_dtype == 6) {
        switch (dst_dtype) {
            case 0: cast_bf16_f32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, (float*)out); break;
            case 1: cast_bf16_f64<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, (double*)out); break;
            case 2: cast_bf16_u8<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, (uint8_t*)out); break;
            case 3: cast_bf16_u32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, (uint32_t*)out); break;
            case 5: cast_bf16_f16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, (__half*)out); break;
            case 7: cast_bf16_f8_e4m3<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, (__nv_fp8_e4m3*)out); break;
        }
        return;
    }
    // From f8_e4m3
    if (src_dtype == 7) {
        switch (dst_dtype) {
            case 0: cast_f8_e4m3_f32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_fp8_e4m3*)inp, (float*)out); break;
            case 1: cast_f8_e4m3_f64<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_fp8_e4m3*)inp, (double*)out); break;
            case 2: cast_f8_e4m3_u8<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_fp8_e4m3*)inp, (uint8_t*)out); break;
            case 3: cast_f8_e4m3_u32<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_fp8_e4m3*)inp, (uint32_t*)out); break;
            case 4: cast_f8_e4m3_i64<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_fp8_e4m3*)inp, (int64_t*)out); break;
            case 5: cast_f8_e4m3_f16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_fp8_e4m3*)inp, (__half*)out); break;
            case 6: cast_f8_e4m3_bf16<<<grid, CAST_BLOCK_SIZE_LAUNCH>>>(numel, num_dims, info, (const __nv_fp8_e4m3*)inp, (__nv_bfloat16*)out); break;
        }
        return;
    }
}

// In-place cast dispatcher (auto mode)
extern "C" void run_cast_mut(
    int32_t src_dtype,
    int32_t dst_dtype,
    size_t numel,
    void* buf
) {
    if (src_dtype >= 0 && src_dtype < 8 && dst_dtype >= 0 && dst_dtype < 8) {
        launch_cast_mut(CAST_MUT_KERNELS[src_dtype][dst_dtype], numel, buf);
    }
}

// Execution mode enum
enum CastMutMode {
    CAST_MUT_MODE_AUTO = 0,
    CAST_MUT_MODE_SINGLE_BLOCK = 1,
    CAST_MUT_MODE_COOPERATIVE = 2
};

// Launch with explicit mode selection
__host__ static void launch_cast_mut_with_mode(
    const cast_mut_kernel_pair& kernels,
    size_t numel,
    void* buf,
    int32_t mode
) {
    if (kernels.fallback == nullptr) {
        return;
    }
    
    switch (mode) {
        case CAST_MUT_MODE_SINGLE_BLOCK:
            kernels.fallback<<<1, CAST_MUT_BLOCK_SIZE>>>(numel, buf);
            break;
            
        case CAST_MUT_MODE_COOPERATIVE:
            if (kernels.coop != nullptr && supports_cooperative_launch()) {
                int num_blocks = get_coop_block_count(kernels.coop);
                int blocks_needed = (numel + CAST_MUT_BLOCK_SIZE - 1) / CAST_MUT_BLOCK_SIZE;
                num_blocks = min(num_blocks, blocks_needed);
                num_blocks = max(num_blocks, 1);
                
                void* args[] = { &numel, &buf };
                cudaLaunchCooperativeKernel(
                    (void*)kernels.coop,
                    dim3(num_blocks),
                    dim3(CAST_MUT_BLOCK_SIZE),
                    args,
                    0,
                    0
                );
            } else {
                kernels.fallback<<<1, CAST_MUT_BLOCK_SIZE>>>(numel, buf);
            }
            break;
            
        case CAST_MUT_MODE_AUTO:
        default:
            launch_cast_mut(kernels, numel, buf);
            break;
    }
}

// In-place cast dispatcher with mode selection
extern "C" void run_cast_mut_with_mode(
    int32_t src_dtype,
    int32_t dst_dtype,
    size_t numel,
    void* buf,
    int32_t mode
) {
    if (src_dtype < 0 || src_dtype >= 8 || dst_dtype < 0 || dst_dtype >= 8) {
        return;
    }
    launch_cast_mut_with_mode(CAST_MUT_KERNELS[src_dtype][dst_dtype], numel, buf, mode);
}

// =============================================================================
// PART 8: QUERY FUNCTIONS
// =============================================================================

/// Returns 1 if cooperative launch is supported, 0 otherwise
extern "C" int32_t cast_mut_supports_cooperative() {
    return supports_cooperative_launch() ? 1 : 0;
}

/// Returns the optimal number of blocks for cooperative launch with given numel
extern "C" int32_t cast_mut_get_optimal_blocks(size_t numel) {
    if (!supports_cooperative_launch()) {
        return 1;
    }
    
    cast_mut_fn_t sample_kernel = cast_mut_f32_f32_coop;
    if (sample_kernel == nullptr) {
        return 1;
    }
    
    int num_blocks = get_coop_block_count(sample_kernel);
    int blocks_needed = (numel + CAST_MUT_BLOCK_SIZE - 1) / CAST_MUT_BLOCK_SIZE;
    return min(num_blocks, blocks_needed);
}
