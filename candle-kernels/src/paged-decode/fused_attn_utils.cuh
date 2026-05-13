#pragma once
// =============================================================================
// FUSED ATTENTION UTILITIES
// =============================================================================
// Device helpers for fused operations within decode attention:
//   1. RoPE (Rotary Position Embedding) applied to Q/K in registers
//   2. Residual addition to attention output before final write
//
// These helpers are designed to be used both by:
//   - Standalone RoPE/residual kernels (for testing and unfused paths)
//   - Fused decode attention kernel (for maximum performance)
//
// Key design principles:
//   - Operate on register-resident data (no extra HBM round-trips)
//   - Support half2/bfloat162 vectorization where possible
//   - Use FMA for numerical precision and performance
//   - Cache cos/sin via __ldg() for read-only texture path
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// =============================================================================
// BASIC TYPE CONVERSIONS (if not already defined by parent kernel)
// =============================================================================

#ifndef FUSED_ATTN_TYPE_CONVERSIONS_DEFINED
#define FUSED_ATTN_TYPE_CONVERSIONS_DEFINED

template <typename T>
__device__ __forceinline__ float fused_to_f32(T v);

template <>
__device__ __forceinline__ float fused_to_f32<float>(float v) { return v; }

template <>
__device__ __forceinline__ float fused_to_f32<__half>(__half v) { return __half2float(v); }

template <>
__device__ __forceinline__ float fused_to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T fused_from_f32(float v);

template <>
__device__ __forceinline__ float fused_from_f32<float>(float v) { return v; }

template <>
__device__ __forceinline__ __half fused_from_f32<__half>(float v) { return __float2half_rn(v); }

template <>
__device__ __forceinline__ __nv_bfloat16 fused_from_f32<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

#endif // FUSED_ATTN_TYPE_CONVERSIONS_DEFINED


// =============================================================================
// ROPE HELPERS FOR REGISTER-RESIDENT VECTORS
// =============================================================================
// RoPE rotates pairs of elements: (x, y) → (x*cos - y*sin, x*sin + y*cos)
// Applied to Q after loading, and to K before writing to cache.
//
// Layouts supported:
//   - Rotary: pairs are d/2 apart: [x0..x_{d/2-1}, y0..y_{d/2-1}]
//   - Interleaved: pairs are adjacent: [(x0,y0), (x1,y1), ...]
//
// The rotary layout is most common for LLaMA/Qwen/etc.
// =============================================================================

/**
 * Apply RoPE rotation to a single pair of float values in registers.
 * 
 * @param x      First element of the pair (will be modified in place)
 * @param y      Second element of the pair (will be modified in place)  
 * @param cos_v  Cosine value for this position/dimension
 * @param sin_v  Sine value for this position/dimension
 *
 * Computes: x_new = x * cos - y * sin
 *           y_new = x * sin + y * cos
 */
__device__ __forceinline__ void rope_rotate_pair_f32(
    float& x, float& y,
    float cos_v, float sin_v
) {
    float x_orig = x;
    float y_orig = y;
    // Use FMA for precision: x_new = fma(x, cos, -y*sin)
    x = __fmaf_rn(x_orig, cos_v, -y_orig * sin_v);
    y = __fmaf_rn(x_orig, sin_v, y_orig * cos_v);
}

/**
 * Apply RoPE to a vector in registers (ROTARY layout).
 * 
 * For rotary layout, vec[0..half_dim-1] pairs with vec[half_dim..dim-1].
 * This function expects:
 *   - vec: register array of size VEC holding this thread's elements
 *   - Each thread holds elements [lane*VEC .. lane*VEC + VEC - 1]
 *   - half_dim = HEAD_DIM / 2
 *
 * @tparam VEC        Number of elements per thread
 * @param  vec        Register array holding thread's Q or K values (modified in-place)
 * @param  cos_cache  Pointer to cos values for this sequence position
 * @param  sin_cache  Pointer to sin values for this sequence position
 * @param  lane       Thread's lane ID within warp (0-31)
 * @param  half_dim   Half of head dimension (HEAD_DIM / 2)
 *
 * Note: cos/sin cache layout is [seq_len, head_dim/2] or similar.
 *       The position index should be applied to get the right base pointer.
 */
template <int VEC>
__device__ __forceinline__ void apply_rope_rotary(
    float* __restrict__ vec,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int lane,
    int half_dim
) {
    // In rotary layout:
    //   Element at global index i (where i < half_dim) pairs with element at i + half_dim
    //   Thread lane handles indices [lane*VEC, lane*VEC + VEC)
    //   
    // For HEAD_DIM=128, VEC=4, half_dim=64:
    //   lane 0:  indices 0-3   pair with 64-67
    //   lane 15: indices 60-63 pair with 124-127
    //   lane 16: indices 64-67 (these ARE the second halves, already paired)
    //   ...
    //
    // So only lanes 0 to (half_dim/VEC - 1) need to do rotation work.
    // Lanes handling the second half don't initiate rotations.
    
    int base_idx = lane * VEC;
    
    if (base_idx < half_dim) {
        // This thread handles elements in the first half - apply rotation
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int d_idx = base_idx + j;  // dimension index (0 to half_dim-1)
            
            // Load cos/sin for this dimension
            float cos_v = __ldg(cos_cache + d_idx);
            float sin_v = __ldg(sin_cache + d_idx);
            
            // Get paired element via warp shuffle
            // The paired element is at d_idx + half_dim, which is in a different thread
            // Thread T has elements [T*VEC, T*VEC+VEC), so element (d_idx + half_dim) is in 
            // thread (d_idx + half_dim) / VEC = lane + half_dim/VEC
            int pair_lane = lane + (half_dim / VEC);
            int pair_reg_idx = j;  // Same register index in the paired thread
            
            float y = __shfl_sync(0xffffffff, vec[pair_reg_idx], pair_lane);
            float x = vec[j];
            
            // Rotate
            float x_new = __fmaf_rn(x, cos_v, -y * sin_v);
            float y_new = __fmaf_rn(x, sin_v, y * cos_v);
            
            vec[j] = x_new;
            
            // Send rotated y back to paired thread
            // The paired thread will receive this via the symmetric shuffle below
        }
    }
    
    // Now handle the second half - receive rotated values from first half
    if (base_idx >= half_dim) {
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int d_idx = base_idx + j - half_dim;  // Original dimension index for this pair
            
            float cos_v = __ldg(cos_cache + d_idx);
            float sin_v = __ldg(sin_cache + d_idx);
            
            // Get the x value from the first-half thread
            int pair_lane = lane - (half_dim / VEC);
            float x = __shfl_sync(0xffffffff, vec[j], pair_lane);  // This gets vec[j] from pair_lane before rotation
            float y = vec[j];
            
            // Rotate the second element
            vec[j] = __fmaf_rn(x, sin_v, y * cos_v);
        }
    }
    
    // Note: The above is a simplified version. In practice, we need both threads
    // to have the original values before either modifies them. A cleaner approach
    // is to do the shuffle exchange first, then compute.
}

/**
 * Apply RoPE to a vector in registers (ROTARY layout) - Simplified version.
 * 
 * This version assumes the caller has already loaded cos/sin values and
 * the vector elements are stored sequentially in registers.
 * Uses shared memory for the paired element exchange.
 *
 * @tparam VEC        Number of elements per thread
 * @tparam HEAD_DIM   Total head dimension
 * @param  vec        Register array holding thread's Q or K values (modified in-place)
 * @param  cos_cache  Pointer to cos values [head_dim/2] for this position
 * @param  sin_cache  Pointer to sin values [head_dim/2] for this position
 * @param  lane       Thread's lane ID within warp (0-31)
 * @param  smem       Shared memory for exchange [HEAD_DIM] (one per warp if needed)
 */
template <int VEC, int HEAD_DIM>
__device__ __forceinline__ void apply_rope_rotary_smem(
    float* __restrict__ vec,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int lane,
    float* __restrict__ smem
) {
    constexpr int HALF_DIM = HEAD_DIM / 2;
    constexpr int WARP_SIZE = 32;
    
    // Step 1: Write all elements to shared memory
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
        smem[lane * VEC + j] = vec[j];
    }
    __syncwarp();
    
    // Step 2: Read paired elements and apply rotation
    int base_idx = lane * VEC;
    
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
        int idx = base_idx + j;
        int pair_idx = (idx < HALF_DIM) ? (idx + HALF_DIM) : (idx - HALF_DIM);
        int d_idx = (idx < HALF_DIM) ? idx : (idx - HALF_DIM);  // cos/sin index
        
        float x = smem[idx];
        float y = smem[pair_idx];
        float cos_v = __ldg(cos_cache + d_idx);
        float sin_v = __ldg(sin_cache + d_idx);
        
        if (idx < HALF_DIM) {
            // First half: x_new = x*cos - y*sin
            vec[j] = __fmaf_rn(x, cos_v, -y * sin_v);
        } else {
            // Second half: y_new = x*sin + y*cos
            vec[j] = __fmaf_rn(x, sin_v, y * cos_v);
        }
    }
}

/**
 * Apply RoPE to a float vector using warp shuffles only (no shared memory).
 * 
 * This is the preferred method when HEAD_DIM allows clean warp-level exchange.
 * Works for HEAD_DIM = 64, 128, 256 where half_dim is multiple of VEC.
 *
 * Constraints:
 *   - HEAD_DIM / 2 must be divisible by VEC
 *   - WARP_SIZE * VEC >= HEAD_DIM
 *
 * @tparam VEC        Number of elements per thread
 * @tparam HEAD_DIM   Total head dimension  
 */
template <int VEC, int HEAD_DIM>
__device__ __forceinline__ void apply_rope_rotary_shuffle(
    float* __restrict__ vec,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int lane
) {
    constexpr int HALF_DIM = HEAD_DIM / 2;
    constexpr int LANES_PER_HALF = HALF_DIM / VEC;  // Lanes covering first half
    
    int base_idx = lane * VEC;
    bool in_first_half = (base_idx < HALF_DIM);
    
    // For each element, we need to exchange with its pair
    // First half lane L exchanges with lane (L + LANES_PER_HALF)
    int pair_lane = in_first_half ? (lane + LANES_PER_HALF) : (lane - LANES_PER_HALF);
    
    // Exchange all VEC elements with paired thread
    float paired[VEC];
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
        paired[j] = __shfl_sync(0xffffffff, vec[j], pair_lane);
    }
    
    // Apply rotation
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
        int d_idx = in_first_half ? (base_idx + j) : (base_idx + j - HALF_DIM);
        float cos_v = __ldg(cos_cache + d_idx);
        float sin_v = __ldg(sin_cache + d_idx);
        
        float x = in_first_half ? vec[j] : paired[j];
        float y = in_first_half ? paired[j] : vec[j];
        
        if (in_first_half) {
            vec[j] = __fmaf_rn(x, cos_v, -y * sin_v);
        } else {
            vec[j] = __fmaf_rn(x, sin_v, y * cos_v);
        }
    }
}

/**
 * Apply RoPE to a typed vector, converting to float for rotation.
 * 
 * @tparam T          Element type (half, bfloat16, float)
 * @tparam VEC        Number of elements per thread
 * @tparam HEAD_DIM   Total head dimension
 */
template <typename T, int VEC, int HEAD_DIM>
__device__ __forceinline__ void apply_rope_to_vec(
    float* __restrict__ vec,
    const T* __restrict__ cos_cache,
    const T* __restrict__ sin_cache,
    int lane
) {
    constexpr int HALF_DIM = HEAD_DIM / 2;
    constexpr int LANES_PER_HALF = HALF_DIM / VEC;
    
    int base_idx = lane * VEC;
    bool in_first_half = (base_idx < HALF_DIM);
    int pair_lane = in_first_half ? (lane + LANES_PER_HALF) : (lane - LANES_PER_HALF);
    
    // Exchange with paired thread
    float paired[VEC];
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
        paired[j] = __shfl_sync(0xffffffff, vec[j], pair_lane);
    }
    
    // Apply rotation
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
        int d_idx = in_first_half ? (base_idx + j) : (base_idx + j - HALF_DIM);
        float cos_v = fused_to_f32(__ldg(cos_cache + d_idx));
        float sin_v = fused_to_f32(__ldg(sin_cache + d_idx));
        
        float x = in_first_half ? vec[j] : paired[j];
        float y = in_first_half ? paired[j] : vec[j];
        
        if (in_first_half) {
            vec[j] = __fmaf_rn(x, cos_v, -y * sin_v);
        } else {
            vec[j] = __fmaf_rn(x, sin_v, y * cos_v);
        }
    }
}


// =============================================================================
// INTERLEAVED ROPE (for models using adjacent pair layout)
// =============================================================================

/**
 * Apply RoPE to a vector with INTERLEAVED layout.
 * 
 * In interleaved layout, pairs are adjacent: [(x0,y0), (x1,y1), ...]
 * So element 0 pairs with element 1, element 2 pairs with element 3, etc.
 *
 * This version handles all VEC sizes:
 * - VEC=1: pairs span adjacent threads, use xor shuffle
 * - VEC even: pairs are within thread (optimal, no shuffles)
 * - VEC odd: one cross-thread pair, use xor shuffle for uniform access
 */
template <typename T, int VEC, int HEAD_DIM>
__device__ __forceinline__ void apply_rope_interleaved(
    float* __restrict__ vec,
    const T* __restrict__ cos_cache,  // [head_dim/2]
    const T* __restrict__ sin_cache,
    int lane
) {
    const int base_idx = lane * VEC;
    
    if constexpr (VEC == 1) {
        // VEC=1: Each thread holds one element. Adjacent threads form pairs.
        // Use xor with 1 to exchange: lane 0 <-> lane 1, lane 2 <-> lane 3, etc.
        
        float my_val = vec[0];
        float pair_val = __shfl_xor_sync(0xffffffff, my_val, 1);
        
        int pair_idx = base_idx / 2;
        float cos_v = fused_to_f32(__ldg(cos_cache + pair_idx));
        float sin_v = fused_to_f32(__ldg(sin_cache + pair_idx));
        
        // Even global indices are x, odd are y
        if ((base_idx & 1) == 0) {
            vec[0] = __fmaf_rn(my_val, cos_v, -pair_val * sin_v);
        } else {
            vec[0] = __fmaf_rn(pair_val, sin_v, my_val * cos_v);
        }
    } else if constexpr (VEC % 2 == 0) {
        // VEC is even: all pairs are within the same thread - optimal path
        #pragma unroll
        for (int j = 0; j < VEC; j += 2) {
            int pair_idx = (base_idx + j) / 2;
            
            float cos_v = fused_to_f32(__ldg(cos_cache + pair_idx));
            float sin_v = fused_to_f32(__ldg(sin_cache + pair_idx));
            
            float x = vec[j];
            float y = vec[j + 1];
            
            vec[j]     = __fmaf_rn(x, cos_v, -y * sin_v);
            vec[j + 1] = __fmaf_rn(x, sin_v, y * cos_v);
        }
    } else {
        // VEC is odd: one element pairs across thread boundary
        // 
        // For VEC=3: 
        //   Lane 0 (base=0): [0,1,2] -> (0,1) in-thread, 2 is x of cross-pair (2,3)
        //   Lane 1 (base=3): [3,4,5] -> 3 is y of cross-pair (2,3), (4,5) in-thread
        //   Lane 2 (base=6): [6,7,8] -> (6,7) in-thread, 8 is x of cross-pair (8,9)
        //   etc.
        //
        // Pattern: if base_idx is even, first element starts a pair (process from start)
        //          if base_idx is odd, first element ends a pair (process from index 1)
        
        const bool base_even = (base_idx & 1) == 0;
        const int start_j = base_even ? 0 : 1;  // Where in-thread pairs start
        
        // Process all complete in-thread pairs
        #pragma unroll
        for (int j = start_j; j + 1 < VEC; j += 2) {
            int pair_idx = (base_idx + j) / 2;
            
            float cos_v = fused_to_f32(__ldg(cos_cache + pair_idx));
            float sin_v = fused_to_f32(__ldg(sin_cache + pair_idx));
            
            float x = vec[j];
            float y = vec[j + 1];
            
            vec[j]     = __fmaf_rn(x, cos_v, -y * sin_v);
            vec[j + 1] = __fmaf_rn(x, sin_v, y * cos_v);
        }
        
        // Handle cross-thread element using xor shuffle (same pattern as VEC=1)
        // Cross-thread element index: if base_even, it's VEC-1; if base_odd, it's 0
        const int cross_local = base_even ? (VEC - 1) : 0;
        const int cross_global = base_idx + cross_local;
        
        float my_val = vec[cross_local];
        float pair_val = __shfl_xor_sync(0xffffffff, my_val, 1);
        
        int pair_idx = cross_global / 2;
        float cos_v = fused_to_f32(__ldg(cos_cache + pair_idx));
        float sin_v = fused_to_f32(__ldg(sin_cache + pair_idx));
        
        // Even global indices are x, odd are y
        if ((cross_global & 1) == 0) {
            vec[cross_local] = __fmaf_rn(my_val, cos_v, -pair_val * sin_v);
        } else {
            vec[cross_local] = __fmaf_rn(pair_val, sin_v, my_val * cos_v);
        }
    }
}

/**
 * Apply RoPE (Rotary Position Embedding) with NON-INTERLEAVED layout (Qwen-style).
 *
 * Non-interleaved layout: first half is x, second half is y
 *   vec = [x0, x1, ..., x_{d/2-1}, y0, y1, ..., y_{d/2-1}]
 *   pairs: (x0, y0), (x1, y1), ..., (x_{d/2-1}, y_{d/2-1})
 *
 * cos_cache/sin_cache: [head_dim/2] - one value per pair
 *
 * Rotation formula for pair i:
 *   x'_i = x_i * cos_i - y_i * sin_i
 *   y'_i = x_i * sin_i + y_i * cos_i
 *
 * For this layout, pairs are at offset HEAD_DIM/2 apart:
 *   x_i is at index i, y_i is at index i + HEAD_DIM/2
 */
template <typename T, int VEC, int HEAD_DIM>
__device__ __forceinline__ void apply_rope_non_interleaved(
    float* __restrict__ vec,
    const T* __restrict__ cos_cache,  // [head_dim/2]
    const T* __restrict__ sin_cache,
    int lane
) {
    constexpr int HALF_DIM = HEAD_DIM / 2;
    constexpr int WARP_SIZE = 32;
    const int base_idx = lane * VEC;
    
    // Each thread processes VEC elements. We need to figure out which elements
    // this thread owns and whether they are in the x-half or y-half.
    //
    // For HEAD_DIM=128, VEC=4, HALF_DIM=64:
    //   Lane 0: indices [0,1,2,3] -> all in x-half
    //   Lane 15: indices [60,61,62,63] -> all in x-half
    //   Lane 16: indices [64,65,66,67] -> all in y-half
    //   Lane 31: indices [124,125,126,127] -> all in y-half
    //
    // For HEAD_DIM=64, VEC=2, HALF_DIM=32:
    //   Lanes 0-15: x-half
    //   Lanes 16-31: y-half
    
    // Check if this thread's elements are in x-half or y-half
    // Since VEC elements are contiguous and HALF_DIM is typically >= 32,
    // all elements in a thread are either all x or all y (no split case for typical configs)
    
    const bool all_in_x_half = (base_idx + VEC - 1) < HALF_DIM;
    const bool all_in_y_half = base_idx >= HALF_DIM;
    
    if (all_in_x_half) {
        // This thread owns x-values. We need y-values from another thread.
        // y_i is at index i + HALF_DIM, so y-lane = lane + HALF_DIM/VEC
        const int y_lane = lane + (HALF_DIM / VEC);
        
        // Exchange with y-lane to get y values
        float y_vals[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            y_vals[j] = __shfl_sync(0xffffffff, vec[j], y_lane);
        }
        
        // Apply rotation: x' = x*cos - y*sin
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int pair_idx = base_idx + j;  // pair index = x index
            float cos_v = fused_to_f32(__ldg(cos_cache + pair_idx));
            float sin_v = fused_to_f32(__ldg(sin_cache + pair_idx));
            vec[j] = __fmaf_rn(vec[j], cos_v, -y_vals[j] * sin_v);
        }
    } else if (all_in_y_half) {
        // This thread owns y-values. We need x-values from another thread.
        // x_i is at index i - HALF_DIM, so x-lane = lane - HALF_DIM/VEC
        const int x_lane = lane - (HALF_DIM / VEC);
        
        // Exchange with x-lane to get x values
        float x_vals[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            x_vals[j] = __shfl_sync(0xffffffff, vec[j], x_lane);
        }
        
        // Apply rotation: y' = x*sin + y*cos
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int pair_idx = base_idx + j - HALF_DIM;  // pair index
            float cos_v = fused_to_f32(__ldg(cos_cache + pair_idx));
            float sin_v = fused_to_f32(__ldg(sin_cache + pair_idx));
            vec[j] = __fmaf_rn(x_vals[j], sin_v, vec[j] * cos_v);
        }
    } else {
        // Split case: thread spans x-half and y-half boundary
        // This can happen when VEC > HALF_DIM / WARP_SIZE
        // For typical configs (HEAD_DIM >= 64), this won't occur, but handle it for safety
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int global_idx = base_idx + j;
            if (global_idx < HALF_DIM) {
                // x-value: need corresponding y-value
                int y_idx = global_idx + HALF_DIM;
                int y_lane = y_idx / VEC;
                int y_offset = y_idx % VEC;
                float y_val = __shfl_sync(0xffffffff, vec[y_offset], y_lane);
                
                int pair_idx = global_idx;
                float cos_v = fused_to_f32(__ldg(cos_cache + pair_idx));
                float sin_v = fused_to_f32(__ldg(sin_cache + pair_idx));
                vec[j] = __fmaf_rn(vec[j], cos_v, -y_val * sin_v);
            } else {
                // y-value: need corresponding x-value
                int x_idx = global_idx - HALF_DIM;
                int x_lane = x_idx / VEC;
                int x_offset = x_idx % VEC;
                float x_val = __shfl_sync(0xffffffff, vec[x_offset], x_lane);
                
                int pair_idx = x_idx;
                float cos_v = fused_to_f32(__ldg(cos_cache + pair_idx));
                float sin_v = fused_to_f32(__ldg(sin_cache + pair_idx));
                vec[j] = __fmaf_rn(x_val, sin_v, vec[j] * cos_v);
            }
        }
    }
}


// =============================================================================
// RESIDUAL ADD HELPERS
// =============================================================================
// Simple in-register addition of residual connection.
// Applied to attention output before final HBM write.
//
// Optionally supports a scale factor for weighted residual:
//   output = attention_out + residual_scale * residual
// =============================================================================

/**
 * Add residual to attention output in registers.
 * 
 * @tparam T          Element type of residual tensor
 * @tparam VEC        Number of elements per thread
 * @param  out_reg    Register array holding attention output (modified in-place)
 * @param  residual   Pointer to residual tensor for this thread's elements
 * @param  lane       Thread's lane ID
 * @param  scale      Scale factor for residual (default 1.0)
 */
template <typename T, int VEC>
__device__ __forceinline__ void add_residual_to_output(
    float* __restrict__ out_reg,
    const T* __restrict__ residual,
    int lane,
    float scale = 1.0f
) {
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
        float res_val = fused_to_f32(__ldg(residual + lane * VEC + j));
        out_reg[j] = __fmaf_rn(res_val, scale, out_reg[j]);
    }
}

/**
 * Add residual with separate base offset.
 * Useful when residual tensor has different stride than output.
 */
template <typename T, int VEC>
__device__ __forceinline__ void add_residual_with_offset(
    float* __restrict__ out_reg,
    const T* __restrict__ residual_base,
    int64_t offset,
    int lane,
    float scale = 1.0f
) {
    const T* residual = residual_base + offset;
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
        float res_val = fused_to_f32(__ldg(residual + lane * VEC + j));
        out_reg[j] = __fmaf_rn(res_val, scale, out_reg[j]);
    }
}


// =============================================================================
// COMBINED ROPE + RESIDUAL PARAMETERS STRUCT
// =============================================================================
// Groups optional fusion parameters for cleaner kernel signatures.
// =============================================================================

template <typename T>
struct FusedAttnParams {
    // RoPE parameters (nullptr to disable)
    const T* cos_cache;       // [max_seq_len, head_dim/2] or [head_dim/2] for single position
    const T* sin_cache;       // Same shape as cos_cache
    const int* positions;     // [batch_size] - sequence positions for cos/sin lookup
    int cos_sin_stride;       // Stride between positions in cos/sin cache (typically head_dim/2)
    bool rope_interleaved;    // True for interleaved layout, false for rotary
    
    // Residual parameters (nullptr to disable)
    const T* residual;        // [batch_size, n_head, head_dim] - residual to add
    float residual_scale;     // Scale factor for residual (typically 1.0)
    
    // Default: all fusion disabled
    __device__ __host__ FusedAttnParams()
        : cos_cache(nullptr)
        , sin_cache(nullptr)
        , positions(nullptr)
        , cos_sin_stride(0)
        , rope_interleaved(false)
        , residual(nullptr)
        , residual_scale(1.0f)
    {}
    
    // Check if RoPE is enabled
    __device__ __forceinline__ bool has_rope() const {
        return cos_cache != nullptr && sin_cache != nullptr;
    }
    
    // Check if residual is enabled
    __device__ __forceinline__ bool has_residual() const {
        return residual != nullptr;
    }
};


// =============================================================================
// POSITION-INDEXED ROPE APPLICATION
// =============================================================================
// For decode, we typically have a per-batch position that indexes into cos/sin.
// =============================================================================

/**
 * Apply RoPE using a position index to look up cos/sin values.
 * 
 * @param vec          Register array to rotate in-place
 * @param cos_cache    Full cos cache [max_positions, head_dim/2]
 * @param sin_cache    Full sin cache [max_positions, head_dim/2]
 * @param position     Sequence position for this batch element
 * @param cos_sin_stride  Stride between positions (typically head_dim/2)
 * @param lane         Thread lane ID
 */
template <typename T, int VEC, int HEAD_DIM>
__device__ __forceinline__ void apply_rope_with_position(
    float* __restrict__ vec,
    const T* __restrict__ cos_cache,
    const T* __restrict__ sin_cache,
    int position,
    int cos_sin_stride,
    int lane
) {
    const T* cos_ptr = cos_cache + (int64_t)position * cos_sin_stride;
    const T* sin_ptr = sin_cache + (int64_t)position * cos_sin_stride;
    apply_rope_to_vec<T, VEC, HEAD_DIM>(vec, cos_ptr, sin_ptr, lane);
}


// =============================================================================
// FUSED K PROCESSING: ROPE + SCATTER TO ARENA
// =============================================================================
// When writing new K to the KV cache, we can fuse RoPE application.
// This happens BEFORE the K is written to the arena.
// =============================================================================

/**
 * Apply RoPE to K values and write to arena (fused RoPE + scatter).
 *
 * @param k_new        New K values in registers (float, will be rotated)
 * @param k_arena      Destination arena for K values
 * @param arena_offset Base offset in arena for this position
 * @param cos_ptr      Cos values for this position
 * @param sin_ptr      Sin values for this position
 * @param lane         Thread lane ID
 */
template <typename T, int VEC, int HEAD_DIM>
__device__ __forceinline__ void fused_rope_and_scatter_k(
    float* __restrict__ k_reg,
    T* __restrict__ k_arena,
    int64_t arena_offset,
    const T* __restrict__ cos_ptr,
    const T* __restrict__ sin_ptr,
    int lane
) {
    // Apply RoPE to K in registers
    apply_rope_to_vec<T, VEC, HEAD_DIM>(k_reg, cos_ptr, sin_ptr, lane);
    
    // Write rotated K to arena
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
        k_arena[arena_offset + lane * VEC + j] = fused_from_f32<T>(k_reg[j]);
    }
}
