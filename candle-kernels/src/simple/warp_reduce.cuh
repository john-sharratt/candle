// =============================================================================
// Warp and Block Reduction Utilities
// =============================================================================
// High-performance reduction primitives using warp shuffle instructions.
// These are used by RMSNorm, LayerNorm, Softmax, and other reduction ops.
//
// Key functions:
//   - warp_reduce_sum(float)   : Sum across a single warp (32 threads)
//   - warp_reduce_sum(float2)  : Sum two values across a warp (for mean/var)
//   - block_reduce_sum<N>()    : Sum across a block with compile-time size
//   - block_reduce_sum_dynamic(): Sum across a block with runtime size
// =============================================================================

#pragma once

#include <cuda_runtime.h>

#define WARP_SIZE 32

// =============================================================================
// Warp-level reductions using shuffle instructions
// =============================================================================

// Reduce a single float across all 32 threads in a warp
// Uses butterfly pattern with __shfl_xor_sync for maximum efficiency
// Returns the sum in ALL threads of the warp
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

// Reduce two floats (e.g., mean and variance) across a warp
// More efficient than two separate reductions
static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, mask, 32);
        a.y += __shfl_xor_sync(0xffffffff, a.y, mask, 32);
    }
    return a;
}

// Reduce a single float, returning result only in lane 0
// Slightly more efficient when only one thread needs the result
static __device__ __forceinline__ float warp_reduce_sum_lane0(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

// =============================================================================
// Block-level reductions with compile-time block size
// =============================================================================

// Reduce across a block with compile-time known size
// Uses warp shuffle + shared memory for multi-warp blocks
// Returns the sum in ALL threads of the block
//
// BLOCK_SIZE must be a power of 2 and <= 1024
// For BLOCK_SIZE <= 32, no shared memory is used
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024, 
                  "BLOCK_SIZE must be between 1 and 1024");
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, 
                  "BLOCK_SIZE must be a power of 2");
    
    // First reduce within each warp
    val = warp_reduce_sum(val);
    
    if constexpr (BLOCK_SIZE <= WARP_SIZE) {
        // Single warp: we're done, all threads have the sum
        return val;
    } else {
        // Multi-warp: need cross-warp reduction via shared memory
        __shared__ float s_warp_sums[32];  // Max 32 warps (1024 threads)
        
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
        
        // Each warp leader writes its partial sum
        if (lane_id == 0) {
            s_warp_sums[warp_id] = val;
        }
        __syncthreads();
        
        // First warp reduces across all warp sums
        // Other threads get 0 to avoid reading uninitialized memory
        float warp_sum = (warp_id == 0 && lane_id < NUM_WARPS) 
                         ? s_warp_sums[lane_id] 
                         : 0.0f;
        
        if (warp_id == 0) {
            warp_sum = warp_reduce_sum(warp_sum);
        }
        
        // Broadcast final result to all threads
        if (threadIdx.x == 0) {
            s_warp_sums[0] = warp_sum;
        }
        __syncthreads();
        
        return s_warp_sums[0];
    }
}

// =============================================================================
// Block-level reductions with runtime block size
// =============================================================================

// Reduce across a block with runtime-determined size
// Slightly less efficient than compile-time version due to runtime conditionals
// Returns the sum in ALL threads of the block
__device__ __forceinline__ float block_reduce_sum_dynamic(float val, int block_size) {
    // First reduce within warp
    val = warp_reduce_sum(val);
    
    if (block_size <= WARP_SIZE) {
        return val;
    }
    
    // Multi-warp path
    __shared__ float s_warp_sums[32];
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    
    if (lane_id == 0) {
        s_warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    float warp_sum = (warp_id == 0 && lane_id < num_warps) 
                     ? s_warp_sums[lane_id] 
                     : 0.0f;
    
    if (warp_id == 0) {
        warp_sum = warp_reduce_sum(warp_sum);
    }
    
    if (threadIdx.x == 0) {
        s_warp_sums[0] = warp_sum;
    }
    __syncthreads();
    
    return s_warp_sums[0];
}

// =============================================================================
// Block-level reduction for float2 (mean and variance)
// =============================================================================

// Reduce float2 across a block with compile-time size
// Used by LayerNorm for computing mean and variance simultaneously
template <int BLOCK_SIZE>
__device__ __forceinline__ float2 block_reduce_sum(float2 val) {
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024, 
                  "BLOCK_SIZE must be between 1 and 1024");
    
    val = warp_reduce_sum(val);
    
    if constexpr (BLOCK_SIZE <= WARP_SIZE) {
        return val;
    } else {
        __shared__ float2 s_warp_sums[32];
        
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
        
        if (lane_id == 0) {
            s_warp_sums[warp_id] = val;
        }
        __syncthreads();
        
        float2 warp_sum;
        if (warp_id == 0 && lane_id < NUM_WARPS) {
            warp_sum = s_warp_sums[lane_id];
        } else {
            warp_sum = make_float2(0.0f, 0.0f);
        }
        
        if (warp_id == 0) {
            warp_sum = warp_reduce_sum(warp_sum);
        }
        
        if (threadIdx.x == 0) {
            s_warp_sums[0] = warp_sum;
        }
        __syncthreads();
        
        return s_warp_sums[0];
    }
}
