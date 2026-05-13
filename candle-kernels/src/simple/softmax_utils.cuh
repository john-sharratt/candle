#pragma once
// =============================================================================
// SOFTMAX UTILITIES - Optimized primitives for softmax computation
// =============================================================================
// Implements safe, numerically stable softmax with key optimizations:
//   - Online algorithm: single-pass max + sum with correction factor
//   - Vectorized memory access (float4, half2)
//   - __ldg() for read-only input access
//   - Fast math intrinsics (__expf, __fdividef)
//   - Proper inter-warp reduction for all block sizes
//   - Template specializations for common ncols values
// =============================================================================

#include "cuda_utils.cuh" 
#include "warp_reduce.cuh"
#include "../fast_exp.cuh"
#include <stdint.h>
#include <float.h>
#include <limits>

// Negative infinity for float and double that works in both host and device code
// Using numeric_limits which is constexpr and works everywhere
#ifndef NEG_INF_F
#define NEG_INF_F (-std::numeric_limits<float>::infinity())
#endif
#ifndef NEG_INF_D  
#define NEG_INF_D (-std::numeric_limits<double>::infinity())
#endif

// =============================================================================
// TYPE CONVERSION HELPERS (for FP8 support)
// =============================================================================

template <typename T>
__device__ __forceinline__ float softmax_to_float(T v) {
    return static_cast<float>(v);
}

template <typename T>
__device__ __forceinline__ T softmax_from_float(float v) {
    return static_cast<T>(v);
}

// FP8E4M3 conversion helpers
__device__ __forceinline__ float softmax_fp8e4m3_to_float(__nv_fp8_e4m3 v) {
#if __CUDA_ARCH__ >= 890
    __nv_fp8_storage_t storage = *reinterpret_cast<const __nv_fp8_storage_t*>(&v);
    return __half2float(__nv_cvt_fp8_to_halfraw(storage, __NV_E4M3));
#else
    uint8_t bits = *reinterpret_cast<const uint8_t*>(&v);
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp = (bits >> 3) & 0xF;
    uint32_t mant = bits & 0x7;
    
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float m = mant / 8.0f;
        float result = ldexpf(m, -6);
        return sign ? -result : result;
    } else if (exp == 15) {
        return __int_as_float(0x7FC00000);
    } else {
        float m = 1.0f + mant / 8.0f;
        float result = ldexpf(m, (int)exp - 7);
        return sign ? -result : result;
    }
#endif
}

__device__ __forceinline__ __nv_fp8_e4m3 softmax_float_to_fp8e4m3(float v) {
#if __CUDA_ARCH__ >= 890
    __nv_fp8_storage_t storage = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 result;
    *reinterpret_cast<__nv_fp8_storage_t*>(&result) = storage;
    return result;
#else
    __nv_fp8_e4m3 result;
    uint8_t* out = reinterpret_cast<uint8_t*>(&result);
    
    uint32_t fbits = __float_as_int(v);
    uint32_t sign = (fbits >> 31) & 1;
    int32_t exp = ((fbits >> 23) & 0xFF) - 127;
    uint32_t mant = fbits & 0x7FFFFF;
    
    if ((fbits & 0x7FFFFFFF) == 0) {
        *out = sign << 7;
        return result;
    }
    if (exp > 8) {
        *out = (sign << 7) | (14 << 3) | 7;
        return result;
    }
    if (exp < -9) {
        *out = sign << 7;
        return result;
    }
    
    int32_t e4m3_exp = exp + 7;
    uint32_t e4m3_mant;
    
    if (e4m3_exp <= 0) {
        int shift = 1 - e4m3_exp + 20;
        e4m3_mant = ((1 << 23) | mant) >> shift;
        e4m3_exp = 0;
    } else {
        e4m3_mant = (mant + (1 << 19)) >> 20;
        if (e4m3_mant >= 8) {
            e4m3_mant = 0;
            e4m3_exp++;
            if (e4m3_exp > 14) {
                *out = (sign << 7) | (14 << 3) | 7;
                return result;
            }
        }
    }
    
    *out = (sign << 7) | (e4m3_exp << 3) | (e4m3_mant & 0x7);
    return result;
#endif
}

template <>
__device__ __forceinline__ float softmax_to_float(__nv_fp8_e4m3 v) {
    return softmax_fp8e4m3_to_float(v);
}

template <>
__device__ __forceinline__ __nv_fp8_e4m3 softmax_from_float(float v) {
    return softmax_float_to_fp8e4m3(v);
}

// =============================================================================
// FAST MATH INTRINSICS
// =============================================================================

// Fast exponential for softmax - uses fast_exp library with Softmax mode (assumes x <= 0)
__device__ __forceinline__ float softmax_exp(float x) {
    return fast_exp::exp<float, fast_exp::Softmax>(x);
}

// Fast division - ~4x faster than standard division
__device__ __forceinline__ float fast_div(float a, float b) {
    return __fdividef(a, b);
}

// Fast reciprocal
__device__ __forceinline__ float fast_rcp(float x) {
    return __frcp_rn(x);
}

// =============================================================================
// WARP-LEVEL MAX REDUCTION
// =============================================================================

// Reduce max across a warp using butterfly pattern
__device__ __forceinline__ float warp_reduce_max(float x) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

// =============================================================================
// BLOCK-LEVEL MAX REDUCTION (for block_size > 32)
// =============================================================================

__device__ __forceinline__ float block_reduce_max_dynamic(float val, int block_size) {
    // First reduce within warp
    val = warp_reduce_max(val);
    
    if (block_size <= WARP_SIZE) {
        return val;
    }
    
    // Multi-warp path using shared memory
    __shared__ float s_warp_max[32];  // Max 32 warps
    
    const int warp_id = threadIdx.y / WARP_SIZE;  // Using y dimension for softmax
    const int lane_id = threadIdx.y % WARP_SIZE;
    const int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    
    if (lane_id == 0) {
        s_warp_max[warp_id] = val;
    }
    __syncthreads();
    
    // First warp reduces across all warp maxes
    float warp_max = (warp_id == 0 && lane_id < num_warps) 
                     ? s_warp_max[lane_id] 
                     : NEG_INF_F;
    
    if (warp_id == 0) {
        warp_max = warp_reduce_max(warp_max);
    }
    
    // Broadcast final result
    if (threadIdx.y == 0) {
        s_warp_max[0] = warp_max;
    }
    __syncthreads();
    
    return s_warp_max[0];
}

// =============================================================================
// BLOCK-LEVEL SUM REDUCTION (for block_size > 32)
// =============================================================================

__device__ __forceinline__ float block_reduce_sum_softmax(float val, int block_size) {
    // First reduce within warp
    val = warp_reduce_sum(val);
    
    if (block_size <= WARP_SIZE) {
        return val;
    }
    
    __shared__ float s_warp_sum[32];
    
    const int warp_id = threadIdx.y / WARP_SIZE;
    const int lane_id = threadIdx.y % WARP_SIZE;
    const int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    
    if (lane_id == 0) {
        s_warp_sum[warp_id] = val;
    }
    __syncthreads();
    
    float warp_sum = (warp_id == 0 && lane_id < num_warps) 
                     ? s_warp_sum[lane_id] 
                     : 0.0f;
    
    if (warp_id == 0) {
        warp_sum = warp_reduce_sum(warp_sum);
    }
    
    if (threadIdx.y == 0) {
        s_warp_sum[0] = warp_sum;
    }
    __syncthreads();
    
    return s_warp_sum[0];
}

// =============================================================================
// ONLINE SOFTMAX ALGORITHM
// =============================================================================
// Computes max and (corrected) exp sum in a single pass:
//   m_new = max(m_old, x)
//   d_new = d_old * exp(m_old - m_new) + exp(x - m_new)
// This eliminates one global memory read pass.

struct OnlineSoftmaxState {
    float max_val;
    float sum_exp;
    
    __device__ __forceinline__ OnlineSoftmaxState() 
        : max_val(NEG_INF_F), sum_exp(0.0f) {}
    
    __device__ __forceinline__ OnlineSoftmaxState(float m, float s) 
        : max_val(m), sum_exp(s) {}
    
    // Update state with a new value
    __device__ __forceinline__ void update(float x) {
        if (x > max_val) {
            // Rescale existing sum and add new element
            sum_exp = sum_exp * softmax_exp(max_val - x) + 1.0f;
            max_val = x;
        } else {
            // Use fmaxf to handle -inf - (-inf) = NaN case branchlessly
            // fmaxf(NaN, -inf) = -inf per IEEE 754-2008, then exp(-inf) = 0
            sum_exp += softmax_exp(fmaxf(x - max_val, NEG_INF_F));
        }
    }
    
    // Merge two states (for reduction)
    __device__ __forceinline__ void merge(const OnlineSoftmaxState& other) {
        if (other.max_val > max_val) {
            sum_exp = sum_exp * softmax_exp(fmaxf(max_val - other.max_val, NEG_INF_F)) + other.sum_exp;
            max_val = other.max_val;
        } else {
            // Use fmaxf to handle -inf - (-inf) = NaN case branchlessly
            sum_exp += other.sum_exp * softmax_exp(fmaxf(other.max_val - max_val, NEG_INF_F));
        }
    }
};

// =============================================================================
// WARP-LEVEL ONLINE SOFTMAX REDUCTION
// =============================================================================

__device__ __forceinline__ OnlineSoftmaxState 
warp_reduce_online_softmax(OnlineSoftmaxState state) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float other_max = __shfl_xor_sync(0xffffffff, state.max_val, mask, 32);
        float other_sum = __shfl_xor_sync(0xffffffff, state.sum_exp, mask, 32);
        OnlineSoftmaxState other(other_max, other_sum);
        state.merge(other);
    }
    return state;
}

// =============================================================================
// BLOCK-LEVEL ONLINE SOFTMAX REDUCTION
// =============================================================================

__device__ __forceinline__ OnlineSoftmaxState 
block_reduce_online_softmax(OnlineSoftmaxState state, int block_size) {
    state = warp_reduce_online_softmax(state);
    
    if (block_size <= WARP_SIZE) {
        return state;
    }
    
    __shared__ float s_max[32];
    __shared__ float s_sum[32];
    
    const int warp_id = threadIdx.y / WARP_SIZE;
    const int lane_id = threadIdx.y % WARP_SIZE;
    const int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    
    if (lane_id == 0) {
        s_max[warp_id] = state.max_val;
        s_sum[warp_id] = state.sum_exp;
    }
    __syncthreads();
    
    OnlineSoftmaxState warp_state;
    if (warp_id == 0 && lane_id < num_warps) {
        warp_state.max_val = s_max[lane_id];
        warp_state.sum_exp = s_sum[lane_id];
    }
    
    if (warp_id == 0) {
        warp_state = warp_reduce_online_softmax(warp_state);
    }
    
    if (threadIdx.y == 0) {
        s_max[0] = warp_state.max_val;
        s_sum[0] = warp_state.sum_exp;
    }
    __syncthreads();
    
    return OnlineSoftmaxState(s_max[0], s_sum[0]);
}

// =============================================================================
// VECTORIZED LOAD HELPERS
// =============================================================================

// Load 4 floats using __ldg (read-only cache)
__device__ __forceinline__ float4 ldg_load_f32x4(const float* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const float4*>(ptr));
}

// Load 2 floats using __ldg
__device__ __forceinline__ float2 ldg_load_f32x2(const float* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const float2*>(ptr));
}

// =============================================================================
// OPTIMIZED SOFTMAX KERNEL - REGISTER-BASED (ncols <= 1024)
// =============================================================================
// For small ncols, keep all intermediate values in registers.
// One thread block per row, threads cooperate within block.

template <typename T, typename ACC>
__device__ void softmax_register_based(
    const T* __restrict__ x,
    T* __restrict__ dst,
    const int ncols
) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int block_size = blockDim.y;
    const int tid = threadIdx.y;
    const int row_offset = row * ncols;
    
    // Phase 1: Online computation of max and sum_exp in single pass
    OnlineSoftmaxState state;
    
    #pragma unroll 4
    for (int col = tid; col < ncols; col += block_size) {
        float val = softmax_to_float(__ldg(x + row_offset + col));
        state.update(val);
    }
    
    // Reduce across block
    state = block_reduce_online_softmax(state, block_size);
    
    const float max_val = state.max_val;
    const float inv_sum = fast_rcp(state.sum_exp);
    
    // Phase 2: Compute final softmax output
    #pragma unroll 4
    for (int col = tid; col < ncols; col += block_size) {
        float val = softmax_to_float(__ldg(x + row_offset + col));
        float result = softmax_exp(val - max_val) * inv_sum;
        dst[row_offset + col] = softmax_from_float<T>(result);
    }
}

// =============================================================================
// OPTIMIZED SOFTMAX - FLOAT SPECIALIZATION WITH FLOAT4 VECTORIZATION
// =============================================================================
// NOTE: float4 vectorized loads/stores require 16-byte alignment.
// We check alignment and fall back to scalar path when misaligned.

template <>
__device__ void softmax_register_based<float, float>(
    const float* __restrict__ x,
    float* __restrict__ dst,
    const int ncols
) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int block_size = blockDim.y;
    const int tid = threadIdx.y;
    const int row_offset = row * ncols;
    
    // Check 16-byte alignment for float4 vectorization
    // Both pointers must be aligned AND row_offset must be multiple of 4 elements (16 bytes)
    const bool x_aligned = ((uintptr_t)x % 16) == 0;
    const bool dst_aligned = ((uintptr_t)dst % 16) == 0;
    const bool row_aligned = (row_offset % 4) == 0;
    const bool can_vectorize = x_aligned && dst_aligned && row_aligned && (ncols >= 4);
    
    if (can_vectorize) {
        // Handle vectorized portion (4 elements at a time)
        const int ncols_vec4 = (ncols / 4) * 4;
        
        // Phase 1: Online max + sum with vectorized loads
        OnlineSoftmaxState state;
        
        // Vectorized loop for aligned portion
        #pragma unroll 2
        for (int col = tid * 4; col < ncols_vec4; col += block_size * 4) {
            float4 vals = ldg_load_f32x4(x + row_offset + col);
            state.update(vals.x);
            state.update(vals.y);
            state.update(vals.z);
            state.update(vals.w);
        }
        
        // Handle remaining elements
        for (int col = ncols_vec4 + tid; col < ncols; col += block_size) {
            float val = __ldg(x + row_offset + col);
            state.update(val);
        }
        
        // Reduce across block
        state = block_reduce_online_softmax(state, block_size);
        
        const float max_val = state.max_val;
        const float inv_sum = fast_rcp(state.sum_exp);
        
        // Phase 2: Compute final softmax with vectorized stores using fast_exp::exp4
        #pragma unroll 2
        for (int col = tid * 4; col < ncols_vec4; col += block_size * 4) {
            float4 vals = ldg_load_f32x4(x + row_offset + col);
            // Subtract max_val from all components
            float4 shifted = make_float4(
                vals.x - max_val,
                vals.y - max_val,
                vals.z - max_val,
                vals.w - max_val
            );
            // Vectorized exp with Softmax mode (all values are <= 0 after max subtraction)
            float4 exp_vals = fast_exp::exp4<float, fast_exp::Softmax>(shifted);
            float4 result = make_float4(
                exp_vals.x * inv_sum,
                exp_vals.y * inv_sum,
                exp_vals.z * inv_sum,
                exp_vals.w * inv_sum
            );
            *reinterpret_cast<float4*>(dst + row_offset + col) = result;
        }
        
        // Handle remaining elements
        for (int col = ncols_vec4 + tid; col < ncols; col += block_size) {
            float val = __ldg(x + row_offset + col);
            float result = softmax_exp(val - max_val) * inv_sum;
            dst[row_offset + col] = result;
        }
    } else {
        // Scalar fallback path for misaligned data
        OnlineSoftmaxState state;
        
        #pragma unroll 4
        for (int col = tid; col < ncols; col += block_size) {
            float val = __ldg(x + row_offset + col);
            state.update(val);
        }
        
        // Reduce across block
        state = block_reduce_online_softmax(state, block_size);
        
        const float max_val = state.max_val;
        const float inv_sum = fast_rcp(state.sum_exp);
        
        // Phase 2: Compute final softmax output (scalar)
        #pragma unroll 4
        for (int col = tid; col < ncols; col += block_size) {
            float val = __ldg(x + row_offset + col);
            float result = softmax_exp(val - max_val) * inv_sum;
            dst[row_offset + col] = result;
        }
    }
}

// =============================================================================
// HALF PRECISION SOFTMAX WITH __half2 VECTORIZATION
// =============================================================================
// NOTE: half2 vectorized loads/stores require 4-byte alignment.
// We check alignment and fall back to scalar path when misaligned.

template <>
__device__ void softmax_register_based<__half, float>(
    const __half* __restrict__ x,
    __half* __restrict__ dst,
    const int ncols
) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int block_size = blockDim.y;
    const int tid = threadIdx.y;
    const int row_offset = row * ncols;
    
    // Check 4-byte alignment for half2 vectorization
    const bool x_aligned = ((uintptr_t)x % 4) == 0;
    const bool dst_aligned = ((uintptr_t)dst % 4) == 0;
    const bool row_aligned = (row_offset % 2) == 0;
    const bool can_vectorize = x_aligned && dst_aligned && row_aligned && (ncols >= 2);
    
    if (can_vectorize) {
        // Handle half2 vectorized portion (2 elements at a time)
        const int ncols_vec2 = (ncols / 2) * 2;
        
        // Phase 1: Online max + sum
        OnlineSoftmaxState state;
        
        #pragma unroll 4
        for (int col = tid * 2; col < ncols_vec2; col += block_size * 2) {
            __half2 vals = *reinterpret_cast<const __half2*>(x + row_offset + col);
            state.update(__half2float(vals.x));
            state.update(__half2float(vals.y));
        }
        
        // Handle remaining element
        if (ncols_vec2 + tid < ncols && tid == 0) {
            state.update(__half2float(x[row_offset + ncols_vec2]));
        }
        
        state = block_reduce_online_softmax(state, block_size);
        
        const float max_val = state.max_val;
        const float inv_sum = fast_rcp(state.sum_exp);
        
        // Phase 2: Compute final softmax with vectorized stores
        #pragma unroll 4
        for (int col = tid * 2; col < ncols_vec2; col += block_size * 2) {
            __half2 vals = *reinterpret_cast<const __half2*>(x + row_offset + col);
            float2 result;
            result.x = softmax_exp(__half2float(vals.x) - max_val) * inv_sum;
            result.y = softmax_exp(__half2float(vals.y) - max_val) * inv_sum;
            __half2 out = __floats2half2_rn(result.x, result.y);
            *reinterpret_cast<__half2*>(dst + row_offset + col) = out;
        }
        
        // Handle remaining element
        if (ncols_vec2 + tid < ncols && tid == 0) {
            float val = __half2float(x[row_offset + ncols_vec2]);
            float result = softmax_exp(val - max_val) * inv_sum;
            dst[row_offset + ncols_vec2] = __float2half(result);
        }
    } else {
        // Scalar fallback for misaligned data
        OnlineSoftmaxState state;
        
        #pragma unroll 4
        for (int col = tid; col < ncols; col += block_size) {
            float val = __half2float(x[row_offset + col]);
            state.update(val);
        }
        
        state = block_reduce_online_softmax(state, block_size);
        
        const float max_val = state.max_val;
        const float inv_sum = fast_rcp(state.sum_exp);
        
        #pragma unroll 4
        for (int col = tid; col < ncols; col += block_size) {
            float val = __half2float(x[row_offset + col]);
            float result = softmax_exp(val - max_val) * inv_sum;
            dst[row_offset + col] = __float2half(result);
        }
    }
}

// =============================================================================
// BFLOAT16 SOFTMAX
// =============================================================================
// NOTE: bfloat162 vectorized loads/stores require 4-byte alignment.
// We check alignment and fall back to scalar path when misaligned.

template <>
__device__ void softmax_register_based<__nv_bfloat16, float>(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ dst,
    const int ncols
) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int block_size = blockDim.y;
    const int tid = threadIdx.y;
    const int row_offset = row * ncols;
    
    // Check 4-byte alignment for bfloat162 vectorization
    const bool x_aligned = ((uintptr_t)x % 4) == 0;
    const bool dst_aligned = ((uintptr_t)dst % 4) == 0;
    const bool row_aligned = (row_offset % 2) == 0;
    const bool can_vectorize = x_aligned && dst_aligned && row_aligned && (ncols >= 2);
    
    if (can_vectorize) {
        const int ncols_vec2 = (ncols / 2) * 2;
        
        OnlineSoftmaxState state;
        
        #pragma unroll 4
        for (int col = tid * 2; col < ncols_vec2; col += block_size * 2) {
            __nv_bfloat162 vals = *reinterpret_cast<const __nv_bfloat162*>(x + row_offset + col);
            state.update(__bfloat162float(vals.x));
            state.update(__bfloat162float(vals.y));
        }
        
        if (ncols_vec2 + tid < ncols && tid == 0) {
            state.update(__bfloat162float(x[row_offset + ncols_vec2]));
        }
        
        state = block_reduce_online_softmax(state, block_size);
        
        const float max_val = state.max_val;
        const float inv_sum = fast_rcp(state.sum_exp);
        
        #pragma unroll 4
        for (int col = tid * 2; col < ncols_vec2; col += block_size * 2) {
            __nv_bfloat162 vals = *reinterpret_cast<const __nv_bfloat162*>(x + row_offset + col);
            float r0 = softmax_exp(__bfloat162float(vals.x) - max_val) * inv_sum;
            float r1 = softmax_exp(__bfloat162float(vals.y) - max_val) * inv_sum;
            __nv_bfloat162 out = __floats2bfloat162_rn(r0, r1);
            *reinterpret_cast<__nv_bfloat162*>(dst + row_offset + col) = out;
        }
        
        if (ncols_vec2 + tid < ncols && tid == 0) {
            float val = __bfloat162float(x[row_offset + ncols_vec2]);
            float result = softmax_exp(val - max_val) * inv_sum;
            dst[row_offset + ncols_vec2] = __float2bfloat16(result);
        }
    } else {
        // Scalar fallback for misaligned data
        OnlineSoftmaxState state;
        
        #pragma unroll 4
        for (int col = tid; col < ncols; col += block_size) {
            float val = __bfloat162float(x[row_offset + col]);
            state.update(val);
        }
        
        state = block_reduce_online_softmax(state, block_size);
        
        const float max_val = state.max_val;
        const float inv_sum = fast_rcp(state.sum_exp);
        
        #pragma unroll 4
        for (int col = tid; col < ncols; col += block_size) {
            float val = __bfloat162float(x[row_offset + col]);
            float result = softmax_exp(val - max_val) * inv_sum;
            dst[row_offset + col] = __float2bfloat16(result);
        }
    }
}

// =============================================================================
// DOUBLE PRECISION SOFTMAX
// =============================================================================

template <>
__device__ void softmax_register_based<double, double>(
    const double* __restrict__ x,
    double* __restrict__ dst,
    const int ncols
) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int block_size = blockDim.y;
    const int tid = threadIdx.y;
    const int row_offset = row * ncols;
    
    // Online max + sum computation
    double max_val = NEG_INF_D;
    double sum_exp = 0.0;
    
    #pragma unroll 4
    for (int col = tid; col < ncols; col += block_size) {
        double val = __ldg(x + row_offset + col);
        if (val > max_val) {
            sum_exp = sum_exp * exp(max_val - val) + 1.0;
            max_val = val;
        } else {
            // Use fmax to handle -inf - (-inf) = NaN case branchlessly
            // fmax(NaN, -inf) = -inf per IEEE 754-2008, then exp(-inf) = 0
            sum_exp += exp(fmax(val - max_val, NEG_INF_D));
        }
    }
    
    // Reduce max across block (using float versions for reduction then convert)
    // For simplicity, use the old 2-pass approach for double
    double global_max = max_val;
    
    // Warp reduce max
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        global_max = fmax(global_max, __shfl_xor_sync(0xffffffff, global_max, mask, 32));
    }
    
    // Block reduce max (if needed)
    if (block_size > WARP_SIZE) {
        __shared__ double s_max[32];
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        const int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
        
        if (lane_id == 0) s_max[warp_id] = global_max;
        __syncthreads();
        
        if (warp_id == 0 && lane_id < num_warps) {
            global_max = s_max[lane_id];
        }
        if (warp_id == 0) {
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                global_max = fmax(global_max, __shfl_xor_sync(0xffffffff, global_max, mask, 32));
            }
        }
        if (tid == 0) s_max[0] = global_max;
        __syncthreads();
        global_max = s_max[0];
    }
    
    // Recompute sum with global max
    double total_sum = 0.0;
    #pragma unroll 4
    for (int col = tid; col < ncols; col += block_size) {
        double val = __ldg(x + row_offset + col);
        total_sum += exp(val - global_max);
    }
    
    // Reduce sum
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        total_sum += __shfl_xor_sync(0xffffffff, total_sum, mask, 32);
    }
    
    if (block_size > WARP_SIZE) {
        __shared__ double s_sum[32];
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        const int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
        
        if (lane_id == 0) s_sum[warp_id] = total_sum;
        __syncthreads();
        
        if (warp_id == 0 && lane_id < num_warps) {
            total_sum = s_sum[lane_id];
        }
        if (warp_id == 0) {
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                total_sum += __shfl_xor_sync(0xffffffff, total_sum, mask, 32);
            }
        }
        if (tid == 0) s_sum[0] = total_sum;
        __syncthreads();
        total_sum = s_sum[0];
    }
    
    const double inv_sum = 1.0 / total_sum;
    
    #pragma unroll 4
    for (int col = tid; col < ncols; col += block_size) {
        double val = __ldg(x + row_offset + col);
        dst[row_offset + col] = exp(val - global_max) * inv_sum;
    }
}
