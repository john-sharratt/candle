// =============================================================================
// Vectorized Load/Store Utilities for CUDA
// =============================================================================
// Type traits and helper functions for efficient vectorized memory access.
// Supports float, double, __half (fp16), and __nv_bfloat16 (bf16).
//
// Key components:
//   - VecTraits<T>: Type traits for vectorization (VEC_SIZE and VecType)
//   - load_and_square_sum(): Load vector and compute sum of squares
//   - store_scaled_vec*(): Store scaled values with vectorization
// =============================================================================

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>

#include <cuda_bf16.h>

// =============================================================================
// Vector Traits: Determine optimal vector size for each type
// =============================================================================

// Default: no vectorization
template <typename T> 
struct VecTraits { 
    static constexpr int VEC_SIZE = 1;
    using VecType = T;
};

// float: Use float4 (128-bit loads, 4 elements)
template <> 
struct VecTraits<float> { 
    static constexpr int VEC_SIZE = 4;
    using VecType = float4;
};

// double: Use double2 (128-bit loads, 2 elements)
template <> 
struct VecTraits<double> { 
    static constexpr int VEC_SIZE = 2;
    using VecType = double2;
};

// __half: Use __half2 (32-bit loads, 2 elements)
// Note: Could use __half4 on newer architectures but half2 is more portable
template <> 
struct VecTraits<__half> { 
    static constexpr int VEC_SIZE = 2;
    using VecType = __half2;
};

// __nv_bfloat16: Use __nv_bfloat162 (32-bit loads, 2 elements)
template <> 
struct VecTraits<__nv_bfloat16> { 
    static constexpr int VEC_SIZE = 2;
    using VecType = __nv_bfloat162;
};

// __nv_fp8_e4m3: No vectorized type available, use scalar
template <> 
struct VecTraits<__nv_fp8_e4m3> { 
    static constexpr int VEC_SIZE = 1;
    using VecType = __nv_fp8_e4m3;
};

// =============================================================================
// Vectorized Load + Sum of Squares
// =============================================================================
// These functions load VEC_SIZE elements and return sum(x_i^2)
// Used in RMSNorm first pass

__device__ __forceinline__ float load_and_square_sum(const float4& v) {
    // FMA chain: ((v.w^2) + v.z^2) + v.y^2) + v.x^2
    // Could also write as sum of individual squares - compiler will optimize
    return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
}

__device__ __forceinline__ float load_and_square_sum(const double2& v) {
    // Convert to float for accumulation (matches RMSNorm f32 accumulation)
    float x = static_cast<float>(v.x);
    float y = static_cast<float>(v.y);
    return x * x + y * y;
}

__device__ __forceinline__ float load_and_square_sum(float v) {
    return v * v;
}

__device__ __forceinline__ float load_and_square_sum(double v) {
    float vf = static_cast<float>(v);
    return vf * vf;
}

__device__ __forceinline__ float load_and_square_sum(const __half2& v) {
    float2 vf = __half22float2(v);
    return vf.x * vf.x + vf.y * vf.y;
}

__device__ __forceinline__ float load_and_square_sum(__half v) {
    float vf = __half2float(v);
    return vf * vf;
}

__device__ __forceinline__ float load_and_square_sum(const __nv_bfloat162& v) {
    float2 vf = __bfloat1622float2(v);
    return vf.x * vf.x + vf.y * vf.y;
}

__device__ __forceinline__ float load_and_square_sum(__nv_bfloat16 v) {
    float vf = __bfloat162float(v);
    return vf * vf;
}

// FP8E4M3 conversion helper
__device__ __forceinline__ float fp8e4m3_to_float(__nv_fp8_e4m3 v) {
#if __CUDA_ARCH__ >= 890
    // Hardware conversion on Ada/Hopper+
    __nv_fp8_storage_t storage = *reinterpret_cast<const __nv_fp8_storage_t*>(&v);
    return __half2float(__nv_cvt_fp8_to_halfraw(storage, __NV_E4M3));
#else
    // Software fallback for SM80-SM88 (Ampere)
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
        return __int_as_float(0x7FC00000);  // quiet NaN
    } else {
        float m = 1.0f + mant / 8.0f;
        float result = ldexpf(m, (int)exp - 7);
        return sign ? -result : result;
    }
#endif
}

__device__ __forceinline__ __nv_fp8_e4m3 float_to_fp8e4m3(float v) {
#if __CUDA_ARCH__ >= 890
    __nv_fp8_storage_t storage = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 result;
    *reinterpret_cast<__nv_fp8_storage_t*>(&result) = storage;
    return result;
#else
    // Software fallback
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
        *out = (sign << 7) | (14 << 3) | 7;  // saturate to max
        return result;
    }
    if (exp < -9) {
        *out = sign << 7;  // underflow to zero
        return result;
    }
    
    int32_t e4m3_exp = exp + 7;
    uint32_t e4m3_mant;
    
    if (e4m3_exp <= 0) {
        int shift = 1 - e4m3_exp + 20;
        e4m3_mant = ((1 << 23) | mant) >> shift;
        e4m3_exp = 0;
    } else {
        e4m3_mant = (mant + (1 << 19)) >> 20;  // round
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

__device__ __forceinline__ float load_and_square_sum(__nv_fp8_e4m3 v) {
    float vf = fp8e4m3_to_float(v);
    return vf * vf;
}

// =============================================================================
// Vectorized Stores (without alpha)
// =============================================================================
// Store scaled values from f32 cache to output type

__device__ __forceinline__ void store_scaled(float* dst, const float* cache, float scale, int idx) {
    dst[idx] = scale * cache[idx];
}

__device__ __forceinline__ void store_scaled_vec4(float* dst, const float* cache, float scale, int idx) {
    float4 out;
    out.x = scale * cache[idx];
    out.y = scale * cache[idx + 1];
    out.z = scale * cache[idx + 2];
    out.w = scale * cache[idx + 3];
    *reinterpret_cast<float4*>(&dst[idx]) = out;
}

__device__ __forceinline__ void store_scaled(double* dst, const float* cache, float scale, int idx) {
    dst[idx] = static_cast<double>(scale * cache[idx]);
}

__device__ __forceinline__ void store_scaled_vec2(double* dst, const float* cache, float scale, int idx) {
    double2 out;
    out.x = static_cast<double>(scale * cache[idx]);
    out.y = static_cast<double>(scale * cache[idx + 1]);
    *reinterpret_cast<double2*>(&dst[idx]) = out;
}

__device__ __forceinline__ void store_scaled(__half* dst, const float* cache, float scale, int idx) {
    dst[idx] = __float2half(scale * cache[idx]);
}

__device__ __forceinline__ void store_scaled_vec2(__half* dst, const float* cache, float scale, int idx) {
    __half2 out = __floats2half2_rn(scale * cache[idx], scale * cache[idx + 1]);
    *reinterpret_cast<__half2*>(&dst[idx]) = out;
}

__device__ __forceinline__ void store_scaled(__nv_bfloat16* dst, const float* cache, float scale, int idx) {
    dst[idx] = __float2bfloat16(scale * cache[idx]);
}

__device__ __forceinline__ void store_scaled_vec2(__nv_bfloat16* dst, const float* cache, float scale, int idx) {
    __nv_bfloat162 out = __floats2bfloat162_rn(scale * cache[idx], scale * cache[idx + 1]);
    *reinterpret_cast<__nv_bfloat162*>(&dst[idx]) = out;
}

__device__ __forceinline__ void store_scaled(__nv_fp8_e4m3* dst, const float* cache, float scale, int idx) {
    dst[idx] = float_to_fp8e4m3(scale * cache[idx]);
}

// =============================================================================
// Vectorized Stores (with alpha scaling)
// =============================================================================
// Store scaled values with additional alpha multiplier

__device__ __forceinline__ void store_scaled_alpha(float* dst, const float* cache, 
                                                    float alpha, float scale, int idx) {
    dst[idx] = scale * cache[idx] * alpha;
}

__device__ __forceinline__ void store_scaled_alpha_vec4(float* dst, const float* cache,
                                                         const float* alpha_vals, float scale, int idx) {
    float4 out;
    out.x = scale * cache[idx] * alpha_vals[0];
    out.y = scale * cache[idx + 1] * alpha_vals[1];
    out.z = scale * cache[idx + 2] * alpha_vals[2];
    out.w = scale * cache[idx + 3] * alpha_vals[3];
    *reinterpret_cast<float4*>(&dst[idx]) = out;
}

__device__ __forceinline__ void store_scaled_alpha(double* dst, const float* cache,
                                                    float alpha, float scale, int idx) {
    dst[idx] = static_cast<double>(scale * cache[idx] * alpha);
}

__device__ __forceinline__ void store_scaled_alpha_vec2(double* dst, const float* cache,
                                                         float alpha0, float alpha1, float scale, int idx) {
    double2 out;
    out.x = static_cast<double>(scale * cache[idx] * alpha0);
    out.y = static_cast<double>(scale * cache[idx + 1] * alpha1);
    *reinterpret_cast<double2*>(&dst[idx]) = out;
}

__device__ __forceinline__ void store_scaled_alpha(__half* dst, const float* cache,
                                                    float alpha, float scale, int idx) {
    dst[idx] = __float2half(scale * cache[idx] * alpha);
}

__device__ __forceinline__ void store_scaled_alpha_vec2(__half* dst, const float* cache,
                                                         float alpha0, float alpha1, float scale, int idx) {
    __half2 out = __floats2half2_rn(scale * cache[idx] * alpha0, 
                                     scale * cache[idx + 1] * alpha1);
    *reinterpret_cast<__half2*>(&dst[idx]) = out;
}

__device__ __forceinline__ void store_scaled_alpha(__nv_bfloat16* dst, const float* cache,
                                                    float alpha, float scale, int idx) {
    dst[idx] = __float2bfloat16(scale * cache[idx] * alpha);
}

__device__ __forceinline__ void store_scaled_alpha_vec2(__nv_bfloat16* dst, const float* cache,
                                                         float alpha0, float alpha1, float scale, int idx) {
    __nv_bfloat162 out = __floats2bfloat162_rn(scale * cache[idx] * alpha0,
                                                scale * cache[idx + 1] * alpha1);
    *reinterpret_cast<__nv_bfloat162*>(&dst[idx]) = out;
}

__device__ __forceinline__ void store_scaled_alpha(__nv_fp8_e4m3* dst, const float* cache,
                                                    float alpha, float scale, int idx) {
    dst[idx] = float_to_fp8e4m3(scale * cache[idx] * alpha);
}

// =============================================================================
// Helper: Cache vectorized loads to f32 shared memory
// =============================================================================
// These helpers load VEC_SIZE elements and store to f32 cache

template <typename T>
__device__ __forceinline__ void cache_to_f32(const T* src, float* cache, int col) {
    cache[col] = static_cast<float>(src[col]);
}

__device__ __forceinline__ void cache_vec4_to_f32(const float4& v, float* cache, int col) {
    cache[col] = v.x;
    cache[col + 1] = v.y;
    cache[col + 2] = v.z;
    cache[col + 3] = v.w;
}

__device__ __forceinline__ void cache_vec2_to_f32(const double2& v, float* cache, int col) {
    cache[col] = static_cast<float>(v.x);
    cache[col + 1] = static_cast<float>(v.y);
}

__device__ __forceinline__ void cache_vec2_to_f32(const __half2& v, float* cache, int col) {
    float2 vf = __half22float2(v);
    cache[col] = vf.x;
    cache[col + 1] = vf.y;
}

__device__ __forceinline__ void cache_vec2_to_f32(const __nv_bfloat162& v, float* cache, int col) {
    float2 vf = __bfloat1622float2(v);
    cache[col] = vf.x;
    cache[col + 1] = vf.y;
}

__device__ __forceinline__ void cache_to_f32(const __nv_fp8_e4m3* src, float* cache, int col) {
    cache[col] = fp8e4m3_to_float(src[col]);
}
