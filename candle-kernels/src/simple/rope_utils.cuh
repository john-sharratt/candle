#pragma once
// =============================================================================
// ROPE (ROTARY POSITION EMBEDDING) UTILITIES
// =============================================================================
// Optimized primitives for RoPE operations in transformer models.
//
// Key optimizations:
//   - Vectorized pair loading (float2, __half2, __nv_bfloat162)
//   - FMA instructions for rotation computation
//   - __ldg() for read-only cos/sin tables
//   - Precomputed reciprocals to eliminate division
//   - Template specializations for common head dimensions
// =============================================================================

#include "cuda_utils.cuh"
#include <stdint.h>

// =============================================================================
// VECTORIZED PAIR TYPES
// =============================================================================
// RoPE operates on pairs of elements: (x, x') -> (x*c - x'*s, x*s + x'*c)
// Using vector types allows loading/storing both elements in one transaction.

template <typename T>
struct rope_pair_traits {
    // Default: no vectorization
    using pair_type = T;
    static constexpr bool has_vector = false;
};

template <>
struct rope_pair_traits<float> {
    using pair_type = float2;
    static constexpr bool has_vector = true;
};

template <>
struct rope_pair_traits<double> {
    using pair_type = double2;
    static constexpr bool has_vector = true;
};

template <>
struct rope_pair_traits<__half> {
    using pair_type = __half2;
    static constexpr bool has_vector = true;
};

template <>
struct rope_pair_traits<__nv_bfloat16> {
    using pair_type = __nv_bfloat162;
    static constexpr bool has_vector = true;
};

// FP8E4M3: no vectorized pair type, use scalar
template <>
struct rope_pair_traits<__nv_fp8_e4m3> {
    using pair_type = __nv_fp8_e4m3;
    static constexpr bool has_vector = false;
};

// =============================================================================
// VECTORIZED LOAD/STORE FOR PAIRS
// =============================================================================

// Load a pair of elements as vector
template <typename T>
__device__ __forceinline__ typename rope_pair_traits<T>::pair_type 
ldg_load_pair(const T* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const typename rope_pair_traits<T>::pair_type*>(ptr));
}

// Store a pair of elements
template <typename T>
__device__ __forceinline__ void 
store_pair(T* __restrict__ ptr, typename rope_pair_traits<T>::pair_type val) {
    *reinterpret_cast<typename rope_pair_traits<T>::pair_type*>(ptr) = val;
}

// =============================================================================
// FMA-BASED ROTATION FOR FLOAT
// =============================================================================
// RoPE rotation: (x, y) -> (x*c - y*s, x*s + y*c)
// Using FMA: result1 = fma(x, c, -y*s), result2 = fma(x, s, y*c)

__device__ __forceinline__ float2 
rope_rotate_f32(float2 xy, float c, float s) {
    float2 result;
    // dst.x = x*c - y*s = fma(x, c, -y*s) = fma(x, c, fma(-y, s, 0)) but simpler:
    result.x = __fmaf_rn(xy.x, c, -xy.y * s);
    result.y = __fmaf_rn(xy.x, s, xy.y * c);
    return result;
}

__device__ __forceinline__ double2 
rope_rotate_f64(double2 xy, double c, double s) {
    double2 result;
    result.x = fma(xy.x, c, -xy.y * s);
    result.y = fma(xy.x, s, xy.y * c);
    return result;
}

// =============================================================================
// HALF PRECISION ROTATION (2x throughput using __half2)
// =============================================================================
__device__ __forceinline__ __half2 
rope_rotate_f16(__half2 xy, __half c, __half s) {
    // Create c2 and s2 pairs for vectorized ops
    __half2 c2 = __half2half2(c);
    __half2 s2 = __half2half2(s);
    
    // xy = (x, y), need to output (x*c - y*s, x*s + y*c)
    // Swizzle: create (y, x) by swapping
    __half2 yx = __lowhigh2highlow(xy);
    
    // x*c and y*c
    __half2 xy_times_c = __hmul2(xy, c2);
    // y*s and x*s  
    __half2 yx_times_s = __hmul2(yx, s2);
    
    // result.x = xy.x * c - xy.y * s = xy_times_c.x - yx_times_s.x (where yx.x = y)
    // result.y = xy.x * s + xy.y * c = yx_times_s.y (where yx.y = x) + xy_times_c.y
    // This doesn't quite work with standard __half2 ops, so we do it element-wise
    __half2 result;
    result.x = __hsub(xy_times_c.x, yx_times_s.x);  // x*c - y*s
    result.y = __hadd(yx_times_s.y, xy_times_c.y);  // x*s + y*c
    
    return result;
}

// Alternative: scalar rotation for __half (may be faster on some GPUs)
__device__ __forceinline__ void
rope_rotate_f16_scalar(__half x, __half y, __half c, __half s, __half& out_x, __half& out_y) {
    // Convert to float for rotation (avoids precision issues)
    float xf = __half2float(x);
    float yf = __half2float(y);
    float cf = __half2float(c);
    float sf = __half2float(s);
    
    out_x = __float2half(__fmaf_rn(xf, cf, -yf * sf));
    out_y = __float2half(__fmaf_rn(xf, sf, yf * cf));
}

// =============================================================================
// BFLOAT16 PRECISION ROTATION
// =============================================================================
__device__ __forceinline__ void
rope_rotate_bf16_scalar(__nv_bfloat16 x, __nv_bfloat16 y, __nv_bfloat16 c, __nv_bfloat16 s,
                        __nv_bfloat16& out_x, __nv_bfloat16& out_y) {
    float xf = __bfloat162float(x);
    float yf = __bfloat162float(y);
    float cf = __bfloat162float(c);
    float sf = __bfloat162float(s);
    
    out_x = __float2bfloat16(__fmaf_rn(xf, cf, -yf * sf));
    out_y = __float2bfloat16(__fmaf_rn(xf, sf, yf * cf));
}

// =============================================================================
// GENERIC ROTATION DISPATCH
// =============================================================================

template <typename T>
__device__ __forceinline__ void
rope_rotate(T x, T y, T c, T s, T& out_x, T& out_y);

template <>
__device__ __forceinline__ void
rope_rotate<float>(float x, float y, float c, float s, float& out_x, float& out_y) {
    out_x = __fmaf_rn(x, c, -y * s);
    out_y = __fmaf_rn(x, s, y * c);
}

template <>
__device__ __forceinline__ void
rope_rotate<double>(double x, double y, double c, double s, double& out_x, double& out_y) {
    out_x = fma(x, c, -y * s);
    out_y = fma(x, s, y * c);
}

template <>
__device__ __forceinline__ void
rope_rotate<__half>(__half x, __half y, __half c, __half s, __half& out_x, __half& out_y) {
    rope_rotate_f16_scalar(x, y, c, s, out_x, out_y);
}

template <>
__device__ __forceinline__ void
rope_rotate<__nv_bfloat16>(__nv_bfloat16 x, __nv_bfloat16 y, __nv_bfloat16 c, __nv_bfloat16 s,
                           __nv_bfloat16& out_x, __nv_bfloat16& out_y) {
    rope_rotate_bf16_scalar(x, y, c, s, out_x, out_y);
}

// FP8E4M3 conversion helpers (forward declared from vec_utils.cuh)
__device__ __forceinline__ float fp8e4m3_to_float_rope(__nv_fp8_e4m3 v) {
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

__device__ __forceinline__ __nv_fp8_e4m3 float_to_fp8e4m3_rope(float v) {
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
__device__ __forceinline__ void
rope_rotate<__nv_fp8_e4m3>(__nv_fp8_e4m3 x, __nv_fp8_e4m3 y, __nv_fp8_e4m3 c, __nv_fp8_e4m3 s,
                           __nv_fp8_e4m3& out_x, __nv_fp8_e4m3& out_y) {
    // Convert to float for rotation (FP8 has very limited precision)
    float xf = fp8e4m3_to_float_rope(x);
    float yf = fp8e4m3_to_float_rope(y);
    float cf = fp8e4m3_to_float_rope(c);
    float sf = fp8e4m3_to_float_rope(s);
    
    out_x = float_to_fp8e4m3_rope(__fmaf_rn(xf, cf, -yf * sf));
    out_y = float_to_fp8e4m3_rope(__fmaf_rn(xf, sf, yf * cf));
}

// =============================================================================
// READ-ONLY CACHE LOADS FOR COS/SIN
// =============================================================================

template <typename T>
__device__ __forceinline__ T ldg_cos_sin(const T* __restrict__ ptr) {
    return __ldg(ptr);
}

// Specializations for types not directly supported by __ldg
template <>
__device__ __forceinline__ __half ldg_cos_sin<__half>(const __half* __restrict__ ptr) {
    return *ptr;  // __half not directly supported
}

template <>
__device__ __forceinline__ __nv_bfloat16 ldg_cos_sin<__nv_bfloat16>(const __nv_bfloat16* __restrict__ ptr) {
    return *ptr;  // __nv_bfloat16 not directly supported
}

// =============================================================================
// FAST INTEGER DIVISION HELPERS
// =============================================================================
// Division by (d/2) can be replaced with multiplication by reciprocal + shift
// when d is known at compile time or is a power of 2.

// Check if value is power of 2
__device__ __forceinline__ bool is_power_of_2(uint32_t v) {
    return v && !(v & (v - 1));
}

// Fast divide by 2
__device__ __forceinline__ uint32_t fast_div2(uint32_t v) {
    return v >> 1;
}

// Fast modulo by power of 2
__device__ __forceinline__ uint32_t fast_mod_pow2(uint32_t v, uint32_t pow2) {
    return v & (pow2 - 1);
}

// =============================================================================
// OPTIMIZED ROPE IMPLEMENTATION FOR INTERLEAVED LAYOUT (ropei)
// =============================================================================
// Layout: [..., (x0, x0'), (x1, x1'), ...] - pairs are adjacent
// This is the simplest case where we can use vectorized pair loads.

template <typename T>
__device__ __forceinline__ void
rope_interleaved_optimized(
    const T* __restrict__ src,
    const T* __restrict__ cos,
    const T* __restrict__ sin,
    T* __restrict__ dst,
    const uint32_t total_pairs,      // bh * td / 2
    const uint32_t half_td,          // td / 2 (precomputed)
    const uint32_t stride_b_pairs    // stride_b / 2 (0 if no batch stride)
) {
    const uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) return;
    
    // Calculate cos/sin index
    uint32_t rope_idx = pair_idx % half_td;
    if (stride_b_pairs > 0) {
        uint32_t b_idx = pair_idx / stride_b_pairs;
        rope_idx += b_idx * half_td;
    }
    
    // Load cos/sin with cache hint
    T c = ldg_cos_sin(cos + rope_idx);
    T s = ldg_cos_sin(sin + rope_idx);
    
    // Load pair, rotate, store
    const uint32_t elem_idx = pair_idx * 2;
    T x = __ldg(src + elem_idx);
    T y = __ldg(src + elem_idx + 1);
    
    T out_x, out_y;
    rope_rotate(x, y, c, s, out_x, out_y);
    
    dst[elem_idx] = out_x;
    dst[elem_idx + 1] = out_y;
}

// Float specialization with vectorized pair load/store
template <>
__device__ __forceinline__ void
rope_interleaved_optimized<float>(
    const float* __restrict__ src,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ dst,
    const uint32_t total_pairs,
    const uint32_t half_td,
    const uint32_t stride_b_pairs
) {
    const uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) return;
    
    uint32_t rope_idx = pair_idx % half_td;
    if (stride_b_pairs > 0) {
        uint32_t b_idx = pair_idx / stride_b_pairs;
        rope_idx += b_idx * half_td;
    }
    
    float c = __ldg(cos + rope_idx);
    float s = __ldg(sin + rope_idx);
    
    // Vectorized pair load
    float2 xy = __ldg(reinterpret_cast<const float2*>(src + pair_idx * 2));
    
    // FMA-based rotation
    float2 result = rope_rotate_f32(xy, c, s);
    
    // Vectorized store
    *reinterpret_cast<float2*>(dst + pair_idx * 2) = result;
}

// Double specialization with vectorized pair load/store
template <>
__device__ __forceinline__ void
rope_interleaved_optimized<double>(
    const double* __restrict__ src,
    const double* __restrict__ cos,
    const double* __restrict__ sin,
    double* __restrict__ dst,
    const uint32_t total_pairs,
    const uint32_t half_td,
    const uint32_t stride_b_pairs
) {
    const uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) return;
    
    uint32_t rope_idx = pair_idx % half_td;
    if (stride_b_pairs > 0) {
        uint32_t b_idx = pair_idx / stride_b_pairs;
        rope_idx += b_idx * half_td;
    }
    
    double c = __ldg(cos + rope_idx);
    double s = __ldg(sin + rope_idx);
    
    double2 xy = __ldg(reinterpret_cast<const double2*>(src + pair_idx * 2));
    double2 result = rope_rotate_f64(xy, c, s);
    *reinterpret_cast<double2*>(dst + pair_idx * 2) = result;
}

// =============================================================================
// OPTIMIZED ROPE FOR ROTARY LAYOUT (rope)
// =============================================================================
// Layout: [..., x0, x1, ..., x_{d/2-1}, x0', x1', ..., x'_{d/2-1}, ...]
// Pairs are separated by d/2 elements.

template <typename T>
__device__ __forceinline__ void
rope_rotary_optimized(
    const T* __restrict__ src,
    const T* __restrict__ cos,
    const T* __restrict__ sin,
    T* __restrict__ dst,
    const uint32_t total_pairs,      // bh * td / 2
    const uint32_t half_td,          // td / 2
    const uint32_t half_d,           // d / 2
    const uint32_t d,                // head dimension
    const uint32_t stride_b_pairs    // stride_b / 2
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;
    
    // Decompose index using multiplications instead of divisions where possible
    // idx = i_bh * half_td + i_td
    // i_td = i_t * half_d + i_d
    const uint32_t i_bh = idx / half_td;
    const uint32_t i_td = idx - half_td * i_bh;
    const uint32_t i_t = i_td / half_d;
    const uint32_t i_d = i_td - half_d * i_t;
    
    // Calculate element indices
    const uint32_t i1 = i_bh * (half_td * 2) + i_t * d + i_d;
    const uint32_t i2 = i1 + half_d;
    
    // Calculate cos/sin index
    uint32_t i_cs = i_t * half_d + i_d;
    if (stride_b_pairs > 0) {
        uint32_t b_idx = idx / stride_b_pairs;
        i_cs += b_idx * half_td;
    }
    
    T c = ldg_cos_sin(cos + i_cs);
    T s = ldg_cos_sin(sin + i_cs);
    
    T x = __ldg(src + i1);
    T y = __ldg(src + i2);
    
    T out_x, out_y;
    rope_rotate(x, y, c, s, out_x, out_y);
    
    dst[i1] = out_x;
    dst[i2] = out_y;
}

// Float specialization with __ldg and FMA
template <>
__device__ __forceinline__ void
rope_rotary_optimized<float>(
    const float* __restrict__ src,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ dst,
    const uint32_t total_pairs,
    const uint32_t half_td,
    const uint32_t half_d,
    const uint32_t d,
    const uint32_t stride_b_pairs
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;
    
    const uint32_t i_bh = idx / half_td;
    const uint32_t i_td = idx - half_td * i_bh;
    const uint32_t i_t = i_td / half_d;
    const uint32_t i_d = i_td - half_d * i_t;
    
    const uint32_t i1 = i_bh * (half_td * 2) + i_t * d + i_d;
    const uint32_t i2 = i1 + half_d;
    
    uint32_t i_cs = i_t * half_d + i_d;
    if (stride_b_pairs > 0) {
        uint32_t b_idx = idx / stride_b_pairs;
        i_cs += b_idx * half_td;
    }
    
    float c = __ldg(cos + i_cs);
    float s = __ldg(sin + i_cs);
    float x = __ldg(src + i1);
    float y = __ldg(src + i2);
    
    // FMA-based rotation
    dst[i1] = __fmaf_rn(x, c, -y * s);
    dst[i2] = __fmaf_rn(x, s, y * c);
}

// =============================================================================
// OPTIMIZED ROPE FOR THD LAYOUT (rope_thd)
// =============================================================================
// Layout: [batch, time, head, dim]

template <typename T>
__device__ __forceinline__ void
rope_thd_optimized(
    const T* __restrict__ src,
    const T* __restrict__ cos,
    const T* __restrict__ sin,
    T* __restrict__ dst,
    const uint32_t total_pairs,      // b * t * h * d / 2
    const uint32_t half_d,           // d / 2
    const uint32_t d,                // head dimension
    const uint32_t h,                // num heads
    const uint32_t t,                // sequence length
    const uint32_t half_td,          // t * d / 2 (for stride_b calc)
    const uint32_t stride_b_pairs    // stride_b / 2 (0 if no batch stride)
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;
    
    // Decompose: idx = i_bth * half_d + i_d
    const uint32_t i_bth = idx / half_d;
    const uint32_t i_d = idx - half_d * i_bth;
    const uint32_t i_t = (i_bth / h) % t;
    
    // Element indices
    const uint32_t i1 = i_bth * d + i_d;
    const uint32_t i2 = i1 + half_d;
    
    // Cos/sin index
    uint32_t i_cs = i_t * half_d + i_d;
    if (stride_b_pairs > 0) {
        uint32_t b_idx = idx / stride_b_pairs;
        i_cs += b_idx * half_td;
    }
    
    T c = ldg_cos_sin(cos + i_cs);
    T s = ldg_cos_sin(sin + i_cs);
    
    T x = __ldg(src + i1);
    T y = __ldg(src + i2);
    
    T out_x, out_y;
    rope_rotate(x, y, c, s, out_x, out_y);
    
    dst[i1] = out_x;
    dst[i2] = out_y;
}

// Float specialization
template <>
__device__ __forceinline__ void
rope_thd_optimized<float>(
    const float* __restrict__ src,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ dst,
    const uint32_t total_pairs,
    const uint32_t half_d,
    const uint32_t d,
    const uint32_t h,
    const uint32_t t,
    const uint32_t half_td,
    const uint32_t stride_b_pairs
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;
    
    const uint32_t i_bth = idx / half_d;
    const uint32_t i_d = idx - half_d * i_bth;
    const uint32_t i_t = (i_bth / h) % t;
    
    const uint32_t i1 = i_bth * d + i_d;
    const uint32_t i2 = i1 + half_d;
    
    uint32_t i_cs = i_t * half_d + i_d;
    if (stride_b_pairs > 0) {
        uint32_t b_idx = idx / stride_b_pairs;
        i_cs += b_idx * half_td;
    }
    
    float c = __ldg(cos + i_cs);
    float s = __ldg(sin + i_cs);
    float x = __ldg(src + i1);
    float y = __ldg(src + i2);
    
    dst[i1] = __fmaf_rn(x, c, -y * s);
    dst[i2] = __fmaf_rn(x, s, y * c);
}
