// The conversions between the kernel compute types, and the scalar and vector
// type utilities.
//
// `to_f32`, `to_half`, `to_bf16` and `from_f32` convert any supported type to
// float, half or bfloat16 through the direct instruction where one exists.
// `from_int` and `to_acc` cover integer and accumulator conversions, and the
// FP8 paths use the LUT where it is faster.

// ============================================================================
// TYPE CONVERSION UTILITIES
// ============================================================================

/// Convert any supported type to float32
template <typename T>
static __device__ __forceinline__ float to_f32(T v);

template <>
__device__ __forceinline__ float to_f32<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ float to_f32<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float to_f32<__nv_fp8_e4m3>(__nv_fp8_e4m3 v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    // Hardware conversion on Ada/Hopper+
    union { __nv_fp8_e4m3 f; __nv_fp8_storage_t s; } vu;
    vu.f = v;
    return __half2float(__nv_cvt_fp8_to_halfraw(vu.s, __NV_E4M3));
#else
    // Software fallback for SM80-SM88 (Ampere)
    // E4M3 format: 1 sign, 4 exponent, 3 mantissa, bias=7
    union { __nv_fp8_e4m3 f; uint8_t u; } vu;
    vu.f = v;
    uint8_t bits = vu.u;
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

template <>
__device__ __forceinline__ float to_f32<__nv_fp8_e5m2>(__nv_fp8_e5m2 v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    union { __nv_fp8_e5m2 f; __nv_fp8_storage_t s; } vu;
    vu.f = v;
    return __half2float(__nv_cvt_fp8_to_halfraw(vu.s, __NV_E5M2));
#else
    // Software fallback: E5M2 format: 1 sign, 5 exponent, 2 mantissa, bias=15
    union { __nv_fp8_e5m2 f; uint8_t u; } vu;
    vu.f = v;
    uint8_t bits = vu.u;
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp = (bits >> 2) & 0x1F;
    uint32_t mant = bits & 0x3;
    
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float m = mant / 4.0f;
        float result = ldexpf(m, -14);
        return sign ? -result : result;
    } else if (exp == 31) {
        return (mant == 0) ? (sign ? -INFINITY : INFINITY) : __int_as_float(0x7FC00000);
    } else {
        float m = 1.0f + mant / 4.0f;
        float result = ldexpf(m, (int)exp - 15);
        return sign ? -result : result;
    }
#endif
}

/// Alias for to_f32 - convert any compute type to float
template <typename T>
static __device__ __forceinline__ float to_float(T v) {
    return to_f32(v);
}

/// Convert any supported type to __half (FP16)
/// Optimized paths for each type - avoids unnecessary intermediate conversions
template <typename T>
static __device__ __forceinline__ __half to_half(T v);

template <>
__device__ __forceinline__ __half to_half<__half>(__half v) {
    return v;  // Already half - no conversion needed
}

template <>
__device__ __forceinline__ __half to_half<float>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ __half to_half<__nv_bfloat16>(__nv_bfloat16 v) {
    // BF16 must go through float (no direct BF16→F16 instruction)
    return __float2half(__bfloat162float(v));
}

template <>
__device__ __forceinline__ __half to_half<__nv_fp8_e4m3>(__nv_fp8_e4m3 v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    // Direct FP8 → half conversion on Ada/Hopper+
    union { __nv_fp8_e4m3 f; __nv_fp8_storage_t s; } vu;
    vu.f = v;
    __half_raw hr = __nv_cvt_fp8_to_halfraw(vu.s, __NV_E4M3);
    union { __half_raw r; __half h; } hu;
    hu.r = hr;
    return hu.h;
#else
    return __float2half(to_f32(v));
#endif
}

template <>
__device__ __forceinline__ __half to_half<__nv_fp8_e5m2>(__nv_fp8_e5m2 v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    // Direct FP8 → half conversion on Ada/Hopper+
    union { __nv_fp8_e5m2 f; __nv_fp8_storage_t s; } vu;
    vu.f = v;
    __half_raw hr = __nv_cvt_fp8_to_halfraw(vu.s, __NV_E5M2);
    union { __half_raw r; __half h; } hu;
    hu.r = hr;
    return hu.h;
#else
    return __float2half(to_f32(v));
#endif
}

/// Convert any supported type to __nv_bfloat16 (BF16)
/// Optimized paths for each type - avoids unnecessary intermediate conversions
template <typename T>
static __device__ __forceinline__ __nv_bfloat16 to_bf16(T v);

template <>
__device__ __forceinline__ __nv_bfloat16 to_bf16<__nv_bfloat16>(__nv_bfloat16 v) {
    return v;  // Already bf16 - no conversion needed
}

template <>
__device__ __forceinline__ __nv_bfloat16 to_bf16<float>(float v) {
    return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 to_bf16<__half>(__half v) {
    // Half must go through float (no direct F16→BF16 instruction)
    return __float2bfloat16(__half2float(v));
}

template <>
__device__ __forceinline__ __nv_bfloat16 to_bf16<__nv_fp8_e4m3>(__nv_fp8_e4m3 v) {
    // FP8 → float → bf16 (no direct path)
    return __float2bfloat16(to_f32(v));
}

template <>
__device__ __forceinline__ __nv_bfloat16 to_bf16<__nv_fp8_e5m2>(__nv_fp8_e5m2 v) {
    // FP8 → float → bf16 (no direct path)
    return __float2bfloat16(to_f32(v));
}

/// Convert float32 to any supported type
template <typename T>
static __device__ __forceinline__ T from_f32(float v);

template <>
__device__ __forceinline__ float from_f32<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ __half from_f32<__half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}

template <>
__device__ __forceinline__ __nv_fp8_e4m3 from_f32<__nv_fp8_e4m3>(float v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __nv_fp8_storage_t storage = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    union { __nv_fp8_e4m3 f; __nv_fp8_storage_t s; } result;
    result.s = storage;
    return result.f;
#else
    // Software fallback for SM80-SM88
    union { __nv_fp8_e4m3 f; uint8_t u; } result;
    
    uint32_t fbits = __float_as_int(v);
    uint32_t sign = (fbits >> 31) & 1;
    int32_t exp = ((fbits >> 23) & 0xFF) - 127;
    uint32_t mant = fbits & 0x7FFFFF;
    
    if ((fbits & 0x7FFFFFFF) == 0) {
        result.u = sign << 7;
        return result.f;
    }
    if (exp > 8) {
        result.u = (sign << 7) | (14 << 3) | 7;  // saturate to max
        return result.f;
    }
    if (exp < -9) {
        result.u = sign << 7;  // underflow to zero
        return result.f;
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
                result.u = (sign << 7) | (14 << 3) | 7;
                return result.f;
            }
        }
    }
    
    result.u = (sign << 7) | (e4m3_exp << 3) | (e4m3_mant & 0x7);
    return result.f;
#endif
}

template <>
__device__ __forceinline__ __nv_fp8_e5m2 from_f32<__nv_fp8_e5m2>(float v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __nv_fp8_storage_t storage = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E5M2);
    union { __nv_fp8_e5m2 f; __nv_fp8_storage_t s; } result;
    result.s = storage;
    return result.f;
#else
    // Software fallback
    union { __nv_fp8_e5m2 f; uint8_t u; } result;
    
    uint32_t fbits = __float_as_int(v);
    uint32_t sign = (fbits >> 31) & 1;
    int32_t exp = ((fbits >> 23) & 0xFF) - 127;
    uint32_t mant = fbits & 0x7FFFFF;
    
    if ((fbits & 0x7FFFFFFF) == 0) {
        result.u = sign << 7;
        return result.f;
    }
    if (exp > 15) {
        result.u = (sign << 7) | (30 << 2) | 3;  // saturate to max
        return result.f;
    }
    if (exp < -16) {
        result.u = sign << 7;
        return result.f;
    }
    
    int32_t e5m2_exp = exp + 15;
    uint32_t e5m2_mant;
    
    if (e5m2_exp <= 0) {
        int shift = 1 - e5m2_exp + 21;
        e5m2_mant = ((1 << 23) | mant) >> shift;
        e5m2_exp = 0;
    } else {
        e5m2_mant = (mant + (1 << 20)) >> 21;
        if (e5m2_mant >= 4) {
            e5m2_mant = 0;
            e5m2_exp++;
            if (e5m2_exp > 30) {
                result.u = (sign << 7) | (30 << 2) | 3;
                return result.f;
            }
        }
    }
    
    result.u = (sign << 7) | (e5m2_exp << 2) | (e5m2_mant & 0x3);
    return result.f;
#endif
}

/// Convert __nv_fp8x2_e4m3 to float2
/// On SM89+ (Ada/Hopper), uses single-instruction __nv_cvt_fp8x2_to_halfraw2
static __device__ __forceinline__ float2 fp8x2_to_f32(__nv_fp8x2_e4m3 v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    // Hardware vectorized conversion: FP8x2 → half2 → float2 in minimal instructions
    union { __nv_fp8x2_e4m3 f; __nv_fp8x2_storage_t s; } vu;
    vu.f = v;
    __half2_raw h2_raw = __nv_cvt_fp8x2_to_halfraw2(vu.s, __NV_E4M3);
    union { __half2_raw r; __half2 h; } hu;
    hu.r = h2_raw;
    return __half22float2(hu.h);
#else
    // Scalar fallback for older architectures
    union { __nv_fp8x2_e4m3 f; uint8_t bytes[2]; } vu;
    vu.f = v;
    union { uint8_t u; __nv_fp8_e4m3 f; } v0, v1;
    v0.u = vu.bytes[0];
    v1.u = vu.bytes[1];
    float2 result;
    result.x = to_f32<__nv_fp8_e4m3>(v0.f);
    result.y = to_f32<__nv_fp8_e4m3>(v1.f);
    return result;
#endif
}

/// Convert 4 consecutive FP8 values to float4
/// On SM89+, uses 2 vectorized __nv_cvt_fp8x2_to_halfraw2 instructions
/// This is faster than 4 scalar conversions and reduces register pressure
static __device__ __forceinline__ float4 fp8x4_to_f32(const __nv_fp8_e4m3* ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    // Load as two fp8x2 and convert using vectorized intrinsics
    const __nv_fp8x2_storage_t* storage_ptr = reinterpret_cast<const __nv_fp8x2_storage_t*>(ptr);
    __half2_raw h2_lo_raw = __nv_cvt_fp8x2_to_halfraw2(storage_ptr[0], __NV_E4M3);
    __half2_raw h2_hi_raw = __nv_cvt_fp8x2_to_halfraw2(storage_ptr[1], __NV_E4M3);
    __half2 h2_lo = *reinterpret_cast<__half2*>(&h2_lo_raw);
    __half2 h2_hi = *reinterpret_cast<__half2*>(&h2_hi_raw);
    float2 f2_lo = __half22float2(h2_lo);
    float2 f2_hi = __half22float2(h2_hi);
    return make_float4(f2_lo.x, f2_lo.y, f2_hi.x, f2_hi.y);
#else
    // Scalar fallback
    float4 result;
    result.x = to_f32<__nv_fp8_e4m3>(ptr[0]);
    result.y = to_f32<__nv_fp8_e4m3>(ptr[1]);
    result.z = to_f32<__nv_fp8_e4m3>(ptr[2]);
    result.w = to_f32<__nv_fp8_e4m3>(ptr[3]);
    return result;
#endif
}

// ============================================================================
// FMA_RN - Type-aware fused multiply-add (a * b + c)
// ============================================================================
// Selects the optimal FMA intrinsic based on type:
//   fma_rn<float>:  __fmaf_rn
//   fma_rn<__half>: __hfma
//   fma_rn<__nv_bfloat16>: __hfma (bf16)
//
// Usage: T result = fma_rn<T>(a, b, c);

template <typename T>
__device__ __forceinline__ T fma_rn(T a, T b, T c);

template <>
__device__ __forceinline__ float fma_rn<float>(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}

template <>
__device__ __forceinline__ __half fma_rn<__half>(__half a, __half b, __half c) {
    return __hfma(a, b, c);
}

template <>
__device__ __forceinline__ __nv_bfloat16 fma_rn<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
    return __hfma(a, b, c);
}

// FP8 types don't have native FMA - go through float
template <>
__device__ __forceinline__ __nv_fp8_e4m3 fma_rn<__nv_fp8_e4m3>(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b, __nv_fp8_e4m3 c) {
    return from_f32<__nv_fp8_e4m3>(__fmaf_rn(to_f32(a), to_f32(b), to_f32(c)));
}

template <>
__device__ __forceinline__ __nv_fp8_e5m2 fma_rn<__nv_fp8_e5m2>(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b, __nv_fp8_e5m2 c) {
    return from_f32<__nv_fp8_e5m2>(__fmaf_rn(to_f32(a), to_f32(b), to_f32(c)));
}

// Vector types - process 2 elements per instruction
template <>
__device__ __forceinline__ float2 fma_rn<float2>(float2 a, float2 b, float2 c) {
    return make_float2(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y));
}

template <>
__device__ __forceinline__ __half2 fma_rn<__half2>(__half2 a, __half2 b, __half2 c) {
    return __hfma2(a, b, c);  // Single instruction for 2 FMAs
}

template <>
__device__ __forceinline__ __nv_bfloat162 fma_rn<__nv_bfloat162>(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
    return __hfma2(a, b, c);  // Single instruction for 2 FMAs
}

// ============================================================================
// FROM_INT - Type-aware integer to float conversion
// ============================================================================
// Selects the optimal intrinsic based on type:
//   from_int<float>:  __int2float_rn
//   from_int<__half>: __int2half_rn
//   from_int<__nv_bfloat16>: __int2bfloat16_rn
//
// Usage: T result = from_int<T>(int_value);

template <typename T>
__device__ __forceinline__ T from_int(int v);

template <>
__device__ __forceinline__ float from_int<float>(int v) {
    return __int2float_rn(v);
}

template <>
__device__ __forceinline__ __half from_int<__half>(int v) {
    return __int2half_rn(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_int<__nv_bfloat16>(int v) {
    return __int2bfloat16_rn(v);
}

// FP8 E4M3 can represent 0-15 exactly - use LUT (faster than float conversion)
// E4M3: 1 sign, 4 exp (bias=7), 3 mantissa
// value = (1 + m/8) * 2^(e-7) for normalized numbers
template <>
__device__ __forceinline__ __nv_fp8_e4m3 from_int<__nv_fp8_e4m3>(int v) {
    static constexpr uint8_t lut[16] = {
        0x00, // 0
        0x38, // 1 = 1.0 * 2^0
        0x40, // 2 = 1.0 * 2^1
        0x44, // 3 = 1.5 * 2^1
        0x48, // 4 = 1.0 * 2^2
        0x4A, // 5 = 1.25 * 2^2
        0x4C, // 6 = 1.5 * 2^2
        0x4E, // 7 = 1.75 * 2^2
        0x50, // 8 = 1.0 * 2^3
        0x51, // 9 = 1.125 * 2^3
        0x52, // 10 = 1.25 * 2^3
        0x53, // 11 = 1.375 * 2^3
        0x54, // 12 = 1.5 * 2^3
        0x55, // 13 = 1.625 * 2^3
        0x56, // 14 = 1.75 * 2^3
        0x57, // 15 = 1.875 * 2^3
    };
    union { __nv_fp8_e4m3 f; uint8_t u; } result;
    result.u = lut[v & 0xF];
    return result.f;
}

// FP8 E5M2 has only 2 mantissa bits - can't represent 9,11,13,14,15 exactly
// Keep float conversion for correctness with rounding
template <>
__device__ __forceinline__ __nv_fp8_e5m2 from_int<__nv_fp8_e5m2>(int v) {
    return from_f32<__nv_fp8_e5m2>(__int2float_rn(v));
}

// ============================================================================
// ACC_ADD - Type-aware addition (returns a + b)
// ============================================================================
// Uses native hardware intrinsics for half/bfloat16.
// Usage: T result = acc_add<T>(a, b);

template <typename T>
__device__ __forceinline__ T acc_add(T a, T b);

template <>
__device__ __forceinline__ float acc_add<float>(float a, float b) {
    return a + b;
}

template <>
__device__ __forceinline__ __half acc_add<__half>(__half a, __half b) {
    return __hadd(a, b);
}

template <>
__device__ __forceinline__ __nv_bfloat16 acc_add<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hadd(a, b);
}

// Vector specializations - process 2 elements per instruction
template <>
__device__ __forceinline__ float2 acc_add<float2>(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

template <>
__device__ __forceinline__ __half2 acc_add<__half2>(__half2 a, __half2 b) {
    return __hadd2(a, b);  // Single instruction for 2 elements
}

template <>
__device__ __forceinline__ __nv_bfloat162 acc_add<__nv_bfloat162>(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __hadd2(a, b);  // Single instruction for 2 elements
}

// ============================================================================
// ACC_MUL - Type-aware multiplication (returns a * b)
// ============================================================================
// Uses native hardware intrinsics for half/bfloat16.
// Usage: T result = acc_mul<T>(a, b);

template <typename T>
__device__ __forceinline__ T acc_mul(T a, T b);

template <>
__device__ __forceinline__ float acc_mul<float>(float a, float b) {
    return a * b;
}

template <>
__device__ __forceinline__ __half acc_mul<__half>(__half a, __half b) {
    return __hmul(a, b);
}

template <>
__device__ __forceinline__ __nv_bfloat16 acc_mul<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hmul(a, b);
}

// Vector specializations - process 2 elements per instruction
template <>
__device__ __forceinline__ float2 acc_mul<float2>(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

template <>
__device__ __forceinline__ __half2 acc_mul<__half2>(__half2 a, __half2 b) {
    return __hmul2(a, b);  // Single instruction for 2 elements
}

template <>
__device__ __forceinline__ __nv_bfloat162 acc_mul<__nv_bfloat162>(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __hmul2(a, b);  // Single instruction for 2 elements
}

// ============================================================================
// TO_ACC - Convert any input type to accumulator type
// ============================================================================
// Converts y values (float, half, bf16, fp8) to acc_t for computation.
// Usage: acc_t val = to_acc<acc_t>(y_value);

// Primary template - goes through float (fallback)
template <typename acc_t, typename y_t>
__device__ __forceinline__ acc_t to_acc(y_t v) {
    if constexpr (std::is_same_v<acc_t, y_t>) {
        return v;  // Same type, no conversion
    } else {
        return from_f32<acc_t>(to_f32(v));  // Go through float
    }
}

// Specializations for optimal direct conversions (avoid float intermediate)

// acc_t=float: use direct to_f32 (single instruction for most types)
template <> __device__ __forceinline__ float to_acc<float, __half>(__half v) {
    return __half2float(v);
}
template <> __device__ __forceinline__ float to_acc<float, __nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
template <> __device__ __forceinline__ float to_acc<float, __nv_fp8_e4m3>(__nv_fp8_e4m3 v) {
    return to_f32(v);  // Uses hardware conversion on SM89+
}
template <> __device__ __forceinline__ float to_acc<float, __nv_fp8_e5m2>(__nv_fp8_e5m2 v) {
    return to_f32(v);
}

// acc_t=__half: use to_half for optimal FP8→half on SM89+
template <> __device__ __forceinline__ __half to_acc<__half, float>(float v) {
    return __float2half(v);
}
template <> __device__ __forceinline__ __half to_acc<__half, __nv_fp8_e4m3>(__nv_fp8_e4m3 v) {
    return to_half(v);  // Direct FP8→half on SM89+, else through float
}
template <> __device__ __forceinline__ __half to_acc<__half, __nv_fp8_e5m2>(__nv_fp8_e5m2 v) {
    return to_half(v);
}
// __half ← __nv_bfloat16: must go through float (no direct instruction)
template <> __device__ __forceinline__ __half to_acc<__half, __nv_bfloat16>(__nv_bfloat16 v) {
    return __float2half(__bfloat162float(v));
}

// acc_t=__nv_bfloat16: direct conversions where possible
template <> __device__ __forceinline__ __nv_bfloat16 to_acc<__nv_bfloat16, float>(float v) {
    return __float2bfloat16(v);
}
template <> __device__ __forceinline__ __nv_bfloat16 to_acc<__nv_bfloat16, __half>(__half v) {
    return __float2bfloat16(__half2float(v));  // Must go through float
}
template <> __device__ __forceinline__ __nv_bfloat16 to_acc<__nv_bfloat16, __nv_fp8_e4m3>(__nv_fp8_e4m3 v) {
    return __float2bfloat16(to_f32(v));
}
template <> __device__ __forceinline__ __nv_bfloat16 to_acc<__nv_bfloat16, __nv_fp8_e5m2>(__nv_fp8_e5m2 v) {
    return __float2bfloat16(to_f32(v));
}

// ============================================================================
// VECTOR SPECIALIZATIONS - Process 2 elements per instruction
// ============================================================================

// float2 ← half2: single instruction conversion
template <> __device__ __forceinline__ float2 to_acc<float2, __half2>(__half2 v) {
    return __half22float2(v);
}

// float2 ← __nv_bfloat162: single instruction conversion  
template <> __device__ __forceinline__ float2 to_acc<float2, __nv_bfloat162>(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
}

// half2 ← float2: single instruction conversion
template <> __device__ __forceinline__ __half2 to_acc<__half2, float2>(float2 v) {
    return __float22half2_rn(v);
}

// half2 ← __nv_bfloat162: must go through float2
template <> __device__ __forceinline__ __half2 to_acc<__half2, __nv_bfloat162>(__nv_bfloat162 v) {
    return __float22half2_rn(__bfloat1622float2(v));
}

// __nv_bfloat162 ← float2: single instruction conversion
template <> __device__ __forceinline__ __nv_bfloat162 to_acc<__nv_bfloat162, float2>(float2 v) {
    return __float22bfloat162_rn(v);
}

// __nv_bfloat162 ← half2: must go through float2
template <> __device__ __forceinline__ __nv_bfloat162 to_acc<__nv_bfloat162, __half2>(__half2 v) {
    return __float22bfloat162_rn(__half22float2(v));
}

// __nv_bfloat162 ← __nv_bfloat162: identity
template <> __device__ __forceinline__ __nv_bfloat162 to_acc<__nv_bfloat162, __nv_bfloat162>(__nv_bfloat162 v) {
    return v;
}
