#pragma once

#include "reduce.cuh"
#include <array>
#include <utility>

// =============================================================================
// MATH AND TYPE CONVERSION UTILITIES FOR QUANTIZED KERNELS
// =============================================================================
// This file provides type conversion and arithmetic primitives used throughout
// the quantized matrix-vector multiplication kernels.
//
// Key functions:
//   - Type conversions: to_f32, from_f32, to_float
//   - Arithmetic: bf16_mul, bf16_fma
//   - Accumulators: convert_dot_to_acc, accumulate, acc_to_float
//   - Initialization: zero_val
//   - FP8 LUT: fp8_e4m3_to_half_lut for fast SM80-88 FP8 conversion
//
// Reductions (warp_reduce_sum_t, warp_reduce_max, bf16_add) are in reduce.cuh
//
// Supports types: float, __half, __nv_bfloat16, __nv_fp8_e4m3, __nv_fp8_e5m2
// Also provides vector variants: float2, float4, __half2, __nv_bfloat162, etc.
//
// =============================================================================

// =============================================================================
// FP8 E4M3 LOOKUP TABLE FOR FAST SOFTWARE CONVERSION (SM80-88)
// =============================================================================
// Precomputed half values for all 256 possible FP8 E4M3 bit patterns.
// Using a LUT reduces ~15 instructions per conversion to a single load.
// 
// E4M3 format: 1 sign, 4 exponent (bias=7), 3 mantissa
// Range: ±448, special: NaN at exp=15
//
// On SM89+ (Ada/Hopper), use hardware __nv_cvt_fp8x2_to_halfraw2 instead.
// =============================================================================

// Host-side initialization function (called once at module load)
// The LUT is stored as uint16_t (raw half bits) to avoid constructor issues
namespace fp8_lut {

// Compile-time computation of FP8 E4M3 → half conversion
constexpr uint16_t fp8_e4m3_to_half_bits(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp = (bits >> 3) & 0xF;
    uint32_t mant = bits & 0x7;
    
    if (exp == 0) {
        if (mant == 0) {
            // Zero (positive or negative)
            return sign ? 0x8000 : 0x0000;
        }
        // Denormal: value = mant/8 * 2^(-6) = mant * 2^(-9)
        // Half denormal: value = mant/1024 * 2^(-14) = mant * 2^(-24)
        // We need to convert E4M3 denormal to half normal or denormal
        // E4M3 denormal value = mant * 2^(-9)
        // To represent in half: need exp such that (1 + m/1024) * 2^(exp-15) = mant * 2^(-9)
        // For mant=1: 2^(-9) = 0.001953125, half can represent this as 2^(-9) with exp=6
        // Actually let's just compute via float at compile time
        // mant=1: 2^-9 = 0.001953125, half bits: exp=6 (2^(-9)), mant=0 → 0x1800
        // Simpler: use the formula directly
        // E4M3 denormal: val = mant/8 * 2^-6 = mant * 2^-9
        // Half normal: val = (1 + m/1024) * 2^(e-15)
        // To match: e-15 = -9 → e=6, and we need m such that 1+m/1024 = mant
        // But mant is 1-7, and 1+m/1024 starts at 1.0, so we can't match directly
        // Instead store as half with appropriate exponent
        // For mant * 2^-9, shift mant to be the "mantissa" of a half at exp=6:
        // half value = mant * 512 at exp=6 would give mant * 2^9 * 2^(6-15) = mant * 2^0 = mant
        // We want mant * 2^-9, so at exp=6: (mant/1024) * 2^(6-15) = mant * 2^-19 (wrong)
        // Let me just compute the correct half bits:
        // mant=1: 2^-9, half: exp=6, mant=0 → 0x1800 is 2^(6-15)=2^-9 ✓
        // mant=2: 2*2^-9=2^-8, half: exp=7, mant=0 → 0x1C00 is 2^-8 ✓
        // mant=3: 3*2^-9, half: exp=6, mant=512 → 0x1A00 is 1.5*2^-9 ✓
        // This is getting complex, just return approximate for denormals
        // For simplicity in constexpr, compute normalized form
        int shift = 2 - ((mant >> 2) ? 0 : (mant >> 1) ? 1 : 2);
        uint32_t half_exp = 6 - shift;
        uint32_t half_mant = (mant << (10 - 3 + shift)) & 0x3FF;
        return (sign << 15) | (half_exp << 10) | half_mant;
    } else if (exp == 15) {
        // NaN (E4M3 has no infinity)
        return 0x7E00;  // quiet NaN
    } else {
        // Normal: value = (1 + mant/8) * 2^(exp-7)
        // Half: value = (1 + m/1024) * 2^(e-15)
        // Match: e-15 = exp-7 → e = exp+8
        // And: mant/8 = m/1024 → m = mant * 128
        uint32_t half_exp = exp + 8;
        uint32_t half_mant = mant << 7;  // mant * 128
        return (sign << 15) | (half_exp << 10) | half_mant;
    }
}

// Compile-time array generator
template<size_t... Is>
constexpr auto make_lut(std::index_sequence<Is...>) {
    return std::array<uint16_t, sizeof...(Is)>{{fp8_e4m3_to_half_bits(Is)...}};
}

// The actual LUT (computed at compile time)
inline constexpr auto fp8_e4m3_lut_data = make_lut(std::make_index_sequence<256>{});

// =============================================================================
// FP8 E5M2 LOOKUP TABLE (5 exponent bits, 2 mantissa bits)
// =============================================================================
// E5M2 format: 1 sign, 5 exponent (bias=15), 2 mantissa
// Range: ±57344, special: Inf at exp=31 mant=0, NaN at exp=31 mant!=0
// E5M2 has larger range but less precision than E4M3.

// Compile-time computation of FP8 E5M2 → half conversion
constexpr uint16_t fp8_e5m2_to_half_bits(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp = (bits >> 2) & 0x1F;  // 5 bits
    uint32_t mant = bits & 0x3;         // 2 bits
    
    if (exp == 0) {
        if (mant == 0) {
            // Zero (positive or negative)
            return sign ? 0x8000 : 0x0000;
        }
        // Denormal: value = mant/4 * 2^(-14) = mant * 2^(-16)
        // Half can represent 2^(-16) as denormal (exp=0, mant=64)
        // mant=1: 2^-16, half denormal: exp=0, mant=64 → 0x0040
        // mant=2: 2^-15, half denormal: exp=0, mant=128 → 0x0080
        // mant=3: 3*2^-16, half denormal: exp=0, mant=192 → 0x00C0
        uint32_t half_mant = mant << 6;  // mant * 64
        return (sign << 15) | half_mant;
    } else if (exp == 31) {
        if (mant == 0) {
            // Infinity
            return (sign << 15) | 0x7C00;
        } else {
            // NaN
            return 0x7E00;  // quiet NaN
        }
    } else {
        // Normal: value = (1 + mant/4) * 2^(exp-15)
        // Half: value = (1 + m/1024) * 2^(e-15)
        // Match: e = exp (same bias!)
        // And: mant/4 = m/1024 → m = mant * 256
        uint32_t half_exp = exp;  // Same bias
        uint32_t half_mant = mant << 8;  // mant * 256
        return (sign << 15) | (half_exp << 10) | half_mant;
    }
}

// Compile-time array generator for E5M2
template<size_t... Is>
constexpr auto make_lut_e5m2(std::index_sequence<Is...>) {
    return std::array<uint16_t, sizeof...(Is)>{{fp8_e5m2_to_half_bits(Is)...}};
}

// The actual E5M2 LUT (computed at compile time)
inline constexpr auto fp8_e5m2_lut_data = make_lut_e5m2(std::make_index_sequence<256>{});

}  // namespace fp8_lut

// Device-accessible constant memory LUTs
// Initialized from fp8_lut::*_lut_data at kernel load time
__device__ __constant__ uint16_t fp8_e4m3_to_half_lut[256];
__device__ __constant__ uint16_t fp8_e5m2_to_half_lut[256];

// Flag to track if LUTs have been initialized (host-side)
inline bool fp8_lut_initialized = false;

// Initialize the FP8 LUTs (call once before any FP8 kernel)
inline void init_fp8_lut() {
    if (!fp8_lut_initialized) {
        cudaMemcpyToSymbol(fp8_e4m3_to_half_lut, fp8_lut::fp8_e4m3_lut_data.data(), 
                          256 * sizeof(uint16_t));
        cudaMemcpyToSymbol(fp8_e5m2_to_half_lut, fp8_lut::fp8_e5m2_lut_data.data(), 
                          256 * sizeof(uint16_t));
        fp8_lut_initialized = true;
    }
}

// Device function to convert FP8 E4M3 byte to half using LUT
__device__ __forceinline__ half fp8_e4m3_lut_to_half(uint8_t bits) {
    uint16_t h_bits = fp8_e4m3_to_half_lut[bits];
    return *reinterpret_cast<half*>(&h_bits);
}

// Device function to convert FP8 E4M3 byte to float using LUT
__device__ __forceinline__ float fp8_e4m3_to_float_lut(uint8_t bits) {
    return __half2float(fp8_e4m3_lut_to_half(bits));
}

// Device function to convert FP8 E5M2 byte to half using LUT
__device__ __forceinline__ half fp8_e5m2_lut_to_half(uint8_t bits) {
    uint16_t h_bits = fp8_e5m2_to_half_lut[bits];
    return *reinterpret_cast<half*>(&h_bits);
}

// Device function to convert FP8 E5M2 byte to float using LUT
__device__ __forceinline__ float fp8_e5m2_to_float_lut(uint8_t bits) {
    return __half2float(fp8_e5m2_lut_to_half(bits));
}

// Convert 4 FP8 E4M3 values (packed in uint32_t) to 2 half2 using LUT
__device__ __forceinline__ void fp8x4_to_half2x2_lut(uint32_t fp8x4, half2& h0, half2& h1) {
    uint16_t h0_lo = fp8_e4m3_to_half_lut[fp8x4 & 0xFF];
    uint16_t h0_hi = fp8_e4m3_to_half_lut[(fp8x4 >> 8) & 0xFF];
    uint16_t h1_lo = fp8_e4m3_to_half_lut[(fp8x4 >> 16) & 0xFF];
    uint16_t h1_hi = fp8_e4m3_to_half_lut[fp8x4 >> 24];
    
    uint32_t h0_bits = h0_lo | (static_cast<uint32_t>(h0_hi) << 16);
    uint32_t h1_bits = h1_lo | (static_cast<uint32_t>(h1_hi) << 16);
    
    h0 = *reinterpret_cast<half2*>(&h0_bits);
    h1 = *reinterpret_cast<half2*>(&h1_bits);
}

// Convert 4 FP8 E5M2 values (packed in uint32_t) to 2 half2 using LUT
__device__ __forceinline__ void fp8x4_to_half2x2_lut_e5m2(uint32_t fp8x4, half2& h0, half2& h1) {
    uint16_t h0_lo = fp8_e5m2_to_half_lut[fp8x4 & 0xFF];
    uint16_t h0_hi = fp8_e5m2_to_half_lut[(fp8x4 >> 8) & 0xFF];
    uint16_t h1_lo = fp8_e5m2_to_half_lut[(fp8x4 >> 16) & 0xFF];
    uint16_t h1_hi = fp8_e5m2_to_half_lut[fp8x4 >> 24];
    
    uint32_t h0_bits = h0_lo | (static_cast<uint32_t>(h0_hi) << 16);
    uint32_t h1_bits = h1_lo | (static_cast<uint32_t>(h1_hi) << 16);
    
    h0 = *reinterpret_cast<half2*>(&h0_bits);
    h1 = *reinterpret_cast<half2*>(&h1_bits);
}

// =============================================================================
// PRMT (BYTE PERMUTE) HELPER FUNCTIONS
// =============================================================================
// PRMT is a powerful PTX instruction that performs byte-level permutation
// on two 32-bit inputs. Each byte of the output is selected from one of 8
// possible source bytes (4 from each input).
//
// Selector nibble: 0-3 = bytes from 'a', 4-7 = bytes from 'b'
// Mode bits in selector byte [7:6]: 0=copy, 1=zero-fill if sign, 2=replicate sign
//
// Examples:
//   prmt(a, b, 0x7610) → [b[3], b[2], a[1], a[0]] (interleave)
//   prmt(a, 0, 0x1010) → [a[1], a[0], a[1], a[0]] (replicate low half)
//   prmt(a, b, 0x0040) → [a[0], 0, a[0], 0] (scatter bytes)
// =============================================================================

// Generic PRMT: output[i] = src_byte[selector_nibble_i]
// where src_byte[0..3] = bytes of 'a', src_byte[4..7] = bytes of 'b'
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t selector) {
    uint32_t result;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(selector));
    return result;
}

// Build half2 pair: [a_byte[i], a_byte[j]] where each byte goes to a 16-bit slot
// Used for constructing (lo, hi) pairs for LOP3 dequantization
// Input: a = [b3, b2, b1, b0] (4 bytes)
// Output: [a_byte[hi_idx] << 16 | a_byte[lo_idx]]
// prmt_pair_from_bytes(a, 0, 2) → [a[2], a[0]] = [(hi>>16)&0xFF | ((lo>>16)&0xFF)<<16]
__device__ __forceinline__ uint32_t prmt_pair_from_bytes(uint32_t a, int lo_idx, int hi_idx) {
    // selector: byte 0 = lo_idx, byte 2 = hi_idx (bytes 1,3 are zero-filled via 0xF nibble)
    uint32_t selector = lo_idx | (0xF << 4) | (hi_idx << 8) | (0xF << 12);
    return prmt(a, 0, selector);
}

// Replicate half to half2: input = uint32_t containing half in low 16 bits
// Output: same half replicated to both slots
// This uses prmt to copy bytes [1,0] to [3,2,1,0]
__device__ __forceinline__ uint32_t prmt_replicate_lo_half(uint32_t h) {
    // h = [?, ?, h1, h0], output = [h1, h0, h1, h0]
    return prmt(h, h, 0x1010);
}

// Replicate high half to both slots
__device__ __forceinline__ uint32_t prmt_replicate_hi_half(uint32_t h) {
    // h = [h3, h2, ?, ?], output = [h3, h2, h3, h2]
    return prmt(h, h, 0x3232);
}

// Extract byte and place in low position of 16-bit slot (for LUT lookup)
// byte_idx: 0-3, specifies which byte to extract
__device__ __forceinline__ uint32_t prmt_extract_byte(uint32_t a, int byte_idx) {
    // Place byte in position 0, zero the rest
    return prmt(a, 0, byte_idx);
}

// =============================================================================
// PRMT PAIR CONSTRUCTION FOR Q4_K/Q6_K LOP3 DEQUANTIZATION
// =============================================================================
// These functions build (lo_byte, hi_byte) pairs needed for half2/bf162 LOP3 
// weight conversion. Given lo=[b3,b2,b1,b0] and hi=[B3,B2,B1,B0], we need:
//   pair0 = [B0, 0, b0, 0]  (bytes 0)
//   pair1 = [B1, 0, b1, 0]  (bytes 1)  
//   pair2 = [B2, 0, b2, 0]  (bytes 2)
//   pair3 = [B3, 0, b3, 0]  (bytes 3)
//
// PRMT selector for pair_i: we want lo_byte[i] at position 0, hi_byte[i] at position 2
// So selector = i | (0xF << 4) | ((4+i) << 8) | (0xF << 12)
// =============================================================================

// Build pair from corresponding bytes of lo and hi
// pair_idx: 0-3, output = [hi_byte[pair_idx], 0, lo_byte[pair_idx], 0]
__device__ __forceinline__ uint32_t prmt_build_lop3_pair(uint32_t lo, uint32_t hi, int pair_idx) {
    // lo bytes are at positions 0-3, hi bytes are at positions 4-7
    // We want: byte0=lo[pair_idx], byte1=0, byte2=hi[pair_idx], byte3=0
    uint32_t selector = pair_idx | (0xF << 4) | ((4 + pair_idx) << 8) | (0xF << 12);
    return prmt(lo, hi, selector);
}

// Compile-time constant versions for better optimization
__device__ __forceinline__ uint32_t prmt_build_lop3_pair_0(uint32_t lo, uint32_t hi) {
    return prmt(lo, hi, 0xF4F0);  // [hi[0], 0, lo[0], 0]
}
__device__ __forceinline__ uint32_t prmt_build_lop3_pair_1(uint32_t lo, uint32_t hi) {
    return prmt(lo, hi, 0xF5F1);  // [hi[1], 0, lo[1], 0]
}
__device__ __forceinline__ uint32_t prmt_build_lop3_pair_2(uint32_t lo, uint32_t hi) {
    return prmt(lo, hi, 0xF6F2);  // [hi[2], 0, lo[2], 0]
}
__device__ __forceinline__ uint32_t prmt_build_lop3_pair_3(uint32_t lo, uint32_t hi) {
    return prmt(lo, hi, 0xF7F3);  // [hi[3], 0, lo[3], 0]
}

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

// ============================================================================
// ACC2_T TYPE TRAIT - Maps scalar acc_t to its vector2 type
// ============================================================================
// Usage: using acc2 = acc2_t<acc_t>;  // float -> float2, __half -> half2

template <typename T> struct acc_vec2;
template <> struct acc_vec2<float> { using type = float2; };
template <> struct acc_vec2<__half> { using type = __half2; };
template <> struct acc_vec2<__nv_bfloat16> { using type = __nv_bfloat162; };

// Convenience alias
template <typename T>
using acc2_t = typename acc_vec2<T>::type;

// ============================================================================
// LO / HI - Extract low/high element from vector2 types
// ============================================================================
// Generic element extraction that works for float2, half2, bfloat162
// Usage: acc_t x = lo(vec2);  acc_t y = hi(vec2);

__device__ __forceinline__ float lo(float2 v) { return v.x; }
__device__ __forceinline__ float hi(float2 v) { return v.y; }

__device__ __forceinline__ __half lo(__half2 v) { return __low2half(v); }
__device__ __forceinline__ __half hi(__half2 v) { return __high2half(v); }

__device__ __forceinline__ __nv_bfloat16 lo(__nv_bfloat162 v) { return __low2bfloat16(v); }
__device__ __forceinline__ __nv_bfloat16 hi(__nv_bfloat162 v) { return __high2bfloat16(v); }

// ============================================================================
// MAKE_ACC2 - Construct acc2_t<T> from two scalars
// ============================================================================
// Creates a packed vector2 from two scalar values.
// Usage: acc2_t<acc_t> v = make_acc2<acc_t>(x, y);

template <typename T>
__device__ __forceinline__ acc2_t<T> make_acc2(T x, T y);

template <>
__device__ __forceinline__ float2 make_acc2<float>(float x, float y) {
    return make_float2(x, y);
}

template <>
__device__ __forceinline__ __half2 make_acc2<__half>(__half x, __half y) {
    return __halves2half2(x, y);
}

template <>
__device__ __forceinline__ __nv_bfloat162 make_acc2<__nv_bfloat16>(__nv_bfloat16 x, __nv_bfloat16 y) {
    return __halves2bfloat162(x, y);
}

// ============================================================================
// SPLAT - Broadcast scalar to both lanes of acc2_t
// ============================================================================
// Usage: acc2_t<acc_t> v2 = splat<acc_t>(scalar);

template <typename T>
__device__ __forceinline__ acc2_t<T> splat(T x);

template <>
__device__ __forceinline__ float2 splat<float>(float x) {
    return make_float2(x, x);
}

template <>
__device__ __forceinline__ __half2 splat<__half>(__half x) {
    return __half2half2(x);
}

template <>
__device__ __forceinline__ __nv_bfloat162 splat<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162bfloat162(x);
}

// ============================================================================
// FROM_INT2 - Convert two ints to acc2_t
// ============================================================================
// Usage: acc2_t<acc_t> v = from_int2<acc_t>(i0, i1);

template <typename T>
__device__ __forceinline__ acc2_t<T> from_int2(int x, int y);

template <>
__device__ __forceinline__ float2 from_int2<float>(int x, int y) {
    return make_float2(float(x), float(y));
}

template <>
__device__ __forceinline__ __half2 from_int2<__half>(int x, int y) {
    return __floats2half2_rn(float(x), float(y));
}

template <>
__device__ __forceinline__ __nv_bfloat162 from_int2<__nv_bfloat16>(int x, int y) {
    return __floats2bfloat162_rn(float(x), float(y));
}

// ============================================================================
// FMA2_RN - Vectorized fused multiply-add: a*b + c
// ============================================================================
// For half2: uses __hfma2 (2 FMAs in 1 instruction)
// For float2: two scalar FMAs
// Usage: acc2_t<acc_t> r = fma2_rn<acc_t>(a, b, c);

template <typename T>
__device__ __forceinline__ acc2_t<T> fma2_rn(acc2_t<T> a, acc2_t<T> b, acc2_t<T> c);

template <>
__device__ __forceinline__ float2 fma2_rn<float>(float2 a, float2 b, float2 c) {
    return make_float2(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y));
}

template <>
__device__ __forceinline__ __half2 fma2_rn<__half>(__half2 a, __half2 b, __half2 c) {
    return __hfma2(a, b, c);
}

template <>
__device__ __forceinline__ __nv_bfloat162 fma2_rn<__nv_bfloat16>(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
    return __hfma2(a, b, c);
}

// ============================================================================
// MUL2 - Vectorized multiply: a*b
// ============================================================================
// For half2: uses __hmul2 (2 muls in 1 instruction)
// Usage: acc2_t<acc_t> r = mul2<acc_t>(a, b);

template <typename T>
__device__ __forceinline__ acc2_t<T> mul2(acc2_t<T> a, acc2_t<T> b);

template <>
__device__ __forceinline__ float2 mul2<float>(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

template <>
__device__ __forceinline__ __half2 mul2<__half>(__half2 a, __half2 b) {
    return __hmul2(a, b);
}

template <>
__device__ __forceinline__ __nv_bfloat162 mul2<__nv_bfloat16>(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __hmul2(a, b);
}

// ============================================================================
// ADD2 - Vectorized add: a+b
// ============================================================================
// For half2: uses __hadd2 (2 adds in 1 instruction)
// Usage: acc2_t<acc_t> r = add2<acc_t>(a, b);

template <typename T>
__device__ __forceinline__ acc2_t<T> add2(acc2_t<T> a, acc2_t<T> b);

template <>
__device__ __forceinline__ float2 add2<float>(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

template <>
__device__ __forceinline__ __half2 add2<__half>(__half2 a, __half2 b) {
    return __hadd2(a, b);
}

template <>
__device__ __forceinline__ __nv_bfloat162 add2<__nv_bfloat16>(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __hadd2(a, b);
}

// ============================================================================
// TO_ACC_LO / TO_ACC_HI - Extract lo/hi element from half2 to acc_t directly
// ============================================================================
// Avoids double conversion: to_acc<float>(__low2half(h2)) does two ops,
// but to_acc_lo<float>(h2) uses __low2float which is a single instruction.
//
// Usage: acc_t y0 = to_acc_lo<acc_t>(y01_h2);
//        acc_t y1 = to_acc_hi<acc_t>(y01_h2);

// Primary templates (through scalar)
template <typename acc_t>
__device__ __forceinline__ acc_t to_acc_lo(__half2 v) {
    return to_acc<acc_t>(__low2half(v));
}

template <typename acc_t>
__device__ __forceinline__ acc_t to_acc_hi(__half2 v) {
    return to_acc<acc_t>(__high2half(v));
}

// Specializations: acc_t=float - use direct __low2float/__high2float
template <>
__device__ __forceinline__ float to_acc_lo<float>(__half2 v) {
    return __low2float(v);
}

template <>
__device__ __forceinline__ float to_acc_hi<float>(__half2 v) {
    return __high2float(v);
}

// Specializations: acc_t=__half - use direct __low2half/__high2half (identity)
template <>
__device__ __forceinline__ __half to_acc_lo<__half>(__half2 v) {
    return __low2half(v);
}

template <>
__device__ __forceinline__ __half to_acc_hi<__half>(__half2 v) {
    return __high2half(v);
}

// Specializations: acc_t=__nv_bfloat16 - must go through float
template <>
__device__ __forceinline__ __nv_bfloat16 to_acc_lo<__nv_bfloat16>(__half2 v) {
    return __float2bfloat16(__low2float(v));
}

template <>
__device__ __forceinline__ __nv_bfloat16 to_acc_hi<__nv_bfloat16>(__half2 v) {
    return __float2bfloat16(__high2float(v));
}

// ============================================================================
// TO_ACC_LO / TO_ACC_HI - Extract from __nv_bfloat162
// ============================================================================

template <typename acc_t>
__device__ __forceinline__ acc_t to_acc_lo(__nv_bfloat162 v) {
    return to_acc<acc_t>(__low2bfloat16(v));
}

template <typename acc_t>
__device__ __forceinline__ acc_t to_acc_hi(__nv_bfloat162 v) {
    return to_acc<acc_t>(__high2bfloat16(v));
}

// Specializations: acc_t=float - use __low2float equivalent
template <>
__device__ __forceinline__ float to_acc_lo<float>(__nv_bfloat162 v) {
    return __bfloat162float(__low2bfloat16(v));
}

template <>
__device__ __forceinline__ float to_acc_hi<float>(__nv_bfloat162 v) {
    return __bfloat162float(__high2bfloat16(v));
}

// Specializations: acc_t=__nv_bfloat16 - identity
template <>
__device__ __forceinline__ __nv_bfloat16 to_acc_lo<__nv_bfloat16>(__nv_bfloat162 v) {
    return __low2bfloat16(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 to_acc_hi<__nv_bfloat16>(__nv_bfloat162 v) {
    return __high2bfloat16(v);
}

// ============================================================================
// BF16 ARITHMETIC HELPERS (bf16_add is in reduce.cuh)
// ============================================================================

/// BF16 multiplication with native hardware support on SM80+
static __device__ __forceinline__ __nv_bfloat16 bf16_mul(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hmul(a, b);
}

/// BF16 fused multiply-add
static __device__ __forceinline__ __nv_bfloat16 bf16_fma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
    return __hfma(a, b, c);
}

// ============================================================================
// ZERO VALUE HELPER
// ============================================================================
// Returns the zero value for any supported type. Useful for initializing
// accumulators and registers in templated code.

template <typename T>
__device__ __forceinline__ T zero_val();

// Scalar types
template <>
__device__ __forceinline__ float zero_val<float>() {
    return 0.0f;
}

template <>
__device__ __forceinline__ double zero_val<double>() {
    return 0.0;
}

template <>
__device__ __forceinline__ int zero_val<int>() {
    return 0;
}

template <>
__device__ __forceinline__ uint32_t zero_val<uint32_t>() {
    return 0u;
}

// Half precision (FP16)
template <>
__device__ __forceinline__ __half zero_val<__half>() {
    return __float2half_rn(0.0f);
}

template <>
__device__ __forceinline__ __half2 zero_val<__half2>() {
    return __float2half2_rn(0.0f);
}

// BFloat16
template <>
__device__ __forceinline__ __nv_bfloat16 zero_val<__nv_bfloat16>() {
    return __float2bfloat16_rn(0.0f);
}

template <>
__device__ __forceinline__ __nv_bfloat162 zero_val<__nv_bfloat162>() {
    return __float2bfloat162_rn(0.0f);
}

// FP8 types
template <>
__device__ __forceinline__ __nv_fp8_e4m3 zero_val<__nv_fp8_e4m3>() {
    union { __nv_fp8_e4m3 f; uint8_t u; } z;
    z.u = 0;
    return z.f;
}

template <>
__device__ __forceinline__ __nv_fp8_e5m2 zero_val<__nv_fp8_e5m2>() {
    union { __nv_fp8_e5m2 f; uint8_t u; } z;
    z.u = 0;
    return z.f;
}

// Vector types (float)
template <>
__device__ __forceinline__ float2 zero_val<float2>() {
    return make_float2(0.0f, 0.0f);
}

template <>
__device__ __forceinline__ float4 zero_val<float4>() {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// Vector types (int)
template <>
__device__ __forceinline__ int2 zero_val<int2>() {
    return make_int2(0, 0);
}

template <>
__device__ __forceinline__ int4 zero_val<int4>() {
    return make_int4(0, 0, 0, 0);
}

template <>
__device__ __forceinline__ uint2 zero_val<uint2>() {
    return make_uint2(0u, 0u);
}

template <>
__device__ __forceinline__ uint4 zero_val<uint4>() {
    return make_uint4(0u, 0u, 0u, 0u);
}

// ============================================================================
// ACCUMULATOR CONVERSION UTILITIES
// ============================================================================

/// Convert float dot product result to accumulator type
/// For float compute: identity (pass through)
/// For BF16 compute: converts float dot product to BF16 for reduced-precision accumulation
/// For FP8 compute: keeps float (FP8 can't accumulate directly)
template <typename acc_t>
static __device__ __forceinline__ acc_t convert_dot_to_acc(float dot_result) {
    if constexpr (std::is_same_v<acc_t, float>) {
        return dot_result;
    } else if constexpr (std::is_same_v<acc_t, __nv_bfloat16>) {
        return from_f32<__nv_bfloat16>(dot_result);
    } else if constexpr (std::is_same_v<acc_t, __half>) {
        return from_f32<__half>(dot_result);
    } else {
        // Fallback for unknown types
        return acc_t(dot_result);
    }
}

/// Accumulate value into accumulator with type-appropriate addition
/// Uses native hardware intrinsics for half/bfloat16, avoiding slow operator+= 
template <typename acc_t>
static __device__ __forceinline__ void accumulate(acc_t& acc, acc_t val) {
    if constexpr (std::is_same_v<acc_t, __half>) {
        acc = __hadd(acc, val);  // Native half add (SM53+)
    } else if constexpr (std::is_same_v<acc_t, __nv_bfloat16>) {
        acc = bf16_add(acc, val);  // Native bf16 add (SM80+)
    } else {
        acc += val;
    }
}

/// Convert accumulator to float for output and cross-warp reduction
/// Handles all supported accumulator types
template <typename acc_t>
static __device__ __forceinline__ float acc_to_float(acc_t val) {
    return to_f32(val);
}

/// Zero value for accumulator type - uses zero_val helper
template <typename acc_t>
static __device__ __forceinline__ acc_t acc_zero() {
    return zero_val<acc_t>();
}

// ============================================================================
// FMA4 - Type-aware 4-element fused multiply-add
// ============================================================================
// Computes: w0*y0 + w1*y1 + w2*y2 + w3*y3 with optimal precision based on
// accumulator type (acc_t) and input type (y_t).
//
// Usage: acc_t result = fma4<acc_t>(w0, w1, w2, w3, y0, y1, y2, y3);
//
// Template params:
//   acc_t - accumulator type (float or __half), specified explicitly
//   y_t   - input type, deduced from arguments
//
// Combinations:
//   fma4<float, float>:  float FMA chain, no conversions
//   fma4<float, __half>: convert y to float, float FMA chain
//   fma4<__half, float>: convert w and y to half, half FMA chain
//   fma4<__half, __half>: convert w to half, y stays half (optimal)
//
// Implementation uses partial template specialization via traits class pattern
// (C++ doesn't allow partial function template specialization).
// ============================================================================

// Primary template (fallback)
template <typename acc_t, typename y_t>
struct fma4_impl {
    __device__ __forceinline__ static acc_t call(
        float w0, float w1, float w2, float w3,
        y_t y0, y_t y1, y_t y2, y_t y3
    ) {
        // Fallback: serial FMA chain with cast
        const float fy0 = to_f32(y0), fy1 = to_f32(y1);
        const float fy2 = to_f32(y2), fy3 = to_f32(y3);
        const float result = __fmaf_rn(w0, fy0,
                             __fmaf_rn(w1, fy1,
                             __fmaf_rn(w2, fy2, w3 * fy3)));
        return static_cast<acc_t>(result);
    }
};

// Partial specialization: float accumulator, any y_t
template <typename y_t>
struct fma4_impl<float, y_t> {
    __device__ __forceinline__ static float call(
        float w0, float w1, float w2, float w3,
        y_t y0, y_t y1, y_t y2, y_t y3
    ) {
        // Convert y to float if needed
        const float fy0 = to_f32(y0);
        const float fy1 = to_f32(y1);
        const float fy2 = to_f32(y2);
        const float fy3 = to_f32(y3);
        // Serial FMA chain: 4 ops (1 MUL + 3 FMA)
        // GPU warps hide latency, so fewer instructions wins over lower dependency
        return __fmaf_rn(w0, fy0,
               __fmaf_rn(w1, fy1,
               __fmaf_rn(w2, fy2, w3 * fy3)));
    }
};

// Partial specialization: __half accumulator, float weights, any y_t
// Uses to_half() for optimal type-specific conversion (no-op for __half,
// direct FP8→half on SM89+, etc.)
template <typename y_t>
struct fma4_impl<__half, y_t> {
    __device__ __forceinline__ static __half call(
        float w0, float w1, float w2, float w3,
        y_t y0, y_t y1, y_t y2, y_t y3
    ) {
        // Convert weights to half
        const __half hw0 = __float2half(w0);
        const __half hw1 = __float2half(w1);
        const __half hw2 = __float2half(w2);
        const __half hw3 = __float2half(w3);
        // Convert y to half using optimized to_half<y_t>
        const __half hy0 = to_half(y0);
        const __half hy1 = to_half(y1);
        const __half hy2 = to_half(y2);
        const __half hy3 = to_half(y3);
        // Serial FMA chain: 4 ops (1 MUL + 3 FMA)
        return __hfma(hw0, hy0,
               __hfma(hw1, hy1,
               __hfma(hw2, hy2, __hmul(hw3, hy3))));
    }
};

// NOTE: No __nv_bfloat16 specialization - the primary template fallback is faster!
// It accumulates in float (fast FMA) and casts only at the end.
// BF16 scalar FMA is slower than float FMA on current GPUs.

// User-facing function: delegates to specialized impl (float weights)
template <typename acc_t, typename y_t>
__device__ __forceinline__ acc_t fma4(
    float w0, float w1, float w2, float w3,
    y_t y0, y_t y1, y_t y2, y_t y3
) {
    return fma4_impl<acc_t, y_t>::call(w0, w1, w2, w3, y0, y1, y2, y3);
}

// ============================================================================
// FMA4 with __half weights - optimized paths when weights are already half
// ============================================================================

// Primary template (fallback) for half weights
template <typename acc_t, typename y_t>
struct fma4_half_impl {
    __device__ __forceinline__ static acc_t call(
        __half w0, __half w1, __half w2, __half w3,
        y_t y0, y_t y1, y_t y2, y_t y3
    ) {
        // Fallback: convert to float, serial FMA chain
        const float fw0 = __half2float(w0), fw1 = __half2float(w1);
        const float fw2 = __half2float(w2), fw3 = __half2float(w3);
        const float fy0 = to_f32(y0), fy1 = to_f32(y1);
        const float fy2 = to_f32(y2), fy3 = to_f32(y3);
        const float result = __fmaf_rn(fw0, fy0,
                             __fmaf_rn(fw1, fy1,
                             __fmaf_rn(fw2, fy2, fw3 * fy3)));
        return static_cast<acc_t>(result);
    }
};

// Partial specialization: float accumulator, __half weights, any y_t
template <typename y_t>
struct fma4_half_impl<float, y_t> {
    __device__ __forceinline__ static float call(
        __half w0, __half w1, __half w2, __half w3,
        y_t y0, y_t y1, y_t y2, y_t y3
    ) {
        // Convert weights and y to float, serial FMA chain
        const float fw0 = __half2float(w0), fw1 = __half2float(w1);
        const float fw2 = __half2float(w2), fw3 = __half2float(w3);
        const float fy0 = to_f32(y0), fy1 = to_f32(y1);
        const float fy2 = to_f32(y2), fy3 = to_f32(y3);
        return __fmaf_rn(fw0, fy0,
               __fmaf_rn(fw1, fy1,
               __fmaf_rn(fw2, fy2, fw3 * fy3)));
    }
};

// Partial specialization: __half accumulator, __half weights, any y_t
// Uses to_half() for optimal type-specific conversion
template <typename y_t>
struct fma4_half_impl<__half, y_t> {
    __device__ __forceinline__ static __half call(
        __half w0, __half w1, __half w2, __half w3,
        y_t y0, y_t y1, y_t y2, y_t y3
    ) {
        // Weights already half, convert y using optimized to_half<y_t>
        const __half hy0 = to_half(y0);
        const __half hy1 = to_half(y1);
        const __half hy2 = to_half(y2);
        const __half hy3 = to_half(y3);
        // Serial FMA chain: 4 ops (1 MUL + 3 FMA)
        return __hfma(w0, hy0,
               __hfma(w1, hy1,
               __hfma(w2, hy2, __hmul(w3, hy3))));
    }
};

// User-facing function: delegates to specialized impl (__half weights)
template <typename acc_t, typename y_t>
__device__ __forceinline__ acc_t fma4(
    __half w0, __half w1, __half w2, __half w3,
    y_t y0, y_t y1, y_t y2, y_t y3
) {
    return fma4_half_impl<acc_t, y_t>::call(w0, w1, w2, w3, y0, y1, y2, y3);
}


// ============================================================================
// MMA (MATRIX MULTIPLY-ACCUMULATE) SUPPORT FOR SM80+
// ============================================================================

/// Pack two 16-bit values into uint32 for MMA operand
template <typename T>
__device__ __forceinline__ uint32_t __pack_half2(T a, T b);

template <>
__device__ __forceinline__ uint32_t __pack_half2<__half>(__half a, __half b) {
    uint32_t result;
    asm("mov.b32 %0, {%1,%2};" : "=r"(result)
        : "h"(__half_as_ushort(a)),
          "h"(__half_as_ushort(b)));
    return result;
}

template <>
__device__ __forceinline__ uint32_t __pack_half2<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) {
    uint32_t result;
    asm("mov.b32 %0, {%1,%2};" : "=r"(result)
        : "h"(__bfloat16_as_ushort(a)),
          "h"(__bfloat16_as_ushort(b)));
    return result;
}

// Float pack - convert to half for MMA
template <>
__device__ __forceinline__ uint32_t __pack_half2<float>(float a, float b) {
    __half ha = __float2half_rn(a);
    __half hb = __float2half_rn(b);
    uint32_t result;
    asm("mov.b32 %0, {%1,%2};" : "=r"(result)
        : "h"(__half_as_ushort(ha)),
          "h"(__half_as_ushort(hb)));
    return result;
}

// FP8 pack - for m16n8k16 MMA we convert to half first (m16n8k32 uses native FP8)
template <>
__device__ __forceinline__ uint32_t __pack_half2<__nv_fp8_e4m3>(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __nv_fp8_storage_t sa = *reinterpret_cast<const __nv_fp8_storage_t*>(&a);
    __nv_fp8_storage_t sb = *reinterpret_cast<const __nv_fp8_storage_t*>(&b);
    __half ha = __nv_cvt_fp8_to_halfraw(sa, __NV_E4M3);
    __half hb = __nv_cvt_fp8_to_halfraw(sb, __NV_E4M3);
    uint32_t result;
    asm("mov.b32 %0, {%1,%2};" : "=r"(result)
        : "h"(*reinterpret_cast<const unsigned short*>(&ha)),
          "h"(*reinterpret_cast<const unsigned short*>(&hb)));
    return result;
#else
    // Fallback: convert to float, then pack as float
    return __pack_half2<float>((float)a, (float)b);
#endif
}

/// m16n8k16 MMA with float32 accumulation (SM80+)
/// A operand: 16×16 (4 uint32 registers, each holding 4 F16 values)
/// B operand: 16×8  (2 uint32 registers, each holding 4 F16 values)
/// C output:  16×8  (4 float registers, accumulated)
template <typename T>
__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
        );
    } else {
        // Default to F16 MMA for half or converted FP8
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
        );
    }
}

/// m16n8k32 MMA with FP8 E4M3 inputs and float32 accumulation (SM89+)
/// A operand: 16×32 (4 uint32 registers, each holding 4 FP8 values)
/// B operand: 32×8  (2 uint32 registers, each holding 4 FP8 values)
/// C output:  16×8  (4 float registers, accumulated) - SAME as F16!
///
/// Note: Processes K=32 per instruction (2× throughput vs F16's K=16)
__device__ __forceinline__ void mma_sync_m16n8k32_row_col_f32_fp8(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
#else
    // Pre-SM89: no FP8 MMA support
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1;
#endif
}

/// Pack 4 consecutive FP8 values from memory into uint32 for MMA operand
/// Simply reinterprets 4 contiguous FP8 bytes as a uint32
__device__ __forceinline__ uint32_t pack_fp8x4(const __nv_fp8_e4m3* ptr) {
    return *reinterpret_cast<const uint32_t*>(ptr);
}

/// Convert 4 F16 values (as 2×half2) → packed FP8x4 uint32
/// Used for converting dequantized weights to FP8 for m16n8k32 MMA
__device__ __forceinline__ uint32_t cvt_f16x4_to_fp8x4(__half2 h01, __half2 h23) {
    __half h0 = __low2half(h01);
    __half h1 = __high2half(h01);
    __half h2 = __low2half(h23);
    __half h3 = __high2half(h23);
    uint32_t result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
    bytes[0] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h0), __NV_SATFINITE, __NV_E4M3);
    bytes[1] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h1), __NV_SATFINITE, __NV_E4M3);
    bytes[2] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h2), __NV_SATFINITE, __NV_E4M3);
    bytes[3] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h3), __NV_SATFINITE, __NV_E4M3);
#else
    // Fallback: use bit-casting for SM80-SM88 (limited FP8 support)
    // Pack 4 __half values using bitwise operations
    uint16_t* shorts = reinterpret_cast<uint16_t*>(&result);
    shorts[0] = *reinterpret_cast<uint16_t*>(&h0);
    shorts[1] = *reinterpret_cast<uint16_t*>(&h1);
#endif
    return result;
}

// ============================================================================
// UNPACK_FP8X4 - Convert 4 packed FP8 bytes to 4 floats
// ============================================================================
// CRITICAL: Must use __nv_cvt_fp8_to_halfraw to interpret raw bits as FP8,
// NOT __nv_fp8_e4m3(storage_t) which treats the value as a number to convert.
__device__ __forceinline__ void unpack_fp8x4(uint32_t packed, float& f0, float& f1, float& f2, float& f3) {
    f0 = __half2float(__nv_cvt_fp8_to_halfraw(packed & 0xFF, __NV_E4M3));
    f1 = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 8) & 0xFF, __NV_E4M3));
    f2 = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 16) & 0xFF, __NV_E4M3));
    f3 = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 24) & 0xFF, __NV_E4M3));
}

