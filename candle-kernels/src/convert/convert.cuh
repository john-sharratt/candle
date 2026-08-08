#pragma once

// =============================================================================
// CONVERT INFRASTRUCTURE - COMMON HEADER
// =============================================================================
// Template-based format conversion for paged attention.
// Supports bidirectional conversion between:
//   - Float formats: F32, F16, BF16, F8E4M3
//   - Quantized formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
//
// Design:
//   - Each format has a specialization of BlockConverter<SrcBlock, DstType>
//   - BlockConverter::load() reads one block and converts to DstType elements
//   - BlockConverter::store() writes DstType elements to one block
//   - Runtime dispatch via load_block_convert() uses format tag from arena metadata
//
// File structure:
//   - convert.cuh (this file): Base template, scalar converters, type traits
//   - block_f32.cuh, block_f16.cuh, etc.: Per-format converter specializations
//   - convert_all.cuh: Unified header with runtime dispatch
// =============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <type_traits>

#include "../arena_table.cuh"
#include "../blocks.cuh"

// =============================================================================
// SCALAR CONVERSION HELPERS
// =============================================================================

// Reciprocal of the 255-step fine-scale quantum (Q4_KS/Q8_KS sub-block scales
// are stored as a uint8 fraction of the coarse `d`). A named compile-time
// constant so the decode reconstruction reads `sa * INV_255` rather than an
// inline `1.0f / 255.0f`; nvcc folds it to the same immediate.
static constexpr float INV_255 = 1.0f / 255.0f;

// Convert any supported type to float
template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <>
__device__ __forceinline__ float to_float<__nv_fp8_e4m3>(__nv_fp8_e4m3 v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __nv_fp8_storage_t storage = *reinterpret_cast<const __nv_fp8_storage_t*>(&v);
    return __half2float(__nv_cvt_fp8_to_halfraw(storage, __NV_E4M3));
#else
    // Software E4M3 → float conversion for SM80-SM88
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

// Convert float to any supported type
template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half_rn(v); }

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

template <>
__device__ __forceinline__ __nv_fp8_e4m3 from_float<__nv_fp8_e4m3>(float v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __nv_fp8_storage_t storage = __nv_cvt_halfraw_to_fp8(__float2half(v), __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 result;
    *reinterpret_cast<__nv_fp8_storage_t*>(&result) = storage;
    return result;
#else
    // Software float → E4M3 conversion for SM80-SM88
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

// =============================================================================
// ALIASES FOR COMPATIBILITY (to_f32/from_f32)
// =============================================================================
// Many kernels use to_f32/from_f32 naming convention

template <typename T>
__device__ __forceinline__ float to_f32(T v) { return to_float(v); }

template <typename T>
__device__ __forceinline__ T from_f32(float v) { return from_float<T>(v); }
// =============================================================================
// BLOCK CONVERTER BASE TEMPLATE
// =============================================================================

template <typename SrcBlock, typename DstType>
struct BlockConverter {
    static __device__ __forceinline__ int load(
        DstType* dst,
        const SrcBlock* src,
        int lane
    ) {
        static_assert(sizeof(SrcBlock) == 0, "BlockConverter not specialized for this type pair");
        return 0;
    }
    
    static __device__ __forceinline__ int store(
        SrcBlock* dst,
        const DstType* src,
        int lane
    ) {
        static_assert(sizeof(SrcBlock) == 0, "BlockConverter not specialized for this type pair");
        return 0;
    }
};

// =============================================================================
// INT8 READ-THROUGH EXTRACTOR (per-format raw int8 + per-block scale)
// =============================================================================
// Typed counterpart to BlockConverter for the V skip-dequant read-through
// (ArenaAccessor::load_head_int8_readthrough_typed, §1A): return a block element
// as a centered int8 plus the FP32 per-(dim,block) scale, with dequant == v*s.
// Specialized (in each block_*.cuh, next to BlockConverter) for every family
// whose dequant is expressible as int8 × one scale:
//   symmetric      Q8_0/Q4_0/Q5_0/Q2_0/Q3_0  + Q8_1 (sum unused) + Q2_S
//   sub-block      Q4_KS/Q8_KS                (scale = d·(sa|sb)/255)
//   sign           Q1_S (±scale) / Q1_A       (sign → ±scale_pos|neg)
//   centroid/anchor Q0 / Q0_M2 / Q0_M4 / Q0_X (v = the int8 codebook value)
// Deliberately NOT specialized — no single int×scale form, so a read-through on
// them is a compile error and the caller (gated by is_int8_readthrough_format)
// keeps them on the FP path:
//   asymmetric "+m" Q4_1/Q5_1/Q2_1/Q3_1/Q2_A  (need an offset term)
//   curve codebook  Q0_V                       (128-entry parametric lookup)
//   floating point  F16/BF16/F32/FP8/R16       (not a block integer)
struct Int8Sample {
    int8_t v;   // centered integer, guaranteed to fit int8
    float  s;   // per-(dim,block) scale; dequant == v * s
};

template <typename SrcBlock> struct BlockInt8 {
    static __device__ __forceinline__ Int8Sample load(const SrcBlock*, int) {
        static_assert(sizeof(SrcBlock) == 0,
                      "BlockInt8 not specialized: format is not an int8 passthrough family");
        return Int8Sample{0, 0.f};
    }
};

// =============================================================================
// TYPE TRAITS
// =============================================================================

// Map ArenaFormat to block type
template <int FORMAT> struct format_to_block;
template <> struct format_to_block<ArenaFormat::F32>     { using type = block_f32; };
template <> struct format_to_block<ArenaFormat::F16>     { using type = block_f16; };
template <> struct format_to_block<ArenaFormat::BF16>    { using type = block_bf16; };
template <> struct format_to_block<ArenaFormat::F8E4M3>  { using type = block_fp8_e4m3; };

template <int FORMAT>
using format_to_block_t = typename format_to_block<FORMAT>::type;

// Map block type to element type
template <typename Block> struct block_elem_type;
template <> struct block_elem_type<block_f32>       { using type = float; };
template <> struct block_elem_type<block_f16>       { using type = __half; };
template <> struct block_elem_type<block_bf16>      { using type = __nv_bfloat16; };
template <> struct block_elem_type<block_fp8_e4m3>  { using type = __nv_fp8_e4m3; };

template <typename Block>
using block_elem_t = typename block_elem_type<Block>::type;

// Map scalar type to ArenaFormat
template <typename T> struct type_to_format;
template <> struct type_to_format<float>           { static constexpr int value = ArenaFormat::F32; };
template <> struct type_to_format<__half>          { static constexpr int value = ArenaFormat::F16; };
template <> struct type_to_format<__nv_bfloat16>   { static constexpr int value = ArenaFormat::BF16; };
template <> struct type_to_format<__nv_fp8_e4m3>   { static constexpr int value = ArenaFormat::F8E4M3; };

template <typename T>
constexpr int type_to_format_v = type_to_format<T>::value;

// Map scalar type to its block type
template <typename T> struct type_to_block;
template <> struct type_to_block<float>           { using type = block_f32; };
template <> struct type_to_block<__half>          { using type = block_f16; };
template <> struct type_to_block<__nv_bfloat16>   { using type = block_bf16; };
template <> struct type_to_block<__nv_fp8_e4m3>   { using type = block_fp8_e4m3; };

template <typename T>
using type_to_block_t = typename type_to_block<T>::type;

// Check if a type is a dtype block
template <typename T> struct is_dtype_block : std::false_type {};
template <> struct is_dtype_block<block_f32>       : std::true_type {};
template <> struct is_dtype_block<block_f16>       : std::true_type {};
template <> struct is_dtype_block<block_bf16>      : std::true_type {};
template <> struct is_dtype_block<block_fp8_e4m3>  : std::true_type {};

template <typename T>
constexpr bool is_dtype_block_v = is_dtype_block<T>::value;

// Runtime helper
template <typename T>
__device__ __forceinline__ constexpr int type_to_arena_format() {
    if constexpr (std::is_same_v<T, float>) return ArenaFormat::F32;
    else if constexpr (std::is_same_v<T, __half>) return ArenaFormat::F16;
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) return ArenaFormat::BF16;
    else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) return ArenaFormat::F8E4M3;
    else return -1;
}

template <typename T>
__device__ __forceinline__ bool format_matches_type(int format) {
    return format == type_to_arena_format<T>();
}
