#pragma once

// =============================================================================
// GEMX DEQUANTIZATION TRAITS
// =============================================================================
//
// Abstract interface for dequantizing quantized weights to MMA-ready fragments.
// Each quant format implements this trait to provide:
//   1. Fragment types (FragB, FragS) for MMA consumption
//   2. Dequant function: packed int → FragB (fp16/bf16)
//   3. Scale application: frag_b = frag_b * scale (+ optional min correction)
//
// DESIGN NOTES:
// - FragB is always Vec<vec2_t, 2> where vec2_t = half2 or bfloat162
// - Dequant produces centered values (e.g., 4-bit: [0-15] → [-8, 7])
// - Scale application happens after dequant, before MMA
// - K-quants with min use apply_scale_with_min for affine dequant
//
// COMPUTE TYPES:
// - half:           Native fp16 tensor core path
// - __nv_bfloat16:  Native bf16 tensor core path  
// - __nv_fp8_e4m3:  FP8 tensor core path (SM89+)
//
// SCALE TYPES:
// - half:  Fast path, scales stored as fp16
// - float: Precision path, scales stored as fp32, converted for multiply
//
// =============================================================================

#include "../block_compact.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

// =============================================================================
// FRAGMENT TYPES (self-contained, mirrors gemx.cuh definitions)
// =============================================================================

template<typename T, int n>
struct GemxVec {
    T elems[n];
    __device__ __forceinline__ T& operator[](int i) { return elems[i]; }
    __device__ __forceinline__ const T& operator[](int i) const { return elems[i]; }
};

// Fragment types for MMA
template <typename compute_t>
struct GemxFragmentTypes;

template <>
struct GemxFragmentTypes<half> {
    using vec2_t = half2;
    using FragB = GemxVec<half2, 2>;    // 4 fp16 values = 2 × half2
    using FragS = GemxVec<half2, 1>;    // scale pair
};

template <>
struct GemxFragmentTypes<__nv_bfloat16> {
    using vec2_t = __nv_bfloat162;
    using FragB = GemxVec<__nv_bfloat162, 2>;  // 4 bf16 values = 2 × bfloat162
    using FragS = GemxVec<__nv_bfloat162, 1>;  // scale pair
};

template <>
struct GemxFragmentTypes<__nv_fp8_e4m3> {
    using vec2_t = __nv_fp8x2_e4m3;
    // FP8 compute uses f16 MMA operands (mma.f32.f16.f16.f32), so FragB is 2 × half2
    using FragB = GemxVec<uint32_t, 2>;  // 4 fp16 values as 2 × uint32 (for m16n8k16)
    using FragS = GemxVec<half2, 1>;     // scales stay in fp16 for precision
};

// =============================================================================
// BLOCK TYPE METADATA - Compute-type-independent traits
// =============================================================================
// Simple trait for block type metadata that doesn't depend on compute_t/scale_t.
// This allows checking has_min for any block type without instantiating the full
// dequant trait (which requires GemxFragmentTypes specialization).

template <typename BlockType>
struct block_type_traits {
    static constexpr bool has_min = false;  // Default: no min (symmetric quant)
};

// Q4_K: affine quantization with min
template <>
struct block_type_traits<block_c_q4_K> {
    static constexpr bool has_min = true;
};

// Q6_K: symmetric quantization (no min)
template <>
struct block_type_traits<block_c_q6_K> {
    static constexpr bool has_min = false;
};

// Q4_0: symmetric quantization (value = d * (q - 8), so has_min = true since min = -8*d)
template <>
struct block_type_traits<block_c_q4_0> {
    static constexpr bool has_min = true;
};

// Q4_1: affine quantization with explicit min (value = d * q + m)
template <>
struct block_type_traits<block_c_q4_1> {
    static constexpr bool has_min = true;
};

// Q5_0: symmetric quantization (value = d * (q - 16))
template <>
struct block_type_traits<block_c_q5_0> {
    static constexpr bool has_min = true;
};

// Q5_1: affine quantization with explicit min (value = d * q + m)
template <>
struct block_type_traits<block_c_q5_1> {
    static constexpr bool has_min = true;
};

// Q8_0: symmetric quantization (value = d * q, q is signed)
// Note: Q8_0 is truly symmetric (q already centered), so has_min = false
template <>
struct block_type_traits<block_c_q8_0> {
    static constexpr bool has_min = false;
};

// Q2_K: affine quantization with min
template <>
struct block_type_traits<block_c_q2_K> {
    static constexpr bool has_min = true;
};

// Q3_K: symmetric quantization (signed values)
template <>
struct block_type_traits<block_c_q3_K> {
    static constexpr bool has_min = false;
};

// Q5_K: affine quantization with min
template <>
struct block_type_traits<block_c_q5_K> {
    static constexpr bool has_min = true;
};

// ─────────────────────────────────────────────────────────────────────────────
// WHY THE INT8 WEIGHT FOLD IS PER-128, NOT PER-SUB (per-32 / per-16)
// ─────────────────────────────────────────────────────────────────────────────
// The live int8 KO fold applies ONE (scale, min) per 128-K per output row. It does
// this deliberately, for tensor-core throughput: the four k32 sub-MMAs of a 128-K tile
// accumulate into a SINGLE int32 before any scale is applied (see
// `grouped_matmul_impl_int8` in kernel.cuh). A per-sub (per-32 / per-16) weight scale
// would force a SEPARATE int32 accumulator per sub plus a per-sub float scale applied
// mid-fold — more accumulator registers, no cross-sub accumulation, and the int8 MMA's
// throughput advantage evaporates. So source formats with finer native scales (Q4_K
// per-32, Q6_K per-16) are RE-QUANTIZED to the coarser per-128 KO affine by `to_ko`; the
// step-up in bit width (`Int8Mode::Precision`) absorbs the granularity loss near-losslessly
// (measured: MXFP4→Q8_KO costs ~0.002 rel_l2 over an exact per-32 int8 — see
// candle-core `mxfp4_int8_matmul_matches_float_baseline`). Per-128 is the design, not a bug.
//
// HISTORICAL NOTE / TRAP REMOVED: three traits used to live here —
// `int8_scales_per_sub_trait` (per-16 "split 2-MMA fold"), `int8_affine_per16_trait`
// (Q2_K per-16 + an "all-ones MMA" for per-16 activation sums), and
// `int8_split_per32_trait`. Their comments described a per-sub fold as if it were live.
// **None of them ever had a consumer** — the split fold was never wired into any kernel.
// They were deleted (2026-08) because their descriptive comments repeatedly misled readers
// into thinking a per-32/per-16 fold existed. If you need a per-sub scale, you are ADDING
// a new fold, not enabling an existing one — and weigh it against the throughput cost above.

// De-interleaved scale storage: when true, the per-32 {scale,-min} dm values do NOT
// live in the weight block — they sit in a separate scale region at the tail of the
// weight tensor (one 16B dm block per quant block, same block index). The fold reads
// them straight from global by index instead of from the staged smem block (sub_dm).
// This keeps the cp.async weight stream pure quants (the MMA skips the float sectors).
template <typename BlockType>
struct is_scale_separate {
    static constexpr bool value = false;
};
template <>
struct is_scale_separate<block_c_q5_KO> {
    static constexpr bool value = true;
};
template <>
struct is_scale_separate<block_c_q4_KO> {
    static constexpr bool value = true;
};
template <>
struct is_scale_separate<block_c_q6_KO> {
    static constexpr bool value = true;
};
template <>
struct is_scale_separate<block_c_q8_KO> {
    static constexpr bool value = true;
};
template <>
struct is_scale_separate<block_c_mxfp4> {
    static constexpr bool value = true;
};
template <>
struct is_scale_separate<block_c_q2_KO> {
    static constexpr bool value = true;
};
// k1024 chunk blocks carry their scales inline (blk.dm) — the int8 fold reads them from
// the chunk, the only "scale-separate" (non-staged) path now used.
template <> struct is_scale_separate<block_c_q4_KO_k1024> { static constexpr bool value = true; };
template <> struct is_scale_separate<block_c_q5_KO_k1024> { static constexpr bool value = true; };
template <> struct is_scale_separate<block_c_q6_KO_k1024> { static constexpr bool value = true; };
template <> struct is_scale_separate<block_c_q8_KO_k1024> { static constexpr bool value = true; };
template <> struct is_scale_separate<block_c_q2_KO_k1024> { static constexpr bool value = true; };
template <> struct is_scale_separate<block_c_mxfp4_k1024> { static constexpr bool value = true; };

// =============================================================================
// K/64 SCALE HELPERS (embedded scales)
// =============================================================================

template <typename compute_t, typename FragB>
__device__ __forceinline__ void scale_frag_simple(FragB& frag_b, float scale) {
    if constexpr (std::is_same_v<compute_t, half>) {
        const half2 s2 = __half2half2(__float2half(scale));
        frag_b[0] = __hmul2(frag_b[0], s2);
        frag_b[1] = __hmul2(frag_b[1], s2);
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        const __nv_bfloat162 s2 = __bfloat162bfloat162(__float2bfloat16(scale));
        frag_b[0] = __hmul2(frag_b[0], s2);
        frag_b[1] = __hmul2(frag_b[1], s2);
    } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
        __nv_fp8_e4m3* p = reinterpret_cast<__nv_fp8_e4m3*>(&frag_b[0]);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            p[i] = __nv_fp8_e4m3(float(p[i]) * scale);
        }
    }
}

template <typename compute_t, typename FragB>
__device__ __forceinline__ void scale_frag_affine(FragB& frag_b, float scale, float bias) {
    if constexpr (std::is_same_v<compute_t, half>) {
        const half2 s2 = __half2half2(__float2half(scale));
        const half2 b2 = __half2half2(__float2half(bias));
        frag_b[0] = __hfma2(frag_b[0], s2, b2);
        frag_b[1] = __hfma2(frag_b[1], s2, b2);
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        const __nv_bfloat162 s2 = __bfloat162bfloat162(__float2bfloat16(scale));
        const __nv_bfloat162 b2 = __bfloat162bfloat162(__float2bfloat16(bias));
        frag_b[0] = __hfma2(frag_b[0], s2, b2);
        frag_b[1] = __hfma2(frag_b[1], s2, b2);
    } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
        __nv_fp8_e4m3* p = reinterpret_cast<__nv_fp8_e4m3*>(&frag_b[0]);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            p[i] = __nv_fp8_e4m3(float(p[i]) * scale + bias);
        }
    }
}

// =============================================================================
// GEMX DEQUANT TRAITS - Base template (must be specialized)
// =============================================================================
//
// Each quant format specializes this template to provide:
//   - FragB:              Fragment type for B operand (dequantized weights)
//   - FragS:              Fragment type for scales
//   - dequant(packed)     → FragB (centered, unscaled)
//   - apply_scale(frag_b, frag_s) → void (in-place scale application)
//   - has_min:            Whether format uses affine quantization (scale + min)
//   - scales_per_ktile:   Scales per 32-elem K-tile (from gemx_tile_traits)
//
// Usage in GEMX kernel:
//   int q = load_from_B_ptr();
//   auto frag_b = DequantTraits::dequant(q);
//   DequantTraits::apply_scale(frag_b, frag_s);
//   mma_sync(frag_c, frag_a, frag_b, frag_c);

template <typename BlockType, typename compute_t, typename scale_t>
struct gemx_dequant_traits {
    // Default: not implemented
    static constexpr bool implemented = false;
    static constexpr bool has_dequant_k64 = false;  // Set to true if dequant_k64_* methods exist
};

// =============================================================================
// LOP3 HELPER (from gemx.cuh)
// =============================================================================
// Ternary logic operation for efficient nibble extraction

#ifndef LOP3_TEMPLATE_DEFINED
#define LOP3_TEMPLATE_DEFINED
template <int lut>
__device__ __forceinline__ int lop3(int a, int b, int c) {
    int res;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
    );
    return res;
}
#endif

// =============================================================================
// COMPUTE TYPE TRAITS (magic constants for dequant)
// =============================================================================

template <typename T>
struct dequant_constants;

template <>
struct dequant_constants<half> {
    // FP16 dequant constants for signed 4-bit (center at 8)
    static constexpr int EX  = 0x64006400;  // Exponent bits for FP16 construction
    static constexpr int SUB = 0x64086408;  // Subtract 8 (zero-point)
    static constexpr int MUL = 0x2c002c00;  // Multiply factor for high nibble
    static constexpr int ADD = 0xd480d480;  // Add constant for high nibble FMA
    
    // Unsigned version (no centering)
    static constexpr uint32_t SUB_UNSIGNED = 0x64006400;
    static constexpr uint32_t ADD_UNSIGNED = 0xd400d400;
};

template <>
struct dequant_constants<__nv_bfloat16> {
    // BF16 dequant constants for signed 4-bit
    static constexpr int EX  = 0x43004300;  // BF16 exponent bits
    static constexpr int SUB = 0x43084308;  // Subtract 8 (zero-point) in BF16
    static constexpr int MUL = 0x3c003c00;  // BF16 multiply factor
    static constexpr int ADD = 0xc300c300;  // BF16 add constant
    
    // Unsigned version
    static constexpr uint32_t SUB_UNSIGNED = 0x43004300;
    static constexpr uint32_t ADD_UNSIGNED = 0xc280c280;
};

template <>
struct dequant_constants<__nv_fp8_e4m3> {
    // FP8 dequant: use FP16 constants (dequant to fp16 then convert)
    static constexpr int EX  = 0x64006400;  // Same as FP16
    static constexpr int SUB = 0x64086408;
    static constexpr int MUL = 0x2c002c00;
    static constexpr int ADD = 0xd480d480;
    
    // Unsigned version
    static constexpr uint32_t SUB_UNSIGNED = 0x64006400;
    static constexpr uint32_t ADD_UNSIGNED = 0xd400d400;
};
