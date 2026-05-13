#pragma once

// =============================================================================
// SCALE TYPE HELPERS
// =============================================================================
// Templates for handling either float2 (K-quants, full precision) or half2
// (simple quants) scale storage.
//
// K-quants benefit from float2 because they pre-compute d * scale_6bit,
// which loses precision when stored as half. Simple quants just copy the
// already-half d value, so half2 is lossless.
// =============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Helper to create a scale pair from two floats
// Specializations for float2 (K-quants) and half2 (simple quants)

template <typename ScaleT>
__device__ __forceinline__ ScaleT make_scale_pair(float s0, float s1);

template <>
__device__ __forceinline__ float2 make_scale_pair<float2>(float s0, float s1) {
    return make_float2(s0, s1);
}

template <>
__device__ __forceinline__ half2 make_scale_pair<half2>(float s0, float s1) {
    return __floats2half2_rn(s0, s1);
}

// Helper to extract x (scale) and y (min/second value) from a scale pair
// For float2: direct access
// For half2: convert to float first

template <typename ScaleT>
__device__ __forceinline__ float get_scale_x(ScaleT sm);

template <typename ScaleT>
__device__ __forceinline__ float get_scale_y(ScaleT sm);

// Passthrough for scalar float (identity)
template <>
__device__ __forceinline__ float get_scale_x<float>(float sm) {
    return sm;
}

template <>
__device__ __forceinline__ float get_scale_y<float>(float sm) {
    return sm;
}

// Passthrough for scalar half (convert to float)
template <>
__device__ __forceinline__ float get_scale_x<__half>(__half sm) {
    return __half2float(sm);
}

template <>
__device__ __forceinline__ float get_scale_y<__half>(__half sm) {
    return __half2float(sm);
}

template <>
__device__ __forceinline__ float get_scale_x<float2>(float2 sm) {
    return sm.x;
}

template <>
__device__ __forceinline__ float get_scale_y<float2>(float2 sm) {
    return sm.y;
}

template <>
__device__ __forceinline__ float get_scale_x<half2>(half2 sm) {
    return __low2float(sm);
}

template <>
__device__ __forceinline__ float get_scale_y<half2>(half2 sm) {
    return __high2float(sm);
}

// Helper to get x and y as half (for loaders that compute in half precision)
template <typename ScaleT>
__device__ __forceinline__ __half get_scale_x_half(ScaleT sm);

template <typename ScaleT>
__device__ __forceinline__ __half get_scale_y_half(ScaleT sm);

// Passthrough for scalar half (identity)
template <>
__device__ __forceinline__ __half get_scale_x_half<__half>(__half sm) {
    return sm;
}

template <>
__device__ __forceinline__ __half get_scale_y_half<__half>(__half sm) {
    return sm;
}

// Passthrough for scalar float (convert to half)
template <>
__device__ __forceinline__ __half get_scale_x_half<float>(float sm) {
    return __float2half(sm);
}

template <>
__device__ __forceinline__ __half get_scale_y_half<float>(float sm) {
    return __float2half(sm);
}

template <>
__device__ __forceinline__ __half get_scale_x_half<float2>(float2 sm) {
    return __float2half(sm.x);
}

template <>
__device__ __forceinline__ __half get_scale_y_half<float2>(float2 sm) {
    return __float2half(sm.y);
}

template <>
__device__ __forceinline__ __half get_scale_x_half<half2>(half2 sm) {
    return __low2half(sm);
}

template <>
__device__ __forceinline__ __half get_scale_y_half<half2>(half2 sm) {
    return __high2half(sm);
}

// ============================================================================
// GENERIC SCALE EXTRACTION - returns requested type directly
// ============================================================================
// get_scale_x_as<T>(sm) - extracts x component as type T
// get_scale_y_as<T>(sm) - extracts y component as type T
//
// Uses optimal path based on return type:
//   T=float → get_scale_x/y (no conversion for float2, __low2float for half2)
//   T=__half → get_scale_x/y_half (no conversion for half2, __float2half for float2)
//   T=other → get float and convert

template <typename RetT, typename ScaleT>
__device__ __forceinline__ RetT get_scale_x_as(ScaleT sm) {
    if constexpr (std::is_same_v<RetT, float>) {
        return get_scale_x(sm);
    } else if constexpr (std::is_same_v<RetT, __half>) {
        return get_scale_x_half(sm);
    } else {
        // Fallback: get float and convert
        return RetT(get_scale_x(sm));
    }
}

template <typename RetT, typename ScaleT>
__device__ __forceinline__ RetT get_scale_y_as(ScaleT sm) {
    if constexpr (std::is_same_v<RetT, float>) {
        return get_scale_y(sm);
    } else if constexpr (std::is_same_v<RetT, __half>) {
        return get_scale_y_half(sm);
    } else {
        // Fallback: get float and convert
        return RetT(get_scale_y(sm));
    }
}

// Type trait for K-quants vs simple quants scale type
// K-quants use float2 for precision, simple quants use half2
// NOTE: Block types are defined in common.cuh which must be included first
template <typename BlockT>
struct scale_type_for;

// =============================================================================
// HALF2 TO ACC2 CONVERSION
// =============================================================================
// Convert from half2 (storage format) to acc2_t<acc_t> (compute format).
// Uses direct intrinsic when possible to avoid split/recombine overhead.

template <typename AccT>
__device__ __forceinline__ AccT convert_half2_to_acc2(half2 h);

template <>
__device__ __forceinline__ float2 convert_half2_to_acc2<float2>(half2 h) {
    return __half22float2(h);
}

template <>
__device__ __forceinline__ half2 convert_half2_to_acc2<half2>(half2 h) {
    return h;  // No conversion needed
}

template <>
__device__ __forceinline__ __nv_bfloat162 convert_half2_to_acc2<__nv_bfloat162>(half2 h) {
    // half2 → float2 → bfloat162
    float2 f = __half22float2(h);
    return __floats2bfloat162_rn(f.x, f.y);
}

// =============================================================================
// ACC TYPE TRAIT FOR ACTIVATION TYPE
// =============================================================================
// Maps activation type to the appropriate accumulator/scale type:
//   float         → float
//   half          → half
//   __nv_bfloat16 → __nv_bfloat16
//   __nv_fp8_e4m3 → half (fp8 has no native FMA, use half)

template <typename act_t>
struct acc_type_for;

template <>
struct acc_type_for<float> {
    using type = float;
};

template <>
struct acc_type_for<half> {
    using type = half;
};

template <>
struct acc_type_for<__nv_bfloat16> {
    using type = __nv_bfloat16;
};

template <>
struct acc_type_for<__nv_fp8_e4m3> {
    using type = half;  // fp8 uses half for accumulation
};

// Convenience alias
template <typename act_t>
using acc_for_act_t = typename acc_type_for<act_t>::type;

// =============================================================================
// ACC2 TYPE TRAIT FOR ACTIVATION TYPE
// =============================================================================
// Maps activation type to the appropriate 2-element accumulator type:
//   float        → float2
//   half         → half2
//   __nv_bfloat16 → __nv_bfloat162
//   fp8          → half2 (fp8 has no native 2-element type)

template <typename act_t>
struct acc2_type_for;

template <>
struct acc2_type_for<float> {
    using type = float2;
};

template <>
struct acc2_type_for<half> {
    using type = half2;
};

template <>
struct acc2_type_for<__nv_bfloat16> {
    using type = __nv_bfloat162;
};

template <>
struct acc2_type_for<__nv_fp8_e4m3> {
    using type = half2;  // fp8 uses half2 for accumulation
};

// Convenience alias
template <typename act_t>
using acc2_for_act_t = typename acc2_type_for<act_t>::type;

// =============================================================================
// LO/HI ACCESSORS FOR ALL ACC2 TYPES
// =============================================================================

__device__ __forceinline__ float lo_acc2(float2 v) { return v.x; }
__device__ __forceinline__ float hi_acc2(float2 v) { return v.y; }

__device__ __forceinline__ half lo_acc2(half2 v) { return __low2half(v); }
__device__ __forceinline__ half hi_acc2(half2 v) { return __high2half(v); }

__device__ __forceinline__ __nv_bfloat16 lo_acc2(__nv_bfloat162 v) { return __low2bfloat16(v); }
__device__ __forceinline__ __nv_bfloat16 hi_acc2(__nv_bfloat162 v) { return __high2bfloat16(v); }

