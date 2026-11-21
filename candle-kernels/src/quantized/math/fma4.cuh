// The four-element fused multiply-add used by the mat-vec dot products,
// specialized by accumulator and weight type.
//
// `fma4` delegates to `fma4_impl` for float and `__half` accumulators, and to
// `fma4_half_impl` when the weights are `__half`. The bfloat16 accumulator uses
// the primary template, which is faster than a specialization.

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
