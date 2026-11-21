// The accumulator vector operations and the type aliases for the two-lane
// accumulator vector.
//
// `acc2_t` is the two-lane accumulator vector, and `lo` and `hi` extract its
// elements. `fma2_rn`, `mul2`, `add2` and `splat` operate on it, and
// `convert_dot_to_acc` and `accumulate` convert and sum values into the
// accumulator type.

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
