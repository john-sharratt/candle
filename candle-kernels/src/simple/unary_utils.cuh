#pragma once
// =============================================================================
// UNARY ELEMENTWISE UTILITIES - Optimized primitives for unary operations
// =============================================================================
// Key optimizations:
//   1. Vectorized memory access (float4 for fp32, half2 for fp16)
//   2. Fast math intrinsics (__expf, __logf, __tanhf, __frcp_rn)
//   3. half2 paired operations for 2x fp16 throughput
//   4. Multiple elements per thread with ILP
//   5. __ldg() for read-only input access
//   6. __restrict__ qualifiers for pointer aliasing hints
//   7. 32-bit index arithmetic
//   8. Separate contiguous vs strided kernel paths
//   9. #pragma unroll for inner loops
// =============================================================================

#include "cuda_utils.cuh"
#include "../fast_exp.cuh"
#include <stdint.h>

// =============================================================================
// FAST MATH INTRINSICS - Uses fast_exp library for exp-based operations
// =============================================================================

__device__ __forceinline__ float fast_expf(float x) { return fast_exp::exp<float>(x); }
__device__ __forceinline__ float fast_logf(float x) { return __logf(x); }  // Fast log intrinsic
__device__ __forceinline__ float fast_tanhf(float x) {
    // Fast tanh approximation: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    // For better precision, use: 1 - 2 / (exp(2x) + 1)
    float e2x = fast_exp::exp<float>(2.0f * x);
    return __fdividef(e2x - 1.0f, e2x + 1.0f);
}
__device__ __forceinline__ float fast_rcp(float x) { return __frcp_rn(x); }
__device__ __forceinline__ float fast_rsqrt(float x) { return rsqrtf(x); }

// =============================================================================
// OPTIMIZED ACTIVATION FUNCTIONS - FLOAT (using fast_exp library)
// =============================================================================

__device__ __forceinline__ float fast_silu_f32(float x) {
    return fast_exp::silu<float>(x);
}

__device__ __forceinline__ float fast_sigmoid_f32(float x) {
    return fast_exp::sigmoid<float>(x);
}

__device__ __forceinline__ float fast_gelu_f32(float x) {
    return fast_exp::gelu<float>(x);
}

__device__ __forceinline__ float fast_gelu_erf_f32(float x) {
    return x * normcdff(x);
}

__device__ __forceinline__ float fast_relu_f32(float x) {
    return fmaxf(x, 0.0f);
}

__device__ __forceinline__ float fast_elu_f32(float x, float alpha) {
    return (x > 0.0f) ? x : alpha * (fast_exp::exp<float>(x) - 1.0f);
}

// =============================================================================
// HALF2 OPTIMIZED FUNCTIONS - Process 2 fp16 values at once
// =============================================================================

// Half2 fast exponential - uses fast_exp library
__device__ __forceinline__ __half2 fast_exp_h2(__half2 x) {
    return fast_exp::exp2<__half>(x);
}

// Half2 SiLU: x * sigmoid(x) - uses fast_exp library vec2 activations
__device__ __forceinline__ __half2 fast_silu_h2(__half2 x) {
    return fast_exp::silu2<__half>(x);
}

// Half2 sigmoid - uses fast_exp library vec2 activations
__device__ __forceinline__ __half2 fast_sigmoid_h2(__half2 x) {
    return fast_exp::sigmoid2<__half>(x);
}

// Half2 GELU (fast approximation) - uses fast_exp library vec2 activations
__device__ __forceinline__ __half2 fast_gelu_h2(__half2 x) {
    return fast_exp::gelu2<__half>(x);
}

// Half2 GELU erf version
__device__ __forceinline__ __half2 fast_gelu_erf_h2(__half2 x) {
    float2 fx;
    fx.x = __half2float(__low2half(x));
    fx.y = __half2float(__high2half(x));
    
    float2 result;
    result.x = fast_gelu_erf_f32(fx.x);
    result.y = fast_gelu_erf_f32(fx.y);
    
    return __floats2half2_rn(result.x, result.y);
}

// Half2 ReLU
__device__ __forceinline__ __half2 fast_relu_h2(__half2 x) {
    __half2 zero = __float2half2_rn(0.0f);
    return __hmax2(x, zero);
}

// Half2 ELU - uses fast_exp library
__device__ __forceinline__ __half2 fast_elu_h2(__half2 x, __half2 alpha) {
    float2 fx, fa;
    fx.x = __half2float(__low2half(x));
    fx.y = __half2float(__high2half(x));
    fa.x = __half2float(__low2half(alpha));
    fa.y = __half2float(__high2half(alpha));
    
    float2 result;
    result.x = (fx.x > 0.0f) ? fx.x : fa.x * (fast_exp::exp<float>(fx.x) - 1.0f);
    result.y = (fx.y > 0.0f) ? fx.y : fa.y * (fast_exp::exp<float>(fx.y) - 1.0f);
    
    return __floats2half2_rn(result.x, result.y);
}

// Half2 tanh
__device__ __forceinline__ __half2 fast_tanh_h2(__half2 x) {
    float2 fx;
    fx.x = __half2float(__low2half(x));
    fx.y = __half2float(__high2half(x));
    
    float2 result;
    result.x = fast_tanhf(fx.x);
    result.y = fast_tanhf(fx.y);
    
    return __floats2half2_rn(result.x, result.y);
}

// Half2 exp - uses fast_exp library
__device__ __forceinline__ __half2 h2_exp(__half2 x) {
    return fast_exp::exp2<__half>(x);
}

// Half2 log
__device__ __forceinline__ __half2 h2_log(__half2 x) {
    return h2log(x);
}

// Half2 sqrt
__device__ __forceinline__ __half2 h2_sqrt(__half2 x) {
    return h2sqrt(x);
}

// Half2 reciprocal
__device__ __forceinline__ __half2 h2_rcp(__half2 x) {
    return h2rcp(x);
}

// Half2 sin
__device__ __forceinline__ __half2 h2_sin(__half2 x) {
    return h2sin(x);
}

// Half2 cos
__device__ __forceinline__ __half2 h2_cos(__half2 x) {
    return h2cos(x);
}

// Half2 negation
__device__ __forceinline__ __half2 h2_neg(__half2 x) {
    return __hneg2(x);
}

// Half2 abs
__device__ __forceinline__ __half2 h2_abs(__half2 x) {
    return __habs2(x);
}

// Half2 square
__device__ __forceinline__ __half2 h2_sqr(__half2 x) {
    return __hmul2(x, x);
}

// Half2 erf (via float conversion)
__device__ __forceinline__ __half2 h2_erf(__half2 x) {
    float2 fx;
    fx.x = __half2float(__low2half(x));
    fx.y = __half2float(__high2half(x));
    return __floats2half2_rn(erff(fx.x), erff(fx.y));
}

// Half2 ceil
__device__ __forceinline__ __half2 h2_ceil(__half2 x) {
    return h2ceil(x);
}

// Half2 floor
__device__ __forceinline__ __half2 h2_floor(__half2 x) {
    return h2floor(x);
}

// Half2 round
__device__ __forceinline__ __half2 h2_round(__half2 x) {
    return h2rint(x);
}

// Half2 normcdf (via float conversion)
__device__ __forceinline__ __half2 h2_normcdf(__half2 x) {
    float2 fx;
    fx.x = __half2float(__low2half(x));
    fx.y = __half2float(__high2half(x));
    return __floats2half2_rn(normcdff(fx.x), normcdff(fx.y));
}

// Half2 sign
__device__ __forceinline__ __half2 h2_sign(__half2 x) {
    float2 fx;
    fx.x = __half2float(__low2half(x));
    fx.y = __half2float(__high2half(x));
    
    float2 result;
    result.x = (fx.x > 0.0f ? 1.0f : 0.0f) - (fx.x < 0.0f ? 1.0f : 0.0f);
    result.y = (fx.y > 0.0f ? 1.0f : 0.0f) - (fx.y < 0.0f ? 1.0f : 0.0f);
    
    return __floats2half2_rn(result.x, result.y);
}

// =============================================================================
// VECTORIZED LOAD/STORE HELPERS
// =============================================================================

// Load 4 floats using __ldg
__device__ __forceinline__ float4 ldg_load_float4(const float* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const float4*>(ptr));
}

// Store 4 floats
__device__ __forceinline__ void store_float4(float* __restrict__ ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

// Load 2 doubles using __ldg
__device__ __forceinline__ double2 ldg_load_double2(const double* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const double2*>(ptr));
}

// Store 2 doubles
__device__ __forceinline__ void store_double2(double* __restrict__ ptr, double2 val) {
    *reinterpret_cast<double2*>(ptr) = val;
}

// Load half2 using __ldg
__device__ __forceinline__ __half2 ldg_load_half2(const __half* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const __half2*>(ptr));
}

// Store half2
__device__ __forceinline__ void store_half2(__half* __restrict__ ptr, __half2 val) {
    *reinterpret_cast<__half2*>(ptr) = val;
}

// Load bfloat16x2 using __ldg
__device__ __forceinline__ __nv_bfloat162 ldg_load_bf162(const __nv_bfloat16* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const __nv_bfloat162*>(ptr));
}

// Store bfloat16x2
__device__ __forceinline__ void store_bf162(__nv_bfloat16* __restrict__ ptr, __nv_bfloat162 val) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = val;
}

// =============================================================================
// OPTIMIZED UNARY KERNEL MACRO - FLOAT32 WITH FLOAT4 VECTORIZATION
// =============================================================================
// Processes 4 elements per thread for better memory bandwidth utilization
// Uses __ldg for read-only access and __restrict__ for aliasing hints
// NOTE: float4 requires 16-byte alignment - we check both src and out pointers

#define UNARY_OP_F32_VEC4(FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* __restrict__ info, \
    const float* __restrict__ inp, \
    float* __restrict__ out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        const float* __restrict__ src = inp ? inp : out; \
        /* Check 16-byte alignment for float4 vectorization */ \
        bool aligned = ((uintptr_t)src % 16 == 0) && ((uintptr_t)out % 16 == 0); \
        \
        if (aligned && numel >= 4) { \
            /* Vectorized path: process 4 elements per thread */ \
            const size_t vec_numel = numel / 4; \
            const size_t vec_offset = vec_numel * 4; \
            \
            /* Main vectorized loop */ \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < vec_numel; i += blockDim.x * gridDim.x) { \
                float4 v = ldg_load_float4(src + i * 4); \
                float4 r; \
                r.x = FUNC(v.x); \
                r.y = FUNC(v.y); \
                r.z = FUNC(v.z); \
                r.w = FUNC(v.w); \
                store_float4(out + i * 4, r); \
            } \
            \
            /* Handle remaining elements */ \
            for (size_t i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                float x = __ldg(src + i); \
                out[i] = FUNC(x); \
            } \
        } else { \
            /* Scalar fallback for unaligned data */ \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                float x = __ldg(src + i); \
                out[i] = FUNC(x); \
            } \
        } \
    } else { \
        /* Strided path: scalar processing */ \
        const float* __restrict__ src = inp; \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            size_t strided_i = get_strided_index(i, num_dims, dims, strides); \
            float x = src ? __ldg(src + strided_i) : out[i]; \
            out[i] = FUNC(x); \
        } \
    } \
}

// =============================================================================
// OPTIMIZED UNARY KERNEL MACRO - FLOAT32 WITH PARAM AND FLOAT4 VECTORIZATION
// =============================================================================
// NOTE: float4 requires 16-byte alignment - we check both src and out pointers

#define UNARY_OP1_F32_VEC4(FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* __restrict__ info, \
    const float param, \
    const float* __restrict__ inp, \
    float* __restrict__ out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        const float* __restrict__ src = inp ? inp : out; \
        /* Check 16-byte alignment for float4 vectorization */ \
        bool aligned = ((uintptr_t)src % 16 == 0) && ((uintptr_t)out % 16 == 0); \
        \
        if (aligned && numel >= 4) { \
            const size_t vec_numel = numel / 4; \
            const size_t vec_offset = vec_numel * 4; \
            \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < vec_numel; i += blockDim.x * gridDim.x) { \
                float4 v = ldg_load_float4(src + i * 4); \
                float4 r; \
                r.x = FUNC(v.x, param); \
                r.y = FUNC(v.y, param); \
                r.z = FUNC(v.z, param); \
                r.w = FUNC(v.w, param); \
                store_float4(out + i * 4, r); \
            } \
            \
            for (size_t i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                float x = __ldg(src + i); \
                out[i] = FUNC(x, param); \
            } \
        } else { \
            /* Scalar fallback for unaligned data */ \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                float x = __ldg(src + i); \
                out[i] = FUNC(x, param); \
            } \
        } \
    } else { \
        const float* __restrict__ src = inp; \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            size_t strided_i = get_strided_index(i, num_dims, dims, strides); \
            float x = src ? __ldg(src + strided_i) : out[i]; \
            out[i] = FUNC(x, param); \
        } \
    } \
}

// =============================================================================
// OPTIMIZED UNARY KERNEL MACRO - HALF WITH HALF2 VECTORIZATION
// =============================================================================
// NOTE: half2 requires 4-byte alignment - we check both src and out pointers

#define UNARY_OP_F16_VEC2(FN_NAME, FUNC_H2, FUNC_SCALAR) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* __restrict__ info, \
    const __half* __restrict__ inp, \
    __half* __restrict__ out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        const __half* __restrict__ src = inp ? inp : out; \
        /* Check 4-byte alignment for half2 vectorization */ \
        bool aligned = ((uintptr_t)src % 4 == 0) && ((uintptr_t)out % 4 == 0); \
        \
        if (aligned && numel >= 2) { \
            const size_t vec_numel = numel / 2; \
            const size_t vec_offset = vec_numel * 2; \
            \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < vec_numel; i += blockDim.x * gridDim.x) { \
                __half2 v = ldg_load_half2(src + i * 2); \
                __half2 r = FUNC_H2(v); \
                store_half2(out + i * 2, r); \
            } \
            \
            /* Handle odd element */ \
            if (vec_offset < numel && blockIdx.x == 0 && threadIdx.x == 0) { \
                __half x = __ldg(src + vec_offset); \
                out[vec_offset] = FUNC_SCALAR(x); \
            } \
        } else { \
            /* Scalar fallback for unaligned data */ \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                __half x = src ? __ldg(src + i) : out[i]; \
                out[i] = FUNC_SCALAR(x); \
            } \
        } \
    } else { \
        const __half* __restrict__ src = inp; \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            size_t strided_i = get_strided_index(i, num_dims, dims, strides); \
            __half x = src ? src[strided_i] : out[i]; \
            out[i] = FUNC_SCALAR(x); \
        } \
    } \
}

// Half with param
// NOTE: half2 requires 4-byte alignment - we check both src and out pointers
#define UNARY_OP1_F16_VEC2(FN_NAME, FUNC_H2, FUNC_SCALAR) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* __restrict__ info, \
    const __half param, \
    const __half* __restrict__ inp, \
    __half* __restrict__ out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    __half2 param2 = __half2half2(param); \
    \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        const __half* __restrict__ src = inp ? inp : out; \
        /* Check 4-byte alignment for half2 vectorization */ \
        bool aligned = ((uintptr_t)src % 4 == 0) && ((uintptr_t)out % 4 == 0); \
        \
        if (aligned && numel >= 2) { \
            const size_t vec_numel = numel / 2; \
            const size_t vec_offset = vec_numel * 2; \
            \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < vec_numel; i += blockDim.x * gridDim.x) { \
                __half2 v = ldg_load_half2(src + i * 2); \
                __half2 r = FUNC_H2(v, param2); \
                store_half2(out + i * 2, r); \
            } \
            \
            if (vec_offset < numel && blockIdx.x == 0 && threadIdx.x == 0) { \
                __half x = __ldg(src + vec_offset); \
                out[vec_offset] = FUNC_SCALAR(x, param); \
            } \
        } else { \
            /* Scalar fallback for unaligned data */ \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                __half x = src ? __ldg(src + i) : out[i]; \
                out[i] = FUNC_SCALAR(x, param); \
            } \
        } \
    } else { \
        const __half* __restrict__ src = inp; \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            size_t strided_i = get_strided_index(i, num_dims, dims, strides); \
            __half x = src ? src[strided_i] : out[i]; \
            out[i] = FUNC_SCALAR(x, param); \
        } \
    } \
}

// =============================================================================
// SIMPLE UNARY OPERATIONS (identity transformations for scalar)
// =============================================================================

__device__ __forceinline__ float id_f32(float x) { return x; }
__device__ __forceinline__ float neg_f32(float x) { return -x; }
__device__ __forceinline__ float sqr_f32(float x) { return x * x; }
__device__ __forceinline__ float sqrt_f32(float x) { return sqrtf(x); }
__device__ __forceinline__ float rcp_f32(float x) { return fast_rcp(x); }
__device__ __forceinline__ float abs_f32(float x) { return fabsf(x); }
__device__ __forceinline__ float ceil_f32(float x) { return ceilf(x); }
__device__ __forceinline__ float floor_f32(float x) { return floorf(x); }
__device__ __forceinline__ float round_f32(float x) { return roundf(x); }
__device__ __forceinline__ float sin_f32(float x) { return sinf(x); }
__device__ __forceinline__ float cos_f32(float x) { return cosf(x); }
__device__ __forceinline__ float exp_f32(float x) { return fast_exp::exp<float>(x); }
__device__ __forceinline__ float log_f32(float x) { return logf(x); }  // Keep standard for accuracy
__device__ __forceinline__ float tanh_f32(float x) { return tanhf(x); }
__device__ __forceinline__ float erf_f32(float x) { return erff(x); }
__device__ __forceinline__ float normcdf_f32(float x) { return normcdff(x); }
__device__ __forceinline__ float sign_f32(float x) { 
    return (x > 0.0f) - (x < 0.0f); 
}
__device__ __forceinline__ float pow_f32(float x, float p) { return powf(x, p); }

// Scalar half operations for remainder handling
__device__ __forceinline__ __half h_id(__half x) { return x; }
__device__ __forceinline__ __half h_neg(__half x) { return __hneg(x); }
__device__ __forceinline__ __half h_sqr(__half x) { return __hmul(x, x); }
__device__ __forceinline__ __half h_sqrt(__half x) { return hsqrt(x); }
__device__ __forceinline__ __half h_rcp(__half x) { return hrcp(x); }
__device__ __forceinline__ __half h_abs(__half x) { return __habs(x); }
__device__ __forceinline__ __half h_exp(__half x) { return hexp(x); }
__device__ __forceinline__ __half h_log(__half x) { return hlog(x); }
__device__ __forceinline__ __half h_sin(__half x) { return hsin(x); }
__device__ __forceinline__ __half h_cos(__half x) { return hcos(x); }
__device__ __forceinline__ __half h_ceil(__half x) { return hceil(x); }
__device__ __forceinline__ __half h_floor(__half x) { return hfloor(x); }
__device__ __forceinline__ __half h_round(__half x) { return hrint(x); }
__device__ __forceinline__ __half h_tanh(__half x) { return __float2half(tanhf(__half2float(x))); }
__device__ __forceinline__ __half h_erf(__half x) { return __float2half(erff(__half2float(x))); }
__device__ __forceinline__ __half h_normcdf(__half x) { return __float2half(normcdff(__half2float(x))); }
__device__ __forceinline__ __half h_silu(__half x) { 
    float fx = __half2float(x);
    return __float2half(fast_silu_f32(fx)); 
}
__device__ __forceinline__ __half h_sigmoid(__half x) { 
    float fx = __half2float(x);
    return __float2half(fast_sigmoid_f32(fx)); 
}
__device__ __forceinline__ __half h_gelu(__half x) { 
    float fx = __half2float(x);
    return __float2half(fast_gelu_f32(fx)); 
}
__device__ __forceinline__ __half h_gelu_erf(__half x) { 
    float fx = __half2float(x);
    return __float2half(fast_gelu_erf_f32(fx)); 
}
__device__ __forceinline__ __half h_relu(__half x) { 
    return __hmax(x, __float2half(0.0f)); 
}
__device__ __forceinline__ __half h_elu(__half x, __half alpha) { 
    float fx = __half2float(x);
    float fa = __half2float(alpha);
    return __float2half(fast_elu_f32(fx, fa)); 
}
__device__ __forceinline__ __half h_sign(__half x) { 
    float fx = __half2float(x);
    return __float2half(sign_f32(fx)); 
}
__device__ __forceinline__ __half h_pow(__half x, __half p) { 
    return __float2half(powf(__half2float(x), __half2float(p))); 
}

// Half2 identity and simple ops
__device__ __forceinline__ __half2 h2_id(__half2 x) { return x; }

// =============================================================================
// OPTIMIZED UNARY KERNEL MACRO - BFLOAT16 WITH BFLOAT162 VECTORIZATION
// =============================================================================

// BF16x2 identity
__device__ __forceinline__ __nv_bfloat162 bf162_id(__nv_bfloat162 x) { return x; }
__device__ __forceinline__ __nv_bfloat16 bf16_id(__nv_bfloat16 x) { return x; }

#define UNARY_OP_BF16_VEC2(FN_NAME, FUNC_VEC2, FUNC_SCALAR) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* __restrict__ info, \
    const __nv_bfloat16* __restrict__ inp, \
    __nv_bfloat16* __restrict__ out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        const __nv_bfloat16* __restrict__ src = inp ? inp : out; \
        /* Check 4-byte alignment for bfloat162 vectorization */ \
        bool aligned = ((uintptr_t)src % 4 == 0) && ((uintptr_t)out % 4 == 0); \
        \
        if (aligned && numel >= 2) { \
            const size_t vec_numel = numel / 2; \
            const size_t vec_offset = vec_numel * 2; \
            \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < vec_numel; i += blockDim.x * gridDim.x) { \
                __nv_bfloat162 v = ldg_load_bf162(src + i * 2); \
                __nv_bfloat162 r = FUNC_VEC2(v); \
                store_bf162(out + i * 2, r); \
            } \
            \
            /* Handle odd element */ \
            if (vec_offset < numel && blockIdx.x == 0 && threadIdx.x == 0) { \
                __nv_bfloat16 x = __ldg(src + vec_offset); \
                out[vec_offset] = FUNC_SCALAR(x); \
            } \
        } else { \
            /* Scalar fallback */ \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                __nv_bfloat16 x = src ? __ldg(src + i) : out[i]; \
                out[i] = FUNC_SCALAR(x); \
            } \
        } \
    } else { \
        const __nv_bfloat16* __restrict__ src = inp; \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            size_t strided_i = get_strided_index(i, num_dims, dims, strides); \
            __nv_bfloat16 x = src ? src[strided_i] : out[i]; \
            out[i] = FUNC_SCALAR(x); \
        } \
    } \
}

// =============================================================================
// OPTIMIZED UNARY KERNEL MACRO - UINT8 WITH UCHAR4 VECTORIZATION
// =============================================================================

// Load 4 uint8 using __ldg
__device__ __forceinline__ uchar4 ldg_load_uchar4(const uint8_t* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const uchar4*>(ptr));
}

// Store 4 uint8
__device__ __forceinline__ void store_uchar4(uint8_t* __restrict__ ptr, uchar4 val) {
    *reinterpret_cast<uchar4*>(ptr) = val;
}

#define UNARY_OP_U8_VEC4(FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* __restrict__ info, \
    const uint8_t* __restrict__ inp, \
    uint8_t* __restrict__ out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        const uint8_t* __restrict__ src = inp ? inp : out; \
        /* Check 4-byte alignment for uchar4 vectorization */ \
        bool aligned = ((uintptr_t)src % 4 == 0) && ((uintptr_t)out % 4 == 0); \
        \
        if (aligned && numel >= 4) { \
            const size_t vec_numel = numel / 4; \
            const size_t vec_offset = vec_numel * 4; \
            \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < vec_numel; i += blockDim.x * gridDim.x) { \
                uchar4 v = ldg_load_uchar4(src + i * 4); \
                uchar4 r; \
                r.x = FUNC(v.x); \
                r.y = FUNC(v.y); \
                r.z = FUNC(v.z); \
                r.w = FUNC(v.w); \
                store_uchar4(out + i * 4, r); \
            } \
            \
            for (size_t i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                uint8_t x = __ldg(src + i); \
                out[i] = FUNC(x); \
            } \
        } else { \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                uint8_t x = src ? __ldg(src + i) : out[i]; \
                out[i] = FUNC(x); \
            } \
        } \
    } else { \
        const uint8_t* __restrict__ src = inp; \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            size_t strided_i = get_strided_index(i, num_dims, dims, strides); \
            uint8_t x = src ? src[strided_i] : out[i]; \
            out[i] = FUNC(x); \
        } \
    } \
}

// =============================================================================
// OPTIMIZED UNARY KERNEL MACRO - UINT32 WITH UINT4 VECTORIZATION
// =============================================================================

// Load 4 uint32 using __ldg
__device__ __forceinline__ uint4 ldg_load_uint4(const uint32_t* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const uint4*>(ptr));
}

// Store 4 uint32
__device__ __forceinline__ void store_uint4(uint32_t* __restrict__ ptr, uint4 val) {
    *reinterpret_cast<uint4*>(ptr) = val;
}

#define UNARY_OP_U32_VEC4(FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* __restrict__ info, \
    const uint32_t* __restrict__ inp, \
    uint32_t* __restrict__ out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        const uint32_t* __restrict__ src = inp ? inp : out; \
        /* Check 16-byte alignment for uint4 vectorization */ \
        bool aligned = ((uintptr_t)src % 16 == 0) && ((uintptr_t)out % 16 == 0); \
        \
        if (aligned && numel >= 4) { \
            const size_t vec_numel = numel / 4; \
            const size_t vec_offset = vec_numel * 4; \
            \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < vec_numel; i += blockDim.x * gridDim.x) { \
                uint4 v = ldg_load_uint4(src + i * 4); \
                uint4 r; \
                r.x = FUNC(v.x); \
                r.y = FUNC(v.y); \
                r.z = FUNC(v.z); \
                r.w = FUNC(v.w); \
                store_uint4(out + i * 4, r); \
            } \
            \
            for (size_t i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                uint32_t x = __ldg(src + i); \
                out[i] = FUNC(x); \
            } \
        } else { \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                uint32_t x = src ? __ldg(src + i) : out[i]; \
                out[i] = FUNC(x); \
            } \
        } \
    } else { \
        const uint32_t* __restrict__ src = inp; \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            size_t strided_i = get_strided_index(i, num_dims, dims, strides); \
            uint32_t x = src ? src[strided_i] : out[i]; \
            out[i] = FUNC(x); \
        } \
    } \
}

// =============================================================================
// OPTIMIZED UNARY KERNEL MACRO - INT64 WITH LONGLONG2 VECTORIZATION
// =============================================================================

// Load 2 int64 using __ldg
__device__ __forceinline__ longlong2 ldg_load_longlong2(const int64_t* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const longlong2*>(ptr));
}

// Store 2 int64
__device__ __forceinline__ void store_longlong2(int64_t* __restrict__ ptr, longlong2 val) {
    *reinterpret_cast<longlong2*>(ptr) = val;
}

#define UNARY_OP_I64_VEC2(FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* __restrict__ info, \
    const int64_t* __restrict__ inp, \
    int64_t* __restrict__ out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        const int64_t* __restrict__ src = inp ? inp : out; \
        /* Check 16-byte alignment for longlong2 vectorization */ \
        bool aligned = ((uintptr_t)src % 16 == 0) && ((uintptr_t)out % 16 == 0); \
        \
        if (aligned && numel >= 2) { \
            const size_t vec_numel = numel / 2; \
            const size_t vec_offset = vec_numel * 2; \
            \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < vec_numel; i += blockDim.x * gridDim.x) { \
                longlong2 v = ldg_load_longlong2(src + i * 2); \
                longlong2 r; \
                r.x = FUNC(v.x); \
                r.y = FUNC(v.y); \
                store_longlong2(out + i * 2, r); \
            } \
            \
            for (size_t i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                int64_t x = __ldg(src + i); \
                out[i] = FUNC(x); \
            } \
        } else { \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                int64_t x = src ? __ldg(src + i) : out[i]; \
                out[i] = FUNC(x); \
            } \
        } \
    } else { \
        const int64_t* __restrict__ src = inp; \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            size_t strided_i = get_strided_index(i, num_dims, dims, strides); \
            int64_t x = src ? src[strided_i] : out[i]; \
            out[i] = FUNC(x); \
        } \
    } \
}

// Identity functions for integer types
__device__ __forceinline__ uint8_t id_u8(uint8_t x) { return x; }
__device__ __forceinline__ uint32_t id_u32(uint32_t x) { return x; }
__device__ __forceinline__ int64_t id_i64(int64_t x) { return x; }

// =============================================================================
// OPTIMIZED UNARY KERNEL MACRO - F8E4M3 WITH 4-BYTE VECTORIZATION
// =============================================================================
// FP8 is 1 byte, so we use uint32_t to load/store 4 elements at once

// Identity for F8E4M3
__device__ __forceinline__ __nv_fp8_e4m3 id_f8e4m3(__nv_fp8_e4m3 x) { return x; }

#define UNARY_OP_F8E4M3_VEC4(FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* __restrict__ info, \
    const __nv_fp8_e4m3* __restrict__ inp, \
    __nv_fp8_e4m3* __restrict__ out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        const __nv_fp8_e4m3* __restrict__ src = inp ? inp : out; \
        /* Check 4-byte alignment for uint32 vectorization (4 x fp8) */ \
        bool aligned = ((uintptr_t)src % 4 == 0) && ((uintptr_t)out % 4 == 0); \
        \
        if (aligned && numel >= 4) { \
            const size_t vec_numel = numel / 4; \
            const size_t vec_offset = vec_numel * 4; \
            const uint32_t* src_vec = reinterpret_cast<const uint32_t*>(src); \
            uint32_t* out_vec = reinterpret_cast<uint32_t*>(out); \
            \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < vec_numel; i += blockDim.x * gridDim.x) { \
                /* For copy, load 4 bytes and store directly */ \
                out_vec[i] = __ldg(src_vec + i); \
            } \
            \
            for (size_t i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                out[i] = FUNC(src[i]); \
            } \
        } else { \
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                __nv_fp8_e4m3 x = src ? src[i] : out[i]; \
                out[i] = FUNC(x); \
            } \
        } \
    } else { \
        const __nv_fp8_e4m3* __restrict__ src = inp; \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            size_t strided_i = get_strided_index(i, num_dims, dims, strides); \
            __nv_fp8_e4m3 x = src ? src[strided_i] : out[i]; \
            out[i] = FUNC(x); \
        } \
    } \
}
