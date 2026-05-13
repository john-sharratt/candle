#define _USE_MATH_DEFINES
#include<math.h>
#include<stdint.h>
#include "cuda_utils.cuh"
#include "unary_utils.cuh"
#include "../fast_exp.cuh"

// =============================================================================
// LEGACY UNARY_OP MACRO - For types without vectorization or simple ops
// =============================================================================
// Now uses __restrict__ and __ldg for read-only access, 32-bit indices

#define UNARY_OP(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* __restrict__ info, \
    const TYPENAME* __restrict__ inp, \
    TYPENAME* __restrict__ out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        const TYPENAME* __restrict__ src = inp ? inp : out; \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = src[i]; \
            out[i] = FUNC; \
        } \
    } \
    else { \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            size_t strided_i = get_strided_index(i, num_dims, dims, strides); \
            TYPENAME x = inp ? inp[strided_i] : out[i]; \
            out[i] = FUNC; \
        } \
    } \
} \

// =============================================================================
// FP8 UNARY_OP MACRO - Standard implementation for __nv_fp8_e4m3
// =============================================================================
#define UNARY_OP_F8E4M3(FN_NAME, FUNC) \
    UNARY_OP(__nv_fp8_e4m3, FN_NAME, FUNC)

#define UNARY_OP1_F8E4M3(FN_NAME, FUNC) \
    UNARY_OP1(__nv_fp8_e4m3, FN_NAME, FUNC)

template<typename T>
__device__ __forceinline__ T gelu_erf_fwd(T x) {
  return x * normcdfg(x);
}

template<typename T>
__device__ __forceinline__ T gelu_fwd(T x) {
    T x_sq = x * x;
    T x_cube = x_sq * x;
    T alpha = x + static_cast<T>(0.044715) * x_cube;
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + tanhg(static_cast<T>(M_2_SQRTPI * M_SQRT1_2) * alpha));
}

template<typename T>
__device__ __forceinline__ T elu_fwd(T x, T alpha) {
  if (x > static_cast<T>(0)) {
    return x;
  }
  return alpha * (expg(x) - static_cast<T>(1));
}

template<typename T>
__device__ __forceinline__ T relu_fwd(T x) {
    T zero = 0.;
    return maxg(x, zero);
}

// SiLU: Generic version uses standard math
template<typename T>
__device__ __forceinline__ T silu_fwd(T x) {
    return x / (static_cast<T>(1) + expg(-x));
}

// SiLU: Float specialization with fast_exp library
template<>
__device__ __forceinline__ float silu_fwd<float>(float x) {
    return fast_exp::silu<float>(x);
}

// SiLU: Double specialization (can't use fast intrinsics)
template<>
__device__ __forceinline__ double silu_fwd<double>(double x) {
    return x / (1.0 + exp(-x));
}

// Sigmoid: Generic version
template<typename T>
__device__ __forceinline__ T sigmoid_fwd(T x) {
    return recipg(static_cast<T>(1) + expg(-x));
}

// Sigmoid: Float specialization with fast_exp library
template<>
__device__ __forceinline__ float sigmoid_fwd<float>(float x) {
    return fast_exp::sigmoid<float>(x);
}

// Sigmoid: Double specialization
template<>
__device__ __forceinline__ double sigmoid_fwd<double>(double x) {
    return 1.0 / (1.0 + exp(-x));
}

#define UNARY_OP1(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* __restrict__ info, \
    const TYPENAME param, \
    const TYPENAME* __restrict__ inp, \
    TYPENAME* __restrict__ out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        const TYPENAME* __restrict__ src = inp ? inp : out; \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = src[i]; \
            out[i] = FUNC; \
        } \
    } \
    else { \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            size_t strided_i = get_strided_index(i, num_dims, dims, strides); \
            TYPENAME x = inp ? inp[strided_i] : out[i]; \
            out[i] = FUNC; \
        } \
    } \
} \

template<typename T>
__device__ T sign_(T t) {
  return static_cast<T>(t > static_cast<T>(0)) - static_cast<T>(t < static_cast<T>(0));
}


// Copy uses vectorized bf16x2 for better memory throughput
UNARY_OP_BF16_VEC2(ucopy_bf16, bf162_id, bf16_id)
UNARY_OP(__nv_bfloat16, uneg_bf16, -x)
UNARY_OP(__nv_bfloat16, urecip_bf16, recipg(x))
UNARY_OP(__nv_bfloat16, uexp_bf16, expg(x))
UNARY_OP(__nv_bfloat16, ulog_bf16, logg(x))
UNARY_OP(__nv_bfloat16, usin_bf16, sing(x))
UNARY_OP(__nv_bfloat16, ucos_bf16, cosg(x))
UNARY_OP(__nv_bfloat16, utanh_bf16, tanhg(x))
UNARY_OP(__nv_bfloat16, uerf_bf16, erfg(x))
UNARY_OP(__nv_bfloat16, uceil_bf16, ceilg(x))
UNARY_OP(__nv_bfloat16, ufloor_bf16, floorg(x))
UNARY_OP(__nv_bfloat16, uround_bf16, roundg(x))
UNARY_OP(__nv_bfloat16, unormcdf_bf16, normcdfg(x))
UNARY_OP(__nv_bfloat16, uabs_bf16, absg(x))
UNARY_OP(__nv_bfloat16, usqr_bf16, x*x)
UNARY_OP(__nv_bfloat16, usqrt_bf16, sqrtg(x))
UNARY_OP(__nv_bfloat16, ugelu_bf16, gelu_fwd(x))
UNARY_OP(__nv_bfloat16, ugelu_erf_bf16, gelu_erf_fwd(x))
UNARY_OP(__nv_bfloat16, urelu_bf16, relu_fwd(x))
UNARY_OP1(__nv_bfloat16, uelu_bf16, elu_fwd(x, param))
UNARY_OP(__nv_bfloat16, usilu_bf16, silu_fwd(x))
UNARY_OP1(__nv_bfloat16, upowf_bf16, powg(x, param))
UNARY_OP(__nv_bfloat16, usign_bf16, sign_(x))
UNARY_OP(__nv_bfloat16, usigmoid_bf16, sigmoid_fwd(x))

// Use vectorized FP8 macro for copy (4 bytes at a time)
UNARY_OP_F8E4M3_VEC4(ucopy_f8_e4m3, id_f8e4m3)
UNARY_OP_F8E4M3(uneg_fp8_e4m3, __nv_fp8_e4m3(-F8E4M3_TO_FLOAT(x)))
UNARY_OP_F8E4M3(urecip_fp8_e4m3, recipg(x))
UNARY_OP_F8E4M3(uexp_fp8_e4m3, expg(x))
UNARY_OP_F8E4M3(ulog_fp8_e4m3, logg(x))
UNARY_OP_F8E4M3(usin_fp8_e4m3, sing(x))
UNARY_OP_F8E4M3(ucos_fp8_e4m3, cosg(x))
UNARY_OP_F8E4M3(utanh_fp8_e4m3, tanhg(x))
UNARY_OP_F8E4M3(uerf_fp8_e4m3, erfg(x))
UNARY_OP_F8E4M3(uceil_fp8_e4m3, ceilg(x))
UNARY_OP_F8E4M3(ufloor_fp8_e4m3, floorg(x))
UNARY_OP_F8E4M3(uround_fp8_e4m3, roundg(x))
UNARY_OP_F8E4M3(unormcdf_fp8_e4m3, normcdfg(x))
UNARY_OP_F8E4M3(uabs_fp8_e4m3, absg(x))
UNARY_OP_F8E4M3(usqr_fp8_e4m3, __nv_fp8_e4m3(F8E4M3_TO_FLOAT(x)*F8E4M3_TO_FLOAT(x)))
UNARY_OP_F8E4M3(usqrt_fp8_e4m3, sqrtg(x))
UNARY_OP_F8E4M3(ugelu_fp8_e4m3, __nv_fp8_e4m3(gelu_fwd(F8E4M3_TO_FLOAT(x))))
UNARY_OP_F8E4M3(ugelu_erf_fp8_e4m3, __nv_fp8_e4m3(gelu_erf_fwd(F8E4M3_TO_FLOAT(x))))
UNARY_OP_F8E4M3(urelu_fp8_e4m3, __nv_fp8_e4m3(relu_fwd(F8E4M3_TO_FLOAT(x))))
UNARY_OP1_F8E4M3(uelu_fp8_e4m3, __nv_fp8_e4m3(elu_fwd(F8E4M3_TO_FLOAT(x), F8E4M3_TO_FLOAT(param))))
UNARY_OP_F8E4M3(usilu_fp8_e4m3, __nv_fp8_e4m3(silu_fwd(F8E4M3_TO_FLOAT(x))))
UNARY_OP1_F8E4M3(upowf_fp8_e4m3, powg(x, param))
UNARY_OP_F8E4M3(usign_fp8_e4m3, __nv_fp8_e4m3(sign_(F8E4M3_TO_FLOAT(x))))
UNARY_OP_F8E4M3(usigmoid_fp8_e4m3, __nv_fp8_e4m3(sigmoid_fwd(F8E4M3_TO_FLOAT(x))))

// =============================================================================
// HALF PRECISION KERNELS - WITH HALF2 VECTORIZATION
// =============================================================================

// Optimized half kernels with half2 vectorization
UNARY_OP_F16_VEC2(ucopy_f16, h2_id, h_id)
UNARY_OP_F16_VEC2(uneg_f16, h2_neg, h_neg)
UNARY_OP_F16_VEC2(urecip_f16, h2_rcp, h_rcp)
UNARY_OP_F16_VEC2(uexp_f16, h2_exp, h_exp)
UNARY_OP_F16_VEC2(ulog_f16, h2_log, h_log)
UNARY_OP_F16_VEC2(usin_f16, h2_sin, h_sin)
UNARY_OP_F16_VEC2(ucos_f16, h2_cos, h_cos)
UNARY_OP_F16_VEC2(utanh_f16, fast_tanh_h2, h_tanh)
UNARY_OP_F16_VEC2(uerf_f16, h2_erf, h_erf)
UNARY_OP_F16_VEC2(uceil_f16, h2_ceil, h_ceil)
UNARY_OP_F16_VEC2(ufloor_f16, h2_floor, h_floor)
UNARY_OP_F16_VEC2(uround_f16, h2_round, h_round)
UNARY_OP_F16_VEC2(unormcdf_f16, h2_normcdf, h_normcdf)
UNARY_OP_F16_VEC2(uabs_f16, h2_abs, h_abs)
UNARY_OP_F16_VEC2(usqr_f16, h2_sqr, h_sqr)
UNARY_OP_F16_VEC2(usqrt_f16, h2_sqrt, h_sqrt)
UNARY_OP_F16_VEC2(ugelu_f16, fast_gelu_h2, h_gelu)
UNARY_OP_F16_VEC2(ugelu_erf_f16, fast_gelu_erf_h2, h_gelu_erf)
UNARY_OP_F16_VEC2(urelu_f16, fast_relu_h2, h_relu)
UNARY_OP1_F16_VEC2(uelu_f16, fast_elu_h2, h_elu)
UNARY_OP_F16_VEC2(usilu_f16, fast_silu_h2, h_silu)
UNARY_OP1_F16_VEC2(upowf_f16, ([](auto x, auto p) { return __floats2half2_rn(powf(__half2float(__low2half(x)), __half2float(__low2half(p))), powf(__half2float(__high2half(x)), __half2float(__high2half(p)))); }), h_pow)
UNARY_OP_F16_VEC2(usign_f16, h2_sign, h_sign)
UNARY_OP_F16_VEC2(usigmoid_f16, fast_sigmoid_h2, h_sigmoid)

// =============================================================================
// INTEGER AND COPY OPERATIONS - Vectorized for better memory throughput
// =============================================================================

UNARY_OP_U8_VEC4(ucopy_u8, id_u8)
UNARY_OP_U32_VEC4(ucopy_u32, id_u32)
UNARY_OP_I64_VEC2(ucopy_i64, id_i64)

// =============================================================================
// FLOAT32 KERNELS - WITH FLOAT4 VECTORIZATION
// =============================================================================

// Copy - vectorized (with alignment check for float4)
extern "C" __global__ void ucopy_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t* __restrict__ info,
    const float* __restrict__ inp,
    float* __restrict__ out
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        const float* __restrict__ src = inp ? inp : out;
        /* Check 16-byte alignment for float4 vectorization */
        bool aligned = ((uintptr_t)src % 16 == 0) && ((uintptr_t)out % 16 == 0);
        
        if (aligned && numel >= 4) {
            const size_t vec_numel = numel / 4;
            const size_t vec_offset = vec_numel * 4;
            
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
                 i < vec_numel; i += blockDim.x * gridDim.x) {
                float4 v = ldg_load_float4(src + i * 4);
                store_float4(out + i * 4, v);
            }
            
            for (size_t i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; 
                 i < numel; i += blockDim.x * gridDim.x) {
                out[i] = __ldg(src + i);
            }
        } else {
            /* Scalar fallback for unaligned data */
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
                 i < numel; i += blockDim.x * gridDim.x) {
                out[i] = __ldg(src + i);
            }
        }
    } else {
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
             i < numel; i += blockDim.x * gridDim.x) {
            size_t strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = inp ? __ldg(inp + strided_i) : out[i];
        }
    }
}

// Negation - vectorized
UNARY_OP_F32_VEC4(uneg_f32, neg_f32)

// Reciprocal - vectorized with fast intrinsic
UNARY_OP_F32_VEC4(urecip_f32, rcp_f32)

// Exp - vectorized with fast intrinsic
UNARY_OP_F32_VEC4(uexp_f32, exp_f32)

// Log - vectorized (keep standard for accuracy)
UNARY_OP_F32_VEC4(ulog_f32, log_f32)

// Sin - vectorized
UNARY_OP_F32_VEC4(usin_f32, sin_f32)

// Cos - vectorized
UNARY_OP_F32_VEC4(ucos_f32, cos_f32)

// Tanh - vectorized
UNARY_OP_F32_VEC4(utanh_f32, tanh_f32)

// Erf - vectorized
UNARY_OP_F32_VEC4(uerf_f32, erf_f32)

// Ceil - vectorized
UNARY_OP_F32_VEC4(uceil_f32, ceil_f32)

// Floor - vectorized
UNARY_OP_F32_VEC4(ufloor_f32, floor_f32)

// Round - vectorized
UNARY_OP_F32_VEC4(uround_f32, round_f32)

// NormCDF - vectorized
UNARY_OP_F32_VEC4(unormcdf_f32, normcdf_f32)

// Abs - vectorized
UNARY_OP_F32_VEC4(uabs_f32, abs_f32)

// Sqr - vectorized
UNARY_OP_F32_VEC4(usqr_f32, sqr_f32)

// Sqrt - vectorized
UNARY_OP_F32_VEC4(usqrt_f32, sqrt_f32)

// GELU - vectorized with fast intrinsics
UNARY_OP_F32_VEC4(ugelu_f32, fast_gelu_f32)

// GELU erf - vectorized
UNARY_OP_F32_VEC4(ugelu_erf_f32, fast_gelu_erf_f32)

// ReLU - vectorized
UNARY_OP_F32_VEC4(urelu_f32, fast_relu_f32)

// ELU - vectorized with param
UNARY_OP1_F32_VEC4(uelu_f32, fast_elu_f32)

// SiLU - vectorized with fast intrinsics
UNARY_OP_F32_VEC4(usilu_f32, fast_silu_f32)

// Pow - vectorized with param
UNARY_OP1_F32_VEC4(upowf_f32, pow_f32)

// Sign - vectorized
UNARY_OP_F32_VEC4(usign_f32, sign_f32)

// Sigmoid - vectorized with fast intrinsics
UNARY_OP_F32_VEC4(usigmoid_f32, fast_sigmoid_f32)

// =============================================================================
// DOUBLE PRECISION KERNELS - WITH DOUBLE2 VECTORIZATION
// =============================================================================

// Double helper functions
__device__ __forceinline__ double id_f64(double x) { return x; }
__device__ __forceinline__ double neg_f64(double x) { return -x; }
__device__ __forceinline__ double sqr_f64(double x) { return x * x; }
__device__ __forceinline__ double sqrt_f64(double x) { return sqrt(x); }
__device__ __forceinline__ double rcp_f64(double x) { return 1.0 / x; }
__device__ __forceinline__ double abs_f64(double x) { return fabs(x); }
__device__ __forceinline__ double ceil_f64(double x) { return ceil(x); }
__device__ __forceinline__ double floor_f64(double x) { return floor(x); }
__device__ __forceinline__ double round_f64(double x) { return round(x); }
__device__ __forceinline__ double sin_f64(double x) { return sin(x); }
__device__ __forceinline__ double cos_f64(double x) { return cos(x); }
__device__ __forceinline__ double exp_f64(double x) { return exp(x); }
__device__ __forceinline__ double log_f64(double x) { return log(x); }
__device__ __forceinline__ double tanh_f64(double x) { return tanh(x); }
__device__ __forceinline__ double erf_f64(double x) { return erf(x); }
__device__ __forceinline__ double normcdf_f64(double x) { return normcdf(x); }
__device__ __forceinline__ double sign_f64(double x) { 
    return (x > 0.0) - (x < 0.0); 
}
__device__ __forceinline__ double pow_f64(double x, double p) { return pow(x, p); }

__device__ __forceinline__ double gelu_f64(double x) {
    const double kSqrt2OverPi = 0.7978845608028654;
    const double kAlpha = 0.044715;
    double x3 = x * x * x;
    double inner = kSqrt2OverPi * (x + kAlpha * x3);
    return 0.5 * x * (1.0 + tanh(inner));
}

__device__ __forceinline__ double gelu_erf_f64(double x) {
    return x * normcdf(x);
}

__device__ __forceinline__ double relu_f64(double x) {
    return fmax(x, 0.0);
}

__device__ __forceinline__ double elu_f64(double x, double alpha) {
    return (x > 0.0) ? x : alpha * (exp(x) - 1.0);
}

__device__ __forceinline__ double silu_f64(double x) {
    return x / (1.0 + exp(-x));
}

__device__ __forceinline__ double sigmoid_f64(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Double vectorized macro
#define UNARY_OP_F64_VEC2(FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* __restrict__ info, \
    const double* __restrict__ inp, \
    double* __restrict__ out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        const size_t vec_numel = numel / 2; \
        const size_t vec_offset = vec_numel * 2; \
        const double* __restrict__ src = inp ? inp : out; \
        \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < vec_numel; i += blockDim.x * gridDim.x) { \
            double2 v = ldg_load_double2(src + i * 2); \
            double2 r; \
            r.x = FUNC(v.x); \
            r.y = FUNC(v.y); \
            store_double2(out + i * 2, r); \
        } \
        \
        for (size_t i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            double x = __ldg(src + i); \
            out[i] = FUNC(x); \
        } \
    } else { \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            size_t strided_i = get_strided_index(i, num_dims, dims, strides); \
            double x = inp ? __ldg(inp + strided_i) : out[i]; \
            out[i] = FUNC(x); \
        } \
    } \
}

#define UNARY_OP1_F64_VEC2(FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t* __restrict__ info, \
    const double param, \
    const double* __restrict__ inp, \
    double* __restrict__ out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        const size_t vec_numel = numel / 2; \
        const size_t vec_offset = vec_numel * 2; \
        const double* __restrict__ src = inp ? inp : out; \
        \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < vec_numel; i += blockDim.x * gridDim.x) { \
            double2 v = ldg_load_double2(src + i * 2); \
            double2 r; \
            r.x = FUNC(v.x, param); \
            r.y = FUNC(v.y, param); \
            store_double2(out + i * 2, r); \
        } \
        \
        for (size_t i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            double x = __ldg(src + i); \
            out[i] = FUNC(x, param); \
        } \
    } else { \
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            size_t strided_i = get_strided_index(i, num_dims, dims, strides); \
            double x = inp ? __ldg(inp + strided_i) : out[i]; \
            out[i] = FUNC(x, param); \
        } \
    } \
}

// Double kernels with vectorization
UNARY_OP_F64_VEC2(ucopy_f64, id_f64)
UNARY_OP_F64_VEC2(uneg_f64, neg_f64)
UNARY_OP_F64_VEC2(urecip_f64, rcp_f64)
UNARY_OP_F64_VEC2(uexp_f64, exp_f64)
UNARY_OP_F64_VEC2(ulog_f64, log_f64)
UNARY_OP_F64_VEC2(usin_f64, sin_f64)
UNARY_OP_F64_VEC2(ucos_f64, cos_f64)
UNARY_OP_F64_VEC2(utanh_f64, tanh_f64)
UNARY_OP_F64_VEC2(uerf_f64, erf_f64)
UNARY_OP_F64_VEC2(uceil_f64, ceil_f64)
UNARY_OP_F64_VEC2(ufloor_f64, floor_f64)
UNARY_OP_F64_VEC2(uround_f64, round_f64)
UNARY_OP_F64_VEC2(unormcdf_f64, normcdf_f64)
UNARY_OP_F64_VEC2(uabs_f64, abs_f64)
UNARY_OP_F64_VEC2(usqr_f64, sqr_f64)
UNARY_OP_F64_VEC2(usqrt_f64, sqrt_f64)
UNARY_OP_F64_VEC2(ugelu_f64, gelu_f64)
UNARY_OP_F64_VEC2(ugelu_erf_f64, gelu_erf_f64)
UNARY_OP_F64_VEC2(urelu_f64, relu_f64)
UNARY_OP1_F64_VEC2(uelu_f64, elu_f64)
UNARY_OP_F64_VEC2(usilu_f64, silu_f64)
UNARY_OP1_F64_VEC2(upowf_f64, pow_f64)
UNARY_OP_F64_VEC2(usign_f64, sign_f64)
UNARY_OP_F64_VEC2(usigmoid_f64, sigmoid_f64)
