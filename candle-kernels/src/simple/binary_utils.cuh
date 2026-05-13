#pragma once
// =============================================================================
// BINARY ELEMENTWISE UTILITIES - Optimized primitives for binary operations
// =============================================================================
// Key optimizations:
//   1. Vectorized memory access (float4 for fp32, half2 for fp16)
//   2. __ldg() for read-only input access (texture cache)
//   3. __restrict__ qualifiers for pointer aliasing hints
//   4. 32-bit index arithmetic
//   5. Fast division for contiguous paths
//   6. Separate contiguous vs strided kernel paths
//   7. half2/bf16x2 paired operations for 2x throughput
// =============================================================================

#include "cuda_utils.cuh"
#include <stdint.h>

// =============================================================================
// FAST MATH HELPERS
// =============================================================================

__device__ __forceinline__ float fast_div_f32(float x, float y) {
    return __fdividef(x, y);
}

// =============================================================================
// VECTORIZED LOAD/STORE HELPERS
// =============================================================================

__device__ __forceinline__ float4 ldg_float4(const float* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const float4*>(ptr));
}

__device__ __forceinline__ void store_float4(float* __restrict__ ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__device__ __forceinline__ double2 ldg_double2(const double* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const double2*>(ptr));
}

__device__ __forceinline__ void store_double2(double* __restrict__ ptr, double2 val) {
    *reinterpret_cast<double2*>(ptr) = val;
}

__device__ __forceinline__ __half2 ldg_half2(const __half* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const __half2*>(ptr));
}

__device__ __forceinline__ void store_half2(__half* __restrict__ ptr, __half2 val) {
    *reinterpret_cast<__half2*>(ptr) = val;
}

__device__ __forceinline__ __nv_bfloat162 ldg_bf162(const __nv_bfloat16* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const __nv_bfloat162*>(ptr));
}

__device__ __forceinline__ void store_bf162(__nv_bfloat16* __restrict__ ptr, __nv_bfloat162 val) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = val;
}

// =============================================================================
// OPTIMIZED BINARY OP MACRO - WITH VECTORIZATION AND __restrict__
// =============================================================================

#define BINARY_OP_OUT(TYPENAME, OUT_TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const unsigned int numel, \
    const size_t num_dims, \
    const size_t* __restrict__ dims_and_strides, \
    const TYPENAME* __restrict__ lhs, \
    const TYPENAME* __restrict__ rhs, \
    OUT_TYPENAME* __restrict__ out \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides); \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    if (lhs_cont && rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = __ldg(lhs + i); \
            TYPENAME y = __ldg(rhs + i); \
            out[i] = FUNC; \
        } \
    } else if (lhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            TYPENAME x = __ldg(lhs + i); \
            TYPENAME y = __ldg(rhs + rhs_i); \
            out[i] = FUNC; \
        } \
    } else if (rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                lhs_i += i_dim * lhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            TYPENAME x = __ldg(lhs + lhs_i); \
            TYPENAME y = __ldg(rhs + i); \
            out[i] = FUNC; \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                lhs_i += i_dim * lhs_strides[d]; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            TYPENAME x = __ldg(lhs + lhs_i); \
            TYPENAME y = __ldg(rhs + rhs_i); \
            out[i] = FUNC; \
        } \
    } \
}

#define BINARY_OP(TYPENAME, FN_NAME, FUNC) \
  BINARY_OP_OUT(TYPENAME, TYPENAME, FN_NAME, FUNC)

// =============================================================================
// VECTORIZED FLOAT32 BINARY OP - 4 ELEMENTS PER THREAD
// =============================================================================

#define BINARY_OP_F32_VEC4(FN_NAME, OP) \
extern "C" __global__ void FN_NAME( \
    const unsigned int numel, \
    const size_t num_dims, \
    const size_t* __restrict__ dims_and_strides, \
    const float* __restrict__ lhs, \
    const float* __restrict__ rhs, \
    float* __restrict__ out \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides); \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    \
    if (lhs_cont && rhs_cont) { \
        /* Vectorized path: process 4 elements per thread */ \
        const unsigned int vec_numel = numel / 4; \
        const unsigned int vec_offset = vec_numel * 4; \
        \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < vec_numel; i += blockDim.x * gridDim.x) { \
            float4 a = ldg_float4(lhs + i * 4); \
            float4 b = ldg_float4(rhs + i * 4); \
            float4 c; \
            c.x = a.x OP b.x; \
            c.y = a.y OP b.y; \
            c.z = a.z OP b.z; \
            c.w = a.w OP b.w; \
            store_float4(out + i * 4, c); \
        } \
        \
        /* Handle remaining elements */ \
        for (unsigned int i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            float x = __ldg(lhs + i); \
            float y = __ldg(rhs + i); \
            out[i] = x OP y; \
        } \
    } else if (lhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            float x = __ldg(lhs + i); \
            float y = __ldg(rhs + rhs_i); \
            out[i] = x OP y; \
        } \
    } else if (rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                lhs_i += i_dim * lhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            float x = __ldg(lhs + lhs_i); \
            float y = __ldg(rhs + i); \
            out[i] = x OP y; \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                lhs_i += i_dim * lhs_strides[d]; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            float x = __ldg(lhs + lhs_i); \
            float y = __ldg(rhs + rhs_i); \
            out[i] = x OP y; \
        } \
    } \
}

// Specialized macro for division with fast math
#define BINARY_OP_F32_DIV_VEC4(FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const unsigned int numel, \
    const size_t num_dims, \
    const size_t* __restrict__ dims_and_strides, \
    const float* __restrict__ lhs, \
    const float* __restrict__ rhs, \
    float* __restrict__ out \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides); \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    \
    if (lhs_cont && rhs_cont) { \
        const unsigned int vec_numel = numel / 4; \
        const unsigned int vec_offset = vec_numel * 4; \
        \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < vec_numel; i += blockDim.x * gridDim.x) { \
            float4 a = ldg_float4(lhs + i * 4); \
            float4 b = ldg_float4(rhs + i * 4); \
            float4 c; \
            c.x = __fdividef(a.x, b.x); \
            c.y = __fdividef(a.y, b.y); \
            c.z = __fdividef(a.z, b.z); \
            c.w = __fdividef(a.w, b.w); \
            store_float4(out + i * 4, c); \
        } \
        \
        for (unsigned int i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = __fdividef(__ldg(lhs + i), __ldg(rhs + i)); \
        } \
    } else if (lhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = __fdividef(__ldg(lhs + i), __ldg(rhs + rhs_i)); \
        } \
    } else if (rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                lhs_i += i_dim * lhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = __fdividef(__ldg(lhs + lhs_i), __ldg(rhs + i)); \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                lhs_i += i_dim * lhs_strides[d]; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = __fdividef(__ldg(lhs + lhs_i), __ldg(rhs + rhs_i)); \
        } \
    } \
}

// Min/Max with fminf/fmaxf vectorized
#define BINARY_OP_F32_MINMAX_VEC4(FN_NAME, MINMAX_FUNC) \
extern "C" __global__ void FN_NAME( \
    const unsigned int numel, \
    const size_t num_dims, \
    const size_t* __restrict__ dims_and_strides, \
    const float* __restrict__ lhs, \
    const float* __restrict__ rhs, \
    float* __restrict__ out \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides); \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    \
    if (lhs_cont && rhs_cont) { \
        const unsigned int vec_numel = numel / 4; \
        const unsigned int vec_offset = vec_numel * 4; \
        \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < vec_numel; i += blockDim.x * gridDim.x) { \
            float4 a = ldg_float4(lhs + i * 4); \
            float4 b = ldg_float4(rhs + i * 4); \
            float4 c; \
            c.x = MINMAX_FUNC(a.x, b.x); \
            c.y = MINMAX_FUNC(a.y, b.y); \
            c.z = MINMAX_FUNC(a.z, b.z); \
            c.w = MINMAX_FUNC(a.w, b.w); \
            store_float4(out + i * 4, c); \
        } \
        \
        for (unsigned int i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = MINMAX_FUNC(__ldg(lhs + i), __ldg(rhs + i)); \
        } \
    } else if (lhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = MINMAX_FUNC(__ldg(lhs + i), __ldg(rhs + rhs_i)); \
        } \
    } else if (rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                lhs_i += i_dim * lhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = MINMAX_FUNC(__ldg(lhs + lhs_i), __ldg(rhs + i)); \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                lhs_i += i_dim * lhs_strides[d]; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = MINMAX_FUNC(__ldg(lhs + lhs_i), __ldg(rhs + rhs_i)); \
        } \
    } \
}

// =============================================================================
// VECTORIZED DOUBLE BINARY OP - 2 ELEMENTS PER THREAD
// =============================================================================

#define BINARY_OP_F64_VEC2(FN_NAME, OP) \
extern "C" __global__ void FN_NAME( \
    const unsigned int numel, \
    const size_t num_dims, \
    const size_t* __restrict__ dims_and_strides, \
    const double* __restrict__ lhs, \
    const double* __restrict__ rhs, \
    double* __restrict__ out \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides); \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    \
    if (lhs_cont && rhs_cont) { \
        const unsigned int vec_numel = numel / 2; \
        const unsigned int vec_offset = vec_numel * 2; \
        \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < vec_numel; i += blockDim.x * gridDim.x) { \
            double2 a = ldg_double2(lhs + i * 2); \
            double2 b = ldg_double2(rhs + i * 2); \
            double2 c; \
            c.x = a.x OP b.x; \
            c.y = a.y OP b.y; \
            store_double2(out + i * 2, c); \
        } \
        \
        for (unsigned int i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; \
             i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = __ldg(lhs + i) OP __ldg(rhs + i); \
        } \
    } else if (lhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = __ldg(lhs + i) OP __ldg(rhs + rhs_i); \
        } \
    } else if (rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                lhs_i += i_dim * lhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = __ldg(lhs + lhs_i) OP __ldg(rhs + i); \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                lhs_i += i_dim * lhs_strides[d]; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = __ldg(lhs + lhs_i) OP __ldg(rhs + rhs_i); \
        } \
    } \
}

// =============================================================================
// VECTORIZED HALF2 BINARY OP - 2 ELEMENTS PER THREAD WITH NATIVE INTRINSICS
// =============================================================================

#define BINARY_OP_F16_VEC2(FN_NAME, H2_OP, SCALAR_OP) \
extern "C" __global__ void FN_NAME( \
    const unsigned int numel, \
    const size_t num_dims, \
    const size_t* __restrict__ dims_and_strides, \
    const __half* __restrict__ lhs, \
    const __half* __restrict__ rhs, \
    __half* __restrict__ out \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides); \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    \
    if (lhs_cont && rhs_cont) { \
        const unsigned int vec_numel = numel / 2; \
        const unsigned int vec_offset = vec_numel * 2; \
        \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < vec_numel; i += blockDim.x * gridDim.x) { \
            __half2 a = ldg_half2(lhs + i * 2); \
            __half2 b = ldg_half2(rhs + i * 2); \
            __half2 c = H2_OP(a, b); \
            store_half2(out + i * 2, c); \
        } \
        \
        /* Handle odd element */ \
        if (vec_offset < numel && blockIdx.x == 0 && threadIdx.x == 0) { \
            out[vec_offset] = SCALAR_OP(lhs[vec_offset], rhs[vec_offset]); \
        } \
    } else if (lhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = SCALAR_OP(lhs[i], rhs[rhs_i]); \
        } \
    } else if (rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                lhs_i += i_dim * lhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = SCALAR_OP(lhs[lhs_i], rhs[i]); \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                lhs_i += i_dim * lhs_strides[d]; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = SCALAR_OP(lhs[lhs_i], rhs[rhs_i]); \
        } \
    } \
}

// Half scalar ops
__device__ __forceinline__ __half h_add(__half a, __half b) { return __hadd(a, b); }
__device__ __forceinline__ __half h_sub(__half a, __half b) { return __hsub(a, b); }
__device__ __forceinline__ __half h_mul(__half a, __half b) { return __hmul(a, b); }
__device__ __forceinline__ __half h_div(__half a, __half b) { return __hdiv(a, b); }
__device__ __forceinline__ __half h_max(__half a, __half b) { return __hmax(a, b); }
__device__ __forceinline__ __half h_min(__half a, __half b) { return __hmin(a, b); }

// =============================================================================
// VECTORIZED BFLOAT16X2 BINARY OP
// =============================================================================

#define BINARY_OP_BF16_VEC2(FN_NAME, BF2_OP, SCALAR_OP) \
extern "C" __global__ void FN_NAME( \
    const unsigned int numel, \
    const size_t num_dims, \
    const size_t* __restrict__ dims_and_strides, \
    const __nv_bfloat16* __restrict__ lhs, \
    const __nv_bfloat16* __restrict__ rhs, \
    __nv_bfloat16* __restrict__ out \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides); \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    \
    if (lhs_cont && rhs_cont) { \
        const unsigned int vec_numel = numel / 2; \
        const unsigned int vec_offset = vec_numel * 2; \
        \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < vec_numel; i += blockDim.x * gridDim.x) { \
            __nv_bfloat162 a = ldg_bf162(lhs + i * 2); \
            __nv_bfloat162 b = ldg_bf162(rhs + i * 2); \
            __nv_bfloat162 c = BF2_OP(a, b); \
            store_bf162(out + i * 2, c); \
        } \
        \
        if (vec_offset < numel && blockIdx.x == 0 && threadIdx.x == 0) { \
            out[vec_offset] = SCALAR_OP(lhs[vec_offset], rhs[vec_offset]); \
        } \
    } else if (lhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = SCALAR_OP(lhs[i], rhs[rhs_i]); \
        } \
    } else if (rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                lhs_i += i_dim * lhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = SCALAR_OP(lhs[lhs_i], rhs[i]); \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                lhs_i += i_dim * lhs_strides[d]; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            out[i] = SCALAR_OP(lhs[lhs_i], rhs[rhs_i]); \
        } \
    } \
}

// BFloat16 scalar ops
__device__ __forceinline__ __nv_bfloat16 bf_add(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }
__device__ __forceinline__ __nv_bfloat16 bf_sub(__nv_bfloat16 a, __nv_bfloat16 b) { return __hsub(a, b); }
__device__ __forceinline__ __nv_bfloat16 bf_mul(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }
__device__ __forceinline__ __nv_bfloat16 bf_div(__nv_bfloat16 a, __nv_bfloat16 b) { return __hdiv(a, b); }
__device__ __forceinline__ __nv_bfloat16 bf_max(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmax(a, b); }
__device__ __forceinline__ __nv_bfloat16 bf_min(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmin(a, b); }
