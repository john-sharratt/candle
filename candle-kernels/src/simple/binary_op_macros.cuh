#include "cuda_utils.cuh"
#include <stdint.h>

// =============================================================================
// OPTIMIZED BINARY ELEMENTWISE OPERATIONS
// =============================================================================
// Key optimizations:
//   1. __ldg() for read-only texture cache access (where supported)
//   2. __restrict__ pointer qualifiers for aliasing hints
//   3. unsigned int for numel (32-bit is sufficient and faster)
//   4. Cache dim_val in local register before division
//   5. Vectorized paths available below
// =============================================================================

// Standard macro with __ldg optimization
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

// Non-__ldg macro for types that don't support __ldg (like fp8)
#define BINARY_OP_OUT_NO_LDG(TYPENAME, OUT_TYPENAME, FN_NAME, FUNC) \
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
            TYPENAME x = lhs[i]; \
            TYPENAME y = rhs[i]; \
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
            TYPENAME x = lhs[i]; \
            TYPENAME y = rhs[rhs_i]; \
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
            TYPENAME x = lhs[lhs_i]; \
            TYPENAME y = rhs[i]; \
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
            TYPENAME x = lhs[lhs_i]; \
            TYPENAME y = rhs[rhs_i]; \
            out[i] = FUNC; \
        } \
    } \
}


#define BINARY_OP(TYPENAME, FN_NAME, FUNC) \
  BINARY_OP_OUT(TYPENAME, TYPENAME, FN_NAME, FUNC)

#define BINARY_OP_NO_LDG(TYPENAME, FN_NAME, FUNC) \
  BINARY_OP_OUT_NO_LDG(TYPENAME, TYPENAME, FN_NAME, FUNC)


// =============================================================================
// VECTORIZED FLOAT32 BINARY OPS - float4 (128-bit) loads for contiguous paths
// =============================================================================
// NOTE: float4 requires 16-byte alignment - we check all pointers

__device__ __forceinline__ float4 ldg_float4(const float* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const float4*>(ptr));
}

__device__ __forceinline__ void store_float4(float* __restrict__ ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

#define BINARY_OP_F32_VEC4(FN_NAME, OP) \
extern "C" __global__ void FN_NAME##_vec4( \
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
        /* Check 16-byte alignment for float4 vectorization */ \
        bool aligned = ((uintptr_t)lhs % 16 == 0) && ((uintptr_t)rhs % 16 == 0) && ((uintptr_t)out % 16 == 0); \
        \
        if (aligned && numel >= 4) { \
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
            for (unsigned int i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                out[i] = __ldg(lhs + i) OP __ldg(rhs + i); \
            } \
        } else { \
            /* Scalar fallback for unaligned data */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                out[i] = __ldg(lhs + i) OP __ldg(rhs + i); \
            } \
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

// Specialized division with __fdividef for float32
// NOTE: float4 requires 16-byte alignment - we check all pointers
#define BINARY_OP_F32_DIV_VEC4(FN_NAME) \
extern "C" __global__ void FN_NAME##_vec4( \
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
        /* Check 16-byte alignment for float4 vectorization */ \
        bool aligned = ((uintptr_t)lhs % 16 == 0) && ((uintptr_t)rhs % 16 == 0) && ((uintptr_t)out % 16 == 0); \
        \
        if (aligned && numel >= 4) { \
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
        } else { \
            /* Scalar fallback for unaligned data */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                out[i] = __fdividef(__ldg(lhs + i), __ldg(rhs + i)); \
            } \
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

// Min/max with fminf/fmaxf
// NOTE: float4 requires 16-byte alignment - we check all pointers
#define BINARY_OP_F32_MINMAX_VEC4(FN_NAME, MINMAX_FUNC) \
extern "C" __global__ void FN_NAME##_vec4( \
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
        /* Check 16-byte alignment for float4 vectorization */ \
        bool aligned = ((uintptr_t)lhs % 16 == 0) && ((uintptr_t)rhs % 16 == 0) && ((uintptr_t)out % 16 == 0); \
        \
        if (aligned && numel >= 4) { \
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
        } else { \
            /* Scalar fallback for unaligned data */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                out[i] = MINMAX_FUNC(__ldg(lhs + i), __ldg(rhs + i)); \
            } \
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
// VECTORIZED DOUBLE BINARY OPS - double2 (128-bit) loads
// =============================================================================

__device__ __forceinline__ double2 ldg_double2(const double* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const double2*>(ptr));
}

__device__ __forceinline__ void store_double2(double* __restrict__ ptr, double2 val) {
    *reinterpret_cast<double2*>(ptr) = val;
}

// NOTE: double2 requires 16-byte alignment - we check all pointers
#define BINARY_OP_F64_VEC2(FN_NAME, OP) \
extern "C" __global__ void FN_NAME##_vec2( \
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
        /* Check 16-byte alignment for double2 vectorization */ \
        bool aligned = ((uintptr_t)lhs % 16 == 0) && ((uintptr_t)rhs % 16 == 0) && ((uintptr_t)out % 16 == 0); \
        \
        if (aligned && numel >= 2) { \
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
        } else { \
            /* Scalar fallback for unaligned data */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                out[i] = __ldg(lhs + i) OP __ldg(rhs + i); \
            } \
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
// VECTORIZED HALF2 BINARY OPS - native half2 intrinsics (SM >= 5.3)
// =============================================================================

__device__ __forceinline__ __half2 ldg_half2(const __half* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const __half2*>(ptr));
}

__device__ __forceinline__ void store_half2(__half* __restrict__ ptr, __half2 val) {
    *reinterpret_cast<__half2*>(ptr) = val;
}

// Scalar helper functions for non-contiguous paths
__device__ __forceinline__ __half h_add(__half a, __half b) { return __hadd(a, b); }
__device__ __forceinline__ __half h_sub(__half a, __half b) { return __hsub(a, b); }
__device__ __forceinline__ __half h_mul(__half a, __half b) { return __hmul(a, b); }
__device__ __forceinline__ __half h_div(__half a, __half b) { return __hdiv(a, b); }
__device__ __forceinline__ __half h_max(__half a, __half b) { return __hmax(a, b); }
__device__ __forceinline__ __half h_min(__half a, __half b) { return __hmin(a, b); }

// NOTE: half2 requires 4-byte alignment - we check all pointers
#define BINARY_OP_F16_VEC2(FN_NAME, H2_OP, SCALAR_OP) \
extern "C" __global__ void FN_NAME##_vec2( \
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
        /* Check 4-byte alignment for half2 vectorization */ \
        bool aligned = ((uintptr_t)lhs % 4 == 0) && ((uintptr_t)rhs % 4 == 0) && ((uintptr_t)out % 4 == 0); \
        \
        if (aligned && numel >= 2) { \
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
            if (vec_offset < numel && blockIdx.x == 0 && threadIdx.x == 0) { \
                out[vec_offset] = SCALAR_OP(lhs[vec_offset], rhs[vec_offset]); \
            } \
        } else { \
            /* Scalar fallback for unaligned data */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                out[i] = SCALAR_OP(lhs[i], rhs[i]); \
            } \
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

// =============================================================================
// VECTORIZED F8E4M3 BINARY OPS - 4-byte (4 x fp8) loads
// =============================================================================
// FP8 is 1 byte, so we load 4 elements as uint32_t for better memory throughput.
// We unpack, convert to float, perform the op, convert back, and repack.
// NOTE: These use software FP8 conversions (F8E4M3_TO_FLOAT / __nv_fp8_e4m3(float))
// which work on all architectures (SM80+). No SM89 hardware intrinsics needed.

#ifndef F8E4M3_TO_FLOAT
#define F8E4M3_TO_FLOAT(x) __half2float(__nv_cvt_fp8_to_halfraw(x.__x, __NV_E4M3))
#endif

// Scalar helpers for f8e4m3
__device__ __forceinline__ __nv_fp8_e4m3 f8_add(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) { 
    return __nv_fp8_e4m3(F8E4M3_TO_FLOAT(a) + F8E4M3_TO_FLOAT(b)); 
}
__device__ __forceinline__ __nv_fp8_e4m3 f8_sub(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) { 
    return __nv_fp8_e4m3(F8E4M3_TO_FLOAT(a) - F8E4M3_TO_FLOAT(b)); 
}
__device__ __forceinline__ __nv_fp8_e4m3 f8_mul(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) { 
    return __nv_fp8_e4m3(F8E4M3_TO_FLOAT(a) * F8E4M3_TO_FLOAT(b)); 
}
__device__ __forceinline__ __nv_fp8_e4m3 f8_div(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) { 
    return __nv_fp8_e4m3(F8E4M3_TO_FLOAT(a) / F8E4M3_TO_FLOAT(b)); 
}
__device__ __forceinline__ __nv_fp8_e4m3 f8_max(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) { 
    return F8E4M3_TO_FLOAT(a) > F8E4M3_TO_FLOAT(b) ? a : b; 
}
__device__ __forceinline__ __nv_fp8_e4m3 f8_min(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) { 
    return F8E4M3_TO_FLOAT(a) < F8E4M3_TO_FLOAT(b) ? a : b; 
}

// NOTE: 4-byte alignment required for uint32 vectorization
#define BINARY_OP_F8E4M3_VEC4(FN_NAME, SCALAR_OP) \
extern "C" __global__ void FN_NAME##_vec4( \
    const unsigned int numel, \
    const size_t num_dims, \
    const size_t* __restrict__ dims_and_strides, \
    const __nv_fp8_e4m3* __restrict__ lhs, \
    const __nv_fp8_e4m3* __restrict__ rhs, \
    __nv_fp8_e4m3* __restrict__ out \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides); \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    \
    if (lhs_cont && rhs_cont) { \
        /* Check 4-byte alignment for uint32 vectorization (4 x fp8) */ \
        bool aligned = ((uintptr_t)lhs % 4 == 0) && ((uintptr_t)rhs % 4 == 0) && ((uintptr_t)out % 4 == 0); \
        \
        if (aligned && numel >= 4) { \
            const unsigned int vec_numel = numel / 4; \
            const unsigned int vec_offset = vec_numel * 4; \
            const uint32_t* lhs_vec = reinterpret_cast<const uint32_t*>(lhs); \
            const uint32_t* rhs_vec = reinterpret_cast<const uint32_t*>(rhs); \
            uint32_t* out_vec = reinterpret_cast<uint32_t*>(out); \
            \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < vec_numel; i += blockDim.x * gridDim.x) { \
                uint32_t a_packed = __ldg(lhs_vec + i); \
                uint32_t b_packed = __ldg(rhs_vec + i); \
                \
                /* Unpack 4 fp8 values */ \
                __nv_fp8_e4m3 a0, a1, a2, a3, b0, b1, b2, b3; \
                a0.__x = (a_packed >> 0) & 0xFF; \
                a1.__x = (a_packed >> 8) & 0xFF; \
                a2.__x = (a_packed >> 16) & 0xFF; \
                a3.__x = (a_packed >> 24) & 0xFF; \
                b0.__x = (b_packed >> 0) & 0xFF; \
                b1.__x = (b_packed >> 8) & 0xFF; \
                b2.__x = (b_packed >> 16) & 0xFF; \
                b3.__x = (b_packed >> 24) & 0xFF; \
                \
                /* Perform operation */ \
                __nv_fp8_e4m3 c0 = SCALAR_OP(a0, b0); \
                __nv_fp8_e4m3 c1 = SCALAR_OP(a1, b1); \
                __nv_fp8_e4m3 c2 = SCALAR_OP(a2, b2); \
                __nv_fp8_e4m3 c3 = SCALAR_OP(a3, b3); \
                \
                /* Repack and store */ \
                uint32_t c_packed = (uint32_t(c0.__x) << 0) | (uint32_t(c1.__x) << 8) | \
                                    (uint32_t(c2.__x) << 16) | (uint32_t(c3.__x) << 24); \
                out_vec[i] = c_packed; \
            } \
            \
            /* Handle remainder */ \
            for (unsigned int i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                out[i] = SCALAR_OP(lhs[i], rhs[i]); \
            } \
        } else { \
            /* Scalar fallback for unaligned data */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                out[i] = SCALAR_OP(lhs[i], rhs[i]); \
            } \
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

// =============================================================================
// VECTORIZED BFLOAT16X2 BINARY OPS (SM >= 8.0)
// =============================================================================

__device__ __forceinline__ __nv_bfloat162 ldg_bf162(const __nv_bfloat16* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const __nv_bfloat162*>(ptr));
}

__device__ __forceinline__ void store_bf162(__nv_bfloat16* __restrict__ ptr, __nv_bfloat162 val) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = val;
}

// Scalar helpers for bfloat16
__device__ __forceinline__ __nv_bfloat16 bf_add(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }
__device__ __forceinline__ __nv_bfloat16 bf_sub(__nv_bfloat16 a, __nv_bfloat16 b) { return __hsub(a, b); }
__device__ __forceinline__ __nv_bfloat16 bf_mul(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }
__device__ __forceinline__ __nv_bfloat16 bf_div(__nv_bfloat16 a, __nv_bfloat16 b) { return __hdiv(a, b); }
__device__ __forceinline__ __nv_bfloat16 bf_max(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmax(a, b); }
__device__ __forceinline__ __nv_bfloat16 bf_min(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmin(a, b); }

// NOTE: bfloat162 requires 4-byte alignment - we check all pointers
#define BINARY_OP_BF16_VEC2(FN_NAME, BF2_OP, SCALAR_OP) \
extern "C" __global__ void FN_NAME##_vec2( \
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
        /* Check 4-byte alignment for bfloat162 vectorization */ \
        bool aligned = ((uintptr_t)lhs % 4 == 0) && ((uintptr_t)rhs % 4 == 0) && ((uintptr_t)out % 4 == 0); \
        \
        if (aligned && numel >= 2) { \
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
        } else { \
            /* Scalar fallback for unaligned data */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                out[i] = SCALAR_OP(lhs[i], rhs[i]); \
            } \
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

// =============================================================================
// IN-PLACE BINARY OPERATIONS (lhs = lhs OP rhs)
// =============================================================================
// These operations modify the left-hand operand in-place, avoiding allocation.
// For in-place ops, lhs MUST be contiguous (we write back to it).
// =============================================================================

// Standard in-place macro - requires lhs to be contiguous
#define BINARY_OP_INPLACE(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const unsigned int numel, \
    const size_t num_dims, \
    const size_t* __restrict__ dims_and_strides, \
    TYPENAME* __restrict__ lhs, \
    const TYPENAME* __restrict__ rhs \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    \
    if (rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = lhs[i]; \
            TYPENAME y = __ldg(rhs + i); \
            lhs[i] = FUNC; \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            TYPENAME x = lhs[i]; \
            TYPENAME y = __ldg(rhs + rhs_i); \
            lhs[i] = FUNC; \
        } \
    } \
}

// Vectorized in-place f32 (float4) with preamble alignment
// Uses preamble approach: process unaligned elements first to reach alignment
#define BINARY_OP_INPLACE_F32_VEC4(FN_NAME, SCALAR_OP) \
extern "C" __global__ void FN_NAME##_vec4( \
    const unsigned int numel, \
    const size_t num_dims, \
    const size_t* __restrict__ dims_and_strides, \
    float* __restrict__ lhs, \
    const float* __restrict__ rhs \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    \
    if (rhs_cont) { \
        constexpr size_t VEC_ALIGN = 16;  /* float4 = 16 bytes */ \
        uintptr_t lhs_offset = reinterpret_cast<uintptr_t>(lhs) & (VEC_ALIGN - 1); \
        uintptr_t rhs_offset = reinterpret_cast<uintptr_t>(rhs) & (VEC_ALIGN - 1); \
        \
        /* If alignment offsets match, we can use preamble to reach alignment */ \
        if (lhs_offset == rhs_offset && numel >= 4) { \
            /* Calculate preamble: elements until we reach 16-byte alignment */ \
            size_t preamble = (lhs_offset == 0) ? 0 : (VEC_ALIGN - lhs_offset) / sizeof(float); \
            if (preamble > numel) preamble = numel; \
            \
            /* Process preamble (scalar, to reach alignment) */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < preamble; i += blockDim.x * gridDim.x) { \
                lhs[i] = SCALAR_OP(lhs[i], rhs[i]); \
            } \
            \
            /* Now process aligned section with float4 */ \
            float* aligned_lhs = lhs + preamble; \
            const float* aligned_rhs = rhs + preamble; \
            const size_t remaining = numel - preamble; \
            const size_t vec_numel = remaining / 4; \
            const size_t vec_offset = vec_numel * 4; \
            \
            float4* lhs4 = (float4*)aligned_lhs; \
            const float4* rhs4 = (const float4*)aligned_rhs; \
            \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < vec_numel; i += blockDim.x * gridDim.x) { \
                float4 a = lhs4[i]; \
                float4 b = __ldg(rhs4 + i); \
                float4 c; \
                c.x = SCALAR_OP(a.x, b.x); \
                c.y = SCALAR_OP(a.y, b.y); \
                c.z = SCALAR_OP(a.z, b.z); \
                c.w = SCALAR_OP(a.w, b.w); \
                lhs4[i] = c; \
            } \
            \
            /* Process tail (up to 3 elements) */ \
            const size_t tail_start = preamble + vec_offset; \
            for (unsigned int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                lhs[i] = SCALAR_OP(lhs[i], rhs[i]); \
            } \
        } else { \
            /* Alignment offsets differ or numel < 4 - use scalar */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                lhs[i] = SCALAR_OP(lhs[i], rhs[i]); \
            } \
        } \
    } else { \
        /* rhs not contiguous - strided access */ \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            lhs[i] = SCALAR_OP(lhs[i], rhs[rhs_i]); \
        } \
    } \
}

// Vectorized in-place f16 (half2) with preamble alignment
#define BINARY_OP_INPLACE_F16_VEC2(FN_NAME, H2_OP, SCALAR_OP) \
extern "C" __global__ void FN_NAME##_vec2( \
    const unsigned int numel, \
    const size_t num_dims, \
    const size_t* __restrict__ dims_and_strides, \
    __half* __restrict__ lhs, \
    const __half* __restrict__ rhs \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    \
    if (rhs_cont) { \
        constexpr size_t VEC_ALIGN = 4;  /* half2 = 4 bytes */ \
        uintptr_t lhs_offset = reinterpret_cast<uintptr_t>(lhs) & (VEC_ALIGN - 1); \
        uintptr_t rhs_offset = reinterpret_cast<uintptr_t>(rhs) & (VEC_ALIGN - 1); \
        \
        /* If alignment offsets match, we can use preamble to reach alignment */ \
        if (lhs_offset == rhs_offset && numel >= 2) { \
            /* Calculate preamble: elements until we reach 4-byte alignment */ \
            size_t preamble = (lhs_offset == 0) ? 0 : (VEC_ALIGN - lhs_offset) / sizeof(__half); \
            if (preamble > numel) preamble = numel; \
            \
            /* Process preamble (scalar, to reach alignment) */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < preamble; i += blockDim.x * gridDim.x) { \
                lhs[i] = SCALAR_OP(lhs[i], rhs[i]); \
            } \
            \
            /* Now process aligned section with half2 */ \
            __half* aligned_lhs = lhs + preamble; \
            const __half* aligned_rhs = rhs + preamble; \
            const size_t remaining = numel - preamble; \
            const size_t vec_numel = remaining / 2; \
            const size_t vec_offset = vec_numel * 2; \
            \
            __half2* lhs2 = (__half2*)aligned_lhs; \
            const __half2* rhs2 = (const __half2*)aligned_rhs; \
            \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < vec_numel; i += blockDim.x * gridDim.x) { \
                __half2 a = lhs2[i]; \
                __half2 b = __ldg(rhs2 + i); \
                lhs2[i] = H2_OP(a, b); \
            } \
            \
            /* Process tail (at most 1 element) */ \
            const size_t tail_start = preamble + vec_offset; \
            if (tail_start < numel && blockIdx.x == 0 && threadIdx.x == 0) { \
                lhs[tail_start] = SCALAR_OP(lhs[tail_start], rhs[tail_start]); \
            } \
        } else { \
            /* Alignment offsets differ or numel < 2 - use scalar */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                lhs[i] = SCALAR_OP(lhs[i], rhs[i]); \
            } \
        } \
    } else { \
        /* rhs not contiguous - strided access */ \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            lhs[i] = SCALAR_OP(lhs[i], rhs[rhs_i]); \
        } \
    } \
}

// Vectorized in-place bf16 (bfloat162) with preamble alignment
#define BINARY_OP_INPLACE_BF16_VEC2(FN_NAME, BF2_OP, SCALAR_OP) \
extern "C" __global__ void FN_NAME##_vec2( \
    const unsigned int numel, \
    const size_t num_dims, \
    const size_t* __restrict__ dims_and_strides, \
    __nv_bfloat16* __restrict__ lhs, \
    const __nv_bfloat16* __restrict__ rhs \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    \
    if (rhs_cont) { \
        constexpr size_t VEC_ALIGN = 4;  /* bfloat162 = 4 bytes */ \
        uintptr_t lhs_offset = reinterpret_cast<uintptr_t>(lhs) & (VEC_ALIGN - 1); \
        uintptr_t rhs_offset = reinterpret_cast<uintptr_t>(rhs) & (VEC_ALIGN - 1); \
        \
        /* If alignment offsets match, we can use preamble to reach alignment */ \
        if (lhs_offset == rhs_offset && numel >= 2) { \
            /* Calculate preamble: elements until we reach 4-byte alignment */ \
            size_t preamble = (lhs_offset == 0) ? 0 : (VEC_ALIGN - lhs_offset) / sizeof(__nv_bfloat16); \
            if (preamble > numel) preamble = numel; \
            \
            /* Process preamble (scalar, to reach alignment) */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < preamble; i += blockDim.x * gridDim.x) { \
                lhs[i] = SCALAR_OP(lhs[i], rhs[i]); \
            } \
            \
            /* Now process aligned section with bfloat162 */ \
            __nv_bfloat16* aligned_lhs = lhs + preamble; \
            const __nv_bfloat16* aligned_rhs = rhs + preamble; \
            const size_t remaining = numel - preamble; \
            const size_t vec_numel = remaining / 2; \
            const size_t vec_offset = vec_numel * 2; \
            \
            __nv_bfloat162* lhs2 = (__nv_bfloat162*)aligned_lhs; \
            const __nv_bfloat162* rhs2 = (const __nv_bfloat162*)aligned_rhs; \
            \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < vec_numel; i += blockDim.x * gridDim.x) { \
                __nv_bfloat162 a = lhs2[i]; \
                __nv_bfloat162 b = ldg_bf162((__nv_bfloat16*)(rhs2 + i)); \
                lhs2[i] = BF2_OP(a, b); \
            } \
            \
            /* Process tail (at most 1 element) */ \
            const size_t tail_start = preamble + vec_offset; \
            if (tail_start < numel && blockIdx.x == 0 && threadIdx.x == 0) { \
                lhs[tail_start] = SCALAR_OP(lhs[tail_start], rhs[tail_start]); \
            } \
        } else { \
            /* Alignment offsets differ or numel < 2 - use scalar */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                lhs[i] = SCALAR_OP(lhs[i], rhs[i]); \
            } \
        } \
    } else { \
        /* rhs not contiguous - strided access */ \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            lhs[i] = SCALAR_OP(lhs[i], rhs[rhs_i]); \
        } \
    } \
}

// Vectorized in-place f8_e4m3 (4 bytes at a time) with preamble alignment
#define BINARY_OP_INPLACE_F8E4M3_VEC4(FN_NAME, SCALAR_OP) \
extern "C" __global__ void FN_NAME##_vec4( \
    const unsigned int numel, \
    const size_t num_dims, \
    const size_t* __restrict__ dims_and_strides, \
    __nv_fp8_e4m3* __restrict__ lhs, \
    const __nv_fp8_e4m3* __restrict__ rhs \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides); \
    \
    if (rhs_cont) { \
        /* f8_e4m3 is 1 byte, uint32 is 4 bytes, so we need 4-byte alignment for vec4 */ \
        const size_t lhs_misalign = ((uintptr_t)lhs) % 4; \
        const size_t rhs_misalign = ((uintptr_t)rhs) % 4; \
        \
        /* Preamble alignment: if both have same misalignment, we can align them together */ \
        if (lhs_misalign == rhs_misalign && numel >= 4) { \
            /* Calculate preamble - elements to process before we reach alignment */ \
            const unsigned int preamble = (lhs_misalign == 0) ? 0 : (unsigned int)(4 - lhs_misalign); \
            \
            /* Process preamble elements (scalar) */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < preamble && i < numel; i += blockDim.x * gridDim.x) { \
                lhs[i] = SCALAR_OP(lhs[i], rhs[i]); \
            } \
            \
            /* Now pointers are aligned, process vectorized portion */ \
            const unsigned int remaining = numel - preamble; \
            const unsigned int vec_numel = remaining / 4; \
            \
            if (vec_numel > 0) { \
                uint32_t* lhs4 = (uint32_t*)(lhs + preamble); \
                const uint32_t* rhs4 = (const uint32_t*)(rhs + preamble); \
                \
                for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                     i < vec_numel; i += blockDim.x * gridDim.x) { \
                    uint32_t a_packed = lhs4[i]; \
                    uint32_t b_packed = rhs4[i]; \
                    __nv_fp8_e4m3 a0, a1, a2, a3, b0, b1, b2, b3; \
                    a0.__x = (a_packed) & 0xFF; \
                    a1.__x = (a_packed >> 8) & 0xFF; \
                    a2.__x = (a_packed >> 16) & 0xFF; \
                    a3.__x = (a_packed >> 24) & 0xFF; \
                    b0.__x = (b_packed) & 0xFF; \
                    b1.__x = (b_packed >> 8) & 0xFF; \
                    b2.__x = (b_packed >> 16) & 0xFF; \
                    b3.__x = (b_packed >> 24) & 0xFF; \
                    __nv_fp8_e4m3 c0 = SCALAR_OP(a0, b0); \
                    __nv_fp8_e4m3 c1 = SCALAR_OP(a1, b1); \
                    __nv_fp8_e4m3 c2 = SCALAR_OP(a2, b2); \
                    __nv_fp8_e4m3 c3 = SCALAR_OP(a3, b3); \
                    uint32_t c_packed = (uint32_t)c0.__x | ((uint32_t)c1.__x << 8) | \
                                       ((uint32_t)c2.__x << 16) | ((uint32_t)c3.__x << 24); \
                    lhs4[i] = c_packed; \
                } \
            } \
            \
            /* Process tail elements */ \
            const unsigned int tail_start = preamble + vec_numel * 4; \
            for (unsigned int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                lhs[i] = SCALAR_OP(lhs[i], rhs[i]); \
            } \
        } else { \
            /* Misalignment differs or numel too small - use scalar fallback */ \
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
                 i < numel; i += blockDim.x * gridDim.x) { \
                lhs[i] = SCALAR_OP(lhs[i], rhs[i]); \
            } \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int dim_val = dims[d]; \
                unsigned int i_dim = tmp_i % dim_val; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dim_val; \
            } \
            lhs[i] = SCALAR_OP(lhs[i], rhs[rhs_i]); \
        } \
    } \
}
