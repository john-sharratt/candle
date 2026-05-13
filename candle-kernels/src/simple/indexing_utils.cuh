#pragma once
// =============================================================================
// INDEXING UTILITIES - Shared helpers for index_select, gather, scatter, etc.
// =============================================================================
// Provides optimized primitives for index-based tensor operations:
//   - max_value<I>() for sentinel detection
//   - Vectorized loads/stores (vec_load, vec_store)
//   - Read-only cache hints via __ldg
//   - Integer division elimination helpers
// =============================================================================

#include "cuda_utils.cuh"
#include <stdint.h>

// =============================================================================
// MAX VALUE SENTINELS - Used to detect "skip" indices (e.g., padding)
// =============================================================================
// These sentinels mark indices that should be zeroed rather than gathered.
// Useful for batched operations where sequences have different lengths.

template <typename T>
__host__ __device__ __forceinline__
constexpr T max_value();

template <>
__host__ __device__ __forceinline__
constexpr int64_t max_value<int64_t>() {
    return 0x7FFFFFFFFFFFFFFFLL;
}

template <>
__host__ __device__ __forceinline__
constexpr uint32_t max_value<uint32_t>() {
    return 0xFFFFFFFFu;
}

template <>
__host__ __device__ __forceinline__
constexpr uint8_t max_value<uint8_t>() {
    return 0xFFu;
}

template <>
__host__ __device__ __forceinline__
constexpr int32_t max_value<int32_t>() {
    return 0x7FFFFFFF;
}

template <>
__host__ __device__ __forceinline__
constexpr int16_t max_value<int16_t>() {
    return 0x7FFF;
}

// =============================================================================
// VECTORIZED MEMORY ACCESS TRAITS
// =============================================================================
// Maps scalar types to their optimal vector types for coalesced memory access.
// Using vec4 (4 elements) provides 4x memory bandwidth improvement.

template <typename T>
struct vec_traits {
    // Default: scalar access (no vectorization)
    static constexpr int vec_size = 1;
    using vec_type = T;
};

template <>
struct vec_traits<float> {
    static constexpr int vec_size = 4;
    using vec_type = float4;
};

template <>
struct vec_traits<double> {
    static constexpr int vec_size = 2;
    using vec_type = double2;
};

template <>
struct vec_traits<uint32_t> {
    static constexpr int vec_size = 4;
    using vec_type = uint4;
};

template <>
struct vec_traits<int64_t> {
    static constexpr int vec_size = 2;
    using vec_type = longlong2;
};

// 16-bit types: use 4 elements (8 bytes = optimal for most GPUs)
template <>
struct vec_traits<__half> {
    static constexpr int vec_size = 4;
    using vec_type = ushort4;  // 4x __half packed as ushort4
};

template <>
struct vec_traits<__nv_bfloat16> {
    static constexpr int vec_size = 4;
    using vec_type = ushort4;  // 4x bf16 packed as ushort4
};

// =============================================================================
// READ-ONLY CACHE LOADS (__ldg)
// =============================================================================
// __ldg() uses the texture cache path, which is faster for read-only data
// that exhibits spatial locality (like embedding tables).
// Note: __ldg is natively supported for int, uint, int2, uint2, int4, uint4,
// long long, unsigned long long, float, float2, float4, double, double2.
// Smaller types use direct loads (they're typically used for indices, not bulk data).

template <typename T>
__device__ __forceinline__ T ldg_load(const T* __restrict__ ptr) {
    return __ldg(ptr);
}

// Specializations for types that don't support __ldg directly
// These use regular loads - small types are typically index values, not bulk data
template <>
__device__ __forceinline__ uint8_t ldg_load<uint8_t>(const uint8_t* __restrict__ ptr) {
    return *ptr;  // Direct load - uint8_t not supported by __ldg
}

template <>
__device__ __forceinline__ int8_t ldg_load<int8_t>(const int8_t* __restrict__ ptr) {
    return *ptr;  // Direct load
}

template <>
__device__ __forceinline__ int16_t ldg_load<int16_t>(const int16_t* __restrict__ ptr) {
    return *ptr;  // Direct load
}

template <>
__device__ __forceinline__ uint16_t ldg_load<uint16_t>(const uint16_t* __restrict__ ptr) {
    return *ptr;  // Direct load
}

// Half-precision types (16-bit)
template <>
__device__ __forceinline__ __half ldg_load<__half>(const __half* __restrict__ ptr) {
    return *ptr;  // Direct load for 16-bit types
}

template <>
__device__ __forceinline__ __nv_bfloat16 ldg_load<__nv_bfloat16>(const __nv_bfloat16* __restrict__ ptr) {
    return *ptr;  // Direct load for 16-bit types
}

// FP8 types (8-bit)
template <>
__device__ __forceinline__ __nv_fp8_e4m3 ldg_load<__nv_fp8_e4m3>(const __nv_fp8_e4m3* __restrict__ ptr) {
    return *ptr;  // Direct load for 8-bit types
}

template <>
__device__ __forceinline__ __nv_fp8_e5m2 ldg_load<__nv_fp8_e5m2>(const __nv_fp8_e5m2* __restrict__ ptr) {
    return *ptr;  // Direct load for 8-bit types
}

// Vector loads through __ldg
__device__ __forceinline__ float4 ldg_load_float4(const float* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const float4*>(ptr));
}

__device__ __forceinline__ double2 ldg_load_double2(const double* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const double2*>(ptr));
}

__device__ __forceinline__ uint4 ldg_load_uint4(const uint32_t* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const uint4*>(ptr));
}

__device__ __forceinline__ longlong2 ldg_load_longlong2(const int64_t* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const longlong2*>(ptr));
}

// =============================================================================
// VECTORIZED STORE HELPERS
// =============================================================================

__device__ __forceinline__ void vec_store_float4(float* __restrict__ ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__device__ __forceinline__ void vec_store_double2(double* __restrict__ ptr, double2 val) {
    *reinterpret_cast<double2*>(ptr) = val;
}

__device__ __forceinline__ void vec_store_uint4(uint32_t* __restrict__ ptr, uint4 val) {
    *reinterpret_cast<uint4*>(ptr) = val;
}

__device__ __forceinline__ void vec_store_longlong2(int64_t* __restrict__ ptr, longlong2 val) {
    *reinterpret_cast<longlong2*>(ptr) = val;
}

// =============================================================================
// FAST INDEX DECOMPOSITION
// =============================================================================
// For index_select with layout: [left, ids_dim, right]
// We need to decompose flat index dst_i into (left_i, id_i, right_i).
// 
// Instead of expensive division/modulo each iteration, we precompute
// multipliers and use multiplication + bit shifts where possible.

// Structure to hold precomputed decomposition factors
struct IndexDecomposer {
    unsigned int ids_times_right;  // ids_dim_size * right_size
    unsigned int right_size;
    
    __device__ __forceinline__ void init(unsigned int ids_dim, unsigned int right) {
        ids_times_right = ids_dim * right;
        right_size = right;
    }
    
    // Decompose flat index into 3D coordinates
    // Returns: left_i, id_i, right_i via output parameters
    __device__ __forceinline__ void decompose(
        unsigned int dst_i,
        unsigned int& left_i,
        unsigned int& id_i, 
        unsigned int& right_i
    ) const {
        left_i = dst_i / ids_times_right;
        unsigned int remainder = dst_i - left_i * ids_times_right;
        id_i = remainder / right_size;
        right_i = remainder - id_i * right_size;
    }
};

// Specialized decomposer for common case: right_size is power of 2
// Uses bit shifts instead of division (5x faster)
template <unsigned int LOG2_RIGHT>
struct IndexDecomposerPow2 {
    unsigned int ids_times_right;
    static constexpr unsigned int RIGHT_MASK = (1u << LOG2_RIGHT) - 1;
    
    __device__ __forceinline__ void init(unsigned int ids_dim) {
        ids_times_right = ids_dim << LOG2_RIGHT;
    }
    
    __device__ __forceinline__ void decompose(
        unsigned int dst_i,
        unsigned int& left_i,
        unsigned int& id_i,
        unsigned int& right_i
    ) const {
        left_i = dst_i / ids_times_right;
        unsigned int remainder = dst_i - left_i * ids_times_right;
        id_i = remainder >> LOG2_RIGHT;
        right_i = remainder & RIGHT_MASK;
    }
};

// =============================================================================
// ROW-BASED INDEX SELECT
// =============================================================================
// For embedding lookups, the common pattern is:
//   - 1D index array of shape [num_indices]
//   - 2D embedding table of shape [vocab_size, embed_dim]
//   - Output shape [num_indices, embed_dim]
//
// This is a special case where left_size=1 and we process one row per index.
// Thread assignment: each thread handles multiple elements within a row.

template <typename T, typename I>
__device__ __forceinline__ void index_select_row(
    const I idx,                          // The index value
    const T* __restrict__ inp,            // Input embedding table
    T* __restrict__ out_row,              // Output row pointer
    const unsigned int embed_dim,         // Embedding dimension (right_size)
    const unsigned int src_dim_size       // Vocab size
) {
    // Handle sentinel (padding) indices
    if (idx == max_value<I>()) {
        // Zero-fill the output row
        for (unsigned int e = threadIdx.x; e < embed_dim; e += blockDim.x) {
            out_row[e] = static_cast<T>(0);
        }
        return;
    }
    
    // Copy embedding row to output
    const T* __restrict__ src_row = inp + static_cast<size_t>(idx) * embed_dim;
    
    #pragma unroll 4
    for (unsigned int e = threadIdx.x; e < embed_dim; e += blockDim.x) {
        out_row[e] = ldg_load(src_row + e);
    }
}

// Vectorized version for float (4x throughput)
template <>
__device__ __forceinline__ void index_select_row<float, int64_t>(
    const int64_t idx,
    const float* __restrict__ inp,
    float* __restrict__ out_row,
    const unsigned int embed_dim,
    const unsigned int src_dim_size
) {
    if (idx == max_value<int64_t>()) {
        const float4 zero4 = make_float4(0.f, 0.f, 0.f, 0.f);
        unsigned int vec_dim = embed_dim / 4;
        for (unsigned int e = threadIdx.x; e < vec_dim; e += blockDim.x) {
            vec_store_float4(out_row + e * 4, zero4);
        }
        // Handle tail elements
        for (unsigned int e = vec_dim * 4 + threadIdx.x; e < embed_dim; e += blockDim.x) {
            out_row[e] = 0.f;
        }
        return;
    }
    
    const float* __restrict__ src_row = inp + idx * embed_dim;
    unsigned int vec_dim = embed_dim / 4;
    
    #pragma unroll 2
    for (unsigned int e = threadIdx.x; e < vec_dim; e += blockDim.x) {
        float4 val = ldg_load_float4(src_row + e * 4);
        vec_store_float4(out_row + e * 4, val);
    }
    // Handle tail elements
    for (unsigned int e = vec_dim * 4 + threadIdx.x; e < embed_dim; e += blockDim.x) {
        out_row[e] = ldg_load(src_row + e);
    }
}

template <>
__device__ __forceinline__ void index_select_row<float, uint32_t>(
    const uint32_t idx,
    const float* __restrict__ inp,
    float* __restrict__ out_row,
    const unsigned int embed_dim,
    const unsigned int src_dim_size
) {
    if (idx == max_value<uint32_t>()) {
        const float4 zero4 = make_float4(0.f, 0.f, 0.f, 0.f);
        unsigned int vec_dim = embed_dim / 4;
        for (unsigned int e = threadIdx.x; e < vec_dim; e += blockDim.x) {
            vec_store_float4(out_row + e * 4, zero4);
        }
        for (unsigned int e = vec_dim * 4 + threadIdx.x; e < embed_dim; e += blockDim.x) {
            out_row[e] = 0.f;
        }
        return;
    }
    
    const float* __restrict__ src_row = inp + static_cast<size_t>(idx) * embed_dim;
    unsigned int vec_dim = embed_dim / 4;
    
    #pragma unroll 2
    for (unsigned int e = threadIdx.x; e < vec_dim; e += blockDim.x) {
        float4 val = ldg_load_float4(src_row + e * 4);
        vec_store_float4(out_row + e * 4, val);
    }
    for (unsigned int e = vec_dim * 4 + threadIdx.x; e < embed_dim; e += blockDim.x) {
        out_row[e] = ldg_load(src_row + e);
    }
}

// =============================================================================
// SHARED MEMORY INDEX CACHE
// =============================================================================
// When multiple threads in a block access the same index, loading it once
// into shared memory reduces global memory traffic.

#define INDICES_SMEM_SIZE 256  // Max indices to cache in shared memory

template <typename I>
__device__ __forceinline__ void load_indices_to_smem(
    const I* __restrict__ ids,
    I* __restrict__ smem_ids,
    const unsigned int ids_dim_size,
    const unsigned int max_to_load
) {
    unsigned int to_load = min(ids_dim_size, max_to_load);
    for (unsigned int i = threadIdx.x; i < to_load; i += blockDim.x) {
        smem_ids[i] = ldg_load(ids + i);
    }
    __syncthreads();
}

// =============================================================================
// CONTIGUITY CHECK RESULTS
// =============================================================================
// Moving is_contiguous check outside the kernel and passing result as template
// parameter eliminates branch divergence within the kernel.

enum class Contiguity : int {
    CONTIGUOUS = 0,
    NON_CONTIGUOUS = 1
};
