#include "cuda_utils.cuh"
#include<stdint.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// =============================================================================
// Alignment helpers
// =============================================================================

template<size_t ALIGN>
__device__ __forceinline__ bool is_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & (ALIGN - 1)) == 0;
}

// Returns number of ELEMENTS (not bytes) needed to reach alignment
template<typename T, size_t ALIGN>
__device__ __forceinline__ size_t elements_to_align(const T* ptr) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t misalign = addr & (ALIGN - 1);
    if (misalign == 0) return 0;
    return (ALIGN - misalign) / sizeof(T);
}

// Check if two pointers have the same alignment offset (for preamble approach)
template<size_t ALIGN>
__device__ __forceinline__ bool same_alignment_offset(const void* a, const void* b) {
    uintptr_t offset_a = reinterpret_cast<uintptr_t>(a) & (ALIGN - 1);
    uintptr_t offset_b = reinterpret_cast<uintptr_t>(b) & (ALIGN - 1);
    return offset_a == offset_b;
}

// =============================================================================
// Type conversion traits - specialize for non-trivial conversions
// =============================================================================

// Default: use implicit C++ conversion
template<typename S, typename T>
struct Convert {
    __device__ __forceinline__ static T apply(S val) {
        return static_cast<T>(val);
    }
};

// Same-type optimization (identity)
template<typename T>
struct Convert<T, T> {
    __device__ __forceinline__ static T apply(T val) {
        return val;
    }
};

// FP8 E4M3 -> any type (through float)
template<typename T>
struct Convert<__nv_fp8_e4m3, T> {
    __device__ __forceinline__ static T apply(__nv_fp8_e4m3 val) {
        float f = __half2float(__nv_cvt_fp8_to_halfraw(val.__x, __NV_E4M3));
        return static_cast<T>(f);
    }
};

// FP8 -> FP8 (identity, override the above)
template<>
struct Convert<__nv_fp8_e4m3, __nv_fp8_e4m3> {
    __device__ __forceinline__ static __nv_fp8_e4m3 apply(__nv_fp8_e4m3 val) {
        return val;
    }
};

// Any type -> FP8 E4M3 (through float)
template<typename S>
struct Convert<S, __nv_fp8_e4m3> {
    __device__ __forceinline__ static __nv_fp8_e4m3 apply(S val) {
        return __nv_fp8_e4m3(static_cast<float>(val));
    }
};

// BF16 <-> F16 need intermediate float on SM80+
template<>
struct Convert<__nv_bfloat16, __half> {
    __device__ __forceinline__ static __half apply(__nv_bfloat16 val) {
        return __float2half(__bfloat162float(val));
    }
};

template<>
struct Convert<__half, __nv_bfloat16> {
    __device__ __forceinline__ static __nv_bfloat16 apply(__half val) {
        return __float2bfloat16(__half2float(val));
    }
};

// BF16 -> uint8_t needs intermediate float
template<>
struct Convert<__nv_bfloat16, uint8_t> {
    __device__ __forceinline__ static uint8_t apply(__nv_bfloat16 val) {
        return static_cast<uint8_t>(__bfloat162float(val));
    }
};

// F16 -> uint8_t needs intermediate float
template<>
struct Convert<__half, uint8_t> {
    __device__ __forceinline__ static uint8_t apply(__half val) {
        return static_cast<uint8_t>(__half2float(val));
    }
};

// =============================================================================
// Packed conversion traits (2 elements at once) for vectorized operations
// =============================================================================

// half2 packed conversions
template<>
struct Convert<__half2, __half2> {
    __device__ __forceinline__ static __half2 apply(__half2 val) { return val; }
};

template<>
struct Convert<float2, __half2> {
    __device__ __forceinline__ static __half2 apply(float2 val) {
        return __float22half2_rn(val);
    }
};

template<>
struct Convert<__half2, float2> {
    __device__ __forceinline__ static float2 apply(__half2 val) {
        return __half22float2(val);
    }
};

// bfloat162 packed conversions
template<>
struct Convert<__nv_bfloat162, __nv_bfloat162> {
    __device__ __forceinline__ static __nv_bfloat162 apply(__nv_bfloat162 val) { return val; }
};

template<>
struct Convert<float2, __nv_bfloat162> {
    __device__ __forceinline__ static __nv_bfloat162 apply(float2 val) {
        return __float22bfloat162_rn(val);
    }
};

template<>
struct Convert<__nv_bfloat162, float2> {
    __device__ __forceinline__ static float2 apply(__nv_bfloat162 val) {
        return __bfloat1622float2(val);
    }
};

// half2 <-> bfloat162 through float2
template<>
struct Convert<__half2, __nv_bfloat162> {
    __device__ __forceinline__ static __nv_bfloat162 apply(__half2 val) {
        float2 f = __half22float2(val);
        return __float22bfloat162_rn(f);
    }
};

template<>
struct Convert<__nv_bfloat162, __half2> {
    __device__ __forceinline__ static __half2 apply(__nv_bfloat162 val) {
        float2 f = __bfloat1622float2(val);
        return __float22half2_rn(f);
    }
};

// =============================================================================
// Vector type mapping: scalar -> packed type
// =============================================================================

template<typename T> struct Vec2Type { using type = void; };
template<typename T> struct Vec4Type { using type = void; };

template<> struct Vec2Type<float> { using type = float2; };
template<> struct Vec4Type<float> { using type = float4; };
template<> struct Vec2Type<double> { using type = double2; };
template<> struct Vec4Type<double> { using type = double4; };
template<> struct Vec2Type<uint32_t> { using type = uint2; };
template<> struct Vec4Type<uint32_t> { using type = uint4; };
template<> struct Vec2Type<int32_t> { using type = int2; };
template<> struct Vec4Type<int32_t> { using type = int4; };
template<> struct Vec2Type<int64_t> { using type = longlong2; };
template<> struct Vec4Type<int64_t> { using type = longlong4; };
template<> struct Vec2Type<uint8_t> { using type = uchar2; };
template<> struct Vec4Type<uint8_t> { using type = uchar4; };

template<> struct Vec2Type<__half> { using type = __half2; };
template<> struct Vec2Type<__nv_bfloat16> { using type = __nv_bfloat162; };

// =============================================================================
// Vec2 conversion traits - encapsulates vectorized conversion parameters
// =============================================================================

template<typename S, typename T>
struct Vec2Traits {
    static constexpr bool HAS_VEC2 = false;
};

// half <-> float
template<>
struct Vec2Traits<__half, float> {
    using SrcVec = __half2;
    using DstVec = float2;
    static constexpr bool HAS_VEC2 = true;
    static constexpr size_t SRC_ALIGN = 4;   // half2 alignment
    static constexpr size_t DST_ALIGN = 8;   // float2 alignment
    static constexpr bool SAME_SIZE = false;
};

template<>
struct Vec2Traits<float, __half> {
    using SrcVec = float2;
    using DstVec = __half2;
    static constexpr bool HAS_VEC2 = true;
    static constexpr size_t SRC_ALIGN = 8;
    static constexpr size_t DST_ALIGN = 4;
    static constexpr bool SAME_SIZE = false;
};

// bfloat16 <-> float
template<>
struct Vec2Traits<__nv_bfloat16, float> {
    using SrcVec = __nv_bfloat162;
    using DstVec = float2;
    static constexpr bool HAS_VEC2 = true;
    static constexpr size_t SRC_ALIGN = 4;
    static constexpr size_t DST_ALIGN = 8;
    static constexpr bool SAME_SIZE = false;
};

template<>
struct Vec2Traits<float, __nv_bfloat16> {
    using SrcVec = float2;
    using DstVec = __nv_bfloat162;
    static constexpr bool HAS_VEC2 = true;
    static constexpr size_t SRC_ALIGN = 8;
    static constexpr size_t DST_ALIGN = 4;
    static constexpr bool SAME_SIZE = false;
};

// bfloat16 <-> half (same size, can use preamble)
template<>
struct Vec2Traits<__nv_bfloat16, __half> {
    using SrcVec = __nv_bfloat162;
    using DstVec = __half2;
    static constexpr bool HAS_VEC2 = true;
    static constexpr size_t SRC_ALIGN = 4;
    static constexpr size_t DST_ALIGN = 4;
    static constexpr bool SAME_SIZE = true;
};

template<>
struct Vec2Traits<__half, __nv_bfloat16> {
    using SrcVec = __half2;
    using DstVec = __nv_bfloat162;
    static constexpr bool HAS_VEC2 = true;
    static constexpr size_t SRC_ALIGN = 4;
    static constexpr size_t DST_ALIGN = 4;
    static constexpr bool SAME_SIZE = true;
};

// double <-> float (different sizes: 8 bytes vs 4 bytes)
template<>
struct Vec2Traits<double, float> {
    using SrcVec = double2;
    using DstVec = float2;
    static constexpr bool HAS_VEC2 = true;
    static constexpr size_t SRC_ALIGN = 16;  // double2 alignment
    static constexpr size_t DST_ALIGN = 8;   // float2 alignment
    static constexpr bool SAME_SIZE = false;
};

template<>
struct Vec2Traits<float, double> {
    using SrcVec = float2;
    using DstVec = double2;
    static constexpr bool HAS_VEC2 = true;
    static constexpr size_t SRC_ALIGN = 8;
    static constexpr size_t DST_ALIGN = 16;
    static constexpr bool SAME_SIZE = false;
};

// Convert specializations for double2 <-> float2
template<>
struct Convert<double2, float2> {
    __device__ __forceinline__ static float2 apply(double2 val) {
        return make_float2(static_cast<float>(val.x), static_cast<float>(val.y));
    }
};

template<>
struct Convert<float2, double2> {
    __device__ __forceinline__ static double2 apply(float2 val) {
        return make_double2(static_cast<double>(val.x), static_cast<double>(val.y));
    }
};

// =============================================================================
// Read-only load wrapper - uses __ldg() for texture cache on supported archs
// =============================================================================

template<typename T>
__device__ __forceinline__ T load_readonly(const T* ptr) {
    // Conservative default: always valid for any type.
    return *ptr;
}

// Fast-path overloads for common scalar types (safe across toolkits).
__device__ __forceinline__ float load_readonly(const float* ptr) {
    return __ldg(ptr);
}

__device__ __forceinline__ double load_readonly(const double* ptr) {
    return __ldg(ptr);
}

__device__ __forceinline__ uint8_t load_readonly(const uint8_t* ptr) {
    return __ldg(ptr);
}

__device__ __forceinline__ uint32_t load_readonly(const uint32_t* ptr) {
    return __ldg(ptr);
}

__device__ __forceinline__ int32_t load_readonly(const int32_t* ptr) {
    return __ldg(ptr);
}

__device__ __forceinline__ int64_t load_readonly(const int64_t* ptr) {
    return __ldg(ptr);
}

// FP8 overload relies on the project-provided __ldg wrapper in cuda_utils.cuh.
__device__ __forceinline__ __nv_fp8_e4m3 load_readonly(const __nv_fp8_e4m3* ptr) {
    return __ldg(ptr);
}

// =============================================================================
// Contiguous kernel - vectorized with loop unrolling (Optimization 1, 3, 4)
// =============================================================================

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_contiguous(
    const size_t numel,
    const S * __restrict__ inp,
    T * __restrict__ out
) {
    constexpr int UNROLL = 4;
    const unsigned int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned int grid_stride = BLOCK_SIZE * gridDim.x;
    
    // Main unrolled loop
    unsigned int i = tid;
    const unsigned int unroll_limit = (numel >= UNROLL * grid_stride) ? numel - (UNROLL - 1) * grid_stride : 0;
    
    while (i < unroll_limit) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            out[i + u * grid_stride] = Convert<S, T>::apply(load_readonly(&inp[i + u * grid_stride]));
        }
        i += UNROLL * grid_stride;
    }
    
    // Remainder
    for (; i < numel; i += grid_stride) {
        out[i] = Convert<S, T>::apply(load_readonly(&inp[i]));
    }
}

// Vectorized float4 specialization for same-size 32-bit types
// Uses preamble to reach alignment if both pointers have same offset
template<typename T, int BLOCK_SIZE>
__device__ void cast_contiguous_vec4_copy(
    const size_t numel,
    const T * __restrict__ inp,
    T * __restrict__ out
) {
    constexpr int UNROLL = 4;
    const unsigned int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned int grid_stride = BLOCK_SIZE * gridDim.x;
    
    // Check if we can use vectorized path
    // For float4: need 16-byte alignment, T is 4 bytes, so need offset % 4 == 0
    constexpr size_t VEC_ALIGN = 16;  // float4 alignment
    
    // If pointers have different alignment offsets, fall back to scalar
    if (!same_alignment_offset<VEC_ALIGN>(inp, out)) {
        // Scalar fallback with unrolling
        unsigned int i = tid;
        const unsigned int unroll_limit = (numel >= UNROLL * grid_stride) ? numel - (UNROLL - 1) * grid_stride : 0;
        while (i < unroll_limit) {
            #pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                out[i + u * grid_stride] = inp[i + u * grid_stride];
            }
            i += UNROLL * grid_stride;
        }
        for (; i < numel; i += grid_stride) {
            out[i] = inp[i];
        }
        return;
    }
    
    // Calculate preamble: elements until we reach alignment
    size_t preamble = elements_to_align<T, VEC_ALIGN>(inp);
    if (preamble > numel) preamble = numel;
    
    // Process preamble (scalar, to reach alignment)
    for (unsigned int i = tid; i < preamble; i += grid_stride) {
        out[i] = inp[i];
    }
    
    // Now both pointers are aligned - process with float4 and unrolling
    const T* aligned_inp = inp + preamble;
    T* aligned_out = out + preamble;
    const size_t remaining = numel - preamble;
    const size_t numel4 = remaining / 4;
    
    const float4* __restrict__ inp4 = reinterpret_cast<const float4*>(aligned_inp);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(aligned_out);
    
    // Unrolled main loop
    unsigned int i = tid;
    const unsigned int unroll_limit = (numel4 >= UNROLL * grid_stride) ? numel4 - (UNROLL - 1) * grid_stride : 0;
    
    while (i < unroll_limit) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            out4[i + u * grid_stride] = inp4[i + u * grid_stride];
        }
        i += UNROLL * grid_stride;
    }
    
    // Remainder of vec4 elements
    for (; i < numel4; i += grid_stride) {
        out4[i] = inp4[i];
    }
    
    // Remainder (up to 3 elements)
    const size_t remainder_start = preamble + numel4 * 4;
    for (unsigned int i = remainder_start + tid; i < numel; i += grid_stride) {
        out[i] = inp[i];
    }
}

// =============================================================================
// Unified Vec2 conversion kernel - handles all vec2-optimizable type pairs
// Uses traits to select alignment strategy at compile time
// =============================================================================

// Helper: Vec2 kernel for different-size types (alignment check + fallback)
template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_contiguous_vec2_different_size(
    const size_t numel,
    const S * __restrict__ inp,
    T * __restrict__ out
) {
    using Traits = Vec2Traits<S, T>;
    using SrcVec = typename Traits::SrcVec;
    using DstVec = typename Traits::DstVec;
    
    constexpr int UNROLL = 4;
    const unsigned int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned int grid_stride = BLOCK_SIZE * gridDim.x;
    
    // Check alignment - different alignments for different-size types
    const bool inp_aligned = is_aligned<Traits::SRC_ALIGN>(inp);
    const bool out_aligned = is_aligned<Traits::DST_ALIGN>(out);
    
    if (!inp_aligned || !out_aligned) {
        // Scalar fallback with unrolling
        unsigned int i = tid;
        const unsigned int unroll_limit = (numel >= UNROLL * grid_stride) ? numel - (UNROLL - 1) * grid_stride : 0;
        while (i < unroll_limit) {
            #pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                out[i + u * grid_stride] = Convert<S, T>::apply(load_readonly(&inp[i + u * grid_stride]));
            }
            i += UNROLL * grid_stride;
        }
        for (; i < numel; i += grid_stride) {
            out[i] = Convert<S, T>::apply(load_readonly(&inp[i]));
        }
        return;
    }
    
    const size_t numel2 = numel / 2;
    const SrcVec* __restrict__ inp2 = reinterpret_cast<const SrcVec*>(inp);
    DstVec* __restrict__ out2 = reinterpret_cast<DstVec*>(out);
    
    // Unrolled main loop
    unsigned int i = tid;
    const unsigned int unroll_limit = (numel2 >= UNROLL * grid_stride) ? numel2 - (UNROLL - 1) * grid_stride : 0;
    
    while (i < unroll_limit) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            out2[i + u * grid_stride] = Convert<SrcVec, DstVec>::apply(inp2[i + u * grid_stride]);
        }
        i += UNROLL * grid_stride;
    }
    
    // Remainder of vec2 elements
    for (; i < numel2; i += grid_stride) {
        out2[i] = Convert<SrcVec, DstVec>::apply(inp2[i]);
    }
    
    // Handle odd element - only one thread across all blocks
    if (numel % 2 && blockIdx.x == 0 && threadIdx.x == 0) {
        out[numel - 1] = Convert<S, T>::apply(load_readonly(&inp[numel - 1]));
    }
}

// Helper: Vec2 kernel for same-size types (preamble approach)
template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_contiguous_vec2_same_size(
    const size_t numel,
    const S * __restrict__ inp,
    T * __restrict__ out
) {
    using Traits = Vec2Traits<S, T>;
    using SrcVec = typename Traits::SrcVec;
    using DstVec = typename Traits::DstVec;
    
    constexpr int UNROLL = 4;
    constexpr size_t VEC_ALIGN = Traits::SRC_ALIGN;  // Same for both when SAME_SIZE
    const unsigned int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned int grid_stride = BLOCK_SIZE * gridDim.x;
    
    // Same-size types can use preamble if alignment offsets match
    if (!same_alignment_offset<VEC_ALIGN>(inp, out)) {
        // Scalar fallback with unrolling
        unsigned int i = tid;
        const unsigned int unroll_limit = (numel >= UNROLL * grid_stride) ? numel - (UNROLL - 1) * grid_stride : 0;
        while (i < unroll_limit) {
            #pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                out[i + u * grid_stride] = Convert<S, T>::apply(load_readonly(&inp[i + u * grid_stride]));
            }
            i += UNROLL * grid_stride;
        }
        for (; i < numel; i += grid_stride) {
            out[i] = Convert<S, T>::apply(load_readonly(&inp[i]));
        }
        return;
    }
    
    // Calculate preamble to reach alignment
    size_t preamble = elements_to_align<S, VEC_ALIGN>(inp);
    if (preamble > numel) preamble = numel;
    
    // Process preamble (scalar)
    for (unsigned int i = tid; i < preamble; i += grid_stride) {
        out[i] = Convert<S, T>::apply(load_readonly(&inp[i]));
    }
    
    // Now both are aligned
    const S* aligned_inp = inp + preamble;
    T* aligned_out = out + preamble;
    const size_t remaining = numel - preamble;
    const size_t numel2 = remaining / 2;
    
    const SrcVec* __restrict__ inp2 = reinterpret_cast<const SrcVec*>(aligned_inp);
    DstVec* __restrict__ out2 = reinterpret_cast<DstVec*>(aligned_out);
    
    // Unrolled main loop
    unsigned int i = tid;
    const unsigned int unroll_limit = (numel2 >= UNROLL * grid_stride) ? numel2 - (UNROLL - 1) * grid_stride : 0;
    
    while (i < unroll_limit) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            out2[i + u * grid_stride] = Convert<SrcVec, DstVec>::apply(inp2[i + u * grid_stride]);
        }
        i += UNROLL * grid_stride;
    }
    
    // Remainder of vec2 elements
    for (; i < numel2; i += grid_stride) {
        out2[i] = Convert<SrcVec, DstVec>::apply(inp2[i]);
    }
    
    // Handle remaining elements after vec2 processing
    const size_t vec_processed = preamble + numel2 * 2;
    for (unsigned int i = vec_processed + tid; i < numel; i += grid_stride) {
        out[i] = Convert<S, T>::apply(load_readonly(&inp[i]));
    }
}

// Unified dispatcher - selects appropriate implementation based on traits
template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_contiguous_vec2(
    const size_t numel,
    const S * __restrict__ inp,
    T * __restrict__ out
) {
    using Traits = Vec2Traits<S, T>;
    // Use SAME_SIZE to dispatch - compiler will optimize away the unused branch
    if (Traits::SAME_SIZE) {
        cast_contiguous_vec2_same_size<S, T, BLOCK_SIZE>(numel, inp, out);
    } else {
        cast_contiguous_vec2_different_size<S, T, BLOCK_SIZE>(numel, inp, out);
    }
}

// =============================================================================
// FP8 E4M3 pack/unpack helpers - reduces code duplication
// =============================================================================

struct FP8x4 {
    // Unpack 4 FP8 values from uint32 to float4
    __device__ __forceinline__ static float4 unpack_to_f32(uint32_t packed) {
        float4 result;
        result.x = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 0) & 0xFF, __NV_E4M3));
        result.y = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 8) & 0xFF, __NV_E4M3));
        result.z = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 16) & 0xFF, __NV_E4M3));
        result.w = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 24) & 0xFF, __NV_E4M3));
        return result;
    }
    
    // Pack 4 floats into uint32 as FP8
    __device__ __forceinline__ static uint32_t pack_from_f32(float4 vals) {
        __nv_fp8_e4m3 v0(vals.x), v1(vals.y), v2(vals.z), v3(vals.w);
        return (uint32_t(v0.__x) << 0) | (uint32_t(v1.__x) << 8) | 
               (uint32_t(v2.__x) << 16) | (uint32_t(v3.__x) << 24);
    }
    
    // Unpack to bfloat162 pair (for bf16 output)
    __device__ __forceinline__ static void unpack_to_bf16x2(uint32_t packed, __nv_bfloat162& out0, __nv_bfloat162& out1) {
        float f0 = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 0) & 0xFF, __NV_E4M3));
        float f1 = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 8) & 0xFF, __NV_E4M3));
        float f2 = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 16) & 0xFF, __NV_E4M3));
        float f3 = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 24) & 0xFF, __NV_E4M3));
        out0 = __float22bfloat162_rn(make_float2(f0, f1));
        out1 = __float22bfloat162_rn(make_float2(f2, f3));
    }
    
    // Pack from bfloat162 pair
    __device__ __forceinline__ static uint32_t pack_from_bf16x2(__nv_bfloat162 bf0, __nv_bfloat162 bf1) {
        float2 f0 = __bfloat1622float2(bf0);
        float2 f1 = __bfloat1622float2(bf1);
        __nv_fp8_e4m3 v0(f0.x), v1(f0.y), v2(f1.x), v3(f1.y);
        return (uint32_t(v0.__x) << 0) | (uint32_t(v1.__x) << 8) | 
               (uint32_t(v2.__x) << 16) | (uint32_t(v3.__x) << 24);
    }
};

// =============================================================================
// FP8 E4M3 <-> F32 optimized conversions (4 elements at once via uint32)
// =============================================================================

// FP8 is 1 byte, so we can process 4 at a time using uint32_t as transport
template<int BLOCK_SIZE>
__device__ void cast_contiguous_vec4_f8_f32(
    const size_t numel,
    const __nv_fp8_e4m3 * __restrict__ inp,
    float * __restrict__ out
) {
    constexpr int UNROLL = 4;
    const unsigned int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned int grid_stride = BLOCK_SIZE * gridDim.x;
    
    // uint32 needs 4-byte alignment, float4 needs 16-byte alignment
    const bool inp_aligned = is_aligned<4>(inp);
    const bool out_aligned = is_aligned<16>(out);
    
    if (!inp_aligned || !out_aligned) {
        // Scalar fallback
        for (unsigned int i = tid; i < numel; i += grid_stride) {
            out[i] = __half2float(__nv_cvt_fp8_to_halfraw(load_readonly(&inp[i]).__x, __NV_E4M3));
        }
        return;
    }
    
    // Process 4 FP8 values at a time
    const size_t numel4 = numel / 4;
    const uint32_t* __restrict__ inp4 = reinterpret_cast<const uint32_t*>(inp);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(out);
    
    // Unrolled main loop
    unsigned int i = tid;
    const unsigned int unroll_limit = (numel4 >= UNROLL * grid_stride) ? numel4 - (UNROLL - 1) * grid_stride : 0;
    
    while (i < unroll_limit) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const unsigned int idx = i + u * grid_stride;
            out4[idx] = FP8x4::unpack_to_f32(inp4[idx]);
        }
        i += UNROLL * grid_stride;
    }
    
    // Remainder of vec4 chunks
    for (; i < numel4; i += grid_stride) {
        out4[i] = FP8x4::unpack_to_f32(inp4[i]);
    }
    
    // Handle remainder (up to 3 elements)
    const unsigned int remainder_start = numel4 * 4;
    const unsigned int remainder_idx = remainder_start + tid;
    if (tid < (numel - remainder_start) && remainder_idx < numel) {
        out[remainder_idx] = __half2float(__nv_cvt_fp8_to_halfraw(load_readonly(&inp[remainder_idx]).__x, __NV_E4M3));
    }
}

template<int BLOCK_SIZE>
__device__ void cast_contiguous_vec4_f32_f8(
    const size_t numel,
    const float * __restrict__ inp,
    __nv_fp8_e4m3 * __restrict__ out
) {
    constexpr int UNROLL = 4;
    const unsigned int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned int grid_stride = BLOCK_SIZE * gridDim.x;
    
    // float4 needs 16-byte alignment, uint32 needs 4-byte alignment
    const bool inp_aligned = is_aligned<16>(inp);
    const bool out_aligned = is_aligned<4>(out);
    
    if (!inp_aligned || !out_aligned) {
        // Scalar fallback
        for (unsigned int i = tid; i < numel; i += grid_stride) {
            out[i] = __nv_fp8_e4m3(load_readonly(&inp[i]));
        }
        return;
    }
    
    // Process 4 floats at a time, output 4 FP8 packed in uint32
    const size_t numel4 = numel / 4;
    const float4* __restrict__ inp4 = reinterpret_cast<const float4*>(inp);
    uint32_t* __restrict__ out4 = reinterpret_cast<uint32_t*>(out);
    
    // Unrolled main loop
    unsigned int i = tid;
    const unsigned int unroll_limit = (numel4 >= UNROLL * grid_stride) ? numel4 - (UNROLL - 1) * grid_stride : 0;
    
    while (i < unroll_limit) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const unsigned int idx = i + u * grid_stride;
            out4[idx] = FP8x4::pack_from_f32(inp4[idx]);
        }
        i += UNROLL * grid_stride;
    }
    
    // Remainder of vec4 chunks
    for (; i < numel4; i += grid_stride) {
        out4[i] = FP8x4::pack_from_f32(inp4[i]);
    }
    
    // Handle remainder (up to 3 elements)
    const unsigned int remainder_start = numel4 * 4;
    const unsigned int remainder_idx = remainder_start + tid;
    if (tid < (numel - remainder_start) && remainder_idx < numel) {
        out[remainder_idx] = __nv_fp8_e4m3(load_readonly(&inp[remainder_idx]));
    }
}

// =============================================================================
// BF16 <-> E4M3 optimized conversions (4 elements at once)
// =============================================================================

template<int BLOCK_SIZE>
__device__ void cast_contiguous_vec4_bf16_f8(
    const size_t numel,
    const __nv_bfloat16 * __restrict__ inp,
    __nv_fp8_e4m3 * __restrict__ out
) {
    constexpr int UNROLL = 4;
    const unsigned int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned int grid_stride = BLOCK_SIZE * gridDim.x;
    
    // bfloat162 needs 4-byte alignment, uint32 needs 4-byte alignment
    const bool inp_aligned = is_aligned<4>(inp);
    const bool out_aligned = is_aligned<4>(out);
    
    if (!inp_aligned || !out_aligned) {
        // Scalar fallback
        for (unsigned int i = tid; i < numel; i += grid_stride) {
            out[i] = __nv_fp8_e4m3(__bfloat162float(load_readonly(&inp[i])));
        }
        return;
    }
    
    // Process 4 bf16 at a time (2x bfloat162), output 4 FP8 packed in uint32
    const size_t numel4 = numel / 4;
    const __nv_bfloat162* __restrict__ inp2 = reinterpret_cast<const __nv_bfloat162*>(inp);
    uint32_t* __restrict__ out4 = reinterpret_cast<uint32_t*>(out);
    
    // Unrolled main loop
    unsigned int i = tid;
    const unsigned int unroll_limit = (numel4 >= UNROLL * grid_stride) ? numel4 - (UNROLL - 1) * grid_stride : 0;
    
    while (i < unroll_limit) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const unsigned int idx = i + u * grid_stride;
            out4[idx] = FP8x4::pack_from_bf16x2(inp2[idx * 2], inp2[idx * 2 + 1]);
        }
        i += UNROLL * grid_stride;
    }
    
    // Remainder of vec4 chunks
    for (; i < numel4; i += grid_stride) {
        out4[i] = FP8x4::pack_from_bf16x2(inp2[i * 2], inp2[i * 2 + 1]);
    }
    
    // Handle remainder (up to 3 elements)
    const unsigned int remainder_start = numel4 * 4;
    const unsigned int remainder_idx = remainder_start + tid;
    if (tid < (numel - remainder_start) && remainder_idx < numel) {
        out[remainder_idx] = __nv_fp8_e4m3(__bfloat162float(load_readonly(&inp[remainder_idx])));
    }
}

template<int BLOCK_SIZE>
__device__ void cast_contiguous_vec4_f8_bf16(
    const size_t numel,
    const __nv_fp8_e4m3 * __restrict__ inp,
    __nv_bfloat16 * __restrict__ out
) {
    constexpr int UNROLL = 4;
    const unsigned int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned int grid_stride = BLOCK_SIZE * gridDim.x;
    
    // uint32 needs 4-byte alignment, bfloat162 needs 4-byte alignment
    const bool inp_aligned = is_aligned<4>(inp);
    const bool out_aligned = is_aligned<4>(out);
    
    if (!inp_aligned || !out_aligned) {
        // Scalar fallback
        for (unsigned int i = tid; i < numel; i += grid_stride) {
            float f = __half2float(__nv_cvt_fp8_to_halfraw(load_readonly(&inp[i]).__x, __NV_E4M3));
            out[i] = __float2bfloat16_rn(f);
        }
        return;
    }
    
    // Process 4 FP8 at a time (uint32), output 4 bf16 as 2x bfloat162
    const size_t numel4 = numel / 4;
    const uint32_t* __restrict__ inp4 = reinterpret_cast<const uint32_t*>(inp);
    __nv_bfloat162* __restrict__ out2 = reinterpret_cast<__nv_bfloat162*>(out);
    
    // Unrolled main loop
    unsigned int i = tid;
    const unsigned int unroll_limit = (numel4 >= UNROLL * grid_stride) ? numel4 - (UNROLL - 1) * grid_stride : 0;
    
    while (i < unroll_limit) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const unsigned int idx = i + u * grid_stride;
            FP8x4::unpack_to_bf16x2(inp4[idx], out2[idx * 2], out2[idx * 2 + 1]);
        }
        i += UNROLL * grid_stride;
    }
    
    // Remainder of vec4 chunks
    for (; i < numel4; i += grid_stride) {
        FP8x4::unpack_to_bf16x2(inp4[i], out2[i * 2], out2[i * 2 + 1]);
    }
    
    // Handle remainder (up to 3 elements)
    const unsigned int remainder_start = numel4 * 4;
    const unsigned int remainder_idx = remainder_start + tid;
    if (tid < (numel - remainder_start) && remainder_idx < numel) {
        float f = __half2float(__nv_cvt_fp8_to_halfraw(load_readonly(&inp[remainder_idx]).__x, __NV_E4M3));
        out[remainder_idx] = __float2bfloat16_rn(f);
    }
}

// =============================================================================
// IN-PLACE CAST OPERATIONS (cast_mut_*)
// =============================================================================
// These kernels convert dtype in-place within the same buffer, avoiding
// allocation of a new tensor. This is memory-efficient for type conversions.
//
// MEMORY SAFETY CHALLENGE:
// In-place conversion has overlapping source/destination regions. Without
// careful ordering, writes can corrupt unread source data.
//
// SOLUTION: Two execution modes for maximum parallelism with safety:
//
// 1. COOPERATIVE GRID MODE (high parallelism):
//    - Uses cooperative_groups for grid-wide synchronization
//    - All blocks read simultaneously → grid.sync() → all blocks write
//    - Requires cudaLaunchCooperativeKernel on host side
//    - Full GPU utilization with memory safety
//
// 2. SINGLE-BLOCK FALLBACK MODE:
//    - For GPUs/drivers without cooperative launch support
//    - Single block with block-level __syncthreads()
//    - Lower parallelism but universally compatible
//
// Three traversal patterns based on size relationship:
// - Shrinking (sizeof(T) < sizeof(S)): Forward traversal
// - Expanding (sizeof(T) > sizeof(S)): Backward traversal  
// - Same-size (sizeof(T) == sizeof(S)): Any direction with staging
//
// Only contiguous tensors supported (no strided access for in-place).
// =============================================================================

// -----------------------------------------------------------------------------
// COOPERATIVE GRID MODE: Maximum parallelism with grid-wide synchronization
// These kernels use cooperative_groups::this_grid().sync() for safety
// MUST be launched with cudaLaunchCooperativeKernel
// -----------------------------------------------------------------------------

// Shrinking cast with cooperative grid sync - forward traversal
template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_shrinking_coop(
    cg::grid_group grid,
    const size_t numel,
    void * __restrict__ buf
) {
    static_assert(sizeof(T) <= sizeof(S), "Use shrinking for sizeof(T) <= sizeof(S)");
    
    S* __restrict__ inp = reinterpret_cast<S*>(buf);
    T* __restrict__ out = reinterpret_cast<T*>(buf);
    
    const size_t global_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t grid_stride = gridDim.x * BLOCK_SIZE;
    
    // Process in grid-stride loop with cooperative sync
    for (size_t base = 0; base < numel; base += grid_stride) {
        const size_t idx = base + global_id;
        
        // Phase 1: All threads across entire grid read their element
        S val;
        bool valid = idx < numel;
        if (valid) {
            val = inp[idx];
        }
        
        // Grid-wide barrier: ensure ALL blocks finish reading before ANY write
        grid.sync();
        
        // Phase 2: All threads write their converted value
        if (valid) {
            out[idx] = Convert<S, T>::apply(val);
        }
        
        // Grid-wide barrier before next iteration
        grid.sync();
    }
}

// Expanding cast with cooperative grid sync - backward traversal
template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_expanding_coop(
    cg::grid_group grid,
    const size_t numel,
    void * __restrict__ buf
) {
    static_assert(sizeof(T) >= sizeof(S), "Use expanding for sizeof(T) >= sizeof(S)");
    
    S* __restrict__ inp = reinterpret_cast<S*>(buf);
    T* __restrict__ out = reinterpret_cast<T*>(buf);
    
    const size_t global_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t grid_stride = gridDim.x * BLOCK_SIZE;
    
    // Calculate total chunks for backward iteration
    const size_t total_chunks = (numel + grid_stride - 1) / grid_stride;
    
    // Process from end to start
    for (size_t chunk = total_chunks; chunk > 0; chunk--) {
        const size_t base = (chunk - 1) * grid_stride;
        const size_t idx = base + global_id;
        
        // Phase 1: All threads read
        S val;
        bool valid = idx < numel;
        if (valid) {
            val = inp[idx];
        }
        
        // Grid-wide barrier
        grid.sync();
        
        // Phase 2: All threads write
        if (valid) {
            out[idx] = Convert<S, T>::apply(val);
        }
        
        // Sync before next chunk
        grid.sync();
    }
}

// Same-size cast with cooperative grid sync - uses registers (no shared memory needed)
template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_same_coop(
    cg::grid_group grid,
    const size_t numel,
    void * __restrict__ buf
) {
    static_assert(sizeof(T) == sizeof(S), "Use same for sizeof(T) == sizeof(S)");
    
    S* __restrict__ inp = reinterpret_cast<S*>(buf);
    T* __restrict__ out = reinterpret_cast<T*>(buf);
    
    const size_t global_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t grid_stride = gridDim.x * BLOCK_SIZE;
    
    for (size_t base = 0; base < numel; base += grid_stride) {
        const size_t idx = base + global_id;
        
        S val;
        bool valid = idx < numel;
        if (valid) {
            val = inp[idx];
        }
        
        grid.sync();
        
        if (valid) {
            out[idx] = Convert<S, T>::apply(val);
        }
        
        grid.sync();
    }
}

// -----------------------------------------------------------------------------
// COOPERATIVE GRID MODE: Vectorized variants (vec2)
// -----------------------------------------------------------------------------

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_shrinking_coop_vec2(
    cg::grid_group grid,
    const size_t numel,
    void * __restrict__ buf
) {
    using Traits = Vec2Traits<S, T>;
    using SrcVec = typename Traits::SrcVec;
    using DstVec = typename Traits::DstVec;
    
    // Alignment check - fall back to scalar if not aligned
    if (!is_aligned<Traits::SRC_ALIGN>(buf)) {
        cast_mut_shrinking_coop<S, T, BLOCK_SIZE>(grid, numel, buf);
        return;
    }
    
    const size_t numel2 = numel / 2;
    SrcVec* __restrict__ inp2 = reinterpret_cast<SrcVec*>(buf);
    DstVec* __restrict__ out2 = reinterpret_cast<DstVec*>(buf);
    
    const size_t global_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t grid_stride = gridDim.x * BLOCK_SIZE;
    
    for (size_t base = 0; base < numel2; base += grid_stride) {
        const size_t idx = base + global_id;
        
        SrcVec val;
        bool valid = idx < numel2;
        if (valid) {
            val = inp2[idx];
        }
        
        grid.sync();
        
        if (valid) {
            out2[idx] = Convert<SrcVec, DstVec>::apply(val);
        }
        
        grid.sync();
    }
    
    // Handle odd element
    if (numel % 2) {
        grid.sync();
        if (global_id == 0) {
            S* inp = reinterpret_cast<S*>(buf);
            T* out = reinterpret_cast<T*>(buf);
            out[numel - 1] = Convert<S, T>::apply(inp[numel - 1]);
        }
        grid.sync();
    }
}

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_expanding_coop_vec2(
    cg::grid_group grid,
    const size_t numel,
    void * __restrict__ buf
) {
    using Traits = Vec2Traits<S, T>;
    using SrcVec = typename Traits::SrcVec;
    using DstVec = typename Traits::DstVec;
    
    if (!is_aligned<Traits::DST_ALIGN>(buf)) {
        cast_mut_expanding_coop<S, T, BLOCK_SIZE>(grid, numel, buf);
        return;
    }
    
    const size_t global_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t grid_stride = gridDim.x * BLOCK_SIZE;
    
    // Handle odd element FIRST (highest index, processed first in backward traversal)
    if (numel % 2) {
        grid.sync();
        if (global_id == 0) {
            S* inp = reinterpret_cast<S*>(buf);
            T* out = reinterpret_cast<T*>(buf);
            out[numel - 1] = Convert<S, T>::apply(inp[numel - 1]);
        }
        grid.sync();
    }
    
    const size_t numel2 = numel / 2;
    SrcVec* __restrict__ inp2 = reinterpret_cast<SrcVec*>(buf);
    DstVec* __restrict__ out2 = reinterpret_cast<DstVec*>(buf);
    
    const size_t total_chunks = (numel2 + grid_stride - 1) / grid_stride;
    
    for (size_t chunk = total_chunks; chunk > 0; chunk--) {
        const size_t base = (chunk - 1) * grid_stride;
        const size_t idx = base + global_id;
        
        SrcVec val;
        bool valid = idx < numel2;
        if (valid) {
            val = inp2[idx];
        }
        
        grid.sync();
        
        if (valid) {
            out2[idx] = Convert<SrcVec, DstVec>::apply(val);
        }
        
        grid.sync();
    }
}

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_same_coop_vec2(
    cg::grid_group grid,
    const size_t numel,
    void * __restrict__ buf
) {
    using Traits = Vec2Traits<S, T>;
    using SrcVec = typename Traits::SrcVec;
    using DstVec = typename Traits::DstVec;
    
    if (!is_aligned<Traits::SRC_ALIGN>(buf)) {
        cast_mut_same_coop<S, T, BLOCK_SIZE>(grid, numel, buf);
        return;
    }
    
    const size_t numel2 = numel / 2;
    SrcVec* __restrict__ inp2 = reinterpret_cast<SrcVec*>(buf);
    DstVec* __restrict__ out2 = reinterpret_cast<DstVec*>(buf);
    
    const size_t global_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t grid_stride = gridDim.x * BLOCK_SIZE;
    
    for (size_t base = 0; base < numel2; base += grid_stride) {
        const size_t idx = base + global_id;
        
        SrcVec val;
        bool valid = idx < numel2;
        if (valid) {
            val = inp2[idx];
        }
        
        grid.sync();
        
        if (valid) {
            out2[idx] = Convert<SrcVec, DstVec>::apply(val);
        }
        
        grid.sync();
    }
    
    // Handle odd element
    if (numel % 2) {
        grid.sync();
        if (global_id == 0) {
            S* inp = reinterpret_cast<S*>(buf);
            T* out = reinterpret_cast<T*>(buf);
            out[numel - 1] = Convert<S, T>::apply(inp[numel - 1]);
        }
        grid.sync();
    }
}

// -----------------------------------------------------------------------------
// SINGLE-BLOCK FALLBACK MODE: For GPUs without cooperative launch support
// These use __syncthreads() within a single block
// MUST be launched with gridDim.x == 1
// -----------------------------------------------------------------------------

// Safe forward traversal for shrinking casts (sizeof(T) < sizeof(S))
// Uses register staging with __syncthreads between read and write phases
// MUST be launched with single block (gridDim.x == 1)
// -----------------------------------------------------------------------------

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_shrinking_phased(
    const size_t numel,
    void * __restrict__ buf
) {
    static_assert(sizeof(T) <= sizeof(S), "Use shrinking for sizeof(T) <= sizeof(S)");
    
    S* __restrict__ inp = reinterpret_cast<S*>(buf);
    T* __restrict__ out = reinterpret_cast<T*>(buf);
    
    // Process in block-sized chunks with explicit synchronization
    // This ensures all reads complete before any writes within each chunk
    for (size_t base = 0; base < numel; base += BLOCK_SIZE) {
        const size_t idx = base + threadIdx.x;
        
        // Phase 1: Read into register
        S val;
        bool valid = idx < numel;
        if (valid) {
            val = inp[idx];
        }
        __syncthreads();  // Ensure ALL reads complete before ANY writes
        
        // Phase 2: Convert and write
        if (valid) {
            out[idx] = Convert<S, T>::apply(val);
        }
        __syncthreads();  // Ensure writes complete before next chunk
    }
}

// -----------------------------------------------------------------------------
// Safe backward traversal for expanding casts (sizeof(T) > sizeof(S))
// Processes from end to start with register staging
// MUST be launched with single block (gridDim.x == 1)
// -----------------------------------------------------------------------------

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_expanding_phased(
    const size_t numel,
    void * __restrict__ buf
) {
    static_assert(sizeof(T) >= sizeof(S), "Use expanding for sizeof(T) >= sizeof(S)");
    
    S* __restrict__ inp = reinterpret_cast<S*>(buf);
    T* __restrict__ out = reinterpret_cast<T*>(buf);
    
    // Process in block-sized chunks from the END, moving backward
    // This ensures we don't overwrite unread source data
    size_t chunks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (size_t chunk = chunks; chunk > 0; chunk--) {
        size_t base = (chunk - 1) * BLOCK_SIZE;
        size_t idx = base + threadIdx.x;
        
        // Phase 1: Read into register
        S val;
        bool valid = idx < numel;
        if (valid) {
            val = inp[idx];
        }
        __syncthreads();  // Ensure ALL reads complete before ANY writes
        
        // Phase 2: Convert and write
        if (valid) {
            out[idx] = Convert<S, T>::apply(val);
        }
        __syncthreads();  // Ensure writes complete before next chunk
    }
}

// -----------------------------------------------------------------------------
// Staged conversion for same-size types (sizeof(T) == sizeof(S))
// Uses shared memory to avoid read-after-write hazards
// MUST be launched with single block (gridDim.x == 1)
// -----------------------------------------------------------------------------

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_staged(
    const size_t numel,
    void * __restrict__ buf
) {
    static_assert(sizeof(T) == sizeof(S), "Use staged for sizeof(T) == sizeof(S)");
    
    __shared__ S shared_buf[BLOCK_SIZE];
    
    S* __restrict__ data_in = reinterpret_cast<S*>(buf);
    T* __restrict__ data_out = reinterpret_cast<T*>(buf);
    
    // Process in block-sized chunks
    for (size_t base = 0; base < numel; base += BLOCK_SIZE) {
        const size_t idx = base + threadIdx.x;
        
        // Phase 1: Load source values into shared memory
        if (idx < numel) {
            shared_buf[threadIdx.x] = data_in[idx];
        }
        __syncthreads();
        
        // Phase 2: Convert and write back
        if (idx < numel) {
            data_out[idx] = Convert<S, T>::apply(shared_buf[threadIdx.x]);
        }
        __syncthreads();  // Ensure writes complete before next chunk
    }
}

// -----------------------------------------------------------------------------
// Vectorized shrinking - vec2 variant with phased execution
// MUST be launched with single block (gridDim.x == 1)
// -----------------------------------------------------------------------------

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_shrinking_vec2(
    const size_t numel,
    void * __restrict__ buf
) {
    using Traits = Vec2Traits<S, T>;
    using SrcVec = typename Traits::SrcVec;
    using DstVec = typename Traits::DstVec;
    
    // Check alignment for vectorized path
    if (!is_aligned<Traits::SRC_ALIGN>(buf)) {
        // Scalar fallback
        cast_mut_shrinking_phased<S, T, BLOCK_SIZE>(numel, buf);
        return;
    }
    
    const size_t numel2 = numel / 2;
    SrcVec* __restrict__ inp2 = reinterpret_cast<SrcVec*>(buf);
    DstVec* __restrict__ out2 = reinterpret_cast<DstVec*>(buf);
    
    // Process vec2 elements with phased execution
    for (size_t base = 0; base < numel2; base += BLOCK_SIZE) {
        const size_t idx = base + threadIdx.x;
        
        // Phase 1: Read vec2 into register
        SrcVec val;
        bool valid = idx < numel2;
        if (valid) {
            val = inp2[idx];
        }
        __syncthreads();
        
        // Phase 2: Convert and write
        if (valid) {
            out2[idx] = Convert<SrcVec, DstVec>::apply(val);
        }
        __syncthreads();
    }
    
    // Handle odd element (scalar)
    if (numel % 2 && threadIdx.x == 0) {
        S* inp = reinterpret_cast<S*>(buf);
        T* out = reinterpret_cast<T*>(buf);
        out[numel - 1] = Convert<S, T>::apply(inp[numel - 1]);
    }
}

// -----------------------------------------------------------------------------
// Vectorized expanding - vec2 variant with backward phased execution
// MUST be launched with single block (gridDim.x == 1)
// -----------------------------------------------------------------------------

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_expanding_vec2(
    const size_t numel,
    void * __restrict__ buf
) {
    using Traits = Vec2Traits<S, T>;
    using SrcVec = typename Traits::SrcVec;
    using DstVec = typename Traits::DstVec;
    
    // Check alignment for vectorized path
    if (!is_aligned<Traits::DST_ALIGN>(buf)) {
        // Scalar fallback
        cast_mut_expanding_phased<S, T, BLOCK_SIZE>(numel, buf);
        return;
    }
    
    // Handle odd element FIRST (it's at the highest address for expanding)
    if (numel % 2 && threadIdx.x == 0) {
        S* inp = reinterpret_cast<S*>(buf);
        T* out = reinterpret_cast<T*>(buf);
        out[numel - 1] = Convert<S, T>::apply(inp[numel - 1]);
    }
    __syncthreads();
    
    const size_t numel2 = numel / 2;
    SrcVec* __restrict__ inp2 = reinterpret_cast<SrcVec*>(buf);
    DstVec* __restrict__ out2 = reinterpret_cast<DstVec*>(buf);
    
    // Process vec2 elements backward with phased execution
    size_t chunks = (numel2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (size_t chunk = chunks; chunk > 0; chunk--) {
        size_t base = (chunk - 1) * BLOCK_SIZE;
        size_t idx = base + threadIdx.x;
        
        // Phase 1: Read vec2 into register
        SrcVec val;
        bool valid = idx < numel2;
        if (valid) {
            val = inp2[idx];
        }
        __syncthreads();
        
        // Phase 2: Convert and write
        if (valid) {
            out2[idx] = Convert<SrcVec, DstVec>::apply(val);
        }
        __syncthreads();
    }
}

// -----------------------------------------------------------------------------
// Vectorized staged (same-size) - vec2 variant with shared memory
// MUST be launched with single block (gridDim.x == 1)
// -----------------------------------------------------------------------------

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_mut_staged_vec2(
    const size_t numel,
    void * __restrict__ buf
) {
    using Traits = Vec2Traits<S, T>;
    using SrcVec = typename Traits::SrcVec;
    using DstVec = typename Traits::DstVec;
    
    __shared__ SrcVec shared_buf[BLOCK_SIZE];
    
    // Check alignment
    if (!is_aligned<Traits::SRC_ALIGN>(buf)) {
        // Fall back to scalar staged
        cast_mut_staged<S, T, BLOCK_SIZE>(numel, buf);
        return;
    }
    
    const size_t numel2 = numel / 2;
    SrcVec* __restrict__ data_in = reinterpret_cast<SrcVec*>(buf);
    DstVec* __restrict__ data_out = reinterpret_cast<DstVec*>(buf);
    
    // Process vec2 chunks with shared memory staging
    for (size_t base = 0; base < numel2; base += BLOCK_SIZE) {
        const size_t idx = base + threadIdx.x;
        
        if (idx < numel2) {
            shared_buf[threadIdx.x] = data_in[idx];
        }
        __syncthreads();
        
        if (idx < numel2) {
            data_out[idx] = Convert<SrcVec, DstVec>::apply(shared_buf[threadIdx.x]);
        }
        __syncthreads();
    }
    
    // Handle odd element (scalar)
    if (numel % 2 && threadIdx.x == 0) {
        S* scalar_in = reinterpret_cast<S*>(buf);
        T* scalar_out = reinterpret_cast<T*>(buf);
        scalar_out[numel - 1] = Convert<S, T>::apply(scalar_in[numel - 1]);
    }
}

// =============================================================================
// Strided kernel - with shared memory coalescing (Optimization 2)
// =============================================================================

template<typename S, typename T, int BLOCK_SIZE>
__device__ void cast_strided(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides,
    const S * __restrict__ inp,
    T * __restrict__ out
) {
    __shared__ S shared_buf[BLOCK_SIZE];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int grid_stride = BLOCK_SIZE * gridDim.x;
    
    for (unsigned int base = blockIdx.x * BLOCK_SIZE; base < numel; base += grid_stride) {
        const unsigned int idx = base + tid;
        
        // Coalesced read from strided input -> shared memory
        if (idx < numel) {
            unsigned int strided_idx = get_strided_index(idx, num_dims, dims, strides);
            shared_buf[tid] = load_readonly(&inp[strided_idx]);
        }
        __syncthreads();
        
        // Convert and coalesced write to output
        if (idx < numel) {
            out[idx] = Convert<S, T>::apply(shared_buf[tid]);
        }
        __syncthreads();
    }
}

// =============================================================================
// Small tensor kernel - single warp, minimal overhead (Optimization 5)
// Uses warp-uniform execution for best efficiency
// =============================================================================

template<typename S, typename T>
__device__ void cast_small(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const S * __restrict__ inp,
    T * __restrict__ out
) {
    const unsigned int tid = threadIdx.x;
    
    // Warp-uniform branch: all threads in warp take same path
    if (info == nullptr) {
        // Contiguous path - simple indexed access
        if (tid < numel) {
            out[tid] = Convert<S, T>::apply(load_readonly(&inp[tid]));
        }
    } else {
        const size_t *dims = info;
        const size_t *strides = info + num_dims;
        
        if (is_contiguous(num_dims, dims, strides)) {
            if (tid < numel) {
                out[tid] = Convert<S, T>::apply(load_readonly(&inp[tid]));
            }
        } else {
            if (tid < numel) {
                unsigned int strided_idx = get_strided_index(tid, num_dims, dims, strides);
                out[tid] = Convert<S, T>::apply(load_readonly(&inp[strided_idx]));
            }
        }
    }
}

// =============================================================================
// Main dispatch kernel (Optimization 4, 8 - branch elimination via dispatch)
// =============================================================================

template<typename S, typename T, int BLOCK_SIZE = 256>
__device__ void cast_kernel(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const S * __restrict__ inp,
    T * __restrict__ out
) {
    // Small tensor path: single block handles everything
    if (numel <= 32 && gridDim.x == 1) {
        cast_small<S, T>(numel, num_dims, info, inp, out);
        return;
    }
    
    // Check contiguity - info being nullptr means contiguous
    if (info == nullptr) {
        cast_contiguous<S, T, BLOCK_SIZE>(numel, inp, out);
    } else {
        const size_t *dims = info;
        const size_t *strides = info + num_dims;
        if (is_contiguous(num_dims, dims, strides)) {
            cast_contiguous<S, T, BLOCK_SIZE>(numel, inp, out);
        } else {
            cast_strided<S, T, BLOCK_SIZE>(numel, num_dims, dims, strides, inp, out);
        }
    }
}

// =============================================================================
// Kernel instantiation macros
// =============================================================================

#define CAST_OP(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const SRC_TYPENAME *inp, \
    DST_TYPENAME *out \
) { \
    cast_kernel<SRC_TYPENAME, DST_TYPENAME, 256>(numel, num_dims, info, inp, out); \
}

// Vectorized specializations for hot paths
#define CAST_OP_VEC4_COPY(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    if (numel <= 32 && gridDim.x == 1) { \
        cast_small<TYPENAME, TYPENAME>(numel, num_dims, info, inp, out); \
        return; \
    } \
    if (info == nullptr) { \
        cast_contiguous_vec4_copy<TYPENAME, 256>(numel, inp, out); \
    } else { \
        const size_t *dims = info; \
        const size_t *strides = info + num_dims; \
        if (is_contiguous(num_dims, dims, strides)) { \
            cast_contiguous_vec4_copy<TYPENAME, 256>(numel, inp, out); \
        } else { \
            cast_strided<TYPENAME, TYPENAME, 256>(numel, num_dims, dims, strides, inp, out); \
        } \
    } \
}

// Unified Vec2 macro - works for any type pair with Vec2Traits specialization
#define CAST_OP_VEC2(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const SRC_TYPENAME *inp, \
    DST_TYPENAME *out \
) { \
    if (numel <= 32 && gridDim.x == 1) { \
        cast_small<SRC_TYPENAME, DST_TYPENAME>(numel, num_dims, info, inp, out); \
        return; \
    } \
    if (info == nullptr) { \
        cast_contiguous_vec2<SRC_TYPENAME, DST_TYPENAME, 256>(numel, inp, out); \
    } else { \
        const size_t *dims = info; \
        const size_t *strides = info + num_dims; \
        if (is_contiguous(num_dims, dims, strides)) { \
            cast_contiguous_vec2<SRC_TYPENAME, DST_TYPENAME, 256>(numel, inp, out); \
        } else { \
            cast_strided<SRC_TYPENAME, DST_TYPENAME, 256>(numel, num_dims, dims, strides, inp, out); \
        } \
    } \
}

// Vectorized FP8 <-> F32 macros
#define CAST_OP_VEC4_F8_F32(FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const __nv_fp8_e4m3 *inp, \
    float *out \
) { \
    if (numel <= 32 && gridDim.x == 1) { \
        cast_small<__nv_fp8_e4m3, float>(numel, num_dims, info, inp, out); \
        return; \
    } \
    if (info == nullptr) { \
        cast_contiguous_vec4_f8_f32<256>(numel, inp, out); \
    } else { \
        const size_t *dims = info; \
        const size_t *strides = info + num_dims; \
        if (is_contiguous(num_dims, dims, strides)) { \
            cast_contiguous_vec4_f8_f32<256>(numel, inp, out); \
        } else { \
            cast_strided<__nv_fp8_e4m3, float, 256>(numel, num_dims, dims, strides, inp, out); \
        } \
    } \
}

#define CAST_OP_VEC4_F32_F8(FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const float *inp, \
    __nv_fp8_e4m3 *out \
) { \
    if (numel <= 32 && gridDim.x == 1) { \
        cast_small<float, __nv_fp8_e4m3>(numel, num_dims, info, inp, out); \
        return; \
    } \
    if (info == nullptr) { \
        cast_contiguous_vec4_f32_f8<256>(numel, inp, out); \
    } else { \
        const size_t *dims = info; \
        const size_t *strides = info + num_dims; \
        if (is_contiguous(num_dims, dims, strides)) { \
            cast_contiguous_vec4_f32_f8<256>(numel, inp, out); \
        } else { \
            cast_strided<float, __nv_fp8_e4m3, 256>(numel, num_dims, dims, strides, inp, out); \
        } \
    } \
}

#define CAST_OP_VEC4_BF16_F8(FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const __nv_bfloat16 *inp, \
    __nv_fp8_e4m3 *out \
) { \
    if (numel <= 32 && gridDim.x == 1) { \
        cast_small<__nv_bfloat16, __nv_fp8_e4m3>(numel, num_dims, info, inp, out); \
        return; \
    } \
    if (info == nullptr) { \
        cast_contiguous_vec4_bf16_f8<256>(numel, inp, out); \
    } else { \
        const size_t *dims = info; \
        const size_t *strides = info + num_dims; \
        if (is_contiguous(num_dims, dims, strides)) { \
            cast_contiguous_vec4_bf16_f8<256>(numel, inp, out); \
        } else { \
            cast_strided<__nv_bfloat16, __nv_fp8_e4m3, 256>(numel, num_dims, dims, strides, inp, out); \
        } \
    } \
}

#define CAST_OP_VEC4_F8_BF16(FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const __nv_fp8_e4m3 *inp, \
    __nv_bfloat16 *out \
) { \
    if (numel <= 32 && gridDim.x == 1) { \
        cast_small<__nv_fp8_e4m3, __nv_bfloat16>(numel, num_dims, info, inp, out); \
        return; \
    } \
    if (info == nullptr) { \
        cast_contiguous_vec4_f8_bf16<256>(numel, inp, out); \
    } else { \
        const size_t *dims = info; \
        const size_t *strides = info + num_dims; \
        if (is_contiguous(num_dims, dims, strides)) { \
            cast_contiguous_vec4_f8_bf16<256>(numel, inp, out); \
        } else { \
            cast_strided<__nv_fp8_e4m3, __nv_bfloat16, 256>(numel, num_dims, dims, strides, inp, out); \
        } \
    } \
}

// Legacy macros for compatibility
#define CAST_OP_FP8(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
    CAST_OP(SRC_TYPENAME, DST_TYPENAME, FN_NAME)

#define CAST_OP_FP8_INTO(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
    CAST_OP(SRC_TYPENAME, DST_TYPENAME, FN_NAME)

#define CAST_THROUGH_OP(SRC_TYPENAME, DST_TYPENAME, INT_TYPENAME, FN_NAME) \
    CAST_OP(SRC_TYPENAME, DST_TYPENAME, FN_NAME)

CAST_OP(__nv_bfloat16, __nv_bfloat16, cast_bf16_bf16)
CAST_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, cast_f8_e4m3_f8_e4m3)

CAST_OP(__nv_bfloat16, uint32_t, cast_bf16_u32)
CAST_OP_VEC2(__nv_bfloat16, float, cast_bf16_f32)  // Vectorized bf16->f32
CAST_OP(__nv_bfloat16, double,   cast_bf16_f64)
CAST_OP(uint8_t, __nv_bfloat16, cast_u8_bf16)
CAST_OP(uint32_t, __nv_bfloat16, cast_u32_bf16)
CAST_OP(int64_t, __nv_bfloat16, cast_i64_bf16)  // i64→bf16
CAST_OP_VEC2(float, __nv_bfloat16, cast_f32_bf16)  // Vectorized f32->bf16
CAST_OP(double,   __nv_bfloat16, cast_f64_bf16)
CAST_THROUGH_OP(__nv_bfloat16, uint8_t, float, cast_bf16_u8)
CAST_OP_VEC2(__nv_bfloat16, __half, cast_bf16_f16)  // Vectorized bf16->f16
CAST_OP_VEC2(__half, __nv_bfloat16, cast_f16_bf16)  // Vectorized f16->bf16

CAST_OP_VEC4_F8_F32(cast_f8_e4m3_f32)  // Vectorized e4m3->f32
CAST_OP_VEC4_F32_F8(cast_f32_f8_e4m3)  // Vectorized f32->e4m3
CAST_OP_FP8(__nv_fp8_e4m3, uint8_t, cast_f8_e4m3_u8)
CAST_OP_FP8(__nv_fp8_e4m3, uint32_t, cast_f8_e4m3_u32)
CAST_OP_FP8(__nv_fp8_e4m3, int64_t, cast_f8_e4m3_i64)
CAST_OP_FP8(__nv_fp8_e4m3, __half, cast_f8_e4m3_f16)
CAST_OP_FP8(__nv_fp8_e4m3, double,  cast_f8_e4m3_f64)
CAST_OP_FP8_INTO(__half,   __nv_fp8_e4m3, cast_f16_f8_e4m3)
CAST_OP_FP8_INTO(double,   __nv_fp8_e4m3, cast_f64_f8_e4m3)
CAST_OP_FP8_INTO(uint8_t,   __nv_fp8_e4m3, cast_u8_f8_e4m3)
CAST_OP_FP8_INTO(uint32_t,  __nv_fp8_e4m3, cast_u32_f8_e4m3)
CAST_OP_FP8_INTO(int64_t,   __nv_fp8_e4m3, cast_i64_f8_e4m3)
CAST_OP_FP8_INTO(int32_t,   __nv_fp8_e4m3, cast_i32_f8_e4m3)
CAST_OP_FP8(__nv_fp8_e4m3, int32_t, cast_f8_e4m3_i32)
CAST_OP_VEC4_F8_BF16(cast_f8_e4m3_bf16)  // Vectorized e4m3->bf16
CAST_OP_VEC4_BF16_F8(cast_bf16_f8_e4m3)  // Vectorized bf16->e4m3

CAST_OP(__half, __half, cast_f16_f16)

CAST_THROUGH_OP(__half, uint8_t,  float, cast_f16_u8)
CAST_OP(__half, uint32_t, cast_f16_u32)
CAST_OP_VEC2(__half, float, cast_f16_f32)  // Vectorized f16->f32
CAST_OP(__half, double,   cast_f16_f64)
CAST_OP(uint8_t,  __half, cast_u8_f16 )
CAST_OP(uint32_t, __half, cast_u32_f16)
CAST_OP(int64_t, __half, cast_i64_f16)  // i64→f16
CAST_OP_VEC2(float, __half, cast_f32_f16)  // Vectorized f32->f16
CAST_OP(double,   __half, cast_f64_f16)

// Use vectorized copy for same-type 32-bit copies
CAST_OP_VEC4_COPY(uint32_t, cast_u32_u32)
CAST_OP(uint32_t, uint8_t,  cast_u32_u8 )
CAST_OP(uint32_t, int64_t,  cast_u32_i64 )
CAST_OP(uint32_t, float,    cast_u32_f32)
CAST_OP(uint32_t, double,   cast_u32_f64)

CAST_OP(uint8_t, uint32_t, cast_u8_u32)
CAST_OP(uint8_t, uint8_t,  cast_u8_u8 )
CAST_OP(uint8_t, int64_t,  cast_u8_i64 )
CAST_OP(uint8_t, float,    cast_u8_f32)
CAST_OP(uint8_t, double,   cast_u8_f64)

CAST_OP(int64_t, uint32_t, cast_i64_u32)
CAST_OP(int64_t, uint8_t,  cast_i64_u8 )
CAST_OP(int64_t, int64_t,  cast_i64_i64 )
CAST_OP(int64_t, float,    cast_i64_f32)
CAST_OP(int64_t, double,   cast_i64_f64)

CAST_OP(float, uint8_t,  cast_f32_u8 )
CAST_OP(float, uint32_t, cast_f32_u32)
CAST_OP(float, int64_t,  cast_f32_i64 )
CAST_OP_VEC4_COPY(float, cast_f32_f32)  // Vectorized f32->f32 copy
CAST_OP(float, double,   cast_f32_f64)

CAST_OP(double, uint8_t,  cast_f64_u8 )
CAST_OP(double, uint32_t, cast_f64_u32)
CAST_OP(double, int64_t,  cast_f64_i64 )
CAST_OP_VEC2(double, float, cast_f64_f32)  // Vectorized double->float
CAST_OP(double, double,   cast_f64_f64)

// =============================================================================
// IN-PLACE CAST KERNEL INSTANTIATION MACROS
// =============================================================================
// Two versions of each kernel:
//   1. _coop suffix: Uses cooperative groups for grid-wide sync (high parallelism)
//      - Launch with cudaLaunchCooperativeKernel
//      - Full GPU utilization
//   2. No suffix: Single-block fallback (universal compatibility)
//      - Launch with <<<1, 256>>>
//      - Lower parallelism but works everywhere
// =============================================================================

// -----------------------------------------------------------------------------
// COOPERATIVE GRID MACROS (high parallelism)
// Launch with cudaLaunchCooperativeKernel, optimal block count for occupancy
// -----------------------------------------------------------------------------

#define CAST_MUT_SHRINKING_COOP(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME##_coop( \
    const size_t numel, \
    void *buf \
) { \
    cg::grid_group grid = cg::this_grid(); \
    cast_mut_shrinking_coop<SRC_TYPENAME, DST_TYPENAME, 256>(grid, numel, buf); \
}

#define CAST_MUT_SHRINKING_VEC2_COOP(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME##_coop( \
    const size_t numel, \
    void *buf \
) { \
    cg::grid_group grid = cg::this_grid(); \
    cast_mut_shrinking_coop_vec2<SRC_TYPENAME, DST_TYPENAME, 256>(grid, numel, buf); \
}

#define CAST_MUT_EXPANDING_COOP(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME##_coop( \
    const size_t numel, \
    void *buf \
) { \
    cg::grid_group grid = cg::this_grid(); \
    cast_mut_expanding_coop<SRC_TYPENAME, DST_TYPENAME, 256>(grid, numel, buf); \
}

#define CAST_MUT_EXPANDING_VEC2_COOP(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME##_coop( \
    const size_t numel, \
    void *buf \
) { \
    cg::grid_group grid = cg::this_grid(); \
    cast_mut_expanding_coop_vec2<SRC_TYPENAME, DST_TYPENAME, 256>(grid, numel, buf); \
}

#define CAST_MUT_SAME_SIZE_COOP(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME##_coop( \
    const size_t numel, \
    void *buf \
) { \
    cg::grid_group grid = cg::this_grid(); \
    cast_mut_same_coop<SRC_TYPENAME, DST_TYPENAME, 256>(grid, numel, buf); \
}

#define CAST_MUT_SAME_SIZE_VEC2_COOP(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME##_coop( \
    const size_t numel, \
    void *buf \
) { \
    cg::grid_group grid = cg::this_grid(); \
    cast_mut_same_coop_vec2<SRC_TYPENAME, DST_TYPENAME, 256>(grid, numel, buf); \
}

// -----------------------------------------------------------------------------
// SINGLE-BLOCK FALLBACK MACROS (universal compatibility)
// Launch with <<<1, 256>>> - works on all GPUs
// -----------------------------------------------------------------------------

#define CAST_MUT_SHRINKING(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    void *buf \
) { \
    cast_mut_shrinking_phased<SRC_TYPENAME, DST_TYPENAME, 256>(numel, buf); \
}

#define CAST_MUT_SHRINKING_VEC2(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    void *buf \
) { \
    cast_mut_shrinking_vec2<SRC_TYPENAME, DST_TYPENAME, 256>(numel, buf); \
}

#define CAST_MUT_EXPANDING(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    void *buf \
) { \
    cast_mut_expanding_phased<SRC_TYPENAME, DST_TYPENAME, 256>(numel, buf); \
}

#define CAST_MUT_EXPANDING_VEC2(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    void *buf \
) { \
    cast_mut_expanding_vec2<SRC_TYPENAME, DST_TYPENAME, 256>(numel, buf); \
}

#define CAST_MUT_SAME_SIZE(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    void *buf \
) { \
    cast_mut_staged<SRC_TYPENAME, DST_TYPENAME, 256>(numel, buf); \
}

#define CAST_MUT_SAME_SIZE_VEC2(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    void *buf \
) { \
    cast_mut_staged_vec2<SRC_TYPENAME, DST_TYPENAME, 256>(numel, buf); \
}

// =============================================================================
// IN-PLACE CAST KERNEL INSTANTIATIONS
// =============================================================================
// Each type conversion gets TWO kernel variants:
//   - cast_mut_X_Y: Single-block fallback (launch with <<<1, 256>>>)
//   - cast_mut_X_Y_coop: Cooperative grid (launch with cudaLaunchCooperativeKernel)
//
// Type sizes for reference:
//   1 byte:  u8, f8_e4m3
//   2 bytes: f16, bf16
//   4 bytes: u32, i32, f32
//   8 bytes: i64, f64
// =============================================================================

// ---------------------------------------------------------------------------
// SHRINKING CASTS (sizeof(T) < sizeof(S)) - Forward traversal
// ---------------------------------------------------------------------------

// 8 bytes -> 4 bytes
CAST_MUT_SHRINKING_VEC2(double, float, cast_mut_f64_f32)
CAST_MUT_SHRINKING_VEC2_COOP(double, float, cast_mut_f64_f32)
CAST_MUT_SHRINKING(double, uint32_t, cast_mut_f64_u32)
CAST_MUT_SHRINKING_COOP(double, uint32_t, cast_mut_f64_u32)
CAST_MUT_SHRINKING(int64_t, float, cast_mut_i64_f32)
CAST_MUT_SHRINKING_COOP(int64_t, float, cast_mut_i64_f32)
CAST_MUT_SHRINKING(int64_t, uint32_t, cast_mut_i64_u32)
CAST_MUT_SHRINKING_COOP(int64_t, uint32_t, cast_mut_i64_u32)

// 8 bytes -> 2 bytes
CAST_MUT_SHRINKING(double, __half, cast_mut_f64_f16)
CAST_MUT_SHRINKING_COOP(double, __half, cast_mut_f64_f16)
CAST_MUT_SHRINKING(int64_t, __half, cast_mut_i64_f16)
CAST_MUT_SHRINKING_COOP(int64_t, __half, cast_mut_i64_f16)
CAST_MUT_SHRINKING(double, __nv_bfloat16, cast_mut_f64_bf16)
CAST_MUT_SHRINKING_COOP(double, __nv_bfloat16, cast_mut_f64_bf16)
CAST_MUT_SHRINKING(int64_t, __nv_bfloat16, cast_mut_i64_bf16)
CAST_MUT_SHRINKING_COOP(int64_t, __nv_bfloat16, cast_mut_i64_bf16)

// 8 bytes -> 1 byte
CAST_MUT_SHRINKING(double, uint8_t, cast_mut_f64_u8)
CAST_MUT_SHRINKING_COOP(double, uint8_t, cast_mut_f64_u8)
CAST_MUT_SHRINKING(int64_t, uint8_t, cast_mut_i64_u8)
CAST_MUT_SHRINKING_COOP(int64_t, uint8_t, cast_mut_i64_u8)
CAST_MUT_SHRINKING(double, __nv_fp8_e4m3, cast_mut_f64_f8_e4m3)
CAST_MUT_SHRINKING_COOP(double, __nv_fp8_e4m3, cast_mut_f64_f8_e4m3)

// 4 bytes -> 2 bytes
CAST_MUT_SHRINKING_VEC2(float, __half, cast_mut_f32_f16)
CAST_MUT_SHRINKING_VEC2_COOP(float, __half, cast_mut_f32_f16)
CAST_MUT_SHRINKING(uint32_t, __half, cast_mut_u32_f16)
CAST_MUT_SHRINKING_COOP(uint32_t, __half, cast_mut_u32_f16)
CAST_MUT_SHRINKING_VEC2(float, __nv_bfloat16, cast_mut_f32_bf16)
CAST_MUT_SHRINKING_VEC2_COOP(float, __nv_bfloat16, cast_mut_f32_bf16)
CAST_MUT_SHRINKING(uint32_t, __nv_bfloat16, cast_mut_u32_bf16)
CAST_MUT_SHRINKING_COOP(uint32_t, __nv_bfloat16, cast_mut_u32_bf16)

// 4 bytes -> 1 byte
CAST_MUT_SHRINKING(float, uint8_t, cast_mut_f32_u8)
CAST_MUT_SHRINKING_COOP(float, uint8_t, cast_mut_f32_u8)
CAST_MUT_SHRINKING(uint32_t, uint8_t, cast_mut_u32_u8)
CAST_MUT_SHRINKING_COOP(uint32_t, uint8_t, cast_mut_u32_u8)
CAST_MUT_SHRINKING(float, __nv_fp8_e4m3, cast_mut_f32_f8_e4m3)
CAST_MUT_SHRINKING_COOP(float, __nv_fp8_e4m3, cast_mut_f32_f8_e4m3)
CAST_MUT_SHRINKING(uint32_t, __nv_fp8_e4m3, cast_mut_u32_f8_e4m3)
CAST_MUT_SHRINKING_COOP(uint32_t, __nv_fp8_e4m3, cast_mut_u32_f8_e4m3)

// 2 bytes -> 1 byte
CAST_MUT_SHRINKING(__half, uint8_t, cast_mut_f16_u8)
CAST_MUT_SHRINKING_COOP(__half, uint8_t, cast_mut_f16_u8)
CAST_MUT_SHRINKING(__nv_bfloat16, uint8_t, cast_mut_bf16_u8)
CAST_MUT_SHRINKING_COOP(__nv_bfloat16, uint8_t, cast_mut_bf16_u8)
CAST_MUT_SHRINKING(__half, __nv_fp8_e4m3, cast_mut_f16_f8_e4m3)
CAST_MUT_SHRINKING_COOP(__half, __nv_fp8_e4m3, cast_mut_f16_f8_e4m3)
CAST_MUT_SHRINKING(__nv_bfloat16, __nv_fp8_e4m3, cast_mut_bf16_f8_e4m3)
CAST_MUT_SHRINKING_COOP(__nv_bfloat16, __nv_fp8_e4m3, cast_mut_bf16_f8_e4m3)

// ---------------------------------------------------------------------------
// EXPANDING CASTS (sizeof(T) > sizeof(S)) - Backward traversal
// ---------------------------------------------------------------------------

// 1 byte -> 2 bytes
CAST_MUT_EXPANDING(uint8_t, __half, cast_mut_u8_f16)
CAST_MUT_EXPANDING_COOP(uint8_t, __half, cast_mut_u8_f16)
CAST_MUT_EXPANDING(uint8_t, __nv_bfloat16, cast_mut_u8_bf16)
CAST_MUT_EXPANDING_COOP(uint8_t, __nv_bfloat16, cast_mut_u8_bf16)
CAST_MUT_EXPANDING(__nv_fp8_e4m3, __half, cast_mut_f8_e4m3_f16)
CAST_MUT_EXPANDING_COOP(__nv_fp8_e4m3, __half, cast_mut_f8_e4m3_f16)
CAST_MUT_EXPANDING(__nv_fp8_e4m3, __nv_bfloat16, cast_mut_f8_e4m3_bf16)
CAST_MUT_EXPANDING_COOP(__nv_fp8_e4m3, __nv_bfloat16, cast_mut_f8_e4m3_bf16)

// 1 byte -> 4 bytes
CAST_MUT_EXPANDING(uint8_t, float, cast_mut_u8_f32)
CAST_MUT_EXPANDING_COOP(uint8_t, float, cast_mut_u8_f32)
CAST_MUT_EXPANDING(uint8_t, uint32_t, cast_mut_u8_u32)
CAST_MUT_EXPANDING_COOP(uint8_t, uint32_t, cast_mut_u8_u32)
CAST_MUT_EXPANDING(__nv_fp8_e4m3, float, cast_mut_f8_e4m3_f32)
CAST_MUT_EXPANDING_COOP(__nv_fp8_e4m3, float, cast_mut_f8_e4m3_f32)
CAST_MUT_EXPANDING(__nv_fp8_e4m3, uint32_t, cast_mut_f8_e4m3_u32)
CAST_MUT_EXPANDING_COOP(__nv_fp8_e4m3, uint32_t, cast_mut_f8_e4m3_u32)

// 1 byte -> 8 bytes
CAST_MUT_EXPANDING(uint8_t, double, cast_mut_u8_f64)
CAST_MUT_EXPANDING_COOP(uint8_t, double, cast_mut_u8_f64)
CAST_MUT_EXPANDING(uint8_t, int64_t, cast_mut_u8_i64)
CAST_MUT_EXPANDING_COOP(uint8_t, int64_t, cast_mut_u8_i64)
CAST_MUT_EXPANDING(__nv_fp8_e4m3, double, cast_mut_f8_e4m3_f64)
CAST_MUT_EXPANDING_COOP(__nv_fp8_e4m3, double, cast_mut_f8_e4m3_f64)
CAST_MUT_EXPANDING(__nv_fp8_e4m3, int64_t, cast_mut_f8_e4m3_i64)
CAST_MUT_EXPANDING_COOP(__nv_fp8_e4m3, int64_t, cast_mut_f8_e4m3_i64)

// 2 bytes -> 4 bytes
CAST_MUT_EXPANDING_VEC2(__half, float, cast_mut_f16_f32)
CAST_MUT_EXPANDING_VEC2_COOP(__half, float, cast_mut_f16_f32)
CAST_MUT_EXPANDING(__half, uint32_t, cast_mut_f16_u32)
CAST_MUT_EXPANDING_COOP(__half, uint32_t, cast_mut_f16_u32)
CAST_MUT_EXPANDING_VEC2(__nv_bfloat16, float, cast_mut_bf16_f32)
CAST_MUT_EXPANDING_VEC2_COOP(__nv_bfloat16, float, cast_mut_bf16_f32)
CAST_MUT_EXPANDING(__nv_bfloat16, uint32_t, cast_mut_bf16_u32)
CAST_MUT_EXPANDING_COOP(__nv_bfloat16, uint32_t, cast_mut_bf16_u32)

// 2 bytes -> 8 bytes
CAST_MUT_EXPANDING(__half, double, cast_mut_f16_f64)
CAST_MUT_EXPANDING_COOP(__half, double, cast_mut_f16_f64)
CAST_MUT_EXPANDING(__half, int64_t, cast_mut_f16_i64)
CAST_MUT_EXPANDING_COOP(__half, int64_t, cast_mut_f16_i64)
CAST_MUT_EXPANDING(__nv_bfloat16, double, cast_mut_bf16_f64)
CAST_MUT_EXPANDING_COOP(__nv_bfloat16, double, cast_mut_bf16_f64)
CAST_MUT_EXPANDING(__nv_bfloat16, int64_t, cast_mut_bf16_i64)
CAST_MUT_EXPANDING_COOP(__nv_bfloat16, int64_t, cast_mut_bf16_i64)

// 4 bytes -> 8 bytes
CAST_MUT_EXPANDING_VEC2(float, double, cast_mut_f32_f64)
CAST_MUT_EXPANDING_VEC2_COOP(float, double, cast_mut_f32_f64)
CAST_MUT_EXPANDING(float, int64_t, cast_mut_f32_i64)
CAST_MUT_EXPANDING_COOP(float, int64_t, cast_mut_f32_i64)
CAST_MUT_EXPANDING(uint32_t, double, cast_mut_u32_f64)
CAST_MUT_EXPANDING_COOP(uint32_t, double, cast_mut_u32_f64)
CAST_MUT_EXPANDING(uint32_t, int64_t, cast_mut_u32_i64)
CAST_MUT_EXPANDING_COOP(uint32_t, int64_t, cast_mut_u32_i64)

// ---------------------------------------------------------------------------
// SAME-SIZE CASTS (sizeof(T) == sizeof(S)) - Staged conversion
// ---------------------------------------------------------------------------

// 1 byte <-> 1 byte
CAST_MUT_SAME_SIZE(uint8_t, uint8_t, cast_mut_u8_u8)
CAST_MUT_SAME_SIZE_COOP(uint8_t, uint8_t, cast_mut_u8_u8)
CAST_MUT_SAME_SIZE(uint8_t, __nv_fp8_e4m3, cast_mut_u8_f8_e4m3)
CAST_MUT_SAME_SIZE_COOP(uint8_t, __nv_fp8_e4m3, cast_mut_u8_f8_e4m3)
CAST_MUT_SAME_SIZE(__nv_fp8_e4m3, uint8_t, cast_mut_f8_e4m3_u8)
CAST_MUT_SAME_SIZE_COOP(__nv_fp8_e4m3, uint8_t, cast_mut_f8_e4m3_u8)
CAST_MUT_SAME_SIZE(__nv_fp8_e4m3, __nv_fp8_e4m3, cast_mut_f8_e4m3_f8_e4m3)
CAST_MUT_SAME_SIZE_COOP(__nv_fp8_e4m3, __nv_fp8_e4m3, cast_mut_f8_e4m3_f8_e4m3)

// 2 bytes <-> 2 bytes
CAST_MUT_SAME_SIZE(__half, __half, cast_mut_f16_f16)
CAST_MUT_SAME_SIZE_COOP(__half, __half, cast_mut_f16_f16)
CAST_MUT_SAME_SIZE(__nv_bfloat16, __nv_bfloat16, cast_mut_bf16_bf16)
CAST_MUT_SAME_SIZE_COOP(__nv_bfloat16, __nv_bfloat16, cast_mut_bf16_bf16)
CAST_MUT_SAME_SIZE_VEC2(__half, __nv_bfloat16, cast_mut_f16_bf16)
CAST_MUT_SAME_SIZE_VEC2_COOP(__half, __nv_bfloat16, cast_mut_f16_bf16)
CAST_MUT_SAME_SIZE_VEC2(__nv_bfloat16, __half, cast_mut_bf16_f16)
CAST_MUT_SAME_SIZE_VEC2_COOP(__nv_bfloat16, __half, cast_mut_bf16_f16)

// 4 bytes <-> 4 bytes
CAST_MUT_SAME_SIZE(float, float, cast_mut_f32_f32)
CAST_MUT_SAME_SIZE_COOP(float, float, cast_mut_f32_f32)
CAST_MUT_SAME_SIZE(uint32_t, uint32_t, cast_mut_u32_u32)
CAST_MUT_SAME_SIZE_COOP(uint32_t, uint32_t, cast_mut_u32_u32)
CAST_MUT_SAME_SIZE(float, uint32_t, cast_mut_f32_u32)
CAST_MUT_SAME_SIZE_COOP(float, uint32_t, cast_mut_f32_u32)
CAST_MUT_SAME_SIZE(uint32_t, float, cast_mut_u32_f32)
CAST_MUT_SAME_SIZE_COOP(uint32_t, float, cast_mut_u32_f32)

// 8 bytes <-> 8 bytes
CAST_MUT_SAME_SIZE(double, double, cast_mut_f64_f64)
CAST_MUT_SAME_SIZE_COOP(double, double, cast_mut_f64_f64)
CAST_MUT_SAME_SIZE(int64_t, int64_t, cast_mut_i64_i64)
CAST_MUT_SAME_SIZE_COOP(int64_t, int64_t, cast_mut_i64_i64)
CAST_MUT_SAME_SIZE(double, int64_t, cast_mut_f64_i64)
CAST_MUT_SAME_SIZE_COOP(double, int64_t, cast_mut_f64_i64)
CAST_MUT_SAME_SIZE(int64_t, double, cast_mut_i64_f64)
CAST_MUT_SAME_SIZE_COOP(int64_t, double, cast_mut_i64_f64)
