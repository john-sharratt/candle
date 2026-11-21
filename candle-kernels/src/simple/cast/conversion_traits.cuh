// The namespace alias, the type-conversion and vector-type traits, and the
// contiguous cast kernels.
//
// `Convert`, `Vec2Type`, `Vec2Traits` and `Vec2Type` choose how each
// element-type pair is converted, including the packed forms that convert two
// or four elements at once. `load_readonly` reads through the read-only cache,
// and `cast_contiguous` with its vectorised variants casts contiguous tensors.

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
