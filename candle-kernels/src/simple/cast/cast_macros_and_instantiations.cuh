// The `CAST_OP` macro family and the kernel instantiations it generates for
// each element-type pair.
//
// The macros that instantiate the cast kernels: the `CAST_OP` family for
// out-of-place casts, the `CAST_MUT_*` family for in-place casts (cooperative
// and single-block forms), and the expanding, shrinking and same-size variants.
// Each instantiation names one element-type pair of the kernels in the earlier
// parts.

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
CAST_THROUGH_OP(__nv_bfloat16, int64_t, float, cast_bf16_i64)
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
// Through f32: an f16 shape arithmetic (`Ceil` → `Cast` to INT64, as the fp16
// converter leaves BiRefNet) had no kernel, and the dispatcher launched
// nothing — the result was whatever the allocation held.
CAST_THROUGH_OP(__half, int64_t,  float, cast_f16_i64)
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
