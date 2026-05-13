#pragma once
// =============================================================================
// CAST_SCALE HELPER
// =============================================================================
// Template specializations for converting float scales to different output types.
// Used by gemx::extract_scales_impl in each loader file.
// =============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace gemx {

template <typename T>
__device__ __forceinline__ T cast_scale(float val);

template <>
__device__ __forceinline__ __half cast_scale<__half>(float val) {
    return __float2half(val);
}

template <>
__device__ __forceinline__ __nv_bfloat16 cast_scale<__nv_bfloat16>(float val) {
    return __float2bfloat16(val);
}

template <>
__device__ __forceinline__ __nv_fp8_e4m3 cast_scale<__nv_fp8_e4m3>(float val) {
    return __nv_fp8_e4m3(val);
}

template <>
__device__ __forceinline__ float cast_scale<float>(float val) {
    return val;
}

} // namespace gemx
