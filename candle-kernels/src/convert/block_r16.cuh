#pragma once

// NOTE: We deliberately avoid __ldg() here.  __ldg uses the read-only texture
// cache which is NOT coherent with stores from the same kernel.  The decode
// kernel's fused KV scatter writes new K values to R16 arena memory and then
// reads them back during attention — __ldg would return stale data.

#include "convert.cuh"

template <>
struct BlockConverter<block_r16, float> {
    static constexpr int BLOCK_SIZE = QK_R16;
    static __device__ __forceinline__ int load(float* dst, const block_r16* src, int lane, float scale) {
        dst[lane] = __half2float(src->d[lane]) / scale;
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_r16* src, int idx, float scale) {
        return __half2float(src->d[idx]) / scale;
    }
};

template <>
struct BlockConverter<block_r16, __half> {
    static constexpr int BLOCK_SIZE = QK_R16;
    static __device__ __forceinline__ int load(__half* dst, const block_r16* src, int lane, float scale) {
        dst[lane] = __float2half_rn(__half2float(src->d[lane]) / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_r16* src, int idx, float scale) {
        return __float2half_rn(__half2float(src->d[idx]) / scale);
    }
};

template <>
struct BlockConverter<block_r16, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK_R16;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_r16* src, int lane, float scale) {
        dst[lane] = __float2bfloat16_rn(__half2float(src->d[lane]) / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_r16* src, int idx, float scale) {
        return __float2bfloat16_rn(__half2float(src->d[idx]) / scale);
    }
};

template <>
struct BlockConverter<block_r16, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK_R16;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_r16* src, int lane, float scale) {
        dst[lane] = from_f32<__nv_fp8_e4m3>(__half2float(src->d[lane]) / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_r16* src, int idx, float scale) {
        return from_f32<__nv_fp8_e4m3>(__half2float(src->d[idx]) / scale);
    }
};
