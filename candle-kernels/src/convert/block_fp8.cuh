#pragma once

#include "convert.cuh"

template <>
struct BlockConverter<block_fp8_e4m3, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = 32;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_fp8_e4m3* src, int lane, float scale) {
        dst[lane] = from_float<__nv_fp8_e4m3>(to_float<__nv_fp8_e4m3>(src->data[lane]) / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ int store(block_fp8_e4m3* dst, const __nv_fp8_e4m3* src, int lane) {
        if (lane < 2) reinterpret_cast<float4*>(dst->data)[lane] = reinterpret_cast<const float4*>(src)[lane];
        return BLOCK_SIZE;
    }
};

template <>
struct BlockConverter<block_fp8_e4m3, float> {
    static constexpr int BLOCK_SIZE = block_fp8_e4m3::QK;
    static __device__ __forceinline__ int load(float* dst, const block_fp8_e4m3* src, int lane, float scale) {
        dst[lane] = to_float<__nv_fp8_e4m3>(src->data[lane]) / scale;
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ int store(block_fp8_e4m3* dst, const float* src, int lane) {
        dst->data[lane] = from_float<__nv_fp8_e4m3>(src[lane]);
        return BLOCK_SIZE;
    }
};

template <>
struct BlockConverter<block_fp8_e4m3, __half> {
    static constexpr int BLOCK_SIZE = block_fp8_e4m3::QK;
    static __device__ __forceinline__ int load(__half* dst, const block_fp8_e4m3* src, int lane, float scale) {
        dst[lane] = __float2half_rn(to_float<__nv_fp8_e4m3>(src->data[lane]) / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ int store(block_fp8_e4m3* dst, const __half* src, int lane) {
        dst->data[lane] = from_float<__nv_fp8_e4m3>(__half2float(src[lane]));
        return BLOCK_SIZE;
    }
};

template <>
struct BlockConverter<block_fp8_e4m3, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = block_fp8_e4m3::QK;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_fp8_e4m3* src, int lane, float scale) {
        dst[lane] = __float2bfloat16_rn(to_float<__nv_fp8_e4m3>(src->data[lane]) / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ int store(block_fp8_e4m3* dst, const __nv_bfloat16* src, int lane) {
        dst->data[lane] = from_float<__nv_fp8_e4m3>(__bfloat162float(src[lane]));
        return BLOCK_SIZE;
    }
};
