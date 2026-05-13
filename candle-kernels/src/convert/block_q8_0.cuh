#pragma once

// Q8_0: 32 elements, 8-bit signed symmetric, per-block F16 scale
// Layout: half d, int8_t qs[32]
// Dequant: x[i] = d * qs[i]

#include "convert.cuh"

template <> struct BlockConverter<block_q8_0, float> {
    static constexpr int BLOCK_SIZE = QK8_0;
    static __device__ __forceinline__ int load(float* dst, const block_q8_0* src, int lane, float scale) {
        dst[lane] = __half2float(src->d) * (float)src->qs[lane] / scale;
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_q8_0* src, int idx, float scale) {
        return __half2float(src->d) * (float)src->qs[idx] / scale;
    }
};

template <> struct BlockConverter<block_q8_0, __half> {
    static constexpr int BLOCK_SIZE = QK8_0;
    static __device__ __forceinline__ int load(__half* dst, const block_q8_0* src, int lane, float scale) {
        dst[lane] = __float2half_rn(__half2float(src->d) * (float)src->qs[lane] / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_q8_0* src, int idx, float scale) {
        return __float2half_rn(__half2float(src->d) * (float)src->qs[idx] / scale);
    }
};

template <> struct BlockConverter<block_q8_0, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK8_0;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q8_0* src, int lane, float scale) {
        dst[lane] = __float2bfloat16_rn(__half2float(src->d) * (float)src->qs[lane] / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q8_0* src, int idx, float scale) {
        return __float2bfloat16_rn(__half2float(src->d) * (float)src->qs[idx] / scale);
    }
};

template <> struct BlockConverter<block_q8_0, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK8_0;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q8_0* src, int lane, float scale) {
        dst[lane] = from_f32<__nv_fp8_e4m3>(__half2float(src->d) * (float)src->qs[lane] / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q8_0* src, int idx, float scale) {
        return from_f32<__nv_fp8_e4m3>(__half2float(src->d) * (float)src->qs[idx] / scale);
    }
};
