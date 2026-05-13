#pragma once
// Q0: INT8 centroid — all elements in block share one centroid value
#include "convert.cuh"

template <> struct BlockConverter<block_q0, float> {
    static constexpr int BLOCK_SIZE = QK_Q0;
    static __device__ __forceinline__ int load(float* dst, const block_q0* src, int lane, float scale) {
        dst[lane] = (float)src->centroid * (1.0f / 127.0f) / scale; return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_q0* src, int, float scale) {
        return (float)src->centroid * (1.0f / 127.0f) / scale;
    }
};
template <> struct BlockConverter<block_q0, __half> {
    static constexpr int BLOCK_SIZE = QK_Q0;
    static __device__ __forceinline__ int load(__half* dst, const block_q0* src, int lane, float scale) {
        dst[lane] = __float2half_rn((float)src->centroid * (1.0f / 127.0f) / scale); return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_q0* src, int, float scale) {
        return __float2half_rn((float)src->centroid * (1.0f / 127.0f) / scale);
    }
};
template <> struct BlockConverter<block_q0, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK_Q0;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q0* src, int lane, float scale) {
        dst[lane] = __float2bfloat16_rn((float)src->centroid * (1.0f / 127.0f) / scale); return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q0* src, int, float scale) {
        return __float2bfloat16_rn((float)src->centroid * (1.0f / 127.0f) / scale);
    }
};
template <> struct BlockConverter<block_q0, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK_Q0;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q0* src, int lane, float scale) {
        dst[lane] = from_f32<__nv_fp8_e4m3>((float)src->centroid * (1.0f / 127.0f) / scale); return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q0* src, int, float scale) {
        return from_f32<__nv_fp8_e4m3>((float)src->centroid * (1.0f / 127.0f) / scale);
    }
};
