#pragma once

// Q3_0: 32 elements, 3-bit symmetric, per-block F16 scale
// Layout: half d, uint8_t qh[4], uint8_t qs[8]
// Dequant: x[i] = d * (q[i] - 3.5)  where q in [0,7]

#include "convert.cuh"

__device__ __forceinline__ int q3_0_get_q(const block_q3_0* src, int idx) {
    const int lo = (src->qs[idx >> 2] >> ((idx & 3) << 1)) & 3;
    const int hi = (src->qh[idx >> 3] >> (idx & 7)) & 1;
    return (hi << 2) | lo;
}

template <> struct BlockConverter<block_q3_0, float> {
    static constexpr int BLOCK_SIZE = QK3_0;
    static __device__ __forceinline__ int load(float* dst, const block_q3_0* src, int lane, float scale) {
        const float d = __half2float(src->d) / scale;
        dst[lane] = __fmaf_rn(d, (float)q3_0_get_q(src, lane), d * -3.5f);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_q3_0* src, int idx, float scale) {
        return __half2float(src->d) * ((float)q3_0_get_q(src, idx) - 3.5f) / scale;
    }
};

template <> struct BlockConverter<block_q3_0, __half> {
    static constexpr int BLOCK_SIZE = QK3_0;
    static __device__ __forceinline__ int load(__half* dst, const block_q3_0* src, int lane, float scale) {
        const float d = __half2float(src->d) / scale;
        dst[lane] = __float2half_rn(d * ((float)q3_0_get_q(src, lane) - 3.5f));
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_q3_0* src, int idx, float scale) {
        return __float2half_rn(__half2float(src->d) * ((float)q3_0_get_q(src, idx) - 3.5f) / scale);
    }
};

template <> struct BlockConverter<block_q3_0, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK3_0;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q3_0* src, int lane, float scale) {
        const float d = __half2float(src->d) / scale;
        dst[lane] = __float2bfloat16_rn(d * ((float)q3_0_get_q(src, lane) - 3.5f));
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q3_0* src, int idx, float scale) {
        return __float2bfloat16_rn(__half2float(src->d) * ((float)q3_0_get_q(src, idx) - 3.5f) / scale);
    }
};

template <> struct BlockConverter<block_q3_0, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK3_0;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q3_0* src, int lane, float scale) {
        const float d = __half2float(src->d) / scale;
        dst[lane] = from_f32<__nv_fp8_e4m3>(__fmaf_rn(d, (float)q3_0_get_q(src, lane), d * -3.5f));
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q3_0* src, int idx, float scale) {
        return from_f32<__nv_fp8_e4m3>(__half2float(src->d) * ((float)q3_0_get_q(src, idx) - 3.5f) / scale);
    }
};
