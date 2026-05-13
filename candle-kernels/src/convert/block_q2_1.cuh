#pragma once
// Q2_1: 2-bit asymmetric with F16 scale + min
// Dequant: x[i] = d * q[i] + m  where q in [0,3]
#include "convert.cuh"

template <> struct BlockConverter<block_q2_1, float> {
    static constexpr int BLOCK_SIZE = QK2_1;
    static __device__ __forceinline__ int load(float* dst, const block_q2_1* src, int lane, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int q = (src->qs[lane >> 2] >> ((lane & 3) << 1)) & 3;
        dst[lane] = __fmaf_rn(d, (float)q, m);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_q2_1* src, int idx, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int q = (src->qs[idx >> 2] >> ((idx & 3) << 1)) & 3;
        return d * (float)q + m;
    }
};
template <> struct BlockConverter<block_q2_1, __half> {
    static constexpr int BLOCK_SIZE = QK2_1;
    static __device__ __forceinline__ int load(__half* dst, const block_q2_1* src, int lane, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int q = (src->qs[lane >> 2] >> ((lane & 3) << 1)) & 3;
        dst[lane] = __float2half_rn(d * (float)q + m);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_q2_1* src, int idx, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int q = (src->qs[idx >> 2] >> ((idx & 3) << 1)) & 3;
        return __float2half_rn(d * (float)q + m);
    }
};
template <> struct BlockConverter<block_q2_1, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK2_1;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q2_1* src, int lane, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int q = (src->qs[lane >> 2] >> ((lane & 3) << 1)) & 3;
        dst[lane] = __float2bfloat16_rn(d * (float)q + m);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q2_1* src, int idx, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int q = (src->qs[idx >> 2] >> ((idx & 3) << 1)) & 3;
        return __float2bfloat16_rn(d * (float)q + m);
    }
};
template <> struct BlockConverter<block_q2_1, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK2_1;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q2_1* src, int lane, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int q = (src->qs[lane >> 2] >> ((lane & 3) << 1)) & 3;
        dst[lane] = from_f32<__nv_fp8_e4m3>(__fmaf_rn(d, (float)q, m));
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q2_1* src, int idx, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int q = (src->qs[idx >> 2] >> ((idx & 3) << 1)) & 3;
        return from_f32<__nv_fp8_e4m3>(d * (float)q + m);
    }
};
