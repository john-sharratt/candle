#pragma once
// Q3_1: 3-bit asymmetric with F16 scale + min
// Low 2 bits in qs[8], high bit in qh[4]
// Dequant: x[i] = d * q[i] + m  where q in [0,7]
#include "convert.cuh"

template <> struct BlockConverter<block_q3_1, float> {
    static constexpr int BLOCK_SIZE = QK3_1;
    static __device__ __forceinline__ int load(float* dst, const block_q3_1* src, int lane, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int lo = (src->qs[lane >> 2] >> ((lane & 3) << 1)) & 3;
        const int hi = (src->qh[lane >> 3] >> (lane & 7)) & 1;
        dst[lane] = __fmaf_rn(d, (float)(lo | (hi << 2)), m);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_q3_1* src, int idx, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int lo = (src->qs[idx >> 2] >> ((idx & 3) << 1)) & 3;
        const int hi = (src->qh[idx >> 3] >> (idx & 7)) & 1;
        return d * (float)(lo | (hi << 2)) + m;
    }
};
template <> struct BlockConverter<block_q3_1, __half> {
    static constexpr int BLOCK_SIZE = QK3_1;
    static __device__ __forceinline__ int load(__half* dst, const block_q3_1* src, int lane, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int lo = (src->qs[lane >> 2] >> ((lane & 3) << 1)) & 3;
        const int hi = (src->qh[lane >> 3] >> (lane & 7)) & 1;
        dst[lane] = __float2half_rn(d * (float)(lo | (hi << 2)) + m);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_q3_1* src, int idx, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int lo = (src->qs[idx >> 2] >> ((idx & 3) << 1)) & 3;
        const int hi = (src->qh[idx >> 3] >> (idx & 7)) & 1;
        return __float2half_rn(d * (float)(lo | (hi << 2)) + m);
    }
};
template <> struct BlockConverter<block_q3_1, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK3_1;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q3_1* src, int lane, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int lo = (src->qs[lane >> 2] >> ((lane & 3) << 1)) & 3;
        const int hi = (src->qh[lane >> 3] >> (lane & 7)) & 1;
        dst[lane] = __float2bfloat16_rn(d * (float)(lo | (hi << 2)) + m);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q3_1* src, int idx, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int lo = (src->qs[idx >> 2] >> ((idx & 3) << 1)) & 3;
        const int hi = (src->qh[idx >> 3] >> (idx & 7)) & 1;
        return __float2bfloat16_rn(d * (float)(lo | (hi << 2)) + m);
    }
};
template <> struct BlockConverter<block_q3_1, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK3_1;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q3_1* src, int lane, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int lo = (src->qs[lane >> 2] >> ((lane & 3) << 1)) & 3;
        const int hi = (src->qh[lane >> 3] >> (lane & 7)) & 1;
        dst[lane] = from_f32<__nv_fp8_e4m3>(__fmaf_rn(d, (float)(lo | (hi << 2)), m));
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q3_1* src, int idx, float scale) {
        const float d = __half2float(src->dm.x) / scale;
        const float m = __half2float(src->dm.y) / scale;
        const int lo = (src->qs[idx >> 2] >> ((idx & 3) << 1)) & 3;
        const int hi = (src->qh[idx >> 3] >> (idx & 7)) & 1;
        return from_f32<__nv_fp8_e4m3>(d * (float)(lo | (hi << 2)) + m);
    }
};
