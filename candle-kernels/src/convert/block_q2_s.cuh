#pragma once
// Q2_S: 2-bit symmetric with INT8 scale
// Dequant: x[i] = d * (q[i] - 1.5)  where d = scale/127
#include "convert.cuh"

template <> struct BlockConverter<block_q2_s, float> {
    static constexpr int BLOCK_SIZE = QK2_S;
    static __device__ __forceinline__ int load(float* dst, const block_q2_s* src, int lane, float scale) {
        const float d = (float)src->scale * (1.0f / 127.0f) / scale;
        const int q = (src->qs[lane >> 2] >> ((lane & 3) << 1)) & 3;
        dst[lane] = __fmaf_rn(d, (float)q, d * -1.5f); return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_q2_s* src, int idx, float scale) {
        const float d = (float)src->scale * (1.0f / 127.0f) / scale;
        const int q = (src->qs[idx >> 2] >> ((idx & 3) << 1)) & 3;
        return d * ((float)q - 1.5f);
    }
};
template <> struct BlockConverter<block_q2_s, __half> {
    static constexpr int BLOCK_SIZE = QK2_S;
    static __device__ __forceinline__ int load(__half* dst, const block_q2_s* src, int lane, float scale) {
        const float d = (float)src->scale * (1.0f / 127.0f) / scale;
        const int q = (src->qs[lane >> 2] >> ((lane & 3) << 1)) & 3;
        dst[lane] = __float2half_rn(d * ((float)q - 1.5f)); return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_q2_s* src, int idx, float scale) {
        const float d = (float)src->scale * (1.0f / 127.0f) / scale;
        const int q = (src->qs[idx >> 2] >> ((idx & 3) << 1)) & 3;
        return __float2half_rn(d * ((float)q - 1.5f));
    }
};
template <> struct BlockConverter<block_q2_s, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK2_S;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q2_s* src, int lane, float scale) {
        const float d = (float)src->scale * (1.0f / 127.0f) / scale;
        const int q = (src->qs[lane >> 2] >> ((lane & 3) << 1)) & 3;
        dst[lane] = __float2bfloat16_rn(d * ((float)q - 1.5f)); return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q2_s* src, int idx, float scale) {
        const float d = (float)src->scale * (1.0f / 127.0f) / scale;
        const int q = (src->qs[idx >> 2] >> ((idx & 3) << 1)) & 3;
        return __float2bfloat16_rn(d * ((float)q - 1.5f));
    }
};
template <> struct BlockConverter<block_q2_s, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK2_S;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q2_s* src, int lane, float scale) {
        const float d = (float)src->scale * (1.0f / 127.0f) / scale;
        const int q = (src->qs[lane >> 2] >> ((lane & 3) << 1)) & 3;
        dst[lane] = from_f32<__nv_fp8_e4m3>(__fmaf_rn(d, (float)q, d * -1.5f)); return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q2_s* src, int idx, float scale) {
        const float d = (float)src->scale * (1.0f / 127.0f) / scale;
        const int q = (src->qs[idx >> 2] >> ((idx & 3) << 1)) & 3;
        return from_f32<__nv_fp8_e4m3>(d * ((float)q - 1.5f));
    }
};
