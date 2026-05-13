#pragma once

// Q4_KS: 32 elements, 4-bit with attention-sink sub-block scaling
// Layout: half d, uint8_t sa, uint8_t sb, uint8_t qs[16]
// Sub-block A (elems 0-3): scale = d * sa/255; Sub-block B (elems 4-31): scale = d * sb/255

#include "convert.cuh"

static __device__ __forceinline__ float q4_ks_decode(const block_q4_ks* src, int idx) {
    const float cd = __half2float(src->d);
    const float da = cd * (src->sa / 255.0f);
    const float db = cd * (src->sb / 255.0f);
    if (idx < 16) {
        const int nibble = (int)(src->qs[idx] & 0xF) - 8;
        return ((idx < 4) ? da : db) * (float)nibble;
    } else {
        const int nibble = (int)(src->qs[idx - 16] >> 4) - 8;
        return db * (float)nibble;
    }
}

template <> struct BlockConverter<block_q4_ks, float> {
    static constexpr int BLOCK_SIZE = QK_Q4_KS;
    static __device__ __forceinline__ int load(float* dst, const block_q4_ks* src, int lane, float scale) {
        dst[lane] = q4_ks_decode(src, lane) / scale;
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_q4_ks* src, int idx, float scale) {
        return q4_ks_decode(src, idx) / scale;
    }
};

template <> struct BlockConverter<block_q4_ks, __half> {
    static constexpr int BLOCK_SIZE = QK_Q4_KS;
    static __device__ __forceinline__ int load(__half* dst, const block_q4_ks* src, int lane, float scale) {
        dst[lane] = __float2half_rn(q4_ks_decode(src, lane) / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_q4_ks* src, int idx, float scale) {
        return __float2half_rn(q4_ks_decode(src, idx) / scale);
    }
};

template <> struct BlockConverter<block_q4_ks, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK_Q4_KS;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q4_ks* src, int lane, float scale) {
        dst[lane] = __float2bfloat16_rn(q4_ks_decode(src, lane) / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q4_ks* src, int idx, float scale) {
        return __float2bfloat16_rn(q4_ks_decode(src, idx) / scale);
    }
};

template <> struct BlockConverter<block_q4_ks, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK_Q4_KS;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q4_ks* src, int lane, float scale) {
        dst[lane] = from_f32<__nv_fp8_e4m3>(q4_ks_decode(src, lane) / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q4_ks* src, int idx, float scale) {
        return from_f32<__nv_fp8_e4m3>(q4_ks_decode(src, idx) / scale);
    }
};
