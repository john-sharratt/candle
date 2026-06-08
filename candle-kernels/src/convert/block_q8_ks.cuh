#pragma once

// Q8_KS: 32 elements, 8-bit signed with attention-sink sub-block scaling
// Layout: half d, uint8_t sa, uint8_t sb, int8_t qs[32]
// Sub-block A (elems 0-3): scale = d * sa/255; Sub-block B (elems 4-31): scale = d * sb/255

#include "convert.cuh"

static __device__ __forceinline__ float q8_ks_decode(const block_q8_ks* src, int idx) {
    const float cd = __half2float(src->d);
    const float da = cd * (src->sa / 255.0f);
    const float db = cd * (src->sb / 255.0f);
    return ((idx < 4) ? da : db) * (float)src->qs[idx];
}

template <> struct BlockConverter<block_q8_ks, float> {
    static constexpr int BLOCK_SIZE = QK_Q8_KS;
    static __device__ __forceinline__ int load(float* dst, const block_q8_ks* src, int lane, float scale) {
        dst[lane] = q8_ks_decode(src, lane) / scale;
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_q8_ks* src, int idx, float scale) {
        return q8_ks_decode(src, idx) / scale;
    }
};

template <> struct BlockConverter<block_q8_ks, __half> {
    static constexpr int BLOCK_SIZE = QK_Q8_KS;
    static __device__ __forceinline__ int load(__half* dst, const block_q8_ks* src, int lane, float scale) {
        dst[lane] = __float2half_rn(q8_ks_decode(src, lane) / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_q8_ks* src, int idx, float scale) {
        return __float2half_rn(q8_ks_decode(src, idx) / scale);
    }
};

template <> struct BlockConverter<block_q8_ks, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK_Q8_KS;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q8_ks* src, int lane, float scale) {
        dst[lane] = __float2bfloat16_rn(q8_ks_decode(src, lane) / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q8_ks* src, int idx, float scale) {
        return __float2bfloat16_rn(q8_ks_decode(src, idx) / scale);
    }
};

template <> struct BlockConverter<block_q8_ks, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK_Q8_KS;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q8_ks* src, int lane, float scale) {
        dst[lane] = from_f32<__nv_fp8_e4m3>(q8_ks_decode(src, lane) / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q8_ks* src, int idx, float scale) {
        return from_f32<__nv_fp8_e4m3>(q8_ks_decode(src, idx) / scale);
    }
};

template <> struct BlockInt8<block_q8_ks> {
    static __device__ __forceinline__ Int8Sample load(const block_q8_ks* b, int e) {
        const float fine = (e < 4) ? (float)b->sa : (float)b->sb;
        return Int8Sample{ b->qs[e], __half2float(b->d) * fine * (1.0f / 255.0f) };
    }
};
