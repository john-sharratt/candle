#pragma once
// Q1_S: 1-bit symmetric with INT8 scale
// Dequant: x[i] = bit ? +blk_scale/127 : -blk_scale/127
#include "convert.cuh"

template <> struct BlockConverter<block_q1_s, float> {
    static constexpr int BLOCK_SIZE = QK1_S;
    static __device__ __forceinline__ int load(float* dst, const block_q1_s* src, int lane, float scale) {
        const float blk_scale = (float)src->scale * (1.0f / 127.0f) / scale;
        const int bit = (src->qs[lane >> 3] >> (lane & 7)) & 1;
        dst[lane] = bit ? blk_scale : -blk_scale; return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_q1_s* src, int idx, float scale) {
        const float blk_scale = (float)src->scale * (1.0f / 127.0f) / scale;
        const int bit = (src->qs[idx >> 3] >> (idx & 7)) & 1;
        return bit ? blk_scale : -blk_scale;
    }
};
template <> struct BlockConverter<block_q1_s, __half> {
    static constexpr int BLOCK_SIZE = QK1_S;
    static __device__ __forceinline__ int load(__half* dst, const block_q1_s* src, int lane, float scale) {
        const float blk_scale = (float)src->scale * (1.0f / 127.0f) / scale;
        const int bit = (src->qs[lane >> 3] >> (lane & 7)) & 1;
        dst[lane] = __float2half_rn(bit ? blk_scale : -blk_scale); return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_q1_s* src, int idx, float scale) {
        const float blk_scale = (float)src->scale * (1.0f / 127.0f) / scale;
        const int bit = (src->qs[idx >> 3] >> (idx & 7)) & 1;
        return __float2half_rn(bit ? blk_scale : -blk_scale);
    }
};
template <> struct BlockConverter<block_q1_s, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK1_S;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q1_s* src, int lane, float scale) {
        const float blk_scale = (float)src->scale * (1.0f / 127.0f) / scale;
        const int bit = (src->qs[lane >> 3] >> (lane & 7)) & 1;
        dst[lane] = __float2bfloat16_rn(bit ? blk_scale : -blk_scale); return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q1_s* src, int idx, float scale) {
        const float blk_scale = (float)src->scale * (1.0f / 127.0f) / scale;
        const int bit = (src->qs[idx >> 3] >> (idx & 7)) & 1;
        return __float2bfloat16_rn(bit ? blk_scale : -blk_scale);
    }
};
template <> struct BlockConverter<block_q1_s, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK1_S;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q1_s* src, int lane, float scale) {
        const float blk_scale = (float)src->scale * (1.0f / 127.0f) / scale;
        const int bit = (src->qs[lane >> 3] >> (lane & 7)) & 1;
        dst[lane] = from_f32<__nv_fp8_e4m3>(bit ? blk_scale : -blk_scale); return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q1_s* src, int idx, float scale) {
        const float blk_scale = (float)src->scale * (1.0f / 127.0f) / scale;
        const int bit = (src->qs[idx >> 3] >> (idx & 7)) & 1;
        return from_f32<__nv_fp8_e4m3>(bit ? blk_scale : -blk_scale);
    }
};

template <> struct BlockInt8<block_q1_s> {
    static __device__ __forceinline__ Int8Sample load(const block_q1_s* b, int e) {
        const int bit = (b->qs[e >> 3] >> (e & 7)) & 1;          // 1-bit sign
        return Int8Sample{ (int8_t)(bit ? (int)b->scale : -(int)b->scale), (1.0f / 127.0f) };
    }
};
