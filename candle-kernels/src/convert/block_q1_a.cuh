#pragma once
// Q1_A: 1-bit asymmetric dequantization
//   sign_bit = 1 → x = +scale_pos / 127
//   sign_bit = 0 → x = -scale_neg / 127
// Branchless via select on the per-element sign bit.
#include "convert.cuh"

static __device__ __forceinline__ float q1_a_elem(const block_q1_a* s, int e) {
    const int sign_bit = (s->qs[e >> 3] >> (e & 7)) & 1;
    const int scale_int = sign_bit ? (int)s->scale_pos : (int)s->scale_neg;
    const float magnitude = (float)scale_int * (1.0f / 127.0f);
    return sign_bit ? magnitude : -magnitude;
}

template <> struct BlockConverter<block_q1_a, float> {
    static constexpr int BLOCK_SIZE = QK1_A;
    static __device__ __forceinline__ int load(float* dst, const block_q1_a* src, int lane, float scale)
    { dst[lane] = q1_a_elem(src, lane) / scale; return BLOCK_SIZE; }
    static __device__ __forceinline__ float load_element(const block_q1_a* src, int e, float scale)
    { return q1_a_elem(src, e) / scale; }
};
template <> struct BlockConverter<block_q1_a, __half> {
    static constexpr int BLOCK_SIZE = QK1_A;
    static __device__ __forceinline__ int load(__half* dst, const block_q1_a* src, int lane, float scale)
    { dst[lane] = __float2half_rn(q1_a_elem(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __half load_element(const block_q1_a* src, int e, float scale)
    { return __float2half_rn(q1_a_elem(src, e) / scale); }
};
template <> struct BlockConverter<block_q1_a, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK1_A;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q1_a* src, int lane, float scale)
    { dst[lane] = __float2bfloat16_rn(q1_a_elem(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q1_a* src, int e, float scale)
    { return __float2bfloat16_rn(q1_a_elem(src, e) / scale); }
};
template <> struct BlockConverter<block_q1_a, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK1_A;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q1_a* src, int lane, float scale)
    { dst[lane] = from_f32<__nv_fp8_e4m3>(q1_a_elem(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q1_a* src, int e, float scale)
    { return from_f32<__nv_fp8_e4m3>(q1_a_elem(src, e) / scale); }
};
