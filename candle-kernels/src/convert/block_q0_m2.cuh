#pragma once
// Q0_M2: INT8 centroid dequantization (two constants + 8-bit quartet mask)
#include "convert.cuh"

static __device__ __forceinline__ float q0_m2_elem(const block_q0_m2* s, int e) {
    return (float)s->centroid[(s->qmask >> (e / 4)) & 1] * (1.0f / 127.0f);
}

template <> struct BlockConverter<block_q0_m2, float> {
    static constexpr int BLOCK_SIZE = QK_Q0_M2;
    static __device__ __forceinline__ int load(float* dst, const block_q0_m2* src, int lane, float scale)
    { dst[lane] = q0_m2_elem(src, lane) / scale; return BLOCK_SIZE; }
    static __device__ __forceinline__ float load_element(const block_q0_m2* src, int e, float scale)
    { return q0_m2_elem(src, e) / scale; }
};
template <> struct BlockConverter<block_q0_m2, __half> {
    static constexpr int BLOCK_SIZE = QK_Q0_M2;
    static __device__ __forceinline__ int load(__half* dst, const block_q0_m2* src, int lane, float scale)
    { dst[lane] = __float2half_rn(q0_m2_elem(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __half load_element(const block_q0_m2* src, int e, float scale)
    { return __float2half_rn(q0_m2_elem(src, e) / scale); }
};
template <> struct BlockConverter<block_q0_m2, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK_Q0_M2;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q0_m2* src, int lane, float scale)
    { dst[lane] = __float2bfloat16_rn(q0_m2_elem(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q0_m2* src, int e, float scale)
    { return __float2bfloat16_rn(q0_m2_elem(src, e) / scale); }
};
template <> struct BlockConverter<block_q0_m2, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK_Q0_M2;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q0_m2* src, int lane, float scale)
    { dst[lane] = from_f32<__nv_fp8_e4m3>(q0_m2_elem(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q0_m2* src, int e, float scale)
    { return from_f32<__nv_fp8_e4m3>(q0_m2_elem(src, e) / scale); }
};
