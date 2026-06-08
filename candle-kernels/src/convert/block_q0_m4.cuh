#pragma once
// Q0_M4: INT8 centroid dequantization (four constants + 32-bit pair mask)
#include "convert.cuh"

static __device__ __forceinline__ float q0_m4_elem(const block_q0_m4* s, int e) {
    return (float)s->centroid[(s->qmask >> (2 * (e / 2))) & 3] * (1.0f / 127.0f);
}

template <> struct BlockConverter<block_q0_m4, float> {
    static constexpr int BLOCK_SIZE = QK_Q0_M4;
    static __device__ __forceinline__ int load(float* dst, const block_q0_m4* src, int lane, float scale)
    { dst[lane] = q0_m4_elem(src, lane) / scale; return BLOCK_SIZE; }
    static __device__ __forceinline__ float load_element(const block_q0_m4* src, int e, float scale)
    { return q0_m4_elem(src, e) / scale; }
};
template <> struct BlockConverter<block_q0_m4, __half> {
    static constexpr int BLOCK_SIZE = QK_Q0_M4;
    static __device__ __forceinline__ int load(__half* dst, const block_q0_m4* src, int lane, float scale)
    { dst[lane] = __float2half_rn(q0_m4_elem(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __half load_element(const block_q0_m4* src, int e, float scale)
    { return __float2half_rn(q0_m4_elem(src, e) / scale); }
};
template <> struct BlockConverter<block_q0_m4, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK_Q0_M4;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q0_m4* src, int lane, float scale)
    { dst[lane] = __float2bfloat16_rn(q0_m4_elem(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q0_m4* src, int e, float scale)
    { return __float2bfloat16_rn(q0_m4_elem(src, e) / scale); }
};
template <> struct BlockConverter<block_q0_m4, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK_Q0_M4;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q0_m4* src, int lane, float scale)
    { dst[lane] = from_f32<__nv_fp8_e4m3>(q0_m4_elem(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q0_m4* src, int e, float scale)
    { return from_f32<__nv_fp8_e4m3>(q0_m4_elem(src, e) / scale); }
};

template <> struct BlockInt8<block_q0_m4> {
    static __device__ __forceinline__ Int8Sample load(const block_q0_m4* b, int e) {
        return Int8Sample{ b->centroid[(b->qmask >> (2 * (e / 2))) & 3], (1.0f / 127.0f) };
    }
};
