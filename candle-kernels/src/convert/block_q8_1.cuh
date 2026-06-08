#pragma once

// Q8_1: 32 elements, 8-bit signed with per-block F16 scale + sum
// Layout: half2 ds, int8_t qs[32]
// Dequant: x[i] = d * qs[i]  (sum field unused)

#include "convert.cuh"

template <> struct BlockConverter<block_q8_1, float> {
    static constexpr int BLOCK_SIZE = QK8_1;
    static __device__ __forceinline__ int load(float* dst, const block_q8_1* src, int lane, float scale) {
        dst[lane] = __half2float(src->ds.x) * (float)src->qs[lane] / scale;
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_q8_1* src, int idx, float scale) {
        return __half2float(src->ds.x) * (float)src->qs[idx] / scale;
    }
};

template <> struct BlockConverter<block_q8_1, __half> {
    static constexpr int BLOCK_SIZE = QK8_1;
    static __device__ __forceinline__ int load(__half* dst, const block_q8_1* src, int lane, float scale) {
        dst[lane] = __float2half_rn(__half2float(src->ds.x) * (float)src->qs[lane] / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_q8_1* src, int idx, float scale) {
        return __float2half_rn(__half2float(src->ds.x) * (float)src->qs[idx] / scale);
    }
};

template <> struct BlockConverter<block_q8_1, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK8_1;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q8_1* src, int lane, float scale) {
        dst[lane] = __float2bfloat16_rn(__half2float(src->ds.x) * (float)src->qs[lane] / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q8_1* src, int idx, float scale) {
        return __float2bfloat16_rn(__half2float(src->ds.x) * (float)src->qs[idx] / scale);
    }
};

template <> struct BlockConverter<block_q8_1, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK8_1;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q8_1* src, int lane, float scale) {
        dst[lane] = from_f32<__nv_fp8_e4m3>(__half2float(src->ds.x) * (float)src->qs[lane] / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q8_1* src, int idx, float scale) {
        return from_f32<__nv_fp8_e4m3>(__half2float(src->ds.x) * (float)src->qs[idx] / scale);
    }
};

template <> struct BlockInt8<block_q8_1> {
    static __device__ __forceinline__ Int8Sample load(const block_q8_1* b, int e) {
        return Int8Sample{ b->qs[e], __half2float(b->ds.x) };   // sum field (ds.y) unused for dequant
    }
};
