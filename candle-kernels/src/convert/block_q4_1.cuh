#pragma once

// Q4_1: 32 elements, 4-bit asymmetric, per-block F16 scale + min
// Layout: half2 dm, uint8_t qs[16]
// Dequant: x[i] = d * nibble[i] + m

#include "convert.cuh"

template <> struct BlockConverter<block_q4_1, float> {
    static constexpr int BLOCK_SIZE = QK4_1;
    static __device__ __forceinline__ int load(float* dst, const block_q4_1* src, int lane, float scale) {
        const __half2 dm = src->dm;
        const float d = __half2float(dm.x) / scale;
        const float m = __half2float(dm.y) / scale;
        const uint8_t byte = src->qs[lane & 15];
        const int nibble = (lane >= 16) ? (byte >> 4) : (byte & 0xF);
        dst[lane] = __fmaf_rn(d, (float)nibble, m);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_q4_1* src, int idx, float scale) {
        const __half2 dm = src->dm;
        const float d = __half2float(dm.x) / scale;
        const float m = __half2float(dm.y) / scale;
        const uint8_t byte = src->qs[idx & 15];
        const int nibble = (idx >= 16) ? (byte >> 4) : (byte & 0xF);
        return d * (float)nibble + m;
    }
};

template <> struct BlockConverter<block_q4_1, __half> {
    static constexpr int BLOCK_SIZE = QK4_1;
    static __device__ __forceinline__ int load(__half* dst, const block_q4_1* src, int lane, float scale) {
        const __half2 dm = src->dm;
        const float d = __half2float(dm.x) / scale;
        const float m = __half2float(dm.y) / scale;
        const uint8_t byte = src->qs[lane & 15];
        const int nibble = (lane >= 16) ? (byte >> 4) : (byte & 0xF);
        dst[lane] = __float2half_rn(d * (float)nibble + m);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_q4_1* src, int idx, float scale) {
        const __half2 dm = src->dm;
        const float d = __half2float(dm.x) / scale;
        const float m = __half2float(dm.y) / scale;
        const uint8_t byte = src->qs[idx & 15];
        const int nibble = (idx >= 16) ? (byte >> 4) : (byte & 0xF);
        return __float2half_rn(d * (float)nibble + m);
    }
};

template <> struct BlockConverter<block_q4_1, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK4_1;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q4_1* src, int lane, float scale) {
        const __half2 dm = src->dm;
        const float d = __half2float(dm.x) / scale;
        const float m = __half2float(dm.y) / scale;
        const uint8_t byte = src->qs[lane & 15];
        const int nibble = (lane >= 16) ? (byte >> 4) : (byte & 0xF);
        dst[lane] = __float2bfloat16_rn(d * (float)nibble + m);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q4_1* src, int idx, float scale) {
        const __half2 dm = src->dm;
        const float d = __half2float(dm.x) / scale;
        const float m = __half2float(dm.y) / scale;
        const uint8_t byte = src->qs[idx & 15];
        const int nibble = (idx >= 16) ? (byte >> 4) : (byte & 0xF);
        return __float2bfloat16_rn(d * (float)nibble + m);
    }
};

template <> struct BlockConverter<block_q4_1, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK4_1;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q4_1* src, int lane, float scale) {
        const __half2 dm = src->dm;
        const float d = __half2float(dm.x) / scale;
        const float m = __half2float(dm.y) / scale;
        const uint8_t byte = src->qs[lane & 15];
        const int nibble = (lane >= 16) ? (byte >> 4) : (byte & 0xF);
        dst[lane] = from_f32<__nv_fp8_e4m3>(__fmaf_rn(d, (float)nibble, m));
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q4_1* src, int idx, float scale) {
        const __half2 dm = src->dm;
        const float d = __half2float(dm.x) / scale;
        const float m = __half2float(dm.y) / scale;
        const uint8_t byte = src->qs[idx & 15];
        const int nibble = (idx >= 16) ? (byte >> 4) : (byte & 0xF);
        return from_f32<__nv_fp8_e4m3>(d * (float)nibble + m);
    }
};
