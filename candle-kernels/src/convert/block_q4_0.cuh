#pragma once

// Q4_0: 32 elements, 4-bit symmetric, per-block F16 scale
// Layout: half d, uint8_t qs[16]
// Dequant: x[i] = d * (nibble[i] - 8)

#include "convert.cuh"

template <> struct BlockConverter<block_q4_0, float> {
    static constexpr int BLOCK_SIZE = QK4_0;
    static __device__ __forceinline__ int load(float* dst, const block_q4_0* src, int lane, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t byte = src->qs[lane & 15];
        const int nibble = (lane >= 16) ? (byte >> 4) : (byte & 0xF);
        dst[lane] = __fmaf_rn(d, (float)nibble, d * -8.f);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_q4_0* src, int idx, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t byte = src->qs[idx & 15];
        const int nibble = (idx >= 16) ? (byte >> 4) : (byte & 0xF);
        return d * ((float)nibble - 8.f);
    }
};

template <> struct BlockConverter<block_q4_0, __half> {
    static constexpr int BLOCK_SIZE = QK4_0;
    static __device__ __forceinline__ int load(__half* dst, const block_q4_0* src, int lane, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t byte = src->qs[lane & 15];
        const int nibble = (lane >= 16) ? (byte >> 4) : (byte & 0xF);
        dst[lane] = __float2half_rn(d * ((float)nibble - 8.f));
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_q4_0* src, int idx, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t byte = src->qs[idx & 15];
        const int nibble = (idx >= 16) ? (byte >> 4) : (byte & 0xF);
        return __float2half_rn(d * ((float)nibble - 8.f));
    }
};

template <> struct BlockConverter<block_q4_0, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK4_0;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q4_0* src, int lane, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t byte = src->qs[lane & 15];
        const int nibble = (lane >= 16) ? (byte >> 4) : (byte & 0xF);
        dst[lane] = __float2bfloat16_rn(d * ((float)nibble - 8.f));
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q4_0* src, int idx, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t byte = src->qs[idx & 15];
        const int nibble = (idx >= 16) ? (byte >> 4) : (byte & 0xF);
        return __float2bfloat16_rn(d * ((float)nibble - 8.f));
    }
};

template <> struct BlockConverter<block_q4_0, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK4_0;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q4_0* src, int lane, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t byte = src->qs[lane & 15];
        const int nibble = (lane >= 16) ? (byte >> 4) : (byte & 0xF);
        dst[lane] = from_f32<__nv_fp8_e4m3>(__fmaf_rn(d, (float)nibble, d * -8.f));
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q4_0* src, int idx, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t byte = src->qs[idx & 15];
        const int nibble = (idx >= 16) ? (byte >> 4) : (byte & 0xF);
        return from_f32<__nv_fp8_e4m3>(d * ((float)nibble - 8.f));
    }
};
