#pragma once

// Q5_0: 32 elements, 5-bit symmetric, per-block F16 scale
// Layout: half d, uint8_t qh[4], uint8_t qs[16]
// Dequant: x[i] = d * (q5[i] - 16)

#include "convert.cuh"

template <> struct BlockConverter<block_q5_0, float> {
    static constexpr int BLOCK_SIZE = QK5_0;
    static __device__ __forceinline__ int load(float* dst, const block_q5_0* src, int lane, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t qs_byte = src->qs[lane & 15];
        const int lo4 = (lane >= 16) ? (qs_byte >> 4) : (qs_byte & 0xF);
        const int hi1 = (src->qh[lane >> 3] >> (lane & 7)) & 1;
        dst[lane] = __fmaf_rn(d, (float)(lo4 | (hi1 << 4)), d * -16.f);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_q5_0* src, int idx, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t qs_byte = src->qs[idx & 15];
        const int lo4 = (idx >= 16) ? (qs_byte >> 4) : (qs_byte & 0xF);
        const int hi1 = (src->qh[idx >> 3] >> (idx & 7)) & 1;
        return d * ((float)(lo4 | (hi1 << 4)) - 16.f);
    }
};

template <> struct BlockConverter<block_q5_0, __half> {
    static constexpr int BLOCK_SIZE = QK5_0;
    static __device__ __forceinline__ int load(__half* dst, const block_q5_0* src, int lane, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t qs_byte = src->qs[lane & 15];
        const int lo4 = (lane >= 16) ? (qs_byte >> 4) : (qs_byte & 0xF);
        const int hi1 = (src->qh[lane >> 3] >> (lane & 7)) & 1;
        dst[lane] = __float2half_rn(d * ((float)(lo4 | (hi1 << 4)) - 16.f));
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_q5_0* src, int idx, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t qs_byte = src->qs[idx & 15];
        const int lo4 = (idx >= 16) ? (qs_byte >> 4) : (qs_byte & 0xF);
        const int hi1 = (src->qh[idx >> 3] >> (idx & 7)) & 1;
        return __float2half_rn(d * ((float)(lo4 | (hi1 << 4)) - 16.f));
    }
};

template <> struct BlockConverter<block_q5_0, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK5_0;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q5_0* src, int lane, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t qs_byte = src->qs[lane & 15];
        const int lo4 = (lane >= 16) ? (qs_byte >> 4) : (qs_byte & 0xF);
        const int hi1 = (src->qh[lane >> 3] >> (lane & 7)) & 1;
        dst[lane] = __float2bfloat16_rn(d * ((float)(lo4 | (hi1 << 4)) - 16.f));
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q5_0* src, int idx, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t qs_byte = src->qs[idx & 15];
        const int lo4 = (idx >= 16) ? (qs_byte >> 4) : (qs_byte & 0xF);
        const int hi1 = (src->qh[idx >> 3] >> (idx & 7)) & 1;
        return __float2bfloat16_rn(d * ((float)(lo4 | (hi1 << 4)) - 16.f));
    }
};

template <> struct BlockConverter<block_q5_0, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK5_0;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q5_0* src, int lane, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t qs_byte = src->qs[lane & 15];
        const int lo4 = (lane >= 16) ? (qs_byte >> 4) : (qs_byte & 0xF);
        const int hi1 = (src->qh[lane >> 3] >> (lane & 7)) & 1;
        dst[lane] = from_f32<__nv_fp8_e4m3>(__fmaf_rn(d, (float)(lo4 | (hi1 << 4)), d * -16.f));
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q5_0* src, int idx, float scale) {
        const float d = __half2float(src->d) / scale;
        const uint8_t qs_byte = src->qs[idx & 15];
        const int lo4 = (idx >= 16) ? (qs_byte >> 4) : (qs_byte & 0xF);
        const int hi1 = (src->qh[idx >> 3] >> (idx & 7)) & 1;
        return from_f32<__nv_fp8_e4m3>(d * ((float)(lo4 | (hi1 << 4)) - 16.f));
    }
};
