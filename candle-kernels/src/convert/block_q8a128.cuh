#pragma once

// q8a128: 128 elements, 8-bit signed, per-128 — ONE F16 {scale, sum} for the
// whole 128-element tile, stored at ds[0]. Layout: half2 ds[4] (ds[0] = the
// tile's {scale, sum}; ds[1..3] are 16-byte-alignment pad), int8_t qs[128].
// Dequant: x[i] = ds[0].x * qs[i]  (sum field ds[0].y unused for dequant)
//
// This is the contiguous q8 *activation* twin of the q8_1 weight block, but the
// int8 matmul folds one activation scale per 128-K MMA accumulation, so the
// block carries a single per-128 scale with a 16-byte-aligned qs run for wide
// cp.async. The BlockConverter below is the per-element decode trait the
// convert/attention path uses, identical in shape to block_q8_1.cuh (4 compute
// types + an Int8Sample read-through).

#include "convert.cuh"

template <> struct BlockConverter<block_q8a128, float> {
    static constexpr int BLOCK_SIZE = QK8A128;
    static __device__ __forceinline__ int load(float* dst, const block_q8a128* src, int lane, float scale) {
        dst[lane] = __half2float(src->ds[0].x) * (float)src->qs[lane] / scale;
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ float load_element(const block_q8a128* src, int idx, float scale) {
        return __half2float(src->ds[0].x) * (float)src->qs[idx] / scale;
    }
};

template <> struct BlockConverter<block_q8a128, __half> {
    static constexpr int BLOCK_SIZE = QK8A128;
    static __device__ __forceinline__ int load(__half* dst, const block_q8a128* src, int lane, float scale) {
        dst[lane] = __float2half_rn(__half2float(src->ds[0].x) * (float)src->qs[lane] / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __half load_element(const block_q8a128* src, int idx, float scale) {
        return __float2half_rn(__half2float(src->ds[0].x) * (float)src->qs[idx] / scale);
    }
};

template <> struct BlockConverter<block_q8a128, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK8A128;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q8a128* src, int lane, float scale) {
        dst[lane] = __float2bfloat16_rn(__half2float(src->ds[0].x) * (float)src->qs[lane] / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q8a128* src, int idx, float scale) {
        return __float2bfloat16_rn(__half2float(src->ds[0].x) * (float)src->qs[idx] / scale);
    }
};

template <> struct BlockConverter<block_q8a128, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK8A128;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q8a128* src, int lane, float scale) {
        dst[lane] = from_f32<__nv_fp8_e4m3>(__half2float(src->ds[0].x) * (float)src->qs[lane] / scale);
        return BLOCK_SIZE;
    }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q8a128* src, int idx, float scale) {
        return from_f32<__nv_fp8_e4m3>(__half2float(src->ds[0].x) * (float)src->qs[idx] / scale);
    }
};

// int8 read-through: q8a128 is per-128 symmetric (sum unused for dequant), so
// element e returns its raw int8 plus the tile's single scale ds[0].x.
template <> struct BlockInt8<block_q8a128> {
    static __device__ __forceinline__ Int8Sample load(const block_q8a128* b, int e) {
        return Int8Sample{ b->qs[e], __half2float(b->ds[0].x) };
    }
};
