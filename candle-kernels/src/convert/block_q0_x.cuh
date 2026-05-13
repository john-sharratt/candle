#pragma once
// Q0_X: Flat block with one outlier escape — dequantization
//
//   v_i8 = bulk_anchor + (e == outlier_idx ? outlier_delta * Q0_X_S_OUTLIER : 0)
//   x    = clamp(v_i8, -127, 127) / 127.0
//
// outlier_packed: [4:0]=outlier_idx (0..31), [7:5]=outlier_delta (signed
// 3-bit, two's complement, [-4..3]). The clamp guards against int8 overflow
// when bulk_anchor + delta*S exceeds [-127, 127].
#include "convert.cuh"

static __device__ __forceinline__ float q0_x_elem(const block_q0_x* s, int e) {
    const uint8_t packed = s->outlier_packed;
    const int outlier_idx = (int)(packed & 0x1F);
    const int delta_u = (int)((packed >> 5) & 0x07);
    const int outlier_delta = delta_u < 4 ? delta_u : delta_u - 8;  // sign-extend 3 bits
    const int delta_scaled = (e == outlier_idx) ? outlier_delta * Q0_X_S_OUTLIER : 0;
    int v_i8 = (int)s->bulk_anchor + delta_scaled;
    v_i8 = max(-127, min(127, v_i8));
    return (float)v_i8 * (1.0f / 127.0f);
}

template <> struct BlockConverter<block_q0_x, float> {
    static constexpr int BLOCK_SIZE = QK_Q0_X;
    static __device__ __forceinline__ int load(float* dst, const block_q0_x* src, int lane, float scale)
    { dst[lane] = q0_x_elem(src, lane) / scale; return BLOCK_SIZE; }
    static __device__ __forceinline__ float load_element(const block_q0_x* src, int e, float scale)
    { return q0_x_elem(src, e) / scale; }
};
template <> struct BlockConverter<block_q0_x, __half> {
    static constexpr int BLOCK_SIZE = QK_Q0_X;
    static __device__ __forceinline__ int load(__half* dst, const block_q0_x* src, int lane, float scale)
    { dst[lane] = __float2half_rn(q0_x_elem(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __half load_element(const block_q0_x* src, int e, float scale)
    { return __float2half_rn(q0_x_elem(src, e) / scale); }
};
template <> struct BlockConverter<block_q0_x, __nv_bfloat16> {
    static constexpr int BLOCK_SIZE = QK_Q0_X;
    static __device__ __forceinline__ int load(__nv_bfloat16* dst, const block_q0_x* src, int lane, float scale)
    { dst[lane] = __float2bfloat16_rn(q0_x_elem(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __nv_bfloat16 load_element(const block_q0_x* src, int e, float scale)
    { return __float2bfloat16_rn(q0_x_elem(src, e) / scale); }
};
template <> struct BlockConverter<block_q0_x, __nv_fp8_e4m3> {
    static constexpr int BLOCK_SIZE = QK_Q0_X;
    static __device__ __forceinline__ int load(__nv_fp8_e4m3* dst, const block_q0_x* src, int lane, float scale)
    { dst[lane] = from_f32<__nv_fp8_e4m3>(q0_x_elem(src, lane) / scale); return BLOCK_SIZE; }
    static __device__ __forceinline__ __nv_fp8_e4m3 load_element(const block_q0_x* src, int e, float scale)
    { return from_f32<__nv_fp8_e4m3>(q0_x_elem(src, e) / scale); }
};
