// SPDX-License-Identifier: MIT
// Q5_1: 32 elements, 5-bit quantization with min (dm.x = delta, dm.y = min)
// Block format: half2 dm + uint8[4] qh (high bits) + uint8[16] qs (low nibbles)
// Reconstruction: x_i = d * (q_lo | (h_i << 4)) + m
//
// DUAL-BLOCK CORE: Process 2 blocks (64 elements) with full warp + vector math
// - 32 lanes: lanes 0-15 handle block 0, lanes 16-31 handle block 1
// - Each lane outputs 2 elements using vectorized stores
// - qh is 32-bit high-bit mask per block, dm contains d (delta) and m (min)

#pragma once

// Core dual-block dequant - half output with half2 FMA vector math
__device__ __forceinline__ void dequantize_q5_1_dual(
    const uint8_t* __restrict__ qs0, const uint32_t qh0,
    const uint8_t* __restrict__ qs1, const uint32_t qh1,
    const __half d0, const __half m0,
    const __half d1, const __half m1,
    __half* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int pair_idx = lane_id & 0xF;
    
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const uint32_t qh = block_idx ? qh1 : qh0;
    const half2 d2 = __half2half2(block_idx ? d1 : d0);
    const half2 mm2 = __half2half2(block_idx ? m1 : m0);
    const uint8_t q = qs[pair_idx];
    const int idx0 = pair_idx * 2;
    const int idx1 = pair_idx * 2 + 1;
    const int h0 = (qh >> idx0) & 1;
    const int h1 = (qh >> idx1) & 1;
    const half2 q2 = __halves2half2(
        __int2half_rn((q & 0xF) | (h0 << 4)),
        __int2half_rn((q >> 4) | (h1 << 4))
    );
    reinterpret_cast<half2*>(dst)[lane_id] = __hfma2(q2, d2, mm2);
}

// Core dual-block dequant - bfloat16 output with bfloat162 FMA vector math
__device__ __forceinline__ void dequantize_q5_1_dual(
    const uint8_t* __restrict__ qs0, const uint32_t qh0,
    const uint8_t* __restrict__ qs1, const uint32_t qh1,
    const __half d0, const __half m0,
    const __half d1, const __half m1,
    __nv_bfloat16* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int pair_idx = lane_id & 0xF;
    
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const uint32_t qh = block_idx ? qh1 : qh0;
    const float d = __half2float(block_idx ? d1 : d0);
    const float m = __half2float(block_idx ? m1 : m0);
    const uint8_t q = qs[pair_idx];
    const int idx0 = pair_idx * 2;
    const int idx1 = pair_idx * 2 + 1;
    const int h0 = (qh >> idx0) & 1;
    const int h1 = (qh >> idx1) & 1;
    const __nv_bfloat162 result = __floats2bfloat162_rn(
        ((q & 0xF) | (h0 << 4)) * d + m,
        ((q >> 4) | (h1 << 4)) * d + m
    );
    reinterpret_cast<__nv_bfloat162*>(dst)[lane_id] = result;
}

// Core dual-block dequant - float output with float2 vector math
__device__ __forceinline__ void dequantize_q5_1_dual(
    const uint8_t* __restrict__ qs0, const uint32_t qh0,
    const uint8_t* __restrict__ qs1, const uint32_t qh1,
    const __half d0, const __half m0,
    const __half d1, const __half m1,
    float* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int pair_idx = lane_id & 0xF;
    
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const uint32_t qh = block_idx ? qh1 : qh0;
    const float d = __half2float(block_idx ? d1 : d0);
    const float m = __half2float(block_idx ? m1 : m0);
    const uint8_t q = qs[pair_idx];
    const int idx0 = pair_idx * 2;
    const int idx1 = pair_idx * 2 + 1;
    const int h0 = (qh >> idx0) & 1;
    const int h1 = (qh >> idx1) & 1;
    reinterpret_cast<float2*>(dst)[lane_id] = make_float2(
        ((q & 0xF) | (h0 << 4)) * d + m,
        ((q >> 4) | (h1 << 4)) * d + m
    );
}

// Core dual-block dequant - FP8 output with fp8x2 vector math
__device__ __forceinline__ void dequantize_q5_1_dual(
    const uint8_t* __restrict__ qs0, const uint32_t qh0,
    const uint8_t* __restrict__ qs1, const uint32_t qh1,
    const __half d0, const __half m0,
    const __half d1, const __half m1,
    __nv_fp8_e4m3* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int pair_idx = lane_id & 0xF;
    
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const uint32_t qh = block_idx ? qh1 : qh0;
    const float d = __half2float(block_idx ? d1 : d0);
    const float m = __half2float(block_idx ? m1 : m0);
    const uint8_t q = qs[pair_idx];
    const int idx0 = pair_idx * 2;
    const int idx1 = pair_idx * 2 + 1;
    const int h0 = (qh >> idx0) & 1;
    const int h1 = (qh >> idx1) & 1;
    reinterpret_cast<__nv_fp8x2_e4m3*>(dst)[lane_id] = __nv_fp8x2_e4m3(make_float2(
        ((q & 0xF) | (h0 << 4)) * d + m,
        ((q >> 4) | (h1 << 4)) * d + m
    ));
}

// Core dual-block dequant - generic compute_t output
template <typename compute_t>
__device__ __forceinline__ void dequantize_q5_1_dual(
    const uint8_t* __restrict__ qs0, const uint32_t qh0,
    const uint8_t* __restrict__ qs1, const uint32_t qh1,
    const __half d0, const __half m0,
    const __half d1, const __half m1,
    compute_t* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int pair_idx = lane_id & 0xF;
    
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const uint32_t qh = block_idx ? qh1 : qh0;
    const float d = __half2float(block_idx ? d1 : d0);
    const float m = __half2float(block_idx ? m1 : m0);
    const uint8_t q = qs[pair_idx];
    const int idx0 = pair_idx * 2;
    const int idx1 = pair_idx * 2 + 1;
    const int h0 = (qh >> idx0) & 1;
    const int h1 = (qh >> idx1) & 1;
    compute_t* out = dst + lane_id * 2;
    out[0] = from_f32<compute_t>(((q & 0xF) | (h0 << 4)) * d + m);
    out[1] = from_f32<compute_t>(((q >> 4) | (h1 << 4)) * d + m);
}

// Block wrappers - extract from AoS structs and call dual-block core

__device__ __forceinline__ void dequantize_block_q5_1(
    const block_q5_1* __restrict__ src,
    __half* __restrict__ dst) {
    dequantize_q5_1_dual(src[0].qs, *reinterpret_cast<const uint32_t*>(src[0].qh),
                         src[1].qs, *reinterpret_cast<const uint32_t*>(src[1].qh),
                         __low2half(src[0].dm), __high2half(src[0].dm),
                         __low2half(src[1].dm), __high2half(src[1].dm), dst);
}

__device__ __forceinline__ void dequantize_block_q5_1(
    const block_q5_1* __restrict__ src,
    __nv_bfloat16* __restrict__ dst) {
    dequantize_q5_1_dual(src[0].qs, *reinterpret_cast<const uint32_t*>(src[0].qh),
                         src[1].qs, *reinterpret_cast<const uint32_t*>(src[1].qh),
                         __low2half(src[0].dm), __high2half(src[0].dm),
                         __low2half(src[1].dm), __high2half(src[1].dm), dst);
}

__device__ __forceinline__ void dequantize_block_q5_1(
    const block_q5_1* __restrict__ src,
    float* __restrict__ dst) {
    dequantize_q5_1_dual(src[0].qs, *reinterpret_cast<const uint32_t*>(src[0].qh),
                         src[1].qs, *reinterpret_cast<const uint32_t*>(src[1].qh),
                         __low2half(src[0].dm), __high2half(src[0].dm),
                         __low2half(src[1].dm), __high2half(src[1].dm), dst);
}

__device__ __forceinline__ void dequantize_block_q5_1(
    const block_q5_1* __restrict__ src,
    __nv_fp8_e4m3* __restrict__ dst) {
    dequantize_q5_1_dual(src[0].qs, *reinterpret_cast<const uint32_t*>(src[0].qh),
                         src[1].qs, *reinterpret_cast<const uint32_t*>(src[1].qh),
                         __low2half(src[0].dm), __high2half(src[0].dm),
                         __low2half(src[1].dm), __high2half(src[1].dm), dst);
}

template <typename compute_t>
__device__ __forceinline__ void dequantize_block_q5_1(
    const block_q5_1* __restrict__ src,
    compute_t* __restrict__ dst) {
    dequantize_q5_1_dual(src[0].qs, *reinterpret_cast<const uint32_t*>(src[0].qh),
                         src[1].qs, *reinterpret_cast<const uint32_t*>(src[1].qh),
                         __low2half(src[0].dm), __high2half(src[0].dm),
                         __low2half(src[1].dm), __high2half(src[1].dm), dst);
}
