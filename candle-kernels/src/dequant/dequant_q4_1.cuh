// SPDX-License-Identifier: MIT
// Q4_1: 32 elements, 4-bit quantization with min+d scales (packed in dm.x/dm.y)
// Block format: half2 dm (dm.x = delta, dm.y = min) + 16 uint8 quantized values (qs)
// Reconstruction: x_i = d * q_i + m
//
// DUAL-BLOCK CORE: Process 2 blocks (64 elements) with full warp + vector math
// - 32 lanes: lanes 0-15 handle block 0, lanes 16-31 handle block 1
// - Each lane outputs 2 elements using vectorized stores
// - Interface: (qs0, qs1, d0, m0, d1, m1, dst) - SoA for MMA compatibility

#pragma once

// Core dual-block dequant - half output
// Split packing: low nibble → element pair_idx, high nibble → element pair_idx+16
__device__ __forceinline__ void dequantize_q4_1_dual(
    const uint8_t* __restrict__ qs0,
    const uint8_t* __restrict__ qs1,
    const __half d0, const __half m0,
    const __half d1, const __half m1,
    __half* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int pair_idx = lane_id & 0xF;
    
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const __half d_h = block_idx ? d1 : d0;
    const __half m_h = block_idx ? m1 : m0;
    const uint8_t q = qs[pair_idx];
    const int out_base = block_idx * 32;
    dst[out_base + pair_idx]      = __hfma(d_h, __int2half_rn(q & 0xF), m_h);
    dst[out_base + pair_idx + 16] = __hfma(d_h, __int2half_rn(q >> 4), m_h);
}

// Core dual-block dequant - bfloat16 output
// Split packing: low nibble → element pair_idx, high nibble → element pair_idx+16
__device__ __forceinline__ void dequantize_q4_1_dual(
    const uint8_t* __restrict__ qs0,
    const uint8_t* __restrict__ qs1,
    const __half d0, const __half m0,
    const __half d1, const __half m1,
    __nv_bfloat16* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int pair_idx = lane_id & 0xF;
    
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const float d = __half2float(block_idx ? d1 : d0);
    const float m = __half2float(block_idx ? m1 : m0);
    const uint8_t q = qs[pair_idx];
    const int out_base = block_idx * 32;
    dst[out_base + pair_idx]      = __float2bfloat16_rn((q & 0xF) * d + m);
    dst[out_base + pair_idx + 16] = __float2bfloat16_rn((q >> 4) * d + m);
}

// Core dual-block dequant - float output
// Split packing: low nibble → element pair_idx, high nibble → element pair_idx+16
__device__ __forceinline__ void dequantize_q4_1_dual(
    const uint8_t* __restrict__ qs0,
    const uint8_t* __restrict__ qs1,
    const __half d0, const __half m0,
    const __half d1, const __half m1,
    float* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int pair_idx = lane_id & 0xF;
    
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const float d = __half2float(block_idx ? d1 : d0);
    const float m = __half2float(block_idx ? m1 : m0);
    const uint8_t q = qs[pair_idx];
    const int out_base = block_idx * 32;
    dst[out_base + pair_idx]      = (q & 0xF) * d + m;
    dst[out_base + pair_idx + 16] = (q >> 4) * d + m;
}

// Core dual-block dequant - FP8 output
// Split packing: low nibble → element pair_idx, high nibble → element pair_idx+16
__device__ __forceinline__ void dequantize_q4_1_dual(
    const uint8_t* __restrict__ qs0,
    const uint8_t* __restrict__ qs1,
    const __half d0, const __half m0,
    const __half d1, const __half m1,
    __nv_fp8_e4m3* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int pair_idx = lane_id & 0xF;
    
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const float d = __half2float(block_idx ? d1 : d0);
    const float m = __half2float(block_idx ? m1 : m0);
    const uint8_t q = qs[pair_idx];
    const int out_base = block_idx * 32;
    dst[out_base + pair_idx]      = __nv_fp8_e4m3((q & 0xF) * d + m);
    dst[out_base + pair_idx + 16] = __nv_fp8_e4m3((q >> 4) * d + m);
}

// Core dual-block dequant - generic compute_t output
// Split packing: low nibble → element pair_idx, high nibble → element pair_idx+16
template <typename compute_t>
__device__ __forceinline__ void dequantize_q4_1_dual(
    const uint8_t* __restrict__ qs0,
    const uint8_t* __restrict__ qs1,
    const __half d0, const __half m0,
    const __half d1, const __half m1,
    compute_t* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int pair_idx = lane_id & 0xF;
    
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const float d = __half2float(block_idx ? d1 : d0);
    const float m = __half2float(block_idx ? m1 : m0);
    const uint8_t q = qs[pair_idx];
    const int out_base = block_idx * 32;
    dst[out_base + pair_idx]      = from_f32<compute_t>((q & 0xF) * d + m);
    dst[out_base + pair_idx + 16] = from_f32<compute_t>((q >> 4) * d + m);
}

// Block wrappers - extract from AoS structs and call dual-block core

__device__ __forceinline__ void dequantize_block_q4_1(
    const block_q4_1* __restrict__ src,
    __half* __restrict__ dst) {
    dequantize_q4_1_dual(src[0].qs, src[1].qs,
                         __low2half(src[0].dm), __high2half(src[0].dm),
                         __low2half(src[1].dm), __high2half(src[1].dm), dst);
}

__device__ __forceinline__ void dequantize_block_q4_1(
    const block_q4_1* __restrict__ src,
    __nv_bfloat16* __restrict__ dst) {
    dequantize_q4_1_dual(src[0].qs, src[1].qs,
                         __low2half(src[0].dm), __high2half(src[0].dm),
                         __low2half(src[1].dm), __high2half(src[1].dm), dst);
}

__device__ __forceinline__ void dequantize_block_q4_1(
    const block_q4_1* __restrict__ src,
    float* __restrict__ dst) {
    dequantize_q4_1_dual(src[0].qs, src[1].qs,
                         __low2half(src[0].dm), __high2half(src[0].dm),
                         __low2half(src[1].dm), __high2half(src[1].dm), dst);
}

__device__ __forceinline__ void dequantize_block_q4_1(
    const block_q4_1* __restrict__ src,
    __nv_fp8_e4m3* __restrict__ dst) {
    dequantize_q4_1_dual(src[0].qs, src[1].qs,
                         __low2half(src[0].dm), __high2half(src[0].dm),
                         __low2half(src[1].dm), __high2half(src[1].dm), dst);
}

template <typename compute_t>
__device__ __forceinline__ void dequantize_block_q4_1(
    const block_q4_1* __restrict__ src,
    compute_t* __restrict__ dst) {
    dequantize_q4_1_dual(src[0].qs, src[1].qs,
                         __low2half(src[0].dm), __high2half(src[0].dm),
                         __low2half(src[1].dm), __high2half(src[1].dm), dst);
}
