// SPDX-License-Identifier: MIT

// ============================================================================
// Q3_K: 256 elements, 3-bit quantization (complex scale encoding)
// Sub-block structure: 256 elements = 8 groups of 32, each group has 2 sub-blocks of 16
// Scale encoding: 6-bit scales packed in 12 bytes (K_SCALE_SIZE)
//
// Q3 formula: dl * (q_2bit - (hmask ? 0 : 4))
// q_2bit ∈ [0,3], so final values ∈ [-4, -3, -2, -1, 0, 1, 2, 3] (8 values → 8-entry LUT!)
//
// Layout:
//   qs[64]: 2 bits per element, packed (4 elements per byte)
//   hmask[32]: 1 bit per element, packed (8 elements per byte)
//   scales[12]: 6-bit scales, complex encoding
//
// WARP-PARALLEL with LUT: 8-entry LUT per sub-block fits easily in first 8 lanes
// ============================================================================

#pragma once

// Helper to decode Q3_K scale (6-bit value from complex packing)
__device__ __forceinline__ int8_t decode_q3_k_scale(const uint8_t* scales, int is) {
    // Scale decoding from llama.cpp - 6-bit scale packed across bytes
    int8_t us;
    if (is < 4) {
        us = (scales[is] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4);
    } else if (is < 8) {
        us = (scales[is] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4);
    } else if (is < 12) {
        us = (scales[is - 8] >> 4) | (((scales[is] >> 4) & 3) << 4);
    } else {
        us = (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);
    }
    return us;
}

// Core templated dequant - generic output (warp-parallel with LUT)
template <int N = QK_K, typename compute_t>
__device__ __forceinline__ void dequantize_q3_K(
    const uint8_t* __restrict__ qs,
    const uint8_t* __restrict__ hmask,
    const uint8_t* __restrict__ scales,
    const float d,
    compute_t* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    const int lane_id = threadIdx.x % WARP_SIZE;
    constexpr int ELEMS_PER_LANE = N / WARP_SIZE;  // 256/32 = 8
    
    // Process 8 elements per lane across 256 total elements
    // Structure: 8 groups of 32 elements, each group has scale index derived from position
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; ++i) {
        const int elem_idx = lane_id + i * WARP_SIZE;
        
        // Decode position within Q3_K structure
        // 256 elements = 128*n + 32*j + position, where n ∈ [0,1], j ∈ [0,3]
        const int n = elem_idx / 128;           // 0 or 1
        const int remainder = elem_idx % 128;
        const int j = remainder / 32;           // 0-3
        const int pos = remainder % 32;         // 0-31
        const int is0 = pos / 16;               // 0 or 1
        const int l = pos % 16;                 // 0-15
        
        // Scale index calculation
        const int is = 8 * n + 2 * j + is0;
        const int8_t us = decode_q3_k_scale(scales, is);
        const float dl = d * float(us - 32);
        
        // Extract 2-bit quant value
        const int shift = 2 * j;
        const uint8_t* q_ptr = qs + 32 * n;
        const int q_2bit = (q_ptr[l + 16 * is0] >> shift) & 3;
        
        // Extract high bit from hmask
        const uint8_t m = 1 << (4 * n + j);
        const int h_bit = (hmask[l + 16 * is0] & m) ? 0 : 4;
        
        // Final value: dl * (q_2bit - h_bit)
        // q_2bit ∈ [0,3], h_bit ∈ {0, 4}, so result ∈ [-4, 3]
        dst[elem_idx] = from_f32<compute_t>(dl * float(q_2bit - h_bit));
    }
}

// Core templated dequant - float output (warp-parallel with LUT)
template <int N = QK_K>
__device__ __forceinline__ void dequantize_q3_K(
    const uint8_t* __restrict__ qs,
    const uint8_t* __restrict__ hmask,
    const uint8_t* __restrict__ scales,
    const float d,
    float* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    const int lane_id = threadIdx.x % WARP_SIZE;
    constexpr int ELEMS_PER_LANE = N / WARP_SIZE;
    
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; ++i) {
        const int elem_idx = lane_id + i * WARP_SIZE;
        
        const int n = elem_idx / 128;
        const int remainder = elem_idx % 128;
        const int j = remainder / 32;
        const int pos = remainder % 32;
        const int is0 = pos / 16;
        const int l = pos % 16;
        
        const int is = 8 * n + 2 * j + is0;
        const int8_t us = decode_q3_k_scale(scales, is);
        const float dl = d * float(us - 32);
        
        const int shift = 2 * j;
        const uint8_t* q_ptr = qs + 32 * n;
        const int q_2bit = (q_ptr[l + 16 * is0] >> shift) & 3;
        
        const uint8_t m = 1 << (4 * n + j);
        const int h_bit = (hmask[l + 16 * is0] & m) ? 0 : 4;
        
        dst[elem_idx] = dl * float(q_2bit - h_bit);
    }
}

// Block wrappers
template <typename compute_t>
__device__ __forceinline__ void dequantize_block_q3_K(
    const block_q3_K* __restrict__ src,
    compute_t* __restrict__ dst) {
    dequantize_q3_K<QK_K>(src->qs, src->hmask, src->scales, __half2float(src->d), dst);
}

__device__ __forceinline__ void dequantize_block_q3_K(
    const block_q3_K* __restrict__ src,
    float* __restrict__ dst) {
    dequantize_q3_K<QK_K>(src->qs, src->hmask, src->scales, __half2float(src->d), dst);
}
