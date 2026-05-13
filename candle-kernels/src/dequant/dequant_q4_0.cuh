// SPDX-License-Identifier: MIT
// Q4_0: 32 elements, 4-bit quantization with per-block absolute scale (no offset)
// Block format: float16 scale (d) + 16 uint8 quantized values (qs)
// Quantized values: 2×uint4, each representing one of 16 levels
// Reconstruction: x_i = d * (q_i - 8)
//
// DUAL-BLOCK CORE: Process 2 blocks (64 elements) with full warp + LUT + shuffle
// - 32 lanes: lanes 0-15 handle block 0, lanes 16-31 handle block 1
// - Each lane outputs 2 elements using vectorized stores
// - LUT optimization: build 16-entry LUT via shuffle to avoid multiply per element

#pragma once

// Core dual-block dequant - half output with LUT + shuffle
// Split packing: low nibble → element pair_idx, high nibble → element pair_idx+16
__device__ __forceinline__ void dequantize_q4_0_dual(
    const uint8_t* __restrict__ qs0,
    const uint8_t* __restrict__ qs1,
    const __half d0,
    const __half d1,
    __half* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int lut_idx = lane_id & 0xF;
    
    // Build LUT: LUT[i] = d * (i - 8)
    const float d = __half2float(block_idx ? d1 : d0);
    const float lut_val = float(lut_idx - 8) * d;
    
    // Load and extract nibbles
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const uint8_t q = qs[lut_idx];
    const int nibble_lo = q & 0xF;
    const int nibble_hi = q >> 4;
    
    // Shuffle-based LUT lookup
    const int base_lane = block_idx ? 16 : 0;
    const float val_lo = __shfl_sync(0xffffffff, lut_val, base_lane + nibble_lo);
    const float val_hi = __shfl_sync(0xffffffff, lut_val, base_lane + nibble_hi);
    
    const int out_base = block_idx * 32;
    dst[out_base + lut_idx]      = __float2half_rn(val_lo);
    dst[out_base + lut_idx + 16] = __float2half_rn(val_hi);
}

// Core dual-block dequant - bfloat16 output with LUT + shuffle
// Split packing: low nibble → element pair_idx, high nibble → element pair_idx+16
__device__ __forceinline__ void dequantize_q4_0_dual(
    const uint8_t* __restrict__ qs0,
    const uint8_t* __restrict__ qs1,
    const __half d0,
    const __half d1,
    __nv_bfloat16* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int lut_idx = lane_id & 0xF;
    
    const float d = __half2float(block_idx ? d1 : d0);
    const float lut_val = float(lut_idx - 8) * d;
    
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const uint8_t q = qs[lut_idx];
    const int nibble_lo = q & 0xF;
    const int nibble_hi = q >> 4;
    
    const int base_lane = block_idx ? 16 : 0;
    const float val_lo = __shfl_sync(0xffffffff, lut_val, base_lane + nibble_lo);
    const float val_hi = __shfl_sync(0xffffffff, lut_val, base_lane + nibble_hi);
    
    const int out_base = block_idx * 32;
    dst[out_base + lut_idx]      = __float2bfloat16_rn(val_lo);
    dst[out_base + lut_idx + 16] = __float2bfloat16_rn(val_hi);
}

// Core dual-block dequant - float output with LUT + shuffle
// Split packing: low nibble → element pair_idx, high nibble → element pair_idx+16
__device__ __forceinline__ void dequantize_q4_0_dual(
    const uint8_t* __restrict__ qs0,
    const uint8_t* __restrict__ qs1,
    const __half d0,
    const __half d1,
    float* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int lut_idx = lane_id & 0xF;
    
    const float d = __half2float(block_idx ? d1 : d0);
    const float lut_val = float(lut_idx - 8) * d;
    
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const uint8_t q = qs[lut_idx];
    const int nibble_lo = q & 0xF;
    const int nibble_hi = q >> 4;
    
    const int base_lane = block_idx ? 16 : 0;
    const float val_lo = __shfl_sync(0xffffffff, lut_val, base_lane + nibble_lo);
    const float val_hi = __shfl_sync(0xffffffff, lut_val, base_lane + nibble_hi);
    
    const int out_base = block_idx * 32;
    dst[out_base + lut_idx]      = val_lo;
    dst[out_base + lut_idx + 16] = val_hi;
}

// Core dual-block dequant - FP8 output with LUT + shuffle
// Split packing: low nibble → element pair_idx, high nibble → element pair_idx+16
__device__ __forceinline__ void dequantize_q4_0_dual(
    const uint8_t* __restrict__ qs0,
    const uint8_t* __restrict__ qs1,
    const __half d0,
    const __half d1,
    __nv_fp8_e4m3* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int lut_idx = lane_id & 0xF;
    
    const float d = __half2float(block_idx ? d1 : d0);
    const float lut_val = float(lut_idx - 8) * d;
    
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const uint8_t q = qs[lut_idx];
    const int nibble_lo = q & 0xF;
    const int nibble_hi = q >> 4;
    
    const int base_lane = block_idx ? 16 : 0;
    const float val_lo = __shfl_sync(0xffffffff, lut_val, base_lane + nibble_lo);
    const float val_hi = __shfl_sync(0xffffffff, lut_val, base_lane + nibble_hi);
    
    const int out_base = block_idx * 32;
    dst[out_base + lut_idx]      = __nv_fp8_e4m3(val_lo);
    dst[out_base + lut_idx + 16] = __nv_fp8_e4m3(val_hi);
}

// Core dual-block dequant - generic compute_t output with LUT + shuffle
// Split packing: low nibble → element pair_idx, high nibble → element pair_idx+16
template <typename compute_t>
__device__ __forceinline__ void dequantize_q4_0_dual(
    const uint8_t* __restrict__ qs0,
    const uint8_t* __restrict__ qs1,
    const __half d0,
    const __half d1,
    compute_t* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int block_idx = lane_id >> 4;
    const int lut_idx = lane_id & 0xF;
    
    const float d = __half2float(block_idx ? d1 : d0);
    const float lut_val = float(lut_idx - 8) * d;
    
    const uint8_t* qs = block_idx ? qs1 : qs0;
    const uint8_t q = qs[lut_idx];
    const int nibble_lo = q & 0xF;
    const int nibble_hi = q >> 4;
    
    const int base_lane = block_idx ? 16 : 0;
    const float val_lo = __shfl_sync(0xffffffff, lut_val, base_lane + nibble_lo);
    const float val_hi = __shfl_sync(0xffffffff, lut_val, base_lane + nibble_hi);
    
    const int out_base = block_idx * 32;
    dst[out_base + lut_idx]      = from_f32<compute_t>(val_lo);
    dst[out_base + lut_idx + 16] = from_f32<compute_t>(val_hi);
}

// Block wrappers - extract from AoS structs and call dual-block core

__device__ __forceinline__ void dequantize_block_q4_0(
    const block_q4_0* __restrict__ src,  // Pointer to 2 consecutive blocks
    __half* __restrict__ dst) {
    dequantize_q4_0_dual(src[0].qs, src[1].qs, src[0].d, src[1].d, dst);
}

__device__ __forceinline__ void dequantize_block_q4_0(
    const block_q4_0* __restrict__ src,
    __nv_bfloat16* __restrict__ dst) {
    dequantize_q4_0_dual(src[0].qs, src[1].qs, src[0].d, src[1].d, dst);
}

__device__ __forceinline__ void dequantize_block_q4_0(
    const block_q4_0* __restrict__ src,
    float* __restrict__ dst) {
    dequantize_q4_0_dual(src[0].qs, src[1].qs, src[0].d, src[1].d, dst);
}

__device__ __forceinline__ void dequantize_block_q4_0(
    const block_q4_0* __restrict__ src,
    __nv_fp8_e4m3* __restrict__ dst) {
    dequantize_q4_0_dual(src[0].qs, src[1].qs, src[0].d, src[1].d, dst);
}

template <typename compute_t>
__device__ __forceinline__ void dequantize_block_q4_0(
    const block_q4_0* __restrict__ src,
    compute_t* __restrict__ dst) {
    dequantize_q4_0_dual(src[0].qs, src[1].qs, src[0].d, src[1].d, dst);
}
