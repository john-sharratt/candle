// SPDX-License-Identifier: MIT

// ============================================================================
// Q_AWQ: AWQ 4-bit quantization with group size 128 in K/128 format
// ============================================================================
// AWQ (Activation-aware Weight Quantization): asymmetric 4-bit with zero-point
// Formula: w = scale * (q - zero) where q ∈ [0, 15]
//
// K/128 block layout (72 bytes):
//   qs[16]: 16 × uint32 packed 4-bit weights (8 per uint32) = 64 bytes
//   scale: half = 2 bytes
//   zero: half = 2 bytes
//   pad: 4 bytes
//
// WARP-PARALLEL with LUT optimization:
// - 128 elements / 32 lanes = 4 elements per lane
// - Build 16-entry LUT: LUT[i] = scale * (i - zero) for i ∈ [0,15]
// - Use shuffle-based lookup for efficient dequantization
// ============================================================================

#pragma once

// Core dequant - float output with LUT + shuffle
__device__ __forceinline__ void dequantize_block_q_awq(
    const block_c_q_awq_k128* __restrict__ src,
    float* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Load scale and zero
    const float scale = __half2float(src->scale);
    const float zero = __half2float(src->zero);
    
    // Build LUT: LUT[i] = scale * (i - zero) for AWQ asymmetric
    const int lut_idx = lane_id & 15;
    const float lut_val = scale * (float(lut_idx) - zero);
    
    // AWQ packs 8 × 4-bit values per uint32
    const uint32_t* qs = src->qs;
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        // Each lane processes 2 uint32s = 16 nibbles = 16 elements
        // But we process 4 elements total per lane, so 1/4 of a uint32
        const int qs_idx = (lane_id * 4 + i * 2) / 8;
        const int nibble_offset = ((lane_id * 4 + i * 2) % 8) * 4;
        
        const uint32_t packed = qs[qs_idx];
        const int nibble0 = (packed >> nibble_offset) & 0xF;
        const int nibble1 = (packed >> (nibble_offset + 4)) & 0xF;
        
        const float val0 = __shfl_sync(0xffffffff, lut_val, nibble0);
        const float val1 = __shfl_sync(0xffffffff, lut_val, nibble1);
        
        dst[lane_id * 4 + i * 2] = val0;
        dst[lane_id * 4 + i * 2 + 1] = val1;
    }
}

// Core dequant - half output with LUT + shuffle
__device__ __forceinline__ void dequantize_block_q_awq(
    const block_c_q_awq_k128* __restrict__ src,
    __half* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    const float scale = __half2float(src->scale);
    const float zero = __half2float(src->zero);
    const int lut_idx = lane_id & 15;
    const float lut_val = scale * (float(lut_idx) - zero);
    
    const uint32_t* qs = src->qs;
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int qs_idx = (lane_id * 4 + i * 2) / 8;
        const int nibble_offset = ((lane_id * 4 + i * 2) % 8) * 4;
        
        const uint32_t packed = qs[qs_idx];
        const int nibble0 = (packed >> nibble_offset) & 0xF;
        const int nibble1 = (packed >> (nibble_offset + 4)) & 0xF;
        
        const float val0 = __shfl_sync(0xffffffff, lut_val, nibble0);
        const float val1 = __shfl_sync(0xffffffff, lut_val, nibble1);
        
        reinterpret_cast<half2*>(dst)[lane_id * 2 + i] = __floats2half2_rn(val0, val1);
    }
}

// Core dequant - bfloat16 output with LUT + shuffle
__device__ __forceinline__ void dequantize_block_q_awq(
    const block_c_q_awq_k128* __restrict__ src,
    __nv_bfloat16* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    const float scale = __half2float(src->scale);
    const float zero = __half2float(src->zero);
    const int lut_idx = lane_id & 15;
    const float lut_val = scale * (float(lut_idx) - zero);
    
    const uint32_t* qs = src->qs;
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int qs_idx = (lane_id * 4 + i * 2) / 8;
        const int nibble_offset = ((lane_id * 4 + i * 2) % 8) * 4;
        
        const uint32_t packed = qs[qs_idx];
        const int nibble0 = (packed >> nibble_offset) & 0xF;
        const int nibble1 = (packed >> (nibble_offset + 4)) & 0xF;
        
        const float val0 = __shfl_sync(0xffffffff, lut_val, nibble0);
        const float val1 = __shfl_sync(0xffffffff, lut_val, nibble1);
        
        reinterpret_cast<__nv_bfloat162*>(dst)[lane_id * 2 + i] = __floats2bfloat162_rn(val0, val1);
    }
}

// Core dequant - FP8 E4M3 output with LUT + shuffle
__device__ __forceinline__ void dequantize_block_q_awq(
    const block_c_q_awq_k128* __restrict__ src,
    __nv_fp8_e4m3* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    const float scale = __half2float(src->scale);
    const float zero = __half2float(src->zero);
    const int lut_idx = lane_id & 15;
    const float lut_val = scale * (float(lut_idx) - zero);
    
    const uint32_t* qs = src->qs;
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int qs_idx = (lane_id * 4 + i * 2) / 8;
        const int nibble_offset = ((lane_id * 4 + i * 2) % 8) * 4;
        
        const uint32_t packed = qs[qs_idx];
        const int nibble0 = (packed >> nibble_offset) & 0xF;
        const int nibble1 = (packed >> (nibble_offset + 4)) & 0xF;
        
        const float val0 = __shfl_sync(0xffffffff, lut_val, nibble0);
        const float val1 = __shfl_sync(0xffffffff, lut_val, nibble1);
        
        reinterpret_cast<__nv_fp8x2_e4m3*>(dst)[lane_id * 2 + i] = __nv_fp8x2_e4m3(make_float2(val0, val1));
    }
}

// Core dequant - generic compute_t output
template <typename compute_t>
__device__ __forceinline__ void dequantize_block_q_awq(
    const block_c_q_awq_k128* __restrict__ src,
    compute_t* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    const float scale = __half2float(src->scale);
    const float zero = __half2float(src->zero);
    const int lut_idx = lane_id & 15;
    const float lut_val = scale * (float(lut_idx) - zero);
    
    const uint32_t* qs = src->qs;
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int qs_idx = (lane_id * 4 + i * 2) / 8;
        const int nibble_offset = ((lane_id * 4 + i * 2) % 8) * 4;
        
        const uint32_t packed = qs[qs_idx];
        const int nibble0 = (packed >> nibble_offset) & 0xF;
        const int nibble1 = (packed >> (nibble_offset + 4)) & 0xF;
        
        const float val0 = __shfl_sync(0xffffffff, lut_val, nibble0);
        const float val1 = __shfl_sync(0xffffffff, lut_val, nibble1);
        
        dst[lane_id * 4 + i * 2] = from_f32<compute_t>(val0);
        dst[lane_id * 4 + i * 2 + 1] = from_f32<compute_t>(val1);
    }
}

// Note: block_c_q_awq is typedef'd to block_c_q_awq_k128, so no separate wrapper needed
