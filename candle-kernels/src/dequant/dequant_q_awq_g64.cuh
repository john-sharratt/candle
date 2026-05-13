// SPDX-License-Identifier: MIT

// ============================================================================
// Q_AWQ_G64: AWQ 4-bit quantization with group size 64 in K/128 format
// ============================================================================
// AWQ (Activation-aware Weight Quantization): asymmetric 4-bit with zero-point
// Formula: w = scale * (q - zero) where q ∈ [0, 15]
//
// Group size 64 means 2 groups per K/128 block.
//
// K/128 block layout (72 bytes):
//   qs[16]: 16 × uint32 packed 4-bit weights (8 per uint32) = 64 bytes
//   scales[2]: 2 × half = 4 bytes
//   zeros[2]: 2 × half = 4 bytes
//
// WARP-PARALLEL with LUT optimization:
// - 128 elements / 32 lanes = 4 elements per lane
// - Two groups: lanes 0-15 handle group 0, lanes 16-31 handle group 1
// - Build 16-entry LUT per group
// ============================================================================

#pragma once

// Core dequant - float output with LUT + shuffle
__device__ __forceinline__ void dequantize_block_q_awq_g64(
    const block_c_q_awq_g64_k128* __restrict__ src,
    float* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int group_idx = lane_id >> 4;  // 0 for lanes 0-15, 1 for lanes 16-31
    
    // Load scale and zero for this group
    const float scale = __half2float(src->scales[group_idx]);
    const float zero = __half2float(src->zeros[group_idx]);
    
    // Build LUT: LUT[i] = scale * (i - zero)
    const int lut_idx = lane_id & 15;
    const float lut_val = scale * (float(lut_idx) - zero);
    
    const uint32_t* qs = src->qs;
    
    // Each lane processes 4 elements from its group
    // Group 0: elements 0-63, Group 1: elements 64-127
    const int base_elem = group_idx * 64 + (lane_id & 15) * 4;
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int elem_idx = base_elem + i * 2;
        const int qs_idx = elem_idx / 8;
        const int nibble_offset = (elem_idx % 8) * 4;
        
        const uint32_t packed = qs[qs_idx];
        const int nibble0 = (packed >> nibble_offset) & 0xF;
        const int nibble1 = (packed >> (nibble_offset + 4)) & 0xF;
        
        // Shuffle within group (base_lane = group_idx * 16)
        const int base_lane = group_idx * 16;
        const float val0 = __shfl_sync(0xffffffff, lut_val, base_lane + nibble0);
        const float val1 = __shfl_sync(0xffffffff, lut_val, base_lane + nibble1);
        
        dst[elem_idx] = val0;
        dst[elem_idx + 1] = val1;
    }
}

// Core dequant - half output with LUT + shuffle
__device__ __forceinline__ void dequantize_block_q_awq_g64(
    const block_c_q_awq_g64_k128* __restrict__ src,
    __half* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int group_idx = lane_id >> 4;
    
    const float scale = __half2float(src->scales[group_idx]);
    const float zero = __half2float(src->zeros[group_idx]);
    const int lut_idx = lane_id & 15;
    const float lut_val = scale * (float(lut_idx) - zero);
    
    const uint32_t* qs = src->qs;
    const int base_elem = group_idx * 64 + (lane_id & 15) * 4;
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int elem_idx = base_elem + i * 2;
        const int qs_idx = elem_idx / 8;
        const int nibble_offset = (elem_idx % 8) * 4;
        
        const uint32_t packed = qs[qs_idx];
        const int nibble0 = (packed >> nibble_offset) & 0xF;
        const int nibble1 = (packed >> (nibble_offset + 4)) & 0xF;
        
        const int base_lane = group_idx * 16;
        const float val0 = __shfl_sync(0xffffffff, lut_val, base_lane + nibble0);
        const float val1 = __shfl_sync(0xffffffff, lut_val, base_lane + nibble1);
        
        reinterpret_cast<half2*>(dst)[elem_idx / 2] = __floats2half2_rn(val0, val1);
    }
}

// Core dequant - bfloat16 output with LUT + shuffle
__device__ __forceinline__ void dequantize_block_q_awq_g64(
    const block_c_q_awq_g64_k128* __restrict__ src,
    __nv_bfloat16* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int group_idx = lane_id >> 4;
    
    const float scale = __half2float(src->scales[group_idx]);
    const float zero = __half2float(src->zeros[group_idx]);
    const int lut_idx = lane_id & 15;
    const float lut_val = scale * (float(lut_idx) - zero);
    
    const uint32_t* qs = src->qs;
    const int base_elem = group_idx * 64 + (lane_id & 15) * 4;
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int elem_idx = base_elem + i * 2;
        const int qs_idx = elem_idx / 8;
        const int nibble_offset = (elem_idx % 8) * 4;
        
        const uint32_t packed = qs[qs_idx];
        const int nibble0 = (packed >> nibble_offset) & 0xF;
        const int nibble1 = (packed >> (nibble_offset + 4)) & 0xF;
        
        const int base_lane = group_idx * 16;
        const float val0 = __shfl_sync(0xffffffff, lut_val, base_lane + nibble0);
        const float val1 = __shfl_sync(0xffffffff, lut_val, base_lane + nibble1);
        
        reinterpret_cast<__nv_bfloat162*>(dst)[elem_idx / 2] = __floats2bfloat162_rn(val0, val1);
    }
}

// Core dequant - FP8 E4M3 output with LUT + shuffle
__device__ __forceinline__ void dequantize_block_q_awq_g64(
    const block_c_q_awq_g64_k128* __restrict__ src,
    __nv_fp8_e4m3* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int group_idx = lane_id >> 4;
    
    const float scale = __half2float(src->scales[group_idx]);
    const float zero = __half2float(src->zeros[group_idx]);
    const int lut_idx = lane_id & 15;
    const float lut_val = scale * (float(lut_idx) - zero);
    
    const uint32_t* qs = src->qs;
    const int base_elem = group_idx * 64 + (lane_id & 15) * 4;
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int elem_idx = base_elem + i * 2;
        const int qs_idx = elem_idx / 8;
        const int nibble_offset = (elem_idx % 8) * 4;
        
        const uint32_t packed = qs[qs_idx];
        const int nibble0 = (packed >> nibble_offset) & 0xF;
        const int nibble1 = (packed >> (nibble_offset + 4)) & 0xF;
        
        const int base_lane = group_idx * 16;
        const float val0 = __shfl_sync(0xffffffff, lut_val, base_lane + nibble0);
        const float val1 = __shfl_sync(0xffffffff, lut_val, base_lane + nibble1);
        
        reinterpret_cast<__nv_fp8x2_e4m3*>(dst)[elem_idx / 2] = __nv_fp8x2_e4m3(make_float2(val0, val1));
    }
}

// Core dequant - generic compute_t output
template <typename compute_t>
__device__ __forceinline__ void dequantize_block_q_awq_g64(
    const block_c_q_awq_g64_k128* __restrict__ src,
    compute_t* __restrict__ dst) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int group_idx = lane_id >> 4;
    
    const float scale = __half2float(src->scales[group_idx]);
    const float zero = __half2float(src->zeros[group_idx]);
    const int lut_idx = lane_id & 15;
    const float lut_val = scale * (float(lut_idx) - zero);
    
    const uint32_t* qs = src->qs;
    const int base_elem = group_idx * 64 + (lane_id & 15) * 4;
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int elem_idx = base_elem + i * 2;
        const int qs_idx = elem_idx / 8;
        const int nibble_offset = (elem_idx % 8) * 4;
        
        const uint32_t packed = qs[qs_idx];
        const int nibble0 = (packed >> nibble_offset) & 0xF;
        const int nibble1 = (packed >> (nibble_offset + 4)) & 0xF;
        
        const int base_lane = group_idx * 16;
        const float val0 = __shfl_sync(0xffffffff, lut_val, base_lane + nibble0);
        const float val1 = __shfl_sync(0xffffffff, lut_val, base_lane + nibble1);
        
        dst[elem_idx] = from_f32<compute_t>(val0);
        dst[elem_idx + 1] = from_f32<compute_t>(val1);
    }
}

// Note: block_c_q_awq_g64 is typedef'd to block_c_q_awq_g64_k128, so no separate wrapper needed
