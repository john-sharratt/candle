// SPDX-License-Identifier: MIT
// Q_AWQ Quantization: float -> 4-bit AWQ with group size 128
//
// AWQ (Activation-aware Weight Quantization) format (80 bytes, 16-byte aligned):
//   - qs[16] (uint32_t): 128 × 4-bit nibbles packed (8 nibbles per u32)
//   - scale (half): scale factor
//   - zero (half): zero point
//   - _pad[3] (uint32_t): padding to 80 bytes
//
// CPU algorithm (k_quants.rs):
//   1. Find min/max in block
//   2. scale = (max - min) / 15, zero = -min / scale
//   3. q = round((x / scale) + zero), clamped to [0, 15]
//   4. Pack: qs[t] = nibble[t*8+0] | (nibble[t*8+1] << 4) | ... | (nibble[t*8+7] << 28)
//
// NOTE: Include via quantize.cuh which defines warp reduce functions.

#pragma once

#ifndef QK_Q_AWQ
#define QK_Q_AWQ 128
#endif

// =============================================================================
// Q_AWQ QUANTIZATION - Match CPU exactly (80-byte block with u32 packing)
// =============================================================================

__device__ __forceinline__ void quantize_block_q_awq(
    const float* __restrict__ src,
    block_q_awq* __restrict__ dst) {
    
    const int lane = threadIdx.x % WARP_SIZE;
    
    // Each thread handles 4 elements (128/32 = 4)
    float vals[4];
    for (int i = 0; i < 4; i++) {
        vals[i] = src[lane * 4 + i];
    }
    
    // Find local min/max
    float local_min = vals[0];
    float local_max = vals[0];
    for (int i = 1; i < 4; i++) {
        local_min = fminf(local_min, vals[i]);
        local_max = fmaxf(local_max, vals[i]);
    }
    
    // Reduce across warp to find global min/max
    float vmin = local_min;
    float vmax = local_max;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        vmin = fminf(vmin, __shfl_xor_sync(0xffffffff, vmin, offset, 32));
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, offset, 32));
    }
    
    // Compute scale and zero - matching CPU exactly
    float range = vmax - vmin;
    float scale, zero, inv_scale;
    if (range < 1e-10f) {
        scale = 0.0f;
        zero = 0.0f;
        inv_scale = 0.0f;
    } else {
        scale = range / 15.0f;
        zero = -vmin / scale;
        inv_scale = 1.0f / scale;
    }
    
    // Store scale, zero, and padding (only lane 0)
    if (lane == 0) {
        dst->scale = __float2half_rn(scale);
        dst->zero = __float2half_rn(zero);
        dst->_pad[0] = 0;
        dst->_pad[1] = 0;
        dst->_pad[2] = 0;
    }
    
    // Quantize values to shared memory
    __shared__ uint8_t shared_q[128];
    for (int i = 0; i < 4; i++) {
        float q = vals[i] * inv_scale + zero;
        q = roundf(q);
        q = fmaxf(0.0f, fminf(15.0f, q));
        shared_q[lane * 4 + i] = (uint8_t)q;
    }
    __syncwarp();
    
    // Pack into u32: lanes 0-15 each write one u32 containing 8 nibbles
    // qs[t] contains elements t*8..t*8+7, nibble j at bits j*4
    if (lane < 16) {
        uint32_t packed = 0;
        for (int j = 0; j < 8; j++) {
            packed |= ((uint32_t)shared_q[lane * 8 + j]) << (j * 4);
        }
        dst->qs[lane] = packed;
    }
}

// =============================================================================
// MULTI-BLOCK QUANTIZATION
// =============================================================================

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q_awq(
    const float* __restrict__ src,
    block_q_awq* __restrict__ dst,
    int num_blocks) {
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    // Shared memory for packing (one per warp)
    __shared__ uint8_t shared_q[4][128];  // Up to 4 warps
    uint8_t* warp_q = shared_q[warp_id % 4];
    
    for (int blk = warp_id + blockIdx.x * warps_per_block; 
         blk < num_blocks; 
         blk += warps_per_block * gridDim.x) {
        
        const float* block_src = src + blk * QK_Q_AWQ;
        block_q_awq* block_dst = dst + blk;
        
        // Load 4 values per thread
        float vals[4];
        for (int i = 0; i < 4; i++) {
            vals[i] = block_src[lane * 4 + i];
        }
        
        // Find min/max
        float local_min = vals[0];
        float local_max = vals[0];
        for (int i = 1; i < 4; i++) {
            local_min = fminf(local_min, vals[i]);
            local_max = fmaxf(local_max, vals[i]);
        }
        
        float vmin = local_min;
        float vmax = local_max;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            vmin = fminf(vmin, __shfl_xor_sync(0xffffffff, vmin, offset, 32));
            vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, offset, 32));
        }
        
        float range = vmax - vmin;
        float scale, zero, inv_scale;
        if (range < 1e-10f) {
            scale = 0.0f;
            zero = 0.0f;
            inv_scale = 0.0f;
        } else {
            scale = range / 15.0f;
            zero = -vmin / scale;
            inv_scale = 1.0f / scale;
        }
        
        if (lane == 0) {
            block_dst->scale = __float2half_rn(scale);
            block_dst->zero = __float2half_rn(zero);
            block_dst->_pad[0] = 0;
            block_dst->_pad[1] = 0;
            block_dst->_pad[2] = 0;
        }
        
        // Quantize
        for (int i = 0; i < 4; i++) {
            float q = vals[i] * inv_scale + zero;
            q = roundf(q);
            q = fmaxf(0.0f, fminf(15.0f, q));
            warp_q[lane * 4 + i] = (uint8_t)q;
        }
        __syncwarp();
        
        // Pack into u32: lanes 0-15 each write one u32
        if (lane < 16) {
            uint32_t packed = 0;
            for (int j = 0; j < 8; j++) {
                packed |= ((uint32_t)warp_q[lane * 8 + j]) << (j * 4);
            }
            block_dst->qs[lane] = packed;
        }
    }
}
