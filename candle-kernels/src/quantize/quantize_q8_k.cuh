// SPDX-License-Identifier: MIT
// Q8_K Quantization: float -> 8-bit K-quant (256 elements per super-block)
//
// Q8_K format stores:
//   - d (float): super-block scale (NOTE: f32 not f16!)
//   - qs[256] (int8_t): quantized values
//   - bsums[16] (int16_t): sum of quants in groups of 16
//
// This is the highest precision K-quant, used as activation format for
// K-quant matrix multiplications.
//
// CPU algorithm from k_quants.rs:
//   1. Find max by absolute value (tracking sign)
//   2. iscale = -128 / max (negative for sign preservation)
//   3. quantize: q = round(iscale * x), clamped to max 127
//   4. d = 1 / iscale
//
// NOTE: Include via quantize.cuh which defines warp reduce functions.

#pragma once

#ifndef QK_K
#define QK_K 256
#endif

// =============================================================================
// Q8_K QUANTIZATION - Match CPU exactly
// =============================================================================
// CPU uses iscale = -128/max (not 127/amax), tracking signed max

__device__ __forceinline__ void quantize_block_q8_K(
    const float* __restrict__ src,
    block_q8_K* __restrict__ dst) {
    
    const int lane = threadIdx.x % WARP_SIZE;
    
    // Load 8 elements per thread via float4 (2 loads)
    float4 v0 = reinterpret_cast<const float4*>(src)[lane * 2];
    float4 v1 = reinterpret_cast<const float4*>(src)[lane * 2 + 1];
    
    // Find local max by absolute value, tracking the actual signed value
    float vals[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    float local_max = 0.0f;
    float local_amax = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float ax = fabsf(vals[i]);
        if (ax > local_amax) {
            local_amax = ax;
            local_max = vals[i];
        }
    }
    
    // Reduce to find max across warp - need to track both amax and signed max
    // Use a two-phase reduction: first amax, then find who has it
    float amax = local_amax;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, amax, offset, 32);
        amax = fmaxf(amax, other);
    }
    
    // Now find which lane has the max and get its signed value
    int winner_lane = -1;
    if (local_amax == amax) {
        winner_lane = lane;
    }
    // Reduce to find the lowest lane with the max (deterministic tie-breaking)
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_xor_sync(0xffffffff, winner_lane, offset, 32);
        if (other >= 0 && (winner_lane < 0 || other < winner_lane)) {
            winner_lane = other;
        }
    }
    // Broadcast the signed max from the winner
    float max_val = __shfl_sync(0xffffffff, local_max, winner_lane, 32);
    
    // CPU algorithm: iscale = -128 / max, d = 1 / iscale
    float d_val, id;
    if (amax == 0.0f) {
        d_val = 0.0f;
        id = 0.0f;
    } else {
        float iscale = -128.0f / max_val;
        d_val = 1.0f / iscale;
        id = iscale;
    }
    
    // Quantize 8 values: q = round(iscale * x), clamp to max 127
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float v = roundf(id * vals[i]);
        // CPU clamps to min(127, v) only - since iscale is negative, values are bounded
        int q = (int)fminf(127.0f, v);
        dst->qs[lane * 8 + i] = (int8_t)q;
    }
    
    // Compute bsums: sum of quants in groups of 16
    // Each group of 16 elements = 2 threads × 8 elements
    int local_sum = 0;
    for (int i = 0; i < 8; i++) {
        local_sum += dst->qs[lane * 8 + i];
    }
    
    // Reduce with partner (adjacent thread)
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, 1, 32);
    
    // Store bsum (every other thread)
    if (lane % 2 == 0) {
        dst->bsums[lane / 2] = (int16_t)local_sum;
    }
    
    // Store scale (f32)
    if (lane == 0) {
        dst->d = d_val;
    }
}

// =============================================================================
// MULTI-BLOCK QUANTIZATION
// =============================================================================

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q8_K(
    const float* __restrict__ src,
    block_q8_K* __restrict__ dst,
    int num_blocks) {
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    for (int blk = warp_id + blockIdx.x * warps_per_block; 
         blk < num_blocks; 
         blk += warps_per_block * gridDim.x) {
        
        const float* block_src = src + blk * QK_K;
        block_q8_K* block_dst = dst + blk;
        
        float4 v0 = reinterpret_cast<const float4*>(block_src)[lane * 2];
        float4 v1 = reinterpret_cast<const float4*>(block_src)[lane * 2 + 1];
        
        float vals[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        float local_max = 0.0f;
        float local_amax = 0.0f;
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float ax = fabsf(vals[i]);
            if (ax > local_amax) {
                local_amax = ax;
                local_max = vals[i];
            }
        }
        
        float amax = local_amax;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, amax, offset, 32);
            amax = fmaxf(amax, other);
        }
        
        int winner_lane = (local_amax == amax) ? lane : 32;
        for (int offset = 16; offset > 0; offset >>= 1) {
            int other = __shfl_xor_sync(0xffffffff, winner_lane, offset, 32);
            winner_lane = min(winner_lane, other);
        }
        float max_val = __shfl_sync(0xffffffff, local_max, winner_lane, 32);
        
        float d_val, id;
        if (amax == 0.0f) {
            d_val = 0.0f;
            id = 0.0f;
        } else {
            float iscale = -128.0f / max_val;
            d_val = 1.0f / iscale;
            id = iscale;
        }
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float v = roundf(id * vals[i]);
            int q = (int)fminf(127.0f, v);
            block_dst->qs[lane * 8 + i] = (int8_t)q;
        }
        
        int local_sum = 0;
        for (int i = 0; i < 8; i++) {
            local_sum += block_dst->qs[lane * 8 + i];
        }
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, 1, 32);
        
        if (lane % 2 == 0) {
            block_dst->bsums[lane / 2] = (int16_t)local_sum;
        }
        
        if (lane == 0) {
            block_dst->d = d_val;
        }
    }
}
