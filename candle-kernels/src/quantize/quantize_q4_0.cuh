// SPDX-License-Identifier: MIT
// Q4_0 Quantization: float -> 4-bit with scale only (OPTIMIZED)
//
// Q4_0 format stores:
//   - d (half): scale factor (negative, GGML convention)
//   - qs[16] (uint8_t): packed 4-bit values (2 per byte)
//
// Block size: 32 elements packed into 16 bytes + 2 byte scale = 18 bytes
// Compression: 4.5 bits per element (vs 32 bits for float)
//
// Optimizations:
//   1. Vectorized float4 loads
//   2. Multiply by reciprocal instead of division
//   3. Warp shuffle for 4-bit packing (no shared memory)
//   4. Vectorized uint64 store for packed nibbles
//
// NOTE: Include via quantize.cuh which defines warp reduce functions.

#pragma once

// Warp reduce functions are defined in quantize.cuh

// =============================================================================
// OPTIMIZED SINGLE-BLOCK QUANTIZATION (32 elements)
// =============================================================================
// 8 threads load float4 each, all 32 threads participate in packing.

__device__ __forceinline__ void quantize_block_q4_0_vec(
    const float* __restrict__ src,
    block_q4_0* __restrict__ dst) {
    
    const int lane = threadIdx.x % WARP_SIZE;
    
    // Load via float4: 8 threads load 4 floats each = 32 floats
    float4 v;
    float local_amax = 0.0f;
    float local_max_val = 0.0f;
    int local_max_idx = 0;  // Global element index
    
    if (lane < 8) {
        v = reinterpret_cast<const float4*>(src)[lane];
        // Find local max magnitude and corresponding value
        // Prefer lower index on tie (like CPU sequential scan)
        float vals[4] = {v.x, v.y, v.z, v.w};
        for (int i = 0; i < 4; i++) {
            float a = fabsf(vals[i]);
            int idx = lane * 4 + i;  // Global element index
            if (a > local_amax || (a == local_amax && idx < local_max_idx)) {
                local_amax = a;
                local_max_val = vals[i];
                local_max_idx = idx;
            }
        }
    }
    
    // Reduce across first 8 lanes: find max magnitude and its value
    float amax = local_amax;
    float max_val = local_max_val;
    int max_idx = local_max_idx;
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        float other_amax = __shfl_xor_sync(0xff, amax, offset, 8);
        float other_val = __shfl_xor_sync(0xff, max_val, offset, 8);
        int other_idx = __shfl_xor_sync(0xff, max_idx, offset, 8);
        if (other_amax > amax || (other_amax == amax && other_idx < max_idx)) {
            amax = other_amax;
            max_val = other_val;
            max_idx = other_idx;
        }
    }
    // Broadcast to all lanes
    max_val = __shfl_sync(0xffffffff, max_val, 0, 32);
    
    // GGML convention: d = max_val / -8 (preserves sign)
    const float d = max_val / -8.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    
    // Each of first 8 threads quantizes 4 values
    uint8_t q[4];
    if (lane < 8) {
        q[0] = (uint8_t)fminf(15.0f, fmaxf(0.0f, v.x * id + 8.5f));
        q[1] = (uint8_t)fminf(15.0f, fmaxf(0.0f, v.y * id + 8.5f));
        q[2] = (uint8_t)fminf(15.0f, fmaxf(0.0f, v.z * id + 8.5f));
        q[3] = (uint8_t)fminf(15.0f, fmaxf(0.0f, v.w * id + 8.5f));
    }
    
    // Pack pairs: element i pairs with element i+16
    // All lanes 0-7 must participate in the shuffle
    uint8_t p0, p1, p2, p3;
    if (lane < 8) {
        p0 = __shfl_sync(0xff, q[0], (lane < 4) ? lane + 4 : lane, 8);
        p1 = __shfl_sync(0xff, q[1], (lane < 4) ? lane + 4 : lane, 8);
        p2 = __shfl_sync(0xff, q[2], (lane < 4) ? lane + 4 : lane, 8);
        p3 = __shfl_sync(0xff, q[3], (lane < 4) ? lane + 4 : lane, 8);
    }
    
    if (lane < 4) {
        // Pack and store byte-by-byte to avoid alignment issues
        // (block_q4_0 is 18 bytes with 2-byte alignment, so qs may not be 4-byte aligned)
        dst->qs[lane * 4 + 0] = q[0] | (p0 << 4);
        dst->qs[lane * 4 + 1] = q[1] | (p1 << 4);
        dst->qs[lane * 4 + 2] = q[2] | (p2 << 4);
        dst->qs[lane * 4 + 3] = q[3] | (p3 << 4);
    }
    
    if (lane == 0) {
        dst->d = __float2half_rn(d);
    }
}
// =============================================================================
// SCALAR FALLBACK (FAST VERSION for transpose+quantize)
// =============================================================================
// Q4_0: d = max_val/-8, q = floor(x/d + 8.5) clamped to [0,15]
// max_val is the element with the largest absolute value, sign preserved.
// This matches the GGML convention used by quantize_blocks_q4_0.

__device__ __forceinline__ void quantize_block_q4_0(
    const float* __restrict__ src,
    block_q4_0* __restrict__ dst) {

    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];

    // Find the element with the largest absolute value (keep its sign).
    float amax = fabsf(xi);
    float max_val = xi;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_amax = __shfl_xor_sync(0xffffffff, amax, offset, 32);
        float other_val  = __shfl_xor_sync(0xffffffff, max_val, offset, 32);
        if (other_amax > amax) { amax = other_amax; max_val = other_val; }
    }
    max_val = __shfl_sync(0xffffffff, max_val, 0, 32);

    // GGML sign convention: d = max_val / -8  (positive if block is negative-dominant)
    const float d = max_val / -8.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    
    // Quantize: q = round(x * id + 8) clamped to [0,15]
    const uint8_t q4 = (uint8_t)fminf(15.0f, fmaxf(0.0f, xi * id + 8.5f));
    
    // Pack pairs via shuffle: element i pairs with element i+16
    const uint8_t q4_partner = __shfl_sync(0xffffffff, q4, lane ^ 16, 32);
    
    if (lane < 16) {
        dst->qs[lane] = q4 | (q4_partner << 4);
    }
    
    if (lane == 0) {
        dst->d = __float2half_rn(d);
    }
}

// =============================================================================
// MULTI-BLOCK QUANTIZATION (VECTORIZED)
// =============================================================================

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q4_0(
    const float* __restrict__ src,
    block_q4_0* __restrict__ dst,
    int num_blocks) {
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    for (int blk = warp_id + blockIdx.x * warps_per_block; 
         blk < num_blocks; 
         blk += warps_per_block * gridDim.x) {
        
        const float* block_src = src + blk * QK4_0;
        block_q4_0* block_dst = dst + blk;
        
        float4 v;
        float local_amax = 0.0f;
        float local_max_val = 0.0f;
        int local_max_idx = 0;
        
        if (lane < 8) {
            v = reinterpret_cast<const float4*>(block_src)[lane];
            float vals[4] = {v.x, v.y, v.z, v.w};
            for (int i = 0; i < 4; i++) {
                float a = fabsf(vals[i]);
                int idx = lane * 4 + i;
                if (a > local_amax || (a == local_amax && idx < local_max_idx)) {
                    local_amax = a;
                    local_max_val = vals[i];
                    local_max_idx = idx;
                }
            }
        }
        
        float amax = local_amax;
        float max_val = local_max_val;
        int max_idx = local_max_idx;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            float other_amax = __shfl_xor_sync(0xff, amax, offset, 8);
            float other_val = __shfl_xor_sync(0xff, max_val, offset, 8);
            int other_idx = __shfl_xor_sync(0xff, max_idx, offset, 8);
            if (other_amax > amax || (other_amax == amax && other_idx < max_idx)) {
                amax = other_amax;
                max_val = other_val;
                max_idx = other_idx;
            }
        }
        max_val = __shfl_sync(0xffffffff, max_val, 0, 32);
        
        const float d = max_val / -8.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        
        uint8_t q[4];
        if (lane < 8) {
            q[0] = (uint8_t)fminf(15.0f, fmaxf(0.0f, v.x * id + 8.5f));
            q[1] = (uint8_t)fminf(15.0f, fmaxf(0.0f, v.y * id + 8.5f));
            q[2] = (uint8_t)fminf(15.0f, fmaxf(0.0f, v.z * id + 8.5f));
            q[3] = (uint8_t)fminf(15.0f, fmaxf(0.0f, v.w * id + 8.5f));
        }
        
        // All lanes in the first 8 must participate in the shuffle
        // Shuffle within groups of 8: lanes 0-3 get values from lanes 4-7
        uint8_t p0, p1, p2, p3;
        if (lane < 8) {
            p0 = __shfl_sync(0xff, q[0], (lane < 4) ? lane + 4 : lane, 8);
            p1 = __shfl_sync(0xff, q[1], (lane < 4) ? lane + 4 : lane, 8);
            p2 = __shfl_sync(0xff, q[2], (lane < 4) ? lane + 4 : lane, 8);
            p3 = __shfl_sync(0xff, q[3], (lane < 4) ? lane + 4 : lane, 8);
        }
        
        if (lane < 4) {
            // Pack and store byte-by-byte to avoid alignment issues
            block_dst->qs[lane * 4 + 0] = q[0] | (p0 << 4);
            block_dst->qs[lane * 4 + 1] = q[1] | (p1 << 4);
            block_dst->qs[lane * 4 + 2] = q[2] | (p2 << 4);
            block_dst->qs[lane * 4 + 3] = q[3] | (p3 << 4);
        }
        
        if (lane == 0) {
            block_dst->d = __float2half_rn(d);
        }
    }
}
