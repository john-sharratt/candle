// SPDX-License-Identifier: MIT
// Q5_0 Quantization: float -> 5-bit with scale only (OPTIMIZED)
//
// Q5_0 format stores:
//   - d (half): delta (scale)
//   - qh[4] (uint8_t): 5th bit of each quant (32 bits total)
//   - qs[16] (uint8_t): lower 4 bits packed (2 per byte)
//
// 5-bit symmetric quantization: maps to [-16, 15]
//   q = round(x / d + 16)  clamped to [0, 31]
//   x ≈ (q - 16) * d
//
// NOTE: Include via quantize.cuh which defines warp reduce functions.

#pragma once

// =============================================================================
// OPTIMIZED SINGLE-BLOCK QUANTIZATION (32 elements)
// =============================================================================

__device__ __forceinline__ void quantize_block_q5_0_vec(
    const float* __restrict__ src,
    block_q5_0* __restrict__ dst) {
    
    const int lane = threadIdx.x % WARP_SIZE;
    
    float4 v;
    float local_amax = 0.0f;
    float local_max_val = 0.0f;
    int local_max_idx = 0;
    
    if (lane < 8) {
        v = reinterpret_cast<const float4*>(src)[lane];
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
    
    // GGML convention: d = max_val / -16 (preserves sign)
    const float d = max_val / -16.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    
    // Quantize to 5-bit: round(x/d + 16), clamp to [0, 31]
    uint8_t q[4];
    uint8_t qh_bits = 0;
    
    if (lane < 8) {
        const int base_idx = lane * 4;
        
        int q0 = (int)fminf(31.0f, fmaxf(0.0f, v.x * id + 16.5f));
        int q1 = (int)fminf(31.0f, fmaxf(0.0f, v.y * id + 16.5f));
        int q2 = (int)fminf(31.0f, fmaxf(0.0f, v.z * id + 16.5f));
        int q3 = (int)fminf(31.0f, fmaxf(0.0f, v.w * id + 16.5f));
        
        // 5th bits: even lanes (0,2,4,6) pack bits 0-3; odd lanes (1,3,5,7) pack bits 4-7.
        // Two adjacent lanes share one qh byte: qh[lane/2] = even_bits | odd_bits.
        const int bit_start = (lane & 1) * 4;
        qh_bits = ((q0 >> 4) & 1) << bit_start;
        qh_bits |= ((q1 >> 4) & 1) << (bit_start + 1);
        qh_bits |= ((q2 >> 4) & 1) << (bit_start + 2);
        qh_bits |= ((q3 >> 4) & 1) << (bit_start + 3);
        
        // Lower 4 bits
        q[0] = q0 & 0xF;
        q[1] = q1 & 0xF;
        q[2] = q2 & 0xF;
        q[3] = q3 & 0xF;
    }
    
    // Pack lower 4 bits: element i pairs with element i+16
    // All lanes 0-7 must participate in shuffle for correctness
    uint8_t p0, p1, p2, p3;
    if (lane < 8) {
        p0 = __shfl_sync(0xff, q[0], (lane < 4) ? lane + 4 : lane, 8);
        p1 = __shfl_sync(0xff, q[1], (lane < 4) ? lane + 4 : lane, 8);
        p2 = __shfl_sync(0xff, q[2], (lane < 4) ? lane + 4 : lane, 8);
        p3 = __shfl_sync(0xff, q[3], (lane < 4) ? lane + 4 : lane, 8);
    }
    if (lane < 4) {
        // Pack and store byte-by-byte to avoid alignment issues
        dst->qs[lane * 4 + 0] = q[0] | (p0 << 4);
        dst->qs[lane * 4 + 1] = q[1] | (p1 << 4);
        dst->qs[lane * 4 + 2] = q[2] | (p2 << 4);
        dst->qs[lane * 4 + 3] = q[3] | (p3 << 4);
    }
    
    // Combine qh bits: adjacent lane pairs (0+1, 2+3, 4+5, 6+7) share one qh byte.
    uint8_t qh_partner;
    if (lane < 8) {
        qh_partner = __shfl_sync(0xff, qh_bits, lane ^ 1, 8);
    }
    if (lane < 8 && (lane & 1) == 0) {
        dst->qh[lane >> 1] = qh_bits | qh_partner;
    }
    
    if (lane == 0) {
        dst->d = __float2half_rn(d);
    }
}

// =============================================================================
// SCALAR FALLBACK (FAST VERSION)
// Q5_0: d = max_val/-16, q = floor(x/d + 16.5) clamped to [0,31]
// max_val is the element with the largest absolute value, sign preserved.

__device__ __forceinline__ void quantize_block_q5_0(
    const float* __restrict__ src,
    block_q5_0* __restrict__ dst) {

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

    // GGML sign convention: d = max_val / -16
    const float d = max_val / -16.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    
    // Quantize to 5 bits [0, 31]
    const int q5 = (int)fminf(31.0f, fmaxf(0.0f, xi * id + 16.5f));
    const uint8_t q4 = q5 & 0xF;
    const uint8_t qh_bit = (q5 >> 4) & 1;
    
    // Pack lower 4 bits
    const uint8_t q4_partner = __shfl_sync(0xffffffff, q4, lane ^ 16, 32);
    if (lane < 16) {
        dst->qs[lane] = q4 | (q4_partner << 4);
    }
    
    // Pack 5th bits into qh[4]
    // Each byte of qh holds bits for 8 consecutive elements
    const int qh_byte = lane / 8;
    const int qh_shift = lane % 8;
    const uint32_t qh_mask = qh_bit << qh_shift;
    
    // Reduce within each group of 8 lanes
    uint32_t qh_combined = qh_mask;
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        qh_combined |= __shfl_xor_sync(0xffffffff, qh_combined, offset, 8);
    }
    
    if (lane % 8 == 0) {
        dst->qh[qh_byte] = (uint8_t)qh_combined;
    }
    
    if (lane == 0) {
        dst->d = __float2half_rn(d);
    }
}

// =============================================================================
// MULTI-BLOCK QUANTIZATION
// =============================================================================

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q5_0(
    const float* __restrict__ src,
    block_q5_0* __restrict__ dst,
    int num_blocks) {
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    for (int blk = warp_id + blockIdx.x * warps_per_block; 
         blk < num_blocks; 
         blk += warps_per_block * gridDim.x) {
        
        const float* block_src = src + blk * QK5_0;
        block_q5_0* block_dst = dst + blk;
        
        const float xi = block_src[lane];
        
        // Find value with maximum magnitude with tie-breaking
        float amax = fabsf(xi);
        float max_val = xi;
        int max_idx = lane;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_amax = __shfl_xor_sync(0xffffffff, amax, offset, 32);
            float other_val = __shfl_xor_sync(0xffffffff, max_val, offset, 32);
            int other_idx = __shfl_xor_sync(0xffffffff, max_idx, offset, 32);
            if (other_amax > amax || (other_amax == amax && other_idx < max_idx)) {
                amax = other_amax;
                max_val = other_val;
                max_idx = other_idx;
            }
        }
        
        const float d = max_val / -16.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        
        const int q5 = (int)fminf(31.0f, fmaxf(0.0f, xi * id + 16.5f));
        const uint8_t q4 = q5 & 0xF;
        const uint8_t qh_bit = (q5 >> 4) & 1;
        
        const uint8_t q4_partner = __shfl_sync(0xffffffff, q4, lane ^ 16, 32);
        if (lane < 16) {
            block_dst->qs[lane] = q4 | (q4_partner << 4);
        }
        
        const int qh_byte = lane / 8;
        const int qh_shift = lane % 8;
        uint32_t qh_combined = qh_bit << qh_shift;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            qh_combined |= __shfl_xor_sync(0xffffffff, qh_combined, offset, 8);
        }
        
        if (lane % 8 == 0) {
            block_dst->qh[qh_byte] = (uint8_t)qh_combined;
        }
        
        if (lane == 0) {
            block_dst->d = __float2half_rn(d);
        }
    }
}
