// SPDX-License-Identifier: MIT
// Q4_1 Quantization: float -> 4-bit with scale and min (OPTIMIZED)
//
// Q4_1 format stores:
//   - dm.x (half): delta (scale)
//   - dm.y (half): min value
//   - qs[16] (uint8_t): packed 4-bit values (2 per byte)
//
// Unlike Q4_0 (symmetric), Q4_1 uses asymmetric quantization:
//   q = round((x - min) / delta)  where delta = (max - min) / 15
//   x ≈ q * delta + min
//
// NOTE: Include via quantize.cuh which defines warp reduce functions.

#pragma once

// quantize_warp_reduce_min is defined in quantize.cuh

// =============================================================================
// OPTIMIZED SINGLE-BLOCK QUANTIZATION (32 elements)
// =============================================================================

__device__ __forceinline__ void quantize_block_q4_1_vec(
    const float* __restrict__ src,
    block_q4_1* __restrict__ dst) {
    
    const int lane = threadIdx.x % WARP_SIZE;
    
    float4 v;
    float local_max = -3.402823466e+38f;
    float local_min = 3.402823466e+38f;
    
    if (lane < 8) {
        v = reinterpret_cast<const float4*>(src)[lane];
        local_max = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
        local_min = fminf(fminf(v.x, v.y), fminf(v.z, v.w));
    }
    
    // Reduce across first 8 lanes
    float vmax = local_max;
    float vmin = local_min;
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(0xff, vmax, offset, 8));
        vmin = fminf(vmin, __shfl_xor_sync(0xff, vmin, offset, 8));
    }
    vmax = __shfl_sync(0xffffffff, vmax, 0, 32);
    vmin = __shfl_sync(0xffffffff, vmin, 0, 32);
    
    // Compute scale and min
    const float d = (vmax - vmin) * (1.0f / 15.0f);
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    
    uint8_t q[4];
    if (lane < 8) {
        q[0] = (uint8_t)fminf(15.0f, fmaxf(0.0f, (v.x - vmin) * id + 0.5f));
        q[1] = (uint8_t)fminf(15.0f, fmaxf(0.0f, (v.y - vmin) * id + 0.5f));
        q[2] = (uint8_t)fminf(15.0f, fmaxf(0.0f, (v.z - vmin) * id + 0.5f));
        q[3] = (uint8_t)fminf(15.0f, fmaxf(0.0f, (v.w - vmin) * id + 0.5f));
    }
    
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
        dst->qs[lane * 4 + 0] = q[0] | (p0 << 4);
        dst->qs[lane * 4 + 1] = q[1] | (p1 << 4);
        dst->qs[lane * 4 + 2] = q[2] | (p2 << 4);
        dst->qs[lane * 4 + 3] = q[3] | (p3 << 4);
    }
    
    if (lane == 0) {
        dst->dm = make_half2(__float2half_rn(d), __float2half_rn(vmin));
    }
}

// =============================================================================
// SCALAR FALLBACK
// =============================================================================

__device__ __forceinline__ void quantize_block_q4_1(
    const float* __restrict__ src,
    block_q4_1* __restrict__ dst) {
    
    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];
    
    float vmax = xi;
    float vmin = xi;
    vmax = quantize_warp_reduce_max(vmax);
    vmin = quantize_warp_reduce_min(vmin);
    
    const float d = (vmax - vmin) * (1.0f / 15.0f);
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    
    const uint8_t q4 = (uint8_t)fminf(15.0f, fmaxf(0.0f, (xi - vmin) * id + 0.5f));
    const uint8_t q4_partner = __shfl_sync(0xffffffff, q4, lane ^ 16, 32);
    
    if (lane < 16) {
        dst->qs[lane] = q4 | (q4_partner << 4);
    }
    
    if (lane == 0) {
        dst->dm = make_half2(__float2half_rn(d), __float2half_rn(vmin));
    }
}

// =============================================================================
// MULTI-BLOCK QUANTIZATION
// =============================================================================

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q4_1(
    const float* __restrict__ src,
    block_q4_1* __restrict__ dst,
    int num_blocks) {
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    for (int blk = warp_id + blockIdx.x * warps_per_block; 
         blk < num_blocks; 
         blk += warps_per_block * gridDim.x) {
        
        const float* block_src = src + blk * QK4_1;
        block_q4_1* block_dst = dst + blk;
        
        float4 v;
        float local_max = -3.402823466e+38f;
        float local_min = 3.402823466e+38f;
        
        if (lane < 8) {
            v = reinterpret_cast<const float4*>(block_src)[lane];
            local_max = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
            local_min = fminf(fminf(v.x, v.y), fminf(v.z, v.w));
        }
        
        float vmax = local_max;
        float vmin = local_min;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            vmax = fmaxf(vmax, __shfl_xor_sync(0xff, vmax, offset, 8));
            vmin = fminf(vmin, __shfl_xor_sync(0xff, vmin, offset, 8));
        }
        vmax = __shfl_sync(0xffffffff, vmax, 0, 32);
        vmin = __shfl_sync(0xffffffff, vmin, 0, 32);
        
        const float d = (vmax - vmin) * (1.0f / 15.0f);
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        
        uint8_t q[4];
        if (lane < 8) {
            q[0] = (uint8_t)fminf(15.0f, fmaxf(0.0f, (v.x - vmin) * id + 0.5f));
            q[1] = (uint8_t)fminf(15.0f, fmaxf(0.0f, (v.y - vmin) * id + 0.5f));
            q[2] = (uint8_t)fminf(15.0f, fmaxf(0.0f, (v.z - vmin) * id + 0.5f));
            q[3] = (uint8_t)fminf(15.0f, fmaxf(0.0f, (v.w - vmin) * id + 0.5f));
        }
        
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
            block_dst->qs[lane * 4 + 0] = q[0] | (p0 << 4);
            block_dst->qs[lane * 4 + 1] = q[1] | (p1 << 4);
            block_dst->qs[lane * 4 + 2] = q[2] | (p2 << 4);
            block_dst->qs[lane * 4 + 3] = q[3] | (p3 << 4);
        }
        
        if (lane == 0) {
            block_dst->dm = make_half2(__float2half_rn(d), __float2half_rn(vmin));
        }
    }
}
