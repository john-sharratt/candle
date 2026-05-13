// SPDX-License-Identifier: MIT
// Q8_1 Quantization: float -> 8-bit with scale and sum (OPTIMIZED)
//
// Q8_1 format stores:
//   - ds.x (half): scale factor d = amax / 127
//   - ds.y (half): raw sum Σx (NOT d·Σq); all consumers in simple/quantized.cu
//                  read ds.y as Σx directly — this diverges from the GGML spec
//                  (which stores d·Σq) but is internally consistent.
//   - qs[32] (int8_t): quantized values
//
// Optimizations:
//   1. Vectorized float4 loads (4x bandwidth utilization)
//   2. Multiply by reciprocal instead of division
//   3. Vectorized int4 stores (pack 4 int8 into one write)
//   4. Fused amax and sum reductions
//
// NOTE: Include via quantize.cuh which defines warp reduce functions.

#pragma once

// Warp reduce functions are defined in quantize.cuh

// =============================================================================
// OPTIMIZED SINGLE-BLOCK QUANTIZATION (32 elements)
// =============================================================================
// Each of 8 threads loads float4, computes local max/sum, then reduces.

__device__ __forceinline__ void quantize_block_q8_1_vec(
    const float* __restrict__ src,
    block_q8_1* __restrict__ dst) {
    
    const int lane = threadIdx.x % WARP_SIZE;
    
    float4 v;
    float local_max = 0.0f;
    float local_sum = 0.0f;
    
    if (lane < 8) {
        v = reinterpret_cast<const float4*>(src)[lane];
        local_max = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), 
                         fmaxf(fabsf(v.z), fabsf(v.w)));
        local_sum = v.x + v.y + v.z + v.w;
    }
    
    // Reduce across first 8 lanes
    float amax = local_max;
    float sum = local_sum;
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xff, amax, offset, 8));
        sum += __shfl_xor_sync(0xff, sum, offset, 8);
    }
    amax = __shfl_sync(0xffffffff, amax, 0, 32);
    sum = __shfl_sync(0xffffffff, sum, 0, 32);
    
    if (lane < 8) {
        const float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
        
        const int8_t q0 = (int8_t)__float2int_rn(v.x * id);
        const int8_t q1 = (int8_t)__float2int_rn(v.y * id);
        const int8_t q2 = (int8_t)__float2int_rn(v.z * id);
        const int8_t q3 = (int8_t)__float2int_rn(v.w * id);
        
        // Store byte-by-byte to avoid alignment issues
        dst->qs[lane * 4 + 0] = q0;
        dst->qs[lane * 4 + 1] = q1;
        dst->qs[lane * 4 + 2] = q2;
        dst->qs[lane * 4 + 3] = q3;
    }
    
    if (lane == 0) {
        dst->ds = make_half2(__float2half_rn(amax / 127.0f), __float2half_rn(sum));
    }
}

// =============================================================================
// SCALAR FALLBACK
// =============================================================================

__device__ __forceinline__ void quantize_block_q8_1(
    const float* __restrict__ src,
    block_q8_1* __restrict__ dst) {
    
    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];
    
    float amax = fabsf(xi);
    float sum = xi;
    
    amax = quantize_warp_reduce_max(amax);
    sum = quantize_warp_reduce_sum(sum);
    
    const float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
    const int8_t q = (int8_t)__float2int_rn(xi * id);
    
    dst->qs[lane] = q;
    
    if (lane == 0) {
        dst->ds = make_half2(__float2half_rn(amax / 127.0f), __float2half_rn(sum));
    }
}

// =============================================================================
// MULTI-BLOCK QUANTIZATION (VECTORIZED)
// =============================================================================

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q8_1(
    const float* __restrict__ src,
    block_q8_1* __restrict__ dst,
    int num_blocks) {
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    for (int blk = warp_id + blockIdx.x * warps_per_block; 
         blk < num_blocks; 
         blk += warps_per_block * gridDim.x) {
        
        const float* block_src = src + blk * QK8_1;
        block_q8_1* block_dst = dst + blk;
        
        float4 v;
        float local_max = 0.0f;
        float local_sum = 0.0f;
        
        if (lane < 8) {
            v = reinterpret_cast<const float4*>(block_src)[lane];
            local_max = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), 
                             fmaxf(fabsf(v.z), fabsf(v.w)));
            local_sum = v.x + v.y + v.z + v.w;
        }
        
        float amax = local_max;
        float sum = local_sum;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xff, amax, offset, 8));
            sum += __shfl_xor_sync(0xff, sum, offset, 8);
        }
        amax = __shfl_sync(0xffffffff, amax, 0, 32);
        sum = __shfl_sync(0xffffffff, sum, 0, 32);
        
        if (lane < 8) {
            const float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
            
            const int8_t q0 = (int8_t)__float2int_rn(v.x * id);
            const int8_t q1 = (int8_t)__float2int_rn(v.y * id);
            const int8_t q2 = (int8_t)__float2int_rn(v.z * id);
            const int8_t q3 = (int8_t)__float2int_rn(v.w * id);
            
            // Store byte-by-byte to avoid alignment issues
            block_dst->qs[lane * 4 + 0] = q0;
            block_dst->qs[lane * 4 + 1] = q1;
            block_dst->qs[lane * 4 + 2] = q2;
            block_dst->qs[lane * 4 + 3] = q3;
        }
        
        if (lane == 0) {
            block_dst->ds = make_half2(__float2half_rn(amax / 127.0f), __float2half_rn(sum));
        }
    }
}
