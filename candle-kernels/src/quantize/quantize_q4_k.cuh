// SPDX-License-Identifier: MIT
// Q4_K Quantization: float -> 4-bit K-quant (256 elements per super-block)
//
// Q4_K format from GGML (follows candle-core/src/quantized/k_quants.rs):
//   - d (half): super-block scale
//   - dmin (half): super-block min scale
//   - scales[12] (uint8_t): 6-bit scales and mins for 8 sub-blocks, packed
//   - qs[128] (uint8_t): 4-bit quants, packed (2 per byte)
//
// Structure: 256 elements in 8 sub-blocks of 32 elements each
// 4-bit unsigned: q ∈ [0,15]
// Dequant: d * scale * q - dmin * min
//
// NOTE: Include via quantize.cuh which defines warp reduce functions.

#pragma once

#ifndef QK_K
#define QK_K 256
#endif

// Helper: nearest_int (matches CPU rounding)
__device__ __forceinline__ int nearest_int_gpu(float v) {
    return (int)roundf(v);
}

// Helper: Compute scale and min for sub-block using make_qkx1_quants logic
// This exactly matches candle-core/src/quantized/utils.rs make_qkx1_quants
__device__ __forceinline__ void make_qkx1_quants_subblock(
    const float* __restrict__ x,
    int n,
    int nmax,
    int ntry,  // number of refinement iterations
    float* __restrict__ scale_out,
    float* __restrict__ min_out) {
    
    // Get min/max
    float min_val = x[0];
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        min_val = fminf(min_val, x[i]);
        max_val = fmaxf(max_val, x[i]);
    }
    
    // If min == max, all values are the same => nothing to do here
    if (max_val == min_val) {
        *scale_out = 0.0f;
        *min_out = 0.0f;
        return;
    }
    
    // Ensure min <= 0.0
    min_val = fminf(min_val, 0.0f);
    
    // Compute scale and inverse scale
    float iscale = (float)nmax / (max_val - min_val);
    float scale = 1.0f / iscale;
    
    // Temporary quantized values (stack allocated, n is small: 32)
    uint8_t l[32];  // n is always 32 for Q4K/Q5K
    for (int i = 0; i < n; i++) {
        l[i] = 0;
    }
    
    for (int iter = 0; iter < ntry; iter++) {
        float sumlx = 0.0f;
        int suml2 = 0;
        bool did_change = false;
        
        for (int i = 0; i < n; i++) {
            int li = nearest_int_gpu(iscale * (x[i] - min_val));
            li = max(0, min(nmax, li));
            uint8_t clamped_li = (uint8_t)li;
            if (clamped_li != l[i]) {
                l[i] = clamped_li;
                did_change = true;
            }
            sumlx += (x[i] - min_val) * (float)li;
            suml2 += li * li;
        }
        
        scale = sumlx / (float)suml2;
        
        // Compute new min: sum of (xi - scale * li) / n
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += x[i] - scale * (float)l[i];
        }
        min_val = sum / (float)n;
        if (min_val > 0.0f) {
            min_val = 0.0f;
        }
        iscale = 1.0f / scale;
        
        if (!did_change) {
            break;
        }
    }
    
    *scale_out = scale;
    *min_out = -min_val;  // Return positive min
}

// Helper: get_scale_min_k4 - extract scale and min from packed format
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// =============================================================================
// Q4_K QUANTIZATION - Single block, warp-cooperative
// =============================================================================

__device__ __forceinline__ void quantize_block_q4_K(
    const float* __restrict__ src,
    block_q4_K* __restrict__ dst) {
    
    const int lane = threadIdx.x % WARP_SIZE;
    
    // Step 1: Load all 256 values into shared memory
    __shared__ float shared_x[QK_K];
    __shared__ float shared_scales[8];  // sub-block scales
    __shared__ float shared_mins[8];    // sub-block mins
    __shared__ uint8_t shared_L[QK_K];  // quantized values [0-15]
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        shared_x[lane + i * 32] = src[lane + i * 32];
    }
    __syncwarp();
    
    // Step 2: Compute sub-block scales and mins (8 sub-blocks of 32 elements)
    // Each of first 8 threads handles one sub-block
    if (lane < 8) {
        const float* sub_x = shared_x + lane * 32;
        make_qkx1_quants_subblock(sub_x, 32, 15, 5, &shared_scales[lane], &shared_mins[lane]);
    }
    __syncwarp();
    
    // Step 3: Find max scale and max min across all sub-blocks
    float max_scale = 0.0f;
    float max_min = 0.0f;
    if (lane < 8) {
        max_scale = shared_scales[lane];
        max_min = shared_mins[lane];
    }
    // Warp reduce for max
    for (int offset = 4; offset > 0; offset >>= 1) {
        float other_scale = __shfl_xor_sync(0xffffffff, max_scale, offset, 32);
        float other_min = __shfl_xor_sync(0xffffffff, max_min, offset, 32);
        max_scale = fmaxf(max_scale, other_scale);
        max_min = fmaxf(max_min, other_min);
    }
    max_scale = __shfl_sync(0xffffffff, max_scale, 0, 32);
    max_min = __shfl_sync(0xffffffff, max_min, 0, 32);
    
    // Step 4: Compute super-block d and dmin
    float d_val = max_scale > 0.0f ? max_scale / 63.0f : 0.0f;
    float dmin_val = max_min > 0.0f ? max_min / 63.0f : 0.0f;
    
    float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
    float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;
    
    // Step 5: Quantize sub-block scales and mins to 6-bit, encode in packed format
    // Use single thread to avoid atomic issues
    __shared__ uint8_t encoded_scales[12];
    __shared__ uint8_t quantized_ls[8];
    __shared__ uint8_t quantized_lm[8];
    
    if (lane < 8) {
        int j = lane;
        quantized_ls[j] = (uint8_t)fminf(63.0f, roundf(inv_scale * shared_scales[j]));
        quantized_lm[j] = (uint8_t)fminf(63.0f, roundf(inv_min * shared_mins[j]));
    }
    __syncwarp();
    
    // Single thread encodes (matches GGML exactly)
    if (lane == 0) {
        for (int i = 0; i < 12; i++) {
            encoded_scales[i] = 0;
        }
        for (int j = 0; j < 8; j++) {
            uint8_t ls = quantized_ls[j];
            uint8_t lm = quantized_lm[j];
            if (j < 4) {
                encoded_scales[j] = ls;
                encoded_scales[j + 4] = lm;
            } else {
                encoded_scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                encoded_scales[j - 4] |= ((ls >> 4) << 6);
                encoded_scales[j] |= ((lm >> 4) << 6);
            }
        }
    }
    __syncwarp();
    
    // Store scales
    if (lane < 12) {
        dst->scales[lane] = encoded_scales[lane];
    }
    
    // Step 6: Quantize values using actual sub-block parameters
    // Decode the quantized scales/mins back to get exact match with dequant
    for (int i = lane; i < QK_K; i += 32) {
        int sub_block = i / 32;
        uint8_t sc, m;
        get_scale_min_k4(sub_block, encoded_scales, sc, m);
        
        float d = d_val * (float)sc;
        float dm = dmin_val * (float)m;
        
        uint8_t q;
        if (d != 0.0f) {
            int l = (int)roundf((shared_x[i] + dm) / d);
            q = (uint8_t)max(0, min(15, l));
        } else {
            q = 0;
        }
        shared_L[i] = q;
    }
    __syncwarp();
    
    // Step 7: Pack into qs[128]
    // For each 64-element group: qs[32] = low[i] | (high[i] << 4)
    // where low = first 32 elements, high = next 32 elements
    for (int half = 0; half < 4; half++) {  // 4 groups of 64 elements
        int base = half * 64;
        for (int l = lane; l < 32; l += 32) {
            uint8_t q_lo = shared_L[base + l];
            uint8_t q_hi = shared_L[base + l + 32];
            dst->qs[half * 32 + l] = q_lo | (q_hi << 4);
        }
    }
    
    // Step 9: Store d and dmin as half2 dm
    if (lane == 0) {
        dst->dm = make_half2(__float2half_rn(d_val), __float2half_rn(dmin_val));
    }
}

// =============================================================================
// MULTI-BLOCK QUANTIZATION
// =============================================================================

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q4_K(
    const float* __restrict__ src,
    block_q4_K* __restrict__ dst,
    int num_blocks) {
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    for (int blk = warp_id + blockIdx.x * warps_per_block; 
         blk < num_blocks; 
         blk += warps_per_block * gridDim.x) {
        
        quantize_block_q4_K(src + blk * QK_K, dst + blk);
    }
}
