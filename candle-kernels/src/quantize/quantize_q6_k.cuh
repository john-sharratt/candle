// SPDX-License-Identifier: MIT
// Q6_K Quantization: float -> 6-bit K-quant (256 elements per super-block)
//
// Q6_K format from GGML (follows candle-core/src/quantized/k_quants.rs):
//   - ql[128] (uint8_t): lower 4 bits of quants, packed specially
//   - qh[64] (uint8_t): upper 2 bits of quants (4 per byte)
//   - scales[16] (int8_t): 8-bit signed scales for 16 sub-blocks
//   - d (half): super-block scale
//
// Structure: 256 elements in 16 sub-blocks of 16 elements each
// 6-bit symmetric: q ∈ [0,63] representing values [-32, 31]
// Dequant: d * scale * (q - 32)
//
// Packing (per 128-element half-block, j = 0 or 128):
//   ql[l] = (q[j+l] & 0xF) | ((q[j+l+64] & 0xF) << 4)
//   ql[l+32] = (q[j+l+32] & 0xF) | ((q[j+l+96] & 0xF) << 4)
//   qh[l] = (q[j+l] >> 4) | ((q[j+l+32] >> 4) << 2) | ((q[j+l+64] >> 4) << 4) | ((q[j+l+96] >> 4) << 6)
//
// NOTE: Include via quantize.cuh which defines warp reduce functions.

#pragma once

#ifndef QK_K
#define QK_K 256
#endif

// Helper: Full make_qx_quants algorithm with RMSE optimization (matches CPU with rmse_type=1)
// For Q6K: nmax=32 (values -32 to 31, stored as 0-63)
// Returns scale and fills ls with quantized values (stored as l + nmax, so [0, 63])
__device__ __forceinline__ float make_qx_quants_full(
    const float* __restrict__ x,
    int n,
    int nmax,
    int8_t* __restrict__ ls) {
    
    // Find max value by absolute value
    float max_val = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            max_val = x[i];
        }
    }
    
    if (amax == 0.0f) {
        // All zero
        for (int i = 0; i < n; i++) {
            ls[i] = nmax;  // Zero value stored as nmax
        }
        return 0.0f;
    }
    
    float iscale = -(float)nmax / max_val;
    
    // Using rmse_type=1: weight by x^2
    float sumlx = 0.0f;
    float suml2 = 0.0f;
    
    for (int i = 0; i < n; i++) {
        int l = (int)roundf(iscale * x[i]);
        l = max(-nmax, min(nmax - 1, l));
        ls[i] = (int8_t)(l + nmax);  // Store as [0, 2*nmax-1]
        float w = x[i] * x[i];  // weight_type = 1
        sumlx += w * x[i] * (float)l;
        suml2 += w * (float)l * (float)l;
    }
    
    float scale = (suml2 > 0.0f) ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;
    
    // Phase 1: 3 iterations of scale refinement
    for (int itry = 0; itry < 3; itry++) {
        iscale = (scale != 0.0f) ? 1.0f / scale : 0.0f;
        float slx = 0.0f;
        float sl2 = 0.0f;
        bool changed = false;
        
        for (int i = 0; i < n; i++) {
            int l = (int)roundf(iscale * x[i]);
            l = max(-nmax, min(nmax - 1, l));
            if (l + nmax != (int)ls[i]) {
                changed = true;
            }
            float w = x[i] * x[i];
            slx += w * x[i] * (float)l;
            sl2 += w * (float)l * (float)l;
        }
        
        if (!changed || sl2 == 0.0f || slx * slx <= best * sl2) {
            break;
        }
        
        // Update ls with new quantized values
        for (int i = 0; i < n; i++) {
            int l = (int)roundf(iscale * x[i]);
            ls[i] = (int8_t)(nmax + max(-nmax, min(nmax - 1, l)));
        }
        sumlx = slx;
        suml2 = sl2;
        scale = sumlx / suml2;
        best = scale * sumlx;
    }
    
    // Phase 2: 5 iterations of individual element optimization
    for (int itry = 0; itry < 5; itry++) {
        int n_changed = 0;
        
        for (int i = 0; i < n; i++) {
            float w = x[i] * x[i];
            int l = (int)ls[i] - nmax;
            float slx = sumlx - w * x[i] * (float)l;
            
            if (slx > 0.0f) {
                float sl2 = suml2 - w * (float)l * (float)l;
                int new_l = (int)roundf(x[i] * sl2 / slx);
                new_l = max(-nmax, min(nmax - 1, new_l));
                
                if (new_l != l) {
                    slx += w * x[i] * (float)new_l;
                    sl2 += w * (float)new_l * (float)new_l;
                    
                    if (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                        ls[i] = (int8_t)(nmax + new_l);
                        sumlx = slx;
                        suml2 = sl2;
                        scale = sumlx / suml2;
                        best = scale * sumlx;
                        n_changed++;
                    }
                }
            }
        }
        
        if (n_changed == 0) {
            break;
        }
    }
    
    return scale;
}

// =============================================================================
// Q6_K QUANTIZATION - Single block, warp-cooperative
// =============================================================================

__device__ __forceinline__ void quantize_block_q6_K(
    const float* __restrict__ src,
    block_q6_K* __restrict__ dst) {
    
    const int lane = threadIdx.x % WARP_SIZE;
    
    // Step 1: Load all 256 values into shared memory
    __shared__ float shared_x[QK_K];
    __shared__ float shared_scales[16];  // sub-block scales
    __shared__ int8_t shared_L[QK_K];    // quantized values [0-63]
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        shared_x[lane + i * 32] = src[lane + i * 32];
    }
    __syncwarp();
    
    // Step 2: Compute sub-block scales using full make_qx_quants algorithm
    // This also fills shared_L with quantized values for this sub-block
    if (lane < 16) {
        const float* sub_x = shared_x + lane * 16;
        int8_t sub_L[16];  // temporary for this sub-block
        
        shared_scales[lane] = make_qx_quants_full(sub_x, 16, 32, sub_L);
        
        // Copy quantized values to shared memory
        for (int i = 0; i < 16; i++) {
            shared_L[lane * 16 + i] = sub_L[i];
        }
    }
    __syncwarp();
    
    // Step 3: Find max scale by absolute value
    float max_scale = 0.0f;
    float max_abs_scale = 0.0f;
    if (lane < 16) {
        float s = shared_scales[lane];
        if (fabsf(s) > max_abs_scale) {
            max_abs_scale = fabsf(s);
            max_scale = s;
        }
    }
    // Warp reduce for max abs
    for (int offset = 8; offset > 0; offset >>= 1) {
        float other_scale = __shfl_xor_sync(0xffffffff, max_scale, offset, 32);
        float other_abs = __shfl_xor_sync(0xffffffff, max_abs_scale, offset, 32);
        if (other_abs > max_abs_scale) {
            max_abs_scale = other_abs;
            max_scale = other_scale;
        }
    }
    max_scale = __shfl_sync(0xffffffff, max_scale, 0, 32);
    
    // Step 4: Compute super-block d and quantized scales
    float iscale = (max_scale != 0.0f) ? -128.0f / max_scale : 0.0f;
    float d_val = (iscale != 0.0f) ? 1.0f / iscale : 0.0f;
    
    __shared__ int8_t quantized_scales[16];
    
    if (lane < 16) {
        int s = (int)roundf(iscale * shared_scales[lane]);
        s = max(-127, min(127, s));
        quantized_scales[lane] = (int8_t)s;
        dst->scales[lane] = (int8_t)s;
    }
    __syncwarp();
    
    // Step 5: Re-quantize values using actual quantized scales (to match dequant)
    for (int i = lane; i < QK_K; i += 32) {
        int sub_block = i / 16;
        int8_t sc = quantized_scales[sub_block];
        float d = d_val * (float)sc;
        
        int8_t q;
        if (d != 0.0f) {
            int l = (int)roundf(shared_x[i] / d);
            l = max(-32, min(31, l)) + 32;  // Map to [0, 63]
            q = (int8_t)l;
        } else {
            q = 32;  // 0 value
        }
        shared_L[i] = q;
    }
    __syncwarp();
    
    // Step 6: Pack into ql[128] and qh[64]
    // Process in two 128-element half-blocks (n = 0, 128)
    
    // Clear output arrays
    for (int i = lane; i < 128; i += 32) {
        dst->ql[i] = 0;
    }
    for (int i = lane; i < 64; i += 32) {
        dst->qh[i] = 0;
    }
    __syncwarp();
    
    for (int n = 0; n < 2; n++) {  // n=0: j=0, n=1: j=128
        int j = n * 128;
        int ql_base = n * 64;
        int qh_base = n * 32;
        
        for (int l = lane; l < 32; l += 32) {
            int8_t q1 = shared_L[j + l];
            int8_t q2 = shared_L[j + l + 32];
            int8_t q3 = shared_L[j + l + 64];
            int8_t q4 = shared_L[j + l + 96];
            
            // Pack low 4 bits into ql
            dst->ql[ql_base + l] = ((q1 & 0xF) | ((q3 & 0xF) << 4));
            dst->ql[ql_base + l + 32] = ((q2 & 0xF) | ((q4 & 0xF) << 4));
            
            // Pack high 2 bits into qh
            dst->qh[qh_base + l] = ((q1 >> 4) & 3) | 
                                   (((q2 >> 4) & 3) << 2) |
                                   (((q3 >> 4) & 3) << 4) |
                                   (((q4 >> 4) & 3) << 6);
        }
    }
    
    // Step 7: Store super-block scale
    if (lane == 0) {
        dst->d = __float2half_rn(d_val);
    }
}

// =============================================================================
// MULTI-BLOCK QUANTIZATION
// =============================================================================

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q6_K(
    const float* __restrict__ src,
    block_q6_K* __restrict__ dst,
    int num_blocks) {
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    for (int blk = warp_id + blockIdx.x * warps_per_block; 
         blk < num_blocks; 
         blk += warps_per_block * gridDim.x) {
        
        quantize_block_q6_K(src + blk * QK_K, dst + blk);
    }
}
