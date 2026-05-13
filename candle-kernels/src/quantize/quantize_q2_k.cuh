// SPDX-License-Identifier: MIT
// Q2_K Quantization: float -> 2-bit K-quant (256 elements per super-block)
//
// Q2_K format:
//   - scales[16] (uint8_t): packed 4-bit scales (low nibble) and mins (high nibble)
//   - qs[64] (uint8_t): 2-bit quants (4 per byte)
//   - dm (half2): x=d (super-block scale), y=dmin (super-block min scale)
//
// Memory layout:
//   - 16 sub-blocks of 16 elements each
//   - Each sub-block has scale[i]&0xF and min scale[i]>>4
//   - Dequant: y = d * (scale&0xF) * q - dmin * (scale>>4)

#pragma once

#ifndef QK_K
#define QK_K 256
#endif

// Helper: Serial make_qkx1_quants for sub-block (exact CPU match)
// n must be 16 for Q2K sub-blocks
__device__ __forceinline__ void make_qkx1_quants_subblock_q2k(
    const float* __restrict__ x,
    int n,
    int nmax,   // 3 for Q2K (2-bit values: 0-3)
    int ntry,   // 5 iterations
    float* __restrict__ scale_out,
    float* __restrict__ min_out) {
    
    // Get min/max
    float min_val = x[0];
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        min_val = fminf(min_val, x[i]);
        max_val = fmaxf(max_val, x[i]);
    }
    
    // If min == max, all values are the same
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
    
    // Temporary quantized values (stack allocated, n=16 for Q2K)
    uint8_t l[16];
    for (int i = 0; i < n; i++) {
        l[i] = 0;
    }
    
    for (int iter = 0; iter < ntry; iter++) {
        float sumlx = 0.0f;
        int suml2 = 0;
        bool did_change = false;
        
        for (int i = 0; i < n; i++) {
            int li = (int)roundf(iscale * (x[i] - min_val));
            li = max(0, min(nmax, li));
            uint8_t clamped_li = (uint8_t)li;
            if (clamped_li != l[i]) {
                l[i] = clamped_li;
                did_change = true;
            }
            sumlx += (x[i] - min_val) * (float)li;
            suml2 += li * li;
        }
        
        if (suml2 > 0) {
            scale = sumlx / (float)suml2;
        }
        
        // Compute new min: sum of (xi - scale * li) / n
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += x[i] - scale * (float)l[i];
        }
        min_val = sum / (float)n;
        if (min_val > 0.0f) {
            min_val = 0.0f;
        }
        iscale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        
        if (!did_change) {
            break;
        }
    }
    
    *scale_out = scale;
    *min_out = -min_val;  // Return positive min
}

// =============================================================================
// Q2_K QUANTIZATION - CPU-matching algorithm with serial iterative refinement
// Uses shared memory for the full 256-element block
// =============================================================================

__device__ __forceinline__ void quantize_block_q2_K(
    const float* __restrict__ src,
    block_q2_K* __restrict__ dst) {
    
    const int lane = threadIdx.x % WARP_SIZE;
    
    // Shared memory - declared inside function for proper scoping
    __shared__ float shared_x[QK_K];
    __shared__ float shared_scales[16];
    __shared__ float shared_mins[16];
    __shared__ uint8_t shared_L[QK_K];
    
    // Step 1: Load all 256 values into shared memory
    #pragma unroll
    for (int i = lane; i < QK_K; i += 32) {
        shared_x[i] = src[i];
    }
    __syncwarp();
    
    // Step 2: Process 16 sub-blocks (16 elements each)
    // Each of first 16 threads handles one sub-block using serial loop (matches CPU exactly)
    if (lane < 16) {
        const float* sub_x = shared_x + lane * 16;
        make_qkx1_quants_subblock_q2k(sub_x, 16, 3, 5, &shared_scales[lane], &shared_mins[lane]);
    }
    __syncwarp();
    
    // Step 3: Find max scale and max min across all 16 sub-blocks
    float max_scale = 0.0f, max_min = 0.0f;
    if (lane < 16) {
        max_scale = shared_scales[lane];
        max_min = shared_mins[lane];
    }
    
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        max_scale = fmaxf(max_scale, __shfl_xor_sync(0xffffffff, max_scale, offset, 32));
        max_min = fmaxf(max_min, __shfl_xor_sync(0xffffffff, max_min, offset, 32));
    }
    
    // Broadcast to all lanes
    max_scale = __shfl_sync(0xffffffff, max_scale, 0, 32);
    max_min = __shfl_sync(0xffffffff, max_min, 0, 32);
    
    // Step 4: Compute super-block d and dmin
    const float Q4SCALE = 15.0f;
    float d_val = max_scale / Q4SCALE;
    float dmin_val = max_min / Q4SCALE;
    float id = (max_scale > 0.0f) ? Q4SCALE / max_scale : 0.0f;
    float im = (max_min > 0.0f) ? Q4SCALE / max_min : 0.0f;
    
    // Step 5: Quantize sub-block scales and mins to 4-bit and store
    if (lane < 16) {
        int scale_q = (int)roundf(shared_scales[lane] * id);
        int min_q = (int)roundf(shared_mins[lane] * im);
        scale_q = max(0, min(15, scale_q));
        min_q = max(0, min(15, min_q));
        dst->scales[lane] = (uint8_t)(scale_q | (min_q << 4));
    }
    __syncwarp();
    
    // Step 6: Re-quantize values using the reconstructed parameters (exact match with dequant)
    for (int sb = 0; sb < 16; sb++) {
        uint8_t sc_packed = dst->scales[sb];
        float d_sub = d_val * (float)(sc_packed & 0xF);
        float dm_sub = dmin_val * (float)(sc_packed >> 4);
        
        if (lane < 16) {
            int elem_idx = sb * 16 + lane;
            float val = shared_x[elem_idx];
            
            uint8_t q;
            if (d_sub > 0.0f) {
                int li = (int)roundf((val + dm_sub) / d_sub);
                q = (uint8_t)max(0, min(3, li));
            } else {
                q = 0;
            }
            shared_L[elem_idx] = q;
        }
        __syncwarp();
    }
    
    // Step 7: Pack into qs[64]
    // qs[ll] = q[ll] | (q[ll+32]<<2) | (q[ll+64]<<4) | (q[ll+96]<<6) for ll in [0,32)
    // qs[32+ll] = q[128+ll] | (q[160+ll]<<2) | (q[192+ll]<<4) | (q[224+ll]<<6)
    if (lane < 32) {
        uint8_t q0 = shared_L[lane];
        uint8_t q1 = shared_L[lane + 32];
        uint8_t q2 = shared_L[lane + 64];
        uint8_t q3 = shared_L[lane + 96];
        dst->qs[lane] = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6);
    }
    __syncwarp();
    
    if (lane < 32) {
        uint8_t q0 = shared_L[128 + lane];
        uint8_t q1 = shared_L[160 + lane];
        uint8_t q2 = shared_L[192 + lane];
        uint8_t q3 = shared_L[224 + lane];
        dst->qs[32 + lane] = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6);
    }
    
    // Step 8: Store d and dmin as half2 dm
    if (lane == 0) {
        dst->dm = make_half2(__float2half_rn(d_val), __float2half_rn(dmin_val));
    }
}

// =============================================================================
// MULTI-BLOCK QUANTIZATION
// =============================================================================

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q2_K(
    const float* __restrict__ src,
    block_q2_K* __restrict__ dst,
    int num_blocks) {
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    for (int blk = warp_id + blockIdx.x * warps_per_block; 
         blk < num_blocks; 
         blk += warps_per_block * gridDim.x) {
        
        quantize_block_q2_K(src + blk * QK_K, dst + blk);
    }
}
