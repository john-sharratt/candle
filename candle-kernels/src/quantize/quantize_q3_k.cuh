// SPDX-License-Identifier: MIT
// Q3_K Quantization: float -> 3-bit K-quant (256 elements per super-block)
//
// Q3_K format from GGML (follows candle-core/src/quantized/k_quants.rs):
//   - hmask[32] (uint8_t): high bits of quants (bit indicates q >= 4)
//   - qs[64] (uint8_t): low 2 bits of quants, packed 4 per byte
//   - scales[12] (uint8_t): 6-bit scales for 16 sub-blocks, encoded specially
//   - d (half): super-block scale
//
// Structure: 256 elements in 16 sub-blocks of 16 elements each
// 3-bit symmetric: q ∈ [0,7] representing values in [-4, 3]
// Dequant: dl * (q_2bit - (hmask ? 0 : 4)) where dl = d * (scale - 32)
//
// CPU algorithm (k_quants.rs):
//   1. For each 16-elem sub-block: scale = make_q3_quants(x, 4, true)
//      - make_q3_quants uses RMSE optimization with 5 iterations
//   2. Find max scale by absolute value
//   3. iscale = -32 / max_scale, d = 1 / iscale
//   4. Quantize scales: l = round(iscale * scale), clamp to [-32,31], add 32
//   5. Re-quantize values using reconstructed dl = d * (sc - 32)
//   6. Encode hmask (high bit) and qs (low 2 bits)
//
// NOTE: Include via quantize.cuh which defines warp reduce functions.

#pragma once

#ifndef QK_K
#define QK_K 256
#endif

#ifndef K_SCALE_SIZE
#define K_SCALE_SIZE 12
#endif

// =============================================================================
// make_q3_quants: RMSE-optimized quantization for 3-bit values
// =============================================================================
// Returns the scale and fills L with quantized values in [0, 2*nmax-1]
// Matches CPU algorithm in utils.rs with do_rmse=true

__device__ __forceinline__ float make_q3_quants_serial(
    const float* __restrict__ x,
    int n,
    int nmax,
    int8_t* __restrict__ L) {
    
    // Find max by absolute value
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
        for (int i = 0; i < n; i++) L[i] = (int8_t)nmax;
        return 0.0f;
    }
    
    // Initial quantization
    float iscale = -(float)nmax / max_val;
    float sumlx = 0.0f;
    float suml2 = 0.0f;
    
    for (int i = 0; i < n; i++) {
        int li = (int)roundf(iscale * x[i]);
        li = max(-nmax, min(nmax - 1, li));
        L[i] = (int8_t)li;
        float w = x[i] * x[i];
        sumlx += w * x[i] * (float)li;
        suml2 += w * (float)(li * li);
    }
    
    // RMSE optimization: 5 iterations matching CPU
    for (int iter = 0; iter < 5; iter++) {
        int n_changed = 0;
        for (int i = 0; i < n; i++) {
            float w = x[i] * x[i];
            float slx = sumlx - w * x[i] * (float)L[i];
            if (slx > 0.0f) {
                float sl2 = suml2 - w * (float)(L[i] * L[i]);
                int new_l = (int)roundf(x[i] * sl2 / slx);
                new_l = max(-nmax, min(nmax - 1, new_l));
                if (new_l != (int)L[i]) {
                    float new_slx = slx + w * x[i] * (float)new_l;
                    float new_sl2 = sl2 + w * (float)(new_l * new_l);
                    if (new_sl2 > 0.0f && new_slx * new_slx * suml2 > sumlx * sumlx * new_sl2) {
                        L[i] = (int8_t)new_l;
                        sumlx = new_slx;
                        suml2 = new_sl2;
                        n_changed++;
                    }
                }
            }
        }
        if (n_changed == 0) break;
    }
    
    // Shift L from [-nmax, nmax-1] to [0, 2*nmax-1]
    for (int i = 0; i < n; i++) {
        L[i] += (int8_t)nmax;
    }
    
    return (suml2 > 0.0f) ? (sumlx / suml2) : (1.0f / iscale);
}

// =============================================================================
// Q3_K QUANTIZATION - Single block, serial per-thread processing
// =============================================================================
// Uses shared memory for block-wide coordination, serial loops match CPU

__device__ __forceinline__ void quantize_block_q3_K(
    const float* __restrict__ src,
    block_q3_K* __restrict__ dst) {
    
    const int lane = threadIdx.x % WARP_SIZE;
    
    // Shared memory for block data
    __shared__ float shared_x[QK_K];
    __shared__ int8_t shared_L[QK_K];
    __shared__ float shared_scales[16];
    
    // Step 1: Load all 256 values into shared memory
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        shared_x[lane + i * 32] = src[lane + i * 32];
    }
    __syncwarp();
    
    // Step 2: Each of the first 16 threads processes one 16-element sub-block
    if (lane < 16) {
        int8_t sub_L[16];
        shared_scales[lane] = make_q3_quants_serial(
            shared_x + lane * 16, 16, 4, sub_L);
        
        // Store to shared memory
        for (int i = 0; i < 16; i++) {
            shared_L[lane * 16 + i] = sub_L[i];
        }
    }
    __syncwarp();
    
    // Step 3: Find max scale by absolute value (serial, matching CPU)
    float max_scale = 0.0f;
    if (lane == 0) {
        for (int j = 0; j < 16; j++) {
            float s = shared_scales[j];
            if (fabsf(s) > fabsf(max_scale)) {
                max_scale = s;
            }
        }
    }
    max_scale = __shfl_sync(0xffffffff, max_scale, 0, 32);
    
    // Step 4: Compute super-block scale and quantize sub-block scales
    __shared__ int8_t quantized_scales[16];
    
    if (lane == 0) {
        if (max_scale != 0.0f) {
            float iscale = -32.0f / max_scale;
            for (int j = 0; j < 16; j++) {
                int l_val = (int)roundf(iscale * shared_scales[j]);
                l_val = max(-32, min(31, l_val)) + 32;
                quantized_scales[j] = (int8_t)l_val;
            }
            dst->d = __float2half_rn(1.0f / iscale);
        } else {
            for (int j = 0; j < 16; j++) {
                quantized_scales[j] = 32;
            }
            dst->d = __float2half_rn(0.0f);
        }
    }
    __syncwarp();
    
    // Step 5: Encode scales into 12-byte packed format
    if (lane == 0) {
        // Clear scales array
        for (int i = 0; i < 12; i++) {
            dst->scales[i] = 0;
        }
        
        for (int j = 0; j < 16; j++) {
            int l = quantized_scales[j];
            if (j < 8) {
                dst->scales[j] = (l & 0xF);
            } else {
                dst->scales[j - 8] |= ((l & 0xF) << 4);
            }
            int high_bits = l >> 4;
            dst->scales[8 + (j % 4)] |= (high_bits << (2 * (j / 4)));
        }
    }
    __syncwarp();
    
    // Step 6: Re-quantize using reconstructed dl = d * (sc - 32)
    float d_val = __half2float(dst->d);
    if (lane < 16) {
        // Reconstruct scale the same way CPU does during re-quantization
        int sc;
        if (lane < 8) {
            sc = dst->scales[lane] & 0xF;
        } else {
            sc = dst->scales[lane - 8] >> 4;
        }
        sc |= (((dst->scales[8 + lane % 4] >> (2 * (lane / 4))) & 3) << 4);
        sc = sc - 32;
        
        float dl = d_val * (float)sc;
        
        if (dl != 0.0f) {
            for (int i = 0; i < 16; i++) {
                int l_val = (int)roundf(shared_x[lane * 16 + i] / dl);
                l_val = max(-4, min(3, l_val)) + 4;
                shared_L[lane * 16 + i] = (int8_t)l_val;
            }
        } else {
            for (int i = 0; i < 16; i++) {
                shared_L[lane * 16 + i] = 4;  // Zero maps to 4
            }
        }
    }
    __syncwarp();
    
    // Step 7: Encode hmask and qs
    // Clear output arrays
    if (lane < 32) {
        dst->hmask[lane] = 0;
        dst->qs[lane] = 0;
        dst->qs[lane + 32] = 0;
    }
    __syncwarp();
    
    // Encode hmask: CPU iterates through L sequentially, setting bits
    // hmask[m] |= hm where m cycles 0..31 and hm cycles through bits
    if (lane == 0) {
        int m = 0;
        uint8_t hm = 1;
        for (int i = 0; i < QK_K; i++) {
            if (shared_L[i] > 3) {
                dst->hmask[m] |= hm;
                shared_L[i] -= 4;  // Remove high bit from L
            }
            m++;
            if (m == 32) {
                m = 0;
                hm <<= 1;
            }
        }
    }
    __syncwarp();
    
    // Encode qs: pack low 2 bits, 4 values per byte
    // CPU: qs[j/4 + l] = L[j+l] | (L[j+l+32]<<2) | (L[j+l+64]<<4) | (L[j+l+96]<<6)
    for (int half = 0; half < 2; half++) {
        int base = half * 128;
        for (int l = lane; l < 32; l += 32) {
            uint8_t packed = ((uint8_t)shared_L[base + l] & 3)
                           | (((uint8_t)shared_L[base + l + 32] & 3) << 2)
                           | (((uint8_t)shared_L[base + l + 64] & 3) << 4)
                           | (((uint8_t)shared_L[base + l + 96] & 3) << 6);
            dst->qs[half * 32 + l] = packed;
        }
    }
}

// =============================================================================
// MULTI-BLOCK QUANTIZATION
// =============================================================================

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q3_K(
    const float* __restrict__ src,
    block_q3_K* __restrict__ dst,
    int num_blocks) {
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    for (int blk = warp_id + blockIdx.x * warps_per_block; 
         blk < num_blocks; 
         blk += warps_per_block * gridDim.x) {
        
        quantize_block_q3_K(src + blk * QK_K, dst + blk);
    }
}
