// SPDX-License-Identifier: MIT
// Q3_0 Quantization: float -> 3-bit symmetric with per-block scale
//
// Q3_0 format stores:
//   - d (half): scale factor, d = amax / 3.5
//   - qh[4] (uint8_t): high (3rd) bit of each of the 32 quants (8 per byte)
//   - qs[8] (uint8_t): low 2 bits of each quant, 4 per byte
//
// Block size: 32 elements → 14 bytes total (2 + 4 + 8)
// Compression: 3.5 bits per element (vs 32 bits for float)
//
// Quantization levels: 0..7, centered at 3.5
//   decode: x = d * (q - 3.5)
//   encode: q = clamp(round(x * (3.5/amax) + 3.5), 0, 7)
//
// NOTE: Include via quantize.cuh which defines warp reduce functions.

#pragma once

// =============================================================================
// OPTIMIZED SINGLE-BLOCK QUANTIZATION (32 elements, vectorized)
// =============================================================================
// 8 threads load float4 each (8 x 4 = 32 floats).
// Each thread encodes 4 quants: low 2 bits into 1 qs byte, high bits into qh.

__device__ __forceinline__ void quantize_block_q3_0_vec(
    const float* __restrict__ src,
    block_q3_0* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;

    float4 v;
    float local_max = 0.0f;

    if (lane < 8) {
        v = reinterpret_cast<const float4*>(src)[lane];
        local_max = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                         fmaxf(fabsf(v.z), fabsf(v.w)));
    }

    // Reduce max across first 8 lanes, broadcast to all 32
    float amax = local_max;
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xff, amax, offset, 8));
    amax = __shfl_sync(0xffffffff, amax, 0, 32);

    if (lane < 8) {
        const float id = (amax != 0.0f) ? 3.5f / amax : 0.0f;

        auto encode = [&](float x) -> uint8_t {
            return (uint8_t)fminf(7.0f, fmaxf(0.0f, roundf(x * id + 3.5f)));
        };

        uint8_t q0 = encode(v.x);
        uint8_t q1 = encode(v.y);
        uint8_t q2 = encode(v.z);
        uint8_t q3 = encode(v.w);

        // Pack low 2 bits into qs[lane]
        dst->qs[lane] = (q0 & 3) | ((q1 & 3) << 2) | ((q2 & 3) << 4) | ((q3 & 3) << 6);

        // Pack high bits:
        // Threads 0,1 → qh[0]; threads 2,3 → qh[1]; 4,5 → qh[2]; 6,7 → qh[3]
        // Even thread: bits 0..3 of its qh byte; odd thread: bits 4..7
        uint8_t my_qh = ((q0 >> 2) & 1) | (((q1 >> 2) & 1) << 1)
                      | (((q2 >> 2) & 1) << 2) | (((q3 >> 2) & 1) << 3);
        if (lane & 1) my_qh <<= 4;  // odd lanes contribute upper nibble

        // Combine with partner lane via XOR-shuffle within pairs
        uint8_t partner_qh = (uint8_t)__shfl_xor_sync(0xff, (int)my_qh, 1, 8);
        uint8_t qh_byte = my_qh | partner_qh;

        if ((lane & 1) == 0) {
            dst->qh[lane >> 1] = qh_byte;
        }
    }

    if (lane == 0)
        dst->d = __float2half_rn(amax / 3.5f);
}

// =============================================================================
// SCALAR FALLBACK (one element per thread, full-warp reduction)
// Used by transpose_batch.cuh for fused transpose+quantize.
// =============================================================================

__device__ __forceinline__ void quantize_block_q3_0(
    const float* __restrict__ src,
    block_q3_0* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];

    float amax = fabsf(xi);
    amax = quantize_warp_reduce_max(amax);

    const float d = amax / 3.5f;
    const float id = (amax != 0.0f) ? 3.5f / amax : 0.0f;

    // q = clamp(round(xi * id + 3.5), 0, 7)
    const uint8_t q3 = (uint8_t)fminf(7.0f, fmaxf(0.0f, roundf(xi * id + 3.5f)));

    // Pack low 2 bits into qs[lane/4] at bit position (lane%4)*2.
    // Each group of 4 lanes (lane & ~3) writes one qs byte.
    uint8_t packed_qs = (q3 & 3) << (2 * (lane & 3));
    packed_qs |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)packed_qs, 1, 32);
    packed_qs |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)packed_qs, 2, 32);
    if ((lane & 3) == 0) {
        dst->qs[lane >> 2] = packed_qs;
    }

    // Pack high bit into qh[lane/8] at bit position lane%8.
    // Each group of 8 lanes writes one qh byte.
    uint8_t packed_qh = ((q3 >> 2) & 1) << (lane & 7);
    packed_qh |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)packed_qh, 1, 32);
    packed_qh |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)packed_qh, 2, 32);
    packed_qh |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)packed_qh, 4, 32);
    if ((lane & 7) == 0) {
        dst->qh[lane >> 3] = packed_qh;
    }

    if (lane == 0) {
        dst->d = __float2half_rn(d);
    }
}

// =============================================================================
// MULTI-BLOCK QUANTIZATION (grid-stride, vectorized)
// =============================================================================

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q3_0(
    const float* __restrict__ src,
    block_q3_0* __restrict__ dst,
    int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks;
         blk += warps_per_block * gridDim.x) {

        const float* block_src = src + blk * QK3_0;
        block_q3_0*  block_dst = dst + blk;

        float4 v;
        float local_max = 0.0f;

        if (lane < 8) {
            v = reinterpret_cast<const float4*>(block_src)[lane];
            local_max = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                             fmaxf(fabsf(v.z), fabsf(v.w)));
        }

        float amax = local_max;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xff, amax, offset, 8));
        amax = __shfl_sync(0xffffffff, amax, 0, 32);

        if (lane < 8) {
            const float id = (amax != 0.0f) ? 3.5f / amax : 0.0f;

            auto encode = [&](float x) -> uint8_t {
                return (uint8_t)fminf(7.0f, fmaxf(0.0f, roundf(x * id + 3.5f)));
            };

            uint8_t q0 = encode(v.x);
            uint8_t q1 = encode(v.y);
            uint8_t q2 = encode(v.z);
            uint8_t q3_val = encode(v.w);

            block_dst->qs[lane] = (q0 & 3) | ((q1 & 3) << 2)
                                            | ((q2 & 3) << 4)
                                            | ((q3_val & 3) << 6);

            uint8_t my_qh = ((q0 >> 2) & 1) | (((q1 >> 2) & 1) << 1)
                          | (((q2 >> 2) & 1) << 2) | (((q3_val >> 2) & 1) << 3);
            if (lane & 1) my_qh <<= 4;

            uint8_t partner_qh = (uint8_t)__shfl_xor_sync(0xff, (int)my_qh, 1, 8);
            uint8_t qh_byte = my_qh | partner_qh;

            if ((lane & 1) == 0) {
                block_dst->qh[lane >> 1] = qh_byte;
            }
        }

        if (lane == 0)
            block_dst->d = __float2half_rn(amax / 3.5f);
    }
}
