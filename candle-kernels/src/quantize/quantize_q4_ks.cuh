// SPDX-License-Identifier: MIT
// Q4_KS Quantization: float -> 4-bit with attention-sink sub-block scaling
//
// Q4_KS format: [ half d | uint8_t sa | uint8_t sb | uint8_t qs[16] ] = 20 bytes
//   d   = amax_all / 7.0f  (coarse scale over all 32 elements)
//   sa  = round(amax_A / amax_all * 255), clamped to [1, 255]  (fine scale for elems 0-3)
//   sb  = round(amax_B / amax_all * 255), clamped to [1, 255]  (fine scale for elems 4-31)
//   actual_d_A = d * sa / 255.0  (actual scale for sub-block A)
//   actual_d_B = d * sb / 255.0  (actual scale for sub-block B)
//   q[k] = clamp(round(x[k] / actual_d), -7, 7) + 8  (biased unsigned nibble)
//   qs[k] = q[k] | (q[k+16] << 4)  for k in 0..16  (GGML nibble packing)
//
// NOTE: Include via quantize.cuh which defines warp reduce functions.

#pragma once

// =============================================================================
// SCALAR SINGLE-BLOCK QUANTIZATION (32 elements, 1 warp)
// =============================================================================

__device__ __forceinline__ void quantize_block_q4_ks(
    const float* __restrict__ src,
    block_q4_ks* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];
    const float axi = fabsf(xi);

    // Compute max absolute value for each sub-block using masked warp reduces
    float amax_a = quantize_warp_reduce_max(lane < 4 ? axi : 0.0f);
    float amax_b = quantize_warp_reduce_max(lane >= 4 ? axi : 0.0f);
    float amax = fmaxf(amax_a, amax_b);

    const float coarse_d = (amax != 0.0f) ? amax * (1.0f / 7.0f) : 0.0f;

    uint8_t sa, sb;
    if (amax == 0.0f) {
        sa = 255; sb = 255;
    } else {
        sa = (uint8_t)fmaxf(1.0f, fminf(255.0f, roundf(amax_a / amax * 255.0f)));
        sb = (uint8_t)fmaxf(1.0f, fminf(255.0f, roundf(amax_b / amax * 255.0f)));
    }

    const float actual_d = (lane < 4)
        ? (coarse_d * sa * (1.0f / 255.0f))
        : (coarse_d * sb * (1.0f / 255.0f));

    const int8_t q = (actual_d != 0.0f)
        ? (int8_t)fminf(7.0f, fmaxf(-7.0f, roundf(xi / actual_d)))
        : 0;
    const uint8_t quant = (uint8_t)((int)q + 8);  // biased [0, 15]

    // Pack nibbles: qs[k] = low_nibble(k) | (high_nibble(k+16) << 4)
    // All 32 lanes must participate in the shuffle to avoid divergence deadlock.
    // Lanes 16-31 discard the result.
    const uint8_t hi = (uint8_t)__shfl_sync(0xffffffff, (int)quant, lane + 16, 32);
    if (lane < 16) {
        dst->qs[lane] = quant | (hi << 4);
    }

    if (lane == 0) {
        dst->d  = __float2half_rn(coarse_d);
        dst->sa = sa;
        dst->sb = sb;
    }
}

// =============================================================================
// VECTORIZED SINGLE-BLOCK (float4 loads, 8 active lanes) — byte-identical to scalar.
// Sub-block A = elements 0-3 (lane 0); B = elements 4-31 (lanes 1-7). The GGML nibble pack
// `qs[k] = q[k] | q[k+16]<<4` pairs lane L (elems 4L..4L+3, all < 16) with lane L+4 (elems
// 4L+16..4L+19), so lane L<4 gathers L+4's four quants via one packed shuffle.
// =============================================================================
__device__ __forceinline__ void quantize_block_q4_ks_vec(
    const float* __restrict__ src,
    block_q4_ks* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    float local_max = 0.0f;
    if (lane < 8) {
        v = reinterpret_cast<const float4*>(src)[lane];
        local_max = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    }
    const float amax_a = quantize_warp_reduce_max((lane == 0) ? local_max : 0.0f);
    const float amax_b = quantize_warp_reduce_max((lane >= 1 && lane < 8) ? local_max : 0.0f);
    const float amax = fmaxf(amax_a, amax_b);
    const float coarse_d = (amax != 0.0f) ? amax * (1.0f / 7.0f) : 0.0f;
    uint8_t sa, sb;
    if (amax == 0.0f) {
        sa = 255; sb = 255;
    } else {
        sa = (uint8_t)fmaxf(1.0f, fminf(255.0f, roundf(amax_a / amax * 255.0f)));
        sb = (uint8_t)fmaxf(1.0f, fminf(255.0f, roundf(amax_b / amax * 255.0f)));
    }
    const float actual_d = (lane == 0) ? (coarse_d * sa * (1.0f / 255.0f)) : (coarse_d * sb * (1.0f / 255.0f));
    // Four biased nibbles for this lane's elements, packed into one int for the cross-lane gather.
    uint8_t b0 = 8, b1 = 8, b2 = 8, b3 = 8;
    if (lane < 8 && actual_d != 0.0f) {
        b0 = (uint8_t)((int)(int8_t)fminf(7.0f, fmaxf(-7.0f, roundf(v.x / actual_d))) + 8);
        b1 = (uint8_t)((int)(int8_t)fminf(7.0f, fmaxf(-7.0f, roundf(v.y / actual_d))) + 8);
        b2 = (uint8_t)((int)(int8_t)fminf(7.0f, fmaxf(-7.0f, roundf(v.z / actual_d))) + 8);
        b3 = (uint8_t)((int)(int8_t)fminf(7.0f, fmaxf(-7.0f, roundf(v.w / actual_d))) + 8);
    }
    const unsigned myq = (unsigned)b0 | ((unsigned)b1 << 8) | ((unsigned)b2 << 16) | ((unsigned)b3 << 24);
    // lane L<4 reads lane L+4's quants (the high nibbles for qs[4L..4L+3]). All lanes participate.
    const unsigned hiq = __shfl_sync(0xffffffff, myq, lane + 4, 32);
    if (lane < 4) {
        dst->qs[lane * 4 + 0] = (uint8_t)(b0 | (((hiq) & 0xff) << 4));
        dst->qs[lane * 4 + 1] = (uint8_t)(b1 | (((hiq >> 8) & 0xff) << 4));
        dst->qs[lane * 4 + 2] = (uint8_t)(b2 | (((hiq >> 16) & 0xff) << 4));
        dst->qs[lane * 4 + 3] = (uint8_t)(b3 | (((hiq >> 24) & 0xff) << 4));
    }
    if (lane == 0) {
        dst->d  = __float2half_rn(coarse_d);
        dst->sa = sa;
        dst->sb = sb;
    }
}

// =============================================================================
// MULTI-BLOCK QUANTIZATION (grid-stride, 1 warp per block)
// =============================================================================

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q4_ks(
    const float* __restrict__ src,
    block_q4_ks* __restrict__ dst,
    int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks;
         blk += warps_per_block * gridDim.x) {

        const float* block_src = src + blk * QK_Q4_KS;
        block_q4_ks*  block_dst = dst + blk;

        const float xi = block_src[lane];
        const float axi = fabsf(xi);

        float amax_a = quantize_warp_reduce_max(lane < 4 ? axi : 0.0f);
        float amax_b = quantize_warp_reduce_max(lane >= 4 ? axi : 0.0f);
        float amax = fmaxf(amax_a, amax_b);

        const float coarse_d = (amax != 0.0f) ? amax * (1.0f / 7.0f) : 0.0f;

        uint8_t sa, sb;
        if (amax == 0.0f) {
            sa = 255; sb = 255;
        } else {
            sa = (uint8_t)fmaxf(1.0f, fminf(255.0f, roundf(amax_a / amax * 255.0f)));
            sb = (uint8_t)fmaxf(1.0f, fminf(255.0f, roundf(amax_b / amax * 255.0f)));
        }

        const float actual_d = (lane < 4)
            ? (coarse_d * sa * (1.0f / 255.0f))
            : (coarse_d * sb * (1.0f / 255.0f));

        const int8_t q = (actual_d != 0.0f)
            ? (int8_t)fminf(7.0f, fmaxf(-7.0f, roundf(xi / actual_d)))
            : 0;
        const uint8_t quant = (uint8_t)((int)q + 8);

        // All 32 lanes must participate in the shuffle to avoid divergence deadlock.
        // Lanes 16-31 discard the result.
        const uint8_t hi = (uint8_t)__shfl_sync(0xffffffff, (int)quant, lane + 16, 32);
        if (lane < 16) {
            block_dst->qs[lane] = quant | (hi << 4);
        }

        if (lane == 0) {
            block_dst->d  = __float2half_rn(coarse_d);
            block_dst->sa = sa;
            block_dst->sb = sb;
        }
    }
}

// =============================================================================
// KERNEL ENTRY POINT
// =============================================================================

extern "C" __global__ void quantize_tensor_q4_ks(
    const float* __restrict__ src,
    block_q4_ks* __restrict__ dst,
    int num_blocks)
{
    quantize_blocks_q4_ks<1>(src, dst, num_blocks);
}
