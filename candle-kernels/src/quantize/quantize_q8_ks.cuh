// SPDX-License-Identifier: MIT
// Q8_KS Quantization: float -> 8-bit signed integer with attention-sink sub-block scaling
//
// Q8_KS format: [ half d | uint8_t sa | uint8_t sb | int8_t qs[32] ] = 36 bytes
//   d   = amax_all / 127.0f  (coarse scale over all 32 elements)
//   sa  = round(amax_A / amax_all * 255), clamped to [1, 255]  (fine scale for elems 0-3)
//   sb  = round(amax_B / amax_all * 255), clamped to [1, 255]  (fine scale for elems 4-31)
//   actual_d_A = d * sa / 255.0  (actual scale for sub-block A)
//   actual_d_B = d * sb / 255.0  (actual scale for sub-block B)
//   qs[k] = clamp(round(x[k] / actual_d), -127, 127)
//
// NOTE: Include via quantize.cuh which defines warp reduce functions.

#pragma once

// =============================================================================
// SCALAR SINGLE-BLOCK QUANTIZATION (32 elements, 1 warp)
// =============================================================================

__device__ __forceinline__ void quantize_block_q8_ks(
    const float* __restrict__ src,
    block_q8_ks* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];
    const float axi = fabsf(xi);

    float amax_a = quantize_warp_reduce_max(lane < 4 ? axi : 0.0f);
    float amax_b = quantize_warp_reduce_max(lane >= 4 ? axi : 0.0f);
    float amax = fmaxf(amax_a, amax_b);

    const float coarse_d = (amax != 0.0f) ? amax * (1.0f / 127.0f) : 0.0f;

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
        ? (int8_t)fminf(127.0f, fmaxf(-127.0f, roundf(xi / actual_d)))
        : 0;

    dst->qs[lane] = q;

    if (lane == 0) {
        dst->d  = __float2half_rn(coarse_d);
        dst->sa = sa;
        dst->sb = sb;
    }
}

// =============================================================================
// VECTORIZED SINGLE-BLOCK (float4 loads, 8 active lanes) — byte-identical to scalar.
// Sub-block A = elements 0-3 (lane 0's float4); B = elements 4-31 (lanes 1-7). amax is
// order-invariant, so the reduction restructure changes no bytes.
// =============================================================================
__device__ __forceinline__ void quantize_block_q8_ks_vec(
    const float* __restrict__ src,
    block_q8_ks* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    float4 v;
    float local_max = 0.0f;
    if (lane < 8) {
        v = reinterpret_cast<const float4*>(src)[lane];
        local_max = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    }
    const float amax_a = quantize_warp_reduce_max((lane == 0) ? local_max : 0.0f);
    const float amax_b = quantize_warp_reduce_max((lane >= 1 && lane < 8) ? local_max : 0.0f);
    const float amax = fmaxf(amax_a, amax_b);
    const float coarse_d = (amax != 0.0f) ? amax * (1.0f / 127.0f) : 0.0f;
    uint8_t sa, sb;
    if (amax == 0.0f) {
        sa = 255; sb = 255;
    } else {
        sa = (uint8_t)fmaxf(1.0f, fminf(255.0f, roundf(amax_a / amax * 255.0f)));
        sb = (uint8_t)fmaxf(1.0f, fminf(255.0f, roundf(amax_b / amax * 255.0f)));
    }
    if (lane < 8) {
        const float actual_d = (lane == 0) ? (coarse_d * sa * (1.0f / 255.0f)) : (coarse_d * sb * (1.0f / 255.0f));
        const int8_t q0 = (actual_d != 0.0f) ? (int8_t)fminf(127.0f, fmaxf(-127.0f, roundf(v.x / actual_d))) : 0;
        const int8_t q1 = (actual_d != 0.0f) ? (int8_t)fminf(127.0f, fmaxf(-127.0f, roundf(v.y / actual_d))) : 0;
        const int8_t q2 = (actual_d != 0.0f) ? (int8_t)fminf(127.0f, fmaxf(-127.0f, roundf(v.z / actual_d))) : 0;
        const int8_t q3 = (actual_d != 0.0f) ? (int8_t)fminf(127.0f, fmaxf(-127.0f, roundf(v.w / actual_d))) : 0;
        dst->qs[lane * 4 + 0] = q0;
        dst->qs[lane * 4 + 1] = q1;
        dst->qs[lane * 4 + 2] = q2;
        dst->qs[lane * 4 + 3] = q3;
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
__device__ __forceinline__ void quantize_blocks_q8_ks(
    const float* __restrict__ src,
    block_q8_ks* __restrict__ dst,
    int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks;
         blk += warps_per_block * gridDim.x) {

        const float* block_src = src + blk * QK_Q8_KS;
        block_q8_ks*  block_dst = dst + blk;

        const float xi = block_src[lane];
        const float axi = fabsf(xi);

        float amax_a = quantize_warp_reduce_max(lane < 4 ? axi : 0.0f);
        float amax_b = quantize_warp_reduce_max(lane >= 4 ? axi : 0.0f);
        float amax = fmaxf(amax_a, amax_b);

        const float coarse_d = (amax != 0.0f) ? amax * (1.0f / 127.0f) : 0.0f;

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
            ? (int8_t)fminf(127.0f, fmaxf(-127.0f, roundf(xi / actual_d)))
            : 0;

        block_dst->qs[lane] = q;

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

extern "C" __global__ void quantize_tensor_q8_ks(
    const float* __restrict__ src,
    block_q8_ks* __restrict__ dst,
    int num_blocks)
{
    quantize_blocks_q8_ks<1>(src, dst, num_blocks);
}
