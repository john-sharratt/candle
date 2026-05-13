// SPDX-License-Identifier: MIT
// Q3_1 Quantization: float -> 3-bit asymmetric with F16 scale + F16 min
// Block: { half2 dm; uint8_t qh[4]; uint8_t qs[8]; } = 16 bytes = 4.00 BPE
// 3-bit unsigned [0..7], low 2 bits in qs, high bit in qh
// decode: q * d + m
#pragma once

__device__ __forceinline__ void quantize_block_q3_1_vec(
    const float* __restrict__ src,
    block_q3_1* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;

    float4 v;
    float local_max = -3.402823466e+38f;
    float local_min = 3.402823466e+38f;
    if (lane < 8) {
        v = reinterpret_cast<const float4*>(src)[lane];
        local_max = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
        local_min = fminf(fminf(v.x, v.y), fminf(v.z, v.w));
    }

    float vmax = local_max, vmin = local_min;
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(0xff, vmax, offset, 8));
        vmin = fminf(vmin, __shfl_xor_sync(0xff, vmin, offset, 8));
    }
    vmax = __shfl_sync(0xffffffff, vmax, 0, 32);
    vmin = __shfl_sync(0xffffffff, vmin, 0, 32);

    const float d = (vmax - vmin) / 7.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    if (lane < 8) {
        auto encode = [&](float x) -> uint8_t {
            return (uint8_t)fminf(7.0f, fmaxf(0.0f, roundf((x - vmin) * id)));
        };

        uint8_t q0 = encode(v.x);
        uint8_t q1 = encode(v.y);
        uint8_t q2 = encode(v.z);
        uint8_t q3 = encode(v.w);

        // Low 2 bits: qs[lane] = (q0&3) | ((q1&3)<<2) | ((q2&3)<<4) | ((q3&3)<<6)
        dst->qs[lane] = (q0 & 3) | ((q1 & 3) << 2) | ((q2 & 3) << 4) | ((q3 & 3) << 6);

        // High bit: qh bytes
        // Elements: lane*4+0..lane*4+3 map to indices in qh
        // qh[j/8] bit (j%8) = high bit of element j
        int base = lane * 4; // element indices: base, base+1, base+2, base+3
        uint8_t hi_bits = 0;
        hi_bits |= ((q0 >> 2) & 1) << (base & 7);
        hi_bits |= ((q1 >> 2) & 1) << ((base + 1) & 7);
        hi_bits |= ((q2 >> 2) & 1) << ((base + 2) & 7);
        hi_bits |= ((q3 >> 2) & 1) << ((base + 3) & 7);

        // Lanes 0-1 write qh[0], lanes 2-3 write qh[1], etc.
        uint8_t partner_hi = (uint8_t)__shfl_xor_sync(0xff, (int)hi_bits, 1, 8);
        if ((lane & 1) == 0) {
            dst->qh[lane >> 1] = hi_bits | partner_hi;
        }
    }
    if (lane == 0)
        dst->dm = __halves2half2(__float2half_rn(d), __float2half_rn(vmin));
}

__device__ __forceinline__ void quantize_block_q3_1(
    const float* __restrict__ src,
    block_q3_1* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];

    float vmax = xi, vmin = xi;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, offset, 32));
        vmin = fminf(vmin, __shfl_xor_sync(0xffffffff, vmin, offset, 32));
    }

    const float d = (vmax - vmin) / 7.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const uint8_t q = (uint8_t)fminf(7.0f, fmaxf(0.0f, roundf((xi - vmin) * id)));

    // Low 2 bits: pack groups of 4
    uint8_t lo = (q & 3) << (2 * (lane & 3));
    lo |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)lo, 1, 32);
    lo |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)lo, 2, 32);
    if ((lane & 3) == 0)
        dst->qs[lane >> 2] = lo;

    // High bit: pack groups of 8
    uint8_t hi = ((q >> 2) & 1) << (lane & 7);
    #pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1)
        hi |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)hi, offset, 32);
    if ((lane & 7) == 0)
        dst->qh[lane >> 3] = hi;

    if (lane == 0)
        dst->dm = __halves2half2(__float2half_rn(d), __float2half_rn(vmin));
}

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q3_1(
    const float* __restrict__ src,
    block_q3_1* __restrict__ dst,
    int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks;
         blk += warps_per_block * gridDim.x) {

        const float* block_src = src + blk * QK3_1;
        block_q3_1* block_dst = dst + blk;

        const float xi = block_src[lane];
        float vmax = xi, vmin = xi;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, offset, 32));
            vmin = fminf(vmin, __shfl_xor_sync(0xffffffff, vmin, offset, 32));
        }

        const float d = (vmax - vmin) / 7.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        const uint8_t q = (uint8_t)fminf(7.0f, fmaxf(0.0f, roundf((xi - vmin) * id)));

        uint8_t lo = (q & 3) << (2 * (lane & 3));
        lo |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)lo, 1, 32);
        lo |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)lo, 2, 32);
        if ((lane & 3) == 0)
            block_dst->qs[lane >> 2] = lo;

        uint8_t hi = ((q >> 2) & 1) << (lane & 7);
        #pragma unroll
        for (int offset = 1; offset < 8; offset <<= 1)
            hi |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)hi, offset, 32);
        if ((lane & 7) == 0)
            block_dst->qh[lane >> 3] = hi;

        if (lane == 0)
            block_dst->dm = __halves2half2(__float2half_rn(d), __float2half_rn(vmin));
    }
}
