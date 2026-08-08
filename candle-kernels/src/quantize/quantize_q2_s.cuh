// SPDX-License-Identifier: MIT
// Q2_S Quantization: float -> 2-bit symmetric with INT8 scale
// Block: { int8_t scale; uint8_t qs[8]; } = 9 bytes = 2.25 BPE
// decode: d * (q - 1.5)  where q in [0,3], d = scale/127.0, d = amax/1.5
// Outer palette scale provides range; no FP8 used.
#pragma once

__device__ __forceinline__ void quantize_block_q2_s_vec(
    const float* __restrict__ src,
    block_q2_s* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;

    float4 v;
    float local_max = 0.0f;
    if (lane < 8) {
        v = reinterpret_cast<const float4*>(src)[lane];
        local_max = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                           fmaxf(fabsf(v.z), fabsf(v.w)));
    }

    float amax = local_max;
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xff, amax, offset, 8));
    amax = __shfl_sync(0xffffffff, amax, 0, 32);

    // Encode d as INT8 then decode for round-trip consistency.
    const int8_t scale = (int8_t)__float2int_rn(fminf(127.0f, (amax * (1.0f / 1.5f)) * 127.0f));
    const float d = (float)scale * (1.0f / 127.0f);
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    if (lane < 8) {
        auto encode = [&](float x) -> uint8_t {
            return (uint8_t)fminf(3.0f, fmaxf(0.0f, roundf(x * id + 1.5f)));
        };
        dst->qs[lane] = encode(v.x) | (encode(v.y) << 2)
                        | (encode(v.z) << 4) | (encode(v.w) << 6);
    }

    if (lane == 0)
        dst->scale = scale;
}

__device__ __forceinline__ void quantize_block_q2_s(
    const float* __restrict__ src,
    block_q2_s* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];

    float amax = fabsf(xi);
    amax = quantize_warp_reduce_max(amax);

    // Encode d as INT8 then decode for round-trip consistency.
    const int8_t scale = (int8_t)__float2int_rn(fminf(127.0f, (amax * (1.0f / 1.5f)) * 127.0f));
    const float d = (float)scale * (1.0f / 127.0f);
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const uint8_t q2 = (uint8_t)fminf(3.0f, fmaxf(0.0f, roundf(xi * id + 1.5f)));

    uint8_t packed = (q2 & 3) << (2 * (lane & 3));
    packed |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)packed, 1, 32);
    packed |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)packed, 2, 32);

    if ((lane & 3) == 0)
        dst->qs[lane >> 2] = packed;

    if (lane == 0)
        dst->scale = scale;
}

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q2_s(
    const float* __restrict__ src,
    block_q2_s* __restrict__ dst,
    int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks;
         blk += warps_per_block * gridDim.x) {

        const float* block_src = src + blk * QK2_S;
        block_q2_s* block_dst = dst + blk;

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

        const int8_t scale = (int8_t)__float2int_rn(fminf(127.0f, (amax * (1.0f / 1.5f)) * 127.0f));
        const float d = (float)scale * (1.0f / 127.0f);
        if (lane < 8) {
            const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
            auto encode = [&](float x) -> uint8_t {
                return (uint8_t)fminf(3.0f, fmaxf(0.0f, roundf(x * id + 1.5f)));
            };
            block_dst->qs[lane] = encode(v.x) | (encode(v.y) << 2)
                                  | (encode(v.z) << 4) | (encode(v.w) << 6);
        }
        if (lane == 0)
            block_dst->scale = scale;
    }
}
