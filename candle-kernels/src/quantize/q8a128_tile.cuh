#pragma once

// One q8a128 tile, written by the warp that holds it.
//
// The single definition of the q8a128 quantization arithmetic. A warp holds one
// 128-element tile, lane `l` owning elements [4l, 4l+4); the per-lane amax and Σ
// reduce across the warp in a five-step xor butterfly, and lane 0 writes the
// tile's (scale, sum). Every producer of the operand calls this — the standalone
// `quantize_q8a128_kernel` and each fused epilogue that emits the operand from
// values it already holds — so a fused producer writes, from the same floats,
// exactly the bytes the standalone quantize would have.
//
// `sum_norm`: 0 stores the raw Σx, 1 stores Σx/amax (see blocks.cuh). `id`
// carries the amax==0 guard, so a dead tile stores {0, 0} either way.
//
// The whole warp must call it together: the butterfly is a full-mask shuffle.

#include "../blocks.cuh"
#include <cuda_fp16.h>

__device__ __forceinline__ void emit_q8a128_tile(
    uint8_t* __restrict__ obytes, int tile, int lane,
    float x0, float x1, float x2, float x3, int sum_norm)
{
    float amax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
    // `__fadd_rn`, not `+`: a fused producer hands over x0..x3 straight from multiplies,
    // and nvcc's default `-fmad` could contract a multiply into this sum as an FMA — a
    // different Σx from the standalone quantize, whose inputs come from memory. The
    // intrinsic is never contracted, so every producer sums the same rounded floats.
    float s = __fadd_rn(__fadd_rn(__fadd_rn(x0, x1), x2), x3);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, 32));
        s += __shfl_xor_sync(0xffffffff, s, off, 32);
    }
    // Every division and multiply below is an IEEE round-to-nearest intrinsic.
    // This header is compiled into each producer's own translation unit, and a
    // unit built with `-prec-div=false` / `--use_fast_math` would otherwise compute
    // `127/amax` approximately — one ulp of `id` is enough to move a value sitting
    // on a .5 boundary to the next int8 code, and the fused producer's bytes stop
    // being the standalone quantize's. The intrinsics are the same instructions
    // whatever the unit's flags.
    const float id = (amax != 0.f) ? __fdiv_rn(127.f, amax) : 0.f;

    // q8a1024 flat-grouped placement: qs and ds de-interleaved into the tile's
    // super-block slot (see blocks.cuh).
    *reinterpret_cast<char4*>(obytes + q8a1024_qs_off(tile) + lane * 4) = make_char4(
        (int8_t)__float2int_rn(__fmul_rn(x0, id)),
        (int8_t)__float2int_rn(__fmul_rn(x1, id)),
        (int8_t)__float2int_rn(__fmul_rn(x2, id)),
        (int8_t)__float2int_rn(__fmul_rn(x3, id)));
    if (lane == 0) {
        // Raw Σx, or Σx normalised by amax: |Σx/amax| ≤ 128 whatever the
        // activation's magnitude, where the raw Σx overflows f16 above 65504.
        // `s` passes through unmultiplied on the raw arm, so those bytes are the
        // plain sum's.
        const float s_store = sum_norm ? __fmul_rn(__fmul_rn(s, id), 1.f / 127.f) : s;
        half2* ds = reinterpret_cast<half2*>(obytes + q8a1024_ds_off(tile));
        ds[0] = make_half2(__float2half_rn(__fdiv_rn(amax, 127.f)), __float2half_rn(s_store));
    }
}
