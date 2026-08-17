#pragma once

// =============================================================================
// MXFP4_KO DEQUANT TRAIT — per-sub fold for the q8a128 int8 path
// =============================================================================
// MXFP4 stores per-32 blocks: 16 packed E2M1 nibbles (codebook indices [0,15])
// + one E8M0 power-of-two scale byte. The int8 tensor cores run one m16n8k32
// MMA per 32-K sub anyway, so each sub's int32 accumulator is folded with its
// OWN scale `2^(e_sub-128)` in FP32 (`is_mxfp4_persub` branch in kernel.cuh)
// — the dequant here is therefore a PURE codebook expansion: nibble → signed
// int8 magnitude, no per-element arithmetic beyond the table lookup. (A prior
// design aligned the four subs onto their common e_max by shifting each
// decoded mantissa right — ncu measured those per-element shift-rounds as
// 13.4B ALU instructions per prefill-scale launch, HALF the kernel's whole
// instruction stream, with the tensor pipe near-idle. The per-sub FP fold
// deletes them; it is also the more faithful arithmetic — no truncation.)
//
// Storage layout is unchanged (512 B nibbles + 32 B per-sub E8M0 + 32 B baked
// per-row dm): existing expert-pack files remain valid. The baked `dm` is not
// read on this path — the per-sub scales supersede it — but stays in the
// layout so the repack format and its fingerprint are untouched.
//
// The ×0.5 that turns the integer codebook {0,1,2,3,4,6,8,12} back into the
// real E2M1 magnitudes {0,.5,1,1.5,2,3,4,6} is folded into the "half" scale
// decode, exactly mirroring `ko_quant::e8m0_to_f32_half`.

#include "gemx_dequant.cuh"   // gemx_dequant_traits base template + block types
#include "q4_K.cuh"           // gemx_dequant_traits<block_c_q4_KO> base (inherited)

// E2M1 magnitude table {0,1,2,3,4,6,8,12} packed as nibbles (idx 0→low), sign in bit 3.
// Branch-free, register-only — no constant-memory table, so this header is ODR-safe across
// every translation unit that includes it.
__device__ __forceinline__ int mxfp4_kvalue_i8(int nib) {
    const int mag = (int)((0xC8643210u >> ((nib & 7) * 4)) & 0xF);
    return (nib & 8) ? -mag : mag;
}

// Map 4 MXFP4 nibbles (one per byte of `nib4`, each already masked to [0,15]) → 4
// signed int8 codebook bytes, packed back into a uint32 the m16n8k32 MMA reads.
__device__ __forceinline__ uint32_t mxfp4_codebook_x4(uint32_t nib4) {
    uint32_t out = 0;
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int nib = (int)((nib4 >> (b * 8)) & 0xF);
        out |= ((uint32_t)((uint8_t)(int8_t)mxfp4_kvalue_i8(nib))) << (b * 8);
    }
    return out;
}

// `2^(e-128)` as fp32, subnormal-aware — the exact mirror of
// `ko_quant::e8m0_to_f32_half` (the CPU repack/oracle side).
__device__ __forceinline__ float mxfp4_e8m0_half(int e) {
    const uint32_t bits = e < 2 ? (0x00200000u << e) : ((uint32_t)(e - 1) << 23);
    return __uint_as_float(bits);
}

// MXFP4_KO K/1024 chunk — LANE-MAJOR, mirrors the block_c_q4_KO_k1024 quant read (one int4
// LDS at lane*16 pulls this lane's 4 sub-uint32s) but decodes MXFP4 codebook nibbles.
template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_mxfp4_k1024, compute_t, scale_t>
    : gemx_dequant_traits<block_c_q4_KO, compute_t, scale_t> {
    __device__ __forceinline__ static void dequant_all_subs_int8(
        const uint8_t* __restrict__ chunk, int lane, uint32_t (&b_frags)[4][2])
    {
        // This lane's 4 subs: low nibbles → K[q3*4 .. +3], high nibbles → K[q3*4+16 .. +19].
        const int4 vv = *reinterpret_cast<const int4*>(chunk + lane * 16);
        const uint32_t s4[4] = {(uint32_t)vv.x, (uint32_t)vv.y, (uint32_t)vv.z, (uint32_t)vv.w};
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            b_frags[sub][0] = mxfp4_codebook_x4(s4[sub] & 0x0F0F0F0Fu);
            b_frags[sub][1] = mxfp4_codebook_x4((s4[sub] >> 4) & 0x0F0F0F0Fu);
        }
    }

    // The two FOLD rows this thread owns (`rl`, `rl+1` in the kernel's C
    // fragment layout): their four per-sub E8M0 scales each, decoded to fp32.
    // One 8-byte load from the chunk's scale tail.
    __device__ __forceinline__ static void load_sub_scales(
        const uint8_t* __restrict__ chunk, int rl, float (&f0)[4], float (&f1)[4])
    {
        const uchar4 e0 = *reinterpret_cast<const uchar4*>(chunk + 512 + rl * 4);
        const uchar4 e1 = *reinterpret_cast<const uchar4*>(chunk + 512 + (rl + 1) * 4);
        f0[0] = mxfp4_e8m0_half(e0.x);
        f0[1] = mxfp4_e8m0_half(e0.y);
        f0[2] = mxfp4_e8m0_half(e0.z);
        f0[3] = mxfp4_e8m0_half(e0.w);
        f1[0] = mxfp4_e8m0_half(e1.x);
        f1[1] = mxfp4_e8m0_half(e1.y);
        f1[2] = mxfp4_e8m0_half(e1.z);
        f1[3] = mxfp4_e8m0_half(e1.w);
    }
};
