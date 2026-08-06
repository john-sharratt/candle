#pragma once

// =============================================================================
// MXFP4_KO DEQUANT TRAIT — native-MXFP4 exponent-collapse for the q8a128 int8 path
// =============================================================================
// MXFP4 stores per-32 blocks: 16 packed E2M1 nibbles (codebook indices [0,15]) + one
// E8M0 power-of-two scale byte. The int8 tensor-core fold wants ONE scale per 128-K row
// so the four k32 sub-MMAs collapse into a single int32. Because the E8M0 scale is a
// PURE power of two, we can align the four per-32 subs of a row onto their common largest
// exponent `e_max` by shifting each sub's decoded int8 mantissa right by (e_max - e_sub) —
// the byte difference IS the exponent difference (E8M0 is uniform, even across the
// subnormal boundary). After the shift all four subs share the single per-128 scale
// `2^(e_max-128)` (precomputed into `dm[row]` at repack, min=0 since E2M1 is centred),
// so the generic int8 fold in kernel.cuh consumes them unchanged. Weights stay 4-bit in
// storage; the collapse is purely in-register here. Validated bit-for-bit against the CPU
// oracle `ko_quant::mxfp4_collapse_int8_matmul`.

#include "gemx_dequant.cuh"   // gemx_dequant_traits base template + block types
#include "q4_K.cuh"           // gemx_dequant_traits<block_c_q4_KO> base (inherited)

// E2M1 magnitude table {0,1,2,3,4,6,8,12} packed as nibbles (idx 0→low), sign in bit 3.
// Branch-free, register-only — no constant-memory table, so this header is ODR-safe across
// every translation unit that includes it.
__device__ __forceinline__ int mxfp4_kvalue_i8(int nib) {
    const int mag = (int)((0xC8643210u >> ((nib & 7) * 4)) & 0xF);
    return (nib & 8) ? -mag : mag;
}

// Round `w / 2^shift` to the nearest integer, half AWAY from zero — matches the CPU
// `ko_quant::shift_round`. `shift` is small (0..a few) so the divide is cheap.
__device__ __forceinline__ int mxfp4_shift_round(int w, int shift) {
    if (shift == 0) return w;
    const int denom = 1 << shift;
    const int half = denom >> 1;
    return w >= 0 ? (w + half) / denom : -(((-w) + half) / denom);
}

// Map 4 MXFP4 nibbles (one per byte of `nib4`, each already masked to [0,15]) → 4
// shifted-codebook signed int8 bytes, packed back into a uint32 the m16n8k32 MMA reads.
__device__ __forceinline__ uint32_t mxfp4_codebook_shift_x4(uint32_t nib4, int shift) {
    uint32_t out = 0;
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int nib = (int)((nib4 >> (b * 8)) & 0xF);
        const int v = mxfp4_shift_round(mxfp4_kvalue_i8(nib), shift);
        out |= ((uint32_t)((uint8_t)(int8_t)v)) << (b * 8);
    }
    return out;
}

// MXFP4_KO K/1024 chunk — LANE-MAJOR, mirrors the block_c_q4_KO_k1024 quant read (one int4
// LDS at lane*16 pulls this lane's 4 sub-uint32s) but decodes MXFP4 codebook nibbles and
// applies the per-sub exponent-collapse shift. `dm[row]` (read by the kernel fold) already
// holds the collapsed per-128 scale; here we only produce the shifted int8 b_frags.
template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_mxfp4_k1024, compute_t, scale_t>
    : gemx_dequant_traits<block_c_q4_KO, compute_t, scale_t> {
    __device__ __forceinline__ static void dequant_all_subs_int8(
        const uint8_t* __restrict__ chunk, int lane, uint32_t (&b_frags)[4][2])
    {
        const int row = lane >> 2;
        // This lane's 4 subs: low nibbles → K[q3*4 .. +3], high nibbles → K[q3*4+16 .. +19].
        const int4 vv = *reinterpret_cast<const int4*>(chunk + lane * 16);
        const uint32_t s4[4] = {(uint32_t)vv.x, (uint32_t)vv.y, (uint32_t)vv.z, (uint32_t)vv.w};
        // This row's four per-sub E8M0 scale bytes (chunk tail, after the 512 B nibble region).
        const uchar4 eb = *reinterpret_cast<const uchar4*>(chunk + 512 + row * 4);
        const int e0 = (int)eb.x, e1 = (int)eb.y, e2 = (int)eb.z, e3 = (int)eb.w;
        const int emax = max(max(e0, e1), max(e2, e3));
        const int esub[4] = {e0, e1, e2, e3};
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const int shift = emax - esub[sub];   // byte diff == exponent diff (pure pow-2)
            b_frags[sub][0] = mxfp4_codebook_shift_x4(s4[sub] & 0x0F0F0F0Fu, shift);
            b_frags[sub][1] = mxfp4_codebook_shift_x4((s4[sub] >> 4) & 0x0F0F0F0Fu, shift);
        }
    }
};
