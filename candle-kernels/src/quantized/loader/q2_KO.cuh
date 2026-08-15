#pragma once

// =============================================================================
// Q2_KO DEQUANT TRAIT — 2-bit affine KO twin for the q8a128 int8 path
// =============================================================================
// Q2_KO is the smallest KO weight: per-128 affine (scale, min), value 0..3. Its 256 B quant
// region is the SAME 2-bit crumb stream Q6_KO carries for its high-2-bits — for `(lane q3, sub)`
// the uint16 at `lane*8 + sub*2` is `{cr0, cr1}`, `cr0` packing the 4 LOW-half values
// (K = q3*4 + {0..3}) at bit positions 0,2,4,6 and `cr1` the 4 HIGH-half values — but here the
// crumb IS the whole value, not Q6's high two bits. The int8 fold in kernel.cuh reads the shared
// per-128 (scale, min) from `dm[row]` (chunk tail), so this trait only produces the unsigned int8
// b_frags. Byte-identical to CPU `ko_quant::quantize_q2_ko`.

#include "gemx_dequant.cuh"   // gemx_dequant_traits base template + block types
#include "q4_K.cuh"           // gemx_dequant_traits<block_c_q4_KO> base (inherited: has_min, sub_dm)

// Spread a byte's four 2-bit crumbs (bits 2j = value j) into four int8 lanes of a uint32
// (byte j = value j, in [0,3]). Same pattern as the q2_K loader's `unpack_field_int8`.
__device__ __forceinline__ uint32_t q2ko_spread_crumbs(uint32_t c) {
    return (c & 0x03u) | ((c & 0x0Cu) << 6) | ((c & 0x30u) << 12) | ((c & 0xC0u) << 18);
}

// Q2_KO K/1024 chunk — LANE-MAJOR. ONE int2 LDS at `lane*8` pulls this lane's 4 subs'
// crumb-uint16s (the 256 B quant region, no separate ql). Per sub: `cr0` → b_frag[0] (the 4
// low-half values K[q3*4 .. +3]), `cr1` → b_frag[1] (the 4 high-half values K[q3*4+16 .. +19]),
// each crumb spread into bits 0-1 of its output byte. Values UNSIGNED (0..3); the per-128
// (scale, min) fold handles the offset. Mirrors the Q6_KO k1024 read minus the 4-bit ql.
template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q2_KO_k1024, compute_t, scale_t>
    : gemx_dequant_traits<block_c_q4_KO, compute_t, scale_t> {
    __device__ __forceinline__ static void dequant_all_subs_int8(
        const uint8_t* __restrict__ chunk, int lane, uint32_t (&b_frags)[4][2])
    {
        const int2 cc = *reinterpret_cast<const int2*>(chunk + lane * 8);
        const uint32_t c2[2] = {(uint32_t)cc.x, (uint32_t)cc.y};  // c2[0]: subs 0,1  c2[1]: subs 2,3
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const uint32_t qh16 = (c2[sub >> 1] >> ((sub & 1) * 16)) & 0xFFFFu;
            b_frags[sub][0] = q2ko_spread_crumbs(qh16 & 0xFFu);          // cr0: low-half values
            b_frags[sub][1] = q2ko_spread_crumbs((qh16 >> 8) & 0xFFu);   // cr1: high-half values
        }
    }
};
