#pragma once

// =============================================================================
// Q3_KO DEQUANT TRAIT — 3-bit affine KO twin for the q8a128 int8 path
// =============================================================================
// Q3_KO is `Q3_K`'s same-width twin, and it exists because there was none: `to_ko` used to
// round Q3_K *up* to Q4_KO, which costs nothing while weights are resident and is PCIe bytes on
// every forward once they stream (`docs/qwen38_layer_streaming.md` §2.3).
//
// It carries NO `ql` plane. A 3-bit value is spent across the two auxiliary planes the other
// twins already define, and this trait is literally their two reads composed:
//
//   bits 0-1  the 256 B **crumb** region, byte-identical to Q2_KO's — for `(lane q3, sub)` the
//             uint16 at `lane*8 + sub*2` is `{cr0, cr1}`, `cr0` packing the 4 LOW-half values
//             (K = q3*4 + {0..3}) at bit positions 0,2,4,6 and `cr1` the 4 HIGH-half values.
//   bit 2     the 128 B **hi** region at offset 256, byte-identical to Q5_KO's 5th-bit region —
//             the byte at `256 + lane*4 + sub` holds `hb0` (low half) in its low nibble and
//             `hb1` (high half) in its high nibble, bit `i` of each serving K index `i`.
//
// The ONLY difference from Q5_KO's hi read is where the bit lands: `<< 2` here (above the
// crumb) against Q5_KO's `<< 4` (above the ql). The int8 fold in kernel.cuh reads the shared
// per-128 (scale, min) from `dm[row]` at the chunk tail (offset 384), so this trait produces
// only the unsigned int8 b_frags. Byte-identical to CPU `ko_quant::quantize_ko(.., Q3_KO)`.

#include "gemx_dequant.cuh"   // gemx_dequant_traits base template + block types
#include "q4_K.cuh"           // gemx_dequant_traits<block_c_q4_KO> base (inherited: has_min, sub_dm)
#include "q2_KO.cuh"          // q2ko_spread_crumbs — the crumb plane is byte-identical

// Q3_KO K/1024 chunk — LANE-MAJOR. One int2 LDS at `lane*8` pulls this lane's 4 subs' crumb
// uint16s (256 B region, no ql); one uint32 LDS at `256 + lane*4` pulls its 4 subs' hi bytes.
// Per sub: crumb `cr0` + hi `hb0` → b_frag[0] (the 4 low-half values K[q3*4 .. +3]), `cr1` +
// `hb1` → b_frag[1] (the 4 high-half values K[q3*4+16 .. +19]). Values UNSIGNED (0..7); the
// per-128 (scale, min) fold handles the offset.
template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q3_KO_k1024, compute_t, scale_t>
    : gemx_dequant_traits<block_c_q4_KO, compute_t, scale_t> {
    __device__ __forceinline__ static void dequant_all_subs_int8(
        const uint8_t* __restrict__ chunk, int lane, uint32_t (&b_frags)[4][2])
    {
        const int2 cc = *reinterpret_cast<const int2*>(chunk + lane * 8);
        const uint32_t c2[2] = {(uint32_t)cc.x, (uint32_t)cc.y};  // c2[0]: subs 0,1  c2[1]: subs 2,3
        const uint32_t hi4 = *reinterpret_cast<const uint32_t*>(chunk + 256 + lane * 4);
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const uint32_t qh16 = (c2[sub >> 1] >> ((sub & 1) * 16)) & 0xFFFFu;
            const uint32_t hbb = (hi4 >> (sub * 8)) & 0xFFu;
            const uint32_t hb0 = hbb & 0xFu;
            const uint32_t hb1 = (hbb >> 4) & 0xFu;
            // `hb * 0x00204081 & 0x01010101` spreads the 4 bits one per output byte (bit 0 of
            // byte j = bit j of hb); `<< 2` seats each above its 2-bit crumb.
            b_frags[sub][0] = q2ko_spread_crumbs(qh16 & 0xFFu)
                            | (((hb0 * 0x00204081u) & 0x01010101u) << 2);
            b_frags[sub][1] = q2ko_spread_crumbs((qh16 >> 8) & 0xFFu)
                            | (((hb1 * 0x00204081u) & 0x01010101u) << 2);
        }
    }
};
