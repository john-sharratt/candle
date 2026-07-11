#pragma once
// ============================================================================
// Palette routing rank computation — shared by the FP16 prefill kernel and
// the INT8 prefix-attention prefill kernel.
//
// A KvHead's pal_map packs one 2-bit palette id per head dimension
// (little-endian, 4 dims per byte). A dimension's storage location inside
// its palette's arena region is its RANK: the number of lower-indexed
// dimensions routed to the same palette. This computes both in O(HD/16)
// word ops via XOR-match + popcount.
// ============================================================================

#include <stdint.h>

/// Compute palette index and rank-within-palette for global dim `tid`.
/// pal_map: HD/4 bytes, 2 bits per dim, little-endian packed.
__device__ __forceinline__ void prefill_pal_rank(
    const uint8_t* pal_map, int tid, int* out_p, int* out_rank)
{
    int my_p = (pal_map[tid >> 2] >> (2 * (tid & 3))) & 0x3;
    const uint32_t* pm = (const uint32_t*)pal_map;
    int word_idx = tid >> 4;
    int partial  = tid & 15;

    auto match = [my_p](uint32_t w) -> uint32_t {
        uint32_t b0 = w & 0x55555555u;
        uint32_t b1 = (w >> 1) & 0x55555555u;
        uint32_t m0 = (my_p & 1) ? b0 : (~b0 & 0x55555555u);
        uint32_t m1 = (my_p & 2) ? b1 : (~b1 & 0x55555555u);
        return m0 & m1;
    };

    int rank = 0;
    for (int i = 0; i < word_idx; i++) {
        rank += __popc(match(pm[i]));
    }
    if (partial > 0) {
        uint32_t m = match(pm[word_idx]);
        m &= (1u << (partial * 2)) - 1;
        rank += __popc(m);
    }

    *out_p    = my_p;
    *out_rank = rank;
}
