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

/// The dims of palette `p` among the 16 a map word routes — one bit (at the
/// dim's even position) per dim whose 2-bit id is `p`.
__device__ __forceinline__ uint32_t pal_word_match(uint32_t w, int p)
{
    const uint32_t b0 = w & 0x55555555u;
    const uint32_t b1 = (w >> 1) & 0x55555555u;
    const uint32_t m0 = (p & 1) ? b0 : (~b0 & 0x55555555u);
    const uint32_t m1 = (p & 2) ? b1 : (~b1 & 0x55555555u);
    return m0 & m1;
}

/// Every dim's rank byte (`palette << 6 | rank`) of one palette map, written
/// to `rank[HD]` — `prefill_pal_rank` for dims `{lane + 32w}`, computed by the
/// warp from the map words it already holds (lane `l < HD/16` holds word `l`,
/// any value elsewhere) rather than by re-reading the map from global memory
/// one word per earlier word of each dim. Warp-collective.
///
/// A dim's rank is the per-palette count of every earlier word plus the
/// matching dims below it in its own word: the first term is an exclusive
/// scan over the words, one per palette, read back with a shuffle.
template <int HD>
__device__ __forceinline__ void i8_rank_from_map_words(uint32_t word, int lane, uint8_t* rank)
{
    constexpr int MAP_WORDS = HD / 16;
    static_assert(MAP_WORDS <= 32, "a palette map must fit one word per lane");
    static_assert(HD <= 0xffff, "a rank count packs into 16 bits");
    // Exclusive per-palette prefix counts over the words, packed two to a word.
    uint32_t pack01 = 0u, pack23 = 0u;
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t own = (lane < MAP_WORDS) ? (uint32_t)__popc(pal_word_match(word, p)) : 0u;
        uint32_t incl = own;
        #pragma unroll
        for (int off = 1; off < MAP_WORDS; off <<= 1) {
            const uint32_t t = __shfl_up_sync(0xffffffffu, incl, off);
            if (lane >= off) incl += t;
        }
        const uint32_t excl = incl - own;
        if (p < 2) pack01 |= excl << (16 * p);
        else       pack23 |= excl << (16 * (p - 2));
    }
    #pragma unroll
    for (int w = 0; w < HD / 32; ++w) {
        const int d = lane + 32 * w;
        const int widx = d >> 4;
        const int partial = d & 15;
        const uint32_t wv = __shfl_sync(0xffffffffu, word, widx);
        const uint32_t p01 = __shfl_sync(0xffffffffu, pack01, widx);
        const uint32_t p23 = __shfl_sync(0xffffffffu, pack23, widx);
        const int p = (int)((wv >> (2 * partial)) & 0x3u);
        const uint32_t before = (p < 2) ? (p01 >> (16 * p)) & 0xffffu
                                        : (p23 >> (16 * (p - 2))) & 0xffffu;
        const uint32_t below = pal_word_match(wv, p) & ((1u << (partial * 2)) - 1u);
        const int r = (int)before + __popc(below);
        rank[d] = (uint8_t)((p << 6) | r);
    }
}
