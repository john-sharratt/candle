// Quad loads, their dtype dispatch and conversion, and the palette decode of
// the block path.
//
// Holds the vector-readable quad loads and their dtype dispatch
// (`quad_raw_load`, `quad_raw_cvt`, `quad_pair_dispatch`), and the per-quad
// probes and `tile_quads_vec` that read a pair of quads.
// Holds `tile_stage_ptr` and `tile_k_block_decode_palettes`, which decode the
// palettes of a group by palette and rank.

/// What a lane knows about one quad of one side before decoding it: the
/// rank-table word covering its four dims, the group's live mask and
/// first token, the palette format of its first dim, and whether the four
/// ranks are consecutive from a multiple of 4 inside ONE palette (`vec`:
/// a dtype palette is then one vector per token).
struct QuadInfo {
    uint32_t tbw, live, fmt;
    int within;
    bool vec;
};
template <int HEAD_DIM>
__device__ __forceinline__ QuadInfo tile_quad_info(
    const TileExt& ext, const TileGroups& tg, const uint8_t* s_tbl, int side, int gi, int d0)
{
    QuadInfo q;
    const uint32_t desc = tg.desc[gi];
    q.live = grp_mask(desc);
    q.within = grp_within(desc);
    q.tbw = *(const uint32_t*)&s_tbl[(gi * 2 + side) * HEAD_DIM + d0];
    const uint32_t b0 = q.tbw & 0xffu;
    q.vec = (q.tbw - 0x03020100u) == b0 * 0x01010101u && (b0 & 3u) == 0u;
    q.fmt = ext.fmt[gi][side][(b0 >> 6) & (N_PALETTE - 1)];
    return q;
}

/// Two vector-readable quads as loaded, before conversion: per quad, per
/// token, the words of its four dims' vector (four for F32, two for the
/// 16-bit formats, one for FP8; the rest are not written). `meta` packs
/// what the convert needs — byte 0 and 1 the quads' formats, nibbles 4
/// and 5 their live masks — and `inv` their palette scale reciprocals.
///
/// `W` is the widest format the holder admits, and is what the words cost
/// in registers: a pair converted in the same breath as it is loaded
/// (`tile_quads_vec`, the K side) holds four, so F32 rides the vector
/// path too; a pair held in flight across other work (the V side, loaded
/// ahead of QK and converted after the softmax) holds two — the width of
/// the formats the arenas actually store, `i8_is_narrow_dtype_format` —
/// so the held words cost half the converted floats' registers rather
/// than all of them. F32 on that side decodes through the block path.
template <int W>
struct QuadRaw {
    uint32_t w[2][GROUP_TOK][W];
    uint32_t meta;
    float inv[2];
};

__device__ __forceinline__ uint32_t quad_raw_meta(
    uint32_t fmt0, uint32_t fmt1, uint32_t live0, uint32_t live1)
{
    return fmt0 | (fmt1 << 8) | (live0 << 16) | (live1 << 20);
}

/// Where a pair of vector-readable quads is read from: per quad its
/// palette base, rank, first token and format. A DEAD quad (live mask 0)
/// takes the live quad's address and format so its loads issue with the
/// other's instead of behind a branch of their own — the convert discards
/// its words. `any` is false when neither quad is live.
struct QuadPair {
    const char* base[2];
    int rank[2];
    int within[2];
    uint32_t fmt[2];
    bool any;
};

template <int HEAD_DIM, int W>
__device__ __forceinline__ QuadPair tile_quads_vec_resolve(
    const TileExt& ext, const QuadInfo (&q)[2], int side, const int (&gi)[2], QuadRaw<W>& raw)
{
    QuadPair pr;
    const bool d0 = q[0].live == 0u, d1 = q[1].live == 0u;
    pr.any = !(d0 && d1);
    // Selects, not an index into q[] — a runtime index would put the
    // pair in local memory.
    const uint32_t tbw0 = d0 ? q[1].tbw : q[0].tbw;
    const uint32_t tbw1 = d1 ? q[0].tbw : q[1].tbw;
    const int g0 = d0 ? gi[1] : gi[0];
    const int g1 = d1 ? gi[0] : gi[1];
    pr.within[0] = d0 ? q[1].within : q[0].within;
    pr.within[1] = d1 ? q[0].within : q[1].within;
    pr.fmt[0] = d0 ? q[1].fmt : q[0].fmt;
    pr.fmt[1] = d1 ? q[0].fmt : q[1].fmt;
    const int p0 = (int)(tbw0 >> 6) & (N_PALETTE - 1);
    const int p1 = (int)(tbw1 >> 6) & (N_PALETTE - 1);
    pr.base[0] = ext.gbase[g0][side][p0];
    pr.base[1] = ext.gbase[g1][side][p1];
    pr.rank[0] = (int)(tbw0 & 63u);
    pr.rank[1] = (int)(tbw1 & 63u);
    // Correctly rounded under fast math too: a Q0_V block folds this into its
    // header, and every Q0_V read must agree on it (block_q0_v.cuh).
    raw.inv[0] = ext.inv[g0][side][p0];
    raw.inv[1] = ext.inv[g1][side][p1];
    raw.meta = quad_raw_meta(pr.fmt[0], pr.fmt[1], q[0].live, q[1].live);
    return pr;
}

/// The loads of quad `i`: one aligned vector per token, four back to
/// back. The group's four tokens lie inside the slice (an aligned quad
/// of it), so a masked token's load issues with the others.
template <int HEAD_DIM, typename Tag, int W>
__device__ __forceinline__ void quad_raw_load(
    Tag tag, const QuadPair& pr, int i, uint32_t (&w)[GROUP_TOK][W])
{
    constexpr int SUB = HEAD_DIM / N_PALETTE;
    const auto quad = i8_tag_quad(tag, pr.base[i], pr.rank[i], SUB, 1.f);
    #pragma unroll
    for (int j = 0; j < GROUP_TOK; ++j) quad.raw4(pr.within[i] + j, w[j]);
}

/// The convert of one quad: xi[k][j] is dim k at token j. A dead quad is
/// zero. A masked token of a live quad is zeroed on the V side (`side`
/// 1): the tile's per-dim scale is an absmax over all its tokens, and the
/// bytes of a token the slice never wrote are whatever the arena held. On
/// the K side it is kept as read: K's scale is per token, so a masked
/// token's bytes reach only its own column, whose score the softmax
/// forces to -inf regardless of what the column holds — even a NaN, which
/// the select discards. (The K callers skip a dead group warp-uniformly
/// before loading, so a dead quad never reaches this on side 0.)
template <typename Tag, int W>
__device__ __forceinline__ void quad_raw_cvt(
    Tag, const uint32_t (&w)[GROUP_TOK][W], float inv, uint32_t live, int side,
    float (&xi)[4][GROUP_TOK])
{
    using E = typename Tag::Elem;
    #pragma unroll
    for (int j = 0; j < GROUP_TOK; ++j) {
        // A dead token is masked on its packed words, before the conversion
        // (one select per word, not per value): the zero word of every dtype
        // converts to +0.f, the value the mask used to select.
        const bool l = side == 0 || ((live >> j) & 1u);
        uint32_t wm[W];
        #pragma unroll
        for (int k = 0; k < W; ++k) wm[k] = l ? w[j][k] : 0u;
        float o[4];
        i8_dtype_cvt4<E>(wm, o);
        #pragma unroll
        for (int k = 0; k < 4; ++k) xi[k][j] = o[k];
    }
    if (inv != 1.f) {
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            #pragma unroll
            for (int j = 0; j < GROUP_TOK; ++j) xi[k][j] *= inv;
    }
}

/// The vector path's format set, by the width of words a holder keeps per
/// token: `QuadRaw<4>` admits every dtype format, `QuadRaw<2>` the narrow
/// ones.
template <int W>
__device__ __forceinline__ bool quad_raw_admits(uint32_t fmt)
{
    static_assert(W == 2 || W == 4, "a QuadRaw holds two or four words per token");
    if constexpr (W == 4) return i8_is_dtype_format(fmt);
    else                  return i8_is_narrow_dtype_format(fmt);
}
template <int W, typename F>
__device__ __forceinline__ void quad_raw_dispatch(uint32_t fmt, F&& f)
{
    if constexpr (W == 4) i8_with_dtype_format((int)fmt, f);
    else                  i8_with_narrow_dtype_format((int)fmt, f);
}

/// Run `f(tag, lo, hi)` for the quads lo..hi of the pair under their
/// format's tag: one dispatch for both when their formats agree, which
/// they do whenever both groups sit in one arena block, so the body sees
/// both quads at once and can put every load ahead of every convert.
template <int W, typename F>
__device__ __forceinline__ void quad_pair_dispatch(uint32_t fmt0, uint32_t fmt1, F&& f)
{
    if (fmt0 == fmt1) {
        quad_raw_dispatch<W>(fmt0, [&](auto tag) { f(tag, 0, 1); });
    } else {
        quad_raw_dispatch<W>(fmt0, [&](auto tag) { f(tag, 0, 0); });
        quad_raw_dispatch<W>(fmt1, [&](auto tag) { f(tag, 1, 1); });
    }
}

/// Issue the loads of two vector-readable dtype quads — all eight back
/// to back — into `raw`, to be converted later by `tile_quads_vec_cvt`.
/// A pair with no live quad loads nothing.
///
/// Every word is written before the loads issue. The format arms are
/// if-converted into predicated loads, and a predicated load only
/// partially defines its register, so without a full definition ahead of
/// them the words' live ranges reach back through the tile loop to the
/// previous window's — pinning the pair's registers across the whole tile,
/// K side included, rather than from here to the convert.
template <int HEAD_DIM, int W>
__device__ __forceinline__ void tile_quads_vec_load(
    const TileExt& ext, const QuadInfo (&q)[2], int side, const int (&gi)[2], QuadRaw<W>& raw)
{
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < GROUP_TOK; ++j)
            #pragma unroll
            for (int k = 0; k < W; ++k) raw.w[i][j][k] = 0u;
    const QuadPair pr = tile_quads_vec_resolve<HEAD_DIM>(ext, q, side, gi, raw);
    if (!pr.any) return;
    quad_pair_dispatch<W>(pr.fmt[0], pr.fmt[1], [&](auto tag, int lo, int hi) {
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            if (i >= lo && i <= hi) quad_raw_load<HEAD_DIM>(tag, pr, i, raw.w[i]);
    });
}

/// Convert a loaded pair: x[i][k][j] is quad i's dim k at token j.
template <int W>
__device__ __forceinline__ void tile_quads_vec_cvt(
    const QuadRaw<W>& raw, int side, float (&x)[2][4][GROUP_TOK])
{
    const uint32_t live[2] = { (raw.meta >> 16) & 0xfu, (raw.meta >> 20) & 0xfu };
    if ((live[0] | live[1]) == 0u) {
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                #pragma unroll
                for (int j = 0; j < GROUP_TOK; ++j) x[i][k][j] = 0.f;
        return;
    }
    quad_pair_dispatch<W>(raw.meta & 0xffu, (raw.meta >> 8) & 0xffu, [&](auto tag, int lo, int hi) {
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            if (i >= lo && i <= hi) quad_raw_cvt(tag, raw.w[i], raw.inv[i], live[i], side, x[i]);
    });
}

/// One token of one quad on the vector path: o[k] is dim d0 + k of group
/// gi at its token j, converted and scaled; zero for a dead quad or a
/// masked token. The V warps read the ninth quad's tokens through this, a
/// token per lane-octet, under the vote that put the tile on the vector
/// path (every live quad vector-readable, the ninth included).
template <int HEAD_DIM, int W>
__device__ __forceinline__ void tile_quad_vec_token(
    const TileExt& ext, const TileGroups& tg, const uint8_t* s_tbl, int side, int gi, int d0,
    int j, float (&o)[4])
{
    constexpr int SUB = HEAD_DIM / N_PALETTE;
    #pragma unroll
    for (int k = 0; k < 4; ++k) o[k] = 0.f;
    const QuadInfo q = tile_quad_info<HEAD_DIM>(ext, tg, s_tbl, side, gi, d0);
    if (((q.live >> j) & 1u) == 0u) return;
    const int p = (int)(q.tbw >> 6) & (N_PALETTE - 1);
    const int rank = (int)(q.tbw & 63u);
    const float inv = ext.inv[gi][side][p];
    uint32_t w[W];
    #pragma unroll
    for (int k = 0; k < W; ++k) w[k] = 0u;
    quad_raw_dispatch<W>(q.fmt, [&](auto tag) {
        using E = typename decltype(tag)::Elem;
        const auto quad = i8_tag_quad(tag, ext.gbase[gi][side][p], rank, SUB, 1.f);
        quad.raw4(q.within + j, w);
        i8_dtype_cvt4<E>(w, o);
    });
    if (inv != 1.f) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) o[k] *= inv;
    }
}

/// Two quads of one side: x[i][k][j] is dim d0[i] + k of group gi[i] at
/// its token j. The quad's four rank bytes are one word of the rank table
/// (d0 is a multiple of 4). A dead group's quad is zero; a masked token of
/// a live group reads as 0.
///
/// `tile_quads_probe<HEAD_DIM, W>` reads both quads' descriptors and says
/// whether they take the vector path: every live quad a vector-readable
/// quad of a format a `QuadRaw<W>` admits, loaded by `tile_quads_vec_load`
/// and converted by `tile_quads_vec_cvt` in a few dozen instructions —
/// `tile_quads_vec` is the two back to back. A warp whose lanes do not all
/// take it decodes its whole group by the block path below instead.
template <int HEAD_DIM, int W>
__device__ __forceinline__ bool tile_quads_probe(
    const TileExt& ext, const TileGroups& tg, const uint8_t* s_tbl, int side,
    const int (&gi)[2], const int (&d0)[2], QuadInfo (&q)[2])
{
    bool fast = true;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        q[i] = tile_quad_info<HEAD_DIM>(ext, tg, s_tbl, side, gi[i], d0[i]);
        fast = fast && (q[i].live == 0u || (q[i].vec && quad_raw_admits<W>(q[i].fmt)));
    }
    return fast;
}

template <int HEAD_DIM>
__device__ __forceinline__ void tile_quads_vec(
    const TileExt& ext, const QuadInfo (&q)[2], int side, const int (&gi)[2],
    float (&x)[2][4][GROUP_TOK])
{
    // Load and convert under ONE dispatch: the words then live only inside
    // the format's own arm, at that format's width, rather than at F32's
    // four per token across a dispatch boundary.
    QuadRaw<4> raw;
    const QuadPair pr = tile_quads_vec_resolve<HEAD_DIM>(ext, q, side, gi, raw);
    if (!pr.any) {
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                #pragma unroll
                for (int j = 0; j < GROUP_TOK; ++j) x[i][k][j] = 0.f;
        return;
    }
    const uint32_t live[2] = { q[0].live, q[1].live };
    quad_pair_dispatch<4>(pr.fmt[0], pr.fmt[1], [&](auto tag, int lo, int hi) {
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            if (i >= lo && i <= hi) quad_raw_load<HEAD_DIM>(tag, pr, i, raw.w[i]);
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            if (i >= lo && i <= hi) quad_raw_cvt(tag, raw.w[i], raw.inv[i], live[i], side, x[i]);
    });
}

// ============================================================================
// The block path: decode by (palette, rank), not by dim.
//
// A quantised or channel-mapped group is decoded SLOT-wise — a slot is one
// (palette, rank), the unit the arena actually stores (a quant palette's
// block, a dtype palette's column) — and every slot of a palette shares
// the palette's format, so a warp that takes a palette's 64 slots two per
// lane decodes them under ONE warp-uniform format dispatch: no lane ever
// waits for another lane's format body. The slot's dim comes from the
// group's INVERSE rank table (dim of (palette, rank)); the seal gives
// every palette exactly SUB dims, so every slot of a live group is a live
// dim and the inverse is a permutation.
//
//   K — warp = group, a lane the ranks lane and lane + 32 of all four
//       palettes; the decoded tokens are staged as half2 pairs by dim,
//       read back by quad (the RoPE pairing), then rotated and quantised
//       exactly as the vector path's quads are.
//   V — warp = palette, a lane the ranks lane and lane + 32 across all
//       eight groups, held as half2 pairs; the per-dim tile absmax is
//       gathered through a shared atomic max because a slot's dim is the
//       GROUP's map's (a sparse tile's groups can come from slices sealed
//       under different maps), and the quantised words go straight into
//       the transposed slab.
//
// The format bodies are the bulk of the kernel's code — one copy of K's
// and one of V's, each instantiated for every arena format — and are in
// line: a call's arguments and results go through local memory, and a
// sealed tile pays that on every group.
// ============================================================================

// A group's tokens are one aligned quad: `i8_pal_rank_quad4` (int8_elem.cuh)
// decodes a slot's four at once, with no unaligned decoder compiled in.
static_assert(GROUP_TOK == 4, "a group is one aligned token quad");

/// Where dim `d` of a K group is staged: 8 bytes per dim (the group's
/// four tokens as two half2 words), dims below HEAD_DIM / 2 at `lo`, the
/// rest at `hi`.
template <int HEAD_DIM>
__device__ __forceinline__ uint8_t* tile_stage_ptr(uint8_t* lo, uint8_t* hi, int d)
{
    return d < HEAD_DIM / 2 ? lo + d * 8 : hi + (d - HEAD_DIM / 2) * 8;
}

__device__ __forceinline__ uint32_t half2_bits(float a, float b)
{
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

/// K, block path: decode the `NP` palettes from `p0` of group g — all of
/// one format `tag` — and stage each dim's four tokens at
/// `tile_stage_ptr(d)`. Every slot's loads issue before any slot's store
/// (a store sits between the loads otherwise, and each slot then waits a
/// full memory latency for the one before it); `NP` bounds the raw words
/// held across that gap.
template <int HEAD_DIM, int NP, typename Tag>
__device__ __forceinline__ void tile_k_block_decode_palettes(
    Tag tag, const TileExt& ext, const uint8_t* s_inv_g, int g, int within, int lane,
    uint8_t* stg_lo, uint8_t* stg_hi, int stage_half, int p0)
{
    constexpr int SUB = HEAD_DIM / N_PALETTE;
    uint2 hw[NP][2];
    #pragma unroll
    for (int pi = 0; pi < NP; ++pi) {
        const char* base = ext.gbase[g][0][p0 + pi];
        const float inv = ext.inv[g][0][p0 + pi];
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            float o[GROUP_TOK];
            i8_pal_rank_quad4<true>(tag, base, lane + WARP_SIZE * s, SUB, inv, within, o);
            hw[pi][s] = make_uint2(half2_bits(o[0], o[1]), half2_bits(o[2], o[3]));
        }
    }
    #pragma unroll
    for (int pi = 0; pi < NP; ++pi)
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int d = s_inv_g[(p0 + pi) * SUB + lane + WARP_SIZE * s];
            if (stage_half < 0) {
                *reinterpret_cast<uint2*>(tile_stage_ptr<HEAD_DIM>(stg_lo, stg_hi, d)) = hw[pi][s];
            } else if ((d >= HEAD_DIM / 2) == (stage_half == 1)) {
                // One half's dims only, all into the private buffer.
                *reinterpret_cast<uint2*>(stg_lo + (d - stage_half * (HEAD_DIM / 2)) * 8) = hw[pi][s];
            }
        }
}
