// K and V block decode, the staged K reads, the Q0_V read-through, and
// `tile_k_decode`.
//
// Holds `tile_k_block_decode`, which decodes every slot of a group's K, and
// `tile_v_block_decode` and `tile_v_block_quantise`, which decode a V slot and
// requantise it once the tile's absmax is complete.
// Holds `tile_v_block_path` and `tile_v_q0v_readthrough` for the V warps, the
// 16-byte `cp.async` copy, the staged quad path, and `tile_k_decode`, which
// composes the K decode.

/// K, block path, warp = group g: decode every slot of the group's K and
/// stage each dim's four tokens at `tile_stage_ptr(d)`. `s_inv_g` is the
/// group's K inverse table; `within` the group's first token (an aligned
/// quad of the slice). Masked tokens are staged as read — K's scale is
/// per token, and a masked column's score is forced to -inf.
///
/// A group whose four palettes share a format — nearly every group —
/// decodes them two at a time in one format body (four slots' raw words
/// fit the registers where eight would spill; the fence stops the
/// compiler hoisting the second pair's loads over the first pair's
/// stores). Otherwise each palette runs its own format's body alone: a
/// body per format over all four palettes would decode the others'
/// slots again for nothing, and a palette's loads cannot be hoisted
/// above the previous palette's format dispatch either way.
///
/// `stage_half` −1 stages every dim, the lower half's at `stg_lo` and the
/// upper's at `stg_hi` (`tile_stage_ptr`); 0 or 1 stages only that half's
/// dims, all of them into `stg_lo`, for a group whose own K rows cannot serve
/// as the upper half's stage (see the caller).
template <int HEAD_DIM>
__device__ __forceinline__ void tile_k_block_decode(
    const TileExt& ext, const uint8_t* s_inv_g, int g, int within, int lane,
    uint8_t* stg_lo, uint8_t* stg_hi, int stage_half)
{
    constexpr int SUB = HEAD_DIM / N_PALETTE;
    static_assert(SUB == 2 * WARP_SIZE, "a lane holds two ranks of each palette");
    static_assert(N_PALETTE % 2 == 0, "palettes decode in pairs");
    if ((within & 3) != 0) __trap();
    const uint32_t f4 = *reinterpret_cast<const uint32_t*>(&ext.fmt[g][0][0]);
    const uint32_t f = f4 & 0xffu;
    if (f4 == f * 0x01010101u) {
        i8_with_format((int)f, [&](auto tag) {
            #pragma unroll
            for (int pp = 0; pp < N_PALETTE; pp += 2) {
                if (pp > 0) asm volatile("" ::: "memory");
                tile_k_block_decode_palettes<HEAD_DIM, 2>(tag, ext, s_inv_g, g, within, lane,
                                                          stg_lo, stg_hi, stage_half, pp);
            }
        });
    } else {
        // A real loop, not unrolled: each palette's body is a dispatch over
        // every format, so unrolling pasted that whole dispatch once per
        // palette into the kernel — four copies on a path that runs only for
        // a group whose palettes differ in format. One palette's loads cannot
        // move above the previous palette's dispatch either way, so the
        // unroll bought no overlap.
        #pragma unroll 1
        for (int p = 0; p < N_PALETTE; ++p)
            i8_with_format((int)((f4 >> (8 * p)) & 0xffu), [&](auto tag) {
                tile_k_block_decode_palettes<HEAD_DIM, 1>(tag, ext, s_inv_g, g, within, lane,
                                                          stg_lo, stg_hi, stage_half, p);
            });
    }
}

/// Read a staged quad back: xi[k][j] is dim d0 + k at token j.
/// A quad's staged dims, raw: 4 dims × 4 tokens of half, 8 words. Held
/// this way across a second staging round it is half the registers of the
/// converted floats.
template <int HEAD_DIM>
__device__ __forceinline__ void tile_k_stage_raw(
    uint8_t* stg_lo, uint8_t* stg_hi, int d0, uint4 (&w)[2])
{
    const uint8_t* p = tile_stage_ptr<HEAD_DIM>(stg_lo, stg_hi, d0);
    w[0] = *reinterpret_cast<const uint4*>(p);
    w[1] = *reinterpret_cast<const uint4*>(p + 16);
}

__device__ __forceinline__ void tile_k_stage_cvt(const uint4 (&w)[2], float (&xi)[4][GROUP_TOK])
{
    const uint32_t ww[4][2] = { { w[0].x, w[0].y }, { w[0].z, w[0].w },
                                { w[1].x, w[1].y }, { w[1].z, w[1].w } };
    #pragma unroll
    for (int k = 0; k < 4; ++k)
        #pragma unroll
        for (int h = 0; h < GROUP_TOK / 2; ++h) {
            const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&ww[k][h]));
            xi[k][2 * h] = f2.x;
            xi[k][2 * h + 1] = f2.y;
        }
}

template <int HEAD_DIM>
__device__ __forceinline__ void tile_k_stage_read(
    uint8_t* stg_lo, uint8_t* stg_hi, int d0, float (&xi)[4][GROUP_TOK])
{
    uint4 w[2];
    tile_k_stage_raw<HEAD_DIM>(stg_lo, stg_hi, d0, w);
    tile_k_stage_cvt(w, xi);
}

__device__ __forceinline__ uint4 sel4(bool keep, const uint4 a, const uint4 b)
{
    return make_uint4(keep ? a.x : b.x, keep ? a.y : b.y, keep ? a.z : b.z, keep ? a.w : b.w);
}

/// V, block path, one slot (palette p, rank `rank`) across the tile's
/// groups: w[g][h] holds tokens 4g + 2h, +1 as a half2 pair — zero for a
/// dead group or a masked token — and each live group's absmax is folded
/// into the tile absmax of the dim its map gives the slot, in `s_vabs`.
/// `NS` is the slot count decoded: TILE_GROUPS on an eight-quad tile (the
/// ninth slot is dead and costs nothing), TILE_SLOTS when the ninth is live.
/// Chosen under a block-uniform branch by the caller, so neither
/// instantiation carries a runtime predicate over its arrays.
template <int HEAD_DIM, int NS>
__device__ __forceinline__ void tile_v_block_decode(
    const TileExt& ext, const TileGroups& tg, const uint8_t* s_inv, int* s_vabs,
    uint32_t live_g, int p, int rank, uint32_t (&w)[TILE_SLOTS][GROUP_TOK / 2])
{
    static_assert(NS == TILE_GROUPS || NS == TILE_SLOTS, "eight quads, or eight and the ninth");
    constexpr int SUB = HEAD_DIM / N_PALETTE;
    #pragma unroll
    for (int g = 0; g < TILE_SLOTS; ++g)
        #pragma unroll
        for (int h = 0; h < GROUP_TOK / 2; ++h) w[g][h] = 0u;
    uint32_t rem = live_g;
    while (rem != 0u) {
        const uint32_t f = ext.fmt[__ffs(rem) - 1][1][p];
        uint32_t sel = 0u;
        #pragma unroll
        for (int g = 0; g < NS; ++g)
            if (((rem >> g) & 1u) != 0u && ext.fmt[g][1][p] == f) sel |= 1u << g;
        // Every group decodes in straight-line code — one that is dead or
        // of another format reads the first selected group's slot again
        // and drops the result — so the eight slot loads issue back to
        // back. A skip around a group's loads compiles to a branch, and
        // the next group's loads then wait behind this group's latency.
        const int g0 = __ffs(sel) - 1;
        i8_with_format((int)f, [&](auto tag) {
            // The atomics go out after every group is decoded: a load is
            // not moved across an atomic, so one inside the loop would
            // hold the next group's loads behind this group's latency. A
            // dtype palette's slot is four loads where a block's is two,
            // so those bodies keep half the groups in flight at a time
            // (the fence between the halves is what stops the compiler
            // hoisting the second half's loads and spilling for them).
            constexpr int IN_FLIGHT = I8IsDtypeTag<decltype(tag)>::value ? TILE_GROUPS / 2 : TILE_GROUPS;
            float a[TILE_SLOTS];
            // The ninth slot (NS == TILE_SLOTS) rides in the LAST round, its
            // load in flight with that round's: neither an extra round (a
            // fence and an exposed latency per call) nor a runtime skip
            // around it (which would put the whole word array under a
            // predicate, and so in local memory — measured at the gate as
            // 1.5 KB of spills). `gn` folds at unroll time.
            #pragma unroll
            for (int gb = 0; gb < TILE_GROUPS; gb += IN_FLIGHT) {
                if (gb > 0) asm volatile("" ::: "memory");
                const int gn = (NS > TILE_GROUPS && gb + IN_FLIGHT >= TILE_GROUPS) ? IN_FLIGHT + 1
                                                                                   : IN_FLIGHT;
                #pragma unroll
                for (int gi = 0; gi < IN_FLIGHT + 1; ++gi) {
                    if (gi >= gn) break;
                    const int g = gb + gi;
                    const bool on = ((sel >> g) & 1u) != 0u;
                    const int gs = on ? g : g0;
                    const uint32_t desc = tg.desc[gs];
                    const uint32_t live = grp_mask(desc);
                    float o[GROUP_TOK];
                    i8_pal_rank_quad4<false>(tag, ext.gbase[gs][1][p], rank, SUB, ext.inv[gs][1][p],
                                   grp_within(desc), o);
                    #pragma unroll
                    for (int j = 0; j < GROUP_TOK; ++j)
                        if (((live >> j) & 1u) == 0u) o[j] = 0.f;
                    a[g] = fmaxf(fmaxf(fabsf(o[0]), fabsf(o[1])), fmaxf(fabsf(o[2]), fabsf(o[3])));
                    #pragma unroll
                    for (int h = 0; h < GROUP_TOK / 2; ++h)
                        w[g][h] = on ? half2_bits(o[2 * h], o[2 * h + 1]) : w[g][h];
                }
            }
            #pragma unroll
            for (int g = 0; g < NS; ++g) {
                const bool on = ((sel >> g) & 1u) != 0u;
                const int gs = on ? g : g0;
                const int d = s_inv[(gs * 2 + 1) * HEAD_DIM + p * SUB + rank];
                // Non-negative floats order as their bit patterns.
                if (on) atomicMax(&s_vabs[d], __float_as_int(a[g]));
            }
        });
        rem &= ~sel;
    }
}

/// V, block path, one slot after the tile absmax is complete: quantise
/// each group's four tokens by their dim's tile scale into that dim's
/// word of the transposed slab. A dead group's word is zero (its map is
/// the identity, so the slot's dim is p · SUB + rank there).
template <int HEAD_DIM, int NS>
__device__ __forceinline__ void tile_v_block_quantise(
    const uint32_t (&w)[TILE_SLOTS][GROUP_TOK / 2], const TileGroups& tg, const uint8_t* s_inv,
    const int* s_vabs, int8_t* s_v8t, int p, int rank)
{
    static_assert(NS == TILE_GROUPS || NS == TILE_SLOTS, "eight quads, or eight and the ninth");
    constexpr int SUB = HEAD_DIM / N_PALETTE;
    #pragma unroll
    for (int g = 0; g < NS; ++g) {
        // A dead group has no columns; a live one's tokens go to its own,
        // whole and 4-aligned as one word, otherwise a byte per live token
        // (its word would cover a neighbouring group's columns).
        const uint32_t lv = grp_mask(tg.desc[g]);
        if (lv == 0u) continue;
        const int d = s_inv[(g * 2 + 1) * HEAD_DIM + p * SUB + rank];
        const float a = __int_as_float(s_vabs[d]);
        // The same arithmetic as the vector path's `v_quantise`, to the
        // rounding: a token quantises to the same byte whichever path carries
        // it, which a multi-pass tile — vector-readable chunks taken through
        // this path — depends on to match its single-pass equivalent.
        const float scale = a / 127.f;
        const float inv = (scale > 0.f) ? 1.f / scale : 0.f;
        float v[GROUP_TOK];
        #pragma unroll
        for (int h = 0; h < GROUP_TOK / 2; ++h) {
            const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&w[g][h]));
            v[2 * h] = f2.x;
            v[2 * h + 1] = f2.y;
        }
        const int c0 = (int)tg.col0[g];
        if (lv == 15u && (c0 & 3) == 0) {
            *reinterpret_cast<uint32_t*>(s_v8t + sw_off<TILE_TOK>(d, c0)) = i8_pack4(v, inv);
        } else {
            #pragma unroll
            for (int j = 0; j < GROUP_TOK; ++j)
                if ((lv >> j) & 1u) s_v8t[sw_off<TILE_TOK>(d, c0 + j)] = i8_quant(v[j], inv);
        }
    }
}

/// V, block path, one V warp's whole tile: palette p = the warp, a lane the
/// ranks `lane` and `lane + 32`. The two slots decode through one copy of
/// the format bodies (the loop is not unrolled), their words kept apart by
/// a select; the tile absmax is complete once every V warp is at the V
/// barrier, which orders the atomics before the reads. `NS` as above,
/// chosen by the caller under a block-uniform branch.
template <int HEAD_DIM, int NS>
__device__ __forceinline__ void tile_v_block_path(
    const TileExt& ext, const TileGroups& tg, const uint8_t* s_inv, int* s_vabs,
    uint32_t live_g, int p, int lane, int8_t* s_v8t)
{
    uint32_t w0[TILE_SLOTS][GROUP_TOK / 2], w1[TILE_SLOTS][GROUP_TOK / 2];
    #pragma unroll
    for (int gg = 0; gg < TILE_SLOTS; ++gg)
        #pragma unroll
        for (int h = 0; h < GROUP_TOK / 2; ++h) w0[gg][h] = 0u;
    #pragma unroll 1
    for (int s = 0; s < 2; ++s) {
        uint32_t wt[TILE_SLOTS][GROUP_TOK / 2];
        tile_v_block_decode<HEAD_DIM, NS>(ext, tg, s_inv, s_vabs, live_g, p, lane + WARP_SIZE * s, wt);
        #pragma unroll
        for (int gg = 0; gg < TILE_SLOTS; ++gg)
            #pragma unroll
            for (int h = 0; h < GROUP_TOK / 2; ++h) {
                w0[gg][h] = (s == 0) ? wt[gg][h] : w0[gg][h];
                w1[gg][h] = wt[gg][h];
            }
    }
    asm volatile("bar.sync %0, %1;" :: "n"(TILE_VABS_BARRIER), "n"(TILE_V_WARPS * WARP_SIZE) : "memory");
    tile_v_block_quantise<HEAD_DIM, NS>(w0, tg, s_inv, s_vabs, s_v8t, p, lane);
    tile_v_block_quantise<HEAD_DIM, NS>(w1, tg, s_inv, s_vabs, s_v8t, p, lane + WARP_SIZE);
}

/// V, Q0_V read-through, one V warp's whole tile: palette p = the warp, a
/// lane the ranks `lane` and `lane + 32`. Taken when every live group of a
/// single-pass tile sits in ONE slice and all four of its V palettes are
/// Q0_V, so each dim's tokens across the tile are one Q0_V block, whose
/// value at token t is
///
///   v[t] = S · base[t + 2p] + C       (S = ±scale·r, C = centroid·r)
///
/// with one (S, C) per dim for the whole tile. The block's base bytes ARE an
/// int8 V row with the per-dim scale S, so they go into the transposed slab
/// as they are — no decode to FP, no tile absmax, no requantisation — and the
/// centroid comes back in the PV epilogue as C · Σp over the tile's row.
/// Per dim this writes S into `s_v_scale` and C's bits into the tile's absmax
/// buffer, which the block path would otherwise use and this one does not.
/// All nine slots are walked: the ninth's descriptor is dead on an
/// eight-quad tile, so it is skipped by the same check as any dead group.
template <int HEAD_DIM>
__device__ __forceinline__ void tile_v_q0v_readthrough(
    const TileExt& ext, const TileGroups& tg, const uint8_t* s_inv, int* s_vabs,
    __half* s_v_scale, int g0, int p, int lane, int8_t* s_v8t)
{
    constexpr int SUB = HEAD_DIM / N_PALETTE;
    const float r = ext.inv[g0][1][p];
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        const int rank = lane + WARP_SIZE * s;
        const int d = s_inv[(g0 * 2 + 1) * HEAD_DIM + p * SUB + rank];
        const Q0VHeader h = q0_v_header<false>(
            reinterpret_cast<const block_q0_v*>(ext.gbase[g0][1][p]) + rank, r);
        s_v_scale[d] = __float2half(h.s);
        s_vabs[d] = __float_as_int(h.c);
        #pragma unroll
        for (int g = 0; g < TILE_SLOTS; ++g) {
            const uint32_t lv = grp_mask(tg.desc[g]);
            if (lv == 0u) continue;
            const int within = grp_within(tg.desc[g]);
            const int c0 = (int)tg.col0[g];
            if (lv == 15u && (c0 & 3) == 0) {
                // A whole group: four consecutive base bytes as one word
                // (within ≤ 28, so the two words lie inside the 64-byte row).
                const uintptr_t a = reinterpret_cast<uintptr_t>(h.row + within);
                const uint32_t* w4 = reinterpret_cast<const uint32_t*>(a & ~(uintptr_t)3);
                const uint32_t w = __funnelshift_r(__ldg(w4), __ldg(w4 + 1), (uint32_t)(a & 3) * 8u);
                *reinterpret_cast<uint32_t*>(s_v8t + sw_off<TILE_TOK>(d, c0)) = w;
            } else {
                #pragma unroll
                for (int j = 0; j < GROUP_TOK; ++j)
                    if ((lv >> j) & 1u) s_v8t[sw_off<TILE_TOK>(d, c0 + j)] = (int8_t)__ldg(h.row + within + j);
            }
        }
    }
}

/// One 16-byte global → shared copy through the async proxy, waited for by
/// `cp_async_wait` (decode_helpers.cuh). `.ca`: small, reread rows.
__device__ __forceinline__ void tile_cp_async_16(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src)
                 : "memory");
}

/// `tile_quads_vec` with quad A (dims d0..d0+3, in the head's lower half)
/// read from the group's staged rows — `k_raw` is [token][lower-half dim]
/// at the palette element width — and quad B loaded from the arena as
/// usual, issued first so it is in flight while A is read back.
template <int HEAD_DIM>
__device__ __forceinline__ void tile_quads_vec_staged(
    const TileExt& ext, const QuadInfo (&q)[2], const int (&gi)[2],
    float (&x)[2][4][GROUP_TOK], const uint8_t* k_raw, int d0)
{
    QuadRaw<4> raw;
    const QuadPair pr = tile_quads_vec_resolve<HEAD_DIM>(ext, q, 0, gi, raw);
    const uint32_t live[2] = { q[0].live, q[1].live };
    quad_pair_dispatch<4>(pr.fmt[0], pr.fmt[1], [&](auto tag, int lo, int hi) {
        if (1 >= lo && 1 <= hi) quad_raw_load<HEAD_DIM>(tag, pr, 1, raw.w[1]);
        if (0 >= lo && 0 <= hi) {
            if constexpr (I8IsDtypeTag<decltype(tag)>::value) {
                using E = typename decltype(tag)::Elem;
                constexpr int ROW = (HEAD_DIM / 2) * (int)sizeof(E);
                #pragma unroll
                for (int j = 0; j < GROUP_TOK; ++j) {
                    const uint8_t* p = k_raw + j * ROW + d0 * (int)sizeof(E);
                    if constexpr (sizeof(E) == 2) {
                        const uint2 v = *reinterpret_cast<const uint2*>(p);
                        raw.w[0][j][0] = v.x;
                        raw.w[0][j][1] = v.y;
                    } else if constexpr (sizeof(E) == 1) {
                        raw.w[0][j][0] = *reinterpret_cast<const uint32_t*>(p);
                    } else {
                        const uint4 v = *reinterpret_cast<const uint4*>(p);
                        raw.w[0][j][0] = v.x; raw.w[0][j][1] = v.y;
                        raw.w[0][j][2] = v.z; raw.w[0][j][3] = v.w;
                    }
                }
            } else {
                __trap();   // a staged tile's quads are dtype quads by the vote that staged it
            }
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            if (i >= lo && i <= hi) quad_raw_cvt(tag, raw.w[i], raw.inv[i], live[i], 0, x[i]);
    });
}

// ============================================================================
// K decode
// ============================================================================
// One group's K rows into the tile's K slab: warp = group `kg`, lane = quad.
// A lane's two quads are its four RoPE pairs — half-split: dims 4q..4q+3 with
// 128+4q..; interleaved: dims 8q..8q+7 as (8q+2i, 8q+2i+1) — and both
// pairings read the same four frequencies 4q..4q+3. The lane looks up the
// rotation at the group's first live token in the slot's rung table
// (`rope_table.cuh`) and the per-position step, that table's `LO` row 1, and
// the later tokens' angles follow by rotating the pair in-register — a
// token's angle is never further than three steps from a lookup, so the drift
// is a few ulp, far under the int8 quantisation the row is about to take. A
// pass-through frequency's rotation and step are exactly (1, 0). On the vector
// path the rope values load ahead of the quad loads so every load of the
// group is in flight together; on the block path they load after the decode,
// so the 16 rope values are not live across its format bodies.
// Scale windows: half-split, a lane's quad A lies in window q>>3 and quad B
// in 4 + (q>>3), so the window absmax is a reduce over the 8 lanes sharing
// q>>3; interleaved, the eight dims lie in window q>>2, a reduce over 4
// lanes. Each token's quantised quads land as one word each. A dead group
// is skipped outright (warp-uniform); a masked token inside a live group
// reads as 0 — its column's score is forced to -inf by the softmax, so its
// int8 row is never read.
//
// A free function over explicit operands, not a lambda over the kernel's
// scope: a closure of this size is one NVCC outlines into a real call, which
// puts every captured variable in addressable local memory and spills the
// caller's live registers around the call. Force-inlined at its one site it
// is register code, as the kernel body it sits in.
//
// `s_kstg` is the K staging slab: `span` is the calling warp's private 4-row
// span of it (the warp's index — a warp decoding the ninth quad as a second
// round reuses its own span, its first group's rows already stored); `s_k8`
// and `s_k_scale` are the tile's K slab and its per-window scales.
template <int HEAD_DIM, bool ROPE_INTERLEAVED>
__device__ __forceinline__ void tile_k_decode(
    const TileExt& ext, const TileGroups& tg, const uint8_t* s_tbl, const uint8_t* s_inv,
    uint8_t* s_kstg, int span, int8_t* s_k8, __half (*s_k_scale)[HEAD_DIM / 32],
    const RopeView& rope, int kg, int lane, const uint8_t* k_raw)
{
    const uint32_t klive = grp_mask(tg.desc[kg]);
    if (klive == 0u) return;
    // The first live token: a group at the head of a slice whose offset
    // falls inside it has leading masked tokens, and those sit before the
    // table's first row.
    const int j0 = __ffs(klive) - 1;
    const int rope0 = tg.rope0[kg] + j0;
    const int dA0 = ROPE_INTERLEAVED ? 8 * lane : 4 * lane;
    const int dB0 = ROPE_INTERLEAVED ? dA0 + 4 : dA0 + HEAD_DIM / 2;
    float rc[4], rs[4], dc[4], ds[4];
    // A lane's four rotary frequencies are 4·lane + i in both pairings: the
    // rotation at the group's first live token from the slot's rung table,
    // and the unit step (its `LO` row 1) the token loop below walks by.
    auto load_rope = [&]() {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            rope_cs_at(rope, rope0, 4 * lane + i, rc[i], rs[i]);
            rope_cs_step(rope, 4 * lane + i, dc[i], ds[i]);
        }
    };
    const int kgi[2] = { kg, kg };
    const int kd0[2] = { dA0, dB0 };
    float x[2][4][GROUP_TOK];
    QuadInfo kq[2];
    const bool kfast = tile_quads_probe<HEAD_DIM, 4>(ext, tg, s_tbl, 0, kgi, kd0, kq);
    if (__builtin_expect(__all_sync(0xffffffffu, kfast), 1)) {
        load_rope();
        if (k_raw != nullptr) {
            // The warp's own copies: every lane waits for its groups, then
            // the warp converges so each lane sees the others' rows.
            cp_async_wait<0, true>();
            __syncwarp();
            tile_quads_vec_staged<HEAD_DIM>(ext, kq, kgi, x, k_raw, dA0);
        } else {
            tile_quads_vec<HEAD_DIM>(ext, kq, 0, kgi, x);
        }
    } else {
        // A staged tile is a vector tile by the vote that staged it; the
        // block path's staging slab is holding the copies.
        if (k_raw != nullptr) __trap();
        // The block path stages the group's dims — 8 B per dim, two 4-row
        // spans — before reading them back by quad. The staging slab's rows
        // of the group's index are this warp's alone. A group's rows are its
        // COLUMNS of the K slab, which are its own to scribble on only when
        // the group is whole and on the 4-grid. A quad at a hole's edge
        // shares its four rows with a neighbour another warp may be writing,
        // so it stages one half of the head at a time through the private
        // span instead — a second decode of the group, paid by the two quads
        // a hole costs a window.
        uint8_t* stg_lo = s_kstg + span * 4 * HEAD_DIM;
        const int kc0 = (int)tg.col0[kg];
        const uint8_t* s_inv_g = s_inv + (kg * 2) * HEAD_DIM;
        const int within = grp_within(tg.desc[kg]);
        if (klive == 15u && (kc0 & 3) == 0) {
            uint8_t* stg_hi = reinterpret_cast<uint8_t*>(s_k8) + kc0 * HEAD_DIM;
            tile_k_block_decode<HEAD_DIM>(ext, s_inv_g, kg, within, lane, stg_lo, stg_hi, -1);
            __syncwarp();
            load_rope();
            tile_k_stage_read<HEAD_DIM>(stg_lo, stg_hi, dA0, x[0]);
            tile_k_stage_read<HEAD_DIM>(stg_lo, stg_hi, dB0, x[1]);
            __syncwarp();
        } else {
            // Half h's dims land at stg_lo + (d − h·HEAD_DIM/2)·8:
            // `tile_stage_ptr(lo, hi, d)` reads the upper half at
            // `hi + (d − HEAD_DIM/2)·8`, so `hi = lo` resolves both halves
            // to the private span.
            //
            // What a round stages is read back RAW (8 words a quad) and
            // converted after both rounds, and the rope rows load after
            // them too: the second round's decode then carries 8 live words
            // of the first, not 16 floats and 16 rope values, which is the
            // difference between this path fitting the register budget and
            // it parking the PV accumulators in local memory.
            //
            // Half-split, quad A is always in the lower half and quad B in
            // the upper, so each round reads its own quad — the round is
            // the unrolled loop's index, no runtime predicate. Interleaved,
            // a lane's two quads share a half that depends on the lane, so
            // each round reads both and keeps the round that staged their
            // half by select: written unconditionally on every path, because
            // an array written under a runtime predicate is one NVCC moves
            // to local memory for the whole kernel, vector path included.
            uint4 wa[2] = { make_uint4(0u, 0u, 0u, 0u), make_uint4(0u, 0u, 0u, 0u) };
            uint4 wb[2] = { make_uint4(0u, 0u, 0u, 0u), make_uint4(0u, 0u, 0u, 0u) };
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                tile_k_block_decode<HEAD_DIM>(ext, s_inv_g, kg, within, lane, stg_lo, stg_lo, h);
                __syncwarp();
                if constexpr (ROPE_INTERLEAVED) {
                    const bool keep = (dA0 >= HEAD_DIM / 2) == (h == 1);
                    uint4 ta[2], tb[2];
                    tile_k_stage_raw<HEAD_DIM>(stg_lo, stg_lo, dA0, ta);
                    tile_k_stage_raw<HEAD_DIM>(stg_lo, stg_lo, dB0, tb);
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        wa[i] = sel4(keep, ta[i], wa[i]);
                        wb[i] = sel4(keep, tb[i], wb[i]);
                    }
                } else {
                    if (h == 0) tile_k_stage_raw<HEAD_DIM>(stg_lo, stg_lo, dA0, wa);
                    else        tile_k_stage_raw<HEAD_DIM>(stg_lo, stg_lo, dB0, wb);
                }
                __syncwarp();
            }
            tile_k_stage_cvt(wa, x[0]);
            tile_k_stage_cvt(wb, x[1]);
            load_rope();
        }
    }
    #pragma unroll
    for (int j = 0; j < GROUP_TOK; ++j) {
        // Step the angle past every token from the first live one on,
        // masked or not, so (rc, rs) is token j's.
        if (j > 0 && j > j0) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float c = rc[i] * dc[i] - rs[i] * ds[i];
                rs[i] = rs[i] * dc[i] + rc[i] * ds[i];
                rc[i] = c;
            }
        }
        if (!((klive >> j) & 1u)) continue;   // warp-uniform
        // The token's column is its logical position's, not the group's
        // slot: a masked j is one outside the window.
        const int tok = (int)tg.col0[kg] + j;
        // r[0..3] is quad A rotated, r[4..7] quad B.
        float r[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float c = rc[i], s = rs[i];
            float lo, hi;
            if constexpr (ROPE_INTERLEAVED) {
                lo = (i < 2) ? x[0][2 * i][j] : x[1][2 * i - 4][j];
                hi = (i < 2) ? x[0][2 * i + 1][j] : x[1][2 * i - 3][j];
            } else {
                lo = x[0][i][j];
                hi = x[1][i][j];
            }
            const float ra = lo * c - hi * s;
            const float rb = lo * s + hi * c;
            if constexpr (ROPE_INTERLEAVED) { r[2 * i] = ra; r[2 * i + 1] = rb; }
            else { r[i] = ra; r[4 + i] = rb; }
        }
        float aA = fmaxf(fmaxf(fabsf(r[0]), fabsf(r[1])), fmaxf(fabsf(r[2]), fabsf(r[3])));
        float aB = fmaxf(fmaxf(fabsf(r[4]), fabsf(r[5])), fmaxf(fabsf(r[6]), fabsf(r[7])));
        if constexpr (ROPE_INTERLEAVED) {
            aA = fmaxf(aA, aB);
            aA = fmaxf(aA, __shfl_xor_sync(0xffffffffu, aA, 1));
            aA = fmaxf(aA, __shfl_xor_sync(0xffffffffu, aA, 2));
            aB = aA;
        } else {
            #pragma unroll
            for (int off = 1; off < 8; off <<= 1) {
                aA = fmaxf(aA, __shfl_xor_sync(0xffffffffu, aA, off));
                aB = fmaxf(aB, __shfl_xor_sync(0xffffffffu, aB, off));
            }
        }
        const float scaleA = aA / 127.f;
        const float scaleB = aB / 127.f;
        const float invA = (scaleA > 0.f) ? 1.f / scaleA : 0.f;
        const float invB = (scaleB > 0.f) ? 1.f / scaleB : 0.f;
        const float qa[4] = { r[0], r[1], r[2], r[3] };
        const float qb[4] = { r[4], r[5], r[6], r[7] };
        const uint32_t wa = i8_pack4(qa, invA);
        const uint32_t wb = i8_pack4(qb, invB);
        if constexpr (ROPE_INTERLEAVED) {
            *(uint2*)(s_k8 + sw_off<HEAD_DIM>(tok, dA0)) = make_uint2(wa, wb);
            if ((lane & 3) == 0) s_k_scale[tok][lane >> 2] = __float2half(scaleA);
        } else {
            *(uint32_t*)(s_k8 + sw_off<HEAD_DIM>(tok, dA0)) = wa;
            *(uint32_t*)(s_k8 + sw_off<HEAD_DIM>(tok, dB0)) = wb;
            if ((lane & 7) == 0) {
                s_k_scale[tok][lane >> 3] = __float2half(scaleA);
                s_k_scale[tok][(HEAD_DIM / 2 / 32) + (lane >> 3)] = __float2half(scaleB);
            }
        }
    }
}
