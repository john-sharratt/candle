#pragma once
// ============================================================================
// ONE WARP'S FOUR KEY COLUMNS — decoded, rotated and quantised
// ============================================================================
//
// The K/V staging arithmetic of the INT8 prefill kernel, in one place for its
// two callers: the attention kernel's own tile staging (a warp's four columns
// straight into the tile slabs) and the pre-staging pass that runs it once
// per position for a bulk launch (`kv_prestage_kernel.cuh`). Both hand it the
// same four consecutive positions and receive the same values in the same
// order, so a column staged either way carries the same bits.
//
// A column at position `pos` is:
//   - past `kv_len`: zero K, zero V;
//   - at or past `prefix_len`: a fresh token, read from the packed K/V inputs;
//   - below `prefix_len`: a sealed token, decoded from the arena through its
//     slice's palette metadata (bound per warp, re-ranked only when a newly
//     bound slice's palette maps differ).
// When all four positions are sealed, consecutive in one slice, and each
// side's palettes share one format, the four are decoded together as a quad
// (`i8_pal_rank_load4`): the block header once, one format dispatch per
// window pair. Otherwise each column is decoded on its own.
//
// K is rotated by the sequence's RoPE rung at its own position and quantised
// per (column, 32-dim window) against the window's absmax over the warp; V is
// handed over as FP32 for the caller to store as FP16.
// ============================================================================

#include <cuda_fp16.h>
#include <stdint.h>
#include <type_traits>
#include "../arena_table.cuh"
#include "../paged-decode/slot_types.cuh"
#include "../convert/convert_all.cuh"
#include "../convert/int8_elem.cuh"
#include "../rope/rope_table.cuh"
#include "pal_rank.cuh"
#include "rope_hoist.cuh"

namespace prefill_int8 {

/// One warp's binding of the slice its current column lives in: per side
/// ([0] = K, [1] = V) and palette, the global decode base, palette scale and
/// its reciprocal, format and quant block bytes (0 ⇒ dtype element
/// addressing); the rank byte (`palette << 6 | rank`) of every natural dim;
/// and the palette-map words those ranks were computed under. Lives in shared
/// memory, one per warp; every value is warp-uniform, so reads are
/// broadcasts, and a lane's rank reads of dims {lane + 32w} touch 32
/// consecutive bytes.
///
/// The reciprocal is `__frcp_rn(scl)`, taken once at the bind: the quad
/// decode scales by it, and taken per window it was a slow-path branch
/// between every window's reads — a block boundary no read is hoisted across.
template <int HEAD_DIM>
struct WarpPalette {
    static constexpr int MAP_WORDS = HEAD_DIM / 16;
    const char* base[2][N_PALETTE];
    float scl[2][N_PALETTE];
    float rscl[2][N_PALETTE];
    int fmt[2][N_PALETTE];
    int bb[2][N_PALETTE];
    uint8_t rank[2][HEAD_DIM];
    uint32_t map[2][MAP_WORDS];
};

/// Bind slice `sl_idx` into `wp`. Warp-collective. `bound_slice` is the
/// warp's register record of what `wp` holds (-1 before the first bind).
template <int HEAD_DIM>
__device__ __forceinline__ void i8_bind_slice(
    WarpPalette<HEAD_DIM>& wp, int& bound_slice, const SlotHeader& slot_hdr,
    int sl_idx, int n_kv_head, int kv_head_idx, int lane)
{
    constexpr int N_WIN = HEAD_DIM / 32;
    constexpr int MAP_WORDS = WarpPalette<HEAD_DIM>::MAP_WORDS;
    static_assert(MAP_WORDS <= 32, "a palette map must fit one word per lane");
    const uint8_t* sl = get_slice<HEAD_DIM>(slot_hdr.slices_ptr, sl_idx, n_kv_head);
    const uint8_t* head = get_head<HEAD_DIM>(sl, kv_head_idx);
    if (lane < 2 * N_PALETTE) {
        const int side = lane / N_PALETTE;
        const int p = lane - side * N_PALETTE;
        const int fmt = side ? kvhead_v_fmt<HEAD_DIM>(head, p)
                             : kvhead_k_fmt<HEAD_DIM>(head, p);
        const int es = ArenaFormat::float_elem_size(fmt);
        wp.base[side][p] = (const char*)(uintptr_t)(
            side ? kvhead_v_ptr<HEAD_DIM>(head, p) : kvhead_k_ptr<HEAD_DIM>(head, p));
        wp.fmt[side][p] = fmt;
        wp.bb[side][p] = (es == 0) ? ArenaAccessor::get_quant_block_bytes(fmt) : 0;
        const float scl = side ? kvhead_v_scale<HEAD_DIM>(head, p)
                               : kvhead_k_scale<HEAD_DIM>(head, p);
        wp.scl[side][p] = scl;
        wp.rscl[side][p] = __frcp_rn(scl);
    }
    // Consecutive slices usually share routing: re-rank only when the maps
    // differ from the ones the rank bytes were computed under.
    const uint8_t* k_pal = kvhead_k_pal_map<HEAD_DIM>(head);
    const uint8_t* v_pal = kvhead_v_pal_map<HEAD_DIM>(head);
    const uint32_t kw = (lane < MAP_WORDS) ? ((const uint32_t*)k_pal)[lane] : 0u;
    const uint32_t vw = (lane < MAP_WORDS) ? ((const uint32_t*)v_pal)[lane] : 0u;
    bool same = (bound_slice >= 0);
    if (lane < MAP_WORDS)
        same = same && (kw == wp.map[0][lane]) && (vw == wp.map[1][lane]);
    same = __all_sync(0xffffffffu, same);
    if (!same) {
        if (lane < MAP_WORDS) {
            wp.map[0][lane] = kw;
            wp.map[1][lane] = vw;
        }
        i8_rank_from_map_words<HEAD_DIM>(kw, lane, wp.rank[0]);
        i8_rank_from_map_words<HEAD_DIM>(vw, lane, wp.rank[1]);
    }
    bound_slice = sl_idx;
    __syncwarp();
}

/// `int8_elem::i8_arena_elem` with the format resolved to `tag` up front:
/// the element at (`within`, `rank`) of a palette whose format is the tag's,
/// decoded as that function decodes it — a quant block's
/// `i8_dequant_elem` (value / scale through the block converter), or a dtype
/// element converted and divided by `scale`. With the format a compile-time
/// tag there is no per-element dispatch, so a run of these reads issues
/// together.
template <bool IS_K, typename Tag>
__device__ __forceinline__ float i8_arena_elem_of(
    Tag, const char* base, int rank, int within, float scale, int sub)
{
    if constexpr (int8_elem::I8IsDtypeTag<Tag>::value) {
        using E = typename Tag::Elem;
        const E* pe = reinterpret_cast<const E*>(base) + ((int64_t)within * sub + rank);
        float v;
        if constexpr (std::is_same_v<E, __half>) {
            v = __half2float(*pe);
        } else if constexpr (std::is_same_v<E, __nv_bfloat16>) {
            v = __bfloat162float(*pe);
        } else if constexpr (std::is_same_v<E, float>) {
            v = *pe;
        } else {
            static_assert(std::is_same_v<E, __nv_fp8_e4m3>, "the arena's dtype palettes");
            v = to_float<__nv_fp8_e4m3>(*pe);
        }
        return v / scale;
    } else {
        using B = typename Tag::Block;
        const char* blk = base + (int64_t)rank * sizeof(B);
        if constexpr (std::is_same_v<B, block_q0_v>) {
            return q0_v_load_element_f32<IS_K>((const block_q0_v*)blk, within, scale);
        } else {
            return BlockConverter<B, float>::load_element((const B*)blk, within, scale);
        }
    }
}

/// Decode the four columns at positions `pos0 .. pos0 + 3` of the sequence
/// whose packed rows start at `q_start`, for KV head `kv_head_idx`.
/// Warp-collective; every lane calls the sinks:
///
///   `k_sink(tt, w, code, scale)` — column `tt`'s window `w`: this lane's
///     int8 code (dim `lane + 32 w`) and the window's scale (FP32, the same
///     on every lane);
///   `v_sink(tt, w, v)` — column `tt`'s V at dim `lane + 32 w`.
///
/// A dead warp (no block in the tile) passes `pos0 = kv_len`: every column
/// is past the sequence and stages as zeros.
template <typename QT, int HEAD_DIM, typename KSink, typename VSink>
__device__ __forceinline__ void i8_stage_quad(
    WarpPalette<HEAD_DIM>& wp, int& bound_slice, const SlotHeader& slot_hdr,
    int n_kv_head, int kv_head_idx,
    const QT* __restrict__ k_packed, const QT* __restrict__ v_packed, int q_start,
    int prefix_len, int kv_len, int pos0, int lane,
    int rope_interleaved, const RopeView& rope,
    KSink&& k_sink, VSink&& v_sink)
{
    using int8_elem::I8IsDtypeTag;
    using int8_elem::i8_arena_elem;
    using int8_elem::i8_block_run4;
    using int8_elem::i8_dtype_rank_load4;
    using int8_elem::i8_pal_rank_quad4;
    using int8_elem::i8_quant;
    using int8_elem::i8_with_format;
    using int8_elem::qt_to_f32;
    constexpr int N_WIN = HEAD_DIM / 32;
    constexpr int SUB = HEAD_DIM / N_PALETTE;
    constexpr int COLS = 4;

    // A K window of one column: the window's absmax over the warp sets its
    // scale.
    const auto stage_k_window = [&](int tt, int w, float x) {
        float a = fabsf(x);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
        const float scale = a / 127.f;
        const float inv = (scale > 0.f) ? 1.f / scale : 0.f;
        k_sink(tt, w, i8_quant(x, inv), scale);
    };

    // The quad path. A warp's four columns are four consecutive positions;
    // when all four are sealed and sit consecutively in one slice, each dim's
    // four tokens are a run of the same block (quant) or four strided
    // elements (dtype), decoded together by `i8_pal_rank_load4`: the block
    // header once instead of four times, and when each side's palettes share
    // one format — nearly every slice — one format dispatch per window pair
    // instead of one per element. K is finished one RoPE pair of windows at a
    // time (RoPE rotates window w against w + N_WIN/2 and each window has its
    // own scale), so only that pair's four tokens are live at once: holding
    // every window's quad across the token loop tripled the kernel's register
    // spill.
    bool quad = false;
    if (pos0 + 3 < prefix_len && pos0 + 3 < kv_len) {
        int sl0, ib0, sl3, ib3;
        resolve_pos(slot_hdr, pos0, sl0, ib0);
        resolve_pos(slot_hdr, pos0 + 3, sl3, ib3);
        if (sl0 == sl3 && ib3 == ib0 + 3) {
            if (sl0 != bound_slice)
                i8_bind_slice<HEAD_DIM>(wp, bound_slice, slot_hdr, sl0, n_kv_head, kv_head_idx, lane);
            const int fk = wp.fmt[0][0];
            const int fv = wp.fmt[1][0];
            bool uniform = true;
            #pragma unroll
            for (int p = 1; p < N_PALETTE; ++p)
                uniform = uniform && wp.fmt[0][p] == fk && wp.fmt[1][p] == fv;
            if (uniform) {
                quad = true;
                // Windows wa and wb's four tokens of rank table `side` at quad
                // `ib0` — `i8_pal_rank_load4` for each, with the one branch
                // it takes (a quant block's aligned quad or unaligned run)
                // taken once for both, so both windows' reads go out together.
                const auto load_pair = [&](auto tag, auto is_k_const, int wa, int wb,
                                           float (&oa)[4], float (&ob)[4]) {
                    using Tag = decltype(tag);
                    constexpr bool IS_K = decltype(is_k_const)::value;
                    constexpr int side = IS_K ? 0 : 1;
                    const int ta = wp.rank[side][lane + 32 * wa];
                    const int tb = wp.rank[side][lane + 32 * wb];
                    const int pa = (ta >> 6) & (N_PALETTE - 1);
                    const int pb = (tb >> 6) & (N_PALETTE - 1);
                    const char* base_a = wp.base[side][pa];
                    const char* base_b = wp.base[side][pb];
                    const float ra = wp.rscl[side][pa];
                    const float rb = wp.rscl[side][pb];
                    if constexpr (I8IsDtypeTag<Tag>::value) {
                        i8_dtype_rank_load4<Tag>(base_a, ta & 63, SUB, ra, ib0, oa);
                        i8_dtype_rank_load4<Tag>(base_b, tb & 63, SUB, rb, ib0, ob);
                    } else if ((ib0 & 3) == 0) {
                        i8_pal_rank_quad4<IS_K>(tag, base_a, ta & 63, SUB, ra, ib0, oa);
                        i8_pal_rank_quad4<IS_K>(tag, base_b, tb & 63, SUB, rb, ib0, ob);
                    } else {
                        using B = typename Tag::Block;
                        i8_block_run4<B, IS_K>(
                            reinterpret_cast<const B*>(base_a + (int64_t)(ta & 63) * sizeof(B)),
                            ib0, ra, oa);
                        i8_block_run4<B, IS_K>(
                            reinterpret_cast<const B*>(base_b + (int64_t)(tb & 63) * sizeof(B)),
                            ib0, rb, ob);
                    }
                };
                using KSide = std::integral_constant<bool, true>;
                using VSide = std::integral_constant<bool, false>;
                #pragma unroll 1
                for (int w = 0; w < N_WIN / 2; ++w) {
                    // The pair's rotations for the four tokens go out first,
                    // so their table reads fly beside the arena reads below
                    // instead of one after another behind them.
                    I8RopePair rp[4];
                    #pragma unroll
                    for (int tt = 0; tt < 4; ++tt)
                        rp[tt] = i8_rope_pair_cs<N_WIN>(rope, pos0 + tt, w, lane, rope_interleaved);
                    float lo[4], hi[4];
                    i8_with_format(fk, [&](auto tag) {
                        load_pair(tag, KSide{}, w, w + N_WIN / 2, lo, hi);
                    });
                    #pragma unroll
                    for (int tt = 0; tt < 4; ++tt) {
                        i8_rope_pair_apply(lo[tt], hi[tt], rp[tt], lane, rope_interleaved);
                        stage_k_window(tt, w, lo[tt]);
                        stage_k_window(tt, w + N_WIN / 2, hi[tt]);
                    }
                }
                // V two windows per format dispatch, as K: both windows'
                // reads are in flight together.
                static_assert(N_WIN % 2 == 0, "V windows decode in pairs");
                #pragma unroll 1
                for (int w = 0; w < N_WIN; w += 2) {
                    float va[4], vb[4];
                    i8_with_format(fv, [&](auto tag) {
                        load_pair(tag, VSide{}, w, w + 1, va, vb);
                    });
                    #pragma unroll
                    for (int tt = 0; tt < 4; ++tt)
                        v_sink(tt, w, va[tt]);
                    #pragma unroll
                    for (int tt = 0; tt < 4; ++tt)
                        v_sink(tt, w + 1, vb[tt]);
                }
            }
        }
    }
    #pragma unroll 1
    for (int tt = 0; tt < COLS && !quad; ++tt) {
        const int pos = pos0 + tt;
        // K stays in registers for RoPE (pairs (w, w + N_WIN/2) are
        // in-thread); V goes straight to its sink, one dim at a time.
        float x[N_WIN];
        if (pos >= kv_len) {
            #pragma unroll
            for (int w = 0; w < N_WIN; ++w) {
                x[w] = 0.f;
                v_sink(tt, w, 0.f);
            }
        } else if (pos >= prefix_len) {
            const int tok = pos - prefix_len; // fresh token index
            const QT* kr = k_packed + ((int64_t)(q_start + tok) * n_kv_head + kv_head_idx) * HEAD_DIM;
            const QT* vr = v_packed + ((int64_t)(q_start + tok) * n_kv_head + kv_head_idx) * HEAD_DIM;
            #pragma unroll
            for (int w = 0; w < N_WIN; ++w) {
                x[w] = qt_to_f32<QT>(kr[lane + 32 * w]);
                v_sink(tt, w, qt_to_f32<QT>(vr[lane + 32 * w]));
            }
        } else {
            int sl_idx, in_blk;
            resolve_pos(slot_hdr, pos, sl_idx, in_blk);
            if (sl_idx != bound_slice)
                i8_bind_slice<HEAD_DIM>(wp, bound_slice, slot_hdr, sl_idx, n_kv_head, kv_head_idx, lane);
            const int fk = wp.fmt[0][0];
            const int fv = wp.fmt[1][0];
            bool uniform = true;
            #pragma unroll
            for (int p = 1; p < N_PALETTE; ++p)
                uniform = uniform && wp.fmt[0][p] == fk && wp.fmt[1][p] == fv;
            if (uniform) {
                // Each side's palettes share one format: one dispatch per
                // side, and every window's read inside it goes out together
                // — the per-element dispatch below serialises one global
                // round trip per window. Same decode, element for element.
                #pragma unroll
                for (int w = 0; w < N_WIN; ++w) x[w] = 0.f;
                i8_with_format(fk, [&](auto tag) {
                    #pragma unroll
                    for (int w = 0; w < N_WIN; ++w) {
                        const int t = wp.rank[0][lane + 32 * w];
                        const int p = (t >> 6) & (N_PALETTE - 1);
                        x[w] = i8_arena_elem_of<true>(tag, wp.base[0][p], t & 63, in_blk,
                                                      wp.scl[0][p], SUB);
                    }
                });
                float v[N_WIN];
                #pragma unroll
                for (int w = 0; w < N_WIN; ++w) v[w] = 0.f;
                i8_with_format(fv, [&](auto tag) {
                    #pragma unroll
                    for (int w = 0; w < N_WIN; ++w) {
                        const int t = wp.rank[1][lane + 32 * w];
                        const int p = (t >> 6) & (N_PALETTE - 1);
                        v[w] = i8_arena_elem_of<false>(tag, wp.base[1][p], t & 63, in_blk,
                                                       wp.scl[1][p], SUB);
                    }
                });
                #pragma unroll
                for (int w = 0; w < N_WIN; ++w) v_sink(tt, w, v[w]);
            } else {
                // Palettes of mixed formats. K is unrolled: its windows stay
                // in registers for RoPE. V goes to its sink a window at a
                // time, so its loop is rolled — each window's element read is
                // a dispatch over every format, and unrolling it put one
                // inlined copy per window in the kernel.
                #pragma unroll
                for (int w = 0; w < N_WIN; ++w) {
                    const int d = lane + 32 * w;
                    const int tk = wp.rank[0][d];
                    const int pk = (tk >> 6) & (N_PALETTE - 1);
                    x[w] = i8_arena_elem<true>(wp.fmt[0][pk], wp.bb[0][pk],
                                               wp.base[0][pk], tk & 63, in_blk,
                                               wp.scl[0][pk], SUB);
                }
                #pragma unroll 1
                for (int w = 0; w < N_WIN; ++w) {
                    const int d = lane + 32 * w;
                    const int tv = wp.rank[1][d];
                    const int pv = (tv >> 6) & (N_PALETTE - 1);
                    const float v = i8_arena_elem<false>(wp.fmt[1][pv], wp.bb[1][pv],
                                                         wp.base[1][pv], tv & 63, in_blk,
                                                         wp.scl[1][pv], SUB);
                    v_sink(tt, w, v);
                }
            }
        }
        if (pos < kv_len)
            i8_rope_row<N_WIN>(x, pos, lane, rope_interleaved, rope);
        #pragma unroll
        for (int w = 0; w < N_WIN; ++w) stage_k_window(tt, w, x[w]);
    }
}

} // namespace prefill_int8
