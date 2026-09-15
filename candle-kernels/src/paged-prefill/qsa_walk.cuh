#pragma once
// ============================================================================
// QSA BLOCK WALK — the prefill kernel's selected-block cursor
// ============================================================================
//
// A prefill block serves a run of consecutive packed query rows (one per
// query token; every head of the group shares the token's selection). The
// block-sparse read visits only the selection blocks (`ratio` consecutive
// positions each, `qsa_select.cuh`) at least one of those rows selects: the
// walk below merges the rows' ascending entry lists into the ascending
// sequence of selected blocks, and the tile loop packs consecutive selected
// blocks into one 32-column tile. Each step also reports, per row, which
// cells of the block that row selects, so the tile's per-column mask is a
// bit test instead of a per-key search.
//
// Every warp runs its own copy: the state is a handful of registers per lane
// (lane l owns rows l and l + 32 of the block's run — at most 64 rows, the
// widest M the kernel packs), the entry loads are L1 hits after the first
// touch, and a warp-private cursor needs no smem handoff or barrier to stay
// block-uniform — each warp derives the identical block sequence from the
// identical lists.
//
// A dense row (`QSA_DENSE_ROW`, or every row when the launch carries no
// selection) selects every cell of every block, so it contributes each
// successive block in turn; the walk over such a run steps through the whole
// prefix at `ratio` positions per step.
// ============================================================================

#include <stdint.h>
#include "../qsa_select.cuh"

namespace prefill_int8 {

constexpr int QSA_WALK_ROWS_PER_LANE = 2;
constexpr int QSA_WALK_MAX_ROWS = 32 * QSA_WALK_ROWS_PER_LANE;

// No selected block at or past the bound — the walk is over.
constexpr int QSA_WALK_END = 0x7fffffff;

struct QsaWalk {
    // The table base is the launch's — a kernel parameter, so the cursor
    // keeps 32-bit row offsets into it rather than a 64-bit pointer each.
    const uint32_t* base;
    uint32_t e[QSA_WALK_ROWS_PER_LANE];        // this lane's rows' entry-list offsets
    uint32_t n[QSA_WALK_ROWS_PER_LANE];        // entries per row; 0 = no row here
    uint32_t c[QSA_WALK_ROWS_PER_LANE];        // cursor: first entry not yet passed
    uint32_t dense;                            // bit h: this lane's row h attends everything
    int ratio;                                 // positions per block
    // Page layout for this run, bound once — see `QsaSel::pages`. A prefill
    // block serves a run of consecutive query rows of ONE sequence, so every
    // row here shares a window and the block→position map is a property of the
    // run rather than of the row. Null when the sequence forwarded its whole
    // prefix, and then a block starts at `block * ratio`.
    const uint2* pages;
    uint2 win;

    // The first key position of `block`, through this run's page layout.
    __device__ __forceinline__ int start_of(uint32_t block) const {
        if (pages == nullptr) return (int)block * ratio;
        uint32_t lo = win.x, hi = win.x + win.y;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (pages[mid].y <= block) lo = mid; else hi = mid;
        }
        const uint2 p = pages[lo];
        return (int)(p.x + (block - p.y) * (uint32_t)ratio);
    }

    // The cells `block` (starting at `first`) spans, at most `ratio` — the
    // walk's copy of `qsa_block_width`. A page's last block is short, and an
    // entry's full count would reach into the next block: attending its first
    // position twice when that block is selected too, and once when it is not.
    __device__ __forceinline__ int width_of(uint32_t block, int first) const {
        if (pages == nullptr) return ratio;
        return min(ratio, start_of(block + 1u) - first);
    }

    // Bind rows [row_base, row_base + n_rows) of `sel`. Called by every lane.
    __device__ __forceinline__ void init(const QsaSel& sel, int row_base, int n_rows, int lane) {
        base = sel.entries;
        ratio = sel.ratio;
        pages = sel.pages;
        win = (sel.pages != nullptr) ? sel.page_win[row_base] : make_uint2(0u, 0u);
        dense = 0u;
        #pragma unroll
        for (int h = 0; h < QSA_WALK_ROWS_PER_LANE; ++h) {
            const int r = lane + 32 * h;
            n[h] = 0u;
            c[h] = 0u;
            e[h] = 0u;
            if (r < n_rows) {
                const int row = row_base + r;
                const uint32_t cnt = sel.cnt[row];
                if (cnt == QSA_DENSE_ROW) {
                    dense |= 1u << h;
                } else {
                    n[h] = cnt;
                    e[h] = (uint32_t)row * sel.stride;
                }
            }
        }
    }

    // Bind `n_rows` rows that all attend everything, `block` positions per
    // step — the launch with no selection at all.
    __device__ __forceinline__ void init_dense(int n_rows, int block, int lane) {
        base = nullptr;
        ratio = block;
        pages = nullptr;
        win = make_uint2(0u, 0u);
        dense = 0u;
        #pragma unroll
        for (int h = 0; h < QSA_WALK_ROWS_PER_LANE; ++h) {
            n[h] = 0u;
            c[h] = 0u;
            e[h] = 0u;
            if (lane + 32 * h < n_rows) dense |= 1u << h;
        }
    }

    // The start of the lowest step at or past `bound` some row reads, or
    // QSA_WALK_END; `end` receives where that step stops, which is the
    // `bound` of the next call. Warp-collective: every lane returns the same
    // pair. Amortised, a cursor moves once per entry over the whole walk.
    //
    // **A step is one block, cut short by whatever another row reads next.**
    // A sparse row's block runs from its start for its width (`width_of`:
    // `ratio`, or less for a page's last block); a dense row reads every
    // position from `bound`. The step ends at the chosen block's end or at the
    // next position any row proposes, whichever is first — so a dense row
    // beside a ragged block covers the gap up to that block and then the block
    // itself, and no position is read twice or skipped. Without pages every
    // block is `ratio` wide and every proposal a multiple of it, so a step is
    // exactly one block, as the walk always stepped.
    //
    // `mask[h]` receives, for this lane's row h, one bit per cell of the step
    // that the row selects (bit c ⇔ position start + c), zero when the row
    // reads nothing in it.
    __device__ __forceinline__ int next(int bound, uint32_t (&mask)[QSA_WALK_ROWS_PER_LANE],
                                        int& end) {
        int best = QSA_WALK_END;
        int first_h[QSA_WALK_ROWS_PER_LANE];
        int cells_h[QSA_WALK_ROWS_PER_LANE];
        int width_h[QSA_WALK_ROWS_PER_LANE];
        #pragma unroll
        for (int h = 0; h < QSA_WALK_ROWS_PER_LANE; ++h) {
            first_h[h] = QSA_WALK_END;
            cells_h[h] = 0;
            width_h[h] = 0;
            if (dense & (1u << h)) {
                first_h[h] = bound;
                cells_h[h] = ratio;
                width_h[h] = ratio;
                best = bound;
                continue;
            }
            while (c[h] < n[h]) {
                const uint32_t ent = base[e[h] + c[h]];
                const uint32_t blk = ent >> QSA_CELL_BITS;
                const int first = start_of(blk);
                const int width = width_of(blk, first);
                // Consumed once the walk is past the block's start: the step
                // that took it ended at or after the block's own end.
                if (first < bound) {
                    c[h] += 1u;
                    continue;
                }
                first_h[h] = first;
                cells_h[h] = min((int)(ent & ((1u << QSA_CELL_BITS) - 1u)) + 1, width);
                width_h[h] = width;
                best = min(best, first);
                break;
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            best = min(best, __shfl_xor_sync(0xffffffffu, best, off));
        // The step's end: a row at `best` stops it at its own block's end, a
        // row proposing a later position stops it there.
        int stop = QSA_WALK_END;
        #pragma unroll
        for (int h = 0; h < QSA_WALK_ROWS_PER_LANE; ++h) {
            if (first_h[h] == QSA_WALK_END) continue;
            stop = min(stop, first_h[h] == best ? first_h[h] + width_h[h] : first_h[h]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            stop = min(stop, __shfl_xor_sync(0xffffffffu, stop, off));
        // At least one position forward, whatever the table says: a page that
        // claimed more blocks than its tokens fill gives a width of zero or
        // less, and a step that ended where it began would be returned again
        // on every call — the tile loop spinning on one block for good. With a
        // well-formed table every step is at least one cell wide already.
        if (best != QSA_WALK_END) stop = max(stop, best + 1);
        end = stop;
        const int span = (best == QSA_WALK_END) ? 0 : stop - best;
        #pragma unroll
        for (int h = 0; h < QSA_WALK_ROWS_PER_LANE; ++h) {
            const int cells = (first_h[h] == best) ? max(0, min(cells_h[h], span)) : 0;
            mask[h] = cells >= 32 ? ~0u : ((1u << cells) - 1u);
        }
        return best;
    }
};

} // namespace prefill_int8
