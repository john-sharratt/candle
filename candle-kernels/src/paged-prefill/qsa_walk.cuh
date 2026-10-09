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
// (lane l owns rows l, l + 32, … of the block's run — `ROWS` slots, enough for
// the widest M the kernel packs at its head dim), and a warp-private cursor
// needs no smem handoff or barrier to stay block-uniform — each warp derives
// the identical block sequence from the identical lists.
//
// The walk is a serial chain — a step cannot start before the one it follows
// has chosen its block — so each step is kept free of memory round trips:
//
//   - a row's cursor holds its current entry AND the one after it in
//     registers, and moving the cursor issues the read of the entry after
//     that; a step reads entries that arrived during an earlier step. The
//     current entry's block start and width are resolved once, when the
//     cursor reaches it, not on every step that compares against it;
//   - a row's page-layout lookup remembers the page its last block fell in
//     and the first block of the next page, so the ascending walk resolves a
//     block's start by two compares instead of a binary search over the
//     table — the search runs only when a block leaves that page;
//   - the cross-lane minima are one `redux.sync` each, not a shuffle tree.
//
// A dense row (`QSA_DENSE_ROW`, or every row when the launch carries no
// selection) selects every cell of every block, so it contributes each
// successive block in turn; the walk over such a run steps through the whole
// prefix at `ratio` positions per step.
// ============================================================================

#include <stdint.h>
#include "../qsa_select.cuh"

namespace prefill_int8 {

// No selected block at or past the bound — the walk is over.
constexpr int QSA_WALK_END = 0x7fffffff;

/// No page remembered yet.
constexpr uint32_t QSA_WALK_NO_PAGE = 0xffffffffu;

template <int ROWS>
struct QsaWalk {
    static_assert(ROWS >= 1 && ROWS <= 2, "a lane owns one or two rows of the run");
    /// The widest run this walk binds.
    static constexpr int MAX_ROWS = 32 * ROWS;

    // The table base is the launch's — a kernel parameter, so the cursor
    // keeps 32-bit row offsets into it rather than a 64-bit pointer each.
    const uint32_t* base;
    uint32_t e[ROWS];        // this lane's rows' entry-list offsets
    uint32_t n[ROWS];        // entries per row; 0 = no row here
    uint32_t c[ROWS];        // cursor: first entry not yet passed
    uint32_t cur[ROWS];      // entry c (valid while c < n)
    uint32_t nxt[ROWS];      // entry c + 1 (valid while c + 1 < n)
    int cur_first[ROWS];     // entry c's block start (valid while c < n)
    int cur_width[ROWS];     // entry c's block width (valid while c < n)
    uint32_t dense;          // bit h: this lane's row h attends everything
    int ratio;               // positions per block
    // Page layout for this run, bound once — see `QsaSel::pages`. A prefill
    // block serves a run of consecutive query rows of ONE sequence, so every
    // row here shares a window and the block→position map is a property of the
    // run rather than of the row. Null when the sequence forwarded its whole
    // prefix, and then a block starts at `block * ratio`.
    const uint2* pages;
    uint2 win;
    // The page the lane's last lookup landed in: its index (or
    // `QSA_WALK_NO_PAGE`), its `{tokens_before, blocks_before}`, and the first
    // block of the page after it within the window (`UINT32_MAX` past the
    // window's last page).
    uint32_t pg;
    uint2 pg_at;
    uint32_t pg_end;

    // The page holding `block`, by binary search over the window: the last
    // page whose first block is at or below `block`.
    __device__ __forceinline__ uint32_t page_search(uint32_t block) const {
        uint32_t lo = win.x, hi = win.x + win.y;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (pages[mid].y <= block) lo = mid; else hi = mid;
        }
        return lo;
    }

    // The first key position of `block`, through this run's page layout —
    // without touching the remembered page (the seek's probes jump about).
    __device__ __forceinline__ int search_start(uint32_t block) const {
        if (pages == nullptr) return (int)block * ratio;
        const uint2 p = pages[page_search(block)];
        return (int)(p.x + (block - p.y) * (uint32_t)ratio);
    }

    // The first key position of `block`. The remembered page answers when
    // `block` falls in it — it is then exactly the last page at or below
    // `block`, as every later page starts at or past `pg_end`. A block past
    // it is found by galloping forward from the page after it (the walk
    // ascends, so the page sought is usually a few on), any other block by a
    // search over the window; either way the answer is remembered. Both find
    // the last page whose first block is at or below `block`: `page_search`'s
    // answer, whichever bracket the bisection starts from.
    __device__ __forceinline__ int start_of(uint32_t block) {
        if (pages == nullptr) return (int)block * ratio;
        if (pg == QSA_WALK_NO_PAGE || block < pg_at.y || block >= pg_end) {
            const uint32_t end = win.x + win.y;
            if (pg != QSA_WALK_NO_PAGE && block >= pg_end) {
                // pages[pg + 1] starts at pg_end <= block, so it is in range.
                uint32_t lo = pg + 1, step = 1;
                while (lo + step < end && pages[lo + step].y <= block) {
                    lo += step;
                    step <<= 1;
                }
                uint32_t hi = min(lo + step, end);
                while (lo + 1 < hi) {
                    const uint32_t mid = (lo + hi) >> 1;
                    if (pages[mid].y <= block) lo = mid; else hi = mid;
                }
                pg = lo;
            } else {
                pg = page_search(block);
            }
            pg_at = pages[pg];
            pg_end = (pg + 1 < end) ? pages[pg + 1].y : 0xffffffffu;
        }
        return (int)(pg_at.x + (block - pg_at.y) * (uint32_t)ratio);
    }

    // The cells `block` (starting at `first`) spans, at most `ratio` — the
    // walk's copy of `qsa_block_width`. A page's last block is short, and an
    // entry's full count would reach into the next block: attending its first
    // position twice when that block is selected too, and once when it is not.
    __device__ __forceinline__ int width_of(uint32_t block, int first) {
        if (pages == nullptr) return ratio;
        return min(ratio, start_of(block + 1u) - first);
    }

    // Row h's current entry's block start and width, resolved once per entry
    // rather than on every step that looks at it.
    __device__ __forceinline__ void resolve_cursor(int h) {
        if (c[h] < n[h]) {
            const uint32_t blk = cur[h] >> QSA_CELL_BITS;
            cur_first[h] = start_of(blk);
            cur_width[h] = width_of(blk, cur_first[h]);
        }
    }

    // Row h's cursor at `at`: the entry there and the one after it.
    __device__ __forceinline__ void load_cursor(int h, uint32_t at) {
        c[h] = at;
        cur[h] = (at < n[h]) ? base[e[h] + at] : 0u;
        nxt[h] = (at + 1u < n[h]) ? base[e[h] + at + 1u] : 0u;
        cur_first[h] = 0;
        cur_width[h] = 0;
        resolve_cursor(h);
    }

    // Row h's cursor one entry on; the read of the entry after the new one
    // goes out now and is consumed a step later.
    __device__ __forceinline__ void advance(int h) {
        c[h] += 1u;
        cur[h] = nxt[h];
        nxt[h] = (c[h] + 1u < n[h]) ? base[e[h] + c[h] + 1u] : 0u;
        resolve_cursor(h);
    }

    // Bind rows [row_base, row_base + n_rows) of `sel`. Called by every lane.
    __device__ __forceinline__ void init(const QsaSel& sel, int row_base, int n_rows, int lane) {
        base = sel.entries;
        ratio = sel.ratio;
        pages = sel.pages;
        win = (sel.pages != nullptr) ? sel.page_win[row_base] : make_uint2(0u, 0u);
        pg = QSA_WALK_NO_PAGE;
        pg_at = make_uint2(0u, 0u);
        pg_end = 0u;
        dense = 0u;
        #pragma unroll
        for (int h = 0; h < ROWS; ++h) {
            const int r = lane + 32 * h;
            n[h] = 0u;
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
            load_cursor(h, 0u);
        }
    }

    // Bind `n_rows` rows that all attend everything, `block` positions per
    // step — the launch with no selection at all.
    __device__ __forceinline__ void init_dense(int n_rows, int block, int lane) {
        base = nullptr;
        ratio = block;
        pages = nullptr;
        win = make_uint2(0u, 0u);
        pg = QSA_WALK_NO_PAGE;
        pg_at = make_uint2(0u, 0u);
        pg_end = 0u;
        dense = 0u;
        #pragma unroll
        for (int h = 0; h < ROWS; ++h) {
            n[h] = 0u;
            c[h] = 0u;
            e[h] = 0u;
            cur[h] = 0u;
            nxt[h] = 0u;
            cur_first[h] = 0;
            cur_width[h] = 0;
            if (lane + 32 * h < n_rows) dense |= 1u << h;
        }
    }

    // Whether no row of the run walks an entry list — every row is dense, or
    // reads nothing. Such a walk steps `ratio` positions from wherever its
    // bound stands, so a tile of it is the 32 positions from the bound.
    // Warp-collective.
    __device__ __forceinline__ bool all_dense() const {
        uint32_t any = 0u;
        #pragma unroll
        for (int h = 0; h < ROWS; ++h) any |= n[h];
        return !__any_sync(0xffffffffu, any != 0u);
    }

    // Move every sparse row's cursor to its first entry whose block starts at
    // or past `pos` — what `next` would reach by consuming the entries below
    // `pos` one step at a time, in a binary search per row. A dense row has no
    // cursor; a walk that seeks must have none (a dense row's step runs from
    // `bound` whatever it covers, so it has no entry to seek to).
    __device__ __forceinline__ void seek(int pos) {
        #pragma unroll
        for (int h = 0; h < ROWS; ++h) {
            uint32_t lo = c[h], hi = n[h];
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if (search_start(base[e[h] + mid] >> QSA_CELL_BITS) < pos) lo = mid + 1;
                else hi = mid;
            }
            load_cursor(h, lo);
        }
    }

    // The first position of the entry `ordinal` of this run's first row
    // (lane 0, slot 0) — where the `split`-th of `splits` equal shares of that
    // row's list begins. 0 for the first share, `QSA_WALK_END` past the last.
    // Warp-collective: every lane returns lane 0's answer.
    __device__ __forceinline__ int share_start(int split, int splits) const {
        int pos = 0;
        if (split >= splits) {
            pos = QSA_WALK_END;
        } else if (split > 0 && n[0] > 0) {
            const uint32_t ordinal = (uint32_t)(((uint64_t)n[0] * (uint32_t)split) / (uint32_t)splits);
            pos = search_start(base[e[0] + ordinal] >> QSA_CELL_BITS);
        }
        return __shfl_sync(0xffffffffu, pos, 0);
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
    __device__ __forceinline__ int next(int bound, uint32_t (&mask)[ROWS], int& end) {
        int best = QSA_WALK_END;
        int first_h[ROWS];
        int cells_h[ROWS];
        int width_h[ROWS];
        #pragma unroll
        for (int h = 0; h < ROWS; ++h) {
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
                const uint32_t ent = cur[h];
                const int first = cur_first[h];
                const int width = cur_width[h];
                // Consumed once the walk is past the block's start: the step
                // that took it ended at or after the block's own end.
                if (first < bound) {
                    advance(h);
                    continue;
                }
                first_h[h] = first;
                cells_h[h] = min((int)(ent & ((1u << QSA_CELL_BITS) - 1u)) + 1, width);
                width_h[h] = width;
                best = min(best, first);
                break;
            }
        }
        best = __reduce_min_sync(0xffffffffu, best);
        // The step's end: a row at `best` stops it at its own block's end, a
        // row proposing a later position stops it there.
        int stop = QSA_WALK_END;
        #pragma unroll
        for (int h = 0; h < ROWS; ++h) {
            if (first_h[h] == QSA_WALK_END) continue;
            stop = min(stop, first_h[h] == best ? first_h[h] + width_h[h] : first_h[h]);
        }
        stop = __reduce_min_sync(0xffffffffu, stop);
        // At least one position forward, whatever the table says: a page that
        // claimed more blocks than its tokens fill gives a width of zero or
        // less, and a step that ended where it began would be returned again
        // on every call — the tile loop spinning on one block for good. With a
        // well-formed table every step is at least one cell wide already.
        if (best != QSA_WALK_END) stop = max(stop, best + 1);
        end = stop;
        const int span = (best == QSA_WALK_END) ? 0 : stop - best;
        #pragma unroll
        for (int h = 0; h < ROWS; ++h) {
            const int cells = (first_h[h] == best) ? max(0, min(cells_h[h], span)) : 0;
            mask[h] = cells >= 32 ? ~0u : ((1u << cells) - 1u);
        }
        return best;
    }
};

} // namespace prefill_int8
