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

    // Bind rows [row_base, row_base + n_rows) of `sel`. Called by every lane.
    __device__ __forceinline__ void init(const QsaSel& sel, int row_base, int n_rows, int lane) {
        base = sel.entries;
        ratio = sel.ratio;
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
        dense = 0u;
        #pragma unroll
        for (int h = 0; h < QSA_WALK_ROWS_PER_LANE; ++h) {
            n[h] = 0u;
            c[h] = 0u;
            e[h] = 0u;
            if (lane + 32 * h < n_rows) dense |= 1u << h;
        }
    }

    // The start of the lowest block at or past `bound` some row selects, or
    // QSA_WALK_END. `bound` must be a block start (a multiple of `ratio`):
    // every entry names one whole block, so the cursors pass exactly the
    // entries below the bound and the survivor's first cell is the block's
    // start. Warp-collective: every lane returns the same value. Amortised,
    // a cursor moves once per entry over the whole walk.
    //
    // `mask[h]` receives, for this lane's row h, one bit per cell of the
    // returned block that the row selects (bit c ⇔ position start + c),
    // zero when the row does not select that block.
    __device__ __forceinline__ int next(int bound, uint32_t (&mask)[QSA_WALK_ROWS_PER_LANE]) {
        int best = QSA_WALK_END;
        int first_h[QSA_WALK_ROWS_PER_LANE];
        int cells_h[QSA_WALK_ROWS_PER_LANE];
        #pragma unroll
        for (int h = 0; h < QSA_WALK_ROWS_PER_LANE; ++h) {
            first_h[h] = QSA_WALK_END;
            cells_h[h] = 0;
            if (dense & (1u << h)) {
                first_h[h] = bound;
                cells_h[h] = ratio;
                best = bound;
                continue;
            }
            while (c[h] < n[h]) {
                const uint32_t ent = base[e[h] + c[h]];
                const int first = (int)(ent >> QSA_CELL_BITS) * ratio;
                const int cells = (int)(ent & ((1u << QSA_CELL_BITS) - 1u)) + 1;
                if (first + cells - 1 < bound) {
                    c[h] += 1u;
                    continue;
                }
                first_h[h] = first;
                cells_h[h] = cells;
                best = min(best, first);
                break;
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            best = min(best, __shfl_xor_sync(0xffffffffu, best, off));
        #pragma unroll
        for (int h = 0; h < QSA_WALK_ROWS_PER_LANE; ++h) {
            const bool hit = (first_h[h] == best);
            mask[h] = !hit ? 0u : (cells_h[h] >= 32 ? ~0u : ((1u << cells_h[h]) - 1u));
        }
        return best;
    }
};

} // namespace prefill_int8
