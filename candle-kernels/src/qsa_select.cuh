#pragma once
// ============================================================================
// QSA SELECTION — the attention side of the block-sparse read
// ============================================================================
//
// Qwen3.8-Flash-Next's full-attention layers attend a SELECTED subset of the
// prefix: an indexer scores `ratio`-cell blocks and each query keeps the best
// `top_k` positions plus its own trailing tail (at most 2051 cells for the
// released checkpoint, whatever the context depth). The host builds one
// ascending entry list per query row; this header is the predicate the
// attention kernels apply.
//
// It is the exact mirror of the Rust definition in
// `candle-transformers/src/models/qwen4exp/qsa_select.rs` — that module states
// the semantics and owns the packing; changing one without the other makes the
// engine disagree with its own oracle.
//
//   entry     = (block << 2) | (cells - 1)   — the block's LOWEST `cells` cells
//   sel_cnt   = number of entries for this row, ascending by block
//             = QSA_DENSE_ROW when the row attends everything visible
//
// A null `entries` pointer means no layer-side restriction at all — the state
// every model without QSA is in, and the state a QSA layer whose
// `compress_ratio` is 0 is in.
// ============================================================================

#include <stdint.h>

#define QSA_CELL_BITS 2
#define QSA_DENSE_ROW 0xFFFFFFFFu

// The per-launch selection view. `stride` is the row pitch of `entries` in
// u32s; `ratio` the layer's compression ratio (cells per block).
struct QsaSel {
    const uint32_t* entries;  // [n_rows, stride], ascending by block
    const uint32_t* cnt;      // [n_rows]
    // Page layout, for a sequence whose prefix arrived as independently sealed
    // pieces. `{tokens_before, blocks_before}` per page, ascending, concatenated
    // over the launch's sequences; `page_win[row]` is `{offset, count}` into it
    // for that row's sequence.
    //
    // **Null for a sequence that forwarded everything it holds**, which is every
    // sequence outside the projection path — and then block `b` covers
    // `[b*ratio, (b+1)*ratio)` exactly as it always did. The walk below reduces
    // to that arithmetic with one page `{0, 0}`, so the null case is a
    // dereference saved, not a different rule.
    const uint2* pages;
    const uint2* page_win;    // [n_rows]
    int stride;
    int ratio;
};

// The page holding `pos` (or the block `b`), by binary search over the ascending
// prefix sums. `key_is_block` picks which member of the pair to compare.
__device__ __forceinline__ uint2 qsa_page_for(
    const QsaSel& sel, int row, uint32_t key, bool key_is_block)
{
    const uint2 w = sel.page_win[row];
    uint32_t lo = w.x, hi = w.x + w.y;
    // Last page whose prefix is at or below `key`. The list is non-empty by
    // construction and its first entry is {0, 0}, so `lo` always lands.
    while (lo + 1 < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        const uint2 p = sel.pages[mid];
        const uint32_t at = key_is_block ? p.y : p.x;
        if (at <= key) lo = mid; else hi = mid;
    }
    return sel.pages[lo];
}

// The block covering key position `pos` for `row`.
__device__ __forceinline__ uint32_t qsa_block_of(const QsaSel& sel, int row, int pos) {
    if (sel.pages == nullptr) return (uint32_t)(pos / sel.ratio);
    const uint2 p = qsa_page_for(sel, row, (uint32_t)pos, false);
    return p.y + ((uint32_t)pos - p.x) / (uint32_t)sel.ratio;
}

// The first key position of `block` for `row` — the inverse of
// [`qsa_block_of`], which the prefill walk steps the prefix by.
__device__ __forceinline__ int qsa_block_start(const QsaSel& sel, int row, uint32_t block) {
    if (sel.pages == nullptr) return (int)block * sel.ratio;
    const uint2 p = qsa_page_for(sel, row, block, true);
    return (int)(p.x + (block - p.y) * (uint32_t)sel.ratio);
}

// The cells `block` (whose first position is `start`) spans for `row`, at most
// `ratio` — how many of an entry's cells a kernel that walks ENTRIES may read.
//
// Uniformly every block is `ratio` wide. Through a page layout a page ends where
// its tokens did, so its last block is short, and the position past it is the
// next block's first — decided by that block's own entry, not this one. An entry
// names its block's lowest cells, so its count clamped to this width is exactly
// the set `qsa_selects` accepts for it: the positions up to the next block's
// start, at most `ratio` of them. Where the next block starts right after a
// short block that is the short block's own positions; where an unindexed span
// sits between them it also covers the span's first cells, up to `ratio` from
// the block's start — which `qsa_selects` maps to this block as well.
//
// `block * ratio` is the start only for a sequence with no pages; through a page
// layout every block behind the first short one sits elsewhere, so a kernel
// that walks entries takes both the start and this width from the table.
__device__ __forceinline__ int qsa_block_width_from(
    const QsaSel& sel, int row, uint32_t block, int start)
{
    if (sel.pages == nullptr) return sel.ratio;
    return min(sel.ratio, qsa_block_start(sel, row, block + 1u) - start);
}

__device__ __forceinline__ bool qsa_active(const QsaSel& sel) {
    return sel.entries != nullptr;
}

// Whether `row` restricts its read at all. A dense row takes the causal mask
// alone, which is one comparison instead of a search per key.
__device__ __forceinline__ bool qsa_row_dense(const QsaSel& sel, int row) {
    return sel.cnt[row] == QSA_DENSE_ROW;
}

// Whether `row` attends key position `pos`.
//
// Binary search for the entry naming `pos / ratio`, then test `pos % ratio`
// against its cell count. The list is ascending and at most
// `top_k / ratio + 2` long (514 for the released checkpoint), so this is ≤ 10
// iterations.
__device__ __forceinline__ bool qsa_selects(const QsaSel& sel, int row, int pos) {
    const uint32_t n = sel.cnt[row];
    if (n == QSA_DENSE_ROW) return true;
    const uint32_t* e = sel.entries + (int64_t)row * sel.stride;
    const uint32_t want = qsa_block_of(sel, row, pos);
    const uint32_t cell = (uint32_t)(pos - qsa_block_start(sel, row, want));
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        uint32_t blk = e[mid] >> QSA_CELL_BITS;
        if (blk < want) {
            lo = mid + 1;
        } else if (blk > want) {
            hi = mid;
        } else {
            return cell <= (e[mid] & ((1u << QSA_CELL_BITS) - 1u));
        }
    }
    return false;
}
