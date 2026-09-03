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
    int stride;
    int ratio;
};

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
    const uint32_t want = (uint32_t)(pos / sel.ratio);
    const uint32_t cell = (uint32_t)(pos - (int)want * sel.ratio);
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
