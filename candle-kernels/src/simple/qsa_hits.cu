// =============================================================================
// qsa_hits — count, per placed piece, the blocks a selection chose
// =============================================================================
//
// The hit-driven projection (`docs/hit_driven_projection.md` §6) moves each
// placed piece's score by how often the model's own indexer reads it. At a
// sampled layer, every block a query's selection names is one hit for the
// piece the block belongs to.
//
// One launch over a whole wave's rows, batched across its sequences (hot-path
// invariant 5): rows of one sequence are contiguous, and a SPAN table names
// each sequence's rows, its block→piece map and its counters — a descriptor
// table rather than one packed block (invariant 2b). Nothing is read back
// here; the counters are read at the projection boundary.
//
//   entries  [n_rows, entry_stride] u32 — the selection, `(block << 2) | (cells − 1)`
//   cnt      [n_rows] u32 — entries per row, or QSA_DENSE_ROW
//   spans    [n_spans] HitSpan — row range, block→piece map, counters
//
// A dense row made no selection — its context is inside the budget, so there
// is nothing to judge between — and counts nothing. The tail entry names the
// query's own block, which belongs to no placed piece; a block past a map's
// end, or mapped to QSA_NO_PIECE, is no piece's.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../qsa_select.cuh"

#define QSA_NO_PIECE 0xFFFFFFFFu

struct HitSpan {
    uint32_t row_lo;       // first row of the sequence in the launch
    uint32_t row_hi;       // one past its last
    uint32_t map_len;      // blocks the map covers
    uint32_t n_pieces;     // counters
    const uint32_t* map;   // [map_len] block → piece, QSA_NO_PIECE for none
    uint32_t* counters;    // [n_pieces] hits, accumulated
};

namespace qsa_hits {

constexpr int THREADS = 128;

// Grid: (row blocks, spans). Block (x, y) walks rows of span `y`.
__global__ void __launch_bounds__(THREADS) qsa_count_hits_kernel(
    const uint32_t* __restrict__ entries,
    int entry_stride,
    const uint32_t* __restrict__ cnt,
    const HitSpan* __restrict__ spans
) {
    const HitSpan s = spans[blockIdx.y];
    for (uint32_t row = s.row_lo + blockIdx.x; row < s.row_hi; row += gridDim.x) {
        const uint32_t n = cnt[row];
        if (n == QSA_DENSE_ROW) continue;
        const uint32_t* e = entries + (size_t)row * (size_t)entry_stride;
        for (uint32_t i = threadIdx.x; i < n; i += THREADS) {
            const uint32_t b = e[i] >> QSA_CELL_BITS;
            if (b >= s.map_len) continue;
            const uint32_t p = s.map[b];
            if (p == QSA_NO_PIECE || p >= s.n_pieces) continue;
            atomicAdd(&s.counters[p], 1u);
        }
    }
}

} // namespace qsa_hits

// `spans` is device memory holding `n_spans` HitSpan; `max_rows` the most rows
// any span has, which sizes the grid.
extern "C" void run_qsa_count_hits(
    const uint32_t* entries,
    int32_t entry_stride,
    const uint32_t* cnt,
    const void* spans,
    int32_t n_spans,
    int32_t max_rows,
    void* stream
) {
    if (n_spans <= 0 || max_rows <= 0) return;
    const unsigned rows = (unsigned)(max_rows < 64 ? max_rows : 64);
    dim3 grid(rows, (unsigned)n_spans);
    qsa_hits::qsa_count_hits_kernel<<<grid, qsa_hits::THREADS, 0, (cudaStream_t)stream>>>(
        entries, entry_stride, cnt, (const HitSpan*)spans);
}
