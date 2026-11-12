// =============================================================================
// qsa_hits — count, per placed piece, the blocks a selection chose, and
// publish the counts where the host reads them without waiting
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
// table rather than one packed block (invariant 2b).
//
//   entries  [n_rows, entry_stride] u32 — the selection, `(block << 2) | (cells − 1)`
//   cnt      [n_rows] u32 — entries per row, or QSA_DENSE_ROW
//   spans    [n_spans] HitSpan — row range, block→piece map, counters, summary
//
// A dense row made no selection — its context is inside the budget, so there
// is nothing to judge between — and counts nothing, not even as a query. The
// tail entry names the query's own block, which belongs to no placed piece; a
// block past a map's end, or mapped to QSA_NO_PIECE, is no piece's.
//
// The counters hold `n_pieces` hits, then the queries counted, then the
// publish sequence word, all cumulative over the declaration. After a wave's
// last sampled layer, the publish launch copies each counter set into its
// mapped summary block (`qwen4exp/hit_summary.rs` gives the layout): the
// sequence word first, the counts, the sequence word again last, each behind
// a system fence. Nothing is read back; the host reads the block at the next
// projection boundary, and never waits for it.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../qsa_select.cuh"

#define QSA_NO_PIECE 0xFFFFFFFFu

// Words ahead of the counts in a summary block: begin, token, generation,
// end, queries.
#define HIT_SUMMARY_HEADER 5u

struct HitSpan {
    uint32_t row_lo;       // first row of the sequence in the launch
    uint32_t row_hi;       // one past its last
    uint32_t map_len;      // blocks the map covers
    uint32_t n_pieces;     // hit counters
    const uint32_t* map;   // [map_len] block → piece, QSA_NO_PIECE for none
    uint32_t* counters;    // [n_pieces + 2] hits, queries, publish sequence
    uint32_t* summary;     // mapped [HIT_SUMMARY_HEADER + n_pieces] block
    uint32_t token;        // the holder the summary belongs to
    uint32_t generation;   // the holder's declaration
    uint32_t publish;      // 1 on the one span per counter set that publishes
    uint32_t pad;
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
        if (threadIdx.x == 0) atomicAdd(&s.counters[s.n_pieces], 1u);
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

// Grid: one block per span. A span that does not publish — a view counting
// into counters another span already publishes — exits.
__global__ void __launch_bounds__(THREADS) qsa_publish_hits_kernel(
    const HitSpan* __restrict__ spans
) {
    const HitSpan s = spans[blockIdx.x];
    if (!s.publish) return;
    volatile uint32_t* out = s.summary;
    __shared__ uint32_t seq;
    if (threadIdx.x == 0) {
        uint32_t next = s.counters[s.n_pieces + 1] + 1u;
        if (next == 0u) next = 1u;  // zero reads as never published
        s.counters[s.n_pieces + 1] = next;
        seq = next;
        out[0] = next;
        out[1] = s.token;
        out[2] = s.generation;
        __threadfence_system();
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i <= s.n_pieces; i += THREADS) {
        // Word 4 is the queries counter, which follows the hits on the device.
        const uint32_t v = (i == 0) ? s.counters[s.n_pieces] : s.counters[i - 1];
        out[4 + i] = v;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        out[3] = seq;
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

// Publish every publishing span's counters into its summary block.
extern "C" void run_qsa_publish_hits(const void* spans, int32_t n_spans, void* stream) {
    if (n_spans <= 0) return;
    qsa_hits::qsa_publish_hits_kernel<<<(unsigned)n_spans, qsa_hits::THREADS, 0,
                                        (cudaStream_t)stream>>>((const HitSpan*)spans);
}
