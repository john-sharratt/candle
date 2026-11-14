// =============================================================================
// qsa_hits — count, per declared round, the blocks a selection chose, keep the
// stride's queries for the shadow scan, and publish the counts where the host
// reads them without waiting
// =============================================================================
//
// The hit-driven projection (`docs/hit_driven_projection.md` §6) moves each
// round's score by how often the model's own indexer reads it. At a sampled
// layer, every block a query's selection names is one hit for the round the
// block belongs to.
//
// One launch over a whole wave's rows, batched across its sequences (hot-path
// invariant 5): rows of one sequence are contiguous, and a SPAN table names
// each sequence's rows, its block→round map, its counters and its kept-query
// buffer — a descriptor table rather than one packed block (invariant 2b).
//
//   entries  [n_rows, entry_stride] u32 — the selection, `(block << 2) | (cells − 1)`
//   cnt      [n_rows] u32 — entries per row, or QSA_DENSE_ROW
//   q        [n_rows, q_words] f32 — each row's indexer query, rotated
//   cuts     [n_rows, QSA_CUT_WORDS] u32 — each row's cut record
//   spans    [n_spans] HitSpan
//
// A dense row made no selection — its context is inside the budget, so there
// is nothing to judge between — and counts nothing, not even as a query. The
// tail entry names the query's own block, which belongs to no round; a block
// past a map's end, or mapped to QSA_NO_PIECE, is no round's.
//
// **The stride's queries are kept here, not in a launch of their own.** The
// last QSA_KEEP_SLOTS rows of each span — a decoded stride's tokens, a
// prefill's last rows — are copied with their cut records into the span's
// kept-query buffer at this layer, slot `position mod QSA_KEEP_SLOTS`, where
// the shadow scan (`qsa_shadow.cu`) reads them at the boundary. A dense row's
// record carries QSA_NO_CUT, so the scan passes it by.
//
// The counters hold `n_pieces` hits, then the queries counted, the shadow
// queries counted, and the publish sequence word, all cumulative over the
// declaration. After a wave's last sampled layer, the publish launch copies
// each counter set into its mapped summary block (`qwen4exp/hit_summary.rs`
// gives the layout): the sequence word first, the counts, the sequence word
// again last, each behind a system fence. Nothing is read back; the host reads
// the block at the next projection boundary, and never waits for it.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../qsa_select.cuh"

#define QSA_NO_PIECE 0xFFFFFFFFu

// Words ahead of the counts in a summary block: begin, token, generation,
// end, queries, shadow queries.
#define HIT_SUMMARY_HEADER 6u

// Kept queries per sequence per sampled layer (`qsa_shadow.cu`).
#define QSA_KEEP_SLOTS 64u

struct HitSpan {
    uint32_t row_lo;       // first row of the sequence in the launch
    uint32_t row_hi;       // one past its last
    uint32_t map_len;      // blocks the map covers
    uint32_t n_pieces;     // hit counters
    const uint32_t* map;   // [map_len] block → round, QSA_NO_PIECE for none
    uint32_t* counters;    // [n_pieces + 3] hits, queries, shadow queries, publish sequence
    uint32_t* summary;     // mapped [HIT_SUMMARY_HEADER + n_pieces] block
    uint32_t* keep;        // [layers, QSA_KEEP_SLOTS, q_words + QSA_CUT_WORDS] kept queries
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
    const float* __restrict__ q,
    int q_words,
    const uint32_t* __restrict__ cuts,
    int layer,
    const HitSpan* __restrict__ spans
) {
    const HitSpan s = spans[blockIdx.y];
    const uint32_t keep_from = s.row_hi - s.row_lo > QSA_KEEP_SLOTS ? s.row_hi - QSA_KEEP_SLOTS : s.row_lo;
    const int rec_words = q_words + QSA_CUT_WORDS;
    for (uint32_t row = s.row_lo + blockIdx.x; row < s.row_hi; row += gridDim.x) {
        const uint32_t* cut = cuts + (size_t)row * QSA_CUT_WORDS;
        if (row >= keep_from) {
            const uint32_t slot = cut[1] % QSA_KEEP_SLOTS;
            uint32_t* dst = s.keep + ((size_t)layer * QSA_KEEP_SLOTS + slot) * (size_t)rec_words;
            const uint32_t* src = reinterpret_cast<const uint32_t*>(q) + (size_t)row * q_words;
            for (int i = threadIdx.x; i < q_words; i += THREADS) dst[i] = src[i];
            if (threadIdx.x < QSA_CUT_WORDS) dst[q_words + threadIdx.x] = cut[threadIdx.x];
        }
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
        uint32_t next = s.counters[s.n_pieces + 2] + 1u;
        if (next == 0u) next = 1u;  // zero reads as never published
        s.counters[s.n_pieces + 2] = next;
        seq = next;
        out[0] = next;
        out[1] = s.token;
        out[2] = s.generation;
        __threadfence_system();
    }
    __syncthreads();
    // Words 4 and 5 are the queries and the shadow queries, which follow the
    // hits on the device.
    for (uint32_t i = threadIdx.x; i < s.n_pieces + 2u; i += THREADS) {
        const uint32_t v = (i < 2u) ? s.counters[s.n_pieces + i] : s.counters[i - 2u];
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
// any span has, which sizes the grid. `layer` is the sampled layer's index
// into each span's kept-query buffer.
extern "C" void run_qsa_count_hits(
    const uint32_t* entries,
    int32_t entry_stride,
    const uint32_t* cnt,
    const float* q,
    int32_t q_words,
    const uint32_t* cuts,
    int32_t layer,
    const void* spans,
    int32_t n_spans,
    int32_t max_rows,
    void* stream
) {
    if (n_spans <= 0 || max_rows <= 0) return;
    const unsigned rows = (unsigned)(max_rows < 64 ? max_rows : 64);
    dim3 grid(rows, (unsigned)n_spans);
    qsa_hits::qsa_count_hits_kernel<<<grid, qsa_hits::THREADS, 0, (cudaStream_t)stream>>>(
        entries, entry_stride, cnt, q, q_words, cuts, layer, (const HitSpan*)spans);
}

// Publish every publishing span's counters into its summary block.
extern "C" void run_qsa_publish_hits(const void* spans, int32_t n_spans, void* stream) {
    if (n_spans <= 0) return;
    qsa_hits::qsa_publish_hits_kernel<<<(unsigned)n_spans, qsa_hits::THREADS, 0,
                                        (cudaStream_t)stream>>>((const HitSpan*)spans);
}
