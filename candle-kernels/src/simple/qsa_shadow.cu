// =============================================================================
// qsa_shadow — what the indexer would have read of the pool it cannot see
// =============================================================================
//
// The real selection ranks only the slot, so on its own it can keep a round
// in the slot and never bring one in. The shadow scan asks, for every pooled
// round that is not loaded, what the indexer would have selected had it been
// there (`docs/hit_driven_projection.md` §6.3), and counts it into the same
// counters as a real hit.
//
// Each sequence keeps, per sampled layer, up to KEEP_SLOTS queries of its
// stride — rotated at their own positions, with their cut records
// (`../qsa_select.cuh`) — written by the hit count (`qsa_hits.cu`). At the
// boundary each kept query `q` at position `p`, with `C` candidate blocks and
// cut `c`, is scored against the sequence's unloaded rows:
//
//   - The rows are cut into windows of `C` rows in pool order. A window of `L`
//     rows is seated directly behind the query: its row `j` rotates at
//     `(C − L + j) · ratio`, so a full window sits at the positions the query's
//     own context does, and the query is the one the real selection used.
//   - Each window ranks its rows as the real selection ranks the slot: the
//     query's budget, `(top_k + ratio − 1 − tail)` cells, buys `keep` blocks.
//   - A row is one hit for its round when it is in its window's top `keep`
//     AND scores above the cut — the first keeps it to what the indexer would
//     choose from a context that size, the second keeps a window of nothing
//     relevant from handing out a full budget of hits.
//
// So a window's hits are the top `min(keep, n_above)` of the rows scoring
// above the cut, ranked `(score desc, row asc)`: the streaming survivors of
// `../qsa_survivors.cuh` over the keys of the rows above the cut. Each kept
// query adds one to the counters' shadow-query word, the denominator an
// unloaded round's hits are divided by (§6.4).
//
// One launch over the wave's due sequences: a SPAN table names each one's kept
// queries, its unloaded rows (a page descriptor table read in place, hot-path
// invariant 2b), their rows' counters and the rung its keys rotate on.
// Nothing is attended and nothing is read back.
//
// The page descriptors and their row prefixes travel in the same upload as the
// spans, after them, and a span names its own by WORD OFFSETS from the table's
// start: the descriptors hold the pages' addresses as they stand at the scan,
// since a compaction between forwards moves a page, and one upload carries all
// of it.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "../qsa_select.cuh"
#include "../qsa_survivors.cuh"
#include "../rope/rope_table.cuh"

// Kept queries a sequence holds per sampled layer: a decoded stride, or the
// last rows of a prefill.
#define QSA_KEEP_SLOTS 64

// i64 words per page descriptor: `{keys, cstride, rstride, delta}`, the
// paged scorer's (`qsa_score_paged.cu`). `delta` is unused: a shadow row's
// position is its seat in its window.
#define QSA_SHADOW_PAGE_WORDS 4

// The most index heads a kept query has.
#define QSA_SHADOW_MAX_HEADS 8

struct ShadowSpan {
    const uint32_t* keep;       // [KEEP_SLOTS, keep_words]: q as f32 bits, then the cut record
    const uint32_t* row_map;    // [n_rows] row → counter
    uint32_t* counters;         // [n_pieces + tail]: hits, queries, shadow queries, …
    const float* tab;           // the factored RoPE table of the sequence's rung
    uint32_t pages_off;         // words from the table's start to [n_pages * PAGE_WORDS] i64
    uint32_t first_off;         // words to [n_pages + 1] u32, the exclusive prefix over rows
    uint32_t n_rows;
    uint32_t n_pages;
    uint32_t n_pieces;
    uint32_t mask_lo;           // kept slots 0..31
    uint32_t mask_hi;           // kept slots 32..63
    uint32_t pad;
};

namespace qsa_shadow {

using namespace qsa_survivors;

// Page holding row `g`, by binary search over the exclusive prefix sum.
__device__ __forceinline__ int page_of(const uint32_t* __restrict__ first, int P, uint32_t g) {
    int lo = 0, hi = P;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (__ldg(first + mid) <= g) lo = mid; else hi = mid;
    }
    return lo;
}

// Grid: (KEEP_SLOTS, window groups, spans). Block (x, y, z) scores kept query
// `x` of span `z` against windows `y, y + gridDim.y, …` of its unloaded rows.
__global__ void __launch_bounds__(THREADS) qsa_shadow_kernel(
    const ShadowSpan* __restrict__ spans,
    int heads,
    int head_dim,
    int pairs,
    int ratio,
    int top_k
) {
    const ShadowSpan s = spans[blockIdx.z];
    const uint32_t* table = reinterpret_cast<const uint32_t*>(spans);
    const long long* pages = reinterpret_cast<const long long*>(table + s.pages_off);
    const uint32_t* page_first = table + s.first_off;
    const int slot = (int)blockIdx.x;
    const uint32_t bit = slot < 32 ? (s.mask_lo >> slot) & 1u : (s.mask_hi >> (slot - 32)) & 1u;
    if (!bit || s.n_rows == 0) return;
    const int qwords = heads * head_dim;
    const uint32_t* rec = s.keep + (size_t)slot * (size_t)(qwords + QSA_CUT_WORDS);
    const uint32_t* cut_rec = rec + qwords;
    const float cut = __uint_as_float(cut_rec[0]);
    // A dense row, or one that ranked nothing, keeps no query.
    if (isnan(cut)) return;
    const int cand = (int)cut_rec[2];
    const int tail = (int)cut_rec[3];
    if (cand <= 0) return;
    const int budget = top_k + ratio - 1 - tail;
    int keep = budget / ratio + (budget % ratio > 0 ? 1 : 0);
    if (keep > MAX_KEEP) keep = MAX_KEEP;
    if (keep <= 0) return;
    const int tid = (int)threadIdx.x;
    if (blockIdx.y == 0 && tid == 0) atomicAdd(&s.counters[s.n_pieces + 1], 1u);

    __shared__ float sq[QSA_SHADOW_MAX_HEADS * 256];
    __shared__ unsigned long long buf[CAP];
    __shared__ int s_count;
    __shared__ unsigned long long s_thr;
    __shared__ int s_hist[BINS];
    __shared__ int s_pick[3];
    const Survivors sv = {buf, &s_count, &s_thr, s_hist, s_pick};
    for (int i = tid; i < qwords; i += THREADS) sq[i] = __uint_as_float(rec[i]);
    __syncthreads();

    const float2* tab = reinterpret_cast<const float2*>(s.tab);
    const int n_rows = (int)s.n_rows;
    const int n_win = (n_rows + cand - 1) / cand;
    for (int w = (int)blockIdx.y; w < n_win; w += (int)gridDim.y) {
        const int lo = w * cand;
        const int hi = lo + cand < n_rows ? lo + cand : n_rows;
        const int seat = cand - (hi - lo);
        survivors_reset(sv, tid);
        int n = 0;
        for (int base = lo; base < hi; base += THREADS) {
            const int g = base + tid;
            unsigned long long key = 0ull;
            if (g < hi) {
                const int p = page_of(page_first, (int)s.n_pages, (uint32_t)g);
                const long long* pd = pages + (size_t)p * QSA_SHADOW_PAGE_WORDS;
                const float* kf = reinterpret_cast<const float*>((uintptr_t)__ldg(pd));
                const long long cstride = __ldg(pd + 1);
                const long long rstride = __ldg(pd + 2);
                const int local = g - (int)__ldg(page_first + p);
                const float* kb = kf + (long long)local * rstride * 4;
                // Channel `c` of this key, un-rotated: group `c/4` at `cstride`
                // float4s apart.
                auto raw = [&](int c) { return kb[(long long)(c >> 2) * cstride * 4 + (c & 3)]; };
                const int pos = (seat + (g - lo)) * ratio;
                float acc[QSA_SHADOW_MAX_HEADS];
#pragma unroll
                for (int h = 0; h < QSA_SHADOW_MAX_HEADS; ++h) acc[h] = 0.f;
                for (int c = 0; c < pairs; ++c) {
                    float kl = raw(c), kh = raw(c + pairs);
                    rope_f_rotate(kl, kh, rope_f_lookup(tab, pairs, pos, c));
                    for (int h = 0; h < heads; ++h) {
                        const float* qh = sq + h * head_dim;
                        acc[h] += qh[c] * kl + qh[c + pairs] * kh;
                    }
                }
                for (int c = 2 * pairs; c < head_dim; ++c) {
                    const float kv = raw(c);
                    for (int h = 0; h < heads; ++h) acc[h] += sq[h * head_dim + c] * kv;
                }
                float score = 0.f;
                for (int h = 0; h < heads; ++h) score += fmaxf(acc[h], 0.f);
                if (score > cut) key = score_key(score, g - lo);
            }
            survivors_offer(sv, key, keep, n, tid);
        }
        if (n == 0) continue;
        survivors_finish(sv, keep, n, tid);
        const int kept = *sv.count;
        for (int r = tid; r < kept; r += THREADS) {
            const uint32_t g = (uint32_t)lo + key_block(sv.buf[r]);
            const uint32_t piece = __ldg(s.row_map + g);
            if (piece < s.n_pieces) atomicAdd(&s.counters[piece], 1u);
        }
        __syncthreads();
    }
}

} // namespace qsa_shadow

// `spans` is device memory holding `n_spans` ShadowSpan followed by the page
// descriptors and row prefixes they name; `max_windows` the most windows any
// kept query cuts its span's rows into, which sizes the grid.
extern "C" void run_qsa_shadow(
    const void* spans,
    int32_t n_spans,
    int32_t max_windows,
    int32_t heads,
    int32_t head_dim,
    int32_t pairs,
    int32_t ratio,
    int32_t top_k,
    void* stream
) {
    if (n_spans <= 0 || max_windows <= 0) return;
    if (heads > QSA_SHADOW_MAX_HEADS || heads * head_dim > QSA_SHADOW_MAX_HEADS * 256) return;
    const unsigned groups = (unsigned)(max_windows < 16 ? max_windows : 16);
    dim3 grid(QSA_KEEP_SLOTS, groups, (unsigned)n_spans);
    qsa_shadow::qsa_shadow_kernel<<<grid, qsa_shadow::THREADS, 0, (cudaStream_t)stream>>>(
        (const ShadowSpan*)spans, heads, head_dim, pairs, ratio, top_k);
}
