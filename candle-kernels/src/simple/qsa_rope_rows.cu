// =============================================================================
// QSA: rotate rows at their positions from the factored RoPE table
// =============================================================================
// The QSA index stores its keys un-rotated, and the paged scorer rotates each
// key as it loads it. Two things still need rows rotated ahead of a read, and
// this kernel is both of them:
//
//   * the indexer's QUERIES, once per layer per wave — every row at its own
//     absolute position, all heads of a row at that row's position;
//   * the live tail's keys when a span is too wide for the paged scorer and is
//     scored by cuBLAS instead — rotated into a scratch buffer that lives for
//     one launch (`IndexCache::score_rows`), at `tail_base + j · ratio`.
//
// Positions come either from a per-group array (`pos`, one entry per
// `rows_per_pos` consecutive rows) or, when `pos` is null, from the affine
// `pos_base + group · pos_step`.
//
// Each group rotates at its own rung of the model's `RopeRungs`: from
// `group_rung` (one entry per group — a wave's queries, whose sequences sit on
// different rungs) or, when that is null, at `rung` (one sequence's keys). A
// rung past the set traps (`rope_view`), so no group reads another's table.
//
// NeoX half-split over the rotary width: channel `c < pairs` pairs with
// `c + pairs`, and `[2·pairs, d)` is copied through. With `q_scale` set, the
// rotated channels are multiplied by the rung's `m²` — a query's scale
// (§4.2); keys take none.
//
// One thread per `(row, channel)` in the lower half of the rotary width plus
// the pass-through channels. A memory-bound pass over a few hundred KB; its job
// is to be one launch, not to be clever.

#include <cuda_runtime.h>
#include <stdint.h>

#include "../rope/rope_table.cuh"

namespace qsa_rope_rows {

__global__ void rope_rows_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n_rows,
    int d,
    int rows_per_pos,
    const uint32_t* __restrict__ pos,
    long long pos_base,
    int pos_step,
    const RopeRungs rungs,
    const uint32_t* __restrict__ group_rung,
    uint32_t rung,
    int q_scale
) {
    const int pairs = (int)rungs.pairs;
    // Work items per row: the `pairs` rotary pairs, then the pass-through
    // channels one each.
    const int per_row = d - pairs;
    const long long total = (long long)n_rows * per_row;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        const int row = (int)(i / per_row);
        const int k = (int)(i - (long long)row * per_row);
        const float* s = src + (long long)row * d;
        float* o = dst + (long long)row * d;
        if (k < pairs) {
            const int group = row / rows_per_pos;
            const int p = pos != nullptr
                ? (int)__ldg(pos + group)
                : (int)(pos_base + (long long)group * pos_step);
            const RopeView v = rope_view(
                rungs, group_rung != nullptr ? __ldg(group_rung + group) : rung);
            const float scale = q_scale ? v.q_scale : 1.f;
            float lo = s[k];
            float hi = s[k + pairs];
            rope_f_rotate(lo, hi, rope_f_lookup(v.tab, pairs, p, k));
            o[k] = lo * scale;
            o[k + pairs] = hi * scale;
        } else {
            // Pass-through channel `c = k + pairs`, which is `>= 2·pairs`.
            const int c = k + pairs;
            o[c] = s[c];
        }
    }
}

} // namespace qsa_rope_rows

extern "C" void run_qsa_rope_rows(
    const float* src,
    float* dst,
    int32_t n_rows,
    int32_t d,
    int32_t rows_per_pos,
    const uint32_t* pos,
    long long pos_base,
    int32_t pos_step,
    RopeRungs rungs,
    const uint32_t* group_rung,
    uint32_t rung,
    int32_t q_scale,
    void* stream
) {
    const int pairs = (int)rungs.pairs;
    if (n_rows <= 0 || d <= 0 || pairs <= 0 || 2 * pairs > d || rows_per_pos <= 0) return;
    const long long total = (long long)n_rows * (d - pairs);
    const int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    qsa_rope_rows::rope_rows_kernel<<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(
        src, dst, n_rows, d, rows_per_pos, pos, pos_base, pos_step,
        rungs, group_rung, rung, q_scale);
}
