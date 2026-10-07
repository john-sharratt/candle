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
// Rows come either from one dense `src` (`[n_rows, d]`), or — when `src_pages`
// is non-null — from a PAGE TABLE: row `r` at `src_pages[r / rows_per_src_page]
// + (r % rows_per_src_page) · d`. The live tail's keys are paged (one arena slot
// per page), so reading them through the table is what spares the caller a
// gather into one dense block (hot-path invariant 2b).
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
    const long long* __restrict__ src_pages,
    int rows_per_src_page,
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
        const float* s = src_pages != nullptr
            ? (const float*)(uintptr_t)__ldg(src_pages + row / rows_per_src_page)
                  + (long long)(row % rows_per_src_page) * d
            : src + (long long)row * d;
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

// The indexer's queries take an RMS norm with a per-channel weight before they
// rotate: `y = x / sqrt(Σx²·(1/d) + eps) · w`, then the rotation above. That
// was seven elementwise launches ahead of this one; here a warp owns a row,
// folds its square sum with an xor tree, writes the normed row to shared
// memory, and rotates it from there with the same lookup and scale. The
// operations and their order are the eager chain's — multiply by `1/d`, add
// `eps`, square root, divide, weight — apart from the order of the square sum.
constexpr int NORM_WARPS = 4;
constexpr int NORM_MAX_D = 256;

__global__ void rope_rows_norm_kernel(
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
    int q_scale,
    const float* __restrict__ norm_w,
    float eps
) {
    __shared__ float s_row[NORM_WARPS][NORM_MAX_D];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * NORM_WARPS + warp;
    if (row >= n_rows) return;
    const float* s = src + (long long)row * d;
    float* o = dst + (long long)row * d;

    float ss = 0.f;
    for (int c = lane; c < d; c += 32) {
        const float x = s[c];
        ss = fmaf(x, x, ss);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float ms = ss * (1.f / (float)d);
    const float denom = sqrtf(ms + eps);
    for (int c = lane; c < d; c += 32) s_row[warp][c] = s[c] / denom * __ldg(norm_w + c);
    __syncwarp();

    const int pairs = (int)rungs.pairs;
    const int group = row / rows_per_pos;
    const int p = pos != nullptr ? (int)__ldg(pos + group)
                                 : (int)(pos_base + (long long)group * pos_step);
    const RopeView v = rope_view(rungs, group_rung != nullptr ? __ldg(group_rung + group) : rung);
    const float scale = q_scale ? v.q_scale : 1.f;
    for (int k = lane; k < d - pairs; k += 32) {
        if (k < pairs) {
            float lo = s_row[warp][k];
            float hi = s_row[warp][k + pairs];
            rope_f_rotate(lo, hi, rope_f_lookup(v.tab, pairs, p, k));
            o[k] = lo * scale;
            o[k + pairs] = hi * scale;
        } else {
            const int c = k + pairs;
            o[c] = s_row[warp][c];
        }
    }
}

} // namespace qsa_rope_rows

// The rows of a dense `src`, RMS-normed with `norm_w` and `eps`, then rotated as
// `run_qsa_rope_rows` rotates them. `d` must not exceed `NORM_MAX_D`.
extern "C" void run_qsa_rope_rows_norm(
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
    const float* norm_w,
    float eps,
    void* stream
) {
    const int pairs = (int)rungs.pairs;
    if (n_rows <= 0 || d <= 0 || d > qsa_rope_rows::NORM_MAX_D || pairs <= 0 || 2 * pairs > d
        || rows_per_pos <= 0) return;
    const unsigned blocks =
        (unsigned)((n_rows + qsa_rope_rows::NORM_WARPS - 1) / qsa_rope_rows::NORM_WARPS);
    qsa_rope_rows::rope_rows_norm_kernel<<<blocks, qsa_rope_rows::NORM_WARPS * 32, 0,
                                           (cudaStream_t)stream>>>(
        src, dst, n_rows, d, rows_per_pos, pos, pos_base, pos_step, rungs, group_rung, rung,
        q_scale, norm_w, eps);
}

extern "C" void run_qsa_rope_rows(
    const float* src,
    const long long* src_pages,
    int32_t rows_per_src_page,
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
    if (src_pages != nullptr && rows_per_src_page <= 0) return;
    const long long total = (long long)n_rows * (d - pairs);
    const int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    qsa_rope_rows::rope_rows_kernel<<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(
        src, src_pages, rows_per_src_page, dst, n_rows, d, rows_per_pos, pos, pos_base, pos_step,
        rungs, group_rung, rung, q_scale);
}
