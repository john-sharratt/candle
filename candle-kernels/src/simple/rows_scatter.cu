// =============================================================================
// Batched row scatter: many (source run → destination offset) copies, one launch
// =============================================================================
// Appending a wave's completed compressor groups to the per-session corpus
// galleries is a scatter, not a copy: every session takes its own slice of one
// fleet-wide pooled block and writes it at its own gallery's current length, in
// four separate arrays (Indexer keys, positions, packed sign bits, and the
// three regions of the two-region latent cache). Expressed with `slice_set` that
// is one launch per array per session per compression layer — for a 16-session
// wave, hundreds of ~1 us launches per step doing a few kilobytes of work each,
// which is launch overhead almost end to end.
//
// This kernel takes them as a DESCRIPTOR TABLE (hot-path invariant 2b): a list
// of runs, each `{src, src row stride, dst, dst row stride, rows, words}`, all
// copied in ONE launch. Destination offsets are baked into `dst` host-side, so
// the kernel only walks rows.
//
// Rows are copied as 32-BIT WORDS regardless of the element type, the same trick
// `corpus_gather_rows_batched` uses: every array this scatters has a row width
// that is a multiple of 4 bytes (f32 and u32 rows trivially, the int8 nope band
// at 448 B, the bf16 rope tail at 128 B), so one word-typed kernel serves all of
// them and the table can mix arrays of different element types in a single
// launch. The host asserts the multiple.
//
// Table layout, array-of-structs, ROWS_SCATTER_WORDS i64 per run:
//     [0] src base pointer      [1] src row stride (WORDS)
//     [2] dst base pointer      [3] dst row stride (WORDS)
//     [4] rows                  [5] words per row
//
// grid.y indexes the run, grid.x tiles its elements. Runs are ragged — a
// position array is one word per row, a latent band is 112 — so the grid is
// sized by the widest run and narrower ones exit on the bounds check. That
// wastes blocks, not bandwidth, and it is what keeps the launch count at one.

#include <cuda_runtime.h>

#define ROWS_SCATTER_WORDS 6
// Runs whose descriptor fits in kernel parameters. 8 × 6 × 8 B = 384 B of the
// 4 KB limit, which comfortably covers every projection split (four runs) while
// leaving the gallery's hundreds on the pointer path.
#define ROWS_SCATTER_INLINE_MAX 8

struct RowsScatterInline {
    long long w[ROWS_SCATTER_WORDS * ROWS_SCATTER_INLINE_MAX];
};

// The grid is (column tiles, runs, row chunks) and **nothing divides**.
//
// The first version indexed a run flatly, `i` over `rows·words`, and recovered
// the row with `r = i / words` — a 64-bit division *per element*. That is
// invisible at the geometry this kernel was written for (a gallery append is
// tens of kilobytes, so the divisions are lost in the launch) and ruinous at the
// one it also serves now: splitting a stacked projection is four runs of
// 10240/6144/48/48 words over every row of the block, ~40M elements, and the
// flat form measured **437 GB/s — 38% SLOWER than the per-part `contiguous`
// copies it replaces**.
//
// Giving the row its own grid axis removes the division outright: `c` comes from
// `blockIdx.x`, `r` from `blockIdx.z`, and consecutive threads hold consecutive
// columns, which is also what makes the accesses coalesce. Runs are ragged, so
// the grid is sized by the widest and the longest and narrower or shorter runs
// exit on their bounds checks — that wastes blocks, not bandwidth, and it is
// what keeps the launch count at one.
//
// VECTORISATION is decided **per run**, not per element: a run whose width, both
// row strides and both base pointers are all 16-byte aligned moves `uint4`, and
// otherwise it moves words. The test is `blockIdx.y`-uniform, so the branch is
// outside the inner loop and costs the scalar path nothing — the objection §0.4
// rule 2 raises against runtime branches is that they put a predicate in
// everyone's inner loop, which this does not. A template parameter cannot serve
// here because one launch deliberately mixes runs of different widths.
template <typename T>
__device__ __forceinline__ void rows_scatter_body(
    const T* __restrict__ src, long long src_s,
    T* __restrict__ dst, long long dst_s,
    int rows, int cols)
{
    const int c0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int cstride = gridDim.x * blockDim.x;
    for (int r = blockIdx.z; r < rows; r += gridDim.z) {
        const T* __restrict__ s = src + r * src_s;
        T* __restrict__ d = dst + r * dst_s;
        for (int c = c0; c < cols; c += cstride) {
            d[c] = s[c];
        }
    }
}

__device__ __forceinline__ void rows_scatter_run(const long long* d)
{
    const unsigned int* __restrict__ src = (const unsigned int*)d[0];
    const long long src_s = d[1];
    unsigned int* __restrict__ dst = (unsigned int*)d[2];
    const long long dst_s = d[3];
    const int rows = (int)d[4];
    const int words = (int)d[5];

    // `uint4` needs 16 B on both sides: the width, both row strides (so every
    // row starts aligned too) and both bases. Word strides are 4 B each, so the
    // stride test is a multiple of four.
    const bool vec_ok =
        (words & 3) == 0 && (src_s & 3) == 0 && (dst_s & 3) == 0 &&
        ((((unsigned long long)src) | ((unsigned long long)dst)) & 15ull) == 0;

    if (vec_ok) {
        rows_scatter_body<uint4>(
            (const uint4*)src, src_s >> 2, (uint4*)dst, dst_s >> 2, rows, words >> 2);
    } else {
        rows_scatter_body<unsigned int>(src, src_s, dst, dst_s, rows, words);
    }
}

// The descriptor by POINTER: the table lives in the pinned arena and every block
// reads its six words **over PCIe**, uncached and at host latency.
//
// That is the right trade for a gallery append, whose own note records it — the
// table is large (384 runs), an upload costs 24 us, and the launch is a handful
// of blocks that hide the reads. It is the wrong trade for a wide split, which
// launches hundreds of blocks that each stall on the same read before they can
// begin: measured 54.8 us of kernel time at **0.73% SM and 1.51% DRAM** for a
// 2.1 MB copy, i.e. entirely descriptor latency. Use `..._inline` below when the
// run count is small enough to pass the table in kernel parameters instead.
extern "C" __global__ void rows_scatter_kernel(
    const long long* __restrict__ desc,
    int n_runs)
{
    const int e = blockIdx.y;
    if (e >= n_runs) return;
    rows_scatter_run(desc + (long long)e * ROWS_SCATTER_WORDS);
}

// The descriptor by VALUE, for a few runs.
//
// Kernel parameters live in constant memory: broadcast to every thread, cached,
// and read without touching the bus at all. `ROWS_SCATTER_INLINE_MAX` runs is
// `6 · 8 · MAX` bytes of the 4 KB parameter space, which is what bounds it — a
// gallery append's 384 runs cannot fit and takes the pointer form above. The
// split of a stacked projection is four runs and always fits.
//
// Two entry points rather than a branch inside one, because the difference is
// *where the descriptor lives* — a parameter cannot be conditionally a pointer.
extern "C" __global__ void rows_scatter_inline_kernel(
    const long long* __restrict__ desc_unused,
    int n_runs,
    RowsScatterInline table)
{
    (void)desc_unused;
    const int e = blockIdx.y;
    if (e >= n_runs) return;
    rows_scatter_run(&table.w[e * ROWS_SCATTER_WORDS]);
}

extern "C" void run_rows_scatter(
    const long long* desc,
    const long long* host_desc,
    int n_runs,
    int max_elems,
    int max_rows,
    void* stream)
{
    if (n_runs <= 0 || max_elems <= 0 || max_rows <= 0) return;
    const int threads = 256;
    // Column tiles cover the widest run's *vector* width — sizing on the scalar
    // width would launch four times the blocks the vector path needs and leave
    // three quarters of them exiting immediately. A run that cannot vectorise
    // covers the rest through the column grid-stride.
    int ctiles = ((max_elems >> 2) + threads - 1) / threads;
    if (ctiles < 1) ctiles = 1;
    if (ctiles > 64) ctiles = 64;
    // Row chunks: enough blocks to fill the device without making the grid
    // scale with a prefill's row count. The row grid-stride covers the rest.
    int rchunks = max_rows;
    if (rchunks > 64) rchunks = 64;
    dim3 grid((unsigned)ctiles, (unsigned)n_runs, (unsigned)rchunks);
    // Few enough runs to carry the table in parameters? Then do — it is the
    // difference between every block stalling on a PCIe read and none of them
    // touching the bus. `host_desc` is the same words the device table holds;
    // the caller has them already, so this costs nothing to supply.
    if (host_desc != nullptr && n_runs <= ROWS_SCATTER_INLINE_MAX) {
        RowsScatterInline table;
        for (int i = 0; i < n_runs * ROWS_SCATTER_WORDS; ++i) {
            table.w[i] = host_desc[i];
        }
        rows_scatter_inline_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
            desc, n_runs, table);
        return;
    }
    rows_scatter_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(desc, n_runs);
}
