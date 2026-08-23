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

extern "C" __global__ void rows_scatter_kernel(
    const long long* __restrict__ desc,
    int n_runs)
{
    const int e = blockIdx.y;
    if (e >= n_runs) return;
    const long long* __restrict__ d = desc + (long long)e * ROWS_SCATTER_WORDS;
    const unsigned int* __restrict__ src = (const unsigned int*)d[0];
    const long long src_s = d[1];
    unsigned int* __restrict__ dst = (unsigned int*)d[2];
    const long long dst_s = d[3];
    const int rows = (int)d[4];
    const int words = (int)d[5];

    const long long total = (long long)rows * (long long)words;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        const long long r = i / words;
        const long long c = i - r * words;
        dst[r * dst_s + c] = src[r * src_s + c];
    }
}

extern "C" void run_rows_scatter(
    const long long* desc,
    int n_runs,
    int max_elems,
    void* stream)
{
    if (n_runs <= 0 || max_elems <= 0) return;
    const int threads = 256;
    int tiles = (max_elems + threads - 1) / threads;
    // The runs are small — a wave's whole append is tens of kilobytes — so cap
    // the tile count and let the grid-stride loop cover anything larger rather
    // than launching a block per 256 words of the widest run.
    if (tiles > 64) tiles = 64;
    dim3 grid(tiles, n_runs, 1);
    rows_scatter_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(desc, n_runs);
}
