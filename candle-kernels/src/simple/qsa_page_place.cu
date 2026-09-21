// =============================================================================
// QSA index page PLACEMENT: channel-block a page for the scorer
// =============================================================================
// An index page's rows are the pooled, normed, UN-rotated block keys of one
// sealed unit, stored `[rows, d]` as a record holds them. The paged scorer reads
// keys channel-blocked as `[d/4, rows, 4]`, which makes a warp's key read 512
// contiguous bytes instead of 32 scattered ones — so a page is laid out that way
// once, when it is placed, rather than once per score.
//
// Placing carries no rotation. The scorer rotates each key as it loads it, at
// the key's own position (`qsa_score_paged.cu`), so a page's layout is the same
// wherever it sits and whatever RoPE schedule reads it.
//
// ---- Shape of the work ------------------------------------------------------
//
// Pure bandwidth: `rows · d` floats in, the same out. One side of a transpose is
// always strided, so a block stages a `TILE_R × d` tile in shared memory —
// coalesced `float4` reads in, coalesced `float4` writes out, and the strided
// axis paid in shared where it is cheap.
//
//   load    consecutive threads take consecutive channel groups of one row
//   store   consecutive threads take consecutive ROWS of one channel group,
//           which is exactly the `[d/4, rows, 4]` order the scorer reads
//
// ---- Batching ---------------------------------------------------------------
//
// A projection places many pages at once, so the launch is batched over a
// descriptor table (hot-path invariant 5): one job per page. `gridDim.y`
// indexes the job and `gridDim.x` the row tile, sized to the widest job; a job
// with fewer tiles retires its extra blocks in the first instruction.
//
// Table layout: array-of-structs, `QSA_PLACE_JOB_WORDS` i64 per job.
//
//     [0] src    const float*  the page's `[rows, d]` keys
//     [1] dst    float*        the page's `[d/4, rows, 4]` placement
//     [2] rows

#include <cuda_runtime.h>
#include <stdint.h>

#define QSA_PLACE_JOB_WORDS 3
#define QSA_PLACE_THREADS 256

namespace qsa_page_place {

/// Padding, in `float4`s, between shared-tile rows.
///
/// The store phase walks a channel group down the tile — stride `D4 + PAD`
/// `float4`s — so an unpadded tile would put every row of a group in the same
/// bank quartet. One `float4` of padding shifts each row by four banks.
constexpr int PAD = 1;

/// One `TILE_R × d` tile of one page.
///
/// `TILE_R` is a template parameter because it is the kernel's only real dial
/// and it pulls two ways: it is the length of the store phase's contiguous run
/// (coalescing) and it multiplies the shared footprint (occupancy). At the
/// released geometry `d = 128`, so a tile is `TILE_R · 33` `float4` — 8.4 KB at
/// 16, 16.9 KB at 32.
///
/// **Where it is measured, it is finished.** `ncu` on the two shapes that
/// actually stream — 32 pages × 512 rows and one 32,768-row section — puts this
/// at **87–88% of peak DRAM**, 95–97% warp occupancy, **40 registers a thread**,
/// and shared bank conflicts under 1% of shared instructions. Every tile is
/// within 1% there, because the tile cannot move a kernel the memory bus is
/// holding. The dial only separates the shapes that fit L2; the sweep in
/// `tests/qsa_page_place_bench.rs` carries those numbers and the choice.
template <int TILE_R>
__global__ __launch_bounds__(QSA_PLACE_THREADS) void place_kernel(
    const long long* __restrict__ jobs,
    int d,
    int n_jobs
) {
    const int j = blockIdx.y;
    if (j >= n_jobs) return;

    const long long* job = jobs + (long long)j * QSA_PLACE_JOB_WORDS;
    const float4* src = (const float4*)(uintptr_t)job[0];
    float4* dst = (float4*)(uintptr_t)job[1];
    const int rows = (int)job[2];

    const int r0 = blockIdx.x * TILE_R;
    if (r0 >= rows) return;                      // a narrower job's spare tiles
    const int nr = min(TILE_R, rows - r0);

    const int D4 = d >> 2;
    const int S = D4 + PAD;                      // shared row pitch, in float4
    const int tid = threadIdx.x;

    extern __shared__ float4 sh[];

    // ---- load: coalesced along the channel axis -------------------------
    for (int i = tid; i < nr * D4; i += QSA_PLACE_THREADS) {
        const int r = i / D4;
        const int cg = i - r * D4;
        sh[r * S + cg] = __ldg(src + (long long)(r0 + r) * D4 + cg);
    }
    __syncthreads();

    // ---- store: coalesced along the ROW axis, which is the scorer's -----
    // `dst` is `[d/4, rows, 4]`, so channel group `cg` of row `r` sits at
    // `cg · rows + r` float4s: consecutive threads take consecutive rows and the
    // warp writes one contiguous run.
    for (int i = tid; i < D4 * nr; i += QSA_PLACE_THREADS) {
        const int cg = i / nr;
        const int r = i - cg * nr;
        dst[(long long)cg * rows + r0 + r] = sh[r * S + cg];
    }
}

} // namespace qsa_page_place

/// Launch at a given row tile. `max_rows` sizes the grid's row axis to the
/// widest job; narrower jobs retire their spare blocks immediately.
#define QSA_PLACE_LAUNCH(TR)                                                       \
    {                                                                              \
        const int D4 = d >> 2;                                                     \
        const size_t shmem = (size_t)(TR) * (D4 + qsa_page_place::PAD)             \
                                 * sizeof(float4);                                 \
        dim3 grid((unsigned)((max_rows + (TR) - 1) / (TR)), (unsigned)n_jobs);     \
        qsa_page_place::place_kernel<TR>                                           \
            <<<grid, QSA_PLACE_THREADS, shmem, (cudaStream_t)stream>>>(            \
                jobs, d, n_jobs);                                                  \
        return;                                                                    \
    }

/// Write every job's page in the scorer's channel-blocked layout.
///
/// `tile_r` selects the row tile; `0` takes the tuned default. The explicit
/// values exist for `tests/qsa_page_place_bench.rs`, which sweeps them — the
/// production caller passes `0` and gets whatever that sweep last settled on.
extern "C" void run_qsa_page_place(
    const long long* jobs,
    int32_t d,
    int32_t n_jobs,
    int32_t max_rows,
    int32_t tile_r,
    void* stream
) {
    if (n_jobs <= 0 || d <= 0 || max_rows <= 0) return;
    switch (tile_r) {
        case 8:  QSA_PLACE_LAUNCH(8)
        case 16: QSA_PLACE_LAUNCH(16)
        case 32: QSA_PLACE_LAUNCH(32)
        case 64: QSA_PLACE_LAUNCH(64)
        default: break;
    }
    // The default. See the bench's notes for the sweep that chose it.
    QSA_PLACE_LAUNCH(16)
}
