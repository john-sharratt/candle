// =============================================================================
// QSA index page PLACEMENT: rotate to the placement frame + channel-block
// =============================================================================
// An index page is a position-free artifact: its rows are the pooled, normed,
// roped block keys of one sealed unit, in that unit's own frame. Placing it in a
// projection gives it a position, and RoPE rotations compose additively — so the
// page is brought into the placement's frame by ONE further rotation through the
// difference between the two, applied once when the page is placed rather than
// once per score.
//
// That is the whole reason this kernel is separate from the scorer. The paged
// scorer's economy is that a key `float4` is read once and reused across
// `TILE_R × H` query accumulators; rotating there would add a `rope_dim/2`
// cos/sin row per CANDIDATE against a 512-byte key, on the traffic that is
// already its bottleneck. Here the rotation is a constant over the whole page,
// so it costs one pass over a buffer the placement had to build anyway.
//
// ---- What it replaces -------------------------------------------------------
//
// The scorer reads keys channel-blocked as `[d/4, rows, 4]`, which makes a
// warp's key read 512 contiguous bytes instead of 32 scattered ones. That
// staging used to be `reshape → transpose → contiguous` — three tensor ops and
// an allocate-plus-copy (hot-path invariant 2) that could not carry a rotation.
// One kernel does both: the transpose is the pass the placement pays anyway, and
// the rotation rides inside it.
//
// ---- Shape of the work ------------------------------------------------------
//
// Pure bandwidth: `rows · d` floats in, the same out, with `rope_dim` channels of
// each row touched by two multiplies and an add. One side of a transpose is
// always strided, so a block stages a `TILE_R × d` tile in shared memory —
// coalesced `float4` reads in, coalesced `float4` writes out, and the strided
// axis paid in shared where it is cheap.
//
//   load    consecutive threads take consecutive channel groups of one row
//   rotate  scalar over the rotary width only — `[rope_dim, d)` is untouched
//   store   consecutive threads take consecutive ROWS of one channel group,
//           which is exactly the `[d/4, rows, 4]` order the scorer reads
//
// The rotation is scalar rather than `float4` deliberately: pairing channel `c`
// with `c + rope_dim/2` is only a `float4`-group pairing when `rope_dim/2` is a
// multiple of four, and the oracle geometries the tests run are not. It touches
// `rope_dim` of `d` channels once, against two full `float4` passes over all of
// them — the vector win is on the axis that carries the bytes.
//
// ---- Batching ---------------------------------------------------------------
//
// A placement rebuilds every page of every attention layer, so the launch is
// batched over a descriptor table (hot-path invariant 5): one job per
// (page, layer), each with its own rows and its own delta. `gridDim.y` indexes
// the job and `gridDim.x` the row tile, sized to the widest job; a job with
// fewer tiles retires its extra blocks in the first instruction.
//
// Table layout: array-of-structs, `QSA_PLACE_JOB_WORDS` i64 per job.
//
//     [0] src    const float*  the page's `[rows, d]` prepared keys
//     [1] dst    float*        this placement's `[d/4, rows, 4]` staging
//     [2] rows
//     [3] delta  the rotation, `placement_base − roped_base`, in tokens —
//                SIGNED, because a seal rotates a page back to zero to make it
//                position-free while a placement rotates it forward to a
//                position. The tables are indexed by |delta| and a rotation
//                through a negative angle is the same cosine with the sine
//                negated, so both directions are one expression.

#include <cuda_runtime.h>
#include <stdint.h>

#define QSA_PLACE_JOB_WORDS 4
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
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    int d,
    int rope_dim,
    int n_jobs
) {
    const int j = blockIdx.y;
    if (j >= n_jobs) return;

    const long long* job = jobs + (long long)j * QSA_PLACE_JOB_WORDS;
    const float4* src = (const float4*)(uintptr_t)job[0];
    float4* dst = (float4*)(uintptr_t)job[1];
    const int rows = (int)job[2];
    const int delta = (int)job[3];

    const int r0 = blockIdx.x * TILE_R;
    if (r0 >= rows) return;                      // a narrower job's spare tiles
    const int nr = min(TILE_R, rows - r0);

    const int D4 = d >> 2;
    const int S = D4 + PAD;                      // shared row pitch, in float4
    const int half = rope_dim >> 1;
    const int tid = threadIdx.x;

    // `[TILE_R · S] float4` of tile, then the delta's cos and sin rows. Staging
    // the two rows once per block is what keeps the rotation off global memory:
    // every row of the tile turns through the SAME angle, so `half` floats
    // apiece serve the whole tile.
    extern __shared__ float4 sh[];
    float* shf = (float*)sh;                     // the tile, addressed by channel
    float* co_row = (float*)(sh + TILE_R * S);
    float* si_row = co_row + half;

    // `cos(-x) = cos(x)` and `sin(-x) = -sin(x)`, so one table row serves both
    // directions and the sign rides on the staged sine.
    const int mag = delta < 0 ? -delta : delta;
    const float sgn = delta < 0 ? -1.0f : 1.0f;
    for (int i = tid; i < half; i += QSA_PLACE_THREADS) {
        co_row[i] = __ldg(cos_tab + (long long)mag * half + i);
        si_row[i] = sgn * __ldg(sin_tab + (long long)mag * half + i);
    }

    // ---- load: coalesced along the channel axis -------------------------
    for (int i = tid; i < nr * D4; i += QSA_PLACE_THREADS) {
        const int r = i / D4;
        const int cg = i - r * D4;
        sh[r * S + cg] = __ldg(src + (long long)(r0 + r) * D4 + cg);
    }
    __syncthreads();

    // ---- rotate: NeoX half-split, rotary width only ---------------------
    // Channel `c < half` pairs with `c + half`; `[rope_dim, d)` passes through
    // untouched, which is why the loop is bounded by `half` and not by `d`.
    //
    // **No barrier inside the loop, and none is needed.** Item `(r, c)` owns the
    // pair `(c, c + half)` of row `r` outright — no other item reads or writes
    // either element — so the in-place update carries no cross-thread hazard and
    // needs no second buffer. Which is the only reason it can be a grid-stride
    // loop at all: a `__syncthreads()` inside one is undefined, because threads
    // do not all run the same number of iterations.
    for (int i = tid; i < nr * half; i += QSA_PLACE_THREADS) {
        const int r = i / half;
        const int c = i - r * half;
        float* row = shf + (long long)r * S * 4;
        const float lo = row[c];
        const float hi = row[c + half];
        const float co = co_row[c];
        const float si = si_row[c];
        row[c] = lo * co - hi * si;
        row[c + half] = hi * co + lo * si;
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

/// Launch at a given row tile. `max_tiles` sizes the grid's row axis to the
/// widest job; narrower jobs retire their spare blocks immediately.
#define QSA_PLACE_LAUNCH(TR)                                                       \
    {                                                                              \
        const int D4 = d >> 2;                                                     \
        const size_t shmem = (size_t)(TR) * (D4 + qsa_page_place::PAD)             \
                                 * sizeof(float4)                                  \
                             + (size_t)rope_dim * sizeof(float);                   \
        dim3 grid((unsigned)((max_rows + (TR) - 1) / (TR)), (unsigned)n_jobs);     \
        qsa_page_place::place_kernel<TR>                                           \
            <<<grid, QSA_PLACE_THREADS, shmem, (cudaStream_t)stream>>>(            \
                jobs, cos_tab, sin_tab, d, rope_dim, n_jobs);                      \
        return;                                                                    \
    }

/// Rotate every job's page into its placement frame and write the scorer's
/// channel-blocked staging.
///
/// `tile_r` selects the row tile; `0` takes the tuned default. The explicit
/// values exist for `tests/qsa_page_place_bench.rs`, which sweeps them — the
/// production caller passes `0` and gets whatever that sweep last settled on.
extern "C" void run_qsa_page_place(
    const long long* jobs,
    const float* cos_tab,
    const float* sin_tab,
    int32_t d,
    int32_t rope_dim,
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
