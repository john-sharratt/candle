// =============================================================================
// QSA index scoring over a RAGGED, PAGED index
// =============================================================================
// The dense scorer takes one contiguous `[n_blocks, D]` key buffer and hands it
// to cuBLAS as the transposed right operand. That shape is unavailable once a
// sequence's index is reconstructed from per-turn pieces: the pieces are
// separately allocated, they arrive in whatever order the turns were sealed, and
// the last row of each piece covers FEWER than `ratio` tokens because a turn
// boundary does not land on a block boundary.
//
// So this kernel takes a DESCRIPTOR TABLE instead of a base pointer — one
// `{keys, first_row}` pair per page — and reads each page in place
// (hot-path invariant 2b). Nothing is concatenated, which is the whole point:
// materialising the window would copy every key of every turn on the step that
// reconstructs, and the pieces are exactly the buffers the turn records already
// hold.
//
//     out[r, g] = Σ_h relu( q[r, h, :] · key_g[:] )        for g <  cnt[r]
//               = -1e30                                    for g >= cnt[r]
//
// `g` is a GLOBAL row index across the concatenated pages; `page_first` is the
// exclusive prefix sum of page row counts, so `page_first[p] <= g <
// page_first[p+1]` names the page and `g - page_first[p]` the row inside it.
//
// **The mask is still a prefix, and that is what keeps the ragged case cheap.**
// Rows are ordered by token position, so "wholly below this query" remains
// `g < cnt[r]` however wide each row is — the caller folds the variable widths
// into `cnt` on the host, where the page table already lives, and the kernel
// never needs a per-row width or position. The `-1e30` matches
// `indexer_score_reduce`'s padding exactly: a masked column must lose to a
// genuinely negative score in the top-k that follows, which a 0 would not.
//
// ## Shape of the work
//
// One thread owns one candidate row and walks `D` with `float4` loads, holding
// `TILE_R × H` accumulators. The key is read ONCE and reused across every query
// row in the block's tile, which is where the arithmetic intensity comes from:
// at the production geometry (`D = 128`, `H = 4`, `TILE_R = 4`) a thread reads
// 512 B of key and does 2,048 FMAs against it.
//
// Candidates on the x axis and query rows on the y: at decode the row count is
// the number of sessions that selected this step — single digits — so tiling
// rows instead would leave the grid on a handful of SMs at exactly the depth
// where the candidate axis is tens of thousands of blocks wide.
//
// The page search is a binary search per thread over `page_first`, which is
// `log2(P)` `__ldg`s of a small array every thread in the launch reads — L2-hot
// after the first block. A block spans 256 consecutive candidates and a page is
// a turn, so in practice every thread in a block resolves to the same one or
// two pages.

#include <cuda_runtime.h>

#define QSA_PAGED_THREADS 256

/// Rows of the query tile one block holds in shared memory. Four keeps the
/// tile at `4 × H × D` floats — 8 KB at the production geometry — which leaves
/// occupancy set by registers rather than by shared memory.
#define QSA_PAGED_TILE_R 4

/// Masked-column score. Bit-identical to `indexer_score_reduce`'s padding.
#define QSA_PAGED_MASK (-1e30f)

/// Page holding `g`, by binary search over the exclusive prefix sum.
///
/// `page_first` has `P + 1` entries and is strictly ascending, so the loop
/// closes on the unique `p` with `page_first[p] <= g < page_first[p+1]`.
__device__ __forceinline__ int qsa_page_of(
    const unsigned int* __restrict__ page_first, int P, unsigned int g)
{
    int lo = 0, hi = P;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (__ldg(page_first + mid) <= g) lo = mid; else hi = mid;
    }
    return lo;
}

/// `H`, the row tile and the candidate tile as template parameters: the
/// accumulator array must be a compile-time size to live in registers rather
/// than local memory, and the inner loop wants its trip count known so the
/// `float4` body unrolls.
///
/// **`CPT` is what pays for the query tile.** A thread reads `TILE_R × H`
/// query `float4`s from shared per channel step, and every lane of the warp
/// reads the SAME address — a broadcast, so each one is a single L1 wavefront
/// rather than 32. Cheap per lane, but still one wavefront per instruction, and
/// at `CPT = 1` there are 32 of them against 128 FMAs. Holding `CPT` candidates
/// in flight reuses each query `float4` across all of them, so the query traffic
/// per unit of arithmetic falls as `1/CPT` while the accumulator array grows as
/// `TILE_R × H × CPT`. That product is the register budget, and it is why the
/// deep-row arm takes a shallow candidate tile and vice versa.
///
/// The `CPT` candidates are `QSA_PAGED_THREADS` apart, not adjacent: adjacent
/// would put each thread's four keys 4·D floats apart and make the warp's load
/// four times as scattered. Strided keeps consecutive lanes on consecutive
/// candidates, which is the least-bad pattern available for a key buffer stored
/// candidate-major.
/// **No `minBlocksPerMultiprocessor` floor, deliberately.** At `(4, 2)` nvcc
/// picks 64 registers on its own — exactly four blocks per SM, and 64 registers
/// admits only 1024 of an SM's 1536 threads, so occupancy stops at 66.7%.
/// Asking for a fifth block caps the budget at 51 registers, which this tile's
/// live values do not fit: it spilled, and measured 0.094 ms → 0.152 at 128K
/// and 64 rows. The compiler's own choice was already the right one.
template <int TILE_R, int H, int CPT>
__global__ __launch_bounds__(QSA_PAGED_THREADS) void qsa_score_paged_kernel(
    const float* __restrict__ q,                     // [rows*H, D] contiguous
    const unsigned long long* __restrict__ page_keys, // [P] device addresses
    const unsigned int* __restrict__ page_first,      // [P+1] exclusive prefix
    const unsigned int* __restrict__ cnt,             // [rows] valid prefix
    float* __restrict__ out,                          // [rows, n_cand], `out_s` apart
    int rows, int D, int n_cand, int P,
    long long out_s, long long row_base)
{
    extern __shared__ float sq[];                     // TILE_R * H * D

    // **One thread owns all `TILE_R` rows, and that is the cheaper arrangement
    // even though it is the register-hungrier one.** Spreading the rows over
    // `TILE_R` threads of the block shrinks the accumulator from
    // `TILE_R × H × CPT` to `H × CPT` — 64 registers down to about 40, which
    // would lift occupancy from 66.7% to 100% — and it was tried. It measured
    // **slower**, 0.094 ms → 0.115 at 128K and 64 rows.
    //
    // The reason is what the row axis is actually reusing. Stacked, a key
    // `float4` is loaded once and serves every row from a register. Spread, the
    // `TILE_R` threads sharing that candidate each load it, so the coalesced
    // global read happens `TILE_R` times and only L1 absorbs the difference:
    // per warp per channel step the cost goes from `4·CPT + TILE_R·H` wavefronts
    // for `TILE_R·H·CPT·4` FMAs to `4·CPT + H` for `H·CPT·4` — twice as many
    // wavefronts per unit of arithmetic. A broadcast from shared is far cheaper
    // than a redundant load from global, so the row axis belongs in registers.
    const int r0 = blockIdx.y * TILE_R;
    const int nr = min(TILE_R, rows - r0);
    if (nr <= 0) return;

    // The query tile, once per block. Consecutive threads take consecutive
    // floats, so the read coalesces across the whole tile.
    const int qelems = nr * H * D;
    const float* qsrc = q + (long long)(r0 * H) * D;
    for (int i = threadIdx.x; i < qelems; i += QSA_PAGED_THREADS) sq[i] = qsrc[i];
    __syncthreads();

    const int D4 = D >> 2;
    const int span = QSA_PAGED_THREADS * CPT;

    for (int g0 = blockIdx.x * span + threadIdx.x; g0 < n_cand; g0 += gridDim.x * span) {
        // Resolve every candidate's page up front, so the channel loop holds
        // only pointers and the binary searches overlap each other.
        //
        // A page is stored `[D/4, rows_p, 4]`, so candidate `g`'s channel group
        // `c` sits at `(c · rows_p + local) · 4` floats: consecutive lanes are
        // consecutive `local`, hence consecutive `float4`s, hence one coalesced
        // 512-byte read per warp per step.
        // `g0` is inside the range by the loop condition, so it is always a
        // valid candidate. A tail slot points at `g0`'s key instead of at null:
        // the epilogue drops its result anyway, and a null test inside the
        // channel loop would put a branch between the loads and re-serialise
        // them — the loop below is unrolled precisely so several key loads are
        // in flight at once, and a branch in the middle costs exactly that.
        const float4* kb[CPT];
        int pitch[CPT];
        int gj[CPT];
        #pragma unroll
        for (int j = 0; j < CPT; ++j) {
            const int g = g0 + j * QSA_PAGED_THREADS;
            gj[j] = g;
            const int gv = g < n_cand ? g : g0;
            const int p = qsa_page_of(page_first, P, (unsigned int)gv);
            const unsigned int first = __ldg(page_first + p);
            pitch[j] = (int)(__ldg(page_first + p + 1) - first);
            kb[j] = reinterpret_cast<const float4*>((const float*)__ldg(page_keys + p))
                + (gv - (int)first);
        }

        float acc[TILE_R][H][CPT];
        #pragma unroll
        for (int r = 0; r < TILE_R; ++r)
            #pragma unroll
            for (int hh = 0; hh < H; ++hh)
                #pragma unroll
                for (int j = 0; j < CPT; ++j) acc[r][hh][j] = 0.f;

        // One pass over the keys, every query row of the tile riding on every
        // candidate of the tile.
        //
        // **Where the latency has to be hidden decides the unroll.** `D` is a
        // runtime argument, so an un-unrolled loop issues one key load, waits
        // on it, and does `TILE_R × H × CPT × 4` FMAs. A tiled arm covers that
        // wait with other warps and with its own `CPT` loads already in flight,
        // and unrolling it only holds more `float4`s live — measured, unrolling
        // the `(4, 4)` arm cost 0.101 ms → 0.121 at 128K and 64 rows, purely in
        // occupancy.
        //
        // The `(1, 1)` arm is different in kind: it is chosen *because* the grid
        // could not be filled, so there are no other warps and the latency has
        // to be hidden inside the thread. Measured at 8K depth and one query
        // row, that arm was 19.2 µs of kernel time for 1 MFMA of arithmetic.
        constexpr int UNROLL = (TILE_R == 1 && CPT == 1) ? 4 : 1;
        #pragma unroll UNROLL
        for (int c = 0; c < D4; ++c) {
            float4 kv[CPT];
            #pragma unroll
            for (int j = 0; j < CPT; ++j) {
                kv[j] = __ldg(kb[j] + (long long)c * pitch[j]);
            }
            #pragma unroll
            for (int r = 0; r < TILE_R; ++r) {
                if (r >= nr) break;
                #pragma unroll
                for (int hh = 0; hh < H; ++hh) {
                    const float4 qv =
                        reinterpret_cast<const float4*>(sq + (r * H + hh) * D)[c];
                    // **Summed into a temporary, deliberately, and it is not the
                    // instruction-cheapest form.** This compiles to FMUL +
                    // 3×FFMA + FADD: five math instructions for four
                    // multiply-adds, and the measured FFMA count is exactly
                    // 0.75× the arithmetic these shapes call for, which is that
                    // ratio. Accumulating in place with four `fmaf`s instead is
                    // 4-for-4 and was tried — it measured **slower**, 0.096 ms →
                    // 0.101 at 128K and 64 rows.
                    //
                    // The reason is the dependency chain, not the instruction
                    // count. In this form the temporary is independent of `acc`,
                    // so the accumulator's critical path is one FADD per channel
                    // group — 32 links. Accumulating in place puts all four FMAs
                    // on that path and makes it 128 links, and with 3.6 active
                    // warps per scheduler there is not enough other work to
                    // cover it. Instruction count is the cheaper thing to spend
                    // here.
                    #pragma unroll
                    for (int j = 0; j < CPT; ++j) {
                        acc[r][hh][j] += qv.x * kv[j].x + qv.y * kv[j].y
                                       + qv.z * kv[j].z + qv.w * kv[j].w;
                    }
                }
            }
        }

        #pragma unroll
        for (int r = 0; r < TILE_R; ++r) {
            if (r >= nr) break;
            const unsigned int valid = __ldg(cnt + r0 + r);
            #pragma unroll
            for (int j = 0; j < CPT; ++j) {
                if (gj[j] >= n_cand) continue;
                float s = QSA_PAGED_MASK;
                if ((unsigned int)gj[j] < valid) {
                    s = 0.f;
                    #pragma unroll
                    for (int hh = 0; hh < H; ++hh) s += fmaxf(acc[r][hh][j], 0.f);
                }
                out[(row_base + r0 + r) * out_s + gj[j]] = s;
            }
        }
    }
}

/// Generic arm: runtime `H`, and `D` not a multiple of four. One accumulator
/// per row held in a small local array, scalar loads. Correctness for the
/// oracle geometries the tests run (`head_dim` 16, odd head counts); the
/// templated kernel above is what production takes.
__global__ __launch_bounds__(QSA_PAGED_THREADS) void qsa_score_paged_generic_kernel(
    const float* __restrict__ q,
    const unsigned long long* __restrict__ page_keys,
    const unsigned int* __restrict__ page_first,
    const unsigned int* __restrict__ cnt,
    float* __restrict__ out,
    int rows, int H, int D, int n_cand, int P,
    long long out_s, long long row_base)
{
    for (int g = blockIdx.x * QSA_PAGED_THREADS + threadIdx.x; g < n_cand;
         g += gridDim.x * QSA_PAGED_THREADS) {
        const int p = qsa_page_of(page_first, P, (unsigned int)g);
        const unsigned int first = __ldg(page_first + p);
        const int pitch = (int)(__ldg(page_first + p + 1) - first);
        // Same `[D/4, rows_p, 4]` staging as the templated arm; `D` need not be
        // a multiple of four here, so the channel index is unpacked by hand.
        const float* kb = (const float*)__ldg(page_keys + p) + (long long)(g - (int)first) * 4;
        for (int r = 0; r < rows; ++r) {
            const unsigned int valid = __ldg(cnt + r);
            if ((unsigned int)g >= valid) {
                out[(row_base + r) * out_s + g] = QSA_PAGED_MASK;
                continue;
            }
            float s = 0.f;
            for (int hh = 0; hh < H; ++hh) {
                const float* qr = q + ((long long)r * H + hh) * D;
                float d = 0.f;
                for (int c = 0; c < D; ++c) {
                    d += qr[c] * kb[(long long)(c >> 2) * pitch * 4 + (c & 3)];
                }
                s += fmaxf(d, 0.f);
            }
            out[(row_base + r) * out_s + g] = s;
        }
    }
}

/// Launch at a given row-tile depth.
///
/// **`TILE_R` is an L2 dial, not an occupancy one.** The grid is
/// `(n_cand / threads) × (rows / TILE_R)`, and every row-tile re-reads every key
/// it touches — so key traffic through L2 falls as `rows / TILE_R` while
/// register pressure rises as `TILE_R × H` accumulators. Measured at 128K depth
/// and 64 query rows, `TILE_R = 4` put L2 at 87.8% of peak against 40% compute:
/// the kernel was reading the same 16 MiB of keys sixteen times.
/// Candidates a block covers: every thread carries `CPT` of them.
#define QSA_PAGED_SPAN(TR, CP) (QSA_PAGED_THREADS * (CP))

#define QSA_PAGED_LAUNCH_T(HV, TR, CP)                                              \
    {                                                                               \
        const size_t shmem = (size_t)(TR) * (HV) * D * sizeof(float);               \
        const int span = QSA_PAGED_SPAN(TR, CP);                                    \
        int bx = (n_cand + span - 1) / span;                                        \
        if (bx < 1) bx = 1;                                                         \
        if (bx > 65535) bx = 65535;                                                 \
        dim3 grid(bx, (rows + (TR) - 1) / (TR));                                     \
        qsa_score_paged_kernel<TR, HV, CP>                                          \
            <<<grid, QSA_PAGED_THREADS, shmem, stream>>>(                           \
                q, page_keys, page_first, cnt, out, rows, D, n_cand, P, out_s,      \
                row_base);                                                          \
        return;                                                                     \
    }

/// SM count of the current device, read once.
///
/// The tile choice below is a grid-occupancy decision, so it needs the size of
/// the machine it is filling. A function-local static is initialised exactly
/// once and thread-safely under C++11, and the attribute query is a driver
/// lookup rather than a full `cudaGetDeviceProperties`.
static int qsa_paged_sm_count()
{
    static int sm = 0;
    if (sm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        int v = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess
            || v <= 0) {
            v = 64;                                   // a plausible floor, not a guess at this part
        }
        sm = v;
    }
    return sm;
}

/// Blocks the grid would hold at a given tile pair.
#define QSA_PAGED_GRID(TR, CP)                                                      \
    ((long long)((rows + (TR) - 1) / (TR))                                          \
     * (long long)((n_cand + QSA_PAGED_SPAN(TR, CP) - 1) / QSA_PAGED_SPAN(TR, CP)))

/// Pick the row tile and the candidate tile.
///
/// **Two dials pulling opposite ways, and the shape decides which one is
/// starved.** Both tiles buy reuse — a deeper row tile divides the key traffic
/// through L2, a deeper candidate tile divides the query traffic through L1 —
/// and the wavefronts-per-FMA cost is `(CPT + TILE_R) / (4 · TILE_R · CPT · H)`,
/// which for a fixed accumulator budget `TILE_R × H × CPT` is smallest when the
/// two are equal. But every unit of tile is a unit the grid does not get: the
/// grid is `⌈rows/TILE_R⌉ × ⌈n_cand/(threads · CPT)⌉`, so the tile that reads
/// the least memory can also leave two thirds of the device idle.
///
/// Measured, at 128K depth and 256 pages: at 64 query rows `(4, 4)` runs 0.101 ms
/// and `(4, 1)` runs 0.113 — reuse wins, because there are 512 blocks either way.
/// At 8 rows the same pair inverts, 0.035 against 0.031, for one reason: `(4, 4)`
/// leaves a grid of 64 blocks on a 110-SM part. Neither tile is right; the
/// **rule** is right.
///
/// So the arms are searched from most reuse to least, and the first one whose
/// grid fills the device wins. If none does — a shallow index, where there is
/// simply not enough work to go round — the last arm is the one with the most
/// blocks, because at that size the kernel is latency-bound and parallelism is
/// the only lever left.
#define QSA_PAGED_LAUNCH(HV)                                                        \
    case HV: {                                                                      \
        const long long fill = 4LL * qsa_paged_sm_count();                          \
        if (rows >= 4 && QSA_PAGED_GRID(4, 2) >= fill) QSA_PAGED_LAUNCH_T(HV, 4, 2) \
        if (rows >= 2 && QSA_PAGED_GRID(2, 2) >= fill) QSA_PAGED_LAUNCH_T(HV, 2, 2) \
        if (rows >= 2 && QSA_PAGED_GRID(2, 1) >= fill) QSA_PAGED_LAUNCH_T(HV, 2, 1) \
        QSA_PAGED_LAUNCH_T(HV, 1, 1)                                                \
    }

extern "C" void run_qsa_score_paged(
    const float* q,
    const unsigned long long* page_keys,
    const unsigned int* page_first,
    const unsigned int* cnt,
    float* out,
    int rows, int H, int D, int n_cand, int P,
    long long out_s, long long row_base,
    cudaStream_t stream)
{
    if (rows <= 0 || n_cand <= 0 || P <= 0) return;
    // Enough blocks to fill the device on the candidate axis; the grid-stride
    // loop absorbs whatever is left. The templated arms size their own grid
    // from their candidate tile; this is the generic arm's.
    const int want = (n_cand + QSA_PAGED_THREADS - 1) / QSA_PAGED_THREADS;
    const int blocks_x = want < 1 ? 1 : (want > 65535 ? 65535 : want);

    if ((D & 3) == 0) {
        switch (H) {
            QSA_PAGED_LAUNCH(1)
            QSA_PAGED_LAUNCH(2)
            QSA_PAGED_LAUNCH(4)
            QSA_PAGED_LAUNCH(8)
            default: break;
        }
    }
    qsa_score_paged_generic_kernel<<<blocks_x, QSA_PAGED_THREADS, 0, stream>>>(
        q, page_keys, page_first, cnt, out, rows, H, D, n_cand, P, out_s, row_base);
}
