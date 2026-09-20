// =============================================================================
// QSA index scoring over a RAGGED, PAGED index, rotating each key on load
// =============================================================================
// The index's keys are stored UN-rotated: pooled and normed, carrying no
// position (`docs/progressive_yarn.md` §7). This kernel rotates each key as it
// reads it, at the key's own position and from the slot's factored RoPE table,
// then scores it — which is exactly what the reference does
// (`apply_qsa_rope` on the compressed keys at their first positions), and what
// lets a page sit anywhere and a slot change RoPE schedule without a byte of
// the index changing.
//
// The index is NOT one buffer. It is a list of pages — per-turn pieces, each
// separately allocated, ragged (a turn boundary does not land on a block
// boundary), plus the sequence's live tail — so the kernel takes a DESCRIPTOR
// TABLE and reads each page in place (hot-path invariant 2b):
//
//     page p:  { keys, cstride, rstride, delta }            (i64 each)
//
//   * `keys` addresses the page's `float4`s; channel group `c` of the page's
//     row `j` is at `keys + c·cstride + j·rstride`. A placed page is
//     channel-blocked (`cstride = rows`, `rstride = 1`), which makes a warp's
//     read contiguous; the live tail is row-major (`cstride = 1`,
//     `rstride = D/4`) because that is how the append writes it.
//   * `delta` places the page: global row `g` of page `p` sits at position
//     `delta + g · ratio`. It is `base − page_first[p] · ratio`, signed.
//
//     out[r, g] = Σ_h relu( q[r, h, :] · rope(key_g, pos(g)) )    for g <  cnt[r]
//               = -1e30                                            for g >= cnt[r]
//
// The queries arrive already rotated (`qsa_rope_rows.cu`), once per layer.
//
// **The mask is still a prefix, and that is what keeps the ragged case cheap.**
// Rows are ordered by token position, so "wholly below this query" remains
// `g < cnt[r]` however wide each row is — the caller folds the variable widths
// into `cnt` on the host. The `-1e30` matches `indexer_score_reduce`'s padding
// exactly: a masked column must lose to a genuinely negative score in the top-k
// that follows, which a 0 would not.
//
// ## Shape of the work
//
// One thread owns one candidate row and walks `D` with `float4` loads, holding
// `TILE_R × H` accumulators. The key is read ONCE, rotated ONCE, and reused
// across every query row in the block's tile: at the production geometry
// (`D = 128`, `H = 4`, `TILE_R = 4`) a thread reads 512 B of key, rotates its
// 64 rotary channels, and does 2,048 FMAs against it.
//
// The rotary pairs are NeoX half-split within the rotary width: pair `i` is
// channels `(i, i + pairs)`, which in `float4` groups is group `c = i/4` with
// its partner `c + R`, `R = pairs/4`. So the channel walk is two loops — the
// `R` rotary groups with their partners, then the pass-through groups `[2R, D/4)`
// unchanged.
//
// Candidates on the x axis and query rows on the y: at decode the row count is
// the number of sessions that selected this step — single digits — so tiling
// rows instead would leave the grid on a handful of SMs at exactly the depth
// where the candidate axis is tens of thousands of blocks wide.

#include <cuda_runtime.h>
#include <stdint.h>

#include "../rope/rope_table.cuh"

#define QSA_PAGED_THREADS 256

/// Blocks per SM the templated kernel is compiled to fit: 80 registers.
#define QSA_PAGED_MIN_BLOCKS 3

/// Rows of the query tile one block holds in shared memory. Four keeps the
/// tile at `4 × H × D` floats — 8 KB at the production geometry — which leaves
/// occupancy set by registers rather than by shared memory.
#define QSA_PAGED_TILE_R 4

/// i64 words per page descriptor: `{keys, cstride, rstride, delta}`.
#define QSA_PAGED_PAGE_WORDS 4

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

/// Where candidate `g` lives: its first `float4`, its channel-group stride, and
/// its position.
struct QsaCand {
    const float4* kb;
    long long cstride;
    int pos;
    int page;
};

__device__ __forceinline__ QsaCand qsa_resolve(
    const long long* __restrict__ pages,
    const unsigned int* __restrict__ page_first,
    int P, int g, int ratio)
{
    const int p = qsa_page_of(page_first, P, (unsigned int)g);
    const long long* pd = pages + (long long)p * QSA_PAGED_PAGE_WORDS;
    const float4* base = reinterpret_cast<const float4*>((uintptr_t)__ldg(pd));
    const long long cstride = __ldg(pd + 1);
    const long long rstride = __ldg(pd + 2);
    const long long delta = __ldg(pd + 3);
    const int local = g - (int)__ldg(page_first + p);
    QsaCand out;
    out.kb = base + (long long)local * rstride;
    out.cstride = cstride;
    out.pos = (int)(delta + (long long)g * ratio);
    out.page = p;
    return out;
}

/// `(sin, cos)` of `w + l`, from `(sin, cos)` of `w` and of `l`.
__device__ __forceinline__ float2 qsa_compose(float2 w, float2 l)
{
    float2 r;
    r.x = w.x * l.y + w.y * l.x;
    r.y = w.y * l.y - w.x * l.x;
    return r;
}

/// Full-warp mask. The candidate loop is warp-uniform, so every shuffle and
/// vote inside it has all 32 lanes present.
#define QSA_FULL 0xffffffffu

/// Page runs a warp shares warp terms across. A warp's 32 candidates are
/// consecutive rows, so they touch at most `⌈31 / rows⌉ + 1` pages: two once
/// pages hold 32 rows, four once they hold 11. Past four, each lane looks its
/// own rotation up.
#define QSA_RUNS 4

/// `H`, the row tile and the candidate tile as template parameters: the
/// accumulator array must be a compile-time size to live in registers rather
/// than local memory, and the inner loop wants its trip count known so the
/// `float4` body unrolls.
///
/// **`CPT` is what pays for the query tile.** A thread reads `TILE_R × H`
/// query `float4`s from shared per channel step, and every lane of the warp
/// reads the SAME address — a broadcast. Holding `CPT` candidates in flight
/// reuses each query `float4` across all of them, so the query traffic per unit
/// of arithmetic falls as `1/CPT` while the accumulator array grows as
/// `TILE_R × H × CPT`. That product is the register budget, and it is why the
/// deep-row arm takes a shallow candidate tile and vice versa.
///
/// The `CPT` candidates are `QSA_PAGED_THREADS` apart, not adjacent: adjacent
/// would put each thread's keys `D` floats apart and make the warp's load
/// scattered. Strided keeps consecutive lanes on consecutive candidates.
///
/// **Where a key's rotation comes from.** Lane `L` of a warp holds candidate
/// `G + L`, and a page's row `g` sits at `delta + g·ratio`, so every lane on
/// one page sits at `W + L·ratio` with `W = delta + G·ratio` shared by the
/// page's lanes. The rotation at each is the rotation at `W` composed with the
/// step table's entry at `L`. So each page run in the warp — at most
/// `QSA_RUNS` of them — has its `W` looked up once, one lane per pair, and left
/// in shared memory; every lane reads its own step entry from shared memory;
/// and no candidate reads the table from L2. A warp spanning more runs than
/// that reads each lane's rotation from the table.
///
/// **Three blocks per SM.** Left to itself the compiler spends 123 registers on
/// the `(4, 4, 2)` arm, which holds two blocks — 16 warps — while shared memory
/// has room for three; with the SM issuing every 1.8 cycles for want of an
/// eligible warp, the third block's warps are worth more than the registers.
template <int TILE_R, int H, int CPT>
__global__ __launch_bounds__(QSA_PAGED_THREADS, QSA_PAGED_MIN_BLOCKS) void qsa_score_paged_kernel(
    const float* __restrict__ q,                     // [rows*H, D], rotated
    const long long* __restrict__ pages,             // [P * PAGE_WORDS]
    const unsigned int* __restrict__ page_first,     // [P+1] exclusive prefix
    const unsigned int* __restrict__ cnt,            // [rows] valid prefix
    const float2* __restrict__ tab,                  // factored RoPE table
    const float2* __restrict__ steps,                // [pairs][32] step table
    float* __restrict__ out,                         // [rows, n_cand], `out_s` apart
    int rows, int D, int n_cand, int P, int pairs, int ratio,
    long long out_s, long long row_base)
{
    // The query tile, then the step table, then each warp's `W` rotations.
    extern __shared__ float sq[];                     // TILE_R * H * D
    float2* step_s = reinterpret_cast<float2*>(sq + TILE_R * H * D);   // [pairs][32]
    float2* warp_s = step_s + pairs * 32;             // [warps][CPT][RUNS][pairs]

    // **One thread owns all `TILE_R` rows.** Spreading the rows over `TILE_R`
    // threads shrinks the accumulator, and was measured slower (0.094 ms →
    // 0.115 at 128K and 64 rows): stacked, a key `float4` is loaded once and
    // serves every row from a register; spread, each thread sharing that
    // candidate loads it again. A broadcast from shared is far cheaper than a
    // redundant load from global, so the row axis belongs in registers.
    const int r0 = blockIdx.y * TILE_R;
    const int nr = min(TILE_R, rows - r0);
    if (nr <= 0) return;

    // The query tile, once per block. Consecutive threads take consecutive
    // floats, so the read coalesces across the whole tile.
    const int qelems = nr * H * D;
    const float* qsrc = q + (long long)(r0 * H) * D;
    for (int i = threadIdx.x; i < qelems; i += QSA_PAGED_THREADS) sq[i] = qsrc[i];
    for (int i = threadIdx.x; i < pairs * 32; i += QSA_PAGED_THREADS) step_s[i] = __ldg(steps + i);
    __syncthreads();

    const int D4 = D >> 2;
    const int R = pairs >> 2;                         // rotary float4 groups
    const int span = QSA_PAGED_THREADS * CPT;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // Warp-uniform: the loop runs on the warp's first candidate, so a warp's
    // lanes enter and leave it together and the shuffles inside are legal.
    for (int w0 = blockIdx.x * span + (threadIdx.x & ~31); w0 < n_cand;
         w0 += gridDim.x * span) {
        const int g0 = w0 + lane;
        // The previous pass's `W` rotations are read until every lane has left
        // its channel loops; this pass overwrites them.
        __syncwarp();
        // Resolve every candidate up front, so the channel loops hold only
        // pointers and the page searches overlap each other. A slot past the
        // end points at the last candidate's key instead of at null: the
        // epilogue drops its result, and a null test inside the channel loop
        // would put a branch between the loads and re-serialise them.
        const float4* kb[CPT];
        long long cs[CPT];
        int pj[CPT];
        int gj[CPT];
        // `run · 32 + (lane − leader)`: this lane's page run and its step from
        // the run's first lane, or -1 when the warp spans too many runs and
        // each lane looks its own rotation up.
        int run[CPT];
        #pragma unroll
        for (int j = 0; j < CPT; ++j) {
            const int g = g0 + j * QSA_PAGED_THREADS;
            gj[j] = g;
            const QsaCand cd = qsa_resolve(pages, page_first, P, g < n_cand ? g : n_cand - 1, ratio);
            kb[j] = cd.kb;
            cs[j] = cd.cstride;
            pj[j] = cd.pos;
            // Lanes on one page are a run, and `W` is the position of the
            // run's first lane — a real key's, so never negative, which
            // `delta + w0·ratio` would be for a page starting mid-warp. A lane
            // past the end resolved to the last candidate, so it joins that
            // page's run with a rotation that is wrong for it — harmless, as
            // its result is dropped — and is never a run's first lane.
            const unsigned same = __match_any_sync(QSA_FULL, cd.page);
            const int leader = __ffs(same) - 1;
            const unsigned leaders = __ballot_sync(QSA_FULL, lane == leader);
            const int nruns = __popc(leaders);
            if (nruns <= QSA_RUNS) {
                run[j] = (__popc(leaders & ((1u << leader) - 1u)) << 5) | (lane - leader);
                float2* wt = warp_s + ((warp * CPT + j) * QSA_RUNS) * pairs;
                unsigned rest = leaders;
                for (int k = 0; k < nruns; ++k) {
                    const int wk = __shfl_sync(QSA_FULL, cd.pos, __ffs(rest) - 1);
                    rest &= rest - 1u;
                    for (int i = lane; i < pairs; i += 32) {
                        wt[k * pairs + i] = rope_f_lookup(tab, pairs, wk, i);
                    }
                }
            } else {
                run[j] = -1;
            }
        }
        __syncwarp();

        float acc[TILE_R][H][CPT];
        #pragma unroll
        for (int r = 0; r < TILE_R; ++r)
            #pragma unroll
            for (int hh = 0; hh < H; ++hh)
                #pragma unroll
                for (int j = 0; j < CPT; ++j) acc[r][hh][j] = 0.f;

        // ---- rotary groups: each with its partner, rotated on load ----------
        for (int c = 0; c < R; ++c) {
            float4 kl[CPT], kh[CPT];
            #pragma unroll
            for (int j = 0; j < CPT; ++j) {
                kl[j] = __ldg(kb[j] + (long long)c * cs[j]);
                kh[j] = __ldg(kb[j] + (long long)(c + R) * cs[j]);
                const int i0 = c << 2;
                float2 sc[4];
                if (run[j] >= 0) {
                    const float2* wt =
                        warp_s + ((warp * CPT + j) * QSA_RUNS + (run[j] >> 5)) * pairs + i0;
                    const int l = run[j] & 31;
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                        sc[e] = qsa_compose(wt[e], step_s[(i0 + e) * 32 + l]);
                } else {
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                        sc[e] = rope_f_lookup(tab, pairs, pj[j], i0 + e);
                }
                rope_f_rotate(kl[j].x, kh[j].x, sc[0]);
                rope_f_rotate(kl[j].y, kh[j].y, sc[1]);
                rope_f_rotate(kl[j].z, kh[j].z, sc[2]);
                rope_f_rotate(kl[j].w, kh[j].w, sc[3]);
            }
            #pragma unroll
            for (int r = 0; r < TILE_R; ++r) {
                if (r >= nr) break;
                #pragma unroll
                for (int hh = 0; hh < H; ++hh) {
                    const float4 ql =
                        reinterpret_cast<const float4*>(sq + (r * H + hh) * D)[c];
                    const float4 qh =
                        reinterpret_cast<const float4*>(sq + (r * H + hh) * D)[c + R];
                    #pragma unroll
                    for (int j = 0; j < CPT; ++j) {
                        acc[r][hh][j] += ql.x * kl[j].x + ql.y * kl[j].y
                                       + ql.z * kl[j].z + ql.w * kl[j].w
                                       + qh.x * kh[j].x + qh.y * kh[j].y
                                       + qh.z * kh[j].z + qh.w * kh[j].w;
                    }
                }
            }
        }

        // ---- pass-through groups: read as stored ---------------------------
        //
        // **Summed into a temporary, deliberately.** This compiles to FMUL +
        // 3×FFMA + FADD, five instructions for four multiply-adds; accumulating
        // in place with four `fmaf`s is 4-for-4 and measured slower (0.096 ms →
        // 0.101 at 128K and 64 rows). The temporary is independent of `acc`, so
        // the accumulator's critical path is one FADD per channel group rather
        // than four FMAs.
        for (int c = 2 * R; c < D4; ++c) {
            float4 kv[CPT];
            #pragma unroll
            for (int j = 0; j < CPT; ++j) {
                kv[j] = __ldg(kb[j] + (long long)c * cs[j]);
            }
            #pragma unroll
            for (int r = 0; r < TILE_R; ++r) {
                if (r >= nr) break;
                #pragma unroll
                for (int hh = 0; hh < H; ++hh) {
                    const float4 qv =
                        reinterpret_cast<const float4*>(sq + (r * H + hh) * D)[c];
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

/// Generic arm: runtime `H`, and `D` or the rotary width not a multiple of the
/// `float4` group. Scalar loads, one accumulator per row. Correctness for the
/// oracle geometries the tests run (`head_dim` 16, odd head counts); the
/// templated kernel above is what production takes.
__global__ __launch_bounds__(QSA_PAGED_THREADS) void qsa_score_paged_generic_kernel(
    const float* __restrict__ q,
    const long long* __restrict__ pages,
    const unsigned int* __restrict__ page_first,
    const unsigned int* __restrict__ cnt,
    const float2* __restrict__ tab,
    float* __restrict__ out,
    int rows, int H, int D, int n_cand, int P, int pairs, int ratio,
    long long out_s, long long row_base)
{
    for (int g = blockIdx.x * QSA_PAGED_THREADS + threadIdx.x; g < n_cand;
         g += gridDim.x * QSA_PAGED_THREADS) {
        const QsaCand cd = qsa_resolve(pages, page_first, P, g, ratio);
        const float* kf = reinterpret_cast<const float*>(cd.kb);
        // Channel `c` of this key, un-rotated: group `c/4` at `cstride` apart.
        auto raw = [&](int c) { return kf[(long long)(c >> 2) * cd.cstride * 4 + (c & 3)]; };
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
                    float k;
                    if (c < pairs) {
                        float lo = raw(c), hi = raw(c + pairs);
                        rope_f_rotate(lo, hi, rope_f_lookup(tab, pairs, cd.pos, c));
                        k = lo;
                    } else if (c < 2 * pairs) {
                        float lo = raw(c - pairs), hi = raw(c);
                        rope_f_rotate(lo, hi, rope_f_lookup(tab, pairs, cd.pos, c - pairs));
                        k = hi;
                    } else {
                        k = raw(c);
                    }
                    d += qr[c] * k;
                }
                s += fmaxf(d, 0.f);
            }
            out[(row_base + r) * out_s + g] = s;
        }
    }
}

/// Candidates a block covers: every thread carries `CPT` of them.
#define QSA_PAGED_SPAN(TR, CP) (QSA_PAGED_THREADS * (CP))

/// Launch at a given tile pair.
///
/// **`TILE_R` is an L2 dial, not an occupancy one.** The grid is
/// `(n_cand / threads) × (rows / TILE_R)`, and every row-tile re-reads every key
/// it touches — so key traffic through L2 falls as `rows / TILE_R` while
/// register pressure rises as `TILE_R × H` accumulators.
#define QSA_PAGED_LAUNCH_T(HV, TR, CP)                                              \
    {                                                                               \
        const size_t shmem = (size_t)(TR) * (HV) * D * sizeof(float)                \
            + (size_t)pairs * 32 * sizeof(float2)                                   \
            + (size_t)(QSA_PAGED_THREADS / 32) * (CP) * QSA_RUNS * pairs            \
                  * sizeof(float2);                                                 \
        const int span = QSA_PAGED_SPAN(TR, CP);                                    \
        int bx = (n_cand + span - 1) / span;                                        \
        if (bx < 1) bx = 1;                                                         \
        if (bx > 65535) bx = 65535;                                                 \
        dim3 grid(bx, (rows + (TR) - 1) / (TR));                                     \
        qsa_score_paged_kernel<TR, HV, CP>                                          \
            <<<grid, QSA_PAGED_THREADS, shmem, stream>>>(                           \
                q, pages, page_first, cnt, tab, steps2, out, rows, D, n_cand, P,    \
                pairs, ratio, out_s, row_base);                                     \
        return;                                                                     \
    }

/// SM count of the current device, read once.
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

/// Pick the row tile and the candidate tile: searched from most reuse to least,
/// and the first whose grid fills the device wins. If none does — a shallow
/// index — the last arm is the one with the most blocks, because at that size
/// the kernel is latency-bound and parallelism is the only lever left.
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
    const long long* pages,
    const unsigned int* page_first,
    const unsigned int* cnt,
    const float* tab,
    const float* steps,
    float* out,
    int rows, int H, int D, int n_cand, int P, int pairs, int ratio,
    long long out_s, long long row_base,
    cudaStream_t stream)
{
    if (rows <= 0 || n_cand <= 0 || P <= 0) return;
    const float2* tab2 = reinterpret_cast<const float2*>(tab);
    const float2* steps2 = reinterpret_cast<const float2*>(steps);
    // The templated arms need whole `float4` groups on both sides of the
    // rotary split; anything else takes the generic arm.
    if ((D & 3) == 0 && (pairs & 3) == 0 && 2 * pairs <= D) {
        const float2* tab = tab2;
        switch (H) {
            QSA_PAGED_LAUNCH(1)
            QSA_PAGED_LAUNCH(2)
            QSA_PAGED_LAUNCH(4)
            QSA_PAGED_LAUNCH(8)
            default: break;
        }
    }
    const int want = (n_cand + QSA_PAGED_THREADS - 1) / QSA_PAGED_THREADS;
    const int blocks_x = want < 1 ? 1 : (want > 65535 ? 65535 : want);
    qsa_score_paged_generic_kernel<<<blocks_x, QSA_PAGED_THREADS, 0, stream>>>(
        q, pages, page_first, cnt, tab2, out, rows, H, D, n_cand, P, pairs, ratio,
        out_s, row_base);
}
