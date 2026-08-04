// Tensor-core Binary Directional Provenance (BDP) scan — b1 BMMA backend.
//
// The scalar scan (`bdp_scan.cu`) spends its cycles hand-broadcasting query
// words to popcount pipelines (~64 issue slots per gallery token). This backend
// feeds the SAME computation to the 1-bit tensor cores instead: the
// `BMMA.88128.XOR.POPC` instruction computes an 8x8 tile of
// `popcount(query XOR token)` over 128-bit chunks in one warp op, and
// `agreement = 512 - xor_popc` — an exact integer transform, so the two
// backends' integer statistics (per-case max, sum, sumsq) are IDENTICAL and the
// final votes go through the shared `bdp_vote` finalize.
//
// Structure (two kernels, one launcher):
//  * `bdp_bmma_accum_kernel` — grid (token-chunk, group, query-tile); each CTA
//    stages a 64-token chunk + 32-query tile in shared (K-chunk-major so every
//    wmma fragment pointer is 32-byte aligned, ldm = 128 bits), runs the b1
//    BMMA tiles, and reduces per-(query, case) max / sum / sumsq in shared
//    before flushing once to global accumulators. Cases are DENSE-RANKED within
//    the chunk (gallery cases are non-decreasing over the scan order — the
//    index builder sorts each segment's windows by case), so the shared
//    accumulators stay bounded even when case ids have gaps.
//  * `bdp_bmma_finalize_kernel` — one thread per (query, group, segment): scans
//    the segment's case range for the leader/runner-up and emits the same
//    `(out_case, z*margin)` pairs as the scalar kernel via `bdp_vote`.
//
// The flattened chunk grid also removes the scalar kernel's two structural
// limits: no per-segment block tail (a 3-token file no longer occupies a whole
// CTA) and no shared-memory dependence on the largest segment's case count.
//
// Hardware gate: b1 BMMA exists on sm_75..sm_89 (Turing/Ampere/Ada) and was
// dropped on Hopper/Blackwell. Device code compiles to a stub outside that
// range and the launcher reports "unsupported" at runtime so the caller falls
// back to the scalar kernel.

#include <cstdint>
#include <cuda_runtime.h>

#include "bdp_vote.cuh"

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 900)
#define BDP_BMMA_DEVICE_OK 1
#include <mma.h>
using namespace nvcuda;
#else
#define BDP_BMMA_DEVICE_OK 0
#endif

// Gallery tokens per CTA chunk (8 BMMA tiles of 8 tokens).
#define BMMA_TC 64
// Query tokens per CTA tile (4 BMMA rows of 8 queries).
#define BMMA_QT 32
// The locked folded group width this backend requires: 8 u64 = 512 bits =
// 4 K-chunks of 128. The launcher rejects any other geometry.
#define BMMA_GW 8
// Tokens per arena page (mirrors the arena's PAGE_TOKENS / scalar kernel).
#define BMMA_PAGE_TOKENS 32
// Sentinel rank for out-of-range (padding) tokens.
#define BMMA_RANK_INVALID 0xFFFFu
// Rank slots held in shared per pass. A 64-token chunk usually spans 1-3 cases;
// capping the accumulators at 32 slots halves their footprint (12 KB), which
// doubles resident CTAs per SM — the kernel is occupancy-bound. The rare chunk
// with more distinct cases (all-tiny exchanges) reruns its tiles once per
// 32-rank window; the recompute is confined to those chunks.
#define BMMA_RANK_CAP 32

extern "C" __global__ void __launch_bounds__(256) bdp_bmma_accum_kernel(
    const unsigned int *__restrict__ gallery_case,   // n_tokens GLOBAL case ids
    const unsigned long long *__restrict__ probe_words, // token-major n_probe * wpt
    const unsigned long long *__restrict__ page_ptr, // page device addresses
    const unsigned int *__restrict__ pos_map,        // n_tokens (page<<5)|in_pg
    int n_tokens,
    int n_probe_tokens,
    int n_groups,
    int n_cases,
    int wpt,
    unsigned int *__restrict__ case_max,             // n_probe * n_groups * n_cases
    unsigned long long *__restrict__ case_sum,      // n_probe * n_groups * n_cases
    unsigned long long *__restrict__ case_sumsq)     // n_probe * n_groups * n_cases
{
#if BDP_BMMA_DEVICE_OK && defined(__CUDA_ARCH__)
    const int j0 = blockIdx.x * BMMA_TC;
    const int g = blockIdx.y;
    if (j0 >= n_tokens) {
        return;
    }

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // K-chunk-major staging: chunk c's rows are contiguous 128-bit records, so
    // every fragment pointer is 32-byte aligned and ldm is exactly 128 bits.
    __shared__ __align__(32) unsigned long long sh_a[4][BMMA_QT * 2];
    __shared__ __align__(32) unsigned long long sh_b[4][BMMA_TC * 2];
    __shared__ __align__(32) int sh_acc[8][64]; // per-warp 8x8 tile staging
    __shared__ unsigned int sh_case[BMMA_TC];
    __shared__ unsigned short sh_rank[BMMA_TC];
    __shared__ unsigned int sh_rank_case[BMMA_TC];
    __shared__ unsigned int sh_chg[2]; // per-warp case-change ballot masks
    __shared__ int sh_nranks;
    // Per-(query, windowed rank) accumulators for one query tile over this
    // chunk. sum <= 64*512 and sumsq <= 64*512^2 = 16.8M, both comfortably u32.
    __shared__ unsigned int sh_max[BMMA_QT * BMMA_RANK_CAP];
    __shared__ unsigned int sh_sum[BMMA_QT * BMMA_RANK_CAP];
    __shared__ unsigned int sh_sq[BMMA_QT * BMMA_RANK_CAP];

    const size_t group_page_off = (size_t)g * BMMA_PAGE_TOKENS * BMMA_GW;

    // Stage the gallery chunk ONCE — it is reused across every query tile below.
    // Out-of-range tokens stage zero words (their tiles compute, the epilogue
    // skips them).
    for (int idx = tid; idx < BMMA_TC * BMMA_GW; idx += blockDim.x) {
        const int t = idx >> 3;
        const int k = idx & 7;
        const int j = j0 + t;
        unsigned long long w = 0ull;
        if (j < n_tokens) {
            const unsigned int pm = pos_map[j];
            const unsigned long long *tok =
                (const unsigned long long *)page_ptr[pm >> 5] + group_page_off +
                (size_t)(pm & (BMMA_PAGE_TOKENS - 1)) * BMMA_GW;
            w = tok[k];
        }
        sh_b[k >> 1][t * 2 + (k & 1)] = w;
    }
    // Stage per-token global case ids (sentinel for the padded tail).
    for (int t = tid; t < BMMA_TC; t += blockDim.x) {
        const int j = j0 + t;
        sh_case[t] = (j < n_tokens) ? gallery_case[j] : 0xFFFFFFFFu;
    }
    __syncthreads();

    // Dense-rank the chunk's cases in parallel. Case ids are non-decreasing
    // over the scan order (the index builder sorts each segment's windows by
    // case, and ids are globally cumulative across segments), so a token's rank
    // is the prefix-popcount of case-change flags — two warp ballots. Padding
    // is always a suffix, so it never breaks the prefix.
    if (tid < BMMA_TC) {
        const unsigned int c = sh_case[tid];
        const bool valid = (c != 0xFFFFFFFFu);
        const bool changed = valid && (tid == 0 || c != sh_case[tid - 1]);
        const unsigned int m = __ballot_sync(0xFFFFFFFFu, changed);
        if ((tid & 31) == 0) {
            sh_chg[tid >> 5] = m;
        }
    }
    __syncthreads();
    if (tid < BMMA_TC) {
        const unsigned int c = sh_case[tid];
        if (c == 0xFFFFFFFFu) {
            sh_rank[tid] = BMMA_RANK_INVALID;
        } else {
            int r;
            if (tid < 32) {
                r = (int)__popc(sh_chg[0] & (0xFFFFFFFFu >> (31 - tid))) - 1;
            } else {
                r = (int)__popc(sh_chg[0]) +
                    (int)__popc(sh_chg[1] & (0xFFFFFFFFu >> (63 - tid))) - 1;
            }
            sh_rank[tid] = (unsigned short)r;
            // The first token of each run records the rank -> case mapping.
            if (tid == 0 || c != sh_case[tid - 1]) {
                sh_rank_case[r] = c;
            }
        }
        if (tid == 0) {
            sh_nranks = __popc(sh_chg[0]) + __popc(sh_chg[1]);
        }
    }
    __syncthreads();
    const int nranks = sh_nranks;
    if (nranks == 0) {
        return; // whole chunk past the token end (empty launch tail)
    }

    // Warp assignment: 8 warps = 4 query-blocks x 2 token-halves.
    const int qb = warp & 3;  // query block: rows [qb*8, qb*8+8)
    const int th = warp >> 2; // token half: tiles [th*4, th*4+4)

    // The warp's gallery fragments are invariant across every query tile —
    // preload all 4 tiles x 4 K-chunks into registers (16 regs) so the query
    // loop below never re-reads the staged gallery from shared.
    wmma::fragment<wmma::matrix_b, 8, 8, 128, wmma::experimental::precision::b1,
                   wmma::col_major>
        b_frag[4][4];
#pragma unroll
    for (int tile = 0; tile < 4; tile++) {
        const int t0 = th * 32 + tile * 8;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            wmma::load_matrix_sync(b_frag[tile][c], &sh_b[c][t0 * 2], 128);
        }
    }

    // Loop the query tiles INSIDE the CTA: the staged gallery chunk, its ranks,
    // and the B fragments amortize across all tiles (staging traffic and launch
    // count drop by the tile count vs a query-tile grid dimension).
    const int n_qtiles = (n_probe_tokens + BMMA_QT - 1) / BMMA_QT;
    for (int qt = 0; qt < n_qtiles; qt++) {
        const int q0 = qt * BMMA_QT;

        // Stage this query tile (zero rows past the probe end — flush skips
        // them).
        for (int idx = tid; idx < BMMA_QT * BMMA_GW; idx += blockDim.x) {
            const int qq = idx >> 3;
            const int k = idx & 7;
            const int q = q0 + qq;
            unsigned long long w = 0ull;
            if (q < n_probe_tokens) {
                w = probe_words[(size_t)q * wpt + (size_t)g * BMMA_GW + k];
            }
            sh_a[k >> 1][qq * 2 + (k & 1)] = w;
        }
        __syncthreads();

        // The warp's query fragments — one per K-chunk, live across the rank
        // windows below.
        wmma::fragment<wmma::matrix_a, 8, 8, 128,
                       wmma::experimental::precision::b1, wmma::row_major>
            a_frag[4];
#pragma unroll
        for (int c = 0; c < 4; c++) {
            wmma::load_matrix_sync(a_frag[c], &sh_a[c][qb * 8 * 2], 128);
        }

        // Rank windows: almost always a single pass (nranks <= 32); a chunk of
        // all-tiny exchanges reruns its tiles once per window.
        for (int rw = 0; rw < nranks; rw += BMMA_RANK_CAP) {
            const int wcount =
                (nranks - rw) < BMMA_RANK_CAP ? (nranks - rw) : BMMA_RANK_CAP;
            for (int idx = tid; idx < BMMA_QT * wcount; idx += blockDim.x) {
                const int base = (idx / wcount) * BMMA_RANK_CAP + (idx % wcount);
                sh_max[base] = 0u;
                sh_sum[base] = 0u;
                sh_sq[base] = 0u;
            }
            __syncthreads();

#pragma unroll
            for (int tile = 0; tile < 4; tile++) {
                const int t0 = th * 32 + tile * 8;
                wmma::fragment<wmma::accumulator, 8, 8, 128, int> acc;
                wmma::fill_fragment(acc, 0);
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    wmma::bmma_sync(acc, a_frag[c], b_frag[tile][c], acc,
                                    wmma::experimental::bmmaBitOpXOR,
                                    wmma::experimental::bmmaAccumulateOpPOPC);
                }
                wmma::store_matrix_sync(sh_acc[warp], acc, 8,
                                        wmma::mem_row_major);
                __syncwarp();
                // Run-merged epilogue: 16 lanes each own half a query row (4
                // token columns) and merge consecutive same-rank columns in
                // REGISTERS before touching shared — one atomic triple per
                // (row-half, rank run), not per element. Cases arrive in runs
                // (tokens are case-sorted), so a tile is typically one run:
                // this removes the 8-way same-address serialization a
                // per-element epilogue suffers. Merging max / sum / sumsq over
                // integers is exact.
                if (lane < 16) {
                    const int row = lane >> 1;
                    const int c0col = (lane & 1) * 4;
                    const int base_q = (qb * 8 + row) * BMMA_RANK_CAP - rw;
                    int run_rank = -1;
                    unsigned int rmax = 0u, rsum = 0u, rsq = 0u;
#pragma unroll
                    for (int jc = c0col; jc < c0col + 4; jc++) {
                        const unsigned short rk = sh_rank[t0 + jc];
                        // Skip padding and ranks outside this window.
                        if ((unsigned int)((int)rk - rw) >= (unsigned int)wcount) {
                            continue;
                        }
                        const unsigned int ag =
                            512u - (unsigned int)sh_acc[warp][row * 8 + jc];
                        if ((int)rk != run_rank) {
                            if (run_rank >= 0) {
                                atomicMax(&sh_max[base_q + run_rank], rmax);
                                atomicAdd(&sh_sum[base_q + run_rank], rsum);
                                atomicAdd(&sh_sq[base_q + run_rank], rsq);
                            }
                            run_rank = (int)rk;
                            rmax = ag;
                            rsum = ag;
                            rsq = ag * ag;
                        } else {
                            rmax = rmax > ag ? rmax : ag;
                            rsum += ag;
                            rsq += ag * ag;
                        }
                    }
                    if (run_rank >= 0) {
                        atomicMax(&sh_max[base_q + run_rank], rmax);
                        atomicAdd(&sh_sum[base_q + run_rank], rsum);
                        atomicAdd(&sh_sq[base_q + run_rank], rsq);
                    }
                }
                __syncwarp();
            }
            __syncthreads();

            // Flush this window's per-(query, rank) statistics to the global
            // per-case accumulators — one atomic triple per pair, skipping
            // padded queries — then rejoin before the next zero/restage.
            for (int idx = tid; idx < BMMA_QT * wcount; idx += blockDim.x) {
                const int qq = idx / wcount;
                const int r = idx % wcount;
                const int q = q0 + qq;
                if (q >= n_probe_tokens) {
                    continue;
                }
                const unsigned int c = sh_rank_case[rw + r];
                const size_t gbase = ((size_t)q * n_groups + g) * n_cases + c;
                const int sbase = qq * BMMA_RANK_CAP + r;
                atomicMax(&case_max[gbase], sh_max[sbase]);
                atomicAdd(&case_sum[gbase], (unsigned long long)sh_sum[sbase]);
                atomicAdd(&case_sumsq[gbase], (unsigned long long)sh_sq[sbase]);
            }
            __syncthreads();
        }
    }
#else
    // Stub on architectures without b1 BMMA — never launched (the launcher
    // reports unsupported at runtime).
    (void)gallery_case; (void)probe_words; (void)page_ptr; (void)pos_map;
    (void)n_tokens; (void)n_probe_tokens; (void)n_groups; (void)n_cases;
    (void)wpt; (void)case_max; (void)case_sum; (void)case_sumsq;
#endif
}

// One thread per (query, group, segment): leader/runner-up over the segment's
// case range -> the same (out_case, z*margin) pairs as the scalar kernel.
extern "C" __global__ void bdp_bmma_finalize_kernel(
    const unsigned int *__restrict__ case_max,
    const unsigned long long *__restrict__ case_sum,
    const unsigned long long *__restrict__ case_sumsq,
    const int *__restrict__ seg_tok_start,  // n_segments+1
    const int *__restrict__ seg_case_start, // n_segments+1
    int n_probe_tokens,
    int n_groups,
    int n_segments,
    int n_cases,
    int *__restrict__ out_case,   // n_probe * n_groups * n_segments
    float *__restrict__ out_vote) // n_probe * n_groups * n_segments
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_probe_tokens * n_groups * n_segments;
    if (idx >= total) {
        return;
    }
    // idx decomposes exactly as the output layout (q * n_groups + g) *
    // n_segments + s, so idx IS the output index.
    const int s = idx % n_segments;
    const int rem = idx / n_segments;
    const int g = rem % n_groups;
    const int q = rem / n_groups;

    const int n_gal = seg_tok_start[s + 1] - seg_tok_start[s];
    const int c0 = seg_case_start[s];
    const int c1 = seg_case_start[s + 1];
    if (c1 <= c0 || n_gal <= 0) {
        out_case[idx] = -1;
        out_vote[idx] = 0.0f;
        return;
    }

    // Leader/runner-up over the segment's cases, ascending with strict
    // comparisons — identical tie behaviour to the scalar kernel's top-2 loop.
    unsigned int top1 = 0u, top2 = 0u;
    int top1c = -1;
    unsigned long long sum = 0ull;
    unsigned long long sumsq = 0ull;
    const size_t base = ((size_t)q * n_groups + g) * n_cases;
    for (int c = c0; c < c1; c++) {
        const unsigned int m = case_max[base + c];
        sum += case_sum[base + c];
        sumsq += case_sumsq[base + c];
        if (m > top1) {
            top2 = top1;
            top1 = m;
            top1c = c;
        } else if (m > top2) {
            top2 = m;
        }
    }
    if (top1c < 0) {
        out_case[idx] = -1;
        out_vote[idx] = 0.0f;
    } else {
        out_case[idx] = top1c; // already a GLOBAL case id
        out_vote[idx] = bdp_vote(top1, top2, sum, sumsq, n_gal);
    }
}

// Whether this backend can run on the current device: the hardware must have
// b1 BMMA tensor cores (sm_75..sm_89 — Turing/Ampere/Ada; dropped on
// Hopper/Blackwell) AND the embedded fatbin must hold a loadable kernel image
// for it — SASS is minor-version-specific and the build targets a fixed arch
// set, so the capability range alone over-promises (e.g. an sm_86 card has the
// hardware but no image). `cudaFuncGetAttributes` fails exactly when no image
// loads for the current device. The Rust side falls down its backend ladder
// when this reports 0.
// Failure stages encoded into the launcher's negative return code as
// `-(stage * 1000 + cudaError)`. A bare `-(int)err` named the error but not
// WHERE it came from, so an `InvalidConfiguration` could equally be a bad grid,
// a shared-memory overflow, or a stale error from unrelated work.
#define BDP_BMMA_STAGE_ALLOC 1
#define BDP_BMMA_STAGE_MEMSET 2
#define BDP_BMMA_STAGE_ACCUM 3
#define BDP_BMMA_STAGE_FINALIZE 4

extern "C" int bdp_bmma_supported() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        return 0;
    }
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) !=
            cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) !=
            cudaSuccess) {
        return 0;
    }
    const int cc = major * 10 + minor;
    if (cc < 75 || cc >= 90) {
        return 0;
    }
    cudaFuncAttributes attr;
    if (cudaFuncGetAttributes(&attr, (const void *)bdp_bmma_accum_kernel) !=
        cudaSuccess) {
        (void)cudaGetLastError(); // clear the sticky probe error
        return 0;
    }
    return 1;
}

// Host launcher: allocates the per-scan global accumulators stream-ordered,
// zeroes them, runs accumulate + finalize, and frees them. Returns 0 on
// success, 1 when this device/geometry cannot run the BMMA path (caller falls
// back to the scalar kernel), or a negative cudaError code.
extern "C" int run_bmma_bdp_scan(
    const unsigned int *gallery_case,
    const unsigned long long *probe_words,
    const int *seg_tok_start,
    const int *seg_case_start,
    const unsigned long long *page_ptr,
    const unsigned int *pos_map,
    int n_tokens,
    int n_probe_tokens,
    int n_groups,
    int n_segments,
    int n_cases,
    int gw,
    int wpt,
    int *out_case,
    float *out_vote,
    void *stream)
{
    if (!bdp_bmma_supported() || gw != BMMA_GW || page_ptr == nullptr ||
        pos_map == nullptr) {
        return 1;
    }
    if (n_tokens <= 0 || n_probe_tokens <= 0 || n_groups <= 0 ||
        n_segments <= 0 || n_cases <= 0) {
        return 1;
    }
    cudaStream_t s = (cudaStream_t)stream;
    // Discard any error left pending on this thread by unrelated earlier work.
    // Without this the `cudaGetLastError()` calls below can pick up someone
    // else's failure and report it as ours — a launch that never actually
    // failed then looks like a BMMA rejection.
    (void)cudaGetLastError();
    const size_t nc = (size_t)n_probe_tokens * n_groups * n_cases;

    unsigned int *d_max = nullptr;
    unsigned long long *d_sum = nullptr;
    unsigned long long *d_sq = nullptr;
    // Which step failed, for the stage-encoded return code (see the header
    // comment on BDP_BMMA_STAGE_*). "insufficient" told us nothing about which
    // of six CUDA calls actually rejected the work.
    int stage = BDP_BMMA_STAGE_ALLOC;
    cudaError_t err = cudaMallocAsync((void **)&d_max, nc * sizeof(unsigned int), s);
    if (err == cudaSuccess) {
        err = cudaMallocAsync((void **)&d_sum, nc * sizeof(unsigned long long), s);
    }
    if (err == cudaSuccess) {
        err = cudaMallocAsync((void **)&d_sq, nc * sizeof(unsigned long long), s);
    }
    if (err == cudaSuccess) {
        stage = BDP_BMMA_STAGE_MEMSET;
        err = cudaMemsetAsync(d_max, 0, nc * sizeof(unsigned int), s);
    }
    if (err == cudaSuccess) {
        err = cudaMemsetAsync(d_sum, 0, nc * sizeof(unsigned long long), s);
    }
    if (err == cudaSuccess) {
        err = cudaMemsetAsync(d_sq, 0, nc * sizeof(unsigned long long), s);
    }
    if (err == cudaSuccess) {
        stage = BDP_BMMA_STAGE_ACCUM;
        const dim3 grid((n_tokens + BMMA_TC - 1) / BMMA_TC, n_groups);
        bdp_bmma_accum_kernel<<<grid, 256, 0, s>>>(
            gallery_case, probe_words, page_ptr, pos_map, n_tokens,
            n_probe_tokens, n_groups, n_cases, wpt, d_max, d_sum, d_sq);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        stage = BDP_BMMA_STAGE_FINALIZE;
        const int total = n_probe_tokens * n_groups * n_segments;
        const int blocks = (total + 255) / 256;
        bdp_bmma_finalize_kernel<<<blocks, 256, 0, s>>>(
            d_max, d_sum, d_sq, seg_tok_start, seg_case_start, n_probe_tokens,
            n_groups, n_segments, n_cases, out_case, out_vote);
        err = cudaGetLastError();
    }
    // Stream-ordered frees run after the kernels complete; free whatever was
    // allocated even on the error path.
    if (d_max != nullptr) {
        cudaFreeAsync(d_max, s);
    }
    if (d_sum != nullptr) {
        cudaFreeAsync(d_sum, s);
    }
    if (d_sq != nullptr) {
        cudaFreeAsync(d_sq, s);
    }
    return (err == cudaSuccess) ? 0 : -(stage * 1000 + (int)err);
}
