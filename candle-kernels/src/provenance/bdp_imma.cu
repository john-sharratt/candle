// INT8 tensor-core Binary Directional Provenance (BDP) scan — IMMA backend.
//
// The Blackwell-portable sibling of the b1 backend (`bdp_bmma.cu`): Hopper and
// Blackwell dropped the 1-bit BMMA path, but INT8 MMA (`mma.m16n8k32.s8`) is a
// mainstream tensor format on every arch from Ampere up — including sm_120. The
// sign bits stay PACKED in the arena exactly as for the other backends, and
// are expanded into 0/1 s8 fragment lanes on chip, each exactly once: a warp's
// gallery fragments into registers once per chunk, held across every query
// tile, and a query tile's fragments into shared once per block, in the image
// each lane loads with one 16-byte read. Expanding both inside the MMA loop
// redid the gallery's per query tile and the query's per warp, and that ALU
// work — not the MMA — was the kernel. With both operands 0/1-encoded the MMA
// accumulates m11 = popcount(q AND t), which relates to the agreement by an
// exact integer identity using the per-row popcounts staged alongside the bits:
//
//   agreement = popc(XNOR(q,t)) = 512 - popc(q) - popc(t) + 2*m11
//
// (0/1 beats a +/-1 encoding by two ops per fragment — measured ~25% off the
// whole kernel.) The per-case integer statistics are IDENTICAL to both other
// backends, and the shared finalize (`bdp_bmma_finalize_kernel` +
// `bdp_vote.cuh`) emits bit-matching votes.
//
// Uses inline-PTX `mma` (not wmma): the m16n8k32 fragment<->thread mappings are
// documented in the PTX ISA, which is what makes building the fragments from
// the packed bits legitimate. A warp computes a 16x8 (query x token) tile per 16 K-steps; the
// accumulator lives entirely in registers, so the epilogue run-merges straight
// from registers into the shared per-(query, rank) accumulators — no staging
// buffer at all. Everything else (chunk grid, ballot dense-ranking, rank
// windows, global accumulators, finalize) mirrors `bdp_bmma.cu`.

#include <cstdint>
#include <cuda_runtime.h>

#include "bdp_chunk.cuh"
#include "bdp_vote.cuh"
#include "../grow_scratch.cuh"

// m16n8k32.s8 requires sm_80+ and — the point of this backend — has no upper
// arch bound: the sm_120 (Blackwell) compilation emits the real kernel.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
#define BDP_IMMA_DEVICE_OK 1
#else
#define BDP_IMMA_DEVICE_OK 0
#endif

// Gallery tokens per CTA chunk (4 warp token-quarters of 16 = 2 MMA n-tiles).
#define IMMA_TC 64
// Query tokens per CTA tile (2 MMA m-blocks of 16).
#define IMMA_QT 32
// The locked folded group width this backend requires: 8 u64 = 512 bits =
// 16 K-steps of 32. The launcher rejects any other geometry.
#define IMMA_GW 8
// Tokens per arena page (mirrors the arena's PAGE_TOKENS).
#define IMMA_PAGE_TOKENS 32
// Sentinel rank for out-of-range (padding) tokens.
#define IMMA_RANK_INVALID 0xFFFFu
// Rank slots held in shared per pass (see bdp_bmma.cu — same windowing).
#define IMMA_RANK_CAP 32
// Row stride of the shared per-(query, rank) accumulators: one past the rank
// cap, so the rows a warp touches at once — eight distinct rows at the same
// rank in the one-case epilogue, 32 rows at rank 0 when zeroing and flushing a
// one-rank window — fall in different banks. At a stride of 32 every one of
// them hit the same bank.
#define IMMA_ACC_STRIDE (IMMA_RANK_CAP + 1)
// Padded u64 words per staged bit-record: 8 data + 1 pad so consecutive rows
// land in different shared banks for the fragment-expansion reads.
#define IMMA_ROW_WORDS 9

#if BDP_IMMA_DEVICE_OK && defined(__CUDA_ARCH__)
// Expand 4 consecutive sign bits into 4 s8 lanes of 0/1 (bit set -> 1). Byte i
// of the result is bit i — the LSB-first lane order of the .b32 fragment
// registers. Both operands use the 0/1 encoding, so the MMA accumulates
// m11 = popcount(q AND t); with the per-row popcounts staged alongside the
// bits, agreement = 512 - popc(q) - popc(t) + 2*m11 — exact integers, and two
// ops cheaper per fragment than a +/-1 expansion.
__device__ __forceinline__ unsigned int bdp_expand4(unsigned int nib) {
    return (nib * 0x00204081u) & 0x01010101u;
}
#endif

extern "C" __global__ void __launch_bounds__(256) bdp_imma_accum_kernel(
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
#if BDP_IMMA_DEVICE_OK && defined(__CUDA_ARCH__)
    const int j0 = blockIdx.x * IMMA_TC;
    const int g = blockIdx.y;
    if (j0 >= n_tokens) {
        return;
    }

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // Packed BIT records (not expanded bytes): row-padded so the 8 fragment
    // rows a warp reads per expansion hit distinct banks.
    __shared__ unsigned long long sh_a[IMMA_QT * IMMA_ROW_WORDS];
    __shared__ unsigned long long sh_b[IMMA_TC * IMMA_ROW_WORDS];
    // The query tile's A fragments, expanded: [query block][K-step][lane], each
    // the four .b32 registers that lane hands the MMA.
    __shared__ uint4 sh_afrag[2 * 16 * 32];
    // Per-row popcounts backing the 0/1-encoding agreement identity.
    __shared__ unsigned int sh_pq[IMMA_QT];
    __shared__ unsigned int sh_pt[IMMA_TC];
    __shared__ unsigned int sh_case[IMMA_TC];
    __shared__ unsigned short sh_rank[IMMA_TC];
    __shared__ unsigned int sh_rank_case[IMMA_TC];
    __shared__ unsigned int sh_chg[2]; // per-warp case-change ballot masks
    __shared__ int sh_nranks;
    // Per-(query, windowed rank) accumulators — same widths/rationale as the
    // b1 backend (sum <= 64*512, sumsq <= 64*512^2, both u32 per chunk).
    __shared__ unsigned int sh_max[IMMA_QT * IMMA_ACC_STRIDE];
    __shared__ unsigned int sh_sum[IMMA_QT * IMMA_ACC_STRIDE];
    __shared__ unsigned int sh_sq[IMMA_QT * IMMA_ACC_STRIDE];

    const size_t group_page_off = (size_t)g * IMMA_PAGE_TOKENS * IMMA_GW;

    // Stage the gallery chunk's bit records ONCE (reused by every query tile).
    for (int idx = tid; idx < IMMA_TC * IMMA_GW; idx += blockDim.x) {
        const int t = idx >> 3;
        const int k = idx & 7;
        const int j = j0 + t;
        unsigned long long w = 0ull;
        if (j < n_tokens) {
            const unsigned int pm = pos_map[j];
            const unsigned long long *tok =
                (const unsigned long long *)page_ptr[pm >> 5] + group_page_off +
                (size_t)(pm & (IMMA_PAGE_TOKENS - 1)) * IMMA_GW;
            w = tok[k];
        }
        sh_b[t * IMMA_ROW_WORDS + k] = w;
    }
    // Stage per-token global case ids (sentinel for the padded tail).
    for (int t = tid; t < IMMA_TC; t += blockDim.x) {
        const int j = j0 + t;
        sh_case[t] = (j < n_tokens) ? gallery_case[j] : 0xFFFFFFFFu;
    }
    __syncthreads();

    // Per-token popcounts (zero for the padded tail's zero words).
    for (int t = tid; t < IMMA_TC; t += blockDim.x) {
        unsigned int p = 0u;
#pragma unroll
        for (int k = 0; k < IMMA_GW; k++) {
            p += (unsigned int)__popcll(sh_b[t * IMMA_ROW_WORDS + k]);
        }
        sh_pt[t] = p;
    }

    // Parallel dense-rank (identical to the b1 backend): case ids are
    // non-decreasing over the scan order, rank = prefix-popcount of changes.
    if (tid < IMMA_TC) {
        const unsigned int c = sh_case[tid];
        const bool valid = (c != 0xFFFFFFFFu);
        const bool changed = valid && (tid == 0 || c != sh_case[tid - 1]);
        const unsigned int m = __ballot_sync(0xFFFFFFFFu, changed);
        if ((tid & 31) == 0) {
            sh_chg[tid >> 5] = m;
        }
    }
    __syncthreads();
    if (tid < IMMA_TC) {
        const unsigned int c = sh_case[tid];
        if (c == 0xFFFFFFFFu) {
            sh_rank[tid] = IMMA_RANK_INVALID;
        } else {
            int r;
            if (tid < 32) {
                r = (int)__popc(sh_chg[0] & (0xFFFFFFFFu >> (31 - tid))) - 1;
            } else {
                r = (int)__popc(sh_chg[0]) +
                    (int)__popc(sh_chg[1] & (0xFFFFFFFFu >> (63 - tid))) - 1;
            }
            sh_rank[tid] = (unsigned short)r;
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

    // Warp assignment: 8 warps = 2 query m-blocks x 4 token quarters; a quarter
    // is 16 tokens = 2 MMA n-tiles of 8.
    const int qb = warp & 1;  // query block: rows [qb*16, qb*16+16)
    const int tq = warp >> 1; // token quarter: tokens [tq*16, tq*16+16)
    const int grp = lane >> 2;   // fragment group id (0..7)
    const int tig = lane & 3;    // thread-in-group (0..3)

    // This warp's gallery (B) fragments, expanded ONCE and held in registers
    // for every query tile. They depend only on the chunk, so expanding them
    // inside the query loop redid the same work once per tile. Documented
    // m16n8k32.s8 B layout (col-major 32x8): b0=(klo, grp) b1=(khi, grp), with
    // klo = tig*4..+4 and khi = klo+16 — both carved from word ks/2.
    unsigned int bfrag[2][16][2];
#pragma unroll
    for (int tile = 0; tile < 2; tile++) {
        const unsigned long long *brow = &sh_b[(tq * 16 + tile * 8 + grp) * IMMA_ROW_WORDS];
#pragma unroll
        for (int ks = 0; ks < 16; ks++) {
            const unsigned int et =
                (unsigned int)(brow[ks >> 1] >> ((ks & 1) * 32 + tig * 4));
            bfrag[tile][ks][0] = bdp_expand4(et & 0xFu);
            bfrag[tile][ks][1] = bdp_expand4((et >> 16) & 0xFu);
        }
    }

    const int n_qtiles = (n_probe_tokens + IMMA_QT - 1) / IMMA_QT;
    for (int qt = 0; qt < n_qtiles; qt++) {
        const int q0 = qt * IMMA_QT;

        // Stage this query tile's bit records (zero rows past the probe end —
        // the flush skips them).
        for (int idx = tid; idx < IMMA_QT * IMMA_GW; idx += blockDim.x) {
            const int qq = idx >> 3;
            const int k = idx & 7;
            const int q = q0 + qq;
            unsigned long long w = 0ull;
            if (q < n_probe_tokens) {
                w = probe_words[(size_t)q * wpt + (size_t)g * IMMA_GW + k];
            }
            sh_a[qq * IMMA_ROW_WORDS + k] = w;
        }
        __syncthreads();
        // Per-query popcounts for this tile (zero rows -> zero).
        for (int qq = tid; qq < IMMA_QT; qq += blockDim.x) {
            unsigned int p = 0u;
#pragma unroll
            for (int k = 0; k < IMMA_GW; k++) {
                p += (unsigned int)__popcll(sh_a[qq * IMMA_ROW_WORDS + k]);
            }
            sh_pq[qq] = p;
        }
        // The tile's query (A) fragments, expanded once by the whole block into
        // the exact register image each lane loads — one 16-byte shared read per
        // K-step. Every token-quarter warp reads the same rows, so expanding in
        // each warp did the same work four times over. Documented m16n8k32.s8 A
        // layout (row-major 16x32): a0=(grp, klo) a1=(grp+8, klo) a2=(grp, khi)
        // a3=(grp+8, khi).
        for (int e = tid; e < 2 * 16 * 32; e += blockDim.x) {
            const int eb = e >> 9;          // query block
            const int ks = (e >> 5) & 15;   // K-step
            const int el = e & 31;          // lane
            const int eg = el >> 2;
            const int et = el & 3;
            const unsigned long long *r0 = &sh_a[(eb * 16 + eg) * IMMA_ROW_WORDS];
            const unsigned long long *r1 = r0 + 8 * IMMA_ROW_WORDS;
            const int sh = (ks & 1) * 32 + et * 4;
            const unsigned int ea = (unsigned int)(r0[ks >> 1] >> sh);
            const unsigned int ec = (unsigned int)(r1[ks >> 1] >> sh);
            sh_afrag[e] = make_uint4(bdp_expand4(ea & 0xFu), bdp_expand4(ec & 0xFu),
                                     bdp_expand4((ea >> 16) & 0xFu),
                                     bdp_expand4((ec >> 16) & 0xFu));
        }

        // Rank windows: almost always one pass (nranks <= 32); a chunk of
        // all-tiny exchanges reruns its tiles once per window.
        for (int rw = 0; rw < nranks; rw += IMMA_RANK_CAP) {
            const int wcount =
                (nranks - rw) < IMMA_RANK_CAP ? (nranks - rw) : IMMA_RANK_CAP;
            for (int idx = tid; idx < IMMA_QT * wcount; idx += blockDim.x) {
                const int base =
                    (idx / wcount) * IMMA_ACC_STRIDE + (idx % wcount);
                sh_max[base] = 0u;
                sh_sum[base] = 0u;
                sh_sq[base] = 0u;
            }
            __syncthreads();

            // The warp's two 16x8 tiles (documented m16n8 C layout: c[0]=(grp,
            // tig*2), c[1]=(grp, tig*2+1), c[2]/c[3] = rows +8). Both operands
            // arrive expanded: A as one 16-byte shared read of the image the
            // block built for this tile, B from the registers this warp filled
            // for the chunk.
            int acc[2][4];
            const uint4 *afrag = &sh_afrag[qb * 16 * 32 + lane];
#pragma unroll
            for (int tile = 0; tile < 2; tile++) {
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                for (int ks = 0; ks < 16; ks++) {
                    const uint4 a = afrag[ks * 32];
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(bfrag[tile][ks][0]),
                          "r"(bfrag[tile][ks][1]));
                }
                acc[tile][0] = c0;
                acc[tile][1] = c1;
                acc[tile][2] = c2;
                acc[tile][3] = c3;
            }

            // With the 0/1 encoding the MMA accumulated m11 = popc(q AND t), so
            // agreement = 512 - popc(q) - popc(t) + 2*m11 — exact.
            const unsigned int pq0 = sh_pq[qb * 16 + grp];
            const unsigned int pq1 = sh_pq[qb * 16 + grp + 8];

            // **One case under the whole quarter — the common case.** A case is
            // an exchange, thousands of tokens long, so the 16 tokens a warp
            // holds almost always share one rank. Then each row's statistics
            // reduce in registers — the lane's four agreements, then the row's
            // four lanes by shuffle — and one lane per row updates the shared
            // accumulators: three atomics per row instead of three per lane per
            // tile, all on distinct rows. The per-lane shared atomics, four
            // lanes to an address, were what held the kernel: L1 77.8% busy at
            // 21% compute (ncu, sm_120). Max and integer sums are exact in any
            // order, so the statistics are the per-token path's bit for bit.
            const unsigned short rq = sh_rank[tq * 16];
            const bool one_case = __all_sync(0xFFFFFFFFu, sh_rank[tq * 16 + (lane & 15)] == rq) &&
                                  (unsigned int)((int)rq - rw) < (unsigned int)wcount;
            if (one_case) {
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    const unsigned int pq = half ? pq1 : pq0;
                    unsigned int mx = 0u, sm = 0u, sq = 0u;
#pragma unroll
                    for (int tile = 0; tile < 2; tile++) {
                        const int n0 = tq * 16 + tile * 8 + tig * 2;
#pragma unroll
                        for (int i = 0; i < 2; i++) {
                            const unsigned int ag =
                                (unsigned int)(512 + 2 * acc[tile][half * 2 + i]) - pq -
                                sh_pt[n0 + i];
                            mx = ag > mx ? ag : mx;
                            sm += ag;
                            sq += ag * ag;
                        }
                    }
#pragma unroll
                    for (int off = 1; off < 4; off <<= 1) {
                        const unsigned int omx = __shfl_xor_sync(0xFFFFFFFFu, mx, off);
                        mx = omx > mx ? omx : mx;
                        sm += __shfl_xor_sync(0xFFFFFFFFu, sm, off);
                        sq += __shfl_xor_sync(0xFFFFFFFFu, sq, off);
                    }
                    if (tig == 0) {
                        const int at = (qb * 16 + grp + half * 8) * IMMA_ACC_STRIDE - rw + rq;
                        atomicMax(&sh_max[at], mx);
                        atomicAdd(&sh_sum[at], sm);
                        atomicAdd(&sh_sq[at], sq);
                    }
                }
            }
#pragma unroll
            for (int tile = 0; tile < 2; tile++) {
                if (one_case) {
                    break;
                }
                const int t0 = tq * 16 + tile * 8;
                const int c0 = acc[tile][0];
                const int c1 = acc[tile][1];
                const int c2 = acc[tile][2];
                const int c3 = acc[tile][3];

                // Per-token epilogue for a quarter that spans a case boundary:
                // this thread's 4 accumulator elements are (row grp, tokens
                // n0/n0+1) and (row grp+8, same tokens) — the adjacent-token
                // pair merges in registers when it shares a rank.
                const int n0 = t0 + tig * 2;
                const unsigned short rk0 = sh_rank[n0];
                const unsigned short rk1 = sh_rank[n0 + 1];
                const bool ok0 =
                    (unsigned int)((int)rk0 - rw) < (unsigned int)wcount;
                const bool ok1 =
                    (unsigned int)((int)rk1 - rw) < (unsigned int)wcount;
                const unsigned int pt0 = sh_pt[n0];
                const unsigned int pt1 = sh_pt[n0 + 1];
                const unsigned int ag00 =
                    (unsigned int)(512 + 2 * c0) - pq0 - pt0;
                const unsigned int ag01 =
                    (unsigned int)(512 + 2 * c1) - pq0 - pt1;
                const unsigned int ag10 =
                    (unsigned int)(512 + 2 * c2) - pq1 - pt0;
                const unsigned int ag11 =
                    (unsigned int)(512 + 2 * c3) - pq1 - pt1;
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int row = qb * 16 + grp + half * 8;
                    const unsigned int agA = half ? ag10 : ag00;
                    const unsigned int agB = half ? ag11 : ag01;
                    const int base = row * IMMA_ACC_STRIDE - rw;
                    if (ok0 && ok1 && rk0 == rk1) {
                        const unsigned int mx = agA > agB ? agA : agB;
                        atomicMax(&sh_max[base + rk0], mx);
                        atomicAdd(&sh_sum[base + rk0], agA + agB);
                        atomicAdd(&sh_sq[base + rk0], agA * agA + agB * agB);
                    } else {
                        if (ok0) {
                            atomicMax(&sh_max[base + rk0], agA);
                            atomicAdd(&sh_sum[base + rk0], agA);
                            atomicAdd(&sh_sq[base + rk0], agA * agA);
                        }
                        if (ok1) {
                            atomicMax(&sh_max[base + rk1], agB);
                            atomicAdd(&sh_sum[base + rk1], agB);
                            atomicAdd(&sh_sq[base + rk1], agB * agB);
                        }
                    }
                }
            }
            // Explicit reconvergence after the divergent epilogue: the next
            // query tile's mma.sync requires all 32 lanes converged.
            __syncwarp();
            __syncthreads();

            // Flush this window to the global per-case accumulators (identical
            // to the b1 backend), then rejoin before the next zero/restage.
            for (int idx = tid; idx < IMMA_QT * wcount; idx += blockDim.x) {
                const int qq = idx / wcount;
                const int r = idx % wcount;
                const int q = q0 + qq;
                if (q >= n_probe_tokens) {
                    continue;
                }
                const unsigned int c = sh_rank_case[rw + r];
                const size_t gbase = ((size_t)q * n_groups + g) * n_cases + c;
                const int sbase = qq * IMMA_ACC_STRIDE + r;
                atomicMax(&case_max[gbase], sh_max[sbase]);
                atomicAdd(&case_sum[gbase], (unsigned long long)sh_sum[sbase]);
                atomicAdd(&case_sumsq[gbase], (unsigned long long)sh_sq[sbase]);
            }
            __syncthreads();
        }
    }
#else
    // Stub below sm_80 — never launched (the launcher gates at runtime).
    (void)gallery_case; (void)probe_words; (void)page_ptr; (void)pos_map;
    (void)n_tokens; (void)n_probe_tokens; (void)n_groups; (void)n_cases;
    (void)wpt; (void)case_max; (void)case_sum; (void)case_sumsq;
#endif
}

// The per-(query, group, segment) finalize is backend-agnostic — reuse the b1
// backend's kernel (same global accumulator layout, same `bdp_vote`).
extern "C" __global__ void bdp_bmma_finalize_kernel(
    const unsigned int *case_max, const unsigned long long *case_sum,
    const unsigned long long *case_sumsq, const int *seg_tok_start,
    const int *seg_case_start, int n_probe_tokens, int n_groups, int n_segments,
    int n_cases, int *out_case, float *out_vote);

// Whether this backend can run on the current device: the hardware must have
// INT8 MMA (sm_80+) AND the embedded fatbin must hold a loadable kernel image
// for it. The build ships sm_89 and sm_120 SASS plus compute_120 PTX, so e.g.
// an sm_90 (Hopper) card has the instruction but no loadable image — the probe
// (`cudaFuncGetAttributes` fails exactly when no image loads) keeps this
// honest without hard-coding the build's arch list. The Rust side prefers b1
// BMMA where it exists and falls down its ladder when this reports 0.
extern "C" int bdp_imma_supported() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        return 0;
    }
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) !=
        cudaSuccess) {
        return 0;
    }
    if (major < 8) {
        return 0;
    }
    cudaFuncAttributes attr;
    if (cudaFuncGetAttributes(&attr, (const void *)bdp_imma_accum_kernel) !=
        cudaSuccess) {
        (void)cudaGetLastError(); // clear the sticky probe error
        return 0;
    }
    return 1;
}

// The stage of a failed launch, encoded in the return code as
// `-(stage * 1000 + cudaError)` — the same values `run_bmma_bdp_scan` returns
// and `gallery_arena::scan::launch_failure` decodes, so a failure names WHICH of
// the launcher's CUDA calls rejected the work, not just the error.
#define IMMA_STAGE_ALLOC 1
#define IMMA_STAGE_MEMSET 2
#define IMMA_STAGE_ACCUM 3
#define IMMA_STAGE_FINALIZE 4

// Host launcher — the IMMA twin of `run_bmma_bdp_scan`: same temp accumulators,
// same finalize, same return contract (0 ok, 1 unsupported device/geometry,
// negative stage-encoded cudaError).
extern "C" int run_imma_bdp_scan(
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
    if (!bdp_imma_supported() || gw != IMMA_GW || page_ptr == nullptr ||
        pos_map == nullptr) {
        return 1;
    }
    if (n_tokens <= 0 || n_probe_tokens <= 0 || n_groups <= 0 ||
        n_segments <= 0 || n_cases <= 0) {
        return 1;
    }
    cudaStream_t s = (cudaStream_t)stream;
    // Discard any error left pending on this thread by unrelated earlier work,
    // as `run_bmma_bdp_scan` does. The `cudaGetLastError()` reads below return
    // whatever is pending, so without this an earlier, unchecked launch's
    // failure is reported as this backend's — on every call, since the error
    // stays pending until someone reads it — and the scan falls to the scalar
    // rung for a launch that never failed.
    (void)cudaGetLastError();

    // Probe tokens in chunks sized to `BDP_SCRATCH_BYTES` (`bdp_probe_chunk`) —
    // see `run_bmma_bdp_scan`, which does the same.
    const int chunk = bdp_probe_chunk(n_groups, n_cases);
    const size_t nc = (size_t)chunk * n_groups * n_cases;

    // The three accumulators, carved from one grow-only scratch the launcher
    // keeps (`grow_scratch.cuh`) — see `run_bmma_bdp_scan`, which does the same.
    static void *s_block = nullptr;
    static size_t s_cap = 0;
    static unsigned long long s_epoch = 0;
    const size_t max_bytes = (nc * sizeof(unsigned int) + 255) & ~(size_t)255;
    const size_t acc_bytes = (nc * sizeof(unsigned long long) + 255) & ~(size_t)255;
    unsigned int *d_max = nullptr;
    unsigned long long *d_sum = nullptr;
    unsigned long long *d_sq = nullptr;
    int stage = IMMA_STAGE_ALLOC;
    cudaError_t err = grow_scratch(&s_block, &s_cap, &s_epoch, max_bytes + 2 * acc_bytes);
    if (err == cudaSuccess) {
        d_max = (unsigned int *)s_block;
        d_sum = (unsigned long long *)((char *)s_block + max_bytes);
        d_sq = (unsigned long long *)((char *)s_block + max_bytes + acc_bytes);
    }
    for (int q0 = 0; err == cudaSuccess && q0 < n_probe_tokens; q0 += chunk) {
        const int nq = (n_probe_tokens - q0) < chunk ? (n_probe_tokens - q0) : chunk;
        const size_t ncq = (size_t)nq * n_groups * n_cases;
        const size_t out_at = (size_t)q0 * n_groups * n_segments;
        stage = IMMA_STAGE_MEMSET;
        err = cudaMemsetAsync(d_max, 0, ncq * sizeof(unsigned int), s);
        if (err == cudaSuccess) {
            err = cudaMemsetAsync(d_sum, 0, ncq * sizeof(unsigned long long), s);
        }
        if (err == cudaSuccess) {
            err = cudaMemsetAsync(d_sq, 0, ncq * sizeof(unsigned long long), s);
        }
        if (err == cudaSuccess) {
            stage = IMMA_STAGE_ACCUM;
            const dim3 grid((n_tokens + IMMA_TC - 1) / IMMA_TC, n_groups);
            bdp_imma_accum_kernel<<<grid, 256, 0, s>>>(
                gallery_case, probe_words + (size_t)q0 * wpt, page_ptr, pos_map, n_tokens,
                nq, n_groups, n_cases, wpt, d_max, d_sum, d_sq);
            err = cudaGetLastError();
        }
        if (err == cudaSuccess) {
            stage = IMMA_STAGE_FINALIZE;
            const int total = nq * n_groups * n_segments;
            const int blocks = (total + 255) / 256;
            bdp_bmma_finalize_kernel<<<blocks, 256, 0, s>>>(
                d_max, d_sum, d_sq, seg_tok_start, seg_case_start, nq,
                n_groups, n_segments, n_cases, out_case + out_at, out_vote + out_at);
            err = cudaGetLastError();
        }
    }
    return (err == cudaSuccess) ? 0 : -(stage * 1000 + (int)err);
}
