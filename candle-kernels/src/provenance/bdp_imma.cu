// INT8 tensor-core Binary Directional Provenance (BDP) scan — IMMA backend.
//
// The Blackwell-portable sibling of the b1 backend (`bdp_bmma.cu`): Hopper and
// Blackwell dropped the 1-bit BMMA path, but INT8 MMA (`mma.m16n8k32.s8`) is a
// mainstream tensor format on every arch from Ampere up — including sm_120. The
// sign bits stay PACKED in the arena exactly as for the other backends; each
// thread expands the bits it needs into 0/1 s8 fragment lanes in REGISTERS
// while building its MMA operands, so the 8x data inflation never exists in
// memory. With both operands 0/1-encoded the MMA accumulates
// m11 = popcount(q AND t), which relates to the agreement by an exact integer
// identity using the per-row popcounts staged alongside the bits:
//
//   agreement = popc(XNOR(q,t)) = 512 - popc(q) - popc(t) + 2*m11
//
// (0/1 beats a +/-1 encoding by two ops per fragment — measured ~25% off the
// whole kernel.) The per-case integer statistics are IDENTICAL to both other
// backends, and the shared finalize (`bdp_bmma_finalize_kernel` +
// `bdp_vote.cuh`) emits bit-matching votes.
//
// Uses inline-PTX `mma` (not wmma): the m16n8k32 fragment<->thread mappings are
// documented in the PTX ISA, which is what makes register-side bit expansion
// legitimate. A warp computes a 16x8 (query x token) tile per 16 K-steps; the
// accumulator lives entirely in registers, so the epilogue run-merges straight
// from registers into the shared per-(query, rank) accumulators — no staging
// buffer at all. Everything else (chunk grid, ballot dense-ranking, rank
// windows, global accumulators, finalize) mirrors `bdp_bmma.cu`.

#include <cstdint>
#include <cuda_runtime.h>

#include "bdp_vote.cuh"

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
    __shared__ unsigned int sh_max[IMMA_QT * IMMA_RANK_CAP];
    __shared__ unsigned int sh_sum[IMMA_QT * IMMA_RANK_CAP];
    __shared__ unsigned int sh_sq[IMMA_QT * IMMA_RANK_CAP];

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

        // Rank windows: almost always one pass (nranks <= 32); a chunk of
        // all-tiny exchanges reruns its tiles once per window.
        for (int rw = 0; rw < nranks; rw += IMMA_RANK_CAP) {
            const int wcount =
                (nranks - rw) < IMMA_RANK_CAP ? (nranks - rw) : IMMA_RANK_CAP;
            for (int idx = tid; idx < IMMA_QT * wcount; idx += blockDim.x) {
                const int base =
                    (idx / wcount) * IMMA_RANK_CAP + (idx % wcount);
                sh_max[base] = 0u;
                sh_sum[base] = 0u;
                sh_sq[base] = 0u;
            }
            __syncthreads();

#pragma unroll
            for (int tile = 0; tile < 2; tile++) {
                const int t0 = tq * 16 + tile * 8;
                // Accumulators for the warp's 16x8 tile (documented m16n8 C
                // layout: c0=(grp, tig*2), c1=(grp, tig*2+1), c2/c3 = rows +8).
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                const unsigned long long *arow0 =
                    &sh_a[(qb * 16 + grp) * IMMA_ROW_WORDS];
                const unsigned long long *arow1 = arow0 + 8 * IMMA_ROW_WORDS;
                const unsigned long long *brow =
                    &sh_b[(t0 + grp) * IMMA_ROW_WORDS];
#pragma unroll
                for (int ks = 0; ks < 16; ks++) {
                    // Documented m16n8k32.s8 A layout (row-major 16x32):
                    //   a0=(grp,        klo) a1=(grp+8, klo)
                    //   a2=(grp,        khi) a3=(grp+8, khi)
                    // with klo = tig*4..+4, khi = klo+16; B (col-major 32x8):
                    //   b0=(klo, grp) b1=(khi, grp). klo and khi share word
                    //   ks/2 (in-word offsets <= 60), so each operand row is one
                    //   shared broadcast load and one shift per step, with both
                    //   nibbles carved from the shifted chunk.
                    const int w = ks >> 1;
                    const int s0 = (ks & 1) * 32 + tig * 4;
                    const unsigned int ea = (unsigned int)(arow0[w] >> s0);
                    const unsigned int eb = (unsigned int)(arow1[w] >> s0);
                    const unsigned int et = (unsigned int)(brow[w] >> s0);
                    const unsigned int a0 = bdp_expand4(ea & 0xFu);
                    const unsigned int a1 = bdp_expand4(eb & 0xFu);
                    const unsigned int a2 = bdp_expand4((ea >> 16) & 0xFu);
                    const unsigned int a3 = bdp_expand4((eb >> 16) & 0xFu);
                    const unsigned int b0 = bdp_expand4(et & 0xFu);
                    const unsigned int b1 = bdp_expand4((et >> 16) & 0xFu);
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }

                // Register epilogue: this thread's 4 accumulator elements are
                // (row grp, tokens n0/n0+1) and (row grp+8, same tokens) — the
                // adjacent-token pair merges in registers when it shares a rank
                // (the common case), halving the shared atomics. With the 0/1
                // encoding the MMA accumulated m11 = popc(q AND t), so
                // agreement = 512 - popc(q) - popc(t) + 2*m11 — exact.
                const int n0 = t0 + tig * 2;
                const unsigned short rk0 = sh_rank[n0];
                const unsigned short rk1 = sh_rank[n0 + 1];
                const bool ok0 =
                    (unsigned int)((int)rk0 - rw) < (unsigned int)wcount;
                const bool ok1 =
                    (unsigned int)((int)rk1 - rw) < (unsigned int)wcount;
                const unsigned int pq0 = sh_pq[qb * 16 + grp];
                const unsigned int pq1 = sh_pq[qb * 16 + grp + 8];
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
                    const int base = row * IMMA_RANK_CAP - rw;
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
                // Explicit reconvergence after the divergent epilogue: the next
                // tile iteration's mma.sync requires all 32 lanes converged.
                __syncwarp();
            }
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
                const int sbase = qq * IMMA_RANK_CAP + r;
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

// Host launcher — the IMMA twin of `run_bmma_bdp_scan`: same temp accumulators,
// same finalize, same return contract (0 ok, 1 unsupported device/geometry,
// negative cudaError).
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
    const size_t nc = (size_t)n_probe_tokens * n_groups * n_cases;

    unsigned int *d_max = nullptr;
    unsigned long long *d_sum = nullptr;
    unsigned long long *d_sq = nullptr;
    cudaError_t err = cudaMallocAsync((void **)&d_max, nc * sizeof(unsigned int), s);
    if (err == cudaSuccess) {
        err = cudaMallocAsync((void **)&d_sum, nc * sizeof(unsigned long long), s);
    }
    if (err == cudaSuccess) {
        err = cudaMallocAsync((void **)&d_sq, nc * sizeof(unsigned long long), s);
    }
    if (err == cudaSuccess) {
        err = cudaMemsetAsync(d_max, 0, nc * sizeof(unsigned int), s);
    }
    if (err == cudaSuccess) {
        err = cudaMemsetAsync(d_sum, 0, nc * sizeof(unsigned long long), s);
    }
    if (err == cudaSuccess) {
        err = cudaMemsetAsync(d_sq, 0, nc * sizeof(unsigned long long), s);
    }
    if (err == cudaSuccess) {
        const dim3 grid((n_tokens + IMMA_TC - 1) / IMMA_TC, n_groups);
        bdp_imma_accum_kernel<<<grid, 256, 0, s>>>(
            gallery_case, probe_words, page_ptr, pos_map, n_tokens,
            n_probe_tokens, n_groups, n_cases, wpt, d_max, d_sum, d_sq);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        const int total = n_probe_tokens * n_groups * n_segments;
        const int blocks = (total + 255) / 256;
        bdp_bmma_finalize_kernel<<<blocks, 256, 0, s>>>(
            d_max, d_sum, d_sq, seg_tok_start, seg_case_start, n_probe_tokens,
            n_groups, n_segments, n_cases, out_case, out_vote);
        err = cudaGetLastError();
    }
    if (d_max != nullptr) {
        cudaFreeAsync(d_max, s);
    }
    if (d_sum != nullptr) {
        cudaFreeAsync(d_sum, s);
    }
    if (d_sq != nullptr) {
        cudaFreeAsync(d_sq, s);
    }
    return (err == cudaSuccess) ? 0 : -(int)err;
}
