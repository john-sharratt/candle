// Batched Binary Directional Provenance (BDP) flat scan.
//
// The CPU reference is `candle_conversation::provenance::score_packed`: for each
// (query token, layer-group) it scans the whole gallery computing per-case best
// sign-agreement (popcount of XNOR over the group's words) plus the agreement
// sum/sumsq, then emits one `z * margin` vote for the leading case. The needle
// gate + per-case tally run on the host, identical to the CPU path — so this
// kernel only produces the per-(query-token, group) `(leading_case, vote)` pairs.
//
// This is the SCALAR backend — the fallback for devices without b1 BMMA tensor
// cores (Hopper/Blackwell dropped them) and the differential-testing oracle for
// the production tensor-core backend in `bdp_bmma.cu` (which computes the same
// integer statistics ~2-3x faster and shares `bdp_vote.cuh`, so their votes
// bit-match).
//
// The design is the empirically-fastest scalar form of several tried
// (thread-per-query-token with a shared gallery and scalar accumulators, and a
// broadcast-read variant, both measured ~3x slower): a **query-token tile per
// block** with the tile loop **fully unrolled**, issuing `BDP_TQ` independent
// popcount chains per gallery token (high ILP). The paged scan is
// **latency-bound** (Nsight: ~63% memory-pipe, ~48% of stalls waiting on the
// shared-memory `qg` reads, DRAM idle / L2-resident). Two tuned knobs mitigate
// that: `BDP_GTOK` register-blocks the gallery dimension so each `qg` word is
// read from shared once and reused across BDP_GTOK tokens, and
// `__launch_bounds__(128, 6)` trades a little ILP for occupancy. The tuned point
// is `BDP_TQ=8, BDP_GTOK=2, minBlocks=6`; a sweep found `BDP_GTOK>2` spills
// registers and `minBlocks=4`/`BDP_TQ=16` regress. The remaining wall is the
// shared-`qg` broadcast volume, inherent to scalar issue — the tensor-core
// backend removes it entirely by folding the broadcast into the BMMA operand.
//
// Layout tricks:
//  * **Group-major** gallery `[group][token][gw]` → consecutive threads read
//    consecutive words (coalesced), not 192-byte-strided.
//  * Vectorized `ulonglong4` gallery loads (the locked 8-word group = two loads).
//  * The full unroll keeps the `lsum`/`lsumsq` accumulators register-resident
//    (constant-indexed); a partial unroll would push them to local memory.
//
// Batching: `probe_words` concatenates every request's probe tokens, one launch
// per wave. Integer reductions (u32 max, u64 sums) are exact/order-independent;
// the final float math is fast-math f32 (~ULP off the CPU — irrelevant for a
// ranking; validated approximately + identical argmax).

#include <cstdint>
#include <cuda_runtime.h>

#include "bdp_vote.cuh"

// Take the CUDA error pending on the calling thread, if any, and clear it:
// `cudaSuccess` (0) when there is none.
//
// Most launchers in this crate never read the launch result, so a launch that
// fails — an empty grid, an oversized block — leaves its error pending until
// the next `cudaGetLastError()` on the thread, which belongs to whoever happens
// to call it. The provenance scan takes it BEFORE its own launches so that it
// is reported as an earlier launch's, not as the scan's.
extern "C" int bdp_take_pending_error() {
    return (int)cudaGetLastError();
}

// Locked folded-signature group width (4 heads x 128 bits = 8 u64).
#define BDP_MAX_GW 8
// Query tokens per block tile — the gallery-reuse / ILP factor.
#define BDP_TQ 8
// Gallery tokens processed per thread per iteration (register blocking): each
// query word `qg[i][k]` is read from shared ONCE and reused across BDP_GTOK
// gallery tokens, cutting the dominant shared-memory (`qg`) MIO traffic
// ~BDP_GTOK×. Each thread carries BDP_GTOK tokens' words in registers.
#define BDP_GTOK 2
// Tokens per gallery page in the paged path (mirrors the arena's PAGE_TOKENS).
#define BDP_PAGE_TOKENS 32

// Segmented form: the gallery tokens are sorted by SEGMENT (a code-read file /
// timeline), and each segment owns a contiguous token range AND a contiguous
// case (exchange) range. The block grid gains a segment dimension (`blockIdx.y`),
// and each block scans ONLY its segment's tokens — so `case_max`, the agreement
// mean/std, and the leader/runner-up are all computed WITHIN the segment. That is
// exactly the belief scan's per-file z / margin, done in one launch. The
// non-segmented case is just `n_segments == 1` (one segment spanning everything),
// which reproduces the original global-z behaviour bit-for-bit.
//
// The popcount hot loop is UNCHANGED — only the token range, the (segment-local)
// case index, the z's `n_gal`, and the output index differ.
//
// Two gallery layouts, selected by whether `page_ptr` is null:
//  * CONTIGUOUS (`page_ptr == nullptr`): one group-major buffer `gallery_words`;
//    token `j`'s group-`g` words are at `g*n_tokens*gw + j*gw`.
//  * PAGED (`page_ptr != nullptr`): the gallery lives in the resident arena as
//    group-major pages of BDP_PAGE_TOKENS tokens each, scanned through page
//    entries: slot `j` is slot `j & 31` of page `entry_page[j >> 5]`, scanned
//    when bit `j & 31` of `entry_slots[j >> 5]` is set, voting for the entry's
//    case `gallery_case[j >> 5]`; `page_ptr[page * n_groups + g]` is the absolute
//    device address of launch-group `g`'s strip in that page (a descriptor per
//    (page, group): a page holds only the groups its turn keeps). A segment's
//    range is in slots; its vote's mean and spread are over the tokens it
//    scans, `seg_tok_count`.
extern "C" __global__ void __launch_bounds__(128, 6) bdp_scan_kernel(
    const unsigned long long *__restrict__ gallery_words, // GROUP-major buffer, or null (paged)
    const unsigned int *__restrict__ gallery_case,        // GLOBAL case ids: per token (contiguous), per entry (paged)
    const unsigned long long *__restrict__ probe_words,   // token-major: n_probe_tokens * wpt
    const int *__restrict__ seg_tok_start,                // n_segments+1 (token / slot range per segment)
    const int *__restrict__ seg_tok_count,                // n_segments+1 (tokens each segment scans, prefix)
    const int *__restrict__ seg_case_start,               // n_segments+1 (case range per segment)
    const unsigned long long *__restrict__ page_ptr,      // paged: n_pages*n_groups (page, group) addresses, or null
    const unsigned int *__restrict__ entry_page,          // paged: the page each entry scans, or null
    const unsigned int *__restrict__ entry_slots,         // paged: its page's slots it scans, a bit each, or null
    const int *__restrict__ seg_top_k,                    // n_segments: cases a token votes for, or null (1 each)
    int n_probe_tokens,
    int n_groups,
    int n_segments,
    int max_seg_cases, // shared-mem case stride (>= every segment's case count)
    int gw,            // words per layer-group (<= BDP_MAX_GW)
    int wpt,           // words per token (n_groups * gw)
    int max_k,         // votes per output row (>= every segment's top_k)
    int *__restrict__ out_case,   // n_probe_tokens * n_groups * n_segments * max_k
    float *__restrict__ out_vote, // n_probe_tokens * n_groups * n_segments * max_k
    BdpStats *__restrict__ out_stats) // statistics mode: one row per (token, group, segment); null for votes
{
    const int tile = blockIdx.x / n_groups;
    const int g = blockIdx.x % n_groups;
    const int s = blockIdx.y;
    const int q0 = tile * BDP_TQ;
    if (q0 >= n_probe_tokens || s >= n_segments) {
        return;
    }
    int tq = n_probe_tokens - q0;
    if (tq > BDP_TQ) {
        tq = BDP_TQ;
    }

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int tok0 = seg_tok_start[s];
    const int tok1 = seg_tok_start[s + 1];
    const int case0 = seg_case_start[s];
    const int seg_nc = seg_case_start[s + 1] - case0;

    // Output row for this (tile, group, segment): query token i's `max_k`
    // votes start at (((q0+i)*n_groups + g)*n_segments + s) * max_k.
    // Empty segment ⇒ no vote.
    if (seg_nc <= 0 || tok1 <= tok0) {
        for (int i = tid; i < tq; i += nthreads) {
            if (out_stats != nullptr) {
                bdp_stats_emit(nullptr, 0, case0, 0ull, 0ull,
                               out_stats + (((q0 + i) * n_groups + g) * n_segments + s));
                continue;
            }
            const int oi = (((q0 + i) * n_groups + g) * n_segments + s) * max_k;
            for (int r = 0; r < max_k; r++) {
                out_case[oi + r] = -1;
                out_vote[oi + r] = 0.0f;
            }
        }
        return;
    }

    // Dynamic shared: per-query-token per-(segment-local)-case best agreement —
    // [BDP_TQ][max_seg_cases]. Sized once at launch to the largest segment.
    extern __shared__ unsigned int s_case_max[];
    __shared__ unsigned long long s_sum[BDP_TQ];
    __shared__ unsigned long long s_sumsq[BDP_TQ];
    __shared__ unsigned long long qg[BDP_TQ * BDP_MAX_GW];

    // Zero only the cells this segment actually uses ([0,tq) x [0,seg_nc)), not
    // the full `tq * max_seg_cases` stride — `max_seg_cases` is the LARGEST
    // segment's case count (can be hundreds), so zeroing the whole row for a
    // 3-case file wasted ~two orders of magnitude of shared writes. The top-2 loop
    // only reads `c < seg_nc`, so untouched cells are never observed.
    for (int idx = tid; idx < tq * seg_nc; idx += nthreads) {
        const int i = idx / seg_nc;
        const int c = idx % seg_nc;
        s_case_max[(size_t)i * max_seg_cases + c] = 0u;
    }
    for (int i = tid; i < tq; i += nthreads) {
        s_sum[i] = 0ull;
        s_sumsq[i] = 0ull;
    }
    // The tile's query-token group words (probe stays token-major — tiny).
    for (int idx = tid; idx < tq * gw; idx += nthreads) {
        const int i = idx / gw;
        const int k = idx % gw;
        qg[i * gw + k] = probe_words[(size_t)(q0 + i) * wpt + (size_t)g * gw + k];
    }
    __syncthreads();

    const bool paged = (page_ptr != nullptr);
    // Contiguous: group g's strip is the whole gallery's group-g plane.
    // Paged: `page_ptr[page * n_groups + g]` is group g's strip in that page —
    // a page holds only the groups its turn keeps, packed, so the address is
    // per (page, group), not a fixed offset into the page.
    const unsigned long long *g_base =
        paged ? nullptr : gallery_words + (size_t)g * seg_tok_start[n_segments] * gw;
    unsigned long long lsum[BDP_TQ];
    unsigned long long lsumsq[BDP_TQ];
    for (int i = 0; i < tq; i++) {
        lsum[i] = 0ull;
        lsumsq[i] = 0ull;
    }

    // Register-blocked over BDP_GTOK gallery tokens per thread: threads still read
    // consecutive tokens per stripe (coalesced), but each stripe is `nthreads`
    // apart so a thread carries BDP_GTOK tokens' words at once. The per-query word
    // `qg[i][k]` is then loaded from shared once and reused across all BDP_GTOK,
    // which is the whole point (shared-`qg` reads were the dominant stall).
    for (int j0 = tok0 + tid; j0 < tok1; j0 += nthreads * BDP_GTOK) {
        unsigned long long tw[BDP_GTOK][BDP_MAX_GW];
        unsigned int cc[BDP_GTOK];
        bool ok[BDP_GTOK];
#pragma unroll
        for (int b = 0; b < BDP_GTOK; b++) {
            const int j = j0 + b * nthreads;
            ok[b] = (j < tok1);
            if (!ok[b]) {
                continue;
            }
            const unsigned long long *tok;
            if (paged) {
                const int e = j >> 5;
                const int sl = j & (BDP_PAGE_TOKENS - 1);
                if (((entry_slots[e] >> sl) & 1u) == 0u) {
                    ok[b] = false; // a slot its entry does not scan
                    continue;
                }
                tok = (const unsigned long long *)page_ptr[(size_t)entry_page[e] * n_groups + g] +
                      (size_t)sl * gw;
                cc[b] = gallery_case[e] - (unsigned int)case0;
            } else {
                tok = g_base + (size_t)j * gw;
                cc[b] = gallery_case[j] - (unsigned int)case0;
            }
            if (gw == BDP_MAX_GW) {
                const ulonglong4 *t4 = reinterpret_cast<const ulonglong4 *>(tok);
                const ulonglong4 a = t4[0];
                const ulonglong4 c4 = t4[1];
                tw[b][0] = a.x; tw[b][1] = a.y; tw[b][2] = a.z; tw[b][3] = a.w;
                tw[b][4] = c4.x; tw[b][5] = c4.y; tw[b][6] = c4.z; tw[b][7] = c4.w;
            } else {
#pragma unroll
                for (int k = 0; k < BDP_MAX_GW; k++) {
                    if (k < gw) {
                        tw[b][k] = tok[k];
                    }
                }
            }
        }
        // Full unroll over the compile-time tile: keeps `lsum`/`lsumsq`
        // register-resident and issues BDP_TQ independent popcount chains (ILP).
#pragma unroll
        for (int i = 0; i < BDP_TQ; i++) {
            if (i >= tq) {
                continue;
            }
            const unsigned long long *qi = qg + (size_t)i * gw;
            unsigned int ag[BDP_GTOK];
#pragma unroll
            for (int b = 0; b < BDP_GTOK; b++) {
                ag[b] = 0u;
            }
#pragma unroll
            for (int k = 0; k < BDP_MAX_GW; k++) {
                if (k < gw) {
                    const unsigned long long qw = qi[k]; // ONE shared read
#pragma unroll
                    for (int b = 0; b < BDP_GTOK; b++) {
                        ag[b] += __popcll(~(qw ^ tw[b][k]));
                    }
                }
            }
#pragma unroll
            for (int b = 0; b < BDP_GTOK; b++) {
                if (ok[b]) {
                    atomicMax(&s_case_max[(size_t)i * max_seg_cases + cc[b]], ag[b]);
                    lsum[i] += ag[b];
                    lsumsq[i] += (unsigned long long)ag[b] * (unsigned long long)ag[b];
                }
            }
        }
    }
    for (int i = 0; i < tq; i++) {
        atomicAdd(&s_sum[i], lsum[i]);
        atomicAdd(&s_sumsq[i], lsumsq[i]);
    }
    __syncthreads();

    // One thread per query token in the tile: its best cases (WITHIN the
    // segment) → z*margin votes, z's mean/std over the segment's tokens. A
    // single-vote segment keeps the leader/runner-up loop; a top-k segment
    // keeps its best k+1 in registers.
    const int k = seg_top_k ? seg_top_k[s] : 1;
    const int n_gal = seg_tok_count[s + 1] - seg_tok_count[s];
    for (int i = tid; i < tq; i += nthreads) {
        const unsigned int *cmax = s_case_max + (size_t)i * max_seg_cases;
        if (out_stats != nullptr) {
            bdp_stats_emit(cmax, seg_nc, case0, s_sum[i], s_sumsq[i],
                           out_stats + (((q0 + i) * n_groups + g) * n_segments + s));
            continue;
        }
        const int oi = (((q0 + i) * n_groups + g) * n_segments + s) * max_k;
        if (k == 1) {
            unsigned int top1 = 0u, top2 = 0u;
            int top1c = -1;
            for (int c = 0; c < seg_nc; c++) {
                const unsigned int m = cmax[c];
                if (m > top1) {
                    top2 = top1;
                    top1 = m;
                    top1c = c;
                } else if (m > top2) {
                    top2 = m;
                }
            }
            if (top1c < 0) {
                out_case[oi] = -1;
                out_vote[oi] = 0.0f;
            } else {
                out_case[oi] = case0 + top1c; // GLOBAL case id
                out_vote[oi] = bdp_vote(top1, top2, s_sum[i], s_sumsq[i], n_gal);
            }
            for (int r = 1; r < max_k; r++) {
                out_case[oi + r] = -1;
                out_vote[oi + r] = 0.0f;
            }
        } else {
            BdpRanks ranks;
            bdp_ranks_init(ranks);
            for (int c = 0; c < seg_nc; c++) {
                bdp_ranks_offer(ranks, k, cmax[c], c);
            }
            bdp_ranks_emit(ranks, k, max_k, case0, s_sum[i], s_sumsq[i], n_gal,
                           out_case + oi, out_vote + oi);
        }
    }
}

// Host launcher: one block per (query-token tile × group, segment), `threads`
// per block, dynamic shared = BDP_TQ * max_seg_cases * sizeof(u32). Launched on
// the caller's stream. The non-segmented (global-z) scan is `n_segments == 1`
// with `seg_tok_start = {0, n_tokens}` and `seg_case_start = {0, n_cases}`.
extern "C" void run_batched_bdp_scan(
    const unsigned long long *gallery_words, // or null when paged
    const unsigned int *gallery_case,
    const unsigned long long *probe_words,
    const int *seg_tok_start,
    const int *seg_tok_count,           // contiguous: seg_tok_start itself
    const int *seg_case_start,
    const unsigned long long *page_ptr, // paged: n_pages*n_groups (page, group) addresses, or null
    const unsigned int *entry_page,     // paged: the page each entry scans, or null
    const unsigned int *entry_slots,    // paged: its page's slots it scans, a bit each, or null
    const int *seg_top_k,               // n_segments: cases a token votes for, or null (1 each)
    int n_probe_tokens,
    int n_groups,
    int n_segments,
    int max_seg_cases,
    int gw,
    int wpt,
    int max_k,
    int *out_case,
    float *out_vote,
    BdpStats *out_stats, // statistics mode, or null for votes
    void *stream)
{
    if (n_probe_tokens <= 0 || n_groups <= 0 || n_segments <= 0 || max_seg_cases <= 0 ||
        max_k <= 0 || max_k > BDP_MAX_TOP_K) {
        return;
    }
    const int n_tiles = (n_probe_tokens + BDP_TQ - 1) / BDP_TQ;
    dim3 blocks(n_tiles * n_groups, n_segments);
    const int threads = 128;
    const size_t shmem = (size_t)BDP_TQ * max_seg_cases * sizeof(unsigned int);
    cudaStream_t s = (cudaStream_t)stream;
    bdp_scan_kernel<<<blocks, threads, shmem, s>>>(
        gallery_words, gallery_case, probe_words,
        seg_tok_start, seg_tok_count, seg_case_start, page_ptr, entry_page, entry_slots,
        seg_top_k, n_probe_tokens, n_groups, n_segments, max_seg_cases, gw, wpt, max_k,
        out_case, out_vote, out_stats);
}
