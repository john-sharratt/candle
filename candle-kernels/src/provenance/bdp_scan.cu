// Batched Binary Directional Provenance (BDP) flat scan.
//
// The CPU reference is `candle_conversation::provenance::score_packed`: for each
// (query token, layer-group) it scans the whole gallery computing per-case best
// sign-agreement (popcount of XNOR over the group's words) plus the agreement
// sum/sumsq, then emits one `z * margin` vote for the leading case. The needle
// gate + per-case tally run on the host, identical to the CPU path — so this
// kernel only produces the per-(query-token, group) `(leading_case, vote)` pairs.
//
// This design is the empirically-fastest of several tried (thread-per-query-token
// with a shared gallery and scalar accumulators, and a broadcast-read variant,
// both measured ~3x slower): a **query-token tile per block** with the tile loop
// **fully unrolled**. The unroll is what makes it fast — it issues `BDP_TQ`
// independent popcount chains per gallery token (high ILP), keeping the popcount
// pipelines saturated even at the low (~12%) occupancy the resulting 200-register
// footprint implies. That is the Volkov "high-ILP, low-occupancy" regime; forcing
// higher occupancy (register caps / less unroll) measured *slower*.
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

// Locked folded-signature group width (4 heads x 128 bits = 8 u64).
#define BDP_MAX_GW 8
// Query tokens per block tile — the gallery-reuse / ILP factor.
#define BDP_TQ 8

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
extern "C" __global__ void bdp_scan_kernel(
    const unsigned long long *__restrict__ gallery_words, // GROUP-major: [g][token][gw], tokens sorted by segment
    const unsigned int *__restrict__ gallery_case,        // n_tokens (GLOBAL case id)
    const unsigned long long *__restrict__ probe_words,   // token-major: n_probe_tokens * wpt
    const int *__restrict__ seg_tok_start,                // n_segments+1 (token range per segment)
    const int *__restrict__ seg_case_start,               // n_segments+1 (case range per segment)
    int n_probe_tokens,
    int n_groups,
    int n_segments,
    int max_seg_cases, // shared-mem case stride (>= every segment's case count)
    int gw,            // words per layer-group (<= BDP_MAX_GW)
    int wpt,           // words per token (n_groups * gw)
    int *__restrict__ out_case,   // n_probe_tokens * n_groups * n_segments
    float *__restrict__ out_vote) // n_probe_tokens * n_groups * n_segments
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

    // Output index for this (tile, group, segment): query token i is at
    // ((q0+i)*n_groups + g)*n_segments + s.
    // Empty segment ⇒ no vote.
    if (seg_nc <= 0 || tok1 <= tok0) {
        for (int i = tid; i < tq; i += nthreads) {
            const int oi = ((q0 + i) * n_groups + g) * n_segments + s;
            out_case[oi] = -1;
            out_vote[oi] = 0.0f;
        }
        return;
    }

    // Dynamic shared: per-query-token per-(segment-local)-case best agreement —
    // [BDP_TQ][max_seg_cases]. Sized once at launch to the largest segment.
    extern __shared__ unsigned int s_case_max[];
    __shared__ unsigned long long s_sum[BDP_TQ];
    __shared__ unsigned long long s_sumsq[BDP_TQ];
    __shared__ unsigned long long qg[BDP_TQ * BDP_MAX_GW];

    for (int i = tid; i < tq * max_seg_cases; i += nthreads) {
        s_case_max[i] = 0u;
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

    const unsigned long long *g_base = gallery_words + (size_t)g * seg_tok_start[n_segments] * gw;
    unsigned long long lsum[BDP_TQ];
    unsigned long long lsumsq[BDP_TQ];
    for (int i = 0; i < tq; i++) {
        lsum[i] = 0ull;
        lsumsq[i] = 0ull;
    }

    for (int j = tok0 + tid; j < tok1; j += nthreads) {
        // Load the gallery token's group words ONCE, reuse across the tile. The
        // locked 8-word group is two vectorized 32-byte (ulonglong4) loads.
        unsigned long long tw[BDP_MAX_GW];
        const unsigned long long *tok = g_base + (size_t)j * gw;
        if (gw == BDP_MAX_GW) {
            const ulonglong4 *t4 = reinterpret_cast<const ulonglong4 *>(tok);
            const ulonglong4 a = t4[0];
            const ulonglong4 b = t4[1];
            tw[0] = a.x; tw[1] = a.y; tw[2] = a.z; tw[3] = a.w;
            tw[4] = b.x; tw[5] = b.y; tw[6] = b.z; tw[7] = b.w;
        } else {
#pragma unroll
            for (int k = 0; k < BDP_MAX_GW; k++) {
                if (k < gw) {
                    tw[k] = tok[k];
                }
            }
        }
        // Segment-local case index (cases are contiguous per segment).
        const unsigned int c = gallery_case[j] - (unsigned int)case0;
        // Full unroll over the compile-time tile: keeps `lsum`/`lsumsq`
        // register-resident and issues BDP_TQ independent popcount chains (ILP).
#pragma unroll
        for (int i = 0; i < BDP_TQ; i++) {
            if (i >= tq) {
                continue;
            }
            const unsigned long long *qi = qg + (size_t)i * gw;
            unsigned int ag = 0u;
#pragma unroll
            for (int k = 0; k < BDP_MAX_GW; k++) {
                if (k < gw) {
                    ag += __popcll(~(qi[k] ^ tw[k]));
                }
            }
            atomicMax(&s_case_max[(size_t)i * max_seg_cases + c], ag);
            lsum[i] += ag;
            lsumsq[i] += (unsigned long long)ag * (unsigned long long)ag;
        }
    }
    for (int i = 0; i < tq; i++) {
        atomicAdd(&s_sum[i], lsum[i]);
        atomicAdd(&s_sumsq[i], lsumsq[i]);
    }
    __syncthreads();

    // One thread per query token in the tile: leader/runner-up (WITHIN the
    // segment) → z*margin vote, z's mean/std over the segment's tokens.
    for (int i = tid; i < tq; i += nthreads) {
        const unsigned int *cmax = s_case_max + (size_t)i * max_seg_cases;
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
        const int oi = ((q0 + i) * n_groups + g) * n_segments + s;
        if (top1c < 0) {
            out_case[oi] = -1;
            out_vote[oi] = 0.0f;
        } else {
            const float n_gal = (float)(tok1 - tok0);
            const float mean = (float)s_sum[i] / n_gal;
            float var = (float)s_sumsq[i] / n_gal - mean * mean;
            if (var < 1e-6f) {
                var = 1e-6f;
            }
            float z = ((float)top1 - mean) / sqrtf(var);
            if (z < 0.0f) {
                z = 0.0f;
            }
            const float margin = (float)(top1 - top2);
            out_case[oi] = case0 + top1c; // GLOBAL case id
            out_vote[oi] = z * margin;
        }
    }
}

// Host launcher: one block per (query-token tile × group, segment), `threads`
// per block, dynamic shared = BDP_TQ * max_seg_cases * sizeof(u32). Launched on
// the caller's stream. The non-segmented (global-z) scan is `n_segments == 1`
// with `seg_tok_start = {0, n_tokens}` and `seg_case_start = {0, n_cases}`.
extern "C" void run_batched_bdp_scan(
    const unsigned long long *gallery_words,
    const unsigned int *gallery_case,
    const unsigned long long *probe_words,
    const int *seg_tok_start,
    const int *seg_case_start,
    int n_probe_tokens,
    int n_groups,
    int n_segments,
    int max_seg_cases,
    int gw,
    int wpt,
    int *out_case,
    float *out_vote,
    void *stream)
{
    if (n_probe_tokens <= 0 || n_groups <= 0 || n_segments <= 0 || max_seg_cases <= 0) {
        return;
    }
    const int n_tiles = (n_probe_tokens + BDP_TQ - 1) / BDP_TQ;
    dim3 blocks(n_tiles * n_groups, n_segments);
    const int threads = 128;
    const size_t shmem = (size_t)BDP_TQ * max_seg_cases * sizeof(unsigned int);
    cudaStream_t s = (cudaStream_t)stream;
    bdp_scan_kernel<<<blocks, threads, shmem, s>>>(
        gallery_words, gallery_case, probe_words,
        seg_tok_start, seg_case_start,
        n_probe_tokens, n_groups, n_segments, max_seg_cases, gw, wpt,
        out_case, out_vote);
}
