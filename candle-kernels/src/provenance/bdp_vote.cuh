// Shared finalize math for the BDP scan backends.
//
// Both the scalar scan (`bdp_scan.cu`) and the tensor-core finalize
// (`bdp_bmma.cu`) turn a segment's per-case agreement statistics into the
// `z * margin` vote through THIS function, compiled in the same archive group
// with the same flags — so the two backends' float math is one code path and
// their votes bit-match whenever their (exact, integer) inputs match.
#pragma once

// z*margin vote for a segment leader: `top1`/`top2` are the leading and
// runner-up per-case best agreements, `sum`/`sumsq` the agreement sum and
// sum-of-squares over the segment's `n_gal` scanned tokens.
__device__ __forceinline__ float bdp_vote(unsigned int top1, unsigned int top2,
                                          unsigned long long sum,
                                          unsigned long long sumsq, int n_gal) {
    const float n = (float)n_gal;
    const float mean = (float)sum / n;
    float var = (float)sumsq / n - mean * mean;
    if (var < 1e-6f) {
        var = 1e-6f;
    }
    float z = ((float)top1 - mean) / sqrtf(var);
    if (z < 0.0f) {
        z = 0.0f;
    }
    const float margin = (float)(top1 - top2);
    return z * margin;
}

// The most cases one probe token votes for in a segment. Mirrors the Rust
// `VoteLaw::MAX_TOP_K`.
#define BDP_MAX_TOP_K 8

// A segment's best `k` cases by agreement, plus the `(k+1)`-th — the floor
// each of the `k` votes' margin is taken over — held in registers: every
// index below is a compile-time constant under the unrolled loops.
struct BdpRanks {
    unsigned int a[BDP_MAX_TOP_K + 1];
    int c[BDP_MAX_TOP_K + 1];
    // The `(k+1)`-th agreement so far: a case at or below it changes nothing.
    unsigned int floor;
};

__device__ __forceinline__ void bdp_ranks_init(BdpRanks &r) {
#pragma unroll
    for (int j = 0; j <= BDP_MAX_TOP_K; j++) {
        r.a[j] = 0u;
        r.c[j] = -1;
    }
    r.floor = 0u;
}

// Offer case `c` at agreement `m` to the best `k + 1`, `1 <= k <= BDP_MAX_TOP_K`.
// Cases are offered in ascending order and only a strictly greater agreement
// displaces, so ties rank by case id, as the single-vote loop's do. One compare
// for a case that does not place.
__device__ __forceinline__ void bdp_ranks_offer(BdpRanks &r, int k, unsigned int m, int c) {
    if (m <= r.floor) {
        return;
    }
    unsigned int ca = m;
    int cc = c;
#pragma unroll
    for (int j = 0; j <= BDP_MAX_TOP_K; j++) {
        if (j <= k && ca > r.a[j]) {
            const unsigned int ta = r.a[j];
            const int tc = r.c[j];
            r.a[j] = ca;
            r.c[j] = cc;
            ca = ta;
            cc = tc;
        }
        if (j == k) {
            r.floor = r.a[j];
        }
    }
}

// Write a token's `max_k` votes in one segment: its best `k` cases, case `r`
// voting `z_r * (a_r - a_k)` through `bdp_vote`, and `-1` / `0` for the rest
// (fewer cases than `k` scored, or `k < max_k`). `case_base` turns a
// segment-local case into a global one.
__device__ __forceinline__ void bdp_ranks_emit(const BdpRanks &r, int k, int max_k, int case_base,
                                               unsigned long long sum,
                                               unsigned long long sumsq, int n_gal,
                                               int *__restrict__ out_case,
                                               float *__restrict__ out_vote) {
#pragma unroll
    for (int j = 0; j < BDP_MAX_TOP_K; j++) {
        if (j < max_k) {
            const bool live = j < k && r.c[j] >= 0;
            out_case[j] = live ? case_base + r.c[j] : -1;
            out_vote[j] = live ? bdp_vote(r.a[j], r.floor, sum, sumsq, n_gal) : 0.0f;
        }
    }
}
