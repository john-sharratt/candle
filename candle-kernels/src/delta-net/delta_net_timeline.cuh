#pragma once
// Gated DeltaNet — the timeline: a short span's contribution to the state,
// stored as the generators the kernels already compute instead of as an
// advanced state (docs/deltanet_timeline.md).
//
// A layer's state slot is [S | conv tail | timeline]. A span of at most
// DNT_ROWS rows — a decode row, a speculative verify block, a short prompt —
// reads S and the tail and writes neither; it writes one timeline row per
// token, and the fold (delta_net_fold_kernel.cuh) applies the first `m` of
// them to S and the tail once `m` — the rows the step keeps — is known.
//
// The timeline is field-major, every field DNT_ROWS rows deep:
//   g [R][h_v]          G[t] — the span's inclusive log-decay scan at row t
//   k [R][h_k][D]       row t's l2-normed, unscaled key, once per K head
//   v [R][h_v][D]       row t's v_new = u − w·S₀ᵀ, per V head
//   x [R][channels]     row t's raw conv input, for the conv tail
// R = DNT_ROWS is a multiple of 8, so every field starts 32-byte aligned
// whatever h_v is, and the k/v/x rows are whole float4s (D and channels are
// multiples of 128).
//
// Mirrored in Rust by `TimelineLayout` (delta_net/timeline.rs); a test there
// pins the offsets.

#include "delta_net_common.cuh"

// Rows a timeline holds — the longest span the short-span kernel runs
// (DNS_ROWS). Must match `DELTA_NET_TIMELINE_ROWS` in api.rs.
#define DNT_ROWS 8

namespace delta_net {

struct DnTimeline {
    float* g;
    float* k;
    float* v;
    float* x;
};

// The four fields of the timeline at `base`, for h_v V heads, h_k K heads of
// DN_HEAD_DIM, and `channels` conv channels.
__device__ __forceinline__ DnTimeline dn_timeline(float* base, int h_v, int h_k, int channels) {
    DnTimeline t;
    t.g = base;
    t.k = t.g + (size_t)DNT_ROWS * h_v;
    t.v = t.k + (size_t)DNT_ROWS * h_k * DN_HEAD_DIM;
    t.x = t.v + (size_t)DNT_ROWS * h_v * DN_HEAD_DIM;
    return t;
}

// One element of the state phase's rank-1 accumulation: acc += v · kg, as one
// rounded FMA. Written as an intrinsic because the DeltaNet kernels build with
// --use_fast_math, where contraction is the compiler's choice: the short-span
// kernel, the three-kernel state pass, the decode step and the fold must all
// round this the same way for the fold to reproduce the state the scan writes.
__device__ __forceinline__ float dn_rank1_acc(float v, float kg, float acc) {
    return __fmaf_rn(v, kg, acc);
}

// k weighted by the decay from its row to the span's last: one rounded multiply.
__device__ __forceinline__ float dn_decayed_key(float k, float e) {
    return __fmul_rn(k, e);
}

// The state phase's store: S · e^{G_last} + acc, as one rounded FMA.
__device__ __forceinline__ float dn_state_update(float s, float decay, float acc) {
    return __fmaf_rn(s, decay, acc);
}

} // namespace delta_net
