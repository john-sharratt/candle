#pragma once
// ============================================================================
// RoPE FOR THE INT8 PREFILL STAGING — table reads issued ahead of the rotation
// ============================================================================
//
// The prefill kernel rotates Q rows and staged K columns a window pair at a
// time. `rope_cs_at` answers a pass-through frequency with an early return, and
// that branch is a basic-block boundary the compiler will not move a load
// across: every pair's two table reads were issued only after the previous
// pair's rotation had consumed its own, so a row paid one L2 round trip per
// pair, one after another — and a staged K quad paid four per window pair,
// one per token, on top of the arena read they rotate.
//
// `i8_rope_cs` is `rope_cs_at` spelled without the branch: a pass-through
// frequency reads a one-row identity table instead of skipping the read, and
// the identity is selected after. Straight-line code lets every read of a row
// or a quad issue together, ahead of the values they rotate. The products are
// `rope_cs_at`'s own — the same `rope_f_lookup`, the same scale multiply — and
// the rotations below are `int8_elem::i8_apply_rope_pair`'s expressions
// unchanged, so a rotated value carries the same bits either way.
// ============================================================================

#include <cuda_runtime.h>
#include "../rope/rope_table.cuh"

namespace prefill_int8 {

/// The table a pass-through frequency reads: one `(sin, cos)` row, valid at
/// every index `rope_f_lookup` forms with `pairs == 0`. Its value never reaches
/// a result — the identity is selected over it.
static __device__ const float2 I8_ROPE_IDENTITY_ROW = {0.f, 1.f};

/// `cos` and `sin` of one frequency.
struct I8RopeCs {
    float c;
    float s;
};

/// `rope_cs_at(v, pos, f, c, s)`, branch-free.
__device__ __forceinline__ I8RopeCs i8_rope_cs(const RopeView& v, int pos, int f)
{
    const bool rot = f < v.pairs;
    const float2* tab = rot ? v.tab : &I8_ROPE_IDENTITY_ROW;
    const float2 sc = rope_f_lookup(tab, rot ? v.pairs : 0, rot ? pos : 0, rot ? f : 0);
    I8RopeCs r;
    r.c = rot ? sc.y * v.scale : 1.f;
    r.s = rot ? sc.x * v.scale : 0.f;
    return r;
}

/// The rotation of one window pair — window `w` (`lo`) against window
/// `w + N_WIN/2` (`hi`) — at one position, read ahead of the values it turns.
/// The rotary layout reads one frequency for both windows; the interleaved
/// layout one per window. Both are read whatever the layout, so the read is
/// branch-free; the rotary layout's second read is the first's address again.
/// (Skipping that read behind the layout test measured ~3% slower: the branch
/// splits the run of reads it sits in.)
struct I8RopePair {
    I8RopeCs lo;
    I8RopeCs hi;
};

template <int N_WIN>
__device__ __forceinline__ I8RopePair i8_rope_pair_cs(
    const RopeView& rope, int pos, int w, int lane, int rope_interleaved)
{
    const int f_lo = rope_interleaved ? (lane + 32 * w) >> 1 : lane + 32 * w;
    const int f_hi = rope_interleaved ? (lane + 32 * (w + N_WIN / 2)) >> 1 : f_lo;
    I8RopePair p;
    p.lo = i8_rope_cs(rope, pos, f_lo);
    p.hi = i8_rope_cs(rope, pos, f_hi);
    return p;
}

/// Rotate `(lo, hi)` by `p` — `i8_apply_rope_pair`'s arithmetic, expression for
/// expression. Warp-collective under the interleaved layout (one `lane ^ 1`
/// shuffle per window).
__device__ __forceinline__ void i8_rope_pair_apply(
    float& lo, float& hi, const I8RopePair& p, int lane, int rope_interleaved)
{
    if (rope_interleaved) {
        const float sign = (lane & 1) ? 1.f : -1.f;
        float c = p.lo.c, s = p.lo.s;
        const float plo = __shfl_sync(0xffffffffu, lo, lane ^ 1);
        lo = lo * c + sign * plo * s;
        c = p.hi.c;
        s = p.hi.s;
        const float phi = __shfl_sync(0xffffffffu, hi, lane ^ 1);
        hi = hi * c + sign * phi * s;
    } else {
        const float c = p.lo.c, s = p.lo.s;
        const float l = lo, h = hi;
        lo = l * c - h * s;
        hi = l * s + h * c;
    }
}

/// RoPE over a whole register row `x[N_WIN]` (lane `l` holds dims
/// `{l + 32w}`) — `int8_elem::i8_apply_rope` with every pair's table reads
/// issued before the first rotation.
template <int N_WIN>
__device__ __forceinline__ void i8_rope_row(
    float (&x)[N_WIN], int pos, int lane, int rope_interleaved, const RopeView& rope)
{
    static_assert(N_WIN % 2 == 0, "windows rotate in pairs: HEAD_DIM is a multiple of 64");
    I8RopePair p[N_WIN / 2];
    #pragma unroll
    for (int w = 0; w < N_WIN / 2; ++w)
        p[w] = i8_rope_pair_cs<N_WIN>(rope, pos, w, lane, rope_interleaved);
    #pragma unroll
    for (int w = 0; w < N_WIN / 2; ++w)
        i8_rope_pair_apply(x[w], x[w + N_WIN / 2], p[w], lane, rope_interleaved);
}

} // namespace prefill_int8
