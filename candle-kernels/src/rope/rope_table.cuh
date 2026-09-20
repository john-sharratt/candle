#pragma once
// =============================================================================
// The engine's one RoPE table: two small tables and the angle-addition identity
// =============================================================================
// A position splits as `pos = hi·2¹⁰ + lo`, and a rotation through a sum of
// angles is the product of the two rotations. So a `HI` block of `(sin, cos)`
// at `hi·2¹⁰` and a `LO` block at `lo` cover every position below 2²¹ from
// `2048 + 1024` rows per rotary pair — 768 KiB at 32 pairs, resident in L2 —
// where a table with one row per position is a gigabyte at a million tokens
// (`docs/progressive_yarn.md` §5).
//
// Layout (built on the host by `models::rope_schedule::table::build`, the same
// as DeepSeek's latent table in `latent_common.cuh`): `float2 (sin, cos)`, the
// `HI` block `[2048][pairs]` first, then the `LO` block `[1024][pairs]`,
// frequency innermost. Only rotary pairs are stored; a frequency index at or
// past `pairs` is a pass-through pair, the identity, by select.
//
// The combine is spelled with `_rn` intrinsics so the compiler cannot contract
// it into FMAs: `table::lookup` is its host mirror, operation for operation,
// and a contraction on one side only would make the two disagree in the last
// bit.
//
// ## Rungs
//
// A model's schedule has one table per rung, all the same size, laid end to
// end, and a per-rung scale for Q's rotary pairs — YaRN's `m²`, exactly 1.0 on
// a rung without temperature (§4.2). A kernel takes the whole set as one
// `RopeRungs` launch argument and each sequence's rung from its own
// `SlotHeader.rope_rung`, so nothing rung-dependent is shared by two
// sequences' rows (§6).

#include <cuda_runtime.h>
#include <stdint.h>

#define ROPE_F_LO_BITS 10
#define ROPE_F_LO_DIM 1024
#define ROPE_F_HI_DIM 2048
/// Rows of one rung's table: the `HI` block then the `LO` block.
#define ROPE_F_ROWS (ROPE_F_HI_DIM + ROPE_F_LO_DIM)

/// Every rung of a model's schedule. Passed by value; mirrors the Rust
/// `RopeRungsFfi` field for field.
struct RopeRungs {
    const float2* tables;   // [n_rungs][ROPE_F_ROWS · pairs]
    const float* q_scale;   // [n_rungs], m² per rung
    uint32_t n_rungs;
    uint32_t pairs;         // rotary pairs P
};
static_assert(sizeof(RopeRungs) == 24, "RopeRungs mirrors the 24-byte RopeRungsFfi");

/// One sequence's rotation: its rung's table, and the scale this view applies
/// to rotary pairs — 1 for K, the rung's `m²` for Q (`RopeView::for_q`).
struct RopeView {
    const float2* tab;
    int pairs;
    float scale;
    float q_scale;

    __device__ __forceinline__ RopeView for_q() const {
        RopeView v = *this;
        v.scale = q_scale;
        return v;
    }
};

/// The view for rung `rung`. A rung past the schedule traps rather than read
/// the next table — or past the allocation.
__device__ __forceinline__ RopeView rope_view(const RopeRungs& r, uint32_t rung)
{
    if (rung >= r.n_rungs) __trap();
    RopeView v;
    v.tab = r.tables + (size_t)rung * ROPE_F_ROWS * r.pairs;
    v.pairs = (int)r.pairs;
    v.scale = 1.f;
    v.q_scale = __ldg(r.q_scale + rung);
    return v;
}

/// `(sin, cos)` of pair `i < pairs` at `pos`. `pos` is clamped to the table's
/// reach, so a position past it reads the last `HI` row rather than past the
/// allocation.
__device__ __forceinline__ float2 rope_f_lookup(
    const float2* __restrict__ tab, int pairs, int pos, int i)
{
    int hi = pos >> ROPE_F_LO_BITS;
    hi = hi < ROPE_F_HI_DIM - 1 ? hi : ROPE_F_HI_DIM - 1;
    const int lo = pos & (ROPE_F_LO_DIM - 1);
    const float2 h = __ldg(tab + hi * pairs + i);
    const float2 l = __ldg(tab + (ROPE_F_HI_DIM + lo) * pairs + i);
    float2 r;
    r.x = __fadd_rn(__fmul_rn(h.x, l.y), __fmul_rn(h.y, l.x));   // sin
    r.y = __fsub_rn(__fmul_rn(h.y, l.y), __fmul_rn(h.x, l.x));   // cos
    return r;
}

/// `cos` and `sin` of frequency `f` at `pos` under `v`, scaled by `v.scale`;
/// the identity for a pass-through frequency (`f >= pairs`). The same
/// frequency-indexed contract the paged kernels have always read: the
/// half-split pairing `(d, d + HD/2)` asks for frequency `d`, the interleaved
/// pairing `(2i, 2i + 1)` for `i`.
__device__ __forceinline__ void rope_cs_at(
    const RopeView& v, int pos, int f, float& c, float& s)
{
    if (f >= v.pairs) {
        c = 1.f;
        s = 0.f;
        return;
    }
    const float2 sc = rope_f_lookup(v.tab, v.pairs, pos, f);
    c = sc.y * v.scale;
    s = sc.x * v.scale;
}

/// `cos` and `sin` of frequency `f` one position on — `LO` row 1, the unit
/// step a kernel walks consecutive tokens by — unscaled; the identity for a
/// pass-through frequency.
__device__ __forceinline__ void rope_cs_step(const RopeView& v, int f, float& c, float& s)
{
    if (f >= v.pairs) {
        c = 1.f;
        s = 0.f;
        return;
    }
    const float2 sc = __ldg(v.tab + (ROPE_F_HI_DIM + 1) * v.pairs + f);
    c = sc.y;
    s = sc.x;
}

/// Rotate NeoX pair `(lo, hi)` by `sc = (sin, cos)`.
__device__ __forceinline__ void rope_f_rotate(float& lo, float& hi, float2 sc)
{
    const float a = lo * sc.y - hi * sc.x;
    const float b = hi * sc.y + lo * sc.x;
    lo = a;
    hi = b;
}
