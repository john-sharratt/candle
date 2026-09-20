//! The factored RoPE table: two small tables joined by the angle-addition
//! identity.
//!
//! RoPE rotates rotary pair `i` at position `p` by the angle `p·ω_i`, and that
//! angle is linear in `p`. Writing `p = h·2¹⁰ + l` splits it into two angles,
//! and a rotation by a sum is the product of the two rotations:
//!
//! ```text
//! sin(α_h + α_l) = sin α_h · cos α_l + cos α_h · sin α_l
//! cos(α_h + α_l) = cos α_h · cos α_l − sin α_h · sin α_l
//! ```
//!
//! So a `HI` block of `(sin, cos)` at `h·2¹⁰` and a `LO` block at `l` reproduce
//! every position below [`ROPE_REACH`] from `2048 + 1024` rows instead of two
//! million. At 32 rotary pairs that is 768 KiB, small enough to stay in L2.
//!
//! **The layout is DeepSeek's latent table's** (`latent_common.cuh`): `float2`
//! entries of `(sin, cos)`, the `HI` block first, frequency innermost. The
//! device lookup is `rope/rope_table.cuh`'s `rope_f_lookup`, and [`lookup`] here
//! is its host mirror, operation for operation.
//!
//! **Precision.** Every `HI` entry is the f64 sine or cosine of an exact f64
//! angle (`pos` has at most 21 bits and `ω` is an f32, so `pos·ω` fits in 53),
//! rounded once to f32. The `LO` rows use the schedule's
//! [`AngleArithmetic`](super::angle::AngleArithmetic): exact the same way, or
//! the reference's f32 product for a model calibrated on it. Either way the
//! combine adds a bounded number of roundings, so the error does not grow with
//! position — unlike an f32 angle `pos·ω` over the whole position, whose
//! rounding alone is `0.5 ulp(pos·ω)`, about 0.008 rad at 200K on the fastest
//! pair. [`LOOKUP_ERROR_BOUND`] is the bound for exact rows, derived rather
//! than tuned.

use super::angle::AngleArithmetic;

/// Low bits of a position: the `LO` block's row index.
pub const ROPE_LO_BITS: usize = 10;
/// Rows in the `LO` block.
pub const ROPE_LO_DIM: usize = 1 << ROPE_LO_BITS;
/// Rows in the `HI` block.
pub const ROPE_HI_DIM: usize = 2048;
/// Positions the table covers: every position below this is exact.
pub const ROPE_REACH: usize = ROPE_HI_DIM << ROPE_LO_BITS;

/// Worst-case absolute error of one [`lookup`] against the f64 sine or cosine.
///
/// Each entry is within one rounding of its f64 truth: `u = 2⁻²⁴`, relative,
/// of a value no larger than one. A product of two such entries is then within
/// `2u` of the true product, plus `u` for rounding the product itself; the sum
/// or difference of two products carries both, plus one more rounding for the
/// result. Seven units.
pub const LOOKUP_ERROR_BOUND: f32 = 7.0 * f32::EPSILON / 2.0;

/// Plain RoPE inverse frequencies over a rotary width of `rope_dim`.
///
/// Computed exactly as every table in the engine has computed them — in f32,
/// `1 / θ^(2i/d)` — so a model rotating from this table uses the frequencies it
/// always has. Only the angle's precision changes.
pub fn plain_inv_freq(rope_dim: usize, theta: f32) -> Vec<f32> {
    (0..rope_dim / 2)
        .map(|i| 1f32 / theta.powf(2.0 * i as f32 / rope_dim as f32))
        .collect()
}

/// The table for one set of frequencies, as `f32`s: `(sin, cos)` pairs, `HI`
/// rows then `LO` rows, frequency innermost. `HI` rows are exact; `LO` rows
/// take `lo`'s arithmetic.
pub fn build(inv_freq: &[f32], lo: AngleArithmetic) -> Vec<f32> {
    let pairs = inv_freq.len();
    let mut out = Vec::with_capacity((ROPE_HI_DIM + ROPE_LO_DIM) * pairs * 2);
    let rows = (0..ROPE_HI_DIM)
        .map(|h| (h << ROPE_LO_BITS, AngleArithmetic::Exact))
        .chain((0..ROPE_LO_DIM).map(|l| (l, lo)));
    for (pos, arith) in rows {
        for &w in inv_freq {
            let (s, c) = arith.sin_cos(pos, w);
            out.push(s);
            out.push(c);
        }
    }
    out
}

/// `(sin, cos)` of pair `pair` at `pos`, from a table [`build`] made.
///
/// The host mirror of `rope_f_lookup`: the same two entries, the same four
/// products and the same two sums, each rounded to f32 in the same order.
/// Rust never contracts a multiply and an add, which is what makes the mirror
/// exact against a kernel that spells every operation `_rn`.
pub fn lookup(table: &[f32], pairs: usize, pos: usize, pair: usize) -> (f32, f32) {
    assert!(
        pos < ROPE_REACH,
        "rope position {pos} is past the table's reach {ROPE_REACH}"
    );
    let h = pos >> ROPE_LO_BITS;
    let l = pos & (ROPE_LO_DIM - 1);
    let hi = (h * pairs + pair) * 2;
    let lo = ((ROPE_HI_DIM + l) * pairs + pair) * 2;
    let (hs, hc) = (table[hi], table[hi + 1]);
    let (ls, lc) = (table[lo], table[lo + 1]);
    let s = hs * lc + hc * ls;
    let c = hc * lc - hs * ls;
    (s, c)
}

/// Lanes in a step table — one warp.
pub const STEP_LANES: usize = 32;

/// The widest block stride a step table is built for: the QSA index pools at
/// most four tokens to a block.
pub const MAX_STEP: usize = 4;

/// A step table: `(sin, cos)` at `L · step` for every lane `L < 32` and every
/// pair, laid out `[pair][lane]`, in `lo`'s arithmetic — the table's own `LO`
/// rows at those positions, so a step composes exactly as a lookup would.
///
/// A scorer whose 32 lanes read 32 consecutive block keys finds them at
/// `W + L · step`, where `W` is the warp's first key's position. The rotation
/// at each lane is then the warp's rotation at `W` composed with this table's
/// entry at `L` — one shared-memory read per lane per pair, instead of two
/// table rows from L2. Pair-major, lane-minor, so a warp reading one pair
/// touches 32 consecutive entries.
pub fn build_steps(inv_freq: &[f32], step: usize, lo: AngleArithmetic) -> Vec<f32> {
    let mut out = Vec::with_capacity(inv_freq.len() * STEP_LANES * 2);
    for &w in inv_freq {
        for lane in 0..STEP_LANES {
            let (s, c) = lo.sin_cos(lane * step, w);
            out.push(s);
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn freqs() -> Vec<f32> {
        // The hybrid lineage's rotary geometry: 64 rotary dims at θ = 1e7.
        plain_inv_freq(64, 1e7)
    }

    /// The layout is DeepSeek's: `HI` rows first, `(sin, cos)` per pair,
    /// frequency innermost, and 768 KiB at 32 pairs.
    #[test]
    fn the_table_has_the_latent_layout_and_size() {
        let t = build(&freqs(), AngleArithmetic::Exact);
        assert_eq!(t.len(), (2048 + 1024) * 32 * 2);
        assert_eq!(t.len() * 4, 768 * 1024);
        // HI row 1 is position 1024; LO row 1 is position 1.
        let w = freqs()[3] as f64;
        let hi1 = (32 + 3) * 2;
        assert_eq!(t[hi1], (1024.0 * w).sin() as f32);
        assert_eq!(t[hi1 + 1], (1024.0 * w).cos() as f32);
        let lo1 = ((2048 + 1) * 32 + 3) * 2;
        assert_eq!(t[lo1], w.sin() as f32);
        assert_eq!(t[lo1 + 1], w.cos() as f32);
    }

    /// Position zero is the identity rotation, exactly.
    #[test]
    fn position_zero_is_exactly_the_identity() {
        for arith in [AngleArithmetic::Exact, AngleArithmetic::F32Product] {
            let t = build(&freqs(), arith);
            for pair in 0..32 {
                assert_eq!(lookup(&t, 32, 0, pair), (0.0, 1.0), "pair {pair}");
            }
        }
    }

    /// Under the reference arithmetic, every position below 2¹⁰ looks up
    /// exactly the lineage's former per-position entry — `pos as f32 * ω`,
    /// f32 `sin` and `cos` — because the `HI` row there is the identity.
    #[test]
    fn f32_product_rows_reproduce_the_reference_below_the_lo_span() {
        let f = freqs();
        let t = build(&f, AngleArithmetic::F32Product);
        for pos in [1usize, 2, 17, 511, 700, 1023] {
            for (pair, &w) in f.iter().enumerate() {
                let a = pos as f32 * w;
                assert_eq!(
                    lookup(&t, 32, pos, pair),
                    (a.sin(), a.cos()),
                    "pos {pos} pair {pair}"
                );
            }
        }
    }

    /// The `HI` rows are exact under either arithmetic, so the two tables
    /// differ only in the `LO` block.
    #[test]
    fn hi_rows_are_exact_under_either_arithmetic() {
        let f = freqs();
        let exact = build(&f, AngleArithmetic::Exact);
        let refr = build(&f, AngleArithmetic::F32Product);
        let hi_len = ROPE_HI_DIM * 32 * 2;
        assert_eq!(exact[..hi_len], refr[..hi_len]);
        assert_ne!(exact[hi_len..], refr[hi_len..]);
    }

    /// Past the `LO` span the reference table's error stays bounded: an exact
    /// `HI` term plus one reference `LO` term, whose angle is within
    /// `0.5 ulp(1023·ω)` of exact — never the whole position's rounding.
    #[test]
    fn f32_product_error_is_bounded_at_depth() {
        let f = freqs();
        let t = build(&f, AngleArithmetic::F32Product);
        // 0.5 ulp of an angle below 1024 is at most 2⁻¹⁴ rad.
        let lo_angle = 2f32.powi(-14);
        for &pos in &[1024usize, 200_000, 1_048_575, ROPE_REACH - 1] {
            for (pair, &w) in f.iter().enumerate() {
                let (s, c) = lookup(&t, 32, pos, pair);
                let a = pos as f64 * w as f64;
                let bound = LOOKUP_ERROR_BOUND + lo_angle;
                assert!(
                    (s as f64 - a.sin()).abs() as f32 <= bound,
                    "sin {pos}/{pair}"
                );
                assert!(
                    (c as f64 - a.cos()).abs() as f32 <= bound,
                    "cos {pos}/{pair}"
                );
            }
        }
    }

    /// The frequencies are today's, bit for bit — only the angle's precision
    /// changes.
    #[test]
    fn plain_frequencies_are_the_ones_every_table_already_used() {
        let f = plain_inv_freq(64, 1e7);
        assert_eq!(f.len(), 32);
        for (i, &w) in f.iter().enumerate() {
            assert_eq!(w, 1f32 / 1e7f32.powf(2.0 * i as f32 / 64.0));
        }
        assert_eq!(f[0], 1.0);
    }

    /// Every lookup across the reach is within the derived bound of the f64
    /// truth: every `HI` row, a spread of `LO` rows including both ends, every
    /// pair.
    #[test]
    fn every_lookup_is_within_the_derived_bound() {
        let f = freqs();
        let t = build(&f, AngleArithmetic::Exact);
        let mut worst = 0f32;
        for h in 0..ROPE_HI_DIM {
            for &l in &[0usize, 1, 2, 511, 512, 1022, 1023] {
                let pos = (h << ROPE_LO_BITS) + l;
                for (pair, &w) in f.iter().enumerate() {
                    let (s, c) = lookup(&t, 32, pos, pair);
                    let a = pos as f64 * w as f64;
                    let es = (s as f64 - a.sin()).abs() as f32;
                    let ec = (c as f64 - a.cos()).abs() as f32;
                    worst = worst.max(es).max(ec);
                }
            }
        }
        assert!(
            worst <= LOOKUP_ERROR_BOUND,
            "worst lookup error {worst:e} exceeds the derived bound {LOOKUP_ERROR_BOUND:e}"
        );
    }

    /// A step table's entry at lane `L` is the rotation at `L · step`, and
    /// composing it with the factored lookup at `W` lands within the derived
    /// bound (plus one composition) of the f64 rotation at `W + L · step`.
    #[test]
    fn a_step_table_composes_with_a_lookup_to_the_later_position() {
        let f = freqs();
        let t = build(&f, AngleArithmetic::Exact);
        for step in 1..=MAX_STEP {
            let s = build_steps(&f, step, AngleArithmetic::Exact);
            assert_eq!(s.len(), 32 * STEP_LANES * 2);
            for &w_pos in &[0usize, 1000, 1_048_000, ROPE_REACH - 200] {
                for lane in [0usize, 1, 17, 31] {
                    for pair in [0usize, 5, 31] {
                        let (ws, wc) = lookup(&t, 32, w_pos, pair);
                        let e = (pair * STEP_LANES + lane) * 2;
                        let (ls, lc) = (s[e], s[e + 1]);
                        let sin = ws * lc + wc * ls;
                        let cos = wc * lc - ws * ls;
                        let a = (w_pos + lane * step) as f64 * f[pair] as f64;
                        let bound = LOOKUP_ERROR_BOUND + 5.0 * f32::EPSILON;
                        assert!(
                            (sin as f64 - a.sin()).abs() as f32 <= bound,
                            "sin at {w_pos}+{lane}·{step}"
                        );
                        assert!(
                            (cos as f64 - a.cos()).abs() as f32 <= bound,
                            "cos at {w_pos}+{lane}·{step}"
                        );
                    }
                }
            }
        }
    }

    /// A position past the reach is refused rather than read off the end.
    #[test]
    #[should_panic(expected = "past the table's reach")]
    fn a_position_past_the_reach_is_refused() {
        let t = build(&freqs(), AngleArithmetic::Exact);
        let _ = lookup(&t, 32, ROPE_REACH, 0);
    }
}
