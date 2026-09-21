//! The arithmetic a table's `LO` rows compute their angle in.
//!
//! A lookup below 2¹⁰ returns its `LO` row unchanged (the `HI` row there is the
//! identity), so these rows ARE the rotation at every position a short context
//! reaches — and a model's calibrated behaviour belongs to the arithmetic it
//! was calibrated on, not to the most exact one available. Two arithmetics are
//! in use:
//!
//! * [`AngleArithmetic::Exact`] — the f64 angle `pos·ω` of the f32 frequency,
//!   its sine and cosine in f64, rounded once. What the GQA path's per-position
//!   table computed (`compute_rope_cs`), so those models rotate bit for bit as
//!   they always have.
//! * [`AngleArithmetic::F32Product`] — the reference's arithmetic: `pos · ω`
//!   rounded to f32, its sine and cosine in f32. What HF and llama.cpp compute,
//!   and what the hybrid lineage's own table computed
//!   (`RotaryLayout::rope_table`), which its KV-compression rows were derived
//!   against. Up to ~3e-5 rad from the exact angle at position 1023 on the
//!   fastest pair; replacing it with the exact angle moved first-token
//!   near-ties on those rows' gates.
//!
//! The `HI` rows are exact under both: past 2¹⁰ the angle is an exact `HI` term
//! plus one `LO` term, so its error stays bounded by the `LO` row's instead of
//! growing with position as a whole-position f32 product's does.

/// The arithmetic of a table's `LO` rows and step tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleArithmetic {
    /// f64 angle, f64 sine and cosine, one rounding.
    Exact,
    /// f32 product `pos · ω`, f32 sine and cosine — the reference's.
    F32Product,
}

impl AngleArithmetic {
    /// `(sin, cos)` of the angle at `pos` for frequency `w`.
    pub fn sin_cos(self, pos: usize, w: f32) -> (f32, f32) {
        match self {
            Self::Exact => {
                let (s, c) = (pos as f64 * w as f64).sin_cos();
                (s as f32, c as f32)
            }
            Self::F32Product => {
                let a = pos as f32 * w;
                (a.sin(), a.cos())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact is the f64 angle, rounded once.
    #[test]
    fn exact_is_the_f64_angle_rounded_once() {
        let w = 0.604_199_7f32;
        let a = 1023.0f64 * w as f64;
        assert_eq!(
            AngleArithmetic::Exact.sin_cos(1023, w),
            (a.sin() as f32, a.cos() as f32)
        );
    }

    /// F32Product is the lineage's former per-position table entry, operation
    /// for operation: `pos as f32 * ω`, then f32 `sin` and `cos`.
    #[test]
    fn f32_product_is_the_reference_arithmetic() {
        let w = 0.604_199_7f32;
        let a = 1023f32 * w;
        assert_eq!(
            AngleArithmetic::F32Product.sin_cos(1023, w),
            (a.sin(), a.cos())
        );
    }

    /// The two differ where the f32 product rounds — which is the whole
    /// reason a model's arithmetic is carried rather than assumed.
    #[test]
    fn the_two_arithmetics_differ_on_the_fastest_pair() {
        let differ = (1..1024).any(|p| {
            AngleArithmetic::Exact.sin_cos(p, 1.0) != AngleArithmetic::F32Product.sin_cos(p, 1.0)
        });
        assert!(differ);
    }

    /// Position zero is the identity under both.
    #[test]
    fn position_zero_is_the_identity_under_both() {
        for a in [AngleArithmetic::Exact, AngleArithmetic::F32Product] {
            assert_eq!(a.sin_cos(0, 1.0), (0.0, 1.0));
        }
    }
}
