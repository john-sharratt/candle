//! The Qwen3.5 / 3.6 lineage's RoPE schedule (`docs/progressive_yarn.md` §2).
//!
//! One schedule for the whole lineage — Qwen3.5-0.8B and 9B, Qwen3.6-35B-A3B
//! and its AntiLoop fine-tune — because they share the RoPE: 64 rotary dims at
//! θ 1e7, trained to 262,144 positions. Past that a slot moves up to the YaRN
//! factors Qwen publishes for the lineage: 2.0 "if the typical context length
//! … is 524,288", and 4.0 for the full extension to 1,010,000. Rung 1 is the
//! trained RoPE exactly, so nothing changes below 262,144.
//!
//! Its tables' `LO` rows take the reference's f32 angle product
//! ([`AngleArithmetic::F32Product`]): the lineage's KV-compression rows were
//! derived on that arithmetic, and the exact angle moves their gates' edges.

use crate::models::rope_schedule::{AngleArithmetic, RopeSchedule, Rung};

use super::config::Qwen35Config;

/// The lineage's trained window.
pub const LINEAGE_L0: usize = 262_144;

/// The lineage's longest supported reach.
pub const LINEAGE_MAX: usize = 1_010_000;

/// The schedule over `cfg`'s rotary width and base.
pub fn lineage_schedule(cfg: &Qwen35Config) -> candle::Result<RopeSchedule> {
    lineage(cfg.rope_dim, cfg.rope_theta)
}

/// The schedule over a rotary width of `rope_dim` at base `theta`.
fn lineage(rope_dim: usize, theta: f32) -> candle::Result<RopeSchedule> {
    RopeSchedule::yarn(
        rope_dim,
        theta,
        LINEAGE_L0,
        vec![
            Rung {
                ceiling: LINEAGE_L0,
                factor: 1.0,
            },
            Rung {
                ceiling: 524_288,
                factor: 2.0,
            },
            Rung {
                ceiling: LINEAGE_MAX,
                factor: 4.0,
            },
        ],
        true,
    )
    .map(|s| s.with_lo_angle(AngleArithmetic::F32Product))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lineage rotates below 2¹⁰ exactly as its former per-position table
    /// did: the reference's f32 product.
    #[test]
    fn the_lineage_takes_the_reference_angle() {
        let s = lineage(64, 1e7).unwrap();
        assert_eq!(s.lo_angle(), AngleArithmetic::F32Product);
        assert_eq!(s.ceilings(), vec![LINEAGE_L0, 524_288, LINEAGE_MAX]);
    }
}
