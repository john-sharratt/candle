//! Qwen3.8-Flash-Next's RoPE schedule (`docs/progressive_yarn.md` §2).
//!
//! 64 rotary dims at θ 1e7, trained to 262,144 positions, then the YaRN
//! factors vLLM's Flash-Next recipe publishes: 2.0 to 524,288 and 4.0 to
//! 1,000,000. The attention and the QSA indexer rotate with the same schedule
//! — the reference hands the indexer the attention's own rotary embedding, `m`
//! included (§12).
//!
//! Its tables' `LO` rows take the reference's f32 angle product
//! ([`AngleArithmetic::F32Product`]), the arithmetic its KV-compression row was
//! derived on, as the rest of the lineage does (`qwen35::rope`).

use crate::models::rope_schedule::{AngleArithmetic, RopeSchedule, Rung};

use super::config::Qwen4ExpConfig;

/// The trained window.
pub const FLASH_NEXT_L0: usize = 262_144;

/// The longest supported reach.
pub const FLASH_NEXT_MAX: usize = 1_000_000;

/// The schedule over `cfg`'s rotary width and base.
pub fn flash_next_schedule(cfg: &Qwen4ExpConfig) -> candle::Result<RopeSchedule> {
    flash_next(cfg.rope_dim, cfg.rope_theta)
}

/// The schedule over a rotary width of `rope_dim` at base `theta`.
fn flash_next(rope_dim: usize, theta: f32) -> candle::Result<RopeSchedule> {
    RopeSchedule::yarn(
        rope_dim,
        theta,
        FLASH_NEXT_L0,
        vec![
            Rung {
                ceiling: FLASH_NEXT_L0,
                factor: 1.0,
            },
            Rung {
                ceiling: 524_288,
                factor: 2.0,
            },
            Rung {
                ceiling: FLASH_NEXT_MAX,
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

    /// Flash-Next rotates below 2¹⁰ exactly as its former per-position table
    /// did: the reference's f32 product.
    #[test]
    fn flash_next_takes_the_reference_angle() {
        let s = flash_next(64, 1e7).unwrap();
        assert_eq!(s.lo_angle(), AngleArithmetic::F32Product);
        assert_eq!(s.ceilings(), vec![FLASH_NEXT_L0, 524_288, FLASH_NEXT_MAX]);
    }
}
