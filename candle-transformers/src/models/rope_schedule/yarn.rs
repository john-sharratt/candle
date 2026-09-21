//! The YaRN frequency transform and its attention temperature.
//!
//! The reference is vLLM's `YaRNScalingRotaryEmbedding`, which the Qwen cards
//! point at, and DeepSeek-V4's `precompute_freqs_cis`, which is the same
//! transform. Over a rotary width `d` with factor `s` and trained window `L₀`:
//!
//! ```text
//! base_i   = θ^(2i/d)
//! extrap_i = 1 / base_i
//! interp_i = 1 / (s · base_i)
//! cdim(r)  = d · ln(L₀ / (2π r)) / (2 ln θ)
//! low      = max(floor(cdim(β_fast)), 0)
//! high     = min(ceil (cdim(β_slow)), d − 1)       (high += 0.001 if low == high)
//! ramp_i   = clamp((i − low) / (high − low), 0, 1)
//! inv_i    = interp_i · ramp_i + extrap_i · (1 − ramp_i)
//! m        = 1 if s ≤ 1, else 0.1 · ln(s) + 1
//! ```
//!
//! Computed in f64 throughout; a caller narrows to f32 once.

/// The YaRN-adjusted inverse frequencies, `rope_dim / 2` of them.
///
/// `original_seq_len == 0` disables YaRN and returns the plain frequencies
/// `1 / θ^(2i/d)`.
pub fn yarn_freqs(
    rope_dim: usize,
    theta: f64,
    original_seq_len: usize,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> Vec<f64> {
    let half = rope_dim / 2;
    let mut freqs: Vec<f64> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f64 / rope_dim as f64))
        .collect();
    if original_seq_len > 0 {
        let (low, high) = correction_range(beta_fast, beta_slow, rope_dim, theta, original_seq_len);
        for (i, f) in freqs.iter_mut().enumerate() {
            // smooth = 1 − ramp: the share of the raw frequency kept.
            let smooth = 1.0 - ramp(low, high, i);
            *f = *f / factor * (1.0 - smooth) + *f * smooth;
        }
    }
    freqs
}

/// The rotary dimension at which a frequency completes `num_rotations` turns
/// over `max_seq` positions.
fn correction_dim(num_rotations: f64, dim: usize, base: f64, max_seq: usize) -> f64 {
    dim as f64 * ((max_seq as f64) / (num_rotations * 2.0 * std::f64::consts::PI)).ln()
        / (2.0 * base.ln())
}

/// `(low, high)`: the pair indices between which YaRN blends from the raw
/// frequencies to the interpolated ones.
pub fn correction_range(
    low_rot: f64,
    high_rot: f64,
    dim: usize,
    base: f64,
    max_seq: usize,
) -> (f64, f64) {
    let low = correction_dim(low_rot, dim, base, max_seq).floor();
    let high = correction_dim(high_rot, dim, base, max_seq).ceil();
    (low.max(0.0), high.min((dim - 1) as f64))
}

/// `linear_ramp_factor(min, max, dim)[i]`, clamped to `[0, 1]`.
fn ramp(min: f64, max: f64, i: usize) -> f64 {
    let max = if min == max { max + 0.001 } else { max };
    (((i as f64) - min) / (max - min)).clamp(0.0, 1.0)
}

/// YaRN's attention temperature `m` for a factor `s`: `0.1·ln(s) + 1`, and
/// exactly 1 at `s ≤ 1`.
///
/// vLLM scales its cos/sin cache by `m`, so Q and K each gain `m` and the
/// rotary logits `m²`. The engine applies `m²` to Q's rotary pairs alone
/// (`docs/progressive_yarn.md` §4.2).
pub fn mscale(factor: f64) -> f64 {
    if factor <= 1.0 {
        1.0
    } else {
        0.1 * factor.ln() + 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blend bounds `docs/progressive_yarn.md` §4.1 tabulates, worked by
    /// hand: Qwen3 (d 128, θ 1e6, L₀ 32,768) and the hybrid lineage (d 64,
    /// θ 1e7, L₀ 262,144), both at β = 32/1.
    #[test]
    fn the_blend_bounds_are_the_tabulated_ones() {
        assert_eq!(correction_range(32.0, 1.0, 128, 1e6, 32_768), (23.0, 40.0));
        assert_eq!(correction_range(32.0, 1.0, 64, 1e7, 262_144), (14.0, 22.0));
    }

    /// Pairs below `low` keep their raw frequency, pairs from `high` on are
    /// divided by the factor exactly, and the pairs between blend.
    #[test]
    fn kept_blended_and_interpolated_pairs() {
        let plain = yarn_freqs(64, 1e7, 0, 1.0, 32.0, 1.0);
        let f = yarn_freqs(64, 1e7, 262_144, 4.0, 32.0, 1.0);
        assert_eq!(f.len(), 32);
        for i in 0..=14 {
            assert_eq!(f[i], plain[i], "pair {i} is kept");
        }
        for i in 15..22 {
            assert!(f[i] < plain[i] && f[i] > plain[i] / 4.0, "pair {i} blends");
        }
        for i in 22..32 {
            assert_eq!(
                f[i],
                plain[i] / 4.0 * 1.0 + plain[i] * 0.0,
                "pair {i} is interpolated"
            );
        }
        // Pair 18 is (18 − 14) / (22 − 14) = half way.
        assert_eq!(f[18], plain[18] / 4.0 * 0.5 + plain[18] * 0.5);
    }

    /// `m` at the two published factors.
    #[test]
    fn the_temperature_at_the_published_factors() {
        assert_eq!(mscale(1.0), 1.0);
        assert_eq!(mscale(0.5), 1.0);
        assert_eq!(mscale(2.0), 0.1 * 2f64.ln() + 1.0);
        assert!((mscale(2.0) - 1.069_314_718).abs() < 1e-9);
        assert!((mscale(4.0) - 1.138_629_436).abs() < 1e-9);
    }

    /// DeepSeek-V4's compressing-layer frequencies are the ones its kernels
    /// have always used: raw pairs 0..low untouched, the transform unchanged
    /// by the move.
    #[test]
    fn deepseek_frequencies_are_unchanged() {
        let f = yarn_freqs(64, 160_000.0, 65_536, 16.0, 32.0, 1.0);
        let (low, high) = correction_range(32.0, 1.0, 64, 160_000.0, 65_536);
        let plain = yarn_freqs(64, 160_000.0, 0, 1.0, 32.0, 1.0);
        for i in 0..32 {
            let smooth = 1.0 - ramp(low, high, i);
            assert_eq!(f[i], plain[i] / 16.0 * (1.0 - smooth) + plain[i] * smooth);
        }
        assert_eq!(f[0], 1.0);
    }
}
