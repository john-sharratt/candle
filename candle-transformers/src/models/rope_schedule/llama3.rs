//! Llama3 RoPE scaling: the published-parameter formula, and the frequencies a
//! GGUF states directly as `rope_freqs.weight`.
//!
//! llama.cpp's converter writes Llama3 scaling only as that tensor — one
//! divisor per rotary pair — never as metadata keys. A file that carries it is
//! stating its own scaling, so the tensor is the source; the formula is what
//! the tests check the tensor against, and what the loader builds from a
//! file's `llama.rope.scaling.*` keys when it has those instead
//! (`docs/progressive_yarn.md` §9).

use crate::models::llama::{Llama3RopeConfig, Llama3RopeType};

/// The RoPE inverse frequencies for Llama, `head_dim / 2` of them.
///
/// Plain `1 / θ^(i/d)` without scaling; with Llama3 scaling, the low-frequency
/// pairs are divided by the factor, the high-frequency pairs kept, and the band
/// between blended.
pub fn llama_inv_freq(
    head_dim: usize,
    freq_base: f32,
    rope_scaling: Option<Llama3RopeConfig>,
) -> Vec<f32> {
    let default_inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();

    match rope_scaling {
        None
        | Some(Llama3RopeConfig {
            rope_type: Llama3RopeType::Default,
            ..
        }) => default_inv_freq,
        Some(rope_scaling) => {
            use std::f32::consts::PI;

            let low_freq_wavelen =
                rope_scaling.original_max_position_embeddings as f32 / rope_scaling.low_freq_factor;
            let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                / rope_scaling.high_freq_factor;

            default_inv_freq
                .into_iter()
                .map(|freq| {
                    let wavelen = 2. * PI / freq;
                    if wavelen < high_freq_wavelen {
                        freq
                    } else if wavelen > low_freq_wavelen {
                        freq / rope_scaling.factor
                    } else {
                        let smooth = (rope_scaling.original_max_position_embeddings as f32
                            / wavelen
                            - rope_scaling.low_freq_factor)
                            / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                        (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                    }
                })
                .collect::<Vec<_>>()
        }
    }
}

/// The frequencies a `rope_freqs.weight` tensor states: the plain frequency of
/// each pair divided by that pair's factor, as llama.cpp applies it.
pub fn from_rope_freqs(
    head_dim: usize,
    freq_base: f32,
    factors: &[f32],
) -> candle::Result<Vec<f32>> {
    let plain = llama_inv_freq(head_dim, freq_base, None);
    if factors.len() != plain.len() {
        candle::bail!(
            "rope_freqs.weight has {} factors for {} rotary pairs",
            factors.len(),
            plain.len()
        );
    }
    Ok(plain.iter().zip(factors).map(|(f, d)| f / d).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hermes_3b() -> Llama3RopeConfig {
        Llama3RopeConfig {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
            rope_type: Llama3RopeType::Llama3,
        }
    }

    /// The tensor's divisors reproduce the formula when they are the formula's
    /// own ratios — the round trip every Hermes file must satisfy.
    #[test]
    fn a_tensor_of_the_formulas_ratios_reproduces_the_formula() {
        let plain = llama_inv_freq(128, 5e5, None);
        let scaled = llama_inv_freq(128, 5e5, Some(hermes_3b()));
        let factors: Vec<f32> = plain.iter().zip(&scaled).map(|(p, s)| p / s).collect();
        let back = from_rope_freqs(128, 5e5, &factors).unwrap();
        for (i, (b, s)) in back.iter().zip(&scaled).enumerate() {
            assert!(
                (b - s).abs() <= s * 2.0 * f32::EPSILON,
                "pair {i}: {b} vs {s}"
            );
        }
    }

    /// Llama3 scaling divides the low-frequency pairs at every position: the
    /// fastest pair is untouched and the slowest is divided by the full factor.
    #[test]
    fn fastest_kept_slowest_divided() {
        let plain = llama_inv_freq(128, 5e5, None);
        let scaled = llama_inv_freq(128, 5e5, Some(hermes_3b()));
        assert_eq!(scaled[0], plain[0]);
        assert_eq!(scaled[63], plain[63] / 32.0);
    }

    /// A tensor of the wrong width is refused rather than zipped short.
    #[test]
    fn a_tensor_of_the_wrong_width_is_refused() {
        assert!(from_rope_freqs(128, 5e5, &[1.0; 63]).is_err());
    }
}
