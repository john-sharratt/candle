use crate::models::llama::{Llama3RopeConfig, Llama3RopeType};

/// Compute the RoPE inverse frequencies (`inv_freq`) for Llama.
///
/// For Llama-1/2 style RoPE, this is the standard $1 / \theta^{i/d}$ curve.
/// For Llama-3-style RoPE scaling, this applies the piecewise scaling described
/// by the `Llama3RopeConfig` metadata.
///
/// Returns a vector of length `head_dim/2`.
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

            let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                / rope_scaling.low_freq_factor;
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
                        let smooth = (rope_scaling.original_max_position_embeddings as f32 / wavelen
                            - rope_scaling.low_freq_factor)
                            / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                        (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                    }
                })
                .collect::<Vec<_>>()
        }
    }
}
