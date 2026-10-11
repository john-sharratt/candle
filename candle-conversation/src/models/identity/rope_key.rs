//! The RoPE schedule as part of a model's identity.
//!
//! Keys are stored rotated, so a K/V block is valid only under the schedule
//! that rotated it: a different preset, a different rung table or a different
//! minimum YaRN factor makes every stored key wrong while every shape still
//! matches. Floats are written as their bit patterns, so two schedules are the
//! same key exactly when they rotate identically.

use crate::models::RopePreset;

/// The canonical text of a RoPE schedule and its minimum YaRN factor.
pub fn rope_key(preset: &RopePreset, min_factor: Option<f32>) -> String {
    let schedule = match preset {
        RopePreset::FileStated => "file-stated".to_string(),
        RopePreset::Lineage => "lineage".to_string(),
        RopePreset::ProgressiveYarn { theta, l0, rungs } => {
            let rungs: Vec<String> = rungs
                .iter()
                .map(|r| format!("{}x{:08x}", r.ceiling, r.factor.to_bits()))
                .collect();
            format!(
                "progressive-yarn theta={:08x} l0={l0} rungs={}",
                theta.to_bits(),
                rungs.join(",")
            )
        }
    };
    match min_factor {
        Some(m) => format!("{schedule} min={:08x}", m.to_bits()),
        None => format!("{schedule} min=none"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3s_schedule_is_this_text() {
        assert_eq!(
            rope_key(&RopePreset::qwen3(), None),
            "progressive-yarn theta=49742400 l0=32768 \
             rungs=32768x3f800000,65536x40000000,131072x40800000 min=none"
        );
    }

    #[test]
    fn the_minimum_factor_is_part_of_it() {
        assert_eq!(
            rope_key(&RopePreset::Lineage, Some(2.0)),
            "lineage min=40000000"
        );
        assert_eq!(
            rope_key(&RopePreset::FileStated, None),
            "file-stated min=none"
        );
    }
}
