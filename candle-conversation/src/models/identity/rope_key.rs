//! The RoPE schedule as part of a model's identity.
//!
//! A K/V block is valid only under the schedule its layers attended by: a
//! different preset or a different rung table changes every key a later layer
//! computed, while every shape still matches. Floats are written as their bit
//! patterns, so two schedules are the same key exactly when they rotate
//! identically.

use crate::models::RopePreset;

/// The canonical text of a RoPE schedule.
pub fn rope_key(preset: &RopePreset) -> String {
    match preset {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3s_schedule_is_this_text() {
        assert_eq!(
            rope_key(&RopePreset::qwen3()),
            "progressive-yarn theta=49742400 l0=32768 \
             rungs=32768x3f800000,65536x40000000,131072x40800000"
        );
    }

    #[test]
    fn a_preset_without_rungs_is_its_name() {
        assert_eq!(rope_key(&RopePreset::Lineage), "lineage");
        assert_eq!(rope_key(&RopePreset::FileStated), "file-stated");
    }
}
