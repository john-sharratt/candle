//! Every preset's RoPE schedule agrees with its architecture.

use super::{Model, ModelArch, RopePreset};

/// A lineage preset names `Lineage`, which its loader carries; a GQA preset
/// names a schedule the builder can build, never `Lineage`.
#[test]
fn every_preset_names_a_schedule_its_arch_runs() {
    let mut all = Model::PRESETS.to_vec();
    all.push(Model::Qwen36_35B_A3B_AntiLoop_StyleTune);
    for m in all {
        let s = m.clone().preset_spec();
        let lineage = s.arch.file_rope() == RopePreset::Lineage;
        assert_eq!(
            s.rope == RopePreset::Lineage,
            lineage,
            "{m:?}: {:?} names {:?}",
            s.arch,
            s.rope
        );
    }
}

/// Qwen3 dense and the original 30B-A3B run Qwen's published progressive
/// schedule; the rest of the GQA presets run what their files state.
#[test]
fn the_qwen3_presets_are_progressive() {
    for m in [
        Model::Qwen3_8B_Q4,
        Model::Qwen3_8B_Q6,
        Model::Qwen3_14B_Q4,
        Model::Qwen3_14B_Q5,
        Model::Qwen3_14B_Q6,
        Model::Qwen3_30B_A3B_Q4,
        Model::Qwen3_30B_A3B_Q6,
    ] {
        assert_eq!(m.clone().preset_spec().rope, RopePreset::qwen3(), "{m:?}");
    }
    for m in [
        Model::Qwen2_0_5B,
        Model::Hermes3_3B_Q6,
        Model::Hermes3_70B_Q4,
    ] {
        assert_eq!(
            m.clone().preset_spec().rope,
            RopePreset::FileStated,
            "{m:?}"
        );
    }
}

/// A local GGUF takes its arch's loader schedule, or what the file states.
#[test]
fn a_local_file_runs_what_it_states() {
    assert_eq!(ModelArch::Qwen3.file_rope(), RopePreset::FileStated);
    assert_eq!(ModelArch::Llama.file_rope(), RopePreset::FileStated);
    assert_eq!(ModelArch::Qwen35Hybrid.file_rope(), RopePreset::Lineage);
    assert_eq!(ModelArch::Qwen4Exp.file_rope(), RopePreset::Lineage);
    assert_eq!(ModelArch::DeepSeekV4.file_rope(), RopePreset::Lineage);
}
