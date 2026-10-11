//! The one model every tenant runs.
//!
//! Prescribed, not chosen: no candidate lists, no VRAM ladder, no per-daemon
//! choice (`docs/shared_model_host.md` §5). Each tenant declares what it needs
//! *of* a model ([`candle_conversation::Capabilities`]) and is refused at start
//! when this one lacks it. A substrate records the model that wrote it and
//! refuses any other, so changing this is a code change that every substrate
//! written under the previous model refuses until it is moved aside.

use candle_conversation::models::{Model, ModelSpec};
use serde::Serialize;

/// The model every tenant runs: Qwen3.8-Flash-Next with Q4_KO experts over a
/// Q8_0 trunk — 250 B total, ~13 B active, its engine artifact prepared locally
/// from the pinned upstream releases. It runs on every card in the fleet at this
/// one quant: the expert cache streams what VRAM cannot hold.
pub const MODEL: Model = Model::Qwen38_FlashNext_Q4KO;

/// The model as `/v1/status` and `/v1/telemetry` name it.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelView {
    /// The checkpoint's file name.
    pub name: String,
    /// The repository it is fetched from.
    pub repo: String,
    /// The weights' published size on disk, in bytes. `None` for an artifact
    /// prepared locally from its sources, whose size nothing publishes.
    pub bytes: Option<u64>,
}

impl ModelView {
    /// [`MODEL`], as its spec — after any local `models.override.yaml` —
    /// names it.
    pub fn of_prescribed() -> Self {
        Self::of(&MODEL.spec())
    }

    fn of(spec: &ModelSpec) -> Self {
        Self {
            name: spec.model_filename.clone(),
            repo: spec.model_repo.clone(),
            bytes: (!spec.prepared_from_source).then_some(spec.model_bytes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_conversation::models::{DialectType, ModelArch};

    /// The prescribed model is Flash-Next, speaks the Qwen3.5 dialect both
    /// tenants' prompts are written in, and is a named preset.
    #[test]
    fn the_prescribed_model_is_flash_next_at_q4_ko() {
        let spec = MODEL.spec();
        assert!(matches!(spec.arch, ModelArch::Qwen4Exp));
        assert!(matches!(spec.chat_format, DialectType::Qwen35));
        assert_eq!(
            MODEL.override_key().as_deref(),
            Some("Qwen38_FlashNext_Q4KO")
        );
    }

    #[test]
    fn the_view_names_the_prescribed_checkpoint() {
        let v = ModelView::of_prescribed();
        assert_eq!(v.repo, MODEL.spec().model_repo);
        assert_eq!(v.name, MODEL.spec().model_filename);
    }

    /// A published checkpoint states its size; an artifact prepared locally
    /// states none rather than a zero.
    #[test]
    fn only_a_published_checkpoint_states_its_size() {
        let published = ModelView::of(&Model::Qwen36_35B_A3B_AntiLoop_StyleTune.preset_spec());
        assert_eq!(published.bytes, Some(21_713_463_520));
        let prepared = ModelView::of(&Model::Qwen38_FlashNext_Q4KO.preset_spec());
        assert_eq!(prepared.bytes, None);
    }
}
