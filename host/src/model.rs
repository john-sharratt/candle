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

/// The model every tenant runs: the AntiLoop + StyleTune fine-tune of
/// Qwen3.6-35B-A3B — the AntiLoop trunk at Q4_K_M with StyleTune's
/// `output.weight` and an MTP gate donor. 21.7 GB, so it runs on every card in
/// the fleet: the expert cache streams what VRAM cannot hold.
pub const MODEL: Model = Model::Qwen36_35B_A3B_AntiLoop_StyleTune;

/// The model as `/v1/status` and `/v1/telemetry` name it.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelView {
    /// The checkpoint's file name.
    pub name: String,
    /// The repository it is fetched from.
    pub repo: String,
    /// The weights' size on disk, in bytes.
    pub bytes: u64,
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
            bytes: spec.model_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_conversation::models::{DialectType, ModelArch};

    /// The prescribed model is the hybrid lineage, speaks the Qwen3.5 dialect
    /// both tenants' prompts are written in, and is a named preset.
    #[test]
    fn the_prescribed_model_is_the_qwen35_hybrid_fine_tune() {
        let spec = MODEL.spec();
        assert!(matches!(spec.arch, ModelArch::Qwen35Hybrid));
        assert!(matches!(spec.chat_format, DialectType::Qwen35));
        assert_eq!(
            MODEL.override_key().as_deref(),
            Some("Qwen36_35B_A3B_AntiLoop_StyleTune")
        );
    }

    #[test]
    fn the_view_names_the_prescribed_checkpoint() {
        let v = ModelView::of_prescribed();
        assert_eq!(v.repo, MODEL.spec().model_repo);
        assert!(v.bytes > 0);
    }
}
