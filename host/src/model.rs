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

/// The model every tenant runs: Qwen3.8-Flash-Next — a `Q8_0` trunk with the
/// routed experts at the card's rung (`Q3_KO` under 64 GiB, `Q4_KO` above). It
/// runs on every card in the fleet: the expert cache streams what VRAM cannot
/// hold, and every rung is the same model, so a substrate written on one card
/// opens on another.
pub const MODEL: Model = Model::Qwen38_FlashNext;

/// The model as `/v1/status` and `/v1/telemetry` name it.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelView {
    /// The checkpoint's file name.
    pub name: String,
    /// The repository it is fetched from.
    pub repo: String,
    /// The weights' published size on disk, in bytes — `0` for a model built
    /// on this machine, which has no published length.
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

    /// The prescribed model is Flash-Next, speaks the Qwen3.5 dialect both
    /// tenants' prompts are written in, and is a named preset.
    #[test]
    fn the_prescribed_model_is_flash_next() {
        let spec = MODEL.spec();
        assert!(matches!(spec.arch, ModelArch::Qwen4Exp));
        assert!(matches!(spec.chat_format, DialectType::Qwen35));
        assert_eq!(MODEL.override_key().as_deref(), Some("Qwen38_FlashNext"));
    }

    #[test]
    fn the_view_names_the_prescribed_checkpoint() {
        let v = ModelView::of_prescribed();
        let spec = MODEL.spec();
        assert_eq!(v.repo, spec.model_repo);
        assert_eq!(v.name, spec.model_filename);
        assert_eq!(v.bytes, spec.model_bytes);
    }
}
