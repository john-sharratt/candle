//! A model as the host reports it.
//!
//! The host prescribes no model: each tenant names the one it runs
//! ([`crate::Tenant::model`], `docs/parasite_model.md` §4), checks it against
//! what it needs of a model ([`candle_conversation::Capabilities`]) at start,
//! and binds its substrate to it. A substrate records the model that wrote it
//! and refuses any other (`docs/shared_model_host.md` §4), so changing a
//! tenant's model is a code change that its substrate refuses until it is moved
//! aside.

use std::path::Path;

use candle_conversation::models::{Model, ModelSpec};
use serde::Serialize;

/// A model as `/v1/status` and `/v1/telemetry` name it.
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
    /// `model`, as its spec — after any local `models.override.yaml` — names
    /// it.
    pub fn of_model(model: &Model) -> Self {
        Self::of(&model.clone().spec())
    }

    fn of(spec: &ModelSpec) -> Self {
        Self {
            name: spec.model_filename.clone(),
            repo: spec.model_repo.clone(),
            bytes: spec.model_bytes,
        }
    }

    /// The model as a prompt names it: the checkpoint's file name without its
    /// extension.
    pub fn release(&self) -> &str {
        release_name(&self.name)
    }
}

/// `file_name` without its extension — `Qwen3.8-Flash-Next.gguf` is
/// `Qwen3.8-Flash-Next`; a name with none is itself.
fn release_name(file_name: &str) -> &str {
    Path::new(file_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(file_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_view_names_the_models_checkpoint() {
        let model = Model::Qwen38_FlashNext;
        let v = ModelView::of_model(&model);
        let spec = model.spec();
        assert_eq!(v.repo, spec.model_repo);
        assert_eq!(v.name, spec.model_filename);
        assert_eq!(v.bytes, spec.model_bytes);
    }

    /// A prompt names the release, not the file: the extension goes, and the
    /// dots inside the name stay.
    #[test]
    fn the_release_is_the_file_name_without_its_extension() {
        assert_eq!(
            release_name("Qwen3.8-Flash-Next.gguf"),
            "Qwen3.8-Flash-Next"
        );
        assert_eq!(
            release_name("Qwen3.8-Flash-Next-Q4_KOEXP-189dd518216f.gguf"),
            "Qwen3.8-Flash-Next-Q4_KOEXP-189dd518216f"
        );
        assert_eq!(release_name("model"), "model");
    }
}
