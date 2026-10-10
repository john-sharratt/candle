//! Binding a substrate to the model that writes it.
//!
//! Everything model-derived in a substrate — K/V, recurrent snapshots,
//! provenance signatures, index pages, content-addressed sections, token ids —
//! is valid only under the model that computed it. So a substrate records its
//! model's [`ModelKey`] and opens under that model alone: a different model is
//! refused before it writes or reads anything, and nothing is regenerated,
//! injected or tombstoned (`docs/shared_model_host.md` §4).
//!
//! A substrate written before keys carries the spec text that preceded them. It
//! is checked once against the running model — architecture, repository,
//! checkpoint name and the recorded tokenizer — and, if it matches, the key is
//! written over it and decides from then on.

use std::path::PathBuf;

use crate::models::identity::{ModelIdentity, ModelKey, RecordedModel};

use super::record::RecordType;
use super::{PersistenceError, Result, SharedSubstrate, SubstratePersistence};

/// What binding found, and what it wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelBinding {
    /// The substrate already records this key. Nothing written.
    Same,
    /// A new substrate: the key is written.
    New,
    /// A substrate written before keys whose spec matches the running model:
    /// the key is written over it.
    Adopted,
    /// The same model under a different description — moved repository or
    /// renamed file. The key is rewritten so the record names it as it is now.
    Redescribed,
}

/// The decision, made from what the record holds and nothing else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Decision {
    Keep,
    Write(ModelBinding),
    Refuse,
}

/// Decide whether a substrate recording `recorded` (and a tokenizer digest
/// `recorded_tokenizer`) opens under `running`.
pub(super) fn decide(
    recorded: Option<&RecordedModel>,
    recorded_tokenizer: Option<[u8; 32]>,
    running: &ModelKey,
) -> Decision {
    match recorded {
        None => Decision::Write(ModelBinding::New),
        Some(RecordedModel::Key(k)) if k == running => Decision::Keep,
        Some(RecordedModel::Key(k)) if k.same_model(running) => {
            Decision::Write(ModelBinding::Redescribed)
        }
        Some(RecordedModel::Key(_)) => Decision::Refuse,
        Some(RecordedModel::Legacy(spec)) => {
            let vocabulary_matches = recorded_tokenizer.is_none_or(|t| t == running.tokenizer);
            if spec.matches(running) && vocabulary_matches {
                Decision::Write(ModelBinding::Adopted)
            } else {
                Decision::Refuse
            }
        }
    }
}

impl SubstratePersistence {
    /// The model this substrate records, decoded. `None` on a substrate no
    /// model has written yet.
    pub fn recorded_model(&self) -> Option<std::result::Result<RecordedModel, String>> {
        self.model_record.as_deref().map(RecordedModel::decode)
    }

    /// Bind this substrate to `key`, or refuse with
    /// [`PersistenceError::WrongModel`].
    ///
    /// Appends a `ModelKey` record when the substrate records none, records the
    /// spec that preceded keys, or describes the same model differently. A
    /// read-only handle refuses exactly as a writable one does and writes
    /// nothing. The caller commits.
    pub fn bind_model(&mut self, key: &ModelKey) -> Result<ModelBinding> {
        let recorded = self
            .recorded_model()
            .transpose()
            .map_err(|e| PersistenceError::Corrupt(format!("{}: {e}", self.dir().display())))?;
        match decide(recorded.as_ref(), self.tokenizer_sha256, key) {
            Decision::Keep => Ok(ModelBinding::Same),
            Decision::Refuse => Err(PersistenceError::WrongModel {
                dir: self.dir().to_path_buf(),
                recorded: recorded.map(|r| r.describe()).unwrap_or_default(),
                running: key.describe(),
            }),
            Decision::Write(binding) => {
                if !self.is_read_only() {
                    let payload = key.encode();
                    self.append_record(RecordType::ModelKey, 0, 0, 0, 0, 0, &payload)?;
                    self.model_record = Some(payload);
                }
                Ok(binding)
            }
        }
    }
}

/// A substrate bound to the model that is about to run on it — the only form
/// in which an engine accepts one, so no engine can open a substrate its model
/// did not write.
#[derive(Clone, Debug)]
pub struct BoundSubstrate {
    shared: SharedSubstrate,
    key: ModelKey,
}

impl BoundSubstrate {
    /// Bind `shared` to `identity`: refuse a substrate another model wrote,
    /// record the key on one no model has, and embed the model's
    /// `tokenizer.json` so the log detokenizes offline.
    pub fn bind(shared: SharedSubstrate, identity: &ModelIdentity) -> Result<Self> {
        {
            let mut persistence = shared.persistence.lock().unwrap_or_else(|e| e.into_inner());
            let binding = persistence.bind_model(identity.key())?;
            let wrote_tokenizer = persistence.set_tokenizer(identity.tokenizer_json())?;
            if binding != ModelBinding::Same || wrote_tokenizer {
                persistence.commit()?;
            }
            tracing::info!(
                ?binding,
                model = %identity.key().describe(),
                dir = %persistence.dir().display(),
                "substrate bound to its model"
            );
        }
        Ok(Self {
            shared,
            key: identity.key().clone(),
        })
    }

    /// The bound substrate.
    pub fn shared(&self) -> &SharedSubstrate {
        &self.shared
    }

    /// The model the substrate is bound to.
    pub fn model_key(&self) -> &ModelKey {
        &self.key
    }

    /// The directory the substrate lives in.
    pub fn dir(&self) -> PathBuf {
        self.shared
            .persistence
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .dir()
            .to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::models::identity::LegacySpec;

    fn key() -> ModelKey {
        ModelKey {
            arch: "Qwen35Hybrid".into(),
            repo: "mradermacher/Qwen3.6-35B-A3B-AntiLoop-GGUF".into(),
            file: "Qwen3.6-35B-A3B-AntiLoop.gguf".into(),
            weights: [7; 32],
            tokenizer: [1; 32],
            rope: "lineage min=none".into(),
        }
    }

    fn legacy() -> LegacySpec {
        LegacySpec {
            arch: "Qwen35Hybrid".into(),
            model_repo: "mradermacher/Qwen3.6-35B-A3B-AntiLoop-GGUF".into(),
            model_filename: "Qwen3.6-35B-A3B-AntiLoop.gguf".into(),
        }
    }

    #[test]
    fn a_new_substrate_takes_the_key() {
        assert_eq!(
            decide(None, None, &key()),
            Decision::Write(ModelBinding::New)
        );
    }

    #[test]
    fn the_same_key_opens_and_writes_nothing() {
        let k = RecordedModel::Key(key());
        assert_eq!(decide(Some(&k), Some([1; 32]), &key()), Decision::Keep);
    }

    #[test]
    fn other_weights_are_refused() {
        let mut other = key();
        other.weights = [8; 32];
        let k = RecordedModel::Key(other);
        assert_eq!(decide(Some(&k), None, &key()), Decision::Refuse);
    }

    #[test]
    fn another_rope_schedule_is_refused() {
        let mut other = key();
        other.rope = "lineage min=40000000".into();
        let k = RecordedModel::Key(other);
        assert_eq!(decide(Some(&k), None, &key()), Decision::Refuse);
    }

    #[test]
    fn a_moved_repository_is_redescribed_not_refused() {
        let mut moved = key();
        moved.repo = "a-mirror/AntiLoop".into();
        let k = RecordedModel::Key(moved);
        assert_eq!(
            decide(Some(&k), None, &key()),
            Decision::Write(ModelBinding::Redescribed)
        );
    }

    #[test]
    fn a_matching_older_spec_is_adopted() {
        let l = RecordedModel::Legacy(legacy());
        assert_eq!(
            decide(Some(&l), Some([1; 32]), &key()),
            Decision::Write(ModelBinding::Adopted)
        );
        // With no tokenizer recorded, the spec alone decides.
        assert_eq!(
            decide(Some(&l), None, &key()),
            Decision::Write(ModelBinding::Adopted)
        );
    }

    #[test]
    fn an_older_spec_for_another_model_is_refused() {
        let mut flash = legacy();
        flash.arch = "Qwen4Exp".into();
        let l = RecordedModel::Legacy(flash);
        assert_eq!(decide(Some(&l), Some([1; 32]), &key()), Decision::Refuse);
    }

    #[test]
    fn an_older_spec_under_another_vocabulary_is_refused() {
        let l = RecordedModel::Legacy(legacy());
        assert_eq!(decide(Some(&l), Some([2; 32]), &key()), Decision::Refuse);
    }
}
