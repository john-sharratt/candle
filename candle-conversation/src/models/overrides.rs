//! Applying [`models.override.yaml`](candle_transformers::model_overrides) to a
//! [`ModelSpec`].
//!
//! The document itself — where it lives, how it reaches the binary, what it can
//! and cannot change — is described by
//! [`candle_transformers::model_overrides`], which owns the file because the
//! engine's own pinned checkpoints need it too and sit a crate below this one.
//!
//! What is here is the half that cannot live down there: mapping the parsed
//! override onto a [`ModelSpec`], a type this crate defines. One document, two
//! consumers, and the parse happens once in the crate both can see.

use candle_transformers::model_overrides;

use super::{LoraSpec, ModelSpec};

/// Apply the override for `key`, if the document names one.
pub(super) fn apply(key: &str, spec: ModelSpec) -> ModelSpec {
    match model_overrides::model(key) {
        Some(o) => merge(&o, spec),
        None => spec,
    }
}

fn merge(o: &model_overrides::ModelOverride, mut spec: ModelSpec) -> ModelSpec {
    if let Some(r) = &o.repo {
        // **What this override replaced, kept so a defect in the replacement can be repaired
        // from it.** A fine-tune whose conversion quantized the DeltaNet recurrent gates loads
        // cleanly and then generates incoherent text; the gates can be read from the base
        // checkpoint instead, and the base is definitionally the preset standing here.
        //
        // Recorded rather than configured, because there is exactly one right answer and
        // asking an operator for it would invite a wrong one — a donor from another model
        // would load and be subtly wrong. Nothing is fetched unless the replacement turns out
        // to need it, so this costs a preset that is never downloaded nothing at all.
        //
        // **A preset that already names its own base keeps it.** The displaced preset may itself
        // be a fine-tune whose gates are the defect being repaired — the hybrid's AntiLoop stores
        // them at Q4_K — and "repairing" from it would read the same quantized gates back. The
        // base it names is still the base of anything tuned from that model.
        let displaced = (
            spec.model_repo.clone(),
            spec.model_rev.clone(),
            spec.model_filename.clone(),
        );
        spec.gate_donor = spec.gate_donor.take().or(Some(displaced));
        // **Borrowed tensors do not follow a change of checkpoint.** They were chosen against the
        // preset's own file — a head that matches its shape, from a tune picked to go with it —
        // and grafting them onto whatever an operator substitutes would run a model nobody chose.
        // A filename-only override stays in the preset's repository and keeps them.
        spec.tensor_overrides.clear();
        spec.model_repo = r.clone();
        // **The preset's commit does not survive a change of repo.** A SHA identifies a commit
        // in the repository that produced it and means nothing in another one — carrying it
        // over would ask the hub for a revision that does not exist there, and the override
        // would fail to resolve for a reason naming neither the override nor the preset. An
        // override pins its own checkpoint with `revision:` or accepts `main`.
        spec.model_rev = o.revision.clone().unwrap_or_default();
    }
    if let Some(f) = &o.filename {
        spec.model_filename = f.clone();
    }
    // A revision named without a repo pins the preset's own checkpoint, which is the spelling
    // for "the preset's model, but hold it still".
    if o.repo.is_none() {
        if let Some(rev) = &o.revision {
            spec.model_rev = rev.clone();
        }
    }
    if let Some(b) = o.bytes {
        spec.model_bytes = b;
    }
    // Replacement, not merge — see `ModelOverride::loras` for why: the
    // repository's example adapter has no business riding somebody else's
    // checkpoint.
    if let Some(ls) = &o.loras {
        spec.loras = ls
            .iter()
            .map(|l| LoraSpec {
                name: l.name.clone(),
                repo: l.repo.clone(),
                revision: l.revision.clone(),
            })
            .collect();
    }
    spec
}

/// Which presets and checkpoints this machine overrides, for startup logging.
///
/// An operator reading a console that names a model they did not expect should
/// learn why from one line, rather than by going looking for a build script.
pub fn active_keys() -> Vec<String> {
    model_overrides::active_keys()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::model_overrides::{LoraOverride, ModelOverride};

    fn base() -> ModelSpec {
        super::super::Model::Qwen35_9B_Q6.preset_spec()
    }

    /// Merging an all-absent override is the identity.
    ///
    /// Deliberately *not* asserted through [`apply`], which reads the document
    /// embedded on whichever machine is running the test — and this machine may
    /// well have a real override. A test that asserted "nothing is overridden"
    /// would then fail because the feature works, which is the least useful
    /// failure a test can produce.
    #[test]
    fn an_absent_override_changes_nothing() {
        let before = base();
        let after = merge(&ModelOverride::default(), base());
        assert_eq!(after.model_repo, before.model_repo);
        assert_eq!(after.model_filename, before.model_filename);
        assert_eq!(after.model_bytes, before.model_bytes);
        assert_eq!(after.loras, before.loras);
    }

    /// The whole point: a preset's coordinates are replaced by a fine-tune's,
    /// and everything the override cannot reach survives.
    #[test]
    fn an_override_replaces_the_checkpoint_coordinates() {
        let o = ModelOverride {
            repo: Some("Someone/Some-Finetune-GGUF".into()),
            filename: Some("finetune-Q6_K.gguf".into()),
            bytes: Some(8_758_786_272),
            revision: None,
            loras: None,
        };
        let s = merge(&o, base());
        assert_eq!(s.model_repo, "Someone/Some-Finetune-GGUF");
        assert_eq!(s.model_filename, "finetune-Q6_K.gguf");
        assert_eq!(s.model_bytes, 8_758_786_272);
        // The preset's commit does not follow its repo out the door: a SHA names a commit in
        // the repository that produced it, so carrying it to another one would ask the hub
        // for a revision that does not exist there.
        assert_eq!(
            s.model_rev, "",
            "the preset's pin must not survive a repo change"
        );
        assert_eq!(
            s.gate_donor,
            Some((base().model_repo, base().model_rev, base().model_filename)),
            "the displaced checkpoint is recorded whole, commit included, so the gate donor \
             resolves the same bytes the preset would have"
        );

        // An override names a checkpoint, not a different architecture.
        assert_eq!(s.arch, base().arch);
        assert_eq!(s.tokenizer_repo, base().tokenizer_repo);
        assert_eq!(s.max_seq_len, base().max_seq_len);
        // `loras` absent entirely — whatever the preset carries survives, which is nothing.
        assert_eq!(s.loras, base().loras);
    }

    /// **A preset with its own base keeps it, and its borrowed tensors stay with its repo.**
    ///
    /// The hybrid names the stock checkpoint as its donor because its own trunk quantized the
    /// recurrent gates, so recording the displaced trunk as the donor would "repair" from the
    /// defect. And its head was chosen for its trunk: a substituted repo drops it, while a
    /// filename-only override — another quant of the same checkpoint — keeps it.
    #[test]
    fn an_override_keeps_a_presets_own_base_and_drops_its_borrowed_tensors() {
        let preset = super::super::Model::Qwen36_35B_A3B_AntiLoop_StyleTune.preset_spec();
        assert!(preset.gate_donor.is_some() && !preset.tensor_overrides.is_empty());

        let o = ModelOverride {
            repo: Some("Someone/Another-Finetune-GGUF".into()),
            ..Default::default()
        };
        let s = merge(&o, preset.clone());
        assert_eq!(
            s.gate_donor, preset.gate_donor,
            "the preset's own base survives"
        );
        assert!(
            s.tensor_overrides.is_empty(),
            "a new repo drops the borrowed head"
        );

        let o = ModelOverride {
            filename: Some("Qwen3.6-35B-A3B-AntiLoop.Q4_K_S.gguf".into()),
            ..Default::default()
        };
        let s = merge(&o, preset.clone());
        assert_eq!(s.tensor_overrides, preset.tensor_overrides);
        assert_eq!(s.gate_donor, preset.gate_donor);
    }

    /// A partial override leaves what it does not name.
    #[test]
    fn omitted_fields_keep_the_presets_own() {
        let o = ModelOverride {
            repo: Some("Someone/Only-The-Repo".into()),
            ..Default::default()
        };
        let s = merge(&o, base());
        assert_eq!(s.model_repo, "Someone/Only-The-Repo");
        assert_eq!(s.model_filename, base().model_filename);
        assert_eq!(s.model_bytes, base().model_bytes);
    }

    /// An override may pin its own checkpoint, and may pin the preset's without moving it.
    #[test]
    fn an_override_can_pin_a_revision() {
        // With a repo: the override's commit, in the override's repository.
        let o = ModelOverride {
            repo: Some("Someone/Some-Finetune-GGUF".into()),
            revision: Some("f".repeat(40)),
            ..Default::default()
        };
        assert_eq!(merge(&o, base()).model_rev, "f".repeat(40));

        // Without one: the preset's own checkpoint, held at a named commit. The repo is
        // untouched, which is what makes this "the preset's model, but hold it still".
        let o = ModelOverride {
            revision: Some("e".repeat(40)),
            ..Default::default()
        };
        let s = merge(&o, base());
        assert_eq!(s.model_rev, "e".repeat(40));
        assert_eq!(s.model_repo, base().model_repo);
        assert_eq!(
            s.gate_donor, None,
            "nothing was displaced, so there is no donor to record"
        );
    }

    /// **Adapters are replaced, not merged.** The repository's example adapter
    /// must not stay attached to somebody else's checkpoint: it would load
    /// against weights it was never trained on and degrade whatever asked for
    /// it, with nothing to indicate why.
    #[test]
    fn overriding_the_adapters_replaces_them_wholesale() {
        // The preset carries none of its own, so the adapter to displace is put there here.
        // Stating it locally is also what keeps this test about `merge` rather than about
        // whichever adapters a preset happens to ship — the coupling that made an earlier
        // version of it fail the day the preset stopped shipping one.
        let mut preset = base();
        preset.loras = vec![LoraSpec {
            name: "carried".into(),
            repo: "Someone/Carried-LoRA".into(),
            revision: "0".repeat(40),
        }];

        let o = ModelOverride {
            loras: Some(vec![LoraOverride {
                name: "rp".into(),
                repo: "Someone/Some-LoRA".into(),
                revision: "abc123".into(),
            }]),
            ..Default::default()
        };
        let s = merge(&o, preset);
        assert_eq!(s.loras.len(), 1);
        assert_eq!(s.loras[0].name, "rp");
        assert_eq!(s.loras[0].revision, "abc123");
        assert!(
            !s.loras.iter().any(|l| l.name == "carried"),
            "the preset's adapter survived an override that replaced the adapters"
        );
    }

    /// An empty list removes the preset's adapters — distinct from omitting the
    /// key, which keeps them. Both spellings are legitimate and they must not
    /// mean the same thing.
    #[test]
    fn an_empty_adapter_list_removes_them() {
        let o = ModelOverride {
            loras: Some(Vec::new()),
            ..Default::default()
        };
        assert!(merge(&o, base()).loras.is_empty());
    }

    /// **The override reaches `Model::spec`, and `preset_spec` escapes it.**
    ///
    /// The pair is the contract every other consumer depends on: one call site
    /// resolves overrides so nothing else has to, and one call site does not so
    /// the override machinery can still see what it is overriding.
    ///
    /// Holds on a machine with an override and on one without. Either way
    /// `spec()` equals `apply(key, preset_spec())`, which is the property that
    /// would break if some caller started reading the preset directly.
    #[test]
    fn spec_is_the_preset_with_the_document_applied() {
        use super::super::Model;
        let key = Model::Qwen35_9B_Q6
            .override_key()
            .expect("a preset is overridable");
        assert_eq!(key, "Qwen35_9B_Q6", "the key is the variant's identifier");

        let resolved = Model::Qwen35_9B_Q6.spec();
        let preset = Model::Qwen35_9B_Q6.preset_spec();
        let expected = apply(&key, Model::Qwen35_9B_Q6.preset_spec());
        assert_eq!(resolved.model_repo, expected.model_repo);
        assert_eq!(resolved.model_filename, expected.model_filename);
        assert_eq!(resolved.model_bytes, expected.model_bytes);
        assert_eq!(resolved.loras, expected.loras);

        // Whatever the document says, these are the preset's to decide.
        assert_eq!(resolved.arch, preset.arch);
        assert_eq!(resolved.tokenizer_repo, preset.tokenizer_repo);
        assert_eq!(resolved.max_seq_len, preset.max_seq_len);
    }

    /// `Model::Custom` is not overridable: it is already whatever the caller
    /// constructed, and a file on disk rewriting it would be the opposite of
    /// what `Custom` is for.
    #[test]
    fn a_custom_model_has_no_override_key() {
        use super::super::Model;
        assert!(Model::Custom(base()).override_key().is_none());
    }

    /// Every preset is reachable by its own name — the name a `--model` flag
    /// or `models.override.yaml` gives resolves to exactly that variant — and
    /// a name that is no preset resolves to nothing rather than to a default.
    #[test]
    fn every_preset_resolves_from_its_own_key() {
        use super::super::Model;
        for m in Model::PRESETS {
            let key = m.override_key().expect("a preset is overridable");
            let back = Model::from_override_key(&key).expect("its own key resolves");
            assert_eq!(back.override_key().as_deref(), Some(key.as_str()));
        }
        assert_eq!(
            Model::from_override_key("Qwen35_0_8B_Q8")
                .and_then(|m| m.override_key())
                .as_deref(),
            Some("Qwen35_0_8B_Q8")
        );
        assert!(Model::from_override_key("NoSuchModel").is_none());
    }
}
