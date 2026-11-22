//! Tests for the model identity a substrate is bound to.
//!
//! A pack without a tokenizer refuses to start. A fine-tune swapped in under the
//! same name and vocabulary is refused, since the tokenizer check alone passed
//! exactly that case, and the refusal comes at bind, before an engine is built. Every
//! Flash-Next rung names the same release, so a substrate written under one rung
//! opens under another.

#[cfg(test)]
pub(super) mod identity_tests {
    use std::path::Path;

    use candle::quantized::gguf_file::Value;
    use candle_transformers::models::model_pack::keys::TOKENIZER_JSON;
    use candle_transformers::models::model_pack::SourceRecord;
    use candle_transformers::models::quantized_qwen38_moe::engine_model_digest;

    use crate::error::ConversationError;
    use crate::models::identity::{write_pack_header, WeightFiles};
    use crate::models::Model;

    const TOKENIZER: &str = r#"{"model":{"vocab":{"a":0}}}"#;

    /// A pack header built from one checkpoint whose whole-file SHA-256 is
    /// `sha` repeated, carrying `tokenizer` when given.
    fn pack(path: &Path, sha: char, tokenizer: Option<&str>) {
        let source = SourceRecord {
            role: "checkpoint".into(),
            repo: "org/m".into(),
            rev: "r1".into(),
            file: "m.gguf".into(),
            len: 4096,
            sha256: sha.to_string().repeat(64),
        };
        let extra: Vec<(&str, Value)> = tokenizer
            .map(|t| (TOKENIZER_JSON, Value::String(t.into())))
            .into_iter()
            .collect();
        write_pack_header(path, &[source], &extra);
    }

    /// **A pack without a tokenizer refuses the start.** The key a substrate is
    /// checked against hashes the vocabulary, so a model without one has
    /// nothing to check — opening the substrate anyway is how a different
    /// tokenizer's KV came to be served.
    #[test]
    fn a_pack_without_a_tokenizer_refuses_the_start() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("m.pack.gguf");
        pack(&path, 'a', None);
        let err = Model::Qwen35_0_8B_Q8
            .builder()
            .model_identity(&WeightFiles::single(&path))
            .expect_err("a model whose tokenizer cannot be read must not be identified");
        let ConversationError::Tokenizer(message) = err else {
            panic!("refused for the wrong reason: {err}");
        };
        let expected = format!(
            "read the tokenizer of {} to identify the model: ",
            path.display()
        );
        assert!(
            message.starts_with(&expected),
            "the refusal must name the pack it could not read: {message}"
        );
    }

    /// **A fine-tune swapped in under the same name and vocabulary is refused.**
    /// The tokenizer check alone passed exactly this case: same file name, same
    /// `tokenizer.json`, different weights — and every stored K/V was the other
    /// model's. The refusal comes at bind, before an engine is built, and
    /// leaves the substrate bound to the model that wrote it.
    #[test]
    fn a_checkpoint_with_the_same_tokenizer_cannot_open_anothers_substrate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().join("base.pack.gguf");
        let tuned = dir.path().join("tuned.pack.gguf");
        pack(&base, 'a', Some(TOKENIZER));
        pack(&tuned, 'b', Some(TOKENIZER));
        let workspace = dir.path().join("workspace");
        let builder = Model::Qwen35_0_8B_Q8.builder().workspace_path(&workspace);

        let first = builder.model_identity(&WeightFiles::single(&base)).unwrap();
        let bound = builder
            .bind_substrate(&first)
            .expect("a new substrate binds");
        drop(bound);

        let second = builder
            .model_identity(&WeightFiles::single(&tuned))
            .unwrap();
        assert_eq!(first.key().tokenizer, second.key().tokenizer);
        assert_ne!(first.key().weights, second.key().weights);
        let err = builder
            .bind_substrate(&second)
            .expect_err("another checkpoint's weights must not open this substrate");
        let message = err.to_string();
        assert!(
            message.contains(&format!(
                "belongs to {}, not {}",
                first.key().describe(),
                second.key().describe()
            )),
            "the refusal must name both models: {message}"
        );

        let reopened = builder
            .bind_substrate(&first)
            .expect("the model that wrote the substrate still opens it");
        assert_eq!(reopened.model_key(), first.key());
    }

    /// **Flash-Next is one model at every rung.** Two cards build packs from
    /// different artifacts — another expert width, another file — and each names
    /// the release, so a substrate written on one opens on the other.
    #[test]
    fn every_flash_next_rung_opens_the_same_substrate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let q3 = dir.path().join("q3.pack.gguf");
        let q4 = dir.path().join("q4.pack.gguf");
        pack(&q3, 'a', Some(TOKENIZER));
        pack(&q4, 'b', Some(TOKENIZER));
        let workspace = dir.path().join("workspace");
        let builder = Model::Qwen38_FlashNext.builder().workspace_path(&workspace);

        let small = builder.model_identity(&WeightFiles::single(&q3)).unwrap();
        let large = builder.model_identity(&WeightFiles::single(&q4)).unwrap();
        assert_eq!(small.key(), large.key());
        assert_eq!(small.key().weights, engine_model_digest());

        drop(
            builder
                .bind_substrate(&small)
                .expect("a new substrate binds"),
        );
        let reopened = builder
            .bind_substrate(&large)
            .expect("the other rung opens the substrate");
        assert_eq!(reopened.model_key(), small.key());
    }
}
