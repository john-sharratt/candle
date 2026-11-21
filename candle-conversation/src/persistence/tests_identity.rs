//! Tests that a substrate is bound to the first model that writes it and refuses any other.
//!
//! A substrate written before keys adopts the key of the model its spec names, and the adopted key then decides, so other weights under the same name are another model. The tokenizer binding is one-way: a second tokenizer is refused on the handle that bound it and after a reopen, because the hash is recovered from disk.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::models::identity::RecordedModel;

    use super::super::tests_support::tests::{model, recorded, tmp_dir};

    /// A substrate takes the first model's key, opens again under it without
    /// writing, rewrites the record when the same model is described anew,
    /// and refuses another model — on the handle that bound it and on a reopen.
    #[test]
    fn a_substrate_is_bound_to_the_first_model_that_writes_it() {
        let dir = tmp_dir("modelkey");
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(
                sp.bind_model(&model("qwen3-30b-a3b", 1)).unwrap(),
                ModelBinding::New
            );
            let off1 = sp.manifest().model_key.unwrap().offset;
            assert_eq!(
                sp.bind_model(&model("qwen3-30b-a3b", 1)).unwrap(),
                ModelBinding::Same
            );
            assert_eq!(
                sp.manifest().model_key.unwrap().offset,
                off1,
                "the same key appends no record"
            );
            assert_eq!(
                sp.bind_model(&model("qwen3-30b-a3b-mirror", 1)).unwrap(),
                ModelBinding::Redescribed
            );
            assert!(sp.manifest().model_key.unwrap().offset > off1);
            let err = sp.bind_model(&model("qwen3-235b", 2)).unwrap_err();
            let msg = err.to_string();
            let recorded_model = model("qwen3-30b-a3b-mirror", 1).describe();
            let running_model = model("qwen3-235b", 2).describe();
            assert!(
                recorded_model.starts_with("qwen3-30b-a3b-mirror (arch Qwen3, weights 01010101…")
                    && running_model.starts_with("qwen3-235b (arch Qwen3, weights 02020202…"),
                "{recorded_model} / {running_model}"
            );
            assert!(
                msg.contains(&format!(
                    "belongs to {recorded_model}, not {running_model}."
                )),
                "the refusal names both models: {msg}"
            );
            sp.commit().unwrap();
        }
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(
                recorded(&sp),
                Some(RecordedModel::Key(model("qwen3-30b-a3b-mirror", 1)))
            );
            assert!(matches!(
                sp.bind_model(&model("qwen3-235b", 2)),
                Err(PersistenceError::WrongModel { .. })
            ));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A substrate written before keys records the spec text. A model that
    /// matches it is adopted — the key is written and decides from then on —
    /// and one that does not is refused.
    #[test]
    fn an_older_substrate_adopts_the_key_of_the_model_its_spec_names() {
        let spec = b"candle-conversation model-spec v1\n\
            arch=Qwen3\n\
            chat_format=ChatML\n\
            model_repo=test/repo\n\
            model_filename=qwen3-8b.gguf\n\
            tokenizer_repo=Qwen/Qwen3-8B\n\
            max_seq_len=40960\n";
        let dir = tmp_dir("modelkey_legacy");
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.append_record(RecordType::ModelKey, 0, 0, 0, 0, 0, spec)
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert!(matches!(
                sp.bind_model(&model("qwen3-14b.gguf", 1)),
                Err(PersistenceError::WrongModel { .. })
            ));
            assert_eq!(
                sp.bind_model(&model("qwen3-8b.gguf", 1)).unwrap(),
                ModelBinding::Adopted
            );
            sp.commit().unwrap();
        }
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(
                recorded(&sp),
                Some(RecordedModel::Key(model("qwen3-8b.gguf", 1)))
            );
            // Adopted, the key decides: other weights under the same name are
            // another model.
            assert!(matches!(
                sp.bind_model(&model("qwen3-8b.gguf", 2)),
                Err(PersistenceError::WrongModel { .. })
            ));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A substrate binds to one tokenizer, and a second one is refused.
    ///
    /// This used to assert that "changed bytes are re-embedded", which is the
    /// behaviour that let a model swap rewrite the vocabulary out from under
    /// every turn already sealed in the log — the ids stay, their meaning
    /// changes, and nothing reports it. The binding is the invariant now.
    #[test]
    fn tokenizer_binds_the_substrate_and_a_second_one_is_refused() {
        let dir = tmp_dir("tok_hash");
        let v1 = vec![1u8; 4096];
        let v2 = vec![2u8; 4096];
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert!(sp.set_tokenizer(&v1).unwrap(), "first tokenizer is written");
            assert!(
                !sp.set_tokenizer(&v1).unwrap(),
                "identical bytes are a hash-match no-op"
            );
            let err = sp
                .set_tokenizer(&v2)
                .expect_err("a different tokenizer must be refused, not appended");
            let msg = err.to_string();
            assert!(
                msg.contains("bound to a different tokenizer"),
                "the refusal must name the cause: {msg}"
            );
            assert_eq!(
                sp.tokenizer_sha256(),
                Some(sha256(&v1)),
                "a refused binding leaves the recorded tokenizer untouched"
            );
            sp.commit().unwrap();
        }
        {
            // The hash is recovered from disk, so the binding survives a
            // reopen — including the refusal.
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(sp.tokenizer_sha256(), Some(sha256(&v1)));
            assert!(
                !sp.set_tokenizer(&v1).unwrap(),
                "reopened log recognises the unchanged tokenizer by hash"
            );
            assert!(
                sp.set_tokenizer(&v2).is_err(),
                "the binding is still enforced after a reopen"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The tokenizer bytes are the `Tokenizer` record's payload — the
    /// log is a self-contained substrate image with no companion files.
    /// (Payload bit-rot coverage lives with `read_record_at`, which
    /// CRC-verifies every record it returns.)
    #[test]
    fn tokenizer_bytes_live_in_the_log() {
        let dir = tmp_dir("tok_embed");
        let bytes = (0..8192u32).map(|i| (i % 251) as u8).collect::<Vec<u8>>();
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert!(sp.set_tokenizer(&bytes).unwrap());
            sp.commit().unwrap();
            let tok_loc = sp.manifest().tokenizer.expect("manifest has tokenizer loc");
            assert_eq!(
                tok_loc.payload_len,
                bytes.len() as u64,
                "Tokenizer record payload must be the full tokenizer bytes"
            );
            assert_eq!(sp.read_tokenizer_bytes().unwrap(), Some(bytes.clone()));
        }
        // Bytes survive a reopen, recovered from the record on demand.
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(sp.tokenizer_sha256(), Some(sha256(&bytes)));
            assert_eq!(sp.read_tokenizer_bytes().unwrap(), Some(bytes.clone()));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn model_key_survives_reopen() {
        let dir = tmp_dir("ms_reopen");
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.bind_model(&model("the-model", 1)).unwrap();
            sp.set_template(b"the-template").unwrap();
            sp.commit().unwrap();
        }
        {
            let sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(
                recorded(&sp),
                Some(RecordedModel::Key(model("the-model", 1)))
            );
            assert_eq!(sp.template(), Some(b"the-template".as_slice()));
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
