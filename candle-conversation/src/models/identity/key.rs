//! The model key and the record it is stored as.
//!
//! The record is text, one `name=value` per line under a versioned first line,
//! so `substrate_inspect` and an error message can show it as it is. Substrates
//! written before the key carry the older spec text in the same record; it is
//! decoded as a [`LegacySpec`] and checked once against the running model
//! (`docs/shared_model_host.md` §4.2).

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::models::ModelSpec;

/// First line of a key record.
const KEY_HEADER: &str = "candle-conversation model-key v1";

/// First line of the spec record substrates carried before the key.
const LEGACY_HEADER: &str = "candle-conversation model-spec v1";

/// What a substrate's model-derived state belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelKey {
    /// The architecture, as `ModelArch` names it: `Qwen35Hybrid`, `Qwen4Exp`, …
    pub arch: String,
    /// The repository the checkpoint was resolved from — a description, not
    /// identity: the same file from a mirror is the same model.
    pub repo: String,
    /// The checkpoint's name — its GGUF `general.name` with `.gguf`, as the
    /// builder names the spec once it has read the pack — a description, like
    /// `repo`.
    pub file: String,
    /// The weights: [`super::WeightFiles::digest`] over every file the model
    /// is loaded from.
    pub weights: [u8; 32],
    /// SHA-256 of `tokenizer.json`.
    pub tokenizer: [u8; 32],
    /// The RoPE schedule positions are rotated under — [`super::rope_key`].
    pub rope: String,
}

impl ModelKey {
    /// Whether `other` is the same model: the same architecture, weights,
    /// vocabulary and RoPE schedule. Where it was resolved from is not part of
    /// it.
    pub fn same_model(&self, other: &ModelKey) -> bool {
        self.arch == other.arch
            && self.weights == other.weights
            && self.tokenizer == other.tokenizer
            && self.rope == other.rope
    }

    /// The record payload.
    pub fn encode(&self) -> Vec<u8> {
        format!(
            "{KEY_HEADER}\n\
             arch={}\n\
             repo={}\n\
             file={}\n\
             weights={}\n\
             tokenizer={}\n\
             rope={}\n",
            self.arch,
            self.repo,
            self.file,
            hex(&self.weights),
            hex(&self.tokenizer),
            self.rope,
        )
        .into_bytes()
    }

    /// One line naming the model for a person: the file, the architecture, the
    /// first bytes of the weights digest and the RoPE schedule — everything two
    /// keys can differ in, so a refusal never names the same text twice.
    pub fn describe(&self) -> String {
        format!(
            "{} (arch {}, weights {}…, tokenizer {}…, rope {})",
            self.file,
            self.arch,
            &hex(&self.weights)[..8],
            &hex(&self.tokenizer)[..8],
            self.rope
        )
    }
}

/// A model as a substrate recorded it before the key: the spec text, which
/// names where the checkpoint came from but not what is in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacySpec {
    pub arch: String,
    pub model_repo: String,
    pub model_filename: String,
}

impl LegacySpec {
    /// The model `spec` names, before its pack is read: the architecture by the
    /// name a key records it under, and the spec's repository and file. The
    /// file is the spec's, not the checkpoint's `general.name` a recorded model
    /// holds — the builder takes that from the pack.
    pub fn of(spec: &ModelSpec) -> Self {
        Self {
            arch: format!("{:?}", spec.arch),
            model_repo: spec.model_repo.clone(),
            model_filename: spec.model_filename.clone(),
        }
    }

    /// Whether `key` is the model this spec describes, as far as the spec can
    /// say: the same architecture, repository and checkpoint name. Revision,
    /// tensor overrides, adapters and gate donor were never recorded, so a
    /// change confined to them is not visible here.
    pub fn matches(&self, key: &ModelKey) -> bool {
        self.arch == key.arch && self.model_repo == key.repo && self.model_filename == key.file
    }

    /// One line naming the model for a person.
    pub fn describe(&self) -> String {
        format!(
            "{} (arch {}, from {}; recorded before model keys)",
            self.model_filename, self.arch, self.model_repo
        )
    }
}

/// What a substrate's model record holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordedModel {
    Key(ModelKey),
    Legacy(LegacySpec),
}

impl RecordedModel {
    /// Decode a model record's payload — a key, or the spec text that came
    /// before it. Anything else is refused: a substrate whose model cannot be
    /// read cannot be checked.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| format!("the model record is not UTF-8: {e}"))?;
        let mut lines = text.lines();
        let header = lines.next().unwrap_or("");
        let mut fields: HashMap<&str, &str> = HashMap::new();
        for line in lines {
            let (name, value) = line
                .split_once('=')
                .ok_or_else(|| format!("the model record has a line with no `=`: {line:?}"))?;
            fields.insert(name, value);
        }
        let field = |name: &str| -> Result<String, String> {
            fields
                .get(name)
                .map(|v| v.to_string())
                .ok_or_else(|| format!("the model record has no `{name}`"))
        };
        match header {
            KEY_HEADER => Ok(RecordedModel::Key(ModelKey {
                arch: field("arch")?,
                repo: field("repo")?,
                file: field("file")?,
                weights: unhex(&field("weights")?)?,
                tokenizer: unhex(&field("tokenizer")?)?,
                rope: field("rope")?,
            })),
            LEGACY_HEADER => Ok(RecordedModel::Legacy(LegacySpec {
                arch: field("arch")?,
                model_repo: field("model_repo")?,
                model_filename: field("model_filename")?,
            })),
            other => Err(format!("the model record's first line is {other:?}")),
        }
    }

    /// One line naming the recorded model for a person.
    pub fn describe(&self) -> String {
        match self {
            RecordedModel::Key(k) => k.describe(),
            RecordedModel::Legacy(l) => l.describe(),
        }
    }
}

/// A model's key together with the vocabulary it hashes — the pair a
/// substrate is bound to, since the substrate also embeds `tokenizer.json` so
/// the log can be detokenized offline.
#[derive(Debug, Clone)]
pub struct ModelIdentity {
    key: ModelKey,
    tokenizer_json: Vec<u8>,
}

impl ModelIdentity {
    /// The identity of a model. Every text field is one line of the record, so
    /// a line break in any of them is refused.
    pub fn new(
        arch: String,
        repo: String,
        file: String,
        weights: [u8; 32],
        tokenizer_json: Vec<u8>,
        rope: String,
    ) -> Result<Self, String> {
        for (name, value) in [
            ("arch", &arch),
            ("repo", &repo),
            ("file", &file),
            ("rope", &rope),
        ] {
            if value.contains(['\n', '\r']) {
                return Err(format!("the model's {name} has a line break: {value:?}"));
            }
        }
        let tokenizer = Sha256::digest(&tokenizer_json).into();
        Ok(Self {
            key: ModelKey {
                arch,
                repo,
                file,
                weights,
                tokenizer,
                rope,
            },
            tokenizer_json,
        })
    }

    pub fn key(&self) -> &ModelKey {
        &self.key
    }

    /// This model run under `rope` — the schedule of the engine a substrate
    /// attaches to, which is what will rotate every key it writes. A shared
    /// engine's schedule is every tenant's merged
    /// (`crate::engine_slot::CoreRequests`), so a tenant's own builder does not
    /// know it.
    pub fn under_rope(mut self, rope: String) -> Self {
        self.key.rope = rope;
        self
    }

    pub fn tokenizer_json(&self) -> &[u8] {
        &self.tokenizer_json
    }
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Result<[u8; 32], String> {
    if s.len() != 64 {
        return Err(format!("a digest is 64 hex digits, not {}: {s:?}", s.len()));
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16)
            .map_err(|_| format!("a digest is hex: {s:?}"))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::models::Model;

    fn key() -> ModelKey {
        ModelKey {
            arch: "Qwen35Hybrid".into(),
            repo: "mradermacher/Qwen3.6-35B-A3B-AntiLoop-GGUF".into(),
            file: "Qwen3.6-35B-A3B-AntiLoop.gguf".into(),
            weights: [0xab; 32],
            tokenizer: [0x01; 32],
            rope: "lineage min=none".into(),
        }
    }

    #[test]
    fn a_key_encodes_to_these_bytes() {
        let expected = "candle-conversation model-key v1\n\
            arch=Qwen35Hybrid\n\
            repo=mradermacher/Qwen3.6-35B-A3B-AntiLoop-GGUF\n\
            file=Qwen3.6-35B-A3B-AntiLoop.gguf\n\
            weights=abababababababababababababababababababababababababababababababab\n\
            tokenizer=0101010101010101010101010101010101010101010101010101010101010101\n\
            rope=lineage min=none\n";
        assert_eq!(key().encode(), expected.as_bytes());
    }

    #[test]
    fn a_key_round_trips() {
        assert_eq!(
            RecordedModel::decode(&key().encode()).unwrap(),
            RecordedModel::Key(key())
        );
    }

    /// The exact text `ModelBuilder` wrote before the key.
    #[test]
    fn the_older_spec_record_decodes_as_legacy() {
        let old = b"candle-conversation model-spec v1\n\
            arch=Qwen4Exp\n\
            chat_format=ChatML\n\
            model_repo=unsloth/Qwen3.8-Flash-Next-GGUF\n\
            model_filename=Qwen3.8-Flash-Next.gguf\n\
            tokenizer_repo=Qwen/Qwen3.8-Flash-Next\n\
            max_seq_len=1048576\n";
        assert_eq!(
            RecordedModel::decode(old).unwrap(),
            RecordedModel::Legacy(LegacySpec {
                arch: "Qwen4Exp".into(),
                model_repo: "unsloth/Qwen3.8-Flash-Next-GGUF".into(),
                model_filename: "Qwen3.8-Flash-Next.gguf".into(),
            })
        );
    }

    #[test]
    fn a_legacy_spec_matches_on_arch_repo_and_file() {
        let legacy = LegacySpec {
            arch: "Qwen35Hybrid".into(),
            model_repo: "mradermacher/Qwen3.6-35B-A3B-AntiLoop-GGUF".into(),
            model_filename: "Qwen3.6-35B-A3B-AntiLoop.gguf".into(),
        };
        assert!(legacy.matches(&key()));
        let mut other = key();
        other.file = "Qwen3.6-35B-A3B.gguf".into();
        assert!(!legacy.matches(&other));
    }

    /// A spec names its model by the arch's variant name, and the repository
    /// and the file as the spec has them.
    #[test]
    fn a_spec_names_its_model_by_arch_repository_and_file() {
        let spec = Model::Qwen2_0_5B.preset_spec();
        assert_eq!(
            LegacySpec::of(&spec),
            LegacySpec {
                arch: "Qwen2".into(),
                model_repo: spec.model_repo.clone(),
                model_filename: spec.model_filename.clone(),
            }
        );
    }

    #[test]
    fn where_a_model_came_from_is_not_its_identity() {
        let mut moved = key();
        moved.repo = "a-mirror/Qwen3.6".into();
        assert!(key().same_model(&moved));
        let mut retrained = key();
        retrained.weights[0] = 0;
        assert!(!key().same_model(&retrained));
        let mut rerotated = key();
        rerotated.rope = "lineage min=40000000".into();
        assert!(!key().same_model(&rerotated));
    }

    #[test]
    fn an_unknown_record_is_refused() {
        assert!(RecordedModel::decode(b"qwen3-30b-a3b").is_err());
        let short = b"candle-conversation model-key v1\narch=Qwen3\n";
        assert!(RecordedModel::decode(short).is_err());
    }

    #[test]
    fn describe_names_everything_two_keys_can_differ_in() {
        assert_eq!(
            key().describe(),
            "Qwen3.6-35B-A3B-AntiLoop.gguf (arch Qwen35Hybrid, weights abababab…, \
             tokenizer 01010101…, rope lineage min=none)"
        );
    }

    /// An identity taken under another engine's schedule keeps everything but
    /// the schedule.
    #[test]
    fn an_identity_under_another_schedule_changes_only_its_rope() {
        let own = ModelIdentity::new(
            "Qwen35Hybrid".into(),
            "r".into(),
            "f".into(),
            [7; 32],
            b"tok".to_vec(),
            "lineage min=none".into(),
        )
        .unwrap();
        let under = own.clone().under_rope("lineage min=40000000".into());
        assert_eq!(under.key().rope, "lineage min=40000000");
        assert_eq!(under.key().weights, own.key().weights);
        assert_eq!(under.key().tokenizer, own.key().tokenizer);
        assert_eq!(under.tokenizer_json(), own.tokenizer_json());
    }

    #[test]
    fn the_identity_hashes_the_tokenizer_and_refuses_a_line_break() {
        let id = ModelIdentity::new(
            "Qwen3".into(),
            "r".into(),
            "f".into(),
            [0; 32],
            b"abc".to_vec(),
            "file-stated min=none".into(),
        )
        .unwrap();
        assert_eq!(
            hex(&id.key().tokenizer),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert!(ModelIdentity::new(
            "Qwen3".into(),
            "r\nx".into(),
            "f".into(),
            [0; 32],
            Vec::new(),
            "lineage min=none".into(),
        )
        .is_err());
    }
}
