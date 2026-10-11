//! What a model pack was built from, recorded in place of the files themselves.
//!
//! A pack's source checkpoints are deleted once it is built, so "which
//! checkpoint is this" is answered by a record written at build time: each
//! source's repo, revision, file name, length and a SHA-256 over the whole file.
//! The whole-file hash is affordable because it is taken once per build, beside
//! the build rather than ahead of it (`digest`).

use super::digest::PendingDigests;
use super::keys::{source_key, SOURCE_COUNT};
use candle::quantized::gguf_file::Value;
use candle::Result;
use std::collections::HashMap;

/// A build's sources as the pack records them: each source's record, and the
/// SHA-256s still being taken for the records that hold a placeholder.
pub struct Provenance {
    pub records: Vec<SourceRecord>,
    pub digests: PendingDigests,
}

impl Provenance {
    /// Records that already carry their digests.
    pub fn complete(records: Vec<SourceRecord>) -> Self {
        Self {
            records,
            digests: PendingDigests::none(),
        }
    }
}

/// One source file of a pack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRecord {
    /// What the file contributed: `checkpoint`, `mtp`, `gate-donor`,
    /// `override:<tensor>`, or a prepared artifact's `recipe`.
    pub role: String,
    pub repo: String,
    /// The pinned revision, or empty for a file resolved by name alone.
    pub rev: String,
    pub file: String,
    pub len: u64,
    /// Lowercase hex SHA-256 of the whole file.
    pub sha256: String,
}

/// The metadata entries recording `sources`, in order.
pub fn to_metadata(sources: &[SourceRecord]) -> Vec<(String, Value)> {
    let mut out = vec![(SOURCE_COUNT.to_string(), Value::U32(sources.len() as u32))];
    for (i, s) in sources.iter().enumerate() {
        out.push((source_key(i, "role"), Value::String(s.role.clone())));
        out.push((source_key(i, "repo"), Value::String(s.repo.clone())));
        out.push((source_key(i, "rev"), Value::String(s.rev.clone())));
        out.push((source_key(i, "file"), Value::String(s.file.clone())));
        out.push((source_key(i, "len"), Value::U64(s.len)));
        out.push((source_key(i, "sha256"), Value::String(s.sha256.clone())));
    }
    out
}

/// The sources a pack's metadata records.
pub fn from_metadata(metadata: &HashMap<String, Value>) -> Result<Vec<SourceRecord>> {
    let count = metadata
        .get(SOURCE_COUNT)
        .ok_or_else(|| candle::Error::Msg(format!("model pack: no {SOURCE_COUNT}")))?
        .to_u32()? as usize;
    let string = |i: usize, field: &str| -> Result<String> {
        let key = source_key(i, field);
        Ok(metadata
            .get(&key)
            .ok_or_else(|| candle::Error::Msg(format!("model pack: no {key}")))?
            .to_string()?
            .clone())
    };
    (0..count)
        .map(|i| {
            let len_key = source_key(i, "len");
            Ok(SourceRecord {
                role: string(i, "role")?,
                repo: string(i, "repo")?,
                rev: string(i, "rev")?,
                file: string(i, "file")?,
                len: metadata
                    .get(&len_key)
                    .ok_or_else(|| candle::Error::Msg(format!("model pack: no {len_key}")))?
                    .to_u64()?,
                sha256: string(i, "sha256")?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two() -> Vec<SourceRecord> {
        vec![
            SourceRecord {
                role: "checkpoint".into(),
                repo: "org/model-GGUF".into(),
                rev: "abc123".into(),
                file: "model-Q4_K_M.gguf".into(),
                len: 17_280_000_000,
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
            },
            SourceRecord {
                role: "mtp".into(),
                repo: "org/model-GGUF".into(),
                rev: String::new(),
                file: "mtp.gguf".into(),
                len: 4096,
                sha256: "00".repeat(32),
            },
        ]
    }

    /// The entries are these keys, in this order — the order the pack writes
    /// them, so the GGUF part of two builds of one model is byte-identical.
    #[test]
    fn the_entries_are_these_keys_in_order() {
        let keys: Vec<String> = to_metadata(&two()).into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys.len(), 13);
        assert_eq!(keys[0], "zen.source.count");
        assert_eq!(keys[1], "zen.source.0.role");
        assert_eq!(keys[6], "zen.source.0.sha256");
        assert_eq!(keys[12], "zen.source.1.sha256");
    }

    #[test]
    fn a_record_round_trips() {
        let map: HashMap<String, Value> = to_metadata(&two()).into_iter().collect();
        assert_eq!(from_metadata(&map).unwrap(), two());
    }

    /// A pack missing a field is refused by name, not read as an empty source.
    #[test]
    fn a_missing_field_is_named() {
        let mut map: HashMap<String, Value> = to_metadata(&two()).into_iter().collect();
        map.remove("zen.source.1.sha256");
        let e = from_metadata(&map).unwrap_err().to_string();
        assert!(e.contains("zen.source.1.sha256"), "{e}");
    }
}
