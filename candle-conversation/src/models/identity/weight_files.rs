//! Every file a model's weights are loaded from, and their joint digest.
//!
//! A model loads from its model pack (`docs/self_contained_model_packs.md`),
//! which already holds every checkpoint the model is made of — the trunk, the
//! tensors an override takes from another checkpoint, a gate donor — and
//! records each one's whole-file SHA-256. Adapters stay outside it. The
//! builder resolves exactly what its loader will read into a [`WeightFiles`],
//! and the same value feeds both the load and the key, so the key names what
//! was loaded.

use std::io;
#[cfg(test)]
use std::io::Write;
use std::path::{Path, PathBuf};

#[cfg(test)]
use candle::quantized::gguf_file::{self, Value};
#[cfg(test)]
use candle_transformers::models::model_pack::keys::{VERSION, VERSION_KEY};
use candle_transformers::models::model_pack::provenance::from_metadata;
#[cfg(test)]
use candle_transformers::models::model_pack::provenance::to_metadata;
use candle_transformers::models::model_pack::ModelPack;
#[cfg(test)]
use candle_transformers::models::model_pack::SourceRecord;
use sha2::{Digest, Sha256};

use super::fingerprint::file_digest;

/// The files a model is loaded from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightFiles {
    /// The model pack.
    pub pack: PathBuf,
    /// Adapters, as `(name, directory)`.
    pub adapters: Vec<(String, PathBuf)>,
}

impl WeightFiles {
    /// A model loaded from its pack and nothing else.
    pub fn single(pack: impl Into<PathBuf>) -> Self {
        Self {
            pack: pack.into(),
            adapters: Vec::new(),
        }
    }

    /// The digest of the weights: SHA-256 over the tag `weights v2\0`, then per
    /// source the pack records, in its order, `source\0<role>\0<sha256>\0`; then
    /// per adapter `adapter\0<name>\0`, then each regular file in its directory,
    /// by name, as `<file name>\0` and its [`file_digest`].
    ///
    /// **The sources, not the pack's bytes.** A pack is built per card — its
    /// int8 mode and its sections' layout differ between machines — while the
    /// model it holds does not, and a substrate written on one machine must open
    /// on the others. Only the header is read, so this costs milliseconds.
    pub fn digest(&self) -> io::Result<[u8; 32]> {
        let invalid = |e: candle::Error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {e}", self.pack.display()),
            )
        };
        let header = ModelPack::read_header(&self.pack).map_err(invalid)?;
        let sources = from_metadata(&header.metadata).map_err(invalid)?;
        let mut hasher = Sha256::new();
        hasher.update(b"weights v2\0");
        for s in &sources {
            hasher.update(b"source\0");
            hasher.update(s.role.as_bytes());
            hasher.update(b"\0");
            hasher.update(s.sha256.as_bytes());
            hasher.update(b"\0");
        }
        for (name, dir) in &self.adapters {
            hasher.update(b"adapter\0");
            hasher.update(name.as_bytes());
            hasher.update(b"\0");
            for file in adapter_files(dir)? {
                let file_name = file.file_name().map(|n| n.to_string_lossy().into_owned());
                hasher.update(file_name.unwrap_or_default().as_bytes());
                hasher.update(b"\0");
                hasher.update(file_digest(&file)?);
            }
        }
        Ok(hasher.finalize().into())
    }
}

/// The regular files directly in `dir`, sorted by name.
fn adapter_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            files.push(entry.path());
        }
    }
    files.sort();
    Ok(files)
}

/// A pack header holding `sources`, and `extra` metadata beside them — enough
/// for [`WeightFiles::digest`], which reads nothing else.
#[cfg(test)]
pub(crate) fn write_pack_header(path: &Path, sources: &[SourceRecord], extra: &[(&str, Value)]) {
    let mut entries: Vec<(String, Value)> = vec![(VERSION_KEY.to_string(), Value::U32(VERSION))];
    entries.extend(to_metadata(sources));
    entries.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
    let refs: Vec<(&str, &Value)> = entries.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mut f = std::fs::File::create(path).unwrap();
    gguf_file::write(&mut f, &refs, &[]).unwrap();
    f.flush().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_transformers::models::model_pack::keys::INT8_MODE;

    fn source(role: &str, sha: char) -> SourceRecord {
        SourceRecord {
            role: role.into(),
            repo: "org/m".into(),
            rev: "r1".into(),
            file: format!("{role}.gguf"),
            len: 4096,
            sha256: sha.to_string().repeat(64),
        }
    }

    fn hex(d: &[u8; 32]) -> String {
        d.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The digest of a fixed pack is a fixed value: the definition is pinned, so
    /// a change to it shows here before it makes every substrate refuse its own
    /// model.
    #[test]
    fn a_fixed_packs_digest_is_a_fixed_value() {
        let dir = tempfile::tempdir().unwrap();
        let pack = dir.path().join("m.pack.gguf");
        write_pack_header(&pack, &[source("checkpoint", 'a')], &[]);
        assert_eq!(
            hex(&WeightFiles::single(&pack).digest().unwrap()),
            "173f3104b7bc4a947a9f8a3e6e838fc95cd7f8cda280677e13e861abf2035791",
        );
    }

    /// Every source the model is made of counts — a different file under a role,
    /// and the role a file plays — and so does an adapter's every byte.
    #[test]
    fn every_part_of_a_model_moves_the_digest() {
        let dir = tempfile::tempdir().unwrap();
        let at = |name: &str, sources: &[SourceRecord]| {
            let p = dir.path().join(name);
            write_pack_header(&p, sources, &[]);
            WeightFiles::single(p).digest().unwrap()
        };
        let base = at("base", &[source("checkpoint", 'a')]);
        let other_trunk = at("trunk", &[source("checkpoint", 'b')]);
        let with_head = at(
            "head",
            &[
                source("checkpoint", 'a'),
                source("override:output.weight", 'c'),
            ],
        );
        let as_donor = at(
            "donor",
            &[source("checkpoint", 'a'), source("gate-donor", 'c')],
        );
        assert_ne!(other_trunk, base);
        assert_ne!(with_head, base);
        assert_ne!(as_donor, with_head, "the role a file plays is part of it");

        let pack = dir.path().join("base");
        let adapter = dir.path().join("lora");
        std::fs::create_dir(&adapter).unwrap();
        std::fs::write(adapter.join("adapter_config.json"), b"{}").unwrap();
        std::fs::write(adapter.join("adapter_model.safetensors"), b"v1").unwrap();
        let with_adapter = WeightFiles {
            pack: pack.clone(),
            adapters: vec![("style".into(), adapter.clone())],
        };
        let d = with_adapter.digest().unwrap();
        assert_ne!(d, base);
        std::fs::write(adapter.join("adapter_model.safetensors"), b"v2").unwrap();
        assert_ne!(with_adapter.digest().unwrap(), d);

        // And a function of the files: computed again, the same.
        assert_eq!(WeightFiles::single(&pack).digest().unwrap(), base);
    }

    /// **A pack built for another card is the same model.** Its int8 mode and
    /// layout are the card's, not the model's, so they leave the digest alone —
    /// which is what lets one machine's substrate open on another's.
    #[test]
    fn what_a_card_decides_leaves_the_digest_alone() {
        let dir = tempfile::tempdir().unwrap();
        let sources = [source("checkpoint", 'a')];
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        write_pack_header(&a, &sources, &[(INT8_MODE, Value::U64(1))]);
        write_pack_header(&b, &sources, &[(INT8_MODE, Value::U64(2))]);
        assert_eq!(
            WeightFiles::single(&a).digest().unwrap(),
            WeightFiles::single(&b).digest().unwrap()
        );
    }

    #[test]
    fn a_file_that_is_not_a_pack_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plain.gguf");
        std::fs::write(&path, b"plain text").unwrap();
        let err = WeightFiles::single(&path).digest().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}
