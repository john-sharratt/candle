//! The records a repository replicates as (`docs/substrate_git.md` §3).
//!
//! | Key | Value |
//! |---|---|
//! | `git/pack/<repo>/<sha256>` | [`PackEntry`]: the `Blob` of that SHA-256 is a pack of `<repo>`, its index another `Blob` |
//! | `git/ref/<repo>/<ref>` | [`RefEntry`]: where `<ref>` points, and the packs it needs |
//! | `git/slice/<repo>/<oid>` | A commit an unfinished sliced fetch has taken |
//!
//! Both are keyed objects, last writer wins by version stamp. A ref is only
//! ever written by a member that has checked it descends from the one it
//! replaces, unless it says `forced` (`docs/substrate_git.md` §5.3).

use std::collections::BTreeMap;

use anyhow::{bail, Context};
use serde::{Deserialize, Serialize};
use zend_vfs::RefName;

/// What a member's log holds and writes: blobs by SHA-256, and keyed JSON
/// objects. `SharedSubstrate` is one (feature `substrate`); tests use a map.
pub trait Records: Send + Sync {
    /// Keep `bytes` as a blob; their SHA-256.
    fn put_blob(&self, bytes: Vec<u8>) -> anyhow::Result<[u8; 32]>;
    /// The blob named `sha256`, or `None` while it has not arrived.
    fn blob(&self, sha256: &[u8; 32]) -> anyhow::Result<Option<Vec<u8>>>;
    /// Whether the blob named `sha256` has arrived, without reading it.
    fn has_blob(&self, sha256: &[u8; 32]) -> bool;
    /// Write the keyed object `key`.
    fn put_object(&self, key: &str, value: serde_json::Value) -> anyhow::Result<()>;
    /// Retire the keyed object `key`.
    fn delete_object(&self, key: &str) -> anyhow::Result<()>;
    /// Every live keyed object whose key starts with `prefix`.
    fn objects_under(&self, prefix: &str) -> Vec<(String, serde_json::Value)>;
}

/// Who may import a repository from `origin`: one member of the group at a
/// time (`docs/substrate_git.md` §5.1).
pub trait Lease: Send + Sync {
    /// Take or renew `key`; whether this member holds it now.
    fn hold(&self, key: &str) -> bool;
}

/// A member alone: it always holds every lease. A daemon with no cluster.
pub struct Alone;

impl Lease for Alone {
    fn hold(&self, _key: &str) -> bool {
        true
    }
}

/// `git/pack/<repo>/<sha256>`: the blob of that SHA-256 is a pack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackEntry {
    /// The pack's trailing checksum, in hex: the name git gives its file.
    pub checksum: String,
    /// How many objects it holds.
    pub objects: u32,
    /// The SHA-256 of the blob holding its `.idx` (`docs/substrate_git.md`
    /// §3).
    pub index: String,
}

/// `git/ref/<repo>/<ref>`: where a ref points.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefEntry {
    /// The object it points at, or `None` for a deleted ref.
    pub oid: Option<String>,
    /// The SHA-256s of the packs written with it: what `oid` reaches beyond
    /// what the log already held. The ref waits for them.
    #[serde(default)]
    pub needs: Vec<String>,
    /// Allowed to replace a ref it does not descend from.
    #[serde(default)]
    pub forced: bool,
}

/// One repository's records in a member's log.
pub struct Log<'a> {
    records: &'a dyn Records,
    repo: &'a str,
}

impl<'a> Log<'a> {
    /// `repo`'s records in `records`. Refuses a name that is not one folder.
    pub fn new(records: &'a dyn Records, repo: &'a str) -> anyhow::Result<Self> {
        check_name(repo)?;
        Ok(Self { records, repo })
    }

    /// The repository's name.
    pub fn repo(&self) -> &str {
        self.repo
    }

    /// Every pack, by the SHA-256 of its blob.
    pub fn packs(&self) -> anyhow::Result<BTreeMap<String, PackEntry>> {
        let prefix = packs_prefix(self.repo);
        self.records
            .objects_under(&prefix)
            .into_iter()
            .map(|(key, value)| {
                let sha = key[prefix.len()..].to_string();
                let entry = serde_json::from_value(value).with_context(|| key.clone())?;
                Ok((sha, entry))
            })
            .collect()
    }

    /// Every ref, by its full name.
    pub fn refs(&self) -> anyhow::Result<BTreeMap<String, RefEntry>> {
        let prefix = refs_prefix(self.repo);
        self.records
            .objects_under(&prefix)
            .into_iter()
            .map(|(key, value)| {
                let name = key[prefix.len()..].to_string();
                let entry = serde_json::from_value(value).with_context(|| key.clone())?;
                Ok((name, entry))
            })
            .collect()
    }

    /// Write a pack: its blob and its index's, then the entry naming both. Its
    /// SHA-256 in hex.
    pub fn put_pack(
        &self,
        pack: Vec<u8>,
        index: Vec<u8>,
        checksum: &str,
        objects: u32,
    ) -> anyhow::Result<String> {
        let sha = hex(&self.records.put_blob(pack)?);
        let index = hex(&self.records.put_blob(index)?);
        let entry = PackEntry {
            checksum: checksum.to_string(),
            objects,
            index,
        };
        self.records
            .put_object(&pack_key(self.repo, &sha), serde_json::to_value(&entry)?)?;
        Ok(sha)
    }

    /// Whether the pack `sha` and its index have both arrived.
    pub fn arrived(&self, sha: &str, entry: &PackEntry) -> anyhow::Result<bool> {
        Ok(self.records.has_blob(&unhex(sha)?) && self.records.has_blob(&unhex(&entry.index)?))
    }

    /// The blob of the index `entry` names, or `None` while it has not
    /// arrived.
    pub fn index_bytes(&self, entry: &PackEntry) -> anyhow::Result<Option<Vec<u8>>> {
        self.records.blob(&unhex(&entry.index)?)
    }

    /// The commits an unfinished sliced fetch has taken.
    pub fn slices(&self) -> Vec<String> {
        let prefix = slices_prefix(self.repo);
        self.records
            .objects_under(&prefix)
            .into_iter()
            .map(|(key, _)| key[prefix.len()..].to_string())
            .collect()
    }

    /// Keep commit `oid` as taken by a sliced fetch, or forget it.
    pub fn set_slice(&self, oid: &str, kept: bool) -> anyhow::Result<()> {
        let key = format!("{}{oid}", slices_prefix(self.repo));
        match kept {
            true => self.records.put_object(&key, serde_json::Value::Bool(true)),
            false => self.records.delete_object(&key),
        }
    }

    /// Write where `name` points.
    pub fn put_ref(&self, name: &RefName, entry: &RefEntry) -> anyhow::Result<()> {
        self.records
            .put_object(&ref_key(self.repo, name), serde_json::to_value(entry)?)
    }

    /// The blob of the pack named `sha` (hex), or `None` while it has not
    /// arrived.
    pub fn pack_bytes(&self, sha: &str) -> anyhow::Result<Option<Vec<u8>>> {
        self.records.blob(&unhex(sha)?)
    }
}

/// The lease the importer of `repo` holds.
pub fn import_lease(repo: &str) -> String {
    format!("git/{repo}")
}

fn packs_prefix(repo: &str) -> String {
    format!("git/pack/{repo}/")
}

fn refs_prefix(repo: &str) -> String {
    format!("git/ref/{repo}/")
}

fn slices_prefix(repo: &str) -> String {
    format!("git/slice/{repo}/")
}

fn pack_key(repo: &str, sha: &str) -> String {
    format!("{}{sha}", packs_prefix(repo))
}

fn ref_key(repo: &str, name: &RefName) -> String {
    format!("{}{}", refs_prefix(repo), name.as_str())
}

/// A repository's name is one plain folder name, as `workspace.yaml` names
/// one (`docs/zend_workspace_execution.md` §3.1).
fn check_name(repo: &str) -> anyhow::Result<()> {
    let bad = repo.is_empty()
        || repo == "."
        || repo == ".."
        || repo.contains(['/', '\\', ':'])
        || repo.starts_with('-');
    if bad {
        bail!("`{repo}` is not a repository's name");
    }
    Ok(())
}

/// `bytes` in lowercase hex.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn unhex(text: &str) -> anyhow::Result<[u8; 32]> {
    let mut out = [0u8; 32];
    if text.len() != 64 {
        bail!("`{text}` is not a SHA-256");
    }
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16)
            .with_context(|| format!("`{text}` is not a SHA-256"))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repository_name_is_one_folder() {
        for bad in ["", ".", "..", "a/b", "a\\b", "c:", "-x"] {
            assert!(check_name(bad).is_err(), "{bad}");
        }
        assert!(check_name("battle-mind").is_ok());
    }

    #[test]
    fn hex_round_trips_a_sha256() {
        let sha = [7u8; 32];
        assert_eq!(unhex(&hex(&sha)).unwrap(), sha);
        assert!(unhex("zz").is_err());
    }
}
