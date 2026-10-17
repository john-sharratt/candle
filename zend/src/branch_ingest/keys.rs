//! Content keys — what an ingested unit is found by, whatever commit it was
//! found on (`docs/zend_branch_ingest.md` §6).

use std::fmt::Write;

use sha2::{Digest, Sha256};
use zend_vfs::Oid;

use crate::repo_scan::types::ModuleHint;

/// Conversation metadata: the unit's content key. Written only once its
/// ingest succeeds, so a conversation carrying it is committed.
pub const CONTENT_KEY: &str = "content_key";

/// Conversation metadata: the bits per weight of the model that wrote the unit
/// — `3` for a Q3 pack, `4` for a Q4 — as `candle_conversation::guest::
/// checkpoint::weight_bits` reads it. Optional: a unit written before it has
/// none, and a key held by one such is judged by stamp alone
/// (`super::plan::plan`).
pub const QUANT_KEY: &str = "quant";

/// The bits per weight of the model this process ingests with, set once the
/// model pack is known; `None` before, or when the pack names no file type.
static MODEL_BITS: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();

/// Record the model pack's bits per weight, once per process.
pub fn set_model_bits(bits: Option<u32>) {
    let _ = MODEL_BITS.set(bits);
}

/// What a unit ingested now is tagged with under [`QUANT_KEY`].
pub fn model_bits() -> Option<u32> {
    MODEL_BITS.get().copied().flatten()
}

/// Every conversation's [`QUANT_KEY`], read as bits per weight; one that has
/// none, or one that does not parse, is absent. Takes the substrate lock
/// itself, so it is called before a read guard is held.
pub fn quant_tags(
    engine: &candle_conversation::ConversationEngine,
) -> std::collections::HashMap<candle_conversation::projection::TimelineId, u32> {
    engine
        .conversations_with_metadata_key(QUANT_KEY)
        .into_iter()
        .filter_map(|(timeline, bits)| Some((timeline, bits.parse().ok()?)))
        .collect()
}

/// Conversation metadata: a file unit's blob id.
pub const BLOB_KEY: &str = "blob";

/// Conversation metadata: how many lines a file unit's file holds — what a
/// fast-path answer tells the model it already has.
pub const LINES_KEY: &str = "lines";

/// Conversation metadata: every branch whose tip holds the unit, comma
/// separated in walk order — the repository's default branch first
/// ([`branches_value`]). Not part of the key: one unit on many branches is one
/// conversation, and which branches hold it moves as they do, so each pass
/// rewrites it when it changed.
pub const BRANCHES_KEY: &str = "branches";

/// Conversation metadata: the commit a file unit's conversation read the file
/// at. Written once and never moved — it names the version the conversation
/// holds, which every branch sharing that blob shares, so it is what tells two
/// readings of one path apart where a branch list would not.
pub const COMMIT_KEY: &str = "commit";

/// The version of what an ingested unit's turns say beyond its content — the
/// calls and responses rendered around it. Part of every key, so a change to
/// that rendering re-reads every unit: each one's new key is queued, its old
/// conversation stays readable until the replacement commits, and the next pass
/// tombstones it (`plan`).
///
/// 2: pages count from 1 — `file_read`'s and `file_list`'s `page`, and the
/// `page=1 pages=N` fence. A unit read under 1 holds zero-based calls and
/// fences, and nothing the conversation is shown should teach those.
///
/// Every key states its version as a `v<N>:` prefix ([`key_version`]), so a
/// member can tell a unit read under a NEWER version — a peer's, upgraded before
/// it — from one it should replace, and leaves it alone.
pub const KEY_VERSION: u32 = 2;

/// The version a key was written under: its `v<N>:` prefix, and 1 for a key
/// with none — the form every key had before versions.
pub fn key_version(key: &str) -> u32 {
    key.strip_prefix('v')
        .and_then(|rest| rest.split_once(':'))
        .and_then(|(n, _)| n.parse().ok())
        .unwrap_or(1)
}

/// [`BRANCHES_KEY`]'s value for `names`.
pub fn branches_value(names: &[String]) -> String {
    names.join(",")
}

/// A file's key: the [`KEY_VERSION`], its workspace-relative path and its blob
/// id — `v2:candle/zend/src/main.rs@3f73f872…`. The path is in it because the
/// ingested conversation names the path; the blob id is git's hash of the
/// bytes `file_read` returns.
pub fn file_key(path: &str, blob: &Oid) -> String {
    format!("v{KEY_VERSION}:{path}@{blob}")
}

/// What a folder's listing turn shows: how many entries the folder holds,
/// and the entries on the listing's first page, workspace-relative, a folder
/// ending in `/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listing<'a> {
    pub total: usize,
    pub page: &'a [String],
}

/// A folder's key: `v<N>:` and then SHA-256, in hex, over the [`KEY_VERSION`],
/// the folder (workspace-relative, `candle/zend/src/`, or `.` for the
/// workspace), its [`Listing`], and the
/// manifest hint its request shows — everything its turns show but the sizes
/// the listing prints beside each file, which move with every edit of a file
/// the folder only names.
///
/// **The hint, not the manifest's bytes.** The request shows the hint the
/// manifest gives (`(crate: candle-nn)`, `(Cargo workspace root)`), never the
/// manifest itself, so two versions of a `Cargo.toml` that give the same hint
/// — a dependency bumped, a version raised — make the same turns. Keyed on the
/// manifest's blob, each such edit split the folder into another conversation
/// saying the same thing: `candle/` stood ten times over seven distinct
/// listings across its branches.
pub fn dir_key(dir: &str, listing: &Listing<'_>, hint: Option<&ModuleHint>) -> String {
    let mut h = Sha256::new();
    h.update(format!("v{KEY_VERSION}\n").as_bytes());
    h.update(dir.as_bytes());
    h.update(b"\n");
    h.update(listing.total.to_string().as_bytes());
    h.update(b"\n");
    for entry in listing.page {
        h.update(entry.as_bytes());
        h.update(b"\n");
    }
    if let Some(hint) = hint {
        h.update(b"\0hint\0");
        h.update(hint.render().as_bytes());
    }
    let mut out = format!("v{KEY_VERSION}:");
    for b in h.finalize() {
        let _ = write!(out, "{b:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "ce013625030ba8dba906f756967f9e9ca394464a";

    fn oid(hex: &str) -> Oid {
        Oid::parse(hex).unwrap()
    }

    fn page(entries: &[&str]) -> Vec<String> {
        entries.iter().map(|e| e.to_string()).collect()
    }

    #[test]
    fn the_branches_value_keeps_walk_order() {
        let names = ["main".to_string(), "qwen38-moe".to_string()];
        assert_eq!(branches_value(&names), "main,qwen38-moe");
        assert_eq!(branches_value(&[]), "");
    }

    #[test]
    fn a_file_key_is_its_path_and_blob() {
        assert_eq!(
            file_key("candle/zend/src/main.rs", &oid(A)),
            "v2:candle/zend/src/main.rs@ce013625030ba8dba906f756967f9e9ca394464a"
        );
    }

    /// The exact digests — computed independently of this code over the
    /// layout's bytes — so the layout the substrate's keys were written under
    /// cannot drift silently.
    #[test]
    fn a_dir_key_is_the_sha256_of_its_evidence() {
        let entries = page(&["r/a/sub/", "r/a/x.rs", "r/a/y.rs"]);
        let listing = Listing {
            total: 3,
            page: &entries,
        };
        assert_eq!(
            dir_key("r/a/", &listing, None),
            "v2:717794e220eb93b6dd9585eca5034757489450c3b49532f42057f23df22e749c"
        );
        let hint = ModuleHint::CargoPackage {
            name: "demo".into(),
        };
        assert_eq!(
            dir_key("r/a/", &listing, Some(&hint)),
            "v2:8fea126301149e52c287e26a280d4e7681799142cabd277541b19c1c7c0be435"
        );
    }

    /// Every key states the version it was written under, and a key from
    /// before versions reads as 1.
    #[test]
    fn a_keys_version_is_read_from_its_prefix() {
        let listing = Listing {
            total: 0,
            page: &[],
        };
        assert_eq!(key_version(&file_key("r/a.rs", &oid(A))), KEY_VERSION);
        assert_eq!(key_version(&dir_key("r/", &listing, None)), KEY_VERSION);
        assert_eq!(
            key_version(&format!("r/a.rs@{A}")),
            1,
            "a file key before versions"
        );
        assert_eq!(
            key_version("b7b7753a12c14492a0ff6e2f48638351766fdaa2ba4336eaa42cd14d8c4f146a"),
            1,
            "a folder key before versions"
        );
        assert_eq!(key_version("v3:r/a.rs@x"), 3);
        assert_eq!(
            key_version("vendor/a.rs@x"),
            1,
            "a path starting with v is not a version"
        );
    }

    /// Every part of what the folder shows moves its key: the hint, an entry,
    /// the entry count past the page, the folder itself.
    #[test]
    fn what_a_folder_shows_moves_its_key() {
        let one = page(&["r/a/x.rs"]);
        let two = page(&["r/a/x.rs", "r/a/z/"]);
        let demo = ModuleHint::CargoPackage {
            name: "demo".into(),
        };
        let key = |dir: &str, total: usize, entries: &[String], hint: &ModuleHint| {
            let listing = Listing {
                total,
                page: entries,
            };
            dir_key(dir, &listing, Some(hint))
        };
        let base = key("r/a/", 1, &one, &demo);
        for other in [
            key("r/a/", 1, &one, &ModuleHint::CargoWorkspace),
            key("r/a/", 2, &two, &demo),
            key("r/a/", 51, &one, &demo),
            key("r/b/", 1, &one, &demo),
        ] {
            assert_ne!(base, other);
        }
        assert_eq!(base, key("r/a/", 1, &one, &demo));
    }
}
