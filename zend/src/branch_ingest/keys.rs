//! Content keys — what an ingested unit is found by, whatever commit it was
//! found on (`docs/zend_branch_ingest.md` §6).

use std::fmt::Write;

use sha2::{Digest, Sha256};
use zend_vfs::Oid;

use crate::repo_scan::types::ModuleHint;

/// Conversation metadata: the unit's content key. Written only once its
/// ingest succeeds, so a conversation carrying it is committed.
pub const CONTENT_KEY: &str = "content_key";

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

/// [`BRANCHES_KEY`]'s value for `names`.
pub fn branches_value(names: &[String]) -> String {
    names.join(",")
}

/// A file's key: its workspace-relative path and its blob id —
/// `candle/zend/src/main.rs@3f73f872…`. The path is in it because the
/// ingested conversation names the path; the blob id is git's hash of the
/// bytes `file_read` returns.
pub fn file_key(path: &str, blob: &Oid) -> String {
    format!("{path}@{blob}")
}

/// What a folder's listing turn shows: how many entries the folder holds,
/// and the entries on the listing's first page, workspace-relative, a folder
/// ending in `/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listing<'a> {
    pub total: usize,
    pub page: &'a [String],
}

/// A folder's key: SHA-256, in hex, over the folder (workspace-relative,
/// `candle/zend/src/`, or `.` for the workspace), its [`Listing`], and the
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
    let mut out = String::with_capacity(64);
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
            "candle/zend/src/main.rs@ce013625030ba8dba906f756967f9e9ca394464a"
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
            "b7b7753a12c14492a0ff6e2f48638351766fdaa2ba4336eaa42cd14d8c4f146a"
        );
        let hint = ModuleHint::CargoPackage {
            name: "demo".into(),
        };
        assert_eq!(
            dir_key("r/a/", &listing, Some(&hint)),
            "76de6507dbd2ffaec073b92ea3734d95c90bdbf8778ee1952081a80b6d53f1eb"
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
