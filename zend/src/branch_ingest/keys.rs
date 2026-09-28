//! Content keys — what an ingested unit is found by, whatever commit it was
//! found on (`docs/zend_branch_ingest.md` §6).

use std::fmt::Write;

use sha2::{Digest, Sha256};
use zend_vfs::Oid;

/// Conversation metadata: the unit's content key. Written only once its
/// ingest succeeds, so a conversation carrying it is committed.
pub const CONTENT_KEY: &str = "content_key";

/// Conversation metadata: a file unit's blob id.
pub const BLOB_KEY: &str = "blob";

/// Conversation metadata: how many lines a file unit's file holds — what a
/// fast-path answer tells the model it already has.
pub const LINES_KEY: &str = "lines";

/// A file's key: its workspace-relative path and its blob id —
/// `candle/zend/src/main.rs@3f73f872…`. The path is in it because the
/// ingested conversation names the path; the blob id is git's hash of the
/// bytes `file_read` returns.
pub fn file_key(path: &str, blob: &Oid) -> String {
    format!("{path}@{blob}")
}

/// A file a folder's turns show the content of — its anchor or a manifest —
/// named by its workspace-relative path and blob id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shown<'a> {
    pub path: &'a str,
    pub blob: &'a Oid,
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
/// path and blob id of its anchor and of every manifest — everything its
/// turns show but the sizes the listing prints beside each file, which move
/// with every edit of a file the folder only names.
pub fn dir_key(
    dir: &str,
    listing: &Listing<'_>,
    anchor: Option<Shown<'_>>,
    manifests: &[Shown<'_>],
) -> String {
    let mut h = Sha256::new();
    h.update(dir.as_bytes());
    h.update(b"\n");
    h.update(listing.total.to_string().as_bytes());
    h.update(b"\n");
    for entry in listing.page {
        h.update(entry.as_bytes());
        h.update(b"\n");
    }
    if let Some(a) = anchor {
        shown(&mut h, "anchor", &a);
    }
    for m in manifests {
        shown(&mut h, "manifest", m);
    }
    let mut out = String::with_capacity(64);
    for b in h.finalize() {
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn shown(h: &mut Sha256, role: &str, file: &Shown<'_>) {
    h.update(b"\0");
    h.update(role.as_bytes());
    h.update(b"\0");
    h.update(file.path.as_bytes());
    h.update(b"\0");
    h.update(file.blob.as_str().as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "ce013625030ba8dba906f756967f9e9ca394464a";
    const B: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

    fn oid(hex: &str) -> Oid {
        Oid::parse(hex).unwrap()
    }

    fn page(entries: &[&str]) -> Vec<String> {
        entries.iter().map(|e| e.to_string()).collect()
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
            dir_key("r/a/", &listing, None, &[]),
            "b7b7753a12c14492a0ff6e2f48638351766fdaa2ba4336eaa42cd14d8c4f146a"
        );
        let (a, b) = (oid(A), oid(B));
        let anchor = Shown {
            path: "r/a/mod.rs",
            blob: &a,
        };
        let manifest = Shown {
            path: "r/a/Cargo.toml",
            blob: &b,
        };
        assert_eq!(
            dir_key("r/a/", &listing, Some(anchor), &[manifest]),
            "d05d729d5b482b0915ec18f3a9095495dc3d8d79ebf212caa6d0e32a2c086a0b"
        );
    }

    /// Every part of what the folder shows moves its key: the anchor's blob,
    /// an entry, the entry count past the page, the folder itself.
    #[test]
    fn what_a_folder_shows_moves_its_key() {
        let (a, b) = (oid(A), oid(B));
        let one = page(&["r/a/x.rs"]);
        let two = page(&["r/a/x.rs", "r/a/z/"]);
        let key = |dir: &str, total: usize, entries: &[String], blob: &Oid| {
            let listing = Listing {
                total,
                page: entries,
            };
            let anchor = Shown {
                path: "r/a/mod.rs",
                blob,
            };
            dir_key(dir, &listing, Some(anchor), &[])
        };
        let base = key("r/a/", 1, &one, &a);
        for other in [
            key("r/a/", 1, &one, &b),
            key("r/a/", 2, &two, &a),
            key("r/a/", 51, &one, &a),
            key("r/b/", 1, &one, &a),
        ] {
            assert_ne!(base, other);
        }
        assert_eq!(base, key("r/a/", 1, &one, &a));
    }
}
