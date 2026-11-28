//! What a file's members are sealed behind ahead of the file itself: its
//! repository's root listing, then the listing of the folder it is in
//! (`docs/placed_section_layout.md` §3.5e).
//!
//! A section sealed with nothing before it reads as the only one of its kind:
//! on BlackWorld (2026-11-28) the decode took `yenbot/Cargo.toml`'s `serde`
//! line for candle's whenever yenbot's sat nearer the question, and a fresh
//! prefill of the same prompt never did. Sealed behind the listing that names
//! the repository it is in, and behind its own folder's, every section is
//! computed as a part of that repository and that folder — its bearings —
//! read after their trees, not as the first thing read. The listings' K/V is
//! the one the knowledge base holds already, reused as a seed chain reuses its
//! sections, so the lead costs no prefill of its own.
//!
//! The repository layer (`repo_map`) is ingested before the files in every
//! pass (`branch_ingest`), so a file finds its listings sealed; a listing this
//! process holds no version of is left out of the lead, and a file with none
//! is sealed behind nothing.

use candle_conversation::files::names::chunk_name;
use candle_conversation::files::{files_owner, FileRecord};
use candle_conversation::projection::{Conversation, SectionId};

use super::records;

/// The listings the file at `path` is led by, in order: its repository's
/// root, then the folder it is in when that is not the root. None for a
/// folder's own listing, which is what leads.
pub fn listing_paths(path: &str) -> Vec<String> {
    if path.ends_with('/') {
        return Vec::new();
    }
    let repo = path.split('/').next().unwrap_or(path);
    let root = format!("{repo}/");
    let folder = path
        .rsplit_once('/')
        .map_or(root.clone(), |(f, _)| format!("{f}/"));
    match folder == root {
        true => vec![root],
        false => vec![root, folder],
    }
}

/// Of a repository's listing `versions`, the one a file on `branches` is read
/// after: the first that shares a branch with it, in the file's branch order;
/// else the first.
pub fn version_for<'a>(versions: &'a [FileRecord], branches: &[String]) -> Option<&'a FileRecord> {
    branches
        .iter()
        .find_map(|b| versions.iter().find(|v| v.branches.contains(b)))
        .or_else(|| versions.first())
}

/// The sections the file at `path`, on `branches`, is sealed behind: each of
/// its listings' ([`listing_paths`]) in order, a listing included only when
/// this process holds every section of it.
pub fn lead_of(conversation: &Conversation, path: &str, branches: &[String]) -> Vec<SectionId> {
    listing_paths(path)
        .iter()
        .flat_map(|listing| held_listing(conversation, listing, branches))
        .collect()
}

/// The sections of the version of `listing` a file on `branches` is read
/// after, when this process holds all of them; none otherwise.
fn held_listing(conversation: &Conversation, listing: &str, branches: &[String]) -> Vec<SectionId> {
    let versions = records::of_path(conversation, listing);
    let Some(record) = version_for(&versions, branches) else {
        return Vec::new();
    };
    let read = conversation.read();
    record
        .chunks
        .iter()
        .map(|c| {
            read.owned_section_named(
                files_owner(),
                &chunk_name(&record.path, &record.blob, c.lines),
            )
        })
        .collect::<Option<Vec<SectionId>>>()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use candle_conversation::files::{ChunkRecord, FileUnit, LineSpan, MemberRef};
    use candle_conversation::persistence::content_hash::ContentHash;
    use candle_conversation::persistence::streams::ContentAddress;

    use super::*;

    fn listing(blob: &str, branches: &[&str]) -> FileRecord {
        folder("candle/", blob, branches)
    }

    fn folder(path: &str, blob: &str, branches: &[&str]) -> FileRecord {
        FileRecord {
            content_key: format!("v1:{path}@{blob}"),
            path: path.into(),
            blob: blob.into(),
            unit: FileUnit::Folder,
            lines: 4,
            lang: String::new(),
            branches: branches.iter().map(|b| b.to_string()).collect(),
            chunks: vec![ChunkRecord {
                lines: LineSpan { first: 1, last: 4 },
                member: MemberRef {
                    address: ContentAddress {
                        prefix_hash: ContentHash { lo: 1, hi: 0 },
                        section_hash: ContentHash { lo: 2, hi: 0 },
                    },
                    tokens: 30,
                },
            }],
        }
    }

    /// **A file is led by its repository's root, then its own folder**; a
    /// root file by the root alone, a folder's listing by nothing.
    #[test]
    fn a_file_is_led_by_its_root_then_its_folder() {
        assert_eq!(listing_paths("candle/Cargo.toml"), vec!["candle/"]);
        assert_eq!(
            listing_paths("yenbot/src/bin/main.rs"),
            vec!["yenbot/", "yenbot/src/bin/"]
        );
        assert!(listing_paths("candle/").is_empty());
        assert!(listing_paths("candle/src/").is_empty());
    }

    /// **The listing version shares the file's branch**, the file's first
    /// branch first; with none shared, the first version.
    #[test]
    fn the_listing_version_shares_the_files_branch() {
        let versions = [listing("aa", &["main"]), listing("bb", &["dev", "main"])];
        let on = |b: &[&str]| -> Vec<String> { b.iter().map(|s| s.to_string()).collect() };
        assert_eq!(version_for(&versions, &on(&["dev"])).unwrap().blob, "bb");
        assert_eq!(version_for(&versions, &on(&["main"])).unwrap().blob, "aa");
        assert_eq!(version_for(&versions, &on(&["other"])).unwrap().blob, "aa");
        assert!(version_for(&[], &on(&["main"])).is_none());
    }

    /// **A file is led by its listing's sections once this process holds them
    /// all; the listing itself, and a file whose listing is not held, by
    /// nothing.**
    #[test]
    fn a_file_is_led_by_its_held_listing() {
        let conversation = Conversation::new();
        records::put(&conversation, &listing("aa", &["main"])).unwrap();
        let main = vec!["main".to_string()];
        assert!(lead_of(&conversation, "candle/Cargo.toml", &main).is_empty());
        let section = conversation.allocate_owned_section().expect("an owned id");
        conversation.register_owned_section(files_owner(), "candle/@aa#1-4", section);
        assert_eq!(
            lead_of(&conversation, "candle/Cargo.toml", &main),
            vec![section]
        );
        assert!(lead_of(&conversation, "candle/", &main).is_empty());
        assert!(lead_of(&conversation, "yenbot/Cargo.toml", &main).is_empty());
        // A deeper file: the root, then its folder once that is held too.
        records::put(&conversation, &folder("candle/src/", "cc", &["main"])).unwrap();
        assert_eq!(
            lead_of(&conversation, "candle/src/lib.rs", &main),
            vec![section],
            "the folder is not held yet"
        );
        let inner = conversation.allocate_owned_section().expect("an owned id");
        conversation.register_owned_section(files_owner(), "candle/src/@cc#1-4", inner);
        assert_eq!(
            lead_of(&conversation, "candle/src/lib.rs", &main),
            vec![section, inner]
        );
    }
}
