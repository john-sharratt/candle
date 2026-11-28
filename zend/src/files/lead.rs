//! What a file's members are sealed behind ahead of the file itself: its
//! repository's root listing (`docs/placed_section_layout.md` §3.5e).
//!
//! A section sealed with nothing before it reads as the only one of its kind:
//! on BlackWorld (2026-11-28) the decode took `yenbot/Cargo.toml`'s `serde`
//! line for candle's whenever yenbot's sat nearer the question, and a fresh
//! prefill of the same prompt never did. Sealed behind the listing that names
//! the repository it is in, every section is computed as a part of that
//! repository read after its tree, not as the first thing read. The listing's
//! K/V is the one the knowledge base holds already, reused as a seed chain
//! reuses its sections, so the lead costs no prefill of its own.
//!
//! The repository layer (`repo_map`) is ingested before the files in every
//! pass (`branch_ingest`), so a file finds its listing sealed; one that does
//! not — a repository whose listing this process holds no version of — is
//! sealed behind nothing, as before.

use candle_conversation::files::names::chunk_name;
use candle_conversation::files::{files_owner, FileRecord};
use candle_conversation::projection::{Conversation, SectionId};

use super::records;

/// The path of the root listing of the repository `path` is in: its first
/// component, as a folder.
pub fn listing_path(path: &str) -> String {
    let repo = path.split('/').next().unwrap_or(path);
    format!("{repo}/")
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

/// The sections the file at `path`, on `branches`, is sealed behind: its
/// repository listing's, in order, when this process holds every one of
/// them; none otherwise, and none for the listing itself.
pub fn lead_of(conversation: &Conversation, path: &str, branches: &[String]) -> Vec<SectionId> {
    let listing = listing_path(path);
    if path == listing {
        return Vec::new();
    }
    let versions = records::of_path(conversation, &listing);
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
        FileRecord {
            content_key: format!("v1:candle/@{blob}"),
            path: "candle/".into(),
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

    /// **A file's listing is its repository's root**, whatever its depth.
    #[test]
    fn a_files_listing_is_its_repositorys_root() {
        assert_eq!(listing_path("candle/Cargo.toml"), "candle/");
        assert_eq!(listing_path("yenbot/src/main.rs"), "yenbot/");
        assert_eq!(listing_path("candle/"), "candle/");
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
    }
}
