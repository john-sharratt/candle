//! A file of the knowledge base as the substrate records it: a custom object
//! under `files/{path}@{blob}`, naming the file and every member it was sealed
//! as (`docs/provenance_system_collection.md` §12 D1, D15).
//!
//! The record names members by their content address, never by a section id:
//! ids are this process's, handed out when a member is sealed or restored, so
//! a record read back after a restart or arriving from another member resolves
//! its members by address.

use serde::{Deserialize, Serialize};

use super::key::{FileKey, LineSpan, SectionKey};
use crate::persistence::record::CustomObjectPayload;
use crate::persistence::streams::ContentAddress;

/// Where every file record's key starts.
pub const FILES_PREFIX: &str = "files/";

/// What a record describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileUnit {
    /// A file's text.
    File,
    /// A folder's listing.
    Folder,
}

/// A member a record names: where it is sealed, and its size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemberRef {
    pub address: ContentAddress,
    pub tokens: u32,
}

/// One chunk of the file and the member it is sealed as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub lines: LineSpan,
    pub member: MemberRef,
}

/// A file of the knowledge base.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileRecord {
    /// `v{N}:{path}@{blob}`, whose digest is the file's [`FileKey`].
    pub content_key: String,
    /// The workspace path, repository first.
    pub path: String,
    /// The git blob the text was read from (a folder: its listing's hash).
    pub blob: String,
    pub unit: FileUnit,
    /// The file's line count.
    pub lines: u32,
    /// The language its extension names (`rs`, `md`, …).
    pub lang: String,
    /// The branches the blob was read on.
    pub branches: Vec<String>,
    /// The sections in line order, each sealed behind the ones before it.
    pub chunks: Vec<ChunkRecord>,
}

impl FileRecord {
    /// The file's key.
    pub fn key(&self) -> FileKey {
        FileKey::of(&self.content_key)
    }

    /// The key of section `i` (`docs/placed_section_layout.md` §3.5c).
    pub fn section_key(&self, i: usize) -> SectionKey {
        SectionKey::of(&self.content_key, self.chunks[i].lines)
    }

    /// Every section's key, in line order.
    pub fn section_keys(&self) -> Vec<SectionKey> {
        (0..self.chunks.len())
            .map(|i| self.section_key(i))
            .collect()
    }

    /// The custom-object key the record is stored under.
    pub fn object_key(&self) -> String {
        object_key(&self.path, &self.blob)
    }

    /// The record as the custom object that stores it.
    pub fn to_object(&self) -> CustomObjectPayload {
        CustomObjectPayload::new(
            self.object_key(),
            serde_json::to_value(self).expect("a file record serializes"),
        )
    }

    /// The record a custom object stores, or `None` for one that is not a
    /// live file record.
    pub fn from_object(obj: &CustomObjectPayload) -> Option<FileRecord> {
        if obj.ceased || !obj.key.starts_with(FILES_PREFIX) {
            return None;
        }
        serde_json::from_value(obj.blob.clone()).ok()
    }

    /// The tokens every section of the record holds.
    pub fn tokens(&self) -> u32 {
        self.chunks.iter().map(|c| c.member.tokens).sum::<u32>()
    }
}

/// The custom-object key of the record of `path` at `blob`.
pub fn object_key(path: &str, blob: &str) -> String {
    format!("{FILES_PREFIX}{path}@{blob}")
}

/// The prefix every record of `path` sits under, whatever its blob.
pub fn path_prefix(path: &str) -> String {
    format!("{FILES_PREFIX}{path}@")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::content_hash::ContentHash;

    fn address(lo: u64) -> ContentAddress {
        ContentAddress {
            prefix_hash: ContentHash { lo, hi: 0 },
            section_hash: ContentHash { lo: lo + 1, hi: 0 },
        }
    }

    fn member(lo: u64, tokens: u32) -> MemberRef {
        MemberRef {
            address: address(lo),
            tokens,
        }
    }

    fn record() -> FileRecord {
        FileRecord {
            content_key: "v1:candle/a.rs@ab12".into(),
            path: "candle/a.rs".into(),
            blob: "ab12".into(),
            unit: FileUnit::File,
            lines: 260,
            lang: "rs".into(),
            branches: vec!["main".into()],
            chunks: vec![
                ChunkRecord {
                    lines: LineSpan {
                        first: 1,
                        last: 200,
                    },
                    member: member(3, 2900),
                },
                ChunkRecord {
                    lines: LineSpan {
                        first: 201,
                        last: 260,
                    },
                    member: member(5, 700),
                },
            ],
        }
    }

    /// **A file record round-trips as its custom object**, under
    /// `files/{path}@{blob}`, its JSON pinned.
    #[test]
    fn a_file_record_round_trips_as_its_custom_object() {
        let r = record();
        let obj = r.to_object();
        assert_eq!(obj.key, "files/candle/a.rs@ab12");
        let json = serde_json::to_string(&obj.blob).unwrap();
        let address = |lo: u64| serde_json::to_string(&address(lo)).unwrap();
        assert_eq!(
            json,
            format!(
                r#"{{"content_key":"v1:candle/a.rs@ab12","path":"candle/a.rs","blob":"ab12","unit":"file","lines":260,"lang":"rs","branches":["main"],"chunks":[{{"lines":{{"first":1,"last":200}},"member":{{"address":{},"tokens":2900}}}},{{"lines":{{"first":201,"last":260}},"member":{{"address":{},"tokens":700}}}}]}}"#,
                address(3),
                address(5)
            )
        );
        assert_eq!(FileRecord::from_object(&obj), Some(r.clone()));
        assert_eq!(r.tokens(), 2900 + 700);
        assert_eq!(r.key(), FileKey::of("v1:candle/a.rs@ab12"));
        assert_eq!(
            r.section_keys(),
            vec![
                SectionKey::of("v1:candle/a.rs@ab12", r.chunks[0].lines),
                SectionKey::of("v1:candle/a.rs@ab12", r.chunks[1].lines),
            ]
        );
    }

    /// A tombstone, or another prefix's object, is no record.
    #[test]
    fn a_ceased_or_foreign_object_is_no_record() {
        assert_eq!(
            FileRecord::from_object(&CustomObjectPayload::ceased("files/candle/a.rs@ab12")),
            None
        );
        let mut obj = record().to_object();
        obj.key = "journal/x".into();
        assert_eq!(FileRecord::from_object(&obj), None);
        assert_eq!(path_prefix("candle/a.rs"), "files/candle/a.rs@");
    }
}
