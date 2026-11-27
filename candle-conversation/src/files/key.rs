//! What names a file of the knowledge base and a run of its lines.
//!
//! A file is named by its **content key**, `v{N}:{path}@{blob}` — the
//! notation's version, the workspace path, and the git blob (a folder: the
//! hash of its listing). [`FileKey`] is that key digested to 64 bits; the path
//! and blob themselves live on the file's record. A [`LineSpan`] counts lines
//! as `file_read`'s pages do, from 1 and inclusive at both ends.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::persistence::content_hash::hash_bytes;
use crate::projection::TimelineId;

/// A file of the knowledge base: the first 8 bytes of [`hash_bytes`] over its
/// content key, read little-endian — the hash's `lo` lane, the first half of
/// `ContentHash::to_bytes`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FileKey(pub u64);

impl FileKey {
    /// The key of the file whose content key is `content_key`.
    pub fn of(content_key: &str) -> FileKey {
        let bytes = hash_bytes(content_key.as_bytes()).to_bytes();
        let mut first = [0u8; 8];
        first.copy_from_slice(&bytes[..8]);
        FileKey(u64::from_le_bytes(first))
    }

    /// The key as 16 lower-case hex digits, zero-padded.
    pub fn hex(self) -> String {
        format!("{:016x}", self.0)
    }

    /// The key [`FileKey::hex`] wrote, or `None` for anything that is not
    /// exactly 16 hex digits.
    pub fn from_hex(hex: &str) -> Option<FileKey> {
        if hex.len() != 16 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        u64::from_str_radix(hex, 16).ok().map(FileKey)
    }

    /// The file as the id the scan and the calibration index name it by,
    /// where a reading's timeline once stood: its key as the raw id, never
    /// the reserved 0.
    pub fn scan_id(self) -> TimelineId {
        TimelineId::from_raw(self.0.max(1)).expect("an id made with max(1) is never the reserved 0")
    }

    /// The file a [`FileKey::scan_id`] names.
    pub fn of_scan_id(id: TimelineId) -> FileKey {
        FileKey(id.raw())
    }
}

impl fmt::Display for FileKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.hex())
    }
}

/// The content key of `path` at `blob` under notation version `version`:
/// `v{version}:{path}@{blob}`.
pub fn content_key(version: u32, path: &str, blob: &str) -> String {
    format!("v{version}:{path}@{blob}")
}

/// Lines `first..=last` of a file, numbered from 1.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct LineSpan {
    pub first: u32,
    pub last: u32,
}

impl LineSpan {
    /// Whether this span starts on the line after `before` ends: the two are
    /// one run of the file.
    pub fn continues(self, before: LineSpan) -> bool {
        before.last + 1 == self.first
    }
}

/// A chunk of a file: the file and the lines it holds.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct ChunkKey {
    pub file: FileKey,
    pub lines: LineSpan,
}

/// A section of a file of the knowledge base — the unit provenance scores, a
/// calibration votes for and the fill places (`docs/placed_section_layout.md`
/// §3.5c): the first 8 bytes of [`hash_bytes`] over `{content key}#{first}-
/// {last}`, read little-endian, as [`FileKey`] is over the content key alone.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SectionKey(pub u64);

impl SectionKey {
    /// The key of the section holding `lines` of the file whose content key
    /// is `content_key`.
    pub fn of(content_key: &str, lines: LineSpan) -> SectionKey {
        let name = format!("{content_key}#{}-{}", lines.first, lines.last);
        SectionKey(FileKey::of(&name).0)
    }

    /// The key as 16 lower-case hex digits, zero-padded.
    pub fn hex(self) -> String {
        format!("{:016x}", self.0)
    }

    /// The section as the id the scan and the calibration index name it by:
    /// its key as the raw id, never the reserved 0.
    pub fn scan_id(self) -> TimelineId {
        TimelineId::from_raw(self.0.max(1)).expect("an id made with max(1) is never the reserved 0")
    }

    /// The section a [`SectionKey::scan_id`] names.
    pub fn of_scan_id(id: TimelineId) -> SectionKey {
        SectionKey(id.raw())
    }
}

impl fmt::Display for SectionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.hex())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A file's key is the first 8 bytes of the content hash of its content
    /// key**, little-endian: pinned as a number, so a change to the hash or to
    /// the byte order cannot pass unnoticed.
    #[test]
    fn file_key_is_the_digest_of_its_content_key() {
        let key = content_key(1, "candle/zend/src/ingest_lease.rs", "3f2a9c");
        assert_eq!(key, "v1:candle/zend/src/ingest_lease.rs@3f2a9c");
        let bytes = hash_bytes(key.as_bytes()).to_bytes();
        assert_eq!(
            FileKey::of(&key),
            FileKey(u64::from_le_bytes(bytes[..8].try_into().unwrap()))
        );
        assert_eq!(FileKey::of(&key), FileKey(0x473e_088d_5653_2900));
        assert_eq!(FileKey::of(&key).hex(), "473e088d56532900");
    }

    #[test]
    fn a_file_key_round_trips_through_its_hex() {
        let k = FileKey(0x00ab_cdef_0123_4567);
        assert_eq!(k.hex(), "00abcdef01234567");
        assert_eq!(k.to_string(), "00abcdef01234567");
        assert_eq!(FileKey::from_hex("00abcdef01234567"), Some(k));
        assert_eq!(FileKey::from_hex("abcdef01234567"), None);
        assert_eq!(FileKey::from_hex("+0abcdef01234567"), None);
        assert_eq!(FileKey::from_hex("00abcdef0123456g"), None);
    }

    /// **A file's scan id is its key**, and names the file back.
    #[test]
    fn a_file_key_round_trips_through_its_scan_id() {
        let k = FileKey(0x473e_088d_5653_2900);
        assert_eq!(k.scan_id().raw(), 0x473e_088d_5653_2900);
        assert_eq!(FileKey::of_scan_id(k.scan_id()), k);
    }

    /// **A section's key is the digest of its file's content key and its
    /// lines**, pinned as a number; its scan id names it back.
    #[test]
    fn a_section_key_is_the_digest_of_its_file_and_lines() {
        let key = content_key(1, "candle/zend/src/ingest_lease.rs", "3f2a9c");
        let lines = LineSpan {
            first: 201,
            last: 400,
        };
        let s = SectionKey::of(&key, lines);
        assert_eq!(s.0, FileKey::of(&format!("{key}#201-400")).0);
        assert_ne!(s.0, FileKey::of(&key).0);
        assert_eq!(SectionKey::of_scan_id(s.scan_id()), s);
        assert_eq!(s.hex().len(), 16);
    }

    #[test]
    fn a_file_key_serializes_as_its_number() {
        let k = ChunkKey {
            file: FileKey(7),
            lines: LineSpan {
                first: 1,
                last: 200,
            },
        };
        let json = serde_json::to_string(&k).unwrap();
        assert_eq!(json, r#"{"file":7,"lines":{"first":1,"last":200}}"#);
        assert_eq!(serde_json::from_str::<ChunkKey>(&json).unwrap(), k);
    }

    #[test]
    fn a_span_continues_the_one_ending_on_the_line_before() {
        let a = LineSpan {
            first: 1,
            last: 200,
        };
        assert!(LineSpan {
            first: 201,
            last: 400
        }
        .continues(a));
        assert!(!LineSpan {
            first: 202,
            last: 400
        }
        .continues(a));
        assert!(!a.continues(a));
    }
}
