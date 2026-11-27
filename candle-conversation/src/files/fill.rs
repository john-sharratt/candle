//! What of a kind's files a projection places
//! (`docs/provenance_system_collection.md` §4).
//!
//! The kind's files go strongest first under the kind's share of the room. A
//! file that fits is placed whole — every section; one that does not fit is
//! placed as many of its leading sections as fit, the file's own opening
//! first, and the next file is tried in what remains. A file none of whose
//! sections fit is not placed. Each section carries its own heading and
//! closing line, so a placed file costs its sections' tokens and nothing more.

use super::key::FileKey;
use super::record::FileRecord;

/// One file as a projection places it: its key and the indices of its
/// sections placed, in line order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilePlacement {
    pub file: FileKey,
    pub chunks: Vec<usize>,
}

/// The files of `ranked` — strongest first, each found in `record_of` — placed
/// under `cap` tokens.
pub fn choose<'a>(
    ranked: &[(FileKey, f32)],
    record_of: impl Fn(FileKey) -> Option<&'a FileRecord>,
    cap: usize,
) -> Vec<FilePlacement> {
    let mut left = cap;
    let mut out = Vec::new();
    for &(file, _) in ranked {
        let Some(record) = record_of(file) else {
            continue;
        };
        let mut room = left;
        let mut chunks = Vec::new();
        for (i, c) in record.chunks.iter().enumerate() {
            let t = c.member.tokens as usize;
            if t > room {
                break;
            }
            room -= t;
            chunks.push(i);
        }
        if chunks.is_empty() {
            continue;
        }
        left = room;
        out.push(FilePlacement { file, chunks });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::{ChunkRecord, FileUnit, LineSpan, MemberRef};
    use crate::persistence::content_hash::ContentHash;
    use crate::persistence::streams::ContentAddress;

    fn member(tokens: u32) -> MemberRef {
        MemberRef {
            address: ContentAddress {
                prefix_hash: ContentHash::ZERO,
                section_hash: ContentHash::ZERO,
            },
            tokens,
        }
    }

    fn record(path: &str, chunks: &[u32]) -> FileRecord {
        FileRecord {
            content_key: format!("v1:{path}@b"),
            path: path.into(),
            blob: "b".into(),
            unit: FileUnit::File,
            lines: 1,
            lang: "rs".into(),
            branches: vec![],
            chunks: chunks
                .iter()
                .enumerate()
                .map(|(i, &t)| ChunkRecord {
                    lines: LineSpan {
                        first: i as u32 * 200 + 1,
                        last: i as u32 * 200 + 200,
                    },
                    member: member(t),
                })
                .collect(),
        }
    }

    /// **Strongest first, whole when it fits, its leading sections when it
    /// does not, and the next file tried in what remains**: under 1,000
    /// tokens, a (300 + 300) whole; b's first section (200) fits the 400 left
    /// and its second (400) does not; c (500) no longer fits the 200 left.
    #[test]
    fn files_are_placed_strongest_first_whole_or_by_their_leading_sections() {
        let a = record("a.rs", &[300, 300]);
        let b = record("b.rs", &[200, 400]);
        let c = record("c.rs", &[500]);
        let records = [a.clone(), b.clone(), c.clone()];
        let ranked = [(a.key(), 900.0), (b.key(), 800.0), (c.key(), 700.0)];
        let placed = choose(&ranked, |k| records.iter().find(|r| r.key() == k), 1_000);
        assert_eq!(
            placed,
            vec![
                FilePlacement {
                    file: a.key(),
                    chunks: vec![0, 1]
                },
                FilePlacement {
                    file: b.key(),
                    chunks: vec![0]
                },
            ]
        );
        assert!(choose(&ranked, |_| None, 1_000).is_empty());
        assert!(choose(&ranked, |k| records.iter().find(|r| r.key() == k), 12).is_empty());
    }
}
