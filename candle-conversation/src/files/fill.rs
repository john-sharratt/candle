//! What of a kind's files a projection places
//! (`docs/provenance_system_collection.md` §4).
//!
//! The kind's files go strongest first under the kind's share of the room. A
//! file that fits is placed whole — its heading, every chunk, its closing — as
//! the probes placed files; one that does not fit is placed as many of its
//! leading chunks as fit, the file's own opening first, and the next file is
//! tried in what remains. A file none of whose chunks fit is not placed.

use super::key::FileKey;
use super::record::FileRecord;

/// The tokens a placed file costs beyond its chunks: its heading and its
/// closing piece.
pub fn frame_tokens(record: &FileRecord, closing: u32) -> usize {
    record.heading.tokens as usize + closing as usize
}

/// One file as a projection places it: its key and the indices of its
/// chunks placed, in line order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilePlacement {
    pub file: FileKey,
    pub chunks: Vec<usize>,
}

/// The files of `ranked` — strongest first, each found in `record_of` — placed
/// under `cap` tokens, a closing piece costing `closing` tokens.
pub fn choose<'a>(
    ranked: &[(FileKey, f32)],
    record_of: impl Fn(FileKey) -> Option<&'a FileRecord>,
    cap: usize,
    closing: u32,
) -> Vec<FilePlacement> {
    let mut left = cap;
    let mut out = Vec::new();
    for &(file, _) in ranked {
        let Some(record) = record_of(file) else {
            continue;
        };
        let frame = frame_tokens(record, closing);
        if left <= frame {
            continue;
        }
        let mut room = left - frame;
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
            heading: member(10),
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

    /// **Strongest first, whole when it fits, its leading chunks when it does
    /// not, and the next file tried in what remains**: under 1,000 tokens, a
    /// (10 + 300 + 300 + 2 = 612) whole; b's first chunk (10 + 200 + 2) fits
    /// the 388 left and its second does not; c (10 + 500 + 2) no longer fits.
    #[test]
    fn files_are_placed_strongest_first_whole_or_by_their_leading_chunks() {
        let a = record("a.rs", &[300, 300]);
        let b = record("b.rs", &[200, 400]);
        let c = record("c.rs", &[500]);
        let records = [a.clone(), b.clone(), c.clone()];
        let ranked = [(a.key(), 900.0), (b.key(), 800.0), (c.key(), 700.0)];
        let placed = choose(&ranked, |k| records.iter().find(|r| r.key() == k), 1_000, 2);
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
        assert!(choose(&ranked, |_| None, 1_000, 2).is_empty());
        assert!(choose(&ranked, |k| records.iter().find(|r| r.key() == k), 12, 2).is_empty());
    }
}
