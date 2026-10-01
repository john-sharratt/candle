//! What a running maintenance op must no longer relocate.
//!
//! A maintenance op is planned under one hold of the persistence lock and then
//! relocates its records in batches, releasing the lock between them so seal
//! writes and cold loads wait for one batch rather than a whole segment. Every
//! record the plan names is copied *forward* into the active segment, so a
//! record written in that window by anyone else lands BEFORE the planned copy
//! that follows it — and the log walk on the next load lets the later record
//! win. Two kinds of write make a planned copy wrong:
//!
//! - **A newer copy of the same record.** A `Chunk` re-sealed (a partial tail
//!   sealed final) or a stream's `Tokens` rewritten: the planned copy is
//!   superseded, and relocating it would roll the record back.
//! - **A section retired.** A `SectionTombstone` ends a section's generation,
//!   and a later declaration of the same content address revives it as a new
//!   one. The old generation's chunks and tokens relocated after the tombstone
//!   would load as part of whatever comes after it — the rebuild never happens,
//!   or the rebuilt section carries leftovers of the retired one.
//!
//! The watch records both from the moment the op is planned. It is also the
//! op's presence: a handle runs one maintenance op at a time, and a second
//! plan while one is in flight would relocate out of segments the first is
//! about to unlink.

use std::collections::HashSet;

use super::record::{RecordHeader, RecordType};

/// The writes made since a maintenance op was planned that its relocation
/// must step around.
#[derive(Debug, Default)]
pub struct RelocationWatch {
    /// `(stream_id, chunk_index)` of every `Chunk` appended.
    chunks: HashSet<(u64, u64)>,
    /// Every stream whose `Tokens` record was appended.
    tokens: HashSet<u64>,
    /// Every section stream a `SectionTombstone` retired.
    retired: HashSet<u64>,
}

impl RelocationWatch {
    /// Note one appended record.
    pub fn record(&mut self, header: &RecordHeader) {
        match header.record_type {
            RecordType::Chunk => {
                self.chunks.insert((header.stream_id, header.chunk_index));
            }
            RecordType::Tokens => {
                self.tokens.insert(header.stream_id);
            }
            RecordType::SectionTombstone => {
                self.retired.insert(header.stream_id);
            }
            _ => {}
        }
    }

    /// Whether the planned copy of chunk `index` of `stream` is still the one
    /// to carry forward.
    pub fn keeps_chunk(&self, stream: u64, index: u64) -> bool {
        !self.chunks.contains(&(stream, index)) && !self.retired.contains(&stream)
    }

    /// Whether the planned copy of `stream`'s `Tokens` is still the one to
    /// carry forward.
    pub fn keeps_tokens(&self, stream: u64) -> bool {
        !self.tokens.contains(&stream) && !self.retired.contains(&stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(record_type: RecordType, stream_id: u64, chunk_index: u64) -> RecordHeader {
        RecordHeader {
            record_type,
            format: 0,
            payload_len: 0,
            crc: 0,
            stream_id,
            chunk_index,
            token_count: 0,
        }
    }

    #[test]
    fn an_unwatched_record_is_carried() {
        let w = RelocationWatch::default();
        assert!(w.keeps_chunk(7, 0));
        assert!(w.keeps_tokens(7));
    }

    /// A re-appended chunk supersedes only its own key.
    #[test]
    fn a_rewritten_chunk_drops_only_that_chunk() {
        let mut w = RelocationWatch::default();
        w.record(&header(RecordType::Chunk, 7, 2));
        assert!(!w.keeps_chunk(7, 2));
        assert!(w.keeps_chunk(7, 1));
        assert!(w.keeps_chunk(8, 2));
        assert!(
            w.keeps_tokens(7),
            "a chunk write leaves the stream's tokens alone"
        );
    }

    /// Tokens are one record per stream, keyed by the stream alone.
    #[test]
    fn rewritten_tokens_drop_that_streams_tokens() {
        let mut w = RelocationWatch::default();
        w.record(&header(RecordType::Tokens, 7, 0));
        assert!(!w.keeps_tokens(7));
        assert!(w.keeps_tokens(8));
        assert!(
            w.keeps_chunk(7, 0),
            "a tokens write leaves the stream's chunks alone"
        );
    }

    /// A retired section carries none of its old generation past the
    /// tombstone — every chunk and its tokens.
    #[test]
    fn a_retired_section_drops_its_whole_old_generation() {
        let mut w = RelocationWatch::default();
        w.record(&header(RecordType::SectionTombstone, 40, 0));
        assert!(!w.keeps_chunk(40, 0));
        assert!(!w.keeps_chunk(40, 9));
        assert!(!w.keeps_tokens(40));
        assert!(w.keeps_chunk(41, 0));
        assert!(w.keeps_tokens(41));
    }

    /// Records that are re-emitted or keyed elsewhere are not watched.
    #[test]
    fn other_record_types_are_ignored() {
        let mut w = RelocationWatch::default();
        w.record(&header(RecordType::Snapshot, 7, 0));
        w.record(&header(RecordType::Tombstone, 7, 0));
        assert!(w.keeps_chunk(7, 0));
        assert!(w.keeps_tokens(7));
    }
}
