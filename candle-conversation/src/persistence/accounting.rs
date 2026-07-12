//! Live/dead byte accounting for the redo log.
//!
//! The log is append-only and last-writer-wins: every re-append of the
//! same key (a partial-tail re-snapshot, a re-written `Tokens` record, a
//! superseded singleton) leaves the previous record behind as
//! unreachable dead weight on disk. This module tracks that dead weight
//! incrementally — O(1) per append — so the automatic compaction
//! trigger never has to walk the log to measure it.
//!
//! Keys are derived from the record header alone: `Chunk` is keyed by
//! `(stream_id, chunk_index)`; the per-stream last-writer-wins types
//! (`Tokens`, `Signatures`, `StreamDecl`, `Commit`, `ProjectionEvents`)
//! by `stream_id`; the workspace singletons by type. The timeline-keyed
//! metadata types (`Label`, `ConvState`, `TreeMetadata`, `DebugId`,
//! `Tombstone`) carry their key inside the payload, which the header
//! scan doesn't decode — they are sector-sized records whose dead
//! weight is negligible next to chunk bytes, so they are left out and
//! the dead estimate stays conservative (it can under-count dead
//! weight, never over-count it).
//!
//! Records made dead by a **tombstone** (every record of a deleted
//! timeline's streams) are also invisible to the header-keyed map;
//! `Substrate::tombstoned_stream_bytes` sums them from the in-RAM
//! stream index and the compaction trigger adds that on top.

use std::collections::HashMap;

use super::record::{RecordHeader, RecordType};

/// Incremental last-writer-wins byte accounting over appended records.
#[derive(Debug, Default)]
pub struct RecordAccounting {
    /// Padded on-disk size of the current live record per key.
    live_sizes: HashMap<(RecordType, u64, u64), u64>,
    /// Total padded bytes of superseded (dead) records.
    dead_bytes: u64,
}

impl RecordAccounting {
    pub fn new() -> RecordAccounting {
        RecordAccounting::default()
    }

    /// Note one appended (or recovery-walked) record. O(1): when the
    /// key was already live, its previous on-disk size becomes dead
    /// weight.
    pub fn record(&mut self, header: &RecordHeader, padded_size: u64) {
        let key = match header.record_type {
            RecordType::Chunk => (RecordType::Chunk, header.stream_id, header.chunk_index),
            RecordType::Tokens
            | RecordType::Signatures
            | RecordType::StreamDecl
            | RecordType::Commit
            | RecordType::ProjectionEvents => (header.record_type, header.stream_id, 0),
            RecordType::ModelSpec
            | RecordType::Template
            | RecordType::Tokenizer
            | RecordType::ToolSummary => (header.record_type, 0, 0),
            // Payload-keyed metadata records — excluded (see module doc).
            // `HeaderIndex` records are excluded too: they're derived
            // data with no supersession key, reclaimed wholesale at
            // compaction.
            RecordType::Label
            | RecordType::ConvState
            | RecordType::TreeMetadata
            | RecordType::DebugId
            | RecordType::Tombstone
            | RecordType::HeaderIndex
            | RecordType::Unknown => return,
        };
        if let Some(old) = self.live_sizes.insert(key, padded_size) {
            self.dead_bytes += old;
        }
    }

    /// Total padded bytes of superseded records seen so far.
    pub fn dead_bytes(&self) -> u64 {
        self.dead_bytes
    }

    /// Drop all state — called when the log is rewritten (compaction)
    /// right before the new file is re-walked into fresh accounting.
    pub fn reset(&mut self) {
        self.live_sizes.clear();
        self.dead_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(rt: RecordType, stream_id: u64, chunk_index: u64) -> RecordHeader {
        RecordHeader {
            record_type: rt,
            format: 0,
            payload_len: 0,
            crc: 0,
            stream_id,
            chunk_index,
            token_count: 0,
        }
    }

    #[test]
    fn superseded_chunk_counts_as_dead() {
        let mut acc = RecordAccounting::new();
        acc.record(&header(RecordType::Chunk, 5, 0), 4096);
        acc.record(&header(RecordType::Chunk, 5, 1), 4096);
        assert_eq!(acc.dead_bytes(), 0, "distinct keys are all live");
        acc.record(&header(RecordType::Chunk, 5, 0), 8192);
        assert_eq!(acc.dead_bytes(), 4096, "the first (5,0) write is dead");
        acc.record(&header(RecordType::Chunk, 5, 0), 8192);
        assert_eq!(acc.dead_bytes(), 4096 + 8192);
    }

    #[test]
    fn per_stream_and_singleton_keys() {
        let mut acc = RecordAccounting::new();
        acc.record(&header(RecordType::Tokens, 1, 0), 4096);
        acc.record(&header(RecordType::Tokens, 2, 0), 4096);
        acc.record(&header(RecordType::Tokens, 1, 0), 4096);
        assert_eq!(acc.dead_bytes(), 4096, "streams don't shadow each other");
        acc.record(&header(RecordType::ModelSpec, 0, 0), 4096);
        acc.record(&header(RecordType::ModelSpec, 0, 0), 4096);
        assert_eq!(acc.dead_bytes(), 8192, "superseded singleton is dead");
    }

    /// Commit records reuse `chunk_index` as `through_index` — they must
    /// key per stream, not per index, or every re-commit looks live.
    #[test]
    fn commits_key_per_stream_not_per_index() {
        let mut acc = RecordAccounting::new();
        acc.record(&header(RecordType::Commit, 9, 3), 4096);
        acc.record(&header(RecordType::Commit, 9, 7), 4096);
        assert_eq!(acc.dead_bytes(), 4096);
    }

    /// Payload-keyed metadata types and the derived `HeaderIndex`
    /// records are excluded — never counted dead.
    #[test]
    fn payload_keyed_types_are_skipped() {
        let mut acc = RecordAccounting::new();
        for _ in 0..3 {
            acc.record(&header(RecordType::Label, 0, 0), 4096);
            acc.record(&header(RecordType::TreeMetadata, 0, 0), 4096);
            acc.record(&header(RecordType::HeaderIndex, 0, 0), 4096);
        }
        assert_eq!(acc.dead_bytes(), 0);
    }

    #[test]
    fn reset_clears_everything() {
        let mut acc = RecordAccounting::new();
        acc.record(&header(RecordType::Chunk, 1, 0), 4096);
        acc.record(&header(RecordType::Chunk, 1, 0), 4096);
        assert_eq!(acc.dead_bytes(), 4096);
        acc.reset();
        assert_eq!(acc.dead_bytes(), 0);
        acc.record(&header(RecordType::Chunk, 1, 0), 4096);
        assert_eq!(acc.dead_bytes(), 0, "post-reset state starts fresh");
    }
}
