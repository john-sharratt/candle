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
//! (`Tokens`, `StreamDecl`, `Commit`, `ProjectionEvents`, `WideQSig`)
//! by `stream_id`; the workspace singletons by type. `ConvState` carries its
//! timeline in the header's `stream_id` and is keyed by it. The other
//! timeline-keyed metadata types (`Label`, `TreeMetadata`, `DebugId`,
//! `Tombstone`) carry their key inside the payload, which the header
//! scan doesn't decode — they are sector-sized records whose dead
//! weight is negligible next to chunk bytes, so they are left out and
//! the dead estimate stays conservative (it can under-count dead
//! weight, never over-count it).
//!
//! Records made dead by a **timeline tombstone** (every record of a deleted
//! timeline's streams) are invisible to the header-keyed map, since the
//! tombstone names its timeline in the payload; `Substrate::tombstoned_stream_bytes`
//! sums them from the in-RAM stream index and the compaction trigger adds
//! that on top. A **section tombstone** names its stream in the header, so it
//! retires that stream's records here directly — see [`RecordAccounting::record`].

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
    ///
    /// A `SectionTombstone` also retires every live record its stream owns:
    /// the tombstone ends that generation of the section
    /// (`Substrate::reset_tombstoned_stream`), and a fresh prefill of the same
    /// content writes a new one under the same keys — which, with the old keys
    /// retired here, count as new rather than as superseding the dead bytes a
    /// second time. That retirement walks the live map, which is fine for a
    /// record written once per section rebuild or repair.
    pub fn record(&mut self, header: &RecordHeader, padded_size: u64) {
        if header.record_type == RecordType::SectionTombstone {
            self.retire_stream(header.stream_id);
        }
        let key = match header.record_type {
            RecordType::Chunk => (RecordType::Chunk, header.stream_id, header.chunk_index),
            // A conversation's file events and tombstones are keyed by timeline
            // and sequence number, each unique: nothing supersedes one. An
            // event dies by a tombstone or with its timeline, and the index
            // that sees that says so through `retire`.
            RecordType::VfsEvent | RecordType::VfsTombstone => {
                (header.record_type, header.stream_id, header.chunk_index)
            }
            // `Snapshot` is header-keyed by a synthetic per-timeline stream
            // id: the newest snapshot supersedes the previous one here — this
            // insert-returning-old IS the single-tail tombstone (design doc
            // `qwen35_qwen38_models.md` §5.2).
            RecordType::Tokens
            | RecordType::StreamDecl
            | RecordType::Commit
            | RecordType::ProjectionEvents
            | RecordType::WideQSig
            | RecordType::TurnIndexPage
            // `Npc` carries its `npc_id` in the header's `stream_id`, so the
            // newest record for a character supersedes every earlier one here
            // mechanically — the same trick `Snapshot` uses for its per-timeline
            // tail. Putting the key in the header rather than the payload is
            // what lets supersession be seen without decoding anything.
            | RecordType::Npc
            | RecordType::Snapshot
            // A branch checkpoint supersedes by the same rule: one live record
            // per branch, keyed by the branch's content prefix in the header.
            | RecordType::BranchCheckpoint
            // A section tombstone is keyed by the section's `StreamId` in the
            // header too — one live marker per section, same mechanical
            // supersession as `Npc` and the branch checkpoint.
            | RecordType::SectionTombstone
            // A conversation's state carries its timeline id in the header's
            // `stream_id` and is written whole every time, so the newest record
            // supersedes the last by the same rule.
            | RecordType::ConvState => (header.record_type, header.stream_id, 0),
            RecordType::ModelSpec | RecordType::Template | RecordType::Tokenizer => {
                (header.record_type, 0, 0)
            }
            // Payload-keyed metadata records — excluded (see module doc).
            // `HeaderIndex` records are excluded too: they're derived
            // data with no supersession key, reclaimed wholesale at
            // compaction. `Distilled` markers are payload-keyed and
            // consumed by the next compaction pass.
            RecordType::Label
            | RecordType::TreeMetadata
            | RecordType::DebugId
            | RecordType::Tombstone
            | RecordType::Distilled
            | RecordType::TurnCoupling
            | RecordType::HeaderIndex
            | RecordType::Unknown => return,
        };
        if let Some(old) = self.live_sizes.insert(key, padded_size) {
            self.dead_bytes += old;
        }
    }

    /// Count the live record keyed `(rt, stream_id, index)` as dead — for a
    /// record nothing supersedes but something else killed: a file event its
    /// tombstone named, or whose timeline was tombstoned. A key not live is
    /// left alone, so retiring twice counts once.
    pub fn retire(&mut self, rt: RecordType, stream_id: u64, index: u64) {
        if let Some(size) = self.live_sizes.remove(&(rt, stream_id, index)) {
            self.dead_bytes += size;
        }
    }

    /// Count every live record `stream_id` owns as dead — its chunks and its
    /// per-stream records. Keyed only on the stream-owned types, since
    /// `Snapshot`, `Npc` and `ConvState` reuse the header's `stream_id` for
    /// ids of their own.
    fn retire_stream(&mut self, stream_id: u64) {
        let mut retired = 0u64;
        self.live_sizes.retain(|&(rt, sid, _), size| {
            let owned = sid == stream_id
                && matches!(
                    rt,
                    RecordType::Chunk
                        | RecordType::Tokens
                        | RecordType::StreamDecl
                        | RecordType::Commit
                        | RecordType::ProjectionEvents
                        | RecordType::WideQSig
                        | RecordType::TurnIndexPage
                );
            if owned {
                retired += *size;
            }
            !owned
        });
        self.dead_bytes += retired;
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

    /// **A conversation's new state supersedes its last**, keyed by the
    /// timeline in the header, and never another conversation's.
    #[test]
    fn conv_state_supersedes_per_timeline() {
        let mut acc = RecordAccounting::new();
        acc.record(&header(RecordType::ConvState, 11, 0), 4096);
        acc.record(&header(RecordType::ConvState, 12, 0), 4096);
        assert_eq!(acc.dead_bytes(), 0, "two conversations, both live");
        acc.record(&header(RecordType::ConvState, 11, 0), 4096);
        assert_eq!(acc.dead_bytes(), 4096, "11's first state is dead");
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

    /// **A file event is live until something retires it** — its sequence
    /// number is unique, so no later record supersedes it — and retiring it
    /// twice counts it once.
    #[test]
    fn a_file_event_is_live_until_retired() {
        let mut acc = RecordAccounting::new();
        acc.record(&header(RecordType::VfsEvent, 7, 1), 4096);
        acc.record(&header(RecordType::VfsEvent, 7, 2), 8192);
        assert_eq!(acc.dead_bytes(), 0);
        acc.retire(RecordType::VfsEvent, 7, 2);
        acc.retire(RecordType::VfsEvent, 7, 2);
        assert_eq!(acc.dead_bytes(), 8192);
        acc.retire(RecordType::VfsEvent, 7, 9);
        assert_eq!(acc.dead_bytes(), 8192, "a key never live retires nothing");
    }

    /// **A section tombstone retires its stream's records, and only its
    /// stream's** — a fresh generation written under the same keys afterwards
    /// starts live without counting the dead ones a second time.
    #[test]
    fn a_section_tombstone_retires_its_stream() {
        let mut acc = RecordAccounting::new();
        acc.record(&header(RecordType::StreamDecl, 40, 0), 4096);
        acc.record(&header(RecordType::Chunk, 40, 0), 8192);
        acc.record(&header(RecordType::Chunk, 40, 1), 8192);
        acc.record(&header(RecordType::Tokens, 40, 0), 4096);
        acc.record(&header(RecordType::Chunk, 41, 0), 8192);
        acc.record(&header(RecordType::Npc, 40, 0), 4096);
        assert_eq!(acc.dead_bytes(), 0);

        acc.record(&header(RecordType::SectionTombstone, 40, 0), 4096);
        assert_eq!(
            acc.dead_bytes(),
            4096 + 8192 + 8192 + 4096,
            "stream 40's decl, chunks and tokens; not stream 41, not the Npc keyed 40"
        );

        acc.record(&header(RecordType::StreamDecl, 40, 0), 4096);
        acc.record(&header(RecordType::Chunk, 40, 0), 8192);
        assert_eq!(
            acc.dead_bytes(),
            4096 + 8192 + 8192 + 4096,
            "the new generation is live and supersedes nothing"
        );
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
