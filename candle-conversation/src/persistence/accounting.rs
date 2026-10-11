//! Live/dead byte accounting for the redo log.
//!
//! The log is append-only, and per key the record with the greatest version
//! stamp is the live one: every superseded record of a key (a partial-tail
//! re-snapshot, a re-written `Tokens` record, a superseded singleton) is left
//! behind as unreachable dead weight on disk — the older one when they arrive
//! in stamp order, the newcomer itself when it arrives after a greater one.
//! This module tracks that dead weight incrementally — O(1) per append — so
//! the automatic compaction trigger never has to walk the log to measure it.
//!
//! Keys are derived from the record header alone: `Chunk` is keyed by
//! `(stream_id, chunk_index)`; the per-stream last-writer-wins types
//! (`Tokens`, `StreamDecl`, `Commit`, `ProjectionEvents`, `WideQSig`)
//! by `stream_id`; the workspace singletons by type. A stream's own records
//! are keyed by their writer's origin too: two members' turn *N* share a
//! stream id and are two streams (`docs/replicated_substrate.md` §6.6), so
//! neither supersedes the other. `ConvState` carries its
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

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use super::identity::RecordIdentity;
use super::record::{RecordHeader, RecordType, SectionTombstonePayload};

/// The current record of one key: its padded on-disk size and its identity.
/// A snapshot retraction is current with size zero — it names nothing to
/// carry, and still outranks every older snapshot that arrives after it.
#[derive(Debug, Clone, Copy)]
struct Live {
    size: u64,
    id: RecordIdentity,
}

/// One record key: its type, the header's `stream_id`, a sub-key (a chunk's
/// index, a file event's sequence number), and the writer's origin for a
/// stream's own records ([`stream_owned`]), zero for every other.
type Key = (RecordType, u64, u64, u64);

/// The record types a stream owns — written by the member that declared the
/// stream, under its id.
fn stream_owned(rt: RecordType) -> bool {
    matches!(
        rt,
        RecordType::Chunk
            | RecordType::Tokens
            | RecordType::StreamDecl
            | RecordType::Commit
            | RecordType::ProjectionEvents
            | RecordType::WideQSig
            | RecordType::TurnIndexPage
    )
}

/// Incremental byte accounting over appended records: per key, the record with
/// the greatest version stamp is live (`docs/replicated_substrate.md` §6.5).
/// Records reach the log in arrival order, so one ranked below the record a
/// key already holds is dead the moment it lands.
#[derive(Debug, Default)]
pub struct RecordAccounting {
    /// The current live record per key.
    live_sizes: HashMap<Key, Live>,
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
    /// A `SectionTombstone` also retires every live record of the stream its
    /// generation was written by: the tombstone ends that generation of the
    /// section (`Substrate::end_section`), and a fresh prefill of the same
    /// content writes a new one under the same keys — which, with the old keys
    /// retired here, count as new rather than as superseding the dead bytes a
    /// second time. That retirement walks the live map, which is fine for a
    /// record written once per section rebuild or repair. `payload` is read
    /// for nothing else.
    pub fn record(&mut self, header: &RecordHeader, payload: &[u8], padded_size: u64) {
        if header.record_type == RecordType::SectionTombstone {
            let writer = SectionTombstonePayload::decode(payload)
                .ok()
                .and_then(|p| p.generation)
                .map_or(0, |g| g.writer);
            self.retire_stream(header.stream_id, writer);
        }
        // A snapshot retraction kills the stream's tail when it outranks it,
        // and is itself never carried forward, so it is dead the moment it
        // lands — but it stays the key's current record, so an older snapshot
        // arriving later is dead too.
        if header.retracts_snapshot() {
            self.offer(
                (RecordType::Snapshot, header.stream_id, 0, 0),
                Live {
                    size: 0,
                    id: header.identity(),
                },
            );
            self.dead_bytes += padded_size;
            return;
        }
        let (rt, stream_id, sub) = match header.record_type {
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
            // `CustomObject` carries its key's hash in the header's `stream_id`
            // (see `CustomObjectPayload::stream_id`), so a re-write for the same
            // key supersedes the previous one here mechanically — the resident
            // twin of `StreamDecl`, re-emitted from RAM at compaction.
            | RecordType::CustomObject
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
            // The deleted set is one record, rewritten whole by each compaction
            // — a singleton like these.
            RecordType::ModelKey
            | RecordType::Template
            | RecordType::Tokenizer
            | RecordType::Deleted => (header.record_type, 0, 0),
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
        let writer = if stream_owned(rt) { header.origin } else { 0 };
        self.offer(
            (rt, stream_id, sub, writer),
            Live {
                size: padded_size,
                id: header.identity(),
            },
        );
    }

    /// `incoming` becomes `key`'s live record when it supersedes the one held,
    /// which turns dead; otherwise `incoming` is dead on arrival.
    fn offer(&mut self, key: Key, incoming: Live) {
        match self.live_sizes.entry(key) {
            Entry::Occupied(mut held) => {
                if incoming.id.supersedes(&held.get().id) {
                    self.dead_bytes += held.get().size;
                    held.insert(incoming);
                } else {
                    self.dead_bytes += incoming.size;
                }
            }
            Entry::Vacant(slot) => {
                slot.insert(incoming);
            }
        }
    }

    /// Count the live record keyed `(rt, stream_id, index)` as dead — for a
    /// record nothing supersedes but something else killed: a file event its
    /// tombstone named, or whose timeline was tombstoned. A key not live is
    /// left alone, so retiring twice counts once.
    pub fn retire(&mut self, rt: RecordType, stream_id: u64, index: u64) {
        if let Some(live) = self.live_sizes.remove(&(rt, stream_id, index, 0)) {
            self.dead_bytes += live.size;
        }
    }

    /// Count every live record `writer`'s stream `stream_id` owns as dead — its
    /// chunks and its per-stream records. Keyed only on the stream-owned
    /// types, since `Snapshot`, `Npc` and `ConvState` reuse the header's
    /// `stream_id` for ids of their own.
    fn retire_stream(&mut self, stream_id: u64, writer: u64) {
        let mut retired = 0u64;
        self.live_sizes.retain(|&(rt, sid, _, w), live| {
            let owned = sid == stream_id && w == writer && stream_owned(rt);
            if owned {
                retired += live.size;
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
    use crate::persistence::record::SectionGeneration;

    /// Note a record whose payload the accounting does not read.
    fn record(acc: &mut RecordAccounting, header: &RecordHeader, size: u64) {
        acc.record(header, &[], size);
    }

    fn header(rt: RecordType, stream_id: u64, chunk_index: u64) -> RecordHeader {
        RecordHeader {
            record_type: rt,
            format: 0,
            payload_len: 0,
            crc: 0,
            stream_id,
            chunk_index,
            token_count: 0,
            ..RecordHeader::NONE
        }
    }

    /// A retraction ends the stream's snapshot tail and is never carried
    /// itself, so both records are dead the moment it lands.
    #[test]
    fn a_snapshot_retraction_kills_the_tail_and_itself() {
        let mut acc = RecordAccounting::new();
        let mut snapshot = header(RecordType::Snapshot, 7, 0);
        snapshot.payload_len = 1000;
        record(&mut acc, &snapshot, 4096);
        assert_eq!(acc.dead_bytes(), 0);
        record(&mut acc, &header(RecordType::Snapshot, 7, 0), 64);
        assert_eq!(acc.dead_bytes(), 4096 + 64);
        // The next real snapshot is live again, superseding nothing.
        record(&mut acc, &snapshot, 4096);
        assert_eq!(acc.dead_bytes(), 4096 + 64);
    }

    /// **A record that arrives after a greater one is dead on arrival**, and
    /// the greater one stays live: what counts is the stamp, not the order the
    /// two reached the log in.
    #[test]
    fn an_older_record_arriving_late_is_dead_on_arrival() {
        let stamped = |hlc: u64| RecordHeader {
            hlc,
            origin: 3,
            oseq: hlc,
            ..header(RecordType::Tokens, 9, 0)
        };
        let mut acc = RecordAccounting::new();
        record(&mut acc, &stamped(20), 4096);
        record(&mut acc, &stamped(10), 8192);
        assert_eq!(acc.dead_bytes(), 8192, "the late, older record is dead");
        record(&mut acc, &stamped(30), 4096);
        assert_eq!(acc.dead_bytes(), 8192 + 4096, "the newer one supersedes 20");
        // A relocated copy of the live record keeps its identity: the old
        // copy is dead and the copy is live.
        record(&mut acc, &stamped(30), 4096);
        assert_eq!(acc.dead_bytes(), 8192 + 4096 + 4096);
    }

    /// A retraction outranks the snapshot before it and every older one that
    /// arrives after it.
    #[test]
    fn a_retraction_refuses_an_older_snapshot_arriving_late() {
        let at = |hlc: u64, len: u64| RecordHeader {
            hlc,
            origin: 3,
            oseq: hlc,
            payload_len: len,
            ..header(RecordType::Snapshot, 7, 0)
        };
        let mut acc = RecordAccounting::new();
        record(&mut acc, &at(10, 1000), 4096);
        record(&mut acc, &at(20, 0), 64);
        assert_eq!(acc.dead_bytes(), 4096 + 64);
        record(&mut acc, &at(15, 1000), 4096);
        assert_eq!(
            acc.dead_bytes(),
            4096 + 64 + 4096,
            "older than the retraction"
        );
        record(&mut acc, &at(25, 1000), 4096);
        assert_eq!(acc.dead_bytes(), 4096 + 64 + 4096, "newer: live again");
    }

    /// Two writers' chunk 0 of one stream id are two streams' records: neither
    /// supersedes the other, whatever their stamps.
    #[test]
    fn two_writers_records_of_one_stream_id_are_both_live() {
        let by = |origin: u64, hlc: u64| RecordHeader {
            hlc,
            origin,
            oseq: hlc,
            ..header(RecordType::Chunk, 5, 0)
        };
        let mut acc = RecordAccounting::new();
        record(&mut acc, &by(3, 10), 4096);
        record(&mut acc, &by(8, 20), 4096);
        assert_eq!(acc.dead_bytes(), 0);
        record(&mut acc, &by(3, 30), 4096);
        assert_eq!(acc.dead_bytes(), 4096, "the first writer's re-write");
        // A section tombstone ends one writer's generation: that writer's
        // records die, the other writer's stay live.
        let ends = SectionTombstonePayload {
            generation: Some(SectionGeneration { writer: 3, oseq: 1 }),
            reason: None,
        }
        .encode();
        acc.record(&header(RecordType::SectionTombstone, 5, 0), &ends, 64);
        assert_eq!(acc.dead_bytes(), 4096 * 2);
    }

    #[test]
    fn superseded_chunk_counts_as_dead() {
        let mut acc = RecordAccounting::new();
        record(&mut acc, &header(RecordType::Chunk, 5, 0), 4096);
        record(&mut acc, &header(RecordType::Chunk, 5, 1), 4096);
        assert_eq!(acc.dead_bytes(), 0, "distinct keys are all live");
        record(&mut acc, &header(RecordType::Chunk, 5, 0), 8192);
        assert_eq!(acc.dead_bytes(), 4096, "the first (5,0) write is dead");
        record(&mut acc, &header(RecordType::Chunk, 5, 0), 8192);
        assert_eq!(acc.dead_bytes(), 4096 + 8192);
    }

    #[test]
    fn per_stream_and_singleton_keys() {
        let mut acc = RecordAccounting::new();
        record(&mut acc, &header(RecordType::Tokens, 1, 0), 4096);
        record(&mut acc, &header(RecordType::Tokens, 2, 0), 4096);
        record(&mut acc, &header(RecordType::Tokens, 1, 0), 4096);
        assert_eq!(acc.dead_bytes(), 4096, "streams don't shadow each other");
        record(&mut acc, &header(RecordType::ModelKey, 0, 0), 4096);
        record(&mut acc, &header(RecordType::ModelKey, 0, 0), 4096);
        assert_eq!(acc.dead_bytes(), 8192, "superseded singleton is dead");
    }

    /// Commit records reuse `chunk_index` as `through_index` — they must
    /// key per stream, not per index, or every re-commit looks live.
    #[test]
    fn commits_key_per_stream_not_per_index() {
        let mut acc = RecordAccounting::new();
        record(&mut acc, &header(RecordType::Commit, 9, 3), 4096);
        record(&mut acc, &header(RecordType::Commit, 9, 7), 4096);
        assert_eq!(acc.dead_bytes(), 4096);
    }

    /// **A conversation's new state supersedes its last**, keyed by the
    /// timeline in the header, and never another conversation's.
    #[test]
    fn conv_state_supersedes_per_timeline() {
        let mut acc = RecordAccounting::new();
        record(&mut acc, &header(RecordType::ConvState, 11, 0), 4096);
        record(&mut acc, &header(RecordType::ConvState, 12, 0), 4096);
        assert_eq!(acc.dead_bytes(), 0, "two conversations, both live");
        record(&mut acc, &header(RecordType::ConvState, 11, 0), 4096);
        assert_eq!(acc.dead_bytes(), 4096, "11's first state is dead");
    }

    /// Payload-keyed metadata types and the derived `HeaderIndex`
    /// records are excluded — never counted dead.
    #[test]
    fn payload_keyed_types_are_skipped() {
        let mut acc = RecordAccounting::new();
        for _ in 0..3 {
            record(&mut acc, &header(RecordType::Label, 0, 0), 4096);
            record(&mut acc, &header(RecordType::TreeMetadata, 0, 0), 4096);
            record(&mut acc, &header(RecordType::HeaderIndex, 0, 0), 4096);
        }
        assert_eq!(acc.dead_bytes(), 0);
    }

    /// **A file event is live until something retires it** — its sequence
    /// number is unique, so no later record supersedes it — and retiring it
    /// twice counts it once.
    #[test]
    fn a_file_event_is_live_until_retired() {
        let mut acc = RecordAccounting::new();
        record(&mut acc, &header(RecordType::VfsEvent, 7, 1), 4096);
        record(&mut acc, &header(RecordType::VfsEvent, 7, 2), 8192);
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
        record(&mut acc, &header(RecordType::StreamDecl, 40, 0), 4096);
        record(&mut acc, &header(RecordType::Chunk, 40, 0), 8192);
        record(&mut acc, &header(RecordType::Chunk, 40, 1), 8192);
        record(&mut acc, &header(RecordType::Tokens, 40, 0), 4096);
        record(&mut acc, &header(RecordType::Chunk, 41, 0), 8192);
        record(&mut acc, &header(RecordType::Npc, 40, 0), 4096);
        assert_eq!(acc.dead_bytes(), 0);

        record(&mut acc, &header(RecordType::SectionTombstone, 40, 0), 4096);
        assert_eq!(
            acc.dead_bytes(),
            4096 + 8192 + 8192 + 4096,
            "stream 40's decl, chunks and tokens; not stream 41, not the Npc keyed 40"
        );

        record(&mut acc, &header(RecordType::StreamDecl, 40, 0), 4096);
        record(&mut acc, &header(RecordType::Chunk, 40, 0), 8192);
        assert_eq!(
            acc.dead_bytes(),
            4096 + 8192 + 8192 + 4096,
            "the new generation is live and supersedes nothing"
        );
    }

    #[test]
    fn reset_clears_everything() {
        let mut acc = RecordAccounting::new();
        record(&mut acc, &header(RecordType::Chunk, 1, 0), 4096);
        record(&mut acc, &header(RecordType::Chunk, 1, 0), 4096);
        assert_eq!(acc.dead_bytes(), 4096);
        acc.reset();
        assert_eq!(acc.dead_bytes(), 0);
        record(&mut acc, &header(RecordType::Chunk, 1, 0), 4096);
        assert_eq!(acc.dead_bytes(), 0, "post-reset state starts fresh");
    }
}
