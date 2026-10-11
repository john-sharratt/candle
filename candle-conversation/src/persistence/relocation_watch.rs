//! What a running maintenance op must no longer relocate, and what its
//! re-emission must write again after its planned copies.
//!
//! A maintenance op is planned under one hold of the persistence lock and then
//! re-emits its resident set and relocates its records in batches, releasing
//! the lock between them so seal writes and cold loads wait for one batch
//! rather than a whole segment. Every record the plan names is copied
//! *forward* into the active segment, so a record written in that window by
//! anyone else lands BEFORE the planned copy that follows it — and the log walk
//! on the next load lets the later record win. Three kinds of write make a
//! planned copy wrong:
//!
//! - **A newer copy of the same record.** A `Chunk` re-sealed (a partial tail
//!   sealed final) or a stream's `Tokens` rewritten: the planned copy is
//!   superseded, and relocating it would roll the record back, so it is
//!   skipped. A resident metadata record — a decl, a signature window, a commit
//!   mark, a label, a debug id, a tree node — written again is different: its
//!   planned copy is still appended, and the newer writes are appended again
//!   after it. Skipping would be wrong for two reasons. A label merges on
//!   replay (a record may carry only a conversation id, only a title or only
//!   custom fields), so its planned copy holds fields the newer one does not.
//!   And replay applies a timeline's metadata only once its turn declaration
//!   has registered the timeline: a write made mid-op sits *before* the
//!   re-emitted declaration whenever the old one was in a target segment, so
//!   a newer copy left where it was is dropped on the next load.
//! - **A section retired.** A `SectionTombstone` ends a section's generation,
//!   and a later declaration of the same content address revives it as a new
//!   one. The old generation's chunks, tokens and metadata carried after the
//!   tombstone would load as part of whatever comes after it — the rebuild
//!   never happens, or the rebuilt section carries leftovers of the retired one.
//! - **A section revived.** The converse: a planned tombstone re-emitted after
//!   the declaration that started the next generation would retire it again.
//!
//! The watch records these from the moment the op is planned. It is also the
//! op's presence: a handle runs one maintenance op at a time, and a second
//! plan while one is in flight would relocate out of segments the first is
//! about to unlink.
//!
//! A stream's own records are watched per writer: two members' turn *N* share
//! a stream id and are two streams (`docs/replicated_substrate.md` §6.6), so a
//! chunk one of them re-seals supersedes nothing of the other's.

use std::collections::{HashMap, HashSet};

use super::identity::RecordIdentity;
use super::manifest::decode_label_payload;
use super::record::{
    DebugIdPayload, DistillPayload, RecordHeader, RecordType, SectionTombstonePayload,
    TreeMetadataPayload,
};

/// The identity under which one resident record replaces (or, for a label,
/// merges into) another of the same type on replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Key {
    /// Keyed by the header's `stream_id` and the writer: one record of this
    /// type per writer's stream (or, for `ConvState` and `CustomObject`, per
    /// timeline / object key, which they carry in the same field, with writer
    /// zero).
    Header(RecordType, u64, u64),
    /// Keyed by the timeline named in the payload.
    Timeline(RecordType, u64),
    /// A summary-forest node, keyed by `(timeline, turn)`.
    TreeNode(u64, u32),
}

/// The key a resident record of type `rt` written by `writer` replays under,
/// or `None` for a record no later write replaces (a timeline or turn
/// tombstone, a coupling — each only ever adds) or that is not resident at all.
fn key(rt: RecordType, stream_id: u64, writer: u64, payload: &[u8]) -> Option<Key> {
    match rt {
        RecordType::StreamDecl
        | RecordType::ProjectionEvents
        | RecordType::WideQSig
        | RecordType::TurnIndexPage
        | RecordType::Commit => Some(Key::Header(rt, stream_id, writer)),
        RecordType::ConvState | RecordType::CustomObject => Some(Key::Header(rt, stream_id, 0)),
        RecordType::Label => decode_label_payload(payload)
            .ok()
            .map(|(timeline, _)| Key::Timeline(rt, timeline)),
        RecordType::DebugId => DebugIdPayload::decode(payload)
            .ok()
            .map(|p| Key::Timeline(rt, p.timeline_id)),
        RecordType::Distilled => DistillPayload::decode(payload)
            .ok()
            .map(|p| Key::Timeline(rt, p.timeline_id)),
        RecordType::TreeMetadata => TreeMetadataPayload::decode(payload)
            .ok()
            .map(|p| Key::TreeNode(p.timeline_id, p.turn_index)),
        _ => None,
    }
}

/// Whether `rt` is keyed by the stream in its header — the records a section's
/// retirement ends along with its chunks.
fn stream_keyed(rt: RecordType) -> bool {
    matches!(
        rt,
        RecordType::StreamDecl
            | RecordType::ProjectionEvents
            | RecordType::WideQSig
            | RecordType::TurnIndexPage
            | RecordType::Commit
            | RecordType::SectionTombstone
    )
}

/// One record written since the plan, as the re-emission appends it again:
/// the header's `chunk_index` (a `Commit` carries its mark there), the
/// payload, and the identity it was written with, which its copy keeps.
pub type Rewrite = (u64, Vec<u8>, RecordIdentity);

/// What a planned resident record's re-emission should do.
#[derive(Debug, PartialEq, Eq)]
pub enum Reemit {
    /// Append the planned copy.
    Carry,
    /// Append nothing: the section's retirement, or its revival, stands after
    /// the plan already.
    Skip,
    /// Append the planned copy, then these writes made since the plan, in the
    /// order they were made — so the newest is last, and after the
    /// re-emitted declaration that registers its timeline on replay.
    CarryThen(Vec<Rewrite>),
}

/// The writes made since a maintenance op was planned that its relocation and
/// re-emission must step around.
#[derive(Debug, Default)]
pub struct RelocationWatch {
    /// `(stream_id, chunk_index, writer)` of every `Chunk` appended.
    chunks: HashSet<(u64, u64, u64)>,
    /// `(stream_id, writer)` of every `Tokens` record appended.
    tokens: HashSet<(u64, u64)>,
    /// `(stream_id, writer)` of every section stream a `SectionTombstone`
    /// retired — the writer of the generation it ended, zero for one written
    /// before generations.
    retired: HashSet<(u64, u64)>,
    /// Every stream a `StreamDecl` was appended for.
    declared: HashSet<u64>,
    /// Every resident record written again, by key, in write order.
    rewrites: HashMap<Key, Vec<Rewrite>>,
}

impl RelocationWatch {
    /// Note one appended record.
    pub fn record(&mut self, header: &RecordHeader, payload: &[u8]) {
        let rt = header.record_type;
        match rt {
            RecordType::Chunk => {
                self.chunks
                    .insert((header.stream_id, header.chunk_index, header.origin));
            }
            RecordType::Tokens => {
                self.tokens.insert((header.stream_id, header.origin));
            }
            RecordType::SectionTombstone => {
                let writer = SectionTombstonePayload::decode(payload)
                    .ok()
                    .and_then(|p| p.generation)
                    .map_or(0, |g| g.writer);
                self.retired.insert((header.stream_id, writer));
            }
            RecordType::StreamDecl => {
                self.declared.insert(header.stream_id);
            }
            _ => {}
        }
        if let Some(k) = key(rt, header.stream_id, header.origin, payload) {
            self.rewrites.entry(k).or_default().push((
                header.chunk_index,
                payload.to_vec(),
                header.identity(),
            ));
        }
    }

    /// Whether the planned copy of chunk `index` of `writer`'s `stream` is
    /// still the one to carry forward.
    pub fn keeps_chunk(&self, stream: u64, index: u64, writer: u64) -> bool {
        !self.chunks.contains(&(stream, index, writer)) && !self.retired.contains(&(stream, writer))
    }

    /// Whether the planned copy of `writer`'s `stream`'s `Tokens` is still the
    /// one to carry forward.
    pub fn keeps_tokens(&self, stream: u64, writer: u64) -> bool {
        !self.tokens.contains(&(stream, writer)) && !self.retired.contains(&(stream, writer))
    }

    /// What to do with a planned resident record of type `rt`, header stream
    /// `stream_id`, written by `writer`, carrying `payload`.
    ///
    /// A planned tombstone that names a generation is always carried: it ends
    /// only that generation, so a section declared again since is untouched
    /// by it. One written before generations ends whatever unstamped stream
    /// stands at its place in the log, so it is not carried past a declaration
    /// made since.
    pub fn reemit(&self, rt: RecordType, stream_id: u64, writer: u64, payload: &[u8]) -> Reemit {
        if stream_keyed(rt) && self.retired.contains(&(stream_id, writer)) {
            return Reemit::Skip;
        }
        if rt == RecordType::SectionTombstone
            && self.declared.contains(&stream_id)
            && SectionTombstonePayload::decode(payload)
                .ok()
                .is_some_and(|p| p.generation.is_none())
        {
            return Reemit::Skip;
        }
        // The planned copy is only decoded when something could have been
        // written over it — the common op runs with nothing written beside it.
        if self.rewrites.is_empty() {
            return Reemit::Carry;
        }
        match key(rt, stream_id, writer, payload).and_then(|k| self.rewrites.get(&k)) {
            Some(newer) => Reemit::CarryThen(newer.clone()),
            None => Reemit::Carry,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::persistence::manifest::encode_label_payload;
    use crate::persistence::record::{SectionGeneration, TombstonePayload};

    /// A header stamped with identity `oseq` of origin 9, so a test can tell
    /// which write a carried copy took its identity from.
    fn header(record_type: RecordType, stream_id: u64, chunk_index: u64) -> RecordHeader {
        RecordHeader {
            record_type,
            format: 0,
            payload_len: 0,
            crc: 0,
            stream_id,
            chunk_index,
            token_count: 0,
            ..RecordHeader::NONE
        }
    }

    fn stamped(header: RecordHeader, oseq: u64) -> RecordHeader {
        header.with_identity(id(oseq))
    }

    fn id(oseq: u64) -> RecordIdentity {
        RecordIdentity {
            origin: 9,
            oseq,
            gen: 0,
            epoch: 0,
            hlc: oseq << 16,
        }
    }

    fn debug_id(timeline_id: u64, id: &str) -> Vec<u8> {
        DebugIdPayload {
            timeline_id,
            debug_id: id.to_string(),
        }
        .encode()
    }

    fn tree_node(timeline_id: u64, turn_index: u32, tree_height: u8) -> Vec<u8> {
        TreeMetadataPayload {
            timeline_id,
            turn_index,
            kind: 0,
            tree_height,
            children: Vec::new(),
        }
        .encode()
    }

    fn label(timeline: u64, conv_id: &str, title: &str) -> Vec<u8> {
        encode_label_payload(timeline, conv_id, title, &BTreeMap::new())
    }

    #[test]
    fn an_unwatched_record_is_carried() {
        let w = RelocationWatch::default();
        assert!(w.keeps_chunk(7, 0, 0));
        assert!(w.keeps_tokens(7, 0));
        assert_eq!(w.reemit(RecordType::StreamDecl, 7, 0, &[]), Reemit::Carry);
        assert_eq!(
            w.reemit(RecordType::DebugId, 0, 0, &debug_id(3, "a")),
            Reemit::Carry
        );
    }

    /// A re-appended chunk supersedes only its own key.
    #[test]
    fn a_rewritten_chunk_drops_only_that_chunk() {
        let mut w = RelocationWatch::default();
        w.record(&header(RecordType::Chunk, 7, 2), &[]);
        assert!(!w.keeps_chunk(7, 2, 0));
        assert!(w.keeps_chunk(7, 1, 0));
        assert!(w.keeps_chunk(8, 2, 0));
        assert!(
            w.keeps_tokens(7, 0),
            "a chunk write leaves the stream's tokens alone"
        );
    }

    /// Two writers' records of one stream id are two streams': one writer's
    /// re-sealed chunk and tokens leave the other's planned copies to carry,
    /// and its re-written decl is not appended after the other's.
    #[test]
    fn one_writers_rewrite_leaves_the_others_stream_alone() {
        let mut w = RelocationWatch::default();
        w.record(&stamped(header(RecordType::Chunk, 7, 2), 1), &[]);
        w.record(&stamped(header(RecordType::Tokens, 7, 0), 2), &[]);
        w.record(&stamped(header(RecordType::StreamDecl, 7, 0), 3), b"decl");
        assert!(!w.keeps_chunk(7, 2, 9));
        assert!(w.keeps_chunk(7, 2, 4), "the other writer's chunk 2");
        assert!(!w.keeps_tokens(7, 9));
        assert!(w.keeps_tokens(7, 4));
        assert_eq!(
            w.reemit(RecordType::StreamDecl, 7, 4, b"theirs"),
            Reemit::Carry
        );
        assert_eq!(
            w.reemit(RecordType::StreamDecl, 7, 9, b"older"),
            Reemit::CarryThen(vec![(0, b"decl".to_vec(), id(3))])
        );
    }

    /// Tokens are one record per stream, keyed by the stream alone.
    #[test]
    fn rewritten_tokens_drop_that_streams_tokens() {
        let mut w = RelocationWatch::default();
        w.record(&header(RecordType::Tokens, 7, 0), &[]);
        assert!(!w.keeps_tokens(7, 0));
        assert!(w.keeps_tokens(8, 0));
        assert!(
            w.keeps_chunk(7, 0, 0),
            "a tokens write leaves the stream's chunks alone"
        );
    }

    fn section_tombstone(generation: Option<SectionGeneration>) -> Vec<u8> {
        SectionTombstonePayload {
            generation,
            reason: None,
        }
        .encode()
    }

    /// A retired section carries none of its old generation past the
    /// tombstone — its writer's chunks, tokens, and stream-keyed metadata —
    /// and leaves another writer's stream of the same section to carry.
    #[test]
    fn a_retired_section_drops_its_whole_old_generation() {
        let mut w = RelocationWatch::default();
        let ends = section_tombstone(Some(SectionGeneration { writer: 4, oseq: 3 }));
        w.record(&header(RecordType::SectionTombstone, 40, 0), &ends);
        assert!(!w.keeps_chunk(40, 0, 4));
        assert!(!w.keeps_chunk(40, 9, 4));
        assert!(!w.keeps_tokens(40, 4));
        assert_eq!(w.reemit(RecordType::StreamDecl, 40, 4, &[]), Reemit::Skip);
        assert_eq!(w.reemit(RecordType::WideQSig, 40, 4, &[]), Reemit::Skip);
        assert!(w.keeps_chunk(40, 0, 6), "another writer's stream");
        assert!(w.keeps_tokens(40, 6));
        assert!(w.keeps_chunk(41, 0, 4));
        assert!(w.keeps_tokens(41, 4));
        assert_eq!(w.reemit(RecordType::StreamDecl, 41, 4, &[]), Reemit::Carry);
        // A tombstone written before generations retires the unstamped stream.
        w.record(
            &header(RecordType::SectionTombstone, 42, 0),
            &section_tombstone(None),
        );
        assert!(!w.keeps_chunk(42, 0, 0));
        assert!(w.keeps_chunk(42, 0, 4));
    }

    /// A section declared again since the plan has started its next
    /// generation: a planned tombstone written before generations must not
    /// retire it a second time, and one that names the generation it ended
    /// cannot, so it is carried.
    #[test]
    fn a_revived_section_keeps_no_planned_tombstone() {
        let mut w = RelocationWatch::default();
        w.record(&header(RecordType::StreamDecl, 40, 0), b"decl");
        assert_eq!(
            w.reemit(
                RecordType::SectionTombstone,
                40,
                0,
                &section_tombstone(None)
            ),
            Reemit::Skip
        );
        let named = section_tombstone(Some(SectionGeneration { writer: 4, oseq: 3 }));
        assert_eq!(
            w.reemit(RecordType::SectionTombstone, 40, 0, &named),
            Reemit::Carry
        );
        assert_eq!(
            w.reemit(
                RecordType::SectionTombstone,
                41,
                0,
                &section_tombstone(None)
            ),
            Reemit::Carry
        );
    }

    /// A rewritten record follows its planned copy under its own key only —
    /// the same type for another stream, timeline or turn is carried alone —
    /// and a commit mark carries its header value with it.
    #[test]
    fn a_rewritten_record_follows_only_its_own_key() {
        let mut w = RelocationWatch::default();
        w.record(&stamped(header(RecordType::Commit, 7, 3), 11), &[]);
        w.record(&header(RecordType::DebugId, 0, 0), &debug_id(5, "new"));
        w.record(&header(RecordType::TreeMetadata, 0, 0), &tree_node(5, 2, 1));

        assert_eq!(
            w.reemit(RecordType::Commit, 7, 9, &[]),
            Reemit::CarryThen(vec![(3, Vec::new(), id(11))]),
            "the newer write is carried with the identity it was written under"
        );
        assert_eq!(w.reemit(RecordType::Commit, 8, 9, &[]), Reemit::Carry);
        assert_eq!(
            w.reemit(RecordType::WideQSig, 7, 9, &[]),
            Reemit::Carry,
            "a commit mark is followed by no other record of its stream"
        );
        assert_eq!(
            w.reemit(RecordType::DebugId, 0, 0, &debug_id(5, "old")),
            Reemit::CarryThen(vec![(0, debug_id(5, "new"), RecordIdentity::NONE)])
        );
        assert_eq!(
            w.reemit(RecordType::DebugId, 0, 0, &debug_id(6, "old")),
            Reemit::Carry
        );
        assert_eq!(
            w.reemit(RecordType::TreeMetadata, 0, 0, &tree_node(5, 2, 0)),
            Reemit::CarryThen(vec![(0, tree_node(5, 2, 1), RecordIdentity::NONE)])
        );
        assert_eq!(
            w.reemit(RecordType::TreeMetadata, 0, 0, &tree_node(5, 3, 0)),
            Reemit::Carry
        );
    }

    /// A label merges on replay, so its planned copy is carried and every
    /// newer partial label follows it, in the order they were written.
    #[test]
    fn a_relabelled_timeline_carries_its_label_then_the_newer_ones() {
        let mut w = RelocationWatch::default();
        let first = label(5, "", "renamed");
        let second = label(5, "", "renamed again");
        w.record(&stamped(header(RecordType::Label, 0, 0), 1), &first);
        w.record(&stamped(header(RecordType::Label, 0, 0), 2), &second);
        assert_eq!(
            w.reemit(RecordType::Label, 0, 0, &label(5, "conv-5", "original")),
            Reemit::CarryThen(vec![(0, first, id(1)), (0, second, id(2))])
        );
        assert_eq!(
            w.reemit(RecordType::Label, 0, 0, &label(6, "conv-6", "other")),
            Reemit::Carry
        );
    }

    /// Records that only ever add are carried alone whatever was written
    /// beside them.
    #[test]
    fn additive_records_are_carried_alone() {
        let mut w = RelocationWatch::default();
        let tomb = TombstonePayload {
            timeline_id: 5,
            turn_index: None,
            turn_writer: 0,
            reason: None,
        }
        .encode();
        w.record(&header(RecordType::Tombstone, 0, 0), &tomb);
        w.record(&header(RecordType::Snapshot, 7, 0), &[]);
        assert_eq!(w.reemit(RecordType::Tombstone, 0, 0, &tomb), Reemit::Carry);
        assert!(w.keeps_chunk(7, 0, 0));
        assert!(w.keeps_tokens(7, 0));
    }
}
