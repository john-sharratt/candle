//! Whether a keyed value here outranks a record arriving for it
//! (`docs/replicated_substrate.md` §6.5).
//!
//! A keyed value holds the record with the greatest stamp, so a record ranked
//! below the one held loses on arrival, whenever it comes: written, it would
//! sit in the log dead, and the next rewrite would drop it. It is skipped and
//! marked applied instead.
//!
//! The question is about stamps, never content. Two records carrying the same
//! bytes under different stamps are still two values: every member must hold
//! the greater stamp, because a third record ranked between them wins against
//! one and loses against the other, and members that kept different stamps for
//! one value would diverge on it.
//!
//! Only values one record sets are answered here. A `Label` merges field by
//! field and a `Commit` takes the greatest mark, so neither has one record to
//! lose to, and a set insert never loses at all.

use super::{StreamKey, Substrate, TimelineIds};
use crate::persistence::identity::RecordIdentity;
use crate::persistence::manifest::decode_conv_state_payload;
use crate::persistence::record::{DebugIdPayload, RecordHeader, RecordType, TreeMetadataPayload};
use crate::persistence::streams::StreamId;
use crate::projection::{TimelineId, TurnIndex};

impl Substrate {
    /// Whether the value `header`'s record sets is held here under a greater
    /// stamp — see the module docs.
    pub fn outranks(&self, header: &RecordHeader, payload: &[u8]) -> bool {
        let id = header.identity();
        self.held_identity(header, payload, &id)
            .is_some_and(|held| !id.supersedes(&held))
    }

    /// The identity of the record behind the value `header`'s record sets, or
    /// `None` when no single record holds it.
    fn held_identity(
        &self,
        header: &RecordHeader,
        payload: &[u8],
        id: &RecordIdentity,
    ) -> Option<RecordIdentity> {
        let stream = StreamId(header.stream_id);
        let physical = || self.streams.physical(StreamKey::of(stream, id));
        match header.record_type {
            RecordType::StreamDecl => physical().map(|rt| rt.ids.decl),
            RecordType::Tokens => physical().map(|rt| rt.ids.tokens),
            RecordType::ProjectionEvents => physical().map(|rt| rt.ids.projection_events),
            RecordType::WideQSig => physical().map(|rt| rt.ids.wide_q_sigs),
            RecordType::TurnIndexPage => physical().map(|rt| rt.ids.index_page),
            RecordType::Chunk => {
                physical().and_then(|rt| rt.ids.chunks.get(&header.chunk_index).copied())
            }
            RecordType::Snapshot => self.snapshot_ids.get(&stream).copied(),
            RecordType::CustomObject => self.custom_object_ids.get(&header.stream_id).copied(),
            RecordType::ConvState => {
                let (timeline, _) = decode_conv_state_payload(payload).ok()?;
                self.record_ids_of(timeline).map(|ids| ids.state)
            }
            RecordType::DebugId => {
                let p = DebugIdPayload::decode(payload).ok()?;
                self.record_ids_of(p.timeline_id).map(|ids| ids.debug_id)
            }
            RecordType::TreeMetadata => {
                let p = TreeMetadataPayload::decode(payload).ok()?;
                self.record_ids_of(p.timeline_id)
                    .and_then(|ids| ids.tree_meta.get(&TurnIndex(p.turn_index)).copied())
            }
            RecordType::Label
            | RecordType::Commit
            | RecordType::TurnCoupling
            | RecordType::Tombstone
            | RecordType::SectionTombstone
            | RecordType::Distilled
            | RecordType::VfsEvent
            | RecordType::VfsTombstone
            | RecordType::Npc
            | RecordType::BranchCheckpoint
            | RecordType::ModelKey
            | RecordType::Template
            | RecordType::Tokenizer
            | RecordType::HeaderIndex
            | RecordType::Replica
            | RecordType::Rebase
            | RecordType::Deleted
            | RecordType::Unknown => None,
        }
    }

    /// The record identities behind registered timeline `timeline`'s values.
    fn record_ids_of(&self, timeline: u64) -> Option<&TimelineIds> {
        let timeline = TimelineId::from_raw(timeline)?;
        self.timelines.get(&timeline).map(|tl| &tl.ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::manifest::ChunkLoc;
    use crate::persistence::segment::SegmentId;

    fn id(origin: u64, hlc: u64) -> RecordIdentity {
        RecordIdentity {
            origin,
            oseq: hlc,
            gen: 0,
            epoch: 0,
            hlc,
        }
    }

    fn chunk_header(chunk_index: u64, id: RecordIdentity) -> RecordHeader {
        RecordHeader {
            record_type: RecordType::Chunk,
            stream_id: 77,
            chunk_index,
            ..RecordHeader::NONE
        }
        .with_identity(id)
    }

    fn loc() -> ChunkLoc {
        ChunkLoc {
            segment: SegmentId(1),
            offset: 0,
            payload_len: 1,
            record_size: 4096,
            token_count: 32,
            format: 4,
        }
    }

    /// A chunk ranked below the one held is outranked; a greater one, another
    /// index and another writer's stream are not.
    #[test]
    fn a_lesser_chunk_is_outranked() {
        let mut sub = Substrate::new();
        sub.apply_chunk_loc(StreamId(77), 0, loc(), id(0xB, 20));
        assert!(sub.outranks(&chunk_header(0, id(0xB, 10)), &[]));
        assert!(!sub.outranks(&chunk_header(0, id(0xB, 30)), &[]));
        assert!(!sub.outranks(&chunk_header(1, id(0xB, 10)), &[]));
        assert!(!sub.outranks(&chunk_header(0, id(0xC, 10)), &[]));
    }

    /// A value nothing holds yet outranks nothing.
    #[test]
    fn an_empty_key_outranks_nothing() {
        let sub = Substrate::new();
        assert!(!sub.outranks(&chunk_header(0, id(0xB, 1)), &[]));
    }
}
