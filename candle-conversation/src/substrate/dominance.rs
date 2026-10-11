//! Whether a tombstone here covers a record arriving from another member
//! (`docs/replicated_substrate.md` §6.5).
//!
//! A tombstone is a fact about what it names, not a value: every record for a
//! dead timeline, a retired turn or an ended section generation is dead at
//! read, whichever arrived first. A record arriving after the tombstone that
//! covers it is therefore not written at all — it would be appended dead and
//! dropped by the next rewrite — and is marked applied, so no catch-up asks for
//! it again.
//!
//! The answer is only ever *yes* when a tombstone is held here. A stream's own
//! records name their stream, not their timeline, so one arriving before its
//! turn's declaration cannot be judged and is written; the declaration's
//! arrival then makes it dead at read like any other.

use super::{StreamKey, Substrate, TurnMark};
use crate::persistence::identity::RecordIdentity;
use crate::persistence::manifest::{decode_conv_state_payload, decode_label_payload};
use crate::persistence::record::{
    DebugIdPayload, RecordHeader, RecordType, SectionGeneration, TreeMetadataPayload,
    TurnCouplingPayload,
};
use crate::persistence::streams::{StreamDecl, StreamId, TurnDecl};
use crate::projection::TimelineId;

impl Substrate {
    /// Whether `timeline` is dead: tombstoned and not distilled.
    pub fn is_timeline_dead(&self, timeline: u64) -> bool {
        TimelineId::from_raw(timeline).is_some_and(|tl| {
            self.tombstoned_timelines.contains(&tl) && !self.distilled_timelines.contains_key(&tl)
        })
    }

    /// Whether a tombstone held here covers the record `header` heads, whose
    /// payload is `payload` — see the module docs. A tombstone itself, a
    /// distillation, and records no timeline or section owns are never
    /// covered.
    pub fn dominates(&self, header: &RecordHeader, payload: &[u8]) -> bool {
        let id = header.identity();
        let stream = StreamId(header.stream_id);
        match header.record_type {
            RecordType::StreamDecl => match StreamDecl::decode(payload) {
                Ok(StreamDecl::Turn(decl)) => self.covers_turn(&decl, id.origin),
                Ok(StreamDecl::PromptSection(_)) => self.ended_sections.is_ended(
                    stream,
                    SectionGeneration {
                        writer: id.origin,
                        oseq: id.oseq,
                    },
                ),
                Err(_) => false,
            },
            RecordType::Chunk
            | RecordType::Tokens
            | RecordType::Commit
            | RecordType::WideQSig
            | RecordType::TurnIndexPage
            | RecordType::ProjectionEvents => self.covers_stream_record(stream, &id),
            RecordType::Label => {
                decode_label_payload(payload).is_ok_and(|(tl, _)| self.is_timeline_dead(tl))
            }
            RecordType::ConvState => {
                decode_conv_state_payload(payload).is_ok_and(|(tl, _)| self.is_timeline_dead(tl))
            }
            RecordType::TreeMetadata => TreeMetadataPayload::decode(payload)
                .is_ok_and(|p| self.is_timeline_dead(p.timeline_id)),
            RecordType::DebugId => {
                DebugIdPayload::decode(payload).is_ok_and(|p| self.is_timeline_dead(p.timeline_id))
            }
            RecordType::TurnCoupling => TurnCouplingPayload::decode(payload)
                .is_ok_and(|p| self.is_timeline_dead(p.timeline_id)),
            RecordType::Snapshot => self.retired_snapshots.contains(&stream),
            RecordType::VfsEvent | RecordType::VfsTombstone => {
                self.is_timeline_dead(header.stream_id)
            }
            RecordType::ModelKey
            | RecordType::Template
            | RecordType::Tokenizer
            | RecordType::HeaderIndex
            | RecordType::Tombstone
            | RecordType::SectionTombstone
            | RecordType::Distilled
            | RecordType::BranchCheckpoint
            | RecordType::Npc
            | RecordType::CustomObject
            | RecordType::Replica
            | RecordType::Rebase
            | RecordType::Deleted
            | RecordType::Unknown => false,
        }
    }

    /// Whether `decl`, as `writer` wrote it, names a dead timeline or a
    /// retired turn.
    fn covers_turn(&self, decl: &TurnDecl, writer: u64) -> bool {
        if self.is_timeline_dead(decl.timeline_id) {
            return true;
        }
        TimelineId::from_raw(decl.timeline_id).is_some_and(|timeline| {
            self.tombstoned_turns.contains(&TurnMark {
                timeline,
                turn_index: decl.turn_index,
                writer,
            })
        })
    }

    /// Whether a record of `stream` written as `id` is covered: its turn's
    /// declaration is held and covered, or it carries a section generation
    /// that has ended or that the declaration held has superseded.
    fn covers_stream_record(&self, stream: StreamId, id: &RecordIdentity) -> bool {
        let key = StreamKey::of(stream, id);
        if let Some(turn) = self.streams.turn(key) {
            return self.covers_turn(turn.decl, key.writer);
        }
        !self.admits(key, id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::content_hash::{hash_bytes, ContentHash};
    use crate::persistence::record::{
        DistillMode, DistillPayload, SectionTombstonePayload, TombstonePayload,
    };
    use crate::persistence::streams::{ContentAddress, SectionDecl};
    use crate::projection::{GroupId, LayerId};

    const B: u64 = 0xB;

    fn id(oseq: u64) -> RecordIdentity {
        RecordIdentity {
            origin: B,
            oseq,
            gen: 0,
            epoch: 0,
            hlc: oseq << 16,
        }
    }

    fn turn_decl(timeline: u64, turn_index: u32) -> TurnDecl {
        TurnDecl {
            timeline_id: timeline,
            turn_index,
            turn_id_day: 0,
            turn_id_seq: turn_index + 1,
            role: 1,
            block_start: 0,
            block_end: 1,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        }
    }

    fn header(rt: RecordType, stream: u64, id: RecordIdentity) -> RecordHeader {
        RecordHeader {
            record_type: rt,
            stream_id: stream,
            ..RecordHeader::NONE
        }
        .with_identity(id)
    }

    fn tombstone(sub: &mut Substrate, timeline: u64, turn_index: Option<u32>) {
        sub.apply_tombstone(&TombstonePayload {
            timeline_id: timeline,
            turn_index,
            turn_writer: B,
            reason: None,
        });
    }

    /// A dead timeline covers its declarations, its turns' records once their
    /// declaration is held, and its timeline-scoped records; a live one covers
    /// nothing.
    #[test]
    fn a_dead_timeline_covers_everything_it_names() {
        let mut sub = Substrate::new();
        let decl = StreamDecl::Turn(turn_decl(40, 0));
        let sid = decl.stream_id();
        let decl_header = header(RecordType::StreamDecl, sid.0, id(1));
        let tokens = header(RecordType::Tokens, sid.0, id(2));
        let coupling = TurnCouplingPayload {
            timeline_id: 40,
            from_turn: 0,
        }
        .encode();
        let coupling_header = header(RecordType::TurnCoupling, 0, id(3));

        assert!(!sub.dominates(&decl_header, &decl.encode()));
        assert!(!sub.dominates(&tokens, &[]));
        assert!(!sub.dominates(&coupling_header, &coupling));

        tombstone(&mut sub, 40, None);
        assert!(sub.dominates(&decl_header, &decl.encode()));
        assert!(sub.dominates(&coupling_header, &coupling));
        assert!(
            !sub.dominates(&tokens, &[]),
            "a stream record ahead of its declaration names no timeline"
        );
        sub.register_timeline(
            TimelineId::for_test(40),
            LayerId::from_raw(1).unwrap(),
            GroupId::from_raw(1).unwrap(),
        );
        sub.apply_stream_decl(sid, decl.clone(), id(1));
        assert!(sub.dominates(&tokens, &[]));
    }

    /// A distilled timeline is tombstoned but not dead: nothing is covered.
    #[test]
    fn a_distilled_timeline_covers_nothing() {
        let mut sub = Substrate::new();
        tombstone(&mut sub, 41, None);
        sub.apply_distill(&DistillPayload {
            timeline_id: 41,
            mode: DistillMode::ProvenanceOnly,
        });
        let decl = StreamDecl::Turn(turn_decl(41, 0));
        let h = header(RecordType::StreamDecl, decl.stream_id().0, id(1));
        assert!(!sub.dominates(&h, &decl.encode()));
    }

    /// A turn tombstone covers the turn as its writer wrote it, and no other
    /// writer's turn at that index.
    #[test]
    fn a_turn_tombstone_covers_its_writers_turn_only() {
        let mut sub = Substrate::new();
        tombstone(&mut sub, 42, Some(3));
        let decl = StreamDecl::Turn(turn_decl(42, 3));
        let sid = decl.stream_id().0;
        assert!(sub.dominates(&header(RecordType::StreamDecl, sid, id(1)), &decl.encode()));
        let other = RecordIdentity {
            origin: 0xC,
            ..id(1)
        };
        assert!(!sub.dominates(&header(RecordType::StreamDecl, sid, other), &decl.encode()));
    }

    /// An ended section generation covers its declaration and every record
    /// that carries it; the next generation stands.
    #[test]
    fn an_ended_generation_covers_its_records_only() {
        let mut sub = Substrate::new();
        let decl = StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress {
                prefix_hash: ContentHash { lo: 9, hi: 0 },
                section_hash: hash_bytes(b"system"),
            },
            debug_name: "system".to_string(),
        });
        let sid = decl.stream_id();
        sub.apply_section_tombstone(
            sid,
            &SectionTombstonePayload {
                generation: Some(SectionGeneration { writer: B, oseq: 5 }),
                reason: None,
            },
        );
        let ended_decl = header(RecordType::StreamDecl, sid.0, id(5));
        assert!(sub.dominates(&ended_decl, &decl.encode()));
        let ended_chunk = header(
            RecordType::Chunk,
            sid.0,
            RecordIdentity { epoch: 5, ..id(6) },
        );
        assert!(sub.dominates(&ended_chunk, &[]));
        let next_decl = header(RecordType::StreamDecl, sid.0, id(7));
        assert!(!sub.dominates(&next_decl, &decl.encode()));
        let next_chunk = header(
            RecordType::Chunk,
            sid.0,
            RecordIdentity { epoch: 7, ..id(8) },
        );
        assert!(!sub.dominates(&next_chunk, &[]));
    }
}
