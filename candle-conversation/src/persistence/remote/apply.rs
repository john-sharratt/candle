//! Applying a batch of records other members wrote
//! (`docs/replicated_substrate.md` §6.5).

use std::collections::BTreeSet;

use super::super::identity::{RecordIdentity, GEN_DETACHED};
use super::super::record::{Record, RecordHeader, RecordType, SectionTombstonePayload};
use super::super::replica::DeferredRecord;
use super::super::streams::{StreamDecl, StreamId};
use super::super::walker::WalkEntry;
use super::super::{record_snapshot_loc, vfs, Result, SubstratePersistence};
use super::adopt::adopt_complete_turns;
use super::incoming::Incoming;
use super::outcome::{Refusal, RemoteBatch, RemoteOutcome};
use crate::substrate::{StreamKey, Substrate};

/// Whether records of type `rt` never leave the member that wrote them
/// (§6.1): the header index, the model binding, the projection schema, the
/// branch-checkpoint cache, and this member's own housekeeping.
fn is_local_only(rt: RecordType) -> bool {
    matches!(
        rt,
        RecordType::HeaderIndex
            | RecordType::ModelKey
            | RecordType::Tokenizer
            | RecordType::Template
            | RecordType::BranchCheckpoint
            | RecordType::Replica
            | RecordType::Rebase
            | RecordType::Deleted
    )
}

/// Whether the record declares a turn. A turn's declaration is written even
/// when a tombstone here covers it, dead at read like the rest of its
/// timeline: it is the one record of the turn that names the timeline, so the
/// stream's own records arriving after it are covered too. Skipped, it would
/// leave them to arrive as records of no timeline, kept for a declaration that
/// never comes.
fn declares_turn(h: &RecordHeader, payload: &[u8]) -> bool {
    h.record_type == RecordType::StreamDecl
        && matches!(StreamDecl::decode(payload), Ok(StreamDecl::Turn(_)))
}

/// Whether a record of type `rt` is one of a turn's own: the pieces whose
/// arrival can complete it.
fn completes_turns(rt: RecordType) -> bool {
    matches!(
        rt,
        RecordType::StreamDecl | RecordType::Tokens | RecordType::Chunk
    )
}

impl SubstratePersistence {
    /// Apply `records`, each the bytes of one record another member wrote, to
    /// this store and to `substrate` — under the substrate's write lock, which
    /// the caller holds for the whole batch, so a projection sees the batch
    /// whole or not at all.
    ///
    /// Each record is decided in turn: a type this build does not decode is
    /// deferred; one applied already is a duplicate; one a tombstone here
    /// covers is dominated, and one whose value is held under a greater stamp
    /// is superseded — both marked applied and not written, but for a turn's
    /// declaration, which is written dead (see [`declares_turn`]); anything
    /// else is
    /// verified and appended under the identity it carries, then applied as
    /// the walk applies it at the next open. Then every turn the batch
    /// completed is adopted, unless the batch moved a turn this member holds,
    /// which needs a reload instead. `n_layers` is the model's K/V layer count,
    /// which says when a turn's chunks are all here.
    ///
    /// A refused record leaves the batch going; an error writing one stops
    /// it.
    pub fn apply_remote(
        &mut self,
        substrate: &mut Substrate,
        records: &[&[u8]],
        n_layers: usize,
    ) -> Result<RemoteBatch> {
        let mut outcomes = Vec::with_capacity(records.len());
        let mut turns = BTreeSet::new();
        for bytes in records {
            outcomes.push(self.apply_one(substrate, bytes, &mut turns)?);
        }
        let reload = substrate.take_reload_due();
        let adopted = if reload {
            Vec::new()
        } else {
            adopt_complete_turns(self, substrate, &turns, n_layers)?
        };
        Ok(RemoteBatch {
            outcomes,
            adopted,
            reload,
        })
    }

    fn apply_one(
        &mut self,
        substrate: &mut Substrate,
        bytes: &[u8],
        turns: &mut BTreeSet<StreamKey>,
    ) -> Result<RemoteOutcome> {
        let incoming = match Incoming::parse(bytes) {
            Ok(incoming) => incoming,
            Err(e) => return Ok(RemoteOutcome::Refused(Refusal::Malformed(e.to_string()))),
        };
        let h = incoming.header;
        let id = h.identity();
        if id.is_none() {
            return Ok(RemoteOutcome::Refused(Refusal::Unidentified));
        }
        if id.gen == GEN_DETACHED {
            return Ok(RemoteOutcome::Refused(Refusal::Detached));
        }
        if h.record_type == RecordType::Unknown {
            self.defer(DeferredRecord {
                origin: id.origin,
                oseq: id.oseq,
                record_type: incoming.type_name,
            });
            return Ok(RemoteOutcome::Deferred);
        }
        if is_local_only(h.record_type) {
            return Ok(RemoteOutcome::Refused(Refusal::LocalOnly(h.record_type)));
        }
        if self.is_applied(id.origin, id.oseq) {
            return Ok(RemoteOutcome::Duplicate);
        }
        if id.origin == self.node() {
            return Ok(RemoteOutcome::Refused(Refusal::OwnOrigin));
        }
        self.minter.observe(&id);
        if substrate.dominates(&h, &incoming.payload) && !declares_turn(&h, &incoming.payload) {
            self.mark_applied(&id);
            return Ok(RemoteOutcome::Dominated);
        }
        if substrate.outranks(&h, &incoming.payload) {
            self.mark_applied(&id);
            return Ok(RemoteOutcome::Superseded);
        }
        if let Err(refusal) = incoming.verify() {
            return Ok(RemoteOutcome::Refused(refusal));
        }
        let at = self.append_carried(
            h.record_type,
            h.format,
            h.stream_id,
            h.chunk_index,
            h.token_count,
            h.crc,
            &incoming.payload,
            id,
        )?;
        let entry = WalkEntry {
            segment: at.segment,
            offset: at.offset,
            record: Record {
                header: h,
                payload: incoming.payload,
            },
            size: at.size,
        };
        self.index_remote(&entry);
        substrate.apply_walker_entry(&entry);
        if completes_turns(h.record_type) {
            turns.insert(StreamKey::of(StreamId(h.stream_id), &id));
        }
        Ok(RemoteOutcome::Written(at))
    }

    /// What the walk at open keeps beside the substrate for a record, and an
    /// append of this member's own does not: a timeline tombstone retires the
    /// conversation's snapshot tail and file events, a file event joins its
    /// timeline's index, and a section tombstone naming a generation this
    /// member is writing ends it here too.
    fn index_remote(&mut self, entry: &WalkEntry) {
        record_snapshot_loc(&mut self.snapshot_locs, entry);
        vfs::record_vfs_loc(&mut self.vfs_index, &mut self.accounting, entry);
        let h = &entry.record.header;
        if h.record_type == RecordType::SectionTombstone {
            if let Some(g) = SectionTombstonePayload::decode(&entry.record.payload)
                .ok()
                .and_then(|p| p.generation)
            {
                if g.writer == self.node() {
                    self.minter.end_section(h.stream_id, g.oseq);
                }
            }
        }
    }

    /// Record `id` as applied without writing it.
    fn mark_applied(&mut self, id: &RecordIdentity) {
        if self.replica.applied.insert(id.origin, id.oseq) {
            self.replica_dirty = true;
        }
    }
}
