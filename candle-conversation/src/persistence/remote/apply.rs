//! Applying a batch of records other members wrote
//! (`docs/replicated_substrate.md` §6.5).

use std::collections::BTreeSet;

use super::super::identity::{RecordIdentity, GEN_DETACHED};
use super::super::record::{Record, RecordHeader, RecordType, SectionTombstonePayload};
use super::super::replica::DeferredRecord;
use super::super::streams::{StreamDecl, StreamId};
use super::super::walker::WalkEntry;
use super::super::{blob, record_snapshot_loc, vfs, Result, SubstratePersistence};
use super::adopt::adopt_complete_turns;
use super::incoming::Incoming;
use super::outcome::{Checked, Refusal, RemoteBatch, RemoteOutcome};
use crate::substrate::{StreamKey, Substrate};

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
    ///
    /// The checks of [`Self::check_remote`] run first; a caller holding locks
    /// readers wait on runs them before it takes them and hands them to
    /// [`Self::apply_remote_checked`].
    pub fn apply_remote(
        &mut self,
        substrate: &mut Substrate,
        records: &[&[u8]],
        n_layers: usize,
    ) -> Result<RemoteBatch> {
        let checks = Self::check_remote(records);
        self.apply_remote_checked(substrate, records, &checks, n_layers)
    }

    /// Whether each of `records` is what its writer wrote — its digest, and
    /// its CRC for any type but a chunk ([`Incoming::verify`]). Reads every
    /// payload byte and touches no state, so it needs no lock: verifying
    /// under the substrate's write lock held every reader of the substrate
    /// behind a full pass over every byte the batch carried.
    pub fn check_remote(records: &[&[u8]]) -> Vec<Checked> {
        records
            .iter()
            .map(|bytes| match Incoming::parse(bytes) {
                Ok(incoming) => incoming.verify(),
                Err(e) => Err(Refusal::Malformed(e.to_string())),
            })
            .collect()
    }

    /// [`Self::apply_remote`] with each record's [`Self::check_remote`]
    /// result already in hand, one per record and in order. A record that
    /// failed its check is refused where the check stands in the decision —
    /// after the duplicate, dominated and superseded cases, which hold
    /// whatever its bytes — so the outcomes are what checking in place gave.
    pub fn apply_remote_checked(
        &mut self,
        substrate: &mut Substrate,
        records: &[&[u8]],
        checks: &[Checked],
        n_layers: usize,
    ) -> Result<RemoteBatch> {
        debug_assert_eq!(records.len(), checks.len());
        let mut outcomes = Vec::with_capacity(records.len());
        let mut turns = BTreeSet::new();
        for (bytes, checked) in records.iter().zip(checks) {
            outcomes.push(self.apply_one(substrate, bytes, checked, &mut turns)?);
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
        checked: &Checked,
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
        if h.record_type.is_local_only() {
            return Ok(RemoteOutcome::Refused(Refusal::LocalOnly(h.record_type)));
        }
        if self.is_applied(id.origin, id.oseq) {
            return Ok(RemoteOutcome::Duplicate);
        }
        // This node's own record, not held: a gap a purge left, which another
        // member's copy fills (§7.8) — or an `oseq` it never minted.
        if id.origin == self.node() && id.oseq >= self.minter.origination().next_oseq() {
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
        if let Err(refusal) = checked {
            return Ok(RemoteOutcome::Refused(refusal.clone()));
        }
        // One of this node's own is taken into its written set before the
        // append, so the append's tap does not send it out again: the member
        // it came from holds it. A failed append leaves the gap open on disk —
        // the replica record that would close it is written by a commit.
        if id.origin == self.node() {
            self.mark_applied(&id);
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
    /// timeline's index, a blob joins the blobs, and a section tombstone naming
    /// a generation this member is writing ends it here too.
    fn index_remote(&mut self, entry: &WalkEntry) {
        record_snapshot_loc(&mut self.snapshot_locs, entry);
        vfs::record_vfs_loc(&mut self.vfs_index, &mut self.accounting, entry);
        blob::record_blob_loc(&mut self.blob_index, entry);
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

    /// Record `id` as applied — into the replica, or for one of this node's
    /// own a purge dropped, into its written set.
    fn mark_applied(&mut self, id: &RecordIdentity) {
        self.mark_gone(id.origin, id.oseq, id.oseq);
    }
}
