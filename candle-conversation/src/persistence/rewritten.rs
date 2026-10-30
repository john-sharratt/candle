//! What a full rewrite of the log leaves the substrate to do
//! (`docs/substrate_lock_free.md` §9.3).
//!
//! A compaction writes every live record into one fresh segment. The records
//! keep their identities, so applying the fresh segment's records to the
//! substrate a second time changes nothing but where each one is: a value the
//! substrate holds under the same identity takes the new location, and one a
//! newer record replaced since the rewrite read the substrate keeps the newer
//! one — the same record again supersedes itself, an older one does not. That
//! makes the rewrite's in-RAM
//! side an edit like any other — it never clears the substrate and rebuilds it,
//! so an edit made while the rewrite ran survives it.
//!
//! What the rewrite did not carry still points into the old segments, and is
//! dropped after the re-apply: the locations into them, the declarations of
//! streams it dropped wholesale, and the tombstones it expired. Only then are
//! the old segments unlinked, so a reader holding an old location reads it
//! until the substrate stops naming it.

use std::collections::HashSet;

use super::identity::RecordIdentity;
use super::record::RecordType;
use super::segment::SegmentId;
use super::streams::StreamId;
use super::tombstone_expiry::Expired;
use super::walker::WalkEntry;
use crate::substrate::{StreamKey, Substrate};

/// Walked records applied per substrate edit when a rewrite is applied to a
/// shared root ([`Rewritten::batches`]).
pub const REAPPLY_BATCH: usize = 4096;

/// See the module documentation.
pub struct Rewritten {
    /// The fresh segment's records, bulk payloads stripped: the substrate
    /// reads a chunk's, tokens' or snapshot's header, never its bytes.
    walked: Vec<WalkEntry>,
    /// The segments the rewrite replaced.
    old_segments: HashSet<SegmentId>,
    /// Stream declarations the substrate held when the rewrite read it and
    /// the rewrite did not carry, with the identity each was held under.
    dropped_decls: Vec<(StreamKey, RecordIdentity)>,
    /// The tombstones the rewrite expired.
    expired: Expired,
}

impl Rewritten {
    pub(super) fn new(
        walked: Vec<WalkEntry>,
        old_segments: HashSet<SegmentId>,
        read: &Substrate,
        expired: Expired,
    ) -> Self {
        let carried: HashSet<StreamKey> = walked
            .iter()
            .filter(|e| e.record.header.record_type == RecordType::StreamDecl)
            .map(|e| {
                let h = &e.record.header;
                StreamKey::of(StreamId(h.stream_id), &h.identity())
            })
            .collect();
        let dropped_decls = read
            .all_physical_streams()
            .filter(|(key, rt)| rt.decl.is_some() && !carried.contains(key))
            .map(|(key, rt)| (key, rt.ids.decl))
            .collect();
        Self {
            walked,
            old_segments,
            dropped_decls,
            expired,
        }
    }

    /// A walked record kept for its re-apply: everything the substrate reads
    /// from it, without the bulk bytes it never reads.
    pub(super) fn keep(entry: &WalkEntry) -> WalkEntry {
        let mut kept = entry.clone();
        if matches!(
            entry.record.header.record_type,
            RecordType::Chunk
                | RecordType::Tokens
                | RecordType::Snapshot
                | RecordType::BranchCheckpoint
        ) {
            kept.record.payload = Vec::new();
        }
        kept
    }

    /// The segments to unlink once the substrate names none of them.
    pub fn old_segments(&self) -> &HashSet<SegmentId> {
        &self.old_segments
    }

    /// The whole re-apply, then [`Self::settle`] — for a substrate no other
    /// thread edits.
    pub fn apply_to(&self, substrate: &mut Substrate) {
        for entry in &self.walked {
            substrate.apply_walker_entry(entry);
        }
        self.settle(substrate);
    }

    /// The re-apply in pieces of [`REAPPLY_BATCH`] records, each a valid edit
    /// on its own — for a shared root, where one edit of every record would be
    /// retried whole each time another writer published first. Run
    /// [`Self::settle`] after the last.
    pub fn batches(&self) -> impl Iterator<Item = &[WalkEntry]> + '_ {
        self.walked.chunks(REAPPLY_BATCH)
    }

    /// Drop what the rewrite did not carry: every location still into an old
    /// segment, each stream it dropped wholesale whose declaration is still the
    /// one it read, and the tombstones it expired; then repoint the residences'
    /// cold references at the relocated chunks. Idempotent.
    pub fn settle(&self, substrate: &mut Substrate) {
        substrate.drop_locations_into(&self.old_segments);
        for &(key, decl) in &self.dropped_decls {
            substrate.forget_dropped_stream(key, decl);
        }
        substrate.forget_expired_tombstones(&self.expired);
        substrate.refresh_cold_refs();
    }
}
