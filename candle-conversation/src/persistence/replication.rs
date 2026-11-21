//! Tracks which records of each origin this member holds, so records replicate to peers and a catch-up asks only for what is missing.
//!
//! Every record with an identity is located, another origin's record is applied, and a replica record stores what this member has applied, so the record a walk reads says which gaps still stand. Per-stream metadata records are tracked by location so that their current copy counts as live weight. The snapshot map drops a timeline's tail when the timeline is tombstoned or distilled, and both rewrite paths agree because the map does that dropping rather than each rewrite.

use super::*;

/// The [`MetadataKey`] of a record with header `h`.
pub(super) fn metadata_key(h: &RecordHeader) -> MetadataKey {
    metadata_key_parts(h.record_type, h.stream_id, h.origin)
}

/// The [`MetadataKey`] of a record of type `rt` on `stream_id`, written by
/// `origin`.
pub(super) fn metadata_key_parts(rt: RecordType, stream_id: u64, origin: u64) -> MetadataKey {
    let writer = match rt {
        RecordType::CustomObject
        | RecordType::Deleted
        | RecordType::Replica
        | RecordType::Rebase => 0,
        _ => origin,
    };
    (rt, stream_id, writer)
}

/// Whether a record type is a per-stream metadata record whose current-copy
/// location is tracked in [`SubstratePersistence::metadata_locs`] so it counts
/// as live weight (see that field). These are the records the maintenance
/// resident-set re-emits and that carry no location in the substrate index.
pub(super) fn is_tracked_metadata(rt: RecordType) -> bool {
    // `Snapshot` is deliberately NOT here: its payload never lives in RAM, so
    // the maintenance resident-set (which re-encodes from RAM) cannot carry
    // it. It is tracked through the substrate index and relocated verbatim
    // like `Tokens` — see `segment_liveness` and `gather_relocations`.
    matches!(
        rt,
        RecordType::StreamDecl
            | RecordType::ProjectionEvents
            | RecordType::WideQSig
            | RecordType::TurnIndexPage
            | RecordType::Commit
            | RecordType::CustomObject
            | RecordType::Deleted
            | RecordType::Replica
            | RecordType::Rebase
    )
}

/// The location of a walked record.
pub(super) fn walked_loc(entry: &walker::WalkEntry) -> RecordLoc {
    RecordLoc {
        segment: entry.segment,
        offset: entry.offset,
        payload_len: entry.record.header.payload_len,
        record_size: entry.size,
    }
}

/// Offer a walked payload-keyed resident record's location to `map` — the
/// greatest stamp per key holds it. The load / compact walk uses this; the
/// runtime append path uses [`SubstratePersistence::track_resident_loc`].
pub(super) fn record_resident_loc(map: &mut StampedLocs<ResidentKey>, entry: &walker::WalkEntry) {
    let h = &entry.record.header;
    if !keyed_by_payload(h.record_type) {
        return;
    }
    let id = h.identity();
    if let Some(key) = resident_key(h.record_type, h.stream_id, &entry.record.payload, &id) {
        map.offer(key, Some(walked_loc(entry)), id);
    }
}

/// Offer a walked record's metadata location to `map` — the greatest stamp per
/// key holds it. The load / compact walk uses this before the
/// `SubstratePersistence` exists; the runtime append path uses
/// [`SubstratePersistence::track_metadata_loc`].
pub(super) fn record_metadata_loc(map: &mut StampedLocs<MetadataKey>, entry: &walker::WalkEntry) {
    let h = &entry.record.header;
    if is_tracked_metadata(h.record_type) {
        map.offer(metadata_key(h), Some(walked_loc(entry)), h.identity());
    }
}

/// `entry` with its identity fixed: a record this node wrote cut off carries
/// the generation a rejoin fixed for it ([`RebaseMap`]); any other record is
/// itself.
pub(super) fn fix_walked<'a>(entry: &'a WalkEntry, rebases: &RebaseMap) -> Cow<'a, WalkEntry> {
    let id = entry.record.header.identity();
    let fixed = rebases.fix(id);
    if fixed == id {
        return Cow::Borrowed(entry);
    }
    let mut owned = entry.clone();
    owned.record.header = owned.record.header.with_identity(fixed);
    Cow::Owned(owned)
}

/// Take one walked record into the replica and the location index: every
/// record with an identity is located, another origin's is applied, and a
/// `Replica` record's sets join what is held. Both only grow, so the order the
/// walk meets them in does not matter.
///
/// A `Replica` record's entry for `node` itself is not the replica's: it is
/// this node's own written set as the record saw it, its gaps what a purge
/// dropped (§7.8). The last record walked is the latest written, so it alone
/// says which gaps still stand — `own_kept` takes its entry, or `None` when it
/// has none.
pub(super) fn observe_replica(
    replica: &mut ReplicaPayload,
    locations: &mut LocationIndex,
    own_kept: &mut Option<OriginApplied>,
    node: u64,
    entry: &walker::WalkEntry,
) {
    let h = &entry.record.header;
    let id = h.identity();
    if !id.is_none() {
        locations.insert(
            h,
            Located {
                segment: entry.segment,
                offset: entry.offset,
            },
        );
        if id.origin != node {
            replica.applied.insert(id.origin, id.oseq);
        }
    }
    if h.record_type == RecordType::Replica {
        if let Ok(mut held) = ReplicaPayload::decode(&entry.record.payload) {
            *own_kept = held.applied.remove(node);
            replica.applied.merge(&held.applied);
            replica.deferred.extend(held.deferred);
        }
    }
}

/// Take off the deferred set every record whose type this build decodes: it is
/// wanted again, so the next catch-up fetches it (§6.5). Returns whether any
/// was.
pub(super) fn release_known_deferred(replica: &mut ReplicaPayload) -> bool {
    let before = replica.deferred.len();
    replica
        .deferred
        .retain(|d| RecordType::from_name(&d.record_type) == RecordType::Unknown);
    replica.deferred.len() != before
}

/// Offer one walked record to the NPC-location map — the greatest stamp per
/// character holds it. Keyed by the header's `stream_id`, which for an `Npc`
/// record is the `npc_id` — so the walk never has to decode a payload to know
/// which character it belongs to.
///
/// A tombstoned character keeps its entry: `state: "tombstoned"` is a
/// superseding record like any other, and the id must stay taken so the acts
/// that already name it still resolve. Dropping it here would let a later
/// character reuse the id.
pub(super) fn record_npc_loc(map: &mut StampedLocs<u64>, entry: &walker::WalkEntry) {
    let h = &entry.record.header;
    if h.record_type == RecordType::Npc {
        map.offer(h.stream_id, Some(walked_loc(entry)), h.identity());
    }
}

/// Mirror one walked record into the snapshot-location map: a `Snapshot`
/// record offers its location and a retraction withdraws it, the greatest
/// stamp per stream holding; a timeline-scoped `Tombstone` removes the
/// timeline's entry, so a snapshot record replayed before its timeline's
/// tombstone never reads as live. The load / compact walk uses this before
/// the `SubstratePersistence` exists; the runtime append path uses
/// [`SubstratePersistence::track_snapshot_loc`] plus the removal in
/// [`SubstratePersistence::write_tombstone`].
pub(super) fn record_snapshot_loc(map: &mut StampedLocs<u64>, entry: &walker::WalkEntry) {
    let h = &entry.record.header;
    match h.record_type {
        RecordType::Snapshot if h.retracts_snapshot() => {
            map.offer(h.stream_id, None, h.identity());
        }
        RecordType::Snapshot | RecordType::BranchCheckpoint => {
            map.offer(h.stream_id, Some(walked_loc(entry)), h.identity());
        }
        RecordType::Tombstone => {
            if let Ok(p) = TombstonePayload::decode(&entry.record.payload) {
                if p.turn_index.is_none() {
                    map.remove(&snapshot_stream_id(p.timeline_id).0);
                }
            }
        }
        // A `Distilled` marker retires the tail for the same reason a tombstone
        // does: the recurrent state IS content, derived from a token stream that
        // distillation has shed, and useless without the K/V it was computed
        // against. A distilled conversation is never resumed — that is what
        // distillation means — so the state describes a resume that cannot
        // happen.
        //
        // Dropping it HERE, at the map, rather than at each rewrite is what
        // makes the two rewrite paths agree without either having to remember.
        // They did not: `collect_live_records` gated on the distilled set while
        // incremental maintenance relocated the same tails forward on every
        // pass, forever, on the reasoning that "the substrate map only ever
        // holds live conversations' tails" — true of tombstone, which removes
        // the entry, and false of distill, which did not. Measured on a
        // production store: 766 distilled timelines carrying ~116 MiB of state
        // each, ~89 GB of a 120 GB log, none of it resumable.
        RecordType::Distilled => {
            if let Ok(p) = record::DistillPayload::decode(&entry.record.payload) {
                map.remove(&snapshot_stream_id(p.timeline_id).0);
            }
        }
        _ => {}
    }
}

impl SubstratePersistence {
    /// The runtime twin of [`observe_replica`]: a record just placed at
    /// `(segment, offset)` is where its identity is now; another origin's
    /// record is applied; and one of this node's own, written for the first
    /// time and not cut off, is tapped (§6.3) — unless it is a type that never
    /// leaves this member (§6.1). Every member refuses such a record, so sent
    /// live it was refused, left a hole in the peer's applied set, and cost a
    /// gap catch-up to list it gone; the workspace template every restart
    /// writes was refused that way on both members on 2026-11-11. It still
    /// counts as written, so a peer's catch-up lists it gone. A compaction
    /// places records again, so one written cut off is judged by the
    /// generation a rejoin fixed for it, not the provisional one its header
    /// still carries.
    pub(super) fn track_identity(&mut self, h: &RecordHeader, segment: SegmentId, offset: u64) {
        let id = h.identity();
        if id.is_none() {
            return;
        }
        self.locations.insert(h, Located { segment, offset });
        if id.origin != self.node() {
            if self.replica.applied.insert(id.origin, id.oseq) {
                self.replica_dirty = true;
            }
        } else if self.rebases.fix(id).gen == GEN_DETACHED {
            self.own_hold.wrote_detached(id.oseq);
        } else if self.own_written.insert(id.oseq) && !h.record_type.is_local_only() {
            if let Some(tap) = &self.tap {
                tap(id.oseq);
            }
        }
    }

    /// Tell `tap` the `oseq` of every record this node writes from now on that
    /// may be sent, as it is appended — and of what it wrote cut off, when a
    /// rejoin fixes it (§6.3, §6.7). It is called on the log's owner thread,
    /// so it must not block.
    pub fn set_tap(&mut self, tap: Tap) {
        self.tap = Some(tap);
    }

    /// Every own `oseq` written and sendable: what this node can say it holds
    /// of itself.
    pub fn own_written(&self) -> &OriginApplied {
        &self.own_written
    }

    /// Every own `oseq` up to this one is written, or known gone: what this
    /// member may tell another it holds of itself (§7.8, R10).
    ///
    /// Not [`OriginApplied::top`]. A record is stamped when its value changes
    /// and appended when the log's owner reaches its job, so this node's
    /// `oseq`s reach the log out of order, and the top can stand above a record
    /// still queued — a peer told it held the top would take that record for
    /// gone. A gap a purge left at open is gone for good and holds nothing back;
    /// a gap above what the log held at open is a record still on its way, and
    /// what may be told stops below it. A record whose append failed is never
    /// coming ([`Self::mark_unwritten`]), so it does not hold the count back.
    pub fn own_sendable_through(&self) -> u64 {
        let written = &self.own_written;
        let hold = written
            .gaps()
            .into_iter()
            .filter(|&(start, _)| start > self.open_top)
            .find(|&(start, end)| !self.own_unwritten.contains_range(start, end));
        self.own_hold.observe(hold, written.top(), self.open_top);
        hold.map_or(written.top(), |(start, _)| start - 1)
    }

    /// `id`, minted by this node, will never be written: its append failed.
    /// It holds nothing back ([`Self::own_sendable_through`]).
    pub fn mark_unwritten(&mut self, id: RecordIdentity) {
        if id.origin == self.node() && id.oseq > 0 {
            self.own_unwritten.insert(id.oseq);
        }
    }

    /// What this member holds of every origin, its own written records
    /// included — what a catch-up starts from and a restart offers (§7.2,
    /// §7.5).
    pub fn applied_set(&self) -> AppliedSet {
        let mut set = self.replica.applied.clone();
        if self.own_written.top() > 0 {
            set.set_origin(self.node(), self.own_written.clone());
        }
        set
    }

    /// Every `oseq` of `origin` up to this one is applied here or known gone —
    /// where a catch-up of `origin` starts (§7.2). This node's own records are
    /// all its own `oseq`s so far.
    pub fn applied_through(&self, origin: u64) -> u64 {
        if origin == self.node() {
            self.minter.origination().next_oseq() - 1
        } else {
            self.replica.applied.through(origin)
        }
    }

    /// Whether `(origin, oseq)` is applied here or known gone. Of this node's
    /// own, every `oseq` minted so far but a gap a purge left (§7.8): what it
    /// wrote past its written set's top is held, cut off or not.
    pub fn is_applied(&self, origin: u64, oseq: u64) -> bool {
        if origin == self.node() {
            oseq != 0
                && oseq < self.minter.origination().next_oseq()
                && (oseq > self.own_written.top() || self.own_written.contains(oseq))
        } else {
            self.replica.applied.contains(origin, oseq)
        }
    }

    /// Record `start..=end` of `origin` as gone — `oseq`s a catch-up learned
    /// no member holds any more (§7.2), this node's own among them when a
    /// purge dropped them (§7.8). The next commit persists it.
    ///
    /// Of this node's own, only what lies at or below its written top is taken:
    /// above it is what it has not written yet, and covering that would leave a
    /// record it writes there untapped. A peer saying such an `oseq` is gone
    /// holds this origin past what this node wrote — said, because a catch-up
    /// then lists the same span every time it asks.
    pub fn mark_gone(&mut self, origin: u64, start: u64, end: u64) {
        let added = if origin == self.node() {
            let top = self.own_written.top();
            if end > top {
                tracing::warn!(
                    node = format!("{origin:016x}"),
                    from = start.max(top + 1),
                    to = end,
                    written_top = top,
                    next_oseq = self.minter.origination().next_oseq(),
                    "a peer lists own oseqs gone above what this node has written: not taken"
                );
            }
            self.own_written.insert_range(start, end.min(top))
        } else {
            self.replica.applied.insert_range(origin, start, end)
        };
        if added {
            self.replica_dirty = true;
        }
    }

    /// Defer a record this build does not decode: it is neither written nor
    /// applied, and a catch-up does not ask for it again until a build that
    /// knows its type opens the substrate (§6.5). The next commit persists it.
    pub fn defer(&mut self, deferred: DeferredRecord) {
        if self.replica.deferred.insert(deferred) {
            self.replica_dirty = true;
        }
    }

    /// Whether `(origin, oseq)` is deferred.
    pub fn is_deferred(&self, origin: u64, oseq: u64) -> bool {
        self.replica
            .deferred
            .iter()
            .any(|d| d.origin == origin && d.oseq == oseq)
    }

    /// What this member has applied of other origins, and what it deferred.
    pub fn replica(&self) -> &ReplicaPayload {
        &self.replica
    }

    /// Where `(origin, oseq)` is, if this member holds it.
    pub fn location_of(&self, origin: u64, oseq: u64) -> Option<Located> {
        self.locations.get(origin, oseq)
    }

    /// This member is cut off from every cluster: what it writes from here on
    /// is detached, under the provisional generation, until [`Self::rejoin`]
    /// (§6.7).
    pub fn detach(&self) {
        self.minter.detach(None);
    }

    /// Whether this member is cut off — detached since [`Self::detach`], or
    /// found so at open: it holds records it wrote detached that no rejoin
    /// fixed.
    pub fn is_detached(&self) -> bool {
        self.minter.origination().is_detached()
    }

    /// The generations fixed for what this node wrote cut off.
    pub fn rebases(&self) -> &RebaseMap {
        &self.rebases
    }

    /// This member rejoined (§6.7). When it wrote anything while cut off, a
    /// generation is decided for it through `generations`, fixed for every
    /// record it wrote detached by a `Rebase` record made durable here, and
    /// minted from here on; the entry is returned. When it wrote nothing it
    /// has nothing to place, takes no generation, and returns `None`.
    ///
    /// A substrate built from this store holds those records' values under the
    /// provisional generation still: the caller reloads it, as every rejoin
    /// does (§6.6). The fixed records are tapped, so the caller lets the
    /// substrate writer drain first: a record minted cut off and still queued
    /// would be tapped before it could be read.
    pub fn rejoin(&mut self, generations: &dyn MintGeneration) -> Result<Option<RebaseEntry>> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
        let origination = self.minter.origination();
        let Some((from, through)) = origination.detached_range() else {
            self.minter.attach(origination.gen());
            return Ok(None);
        };
        let entry = RebaseEntry {
            origin: self.node(),
            from,
            through,
            gen: generations.mint_generation(origination.gen()),
        };
        self.rebases.add(entry);
        let payload = self.rebases.encode();
        self.append_carried(
            RecordType::Rebase,
            0,
            0,
            0,
            0,
            0,
            &payload,
            RecordIdentity::NONE,
        )?;
        self.commit()?;
        self.minter.attach(entry.gen);
        // Fixed, they may be sent now (§6.7 step 3). The caller has let the
        // writer drain first, so each of them is in the log to be read.
        self.own_written.insert_range(from, through);
        if let Some(tap) = &self.tap {
            for oseq in from..=through {
                tap(oseq);
            }
        }
        Ok(Some(entry))
    }

    /// The replica as its record holds it: every other origin's applied set,
    /// and this node's own written set, whose gaps are what a purge dropped
    /// and no catch-up has returned (§7.8). The walk takes the own entry from
    /// the latest record alone ([`observe_replica`]), so once the store holds
    /// any replica record every record carries it: one without would leave an
    /// older record's gaps standing whenever maintenance reclaimed the newer
    /// first. A node that has never applied another origin, holds no replica
    /// record and has no gaps writes none.
    pub(super) fn replica_record(&self) -> ReplicaPayload {
        let mut record = self.replica.clone();
        let holds_one = self.facts.holds(1u32 << RecordType::Replica.tag());
        let needed = holds_one || !self.replica.is_empty() || !self.own_written.gaps().is_empty();
        if self.own_written.top() > 0 && needed {
            record
                .applied
                .set_origin(self.node(), self.own_written.clone());
        }
        record
    }

    /// Write the replica when it changed since it was last written — at a
    /// commit, before the sync that makes it durable. Local housekeeping: it
    /// carries no identity and is never replicated.
    pub(super) fn write_replica_if_changed(&mut self) -> Result<()> {
        if !self.replica_dirty {
            return Ok(());
        }
        let payload = self.replica_record().encode();
        self.append_carried(
            RecordType::Replica,
            0,
            0,
            0,
            0,
            0,
            &payload,
            RecordIdentity::NONE,
        )?;
        self.replica_dirty = false;
        Ok(())
    }
}
