//! Gathers everything a maintenance op needs under a brief read lock.
//!
//! The execute phase works from this plan alone, so the slow relocation I/O
//! holds no substrate lock. Records the substrate holds only on disk, such as
//! characters, are listed by location and carried verbatim, because nothing in
//! this process can rebuild them. Each planned resident record keeps the
//! identity of the copy it was taken from, and a write made after the plan is
//! re-applied behind it.

use super::*;

/// One already-encoded record to relocate verbatim from a source segment: its
/// on-disk `(offset, record_size)`. The copy's bookkeeping reads the header
/// out of the copied bytes ([`SubstratePersistence::append_raw_record`]), so it
/// carries the record's identity and digest exactly as written.
pub(super) struct RawReloc {
    pub(super) offset: u64,
    pub(super) record_size: u64,
}

/// A planned maintenance op with everything the execute phase needs, gathered
/// from the substrate under a **brief** read lock — so the slow relocation I/O
/// runs without holding the substrate lock (mirrors the persistence thread's
/// hot→warm / warm→cold discipline: snapshot locked, I/O unlocked, install
/// locked). See [`SubstratePersistence::plan_maintenance`].
pub struct MaintenancePlan {
    pub(super) op: MaintenanceOp,
    /// The resident records whose current copy the op is about to drop
    /// ([`SubstratePersistence::resident_to_carry`]), each with the identity of
    /// the record holding its value when the plan was made — which the copy
    /// keeps. A write made since the plan is carried after it under its own
    /// ([`RelocationWatch`]).
    pub(super) resident: Vec<Resident>,
    /// `(stream, chunk_index, source_loc)` chunks to relocate off the targets.
    pub(super) chunk_relocs: Vec<(StreamKey, u64, ChunkLoc)>,
    /// `(stream, source_loc)` `Tokens` records to relocate.
    pub(super) token_relocs: Vec<(StreamKey, RecordLoc)>,
    /// `(stream, source_loc)` QSA index pages held in the log to relocate —
    /// carried by location like `Tokens`, since their bytes are not in RAM
    /// (`IndexPage::Stored`).
    pub(super) index_page_relocs: Vec<(StreamKey, RecordLoc)>,
    /// `(snapshot stream, source_loc)` recurrent-state snapshots to relocate.
    pub(super) snapshot_relocs: Vec<(StreamId, RecordLoc)>,
    /// `(branch stream, source_loc)` prompt branch checkpoints to relocate.
    pub(super) branch_checkpoint_relocs: Vec<(StreamId, RecordLoc)>,
    /// `(npc stream, source_loc)` character records to relocate.
    ///
    /// **Relocated, never re-emitted.** An NPC's payload is not in this
    /// process's RAM — the registry that owns it lives in the daemon above — so
    /// unlike every other resident record it cannot be rebuilt from substrate
    /// state and must be carried across byte-for-byte, exactly as `Snapshot` is.
    /// Omitting it deleted characters whenever the segment holding them was
    /// maintained.
    ///
    /// Counting an NPC's bytes as live (see `live_bytes_by_segment`) is not a
    /// substitute — it only keeps a segment off the `live_bytes == 0` Drop path,
    /// while `Compact` and `Combine` qualify on a dead *ratio* and retire the
    /// segment just the same.
    pub(super) npc_relocs: Vec<(StreamId, RecordLoc)>,
    /// A live conversation's file events and tombstones to relocate —
    /// carried by location for the reason characters are: their bodies are
    /// not in this process's RAM. A tombstone is carried as long as its
    /// conversation lives, because copies of the events it killed can sit in
    /// segments this op leaves alone, and it has to stay after them.
    pub(super) vfs_relocs: Vec<VfsReloc>,
    /// Blobs to relocate, by location: their bytes are never in this
    /// process's RAM either.
    pub(super) blob_relocs: Vec<(BlobKey, RecordLoc)>,
    /// `(type, source_loc)` singleton records to relocate.
    pub(super) singleton_relocs: Vec<(RecordType, RecordLoc)>,
}

/// One file event or tombstone to relocate, keyed as the index keys it.
#[derive(Clone, Copy, Debug)]
pub(in super::super) struct VfsReloc {
    pub(in super::super) rt: RecordType,
    pub(super) timeline: u64,
    pub(super) id: VfsEventId,
    pub(in super::super) loc: RecordLoc,
}

impl MaintenancePlan {
    /// The op this plan executes.
    pub fn op(&self) -> MaintenanceOp {
        self.op
    }
}

/// An op a liveness scan picked, and the stats it picked it from — what
/// [`SubstratePersistence::plan_chosen`] gathers a plan for.
pub struct Chosen {
    pub(super) op: MaintenanceOp,
    pub(super) stats: Vec<SegmentStat>,
    /// The scan's own time, for the plan's log line.
    pub(super) liveness_ms: u64,
}

impl MaintenancePlan {
    /// Every record this plan found in a target that `substrate`'s index still
    /// names there — read after the apply, so a record the apply repointed, or
    /// one rewritten since the plan, is not among them.
    pub(super) fn stranded_in(&self, substrate: &Substrate) -> Stranded {
        let record =
            |kind, stream_id: StreamId, writer, chunk_index, segment, offset| StrandedRecord {
                kind,
                stream_id: stream_id.0,
                writer,
                chunk_index,
                segment,
                offset,
            };
        let chunks = self.chunk_relocs.iter().map(|(key, idx, old)| {
            let (seg, off) = (old.segment, old.offset);
            record(
                StrandedKind::Chunk,
                key.stream_id,
                key.writer,
                *idx,
                seg,
                off,
            )
        });
        let tokens = self.token_relocs.iter().map(|(key, old)| {
            let (seg, off) = (old.segment, old.offset);
            record(StrandedKind::Tokens, key.stream_id, key.writer, 0, seg, off)
        });
        let pages = self.index_page_relocs.iter().map(|(key, old)| {
            let (seg, off) = (old.segment, old.offset);
            record(
                StrandedKind::IndexPage,
                key.stream_id,
                key.writer,
                0,
                seg,
                off,
            )
        });
        let snapshots = self
            .snapshot_relocs
            .iter()
            .map(|(sid, old)| record(StrandedKind::Snapshot, *sid, 0, 0, old.segment, old.offset));
        let branches = self.branch_checkpoint_relocs.iter().map(|(sid, old)| {
            let (seg, off) = (old.segment, old.offset);
            record(StrandedKind::BranchCheckpoint, *sid, 0, 0, seg, off)
        });
        let mut out = Stranded::default();
        for r in chunks
            .chain(tokens)
            .chain(pages)
            .chain(snapshots)
            .chain(branches)
            .filter(|r| r.named_in(substrate))
        {
            out.push(r);
        }
        out
    }
}

impl SubstratePersistence {
    /// Run one maintenance op end to end (plan → execute → apply → drop) under
    /// the caller's already-held substrate lock. Convenience for single-threaded
    /// callers (tests); the daemon uses the **phased** API — [`Self::plan_maintenance`]
    /// → [`Self::execute_maintenance`] → [`MaintenanceResult::apply_to_substrate`]
    /// → [`Self::finish_maintenance`] — so the slow relocation I/O never holds
    /// the substrate write lock. Returns the op that ran, or `None` for a no-op.
    pub fn run_maintenance(&mut self, substrate: &mut Substrate) -> Result<Option<MaintenanceOp>> {
        let Some(plan) = self.plan_maintenance(substrate, false)? else {
            return Ok(None);
        };
        let op = plan.op;
        let result = self.execute_maintenance(&plan)?;
        let stranded = result.apply_to_substrate(&plan, substrate);
        unlink_now(self.finish_maintenance(&plan, &stranded)?)?;
        Ok(Some(op))
    }

    /// Count of sealed segments plus the active — surfaced in the daemon's
    /// status for the GUI's compaction indicator.
    pub fn segment_count(&self) -> usize {
        self.segments.sealed_ids().len() + 1
    }

    /// **Phase 1** — pick an op and gather everything needed to execute it,
    /// reading the substrate + manifest. Runs under a brief read lock; returns
    /// `None` for a no-op pass. The gathered plan (resident snapshot +
    /// relocation worklist) lets the slow I/O phase run without the substrate
    /// lock.
    ///
    /// A read-only handle plans no op, forced or not: maintaining the store is
    /// the business of the process that owns it.
    pub fn plan_maintenance(
        &self,
        substrate: &Substrate,
        force: bool,
    ) -> Result<Option<MaintenancePlan>> {
        let indexes = self.live_indexes();
        Ok(indexes
            .choose_maintenance(substrate, &self.segment_sizes()?, force)
            .map(|chosen| indexes.plan_chosen(substrate, chosen)))
    }

    /// [`Self::plan_maintenance`] restricted to dropping a fully-dead segment:
    /// what a member whose volume is full may still run, since a drop needs no
    /// scratch space (`docs/replicated_substrate.md` §7.7).
    pub fn plan_drop(&self, substrate: &Substrate) -> Result<Option<MaintenancePlan>> {
        let indexes = self.live_indexes();
        Ok(indexes
            .choose_drop(substrate, &self.segment_sizes()?)
            .map(|chosen| indexes.plan_chosen(substrate, chosen)))
    }

    /// Live record bytes per segment ([`LiveIndexes::segment_liveness`]).
    pub fn segment_liveness(&self, substrate: &Substrate) -> HashMap<SegmentId, u64> {
        self.live_indexes().segment_liveness(substrate)
    }
}

impl LiveIndexes {
    /// The first half of a plan: the per-segment liveness scan and the op it
    /// picks, or `None` for a no-op pass. Apart from [`Self::plan_chosen`] so
    /// the plan reads a later version of the substrate than the scan did. The
    /// chosen op stays valid across the gap: only maintenance retires a
    /// segment, and it runs one op at a time.
    pub fn choose_maintenance(
        &self,
        substrate: &Substrate,
        sizes: &SegmentSizes,
        force: bool,
    ) -> Option<Chosen> {
        self.choose_with(substrate, sizes, |stats| pick_maintenance_op(stats, force))
    }

    /// [`Self::choose_maintenance`] restricted to dropping a fully-dead
    /// segment ([`SubstratePersistence::plan_drop`]).
    pub fn choose_drop(&self, substrate: &Substrate, sizes: &SegmentSizes) -> Option<Chosen> {
        self.choose_with(substrate, sizes, pick_drop)
    }

    pub(super) fn choose_with(
        &self,
        substrate: &Substrate,
        sizes: &SegmentSizes,
        pick: impl Fn(&[SegmentStat]) -> Option<MaintenanceOp>,
    ) -> Option<Chosen> {
        if self.is_read_only() || sizes.sealed.is_empty() {
            return None;
        }
        let t_liveness = Instant::now();
        let liveness = self.segment_liveness(substrate);
        let liveness_ms = t_liveness.elapsed().as_millis() as u64;
        self.note_dead_ratio(sizes.total_record_bytes, liveness.values().sum());
        // A segment kept for a record the index still names there would be
        // picked, kept and picked again, and no other segment compacted.
        let held = self.stranded.held_in(substrate);
        let stats: Vec<SegmentStat> = sizes
            .sealed
            .iter()
            .filter(|s| !held.contains(&s.id))
            .map(|s| SegmentStat {
                id: s.id,
                total_bytes: s.total_bytes,
                live_bytes: liveness.get(&s.id).copied().unwrap_or(0),
                age_secs: s.age_secs,
            })
            .collect();
        pick(&stats).map(|op| Chosen {
            op,
            stats,
            liveness_ms,
        })
    }

    /// The second half of a plan: everything `chosen`'s op relocates and
    /// re-emits, gathered so the relocation's batches run from a list.
    pub fn plan_chosen(&self, substrate: &Substrate, chosen: Chosen) -> MaintenancePlan {
        let Chosen {
            op,
            stats,
            liveness_ms,
        } = chosen;
        // Re-emit only the resident records whose current copy the op is about
        // to drop (see `resident_to_carry`).
        let t_resident = Instant::now();
        let resident = self.resident_to_carry(substrate, &op.targets());
        let resident_ms = t_resident.elapsed().as_millis() as u64;
        let t_relocs = Instant::now();
        let (
            chunk_relocs,
            token_relocs,
            snapshot_relocs,
            branch_checkpoint_relocs,
            singleton_relocs,
        ) = self.gather_relocations(substrate, &op.targets());
        let npc_relocs = self.npc_relocations(&op.targets());
        let vfs_relocs = self.vfs_relocations(substrate, &op.targets());
        let blob_relocs = self.blob_relocations(&op.targets());
        let index_page_relocs = self.gather_index_page_relocs(substrate, &op.targets());
        let relocs_ms = t_relocs.elapsed().as_millis() as u64;
        // What this op carries off the target segments, by type. The incremental
        // path is the one that actually runs on a busy store — the 143 GB store
        // burned through ~370 segment generations without a single full
        // compaction — so this is where a missing carrier shows up first. Two
        // small tallies over vectors already in RAM, once per op (which is at
        // most a few times a minute), and the summary string is built only for
        // the line itself.
        let mut resident_census = RecordCensus::new();
        for r in &resident {
            resident_census.record(r.rt);
        }
        // Why the op ran: its targets' sizes, so a store that keeps compacting
        // the same data can be read straight off the log.
        let (target_bytes, target_live, target_age) =
            stats.iter().filter(|s| op.targets().contains(&s.id)).fold(
                (0u64, 0u64, None),
                |(t, l, a): (u64, u64, Option<u64>), s| {
                    let age = a.map_or(s.age_secs, |a| a.min(s.age_secs));
                    (t + s.total_bytes, l + s.live_bytes, Some(age))
                },
            );
        // An op with no matching targets has no age; log 0 rather than a
        // sentinel that reads as a number.
        let target_age = target_age.unwrap_or(0);
        tracing::info!(
            target: "candle_conversation::persistence::census",
            dir = %self.dir().display(),
            op = op.label(),
            target_bytes,
            target_live,
            target_dead_pct = (target_bytes.saturating_sub(target_live) * 100)
                .checked_div(target_bytes)
                .unwrap_or(0),
            target_age_secs = target_age,
            reemit = resident_census.total(),
            chunks = chunk_relocs.len(),
            tokens = token_relocs.len(),
            snapshots = snapshot_relocs.len(),
            branches = branch_checkpoint_relocs.len(),
            singletons = singleton_relocs.len(),
            npcs = npc_relocs.len(),
            vfs = vfs_relocs.len(),
            blobs = blob_relocs.len(),
            index_pages = index_page_relocs.len(),
            liveness_ms,
            resident_ms,
            relocs_ms,
            "maintenance {:?}: re-emitting {}",
            op,
            resident_census.summary()
        );
        MaintenancePlan {
            op,
            resident,
            chunk_relocs,
            token_relocs,
            index_page_relocs,
            snapshot_relocs,
            branch_checkpoint_relocs,
            singleton_relocs,
            npc_relocs,
            vfs_relocs,
            blob_relocs,
        }
    }

    /// The resident records an op targeting `targets` must re-emit.
    ///
    /// **Only what the op is about to drop.** Every resident type's current
    /// copy is location-tracked (`metadata_locs` / `resident_locs`), and a
    /// record is re-emitted exactly when that copy sits in a target segment:
    /// anywhere else, the drop cannot touch it. Re-emitting the whole set
    /// instead wrote every live turn's index page, signature and events forward
    /// on every op — 2.2 GB across ~12,900 records on the live daemon. That
    /// re-emission spilled over the active segment, left the previous op's
    /// copies dead in the segments just sealed, and made them the next op's
    /// targets: one compaction every 15–60 s on segments barely a minute old,
    /// each rewriting the set again and carrying the same chunks forward each
    /// time. Section tombstones were the last type carried that way, and grew
    /// with every section a character retired: 12,000 of them re-emitted on
    /// each op held the persistence lock for 18 s.
    ///
    /// A record with no location entry is re-emitted unconditionally — the map
    /// has no copy to vouch for, so the op makes one.
    pub(super) fn resident_to_carry(
        &self,
        substrate: &Substrate,
        targets: &[SegmentId],
    ) -> Vec<Resident> {
        let mut set = self.resident_set(substrate);
        set.retain(|r| {
            self.current_resident_loc(r)
                .is_none_or(|loc| targets.contains(&loc.segment))
        });
        set
    }

    /// Where the current on-disk copy of `r` is, as the location maps record
    /// it — keyed as an append keys it, by the record's identity. `None` when
    /// the map holds no entry for it.
    pub(super) fn current_resident_loc(&self, r: &Resident) -> Option<RecordLoc> {
        let loc = if is_tracked_metadata(r.rt) {
            self.metadata_locs
                .get(&metadata_key_parts(r.rt, r.stream_id, r.id.origin))
        } else if keyed_by_payload(r.rt) {
            resident_key(r.rt, r.stream_id, &r.payload, &r.id)
                .and_then(|k| self.resident_locs.get(&k))
        } else {
            None
        };
        loc.copied()
    }
}
