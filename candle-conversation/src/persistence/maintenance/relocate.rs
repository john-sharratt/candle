//! Moves an op's records into the active segment, one lock-bounded batch at a
//! time.
//!
//! The persistence lock is held for one batch, so a seal or a cold load waits
//! for a bounded amount of work rather than the whole op. Chunks are copied
//! verbatim from coalesced reads, without decoding. A record rewritten since the
//! plan is skipped, or its newer writes are appended after the planned copy, so
//! a reload never restores an older value. An op cannot complete while batches
//! remain.

use super::*;

/// Record bytes one batch re-emits or relocates under a single hold of the
/// persistence lock. Relocation reads and re-appends at ~100 MB/s (a 3.6 GB
/// segment took 37 s), so this bounds what a seal write or cold load waits
/// behind maintenance to roughly 150 ms.
pub const RELOCATION_BATCH_BYTES: u64 = 16 * 1024 * 1024;

/// Resident records one batch re-emits under a single hold, whatever their
/// size. Most resident records are a few hundred bytes — a section tombstone
/// is under one sector — so a byte budget alone let one batch take every
/// record of a large set, and each still costs an append's bookkeeping.
pub const RELOCATION_BATCH_RECORDS: usize = 1024;

/// A maintenance op in progress, carried between the batches of
/// [`SubstratePersistence::relocate_batch`].
pub struct MaintenanceRun {
    /// How many of the plan's `resident` records have been taken.
    pub(super) resident_cursor: usize,
    /// How many of the plan's `chunk_relocs` have been taken.
    pub(super) chunk_cursor: usize,
    /// `(stream, chunk_index, old_loc, new_loc)` of every chunk moved so far.
    pub(super) chunk_updates: Vec<(StreamKey, u64, ChunkLoc, ChunkLoc)>,
    /// Which tail stage [`SubstratePersistence::relocate_tail_batch`] is in.
    /// Tokens, snapshots, branch checkpoints, singletons, characters, file
    /// events, blobs, index pages, then done.
    pub(super) tail_stage: u8,
    /// How far into the current tail stage the cursor has walked.
    pub(super) tail_cursor: usize,
    pub(super) token_updates: Vec<(StreamKey, RecordLoc, RecordLoc)>,
    pub(super) index_page_updates: Vec<(StreamKey, RecordLoc, RecordLoc)>,
    pub(super) snapshot_updates: Vec<(StreamId, RecordLoc, RecordLoc)>,
    pub(super) branch_checkpoint_updates: Vec<(StreamId, RecordLoc, RecordLoc)>,
}

/// Tail stages of [`SubstratePersistence::relocate_tail_batch`], in order.
pub(super) const TAIL_TOKENS: u8 = 0;

pub(super) const TAIL_SNAPSHOTS: u8 = 1;

pub(super) const TAIL_BRANCHES: u8 = 2;

pub(super) const TAIL_SINGLETONS: u8 = 3;

pub(super) const TAIL_NPCS: u8 = 4;

pub(super) const TAIL_VFS: u8 = 5;

pub(super) const TAIL_BLOBS: u8 = 6;

pub(super) const TAIL_INDEX_PAGES: u8 = 7;

pub(super) const TAIL_DONE: u8 = 8;

/// Skipped records examined under one hold before the lock is released, so a
/// stage of nothing-but-superseded entries cannot sit on the lock.
pub(super) const TAIL_EXAMINE_CAP: usize = 4096;

impl MaintenanceRun {
    /// The index updates the batches accumulated, once every stage has run —
    /// taken out of the run, which an op held across passes still owns.
    pub fn take_result(&mut self) -> MaintenanceResult {
        MaintenanceResult {
            chunk_updates: std::mem::take(&mut self.chunk_updates),
            token_updates: std::mem::take(&mut self.token_updates),
            index_page_updates: std::mem::take(&mut self.index_page_updates),
            snapshot_updates: std::mem::take(&mut self.snapshot_updates),
            branch_checkpoint_updates: std::mem::take(&mut self.branch_checkpoint_updates),
        }
    }
}

impl SubstratePersistence {
    /// **Phase 2** — the slow I/O: append the resident set, then relocate each
    /// target segment's live records into the active, then commit. Holds only
    /// the persistence lock; the caller does **not** hold the substrate lock
    /// here, so decode's in-RAM projection reads/writes proceed. The relocated
    /// copies are durable (fsynced) before any source is unlinked. Returns the
    /// new locations for [`MaintenanceResult::apply_to_substrate`]. A
    /// read-only handle refuses with [`PersistenceError::ReadOnly`].
    pub fn execute_maintenance(&mut self, plan: &MaintenancePlan) -> Result<MaintenanceResult> {
        let mut run = self.begin_maintenance(plan)?;
        while self.relocate_batch(plan, &mut run, u64::MAX)? {}
        self.complete_maintenance(plan, run)
    }

    /// Start watching the writes the op about to run must step around — called
    /// in the same hold of this handle's lock as [`Self::plan_maintenance`], so
    /// no append can fall between the plan and the watch. See
    /// [`RelocationWatch`].
    ///
    /// # Panics
    ///
    /// When an op is already in flight ([`Self::relocation_in_flight`]): two
    /// ops would share one watch, and the second would relocate out of
    /// segments the first is about to unlink. Callers check first.
    pub fn watch_relocation(&mut self) {
        assert!(
            self.relocation_watch.is_none(),
            "a second maintenance op was planned while one is relocating"
        );
        self.relocation_watch = Some(RelocationWatch::default());
    }

    /// Whether a maintenance op is between its plan and the unlink of its
    /// sources (or its failure). A caller that finds one in flight plans none.
    pub fn relocation_in_flight(&self) -> bool {
        self.relocation_watch.is_some()
    }

    /// Stop watching — the op finished or was abandoned.
    pub fn end_relocation_watch(&mut self) {
        self.relocation_watch = None;
    }

    /// Put back what an op that ends **without its apply** moved on the
    /// persistence side: an op that failed part-way, or was abandoned. Returns
    /// how many entries were restored.
    ///
    /// A relocated copy keeps its record's identity, so its append moves the
    /// live-tail maps onto it at once (`track_snapshot_loc`,
    /// `track_metadata_loc`), while the substrate index moves only at
    /// [`MaintenanceResult::apply_to_substrate`]. An op that never applies
    /// would leave the maps on the copy and the index on the source, and the
    /// next op over the source's segment would plan the source from the index,
    /// filter it out against the maps, and unlink it under the index: a record
    /// indexed but unreadable until a restart re-walks the log. Restored, the
    /// maps name the source the index names, under the same identity; the
    /// copies are dead weight a later op reclaims.
    ///
    /// **Only where `substrate`'s index still names the source.** An entry the
    /// index has moved onto the copy — an apply that ran — keeps the copy:
    /// put back, the maps would name the source while the index named the
    /// copy, the same divergence the other way. `why` is what ended the op,
    /// logged beside the repair.
    ///
    /// The byte accounting needs nothing: the relocation counted the source's
    /// bytes dead and the copy took its record's live entry at the same size,
    /// and once the maps name the source again, the copy is the record nothing
    /// indexes — the same bytes, dead.
    pub fn unwind_relocation(
        &mut self,
        plan: &MaintenancePlan,
        run: &MaintenanceRun,
        substrate: &Substrate,
        why: &str,
    ) -> usize {
        let mut restored = 0;
        for (sid, old, new) in &run.snapshot_updates {
            if self.snapshot_locs.get(&sid.0) == Some(new)
                && substrate.recurrent_snapshot_loc(*sid) == Some(*old)
            {
                if let Some(id) = self.snapshot_locs.identity(&sid.0) {
                    restored += usize::from(self.snapshot_locs.offer(sid.0, Some(*old), id));
                }
            }
        }
        for (sid, old, new) in &run.branch_checkpoint_updates {
            if self.snapshot_locs.get(&sid.0) == Some(new)
                && substrate.branch_checkpoint_loc(*sid) == Some(*old)
            {
                if let Some(id) = self.snapshot_locs.identity(&sid.0) {
                    restored += usize::from(self.snapshot_locs.offer(sid.0, Some(*old), id));
                }
            }
        }
        for (key, old, new) in &run.index_page_updates {
            let page = metadata_key_parts(RecordType::TurnIndexPage, key.stream_id.0, key.writer);
            let indexed = substrate
                .stream_at(*key)
                .and_then(|s| s.index_page.as_ref().and_then(IndexPage::stored));
            if self.metadata_locs.get(&page) == Some(new) && indexed == Some(*old) {
                if let Some(id) = self.metadata_locs.identity(&page) {
                    restored += usize::from(self.metadata_locs.offer(page, Some(*old), id));
                }
            }
        }
        if restored > 0 {
            tracing::warn!(
                target: "candle_conversation::persistence::maintenance",
                op = plan.op.label(),
                restored,
                ended_by = why,
                "segment-maintenance: an op ended before its apply; the live-tail maps \
                 name its sources again",
            );
        }
        restored
    }

    /// The first step of [`Self::execute_maintenance`]: start the run the
    /// batches of [`Self::relocate_batch`] advance. Writes nothing.
    pub fn begin_maintenance(&mut self, plan: &MaintenancePlan) -> Result<MaintenanceRun> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
        Ok(MaintenanceRun {
            resident_cursor: 0,
            chunk_cursor: 0,
            chunk_updates: Vec::with_capacity(plan.chunk_relocs.len()),
            tail_stage: TAIL_TOKENS,
            tail_cursor: 0,
            token_updates: Vec::new(),
            index_page_updates: Vec::new(),
            snapshot_updates: Vec::new(),
            branch_checkpoint_updates: Vec::new(),
        })
    }

    /// Advance the op by one batch — up to `budget_bytes` of records, at least
    /// one — and commit it. The resident set is re-emitted first, then the
    /// target segments' chunks are relocated. Answers whether either remains.
    ///
    /// This is the unit the caller takes the lock for, so a seal write or a
    /// cold load waits for one batch instead of the whole op. Both halves are
    /// large: the chunks are gigabytes for a big segment, and the resident set
    /// is every live metadata record in the store — ~35,700 of them on the live
    /// daemon, whose single-hold re-emission held the lock for 44–52 s and
    /// stalled every finishing turn's token write behind it.
    pub fn relocate_batch(
        &mut self,
        plan: &MaintenancePlan,
        run: &mut MaintenanceRun,
        budget_bytes: u64,
    ) -> Result<bool> {
        if run.resident_cursor < plan.resident.len() {
            self.reemit_resident_batch(plan, run, budget_bytes)?;
            return Ok(true);
        }
        self.relocate_chunk_batch(plan, run, budget_bytes)
    }

    /// Re-emit the next run of the plan's resident set — up to `budget_bytes`
    /// of appended records and [`RELOCATION_BATCH_RECORDS`] planned records,
    /// at least one — and commit it.
    ///
    /// Re-emitted from in-RAM state (not read from disk), so the normal
    /// encoding append. A record written since the plan is followed by its
    /// newer writes, appended again, the way the watch says (see [`Reemit`]):
    /// the lock is released between batches, and a planned copy left as the
    /// last word would roll its record back on the next load.
    ///
    /// **The watch is set aside while the op appends its own copies.** It is
    /// there to see other writers; fed the re-emission, it would read a
    /// section's re-emitted declaration as a revival and skip the same
    /// section's planned tombstone a few records later.
    pub(super) fn reemit_resident_batch(
        &mut self,
        plan: &MaintenancePlan,
        run: &mut MaintenanceRun,
        budget_bytes: u64,
    ) -> Result<()> {
        let watch = self.relocation_watch.take();
        let appended = self.reemit_resident_run(plan, run, budget_bytes, watch.as_ref());
        self.relocation_watch = watch;
        appended?;
        self.commit()
    }

    /// The appends of [`Self::reemit_resident_batch`], judged against `watch`.
    pub(super) fn reemit_resident_run(
        &mut self,
        plan: &MaintenancePlan,
        run: &mut MaintenanceRun,
        budget_bytes: u64,
        watch: Option<&RelocationWatch>,
    ) -> Result<()> {
        let mut bytes = 0u64;
        let end = plan
            .resident
            .len()
            .min(run.resident_cursor + RELOCATION_BATCH_RECORDS);
        while run.resident_cursor < end && (bytes == 0 || bytes < budget_bytes) {
            let r = &plan.resident[run.resident_cursor];
            run.resident_cursor += 1;
            let verdict = watch.map_or(Reemit::Carry, |w| {
                w.reemit(r.rt, r.stream_id, r.id.origin, &r.payload)
            });
            let newer = match verdict {
                Reemit::Skip => continue,
                Reemit::Carry => Vec::new(),
                Reemit::CarryThen(newer) => newer,
            };
            bytes += self
                .append_carried(r.rt, 0, r.stream_id, r.chunk_index, 0, 0, &r.payload, r.id)?
                .size;
            for (chunk_index, payload, id) in &newer {
                bytes += self
                    .append_carried(r.rt, 0, r.stream_id, *chunk_index, 0, 0, payload, *id)?
                    .size;
            }
        }
        Ok(())
    }

    /// Relocate the next run of the plan's chunks — up to `budget_bytes` of
    /// records, at least one — and commit them. Answers whether chunks remain.
    ///
    /// A chunk re-appended since the plan is skipped (see `relocation_watch`).
    ///
    /// Grouped by source segment and moved with **coalesced reads + verbatim
    /// staging** (no decode / CRC / re-encode), so the fast block-read path is
    /// used rather than a syscall per record.
    pub(super) fn relocate_chunk_batch(
        &mut self,
        plan: &MaintenancePlan,
        run: &mut MaintenanceRun,
        budget_bytes: u64,
    ) -> Result<bool> {
        let rest = &plan.chunk_relocs[run.chunk_cursor..];
        let mut take = 0usize;
        let mut bytes = 0u64;
        for &(_, _, old) in rest {
            if take > 0 && bytes + old.record_size > budget_bytes {
                break;
            }
            bytes += old.record_size;
            take += 1;
        }
        let batch = &rest[..take];
        run.chunk_cursor += take;
        let keeps = |key: StreamKey, idx: u64| {
            self.relocation_watch
                .as_ref()
                .is_none_or(|w| w.keeps_chunk(key.stream_id.0, idx, key.writer))
        };
        let mut by_seg: BTreeMap<SegmentId, Vec<(StreamKey, u64, ChunkLoc)>> = BTreeMap::new();
        for &(key, idx, old) in batch {
            if keeps(key, idx) {
                by_seg.entry(old.segment).or_default().push((key, idx, old));
            }
        }
        for (source, recs) in by_seg {
            let items: Vec<RawReloc> = recs
                .iter()
                .map(|&(_, _, old)| RawReloc {
                    offset: old.offset,
                    record_size: old.record_size,
                })
                .collect();
            let new = self.relocate_raw_from_segment(source, &items)?;
            for ((sid, idx, old), at) in recs.into_iter().zip(new) {
                run.chunk_updates.push((
                    sid,
                    idx,
                    old,
                    ChunkLoc {
                        segment: at.segment,
                        offset: at.offset,
                        payload_len: old.payload_len,
                        record_size: at.size,
                        token_count: old.token_count,
                        format: old.format,
                    },
                ));
            }
        }
        // Durable per batch, so the op's closing commit — which the durability
        // barrier before the unlink needs — does not fsync the whole relocation
        // in one hold.
        self.commit()?;
        Ok(run.chunk_cursor < plan.chunk_relocs.len())
    }

    /// The last step of [`Self::execute_maintenance`]: the small record types,
    /// each behind its own supersession check. [`Self::relocate_tail_batch`]
    /// does the I/O one budget at a time; this runs that to the end under the
    /// caller's hold and hands back the index updates. The watch stays up
    /// until [`Self::finish_maintenance`] unlinks a source.
    ///
    /// # Panics
    ///
    /// When [`Self::relocate_batch`] has not run the op to the end: a resident
    /// record or chunk not yet carried would be unlinked with the sources.
    pub fn complete_maintenance(
        &mut self,
        plan: &MaintenancePlan,
        mut run: MaintenanceRun,
    ) -> Result<MaintenanceResult> {
        assert!(
            run.resident_cursor == plan.resident.len()
                && run.chunk_cursor == plan.chunk_relocs.len(),
            "a maintenance op was completed with batches still to run: \
             {}/{} resident records, {}/{} chunks",
            run.resident_cursor,
            plan.resident.len(),
            run.chunk_cursor,
            plan.chunk_relocs.len(),
        );
        while self.relocate_tail_batch(plan, &mut run, u64::MAX)? {}
        Ok(run.take_result())
    }

    /// Advance the tail — tokens, snapshots, branch checkpoints, singletons,
    /// characters, file events, blobs — by up to `budget_bytes`, and commit
    /// what moved. Answers whether any of the tail remains.
    ///
    /// This is the unit the caller takes the persistence lock for, the same
    /// bound as [`Self::relocate_batch`]. A seal, a cold load, and any engine
    /// caller waiting on this lock wait for one batch.
    pub fn relocate_tail_batch(
        &mut self,
        plan: &MaintenancePlan,
        run: &mut MaintenanceRun,
        budget_bytes: u64,
    ) -> Result<bool> {
        assert!(
            run.resident_cursor == plan.resident.len()
                && run.chunk_cursor == plan.chunk_relocs.len(),
            "a maintenance tail batch ran with relocation still to do: \
             {}/{} resident records, {}/{} chunks",
            run.resident_cursor,
            plan.resident.len(),
            run.chunk_cursor,
            plan.chunk_relocs.len(),
        );
        let mut moved = 0u64;
        let mut examined = 0usize;
        while run.tail_stage < TAIL_DONE
            && examined < TAIL_EXAMINE_CAP
            && (moved == 0 || moved < budget_bytes)
        {
            let room = budget_bytes.saturating_sub(moved).max(1);
            let (n, bytes, stage_done) = self.relocate_tail_stage(plan, run, room)?;
            examined += n;
            moved += bytes;
            if stage_done {
                run.tail_stage += 1;
                run.tail_cursor = 0;
            }
            // An empty stage falls through to the next one in this hold. A
            // stage that stopped mid-way (budget, or a run of skips) releases
            // the lock so a seal or a cold load gets in.
            if n == 0 && !stage_done {
                break;
            }
            if !stage_done && moved >= budget_bytes {
                break;
            }
        }
        if moved > 0 {
            self.commit()?;
        }
        Ok(run.tail_stage < TAIL_DONE)
    }

    pub(super) fn relocate_tail_stage(
        &mut self,
        plan: &MaintenancePlan,
        run: &mut MaintenanceRun,
        budget: u64,
    ) -> Result<(usize, u64, bool)> {
        match run.tail_stage {
            TAIL_TOKENS => {
                let (n, bytes, done, batch) = Self::take_budgeted(
                    &plan.token_relocs,
                    &mut run.tail_cursor,
                    budget,
                    |(_, old)| old.record_size,
                    |(key, _)| {
                        self.relocation_watch
                            .as_ref()
                            .is_none_or(|w| w.keeps_tokens(key.stream_id.0, key.writer))
                    },
                );
                self.extend_stream_updates(&batch, &mut run.token_updates)?;
                Ok((n, bytes, done))
            }
            TAIL_SNAPSHOTS => {
                let (n, bytes, done, batch) = Self::take_budgeted(
                    &plan.snapshot_relocs,
                    &mut run.tail_cursor,
                    budget,
                    |(_, old)| old.record_size,
                    |(sid, old)| self.snapshot_locs.get(&sid.0) == Some(old),
                );
                self.extend_stream_updates(&batch, &mut run.snapshot_updates)?;
                Ok((n, bytes, done))
            }
            TAIL_BRANCHES => {
                let (n, bytes, done, batch) = Self::take_budgeted(
                    &plan.branch_checkpoint_relocs,
                    &mut run.tail_cursor,
                    budget,
                    |(_, old)| old.record_size,
                    |(sid, old)| self.snapshot_locs.get(&sid.0) == Some(old),
                );
                self.extend_stream_updates(&batch, &mut run.branch_checkpoint_updates)?;
                Ok((n, bytes, done))
            }
            TAIL_SINGLETONS => self.take_singletons(plan, run, budget),
            TAIL_NPCS => self.take_npcs(plan, run, budget),
            TAIL_VFS => self.take_vfs(plan, run, budget),
            TAIL_BLOBS => self.take_blobs(plan, run, budget),
            TAIL_INDEX_PAGES => {
                let (n, bytes, done, batch) = Self::take_budgeted(
                    &plan.index_page_relocs,
                    &mut run.tail_cursor,
                    budget,
                    |(_, old)| old.record_size,
                    |(key, old)| {
                        self.metadata_locs.get(&metadata_key_parts(
                            RecordType::TurnIndexPage,
                            key.stream_id.0,
                            key.writer,
                        )) == Some(old)
                    },
                );
                self.extend_stream_updates(&batch, &mut run.index_page_updates)?;
                Ok((n, bytes, done))
            }
            _ => Ok((0, 0, true)),
        }
    }

    /// Walk `items` from `cursor`, keeping those `keep` still calls live, until
    /// `budget` bytes are kept or [`TAIL_EXAMINE_CAP`] entries have been
    /// looked at. The first kept record is taken even when it is larger than
    /// the budget. A run of superseded entries releases the lock at the cap
    /// instead of scanning the rest of the stage. Returns
    /// `(examined, bytes, stage_done, kept)`.
    pub(super) fn take_budgeted<T: Copy>(
        items: &[T],
        cursor: &mut usize,
        budget: u64,
        size_of: impl Fn(&T) -> u64,
        keep: impl Fn(&T) -> bool,
    ) -> (usize, u64, bool, Vec<T>) {
        let start = *cursor;
        let mut bytes = 0u64;
        let mut taken = Vec::new();
        while *cursor < items.len() && *cursor - start < TAIL_EXAMINE_CAP {
            let item = items[*cursor];
            if !keep(&item) {
                *cursor += 1;
                continue;
            }
            let size = size_of(&item);
            if !taken.is_empty() && bytes + size > budget {
                break;
            }
            *cursor += 1;
            bytes += size;
            taken.push(item);
            if bytes >= budget {
                break;
            }
        }
        let done = *cursor >= items.len();
        (*cursor - start, bytes, done, taken)
    }

    pub(super) fn extend_stream_updates<K: Copy>(
        &mut self,
        batch: &[(K, RecordLoc)],
        into: &mut Vec<(K, RecordLoc, RecordLoc)>,
    ) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        into.extend(self.relocate_stream_records(batch)?);
        Ok(())
    }

    /// `ModelKey`, `Template`, and `Tokenizer` — re-read and re-appended so
    /// `manifest.ingest` repoints them. A singleton rewritten since the plan
    /// is skipped: its newer copy is not on a target segment, and appending
    /// the planned one would install the stale bytes as the manifest's copy.
    pub(super) fn take_singletons(
        &mut self,
        plan: &MaintenancePlan,
        run: &mut MaintenanceRun,
        budget: u64,
    ) -> Result<(usize, u64, bool)> {
        let (n, bytes, done, batch) = Self::take_budgeted(
            &plan.singleton_relocs,
            &mut run.tail_cursor,
            budget,
            |(_, old)| old.record_size,
            |(rt, old)| self.singleton_loc(*rt) == Some(*old),
        );
        for &(rt, old) in &batch {
            let rec = self
                .segments
                .read_record_at(old.segment, old.offset, old.record_size)?;
            self.append_carried(rt, 0, 0, 0, 0, 0, &rec.payload, rec.header.identity())?;
        }
        Ok((n, bytes, done))
    }

    pub(super) fn singleton_loc(&self, rt: RecordType) -> Option<RecordLoc> {
        match rt {
            RecordType::ModelKey => self.manifest.model_key,
            RecordType::Template => self.manifest.template,
            RecordType::Tokenizer => self.manifest.tokenizer,
            _ => None,
        }
    }

    /// Characters, grouped by source. `append_raw_record` repoints `npc_locs`,
    /// so nothing rides back through [`MaintenanceResult`]. A character edited
    /// since the plan is skipped: relocating the stale copy would append it
    /// after the newer one, and the next reload would install the edit's
    /// predecessor.
    pub(super) fn take_npcs(
        &mut self,
        plan: &MaintenancePlan,
        run: &mut MaintenanceRun,
        budget: u64,
    ) -> Result<(usize, u64, bool)> {
        let (n, bytes, done, batch) = Self::take_budgeted(
            &plan.npc_relocs,
            &mut run.tail_cursor,
            budget,
            |(_, old)| old.record_size,
            |(sid, old)| self.npc_locs.get(&sid.0) == Some(old),
        );
        if !batch.is_empty() {
            let _ = self.relocate_stream_records(&batch)?;
        }
        Ok((n, bytes, done))
    }

    /// File events and tombstones still named by the index at the planned
    /// location. A tombstone written since the plan, or a conversation
    /// tombstoned since, drops the entry: relocating it would put a dead
    /// record after the record that killed it.
    pub(super) fn take_vfs(
        &mut self,
        plan: &MaintenancePlan,
        run: &mut MaintenanceRun,
        budget: u64,
    ) -> Result<(usize, u64, bool)> {
        let (n, bytes, done, batch) = Self::take_budgeted(
            &plan.vfs_relocs,
            &mut run.tail_cursor,
            budget,
            |r| r.loc.record_size,
            |r| self.vfs_still_planned(r),
        );
        let moved = self.relocate_raw_batch(
            &batch,
            |r| r.loc.segment,
            |r| RawReloc {
                offset: r.loc.offset,
                record_size: r.loc.record_size,
            },
        )?;
        for (r, at) in moved {
            let new = at.loc(r.loc.payload_len);
            if r.rt == RecordType::VfsEvent {
                self.vfs_index.repoint_event(r.timeline, r.id, r.loc, new);
            } else {
                self.vfs_index
                    .repoint_tombstone(r.timeline, r.id, r.loc, new);
            }
        }
        Ok((n, bytes, done))
    }

    pub(super) fn vfs_still_planned(&self, r: &VfsReloc) -> bool {
        self.vfs_index.timeline(r.timeline).is_some_and(|tl| {
            let at = if r.rt == RecordType::VfsEvent {
                tl.events()
            } else {
                tl.tombstones()
            };
            at.get(&r.id) == Some(&r.loc)
        })
    }

    /// Blobs still at the location the plan recorded.
    pub(super) fn take_blobs(
        &mut self,
        plan: &MaintenancePlan,
        run: &mut MaintenanceRun,
        budget: u64,
    ) -> Result<(usize, u64, bool)> {
        let (n, bytes, done, batch) = Self::take_budgeted(
            &plan.blob_relocs,
            &mut run.tail_cursor,
            budget,
            |(_, old)| old.record_size,
            |(key, old)| self.blob_index.get(*key) == Some(*old),
        );
        let moved = self.relocate_raw_batch(
            &batch,
            |(_, old)| old.segment,
            |(_, old)| RawReloc {
                offset: old.offset,
                record_size: old.record_size,
            },
        )?;
        for ((key, old), at) in moved {
            self.blob_index.repoint(key, old, at.loc(old.payload_len));
        }
        Ok((n, bytes, done))
    }

    /// Relocate `batch` verbatim, grouped by source segment, and return each
    /// item with where its copy landed.
    pub(super) fn relocate_raw_batch<T: Copy>(
        &mut self,
        batch: &[T],
        segment_of: impl Fn(&T) -> SegmentId,
        raw_of: impl Fn(&T) -> RawReloc,
    ) -> Result<Vec<(T, Appended)>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        let mut by_seg: BTreeMap<SegmentId, Vec<T>> = BTreeMap::new();
        for item in batch {
            by_seg.entry(segment_of(item)).or_default().push(*item);
        }
        let mut out = Vec::with_capacity(batch.len());
        for (source, recs) in by_seg {
            let items: Vec<RawReloc> = recs.iter().map(&raw_of).collect();
            let new = self.relocate_raw_from_segment(source, &items)?;
            for (item, at) in recs.into_iter().zip(new) {
                out.push((item, at));
            }
        }
        Ok(out)
    }

    /// Relocate a set of already-encoded records from `source` into the active
    /// via **coalesced stripe reads + verbatim staging**. Records are sorted by
    /// offset and adjacent ones read in a single I/O; each is staged byte-for-
    /// byte (its CRC and identity preserved — no re-encode). Returns where each
    /// item landed, in input order.
    pub(super) fn relocate_raw_from_segment(
        &mut self,
        source: SegmentId,
        items: &[RawReloc],
    ) -> Result<Vec<Appended>> {
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_unstable_by_key(|&i| items[i].offset);
        let spans: Vec<(u64, u64)> = order
            .iter()
            .map(|&k| (items[k].offset, items[k].record_size))
            .collect();
        let unplaced = Appended {
            segment: source,
            offset: 0,
            size: 0,
            identity: RecordIdentity::NONE,
        };
        let mut new_locs = vec![unplaced; items.len()];
        let mut buf: Vec<u8> = Vec::new();
        for (i, j) in stripes(&spans, MAX_STRIPE_BYTES) {
            let start = spans[i].0;
            let end = spans[j - 1].0 + spans[j - 1].1;
            let len = (end - start) as usize;
            if buf.len() < len {
                buf.resize(len, 0);
            }
            self.segments.read_into(source, start, &mut buf[..len])?;
            for &k in &order[i..j] {
                let it = &items[k];
                let within = (it.offset - start) as usize;
                let raw = &buf[within..within + it.record_size as usize];
                new_locs[k] = self.append_raw_record(raw)?;
            }
        }
        Ok(new_locs)
    }

    /// Relocate per-stream `RecordLoc`-shaped records (`Tokens` / `Snapshot` /
    /// `BranchCheckpoint`) verbatim into the active segment — grouped by source
    /// segment for coalesced stripe reads — returning `(stream, old_loc,
    /// new_loc)` per input record, the stream as the caller keyed it.
    pub(super) fn relocate_stream_records<K: Copy>(
        &mut self,
        relocs: &[(K, RecordLoc)],
    ) -> Result<Vec<(K, RecordLoc, RecordLoc)>> {
        let mut updates = Vec::with_capacity(relocs.len());
        let mut by_seg: BTreeMap<SegmentId, Vec<(K, RecordLoc)>> = BTreeMap::new();
        for &(sid, old) in relocs {
            by_seg.entry(old.segment).or_default().push((sid, old));
        }
        for (source, recs) in by_seg {
            let items: Vec<RawReloc> = recs
                .iter()
                .map(|&(_, old)| RawReloc {
                    offset: old.offset,
                    record_size: old.record_size,
                })
                .collect();
            let new = self.relocate_raw_from_segment(source, &items)?;
            for ((sid, old), at) in recs.into_iter().zip(new) {
                updates.push((sid, old, at.loc(old.payload_len)));
            }
        }
        Ok(updates)
    }

    /// The op's last step: take the drained source segments out of the log
    /// and its location indexes, and record the op for the status indicator.
    /// Runs on the log's owner, after the index apply. Safe because the index
    /// no longer points at any source (the apply repointed the live records;
    /// the rest were dead). The sources' files come back as
    /// [`RetiredSegments`], for the caller to unlink off the owner. A
    /// read-only handle refuses with [`PersistenceError::ReadOnly`].
    ///
    /// **A target holding a `stranded` record stays.** The index still names a
    /// record in it that the relocation never carried
    /// ([`MaintenanceResult::apply_to_substrate`]); dropped, the record would
    /// be indexed but unreadable. Each such record is logged with where the
    /// substrate and the persistence side each place it, and its segment is
    /// left linked, its relocated copies standing beside it, for a later pass.
    pub fn finish_maintenance(
        &mut self,
        plan: &MaintenancePlan,
        stranded: &Stranded,
    ) -> Result<RetiredSegments> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
        // The op ends here whether or not the unlink succeeds: nothing of it
        // runs on the owner after this call.
        self.end_relocation_watch();
        for record in stranded.records() {
            // The persistence side's map is the one the relocation filtered
            // by, so its entry beside the index's is the divergence.
            let persisted = match record.kind {
                StrandedKind::Snapshot | StrandedKind::BranchCheckpoint => self
                    .snapshot_locs
                    .get(&record.stream_id)
                    .map(|loc| (loc.segment.0, loc.offset)),
                StrandedKind::IndexPage => self
                    .metadata_locs
                    .get(&metadata_key_parts(
                        RecordType::TurnIndexPage,
                        record.stream_id,
                        record.writer,
                    ))
                    .map(|loc| (loc.segment.0, loc.offset)),
                StrandedKind::Chunk | StrandedKind::Tokens => None,
            };
            tracing::warn!(
                target: "candle_conversation::persistence::maintenance",
                kind = ?record.kind,
                stream_id = record.stream_id,
                writer = record.writer,
                substrate_segment = record.segment.0,
                substrate_offset = record.offset,
                persisted = ?persisted,
                "segment-maintenance: the index still names a record the relocation did not \
                 carry; its segment stays linked",
            );
        }
        let targets: Vec<SegmentId> = plan
            .op
            .targets()
            .into_iter()
            .filter(|t| !stranded.holds(*t))
            .collect();
        let mut retired = RetiredSegments::default();
        for &t in &targets {
            self.segments.retire_sealed(t, &mut retired)?;
        }
        // Drop metadata-location entries that pointed into the now-unlinked
        // segments. Live metadata whose current copy lived here was re-pointed
        // forward by the resident re-emission; any residue is a tombstoned
        // stream's metadata, which is genuinely gone with its segment. Leaving
        // stale entries would mis-attribute live bytes to a segment that no
        // longer exists.
        self.metadata_locs
            .retain(|_, loc| !targets.contains(&loc.segment));
        // The same for the payload-keyed resident records: their live copies
        // were re-emitted forward, so an entry still pointing into a dropped
        // segment names bytes that are gone.
        self.resident_locs
            .retain(|_, loc| !targets.contains(&loc.segment));
        // And for a record found by its identity: one still located in a
        // dropped segment was not carried forward, so this member no longer
        // holds it.
        self.locations.forget_segments(&targets);
        // Keep what this op stranded for the next choice to step around, and
        // let go of the records of segments no longer in the log.
        self.stranded.extend(stranded);
        let sealed: HashSet<SegmentId> = self.segments.sealed_ids().iter().copied().collect();
        self.stranded.retain_segments(|s| sealed.contains(&s));
        let unix = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.last_maintenance = Some((plan.op.label(), unix));
        tracing::trace!(op = ?plan.op, "substrate maintenance op applied");
        Ok(retired)
    }
}
