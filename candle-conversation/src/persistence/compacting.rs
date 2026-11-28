//! Reclaims dead weight from the log by rewriting only its live records, so the log's size tracks what is still referenced.
//!
//! Compaction is triggered by a dead-byte ratio taken from the same per-segment liveness that background maintenance uses, since the incremental counter climbs without bound under maintenance. A rewrite reads one version of the store and then re-applies the edits published meanwhile, so a shared root keeps every edit made during the rewrite. The rewrite logs the record classes it carries against those the store held at open, because a class that is dropped raises nothing on its own. An inherited log is never compacted by a child that only reads it.

use super::*;

impl SubstratePersistence {
    /// How long a timeline tombstone is kept before a compaction may shrink it
    /// to its deleted-set entry — the member's `--tombstone-retention`. Only
    /// when its bytes are reclaimed depends on it; the timeline stays dead.
    pub fn set_tombstone_retention(&mut self, retention: Duration) {
        self.tombstone_retention = retention;
    }

    /// The last background-maintenance op applied — `(label, unix_secs)` — or
    /// `None` if none has run this session. Surfaced in the daemon status.
    pub fn last_maintenance(&self) -> Option<(String, u64)> {
        self.last_maintenance.clone()
    }

    /// Whether the active log has accumulated enough dead weight to justify
    /// a compaction pass — the dead-byte ratio (§5.8) crossing
    /// [`COMPACTION_DEAD_RATIO_THRESHOLD`] on a log at least
    /// [`COMPACTION_MIN_LOG_BYTES`] long.
    ///
    /// Cheap enough to poll every persistence-thread pass: the dead-byte
    /// counter is maintained O(1) per append (see [`accounting`]), and the
    /// tombstoned-stream sum is a walk of the in-RAM stream index — no
    /// disk I/O.
    pub fn should_compact(&self, substrate: &Substrate) -> bool {
        let total = self.segments.total_record_bytes().unwrap_or(0);
        total >= COMPACTION_MIN_LOG_BYTES
            && self.dead_ratio(substrate) >= COMPACTION_DEAD_RATIO_THRESHOLD
    }

    /// Fraction of the store's record bytes that are dead weight, computed from
    /// the SAME per-segment liveness the background maintenance uses: `dead =
    /// Σ(segment_total − segment_live)` over the whole log. This is a real
    /// fraction in `[0, 1]` — it can never exceed 100%, and it drops as
    /// maintenance reclaims dead segments.
    ///
    /// It deliberately does NOT use the incremental `accounting.dead_bytes()`
    /// counter: that counter is a whole-log cumulative total reset only by the
    /// global `compact()` (a load-time / manual path), so under the daemon's
    /// per-segment maintenance it climbs without bound — every relocation append
    /// supersedes the moved-from copy and adds to it — and the ratio sails past
    /// 100% into meaningless territory. Liveness-derived is scope-consistent
    /// (numerator and denominator span the same segments) and self-correcting.
    pub fn dead_ratio(&self, substrate: &Substrate) -> f32 {
        let total = self.segments.total_record_bytes().unwrap_or(0);
        if total == 0 {
            return 0.0;
        }
        let live: u64 = self.segment_liveness(substrate).values().sum();
        self.note_dead_ratio(total, live)
    }

    /// Record the dead ratio of `total` record bytes of which `live` are live,
    /// and return it — what [`Self::scanned_dead_ratio`] answers until the next
    /// liveness scan, this one's or maintenance's.
    pub(crate) fn note_dead_ratio(&self, total: u64, live: u64) -> f32 {
        let ratio = dead_ratio_of(total, live);
        self.facts.set_dead_ratio(ratio);
        ratio
    }

    /// The dead ratio the last liveness scan found — maintenance's plan or a
    /// [`Self::dead_ratio`] — without a scan of its own; `None` before the first.
    ///
    /// For a viewer: the scan walks the whole substrate under its read lock,
    /// up to 0.86 s on BlackWorld (2026-10-29), and the substrate page asked
    /// for it on every refresh — a hold every writer, the scheduler's among
    /// them, waited out. Maintenance scans every pass anyway.
    pub fn scanned_dead_ratio(&self) -> Option<f32> {
        self.facts.dead_ratio()
    }

    /// What this log publishes for any thread to read without the owner.
    pub fn facts(&self) -> LogFacts {
        self.facts.clone()
    }

    /// Compact the store — reclaim dead weight by rewriting only the **live**
    /// records into a fresh single segment, then dropping every prior segment.
    ///
    /// Quiesces in-flight writes, collects the live winners across every
    /// segment (routing each read to the segment holding it), writes them to
    /// a compacted scratch file, and adopts it as the sole active segment
    /// ([`SegmentedLog::adopt_compacted`]). The in-RAM manifest, substrate
    /// index, and metadata caches are rebuilt from the compacted segment.
    /// Inherited logs are untouched — a child never compacts a base it only
    /// reads (§13.5). A read-only handle refuses with
    /// [`PersistenceError::ReadOnly`].
    ///
    /// `progress` is told the records written of the live set — `(0, 0)` while
    /// that set is still being collected.
    ///
    /// For a substrate no other thread edits; [`Self::compact_shared`] is the
    /// same rewrite against a root other threads edit while it runs.
    pub fn compact(
        &mut self,
        substrate: &mut Substrate,
        progress: Option<&dyn Fn(usize, usize)>,
    ) -> Result<()> {
        let rewritten = self.rewrite(substrate, progress, None)?;
        rewritten.reapply(substrate);
        self.refuse_untaken(&rewritten, &rewritten.untaken(substrate))?;
        rewritten.settle(substrate);
        self.unlink_rewritten(&rewritten)
    }

    /// [`Self::compact`] against a shared root: the rewrite reads one version,
    /// and its re-apply is a run of ordinary edits ([`Rewritten::batches`]), so
    /// every edit another thread publishes meanwhile survives it. The old
    /// segments are unlinked once the last edit is published.
    pub fn compact_shared(
        &mut self,
        substrate: &SubstrateLock,
        progress: Option<&(dyn Fn(usize, usize) + Sync)>,
    ) -> Result<()> {
        let read = substrate.read();
        let progress = progress.map(|p| p as &dyn Fn(usize, usize));
        let rewritten = self.rewrite(&read, progress, None)?;
        drop(read);
        for batch in rewritten.batches() {
            substrate.update(|s| {
                for entry in batch {
                    s.apply_walker_entry(entry);
                }
            });
        }
        let untaken = rewritten.untaken(&substrate.read());
        self.refuse_untaken(&rewritten, &untaken)?;
        substrate.update(|s| rewritten.settle(s));
        self.unlink_rewritten(&rewritten)
    }

    /// `Ok` when the re-apply took back everything `rewritten` carried.
    /// Otherwise the segments it replaced are kept as sealed segments — the
    /// untaken records are located there and nowhere else — and the
    /// compaction ends without settling or unlinking anything, naming what it
    /// could not take back.
    pub(super) fn refuse_untaken(
        &mut self,
        rewritten: &Rewritten,
        untaken: &[Untaken],
    ) -> Result<()> {
        if untaken.is_empty() {
            return Ok(());
        }
        self.segments.keep_replaced(rewritten.old_segments());
        let chunks: usize = untaken.iter().map(|u| u.carried - u.taken).sum();
        tracing::error!(
            target: "candle_conversation::persistence::census",
            dir = %self.dir().display(),
            streams = untaken.len(),
            chunks,
            first = ?&untaken[..untaken.len().min(8)],
            "a compaction's re-apply did not take back every chunk it carried — the replaced \
             segments are kept, nothing was forgotten, and the store reads as it did before"
        );
        Err(PersistenceError::RewriteUntaken {
            streams: untaken.len(),
            chunks,
        })
    }

    /// Unlink the segments `rewritten` replaced: the substrate names none of
    /// them any more.
    pub(super) fn unlink_rewritten(&mut self, rewritten: &Rewritten) -> Result<()> {
        self.segments.unlink_replaced(rewritten.old_segments())
    }

    /// Adopt a store written before identities (§6.2, [`adoption`]): one
    /// compaction stamps every record that carries none as this node's own.
    /// Run by the first writable open of such a store, which then opens it
    /// again; a no-op on a store that needs none. `progress` counts the
    /// records rewritten, as [`Self::compact`]'s does.
    pub(super) fn adopt(
        &mut self,
        substrate: &mut Substrate,
        progress: &dyn Fn(usize, usize),
    ) -> Result<()> {
        if !self.predates_identities() {
            return Ok(());
        }
        tracing::info!(
            target: "candle_conversation::persistence::identity",
            node = format!("{:016x}", self.node()),
            dir = %self.dir().display(),
            unstamped = %self.unstamped.summary(),
            "this store was written before records carried an identity — rewriting it once, \
             with every record stamped as this node's own"
        );
        let mut adoption = Adoption::new(self.minter.clone());
        let rewritten = self.rewrite(substrate, Some(progress), Some(&mut adoption))?;
        // The caller opens the adopted store again into a fresh substrate, so
        // this one is not re-applied.
        self.unlink_rewritten(&rewritten)?;
        self.unstamped = RecordCensus::new();
        Ok(())
    }

    /// The log rewrite behind [`Self::compact`], stamping as it writes when
    /// `adoption` is given. Reads `substrate` and changes nothing in it: what
    /// the substrate has to do about the rewrite comes back as a [`Rewritten`].
    pub(super) fn rewrite(
        &mut self,
        substrate: &Substrate,
        progress: Option<&dyn Fn(usize, usize)>,
        adoption: Option<&mut Adoption>,
    ) -> Result<Rewritten> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
        // Records written of the live set, for the loading screen: `(0, 0)`
        // until the set is collected.
        let report = |done: usize, total: usize| {
            if let Some(p) = progress {
                p(done, total);
            }
        };
        report(0, 0);

        // 1. Quiesce — every staged write is now durable on disk.
        self.segments.commit()?;

        // 2. Collect the live winners, in dependency order — sourced from
        //    substrate state (per-entity) + manifest singletons. Each record
        //    is read from the segment that physically holds it.
        let dir = self.segments.dir().to_path_buf();
        // Planning only — no disk reads; each read-back record carries its
        // source location, read back coalesced + staged verbatim in step 3.
        let expiry = TombstoneExpiry {
            now_ms: now_ms(),
            retention: self.tombstone_retention,
        };
        // What this compaction drops: the substrate forgets it once the
        // rewritten log is walked back in, or maintenance would carry it again.
        let expired = expiry.expired(substrate, &self.marker_ids);
        let live = compaction::collect_live_records(
            &self.manifest,
            substrate,
            &self.npc_locs,
            &self.vfs_index,
            &self.blob_index,
            &self.marker_ids,
            self.node(),
            expiry,
            &Housekeeping {
                replica: self.replica_record(),
                rebases: self.rebases.clone(),
            },
        );
        // What this compaction is carrying forward, by type, against what the
        // store held when it opened. A class present at open and absent here is
        // being deleted — the failure this whole module exists to make visible,
        // and one that raises nothing on its own. One pass over a Vec already in
        // RAM, once per compaction.
        {
            let mut carried = RecordCensus::new();
            for item in &live {
                carried.record(item.header().record_type);
            }
            let lost = carried.types_lost_against(&self.open_census);
            if lost.is_empty() {
                tracing::info!(
                    target: "candle_conversation::persistence::census",
                    total = carried.total(),
                    "compaction carrying forward: {}",
                    carried.summary()
                );
            } else {
                // `warn`, and naming the classes: every type the store held has
                // a carrier in `survival.rs`, so a type going to zero here is
                // either a genuine emptying (a tombstoned timeline's last turn)
                // or a rewrite that forgot one. Both are worth a line that says
                // which.
                tracing::warn!(
                    target: "candle_conversation::persistence::census",
                    total = carried.total(),
                    dropped = ?lost,
                    "compaction is carrying NO records of {} type(s) the store held at open \
                     — verify each is a legitimate emptying and not a missing carrier \
                     (persistence/survival.rs): {}",
                    lost.len(),
                    carried.summary()
                );
            }
        }
        let total = live.len();
        report(0, total);

        // 3. Write the live set into a compacted scratch file (a fresh index
        //    chain interleaved), reading read-back records off the source
        //    segments in coalesced stripes, then adopt it as the sole active
        //    segment, dropping every prior segment.
        let scratch = dir.join(".compact");
        let (new_log, new_manifest) = {
            let segments = &mut self.segments;
            compaction::write_compacted_log(
                &scratch,
                &live,
                &mut |seg, off, dest| segments.read_into(seg, off, dest),
                adoption,
                &mut |done| report(done, total),
            )?
        };
        let old_segments = self.segments.adopt_compacted(new_log, &scratch)?;

        // 4. Adopt the compacted manifest and refresh the metadata caches.
        //    `write_compacted_log` stamped every singleton with `FIRST_SEGMENT`,
        //    but `adopt_compacted` names the compacted segment with a fresh id
        //    (crash safety) and wrote ALL records into that one segment — so
        //    re-stamp the singletons at the actual active id before reading them.
        self.manifest = new_manifest;
        let adopted = self.segments.active_id();
        for loc in [
            &mut self.manifest.model_key,
            &mut self.manifest.template,
            &mut self.manifest.tokenizer,
        ]
        .into_iter()
        .flatten()
        {
            loc.segment = adopted;
        }
        self.model_record = self
            .manifest
            .model_key
            .map(|loc| {
                self.segments
                    .read_record_at(loc.segment, loc.offset, loc.record_size)
                    .map(|r| r.payload)
            })
            .transpose()?;
        self.template = self
            .manifest
            .template
            .map(|loc| {
                self.segments
                    .read_record_at(loc.segment, loc.offset, loc.record_size)
                    .map(|r| r.payload)
            })
            .transpose()?;
        self.tokenizer_sha256 = self
            .manifest
            .tokenizer
            .map(|loc| {
                let r = self
                    .segments
                    .read_record_at(loc.segment, loc.offset, loc.record_size)?;
                Ok::<_, PersistenceError>(sha256(&r.payload))
            })
            .transpose()?;
        // 5. The substrate still holds offsets into the OLD segments. The walk
        //    of the compacted segment below is kept for the caller to re-apply
        //    ([`Rewritten`]), which moves each value to its new offset. The
        //    dead-weight accounting restarts from the fresh segment in the same
        //    pass (a just-compacted store is all-live).
        self.accounting.reset();
        // Metadata and snapshot locations point at the OLD segments; rebuild
        // them from the freshly-compacted segment in the same recovery pass
        // (mirrors the load walk in `from_dir_with_sink`).
        self.metadata_locs.clear();
        self.resident_locs.clear();
        self.snapshot_locs.clear();
        self.npc_locs.clear();
        self.vfs_index.clear();
        self.blob_index.clear();
        // The marker identities are rebuilt the same way: every carried marker
        // kept its own, so the rebuilt table holds what the old one did for
        // every marker still live, and nothing for one the compaction dropped.
        self.marker_ids = MarkerIds::new();
        // Every record moved, so every location is rebuilt; the applied set
        // only grows, and the rewrite carried it whole.
        self.locations.clear();
        let node = self.node();
        let accounting = &mut self.accounting;
        let metadata_locs = &mut self.metadata_locs;
        let resident_locs = &mut self.resident_locs;
        let snapshot_locs = &mut self.snapshot_locs;
        let npc_locs = &mut self.npc_locs;
        let vfs_index = &mut self.vfs_index;
        let blob_index = &mut self.blob_index;
        let marker_ids = &mut self.marker_ids;
        let replica = &mut self.replica;
        let locations = &mut self.locations;
        let rebases = &self.rebases;
        // The carried record's own entry is `own_written` as it stands, which
        // this handle already holds.
        let mut own_kept = None;
        let mut walked: Vec<WalkEntry> = Vec::with_capacity(total);
        let (last_index, tail_digests) = self.segments.recover_active_with_sink(|entry| {
            let fixed = fix_walked(entry, rebases);
            let entry = fixed.as_ref();
            observe_replica(replica, locations, &mut own_kept, node, entry);
            let h = &entry.record.header;
            marker_ids.record(
                h.record_type,
                h.stream_id,
                &entry.record.payload,
                h.identity(),
            );
            accounting.record(&entry.record.header, &entry.record.payload, entry.size);
            record_metadata_loc(metadata_locs, entry);
            record_resident_loc(resident_locs, entry);
            record_snapshot_loc(snapshot_locs, entry);
            record_npc_loc(npc_locs, entry);
            vfs::record_vfs_loc(vfs_index, accounting, entry);
            blob::record_blob_loc(blob_index, entry);
            walked.push(Rewritten::keep(entry));
        })?;
        // The compacted segment carries a fresh index chain; chain the next
        // flush onto it.
        self.last_index = last_index;
        self.pending_index = tail_digests;
        report(total, total);
        Ok(Rewritten::new(walked, old_segments, substrate, expired))
    }
}
