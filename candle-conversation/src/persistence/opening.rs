//! Opens the persistence layer: finds the segment set, recovers the manifest and substrate from it, and refuses a store the model cannot open.
//!
//! The walk that builds the manifest also drives every record through the substrate and an optional sink, so the substrate is populated in one pass. A writable open takes the directory's writer lock before it reads a byte, so a second writer is refused before anything changes. A store written before identities is adopted once, after the sink has seen it, and a refused store is left as it was found. A read-only open never creates or writes anything under the segment directory, and beside a live writer a segment may vanish mid-open, in which case the open starts over empty.

use super::*;

impl SubstratePersistence {
    /// Open the persistence layer at `<cwd>/substrate/substrate.log`,
    /// creating the directory and file if absent and recovering the
    /// manifest if present.
    pub fn open() -> Result<SubstratePersistence> {
        let cwd = std::env::current_dir()?;
        SubstratePersistence::open_in(&cwd)
    }

    /// Open the persistence layer at `<dir>/substrate/` (the segment set).
    pub fn open_in(dir: &Path) -> Result<SubstratePersistence> {
        Self::from_dir_with_sink(&dir.join(SUBSTRATE_DIR), &[], false, Purge::Nothing, |_| {})
    }

    /// Open the persistence layer and drive every record through
    /// [`Substrate::apply_walker_entry`] in the same pass that builds
    /// the manifest's singleton offsets.  The substrate ends up fully
    /// populated (streams, labels, conv_states, tree metadata, debug
    /// ids) by the time this returns.
    ///
    /// Replaces the legacy two-phase `open_in` + `reconstruct →
    /// mirror_from_manifest` pattern with a single walker pass.  No
    /// per-entity state is mirrored from manifest to substrate
    /// afterwards because the manifest's per-entity fields are
    /// `#[serde(skip)]` and the substrate is the authoritative
    /// in-RAM index.
    ///
    /// A store written before identities is adopted here, once ([`adoption`]).
    /// Opened for no model ([`Admit::Unbound`]): a host that runs one binds
    /// through [`SharedSubstrate::open_in`].
    pub fn open_in_with_substrate(
        dir: &Path,
        substrate: &mut Substrate,
    ) -> Result<SubstratePersistence> {
        Self::open_adopting(
            &dir.join(SUBSTRATE_DIR),
            substrate,
            Admit::Unbound,
            |_| {},
            &|_, _, _| {},
        )
    }

    /// The writable open with a substrate: walk the store into `substrate`
    /// and `sink`, refuse it when `admit`'s model cannot open it, and when it
    /// was written before identities, adopt it and open it again into a fresh
    /// substrate. The refusal comes before the adoption, so a refused store is
    /// left as it was found. `sink` sees the walk once: the adoption changes
    /// record headers and the payloads that name a writer — turn tombstones and
    /// file events — and leaves every other payload as it was.
    ///
    /// `progress` is told the stage it is in and how far through it the open
    /// is: MiB of log walked against the store's size for a walk, records
    /// rewritten of the live set for the adoption ([`OpenStage::unit`]).
    ///
    /// A member away too long ([`joined`]) walks without what it can no longer
    /// trust — `sink` never sees it — and compacts it away before it returns.
    pub(super) fn open_adopting<F>(
        dir: &Path,
        substrate: &mut Substrate,
        admit: Admit<'_>,
        mut sink: F,
        progress: OpenProgress<'_>,
    ) -> Result<Self>
    where
        F: FnMut(&WalkEntry),
    {
        let last_joined = joined::read(dir)?;
        let purge = Purge::after(last_joined, now_ms());
        let total = store_mib(dir);
        progress(OpenStage::Reading, 0, total);
        let mut meter = MibMeter::default();
        let mut sp = Self::from_dir_with_sink(dir, &[], false, purge, |entry| {
            substrate.apply_walker_entry(entry);
            sink(entry);
            if let Some(mib) = meter.add(entry.size) {
                progress(OpenStage::Reading, mib.min(total), total);
            }
        })?;
        progress(OpenStage::Reading, total, total);
        // Before anything below rewrites the log, so a rewrite does not carry
        // a blob its tombstone retired.
        sp.forget_retired_blobs(substrate);
        sp.admit(admit)?;
        if sp.predates_identities() {
            sp.adopt(substrate, &|done, total| {
                progress(OpenStage::Stamping, done, total)
            })?;
            drop(sp);
            *substrate = Substrate::new();
            let total = store_mib(dir);
            progress(OpenStage::Rereading, 0, total);
            let mut meter = MibMeter::default();
            sp = Self::from_dir_with_sink(dir, &[], false, Purge::Nothing, |entry| {
                substrate.apply_walker_entry(entry);
                if let Some(mib) = meter.add(entry.size) {
                    progress(OpenStage::Rereading, mib.min(total), total);
                }
            })?;
            progress(OpenStage::Rereading, total, total);
            sp.forget_retired_blobs(substrate);
        } else if purge.drops_any() {
            sp.compact(
                substrate,
                Some(&|done, total| progress(OpenStage::Purging, done, total)),
            )?;
        }
        // Judged now: a store never judged starts its clock here, and one just
        // purged starts it again rather than purging at every open until the
        // replicator next marks it joined.
        if last_joined.is_none() || purge.drops_any() {
            joined::write(dir, now_ms())?;
        }
        // The substrate locates what it does not hold; this is what reads it.
        substrate
            .payloads()
            .set_reader(Arc::new(sp.sealed_reader()));
        Ok(sp)
    }

    /// Whether the open met records written before identities.
    pub(super) fn predates_identities(&self) -> bool {
        self.unstamped.total() > 0
    }

    /// As [`Self::open_in_with_substrate`], but **read-only**: the segment set
    /// is opened with [`SegmentedLog::open_read_only_with_sink`], so nothing
    /// under `<dir>/substrate/` is ever created, renamed, deleted, truncated,
    /// grown or written through this handle, and the store must already exist.
    ///
    /// Every durable operation keeps its signature and answers without
    /// touching disk: the appends refuse with [`PersistenceError::ReadOnly`]
    /// and stage nothing; `commit`, `commit_if_pending` and `seal_active`
    /// succeed with nothing to do; `set_template` reports that nothing was
    /// written; `bind_model` and `set_tokenizer` still refuse a model or
    /// vocabulary other than the recorded one but never record one;
    /// maintenance plans no op and compaction refuses.
    pub fn open_in_with_substrate_read_only(
        dir: &Path,
        substrate: &mut Substrate,
    ) -> Result<SubstratePersistence> {
        Self::open_store_read_only(&dir.join(SUBSTRATE_DIR), substrate)
    }

    /// [`Self::open_in_with_substrate_read_only`] on the store in `dir`
    /// itself — the segment directory, wherever it is, rather than a
    /// workspace's `substrate/`: a copy set aside beside the live store opens
    /// as it lies.
    pub fn open_store_read_only(
        dir: &Path,
        substrate: &mut Substrate,
    ) -> Result<SubstratePersistence> {
        // Beside a live writer a segment can be compacted away mid-open; the
        // open starts over from an empty substrate (`vanished_segment`).
        let dir = dir.to_path_buf();
        let substrate = RefCell::new(substrate);
        let mut sp = retry_vanished(
            || {
                Self::from_dir_with_sink(&dir, &[], true, Purge::Nothing, |entry| {
                    substrate.borrow_mut().apply_walker_entry(entry)
                })
            },
            || **substrate.borrow_mut() = Substrate::new(),
        )?;
        sp.forget_retired_blobs(&substrate.borrow());
        // The substrate locates what it does not hold; this is what reads it.
        substrate
            .borrow()
            .payloads()
            .set_reader(Arc::new(sp.sealed_reader()));
        Ok(sp)
    }

    /// As [`Self::open_in_with_substrate`], but the caller also sees every
    /// record as it is walked.
    ///
    /// For record classes the substrate does not interpret — [`RecordType::Npc`]
    /// is the one that exists today — this is how their owner rebuilds its
    /// index without a second pass over the log. Reading them back through
    /// [`Self::npc_locs`] afterwards would work too, but costs one seek per
    /// character where this costs none: the bytes are already in hand.
    ///
    /// The sink sees records in log order, so last-writer-wins is simply the
    /// last value the sink is handed for a key. A store `admit`'s model cannot
    /// open is refused; one written before identities is adopted here, once,
    /// after the sink has seen it ([`Self::open_adopting`]).
    pub fn open_in_with_substrate_and_sink<F>(
        dir: &Path,
        substrate: &mut Substrate,
        admit: Admit<'_>,
        sink: F,
        progress: OpenProgress<'_>,
    ) -> Result<SubstratePersistence>
    where
        F: FnMut(&WalkEntry),
    {
        Self::open_adopting(&dir.join(SUBSTRATE_DIR), substrate, admit, sink, progress)
    }

    /// Open over an ordered list of paths. The last entry is the active,
    /// writable **segment directory** (`substrate/`); every earlier entry is
    /// an inherited read-only single-file log, loaded through the shared
    /// cache (§13.5).
    pub fn open_concat(logs: &[PathBuf]) -> Result<SubstratePersistence> {
        let (active_dir, inherited) = logs.split_last().ok_or_else(|| {
            PersistenceError::Corrupt("open_concat needs at least one log".into())
        })?;
        Self::from_dir_with_sink(active_dir, inherited, false, Purge::Nothing, |_| {})
    }

    /// Open the segment set in `dir` (the `substrate/` directory) with the
    /// listed inherited single-file logs, driving every recovered record
    /// through `sink` in the same pass that builds the manifest and the
    /// dead-weight accounting. `read_only` opens the segment set with
    /// [`SegmentedLog::open_read_only_with_sink`] instead of
    /// [`SegmentedLog::open_with_sink`].
    ///
    /// A record `purge` drops is walked past as if it were not there — but for
    /// the clock and the `oseq` counter, which it still advances, so this node
    /// never mints an identity it issued before. The replica then forgets every
    /// other origin, and this node's own dropped `oseq`s are gaps in what it
    /// has written, for a catch-up to fill ([`joined`]).
    pub(super) fn from_dir_with_sink<F>(
        dir: &Path,
        inherited: &[PathBuf],
        read_only: bool,
        purge: Purge,
        mut sink: F,
    ) -> Result<SubstratePersistence>
    where
        F: FnMut(&walker::WalkEntry),
    {
        // One writer per substrate: taken before a byte is read, so a refused
        // open costs nothing. A read-only handle takes none.
        let writer_lock = if read_only {
            None
        } else {
            Some(WriterLock::acquire(dir)?)
        };
        // The node, once the directory is ours: a read-only handle takes the
        // one the directory names, if any, and mints nothing.
        let (node, node_source) = if read_only {
            (read_node(dir)?.unwrap_or(0), None)
        } else {
            let (node, source) = load_or_mint(dir)?;
            if let NodeSource::Reminted { previous } = source {
                tracing::warn!(
                    target: "candle_conversation::persistence::identity",
                    previous = format!("{previous:016x}"),
                    node = format!("{node:016x}"),
                    dir = %dir.display(),
                    "this substrate was minted in another directory or on another machine — \
                     it writes under a fresh node from here on, so it can never issue a \
                     record its original also issues"
                );
            }
            (node, Some(source))
        };
        let mut origination = Origination::new(node);
        let mut own_sections = OwnSections::default();
        let mut replica = ReplicaPayload::default();
        let mut locations = LocationIndex::default();
        let mut marker_ids = MarkerIds::new();
        let mut inherited_subs = Vec::with_capacity(inherited.len());
        for path in inherited {
            inherited_subs.push(InheritedSubstrate::load(path)?);
        }

        // Recover every segment (ascending id, active last) — feeding the
        // dead-weight accounting and the caller's sink in the same pass that
        // builds the combined singleton manifest.
        let mut accounting = RecordAccounting::new();
        let mut metadata_locs = StampedLocs::new();
        let mut resident_locs = StampedLocs::new();
        let mut snapshot_locs = StampedLocs::new();
        let mut npc_locs = StampedLocs::new();
        let mut vfs_index = VfsIndex::new();
        let mut blob_index = BlobIndex::new();
        // What the store actually holds, by type. One array increment per record
        // in a walk that already visits every record — see `RecordCensus`.
        let mut census = RecordCensus::new();
        // What this node wrote cut off carries the provisional generation until
        // a rejoin fixed it, and the `Rebase` record that says so is written
        // after the records it fixes. So those records are held back until the
        // walk has read every `Rebase`, then applied with the generation fixed
        // for them — nothing reaches the substrate, or any index, unfixed.
        let rebases = RefCell::new(RebaseMap::default());
        let held: RefCell<Vec<WalkEntry>> = RefCell::new(Vec::new());
        let draining = Cell::new(false);
        let mut unstamped = RecordCensus::new();
        let mut own_kept = None;
        let mut dropped_own = OriginApplied::default();
        let mut dropped_foreign = 0u64;
        let mut walk = |entry: &WalkEntry| {
            let h = &entry.record.header;
            if adoption::predates_identities(h) {
                unstamped.record(h.record_type);
            }
            if h.record_type == RecordType::Rebase {
                if let Ok(map) = RebaseMap::decode(&entry.record.payload) {
                    rebases.borrow_mut().merge(&map);
                }
            }
            if h.gen == GEN_DETACHED && h.origin == node && !draining.get() {
                held.borrow_mut().push(entry.clone());
                return;
            }
            let fixed = fix_walked(entry, &rebases.borrow());
            let entry = fixed.as_ref();
            let h = &entry.record.header;
            let id = h.identity();
            origination.observe(&id);
            if purge.drops(node, &id) {
                if id.origin == node {
                    dropped_own.insert(id.oseq);
                } else {
                    dropped_foreign += 1;
                }
                return;
            }
            own_sections.observe(node, h, &entry.record.payload);
            observe_replica(&mut replica, &mut locations, &mut own_kept, node, entry);
            marker_ids.record(
                h.record_type,
                h.stream_id,
                &entry.record.payload,
                h.identity(),
            );
            accounting.record(&entry.record.header, &entry.record.payload, entry.size);
            census.record(entry.record.header.record_type);
            record_metadata_loc(&mut metadata_locs, entry);
            record_resident_loc(&mut resident_locs, entry);
            record_snapshot_loc(&mut snapshot_locs, entry);
            record_npc_loc(&mut npc_locs, entry);
            vfs::record_vfs_loc(&mut vfs_index, &mut accounting, entry);
            blob::record_blob_loc(&mut blob_index, entry);
            sink(entry);
        };
        let opened = if read_only {
            SegmentedLog::open_read_only_with_sink(dir, &mut walk)?
        } else {
            SegmentedLog::open_with_sink(dir, &mut walk)?
        };
        draining.set(true);
        let held = held.take();
        // A detached record no `Rebase` covers was written since this node was
        // last cut off and has not rejoined: it is cut off still, from the
        // first such record.
        let still_detached = held
            .iter()
            .map(|e| rebases.borrow().fix(e.record.header.identity()))
            .filter(|id| id.gen == GEN_DETACHED)
            .map(|id| id.oseq)
            .min();
        for entry in &held {
            walk(entry);
        }
        let rebases = rebases.into_inner();
        if still_detached.is_some() {
            origination.detach(still_detached);
        }
        // Every own record the store holds may be sent, but what it wrote cut
        // off and no rejoin fixed (§6.3, §6.7), and what a purge dropped and no
        // catch-up has returned yet — the gaps the replica's own entry kept,
        // and the ones this open just made (§7.8).
        let mut own_missing = dropped_own;
        if let Some(kept) = own_kept {
            for (start, end) in kept.gaps() {
                own_missing.insert_range(start, end);
            }
        }
        if purge.drops_any() {
            replica = ReplicaPayload::default();
            tracing::warn!(
                target: "candle_conversation::persistence::joined",
                node = format!("{node:016x}"),
                dir = %dir.display(),
                ?purge,
                dropped_foreign,
                own_missing = own_missing.ranges().iter().map(|(s, e)| e - s + 1).sum::<u64>(),
                "this member was away longer than a tombstone may be missed — dropping what \
                 it can no longer trust; a catch-up refetches what the cluster still holds"
            );
        }
        let own_top = still_detached.unwrap_or(origination.next_oseq()) - 1;
        let own_written = OriginApplied::from_top_and_gaps(
            own_top,
            own_missing
                .ranges()
                .into_iter()
                .filter(|(start, _)| *start <= own_top)
                .map(|(start, end)| (start, end.min(own_top))),
        );
        let segmented_log::OpenedSegments {
            mut segments,
            manifest,
            last_index,
            tail_digests,
            recovered_records,
        } = opened;
        // The census at open is the "before" of every before/after: compare two
        // restarts and a class that a rewrite dropped in between reads as a
        // count that went to zero.
        tracing::info!(
            target: "candle_conversation::persistence::census",
            total = census.total(),
            "substrate census at open: {}",
            census.summary()
        );

        let model_record = manifest
            .model_key
            .map(|loc| {
                segments
                    .read_record_at(loc.segment, loc.offset, loc.record_size)
                    .map(|r| r.payload)
            })
            .transpose()?;
        let template = manifest
            .template
            .map(|loc| {
                segments
                    .read_record_at(loc.segment, loc.offset, loc.record_size)
                    .map(|r| r.payload)
            })
            .transpose()?;
        let tokenizer_sha256 = manifest
            .tokenizer
            .map(|loc| {
                let r = segments.read_record_at(loc.segment, loc.offset, loc.record_size)?;
                Ok::<_, PersistenceError>(sha256(&r.payload))
            })
            .transpose()?;

        let mut sp = SubstratePersistence {
            segments,
            manifest,
            inherited: inherited_subs,
            model_record,
            tokenizer_sha256,
            template,
            accounting,
            pending_index: tail_digests,
            last_index,
            recovered_records,
            last_maintenance: None,
            relocation_watch: None,
            stranded: Stranded::default(),
            metadata_locs,
            resident_locs,
            snapshot_locs,
            npc_locs,
            vfs_index,
            blob_index,
            facts: LogFacts::new(census.held_mask()),
            open_census: census,
            writer_lock,
            minter: Minter::with_sections(origination, own_sections.live()),
            node_source,
            marker_ids,
            tombstone_retention: DEFAULT_TOMBSTONE_RETENTION,
            replica_dirty: (release_known_deferred(&mut replica) || purge.drops_any())
                && !read_only,
            replica,
            locations,
            applied_mix: AppliedMix::default(),
            rebases,
            unstamped,
            tap: None,
            open_top: own_written.top(),
            own_hold: OwnHold::default(),
            own_unwritten: OriginApplied::default(),
            own_written,
        };
        // Self-heal a large un-indexed tail (a crash window, or a log
        // that predates the index chain entirely): flush it now so the
        // next open takes the chain path instead of re-walking it. A
        // read-only handle leaves the tail as it found it — the next
        // writable open heals it.
        if !read_only && sp.pending_index.len() >= INDEX_FLUSH_ENTRIES {
            sp.flush_header_index()?;
        }
        Ok(sp)
    }

    /// The active log's manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// All inherited (read-only) substrates, in registration order.
    /// Each carries its own manifest.  Exposed so the substrate-side
    /// reload can mirror per-stream state across all inherited logs
    /// without each call site re-deriving the list.
    pub fn inherited_substrates(&self) -> &[Arc<InheritedSubstrate>] {
        &self.inherited
    }

    /// Whether this handle refuses writes — opened read-only
    /// ([`Self::open_in_with_substrate_read_only`]), or released
    /// ([`Self::release_writer`]).
    pub fn is_read_only(&self) -> bool {
        self.segments.is_read_only()
    }

    /// Number of inherited logs.
    pub fn inherited_count(&self) -> usize {
        self.inherited.len()
    }

    /// Every character's current record location, keyed by `npc_id`.
    ///
    /// The `substrate/` directory this store's segments live in.
    pub fn dir(&self) -> &Path {
        self.segments.dir()
    }
}
