//! Keeps each sequence's recurrent (DeltaNet) state in step with its KV, since
//! that state has no per-session storage of its own.
//!
//! The state is owned by the model and outlives every session, so it is reset
//! whenever a sequence stands at offset zero. Without that, a new session, a
//! fork or a truncation would leave the last conversation's state in place, and
//! the model would answer as though it remembered a conversation the prompt
//! never had. A wave takes each sequence's state out of the map, runs, and puts
//! it back on success and on failure alike. A fork copies state for a branch,
//! and a snapshot is held until its state is first wanted, so a waiting fork
//! holds no device memory.

use super::*;

impl HybridBatched {
    /// Ensure every sequence in `seqs` has recurrent state consistent with how
    /// many tokens its KV holds.
    ///
    /// **A sequence standing at offset 0 has no history, so its recurrent state
    /// must be the sequence-start value** — and that is a reset, not merely a
    /// create. The paged KV is owned by the session and disappears with it;
    /// this map is owned by the *model* and outlives every session, because a
    /// DeltaNet layer has no per-session storage to put it in. So a new
    /// session, a fork, or a truncation back to nothing all hand a sequence id
    /// back with an empty cache while the entry here still holds whatever the
    /// last conversation left in it. Keying the reset off the offset ties the
    /// two together: the recurrence follows the KV, which is the thing the
    /// scheduler actually manages.
    ///
    /// Missed, this is close to invisible. The cache is empty so attention is
    /// correct, the shapes all match, nothing errors — the model simply
    /// answers as though it remembers a conversation the prompt never had.
    ///
    /// A sequence whose span this forward is longer than [`TIMELINE_ROWS`] —
    /// a prefill chunk — also gets its prefill scratch here, between forwards,
    /// where the claim is legal: the span writes its advanced state there and
    /// the commit exchanges it in.
    pub fn ensure_recurrent(
        &self,
        seqs: &[usize],
        offsets: &[usize],
        q_lens: &[usize],
    ) -> Result<()> {
        if seqs.len() != offsets.len() || seqs.len() != q_lens.len() {
            candle::bail!(
                "ensure_recurrent: {} sequences against {} offsets and {} span lengths",
                seqs.len(),
                offsets.len(),
                q_lens.len()
            );
        }
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        for ((&seq, &offset), &q_len) in seqs.iter().zip(offsets).zip(q_lens) {
            // A snapshot held for this sequence becomes its store before the
            // wave reads it. Built seeded, so the offset-0 rule below keeps it.
            if let Entry::Vacant(slot) = map.entry(seq) {
                if let Some(store) = self.build_staged(seq)? {
                    slot.insert(store);
                }
            }
            match map.entry(seq) {
                Entry::Vacant(slot) => {
                    slot.insert(RecurrentStateStore::new(
                        &self.model.cfg.layer_kinds,
                        &self.model.cfg.delta_net,
                        &self.model.device,
                    )?);
                }
                Entry::Occupied(mut slot) => {
                    // The seeded flag is consumed on the FIRST wave either
                    // way. It protects the one wave that follows a fork or a
                    // restore — the wave whose slot legitimately stands at
                    // offset 0 while holding state that was put there on
                    // purpose. A flag that outlived that wave would go on to
                    // suppress a later, genuine reset, which is the recycled
                    // slot defect wearing the fix's own clothes.
                    let seeded = slot.get_mut().take_seeded();
                    // Zeroed on the slots the store holds, never rebuilt: a
                    // dropped store's slots return only once nothing views
                    // them, so a rebuild claims a second store's worth while
                    // the first is still held.
                    if offset == 0 && !seeded {
                        slot.get_mut().reset()?;
                    }
                }
            }
            if q_len > TIMELINE_ROWS {
                map.get_mut(&seq).expect("ensured above").ensure_scratch()?;
            }
        }
        Ok(())
    }

    /// Drop a sequence's recurrent state.
    ///
    /// The scheduler releases KV slots when a sequence ends; the recurrent
    /// half has to be released with it or the map grows for the life of the
    /// process.
    pub fn release_recurrent(&self, seq: usize) -> Result<()> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        map.remove(&seq);
        self.staged_map()?.remove(&seq);
        // An armed block names rows of the state that just went away, and the
        // draft head's history belongs to the same sequence.
        self.armed_map()?.remove(&seq);
        self.release_draft(seq);
        Ok(())
    }

    /// Give `child` a copy of `parent`'s recurrent state.
    ///
    /// Reservation bytes recurrent state holds on this model's device — every state
    /// arena's regions, whatever is in them.
    ///
    /// The arenas and not a sum over the stores: the stores share arenas, so their
    /// sum leaves out every arena's free slots and unused tail, and any slot a
    /// handle kept after its store was dropped. See
    /// [`RecurrentStateStore::arena_reserved_bytes`].
    pub fn recurrent_reserved_bytes(&self) -> usize {
        RecurrentStateStore::arena_reserved_bytes(&self.model.device)
    }

    /// Compact the state arenas every sequence's recurrent state lives in — see
    /// [`compact_stores`]. Between forwards; holds the map for the pass, which is
    /// what keeps a wave from opening on a store mid-move.
    pub fn compact_recurrent(&self, max_moves: usize) -> Result<RecurrentCompaction> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        compact_stores(
            map.values_mut(),
            &self.model.cfg.delta_net,
            &self.model.device,
            max_moves,
        )
    }

    /// What one sequence's state costs, whether or not one is standing.
    ///
    /// **Priced from the geometry, never from residency.** Every store this
    /// model builds has the same shape, so the config answers for all of them —
    /// and it answers at the one moment residency cannot, which is the moment
    /// admission actually asks. See
    /// [`RecurrentStateStore::reserved_bytes_for`] for why a mean over the live
    /// stores is not a substitute: it divides a sum over every store the
    /// process holds by a count of what is merely in flight, so it climbs with
    /// the number of *idle* conversations.
    ///
    /// **A sequence arrives prefilling**, so the price is its store and its
    /// prefill scratch — one more slot per DeltaNet layer, held until the
    /// sequence decodes (`docs/deltanet_timeline.md` §7). A decoding sequence
    /// holds [`RecurrentStateStore::reserved_bytes_for`] alone.
    ///
    /// This takes no lock, which is the other half of its value here —
    /// admission asks on the scheduler thread while forwards hold the map.
    pub fn recurrent_store_bytes(&self) -> usize {
        let (kinds, dims) = (&self.model.cfg.layer_kinds, &self.model.cfg.delta_net);
        RecurrentStateStore::reserved_bytes_for(kinds, dims)
            + RecurrentStateStore::scratch_bytes_for(kinds, dims)
    }

    /// The turn loop carves a child slot per turn and decodes on it, borrowing
    /// the parent's KV blocks zero-copy. State cannot be borrowed the same way
    /// — the child advances it — so it is copied device-to-device
    /// ([`RecurrentStateStore::fork_from`]), and the copy is seeded so the
    /// child's first wave does not reset it back to zero.
    ///
    /// A parent whose snapshot is still held has no store to copy, and the
    /// child is held the same snapshot rather than a store built from it: its
    /// store is built when its state is first wanted, at its first wave, so a
    /// fork waiting for its turn holds no device ground. The parent keeps
    /// holding its snapshot for its next view — one store per turn in flight,
    /// never one idling on the parent beside it.
    ///
    /// Errors when the parent carries no state: a fork of nothing is a caller
    /// bug, and returning quietly would hand the child zeros — which is the
    /// defect this whole path exists to remove, reintroduced as an error path.
    pub fn fork_recurrent(&self, parent: usize, child: usize) -> Result<()> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        match map.get(&parent) {
            Some(store) => {
                let forked = store.fork_from()?;
                map.insert(child, forked);
            }
            None => {
                let mut staged = self.staged_map()?;
                let held = staged.get(&parent).cloned().ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "qwen35: fork_recurrent from sequence {parent}, which carries no \
                         recurrent state — handing {child} zeros here is the amnesia this \
                         path exists to prevent"
                    ))
                })?;
                staged.insert(child, held);
            }
        }
        Ok(())
    }

    /// Move `child`'s state onto `parent` — the linear join at `finalize_view`.
    ///
    /// A move, so the child's entry is gone afterwards and the parent's old
    /// state is dropped. That is correct precisely because a view is a linear
    /// continuation: it entered with a copy of the parent's state and advanced
    /// it over the turn's tokens, so what it holds now is what the parent's
    /// state becomes.
    ///
    /// A child still holding only the snapshot it forked never ran a wave, so
    /// it advanced nothing: its snapshot goes and the parent keeps what it has.
    ///
    /// Errors when the child carries none, for the same reason as
    /// [`Self::fork_recurrent`]: silently leaving the parent's stale state in
    /// place would lose the turn without saying so.
    pub fn move_recurrent(&self, child: usize, parent: usize) -> Result<()> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        if !map.contains_key(&child) && self.staged_map()?.remove(&child).is_some() {
            return Ok(());
        }
        let store = map.remove(&child).ok_or_else(|| {
            candle::Error::Msg(format!(
                "qwen35: move_recurrent from sequence {child}, which carries no \
                 recurrent state — the turn's decode would be silently lost"
            ))
        })?;
        map.insert(parent, store);
        // The parent's state is the view's continuation now, so a snapshot it
        // was still holding describes a point it has moved past.
        self.staged_map()?.remove(&parent);
        Ok(())
    }

    /// Read a sequence's state back as the snapshot record's layer rows.
    ///
    /// `None` when the sequence carries no state — a slot that has never run a
    /// wave has nothing worth persisting, and writing a zero snapshot would be
    /// worse than writing none: resume would install it and report success.
    ///
    /// A snapshot is taken at a turn's seal, which ends any prefill the
    /// sequence was in, so its prefill scratch goes back here: the next long
    /// span claims it again.
    pub fn export_recurrent(&self, seq: usize) -> Result<Option<(u64, Vec<ExportedLayerState>)>> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        let Some(store) = map.get_mut(&seq) else {
            // A held snapshot is this sequence's state as surely as a store
            // would be: read it back as it stands, without building one.
            return Ok(self
                .staged_map()?
                .get(&seq)
                .map(|s| (s.schedule_hash, s.layers.to_vec())));
        };
        // `export` refuses mid-wave itself; the seal runs outside the wave, so
        // reaching that error means the ordering broke, not that the caller
        // needs a retry.
        let layers = store.export()?;
        store.release_scratch();
        Ok(Some((store.schedule_hash(), layers)))
    }

    /// Scatter a snapshot into a sequence's state — the resume path.
    ///
    /// Creates the store if the slot has none yet (the normal case: resume runs
    /// at `create_sequence`, before any wave). `import` validates the schedule
    /// hash and every layer's geometry before touching a tensor, and marks the
    /// store seeded so the first wave's `offset == 0` reset does not undo it.
    pub fn restore_recurrent(
        &self,
        seq: usize,
        schedule_hash: u64,
        layers: &[ExportedLayerState],
    ) -> Result<()> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        // A restored state supersedes any snapshot held back for this sequence.
        self.staged_map()?.remove(&seq);
        let store = match map.entry(seq) {
            Entry::Occupied(slot) => slot.into_mut(),
            Entry::Vacant(slot) => slot.insert(RecurrentStateStore::new(
                &self.model.cfg.layer_kinds,
                &self.model.cfg.delta_net,
                &self.model.device,
            )?),
        };
        store.import(schedule_hash, layers)
    }

    /// Hold a snapshot for `seq` until its state is first wanted — see the
    /// `staged` field. Checked against this model's geometry now, so a foreign
    /// layout is refused here exactly as [`Self::restore_recurrent`] refuses it.
    ///
    /// A sequence that already holds a store takes the snapshot at once: held
    /// back, the store's older state would be what its next wave read.
    pub fn stage_recurrent(
        &self,
        seq: usize,
        schedule_hash: u64,
        layers: Arc<[ExportedLayerState]>,
    ) -> Result<()> {
        RecurrentStateStore::check_snapshot(
            &self.model.cfg.layer_kinds,
            &self.model.cfg.delta_net,
            schedule_hash,
            &layers,
        )?;
        if self.has_recurrent(seq)? {
            return self.restore_recurrent(seq, schedule_hash, &layers);
        }
        self.staged_map()?.insert(
            seq,
            StagedSnapshot {
                schedule_hash,
                layers,
            },
        );
        Ok(())
    }

    /// The held-snapshot map, locked.
    pub(super) fn staged_map(&self) -> Result<MutexGuard<'_, HashMap<usize, StagedSnapshot>>> {
        self.staged
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: staged snapshot lock poisoned".into()))
    }

    /// A store built from the snapshot held for `seq`, if one is held; the
    /// snapshot is consumed only when the build succeeds, so a build the span
    /// refuses can be retried.
    pub(super) fn build_staged(&self, seq: usize) -> Result<Option<RecurrentStateStore>> {
        let mut staged = self.staged_map()?;
        let Some(snap) = staged.get(&seq) else {
            return Ok(None);
        };
        let store = snap.build(
            &self.model.cfg.layer_kinds,
            &self.model.cfg.delta_net,
            &self.model.device,
        )?;
        staged.remove(&seq);
        Ok(Some(store))
    }

    /// Whether a sequence currently carries recurrent state — for the
    /// scheduler's fork/move wiring, which must not call either on a slot the
    /// model has never seen (a view carved before the parent's first wave).
    /// A held snapshot is not a store: see [`Self::holds_recurrent`].
    pub fn has_recurrent(&self, seq: usize) -> Result<bool> {
        Ok(self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?
            .contains_key(&seq))
    }

    /// Whether a sequence has recurrent state to fork — a store, or a snapshot
    /// held until one is wanted.
    pub fn holds_recurrent(&self, seq: usize) -> Result<bool> {
        Ok(self.has_recurrent(seq)? || self.staged_map()?.contains_key(&seq))
    }

    /// How many sequences currently carry recurrent state.
    pub fn recurrent_len(&self) -> Result<usize> {
        Ok(self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?
            .len())
    }

    /// Open a wave over `seqs`' recurrent state.
    ///
    /// The wave-atomicity contract: a wave writes timelines and the prefill
    /// scratch, never a live state, so every wave that begins must either commit
    /// or roll back and a rolled-back one leaves no trace.
    pub fn begin_recurrent_wave(&self, seqs: &[usize]) -> Result<()> {
        self.for_each_store(seqs, |s| s.begin_wave())
    }

    /// Close the wave over `seqs` ([`commit_stores`]): fold every short span it
    /// ran for a sequence outside the armed cohort — all of its rows, in ONE
    /// launch over every such sequence and layer — then commit. The armed
    /// cohort's layers stay pending for the reconcile, which alone knows how many
    /// rows each keeps. On an error nothing is committed and the caller rolls the
    /// wave back.
    pub fn commit_recurrent_wave(&self, seqs: &[usize]) -> Result<()> {
        let armed: Vec<usize> = self.armed_map()?.keys().copied().collect();
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        if let Some(seq) = seqs.iter().find(|s| !map.contains_key(s)) {
            candle::bail!(
                "qwen35: sequence {seq} has no recurrent state — \
                 `ensure_recurrent` was not called for this wave"
            );
        }
        let mut stores: Vec<(&mut RecurrentStateStore, bool)> = map
            .iter_mut()
            .filter(|(seq, _)| seqs.contains(seq))
            .map(|(seq, store)| (store, !armed.contains(seq)))
            .collect();
        commit_stores(&mut stores)
    }

    pub fn rollback_recurrent_wave(&self, seqs: &[usize]) -> Result<()> {
        self.for_each_store(seqs, |s| s.rollback_wave())
    }

    /// Lift the wave's sequences' state out of the map, in `seqs` order.
    ///
    /// The layer sweep mixes every sequence against its own state in one pass,
    /// so it needs `&mut` to all of them at once — which a `HashMap` cannot
    /// hand out. Taking them out and putting them back is the borrow-checkable
    /// form of that, and it is why [`Self::put_recurrent`] must run on the
    /// failure path too: a sequence whose state stayed lifted would be missing
    /// from the map for every later wave.
    pub fn take_recurrent(&self, seqs: &[usize]) -> Result<Vec<RecurrentStateStore>> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        let mut out = Vec::with_capacity(seqs.len());
        for &seq in seqs {
            out.push(map.remove(&seq).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "qwen35: sequence {seq} has no recurrent state — \
                     `ensure_recurrent` was not called for this wave"
                ))
            })?);
        }
        Ok(out)
    }

    /// Put lifted state back. Runs on both the success and the failure path.
    pub fn put_recurrent(&self, seqs: &[usize], stores: Vec<RecurrentStateStore>) -> Result<()> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        for (&seq, store) in seqs.iter().zip(stores) {
            map.insert(seq, store);
        }
        Ok(())
    }

    pub(super) fn for_each_store(
        &self,
        seqs: &[usize],
        mut f: impl FnMut(&mut RecurrentStateStore) -> Result<()>,
    ) -> Result<()> {
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        for &seq in seqs {
            let store = map.get_mut(&seq).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "qwen35: sequence {seq} has no recurrent state — \
                     `ensure_recurrent` was not called for this wave"
                ))
            })?;
            f(store)?;
        }
        Ok(())
    }
}
