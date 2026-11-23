//! The GDN recurrent state each sequence carries, and how it is claimed, forked, moved, parked and written to a snapshot.
//!
//! A view carve copies all three carried classes together, because a child with the parent's recurrence but an empty index would attend over blocks its selector never scored. The state is claimed at admission so a refusal is an admission refusal, and a store is parked once its last seal is durable, so an idle conversation holds no device ground. A parked state is refused rather than replaced by a sequence-start store, which would answer fluently and remember nothing.

use super::*;

/// The carried per-sequence state, and what persisting it costs.
///
/// This model carries four classes outside the paged K/V, and they divide on
/// one question — **is it cheaper to store or to rebuild?**
///
/// * **GDN recurrence** — an accumulated sum with no per-token decomposition.
///   Not derivable from anything; must be stored. [`RecurrentStateStore`]
///   already exports and imports it, geometry-checked against a schedule hash,
///   and is shared with the hybrid lineage.
/// * **PLE** — a `(conv_kernel − 1) × ngram_size` row convolution history plus
///   the last `ngram_size − 1` token ids. Kilobytes: stored, in the snapshot's
///   opaque auxiliary blob.
/// * **QSA index** — `n_blocks × head_dim` keys per attention layer, which at
///   128K tokens and ratio 4 is ~16 MiB a layer and ~192 MiB a sequence. That
///   is not a per-turn snapshot. The keys are the indexer's own projection of
///   each layer's hidden states, not a function of K, so they cannot be rebuilt
///   from resident K/V: the live tail is stored in the seal's auxiliary blob
///   (`seal_live`, restored by [`Self::restore_aux_state`]) and the injected
///   pages ahead of it come back with the projection that injects their K/V. An
///   index dropped any other way costs a re-prefill of its context.
/// * **Draft seeds** — one residual row per sequence, and only meaningful
///   inside the wave that produced it. Neither stored nor rebuilt: a resumed
///   sequence simply drafts nothing until its first wave seeds it, which costs
///   one step of speculation and no correctness.
impl Qwen4ExpBatched {
    /// Whether `seq` carries any of the recurrent classes yet.
    ///
    /// The fork and move sites consult this rather than erroring, because a
    /// view can legitimately be carved before its parent has ever run a wave —
    /// a brand-new conversation's first turn does exactly that, and there the
    /// child correctly starts from the sequence-start value.
    pub fn has_recurrent(&self, seq: usize) -> Result<bool> {
        Ok(self
            .recurrent
            .read()
            .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?
            .contains_key(&seq))
    }

    /// How many sequences currently carry recurrent state.
    /// The sequences holding a recurrent store, sorted.
    pub fn recurrent_holders(&self) -> Vec<usize> {
        let mut seqs: Vec<usize> = self
            .recurrent
            .read()
            .map(|r| r.keys().copied().collect())
            .unwrap_or_default();
        seqs.sort_unstable();
        seqs
    }

    pub fn recurrent_len(&self) -> Result<usize> {
        Ok(self
            .recurrent
            .read()
            .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?
            .len())
    }

    /// A view carve: `child` begins as an independent copy of `parent`'s
    /// carried state.
    ///
    /// **All three copied classes move together or none does.** The K/V the
    /// child borrows is one history; a child holding the parent's GDN state but
    /// an empty index would attend over blocks its selector never scored, and
    /// one holding the index but a zero PLE would inject an n-gram embedding
    /// computed from a window it never saw. Both read as a plausible answer.
    pub fn fork_recurrent(&self, parent: usize, child: usize) -> Result<()> {
        self.refuse_parked(parent)?;
        // The child's state is the parent's from here, not sequence-start.
        self.clear_pristine(child);
        // **Each carried class is forked on its own terms.** They are populated
        // by different things and a parent can hold one without the others:
        // the recurrent store comes from running a wave, the PLE state from the
        // same, but the QSA index also comes from INJECTION — a projection
        // installs pages into a slot that has never decoded a token.
        //
        // This used to `return Ok(())` when the parent had no recurrent store,
        // on the reasoning that a parent which has not run a wave has nothing to
        // copy. True of the recurrence, false of the index, and the early return
        // took all three out together. The base conversation is exactly that
        // shape — an Arc-injected prefix of sections, never decoded — so every
        // conversation forked from it inherited a slot holding the base's K/V
        // and none of the index describing it. Silent until the prefix passed
        // the QSA identity threshold, at which point every first turn failed
        // with `a query at position N needs M blocks but the index cache holds
        // 0 in injected pages`.
        // A parent whose snapshot is still held has no store to copy, and the
        // child takes the snapshot held, not a store built from it: the store is
        // claimed at the child's admission (`claim_carried_state` installs a
        // held snapshot), so a fork waiting to be admitted holds no device
        // ground. An ephemeral fork can wait a long time — it is taken at its
        // parent's turn boundary and queued like any turn — and building its
        // store here put ~288 MiB per fork on the card before it was admitted.
        // The parent keeps holding its snapshot for its next view: one store
        // per turn in flight, never a second one idling on the parent.
        let inherited = {
            let map = self
                .recurrent
                .read()
                .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?;
            match map.get(&parent) {
                Some(store) => Some(Inherited::Store(store.fork_from()?)),
                None => self
                    .staged
                    .read()
                    .map_err(|_| candle::Error::Msg("qwen4exp: staged lock poisoned".into()))?
                    .get(&parent)
                    .cloned()
                    .map(Inherited::Staged),
            }
        };
        let ple = {
            let map = self
                .ple
                .read()
                .map_err(|_| candle::Error::Msg("qwen4exp: ple lock poisoned".into()))?;
            map.get(&parent).map(PleState::snapshot)
        };
        let index = {
            let map = self
                .index
                .read()
                .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
            match map.get(&parent) {
                Some(caches) => Some(
                    caches
                        .iter()
                        .map(IndexCache::fork)
                        .collect::<Result<Vec<_>>>()?,
                ),
                None => None,
            }
        };
        // Whether the parent has ever run a wave, read before its state is moved
        // into the child: it decides what a missing index means below.
        let parent_ran = inherited.is_some();
        match inherited {
            Some(Inherited::Store(store)) => {
                self.recurrent
                    .write()
                    .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?
                    .insert(child, store);
            }
            Some(Inherited::Staged(snapshot)) => {
                self.staged
                    .write()
                    .map_err(|_| candle::Error::Msg("qwen4exp: staged lock poisoned".into()))?
                    .insert(child, snapshot);
            }
            None => {}
        }
        if let Some(p) = ple {
            self.ple
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: ple lock poisoned".into()))?
                .insert(child, p);
        }
        match index {
            Some(i) => {
                // What the child actually inherited, per layer. A carve that
                // silently hands over nothing looks identical downstream to a
                // model that keeps no index at all, and the difference decides
                // whether to look at the fork or at the parent's prefix.
                let inherited: Vec<usize> = i
                    .iter()
                    .zip(self.attention_ratios())
                    .filter(|(_, r)| *r > 0)
                    .map(|(c, r)| c.indexed_tokens(r))
                    .collect();
                tracing::trace!(
                    target: "candle_conversation::scheduler::reproject",
                    parent,
                    child,
                    layers = i.len(),
                    inherited_min = inherited.iter().copied().min().unwrap_or(0),
                    inherited_max = inherited.iter().copied().max().unwrap_or(0),
                    "qwen4exp: forked index caches to the view",
                );
                self.index
                    .write()
                    .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?
                    .insert(child, i);
            }
            // The parent had no caches at all. After a parent has run a wave —
            // it holds recurrent state — that is a loss: the child starts empty
            // and every token the view borrows is unindexed from birth. A parent
            // that has never run one is a fresh slot with nothing to inherit;
            // the tree summariser's scratch turn is carved from exactly that,
            // on whatever id a finalized view just freed.
            None if parent_ran => tracing::warn!(
                parent,
                child,
                "qwen4exp: the view's parent holds no index caches, so the carve \
                 inherits none — the borrowed K/V is unindexed from the start",
            ),
            None => {}
        }
        // The view counts into the parent's hits: the parent's projection
        // declared the pieces, the view's rows are the reads (`super::hits`).
        {
            let mut hits = self
                .hits
                .write()
                .map_err(|_| candle::Error::Msg("hits lock poisoned".into()))?;
            fork_holder(&mut hits, parent, child)?;
        }
        self.mark_seeded(child)?;
        Ok(())
    }

    /// Claim `seq`'s carried state now — the recurrent store, the PLE history
    /// and its spare, the draft head's seeds — whatever of it the sequence does
    /// not already hold.
    ///
    /// **What admission calls, so a refusal is an admission refusal.** Claimed
    /// lazily at wave entry instead, the store came after the weight side had
    /// taken spare ground back and after the wave's own K/V, and a refusal there
    /// failed every member of the cohort; claimed here, between forwards, the
    /// weight side can sell for it, and a refusal leaves the turn queued. What
    /// this creates is sequence-start state, so it is marked [`Self::pristine`]
    /// and the first wave keeps it. A store a fork already copied is left alone,
    /// and a snapshot held for the sequence becomes its store here.
    pub fn claim_carried_state(&self, seq: usize) -> Result<()> {
        let t = Instant::now();
        self.refuse_parked(seq)?;
        ClaimPart::Parked.charge(t.elapsed());
        let t = Instant::now();
        self.install_staged(seq)?;
        ClaimPart::Install.charge(t.elapsed());
        let cfg = &self.model.cfg;
        let hc_dim = cfg.hc.count * cfg.hidden_size;
        let dev = &self.model.device;
        let mut created = false;
        let t = Instant::now();
        {
            let mut rec = self
                .recurrent
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?;
            if let Entry::Vacant(e) = rec.entry(seq) {
                e.insert(RecurrentStateStore::new(
                    &cfg.layer_kinds,
                    &cfg.delta_net,
                    dev,
                )?);
                created = true;
            }
        }
        ClaimPart::Recurrent.charge(t.elapsed());
        let t = Instant::now();
        {
            let mut ple = self
                .ple
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: ple lock poisoned".into()))?;
            if let Entry::Vacant(e) = ple.entry(seq) {
                e.insert(PleState::zeros(&cfg.ple, hc_dim, dev)?);
            }
            ple.get_mut(&seq).expect("inserted above").ensure_spare()?;
        }
        ClaimPart::Ple.charge(t.elapsed());
        let t = Instant::now();
        if self.model.mtp.is_some() {
            let dims = (1, cfg.hc.count, cfg.hidden_size);
            let mut seeds = self
                .seeds
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: seed lock poisoned".into()))?;
            if let Entry::Vacant(e) = seeds.entry(seq) {
                e.insert(state_buffer(dev, dims, true)?);
            }
            let mut spares = self
                .seed_spares
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: seed lock poisoned".into()))?;
            if let Entry::Vacant(e) = spares.entry(seq) {
                // Written by the carry before it is read (invariant 6).
                e.insert(state_buffer(dev, dims, false)?);
            }
        }
        ClaimPart::Seeds.charge(t.elapsed());
        if created {
            self.pristine
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: pristine lock poisoned".into()))?
                .insert(seq);
        }
        Ok(())
    }

    /// Forget that `seq`'s carried state is pristine: something wrote it, or
    /// it is going away.
    pub(super) fn clear_pristine(&self, seq: usize) -> bool {
        self.pristine
            .write()
            .map(|mut p| p.remove(&seq))
            .unwrap_or(false)
    }

    /// Declare `seq`'s carried state deliberately installed, so the next wave
    /// does not reset it out from under the caller. See the `seeded` field.
    pub(in super::super) fn mark_seeded(&self, seq: usize) -> Result<()> {
        self.seeded
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: seeded lock poisoned".into()))?
            .insert(seq);
        Ok(())
    }

    /// A view finalizes: `child`'s state becomes `parent`'s.
    ///
    /// A move, not a merge — a view is a linear continuation of its parent, so
    /// what it holds now is what the parent's state becomes. The child's
    /// entries are gone afterwards and the parent's previous ones are dropped.
    pub fn move_recurrent(&self, child: usize, parent: usize) -> Result<()> {
        self.refuse_parked(child)?;
        // Its counting moves with it, whatever state it carries: a
        // reprojection's fresh view counts into its conversation's counters
        // again (`super::hits::move_holder`).
        {
            let mut hits = self
                .hits
                .write()
                .map_err(|_| candle::Error::Msg("hits lock poisoned".into()))?;
            move_holder(&mut hits, child, parent)?;
        }
        // A view claimed its state at admission and never ran a wave: what it
        // holds is sequence-start zeros, not a continuation, and moving it would
        // wipe the parent's. It goes; the parent keeps what it had — the same
        // answer as a view that never claimed any.
        if self.clear_pristine(child) {
            self.recurrent
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?
                .remove(&child);
            self.ple
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: ple lock poisoned".into()))?
                .remove(&child);
            self.seeds
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: seeds lock poisoned".into()))?
                .remove(&child);
            self.seed_spares
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: seeds lock poisoned".into()))?
                .remove(&child);
            return Ok(());
        }
        let t_lock = Instant::now();
        // One write: take the child's store and put it under the parent. The
        // parent's displaced store leaves the lock with it and is dropped
        // outside, its slots going back to their arenas without holding the map
        // every wave reads. The parent's stale staged snapshot, removed below,
        // returns its host rows to `host_rows` when it is the last holder.
        let displaced = {
            let mut map = self
                .recurrent
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?;
            let lock_us = t_lock.elapsed().as_micros() as u64;
            let store = match map.remove(&child) {
                Some(s) => s,
                // The view never ran a wave; the parent keeps what it had.
                None => return Ok(()),
            };
            (map.insert(parent, store), lock_us)
        };
        let (displaced, lock_us) = displaced;
        let t_drop = Instant::now();
        let displaced_store = displaced.is_some();
        drop(displaced);
        let drop_us = t_drop.elapsed().as_micros() as u64;
        self.unpark(parent);
        // The parent's state is the view's continuation now, so a snapshot it
        // was still holding describes a point it has moved past.
        let t_staged = Instant::now();
        let stale = self
            .staged
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: staged lock poisoned".into()))?
            .remove(&parent);
        drop(stale);
        let staged_us = t_staged.elapsed().as_micros() as u64;
        note("recurrent move split", || {
            format!(
                "child={child} parent={parent} lock_us={lock_us} drop_us={drop_us} \
                 displaced={displaced_store} staged_us={staged_us}"
            )
        });
        {
            let mut map = self
                .ple
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: ple lock poisoned".into()))?;
            if let Some(p) = map.remove(&child) {
                map.insert(parent, p);
            }
        }
        {
            let mut map = self
                .index
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
            if let Some(i) = map.remove(&child) {
                map.insert(parent, i);
            }
        }
        {
            let mut map = self
                .seeds
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: seeds lock poisoned".into()))?;
            if let Some(s) = map.remove(&child) {
                map.insert(parent, s);
            }
        }
        {
            // The spare follows its seed, so the parent's pair stays two
            // distinct buffers; a parent whose seed was not replaced keeps its
            // own spare.
            let mut map = self
                .seed_spares
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: seeds lock poisoned".into()))?;
            if let Some(s) = map.remove(&child) {
                map.insert(parent, s);
            }
        }
        Ok(())
    }

    /// Read `seq`'s GDN state back as the snapshot record's layer rows.
    ///
    /// `None` when the sequence carries none — a slot that has never run a wave
    /// has nothing worth persisting, and writing a zero snapshot would be worse
    /// than writing none: resume would install it and report success.
    ///
    /// **This is the GDN state only.** The PLE window rides in the record's
    /// auxiliary blob ([`Self::export_aux_state`]) and the QSA index is rebuilt
    /// rather than stored — see this impl block's header for why the three
    /// classes are treated differently.
    ///
    /// A snapshot is taken at a turn's seal, which ends any prefill the
    /// sequence was in, so its prefill scratch goes back here: the next long
    /// span claims it again.
    pub fn export_recurrent(&self, seq: usize) -> Result<Option<(u64, Vec<ExportedLayerState>)>> {
        self.refuse_parked(seq)?;
        let mut map = self
            .recurrent
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?;
        let Some(store) = map.get_mut(&seq) else {
            // A held snapshot is this sequence's state as surely as a store
            // would be: read it back as it stands, without building one.
            let staged = self
                .staged
                .read()
                .map_err(|_| candle::Error::Msg("qwen4exp: staged lock poisoned".into()))?;
            return Ok(staged
                .get(&seq)
                .map(|s| (s.schedule_hash, s.layers.to_vec())));
        };
        // `export` refuses mid-wave itself; the seal runs outside the wave, so
        // reaching that error means the ordering broke.
        let layers = store.export()?;
        store.release_scratch();
        Ok(Some((store.schedule_hash(), layers)))
    }

    /// Scatter a snapshot into `seq`'s GDN state — the resume path.
    ///
    /// Creates the store if the slot has none yet, which is the normal case:
    /// resume runs at `create_sequence`, before any wave. `import` validates
    /// the schedule hash and every layer's geometry before touching a tensor.
    pub fn restore_recurrent(
        &self,
        seq: usize,
        schedule_hash: u64,
        layers: &[ExportedLayerState],
    ) -> Result<()> {
        // A restored state is the record's, not sequence-start, and it
        // supersedes any snapshot held back for this sequence.
        self.clear_pristine(seq);
        self.unpark(seq);
        self.staged
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: staged lock poisoned".into()))?
            .remove(&seq);
        let cfg = &self.model.cfg;
        let mut map = self
            .recurrent
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?;
        let store = match map.entry(seq) {
            std::collections::hash_map::Entry::Occupied(slot) => slot.into_mut(),
            std::collections::hash_map::Entry::Vacant(slot) => slot.insert(
                RecurrentStateStore::new(&cfg.layer_kinds, &cfg.delta_net, &self.model.device)?,
            ),
        };
        store.import(schedule_hash, layers)?;
        drop(map);
        self.mark_seeded(seq)
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
        let cfg = &self.model.cfg;
        RecurrentStateStore::check_snapshot(
            &cfg.layer_kinds,
            &cfg.delta_net,
            schedule_hash,
            &layers,
        )?;
        if self.has_recurrent(seq)? {
            return self.restore_recurrent(seq, schedule_hash, &layers);
        }
        self.clear_pristine(seq);
        self.unpark(seq);
        self.staged
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: staged lock poisoned".into()))?
            .insert(
                seq,
                StagedSnapshot {
                    schedule_hash,
                    layers,
                },
            );
        self.mark_seeded(seq)
    }

    /// Drop `seq`'s GDN store: its last seal's snapshot is durable and the
    /// sequence is out of every forward, so the store's device ground is idle
    /// ground. The PLE history, seeds and QSA index stay — they are small, or
    /// rebuilt from pages, and none of them is what holds the span.
    ///
    /// An idle conversation keeps its slot for minutes, and its store with it:
    /// on blackworld (2026-10-18) 155 recurrent regions, 2.37 GB, held mostly
    /// by conversations no forward was running.
    pub fn park_recurrent(&self, seq: usize) -> Result<bool> {
        {
            let mut map = self
                .recurrent
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?;
            match map.get(&seq) {
                Some(store) if !store.wave_open() => {
                    map.remove(&seq);
                }
                _ => return Ok(false),
            }
        }
        self.clear_pristine(seq);
        self.parked
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: parked lock poisoned".into()))?
            .insert(seq);
        Ok(true)
    }

    /// Whether `seq`'s state is parked in its record.
    pub fn recurrent_parked(&self, seq: usize) -> bool {
        self.parked.read().is_ok_and(|p| p.contains(&seq))
    }

    /// Refuse a want of `seq`'s state while it is parked and nothing has been
    /// staged back for it.
    pub(super) fn refuse_parked(&self, seq: usize) -> Result<()> {
        if self.recurrent_parked(seq) && !self.has_recurrent(seq)? {
            let staged = self
                .staged
                .read()
                .map_err(|_| candle::Error::Msg("qwen4exp: staged lock poisoned".into()))?
                .contains_key(&seq);
            if !staged {
                candle::bail!(
                    "{}: sequence {seq} must be restored from its record before its state \
                     is read",
                    crate::models::batched_inference::RECURRENT_PARKED_MARKER
                );
            }
        }
        Ok(())
    }

    /// `seq`'s state is written from elsewhere: no longer the record's alone.
    pub(super) fn unpark(&self, seq: usize) {
        if let Ok(mut p) = self.parked.write() {
            p.remove(&seq);
        }
    }

    /// Build `seq`'s store from the snapshot held for it, if one is held. A
    /// build the span refuses keeps the snapshot held, so the claim that asked
    /// can be retried.
    pub(super) fn install_staged(&self, seq: usize) -> Result<()> {
        let Some(snap) = self
            .staged
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: staged lock poisoned".into()))?
            .remove(&seq)
        else {
            return Ok(());
        };
        let cfg = &self.model.cfg;
        let t = Instant::now();
        let built = snap.build(&cfg.layer_kinds, &cfg.delta_net, &self.model.device);
        ClaimPart::Build.charge(t.elapsed());
        match built {
            Ok(store) => {
                self.recurrent
                    .write()
                    .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?
                    .insert(seq, store);
                self.unpark(seq);
                Ok(())
            }
            Err(e) => {
                self.staged
                    .write()
                    .map_err(|_| candle::Error::Msg("qwen4exp: staged lock poisoned".into()))?
                    .insert(seq, snap);
                Err(e)
            }
        }
    }

    /// The carried state that is not a DeltaNet layer stack — the PLE window
    /// and one QSA index page per attention layer — for the turn record's
    /// auxiliary slot.
    ///
    /// Each layer's record carries the cache's completed rows AND its carried
    /// open block, so a resumed sequence's index covers exactly the tokens its
    /// restored K/V does — including the `T mod ratio` that had not completed a
    /// row when the turn ended, which is the usual case rather than an edge one.
    /// Rows above `n_blocks` (and `n_open`) are dead until an append writes them
    /// and are not persisted.
    ///
    /// `last_cells` is `ratio`: every row this cache completed is a full one.
    /// A page whose last row is SHORT is what a per-turn seal emits after
    /// [`IndexCache::flush_open_block`], for a window reconstructed from several
    /// turns' pieces — the container's format is the same either way, which is
    /// what lets both forms share one record.
    pub fn export_aux_state(&self, seq: usize) -> Result<Option<Vec<u8>>> {
        let ple_blob = {
            let map = self
                .ple
                .read()
                .map_err(|_| candle::Error::Msg("qwen4exp: ple lock poisoned".into()))?;
            match map.get(&seq) {
                Some(state) => state.encode()?,
                None => return Ok(None),
            }
        };
        let layers = {
            let map = self
                .index
                .read()
                .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
            match map.get(&seq) {
                Some(caches) => {
                    let ratios = self.attention_ratios();
                    let mut v = Vec::with_capacity(caches.len());
                    for (c, ratio) in caches.iter().zip(ratios.iter()) {
                        // The live tail's rows as they are stored — un-rotated,
                        // so position-free — and `restore_aux_state` restores
                        // them into a cache whose tail opens at zero. (The
                        // injected pages ahead of the tail are not exported at
                        // all — a resume rebuilds them from the projection, and
                        // `indexed_tokens` is what reports the shortfall if one
                        // does not.)
                        v.push(seal_live(c, (*ratio).max(1))?);
                    }
                    v
                }
                None => Vec::new(),
            }
        };
        Ok(Some(paged_index::encode_aux(&ple_blob, &layers)?))
    }

    /// Install a PLE window and the QSA index pages from a snapshot's
    /// auxiliary blob.
    ///
    /// Refuses a page count that does not match this checkpoint's KV layers
    /// rather than installing a partial index: a sequence whose deepest
    /// attention layers had no index would select against an empty candidate
    /// set there and attend to nothing, which reads as a retrieval that simply
    /// found nothing relevant.
    pub fn restore_aux_state(&self, seq: usize, blob: &[u8]) -> Result<()> {
        let cfg = &self.model.cfg;
        let (ple_blob, layers) = paged_index::decode_aux(blob)?;
        let state = PleState::decode(
            &ple_blob,
            cfg.ple.conv_history(),
            cfg.hc.dim(cfg.hidden_size),
            &self.model.device,
        )?;
        let want = cfg.kv_layers().total();
        if !layers.is_empty() && layers.len() != want {
            candle::bail!(
                "qwen4exp: the snapshot carries {} index pages but this checkpoint has {want} \
                 KV layers — a partial index would select against an empty candidate set on \
                 the layers it does not cover",
                layers.len()
            );
        }
        self.ple
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: ple lock poisoned".into()))?
            .insert(seq, state);
        if !layers.is_empty() {
            let mut caches = Vec::with_capacity(layers.len());
            for s in &layers {
                caches.push(IndexCache::from_rows(
                    &s.keys,
                    &s.open,
                    cfg.indexer.head_dim,
                    &self.model.device,
                )?);
            }
            self.index
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?
                .insert(seq, caches);
        }
        self.mark_seeded(seq)
    }

    /// Bytes the carried state holds for every live sequence.
    ///
    /// Reported because it is large and it moves: this is a span-reservation
    /// tenant, and a total that omits it makes the partition look emptier than
    /// it is — the same blindness that let the dense weights hide.
    pub fn recurrent_reserved_bytes(&self) -> usize {
        // Every tenant arena's regions, not a sum over the holders: holders share
        // arenas, so their sum leaves out the free slots and unused tails.
        let gdn = RecurrentStateStore::arena_reserved_bytes(&self.model.device);
        let idx = arena_regions(&self.model.device, SlotTenant::QsaIndex) * SpanRegion::bytes();
        gdn + idx
    }

    /// Compact every arena the carried state lives in: each sequence's GDN state
    /// ([`compact_stores`]) and each QSA index cache ([`compact_index_caches`]).
    /// The report sums both. Between forwards; the locks are held for the pass,
    /// which is what keeps a wave from opening on a store or a cache mid-move.
    pub fn compact_recurrent(&self, max_moves: usize) -> Result<RecurrentCompaction> {
        let mut map = self
            .recurrent
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: recurrent lock poisoned".into()))?;
        let gdn = compact_stores(
            map.values_mut(),
            &self.model.cfg.delta_net,
            &self.model.device,
            max_moves,
        )?;
        let mut idx = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;
        let mut caches: Vec<&mut IndexCache> = idx.values_mut().flatten().collect();
        let retired: Vec<Arc<ResidentPage>> = self
            .retired
            .read()
            .map_err(|_| candle::Error::Msg("qwen4exp: retired lock poisoned".into()))?
            .values()
            .flatten()
            .cloned()
            .collect();
        let qsa = compact_index_caches(
            &mut caches,
            &retired,
            self.model.cfg.indexer.head_dim,
            &self.model.device,
            max_moves,
        )?;
        let (planned, moved) = self.compact_carried(max_moves)?;
        // The carried buffers share the GDN state's tenant, so its region count
        // (read after both passes) already covers them.
        let regions_after =
            arena_regions(&self.model.device, SlotTenant::RecurrentState) + qsa.regions_after;
        Ok(RecurrentCompaction {
            planned: gdn.planned + qsa.planned + planned,
            moved: gdn.moved + qsa.moved + moved,
            regions_before: gdn.regions_before + qsa.regions_before,
            regions_after,
        })
    }

    /// Compact the buffers a sequence carries beside its GDN state in the same
    /// tenant — the PLE block's conv histories and the draft head's seeds — each
    /// geometry's pool by its own pass, every holder relocated through
    /// [`relocate_tensor`]. Answers the moves planned and made.
    ///
    /// **They pin the frontier as surely as a state block.** They are claimed at
    /// their own strides, so the GDN pass, keyed by the state block's, never plans
    /// a move for them: on WhiteWorld (a09d18907, 2026-10-19) recurrent state held
    /// the top region over 17 to 27 holes for forty seconds with no sweep in
    /// flight and no store open, every lowering gaining nothing. A holder a
    /// snapshot still shares keeps its old slot alive until the snapshot goes,
    /// which costs reclaim, never correctness.
    pub(super) fn compact_carried(&self, max_moves: usize) -> Result<(usize, usize)> {
        let poisoned = || candle::Error::Msg("qwen4exp: carried-state lock poisoned".into());
        let mut ple = self.ple.write().map_err(|_| poisoned())?;
        let mut seeds = self.seeds.write().map_err(|_| poisoned())?;
        let mut spares = self.seed_spares.write().map_err(|_| poisoned())?;
        let mut holders: Vec<&mut Tensor> = ple
            .values_mut()
            .flat_map(|s| std::iter::once(&mut s.conv_hist).chain(s.spare_hist.as_mut()))
            .chain(seeds.values_mut())
            .chain(spares.values_mut())
            .collect();
        let mut sizes: Vec<usize> = holders
            .iter()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .collect();
        sizes.sort_unstable();
        sizes.dedup();
        let (mut planned, mut moved) = (0usize, 0usize);
        for bytes in sizes {
            let moves = plan_slot_moves(
                &self.model.device,
                SlotTenant::RecurrentState,
                bytes,
                max_moves,
            )?;
            planned += moves.len();
            let mut by_source: HashMap<u64, ArenaSlot> =
                moves.into_iter().map(|m| (m.src, m.dst)).collect();
            for t in holders
                .iter_mut()
                .filter(|t| t.elem_count() * t.dtype().size_in_bytes() == bytes)
            {
                moved += usize::from(relocate_tensor(t, &mut by_source)?);
            }
        }
        Ok((planned, moved))
    }

    /// What admitting **one** more sequence costs in carried state.
    ///
    /// The GDN store, priced from the geometry
    /// ([`RecurrentStateStore::reserved_bytes_for`]) — every store this model
    /// builds has the same shape, so the config answers for all of them, and it
    /// answers before the first one exists, which is when admission asks — and,
    /// because a sequence arrives prefilling, its prefill scratch: one more slot
    /// per DeltaNet layer, held until it decodes (`docs/deltanet_timeline.md` §7).
    ///
    /// **The index caches are deliberately not in it.** They are the other half
    /// of [`Self::recurrent_reserved_bytes`], but their key pages grow with the
    /// context the sequence has already decoded, so a *new* sequence brings only
    /// its open-block buffers — a few KiB per layer. Charging a fresh turn for the
    /// index a long conversation has accumulated would price arrivals by the
    /// depth of the sequences already resident, which is the mistake this
    /// function exists to end.
    pub fn recurrent_store_bytes(&self) -> usize {
        let cfg = &self.model.cfg;
        RecurrentStateStore::reserved_bytes_for(&cfg.layer_kinds, &cfg.delta_net)
            + RecurrentStateStore::scratch_bytes_for(&cfg.layer_kinds, &cfg.delta_net)
    }
}
