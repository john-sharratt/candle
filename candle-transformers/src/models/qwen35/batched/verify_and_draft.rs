//! Carries the speculative-decoding bookkeeping for each sequence: the verify
//! cohort, the draft seed and the KV layer the draft head writes.
//!
//! A verify block is armed with its rows held pending past the forward's commit,
//! then either folded with the rows the accept kept or discarded, so a rejected
//! tail leaves the state as it was. The draft seed is the hidden state at the
//! last accepted row, not the last row the wave computed. A wrong row changes no
//! token and only lowers acceptance, so it is easy to miss. A checkpoint without
//! a head drafts nothing, and a seed from another activation dtype is not used,
//! so a draft never proposes from noise.

use super::*;

impl HybridBatched {
    /// The sequences this wave must score every prefill row of. Empty on every
    /// wave but a speculative verify.
    pub fn verify_row_seqs(&self) -> Result<Vec<usize>> {
        Ok(self
            .verify_rows
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: verify_rows lock poisoned".into()))?
            .clone())
    }

    /// Name the verifying sequences for the next forward.
    pub fn set_verify_row_seqs(&self, seqs: &[usize]) -> Result<()> {
        *self
            .verify_rows
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: verify_rows lock poisoned".into()))? =
            seqs.to_vec();
        Ok(())
    }

    /// Arm this step's verify cohort — `(sequence, block length)` — replacing
    /// whatever the last step armed. Their DeltaNet layers stay pending past the
    /// forward's commit until [`Self::fold_armed`] or [`Self::disarm_verify`].
    ///
    /// **A block is at most [`TIMELINE_ROWS`] rows**, refused here rather than
    /// run: a longer span writes the prefill scratch, which cannot keep only a
    /// prefix, so it could never be partially accepted.
    pub fn arm_verify(&self, blocks: &[(usize, usize)]) -> Result<()> {
        if let Some(&(seq, len)) = blocks.iter().find(|&&(_, len)| len > TIMELINE_ROWS) {
            candle::bail!(
                "qwen35 verify: sequence {seq}'s {len}-row block is deeper than the \
                 {TIMELINE_ROWS}-row timeline — at most {} drafts a block",
                TIMELINE_ROWS - 1
            );
        }
        *self.armed_map()? = blocks.iter().copied().collect();
        Ok(())
    }

    /// The armed cohort, locked.
    pub(super) fn armed_map(&self) -> Result<MutexGuard<'_, HashMap<usize, usize>>> {
        self.armed
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: armed cohort lock poisoned".into()))
    }

    /// Whether `seq` is in the armed cohort, and its block's length.
    pub fn armed_block(&self, seq: usize) -> Result<Option<usize>> {
        Ok(self.armed_map()?.get(&seq).copied())
    }

    /// Take `seqs` out of the armed cohort and drop their pending timelines: the
    /// step that armed them ends without keeping any of their rows, so their
    /// state stands exactly as the step found it.
    pub fn disarm_verify(&self, seqs: &[usize]) -> Result<()> {
        // Released before the map is taken: `release_recurrent` locks the two the
        // other way round.
        {
            let mut armed = self.armed_map()?;
            for seq in seqs {
                armed.remove(seq);
            }
        }
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        for seq in seqs {
            if let Some(store) = map.get_mut(seq) {
                store.discard_pending();
            }
        }
        Ok(())
    }

    /// Fold the armed sequences' kept rows — `(sequence, rows kept)` — and take
    /// them out of the cohort: the reconcile's whole DeltaNet half, ONE fold
    /// launch over every sequence and recurrent layer.
    ///
    /// A sequence keeping every row of its block folds them all; one keeping
    /// none drops its timelines. Every job must name an armed sequence and keep
    /// at most its block's rows.
    pub fn fold_armed(&self, jobs: &[(usize, usize)]) -> Result<()> {
        let mut armed = self.armed_map()?;
        for &(seq, kept) in jobs {
            let len = armed.get(&seq).copied().ok_or_else(|| {
                candle::Error::Msg(format!(
                    "qwen35 verify: a fold for sequence {seq}, which is not in the armed cohort"
                ))
            })?;
            if kept > len {
                candle::bail!(
                    "qwen35 verify: sequence {seq} keeps {kept} rows of a {len}-row block"
                );
            }
        }
        // Out of the cohort before the fold runs: whether it succeeds or not,
        // these rows are consumed — a failed fold's must not survive to be
        // folded by a later step.
        for &(seq, _) in jobs {
            armed.remove(&seq);
        }
        drop(armed);
        let mut map = self
            .recurrent
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: recurrent state lock poisoned".into()))?;
        let mut stores: HashMap<usize, &mut RecurrentStateStore> = map
            .iter_mut()
            .filter(|(seq, _)| jobs.iter().any(|&(s, _)| s == **seq))
            .map(|(seq, st)| (*seq, st))
            .collect();
        let mut folds: Vec<(&mut RecurrentStateStore, Option<usize>)> =
            Vec::with_capacity(jobs.len());
        for &(seq, kept) in jobs {
            let store = stores.remove(&seq).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "qwen35 verify: armed sequence {seq} has no recurrent state to fold"
                ))
            })?;
            folds.push((store, Some(kept)));
        }
        fold_stores(&mut folds)
    }

    // ── The MTP draft head ────────────────────────────────────────────────
    //
    // `super::mtp` is the head itself and `super::draft` is how the engine runs
    // it; this is the per-sequence bookkeeping between them — one seed hidden
    // per sequence, and the KV layer the head writes.

    /// Whether this model carries a draft head at all.
    pub fn has_drafter(&self) -> bool {
        self.model.mtp.is_some()
    }

    /// This checkpoint's draft budget for a wave of `width` sequences.
    ///
    /// Gated on the head actually being loaded, not just on the ladder being
    /// non-empty: a checkpoint converted without the NextN tensors would
    /// otherwise have every wave pay a drafting call that can only return
    /// nothing.
    pub fn draft_budget_for(&self, width: usize) -> usize {
        if !self.has_drafter() {
            return 0;
        }
        // Bounded by the timeline: a block of `k` drafts verifies `k + 1` rows,
        // and a block deeper than the timeline is refused (`arm_verify`). The
        // rewind costs no memory a deeper block would add to — each DeltaNet
        // layer's timeline is a fixed part of its slot — so depth is otherwise
        // the ladder's alone.
        self.draft.budget(width).min(TIMELINE_ROWS - 1)
    }

    /// The drafted-token cost this checkpoint's ladder carries.
    pub fn draft_token_cost_for(&self) -> f32 {
        self.draft.token_cost()
    }

    /// The KV layer the draft head writes, past every trunk attention layer.
    /// `None` on a checkpoint without a head, where no such layer is allocated.
    pub fn mtp_kv_layer(&self) -> Option<usize> {
        mtp_kv_layer(&self.model.cfg)
    }

    /// The KV layers a wave over `[layer_start, layer_end)` touches — see
    /// [`wave_kv_range`], which is where the rule and its test live.
    pub fn kv_layer_range(&self, layer_start: usize, layer_end: usize) -> (usize, usize) {
        wave_kv_range(&self.model.cfg, layer_start, layer_end)
    }

    /// Propose up to `max_len` tokens after `committed` for each sequence.
    ///
    /// Empty — a plain decode step — when there is no head, or when the
    /// sequence has no seed yet (the first step after prefill, before any wave
    /// has scored a row for it). A seedless sequence is left out rather than
    /// drafted from zeros, which would propose noise and waste the verify row
    /// it costs.
    pub fn mtp_draft(
        &self,
        session: &mut BatchedInferenceSession,
        seqs: &[usize],
        committed: &[u32],
        max_len: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let empty = || vec![Vec::new(); seqs.len()];
        if !self.has_drafter() || max_len == 0 || seqs.is_empty() {
            return Ok(empty());
        }
        if committed.len() != seqs.len() {
            candle::bail!(
                "qwen35 mtp: {} committed tokens for {} sequences",
                committed.len(),
                seqs.len()
            );
        }

        let want = session.activation_dtype();
        let (draftable, seeds): (Vec<usize>, Vec<Tensor>) = {
            let mut map = self
                .seed
                .lock()
                .map_err(|_| candle::Error::Msg("qwen35: seed lock poisoned".into()))?;
            let mut idx = Vec::with_capacity(seqs.len());
            let mut seeds = Vec::with_capacity(seqs.len());
            for (i, seq) in seqs.iter().enumerate() {
                match map.get(seq) {
                    // A seed captured under a DIFFERENT activation dtype belongs
                    // to a session that no longer exists — the harness builds one
                    // session per config and a sequence index is reused across
                    // them, so the seed left behind is a previous run's hidden.
                    // Concatenating it with this run's embedding is a dtype
                    // mismatch, and the sequence simply drafts nothing until the
                    // next wave captures a fresh one.
                    Some(s) if s.dtype() != want => {
                        map.remove(seq);
                    }
                    Some(s) => {
                        idx.push(i);
                        seeds.push(s.clone());
                    }
                    None => {}
                }
            }
            (idx, seeds)
        };
        if draftable.is_empty() {
            return Ok(empty());
        }

        let cohort: Vec<usize> = draftable.iter().map(|&i| seqs[i]).collect();
        let toks: Vec<u32> = draftable.iter().map(|&i| committed[i]).collect();
        let drafted = draft_cohort(self, session, &cohort, &toks, &seeds, max_len)?;

        let mut out = empty();
        for (k, &i) in draftable.iter().enumerate() {
            out[i] = drafted[k].clone();
        }
        Ok(out)
    }

    /// Take each sequence's next seed from the wave that just ran: the trunk's
    /// hidden at the LAST ACCEPTED position of its block.
    ///
    /// `jobs` pairs a sequence with how many of its rows the accept kept, so
    /// row `kept - 1` of the buffer the wave captured is the one. Which row
    /// that is cannot be known during the wave — it is decided by comparing the
    /// target's argmaxes against the proposal — which is why the wave captures
    /// the whole block and this picks from it.
    ///
    /// **`None` means "every row this wave captured", and it is not a synonym
    /// for one.** A caller that kept the whole block knows it kept the whole
    /// block; it does not necessarily know how long the block was, because the
    /// accept classifies by *offset* and only a rewind carries a row count.
    /// Asking here rather than assuming is the difference between seeding from
    /// the last accepted row and seeding from the block's first — an error that
    /// changes no token, because verify keeps only the target's own argmaxes,
    /// and shows up solely as acceptance decaying toward 1.00.
    ///
    /// Nothing else is caught up here. The head's KV took every one of those
    /// positions inside the wave, as a layer, and the rejected tail truncates
    /// away with every other layer's.
    pub fn mtp_take_seeds(&self, jobs: &[(usize, Option<usize>)]) -> Result<()> {
        if !self.has_drafter() {
            return Ok(());
        }
        let mut picked: Vec<(usize, Tensor)> = Vec::with_capacity(jobs.len());
        for &(seq, kept) in jobs {
            let Some(rows) = self.hidden_buffer(seq)? else {
                continue;
            };
            // The wave's own row count, never the buffer's: buffers only grow,
            // so `dim(0)` is a high-water mark from whichever earlier block was
            // widest.
            let Some(filled) = self.captured_rows(seq)? else {
                continue;
            };
            let kept = kept.unwrap_or(filled);
            if kept == 0 {
                continue;
            }
            if kept > filled || filled > rows.dim(0)? {
                candle::bail!(
                    "qwen35 mtp: sequence {seq} accepted {kept} of {filled} captured rows \
                     in a {}-row buffer",
                    rows.dim(0)?
                )
            }
            picked.push((seq, rows.narrow(0, kept - 1, 1)?.contiguous()?));
        }
        let mut map = self
            .seed
            .lock()
            .map_err(|_| candle::Error::Msg("qwen35: seed lock poisoned".into()))?;
        for (seq, seed) in picked {
            map.insert(seq, seed);
        }
        Ok(())
    }

    /// Drop a sequence's draft state.
    pub fn release_draft(&self, seq: usize) {
        if let Ok(mut m) = self.seed.lock() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.verify_hidden.lock() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.capture_rows.lock() {
            m.remove(&seq);
        }
    }
}
