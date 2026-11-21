//! Places sealed or borrowed K/V pieces onto a sequence's tail or into its
//! middle while keeping every layer on the same token windows.
//!
//! Input is reconciled to the common prefix before it enters, so a per-layer
//! skew cannot desync the slot and wedge later decodes. Appends go onto content
//! and never onto a layer's empty writer chunk. An empty chunk left in place
//! turns a lossless block-count skew into an interior hole, which the tail
//! healing path refuses to repair. Each per-layer loop is atomic, so a failure
//! rolls back every layer already injected and no prefix of layers runs ahead.

use super::*;

impl BatchedInferenceSession {
    /// Append the sealed chunks of `sealed_per_layer` onto the tail of
    /// `seq_idx` as live chunk windows.  Pure metadata — no DMA.
    ///
    /// The source is first run through [`reconcile_sealed_prefix`], so a
    /// per-layer creep skew is cut to the common prefix rather than injected
    /// verbatim (which would desync the slot and wedge every later decode), and
    /// the per-layer loop is atomic — a mid-loop failure rolls back every layer
    /// already injected. After reconciliation the layers are uniform, so the
    /// appended token count (which advances the sequence's logical offset) and
    /// the block range are read from layer 0.
    ///
    /// A reconciled source is reported, not logged: the caller knows which
    /// stored unit it injected and is the one that can repair the record, so
    /// the cut is not met again on every later injection of the same unit.
    pub fn inject_sealed_at_tail(
        &mut self,
        seq_idx: usize,
        sealed_per_layer: &[candle_nn::kv_cache::SealedSequence],
    ) -> Result<SealedInjection> {
        if sealed_per_layer.len() != self.backings.len() {
            candle::bail!(
                "inject_sealed_at_tail: got {} layers but session has {} backings",
                sealed_per_layer.len(),
                self.backings.len()
            );
        }
        // Reconcile a per-layer creep skew before injecting. If the sealed
        // section has one layer group a filled chunk ahead of the others,
        // injecting it verbatim desyncs the slot per layer and wedges every
        // later decode (`unify_decode_layout` cannot heal a mid-history,
        // Arc-shared divergence). Cut every layer to the common prefix so only a
        // consistent turn enters the slot — this both blocks new corruption and
        // lets an already-skewed section (from a substrate sealed before the
        // seal-time guard, or that slipped past it) reload and continue.
        let reconciled = reconcile_sealed_prefix(sealed_per_layer);
        let (source, cut_chunks): (&[candle_nn::kv_cache::SealedSequence], usize) =
            match reconciled.as_deref() {
                Some(fixed) => {
                    let cut = sealed_per_layer
                        .iter()
                        .zip(fixed.iter())
                        .map(|(s, f)| s.chunks.len() - f.chunks.len())
                        .max()
                        .unwrap_or(0);
                    (fixed, cut)
                }
                None => (sealed_per_layer, 0),
            };

        // **Append onto content, never onto a layer's empty writer chunk.**
        // Reconciling the source above cannot see this one: it is the
        // *destination*'s skew, not the incoming record's — a layer carrying a
        // trailing empty chunk on `seq_idx` takes the injected content one
        // block further along than a layer without one. That is how a silent,
        // lossless block-count skew (padding from `reconcile_block_counts`, or
        // a creep's phantom writer chunk — both benign while the extra block
        // is empty) becomes an *interior* hole the instant anything is
        // appended, and an interior hole is precisely what
        // `heal_tail_divergence` refuses to repair. Trimming first is lossless
        // for the same reason the padding was: an empty chunk holds no token,
        // so no position moves. The caller pushes a fresh writer chunk after an
        // inject when it needs one (`push_empty_if_sealed`).
        self.trim_empty_tail_chunks(seq_idx)?;

        // Atomic per-layer inject: capture each layer's pre-inject block count
        // and roll every injected layer back if a later one fails, so a mid-loop
        // error can never leave a prefix of layers ahead of the rest — the exact
        // permanent skew this whole path exists to avoid (cf. `reserve_glue_gap`,
        // whose loop carries the same rollback for the same reason).
        let mut range = (0usize, 0usize);
        let mut pre_counts: Vec<usize> = Vec::with_capacity(self.backings.len());
        let rollback = |backings: &[ChunkedKvBacking], pre: &[usize]| {
            for (li, &c) in pre.iter().enumerate() {
                let _ = backings[li].truncate_sequence_to_blocks(seq_idx, c);
            }
        };
        for (i, (backing, sealed)) in self.backings.iter().zip(source.iter()).enumerate() {
            let pre = backing.sequence_block_count(seq_idx).unwrap_or(0);
            pre_counts.push(pre);
            match backing.inject_sealed_at_tail(seq_idx, sealed) {
                Ok(r) => {
                    if i == 0 {
                        range = r;
                    }
                }
                Err(e) => {
                    rollback(&self.backings, &pre_counts);
                    return Err(e);
                }
            }
        }
        // Every layer now holds the reconciled (identical) windows, so the token
        // count is uniform — layer 0 speaks for all of them.
        let tokens_added: usize = source
            .first()
            .map(|s| s.chunks.iter().map(|c| c.token_count as usize).sum())
            .unwrap_or(0);
        if tokens_added > 0 {
            if let Some(Some(state)) = self.sequences.get_mut(seq_idx) {
                state.offset += tokens_added;
            }
        }
        // Walked only when something was cut: an aligned piece lost nothing.
        let cut_tokens = if cut_chunks == 0 {
            0
        } else {
            sealed_per_layer
                .iter()
                .map(|s| {
                    s.chunks
                        .iter()
                        .map(|c| c.token_count as usize)
                        .sum::<usize>()
                })
                .max()
                .unwrap_or(0)
                .saturating_sub(tokens_added)
        };
        Ok(SealedInjection {
            range,
            tokens: tokens_added,
            cut_chunks,
            cut_tokens,
        })
    }

    /// Replace the sealed blocks `[start, end)` of `seq_idx` with
    /// `sealed_per_layer` on every layer, leaving every block outside the range
    /// where it stands — the mid-sequence counterpart of
    /// [`Self::inject_sealed_at_tail`], and as pure: no DMA, no kernel.
    ///
    /// What lets a rebuild put a changed piece where the projection placed it
    /// without re-injecting every unchanged piece after it: each block's RoPE
    /// base is the cumulative usage before it, so the blocks after the range
    /// move with the splice. An empty piece evicts the range; an empty range
    /// inserts.
    ///
    /// The source is reconciled as an injection's is, the per-layer loop is
    /// atomic — a failure puts back every layer already spliced — and the
    /// sequence's logical offset moves by the tokens inserted less the tokens
    /// removed. The returned range is where the inserted chunks now sit.
    pub fn splice_sealed(
        &mut self,
        seq_idx: usize,
        start: usize,
        end: usize,
        sealed_per_layer: &[candle_nn::kv_cache::SealedSequence],
    ) -> Result<SealedInjection> {
        if sealed_per_layer.len() != self.backings.len() {
            candle::bail!(
                "splice_sealed: got {} layers but session has {} backings",
                sealed_per_layer.len(),
                self.backings.len()
            );
        }
        let reconciled = reconcile_sealed_prefix(sealed_per_layer);
        let (source, cut_chunks): (&[candle_nn::kv_cache::SealedSequence], usize) =
            match reconciled.as_deref() {
                Some(fixed) => {
                    let cut = sealed_per_layer
                        .iter()
                        .zip(fixed.iter())
                        .map(|(s, f)| s.chunks.len() - f.chunks.len())
                        .max()
                        .unwrap_or(0);
                    (fixed, cut)
                }
                None => (sealed_per_layer, 0),
            };
        // What the range holds now, per layer: the tokens it gives up, and what
        // a failed splice puts back.
        let removed = self.snapshot_sequence_blocks(seq_idx, start, end)?;
        let removed_tokens = removed.first().map_or(0, |s| s.token_count);
        let offset = match self.sequences.get(seq_idx) {
            Some(Some(state)) => state.offset,
            _ => candle::bail!("splice_sealed: sequence {seq_idx} not allocated"),
        };
        if removed_tokens > offset {
            candle::bail!(
                "splice_sealed: [{start}, {end}) holds {removed_tokens} tokens but sequence \
                 {seq_idx} stands at {offset}"
            );
        }
        for (done, (backing, sealed)) in self.backings.iter().zip(source.iter()).enumerate() {
            if let Err(e) = backing.splice_sequence_blocks(seq_idx, start, end, sealed) {
                for ((b, put), took) in self
                    .backings
                    .iter()
                    .zip(&removed)
                    .zip(source.iter())
                    .take(done)
                {
                    let _ =
                        b.splice_sequence_blocks(seq_idx, start, start + took.chunks.len(), put);
                }
                return Err(e);
            }
        }
        let tokens_added: usize = source
            .first()
            .map(|s| s.chunks.iter().map(|c| c.token_count as usize).sum())
            .unwrap_or(0);
        if let Some(Some(state)) = self.sequences.get_mut(seq_idx) {
            state.offset = offset - removed_tokens + tokens_added;
        }
        let cut_tokens = if cut_chunks == 0 {
            0
        } else {
            sealed_per_layer
                .iter()
                .map(|s| {
                    s.chunks
                        .iter()
                        .map(|c| c.token_count as usize)
                        .sum::<usize>()
                })
                .max()
                .unwrap_or(0)
                .saturating_sub(tokens_added)
        };
        let inserted = source.first().map_or(0, |s| s.chunks.len());
        Ok(SealedInjection {
            range: (start, start + inserted),
            tokens: tokens_added,
            cut_chunks,
            cut_tokens,
        })
    }

    /// Force-push a fresh empty writer chunk onto every layer's view
    /// of `seq_idx`.  Used by cumulative section ingest right after
    /// [`inject_sealed_at_tail`] populates a scratch slot with prefix
    /// sections' substrate chunks: the prefix's last partial chunk
    /// is Arc-shared with the substrate, so writing into it would
    /// mutate bytes other holders see as immutable.  Pushing a fresh
    /// chunk here makes the slot's tail a writer-owned chunk; the
    /// upcoming prefill writes start at that chunk's position 0
    /// (logical position = `prefix_token_count`).
    pub fn push_empty_writer_chunk(&mut self, seq_idx: usize) -> Result<()> {
        for backing in &self.backings {
            backing.push_empty_writer_chunk(seq_idx)?;
        }
        Ok(())
    }

    /// Update the offset for a sequence after processing tokens.
    pub fn advance_sequence(&mut self, idx: usize, tokens_processed: usize) -> Result<()> {
        if idx >= self.sequences.len() {
            candle::bail!("invalid sequence index {}", idx);
        }
        if let Some(ref mut state) = self.sequences[idx] {
            state.offset += tokens_processed;
            Ok(())
        } else {
            candle::bail!("sequence {} not allocated", idx);
        }
    }

    /// Inject pre-populated prefix chunks into a sequence across all layers.
    ///
    /// The sequence must already be allocated (via `create_sequence`). Each
    /// layer backing gets the same chunk_ids/rope_positions/seq_len.
    pub fn inject_prefix(
        &self,
        seq_idx: usize,
        chunk_ids: &[HeadGids],
        seq_len: usize,
    ) -> Result<()> {
        for backing in &self.backings {
            backing.inject_prefix_chunks(seq_idx, chunk_ids, seq_len)?;
        }
        Ok(())
    }

    /// Append borrowed chunk references to an existing sequence across all layers.
    ///
    /// Extends the sequence's block table starting from its current `block_count`.
    /// Does not change `block_usage`. Advances the backing's length tracking
    /// by `token_count`.
    ///
    /// The caller must call `advance_sequence(seq_idx, token_count)` afterwards
    /// to update the session-level offset.
    pub fn append_borrowed_chunks_cow(
        &self,
        seq_idx: usize,
        chunk_ids: &[HeadGids],
        token_count: usize,
    ) -> Result<()> {
        for backing in &self.backings {
            backing.append_borrowed_chunks_cow(seq_idx, chunk_ids, token_count)?;
        }
        Ok(())
    }
}
