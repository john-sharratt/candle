//! Keeps every layer's block list for a sequence in agreement, so one block index
//! names the same token window on all of them.
//!
//! Layers can differ by a trailing empty writer chunk after a windowed creep
//! prefill, which is harmless only until something is appended after it.
//! Reconciliation pads lagging layers up to the common count before a sequence
//! is treated as uniform. Trimming removes the empty tail before an append,
//! which is lossless because an empty chunk holds no token, so each layer's next
//! write lands at the same index. Reserving a glue gap is atomic: a failure rolls
//! back the layers already pushed.

use super::*;

impl BatchedInferenceSession {
    /// How many blocks of `seq_idx` hold content: on every layer, its block
    /// count without the empty (0-token) writer chunks at its tail — the
    /// fewest across layers. `None` when the slot is unallocated on every layer.
    ///
    /// **A record of where the slot stood after something was placed counts
    /// this, never [`Self::sequence_block_count`].** An empty writer chunk at
    /// the tail is where the *next* write lands — a glue gap's reservation
    /// pushes one behind it, and [`Self::inject_sealed_at_tail`] trims it and
    /// lands its first chunk at that very index. A count that included it then
    /// ends inside whatever was placed next, and cutting the slot back to it
    /// keeps that piece's first chunk: measured as a kept prefix whose slot
    /// offset ran exactly one 32-token chunk past its logical position.
    pub fn sequence_content_block_count(&self, seq_idx: usize) -> Result<Option<usize>> {
        let mut fewest: Option<usize> = None;
        for backing in &self.backings {
            let Some((_, held)) = backing.content_extent(seq_idx)? else {
                continue;
            };
            fewest = Some(fewest.map_or(held, |f| f.min(held)));
        }
        Ok(fewest)
    }

    /// Drop every layer's trailing empty (0-token) chunks from `seq_idx`, so
    /// each layer ends flush with the last token it actually holds.
    ///
    /// The inverse of [`Self::reconcile_block_counts`], and the one the layers
    /// need before anything is **appended**. Padding equalises the block count
    /// of a slot that is about to be read as one uniform thing; it is the wrong
    /// move before a write, because the pad then sits between the old content
    /// and the new on exactly the layers that were short. Trimming is lossless
    /// by the same argument that makes padding lossless — an empty chunk holds
    /// no token — and the holes an empty chunk carries stay pending at the new
    /// tail (`ChunkedKvBacking::trim_empty_tail`), so no position moves either.
    /// It leaves every layer's next append at the same block index by
    /// construction.
    pub fn trim_empty_tail_chunks(&mut self, seq_idx: usize) -> Result<()> {
        for backing in &self.backings {
            // A slot holding more blocks than the backing's `max_blocks` is a
            // real invariant violation — the sequence has outgrown the arena's
            // own capacity — and `content_extent` refuses it rather than
            // silently keeping everything.
            let Some((blocks, keep)) = backing.content_extent(seq_idx)? else {
                continue;
            };
            if keep < blocks {
                backing.trim_empty_tail(seq_idx)?;
            }
        }
        Ok(())
    }

    /// Pad every lagging layer's block count up to the max with an empty
    /// (0-token) writer chunk, so every layer of `seq_idx` agrees on how many
    /// blocks it holds.
    ///
    /// **During a windowed creep prefill, a layer in an earlier window pushes
    /// an empty writer chunk for the next window ahead of layers in a later
    /// window still pending resume**, so the layers' block counts differ by
    /// one even though their materialised token counts match (cf. dcd075e0,
    /// which fixed the same incremental-fill skew for
    /// `sequence_backing_tokens`). Left unreconciled, that skew is exactly
    /// what [`Self::sequence_block_count`]'s doc assumes never survives past
    /// a phantom empty chunk — but nothing stops a LATER write from landing
    /// in that chunk on the ahead layers while the lagging layers are still
    /// behind, at which point the "extra block" is no longer empty and the
    /// layers permanently disagree on token content, not just block count.
    /// **Any caller about to treat a sequence's per-layer view as uniform and
    /// then persist or reuse that view — capturing a `CapturedSpan`, sealing
    /// a turn, reserving a glue gap — must reconcile first**, while the extra
    /// blocks are still genuinely empty and padding is lossless. Padding
    /// (not truncating) the ahead layer is what's safe: truncating would drop
    /// a writer chunk a co-batched decode may target.
    pub fn reconcile_block_counts(&mut self, seq_idx: usize) -> Result<()> {
        let max_blocks = (0..self.backings.len())
            .filter_map(|li| self.backings[li].sequence_block_count(seq_idx))
            .max()
            .unwrap_or(0);
        for backing in &mut self.backings {
            // `None` (unallocated) reads as `max_blocks`, so the loop is skipped and
            // never spins; a successful push raises the count by one toward `max`.
            while backing.sequence_block_count(seq_idx).unwrap_or(max_blocks) < max_blocks {
                backing.push_empty_writer_chunk(seq_idx)?;
            }
        }
        Ok(())
    }

    /// Reserve a full-by-construction glue gap across every layer's backing.
    /// Returns `(gap_block_index, in_blk_base)` — the block index (identical
    /// across layers) and the first valid slot of the gap's tail window, into
    /// which the glue forward scatters the island's K/V. The glue forward
    /// later fills the gap by explicit `(slice, in_blk)` write target; until
    /// then its K/V is uninitialised but never read (the kernel scatters
    /// before it streams).
    ///
    /// This is the interleaved-glue primitive: because the gap is a real
    /// chunk with `usage = n_tokens` sitting at its logical position, the
    /// cumulative-usage `rope_base` of every later chunk equals its true
    /// sequence position — so decode and glue share one positional
    /// convention (`slice_rope`) with no `col_actual_pos` side channel.
    pub fn reserve_glue_gap(&mut self, seq_idx: usize, n_tokens: u32) -> Result<(usize, u32)> {
        // `reserve_glue_gap_chunk` MUTATES each layer (pushes a gap chunk + a writer
        // chunk). This loop must therefore be ATOMIC: if it bails mid-way — because a
        // later layer's gap index diverges, or a per-layer reservation errors — the
        // layers already pushed must be ROLLED BACK, or the slot is left one chunk
        // longer on those layers and EVERY subsequent reservation diverges harder,
        // permanently wedging the sequence. Each layer's returned `idx` is its
        // PRE-push block count (gap_idx = block_count-1 taken right after the gap
        // push), so truncating a pushed layer back to `idx` blocks restores it
        // exactly. (The primary fix is deferring the pinned working set from the
        // warm→cold gather so a live slot's layers stay uniform; this keeps a
        // residual divergence a clean, retryable turn error instead of corruption.)
        // Reconcile per-layer block counts BEFORE reserving — see
        // `reconcile_block_counts`. The reservation needs ONE gap index across
        // every layer, so a lagging layer must be padded up first.
        self.reconcile_block_counts(seq_idx)?;

        let mut gap: Option<(usize, u32)> = None;
        let mut pre_counts: Vec<usize> = Vec::with_capacity(self.backings.len());
        let rollback = |backings: &[ChunkedKvBacking], pre: &[usize]| {
            for (li, &c) in pre.iter().enumerate() {
                let _ = backings[li].truncate_sequence_to_blocks(seq_idx, c);
            }
        };
        for li in 0..self.backings.len() {
            let (idx, in_blk_base) =
                match self.backings[li].reserve_glue_gap_chunk(seq_idx, n_tokens) {
                    Ok(v) => v,
                    Err(e) => {
                        rollback(&self.backings, &pre_counts);
                        return Err(e);
                    }
                };
            pre_counts.push(idx);
            match gap {
                None => gap = Some((idx, in_blk_base)),
                Some((g, _)) if g != idx => {
                    rollback(&self.backings, &pre_counts);
                    candle::bail!(
                        "reserve_glue_gap: layer gap index diverged ({g} != {idx}) for slot {seq_idx}"
                    );
                }
                _ => {}
            }
        }
        if let Some(Some(state)) = self.sequences.get_mut(seq_idx) {
            state.offset += n_tokens as usize;
        } else {
            rollback(&self.backings, &pre_counts);
            candle::bail!("reserve_glue_gap: sequence {seq_idx} not allocated");
        }
        gap.ok_or_else(|| candle::Error::Msg("reserve_glue_gap: no layers".into()))
    }

    /// Truncate every backing's view of `seq_idx` to `block_count`
    /// chunks; reset the session's logical offset to the resulting
    /// token count (read from layer 0).
    ///
    /// Lets the scheduler reset a persistent conversation sequence to
    /// its system-prompt baseline before injecting the next turn's
    /// projection.
    /// Cut a sequence's chunk list back to `block_count` blocks.
    ///
    /// **A non-zero target is not automatically a rewind of live state.** Two
    /// callers legitimately pass one, and neither leaves a model's per-sequence
    /// state describing tokens the K/V no longer holds:
    ///
    /// - the projection assembler's `cut_slot`, which cuts the slot after the
    ///   pieces it keeps *and* truncates the per-position state to the same
    ///   point in the same step, so both sides move together;
    /// - `reserve_glue_gap`'s error-path rollback, which releases chunks a
    ///   failed call had just reserved and that nothing has decoded into.
    ///
    /// The operation that *is* a state-corrupting rewind is the token-granular
    /// [`Self::truncate_sequence_to_tokens`], reached only from speculative
    /// decode, and it is refused up front for a model that
    /// [`ManagedBatchedModel::carries_recurrent_state`].
    pub fn truncate_sequence_to_blocks(
        &mut self,
        seq_idx: usize,
        block_count: usize,
    ) -> Result<()> {
        let mut new_offset: Option<usize> = None;
        for backing in &self.backings {
            let n = backing.truncate_sequence_to_blocks(seq_idx, block_count)?;
            if new_offset.is_none() {
                new_offset = Some(n);
            }
        }
        if let Some(o) = new_offset {
            if let Some(Some(state)) = self.sequences.get_mut(seq_idx) {
                state.offset = o;
            }
        }
        Ok(())
    }

    /// Truncate `seq_idx`'s KV to an exact `target_tokens` count across every backing, and set
    /// the sequence offset to match. Token-granular counterpart to
    /// [`Self::truncate_sequence_to_blocks`] — the primitive the generic speculative-decode
    /// driver uses to roll back the rejected tail of a verified block.
    pub fn truncate_sequence_to_tokens(
        &mut self,
        seq_idx: usize,
        target_tokens: usize,
    ) -> Result<()> {
        for backing in &self.backings {
            backing.truncate_sequence_to_tokens(seq_idx, target_tokens)?;
        }
        if let Some(Some(state)) = self.sequences.get_mut(seq_idx) {
            state.offset = target_tokens;
            // **The per-layer caches carry their own length, and it is not the
            // backing's.** Leaving them is how a rewound sequence walks into
            // the next wave with two answers for how long it is: the wave's
            // pre-claim ensures a write chunk at the SESSION offset, while the
            // decode path primes its write slot from `KvCache::current_seq_len`
            // — so a stale cache asks for a chunk nobody claimed, and the arena
            // refuses to create one from inside the forward that owns the
            // partition. That is not a hypothetical: it is what a speculative
            // sequence did on the step after a partial accept, once it had
            // dropped back to a plain decode row.
            for cache in state.caches.caches.iter_mut() {
                cache.truncate_to_offset(target_tokens)?;
            }
        }
        Ok(())
    }
}
