//! Making sure a sequence has writable chunks for the tokens about to be written.
//!
//! Each write counts and allocates the chunks it needs before any guard is taken,
//! because an allocation can reach a compaction that needs a write guard on every
//! layer's block table. The tail is then made writable and the new blocks are
//! installed under one guard, so a reader never sees blocks pushed with the tail
//! still unreplaced. The steady decode step returns before taking the write guard.

use super::*;

impl ChunkedKvBacking {
    /// Ensure that chunks needed to write `add` tokens at `offsets` are allocated.
    ///
    /// `offsets` must have exactly `batch_capacity()` elements, one per sequence slot.
    pub fn ensure_for_offsets(&self, offsets: &[usize], adds: &[usize]) -> Result<()> {
        let batch = self.batch_capacity();
        if offsets.len() != batch {
            candle::bail!(
                "offset count mismatch: got {} offsets for chunked backing batch {}",
                offsets.len(),
                batch
            )
        }
        if adds.len() != offsets.len() {
            candle::bail!(
                "ensure_for_offsets: {} adds for {} offsets",
                adds.len(),
                offsets.len()
            )
        }
        if adds.iter().all(|&a| a == 0) {
            return Ok(());
        }

        let mut required_max_blocks = 1usize;
        for (i, &off) in offsets.iter().enumerate() {
            let end_pos = off.saturating_add(adds[i]).saturating_sub(1);
            let need_blocks = (end_pos / CHUNK_SIZE) + 1;
            required_max_blocks = cmp::max(required_max_blocks, need_blocks);
        }
        self.ensure_max_blocks(required_max_blocks)?;

        let chunk_size = CHUNK_SIZE;

        // Count first, allocate WITHOUT the guard, then install. `alloc_block_chunks`
        // can reach `request_global_compact`, which needs a write guard on every
        // layer's block table; allocating under one made that a self-deadlock and,
        // once the compactor was made non-blocking, a permanent no-op — so arena
        // compaction could never run from the prefill path that needs it most.
        // See `ensure_for_batch_entries` for the full rationale.
        // Plan blocks AND predict tail needs in one read pass, allocate both with
        // no guard held, then mutate under ONE write guard — extending a sequence
        // and making its tail writable must be atomic (a reader seeing blocks
        // pushed but the tail unreplaced would write into a full or closed-quant
        // chunk). Predicting before the installs over-estimates safely: a freshly
        // pushed block is writable, so installs only shrink the need.
        let mut plan: Vec<(usize, usize)> = Vec::new();
        let mut tail_maybe: Vec<usize> = Vec::new();
        {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for (b, &off) in offsets.iter().enumerate() {
                if state.sequences[b].is_none() || adds[b] == 0 {
                    continue;
                }
                let end_pos = off.saturating_add(adds[b]).saturating_sub(1);
                let need_blocks = (end_pos / chunk_size) + 1;
                let slot = state.sequences[b].as_ref().unwrap();
                let missing = (0..need_blocks)
                    .filter(|&blk| slot.chunk_at(blk).is_none())
                    .count();
                if missing > 0 {
                    plan.push((b, missing));
                }
            }
            for b in 0..batch {
                if state.sequences[b].is_some() && self.tail_needs_new_block(&state, b) {
                    tail_maybe.push(b);
                }
            }
        }
        let mut prealloc: Vec<(usize, Vec<_>)> = Vec::with_capacity(plan.len());
        for (b, missing) in plan {
            let mut cws = Vec::with_capacity(missing);
            for _ in 0..missing {
                cws.push(self.alloc_block_chunks(0, 0)?);
            }
            prealloc.push((b, cws));
        }
        let mut tail_spares: Vec<(usize, _)> = Vec::with_capacity(tail_maybe.len());
        for b in tail_maybe {
            tail_spares.push((b, self.alloc_block_chunks(0, 0)?));
        }

        // Fast path — see `ensure_for_batch_entries`.
        if prealloc.is_empty() && tail_spares.is_empty() {
            return Ok(());
        }

        {
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for (b, cws) in prealloc {
                let Some(slot) = state.sequences[b].as_mut() else {
                    continue;
                };
                for cw in cws {
                    // Seal the current write target to full capacity. We are
                    // allocating a new block because the write range extends past
                    // the previous last block; the kernel will fill every remaining
                    // position in that block. The seal must stay immediately before
                    // its push.
                    if let Some(last) = slot.last_chunk_mut() {
                        let cur_offset = last.offset;
                        let capacity = chunk_size - cur_offset as usize;
                        last.usage = capacity as u32;
                    }
                    slot.push_chunk(cw);
                }
            }
            // Writable-tail pass, same guard.
            for (b, cw) in tail_spares {
                if state.sequences[b].is_some() && self.tail_needs_new_block(&state, b) {
                    let slot = state.sequences[b].as_mut().unwrap();
                    slot.push_chunk(cw);
                }
            }
        }

        Ok(())
    }

    pub fn ensure_for_batch_entries(&self, entries: &[(usize, usize)], add: usize) -> Result<()> {
        if entries.is_empty() || add == 0 {
            return Ok(());
        }

        let batch = self.batch_capacity();
        for &(batch_idx, _off) in entries.iter() {
            if batch_idx >= batch {
                candle::bail!(
                    "batch_idx {} out of range for chunked backing (capacity {})",
                    batch_idx,
                    batch
                )
            }
        }

        let chunk_size = CHUNK_SIZE;

        // Under cum_token addressing the slot's `state.offset` is the
        // sum of slice.usage — NOT chunk_count × CHUNK_SIZE.  We can't
        // use positional math (`(offset + add) / CHUNK_SIZE`) to count
        // needed chunks because partial-tail slices (from injected
        // prefix sections) make positional and cum_token indices
        // diverge.  Instead, compute how many additional chunks each
        // slot needs based on the actual remaining capacity in
        // existing chunks starting at the first empty (or last
        // partial) slice — the same `write_slice` rule used in
        // `slot_state.rs`.
        let mut alloc_plan: Vec<(usize, usize)> = Vec::with_capacity(entries.len());
        let mut tail_maybe: Vec<usize> = Vec::new();
        let mut required_max_blocks = 1usize;
        {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for &(batch_idx, _off) in entries.iter() {
                let (current_chunks, available) =
                    match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                        Some(slot) => {
                            let chunks = slot.chunks_slice();
                            // Writer-owned capacity ONLY: chunks before
                            // `writer_start_idx` are Arc-shared/sealed, and a
                            // partial sealed tail is a GAP — its free slots
                            // are dead, never a write target. When the
                            // boundary sits past the last chunk (a freshly
                            // injected prefix), available is ZERO; clamping
                            // into the sealed tail counts its gap as writer
                            // capacity and under-allocates the write region
                            // by up to one chunk.
                            let start = slot.writer_start_idx();
                            let avail: usize = if start >= chunks.len() {
                                0
                            } else {
                                chunks[start..]
                                    .iter()
                                    .map(|c| chunk_size - (c.offset as usize + c.usage as usize))
                                    .sum()
                            };
                            (chunks.len(), avail)
                        }
                        None => (0usize, 0usize),
                    };
                let needed_extra = add.saturating_sub(available);
                let additional_chunks = needed_extra.div_ceil(chunk_size);
                let new_total_chunks = current_chunks + additional_chunks;
                required_max_blocks = cmp::max(required_max_blocks, new_total_chunks.max(1));
                alloc_plan.push((batch_idx, additional_chunks));
                // Predict the tail need in the SAME read pass — this runs once per
                // layer per decode step, so a second guard acquisition here is
                // pure overhead on the hot path.
                if self.tail_needs_new_block(&state, batch_idx) {
                    tail_maybe.push(batch_idx);
                }
            }
        }
        self.ensure_max_blocks(required_max_blocks)?;

        // Allocate BEFORE taking the state guard.
        //
        // `alloc_block_chunks` can reach `request_global_compact`, and compaction
        // needs a write guard on EVERY layer's block table to remap relocated
        // GIDs. Allocating while holding one of those guards made that a
        // self-deadlock; once the compactor was made non-blocking it became a
        // permanent no-op instead, so arena compaction could never run from the
        // path that needs it most. The chunk counts are already known from
        // `alloc_plan`, so the allocation needs no state access at all.
        let mut prealloc: Vec<(usize, Vec<_>)> = Vec::with_capacity(alloc_plan.len());
        for (batch_idx, additional_chunks) in alloc_plan {
            let mut cws = Vec::with_capacity(additional_chunks);
            for _ in 0..additional_chunks {
                cws.push(self.alloc_block_chunks(0, 0)?);
            }
            prealloc.push((batch_idx, cws));
        }

        // Predict which tails will need a fresh block, so their chunks can be
        // allocated alongside the rest and ALL mutation can happen under ONE
        // guard. The prediction is a safe over-estimate: installing a block makes
        // that sequence's tail fresh and therefore writable, so the block installs
        // below can only ever *reduce* this set, never grow it. Spares that turn
        // out to be unnecessary simply drop, returning their GIDs to the pool.
        let mut tail_spares: Vec<(usize, _)> = Vec::with_capacity(tail_maybe.len());
        for batch_idx in tail_maybe {
            tail_spares.push((batch_idx, self.alloc_block_chunks(0, 0)?));
        }

        // FAST PATH. This runs 48x per decode step (once per layer), and on a
        // normal step the block is not full and the tail is still writable, so
        // there is nothing to install. Returning before the write guard keeps the
        // steady-state decode cost at one read guard, as it was before the
        // allocation was hoisted out of the mutation guard.
        if prealloc.is_empty() && tail_spares.is_empty() {
            return Ok(());
        }

        // Single guard for every mutation. Extending a sequence and making its
        // tail writable must be ATOMIC: a reader that observes blocks pushed but
        // the tail not yet replaced would write into a full or closed-quant chunk.
        {
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for (batch_idx, cws) in prealloc {
                // Auto-allocate slot if needed (mirrors ensure_for_offset behavior).
                if state.sequences[batch_idx].is_none() {
                    state.sequences[batch_idx] = Some(self.make_sequence_state()?);
                }
                let slot = state.sequences[batch_idx].as_mut().unwrap();
                // Each push marks the slot buffer appended-to, so its next sync
                // serialises the new tail rather than the whole table.
                for cw in cws {
                    slot.push_chunk(cw);
                }
            }

            // Writable-tail pass, still under the same guard.
            for (batch_idx, cw) in tail_spares {
                if self.tail_needs_new_block(&state, batch_idx) {
                    let slot = state.sequences[batch_idx].as_mut().unwrap();
                    slot.push_chunk(cw);
                }
            }
        }

        Ok(())
    }

    /// Whether `batch_idx`'s tail block can still be written into, or a fresh
    /// block must be pushed. Pure read over state + arena storage; extracted so
    /// the decision can be made under a guard while the allocation it implies
    /// happens outside one.
    ///
    /// **A tail below the writer boundary is sealed, whatever room it has.**
    /// `inject_sealed_at_tail` puts every injected chunk under the boundary,
    /// partial tail included; a write that then lands by position inside that
    /// partial chunk bumps a sealed chunk's usage, and the rollback that should
    /// undo it (`truncate_sequence_to_tokens`) clamps at the sealed boundary
    /// and cannot. Measured on a resumed decode lease: the drafter's two rows
    /// went into a 27-token sealed tail, the head layer read 2050 tokens
    /// against an offset of 2048, and the header build refused the slot on
    /// every wave it joined. So a sealed tail needs a fresh block exactly as a
    /// full or a block-quantized one does.
    pub(super) fn tail_needs_new_block(
        &self,
        state: &super::super::types::BlockTableState,
        batch_idx: usize,
    ) -> bool {
        let needs: Option<bool> = state.sequences[batch_idx].as_ref().and_then(|s| {
            let cw = s.last_chunk()?;
            debug_assert!(
                cw.gids.iter().all(|g| g.strong_count() <= cw.gids.len()),
                "tail block must not be shared — fork should have copied it"
            );
            let sealed = s.block_count() <= s.writer_start_idx();
            let is_full = (cw.offset as usize + cw.usage as usize) >= CHUNK_SIZE;
            if sealed || is_full {
                Some(true)
            } else {
                // A block-quantized band cannot take a partial-token append —
                // writes have to land on whole blocks — so a tail that has
                // already been compressed needs a fresh block behind it.
                //
                // The question is asked of the chunk's own band tags, not of
                // the arenas its gids point into: a size-class arena holds
                // whatever fits its stride, so "is this arena quantized" has no
                // answer (`docs/archived/arena_unification.md` principle 8). R16 is
                // excluded because it is the raw active-K capture format, not
                // compression — the writer chunk's own K side is R16 and must
                // stay writable.
                let quantized_band = |tags: &[u8]| {
                    tags.iter()
                        .copied()
                        .map(ArenaFormatTag::from_u8)
                        .any(|t| t.is_quantized() && t != ArenaFormatTag::R16)
                };
                if quantized_band(&cw.k_fmt) || quantized_band(&cw.v_fmt) {
                    Some(true)
                } else {
                    None
                }
            }
        });
        matches!(needs, Some(true))
    }

    pub fn ensure_for_offset(&self, batch_idx: usize, offset: usize, add: usize) -> Result<()> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }
        if add == 0 {
            return Ok(());
        }

        let end_pos = offset.saturating_add(add).saturating_sub(1);
        let need_blocks = (end_pos / CHUNK_SIZE) + 1;
        self.ensure_max_blocks(need_blocks)?;

        // Count first, allocate WITHOUT the guard, then install. `alloc_block_chunks`
        // can reach `request_global_compact`, which needs a write guard on every
        // layer's block table; allocating under one made that a self-deadlock.
        // See `ensure_for_batch_entries` for the full rationale.
        // Count blocks AND predict the tail need in one read pass, allocate both
        // without a guard, then mutate under ONE write guard. Extending a sequence
        // and making its tail writable must be atomic: a reader that saw blocks
        // pushed but the tail not yet replaced would write into a full or
        // closed-quant chunk. Predicting the tail before the installs is a safe
        // over-estimate — a freshly pushed block is itself writable, so installs
        // can only shrink the need. An unused spare drops, returning its GIDs.
        let (missing, maybe_tail) = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let have = state.sequences[batch_idx]
                .as_ref()
                .map(|s| {
                    (0..need_blocks)
                        .filter(|&b| s.chunk_at(b).is_some())
                        .count()
                })
                .unwrap_or(0);
            (
                need_blocks.saturating_sub(have),
                self.tail_needs_new_block(&state, batch_idx),
            )
        };
        let mut fresh = Vec::with_capacity(missing);
        for _ in 0..missing {
            fresh.push(self.alloc_block_chunks(0, 0)?);
        }
        let tail_spare = if maybe_tail {
            Some(self.alloc_block_chunks(0, 0)?)
        } else {
            None
        };

        // Fast path — see `ensure_for_batch_entries`.
        if fresh.is_empty() && tail_spare.is_none() {
            return Ok(());
        }

        {
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            // Auto-allocate slot if needed
            if state.sequences[batch_idx].is_none() {
                state.sequences[batch_idx] = Some(self.make_sequence_state()?);
            }
            {
                let slot = state.sequences[batch_idx].as_mut().unwrap();
                // Under cum_token addressing we never bump the previous tail's
                // usage when allocating a new chunk. See `ensure_for_batch_entries`.
                for cw in fresh {
                    if (0..need_blocks).any(|b| slot.chunk_at(b).is_none()) {
                        slot.push_chunk(cw);
                    }
                }
            }
            // Writable-tail pass, same guard.
            if let Some(cw) = tail_spare {
                if self.tail_needs_new_block(&state, batch_idx) {
                    let slot = state.sequences[batch_idx].as_mut().unwrap();
                    slot.push_chunk(cw);
                }
            }
        }

        Ok(())
    }

    /// Allocate writer capacity for a write of logical tokens `offset..offset +
    /// add`, by the placement rule `write_contiguous` writes them by
    /// (`write_placement`): positions the sequence already holds need nothing,
    /// the rest need writer capacity from the writer boundary.
    pub(in super::super) fn ensure_for_append(
        &self,
        batch_idx: usize,
        offset: usize,
        add: usize,
    ) -> Result<()> {
        let appended = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            appended_tokens(state.sequences[batch_idx].as_ref(), offset, add)
        };
        self.ensure_for_batch_entries(&[(batch_idx, offset)], appended)
    }
}
