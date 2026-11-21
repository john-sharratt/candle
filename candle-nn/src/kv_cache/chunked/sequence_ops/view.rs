//! Creates read-only view slots that borrow a parent's chunks, and returns the
//! tokens a view writes to its parent when it is finished.
//!
//! `borrow_into_view` clones the parent's chunk table by Arc, with no allocation
//! and no device call, so every layer's prefix can be borrowed at once. Writes
//! land in a fresh active chunk opened after the borrow, so a borrowed chunk is
//! never extended. `finalize_view` truncates the parent to the count it had
//! before the view, then moves the blocks the view wrote into it. A missing
//! parent chunk is reported as an error rather than replaced, since a sentinel
//! would count a phantom chunk of tokens.

use super::*;

impl ChunkedKvBacking {
    /// Create a view sequence that borrows blocks from a parent sequence.
    ///
    /// The view holds the blocks specified by `visible_block_ranges` (each range is
    /// `[start_blk, end_blk)` in parent coordinates) as its initial KV context.
    /// New tokens written to the view continue sequentially from the last borrowed
    /// parent block.
    ///
    /// `view_batch` must already be an allocated slot (via `alloc_sequence`).
    /// `parent_batch` must be an allocated slot.
    ///
    /// Returns `(borrowed_block_count, borrowed_token_count)`. All borrowed
    /// chunks are read-only Arc clones of parent's chunks; writes land in a
    /// fresh active chunk pushed after them. `borrowed_block_count` must be
    /// passed unchanged to [`finalize_view`].
    pub fn create_view_sequence(
        &self,
        view_batch: usize,
        parent_batch: usize,
        visible_block_ranges: &[(usize, usize)],
    ) -> Result<(usize, usize)> {
        let borrowed = self.borrow_into_view(view_batch, parent_batch, visible_block_ranges)?;
        if borrowed.0 > 0 {
            self.open_view_writer(view_batch)?;
        }
        Ok(borrowed)
    }

    /// The borrow half of [`Self::create_view_sequence`]: the view slot holds
    /// the parent's visible blocks and nothing else — no writer chunk yet, see
    /// [`Self::open_view_writer`].
    ///
    /// Host bookkeeping only — Arc clones of the parent's chunk table, no
    /// allocation and no device call — so a session borrows every layer's
    /// prefix at once rather than one layer after another.
    pub fn borrow_into_view(
        &self,
        view_batch: usize,
        parent_batch: usize,
        visible_block_ranges: &[(usize, usize)],
    ) -> Result<(usize, usize)> {
        let parent_blocks: Vec<usize> = visible_block_ranges
            .iter()
            .flat_map(|&(start, end)| start..end)
            .collect();
        let total_parent_blocks = parent_blocks.len();
        if total_parent_blocks == 0 {
            return Ok((0, 0));
        }

        self.ensure_max_blocks(total_parent_blocks)?;

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        // Validate parent
        let parent_slot = state
            .sequences
            .get(parent_batch)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "create_view_sequence: parent slot {} not allocated",
                    parent_batch
                ))
            })?;
        let max_parent_needed = parent_blocks.iter().copied().max().unwrap_or(0);
        if max_parent_needed >= parent_slot.block_count() {
            candle::bail!(
                "create_view_sequence: parent slot {} has {} blocks but range requests block {}",
                parent_batch,
                parent_slot.block_count(),
                max_parent_needed
            );
        }
        for &pb in &parent_blocks {
            if parent_slot.chunk_at(pb).is_none() {
                candle::bail!(
                    "create_view_sequence: parent slot {} block {} is not allocated",
                    parent_batch,
                    pb
                );
            }
        }

        // Validate view slot
        if state
            .sequences
            .get(view_batch)
            .and_then(|s| s.as_ref())
            .is_none()
        {
            candle::bail!(
                "create_view_sequence: view slot {} not allocated",
                view_batch
            );
        }

        // --- Collect borrowed-block metadata from parent (before mutating state) ---
        // Each entry carries the parent's gids + pal/scale state so a CoW share
        // points the view at the same arena bytes *and* preserves the routing
        // and outer scaling needed to decode them.
        let _parent_seq_len = state.sequences[parent_batch].as_ref().unwrap().seq_len();
        // No rope position is carried: a view's positions are its cumulative
        // usage, read at decode like any slot's. Computing one here per block
        // was a prefix sum per block — O(n²) over the borrow, 75 M additions per
        // layer at a 12K-block prefix, for a value nothing read.
        let borrowed_meta: Vec<ChunkWindow> = {
            let ps = state.sequences[parent_batch].as_ref().unwrap();
            let parent_len = ps.seq_len();
            parent_blocks
                .iter()
                .map(|&parent_blk| {
                    // A missing parent chunk is a HOLE in a live slot's block
                    // table — the parent lost backing between the caller's block
                    // enumeration and this borrow (eviction/truncate racing view
                    // creation). Substituting a sentinel here would count a full
                    // phantom chunk of tokens ("computed write len N is invalid"
                    // downstream) and hand the kernels a base-0 pointer
                    // (CUDA_ERROR_ILLEGAL_ADDRESS). Fail loudly with the exact
                    // hole instead.
                    let Some(cw) = ps.chunk_at(parent_blk) else {
                        candle::bail!(
                            "create_view_sequence: parent slot {parent_batch} block \
                             {parent_blk} has no chunk (parent seq_len {parent_len}, \
                             {} blocks borrowed) — live block table lost its backing",
                            parent_blocks.len()
                        );
                    };
                    // The view borrows the parent block's window whole: its GIDs by
                    // Arc clone — same physical placement — so the parent's
                    // resident record (if any) is valid for the view too, and the
                    // reproject glue reads it instead of rebuilding; its formats,
                    // scales and captured `sign(Q)` with them.
                    Ok(cw.clone())
                })
                .collect::<candle::Result<Vec<_>>>()?
        };
        let borrowed_token_count: usize = borrowed_meta.iter().map(|cw| cw.usage as usize).sum();

        // All projected chunks are read-only. The view borrows EVERY parent
        // block (including a partial tail) via Arc clone — no COW. New tokens
        // never extend a borrowed chunk; they always land in a fresh active
        // chunk pushed at the end of the borrow loop. This collapses two
        // older concerns at once:
        //   - no Q→R16 elevation needed for closed-quant partials,
        //   - the tail-is-shared debug class becomes structurally impossible.
        let borrowed_count = parent_blocks.len();

        // Clone parent blocks (Arc shared, read-only) into the view in one
        // replacement — whatever the slot held goes back to the pool via RAII.
        let chunks = borrowed_meta;
        let view = state.sequences[view_batch].as_mut().unwrap();
        view.replace_chunks(chunks);
        // **The borrow is immutable the moment it lands**, not once a writer
        // opens behind it. A slot whose write boundary still reads 0 claims
        // every borrowed block — sealed, record-bearing history it shares with
        // its parent — as its own writable region, which is exactly what
        // compaction must refuse to move. Set here, nothing between this borrow
        // and `open_view_writer` (which keeps the same boundary) can see it as
        // anything else.
        view.set_writer_start_idx(borrowed_count);
        drop(state);

        Ok((borrowed_count, borrowed_token_count))
    }

    /// Push the fresh active chunk a view's writes land in, past the prefix
    /// [`Self::borrow_into_view`] borrowed.
    ///
    /// Allocated in the active K/V arena keys (R16 K + F16 V on GPU) so
    /// decode/prefill kernels can write directly. Its gids have
    /// strong_count = 1, so the tail of the slot is unshared by construction.
    pub fn open_view_writer(&self, view_batch: usize) -> Result<()> {
        let active_cw = self.alloc_block_chunks(0, 0)?;
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let vs = state
            .sequences
            .get_mut(view_batch)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "open_view_writer: view slot {view_batch} not allocated"
                ))
            })?;
        vs.push_chunk(active_cw);
        // Writes start at the fresh active chunk, one past the borrowed prefix.
        let view_block_count = vs.block_count();
        vs.set_writer_start_idx(view_block_count.saturating_sub(1));
        Ok(())
    }

    /// Finish a view sequence and transfer its newly-written blocks to the parent.
    ///
    /// Blocks `original_view_block_count..view.block_count` (the blocks written during
    /// this turn) are moved from `view_batch` to `parent_batch` (appended after the
    /// parent's existing blocks).  The view slot is then freed.
    ///
    /// `original_view_block_count` is the value returned by [`create_view_sequence`].
    pub fn finalize_view(
        &self,
        view_batch: usize,
        parent_batch: usize,
        original_view_block_count: usize,
    ) -> Result<()> {
        // Read block counts under read lock to determine whether we need to grow
        let (view_block_count, parent_block_count) = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let vbc = state
                .sequences
                .get(view_batch)
                .and_then(|s| s.as_ref())
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "finalize_view: view slot {} not allocated",
                        view_batch
                    ))
                })?
                .block_count();
            let pbc = state
                .sequences
                .get(parent_batch)
                .and_then(|s| s.as_ref())
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "finalize_view: parent slot {} not allocated",
                        parent_batch
                    ))
                })?
                .block_count();
            (vbc, pbc)
        };

        if view_block_count < original_view_block_count {
            candle::bail!(
                "finalize_view: view slot {} has {} blocks, \
                 less than original_view_block_count {}",
                view_batch,
                view_block_count,
                original_view_block_count
            );
        }
        let new_count = view_block_count - original_view_block_count;
        if new_count == 0 {
            // Nothing was written; just free the view slot
            return self.free_sequence(view_batch);
        }

        self.ensure_max_blocks(parent_block_count + new_count)?;

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        // Re-read after potential grow
        let view_block_count = state.sequences[view_batch]
            .as_ref()
            .map_or(0, |s| s.block_count());
        let new_count = view_block_count.saturating_sub(original_view_block_count);
        if new_count == 0 {
            drop(state);
            return self.free_sequence(view_batch);
        }

        // Transfer new blocks from view to parent.
        // view.chunks[0..original_view_block_count] are the borrowed prefix (shared GIDs).
        // view.chunks[original_view_block_count..] are new/COW blocks written by the view.
        //
        // Truncate parent to original_view_block_count first: when the parent had a
        // partial tail (borrowed_count = full_blocks, not full_blocks+1), the partial
        // block is at parent.chunks[original_view_block_count] and was COW-extended by
        // the view.  Truncating removes the stale partial entry; extending adds the
        // updated version from the view.
        {
            let vs = state.sequences[view_batch].as_mut().unwrap();
            let new_blocks: Vec<ChunkWindow> = vs.split_off_chunks(original_view_block_count);

            let ps = state.sequences[parent_batch].as_mut().unwrap();
            ps.truncate_chunks(original_view_block_count);
            ps.extend_chunks(new_blocks);
        }

        // Free the view slot (borrowed prefix chunks dropped, decrementing Arc refcount)
        state.sequences[view_batch] = None;

        Ok(())
    }
}
