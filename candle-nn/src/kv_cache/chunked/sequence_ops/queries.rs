//! Reads the per-block metadata of a sequence slot: usage, RoPE positions, chunk
//! identifiers and chunk references.
//!
//! Each reader returns what the block table holds, and an unallocated slot is
//! reported as absent. `content_extent` reads the block count and the extent of
//! the content under one lock. `chunk_rope_positions` is a running sum over the
//! usage of the earlier blocks. `block_usage` reports the active block as zero,
//! because the exact length of the active block is known only to the caller.

use super::*;

impl ChunkedKvBacking {
    /// Get per-block valid token counts for a sequence slot.
    ///
    /// Returns a `Vec<u32>` of length `max_blocks`.  Each entry is the `usage`
    /// field from the corresponding `ChunkMeta` logical-block entry.  Unused
    /// logical-block positions return 0.
    ///
    /// The usage of the active block is returned as 0 (the exact value is only
    /// available if the caller supplies the current seq_len; use
    /// [`chunk_meta_row`] directly for full metadata).
    pub fn block_usage(&self, batch_idx: usize) -> Vec<u32> {
        let state = match self.state.read() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let max_blocks = state.max_blocks;
        let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
            Some(s) => s,
            None => return vec![0u32; max_blocks],
        };
        let mut result = vec![0u32; max_blocks];
        for (i, cw) in slot.chunks_slice().iter().enumerate() {
            if i < max_blocks {
                result[i] = cw.usage;
            }
        }
        result
    }

    /// A slot's block count and its content extent — the last block holding a
    /// token, plus one — read under one lock with nothing allocated: what
    /// [`Self::block_usage`] answers when all a caller wants is where the
    /// content ends. `None` when the slot is not allocated; an error when the
    /// slot holds more blocks than the backing's `max_blocks`, which is the
    /// sequence outgrowing the arena's capacity.
    pub fn content_extent(&self, batch_idx: usize) -> Result<Option<(usize, usize)>> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let Some(slot) = state.sequences.get(batch_idx).and_then(|s| s.as_ref()) else {
            return Ok(None);
        };
        let chunks = slot.chunks_slice();
        if chunks.len() > state.max_blocks {
            candle::bail!(
                "content_extent: seq {batch_idx} holds {} block(s), past the backing's \
                 {} max_blocks",
                chunks.len(),
                state.max_blocks
            );
        }
        let held = chunks
            .iter()
            .rposition(|c| c.usage != 0)
            .map_or(0, |i| i + 1);
        Ok(Some((chunks.len(), held)))
    }

    /// Count the chunks currently held by a sequence slot.
    ///
    /// This is the **authoritative** block-count for a slot.  Callers
    /// must use this rather than `sequence_offset.div_ceil(CHUNK_SIZE)`,
    /// which only matches when every chunk in the slot is full.  After
    /// `inject_sealed_at_tail` materialises sealed sections back-to-
    /// back, each section's trailing partial chunk stays a separate
    /// `ChunkWindow` — total token count divided by CHUNK_SIZE then
    /// *under-counts* the real chunk total, leaving the slot's tail
    /// invisible to anything that uses the divided value as a block
    /// range bound (e.g. `create_view_sequence`'s borrow ranges).
    ///
    /// Returns `None` if the slot is not allocated.
    pub fn sequence_block_count(&self, batch_idx: usize) -> Option<usize> {
        let state = self.state.read().ok()?;
        state
            .sequences
            .get(batch_idx)
            .and_then(|s| s.as_ref())
            .map(|s| s.block_count())
    }

    /// Get per-block canonical RoPE start positions for a sequence slot.
    ///
    /// Returns the `rope_base` field from each `ChunkMeta` logical-block entry.
    /// Unused positions return 0.
    pub fn chunk_rope_positions(&self, batch_idx: usize) -> Vec<i32> {
        let state = match self.state.read() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let max_blocks = state.max_blocks;
        let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
            Some(s) => s,
            None => return vec![0i32; max_blocks],
        };
        // One running sum: `rope_pos(i)` re-sums every chunk before `i`.
        let mut positions = vec![0i32; max_blocks];
        let mut at = slot.rope_pos(0);
        for (pos, cw) in positions.iter_mut().zip(slot.chunks_slice()) {
            *pos = at;
            at += cw.usage as i32;
        }
        positions
    }

    /// Read the per-head GID vectors for a sequence slot's block table.
    ///
    /// Returns one `Vec<ChunkGid>` per block (length = `2 * n_kv_head`),
    /// preserving per-head arena assignments.
    pub fn slot_chunk_ids(&self, batch_idx: usize) -> Result<Vec<HeadGids>> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let slot = state
            .sequences
            .get(batch_idx)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "slot_chunk_ids: batch_idx {} is not allocated",
                    batch_idx
                ))
            })?;
        let ids: Vec<HeadGids> = (0..slot.block_count())
            .map(|i| {
                slot.chunk_at(i)
                    .map(|cw| cw.gids.clone())
                    .unwrap_or_else(|| HeadGids::from_vec(Vec::new()))
            })
            .collect();
        Ok(ids)
    }

    /// Get chunk references for a sequence's blocks.
    ///
    /// Returns the full per-head `HeadGids` for each allocated block in the
    /// sequence.  These contain addressing info for attention kernels without
    /// holding tensor data directly.
    ///
    /// # Arguments
    /// * `batch_idx` - Sequence slot index
    /// * `block_range` - Range of blocks to get (None = all allocated blocks)
    pub fn get_chunk_refs(
        &self,
        batch_idx: usize,
        block_range: Option<Range<usize>>,
    ) -> Result<Vec<HeadGids>> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }

        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let slot = match state.sequences[batch_idx].as_ref() {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };

        let range = block_range.unwrap_or(0..slot.block_count());
        let end = range.end.min(slot.block_count());

        let mut refs = Vec::with_capacity(end.saturating_sub(range.start));
        for blk in range.start..end {
            if let Some(cw) = slot.chunk_at(blk) {
                refs.push(cw.gids.clone());
            }
        }

        Ok(refs)
    }

    /// Get chunk references for a sequence, optionally restricted to a block range.
    ///
    /// Rope positions are stored per-block in `ChunkWindow` (set at
    /// turn-boundary recording time) so no separate base parameter is needed.
    pub fn get_chunk_refs_with_rope(
        &self,
        batch_idx: usize,
        block_range: Option<Range<usize>>,
    ) -> Result<Vec<HeadGids>> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }

        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let slot = match state.sequences[batch_idx].as_ref() {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };

        let range = block_range.unwrap_or(0..slot.block_count());
        let end = range.end.min(slot.block_count());

        let mut refs = Vec::with_capacity(end.saturating_sub(range.start));
        for blk in range.start..end {
            if let Some(cw) = slot.chunk_at(blk) {
                refs.push(cw.gids.clone());
            }
        }

        Ok(refs)
    }
}
