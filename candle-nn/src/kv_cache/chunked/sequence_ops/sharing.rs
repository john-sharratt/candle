//! Shares prefix chunks between sequences and appends borrowed chunks, without
//! copying data.
//!
//! `share_prefix` makes the target's prefix the source's chunks, cloned by Arc,
//! and drains the target's own prefix, which is never the target's to write.
//! `inject_prefix_chunks` and `append_borrowed_chunks_cow` take the palette and
//! scale metadata of their chunks from the donor that owns the first matched
//! identifier, and fall back to identity values if no donor exists.
//! `ensure_block_writable` performs copy-on-write under the write lock, although
//! the read-only projection model keeps a writer's block unshared already.

use super::*;

impl ChunkedKvBacking {
    /// Append borrowed chunk references to an existing sequence's block table.
    ///
    /// Unlike [`inject_prefix_chunks`] (which resets a fresh, empty slot),
    /// this method extends an existing slot that already has content in its
    /// KV cache. Used for mid-sequence boundary injection.
    ///
    /// # Differences from `inject_prefix_chunks`
    /// - Starts writing at `slot.block_count` (not at 0)
    /// - Does **not** change `block_usage` (caller sets validity)
    /// - Advances `block_count` and the backing's length tracking by `token_count`
    ///
    /// RoPE positions for borrowed blocks are computed at read time as
    /// `blk * chunk_size` (dense sequential from the slot's natural layout).
    ///
    /// The caller is responsible for advancing the session offset afterwards
    /// via `advance_sequence(batch_idx, token_count)`.
    ///
    /// # Arguments
    /// * `batch_idx` - Target slot (must already be allocated)
    /// * `chunk_ids` - Per-block GID vectors to append to the block table, in
    ///   order. Each `HeadGids` has length `2 * n_kv_head`.
    /// * `token_count` - Logical token count of the appended boundary
    pub fn append_borrowed_chunks_cow(
        &self,
        batch_idx: usize,
        chunk_ids: &[HeadGids],
        _token_count: usize,
    ) -> Result<()> {
        // Read current block_count and max_blocks under a read lock first.
        let (start_block, current_max) = {
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
                        "append_borrowed_chunks_cow: batch_idx {} is not allocated",
                        batch_idx
                    ))
                })?;
            (slot.block_count(), state.max_blocks)
        };

        let need_blocks = start_block + chunk_ids.len();

        // Grow max_blocks if needed (drop+re-acquire write lock).
        if need_blocks > current_max {
            self.ensure_max_blocks(need_blocks)?;
        }

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        // Re-verify slot is still allocated after potential lock drop.
        if state
            .sequences
            .get(batch_idx)
            .and_then(|s| s.as_ref())
            .is_none()
        {
            candle::bail!(
                "append_borrowed_chunks_cow: batch_idx {} is not allocated",
                batch_idx
            );
        }

        // Resolve donor pal/scale from the source ChunkWindow that owns the
        // first gid of each borrowed block, so the new entries decode the
        // shared arena bytes correctly. Falls back to identity/unity if the
        // donor chunk can't be found (defensive — shouldn't happen).
        let n = chunk_ids.len();
        struct DonorMeta {
            k_pal: Arc<Vec<u8>>,
            v_pal: Arc<Vec<u8>>,
            k_scale: Arc<Vec<f32>>,
            v_scale: Arc<Vec<f32>>,
            k_fmt: Arc<Vec<u8>>,
            v_fmt: Arc<Vec<u8>>,
            q_sign: Option<QSign>,
        }
        let donors: Vec<DonorMeta> = chunk_ids
            .iter()
            .map(|block_ids| {
                let first_raw = block_ids.iter().next().map(|g| g.raw()).unwrap_or(-1);
                let mut donor = None;
                'search: for seq in state.sequences.iter().flatten() {
                    for cw in seq.chunks_slice().iter() {
                        if cw.gids.iter().any(|g| g.raw() == first_raw) {
                            donor = Some(DonorMeta {
                                k_pal: cw.k_pal.clone(),
                                v_pal: cw.v_pal.clone(),
                                k_scale: cw.k_scale.clone(),
                                v_scale: cw.v_scale.clone(),
                                k_fmt: cw.k_fmt.clone(),
                                v_fmt: cw.v_fmt.clone(),
                                q_sign: cw.q_sign.clone(),
                            });
                            break 'search;
                        }
                    }
                }
                donor.unwrap_or_else(|| DonorMeta {
                    k_pal: self.inner.identity_pal.clone(),
                    v_pal: self.inner.identity_pal.clone(),
                    k_scale: self.inner.identity_scale.clone(),
                    v_scale: self.inner.identity_scale.clone(),
                    // No donor found: the block is being borrowed without a
                    // live chunk describing it, so fall back to the active
                    // formats — the same default the identity pal/scale
                    // represent for their fields.
                    k_fmt: self.inner.active_k_fmt.clone(),
                    v_fmt: self.inner.active_v_fmt.clone(),
                    q_sign: None,
                })
            })
            .collect();

        // Update slot: append blocks with their full per-head GID vectors.
        let slot = state.sequences[batch_idx].as_mut().unwrap();

        // All borrowed blocks go into the flat chunks vec.
        // Full blocks (all but last) get usage = CHUNK_SIZE.
        // Last appended block gets usage = 0 (updated later by set_len).
        for i in 0..n.saturating_sub(1) {
            slot.push_chunk(ChunkWindow {
                gids: chunk_ids[i].clone(),
                usage: CHUNK_SIZE as u32,
                offset: 0,
                k_pal: donors[i].k_pal.clone(),
                v_pal: donors[i].v_pal.clone(),
                k_scale: donors[i].k_scale.clone(),
                v_scale: donors[i].v_scale.clone(),
                k_fmt: donors[i].k_fmt.clone(),
                v_fmt: donors[i].v_fmt.clone(),
                meta: None,
                q_sign: donors[i].q_sign.clone(),
            });
        }
        if n > 0 {
            slot.push_chunk(ChunkWindow {
                gids: chunk_ids[n - 1].clone(),
                usage: 0,
                offset: 0,
                k_pal: donors[n - 1].k_pal.clone(),
                v_pal: donors[n - 1].v_pal.clone(),
                k_scale: donors[n - 1].k_scale.clone(),
                v_scale: donors[n - 1].v_scale.clone(),
                k_fmt: donors[n - 1].k_fmt.clone(),
                v_fmt: donors[n - 1].v_fmt.clone(),
                meta: None,
                q_sign: donors[n - 1].q_sign.clone(),
            });
        }

        drop(state);

        Ok(())
    }

    /// Inject pre-existing chunk IDs into a sequence slot's block table.
    ///
    /// Populates a borrower's block table with chunk IDs cloned from a
    /// source slot.  Each `chunk_id` gets a `ChunkGid` entry cloned
    /// from the source.
    ///
    /// This does NOT allocate new chunks — it references existing ones via
    /// Arc-clone of handles looked up from the source.  The caller must ensure
    /// the chunk IDs are valid and owned by a prototype slot that outlives
    /// all borrowers.
    ///
    /// After injection, blocks `0..n-2` are committed to `recorded_metas` as
    /// full blocks (offset=0, usage=chunk_size, sequential rope positions).
    /// Block `n-1` is set as the *active* block with `active_chunk_offset=0`;
    /// its usage is computed
    /// dynamically by `chunk_meta_row(seq_len)`.
    ///
    /// # Arguments
    /// * `batch_idx` - Target slot (must already be allocated)
    /// * `chunk_ids` - Per-block GID vectors to write into block table, in
    ///   order. Each `HeadGids` has length `2 * n_kv_head`.
    /// * `seq_len` - Real token count of the injected prefix
    pub fn inject_prefix_chunks(
        &self,
        batch_idx: usize,
        chunk_ids: &[HeadGids],
        seq_len: usize,
    ) -> Result<()> {
        let chunk_size = CHUNK_SIZE;

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        // Ensure enough blocks
        let need_blocks = chunk_ids.len();
        if need_blocks > state.max_blocks {
            drop(state);
            self.ensure_max_blocks(need_blocks)?;
            state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        }

        // Verify target slot is allocated
        if state
            .sequences
            .get(batch_idx)
            .and_then(|s| s.as_ref())
            .is_none()
        {
            candle::bail!(
                "inject_prefix_chunks: batch_idx {} is not allocated",
                batch_idx
            );
        }

        // Phase 1: For each block's gid vector, find the matching live GIDs
        // in existing slots and clone them (bumps Arc refcount). Also capture
        // the source ChunkWindow's pal_map / outer scale so the new chunk
        // decodes the same arena bytes correctly. We use the source chunk
        // that owns the FIRST gid of each block as the canonical donor.
        struct ResolvedBlock {
            gids: HeadGids,
            k_pal: Arc<Vec<u8>>,
            v_pal: Arc<Vec<u8>>,
            k_scale: Arc<Vec<f32>>,
            v_scale: Arc<Vec<f32>>,
            k_fmt: Arc<Vec<u8>>,
            v_fmt: Arc<Vec<u8>>,
            q_sign: Option<QSign>,
        }
        let mut resolved_blocks: Vec<ResolvedBlock> = Vec::with_capacity(need_blocks);
        for block_ids in chunk_ids.iter() {
            let mut resolved = Vec::with_capacity(block_ids.len());
            // Donor metadata: pal/scale from the source ChunkWindow that owns
            // the first matched gid. Defaults to identity/unity if the donor
            // chunk can't be found (defensive fallback — shouldn't happen).
            let mut donor_k_pal: Option<Arc<Vec<u8>>> = None;
            let mut donor_v_pal: Option<Arc<Vec<u8>>> = None;
            let mut donor_k_scale: Option<Arc<Vec<f32>>> = None;
            let mut donor_v_scale: Option<Arc<Vec<f32>>> = None;
            let mut donor_k_fmt: Option<Arc<Vec<u8>>> = None;
            let mut donor_v_fmt: Option<Arc<Vec<u8>>> = None;
            let mut donor_q_sign: Option<QSign> = None;
            for (idx, chunk_id) in block_ids.iter().enumerate() {
                let raw = chunk_id.raw();
                // Walk every live slot's chunks once to find the donor
                // ChunkWindow. We capture both the cloned gid and (only on the
                // first iteration) the donor's pal/scale.
                let mut found_gid = None;
                'search: for seq in state.sequences.iter().flatten() {
                    for cw in seq.chunks_slice().iter() {
                        if let Some(g) = cw.gids.iter().find(|g| g.raw() == raw) {
                            found_gid = Some(g.clone());
                            if idx == 0 {
                                donor_k_pal = Some(cw.k_pal.clone());
                                donor_v_pal = Some(cw.v_pal.clone());
                                donor_k_scale = Some(cw.k_scale.clone());
                                donor_v_scale = Some(cw.v_scale.clone());
                                donor_k_fmt = Some(cw.k_fmt.clone());
                                donor_v_fmt = Some(cw.v_fmt.clone());
                                donor_q_sign = cw.q_sign.clone();
                            }
                            break 'search;
                        }
                    }
                }
                let gid = found_gid.ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "inject_prefix_chunks: chunk_id {} not found in any slot",
                        raw
                    ))
                })?;
                resolved.push(gid);
            }
            resolved_blocks.push(ResolvedBlock {
                gids: HeadGids::from_vec(resolved),
                k_pal: donor_k_pal.unwrap_or_else(|| self.inner.identity_pal.clone()),
                v_pal: donor_v_pal.unwrap_or_else(|| self.inner.identity_pal.clone()),
                k_scale: donor_k_scale.unwrap_or_else(|| self.inner.identity_scale.clone()),
                v_scale: donor_v_scale.unwrap_or_else(|| self.inner.identity_scale.clone()),
                k_fmt: donor_k_fmt.unwrap_or_else(|| self.inner.active_k_fmt.clone()),
                v_fmt: donor_v_fmt.unwrap_or_else(|| self.inner.active_v_fmt.clone()),
                q_sign: donor_q_sign,
            });
        }

        // Phase 2: Write into the target slot.
        let slot = state.sequences[batch_idx].as_mut().unwrap();
        slot.clear_chunks();

        // All blocks go into the flat chunks vec.
        // Full blocks 0..n-2 get usage = chunk_size.
        // Last block (n-1) gets the partial usage derived from seq_len.
        for rb in resolved_blocks.iter().take(need_blocks.saturating_sub(1)) {
            slot.push_chunk(ChunkWindow {
                gids: rb.gids.clone(),
                usage: chunk_size as u32,
                offset: 0,
                k_pal: rb.k_pal.clone(),
                v_pal: rb.v_pal.clone(),
                k_scale: rb.k_scale.clone(),
                v_scale: rb.v_scale.clone(),
                k_fmt: rb.k_fmt.clone(),
                v_fmt: rb.v_fmt.clone(),
                meta: None,
                q_sign: rb.q_sign.clone(),
            });
        }
        if need_blocks > 0 {
            let last_usage = seq_len.saturating_sub((need_blocks - 1) * chunk_size);
            let rb = &resolved_blocks[need_blocks - 1];
            slot.push_chunk(ChunkWindow {
                gids: rb.gids.clone(),
                usage: last_usage as u32,
                offset: 0,
                k_pal: rb.k_pal.clone(),
                v_pal: rb.v_pal.clone(),
                k_scale: rb.k_scale.clone(),
                v_scale: rb.v_scale.clone(),
                k_fmt: rb.k_fmt.clone(),
                v_fmt: rb.v_fmt.clone(),
                meta: None,
                q_sign: rb.q_sign.clone(),
            });
        }

        Ok(())
    }

    /// Share prefix chunks from source sequence to target sequence.
    ///
    /// After this call, both sequences will reference the same chunk storage
    /// for the first `floor(prefix_tokens / chunk_size)` blocks. No data copy.
    ///
    /// The target sequence's existing blocks in the prefix region are freed
    /// (if solely owned) before sharing.
    ///
    /// Note: Freed target chunks go to target's free pool, which may affect
    /// allocation locality if chunks were originally allocated elsewhere.
    ///
    /// Returns the actual number of tokens covered by shared blocks (floored to chunk boundary).
    pub fn share_prefix(
        &self,
        source_batch: usize,
        target_batch: usize,
        prefix_tokens: usize,
    ) -> Result<usize> {
        let batch = self.batch_capacity();
        if source_batch >= batch {
            candle::bail!(
                "source_batch {} out of range for chunked backing (capacity {})",
                source_batch,
                batch
            )
        }
        if target_batch >= batch {
            candle::bail!(
                "target_batch {} out of range for chunked backing (capacity {})",
                target_batch,
                batch
            )
        }
        if source_batch == target_batch {
            candle::bail!("cannot share prefix with self")
        }

        let num_blocks = prefix_tokens / CHUNK_SIZE;
        if num_blocks == 0 {
            return Ok(0);
        }

        self.ensure_max_blocks(num_blocks)?;

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        // Validate source slot is allocated and has enough blocks
        let source_slot = state.sequences[source_batch].as_ref().ok_or_else(|| {
            candle::Error::Msg(format!("source slot {} not allocated", source_batch))
        })?;
        if num_blocks > source_slot.block_count() {
            candle::bail!(
                "source sequence {} only has {} blocks allocated, cannot share {} blocks",
                source_batch,
                source_slot.block_count(),
                num_blocks
            )
        }

        // Validate all source blocks in range are actually allocated
        for blk in 0..num_blocks {
            if source_slot.chunk_at(blk).is_none() {
                candle::bail!(
                    "source sequence {} block {} is not allocated",
                    source_batch,
                    blk
                )
            }
        }

        // Ensure target slot is allocated
        if state.sequences[target_batch].is_none() {
            state.sequences[target_batch] = Some(self.make_sequence_state()?);
        }

        // Free target's existing blocks in the prefix region (if solely owned)
        let drain_count = {
            let target_slot = state.sequences[target_batch].as_mut().unwrap();
            // Remove committed chunks in 0..num_blocks (drain from front)
            let drain_count = num_blocks.min(target_slot.block_count());
            // Drained GIDs are dropped here via RAII, returning to pool
            target_slot.drain_front_chunks(drain_count);
            drain_count
        };

        // Share source's chunks with target (clone increments Gid refcount)
        // We need to insert the shared chunks at the front
        {
            let source_slot = state.sequences[source_batch].as_ref().unwrap();
            let source_chunks: Vec<ChunkWindow> = (0..num_blocks)
                .map(|blk| {
                    let cw = source_slot.chunk_at(blk).unwrap();
                    ChunkWindow {
                        gids: cw.gids.clone(),
                        usage: CHUNK_SIZE as u32,
                        offset: 0,
                        k_pal: cw.k_pal.clone(),
                        v_pal: cw.v_pal.clone(),
                        k_scale: cw.k_scale.clone(),
                        v_scale: cw.v_scale.clone(),
                        k_fmt: cw.k_fmt.clone(),
                        v_fmt: cw.v_fmt.clone(),
                        // Fork shares the source chunk's GIDs (Arc bump), so its
                        // resident record's pointers stay valid — share it too.
                        meta: cw.meta.clone(),
                        q_sign: cw.q_sign.clone(),
                    }
                })
                .collect();
            let target_slot = state.sequences[target_batch].as_mut().unwrap();
            // Prepend shared chunks before any remaining target chunks
            target_slot.prepend_chunks(source_chunks);
            // The shared prefix is the source's sealed history: never the
            // target's to write. The target's own writer region keeps its place
            // behind it, moved by the prefix in and the front drained out.
            let own_writer = target_slot
                .writer_start_idx()
                .saturating_sub(drain_count)
                .saturating_add(num_blocks);
            target_slot.set_writer_start_idx(own_writer.max(num_blocks));
        }

        drop(state);

        Ok(num_blocks * CHUNK_SIZE)
    }

    /// Check if a block is shared (referenced by multiple sequences).
    ///
    /// WARNING: The result is informational only. Do not use this to make
    /// decisions about whether to write - use `ensure_block_writable` instead,
    /// which atomically checks and performs COW under the write lock.
    /// Get the number of blocks currently allocated for a sequence.
    pub fn seq_blocks_count(&self, batch_idx: usize) -> Result<usize> {
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
        Ok(state.sequences[batch_idx]
            .as_ref()
            .map_or(0, |s| s.block_count()))
    }

    /// Ensure a block is writable (not shared).  Returns the block's per-head GIDs.
    /// This is the safe way to ensure a block can be written to without
    /// affecting other sequences that may share it.
    pub fn ensure_block_writable(&self, batch_idx: usize, block_idx: usize) -> Result<HeadGids> {
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

        let slot = state.sequences[batch_idx]
            .as_ref()
            .ok_or_else(|| candle::Error::Msg(format!("slot {} not allocated", batch_idx)))?;

        if block_idx >= slot.block_count() {
            candle::bail!("block_idx out of range")
        }

        let cw = slot
            .chunk_at(block_idx)
            .ok_or_else(|| candle::Error::Msg("block not allocated".into()))?;

        // The read-only projection model guarantees structurally that any
        // block reached by a writer is unshared: projection borrows parent
        // chunks read-only and pushes a fresh active chunk for writes, so
        // the tail's gids always have strong_count = 1.

        Ok(cw.gids.clone())
    }
}
