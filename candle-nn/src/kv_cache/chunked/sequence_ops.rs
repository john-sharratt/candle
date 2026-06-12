//! Sequence allocation and management operations for ChunkedKvBacking.
//!
//! This module contains methods for:
//! - Allocating and freeing sequences (slots)
//! - Sharing prefix blocks between sequences (COW)
//! - Forking sequences for beam search / speculative decoding
//! - Copy-on-write operations for shared blocks
//! - Block reference queries

use std::cmp;
use std::ops::Range;
use std::sync::Arc;

use candle::Result;

use super::head_gids::GIDS_PER_HEAD;
use crate::kv_cache::arena_table::N_PALETTE;
use crate::CHUNK_SIZE;

use super::gid_pool::ChunkGid;
use super::head_gids::HeadGids;
use super::types::{ChunkWindow, SealedChunk, SealedSequence};
use super::{Arena, BlockTableState, ChunkedKvBacking, SequenceState};

impl ChunkedKvBacking {
    /// Create a new [`SequenceState`] bound to this backing's device stream.
    ///
    /// Under the `cuda` feature the state receives a clone of the device's
    /// `CudaStream` so that async H->D copies can be issued from the guard.
    #[inline]
    pub(super) fn make_sequence_state(&self) -> Result<SequenceState> {
        #[cfg(feature = "cuda")]
        {
            let stream = match &self.inner.device {
                candle::Device::Cuda(dev) => Some(dev.cuda_stream()),
                _ => None,
            };
            Ok(SequenceState::new(stream))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(SequenceState::new())
        }
    }
}

impl ChunkedKvBacking {
    /// Return the first unallocated slot index in this backing.
    ///
    /// Scans `state.slots` and returns the index of the first `None` entry,
    /// or `state.slots.len()` when all current slots are occupied (caller must
    /// grow capacity).  This is the backing-level slot allocator used when
    /// multiple sessions share the same `BackingInner` pool â€” consulting the
    /// backing ensures sibling sessions get non-overlapping slot indices.
    pub fn first_free_slot(&self) -> Result<usize> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        Ok(state
            .sequences
            .iter()
            .position(|s| s.is_none())
            .unwrap_or(state.sequences.len()))
    }

    /// Allocate a new sequence slot.
    ///
    /// Returns a batch index that can be used with other methods.
    /// Automatically grows capacity if all slots are in use.
    pub fn alloc_sequence(&self) -> Result<usize> {
        // First try to find a free slot
        {
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

            for (idx, slot) in state.sequences.iter_mut().enumerate() {
                if slot.is_none() {
                    *slot = Some(self.make_sequence_state()?);
                    return Ok(idx);
                }
            }
        }

        // No free slot - grow capacity and allocate
        let old_capacity = self.batch_capacity();
        let new_capacity = cmp::max(old_capacity * 2, old_capacity + 1);
        self.grow_batch_capacity(new_capacity)?;

        // Now allocate in the newly available slot
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        state.sequences[old_capacity] = Some(self.make_sequence_state()?);
        Ok(old_capacity)
    }

    /// Ensure a specific sequence slot is allocated.
    ///
    /// This is used when externally managing batch indices (e.g., when migrating
    /// contiguous caches to chunked backing at specific slot indices).
    /// If the slot is already allocated, this is a no-op.
    /// Grows capacity if batch_idx >= current capacity.
    pub fn ensure_sequence_allocated(&self, batch_idx: usize) -> Result<()> {
        // Grow capacity if needed
        let current_capacity = self.batch_capacity();
        if batch_idx >= current_capacity {
            let new_capacity = cmp::max(batch_idx + 1, current_capacity * 2);
            self.grow_batch_capacity(new_capacity)?;
        }

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        if state.sequences[batch_idx].is_none() {
            state.sequences[batch_idx] = Some(self.make_sequence_state()?);
        }

        Ok(())
    }

    /// Free a sequence slot, returning it to the pool.
    ///
    /// This frees the underlying KV cache blocks (respecting COW sharing)
    /// and marks the slot as available for reuse.
    pub fn free_sequence(&self, batch_idx: usize) -> Result<()> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        // Take the slot (marks it as free)
        let slot = match state.sequences[batch_idx].take() {
            Some(s) => s,
            None => return Ok(()), // Already free
        };

        // GIDs in the slot are dropped here via RAII, returning to pool
        drop(slot);
        drop(state);
        Ok(())
    }

    pub fn set_len(&self, batch_idx: usize, len: usize) {
        if let Ok(mut state) = self.state.write() {
            if let Some(Some(seq)) = state.sequences.get_mut(batch_idx) {
                let n = seq.block_count();
                if n == 0 {
                    return;
                }
                // Under cum_token addressing a prefill of N tokens
                // fills consecutive chunks starting at the writer
                // boundary — the same selection rule the slot's
                // position_map and `ensure_for_batch_entries` use.
                // Chunks at index < writer_start_idx are Arc-shared
                // with substrate/parent and MUST NOT be modified.
                let prior_total: usize = seq.chunks_slice().iter().map(|c| c.usage as usize).sum();
                if len <= prior_total {
                    return;
                }
                let mut remaining = len - prior_total;
                let chunk_size = CHUNK_SIZE;
                let writer_start = seq.writer_start_idx();
                let mut idx = writer_start.min(n.saturating_sub(1));
                while remaining > 0 && idx < n {
                    let cap = {
                        let c = &seq.chunks_slice()[idx];
                        chunk_size - (c.offset as usize + c.usage as usize)
                    };
                    let take = remaining.min(cap);
                    seq.chunk_at_mut(idx).unwrap().usage += take as u32;
                    remaining -= take;
                    idx += 1;
                }
                // DO NOT call seq.patch_host_lens() here. patch_host_lens writes
                // directly to `buf` (the pinned host DMA source) outside of any
                // GpuChunksGuard. If a GpuChunksGuard::drop issued an async
                // memcpy_htod earlier in this step, the DMA engine may still be
                // reading `buf` when patch_host_lens writes to it — a host-memory
                // data race that causes ws_len in the GPU buffer to be incremented
                // by more than 1 per step, eventually triggering the
                // ws_offset+ws_len >= CHUNK_SIZE assertion in the decode kernel.
                // `buf` is always fully rewritten by rebuild_decode (REBUILD path)
                // or update_chunk (dirty-flush path) before any DMA, so keeping
                // `buf` manually in sync here is both redundant and unsafe.
            }
        }
    }

    /// Set one block's window geometry — its skip `offset` and valid token
    /// `usage` — directly.
    ///
    /// Used by the resume / cold-load path: after [`Self::alloc_sealed_block`]
    /// allocates a chunk's GIDs, the persisted `offset` / `token_count` must
    /// be stamped onto the `ChunkWindow` so [`Self::record_turn`] snapshots
    /// the correct window. Errors if the slot or block is not allocated.
    pub fn set_block_window(
        &self,
        batch_idx: usize,
        block_idx: usize,
        offset: u16,
        usage: u32,
    ) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let seq = state
            .sequences
            .get_mut(batch_idx)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| {
                candle::Error::Msg(format!("set_block_window: slot {batch_idx} not allocated"))
            })?;
        let cw = seq.chunk_at_mut(block_idx).ok_or_else(|| {
            candle::Error::Msg(format!(
                "set_block_window: block {block_idx} not allocated in slot {batch_idx}"
            ))
        })?;
        cw.offset = offset;
        cw.usage = usage;
        Ok(())
    }

    /// Test-only: set the decode writer-start block index for `batch_idx`.
    ///
    /// Blocks before `writer_start` are treated as sealed (never selected as the
    /// decode writer). Combined with [`Self::set_block_window`], this builds the
    /// substrate-seal gap — a partial sealed chunk (`usage < CHUNK_SIZE`) followed
    /// by a fresh writer chunk — that the decode-kernel gap-handling tests need.
    /// Invalidates the cached GPU slot buffer so the next decode rebuilds it.
    pub fn test_set_writer_start(&self, batch_idx: usize, writer_start: usize) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let seq = state
            .sequences
            .get_mut(batch_idx)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "test_set_writer_start: slot {batch_idx} not allocated"
                ))
            })?;
        seq.set_writer_start_idx(writer_start);
        seq.invalidate_gpu_chunks();
        Ok(())
    }

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
    /// * `chunk_ids` - Per-block GID vectors to append to the block table, in order.
    ///                 Each `HeadGids` has length `2 * n_kv_head`.
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
            });
        }

        drop(state);

        Ok(())
    }

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
        let mut positions = vec![0i32; max_blocks];
        for i in 0..slot.block_count().min(max_blocks) {
            positions[i] = slot.rope_pos(i);
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

    /// Inject pre-existing chunk IDs into a sequence slot's block table.
    ///
    /// Populates a borrower's block table with chunk IDs cloned from a
    /// source slot.  Each `chunk_id` gets a `ChunkGid` entry cloned
    /// from the source.
    ///
    /// This does NOT allocate new chunks â€” it references existing ones via
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
    /// * `chunk_ids` - Per-block GID vectors to write into block table, in order.
    ///                 Each `HeadGids` has length `2 * n_kv_head`.
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
            });
        }

        // Phase 2: Write into the target slot.
        let slot = state.sequences[batch_idx].as_mut().unwrap();
        slot.clear_chunks();

        // All blocks go into the flat chunks vec.
        // Full blocks 0..n-2 get usage = chunk_size.
        // Last block (n-1) gets the partial usage derived from seq_len.
        for i in 0..need_blocks.saturating_sub(1) {
            let rb = &resolved_blocks[i];
            slot.push_chunk(ChunkWindow {
                gids: rb.gids.clone(),
                usage: chunk_size as u32,
                offset: 0,
                k_pal: rb.k_pal.clone(),
                v_pal: rb.v_pal.clone(),
                k_scale: rb.k_scale.clone(),
                v_scale: rb.v_scale.clone(),
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
        {
            let target_slot = state.sequences[target_batch].as_mut().unwrap();
            // Remove committed chunks in 0..num_blocks (drain from front)
            let drain_count = num_blocks.min(target_slot.block_count());
            // Drained GIDs are dropped here via RAII, returning to pool
            target_slot.drain_front_chunks(drain_count);
        }

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
                    }
                })
                .collect();
            let target_slot = state.sequences[target_batch].as_mut().unwrap();
            // Prepend shared chunks before any remaining target chunks
            target_slot.prepend_chunks(source_chunks);
        }

        drop(state);

        Ok(num_blocks * CHUNK_SIZE)
    }

    /// Fork (clone) a sequence's KV cache to another sequence slot.
    ///
    /// This creates a copy of the source sequence's cache in the target slot:
    /// - Complete chunks are shared via COW (copy-on-write) - no data copy
    /// - The partial/remainder tokens in the last incomplete chunk are copied
    ///   into a new chunk allocated for the target
    ///
    /// This is useful for beam search, speculative decoding, or any scenario
    /// where you need to branch from an existing sequence state.
    ///
    /// # Arguments
    /// * `source_batch` - The source sequence to fork from
    /// * `target_batch` - The target sequence slot (will be overwritten)
    /// * `seq_len` - The number of tokens in the source sequence
    ///
    /// # Returns
    /// The number of tokens in the forked sequence (same as `seq_len`)
    pub fn fork_sequence(
        &self,
        source_batch: usize,
        target_batch: usize,
        seq_len: usize,
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
            candle::bail!("cannot fork sequence to itself")
        }
        if seq_len == 0 {
            // Just free target if any
            self.free_sequence(target_batch)?;
            return Ok(0);
        }

        let full_blocks = seq_len / CHUNK_SIZE;
        let remainder = seq_len % CHUNK_SIZE;
        let total_blocks = if remainder > 0 {
            full_blocks + 1
        } else {
            full_blocks
        };

        if total_blocks > 0 {
            self.ensure_max_blocks(total_blocks)?;
        }

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        // Validate source slot is allocated and has enough blocks
        let source_slot = state.sequences[source_batch].as_ref().ok_or_else(|| {
            candle::Error::Msg(format!("source slot {} not allocated", source_batch))
        })?;
        if total_blocks > source_slot.block_count() {
            candle::bail!(
                "source sequence {} only has {} blocks allocated, need {} for seq_len {}",
                source_batch,
                source_slot.block_count(),
                total_blocks,
                seq_len
            )
        }

        // Validate all needed source blocks are allocated
        for blk in 0..total_blocks {
            if source_slot.chunk_at(blk).is_none() {
                candle::bail!(
                    "source sequence {} block {} is not allocated",
                    source_batch,
                    blk
                )
            }
        }

        // Free target's existing blocks if slot exists
        // Collect chunks to free first to avoid borrow conflicts
        if let Some(ts) = state.sequences[target_batch].as_mut() {
            // Cleared GIDs are dropped via RAII, returning to pool
            ts.clear_chunks();
        }

        // Ensure target slot exists
        if state.sequences[target_batch].is_none() {
            state.sequences[target_batch] = Some(self.make_sequence_state()?);
        }

        // Share full blocks (COW) - clone increments Gid refcount
        {
            let source_slot = state.sequences[source_batch].as_ref().unwrap();
            let source_chunks: Vec<ChunkWindow> = (0..full_blocks)
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
                    }
                })
                .collect();
            let target_slot = state.sequences[target_batch].as_mut().unwrap();
            target_slot.replace_chunks(source_chunks);
        }

        // Handle the last block with remainder -- copy per-head data from
        // potentially different source arenas into a single new float chunk.
        if remainder > 0 {
            let last_blk = full_blocks;
            let n_kv_head = self.inner.n_kv_head;
            let sub_head_dim = self.inner.head_dim / N_PALETTE;

            // Clone the source block's full per-head GID vector AND its
            // pal_map / outer-scale state — the partial-block copy below
            // preserves the source's encoded byte semantics, so the new
            // chunk needs the same metadata to decode correctly.
            let (source_gids, source_k_pal, source_v_pal, source_k_scale, source_v_scale): (
                HeadGids,
                Arc<Vec<u8>>,
                Arc<Vec<u8>>,
                Arc<Vec<f32>>,
                Arc<Vec<f32>>,
            ) = {
                let cw = state.sequences[source_batch]
                    .as_ref()
                    .unwrap()
                    .chunk_at(last_blk)
                    .unwrap();
                (
                    cw.gids.clone(),
                    cw.k_pal.clone(),
                    cw.v_pal.clone(),
                    cw.k_scale.clone(),
                    cw.v_scale.clone(),
                )
            };

            // Allocate new chunks using active_k_arena_key (R16 on GPU, Float on CPU)
            // so the active decode kernel can write directly into them.
            let k_key = self.active_k_arena_key();
            let v_key = self.active_v_arena_key();
            let target_gids = self.inner.storage.write(|arena_state| {
                let mut gid_vec: Vec<ChunkGid> = Vec::with_capacity(GIDS_PER_HEAD * n_kv_head);

                for i in 0..(GIDS_PER_HEAD * n_kv_head) {
                    let key = if i % 2 == 0 {
                        k_key.clone()
                    } else {
                        v_key.clone()
                    };
                    let gid = self.alloc_chunk_with_arenas(arena_state, key)?;
                    gid_vec.push(gid);
                }

                // Pass 1 (immutable): clone Quantized source arenas once per unique
                // arena index.  This frees the immutable borrow so pass 2 can call
                // arenas_mut() even when src and dst land in the same arena.
                let quant_clones: std::collections::HashMap<usize, candle::quantized::QTensor> = {
                    let arenas = arena_state.arenas();
                    let mut map = std::collections::HashMap::new();
                    for src_gid in source_gids.iter() {
                        let ai = src_gid.arena_idx();
                        if let Some(Arena::Quantized { data, .. }) = arenas.get(&ai) {
                            map.entry(ai).or_insert_with(|| data.clone());
                        }
                    }
                    map
                };

                // Pass 2 (mutable): copy each GID's chunk to its destination.
                let elems_per_chunk = CHUNK_SIZE * sub_head_dim;
                let arenas = arena_state.arenas_mut();
                for (i, src_gid) in source_gids.iter().enumerate() {
                    let src_arena_idx = src_gid.arena_idx();
                    let src_chunk_idx = src_gid.chunk_idx();
                    let dst_gid = &gid_vec[i];
                    let dst_arena_idx = dst_gid.arena_idx();
                    let dst_chunk_idx = dst_gid.chunk_idx();

                    if let Some(src_clone) = quant_clones.get(&src_arena_idx) {
                        // Quantized source (R16 on GPU): byte-level copy to same-format
                        // dst, or dequantize if dst is Float (CPU / format-transition).
                        let src_dtype = src_clone.dtype();
                        let src_off = src_chunk_idx * elems_per_chunk;
                        let dst_off = dst_chunk_idx * elems_per_chunk;
                        match arenas.get_mut(&dst_arena_idx) {
                            Some(Arena::Quantized { data: dst_q, .. }) => {
                                if src_dtype == dst_q.dtype() {
                                    dst_q.slice_range_copy(
                                        src_clone,
                                        src_off,
                                        dst_off,
                                        elems_per_chunk,
                                    )?;
                                } else {
                                    candle::bail!(
                                        "fork_sequence: quant dtype mismatch ({:?} vs {:?})",
                                        src_dtype,
                                        dst_q.dtype()
                                    );
                                }
                            }
                            Some(Arena::Float { data: dst_data, .. }) => {
                                // Quantized source → Float dst: layouts differ.
                                // Quant/R16 arenas store blocks in DIM-MAJOR
                                // order within each chunk (block `d` holds
                                // 32 tokens of dim `d`); `dequantize_f16`
                                // walks blocks in storage order so its flat
                                // output per chunk is (dim, token). Float
                                // arenas expect (chunk, token, dim). Reshape
                                // in the native (chunk, dim, token) order,
                                // then transpose dim↔token before writing.
                                let device = dst_data.device().clone();
                                let dst_dtype = dst_data.dtype();
                                let kv_float = src_clone.dequantize_f16(&device)?;
                                let s_hdim = dst_data.dim(2)?;
                                let t = kv_float.elem_count() / (CHUNK_SIZE * s_hdim);
                                let kv_r = kv_float.reshape((t, s_hdim, CHUNK_SIZE))?;
                                let hd = kv_r
                                    .narrow(0, src_chunk_idx, 1)?
                                    .transpose(1, 2)?
                                    .contiguous()?
                                    .to_dtype(dst_dtype)?;
                                dst_data.slice_set(&hd, 0, dst_chunk_idx)?;
                            }
                            None => candle::bail!(
                                "fork_sequence: dst arena {} not found",
                                dst_arena_idx
                            ),
                        }
                    } else {
                        // Float source: extract owned slice (NLL ends the immutable
                        // arenas borrow), then mutably access the destination.
                        let head_data = match arenas.get(&src_arena_idx) {
                            Some(Arena::Float { data, .. }) => {
                                data.narrow(0, src_chunk_idx, 1)?.copy()?
                            }
                            None => candle::bail!(
                                "fork_sequence: src arena {} not found",
                                src_arena_idx
                            ),
                            _ => candle::bail!(
                                "fork_sequence: unexpected arena type at {}",
                                src_arena_idx
                            ),
                        };
                        match arenas.get_mut(&dst_arena_idx) {
                            Some(Arena::Float { data, .. }) => {
                                data.slice_set(&head_data, 0, dst_chunk_idx)?
                            }
                            _ => candle::bail!("fork_sequence: Float src but non-Float dst"),
                        }
                    }
                }

                Ok(gid_vec)
            })??;

            // Push remainder block with per-head GIDs. The arenas are freshly
            // allocated but the data was *copied* from the source, so the
            // chunk inherits the source's pal_map and outer-scale metadata
            // (otherwise the decoder would misinterpret the copied bytes).
            let gids = HeadGids::from_vec(target_gids);
            state.sequences[target_batch]
                .as_mut()
                .unwrap()
                .push_chunk(ChunkWindow {
                    gids,
                    usage: remainder as u32,
                    offset: 0,
                    k_pal: source_k_pal,
                    v_pal: source_v_pal,
                    k_scale: source_k_scale,
                    v_scale: source_v_scale,
                });
        }

        // block_count() = chunks.len() = total_blocks. âœ“

        drop(state);

        Ok(seq_len)
    }

    /// Fork a sequence, automatically allocating a new slot for the target.
    ///
    /// This combines `alloc_sequence` and `fork_sequence` into a single operation.
    /// Complete blocks are shared (COW), partial blocks are copied.
    ///
    /// Returns the batch index of the newly allocated target sequence.
    pub fn fork_sequence_alloc(&self, source_batch: usize, seq_len: usize) -> Result<usize> {
        let target_batch = self.alloc_sequence()?;

        match self.fork_sequence(source_batch, target_batch, seq_len) {
            Ok(_) => Ok(target_batch),
            Err(e) => {
                // Clean up on error
                let _ = self.free_sequence(target_batch);
                Err(e)
            }
        }
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

    /// Internal: writable-range gate. Under the read-only projection model
    /// the tail (and any newly-allocated block past it) is unshared by
    /// construction, so no COW is ever required here. The function remains
    /// as a hook in case future paths want a writability re-check; today it
    /// always reports "no COW occurred."
    pub(super) fn ensure_blocks_writable_locked(
        &self,
        state: &mut BlockTableState,
        batch_idx: usize,
        _start_block: usize,
        _end_block: usize,
    ) -> Result<bool> {
        if state.sequences[batch_idx].is_none() {
            return Ok(false);
        }
        Ok(false)
    }

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
    /// fresh active chunk pushed after the borrow loop. `borrowed_block_count`
    /// must be passed unchanged to [`finalize_view`].
    pub fn create_view_sequence(
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

        let chunk_size = CHUNK_SIZE;

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
        let borrowed_meta: Vec<(
            HeadGids,
            u32,
            u16,
            i32,
            Arc<Vec<u8>>,
            Arc<Vec<u8>>,
            Arc<Vec<f32>>,
            Arc<Vec<f32>>,
        )> = {
            let ps = state.sequences[parent_batch].as_ref().unwrap();
            parent_blocks
                .iter()
                .enumerate()
                .map(|(_view_blk, &parent_blk)| {
                    let cw = ps.chunk_at(parent_blk);
                    let gids = cw
                        .map(|cw| cw.gids.clone())
                        .unwrap_or_else(|| HeadGids::uniform(ChunkGid::detached(-1), 1));
                    let (usage, offset) = if let Some(cw) = cw {
                        (cw.usage, cw.offset)
                    } else {
                        (chunk_size as u32, 0u16)
                    };
                    let (k_pal, v_pal, k_scale, v_scale) = match cw {
                        Some(cw) => (
                            cw.k_pal.clone(),
                            cw.v_pal.clone(),
                            cw.k_scale.clone(),
                            cw.v_scale.clone(),
                        ),
                        None => (
                            self.inner.identity_pal.clone(),
                            self.inner.identity_pal.clone(),
                            self.inner.identity_scale.clone(),
                            self.inner.identity_scale.clone(),
                        ),
                    };
                    (
                        gids,
                        usage,
                        offset,
                        ps.rope_pos(parent_blk),
                        k_pal,
                        v_pal,
                        k_scale,
                        v_scale,
                    )
                })
                .collect()
        };
        let borrowed_token_count: usize = borrowed_meta
            .iter()
            .map(|(_, usage, ..)| *usage as usize)
            .sum();

        // All projected chunks are read-only. The view borrows EVERY parent
        // block (including a partial tail) via Arc clone — no COW. New tokens
        // never extend a borrowed chunk; they always land in a fresh active
        // chunk pushed at the end of the borrow loop. This collapses two
        // older concerns at once:
        //   - no Q→R16 elevation needed for closed-quant partials,
        //   - the tail-is-shared debug class becomes structurally impossible.
        let borrowed_count = parent_blocks.len();

        // Free any blocks currently in the view slot (if solely owned)
        {
            let vs = state.sequences[view_batch].as_mut().unwrap();
            // GIDs are dropped via RAII, returning to pool
            vs.clear_chunks();
        }

        // Clone parent blocks (Arc shared, read-only) and push one fresh
        // empty active chunk so the next write has somewhere unshared to land.
        {
            let vs = state.sequences[view_batch].as_mut().unwrap();
            for (
                source_gids,
                usage,
                offset,
                _rope_base,
                source_k_pal,
                source_v_pal,
                source_k_scale,
                source_v_scale,
            ) in borrowed_meta.into_iter()
            {
                vs.push_chunk(ChunkWindow {
                    gids: source_gids,
                    usage,
                    offset,
                    k_pal: source_k_pal,
                    v_pal: source_v_pal,
                    k_scale: source_k_scale,
                    v_scale: source_v_scale,
                });
            }
        }

        // Push the fresh active chunk that writes will land in. Allocated in
        // the active K/V arena keys (R16 K + F16 V on GPU) so decode/prefill
        // kernels can write directly. Its gids have strong_count = 1, so the
        // tail of the slot is unshared by construction — no COW step needed.
        {
            let active_cw = self.alloc_block_chunks(0, 0)?;
            let vs = state.sequences[view_batch].as_mut().unwrap();
            vs.push_chunk(active_cw);
        }

        // Writes start at the fresh active chunk (one past the borrowed prefix).
        {
            let vs = state.sequences[view_batch].as_mut().unwrap();
            let view_block_count = vs.block_count();
            // borrowed_count blocks borrowed read-only, then one fresh active.
            // The active chunk is at index `borrowed_count` and is where new
            // tokens land.
            vs.set_writer_start_idx(view_block_count.saturating_sub(1));
        }

        drop(state);

        Ok((borrowed_count, borrowed_token_count))
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

    /// Snapshot the current turn as a [`SealedSequence`] of windows.
    ///
    /// Every chunk currently in the slot's block table is recorded
    /// as a [`SealedChunk`] window — full blocks *and* the partial
    /// trailing block.  Dropping the partial would silently lose
    /// up to `CHUNK_SIZE - 1` tokens per sealed unit; when many
    /// units are projected back-to-back (e.g. the system-prompt
    /// section catalog) the gaps stack up and the destination's
    /// kernel-visible KV ends up shorter than the projection
    /// reports.
    ///
    /// Sharing the partial chunk via Arc into many destination slots
    /// is safe because the architecture enforces
    /// **single-writer-per-partial-chunk** at the resume layer: at
    /// most one slot may be bound to the chunk's owning
    /// `(layer, group, instance)` target at any moment, and only that
    /// slot extends the chunk.  All other holders are read-only and
    /// observe the writer's `usage` advancing through the Arc — that
    /// observed mutability is intentional (the substrate's view of
    /// the timeline tracks the writer's progress, not lags it).
    ///
    /// Refcount keeps the chunk's memory alive across freed slots;
    /// substrate metadata updates are atomic; projection sees a
    /// consistent snapshot.
    ///
    /// # Parameters
    /// * `batch_idx` — slot index of the sequence
    pub fn record_turn(&self, batch_idx: usize) -> Result<SealedSequence> {
        let chunk_size = CHUNK_SIZE;

        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
            Some(s) => s,
            None => {
                candle::bail!("record_turn: batch_idx {} is not allocated", batch_idx)
            }
        };

        // Pre-compute per-arena byte strides once for the whole record call.
        // `chunk_byte_stride` is the byte size of one physical chunk slot in that
        // arena; summing over unique arena indices gives the total bytes for a
        // SealedChunk's GID set.
        let arena_infos = self.resolve_arena_info().unwrap_or_default();

        // Build SealedChunks from every block in the slot's block
        // table, including the trailing partial.  No positional state
        // is captured — K bytes in the chunks are un-rotated, and
        // RoPE is applied at the latest responsible moment by the
        // attention kernel using a slice_rope recomputed from the
        // destination slot's cumulative usage.  See `SealedChunk`
        // docs.
        let sealed_chunks: Vec<SealedChunk> = slot
            .chunks_slice()
            .iter()
            .map(|cw| {
                let byte_size = cw.gids.arena_byte_size(&arena_infos);
                SealedChunk {
                    gids: cw.gids.clone(),
                    offset: cw.offset,
                    token_count: cw.usage as u16,
                    k_pal: cw.k_pal.clone(),
                    v_pal: cw.v_pal.clone(),
                    k_scale: cw.k_scale.clone(),
                    v_scale: cw.v_scale.clone(),
                    byte_size,
                }
            })
            .collect();

        let token_count = sealed_chunks.iter().map(|sc| sc.token_count as usize).sum();
        let location = self.inner.storage.default_location();
        Ok(SealedSequence {
            chunks: sealed_chunks,
            token_count,
            chunk_size,
            location,
        })
    }

    /// Ensure the last chunk is a writable (float, uniquely-owned) block.
    ///
    /// Must be called before each inference pass.  Pushes a new empty float
    /// chunk if the current tail is quantized or COW-shared (refcount > 1).
    /// Returns `true` if a new chunk was pushed.
    pub fn ensure_writable_tail(&self, batch_idx: usize) -> Result<bool> {
        // Fast check under read lock: determine what action is needed.
        // The tail block is never shared after fork/view — fork_sequence and
        // create_view_sequence copy partial tails at fork time.  So we only
        // need to check: empty, full, or quantized → push new float block;
        // partial + float → already writable.
        let needs_new_block: Option<bool> = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                Some(s) => s,
                None => return Ok(false),
            };
            if slot.is_empty() {
                Some(true) // empty → push new block
            } else {
                let cw = slot.last_chunk().unwrap();
                debug_assert!(
                    cw.gids.iter().all(|g| g.strong_count() <= cw.gids.len()),
                    "tail block must not be shared — fork should have copied it"
                );
                let is_full = (cw.offset as usize + cw.usage as usize) >= CHUNK_SIZE;
                if is_full {
                    Some(true)
                } else {
                    // If any head lives in a quantized arena the whole block
                    // must be replaced with a fresh float block.
                    let any_quantized = self.inner.storage.read(|s| {
                        cw.gids.iter().any(|g| {
                            s.arena_key(g.arena_idx())
                                .map(|k| {
                                    matches!(k.format, crate::kv_cache::KvFormat::Quantized(_))
                                })
                                .unwrap_or(true)
                        })
                    })?;
                    if any_quantized {
                        Some(true)
                    } else {
                        None // partial + float → already writable
                    }
                }
            }
        };

        match needs_new_block {
            None => return Ok(false), // already writable
            Some(true) => {
                // Push new empty float block.
                let current_block_count = {
                    let state = self
                        .state
                        .read()
                        .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
                    state
                        .sequences
                        .get(batch_idx)
                        .and_then(|s| s.as_ref())
                        .map(|s| s.block_count())
                        .unwrap_or(0)
                };
                self.ensure_max_blocks(current_block_count + 1)?;
                let mut state = self
                    .state
                    .write()
                    .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
                let cw = self.alloc_block_chunks(0, 0)?;
                if let Some(Some(slot)) = state.sequences.get_mut(batch_idx) {
                    slot.push_chunk(cw);
                }
            }
            _ => unreachable!(),
        }

        Ok(true)
    }

    /// Truncate the sequence at `batch_idx` to keep only the first
    /// `block_count` chunks; everything beyond is dropped (their
    /// `ChunkGid`s fall and physical chunks return to the pool when
    /// their refcount reaches zero).
    ///
    /// Resets the sequence's logical token offset to the sum of
    /// usages of the retained chunks.  Used by the SubmitTurn handler
    /// to reset a persistent conversation sequence to its
    /// system-prompt baseline before injecting the next turn's
    /// projection.
    /// Snapshot the slot's writer-owned tail chunks (`[writer_start_idx..end)`)
    /// and remove them from the slot.
    ///
    /// Used by the stateless-slot rebuild path: the scheduler takes
    /// this snapshot before calling [`Self::truncate_sequence_to_blocks`]
    /// + [`Self::inject_sealed_at_tail`] to refresh the prefix, then
    /// restores the tail via [`Self::extend_writer_tail`]. The returned
    /// [`WriterTail`] holds RAII refs that keep the underlying arena
    /// chunks alive across the truncate, so no bytes are copied.
    ///
    /// At turn-boundary projection (the common case) the tail is empty
    /// and this is effectively a no-op.
    pub fn split_off_writer_tail(
        &self,
        batch_idx: usize,
    ) -> Result<super::types::WriterTail> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let slot = state.sequences[batch_idx].as_mut().ok_or_else(|| {
            candle::Error::Msg(format!(
                "split_off_writer_tail: slot {} not allocated",
                batch_idx
            ))
        })?;
        let writer_start = slot.writer_start_idx();
        let chunks = slot.split_off_chunks(writer_start);
        Ok(super::types::WriterTail { chunks })
    }

    /// Restore the writer-owned tail chunks captured by
    /// [`Self::split_off_writer_tail`]. Appends them to the slot's
    /// chunk list after whatever prefix has been re-injected; the
    /// writer boundary stays where the prefix's
    /// [`Self::inject_sealed_at_tail`] left it.
    pub fn extend_writer_tail(
        &self,
        batch_idx: usize,
        tail: super::types::WriterTail,
    ) -> Result<()> {
        if tail.is_empty() {
            return Ok(());
        }
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }
        let new_total = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let slot = state.sequences[batch_idx].as_ref().ok_or_else(|| {
                candle::Error::Msg(format!(
                    "extend_writer_tail: slot {} not allocated",
                    batch_idx
                ))
            })?;
            slot.block_count() + tail.chunks.len()
        };
        self.ensure_max_blocks(new_total)?;
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let slot = state.sequences[batch_idx].as_mut().ok_or_else(|| {
            candle::Error::Msg(format!(
                "extend_writer_tail: slot {} not allocated",
                batch_idx
            ))
        })?;
        slot.extend_chunks(tail.chunks);
        Ok(())
    }

    pub fn truncate_sequence_to_blocks(
        &self,
        batch_idx: usize,
        block_count: usize,
    ) -> Result<usize> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let slot = state.sequences[batch_idx].as_mut().ok_or_else(|| {
            candle::Error::Msg(format!(
                "truncate_sequence_to_blocks: slot {} not allocated",
                batch_idx
            ))
        })?;
        slot.truncate_chunks(block_count);
        // Reset writer boundary: after truncate everything before
        // `block_count` is whatever the caller had (typically nothing
        // for `truncate(0)` or Arc-shared for partial truncates).
        // Clamp the existing writer_start to be ≤ the new chunk count
        // so any writes go into existing chunks past it.
        if slot.writer_start_idx() > slot.block_count() {
            slot.set_writer_start_idx(slot.block_count());
        }
        let new_tokens: usize = slot.chunks_slice().iter().map(|c| c.usage as usize).sum();
        Ok(new_tokens)
    }

    /// Append the sealed chunks of `sealed` onto the tail of the
    /// sequence at `batch_idx` as live `ChunkWindow`s.
    ///
    /// Pure metadata — no kernel, no DMA.  Returns `(block_start,
    /// block_end)` — the absolute block range where the appended
    /// chunks now live in the target sequence.  The caller is
    /// responsible for advancing the sequence's logical token offset
    /// after this returns.
    ///
    /// Used by the projection path to inject GPU-resident chunks
    /// (returned from the upload cache) into the freshly-allocated
    /// parent sequence the next turn will decode against.
    pub fn inject_sealed_at_tail(
        &self,
        batch_idx: usize,
        sealed: &SealedSequence,
    ) -> Result<(usize, usize)> {
        use super::types::ChunkWindow;

        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }

        let n = sealed.chunks.len();
        let block_start = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            state
                .sequences
                .get(batch_idx)
                .and_then(|s| s.as_ref())
                .map(|s| s.block_count())
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "inject_sealed_at_tail: slot {} not allocated",
                        batch_idx
                    ))
                })?
        };

        if n == 0 {
            return Ok((block_start, block_start));
        }

        self.ensure_max_blocks(block_start + n)?;

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let slot = state.sequences[batch_idx].as_mut().ok_or_else(|| {
            candle::Error::Msg(format!(
                "inject_sealed_at_tail: slot {} not allocated",
                batch_idx
            ))
        })?;

        // Injected chunks are Arc-shared with substrate/parent —
        // advance the writer boundary past them so subsequent writes
        // never land in their physical positions.
        slot.set_writer_start_idx(block_start + n);
        let windows: Vec<ChunkWindow> = sealed
            .chunks
            .iter()
            .map(|sc| ChunkWindow {
                gids: sc.gids.clone(),
                usage: sc.token_count as u32,
                offset: sc.offset,
                k_pal: Arc::clone(&sc.k_pal),
                v_pal: Arc::clone(&sc.v_pal),
                k_scale: Arc::clone(&sc.k_scale),
                v_scale: Arc::clone(&sc.v_scale),
            })
            .collect();
        slot.extend_chunks(windows);
        let block_end = slot.block_count();

        Ok((block_start, block_end))
    }
}

// The deleted `cow_partial_tail_if_shared` machinery lived here.
// It existed to copy a slot's partial tail when the tail's
// `ChunkGid`s were shared with another holder (e.g. the substrate's
// pinned section).  Under the current design that scenario is
// reachable but harmless, because the architecture's actual rule is
// **single-writer-per-partial-chunk** rather than the stricter
// "all shared chunks are immutable" the deleted machinery assumed.
//
// `record_turn` seals every chunk in the slot, including the
// trailing partial.  The resulting `SealedSequence` is pinned in
// the substrate and projected into destination slots via
// `inject_sealed_at_tail`, which Arc-clones each `ChunkGid` as a
// fresh `ChunkWindow`.  Many slots can therefore hold the same
// `ChunkGid` simultaneously.
//
// What keeps this safe is the upstream constraint: at any moment,
// at most one slot is the *active writer* for a given
// `(layer, group, instance)` target — the resume-check rejects a
// second attempt to bind a sequence to a target that is already
// resumed.  The unique active writer is the unique slot whose tail
// chunk is also the substrate's tail chunk for that target.  All
// other holders (the substrate's metadata pointer, BDP scans, sibling
// slots that projected the section/turn purely as read context) are
// read-only.  When the active writer extends the tail chunk's usage
// from N to N+1, the read-only holders see the change through their
// Arc — that's *intended*: the substrate's view of the timeline is
// supposed to track the writer's progress, not lag it.
//
// Sections work identically: each section is bound to a specific
// `(layer, group)` and a destination slot becomes the unique writer
// past its tail.  Other slots that project the same section get the
// `ChunkGid`s read-only via Arc and write their own continuation into
// fresh chunks they allocate; they never extend the section's tail
// chunk, so there is no contention.
