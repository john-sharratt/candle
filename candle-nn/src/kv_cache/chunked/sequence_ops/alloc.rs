//! Allocates and frees the sequence slots of a chunked KV backing.
//!
//! A slot is a batch index into the backing's block table. Allocation takes the
//! first free slot and grows capacity when none is free, so sessions that share
//! one pool get non-overlapping indices. Freeing returns the slot's chunks
//! through RAII. The detached form releases the device buffer on the caller's
//! thread and hands back only the chunk table, whose drops make no device call.
//! `set_len` changes only the host chunk table, since the decode kernel advances
//! its own length on the device, and chunks shared with a parent are never
//! modified.

use super::*;

impl ChunkedKvBacking {
    /// Create a new [`SequenceState`] bound to this backing's device.
    ///
    /// Under the `cuda` feature the state receives a handle on the device, from
    /// which its guard records or issues its slot-state uploads.
    #[inline]
    pub(in super::super) fn make_sequence_state(&self) -> Result<SequenceState> {
        #[cfg(feature = "cuda")]
        {
            let device = match &self.inner.device {
                candle::Device::Cuda(dev) => Some(dev.clone()),
                _ => None,
            };
            Ok(SequenceState::new(device))
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
    /// multiple sessions share the same `BackingInner` pool — consulting the
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

    /// [`Self::free_sequence`], with the block table handed back instead of
    /// dropped.
    ///
    /// The slot and its device buffer are released here, on the caller's
    /// thread. What comes back is only the chunk table — RAII gids and Arc'd
    /// metadata, whose drops make no device call — so a session freeing a long
    /// sequence drops every layer's table at once. A freed view's table is the
    /// parent's whole prefix, borrowed: thousands of refcount decrements per
    /// layer that release nothing.
    pub fn free_sequence_detached(&self, batch_idx: usize) -> Result<DetachedChunks> {
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
        let Some(mut slot) = state.sequences[batch_idx].take() else {
            return Ok(DetachedChunks::none());
        };
        let (chunks, pins) = slot.take_chunks();
        drop(slot);
        drop(state);
        Ok(DetachedChunks { chunks, pins })
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
                let mut first_filled = None;
                while remaining > 0 && idx < n {
                    let cap = {
                        let c = &seq.chunks_slice()[idx];
                        chunk_size - (c.offset as usize + c.usage as usize)
                    };
                    let take = remaining.min(cap);
                    if take > 0 && first_filled.is_none() {
                        first_filled = Some(idx);
                    }
                    seq.chunk_at_mut(idx).unwrap().usage += take as u32;
                    remaining -= take;
                    idx += 1;
                }
                // Only the host chunk table changes here. The serialised decode
                // slot buffer is left alone: the decode kernel advances its
                // writer length on the device, and a commit made outside the
                // decode kernel marks it (`SequenceState::mark_decode_writer_stale`)
                // for the next sync to re-serialise — from the first chunk this
                // filled, every one before it being full and untouched.
                if let Some(first) = first_filled {
                    seq.note_fill_from(first);
                }
            }
        }
    }
}
