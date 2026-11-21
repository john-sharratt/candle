//! The edits that change which chunks a sequence's block table holds, each
//! keeping the cached GPU slot-state buffer true to the host chunk list.
//!
//! The decode hot path trusts that cached buffer whenever it exists, so every
//! structural change invalidates it at the mutation site rather than leaving
//! the decode step to notice a mismatch. An append is the exception: each
//! serialised entry stays a valid prefix of the list, so the buffer is marked
//! for the next sync to extend with the new tail. Re-serialising the whole list
//! at depth on every layer was the dominant host cost of a deep decode step.
//! A detached block table releases its chunks on whichever thread drops it and
//! touches no device.

use super::*;

/// A freed sequence's block table, detached from its slot
/// ([`ChunkedKvBacking::free_sequence_detached`](super::ChunkedKvBacking::free_sequence_detached)).
///
/// Dropping it releases the chunks and the slot-state serialisation's pins on
/// them, on whichever thread drops it. Nothing in it touches the device.
pub struct DetachedChunks {
    pub(crate) chunks: Vec<ChunkWindow>,
    /// Held only to be dropped with the chunks, on the dropping thread.
    #[allow(dead_code)]
    pub(crate) pins: Arc<Vec<ChunkPin>>,
}

impl DetachedChunks {
    /// Nothing detached.
    pub(crate) fn none() -> Self {
        Self {
            chunks: Vec::new(),
            pins: Arc::new(Vec::new()),
        }
    }

    /// Chunks held.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Whether it holds no chunk.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

impl SequenceState {
    // Chunk-only mutation primitives (no GPU params required)
    //
    // Structural operations eagerly reset the GPU backing buffer because the
    // cached decode slot state is trusted on the hot path. Any change to the
    // chunk vector shape therefore must invalidate the cache immediately at the
    // mutation site rather than relying on decode-time mismatch detection —
    // except an append, which leaves every serialised entry a valid prefix of
    // the list and is marked for the next sync to extend.

    /// Append one `ChunkWindow` to the tail and mark the cached GPU slot buffer
    /// appended-to, so the next sync serialises the new tail
    /// (`GpuChunksGuard::extend_decode`) instead of the whole list. At depth the
    /// whole list is thousands of chunks, re-serialised and re-uploaded per
    /// layer at every 32-token boundary — the dominant host cost of a deep
    /// decode step.
    #[inline]
    pub(crate) fn push_chunk(&mut self, cw: ChunkWindow) {
        self.chunks.push(cw);
        self.gpu_chunks.mark_appended();
    }

    /// Clear all chunks and clear the GPU buffer.
    pub(crate) fn clear_chunks(&mut self) {
        self.chunks.clear();
        self.gap_cursor = 0;
        self.gpu_chunks.as_mut().clear();
    }

    /// Move every chunk out and clear the GPU buffer, handing the chunks and
    /// the serialisation's pins on them to the caller to drop.
    pub(crate) fn take_chunks(&mut self) -> (Vec<ChunkWindow>, Arc<Vec<ChunkPin>>) {
        let chunks = std::mem::take(&mut self.chunks);
        self.gap_cursor = 0;
        let pins = self.gpu_chunks.as_mut().clear_detached();
        (chunks, pins)
    }

    /// Drain the first `n` chunks (RAII-drops their GIDs) and clear the GPU buffer.
    pub(crate) fn drain_front_chunks(&mut self, n: usize) {
        let n = n.min(self.chunks.len());
        self.chunks.drain(..n).for_each(drop);
        self.gap_cursor = cursor_after_front_eviction(self.gap_cursor, n);
        if n > 0 {
            self.gpu_chunks.as_mut().clear();
        }
    }

    /// Prepend `prefix` before all existing chunks; GPU buffer is cleared.
    pub(crate) fn prepend_chunks(&mut self, mut prefix: Vec<ChunkWindow>) {
        self.gap_cursor = cursor_after_splice(self.gap_cursor, 0, 0, prefix.len());
        prefix.extend(std::mem::take(&mut self.chunks));
        self.chunks = prefix;
        self.gpu_chunks.as_mut().clear();
    }

    /// Replace all chunks with `new_chunks` and clear the GPU buffer.
    pub(crate) fn replace_chunks(&mut self, new_chunks: Vec<ChunkWindow>) {
        self.chunks = new_chunks;
        self.gap_cursor = 0;
        self.gpu_chunks.as_mut().clear();
    }

    /// Replace the sealed chunks `[start, end)` with `windows`, leaving every
    /// chunk outside the range where it stands, in order; the GPU buffer is
    /// cleared because the layout changed.
    ///
    /// **Sealed history only.** The range must lie below the writer boundary:
    /// chunks there are Arc-shared prefix that nothing writes into, so moving
    /// them is metadata, and every position after the splice is re-derived from
    /// cumulative usage when the slot is next read. The boundary moves by the
    /// change in count, so the writer region keeps exactly its chunks.
    pub(crate) fn splice_sealed_chunks(
        &mut self,
        start: usize,
        end: usize,
        windows: Vec<ChunkWindow>,
    ) -> candle::Result<()> {
        if start > end || end > self.writer_start_idx {
            candle::bail!(
                "splice_sealed_chunks: [{start}, {end}) is not sealed history below the \
                 writer boundary {} ({} chunks)",
                self.writer_start_idx,
                self.chunks.len()
            );
        }
        let inserted = windows.len();
        self.chunks.splice(start..end, windows);
        self.writer_start_idx = self.writer_start_idx - (end - start) + inserted;
        self.gap_cursor = cursor_after_splice(self.gap_cursor, start, end, inserted);
        self.gpu_chunks.as_mut().clear();
        Ok(())
    }

    /// Drain all chunks from index `at` onward into a `Vec`.  GPU buffer is
    /// cleared because chunk count changes.
    pub(crate) fn split_off_chunks(&mut self, at: usize) -> Vec<ChunkWindow> {
        let tail = self.chunks.split_off(at);
        self.gap_cursor = cursor_after_truncate(self.gap_cursor, self.chunks.len());
        self.gpu_chunks.as_mut().clear();
        tail
    }

    /// Truncate to `n` chunks.  If the length actually decreases the GPU
    /// buffer is cleared.
    pub(crate) fn truncate_chunks(&mut self, n: usize) {
        if n < self.chunks.len() {
            self.chunks.truncate(n);
            self.gap_cursor = cursor_after_truncate(self.gap_cursor, n);
            self.gpu_chunks.as_mut().clear();
        }
    }

    /// Extend with additional `ChunkWindow`s and invalidate the cached GPU
    /// slot buffer because the chunk layout has changed.
    ///
    /// An appended chunk the paged stores cannot write — a partial a mid-turn
    /// seal quantized, arriving with a restored writer tail or a view folded
    /// back into its parent — moves the writer boundary past it, so it is
    /// never the chunk a write resolves to. See
    /// [`Self::seal_unwritable_writer_region`].
    pub(crate) fn extend_chunks(&mut self, iter: impl IntoIterator<Item = ChunkWindow>) {
        self.chunks.extend(iter);
        self.gpu_chunks.as_mut().clear();
        self.seal_unwritable_writer_region();
    }
}
