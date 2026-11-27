//! The sliding-window ring's eviction decision, and the absolute-position
//! bookkeeping that keeps a window's keys at their true positions.
//!
//! A query attends only to keys inside its window, so a front chunk is freed
//! once every one of its tokens has left the window. That decision is a pure
//! function, so its off-by-one boundary can be tested without any arenas. The
//! writer chunk and everything past the writer boundary are never evicted.
//! Evicted tokens are counted in a base position that offsets every chunk's
//! rope base, so the resident window keeps its absolute frame.

use super::*;

/// Number of front chunks fully outside the sliding window — the pure
/// decision behind [`SequenceState::evict_front_window`], factored out so the
/// off-by-one boundary is unit-testable without arenas.
///
/// `usages[i]` is the positions chunk `i` takes — its lead and its valid tokens
/// ([`ChunkWindow::span`]); `writer_start` bounds how far
/// the sweep may go (never evicts the writer or a shared prefix); `base_pos` is
/// the absolute position of the first resident token; the window ends at
/// `abs_pos` and spans `window_size` tokens. A chunk at absolute
/// `[start, start+usage)` is evictable iff `start + usage ≤ abs_pos −
/// window_size + 1` (its highest position is below the lowest in-window one).
pub(crate) fn front_evict_count(
    usages: &[u32],
    writer_start: usize,
    base_pos: u32,
    window_size: usize,
    abs_pos: usize,
) -> usize {
    if window_size == 0 {
        return 0;
    }
    // Lowest absolute position still inside the window.
    let lo = abs_pos.saturating_sub(window_size) + 1;
    let max_evict = writer_start.min(usages.len());
    let mut start = base_pos as usize;
    let mut drained = 0usize;
    for &usage in usages.iter().take(max_evict) {
        let end = start + usage as usize; // one past the chunk's last token
        if end <= lo {
            drained += 1;
            start = end;
        } else {
            break;
        }
    }
    drained
}

impl SequenceState {
    #[cfg(feature = "cuda")]
    pub(in super::super) fn new(device: Option<CudaDevice>) -> Self {
        Self {
            chunks: Vec::new(),
            gpu_chunks: GpuChunks::new(device),
            writer_start_idx: 0,
            gap_cursor: 0,
            base_pos: 0,
            pending_lead: 0,
            decode_stale_from: None,
            fill_from: None,
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub(in super::super) fn new() -> Self {
        Self {
            chunks: Vec::new(),
            gpu_chunks: GpuChunks::new(),
            writer_start_idx: 0,
            gap_cursor: 0,
            base_pos: 0,
            pending_lead: 0,
            decode_stale_from: None,
            fill_from: None,
        }
    }

    /// Total allocated blocks.
    #[inline]
    pub(in super::super) fn block_count(&self) -> usize {
        self.chunks.len()
    }

    /// Total token count across all chunks.
    #[inline]
    pub(in super::super) fn seq_len(&self) -> usize {
        self.chunks.iter().map(|c| c.usage as usize).sum()
    }

    /// The positions the slot leaves empty: every chunk's lead and the lead
    /// pending at the tail. The position frontier is the token count plus
    /// this, since every hole lies before the tail.
    #[inline]
    pub(in super::super) fn lead_total(&self) -> usize {
        self.chunks.iter().map(|c| c.lead as usize).sum::<usize>() + self.pending_lead as usize
    }

    /// The position the next token the tail gains is read at: the evicted
    /// front, every chunk's lead and tokens, and the lead pending at the tail.
    #[inline]
    pub(in super::super) fn position_frontier(&self) -> usize {
        self.base_pos as usize
            + self.chunks.iter().map(|c| c.span() as usize).sum::<usize>()
            + self.pending_lead as usize
    }

    /// The slot's positions against its tokens, as runs `[position, skipped]`
    /// ascending by position: from `position` on, until the next run, the
    /// token a position holds is `position − skipped` — `skipped` being the
    /// evicted front and every hole before it. The first run opens at 0; each
    /// chunk with a lead opens one at its first token, and a lead pending at
    /// the tail one at the frontier. A slot without holes is one run.
    pub(in super::super) fn position_runs(&self) -> Vec<[u32; 2]> {
        let mut skipped = self.base_pos;
        let mut at = self.base_pos;
        let mut runs = vec![[0, skipped]];
        for c in &self.chunks {
            if c.lead > 0 {
                skipped += c.lead;
                runs.push([at + c.lead, skipped]);
            }
            at += c.span();
        }
        if self.pending_lead > 0 {
            skipped += self.pending_lead;
            runs.push([at + self.pending_lead, skipped]);
        }
        runs
    }

    /// Leave `n` positions empty at the tail: the next chunk the tail gains
    /// takes them as its lead. A tail that is an empty writer chunk takes them
    /// at once; one holding tokens is closed by the caller first, so no token
    /// is written after the hole into the chunk before it.
    pub(in super::super) fn push_lead(&mut self, n: u32) {
        if n == 0 {
            return;
        }
        match self.chunks.last_mut() {
            Some(tail) if tail.usage == 0 => {
                tail.lead += n;
                self.gpu_chunks.as_mut().clear();
            }
            _ => self.pending_lead += n,
        }
    }

    /// Hand the pending lead to `cw`, the chunk the tail is about to gain.
    #[inline]
    pub(in super::super) fn take_pending_lead(&mut self, cw: &mut ChunkWindow) {
        cw.lead += std::mem::take(&mut self.pending_lead);
    }

    /// Drop the lead pending at the tail: the tail it was left after is gone.
    #[inline]
    pub(in super::super) fn clear_pending_lead(&mut self) {
        self.pending_lead = 0;
    }

    /// The sliding-window ring's evicted-front count — the absolute position of
    /// the first resident token. Zero until the ring first slides.
    #[inline]
    pub(in super::super) fn base_pos(&self) -> u32 {
        self.base_pos
    }

    /// Seed the evicted-front count directly (turn-seal ring restore): the
    /// remaining resident chunks then serialise ABSOLUTE `rope_base` positions
    /// from this base (`rope_pos = base_pos + Σ preceding usage`), so a resumed
    /// window continues the original absolute frame.
    #[inline]
    pub(in super::super) fn set_base_pos(&mut self, v: u32) {
        self.base_pos = v;
    }

    /// Get the `ChunkWindow` for block `blk`.
    #[inline]
    pub(in super::super) fn chunk_at(&self, blk: usize) -> Option<&ChunkWindow> {
        self.chunks.get(blk)
    }

    /// Replace the full per-head GID vector AND the per-block palette/scale
    /// metadata at block `blk`.
    ///
    /// `gids` must have length `GIDS_PER_HEAD * n_kv_head`.
    /// `k_pal` / `v_pal` must each be `n_kv_head × (head_dim/4)` bytes (or empty
    /// for identity routing).
    /// `k_scale` / `v_scale` must each be `n_kv_head × N_PALETTE` f32 values
    /// (or empty for unity scale = no outer scaling).
    ///
    /// Pal/scale **must** be passed explicitly — they describe how to interpret
    /// the bytes stored at `gids`, so they have to track every gid mutation.
    /// Callers that want to preserve existing semantics should read the chunk's
    /// current values first and pass them through.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn set_block_gids(
        &mut self,
        blk: usize,
        gids: HeadGids,
        k_pal: Arc<Vec<u8>>,
        v_pal: Arc<Vec<u8>>,
        k_scale: Arc<Vec<f32>>,
        v_scale: Arc<Vec<f32>>,
        k_fmt: Arc<Vec<u8>>,
        v_fmt: Arc<Vec<u8>>,
    ) {
        if blk < self.chunks.len() {
            let cw = &mut self.chunks[blk];
            cw.gids = gids;
            cw.k_pal = k_pal;
            cw.v_pal = v_pal;
            cw.k_scale = k_scale;
            cw.v_scale = v_scale;
            // The format tags travel with the gids and MUST be replaced with
            // them. A cold-load or elevate re-points this window at chunks in
            // whatever formats were persisted; leaving the window's previous
            // tags in place (the active R16/F16 a freshly-allocated writer
            // window carries) would make every reader decode those chunks as
            // raw floats. The pal/scale fields above have the same lifecycle
            // for the same reason.
            cw.k_fmt = k_fmt;
            cw.v_fmt = v_fmt;
            // Any GID mutation (defrag remap, cold-load reinjection) invalidates
            // the resident record's per-palette pointers. Drop it so the host
            // serializer falls back to per-forward scratch heads rather than
            // emitting a stale `kvheads_ptr`.
            cw.meta = None;
        }
    }

    /// Compute the ABSOLUTE RoPE base position for block `blk`: the count of
    /// tokens evicted off the front (`base_pos`), the positions all preceding
    /// resident blocks take (leads and tokens), and the block's own lead.
    /// `base_pos` is zero until the sliding-window ring evicts. Past the last
    /// block it is the position frontier.
    #[inline]
    pub(in super::super) fn rope_pos(&self, blk: usize) -> i32 {
        let count = blk.min(self.chunks.len());
        let before: i32 = self
            .chunks
            .iter()
            .take(count)
            .map(|c| c.span() as i32)
            .sum();
        let own = match self.chunks.get(blk) {
            Some(c) => c.lead,
            None => self.pending_lead,
        };
        self.base_pos as i32 + before + own as i32
    }

    /// Slide the sliding-window ring: drop every FRONT chunk that has fully
    /// exited the `window_size`-token window ending at absolute query position
    /// `abs_pos`, returning the new [`Self::base_pos`] (total evicted tokens).
    ///
    /// A query at `abs_pos` attends window keys with `key_pos > abs_pos −
    /// window_size` (the kernel's causal+window mask), so the lowest in-window
    /// absolute position is `abs_pos − window_size + 1`. A front chunk spanning
    /// absolute `[start, start+usage)` is fully out of window once
    /// `start + usage ≤ abs_pos − window_size + 1` — none of its tokens can be
    /// attended, so freeing it changes nothing the kernel reads while bounding
    /// the resident set (and the per-step tile walk) to `O(window_size)`.
    /// Older tokens are already folded into the compressed corpus (guaranteed
    /// by `window_size ≥ compress_ratio`), so no attended information is lost.
    ///
    /// Never touches the writer chunk or anything at/after `writer_start_idx`.
    /// Draining bumps `base_pos` (keeping remaining chunks' absolute positions
    /// intact via [`Self::rope_pos`] / the serialised `rope_base`) and shifts
    /// the writer boundary down by the number dropped.
    pub(crate) fn evict_front_window(&mut self, window_size: usize, abs_pos: usize) -> u32 {
        // The positions each chunk takes, its lead included, so a hole in front
        // of a chunk counts toward how far the window has passed it.
        let spans: Vec<u32> = self.chunks.iter().map(ChunkWindow::span).collect();
        let drained = front_evict_count(
            &spans,
            self.writer_start_idx,
            self.base_pos,
            window_size,
            abs_pos,
        );
        if drained > 0 {
            let evicted: u32 = spans[..drained].iter().copied().sum();
            self.drain_front_chunks(drained); // RAII-frees GIDs + clears GPU buffer
            self.base_pos = self.base_pos.wrapping_add(evicted);
            self.writer_start_idx = self.writer_start_idx.saturating_sub(drained);
        }
        self.base_pos
    }

    // Read-only accessors

    /// `true` when no chunks have been allocated yet.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Borrow the last `ChunkWindow` (the active tail), if any.
    #[inline]
    pub(crate) fn last_chunk(&self) -> Option<&ChunkWindow> {
        self.chunks.last()
    }

    /// Mutably borrow the last `ChunkWindow` (the active tail), if any.
    #[inline]
    pub(crate) fn last_chunk_mut(&mut self) -> Option<&mut ChunkWindow> {
        self.chunks.last_mut()
    }

    /// Mutably borrow the `ChunkWindow` at block `blk`, if any.
    #[inline]
    pub(crate) fn chunk_at_mut(&mut self, blk: usize) -> Option<&mut ChunkWindow> {
        self.chunks.get_mut(blk)
    }

    /// Return all chunks as a slice for iteration / indexing.
    #[inline]
    /// Mutable view of all chunks (test helper / direct manipulation).
    #[allow(dead_code)]
    pub(crate) fn chunks_slice_mut(&mut self) -> &mut [ChunkWindow] {
        &mut self.chunks
    }

    pub(crate) fn chunks_slice(&self) -> &[ChunkWindow] {
        &self.chunks
    }

    /// Cheap host-side validation for decode.
    ///
    /// This mirrors the kernel's write-slice invariant and rejects obviously
    /// stale slot-state before the paged decode kernel is launched.
    pub(crate) fn validate_decode_state(
        &self,
        batch_idx: usize,
        seq_offset: usize,
    ) -> candle::Result<()> {
        let Some(tail) = self.last_chunk() else {
            candle::bail!(
                "chunked decode validation failed for batch_idx {}: sequence has no allocated tail chunk",
                batch_idx
            );
        };

        let host_n = self.block_count();
        let cached_n = self.gpu_chunks.n_chunks();
        // A buffer chunks were appended past holds a prefix of the list, which
        // the sync extends before anything reads it.
        let extending = self.gpu_chunks.appended() && cached_n < host_n;
        if cached_n != 0 && cached_n != host_n && !extending {
            candle::bail!(
                "chunked decode validation failed for batch_idx {} at offset {}: cached decode slot count {} does not match host chunk count {}",
                batch_idx,
                seq_offset,
                cached_n,
                host_n
            );
        }

        let last_rope_base: usize = if host_n > 1 {
            self.chunks[..host_n - 1]
                .iter()
                .map(|c| c.usage as usize)
                .sum()
        } else {
            0
        };
        let expected_write_len = seq_offset.saturating_sub(last_rope_base);
        if expected_write_len >= CHUNK_SIZE {
            candle::bail!(
                "chunked decode validation failed for batch_idx {} at offset {}: computed write len {} is invalid for chunk_size {} (tail_offset={}, host_tail_usage={})",
                batch_idx,
                seq_offset,
                expected_write_len,
                CHUNK_SIZE,
                tail.offset,
                tail.usage
            );
        }

        let tail_fill = tail.offset as usize + expected_write_len;
        if tail_fill >= CHUNK_SIZE {
            candle::bail!(
                "chunked decode validation failed for batch_idx {} at offset {}: writable tail is already full/stale (tail_offset={}, expected_write_len={}, host_tail_usage={}, chunk_size={})",
                batch_idx,
                seq_offset,
                tail.offset,
                expected_write_len,
                tail.usage,
                CHUNK_SIZE
            );
        }

        Ok(())
    }
}
