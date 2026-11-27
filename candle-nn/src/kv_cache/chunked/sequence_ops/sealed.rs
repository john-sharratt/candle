//! Appends or splices sealed chunks into a sequence, sharing their memory
//! without a copy.
//!
//! `inject_sealed_at_tail` and `splice_sequence_blocks` are pure metadata
//! operations that clone the chunk handles, so the injected chunks keep their
//! records valid. An injection invalidates the cached device slot state
//! unconditionally, because a same-count replacement would otherwise leave
//! embedded arena pointers naming chunks that have since been freed or reused.
//! Each sealed chunk has at most one active writer, and every other holder of it
//! is read-only.

use super::*;

impl ChunkedKvBacking {
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
        slot.extend_chunks(sealed_windows(sealed));
        // The injection changed the slot's chunk layout out from under the
        // cached device-side slot-state (GpuChunks). The cache's count-mismatch
        // detection catches a GROWN table, but a same-count window replacement
        // (steady-state reprojection: truncate re-inject with an equal number
        // of chunks) leaves the stale serialization live — its embedded arena
        // base pointers keep addressing the PREVIOUS windows' chunks, whose
        // gids this injection just dropped. Once those arenas empty, the sweep
        // frees and RECYCLES their indices, and the next decode launch reads
        // through freed (or re-tenanted) memory. Invalidate unconditionally —
        // O(1); the next decode sync rebuilds from the current table.
        slot.invalidate_gpu_chunks();
        let block_end = slot.block_count();

        Ok((block_start, block_end))
    }

    /// Replace the sealed blocks `[start, end)` of the sequence at `batch_idx`
    /// with `sealed`'s chunks, leaving every block outside the range where it
    /// stands. Returns the block the inserted chunks end at.
    ///
    /// The mid-sequence counterpart of [`Self::inject_sealed_at_tail`], and as
    /// pure: no kernel, no DMA, Arc-shared gids. With `sealed` empty it evicts
    /// the range; with `start == end` it inserts. A rebuild uses it to put a
    /// changed piece where the projection placed it without re-injecting the
    /// unchanged pieces after it — every later block's RoPE base is cumulative
    /// usage, so it moves with them. The caller moves the sequence's logical
    /// offset by the change in tokens.
    pub fn splice_sequence_blocks(
        &self,
        batch_idx: usize,
        start: usize,
        end: usize,
        sealed: &SealedSequence,
    ) -> Result<usize> {
        let held = {
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
                        "splice_sequence_blocks: slot {batch_idx} not allocated"
                    ))
                })?
        };
        let n = sealed.chunks.len();
        self.ensure_max_blocks(held.saturating_sub(end.saturating_sub(start)) + n)?;
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let slot = state.sequences[batch_idx].as_mut().ok_or_else(|| {
            candle::Error::Msg(format!(
                "splice_sequence_blocks: slot {batch_idx} not allocated"
            ))
        })?;
        slot.splice_sealed_chunks(start, end, sealed_windows(sealed))?;
        Ok(start + n)
    }
}

/// `sealed`'s chunks as windows a slot holds: Arc-shared gids, the chunk's own
/// fill, and its resident record — the injected chunk names the sealed chunk's
/// gids, so the record stays valid.
pub(super) fn sealed_windows(sealed: &SealedSequence) -> Vec<ChunkWindow> {
    sealed
        .chunks
        .iter()
        .map(|sc| ChunkWindow {
            gids: sc.gids.clone(),
            usage: sc.token_count as u32,
            lead: 0,
            offset: sc.offset,
            k_pal: Arc::clone(&sc.k_pal),
            v_pal: Arc::clone(&sc.v_pal),
            k_scale: Arc::clone(&sc.k_scale),
            v_scale: Arc::clone(&sc.v_scale),
            k_fmt: Arc::clone(&sc.k_fmt),
            v_fmt: Arc::clone(&sc.v_fmt),
            meta: sc.meta.clone(),
            q_sign: sc.q_sign.clone(),
        })
        .collect()
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
