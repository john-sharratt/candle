//! Snapshots a slot's chunks as a sealed record and truncates or trims a slot
//! back to an earlier point.
//!
//! `record_turn` captures every chunk, including the partial trailing one,
//! because dropping the partial chunk would lose up to a chunk of tokens from
//! the sealed unit. The snapshot shares the live chunks, so a resident record
//! stays valid. Truncation by blocks or by tokens frees the chunks past the cut.
//! A cut that falls in sealed ground is clamped to the sealed boundary. A trim
//! that stays inside the landing chunk changes only that chunk's length, and
//! marks the chunk for the next sync instead of re-serialising the whole slot.

use super::*;

impl ChunkedKvBacking {
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
        self.record_turn_blocks(batch_idx, 0, usize::MAX)
    }

    /// Like [`Self::record_turn`] but restricted to the block-index range
    /// `[start_block, end_block)` (clamped to the slot's block count). The walk
    /// cost scales with the RANGE, not the slot — the projection assembler's
    /// glue-island capture snapshots a couple of chunks out of a
    /// multi-hundred-block slot, where a whole-slot record per layer costs tens
    /// of milliseconds per wave. Per-chunk sealed metadata is self-contained
    /// (un-rotated K, positions recomputed from the destination's cumulative
    /// usage), so a ranged record equals the full record sliced to the range.
    pub fn record_turn_blocks(
        &self,
        batch_idx: usize,
        start_block: usize,
        end_block: usize,
    ) -> Result<SealedSequence> {
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
        let all_chunks = slot.chunks_slice();
        let end = end_block.min(all_chunks.len());
        let start = start_block.min(end);
        let range_chunks = &all_chunks[start..end];

        // A chunk's byte size sums its bands' own payloads, so this needs the
        // geometry and nothing from the arenas. It used to resolve a per-arena
        // stride snapshot here, which dominated the per-scope cost.
        let elems_per_chunk = self.inner.elems_per_chunk();

        // Build SealedChunks from every block in the requested range of the
        // slot's block table, including a trailing partial.  No positional
        // state is captured — K bytes in the chunks are un-rotated, and
        // RoPE is applied at the latest responsible moment by the
        // attention kernel using a slice_rope recomputed from the
        // destination slot's cumulative usage.  See `SealedChunk`
        // docs.
        let sealed_chunks: Vec<SealedChunk> = range_chunks
            .iter()
            .map(|cw| {
                let byte_size = cw.byte_size(elems_per_chunk);
                SealedChunk {
                    gids: cw.gids.clone(),
                    offset: cw.offset,
                    token_count: cw.usage as u16,
                    k_pal: cw.k_pal.clone(),
                    v_pal: cw.v_pal.clone(),
                    k_scale: cw.k_scale.clone(),
                    v_scale: cw.v_scale.clone(),
                    k_fmt: cw.k_fmt.clone(),
                    v_fmt: cw.v_fmt.clone(),
                    byte_size,
                    // Snapshot shares the live chunk's GIDs, so its record (if
                    // resident) stays valid — propagate the handle.
                    meta: cw.meta.clone(),
                    q_sign: cw.q_sign.clone(),
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

    /// Snapshot the slot's writer-owned tail chunks (`[writer_start_idx..end)`)
    /// and remove them from the slot.
    ///
    /// Used by the stateless-slot rebuild path: the scheduler takes this
    /// snapshot before calling [`Self::truncate_sequence_to_blocks`] and
    /// [`Self::inject_sealed_at_tail`] to refresh the prefix, then restores the
    /// tail via [`Self::extend_writer_tail`]. The returned [`WriterTail`] holds
    /// RAII refs that keep the underlying arena chunks alive across the
    /// truncate, so no bytes are copied.
    ///
    /// At turn-boundary projection (the common case) the tail is empty
    /// and this is effectively a no-op.
    pub fn split_off_writer_tail(
        &self,
        batch_idx: usize,
    ) -> Result<super::super::types::WriterTail> {
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
        Ok(super::super::types::WriterTail { chunks })
    }

    /// Restore the writer-owned tail chunks captured by
    /// [`Self::split_off_writer_tail`]. Appends them to the slot's
    /// chunk list after whatever prefix has been re-injected; the
    /// writer boundary stays where the prefix's
    /// [`Self::inject_sealed_at_tail`] left it — unless the tail carries a
    /// chunk the paged stores cannot write (a mid-turn seal quantizes the
    /// partial writer too), in which case the extend moves it past that chunk.
    pub fn extend_writer_tail(
        &self,
        batch_idx: usize,
        tail: super::super::types::WriterTail,
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
        // so any writes go into existing chunks past it. Callers that
        // need the kept blocks fully immutable follow up with [`Self::seal_writer_boundary`].
        if slot.writer_start_idx() > slot.block_count() {
            slot.set_writer_start_idx(slot.block_count());
        }
        let new_tokens: usize = slot.chunks_slice().iter().map(|c| c.usage as usize).sum();
        Ok(new_tokens)
    }

    /// Mark every block the slot currently holds as immutable prefix: later
    /// writes append AFTER them in a fresh writer chunk, never extend a kept
    /// partial/empty tail block. A caller that seals a range anchored at
    /// exactly the kept block count needs this after its truncate — a write
    /// landing below the anchor would fall outside the sealed span and
    /// silently drop from persistence.
    pub fn seal_writer_boundary(&self, batch_idx: usize) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let slot = state.sequences[batch_idx].as_mut().ok_or_else(|| {
            candle::Error::Msg(format!(
                "seal_writer_boundary: slot {} not allocated",
                batch_idx
            ))
        })?;
        slot.set_writer_start_idx(slot.block_count());
        Ok(())
    }

    /// Truncate the sequence's **writer-owned tail** to `target_tokens`
    /// cum-tokens. Makes an offset-`N` re-prefill idempotent: a caller (e.g.
    /// the bench harness's repeat loop) that re-runs a prefill at the same
    /// offset must not stack stale tail chunks on top of the previous run.
    /// No-op when the sequence already holds ≤ `target_tokens` tokens (growth
    /// is handled by `set_len`).
    ///
    /// **Sealed ground is not this operation's to cut.** Chunks below
    /// `writer_start_idx` are Arc-shared projected layout — substrate borrows,
    /// injected sections, and reprojection's reserved glue gaps — so a target
    /// that falls inside them is clamped up to the sealed boundary and only
    /// the writable chunks beyond it are freed. That case is not hypothetical:
    /// a slot with a **deferred glue fire** counts its reserved gap tokens in
    /// the chunk windows (the reproject stamps them at assembly so the fire
    /// can scatter into place) but not in the scheduler's offset (glue "must
    /// not advance its slot" until it fires), so every such slot arrives here
    /// with `target` short of the sealed cum by exactly the pending glue. This
    /// used to be a bail that one caller silently discarded — which made the
    /// no-op accidental and the first caller to *propagate* errors killed the
    /// wave with `target 1334 cuts into the Arc-shared prefix`. The clamp is
    /// that no-op as a contract instead of an accident, and it is also what
    /// keeps stale-tail cutting alive on the same slot: writable chunks past
    /// the sealed boundary still go.
    pub fn truncate_sequence_to_tokens(
        &self,
        batch_idx: usize,
        target_tokens: usize,
    ) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let slot = match state.sequences.get_mut(batch_idx).and_then(|s| s.as_mut()) {
            Some(s) => s,
            None => return Ok(()),
        };
        let n = slot.block_count();
        if n == 0 {
            return Ok(());
        }
        // `target_tokens` is ABSOLUTE (the session offset space); chunk usages
        // are RESIDENT (the sliding-window ring drops `base_pos` front tokens).
        // Translate, or a truncate on a slid ring silently no-ops (`resident
        // total <= absolute target` is always true once the ring has slid) —
        // the speculative rollback then leaves rejected draft rows committed
        // and the arena length drifts ahead of the true content every partial
        // accept. Zero until the first slide, so unslid rings are unchanged.
        let target_tokens = target_tokens.saturating_sub(slot.base_pos() as usize);
        let total: usize = slot.chunks_slice().iter().map(|c| c.usage as usize).sum();
        if total <= target_tokens {
            return Ok(());
        }
        let writer_start = slot.writer_start_idx();
        let sealed_cum: usize = slot.chunks_slice()[..writer_start.min(n)]
            .iter()
            .map(|c| c.usage as usize)
            .sum();
        let target_tokens = target_tokens.max(sealed_cum);
        if total <= target_tokens {
            return Ok(());
        }
        let mut cum = 0usize;
        let mut patch_writer = false;
        for i in 0..n {
            let usage = slot.chunks_slice()[i].usage as usize;
            // The `i + 1` bound keeps the landing at or past the last sealed
            // chunk even when trailing sealed chunks hold zero tokens — landing
            // earlier would `truncate_chunks` sealed layout away on a token
            // count that never covered it.
            if cum + usage >= target_tokens && i + 1 >= writer_start.min(n) {
                let keep = target_tokens - cum;
                // The clamp above puts the cut at or past the sealed boundary:
                // a chunk below `writer_start` can be landed on only with
                // `keep == usage`, which trims nothing from it.
                // Keep chunks 0..=i (chunk i trimmed to `keep` tokens, possibly 0 →
                // it becomes the empty writer chunk) and free everything after.
                slot.chunk_at_mut(i).unwrap().usage = keep as u32;
                slot.truncate_chunks(i + 1);
                if slot.writer_start_idx() > slot.block_count() {
                    slot.set_writer_start_idx(slot.block_count());
                }
                if i + 1 == n {
                    // The trim stayed INSIDE the landing chunk — no chunk was
                    // freed, so the cached decode slot buffer differs from
                    // host state in exactly one field: that chunk's length.
                    // Mark the writer region (below, after the state lock)
                    // for the next sync instead of dropping the whole
                    // serialized buffer. A speculative partial accept trims
                    // here on nearly every step, and an invalidate would force
                    // the next verify's full 43-layer re-serialisation — the
                    // mark is what lets a partial-accept step keep
                    // plain-wave-cost metadata.
                    //
                    // **From the landing chunk, which is the one that changed.**
                    // Left unnamed, the mark falls back to the writer boundary —
                    // chunk 0 for a turn written from its first token, so every
                    // rollback re-serialised the whole sequence on every layer:
                    // 1.6 ms a layer at 128K, enough to leave the GPU waiting on
                    // the host for most of each verify step.
                    slot.note_fill_from(i);
                    patch_writer = true;
                } else {
                    // Chunks were freed: the serialized chunk count is wrong
                    // and only a rebuild can fix it.
                    slot.invalidate_gpu_chunks();
                }
                break;
            }
            cum += usage;
        }
        drop(state);
        if patch_writer {
            self.mark_decode_writer_stale(&[(batch_idx, 0)])?;
        }
        Ok(())
    }
}
