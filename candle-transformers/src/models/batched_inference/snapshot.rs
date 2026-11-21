//! Captures a sequence's KV as sealed snapshots, one per layer, for turn records
//! and for moving the sequence between the hot and warm tiers.
//!
//! A trailing empty writer chunk is structure rather than content, and the layers
//! do not agree on having one. Every snapshot path drops it, since keeping it
//! would leave an interior empty chunk once anything is appended. The whole, tail
//! and block-range snapshots must agree, because callers resolve a range against
//! one and slice another. Tier moves run per layer and carry the same token
//! layout the attention kernel reads.

use super::*;

impl BatchedInferenceSession {
    /// Record a turn boundary for a sequence, capturing the current KV state.
    ///
    /// Commits the active window to `recorded_metas` and returns a `SealedSequence`
    /// snapshot of the just-committed turn.
    ///
    /// No-op when the sequence has zero tokens.
    pub fn record_turn(&mut self, idx: usize) -> Result<candle_nn::kv_cache::SealedSequence> {
        let backing = self
            .backings
            .first()
            .ok_or_else(|| candle::Error::Msg("record_turn: no backings".into()))?;
        backing
            .record_turn(idx)
            .map_err(|e| candle::Error::Msg(format!("record_turn: {e}")))
    }

    /// Read the per-head GID vectors for a sequence's block table (from layer 0).
    ///
    /// Returns one `HeadGids` per block (length = `2 * n_kv_head`),
    /// preserving per-head arena assignments.
    /// All layers share the same block table, so reading from layer 0 suffices.
    pub fn slot_chunk_ids(&self, seq_idx: usize) -> Result<Vec<HeadGids>> {
        if let Some(backing) = self.backings.first() {
            backing.slot_chunk_ids(seq_idx)
        } else {
            Ok(Vec::new())
        }
    }

    /// Snapshot the current KV state of a sequence as a [`SealedSequence`].
    ///
    /// Captures the sequence's current chunk IDs and token counts.  The last
    /// chunk may be partial (fewer than `chunk_size` tokens); its token count
    /// is computed from the current sequence offset.  In the turn-recording
    /// design this also **commits** the turn boundary (moves the active window
    /// to `recorded_metas`).
    ///
    /// Returns an error if the sequence is not allocated.
    /// The trailing empty chunk is dropped here for the same reason
    /// [`Self::snapshot_sequence_per_layer`] drops it, and the two **must** agree:
    /// they are two views of the same layer, and callers resolve a block range
    /// against one and then slice the other. Without this they differed by exactly
    /// one whenever the writer chunk was empty — `record_turn` reports it because the
    /// slot really holds it — and `resolve_seal_range` clamped a seal to the longer
    /// count. Measured: `seal range 40..49 is outside layer 0's 48 sealed chunk(s)`,
    /// deterministic on every boot from a fresh substrate, which made the daemon
    /// unable to start at all; an existing substrate hid it by leaving nothing to
    /// ingest. Lossless, by the argument that justifies the drop downstream: an empty
    /// chunk holds no token, so no range and no position moves.
    pub fn snapshot_sequence(&self, idx: usize) -> Result<candle_nn::kv_cache::SealedSequence> {
        let backing = self
            .backings
            .first()
            .ok_or_else(|| candle::Error::Msg("snapshot_sequence: no backings".into()))?;
        let mut seq = backing
            .record_turn(idx)
            .map_err(|e| candle::Error::Msg(format!("snapshot_sequence: {e}")))?;
        seq.drop_empty_tail();
        Ok(seq)
    }

    /// Snapshot a sequence into per-layer `SealedSequence`s, one
    /// entry per backing.  Used at turn-end to capture per-layer KV
    /// bytes ready for `seal_to_cpu_per_layer`.
    pub fn snapshot_sequence_per_layer(
        &self,
        idx: usize,
    ) -> Result<Vec<candle_nn::kv_cache::SealedSequence>> {
        let mut out = Vec::with_capacity(self.backings.len());
        for backing in &self.backings {
            let mut seq = backing
                .record_turn(idx)
                .map_err(|e| candle::Error::Msg(format!("snapshot_sequence_per_layer: {e}")))?;
            // **A trailing empty chunk is structure, not content, and it does
            // not travel.** `record_turn` takes every block a layer holds, the
            // empty writer chunk included, and the layers do not agree on
            // whether they have one: `reconcile_block_counts` pads a lagging
            // layer with exactly such a chunk, and a windowed creep leaves one
            // on the layers of an earlier window. Captured, that difference is
            // harmless only until something is appended after it — then the
            // empty chunk is *interior*, the layers' token windows are offset
            // by one chunk against each other, and no repair downstream can
            // tell which side is right (`heal_tail_divergence` refuses it as
            // mid-history corruption). Dropping it here loses nothing (an
            // empty chunk adds nothing to the cumulative count, so no position
            // moves) and is what the batched seal path has always done before
            // persisting — see `SealedSequence::drop_empty_tail`.
            seq.drop_empty_tail();
            out.push(seq);
        }
        assert_sealed_layers_aligned(&out, idx, "snapshot_sequence_per_layer")?;
        Ok(out)
    }

    /// [`Self::snapshot_sequence_per_layer`] from block `start_block` to the
    /// end, on every layer — the full snapshot sliced to `[start_block..]`,
    /// trailing empty writer chunk dropped the same way, at the cost of the
    /// tail alone.
    ///
    /// What a reprojection takes of the turn it is decoding: a few blocks off
    /// the end of a slot that holds the whole projected context — ~9,300
    /// blocks a layer at a 300K-token prompt, every one of which the full
    /// snapshot built a sealed record for on every reprojection only for all
    /// but the tail to be thrown away.
    pub fn snapshot_sequence_tail(
        &self,
        idx: usize,
        start_block: usize,
    ) -> Result<Vec<candle_nn::kv_cache::SealedSequence>> {
        let mut out = Vec::with_capacity(self.backings.len());
        for backing in &self.backings {
            let mut seq = backing
                .record_turn_blocks(idx, start_block, usize::MAX)
                .map_err(|e| candle::Error::Msg(format!("snapshot_sequence_tail: {e}")))?;
            // See `snapshot_sequence_per_layer`: the trailing empty chunk is
            // structure, and the layers disagree about having one.
            seq.drop_empty_tail();
            out.push(seq);
        }
        assert_sealed_layers_aligned(&out, idx, "snapshot_sequence_tail")?;
        Ok(out)
    }

    /// [`Self::snapshot_sequence_per_layer`] restricted to the block-index range
    /// `[start_block, end_block)` on every layer. Cost scales with the range —
    /// the glue-island capture snapshots a couple of chunks out of a
    /// multi-hundred-block slot. The caller guarantees the layers' block tables
    /// are aligned over the range (the projection walk builds them uniformly;
    /// same contract as `slice_per_layer_sealed` over a full snapshot).
    pub fn snapshot_sequence_blocks(
        &self,
        idx: usize,
        start_block: usize,
        end_block: usize,
    ) -> Result<Vec<candle_nn::kv_cache::SealedSequence>> {
        let mut out = Vec::with_capacity(self.backings.len());
        for backing in &self.backings {
            out.push(
                backing
                    .record_turn_blocks(idx, start_block, end_block)
                    .map_err(|e| candle::Error::Msg(format!("snapshot_sequence_blocks: {e}")))?,
            );
        }
        // The alignment this method's contract *assumes* of its caller, checked
        // rather than trusted — see `assert_sealed_layers_aligned`.
        assert_sealed_layers_aligned(&out, idx, "snapshot_sequence_blocks")?;
        Ok(out)
    }

    /// Migrate a per-layer sealed snapshot from the GPU (hot) tier to the CPU (warm) tier.
    ///
    /// `sealed` must have one entry per layer (same length as `self.backings()`).
    /// Each entry is migrated by the matching layer's [`ChunkedKvBacking`].
    /// Returns a new per-layer vec with CPU-resident [`SealedSequence`]s.
    pub fn sealed_to_cpu(
        &self,
        sealed: &[candle_nn::kv_cache::SealedSequence],
    ) -> Result<Vec<candle_nn::kv_cache::SealedSequence>> {
        sealed
            .iter()
            .zip(self.backings.iter())
            .map(|(seq, backing)| backing.migrate_sealed_to_cpu(seq))
            .collect()
    }

    /// Migrate a per-layer sealed snapshot from the CPU (warm) tier back to the GPU (hot) tier.
    ///
    /// Symmetric inverse of [`sealed_to_cpu`].
    pub fn sealed_to_gpu(
        &self,
        sealed: &[candle_nn::kv_cache::SealedSequence],
    ) -> Result<Vec<candle_nn::kv_cache::SealedSequence>> {
        sealed
            .iter()
            .zip(self.backings.iter())
            .map(|(seq, backing)| backing.migrate_sealed_to_gpu(seq))
            .collect()
    }

    /// Read raw quantized bytes for one block across **all** layers.
    ///
    /// Called per-block during Hot→Warm eviction.  Returns one `(k_bytes, v_bytes)`
    /// pair per layer in the session.  The bytes are in the token-oriented layout
    /// produced by `reconcile` — identical to what the attention kernel reads.
    ///
    /// # Errors
    /// Fails if any layer's block is still in a float arena (not yet reconciled).
    pub fn read_all_layers_raw_sealed_chunk(
        &self,
        seq_idx: usize,
        block_idx: usize,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.backings
            .iter()
            .map(|backing| backing.read_raw_sealed_chunk(seq_idx, block_idx))
            .collect()
    }

    /// Write raw quantized bytes for one block across **all** layers.
    ///
    /// Called per-block during Warm→Hot restore.  The bytes must be in the
    /// same token-oriented layout that `reconcile` produces (as returned by
    /// [`read_all_layers_raw_sealed_chunk`]).  No format conversion is performed.
    ///
    /// After writing all blocks, call `advance_sequence` with the total token count.
    pub fn write_all_layers_raw_sealed_chunk(
        &self,
        seq_idx: usize,
        block_idx: usize,
        kv_per_layer: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<()> {
        for (backing, (k_bytes, v_bytes)) in self.backings.iter().zip(kv_per_layer.iter()) {
            backing.write_raw_sealed_chunk(
                seq_idx,
                block_idx,
                k_bytes,
                v_bytes,
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
            )?;
        }
        Ok(())
    }
}
