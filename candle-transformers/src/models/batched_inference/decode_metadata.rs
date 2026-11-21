//! Builds the device-resident slot headers that decode and prefill kernels read,
//! for one contiguous group of layers per forward.
//!
//! Headers are packed for the group, so a layer's kernel index is its offset from
//! the group's first layer. The group's write capacity is reconciled once, which
//! makes the whole stack one build rather than one per layer, and each layer is
//! then checked against the first layer's write slice. A layer whose block table
//! disagreed would scatter the token into one chunk while attention read another,
//! silently. Sequences whose writes go through explicit gap chunks are kept out
//! of the write-chunk allocation, and sequences that mutate the arena during a
//! forward get an immutable snapshot of their slot state.

use super::*;

impl BatchedInferenceSession {
    /// Build decode kernel metadata for ALL layers from scratch.
    ///
    /// Rebuilds slice and header data directly from backing arenas on every call.
    /// Pass `seq_indices` in the same order used for the forward pass batch.
    /// Returns the device-resident `SlotHeader` buffer holding every layer's
    /// headers (must stay alive until after the kernels) and the per-layer byte
    /// stride; the buffer is `None` when `seq_indices` is empty.
    #[cfg(feature = "cuda")]
    pub fn build_decode_metadata(
        &self,
        seq_indices: &[usize],
        generation: &Generation,
    ) -> Result<(Option<GpuBuf>, u64)> {
        // The STREAM's layers, not every backing: the group a wave steps is
        // reconciled to a common block structure, and a draft head's layer —
        // which this sweep never writes — would drag that repair across the
        // whole group on every decode step, moving the trunk's own write slices
        // under it. The head's pass builds metadata for its layer alone.
        self.build_decode_metadata_at(
            0..self.stream_layers,
            seq_indices,
            generation,
            &[],
            &[],
            &[],
        )
    }

    /// [`Self::build_decode_metadata`] with explicit per-sequence offset
    /// overrides: each `(seq, offset)` pair serializes that slot AS IF the
    /// sequence stood at `offset`, without touching session state. The wave
    /// prefill uses this to pre-build one header snapshot per absorbed token
    /// (token `t` attends `[0, base+t)` + itself — exactly the per-token
    /// decode regime), while other sequences serialize their live offsets.
    ///
    /// `non_writer` names sequences whose rows do NOT scatter through the
    /// decode write slot (glue rows write into explicitly reserved gap chunks
    /// instead). They are EXCLUDED from the `+1` write-chunk ensure: applying
    /// it would allocate a spurious empty chunk past a full/gap tail, which
    /// permanently inflates the slot's block count — the turn-seal range
    /// `[turn_start_parent_blocks, block_count)` then misses the turn's real
    /// blocks and the turn silently never persists.
    ///
    /// `snapshot_seqs` names the sequences whose slot state must be captured as
    /// an IMMUTABLE snapshot copy into `generation` (their `slices_ptr` survives
    /// a later chunk-boundary reallocation) — i.e. any sequence that mutates the
    /// arena during this forward: a prefill absorbing across chunk boundaries, a
    /// glue gap-scatter. Every OTHER sequence (a plain decode row, whose write
    /// chunk is pre-ensured so it never reallocs) keeps the zero-copy LIVE
    /// pointer + on-device write-len commit — the cheap decode path. Passing
    /// `&[]` makes every row live.
    ///
    /// `layers` names the contiguous group of KV layers to describe. The whole
    /// stack (`0..num_layers`) is the ordinary answer, and the group's one
    /// write-capacity reconciliation is what makes it one build rather than
    /// N — every layer of the group must therefore agree on block structure,
    /// which [`ChunkedKvBacking::ensure_for_batch_entries_all`] establishes
    /// over the group before each layer's headers are checked against the
    /// first's.
    ///
    /// A NARROWER group exists for exactly one reason: a layer that legitimately
    /// stands at a different length from the rest. The MTP draft head is one —
    /// its layer runs one position ahead of the trunk's for each speculative
    /// token it proposes, and a whole-stack build would see that as divergence
    /// and "heal" it by truncating the head's proposal away. Asking for
    /// `head..head + 1` describes that layer against its own length, and the
    /// group is trivially self-consistent.
    ///
    /// Headers are packed for the GROUP, so the kernel index of layer `l` is
    /// `l - layers.start`, not `l`.
    pub fn build_decode_metadata_at(
        &self,
        layers: Range<usize>,
        seq_indices: &[usize],
        generation: &Generation,
        offset_overrides: &[(usize, usize)],
        non_writer: &[usize],
        snapshot_seqs: &[usize],
    ) -> Result<(Option<GpuBuf>, u64)> {
        let n_active = seq_indices.len();
        if n_active == 0 {
            return Ok((None, 0));
        }
        if layers.start >= layers.end || layers.end > self.num_layers {
            candle::bail!(
                "decode metadata: layer group {:?} is empty or past the session's \
                 {} KV layers",
                layers,
                self.num_layers
            )
        }
        let group = &self.backings[layers.clone()];
        // Per-row snapshot decision (layer-invariant), aligned with `seq_indices`.
        let snapshot_mask: Vec<bool> = seq_indices
            .iter()
            .map(|s| snapshot_seqs.contains(s))
            .collect();

        let header_stride = n_active * SLOT_HEADER_BYTES;
        let mut all_headers: Vec<u8> = Vec::with_capacity(group.len() * header_stride);

        // Pre-compute per-sequence offsets once (same for all layers).
        let seq_offsets: Vec<(usize, usize)> = seq_indices
            .iter()
            .map(|&seq_idx| {
                let offset = offset_overrides
                    .iter()
                    .find(|(s, _)| *s == seq_idx)
                    .map(|&(_, o)| o)
                    .unwrap_or_else(|| {
                        self.sequences
                            .get(seq_idx)
                            .and_then(|s| s.as_ref())
                            .map_or(0, |s| s.offset)
                    });
                (seq_idx, offset)
            })
            .collect();
        // Each sequence's rung, from the reach this step writes: its offset
        // plus the decoded token. Layer-invariant, so taken once.
        let rungs: Vec<u32> = seq_offsets
            .iter()
            .map(|&(_, offset)| self.rope_rung_for(offset + 1))
            .collect::<Result<_>>()?;

        // `(slice count, write slice)` per sequence, read from the group's
        // FIRST layer. The decode kernels address the write chunk through
        // each layer's own header, so every layer's answer is checked
        // against the first's below rather than assumed equal: a layer whose
        // block table disagrees would scatter the token into one chunk and
        // attend to another. (The decode headers carry no position map —
        // no decode kernel reads one; the prefill and glue uploads in
        // `prefill_utils` build theirs per forward.)
        let mut slot_shape: Vec<(u32, u32)> = Vec::with_capacity(n_active);
        // Ensure backings are sized for the upcoming decode write so the slot's
        // chunks reflect the post-write layout when we read them, and reconcile
        // any block-structure skew between layers — the shape read below from
        // the first layer must describe all of them.
        //
        // Every layer of the GROUP at once, and once per decode step rather than
        // once per layer: only the layers that need an allocation take the write
        // guard, which is what the per-layer form cost 48 times per decoded
        // token in a steady state where the answer is "nothing to allocate".
        //
        // Only for sequences that actually decode-write — a speculative-verify
        // slot replays already-written positions and must not have a write
        // chunk pre-allocated (see `non_writer`).
        let writer_offsets: Vec<(usize, usize)> = seq_offsets
            .iter()
            .copied()
            .filter(|(s, _)| !non_writer.contains(s))
            .collect();
        {
            let _g = span("decode:ensure_layout");
            ChunkedKvBacking::ensure_for_batch_entries_all(group, &writer_offsets, 1)?;
        }
        for &(seq_idx, seq_offset) in &seq_offsets {
            #[cfg(debug_assertions)]
            {
                let cum_tokens: usize = group[0]
                    .visit_live_chunks(seq_idx, |it| it.map(|c| c.token_count as usize).sum())
                    .unwrap_or(0);
                debug_assert_eq!(
                    cum_tokens, seq_offset,
                    "decode metadata: cum_tokens {cum_tokens} != state.offset {seq_offset} for seq {seq_idx}",
                );
            }
            #[cfg(not(debug_assertions))]
            let _ = seq_offset;
            // The new token lands in the WRITE chunk — the first non-full
            // chunk from `writer_start_idx`, NOT the last chunk (which may be a
            // trailing empty sitting past the writer). The same rule as the
            // `write_slice` in `sync_decode_gpu_chunks`, which every layer is
            // checked against below. Read in place: this runs every decode
            // step, and a copy of the chunk list here cost one clone per block.
            let (n_ch, wi) = group[0].decode_write_shape(seq_idx).unwrap_or((0, 0));
            slot_shape.push((n_ch as u32, wi as u32));
        }

        // Each sync kind's host time and slot count over the group's layers: a
        // reuse hands out the live buffer, a resync first re-serialises its stale
        // writer region, an extend its appended tail, a rebuild the whole slot.
        let mut slot_reuse = (std::time::Duration::ZERO, 0u64);
        let mut slot_resync = (std::time::Duration::ZERO, 0u64);
        let mut slot_extend = (std::time::Duration::ZERO, 0u64);
        let mut slot_rebuild = (std::time::Duration::ZERO, 0u64);
        // The one batched upload per layer that copies every slot re-serialised.
        let mut slot_upload = (std::time::Duration::ZERO, 0u64);

        for (slot, backing) in group.iter().enumerate() {
            // Capacity for the upcoming write is ensured for EVERY layer of the
            // group above, before this loop — see `ensure_for_batch_entries_all`.
            let layer_idx = layers.start + slot;

            let arena_info = backing.resolve_arena_info()?;

            // Incrementally sync each sequence's GPU slot-state buffer.
            // Common case: the cached GPU buffer is already valid and we only
            // reuse its pointer. Chunk-boundary case: the layer rebuilds once
            // from authoritative CPU chunk state.
            // The wave prefill builds every per-token snapshot up front, then
            // runs the layer loop, so its `slices_ptr` must be an immutable copy
            // that survives later chunk-boundary reallocations of the live
            // buffer. Live decode (one metadata build per step, used immediately)
            // keeps the zero-copy live pointer + on-device write-len commit.
            // Per-row live-or-snapshot: only the rows in `snapshot_seqs` pay the
            // immutable copy; decode rows keep the live pointer (mask all-false ⇒
            // the whole wave is the cheap live path).
            let (seq_ptrs, sync_stats) = backing.sync_decode_gpu_chunks_snapshot(
                &seq_offsets,
                &arena_info,
                generation,
                &snapshot_mask,
            )?;
            slot_reuse.0 += sync_stats.reuse_time;
            slot_reuse.1 += sync_stats.reuses;
            slot_resync.0 += sync_stats.resync_time;
            slot_resync.1 += sync_stats.resyncs;
            slot_extend.0 += sync_stats.extend_time;
            slot_extend.1 += sync_stats.extends;
            slot_rebuild.0 += sync_stats.rebuild_time;
            slot_rebuild.1 += sync_stats.rebuilds;
            if sync_stats.resyncs + sync_stats.extends + sync_stats.rebuilds > 0 {
                slot_upload.0 += sync_stats.upload_time;
                slot_upload.1 += 1;
            }

            // Append this layer's headers (one `SlotHeader` per sequence; the
            // position map field is 0 — see `slot_shape` above).
            for (i, &(ptr, n_slices, write_slice)) in seq_ptrs.iter().enumerate() {
                // The kernel SCATTERS the new token through this layer's own
                // `write_slice`. A layer whose block table disagrees with the
                // group's writes the token into one chunk while the group's
                // shared selection attends to another — silently, with no
                // fault and no wrong-looking number anywhere.
                // `ensure_for_batch_entries_all` reconciles the layers before
                // this point; this is where that is worth confirming, because
                // both values are already in hand.
                let (exp_slices, exp_write) = slot_shape[i];
                if (n_slices, write_slice) != (exp_slices, exp_write) {
                    candle::bail!(
                        "decode metadata: layer {layer_idx} describes sequence {} as \
                         {n_slices} slices writing slice {write_slice}, but layer {} of \
                         the group {layers:?} describes it as {exp_slices} slices writing \
                         slice {exp_write}",
                        seq_offsets[i].0,
                        layers.start
                    )
                }
                SlotHeaderHost {
                    n_slices,
                    write_slice,
                    slices_ptr: ptr,
                    position_map_ptr: 0,
                    rope_rung: rungs[i],
                }
                .write(&mut all_headers);
            }
        }

        for (name, (time, count)) in [
            ("decode:slot_reuse", slot_reuse),
            ("decode:slot_resync", slot_resync),
            ("decode:slot_extend", slot_extend),
            ("decode:slot_rebuild", slot_rebuild),
            ("decode:slot_upload", slot_upload),
        ] {
            if count > 0 {
                pipeline_record_duration(name, time, count);
            }
        }

        // The group's headers go to DEVICE memory in one pinned → GPU copy.
        // Every block of every attention launch opens on its slot header and
        // hangs its whole slice chain off it, so the header is the one
        // latency-critical read of the kernel: over the zero-copy mapping
        // that is a PCIe round trip per warp (measured ~11% of the sparse
        // decode kernel), resident it is an L2 hit.
        let total = group.len() * header_stride;
        let mut pinned = generation.alloc(total)?;
        pinned.copy_from_slice(&all_headers);
        let gpu_buf = generation.submit_resident(pinned)?;
        Ok((Some(gpu_buf), header_stride as u64))
    }
}
