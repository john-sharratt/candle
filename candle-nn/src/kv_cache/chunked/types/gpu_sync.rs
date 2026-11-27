//! Keeps a sequence's cached GPU slot-state buffer in step with its host chunk
//! list, so the decode kernel never reads a stale or misshapen table.
//!
//! Commits made outside the decode kernel, such as a prefill's length writes,
//! only mark the writer region stale. The next sync re-serialises the chunks
//! those commits filled, in one upload rather than one per chunk. Appends extend
//! the buffer's tail, and a layout that cannot hold them is rebuilt in full. The
//! sync also checks the entry count the kernel will be promised against the
//! count the buffer holds, because a larger promise would walk off the end of
//! the slot into unmapped memory and fault the whole device context.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecodeGpuChunksSyncKind {
    Empty,
    Rebuild,
    /// The live buffer served as it stood.
    Reuse,
    /// A commit outside the decode kernel had marked the writer region stale,
    /// and its `entries` were re-serialised before the live buffer was handed
    /// back — from the first chunk a pending commit changed to the writer.
    Resync {
        entries: usize,
    },
    /// Chunks had been appended, and the buffer was brought up to them by
    /// serialising `entries` of its tail — from the previous writer, or from
    /// the first chunk a pending commit filled when that is earlier.
    Extend {
        entries: usize,
    },
}

impl SequenceState {
    /// Clear the GPU slot-state buffer, forcing a full rebuild on the next decode step.
    ///
    /// Called after prefill to mark the buffer stale — the prefill kernel wrote
    /// KV data directly and may have changed chunk layout without going through
    /// the decode self-increment path.
    pub(crate) fn invalidate_gpu_chunks(&mut self) {
        self.gpu_chunks.as_mut().clear();
    }

    /// Whether a decode slot buffer is cached for this sequence. A sequence
    /// that has never decoded, or whose buffer a structural mutation cleared,
    /// has none, and rebuilds it in full on its next sync.
    pub(crate) fn has_decode_gpu_chunks(&self) -> bool {
        self.gpu_chunks.n_chunks() != 0
    }

    /// Record a commit made outside the decode kernel — a prefill, a glue
    /// writeback, a speculative verify block, a stencil static run.
    ///
    /// The decode kernel advances the buffer's writer length itself, on the
    /// device; `set_len` advances only the host. Left alone, the next decode
    /// reuses the buffer at the pre-commit length, writes its token over a
    /// committed one, and leaves a slot the host counts unwritten. So the
    /// buffer is marked, and [`Self::sync_decode_gpu_chunks`] — which every
    /// reader of the buffer goes through, the prefill header build as well as
    /// the decode metadata build — re-serialises its writer region before
    /// handing it out.
    ///
    /// Nothing is serialised or uploaded here. A prefill commits once per
    /// layer with that layer's attention still queued on the stream, so work
    /// done at the commit is done while the host should be running ahead, and
    /// an upload issued there queues behind the layer and holds a staging
    /// buffer until it runs. Deferred to the sync, the region is re-serialised
    /// once, by the step that reads it, however many commits came before. A
    /// sequence with no buffer is left unmarked: its next sync rebuilds it.
    pub(crate) fn mark_decode_writer_stale(&mut self) {
        let from = self.fill_from.take().unwrap_or(self.writer_start_idx);
        if self.has_decode_gpu_chunks() {
            self.decode_stale_from = Some(self.decode_stale_from.map_or(from, |s| s.min(from)));
        }
    }

    /// A rollback: the forward taken back may have committed lengths on the
    /// device that the host never recorded, so the decode writer chunk is
    /// re-serialised from the host state by the next sync.
    ///
    /// The decode kernel advances its write chunk's length on the device as it
    /// writes, and the host counts the token only once the step has run. A
    /// forward that fails after its decode attention ran therefore leaves the
    /// device a token ahead of a host that never moved, and a cut to the host's
    /// own length has nothing to trim — so it marks nothing, and the next decode
    /// reused the buffer and attended the failed attempt's token. Only the
    /// writer chunk can be ahead: one decoded token lands in it.
    pub(crate) fn mark_decode_rolled_back(&mut self) {
        if !self.chunks.is_empty() {
            self.note_fill_from(self.decode_write_chunk_idx());
        }
        self.mark_decode_writer_stale();
    }

    /// Record that a host-side length commit changed chunk `idx` first — the
    /// earliest of every such commit since the last sync is kept.
    pub(crate) fn note_fill_from(&mut self, idx: usize) {
        self.fill_from = Some(self.fill_from.map_or(idx, |f| f.min(idx)));
    }

    /// Re-serialise the cached buffer's writer region from the host state —
    /// the work [`Self::mark_decode_writer_stale`] leaves to the next sync.
    ///
    /// `set_len` tops up usages from the writer boundary through consecutive
    /// chunks to the sequence length, so a commit that crossed into a later
    /// chunk left every chunk it filled on the way stale, not only the writer.
    /// The buffer, built from the same chunks (they are allocated up front, so
    /// no append clears it during the write), therefore has the WRITER
    /// REGION's slices — every chunk from the writer boundary to the current
    /// writer — re-serialised under the guard (async H→D on drop); the chunks
    /// below the boundary are shared and unwritten. This is O(chunks the
    /// commit wrote) against dropping and re-uploading the entire per-layer
    /// table, which at depth costs megabytes of pinned realloc and a stream
    /// sync per layer. A missing buffer is left for the sync's full rebuild; a
    /// shape mismatch (defensive) falls back to full invalidation, which the
    /// same sync then rebuilds.
    ///
    /// Answers with the entries re-serialised: zero when there was no buffer to
    /// bring up to date or it was invalidated instead.
    pub(super) fn resync_decode_writer_region(
        &mut self,
        from: usize,
        n_kv_head: usize,
        head_dim: usize,
        arena_info: &[ResolvedArenaInfo],
    ) -> candle::Result<usize> {
        let gpu_n = self.gpu_chunks.n_chunks();
        if gpu_n == 0 {
            return Ok(0);
        }
        if gpu_n != self.chunks.len() {
            self.invalidate_gpu_chunks();
            return Ok(0);
        }
        let wi = self.decode_write_chunk_idx();
        if wi >= gpu_n {
            self.invalidate_gpu_chunks();
            return Ok(0);
        }
        // Every chunk a prefill can have written, not only the writer:
        // `set_len` fills consecutive chunks, so the chunks it filled on the
        // way to the writer carry the buffer's pre-prefill lengths exactly as
        // the writer does — a 200-token prompt leaves six sealed chunks at
        // length 0, and the decode attends the last eight tokens alone. `from`
        // is the first chunk the earliest pending commit filled (or the writer
        // boundary at that commit, when it did not say), so it covers them all.
        let start = from.min(wi);
        // One guard for the whole region: one upload, not one per chunk.
        let region: Vec<usize> = (start..=wi).collect();
        self.update_gpu_chunks_bulk(&region, n_kv_head, head_dim, arena_info)?;
        Ok(region.len())
    }

    /// Re-serialise the GPU buffer slot at `blk` from the current host state
    /// (including updated GIDs), scheduling an async H→D copy on the guard drop.
    ///
    /// No-op if the GPU buffer has not been initialised yet (avoids errors for
    /// sequences that haven't gone through their first decode step).
    pub(in super::super) fn update_gpu_chunk(
        &mut self,
        blk: usize,
        n_kv_head: usize,
        head_dim: usize,
        arena_info: &[ResolvedArenaInfo],
    ) -> candle::Result<()> {
        if self.gpu_chunks.n_chunks() == 0 {
            return Ok(());
        }
        let rope_base: u32 = self.base_pos
            + self.chunks[..blk]
                .iter()
                .map(ChunkWindow::span)
                .sum::<u32>()
            + self.chunks[blk].lead;
        let SequenceState {
            ref chunks,
            ref mut gpu_chunks,
            ..
        } = *self;
        gpu_chunks.as_mut().update_chunk(
            blk,
            &chunks[blk],
            n_kv_head,
            head_dim,
            rope_base,
            arena_info,
        )
    }

    /// Bulk variant of [`Self::update_gpu_chunk`] — serialises every
    /// block in `block_indices` under a **single** `GpuChunksGuard`,
    /// so the guard drop coalesces adjacent indices into one
    /// `memcpy_htod` per contiguous run (instead of one per block).
    ///
    /// Used by the cold-load `alloc_sealed_blocks_bulk` path and by
    /// [`Self::resync_decode_writer_region`], so each pushes its chunks as
    /// one batched upload where the indices allow.
    pub(in super::super) fn update_gpu_chunks_bulk(
        &mut self,
        block_indices: &[usize],
        n_kv_head: usize,
        head_dim: usize,
        arena_info: &[ResolvedArenaInfo],
    ) -> candle::Result<()> {
        if block_indices.is_empty() || self.gpu_chunks.n_chunks() == 0 {
            return Ok(());
        }
        // Seeded at `base_pos` so evicted-front slots keep absolute rope
        // positions; captured before the disjoint-field destructure below.
        let base_pos = self.base_pos;
        let SequenceState {
            ref chunks,
            ref mut gpu_chunks,
            ..
        } = *self;
        // Prefix-sum cumulative usage once so each block's rope_base
        // is an O(1) lookup. The previous `chunks[..blk].iter().sum()`
        // was O(blk) per block — quadratic over a layer's blocks.
        // Only as far as the highest block asked for: a resync touches the
        // last few chunks of a deep sequence, and bases past it are unused.
        let last = block_indices.iter().copied().max().unwrap_or(0);
        let mut rope_bases: Vec<u32> = Vec::with_capacity(last + 1);
        let mut acc: u32 = base_pos;
        for c in chunks.iter().take(last + 1) {
            acc = acc.wrapping_add(c.lead);
            rope_bases.push(acc);
            acc = acc.wrapping_add(c.usage);
        }
        // One guard, scoped over the whole loop — its drop coalesces
        // every dirty chunk into the fewest contiguous memcpy_htod
        // runs the index set allows.
        let mut guard = gpu_chunks.as_mut();
        for &blk in block_indices {
            guard.update_chunk(
                blk,
                &chunks[blk],
                n_kv_head,
                head_dim,
                rope_bases[blk],
                arena_info,
            )?;
        }
        Ok(())
    }

    /// The chunks the serialised slot-state references, for a consumer to hold
    /// across its launch (see `GpuChunks::pins`).
    pub(crate) fn gpu_chunk_pins(&self) -> Arc<Vec<ChunkPin>> {
        self.gpu_chunks.pins()
    }

    /// Rebuild the GPU slot-state buffer for decode using the true sequence length.
    ///
    /// Serialises all chunks into the host copy with per-chunk
    /// `rope_base` values derived from cumulative usage, then uploads to the
    /// device buffer asynchronously.
    ///
    /// Returns `(raw_device_ptr, n_chunks, write_chunk_idx)` where:
    /// - `raw_device_ptr` is the GPU base pointer for the decode kernel,
    /// - `n_chunks` is the number of serialised chunk entries,
    /// - `write_chunk_idx` is the index of the last (writable) chunk.
    ///
    /// `seq_offset` is the current sequence length used to derive the true
    /// token count for the write chunk (overrides the potentially-stale
    /// `chunk.usage` field).
    pub(crate) fn rebuild_decode_gpu_chunks(
        &mut self,
        n_kv_head: usize,
        head_dim: usize,
        seq_offset: usize,
        arena_info: &[ResolvedArenaInfo],
    ) -> candle::Result<(u64, u32, u32)> {
        let n = self.chunks.len();
        if n == 0 {
            return Ok((0, 0, 0));
        }
        // The writer is the first non-full chunk from `writer_start_idx`, NOT
        // `host_n - 1`: trailing empty chunks (a fresh writer past a sealed
        // partial chunk) must be skipped. Its rope base is the cum-token sum of
        // the chunks before it, and its GPU length is the seq_offset-derived
        // fill (overriding the possibly-stale stored usage).
        let wi = self.decode_write_chunk_idx();
        let before_wi: usize = self.chunks[..wi].iter().map(|c| c.usage as usize).sum();
        let write_len = seq_offset.saturating_sub(before_wi) as u16;

        // Borrow chunks and gpu_chunks as disjoint fields simultaneously.
        let base_pos = self.base_pos;
        let SequenceState {
            ref chunks,
            ref mut gpu_chunks,
            ..
        } = *self;
        gpu_chunks.as_mut().rebuild_decode(
            chunks, n_kv_head, head_dim, arena_info, write_len, wi, base_pos,
        )?;
        // Serialised from the host state just now, so no commit is pending.
        self.decode_stale_from = None;

        let ptr = self.gpu_chunks.raw_device_ptr();
        Ok((ptr, n as u32, wi as u32))
    }

    /// Extend the GPU slot-state buffer over the chunks appended since its last
    /// serialisation, returning its header triple and the entries serialised.
    /// `None` — nothing written — when its layout cannot hold them, for the
    /// caller to rebuild.
    ///
    /// The tail is serialised from the earliest entry that can differ from
    /// what a rebuild would write: the previous serialisation's writer (its
    /// length was the offset-derived one, and the kernel has advanced its
    /// device copy since), or, if a commit marked the writer region stale, the
    /// start of that region as [`Self::resync_decode_writer_region`] takes it.
    /// Every entry before it is unchanged, so the buffer comes out as a rebuild
    /// would leave it.
    pub(super) fn extend_decode_gpu_chunks(
        &mut self,
        n_kv_head: usize,
        head_dim: usize,
        seq_offset: usize,
        arena_info: &[ResolvedArenaInfo],
    ) -> candle::Result<Option<((u64, u32, u32), usize)>> {
        let n = self.chunks.len();
        let wi = self.decode_write_chunk_idx();
        let before_wi: usize = self.chunks[..wi].iter().map(|c| c.usage as usize).sum();
        let write_len = seq_offset.saturating_sub(before_wi) as u16;
        // Never past the current writer, whose length is the offset-derived one.
        let mut start = self.gpu_chunks.write_idx().min(wi);
        if let Some(from) = self.decode_stale_from {
            start = start.min(from);
        }
        let base_pos = self.base_pos;
        let SequenceState {
            ref chunks,
            ref mut gpu_chunks,
            ..
        } = *self;
        let Some(entries) = gpu_chunks.as_mut().extend_decode(
            chunks, n_kv_head, head_dim, arena_info, write_len, wi, base_pos, start,
        )?
        else {
            return Ok(None);
        };
        self.decode_stale_from = None;
        Ok(Some((
            (self.gpu_chunks.raw_device_ptr(), n as u32, wi as u32),
            entries,
        )))
    }

    /// The entry count a decode header promises the kernel must be the entry
    /// count the device buffer was built for.
    ///
    /// Split out so the rule is arithmetic that tests without a device — the
    /// same reason [`super::region_pool`]'s `tier_fits` and `claimable` are
    /// free functions.
    ///
    /// `host_n > gpu_n` is the dangerous direction and the only one that can
    /// reach a kernel: the header's walk runs off the end of the slot. `host_n <
    /// gpu_n` is merely stale, but it is equally a sign that a `chunks` mutation
    /// skipped its `gpu_chunks.clear()`, so both are refused.
    pub(super) fn slot_counts_agree(host_n: usize, gpu_n: usize) -> candle::Result<()> {
        if host_n == gpu_n {
            return Ok(());
        }
        candle::bail!(
            "decode slot-state: the host chunk list holds {host_n} chunks but the device \
             buffer was built for {gpu_n}. The header would tell the kernel to walk \
             {host_n} entries through a buffer holding {gpu_n}, reading past the slot — and \
             past its slab, unmapped — as slice records whose pointers are then \
             dereferenced. A mutator of `chunks` did not clear `gpu_chunks`."
        )
    }

    /// Synchronise the GPU slot-state buffer for decode.
    ///
    /// The decode hot path trusts the cached GPU slot buffer whenever it exists.
    /// Structural mutations must therefore invalidate it eagerly at the point of
    /// mutation rather than relying on decode-time mismatch heuristics. A
    /// commit made outside the decode kernel only marks it
    /// ([`Self::mark_decode_writer_stale`]), and its writer region is
    /// re-serialised here, first, so no reader is handed pre-commit lengths.
    /// An append ([`Self::push_chunk`]) is extended here — the new tail
    /// serialised, the region a pending commit marked folded into it — or,
    /// when the layout cannot hold it, rebuilt.
    /// `arena_info` covers every arena the rebuild below may serialise, so it
    /// covers the writer region's.
    pub(crate) fn sync_decode_gpu_chunks(
        &mut self,
        n_kv_head: usize,
        head_dim: usize,
        seq_offset: usize,
        arena_info: &[ResolvedArenaInfo],
    ) -> candle::Result<((u64, u32, u32), DecodeGpuChunksSyncKind)> {
        // A fill no stale mark claimed was the decode kernel's own, already on
        // the device; one a later commit marks is noted again by its own
        // `set_len`.
        self.fill_from = None;
        // A slot standing on held ground being vacated moves the way a promotion
        // does: a rebuild into a slot claimed afresh, from the host state.
        if !self.chunks.is_empty() && self.gpu_chunks.asked_to_move() {
            let rebuilt =
                self.rebuild_decode_gpu_chunks(n_kv_head, head_dim, seq_offset, arena_info)?;
            return Ok((rebuilt, DecodeGpuChunksSyncKind::Rebuild));
        }
        if self.gpu_chunks.appended() {
            if let Some((extended, entries)) =
                self.extend_decode_gpu_chunks(n_kv_head, head_dim, seq_offset, arena_info)?
            {
                return Ok((extended, DecodeGpuChunksSyncKind::Extend { entries }));
            }
            let rebuilt =
                self.rebuild_decode_gpu_chunks(n_kv_head, head_dim, seq_offset, arena_info)?;
            return Ok((rebuilt, DecodeGpuChunksSyncKind::Rebuild));
        }
        let mut resynced = None;
        if let Some(from) = self.decode_stale_from {
            // Cleared only once the region is current: a resync that fails
            // leaves the mark for the next sync, not a buffer that reads as up
            // to date and is not.
            let entries =
                self.resync_decode_writer_region(from, n_kv_head, head_dim, arena_info)?;
            self.decode_stale_from = None;
            resynced = Some(entries);
        }
        let host_n = self.chunks.len();
        if host_n == 0 {
            return Ok(((0, 0, 0), DecodeGpuChunksSyncKind::Empty));
        }

        if self.gpu_chunks.n_chunks() == 0 {
            let rebuilt =
                self.rebuild_decode_gpu_chunks(n_kv_head, head_dim, seq_offset, arena_info)?;
            return Ok((rebuilt, DecodeGpuChunksSyncKind::Rebuild));
        }

        // **The count the kernel is promised must be the count the buffer holds.**
        //
        // `host_n` is the HOST chunk list's length; the device buffer was built
        // for `gpu_chunks.n_chunks()`. The kernel indexes the buffer with the
        // former, so if it ever exceeded the latter the walk would run off the
        // end of the slot — and past the end of its slab it is unmapped memory,
        // read as `KvHead` records whose `kvheads_ptr` fields are then
        // dereferenced. That is a device-side out-of-range access on every warp
        // at once: it poisons the context and takes the process down, with the
        // error surfacing at whatever unrelated call synchronises next.
        //
        // The two agree today, and only because all seven mutators of `chunks`
        // remember to say so: `push_chunk` marks the append, which the extend
        // above serves, and `clear_chunks`, `drain_front_chunks`,
        // `prepend_chunks`, `replace_chunks`, `truncate_chunks` and
        // `extend_chunks` each `gpu_chunks.clear()`, which forces the rebuild
        // above. That is an invariant held by discipline across seven call
        // sites; an eighth mutator that forgets would not fail here, it would
        // fail in a kernel with no way back to the omission. Checking it costs
        // an integer compare against two numbers already in hand — no launch,
        // no allocation — so it is
        // checked rather than trusted (AGENTS.md invariant 1b: validate the
        // agreement, do not paper over it by rebuilding defensively).
        Self::slot_counts_agree(host_n, self.gpu_chunks.n_chunks())?;
        let wi = self.decode_write_chunk_idx();
        let ptr = self.gpu_chunks.raw_device_ptr();
        let kind = match resynced {
            Some(entries) => DecodeGpuChunksSyncKind::Resync { entries },
            None => DecodeGpuChunksSyncKind::Reuse,
        };
        Ok(((ptr, host_n as u32, wi as u32), kind))
    }

    /// Snapshot the current slot-state into the stager `generation`, returning
    /// the immutable copy's device pointer. Call after `sync_decode_gpu_chunks`
    /// has brought `gpu_chunks` to the desired offset. See
    /// [`super::gpu_chunks::GpuChunks::snapshot_into_generation`].
    pub(crate) fn snapshot_gpu_chunks_into(
        &mut self,
        generation: &candle::quantized::pinned_staging::Generation,
        seq_offset: usize,
    ) -> candle::Result<u64> {
        // The write chunk's per-token length is not carried in the live buffer
        // between snapshots (see `snapshot_into_generation`); derive it from the
        // sequence offset (tokens already in the write chunk before this token).
        let wi = self.decode_write_chunk_idx();
        let rope_base: usize = self.chunks[..wi.min(self.chunks.len())]
            .iter()
            .map(|c| c.usage as usize)
            .sum();
        let write_len = seq_offset.saturating_sub(rope_base) as u16;
        self.gpu_chunks
            .snapshot_into_generation(generation, wi, write_len)
    }
}
