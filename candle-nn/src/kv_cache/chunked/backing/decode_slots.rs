//! Keeps each sequence's cached decode slot buffer consistent with its chunks.
//!
//! A slot buffer caches band and record addresses, and it is reused whenever the
//! chunk count still matches. A compaction never changes that count, so ground
//! that moves must invalidate the buffers that name it. A buffer is dropped only
//! when it names replaced ground, since clearing a clean one costs a
//! re-serialization. Re-serialized slots are uploaded in one batch under the state
//! lock.

use super::*;

impl ChunkedKvBacking {
    /// Throw away every cached decode buffer in this backing that was serialised from
    /// ground a compaction's sweep replaced, and answer which slots those were.
    ///
    /// **A buffer caches addresses it does not own the meaning of.** Each slice header
    /// carries a `kvheads_ptr` — its chunk's record address — and band addresses behind
    /// it, and the buffer is reused whenever the chunk *count* agrees, which a compaction
    /// never changes. `SequenceState::rewrite_for_compaction` clears the buffer of a slot
    /// whose own chunks it rewrote; this catches a buffer that names replaced ground
    /// through anything else.
    ///
    /// **Decided from the buffer's own pins, not by clearing everything.** Every buffer
    /// holds a [`ChunkPin`](super::gpu_chunks::ChunkPin) per serialised chunk, so it can
    /// say exactly which bands and records its headers name. A buffer naming none of what
    /// the sweep replaced is still exact and is kept: clearing it would cost its slot a
    /// host re-serialisation, and would make the count below mean nothing.
    ///
    /// A non-empty answer is therefore a real finding: a buffer whose slot the sweep did
    /// not rewrite, naming ground it moved other holders off. Not a correctness fault on
    /// its own, since the pins keep that ground alive for as long as the buffer is held,
    /// but it is a holder the sweep does not reach and ground the pass cannot reclaim.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn invalidate_decode_buffers_naming(
        &self,
        replaced: &Replaced,
    ) -> Result<Vec<usize>> {
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let mut cleared = Vec::new();
        for (slot, entry) in state.sequences.iter_mut().enumerate() {
            let Some(seq) = entry.as_mut() else { continue };
            if seq.has_decode_gpu_chunks() && replaced.names_any(&seq.gpu_chunk_pins()) {
                seq.invalidate_gpu_chunks();
                cleared.push(slot);
            }
        }
        Ok(cleared)
    }

    /// Get the current batch capacity (number of sequence slots).
    pub fn batch_capacity(&self) -> usize {
        self.state
            .read()
            .expect("chunked state lock poisoned")
            .sequences
            .len()
    }

    /// Grow the batch capacity to accommodate more sequences.
    ///
    /// This expands the block table and slots vector to hold `new_capacity` sequences.
    /// Existing sequences are preserved.
    pub fn grow_batch_capacity(&self, new_capacity: usize) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        if new_capacity <= state.sequences.len() {
            return Ok(());
        }

        // Grow the slots vector (block table is derived from slots on demand)
        state.sequences.resize(new_capacity, None);

        Ok(())
    }

    /// Begin a stager generation guard.
    ///
    /// While the returned [`Generation`] is alive, the pinned arena will not be
    /// reset — all device-mapped pointers from `submit()` remain valid.  When
    /// the last generation drops, the stream is synchronised and arenas are
    /// reclaimed.
    ///
    /// Returns `None` on non-CUDA devices or if no stager is configured.
    pub fn begin_stager_generation(&self) -> Option<Generation> {
        Some(self.inner.pinned_stager.begin_generation())
    }

    /// Begin a stager generation, panicking if no stager is configured.
    ///
    /// Use this when a Generation is mandatory (e.g. during forward passes
    /// that will run quantization kernels).
    pub fn begin_stager_generation_required(&self) -> Generation {
        self.inner.pinned_stager.begin_generation()
    }

    /// Returns a snapshot of the live chunks for a sequence as [`SealedChunk`]s.
    ///
    /// Used by the slot pool system to build `SlotStateHost` entries from current
    /// chunk data without waiting for a full turn seal.
    ///
    /// No positional state is captured — RoPE is applied at the
    /// latest responsible moment by the attention kernel using a
    /// `slice_rope` recomputed from the destination slot's cumulative
    /// usage.  See [`SealedChunk`] docs.
    ///
    /// Returns `None` if the sequence slot is not allocated.
    /// First writer-owned chunk index for the given sequence slot.
    /// Returns `None` if the slot is unallocated.  See
    /// [`super::types::SequenceState::writer_start_idx`].
    pub fn writer_start_idx_for_seq(&self, batch_idx: usize) -> Option<usize> {
        let state = self.state.read().ok()?;
        let seq = state.sequences.get(batch_idx)?.as_ref()?;
        Some(seq.writer_start_idx())
    }

    /// The sequence's first chunk not yet handed to a gap quantize. `None` if
    /// the slot is unallocated.
    pub(crate) fn gap_cursor_for_seq(&self, batch_idx: usize) -> Option<usize> {
        let state = self.state.read().ok()?;
        let seq = state.sequences.get(batch_idx)?.as_ref()?;
        Some(seq.gap_cursor())
    }

    /// Move the sequence's gap cursor to `cursor`. No-op on an unallocated slot.
    pub(crate) fn set_gap_cursor_for_seq(&self, batch_idx: usize, cursor: usize) {
        let Ok(mut state) = self.state.write() else {
            return;
        };
        if let Some(Some(seq)) = state.sequences.get_mut(batch_idx) {
            seq.set_gap_cursor(cursor);
        }
    }

    /// This layer's shape of the sequence from chunk `from` on, the part a gap
    /// reads (`gap::table`). `None` if the slot is unallocated.
    pub(crate) fn gap_layer_shape(&self, batch_idx: usize, from: usize) -> Option<LayerShape> {
        let state = self.state.read().ok()?;
        let seq = state.sequences.get(batch_idx)?.as_ref()?;
        let writer_start = seq.writer_start_idx();
        let first = from.max(writer_start).min(seq.block_count());
        let chunks = &seq.chunks_slice()[first..];
        Some(LayerShape {
            writer_start,
            first,
            usage: chunks.iter().map(|c| c.usage).collect(),
            native: chunks.iter().map(ChunkWindow::takes_writes).collect(),
        })
    }

    /// `(block count, write chunk)` for a sequence — the shape a decode header
    /// describes, read without copying the chunk list. The write chunk is
    /// [`super::types::SequenceState::decode_write_chunk_idx`], the rule the
    /// slot buffer's `write_slice` follows. `None` if the slot is unallocated.
    pub fn decode_write_shape(&self, batch_idx: usize) -> Option<(usize, usize)> {
        let state = self.state.read().ok()?;
        let seq = state.sequences.get(batch_idx)?.as_ref()?;
        Some((seq.block_count(), seq.decode_write_chunk_idx()))
    }

    /// Per-chunk `(offset, len, cum_before)` real-token window for a sequence — the
    /// exact layout attention reads (writer chunk gets the `seq_offset`-derived
    /// length). A provenance / diagnostic gather consults this to check only real
    /// slots and skip partial-chunk padding. See
    /// [`super::types::SequenceState::provenance_chunk_layout`].
    pub fn provenance_chunk_layout(
        &self,
        batch_idx: usize,
        seq_offset: usize,
    ) -> Vec<(u16, u16, usize)> {
        let Ok(state) = self.state.read() else {
            return Vec::new();
        };
        match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
            Some(seq) => seq.provenance_chunk_layout(seq_offset),
            None => Vec::new(),
        }
    }

    /// Visit `batch_idx`'s live chunks as borrowed [`LiveChunkRef`]s under
    /// the state read lock — the zero-clone path for per-forward metadata
    /// builds, avoiding the per-chunk clones + `arena_byte_size` walks that
    /// dominate host cost at deep prefixes. [`Self::live_chunks_as_sealed`]
    /// serves callers that need OWNED snapshots (seal/persist/migration).
    pub fn visit_live_chunks<R>(
        &self,
        batch_idx: usize,
        f: impl FnOnce(&mut dyn Iterator<Item = LiveChunkRef<'_>>) -> R,
    ) -> Option<R> {
        let state = self.state.read().ok()?;
        let seq = state.sequences.get(batch_idx)?.as_ref()?;
        let mut it = seq.chunks_slice().iter().map(|cw| LiveChunkRef {
            gids: &cw.gids,
            offset: cw.offset,
            token_count: cw.usage as u16,
            k_pal: cw.k_pal.as_slice(),
            v_pal: cw.v_pal.as_slice(),
            k_scale: cw.k_scale.as_slice(),
            v_scale: cw.v_scale.as_slice(),
            k_fmt: cw.k_fmt.as_slice(),
            v_fmt: cw.v_fmt.as_slice(),
            meta: cw.meta.as_ref(),
        });
        Some(f(&mut it))
    }

    /// Drop the sequence's cached GPU slot-state buffer so the next metadata
    /// build re-serialises it from the authoritative CPU chunk state (the
    /// REBUILD path). Needed when host bookkeeping has changed slice lengths or
    /// windows below the writer boundary — `set_len` after a truncate, a block
    /// re-windowed in place — without touching the serialized buffer:
    /// `set_len` never writes it, and a buffer marked by
    /// [`Self::mark_decode_writer_stale`] has only its writer region
    /// re-serialised, so a later build's REUSE path would snapshot stale
    /// lengths.
    pub fn invalidate_decode_slot(&self, batch_idx: usize) {
        if let Ok(mut state) = self.state.write() {
            if let Some(Some(seq)) = state.sequences.get_mut(batch_idx) {
                seq.invalidate_gpu_chunks();
            }
        }
    }

    pub fn live_chunks_as_sealed(&self, batch_idx: usize) -> Option<Vec<SealedChunk>> {
        let elems = self.inner.elems_per_chunk();
        let state = self.state.read().ok()?;
        let seq = state.sequences.get(batch_idx)?.as_ref()?;
        let chunks: Vec<SealedChunk> = seq
            .chunks_slice()
            .iter()
            .map(|cw| {
                let byte_size = cw.byte_size(elems);
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
                    // Live snapshot shares the chunk's GIDs — propagate the
                    // resident record handle so the prefill/glue serializer can
                    // emit its `kvheads_ptr` instead of rebuilding heads.
                    meta: cw.meta.clone(),
                    q_sign: cw.q_sign.clone(),
                }
            })
            .collect();
        Some(chunks)
    }

    /// Record a commit made outside the decode kernel on each sequence's
    /// cached decode slot buffer — see
    /// [`super::types::SequenceState::mark_decode_writer_stale`]. The writer
    /// region is re-serialised by the next sync that reads the buffer, not
    /// here: this runs once per layer of every prefill, with that layer's
    /// attention still queued, and costs one state lock and a flag per
    /// sequence — no arena resolve, no serialisation, no upload.
    pub fn mark_decode_writer_stale(&self, batch_entries: &[(usize, usize)]) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        for &(seq_idx, _) in batch_entries {
            if let Some(Some(seq)) = state.sequences.get_mut(seq_idx) {
                seq.mark_decode_writer_stale();
            }
        }
        Ok(())
    }

    /// Lightweight host-side validation for the paged decode hot path.
    ///
    /// This is intended to catch full or stale tail-slice metadata before the
    /// CUDA kernel is launched, while keeping the check effectively O(batch).
    pub fn validate_decode_batch_state(&self, batch_entries: &[(usize, usize)]) -> Result<()> {
        if batch_entries.is_empty() {
            return Ok(());
        }
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        for &(seq_idx, seq_offset) in batch_entries {
            let seq = state
                .sequences
                .get(seq_idx)
                .and_then(|s| s.as_ref())
                .ok_or_else(|| {
                    let cap = state.sequences.len();
                    let alloc_map: Vec<String> = state
                        .sequences
                        .iter()
                        .enumerate()
                        .map(|(i, s)| format!("{}={}", i, if s.is_some() { "S" } else { "_" }))
                        .collect();
                    candle::Error::Msg(format!(
                        "chunked decode validation failed for batch_idx {}: sequence slot is not allocated (capacity={}, entries={:?}, slots=[{}])",
                        seq_idx,
                        cap,
                        batch_entries,
                        alloc_map.join(",")
                    ))
                })?;
            seq.validate_decode_state(seq_idx, seq_offset)?;
        }
        Ok(())
    }

    /// Check that no two sessions in `batch_entries` share a raw GID value for
    /// any K or V head/palette slot.  Shared GIDs across sessions indicate
    /// aliased arena storage and will cause cross-session KV contamination.
    ///
    /// Returns a list of human-readable violation strings (empty = clean).
    /// Each entry reports the GID, computed GPU pointer, and both owner locations.
    pub fn validate_gid_uniqueness(
        &self,
        batch_entries: &[(usize, usize)],
    ) -> candle::Result<Vec<String>> {
        use crate::kv_cache::arena_table::N_PALETTE;
        use std::collections::HashMap;

        if batch_entries.is_empty() {
            return Ok(Vec::new());
        }

        // Resolve arena info outside the state lock.
        let arena_info = self.resolve_arena_info()?;

        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let n_kv_head = self.inner.n_kv_head;

        // Maps raw GID → (batch_idx, blk, head, pal, is_k)
        let mut seen: HashMap<i64, (usize, usize, usize, usize, bool)> = HashMap::new();
        let mut violations: Vec<String> = Vec::new();

        for &(batch_idx, _) in batch_entries {
            let seq = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                Some(s) => s,
                None => continue,
            };
            for blk in 0..seq.block_count() {
                let cw = match seq.chunk_at(blk) {
                    Some(c) => c,
                    None => continue,
                };
                for h in 0..n_kv_head {
                    for p in 0..N_PALETTE {
                        let k_gid = cw.gids.k_gid_pal(h, p);
                        let v_gid = cw.gids.v_gid_pal(h, p);

                        let check =
                            |raw: i64,
                             is_k: bool,
                             seen: &mut HashMap<i64, (usize, usize, usize, usize, bool)>,
                             violations: &mut Vec<String>| {
                                let stride = GID_STRIDE;
                                let ptr = arena_info
                                    .get(raw as usize / stride)
                                    .map(|ai| {
                                        ai.base_ptr
                                            + (raw as usize % stride) as u64
                                                * ai.chunk_byte_stride as u64
                                    })
                                    .unwrap_or(0);

                                if let Some(&(ob, obl, oh, op, oik)) = seen.get(&raw) {
                                    if ob != batch_idx {
                                        violations.push(format!(
                                            "{}-GID ALIAS: gid={raw} ptr={ptr:#x} \
                                         → session {batch_idx} blk={blk} h={h} p={p} \
                                         vs session {ob} blk={obl} h={oh} p={op} {}",
                                            if is_k { "K" } else { "V" },
                                            if oik { "(K)" } else { "(V)" },
                                        ));
                                    }
                                } else {
                                    seen.insert(raw, (batch_idx, blk, h, p, is_k));
                                }
                            };

                        check(k_gid.raw(), true, &mut seen, &mut violations);
                        check(v_gid.raw(), false, &mut seen, &mut violations);
                    }
                }
            }
        }

        Ok(violations)
    }

    /// Synchronise decode slot-state buffers for the selected sequences and
    /// return both the raw slot pointers and aggregated rebuild-versus-reuse
    /// timing stats. Every slot re-serialised is uploaded by one scatter staged
    /// through `generation` (see `slot_upload_batch`).
    pub fn sync_decode_gpu_chunks(
        &self,
        batch_entries: &[(usize, usize)],
        arena_info: &[crate::kv_cache::arena_table::ResolvedArenaInfo],
        generation: &Generation,
    ) -> candle::Result<(
        Vec<(u64, u32, u32)>,
        Vec<Arc<Vec<super::super::gpu_chunks::ChunkPin>>>,
        DecodeGpuChunkSyncStats,
    )> {
        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let mut results = Vec::with_capacity(batch_entries.len());
        // Each slot's serialisation pins, taken under the lock this call
        // already holds — the chunks its slice array references, for a caller
        // to hold until its launch has retired (see `GpuChunks::pins`). A
        // second pass under a second lock cost more than the pinning it
        // replaced, on a lock the persistence thread contends.
        let mut pins = Vec::with_capacity(batch_entries.len());
        let mut stats = DecodeGpuChunkSyncStats::default();
        // Flushed below under the state lock still held, which owns every slot
        // this loop writes.
        #[cfg(feature = "cuda")]
        let batch = SlotUploadBatch::open(generation)?;
        #[cfg(not(feature = "cuda"))]
        let _ = generation;
        for &(seq_idx, seq_offset) in batch_entries {
            let t_sync = std::time::Instant::now();
            let (result, sync_kind) = if let Some(Some(seq)) = state.sequences.get_mut(seq_idx) {
                seq.validate_decode_state(seq_idx, seq_offset)?;
                let synced =
                    seq.sync_decode_gpu_chunks(n_kv_head, head_dim, seq_offset, arena_info)?;
                pins.push(seq.gpu_chunk_pins());
                synced
            } else {
                pins.push(Arc::new(Vec::new()));
                ((0, 0, 0), DecodeGpuChunksSyncKind::Empty)
            };
            let elapsed = t_sync.elapsed();
            match sync_kind {
                DecodeGpuChunksSyncKind::Rebuild => {
                    stats.rebuilds += 1;
                    stats.rebuild_time += elapsed;
                }
                DecodeGpuChunksSyncKind::Reuse => {
                    stats.reuses += 1;
                    stats.reuse_time += elapsed;
                }
                DecodeGpuChunksSyncKind::Resync { entries } => {
                    stats.resyncs += 1;
                    stats.resync_entries += entries as u64;
                    stats.resync_time += elapsed;
                }
                DecodeGpuChunksSyncKind::Extend { entries } => {
                    stats.extends += 1;
                    stats.extend_entries += entries as u64;
                    stats.extend_time += elapsed;
                }
                DecodeGpuChunksSyncKind::Empty => {
                    stats.empty += 1;
                }
            }
            results.push(result);
        }
        #[cfg(feature = "cuda")]
        {
            let t_upload = std::time::Instant::now();
            batch.flush()?;
            stats.upload_time = t_upload.elapsed();
        }
        drop(state);
        Ok((results, pins, stats))
    }

    /// Like [`Self::sync_decode_gpu_chunks`], but each returned `slices_ptr` is
    /// an immutable copy of that sequence's slot-state placed in the pinned
    /// stager `generation` (device pointer stable for the whole forward),
    /// instead of the live `gpu_chunks` buffer (which reallocates on the next
    /// chunk append). Used by the metadata builder so per-token snapshots built
    /// up front survive later chunk-boundary rebuilds. See
    /// [`super::gpu_chunks::GpuChunks::snapshot_into_generation`].
    pub fn sync_decode_gpu_chunks_snapshot(
        &self,
        batch_entries: &[(usize, usize)],
        arena_info: &[crate::kv_cache::arena_table::ResolvedArenaInfo],
        generation: &candle::quantized::pinned_staging::Generation,
        snapshot_mask: &[bool],
    ) -> candle::Result<(Vec<(u64, u32, u32)>, DecodeGpuChunkSyncStats)> {
        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let mut results = Vec::with_capacity(batch_entries.len());
        let mut stats = DecodeGpuChunkSyncStats::default();
        // Every slot this loop re-serialises is uploaded by ONE scatter when the
        // batch flushes below — under the state lock still held here, which owns
        // every one of those slots (see `slot_upload_batch`).
        #[cfg(feature = "cuda")]
        let batch = SlotUploadBatch::open(generation)?;
        // A speculative-verify wave carries one entry per BLOCK POSITION, so a
        // sequence appears `block_len` times with an identical (seq, offset)
        // key — and an identical snapshot. Snapshot once per key and reuse the
        // copy for the duplicate rows (the state lock is held across the loop,
        // so nothing can mutate the slot buffer between duplicates).
        let mut snap_cache: HashMap<(usize, usize), (u64, u32, u32)> = HashMap::new();
        for (i, &(seq_idx, seq_offset)) in batch_entries.iter().enumerate() {
            let t_sync = std::time::Instant::now();
            // Per-entry: a sequence that mutates the arena during this forward
            // (a prefill absorbing across chunk boundaries, a glue gap-scatter)
            // needs an IMMUTABLE snapshot copy so its `slices_ptr` survives the
            // reallocation; a plain decode row never reallocs (its write chunk is
            // pre-ensured), so it keeps the zero-copy LIVE pointer + on-device
            // write-len commit — the cheap Qwen/Llama decode path.
            let want_snapshot = snapshot_mask.get(i).copied().unwrap_or(true);
            if want_snapshot {
                if let Some(&cached) = snap_cache.get(&(seq_idx, seq_offset)) {
                    stats.reuses += 1;
                    results.push(cached);
                    continue;
                }
            }
            let (result, sync_kind) = if let Some(Some(seq)) = state.sequences.get_mut(seq_idx) {
                seq.validate_decode_state(seq_idx, seq_offset)?;
                let ((live_ptr, n_slices, write_slice), kind) =
                    seq.sync_decode_gpu_chunks(n_kv_head, head_dim, seq_offset, arena_info)?;
                let ptr = if want_snapshot {
                    let ptr = seq.snapshot_gpu_chunks_into(generation, seq_offset)?;
                    snap_cache.insert((seq_idx, seq_offset), (ptr, n_slices, write_slice));
                    ptr
                } else {
                    live_ptr
                };
                ((ptr, n_slices, write_slice), kind)
            } else {
                ((0, 0, 0), DecodeGpuChunksSyncKind::Empty)
            };
            let elapsed = t_sync.elapsed();
            match sync_kind {
                DecodeGpuChunksSyncKind::Rebuild => {
                    stats.rebuilds += 1;
                    stats.rebuild_time += elapsed;
                }
                DecodeGpuChunksSyncKind::Reuse => {
                    stats.reuses += 1;
                    stats.reuse_time += elapsed;
                }
                DecodeGpuChunksSyncKind::Resync { entries } => {
                    stats.resyncs += 1;
                    stats.resync_entries += entries as u64;
                    stats.resync_time += elapsed;
                }
                DecodeGpuChunksSyncKind::Extend { entries } => {
                    stats.extends += 1;
                    stats.extend_entries += entries as u64;
                    stats.extend_time += elapsed;
                }
                DecodeGpuChunksSyncKind::Empty => {
                    stats.empty += 1;
                }
            }
            results.push(result);
        }
        #[cfg(feature = "cuda")]
        {
            let t_upload = std::time::Instant::now();
            batch.flush()?;
            stats.upload_time = t_upload.elapsed();
        }
        drop(state);
        Ok((results, stats))
    }

    /// Returns the storage policy for sealed chunks.
    ///
    /// This determines the target format for sealed chunks after reconcile().
    pub fn storage_policy(&self) -> StoragePolicy {
        let k_format = self.inner.storage.k_format();
        let v_format = self.inner.storage.v_format();
        let location = self.inner.storage.default_location();
        match (location, k_format, v_format) {
            (ArenaLocation::Gpu, KvFormat::Float(dt), KvFormat::Float(_)) => {
                StoragePolicy::GpuFloat(dt)
            }
            (ArenaLocation::Gpu, KvFormat::Quantized(kf), KvFormat::Quantized(_)) => {
                StoragePolicy::GpuQuant(kf)
            }
            (ArenaLocation::Cpu, KvFormat::Float(dt), KvFormat::Float(_)) => {
                StoragePolicy::CpuFloat(dt)
            }
            (ArenaLocation::Cpu, KvFormat::Quantized(kf), KvFormat::Quantized(_)) => {
                StoragePolicy::CpuQuant(kf)
            }
            // Mixed float/quant — treat as quant with the quant format
            (ArenaLocation::Gpu, KvFormat::Quantized(kf), KvFormat::Float(_)) => {
                StoragePolicy::GpuQuant(kf)
            }
            (ArenaLocation::Gpu, KvFormat::Float(_), KvFormat::Quantized(vf)) => {
                StoragePolicy::GpuQuant(vf)
            }
            (ArenaLocation::Cpu, KvFormat::Quantized(kf), KvFormat::Float(_)) => {
                StoragePolicy::CpuQuant(kf)
            }
            (ArenaLocation::Cpu, KvFormat::Float(_), KvFormat::Quantized(vf)) => {
                StoragePolicy::CpuQuant(vf)
            }
        }
    }
}
