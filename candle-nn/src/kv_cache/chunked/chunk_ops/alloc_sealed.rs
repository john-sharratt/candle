//! Allocates the chunks for sealed blocks and registers them on the block table,
//! one block at a time or a whole layer at once.
//!
//! A block's chunks are allocated in the arenas of their formats. A chunk's
//! format is the majority vote of its blocks, so a chunk stays in a single arena
//! format. Bulk allocation does a layer's blocks under one lock and publishes the
//! GPU metadata once, because each lock acquisition on the cold-load path costs
//! something measurable. The arena addresses are resolved after every allocation,
//! since a new allocation can register an arena whose addresses would otherwise
//! be stale.

use super::*;

/// The arena keys a run of per-band formats allocates from.
///
/// The one place formats are turned into keys, so the "can these bands share a
/// contiguous run" test is asked of the keys everywhere rather than of the
/// formats in one place and the keys in another.
pub(super) fn band_keys(
    fmts: &[KvFormat],
    backing: &ChunkedKvBacking,
    location: ArenaLocation,
) -> Result<Vec<ArenaKey>> {
    fmts.iter()
        .map(|f| backing.inner.arena_key_for(*f, location))
        .collect()
}

/// Per-block parameter bundle for [`ChunkedKvBacking::alloc_sealed_blocks_bulk`].
///
/// One spec carries everything that one call to
/// [`ChunkedKvBacking::alloc_sealed_block`] +
/// [`ChunkedKvBacking::set_block_window`] would need. The cold-load
/// path constructs a `Vec<BlockAllocSpec>` for an entire turn's worth
/// of blocks (per layer) and submits them in one bulk call, collapsing
/// the per-block `state.write()` lock churn (3 acquisitions/block ×
/// thousands of blocks) into 1 acquisition/layer.
#[derive(Clone)]
pub struct BlockAllocSpec {
    pub block_idx: usize,
    pub k_formats: Vec<KvFormat>,
    pub v_formats: Vec<KvFormat>,
    pub k_pal: std::sync::Arc<Vec<u8>>,
    pub v_pal: std::sync::Arc<Vec<u8>>,
    pub k_scale: std::sync::Arc<Vec<f32>>,
    pub v_scale: std::sync::Arc<Vec<f32>>,
    pub offset: u16,
    pub usage: u32,
}

impl ChunkedKvBacking {
    /// Aggregate per-block format tags into a single per-chunk format.
    ///
    /// Uses majority vote: the format selected by the most blocks wins.
    /// F16 sentinel (99) votes are counted and if they form the majority:
    /// - With `has_float_fallback=true`: returns `None` (chunk stays float)
    /// - With `has_float_fallback=false`: returns the first (highest fidelity) candidate
    ///
    /// Ties are broken towards higher fidelity (lower candidate index).
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    pub(super) fn aggregate_chunk_format(
        tags: &[i32],
        candidates: &[crate::kv_cache::QuantFormat],
        has_float_fallback: bool,
    ) -> Result<Option<crate::kv_cache::QuantFormat>> {
        use candle::quantized::cuda::select_qtype_to_ggml;

        if candidates.is_empty() {
            candle::bail!("aggregate_chunk_format: empty candidate list");
        }

        // Count votes per candidate index. Index `candidates.len()` = F16 sentinel.
        let mut votes = vec![0u32; candidates.len() + 1];
        let f16_idx = candidates.len(); // sentinel index

        for &tag in tags {
            if tag == SELECT_FMT_F16 {
                votes[f16_idx] += 1;
                continue;
            }
            let block_ggml = select_qtype_to_ggml(tag)?;
            if let Some(idx) = candidates
                .iter()
                .position(|c| c.to_ggml_dtype() == block_ggml)
            {
                votes[idx] += 1;
            }
            // Unknown tags are ignored
        }

        // Find the candidate with the most votes.
        // On tie, prefer the candidate with the HIGHER index (more aggressive).
        let mut best_idx = 0usize;
        let mut best_count = votes[0];
        for (idx, &count) in votes.iter().enumerate().skip(1) {
            if count > best_count {
                best_count = count;
                best_idx = idx;
            }
        }

        if best_idx == f16_idx {
            if has_float_fallback {
                // Float fallback: chunk stays in its float arena
                Ok(None)
            } else {
                // No float fallback: use highest fidelity quantized candidate
                Ok(Some(candidates[0]))
            }
        } else {
            Ok(Some(candidates[best_idx]))
        }
    }

    /// Write one prepacked chunk directly into the backing's arena storage.
    ///
    /// The provided bytes must already match the backing's configured per-side layout.
    /// This is used for zero-copy restore or benchmark staging where K is already packed
    /// as R16 and V is already packed as F16.
    ///
    /// After writing all chunks for a sequence, the caller must call
    /// `advance_sequence` with the total actual token count.
    /// Restore a sealed chunk from raw byte streams.
    ///
    /// The K/V byte streams contain `n_kv_head × N_PALETTE` palette sub-chunks
    /// concatenated. The pal/scale arguments describe how the encoded bytes
    /// should be interpreted on decode:
    ///
    /// - `k_pal` / `v_pal`: per-head packed palette maps (`n_kv_head × head_dim/4`
    ///   bytes each). Pass an empty `Arc<Vec<u8>>` for identity routing.
    /// - `k_scale` / `v_scale`: per-head outer scales (`n_kv_head × N_PALETTE`
    ///   f32 each). Pass an empty `Arc<Vec<f32>>` for unity (no outer scaling).
    ///
    /// Pass empty Arcs when restoring data that was encoded with neither
    /// palette routing nor outer scaling (e.g. R16 / float captures).
    #[allow(clippy::too_many_arguments)]
    pub fn write_raw_sealed_chunk(
        &self,
        batch_idx: usize,
        block_idx: usize,
        k_bytes: &[u8],
        v_bytes: &[u8],
        k_pal: std::sync::Arc<Vec<u8>>,
        v_pal: std::sync::Arc<Vec<u8>>,
        k_scale: std::sync::Arc<Vec<f32>>,
        v_scale: std::sync::Arc<Vec<f32>>,
    ) -> Result<()> {
        let n_kv_head = self.inner.n_kv_head;
        let chunk_size = CHUNK_SIZE;
        let head_dim = self.inner.head_dim;

        let k_format = self.inner.storage.k_format();
        let v_format = self.inner.storage.v_format();

        // Palette4: each chunk = chunk_size * sub_head_dim (one palette sub-band)
        let sub_head_dim = head_dim / N_PALETTE;
        let elems_per_head = chunk_size * sub_head_dim;

        // k_bytes/v_bytes contain ALL n_kv_head * N_PALETTE palette sub-chunks
        // concatenated: [h0p0, h0p1, h0p2, h0p3, h1p0, ...].
        // Split per-(head, palette) and write each to its own flat chunk.
        let k_head_bytes = match k_format {
            KvFormat::Quantized(qf) => {
                let ggml = qf.to_ggml_dtype();
                let blocks = elems_per_head / ggml.block_size();
                blocks * ggml.type_size()
            }
            KvFormat::Float(dtype) => elems_per_head * dtype.size_in_bytes(),
        };
        let v_head_bytes = match v_format {
            KvFormat::Quantized(qf) => {
                let ggml = qf.to_ggml_dtype();
                let v_head_blocks = elems_per_head / ggml.block_size();
                v_head_blocks * ggml.type_size()
            }
            KvFormat::Float(dtype) => elems_per_head * dtype.size_in_bytes(),
        };
        let expected_k_bytes = n_kv_head * N_PALETTE * k_head_bytes;
        let expected_v_bytes = n_kv_head * N_PALETTE * v_head_bytes;
        if k_bytes.len() != expected_k_bytes {
            candle::bail!(
                "write_raw_sealed_chunk: K byte length mismatch, got {}, expected {}",
                k_bytes.len(),
                expected_k_bytes
            );
        }
        if v_bytes.len() != expected_v_bytes {
            candle::bail!(
                "write_raw_sealed_chunk: V byte length mismatch, got {}, expected {}",
                v_bytes.len(),
                expected_v_bytes
            );
        }

        // Allocate the block's GIDs through the single allocation keystone —
        // it also registers them on the block table with the palettes/scales.
        // This restore path is uniform-format, so every sub-band uses the
        // same K/V format.
        let n_sub = n_kv_head * N_PALETTE;
        let gids = self.alloc_sealed_block(
            batch_idx,
            block_idx,
            &vec![k_format; n_sub],
            &vec![v_format; n_sub],
            k_pal,
            v_pal,
            k_scale,
            v_scale,
        )?;

        // Upload each `(head, palette, side)` byte slice into its own slot.
        //
        // One byte write per slot, whatever the format. The bytes arriving here
        // are already the format's own image — that is what "raw" means in this
        // function's name — and a slot is a run of bytes, so there is nothing to
        // decode, transpose, or re-quantize on the way in. The former per-dtype
        // ladder (R16 memcpy / ggml re-wrap / one arm per float width) existed
        // only because the destination was a typed tensor.
        let upload =
            |s: &mut ArenaStorageState, gid: &ChunkGid, slice: &[u8], side: &str| -> Result<()> {
                let ai = gid.arena_idx();
                let arena = s
                    .arenas_mut()
                    .get_mut(&ai)
                    .ok_or_else(|| candle::Error::Msg(format!("{side} arena {ai} not found")))?;
                arena.write_slot_from_bytes(gid.chunk_idx(), slice)
            };

        for h in 0..n_kv_head {
            for p in 0..N_PALETTE {
                let slot_idx = h * N_PALETTE + p;
                let k_start = slot_idx * k_head_bytes;
                let k_slice = &k_bytes[k_start..k_start + k_head_bytes];
                let v_start = slot_idx * v_head_bytes;
                let v_slice = &v_bytes[v_start..v_start + v_head_bytes];

                let gid_offset = h * GIDS_PER_HEAD + p * 2;
                let k_gid = &gids.0[gid_offset];
                let v_gid = &gids.0[gid_offset + 1];

                self.inner.storage.try_write(|s| {
                    upload(s, k_gid, k_slice, "k")?;
                    upload(s, v_gid, v_slice, "v")
                })?;
            }
        }

        Ok(())
    }

    /// Allocate one sealed block's GIDs — the **allocation keystone** (§16.12).
    ///
    /// This is the allocation half of [`Self::write_raw_sealed_chunk`], with
    /// no byte I/O: it materialises the destination slot, allocates the
    /// per-`(head, palette)` chunk GIDs in arenas of `k_format` / `v_format`,
    /// and registers them on the block table with the chunk's palettes and
    /// scales. It is the **single source of truth** for the GID shape a chunk
    /// of a given `KvFormat` needs.
    ///
    /// The returned [`HeadGids`] addresses freshly-allocated, uninitialised
    /// chunks. Fill them either by uploading raw bytes
    /// (`write_raw_sealed_chunk`) or by snapshotting the slot with
    /// [`Self::record_turn`] and scattering bytes through the
    /// `resolve_sealed_chunk_ptrs` migration path (resume / cold-load).
    #[allow(clippy::too_many_arguments)]
    pub fn alloc_sealed_block(
        &self,
        batch_idx: usize,
        block_idx: usize,
        k_formats: &[KvFormat],
        v_formats: &[KvFormat],
        k_pal: std::sync::Arc<Vec<u8>>,
        v_pal: std::sync::Arc<Vec<u8>>,
        k_scale: std::sync::Arc<Vec<f32>>,
        v_scale: std::sync::Arc<Vec<f32>>,
    ) -> Result<HeadGids> {
        let n_kv_head = self.inner.n_kv_head;
        let want = n_kv_head * N_PALETTE;
        if k_formats.len() != want || v_formats.len() != want {
            candle::bail!(
                "alloc_sealed_block: expected {want} per-sub-band formats, got k={} v={}",
                k_formats.len(),
                v_formats.len()
            );
        }
        let location = self.inner.storage.default_location();

        // Ensure the block table has room for this block index and materialize
        // the destination sequence slot before attaching the fresh GIDs.
        self.ensure_max_blocks(block_idx + 1)?;
        self.ensure_for_offset(batch_idx, block_idx * CHUNK_SIZE, CHUNK_SIZE)?;

        // Allocate GIDS_PER_HEAD * n_kv_head destination chunks, each
        // `(head, palette sub-band)` in its own adaptive `KvFormat`.
        let mut gid_vec: Vec<ChunkGid> = Vec::with_capacity(GIDS_PER_HEAD * n_kv_head);
        {
            let _state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            // Per (head, side): a uniform band group allocates one CONTIGUOUS
            // run for select/QREL walk locality (see `alloc_chunk_run_for_key`;
            // correctness is per-band via `resolve_band_source`, not the run
            // layout); mixed groups allocate per band.
            for h in 0..n_kv_head {
                let base = h * N_PALETTE;
                let alloc_side = |fmts: &[KvFormat]| -> Result<Vec<ChunkGid>> {
                    // Eligibility is a question about *keys*, not formats: four
                    // bands in different formats that share a size class share
                    // an arena, so they can still be one contiguous run. Under
                    // per-format arenas this test could only pass for identical
                    // formats; classes are coarser, so it now fires strictly
                    // more often.
                    let keys = band_keys(&fmts[base..base + N_PALETTE], self, location)?;
                    if keys.iter().all(|k| *k == keys[0]) {
                        self.alloc_chunk_run_for_key(keys[0], N_PALETTE)
                    } else {
                        keys.into_iter()
                            .map(|k| self.alloc_chunk_for_key(k))
                            .collect()
                    }
                };
                let k_run = alloc_side(k_formats)?;
                let v_run = alloc_side(v_formats)?;
                for (k_gid, v_gid) in k_run.into_iter().zip(v_run) {
                    gid_vec.push(k_gid);
                    gid_vec.push(v_gid);
                }
            }
        }

        // Register per-head GIDs, palette maps, outer scales and the band
        // format tags on the block table, then refresh the live decode slot.
        // The tags are the *destination* formats — this window is being
        // re-pointed at sealed chunks whose layout is whatever was persisted,
        // not the active R16/F16 a fresh writer window carries.
        let gids = HeadGids::from_vec(gid_vec);
        let arena_info = self.resolve_arena_info()?;
        self.set_block_gids_sharded_and_update_gpu(
            batch_idx,
            block_idx,
            gids.clone(),
            k_pal,
            v_pal,
            k_scale,
            v_scale,
            std::sync::Arc::new(k_formats.iter().map(|f| f.to_tag()).collect()),
            std::sync::Arc::new(v_formats.iter().map(|f| f.to_tag()).collect()),
            &arena_info,
        )?;
        Ok(gids)
    }

    /// Bulk variant of [`Self::alloc_sealed_block`] + [`Self::set_block_window`]
    /// that does all the per-block work for one layer under a **single**
    /// per-layer `state.write()` critical section.
    ///
    /// Per the legacy path, each block paid:
    ///   - 1 `state.write()` for the `alloc_chunk_for_key` loop,
    ///   - 1 `state.write()` for `set_block_gids_sharded_and_update_gpu`,
    ///   - 1 `state.write()` for `set_block_window`,
    ///   - 1 `resolve_arena_info` call (CPU walk over arenas).
    ///
    /// For a Qwen3-30B-A3B turn (~1500 blocks × 48 layers) that was
    /// ~216 000 lock acquisitions and ~72 000 arena-info walks. The bulk
    /// call collapses all of them to 1 acquisition + 1 arena-info walk
    /// per layer (48 each per turn).
    ///
    /// All blocks must satisfy the same per-sub-band shape constraint as
    /// `alloc_sealed_block` (`k_formats.len() == v_formats.len() ==
    /// n_kv_head × N_PALETTE`). Block indices need not be contiguous
    /// but must each be `< max_blocks` after the upfront grow.
    ///
    /// On error, the lock guard drops and any partial allocations remain
    /// in place — the caller is expected to `free_sequence` the slot to
    /// recycle them.
    /// Pre-allocate the slot's block table + sequence chunks for a
    /// cold-load. Caller pairs this with [`Self::alloc_sealed_blocks_bulk`]:
    /// invoke once per layer at the start of the cold-load (with the
    /// turn's total `chunks_per_layer`), then the per-unit bulk allocs
    /// inherit the capacity and skip their own `ensure_*` work
    /// (saving 2 × `state.write()` acquires per bulk call).
    pub fn ensure_capacity_for_blocks(&self, batch_idx: usize, n_blocks: usize) -> Result<()> {
        if n_blocks == 0 {
            return Ok(());
        }
        self.ensure_max_blocks(n_blocks)?;
        self.ensure_for_offset(batch_idx, 0, n_blocks * CHUNK_SIZE)?;
        Ok(())
    }

    /// Returns the alloc'd `HeadGids` alongside the **freshly-resolved**
    /// `arena_info` snapshot used for the GPU metadata push. The caller
    /// can hand both directly to
    /// [`Self::resolve_block_ptrs_from_hgids`] on the same backing —
    /// no `state.read()` lock, no per-block dedup. The arena_info is
    /// valid as long as no arena-mutating call runs in between (true
    /// for the cold-load pipeline's per-unit, per-layer pair).
    pub fn alloc_sealed_blocks_bulk(
        &self,
        batch_idx: usize,
        specs: &[BlockAllocSpec],
        pool_us_out: &mut u64,
        register_us_out: &mut u64,
        gpu_push_us_out: &mut u64,
    ) -> Result<(
        Vec<HeadGids>,
        Vec<crate::kv_cache::arena_table::ResolvedArenaInfo>,
    )> {
        if specs.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;
        let want = n_kv_head * N_PALETTE;
        for spec in specs {
            if spec.k_formats.len() != want || spec.v_formats.len() != want {
                candle::bail!(
                    "alloc_sealed_blocks_bulk: expected {want} per-sub-band formats, \
                     got k={} v={} (block {})",
                    spec.k_formats.len(),
                    spec.v_formats.len(),
                    spec.block_idx
                );
            }
        }
        let location = self.inner.storage.default_location();

        // **Caller contract**: the slot must already have at least
        // `max(spec.block_idx) + 1` chunks allocated before entering
        // this function. The cold-load pipeline ensures this once up
        // front via [`Self::ensure_capacity_for_blocks`], which avoids
        // the per-call `ensure_max_blocks` + `ensure_for_offset` lock
        // churn (each acquires `state.write()` — 2 acquires/call ×
        // 82 calls/turn = ~10 ms wall-clock on a 1824-chunk turn).

        // `resolve_arena_info` must be called **after** the per-spec
        // gid allocs below (which can register fresh arena indices via
        // `pool.register_arena` when a format's pool is exhausted). We
        // hold off on resolving until just before the GPU metadata
        // push so it reflects every arena that the alloc grew. Doing
        // it earlier would leave the `update_gpu_chunks_bulk` call
        // with stale `base_ptr` / `chunk_byte_stride` for the freshly-
        // registered arenas — silent corruption with `R16`-formatted
        // chunks, or "arena index N out of range" downstream.

        // Single critical section covering: pool allocs, block_gid
        // registration, window stamping, then a **single batched**
        // GPU chunk-metadata push at the end (one guard for the
        // whole layer's blocks, so the drop coalesces adjacent
        // indices into a few `memcpy_htod` runs instead of one per
        // block).
        // ── Phase 1: batched GID alloc + scatter into per-spec slots ─
        //
        // For each (spec, sub-band, K|V) gid, record its **destination**
        // (s_idx, gid_idx) under its `ArenaKey`. Then allocate per-key
        // in one batched call and scatter the freshly-claimed gids
        // straight into per-spec `MaybeUninit<ChunkGid>` buffers.
        //
        // Phase 2 below just hands each filled buffer to
        // `HeadGids::from_vec` — **no inner stitch loop, no HashMap
        // lookups in the locked section**. The cold-load shape (~600
        // gids/spec × ~22 specs/call) used to spend ~1,400 HashMap
        // lookups + IntoIter::next per call inside `state.write()`.
        //
        // Layout per spec: K(h, p) → gid_idx = sub*2, V(h, p) → sub*2+1,
        // where sub = h*N_PALETTE + p (matches the historical inner
        // push order).
        use std::mem::MaybeUninit;
        let t_pool = std::time::Instant::now();
        let gids_per_spec = GIDS_PER_HEAD * n_kv_head;
        // Uniform (head, side) band groups allocate as CONTIGUOUS runs for
        // select/QREL walk locality (correctness is per-band via
        // `resolve_band_source` — see `alloc_chunk_run_for_key`); only
        // mixed-format groups fall through to the batched singleton path below.
        let mut run_groups: Vec<(usize, usize, bool, ArenaKey)> = Vec::new();
        let mut positions_per_key: ahash::HashMap<ArenaKey, Vec<(usize, usize)>> =
            ahash::HashMap::with_capacity_and_hasher(8, ahash::RandomState::new());
        for (s_idx, spec) in specs.iter().enumerate() {
            for h in 0..n_kv_head {
                let base = h * N_PALETTE;
                // Same key-not-format eligibility test as `alloc_sealed_block`:
                // bands that share a size class share an arena and can form one
                // contiguous run whatever their formats.
                for (is_v, fmts) in [(false, &spec.k_formats), (true, &spec.v_formats)] {
                    let keys = band_keys(&fmts[base..base + N_PALETTE], self, location)?;
                    if keys.iter().all(|k| *k == keys[0]) {
                        run_groups.push((s_idx, base, is_v, keys[0]));
                    } else {
                        for (p, k) in keys.into_iter().enumerate() {
                            positions_per_key
                                .entry(k)
                                .or_default()
                                .push((s_idx, (base + p) * 2 + usize::from(is_v)));
                        }
                    }
                }
            }
        }
        // Per-spec uninitialised buffers — `set_len` is free (no
        // per-element init), and every slot is written by the
        // scatter below.
        let mut spec_buffers: Vec<Vec<MaybeUninit<super::super::gid_pool::ChunkGid>>> = (0..specs
            .len())
            .map(|_| {
                let mut v: Vec<MaybeUninit<super::super::gid_pool::ChunkGid>> =
                    Vec::with_capacity(gids_per_spec);
                // SAFETY: every slot is written by the per-key scatter
                // below before any read. The Vec is consumed into an
                // initialised `Vec<ChunkGid>` once the scatter is
                // complete.
                unsafe { v.set_len(gids_per_spec) };
                v
            })
            .collect();
        for (key, positions) in positions_per_key {
            let gids = self.alloc_chunks_for_key_bulk(key, positions.len())?;
            for (gid, (s_idx, gid_idx)) in gids.into_iter().zip(positions) {
                spec_buffers[s_idx][gid_idx].write(gid);
            }
        }
        for (s_idx, base, is_v, key) in run_groups {
            let run = self.alloc_chunk_run_for_key(key, N_PALETTE)?;
            for (p, gid) in run.into_iter().enumerate() {
                let gid_idx = (base + p) * 2 + usize::from(is_v);
                spec_buffers[s_idx][gid_idx].write(gid);
            }
        }
        // Phase 1 wrote every position in every spec buffer (the
        // position lists were built to cover exactly one slot per
        // (spec, sub, K|V) tuple). Reinterpret each buffer as a
        // fully-initialised `Vec<ChunkGid>`.
        let per_spec_gids: Vec<Vec<super::super::gid_pool::ChunkGid>> = spec_buffers
            .into_iter()
            .map(|buf| {
                let mut buf = std::mem::ManuallyDrop::new(buf);
                let ptr = buf.as_mut_ptr() as *mut super::super::gid_pool::ChunkGid;
                let len = buf.len();
                let cap = buf.capacity();
                // SAFETY: every position in `buf` was written in
                // Phase 1; the layout of `MaybeUninit<T>` matches `T`.
                unsafe { Vec::from_raw_parts(ptr, len, cap) }
            })
            .collect();
        *pool_us_out += t_pool.elapsed().as_micros() as u64;

        // ── Phase 2: register blocks under the state write lock ──────
        let mut hgids_per_block: Vec<HeadGids> = Vec::with_capacity(specs.len());
        let mut updated_blocks: Vec<usize> = Vec::with_capacity(specs.len());
        // Whether any loaded block's window differs from the one it replaces.
        let mut layout_changed = false;
        // Declared outside the `state.write()` scope so we can return
        // it to the caller (immediately reused by
        // `resolve_block_ptrs_from_hgids` to skip its own arena walk).
        let arena_info: Vec<crate::kv_cache::arena_table::ResolvedArenaInfo>;
        // Per-block snapshots (blk + frozen fields) for the resident-record build,
        // captured under the state lock but built/uploaded AFTER it is released so
        // the device upload never serializes the per-forward readers.
        // `(block, gids, k_pal, v_pal, k_scale, v_scale, k_fmt, v_fmt)` — the
        // owning form of `ChunkRecordSrc`, since the borrow cannot outlive the
        // state lock the snapshot is taken under.
        type RecordSrc = (
            usize,
            HeadGids,
            std::sync::Arc<Vec<u8>>,
            std::sync::Arc<Vec<u8>>,
            std::sync::Arc<Vec<f32>>,
            std::sync::Arc<Vec<f32>>,
            std::sync::Arc<Vec<u8>>,
            std::sync::Arc<Vec<u8>>,
        );
        let mut record_srcs: Vec<RecordSrc> = Vec::new();
        let mut per_spec_iter = per_spec_gids.into_iter();
        {
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

            let t_register = std::time::Instant::now();
            for spec in specs {
                let gid_vec = per_spec_iter
                    .next()
                    .expect("per_spec_gids has one entry per spec");
                let gids = HeadGids::from_vec(gid_vec);

                // Register on the block table + stamp the window.
                // GPU chunk-metadata push is deferred until all
                // blocks have been registered (see batched call
                // below).
                if let Some(slot) = state.sequences.get_mut(batch_idx).and_then(|s| s.as_mut()) {
                    slot.set_block_gids(
                        spec.block_idx,
                        gids.clone(),
                        std::sync::Arc::clone(&spec.k_pal),
                        std::sync::Arc::clone(&spec.v_pal),
                        std::sync::Arc::clone(&spec.k_scale),
                        std::sync::Arc::clone(&spec.v_scale),
                        // The spec's per-band formats ARE the persisted tag
                        // bytes (`KvFormat::to_tag` is the `ArenaFormatTag`
                        // discriminant), so this closes the loop from disk back
                        // to the window that describes the loaded chunks.
                        std::sync::Arc::new(spec.k_formats.iter().map(|f| f.to_tag()).collect()),
                        std::sync::Arc::new(spec.v_formats.iter().map(|f| f.to_tag()).collect()),
                    );
                    if let Some(cw) = slot.chunk_at_mut(spec.block_idx) {
                        layout_changed |= cw.offset != spec.offset || cw.usage != spec.usage;
                        cw.offset = spec.offset;
                        cw.usage = spec.usage;
                    }
                    updated_blocks.push(spec.block_idx);
                }
                hgids_per_block.push(gids);
            }
            *register_us_out += t_register.elapsed().as_micros() as u64;

            // ── Phase 3: resolve arena_info + GPU metadata push ─────
            //
            // Resolve arena_info NOW — after every spec's allocs have
            // potentially registered new arenas — so the GPU metadata
            // push reflects every arena the chunks point into. (See
            // the comment block before the critical section.)
            let t_gpu_push = std::time::Instant::now();
            arena_info = self.resolve_arena_info()?;
            if let Some(slot) = state.sequences.get_mut(batch_idx).and_then(|s| s.as_mut()) {
                // A block whose window moved moves the rope of every block
                // after it, which re-serialising the loaded blocks alone would
                // leave as it was. Drop the buffer then: the next decode sync
                // rebuilds every rope from the host. Unchanged windows only
                // swapped their gids, and the loaded blocks are all that moved.
                if layout_changed {
                    slot.invalidate_gpu_chunks();
                } else {
                    slot.update_gpu_chunks_bulk(&updated_blocks, n_kv_head, head_dim, &arena_info)?;
                }
            }
            *gpu_push_us_out += t_gpu_push.elapsed().as_micros() as u64;

            // Snapshot the frozen fields of each loaded chunk while we hold the
            // lock; the actual record serialize + device upload happens below,
            // off the lock.
            if let Some(slot) = state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                for &blk in &updated_blocks {
                    if let Some(cw) = slot.chunk_at(blk) {
                        record_srcs.push((
                            blk,
                            cw.gids.clone(),
                            cw.k_pal.clone(),
                            cw.v_pal.clone(),
                            cw.k_scale.clone(),
                            cw.v_scale.clone(),
                            cw.k_fmt.clone(),
                            cw.v_fmt.clone(),
                        ));
                    }
                }
            }
        }

        // Build resident KV-head records at the loaded chunks' GPU placement.
        // Cold-load (and warm→hot elevate that routes through here) is the point
        // at which these sealed chunks become device-resident; building the record
        // once here lets every subsequent reproject glue forward read it via
        // `kvheads_ptr` instead of rebuilding heads per layer. Done OFF the state
        // lock and as a single coalesced upload.
        if !record_srcs.is_empty() {
            let refs: Vec<ChunkRecordSrc<'_>> = record_srcs
                .iter()
                .map(|(_, g, kp, vp, ks, vs, kf, vf)| ChunkRecordSrc {
                    gids: g,
                    k_pal: kp.as_slice(),
                    v_pal: vp.as_slice(),
                    k_scale: ks.as_slice(),
                    v_scale: vs.as_slice(),
                    k_fmt: kf.as_slice(),
                    v_fmt: vf.as_slice(),
                })
                .collect();
            let metas = self.build_meta_records(&refs, &arena_info)?;
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            if let Some(slot) = state.sequences.get_mut(batch_idx).and_then(|s| s.as_mut()) {
                let mut sealed_end = 0;
                for ((blk, ..), meta) in record_srcs.iter().zip(metas) {
                    if let Some(cw) = slot.chunk_at_mut(*blk) {
                        cw.meta = meta;
                        sealed_end = sealed_end.max(blk + 1);
                    }
                }
                // A block that carries a record is sealed history, read through
                // that record — never the slot's to write. The write boundary
                // moves past it, so nothing reads it as a writer chunk.
                if slot.writer_start_idx() < sealed_end {
                    slot.set_writer_start_idx(sealed_end);
                }
            }
        }
        Ok((hgids_per_block, arena_info))
    }
}
