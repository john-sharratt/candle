//! Stages the paged selection inputs on the device, from F32 test chunks or from
//! the live arenas, and runs the fused palette-4 selection over them.
//!
//! The K arena is staged as R16, carrying the raw F16 value and the captured query
//! value per block, because the kernel needs the query to compute relevance. The V
//! arena stays F32, as it is during selection in production. Chunks arrive
//! token-major within each head and palette band, and the R16 layout is
//! dim-major, so the staging transposes each band on the way in. The per-head
//! table and the head-gid table are uploaded as owned buffers, not from the pinned
//! staging arena, since the kernel reads them asynchronously and the arena's bytes
//! can be recycled under an in-flight kernel. A selection over a chunk range
//! narrows both tables to that range, so a chunk's row index is relative to the
//! buffer it is in. A multi-layer selection concatenates every layer's tables into
//! one upload, which gives the same result as selecting each layer separately. A
//! selection is refused when the gids do not match the current arenas, since an
//! error there would surface as an out-of-bounds write on the device, and when the
//! head dimension is one the kernel is not instantiated for.

use super::*;

impl PagedSelectionGpuInputs {
    /// A device block of at least `bytes` for one palette-4 selection.
    pub(super) fn palette_block(&self, bytes: usize) -> Result<u64> {
        let bytes = bytes.max(256);
        let mut held = self.palette_block.lock().unwrap();
        if let Some((at, len)) = *held {
            if len >= bytes {
                return Ok(at);
            }
        }
        if let Some(bump) = self._bump_gen.as_ref() {
            let at = bump.alloc(bytes, 256)?.ptr;
            *held = Some((at, bytes));
            return Ok(at);
        }
        // No span to carve from — CPU-staged test inputs. The block is this
        // struct's to hold and is recorded as a driver allocation.
        let mut owned = self.palette_owned.lock().unwrap();
        // SAFETY: the selection writes its inputs and clears its outputs
        // before the kernel reads either.
        let slice = unsafe { self.dev.alloc::<u8>(bytes)? };
        let at = slice.device_ptr(&self.dev.cuda_stream()).0;
        *owned = Some(slice);
        *held = Some((at, bytes));
        Ok(at)
    }

    /// One fused palette-4 selection over these inputs, read back.
    pub(super) fn run_palette4(
        &self,
        k_ggml: &[GgmlDType],
        v_ggml: &[GgmlDType],
        thresholds: [f32; 4],
        valid_ranges: &[i32],
    ) -> Result<Palette4Result> {
        let blocks_per_head = self.checked_blocks_per_head()?;
        let n_chunks = self.chunk_gids_keepalive.len();
        let layout = Palette4Layout::new(
            n_chunks,
            self.n_kv_head,
            blocks_per_head,
            k_ggml.len(),
            v_ggml.len(),
        );
        let block = self.palette_block(layout.bytes())?;
        // SAFETY: both tables are staged for these inputs and live as long as
        // `self`; `block` holds `layout.bytes()` and nothing else touches it.
        unsafe {
            select_palette4_in(
                self.per_head_table_buf.dev_ptr(),
                self.head_gids_buf.dev_ptr(),
                n_chunks,
                k_ggml,
                v_ggml,
                thresholds,
                blocks_per_head,
                self.n_kv_head,
                self.arena_chunks,
                valid_ranges,
                block,
                &self.dev.cuda_stream(),
            )
        }
    }

    /// Stage `k_chunks` / `v_chunks` as F32 float arenas. Each chunk is
    /// `[H][P][T][D']` flat f32 of length `n_kv_head * head_dim * chunk_size` —
    /// palette bands in order, each token-major — the layout the float arenas
    /// hold in production and the dump readers produce.
    pub fn from_f32_chunks(
        k_chunks: &[&[f32]],
        v_chunks: &[&[f32]],
        blocks_per_chunk: usize,
        n_kv_head: usize,
        _arena_chunks: usize,
        generation: Option<&Generation>,
        dev: &candle::CudaDevice,
    ) -> Result<(ChunkedKvBacking, Self)> {
        if k_chunks.len() != v_chunks.len() {
            candle::bail!("paged selection input length mismatch");
        }
        if blocks_per_chunk == 0 {
            candle::bail!("blocks_per_chunk must be > 0");
        }
        if n_kv_head == 0 || !blocks_per_chunk.is_multiple_of(n_kv_head) {
            candle::bail!(
                "blocks_per_chunk ({blocks_per_chunk}) must be divisible by n_kv_head ({n_kv_head})"
            );
        }
        let head_dim = blocks_per_chunk / n_kv_head;
        if !head_dim.is_multiple_of(N_PALETTE) {
            candle::bail!("head_dim ({head_dim}) must be divisible by N_PALETTE ({N_PALETTE})");
        }
        let chunk_size = SELECT_BLOCK;
        let sub_head_dim = head_dim / N_PALETTE;
        let elems_per_band = sub_head_dim * chunk_size;

        let device = candle::Device::Cuda(dev.clone());
        let n_chunks = k_chunks.len();
        let backing = ChunkedKvBacking::new_with_format(
            n_chunks,
            n_kv_head,
            head_dim,
            KvFormat::Float(candle::DType::F32),
            KvFormat::Float(candle::DType::F32),
            &device,
            chunk_size,
        )?;

        for (slot, (k_chunk, v_chunk)) in k_chunks.iter().zip(v_chunks.iter()).enumerate() {
            // The chunk's bands in `write_raw_sealed_chunk` order ([h0p0, h0p1,
            // h0p2, h0p3, h1p0, ...]), each already token-major.
            let mut k_bytes = Vec::with_capacity(n_kv_head * N_PALETTE * elems_per_band * 4);
            let mut v_bytes = Vec::with_capacity(n_kv_head * N_PALETTE * elems_per_band * 4);
            for h in 0..n_kv_head {
                for p in 0..N_PALETTE {
                    let start = (h * head_dim + p * sub_head_dim) * chunk_size;
                    k_bytes.extend(
                        k_chunk[start..start + elems_per_band]
                            .iter()
                            .flat_map(|&f| f.to_le_bytes()),
                    );
                    v_bytes.extend(
                        v_chunk[start..start + elems_per_band]
                            .iter()
                            .flat_map(|&f| f.to_le_bytes()),
                    );
                }
            }
            backing.write_raw_sealed_chunk(
                slot,
                0,
                &k_bytes,
                &v_bytes,
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
            )?;
        }

        let all_slots: Vec<usize> = (0..n_chunks).collect();
        let inputs = Self::from_backing_owned(&backing, &all_slots, generation, dev)?;
        Ok((backing, inputs))
    }

    /// Build a paged-selection input set with K stored as R16 (raw F16 + captured
    /// Q values), matching what production stores during prefill/decode.
    ///
    /// Use this instead of [`from_f32_chunks`] when the caller has Q-projection
    /// values per chunk (e.g., loaded from a v4 KV dump): the GPU selection
    /// kernel reads `block_r16->q[]` to compute per-block Q-relevance, drives
    /// the IQR threshold scaling, and selects formats per-(head, palette).
    /// Without R16+Q the kernel sees Q=0 everywhere, falls back to the
    /// geometric-mean threshold, and the per-block format distribution does
    /// not match what production produces.
    ///
    /// `k_chunks`, `v_chunks`, and `q_chunks` are `[H][P][T][D']` flat f32
    /// arrays of length `n_kv_head * head_dim * chunk_size` — palette-segmented,
    /// then token-major within each (head, palette) band — the layout of
    /// `dump_sequence_*_chunks`, the v3/v4 dump readers, and the float arenas.
    /// V is kept as F32 (matches production where V is always float during
    /// the selection phase).
    pub fn from_f32_chunks_with_q(
        k_chunks: &[&[f32]],
        v_chunks: &[&[f32]],
        q_chunks: &[&[f32]],
        blocks_per_chunk: usize,
        n_kv_head: usize,
        _arena_chunks: usize,
        generation: Option<&Generation>,
        dev: &candle::CudaDevice,
    ) -> Result<(ChunkedKvBacking, Self)> {
        if k_chunks.len() != v_chunks.len() || k_chunks.len() != q_chunks.len() {
            candle::bail!("paged selection input length mismatch (k/v/q must match)");
        }
        if blocks_per_chunk == 0 {
            candle::bail!("blocks_per_chunk must be > 0");
        }
        if n_kv_head == 0 || !blocks_per_chunk.is_multiple_of(n_kv_head) {
            candle::bail!(
                "blocks_per_chunk ({blocks_per_chunk}) must be divisible by n_kv_head ({n_kv_head})"
            );
        }
        let head_dim = blocks_per_chunk / n_kv_head;
        if !head_dim.is_multiple_of(N_PALETTE) {
            candle::bail!("head_dim ({head_dim}) must be divisible by N_PALETTE ({N_PALETTE})");
        }
        let chunk_size = SELECT_BLOCK;
        let sub_head_dim = head_dim / N_PALETTE;
        let elems_per_band = sub_head_dim * chunk_size;

        let device = candle::Device::Cuda(dev.clone());
        let n_chunks = k_chunks.len();
        // K arena = R16 (raw F16 + Q-capture). V arena stays F32 (matches the
        // production prefill/decode layout where V is always float during
        // selection).
        let backing = ChunkedKvBacking::new_with_format(
            n_chunks,
            n_kv_head,
            head_dim,
            KvFormat::Quantized(QuantFormat::R16),
            KvFormat::Float(candle::DType::F32),
            &device,
            chunk_size,
        )?;

        // R16 byte layout per (head, palette) sub-band:
        //   sub_head_dim blocks × 128 bytes/block
        //   block[d] = { F16 d[CHUNK_SIZE]   // K values for tokens 0..32 at dim d
        //              , F16-as-u16 q[CHUNK_SIZE] }  // Q values for tokens 0..32 at dim d
        //
        // Source layout: `dump_sequence_*_chunks` and the v3/v4 dump readers
        // produce `[H][P][T][D']` flat f32 — palette-segmented, then
        // **token-major** within each (h, p) sub-band (matches the F32 arena
        // shape `(1, CHUNK_SIZE, sub_head_dim)`).  R16's dim-major block
        // layout requires transposing each sub-band on the way in.
        let r16_bytes_per_band = sub_head_dim * 128;
        for (slot, ((k_chunk, v_chunk), q_chunk)) in k_chunks
            .iter()
            .zip(v_chunks.iter())
            .zip(q_chunks.iter())
            .enumerate()
        {
            let mut k_bytes = Vec::with_capacity(n_kv_head * N_PALETTE * r16_bytes_per_band);
            let mut v_bytes = Vec::with_capacity(n_kv_head * N_PALETTE * elems_per_band * 4);
            for h in 0..n_kv_head {
                for p in 0..N_PALETTE {
                    let start = (h * head_dim + p * sub_head_dim) * chunk_size;
                    // Pack 32 R16 blocks (one per dim) for this (h, p) sub-band.
                    // Source is [T][D'] token-major; iterate d outer to gather all
                    // 32 tokens for that dim into block[d].d[] / block[d].q[].
                    for d in 0..sub_head_dim {
                        for t in 0..chunk_size {
                            let src = start + t * sub_head_dim + d;
                            let k_h = half::f16::from_f32(k_chunk[src]);
                            k_bytes.extend_from_slice(&k_h.to_le_bytes());
                        }
                        for t in 0..chunk_size {
                            let src = start + t * sub_head_dim + d;
                            let q_h = half::f16::from_f32(q_chunk[src]);
                            k_bytes.extend_from_slice(&q_h.to_le_bytes());
                        }
                    }
                    // V is F32 — same token-major layout as `from_f32_chunks`.
                    v_bytes.extend(
                        v_chunk[start..start + elems_per_band]
                            .iter()
                            .flat_map(|&f| f.to_le_bytes()),
                    );
                }
            }
            backing.write_raw_sealed_chunk(
                slot,
                0,
                &k_bytes,
                &v_bytes,
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
            )?;
        }

        let all_slots: Vec<usize> = (0..n_chunks).collect();
        let inputs = Self::from_backing_owned(&backing, &all_slots, generation, dev)?;
        Ok((backing, inputs))
    }

    pub(super) fn from_backing_owned(
        backing: &ChunkedKvBacking,
        batch_indices: &[usize],
        generation: Option<&Generation>,
        dev: &candle::CudaDevice,
    ) -> Result<Self> {
        let mut chunk_gids_keepalive = Vec::with_capacity(batch_indices.len());
        for &batch_idx in batch_indices {
            let sealed = backing.live_chunks_as_sealed(batch_idx).ok_or_else(|| {
                candle::Error::Msg(format!("missing live chunks for batch slot {batch_idx}"))
            })?;
            let chunk = sealed.first().ok_or_else(|| {
                candle::Error::Msg(format!(
                    "no paged chunk recorded for batch slot {batch_idx}"
                ))
            })?;
            chunk_gids_keepalive.push(ChunkBands::from_sealed(chunk));
        }

        Self::from_head_gids(
            std::sync::Arc::clone(&backing.inner),
            &chunk_gids_keepalive,
            generation,
            dev,
        )
    }

    pub fn from_backing(
        backing: &ChunkedKvBacking,
        batch_indices: &[usize],
        generation: Option<&Generation>,
        dev: &candle::CudaDevice,
    ) -> Result<Self> {
        Self::from_backing_owned(backing, batch_indices, generation, dev)
    }

    pub fn from_head_gids(
        backing: Arc<BackingInner>,
        chunk_gids_keepalive: &[ChunkBands],
        generation: Option<&Generation>,
        dev: &candle::CudaDevice,
    ) -> Result<Self> {
        // Refuse to launch the selection kernel over gids that don't match the
        // current storage arenas — a named host error beats a silent GPU OOB.
        backing.validate_selection_gids(chunk_gids_keepalive, "single-layer")?;
        let per_head_table = backing.per_head_table_host(chunk_gids_keepalive)?;
        // The guard opens *before* the uploads it bounds. Creating it in the
        // struct literal below left the bumps briefly unguarded, so another
        // thread's last-generation drop could reset the cursor between the
        // upload and the guard.
        let bump_gen = bump_generation(None, dev);
        let per_head_table_buf = stage_table(&per_head_table, bump_gen.as_ref(), dev)?;
        let n_kv_head = backing.n_kv_head;

        let blocks_per_chunk = n_kv_head * backing.head_dim;
        let chunk_gids_keepalive = chunk_gids_keepalive.to_vec();
        // Stage every palette band's gid (HeadGids natural order:
        // head * GIDS_PER_HEAD + palette * 2 + is_v) — the selection kernels
        // address each band through its own gid, never by extrapolating from
        // band 0.
        let head_gids: Vec<i64> = chunk_gids_keepalive
            .iter()
            .flat_map(|c| {
                let gids = &c.gids;
                (0..n_kv_head).flat_map(move |head_idx| {
                    (0..N_PALETTE).flat_map(move |p| {
                        [
                            gids.k_gid_pal(head_idx, p).raw(),
                            gids.v_gid_pal(head_idx, p).raw(),
                        ]
                    })
                })
            })
            .collect();
        // OWNED upload, never the pinned-stager arena: an arena-backed GpuBuf's
        // bytes are valid only while the staging generation chain stays alive,
        // and the selection kernel consumes this table asynchronously — a
        // last-generation drop on ANOTHER thread syncs one stream and resets
        // the arena, recycling these bytes under the in-flight kernel (the
        // clobbered-gids OOB). The table is tiny; the owned CudaSlice is held
        // by the GpuBuf and freed stream-ordered after the kernel — safe.
        let _ = generation;
        let head_gids_buf = stage_i64_slice(&head_gids, None, bump_gen.as_ref(), dev)?;

        Ok(Self {
            chunk_gids_keepalive,
            per_head_table,
            per_head_table_buf,
            head_gids,
            head_gids_buf,
            blocks_per_chunk,
            n_kv_head,
            arena_chunks: GID_STRIDE,
            dev: dev.clone(),
            // Staged through `bump_gen` above, so the buffers borrow the span
            // it holds open.
            _bump_gen: bump_gen,
            palette_block: Mutex::new(None),
            palette_owned: Mutex::new(None),
        })
    }

    /// Cross-layer selection inputs: concatenate every layer's per-head arena
    /// table and gids into ONE upload, so a single fused selection kernel + a
    /// single set of readbacks replaces the per-layer 48× launch/readback loop
    /// that dominates the hot→warm drain. Each layer's `arena_idx` is offset by
    /// the running arena count, keeping the unified table densely indexable as
    /// `table[arena_idx * n_kv_head + head]`. Returns the unified inputs plus the
    /// per-layer chunk counts, so the caller can split the flat results back out
    /// per layer.
    ///
    /// Bit-identical to running [`Self::from_head_gids`] per layer and selecting
    /// each separately: the offset is a pure relabelling of arena rows the kernel
    /// reads by pointer, and chunks never interact across the selection.
    pub fn from_head_gids_multi(
        backings: &[Arc<BackingInner>],
        chunk_gids_per_layer: &[Vec<ChunkBands>],
        generation: Option<&Generation>,
        dev: &candle::CudaDevice,
    ) -> Result<(Self, Vec<usize>)> {
        if backings.len() != chunk_gids_per_layer.len() {
            candle::bail!("from_head_gids_multi: backings/gids layer count mismatch");
        }
        if backings.is_empty() {
            candle::bail!("from_head_gids_multi: no layers");
        }
        let n_kv_head = backings[0].n_kv_head;
        let head_dim = backings[0].head_dim;

        let mut unified_table: Vec<i64> = Vec::new();
        let mut unified_head_gids: Vec<i64> = Vec::new();
        let mut chunk_counts: Vec<usize> = Vec::with_capacity(backings.len());
        let mut chunk_gids_keepalive: Vec<ChunkBands> = Vec::new();

        for (layer_idx, (backing, gids_layer)) in
            backings.iter().zip(chunk_gids_per_layer).enumerate()
        {
            if backing.n_kv_head != n_kv_head || backing.head_dim != head_dim {
                candle::bail!("from_head_gids_multi: heterogeneous layer geometry");
            }
            // Refuse to launch the fused selection over gids that don't match
            // this layer's current storage arenas (see validate_selection_gids).
            backing.validate_selection_gids(gids_layer, &format!("multi layer {layer_idx}"))?;
            // Rows are keyed by (chunk, head) and chunks concatenate, so each
            // layer's rows land at the right global index by appending — no
            // arena rebasing, and no `gid_off` added to the gids. That rebasing
            // existed only because the row key came *from* the gid; it does
            // not any more.
            unified_table.extend_from_slice(&backing.per_head_table_host(gids_layer)?);
            for c in gids_layer.iter() {
                c.gids
                    .extend_selection_gids(n_kv_head, &mut unified_head_gids);
            }
            chunk_counts.push(gids_layer.len());
            chunk_gids_keepalive.extend(gids_layer.iter().cloned());
        }

        // Fused selection-table size. Worth logging because this step changes
        // the table's *exponent*, not just its constant: keyed by arena it grew
        // with the arena count — i.e. with the fragmentation this initiative
        // exists to delete, so a sicker pool made every drain more expensive.
        // Keyed by chunk it grows with the work actually being selected.
        tracing::debug!(
            layers = backings.len(),
            chunks = chunk_gids_keepalive.len(),
            rows = chunk_gids_keepalive.len() * n_kv_head,
            bytes = unified_table.len() * std::mem::size_of::<i64>(),
            "fused selection table"
        );
        // The span, whatever the caller passed: neither table is staged through
        // the pinned generation (see `from_head_gids`), so a pinned caller that
        // skipped the bump guard sent both uploads to the driver — the
        // persistence thread's most frequent allocation, once per drain.
        let _ = generation;
        let bump_gen = bump_generation(None, dev);
        let per_head_table_buf = stage_table(&unified_table, bump_gen.as_ref(), dev)?;
        let head_gids_buf = stage_i64_slice(&unified_head_gids, None, bump_gen.as_ref(), dev)?;

        Ok((
            Self {
                chunk_gids_keepalive,
                per_head_table: unified_table,
                per_head_table_buf,
                head_gids: unified_head_gids,
                head_gids_buf,
                blocks_per_chunk: n_kv_head * head_dim,
                n_kv_head,
                arena_chunks: GID_STRIDE,
                dev: dev.clone(),
                _bump_gen: bump_gen,
                palette_block: Mutex::new(None),
                palette_owned: Mutex::new(None),
            },
            chunk_counts,
        ))
    }

    pub fn select_chunks(
        &self,
        start_chunk: usize,
        chunk_count: usize,
        generation: Option<&Generation>,
    ) -> Result<Self> {
        let end_chunk = start_chunk + chunk_count;
        let gids_per_chunk = self.n_kv_head * GIDS_PER_HEAD;
        if end_chunk > self.chunk_gids_keepalive.len()
            || end_chunk * gids_per_chunk > self.head_gids.len()
        {
            candle::bail!("paged input chunk range out of bounds");
        }

        let head_gids =
            self.head_gids[start_chunk * gids_per_chunk..end_chunk * gids_per_chunk].to_vec();
        let chunk_gids_keepalive = self.chunk_gids_keepalive[start_chunk..end_chunk].to_vec();
        // The table is keyed by (chunk, head), so it narrows with the gids —
        // the kernel's `chunk_idx` is relative to *this* buffer. Sharing the
        // parent's table would make chunk `start_chunk + i` read row `i`.
        let rows_per_chunk = self.n_kv_head * PerHeadEntry::COLS;
        let per_head_table =
            self.per_head_table[start_chunk * rows_per_chunk..end_chunk * rows_per_chunk].to_vec();
        // OWNED upload — same staging-lifetime hazard as `from_head_gids`.
        let _ = generation;
        let bump_gen = bump_generation(None, &self.dev);
        let head_gids_buf = stage_i64_slice(&head_gids, None, bump_gen.as_ref(), &self.dev)?;
        let per_head_table_buf = stage_table(&per_head_table, bump_gen.as_ref(), &self.dev)?;

        Ok(Self {
            // Re-uploaded through `bump_gen` above, so these borrow the span
            // it holds open.
            _bump_gen: bump_gen,
            palette_block: Mutex::new(None),
            palette_owned: Mutex::new(None),
            chunk_gids_keepalive,
            per_head_table,
            per_head_table_buf,
            head_gids,
            head_gids_buf,
            blocks_per_chunk: self.blocks_per_chunk,
            n_kv_head: self.n_kv_head,
            arena_chunks: self.arena_chunks,
            dev: self.dev.clone(),
        })
    }

    pub fn per_head_table_buf(&self) -> &GpuBuf {
        &self.per_head_table_buf
    }

    pub fn head_gids_buf(&self) -> &GpuBuf {
        &self.head_gids_buf
    }

    pub fn head_gids(&self) -> &[i64] {
        &self.head_gids
    }

    pub fn arena_chunks(&self) -> usize {
        self.arena_chunks
    }

    pub fn dev(&self) -> &candle::CudaDevice {
        &self.dev
    }

    #[inline]
    pub(super) fn blocks_per_head(&self) -> usize {
        // 32-element blocks per head per chunk: one block per head-dimension
        // (32 tokens × 1 dim). Equal to head_dim (= blocks_per_chunk / n_kv_head).
        self.blocks_per_chunk / self.n_kv_head.max(1)
    }

    /// The fused selection kernel is instantiated per head_dim (`HB` template
    /// parameter, dispatched by the CUDA launcher). Anything else would make
    /// the kernel guard return without writing any selection output, so fail
    /// loudly here instead.
    pub(super) fn checked_blocks_per_head(&self) -> Result<usize> {
        let blocks_per_head = self.blocks_per_head();
        if blocks_per_head != 128 && blocks_per_head != 256 {
            candle::bail!(
                "fused KV format selection supports head_dim (blocks_per_head) 128 or 256, got {blocks_per_head}"
            );
        }
        Ok(blocks_per_head)
    }

    pub fn n_chunks(&self) -> usize {
        self.chunk_gids_keepalive.len()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn select_block_formats(
        &self,
        k_candidates: &[SampleFormat],
        v_candidates: &[SampleFormat],
        k_threshold_hi: f32,
        k_threshold_lo: f32,
        v_threshold_hi: f32,
        v_threshold_lo: f32,
    ) -> Result<(
        Vec<SampleFormat>,
        Vec<SampleFormat>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    )> {
        let blocks_per_head = self.checked_blocks_per_head()?;
        let k_ggml: Vec<GgmlDType> = k_candidates
            .iter()
            .copied()
            .map(|f| f.to_ggml_dtype())
            .collect();
        let v_ggml: Vec<GgmlDType> = v_candidates
            .iter()
            .copied()
            .map(|f| f.to_ggml_dtype())
            .collect();

        let n_chunks = self.chunk_gids_keepalive.len();
        let total_heads = n_chunks * self.n_kv_head;
        // Per-block selection is only used on full chunks.
        let valid_ranges_arg = vec![32i32; n_chunks];

        let Palette4Result {
            k_palette_tags: k_pal_tags_cpu,
            v_palette_tags: v_pal_tags_cpu,
            k_pal_scale: k_pal_scale_cpu,
            v_pal_scale: v_pal_scale_cpu,
            k_palette_map: k_map_cpu,
            v_palette_map: v_map_cpu,
            k_head_amax,
            v_head_amax,
            ..
        } = self.run_palette4(
            &k_ggml,
            &v_ggml,
            [
                k_threshold_hi,
                k_threshold_lo,
                v_threshold_hi,
                v_threshold_lo,
            ],
            &valid_ranges_arg,
        )?;

        let total_blocks = total_heads * blocks_per_head;

        let mut k_blk_tags = Vec::with_capacity(total_blocks);
        let mut v_blk_tags = Vec::with_capacity(total_blocks);
        let mut k_blk_scale = Vec::with_capacity(total_blocks);
        let mut v_blk_scale = Vec::with_capacity(total_blocks);

        for bi in 0..total_blocks {
            let head = bi / blocks_per_head;
            let k_slot = (k_map_cpu[bi].clamp(0, 3)) as usize;
            let v_slot = (v_map_cpu[bi].clamp(0, 3)) as usize;
            k_blk_tags.push(SampleFormat::try_from_cuda_tag(
                k_pal_tags_cpu[head * 4 + k_slot],
            )?);
            v_blk_tags.push(SampleFormat::try_from_cuda_tag(
                v_pal_tags_cpu[head * 4 + v_slot],
            )?);
            k_blk_scale.push(k_pal_scale_cpu[head * 4 + k_slot]);
            v_blk_scale.push(v_pal_scale_cpu[head * 4 + v_slot]);
        }

        Ok((
            k_blk_tags,
            v_blk_tags,
            k_blk_scale,
            v_blk_scale,
            k_head_amax,
            v_head_amax,
        ))
    }

    pub fn aggregate_quality_metric_formats_gpu(
        &self,
        k_formats: &[SampleFormat],
        v_formats: &[SampleFormat],
        generation: Option<&Generation>,
    ) -> Result<GpuQualityAggregation> {
        let (k_head_formats, v_head_formats, k_worst_case_formats, v_worst_case_formats) =
            self.reduce_per_head_formats_gpu(k_formats, v_formats, generation)?;
        // `k_formats` / `v_formats` are already palette-expanded per-block tags
        // produced by the fused GPU kernel (`select_kv_format_palette4_paged`)
        // via `select_block_formats`.  Re-running `cpu_palette4_reduce` on top
        // would shuffle blocks across palette slots using a coarser CPU greedy
        // that doesn't see block data — so we just pass the kernel's tags
        // through unchanged.
        let k_palette4_formats = k_formats.to_vec();
        let v_palette4_formats = v_formats.to_vec();
        Ok(GpuQualityAggregation {
            k_head_formats,
            v_head_formats,
            k_worst_case_formats,
            v_worst_case_formats,
            k_palette4_formats,
            v_palette4_formats,
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn reduce_per_head_formats_gpu(
        &self,
        k_formats: &[SampleFormat],
        v_formats: &[SampleFormat],
        generation: Option<&Generation>,
    ) -> Result<(
        Vec<SampleFormat>,
        Vec<SampleFormat>,
        Vec<SampleFormat>,
        Vec<SampleFormat>,
    )> {
        let blocks_per_head = self.blocks_per_head();
        let total_blocks = self.chunk_gids_keepalive.len() * self.n_kv_head * blocks_per_head;
        if k_formats.len() != total_blocks || v_formats.len() != total_blocks {
            candle::bail!("reduce_per_head_formats_gpu length mismatch");
        }
        let k_codes: Vec<i32> = k_formats
            .iter()
            .copied()
            .map(SampleFormat::to_cuda_tag)
            .collect();
        let v_codes: Vec<i32> = v_formats
            .iter()
            .copied()
            .map(SampleFormat::to_cuda_tag)
            .collect();
        let k_buf = stage_i32_slice(&k_codes, generation, self._bump_gen.as_ref(), &self.dev)?;
        let v_buf = stage_i32_slice(&v_codes, generation, self._bump_gen.as_ref(), &self.dev)?;
        let (k_head_gpu, v_head_gpu, k_eff_gpu, v_eff_gpu) = reduce_head_format_stats(
            k_buf.dev_ptr(),
            v_buf.dev_ptr(),
            blocks_per_head,
            self.n_kv_head,
            self.chunk_gids_keepalive.len(),
            &self.dev,
        )?;
        Ok((
            self.dev
                .memcpy_dtov(&k_head_gpu)?
                .into_iter()
                .map(SampleFormat::try_from_cuda_tag)
                .collect::<Result<Vec<_>>>()?,
            self.dev
                .memcpy_dtov(&v_head_gpu)?
                .into_iter()
                .map(SampleFormat::try_from_cuda_tag)
                .collect::<Result<Vec<_>>>()?,
            self.dev
                .memcpy_dtov(&k_eff_gpu)?
                .into_iter()
                .map(SampleFormat::try_from_cuda_tag)
                .collect::<Result<Vec<_>>>()?,
            self.dev
                .memcpy_dtov(&v_eff_gpu)?
                .into_iter()
                .map(SampleFormat::try_from_cuda_tag)
                .collect::<Result<Vec<_>>>()?,
        ))
    }

    /// Fused per-head palette-4 selection: single kernel pass replacing `select_block_formats` + palette reduction.
    ///
    /// Returns `(k_palette4_rows, v_palette4_rows, k_pal_scale_rows, v_pal_scale_rows,
    ///           k_assignments, v_assignments, k_head_amax, v_head_amax)`.
    #[allow(clippy::type_complexity)]
    pub fn select_palette4_formats_fused(
        &self,
        k_candidates: &[SampleFormat],
        v_candidates: &[SampleFormat],
        k_threshold_hi: f32,
        k_threshold_lo: f32,
        v_threshold_hi: f32,
        v_threshold_lo: f32,
        // Per-chunk valid token range, packed (offset << 8) | len with len
        // in [1, 32]; `None` = every chunk is full. Corrects the
        // count-normalized error metrics and sink statistics for partial
        // chunks, and bounds every load: the kernels read a dead slot as
        // zero whatever the recycled ground holds.
        valid_ranges: Option<&[i32]>,
        _generation: Option<&Generation>,
    ) -> Result<(
        Vec<[SampleFormat; 4]>,
        Vec<[SampleFormat; 4]>,
        Vec<[f32; 4]>,
        Vec<[f32; 4]>,
        Vec<u8>,
        Vec<u8>,
        Vec<f32>,
        Vec<f32>,
    )> {
        let n_chunks = self.chunk_gids_keepalive.len();
        let k_ggml: Vec<candle::quantized::GgmlDType> = k_candidates
            .iter()
            .copied()
            .map(|f| f.to_ggml_dtype())
            .collect();
        let v_ggml: Vec<candle::quantized::GgmlDType> = v_candidates
            .iter()
            .copied()
            .map(|f| f.to_ggml_dtype())
            .collect();

        let valid_ranges_arg: Vec<i32> = match valid_ranges {
            Some(r) => {
                if r.len() != n_chunks {
                    candle::bail!("valid_ranges length {} != n_chunks {}", r.len(), n_chunks);
                }
                r.to_vec()
            }
            None => vec![32i32; n_chunks],
        };

        let Palette4Result {
            k_palette_tags: k_pal_tags_cpu,
            v_palette_tags: v_pal_tags_cpu,
            k_pal_scale: k_pal_scale_cpu,
            v_pal_scale: v_pal_scale_cpu,
            k_palette_map: k_map_cpu,
            v_palette_map: v_map_cpu,
            k_head_amax,
            v_head_amax,
            ..
        } = self.run_palette4(
            &k_ggml,
            &v_ggml,
            [
                k_threshold_hi,
                k_threshold_lo,
                v_threshold_hi,
                v_threshold_lo,
            ],
            &valid_ranges_arg,
        )?;

        let to_palette = |flat: &[i32]| -> Result<Vec<[SampleFormat; 4]>> {
            flat.as_chunks::<4>()
                .0
                .iter()
                .map(|c| {
                    Ok([
                        SampleFormat::try_from_cuda_tag(c[0])?,
                        SampleFormat::try_from_cuda_tag(c[1])?,
                        SampleFormat::try_from_cuda_tag(c[2])?,
                        SampleFormat::try_from_cuda_tag(c[3])?,
                    ])
                })
                .collect()
        };
        let to_scale_palette = |flat: &[f32]| -> Vec<[f32; 4]> { flat.as_chunks::<4>().0.to_vec() };

        let k_palette4_rows = to_palette(&k_pal_tags_cpu)?;
        let v_palette4_rows = to_palette(&v_pal_tags_cpu)?;
        let k_pal_scale_rows = to_scale_palette(&k_pal_scale_cpu);
        let v_pal_scale_rows = to_scale_palette(&v_pal_scale_cpu);
        let k_assignments: Vec<u8> = k_map_cpu.iter().map(|&v| v.clamp(0, 3) as u8).collect();
        let v_assignments: Vec<u8> = v_map_cpu.iter().map(|&v| v.clamp(0, 3) as u8).collect();

        Ok((
            k_palette4_rows,
            v_palette4_rows,
            k_pal_scale_rows,
            v_pal_scale_rows,
            k_assignments,
            v_assignments,
            k_head_amax,
            v_head_amax,
        ))
    }
}
